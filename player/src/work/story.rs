//! A song's story planned for the karaoke: the language model's unit of
//! work ([`StoryJob`]). In: the song (its timed lyrics and tags:
//! [`Song`]); out: its plan ([`Plan`]), or why it has none ([`Failure`]);
//! the model it needs: Gemma 4 E4B ([`Gemma`], as a [`Writer`]), opened,
//! with room for [`CONTEXT`] positions ([`memory`] says what that takes
//! before anything is loaded). Whoever runs the model's work (and decides
//! when it is loaded and let go) runs it; the job opens, keeps and lets go
//! of nothing on the GPU but the room for its context, which it makes as
//! it writes and leaves made.
//!
//! The model writes the plan greedily, so the same song always gives the
//! same plan ([`plan`]), its prose told as it comes for the preparation
//! view ([`Heard::Story`]). An answer that does not hold
//! ([`crate::core::story::check::read`]) is asked again once, the error
//! said with the prompt; a second that does not hold either leaves the
//! song with no story (told, and noted in the trace): the karaoke shows
//! its cover as it always has.
//!
//! A plan is kept in `stories` in the data folder
//! ([`crate::store::cache::STORY_PLAN`]), a small file a song, named by a
//! hash of its path, size and modification time, with what made it
//! ([`VERSION`]) and a hash of what it was asked, both compared when it
//! is read: new lyrics, new tags or another prompt, and it is planned
//! again. A song whose answers did not hold is kept as having no story
//! (it would get none again); a model that failed, or an answer cut off
//! at its length, is not (the song is not to blame).

use std::path::{Path, PathBuf};
use std::time::Duration;

use jodel_gpu::llm::MOST_CONTEXT;
use jodel_gpu::{Footprint, Gemma};

use crate::core::prep::Story;
use crate::core::story::prose::Prose;
use crate::core::story::rules::MOST_ANSWER;
use crate::core::story::{check, prompt, Plan, Song};
use crate::core::{Action, Heard};
use crate::store::cache::{self, STORY_PLAN};
use crate::work::outbox::Outbox;

/// What makes a plan, as the files say it: the model (Gemma 4 E4B-it,
/// its weights) written greedily, and how its answer is read. A new
/// model, other weights or another reading raise it, and every song is
/// planned again; a new prompt does so by itself (its hash is kept).
pub const VERSION: u32 = 1;

/// The positions a plan takes at most, its prompt and its answer: as
/// many as the model has room for.
pub const CONTEXT: usize = MOST_CONTEXT;

/// What the job needs on the GPU: the model in the file at `model`, with
/// room for [`CONTEXT`] (only the file's header read).
pub fn memory(model: &Path) -> Result<Footprint, String> {
    Gemma::planned(model, CONTEXT)
}

/// A language model that writes after a prompt (one user turn): the
/// text, greedily, at most `most` tokens, shown to `seen` piece by piece
/// as it comes. A prompt too long for it is the song's failure
/// ([`Failure::Refused`]), anything else its own ([`Failure::Model`]).
pub trait Writer {
    /// What it writes after `prompt`.
    fn write(
        &self,
        prompt: &str,
        most: usize,
        seen: &mut dyn FnMut(&str),
    ) -> Result<Written, Failure>;
}

/// What a writer wrote: the text, whether it ended by itself (not at its
/// limit), and the prompt's and the answer's tokens and times.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Written {
    /// The text.
    pub text: String,
    /// It ended by itself.
    pub stopped: bool,
    /// The prompt's tokens.
    pub prompt_tokens: usize,
    /// How long the prompt took, to the answer's first token.
    pub prompt_time: Duration,
    /// The answer's tokens.
    pub answer_tokens: usize,
    /// How long the answer took.
    pub answer_time: Duration,
}

impl Writer for Gemma {
    fn write(
        &self,
        prompt: &str,
        most: usize,
        seen: &mut dyn FnMut(&str),
    ) -> Result<Written, Failure> {
        let ids = self.tokens().chat(prompt);
        if ids.len() + most > CONTEXT {
            return Err(Failure::Refused(format!(
                "its words are too long for the model ({} tokens asked, at most {})",
                ids.len(),
                CONTEXT - most
            )));
        }
        self.reserve(CONTEXT).map_err(Failure::Model)?;
        let out = self
            .generate_text(&ids, most, |piece| seen(piece))
            .map_err(Failure::Model)?;
        let generated = out.generated;
        Ok(Written {
            text: out.text,
            stopped: generated.stopped,
            prompt_tokens: generated.prompt_tokens,
            prompt_time: generated.prompt_time,
            answer_tokens: generated.tokens.len(),
            answer_time: generated.step_time,
        })
    }
}

/// A try at a plan: what was written, and why it was refused (none:
/// taken).
#[derive(Clone, Debug, PartialEq)]
pub struct Try {
    /// What was written.
    pub written: Written,
    /// Why it was refused.
    pub refused: Option<String>,
}

/// Why a song has no plan.
#[derive(Clone, Debug, PartialEq)]
pub enum Failure {
    /// Its answers did not hold (it would get none again): kept.
    Refused(String),
    /// An answer was cut off at [`MOST_ANSWER`] (the length's fault, not
    /// asked again): not kept.
    CutOff(String),
    /// The model failed: not kept.
    Model(String),
}

impl Failure {
    /// Why, in words.
    pub fn why(&self) -> &str {
        match self {
            Failure::Refused(why) | Failure::CutOff(why) | Failure::Model(why) => why,
        }
    }
}

/// A plan written ([`plan`]), or why there is none, and its tries.
#[derive(Clone, Debug)]
pub struct Planned {
    /// The plan.
    pub plan: Result<Plan, Failure>,
    /// The tries, in order.
    pub tries: Vec<Try>,
}

/// `song`'s plan as `writer` writes it after `asked` (the song's prompt,
/// [`prompt::prompt`]): asked once, and once more with the error if the
/// answer does not hold (not if it was cut off); a song with no lines is
/// not asked at all. `tell` is told the prose as it is written, a
/// note at the retry, and why there is no plan, if there is none.
pub fn plan(writer: &dyn Writer, song: &Song, asked: &str, tell: &mut dyn FnMut(Story)) -> Planned {
    let mut tries = Vec::new();
    // Not the model's to break: with no line to tell of there is no plan.
    if song.lines.is_empty() {
        let none = Failure::Refused(NO_LINES.into());
        return failed(none, tries, tell);
    }
    // A cut-off answer is not asked again: the second prompt is longer
    // and the answer's room the same, so it would be cut off again.
    let why = match attempt(writer, song, asked, tell, &mut tries) {
        Ok(Ok(plan)) => {
            return Planned {
                plan: Ok(plan),
                tries,
            }
        }
        Ok(Err(Failure::Refused(why))) => why,
        Ok(Err(failure)) | Err(failure) => return failed(failure, tries, tell),
    };
    tell(Story::Words(format!(
        "\n(That plan did not hold: {why}. Once more.)\n\n"
    )));
    match attempt(writer, song, &prompt::again(asked, &why), tell, &mut tries) {
        Ok(Ok(plan)) => Planned {
            plan: Ok(plan),
            tries,
        },
        Ok(Err(failure)) | Err(failure) => failed(failure, tries, tell),
    }
}

/// Why a song with no timed sung lines has no plan.
pub const NO_LINES: &str = "the song has no timed sung lines";

/// One try: `writer`'s answer to `prompt`, its prose told as it comes,
/// read as `song`'s plan, and the try kept in `tries`; the plan or why it
/// was refused (an answer that does not hold, or one cut off at
/// [`MOST_ANSWER`]), or the writer's failure.
fn attempt(
    writer: &dyn Writer,
    song: &Song,
    prompt: &str,
    tell: &mut dyn FnMut(Story),
    tries: &mut Vec<Try>,
) -> Result<Result<Plan, Failure>, Failure> {
    let mut prose = Prose::new();
    let written = writer.write(prompt, MOST_ANSWER, &mut |piece| {
        let words = prose.feed(piece);
        if !words.is_empty() {
            tell(Story::Words(words));
        }
    })?;
    let read = check::read(&written.text, song).map_err(|why| {
        if written.stopped {
            Failure::Refused(why)
        } else {
            Failure::CutOff(format!(
                "{why} (the answer was cut off at {MOST_ANSWER} tokens)"
            ))
        }
    });
    tries.push(Try {
        refused: read.as_ref().err().map(|f| f.why().to_owned()),
        written,
    });
    Ok(read)
}

/// No plan, for `failure`: told.
fn failed(failure: Failure, tries: Vec<Try>, tell: &mut dyn FnMut(Story)) -> Planned {
    tell(Story::Words(format!(
        "\nNo story for this song: {}\n",
        failure.why()
    )));
    Planned {
        plan: Err(failure),
        tries,
    }
}

/// What is kept for a song: its plan, or that it has none and why.
#[derive(Clone, Debug, PartialEq)]
pub enum Kept {
    /// Its plan.
    Plan(Box<Plan>),
    /// No story, and why.
    NoStory(String),
}

/// The language model's unit of work for a song: its file (the news it
/// tells is of it), the song as the plan is made from it, where its plan
/// is kept (the data folder's `stories`), and what it asks.
#[derive(Clone, Debug)]
pub struct StoryJob {
    path: PathBuf,
    song: Song,
    dir: PathBuf,
    asked: String,
}

impl StoryJob {
    /// The job of planning `song`, of the file `path`, kept in `dir`.
    pub fn new(path: PathBuf, song: Song, dir: PathBuf) -> StoryJob {
        let asked = prompt::prompt(&song);
        StoryJob {
            path,
            song,
            dir,
            asked,
        }
    }

    /// The song's file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// What is kept for the song by this [`VERSION`] for what it is
    /// asked now; `None`: it is to be planned.
    pub fn kept(&self) -> Option<Kept> {
        let file = crate::store::mover::locate(&self.file(&self.dir)?);
        from_text(&std::fs::read_to_string(file).ok()?, &self.hash())
    }

    /// Plans the song with `writer` ([`plan`]), its news told on `outbox`
    /// ([`Heard::Story`]), and keeps the plan, or that it has none where
    /// that is the song's ([`Failure::Refused`]).
    pub fn run(&self, writer: &dyn Writer, outbox: &Outbox) -> Result<Plan, Failure> {
        let planned = plan(writer, &self.song, &self.asked, &mut |story| {
            outbox.send(Action::Heard {
                path: self.path.clone(),
                part: Heard::Story(story),
            });
        });
        let name = self.path.file_name().unwrap_or_default().to_string_lossy();
        for t in &planned.tries {
            let w = &t.written;
            crate::trace::note(format_args!(
                "story: {name}: {} prompt tokens in {:.1} s, {} written in {:.1} s{}{}",
                w.prompt_tokens,
                w.prompt_time.as_secs_f64(),
                w.answer_tokens,
                w.answer_time.as_secs_f64(),
                if w.stopped { "" } else { " (cut off)" },
                t.refused
                    .as_ref()
                    .map_or_else(String::new, |why| format!(", refused: {why}"))
            ));
        }
        let kept = match &planned.plan {
            Ok(plan) => Kept::Plan(Box::new(plan.clone())),
            Err(Failure::Refused(why)) => Kept::NoStory(why.clone()),
            Err(failure) => {
                crate::trace::note(format_args!("story: {name}: not kept: {}", failure.why()));
                return planned.plan;
            }
        };
        if let Kept::NoStory(why) = &kept {
            crate::trace::note(format_args!("story: {name}: no story: {why}"));
        }
        if let Some(file) = self.file(&crate::store::mover::current(&self.dir)) {
            let text = to_text(&self.hash(), &kept);
            if let Err(e) = durable::write_atomic(&file, text.as_bytes()) {
                crate::trace::note(format_args!("story: {name}: not kept: {e}"));
            }
        }
        planned.plan
    }

    /// The file of the song's plan in `dir` (whether or not it is there);
    /// `None` when the song cannot be read.
    fn file(&self, dir: &Path) -> Option<PathBuf> {
        cache::file_for(dir, &self.path, "story")
    }

    /// What the song is asked, as the file names it: the prompt's hash.
    fn hash(&self) -> String {
        cache::name(&self.asked)
    }
}

/// What starts the line of what made a kept plan's file.
const MADE: &str = "made story";
/// What starts the line of what it was asked.
const ASKED: &str = "asked";
/// The line before its plan.
const PLAN: &str = "plan";
/// What starts the line of why there is none.
const NONE: &str = "none";

/// What is kept for a song as its file holds it: its header, what made
/// it, the hash of what was `asked`, then the plan as JSON, or `none` and
/// why.
pub fn to_text(asked: &str, kept: &Kept) -> String {
    let what = match kept {
        Kept::Plan(plan) => format!("{PLAN}\n{}\n", plan.to_json()),
        Kept::NoStory(why) => format!("{NONE} {}\n", why.replace('\n', " ")),
    };
    STORY_PLAN.wrap(&format!("{MADE} {VERSION}\n{ASKED} {asked}\n{what}"))
}

/// What is kept in a file's `text`, where it was made by this [`VERSION`]
/// for what is `asked`; else (another version of the schema, another
/// maker, another ask, a file not read) none.
pub fn from_text(text: &str, asked: &str) -> Option<Kept> {
    let body = STORY_PLAN.body(text)?;
    let (made, rest) = body.split_once('\n')?;
    let (was, rest) = rest.split_once('\n')?;
    if made != format!("{MADE} {VERSION}") || was != format!("{ASKED} {asked}") {
        return None;
    }
    if let Some(json) = rest.strip_prefix(&format!("{PLAN}\n")) {
        return Plan::from_json(json).ok().map(|p| Kept::Plan(Box::new(p)));
    }
    let why = rest
        .strip_prefix(&format!("{NONE} "))?
        .trim_end_matches('\n');
    Some(Kept::NoStory(why.to_owned()))
}

#[cfg(test)]
mod tests;
