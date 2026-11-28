//! A song's story planned for the karaoke: the language model's unit of
//! work ([`StoryJob`]). In: the song (its timed lyrics and tags:
//! [`Song`]); out: its plan ([`Plan`]), or why it has none ([`Failure`]);
//! the model it needs: Gemma 4 E4B ([`Gemma`], as a [`Writer`]), opened,
//! with room for what a plan can use where the card holds it ([`context`],
//! decided once by whoever runs the model's work; [`Gemma::planned`] says
//! what that takes before anything is loaded). Whoever runs the model's work
//! (and decides when it is loaded and let go) runs it; the job opens,
//! keeps and lets go of nothing on the GPU but the room for its context,
//! which it makes as it writes and lets go once written (the stages count
//! it only while a job runs).
//!
//! The model thinks first (its thinking switched on as its chat template
//! does it), then writes the plan greedily, so the same song always gives
//! the same plan ([`plan`]); the thought is left out of the answer read
//! (its tokens and time are noted in the trace), and the plan's prose is
//! told as it comes for the preparation view
//! ([`crate::core::Heard::Story`]), "thinking…" while the thought goes on.
//! The thought takes what the context leaves after the prompt and the
//! answer's room ([`MOST_ANSWER`]), at most [`MOST_THOUGHT`] tokens (a
//! guard against one that never ends): one cut off at that most is cut
//! off ([`Failure::CutOff`]), one cut off sooner by the context was short
//! of room ([`Failure::Short`]). An answer that does not hold
//! ([`crate::core::story::check::read`]) is asked again once, the error
//! said with the prompt; a second that does not hold either leaves the
//! song with no story (told, and noted in the trace): it can be sung with
//! the game off, on the blurred cover, not in the game
//! ([`crate::core::karaoke::NO_STORY`]).
//!
//! A plan is kept in `stories` in the data folder
//! ([`crate::store::cache::STORY_PLAN`]), a small file a song, named by a
//! hash of its path, size and modification time, with what made it
//! ([`VERSION`]) and a hash of what it was asked and the answer's and the
//! thought's rooms ([`MOST_ANSWER`], [`MOST_THOUGHT`]), both compared when
//! it is read: new lyrics, new tags, another prompt or another room, and
//! it is planned again. A song whose answers did not hold, or whose
//! answer or thought was cut off at its most, is kept as having no story:
//! the model writes greedily, so it would get the same again until the
//! version, the prompt, the rooms or the lyrics change. A plan holds for
//! any context: written greedily, the same thought and answer come in any
//! room they fit. Only a song short of room (its prompt too long, or its
//! thought cut off sooner than its most) is kept with the context it had,
//! and planned again where the card gives a larger one (the caller
//! compares: [`Kept::Short`]). A model that failed is not kept (the song
//! is not to blame: it is tried again).
//!
//! The plan's pictures are drawn by the image model's two jobs
//! ([`draw`]), and every song of the queue is planned, drawn and its
//! scenes taken apart on a thread of its own as a step of its preparation,
//! the models taking turns ([`worker`]).

pub mod draw;
pub mod worker;

use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use jodel_gpu::llm::Take;
use jodel_gpu::{Gemma, Resident};

use crate::core::prep::Story;
use crate::core::story::prose::Prose;
use crate::core::story::rules::{MOST_ANSWER, MOST_PROMPT, MOST_THOUGHT};
use crate::core::story::{check, prompt, Plan, Song};
use crate::store::cache::{self, STORY_PLAN};
use crate::work::stages::plan::{room, MARGIN};

/// What makes a plan, as the files say it: the model (Gemma 4 E4B-it,
/// its weights) written greedily, and how its answer is read. A new
/// model, other weights or another reading raise it, and every song is
/// planned again; a new prompt does so by itself (its hash is kept).
/// Version 1 refused shaders over the budget; version 2 wrote without
/// thinking.
pub const VERSION: u32 = 3;

/// What the process holds on the card beside the stages' models at the
/// least (the window, the track playing's stems), where nothing better is
/// known when the context is worked out ([`context`]).
const BESIDE: u64 = 512 << 20;

/// The positions a plan can use at most: the longest prompt it is given
/// room for, its thought and its answer at their most, the thought's two
/// specials and one token past.
pub const NEED: usize = MOST_PROMPT + MOST_THOUGHT + MOST_ANSWER + 3;

/// The positions a plan takes at most, its prompt, the model's thought
/// and its answer, for the model in the file at `model`: what it can use
/// ([`NEED`]) where the card holds that, else the largest the card's
/// budget holds for the model alone beside what the stages keep free and
/// what else the process holds ([`room`] with [`BESIDE`]), in steps
/// ([`Gemma::fitting`]: from [`jodel_gpu::llm::LEAST_CONTEXT`]). A card's
/// memory is used where it helps the story, never held for positions the
/// story cannot reach. Worked out from the budget as it is now (only the
/// file's header read): Vulkan's moves with what other programs hold, so
/// whoever runs the model decides it once and keeps it, for the stages'
/// count of its room, each plan's room and the comparison of a song short
/// of room alike (`worker::GpuModels`; on a card with room for the need it
/// is always the need).
pub fn context(model: &Path) -> Result<usize, String> {
    let room = jodel_gpu::memory::budget().map(|b| room(b.budget, BESIDE));
    let context = Gemma::fitting(model, room, NEED)?;
    crate::trace::note(format_args!(
        "story: a context of {context} positions (the stages keep {} MB free)",
        MARGIN >> 20
    ));
    Ok(context)
}

/// A language model that thinks, then writes, after a prompt (one user
/// turn): greedily, in a context of `context` positions, its answer at
/// most [`MOST_ANSWER`] tokens, shown to `seen` as it comes
/// ([`Writing`]). A prompt too long for the context is the song's failure
/// there ([`Failure::Short`]), anything else the model's own
/// ([`Failure::Model`]).
pub trait Writer {
    /// What it writes after `prompt`.
    fn write(
        &self,
        prompt: &str,
        context: usize,
        seen: &mut dyn FnMut(Writing),
    ) -> Result<Written, Failure>;
}

/// What a writer shows as it writes: that it thinks (once, as its thought
/// begins), then its answer, piece by piece.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Writing<'a> {
    /// Its thought begun.
    Thinking,
    /// The answer's next piece.
    Answer(&'a str),
}

/// What a writer wrote: the answer's text (no thought), whether it ended
/// by itself (not at a limit, its thought ended), whether it ended still
/// thinking, and the prompt's, the thought's and the answer's tokens and
/// times.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Written {
    /// The answer's text.
    pub text: String,
    /// It ended by itself.
    pub stopped: bool,
    /// It ended within its thought (cut off there, or the thought never
    /// closed).
    pub thinking: bool,
    /// The prompt's tokens.
    pub prompt_tokens: usize,
    /// How long the prompt took, to the answer's first token.
    pub prompt_time: Duration,
    /// The thought's tokens, between its specials.
    pub thought_tokens: usize,
    /// The thought's room: what the context left it, at most
    /// [`MOST_THOUGHT`].
    pub thought_room: usize,
    /// How long the thought took.
    pub thought_time: Duration,
    /// The answer's tokens.
    pub answer_tokens: usize,
    /// How long the answer took.
    pub answer_time: Duration,
}

/// The room for its context is made for the answer and let go after it
/// (the stages count it only while a job runs).
impl Writer for Gemma {
    fn write(
        &self,
        prompt: &str,
        context: usize,
        seen: &mut dyn FnMut(Writing),
    ) -> Result<Written, Failure> {
        let tokens = self.tokens();
        let most = MOST_ANSWER;
        let ids = tokens.chat_thinking(prompt);
        // The answer's room, the thought's two specials and one token past
        // to find either over ([`jodel_gpu::llm::Thinking::most`]); the
        // thought has what the context leaves, at most [`MOST_THOUGHT`].
        let answer = most + 3;
        let Some(left) = context.checked_sub(ids.len() + answer) else {
            return Err(Failure::Short(format!(
                "its words are too long for the model ({} tokens asked, at most {})",
                ids.len(),
                context.saturating_sub(answer)
            )));
        };
        self.reserve(context).map_err(Failure::Model)?;
        let mut out = Written {
            thought_room: left.min(MOST_THOUGHT),
            ..Written::default()
        };
        let mut thinking = tokens.thinking(out.thought_room, most);
        let mut stream = tokens.stream();
        let (mut opened, mut closed) = (None, None);
        let generated = self.generate(&ids, thinking.most(), |got| {
            let now = Instant::now();
            for &t in got {
                match thinking.take(t) {
                    Take::Over => return ControlFlow::Break(()),
                    Take::Opened => {
                        opened = Some(now);
                        seen(Writing::Thinking);
                    }
                    Take::Thought => {}
                    Take::Closed => closed = Some(now),
                    Take::Answer => {
                        let piece = stream.push(t);
                        if !piece.is_empty() {
                            seen(Writing::Answer(&piece));
                            out.text.push_str(&piece);
                        }
                    }
                }
            }
            ControlFlow::Continue(())
        });
        self.release();
        let generated = generated.map_err(Failure::Model)?;
        let rest = stream.finish();
        if !rest.is_empty() {
            seen(Writing::Answer(&rest));
            out.text.push_str(&rest);
        }
        let ended = Instant::now();
        [out.thought_tokens, out.answer_tokens] = thinking.taken();
        out.thinking = thinking.in_thought();
        out.stopped = thinking.ended(generated.stopped);
        out.prompt_tokens = generated.prompt_tokens;
        out.prompt_time = generated.prompt_time;
        out.thought_time = opened.map_or(Duration::ZERO, |o| closed.unwrap_or(ended) - o);
        out.answer_time = generated.step_time.saturating_sub(out.thought_time);
        Ok(out)
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
    /// An answer was cut off at [`MOST_ANSWER`], or its thought at
    /// [`MOST_THOUGHT`] (not asked again: the second prompt is longer and
    /// the rooms the same; written greedily, it would be cut off again):
    /// kept.
    CutOff(String),
    /// The context had no room enough: its prompt too long for it, or its
    /// thought cut off there before [`MOST_THOUGHT`] (not asked again:
    /// written greedily, it would think as long again). Kept for that
    /// context: a larger one plans it again.
    Short(String),
    /// The model failed: not kept.
    Model(String),
}

impl Failure {
    /// Why, in words.
    pub fn why(&self) -> &str {
        match self {
            Failure::Refused(why)
            | Failure::CutOff(why)
            | Failure::Short(why)
            | Failure::Model(why) => why,
        }
    }

    /// Whether it is the song's, for good (the same answer would come
    /// again for the same ask, in a context as large): kept as having no
    /// story. A model's failure is not.
    pub fn for_good(&self) -> bool {
        !matches!(self, Failure::Model(_))
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
/// [`prompt::prompt`]) in a context of `context` positions: asked once,
/// and once more with the error if the answer does not hold (not if it
/// was cut off); a song with no lines is not asked at all. `tell` is told
/// the prose as it is written, a note at the retry, and why there is no
/// plan, if there is none.
pub fn plan(
    writer: &dyn Writer,
    song: &Song,
    (asked, context): (&str, usize),
    tell: &mut dyn FnMut(Story),
) -> Planned {
    let mut tries = Vec::new();
    // Not the model's to break: with no line to tell of there is no plan.
    if song.lines.is_empty() {
        let none = Failure::Refused(NO_LINES.into());
        return failed(none, tries, tell);
    }
    // A cut-off answer is not asked again: the second prompt is longer
    // and the answer's room the same, so it would be cut off again.
    let why = match attempt(writer, song, (asked, context), tell, &mut tries) {
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
    let again = prompt::again(asked, &why);
    match attempt(writer, song, (&again, context), tell, &mut tries) {
        Ok(Ok(plan)) => Planned {
            plan: Ok(plan),
            tries,
        },
        Ok(Err(failure)) | Err(failure) => failed(failure, tries, tell),
    }
}

/// Why a song with no timed sung lines has no plan.
pub const NO_LINES: &str = "the song has no timed sung lines";

/// One try: `writer`'s answer to `prompt` in a context of `context`
/// positions, "thinking…" told while it thinks and its prose as it comes,
/// read as `song`'s plan, and the try kept in `tries`; the plan or why it
/// was refused (an answer that does not hold, one cut off at
/// [`MOST_ANSWER`], a thought that never ended in the context), or the
/// writer's failure.
fn attempt(
    writer: &dyn Writer,
    song: &Song,
    (prompt, context): (&str, usize),
    tell: &mut dyn FnMut(Story),
    tries: &mut Vec<Try>,
) -> Result<Result<Plan, Failure>, Failure> {
    let mut prose = Prose::new();
    let written = writer.write(prompt, context, &mut |writing| match writing {
        Writing::Thinking => tell(Story::Thinking),
        Writing::Answer(piece) => {
            let words = prose.feed(piece);
            if !words.is_empty() {
                tell(Story::Words(words));
            }
        }
    })?;
    let read = if written.thinking {
        let why = format!(
            "the model's thought did not end (cut off after {} tokens)",
            written.thought_tokens
        );
        // Short where its room was full and the context, not its most,
        // made it so; any other end within the thought (at its most, or on
        // a stop token) is the song's for good: written greedily, it ends
        // there again.
        let full = written.thought_tokens == written.thought_room;
        Err(if full && written.thought_room < MOST_THOUGHT {
            Failure::Short(why)
        } else {
            Failure::CutOff(why)
        })
    } else {
        check::read(&written.text, song).map_err(|why| {
            if written.stopped {
                Failure::Refused(why)
            } else {
                Failure::CutOff(format!(
                    "{why} (the answer was cut off at {MOST_ANSWER} tokens)"
                ))
            }
        })
    };
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
    /// No story in a context of `context` positions, and why: it was short
    /// of room ([`Failure::Short`]).
    Short {
        /// The context's positions.
        context: usize,
        /// Why.
        why: String,
    },
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
    /// asked now (one short of room with the context it had: whether a
    /// larger one plans it again is the caller's to say); `None`: it is to
    /// be planned.
    pub fn kept(&self) -> Option<Kept> {
        let file = crate::store::mover::locate(&self.file(&self.dir)?);
        from_text(&std::fs::read_to_string(file).ok()?, &self.hash())
    }

    /// Plans the song with `writer` ([`plan`]) in a context of `context`
    /// positions, its news told to `tell`, and keeps the plan, or that it
    /// has none where that is the song's for good ([`Failure::for_good`];
    /// short of room, with that context).
    pub fn run(
        &self,
        writer: &dyn Writer,
        context: usize,
        tell: &mut dyn FnMut(Story),
    ) -> Result<Plan, Failure> {
        let planned = plan(writer, &self.song, (&self.asked, context), tell);
        let name = crate::work::song_name(&self.path);
        for t in &planned.tries {
            let w = &t.written;
            crate::trace::note(format_args!(
                "story: {name}: {} prompt tokens in {:.1} s, thought {} in {:.1} s, {} written in {:.1} s{}{}",
                w.prompt_tokens,
                w.prompt_time.as_secs_f64(),
                w.thought_tokens,
                w.thought_time.as_secs_f64(),
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
            Err(Failure::Short(why)) => Kept::Short {
                context,
                why: why.clone(),
            },
            Err(failure) if failure.for_good() => Kept::NoStory(failure.why().to_owned()),
            Err(failure) => {
                crate::trace::note(format_args!("story: {name}: not kept: {}", failure.why()));
                return planned.plan;
            }
        };
        if let Kept::NoStory(why) | Kept::Short { why, .. } = &kept {
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

    /// What the song is asked, as the file names it: the hash of the
    /// prompt and the answer's and the thought's rooms (the context is kept
    /// apart, for a song short of room only: [`Kept::Short`]).
    fn hash(&self) -> String {
        cache::name(&format!("{MOST_ANSWER} {MOST_THOUGHT} {}", self.asked))
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
/// What starts the line of the context short of room, and why.
const SHORT: &str = "short";

/// What is kept for a song as its file holds it: its header, what made
/// it, the hash of what was `asked`, then the plan as JSON, or `none` and
/// why, or `short`, the context and why.
pub fn to_text(asked: &str, kept: &Kept) -> String {
    let line = |why: &str| why.replace('\n', " ");
    let what = match kept {
        Kept::Plan(plan) => format!("{PLAN}\n{}\n", plan.to_json()),
        Kept::NoStory(why) => format!("{NONE} {}\n", line(why)),
        Kept::Short { context, why } => format!("{SHORT} {context} {}\n", line(why)),
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
    if let Some(short) = rest.strip_prefix(&format!("{SHORT} ")) {
        let (had, why) = short.trim_end_matches('\n').split_once(' ')?;
        return Some(Kept::Short {
            context: had.parse().ok()?,
            why: why.to_owned(),
        });
    }
    let why = rest
        .strip_prefix(&format!("{NONE} "))?
        .trim_end_matches('\n');
    Some(Kept::NoStory(why.to_owned()))
}

#[cfg(test)]
mod tests;
