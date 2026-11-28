//! A song's story planned for the karaoke: the language model's unit of
//! work ([`StoryJob`]). In: the song (its timed lyrics and tags:
//! [`Song`]); out: its plan ([`Plan`]), or why it has none ([`Failure`]);
//! the model it needs: Gemma 4 E4B ([`Gemma`], as a [`Writer`]), opened,
//! with room for what a plan can use where the card holds it ([`context`],
//! decided once by whoever runs the model's work; [`Gemma::planned`] says
//! what that takes before anything is loaded). Whoever runs the model's work
//! (and decides when it is loaded and let go) runs it; the job opens,
//! keeps and lets go of nothing on the GPU but the room for its context,
//! which its conversation makes as it is first spoken to and lets go once
//! over (the stages count it only while a job runs).
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
//! of room ([`Failure::Short`]). The answer is read forgivingly
//! ([`crate::core::story::check::read`]: a small model slips, and its work
//! is never thrown away for a slip; one cut off at its room keeps every
//! whole scene). What needs the model's judgement goes back to it once, in
//! the same conversation ([`Conversation`]: its prompt and answer still in
//! its cache, so the follow-up costs only its own words): only what is
//! wrong, its choices listed, a small patch for answer, merged into the
//! plan; what the patch leaves wrong is taken as the nearest valid choice.
//! Every mending and the follow-up go into the trace. Only an answer with
//! nothing to draw leaves the song with no story (told, and noted in the
//! trace): it can be sung with the game off, on the blurred cover, not in
//! the game ([`crate::core::karaoke::NO_STORY`]).
//!
//! A plan is kept in `stories` in the data folder
//! ([`crate::store::cache::STORY_PLAN`]), a small file a song, named by a
//! hash of its path, size and modification time, with what made it
//! ([`VERSION`]) and a hash of what it was asked and the rooms of the
//! answer, the thought and the follow-up ([`MOST_ANSWER`],
//! [`MOST_THOUGHT`], [`FOLLOW_UP_THOUGHT`], [`MOST_PATCH`]), both compared
//! when it is read: new lyrics, new tags, another prompt or another room,
//! and it is planned again. A song whose answer had nothing to draw, or
//! whose thought was cut off at its most, is kept as having no story: the
//! model writes greedily, so it would get the same again until the
//! version, the prompt, the rooms or the lyrics change. A plan holds for
//! any context: written greedily, the same thought and answer come in any
//! room they fit. Only a song short of room (its prompt too long, or its
//! thought cut off sooner than its most) is kept with the context it had,
//! and planned again where the card gives a larger one (the caller
//! compares: [`Kept::Short`]); so is a plan whose follow-up was short of
//! room (its questions taken as the nearest valid choices: kept and drawn,
//! and planned again in a larger context). A model that failed is not kept (the song
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
use serde_json::Value;

use crate::core::prep::Story;
use crate::core::story::follow;
use crate::core::story::follow::Question;
use crate::core::story::prose::Prose;
use crate::core::story::rules::{
    FOLLOW_UP_THOUGHT, MOST_ANSWER, MOST_ASK, MOST_PATCH, MOST_PROMPT, MOST_SCENES, MOST_THOUGHT,
};
use crate::core::story::{check, prompt, Plan, Song};
use crate::store::cache::{self, STORY_PLAN};
use crate::work::stages::plan::{room, MARGIN};

/// What makes a plan, as the files say it: the model (Gemma 4 E4B-it,
/// its weights) written greedily, and how its answer is read. A new
/// model, other weights or another reading raise it, and every song is
/// planned again; a new prompt does so by itself (its hash is kept).
/// Version 1 refused shaders over the budget; version 2 wrote without
/// thinking; version 3 asked for scenes by the clock and refused an answer
/// for any rule broken. The follow-up's words, its choices and the most
/// scenes are in the plan's hash ([`StoryJob`]); how they are read is
/// this version's.
pub const VERSION: u32 = 4;

/// What the process holds on the card beside the stages' models at the
/// least (the window, the track playing's stems), where nothing better is
/// known when the context is worked out ([`context`]).
const BESIDE: u64 = 512 << 20;

/// The positions a plan can use at most: the longest prompt it is given
/// room for, its thought and its answer at their most (with the thought's
/// two specials and one token past), then the follow-up's turn at its
/// most (its words, its thought and its patch, the same three, and the
/// last token written run again).
pub const NEED: usize =
    MOST_PROMPT + MOST_THOUGHT + MOST_ANSWER + 3 + MOST_ASK + FOLLOW_UP_THOUGHT + MOST_PATCH + 4;

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

/// A language model that thinks, then writes, greedily, in conversations
/// ([`Conversation`]).
pub trait Writer {
    /// A conversation in a context of `context` positions: its room made
    /// as it is first spoken to, let go when it is dropped.
    fn converse(&self, context: usize) -> Box<dyn Conversation + '_>;
}

/// A conversation with a language model: the user's turns, each answered
/// (the model thinking first), every turn and answer kept in the model's
/// cache, so a turn after the first costs only its own words.
pub trait Conversation {
    /// What the model writes after `words`, the user's next turn (the
    /// first: the prompt): its thought at most `rooms.0` tokens (what the
    /// context leaves it, where that is less), its answer at most
    /// `rooms.1`, shown to `seen` as it comes ([`Writing`]). The context
    /// with no room for the words and the answer is [`Failure::Short`],
    /// anything else the model's own failure ([`Failure::Model`]).
    fn say(
        &mut self,
        words: &str,
        rooms: (usize, usize),
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
    /// The prompt's tokens run (a turn after the first: its own).
    pub prompt_tokens: usize,
    /// How long the prompt took, to the answer's first token.
    pub prompt_time: Duration,
    /// The thought's tokens, between its specials.
    pub thought_tokens: usize,
    /// The thought's room: what the context left it, at most what was
    /// asked.
    pub thought_room: usize,
    /// How long the thought took.
    pub thought_time: Duration,
    /// The answer's tokens.
    pub answer_tokens: usize,
    /// How long the answer took.
    pub answer_time: Duration,
}

/// The room for its context is made for the conversation's first turn
/// and let go when it is over (the stages count it only while a job
/// runs).
impl Writer for Gemma {
    fn converse(&self, context: usize) -> Box<dyn Conversation + '_> {
        Box::new(Talk {
            gemma: self,
            context,
            reserved: false,
            talked: Talked::Not,
        })
    }
}

/// A conversation with Gemma ([`Writer::converse`]): its context, whether
/// its room is made, and what its cache holds.
struct Talk<'a> {
    gemma: &'a Gemma,
    context: usize,
    reserved: bool,
    talked: Talked,
}

/// What a conversation's cache holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Talked {
    /// Nothing yet, or nothing to go on from (a turn failed): the next
    /// turn starts the conversation.
    Not,
    /// Its turns so far: these many tokens, the last this one.
    Holds(usize, u32),
}

impl Drop for Talk<'_> {
    fn drop(&mut self) {
        if self.reserved {
            self.gemma.release();
        }
    }
}

impl Conversation for Talk<'_> {
    fn say(
        &mut self,
        words: &str,
        (thought, most): (usize, usize),
        seen: &mut dyn FnMut(Writing),
    ) -> Result<Written, Failure> {
        let tokens = self.gemma.tokens();
        let (ids, held) = match self.talked {
            Talked::Not => (tokens.chat_thinking(words), 0),
            Talked::Holds(held, _) => (tokens.next_turn(words), held),
        };
        // The answer's room, the thought's two specials and one token past
        // to find either over ([`jodel_gpu::llm::Thinking::most`]); the
        // thought has what the context leaves, at most `thought`.
        let answer = most + 3;
        let Some(left) = self.context.checked_sub(held + ids.len() + answer) else {
            return Err(Failure::Short(format!(
                "its words are too long for the model ({} tokens asked after {held}, at most {})",
                ids.len(),
                self.context.saturating_sub(held + answer)
            )));
        };
        if !self.reserved {
            self.gemma.reserve(self.context).map_err(Failure::Model)?;
            self.reserved = true;
        }
        let mut out = Written {
            thought_room: left.min(thought),
            ..Written::default()
        };
        let mut thinking = tokens.thinking(out.thought_room, most);
        let at_most = thinking.most();
        let mut stream = tokens.stream();
        let (mut opened, mut closed) = (None, None);
        let take = |got: &[u32]| {
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
        };
        let generated = match self.talked {
            Talked::Not => self.gemma.generate(&ids, at_most, take),
            Talked::Holds(held, last) => self.gemma.go_on((held, last), &ids, at_most, take),
        };
        // A turn that failed leaves nothing to go on from.
        self.talked = Talked::Not;
        let generated = generated.map_err(Failure::Model)?;
        let last = generated
            .tokens
            .last()
            .or(ids.last())
            .copied()
            .unwrap_or_default();
        self.talked = Talked::Holds(held + ids.len() + generated.tokens.len(), last);
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

/// A turn of a plan's conversation: what was written, and the follow-up's
/// words it answered (none: the prompt).
#[derive(Clone, Debug, PartialEq)]
pub struct Try {
    /// What was written.
    pub written: Written,
    /// The follow-up's words, where it answered them.
    pub follow_up: Option<String>,
}

/// Why a song has no plan.
#[derive(Clone, Debug, PartialEq)]
pub enum Failure {
    /// Its answer had nothing to draw (it would get none again): kept.
    Refused(String),
    /// Its thought was cut off at [`MOST_THOUGHT`] (written greedily, it
    /// would be cut off again): kept.
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

/// A plan written ([`plan`]), or why there is none, its turns, every
/// mending of its answer and every follow-up, in words (for the trace),
/// and whether its follow-up was short of room in the context (its
/// questions then taken as the nearest valid choices: planned again in a
/// larger one).
#[derive(Clone, Debug)]
pub struct Planned {
    /// The plan.
    pub plan: Result<Plan, Failure>,
    /// The turns, in order.
    pub tries: Vec<Try>,
    /// What was mended and asked.
    pub notes: Vec<String>,
    /// The follow-up was short of room in the context.
    pub cramped: bool,
}

/// `song`'s plan as `writer` writes it after `asked` (the song's prompt,
/// [`prompt::prompt`]) in a context of `context` positions, read
/// forgivingly; what needs the model's judgement asked once more in the
/// same conversation ([`follow_up`]), the patch it answers merged; a song
/// with no lines is not asked at all. `tell` is told the prose as it is
/// written, a note at the follow-up, and why there is no plan, if there is
/// none.
pub fn plan(
    writer: &dyn Writer,
    song: &Song,
    (asked, context): (&str, usize),
    tell: &mut dyn FnMut(Story),
) -> Planned {
    let mut planned = Turns::default();
    // Not the model's to break: with no line to tell of there is no plan.
    if song.lines.is_empty() {
        return failed(planned.with(Err(Failure::Refused(NO_LINES.into()))), tell);
    }
    let mut talk = writer.converse(context);
    let mut prose = Prose::new();
    let said = talk.say(
        asked,
        (MOST_THOUGHT, MOST_ANSWER),
        &mut |writing| match writing {
            Writing::Thinking => tell(Story::Thinking),
            Writing::Answer(piece) => {
                let words = prose.feed(piece);
                if !words.is_empty() {
                    tell(Story::Words(words));
                }
            }
        },
    );
    let written = match said {
        Ok(written) => written,
        Err(failure) => return failed(planned.with(Err(failure)), tell),
    };
    let ended = thought_ended(&written);
    let answer = written.text.clone();
    if !written.stopped && ended.is_ok() {
        planned.notes.push(format!(
            "the answer was cut off at {MOST_ANSWER} tokens: all that is whole kept"
        ));
    }
    planned.tries.push(Try {
        written,
        follow_up: None,
    });
    if let Err(failure) = ended {
        return failed(planned.with(Err(failure)), tell);
    }
    let mut read = match check::read(&answer, song) {
        Ok(read) => read,
        Err(refused) => {
            planned.notes.extend(refused.notes);
            return failed(planned.with(Err(Failure::Refused(refused.why))), tell);
        }
    };
    let questions = read.questions();
    if !questions.is_empty() {
        let (answers, try_, cramped) = follow_up(&mut *talk, &questions, &mut read.notes, tell);
        planned.tries.extend(try_);
        planned.cramped = cramped;
        read.settle(&questions, &answers);
    }
    drop(talk);
    let (plan, notes) = read.plan();
    planned.notes.extend(notes);
    planned.with(Ok(plan))
}

/// A plan's turns, its mendings and whether its follow-up was short of
/// room, as they are gathered ([`plan`]).
#[derive(Default)]
struct Turns {
    tries: Vec<Try>,
    notes: Vec<String>,
    cramped: bool,
}

impl Turns {
    /// The plan written: `plan`, with these turns.
    fn with(self, plan: Result<Plan, Failure>) -> Planned {
        Planned {
            plan,
            tries: self.tries,
            notes: self.notes,
            cramped: self.cramped,
        }
    }
}

/// `questions` asked in `talk` (told to `tell`): their answers in order,
/// the turn, and whether the context left it short of room. Not answered
/// (none for each: the nearest valid choices) where the follow-up's thought
/// does not end, the context has no room for it, or the model fails (its
/// first answer is kept all the same); each said in `notes`.
fn follow_up(
    talk: &mut dyn Conversation,
    questions: &[Question],
    notes: &mut Vec<String>,
    tell: &mut dyn FnMut(Story),
) -> (Vec<Option<Value>>, Option<Try>, bool) {
    let ask = follow::ask(questions);
    tell(Story::Words(format!(
        "\n(Settling {} with the model.)\n",
        match questions.len() {
            1 => "one thing".to_owned(),
            n => format!("{n} things"),
        }
    )));
    let none = vec![None; questions.len()];
    match talk.say(&ask, (FOLLOW_UP_THOUGHT, MOST_PATCH), &mut |_| {}) {
        Ok(written) => {
            let full = written.thought_tokens == written.thought_room;
            let cramped = written.thinking && full && written.thought_room < FOLLOW_UP_THOUGHT;
            let answers = if written.thinking {
                notes.push(
                    "the follow-up's thought did not end: the nearest valid choices taken".into(),
                );
                none
            } else {
                follow::answers(&written.text, questions.len(), notes)
            };
            let turn = Try {
                written,
                follow_up: Some(ask),
            };
            (answers, Some(turn), cramped)
        }
        Err(failure) => {
            notes.push(format!(
                "no follow-up ({}): the nearest valid choices taken",
                failure.why()
            ));
            (none, None, matches!(failure, Failure::Short(_)))
        }
    }
}

/// Why a song with no timed sung lines has no plan.
pub const NO_LINES: &str = "the song has no timed sung lines";

/// Whether the model's thought in `written` ended; else why not: short of
/// room where its room was full and the context, not its most, made it so;
/// any other end within the thought (at its most, or on a stop token) cut
/// off for good (written greedily, it ends there again).
fn thought_ended(written: &Written) -> Result<(), Failure> {
    if !written.thinking {
        return Ok(());
    }
    let why = format!(
        "the model's thought did not end (cut off after {} tokens)",
        written.thought_tokens
    );
    let full = written.thought_tokens == written.thought_room;
    Err(if full && written.thought_room < MOST_THOUGHT {
        Failure::Short(why)
    } else {
        Failure::CutOff(why)
    })
}

/// `planned` with no plan: why told.
fn failed(planned: Planned, tell: &mut dyn FnMut(Story)) -> Planned {
    if let Err(failure) = &planned.plan {
        tell(Story::Words(format!(
            "\nNo story for this song: {}\n",
            failure.why()
        )));
    }
    planned
}

/// What is kept for a song: its plan, or that it has none and why.
#[derive(Clone, Debug, PartialEq)]
pub enum Kept {
    /// Its plan.
    Plan {
        /// The plan.
        plan: Box<Plan>,
        /// The context it was settled in, where its follow-up was short of
        /// room there (planned again in a larger one).
        settled_short_at: Option<usize>,
    },
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
    /// short of room, with that context); each turn, the follow-up and
    /// every mending noted in the trace.
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
                "story: {name}: {}{} prompt tokens in {:.1} s, thought {} in {:.1} s, {} written in {:.1} s{}",
                if t.follow_up.is_some() { "follow-up: " } else { "" },
                w.prompt_tokens,
                w.prompt_time.as_secs_f64(),
                w.thought_tokens,
                w.thought_time.as_secs_f64(),
                w.answer_tokens,
                w.answer_time.as_secs_f64(),
                if w.stopped { "" } else { " (cut off)" },
            ));
            let line = |text: &str| text.trim().replace('\n', " | ");
            if let Some(asked) = &t.follow_up {
                crate::trace::note(format_args!("story: {name}: asked again: {}", line(asked)));
            }
            crate::trace::note(format_args!("story: {name}: answered: {}", line(&w.text)));
        }
        for note in &planned.notes {
            let (how, note) = (note.strip_prefix(check::READ))
                .map_or(("mended:", note.as_str()), |n| {
                    (check::READ, n.trim_start())
                });
            crate::trace::note(format_args!("story: {name}: {how} {note}"));
        }
        let kept = match &planned.plan {
            Ok(plan) => Kept::Plan {
                plan: Box::new(plan.clone()),
                settled_short_at: planned.cramped.then_some(context),
            },
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
    /// prompt, the rooms of the answer, the thought and the follow-up, the
    /// follow-up's words and choices and the most scenes (the context is
    /// kept apart, for a song short of room only: [`Kept::Short`], and a
    /// plan settled short of room).
    fn hash(&self) -> String {
        cache::name(&format!(
            "{MOST_ANSWER} {MOST_THOUGHT} {FOLLOW_UP_THOUGHT} {MOST_PATCH} {MOST_SCENES} {} {}\n{}",
            follow::CHOICES,
            follow::PREAMBLE,
            self.asked
        ))
    }
}

/// What starts the line of what made a kept plan's file.
const MADE: &str = "made story";
/// What starts the line of what it was asked.
const ASKED: &str = "asked";
/// The line before its plan (followed by the context it was settled in,
/// where it was short of room there).
const PLAN: &str = "plan";
/// What starts the line of why there is none.
const NONE: &str = "none";
/// What starts the line of the context short of room, and why.
const SHORT: &str = "short";

/// What is kept for a song as its file holds it: its header, what made
/// it, the hash of what was `asked`, then `plan` (and the context it was
/// settled in, where it was short of room there) and the plan as JSON, or
/// `none` and why, or `short`, the context and why.
pub fn to_text(asked: &str, kept: &Kept) -> String {
    let line = |why: &str| why.replace('\n', " ");
    let what = match kept {
        Kept::Plan {
            plan,
            settled_short_at: None,
        } => format!("{PLAN}\n{}\n", plan.to_json()),
        Kept::Plan {
            plan,
            settled_short_at: Some(context),
        } => format!("{PLAN} {context}\n{}\n", plan.to_json()),
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
        return Plan::from_json(json).ok().map(|p| Kept::Plan {
            plan: Box::new(p),
            settled_short_at: None,
        });
    }
    if let Some(settled) = rest.strip_prefix(&format!("{PLAN} ")) {
        let (context, json) = settled.split_once('\n')?;
        let plan = Plan::from_json(json).ok()?;
        return Some(Kept::Plan {
            plan: Box::new(plan),
            settled_short_at: Some(context.parse().ok()?),
        });
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
