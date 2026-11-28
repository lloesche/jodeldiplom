//! The stories' thread: every song of the queue whose lyrics are settled
//! ([`StoryWorker::want`], in the queue's order from the song playing)
//! gets its story made in the background, as a step of its preparation
//! beside its analysis and its lyrics, told to the app as [`Job::Story`]
//! (working in each of its turns, then ready: [`Facts::Story`]). It goes
//! in rounds ([`round`]), by model, each turn a job of the GPU's stages
//! ([`crate::work::stages`]). A round starts once the queue has settled,
//! no song asked for in [`SETTLE`] (the songs' lyrics come one after
//! another as the queue is got ready, and a round begun with the first
//! alone would plan the next only after it drew the first: each model
//! loaded twice where once does); a song wanted while a round is under
//! way waits for the next:
//!
//! 0. every song looked at, no model run: one whose story is made (its
//!    plan kept, its pictures drawn, its scenes taken apart), or that has
//!    none for good (no lyrics or no timed sung lines, or a plan that did
//!    not hold), is told ready at once ([`StoryFact`]);
//! 1. the plans of the songs still to plan, by the language model
//!    ([`Stage::StoryPlan`], [`crate::work::story::StoryJob`]; a plan kept
//!    is not asked again), in order, while the drawings planned would hold
//!    less than [`ENCODED_AHEAD_MB`] of prompts encoded;
//! 2. then every song's prompts of its pictures still to draw, by the
//!    image model's text encoder ([`Stage::StoryEncode`],
//!    [`Drawing::encode`]);
//! 3. then every song's pictures, by the image model ([`Stage::StoryDraw`],
//!    [`Drawing::draw`]), and its scenes described ([`Drawing::keep`]);
//! 4. then every song's scenes taken apart for the backdrop by the depth
//!    and cut-out models ([`Stage::Scenes`], [`scenes::take_apart`]), the
//!    song told ready and the song playing's story told of it
//!    ([`OnDrawn`]).
//!
//! The turns of a model follow each other, so it stays loaded through
//! them, and on a small card no model is loaded twice in a round. A song
//! no longer wanted (out of the queue, or other lyrics of it), or the
//! worker stopped, is given up before each of its turns and between its
//! pictures, what is done kept; the songs wanted then are taken up in the
//! next round. Each step begun is told to the app ([`Story`]: the queue's
//! caption and the preparation view say it).
//!
//! Between the encoder's turns and the image model's, a round holds its
//! songs' prompts encoded, about 16 MB a picture still to draw
//! ([`PROMPT_BYTES`]): some 400 MB a song of 25 pictures, a queue's worth
//! far more. A round plans only as many songs as would fill
//! [`ENCODED_AHEAD_MB`], and encodes until the prompts held do; the rest
//! wait for the next round, at once (on a small card, each model loaded
//! once more for them), so the song playing next is finished first and the
//! most held is that and one song's more.
//!
//! The models are held each in a [`jodel_gpu::Kept`] registered for its
//! stage ([`GpuModels`]): set up on its first use in a turn, let go by the
//! stages as they plan. A model that cannot be set up, or fails, or a file
//! that cannot be read, leaves the song without its story now (told ready
//! as not made now, [`StoryFact::NotNow`], and noted in the trace): it is
//! left out of the rounds for [`RETRY`], twice as long each time again in
//! a row up to [`MOST_RETRY`] ([`Tried`]), so it neither holds the songs
//! after it off nor loads the models again and again; the thread waits
//! for it ([`Next::At`]), and takes up a new ask at once meanwhile. A
//! model that cannot be set up is told to setup's repair
//! ([`crate::setup::Repair`]), which gets it again where its files are
//! gone or damaged, and has it tried again at once on the song's next
//! try. A picture that is not one
//! any more (damaged) is let go as it is read, and drawn again alone.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use jodel_gpu::image::{Drawer, TextEncoder};
use jodel_gpu::{Gemma, Precision, Resident};
use jodel_lyrics::Lyrics;

use super::draw::{Drawing, Drawn, Encoded, PROMPT_BYTES};
use super::{Failure, Kept, StoryJob};
use crate::core::prep::Story;
use crate::core::story::{Plan, Song};
use crate::core::{Action, Facts, Heard, Job, StoryFact, StorySong};
use crate::setup::Unreadable;
use crate::store::mover::locate;
use crate::work::outbox::Outbox;
use crate::work::scenes::{self, GpuNets, SceneModels, SceneStore, Scenes, TakenApart};
use crate::work::stages::{self, Stage};
use crate::work::story_models::StoryModels;
use crate::work::{song_name, Queue};

/// What is told when a song's story is finished (its scenes drawn and
/// taken apart): the song playing's story, which takes it up if it is
/// that song ([`crate::work::scenes::PlayingStory::on_drawn`]).
pub type OnDrawn = Arc<dyn Fn(&Path) + Send + Sync>;

/// A song's lyrics as settled (for its story): none where it has none,
/// why not where they cannot be read now
/// ([`crate::work::lyrics::settled`] in the app).
pub type LyricsOf = Arc<dyn Fn(&Path) -> Result<Option<Lyrics>, String> + Send + Sync>;

/// The prompts a round holds encoded, in megabytes, and plans as many
/// songs as would fill that, before it leaves the rest of its songs to the
/// next round: about one song's.
pub const ENCODED_AHEAD_MB: usize = 512;

/// How long a song left without its story now (a model failed, a file
/// could not be read) is left out of the rounds the first time; each time
/// again in a row (for the same lyrics) twice as long, up to
/// [`MOST_RETRY`].
pub const RETRY: Duration = Duration::from_secs(60);

/// The longest a song left without its story now is left out.
pub const MOST_RETRY: Duration = Duration::from_secs(15 * 60);

/// How long the thread waits after the last ask before it starts a round:
/// the queue settled, every song whose lyrics came one after another is
/// in the round, each model loaded once for them all. The wait needs no
/// cap: the app asks only when the songs wanted or their lyrics change
/// (`crate::core::work::want_stories`), so it ends once the queue's lyrics
/// are in.
pub const SETTLE: Duration = Duration::from_secs(2);

/// The story's models, each run in its stage's turn.
pub trait Models: Send + Sync {
    /// The context the language model writes in on this card
    /// ([`super::context`]); why there is none now.
    fn context(&self) -> Result<usize, String>;
    /// `job` planned, its news told to `tell`.
    fn plan(&self, job: &StoryJob, tell: &mut dyn FnMut(Story)) -> Result<Plan, Failure>;
    /// The prompts of `drawing`'s pictures still to draw, encoded.
    fn encode(&self, drawing: &Drawing) -> Result<Encoded, String>;
    /// `drawing`'s pictures of `encoded` drawn while `still` wanted, each
    /// picture begun and each scene told to `tell`.
    fn draw(
        &self,
        drawing: &Drawing,
        encoded: &mut Encoded,
        still: &dyn Fn() -> bool,
        tell: &mut dyn FnMut(Story),
    ) -> Result<Drawn, String>;
    /// `scenes` (kept in `store`) taken apart while `still` wanted
    /// ([`scenes::take_apart`]).
    fn take_apart(
        &self,
        scenes: &Scenes,
        store: &SceneStore,
        still: &dyn Fn() -> bool,
    ) -> Result<TakenApart, String>;
}

/// The language model's file and its context on this card, decided once
/// (the first time it can be: the file there, a GPU) and kept: the
/// stages' count of its room, the room each plan makes and the comparison
/// of a song short of room take the same value ([`super::context`] works
/// it out from a budget that moves).
struct Language {
    file: PathBuf,
    context: OnceLock<usize>,
}

impl Language {
    /// Its context, decided on the first ask that can.
    fn context(&self) -> Result<usize, String> {
        if let Some(&context) = self.context.get() {
            return Ok(context);
        }
        let context = super::context(&locate(&self.file))?;
        Ok(*self.context.get_or_init(|| context))
    }

    /// Its file, where it is now.
    fn file(&self) -> PathBuf {
        locate(&self.file)
    }
}

/// The story's models on the GPU, where setup put them: each held for
/// its stage and registered there, set up on its first use in a turn.
pub struct GpuModels {
    /// The language model's file and the context decided for it.
    language: Arc<Language>,
    /// The language model, held for the story's plans.
    writer: Arc<jodel_gpu::Kept<Gemma>>,
    encoder: Arc<jodel_gpu::Kept<TextEncoder>>,
    drawer: Arc<jodel_gpu::Kept<Drawer>>,
    nets: GpuNets,
}

impl GpuModels {
    /// The models in `files`, and the scenes' in `scenes`, registered for
    /// their stages; one that cannot be set up told to `unreadable`
    /// (checked, and got again where it is gone or damaged; then tried
    /// again at once).
    pub fn new(files: &StoryModels, scenes: SceneModels, unreadable: Unreadable) -> GpuModels {
        use crate::work::story_models::{
            ENCODER_DIR, LANGUAGE_FILE, PIPELINE_DIR, TRANSFORMER_DIR,
        };
        let language = Arc::new(Language {
            file: files.language.clone(),
            context: OnceLock::new(),
        });
        let (planned, opened) = (Arc::clone(&language), Arc::clone(&language));
        let told = unreadable.clone();
        let writer = Arc::new(jodel_gpu::Kept::new(
            move || {
                let context = planned.context().ok()?;
                Gemma::planned(&planned.file(), context).ok()
            },
            Box::new(move || {
                let opened = Gemma::open(&opened.file(), Precision::Fast);
                let gemma = told.report(LANGUAGE_FILE, opened)?;
                let name = gemma.gpu_name().to_owned();
                Ok((gemma, name))
            }),
        ));
        unreadable.retries(LANGUAGE_FILE, &writer);
        let encoder = files.encoder.clone();
        let told = unreadable.clone();
        let encoder = Arc::new(jodel_gpu::Kept::new(
            || Some(TextEncoder::planned()),
            Box::new(move || {
                let loaded = TextEncoder::load(&locate(&encoder), Precision::Fast);
                let model = told.report(ENCODER_DIR, loaded)?;
                let name = model.gpu_name().to_owned();
                Ok((model, name))
            }),
        ));
        unreadable.retries(ENCODER_DIR, &encoder);
        let (transformer, pipeline) = (files.transformer.clone(), files.pipeline.clone());
        let told = unreadable.clone();
        let drawer = Arc::new(jodel_gpu::Kept::new(
            || Some(super::draw::memory()),
            Box::new(move || {
                let (t, p) = (locate(&transformer), locate(&pipeline));
                // Either folder may be the one at fault: both checked.
                let loaded = Drawer::load(&t, &p, Precision::Fast);
                let model = told.report(PIPELINE_DIR, told.report(TRANSFORMER_DIR, loaded))?;
                let name = model.gpu_name().to_owned();
                Ok((model, name))
            }),
        ));
        unreadable.retries(TRANSFORMER_DIR, &drawer);
        unreadable.retries(PIPELINE_DIR, &drawer);
        stages::register(Stage::StoryPlan, Arc::clone(&writer) as Arc<dyn Resident>);
        stages::register(
            Stage::StoryEncode,
            Arc::clone(&encoder) as Arc<dyn Resident>,
        );
        stages::register(Stage::StoryDraw, Arc::clone(&drawer) as Arc<dyn Resident>);
        GpuModels {
            language,
            writer,
            encoder,
            drawer,
            nets: GpuNets::new(scenes, unreadable),
        }
    }
}

/// A model that cannot be set up fails the song's plan as a model's
/// failure does ([`Failure::Model`]: not kept); one set up again after
/// its device was lost runs its job again, the plan's words already told
/// not told twice ([`past`]).
impl Models for GpuModels {
    fn context(&self) -> Result<usize, String> {
        self.language.context()
    }

    fn plan(&self, job: &StoryJob, tell: &mut dyn FnMut(Story)) -> Result<Plan, Failure> {
        let context = self.context().map_err(Failure::Model)?;
        let told = &mut 0;
        (self.writer)
            .run(|gemma| Ok(job.run(gemma, context, &mut past(told, &mut *tell))))
            .unwrap_or_else(|why| Err(Failure::Model(why)))
    }

    fn encode(&self, drawing: &Drawing) -> Result<Encoded, String> {
        self.encoder.run(|encoder| drawing.encode(encoder))
    }

    fn draw(
        &self,
        drawing: &Drawing,
        encoded: &mut Encoded,
        still: &dyn Fn() -> bool,
        tell: &mut dyn FnMut(Story),
    ) -> Result<Drawn, String> {
        (self.drawer).run(|drawer| drawing.draw(encoded, drawer, still, &mut *tell))
    }

    fn take_apart(
        &self,
        scenes: &Scenes,
        store: &SceneStore,
        still: &dyn Fn() -> bool,
    ) -> Result<TakenApart, String> {
        scenes::take_apart(scenes, store, &self.nets, still)
    }
}

/// `tell`, but for the story's words past the first `told` bytes: those a
/// run before told (the model writes greedily, so a run again writes the
/// same), `told` counting them; a thought begun among them (told before)
/// is not told again.
fn past<'a>(told: &'a mut usize, tell: &'a mut dyn FnMut(Story)) -> impl FnMut(Story) + 'a {
    let mut seen = 0;
    move |story| match story {
        Story::Thinking if seen < *told => {}
        Story::Words(words) => {
            let end = seen + words.len();
            let skip = (told.saturating_sub(seen)..=words.len())
                .find(|&k| words.is_char_boundary(k))
                .unwrap_or(words.len());
            if skip < words.len() {
                tell(Story::Words(words[skip..].to_owned()));
            }
            *told = (*told).max(end);
            seen = end;
        }
        story => tell(story),
    }
}

/// Where the stories are kept and what they are made from: the plans (the
/// data folder's `stories/`), the songs' scenes, and the songs' lyrics.
pub struct Places {
    /// The plans' folder.
    pub stories: PathBuf,
    /// The songs' scenes.
    pub scenes: SceneStore,
    /// The songs' lyrics.
    pub lyrics: LyricsOf,
}

/// What the thread is asked: the songs wanted, how many times they were
/// asked for and when last, and how long it waits after an ask before it
/// starts a round ([`SETTLE`]).
struct Asked {
    songs: Vec<StorySong>,
    times: u64,
    at: Instant,
    settle: Duration,
}

impl Asked {
    /// Nothing asked yet, a round waiting `settle` after each ask.
    fn new(settle: Duration) -> Asked {
        Asked {
            songs: Vec::new(),
            times: 0,
            // Means nothing before the first ask: no round is due then.
            at: Instant::now(),
            settle,
        }
    }

    /// `songs` asked for at `now`.
    fn ask(&mut self, songs: Vec<StorySong>, now: Instant) {
        self.songs = songs;
        self.times += 1;
        self.at = now;
    }

    /// When the queue has settled: `settle` after the last ask.
    fn settles(&self) -> Instant {
        self.at + self.settle
    }

    /// The songs for a round at `now`, after `next` and with the asks up
    /// to `seen` taken up (counted on to these): where one is due (asked
    /// anew, songs left for it, or a song to be tried again) and the queue
    /// has settled. Else when to look again besides the next ask: once
    /// the queue settles where a round is due, once a song is to be tried
    /// again where one is.
    fn round_due(
        &self,
        next: Next,
        seen: &mut u64,
        now: Instant,
    ) -> Result<Vec<StorySong>, Option<Instant>> {
        let retry = next.due();
        let late = retry.is_some_and(|at| now >= at);
        if !(next == Next::Round || late || self.times != *seen) {
            return Err(retry);
        }
        if now < self.settles() {
            return Err(Some(self.settles()));
        }
        *seen = self.times;
        Ok(self.songs.clone())
    }

    /// Whether `song` is still wanted, of the very same lyrics.
    fn still(&self, song: &StorySong) -> bool {
        self.songs.contains(song)
    }
}

/// What the thread keeps from round to round: the songs told ready (made,
/// or none for good) with the lyrics they were made of
/// ([`StorySong::lyrics`]), not looked at again for them; and the songs
/// left without their story now, each with its lyrics, when it is tried
/// again and how long it waited last ([`RETRY`]).
#[derive(Debug, Default)]
pub struct Tried {
    finished: HashMap<PathBuf, u64>,
    failed: HashMap<PathBuf, Failed>,
}

/// A song left without its story now ([`Tried`]).
#[derive(Clone, Copy, Debug)]
struct Failed {
    lyrics: u64,
    again: Instant,
    wait: Duration,
}

impl Tried {
    /// Whether `song` is left out of a round at `now`: finished for its
    /// lyrics, or left without its story now for them and not to be tried
    /// again yet.
    fn rests(&self, song: &StorySong, now: Instant) -> bool {
        let failed = self.failed.get(&song.path);
        self.finished.get(&song.path) == Some(&song.lyrics)
            || failed.is_some_and(|f| f.lyrics == song.lyrics && now < f.again)
    }

    /// `song` finished for its lyrics.
    fn finish(&mut self, song: &StorySong) {
        self.failed.remove(&song.path);
        self.finished.insert(song.path.clone(), song.lyrics);
    }

    /// `song` left without its story now at `now`: how long it is left
    /// out ([`RETRY`], twice the last for the same lyrics, at most
    /// [`MOST_RETRY`]).
    fn fail(&mut self, song: &StorySong, now: Instant) -> Duration {
        let wait = match self.failed.get(&song.path) {
            Some(f) if f.lyrics == song.lyrics => (f.wait * 2).min(MOST_RETRY),
            _ => RETRY,
        };
        let failed = Failed {
            lyrics: song.lyrics,
            again: now + wait,
            wait,
        };
        self.failed.insert(song.path.clone(), failed);
        wait
    }

    /// When the first of `songs` left without its story now for its
    /// lyrics is to be tried again.
    fn due<'a>(&self, songs: impl Iterator<Item = &'a StorySong>) -> Option<Instant> {
        let failed = |s: &StorySong| self.failed.get(&s.path).filter(|f| f.lyrics == s.lyrics);
        songs.filter_map(|s| failed(s).map(|f| f.again)).min()
    }
}

/// What comes after a round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Next {
    /// Nothing, until the songs wanted change.
    Wait,
    /// Another round at once: it left songs for the next.
    Round,
    /// Another round then (or once the songs wanted change): a song left
    /// without its story now is to be tried again.
    At(Instant),
}

impl Next {
    /// When the thread is to go on by itself, where it is not asked anew
    /// first.
    fn due(self) -> Option<Instant> {
        match self {
            Next::At(at) => Some(at),
            Next::Wait | Next::Round => None,
        }
    }
}

/// The stories' thread.
pub struct StoryWorker {
    asked: Arc<Queue<Asked>>,
}

impl Drop for StoryWorker {
    fn drop(&mut self) {
        self.asked.stop();
    }
}

impl StoryWorker {
    /// Starts the thread: the stories made with `models`, kept in
    /// `places`, told on `outbox`, and each finished told to `on_drawn`.
    pub fn start(
        models: Arc<dyn Models>,
        places: Places,
        outbox: Outbox,
        on_drawn: OnDrawn,
    ) -> StoryWorker {
        let asked = Arc::new(Queue::new(Asked::new(SETTLE)));
        let thread = Arc::clone(&asked);
        std::thread::Builder::new()
            .name("stories".into())
            .spawn(move || {
                jodel_gpu::pace::background();
                // A round once one is due and the queue has settled
                // ([`Asked::round_due`]).
                let (mut seen, mut next, mut tried) = (0, Next::Wait, Tried::default());
                while let Some(songs) =
                    thread.next_by(|a: &mut Asked| a.round_due(next, &mut seen, Instant::now()))
                {
                    let still = |song: &StorySong| !thread.stopped() && thread.lock().still(song);
                    let ask = Ask {
                        models: &*models,
                        places: &places,
                        still: &still,
                        outbox: &outbox,
                        on_drawn: &*on_drawn,
                    };
                    next = round(&songs, &ask, &mut tried);
                }
            })
            .expect("a thread");
        StoryWorker { asked }
    }

    /// The stories of exactly `songs` wanted, in this order: the round
    /// under way gives up the songs no longer wanted, and the next takes
    /// these up once the queue has settled ([`SETTLE`]).
    pub fn want(&self, songs: Vec<StorySong>) {
        self.asked.change(|a| a.ask(songs, Instant::now()));
    }
}

/// What a round works with: the models, where the stories are kept,
/// whether a song is still wanted, where its news go, and what is told of
/// each song finished.
pub struct Ask<'a> {
    /// The models.
    pub models: &'a dyn Models,
    /// Where the stories are kept.
    pub places: &'a Places,
    /// Whether a song is still wanted.
    pub still: &'a dyn Fn(&StorySong) -> bool,
    /// Where its news go.
    pub outbox: &'a Outbox,
    /// Told of each song finished.
    pub on_drawn: &'a dyn Fn(&Path),
}

/// Where a song's story stands in a round: to be planned, drawn (its
/// prompts encoded, or none to encode), its scenes taken apart.
enum Stands {
    /// To be planned.
    Plan(StoryJob),
    /// Planned, its prompts to be encoded.
    Encode(Drawing),
    /// Its prompts encoded (none: every picture drawn), to be drawn.
    Draw(Drawing, Option<Encoded>),
    /// Drawn, its scenes to be taken apart.
    Cut(Scenes),
}

/// The songs of a round, each where its story stands.
type Songs<'a> = Vec<(&'a StorySong, Stands)>;

/// The bytes of prompts a round holds encoded, and plans for, at most
/// ([`ENCODED_AHEAD_MB`]).
const AHEAD: usize = ENCODED_AHEAD_MB << 20;

/// One round: the stories of `songs` (those still wanted that `tried` does
/// not leave out: finished for their lyrics, or left without their story
/// now not long ago; each given up once the ask says it is no longer
/// wanted) looked at, then planned (while the songs planned would hold
/// less than [`ENCODED_AHEAD_MB`] of prompts encoded), then their prompts
/// encoded (while those held are less), then their pictures drawn and
/// described, then their scenes taken apart, by the ask's models in their
/// turns; each song finished (made, or none for good) told ready and kept
/// in `tried`, each left without its story now told so and kept there to
/// be tried again later. What comes next: a round at once for the songs it
/// left, else one when the first song still wanted and left without its
/// story now is to be tried again, else none.
pub fn round(songs: &[StorySong], ask: &Ask, tried: &mut Tried) -> Next {
    let now = Instant::now();
    let looked: Vec<&StorySong> = (songs.iter())
        .filter(|s| !tried.rests(s, now) && (ask.still)(s))
        .collect();
    let mut stands = Songs::new();
    for song in looked {
        match stands_of(song, ask) {
            Ok(stand) => stands.push((song, stand)),
            Err(fact) => finish(ask, tried, song, fact),
        }
    }
    let (mut planned, mut held, mut left) = (0, 0, false);
    let stands = step(stands, ask, |song, stand| {
        plan(ask, tried, (song, stand), (&mut planned, &mut left))
    });
    let stands = step(stands, ask, |song, stand| {
        encode(ask, tried, (song, stand), (&mut held, &mut left))
    });
    let stands = step(stands, ask, |song, stand| draw(ask, tried, song, stand));
    step(stands, ask, |song, stand| cut(ask, tried, song, stand));
    if left {
        return Next::Round;
    }
    match tried.due(songs.iter().filter(|s| (ask.still)(s))) {
        Some(at) => Next::At(at),
        None => Next::Wait,
    }
}

/// `songs` still wanted, each on through `then` (none: no further this
/// round).
fn step<'a>(
    songs: Songs<'a>,
    ask: &Ask,
    mut then: impl FnMut(&StorySong, Stands) -> Option<Stands>,
) -> Songs<'a> {
    (songs.into_iter())
        .filter(|(song, _)| (ask.still)(song))
        .filter_map(|(song, stand)| Some((song, then(song, stand)?)))
        .collect()
}

/// The language model's turn for `song` where it is to be planned (while
/// the prompts of the drawings planned this round, `planned` bytes once
/// encoded, are less than [`ENCODED_AHEAD_MB`]: else `left` for the next
/// round): on to its drawing; finished where its plan did not hold or was
/// cut off (kept as having none: [`Failure::for_good`]), not now where the
/// model failed. A drawing planned before counts its prompts as it
/// passes.
fn plan(
    ask: &Ask,
    tried: &mut Tried,
    (song, stand): (&StorySong, Stands),
    (planned, left): (&mut usize, &mut bool),
) -> Option<Stands> {
    let job = match stand {
        Stands::Plan(job) if *planned < AHEAD => job,
        Stands::Plan(_) => {
            *left = true;
            return None;
        }
        Stands::Encode(drawing) => {
            *planned += drawing.to_encode() * PROMPT_BYTES;
            return Some(Stands::Encode(drawing));
        }
        stand => return Some(stand),
    };
    let made = turn(ask, song, Stage::StoryPlan, Some(Story::Planning), |tell| {
        ask.models.plan(&job, tell)
    });
    match made {
        Ok(plan) => {
            let drawing = Drawing::new(song.path.clone(), plan, &ask.places.scenes);
            *planned += drawing.to_encode() * PROMPT_BYTES;
            Some(Stands::Encode(drawing))
        }
        Err(failure) if failure.for_good() => {
            finish(ask, tried, song, StoryFact::NoStory);
            None
        }
        Err(failure) => {
            not_now(ask, tried, song, failure.why().into());
            None
        }
    }
}

/// The text encoder's turn for `song` where its prompts are to be
/// encoded, the prompts the round holds counted in `held`; none where it
/// holds [`ENCODED_AHEAD_MB`] already (`left` for the next round).
fn encode(
    ask: &Ask,
    tried: &mut Tried,
    (song, stand): (&StorySong, Stands),
    (held, left): (&mut usize, &mut bool),
) -> Option<Stands> {
    let Stands::Encode(drawing) = stand else {
        return Some(stand);
    };
    if *held >= AHEAD {
        *left = true;
        return None;
    }
    if drawing.to_encode() == 0 {
        return Some(Stands::Draw(drawing, None));
    }
    let read = turn(ask, song, Stage::StoryEncode, Some(Story::Reading), |_| {
        ask.models.encode(&drawing)
    });
    match read {
        Ok(prompts) => {
            *held += prompts.bytes();
            Some(Stands::Draw(drawing, Some(prompts)))
        }
        Err(why) => {
            not_now(ask, tried, song, why);
            None
        }
    }
}

/// The image model's turn for `song` where its pictures are to be drawn
/// (each picture begun and each scene told), then its scenes described:
/// on to taking them apart; none where it was given up (what is drawn
/// kept) or not done now.
fn draw(ask: &Ask, tried: &mut Tried, song: &StorySong, stand: Stands) -> Option<Stands> {
    let Stands::Draw(drawing, prompts) = stand else {
        return Some(stand);
    };
    let still = || (ask.still)(song);
    let drawn = match prompts {
        Some(mut prompts) => turn(ask, song, Stage::StoryDraw, None, |tell| {
            ask.models.draw(&drawing, &mut prompts, &still, tell)
        }),
        None => Ok(Drawn::All),
    };
    let kept = match drawn {
        Ok(Drawn::All) => drawing.keep(),
        Ok(Drawn::GivenUp) => return None,
        Err(why) => Err(why),
    };
    match kept.and_then(|()| scenes_of(&song.path, ask.places)) {
        Ok(scenes) => Some(Stands::Cut(scenes)),
        Err(why) => {
            not_now(ask, tried, song, why);
            None
        }
    }
}

/// The scenes' models' turn for `song` where its scenes are to be taken
/// apart: finished (told ready, and the song playing's story told of it),
/// each picture taken apart noted in the trace with its time; none where
/// it was given up (what is taken apart kept) or not done now.
fn cut(ask: &Ask, tried: &mut Tried, song: &StorySong, stand: Stands) -> Option<Stands> {
    let Stands::Cut(scenes) = stand else {
        return Some(stand);
    };
    let still = || (ask.still)(song);
    let store = &ask.places.scenes;
    let cut = turn(ask, song, Stage::Scenes, Some(Story::Cutting), |_| {
        ask.models.take_apart(&scenes, store, &still)
    });
    match cut {
        Ok(TakenApart::All(took)) => {
            let name = song_name(&song.path);
            for (k, t) in took.iter().enumerate() {
                if let Some(t) = t {
                    crate::trace::note(format_args!("scenes: {name} scene {k}: {t}"));
                }
            }
            finish(ask, tried, song, StoryFact::Made);
            (ask.on_drawn)(&song.path);
        }
        Ok(TakenApart::GivenUp) => {}
        Err(why) => not_now(ask, tried, song, why),
    }
    None
}

/// `job` for `song`, in `stage`'s turn: told working while it runs, with
/// the step `begun` (where the job does not tell its own) and the news it
/// tells (given `tell`), noted in the trace with how long it took (its
/// model's setup included) and whether it was done (why not is the
/// caller's to say).
fn turn<T, E>(
    ask: &Ask,
    song: &StorySong,
    stage: Stage,
    begun: Option<Story>,
    job: impl FnOnce(&mut dyn FnMut(Story)) -> Result<T, E>,
) -> Result<T, E> {
    let name = song_name(&song.path);
    stages::run(&[stage], &name, || {
        let started = Instant::now();
        let working = |on| {
            ask.outbox.send(Action::Working {
                path: song.path.clone(),
                job: Job::Story,
                on,
            });
        };
        working(true);
        let tell = &mut teller(ask.outbox, &song.path);
        if let Some(begun) = begun {
            tell(begun);
        }
        let made = job(tell);
        working(false);
        let how = if made.is_ok() { "done" } else { "not done" };
        crate::trace::took(
            format_args!("story: {name}: {stage:?} {how}"),
            started,
            true,
        );
        made
    })
}

/// Where `song`'s story stands, looking at what is kept in the places
/// `ask` has alone: to plan, to draw, to take apart; or how it ends
/// (`Err`): made, none for good (no lyrics, no timed sung lines, a plan
/// that did not hold, or no room for it in a context as large as the
/// language model's now), or not now (its lyrics, its file, its scenes,
/// or the model's context where a song short of room asks for it, not
/// read now).
fn stands_of(song: &StorySong, ask: &Ask) -> Result<Stands, StoryFact> {
    let places = ask.places;
    let lyrics = (places.lyrics)(&song.path).map_err(StoryFact::NotNow)?;
    let lyrics = lyrics.ok_or(StoryFact::NoStory)?;
    let job = job_of(&song.path, &lyrics, places)?;
    let plan = match job.kept() {
        Some(Kept::Plan(plan)) => *plan,
        Some(Kept::NoStory(_)) => return Err(StoryFact::NoStory),
        Some(Kept::Short { context: had, .. }) => {
            let context = ask.models.context().map_err(StoryFact::NotNow)?;
            return match context > had {
                true => Ok(Stands::Plan(job)),
                false => Err(StoryFact::NoStory),
            };
        }
        None => return Ok(Stands::Plan(job)),
    };
    let drawing = Drawing::new(song.path.clone(), plan, &places.scenes);
    if !drawing.drawn() {
        return Ok(Stands::Encode(drawing));
    }
    let scenes = scenes_of(&song.path, places).map_err(StoryFact::NotNow)?;
    match scenes::taken_apart(&scenes, &places.scenes) {
        Ok(true) => Err(StoryFact::Made),
        Ok(false) | Err(_) => Ok(Stands::Cut(scenes)),
    }
}

/// The scenes of the song at `path` as its drawing left them in `places`.
fn scenes_of(path: &Path, places: &Places) -> Result<Scenes, String> {
    (places.scenes.scenes_of(path)).ok_or_else(|| "its scenes are not described".to_owned())
}

/// `song` told ready with `fact`: made, or none for good, kept in `tried`
/// as finished for its lyrics; or not now ([`not_now`]).
fn finish(ask: &Ask, tried: &mut Tried, song: &StorySong, fact: StoryFact) {
    if let StoryFact::NotNow(why) = fact {
        return not_now(ask, tried, song, why);
    }
    tried.finish(song);
    told(ask, song, fact);
}

/// `song` left without its story now, for `why`: told so, noted in the
/// trace, and kept in `tried` to be tried again later ([`Tried::fail`]).
fn not_now(ask: &Ask, tried: &mut Tried, song: &StorySong, why: String) {
    let wait = tried.fail(song, Instant::now());
    let name = song_name(&song.path);
    crate::trace::note(format_args!(
        "story: {name}: no story now: {why} (tried again in {} s)",
        wait.as_secs()
    ));
    told(ask, song, StoryFact::NotNow(why));
}

/// `song`'s story told ready with `fact`.
fn told(ask: &Ask, song: &StorySong, fact: StoryFact) {
    ask.outbox.send(Action::Ready {
        path: song.path.clone(),
        job: Job::Story,
        facts: Some(Facts::Story {
            lyrics: song.lyrics,
            fact,
        }),
    });
}

/// What tells the story of the song at `path` on `outbox`.
fn teller<'a>(outbox: &'a Outbox, path: &'a Path) -> impl FnMut(Story) + 'a {
    move |story| {
        outbox.send(Action::Heard {
            path: path.to_owned(),
            part: Heard::Story(story),
        });
    }
}

/// The plan's job for the song at `path` with `lyrics`, kept in `places`;
/// none for good for a song with no timed sung lines, not now for one
/// that cannot be read now.
fn job_of(path: &Path, lyrics: &Lyrics, places: &Places) -> Result<StoryJob, StoryFact> {
    let info = jodel_playback::source::probe(path)
        .map_err(|e| StoryFact::NotNow(format!("its file could not be read: {e}")))?;
    let song = Song::of(&info, lyrics);
    if song.lines.is_empty() {
        return Err(StoryFact::NoStory);
    }
    Ok(StoryJob::new(path.to_owned(), song, places.stories.clone()))
}

#[cfg(test)]
mod tests;
