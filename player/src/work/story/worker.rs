//! The karaoke stories' thread: the songs Karaoke mode wants a story of
//! ([`StoryWorker::want`]) planned and drawn in rounds ([`round`]), by
//! model, each turn a job of the GPU's stages ([`crate::work::stages`]):
//!
//! 1. every song's plan, by the language model ([`Stage::StoryPlan`],
//!    [`crate::work::story::StoryJob`]; a plan kept is not asked again);
//! 2. then every song's prompts of its pictures still to draw, by the
//!    image model's text encoder ([`Stage::StoryEncode`],
//!    [`Drawing::encode`]);
//! 3. then every song's pictures, by the image model ([`Stage::StoryDraw`],
//!    [`Drawing::draw`]), and its scenes described ([`Drawing::keep`]),
//!    the covers' worker told ([`OnDrawn`]).
//!
//! The turns of a model follow each other, so it stays loaded through
//! them, and on a small card no model is loaded twice in a round. A song
//! no longer wanted (another song, or other lyrics of it), or the worker
//! stopped, is given up before each of its turns and between its
//! pictures, what is done kept; the songs wanted then are taken up in the
//! next round.
//!
//! Between the encoder's turns and the image model's, a round holds its
//! songs' prompts encoded, about 16 MB a picture still to draw: some
//! 400 MB a song of 25 pictures, so `1 + SONGS_AHEAD` songs' worth, 1.2 GB
//! ([`crate::core::karaoke::SONGS_AHEAD`]), were nothing to bound it. Once
//! the songs encoded hold [`ENCODED_AHEAD_MB`], the rest wait for the next
//! round (on a small card, each model loaded once more for them), so the
//! most held is that and one song's more.
//!
//! The models are held each in a [`jodel_gpu::Kept`] registered for its
//! stage ([`GpuModels`]): set up on its first use in a turn, let go by the
//! stages as they plan.
//!
//! A song with no timed sung lines gets no story; a model that cannot be
//! set up leaves the song without one now (noted in the trace), tried
//! again the next time it is wanted.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use jodel_gpu::image::{Drawer, TextEncoder};
use jodel_gpu::{Gemma, Precision, Resident};
use jodel_lyrics::Lyrics;

use super::draw::{Drawing, Drawn, Encoded};
use super::{Failure, Kept, StoryJob};
use crate::core::karaoke::StorySong;
use crate::core::prep::Story;
use crate::core::story::{Plan, Song};
use crate::core::{Action, Heard};
use crate::store::mover::locate;
use crate::work::outbox::Outbox;
use crate::work::scenes::SceneStore;
use crate::work::stages::{self, Stage};
use crate::work::story_models::StoryModels;
use crate::work::{song_name, Queue};

/// What is told when a song's story is drawn and described in its folder:
/// the covers' worker, which takes it up if the song plays
/// ([`crate::work::cover::CoverWorker::on_drawn`]).
pub type OnDrawn = Arc<dyn Fn(&Path) + Send + Sync>;

/// The prompts encoded a round holds, in megabytes, before it leaves the
/// rest of its songs to the next round: about one song's.
pub const ENCODED_AHEAD_MB: usize = 512;

/// The story's models, each run in its stage's turn.
pub trait Models: Send + Sync {
    /// `job` planned, its news told to `tell`.
    fn plan(&self, job: &StoryJob, tell: &mut dyn FnMut(Story)) -> Result<Plan, Failure>;
    /// The prompts of `drawing`'s pictures still to draw, encoded.
    fn encode(&self, drawing: &Drawing) -> Result<Encoded, String>;
    /// `drawing`'s pictures of `encoded` drawn while `still` wanted, each
    /// scene told to `tell`.
    fn draw(
        &self,
        drawing: &Drawing,
        encoded: &mut Encoded,
        still: &dyn Fn() -> bool,
        tell: &mut dyn FnMut(Story),
    ) -> Result<Drawn, String>;
}

/// The story's models on the GPU, where setup put them: each held for
/// its stage and registered there, set up on its first use in a turn.
pub struct GpuModels {
    language: Arc<jodel_gpu::Kept<Gemma>>,
    encoder: Arc<jodel_gpu::Kept<TextEncoder>>,
    drawer: Arc<jodel_gpu::Kept<Drawer>>,
}

impl GpuModels {
    /// The models in `files`, registered for their stages.
    pub fn new(files: &StoryModels) -> GpuModels {
        let (planned, language) = (files.language.clone(), files.language.clone());
        let language = Arc::new(jodel_gpu::Kept::new(
            move || super::memory(&locate(&planned)).ok(),
            Box::new(move || {
                let gemma = Gemma::open(&locate(&language), Precision::Fast)?;
                let name = gemma.gpu_name().to_owned();
                Ok((gemma, name))
            }),
        ));
        let encoder = files.encoder.clone();
        let encoder = Arc::new(jodel_gpu::Kept::new(
            || Some(TextEncoder::planned()),
            Box::new(move || {
                let model = TextEncoder::load(&locate(&encoder), Precision::Fast)?;
                let name = model.gpu_name().to_owned();
                Ok((model, name))
            }),
        ));
        let (transformer, pipeline) = (files.transformer.clone(), files.pipeline.clone());
        let drawer = Arc::new(jodel_gpu::Kept::new(
            || Some(super::draw::memory()),
            Box::new(move || {
                let (t, p) = (locate(&transformer), locate(&pipeline));
                let model = Drawer::load(&t, &p, Precision::Fast)?;
                let name = model.gpu_name().to_owned();
                Ok((model, name))
            }),
        ));
        stages::register(Stage::StoryPlan, Arc::clone(&language) as Arc<dyn Resident>);
        stages::register(
            Stage::StoryEncode,
            Arc::clone(&encoder) as Arc<dyn Resident>,
        );
        stages::register(Stage::StoryDraw, Arc::clone(&drawer) as Arc<dyn Resident>);
        GpuModels {
            language,
            encoder,
            drawer,
        }
    }
}

/// A model that cannot be set up fails the song's plan as a model's
/// failure does ([`Failure::Model`]: not kept), noted in the trace.
/// A model that cannot be set up fails the song's plan as a model's
/// failure does ([`Failure::Model`]: not kept); one set up again after
/// its device was lost runs its job again, the plan's words already told
/// not told twice ([`past`]). Each turn's end is noted in the trace.
impl Models for GpuModels {
    fn plan(&self, job: &StoryJob, tell: &mut dyn FnMut(Story)) -> Result<Plan, Failure> {
        let told = &mut 0;
        let planned = || {
            (self.language)
                .run(|gemma| Ok(job.run(gemma, &mut past(told, &mut *tell))))
                .unwrap_or_else(|why| Err(Failure::Model(why)))
        };
        turn(
            Stage::StoryPlan,
            job.path(),
            planned,
            |planned| match planned {
                Ok(_) => "planned".into(),
                Err(failure) => format!("not planned: {}", failure.why()),
            },
        )
    }

    fn encode(&self, drawing: &Drawing) -> Result<Encoded, String> {
        let encoded = || self.encoder.run(|encoder| drawing.encode(encoder));
        turn(
            Stage::StoryEncode,
            drawing.path(),
            encoded,
            |encoded| match encoded {
                Ok(_) => "its prompts encoded".into(),
                Err(why) => format!("its prompts not encoded: {why}"),
            },
        )
    }

    fn draw(
        &self,
        drawing: &Drawing,
        encoded: &mut Encoded,
        still: &dyn Fn() -> bool,
        tell: &mut dyn FnMut(Story),
    ) -> Result<Drawn, String> {
        let drawn = || (self.drawer).run(|drawer| drawing.draw(encoded, drawer, still, &mut *tell));
        turn(
            Stage::StoryDraw,
            drawing.path(),
            drawn,
            |drawn| match drawn {
                Ok(Drawn::All) => "its pictures drawn".into(),
                Ok(Drawn::GivenUp) => "its pictures given up: no longer wanted".into(),
                Err(why) => format!("its pictures not drawn: {why}"),
            },
        )
    }
}

/// `job`, for the song at `path`, in `stage`'s turn, noted in the trace as
/// `said` says what it made, with how long it took (its model's setup
/// included).
fn turn<T>(
    stage: Stage,
    path: &Path,
    job: impl FnOnce() -> T,
    said: impl FnOnce(&T) -> String,
) -> T {
    let name = song_name(path);
    stages::run(&[stage], &name, || {
        let started = Instant::now();
        let made = job();
        let said = said(&made);
        crate::trace::took(format_args!("story: {name}: {said}"), started, true);
        made
    })
}

/// `tell`, but for the story's words past the first `told` bytes: those a
/// run before told (the model writes greedily, so a run again writes the
/// same), `told` counting them.
fn past<'a>(told: &'a mut usize, tell: &'a mut dyn FnMut(Story)) -> impl FnMut(Story) + 'a {
    let mut seen = 0;
    move |story| match story {
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
        scene => tell(scene),
    }
}

/// Where the stories are kept: the plans (the data folder's `stories/`)
/// and the songs' scenes.
pub struct Places {
    /// The plans' folder.
    pub stories: PathBuf,
    /// The songs' scenes.
    pub scenes: SceneStore,
}

/// What the thread is asked: the songs wanted, and how many times they
/// were asked for.
#[derive(Default)]
struct Asked {
    songs: Vec<StorySong>,
    times: u64,
}

impl Asked {
    /// Whether `song` is still wanted, with the very same lyrics.
    fn still(&self, song: &StorySong) -> bool {
        self.songs.contains(song)
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
    /// `places`, told on `outbox`, and each drawn told to `on_drawn`.
    pub fn start(
        models: Arc<dyn Models>,
        places: Places,
        outbox: Outbox,
        on_drawn: OnDrawn,
    ) -> StoryWorker {
        let asked = Arc::new(Queue::new(Asked::default()));
        let thread = Arc::clone(&asked);
        std::thread::Builder::new()
            .name("stories".into())
            .spawn(move || {
                jodel_gpu::pace::background();
                // A round goes on where one asked anew, or the last left
                // songs for the next.
                let (mut seen, mut left) = (0, false);
                while let Some(songs) = thread.next(|a: &mut Asked| {
                    (left || a.times != seen).then(|| {
                        seen = a.times;
                        a.songs.clone()
                    })
                }) {
                    let still = |song: &StorySong| !thread.stopped() && thread.lock().still(song);
                    left = round(&songs, &*models, &places, &still, &outbox, &*on_drawn);
                }
            })
            .expect("a thread");
        StoryWorker { asked }
    }

    /// The stories of exactly `songs` wanted, in this order: the round
    /// under way gives up the songs no longer wanted, and the next takes
    /// these up.
    pub fn want(&self, songs: Vec<StorySong>) {
        self.asked.change(|a| {
            a.songs = songs;
            a.times += 1;
        });
    }
}

/// One round: the stories of `songs` (each given up once `still` says it
/// is no longer wanted) planned, then their prompts encoded (while the
/// prompts encoded hold less than [`ENCODED_AHEAD_MB`]), then their
/// pictures drawn and described, by `models` in their turns, kept in
/// `places`, the news told on `outbox`, each drawn told to `on_drawn`.
/// Whether it left songs to the next round.
pub fn round(
    songs: &[StorySong],
    models: &dyn Models,
    places: &Places,
    still: &dyn Fn(&StorySong) -> bool,
    outbox: &Outbox,
    on_drawn: &dyn Fn(&Path),
) -> bool {
    // The language model's turns.
    let mut drawings = Vec::new();
    for song in songs.iter().filter(|s| still(s)) {
        if let Some(drawing) = drawing_of(song, models, places, outbox) {
            drawings.push((song, drawing));
        }
    }
    // The text encoder's.
    let (mut encoded, mut held, mut left) = (Vec::new(), 0, false);
    for (song, drawing) in drawings.into_iter().filter(|(s, _)| still(s)) {
        if held >= ENCODED_AHEAD_MB << 20 {
            left = true;
            break;
        }
        let prompts = if drawing.to_draw() {
            // Why not, noted in the trace by its turn.
            let Ok(prompts) = models.encode(&drawing) else {
                continue;
            };
            Some(prompts)
        } else {
            None
        };
        held += prompts.as_ref().map_or(0, Encoded::bytes);
        encoded.push((song, drawing, prompts));
    }
    // The image model's.
    for (song, drawing, prompts) in encoded.into_iter().filter(|(s, _, _)| still(s)) {
        let tell = &mut teller(outbox, &song.path);
        let drawn = match prompts {
            Some(mut prompts) => models.draw(&drawing, &mut prompts, &|| still(song), tell),
            None => Ok(Drawn::All),
        };
        // Given up or failed: noted in the trace by its turn, what is drawn
        // kept.
        if drawn == Ok(Drawn::All) {
            match drawing.keep() {
                Ok(()) => on_drawn(&song.path),
                Err(why) => failed(&song.path, &why),
            }
        }
    }
    left
}

/// The drawing of `song`'s story, not yet drawn: its plan as kept, else
/// planned now by `models` (its news told on `outbox`), kept in `places`;
/// none where it is drawn already, or has no story.
fn drawing_of(
    song: &StorySong,
    models: &dyn Models,
    places: &Places,
    outbox: &Outbox,
) -> Option<Drawing> {
    let job = job_of(&song.path, &song.lyrics, places)?;
    let plan = match job.kept() {
        Some(Kept::Plan(plan)) => *plan,
        Some(Kept::NoStory(_)) => return None,
        // Why there is none told, and noted in the trace, by the job.
        None => models.plan(&job, &mut teller(outbox, &song.path)).ok()?,
    };
    let drawing = Drawing::new(song.path.clone(), plan, &places.scenes);
    (!drawing.drawn()).then_some(drawing)
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
/// none for a song that cannot be read or has no timed sung lines.
fn job_of(path: &Path, lyrics: &Lyrics, places: &Places) -> Option<StoryJob> {
    let info = jodel_playback::source::probe(path)
        .map_err(|e| failed(path, &e.to_string()))
        .ok()?;
    let song = Song::of(&info, lyrics);
    (!song.lines.is_empty()).then(|| StoryJob::new(path.to_owned(), song, places.stories.clone()))
}

/// The song at `path` left without a story now, for `why` (noted in the
/// trace).
fn failed(path: &Path, why: &str) {
    crate::trace::note(format_args!(
        "story: {}: no story now: {why}",
        song_name(path)
    ));
}

#[cfg(test)]
mod tests;
