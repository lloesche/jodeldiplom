//! A song's preparation as it goes, kept for the queue's captions and the
//! right column's preparation view ([`crate::view::prep`]): what each step
//! of its work found so far, as the workers tell it ([`Heard`]), never
//! more than they found. Only work done now is kept (a song whose results
//! were all in the caches has none), for the last [`KEPT`] songs worked
//! on.
//!
//! What the workers tell, in the order it comes:
//!
//! - the analysis: the key once the file is decoded ([`Heard::Key`]), the
//!   stems as each chunk is separated ([`Heard::Separating`]: each stem's
//!   waveform so far, binned over the song) and once all are
//!   ([`Heard::Stems`]), the beat once decided ([`Heard::Beats`]: a
//!   stretch of the song's waveform from a downbeat, with its beats);
//! - the lyrics ([`Heard::Lyrics`]): looked up on LRCLIB, found (and
//!   where), heard from the voice, their words being timed, timed (the
//!   first lines, with their words' times, for a preview);
//! - the song's story ([`Heard::Story`], from [`crate::work::story`]): its
//!   plan begun and written as a language model writes it, its pictures'
//!   prompts read, each picture begun and each scene as the image model
//!   draws it, then its scenes taken apart; or why it was not made now.
//!
//! Each result found is counted ([`Prep::found`]), so a sound can follow
//! it ([`crate::sfx`]). The step a job is at ([`step_of`]) is what every
//! screen says of it: the queue's captions, the preparation view, and
//! what a song waits for before it can be sung ([`crate::core::job_words`]).

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use jodel_rhythm::key::Key;
use jodel_rhythm::waveform::Lane;

use super::{readiness, Heard, Job, Readiness, State};

/// The songs whose preparation is kept: the most recent.
pub const KEPT: usize = 6;
/// The bins each stem's waveform is shown in, over the whole song.
pub const BINS: usize = 64;
/// The bins of the stretch of waveform the beat is shown on.
pub const BEAT_BINS: usize = 160;
/// How long the stretch the beat is shown on is, in seconds.
pub const STRETCH: f64 = 8.0;
/// How long a result found stays as a queue row's caption.
pub const CAPTION_MS: u64 = 2_500;
/// How long a result found takes to come in on the preparation view.
pub const REVEAL_MS: u64 = 1_200;
/// How long the view stays on a result found before it goes on to the
/// next step under way.
pub const HOLD_MS: u64 = 3_000;
/// How long a song's preparation shows after nothing was left to do for
/// it.
pub const SHOWN_MS: u64 = 12_000;
/// The lines of the timed words kept for the preview.
pub const PREVIEW_LINES: usize = 2;
/// The most of a story kept, in bytes (its oldest words go first).
pub const STORY_CHARS: usize = 4_000;
/// The width of a scene's picture as it is told ([`Story::Scene`]: the
/// whole picture made smaller, its shape kept), in pixels: what the
/// preparation view shows it at, a tile of the cover.
pub const SCENE_WIDTH: u32 = 256;

/// The stems separated so far ([`Heard::Separating`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Separating {
    /// The share of the song separated, 0 to 1.
    pub share: f32,
    /// Each stem's loudest ([`jodel_rhythm::stems::Stem::ALL`]' order:
    /// bass, drums, other, vocals) in each of [`BINS`] over the song, on
    /// the waveform's dB scale (0 to 255); 0 where not separated yet.
    pub peaks: [[u8; BINS]; 4],
}

impl Separating {
    /// The stems' `lanes` so far (100 frames a second) in a song of
    /// `frames` frames.
    pub fn of(lanes: [&Lane; 4], frames: usize) -> Separating {
        let frames = frames.max(1);
        let mut peaks = [[0u8; BINS]; 4];
        for (bins, lane) in peaks.iter_mut().zip(lanes) {
            for (k, &p) in lane.peak.iter().enumerate() {
                let bin = (k * BINS / frames).min(BINS - 1);
                bins[bin] = bins[bin].max(p);
            }
        }
        let done = lanes.iter().map(|l| l.len()).min().unwrap_or(0);
        Separating {
            share: (done as f32 / frames as f32).min(1.0),
            peaks,
        }
    }
}

/// The beat as the analysis decided it ([`Heard::Beats`]): its tempo, and
/// a stretch of the song from a downbeat with its waveform and beats.
#[derive(Clone, Debug, PartialEq)]
pub struct BeatsFound {
    /// The tempo, where the beat is clear.
    pub bpm: Option<f64>,
    /// Where the stretch starts, seconds into the song.
    pub from: f64,
    /// How long it is, seconds.
    pub seconds: f64,
    /// The mix's loudest in each of [`BEAT_BINS`] across it, on the
    /// waveform's dB scale.
    pub peaks: Vec<u8>,
    /// Its beats: where across it (0 to 1), and whether each starts a bar.
    pub beats: Vec<(f32, bool)>,
}

impl BeatsFound {
    /// The beat of `analysis` over the mix's waveform `mix` (100 frames a
    /// second): the stretch from the first downbeat a third of the way in
    /// (else the first beat there, else the third).
    pub fn of(analysis: &jodel_rhythm::Analysis, mix: &Lane) -> BeatsFound {
        let duration = analysis.duration.max(mix.len() as f64 / 100.0);
        let third = duration / 3.0;
        let downbeat = (analysis.downbeats.iter())
            .filter_map(|&k| analysis.beats.get(k).copied())
            .find(|&t| t >= third);
        let beat = analysis.beats.iter().copied().find(|&t| t >= third);
        let from = downbeat.or(beat).unwrap_or(third);
        let seconds = STRETCH.min((duration - from).max(0.0));
        let mut peaks = vec![0u8; BEAT_BINS];
        let first = (from * 100.0) as usize;
        let last = ((from + seconds) * 100.0) as usize;
        for (k, &p) in mix.peak.iter().enumerate().take(last).skip(first) {
            let bin = ((k - first) * BEAT_BINS / (last - first).max(1)).min(BEAT_BINS - 1);
            peaks[bin] = peaks[bin].max(p);
        }
        let beats = (analysis.beats.iter().enumerate())
            .filter(|&(_, &t)| t >= from && t < from + seconds)
            .map(|(k, &t)| {
                let across = ((t - from) / seconds.max(1e-9)) as f32;
                (across, analysis.downbeats.binary_search(&k).is_ok())
            })
            .collect();
        BeatsFound {
            bpm: analysis.summary.map(|s| s.bpm).filter(|b| *b > 0.0),
            from,
            seconds,
            peaks,
            beats,
        }
    }
}

/// Where a song's words were found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// LRCLIB.
    Lrclib,
    /// The file's own tags.
    Tags,
    /// A file beside it.
    Beside,
    /// Heard from the voice by the speech recogniser.
    Heard,
}

impl Origin {
    /// Where `source` is.
    pub fn of(source: &jodel_lyrics::Source) -> Origin {
        match source {
            jodel_lyrics::Source::Lrclib => Origin::Lrclib,
            jodel_lyrics::Source::Tag => Origin::Tags,
            jodel_lyrics::Source::Sidecar | jodel_lyrics::Source::Unknown => Origin::Beside,
            jodel_lyrics::Source::Transcribed(_) => Origin::Heard,
        }
    }
}

/// A word of the preview, its times from the preview's start.
#[derive(Clone, Debug, PartialEq)]
pub struct PreviewWord {
    /// As written.
    pub text: String,
    /// When it starts, seconds from the preview's start.
    pub start: f32,
    /// When it ends.
    pub end: f32,
}

/// The first lines of a song's timed words, for a preview of them lit in
/// time.
#[derive(Clone, Debug, PartialEq)]
pub struct Preview {
    /// The lines, each its words.
    pub lines: Vec<Vec<PreviewWord>>,
    /// How long they take, seconds.
    pub seconds: f32,
}

impl Preview {
    /// The first [`PREVIEW_LINES`] lines of `timed` with every word timed,
    /// of three words or more where there are such; `None` with none.
    pub fn of(timed: &jodel_lyrics::Lyrics) -> Option<Preview> {
        let whole = |l: &&jodel_lyrics::Line| {
            !l.is_break
                && !l.words.is_empty()
                && l.words.iter().all(|w| w.start.is_some() && w.end.is_some())
        };
        let long: Vec<&jodel_lyrics::Line> = (timed.lines.iter().filter(whole))
            .filter(|l| l.words.len() >= 3)
            .take(PREVIEW_LINES)
            .collect();
        let lines = if long.is_empty() {
            timed
                .lines
                .iter()
                .filter(whole)
                .take(PREVIEW_LINES)
                .collect()
        } else {
            long
        };
        let start = lines.first()?.words.first()?.start?;
        let lines: Vec<Vec<PreviewWord>> = lines
            .iter()
            .map(|l| {
                l.words
                    .iter()
                    .map(|w| PreviewWord {
                        text: w.text.clone(),
                        start: (w.start.unwrap_or(start) - start) as f32,
                        end: (w.end.unwrap_or(start) - start) as f32,
                    })
                    .collect()
            })
            .collect();
        let seconds = (lines.iter().flatten())
            .map(|w| w.end)
            .fold(0.0f32, f32::max);
        Some(Preview { lines, seconds })
    }
}

/// A step of the lyrics' work, as it begins or finds something
/// ([`Heard::Lyrics`]).
#[derive(Clone, Debug, PartialEq)]
pub enum Words {
    /// Looked up on LRCLIB.
    Looking,
    /// Found, and where.
    Found(Origin),
    /// None found anywhere: heard from the voice.
    Hearing,
    /// Their words being timed against the voice.
    Timing,
    /// Timed: the first lines, for the preview.
    Timed(Arc<Preview>),
}

/// A picture of a story begun by the image model ([`Story::Drawing`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Picture {
    /// A character's portrait: which (from 1), of how many.
    Portrait(usize, usize),
    /// A scene: which (from 1), of how many.
    Scene(usize, usize),
}

/// The song's story ([`Heard::Story`]) as its jobs tell it
/// (`crate::work::story`): its plan begun, the language model's thought
/// before it, then the plan written and told as it is, its pictures'
/// prompts read, each picture begun
/// and each scene told as it lands (each a result found:
/// [`Found::Scene`]), its scenes taken apart (kept as [`StoryPrep`]; how it
/// ends is a fact, [`crate::core::StoryFact`]). The queue's captions and
/// the preparation view show it ([`StoryStand`]).
#[derive(Clone)]
pub enum Story {
    /// Its plan begun by the language model.
    Planning,
    /// The model thinking before it writes the plan.
    Thinking,
    /// The next words of the story, as the model writes them.
    Words(String),
    /// Its pictures' prompts being read by the image model's text encoder.
    Reading,
    /// A picture begun.
    Drawing(Picture),
    /// A scene drawn: which, of how many, and its picture ([`SCENE_WIDTH`]
    /// across, its shape kept).
    Scene {
        /// Which scene, from 0.
        index: usize,
        /// How many scenes the story has.
        of: usize,
        /// Its picture.
        picture: Arc<image::RgbaImage>,
    },
    /// Its scenes being taken apart for the backdrop.
    Cutting,
}

impl fmt::Debug for Story {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Story::Planning => write!(f, "Planning"),
            Story::Thinking => write!(f, "Thinking"),
            Story::Words(words) => write!(f, "Words({words:?})"),
            Story::Reading => write!(f, "Reading"),
            Story::Drawing(picture) => write!(f, "Drawing({picture:?})"),
            Story::Scene { index, of, .. } => write!(f, "Scene({index} of {of})"),
            Story::Cutting => write!(f, "Cutting"),
        }
    }
}

impl PartialEq for Story {
    fn eq(&self, other: &Story) -> bool {
        match (self, other) {
            (
                Story::Scene {
                    index: a,
                    of: m,
                    picture: p,
                },
                Story::Scene {
                    index: b,
                    of: n,
                    picture: q,
                },
            ) => a == b && m == n && Arc::ptr_eq(p, q),
            (Story::Words(a), Story::Words(b)) => a == b,
            (Story::Drawing(a), Story::Drawing(b)) => a == b,
            (Story::Planning, Story::Planning)
            | (Story::Thinking, Story::Thinking)
            | (Story::Reading, Story::Reading)
            | (Story::Cutting, Story::Cutting) => true,
            _ => false,
        }
    }
}

/// A scene of the story as drawn.
#[derive(Clone, Debug)]
pub struct Scene {
    /// Which, from 0.
    pub index: usize,
    /// Its picture.
    pub picture: Arc<image::RgbaImage>,
    /// When it was told.
    pub at: u64,
}

/// What a song's story told of itself so far, its plan's: the step under
/// way last told (planning, thinking, reading, drawing, cutting; the plan's
/// words planning again after the thought), its words as
/// written, its scenes drawn, in order, and how many it has (once one is
/// drawn). A plan begun anew starts it afresh.
#[derive(Clone, Debug, Default)]
pub struct StoryPrep {
    /// The step under way last told.
    pub step: Option<Step>,
    /// The words so far (the last [`STORY_CHARS`] bytes).
    pub text: String,
    /// The scenes drawn so far.
    pub scenes: Vec<Scene>,
    /// How many scenes it has.
    pub of: usize,
}

impl StoryPrep {
    /// Whether anything of it is told: words or a scene.
    pub fn told(&self) -> bool {
        !self.text.is_empty() || !self.scenes.is_empty()
    }

    /// `story` told of at `now`; a scene drawn found (which, of how many).
    fn tell(&mut self, story: &Story, now: u64) -> Option<Found> {
        match story {
            Story::Planning => {
                *self = StoryPrep {
                    step: Some(Step::Planning),
                    ..StoryPrep::default()
                };
            }
            Story::Thinking => self.step = Some(Step::Thinking),
            Story::Reading => self.step = Some(Step::Reading),
            Story::Drawing(picture) => self.step = Some(Step::Drawing(*picture)),
            Story::Cutting => self.step = Some(Step::Cutting),
            Story::Words(words) => {
                // Words come only while it is planned: after the thought,
                // planning again.
                self.step = Some(Step::Planning);
                self.text.push_str(words);
                if self.text.len() > STORY_CHARS {
                    let cut = self.text.len() - STORY_CHARS;
                    let at = (cut..self.text.len())
                        .find(|&k| self.text.is_char_boundary(k))
                        .unwrap_or(self.text.len());
                    self.text.drain(..at);
                }
            }
            Story::Scene { index, of, picture } => {
                self.scenes.retain(|s| s.index != *index);
                self.scenes.push(Scene {
                    index: *index,
                    picture: Arc::clone(picture),
                    at: now,
                });
                self.scenes.sort_by_key(|s| s.index);
                self.of = *of;
                return Some(Found::Scene {
                    n: index + 1,
                    of: *of,
                });
            }
        }
        None
    }
}

/// Where a song's story stands, as every screen says it: the queue's
/// caption and the preparation view ([`crate::view::prep`]), and what a
/// song waits for before it can be sung ([`crate::core::job_words`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StoryStand {
    /// No stories are made here.
    Off,
    /// A worker at it, at this step.
    Making(Step),
    /// Still to come, no worker at it now: the scenes drawn so far, of
    /// how many (none told: 0 of 0).
    Waiting {
        /// Drawn.
        drawn: usize,
        /// Of how many.
        of: usize,
    },
    /// Not made now, and why: tried again later.
    NotNow(String),
    /// Made.
    Made,
    /// None, for good.
    NoStory,
}

impl StoryStand {
    /// Where `path`'s story stands in `state`.
    pub fn of(state: &State, path: &Path) -> StoryStand {
        use crate::core::StoryFact;
        if !state.preparing.contains(&Job::Story) {
            return StoryStand::Off;
        }
        if let Some(step) = doing(state, path, Job::Story) {
            return StoryStand::Making(step);
        }
        // Told with its job (made, none: done; not now: still to come),
        // and forgotten with it.
        match state.facts.get(path).and_then(|f| f.story.clone()) {
            Some(StoryFact::Made) => StoryStand::Made,
            Some(StoryFact::NoStory) => StoryStand::NoStory,
            Some(StoryFact::NotNow(why)) => StoryStand::NotNow(why),
            None => {
                let song = state.prep.song(path);
                StoryStand::Waiting {
                    drawn: song.map_or(0, |s| s.story.scenes.len()),
                    of: song.map_or(0, |s| s.story.of),
                }
            }
        }
    }

    /// Whether the story is still being made (a worker at it, or to come).
    pub fn making(&self) -> bool {
        matches!(self, StoryStand::Making(_) | StoryStand::Waiting { .. })
    }

    /// Where it is, in a few words (none: nothing to say but that it is to
    /// come, or no stories are made here).
    pub fn status(&self) -> Option<String> {
        match self {
            StoryStand::Off | StoryStand::Waiting { of: 0, .. } => None,
            StoryStand::Making(step) => Some(step.caption()),
            StoryStand::Waiting { drawn, of } => {
                Some(format!("{drawn} of {of} scenes drawn, waiting its turn"))
            }
            StoryStand::NotNow(why) => Some(format!("not made now ({why})")),
            StoryStand::Made => Some("made".into()),
            StoryStand::NoStory => Some("none for this song".into()),
        }
    }

    /// Its line, as the preparation view's step says it: its status as a
    /// sentence ("The story" with nothing to say).
    pub fn line(&self) -> String {
        let status = self.status().unwrap_or_else(|| "the story".into());
        let mut chars = status.chars();
        chars.next().map_or_else(String::new, |first| {
            first.to_uppercase().chain(chars).collect()
        })
    }
}

/// A result a step found, as the queue's caption says it.
#[derive(Clone, Debug, PartialEq)]
pub enum Found {
    /// The key.
    Key(Key),
    /// The stems separated.
    Stems,
    /// The tempo.
    Tempo(f64),
    /// The words, and where.
    Words(Origin),
    /// The words timed.
    Timed,
    /// A scene of the story drawn: which (from 1), of how many.
    Scene {
        /// Which.
        n: usize,
        /// Of how many.
        of: usize,
    },
}

impl Found {
    /// What the queue's caption says of it, after its tick.
    pub fn caption(&self) -> String {
        match self {
            Found::Key(key) => format!("{} ✓", key.name()),
            Found::Stems => "stems ✓".into(),
            Found::Tempo(bpm) => format!("{bpm:.0} BPM ✓"),
            Found::Words(Origin::Lrclib) => "words on LRCLIB ✓".into(),
            Found::Words(Origin::Heard) => "words heard ✓".into(),
            Found::Words(_) => "words found ✓".into(),
            Found::Timed => "words timed ✓".into(),
            Found::Scene { n, of } => format!("scene {n} of {of} ✓"),
        }
    }

    /// Its place among the steps (the beat 0 … the story's scenes
    /// [`LAST_STEP`]): the found sound's pitch climbs with it.
    pub fn step(&self) -> u32 {
        match self {
            Found::Tempo(_) => 0,
            Found::Key(_) => 1,
            Found::Stems => 2,
            Found::Words(_) | Found::Timed => 3,
            Found::Scene { .. } => LAST_STEP,
        }
    }
}

/// The last step a result is found at ([`Found::step`]): the story's
/// scenes.
pub const LAST_STEP: u32 = 4;

/// What a song's work found so far, told as it came.
#[derive(Clone, Debug, Default)]
pub struct SongPrep {
    /// The key, once looked for: what was found.
    pub key: Option<Option<Key>>,
    /// The stems separated so far, while they are.
    pub separating: Option<Arc<Separating>>,
    /// Whether every stem is separated.
    pub stems: bool,
    /// The beat, once decided.
    pub beats: Option<Arc<BeatsFound>>,
    /// The lyrics' step under way last told (looking, hearing, timing).
    pub words_step: Option<Words>,
    /// Where the words were found.
    pub origin: Option<Origin>,
    /// The words timed, for the preview.
    pub preview: Option<Arc<Preview>>,
    /// The story, as it told of itself.
    pub story: StoryPrep,
    /// The last result found, and when ([`State::now_ms`]).
    pub found: Option<(Found, u64)>,
    /// When the beat came.
    pub beats_at: Option<u64>,
    /// When the key was looked for.
    pub key_at: Option<u64>,
    /// When the stems were all separated.
    pub stems_at: Option<u64>,
    /// When the words were found, then timed.
    pub words_at: Option<u64>,
    /// When it was last told of.
    pub last: u64,
    /// When nothing was left to do for it, once nothing is.
    pub done_at: Option<u64>,
}

/// The preparation of the last songs worked on.
#[derive(Clone, Debug, Default)]
pub struct Prep {
    /// The songs, the most recently told of last.
    songs: Vec<(PathBuf, SongPrep)>,
    /// The results found this run.
    pub found: u64,
    /// When the last was ([`State::now_ms`]).
    pub found_at: Option<u64>,
}

impl Prep {
    /// What `path`'s work found, if it is kept.
    pub fn song(&self, path: &Path) -> Option<&SongPrep> {
        self.songs.iter().find(|(p, _)| p == path).map(|(_, s)| s)
    }

    /// The songs kept.
    pub fn songs(&self) -> impl Iterator<Item = (&PathBuf, &SongPrep)> {
        self.songs.iter().map(|(p, s)| (p, s))
    }

    /// `path`'s, kept from now as the most recent (made where it was not;
    /// the oldest let go of past [`KEPT`]).
    fn entry(&mut self, path: &Path, now: u64) -> &mut SongPrep {
        let song = match self.songs.iter().position(|(p, _)| p == path) {
            Some(k) => self.songs.remove(k).1,
            None => SongPrep::default(),
        };
        self.songs.push((path.to_owned(), song));
        if self.songs.len() > KEPT {
            self.songs.remove(0);
        }
        let (_, song) = self.songs.last_mut().expect("just pushed");
        song.last = now;
        song
    }
}

/// `found` counted for `song` at `now`.
fn found(prep: &mut Prep, path: &Path, now: u64, found: Found) {
    prep.entry(path, now).found = Some((found, now));
    prep.found += 1;
    prep.found_at = Some(now);
}

/// [`crate::core::Action::Working`]: a worker started (`on`) on `path`:
/// its preparation is kept from now. Its end is followed by [`follow`].
pub(crate) fn working(state: &mut State, path: &Path, on: bool) {
    if on {
        let now = state.now_ms;
        let song = state.prep.entry(path, now);
        song.done_at = None;
    }
}

/// [`crate::core::Action::Heard`]: what a step of `path`'s work found or
/// began.
pub(crate) fn heard(state: &mut State, path: &Path, part: &Heard) {
    let now = state.now_ms;
    let prep = &mut state.prep;
    match part {
        Heard::Key(key) => {
            let song = prep.entry(path, now);
            song.key = Some(*key);
            song.key_at = Some(now);
            if let Some(key) = key {
                found(prep, path, now, Found::Key(*key));
            }
        }
        Heard::Separating(so_far) => {
            prep.entry(path, now).separating = Some(Arc::clone(so_far));
        }
        Heard::Stems => {
            let song = prep.entry(path, now);
            song.stems = true;
            song.stems_at = Some(now);
            found(prep, path, now, Found::Stems);
        }
        Heard::Beats(beats) => {
            let song = prep.entry(path, now);
            song.beats = Some(Arc::clone(beats));
            song.beats_at = Some(now);
            if let Some(bpm) = beats.bpm {
                found(prep, path, now, Found::Tempo(bpm));
            }
        }
        Heard::Lyrics(words) => {
            let song = prep.entry(path, now);
            match words {
                Words::Looking | Words::Hearing | Words::Timing => {
                    song.words_step = Some(words.clone());
                }
                Words::Found(origin) => {
                    song.origin = Some(*origin);
                    song.words_at = Some(now);
                    found(prep, path, now, Found::Words(*origin));
                }
                Words::Timed(preview) => {
                    song.preview = Some(Arc::clone(preview));
                    song.words_at = Some(now);
                    found(prep, path, now, Found::Timed);
                }
            }
        }
        Heard::Story(story) => {
            if let Some(scene) = prep.entry(path, now).story.tell(story, now) {
                found(prep, path, now, scene);
            }
        }
    }
}

/// After a worker's news of `path` (started, ended, a part found, a job
/// done): when nothing is left to do for a song kept, when that was.
pub(crate) fn follow(state: &mut State, path: &Path) {
    let done = readiness(state, path) == Readiness::Done;
    let now = state.now_ms;
    if let Some((_, song)) = state.prep.songs.iter_mut().find(|(p, _)| p == path) {
        match (done, song.done_at) {
            (true, None) => song.done_at = Some(now),
            (false, Some(_)) => song.done_at = None,
            _ => {}
        }
    }
}

/// The step `job` is at for `song` while a worker is at it: the analysis
/// listening for the key, then taking the song apart (while its stems are
/// separated), then finding the beat; the lyrics' and the story's step
/// last told.
pub fn step_of(job: Job, song: Option<&SongPrep>) -> Step {
    match job {
        Job::Rhythm => match song {
            Some(s) if s.key.is_some() && s.separating.is_some() && !s.stems => Step::Stems,
            Some(s) if s.key.is_some() => Step::Beat,
            _ => Step::Key,
        },
        Job::Lyrics => match song.and_then(|s| s.words_step.as_ref()) {
            Some(Words::Hearing) => Step::Hearing,
            Some(Words::Timing) => Step::Timing,
            _ => Step::Looking,
        },
        Job::Story => song.and_then(|s| s.story.step).unwrap_or(Step::Planning),
    }
}

/// The step `job` is at for `path` where a worker is at it now ([`step_of`]).
pub fn doing(state: &State, path: &Path, job: Job) -> Option<Step> {
    let working = state.working.get(path)?.contains(&job);
    working.then(|| step_of(job, state.prep.song(path)))
}

/// A step of a song's preparation, as it runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// Listening for the key (the file decoded).
    Key,
    /// Taking the song apart (its stems separated).
    Stems,
    /// Finding the beat.
    Beat,
    /// Looking up the words.
    Looking,
    /// Listening for the words (heard from the voice).
    Hearing,
    /// Timing the words.
    Timing,
    /// Planning the story (the language model writing it).
    Planning,
    /// The language model thinking before it writes the story's plan.
    Thinking,
    /// Reading the story's pictures' prompts.
    Reading,
    /// Drawing a picture of the story.
    Drawing(Picture),
    /// Taking the story's scenes apart.
    Cutting,
}

impl Step {
    /// What it is doing, as the queue's caption says it.
    pub fn caption(self) -> String {
        match self {
            Step::Key => "listening for the key…".into(),
            Step::Stems => "taking the song apart…".into(),
            Step::Beat => "finding the beat…".into(),
            Step::Looking => "looking up the words…".into(),
            Step::Hearing => "listening for the words…".into(),
            Step::Timing => "timing the words…".into(),
            Step::Planning => "planning the story…".into(),
            Step::Thinking => "thinking…".into(),
            Step::Reading => "reading the scenes' prompts…".into(),
            Step::Drawing(Picture::Portrait(n, of)) => format!("drawing character {n} of {of}…"),
            Step::Drawing(Picture::Scene(n, of)) => format!("drawing scene {n} of {of}…"),
            Step::Cutting => "giving the scenes depth…".into(),
        }
    }
}

/// When the preparation shown next changes by itself: a result's reveal
/// ending, its hold over, a finished song's view going; after the clock
/// ([`State::now_ms`]), so each is ticked once.
pub fn due(state: &State) -> Option<u64> {
    let now = state.now_ms;
    let found = state
        .prep
        .found_at
        .into_iter()
        .flat_map(|at| [at + REVEAL_MS, at + HOLD_MS, at + CAPTION_MS]);
    let done = (state.prep.songs.iter()).filter_map(|(_, s)| s.done_at.map(|at| at + SHOWN_MS));
    found.chain(done).filter(|&t| t > now).min()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scene drawn, tiny.
    fn scene(index: usize) -> Heard {
        Heard::Story(Story::Scene {
            index,
            of: 3,
            picture: Arc::new(image::RgbaImage::new(2, 2)),
        })
    }

    /// The preparation is kept for the last songs told of only; a story
    /// keeps its last words.
    #[test]
    fn the_preparation_is_kept_for_the_last_songs_only() {
        let mut state = State::new("/music".into(), crate::config::Config::default());
        let song = |k: usize| PathBuf::from(format!("/music/{k}.flac"));
        for k in 0..KEPT + 3 {
            heard(&mut state, &song(k), &scene(0));
        }
        let kept: Vec<&PathBuf> = state.prep.songs().map(|(p, _)| p).collect();
        assert_eq!(kept.len(), KEPT);
        assert_eq!(kept[0], &song(3), "the oldest let go of");
        assert_eq!(state.prep.found, (KEPT + 3) as u64);
        let long = "a".repeat(STORY_CHARS);
        for words in [long.as_str(), "the end"] {
            heard(
                &mut state,
                &song(0),
                &Heard::Story(Story::Words(words.into())),
            );
        }
        let story = &state.prep.song(&song(0)).unwrap().story.text;
        assert_eq!(story.len(), STORY_CHARS);
        assert!(story.ends_with("the end"));
    }

    /// Where a story stands, in its words: the step at work as it is said,
    /// a sentence on the preparation view; a story to come with nothing
    /// to say is the story; a failure says why; off, nothing.
    #[test]
    fn a_story_s_stand_says_itself_once() {
        let drawing = StoryStand::Making(Step::Drawing(Picture::Scene(4, 23)));
        assert_eq!(drawing.status().as_deref(), Some("drawing scene 4 of 23…"));
        assert_eq!(drawing.line(), "Drawing scene 4 of 23…");
        let none = StoryStand::Waiting { drawn: 0, of: 0 };
        assert_eq!((none.status(), none.line()), (None, "The story".into()));
        let half = StoryStand::Waiting { drawn: 4, of: 23 };
        assert_eq!(half.line(), "4 of 23 scenes drawn, waiting its turn");
        let failed = StoryStand::NotNow("no GPU".into());
        assert_eq!(failed.line(), "Not made now (no GPU)");
        assert!(!failed.making() && half.making() && drawing.making());
        assert_eq!(StoryStand::Off.status(), None);
        let mut state = State::new("/music".into(), crate::config::Config::default());
        let path = PathBuf::from("/music/a.flac");
        assert_eq!(StoryStand::of(&state, &path), StoryStand::Off);
        state.preparing.push(Job::Story);
        assert_eq!(StoryStand::of(&state, &path), none);
    }

    /// The model's thought is a step of its own until the plan's words
    /// come; a plan begun anew starts the story afresh: its words, its
    /// scenes and their count from before all go.
    #[test]
    fn a_story_planned_again_starts_afresh() {
        let mut state = State::new("/music".into(), crate::config::Config::default());
        let path = PathBuf::from("/music/a.flac");
        let step = |state: &State| state.prep.song(&path).unwrap().story.step;
        heard(&mut state, &path, &Heard::Story(Story::Planning));
        heard(&mut state, &path, &Heard::Story(Story::Thinking));
        assert_eq!(
            step(&state).map(Step::caption).as_deref(),
            Some("thinking…")
        );
        heard(
            &mut state,
            &path,
            &Heard::Story(Story::Words("Once".into())),
        );
        assert_eq!(
            step(&state),
            Some(Step::Planning),
            "writing after the thought"
        );
        let drawing = Story::Drawing(Picture::Scene(1, 3));
        heard(&mut state, &path, &Heard::Story(drawing));
        heard(&mut state, &path, &scene(0));
        let story = &state.prep.song(&path).unwrap().story;
        assert!(story.told() && story.of == 3);
        assert_eq!(story.step, Some(Step::Drawing(Picture::Scene(1, 3))));
        heard(&mut state, &path, &Heard::Story(Story::Planning));
        let story = &state.prep.song(&path).unwrap().story;
        assert!(!story.told() && story.of == 0);
        assert_eq!(story.step, Some(Step::Planning));
    }
}
