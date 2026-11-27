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
//! - the cover once taken apart ([`Heard::Cover`]: its layers, as they
//!   were made);
//! - and, for what is to come, a story of the song written as a language
//!   model writes it ([`crate::work::story`]) and its scenes as an image
//!   model draws them ([`Heard::Story`]): nothing draws them yet.
//!
//! Each result found is counted ([`Prep::found`]), so a sound can follow
//! it ([`crate::sfx`]).

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use jodel_rhythm::key::Key;
use jodel_rhythm::waveform::Lane;

use super::{readiness, Heard, Job, Readiness, State};

/// The songs whose preparation is kept: the most recent.
pub const KEPT: usize = 6;
/// The songs whose cover's layers are kept with it: the most recent (each
/// a few megabytes).
pub const COVERS_KEPT: usize = 2;
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
/// The most of a story's scenes kept.
pub const SCENES: usize = 4;

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

/// A cover taken apart ([`Heard::Cover`]), its layers as they were made.
#[derive(Clone)]
pub struct CoverCut(pub crate::work::cover::Layers);

impl fmt::Debug for CoverCut {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CoverCut({})", self.0.key)
    }
}

impl PartialEq for CoverCut {
    fn eq(&self, other: &CoverCut) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// The seam for a story of the song ([`Heard::Story`]): written by a
/// language model and told as it writes it, its scenes drawn by an image
/// model and told step by step. The plan's job tells its words
/// ([`crate::work::story`]); nothing draws the scenes yet. A worker that
/// writes or draws them sends them on its outbox, and the preparation
/// view shows them ([`crate::view::prep::StoryView`]).
#[derive(Clone)]
pub enum Story {
    /// The next words of the story, as the model writes them.
    Words(String),
    /// A scene drawn so far: which, how far it is (0 to 1), and the
    /// picture as it is now.
    Scene {
        /// Which scene, from 0.
        index: usize,
        /// How far it is drawn, 0 to 1.
        done: f32,
        /// The picture so far.
        picture: Arc<image::RgbaImage>,
    },
}

impl fmt::Debug for Story {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Story::Words(words) => write!(f, "Words({words:?})"),
            Story::Scene { index, done, .. } => write!(f, "Scene({index}, {done})"),
        }
    }
}

impl PartialEq for Story {
    fn eq(&self, other: &Story) -> bool {
        match (self, other) {
            (Story::Words(a), Story::Words(b)) => a == b,
            (
                Story::Scene {
                    index: a,
                    done: d,
                    picture: p,
                },
                Story::Scene {
                    index: b,
                    done: e,
                    picture: q,
                },
            ) => a == b && d == e && Arc::ptr_eq(p, q),
            _ => false,
        }
    }
}

/// A scene of the story as drawn so far.
#[derive(Clone, Debug)]
pub struct Scene {
    /// Which, from 0.
    pub index: usize,
    /// How far it is drawn, 0 to 1.
    pub done: f32,
    /// The picture so far.
    pub picture: Arc<image::RgbaImage>,
    /// When it was told last.
    pub at: u64,
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
    /// The cover taken apart.
    Cover,
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
            Found::Cover => "cover cut ✓".into(),
        }
    }

    /// Its place among the steps (the beat 0 … the cover 4): the found
    /// sound's pitch climbs with it.
    pub fn step(&self) -> u32 {
        match self {
            Found::Tempo(_) => 0,
            Found::Key(_) => 1,
            Found::Stems => 2,
            Found::Words(_) | Found::Timed => 3,
            Found::Cover => 4,
        }
    }
}

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
    /// The cover taken apart.
    pub cover: Option<CoverCut>,
    /// The story so far.
    pub story: String,
    /// Its scenes so far, in order.
    pub scenes: Vec<Scene>,
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
    /// When the cover was taken apart.
    pub cover_at: Option<u64>,
    /// When the story was last told of.
    pub story_at: Option<u64>,
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

    /// The covers' layers let go of but for the [`COVERS_KEPT`] most
    /// recent.
    fn keep_covers(&mut self) {
        let mut kept = 0;
        for (_, song) in self.songs.iter_mut().rev() {
            if song.cover.is_some() {
                kept += 1;
                if kept > COVERS_KEPT {
                    song.cover = None;
                }
            }
        }
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
        Heard::Cover(cut) => {
            let song = prep.entry(path, now);
            song.cover = Some(cut.clone());
            song.cover_at = Some(now);
            found(prep, path, now, Found::Cover);
            prep.keep_covers();
        }
        Heard::Story(story) => {
            let song = prep.entry(path, now);
            song.story_at = Some(now);
            match story {
                Story::Words(words) => {
                    song.story.push_str(words);
                    if song.story.len() > STORY_CHARS {
                        let cut = song.story.len() - STORY_CHARS;
                        let at = (cut..song.story.len())
                            .find(|&k| song.story.is_char_boundary(k))
                            .unwrap_or(song.story.len());
                        song.story.drain(..at);
                    }
                }
                Story::Scene {
                    index,
                    done,
                    picture,
                } => {
                    song.scenes.retain(|s| s.index != *index);
                    song.scenes.push(Scene {
                        index: *index,
                        done: done.clamp(0.0, 1.0),
                        picture: Arc::clone(picture),
                        at: now,
                    });
                    song.scenes.sort_by_key(|s| s.index);
                    song.scenes.truncate(SCENES);
                }
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
/// separated), then finding the beat; the lyrics' step last told; the
/// cover's.
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
        Job::Cover => Step::Cover,
    }
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
    /// Cutting out the cover.
    Cover,
}

impl Step {
    /// What it is doing, as the queue's caption says it.
    pub fn caption(self) -> &'static str {
        match self {
            Step::Key => "listening for the key…",
            Step::Stems => "taking the song apart…",
            Step::Beat => "finding the beat…",
            Step::Looking => "looking up the words…",
            Step::Hearing => "listening for the words…",
            Step::Timing => "timing the words…",
            Step::Cover => "cutting out the cover…",
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
    use crate::work::cover::{Apart, CoverLayers};
    use crate::work::picture::NearMap;

    /// A cover taken apart, tiny.
    fn cut() -> CoverCut {
        let near = NearMap {
            width: 1,
            height: 1,
            near: vec![0],
        };
        CoverCut(Arc::new(Apart::new(
            "tiny".into(),
            CoverLayers {
                plate: image::RgbImage::new(2, 2),
                subject: None,
                depth: near.clone(),
                plate_depth: near,
                outline: image::GrayImage::new(2, 2),
                palette: Vec::new(),
            },
        )))
    }

    /// The preparation is kept for the last songs told of only, and the
    /// covers' layers for fewer of them; a story keeps its last words.
    #[test]
    fn the_preparation_is_kept_for_the_last_songs_only() {
        let mut state = State::new("/music".into(), crate::config::Config::default());
        let song = |k: usize| PathBuf::from(format!("/music/{k}.flac"));
        for k in 0..KEPT + 3 {
            heard(&mut state, &song(k), &Heard::Cover(cut()));
        }
        let kept: Vec<&PathBuf> = state.prep.songs().map(|(p, _)| p).collect();
        assert_eq!(kept.len(), KEPT);
        assert_eq!(kept[0], &song(3), "the oldest let go of");
        let covers = (state.prep.songs())
            .filter(|(_, s)| s.cover.is_some())
            .count();
        assert_eq!(covers, COVERS_KEPT);
        assert!(state.prep.song(&song(KEPT + 2)).unwrap().cover.is_some());
        assert_eq!(state.prep.found, (KEPT + 3) as u64);
        let long = "a".repeat(STORY_CHARS);
        for words in [long.as_str(), "the end"] {
            heard(
                &mut state,
                &song(0),
                &Heard::Story(Story::Words(words.into())),
            );
        }
        let story = &state.prep.song(&song(0)).unwrap().story;
        assert_eq!(story.len(), STORY_CHARS);
        assert!(story.ends_with("the end"));
    }
}
