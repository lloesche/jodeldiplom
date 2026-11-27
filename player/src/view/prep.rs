//! A song's preparation as the screens show it ([`crate::core::prep`]):
//! a queue row's caption ([`caption`]: the step running, then for a moment
//! what it found), and the right column's preparation view while the
//! queue's selected song is being got ready or was just
//! ([`PrepView`]): its steps one under the other, each what it is doing or
//! what it found, and the latest result shown big ([`Hero`]): the beat on
//! a stretch of the song's waveform, its ticks coming in and a pulse on
//! each; the key on a keyboard; the stems' lanes filling as they are
//! separated; the words lit in time as they will be sung. Only what the
//! work found is shown, never a progress it did not tell.
//!
//! The song's story is shown on its cover while it is made
//! ([`Hero::Story`], [`StoryView`]): typed out over it as a language model
//! writes it, then its scenes taking the cover's place tile by tile as an
//! image model draws them, each told as "Scene N of M drawn". Otherwise,
//! and once its story is made, the card is the song's cover itself
//! ([`Hero::Cover`]).

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::core::prep::{
    self, BeatsFound, Found, Origin, SongPrep, Step, StoryStand, CAPTION_MS, HOLD_MS, REVEAL_MS,
    SHOWN_MS,
};
use crate::core::{job_done, list_cursor, readiness, Job, LyricsFacts, Readiness, Side, State};

/// How long the beat pulse stays lit after a beat, in seconds.
pub const PULSE_SECONDS: f32 = 0.12;
/// The rest between two plays of the words' preview, seconds.
const PREVIEW_REST: f32 = 1.5;

/// A queue row's caption.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Caption {
    /// What it says.
    pub text: String,
    /// A result found (else the step running).
    pub found: bool,
}

/// The caption of `path`'s queue row: the last result found, for
/// [`CAPTION_MS`]; else the step running (of the work started last);
/// none while nothing is.
pub fn caption(state: &State, path: &Path) -> Option<Caption> {
    let song = state.prep.song(path);
    if let Some((found, at)) = song.and_then(|s| s.found.as_ref()) {
        if state.now_ms.saturating_sub(*at) < CAPTION_MS {
            return Some(Caption {
                text: found.caption(),
                found: true,
            });
        }
    }
    let job = *state.working.get(path)?.last()?;
    Some(Caption {
        text: prep::step_of(job, song).caption(),
        found: false,
    })
}

/// A step of the preparation, as the view lists it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The beat and tempo.
    Beat,
    /// The key.
    Key,
    /// The stems.
    Stems,
    /// The words.
    Words,
    /// The story.
    Story,
}

/// How a step stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stand {
    /// Still to come.
    Waiting,
    /// Under way.
    Running,
    /// Done, and found.
    Found,
    /// Done, and none to find.
    Missing,
}

/// A step as the view lists it.
#[derive(Clone, Debug, PartialEq)]
pub struct Stage {
    /// Which.
    pub kind: Kind,
    /// How it stands.
    pub stand: Stand,
    /// What it says: what it does, or what it found.
    pub line: String,
    /// How far in the result came, 0 to 1 (1 once it is in, or none came).
    pub reveal: f32,
}

/// The beat shown: a stretch of the waveform, its beats' ticks coming in,
/// a play head going over them with a pulse on each.
#[derive(Clone, Debug, PartialEq)]
pub struct BeatView {
    /// The waveform across the stretch, 0 to 1.
    pub peaks: Vec<f32>,
    /// The ticks in so far: where across (0 to 1), and whether a bar's
    /// first.
    pub ticks: Vec<(f32, bool)>,
    /// Where the play head is across, while it goes.
    pub head: Option<f32>,
    /// How lit the pulse is (a beat just passed), 0 to 1, and whether on
    /// a bar's first.
    pub pulse: (f32, bool),
}

/// The key shown on a keyboard of two octaves from a C.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyView {
    /// Its name.
    pub name: String,
    /// Its chord's notes lit, semitones over the lowest C.
    pub lit: Vec<u8>,
}

/// The stems shown as they are separated.
#[derive(Clone, Debug, PartialEq)]
pub struct StemsView {
    /// The share separated, 0 to 1.
    pub share: f32,
    /// Each stem's name and its waveform so far over the song (0 to 1),
    /// the voice first.
    pub lanes: Vec<(&'static str, Vec<f32>)>,
}

/// The words shown lit in time.
#[derive(Clone, Debug, PartialEq)]
pub struct WordsView {
    /// The lines, each its words and how much each is lit (0 to 1).
    pub lines: Vec<Vec<(String, f32)>>,
}

/// What the view shows big.
#[derive(Clone, Debug, PartialEq)]
pub enum Hero {
    /// The song's cover (a placeholder without one).
    Cover,
    /// The beat.
    Beat(BeatView),
    /// The key.
    Key(KeyView),
    /// The stems.
    Stems(StemsView),
    /// The words.
    Words(WordsView),
    /// The story on the cover, as written and drawn so far.
    Story(StoryView),
}

/// The story as written so far, and its scenes as drawn so far, on the
/// song's cover ([`crate::core::prep::Story`]).
#[derive(Clone, Debug)]
pub struct StoryView {
    /// The words so far.
    pub text: String,
    /// Whether the caret shows after them (while they are written).
    pub caret: bool,
    /// Each scene drawn: which (from 0), its picture and how far it has
    /// faded in, 0 to 1.
    pub scenes: Vec<(usize, Arc<image::RgbaImage>, f32)>,
    /// How many scenes the story has (0: none drawn yet).
    pub of: usize,
}

impl PartialEq for StoryView {
    fn eq(&self, other: &StoryView) -> bool {
        let same = |(i, a, x): &(usize, Arc<image::RgbaImage>, f32),
                    (k, b, y): &(usize, Arc<image::RgbaImage>, f32)| {
            i == k && Arc::ptr_eq(a, b) && x == y
        };
        self.text == other.text
            && self.caret == other.caret
            && self.of == other.of
            && self.scenes.len() == other.scenes.len()
            && self
                .scenes
                .iter()
                .zip(&other.scenes)
                .all(|(a, b)| same(a, b))
    }
}

/// The right column's preparation view.
#[derive(Clone, Debug)]
pub struct PrepView {
    /// The song.
    pub path: PathBuf,
    /// Its title.
    pub title: String,
    /// Whether nothing is left to do for it.
    pub done: bool,
    /// What it says big: the step under way, or the result found.
    pub headline: String,
    /// What is shown big.
    pub hero: Hero,
    /// The steps, in their order.
    pub stages: Vec<Stage>,
}

/// The queue's selected song while its preparation shows: on the main
/// screen's queue, kept, and still to be got ready, or ready since less
/// than [`SHOWN_MS`].
fn shown_song(state: &State) -> Option<(&PathBuf, &SongPrep)> {
    if !state.showing(Side::Queue) {
        return None;
    }
    let (cursor, _) = list_cursor(state, Side::Queue);
    let path = state.queue.paths.get(cursor)?;
    let song = state.prep.song(path)?;
    let now = state.now_ms;
    let shows = match song.done_at {
        Some(at) => now.saturating_sub(at) < SHOWN_MS,
        None => readiness(state, path) != Readiness::Done,
    };
    shows.then_some((path, song))
}

/// The song whose preparation shows: the app reads its cover for the card.
pub fn shown_path(state: &State) -> Option<&Path> {
    shown_song(state).map(|(path, _)| path.as_path())
}

/// The story's scenes of the song whose preparation shows, in order:
/// the window makes their pictures ready.
pub fn shown_scenes(state: &State) -> Vec<Arc<image::RgbaImage>> {
    shown_song(state).map_or_else(Vec::new, |(_, s)| {
        (s.story.scenes.iter())
            .map(|s| Arc::clone(&s.picture))
            .collect()
    })
}

/// Whether the preparation view shows and moves by itself: a worker at its
/// song, a result coming in or held, or its song ready since less than
/// [`SHOWN_MS`]. A song waiting for its work, with nothing new, rests.
pub fn moving(state: &State) -> bool {
    let Some((path, song)) = shown_song(state) else {
        return false;
    };
    let now = state.now_ms;
    state.working.contains_key(path)
        || song
            .found
            .as_ref()
            .is_some_and(|(_, at)| now.saturating_sub(*at) < HOLD_MS)
        || song.done_at.is_some()
}

/// `at`'s share of the way to `at` + `span` by `now`, eased out.
fn reveal(now: u64, at: Option<u64>, span: u64) -> f32 {
    let Some(at) = at else {
        return 1.0;
    };
    let t = (now.saturating_sub(at) as f32 / span as f32).clamp(0.0, 1.0);
    1.0 - (1.0 - t) * (1.0 - t)
}

/// A byte of the waveform's dB scale as a height, 0 to 1.
fn height(byte: u8) -> f32 {
    f32::from(byte) / 255.0
}

impl PrepView {
    /// The view for `state`, while it shows (on the queue's side, its
    /// selected song being got ready or just).
    pub fn of(state: &State) -> Option<PrepView> {
        let (path, song) = shown_song(state)?;
        // What moves, by the clock the frame is drawn at; at rest it
        // stands still.
        let now = state.drawn_ms.unwrap_or(state.now_ms).max(state.now_ms);
        let moves = moving(state);
        let stages = stages(state, path, song, now);
        let title = state
            .details
            .get(path)
            .and_then(|d| d.title.clone())
            .unwrap_or_else(|| {
                path.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
            });
        let (hero, headline) = hero(state, path, song, now, moves);
        let done = song.done_at.is_some();
        let headline = headline.unwrap_or_else(|| {
            let running = stages.iter().find(|s| s.stand == Stand::Running);
            match running {
                Some(s) => s.line.clone(),
                None if done => "Ready".to_owned(),
                None => "Waiting its turn".to_owned(),
            }
        });
        Some(PrepView {
            path: path.clone(),
            title,
            done,
            headline,
            hero,
            stages,
        })
    }
}

/// The steps for `path`, in order: the beat, the key, the stems (where
/// they are separated), the words, the story (where stories are made).
fn stages(state: &State, path: &Path, song: &SongPrep, now: u64) -> Vec<Stage> {
    let working = |job| state.working.get(path).is_some_and(|j| j.contains(&job));
    let done = |job| job_done(state, path, job);
    let facts = state.facts.get(path).cloned().unwrap_or_default();
    let analysing = working(Job::Rhythm);
    let stage = |kind, stand, line: &str, at| Stage {
        kind,
        stand,
        line: line.to_owned(),
        reveal: reveal(now, at, REVEAL_MS),
    };
    let mut out = Vec::new();
    // The beat.
    out.push(match &song.beats {
        Some(b) => match b.bpm {
            Some(bpm) => stage(
                Kind::Beat,
                Stand::Found,
                &format!("Found it: {bpm:.0} BPM"),
                song.beats_at,
            ),
            None => stage(Kind::Beat, Stand::Missing, "No clear beat", song.beats_at),
        },
        None if analysing && song.key.is_some() => {
            stage(Kind::Beat, Stand::Running, "Finding the downbeat…", None)
        }
        None if done(Job::Rhythm) && facts.tempo == Some(true) => {
            stage(Kind::Beat, Stand::Found, "Found it", None)
        }
        None if done(Job::Rhythm) && facts.tempo == Some(false) => {
            stage(Kind::Beat, Stand::Missing, "No clear beat", None)
        }
        None => stage(Kind::Beat, Stand::Waiting, "The beat", None),
    });
    // The key.
    out.push(match song.key {
        Some(Some(key)) => stage(
            Kind::Key,
            Stand::Found,
            &format!("Found it: {}", key.name()),
            song.key_at,
        ),
        Some(None) => stage(Kind::Key, Stand::Missing, "No clear key", song.key_at),
        None if analysing => stage(Kind::Key, Stand::Running, "Listening for the key…", None),
        None if done(Job::Rhythm) && facts.key == Some(true) => {
            stage(Kind::Key, Stand::Found, "Found it", None)
        }
        None if done(Job::Rhythm) && facts.key == Some(false) => {
            stage(Kind::Key, Stand::Missing, "No clear key", None)
        }
        None => stage(Kind::Key, Stand::Waiting, "The key", None),
    });
    // The stems, where they are separated.
    let stems_asked = state.prepared.recipe.as_ref().is_some_and(|r| r.stems)
        || song.separating.is_some()
        || song.stems;
    if stems_asked {
        out.push(if song.stems {
            stage(
                Kind::Stems,
                Stand::Found,
                "Found them: four stems",
                song.stems_at,
            )
        } else if let Some(s) = song.separating.as_ref().filter(|_| analysing) {
            let line = format!("Taking the song apart… {:.0} %", 100.0 * s.share);
            stage(Kind::Stems, Stand::Running, &line, None)
        } else if done(Job::Rhythm) && facts.stems == Some(true) {
            stage(Kind::Stems, Stand::Found, "Found them: four stems", None)
        } else if done(Job::Rhythm) && facts.stems == Some(false) {
            stage(Kind::Stems, Stand::Missing, "Not taken apart", None)
        } else {
            stage(Kind::Stems, Stand::Waiting, "The stems", None)
        });
    }
    // The words.
    let lyrics = working(Job::Lyrics);
    let found_line = |origin| match origin {
        Origin::Lrclib => "Found on LRCLIB",
        Origin::Tags => "Found in the file's tags",
        Origin::Beside => "Found beside the file",
        Origin::Heard => "Heard from the voice",
    };
    out.push(if song.preview.is_some() {
        stage(
            Kind::Words,
            Stand::Found,
            "Timed, word by word",
            song.words_at,
        )
    } else if lyrics {
        let line = match prep::step_of(Job::Lyrics, Some(song)) {
            Step::Hearing => "Listening for the words…",
            Step::Timing => "Timing the words…",
            _ => "Looking up the words…",
        };
        stage(Kind::Words, Stand::Running, line, None)
    } else if let Some(origin) = song.origin {
        stage(Kind::Words, Stand::Found, found_line(origin), song.words_at)
    } else if done(Job::Lyrics) {
        match facts.lyrics.map(|l| l.how) {
            Some(LyricsFacts::Timed | LyricsFacts::Untimed) => {
                stage(Kind::Words, Stand::Found, "Found them", None)
            }
            Some(LyricsFacts::Heard) => stage(Kind::Words, Stand::Found, "Heard them", None),
            Some(LyricsFacts::Instrumental) => stage(
                Kind::Words,
                Stand::Missing,
                "No words: it is instrumental",
                None,
            ),
            _ => stage(Kind::Words, Stand::Missing, "No words found", None),
        }
    } else {
        stage(Kind::Words, Stand::Waiting, "The words", None)
    });
    // The story, where stories are made: where it stands, in its words.
    let story = StoryStand::of(state, path);
    let stand = match story {
        StoryStand::Off => return out,
        StoryStand::Making(_) => Stand::Running,
        StoryStand::Waiting { .. } | StoryStand::NotNow(_) => Stand::Waiting,
        StoryStand::Made => Stand::Found,
        StoryStand::NoStory => Stand::Missing,
    };
    out.push(stage(Kind::Story, stand, &story.line(), None));
    out
}

/// What the view shows big, and its headline where it is a result: the
/// result found last while it is held ([`HOLD_MS`]; a scene drawn shows
/// the story, while it is still being made); else the stems as they are
/// separated; else the story on the cover while it is being made (once
/// anything of it is told, or a worker is at it); else the song's cover.
fn hero(
    state: &State,
    path: &Path,
    song: &SongPrep,
    now: u64,
    moves: bool,
) -> (Hero, Option<String>) {
    let story = StoryStand::of(state, path);
    let held = (song.found.as_ref())
        .filter(|(_, at)| state.now_ms.saturating_sub(*at) < HOLD_MS)
        .filter(|(found, _)| story.making() || !matches!(found, Found::Scene { .. }));
    let separating = song.separating.as_ref().filter(|_| {
        !song.stems
            && state
                .working
                .get(path)
                .is_some_and(|j| j.contains(&Job::Rhythm))
    });
    let shown = |found: &Found| -> Option<(Hero, Option<String>)> {
        let headline = |line: String| Some(line);
        match found {
            Found::Tempo(bpm) => {
                let beats = song.beats.as_ref()?;
                Some((
                    Hero::Beat(beat_view(beats, now, song.beats_at, moves)),
                    headline(format!("Found it: {bpm:.0} BPM")),
                ))
            }
            Found::Key(key) => {
                let t = key.tonic();
                let third = if key.minor() { 3 } else { 4 };
                Some((
                    Hero::Key(KeyView {
                        name: key.name(),
                        lit: vec![t, t + third, t + 7],
                    }),
                    headline(format!("Found it: {}", key.name())),
                ))
            }
            Found::Stems => Some((
                Hero::Stems(stems_view(song.separating.as_deref()?, 1.0)),
                headline("Found them: four stems".to_owned()),
            )),
            Found::Words(_) | Found::Timed => {
                let preview = song.preview.as_ref()?;
                Some((
                    Hero::Words(words_view(preview, now, song.words_at, moves)),
                    headline("The words, in time".to_owned()),
                ))
            }
            Found::Scene { n, of } => Some((
                Hero::Story(story_view(song, now, moves)),
                headline(format!("Scene {n} of {of} drawn")),
            )),
        }
    };
    if let Some(found) = held.and_then(|(f, _)| shown(f)) {
        return found;
    }
    if let Some(s) = separating {
        return (Hero::Stems(stems_view(s, s.share)), None);
    }
    // Built on the cover while it is made (being made now, or to come
    // with some of it told); else the cover.
    match story {
        StoryStand::Making(_) => (Hero::Story(story_view(song, now, moves)), None),
        StoryStand::Waiting { .. } if song.story.told() => {
            (Hero::Story(story_view(song, now, moves)), None)
        }
        _ => (Hero::Cover, None),
    }
}

/// The story of `song` as written and drawn so far, by `now` (its caret
/// blinking while the view `moves`, each scene fading in as it came).
fn story_view(song: &SongPrep, now: u64, moves: bool) -> StoryView {
    let story = &song.story;
    StoryView {
        text: story.text.clone(),
        caret: moves && (now / 500).is_multiple_of(2),
        scenes: (story.scenes.iter())
            .map(|s| {
                let shown = reveal(now, Some(s.at), REVEAL_MS);
                (s.index, Arc::clone(&s.picture), shown)
            })
            .collect(),
        of: story.of,
    }
}

/// The beat of `beats` (found at `at`) by `now`: its ticks coming in left
/// to right over [`REVEAL_MS`], then, while the view `moves`, a play head
/// going over the stretch in real time with a pulse on each beat it
/// passes.
fn beat_view(beats: &BeatsFound, now: u64, at: Option<u64>, moves: bool) -> BeatView {
    let swept = reveal(now, at, REVEAL_MS);
    let ticks: Vec<(f32, bool)> = (beats.beats.iter())
        .filter(|(across, _)| *across <= swept)
        .copied()
        .collect();
    let since = at.map_or(0, |at| now.saturating_sub(at + REVEAL_MS));
    let going = moves && swept >= 1.0 && beats.seconds > 0.0;
    let head = going.then(|| ((since as f64 / 1000.0) % beats.seconds / beats.seconds) as f32);
    let pulse = head
        .and_then(|h| {
            (beats.beats.iter())
                .rfind(|(across, _)| *across <= h)
                .map(|&(across, bar)| {
                    let ago = (h - across) * beats.seconds as f32;
                    ((1.0 - ago / PULSE_SECONDS).max(0.0), bar)
                })
        })
        .unwrap_or((0.0, false));
    BeatView {
        peaks: beats.peaks.iter().map(|&p| height(p)).collect(),
        ticks,
        head,
        pulse,
    }
}

/// The stems of `so_far`, `share` of the way separated.
fn stems_view(so_far: &prep::Separating, share: f32) -> StemsView {
    // The voice first, then the drums, the bass and the rest.
    let order = [(3, "Voice"), (1, "Drums"), (0, "Bass"), (2, "Other")];
    StemsView {
        share,
        lanes: order
            .iter()
            .map(|&(k, name)| (name, so_far.peaks[k].iter().map(|&p| height(p)).collect()))
            .collect(),
    }
}

/// The words of `preview` (timed at `at`) lit by `now`: played over and
/// over while the view `moves`, a rest between; at rest, unlit.
fn words_view(preview: &prep::Preview, now: u64, at: Option<u64>, moves: bool) -> WordsView {
    let round = preview.seconds + PREVIEW_REST;
    let t = if moves {
        (at.map_or(0, |at| now.saturating_sub(at)) as f32 / 1000.0) % round.max(0.1)
    } else {
        -1.0
    };
    WordsView {
        lines: (preview.lines.iter())
            .map(|line| {
                line.iter()
                    .map(|w| {
                        let lit = ((t - w.start) / (w.end - w.start).max(0.05)).clamp(0.0, 1.0);
                        (w.text.clone(), lit)
                    })
                    .collect()
            })
            .collect(),
    }
}

/// An outline of the view, one line a piece with its tone in brackets:
/// for the tests.
impl fmt::Display for PrepView {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let done = if self.done { " done" } else { "" };
        writeln!(f, "[prep{done}] {}", self.title)?;
        writeln!(f, "[headline] {}", self.headline)?;
        let hero = match &self.hero {
            Hero::Cover => "cover".to_owned(),
            Hero::Beat(b) => format!("beat: {} ticks", b.ticks.len()),
            Hero::Key(k) => format!("key: {}", k.name),
            Hero::Stems(s) => format!("stems: {:.0} %", 100.0 * s.share),
            Hero::Words(w) => format!(
                "words: {}",
                (w.lines.iter().flatten())
                    .map(|(t, _)| t.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
            Hero::Story(s) => format!("story: {} ({} of {} scenes)", s.text, s.scenes.len(), s.of),
        };
        writeln!(f, "[hero] {hero}")?;
        for s in &self.stages {
            let stand = match s.stand {
                Stand::Waiting => "waiting",
                Stand::Running => "running",
                Stand::Found => "found",
                Stand::Missing => "missing",
            };
            writeln!(f, "[{stand}] {:?}: {}", s.kind, s.line)?;
        }
        Ok(())
    }
}
