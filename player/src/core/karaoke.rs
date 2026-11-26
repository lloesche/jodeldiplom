//! The Karaoke tab and Karaoke mode.
//!
//! The tab ([`Side::Karaoke`]) sets karaoke up: the voice stem's level, the
//! microphone (listened to, to hear the singer, never played: what plays
//! never changes for it), the singer's delay (its nudge, and a sync that
//! measures it: [`super::sync`], [`super::lag`]), and the button into the
//! mode. While the tab or
//! the mode shows, the voice plays at its level: the vocals lane's gain on
//! the desk, on top of its strip ([`super::stems::lanes_due`]), so what is
//! heard, metered and recorded on the tab is what the mode plays; leaving
//! the tab puts the lane back at 1. The mode ([`Screen::Karaoke`], fullscreen in the
//! window) is where karaoke happens:
//!
//! - entering it stops the song and holds it at its start (entering it
//!   with shift, [`Op::KaraokeEnterHere`], leaves the song where it is),
//!   and a song's end holds the next one there: only [`Op::KaraokeStart`]
//!   (START's pill on the stage, `enter`, `space`) plays a song, with a
//!   count-in of three seconds ([`Countdown`]) that ends as its first word
//!   is sung, drawn from what is heard: the song plays at once and the
//!   count comes in its intro, or, when it is sung from less than three
//!   seconds in, it starts that much later; a song paused in the mode goes
//!   on after a count-in of its own;
//! - a song heard to its end, the game on, shows how it went while the
//!   next is held, or after the last ([`result_shown`]): the first START
//!   during its reveal completes it, the next starts the song;
//! - `p` held, the demo sings ([`super::singing::Demo`]): for showing the
//!   game off, never a best;
//! - `esc` is the only way out: back to the tab (the voice still at its
//!   level), playback where it is (a count-in under way is called off).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use jodel_lyrics::{Detail, Lyrics, Source};
use jodel_playback::{Command, PlayState};

use crate::lyrics::Answer;

use crate::core::keymap::Op;
use crate::core::sliders::Slider;
use crate::core::transition::Status;
use crate::core::{Effect, Gpu, Screen, Side, State};

pub mod controls;
pub mod pointer;
mod result;
pub use result::{clock_ms, result_motion, result_now, result_shown};
pub(crate) use result::{ended, track_started};

/// The numbers a count-in counts: "3", "2", "1", then the music.
pub const COUNT: usize = 3;
/// How long each number of a count-in shows, in seconds.
pub const COUNT_STEP: f64 = 1.0;
/// How long a song's title and artist show alone as it starts, in seconds,
/// before they fade out.
pub const TITLE_HOLD: f64 = 2.0;
/// How long they take to fade out, in seconds.
pub const TITLE_FADE: f64 = 1.0;
/// A count-in: the track, how long each number shows, and the second of
/// the track heard right after "1" (where the singing starts).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Countdown {
    /// The track.
    pub index: usize,
    /// How long each number shows, in seconds ([`COUNT_STEP`]).
    pub beat: f64,
    /// Seconds into the track heard right after "1": its first sung word,
    /// or its start.
    pub on: f64,
    /// Seconds from when the song starts to be heard (its lead's start)
    /// before the count may begin: the title's time, or none.
    pub title: f64,
}

impl Countdown {
    /// The count-in for track `index`: a number a second, then the track
    /// from its start.
    pub fn new(index: usize) -> Countdown {
        Countdown::before(index, 0.0)
    }

    /// The count-in for track `index` whose singing starts `on` seconds in:
    /// "1" ends there.
    pub fn before(index: usize, on: f64) -> Countdown {
        Countdown {
            index,
            beat: COUNT_STEP,
            on: on.max(0.0),
            title: 0.0,
        }
    }

    /// [`Countdown::before`], after the song's title has shown and faded
    /// ([`TITLE_HOLD`], [`TITLE_FADE`]): the music waits for it where the
    /// singing starts too soon.
    pub fn after_title(index: usize, on: f64) -> Countdown {
        Countdown {
            title: TITLE_HOLD + TITLE_FADE,
            ..Countdown::before(index, on)
        }
    }

    /// Whether `t` (seconds into the track) is in the silence before it.
    pub fn in_lead(&self, t: f64) -> bool {
        t < 0.0
    }

    /// Seconds of silence before the track: the title's time and the
    /// count's beats before [`Countdown::on`], less what the track itself
    /// has before it.
    pub fn lead(&self) -> f64 {
        (self.title + COUNT as f64 * self.beat - self.on).max(0.0)
    }

    /// The number shown at `t` seconds into the track (negative in the
    /// silence before it) and how far through its beat, 0 to 1; `None`
    /// before the count and once it is over.
    pub fn at(&self, t: f64) -> Option<(usize, f64)> {
        let left = (self.on - t) / self.beat;
        if left <= 0.0 || left > COUNT as f64 {
            return None;
        }
        let n = left.ceil();
        Some((n as usize, n - left))
    }

    /// Whether the count is over at `t`.
    pub fn over(&self, t: f64) -> bool {
        t >= self.on
    }
}

/// The warning shown with lyrics a speech recogniser heard ([`SongLyrics::heard`]).
pub const HEARD_WARNING: &str =
    "No trusted source has these lyrics: machine learning heard them from the voice, so they may be wrong.";

/// What is known of a song's lyrics.
#[derive(Clone, Debug, PartialEq)]
pub enum SongLyrics {
    /// Asked for.
    Pending,
    /// Found: the text as found (its words only), and as shown (with our
    /// word times once the speech model has them).
    Found {
        /// As found.
        lyrics: Arc<Lyrics>,
        /// As shown.
        shown: Arc<Lyrics>,
    },
    /// The song has no words.
    Instrumental,
    /// None were found.
    NotFound,
    /// The look-up failed (it is tried again later).
    Failed(String),
}

impl SongLyrics {
    /// From a look-up's answer.
    pub fn from_answer(answer: Answer) -> SongLyrics {
        match answer {
            Answer::Found { lyrics, timed } => {
                let shown = timed.unwrap_or_else(|| lyrics.clone());
                SongLyrics::Found {
                    lyrics: Arc::new(lyrics),
                    shown: Arc::new(shown),
                }
            }
            Answer::Instrumental => SongLyrics::Instrumental,
            Answer::NotFound => SongLyrics::NotFound,
            Answer::Failed(why) => SongLyrics::Failed(why),
        }
    }

    /// Whether its lyrics were heard from the voice by a speech recogniser,
    /// no trusted source having them ([`Source::Transcribed`]).
    pub fn heard(&self) -> bool {
        matches!(self, SongLyrics::Found { lyrics, .. } if matches!(lyrics.source, Source::Transcribed(_)))
    }

    /// In a few words, for the tab.
    pub fn describe(&self) -> String {
        match self {
            SongLyrics::Pending => "looking…".into(),
            SongLyrics::Found { lyrics, shown } => {
                let from = match lyrics.source {
                    Source::Tag => "from its tags",
                    Source::Sidecar => "from its .lrc file",
                    Source::Lrclib => "from LRCLIB",
                    Source::Transcribed(_) => {
                        "heard from the voice by machine learning (no trusted source)"
                    }
                    Source::Unknown => "found",
                };
                let timed = match (&shown.timing.by, shown.timing.detail) {
                    (Some(by), Detail::Words) => format!("words timed ({})", by.name),
                    _ => "not timed yet".into(),
                };
                // How well the text fit what the recogniser heard.
                let fit = match shown.timing.fit {
                    Some(f) if f.swapped => format!(
                        "; its first text did not fit the song, this one does ({} % heard)",
                        f.percent
                    ),
                    Some(f) if f.fits() => format!("; fits the song ({} % heard)", f.percent),
                    Some(f) => format!("; may not fit the song (only {} % heard)", f.percent),
                    None => String::new(),
                };
                format!("{from}, {timed}{fit}")
            }
            SongLyrics::Instrumental => "instrumental".into(),
            SongLyrics::NotFound => "none found".into(),
            SongLyrics::Failed(why) => format!("not looked up: {why}"),
        }
    }
}

/// The songs whose lyrics are wanted: while the tab or the mode shows, the one
/// playing (or held), then the next two in the queue; while the Transition tab
/// shows, its tracks' ([`crate::core::transition::tab::lyrics_wanted`]).
pub fn lyrics_wanted(state: &State) -> Vec<PathBuf> {
    if let Some(paths) = crate::core::transition::tab::lyrics_wanted(state) {
        return paths;
    }
    if !(on_tab(state) || in_mode(state)) {
        return Vec::new();
    }
    let Some(first) = song_index(state) else {
        return Vec::new();
    };
    state
        .queue
        .paths
        .iter()
        .skip(first)
        .take(3)
        .cloned()
        .collect()
}

/// The queue index of the song the mode is about: the one heard, else the
/// one held for START, else the queue's current one.
pub fn song_index(state: &State) -> Option<usize> {
    state
        .playing
        .track
        .as_ref()
        .map(|t| t.index)
        .or(state.karaoke.held)
        .or(state.queue.current)
        .filter(|&i| i < state.queue.paths.len())
}

/// The analysis (and waveform) of the song the mode is about, once in.
pub fn rhythm_of(state: &State) -> Option<&super::Rhythmed> {
    let path = state.queue.paths.get(song_index(state)?)?;
    match state.rhythm.get(path)? {
        super::Rhythm::Ready(r) => Some(r),
        _ => None,
    }
}

/// Asks for the lyrics wanted that are not known yet, when what is wanted
/// changed; forgets those no longer wanted.
pub(crate) fn want_lyrics(state: &mut State, fx: &mut Vec<Effect>) {
    let wanted = lyrics_wanted(state);
    if wanted == state.karaoke.lyrics_wanted {
        return;
    }
    let k = &mut state.karaoke;
    k.lyrics
        .retain(|p, l| wanted.contains(p) && !matches!(l, SongLyrics::Failed(_)));
    let ask: Vec<PathBuf> = wanted
        .iter()
        .filter(|p| !k.lyrics.contains_key(*p))
        .cloned()
        .collect();
    for path in &ask {
        k.lyrics.insert(path.clone(), SongLyrics::Pending);
    }
    k.lyrics_wanted = wanted;
    if !ask.is_empty() {
        fx.push(Effect::WantLyrics(ask));
    }
}

/// A song's lyrics arrived.
pub(crate) fn lyrics_arrived(state: &mut State, path: PathBuf, answer: Answer) {
    // Words timed by us: the speech model is here (fetched meanwhile).
    if let Answer::Found {
        timed: Some(timed), ..
    } = &answer
    {
        if timed.timing.detail == Detail::Words && timed.timing.by.is_some() {
            state.karaoke.model = ModelState::Ready;
        }
    }
    if state.karaoke.lyrics_wanted.contains(&path) {
        state
            .karaoke
            .lyrics
            .insert(path, SongLyrics::from_answer(answer));
    }
}

/// The lyrics of the song at `index`, when known.
pub fn lyrics_of(state: &State, index: usize) -> Option<&SongLyrics> {
    state.karaoke.lyrics.get(state.queue.paths.get(index)?)
}

/// The speech model that times the words. Setup gets it before the player
/// starts ([`crate::setup`]); a player started without (tests) has none.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum ModelState {
    /// Not found, and nowhere to fetch it to: the words are shown, not
    /// timed.
    #[default]
    Missing,
    /// Not here yet: fetched when first needed (unless the lyrics are kept
    /// offline).
    Coming,
    /// There.
    Ready,
}

/// The microphone, as the app opened it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MicShown {
    /// Its name, while it is open.
    pub name: Option<String>,
    /// Why it could not be opened.
    pub error: Option<String>,
    /// Open for a while and nothing but exact zeros came in: muted, or
    /// (macOS) the player is not allowed to use the microphone. (What it
    /// hears is [`crate::core::State::mic`].)
    pub silent: bool,
}

/// Whether the microphone is wanted: while the tab or the mode shows, and
/// while the Mixer tab has its strip selected.
pub fn mic_wanted(state: &State) -> bool {
    (state.screen == Screen::Main && state.side == Side::Karaoke)
        || in_mode(state)
        || super::desk::listens(state)
}

/// The Karaoke tab's and mode's state.
#[derive(Clone, Debug, Default)]
pub struct KaraokeState {
    /// The speech model.
    pub model: ModelState,
    /// The microphone.
    pub mic: MicShown,
    /// The lyrics known, by file.
    pub lyrics: HashMap<PathBuf, SongLyrics>,
    /// The files whose lyrics were last wanted, in order.
    pub lyrics_wanted: Vec<PathBuf>,
    /// The slider the arrow keys move on the tab.
    pub focus: Slider,
    /// The slider being dragged with the mouse.
    pub dragging: Option<Slider>,
    /// The track START plays, when known (the one the engine holds).
    pub held: Option<usize>,
    /// The count-in under way.
    pub countdown: Option<Countdown>,
    /// A paused song goes on at this time ([`State::now_ms`]), after the
    /// count-in that shows until then.
    pub resume_at: Option<u64>,
    /// Where the mode's controls are by the pointer ([`controls`]).
    pub controls: controls::Controls,
    /// When the pointer last moved over the mode, or the mode came
    /// ([`clock_ms`]): it hides a while after ([`pointer`]).
    pub pointer_moved: u64,
    /// Where the song playing has been heard from without a jump (the mode
    /// entered where it was, a seek, another song, a pause ended): a
    /// count-in shows only if it fits whole after it. The track, seconds.
    pub counts_from: Option<(usize, f64)>,
    /// Where the song was heard at the last tick, and when
    /// ([`State::now_ms`]): what a jump is told by.
    pub last_heard: Option<(usize, f64, u64)>,
    /// The singing of the song in the mode, scored.
    pub singing: super::singing::Singing,
    /// The demo key, held in the mode: a perfect singer heard in place of
    /// the microphone.
    pub demo: super::singing::Demo,
    /// How late the singer's voice comes, as the scoring takes it, and what
    /// is kept of it.
    pub lag: super::lag::Lag,
    /// The sync shown on the tab.
    pub sync: Option<super::sync::Session>,
    /// How the last song was sung, its reveal and when it was shown, until
    /// the next is started ([`result_shown`]).
    pub result: Option<Arc<super::singing::Shown>>,
    /// How many results were shown (the last one's id).
    pub results: u64,
    /// The songs' loudness as the karaoke plays them, LUFS, as measured
    /// (none: silent, or not read): the game's sounds follow the sung
    /// song's ([`crate::sfx::conductor::calibration`]); one not here yet is
    /// taken as at the reference.
    pub loudness: HashMap<PathBuf, Option<f64>>,
    /// The best scores sung, by song.
    pub bests: crate::scores::Bests,
    /// Whether the best scores changed since they were last saved.
    pub bests_changed: bool,
}

/// Whether the Karaoke tab is what keys act on.
pub fn on_tab(state: &State) -> bool {
    state.screen == Screen::Main && state.side == Side::Karaoke && state.palette.is_none()
}

/// Whether Karaoke mode is on.
pub fn in_mode(state: &State) -> bool {
    state.screen == Screen::Karaoke
}

/// Whether the Karaoke tab or the mode shows (a palette over the tab
/// included): the voice then plays at its level.
pub fn shown(state: &State) -> bool {
    (state.screen == Screen::Main && state.side == Side::Karaoke) || in_mode(state)
}

/// Whether nothing plays in the mode, so START shows.
pub fn waiting(state: &State) -> bool {
    let k = &state.karaoke;
    let counting = k.countdown.is_some() || k.resume_at.is_some();
    // A song held at its start (stopping, or stopped) waits too.
    in_mode(state) && !counting && (state.playing.state != PlayState::Playing || k.held.is_some())
}

/// Whether the seek bar does nothing now: in the mode, while a song is held
/// for START or counting in (its silence before it, or a paused song's
/// count before it goes on).
pub fn seek_refused(state: &State) -> bool {
    in_mode(state)
        && (state.karaoke.held.is_some() || state.karaoke.resume_at.is_some() || in_lead(state))
}

/// The count-in before a paused song goes on at `now_ms`: the number shown
/// and how far through its second, if one is under way.
pub fn resume_count(state: &State, now_ms: u64) -> Option<(usize, f64)> {
    let at = state.karaoke.resume_at?;
    let left = at.saturating_sub(now_ms) as f64 / 1000.0 / COUNT_STEP;
    if left <= 0.0 || left > COUNT as f64 {
        return None;
    }
    let n = left.ceil();
    Some((n as usize, n - left))
}

/// The count-in at the time heard now, if one is under way.
pub fn count_now(state: &State) -> Option<(usize, f64)> {
    if state.karaoke.resume_at.is_some() {
        return resume_count(state, state.now_ms);
    }
    let countdown = state.karaoke.countdown?;
    let heard = state.playing.track.as_ref()?;
    (heard.index == countdown.index)
        .then(|| countdown.at(state.heard()))
        .flatten()
}

/// Carries out `op` on the Karaoke tab or in the mode; `false` when it is
/// not theirs (or neither is shown).
pub(crate) fn apply(state: &mut State, op: Op, fx: &mut Vec<Effect>) -> bool {
    if in_mode(state) && state.palette.is_none() {
        return apply_mode(state, op, fx);
    }
    if !on_tab(state) {
        return false;
    }
    if super::sync::apply(state, op, fx) {
        return true;
    }
    match op {
        Op::KaraokeEnter | Op::Open | Op::Accept => enter(state, fx, false),
        Op::KaraokeEnterHere => enter(state, fx, true),
        Op::MicNext => fx.push(Effect::NextMic),
        Op::KaraokeRecordEcho => fx.push(Effect::RecordEcho),
        Op::KaraokeSync => super::sync::begin(state, fx),
        Op::KaraokeGame | Op::KaraokeNotes | Op::KaraokeKeep => game_setting(state, op, fx),
        Op::Up | Op::Down => focus_by(state, if op == Op::Up { -1 } else { 1 }),
        Op::SliderDown | Op::SliderUp | Op::SliderFineDown | Op::SliderFineUp => {
            slide_key(state, op, fx)
        }
        // Esc does nothing here: only tab and the tab buttons leave.
        Op::Back => {}
        _ => return false,
    }
    true
}

/// The row above (`by` −1) or below the slider the arrow keys move.
fn focus_by(state: &mut State, by: i32) {
    let all = Slider::ALL;
    let at = all
        .iter()
        .position(|&s| s == state.karaoke.focus)
        .unwrap_or(0) as i32;
    state.karaoke.focus = all[(at + by).clamp(0, all.len() as i32 - 1) as usize];
}

/// The game's settings: scoring on or off (`g`), the notes shown or not
/// (`p`), the singing kept or not; saved, and said.
fn game_setting(state: &mut State, op: Op, fx: &mut Vec<Effect>) {
    let config = &mut state.config;
    let said = match op {
        Op::KaraokeGame => {
            config.karaoke_game = !config.karaoke_game;
            if config.karaoke_game {
                "Game on: the singing is scored, each phrase shining".to_owned()
            } else {
                "Game off: just singing, nothing scored".to_owned()
            }
        }
        Op::KaraokeKeep => {
            config.karaoke_keep = !config.karaoke_keep;
            if config.karaoke_keep {
                "Your singing of each song is kept (recordings, beside the settings)".to_owned()
            } else {
                "Your singing is not kept".to_owned()
            }
        }
        _ => {
            config.karaoke_notes = !config.karaoke_notes;
            if config.karaoke_notes {
                "The song's notes show under the line".to_owned()
            } else {
                "The song's notes hidden".to_owned()
            }
        }
    };
    fx.push(Effect::SaveConfig);
    super::update::say_info(state, said);
}

/// [`apply`] in the mode: START, play and pause, the sliders, the next or
/// previous song (held at its start), and `esc`; quitting and the device
/// volume as anywhere; nothing else (no tabs, screens or seeking).
fn apply_mode(state: &mut State, op: Op, fx: &mut Vec<Effect>) -> bool {
    match op {
        Op::Back => leave(state, fx),
        Op::KaraokeStart | Op::Accept | Op::Open => start(state, fx),
        // Space pauses a song playing; anything else is START (never a
        // song going on without its count-in).
        Op::PlayPause => {
            let k = &state.karaoke;
            // The silence before a song is no song to pause (its intro,
            // with the count-in to come, is).
            let leading = k.countdown.is_some() && in_lead(state);
            let counting = leading || k.resume_at.is_some();
            if state.playing.state == PlayState::Playing && k.held.is_none() && !counting {
                fx.push(Effect::Engine(Command::Pause));
            } else if k.resume_at.is_some() {
                // Space again during a resume's count-in: it stays paused.
                state.karaoke.resume_at = None;
            } else {
                start(state, fx);
            }
        }
        Op::Next | Op::Previous => {
            let current = state
                .playing
                .track
                .as_ref()
                .map(|t| t.index)
                .or(state.karaoke.held)
                .or(state.queue.current)
                .unwrap_or(0);
            let index = if op == Op::Next {
                current + 1
            } else {
                current.saturating_sub(1)
            };
            if index < state.queue.paths.len() {
                state.karaoke.countdown = None;
                state.karaoke.resume_at = None;
                state.karaoke.held = Some(index);
                state.queue.cursor = index;
                fx.push(Effect::Engine(Command::Stop));
            }
        }
        Op::Up | Op::Down => state.karaoke.focus = Slider::Voice,
        Op::SliderDown | Op::SliderUp | Op::SliderFineDown | Op::SliderFineUp => {
            slide_key(state, op, fx)
        }
        Op::MicNext => fx.push(Effect::NextMic),
        Op::KaraokeRecordEcho => fx.push(Effect::RecordEcho),
        Op::KaraokeDemo => state.karaoke.demo.hold(),
        Op::Quit | Op::VolumeUp | Op::VolumeDown => return false,
        _ => {}
    }
    true
}

/// Why track `index` cannot be sung yet, if it cannot: its lyrics still
/// looked for, or their words still being timed (a big folder queued is
/// got ready a track at a time), or known to be missing (none found, an
/// instrumental, the look-up failed). Words that cannot be timed (no speech
/// model, or the timing failed) are sung as they are.
pub fn not_singable(state: &State, index: usize) -> Option<&'static str> {
    let path = state.queue.paths.get(index)?;
    let settled = state
        .ready
        .get(path)
        .is_some_and(|jobs| jobs.contains(&super::Job::Lyrics));
    match lyrics_of(state, index) {
        None | Some(SongLyrics::Pending) => Some("still looking for its lyrics"),
        Some(SongLyrics::NotFound) => Some("no lyrics were found for it"),
        Some(SongLyrics::Instrumental) => Some("it is an instrumental"),
        Some(SongLyrics::Failed(_)) => Some("its lyrics could not be looked up"),
        Some(SongLyrics::Found { shown, .. }) => {
            let timed = shown.timing.detail == jodel_lyrics::Detail::Words;
            (!timed && !settled && state.karaoke.model != ModelState::Missing)
                .then_some("its words are still being timed")
        }
    }
}
/// Whether track `index` (if any) cannot be sung yet ([`not_singable`]),
/// said in the status line.
fn refused(state: &mut State, index: Option<usize>) -> bool {
    let Some((index, why)) = index.and_then(|i| not_singable(state, i).map(|why| (i, why))) else {
        return false;
    };
    let title = super::shown_title(state, &state.queue.paths[index]);
    let next = if in_mode(state) {
        "; N for the next song"
    } else {
        ""
    };
    super::update::say_info(state, format!("No karaoke for {title}: {why}{next}"));
    true
}

/// Into the mode: fullscreen (the window), every stem with the voice at
/// its level, songs held at their ends. The song playing (or paused)
/// stops and is held at its start for START, with its count-in; `here`:
/// it goes on where it is instead (nothing stopped, no count-in). A
/// transition armed is called off (one already mixing plays out: the
/// engine cuts no mix it is heard in). A result from before is gone. Not
/// for a song without its lyrics ready ([`not_singable`]): it says why
/// instead.
pub(crate) fn enter(state: &mut State, fx: &mut Vec<Effect>, here: bool) {
    if refused(state, song_index(state)) {
        return;
    }
    state.screen = Screen::Karaoke;
    state.karaoke.result = None;
    // The voice is the mode's only slider.
    state.karaoke.focus = Slider::Voice;
    state.karaoke.controls = controls::Controls::Hidden;
    // The pointer shows as the mode comes, and hides as if it moved now.
    state.karaoke.pointer_moved = clock_ms(state);
    state.karaoke.countdown = None;
    state.karaoke.resume_at = None;
    if state.transition.draft.status == Status::Armed {
        fx.push(Effect::Engine(Command::CancelTransition));
        state.transition.draft.status = Status::Draft;
        state.transition.draft.sent = None;
    }
    let song = state
        .playing
        .track
        .as_ref()
        .map(|t| t.index)
        .or(state.queue.current)
        .or((!state.queue.paths.is_empty()).then_some(state.queue.cursor))
        .filter(|&i| i < state.queue.paths.len());
    let stopped = state.playing.state == PlayState::Stopped;
    // Entered where the song is: a count-in shows only whole from here (a
    // first word under three seconds away gets none).
    if let (true, Some(index)) = (here, state.playing.track.as_ref().map(|t| t.index)) {
        state.karaoke.counts_from = Some((index, state.heard()));
        state.karaoke.last_heard = Some((index, state.heard(), state.now_ms));
    }
    if !here && !stopped {
        fx.push(Effect::Engine(Command::Stop));
    }
    // Where the song goes on, nothing is held (unless nothing plays).
    state.karaoke.held = if here && !stopped { None } else { song };
    fx.push(Effect::Engine(Command::HoldAtEnd(true)));
    match state.gpu {
        Gpu::Present(_) => {}
        Gpu::Looking => {
            super::update::say_info(state, "Still looking for a GPU: the voice stays in for now")
        }
        Gpu::Absent => super::update::say_info(
            state,
            "No GPU for the stems: the voice is turned down only where its stems are kept",
        ),
    }
}

/// Out of the mode, back to the tab (the voice still at its level), songs
/// running into each other again. A song still in the silence before it
/// is called off (it stops); anything else plays on.
pub(crate) fn leave(state: &mut State, fx: &mut Vec<Effect>) {
    state.screen = Screen::Main;
    state.side = Side::Karaoke;
    state.karaoke.resume_at = None;
    if let Some(countdown) = state.karaoke.countdown.take() {
        let heard = state.playing.track.as_ref().map(|t| t.index);
        if heard == Some(countdown.index) && countdown.in_lead(state.heard()) {
            state.karaoke.held = Some(countdown.index);
            fx.push(Effect::Engine(Command::Stop));
        }
    }
    state.karaoke.dragging = None;
    state.karaoke.demo.let_go();
    state.karaoke.controls = controls::Controls::Hidden;
    fx.push(Effect::Engine(Command::HoldAtEnd(false)));
}

/// START: the song held (or the queue's current one) plays from its start
/// after its count-in; a paused song goes on after the same count-in.
fn start(state: &mut State, fx: &mut Vec<Effect>) {
    if state.karaoke.countdown.is_some() || state.karaoke.resume_at.is_some() {
        return;
    }
    // The first START during the result's reveal completes it, no more.
    if result::skipped(state) {
        return;
    }
    state.karaoke.result = None;
    // A song held (being stopped as the mode began) counts as stopped.
    let held = state.karaoke.held.is_some();
    match state.playing.state {
        PlayState::Playing if !held => {}
        PlayState::Paused if !held => {
            let paused = state.playing.track.as_ref().map(|t| t.index);
            if refused(state, paused) {
                return;
            }
            state.karaoke.resume_at =
                Some(state.now_ms + (COUNT as f64 * COUNT_STEP * 1000.0) as u64);
        }
        _ => {
            let index = state
                .karaoke
                .held
                .or(state.queue.current)
                .or((!state.queue.paths.is_empty()).then_some(state.queue.cursor))
                .filter(|&i| i < state.queue.paths.len());
            let Some(index) = index else {
                super::update::say_info(state, "Nothing to sing: queue some songs first");
                return;
            };
            if refused(state, Some(index)) {
                return;
            }
            let countdown = countdown_for(state, index);
            state.karaoke.countdown = Some(countdown);
            state.karaoke.held = None;
            fx.push(Effect::Engine(Command::PlayIn {
                index,
                lead: countdown.lead(),
            }));
        }
    }
}

/// The count-in for track `index`: after its title has shown and faded,
/// ending as its first word is sung, when its words are timed; else before
/// its start (the title stays, with what is known of the lyrics).
pub fn countdown_for(state: &State, index: usize) -> Countdown {
    match first_sung(state, index) {
        Some(on) => Countdown::after_title(index, on),
        None => Countdown::new(index),
    }
}

/// How much of track `index`'s title and artist shows at `t` (seconds into
/// it, as drawn), 0 to 1: all of it until it plays (held for START, or
/// waiting for its lead), then [`TITLE_HOLD`] from when it is first heard
/// (its lead's start, after START), fading out over [`TITLE_FADE`]; once,
/// whatever comes after (the count-in over, its record gone). None for a
/// song heard from further in (the mode entered with shift, a seek, a pause
/// ended).
pub fn title_shown(state: &State, index: usize, t: f64) -> f64 {
    let playing = state.playing.state == PlayState::Playing
        && state
            .playing
            .track
            .as_ref()
            .is_some_and(|tr| tr.index == index);
    let from = match state.karaoke.countdown {
        Some(c) if c.index == index && !playing => return 1.0,
        Some(c) if c.index == index => -c.lead(),
        _ if !playing && t < TITLE_HOLD => return 1.0,
        // Heard without a jump from there: its lead's start after START
        // (also once the count-in is over), else where it was joined.
        _ => counts_from(state, index),
    };
    if from > 0.05 {
        return 0.0;
    }
    (1.0 - (t - from - TITLE_HOLD) / TITLE_FADE).clamp(0.0, 1.0)
}

/// When the first word of track `index` is sung (seconds in), once its
/// words are timed.
pub fn first_sung(state: &State, index: usize) -> Option<f64> {
    let Some(SongLyrics::Found { shown, .. }) = lyrics_of(state, index) else {
        return None;
    };
    shown
        .lines
        .iter()
        .flat_map(|l| &l.words)
        .filter_map(|w| w.start)
        .reduce(f64::min)
}

/// Whether the song heard is in the silence before it (a count-in's lead).
fn in_lead(state: &State) -> bool {
    match (state.karaoke.countdown, &state.playing.track) {
        (Some(c), Some(t)) => t.index == c.index && c.in_lead(state.heard()),
        (Some(_), None) => true,
        _ => false,
    }
}

/// The engine holds track `index` at its start (the song before ended).
pub(crate) fn held(state: &mut State, index: usize) {
    state.karaoke.held = Some(index);
    state.karaoke.countdown = None;
}

/// Notes where the song is heard from when it jumped (another song, a
/// seek, playing again after a pause): the count-ins ([`counts_from`])
/// only show whole after that.
fn note_heard(state: &mut State) {
    let Some(index) = state.playing.track.as_ref().map(|t| t.index) else {
        state.karaoke.last_heard = None;
        return;
    };
    let at = state.heard();
    let playing = state.playing.state == PlayState::Playing;
    let jumped = match state.karaoke.last_heard {
        Some((i, was, then)) if i == index => {
            let went = (state.now_ms.saturating_sub(then)) as f64 / 1000.0;
            // Playing: about as far as the clock went. Not playing: nothing
            // is heard, so only where it plays from again counts.
            playing && (at - was - went).abs() > 1.0
        }
        _ => true,
    };
    if jumped || !playing {
        state.karaoke.counts_from = Some((index, at));
    }
    state.karaoke.last_heard = Some((index, at, state.now_ms));
}

/// Where the song playing (`index`) has been heard from without a jump: a
/// count-in shows only if it fits whole after it. Before anything is known,
/// from its start.
pub fn counts_from(state: &State, index: usize) -> f64 {
    match state.karaoke.counts_from {
        Some((i, at)) if i == index => at,
        _ => 0.0,
    }
}

/// The clock moved: a count-in heard to its end is over; a paused song's
/// goes on once its count-in is.
pub(crate) fn tick(state: &mut State, fx: &mut Vec<Effect>) {
    note_heard(state);
    if let Some(at) = state.karaoke.resume_at {
        if !in_mode(state) || state.playing.state != PlayState::Paused {
            state.karaoke.resume_at = None;
        } else if state.now_ms >= at {
            state.karaoke.resume_at = None;
            fx.push(Effect::Engine(Command::Resume));
        }
    }
    if let Some(countdown) = state.karaoke.countdown {
        let heard = state.playing.track.as_ref().map(|t| t.index);
        let over = match heard {
            Some(index) => index != countdown.index || countdown.over(state.playing.position),
            None => state.playing.state == PlayState::Stopped && state.karaoke.held.is_some(),
        };
        if over {
            state.karaoke.countdown = None;
        }
    }
}

/// The slider the arrow keys move, a step (`op`: down or up, coarse or
/// fine) of its own.
fn slide_key(state: &mut State, op: Op, fx: &mut Vec<Effect>) {
    let slider = state.karaoke.focus;
    let fine = matches!(op, Op::SliderFineDown | Op::SliderFineUp);
    let up = matches!(op, Op::SliderUp | Op::SliderFineUp);
    let value = slider.stepped(slider.value(state), up, fine);
    set(state, slider, value, fx);
}

/// Moves `slider` by `by` of its steps (the wheel over it: up while
/// positive); the arrow keys move it next.
pub(crate) fn step(state: &mut State, slider: Slider, by: i32, fx: &mut Vec<Effect>) {
    state.karaoke.focus = slider;
    let mut value = slider.value(state);
    for _ in 0..by.unsigned_abs() {
        value = slider.stepped(value, by > 0, false);
    }
    set(state, slider, value, fx);
}

/// Puts `slider` at a share of its range (a click or drag on it), at the
/// nearest of its finest steps ([`Slider::unit`]).
pub(crate) fn slide_to(state: &mut State, slider: Slider, share: f64, fx: &mut Vec<Effect>) {
    state.karaoke.focus = slider;
    let fine = f64::from(slider.unit());
    let span = f64::from(slider.most() - slider.least());
    let value = slider.least() + ((share.clamp(0.0, 1.0) * span / fine).round() * fine) as i32;
    set(state, slider, value, fx);
}

/// `slider` at `value`: the voice kept in the settings and heard at it at
/// once on the tab and in the mode (the vocals lane, after the action:
/// [`super::stems::follow`]);
/// the delay through its nudge, kept with the delays
/// ([`super::lag::set_nudge`]).
fn set(state: &mut State, slider: Slider, value: i32, fx: &mut Vec<Effect>) {
    if slider.value(state) == value {
        return;
    }
    match slider {
        Slider::Voice => state.config.karaoke_voice = value.clamp(0, 100) as u8,
        Slider::Delay => {
            let base = (state.karaoke.lag.base * 1000.0).round() as i32;
            return super::lag::set_nudge(state, value - base);
        }
        Slider::Key => return super::key::set(state, value, fx),
    }
    if !fx.contains(&Effect::SaveConfig) {
        fx.push(Effect::SaveConfig);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_count_in_is_a_number_a_second_then_the_song_from_its_start() {
        let c = Countdown::new(0);
        assert_eq!((c.beat, c.on), (1.0, 0.0));
        assert!((c.lead() - 3.0).abs() < 1e-12);
        // "3" as the silence starts, "2" a second later, "1" the last.
        assert_eq!(c.at(-3.0), Some((3, 0.0)));
        assert_eq!(c.at(-1.5).map(|(n, _)| n), Some(2));
        let (n, through) = c.at(-0.25).unwrap();
        assert_eq!(n, 1);
        assert!((through - 0.75).abs() < 1e-9);
        assert_eq!(c.at(0.0), None);
        assert!(c.over(0.0) && !c.over(-0.01));
    }

    /// After START the title shows and fades, then the count runs, then
    /// the singing: the music waits for all that where the singing starts
    /// sooner, and plays at once (counted in over its intro) where not.
    #[test]
    fn the_title_fades_before_the_count_and_the_music_waits_for_it() {
        let title = TITLE_HOLD + TITLE_FADE;
        // Sung from the start: the title and the count in silence.
        let soon = Countdown::after_title(0, 0.0);
        assert!((soon.lead() - (title + 3.0)).abs() < 1e-12);
        // The title (from the lead's start) is gone as "3" comes.
        let starts = -soon.lead();
        assert_eq!(soon.at(starts + title - 0.01), None);
        assert_eq!(soon.at(starts + title).map(|(n, _)| n), Some(3));
        // Sung 2.5 s in: the music waits for what it lacks.
        let near = Countdown::after_title(0, 2.5);
        assert!((near.lead() - (title + 0.5)).abs() < 1e-12);
        // Sung 20 s in: no wait; the count over the intro.
        let far = Countdown::after_title(0, 20.0);
        assert_eq!(far.lead(), 0.0);
        assert_eq!(far.at(17.0).map(|(n, _)| n), Some(3));
    }
}
