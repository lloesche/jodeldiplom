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

use jodel_playback::{Command, PlayState};

use crate::core::keymap::Op;
use crate::core::sliders::Slider;
use crate::core::transition::Status;
use crate::core::{Effect, Gpu, Screen, Side, State};

pub mod controls;
mod countdown;
mod lyrics;
pub mod pointer;
mod result;
pub use countdown::{
    countdown_for, counts_from, first_sung, resume_count, title_shown, Countdown, COUNT,
    COUNT_STEP, TITLE_FADE, TITLE_HOLD,
};
use countdown::{in_lead, note_heard};
pub(crate) use lyrics::{lyrics_arrived, want_lyrics};
pub use lyrics::{lyrics_of, lyrics_wanted, not_singable, ModelState, SongLyrics, HEARD_WARNING};
pub(crate) use result::ended;
pub use result::{clock_ms, result_motion, result_now, result_shown};

/// The queue index of the song the mode is about: the one held for START,
/// else the one heard, else the queue's current one, else the one
/// selected in the queue. The one the tab and the mode show, entering the
/// mode holds, START plays, and `N` and `P` go on from.
pub fn song_index(state: &State) -> Option<usize> {
    state
        .karaoke
        .held
        .or(state.playing.track.as_ref().map(|t| t.index))
        .or(state.queue.current)
        .or(Some(state.queue.cursor))
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
    /// The lyrics editor, while it is open on a song.
    pub editor: Option<super::lyric_edit::LyricEditor>,
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
            if pausable(state) {
                fx.push(Effect::Engine(Command::Pause));
            } else if state.karaoke.resume_at.is_some() {
                // Space again during a resume's count-in: it stays paused.
                state.karaoke.resume_at = None;
            } else {
                start(state, fx);
            }
        }
        Op::Next | Op::Previous => {
            let Some(current) = song_index(state) else {
                return true;
            };
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

/// Whether a song is heard in the mode that `space` pauses: playing, not
/// held for START, not counting in (the silence before a song is no song
/// to pause; its intro, with the count-in to come, is).
pub fn pausable(state: &State) -> bool {
    let k = &state.karaoke;
    let leading = k.countdown.is_some() && in_lead(state);
    let counting = leading || k.resume_at.is_some();
    in_mode(state) && state.playing.state == PlayState::Playing && k.held.is_none() && !counting
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
    let song = song_index(state);
    if refused(state, song) {
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
        // Negative times: the silence before the track.
        if heard == Some(countdown.index) && state.heard() < 0.0 {
            state.karaoke.held = Some(countdown.index);
            fx.push(Effect::Engine(Command::Stop));
        }
    }
    state.karaoke.dragging = None;
    state.karaoke.demo.let_go();
    state.karaoke.controls = controls::Controls::Hidden;
    fx.push(Effect::Engine(Command::HoldAtEnd(false)));
}

/// START: the song the mode is about ([`song_index`]: the one held, else
/// the queue's current one) plays from its start after its count-in; a
/// paused song goes on after the same count-in.
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
    let song = song_index(state);
    match state.playing.state {
        PlayState::Playing if !held => {}
        // Nothing held: the song is the one paused.
        PlayState::Paused if !held => {
            if refused(state, song) {
                return;
            }
            state.karaoke.resume_at =
                Some(state.now_ms + (COUNT as f64 * COUNT_STEP * 1000.0) as u64);
        }
        _ => {
            let Some(index) = song else {
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

/// The engine holds track `index` at its start (the song before ended).
pub(crate) fn held(state: &mut State, index: usize) {
    state.karaoke.held = Some(index);
    state.karaoke.countdown = None;
}

/// A track starts to be heard (START's, one a mix ran into, one played on
/// the tab after the mode): the engine holds none now, and the result of
/// the song before is gone.
pub(crate) fn track_started(state: &mut State) {
    state.karaoke.held = None;
    result::track_started(state);
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
mod tests;
