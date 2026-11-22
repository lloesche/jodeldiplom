//! The song's stems on the desk: drums, bass, vocals and other are strips
//! inside the player's ([`super::desk`]), each heard at its fader, its mute
//! and its solo, and the mixer plays the stems in place of the file only
//! while one of them is touched.
//!
//! - The stem keys (`alt+1` to `alt+4`, the bottom bar's buttons, the
//!   palette) mute a stem's strip, or unmute it: on the first beat of the
//!   next bar at least [`DESK_SOON`] ahead ([`ENGINE_SOON`] while the
//!   engine's own mix is heard, or while no stems are being made yet),
//!   with Stem switching "on the next bar"; at once without bars, or with
//!   it "instantly". A key pressed again before that bar takes the change
//!   back: what is meant is the desk's setting, set at once.
//! - `alt+0` puts every stem's strip back as it starts (unmuted, 0 dB,
//!   not soloed), so the file is heard as it is again, bit-perfect.
//! - The engine makes the playing track's stems ([`MakeStems`]) while a
//!   stem's strip is touched (or a change of one waits for its bar) or the
//!   Mix, Karaoke or Transition tab shows: from the stems kept, else
//!   separated on the GPU (without one, only the kept ones). Otherwise it
//!   makes none, so the GPU and the processor rest.
//! - The Karaoke tab's voice is the vocals lane's gain, a controller's on
//!   top of the strip ([`lanes_due`]): while the tab or the mode shows, 1
//!   elsewhere.
//! - What a transition mixes inside the engine follows the strips
//!   ([`mask`], [`Command::StemMask`]): its tracks are heard as the player's
//!   main there.
//!
//! What the engine and the mixer were last told of all this is kept in
//! [`StemsSent`], and every change goes out from [`follow`] (after any
//! action) or a stem key, once.

use jodel_playback::stems::{Stem, STEMS};
use jodel_playback::{Command, MakeStems};

use super::desk::{DeskChange, StripKind};
use super::lanes::{self, Block};
use super::transition::Status;
use super::update::say_info;
use super::{karaoke, Effect, Gpu, Screen, Side, State};

/// Seconds ahead of what is heard that a stem's strip can be muted on a bar
/// at the soonest: the mixer takes a change for a frame it has yet to mix.
pub const DESK_SOON: f64 = 0.1;

/// The same while the engine's mix is what is heard (a transition armed or
/// under way: it is made some half a second ahead), or while no stems are
/// being made yet (they are there that much later).
pub const ENGINE_SOON: f64 = 1.0;

/// Seconds after a bar that a change on it still has the stems made: its
/// ramps there have long settled.
const DUE_AFTER: f64 = 0.5;

/// What the engine and the mixer were last given of the stems, so a change
/// goes out once ([`follow`]): as a new engine and mixer start (every gain
/// at 1, no stems made) until [`engine_started`] gives them what is due.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StemsSent {
    /// The stems' gains for a transition ([`mask`]).
    pub mask: [f64; STEMS],
    /// The stems' lanes' gains ([`lanes_due`]).
    pub lanes: [f64; STEMS],
    /// When the stems are made ([`make_wanted`]).
    pub make: MakeStems,
}

impl Default for StemsSent {
    fn default() -> Self {
        StemsSent {
            mask: [1.0; STEMS],
            lanes: [1.0; STEMS],
            make: MakeStems::Never,
        }
    }
}

/// A bar a stem's change lands on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bar {
    /// Its first beat, in seconds into the playing track.
    pub seconds: f64,
    /// That beat's frame of the player's output (its stream frame).
    pub frame: u64,
    /// Its number, from 1.
    pub number: usize,
}

/// When the engine should make the playing track's stems: while a stem's
/// strip is touched (or a change of one waits for its bar), or the Mix,
/// Karaoke or Transition tab (or the mode) shows; from the stems kept, else
/// on the GPU (only kept ones without one, and while it is looked for).
pub fn make_wanted(state: &State) -> MakeStems {
    let tab = state.screen == Screen::Main
        && matches!(state.side, Side::Mix | Side::Karaoke | Side::Transition);
    // A change still to land on its bar (an unmute, say) needs the stems
    // until then: the desk plays them up to it.
    let pending = state.desk.stems_pending.is_some_and(|(index, at)| {
        state
            .playing
            .track
            .as_ref()
            .is_some_and(|t| t.index == index)
            && state.heard() < at + DUE_AFTER
    });
    if !(tab || pending || karaoke::in_mode(state) || state.desk.stems_touched()) {
        return MakeStems::Never;
    }
    match state.gpu {
        Gpu::Present(_) => MakeStems::Separate,
        Gpu::Looking | Gpu::Absent => MakeStems::Kept,
    }
}

/// The stems' lanes' gains (drums, bass, vocals, other): the Karaoke tab's
/// voice on the vocals while the tab or the mode shows, else all 1.
pub fn lanes_due(state: &State) -> [f64; STEMS] {
    let voice = if karaoke::shown(state) {
        f64::from(state.config.karaoke_voice.min(100)) / 100.0
    } else {
        1.0
    };
    StripKind::PARTS.map(|k| if k == StripKind::Vocals { voice } else { 1.0 })
}

/// Each stem's gain as the mixer plays it (drums, bass, vocals, other): its
/// strip's fader (0 muted) times its lane's gain, times its solo gate (0
/// while another stem is soloed and neither it nor the player is). A
/// transition mixes its tracks with these ([`Command::StemMask`]).
pub fn mask(state: &State) -> [f64; STEMS] {
    let lanes = lanes_due(state);
    let any_solo = StripKind::PARTS.iter().any(|&k| state.desk.setting(k).solo);
    let player_solo = state.desk.setting(StripKind::Player).solo;
    let mut out = [1.0; STEMS];
    for (k, kind) in StripKind::PARTS.into_iter().enumerate() {
        let s = state.desk.setting(kind);
        let gate = if !any_solo || s.solo || player_solo {
            1.0
        } else {
            0.0
        };
        out[k] = s.gain() * lanes[k] * gate;
    }
    out
}

/// Whether `stem` is off: its strip muted, or its fader all the way down.
pub fn is_off(state: &State, stem: Stem) -> bool {
    let s = state.desk.setting(StripKind::for_stem(stem));
    s.mute || s.db == f64::NEG_INFINITY
}

/// How far ahead of what is heard a stem's change can land now
/// ([`DESK_SOON`], [`ENGINE_SOON`]).
fn lead(state: &State) -> f64 {
    let engine_mixes = state.transition.draft.status != Status::Draft;
    if engine_mixes || state.stems_sent.make == MakeStems::Never {
        ENGINE_SOON
    } else {
        DESK_SOON
    }
}

/// The bar a stem's change lands on now: the first of the playing track far
/// enough after what is heard ([`lead`]), its frame worked out from the
/// last position the engine told and how fast the output went through the
/// track between the last two; `None` with the changes at once (the
/// setting, no bars there, fewer than two positions since the track
/// started, a seek or a jump: the first can come before the mixer has
/// mixed anything of the output again, with no frame of it to time a
/// change on).
pub fn bar(state: &State) -> Option<Bar> {
    if !state.config.stems_on_bar {
        return None;
    }
    let track = state.playing.track.as_ref()?;
    let (index, output_frame, seconds) = state.playing.anchor?;
    let speed = state.playing.speed?;
    if index != track.index {
        return None;
    }
    let a = &lanes::rhythmed(state, Block::Now)?.analysis;
    let after = state.heard() + lead(state);
    let (at, number) = a
        .downbeats
        .iter()
        .enumerate()
        .map(|(k, &d)| (a.beats[d], k + 1))
        .find(|&(t, _)| t >= after)?;
    let frame = output_frame as f64 + (at - seconds) * speed;
    Some(Bar {
        seconds: at,
        frame: frame.round().max(0.0) as u64,
        number,
    })
}

/// `alt+1` to `alt+4`: `stem`'s strip muted, or unmuted, on the next bar.
pub(crate) fn toggle(state: &mut State, stem: Stem, fx: &mut Vec<Effect>) {
    let kind = StripKind::for_stem(stem);
    let on = !state.desk.setting(kind).mute;
    let bar = bar(state);
    mute(state, kind, on, bar, fx);
    told(state, bar, fx);
}

/// `alt+0`: every stem's strip as it starts, the mutes on the next bar, so
/// the file is heard as it is again.
pub(crate) fn reset(state: &mut State, fx: &mut Vec<Effect>) {
    let bar = bar(state);
    for kind in StripKind::PARTS {
        let s = state.desk.setting(kind);
        if s.db != 0.0 {
            state.desk.strips[kind.index()].db = 0.0;
            fx.push(Effect::Desk(DeskChange::Level(kind, 1.0)));
        }
        if s.solo {
            state.desk.strips[kind.index()].solo = false;
            fx.push(Effect::Desk(DeskChange::Solo(kind, false)));
        }
        if s.mute {
            mute(state, kind, false, bar, fx);
        }
    }
    told(state, bar, fx);
}

fn mute(state: &mut State, kind: StripKind, on: bool, bar: Option<Bar>, fx: &mut Vec<Effect>) {
    state.desk.strips[kind.index()].mute = on;
    if let (Some(bar), Some(track)) = (bar, &state.playing.track) {
        // The later of two changes waiting keeps the stems made.
        let later = match state.desk.stems_pending {
            Some((index, at)) if index == track.index => at.max(bar.seconds),
            _ => bar.seconds,
        };
        state.desk.stems_pending = Some((track.index, later));
    }
    fx.push(Effect::Desk(match bar {
        Some(bar) => DeskChange::MuteAt(kind, on, bar.frame),
        None => DeskChange::Mute(kind, on),
    }));
}

/// The seconds into the playing track of a stem's change still to come on
/// a bar, if one is.
fn pending_at(state: &State) -> Option<f64> {
    let (index, at) = state.desk.stems_pending?;
    let playing = state.playing.track.as_ref()?.index == index;
    (playing && state.heard() < at).then_some(at)
}

/// The engine's transitions told the stems' gains when they changed: from
/// `at` seconds into the playing track (a bar), else at once.
fn send_mask(state: &mut State, at: Option<f64>, fx: &mut Vec<Effect>) {
    let gains = mask(state);
    if state.stems_sent.mask != gains {
        state.stems_sent.mask = gains;
        fx.push(Effect::Engine(Command::StemMask { gains, at }));
    }
}

/// After a stem key: the engine's transitions told at the same bar, and
/// what is heard said.
fn told(state: &mut State, bar: Option<Bar>, fx: &mut Vec<Effect>) {
    send_mask(state, bar.map(|b| b.seconds), fx);
    let on: Vec<&str> = StripKind::PARTS
        .iter()
        .filter_map(|k| k.stem())
        .filter(|&s| !is_off(state, s))
        .map(|s| s.id())
        .collect();
    let mut what = if on.len() == STEMS && !state.desk.stems_touched() {
        "Stems: all on, the file as it is".to_owned()
    } else if on.is_empty() {
        "Stems: none (not bit-perfect)".to_owned()
    } else {
        format!("Stems: {} (not bit-perfect)", on.join(", "))
    };
    if let Some(bar) = bar {
        what.push_str(&format!(", from bar {}", bar.number));
    }
    match state.gpu {
        Gpu::Present(_) => {}
        Gpu::Looking => what.push_str("; still looking for a GPU: kept stems only"),
        Gpu::Absent => what.push_str("; no GPU: kept stems only"),
    }
    say_info(state, what);
}

/// After any action: the mixer's lanes, the engine's transitions and when
/// the engine makes stems follow the state when they changed (a fader or a
/// mute on the Mix tab, the Karaoke tab shown or left, its voice moved, a
/// tab shown). A change of the gains while a stem key's waits for its bar
/// goes with it, at that bar.
pub(crate) fn follow(state: &mut State, fx: &mut Vec<Effect>) {
    let lanes = lanes_due(state);
    for (k, kind) in StripKind::PARTS.into_iter().enumerate() {
        if lanes[k] != state.stems_sent.lanes[k] {
            fx.push(Effect::Desk(DeskChange::Lane(kind, lanes[k])));
        }
    }
    state.stems_sent.lanes = lanes;
    let at = pending_at(state);
    send_mask(state, at, fx);
    let make = make_wanted(state);
    if make != state.stems_sent.make {
        state.stems_sent.make = make;
        fx.push(Effect::Engine(Command::MakeStems(make)));
    }
}

/// A new engine and mixer: they start as [`StemsSent::default`] has it,
/// and get what is due.
pub fn engine_started(state: &mut State, fx: &mut Vec<Effect>) {
    state.stems_sent = StemsSent::default();
    follow(state, fx);
}

/// The playing track's stems as parts of the output stopped (`reason`): it
/// is heard as its file from there, whatever the strips say, and it has
/// none until they are asked for again (another track has its own). The
/// strips keep their settings.
pub(crate) fn stopped(state: &mut State, index: usize, reason: &str) {
    state.desk.parts_failed = Some(index);
    say_failed(state, reason);
}

/// A transition's stems (the engine's own mix) stopped: its tracks go on
/// whole; the stems on the desk are not touched by it.
pub(crate) fn transition_stopped(state: &mut State, reason: &str) {
    say_failed(state, reason);
}

fn say_failed(state: &mut State, reason: &str) {
    super::update::say_error(
        state,
        format!("The stems stopped ({reason}); playing the whole mix"),
    );
}
