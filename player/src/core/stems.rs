//! The song's stems on the desk: drums, bass, vocals and other are strips
//! inside the player's ([`super::desk`]), each heard at its fader, its mute
//! and its solo, and the mixer plays the stems in place of the file only
//! while one of them is touched.
//!
//! - The stem keys (`alt+1` to `alt+4`, the bottom bar's buttons, the
//!   palette) mute a stem's strip, or unmute it: on the first beat of the
//!   next bar at least [`DESK_SOON`] ahead ([`SEPARATE_SOON`] while the
//!   stems are still to be separated), with Stem switching "on the next
//!   bar" (also with no stems made yet: the engine starts them on the frame
//!   heard and writes again what it holds); at once without bars, or with
//!   it "instantly". A key pressed again before that bar takes the change
//!   back: what is meant is the desk's setting, set at once.
//! - `alt+0` puts every stem's strip back as it starts (unmuted, 0 dB,
//!   not soloed), so the file is heard as it is again, bit-perfect.
//! - The engine makes the playing track's stems ([`MakeStems`]) while a
//!   stem's strip is touched (or a change of one waits for its bar), the
//!   Mixer, Karaoke or Transition tab shows, or a transition is armed or
//!   mixed (its lanes fade each stem): from the stems kept, else separated
//!   on the GPU (without one, only the kept ones). Otherwise it makes none,
//!   so the GPU and the processor rest.
//! - The Karaoke tab's voice is the vocals lane's gain, a controller's on
//!   top of the strip ([`lanes_due`]): while the tab or the mode shows, 1
//!   elsewhere.
//! - A transition's two tracks play on the same strips, each on a head of
//!   the player's own (`desk_wiring`): the stems' strips act on both, a
//!   bar's change on its frame of the head the playing track is on (the
//!   playing track's bars and the position last told: through the mix the
//!   track handed over from, then the next).
//!
//! What the engine and the mixer were last told of all this is kept in
//! [`StemsSent`], and every change goes out from [`follow`] (after any
//! action) or a stem key, once.

use jodel_playback::stems::{Stem, STEMS};
use jodel_playback::{Command, MakeStems};

use super::desk::{taper, DeskChange, StripKind};
use super::lanes::{self, Block};
use super::transition::Status;
use super::update::say_info;
use super::{karaoke, Effect, Gpu, HeadFrame, Rhythm, Screen, Side, State};

/// Seconds ahead of what is heard that a stem's strip can be muted on a bar
/// at the soonest: the mixer takes a change for a frame it has yet to mix.
pub const DESK_SOON: f64 = 0.1;

/// The same while the playing track's stems are not there yet and must be
/// separated on the GPU first (none kept): the separator takes some 0.4 to
/// 0.5 s for its first chunk, and only then does the engine write again,
/// with the stems, from the frame heard by then. A change on a frame heard
/// before that would be heard late, so it lands on a bar past that first
/// chunk and [`DESK_SOON`] besides, with room to spare.
pub const SEPARATE_SOON: f64 = 0.75;

/// Seconds after a bar that a change on it still has the stems made: its
/// ramps there have long settled.
const DUE_AFTER: f64 = 0.5;

/// What the engine and the mixer were last given of the stems, so a change
/// goes out once ([`follow`]): as a new engine and mixer start (every gain
/// at 1, no stems made) until [`engine_started`] gives them what is due.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StemsSent {
    /// The stems' lanes' gains ([`lanes_due`]).
    pub lanes: [f64; STEMS],
    /// When the stems are made ([`make_wanted`]).
    pub make: MakeStems,
}

impl Default for StemsSent {
    fn default() -> Self {
        StemsSent {
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
    /// That beat's frame of the player's output: of the head the position
    /// last told is of (the playing track's, [`super::Anchor::at`]).
    pub at: HeadFrame,
    /// Its number, from 1.
    pub number: usize,
}

/// When the engine should make the playing track's stems: while a stem's
/// strip is touched (or a change of one waits for its bar), the Mixer,
/// Karaoke or Transition tab (or the mode) shows, or a transition is armed
/// or mixed (both its tracks' stems, each on its head, faded by its lanes);
/// from the stems kept, else on the GPU (only kept ones without one, and
/// while it is looked for).
pub fn make_wanted(state: &State) -> MakeStems {
    let tab = state.screen == Screen::Main
        && matches!(state.side, Side::Mixer | Side::Karaoke | Side::Transition);
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
    let transition = state.transition.draft.status != Status::Draft;
    let shown = tab || karaoke::in_mode(state);
    if !(shown || pending || transition || state.desk.stems_touched()) {
        return MakeStems::Never;
    }
    match state.gpu {
        Gpu::Present(_) => MakeStems::Separate,
        Gpu::Looking | Gpu::Absent => MakeStems::Kept,
    }
}

/// The stems' lanes' gains (drums, bass, vocals, other): the Karaoke tab's
/// voice on the vocals while the tab or the mode shows (its slider read on
/// the fader's taper), else all 1.
pub fn lanes_due(state: &State) -> [f64; STEMS] {
    let voice = if karaoke::shown(state) {
        taper::gain(karaoke::voice_db(state.config.karaoke_voice))
    } else {
        1.0
    };
    StripKind::PARTS.map(|k| if k == StripKind::Vocals { voice } else { 1.0 })
}

/// Whether `stem` is off: its strip muted, or its fader all the way down.
pub fn is_off(state: &State, stem: Stem) -> bool {
    let s = state.desk.setting(StripKind::for_stem(stem));
    s.mute || s.db == f64::NEG_INFINITY
}

/// How far ahead of what is heard a stem's change can land now
/// ([`DESK_SOON`], [`SEPARATE_SOON`]), on the playing track, with a
/// transition armed or mixed too (the desk plays both its tracks'
/// stems on their strips). Stems not there yet start on the frame heard,
/// and the engine writes what it holds again with them once they are made:
/// kept ones at once (some 15 ms), so [`DESK_SOON`] is enough; separated
/// ones once their first chunk is ([`separated_first`]). It cannot while a
/// jump's crossfade is held, a route switch is under way, a transition is
/// set up or a jump waits for its time: a first touch then lands up to
/// about a second late.
fn lead(state: &State) -> f64 {
    if separated_first(state) {
        SEPARATE_SOON
    } else {
        DESK_SOON
    }
}

/// Whether the stems must be separated before a change of them is heard:
/// they are not there (none made, or the desk does not play them yet on
/// every head with an output open), and of the tracks they are for
/// ([`stem_tracks`]) one has none kept (as the queue's facts or its
/// analysis know) and none failed, and the GPU separates them. Without
/// one, stems not kept never come (a stem key says so).
fn separated_first(state: &State) -> bool {
    let there = state.stems_sent.make != MakeStems::Never && state.desk.stems_there();
    let gpu = matches!(state.gpu, Gpu::Present(_));
    let waiting = stem_tracks(state)
        .into_iter()
        .any(|i| !kept(state, i) && !state.desk.parts_failed.contains(&i));
    !there && waiting && gpu
}

/// The tracks the desk plays the stems of: the playing one, and while a
/// transition is armed or mixed its two (each on a head).
fn stem_tracks(state: &State) -> Vec<usize> {
    let mut tracks: Vec<usize> = state.playing.track.iter().map(|t| t.index).collect();
    if let Some(sent) = state.transition.draft.in_force() {
        tracks.extend([sent.plan.from, sent.plan.into]);
    }
    tracks.sort_unstable();
    tracks.dedup();
    tracks
}

/// Whether track `index` has its stems kept, as its analysis or the
/// queue's facts know.
fn kept(state: &State, index: usize) -> bool {
    let Some(path) = state.queue.paths.get(index) else {
        return false;
    };
    let analysed = matches!(state.rhythm.get(path), Some(Rhythm::Ready(r)) if r.analysis.stems);
    analysed || state.facts.get(path).is_some_and(|f| f.stems == Some(true))
}

/// The bar a stem's change lands on now: the first of the playing track far
/// enough after what is heard ([`lead`]), its frame worked out from the
/// last position the engine told and how fast the output went through the
/// track between the last two, on the head that position was told of;
/// `None` with the changes at once (the setting, no bars there, fewer than
/// two positions since the track started, a seek or a jump: the first can
/// come before the mixer has mixed anything of the output again, with no
/// frame of it to time a change on). A transition's hand-over starts the
/// next track on the other head: its bars are timed from its own
/// positions, on its head's frames, once two are told.
pub fn bar(state: &State) -> Option<Bar> {
    if !state.config.stems_on_bar {
        return None;
    }
    let track = state.playing.track.as_ref()?;
    let anchor = state.playing.anchor?;
    let speed = state.playing.speed?;
    if anchor.index != track.index {
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
    let frame = anchor.at.frame as f64 + (at - anchor.seconds) * speed;
    Some(Bar {
        seconds: at,
        at: HeadFrame {
            head: anchor.at.head,
            frame: frame.round().max(0.0) as u64,
        },
        number,
    })
}

/// `alt+1` to `alt+4`: `stem`'s strip muted, or unmuted, on the next bar.
pub(crate) fn toggle(state: &mut State, stem: Stem, fx: &mut Vec<Effect>) {
    let kind = StripKind::for_stem(stem);
    let on = !state.desk.setting(kind).mute;
    let bar = bar(state);
    mute(state, kind, on, bar, fx);
    told(state, bar);
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
    told(state, bar);
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
        Some(bar) => DeskChange::MuteAt(kind, on, bar.at),
        None => DeskChange::Mute(kind, on),
    }));
}

/// After a stem key: what is heard said.
fn told(state: &mut State, bar: Option<Bar>) {
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

/// After any action: the mixer's lanes and when the engine makes stems
/// follow the state when they changed (the Karaoke tab shown or left, its
/// voice moved, a stem's strip touched, a tab shown, a transition armed or
/// done).
pub(crate) fn follow(state: &mut State, fx: &mut Vec<Effect>) {
    let lanes = lanes_due(state);
    for (k, kind) in StripKind::PARTS.into_iter().enumerate() {
        if lanes[k] != state.stems_sent.lanes[k] {
            fx.push(Effect::Desk(DeskChange::Lane(kind, lanes[k])));
        }
    }
    state.stems_sent.lanes = lanes;
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
    if !state.desk.parts_failed.contains(&index) {
        state.desk.parts_failed.push(index);
    }
    say_failed(state, reason);
}

/// A transition's stems stopped ([`jodel_playback::Event::StemsStopped`],
/// no longer sent: a head's parts that stop are told as any parts are):
/// its tracks go on whole.
pub(crate) fn transition_stopped(state: &mut State, reason: &str) {
    say_failed(state, reason);
}

fn say_failed(state: &mut State, reason: &str) {
    super::update::say_error(
        state,
        format!("The stems stopped ({reason}); playing the whole mix"),
    );
}
