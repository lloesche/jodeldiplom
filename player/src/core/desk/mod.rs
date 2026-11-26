//! The Mixer tab ([`crate::core::Side::Mixer`]): the mixer as a desk (a
//! mixing console). A channel
//! strip for the player, a group with a strip for each of its stems inside
//! it (drums, bass, vocals, other), one for the preview and one for the
//! karaoke game's sound effects, each with a fader, mute and solo; the
//! microphone's strip, which is only listened to; the master with the
//! device. Every strip starts at 0 dB and nothing is kept between runs
//! (the desk adds no gain until a fader is moved, and with every stem's
//! strip as it starts the player is heard as its file), but for the
//! Effects strip's fader and mute ([`StripKind::kept`]): the settings keep
//! them (`karaoke_effects`), saved as they change (a fader held, once it
//! is let go).
//!
//! The settings live here and go out as [`Effect::Desk`]; the app sets its
//! strips by them, and again on every new mixer ([`changes`]). What the
//! desk shows comes in once a drawn frame ([`feed`]).

pub mod meter;
pub mod shown;
pub mod strips;
pub mod taper;

use jodel_output::lanes::Parts;
use jodel_output::mixer::{
    DeskView, InputView, MasterView, Meters, MicView, PartsView, Role, StripView,
};

use crate::core::keymap::Op;
use crate::core::update::say_info;
use crate::core::HeadFrame;
use crate::core::{Effect, Screen, Side, State, DOUBLE_CLICK_MS};

pub use meter::{Meter, Pair};
pub use shown::strips_shown;
pub use strips::{Channel, DeskButton, StripKind};

/// What the user set on a strip.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StripSetting {
    /// The fader, in dB (−∞: off).
    pub db: f64,
    /// Muted.
    pub mute: bool,
    /// Soloed.
    pub solo: bool,
}

impl Default for StripSetting {
    fn default() -> Self {
        StripSetting {
            db: 0.0,
            mute: false,
            solo: false,
        }
    }
}

// A fader is never NaN: −∞ to +6 dB.
impl Eq for StripSetting {}

impl StripSetting {
    /// Whether it is touched: anything but as a strip starts.
    pub fn touched(&self) -> bool {
        *self != StripSetting::default()
    }

    /// Its gain: the fader's, 0 while muted.
    pub fn gain(&self) -> f64 {
        if self.mute {
            0.0
        } else {
            taper::gain(self.db)
        }
    }
}

/// A change for the app to make on a strip ([`Effect::Desk`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DeskChange {
    /// The level, a linear gain.
    Level(StripKind, f64),
    /// Muted or not.
    Mute(StripKind, bool),
    /// Muted or not from a frame of the player's output (of its head, as
    /// [`jodel_output::mixer::Fader::at_stream`] takes it): a bar's first
    /// beat.
    MuteAt(StripKind, bool, HeadFrame),
    /// Soloed or not.
    Solo(StripKind, bool),
    /// The controllers' gain of a stem's lane, not the user's strip: the
    /// Karaoke tab's voice on the vocals.
    Lane(StripKind, f64),
    /// The player's own volume ([`DeskState::volume`]), a linear gain.
    Volume(f64),
}

/// A fader held with the mouse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drag {
    /// Which.
    pub strip: StripKind,
}

/// What the mouse does on the desk ([`crate::core::Action::Desk`]): the
/// window's Mixer tab sends these ([`crate::gui::mixer`]), the pointer's
/// positions worked out to the pixel there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DeskAction {
    /// A strip was clicked: it is selected.
    Select(Channel),
    /// A fader was pressed: held from now, its cap at `position` (0 to 1,
    /// [`taper`]). A second press soon after is a double click: 0 dB.
    Grab {
        /// Which.
        strip: StripKind,
        /// Where the cap goes.
        position: f64,
    },
    /// The fader held moved to `position`.
    Slide(f64),
    /// The fader held was let go of.
    Release,
    /// The wheel over a strip: its fader (the master: the device's
    /// volume) by this many steps.
    Wheel(Channel, i32),
    /// A button was clicked.
    Press(DeskButton),
}

/// A fader strip as looked at: the mixer's view of it, and its peaks since
/// the last look.
#[derive(Clone, Debug, PartialEq)]
pub struct StripLook {
    /// The strip.
    pub view: StripView,
    /// Its peaks, one per channel (linear).
    pub peaks: Vec<f32>,
}

/// What the app looked at for a frame ([`feed`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Look {
    /// What goes to the device.
    pub master: MasterView,
    /// The microphone's port.
    pub mic: MicView,
    /// Each fader strip ([`StripKind::index`]), when the mixer has it.
    pub strips: [Option<StripLook>; StripKind::COUNT],
    /// The master's peaks, one per channel (linear).
    pub peaks: Vec<f32>,
    /// The limiter acted since the last look.
    pub limited: bool,
}

impl Look {
    /// The look at `desk` and `meters`, each fader strip found by its
    /// mixer's id (`ids`, by [`StripKind::index`]).
    pub fn of(desk: DeskView, meters: Meters, ids: [u64; StripKind::COUNT]) -> Look {
        let mut strips: [Option<StripLook>; StripKind::COUNT] = Default::default();
        for (slot, id) in strips.iter_mut().zip(ids) {
            *slot = desk.strips.iter().find(|s| s.id == id).map(|s| StripLook {
                view: s.clone(),
                peaks: meters
                    .strips
                    .iter()
                    .find(|(i, _)| *i == id)
                    .map_or_else(Vec::new, |(_, p)| p.clone()),
            });
        }
        Look {
            master: desk.master,
            mic: desk.mic,
            strips,
            peaks: meters.master,
            limited: meters.limited,
        }
    }
}

/// How long the master's LIMIT lamp stays lit after the limiter acted.
pub const LIMIT_HOLD_MS: u64 = 1_000;

/// The meters, as they show.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Readings {
    /// What goes to the device.
    pub master: Pair,
    /// Each fader strip's, after its gain ([`StripKind::index`]).
    pub strips: [Pair; StripKind::COUNT],
    /// The microphone as captured.
    pub mic_in: Meter,
    /// The singer isolated.
    pub mic_voice: Meter,
}

/// The Mixer tab's state.
#[derive(Clone, Debug, Default)]
pub struct DeskState {
    /// The strip the keys act on.
    pub selected: Channel,
    /// What the user set on each fader strip ([`StripKind::index`]): this
    /// run only. A stem's mute is what is meant: one set on a bar is here
    /// at once, and lands there.
    pub strips: [StripSetting; StripKind::COUNT],
    /// The player's own volume in dB, at most 0 (the default: unity): the
    /// master's gain, over everything the desk plays; this run only. The
    /// system's volume is the system's.
    pub volume: f64,
    /// The fader held with the mouse.
    pub dragging: Option<Drag>,
    /// The last press on a fader and when ([`State::now_ms`]), for a
    /// double click.
    pub last_press: Option<(StripKind, u64)>,
    /// The master as last looked at (none before the first look, or with
    /// no engine).
    pub master: Option<MasterView>,
    /// The microphone's port as last looked at.
    pub mic_port: MicView,
    /// Each fader strip as last looked at ([`StripKind::index`]).
    pub views: [Option<StripView>; StripKind::COUNT],
    /// The tracks whose stems stopped
    /// ([`jodel_playback::Event::PartsStopped`]): they have none (the
    /// playing one, and a transition's next track, each on its head).
    pub parts_failed: Vec<usize>,
    /// A stem's change waiting for its bar (the track, seconds into it): the
    /// stems are made until then, even if none is touched any more.
    pub stems_pending: Option<(usize, f64)>,
    /// The meters.
    pub readings: Readings,
    /// When the limiter last acted.
    pub limited_at: Option<u64>,
    /// When the last look came.
    pub looked_at: u64,
}

impl DeskState {
    /// The desk as a run starts: every strip as a strip starts, the
    /// Effects strip as the settings keep it.
    pub fn new(config: &crate::config::Config) -> DeskState {
        let mut desk = DeskState::default();
        desk.strips[StripKind::Effects.index()] = StripSetting {
            solo: false,
            ..config.karaoke_effects
        };
        desk
    }

    /// What is set on `strip`.
    pub fn setting(&self, strip: StripKind) -> StripSetting {
        self.strips[strip.index()]
    }

    /// What `strip`'s own inputs are heard at as the desk is set (a linear
    /// gain): its fader and mute times its group's, and its solo gate as
    /// the mixer's strips have it: while any strip is soloed, a strip is
    /// heard only where it or the group it sits in is.
    pub fn heard_gain(&self, strip: StripKind) -> f64 {
        let soloed = |k: StripKind| self.setting(k).solo;
        let any = StripKind::ALL.into_iter().any(soloed);
        let group = strip.parent();
        if any && !soloed(strip) && !group.is_some_and(soloed) {
            return 0.0;
        }
        self.setting(strip).gain() * group.map_or(1.0, |g| self.setting(g).gain())
    }

    /// `strip` as last looked at.
    pub fn view(&self, strip: StripKind) -> Option<&StripView> {
        self.views[strip.index()].as_ref()
    }

    /// `strip`'s meters.
    pub fn meters(&self, strip: StripKind) -> &Pair {
        &self.readings.strips[strip.index()]
    }

    /// The switch of the player's input between its main and its stems, as
    /// last looked at: the Lead's (the head on air), else the first with
    /// lanes; `None` without an input of lanes.
    pub fn parts(&self) -> Option<PartsView> {
        let inputs = &self.view(StripKind::Player)?.inputs;
        let lead = inputs.iter().find(|i| i.role == Role::Lead);
        lead.and_then(|i| i.parts)
            .or_else(|| inputs.iter().find_map(|i| i.parts))
    }

    /// Whether the player's stems are on the desk, as last looked at: every
    /// head of it with an output open has its parts there (the one on air,
    /// and a transition's next track's), and there is one.
    pub fn stems_there(&self) -> bool {
        let present = |p: Option<PartsView>| p.is_some_and(|p| p.state == Parts::Present);
        let open = self.open_heads();
        if open.is_empty() {
            // None open as looked at: the Lead's parts as it left them.
            return present(self.parts());
        }
        open.iter().all(|i| present(i.parts))
    }

    /// The player's heads with an output open, as last looked at.
    pub fn open_heads(&self) -> Vec<&InputView> {
        self.view(StripKind::Player).map_or(Vec::new(), |v| {
            v.inputs.iter().filter(|i| i.format.is_some()).collect()
        })
    }

    /// Whether the player's stems have strips on the desk: its input has
    /// lanes.
    pub fn has_stems(&self) -> bool {
        self.parts().is_some()
    }

    /// Whether any of the player's stems is touched.
    pub fn stems_touched(&self) -> bool {
        StripKind::PARTS.iter().any(|&k| self.setting(k).touched())
    }

    /// Whether the LIMIT lamp is lit at `now_ms`.
    pub fn limiting(&self, now_ms: u64) -> bool {
        self.limited_at
            .is_some_and(|at| now_ms.saturating_sub(at) < LIMIT_HOLD_MS)
    }

    fn setting_mut(&mut self, strip: StripKind) -> &mut StripSetting {
        &mut self.strips[strip.index()]
    }
}

/// Whether the Mixer tab shows (a palette over it included).
pub fn shown(state: &State) -> bool {
    state.screen == Screen::Main && state.side == Side::Mixer
}

/// Whether the Mixer tab is what keys act on.
pub fn on_tab(state: &State) -> bool {
    shown(state) && state.palette.is_none()
}

/// Whether the Mixer tab listens to the microphone: while it shows with the
/// microphone's strip selected.
pub fn listens(state: &State) -> bool {
    shown(state) && state.desk.selected == Channel::Mic
}

/// Whether the Mixer tab shows something that moves by itself at `now_ms`:
/// a meter lit ([`meter::Meter::lit`]: falling, or a tick or a clip lamp
/// to go out), the LIMIT lamp lit, a head's automation playing, or the
/// microphone's strip listening (its meters move with the room). Otherwise
/// the tab stands still until something changes it.
pub fn moving(state: &State, now_ms: u64) -> bool {
    let desk = &state.desk;
    let r = &desk.readings;
    let lit = r
        .master
        .iter()
        .chain(r.strips.iter().flatten())
        .chain([&r.mic_in, &r.mic_voice])
        .any(|m| m.lit(now_ms));
    let automated = desk
        .views
        .iter()
        .flatten()
        .flat_map(|v| &v.inputs)
        .any(|i| i.automation.as_ref().is_some_and(|a| !a.settled));
    let mic = listens(state) && state.mic.is_some();
    shown(state) && (lit || desk.limiting(now_ms) || automated || mic)
}

/// What the left button let go of does to the desk, wherever the pointer
/// is: a fader held is let go of. The core knows a fader is held, whatever
/// became of the widget that took it (the strips changed meanwhile: stems
/// came, a head opened, the window was resized).
pub fn let_go(state: &State) -> Option<DeskAction> {
    state.desk.dragging.map(|_| DeskAction::Release)
}

/// After any action: a fader held is let go of once the tab is left.
pub(crate) fn settle(state: &mut State, fx: &mut Vec<Effect>) {
    if !shown(state) {
        if let Some(drag) = state.desk.dragging.take() {
            keep(state, drag.strip, fx);
        }
    }
}

/// Every setting of the desk, for a new mixer: its strips start at 0 dB,
/// and the user's settings are made on them again (a mute meant for a bar
/// at once). The stems' lanes follow from [`crate::core::stems`].
pub fn changes(state: &State) -> Vec<DeskChange> {
    StripKind::ALL
        .iter()
        .flat_map(|&strip| {
            let s = state.desk.setting(strip);
            [
                DeskChange::Level(strip, taper::gain(s.db)),
                DeskChange::Mute(strip, s.mute),
                DeskChange::Solo(strip, s.solo),
            ]
        })
        .chain([DeskChange::Volume(taper::gain(state.desk.volume))])
        .collect()
}

/// Moves the player's own volume to `db` (at most 0 dB), sending it when it
/// changed.
pub fn set_volume(state: &mut State, db: f64, fx: &mut Vec<Effect>) {
    let db = db.min(0.0);
    if state.desk.volume != db {
        state.desk.volume = db;
        fx.push(Effect::Desk(DeskChange::Volume(taper::gain(db))));
    }
}

/// A look this long after the last one finds peaks from long ago (the
/// tab was not shown): its meters are not taken.
pub const STALE_MS: u64 = 500;

/// Takes in what the app looked at for a frame drawn at `now_ms`: the
/// desk, and the meters moved by their ballistics (none from a look
/// [`STALE_MS`] after the last), the microphone's from
/// [`State::mic`].
pub fn feed(state: &mut State, look: Look, now_ms: u64) {
    let mic = state.mic;
    let desk = &mut state.desk;
    let fresh = now_ms.saturating_sub(desk.looked_at) <= STALE_MS;
    let r = &mut desk.readings;
    let none: &[f32] = &[];
    meter::take_pair(
        &mut r.master,
        if fresh { &look.peaks } else { none },
        now_ms,
    );
    for (pair, strip) in r.strips.iter_mut().zip(&look.strips) {
        let peaks = strip.as_ref().filter(|_| fresh).map_or(none, |s| &s.peaks);
        meter::take_pair(pair, peaks, now_ms);
    }
    r.mic_in.take(mic.map(|m| m.peak), now_ms);
    r.mic_voice.take(mic.and_then(|m| m.voice), now_ms);
    if fresh && look.limited {
        desk.limited_at = Some(now_ms);
    }
    desk.views = look.strips.map(|s| s.map(|s| s.view));
    desk.master = Some(look.master);
    desk.mic_port = look.mic;
    desk.looked_at = now_ms;
}

/// Carries out `op` on the Mixer tab; `false` when it is not the tab's (or
/// the tab is not shown).
pub(crate) fn apply(state: &mut State, op: Op, fx: &mut Vec<Effect>) -> bool {
    if !on_tab(state) {
        return false;
    }
    let channel = state.desk.selected;
    match op {
        Op::DeskPrevStrip => select_by(state, -1),
        Op::DeskNextStrip => select_by(state, 1),
        Op::Up | Op::Down | Op::PageUp | Op::PageDown | Op::Top | Op::Bottom => {
            fader_key(state, channel, op, fx)
        }
        Op::DeskMute | Op::DeskSolo => match channel {
            Channel::Strip(strip) => {
                let button = if op == Op::DeskMute {
                    DeskButton::Mute(strip)
                } else {
                    DeskButton::Solo(strip)
                };
                press(state, button, fx);
            }
            Channel::Mic => no_fader(state),
            Channel::Main => say_info(state, MAIN_SAID),
        },
        // Esc does nothing here: only tab and the tab buttons leave.
        Op::Back => {}
        _ => return false,
    }
    true
}

/// `strip`'s fader and mute into the settings, saved, when it is a strip
/// they keep ([`StripKind::kept`]) and they changed; not while its fader is
/// held (once it is let go).
fn keep(state: &mut State, strip: StripKind, fx: &mut Vec<Effect>) {
    let held = state.desk.dragging.is_some_and(|d| d.strip == strip);
    if !strip.kept() || held {
        return;
    }
    let mut s = state.desk.setting(strip);
    s.solo = false;
    // Below what the file takes: off.
    if s.db < -100.0 {
        s.db = f64::NEG_INFINITY;
    }
    if state.config.karaoke_effects != s {
        state.config.karaoke_effects = s;
        fx.push(Effect::SaveConfig);
    }
}

/// What the master says to a key it has no use for.
const MAIN_SAID: &str = "The master has no mute or solo: ↑ ↓ move the player's volume";

fn no_fader(state: &mut State) {
    say_info(
        state,
        "The microphone is only listened to: it has no fader and is never played",
    );
}

/// The strips the arrow keys go through: those shown (the stems' only
/// while the player has them, none the narrowing left out).
fn channels(state: &State) -> Vec<Channel> {
    strips_shown(state.fit.desk_width, state.desk.has_stems()).strips
}

fn select_by(state: &mut State, by: isize) {
    let all = channels(state);
    let at = all
        .iter()
        .position(|&c| c == state.desk.selected)
        .unwrap_or(0) as isize;
    let next = (at + by).clamp(0, all.len() as isize - 1) as usize;
    state.desk.selected = all[next];
}

fn fader_key(state: &mut State, channel: Channel, op: Op, fx: &mut Vec<Effect>) {
    match channel {
        Channel::Strip(strip) => {
            let to = fader_step(state.desk.setting(strip).db, op);
            set_level(state, strip, to, fx);
        }
        Channel::Main => {
            let to = fader_step(state.desk.volume, op);
            set_volume(state, to, fx);
        }
        Channel::Mic => no_fader(state),
    }
}

/// Where a fader at `db` goes for the key `op`.
fn fader_step(db: f64, op: Op) -> f64 {
    match op {
        Op::Up => taper::step(db, 1.0),
        Op::Down => taper::step(db, -1.0),
        Op::PageUp => taper::step(db, 6.0),
        Op::PageDown => taper::step(db, -6.0),
        Op::Top => 0.0,
        _ => f64::NEG_INFINITY,
    }
}

/// Sets `strip`'s fader to `db`, sending it when it changed; whether it
/// did.
fn set_level(state: &mut State, strip: StripKind, db: f64, fx: &mut Vec<Effect>) -> bool {
    let s = state.desk.setting_mut(strip);
    if s.db == db {
        return false;
    }
    s.db = db;
    fx.push(Effect::Desk(DeskChange::Level(strip, taper::gain(db))));
    keep(state, strip, fx);
    true
}

fn press(state: &mut State, button: DeskButton, fx: &mut Vec<Effect>) {
    match button {
        DeskButton::Mute(strip) => {
            let s = state.desk.setting_mut(strip);
            s.mute = !s.mute;
            fx.push(Effect::Desk(DeskChange::Mute(strip, s.mute)));
            keep(state, strip, fx);
        }
        DeskButton::Solo(strip) => {
            let s = state.desk.setting_mut(strip);
            s.solo = !s.solo;
            fx.push(Effect::Desk(DeskChange::Solo(strip, s.solo)));
        }
        DeskButton::Echo => toggle_echo(state, fx),
    }
}

/// The microphone's echo canceller on or off (`karaoke_echo`, saved; the
/// app sets the microphone by it).
pub(crate) fn toggle_echo(state: &mut State, fx: &mut Vec<Effect>) {
    state.config.karaoke_echo = !state.config.karaoke_echo;
    fx.push(Effect::SaveConfig);
    let said = if state.config.karaoke_echo {
        "Echo canceller on: the music's echo is taken out of the microphone"
    } else {
        "Echo canceller off: the microphone as it hears"
    };
    say_info(state, said);
}

/// Carries out a mouse action on the desk.
pub(crate) fn act(state: &mut State, action: DeskAction, fx: &mut Vec<Effect>) {
    if !shown(state) {
        return;
    }
    match action {
        DeskAction::Select(channel) => state.desk.selected = channel,
        DeskAction::Grab { strip, position } => {
            state.desk.selected = Channel::Strip(strip);
            let now = state.now_ms;
            let double = state
                .desk
                .last_press
                .is_some_and(|(s, at)| s == strip && now.saturating_sub(at) <= DOUBLE_CLICK_MS);
            if double {
                // A double click: back to 0 dB, nothing held.
                state.desk.last_press = None;
                state.desk.dragging = None;
                set_level(state, strip, 0.0, fx);
                return;
            }
            state.desk.last_press = Some((strip, now));
            state.desk.dragging = Some(Drag { strip });
            set_level(state, strip, taper::snap(taper::db_at(position)), fx);
        }
        DeskAction::Slide(position) => {
            if let Some(drag) = state.desk.dragging {
                let db = taper::snap(taper::db_at(position));
                // A fader moved is no first click of a double click.
                if set_level(state, drag.strip, db, fx) {
                    state.desk.last_press = None;
                }
            }
        }
        DeskAction::Release => {
            if let Some(drag) = state.desk.dragging.take() {
                keep(state, drag.strip, fx);
            }
        }
        DeskAction::Wheel(channel, steps) => match channel {
            Channel::Strip(strip) => {
                let db = taper::step(state.desk.setting(strip).db, f64::from(steps));
                set_level(state, strip, db, fx);
            }
            Channel::Main => {
                let db = taper::step(state.desk.volume, f64::from(steps));
                set_volume(state, db, fx);
            }
            Channel::Mic => {}
        },
        DeskAction::Press(button) => {
            state.desk.selected = match button {
                DeskButton::Mute(strip) | DeskButton::Solo(strip) => Channel::Strip(strip),
                DeskButton::Echo => Channel::Mic,
            };
            press(state, button, fx);
        }
    }
}

/// What a desk op does away from the tab (from the palette elsewhere).
pub(crate) fn elsewhere(state: &mut State) {
    say_info(state, "That is for the Mixer tab (tab)");
}

#[cfg(test)]
mod tests;
