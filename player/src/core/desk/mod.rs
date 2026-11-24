//! The Mixer tab ([`crate::core::Side::Mixer`]): the mixer as a desk. A channel
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
pub mod strips;
pub mod taper;

use jodel_output::lanes::Parts;
use jodel_output::mixer::{
    DeskView, InputView, MasterView, Meters, MicView, PartsView, Role, StripView,
};

use crate::core::keymap::Op;
use crate::core::update::say_info;
use crate::core::HeadFrame;
use crate::core::{Effect, Screen, Side, State, VolumeChange, DOUBLE_CLICK_MS};

pub use meter::{Meter, Pair};
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
        .collect()
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
const MAIN_SAID: &str = "The master adds no gain: ↑ ↓ move the device's own volume";

fn no_fader(state: &mut State) {
    say_info(
        state,
        "The microphone is only listened to: it has no fader and is never played",
    );
}

/// The strips the arrow keys go through: those drawn (the stems' only
/// while the player has them, none the narrowing left out).
fn channels(state: &State) -> Vec<Channel> {
    use crate::core::layout::{self, Layout};
    if let Layout::Main(m) = layout::compute(state) {
        if let Some(desk) = m.desk {
            return desk.channels.iter().map(|c| c.channel).collect();
        }
    }
    let stems = state.desk.has_stems();
    Channel::ALL
        .into_iter()
        .filter(|c| stems || !c.is_part())
        .collect()
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
            let db = state.desk.setting(strip).db;
            let to = match op {
                Op::Up => taper::step(db, 1.0),
                Op::Down => taper::step(db, -1.0),
                Op::PageUp => taper::step(db, 6.0),
                Op::PageDown => taper::step(db, -6.0),
                Op::Top => 0.0,
                _ => f64::NEG_INFINITY,
            };
            set_level(state, strip, to, fx);
        }
        Channel::Main => match op {
            Op::Up => fx.push(Effect::DeviceVolume(VolumeChange::Step(1))),
            Op::Down => fx.push(Effect::DeviceVolume(VolumeChange::Step(-1))),
            _ => say_info(state, MAIN_SAID),
        },
        Channel::Mic => no_fader(state),
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
            Channel::Main => fx.push(Effect::DeviceVolume(VolumeChange::Step(steps))),
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
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::core::{update, Action};
    use jodel_output::mixer::Timebase;

    fn tab() -> State {
        let mut s = State::new("/m".into(), Config::default(), (120, 36));
        s.side = Side::Mixer;
        s
    }

    fn desk_fx(fx: &[Effect]) -> Vec<DeskChange> {
        fx.iter()
            .filter_map(|e| match e {
                Effect::Desk(c) => Some(*c),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn every_control_sends_its_change() {
        let mut s = tab();
        let fx = update(&mut s, Action::Op(Op::Down));
        assert_eq!(
            desk_fx(&fx),
            [DeskChange::Level(StripKind::Player, taper::gain(-1.0))]
        );
        let fx = update(&mut s, Action::Op(Op::DeskMute));
        assert_eq!(desk_fx(&fx), [DeskChange::Mute(StripKind::Player, true)]);
        let fx = update(&mut s, Action::Op(Op::DeskNextStrip));
        assert!(desk_fx(&fx).is_empty());
        let fx = update(&mut s, Action::Op(Op::DeskSolo));
        assert_eq!(desk_fx(&fx), [DeskChange::Solo(StripKind::Preview, true)]);
        let fx = update(&mut s, Action::Op(Op::Bottom));
        assert_eq!(desk_fx(&fx), [DeskChange::Level(StripKind::Preview, 0.0)]);
        let fx = update(&mut s, Action::Op(Op::Top));
        assert_eq!(desk_fx(&fx), [DeskChange::Level(StripKind::Preview, 1.0)]);
        // At 0 dB already: nothing to send.
        assert!(desk_fx(&update(&mut s, Action::Op(Op::Top))).is_empty());
        // The effects' fader: kept in the settings, saved.
        update(&mut s, Action::Op(Op::DeskNextStrip));
        let fx = update(&mut s, Action::Op(Op::Down));
        assert_eq!(
            desk_fx(&fx),
            [DeskChange::Level(StripKind::Effects, taper::gain(-1.0))]
        );
        assert!(fx.contains(&Effect::SaveConfig));
        assert_eq!(s.config.karaoke_effects.db, -1.0);
        let fx = update(&mut s, Action::Op(Op::DeskMute));
        assert!(fx.contains(&Effect::SaveConfig) && s.config.karaoke_effects.mute);
        // The master steps the device's own volume; the microphone has no
        // fader.
        update(&mut s, Action::Op(Op::DeskNextStrip));
        assert!(desk_fx(&update(&mut s, Action::Op(Op::Up))).is_empty());
        update(&mut s, Action::Op(Op::DeskNextStrip));
        let fx = update(&mut s, Action::Op(Op::Up));
        assert!(fx.contains(&Effect::DeviceVolume(VolumeChange::Step(1))));
        assert!(desk_fx(&fx).is_empty());
        // Every setting again for a new mixer: each strip's, the stems'
        // as they start.
        let all = changes(&s);
        assert_eq!(
            all[..3],
            [
                DeskChange::Level(StripKind::Player, taper::gain(-1.0)),
                DeskChange::Mute(StripKind::Player, true),
                DeskChange::Solo(StripKind::Player, false),
            ]
        );
        assert_eq!(
            all[15..18],
            [
                DeskChange::Level(StripKind::Preview, 1.0),
                DeskChange::Mute(StripKind::Preview, false),
                DeskChange::Solo(StripKind::Preview, true),
            ]
        );
        assert_eq!(
            all[18..],
            [
                DeskChange::Level(StripKind::Effects, taper::gain(-1.0)),
                DeskChange::Mute(StripKind::Effects, true),
                DeskChange::Solo(StripKind::Effects, false),
            ]
        );
        for (k, kind) in StripKind::PARTS.into_iter().enumerate() {
            assert_eq!(
                all[3 + 3 * k..6 + 3 * k],
                [
                    DeskChange::Level(kind, 1.0),
                    DeskChange::Mute(kind, false),
                    DeskChange::Solo(kind, false),
                ]
            );
        }
        assert_eq!(all.len(), 21);
    }

    #[test]
    fn the_arrows_go_through_the_stems_only_while_the_player_has_them() {
        let mut s = tab();
        let walk = |s: &mut State| {
            let mut seen = vec![s.desk.selected];
            for _ in 0..8 {
                update(s, Action::Op(Op::DeskNextStrip));
                seen.push(s.desk.selected);
            }
            seen.dedup();
            seen
        };
        assert_eq!(
            walk(&mut s),
            [
                Channel::Strip(StripKind::Player),
                Channel::Strip(StripKind::Preview),
                Channel::Strip(StripKind::Effects),
                Channel::Mic,
                Channel::Main,
            ]
        );
        s.desk.selected = Channel::default();
        feed(&mut s, look_with_stems(), 1_000);
        assert_eq!(walk(&mut s), Channel::ALL);
        // A stem's keys are its strip's: the mute at once on the tab.
        s.desk.selected = Channel::Strip(StripKind::Bass);
        let fx = update(&mut s, Action::Op(Op::DeskMute));
        assert_eq!(desk_fx(&fx), [DeskChange::Mute(StripKind::Bass, true)]);
        assert!(s.desk.stems_touched());
    }

    /// A look with the player's stems on the desk.
    fn look_with_stems() -> Look {
        let mut look = look(0.5, false);
        let player = look.strips[0].as_mut().unwrap();
        player.view.inputs.push(InputView {
            role: Role::Lead,
            gain: 1.0,
            heard: true,
            lanes: Vec::new(),
            parts: Some(PartsView {
                state: Parts::None,
                share: 0.0,
            }),
            format: None,
            queued: 0,
            capacity: 0,
            dropped: 0,
            automation: None,
        });
        look
    }

    /// The effects' strip starts as the settings keep it; its fader
    /// dragged is saved once let go of, not on every step.
    #[test]
    fn the_effects_strip_is_kept_in_the_settings() {
        let config = Config {
            karaoke_effects: StripSetting {
                db: -6.0,
                mute: true,
                solo: false,
            },
            ..Config::default()
        };
        let mut s = State::new("/m".into(), config, (120, 36));
        s.side = Side::Mixer;
        let set = s.desk.setting(StripKind::Effects);
        assert_eq!((set.db, set.mute), (-6.0, true));
        assert!(changes(&s).contains(&DeskChange::Mute(StripKind::Effects, true)));
        let grab = Action::Desk(DeskAction::Grab {
            strip: StripKind::Effects,
            position: 0.3,
        });
        assert!(!update(&mut s, grab).contains(&Effect::SaveConfig));
        let slid = update(&mut s, Action::Desk(DeskAction::Slide(0.2)));
        assert!(!slid.contains(&Effect::SaveConfig));
        assert_eq!(s.config.karaoke_effects.db, -6.0, "not yet");
        let fx = update(&mut s, Action::Desk(DeskAction::Release));
        assert!(fx.contains(&Effect::SaveConfig));
        let db = s.desk.setting(StripKind::Effects).db;
        assert_eq!(s.config.karaoke_effects.db, db);
        // All the way down: off.
        s.desk.selected = Channel::Strip(StripKind::Effects);
        update(&mut s, Action::Op(Op::Bottom));
        assert_eq!(s.config.karaoke_effects.db, f64::NEG_INFINITY);
    }

    #[test]
    fn the_echo_canceller_is_the_setting_and_is_saved() {
        let mut s = tab();
        assert!(s.config.karaoke_echo);
        let fx = update(&mut s, Action::Op(Op::DeskEcho));
        assert!(!s.config.karaoke_echo);
        assert!(fx.contains(&Effect::SaveConfig));
        let fx = update(&mut s, Action::Desk(DeskAction::Press(DeskButton::Echo)));
        assert!(s.config.karaoke_echo);
        assert!(fx.contains(&Effect::SaveConfig));
        assert_eq!(s.desk.selected, Channel::Mic);
    }

    #[test]
    fn away_from_the_tab_the_desk_keys_do_nothing_to_it() {
        let mut s = tab();
        s.side = Side::Browser;
        let fx = update(&mut s, Action::Op(Op::DeskMute));
        assert!(desk_fx(&fx).is_empty());
        assert!(!s.desk.setting(StripKind::Player).mute);
    }

    #[test]
    fn a_fader_flicked_and_taken_again_soon_is_no_double_click() {
        let mut s = tab();
        s.now_ms = 1_000;
        let grab = |position| {
            Action::Desk(DeskAction::Grab {
                strip: StripKind::Player,
                position,
            })
        };
        update(&mut s, grab(0.75));
        update(&mut s, Action::Desk(DeskAction::Slide(0.3)));
        update(&mut s, Action::Desk(DeskAction::Release));
        let flicked = s.desk.setting(StripKind::Player).db;
        assert!(flicked < -20.0, "{flicked}");
        s.now_ms += 200;
        update(&mut s, grab(0.3));
        assert_eq!(s.desk.setting(StripKind::Player).db, flicked);
        assert!(s.desk.dragging.is_some());
    }

    fn look(peak: f32, limited: bool) -> Look {
        let strip = |id, name: &str| StripView {
            id,
            name: name.into(),
            parent: None,
            level: 1.0,
            mute: false,
            solo: false,
            inputs: Vec::new(),
        };
        let desk = DeskView {
            master: MasterView {
                device: "DAC".into(),
                stream: None,
                timebase: Timebase::default(),
                passthrough: true,
                clipped: 0,
            },
            strips: vec![strip(7, "the preview"), strip(3, "the player")],
            mic: MicView::default(),
        };
        let meters = Meters {
            master: vec![peak, peak],
            strips: vec![(7, vec![peak / 2.0]), (3, vec![peak])],
            limited,
        };
        Look::of(desk, meters, [3, 10, 11, 12, 13, 7, 14])
    }

    #[test]
    fn strips_are_found_by_id_and_a_stale_look_takes_no_peaks() {
        let l = look(0.5, true);
        assert_eq!(l.strips[0].as_ref().unwrap().view.id, 3);
        assert_eq!(l.strips[5].as_ref().unwrap().peaks, [0.25]);
        assert!(l.strips[1].is_none(), "no drums on that desk");
        // The first look after none (or long after the last) finds peaks
        // from long ago: neither they nor the limiter count.
        let mut s = tab();
        feed(&mut s, look(0.5, true), 10_000);
        assert_eq!(s.desk.readings.master[0].db, f32::NEG_INFINITY);
        assert!(!s.desk.limiting(10_000));
        assert_eq!(s.desk.view(StripKind::Player).unwrap().id, 3);
        // A fresh one takes them.
        feed(&mut s, look(0.5, true), 10_016);
        assert!(s.desk.readings.master[0].db > -7.0);
        assert!(s.desk.meters(StripKind::Preview)[0].db > -13.0);
        assert!(s.desk.limiting(10_016));
    }
}
