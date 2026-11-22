//! The Mix tab ([`crate::core::Side::Mix`]): the mixer as a desk. A channel
//! strip for the player and one for the preview, each with a fader, mute
//! and solo; the microphone's strip, which is only listened to; the master
//! with the device. Every strip starts at 0 dB and nothing is kept between
//! runs: the desk adds no gain until a fader is moved.
//!
//! The settings live here and go out as [`Effect::Desk`]; the app sets its
//! strips by them, and again on every new mixer ([`changes`]). What the
//! desk shows comes in once a drawn frame ([`feed`]).

pub mod meter;
pub mod mouse;
pub mod taper;

use jodel_output::mixer::{DeskView, MasterView, Meters, MicView, StripView};

use crate::core::keymap::Op;
use crate::core::update::say_info;
use crate::core::{Effect, Screen, Side, State, VolumeChange, DOUBLE_CLICK_MS};

pub use meter::{Meter, Pair};

/// A channel strip with a fader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StripKind {
    /// The song's player.
    Player,
    /// The preview's players.
    Preview,
}

impl StripKind {
    /// Both, left to right.
    pub const ALL: [StripKind; 2] = [StripKind::Player, StripKind::Preview];

    /// The name the app gives the strip on the mixer.
    pub fn name(self) -> &'static str {
        match self {
            StripKind::Player => "Player",
            StripKind::Preview => "Preview",
        }
    }

    /// Where it is in arrays of both ([`StripKind::ALL`]'s order).
    pub fn index(self) -> usize {
        match self {
            StripKind::Player => 0,
            StripKind::Preview => 1,
        }
    }
}

/// A strip of the desk, as the arrow keys go through them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Channel {
    /// A strip with a fader.
    Strip(StripKind),
    /// The microphone's strip: listened to, never played.
    Mic,
    /// The master: what goes to the device.
    Main,
}

impl Channel {
    /// Left to right.
    pub const ALL: [Channel; 4] = [
        Channel::Strip(StripKind::Player),
        Channel::Strip(StripKind::Preview),
        Channel::Mic,
        Channel::Main,
    ];
}

impl Default for Channel {
    /// The player's strip, as the tab opens.
    fn default() -> Self {
        Channel::Strip(StripKind::Player)
    }
}

/// A button on a strip.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeskButton {
    /// Mute.
    Mute(StripKind),
    /// Solo.
    Solo(StripKind),
    /// The microphone's echo canceller on or off.
    Echo,
}

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

/// A change for the app to make on a strip ([`Effect::Desk`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DeskChange {
    /// The level, a linear gain.
    Level(StripKind, f64),
    /// Muted or not.
    Mute(StripKind, bool),
    /// Soloed or not.
    Solo(StripKind, bool),
}

/// A fader held with the mouse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drag {
    /// Which.
    pub strip: StripKind,
    /// How far the cap's middle is below the pointer, in rows: a cap
    /// taken by its edge does not jump.
    pub offset: f64,
}

/// What the mouse does on the desk ([`crate::core::Action::Desk`]).
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
        /// See [`Drag::offset`].
        offset: f64,
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
    pub strips: [Option<StripLook>; 2],
    /// The master's peaks, one per channel (linear).
    pub peaks: Vec<f32>,
    /// The limiter acted since the last look.
    pub limited: bool,
}

impl Look {
    /// The look at `desk` and `meters`, each fader strip found by its
    /// mixer's id (`ids`, by [`StripKind::index`]).
    pub fn of(desk: DeskView, meters: Meters, ids: [u64; 2]) -> Look {
        let mut strips = [None, None];
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
    pub strips: [Pair; 2],
    /// The microphone as captured.
    pub mic_in: Meter,
    /// The singer isolated.
    pub mic_voice: Meter,
}

/// The Mix tab's state.
#[derive(Clone, Debug, Default)]
pub struct DeskState {
    /// The strip the keys act on.
    pub selected: Channel,
    /// What the user set on each fader strip ([`StripKind::index`]): this
    /// run only.
    pub strips: [StripSetting; 2],
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
    pub views: [Option<StripView>; 2],
    /// The meters.
    pub readings: Readings,
    /// When the limiter last acted.
    pub limited_at: Option<u64>,
    /// When the last look came.
    pub looked_at: u64,
}

impl DeskState {
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

    /// Whether the LIMIT lamp is lit at `now_ms`.
    pub fn limiting(&self, now_ms: u64) -> bool {
        self.limited_at
            .is_some_and(|at| now_ms.saturating_sub(at) < LIMIT_HOLD_MS)
    }

    fn setting_mut(&mut self, strip: StripKind) -> &mut StripSetting {
        &mut self.strips[strip.index()]
    }
}

/// Whether the Mix tab shows (a palette over it included).
pub fn shown(state: &State) -> bool {
    state.screen == Screen::Main && state.side == Side::Mix
}

/// Whether the Mix tab is what keys act on.
pub fn on_tab(state: &State) -> bool {
    shown(state) && state.palette.is_none()
}

/// Whether the Mix tab listens to the microphone: while it shows with the
/// microphone's strip selected.
pub fn listens(state: &State) -> bool {
    shown(state) && state.desk.selected == Channel::Mic
}

/// After any action: a fader held is let go of once the tab is left.
pub(crate) fn settle(state: &mut State) {
    if !shown(state) {
        state.desk.dragging = None;
    }
}

/// Every setting of the desk, for a new mixer: its strips start at 0 dB,
/// and the user's settings are made on them again.
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
    let [player, preview] = look.strips;
    desk.views = [player.map(|s| s.view), preview.map(|s| s.view)];
    desk.master = Some(look.master);
    desk.mic_port = look.mic;
    desk.looked_at = now_ms;
}

/// Carries out `op` on the Mix tab; `false` when it is not the tab's (or
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

/// What the master says to a key it has no use for.
const MAIN_SAID: &str = "The master adds no gain: ↑ ↓ move the device's own volume";

fn no_fader(state: &mut State) {
    say_info(
        state,
        "The microphone is only listened to: it has no fader and is never played",
    );
}

fn select_by(state: &mut State, by: isize) {
    let all = Channel::ALL;
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
    true
}

fn press(state: &mut State, button: DeskButton, fx: &mut Vec<Effect>) {
    match button {
        DeskButton::Mute(strip) => {
            let s = state.desk.setting_mut(strip);
            s.mute = !s.mute;
            fx.push(Effect::Desk(DeskChange::Mute(strip, s.mute)));
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
        DeskAction::Grab {
            strip,
            position,
            offset,
        } => {
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
            state.desk.dragging = Some(Drag { strip, offset });
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
        DeskAction::Release => state.desk.dragging = None,
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
    say_info(state, "That is for the Mix tab (tab)");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::core::{update, Action};
    use jodel_output::mixer::Timebase;

    fn tab() -> State {
        let mut s = State::new("/m".into(), Config::default(), (120, 36));
        s.side = Side::Mix;
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
        // The master steps the device's own volume; the microphone has no
        // fader.
        update(&mut s, Action::Op(Op::DeskNextStrip));
        assert!(desk_fx(&update(&mut s, Action::Op(Op::Up))).is_empty());
        update(&mut s, Action::Op(Op::DeskNextStrip));
        let fx = update(&mut s, Action::Op(Op::Up));
        assert!(fx.contains(&Effect::DeviceVolume(VolumeChange::Step(1))));
        assert!(desk_fx(&fx).is_empty());
        // Every setting again for a new mixer.
        assert_eq!(
            changes(&s),
            [
                DeskChange::Level(StripKind::Player, taper::gain(-1.0)),
                DeskChange::Mute(StripKind::Player, true),
                DeskChange::Solo(StripKind::Player, false),
                DeskChange::Level(StripKind::Preview, 1.0),
                DeskChange::Mute(StripKind::Preview, false),
                DeskChange::Solo(StripKind::Preview, true),
            ]
        );
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
                offset: 0.0,
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
        Look::of(desk, meters, [3, 7])
    }

    #[test]
    fn strips_are_found_by_id_and_a_stale_look_takes_no_peaks() {
        let l = look(0.5, true);
        assert_eq!(l.strips[0].as_ref().unwrap().view.id, 3);
        assert_eq!(l.strips[1].as_ref().unwrap().peaks, [0.25]);
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
