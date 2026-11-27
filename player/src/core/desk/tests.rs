//! The desk's tests: each control's change, the keys on the tab and off
//! it, the strips and how each is heard.

use super::*;
use crate::config::Config;
use crate::core::{update, Action};
use jodel_output::mixer::{AutomationView, Timebase};

/// `config`'s state on the Mixer tab of a 960 × 684 window (the old
/// 120 × 36 cells).
fn tab_with(config: Config) -> State {
    let mut s = State::new("/m".into(), config);
    s.fit = crate::gui::metrics::fit(iced::Size::new(960.0, 684.0), &s, 8.0);
    s.side = Side::Mixer;
    s
}

fn tab() -> State {
    tab_with(Config::default())
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
    // The master steps the player's own volume, never above unity; the
    // microphone has no fader.
    update(&mut s, Action::Op(Op::DeskNextStrip));
    assert!(desk_fx(&update(&mut s, Action::Op(Op::Up))).is_empty());
    update(&mut s, Action::Op(Op::DeskNextStrip));
    assert!(desk_fx(&update(&mut s, Action::Op(Op::Up))).is_empty());
    let fx = update(&mut s, Action::Op(Op::Down));
    assert_eq!(desk_fx(&fx), [DeskChange::Volume(taper::gain(-1.0))]);
    assert_eq!(s.desk.volume, -1.0);
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
            DeskChange::Volume(taper::gain(-1.0)),
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
    assert_eq!(all.len(), 22);
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
    let mut s = tab_with(config);
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

/// The tab stands still (the window may rest) until something on it
/// moves: a meter lit, the LIMIT lamp, a head's automation playing, the
/// microphone's strip listening; off the tab nothing of it moves.
#[test]
fn the_tab_moves_only_while_something_on_it_does() {
    let mut s = tab();
    assert!(!moving(&s, 0), "nothing looked at yet");
    // Looks of silence: the meters stay dark.
    feed(&mut s, look(0.0, false), 1_000);
    feed(&mut s, look(0.0, false), 1_250);
    assert!(!moving(&s, 1_250));
    // A peak: the meters light, and go dark again in a few seconds.
    feed(&mut s, look(0.5, false), 1_500);
    assert!(moving(&s, 1_500));
    let mut at = 1_500;
    while moving(&s, at) {
        at += 250;
        feed(&mut s, look(0.0, false), at);
        assert!(at < 10_000, "the meters never went dark");
    }
    // The limiter: its lamp, a while.
    at += 250;
    feed(&mut s, look(0.0, true), at);
    assert!(moving(&s, at));
    assert!(!moving(&s, at + LIMIT_HOLD_MS));
    // A head's automation playing, until it settles.
    let late = at + LIMIT_HOLD_MS;
    let automation = |settled| AutomationView {
        input: 1.0,
        main: 0.5,
        parts: Vec::new(),
        settled,
    };
    let mut player = look_with_stems().strips[0].clone().unwrap().view;
    player.inputs[0].automation = Some(automation(false));
    s.desk.views[StripKind::Player.index()] = Some(player.clone());
    assert!(moving(&s, late));
    player.inputs[0].automation = Some(automation(true));
    s.desk.views[StripKind::Player.index()] = Some(player);
    assert!(!moving(&s, late));
    // The microphone's strip listening, its meters moving with the room.
    s.mic = Some(crate::core::mic::MicReading::default());
    assert!(!moving(&s, late), "the microphone's strip not selected");
    s.desk.selected = Channel::Mic;
    assert!(moving(&s, late));
    // Away from the tab, nothing of it moves.
    s.side = Side::Browser;
    assert!(!moving(&s, late));
}
