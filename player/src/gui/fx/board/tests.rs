use super::*;
use crate::core::shine::Shone;
use crate::gui::fx::{Mod, Signal};
use crate::tiers::Tier;

fn step(tier: Option<Tier>, glow: Glow) -> Step {
    Step {
        tier,
        glow,
        ..Step::default()
    }
}

fn held_at(level: f32) -> Glow {
    Glow {
        level,
        from: level,
        since: -10.0,
    }
}

/// At each row's glow an effect is that row's strength, bit for bit;
/// between two rows the straight line; none below the first row that has
/// it.
#[test]
fn a_held_strength_is_its_row_s_at_the_row_and_eases_between() {
    let mut board = Board::default();
    for fx in Fx::ALL {
        for (t, row) in Tier::ALL.into_iter().zip(&TIERS) {
            let f = board.frame(&step(Some(t), held_at(row.glow)), &Music::NEUTRAL, 1.0);
            let want = strength(row.fx.held, fx);
            let live = f.live(fx);
            assert_eq!(live.strength.to_bits(), want.to_bits(), "{fx:?} {t:?}");
            assert_eq!(live.aim.to_bits(), want.to_bits(), "{fx:?} {t:?}");
            assert_eq!(live.listed.to_bits(), want.to_bits(), "{fx:?} {t:?}");
        }
        let none = board.frame(&step(None, held_at(0.0)), &Music::NEUTRAL, 1.0);
        assert_eq!(none.live(fx).strength, 0.0, "{fx:?}");
    }
    let (a, b) = (&TIERS[1], &TIERS[2]);
    let mid = (a.glow + b.glow) / 2.0;
    let f = board.frame(&step(None, held_at(mid)), &Music::NEUTRAL, 1.0);
    let want = (strength(a.fx.held, Fx::Flames) + strength(b.fx.held, Fx::Flames)) / 2.0;
    assert!((f.live(Fx::Flames).strength - want).abs() < 1e-6);
}

/// The lane's glow and its flames, as the glow eases through and between
/// the rows, bit for bit what they were when each row held them itself
/// (Great's glow round the lane, Amazing's flames).
#[test]
fn the_lane_glow_and_flames_ramp_as_before() {
    let before = |k: f32, values: [f32; 4]| {
        let mut a = (0.0f32, 0.0f32);
        for (row, v) in TIERS.iter().zip(values) {
            let b = (row.glow, v);
            if k < b.0 {
                return a.1 + (b.1 - a.1) * ((k - a.0) / (b.0 - a.0));
            }
            a = b;
        }
        a.1
    };
    let mut board = Board::default();
    for i in 0..=2000 {
        let k = i as f32 / 1600.0;
        let f = board.frame(&step(None, held_at(k)), &Music::NEUTRAL, 1.0);
        let glow = before(k, [0.0, 1.0, 1.0, 1.0]);
        let flames = before(k, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(f.live(Fx::Glow).strength.to_bits(), glow.to_bits(), "{k}");
        let got = f.live(Fx::Flames).strength;
        assert_eq!(got.to_bits(), flames.to_bits(), "{k}");
    }
}

/// The rainbow eases by its own glow: while the glow rises from none or
/// collapses, it is as strong as the row it comes from or goes to lists
/// it, not the held ramp below the first row.
#[test]
fn the_rainbow_is_as_strong_as_the_row_its_glow_eases_by() {
    let mut board = Board::default();
    let rising = Glow {
        level: TIERS[0].glow,
        from: 0.0,
        since: 1.0,
    };
    let lost = Glow {
        level: 0.0,
        from: TIERS[2].glow,
        since: 1.0,
    };
    for (g, clock) in [(rising, 1.05), (lost, 1.4)] {
        let f = board.frame(&step(None, g), &Music::NEUTRAL, clock);
        let live = f.live(Fx::Rainbow);
        assert!(live.strength < 1.0, "{live:?}");
        assert_eq!(live.listed, 1.0);
    }
}

/// A phrase at the top fires its row's jackpot and a step up (the tiers it
/// rose with it) at their strength, once; the next frame nothing; a seek
/// with it (no phrase given) nothing.
#[test]
fn a_phrase_fires_its_row_s_effects_once() {
    let mut board = Board::default();
    let shone = Shone {
        line: 3,
        end: 2.0,
        value: 1.0,
        top: 0.9,
        tier: Some(Tier::TOP),
    };
    let diva = Step {
        phrase: Some(shone),
        ups: 4,
        ..step(Some(Tier::TOP), held_at(1.0))
    };
    let f = board.frame(&diva, &Music::NEUTRAL, 2.0);
    for fx in [Fx::Jackpot, Fx::StepUp] {
        let live = f.live(fx);
        assert!(live.fired && live.strength == 1.0, "{fx:?} {live:?}");
    }
    assert_eq!(f.live(Fx::StepUp).rose, 4);
    assert!(!f.live(Fx::Angels).fired);
    let next = board.frame(&step(Some(Tier::TOP), held_at(1.0)), &Music::NEUTRAL, 2.02);
    assert!(Fx::ALL.iter().all(|&fx| !next.live(fx).fired));
    assert_eq!(next.live(Fx::StepUp).rose, 0);
    let sought = board.frame(&step(Some(Tier::TOP), held_at(1.0)), &Music::NEUTRAL, 1.0);
    assert!(Fx::ALL.iter().all(|&fx| !sought.live(fx).fired));
}

/// No row moves the strength of what it fires: it fires as listed.
#[test]
fn no_row_moves_what_it_fires() {
    for t in &TIERS {
        for u in t.fx.on_phrase.iter().chain(t.fx.on_rung) {
            let moved = u.mods.iter().any(|m| m.param == Param::Strength);
            assert!(!moved, "{} {:?}", t.name, u.fx);
        }
    }
}

static FIRE: [Use; 1] = [Use::of(Fx::EdgeFire, 0.6)];
static RAIN: [Use; 1] = [Use::of(Fx::Rain, 1.0)];
static RUNGS: Effects = Effects {
    on_rung: &RAIN,
    at_rung: &FIRE,
    ..Effects::NONE
};

/// Rows with rung effects: reaching a rung fires its row's and eases in
/// what it holds; losing it eases that out; a seek to another rung snaps
/// there and fires nothing.
#[test]
fn rung_effects_are_held_and_fired_by_rung() {
    let rows = [(0.5, &Effects::NONE), (1.0, &RUNGS)];
    let mut board = Board::default();
    let at = |rung: u32, change: Option<(u32, u32)>| Step {
        at_rung: rung,
        rung: change,
        ..Step::default()
    };
    let m = &Music::NEUTRAL;
    let mut frame = |s: &Step, clock: f64| board.frame_of(rows.into_iter(), s, m, clock);
    let f = frame(&at(1, Some((0, 1))), 1.0);
    let fire = f.live(Fx::EdgeFire).strength;
    assert_eq!(fire, 0.0, "the first rung has none");
    let up = frame(&at(2, Some((1, 2))), 2.0);
    assert!(up.live(Fx::Rain).fired);
    assert_eq!(up.live(Fx::EdgeFire).aim, 0.6);
    assert_eq!(up.live(Fx::EdgeFire).strength, 0.0, "easing in");
    let held = frame(&at(2, None), 3.0);
    assert_eq!(held.live(Fx::EdgeFire).strength, 0.6);
    assert!(!held.live(Fx::Rain).fired);
    let lost = frame(&at(0, Some((2, 0))), 4.0);
    assert!(!lost.live(Fx::Rain).fired, "nothing fires going down");
    assert_eq!(lost.live(Fx::EdgeFire).aim, 0.0);
    let collapse = f64::from(crate::gui::rainbow::COLLAPSE);
    let half = frame(&at(0, None), 4.0 + collapse / 2.0);
    assert!((half.live(Fx::EdgeFire).strength - 0.3).abs() < 1e-6);
    let sought = frame(&at(2, None), 9.0);
    assert_eq!(sought.live(Fx::EdgeFire).strength, 0.6, "snapped");
    assert!(!sought.live(Fx::Rain).fired);
    assert_eq!(sought.live(Fx::Bulbs).strength, 0.0);
}

static BEATING: [Mod; 2] = [
    Mod {
        param: Param::Strength,
        by: Signal::Beat,
        amount: 0.5,
    },
    Mod {
        param: Param::Look(1),
        by: Signal::Stem(jodel_rhythm::stems::Stem::Bass),
        amount: 2.0,
    },
];
static MOVED: [Use; 1] = [Use {
    fx: Fx::Bulbs,
    strength: 0.8,
    mods: &BEATING,
}];
static MOVING: Effects = Effects {
    held: &MOVED,
    ..Effects::NONE
};

/// A row's mods move its effect with the music: its strength by as much
/// again as the beat, its look's number by the bass; still music moves
/// nothing.
#[test]
fn the_music_moves_what_a_row_lists() {
    let mut board = Board::default();
    let s = step(Some(Tier::ALL[0]), held_at(1.0));
    let rows = [(1.0, &MOVING)];
    let still = board.frame_of(rows.into_iter(), &s, &Music::NEUTRAL, 1.0);
    assert_eq!(still.live(Fx::Bulbs).strength, 0.8);
    assert_eq!(still.looks(), Looks::SPECS);
    let mut music = Music {
        beat: 1.0,
        sure: 1.0,
        ..Music::NEUTRAL
    };
    music.stems[jodel_rhythm::stems::Stem::Bass as usize] = 0.25;
    let moved = board.frame_of(rows.into_iter(), &s, &music, 1.0);
    let live = moved.live(Fx::Bulbs);
    assert!((live.strength - 1.2).abs() < 1e-6, "{live:?}");
    let look = moved.looks().0[Fx::Bulbs.index()];
    assert!((look[1] - (Fx::Bulbs.spec().params[1] + 0.5)).abs() < 1e-6);
}
