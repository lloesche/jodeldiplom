use super::*;
use crate::core::shine::{Shone, Tier};
use crate::gui::fx::{Board, Effects, Music, Use};
use crate::gui::particles::sprite;
use crate::gui::rainbow::Glow;
use crate::gui::stage::{over_at, Lane};
use crate::gui::streak::Step;

/// The stage the tests draw over.
const AREA: (f32, f32, f32, f32) = (0.0, 0.0, 1600.0, 900.0);

static SLAMS: [Use; 1] = [Use::of(Fx::Slam, 1.0)];

/// A frame played: its time, the words it letters, the shake.
type Played = (f64, Vec<Letters>, [f32; 2]);

/// The tiers' rows, the top's phrase and rung firing the slam.
fn rows() -> Vec<(f32, Effects)> {
    let top = Tier::TOP.index();
    (TIERS.iter().enumerate())
        .map(|(i, t)| {
            let fx = if i == top {
                Effects {
                    on_phrase: &SLAMS,
                    on_rung: &SLAMS,
                    ..Effects::NONE
                }
            } else {
                Effects::NONE
            };
            (t.glow, fx)
        })
        .collect()
}

/// The singing at the top, a phrase reaching it decided at `t` (or none).
fn step(phrase: Option<f64>) -> Step {
    Step {
        tier: Some(Tier::TOP),
        glow: Glow {
            level: 1.0,
            from: 1.0,
            since: -10.0,
        },
        phrase: phrase.map(|end| Shone {
            line: 0,
            end,
            value: 1.0,
            top: 0.9,
            tier: Some(Tier::TOP),
        }),
        ..Step::default()
    }
}

/// The top's rung reached with `step`.
fn rung(step: Step) -> Step {
    let below = Tier::ALL.iter().rev().nth(1).copied();
    Step {
        at_rung: Some(Tier::TOP),
        rung: Some((below, Some(Tier::TOP))),
        ..step
    }
}

/// What the slam draws over everything at `t`, `step` and `music` then:
/// its words, the shake.
fn over(board: &mut Board, step: &Step, music: &Music, t: f64) -> (Vec<Letters>, [f32; 2]) {
    let rows = rows();
    let frame = board.frame_of(rows.iter().map(|(g, e)| (*g, e)), step, music, t);
    let lane = Lane::new(AREA, t).glass(AREA);
    let (mut shapes, mut sprays, mut letters, mut shake) =
        (Vec::new(), Vec::new(), Vec::new(), [0.0; 2]);
    let mut out = Out {
        shapes: &mut shapes,
        sprays: &mut sprays,
        letters: &mut letters,
        shake: &mut shake,
    };
    board.draw(&over_at(AREA, lane), &frame, &mut out);
    assert!(shapes.is_empty() && sprays.is_empty());
    (letters, shake)
}

/// Fired at `fire` by `step`, with `music`: the frames at 60 a second
/// after it for ten seconds, each its time and what it drew.
fn played(step_fired: Step, music: Music, fire: f64) -> Vec<Played> {
    let mut board = Board::default();
    let (l, k) = over(&mut board, &step_fired, &music, fire);
    let mut frames = vec![(fire, l, k)];
    for f in 1..600 {
        let t = fire + f64::from(f) / 60.0;
        let (l, k) = over(&mut board, &step(None), &music, t);
        frames.push((t, l, k));
    }
    frames
}

/// The dust thrown in `frames`, with the frame it is thrown in.
fn dust(frames: &[Played]) -> Vec<(f64, Spray)> {
    (frames.iter())
        .flat_map(|f| f.1.iter().filter_map(|l| l.burst).map(move |b| (f.0, b)))
        .collect()
}

/// The first frame it shows in, and the frame its dust is thrown in.
fn shown_and_landed(frames: &[Played]) -> (f64, f64) {
    let shown = frames.iter().find(|f| !f.1.is_empty()).expect("it shows").0;
    (shown, dust(frames).first().expect("it lands").0)
}

#[test]
fn it_falls_lands_squashed_springs_back_and_fades() {
    assert_eq!(pose(-DROP - 0.01, DROP), None);
    let start = pose(-DROP + 1e-4, DROP).unwrap();
    assert!(
        start.scale.0 > FROM - 0.01 && start.alpha < SHOWS,
        "{start:?}"
    );
    let mid = pose(-DROP / 2.0, DROP).unwrap();
    assert!(mid.scale.0 < start.scale.0 && mid.alpha > start.alpha);
    let landed = pose(0.0, DROP).unwrap();
    assert!(
        landed.scale.0 > 1.2 && landed.scale.1 < 0.75,
        "squashed: {landed:?}"
    );
    // Springing: taller than it is for a moment, then still.
    let tall = (1..40)
        .filter_map(|k| pose(k as f32 * 0.01, DROP))
        .any(|p| p.scale.1 > 1.05);
    assert!(tall);
    let rest = pose(0.6, DROP).unwrap();
    assert!((rest.scale.0 - 1.0).abs() < 0.01 && (rest.scale.1 - 1.0).abs() < 0.01);
    assert!((rest.alpha - REST).abs() < 0.01, "{rest:?}");
    // The shine sweeps over it once, after it lands.
    assert!(!(0.0..=1.0).contains(&landed.shine));
    assert!((0.0..=1.0).contains(&pose(0.3, DROP).unwrap().shine));
    // Rising, fading, gone.
    let fading = pose(HOLD + FADE * 0.7, DROP).unwrap();
    assert!(fading.alpha < rest.alpha * 0.5 && fading.lift > 0.0);
    assert_eq!(pose(HOLD + FADE, DROP), None);
}

#[test]
fn the_screen_shakes_as_it_lands_and_only_then() {
    assert_eq!(shake(-0.01, 900.0), [0.0; 2]);
    let blow = shake(0.0, 900.0);
    assert!(
        (blow[1] - SHAKE * 900.0).abs() < 1e-3,
        "down at once: {blow:?}"
    );
    let most = (0..400)
        .map(|k| shake(k as f32 * 0.001, 900.0))
        .map(|s| s[0].hypot(s[1]))
        .fold(0.0, f32::max);
    assert!(most <= SHAKE * 900.0 * 1.08, "{most}");
    let later = shake(0.2, 900.0);
    assert!(later[0].hypot(later[1]) < 0.1 * SHAKE * 900.0);
    assert_eq!(shake(SHAKE_FOR, 900.0), [0.0; 2]);
}

/// A phrase reaching a row shouts its tier's name; its rung reached, the
/// multiplier; any row, none named here.
#[test]
fn it_shouts_what_fired_it() {
    for (i, t) in TIERS.iter().enumerate() {
        let name = word(By::Phrase(i)).unwrap();
        assert_eq!(name, format!("{}!", t.name.to_uppercase()));
        let times = format!("×{}", Tier::ALL[i].multiplier());
        assert_eq!(word(By::Rung(i)), Some(times));
    }
    assert_eq!(word(By::Phrase(TIERS.len())), None);
    assert_eq!(word(By::Rung(TIERS.len())), None);
    let name = format!("{}!", Tier::TOP.spec().name.to_uppercase());
    let said = |frames: &[Played]| frames.iter().find_map(|f| f.1.first().cloned()).unwrap();
    let by_phrase = said(&played(step(Some(5.0)), Music::NEUTRAL, 5.0));
    assert_eq!(
        (by_phrase.text.as_str(), by_phrase.shape.params[0]),
        (name.as_str(), SLAM)
    );
    let by_rung = said(&played(rung(step(None)), Music::NEUTRAL, 5.0));
    assert_eq!(by_rung.text, format!("×{}", Tier::TOP.multiplier()));
    // Both in one frame: the phrase's shout; the rung's let go (once).
    let both = played(rung(step(Some(5.0))), Music::NEUTRAL, 5.0);
    assert_eq!(said(&both).text, name);
    assert!(both.iter().flat_map(|f| &f.1).all(|l| l.text == name));
    assert_eq!(dust(&both).len(), 1);
}

/// A phrase's slam: where the grid is sure, on the next bar's first beat
/// (however far), falling in just before; unsure, at once (on the frame it
/// fires, landing). Its dust once, as it lands; the screen shaken then.
#[test]
fn it_lands_on_the_next_downbeat_when_the_grid_is_sure() {
    let fire = 5.0;
    let frame = 1.0 / 60.0;
    // Two and a half beats into a bar of four at 120: the next first in
    // 0.75 s.
    let near = played(
        step(Some(fire)),
        Music::steady(10.5, 120.0, 4, 1.0, fire),
        fire,
    );
    let (shown, landed) = shown_and_landed(&near);
    assert!((landed - (fire + 0.75)).abs() < frame + 1e-9, "{landed}");
    let fall = fire + 0.75 - f64::from(DROP);
    assert!(shown >= fall && shown - fall < 2.0 * frame, "{shown}");
    let thrown = dust(&near);
    assert_eq!(thrown.len(), 1);
    assert!((f64::from(thrown[0].1.start) - (fire + 0.75)).abs() < 1e-4);
    assert_eq!(thrown[0].1.kind, Kind::SlamDust);
    let shaken: Vec<f64> = (near.iter())
        .filter(|f| f.2 != [0.0; 2])
        .map(|f| f.0)
        .collect();
    assert!(!shaken.is_empty() && shaken.iter().all(|&t| t >= fire + 0.75 - 1e-6));
    assert!(shaken
        .iter()
        .all(|&t| t < fire + 0.75 + f64::from(SHAKE_FOR) + 1e-3));
    // Gone a while after it landed.
    let last = near.iter().rev().find(|f| !f.1.is_empty()).unwrap().0;
    assert!(last - landed < f64::from(HOLD + FADE) + 0.8);
    assert!(near.last().unwrap().1.is_empty());
    // A beat into the bar: the next first 1.5 s off, still waited for.
    let far = played(
        step(Some(fire)),
        Music::steady(9.0, 120.0, 4, 1.0, fire),
        fire,
    );
    let (_, landed) = shown_and_landed(&far);
    assert!((landed - (fire + 1.5)).abs() < frame + 1e-9, "{landed}");
    // Unsure: at once, landing on the frame it fires.
    let unsure = played(
        step(Some(fire)),
        Music::steady(10.5, 120.0, 4, 0.2, fire),
        fire,
    );
    assert_eq!(shown_and_landed(&unsure), (fire, fire));
    let at_once = &unsure[0].1[0];
    assert!(at_once.scale.1 < at_once.scale.0, "squashed: {at_once:?}");
}

/// Where a slam lands is where the grid's one rule lands its sound
/// ([`jodel_rhythm::grid::Grid::land`]) for the same song, the same lines
/// sung and the same [`When`]: a phrase's on a bar's first beat with room
/// for the jackpot before a sung line, a rung's on the next beat.
#[test]
fn it_lands_where_its_sound_does() {
    let (tempo, late) = (100.0, 0.3);
    let song = crate::gui::fx::music::steady_song(80, tempo, 4, 0.9);
    let beat = 60.0 / tempo;
    let top = Tier::TOP.index();
    for (beats, by) in [
        (9.7, By::Phrase(top)),
        (13.2, By::Phrase(top)),
        (10.8, By::Rung(top)),
        (10.2, By::Rung(top)),
    ] {
        let now = beats * beat;
        let fire = now - late;
        let mut music = Music::steady(beats, tempo, 4, 0.9, fire);
        // A line sung a few beats on.
        let line = (fire + 4.0 * beat, fire + 7.0 * beat);
        music.lines[0] = Some(line);
        let step = match by {
            By::Phrase(_) => step(Some(fire)),
            By::Rung(_) => rung(step(None)),
        };
        let thrown = dust(&played(step, music, fire));
        let grid = song.grid_at(now).on(now, fire, 1.0);
        let lands = grid.land(fire, when(by, &music), &[line]);
        assert!(
            (f64::from(thrown[0].1.start) - lands).abs() < 1e-4,
            "{by:?} at {now}: {} {lands}",
            thrown[0].1.start
        );
    }
}

/// Sought back before it fired: gone, its dust not thrown.
#[test]
fn a_seek_back_takes_it_away() {
    let mut board = Board::default();
    let music = Music::steady(10.5, 120.0, 4, 1.0, 5.0);
    over(&mut board, &step(Some(5.0)), &music, 5.0);
    let (letters, _) = over(&mut board, &step(None), &music, 3.0);
    assert!(letters.is_empty());
    let (letters, _) = over(&mut board, &step(None), &music, 5.9);
    assert!(letters.is_empty());
}

/// The word never covers what is sung: its foot above the lane's glass
/// (so above every tube there and the word riding in it, however high its
/// note) and the lyric band under that, and it grows only up and out from
/// its foot; it stays on the stage, clear of the multiplier over the
/// playhead.
#[test]
fn it_lands_clear_of_the_tubes_and_the_lyrics() {
    for area in [AREA, (0.0, 0.0, 3024.0, 1964.0), (0.0, 0.0, 800.0, 1200.0)] {
        let lane = Lane::new(area, 0.0);
        let glass = lane.glass(area);
        let At::Over { band, .. } = over_at(area, glass) else {
            panic!("over");
        };
        let ((x, foot), px) = place(area, glass);
        // The highest a tube's word box can reach: its note at the top.
        let tubes = lane.y(1.0) - lane.pill / 2.0;
        assert!(foot <= glass.middle.1 - glass.half.1, "{area:?}");
        assert!(foot < tubes && foot < band.0, "{area:?}: {foot} {tubes}");
        assert!(foot - px * 1.4 > area.1, "{area:?}: on the stage at rest");
        assert_eq!(x, area.0 + area.2 / 2.0);
        assert!(x - area.2 * WIDEST / 2.0 > lane.head, "{area:?}");
    }
}

/// The dust: puffs rolling out either way along the floor, sparkles
/// thrown up, all gone by its end.
#[test]
fn the_dust_rolls_out_and_the_sparkles_fly_up() {
    let s = slam_dust((800.0, 500.0), 300.0, 20.0, 10.0, 7, GOLD);
    let at = |t: f32| -> Vec<Sprite> {
        (0..crate::gui::particles::QUADS)
            .filter_map(|q| sprite(&s, q, 10.0 + t))
            .collect()
    };
    let early = at(0.05);
    assert_eq!(
        early.iter().filter(|p| p.look == Look::Puff).count(),
        PUFFS as usize
    );
    assert!(early.iter().filter(|p| p.look == Look::Sparkle).count() >= SPARKS as usize - 2);
    let late = at(0.5);
    let puffs: Vec<&Sprite> = late.iter().filter(|p| p.look == Look::Puff).collect();
    let left = puffs.iter().map(|p| p.at.0).fold(f32::MAX, f32::min);
    let right = puffs.iter().map(|p| p.at.0).fold(f32::MIN, f32::max);
    assert!(
        left < 800.0 - 150.0 && right > 800.0 + 150.0,
        "{left} {right}"
    );
    assert!(puffs
        .iter()
        .all(|p| p.at.1 <= 500.0 && p.at.1 > 500.0 - 20.0 * 2.0));
    let up = (0..crate::gui::particles::QUADS)
        .filter_map(|q| sprite(&s, q, 10.15))
        .filter(|p| p.look == Look::Sparkle)
        .filter(|p| p.at.1 < 500.0 - 20.0)
        .count();
    assert!(up > 5, "{up}");
    assert!(at(DUST_LIFE + 0.01).is_empty());
}

/// The ink: its face lit from its top, outlined dark round it, none shown
/// at none, the shine whitening as it sweeps.
#[test]
fn the_ink_is_lit_and_outlined() {
    let colours = (GOLD, GOLD.map(|v| v * 0.16));
    let face_top = ink([1.0, 1.0], (0.5, 0.05), (1.0, -1.0), colours);
    let face_low = ink([1.0, 1.0], (0.5, 0.95), (1.0, -1.0), colours);
    assert!(face_top[2] > face_low[2] && face_top[3] == 1.0);
    let rim = ink([0.0, 1.0], (0.5, 0.5), (1.0, -1.0), colours);
    assert_eq!(&rim[..3], &colours.1[..]);
    assert_eq!(ink([0.0, 0.0], (1.1, 0.5), (1.0, -1.0), colours)[3], 0.0);
    assert_eq!(ink([1.0, 1.0], (0.5, 0.5), (0.0, 0.3), colours)[3], 0.0);
    let shine = ink([1.0, 1.0], (0.3, 0.5), (1.0, 0.5), colours);
    let plain = ink([1.0, 1.0], (0.3, 0.5), (1.0, -1.0), colours);
    assert!(shine[2] > plain[2] + 0.2, "{shine:?} {plain:?}");
}

/// The taps: a texel's own coverage at its middle, between texels the
/// straight line; nothing read outside the word's own texels (another
/// word beside it in the atlas); the outline the most within its reach;
/// inside a letter, all of it.
#[test]
fn the_taps_read_only_the_word_s_own_texels() {
    let texel = |x: i32, y: i32| -> f32 {
        let (x, y) = (x.clamp(0, 31), y.clamp(0, 15));
        if x >= 24 {
            0.8 // Another word.
        } else if (8..12).contains(&x) {
            1.0
        } else {
            (x + y) as f32 / 64.0
        }
    };
    let rect = [0.0, 0.0, 24.0, 16.0];
    assert_eq!(cover(&texel, (3.5, 4.5), rect, 2.0)[0], 7.0 / 64.0);
    let between = cover(&texel, (4.0, 4.5), rect, 2.0)[0];
    assert!((between - 7.5 / 64.0).abs() < 1e-6, "{between}");
    assert_eq!(cover(&texel, (9.5, 4.5), rect, 2.0), [1.0, 1.0]);
    assert_eq!(cover(&texel, (26.0, 4.5), rect, 2.0)[0], 0.0, "outside");
    let near = cover(&texel, (6.5, 4.5), rect, 2.0);
    assert!(near[1] >= 0.99 && near[0] < 0.2, "{near:?}");
}
