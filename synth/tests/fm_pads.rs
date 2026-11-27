//! FM and SUPERSAW, each rendered alone: in tune at every rate, bounded at
//! the extremes, the same from the same seed, FM's sidebands where
//! Chowning's Bessel functions put them, its bells ringing out and saying
//! so, its retriggers without a click, and the supersaw's spread centred
//! on the note.

mod common;

use common::measure::{biggest_jump, centroid, cents, magnitude, peak, pitch};
use common::solo;
use jodel_synth::dsp::{self, Rate};
use jodel_synth::engines::fm::{self, Algorithm, Fm, Freq, Operator};
use jodel_synth::engines::supersaw::{self, Supersaw};
use jodel_synth::{Engine, NoteOn, VoiceMod, BLOCK};

/// An operator held at full: at `freq`, `level`.
fn held(freq: Freq, level: f32) -> Operator {
    Operator {
        freq,
        detune: 0.0,
        level,
        attack: 0.0,
        decay: 0.0,
        sustain: 1.0,
        release: 0.0,
        velocity: 0.0,
    }
}

/// A carrier at `carrier` times the note under one modulator at
/// `modulator` times it, index `index`, both held.
fn pair(carrier: f32, modulator: f32, index: f32) -> fm::Params {
    let silent = held(Freq::Ratio(1.0), 0.0);
    fm::Params {
        algorithm: Algorithm::Stack,
        ops: [
            held(Freq::Ratio(carrier), 1.0),
            held(Freq::Ratio(modulator), index),
            silent,
            silent,
        ],
        feedback: 0.0,
        key_scale: 0.0,
    }
}

/// The Bessel function of the first kind J_n(x), by its series.
fn bessel(n: u32, x: f64) -> f64 {
    let mut term = (x / 2.0).powi(n as i32) / (1..=n).map(f64::from).product::<f64>();
    let mut sum = term;
    for m in 1..40 {
        term *= -(x / 2.0).powi(2) / (f64::from(m) * f64::from(m + n));
        sum += term;
    }
    sum
}

fn assert_in_tune(x: &[f32], rate: u32, note: f32, what: &str) {
    let got = pitch(&x[x.len() / 3..], rate as f32);
    let c = cents(got, dsp::hz(note));
    assert!(c.abs() < 5.0, "{what} at {rate} Hz, note {note}: {c} cents");
}

#[test]
fn fm_carriers_at_ratio_one_are_in_tune_at_every_rate() {
    for rate in [44_100, 48_000, 96_000] {
        let e = Fm::new(Rate::new(rate));
        for note in [45.0, 57.0, 69.0, 76.0] {
            let plain = solo(&e, &fm::Params::default(), rate, note, 0.3, 1, 0.0);
            assert_in_tune(&plain, rate, note, "a sine");
            let modulated = solo(&e, &pair(1.0, 1.0, 1.5), rate, note, 0.3, 1, 0.0);
            assert_in_tune(&modulated, rate, note, "1 : 1 at index 1.5");
            let piano = solo(&e, &fm::E_PIANO, rate, note, 0.3, 1, 0.0);
            assert_in_tune(&piano, rate, note, "E-PIANO");
        }
    }
}

#[test]
fn fm_sidebands_follow_the_bessel_functions() {
    // A carrier at 4 f under a modulator at f: its sidebands at (4 + k) f
    // weigh J_k(index) against the carrier's J_0(index).
    let e = Fm::new(Rate::new(48_000));
    let f = f64::from(dsp::hz(57.0));
    let ratio =
        |x: &[f32], k: f64| magnitude(x, (4.0 + k) * f, 48_000.0) / magnitude(x, 4.0 * f, 48_000.0);
    let mut last = 0.0;
    for index in [0.5f32, 1.0, 2.0] {
        let x = solo(&e, &pair(4.0, 1.0, index), 48_000, 57.0, 0.6, 1, 0.0);
        let x = &x[2_400..];
        let j0 = bessel(0, f64::from(index)).abs();
        for k in [1u32, 2] {
            let want = bessel(k, f64::from(index)).abs() / j0;
            for side in [1.0, -1.0] {
                let got = ratio(x, side * f64::from(k));
                assert!(
                    (got / want - 1.0).abs() < 0.03,
                    "index {index}, sideband {}: {got} against {want}",
                    side * f64::from(k)
                );
            }
        }
        // The deeper the index, the louder the first sidebands.
        let first = ratio(x, 1.0);
        assert!(first > last, "index {index}: {first} after {last}");
        last = first;
    }
    // At index 0, a pure sine.
    let x = solo(&e, &pair(4.0, 1.0, 0.0), 48_000, 57.0, 0.6, 1, 0.0);
    assert!(ratio(&x[2_400..], 1.0) < 1e-4);
    // The brightness macro deepens the index: 2 octaves of it double it.
    let deepened = solo(&e, &pair(4.0, 1.0, 0.5), 48_000, 57.0, 0.6, 1, 2.0);
    let want = bessel(1, 1.0) / bessel(0, 1.0);
    let got = ratio(&deepened[2_400..], 1.0);
    assert!((got / want - 1.0).abs() < 0.03, "{got} against {want}");
}

#[test]
fn fm_feedback_brightens_operator_four() {
    let e = Fm::new(Rate::new(48_000));
    let alone = |feedback: f32| {
        let silent = held(Freq::Ratio(1.0), 0.0);
        let p = fm::Params {
            algorithm: Algorithm::Parallel,
            ops: [silent, silent, silent, held(Freq::Ratio(1.0), 1.0)],
            feedback,
            key_scale: 0.0,
        };
        centroid(&solo(&e, &p, 48_000, 57.0, 0.2, 1, 0.0), 48_000.0)
    };
    let (sine, edged) = (alone(0.0), alone(1.2));
    assert!(sine < 300.0 && edged > 2.0 * sine, "{sine} {edged}");
}

#[test]
fn fm_modulators_fade_with_the_notes_envelope() {
    // A modulator with a release of 1 thins its index as the note's
    // envelope falls (the host hands it over as it asks for the amplitude).
    let rate = Rate::new(48_000);
    let e = Fm::new(rate);
    let p = fm::Params {
        ops: [
            held(Freq::Ratio(1.0), 1.0),
            Operator {
                release: 1.0,
                ..held(Freq::Ratio(1.0), 3.0)
            },
            held(Freq::Ratio(1.0), 0.0),
            held(Freq::Ratio(1.0), 0.0),
        ],
        ..fm::Params::default()
    };
    let at = |adsr: f32| {
        let mut v = fm::Voice::default();
        e.note_on(&p, &mut v, &NoteOn::default());
        let m = VoiceMod {
            phase_inc: rate.inc(57.0),
            pitch: 57.0,
            ..VoiceMod::default()
        };
        let mut out = vec![0.0; 100 * BLOCK];
        for block in out.chunks_exact_mut(BLOCK) {
            assert_eq!(e.amp(&p, &mut v, adsr), adsr);
            e.render(&p, &mut v, &m, block);
        }
        centroid(&out, 48_000.0)
    };
    let (held_down, let_go) = (at(1.0), at(0.05));
    assert!(
        let_go < 300.0 && held_down > 2.0 * let_go,
        "{held_down} {let_go}"
    );
}

#[test]
fn fm_stays_bounded_at_the_extremes() {
    let e = Fm::new(Rate::new(48_000));
    for algorithm in fm::ALGORITHMS {
        for corner in 0..16u32 {
            let pick = |bit: u32| (corner >> bit) & 1 == 1;
            let freq = |k: usize| match (pick(0), k) {
                (true, 3) => Freq::Hz(20_000.0),
                (true, _) => Freq::Ratio(20.0),
                (false, _) => Freq::Ratio(0.5 + k as f32),
            };
            let level = if pick(1) { 20.0 } else { 1.0 };
            let p = fm::Params {
                algorithm,
                ops: std::array::from_fn(|k| Operator {
                    attack: if pick(2) { 0.01 } else { 0.0 },
                    release: 2.0,
                    velocity: 1.0,
                    ..held(freq(k), level)
                }),
                feedback: if pick(3) { 4.0 } else { 0.0 },
                key_scale: 0.5,
            };
            for (note, cutoff) in [(24.0, -4.0), (108.0, 4.0), (60.0, 0.0)] {
                let x = solo(&e, &p, 48_000, note, 0.1, 7, cutoff);
                assert!(
                    x.iter().all(|s| s.is_finite() && s.abs() <= 1.0),
                    "{algorithm:?} {corner} {note}: {}",
                    peak(&x)
                );
            }
        }
    }
    // An operator above the band is left out.
    let p = fm::Params {
        ops: [held(Freq::Hz(23_000.0), 1.0); 4],
        ..fm::Params::default()
    };
    assert_eq!(peak(&solo(&e, &p, 48_000, 60.0, 0.1, 1, 0.0)), 0.0);
}

/// Renders `p` on `note` until its engine says it has ended; the seconds
/// that took (at most `limit`).
fn rings_for<E: Engine>(e: &E, p: &E::Params, note: f32, limit: f32) -> f32 {
    let rate = Rate::new(48_000);
    let mut v = E::Voice::default();
    e.note_on(
        p,
        &mut v,
        &NoteOn {
            note,
            velocity: 1.0,
            seed: 1,
            pan: 0.0,
            shape: 0.0,
            glide: None,
            line: 0,
        },
    );
    let m = VoiceMod {
        phase_inc: rate.inc(note),
        pitch: note,
        ..VoiceMod::default()
    };
    let mut out = [0.0; BLOCK];
    let mut blocks = 0;
    while !e.done(&v) && (blocks as f32) < limit * 48_000.0 / BLOCK as f32 {
        e.render(p, &mut v, &m, &mut out);
        blocks += 1;
    }
    blocks as f32 * BLOCK as f32 / 48_000.0
}

#[test]
fn fm_bells_ring_out_and_say_so() {
    let e = Fm::new(Rate::new(48_000));
    // GLASS BELL's longest carrier decays 60 dB in 4 s: 80 dB in 5.33 s.
    let t = rings_for(&e, &fm::GLASS_BELL, 84.0, 20.0);
    assert!((t - 16.0 / 3.0).abs() < 0.05, "{t}");
    // A sustaining patch rings as long as it is held.
    assert_eq!(rings_for(&e, &fm::BRASS, 60.0, 3.0), 3.0);
}

#[test]
fn fm_retriggers_without_a_click() {
    let rate = Rate::new(48_000);
    let e = Fm::new(rate);
    for p in [fm::BRASS, fm::GLASS_BELL, fm::E_PIANO] {
        let mut v = fm::Voice::default();
        let n = NoteOn {
            note: 72.0,
            velocity: 1.0,
            seed: 1,
            pan: 0.0,
            shape: 0.0,
            glide: None,
            line: 0,
        };
        e.note_on(&p, &mut v, &n);
        let m = VoiceMod {
            phase_inc: rate.inc(72.0),
            pitch: 72.0,
            ..VoiceMod::default()
        };
        let mut out = vec![0.0; 40 * BLOCK];
        for block in out[..20 * BLOCK].chunks_exact_mut(BLOCK) {
            e.render(&p, &mut v, &m, block);
        }
        e.note_on(&p, &mut v, &n);
        for block in out[20 * BLOCK..].chunks_exact_mut(BLOCK) {
            e.render(&p, &mut v, &m, block);
        }
        let before = biggest_jump(&out[10 * BLOCK..20 * BLOCK]);
        let across = biggest_jump(&out[20 * BLOCK - 1..21 * BLOCK]);
        assert!(
            across < 2.0 * before + 0.02,
            "{p:?}: {before} then {across}"
        );
    }
}

#[test]
fn supersaws_centre_is_in_tune_at_every_rate() {
    for rate in [44_100, 48_000, 96_000] {
        let e = Supersaw::new(Rate::new(rate));
        for note in [45.0, 57.0, 69.0, 76.0] {
            let centre = supersaw::Params {
                mix: 0.0,
                ..supersaw::SHIMMER_PAD
            };
            let x = solo(&e, &centre, rate, note, 0.3, 1, 0.0);
            assert_in_tune(&x, rate, note, "the centre saw");
            // The shape at -1 closes the spread: every saw on the note (no
            // sub, which would be heard an octave down).
            let wide = supersaw::Params {
                sub: 0.0,
                ..supersaw::BIG_CHORD
            };
            let mut v = supersaw::Voice::default();
            let n = NoteOn {
                note,
                ..NoteOn::default()
            };
            e.note_on(&wide, &mut v, &n);
            let r = Rate::new(rate);
            let m = VoiceMod {
                phase_inc: r.inc(note),
                pitch: note,
                shape: -1.0,
                ..VoiceMod::default()
            };
            let mut closed = vec![0.0; (0.3 * rate as f32) as usize / BLOCK * BLOCK];
            for block in closed.chunks_exact_mut(BLOCK) {
                e.render(&wide, &mut v, &m, block);
            }
            assert_in_tune(&closed, rate, note, "the closed spread");
        }
    }
}

#[test]
fn supersaws_spread_is_centred_on_the_note() {
    // The power within half a semitone of the note, on a one-cent grid,
    // centres on the note: the centre saw, the pairs either side of it.
    let e = Supersaw::new(Rate::new(48_000));
    for seed in [1, 2, 3] {
        let x = solo(&e, &supersaw::SHIMMER_PAD, 48_000, 57.0, 2.5, seed, 0.0);
        let x = &x[4_800..];
        let f = f64::from(dsp::hz(57.0));
        let power: Vec<(f64, f64)> = (-50..=50)
            .map(|c| {
                let c = f64::from(c);
                (c, magnitude(x, f * (c / 1_200.0).exp2(), 48_000.0).powi(2))
            })
            .collect();
        let total: f64 = power.iter().map(|(_, p)| p).sum();
        let mean = power.iter().map(|(c, p)| c * p).sum::<f64>() / total;
        assert!(mean.abs() < 2.0, "seed {seed}: {mean} cents");
    }
}

#[test]
fn supersaws_brightness_opens_its_filter() {
    let e = Supersaw::new(Rate::new(48_000));
    let c = |brightness: f32| {
        centroid(
            &solo(&e, &supersaw::CHOIR_PAD, 48_000, 57.0, 0.3, 1, brightness),
            48_000.0,
        )
    };
    let (dark, open) = (c(0.0), c(2.0));
    assert!(open > 1.5 * dark, "{dark} {open}");
}

#[test]
fn supersaws_phases_come_from_the_seed() {
    let e = Supersaw::new(Rate::new(48_000));
    let p = supersaw::BIG_CHORD;
    let a = solo(&e, &p, 48_000, 60.0, 0.2, 5, 0.0);
    assert_eq!(a, solo(&e, &p, 48_000, 60.0, 0.2, 5, 0.0));
    assert_ne!(a, solo(&e, &p, 48_000, 60.0, 0.2, 6, 0.0));
    // FM draws nothing: the same every time.
    let fm = Fm::new(Rate::new(48_000));
    let b = solo(&fm, &fm::COIN_GLINT, 48_000, 93.0, 0.2, 5, 0.0);
    assert_eq!(b, solo(&fm, &fm::COIN_GLINT, 48_000, 93.0, 0.2, 6, 0.0));
}

#[test]
fn supersaw_stays_bounded_at_the_extremes() {
    let e = Supersaw::new(Rate::new(48_000));
    for corner in 0..64u32 {
        let pick = |bit: u32| ((corner >> bit) & 1) as f32;
        let p = supersaw::Params {
            pairs: if pick(0) == 1.0 { 255 } else { 0 },
            detune: 200.0 * pick(1),
            mix: pick(2),
            sub: pick(3),
            cutoff: 20.0 + 19_980.0 * pick(4),
            res: pick(5),
            key_track: 1.0,
        };
        for (note, cutoff) in [(24.0, -4.0), (120.0, 4.0), (60.0, 0.0)] {
            let x = solo(&e, &p, 48_000, note, 0.1, 7, cutoff);
            assert!(
                x.iter().all(|s| s.is_finite() && s.abs() <= 1.0),
                "{p:?} {note}: {}",
                peak(&x)
            );
        }
    }
}
