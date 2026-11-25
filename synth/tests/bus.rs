//! The bus processors: the hall decays in the time it is set to, its two
//! sides uncorrelated, and at wet 0 leaves the dry sound bit for bit; the
//! limiter never lets a sample past its ceiling, by a smooth gain, and
//! leaves a sound under it as it was; the compressor's gain follows its
//! static curve and its attack.

mod common;

use common::measure::{centroid, db, peak};
use jodel_synth::dsp::{self, Rate};
use jodel_synth::dynamics::{Compressor, CompressorSettings, Limiter, LimiterSettings};
use jodel_synth::engines::drum::{self, Drum};
use jodel_synth::hall::{Hall, HallSettings};
use jodel_synth::{Engine, NoteOn, VoiceMod, BLOCK};

const RATE: u32 = 48_000;

fn rate() -> Rate {
    Rate::new(RATE)
}

/// The hall's answer to a click, both sides, `seconds` long.
fn hall_tail(s: HallSettings, seconds: f32) -> (Vec<f32>, Vec<f32>) {
    let mut h = Hall::new(rate());
    h.set(s);
    let (mut left, mut right) = (Vec::new(), Vec::new());
    for b in 0..(seconds * RATE as f32) as usize / BLOCK {
        let mut send = [0.0; BLOCK];
        if b == 0 {
            send[0] = 1.0;
        }
        let (mut l, mut r) = ([0.0; BLOCK], [0.0; BLOCK]);
        h.process(&send, &mut l, &mut r);
        left.extend(l);
        right.extend(r);
    }
    (left, right)
}

/// The time to -60 dB of `x`, seconds: Schroeder's backward integral of
/// its energy, its slope from -5 to -25 dB taken three times.
fn rt60(x: &[f32]) -> f32 {
    let mut edc = vec![0.0f64; x.len()];
    let mut sum = 0.0;
    for i in (0..x.len()).rev() {
        sum += f64::from(x[i]) * f64::from(x[i]);
        edc[i] = sum;
    }
    let at = |level: f64| {
        let k = edc
            .iter()
            .position(|&e| 10.0 * (e / edc[0]).log10() <= level);
        k.expect("the tail falls that far") as f32 / RATE as f32
    };
    3.0 * (at(-25.0) - at(-5.0))
}

fn plain(decay: f32) -> HallSettings {
    HallSettings {
        decay,
        damp: 0.0,
        predelay: 0.0,
        wet: 1.0,
        ..HallSettings::default()
    }
}

#[test]
fn the_hall_decays_in_its_time() {
    for (decay, size) in [(0.8, 0.3), (2.0, 0.7), (4.0, 1.0)] {
        let (l, r) = hall_tail(
            HallSettings {
                size,
                ..plain(decay)
            },
            decay * 1.5,
        );
        for side in [&l, &r] {
            let t = rt60(side);
            assert!((t / decay - 1.0).abs() < 0.2, "{decay} s: {t} s");
        }
    }
    // Damped, it decays sooner overall, and it stays bounded at the longest.
    let (damped, _) = hall_tail(
        HallSettings {
            damp: 1.0,
            ..plain(2.0)
        },
        3.0,
    );
    assert!(rt60(&damped) < 2.0);
    let (long, _) = hall_tail(plain(30.0), 3.0);
    assert!(long.iter().all(|x| x.is_finite()) && peak(&long) < 1.0);
}

#[test]
fn the_halls_sides_are_uncorrelated() {
    let (l, r) = hall_tail(plain(2.0), 1.5);
    let (l, r) = (&l[4_800..], &r[4_800..]);
    let dot = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum::<f64>()
    };
    let rho = dot(l, r) / (dot(l, l) * dot(r, r)).sqrt();
    assert!(rho.abs() < 0.2, "{rho}");
    // At width 0 the two sides are one.
    let (l0, r0) = hall_tail(
        HallSettings {
            width: 0.0,
            ..plain(2.0)
        },
        0.5,
    );
    assert_eq!(l0, r0);
}

#[test]
fn the_halls_pre_delay_holds_the_tail_back() {
    let (l, _) = hall_tail(
        HallSettings {
            predelay: 0.1,
            ..plain(1.0)
        },
        0.3,
    );
    assert_eq!(peak(&l[..4_800]), 0.0);
    assert!(peak(&l[4_800..]) > 1e-3);
}

#[test]
fn a_dry_hall_is_the_dry_signal_bit_for_bit() {
    let mut h = Hall::new(rate());
    h.set(HallSettings {
        wet: 0.0,
        ..HallSettings::default()
    });
    let mut rng = dsp::seed(9);
    for _ in 0..200 {
        let send: [f32; BLOCK] = std::array::from_fn(|_| dsp::white(&mut rng));
        // Negative zeros too: adding even +0 to one would change it.
        let dry: [[f32; BLOCK]; 2] = std::array::from_fn(|_| {
            std::array::from_fn(|i| if i % 8 == 0 { -0.0 } else { -0.5 * send[i] })
        });
        let (mut l, mut r) = (dry[0], dry[1]);
        h.process(&send, &mut l, &mut r);
        assert_eq!(l.map(f32::to_bits), dry[0].map(f32::to_bits));
        assert_eq!(r.map(f32::to_bits), dry[1].map(f32::to_bits));
    }
}

/// A stress signal: 24 drum hits and loud tones stacked at once and in
/// quick succession, a spike, and a tone at a quarter of the rate whose
/// samples miss its peaks, up to about +24 dBFS.
fn stress() -> Vec<f32> {
    let r = rate();
    let d = Drum::new(r);
    let mut x = vec![0.0f32; RATE as usize];
    for k in 0..24usize {
        let (_, p) = drum::PRESETS[k % drum::PRESETS.len()];
        let mut v = drum::Voice::default();
        d.note_on(
            &p,
            &mut v,
            &NoteOn {
                note: 60.0,
                velocity: 1.0,
                seed: k as u32,
                pan: 0.0,
            },
        );
        let m = VoiceMod {
            phase_inc: r.inc(60.0),
            pitch: 60.0,
            ..VoiceMod::default()
        };
        let at = 4_800 + (k % 6) * 960;
        for block in x[at..].chunks_exact_mut(BLOCK).take(400) {
            let mut y = [0.0; BLOCK];
            d.render(&p, &mut v, &m, &mut y);
            for (s, y) in block.iter_mut().zip(y) {
                *s += 4.0 * y;
            }
        }
    }
    for (i, s) in x[24_000..30_000].iter_mut().enumerate() {
        *s += 3.0 * (i as f32 * 0.07).sin();
    }
    x[31_000] = 12.0;
    for (i, s) in x[36_000..36_400].iter_mut().enumerate() {
        *s = 2.0 * (std::f32::consts::FRAC_PI_2 * i as f32 + std::f32::consts::FRAC_PI_4).sin();
    }
    x
}

/// `left` and `right` through `l`, block by block.
fn limit(l: &mut Limiter, left: &[f32], right: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (mut ol, mut or) = (Vec::new(), Vec::new());
    for (a, b) in left.chunks_exact(BLOCK).zip(right.chunks_exact(BLOCK)) {
        let mut bl: [f32; BLOCK] = a.try_into().unwrap();
        let mut br: [f32; BLOCK] = b.try_into().unwrap();
        l.process(&mut bl, &mut br);
        ol.extend(bl);
        or.extend(br);
    }
    (ol, or)
}

#[test]
fn the_limiter_never_passes_its_ceiling() {
    for ceiling in [-1.0f32, -6.0, 0.0] {
        let mut l = Limiter::new(rate());
        l.set(LimiterSettings {
            ceiling,
            release: 0.05,
        });
        let x = stress();
        let right: Vec<f32> = x.iter().map(|s| -0.7 * s).collect();
        assert!(db(peak(&x)) > 20.0);
        let (out, out_r) = limit(&mut l, &x, &right);
        let c = 10f32.powf(ceiling / 20.0);
        assert!(peak(&out) <= c && peak(&out_r) <= c, "{}", peak(&out));
        // It gets there by a smooth gain, not by clipping: the gain each
        // sample (the output over the input it holds back) moves by at
        // most a look-ahead's share of the way a sample.
        let late = l.latency();
        let span = late - 1;
        let mut last: Option<f32> = None;
        for (i, &y) in out.iter().enumerate().skip(late) {
            let input = x[i - late];
            if input.abs() < 1e-2 {
                last = None;
                continue;
            }
            let g = y / input;
            assert!(g > 0.0 && g <= 1.0 + 1e-6, "{i}: {g}");
            if let Some(prev) = last {
                assert!(
                    (g - prev).abs() <= 1.0 / span as f32 + 1e-3,
                    "{i}: {prev} -> {g}"
                );
            }
            last = Some(g);
        }
    }
}

#[test]
fn under_its_ceiling_the_limiter_only_delays() {
    let mut l = Limiter::new(rate());
    let mut rng = dsp::seed(5);
    let x: Vec<f32> = (0..9_600).map(|_| 0.5 * dsp::white(&mut rng)).collect();
    let (out, _) = limit(&mut l, &x, &x);
    let late = l.latency();
    assert_eq!(&out[late..], &x[..x.len() - late]);
    assert_eq!(l.gain(), 1.0);
}

/// The compressor's output level, dB, for a steady level `level` (dBFS)
/// held for `seconds`.
fn steady(c: &mut Compressor, level: f32, seconds: f32) -> f32 {
    let a = 10f32.powf(level / 20.0);
    let mut out = 0.0;
    for _ in 0..(seconds * RATE as f32) as usize / BLOCK {
        let (mut l, mut r) = ([a; BLOCK], [a; BLOCK]);
        c.process(&mut l, &mut r);
        out = l[BLOCK - 1];
    }
    db(out)
}

#[test]
fn the_compressors_gain_follows_its_static_curve() {
    let s = CompressorSettings {
        threshold: -18.0,
        ratio: 4.0,
        knee: 8.0,
        attack: 0.002,
        release: 0.05,
        makeup: 3.0,
    };
    let mut c = Compressor::new(rate());
    c.set(s);
    let mut level = -40.0;
    while level <= 0.0 {
        let got = steady(&mut c, level, 0.5);
        let want = c.curve(level) + s.makeup;
        assert!((got - want).abs() < 0.02, "{level} dB: {got} {want}");
        level += 1.0;
    }
    // The curve: unity below the knee, the ratio above it, and halfway
    // into the knee's quadratic at the threshold.
    assert_eq!(c.curve(-30.0), -30.0);
    assert!((c.curve(-2.0) - (-18.0 + 16.0 / 4.0)).abs() < 1e-4);
    let mid = -18.0 + (0.25 - 1.0) * 4.0 * 4.0 / 16.0;
    assert!((c.curve(-18.0) - mid).abs() < 1e-4, "{}", c.curve(-18.0));
}

#[test]
fn the_compressor_pulls_down_over_its_attack() {
    let s = CompressorSettings {
        threshold: -20.0,
        ratio: 10.0,
        knee: 0.0,
        attack: 0.01,
        release: 0.2,
        makeup: 0.0,
    };
    let mut c = Compressor::new(rate());
    c.set(s);
    steady(&mut c, -40.0, 0.2);
    assert_eq!(c.reduction(), 0.0);
    let full = 0.0 - c.curve(0.0);
    steady(&mut c, 0.0, 0.01);
    let share = c.reduction() / full;
    assert!((share - (1.0 - (-1.0f32).exp())).abs() < 0.05, "{share}");
}

#[test]
fn the_halls_damping_darkens_its_tail() {
    let at = |damp: f32| {
        let (l, _) = hall_tail(HallSettings { damp, ..plain(2.0) }, 1.2);
        centroid(&l[48_000..49_024], 48_000.0)
    };
    assert!(at(1.0) < 0.7 * at(0.0), "{} {}", at(1.0), at(0.0));
}
