//! The chorus: at a mix of 0 it is the input bit for bit, its taps read
//! only within the delay and depth they are set to (held to the line),
//! the wet signal is never louder than the input, and the spread is what
//! sets the sides apart.

use jodel_synth::chorus::{Chorus, ChorusSettings, Wave, DELAY_MAX, DELAY_MIN};
use jodel_synth::dsp::{self, Rate};
use jodel_synth::BLOCK;

const SR: f32 = 48_000.0;

fn chorus(s: ChorusSettings) -> Chorus {
    let mut c = Chorus::new(Rate::new(48_000));
    c.set(s);
    // Let the delay and the depth glide into place.
    for _ in 0..(0.2 * SR) as usize {
        c.frame(0.0, 0.0);
    }
    c
}

/// Noise, -1 to 1, with some negative zeros among it.
fn noise(n: usize, seed: u32) -> Vec<f32> {
    let mut s = dsp::seed(seed);
    (0..n)
        .map(|i| if i % 7 == 0 { -0.0 } else { dsp::white(&mut s) })
        .collect()
}

#[test]
fn a_mix_of_zero_is_the_input_bit_for_bit() {
    let x = noise(20_000, 1);
    let y = noise(20_000, 2);
    for taps in 1..=3 {
        for wave in [Wave::Sine, Wave::Triangle] {
            let s = ChorusSettings {
                mix: 0.0,
                taps,
                wave,
                rate: 3.0,
                ..ChorusSettings::default()
            };
            let mut c = chorus(ChorusSettings { mix: 0.7, ..s });
            // Wet a while, then dry at once.
            for (l, r) in x.iter().zip(&y).take(5_000) {
                c.frame(*l, *r);
            }
            c.set(s);
            for (l, r) in x.iter().zip(&y) {
                let (ol, or) = c.frame(*l, *r);
                assert_eq!((ol.to_bits(), or.to_bits()), (l.to_bits(), r.to_bits()));
            }
        }
    }
    // The block API too.
    let mut c = chorus(ChorusSettings {
        mix: 0.0,
        ..ChorusSettings::default()
    });
    let mut left: [f32; BLOCK] = std::array::from_fn(|i| x[i]);
    let mut right: [f32; BLOCK] = std::array::from_fn(|i| y[i]);
    c.process(&mut left, &mut right);
    assert!(left.iter().zip(&x).all(|(a, b)| a.to_bits() == b.to_bits()));
    assert!(right
        .iter()
        .zip(&y)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
}

/// Where impulses through a fully wet `s` come out: the first and last
/// sample after each impulse at which anything does, the earliest and the
/// latest over impulses spread across the LFO's cycles.
fn reach(s: ChorusSettings) -> (usize, usize) {
    let mut c = chorus(ChorusSettings { mix: 1.0, ..s });
    let spacing = (0.073 * SR) as usize;
    let (mut first, mut last) = (usize::MAX, 0);
    for _ in 0..60 {
        for n in 0..spacing {
            let x = if n == 0 { 1.0 } else { 0.0 };
            let (l, r) = c.frame(x, x);
            if l.abs() > 1e-9 || r.abs() > 1e-9 {
                first = first.min(n);
                last = last.max(n);
            }
        }
    }
    (first, last)
}

#[test]
fn the_taps_read_within_their_delay_and_depth() {
    let ms = SR / 1_000.0;
    for (delay, depth, taps, wave) in [
        (8.0, 3.0, 1, Wave::Sine),
        (12.0, 2.0, 3, Wave::Triangle),
        (5.0, 0.0, 2, Wave::Sine),
    ] {
        let (first, last) = reach(ChorusSettings {
            delay,
            depth,
            taps,
            wave,
            rate: 5.0,
            ..ChorusSettings::default()
        });
        let (lo, hi) = ((delay - depth) * ms, (delay + depth) * ms);
        assert!(
            first as f32 >= lo.floor() && last as f32 <= hi.ceil() + 1.0,
            "{delay} ms ± {depth}: {first}..{last} against {lo}..{hi}"
        );
        // And it sweeps most of that range.
        assert!(
            (last - first) as f32 >= 0.9 * (hi - lo),
            "{delay} ms ± {depth}: {first}..{last}"
        );
    }
    // Settings past the line are held to it.
    let (first, last) = reach(ChorusSettings {
        delay: 1_000.0,
        depth: 1_000.0,
        rate: 5.0,
        ..ChorusSettings::default()
    });
    assert!(first as f32 >= (DELAY_MIN * ms).floor(), "{first}");
    assert!(
        last as f32 <= (2.0 * DELAY_MAX - DELAY_MIN) * ms + 1.0,
        "{last}"
    );
}

#[test]
fn the_wet_signal_keeps_the_input_level() {
    // The taps summed over the square root of their count: about the
    // input's power, and never past that many times its peak.
    let x = noise(48_000, 3);
    let rms = |v: &[f32]| (v.iter().map(|s| s * s).sum::<f32>() / v.len() as f32).sqrt();
    for mix in [0.3, 1.0] {
        let mut c = chorus(ChorusSettings {
            mix,
            rate: 2.0,
            depth: 4.0,
            ..ChorusSettings::default()
        });
        let mut out = Vec::new();
        for s in &x {
            let (l, r) = c.frame(*s, -*s);
            let most = 3f32.sqrt();
            assert!(
                l.is_finite() && l.abs() <= most && r.abs() <= most,
                "{l} {r}"
            );
            out.push(l);
        }
        if mix == 1.0 {
            let ratio = rms(&out[4_800..]) / rms(&x[4_800..]);
            assert!(ratio > 0.8 && ratio < 1.25, "{ratio}");
        }
    }
}

#[test]
fn the_spread_sets_the_sides_apart() {
    let x = noise(10_000, 4);
    let run = |spread: f32| {
        let mut c = chorus(ChorusSettings {
            spread,
            mix: 1.0,
            ..ChorusSettings::default()
        });
        x.iter().map(|s| c.frame(*s, *s)).collect::<Vec<_>>()
    };
    assert!(run(0.0).iter().all(|(l, r)| l == r));
    let apart = run(1.0);
    let differ = apart.iter().filter(|(l, r)| (l - r).abs() > 1e-3).count();
    assert!(differ > apart.len() / 2, "{differ}");
}
