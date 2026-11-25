//! The shared effects: the delay's echoes land on the beat and fade, and,
//! through the host, the hall, the delay and the chorus at wet 0 leave the
//! dry sound as it was (the hall itself is tested with the bus).

mod common;

use common::measure::peak;
use common::{left, on};
use jodel_synth::chorus::ChorusSettings;
use jodel_synth::delay::{Delay, DelaySettings};
use jodel_synth::dsp::Rate;
use jodel_synth::engines::noise;
use jodel_synth::hall::HallSettings;
use jodel_synth::patch::Adsr;
use jodel_synth::{Bank, Patch, Synth, BLOCK};

#[test]
fn the_delay_echoes_on_the_beat_and_fades() {
    let mut d = Delay::new(Rate::new(48_000));
    // A quarter beat at 120 a minute: 125 ms, 6000 samples.
    d.set(
        DelaySettings {
            beats: 0.25,
            feedback: 0.5,
            damp: 0.0,
            wet: 1.0,
            ping_pong: false,
        },
        120.0,
    );
    let mut out = Vec::new();
    for b in 0..1_000 {
        let mut send = [0.0; BLOCK];
        if b == 0 {
            send[0] = 1.0;
        }
        let (mut l, mut r) = ([0.0; BLOCK], [0.0; BLOCK]);
        d.process([&send, &send], &mut l, &mut r);
        out.extend(l);
    }
    let echoes: Vec<usize> = (0..4)
        .map(|k| {
            let from = k * 6_000 + 3_000;
            from + out[from..from + 6_000]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                .unwrap()
                .0
        })
        .collect();
    assert_eq!(echoes, [6_000, 12_000, 18_000, 24_000]);
    for k in 1..4 {
        let ratio = out[echoes[k]] / out[echoes[k - 1]];
        assert!((ratio - 0.5).abs() < 0.01, "{ratio}");
    }
}

/// A burst of noise through the synth, its sends at `reverb` and `delay`,
/// the effects at `wet`.
fn burst(reverb: f32, delay: f32, wet: f32) -> Vec<f32> {
    let mut bank = Bank::new();
    let id = bank.add(Patch {
        adsr: Adsr {
            release: 0.02,
            ..Adsr::default()
        },
        gate: 0.05,
        reverb,
        delay,
        chorus: delay,
        ..Patch::new(noise::TICK)
    });
    let mut s = Synth::new(48_000, bank, 8);
    s.set_reverb(HallSettings {
        wet,
        ..HallSettings::default()
    });
    s.set_chorus(ChorusSettings {
        mix: wet,
        ..ChorusSettings::default()
    });
    s.set_delay(DelaySettings {
        wet,
        ..DelaySettings::default()
    });
    s.note(0, id, on(60.0, 4), None);
    let mut out = vec![0.0; 2 * 96_000];
    s.render_to(&mut out);
    out
}

#[test]
fn wet_zero_is_the_dry_sound() {
    let dry = burst(0.0, 0.0, 0.5);
    assert_eq!(burst(1.0, 1.0, 0.0), dry);
    // And with them on, the sound goes on past the dry one's end.
    let wet = left(&burst(1.0, 1.0, 0.5));
    assert!(peak(&wet[48_000..]) > 1e-3);
    assert_eq!(peak(&left(&dry)[48_000..]), 0.0);
}
