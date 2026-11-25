//! DRUM: every voice stays bounded, is the same every time and rings out
//! by itself; a kick's sweep falls to its tune, the note moves the tune,
//! the hats are bright and the brightness brightens; a clap comes in
//! bursts; and through the host a drum ends its own voice.

mod common;

use common::measure::{centroid, peak, rms};
use jodel_synth::dsp::Rate;
use jodel_synth::engines::drum::{self, Drum, Kind};
use jodel_synth::{Bank, Engine, NoteOn, Patch, Synth, VoiceMod, BLOCK};

const RATE: u32 = 48_000;

fn all() -> Vec<(&'static str, drum::Params)> {
    drum::PRESETS.into_iter().chain(drum::SLAM).collect()
}

/// `p` hit on `note` at full amplitude (brightness `bright`), rendered
/// until its voice says it is done or `seconds` pass: the sound, and
/// whether it ended.
fn hit(p: &drum::Params, note: f32, seconds: f32, seed: u32, bright: f32) -> (Vec<f32>, bool) {
    let rate = Rate::new(RATE);
    let d = Drum::new(rate);
    let mut v = drum::Voice::default();
    d.note_on(
        p,
        &mut v,
        &NoteOn {
            note,
            velocity: 1.0,
            seed,
            pan: 0.0,
        },
    );
    let m = VoiceMod {
        phase_inc: rate.inc(note),
        pitch: note,
        bright,
        ..VoiceMod::default()
    };
    let mut out = Vec::new();
    for _ in 0..(seconds * RATE as f32) as usize / BLOCK {
        if d.done(&v) {
            return (out, true);
        }
        let mut block = [0.0; BLOCK];
        d.render(p, &mut v, &m, &mut block);
        out.extend(block);
    }
    (out, d.done(&v))
}

#[test]
fn every_voice_is_bounded_the_same_every_time_and_ends() {
    for (name, p) in all() {
        let (x, ended) = hit(&p, 60.0, 10.0, 7, 0.0);
        assert!(ended, "{name} rang on");
        assert!(x.iter().all(|s| s.is_finite()), "{name}");
        // Every part is let ring to -80 dB: 4/3 of its time to -60.
        let longest = p.decay.max(p.snap_decay) * 4.0 / 3.0 + 0.05;
        let seconds = x.len() as f32 / RATE as f32;
        assert!(seconds <= longest, "{name}: {seconds} s");
        assert!(rms(&x[..480]) > 1e-3, "{name}: {}", rms(&x[..480]));
        assert!(peak(&x) <= 1.0, "{name}: {}", peak(&x));
        assert_eq!(x, hit(&p, 60.0, 10.0, 7, 0.0).0, "{name}");
        // The extremes stay bounded too.
        let loud = drum::Params {
            drive: 1.0,
            snappy: 1.0,
            tone: 1.0,
            bend: 48.0,
            ..p
        };
        let (y, _) = hit(&loud, 100.0, 1.0, 3, 4.0);
        assert!(y.iter().all(|s| s.is_finite() && s.abs() <= 1.0), "{name}");
    }
}

/// The frequency of each cycle of `x`, Hz, from its upward zero crossings
/// (interpolated between samples).
fn cycles(x: &[f32]) -> Vec<f32> {
    let mut at = Vec::new();
    for i in 1..x.len() {
        if x[i - 1] < 0.0 && x[i] >= 0.0 {
            at.push((i - 1) as f32 + x[i - 1] / (x[i - 1] - x[i]));
        }
    }
    at.windows(2).map(|w| RATE as f32 / (w[1] - w[0])).collect()
}

#[test]
fn a_kicks_sweep_falls_to_its_tune() {
    for p in [drum::KICK_808, drum::KICK, drum::IMPACT] {
        // The body alone: no click to cross zero.
        let body = drum::Params { snappy: 0.0, ..p };
        let (x, _) = hit(&body, 60.0, 0.4, 1, 0.0);
        let f = cycles(&x);
        for w in f.windows(2) {
            assert!(w[1] <= w[0] * 1.005, "{p:?}: {f:?}");
        }
        let (first, last) = (f[0], f[f.len() - 1]);
        assert!(first > 1.2 * last, "{p:?}: {first} -> {last}");
        assert!((last / p.tune - 1.0).abs() < 0.03, "{p:?}: {last}");
    }
}

#[test]
fn the_note_moves_the_tune() {
    let p = drum::Params {
        snappy: 0.0,
        ..drum::TOM_LOW
    };
    let late = |note: f32| {
        let (x, _) = hit(&p, note, 0.4, 1, 0.0);
        let f = cycles(&x);
        f[f.len() - 1]
    };
    let (at, up) = (late(60.0), late(72.0));
    assert!((at / p.tune - 1.0).abs() < 0.02, "{at}");
    assert!((up / at - 2.0).abs() < 0.04, "{at} {up}");
}

#[test]
fn the_hats_are_bright_and_the_brightness_brightens() {
    let c = |p: &drum::Params, bright: f32| {
        let (x, _) = hit(p, 60.0, 0.3, 1, bright);
        centroid(&x[..2_048], RATE as f32)
    };
    for p in [drum::HAT, drum::OPEN_HAT] {
        assert!(c(&p, 0.0) > 6_000.0, "{p:?}: {}", c(&p, 0.0));
    }
    assert!(c(&drum::CRASH, 0.0) > 4_000.0);
    assert!(
        c(&drum::KICK_808, 0.0) < 400.0,
        "{}",
        c(&drum::KICK_808, 0.0)
    );
    for p in [drum::SNARE, drum::CLAP, drum::CRASH] {
        let (dark, bright) = (c(&p, -1.0), c(&p, 1.0));
        assert!(bright > 1.3 * dark, "{p:?}: {dark} {bright}");
    }
    assert_eq!(drum::HAT.kind, Kind::ClosedHat);
}

#[test]
fn a_clap_comes_in_bursts() {
    let (x, _) = hit(&drum::CLAP, 60.0, 0.1, 1, 0.0);
    let ms = |t: f32| (t * RATE as f32 / 1_000.0) as usize;
    let level = |from: f32, to: f32| rms(&x[ms(from)..ms(to)]);
    // Each burst is louder than the moment before it.
    for onset in [8.5, 17.5, 27.0] {
        let (before, after) = (level(onset - 1.5, onset), level(onset, onset + 1.5));
        assert!(after > 2.0 * before, "{onset} ms: {before} {after}");
    }
    // And a tail follows them.
    assert!(level(40.0, 60.0) > 0.05 * level(0.0, 2.0));
}

#[test]
fn through_the_host_a_drum_ends_its_own_voice() {
    let mut bank = Bank::new();
    let ids: Vec<_> = all()
        .into_iter()
        .map(|(_, p)| {
            bank.add(Patch {
                gate: 10.0,
                ..Patch::new(p.into())
            })
        })
        .collect();
    let mut s = Synth::new(RATE, bank, 32);
    for (k, &id) in ids.iter().enumerate() {
        let n = NoteOn {
            note: 60.0,
            velocity: 1.0,
            seed: k as u32,
            pan: 0.0,
        };
        assert!(s.note(0, id, n, None));
    }
    // A block in, all sound; three seconds on, long before their gate lets
    // them go, none: each has ended itself.
    let mut out = vec![0.0; 2 * BLOCK];
    s.render_to(&mut out);
    assert_eq!(s.voices().count(), ids.len());
    let mut rest = vec![0.0; 2 * 3 * RATE as usize];
    s.render_to(&mut rest);
    assert_eq!(s.voices().count(), 0);
    assert!(out.iter().chain(&rest).all(|x| x.is_finite()));
}
