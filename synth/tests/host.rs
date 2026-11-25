//! The host: notes start on their frame to the sample; voices are taken
//! oldest-released first, fading out without a click, even two in one
//! block; a mono retrigger does not click either; a release ends a voice;
//! the queue drops what it cannot hold; the spreads come from the seed; the
//! glides, the filter envelope, the tremolo and an LFO in beats move what
//! they should; a note's own length overrides the gate; the pool takes 128
//! voices.

mod common;

use common::measure::{biggest_jump, centroid, cents, pitch, rms};
use common::{left, on, render, RATE};
use jodel_synth::engines::analog;
use jodel_synth::patch::{Adsr, Glide, Lfo, LfoRate, Spread};
use jodel_synth::{Bank, NoteOn, Patch, PatchId, Synth, CUES};

/// A plain sine held `gate` seconds, released over `release`.
fn sine(gate: f32, release: f32) -> Patch {
    Patch {
        adsr: Adsr {
            release,
            ..Adsr::default()
        },
        gate,
        ..Patch::new(analog::SINE)
    }
}

/// The same with a soft attack, held long.
fn soft() -> Patch {
    Patch {
        adsr: Adsr {
            attack: 0.02,
            ..Adsr::default()
        },
        ..sine(10.0, 0.05)
    }
}

fn synth(patches: &[Patch], voices: usize) -> (Synth, Vec<PatchId>) {
    let mut bank = Bank::new();
    let ids = patches.iter().map(|p| bank.add(*p)).collect();
    (Synth::new(RATE, bank, voices), ids)
}

/// The largest step of a plain `soft()` sine on `note`, once it holds.
fn steady_jump(note: f32) -> f32 {
    let (mut s, ids) = synth(&[soft()], 1);
    s.note(0, ids[0], on(note, 1), None);
    biggest_jump(&left(&render(&mut s, 24_000))[4_800..])
}

#[test]
fn a_note_starts_on_its_frame_to_the_sample() {
    for at in [0u64, 1, 31, 32, 1_000, 4_097] {
        let (mut s, ids) = synth(&[sine(1.0, 0.1)], 8);
        assert!(s.note(at, ids[0], on(69.0, 1), None));
        let out = left(&render(&mut s, 6_000));
        let first = out.iter().position(|x| *x != 0.0).expect("it sounds");
        // The ramp starts from 0 on the frame itself; the limiter holds the
        // output back by its latency.
        assert_eq!(first as u64, at + 1 + s.latency() as u64, "{at}");
    }
}

#[test]
fn rendering_in_pieces_is_rendering_at_once() {
    let make = || {
        let (mut s, ids) = synth(&[sine(0.05, 0.05)], 8);
        for k in 0..20u64 {
            let n = NoteOn {
                velocity: 0.8,
                pan: 0.3,
                ..on(60.0 + k as f32, k as u32)
            };
            s.note(k * 1_234, ids[0], n, None);
        }
        s
    };
    let whole = render(&mut make(), 30_000);
    let mut s = make();
    let mut pieces = Vec::new();
    for n in [1usize, 17, 31, 32, 33, 500, 7, 29_379] {
        pieces.extend(render(&mut s, n));
    }
    assert_eq!(s.frame(), 30_000);
    assert_eq!(whole, pieces);
}

#[test]
fn a_voice_is_taken_from_the_oldest_let_go() {
    let (mut s, ids) = synth(&[sine(10.0, 5.0), sine(0.01, 5.0)], 32);
    // An old note let go (ringing in its long release), then the pool
    // filled with held ones.
    s.note(0, ids[1], on(40.0, 0), None);
    for k in 1..32u64 {
        s.note(k, ids[0], on(50.0 + k as f32, 0), None);
    }
    render(&mut s, 2_000);
    assert_eq!(s.voices().count(), 32);
    assert!(s.voices().any(|v| v.note == 40.0 && !v.held));
    s.note(2_000, ids[0], on(100.0, 0), None);
    render(&mut s, 64);
    assert_eq!(s.voices().count(), 32);
    assert!(!s.voices().any(|v| v.note == 40.0), "the released one went");
    assert!(s.voices().any(|v| v.note == 100.0));
    // With every voice held, the oldest goes.
    s.note(2_100, ids[0], on(101.0, 0), None);
    render(&mut s, 64);
    assert!(!s.voices().any(|v| v.note == 51.0), "the oldest went");
    assert_eq!(s.stolen(), 2);
}

#[test]
fn a_voice_taken_fades_out_without_a_click() {
    // One voice: a held A4, taken mid-block by an E5.
    let (mut s, ids) = synth(&[soft()], 1);
    s.note(0, ids[0], on(69.0, 1), None);
    s.note(10_013, ids[0], on(76.0, 2), None);
    let out = left(&render(&mut s, 20_000));
    let jump = biggest_jump(&out[4_800..]);
    assert!(jump < 1.5 * steady_jump(76.0), "{jump}");
    // The A4 fades out after the frame, not before it.
    assert!(out[10_013].abs() > 0.0 && out[10_000..10_013].iter().any(|x| x.abs() > 0.1));
    assert_eq!(s.stolen(), 1);
}

#[test]
fn two_notes_on_one_voice_in_one_block_fade_the_first() {
    // One voice, a held A4; two notes land on it in one block: the A4 fades
    // out (never cut), the first of the two never sounds, the second plays.
    let (mut s, ids) = synth(&[soft()], 1);
    s.note(0, ids[0], on(69.0, 1), None);
    s.note(9_605, ids[0], on(72.0, 2), None);
    s.note(9_620, ids[0], on(76.0, 3), None);
    let out = left(&render(&mut s, 20_000));
    let jump = biggest_jump(&out[4_800..]);
    assert!(jump < 1.5 * steady_jump(76.0), "{jump}");
    let notes: Vec<f32> = s.voices().map(|v| v.note).collect();
    assert_eq!(notes, [76.0]);
    let got = pitch(&out[14_400..19_200], RATE as f32);
    assert!(cents(got, 659.255).abs() < 5.0, "{got}");
}

#[test]
fn a_mono_retrigger_does_not_click() {
    let mono = Patch {
        mono: true,
        ..soft()
    };
    for to in [69.0, 74.0] {
        let (mut s, ids) = synth(&[mono], 4);
        s.note(0, ids[0], on(69.0, 1), None);
        for k in 1..6u64 {
            s.note(k * 3_001, ids[0], on(to, k as u32), None);
        }
        let out = left(&render(&mut s, 24_000));
        let jump = biggest_jump(&out[2_400..]);
        assert!(jump < 1.5 * steady_jump(to), "{to}: {jump}");
        assert_eq!(s.voices().count(), 1);
    }
}

#[test]
fn a_release_ends_the_voice_in_silence() {
    let (mut s, ids) = synth(&[sine(0.1, 0.2)], 8);
    s.note(100, ids[0], on(69.0, 9), None);
    let out = render(&mut s, 48_000);
    assert_eq!(s.voices().count(), 0);
    // Sounding for its gate and release (0.3 s), then nothing at all.
    let last = out.iter().rposition(|x| *x != 0.0).unwrap() / 2;
    assert!((last as f32 / 48_000.0 - 0.3).abs() < 0.01, "{last}");
}

#[test]
fn a_note_of_its_own_length_overrides_the_gate() {
    let (mut s, ids) = synth(&[sine(1.0, 0.1)], 8);
    s.note(0, ids[0], on(69.0, 9), Some(4_800));
    let out = render(&mut s, 48_000);
    let last = out.iter().rposition(|x| *x != 0.0).unwrap() / 2;
    assert!((last as f32 / 48_000.0 - 0.2).abs() < 0.01, "{last}");
}

#[test]
fn a_full_queue_drops_and_counts() {
    let (mut s, ids) = synth(&[sine(0.1, 0.1)], 8);
    for k in 0..CUES as u64 {
        assert!(s.note(1_000_000 + k, ids[0], on(60.0, 0), None));
    }
    assert!(!s.note(5, ids[0], on(60.0, 0), None));
    assert_eq!(s.dropped(), 1);
}

#[test]
fn the_spreads_come_from_the_seed() {
    let spread = Patch {
        spread: Spread {
            detune: 50.0,
            level: 6.0,
            start: 0.01,
            lfo_phase: 1.0,
        },
        ..sine(0.2, 0.1)
    };
    let once = |seed: u32| {
        let (mut s, ids) = synth(&[spread], 8);
        s.note(0, ids[0], on(69.0, seed), None);
        render(&mut s, 9_600)
    };
    assert_eq!(once(5), once(5));
    let (a, b) = (left(&once(5)), left(&once(6)));
    assert_ne!(a, b);
    let pa = pitch(&a[2_400..], 48_000.0);
    let pb = pitch(&b[2_400..], 48_000.0);
    assert!(cents(pa, 440.0).abs() <= 50.5 && cents(pb, 440.0).abs() <= 50.5);
    assert!((pa - pb).abs() > 0.01, "{pa} {pb}");
}

#[test]
fn a_mono_patch_keeps_one_voice() {
    let mono = Patch {
        mono: true,
        ..sine(1.0, 0.1)
    };
    let (mut s, ids) = synth(&[mono], 8);
    for k in 0..5u64 {
        s.note(k * 1_000, ids[0], on(60.0 + k as f32, 0), None);
    }
    render(&mut s, 6_000);
    let notes: Vec<f32> = s.voices().map(|v| v.note).collect();
    assert_eq!(notes, [64.0]);
}

#[test]
fn the_pitch_glide_bends_down_to_the_note() {
    let pew = Patch {
        pitch: Glide {
            from: 12.0,
            time: 0.2,
            curve: 1.0,
        },
        ..sine(1.0, 0.1)
    };
    let (mut s, ids) = synth(&[pew], 8);
    s.note(0, ids[0], on(57.0, 0), None);
    let out = left(&render(&mut s, 24_000));
    let early = pitch(&out[480..2_880], 48_000.0);
    let late = pitch(&out[14_400..19_200], 48_000.0);
    assert!(early > 360.0 && early < 440.0, "{early}");
    assert!(cents(late, 220.0).abs() < 5.0, "{late}");
}

#[test]
fn the_filter_envelope_brightens_then_settles() {
    let pluck = Patch {
        adsr: Adsr {
            decay: 0.3,
            sustain: 0.2,
            ..Adsr::default()
        },
        filter_env: 4.0,
        ..sine(1.0, 0.1)
    };
    let pluck = Patch {
        engine: analog::Params {
            cutoff: 300.0,
            ..analog::Params::default()
        }
        .into(),
        ..pluck
    };
    let (mut s, ids) = synth(&[pluck], 8);
    s.note(0, ids[0], on(48.0, 0), None);
    let out = left(&render(&mut s, 48_000));
    let (early, late) = (
        centroid(&out[0..2_048], 48_000.0),
        centroid(&out[38_000..40_048], 48_000.0),
    );
    assert!(early > 2.0 * late, "{early} {late}");
}

/// The level every 10 ms of `x`.
fn envelope(x: &[f32]) -> Vec<f32> {
    x.chunks(480).map(rms).collect()
}

/// The period of `e` (one value every 10 ms), seconds: its strongest
/// autocorrelation between 0.2 and 1.5 s.
fn period(e: &[f32]) -> f32 {
    let mean = e.iter().sum::<f32>() / e.len() as f32;
    let d: Vec<f32> = e.iter().map(|v| v - mean).collect();
    let corr = |lag: usize| d.iter().zip(&d[lag..]).map(|(a, b)| a * b).sum::<f32>();
    let lag = (20..150)
        .max_by(|&a, &b| corr(a).total_cmp(&corr(b)))
        .unwrap();
    lag as f32 * 0.01
}

#[test]
fn a_tremolo_in_beats_follows_the_tempo() {
    let tremolo = Patch {
        lfo: Lfo {
            rate: LfoRate::Beats(1.0),
            amp: 1.0,
            ..Lfo::default()
        },
        ..sine(10.0, 0.1)
    };
    for (bpm, beat) in [(120.0, 0.5), (60.0, 1.0)] {
        let (mut s, ids) = synth(&[tremolo], 8);
        s.set_tempo(bpm);
        s.note(0, ids[0], on(69.0, 0), None);
        let e = envelope(&left(&render(&mut s, 4 * 48_000)));
        assert!((period(&e) - beat).abs() < 0.03, "{bpm}: {}", period(&e));
        // Full depth: it dips to near silence once a beat.
        let (lo, hi) = e[20..]
            .iter()
            .fold((f32::MAX, 0.0f32), |(l, h), v| (l.min(*v), h.max(*v)));
        assert!(lo < 0.1 * hi, "{lo} {hi}");
    }
}

#[test]
fn the_lfo_waits_and_fades_in() {
    let vibrato = Patch {
        lfo: Lfo {
            rate: LfoRate::Hz(6.0),
            amp: 1.0,
            delay: 0.5,
            fade: 0.5,
            ..Lfo::default()
        },
        ..sine(10.0, 0.1)
    };
    let (mut s, ids) = synth(&[vibrato], 8);
    s.note(0, ids[0], on(69.0, 0), None);
    let e = envelope(&left(&render(&mut s, 2 * 48_000)));
    let swing = |w: &[f32]| {
        w.iter().cloned().fold(0.0f32, f32::max) - w.iter().cloned().fold(f32::MAX, f32::min)
    };
    let (before, during, after) = (swing(&e[5..45]), swing(&e[55..95]), swing(&e[110..190]));
    // (A 10 ms window holds no whole number of A4's cycles: a little
    // ripple even with no tremolo.)
    assert!(
        before < 0.02 && during > 0.05 && after > during,
        "{before} {during} {after}"
    );
}

#[test]
fn the_pool_takes_128_voices() {
    let (mut s, ids) = synth(&[sine(10.0, 0.1)], 128);
    for k in 0..128u64 {
        s.note(k * 7, ids[0], on(40.0 + (k % 60) as f32, k as u32), None);
    }
    render(&mut s, 4_800);
    assert_eq!(s.voices().count(), 128);
    assert_eq!(s.stolen(), 0);
    s.note(5_000, ids[0], on(100.0, 0), None);
    render(&mut s, 480);
    assert_eq!((s.voices().count(), s.stolen()), (128, 1));
}

#[test]
fn a_pile_of_sounds_never_passes_the_ceiling() {
    // Sixty-four loud notes at once, a jackpot's worth: the bus's limiter
    // holds them under -1 dBFS.
    let loud = Patch {
        level: 12.0,
        reverb: 1.0,
        ..Patch::new(analog::Params::default())
    };
    let (mut s, ids) = synth(&[loud], 64);
    for k in 0..64u64 {
        s.note(
            k * 3,
            ids[0],
            on(36.0 + (k % 36) as f32, k as u32),
            Some(24_000),
        );
    }
    let out = render(&mut s, 48_000);
    let ceiling = 10f32.powf(-1.0 / 20.0);
    let peak = out.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    assert!(peak <= ceiling + 1e-4 && peak > 0.5 * ceiling, "{peak}");
}
