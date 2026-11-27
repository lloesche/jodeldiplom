//! The physical engines, each rendered alone at 48 kHz: sane levels, no
//! NaN, the same sound from the same seed, in tune, decay times that follow
//! their parameters, re-strikes that stay bounded, beating glass pairs, and
//! vowels whose formants land where the table puts them.

mod common;

use common::measure::{
    biggest_jump, cents_off, envelope_peak, magnitude, noise_peak, ring_time, sane,
};
use common::{held, solo, RATE};
use jodel_synth::dsp::{self, Rate};
use jodel_synth::engines::formant::{self, Formant};
use jodel_synth::engines::modal::{self, Modal};
use jodel_synth::engines::string::{self, KarplusString};
use jodel_synth::{Engine, NoteOn, BLOCK};

const SR: f32 = RATE as f32;

fn rate() -> Rate {
    Rate::new(RATE)
}

/// A note at full amplitude for `seconds`, its brightness moved `bright`.
fn play<E: Engine>(
    e: &E,
    p: &E::Params,
    note: f32,
    bright: f32,
    seed: u32,
    seconds: f32,
) -> Vec<f32> {
    solo(e, p, RATE, note, seconds, seed, bright)
}

const MODAL: modal::Params = modal::Params {
    body: 0.0,
    decay: 0.5,
    brightness: 0.6,
    position: 0.5,
    hardness: 0.6,
    noise: 0.0,
    split: 0.0,
};

const STRING: string::Params = string::Params {
    decay: 0.6,
    brightness: 0.5,
    position: 0.3,
    stiffness: 0.25,
};

const VOICE: formant::Params = formant::Params {
    vowel: 0.0,
    breath: 0.0,
    size: 0.0,
    soft: 0.3,
    width: 1.0,
};

#[test]
fn modal_rings_in_tune_on_every_body() {
    let e = Modal::new(rate());
    for body in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] {
        for note in [48.0, 69.0, 90.0] {
            let p = modal::Params { body, ..MODAL };
            let x = play(&e, &p, note, 0.0, 1, 0.6);
            sane(&x, 3.0, &format!("modal body {body} note {note}"));
            let off = cents_off(&x[2400..], dsp::hz(note), SR);
            assert!(off.abs() < 3.0, "body {body} note {note}: {off:.2} cents");
        }
    }
}

#[test]
fn modal_stays_bounded_at_the_extremes() {
    let e = Modal::new(rate());
    for corner in 0..64u32 {
        let pick = |bit: u32| ((corner >> bit) & 1) as f32;
        let p = modal::Params {
            body: pick(5),
            decay: 0.9 * pick(0),
            brightness: pick(1),
            position: pick(2),
            hardness: pick(3),
            noise: pick(4),
            split: 0.5 * pick(1),
        };
        for (note, cutoff) in [(24.0, -4.0), (108.0, 4.0), (60.0, 0.0)] {
            let x = play(&e, &p, note, cutoff, 7, 0.1);
            assert!(
                x.iter().all(|s| s.is_finite() && s.abs() < 4.0),
                "{p:?} {note}"
            );
        }
    }
}

#[test]
fn modal_decay_follows_the_parameter() {
    let e = Modal::new(rate());
    // The lowest mode rings to -60 dB in 0.02 s x 10^(4 decay), the second
    // (of a harmonic body, at twice the pitch) as long times
    // 2^(-2 (1 - brightness)).
    for decay in [0.3, 0.45, 0.6] {
        let p = modal::Params {
            decay,
            position: 0.3,
            ..MODAL
        };
        let x = play(&e, &p, 60.0, 0.0, 1, 1.5);
        let expected = 0.02 * 10f32.powf(4.0 * decay);
        let measured = ring_time(&x, dsp::hz(60.0), expected / 4.0, SR);
        assert!(
            (measured / expected - 1.0).abs() < 0.1,
            "decay {decay}: rings {measured} s, not {expected}"
        );
        let expected = expected * 2f32.powf(-2.0 * (1.0 - MODAL.brightness));
        let measured = ring_time(&x, 2.0 * dsp::hz(60.0), expected / 4.0, SR);
        assert!(
            (measured / expected - 1.0).abs() < 0.1,
            "decay {decay}: the second mode rings {measured} s, not {expected}"
        );
    }
}

#[test]
fn modal_rings_out_and_says_so() {
    let e = Modal::new(rate());
    let p = modal::Params {
        decay: 0.2,
        body: 0.8,
        ..MODAL
    };
    let mut v = Default::default();
    e.note_on(
        &p,
        &mut v,
        &NoteOn {
            note: 72.0,
            velocity: 1.0,
            seed: 3,
            pan: 0.0,
            shape: 0.0,
            glide: None,
            line: 0,
        },
    );
    let m = held(72.0, 0.0);
    let mut out = [0.0; BLOCK];
    let blocks = (0..3000).take_while(|_| {
        e.render(&p, &mut v, &m, &mut out);
        !e.done(&v)
    });
    assert!(blocks.count() < 3000, "never done");
}

/// `p` on `note`, struck `per_second` times a second for `seconds` on one
/// voice (each strike a retrigger of the ringing one).
fn strikes(p: &modal::Params, note: f32, per_second: f32, seconds: f32) -> Vec<f32> {
    let e = Modal::new(rate());
    let mut v = modal::Voice::default();
    let m = held(note, 0.0);
    let blocks = (seconds * SR) as usize / BLOCK;
    let every = (SR / per_second) as usize / BLOCK;
    let mut out = vec![0.0; blocks * BLOCK];
    for (b, block) in out.chunks_mut(BLOCK).enumerate() {
        if b % every == 0 {
            let n = NoteOn {
                note,
                velocity: 1.0,
                seed: b as u32,
                pan: 0.0,
                shape: 0.0,
                glide: None,
                line: 0,
            };
            e.note_on(p, &mut v, &n);
        }
        e.render(p, &mut v, &m, block);
    }
    out
}

#[test]
fn a_body_struck_fast_stays_bounded() {
    // Struck 24 times a second for 2 s, a bell's level grows towards twice a
    // strike's and no further, without a jump past what one strike makes.
    for (p, note) in [
        (modal::SLOT_BELL, 88.0),
        (modal::BELL, 72.0),
        (modal::GLASS, 80.0),
    ] {
        let once = strikes(&p, note, 0.1, 0.5);
        let fast = strikes(&p, note, 24.0, 2.0);
        let (peak1, peak) = (sane(&once, 4.0, "once"), sane(&fast, 4.0, "fast"));
        assert!(
            peak < 2.2 * peak1,
            "{p:?}: {peak} against one strike's {peak1}"
        );
        let (jump1, jump) = (biggest_jump(&once), biggest_jump(&fast));
        assert!(jump < 2.2 * jump1, "{p:?}: jumps {jump} against {jump1}");
        // And the last second no louder than the first: no build-up.
        let half = fast.len() / 2;
        let late = fast[half..].iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(late < 1.3 * fast[..half].iter().fold(0.0f32, |m, s| m.max(s.abs())));
    }
}

#[test]
fn split_glass_modes_beat() {
    // The lowest pair, 0.4 Hz apart, beats once in 2.5 s: its level dips and
    // rises again over 3 s; unsplit it only decays.
    let e = Modal::new(rate());
    let level = |split: f32| {
        let p = modal::Params {
            split,
            ..modal::GLASS
        };
        let x = play(&e, &p, 76.0, 0.0, 1, 3.0);
        let f = dsp::hz(76.0) as f64;
        (0..6)
            .map(|k| magnitude(&x[k * 24_000..k * 24_000 + 9_600], f, SR))
            .collect::<Vec<f64>>()
    };
    let (beating, plain) = (level(0.4), level(0.0));
    assert!(plain.windows(2).all(|w| w[1] < w[0]), "{plain:?}");
    // Against the plain decay, the pair's level follows |cos(pi 0.4 t)|:
    // near 0 at 1.25 s, back near 1 at 2.5 s.
    let ratio: Vec<f64> = beating.iter().zip(&plain).map(|(b, p)| b / p).collect();
    let dip = ratio.iter().cloned().fold(f64::MAX, f64::min);
    assert!(dip < 0.3 && ratio[5] > 0.7, "{ratio:?}");
}

#[test]
fn string_plucks_in_tune() {
    let e = KarplusString::new(rate());
    for note in [40.0, 57.0, 69.0, 84.0] {
        let x = play(&e, &STRING, note, 0.0, 5, 0.6);
        sane(&x, 3.0, &format!("string note {note}"));
        let off = cents_off(&x[2400..], dsp::hz(note), SR);
        assert!(off.abs() < 3.0, "note {note}: {off:.2} cents");
    }
}

#[test]
fn stiffness_stretches_the_partials() {
    // How many cents the sixth partial sits above six times the pitch (the
    // seventh is 267 cents up).
    let e = KarplusString::new(rate());
    let stretch = |stiffness| {
        let p = string::Params {
            brightness: 1.0,
            stiffness,
            ..STRING
        };
        let x = play(&e, &p, 48.0, 0.0, 5, 0.35);
        let at = |c: i32| {
            magnitude(
                &x[4800..],
                6.0 * dsp::hz(48.0) as f64 * (c as f64 / 1200.0).exp2(),
                SR,
            )
        };
        (-50..=200)
            .max_by(|&a, &b| at(a).total_cmp(&at(b)))
            .unwrap()
    };
    let (plain, stiff) = (stretch(0.0), stretch(1.0));
    assert!(
        plain.abs() <= 3,
        "plain: the sixth partial {plain} cents off"
    );
    assert!(
        stiff > 15,
        "stiff: the sixth partial only {stiff} cents sharp"
    );
}

#[test]
fn string_stays_bounded_at_the_extremes() {
    let e = KarplusString::new(rate());
    for corner in 0..16u32 {
        let pick = |bit: u32| ((corner >> bit) & 1) as f32;
        let p = string::Params {
            decay: pick(0),
            brightness: pick(1),
            position: pick(2),
            stiffness: pick(3),
        };
        for (note, cutoff) in [(12.0, -4.0), (110.0, 4.0), (60.0, 0.0)] {
            let x = play(&e, &p, note, cutoff, 9, 0.2);
            assert!(
                x.iter().all(|s| s.is_finite() && s.abs() < 4.0),
                "{p:?} {note}"
            );
        }
    }
}

#[test]
fn string_decay_follows_the_parameter() {
    let e = KarplusString::new(rate());
    let f0 = dsp::hz(52.0);
    // The fundamental rings to -60 dB in 0.05 s x 400^decay, whatever the
    // brightness; the higher harmonics die sooner the darker the string.
    for decay in [0.3, 0.5, 0.7] {
        let mut second = Vec::new();
        for brightness in [0.2, 1.0] {
            let p = string::Params {
                decay,
                brightness,
                stiffness: 0.0,
                ..STRING
            };
            let x = play(&e, &p, 52.0, 0.0, 1, 1.5);
            let expected = 0.05 * 400f32.powf(decay);
            let measured = ring_time(&x, f0, expected / 4.0, SR);
            assert!(
                (measured / expected - 1.0).abs() < 0.1,
                "decay {decay}, brightness {brightness}: rings {measured} s, not {expected}"
            );
            second.push(ring_time(&x, 2.0 * f0, expected / 4.0, SR) / measured);
        }
        assert!(
            second[0] < 0.9 && second[1] > 0.95,
            "decay {decay}: the second harmonic rings {second:?} of the fundamental"
        );
    }
}

/// Hillenbrand's men: (vowel, F1, F2) of i, a, u.
const VOWELS: [(f32, f32, f32); 3] = [
    (0.0, 342.0, 2322.0),
    (0.5, 768.0, 1333.0),
    (1.0, 378.0, 997.0),
];

#[test]
fn whispered_vowels_put_their_formants_where_the_table_does() {
    // A whisper's source is white noise: its spectrum is the resonators'.
    let e = Formant::new(rate());
    for (vowel, f1, f2) in VOWELS {
        let whisper = formant::Params {
            vowel,
            breath: 1.0,
            ..VOICE
        };
        let x = play(&e, &whisper, 45.0, 0.0, 11, 1.0);
        sane(&x, 3.0, &format!("whisper {vowel}"));
        let p1 = noise_peak(&x[4800..], 0.7 * f1, 1.3 * f1, SR);
        let p2 = noise_peak(&x[4800..], 0.85 * f2, 1.15 * f2, SR);
        assert!(
            (p1 / f1 - 1.0).abs() < 0.05,
            "vowel {vowel}: F1 {p1} Hz, not {f1}"
        );
        assert!(
            (p2 / f2 - 1.0).abs() < 0.05,
            "vowel {vowel}: F2 {p2} Hz, not {f2}"
        );
    }
}

#[test]
fn sung_vowels_peak_near_their_formants() {
    // The glottal pulse's own falling spectrum pulls the peaks of the
    // harmonics' envelope down a little, most for a low F1.
    let e = Formant::new(rate());
    for (vowel, f1, f2) in VOWELS {
        let p = formant::Params { vowel, ..VOICE };
        let x = play(&e, &p, 45.0, 0.0, 11, 0.8);
        sane(&x, 3.0, &format!("vowel {vowel}"));
        let steady = &x[14_400..];
        let f0 = dsp::hz(45.0);
        let p1 = envelope_peak(steady, f0, 0.7 * f1, 1.3 * f1, SR);
        let p2 = envelope_peak(steady, f0, 0.85 * f2, 1.15 * f2, SR);
        assert!(
            (p1 / f1 - 1.0).abs() < 0.2,
            "vowel {vowel}: F1 {p1} Hz, not {f1}"
        );
        assert!(
            (p2 / f2 - 1.0).abs() < 0.1,
            "vowel {vowel}: F2 {p2} Hz, not {f2}"
        );
    }
}

#[test]
fn whispers_and_extremes_stay_bounded() {
    let e = Formant::new(rate());
    for corner in 0..16u32 {
        let pick = |bit: u32| ((corner >> bit) & 1) as f32;
        for vowel in [0.0, 0.6, 1.0] {
            let p = formant::Params {
                vowel,
                breath: pick(0),
                size: pick(2),
                soft: pick(1),
                width: 0.25 + 7.75 * pick(3),
            };
            for (note, cutoff) in [(24.0, -4.0), (96.0, 4.0), (57.0, 0.0)] {
                let x = play(&e, &p, note, cutoff, 13, 0.2);
                sane(&x, 4.0, &format!("{p:?} note {note}"));
            }
        }
    }
}

#[test]
fn the_same_seed_gives_the_same_sound() {
    let noisy = modal::Params {
        noise: 0.7,
        body: 0.9,
        ..MODAL
    };
    let breathy = formant::Params {
        breath: 0.5,
        vowel: 0.3,
        ..VOICE
    };
    let modal = Modal::new(rate());
    let string = KarplusString::new(rate());
    let formant = Formant::new(rate());
    let renders: [&dyn Fn(u32) -> Vec<f32>; 3] = [
        &|seed| play(&modal, &noisy, 76.0, 0.0, seed, 0.3),
        &|seed| play(&string, &STRING, 55.0, 0.0, seed, 0.3),
        &|seed| play(&formant, &breathy, 57.0, 0.0, seed, 0.3),
    ];
    for (i, render) in renders.iter().enumerate() {
        assert_eq!(render(42), render(42), "engine {i}: not repeatable");
        assert_ne!(
            render(42),
            render(43),
            "engine {i}: the seed changes nothing"
        );
    }
}

#[test]
fn strings_borrow_their_lines_and_give_them_back() {
    // The pool's lines go to the first strings plucked; one more is silent
    // and done at once; let one go and the next pluck has its line.
    let e = KarplusString::new(rate());
    let m = held(57.0, 0.0);
    let pluck = |v: &mut string::Voice| {
        e.note_on(&STRING, v, &NoteOn::default());
        let mut out = [0.0; BLOCK];
        e.render(&STRING, v, &m, &mut out);
        out.iter().any(|s| *s != 0.0)
    };
    let mut voices: Vec<string::Voice> = (0..string::LINES).map(|_| Default::default()).collect();
    for v in &mut voices {
        assert!(pluck(v) && !e.done(v));
    }
    let mut extra = string::Voice::default();
    assert!(!pluck(&mut extra) && e.done(&extra));
    e.free(&mut voices[3]);
    let mut next = string::Voice::default();
    assert!(pluck(&mut next));
}
