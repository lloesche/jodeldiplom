//! The palette of the physical engines: bells, a casino bell, a coin, a
//! wine glass, a kalimba, a marimba, plucks, a harp, an "aww", an "ooh"
//! choir and a whisper.

use super::{Cue, Demo};
use jodel_synth::engines::{formant, modal, string};
use jodel_synth::patch::{Adsr, Glide, Lfo, LfoRate, Spread};
use jodel_synth::{Bank, Patch};

/// A struck or plucked body: sounding as long as it rings (its engine ends
/// it), let go after `gate` with a release of `release`.
fn ringing(
    engine: impl Into<jodel_synth::engines::AnyParams>,
    gate: f32,
    release: f32,
    level: f32,
) -> Patch {
    Patch {
        adsr: Adsr {
            release,
            ..Adsr::default()
        },
        gate,
        level,
        reverb: 0.25,
        ..Patch::new(engine)
    }
}

/// The C major scale's steps.
const MAJOR: [f32; 7] = [0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0];

fn bell(b: &mut Bank) -> Vec<Cue> {
    let p = b.add(ringing(modal::BELL, 5.0, 0.8, 0.0));
    vec![(0.0, p, 72.0, 1.0, -0.2), (1.6, p, 67.0, 0.9, 0.2)]
}

fn slot_bells(b: &mut Bank) -> Vec<Cue> {
    // A casino payout bell: two small bells, a clapper between them
    // striking each in turn, 24 strikes a second, then ringing out. Each
    // bell is one voice struck again (its ring damped by each strike, so it
    // shimmers at a ceiling instead of piling up).
    let bell = |b: &mut Bank| {
        b.add(Patch {
            mono: true,
            spread: Spread {
                level: 1.5,
                ..Spread::default()
            },
            ..ringing(modal::SLOT_BELL, 2.0, 0.6, -1.0)
        })
    };
    let (high, low) = (bell(b), bell(b));
    (0..36)
        .map(|k| {
            let (p, note, pan) = if k % 2 == 0 {
                (high, 93.0, -0.25)
            } else {
                (low, 88.0, 0.25)
            };
            (k as f32 / 24.0, p, note, 1.0, pan)
        })
        .collect()
}

fn coin(b: &mut Bank) -> Vec<Cue> {
    // "Ch-ing": a small plate hit twice, a little scrape.
    let p = b.add(ringing(modal::COIN, 1.0, 0.3, 8.0));
    vec![(0.0, p, 93.0, 1.0, 0.0), (0.07, p, 100.0, 1.0, 0.1)]
}

fn glass(b: &mut Bank) -> Vec<Cue> {
    // One wine glass tapped twice, the second softer: its pairs beat.
    let p = b.add(Patch {
        mono: true,
        reverb: 0.2,
        ..ringing(modal::GLASS, 6.0, 1.0, 6.0)
    });
    vec![(0.0, p, 77.0, 1.0, 0.0), (2.2, p, 77.0, 0.6, 0.0)]
}

fn kalimba(b: &mut Bank) -> Vec<Cue> {
    let p = b.add(ringing(modal::KALIMBA, 1.5, 0.4, 8.0));
    [72.0, 76.0, 79.0, 81.0, 79.0, 76.0, 74.0, 72.0]
        .into_iter()
        .enumerate()
        .map(|(k, n)| (k as f32 * 0.25, p, n, 0.8, 0.0))
        .collect()
}

fn marimba(b: &mut Bank) -> Vec<Cue> {
    // Up two octaves of C major.
    let p = b.add(ringing(modal::MARIMBA, 1.0, 0.3, 8.0));
    (0..15)
        .map(|k| {
            let note = 60.0 + 12.0 * (k / 7) as f32 + MAJOR[k % 7];
            (k as f32 * 0.09, p, note, 0.8, -0.3 + 0.04 * k as f32)
        })
        .collect()
}

fn plucks(b: &mut Bank) -> Vec<Cue> {
    // C, E, G, C plucked, twice.
    let p = b.add(ringing(string::PLUCK, 0.8, 0.3, 0.0));
    [60.0, 64.0, 67.0, 72.0, 60.0, 64.0, 67.0, 72.0]
        .into_iter()
        .enumerate()
        .map(|(k, n)| (k as f32 * 0.2, p, n, 0.8, 0.0))
        .collect()
}

fn harp(b: &mut Bank) -> Vec<Cue> {
    // A glissando up three octaves of C major.
    let p = b.add(ringing(string::HARP, 3.0, 0.8, -6.0));
    (0..22)
        .map(|k| {
            let note = 48.0 + 12.0 * (k / 7) as f32 + MAJOR[k % 7];
            (k as f32 * 0.045, p, note, 0.8, -0.4 + 0.04 * k as f32)
        })
        .collect()
}

fn aww(b: &mut Bank) -> Vec<Cue> {
    // A woman's "a" gliding to "o" and down a fourth.
    let p = b.add(Patch {
        adsr: Adsr {
            attack: 0.08,
            release: 0.4,
            ..Adsr::default()
        },
        pitch: Glide {
            from: 5.0,
            time: 1.3,
            curve: 1.0,
        },
        shape: Glide {
            from: -0.5,
            time: 1.3,
            curve: 1.0,
        },
        gate: 1.3,
        reverb: 0.25,
        lfo: Lfo {
            rate: LfoRate::Hz(5.2),
            pitch: 0.35,
            fade: 0.3,
            ..Lfo::default()
        },
        ..Patch::new(formant::AWW)
    });
    vec![(0.0, p, 62.0, 1.0, 0.0)]
}

fn choir(b: &mut Bank) -> Vec<Cue> {
    let p = b.add(Patch {
        adsr: Adsr {
            attack: 0.4,
            release: 0.8,
            ..Adsr::default()
        },
        gate: 3.0,
        level: -4.0,
        reverb: 0.4,
        spread: Spread {
            detune: 8.0,
            lfo_phase: 1.0,
            ..Spread::default()
        },
        lfo: Lfo {
            rate: LfoRate::Hz(5.0),
            pitch: 0.25,
            fade: 0.3,
            ..Lfo::default()
        },
        ..Patch::new(formant::OOH)
    });
    vec![
        (0.0, p, 60.0, 1.0, -0.4),
        (0.0, p, 64.0, 1.0, 0.0),
        (0.0, p, 67.0, 1.0, 0.4),
    ]
}

fn whisper(b: &mut Bank) -> Vec<Cue> {
    // A low, breathy "oo" opening towards "o".
    let p = b.add(Patch {
        adsr: Adsr {
            attack: 0.5,
            release: 1.0,
            ..Adsr::default()
        },
        shape: Glide {
            from: 0.8,
            time: 2.5,
            curve: 1.0,
        },
        gate: 2.5,
        reverb: 0.4,
        ..Patch::new(formant::WHISPER)
    });
    vec![(0.0, p, 43.0, 1.0, 0.0)]
}

pub const DEMOS: [Demo; 11] = [
    Demo {
        name: "bell",
        what: "MODAL church bell, struck twice",
        seconds: 6.5,
        room: 0.7,
        make: bell,
    },
    Demo {
        name: "slot_bells",
        what: "MODAL casino bell: two bells, 24 strikes/s",
        seconds: 3.0,
        room: 0.4,
        make: slot_bells,
    },
    Demo {
        name: "coin",
        what: "MODAL plate, ch-ing",
        seconds: 1.3,
        room: 0.4,
        make: coin,
    },
    Demo {
        name: "glass",
        what: "MODAL wine glass, tapped twice",
        seconds: 7.0,
        room: 0.5,
        make: glass,
    },
    Demo {
        name: "kalimba",
        what: "MODAL tines, a tune",
        seconds: 3.5,
        room: 0.5,
        make: kalimba,
    },
    Demo {
        name: "marimba_run",
        what: "MODAL bars, up two octaves",
        seconds: 3.0,
        room: 0.5,
        make: marimba,
    },
    Demo {
        name: "pluck_arpeggio",
        what: "STRING plucks C E G C",
        seconds: 3.0,
        room: 0.5,
        make: plucks,
    },
    Demo {
        name: "harp_gliss",
        what: "STRING harp, up three octaves",
        seconds: 4.5,
        room: 0.6,
        make: harp,
    },
    Demo {
        name: "aww",
        what: "FORMANT a -> o, down a fourth",
        seconds: 2.0,
        room: 0.5,
        make: aww,
    },
    Demo {
        name: "ooh_choir",
        what: "FORMANT three voices, ooh",
        seconds: 4.0,
        room: 0.7,
        make: choir,
    },
    Demo {
        name: "whisper",
        what: "FORMANT whisper, oo -> o",
        seconds: 4.0,
        room: 0.6,
        make: whisper,
    },
];
