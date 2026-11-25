//! The palette of FM and SUPERSAW: each FM preset in a few notes (the
//! bells struck twice), the pads holding a C major chord, dry and through
//! the chorus, and a sparkle arpeggio through the chorus.

use super::{Cue, Demo};
use jodel_synth::chorus::{ChorusSettings, Wave};
use jodel_synth::engines::{fm, supersaw};
use jodel_synth::patch::{Adsr, Glide, Spread};
use jodel_synth::{Bank, Patch};

/// The chorus the chorused demos' sends go through: a three-tap ensemble,
/// returned at half (the dry demos return none of it).
pub const ENSEMBLE: ChorusSettings = ChorusSettings {
    rate: 0.6,
    depth: 2.0,
    delay: 12.0,
    mix: 0.5,
    spread: 1.0,
    wave: Wave::Sine,
    taps: 3,
};

/// An FM patch: held `gate` seconds (a struck one ends itself sooner),
/// released over `release`, at `level` dB.
fn struck(engine: fm::Params, gate: f32, release: f32, level: f32) -> Patch {
    Patch {
        adsr: Adsr {
            release,
            ..Adsr::default()
        },
        gate,
        level,
        reverb: 0.25,
        chorus: 1.0,
        ..Patch::new(engine)
    }
}

/// A pad: swelling in over `attack`, held `gate` seconds, released over
/// 1.5 s, a little detuned from note to note.
fn pad(engine: supersaw::Params, attack: f32, gate: f32, level: f32) -> Patch {
    Patch {
        adsr: Adsr {
            attack,
            release: 1.5,
            ..Adsr::default()
        },
        gate,
        level,
        reverb: 0.35,
        chorus: 1.0,
        spread: Spread {
            detune: 4.0,
            ..Spread::default()
        },
        ..Patch::new(engine)
    }
}

/// `notes` together at `t`, panned across, on `p`.
fn chord(p: jodel_synth::PatchId, t: f32, notes: &[f32], velocity: f32) -> Vec<Cue> {
    let last = (notes.len() - 1).max(1) as f32;
    notes
        .iter()
        .enumerate()
        .map(|(k, &n)| (t, p, n, velocity, -0.4 + 0.8 * k as f32 / last))
        .collect()
}

fn glass_bell(b: &mut Bank) -> Vec<Cue> {
    // Struck twice, the second softer, then a third over it.
    let p = b.add(struck(fm::GLASS_BELL, 6.0, 1.0, 0.0));
    let mut cues = vec![(0.0, p, 84.0, 1.0, 0.0), (1.6, p, 84.0, 0.8, 0.0)];
    cues.extend(chord(p, 3.2, &[79.0, 88.0], 0.8));
    cues
}

fn e_piano(b: &mut Bank) -> Vec<Cue> {
    let held = b.add(struck(fm::E_PIANO, 1.0, 0.3, -4.0));
    let short = b.add(struck(fm::E_PIANO, 0.25, 0.3, -4.0));
    let mut cues = chord(held, 0.0, &[60.0, 64.0, 67.0], 0.8);
    for (k, n) in [72.0, 76.0, 79.0, 76.0].into_iter().enumerate() {
        cues.push((1.3 + k as f32 * 0.3, short, n, 0.6 + 0.1 * k as f32, 0.1));
    }
    cues
}

fn coin_glint(b: &mut Bank) -> Vec<Cue> {
    // "Ting-ting", twice.
    let p = b.add(struck(fm::COIN_GLINT, 1.0, 0.3, 0.0));
    vec![
        (0.0, p, 93.0, 1.0, 0.0),
        (0.07, p, 100.0, 1.0, 0.1),
        (0.8, p, 93.0, 0.8, -0.1),
        (0.87, p, 100.0, 0.8, 0.0),
    ]
}

fn sparkle(b: &mut Bank) -> Vec<Cue> {
    let p = b.add(struck(fm::SPARKLE, 0.5, 0.2, 0.0));
    [84.0, 88.0, 91.0, 96.0]
        .into_iter()
        .enumerate()
        .map(|(k, n)| (k as f32 * 0.1, p, n, 0.9, -0.3 + 0.2 * k as f32))
        .collect()
}

fn chime(b: &mut Bank) -> Vec<Cue> {
    let p = b.add(struck(fm::CHIME, 5.0, 1.0, 0.0));
    vec![
        (0.0, p, 79.0, 1.0, -0.2),
        (2.0, p, 79.0, 0.8, -0.2),
        (3.0, p, 84.0, 0.8, 0.2),
    ]
}

fn marimba(b: &mut Bank) -> Vec<Cue> {
    // Up two octaves of C major.
    const MAJOR: [f32; 7] = [0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0];
    let p = b.add(struck(fm::MARIMBA, 1.0, 0.3, 0.0));
    (0..15)
        .map(|k| {
            let note = 60.0 + 12.0 * (k / 7) as f32 + MAJOR[k % 7];
            (k as f32 * 0.12, p, note, 0.8, -0.3 + 0.04 * k as f32)
        })
        .collect()
}

fn brass(b: &mut Bank) -> Vec<Cue> {
    let swell = b.add(struck(fm::BRASS, 1.2, 0.3, -6.0));
    let stab = b.add(struck(fm::BRASS, 0.2, 0.2, -6.0));
    let mut cues = chord(swell, 0.0, &[60.0, 64.0, 67.0], 1.0);
    cues.extend(chord(stab, 1.6, &[65.0, 69.0, 72.0], 1.0));
    cues
}

fn bell(b: &mut Bank) -> Vec<Cue> {
    // The Diva's bell: low, struck twice.
    let p = b.add(struck(fm::BELL, 12.0, 2.0, 0.0));
    vec![(0.0, p, 48.0, 1.0, 0.0), (4.0, p, 48.0, 0.85, 0.0)]
}

/// The pads' C major chord.
const C_MAJOR: [f32; 4] = [60.0, 64.0, 67.0, 72.0];

fn shimmer_pad(b: &mut Bank) -> Vec<Cue> {
    // An octave up: the glow under the streak.
    let p = b.add(pad(supersaw::SHIMMER_PAD, 0.8, 3.0, -14.0));
    chord(p, 0.0, &C_MAJOR.map(|n| n + 12.0), 1.0)
}

fn choir_pad(b: &mut Bank) -> Vec<Cue> {
    let p = b.add(pad(supersaw::CHOIR_PAD, 0.6, 3.0, -12.0));
    chord(p, 0.0, &C_MAJOR, 1.0)
}

fn big_chord(b: &mut Bank) -> Vec<Cue> {
    let p = b.add(Patch {
        adsr: Adsr {
            attack: 0.05,
            release: 1.2,
            ..Adsr::default()
        },
        ..pad(supersaw::BIG_CHORD, 0.05, 2.5, -15.0)
    });
    chord(p, 0.0, &[48.0, 60.0, 64.0, 67.0, 72.0], 1.0)
}

fn riser_pad(b: &mut Bank) -> Vec<Cue> {
    // Rising an octave into the chord over 3 s, the filter opening with
    // the swell.
    let p = b.add(Patch {
        pitch: Glide {
            from: -12.0,
            time: 3.0,
            curve: 1.5,
        },
        filter_env: 3.0,
        ..pad(supersaw::RISER_PAD, 3.0, 3.0, -12.0)
    });
    chord(p, 0.0, &C_MAJOR, 1.0)
}

fn sparkle_arpeggio(b: &mut Bank) -> Vec<Cue> {
    // C major pentatonic up and down from C6, sixteenths at 120 a minute,
    // twice round.
    let p = b.add(Patch {
        delay: 0.2,
        ..struck(fm::SPARKLE, 0.2, 0.15, -3.0)
    });
    let notes = [
        84, 86, 88, 91, 93, 96, 98, 100, 103, 100, 98, 96, 93, 91, 88, 86,
    ];
    (0..32)
        .map(|k| {
            let pan = if k % 2 == 0 { -0.25 } else { 0.25 };
            let velocity = if k % 4 == 0 { 1.0 } else { 0.75 };
            (k as f32 * 0.125, p, notes[k % 16] as f32, velocity, pan)
        })
        .collect()
}

/// The demos rendered dry.
pub const DEMOS: [Demo; 12] = [
    Demo {
        name: "fm_glass_bell",
        what: "FM glass bell, struck twice, then two over it",
        seconds: 7.0,
        room: 0.6,
        make: glass_bell,
    },
    Demo {
        name: "fm_e_piano",
        what: "FM electric piano, a chord and a line",
        seconds: 4.0,
        room: 0.5,
        make: e_piano,
    },
    Demo {
        name: "fm_coin_glint",
        what: "FM coin ting-ting, twice",
        seconds: 2.0,
        room: 0.4,
        make: coin_glint,
    },
    Demo {
        name: "fm_sparkle",
        what: "FM sparkle, four notes up",
        seconds: 1.5,
        room: 0.5,
        make: sparkle,
    },
    Demo {
        name: "fm_chime",
        what: "FM chime, struck twice, then a fourth up",
        seconds: 6.0,
        room: 0.6,
        make: chime,
    },
    Demo {
        name: "fm_marimba",
        what: "FM marimba, up two octaves",
        seconds: 3.0,
        room: 0.5,
        make: marimba,
    },
    Demo {
        name: "fm_brass",
        what: "FM brass, a swell and a stab",
        seconds: 2.6,
        room: 0.6,
        make: brass,
    },
    Demo {
        name: "fm_bell",
        what: "FM deep bell, struck twice",
        seconds: 12.0,
        room: 0.8,
        make: bell,
    },
    Demo {
        name: "pad_shimmer",
        what: "SUPERSAW shimmer pad, C major",
        seconds: 5.0,
        room: 0.7,
        make: shimmer_pad,
    },
    Demo {
        name: "pad_choir",
        what: "SUPERSAW choir pad, C major",
        seconds: 5.0,
        room: 0.7,
        make: choir_pad,
    },
    Demo {
        name: "pad_big_chord",
        what: "SUPERSAW big chord, C major",
        seconds: 4.0,
        room: 0.6,
        make: big_chord,
    },
    Demo {
        name: "pad_riser",
        what: "SUPERSAW riser into C major",
        seconds: 5.0,
        room: 0.7,
        make: riser_pad,
    },
];

/// The demos rendered through [`ENSEMBLE`].
pub const CHORUSED: [Demo; 5] = [
    Demo {
        name: "pad_shimmer_chorus",
        what: "SUPERSAW shimmer pad, C major, chorus",
        seconds: 5.0,
        room: 0.7,
        make: shimmer_pad,
    },
    Demo {
        name: "pad_choir_chorus",
        what: "SUPERSAW choir pad, C major, chorus",
        seconds: 5.0,
        room: 0.7,
        make: choir_pad,
    },
    Demo {
        name: "pad_big_chord_chorus",
        what: "SUPERSAW big chord, C major, chorus",
        seconds: 4.0,
        room: 0.6,
        make: big_chord,
    },
    Demo {
        name: "pad_riser_chorus",
        what: "SUPERSAW riser into C major, chorus",
        seconds: 5.0,
        room: 0.7,
        make: riser_pad,
    },
    Demo {
        name: "sparkle_arpeggio_chorus",
        what: "FM sparkle, pentatonic sixteenths at 120, chorus",
        seconds: 5.0,
        room: 0.5,
        make: sparkle_arpeggio,
    },
];
