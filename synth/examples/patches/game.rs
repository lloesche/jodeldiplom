//! The game's own palette, on the four synthesis engines: coins, a cash
//! drawer, a riser, ticks, a "pew", brass, plucks, a slot bell, a drone, a
//! chip arpeggio and an "aww".

use super::{hit, Cue, Demo};
use jodel_synth::engines::{analog, lofi, noise, phase};
use jodel_synth::patch::{Adsr, Glide, Lfo, LfoRate, Spread};
use jodel_synth::{Bank, Patch};

/// PHASE's bell, less bent: rounder at the high notes a bell rings on.
const SOFT_BELL: phase::Params = phase::Params {
    dcw: 0.1,
    ..phase::BELL
};

fn coin(b: &mut Bank) -> Vec<Cue> {
    // B5 then E6 on one voice: the second retriggers the first.
    let p = b.add(Patch {
        adsr: hit(0.45),
        mono: true,
        level: -6.0,
        reverb: 0.1,
        ..Patch::new(lofi::COIN)
    });
    vec![(0.0, p, 83.0, 1.0, 0.0), (0.08, p, 88.0, 1.0, 0.0)]
}

fn drawer(b: &mut Bank) -> Vec<Cue> {
    // The drawer sliding out (noise rising), the rollers' crackle, the
    // stop's clack, and the "ching".
    let slide = b.add(Patch {
        adsr: Adsr {
            attack: 0.02,
            release: 0.06,
            ..Adsr::default()
        },
        pitch: Glide {
            from: -12.0,
            time: 0.25,
            curve: 1.0,
        },
        gate: 0.24,
        level: 2.0,
        ..Patch::new(noise::DRAWER)
    });
    let rollers = b.add(Patch {
        gate: 0.26,
        adsr: Adsr {
            release: 0.03,
            ..Adsr::default()
        },
        level: -6.0,
        ..Patch::new(noise::Params {
            density: 90.0,
            cutoff: 4_500.0,
            ..noise::CRACKLE
        })
    });
    let clack = b.add(Patch {
        adsr: hit(0.04),
        level: -2.0,
        ..Patch::new(noise::Params {
            cutoff: 1_200.0,
            res: 0.6,
            ..noise::TICK
        })
    });
    let ching = b.add(Patch {
        adsr: hit(1.2),
        level: -4.0,
        reverb: 0.35,
        filter_env: 0.4,
        ..Patch::new(SOFT_BELL)
    });
    vec![
        (0.0, slide, 60.0, 1.0, -0.2),
        (0.0, rollers, 60.0, 1.0, -0.2),
        (0.27, clack, 60.0, 1.0, 0.0),
        (0.3, ching, 88.0, 1.0, 0.2),
        (0.3, ching, 95.0, 0.7, 0.3),
    ]
}

fn riser(b: &mut Bank) -> Vec<Cue> {
    let p = b.add(Patch {
        adsr: Adsr {
            attack: 1.3,
            release: 0.3,
            ..Adsr::default()
        },
        pitch: Glide {
            from: -24.0,
            time: 1.5,
            curve: 0.6,
        },
        gate: 1.5,
        level: -3.0,
        reverb: 0.3,
        ..Patch::new(noise::WHOOSH)
    });
    vec![(0.0, p, 72.0, 1.0, 0.0)]
}

fn tick(b: &mut Bank) -> Vec<Cue> {
    let p = b.add(Patch {
        adsr: hit(0.015),
        level: 6.0,
        ..Patch::new(noise::TICK)
    });
    (0..4)
        .map(|k| (k as f32 * 0.12, p, 60.0, 1.0, 0.0))
        .collect()
}

fn pew(b: &mut Bank) -> Vec<Cue> {
    // A synced saw diving two and a half octaves in a fifth of a second,
    // an echo after it.
    let p = b.add(Patch {
        adsr: hit(0.25),
        pitch: Glide {
            from: 30.0,
            time: 0.18,
            curve: 2.5,
        },
        level: 0.0,
        delay: 0.35,
        ..Patch::new(analog::ZAP)
    });
    vec![(0.0, p, 52.0, 1.0, 0.0)]
}

fn brass(b: &mut Bank) -> Vec<Cue> {
    let adsr = Adsr {
        attack: 0.03,
        decay: 0.4,
        sustain: 0.6,
        release: 0.25,
    };
    let spread = Spread {
        detune: 6.0,
        level: 1.0,
        start: 0.006,
        ..Spread::default()
    };
    let saws = b.add(Patch {
        adsr,
        filter_env: 3.0,
        gate: 0.3,
        level: -12.0,
        reverb: 0.25,
        spread,
        ..Patch::new(analog::BRASS)
    });
    let pd = b.add(Patch {
        adsr,
        gate: 0.3,
        level: -13.0,
        reverb: 0.25,
        spread,
        filter_env: 2.6,
        ..Patch::new(phase::BRASS)
    });
    let mut cues = Vec::new();
    for (k, note) in [60.0, 64.0, 67.0, 72.0].into_iter().enumerate() {
        let pan = -0.3 + 0.2 * k as f32;
        cues.push((0.0, saws, note, 1.0, pan));
        cues.push((0.0, pd, note, 1.0, -pan));
    }
    cues
}

fn arpeggio(b: &mut Bank) -> Vec<Cue> {
    // C major pentatonic up two octaves and back, sixteenths at 120 a
    // minute: each on its frame (6000 at 48 kHz).
    let p = b.add(Patch {
        adsr: hit(0.3),
        filter_env: 4.0,
        level: -2.0,
        delay: 0.3,
        reverb: 0.15,
        ..Patch::new(analog::PLUCK)
    });
    let notes = [
        60, 62, 64, 67, 69, 72, 74, 76, 79, 76, 74, 72, 69, 67, 64, 62,
    ];
    notes
        .iter()
        .enumerate()
        .map(|(k, &n)| {
            (
                k as f32 * 0.125,
                p,
                n as f32,
                if k % 4 == 0 { 1.0 } else { 0.8 },
                0.0,
            )
        })
        .collect()
}

fn slot_bell(b: &mut Bank) -> Vec<Cue> {
    // A payout bell ringing fast, two strikers.
    let p = b.add(Patch {
        adsr: hit(0.5),
        level: -6.0,
        reverb: 0.2,
        spread: Spread {
            level: 2.0,
            ..Spread::default()
        },
        filter_env: 0.4,
        ..Patch::new(SOFT_BELL)
    });
    (0..14)
        .map(|k| {
            (
                k as f32 / 14.0,
                p,
                if k % 2 == 0 { 79.0 } else { 83.0 },
                1.0,
                if k % 2 == 0 { -0.3 } else { 0.3 },
            )
        })
        .collect()
}

fn drone(b: &mut Bank) -> Vec<Cue> {
    let swell = Adsr {
        attack: 0.8,
        decay: 0.0,
        sustain: 1.0,
        release: 1.5,
    };
    let p = b.add(Patch {
        adsr: swell,
        lfo: Lfo {
            rate: LfoRate::Hz(0.3),
            pitch: 0.08,
            bright: 0.75,
            ..Lfo::default()
        },
        gate: 2.5,
        level: -14.0,
        reverb: 0.5,
        ..Patch::new(phase::DRONE)
    });
    let wind = b.add(Patch {
        adsr: swell,
        gate: 2.5,
        level: -16.0,
        reverb: 0.5,
        ..Patch::new(noise::WIND)
    });
    vec![
        (0.0, p, 45.0, 1.0, -0.3),
        (0.0, p, 52.0, 1.0, 0.3),
        (0.0, p, 60.0, 0.8, 0.0),
        (0.0, wind, 60.0, 1.0, 0.0),
    ]
}

fn chip_arp(b: &mut Bank) -> Vec<Cue> {
    let p = b.add(Patch {
        adsr: Adsr {
            release: 0.05,
            ..Adsr::default()
        },
        gate: 0.35,
        level: -10.0,
        delay: 0.2,
        ..Patch::new(lofi::ARP)
    });
    [72.0, 77.0, 79.0, 84.0]
        .into_iter()
        .enumerate()
        .map(|(k, n)| (k as f32 * 0.4, p, n, 1.0, 0.0))
        .collect()
}

fn aww(b: &mut Bank) -> Vec<Cue> {
    // "Ah" sliding into "aw", down a fourth.
    let slide = Glide {
        from: 5.0,
        time: 0.6,
        curve: 1.0,
    };
    let a = b.add(Patch {
        adsr: Adsr {
            attack: 0.04,
            release: 0.25,
            ..Adsr::default()
        },
        pitch: slide,
        gate: 0.3,
        level: -9.0,
        reverb: 0.2,
        lfo: Lfo {
            rate: LfoRate::Hz(5.5),
            pitch: 0.14,
            fade: 0.3,
            ..Lfo::default()
        },
        ..Patch::new(lofi::VOX_A)
    });
    let o = b.add(Patch {
        adsr: Adsr {
            attack: 0.3,
            release: 0.3,
            ..Adsr::default()
        },
        pitch: slide,
        gate: 0.55,
        level: -9.0,
        reverb: 0.2,
        lfo: Lfo {
            rate: LfoRate::Hz(5.5),
            pitch: 0.14,
            fade: 0.3,
            ..Lfo::default()
        },
        ..Patch::new(lofi::VOX_O)
    });
    vec![(0.0, a, 62.0, 1.0, 0.0), (0.0, o, 62.0, 1.0, 0.0)]
}

pub const DEMOS: [Demo; 11] = [
    Demo {
        name: "coin",
        what: "LOFI two-note blip",
        seconds: 0.7,
        room: 0.4,
        make: coin,
    },
    Demo {
        name: "drawer",
        what: "NOISE slide + dust + clack + PHASE ching",
        seconds: 1.6,
        room: 0.5,
        make: drawer,
    },
    Demo {
        name: "riser",
        what: "NOISE band rising two octaves",
        seconds: 2.2,
        room: 0.6,
        make: riser,
    },
    Demo {
        name: "tick",
        what: "NOISE band bursts",
        seconds: 0.5,
        room: 0.4,
        make: tick,
    },
    Demo {
        name: "pew",
        what: "ANALOG sync dive + delay",
        seconds: 0.9,
        room: 0.4,
        make: pew,
    },
    Demo {
        name: "brass",
        what: "ANALOG + PHASE brass chord",
        seconds: 1.1,
        room: 0.6,
        make: brass,
    },
    Demo {
        name: "arpeggio",
        what: "ANALOG pluck, pentatonic sixteenths at 120",
        seconds: 3.0,
        room: 0.5,
        make: arpeggio,
    },
    Demo {
        name: "slot_bell",
        what: "PHASE bell, fast strikes",
        seconds: 1.7,
        room: 0.5,
        make: slot_bell,
    },
    Demo {
        name: "drone",
        what: "PHASE ring drone + NOISE wind",
        seconds: 4.5,
        room: 0.85,
        make: drone,
    },
    Demo {
        name: "chip_arp",
        what: "LOFI major arpeggio",
        seconds: 1.8,
        room: 0.4,
        make: chip_arp,
    },
    Demo {
        name: "aww",
        what: "LOFI vowels sliding down",
        seconds: 1.2,
        room: 0.5,
        make: aww,
    },
];
