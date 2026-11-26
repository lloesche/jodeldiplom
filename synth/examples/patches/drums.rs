//! The drums, the hall and the bus dynamics: every drum preset on its own,
//! a four-bar beat at 120 beats a minute cued on exact frames, the slam,
//! a bell dry and through the hall send, and a stack of loud jackpot hits
//! and bells at once, with the bus's compressor off and its limiter at full
//! scale, and with both as they are.

use super::RATE;
use jodel_synth::dynamics::{CompressorSettings, LimiterSettings};
use jodel_synth::engines::{drum, modal, AnyParams};
use jodel_synth::hall::HallSettings;
use jodel_synth::patch::Adsr;
use jodel_synth::{Bank, NoteOn, Patch, Synth, BLOCK};

/// A note: its frame, its patch (an index into the demo's patches), MIDI
/// note, velocity and pan.
type Hit = (u64, usize, f32, f32, f32);

/// A one-shot at `level` dB: held far longer than it rings, so its engine
/// ends it.
fn one_shot(engine: impl Into<AnyParams>, level: f32) -> Patch {
    Patch {
        adsr: Adsr {
            release: 0.5,
            ..Adsr::default()
        },
        gate: 10.0,
        level,
        ..Patch::new(engine.into())
    }
}

/// `hits` of `patches`, `seconds` long (a whole number of blocks), the
/// synth's bus as `setup` leaves it.
fn play(patches: &[Patch], hits: &[Hit], seconds: f32, setup: fn(&mut Synth)) -> Vec<f32> {
    let mut bank = Bank::new();
    let ids: Vec<_> = patches.iter().map(|&p| bank.add(p)).collect();
    let mut synth = Synth::new(RATE, bank, 64);
    setup(&mut synth);
    for (k, &(frame, p, note, velocity, pan)) in hits.iter().enumerate() {
        let n = NoteOn {
            note,
            velocity,
            seed: 1 + k as u32,
            pan,
            shape: 0.0,
        };
        assert!(synth.note(frame, ids[p], n, None));
    }
    let frames = (seconds * RATE as f32) as usize / BLOCK * BLOCK;
    let mut out = vec![0.0; 2 * frames];
    synth.render_to(&mut out);
    out
}

/// The bus as it is: the gentle compressor, the limiter at -1 dBFS.
fn bus(_: &mut Synth) {}

/// The bus without its compressor, its limiter at its highest ceiling,
/// 0 dBFS (it never lets a sample past full scale).
fn raw(s: &mut Synth) {
    s.set_compressor(CompressorSettings {
        ratio: 1.0,
        ..jodel_synth::GENTLE
    });
    s.set_limiter(LimiterSettings {
        ceiling: 0.0,
        ..LimiterSettings::default()
    });
}

/// The bus with a long hall on the reverb send.
fn hall(s: &mut Synth) {
    s.set_reverb(HallSettings {
        size: 0.85,
        decay: 2.8,
        damp: 0.45,
        predelay: 0.025,
        wet: 0.6,
        width: 1.0,
    });
}

/// The frame of sixteenth `step` at 120 beats a minute (6000 a step).
fn step(step: u64) -> u64 {
    step * RATE as u64 / 8
}

fn beat() -> Vec<f32> {
    let patches = [
        one_shot(drum::KICK, 0.0),
        one_shot(drum::SNARE, -3.0),
        one_shot(drum::HAT, -9.0),
        one_shot(drum::OPEN_HAT, -10.0),
        one_shot(drum::CLAP, -6.0),
        one_shot(drum::TOM_HIGH, -4.0),
        one_shot(drum::TOM_LOW, -3.0),
    ];
    let mut hits: Vec<Hit> = Vec::new();
    for bar in 0..4u64 {
        let at = |s: u64| step(16 * bar + s);
        for s in [0, 8, 11] {
            hits.push((at(s), 0, 60.0, 1.0, 0.0));
        }
        for s in [4, 12] {
            hits.push((at(s), 1, 60.0, 1.0, 0.05));
            if bar >= 2 {
                hits.push((at(s), 4, 60.0, 0.8, -0.05));
            }
        }
        for s in (0..16).step_by(2) {
            let (patch, velocity) = if s == 14 { (3, 0.8) } else { (2, 0.9) };
            let accent = if s % 4 == 0 { 1.0 } else { 0.75 };
            hits.push((at(s), patch, 60.0, velocity * accent, 0.3));
        }
        if bar == 3 {
            for (s, patch) in [(13, 5), (14, 6), (15, 6)] {
                hits.push((at(s), patch, 60.0, 0.9, -0.2));
            }
        }
    }
    play(&patches, &hits, 8.5, bus)
}

/// The slam's three layers, and their patches, at `level` dB.
fn slam(level: f32, frame: u64, patches: &mut Vec<Patch>, hits: &mut Vec<Hit>) {
    for (_, p) in drum::SLAM {
        hits.push((frame, patches.len(), 60.0, 1.0, 0.0));
        patches.push(one_shot(p, level));
    }
}

/// The bell struck twice, its send to the hall at `send`, the bus as
/// `setup` leaves it.
fn bell(send: f32, setup: fn(&mut Synth)) -> Vec<f32> {
    let p = Patch {
        reverb: send,
        ..one_shot(modal::BELL, 0.0)
    };
    play(
        &[p],
        &[(0, 0, 72.0, 1.0, -0.2), (step(24), 0, 67.0, 0.9, 0.2)],
        6.0,
        setup,
    )
}

/// Three jackpots a third of a second apart, each a slam, an impact, two
/// casino bells, a coin and a crash, all hit hard and boosted, the bus as
/// `setup` leaves it.
fn jackpot(setup: fn(&mut Synth)) -> Vec<f32> {
    let mut patches = Vec::new();
    let mut hits = Vec::new();
    for k in 0..3u64 {
        let frame = RATE as u64 / 10 + k * RATE as u64 / 3;
        slam(6.0, frame, &mut patches, &mut hits);
        let more = [
            (one_shot(drum::IMPACT, 6.0), 60.0),
            (one_shot(modal::SLOT_BELL, 6.0), 88.0),
            (one_shot(modal::SLOT_BELL, 6.0), 93.0),
            (one_shot(modal::COIN, 6.0), 96.0),
            (one_shot(drum::CRASH, 3.0), 60.0),
        ];
        for (i, (p, note)) in more.into_iter().enumerate() {
            let pan = (i as f32 - 2.0) * 0.2;
            hits.push((frame, patches.len(), note, 1.0, pan));
            patches.push(p);
        }
    }
    play(&patches, &hits, 3.0, setup)
}

/// The demos: their file names, what they are, and their sound.
pub fn processed() -> Vec<(String, String, Vec<f32>)> {
    let mut all = Vec::new();
    for (name, p) in drum::PRESETS {
        let file = name.to_lowercase().replace(' ', "_");
        let seconds = (p.decay.max(p.snap_decay) + 0.3).min(3.0);
        let out = play(&[one_shot(p, 0.0)], &[(0, 0, 60.0, 1.0, 0.0)], seconds, bus);
        all.push((file, format!("DRUM {name}"), out));
    }
    all.push((
        "beat_120".into(),
        "4 bars at 120: kick, snare, hats, clap, toms".into(),
        beat(),
    ));
    let (mut patches, mut hits) = (Vec::new(), Vec::new());
    slam(0.0, 0, &mut patches, &mut hits);
    all.push((
        "slam".into(),
        "SLAM: kick + clap + crash, one frame".into(),
        play(&patches, &hits, 2.2, bus),
    ));
    all.push(("bell_dry".into(), "MODAL bell, dry".into(), bell(0.0, bus)));
    all.push((
        "bell_hall".into(),
        "MODAL bell on the hall send".into(),
        bell(1.0, hall),
    ));
    all.push((
        "jackpot_raw".into(),
        "3 stacked jackpots, no compressor, limiter at 0 dBFS".into(),
        jackpot(raw),
    ));
    all.push((
        "jackpot_limited".into(),
        "the same, the bus's compressor + limiter".into(),
        jackpot(bus),
    ));
    all
}
