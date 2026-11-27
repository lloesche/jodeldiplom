//! The cost of a full pool: every voice sounding at once through the bus
//! (every voice sending to the chorus, the delay and the hall, then the
//! compressor and the limiter), for 10 s at 48 kHz. First the bus alone,
//! then 32 voices of a mix of every engine (in their heavier settings),
//! then 32 of each engine alone, then 128 of the mix and 128 of LOFI. The
//! drums ring out by themselves, so theirs are struck again every quarter
//! second. Prints the real-time factor, the time a block takes, and the
//! size of a voice.
//!
//! ```text
//! cargo run --release -p jodel-synth --example bench
//! ```

use jodel_synth::engines::{
    analog, drum, fm, formant, lofi, modal, noise, phase, string, supersaw, AnyParams, AnyVoice,
};
use jodel_synth::patch::{Lfo, LfoRate};
use jodel_synth::{Bank, NoteOn, Patch, Synth, BLOCK};
use std::time::Instant;

const RATE: u32 = 48_000;
const SECONDS: f32 = 10.0;

/// Every engine, each in a setting that rings the whole 10 s.
fn engines() -> [(&'static str, AnyParams); 10] {
    [
        ("noise", noise::WHOOSH.into()),
        (
            "analog",
            analog::Params {
                sub: 0.4,
                noise: 0.1,
                drive: 0.5,
                ..analog::BRASS
            }
            .into(),
        ),
        (
            "lofi",
            lofi::Params {
                crush: 16_000.0,
                bits: 6,
                ..lofi::VOX_A
            }
            .into(),
        ),
        (
            "phase",
            phase::Params {
                sub: 0.3,
                ..phase::BELL
            }
            .into(),
        ),
        ("modal", modal::BELL.into()),
        (
            "string",
            string::Params {
                decay: 1.0,
                stiffness: 0.5,
                ..string::HARP
            }
            .into(),
        ),
        ("formant", formant::AWW.into()),
        ("fm", fm::BRASS.into()),
        ("supersaw", supersaw::BIG_CHORD.into()),
        ("drum", drum::CRASH.into()),
    ]
}

/// Renders a pool of `voices` with `notes` of them sounding, each on a
/// patch of its own, of `engines` taken in turn, for 10 s; the seconds it
/// took and the voices sounding at the end.
fn run(engines: &[AnyParams], voices: usize, notes: usize) -> (f64, usize) {
    let lfo = Lfo {
        rate: LfoRate::Beats(1.0),
        pitch: 0.1,
        bright: 0.5,
        amp: 0.2,
        shape: 0.3,
        ..Lfo::default()
    };
    let mut bank = Bank::new();
    let ids: Vec<_> = (0..notes)
        .map(|k| {
            let e = engines[k % engines.len()];
            bank.add(Patch {
                gate: 60.0,
                level: -30.0,
                reverb: 0.3,
                delay: 0.2,
                chorus: 0.3,
                mono: matches!(e, AnyParams::Drum(_)),
                lfo,
                ..Patch::new(e)
            })
        })
        .collect();
    let mut synth = Synth::new(RATE, bank, voices);
    let note = |k: usize| NoteOn {
        note: 36.0 + (k % 60) as f32,
        velocity: 1.0,
        seed: k as u32,
        pan: 0.0,
        shape: 0.0,
        glide: None,
        line: 0,
    };
    for (k, &id) in ids.iter().enumerate() {
        synth.note(0, id, note(k), None);
    }
    // A quarter second at a time, the drums struck again at its start.
    let quarter = RATE as usize / 4;
    let mut out = vec![0.0f32; 2 * quarter];
    let start = Instant::now();
    for q in 0..4 * SECONDS as usize {
        if q > 0 {
            for (k, &id) in ids.iter().enumerate() {
                if matches!(engines[k % engines.len()], AnyParams::Drum(_)) {
                    synth.note((q * quarter) as u64, id, note(k), None);
                }
            }
        }
        synth.render_to(&mut out);
    }
    (start.elapsed().as_secs_f64(), synth.voices().count())
}

fn main() {
    let blocks = (SECONDS * RATE as f32) as usize / BLOCK;
    let report = |what: &str, (took, sounding): (f64, usize)| {
        println!(
            "{what:<12} {sounding:>3} voices, {SECONDS} s at {RATE} Hz: {took:.3} s, {:.0}x real time, {:.2} us a block of {BLOCK} ({:.0} us of audio)",
            f64::from(SECONDS) / took,
            took * 1e6 / blocks as f64,
            BLOCK as f64 * 1e6 / f64::from(RATE),
        );
    };
    println!(
        "a voice's engine state: {} bytes; the strings' pool: {} lines of {} bytes",
        std::mem::size_of::<AnyVoice>(),
        string::LINES,
        string::LINE * 4
    );
    let all = engines();
    let mix = all.map(|(_, e)| e);
    report("bus alone", run(&mix, 32, 0));
    report("mix", run(&mix, 32, 32));
    for (name, e) in all {
        report(name, run(&[e], 32, 32));
    }
    report("mix x128", run(&mix, 128, 128));
    report("lofi x128", run(&[all[2].1], 128, 128));
}
