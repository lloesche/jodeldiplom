//! The echo canceller over recordings of it: the player's `r` on the
//! Karaoke tab records what it works with (five float channels: the
//! microphone, the music going to the left and the right speaker, the voice
//! the canceller gave, and that voice with its leftover suppressed). Each
//! file is fed again through one canceller and suppressor, in the order
//! given (as in one session: what they learnt carries over), and each
//! second says how they are doing, with the voice they give now beside the
//! one recorded. What they give now is written beside each file, as
//! FILE.replayed.wav (the canceller's) and FILE.suppressed.wav (as the game
//! hears it; mono, 32-bit float, lined up with the input), and the singer
//! isolated from the latter (`jodel_output::isolate`, as the meter hears
//! it) as FILE.isolated.wav (16 kHz, lined up with the input), with how
//! long that took.
//!
//! Older recordings of three channels (the microphone, the mean of what
//! the speakers played, the voice) are taken too, and files of two (the
//! microphone, the music): a made-up session, with no voice recorded to
//! compare. `--fresh` gives each file a canceller of its own.
//!
//! ```text
//! cargo run --release -p jodel-output --example aec_replay -- [--fresh] RECORDING.wav...
//! ```

use jodel_output::aec::{Whole, BLOCK};
use jodel_output::isolate::{Downsampler, Isolator, DELAY, RATE};
use std::path::Path;
use std::time::Instant;

fn db(x: &[f32]) -> f64 {
    let p = x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / x.len().max(1) as f64;
    10.0 * p.max(1e-14).log10()
}

fn main() {
    let mut paths: Vec<String> = std::env::args().skip(1).collect();
    let fresh = paths.iter().any(|p| p == "--fresh");
    paths.retain(|p| p != "--fresh");
    if paths.is_empty() {
        eprintln!("usage: aec_replay [--fresh] RECORDING.wav...");
        std::process::exit(2);
    }
    let mut aec: Option<Whole> = None;
    for path in &paths {
        println!("{path}");
        if fresh {
            aec = None;
        }
        replay(path, &mut aec);
    }
}

/// A mono 32-bit float WAV.
fn write_wav(path: &str, rate: u32, x: &[f32]) {
    durable::write_atomic(Path::new(path), &jodel_dsp::wav::float32(rate, 1, x))
        .unwrap_or_else(|e| panic!("{path}: {e}"));
}

fn replay(path: &str, kept: &mut Option<Whole>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let wav = jodel_dsp::wav::parse(&b).unwrap_or_else(|| panic!("{path}: not a WAV file"));
    let (channels, rate) = (usize::from(wav.format.channels), wav.format.rate);
    assert!(
        matches!(channels, 2 | 3 | 5),
        "{path}: {channels} channels, not a recording of the canceller"
    );
    let x = wav
        .float32()
        .unwrap_or_else(|| panic!("{path}: not floats"));
    let channel = |c: usize| -> Vec<f32> { x.iter().skip(c).step_by(channels).copied().collect() };
    let mic = channel(0);
    let (left, right) = if channels == 5 {
        (channel(1), channel(2))
    } else {
        (channel(1), channel(1))
    };
    let music: Vec<f32> = left
        .iter()
        .zip(&right)
        .map(|(l, r)| 0.5 * (l + r))
        .collect();
    let voice = match channels {
        3 => Some(channel(2)),
        5 => Some(channel(4)),
        _ => None,
    };
    let whole = kept.get_or_insert_with(|| Whole::new(rate));
    let second = rate as usize;
    let mut seconds = Vec::new();
    let cleaned = whole.run(&mic, &left, &right, |end, status| {
        if end / second != (end - BLOCK) / second {
            seconds.push((end, *status));
        }
    });
    let (out, suppressed) = (&cleaned.cancelled, &cleaned.suppressed);
    for (end, s) in seconds {
        let w = end - second..end;
        let recorded = voice.as_ref().map_or(String::new(), |v| {
            format!(", recorded {:6.1}", db(&v[w.clone()]))
        });
        println!(
            "{:3} s  learnt {:5}  removed {:5.1} dB  delay {:>6}  mic {:6.1}  voice now {:6.1}  suppressed {:6.1}{recorded}  music {:6.1} dB",
            end / second,
            s.learnt,
            s.removed_db,
            s.delay
                .map_or("-".to_owned(), |d| format!("{:.0} ms", d * 1000.0)),
            db(&mic[w.clone()]),
            db(&out[w.clone()]),
            db(&suppressed[w.clone()]),
            db(&music[w]),
        );
    }
    write_wav(&format!("{path}.replayed.wav"), rate, out);
    write_wav(&format!("{path}.suppressed.wav"), rate, suppressed);
    // The singer isolated, block by block as the canceller's thread does.
    let started = Instant::now();
    let (mut down, mut iso) = (Downsampler::new(rate), Isolator::new());
    let (mut at_16k, mut isolated) = (Vec::new(), Vec::new());
    for block in suppressed.chunks(BLOCK) {
        down.process(block, &mut at_16k);
        iso.process(&at_16k, &mut isolated);
        at_16k.clear();
    }
    let took = started.elapsed().as_secs_f64();
    let seconds = out.len() as f64 / f64::from(rate);
    println!(
        "isolated in {took:.2} s for {seconds:.1} s ({:.1} % of a core)",
        100.0 * took / seconds
    );
    let late = DELAY.min(isolated.len());
    write_wav(&format!("{path}.isolated.wav"), RATE, &isolated[late..]);
}
