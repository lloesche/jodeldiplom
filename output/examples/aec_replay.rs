//! The echo canceller over recordings of it: the player's `r` on the
//! Karaoke tab records what it works with (three float channels: the
//! microphone, the music going to the speakers, the voice it gave). Each
//! file is fed again through one canceller, in the order given (as in one
//! session: what it learnt carries over), and each second says how it is
//! doing, with the voice it gives now beside the one recorded. The voice it
//! gives now is written beside each file, as FILE.replayed.wav (mono,
//! 32-bit float), and the singer isolated from it (`jodel_output::isolate`,
//! as the meter hears it) as FILE.isolated.wav (16 kHz, lined up with the
//! input), with how long that took.
//!
//! A file of two float channels (the microphone, the music) is taken too:
//! a made-up session, with no voice recorded to compare. `--fresh` gives
//! each file a canceller of its own.
//!
//! ```text
//! cargo run --release -p jodel-output --example aec_replay -- [--fresh] RECORDING.wav...
//! ```

use jodel_output::aec::{EchoCanceller, BLOCK};
use jodel_output::isolate::{Downsampler, Isolator, DELAY, RATE};
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
    let mut aec: Option<EchoCanceller> = None;
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
    let data = (4 * x.len()) as u32;
    let mut b = Vec::with_capacity(44 + 4 * x.len());
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&3u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&rate.to_le_bytes());
    b.extend_from_slice(&(4 * rate).to_le_bytes());
    b.extend_from_slice(&4u16.to_le_bytes());
    b.extend_from_slice(&32u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data.to_le_bytes());
    for v in x {
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, b).unwrap_or_else(|e| panic!("{path}: {e}"));
}

fn replay(path: &str, kept: &mut Option<EchoCanceller>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let u16_at = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
    let channels = usize::from(u16_at(22));
    let rate = u32::from_le_bytes([b[24], b[25], b[26], b[27]]);
    assert!(
        channels == 2 || channels == 3,
        "{path}: {channels} channels, not a recording of the canceller"
    );
    let at = b
        .windows(4)
        .position(|w| w == b"data")
        .expect("a data chunk")
        + 8;
    let x: Vec<f32> = b[at..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let channel = |c: usize| -> Vec<f32> { x.iter().skip(c).step_by(channels).copied().collect() };
    let (mic, music) = (channel(0), channel(1));
    let voice = (channels == 3).then(|| channel(2));
    let aec = kept.get_or_insert_with(|| EchoCanceller::new(rate));
    let mut out = vec![0.0f32; mic.len()];
    let mut block = [0.0f32; BLOCK];
    let second = rate as usize;
    for k in 0..mic.len() / BLOCK {
        let r = k * BLOCK..(k + 1) * BLOCK;
        aec.process(&mic[r.clone()], &music[r.clone()], &mut block);
        out[r].copy_from_slice(&block);
        let end = (k + 1) * BLOCK;
        if end / second != (end - BLOCK) / second {
            let s = aec.status();
            let w = end - second..end;
            let recorded = voice.as_ref().map_or(String::new(), |v| {
                format!(", recorded {:6.1}", db(&v[w.clone()]))
            });
            println!(
                "{:3} s  learnt {:5}  removed {:5.1} dB  delay {:>6}  mic {:6.1}  voice now {:6.1}{recorded}  music {:6.1} dB",
                end / second,
                s.learnt,
                s.removed_db,
                s.delay
                    .map_or("-".to_owned(), |d| format!("{:.0} ms", d * 1000.0)),
                db(&mic[w.clone()]),
                db(&out[w.clone()]),
                db(&music[w]),
            );
        }
    }
    write_wav(&format!("{path}.replayed.wav"), rate, &out);
    // The singer isolated, block by block as the canceller's thread does.
    let started = Instant::now();
    let (mut down, mut iso) = (Downsampler::new(rate), Isolator::new());
    let (mut at_16k, mut isolated) = (Vec::new(), Vec::new());
    for block in out.chunks(BLOCK) {
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
