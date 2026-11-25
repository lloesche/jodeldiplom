//! The game's palette as demo sounds, each rendered on the synth to a
//! 48 kHz 16-bit stereo WAV in a folder of its set under the one given:
//! `synth` (the four synthesis engines, [`game`]), `synth-physical` (the
//! physical ones, [`physical`]), `synth-fm` (FM and SUPERSAW, dry and
//! through the chorus send, [`fm`]) and `synth-drums` (the drums, the hall
//! send and the bus's dynamics, [`drums`]). A summary of each is printed:
//! its length, peak, RMS and crest, the largest step between two samples,
//! its peak every 0.25 s (a build-up or a clip shows) and its spectral
//! centroid every 0.1 s (so a sweep can be seen going the right way).
//!
//! ```text
//! cargo run --release -p jodel-synth --example patches -- OUT_DIR
//! ```

#[path = "../../tests/common/measure.rs"]
mod measure;

mod drums;
mod fm;
mod game;
mod physical;

use jodel_synth::chorus::ChorusSettings;
use jodel_synth::delay::DelaySettings;
use jodel_synth::hall::HallSettings;
use jodel_synth::patch::Adsr;
use jodel_synth::{Bank, NoteOn, PatchId, Synth};
use std::path::{Path, PathBuf};

const RATE: u32 = 48_000;

/// A note: its time (seconds), patch, MIDI note, velocity and pan.
type Cue = (f32, PatchId, f32, f32, f32);

/// A percussive envelope: no attack, a decay of `decay` seconds to silence.
fn hit(decay: f32) -> Adsr {
    Adsr {
        attack: 0.0,
        decay,
        sustain: 0.0,
        release: 0.05,
    }
}

/// A demo: its name, what it is, its length, the hall's size, and its
/// patches and notes.
struct Demo {
    name: &'static str,
    what: &'static str,
    seconds: f32,
    room: f32,
    make: fn(&mut Bank) -> Vec<Cue>,
}

/// `samples` (interleaved stereo) as a 16-bit WAV.
fn wav(samples: &[f32]) -> Vec<u8> {
    let data = (samples.len() * 2) as u32;
    let mut b = Vec::with_capacity(44 + data as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&RATE.to_le_bytes());
    b.extend_from_slice(&(RATE * 4).to_le_bytes());
    b.extend_from_slice(&4u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data.to_le_bytes());
    for s in samples {
        b.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32_767.0).round() as i16).to_le_bytes());
    }
    b
}

/// `d` rendered, its patches' chorus sends returned as `chorus` says (none:
/// not at all).
fn render(d: &Demo, chorus: Option<ChorusSettings>) -> Vec<f32> {
    let mut bank = Bank::new();
    let cues = (d.make)(&mut bank);
    let mut synth = Synth::new(RATE, bank, 64);
    synth.set_tempo(120.0);
    synth.set_reverb(HallSettings {
        size: d.room,
        decay: 0.6 + 3.0 * d.room,
        wet: 0.35,
        ..HallSettings::default()
    });
    synth.set_delay(DelaySettings::default());
    synth.set_chorus(chorus.unwrap_or(ChorusSettings {
        mix: 0.0,
        ..ChorusSettings::default()
    }));
    for (k, (t, p, note, vel, pan)) in cues.into_iter().enumerate() {
        let frame = (t * RATE as f32).round() as u64;
        let n = NoteOn {
            note,
            velocity: vel,
            seed: 1 + k as u32,
            pan,
        };
        assert!(synth.note(frame, p, n, None));
    }
    let mut out = vec![0.0; 2 * (d.seconds * RATE as f32) as usize];
    synth.render_to(&mut out);
    out
}

/// Renders `demos` into `dir`, through `chorus` as [`render`] says.
fn write(demos: &[Demo], dir: &Path, chorus: Option<ChorusSettings>) {
    for d in demos {
        save(dir, d.name, d.what, &render(d, chorus));
    }
}

/// Writes `out` (interleaved stereo) into `dir` as `name`, printing its
/// summary.
fn save(dir: &Path, name: &str, what: &str, out: &[f32]) {
    std::fs::create_dir_all(dir).expect("the output folder");
    let mono: Vec<f32> = out.chunks_exact(2).map(|f| 0.5 * (f[0] + f[1])).collect();
    let path = dir.join(format!("{name}.wav"));
    durable::write_atomic(&path, &wav(out)).expect("the WAV written");
    let centroids: Vec<String> = measure::centroids(&mono, RATE as f32, 0.1)
        .iter()
        .map(|c| format!("{c:.0}"))
        .collect();
    let quarter = RATE as usize / 4;
    let peaks: Vec<String> = mono
        .chunks(quarter)
        .map(|c| format!("{:.0}", measure::db(measure::peak(c))))
        .collect();
    let (peak, rms) = (
        measure::db(measure::peak(out)),
        measure::db(measure::rms(out)),
    );
    println!(
        "{:<24} {:<48} {:.2} s  peak {:6.1} dBFS  rms {:6.1} dBFS  crest {:4.1} dB  jump {:.3}  clipped {}\n{:25}peak/0.25 s: {}\n{:25}centroid/0.1 s: {}",
        name,
        what,
        mono.len() as f32 / RATE as f32,
        peak,
        rms,
        peak - rms,
        measure::biggest_jump(&mono),
        out.iter().filter(|s| s.abs() > 1.0).count(),
        "",
        peaks.join(" "),
        "",
        centroids.join(" ")
    );
}

fn main() {
    let Some(root) = std::env::args().nth(1).map(PathBuf::from) else {
        eprintln!("usage: patches OUT_DIR (the sets go into its synth, synth-physical, synth-fm and synth-drums)");
        std::process::exit(2);
    };
    write(&game::DEMOS, &root.join("synth"), None);
    write(&physical::DEMOS, &root.join("synth-physical"), None);
    let fm_dir = root.join("synth-fm");
    write(&fm::DEMOS, &fm_dir, None);
    write(&fm::CHORUSED, &fm_dir, Some(fm::ENSEMBLE));
    for (name, what, out) in drums::processed() {
        save(&root.join("synth-drums"), &name, &what, &out);
    }
}
