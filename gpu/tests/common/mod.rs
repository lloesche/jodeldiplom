//! What the speech recogniser's tests hear with: the packed model, where
//! it is here, and pieces of a voice.

use std::f64::consts::TAU;
use std::path::PathBuf;

use jodel_lyrics::asr::packed::PackedModel;

/// The packed recogniser: `JODEL_ASR_MODEL`, else the player's data
/// folder's; `None` (with a note) where it is not.
pub fn model() -> Option<PackedModel> {
    let file = std::env::var_os("JODEL_ASR_MODEL")
        .map(PathBuf::from)
        .or_else(|| {
            let data = if cfg!(windows) {
                std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
            } else if cfg!(target_os = "macos") {
                std::env::var_os("HOME")
                    .map(|h| PathBuf::from(h).join("Library/Application Support"))
            } else {
                std::env::var_os("XDG_DATA_HOME")
                    .map(PathBuf::from)
                    .or_else(|| {
                        std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share"))
                    })
            }?;
            Some(data.join("Jodeldiplom Data/models/qwen3-asr-0.6b-q4.jdlasr"))
        });
    let model = file.and_then(|f| PackedModel::open(&f).ok());
    if model.is_none() {
        eprintln!("no speech recogniser here (JODEL_ASR_MODEL): skipped");
    }
    model
}

/// The recogniser on the GPU; `None` (with a note) where the model or a
/// usable GPU is not.
pub fn recogniser() -> Option<super::GpuAsr> {
    let model = model()?;
    match super::GpuAsr::new(&model) {
        Ok(asr) => Some(asr),
        Err(e) => {
            eprintln!("no usable GPU ({e}): skipped");
            None
        }
    }
}

/// `count` pieces of `seconds` (16 kHz mono): from the voice in
/// `JODEL_ASR_VOICE` (a 16 kHz mono 32-bit float WAV) where it is set, 7 s
/// apart from 20 s on; else sung vowels, a different pitch each.
pub fn pieces(count: usize, seconds: f64) -> Vec<Vec<f32>> {
    let len = (seconds * 16_000.0) as usize;
    if let Some(bytes) = std::env::var_os("JODEL_ASR_VOICE").and_then(|f| std::fs::read(f).ok()) {
        let voice: Vec<f32> = bytes[44..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let pieces: Vec<Vec<f32>> = (0..count)
            .map(|k| (20 + 7 * k) * 16_000)
            .filter(|&at| at + len <= voice.len())
            .map(|at| voice[at..at + len].to_vec())
            .collect();
        if pieces.len() == count {
            return pieces;
        }
    }
    (0..count)
        .map(|k| {
            let f0 = 110.0 + 35.0 * k as f64;
            let mut phase = 0.0;
            (0..len)
                .map(|i| {
                    let t = i as f64 / 16_000.0;
                    phase += f0 * (1.0 + 0.03 * (TAU * 5.0 * t).sin()) / 16_000.0;
                    // Syllables, three a second.
                    let syllable = (TAU * 1.5 * t).sin().abs();
                    let formant = 600.0 + 400.0 * (TAU * 0.7 * t + k as f64).sin();
                    let mut v = 0.0;
                    for h in 1..16 {
                        let f = f0 * h as f64;
                        let near = (-((f - formant) / 300.0).powi(2)).exp();
                        v += near * (TAU * phase * h as f64).sin();
                    }
                    (0.3 * syllable * v) as f32
                })
                .collect()
        })
        .collect()
}
