//! Checks the CPU wav2vec2 against reference outputs made with PyTorch
//! (out of the repository): for each clip, the conv features, the hidden
//! states before and after the transformer, and the logits, as the largest
//! difference relative to the largest value, and how often the most likely
//! token agrees.
//!
//! ```text
//! cargo run --release -p jodel-lyrics --example speech_check -- MODEL.safetensors REF_DIR CLIP...
//! ```
//!
//! `REF_DIR` holds `CLIP.f32` (the voice at 16 kHz) and
//! `CLIP.{features,hidden0,hidden12,logits}.f32`, raw little-endian f32.

use std::path::{Path, PathBuf};
use std::time::Instant;

use jodel_lyrics::speech::Wav2Vec2;

fn read(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// The largest difference relative to the largest reference value.
fn off(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "lengths");
    let scale = want.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
    let worst = got.iter().zip(want).fold(0.0f64, |m, (&a, &b)| {
        m.max((f64::from(a) - f64::from(b)).abs())
    });
    worst / scale.max(1e-12)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model_path = PathBuf::from(&args[0]);
    let dir = PathBuf::from(&args[1]);
    let started = Instant::now();
    let model = Wav2Vec2::open(&model_path).unwrap();
    println!("loaded in {:.2?}", started.elapsed());
    for clip in &args[2..] {
        let voice = read(&dir.join(format!("{clip}.f32")));
        let at = |stage: &str| read(&dir.join(format!("{clip}.{stage}.f32")));
        let started = Instant::now();
        let levels = model.levels(&voice);
        let features = model.features(&levels.normalise(&voice), &levels);
        let h0 = model.embed(features.clone());
        let h12 = model.encode(h0.clone());
        let logits = model.logits(&voice);
        let took = started.elapsed();
        let want = at("logits");
        let frames = want.len() / 32;
        let argmax = |l: &[f32], f: usize| {
            (0..32)
                .max_by(|&a, &b| l[f * 32 + a].total_cmp(&l[f * 32 + b]))
                .unwrap()
        };
        let agree = (0..frames)
            .filter(|&f| argmax(&logits, f) == argmax(&want, f))
            .count();
        println!(
            "{clip}: {} s, {frames} frames in {took:.2?} (twice the network): features {:.2e}, hidden0 {:.2e}, hidden12 {:.2e}, logits {:.2e}, best token agrees {agree}/{frames}",
            voice.len() as f64 / 16_000.0,
            off(&features, &at("features")),
            off(&h0, &at("hidden0")),
            off(&h12, &at("hidden12")),
            off(&logits, &want),
        );
    }
}
