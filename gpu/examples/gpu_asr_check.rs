//! Checks the GPU's Qwen3-ASR (4-bit weights) against the CPU's (the
//! weights rounded the same way) and the official model's tokens: the
//! encoder's output, then the words, and how long each took.
//!
//! ```text
//! cargo run --release -p jodel-gpu --example gpu_asr_check -- MODEL_DIR REF_DIR...
//! ```
//!
//! `REF_DIR`s as `jodel-lyrics`'s `asr_check` takes them (`wav.f32`,
//! `meta.json`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use jodel_gpu::GpuAsr;
use jodel_lyrics::asr::packed::PackedModel;
use jodel_lyrics::asr::{text, Qwen3Asr};
use jodel_tensors::json::Json;

fn read(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
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
    let started = Instant::now();
    let mut cpu = Qwen3Asr::open(Path::new(&args[0])).unwrap();
    let packed = PackedModel::from_checkpoint(Path::new(&args[0]), |_| 4, None, |_, _| {});
    let gpu = GpuAsr::new(&packed.unwrap()).unwrap();
    println!("{}: loaded in {:.2?}", gpu.name(), started.elapsed());
    cpu.round_weights(4, 32, false);
    for dir in &args[1..] {
        let dir = PathBuf::from(dir);
        let voice = read(&dir.join("wav.f32"));
        let meta = Json::parse(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
        let want: Vec<u32> = meta
            .get("generated_ids")
            .and_then(Json::as_array)
            .unwrap()
            .iter()
            .filter_map(|v| v.as_f64().map(|v| v as u32))
            .filter(|&i| !text::is_end(i))
            .collect();
        println!("{}: {:.1} s", dir.display(), voice.len() as f64 / 16_000.0);
        let t = Instant::now();
        let audio = gpu.audio(&voice).unwrap();
        let gpu_took = t.elapsed();
        let (features, frames) = cpu.features(&voice);
        let t = Instant::now();
        let cpu_audio = cpu.encode(&features, frames);
        println!(
            "  audio   {:.2e} against the CPU's (GPU {gpu_took:.2?}, CPU {:.2?})",
            off(&audio, &cpu_audio),
            t.elapsed()
        );
        let t = Instant::now();
        let got = gpu.transcribe(&voice).unwrap();
        let took = t.elapsed();
        let same = got
            .ids
            .iter()
            .zip(&want)
            .take_while(|(a, b)| a == b)
            .count();
        println!(
            "  tokens  {same} of {} as the official model's before a difference ({} written, {took:.2?})",
            want.len(),
            got.ids.len()
        );
        println!("  [{}] {}", got.language, got.text);
        let t = Instant::now();
        let c = cpu.transcribe(&voice, got.ids.len() + 8);
        let same = got
            .ids
            .iter()
            .zip(&c.ids)
            .take_while(|(a, b)| a == b)
            .count();
        println!(
            "  CPU 4-bit: {same} of {} the same ({:.2?}): [{}] {}",
            c.ids.len(),
            t.elapsed(),
            c.language,
            c.text
        );
    }
}
