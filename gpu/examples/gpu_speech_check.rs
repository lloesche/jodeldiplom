//! Checks the speech model on the GPU against the CPU's (and both against
//! PyTorch's logits, made out of the repository), and times a whole song's
//! worth of voice on each.
//!
//! ```text
//! cargo run --release -p jodel-gpu --example gpu_speech_check -- MODEL.safetensors REF_DIR CLIP...
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use jodel_gpu::GpuSpeech;
use jodel_lyrics::ctc::Acoustic;
use jodel_lyrics::speech::Wav2Vec2;

fn read(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// The largest difference relative to the largest value of `want`.
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
    let dir = PathBuf::from(&args[1]);
    let started = Instant::now();
    let cpu = Wav2Vec2::open(Path::new(&args[0])).unwrap();
    let gpu = GpuSpeech::new(&cpu).unwrap();
    println!("set up on {} in {:.2?}", gpu.name(), started.elapsed());
    let mut longest = Vec::new();
    for clip in &args[2..] {
        let voice = read(&dir.join(format!("{clip}.f32")));
        let reference = read(&dir.join(format!("{clip}.logits.f32")));
        let t = Instant::now();
        let on_gpu = gpu.logits(&voice).unwrap();
        let gpu_took = t.elapsed();
        let t = Instant::now();
        let on_cpu = cpu.logits(&voice);
        let cpu_took = t.elapsed();
        println!(
            "{clip}: GPU {gpu_took:.2?} vs CPU {cpu_took:.2?}; GPU against CPU {:.2e}, against PyTorch {:.2e} (CPU against PyTorch {:.2e})",
            off(&on_gpu, &on_cpu),
            off(&on_gpu, &reference),
            off(&on_cpu, &reference)
        );
        longest = voice;
    }
    // Four minutes: the last clip again and again, through the pieces.
    let song: Vec<f32> = longest.iter().copied().cycle().take(16_000 * 240).collect();
    let t = Instant::now();
    let on_gpu = gpu.emissions(&song).unwrap();
    let gpu_took = t.elapsed();
    let t = Instant::now();
    let on_cpu = cpu.emissions(&song).unwrap();
    let cpu_took = t.elapsed();
    println!(
        "4 minutes: GPU {gpu_took:.2?}, CPU {cpu_took:.2?}; emissions apart by {:.2e} at most",
        off(&on_gpu.logp, &on_cpu.logp)
    );
}
