//! Whole voices through the GPU's Qwen3-ASR: for each 16 kHz mono 32-bit
//! float WAV file, a line `FILE<TAB>SECONDS<TAB>TOKENS<TAB>[LANGUAGE] TEXT`.
//! `MODEL` is a packed file (`asr_pack`'s) or the checkpoint's folder, packed
//! on the way (the bits each matrix gets by `JODEL_ASR_BITS`, as
//! `quant::bits_by` reads it, 4 for all by default; the 4-bit ones weighted by
//! the calibration statistics in the file `JODEL_ASR_STATS` names, if any).
//!
//! ```text
//! cargo run --release -p jodel-gpu --example gpu_asr_songs -- MODEL VOICE.wav...
//! ```

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use jodel_gpu::GpuAsr;
use jodel_lyrics::asr::packed::PackedModel;
use jodel_lyrics::asr::Qwen3Asr;

fn read_wav(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes[44..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let started = Instant::now();
    let model = Path::new(&args[0]);
    let packed = if model.is_dir() {
        let cpu = Qwen3Asr::open(model).unwrap();
        let spec = std::env::var("JODEL_ASR_BITS").unwrap_or_else(|_| "4".into());
        let stats = std::env::var_os("JODEL_ASR_STATS").map(|p| {
            jodel_lyrics::asr::quant::Statistics::from_bytes(&std::fs::read(p).unwrap()).unwrap()
        });
        PackedModel::from_model(
            &cpu,
            jodel_lyrics::asr::quant::bits_by(&spec),
            stats.as_ref(),
        )
    } else {
        PackedModel::open(model).unwrap()
    };
    let gpu = GpuAsr::new(&packed).unwrap();
    drop(packed);
    eprintln!("{}: loaded in {:.2?}", gpu.name(), started.elapsed());
    let mut out = std::io::stdout().lock();
    for file in &args[1..] {
        let voice = read_wav(Path::new(file));
        let t = Instant::now();
        let got = gpu.transcribe(&voice).unwrap();
        writeln!(
            out,
            "{file}\t{:.2}\t{}\t[{}] {}",
            t.elapsed().as_secs_f64(),
            got.ids.len(),
            got.language,
            got.text.replace(['\t', '\n'], " ")
        )
        .unwrap();
        out.flush().unwrap();
    }
}
