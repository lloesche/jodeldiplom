//! Checks the CPU Qwen3-ASR against reference outputs made with the
//! official model in PyTorch (out of the repository): the mel filters, the
//! features, the encoder's input to its first layer and after it, its
//! output, the first token's logits (as the largest difference relative to
//! the largest value), and the tokens written greedily.
//!
//! ```text
//! cargo run --release -p jodel-lyrics --example asr_check -- MODEL_DIR REF_DIR...
//! ```
//!
//! `MODEL_DIR` holds `model.safetensors`, `vocab.json` and
//! `tokenizer_config.json`. Each `REF_DIR` holds `wav.f32`,
//! `mel_filters.f32`, `features.f32`, `conv.f32`, `layer0.f32`,
//! `audio.f32`, `logits0.f32` (raw little-endian f32) and `meta.json`
//! (with `generated_ids`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use jodel_lyrics::asr::{argmax, mel, text, Qwen3Asr};
use jodel_lyrics::json::Json;

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
    let started = Instant::now();
    let model = Qwen3Asr::open(Path::new(&args[0])).unwrap();
    println!("loaded in {:.2?}", started.elapsed());
    for dir in &args[1..] {
        let dir = PathBuf::from(dir);
        let at = |name: &str| read(&dir.join(format!("{name}.f32")));
        let meta = Json::parse(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
        let want_ids: Vec<u32> = meta
            .get("generated_ids")
            .and_then(Json::as_array)
            .unwrap()
            .iter()
            .filter_map(|v| v.as_f64().map(|v| v as u32))
            .filter(|&i| !text::is_end(i))
            .collect();
        let voice = at("wav");
        println!("{}: {:.1} s", dir.display(), voice.len() as f64 / 16_000.0);
        println!(
            "  mel filters   {:.2e}",
            off(&mel::filters(), &at("mel_filters"))
        );
        let t = Instant::now();
        let (features, frames) = model.features(&voice);
        println!(
            "  features      {:.2e}  ({frames} frames, {:.2?})",
            off(&features, &at("features")),
            t.elapsed()
        );
        let t = Instant::now();
        let conv = model.embed_audio(&features, frames);
        println!(
            "  conv          {:.2e}  ({:.2?})",
            off(&conv, &at("conv")),
            t.elapsed()
        );
        let layer0 = model.encoder_layers(conv, frames, 1);
        println!("  layer 0       {:.2e}", off(&layer0, &at("layer0")));
        let t = Instant::now();
        let audio = model.encode(&features, frames);
        println!(
            "  audio         {:.2e}  ({:.2?})",
            off(&audio, &at("audio")),
            t.elapsed()
        );
        let t = Instant::now();
        let logits = model.first_logits(&voice);
        let want = at("logits0");
        println!(
            "  logits0       {:.2e}  (top {} want {}, {:.2?})",
            off(&logits, &want),
            argmax(&logits),
            argmax(&want),
            t.elapsed()
        );
        let t = Instant::now();
        let got = model.transcribe(&voice, want_ids.len() + 8);
        let same = got
            .ids
            .iter()
            .zip(&want_ids)
            .take_while(|(a, b)| a == b)
            .count();
        println!(
            "  tokens        {same} of {} agree before the first difference ({} written, {:.2?})",
            want_ids.len(),
            got.ids.len(),
            t.elapsed()
        );
        println!("  [{}] {}", got.language, got.text);
    }
}
