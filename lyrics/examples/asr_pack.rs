//! Packs the recogniser into our own file ([`jodel_lyrics::asr::packed`]):
//! its matrices at 4 bits (or the bits `JODEL_ASR_BITS` gives them, as
//! `quant::bits_by` reads it), weighted by the calibration statistics in the
//! file `JODEL_ASR_STATS` names (the built-in ones if none), the rest as f32,
//! and the vocabulary.
//!
//! ```text
//! cargo run --release -p jodel-lyrics --example asr_pack -- MODEL_DIR OUT.jdlasr
//! ```

use std::path::Path;
use std::time::Instant;

use jodel_lyrics::asr::packed::PackedModel;
use jodel_lyrics::asr::Qwen3Asr;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let started = Instant::now();
    let model = Qwen3Asr::open(Path::new(&args[0])).unwrap();
    let spec = std::env::var("JODEL_ASR_BITS").unwrap_or_else(|_| "4".into());
    let stats = match std::env::var_os("JODEL_ASR_STATS") {
        Some(p) => {
            jodel_lyrics::asr::quant::Statistics::from_bytes(&std::fs::read(p).unwrap()).unwrap()
        }
        None => jodel_lyrics::asr::quant::Statistics::built_in(),
    };
    let packed = PackedModel::from_model(
        &model,
        jodel_lyrics::asr::quant::bits_by(&spec),
        Some(&stats),
    );
    packed.write(Path::new(&args[1])).unwrap();
    let size = std::fs::metadata(&args[1]).map_or(0, |m| m.len());
    println!(
        "{}: {:.1} MB in {:.2?}",
        args[1],
        size as f64 / 1e6,
        started.elapsed()
    );
}
