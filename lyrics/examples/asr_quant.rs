//! What quantising Qwen3-ASR's weights costs: clips transcribed with the
//! weights as they come, then rounded each way named, as text a line each
//! (`SCHEME<TAB>CLIP<TAB>SECONDS<TAB>TEXT`) for comparing outside.
//!
//! ```text
//! cargo run --release -p jodel-lyrics --example asr_quant -- MODEL_DIR SCHEMES VOICE.wav:START:SECONDS...
//! ```
//!
//! `SCHEMES` is a comma-separated list of `full` and `BITS[s|a]GROUP`
//! (`4s32`: 4 bits, symmetric, groups of 32; `4a32`: with a minimum too).
//! The voices are 16 kHz mono 32-bit float WAV files.

use std::path::Path;
use std::time::Instant;

use jodel_lyrics::asr::Qwen3Asr;

/// A 16 kHz mono 32-bit float WAV file's samples from `start` for
/// `seconds`.
fn clip(path: &Path, start: f64, seconds: f64) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let all: Vec<f32> = bytes[44..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let a = ((start * 16_000.0) as usize).min(all.len());
    let b = ((a as f64 + seconds * 16_000.0) as usize).min(all.len());
    all[a..b].to_vec()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = Path::new(&args[0]);
    let clips: Vec<(String, Vec<f32>)> = args[2..]
        .iter()
        .map(|c| {
            let mut f = c.rsplitn(3, ':');
            let seconds: f64 = f.next().unwrap().parse().unwrap();
            let start: f64 = f.next().unwrap().parse().unwrap();
            let path = f.next().unwrap();
            (c.clone(), clip(Path::new(path), start, seconds))
        })
        .collect();
    for scheme in args[1].split(',') {
        let mut model = Qwen3Asr::open(dir).unwrap();
        if scheme != "full" {
            let symmetric = scheme.contains('s');
            let (bits, group) = scheme.split_once(['s', 'a']).unwrap();
            model.round_weights(bits.parse().unwrap(), group.parse().unwrap(), symmetric);
        }
        for (name, voice) in &clips {
            let t = Instant::now();
            let got = model.transcribe(voice, 400);
            println!(
                "{scheme}\t{name}\t{:.1}\t[{}] {}",
                t.elapsed().as_secs_f64(),
                got.language,
                got.text
            );
        }
    }
}
