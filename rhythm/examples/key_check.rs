//! Checks the key finder against the original's values (`ref_key.py`
//! writes them, out of this repository): our spectrum against librosa's,
//! the network layer by layer on librosa's spectrum, and the whole way.
//!
//! ```text
//! cargo run --release -p jodel-rhythm --example key_check -- REF_DIR...
//! ```

use std::path::Path;

use jodel_rhythm::key::{key_from, spectrum, KeyNet, BINS};

fn read(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// The largest difference against the largest magnitude of `want`.
fn error(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let top = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-12);
    got.iter()
        .zip(want)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
        / top
}

fn main() {
    let net = KeyNet::built_in();
    for dir in std::env::args().skip(1) {
        let dir = Path::new(&dir);
        let wav = read(&dir.join("wav.f32"));
        let want_cqt = read(&dir.join("cqt.f32"));
        let started = std::time::Instant::now();
        let (cqt, frames) = spectrum(&wav);
        let took = started.elapsed();
        println!("{}", dir.display());
        println!(
            "  spectrum: {frames} frames (librosa {}), {took:.2?}, largest difference {:.2e} of the largest value",
            want_cqt.len() / BINS,
            error(&cqt, &want_cqt)
        );
        // Bins where it differs most, as dB.
        let mut worst = (0, 0.0f32);
        for b in 0..BINS {
            let (g, w) = (
                &cqt[b * frames..(b + 1) * frames],
                &want_cqt[b * frames..(b + 1) * frames],
            );
            let e = error(g, w);
            if e > worst.1 {
                worst = (b, e);
            }
        }
        println!(
            "  the bin furthest off: {} ({:.2e} of its largest)",
            worst.0, worst.1
        );
        // The network on librosa's own input.
        let input = read(&dir.join("input.f32"));
        let t = input.len() / BINS;
        let (scores, steps) = net.run(&input, t);
        let names = [
            "conv1", "conv2", "pool1", "conv3", "conv4", "pool2", "conv5", "conv6", "pool3",
            "conv7", "conv8", "conv9",
        ];
        for (name, got) in names.iter().zip(&steps) {
            let want = read(&dir.join(format!("{name}.f32")));
            println!("  {name}: {:.2e}", error(got, &want));
        }
        let logits = read(&dir.join("logits.f32"));
        let want: [f32; 24] = logits.try_into().unwrap();
        println!(
            "  scores: {:.2e}; the original's key {} ({}), ours on its spectrum {}",
            error(&scores, &want),
            key_from(&want).name(),
            key_from(&want).camelot(),
            key_from(&scores).name()
        );
        // The whole way, our spectrum.
        let mine: Vec<f32> = (0..BINS)
            .flat_map(|b| cqt[b * frames..b * frames + t].to_vec())
            .collect();
        let (own, _) = net.run(&mine, t);
        let key = key_from(&own);
        println!(
            "  ours the whole way: {} ({}, {:.0} % sure); scores off by {:.2e}",
            key.name(),
            key.camelot(),
            100.0 * key.confidence,
            error(&own, &want)
        );
    }
}
