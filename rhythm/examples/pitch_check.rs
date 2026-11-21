//! Checks the pitch network against the original's values (`ref_fcpe.py`
//! writes them, out of this repository): our spectrogram against the
//! authors', the network layer by layer on their spectrogram, the pitch
//! the whole way. With `--weights FILE` it runs those weights (the full
//! precision ones, say) instead of the built-in ones.
//!
//! ```text
//! cargo run --release -p jodel-rhythm --example pitch_check -- [--weights FILE] REF_DIR...
//! ```

use std::path::Path;
use std::time::Instant;

use jodel_rhythm::net::weights::Weights;
use jodel_rhythm::pitch::{spectrogram, Pitch, PitchNet, BANDS};

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

/// How two pitch tracks agree: frames voiced in one only, and the cents
/// between them where both are (median, 99th percentile, and the share
/// more than 50 cents apart).
fn compare(got: &Pitch, want: &[f32]) -> String {
    let mut apart = Vec::new();
    let mut disagree = 0;
    for (&a, &b) in got.hz.iter().zip(want) {
        match (a > 0.0, b > 0.0) {
            (true, true) => apart.push((1200.0 * (a / b).log2()).abs()),
            (false, false) => {}
            _ => disagree += 1,
        }
    }
    apart.sort_by(f32::total_cmp);
    let at = |q: f64| {
        apart
            .get(((apart.len() as f64 - 1.0) * q) as usize)
            .copied()
            .unwrap_or(0.0)
    };
    let far = apart.iter().filter(|&&c| c > 50.0).count();
    format!(
        "{} voiced in one only of {}; cents apart: median {:.3}, 99 % {:.2}, over 50: {}",
        disagree,
        want.len(),
        at(0.5),
        at(0.99),
        far
    )
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let net: &PitchNet = if args.first().map(String::as_str) == Some("--weights") {
        let file = args.remove(1);
        args.remove(0);
        let bytes = std::fs::read(&file).unwrap_or_else(|e| panic!("{file}: {e}"));
        Box::leak(Box::new(
            PitchNet::load(Weights::parse(&bytes).expect("weights")).expect("the network"),
        ))
    } else {
        PitchNet::built_in()
    };
    for dir in &args {
        let dir = Path::new(dir);
        println!("{}", dir.display());
        let wav = read(&dir.join("wav16k.f32"));
        let want_mel = read(&dir.join("mel.f32"));
        let started = Instant::now();
        let (mel, frames) = spectrogram(&wav);
        println!(
            "  spectrogram: {frames} frames (theirs {}), {:.2?}, largest difference {:.2e}",
            want_mel.len() / BANDS,
            started.elapsed(),
            error(&mel, &want_mel)
        );
        let started = Instant::now();
        let (latent, steps) = net.run(&want_mel, frames, true);
        let took = started.elapsed();
        let names = [
            "stack", "layer0", "layer1", "layer2", "layer3", "layer4", "layer5",
        ];
        for (name, got) in names.iter().zip(&steps) {
            println!(
                "  {name}: {:.2e}",
                error(got, &read(&dir.join(format!("{name}.f32"))))
            );
        }
        println!(
            "  latent: {:.2e} ({took:.2?} for {:.1} s)",
            error(&latent, &read(&dir.join("latent.f32"))),
            wav.len() as f64 / 16_000.0
        );
        let want_f0 = read(&dir.join("f0.f32"));
        println!(
            "  on their spectrogram: {}",
            compare(&net.decode(&latent), &want_f0)
        );
        let started = Instant::now();
        let (own, _) = net.run(&mel, frames, false);
        let took = started.elapsed();
        println!(
            "  in windows against at once: {:.2e} ({took:.2?})",
            error(&own, &net.run(&mel, frames, true).0)
        );
        println!("  the whole way: {}", compare(&net.decode(&own), &want_f0));
    }
}
