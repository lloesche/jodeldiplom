//! The fit check on real files: LRCLIB's text for each, what the speech
//! recogniser hears in its voice, how well that text and LRCLIB's others
//! fit it, and which the player would time.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example fit_check -- FILE...
//! ```
//!
//! Uses the speech recogniser where the player finds it
//! (`lyrics::asr_model_path`), and the GPU.

use std::path::Path;
use std::time::{Duration, Instant};

use jodel_lyrics::fit::lyrics_fit;
use jodel_lyrics::{Found, Lyrics};
use jodeldiplom::lyrics::{asr_model_path, fitting, lrclib_of, separate_16k, track_of};

fn first(l: &Lyrics) -> String {
    l.lines.first().map(|l| l.text()).unwrap_or_default()
}

fn main() {
    let asr = {
        let path = asr_model_path().expect("no speech recogniser found");
        let packed = jodel_lyrics::asr::packed::PackedModel::open(&path).unwrap();
        jodel_gpu::GpuAsr::new(&packed).unwrap()
    };
    let lrclib = lrclib_of();
    let wait = |pause: Duration| {
        std::thread::sleep(pause);
        true
    };
    for file in std::env::args().skip(1) {
        let path = Path::new(&file);
        let Some(track) = track_of(path) else {
            println!("{file}: cannot be read");
            continue;
        };
        let found = match lrclib.fetch(&track, &wait) {
            Ok(Some(record)) => match record.found() {
                Found::Lyrics(l) => l,
                other => {
                    println!("{file}: LRCLIB: {other:?}");
                    continue;
                }
            },
            other => {
                println!("{file}: LRCLIB: {other:?}");
                continue;
            }
        };
        let t = Instant::now();
        let (voice, _) = separate_16k(path, false).unwrap();
        let separated = t.elapsed();
        let t = Instant::now();
        let heard = asr.transcribe(&voice).unwrap();
        let heard_in = t.elapsed();
        println!(
            "{file}\n  voice {separated:.1?}, heard in {heard_in:.1?}: {}…",
            heard.text.chars().take(100).collect::<String>()
        );
        println!(
            "  found   fit {:4.0} %  {}",
            100.0 * lyrics_fit(&found, &heard.text),
            first(&found)
        );
        let others = lrclib.candidates(&track, &wait).unwrap_or_default();
        for o in &others {
            println!(
                "  other   fit {:4.0} %  {}",
                100.0 * lyrics_fit(o, &heard.text),
                first(o)
            );
        }
        let (chosen, fit) = fitting(&found, &heard.text, &|| others.clone());
        println!(
            "  timed:  {} ({})",
            first(&chosen),
            fit.map_or("not judged".into(), |f| format!("fit {}", f.to_text()))
        );
    }
}
