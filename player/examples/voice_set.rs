//! Training material for a voice cleaner: for each song listed, its voice
//! and its band (the mix less the voice), separated on the GPU, mono at
//! 16 kHz, 16-bit, as `OUT/{id}-voice.wav` and `OUT/{id}-band.wav`, and a
//! line `ID<TAB>SECONDS<TAB>FILE` in `OUT/index.tsv`. Songs already in the
//! index are passed over, so a run stopped goes on where it was.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example voice_set -- LIST OUT
//! ```

use std::collections::HashSet;
use std::fmt::Write;
use std::path::Path;
use std::time::Instant;

fn wav16(path: &Path, x: &[f32]) -> std::io::Result<()> {
    let pcm: Vec<i16> = x
        .iter()
        .map(|v| (v.clamp(-1.0, 1.0) * 32767.0).round() as i16)
        .collect();
    durable::write_atomic(path, &jodel_dsp::wav::pcm16(16_000, 1, &pcm))
}

/// A short id for `path`: 12 hex digits of FNV-1a.
fn id(path: &str) -> String {
    format!(
        "{:012x}",
        jodel_dsp::fnv1a(path.as_bytes()) & 0xffff_ffff_ffff
    )
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (list, out) = (&args[0], Path::new(&args[1]));
    let paths: Vec<String> = std::fs::read_to_string(list)
        .unwrap()
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect();
    // The index so far, written whole again after each song (one that
    // cannot be read stops the run: rewritten, its songs would be lost).
    // A song is done once it is in it, its two files written before.
    let index_file = out.join("index.tsv");
    let mut index = match std::fs::read_to_string(&index_file) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => panic!("{}: {e}", index_file.display()),
    };
    let done: HashSet<String> = index
        .lines()
        .filter_map(|l| l.split('\t').next().map(str::to_owned))
        .collect();
    let started = Instant::now();
    for (k, path) in paths.iter().enumerate() {
        let id = id(path);
        if done.contains(&id) {
            continue;
        }
        let (voice_file, band_file) = (
            out.join(format!("{id}-voice.wav")),
            out.join(format!("{id}-band.wav")),
        );
        let t = Instant::now();
        match jodeldiplom::lyrics::timer::separate_16k(Path::new(path), true) {
            Ok((voice, mix)) => {
                let n = voice.len().min(mix.len());
                let band: Vec<f32> = (0..n).map(|i| mix[i] - voice[i]).collect();
                wav16(&voice_file, &voice[..n]).unwrap();
                wav16(&band_file, &band).unwrap();
                writeln!(index, "{id}\t{:.1}\t{path}", n as f64 / 16_000.0).unwrap();
                durable::write_atomic(&index_file, index.as_bytes()).unwrap();
                eprintln!(
                    "{}/{} {id} {:.0} s in {:.1} s ({:.0} min so far)",
                    k + 1,
                    paths.len(),
                    n as f64 / 16_000.0,
                    t.elapsed().as_secs_f64(),
                    started.elapsed().as_secs_f64() / 60.0
                );
            }
            Err(e) => eprintln!("{}/{} {path}: {e}", k + 1, paths.len()),
        }
    }
}
