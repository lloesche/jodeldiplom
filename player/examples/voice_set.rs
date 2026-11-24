//! Training material for a voice cleaner: for each song listed, its voice
//! and its band (the mix less the voice), separated on the GPU, mono at
//! 16 kHz, 16-bit, as `OUT/{id}-voice.wav` and `OUT/{id}-band.wav`, and a
//! line `ID<TAB>SECONDS<TAB>FILE` in `OUT/index.tsv`. Songs already there
//! are passed over, so a run stopped goes on where it was.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example voice_set -- LIST OUT
//! ```

use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn wav16(path: &Path, x: &[f32]) -> std::io::Result<()> {
    let data = (2 * x.len()) as u32;
    let mut b = Vec::with_capacity(44 + 2 * x.len());
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&16_000u32.to_le_bytes());
    b.extend_from_slice(&32_000u32.to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data.to_le_bytes());
    for v in x {
        b.extend_from_slice(&((v.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, b)?;
    std::fs::rename(tmp, path)
}

/// A short id for `path`: 12 hex digits of FNV-1a.
fn id(path: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in path.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{:012x}", h & 0xffff_ffff_ffff)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (list, out) = (&args[0], Path::new(&args[1]));
    std::fs::create_dir_all(out).unwrap();
    let paths: Vec<String> = std::fs::read_to_string(list)
        .unwrap()
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect();
    let mut index = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out.join("index.tsv"))
        .unwrap();
    let started = Instant::now();
    for (k, path) in paths.iter().enumerate() {
        let id = id(path);
        let (voice_file, band_file) = (
            out.join(format!("{id}-voice.wav")),
            out.join(format!("{id}-band.wav")),
        );
        if voice_file.exists() && band_file.exists() {
            continue;
        }
        let t = Instant::now();
        match jodeldiplom::lyrics::timer::separate_16k(Path::new(path), true) {
            Ok((voice, mix)) => {
                let n = voice.len().min(mix.len());
                let band: Vec<f32> = (0..n).map(|i| mix[i] - voice[i]).collect();
                wav16(&voice_file, &voice[..n]).unwrap();
                wav16(&band_file, &band).unwrap();
                writeln!(index, "{id}\t{:.1}\t{path}", n as f64 / 16_000.0).unwrap();
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
