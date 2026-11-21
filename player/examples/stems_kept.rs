//! A track's stems kept as the queue keeps them, and what that costs:
//!
//! ```text
//! cargo run --release -p jodeldiplom --example stems_kept -- FILE [OUT.stems]
//! ```
//!
//! The file is analysed with stems as the player gets a track ready, its
//! stems written to OUT.stems (a temporary file without it). Then: how
//! long that took, how big the file is against the stems as plain 16 bits,
//! how long reading all of it back takes, and how far the kept stems are
//! from separating the track again live (as the engine did before stems
//! were kept).

use std::path::PathBuf;
use std::time::Instant;

use jodel_playback::stems::{Stem, StereoFeed, RATE, STEMS};
use jodel_playback::{Source, StemFile};
use jodel_rhythm::stems::Splitter;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let file: PathBuf = args.next().ok_or("give a FILE")?.into();
    let out: PathBuf = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("jodel-stems-kept.stems"));
    let gpu = jodel_gpu::GpuSeparator::shared().ok_or("no GPU for the stems")?;
    let recipe = jodeldiplom::core::RhythmRecipe {
        methods: jodel_rhythm::Method::ALL.to_vec(),
        stems: true,
    };
    let started = Instant::now();
    let listened = jodeldiplom::rhythm::listen_keeping(&file, &recipe, Some(out.clone()))?;
    println!(
        "analysed with stems in {:.1} s (stems {:.1} s){}",
        started.elapsed().as_secs_f64(),
        listened.timings.stems.map_or(0.0, |d| d.as_secs_f64()),
        listened
            .stems_failed
            .as_ref()
            .map_or(String::new(), |e| format!(", stems failed: {e}"))
    );
    let mut kept = StemFile::open(&out).map_err(|e| e.to_string())?;
    let size = std::fs::metadata(&out).map_err(|e| e.to_string())?.len();
    let frames = kept.frames();
    let plain = frames * STEMS as u64 * 2 * 2;
    println!(
        "{}: {:.1} s of stems, {:.1} MB ({:.0} % of plain 16 bits, {:.1} MB a minute)",
        out.display(),
        frames as f64 / f64::from(RATE),
        size as f64 / 1e6,
        100.0 * size as f64 / plain as f64,
        size as f64 / 1e6 / (frames as f64 / f64::from(RATE) / 60.0)
    );
    let reading = Instant::now();
    let mut all: Vec<[Vec<f32>; 2]> = vec![[Vec::new(), Vec::new()]; STEMS];
    let mut at = 0;
    while at < frames {
        let part = kept
            .read(at, RATE as usize * 10)
            .map_err(|e| e.to_string())?;
        at += part[0][0].len() as u64;
        for (s, [l, r]) in part.into_iter().enumerate() {
            all[s][0].extend(l);
            all[s][1].extend(r);
        }
    }
    let read = reading.elapsed().as_secs_f64();
    println!(
        "read back whole in {:.2} s: {:.0} times as fast as it plays",
        read,
        frames as f64 / f64::from(RATE) / read.max(1e-6)
    );
    // Separated again, live from the start, as a stems run did.
    let separating = Instant::now();
    let mut source = Source::open(&file).map_err(|e| e.to_string())?;
    let channels = usize::from(source.info().channels).max(1);
    let mut feed = StereoFeed::new(source.info().rate);
    let mut splitter = Splitter::new(gpu);
    let mut live: Vec<[Vec<f32>; 2]> = vec![[Vec::new(), Vec::new()]; STEMS];
    let mut failed = None;
    {
        let mut take = |b: &jodel_rhythm::stems::StemBlock| {
            for (s, [l, r]) in b.stems.iter().enumerate() {
                live[s][0].extend_from_slice(l);
                live[s][1].extend_from_slice(r);
            }
        };
        let mut put = |l: Vec<f32>, r: Vec<f32>| {
            if let Err(e) = splitter.push(&l, &r, &mut take) {
                failed.get_or_insert(e);
            }
        };
        let mut block = Vec::new();
        loop {
            block.clear();
            if source
                .read_unquantized(&mut block)
                .map_err(|e| e.to_string())?
                == 0
            {
                break;
            }
            feed.push(&block, channels, &mut put);
        }
        feed.finish(&mut put);
    }
    if let Some(e) = failed {
        return Err(e);
    }
    {
        let mut take = |b: &jodel_rhythm::stems::StemBlock| {
            for (s, [l, r]) in b.stems.iter().enumerate() {
                live[s][0].extend_from_slice(l);
                live[s][1].extend_from_slice(r);
            }
        };
        splitter.finish(&mut take)?;
    }
    println!(
        "separated again live in {:.1} s",
        separating.elapsed().as_secs_f64()
    );
    for (s, stem) in Stem::ALL.iter().enumerate() {
        let (mut worst, mut power, mut error) = (0.0f64, 0.0f64, 0.0f64);
        for ch in 0..2 {
            for (a, b) in all[s][ch].iter().zip(&live[s][ch]) {
                let d = f64::from(*a - *b);
                worst = worst.max(d.abs());
                error += d * d;
                power += f64::from(*b) * f64::from(*b);
            }
        }
        let snr = 10.0 * (power.max(1e-20) / error.max(1e-30)).log10();
        println!(
            "  {stem:?}: largest difference {:.1} dBFS, {snr:.0} dB below the stem",
            20.0 * worst.max(1e-12).log10()
        );
    }
    Ok(())
}
