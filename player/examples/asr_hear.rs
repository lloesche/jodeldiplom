//! The speech recogniser over any number of songs, for calibrating and
//! measuring its quantisation.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example asr_hear -- --pick N --from LIST [--seed S] [--not LIST]...
//! cargo run --release -p jodeldiplom --example asr_hear -- MODEL LIST OUT.tsv [--voices DIR] [--stats FILE]
//! ```
//!
//! - `--pick` prints `N` songs from the files listed in `LIST` (a path a
//!   line), at random but by `--seed` (1), a song an artist (the folder
//!   above the album's) round after round, none of those in the `--not`n//!   lists (nor of their artists).
//! - Else each song of `LIST` is heard by `MODEL` (a packed file, or the
//!   checkpoint's folder packed on the way: the bits by `JODEL_ASR_BITS`, the
//!   4-bit rounding weighted by the statistics `JODEL_ASR_STATS` names), a
//!   line `FILE<TAB>SECONDS<TAB>TOKENS<TAB>LANGUAGE<TAB>TEXT` added to
//!   `OUT.tsv` for each. Songs already in it are passed over, so a run
//!   stopped goes on where it was.
//! - `--voices DIR` keeps each song's separated voice there (16 kHz, 16-bit),
//!   so a second model hears it without separating it again.
//! - `--mix` hears the whole mix instead of the separated voice (for speech:
//!   audio plays and the like), `--most SECONDS` only its start.
//! - `--stats FILE` collects the calibration statistics (each matrix's input
//!   columns' sums of squares, `jodel_lyrics::asr::quant::Statistics`) and adds
//!   them into FILE every ten songs and at the end.

use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use jodel_gpu::GpuAsr;
use jodel_lyrics::asr::packed::PackedModel;
use jodel_lyrics::asr::quant::{bits_by, Statistics};

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn lines(path: &str) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .lines()
        .map(|l| l.trim().trim_start_matches('\u{feff}').to_owned())
        .filter(|l| !l.is_empty())
        .collect()
}

/// The artist of a library file: the folder above its album's (the one
/// above a disc's folder, for a set of discs).
fn artist(path: &str) -> String {
    let p = Path::new(path);
    let mut album = p.parent();
    if album
        .and_then(Path::file_name)
        .and_then(|n| n.to_str())
        .is_some_and(|n| {
            let n = n.to_lowercase();
            (n.starts_with("cd") || n.starts_with("disc") || n.starts_with("disk"))
                && n.chars().any(|c| c.is_ascii_digit())
        })
    {
        album = album.and_then(Path::parent);
    }
    album
        .and_then(Path::parent)
        .and_then(Path::file_name)
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
}

/// `--pick`: `n` songs of `from`, one an artist, none of `not`'s.
fn pick(n: usize, from: &str, seed: u64, not: &[String]) {
    let skip: HashSet<String> = not.iter().flat_map(|f| lines(f)).collect();
    let skip_artists: HashSet<String> = skip.iter().map(|p| artist(p)).collect();
    let mut by_artist: HashMap<String, Vec<String>> = HashMap::new();
    for path in lines(from) {
        let a = artist(&path);
        if !skip.contains(&path) && !skip_artists.contains(&a) {
            by_artist.entry(a).or_default().push(path);
        }
    }
    let mut artists: Vec<String> = by_artist.keys().cloned().collect();
    artists.sort();
    // A small fixed shuffle (splitmix64), the same for the same seed.
    let mut state = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut next = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    for i in (1..artists.len()).rev() {
        artists.swap(i, (next() % (i as u64 + 1)) as usize);
    }
    // Each artist's songs shuffled; then a song an artist, round after
    // round, until there are enough.
    let mut queues: Vec<Vec<String>> = artists
        .iter()
        .map(|a| {
            let mut songs = by_artist[a].clone();
            songs.sort();
            for i in (1..songs.len()).rev() {
                songs.swap(i, (next() % (i as u64 + 1)) as usize);
            }
            songs
        })
        .collect();
    let mut picked = 0;
    while picked < n && queues.iter().any(|q| !q.is_empty()) {
        for q in queues.iter_mut() {
            if picked == n {
                break;
            }
            if let Some(song) = q.pop() {
                println!("{song}");
                picked += 1;
            }
        }
    }
}

/// A song's voice, kept in `voices` (16-bit WAV), else separated (and kept).
fn voice(path: &Path, voices: Option<&Path>) -> Result<Vec<f32>, String> {
    let kept = voices.and_then(|dir| {
        jodeldiplom::lyrics::store::cache_key(path).map(|k| dir.join(format!("{k}.wav")))
    });
    if let Some(file) = kept.as_ref().filter(|f| f.is_file()) {
        let bytes = std::fs::read(file).map_err(|e| e.to_string())?;
        let pcm = jodel_dsp::wav::parse(&bytes)
            .and_then(|w| w.pcm16())
            .ok_or("not a 16-bit WAV file")?;
        return Ok(pcm.iter().map(|&s| f32::from(s) / 32768.0).collect());
    }
    let (v, _) = jodeldiplom::lyrics::timer::separate_16k(path, false)?;
    if let Some(file) = kept {
        let pcm: Vec<i16> = v
            .iter()
            .map(|s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect();
        let _ = durable::write_atomic(&file, &jodel_dsp::wav::pcm16(16_000, 1, &pcm));
    }
    Ok(v)
}

/// The mix of `path`, mono at 16 kHz, at most `most` seconds of it.
fn mix_16k(path: &Path, most: Option<f64>) -> Result<Vec<f32>, String> {
    let mut source = jodel_playback::Source::open(path).map_err(|e| e.to_string())?;
    let (rate, channels) = (
        source.info().rate,
        usize::from(source.info().channels).max(1),
    );
    let limit = most.map_or(usize::MAX, |s| (s * f64::from(rate)) as usize);
    let mut mono: Vec<f64> = Vec::new();
    let mut block = Vec::new();
    while mono.len() < limit {
        block.clear();
        let frames = source
            .read_unquantized(&mut block)
            .map_err(|e| e.to_string())?;
        if frames == 0 {
            break;
        }
        // The source's samples are in 24-bit units (full scale ±2²³).
        for frame in block.chunks(channels) {
            let sum: f64 = frame.iter().sum();
            mono.push(sum / channels as f64 / 8_388_608.0);
        }
    }
    mono.truncate(limit);
    let mut r = jodel_playback::Resampler::new(rate, 16_000, 1);
    let mut out = Vec::new();
    r.process_f64(&mono, &mut out);
    r.finish_f64(&mut out);
    Ok(out.into_iter().map(|v| v as f32).collect())
}

/// The statistics collected so far added into `file`, the collection
/// started afresh.
fn save_stats(gpu: &GpuAsr, file: &Path) {
    let mut kept = std::fs::read(file)
        .ok()
        .and_then(|b| Statistics::from_bytes(&b).ok())
        .unwrap_or_default();
    let new = Statistics {
        matrices: gpu.statistics().unwrap(),
    };
    kept.add(&new);
    durable::write_atomic(file, &kept.to_bytes()).unwrap();
    gpu.calibrate(true).unwrap();
}

/// `file`'s text, nothing if there is no such file yet.
fn so_far(file: &Path) -> String {
    match std::fs::read_to_string(file) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => panic!("{}: {e}", file.display()),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(n) = flag(&args, "--pick") {
        let from = flag(&args, "--from").expect("--pick needs --from LIST");
        let seed = flag(&args, "--seed").map_or(1, |s| s.parse().unwrap());
        let not: Vec<String> = args
            .iter()
            .enumerate()
            .filter(|(_, a)| *a == "--not")
            .filter_map(|(i, _)| args.get(i + 1).cloned())
            .collect();
        return pick(n.parse().unwrap(), from, seed, &not);
    }
    let (model, list, out) = (Path::new(&args[0]), &args[1], PathBuf::from(&args[2]));
    let voices = flag(&args, "--voices").map(PathBuf::from);
    let stats = flag(&args, "--stats").map(PathBuf::from);
    let mix = args.iter().any(|a| a == "--mix");
    let most = flag(&args, "--most").map(|s| s.parse::<f64>().unwrap());
    let started = Instant::now();
    let packed = if model.is_dir() {
        let spec = std::env::var("JODEL_ASR_BITS").unwrap_or_else(|_| "4".into());
        let weights = std::env::var_os("JODEL_ASR_STATS")
            .map(|p| Statistics::from_bytes(&std::fs::read(p).unwrap()).unwrap());
        PackedModel::from_checkpoint(model, bits_by(&spec), weights.as_ref(), |_, _| {}).unwrap()
    } else {
        PackedModel::open(model).unwrap()
    };
    let gpu = GpuAsr::new(&packed).unwrap();
    drop(packed);
    eprintln!("{}: model ready in {:.1?}", gpu.name(), started.elapsed());
    if stats.is_some() {
        gpu.calibrate(true).unwrap();
    }
    // The table so far, written whole again after each song (one that
    // cannot be read stops the run: rewritten, its songs would be lost).
    let mut tsv = so_far(&out);
    let done: HashSet<String> = tsv
        .lines()
        .filter_map(|l| l.split('\t').next().map(str::to_owned))
        .collect();
    let songs = lines(list);
    let (mut heard, mut since_save) = (0, 0);
    for (i, song) in songs.iter().enumerate() {
        if done.contains(song) {
            continue;
        }
        let t = Instant::now();
        let got_voice = if mix {
            mix_16k(Path::new(song), most)
        } else {
            voice(Path::new(song), voices.as_deref())
        };
        let v = match got_voice {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{song}: {e}");
                continue;
            }
        };
        let separated = t.elapsed();
        let t = Instant::now();
        let got = match gpu.transcribe(&v) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("{song}: {e}");
                continue;
            }
        };
        let took = t.elapsed();
        writeln!(
            tsv,
            "{song}\t{:.2}\t{}\t{}\t{}",
            took.as_secs_f64(),
            got.ids.len(),
            got.language,
            got.text.replace(['\t', '\n'], " ")
        )
        .unwrap();
        durable::write_atomic(&out, tsv.as_bytes()).unwrap();
        heard += 1;
        since_save += 1;
        eprintln!(
            "{:>5}/{}  voice {separated:.1?}, heard {took:.1?}, {} tokens  {}",
            i + 1,
            songs.len(),
            got.ids.len(),
            Path::new(song)
                .file_name()
                .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
        );
        if let (Some(file), true) = (&stats, since_save == 10) {
            save_stats(&gpu, file);
            since_save = 0;
        }
    }
    if let (Some(file), true) = (&stats, since_save > 0) {
        save_stats(&gpu, file);
    }
    eprintln!("{heard} songs heard in {:.1?}", started.elapsed());
}
