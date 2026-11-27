//! How well the speech model times sung lines: for songs whose tags carry
//! LRC lyrics (line times), the voice is separated on the GPU, run
//! through wav2vec2 on the CPU, and the words aligned *without* the line
//! times; each line's start is then compared with its LRC time. The same
//! is done on the whole mix, for comparison.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example lyrics_align -- MODEL.safetensors FILE_OR_FOLDER...
//! cargo run --release -p jodeldiplom --example lyrics_align -- MODEL.safetensors AUDIO --lrc LYRICS.lrc
//! ```
//!
//! Folders are searched for FLAC files with timed lyrics in their tags.
//! With `--lrc`, one file is timed against the lines of an LRC file (one
//! from LRCLIB, say), and every line is listed: its LRC time, where the
//! words were heard, and the difference. `--voice-wav OUT.wav` also keeps
//! the separated voice (16 kHz mono, 32-bit float) for other tools, and
//! `--voice-in VOICE.wav` reads one kept before.

use std::path::{Path, PathBuf};
use std::time::Instant;

use jodel_lyrics::aligner::Acoustic;
use jodel_lyrics::ctc::{align_path, heard, voice_pauses, CtcSettings, Hints};
use jodel_lyrics::speech::Wav2Vec2;
use jodel_lyrics::tokens::WAV2VEC2_960H;
use jodel_lyrics::{Detail, Lyrics, Source};

/// The lyrics in a file's tags, when they have line times.
fn timed_lyrics(path: &Path) -> Option<Lyrics> {
    let info = jodel_playback::probe_header(path).ok()?;
    let text = jodel_lyrics::sources::lyrics_tags(&info.tags.all).synced?;
    let lyrics = Lyrics::parse(text, Source::Tag);
    (lyrics.timing.detail >= Detail::Lines).then_some(lyrics)
}

/// The FLAC files below `dir` (or `dir` itself) with timed lyrics.
fn find(dir: &Path, out: &mut Vec<PathBuf>) {
    if dir.is_file() {
        out.push(dir.to_owned());
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            find(&path, out);
        } else if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("flac"))
            && timed_lyrics(&path).is_some()
        {
            out.push(path);
        }
    }
}

/// Each line's start as aligned without line times, against its LRC
/// time: the errors (aligned − LRC), in seconds.
fn line_errors(model: &Wav2Vec2, voice: &[f32], lyrics: &Lyrics) -> Option<Vec<f64>> {
    let emissions = model.emissions(voice)?;
    let mut untimed = lyrics.clone();
    for line in &mut untimed.lines {
        line.time = None;
    }
    let (aligned, path) = align_path(
        &emissions,
        &untimed,
        &WAV2VEC2_960H,
        &CtcSettings::default(),
        &Hints {
            pauses: voice_pauses(voice, &emissions),
            ..Hints::default()
        },
        &[],
    )?;
    let heard = heard(&emissions, &untimed, &WAV2VEC2_960H, &path);
    println!("  heard {heard:.3} (the letters' mean probability on the path)");
    Some(
        lyrics
            .lines
            .iter()
            .zip(&aligned.lines)
            .filter(|(l, _)| !l.words.is_empty())
            .filter_map(|(l, a)| Some(a.words.first()?.start? - l.time?))
            .collect(),
    )
}

/// Median |error|, share within 0.3 s and 1 s, median signed error.
fn summary(errors: &[f64]) -> String {
    if errors.is_empty() {
        return "no lines".into();
    }
    let mut abs: Vec<f64> = errors.iter().map(|e| e.abs()).collect();
    abs.sort_by(f64::total_cmp);
    let mut signed = errors.to_vec();
    signed.sort_by(f64::total_cmp);
    let within = |s: f64| 100.0 * abs.iter().filter(|&&e| e <= s).count() as f64 / abs.len() as f64;
    format!(
        "{} lines: median |err| {:.2} s, ≤0.3 s {:.0} %, ≤1 s {:.0} %, median err {:+.2} s",
        errors.len(),
        abs[abs.len() / 2],
        within(0.3),
        within(1.0),
        signed[signed.len() / 2]
    )
}

/// `samples` (16 kHz mono) as a 32-bit float WAV file.
fn write_wav(path: &Path, samples: &[f32]) {
    durable::write_atomic(path, &jodel_dsp::wav::float32(16_000, 1, samples))
        .expect("the WAV file");
}

/// `--lrc`'s options.
struct AgainstLrc<'a> {
    audio: &'a Path,
    lrc: &'a Path,
    /// Where to keep the separated voice.
    voice_out: Option<&'a Path>,
    /// A voice kept before, read instead of separating again.
    voice_in: Option<&'a Path>,
}

#[path = "research/wav.rs"]
mod wav;

/// Each line's start against its LRC time (aligned − LRC), in seconds.
fn errors_of(timed: &[&jodel_lyrics::Line], aligned: &Lyrics) -> String {
    timed
        .iter()
        .zip(&aligned.lines)
        .filter_map(|(l, a)| Some(a.words.first()?.start? - l.time?))
        .map(|e| format!("{e:.2}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// One file against an LRC file's lines, line by line (on the GPU where
/// there is one, as the player times words). Ends with a `RESULT` line:
/// the seconds each step took, how well the words were heard, and each
/// line's error.
fn against_lrc(model: Wav2Vec2, run: AgainstLrc) {
    let text = std::fs::read_to_string(run.lrc).expect("the LRC file");
    let lyrics = Lyrics::parse(&text, Source::Sidecar);
    let started = Instant::now();
    let voice = match run.voice_in {
        Some(path) => wav::read_wav(path).expect("the WAV file"),
        None => {
            jodeldiplom::work::voice::separate_16k(run.audio, false)
                .expect("the voice")
                .0
        }
    };
    let separated = started.elapsed().as_secs_f64();
    if let Some(path) = run.voice_out {
        write_wav(path, &voice);
    }
    let started = Instant::now();
    let emissions = match jodel_gpu::GpuSpeech::new(&model) {
        Ok(gpu) => gpu.emissions(&voice),
        Err(_) => model.emissions(&voice),
    }
    .expect("emissions");
    let heard_in = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let mut untimed = lyrics.clone();
    untimed.drop_times();
    let settings = CtcSettings::default();
    let pauses = Hints {
        pauses: voice_pauses(&voice, &emissions),
        ..Hints::default()
    };
    let (aligned, path) = align_path(
        &emissions,
        &untimed,
        &WAV2VEC2_960H,
        &settings,
        &pauses,
        &[],
    )
    .expect("a path");
    let heard = heard(&emissions, &untimed, &WAV2VEC2_960H, &path);
    let aligned_in = started.elapsed().as_secs_f64();
    println!("heard {heard:.3}");
    let timed: Vec<&jodel_lyrics::Line> = lyrics
        .lines
        .iter()
        .filter(|l| !l.words.is_empty())
        .collect();
    for (l, a) in timed.iter().zip(&aligned.lines) {
        let lrc = l.time.unwrap_or(f64::NAN);
        let ours = a.words.first().and_then(|w| w.start).unwrap_or(f64::NAN);
        let conf = a.words.iter().map(|w| w.confidence).sum::<f32>() / a.words.len().max(1) as f32;
        println!(
            "{lrc:7.2} {ours:7.2} {:+6.2}  {conf:.2}  {}",
            ours - lrc,
            l.text()
        );
    }
    println!(
        "RESULT separate {separated:.2} speech {heard_in:.2} align {aligned_in:.2} heard {heard:.3} errors {}",
        errors_of(&timed, &aligned)
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = Wav2Vec2::open(Path::new(&args[0])).expect("the model");
    if let Some(k) = args.iter().position(|a| a == "--lrc") {
        let after = |flag: &str| {
            args.iter()
                .position(|a| a == flag)
                .map(|w| Path::new(&args[w + 1]))
        };
        return against_lrc(
            model,
            AgainstLrc {
                audio: Path::new(&args[1]),
                lrc: Path::new(&args[k + 1]),
                voice_out: after("--voice-wav"),
                voice_in: after("--voice-in"),
            },
        );
    }
    let mut files = Vec::new();
    for a in &args[1..] {
        find(Path::new(a), &mut files);
    }
    println!("{} songs with timed lyrics", files.len());
    let (mut all_voice, mut all_mix) = (Vec::new(), Vec::new());
    for path in &files {
        let name = path
            .file_stem()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        let Some(lyrics) = timed_lyrics(path) else {
            continue;
        };
        let started = Instant::now();
        let (voice, mix) = match jodeldiplom::work::voice::separate_16k(path, true) {
            Ok(v) => v,
            Err(e) => {
                println!("{name}: {e}");
                continue;
            }
        };
        let separated = started.elapsed();
        let on_voice = line_errors(&model, &voice, &lyrics).unwrap_or_default();
        let on_mix = line_errors(&model, &mix, &lyrics).unwrap_or_default();
        println!(
            "{name} ({separated:.1?} to separate, {:.1?} in all)",
            started.elapsed()
        );
        println!("  voice: {}", summary(&on_voice));
        println!("  mix:   {}", summary(&on_mix));
        all_voice.extend(on_voice);
        all_mix.extend(on_mix);
    }
    println!("ALL voice: {}", summary(&all_voice));
    println!("ALL mix:   {}", summary(&all_mix));
}
