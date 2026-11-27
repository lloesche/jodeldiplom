//! The word timing on a measurement set, word by word, as the player times
//! a text: each song folder below `ROOT/songs` holds `voice.wav` (its
//! separated voice, 16 kHz mono 32-bit float), `lyrics.lrc` (LRCLIB's
//! synced lyrics), `parts.tsv` and `glimpses.tsv` (what the recogniser
//! heard, rows of `START<TAB>END<TAB>TEXT`; see `lyrics_set`). With
//! `--records DIR`, the text is the song's LRCLIB record kept in
//! `DIR/SONG.json` (its plain text, the synced one's line times laid on
//! it), as the player reads it; else `lyrics.lrc`.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example words_set -- MODEL.safetensors ROOT [--records DIR] VARIANT...
//! ```
//!
//! A variant is `TAG` or `TAG:SETTING=VALUE,...` (`scaffold=0` times
//! without the source's line times). Each writes `words-TAG.tsv` into
//! every song's folder: a row for each word with its line, its place in
//! the line, its text, start, end and confidence, whether it is a backing
//! word, its line's source time and the next line's (`nan` for none), the
//! share of its frames where the voice pauses, its loudest frame against
//! the song's singing (dB), and whether the source's times were trusted
//! for the song (with the offset found). A summary row per song goes to
//! the standard output.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use jodel_lyrics::aligner::Acoustic;
use jodel_lyrics::asr::Transcript;
use jodel_lyrics::ctc::{voice_pauses, CtcSettings, Hints};
use jodel_lyrics::speech::Wav2Vec2;
use jodel_lyrics::{Lyrics, Source};

#[path = "research/parts.rs"]
mod parts;
#[path = "research/wav.rs"]
mod wav;

use parts::read_parts;
use wav::read_wav;

/// A variant: its tag and settings.
struct Variant {
    tag: String,
    settings: CtcSettings,
}

/// The variant `spec` names (`TAG` or `TAG:SETTING=VALUE,...`).
fn variant(spec: &str) -> Variant {
    let (tag, changes) = spec.split_once(':').unwrap_or((spec, ""));
    let mut v = Variant {
        tag: tag.to_owned(),
        settings: CtcSettings::default(),
    };
    for change in changes.split(',').filter(|c| !c.is_empty()) {
        let (name, value) = change
            .split_once('=')
            .unwrap_or_else(|| panic!("not SETTING=VALUE: {change}"));
        let value: f32 = value
            .parse()
            .unwrap_or_else(|_| panic!("not a number: {change}"));
        let s = &mut v.settings;
        match name {
            "window_penalty" => s.window_penalty = value,
            "opening_pull" => s.opening_pull = value,
            "inside_word" => s.inside_word = value,
            "unheard" => s.unheard = value,
            "glimpse_penalty" => s.glimpse_penalty = value,
            "scaffold" => s.scaffold = value,
            _ => panic!("no setting {name}"),
        }
    }
    v
}

/// Each 20 ms frame's level in dB, and the song's singing level (the 95th
/// percentile, as `vocal_stretches` takes it).
fn levels(voice: &[f32]) -> (Vec<f64>, f64) {
    let level: Vec<f64> = voice
        .chunks(320)
        .map(|c| {
            let p = c.iter().map(|&s| f64::from(s).powi(2)).sum::<f64>() / c.len() as f64;
            10.0 * p.max(1e-12).log10()
        })
        .collect();
    let mut sorted = level.clone();
    sorted.sort_by(f64::total_cmp);
    let loud = sorted
        .get(((sorted.len().max(1) - 1) as f64 * 0.95) as usize)
        .copied()
        .unwrap_or(-60.0);
    (level, loud)
}

/// The song's text: from its record (plain, with the synced line times laid
/// on it, as the player reads LRCLIB), else its LRC file's.
fn text_of(dir: &Path, records: Option<&Path>) -> Option<Lyrics> {
    let name = dir.file_name()?.to_string_lossy().into_owned();
    if let Some(records) = records {
        let body = std::fs::read_to_string(records.join(format!("{name}.json"))).ok()?;
        let record = jodel_lyrics::lrclib::parse_get(&body)?;
        return match record.found() {
            jodel_lyrics::Found::Lyrics(l) => Some(l),
            _ => None,
        };
    }
    let text = std::fs::read_to_string(dir.join("lyrics.lrc")).ok()?;
    Some(Lyrics::parse_text(&text, Source::Lrclib))
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let records = args.iter().position(|a| a == "--records").map(|k| {
        let path = args.remove(k + 1);
        args.remove(k);
        PathBuf::from(path)
    });
    if args.len() < 3 {
        eprintln!("usage: words_set MODEL.safetensors ROOT [--records DIR] VARIANT...");
        std::process::exit(2);
    }
    let cpu = Wav2Vec2::open(Path::new(&args[0])).expect("the speech model");
    let gpu = jodel_gpu::GpuSpeech::new(&cpu).ok();
    let model: &dyn Acoustic = match &gpu {
        Some(g) => g,
        None => &cpu,
    };
    let variants: Vec<Variant> = args[2..].iter().map(|s| variant(s)).collect();
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(Path::new(&args[1]).join("songs"))
        .expect("ROOT/songs")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("voice.wav").is_file() && p.join("lyrics.lrc").is_file())
        .collect();
    dirs.sort();
    for dir in &dirs {
        let name = dir.file_name().unwrap_or_default().to_string_lossy();
        let (Some(voice), Some(text)) = (
            read_wav(&dir.join("voice.wav")),
            text_of(dir, records.as_deref()),
        ) else {
            println!("{name}: no voice or text");
            continue;
        };
        let parts = read_parts(&dir.join("parts.tsv")).unwrap_or_default();
        let kept = read_parts(&dir.join("glimpses.tsv")).unwrap_or_default();
        // Those kept, where the voice is heard (as the player hears them).
        let glimpses = jodel_lyrics::glimpses::hear_glimpses(&voice, |piece| {
            let at =
                (piece.as_ptr() as usize - voice.as_ptr() as usize) / std::mem::size_of::<f32>();
            let start = at as f64 / f64::from(jodel_lyrics::VOCALS_RATE);
            Ok(kept
                .iter()
                .find(|p| (p.start - start).abs() < 0.01)
                .map(|p| p.text.clone())
                .unwrap_or_default())
        })
        .unwrap_or_default();
        // As the player: the fit checked against the transcript (its parts'
        // words, as the recogniser joins them), and the repeats it heard put
        // in when the text fits; no other texts to swap in.
        let heard = Transcript {
            text: parts
                .iter()
                .map(|p| p.text.as_str())
                .filter(|t| !t.is_empty())
                .collect::<Vec<_>>()
                .join(" "),
            parts: parts.clone(),
            ..Transcript::default()
        };
        let others = jodel_lyrics::prepare::Text::Found(&Vec::new);
        let (lyrics, fit) = jodel_lyrics::prepare::text(&text, Some(&heard), &others);
        let Some(emissions) = model.emissions(&voice) else {
            println!("{name}: no emissions");
            continue;
        };
        let pauses = voice_pauses(&voice, &emissions);
        let (level, loud) = levels(&voice);
        let fs = emissions.frame_seconds;
        let fit = fit.map_or("-".into(), |f| f.to_text());
        let mut row = format!("{name} fit {fit}");
        for v in &variants {
            let hints = Hints {
                pauses: pauses.clone(),
                windows: jodel_lyrics::anchors::windows(&lyrics, &parts),
                glimpses: glimpses.clone(),
            };
            let vocab = model.vocab();
            let Some(timed) =
                jodel_lyrics::scaffold::align(&emissions, &lyrics, &vocab, &v.settings, &hints)
            else {
                continue;
            };
            // The check of the source's times, on the song aligned without them.
            let verdict = jodel_lyrics::ctc::align_emissions(
                &emissions,
                &lyrics,
                &vocab,
                &v.settings,
                &hints,
            )
            .as_ref()
            .and_then(jodel_lyrics::scaffold::check);
            let trusted = match verdict {
                None => String::from("-"),
                Some(v) if v.trusted() => format!("{:+.2}/{:.2}/{}", v.offset, v.agree, v.lines),
                Some(v) => format!("no{:+.2}/{:.2}/{}", v.offset, v.agree, v.lines),
            };
            let mut tsv = String::new();
            for (l, line) in timed.lines.iter().enumerate() {
                let synced = lyrics.lines[l].synced.map(|s| s.start);
                let next = jodel_lyrics::scaffold::line_end(&lyrics.lines, l);
                for (k, w) in line.words.iter().enumerate() {
                    let (s, e) = (w.start.unwrap_or(f64::NAN), w.end.unwrap_or(f64::NAN));
                    let (mut paused, mut peak, mut n) = (0usize, -120.0f64, 0usize);
                    if !s.is_nan() && !e.is_nan() {
                        let a = (s / fs).floor() as usize;
                        let b = ((e / fs).ceil() as usize).max(a + 1);
                        for &p in pauses.get(a..b.min(pauses.len())).unwrap_or_default() {
                            paused += usize::from(p);
                            n += 1;
                        }
                        for &db in level.get(a..b.min(level.len())).unwrap_or_default() {
                            peak = peak.max(db);
                        }
                    }
                    let _ = writeln!(
                        tsv,
                        "{l}\t{k}\t{}\t{s:.3}\t{e:.3}\t{:.3}\t{}\t{}\t{}\t{:.3}\t{:.1}\t{trusted}",
                        w.text.replace('\t', " "),
                        w.confidence,
                        u8::from(w.backing),
                        synced.map_or("nan".into(), |t| format!("{t:.3}")),
                        next.map_or("nan".into(), |t| format!("{t:.3}")),
                        paused as f64 / n.max(1) as f64,
                        peak - loud,
                    );
                }
            }
            let _ =
                durable::write_atomic(&dir.join(format!("words-{}.tsv", v.tag)), tsv.as_bytes());
            let _ = write!(row, "  {} {trusted}", v.tag);
        }
        println!("{row}");
    }
}
