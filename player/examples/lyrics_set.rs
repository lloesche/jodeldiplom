//! The word timing on a measurement set, against LRCLIB's line times: each
//! song folder below `ROOT/songs` holds `voice.wav` (its separated voice,
//! 16 kHz mono 32-bit float, as `lyrics_align --voice-wav` keeps it),
//! `lyrics.lrc` (LRCLIB's synced lyrics) and, for the pins, `parts.tsv`
//! (what the speech recogniser heard in the transcript's parts) and
//! `glimpses.tsv` (what it heard in its glimpses), both rows of
//! `START<TAB>END<TAB>TEXT`. The text is timed as the player times it,
//! without its line times (wav2vec2 on the GPU, the CTC alignment with the
//! voice's pauses and what the recogniser heard), once for each variant;
//! each line's start is then held against its LRC time.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example lyrics_set -- MODEL.safetensors ROOT [--asr PACKED] VARIANT...
//! ```
//!
//! With `--asr`, a song without `glimpses.tsv` is heard in glimpses by that
//! recogniser (on the GPU) and they are kept there.
//!
//! A variant is `TAG` or `TAG:SETTING=VALUE,...`: the aligner's default
//! settings with those changed (`window_penalty`, `opening_pull`,
//! `inside_word`, `skip_penalty`, `island_heard`, `confident`, `trusted`,
//! `unheard`, `glimpse_penalty`), `pins=0` to time without the transcript's parts and
//! `glimpses=0` without the glimpses (as a text the recogniser has not
//! heard). Each writes `ours-TAG.tsv` into every song's folder, a row for
//! each line with words: its LRC time, its first word's start, its longest
//! word's length, its words and its last word's end, tab-separated (what
//! the research scripts read). A song's emissions are made once for all
//! variants. The last lines printed sum each variant up: the lines within
//! 1 s of LRCLIB's, the first lines over 3 s off, and the lines whose last
//! word ends over [`OVERRUN`] after the next line's LRC time (a word held
//! over a stretch with no words of its own).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use jodel_lyrics::asr::Part;
use jodel_lyrics::ctc::{align_heard, voice_pauses, Acoustic, CtcSettings, Hints};
use jodel_lyrics::speech::Wav2Vec2;
use jodel_lyrics::{Lyrics, Source};

/// A 16 kHz mono 32-bit float WAV file's samples (after a 44-byte header).
fn read_wav(path: &Path) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    Some(
        bytes
            .get(44..)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
    )
}

/// The recogniser's parts or glimpses in `path`, if it is there.
fn read_parts(path: &Path) -> Option<Vec<Part>> {
    let text = std::fs::read_to_string(path).ok()?;
    Some(
        text.lines()
            .filter_map(|row| {
                let mut cells = row.splitn(3, '\t');
                Some(Part {
                    start: cells.next()?.trim().parse().ok()?,
                    end: cells.next()?.trim().parse().ok()?,
                    text: cells.next().unwrap_or("").to_owned(),
                })
            })
            .collect(),
    )
}

/// `parts` as `read_parts` reads them.
fn parts_text(parts: &[Part]) -> String {
    let mut out = String::new();
    for p in parts {
        let _ = writeln!(
            out,
            "{:.2}\t{:.2}\t{}",
            p.start,
            p.end,
            p.text.replace(['\t', '\n'], " ")
        );
    }
    out
}

/// How far past the next line's LRC time a line may end before it counts
/// as running over (into the next line, or the silence after the last).
const OVERRUN: f64 = 1.0;

/// A variant: its tag, its settings, and whether the transcript's parts
/// and the glimpses are used.
struct Variant {
    tag: String,
    settings: CtcSettings,
    pins: bool,
    glimpses: bool,
}

/// The variant `spec` names (`TAG` or `TAG:SETTING=VALUE,...`).
fn variant(spec: &str) -> Variant {
    let (tag, changes) = spec.split_once(':').unwrap_or((spec, ""));
    let mut v = Variant {
        tag: tag.to_owned(),
        settings: CtcSettings::default(),
        pins: true,
        glimpses: true,
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
            "pins" => v.pins = value != 0.0,
            "glimpses" => v.glimpses = value != 0.0,
            "window_penalty" => s.window_penalty = value,
            "opening_pull" => s.opening_pull = value,
            "inside_word" => s.inside_word = value,
            "skip_penalty" => s.skip_penalty = value,
            "island_heard" => s.island_heard = value,
            "confident" => s.confident = value,
            "trusted" => s.trusted = value,
            "unheard" => s.unheard = value,
            "glimpse_penalty" => s.glimpse_penalty = value,
            _ => panic!("no setting {name}"),
        }
    }
    v
}

/// The totals of a variant: lines, lines within 1 s, songs, first lines
/// over 3 s off, lines running over.
#[derive(Default)]
struct Totals {
    lines: usize,
    within: usize,
    songs: usize,
    first_off: usize,
    overrun: usize,
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let asr = args.iter().position(|a| a == "--asr").map(|k| {
        let path = args.remove(k + 1);
        args.remove(k);
        let packed = jodel_lyrics::asr::packed::PackedModel::open(Path::new(&path))
            .expect("the speech recogniser");
        jodel_gpu::GpuAsr::new(&packed).expect("the recogniser on the GPU")
    });
    if args.len() < 3 {
        eprintln!("usage: lyrics_set MODEL.safetensors ROOT [--asr PACKED] VARIANT...");
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
    let mut totals: Vec<Totals> = variants.iter().map(|_| Totals::default()).collect();
    for dir in &dirs {
        let name = dir.file_name().unwrap_or_default().to_string_lossy();
        let (Some(voice), Ok(text)) = (
            read_wav(&dir.join("voice.wav")),
            std::fs::read_to_string(dir.join("lyrics.lrc")),
        ) else {
            continue;
        };
        let mut lyrics = Lyrics::parse(&text, Source::Lrclib);
        // Each line with words: its time and the next line's (one without
        // words too: where the singing stops).
        let lrc: Vec<(Option<f64>, Option<f64>)> = (0..lyrics.lines.len())
            .filter(|&k| !lyrics.lines[k].words.is_empty())
            .map(|k| {
                (
                    lyrics.lines[k].time,
                    lyrics.lines.get(k + 1).and_then(|l| l.time),
                )
            })
            .collect();
        lyrics.lines.retain(|l| !l.words.is_empty());
        lyrics.drop_times();
        let parts = read_parts(&dir.join("parts.tsv")).unwrap_or_default();
        let kept = dir.join("glimpses.tsv");
        let glimpses = match (read_parts(&kept), &asr) {
            // Those kept, where the voice is heard (as the player hears
            // them: kept by an older run, they may have been heard all).
            (Some(g), _) => jodel_lyrics::glimpses::hear_glimpses(&voice, |piece| {
                let at = (piece.as_ptr() as usize - voice.as_ptr() as usize)
                    / std::mem::size_of::<f32>();
                let start = at as f64 / f64::from(jodel_lyrics::VOCALS_RATE);
                Ok(g.iter()
                    .find(|p| (p.start - start).abs() < 0.01)
                    .map(|p| p.text.clone())
                    .unwrap_or_default())
            })
            .unwrap_or_default(),
            (None, Some(asr)) => match asr.glimpses(&voice) {
                Ok(g) => {
                    let _ = std::fs::write(&kept, parts_text(&g));
                    g
                }
                Err(e) => {
                    println!("{name}: no glimpses: {e}");
                    Vec::new()
                }
            },
            (None, None) => Vec::new(),
        };
        let Some(emissions) = model.emissions(&voice) else {
            println!("{name}: no emissions");
            continue;
        };
        let pauses = voice_pauses(&voice, &emissions);
        let mut row = format!("{name}");
        for (v, total) in variants.iter().zip(&mut totals) {
            let hints = Hints {
                pauses: pauses.clone(),
                windows: if v.pins {
                    jodel_lyrics::anchors::windows(&lyrics, &parts)
                } else {
                    Vec::new()
                },
                glimpses: if v.glimpses {
                    glimpses.clone()
                } else {
                    Vec::new()
                },
            };
            let Some((timed, _)) =
                align_heard(&emissions, &lyrics, &model.vocab(), &v.settings, &hints)
            else {
                continue;
            };
            let mut tsv = String::new();
            let (mut within, mut first_err, mut overrun) = (0, None, 0);
            for (line, (at, next)) in timed.lines.iter().zip(&lrc) {
                let start = line.words.first().and_then(|w| w.start).unwrap_or(f64::NAN);
                let end = line
                    .words
                    .iter()
                    .rev()
                    .find_map(|w| w.end)
                    .unwrap_or(f64::NAN);
                let longest = line
                    .words
                    .iter()
                    .filter_map(|w| Some(w.end? - w.start?))
                    .fold(0.0, f64::max);
                let at = at.unwrap_or(f64::NAN);
                let _ = writeln!(tsv, "{at}\t{start}\t{longest}\t{}\t{end}", line.words.len());
                within += usize::from((start - at).abs() <= 1.0);
                overrun += usize::from(next.is_some_and(|n| end > n + OVERRUN));
                first_err.get_or_insert(start - at);
            }
            let _ = std::fs::write(dir.join(format!("ours-{}.tsv", v.tag)), tsv);
            total.lines += timed.lines.len();
            total.within += within;
            total.songs += 1;
            let first = first_err.unwrap_or(f64::NAN);
            total.first_off += usize::from(first.abs() > 3.0);
            total.overrun += overrun;
            let _ = write!(
                row,
                "  {} {within}/{} first {first:+.2} over {overrun}",
                v.tag,
                timed.lines.len()
            );
        }
        println!("{row}");
    }
    for (v, t) in variants.iter().zip(&totals) {
        println!(
            "ALL {}: {} songs {} lines, within 1 s {:.1} %, first lines >3 s off {}, \
             lines ending over {OVERRUN} s after the next's time {}",
            v.tag,
            t.songs,
            t.lines,
            100.0 * t.within as f64 / t.lines.max(1) as f64,
            t.first_off,
            t.overrun
        );
    }
}
