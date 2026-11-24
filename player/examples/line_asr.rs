//! Real transcripts for measuring line breaks: each song's voice heard by
//! the player's own speech recogniser and timed by its aligner, exactly as
//! a song without lyrics is in the player, written to OUT_DIR/ID.json with
//! the song's beats and bars: `{"words": [[text, start, end], ...],
//! "first": [line starts as first cut], "beats": [...], "downbeats": [...]}`.
//!
//! Each line of LIST is a voice folder (`voice.wav`, 16 kHz mono float);
//! the song is the one its `meta.json` names, else the one whose lyrics
//! key the folder is named by (found among the cached analyses). Its beats
//! come from a cached analysis of it, else from analysing it (the file is
//! only read).
//!
//! ```text
//! cargo run --release -p jodeldiplom --example line_asr -- DATA_DIR LIST OUT_DIR
//! ```

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[path = "research/wav.rs"]
mod wav;

fn json_text(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Every cached analysis: its song's path, its key and its text.
fn analyses(data: &Path) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    for entry in fs::read_dir(data.join("analyses")).unwrap().flatten() {
        let Ok(text) = fs::read_to_string(entry.path()) else {
            continue;
        };
        let Some(key) = text
            .lines()
            .find_map(|l| l.strip_prefix("key "))
            .map(str::to_owned)
        else {
            continue;
        };
        let parts: Vec<&str> = key.rsplitn(3, ' ').collect();
        if parts.len() < 3 {
            continue;
        }
        let song = parts[2].to_owned();
        out.push((song, key, text));
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let data = Path::new(&args[0]);
    let list = fs::read_to_string(&args[1]).unwrap();
    let out = Path::new(&args[2]);
    fs::create_dir_all(out).unwrap();
    let models = data.join("models");
    let asr = jodel_lyrics::asr::packed::PackedModel::open(
        &models.join(jodeldiplom::lyrics::models::ASR_FILE),
    )
    .and_then(|p| jodel_gpu::GpuAsr::new(&p))
    .expect("the speech recogniser on the GPU");
    let cpu =
        jodel_lyrics::speech::Wav2Vec2::open(&models.join(jodeldiplom::lyrics::models::MODEL_FILE))
            .unwrap();
    let speech: Arc<dyn jodel_lyrics::aligner::Acoustic + Send + Sync> =
        Arc::new(jodel_gpu::GpuSpeech::new(&cpu).expect("the speech model on the GPU"));
    let cached = analyses(data);
    let recipe = jodeldiplom::core::RhythmRecipe {
        methods: vec![jodel_rhythm::Method::Flux, jodel_rhythm::Method::Network],
        stems: false,
    };
    for line in list.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let dir = Path::new(line.trim_start_matches('\u{feff}'));
        let id = dir.file_name().unwrap().to_string_lossy().into_owned();
        let target = out.join(format!("{id}.json"));
        if target.exists() {
            continue;
        }
        // The song: its meta's path, else the cached analysis whose song
        // the folder's lyrics key names.
        let song: Option<PathBuf> = fs::read_to_string(dir.join("meta.json"))
            .ok()
            .and_then(|m| jodel_lyrics::json::Json::parse(m.trim_start_matches('\u{feff}')))
            .and_then(|j| j.get("path").and_then(|p| p.as_str()).map(PathBuf::from))
            .or_else(|| {
                cached
                    .iter()
                    .find(|(s, _, _)| format!("{:016x}", jodeldiplom::rhythm::hash(s)) == id)
                    .map(|(s, _, _)| PathBuf::from(s.rsplitn(3, ' ').nth(2).unwrap_or(s)))
            });
        let Some(song) = song else {
            println!("{id}: no song found");
            continue;
        };
        let song_text = song.to_string_lossy().into_owned();
        let analysis = cached
            .iter()
            .filter(|(s, _, _)| s.starts_with(&song_text))
            .find_map(|(_, key, text)| jodeldiplom::rhythm::from_text(key, text))
            .or_else(|| jodeldiplom::rhythm::analyze_file(&song, &recipe).ok());
        let voice = wav::read_wav(&dir.join("voice.wav")).expect("the voice");
        let heard = match asr.transcribe(&voice) {
            Ok(h) => h,
            Err(e) => {
                println!("{id}: not heard: {e}");
                continue;
            }
        };
        let text =
            jodeldiplom::lyrics::transcript_lyrics(&heard, jodeldiplom::lyrics::timer::asr_maker());
        let glimpses = asr.glimpses(&voice).unwrap_or_default();
        let timed = jodel_lyrics::aligner::CtcAligner::new(Arc::clone(&speech)).align_heard_in(
            &voice,
            &text,
            &heard.parts,
            &glimpses,
        );
        let mut words = Vec::new();
        let mut first = Vec::new();
        for l in &timed.lines {
            if l.words.is_empty() {
                continue;
            }
            first.push(words.len().to_string());
            for w in &l.words {
                let (s, e) = (w.start.unwrap_or(0.0), w.end.unwrap_or(0.0));
                words.push(format!("[{},{s:.3},{e:.3}]", json_text(&w.text)));
            }
        }
        let (beats, downbeats) = analysis.map_or((Vec::new(), Vec::new()), |a| {
            (
                a.beats
                    .iter()
                    .map(|b| format!("{b:.3}"))
                    .collect::<Vec<_>>(),
                a.downbeats
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>(),
            )
        });
        fs::write(
            &target,
            format!(
                "{{\"words\":[{}],\"first\":[{}],\"beats\":[{}],\"downbeats\":[{}]}}\n",
                words.join(","),
                first.join(","),
                beats.join(","),
                downbeats.join(",")
            ),
        )
        .unwrap();
        println!("{id}: {} words, {} beats", words.len(), beats.len());
    }
}
