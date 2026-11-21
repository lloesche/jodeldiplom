//! Songs whose lyrics have real line breaks, for measuring where a line
//! splitter puts them: each song in the player's caches with LRCLIB lyrics
//! timed word by word, and its beats and bars from its analysis, written
//! as one JSON file a song into OUT_DIR. Only the caches are read (the
//! analyses name their songs; the songs themselves are never opened).
//!
//! ```text
//! cargo run --release -p jodeldiplom --example line_bench -- DATA_DIR OUT_DIR
//! ```
//!
//! Each file: `{"name": ..., "words": [[text, start, end, line], ...],
//! "beats": [...], "downbeats": [...]}`, the breaks after section breaks
//! (`♪`) kept as line numbers, those lines left out.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

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

/// `--score SONGS_DIR`: how close `jodel_lyrics::lines::breaks` comes to the
/// songs' real breaks (precision, recall, F1; exactly and within a word).
fn score(dir: &Path) {
    use jodel_lyrics::json::Json;
    let mut sums = [[0.0f64; 3]; 2];
    let mut n = 0.0;
    for entry in fs::read_dir(dir).unwrap().flatten() {
        let Some(song) = Json::parse(&fs::read_to_string(entry.path()).unwrap()) else {
            continue;
        };
        let words = song.get("words").and_then(Json::as_array).unwrap();
        let mut times = Vec::new();
        let mut ends = Vec::new();
        let mut real = Vec::new();
        let mut last = None;
        for (k, w) in words.iter().enumerate() {
            let w = w.as_array().unwrap();
            times.push((w[1].as_f64().unwrap(), w[2].as_f64().unwrap()));
            // As in the benchmark: no punctuation from LRCLIB's text.
            ends.push(jodel_lyrics::lines::Stop::None);
            let line = w[3].as_f64().unwrap();
            if last.is_some_and(|l| l != line) {
                real.push(k);
            }
            last = Some(line);
        }
        let made = jodel_lyrics::lines::breaks(&times, &ends);
        for (s, slack) in [0usize, 1].into_iter().enumerate() {
            let near = |a: usize, b: usize| a.abs_diff(b) <= slack;
            let hit = made
                .iter()
                .filter(|&&m| real.iter().any(|&r| near(m, r)))
                .count() as f64;
            let found = real
                .iter()
                .filter(|&&r| made.iter().any(|&m| near(m, r)))
                .count() as f64;
            let p = hit / made.len().max(1) as f64;
            let r = found / real.len().max(1) as f64;
            let f = 2.0 * p * r / (p + r).max(1e-9);
            sums[s][0] += p;
            sums[s][1] += r;
            sums[s][2] += f;
        }
        n += 1.0;
    }
    for (s, name) in ["exactly", "within a word"].iter().enumerate() {
        println!(
            "{n} songs {name}: precision {:.2} recall {:.2} F1 {:.2}",
            sums[s][0] / n,
            sums[s][1] / n,
            sums[s][2] / n
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args[0] == "--score" {
        score(Path::new(&args[1]));
        return;
    }
    // `--reline FILE.timed.lrc`: a kept timing's lines as they are and as
    // made again.
    if args[0] == "--reline" {
        let text = fs::read_to_string(&args[1]).unwrap();
        let body = jodeldiplom::cache::LYRICS_TIMED.body(&text).unwrap();
        let lyrics = jodel_lyrics::Lyrics::parse(body, jodel_lyrics::Source::Unknown);
        let (_, timed) = jodeldiplom::lyrics::relined(&lyrics);
        for (name, l) in [("as kept", &lyrics), ("made again", &timed)] {
            println!("--- {name}: {} lines", l.lines.len());
            for line in l.lines.iter().take(24) {
                println!("{:7.2}  {}", line.start().unwrap_or(0.0), line.text());
            }
        }
        return;
    }
    let data = Path::new(&args[0]);
    let out = Path::new(&args[1]);
    fs::create_dir_all(out).unwrap();
    let lyrics_dir = data.join("lyrics");
    let mut written = 0;
    for entry in fs::read_dir(data.join("analyses")).unwrap().flatten() {
        let file = entry.path();
        if file.extension().is_none_or(|e| e != "txt") {
            continue;
        }
        let Ok(text) = fs::read_to_string(&file) else {
            continue;
        };
        // `key PATH SIZE MODIFIED vN METHODS`: the song as the lyrics cache
        // names it is `PATH SIZE MODIFIED`.
        let Some(key) = text.lines().find_map(|l| l.strip_prefix("key ")) else {
            continue;
        };
        let parts: Vec<&str> = key.rsplitn(3, ' ').collect();
        if parts.len() < 3 {
            continue;
        }
        let song = parts[2];
        let path = song.rsplitn(3, ' ').nth(2).unwrap_or(song);
        let lyrics_key = format!("{:016x}", jodeldiplom::rhythm::hash(song));
        let Ok(timed) = fs::read_to_string(lyrics_dir.join(format!("{lyrics_key}.timed.lrc")))
        else {
            continue;
        };
        if !timed.contains("[source:lrclib]") {
            continue;
        }
        let Some(body) = jodeldiplom::cache::LYRICS_TIMED.body(&timed) else {
            continue;
        };
        let lyrics = jodel_lyrics::Lyrics::parse(body, jodel_lyrics::Source::Lrclib);
        let Some(analysis) = jodeldiplom::rhythm::from_text(key, &text) else {
            continue;
        };
        let mut words = Vec::new();
        for (n, line) in lyrics
            .lines
            .iter()
            .filter(|l| !l.words.is_empty())
            .enumerate()
        {
            for w in &line.words {
                if let (Some(s), Some(e)) = (w.start, w.end) {
                    words.push(format!("[{},{s:.3},{e:.3},{n}]", json_text(&w.text)));
                }
            }
        }
        if words.len() < 40 {
            continue;
        }
        let name = path
            .rsplit_once(['\\', '/'])
            .map_or(path, |(_, n)| n)
            .trim_end_matches(".flac")
            .to_owned();
        let beats: Vec<String> = analysis.beats.iter().map(|b| format!("{b:.3}")).collect();
        let downbeats: Vec<String> = analysis.downbeats.iter().map(|d| d.to_string()).collect();
        let json = format!(
            "{{\"name\":{},\"words\":[{}],\"beats\":[{}],\"downbeats\":[{}]}}\n",
            json_text(&name),
            words.join(","),
            beats.join(","),
            downbeats.join(",")
        );
        fs::write(out.join(format!("{lyrics_key}.json")), json).unwrap();
        written += 1;
    }
    println!("{written} songs");
}
