//! Songs for measuring the speech recogniser: each song in the player's
//! caches with LRCLIB's text and its stems kept, written to OUT_DIR/KEY/ as
//! `voice.wav` (its voice from the kept stems, mono 16 kHz 32-bit float,
//! nothing separated again), `lyrics.txt` (LRCLIB's text, a line a line)
//! and `name.txt`. The songs themselves are only looked at (their size and
//! date name their stems), never opened.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example voice_export -- DATA_DIR OUT_DIR
//! ```

use std::fs;
use std::path::Path;

fn wav_f32(samples: &[f32], rate: u32) -> Vec<u8> {
    let data = (samples.len() * 4) as u32;
    let mut out = Vec::with_capacity(44 + data as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 4).to_le_bytes());
    out.extend_from_slice(&4u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data.to_le_bytes());
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// `bytes` as `dir/name`, written whole.
fn put(dir: &Path, name: &str, bytes: impl AsRef<[u8]>) {
    durable::write_atomic(&dir.join(name), bytes.as_ref()).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let data = Path::new(&args[0]);
    let out = Path::new(&args[1]);
    let stems = data.join("stems");
    // `-- DATA_DIR OUT_DIR --song SONG_PATH TEXT_FILE`: one song, its text
    // given (a reference for a song LRCLIB lacks).
    if args.get(2).is_some_and(|a| a == "--song") {
        let path = Path::new(&args[3]);
        let text = fs::read_to_string(&args[4]).unwrap();
        let file = jodeldiplom::stems_cache::open(&stems, path).expect("its stems kept");
        let voice = jodeldiplom::lyrics::timer::voice_16k(file).unwrap();
        let key = jodeldiplom::lyrics::store::cache_key(path).unwrap();
        let dir = out.join(&key);
        let lines: Vec<&str> = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('[') && !l.starts_with('('))
            .collect();
        put(&dir, "lyrics.txt", lines.join("\n") + "\n");
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        put(&dir, "name.txt", &name);
        put(&dir, "voice.wav", wav_f32(&voice, 16_000));
        println!("{key}  {name}  {:.0} s", voice.len() as f64 / 16_000.0);
        return;
    }
    let mut written = 0;
    for entry in fs::read_dir(data.join("analyses")).unwrap().flatten() {
        let Ok(text) = fs::read_to_string(entry.path()) else {
            continue;
        };
        let Some(key) = text.lines().find_map(|l| l.strip_prefix("key ")) else {
            continue;
        };
        let parts: Vec<&str> = key.rsplitn(3, ' ').collect();
        if parts.len() < 3 {
            continue;
        }
        let song = parts[2];
        let path = Path::new(song.rsplitn(3, ' ').nth(2).unwrap_or(song));
        let lyrics_key = format!("{:016x}", jodeldiplom::rhythm::hash(song));
        let dir = out.join(&lyrics_key);
        if dir.join("voice.wav").is_file() {
            continue;
        }
        let Ok(found) = fs::read_to_string(data.join("lyrics").join(format!("{lyrics_key}.lrc")))
        else {
            continue;
        };
        let Some(body) = jodeldiplom::cache::LYRICS_TEXT.body(&found) else {
            continue;
        };
        let lyrics = jodel_lyrics::Lyrics::parse(body, jodel_lyrics::Source::Unknown);
        if lyrics.source != jodel_lyrics::Source::Lrclib || lyrics.lines.len() < 8 {
            continue;
        }
        let Some(file) = jodeldiplom::stems_cache::open(&stems, path) else {
            continue;
        };
        let Ok(voice) = jodeldiplom::lyrics::timer::voice_16k(file) else {
            continue;
        };
        let lines: Vec<String> = lyrics
            .lines
            .iter()
            .filter(|l| !l.words.is_empty())
            .map(|l| l.text())
            .collect();
        put(&dir, "lyrics.txt", lines.join("\n") + "\n");
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        put(&dir, "name.txt", &name);
        // Last: a song with its voice.wav is done (passed over above).
        put(&dir, "voice.wav", wav_f32(&voice, 16_000));
        println!(
            "{lyrics_key}  {name}  {:.0} s",
            voice.len() as f64 / 16_000.0
        );
        written += 1;
    }
    println!("{written} songs");
}
