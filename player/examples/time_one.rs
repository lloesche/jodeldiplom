//! One song timed as the player times it (its kept voice, the recogniser's
//! fit check and repeats, the aligner), from a text file of its lyrics:
//! each line with its start, and the words' times when `--words`.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example time_one -- DATA_DIR SONG LYRICS.lrc [--words]
//! ```

use std::path::Path;

use jodeldiplom::lyrics::{Models, SpeechTimer, WordTimer};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let data = Path::new(&args[0]);
    let song = Path::new(&args[1]);
    let found = std::fs::read_to_string(&args[2]).unwrap();
    let body = jodeldiplom::cache::LYRICS_TEXT
        .body(&found)
        .unwrap_or(&found)
        .to_owned();
    let lyrics = jodel_lyrics::Lyrics::parse(&body, jodel_lyrics::Source::Lrclib);
    let words = args.iter().any(|a| a == "--words");
    let models = Models::new(
        data.join("models").join(jodeldiplom::lyrics::MODEL_FILE),
        data.join("models").join(jodeldiplom::lyrics::ASR_FILE),
    );
    let timer = SpeechTimer::new(models);
    let timed = timer.time(song, &lyrics, &|| Vec::new()).unwrap();
    println!(
        "{} lines (text {}), fit {:?}",
        timed.lines.len(),
        lyrics.lines.len(),
        timed.timing.fit
    );
    for l in &timed.lines {
        let t = l.words.first().and_then(|w| w.start).unwrap_or(-1.0);
        if words {
            let ws: Vec<String> = l
                .words
                .iter()
                .map(|w| {
                    format!(
                        "{}[{:.2}-{:.2}]",
                        w.text,
                        w.start.unwrap_or(-1.0),
                        w.end.unwrap_or(-1.0)
                    )
                })
                .collect();
            println!("{t:7.2}  {}", ws.join(" "));
        } else {
            println!("{t:7.2}  {}", l.text());
        }
    }
}
