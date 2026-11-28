//! One song timed as the player times it (its kept voice, the recogniser's
//! fit check and repeats, the aligner held to the source's synced times
//! where they fit), from its lyrics as the player reads them: a text or
//! LRC file, a file of the lyrics cache, or an LRCLIB record (`.json`, its
//! plain text with the synced one's line times laid on it). Each line with
//! its start, and the words' times when `--words`.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example time_one -- DATA_DIR SONG LYRICS [--words]
//! ```

use std::path::Path;

use jodeldiplom::work::lyrics::models::Models;
use jodeldiplom::work::lyrics::timer::SpeechTimer;
use jodeldiplom::work::lyrics::WordTimer;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let data = Path::new(&args[0]);
    let song = Path::new(&args[1]);
    let found = std::fs::read_to_string(&args[2]).unwrap();
    let lyrics = if args[2].ends_with(".json") {
        match jodel_lyrics::lrclib::parse_get(&found).map(|r| r.found()) {
            Some(jodel_lyrics::Found::Lyrics(l)) => l,
            other => panic!("no lyrics in the record: {other:?}"),
        }
    } else {
        let body = jodeldiplom::store::cache::LYRICS_TEXT
            .body(&found)
            .unwrap_or(&found);
        jodel_lyrics::Lyrics::parse_text(body, jodel_lyrics::Source::Lrclib)
    };
    let synced = lyrics.lines.iter().filter(|l| l.synced.is_some()).count();
    println!(
        "{synced} of {} lines synced by the source",
        lyrics.lines.len()
    );
    let words = args.iter().any(|a| a == "--words");
    let models = Models::new(
        data.join("models")
            .join(jodeldiplom::work::lyrics::models::MODEL_FILE),
        data.join("models")
            .join(jodeldiplom::work::lyrics::models::ASR_FILE),
    );
    let timer = SpeechTimer::new(
        models,
        Some(data.join("stems")),
        jodeldiplom::setup::Unreadable::none(),
    );
    let others = jodel_lyrics::prepare::Text::Found(&Vec::new);
    let timed = timer.time(song, &lyrics, &others).unwrap();
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
