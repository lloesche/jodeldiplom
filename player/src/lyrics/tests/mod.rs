use super::timer::asr_maker;
use super::web::{retry_after, Response, BUSY_WAIT, LONGEST_WAIT, SPACING, USER_AGENT};
use super::*;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::mpsc;

/// `lyrics` with every word timed (a second each, in order), by `by`,
/// as our aligner would.
fn timed_by(lyrics: &Lyrics, by: &str) -> Lyrics {
    let mut timed = lyrics.clone();
    let mut t = 1.0;
    for word in timed.lines.iter_mut().flat_map(|l| &mut l.words) {
        word.start = Some(t);
        word.end = Some(t + 0.8);
        word.confidence = 0.9;
        t += 1.0;
    }
    timed.timing.detail = jodel_lyrics::Detail::Words;
    timed.timing.by = Some(jodel_lyrics::Maker::new(by, 1));
    timed
}

/// A timer that marks what it timed.
struct Marker;

impl WordTimer for Marker {
    fn ready(&self) -> bool {
        true
    }

    fn maker(&self) -> Maker {
        Maker::new("marker", 1)
    }

    fn time(&self, _: &Path, lyrics: &Lyrics, _: &Others) -> Result<Lyrics, String> {
        Ok(timed_by(lyrics, "marker"))
    }
}

/// A timer that also hears: two lines, from any voice.
struct Transcriber;

impl WordTimer for Transcriber {
    fn ready(&self) -> bool {
        true
    }

    fn maker(&self) -> Maker {
        Maker::new("marker", 1)
    }

    fn time(&self, _: &Path, lyrics: &Lyrics, _: &Others) -> Result<Lyrics, String> {
        Ok(timed_by(lyrics, "marker"))
    }

    fn transcribe(&self, _: &Path) -> Result<(Lyrics, Lyrics), String> {
        let heard = jodel_lyrics::asr::Transcript {
            parts: vec![jodel_lyrics::asr::Part {
                start: 0.0,
                end: 30.0,
                text: "Heard this line. And then another one".into(),
            }],
            ..Default::default()
        };
        let text = transcript_lyrics(&heard, asr_maker());
        let timed = timed_by(&text, "marker");
        Ok((text, timed))
    }
}

/// A track no source has lyrics for is heard from its voice instead:
/// marked as heard (no trusted source), timed, kept, and said settled
/// as heard.
#[test]
fn a_track_with_no_lyrics_found_is_heard_instead() {
    let dir = std::env::temp_dir().join(format!("jodel-lyrics-heard-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let song = dir.join("song.flac");
    fs::write(&song, b"not audio").unwrap();
    let probe = |path: &Path| {
        Some(Track {
            path: path.to_owned(),
            ..Track::default()
        })
    };
    let settled = Arc::new(Mutex::new(Vec::new()));
    let told = Arc::clone(&settled);
    let options = Options {
        cache: Some(dir.join("cache")),
        online: false,
        http: Arc::new(Web::new()),
        base: "http://lrclib.test".into(),
        probe: Arc::new(probe),
        timer: Some(Arc::new(Transcriber)),
        working: None,
        done: Some(Arc::new(move |_: &Path, facts| {
            told.lock().unwrap().push(facts)
        })),
    };
    let (tx, rx) = mpsc::channel();
    let worker = LyricsWorker::new(options, move |path, answer| {
        let _ = tx.send((path, answer));
    });
    worker.want(vec![song.clone()]);
    let wait = Duration::from_secs(10);
    // Not "none found" first: the heard lyrics, timed.
    let (path, answer) = rx.recv_timeout(wait).unwrap();
    assert_eq!(path, song);
    let Answer::Found { lyrics, timed } = &answer else {
        panic!("{answer:?}")
    };
    assert!(matches!(lyrics.source, Source::Transcribed(_)));
    let texts: Vec<String> = lyrics.lines.iter().map(|l| l.text()).collect();
    assert_eq!(texts, ["Heard this line.", "And then another one"]);
    assert!(timed.is_some());
    assert_eq!(*settled.lock().unwrap(), [crate::core::LyricsFacts::Heard]);
    // Kept: asked again, it comes from the cache at once, still marked.
    worker.want(vec![song.clone()]);
    let (_, again) = rx.recv_timeout(wait).unwrap();
    assert!(
        matches!(&again, Answer::Found { lyrics, timed: Some(_) } if matches!(lyrics.source, Source::Transcribed(_))),
        "{again:?}"
    );
    drop(worker);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_transcript_is_cut_into_lines_at_its_sentences() {
    let heard = jodel_lyrics::asr::Transcript {
        parts: vec![
            jodel_lyrics::asr::Part {
                start: 0.0,
                end: 20.0,
                text: "Short one! Then a sentence that goes on, and on and on and on and on and on for ever"
                    .into(),
            },
            jodel_lyrics::asr::Part {
                start: 20.0,
                end: 40.0,
                text: "no stops at all here".into(),
            },
        ],
        ..Default::default()
    };
    let lyrics = transcript_lyrics(&heard, asr_maker());
    let texts: Vec<String> = lyrics.lines.iter().map(|l| l.text()).collect();
    assert_eq!(
        texts,
        [
            "Short one!",
            "Then a sentence that goes on,",
            "and on and on and on",
            "and on and on for ever",
            "no stops at all here"
        ]
    );
    assert!(lyrics
        .lines
        .iter()
        .all(|l| l.words.len() <= HEARD_LINE_MOST));
}

#[test]
fn lyrics_found_come_again_with_their_words_timed_and_kept() {
    let dir = std::env::temp_dir().join(format!("jodel-lyrics-timer-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let song = dir.join("song.flac");
    fs::write(&song, b"not audio").unwrap();
    fs::write(dir.join("song.lrc"), "[00:01.00]One line\n").unwrap();
    let probe = |path: &Path| {
        Some(Track {
            path: path.to_owned(),
            ..Track::default()
        })
    };
    let options = Options {
        cache: Some(dir.join("cache")),
        online: false,
        http: Arc::new(Web::new()),
        base: "http://lrclib.test".into(),
        probe: Arc::new(probe),
        timer: Some(Arc::new(Marker)),
        working: None,
        done: None,
    };
    let (tx, rx) = mpsc::channel();
    let worker = LyricsWorker::new(options, move |path, answer| {
        let _ = tx.send((path, answer));
    });
    worker.want(vec![song.clone()]);
    let wait = Duration::from_secs(10);
    let timed = |a: &Answer| match a {
        Answer::Found { timed, .. } => timed.as_ref().and_then(|t| t.timing.by.clone()),
        _ => None,
    };
    // First as found, then timed.
    let (_, first) = rx.recv_timeout(wait).unwrap();
    assert_eq!(timed(&first), None);
    let (path, second) = rx.recv_timeout(wait).unwrap();
    assert_eq!(path, song);
    assert_eq!(timed(&second), Some(jodel_lyrics::Maker::new("marker", 1)));
    // Kept: asked again, it comes timed at once.
    worker.want(vec![song.clone()]);
    let (_, again) = rx.recv_timeout(wait).unwrap();
    assert_eq!(timed(&again), Some(jodel_lyrics::Maker::new("marker", 1)));
    // A timing kept by an older aligner is passed over and made again.
    let cache = Cache::new(dir.join("cache"));
    let key = cache_key(&song).unwrap();
    let mut older = cache.timed(&key).unwrap();
    older.timing.by = Some(Maker::new("marker", 0));
    cache.store_timed(&key, &older).unwrap();
    worker.want(vec![song.clone()]);
    let (_, stale) = rx.recv_timeout(wait).unwrap();
    assert_eq!(timed(&stale), None);
    let (_, renewed) = rx.recv_timeout(wait).unwrap();
    assert_eq!(timed(&renewed), Some(Maker::new("marker", 1)));
    assert_eq!(
        cache.timed(&key).unwrap().timing.by,
        Some(Maker::new("marker", 1))
    );
    drop(worker);
    let _ = fs::remove_dir_all(&dir);
}

/// What the voice of the songs in these tests is heard to sing.
const HEARD: &str = "You search through the years and you wipe away the tears. You find \
    a woman, not a girl, but a feeling like the beginning of the world.";

/// A timer that hears [`HEARD`] and picks the text to time as the
/// speech timer does.
struct Hearer;

impl WordTimer for Hearer {
    fn ready(&self) -> bool {
        true
    }

    fn maker(&self) -> Maker {
        Maker::new("hearer", 1)
    }

    fn time(&self, _: &Path, lyrics: &Lyrics, others: &Others) -> Result<Lyrics, String> {
        let (text, fit) = fitting(lyrics, HEARD, others);
        let mut timed = timed_by(&text, "hearer");
        timed.timing.fit = fit;
        Ok(timed)
    }
}

#[test]
fn a_text_that_fits_is_kept_and_one_that_does_not_is_swapped() {
    let text = |t: &str| Lyrics::parse_text(t, Source::Lrclib);
    let right = text(
        "You search through the years\nAnd you wipe away the tears\n\
         You find a woman, not a girl\nBut a feeling like the beginning of the world",
    );
    let wrong = text("Hello there\nGeneral Kenobi\nYou are a bold one");
    let asked = std::cell::Cell::new(0);
    let others = || {
        asked.set(asked.get() + 1);
        vec![wrong.clone(), right.clone()]
    };
    // Fits: kept, and the others never asked for.
    let (kept, fit) = fitting(&right, HEARD, &others);
    assert_eq!(kept, right);
    assert!(fit.unwrap().percent > 90 && !fit.unwrap().swapped);
    assert_eq!(asked.get(), 0);
    // Does not fit: the other that does.
    let (swapped, fit) = fitting(&wrong, HEARD, &others);
    assert_eq!(swapped, right);
    assert!(fit.unwrap().fits() && fit.unwrap().swapped);
    // None fits: kept as it is, its fit said.
    let (kept, fit) = fitting(&wrong, HEARD, &|| vec![wrong.clone()]);
    assert_eq!(kept, wrong);
    assert!(!fit.unwrap().fits() && !fit.unwrap().swapped);
    // Too little heard to judge: kept, not judged.
    assert_eq!(
        fitting(&wrong, "oh oh yeah", &others),
        (wrong.clone(), None)
    );
}

#[test]
fn a_fetched_text_that_does_not_fit_is_swapped_for_one_that_does() {
    let dir = std::env::temp_dir().join(format!("jodel-lyrics-fit-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let song = dir.join("song.flac");
    fs::write(&song, b"not audio").unwrap();
    let record = |id: u32, text: &str| {
        format!(
            r#"{{"id":{id},"trackName":"Song","artistName":"Band","albumName":"LP",
            "duration":200,"instrumental":false,"plainLyrics":"{text}"}}"#
        )
    };
    let wrong = record(1, "Hello there\\nGeneral Kenobi\\nYou are a bold one");
    let right = record(
        2,
        "You search through the years\\nAnd you wipe away the tears\\n\
         You find a woman, not a girl\\nBut a feeling like the beginning of the world",
    );
    let search = format!("[{wrong},{right}]");
    let web = Stub::new(move |url, _| {
        if url.contains("/api/search") {
            ok(&search)
        } else {
            ok(&wrong)
        }
    });
    let probe = |path: &Path| {
        Some(Track {
            path: path.to_owned(),
            title: Some("Song".into()),
            artist: Some("Band".into()),
            album: Some("LP".into()),
            duration: Some(200.0),
            ..Track::default()
        })
    };
    let options = Options {
        cache: Some(dir.join("cache")),
        online: true,
        http: web.clone(),
        base: "http://lrclib.test".into(),
        probe: Arc::new(probe),
        timer: Some(Arc::new(Hearer)),
        working: None,
        done: None,
    };
    let (tx, rx) = mpsc::channel();
    let worker = LyricsWorker::new(options, move |path, answer| {
        let _ = tx.send((path, answer));
    });
    worker.want(vec![song.clone()]);
    let wait = Duration::from_secs(10);
    let first_line = |a: &Answer| match a {
        Answer::Found { lyrics, .. } => lyrics.lines[0].text(),
        other => panic!("{other:?}"),
    };
    // As found first (the wrong text), then the right one, timed.
    assert_eq!(first_line(&rx.recv_timeout(wait).unwrap().1), "Hello there");
    let (_, timed) = rx.recv_timeout(wait).unwrap();
    assert_eq!(first_line(&timed), "You search through the years");
    assert!(matches!(&timed, Answer::Found { timed: Some(t), .. }
        if t.lines[0].text() == "You search through the years"
            && t.timing.fit.is_some_and(|f| f.swapped && f.fits())));
    assert!(web.asked().iter().any(|u| u.contains("/api/search")));
    // Kept: the right text is the song's from now on.
    let cache = Cache::new(dir.join("cache"));
    let key = cache_key(&song).unwrap();
    let Kept::Current(kept) = cache.text(&key) else {
        panic!("not kept");
    };
    assert_eq!(kept.lines[0].text(), "You search through the years");
    drop(worker);
    let _ = fs::remove_dir_all(&dir);
}

/// A song got ready ahead is timed into the cache without a word to
/// the app, but for its timing starting and ending; wanted then, it
/// comes timed at once.
#[test]
fn songs_got_ready_ahead_are_timed_into_the_cache_quietly() {
    let dir = std::env::temp_dir().join(format!("jodel-lyrics-ahead-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let song = dir.join("song.flac");
    fs::write(&song, b"not audio").unwrap();
    fs::write(dir.join("song.lrc"), "[00:01.00]One line\n").unwrap();
    let probe = |path: &Path| {
        Some(Track {
            path: path.to_owned(),
            ..Track::default()
        })
    };
    let (busy_tx, busy_rx) = mpsc::channel();
    let busy_tx = Mutex::new(busy_tx);
    let (done_tx, done_rx) = mpsc::channel();
    let done_tx = Mutex::new(done_tx);
    let options = Options {
        cache: Some(dir.join("cache")),
        online: false,
        http: Arc::new(Web::new()),
        base: "http://lrclib.test".into(),
        probe: Arc::new(probe),
        timer: Some(Arc::new(Marker)),
        working: Some(Arc::new(move |path: &Path, on: bool| {
            let _ = busy_tx.lock().unwrap().send((path.to_owned(), on));
        })),
        done: Some(Arc::new(move |path: &Path, facts| {
            let _ = done_tx.lock().unwrap().send((path.to_owned(), facts));
        })),
    };
    let (tx, rx) = mpsc::channel();
    let worker = LyricsWorker::new(options, move |path, answer| {
        let _ = tx.send((path, answer));
    });
    worker.prepare(vec![song.clone()]);
    let wait = Duration::from_secs(10);
    assert_eq!(busy_rx.recv_timeout(wait).unwrap(), (song.clone(), true));
    assert_eq!(busy_rx.recv_timeout(wait).unwrap(), (song.clone(), false));
    // Done with once its timing is in the cache (the timing ends before).
    let timed = (song.clone(), crate::core::LyricsFacts::Timed);
    assert_eq!(done_rx.recv_timeout(wait).unwrap(), timed);
    assert!(rx.recv_timeout(Duration::from_millis(200)).is_err());
    worker.want(vec![song.clone()]);
    let (_, answer) = rx.recv_timeout(wait).unwrap();
    let by = match answer {
        Answer::Found { timed, .. } => timed.and_then(|t| t.timing.by),
        _ => None,
    };
    assert_eq!(by, Some(jodel_lyrics::Maker::new("marker", 1)));
    drop(worker);
    let _ = fs::remove_dir_all(&dir);
}

/// How a stub answers the `n`-th request, for `url`.
type Answering = dyn Fn(&str, usize) -> Result<Response, String> + Send + Sync;

/// A web that answers by a rule, and keeps what it was asked.
struct Stub {
    asked: Mutex<Vec<String>>,
    answer: Box<Answering>,
}

impl Stub {
    fn new(
        answer: impl Fn(&str, usize) -> Result<Response, String> + Send + Sync + 'static,
    ) -> Arc<Stub> {
        Arc::new(Stub {
            asked: Mutex::new(Vec::new()),
            answer: Box::new(answer),
        })
    }

    fn asked(&self) -> Vec<String> {
        self.asked.lock().unwrap().clone()
    }
}

impl Http for Stub {
    fn get(&self, url: &str) -> Result<Response, String> {
        let n = {
            let mut asked = self.asked.lock().unwrap();
            asked.push(url.to_owned());
            asked.len() - 1
        };
        (self.answer)(url, n)
    }
}

fn ok(body: &str) -> Result<Response, String> {
    Ok(Response {
        status: 200,
        body: body.into(),
        retry_after: None,
    })
}

fn status(status: u16) -> Result<Response, String> {
    Ok(Response {
        status,
        ..Response::default()
    })
}

const RECORD: &str = r#"{"id":7,"trackName":"Song","artistName":"Band","albumName":"LP",
    "duration":200,"instrumental":false,"plainLyrics":"Hello there\nGeneral Kenobi",
    "syncedLyrics":"[00:12.00]Hello there\n[00:15.50]General Kenobi"}"#;

/// A folder for one test, with a track file in it.
fn place(name: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("jodel-lyrics-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let file = dir.join("song.flac");
    fs::write(&file, b"not really audio").unwrap();
    (dir, file)
}

fn track(file: &Path) -> Track {
    Track {
        path: file.to_owned(),
        title: Some("Song".into()),
        artist: Some("Band".into()),
        album: Some("LP".into()),
        duration: Some(200.4),
        ..Track::default()
    }
}

fn finder(dir: &Path, http: Arc<dyn Http>) -> Finder {
    Finder {
        cache: Some(Cache::new(dir.join("cache"))),
        online: true,
        lrclib: Lrclib::new(http, "http://lrclib.test"),
        timer: None,
    }
}

fn no_wait(_: Duration) -> bool {
    true
}

fn lines(answer: &Answer) -> Vec<String> {
    match answer {
        Answer::Found { lyrics, .. } => lyrics.lines.iter().map(|l| l.text()).collect(),
        other => panic!("no lyrics: {other:?}"),
    }
}

mod lookup;
