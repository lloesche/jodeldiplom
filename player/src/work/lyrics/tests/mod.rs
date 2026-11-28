use std::sync::Mutex;
use std::thread;

use jodel_lyrics::prepare::Text;
use jodel_lyrics::{Found, LyricsSource, Source};

use super::store::{cache_key, Cache, Kept};
use super::timer::asr_maker;
use super::web::{retry_after, Lrclib, Response, BUSY_WAIT, LONGEST_WAIT, SPACING, USER_AGENT};
use super::*;
use jodel_lyrics::lines::transcript_lyrics;
use jodel_lyrics::prepare::fitting;
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

    fn time(&self, _: &Path, lyrics: &Lyrics, _: &Text) -> Result<Lyrics, String> {
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

    fn time(&self, _: &Path, lyrics: &Lyrics, _: &Text) -> Result<Lyrics, String> {
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
    let dir = jodel_testkit::TempDir::new("lyrics-heard");
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
            told.lock().unwrap().push(facts.how)
        })),
        edits: None,
        told: None,
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
}

/// A hearer that takes a while and fails, counting its hearings.
struct SlowFailing(Arc<std::sync::atomic::AtomicUsize>);

impl WordTimer for SlowFailing {
    fn ready(&self) -> bool {
        true
    }

    fn maker(&self) -> Maker {
        Maker::new("marker", 1)
    }

    fn time(&self, _: &Path, lyrics: &Lyrics, _: &Text) -> Result<Lyrics, String> {
        Ok(timed_by(lyrics, "marker"))
    }

    fn transcribe(&self, _: &Path) -> Result<(Lyrics, Lyrics), String> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        thread::sleep(Duration::from_millis(300));
        Err("the GPU failed".into())
    }
}

/// A track no source has lyrics for is heard once a run, however often the
/// queue is got ready again (or the song wanted) while it is heard or
/// after: a hearing that failed is not tried again, the song settled as
/// none found.
#[test]
fn a_track_with_no_lyrics_found_is_heard_once_a_run() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let dir = jodel_testkit::TempDir::new("lyrics-once");
    let song = dir.join("song.flac");
    fs::write(&song, b"not audio").unwrap();
    let probe = |path: &Path| {
        Some(Track {
            path: path.to_owned(),
            ..Track::default()
        })
    };
    let hearings = Arc::new(AtomicUsize::new(0));
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
        timer: Some(Arc::new(SlowFailing(Arc::clone(&hearings)))),
        working: Some(Arc::new(move |_: &Path, on: bool| {
            let _ = busy_tx.lock().unwrap().send(on);
        })),
        done: Some(Arc::new(move |_: &Path, facts| {
            let _ = done_tx.lock().unwrap().send(facts.how);
        })),
        edits: None,
        told: None,
    };
    let (tx, rx) = mpsc::channel();
    let worker = LyricsWorker::new(options, move |path, answer| {
        let _ = tx.send((path, answer));
    });
    let wait = Duration::from_secs(10);
    worker.prepare(vec![song.clone()]);
    assert!(busy_rx.recv_timeout(wait).unwrap(), "heard");
    // Got ready again, and wanted, while it is heard.
    for _ in 0..3 {
        worker.prepare(vec![song.clone()]);
        thread::sleep(Duration::from_millis(20));
    }
    worker.want(vec![song.clone()]);
    let (_, answer) = rx.recv_timeout(wait).unwrap();
    assert_eq!(answer, Answer::NotFound);
    let none = crate::core::LyricsFacts::NotFound;
    assert_eq!(done_rx.recv_timeout(wait).unwrap(), none);
    // And after: settled at once, not heard again.
    worker.want(Vec::new());
    worker.prepare(vec![song.clone()]);
    assert_eq!(done_rx.recv_timeout(wait).unwrap(), none);
    worker.want(vec![song.clone()]);
    let (_, answer) = rx.recv_timeout(wait).unwrap();
    assert_eq!(answer, Answer::NotFound);
    thread::sleep(Duration::from_millis(500));
    assert_eq!(hearings.load(Ordering::SeqCst), 1);
    drop(worker);
}

/// A hearer that hears as [`Transcriber`] does, counting its hearings.
struct CountingHearer(Arc<std::sync::atomic::AtomicUsize>);

impl WordTimer for CountingHearer {
    fn ready(&self) -> bool {
        true
    }

    fn maker(&self) -> Maker {
        Transcriber.maker()
    }

    fn time(&self, path: &Path, lyrics: &Lyrics, text: &Text) -> Result<Lyrics, String> {
        Transcriber.time(path, lyrics, text)
    }

    fn transcribe(&self, path: &Path) -> Result<(Lyrics, Lyrics), String> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Transcriber.transcribe(path)
    }
}

/// A track heard and kept is not heard again after a restart: the next
/// run's worker, on the same cache, finds its heard text and timing there,
/// got ready ahead or wanted.
#[test]
fn a_track_heard_and_kept_is_not_heard_again_after_a_restart() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let dir = jodel_testkit::TempDir::new("lyrics-restart");
    let song = dir.join("song.flac");
    fs::write(&song, b"not audio").unwrap();
    let hearings = Arc::new(AtomicUsize::new(0));
    let run = |hearings: &Arc<AtomicUsize>| {
        let (done_tx, done_rx) = mpsc::channel();
        let done_tx = Mutex::new(done_tx);
        let options = Options {
            cache: Some(dir.join("cache")),
            online: false,
            http: Arc::new(Web::new()),
            base: "http://lrclib.test".into(),
            probe: Arc::new(|path: &Path| {
                Some(Track {
                    path: path.to_owned(),
                    ..Track::default()
                })
            }),
            timer: Some(Arc::new(CountingHearer(Arc::clone(hearings)))),
            working: None,
            done: Some(Arc::new(move |_: &Path, facts| {
                let _ = done_tx.lock().unwrap().send(facts.how);
            })),
            edits: None,
            told: None,
        };
        let (tx, rx) = mpsc::channel();
        let worker = LyricsWorker::new(options, move |path, answer| {
            let _ = tx.send((path, answer));
        });
        (worker, done_rx, rx)
    };
    let wait = Duration::from_secs(10);
    let (worker, done, _) = run(&hearings);
    worker.prepare(vec![song.clone()]);
    let heard = crate::core::LyricsFacts::Heard;
    assert_eq!(done.recv_timeout(wait).unwrap(), heard);
    drop(worker);
    assert_eq!(hearings.load(Ordering::SeqCst), 1);
    // The next run.
    let (worker, done, answers) = run(&hearings);
    worker.prepare(vec![song.clone()]);
    assert_eq!(done.recv_timeout(wait).unwrap(), heard);
    worker.want(vec![song.clone()]);
    let (_, answer) = answers.recv_timeout(wait).unwrap();
    assert!(
        matches!(&answer, Answer::Found { lyrics, timed: Some(_) } if matches!(lyrics.source, Source::Transcribed(_))),
        "{answer:?}"
    );
    assert_eq!(hearings.load(Ordering::SeqCst), 1);
    drop(worker);
}

#[test]
fn lyrics_found_come_again_with_their_words_timed_and_kept() {
    let dir = jodel_testkit::TempDir::new("lyrics-timer");
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
        edits: None,
        told: None,
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
}

/// A timer that keeps the words set by hand where they were set (as the
/// aligner does), and says when it was asked to time a text as edited.
struct Keeper(Arc<Mutex<Vec<bool>>>);

impl WordTimer for Keeper {
    fn ready(&self) -> bool {
        true
    }

    fn maker(&self) -> Maker {
        Maker::new("marker", 1)
    }

    fn time(&self, _: &Path, lyrics: &Lyrics, text: &Text) -> Result<Lyrics, String> {
        let edited = matches!(text, Text::Edited);
        self.0.lock().unwrap().push(edited);
        let mut timed = timed_by(lyrics, "marker");
        for w in timed.lines.iter_mut().flat_map(|l| &mut l.words) {
            w.keep_pin();
        }
        Ok(timed)
    }
}

/// A song edited by hand: its edits kept, the song timed again as edited
/// (only then delivered), its words set by hand where they were set and
/// its marks on; found so again at the next start without timing it; its
/// edits forgotten, its text as found again.
#[test]
fn a_song_edited_by_hand_is_timed_with_its_edits_and_found_so_again() {
    let dir = jodel_testkit::TempDir::new("lyrics-edited");
    let song = dir.join("song.flac");
    fs::write(&song, b"not audio").unwrap();
    fs::write(
        dir.join("song.lrc"),
        "[00:01.00]One line\n[00:03.00]Two here\n",
    )
    .unwrap();
    let probe = |path: &Path| {
        Some(Track {
            path: path.to_owned(),
            ..Track::default()
        })
    };
    let asked = Arc::new(Mutex::new(Vec::new()));
    let options = Options {
        cache: Some(dir.join("cache")),
        timer: Some(Arc::new(Keeper(Arc::clone(&asked)))),
        edits: Some(dir.join("edits")),
        probe: Arc::new(probe),
        ..Options::new(false)
    };
    let start = || {
        let (tx, rx) = mpsc::channel();
        let worker = LyricsWorker::new(options.clone(), move |path, answer| {
            let _ = tx.send((path, answer));
        });
        worker.want(vec![song.clone()]);
        (worker, rx)
    };
    let wait = Duration::from_secs(10);
    let (worker, rx) = start();
    let _ = rx.recv_timeout(wait).unwrap();
    let Answer::Found {
        timed: Some(timed), ..
    } = rx.recv_timeout(wait).unwrap().1
    else {
        panic!("not timed");
    };
    // "line" moved, "Two" not to be sung, "here" written "there".
    let mut edited = timed.clone();
    edited.lines[0].words[1].pin_to(2.2, 2.6);
    edited.lines[1].words[0].edit.not_sung = true;
    edited.lines[1].words[1].text = "there".into();
    let edits = edited.edits();
    worker.edited(song.clone(), Some(edits.clone()));
    let check = |answer: Answer| {
        let Answer::Found {
            lyrics,
            timed: Some(timed),
        } = answer
        else {
            panic!("not timed: {answer:?}");
        };
        let texts: Vec<String> = lyrics.lines.iter().map(|l| l.text()).collect();
        assert_eq!(texts, ["One line", "Two there"]);
        assert_eq!(timed.edits(), edits);
        assert_eq!(timed.lines[0].words[1].edit.pin, Some((2.2, 2.6)));
        assert!(timed.lines[1].words[0].edit.not_sung);
        assert!(timed.lines[1].words[1].is_timed());
    };
    check(rx.recv_timeout(wait).unwrap().1);
    assert_eq!(*asked.lock().unwrap(), [false, true]);
    drop(worker);
    // The next start: as edited, timed, at once (no timing).
    let (worker, rx) = start();
    check(rx.recv_timeout(wait).unwrap().1);
    assert_eq!(asked.lock().unwrap().len(), 2);
    // Forgotten: the text as found, with its timing as kept.
    worker.edited(song.clone(), None);
    let Answer::Found {
        lyrics,
        timed: Some(plain),
    } = rx.recv_timeout(wait).unwrap().1
    else {
        panic!("not timed");
    };
    let texts: Vec<String> = lyrics.lines.iter().map(|l| l.text()).collect();
    assert_eq!(texts, ["One line", "Two here"]);
    let words = |l: &Lyrics| -> Vec<(String, Option<f64>)> {
        let words = l.lines.iter().flat_map(|l| &l.words);
        words.map(|w| (w.text.clone(), w.start)).collect()
    };
    assert_eq!(words(&plain), words(&timed));
    assert!(plain
        .lines
        .iter()
        .flat_map(|l| &l.words)
        .all(|w| !w.edit.is_set()));
    assert!(edits::EditStore::new(dir.join("edits"))
        .load(&song)
        .is_none());
    drop(worker);
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

    fn time(&self, _: &Path, lyrics: &Lyrics, text: &Text) -> Result<Lyrics, String> {
        let Text::Found(others) = text else {
            return Ok(timed_by(lyrics, "hearer"));
        };
        let (text, fit) = fitting(lyrics, HEARD, others);
        let mut timed = timed_by(&text, "hearer");
        timed.timing.fit = fit;
        Ok(timed)
    }
}

#[test]
fn a_fetched_text_that_does_not_fit_is_swapped_for_one_that_does() {
    let dir = jodel_testkit::TempDir::new("lyrics-fit");
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
        edits: None,
        told: None,
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
}

/// A timer that finds the text fits and times it with its last line put
/// in again, as a repeat heard and left out of the text.
struct Repeater;

impl WordTimer for Repeater {
    fn ready(&self) -> bool {
        true
    }

    fn maker(&self) -> Maker {
        Maker::new("repeater", 1)
    }

    fn time(&self, _: &Path, lyrics: &Lyrics, _: &Text) -> Result<Lyrics, String> {
        let mut text = lyrics.clone();
        let last = text.lines.last().cloned().unwrap();
        text.lines.push(jodel_lyrics::Line {
            synced: None,
            ..last
        });
        let mut timed = timed_by(&text, "repeater");
        timed.timing.fit = Some(jodel_lyrics::Fit::new(0.8, false));
        Ok(timed)
    }
}

#[test]
fn a_fitting_text_timed_with_repeats_put_in_is_kept_as_found() {
    let dir = jodel_testkit::TempDir::new("lyrics-repeats");
    let song = dir.join("song.flac");
    fs::write(&song, b"not audio").unwrap();
    let record = r#"{"id":1,"trackName":"Song","artistName":"Band","albumName":"LP",
        "duration":200,"instrumental":false,"plainLyrics":"First line\nThe chorus"}"#;
    let web = Stub::new(move |_, _| ok(record));
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
        timer: Some(Arc::new(Repeater)),
        working: None,
        done: None,
        edits: None,
        told: None,
    };
    let (tx, rx) = mpsc::channel();
    let worker = LyricsWorker::new(options, move |path, answer| {
        let _ = tx.send((path, answer));
    });
    worker.want(vec![song.clone()]);
    let wait = Duration::from_secs(10);
    let texts = |l: &Lyrics| l.lines.iter().map(|l| l.text()).collect::<Vec<_>>();
    let _found = rx.recv_timeout(wait).unwrap();
    let (_, answer) = rx.recv_timeout(wait).unwrap();
    let Answer::Found {
        lyrics,
        timed: Some(timed),
    } = &answer
    else {
        panic!("{answer:?}");
    };
    // Shown with the repeat; the text stays as found, not taken for
    // another one, in the answer and in the cache.
    assert_eq!(texts(timed), ["First line", "The chorus", "The chorus"]);
    assert_eq!(texts(lyrics), ["First line", "The chorus"]);
    let cache = Cache::new(dir.join("cache"));
    let key = cache_key(&song).unwrap();
    let Kept::Current(kept) = cache.text(&key) else {
        panic!("not kept");
    };
    assert_eq!(texts(&kept), ["First line", "The chorus"]);
    assert_eq!(cache.timed(&key).unwrap().lines.len(), 3);
    drop(worker);
}

/// A song got ready ahead is timed into the cache without a word to
/// the app, but for its timing starting and ending; wanted then, it
/// comes timed at once.
#[test]
fn songs_got_ready_ahead_are_timed_into_the_cache_quietly() {
    let dir = jodel_testkit::TempDir::new("lyrics-ahead");
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
            let _ = done_tx.lock().unwrap().send((path.to_owned(), facts.how));
        })),
        edits: None,
        told: None,
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
fn place(name: &str) -> (jodel_testkit::TempDir, PathBuf) {
    let dir = jodel_testkit::TempDir::new(&format!("lyrics-{name}"));
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
        edits: None,
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
