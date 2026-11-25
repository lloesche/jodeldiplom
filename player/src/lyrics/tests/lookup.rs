//! The look-ups: the sources in order, the cache, and LRCLIB over the web.

use super::*;
use std::time::Instant;

#[test]
fn the_tag_then_the_sidecar_then_the_web() {
    let (dir, file) = place("order");
    let web = Stub::new(|_, _| ok(RECORD));
    let f = finder(&dir, web.clone());
    let mut t = track(&file);
    t.tags = vec![("LYRICS".into(), "From the tag".into())];
    fs::write(dir.join("song.lrc"), "[00:01.00]From the sidecar\n").unwrap();
    let answer = f.lookup(&t, 1000, &no_wait);
    assert_eq!(lines(&answer), ["From the tag"]);
    t.tags.clear();
    let answer = f.lookup(&t, 1000, &no_wait);
    assert_eq!(lines(&answer), ["From the sidecar"]);
    // Its words, the sidecar's time kept only as the scaffold.
    let Answer::Found { lyrics, .. } = &answer else {
        unreachable!()
    };
    assert!(lyrics.lines.iter().all(|l| l.time.is_none()));
    let synced = jodel_lyrics::Synced {
        start: 1.0,
        end: None,
    };
    assert_eq!(lyrics.lines[0].synced, Some(synced));
    assert!(web.asked().is_empty());
    fs::remove_file(dir.join("song.lrc")).unwrap();
    let answer = f.lookup(&t, 1000, &no_wait);
    assert_eq!(lines(&answer), ["Hello there", "General Kenobi"]);
    let Answer::Found { lyrics, timed } = &answer else {
        unreachable!()
    };
    assert_eq!(lyrics.source, Source::Lrclib);
    assert_eq!(*timed, None);
    assert_eq!(
        web.asked(),
        [
            "http://lrclib.test/api/get?artist_name=Band&track_name=Song&album_name=LP&duration=200"
        ]
    );
    // Kept, its words and when LRCLIB synced its lines: asked again, the
    // web is not.
    let kept_at = dir
        .join("cache")
        .join(format!("{}.lrc", cache_key(&file).unwrap()));
    let kept = fs::read_to_string(&kept_at).unwrap();
    assert_eq!(
        crate::cache::LYRICS_TEXT.body(&kept).map(str::trim_end),
        Some(
            "[re:jodel-lyrics]\n[source:lrclib]\nHello there\n[#:synced 12.00 15.50]\n\
             General Kenobi\n[#:synced 15.50 -]"
        )
    );
    assert_eq!(f.lookup(&t, 2000, &no_wait), answer);
    assert_eq!(web.asked().len(), 1);
    // A text kept by an older player (with the times it came with as its
    // lines' own, or without LRCLIB's synced times) is not read but kept
    // until fetched again; a timing kept in another version goes.
    let cache = Cache::new(dir.join("cache"));
    let key = cache_key(&file).unwrap();
    let older = [
        "[source:lrclib]\n[00:12.00]Hello there\n",
        "jodeldiplom-cache lyrics 2 by jodeldiplom 0.1.0\n[re:jodel-lyrics]\n[source:lrclib]\n\
         Hello there\nGeneral Kenobi\n",
    ];
    for (n, text) in older.into_iter().enumerate() {
        fs::write(&kept_at, text).unwrap();
        fs::write(
            kept_at.with_extension("timed.lrc"),
            "[00:12.00]<00:12.00>Hello\n",
        )
        .unwrap();
        assert_eq!(cache.text(&key), Kept::Outdated);
        assert!(kept_at.exists());
        assert_eq!(cache.timed(&key), None);
        assert!(!kept_at.with_extension("timed.lrc").exists());
        assert_eq!(f.lookup(&t, 3000, &no_wait), answer);
        assert_eq!(web.asked().len(), 2 + n);
        assert!(matches!(cache.text(&key), Kept::Current(_)));
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_text_kept_by_an_older_player_waits_for_the_web() {
    // Offline (or the web resting after a failure), it is neither taken
    // for no lyrics nor deleted: the look-up failed, to be tried again.
    let (dir, file) = place("outdated");
    let web = Stub::new(|_, _| ok(RECORD));
    let mut f = finder(&dir, web.clone());
    let key = cache_key(&file).unwrap();
    let kept_at = dir.join("cache").join(format!("{key}.lrc"));
    fs::create_dir_all(kept_at.parent().unwrap()).unwrap();
    let old = "jodeldiplom-cache lyrics 2 by jodeldiplom 0.1.0\n[source:lrclib]\nHello there\n";
    fs::write(&kept_at, old).unwrap();
    f.online = false;
    assert_eq!(
        f.lookup(&track(&file), 1, &no_wait),
        Answer::Failed(OUTDATED.into())
    );
    assert_eq!(fs::read_to_string(&kept_at).unwrap(), old);
    assert!(web.asked().is_empty());
    // The worker neither hears the song nor keeps anything else for it.
    let heard = Arc::new(Mutex::new(0usize));
    let options = Options {
        cache: Some(dir.join("cache")),
        online: false,
        http: web.clone(),
        base: "http://lrclib.test".into(),
        probe: Arc::new(|path: &Path| Some(track(path))),
        timer: Some(Arc::new(Counting(Arc::clone(&heard)))),
        working: None,
        done: None,
    };
    let (tx, rx) = mpsc::channel();
    let worker = LyricsWorker::new(options, move |path, answer| {
        let _ = tx.send((path, answer));
    });
    worker.want(vec![file.clone()]);
    let (_, answer) = rx.recv_timeout(Duration::from_secs(10)).unwrap();
    assert_eq!(answer, Answer::Failed(OUTDATED.into()));
    drop(worker);
    assert_eq!(*heard.lock().unwrap(), 0);
    assert_eq!(fs::read_to_string(&kept_at).unwrap(), old);
    // Online, it is fetched again and kept in this version.
    f.online = true;
    assert_eq!(
        lines(&f.lookup(&track(&file), 2, &no_wait)),
        ["Hello there", "General Kenobi"]
    );
    assert!(matches!(
        f.cache.as_ref().unwrap().text(&key),
        Kept::Current(_)
    ));
    let _ = fs::remove_dir_all(&dir);
}

/// A timer that counts how often it is asked to hear a song.
struct Counting(Arc<Mutex<usize>>);

impl WordTimer for Counting {
    fn ready(&self) -> bool {
        true
    }

    fn maker(&self) -> Maker {
        Maker::new("counting", 1)
    }

    fn time(&self, _: &Path, lyrics: &Lyrics, _: &Others) -> Result<Lyrics, String> {
        Ok(timed_by(lyrics, "counting"))
    }

    fn transcribe(&self, _: &Path) -> Result<(Lyrics, Lyrics), String> {
        *self.0.lock().unwrap() += 1;
        Err(TOO_LITTLE_HEARD.into())
    }
}

#[test]
fn lrclib_is_asked_once_a_second_at_most() {
    let (dir, file) = place("spacing");
    let web = Stub::new(|_, _| ok(RECORD));
    // No cache: nothing written between the requests, so the time from
    // one to the next is the look-ups' own.
    let f = Finder {
        cache: None,
        ..finder(&dir, web.clone())
    };
    // Other finders on the same LRCLIB (as the worker's two threads).
    let (g, h) = (f.clone(), f.clone());
    // Each wait asked for, and when.
    let waits = Mutex::new(Vec::new());
    let wait = |d| {
        waits.lock().unwrap().push((d, Instant::now()));
        true
    };
    let first = Instant::now();
    assert_eq!(lines(&f.lookup(&track(&file), 1, &wait)).len(), 2);
    assert!(
        waits.lock().unwrap().is_empty(),
        "the first is asked at once"
    );
    assert_eq!(lines(&g.lookup(&track(&file), 1, &wait)).len(), 2);
    // The second waits what is left of a second from the first: no more,
    // and no less than a second less the time since before the first.
    let waited = waits.lock().unwrap().clone();
    let [(d, at)] = waited[..] else {
        panic!("{waited:?}");
    };
    assert!(d <= SPACING && d + (at - first) >= SPACING, "{waited:?}");
    // Stopped while waiting (its turn a second after the second's, still
    // ahead): nothing asked.
    assert_eq!(
        h.lookup(&track(&file), 1, &|_| false),
        Answer::Failed("stopped".into())
    );
    assert_eq!(web.asked().len(), 2);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_search_follows_when_get_finds_nothing() {
    let (dir, file) = place("search");
    let web = Stub::new(|url, _| {
        if url.contains("/api/get") {
            status(404)
        } else {
            ok(&format!(
                "[{{\"id\":1,\"trackName\":\"Other\",\"artistName\":\"Band\",\"duration\":200,\
                 \"plainLyrics\":\"No\"}},{}]",
                RECORD
            ))
        }
    });
    let f = finder(&dir, web.clone());
    let answer = f.lookup(&track(&file), 1000, &no_wait);
    assert_eq!(lines(&answer), ["Hello there", "General Kenobi"]);
    let asked = web.asked();
    assert_eq!(asked.len(), 2);
    assert_eq!(
        asked[1],
        "http://lrclib.test/api/search?track_name=Song&artist_name=Band"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn nothing_found_is_believed_for_thirty_days() {
    let (dir, file) = place("none");
    let web = Stub::new(|url, _| {
        if url.contains("/api/get") {
            ok(
                r#"{"id":3,"trackName":"Song","artistName":"Band","instrumental":true,
                "plainLyrics":null,"syncedLyrics":null}"#,
            )
        } else {
            ok("[]")
        }
    });
    let f = finder(&dir, web.clone());
    let t = track(&file);
    let day = 24 * 60 * 60;
    assert_eq!(f.lookup(&t, 1000, &no_wait), Answer::Instrumental);
    assert_eq!(
        f.lookup(&t, 1000 + 29 * day, &no_wait),
        Answer::Instrumental
    );
    assert_eq!(web.asked().len(), 1);
    assert_eq!(
        f.lookup(&t, 1000 + 30 * day, &no_wait),
        Answer::Instrumental
    );
    assert_eq!(web.asked().len(), 2);
    // Not found at all: the same.
    let (dir2, file2) = place("none2");
    let empty = Stub::new(|url, _| {
        if url.contains("/api/get") {
            status(404)
        } else {
            ok("[]")
        }
    });
    let f = finder(&dir2, empty.clone());
    assert_eq!(f.lookup(&track(&file2), 5, &no_wait), Answer::NotFound);
    assert_eq!(f.lookup(&track(&file2), 6, &no_wait), Answer::NotFound);
    assert_eq!(empty.asked().len(), 2, "get and search, once");
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::remove_dir_all(&dir2);
}

#[test]
fn a_busy_service_is_waited_out() {
    let (dir, file) = place("busy");
    let web = Stub::new(|_, n| match n {
        0 => Ok(Response {
            status: 429,
            body: String::new(),
            retry_after: Some(Duration::from_secs(3)),
        }),
        1 => status(429),
        _ => ok(RECORD),
    });
    let waits = Mutex::new(Vec::new());
    let wait = |d| {
        waits.lock().unwrap().push(d);
        true
    };
    let f = finder(&dir, web.clone());
    let answer = f.lookup(&track(&file), 1000, &wait);
    assert_eq!(lines(&answer).len(), 2);
    // The busy answers waited out, each try after them in its slot (a
    // second after the last; the waits here pass no time).
    let waited = waits.lock().unwrap().clone();
    assert_eq!(waited.len(), 4, "{waited:?}");
    assert_eq!((waited[0], waited[2]), (Duration::from_secs(3), BUSY_WAIT));
    assert!(
        waited[1] <= SPACING && waited[3] <= 2 * SPACING,
        "{waited:?}"
    );
    // Too long a wait, or stopped: failed, nothing recorded.
    let (dir2, file2) = place("busy2");
    let web = Stub::new(|_, _| {
        Ok(Response {
            status: 429,
            body: String::new(),
            retry_after: Some(LONGEST_WAIT + Duration::from_secs(1)),
        })
    });
    let f = finder(&dir2, web.clone());
    assert!(matches!(
        f.lookup(&track(&file2), 1, &no_wait),
        Answer::Failed(_)
    ));
    let web = Stub::new(|_, _| status(429));
    let f = finder(&dir2, web.clone());
    assert_eq!(
        f.lookup(&track(&file2), 1, &|_| false),
        Answer::Failed("stopped".into())
    );
    assert_eq!(web.asked().len(), 1);
    let web = Stub::new(|_, _| Err("no network".into()));
    let f = finder(&dir2, web.clone());
    assert_eq!(
        f.lookup(&track(&file2), 1, &no_wait),
        Answer::Failed("no network".into())
    );
    assert!(!dir2.join("cache").exists());
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::remove_dir_all(&dir2);
}

#[test]
fn offline_or_nameless_tracks_never_reach_the_web() {
    let (dir, file) = place("offline");
    let web = Stub::new(|_, _| ok(RECORD));
    let mut f = finder(&dir, web.clone());
    f.online = false;
    assert_eq!(f.lookup(&track(&file), 1, &no_wait), Answer::NotFound);
    f.online = true;
    let nameless = Track {
        title: None,
        ..track(&file)
    };
    assert_eq!(f.lookup(&nameless, 1, &no_wait), Answer::NotFound);
    assert!(web.asked().is_empty());
    assert!(!dir.join("cache").exists(), "nothing recorded");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn our_word_times_come_with_the_text_they_were_made_from() {
    let (dir, file) = place("timed");
    let f = finder(&dir, Stub::new(|_, _| status(404)));
    let cache = f.cache.clone().unwrap();
    let key = cache_key(&file).unwrap();
    fs::write(dir.join("song.lrc"), "[00:01.00]Hi there\n").unwrap();
    let Answer::Found { lyrics, timed } = f.lookup(&track(&file), 1, &no_wait) else {
        panic!("no sidecar");
    };
    assert_eq!(timed, None);
    let mut ours = timed_by(&lyrics, "test");
    cache.store_timed(&key, &ours).unwrap();
    let Answer::Found { timed, .. } = f.lookup(&track(&file), 1, &no_wait) else {
        panic!("no sidecar");
    };
    let timed = timed.unwrap();
    assert_eq!(timed.timing, ours.timing);
    assert_eq!(timed.lines[0].words[1].start, ours.lines[0].words[1].start);
    // Made from another source's text: not given.
    ours.source = Source::Lrclib;
    cache.store_timed(&key, &ours).unwrap();
    let Answer::Found { timed, .. } = f.lookup(&track(&file), 1, &no_wait) else {
        panic!("no sidecar");
    };
    assert_eq!(timed, None);
    let _ = fs::remove_dir_all(&dir);
}

/// Serves one request a connection on a local port with `answers`, in
/// order, and hands back each request's head.
fn serve(answers: Vec<String>) -> (String, mpsc::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for answer in answers {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut head = String::new();
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
                head.push_str(&line);
            }
            let _ = tx.send(head);
            let _ = stream.write_all(answer.as_bytes());
            let _ = stream.flush();
        }
    });
    (base, rx)
}

fn http_answer(status: &str, headers: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {status}\r\n{headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

#[test]
fn the_web_client_against_a_local_server() {
    let body = RECORD.replace('\n', " ");
    let (base, heads) = serve(vec![
        http_answer("429 Too Many Requests", "Retry-After: 0\r\n", ""),
        http_answer("200 OK", "Content-Type: application/json\r\n", &body),
    ]);
    let web = Arc::new(Web::with(false));
    let first = web.get(&format!("{base}/x")).unwrap();
    assert_eq!(first.status, 429);
    assert_eq!(first.retry_after, Some(Duration::ZERO));
    let head = heads.recv().unwrap().to_ascii_lowercase();
    assert!(head.starts_with("get /x http/1.1\r\n"), "{head}");
    assert!(head.contains(&format!(
        "user-agent: {}\r\n",
        USER_AGENT.to_ascii_lowercase()
    )));
    // LRCLIB through it: the busy answer waited out, then the record.
    let (base, heads) = serve(vec![
        http_answer("429 Too Many Requests", "Retry-After: 0\r\n", ""),
        http_answer("200 OK", "", &body),
    ]);
    let lrclib = Lrclib::new(web, base);
    let (dir, file) = place("web");
    let found = lrclib.find(&track(&file)).unwrap();
    let Found::Lyrics(lyrics) = found else {
        panic!("no lyrics");
    };
    // Its plain text, no times.
    assert_eq!(lyrics.lines[0].text(), "Hello there");
    assert!(lyrics.lines.iter().all(|l| l.time.is_none()));
    let asked: Vec<String> = heads.iter().take(2).collect();
    assert!(asked
        .iter()
        .all(|h| h.starts_with("GET /api/get?artist_name=Band&")));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_worker_answers_in_the_order_asked() {
    let (dir, file) = place("worker");
    let other = dir.join("other.flac");
    fs::write(&other, b"also not audio").unwrap();
    fs::write(dir.join("other.lrc"), "Beside it\n").unwrap();
    let web = Stub::new(|_, _| ok(RECORD));
    let probe = |path: &Path| {
        (path.extension().is_some_and(|e| e == "flac")).then(|| Track {
            path: path.to_owned(),
            title: Some("Song".into()),
            artist: Some("Band".into()),
            ..Track::default()
        })
    };
    let options = Options {
        cache: Some(dir.join("cache")),
        online: true,
        http: web.clone(),
        base: "http://lrclib.test".into(),
        probe: Arc::new(probe),
        timer: None,
        working: None,
        done: None,
    };
    let (tx, rx) = mpsc::channel();
    let worker = LyricsWorker::new(options, move |path, answer| {
        let _ = tx.send((path, answer));
    });
    worker.want(vec![file.clone(), other.clone(), dir.join("cover.jpg")]);
    let wait = Duration::from_secs(10);
    let (path, answer) = rx.recv_timeout(wait).unwrap();
    assert_eq!(path, file);
    assert_eq!(lines(&answer), ["Hello there", "General Kenobi"]);
    let (path, answer) = rx.recv_timeout(wait).unwrap();
    assert_eq!(path, other);
    assert_eq!(lines(&answer), ["Beside it"]);
    let (_, answer) = rx.recv_timeout(wait).unwrap();
    assert!(matches!(answer, Answer::Failed(_)));
    // Offline: what is kept still comes, the web is not asked.
    worker.set_online(false);
    let third = dir.join("third.flac");
    fs::write(&third, b"x").unwrap();
    worker.want(vec![third, file.clone()]);
    assert_eq!(rx.recv_timeout(wait).unwrap().1, Answer::NotFound);
    assert_eq!(lines(&rx.recv_timeout(wait).unwrap().1).len(), 2);
    assert_eq!(web.asked().len(), 1);
    drop(worker);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_user_agent_names_the_player_only() {
    assert_eq!(
        USER_AGENT,
        format!("jodeldiplom/{}", env!("CARGO_PKG_VERSION"))
    );
    assert_eq!(retry_after(" 120 "), Some(Duration::from_secs(120)));
    assert_eq!(retry_after("Wed, 21 Oct 2015 07:28:00 GMT"), None);
}
