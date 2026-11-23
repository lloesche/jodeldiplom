//! Lyrics of the tracks for karaoke: found in the background, and kept on
//! disk.
//!
//! A [`LyricsWorker`] thread looks up the tracks it is asked for, one at a
//! time, in this order:
//!
//! 1. the file's lyrics tag ([`TagSource`]);
//! 2. a `.lrc` file beside it ([`SidecarSource`]);
//! 3. what was fetched before, from the cache;
//! 4. LRCLIB ([`Lrclib`]): first the one record for the artist, title,
//!    album and length, then a search by title and artist. A record marked
//!    instrumental is kept as "no lyrics". Only when `online` is on.
//!
//! The web requests go one after another, with a `User-Agent` of
//! `jodeldiplom/<version>` and nothing about the user; a `429` (or `503`)
//! answer is waited out for as long as its `Retry-After` says, within
//! reason, and asked again.
//!
//! The cache is `lyrics/` beside `config.toml`, a few files a track named
//! by a hash of its path, size and modification time (as the beat
//! analysis's cache is):
//!
//! - `{key}.lrc`: the text as fetched, after a `[source:...]` line;
//! - `{key}.none`: "nothing found" or "instrumental", with when; asked
//!   again after [`ASK_AGAIN`];
//! - `{key}.timed.lrc`: our own word times, as Enhanced LRC, with a header
//!   naming the source and the aligner, kept by [`Cache::store_timed`].

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use jodel_lyrics::fit::FIT_LEAST;
use jodel_lyrics::lrclib::{self, Record};
use jodel_lyrics::sources::{read_text, SidecarSource, TagSource};
use jodel_lyrics::{Fit, Found, Lyrics, LyricsSource, Maker, Source, Track};

use crate::cache;

/// What the player tells web services it is.
pub const USER_AGENT: &str = concat!("jodeldiplom/", env!("CARGO_PKG_VERSION"));

/// How long "nothing found" is believed before LRCLIB is asked again.
pub const ASK_AGAIN: Duration = Duration::from_secs(30 * 24 * 60 * 60);

/// The longest `Retry-After` waited out; a longer one gives up for now.
pub const LONGEST_WAIT: Duration = Duration::from_secs(120);

/// How long to wait when a busy answer gives no `Retry-After`.
const BUSY_WAIT: Duration = Duration::from_secs(10);

/// How many times one address is asked while the service is busy.
const TRIES: usize = 3;

/// How long a track whose look-up failed (the network, the service) is
/// not looked up on the web again.
const FAILED_PAUSE: Duration = Duration::from_secs(10 * 60);

/// A web service's answer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Response {
    /// The HTTP status.
    pub status: u16,
    /// The body, as text.
    pub body: String,
    /// The `Retry-After` header, when it gives seconds.
    pub retry_after: Option<Duration>,
}

/// Something that makes GET requests.
pub trait Http: Send + Sync {
    /// The answer to `GET url`; `Err` when there was none (no network, a
    /// timeout).
    fn get(&self, url: &str) -> Result<Response, String>;
}

/// The `Retry-After` value in seconds (its date form is not read).
pub fn retry_after(value: &str) -> Option<Duration> {
    value.trim().parse().ok().map(Duration::from_secs)
}

/// GET requests over the web, through `ureq` (HTTPS by rustls).
pub struct Web {
    agent: ureq::Agent,
}

impl Web {
    /// A client with the player's `User-Agent` and a 20-second limit a
    /// request, through the proxy the environment names, if any.
    pub fn new() -> Web {
        Web::with(true)
    }

    fn with(env_proxy: bool) -> Web {
        let mut config = ureq::Agent::config_builder()
            .user_agent(USER_AGENT)
            .http_status_as_error(false)
            .timeout_global(Some(Duration::from_secs(20)));
        if !env_proxy {
            config = config.proxy(None);
        }
        Web {
            agent: config.build().into(),
        }
    }
}

impl Default for Web {
    fn default() -> Self {
        Web::new()
    }
}

impl Http for Web {
    fn get(&self, url: &str) -> Result<Response, String> {
        let mut answer = self.agent.get(url).call().map_err(|e| e.to_string())?;
        let retry = answer
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(retry_after);
        let status = answer.status().as_u16();
        let body = answer
            .body_mut()
            .read_to_string()
            .map_err(|e| e.to_string())?;
        Ok(Response {
            status,
            body,
            retry_after: retry,
        })
    }
}

/// The LRCLIB service as a source.
#[derive(Clone)]
pub struct Lrclib {
    http: Arc<dyn Http>,
    base: String,
}

impl Lrclib {
    /// LRCLIB at `base` ([`lrclib::BASE`]), asked through `http`.
    pub fn new(http: Arc<dyn Http>, base: impl Into<String>) -> Lrclib {
        Lrclib {
            http,
            base: base.into(),
        }
    }

    /// The body of `GET url`: `None` for a 404. A busy answer is waited
    /// out by `wait` (which says `false` to give up) and asked again.
    fn ask(&self, url: &str, wait: &dyn Fn(Duration) -> bool) -> Result<Option<String>, String> {
        for _ in 0..TRIES {
            let answer = self.http.get(url)?;
            match answer.status {
                200 => return Ok(Some(answer.body)),
                404 => return Ok(None),
                429 | 503 => {
                    let pause = answer.retry_after.unwrap_or(BUSY_WAIT);
                    if pause > LONGEST_WAIT {
                        return Err(format!("LRCLIB asks to wait {} s", pause.as_secs()));
                    }
                    if !wait(pause) {
                        return Err("stopped".into());
                    }
                }
                status => return Err(format!("LRCLIB answered {status}")),
            }
        }
        Err("LRCLIB stayed busy".into())
    }

    /// The record LRCLIB has for `track`: the one its `get` finds (when
    /// it has text or is instrumental), else the best of a search. `None`
    /// when there is none, or the track has no title and artist to ask
    /// by.
    pub fn fetch(
        &self,
        track: &Track,
        wait: &dyn Fn(Duration) -> bool,
    ) -> Result<Option<Record>, String> {
        let Some(get) = lrclib::get_url(&self.base, track) else {
            return Ok(None);
        };
        if let Some(record) = self.ask(&get, wait)?.as_deref().and_then(lrclib::parse_get) {
            if record.found() != Found::NotFound {
                return Ok(Some(record));
            }
        }
        let Some(search) = lrclib::search_url(&self.base, track) else {
            return Ok(None);
        };
        let records = self
            .ask(&search, wait)?
            .map(|body| lrclib::parse_search(&body))
            .unwrap_or_default();
        Ok(lrclib::best(&records, track).cloned())
    }

    /// The texts LRCLIB's search has for `track` (each once, at most
    /// [`MOST_CANDIDATES`]), for when the one found does not fit the song.
    pub fn candidates(
        &self,
        track: &Track,
        wait: &dyn Fn(Duration) -> bool,
    ) -> Result<Vec<Lyrics>, String> {
        let Some(search) = lrclib::search_url(&self.base, track) else {
            return Ok(Vec::new());
        };
        let records = self
            .ask(&search, wait)?
            .map(|body| lrclib::parse_search(&body))
            .unwrap_or_default();
        let mut out: Vec<Lyrics> = Vec::new();
        for record in &records {
            if let Found::Lyrics(lyrics) = record.found() {
                let words = jodel_lyrics::fit::lyrics_words(&lyrics);
                if !out
                    .iter()
                    .any(|o| jodel_lyrics::fit::lyrics_words(o) == words)
                {
                    out.push(lyrics);
                }
            }
            if out.len() == MOST_CANDIDATES {
                break;
            }
        }
        Ok(out)
    }
}

/// LRCLIB as the player asks it: through [`Web`], at [`lrclib::BASE`].
pub fn lrclib_of() -> Lrclib {
    Lrclib::new(Arc::new(Web::new()), lrclib::BASE)
}

/// The most of LRCLIB's other texts a song's is compared with.
pub const MOST_CANDIDATES: usize = 12;

impl LyricsSource for Lrclib {
    fn name(&self) -> &str {
        "lrclib"
    }

    fn find(&self, track: &Track) -> Result<Found, String> {
        let wait = |pause| {
            thread::sleep(pause);
            true
        };
        Ok(self
            .fetch(track, &wait)?
            .map_or(Found::NotFound, |r| r.found()))
    }
}

/// Where lyrics are kept: `lyrics` in the data folder of the usual settings
/// ([`crate::dirs::Data::current`]; the player itself is told its own).
pub fn cache_dir() -> Option<PathBuf> {
    crate::dirs::Data::current().map(|d| d.lyrics())
}

/// The cache's name for `path`'s lyrics: a hash of its path, size and
/// modification time (a file retagged is looked up afresh).
pub fn cache_key(path: &Path) -> Option<String> {
    let meta = fs::metadata(path).ok()?;
    let modified = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    let key = format!("{} {} {modified}", path.display(), meta.len());
    Some(format!("{:016x}", crate::rhythm::hash(&key)))
}

/// Seconds since 1970.
pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The first line of a "nothing found" record.
const NONE_HEADER: &str = "jodel-lyrics none";

/// Lyrics kept on disk, in a folder.
#[derive(Clone, Debug)]
pub struct Cache {
    /// The folder.
    pub dir: PathBuf,
}

impl Cache {
    /// The cache in `dir`.
    pub fn new(dir: impl Into<PathBuf>) -> Cache {
        Cache { dir: dir.into() }
    }

    fn file(&self, key: &str, extension: &str) -> PathBuf {
        self.dir.join(format!("{key}.{extension}"))
    }

    fn write(&self, file: &Path, text: &str) -> io::Result<()> {
        // Where it goes now: while the data folder moves, into the new one.
        let file = &crate::mover::current(file);
        if let Some(dir) = file.parent() {
            fs::create_dir_all(dir)?;
        }
        durable::write_atomic(file, text.as_bytes())
    }

    /// The body of the file under `key` with `extension`, when it is of
    /// `schema`'s version; a file of another version is deleted (an older
    /// one may hold times that came with the text).
    fn current(&self, key: &str, extension: &str, schema: cache::Schema) -> Option<String> {
        // Where it is now: while the data folder moves, not yet moved is
        // where it was.
        let file = crate::mover::locate(&self.file(key, extension));
        let text = read_text(&fs::read(&file).ok()?);
        match schema.body(&text) {
            Some(body) => Some(body.to_owned()),
            None => {
                let _ = fs::remove_file(&file);
                None
            }
        }
    }

    /// The fetched text kept under `key`: its words only.
    pub fn text(&self, key: &str) -> Option<Lyrics> {
        let body = self.current(key, "lrc", cache::LYRICS_TEXT)?;
        let lyrics = Lyrics::parse_text(&body, Source::Unknown);
        (!lyrics.is_empty()).then_some(lyrics)
    }

    /// Keeps the words of `lyrics` (and where they came from) under `key`,
    /// without any times; a "nothing found" record there goes.
    pub fn store_text(&self, key: &str, lyrics: &Lyrics) -> io::Result<()> {
        let mut words = lyrics.clone();
        words.drop_times();
        self.write(
            &self.file(key, "lrc"),
            &cache::LYRICS_TEXT.wrap(&words.to_lrc()),
        )?;
        let _ = fs::remove_file(self.file(key, "none"));
        Ok(())
    }

    /// The "nothing found" record under `key` ([`Found::NotFound`] or
    /// [`Found::Instrumental`]), if it is younger than [`ASK_AGAIN`] at
    /// `now` (seconds since 1970).
    pub fn none(&self, key: &str, now: u64) -> Option<Found> {
        let text = fs::read_to_string(self.file(key, "none")).ok()?;
        let mut lines = cache::LYRICS_NONE.body(&text)?.lines();
        if lines.next()? != NONE_HEADER {
            return None;
        }
        let (kind, at) = lines.next()?.split_once(' ')?;
        let at: u64 = at.trim().parse().ok()?;
        if now.saturating_sub(at) >= ASK_AGAIN.as_secs() {
            return None;
        }
        match kind {
            "instrumental" => Some(Found::Instrumental),
            "not-found" => Some(Found::NotFound),
            _ => None,
        }
    }

    /// Records under `key` that `found` (not lyrics) was the answer at
    /// `now`.
    pub fn store_none(&self, key: &str, found: &Found, now: u64) -> io::Result<()> {
        let kind = match found {
            Found::Instrumental => "instrumental",
            Found::NotFound => "not-found",
            Found::Lyrics(_) => return Ok(()),
        };
        self.write(
            &self.file(key, "none"),
            &cache::LYRICS_NONE.wrap(&format!("{NONE_HEADER}\n{kind} {now}\n")),
        )
    }

    /// Our word times kept under `key`.
    pub fn timed(&self, key: &str) -> Option<Lyrics> {
        let body = self.current(key, "timed.lrc", cache::LYRICS_TIMED)?;
        let lyrics = Lyrics::parse(&body, Source::Unknown);
        (!lyrics.is_empty()).then_some(lyrics)
    }

    /// Keeps `lyrics` (timed by us) under `key`, as Enhanced LRC.
    pub fn store_timed(&self, key: &str, lyrics: &Lyrics) -> io::Result<()> {
        self.write(
            &self.file(key, "timed.lrc"),
            &cache::LYRICS_TIMED.wrap(&lyrics.to_lrc()),
        )
    }
}

/// The answer for a track.
#[derive(Clone, Debug, PartialEq)]
pub enum Answer {
    /// Its lyrics, and our word times for them when they are kept.
    Found {
        /// The text, with the times it came with.
        lyrics: Lyrics,
        /// Our times, from the cache, when they were made from a text of
        /// the same source.
        timed: Option<Lyrics>,
    },
    /// The track has no words.
    Instrumental,
    /// Nothing was found.
    NotFound,
    /// It could not be looked up (it may be on another try).
    Failed(String),
}

/// What a look-up uses.
#[derive(Clone)]
pub struct Finder {
    /// The cache, if any.
    pub cache: Option<Cache>,
    /// Whether LRCLIB is asked.
    pub online: bool,
    /// LRCLIB.
    pub lrclib: Lrclib,
    /// What times the words, if anything: a kept timing it did not make is
    /// passed over (and the words timed again) while it can run.
    pub timer: Option<Arc<dyn WordTimer>>,
}

/// Whether `timed`, kept, still stands for `timer`: made by it, or made by
/// anything when it cannot run now (so an older timing is better than
/// none until it can).
fn still_current(timed: &Lyrics, timer: Option<&dyn WordTimer>) -> bool {
    match timer {
        Some(t) if t.ready() => timed.timing.by.as_ref() == Some(&t.maker()),
        _ => true,
    }
}

impl Finder {
    /// The lyrics of `track`, from the first source that has them; `now`
    /// is the time for the cache's records, `wait` waits out a busy
    /// service (saying `false` to give up).
    pub fn lookup(&self, track: &Track, now: u64, wait: &dyn Fn(Duration) -> bool) -> Answer {
        self.lookup_with(track, now, wait, &|_| {})
    }

    /// [`Finder::lookup`], saying to `fetching` when a web request starts
    /// (`true`) and ends.
    pub fn lookup_with(
        &self,
        track: &Track,
        now: u64,
        wait: &dyn Fn(Duration) -> bool,
        fetching: &dyn Fn(bool),
    ) -> Answer {
        let key = cache_key(&track.path);
        let cache = self.cache.as_ref().zip(key.as_deref());
        let found = |lyrics: Lyrics| {
            let timed = cache
                .and_then(|(c, k)| c.timed(k))
                .filter(|t| t.source == lyrics.source)
                .filter(|t| still_current(t, self.timer.as_deref()));
            // Heard lyrics: lines from their words' timing (also those kept
            // before lines were made so).
            if let (Source::Transcribed(_), Some(t)) = (&lyrics.source, &timed) {
                let (lyrics, timed) = relined(t);
                return Answer::Found {
                    lyrics,
                    timed: Some(timed),
                };
            }
            Answer::Found { lyrics, timed }
        };
        let local: [&dyn LyricsSource; 2] = [&TagSource, &SidecarSource];
        for source in local {
            if let Ok(Found::Lyrics(lyrics)) = source.find(track) {
                return found(lyrics);
            }
        }
        if let Some((c, k)) = cache {
            if let Some(lyrics) = c.text(k) {
                return found(lyrics);
            }
            match c.none(k, now) {
                Some(Found::Instrumental) => return Answer::Instrumental,
                Some(_) => return Answer::NotFound,
                None => {}
            }
        }
        if !self.online || lrclib::get_url("", track).is_none() {
            return Answer::NotFound;
        }
        fetching(true);
        let fetched = self.lrclib.fetch(track, wait);
        fetching(false);
        match fetched {
            Ok(record) => {
                let answer = record.as_ref().map_or(Found::NotFound, Record::found);
                if let Some((c, k)) = cache {
                    // The words only: what LRCLIB has of times is not kept.
                    let _ = match &answer {
                        Found::Lyrics(lyrics) => c.store_text(k, lyrics),
                        _ => c.store_none(k, &answer, now),
                    };
                }
                match answer {
                    Found::Lyrics(lyrics) => found(lyrics),
                    Found::Instrumental => Answer::Instrumental,
                    Found::NotFound => Answer::NotFound,
                }
            }
            Err(e) => Answer::Failed(e),
        }
    }
}

/// What a track is, for its lyrics: its tags and length from its headers.
pub fn track_of(path: &Path) -> Option<Track> {
    let info = jodel_playback::probe_header(path).ok()?;
    Some(Track {
        path: path.to_owned(),
        title: info.tags.title.clone(),
        artist: info.tags.artist.clone(),
        album: info.tags.album.clone(),
        duration: info.duration().map(|d| d.as_secs_f64()),
        tags: info.tags.all.clone(),
        vocals: None,
    })
}

/// Reads a track's details from its path.
pub type Probe = dyn Fn(&Path) -> Option<Track> + Send + Sync;

/// How a [`LyricsWorker`] works.
#[derive(Clone)]
pub struct Options {
    /// The cache folder ([`cache_dir`]), if any.
    pub cache: Option<PathBuf>,
    /// Whether LRCLIB is asked (`karaoke_online` in the settings).
    pub online: bool,
    /// How web requests are made.
    pub http: Arc<dyn Http>,
    /// LRCLIB's address.
    pub base: String,
    /// How a track's details are read ([`track_of`]).
    pub probe: Arc<Probe>,
    /// What times the words of lyrics found without our times (`None`:
    /// nothing; they keep the times they came with).
    pub timer: Option<Arc<dyn WordTimer>>,
    /// Told when work on a file starts (`true`) and ends: a web look-up or
    /// a timing.
    pub working: Option<Arc<Working>>,
    /// Told when a file's lyrics are settled in the cache: found and timed
    /// by us (or found, with nothing to time them), none found, or the
    /// look-up failed (asked again later).
    pub done: Option<Arc<Done>>,
}

/// Told when a file's lyrics are settled, and how (for the queue's icons).
pub type Done = dyn Fn(&Path, crate::core::LyricsFacts) + Send + Sync;

/// Told when work on a file starts (`true`) and ends.
pub type Working = dyn Fn(&Path, bool) + Send + Sync;

impl Options {
    /// The player's own: the cache in the data folder, the web through
    /// [`Web`], LRCLIB at [`lrclib::BASE`].
    pub fn new(online: bool) -> Options {
        Options {
            cache: cache_dir(),
            online,
            http: Arc::new(Web::new()),
            base: lrclib::BASE.into(),
            probe: Arc::new(track_of),
            timer: None,
            working: None,
            done: None,
        }
    }
}

struct Wanted {
    queue: VecDeque<PathBuf>,
    /// The tracks last wanted (a timing for another is dropped).
    wanted: Vec<PathBuf>,
    /// Lyrics found, to be timed, in order.
    timing: VecDeque<(PathBuf, Lyrics)>,
    /// Tracks to get into the cache after those, and their lyrics found
    /// to be timed (neither delivered).
    ahead: VecDeque<PathBuf>,
    timing_ahead: VecDeque<(PathBuf, Lyrics)>,
    /// Tracks no source has lyrics for, to be heard by the speech
    /// recogniser ([`WordTimer::transcribe`]), in order; and those to get
    /// into the cache after them (not delivered).
    hearing: VecDeque<PathBuf>,
    hearing_ahead: VecDeque<PathBuf>,
    online: bool,
    stop: bool,
}

/// Looks up lyrics in the background, one track at a time.
pub struct LyricsWorker {
    shared: Arc<(Mutex<Wanted>, Condvar)>,
    timer: Option<Arc<dyn WordTimer>>,
}

impl LyricsWorker {
    /// Starts the thread; each answer goes to `deliver` with its track's
    /// path.
    pub fn new(
        options: Options,
        deliver: impl FnMut(PathBuf, Answer) + Send + 'static,
    ) -> LyricsWorker {
        let shared = Arc::new((
            Mutex::new(Wanted {
                queue: VecDeque::new(),
                wanted: Vec::new(),
                timing: VecDeque::new(),
                ahead: VecDeque::new(),
                timing_ahead: VecDeque::new(),
                hearing: VecDeque::new(),
                hearing_ahead: VecDeque::new(),
                online: options.online,
                stop: false,
            }),
            Condvar::new(),
        ));
        let deliver = Arc::new(Mutex::new(deliver));
        let working: Arc<Working> = options
            .working
            .clone()
            .unwrap_or_else(|| Arc::new(|_, _| {}));
        let done: Arc<Done> = options.done.clone().unwrap_or_else(|| Arc::new(|_, _| {}));
        if let Some(timer) = options.timer.clone() {
            timer.set_online(options.online);
            let (shared, deliver) = (Arc::clone(&shared), Arc::clone(&deliver));
            let cache = options.cache.clone().map(Cache::new);
            let (working, done) = (Arc::clone(&working), Arc::clone(&done));
            let lrclib = Lrclib::new(Arc::clone(&options.http), options.base.clone());
            let probe = Arc::clone(&options.probe);
            thread::spawn(move || {
                // Its GPU work gives way to the window's frames.
                jodel_gpu::pace::background();
                time_words(
                    &shared,
                    &*timer,
                    cache.as_ref(),
                    (&lrclib, &*probe),
                    &deliver,
                    &*working,
                    &*done,
                );
            });
        }
        let kept_timer = options.timer.clone();
        let worker = Arc::clone(&shared);
        let deliver = move |path: PathBuf, answer: Answer| {
            (deliver.lock().unwrap_or_else(|e| e.into_inner()))(path, answer);
        };
        thread::spawn(move || {
            let (lock, ready) = &*worker;
            let mut finder = Finder {
                cache: options.cache.map(Cache::new),
                online: options.online,
                lrclib: Lrclib::new(options.http, options.base),
                timer: options.timer.clone(),
            };
            // Tracks the web failed for, and when.
            let mut failed: HashMap<PathBuf, Instant> = HashMap::new();
            // Waits `pause` unless told to stop.
            let wait = |pause: Duration| {
                let wanted = lock.lock().unwrap_or_else(|e| e.into_inner());
                let (wanted, _) = ready
                    .wait_timeout_while(wanted, pause, |w| !w.stop)
                    .unwrap_or_else(|e| e.into_inner());
                !wanted.stop
            };
            loop {
                // A track wanted, else one to get ready ahead.
                let (path, ahead) = {
                    let mut wanted = lock.lock().unwrap_or_else(|e| e.into_inner());
                    loop {
                        if wanted.stop {
                            return;
                        }
                        if let Some(path) = wanted.queue.pop_front() {
                            finder.online = wanted.online;
                            break (path, false);
                        }
                        if let Some(path) = wanted.ahead.pop_front() {
                            if wanted.wanted.contains(&path) {
                                continue;
                            }
                            finder.online = wanted.online;
                            break (path, true);
                        }
                        wanted = ready.wait(wanted).unwrap_or_else(|e| e.into_inner());
                    }
                };
                let Some(track) = (options.probe)(&path) else {
                    if !ahead {
                        deliver(path, Answer::Failed("the file cannot be read".into()));
                    }
                    continue;
                };
                let resting = failed
                    .get(&path)
                    .is_some_and(|at| at.elapsed() < FAILED_PAUSE);
                let mut asked = finder.clone();
                asked.online &= !resting;
                let answer = asked.lookup_with(&track, now(), &wait, &|on| working(&path, on));
                if matches!(answer, Answer::Failed(_)) {
                    failed.insert(path.clone(), Instant::now());
                } else if !resting {
                    failed.remove(&path);
                }
                {
                    let mut wanted = lock.lock().unwrap_or_else(|e| e.into_inner());
                    if wanted.stop {
                        return;
                    }
                    // Found without our times: timed next, when it can be;
                    // else settled.
                    let mut to_time = false;
                    if let (
                        Answer::Found {
                            lyrics,
                            timed: None,
                        },
                        Some(timer),
                    ) = (&answer, &options.timer)
                    {
                        if !lyrics.is_empty() && timer.ready() {
                            let timing = if ahead {
                                &mut wanted.timing_ahead
                            } else {
                                &mut wanted.timing
                            };
                            timing.push_back((path.clone(), lyrics.clone()));
                            ready.notify_all();
                            to_time = true;
                        }
                    }
                    // None found anywhere: heard from the voice instead, when
                    // it can be (the answer comes from there).
                    if answer == Answer::NotFound
                        && options.timer.as_ref().is_some_and(|t| t.ready())
                    {
                        let hearing = if ahead {
                            &mut wanted.hearing_ahead
                        } else {
                            &mut wanted.hearing
                        };
                        hearing.push_back(path.clone());
                        ready.notify_all();
                        continue;
                    }
                    if !to_time {
                        done(&path, crate::core::LyricsFacts::of(&answer));
                    }
                }
                if !ahead {
                    deliver(path, answer);
                }
            }
        });
        LyricsWorker {
            shared,
            timer: kept_timer,
        }
    }

    /// Looks up exactly `paths`, in this order (what was asked for before
    /// and not started yet is dropped): the playing track, then the next
    /// ones in the queue.
    pub fn want(&self, paths: Vec<PathBuf>) {
        let (lock, ready) = &*self.shared;
        let mut wanted = lock.lock().unwrap_or_else(|e| e.into_inner());
        wanted.timing.retain(|(p, _)| paths.contains(p));
        wanted.wanted = paths.clone();
        wanted.queue = paths.into();
        ready.notify_all();
    }

    /// Looks up `paths` ahead, in this order, after what is wanted, and
    /// times the words of what is found, into the cache only (nothing is
    /// delivered; what was asked for before and not started is dropped).
    pub fn prepare(&self, paths: Vec<PathBuf>) {
        let (lock, ready) = &*self.shared;
        let mut wanted = lock.lock().unwrap_or_else(|e| e.into_inner());
        wanted.timing_ahead.retain(|(p, _)| paths.contains(p));
        wanted.ahead = paths.into();
        ready.notify_all();
    }

    /// Turns the LRCLIB look-ups on or off, from the next track.
    pub fn set_online(&self, online: bool) {
        let (lock, _) = &*self.shared;
        lock.lock().unwrap_or_else(|e| e.into_inner()).online = online;
        if let Some(timer) = &self.timer {
            timer.set_online(online);
        }
    }
}

impl Drop for LyricsWorker {
    fn drop(&mut self) {
        let (lock, ready) = &*self.shared;
        lock.lock().unwrap_or_else(|e| e.into_inner()).stop = true;
        ready.notify_all();
    }
}

/// LRCLIB's other texts of a song, asked for only when needed.
pub type Others<'a> = dyn Fn() -> Vec<Lyrics> + 'a;

/// Times the words of lyrics against a track's voice.
pub trait WordTimer: Send + Sync {
    /// Whether it can run now (its model is there, or can be fetched).
    fn ready(&self) -> bool;

    /// What its timings are made by ([`jodel_lyrics::Timing::by`]): a kept
    /// timing by anything else (an older aligner) is made again.
    fn maker(&self) -> Maker;

    /// `lyrics` with each word timed against the voice of `path`; when
    /// they do not fit what is sung, the best fitting of `others` instead
    /// ([`fitting`]), timed.
    fn time(&self, path: &Path, lyrics: &Lyrics, others: &Others) -> Result<Lyrics, String>;

    /// The words sung in the voice of `path`, heard by a speech recogniser
    /// (for a track no source has lyrics for): the text, its lines cut
    /// where the recogniser's punctuation (or a line's length) says
    /// ([`transcript_lyrics`]), and the text timed. Its source is
    /// [`Source::Transcribed`]: no trusted source, so it may be wrong.
    fn transcribe(&self, _path: &Path) -> Result<(Lyrics, Lyrics), String> {
        Err("no speech recogniser".into())
    }

    /// Whether it may fetch what it is missing (the lyrics' online
    /// setting).
    fn set_online(&self, _online: bool) {}
}

/// Timed heard lyrics with their lines made again where the words' timing
/// says they break ([`jodel_lyrics::lines::relined`]): the text (untimed) and
/// the timing.
pub fn relined(timed: &Lyrics) -> (Lyrics, Lyrics) {
    let timed = jodel_lyrics::lines::relined(timed);
    let mut text = timed.clone();
    text.drop_times();
    (text, timed)
}

/// The most words a line of a transcript gets (a longer sentence is cut
/// into even pieces).
const HEARD_LINE_MOST: usize = 9;

/// Lyrics from what a speech recogniser heard, by `maker`, as first cut
/// before their words are timed (then [`relined`] by the timing): each part's
/// words cut into lines at the ends of sentences (`.`, `!`, `?`, `;`), at
/// commas where a sentence is long, and into even pieces of at most
/// [`HEARD_LINE_MOST`] words; untimed.
pub fn transcript_lyrics(transcript: &jodel_lyrics::asr::Transcript, maker: Maker) -> Lyrics {
    let mut lines = Vec::new();
    for part in &transcript.parts {
        let mut sentence: Vec<&str> = Vec::new();
        let words: Vec<&str> = part.text.split_whitespace().collect();
        for (k, word) in words.iter().enumerate() {
            sentence.push(word);
            let ends = word.ends_with(['.', '!', '?', ';', '…', '。', '！', '？'])
                || (word.ends_with([',', '，']) && sentence.len() >= 5)
                || k + 1 == words.len();
            if ends {
                let pieces = sentence.len().div_ceil(HEARD_LINE_MOST);
                let each = sentence.len().div_ceil(pieces);
                for piece in sentence.chunks(each) {
                    lines.push(jodel_lyrics::Line::from_text(&piece.join(" ")));
                }
                sentence.clear();
            }
        }
    }
    Lyrics {
        lines,
        source: Source::Transcribed(maker),
        timing: jodel_lyrics::Timing::default(),
    }
}

/// Of `lyrics` and `others`, the text to time against a voice in which the
/// recogniser `heard` this: `lyrics` when they fit ([`FIT_LEAST`]) or too
/// little was heard to judge (under [`HEARD_LEAST`] words), else the best
/// fitting of `others` when it fits, else `lyrics` still. And its fit, when
/// judged.
pub fn fitting(lyrics: &Lyrics, heard: &str, others: &Others) -> (Lyrics, Option<Fit>) {
    use jodel_lyrics::fit::{lyrics_fit, words};
    if words(heard).len() < HEARD_LEAST {
        return (lyrics.clone(), None);
    }
    let fit = lyrics_fit(lyrics, heard);
    if fit >= FIT_LEAST {
        return (lyrics.clone(), Some(Fit::new(fit, false)));
    }
    let best = others()
        .into_iter()
        .map(|o| {
            let f = lyrics_fit(&o, heard);
            (o, f)
        })
        .max_by(|a, b| a.1.total_cmp(&b.1));
    match best {
        Some((other, f)) if f >= FIT_LEAST && f > fit => (other, Some(Fit::new(f, true))),
        _ => (lyrics.clone(), Some(Fit::new(fit, false))),
    }
}

/// The least words heard for a fit to be judged (fewer: a voice too quiet
/// or too short to tell).
pub const HEARD_LEAST: usize = 20;

/// The timing thread: takes the lyrics found without our times, in order,
/// times them (a text from LRCLIB that does not fit the song swapped for
/// one of its others that does: [`fitting`]), keeps them in the cache and
/// sends them on (for a track still wanted).
fn time_words<D: FnMut(PathBuf, Answer)>(
    shared: &(Mutex<Wanted>, Condvar),
    timer: &dyn WordTimer,
    cache: Option<&Cache>,
    (lrclib, probe): (&Lrclib, &Probe),
    deliver: &Mutex<D>,
    working: &Working,
    done_with: &Done,
) {
    let (lock, ready) = shared;
    loop {
        // A timing first, else a track to hear.
        let (path, lyrics) = {
            let mut wanted = lock.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if wanted.stop {
                    return;
                }
                if let Some(job) = wanted.timing.pop_front() {
                    break job;
                }
                if let Some(job) = wanted.timing_ahead.pop_front() {
                    break job;
                }
                if let Some(path) = wanted.hearing.pop_front() {
                    drop(wanted);
                    hear(shared, timer, cache, deliver, working, done_with, path);
                    wanted = lock.lock().unwrap_or_else(|e| e.into_inner());
                    continue;
                }
                if let Some(path) = wanted.hearing_ahead.pop_front() {
                    drop(wanted);
                    hear(shared, timer, cache, deliver, working, done_with, path);
                    wanted = lock.lock().unwrap_or_else(|e| e.into_inner());
                    continue;
                }
                wanted = ready.wait(wanted).unwrap_or_else(|e| e.into_inner());
            }
        };
        // Timed meanwhile (wanted and ahead both): not again.
        let key = cache_key(&path);
        let done = cache
            .zip(key.as_deref())
            .and_then(|(c, k)| c.timed(k))
            .filter(|t| t.source == lyrics.source)
            .filter(|t| still_current(t, Some(timer)));
        if let Some(timed) = done {
            done_with(&path, crate::core::LyricsFacts::Timed);
            let still = {
                let wanted = lock.lock().unwrap_or_else(|e| e.into_inner());
                !wanted.stop && wanted.wanted.contains(&path)
            };
            if still {
                let answer = Answer::Found {
                    lyrics,
                    timed: Some(timed),
                };
                (deliver.lock().unwrap_or_else(|e| e.into_inner()))(path, answer);
            }
            continue;
        }
        working(&path, true);
        // LRCLIB's other texts of the song, asked for only when this one
        // does not fit (and only of LRCLIB's: the file's own are kept).
        let others = || -> Vec<Lyrics> {
            let online = {
                let wanted = lock.lock().unwrap_or_else(|e| e.into_inner());
                !wanted.stop && wanted.online
            };
            if lyrics.source != Source::Lrclib || !online {
                return Vec::new();
            }
            let wait = |pause: Duration| {
                thread::sleep(pause);
                true
            };
            probe(&path)
                .and_then(|track| lrclib.candidates(&track, &wait).ok())
                .unwrap_or_default()
        };
        let timed = timer.time(&path, &lyrics, &others);
        working(&path, false);
        let timed = match timed {
            Ok(timed) => timed,
            // The text stays as it came; nothing to say, nothing more to do.
            Err(_) => {
                done_with(&path, crate::core::LyricsFacts::Untimed);
                continue;
            }
        };
        // Another text chosen: it is the song's from now on.
        let swapped =
            jodel_lyrics::fit::lyrics_words(&timed) != jodel_lyrics::fit::lyrics_words(&lyrics);
        let lyrics = if swapped {
            let mut text = timed.clone();
            text.drop_times();
            text
        } else {
            lyrics
        };
        if let (Some(cache), Some(key)) = (cache, key) {
            if swapped {
                let _ = cache.store_text(&key, &lyrics);
            }
            let _ = cache.store_timed(&key, &timed);
        }
        done_with(&path, crate::core::LyricsFacts::Timed);
        let still = {
            let wanted = lock.lock().unwrap_or_else(|e| e.into_inner());
            !wanted.stop && wanted.wanted.contains(&path)
        };
        if still {
            let answer = Answer::Found {
                lyrics,
                timed: Some(timed),
            };
            (deliver.lock().unwrap_or_else(|e| e.into_inner()))(path, answer);
        }
    }
}

/// Hears the words sung in `path` ([`WordTimer::transcribe`]), for a track
/// no source has lyrics for: kept in the cache as the track's text (marked
/// as heard, not from a trusted source) and timing, and delivered when the
/// track is still wanted. Too little heard: kept as having no words. Else
/// (no recogniser here): none found.
fn hear<D: FnMut(PathBuf, Answer)>(
    shared: &(Mutex<Wanted>, Condvar),
    timer: &dyn WordTimer,
    cache: Option<&Cache>,
    deliver: &Mutex<D>,
    working: &Working,
    done_with: &Done,
    path: PathBuf,
) {
    working(&path, true);
    let heard = timer.transcribe(&path);
    working(&path, false);
    let key = cache_key(&path);
    let answer = match heard {
        Ok((text, timed)) => {
            if let (Some(cache), Some(key)) = (cache, key.as_deref()) {
                let _ = cache.store_text(key, &text);
                let _ = cache.store_timed(key, &timed);
            }
            Answer::Found {
                lyrics: text,
                timed: Some(timed),
            }
        }
        Err(e) if e == TOO_LITTLE_HEARD => {
            if let (Some(cache), Some(key)) = (cache, key.as_deref()) {
                let _ = cache.store_none(key, &Found::Instrumental, now());
            }
            Answer::Instrumental
        }
        Err(_) => Answer::NotFound,
    };
    done_with(&path, crate::core::LyricsFacts::of(&answer));
    let still = {
        let (lock, _) = shared;
        let wanted = lock.lock().unwrap_or_else(|e| e.into_inner());
        !wanted.stop && wanted.wanted.contains(&path)
    };
    if still {
        (deliver.lock().unwrap_or_else(|e| e.into_inner()))(path, answer);
    }
}

/// Why a transcription gave nothing: the voice sings too few words.
pub const TOO_LITTLE_HEARD: &str = "too little was heard";
/// The speech model's file name.
pub const MODEL_FILE: &str = "wav2vec2-base-960h.safetensors";
/// Where it is downloaded from (Apache 2.0, by Meta).
pub const MODEL_URL: &str =
    "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/model.safetensors";
/// Its length in bytes.
pub const MODEL_BYTES: u64 = 377_607_901;
/// Its SHA-256.
pub const MODEL_SHA256: &str = "8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a";

/// The speech model where it is (not checked: [`crate::setup`] checks it):
/// `models/` beside the executable, or in the app bundle's
/// `Contents/Resources/models/` (where one sent along with the player
/// sits); else `models/` in the data folder, where setup puts it. `None`
/// when it is in none of them.
pub fn model_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok();
    let beside = exe.as_deref().and_then(Path::parent).map(Path::to_path_buf);
    let data = crate::dirs::Data::current().map(|d| d.0);
    model_in(beside.as_deref(), data.as_deref())
}

/// [`model_path`] for an executable in `exe_dir` and the data folder
/// `data_dir`.
pub fn model_in(exe_dir: Option<&Path>, data_dir: Option<&Path>) -> Option<PathBuf> {
    file_in(MODEL_FILE, exe_dir, data_dir)
}

/// The model file `name` in the places [`model_path`] looks.
fn file_in(name: &str, exe_dir: Option<&Path>, data_dir: Option<&Path>) -> Option<PathBuf> {
    let mut places = Vec::new();
    if let Some(dir) = exe_dir {
        places.push(dir.join(MODEL_DIR));
        places.push(dir.join("..").join("Resources").join(MODEL_DIR));
    }
    if let Some(dir) = data_dir {
        places.push(dir.join(MODEL_DIR));
    }
    places
        .into_iter()
        .map(|d| d.join(name))
        .find(|p| p.is_file())
}

/// The speech recogniser's file: Qwen3-ASR-0.6B (Apache 2.0, by Alibaba's
/// Qwen team) packed at 4 bits ([`jodel_lyrics::asr::packed`]), about 0.5 GB.
pub const ASR_FILE: &str = "qwen3-asr-0.6b-q4.jdlasr";
/// The packed file's length in bytes. Packing is the same on every machine
/// (each group of weights is fitted on its own, in a fixed order), so the
/// file is too: setup checks it against [`ASR_SHA256`]. A change to how it
/// is packed ([`jodel_lyrics::asr::packed::VERSION`], the calibration) makes
/// another file: these two are then measured again.
pub const ASR_BYTES: u64 = 588_661_472;
/// The packed file's SHA-256.
pub const ASR_SHA256: &str = "4a480c989e5d47e225a31fd798ec258c65e4c6c8f1bef51ca6f65df1396df151";
/// The checkpoint it is packed from, at a fixed revision.
pub(crate) const ASR_BASE: &str =
    "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/5eb144179a02acc5e5ba31e748d22b0cf3e303b0";
/// The checkpoint's files: name, length, SHA-256.
pub(crate) const ASR_PARTS: [(&str, u64, &str); 3] = [
    (
        "model.safetensors",
        1_876_091_704,
        "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea",
    ),
    (
        "vocab.json",
        2_776_833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
    (
        "tokenizer_config.json",
        12_487,
        "4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c",
    ),
];

/// The speech recogniser's file, where the build put it (as
/// [`model_path`]).
pub fn asr_model_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok();
    let beside = exe.as_deref().and_then(Path::parent).map(Path::to_path_buf);
    let data = crate::dirs::Data::current().map(|d| d.0);
    asr_model_in(beside.as_deref(), data.as_deref())
}

/// [`asr_model_path`] for an executable in `exe_dir` and the data folder
/// `data_dir`.
pub fn asr_model_in(exe_dir: Option<&Path>, data_dir: Option<&Path>) -> Option<PathBuf> {
    file_in(ASR_FILE, exe_dir, data_dir)
}

/// The folder the speech model sits in, where it is looked for.
pub const MODEL_DIR: &str = "models";

/// The speech model and the speech recogniser, as setup found them
/// ([`crate::setup`]): there, and checked against their built-in checksums.
/// Read through [`crate::mover::locate`]: while the data folder moves, one
/// not yet moved is read where it was.
pub struct Models {
    /// The speech model.
    pub speech: PathBuf,
    /// The speech recogniser.
    pub asr: PathBuf,
}

impl Models {
    /// The models at `speech` and `asr`.
    pub fn new(speech: PathBuf, asr: PathBuf) -> Arc<Models> {
        Arc::new(Models { speech, asr })
    }

    /// Where the speech model is to be read now.
    pub fn speech_now(&self) -> PathBuf {
        crate::mover::locate(&self.speech)
    }

    /// Where the speech recogniser is to be read now.
    pub fn asr_now(&self) -> PathBuf {
        crate::mover::locate(&self.asr)
    }

    /// Whether the speech recogniser is there.
    pub fn has_asr(&self) -> bool {
        jodel_lyrics::asr::packed::PackedModel::is_current(&self.asr_now())
    }
}

/// The word timing of the speech model: the voice separated on the GPU,
/// wav2vec2 on the GPU or half the processor's threads (the rest left to
/// playback), and the CTC alignment over the whole song (no times but
/// ours). Any text is first checked against what the speech recogniser
/// hears in the voice (on the GPU, where it and the recogniser are there),
/// one from LRCLIB swapped for one of LRCLIB's others when it does not fit
/// ([`fitting`]); what it heard, in the transcript's parts and in glimpses
/// ([`jodel_lyrics::glimpses`]), pins the words to where it was sung.
pub struct SpeechTimer {
    models: Arc<Models>,
    model: Mutex<Option<Arc<dyn jodel_lyrics::ctc::Acoustic + Send + Sync>>>,
    /// The recogniser once read: `None` inside when it cannot run here.
    asr: Mutex<Option<Option<Arc<jodel_gpu::GpuAsr>>>>,
}

impl SpeechTimer {
    /// The timer for `models` (each read when first needed).
    pub fn new(models: Arc<Models>) -> SpeechTimer {
        SpeechTimer {
            models,
            model: Mutex::new(None),
            asr: Mutex::new(None),
        }
    }

    /// The speech recogniser on the GPU, when it is there and there is one.
    fn recogniser(&self) -> Option<Arc<jodel_gpu::GpuAsr>> {
        if !self.models.has_asr() {
            return None;
        }
        let mut kept = self.asr.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(asr) = kept.as_ref() {
            return asr.clone();
        }
        let asr = jodel_lyrics::asr::packed::PackedModel::open(&self.models.asr_now())
            .and_then(|packed| jodel_gpu::GpuAsr::new(&packed))
            .ok()
            .map(Arc::new);
        *kept = Some(asr.clone());
        asr
    }
}

impl WordTimer for SpeechTimer {
    fn ready(&self) -> bool {
        self.models.speech_now().is_file()
    }

    fn maker(&self) -> Maker {
        jodel_lyrics::ctc::aligner_maker(&jodel_lyrics::speech::maker())
    }

    fn time(&self, path: &Path, lyrics: &Lyrics, others: &Others) -> Result<Lyrics, String> {
        use jodel_lyrics::ctc::CtcAligner;
        let model = self.acoustic()?;
        let voice = track_voice(path)?;
        // Any text checked against what is sung (one from LRCLIB swapped
        // for one of its others when it does not fit: `others` has none
        // for the file's own); what was heard, in the transcript's parts
        // and in glimpses, also pins the words to where it was.
        let (lyrics, fit, parts, glimpses) = match self.recogniser() {
            Some(asr) => match asr.transcribe(&voice) {
                Ok(heard) => {
                    let (lyrics, fit) = fitting(lyrics, &heard.text, others);
                    // A text that fits, with the repeats it leaves out.
                    let fits = fit.is_some_and(|f| f32::from(f.percent) >= 100.0 * FIT_LEAST);
                    let lyrics = if fits {
                        jodel_lyrics::repeats::with_repeats(&lyrics, &heard.parts)
                    } else {
                        lyrics
                    };
                    let glimpses = asr.glimpses(&voice).unwrap_or_default();
                    (lyrics, fit, heard.parts, glimpses)
                }
                Err(_) => (lyrics.clone(), None, Vec::new(), Vec::new()),
            },
            None => (lyrics.clone(), None, Vec::new(), Vec::new()),
        };
        let mut timed = CtcAligner::new(model).align_heard_in(&voice, &lyrics, &parts, &glimpses);
        timed.timing.fit = fit;
        Ok(timed)
    }

    fn transcribe(&self, path: &Path) -> Result<(Lyrics, Lyrics), String> {
        use jodel_lyrics::ctc::CtcAligner;
        let asr = self.recogniser().ok_or("no speech recogniser here")?;
        let model = self.acoustic()?;
        let voice = track_voice(path)?;
        let heard = asr.transcribe(&voice)?;
        if jodel_lyrics::fit::words(&heard.text).len() < HEARD_LEAST {
            return Err(TOO_LITTLE_HEARD.into());
        }
        let text = transcript_lyrics(&heard, asr_maker());
        let glimpses = asr.glimpses(&voice).unwrap_or_default();
        let timed = CtcAligner::new(model).align_heard_in(&voice, &text, &heard.parts, &glimpses);
        // The lines made again where the words' timing says they break.
        Ok(relined(&timed))
    }
}

impl SpeechTimer {
    /// The speech model that times the words, read once: on the GPU where
    /// there is one (25 times as fast, and the processor left to
    /// playback), else on half the processor's threads.
    fn acoustic(&self) -> Result<Arc<dyn jodel_lyrics::ctc::Acoustic + Send + Sync>, String> {
        if !self.models.speech_now().is_file() {
            return Err("the speech model is not here".into());
        }
        let mut kept = self.model.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(m) = kept.as_ref() {
            return Ok(Arc::clone(m));
        }
        let mut cpu = jodel_lyrics::speech::Wav2Vec2::open(&self.models.speech_now())?;
        let m: Arc<dyn jodel_lyrics::ctc::Acoustic + Send + Sync> =
            match jodel_gpu::GpuSpeech::new(&cpu) {
                Ok(gpu) => Arc::new(gpu),
                Err(_) => {
                    let threads = thread::available_parallelism().map_or(2, |n| n.get());
                    cpu.set_threads((threads / 2).max(1));
                    Arc::new(cpu)
                }
            };
        *kept = Some(Arc::clone(&m));
        Ok(m)
    }
}

/// What made a transcript's text: the speech recogniser, as packed.
pub fn asr_maker() -> Maker {
    Maker::new(
        "qwen3-asr-0.6b-q4",
        u32::from(jodel_lyrics::asr::packed::VERSION),
    )
}

/// The voice of `path`, mono at 16 kHz: from the track's kept stems, else
/// separated now.
fn track_voice(path: &Path) -> Result<Vec<f32>, String> {
    let kept = crate::stems_cache::dir().and_then(|dir| crate::stems_cache::open(&dir, path));
    match kept {
        Some(file) => voice_16k(file),
        None => Ok(separate_16k(path, false)?.0),
    }
}
/// The voice in a track's kept stems, mono at 16 kHz: the vocals' two
/// channels averaged and resampled. Nothing is separated.
pub fn voice_16k(mut file: jodel_playback::StemFile) -> Result<Vec<f32>, String> {
    use jodel_playback::stems::Stem;
    let vocals = Stem::ALL
        .iter()
        .position(|&s| s == Stem::Vocals)
        .unwrap_or(2);
    let mut r = jodel_playback::Resampler::new(jodel_playback::stems::RATE, 16_000, 1);
    let (mut voice, mut out) = (Vec::new(), Vec::new());
    let mut at = 0;
    while at < file.frames() {
        let stems = file
            .read(at, jodel_playback::stems::RATE as usize * 10)
            .map_err(|e| e.to_string())?;
        let [l, rt] = &stems[vocals];
        if l.is_empty() {
            break;
        }
        at += l.len() as u64;
        voice.clear();
        voice.extend(
            l.iter()
                .zip(rt)
                .map(|(a, b)| (f64::from(*a) + f64::from(*b)) / 2.0),
        );
        r.process_f64(&voice, &mut out);
    }
    r.finish_f64(&mut out);
    Ok(out.into_iter().map(|v| v as f32).collect())
}

/// The voice of `path` (and, with `mix`, the whole mix), mono at 16 kHz:
/// the file decoded, separated into its stems on the GPU, the vocals'
/// two channels averaged.
pub fn separate_16k(path: &Path, mix: bool) -> Result<(Vec<f32>, Vec<f32>), String> {
    use jodel_playback::stems::{Stem, StereoFeed};
    use jodel_rhythm::stems::{Splitter, StemBlock};
    let gpu = if jodel_gpu::pace::is_background() {
        jodel_gpu::GpuSeparator::background()
    } else {
        jodel_gpu::GpuSeparator::shared()
    }
    .ok_or("no GPU for the stems")?;
    let vocals = Stem::ALL
        .iter()
        .position(|&s| s == Stem::Vocals)
        .unwrap_or(2);
    let mut source = jodel_playback::Source::open(path).map_err(|e| e.to_string())?;
    let (rate, channels) = (
        source.info().rate,
        usize::from(source.info().channels).max(1),
    );
    let mut feed = StereoFeed::new(rate);
    let mut splitter = Splitter::overlapped(gpu);
    let mut voice: Vec<f64> = Vec::new();
    let mut whole: Vec<f64> = Vec::new();
    let mut take = |b: &StemBlock| {
        let [l, r] = b.stems[vocals];
        if voice.len() < b.start + l.len() {
            voice.resize(b.start + l.len(), 0.0);
        }
        for (i, (a, c)) in l.iter().zip(r).enumerate() {
            voice[b.start + i] = (f64::from(*a) + f64::from(*c)) / 2.0;
        }
    };
    let mut failed = None;
    let mut put = |left: Vec<f32>, right: Vec<f32>| {
        if mix {
            whole.extend(
                left.iter()
                    .zip(&right)
                    .map(|(a, b)| (f64::from(*a) + f64::from(*b)) / 2.0),
            );
        }
        if let Err(e) = splitter.push(&left, &right, &mut take) {
            failed.get_or_insert(e);
        }
    };
    let mut block = Vec::new();
    loop {
        block.clear();
        let frames = source
            .read_unquantized(&mut block)
            .map_err(|e| e.to_string())?;
        if frames == 0 {
            break;
        }
        feed.push(&block, channels, &mut put);
    }
    feed.finish(&mut put);
    if let Some(e) = failed {
        return Err(e);
    }
    splitter.finish(&mut take)?;
    let to_16k = |x: &[f64]| {
        let mut r = jodel_playback::Resampler::new(jodel_playback::stems::RATE, 16_000, 1);
        let mut out = Vec::new();
        r.process_f64(x, &mut out);
        r.finish_f64(&mut out);
        out.into_iter().map(|v| v as f32).collect::<Vec<f32>>()
    };
    Ok((to_16k(&voice), to_16k(&whole)))
}

/// Downloads `url` to `to` (through a file beside it, checked against its
/// length `bytes` and its SHA-256 `sha256` before it takes its name),
/// telling `progress` the bytes so far and in all.
pub(crate) fn download(
    url: &str,
    to: &Path,
    bytes: u64,
    sha256: &str,
    mut progress: impl FnMut(u64, u64),
) -> Result<(), String> {
    use std::io::{Read, Write};
    if let Some(dir) = to.parent() {
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let mut name = to.file_name().unwrap_or_default().to_os_string();
    name.push(".partial");
    let partial = to.with_file_name(name);
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .user_agent(USER_AGENT)
        .build()
        .into();
    let mut response = agent
        .get(url)
        .call()
        .map_err(|e| format!("the download failed: {e}"))?;
    let mut body = response.body_mut().with_config().limit(bytes + 1).reader();
    let mut file = fs::File::create(&partial).map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    let mut done = 0u64;
    loop {
        let n = body
            .read(&mut buf)
            .map_err(|e| format!("the download failed: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        hash.update(&buf[..n]);
        done += n as u64;
        progress(done, bytes);
    }
    // On the disk before it takes its name.
    file.sync_all().map_err(|e| e.to_string())?;
    drop(file);
    let digest = hash.hex();
    if done != bytes || digest != sha256 {
        let _ = fs::remove_file(&partial);
        return Err(format!(
            "the download of {url} is not the file ({done} bytes, SHA-256 {digest})"
        ));
    }
    durable::rename_durable(&partial, to).map_err(|e| e.to_string())
}

/// SHA-256 (FIPS 180-4), to check a download.
pub struct Sha256 {
    state: [u32; 8],
    block: Vec<u8>,
    length: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Sha256::new()
    }
}

impl Sha256 {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    /// A hash of nothing yet.
    pub fn new() -> Sha256 {
        Sha256 {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            block: Vec::with_capacity(64),
            length: 0,
        }
    }

    /// Takes in `data`.
    pub fn update(&mut self, mut data: &[u8]) {
        self.length += data.len() as u64;
        while !data.is_empty() {
            let take = (64 - self.block.len()).min(data.len());
            self.block.extend_from_slice(&data[..take]);
            data = &data[take..];
            if self.block.len() == 64 {
                let block: [u8; 64] = self.block[..].try_into().unwrap();
                self.compress(&block);
                self.block.clear();
            }
        }
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (i, word) in block.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(*word);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for (&k, &wi) in Self::K.iter().zip(&w) {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(k)
                .wrapping_add(wi);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            (h, g, f, e, d, c, b, a) = (g, f, e, d.wrapping_add(t1), c, b, a, t1.wrapping_add(t2));
        }
        for (s, v) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *s = s.wrapping_add(v);
        }
    }

    /// The digest, in lowercase hex.
    pub fn hex(mut self) -> String {
        let bits = self.length.wrapping_mul(8);
        self.update(&[0x80]);
        while self.block.len() != 56 {
            self.update(&[0]);
        }
        self.update(&bits.to_be_bytes());
        self.state.iter().map(|w| format!("{w:08x}")).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;

    #[test]
    fn sha256_gives_the_standard_digests() {
        let hex = |data: &[u8]| {
            let mut h = Sha256::new();
            h.update(data);
            h.hex()
        };
        assert_eq!(
            hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // Two blocks, fed in odd pieces.
        let long = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        let mut h = Sha256::new();
        for piece in long.chunks(7) {
            h.update(piece);
        }
        assert_eq!(
            h.hex(),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

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
        assert_eq!(
            cache.text(&key).unwrap().lines[0].text(),
            "You search through the years"
        );
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
            done: None,
        };
        let (tx, rx) = mpsc::channel();
        let worker = LyricsWorker::new(options, move |path, answer| {
            let _ = tx.send((path, answer));
        });
        worker.prepare(vec![song.clone()]);
        let wait = Duration::from_secs(10);
        assert_eq!(busy_rx.recv_timeout(wait).unwrap(), (song.clone(), true));
        assert_eq!(busy_rx.recv_timeout(wait).unwrap(), (song.clone(), false));
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
        // Its words only: the sidecar's time is gone.
        let Answer::Found { lyrics, .. } = &answer else {
            unreachable!()
        };
        assert!(lyrics.lines.iter().all(|l| l.time.is_none()));
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
        // Kept, its words only: asked again, the web is not.
        let kept_at = dir
            .join("cache")
            .join(format!("{}.lrc", cache_key(&file).unwrap()));
        let kept = fs::read_to_string(&kept_at).unwrap();
        assert_eq!(
            crate::cache::LYRICS_TEXT.body(&kept).map(str::trim_end),
            Some("[re:jodel-lyrics]\n[source:lrclib]\nHello there\nGeneral Kenobi")
        );
        assert_eq!(f.lookup(&t, 2000, &no_wait), answer);
        assert_eq!(web.asked().len(), 1);
        // A text kept by an older player (with the times it came with) is
        // deleted when read, and the words asked for again.
        fs::write(&kept_at, "[source:lrclib]\n[00:12.00]Hello there\n").unwrap();
        fs::write(
            kept_at.with_extension("timed.lrc"),
            "[00:12.00]<00:12.00>Hello\n",
        )
        .unwrap();
        let cache = Cache::new(dir.join("cache"));
        let key = cache_key(&file).unwrap();
        assert_eq!(cache.text(&key), None);
        assert_eq!(cache.timed(&key), None);
        assert!(!kept_at.exists() && !kept_at.with_extension("timed.lrc").exists());
        assert_eq!(f.lookup(&t, 3000, &no_wait), answer);
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
        assert_eq!(*waits.lock().unwrap(), [Duration::from_secs(3), BUSY_WAIT]);
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
}
