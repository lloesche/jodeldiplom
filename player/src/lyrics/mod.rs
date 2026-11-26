//! Lyrics of the tracks for karaoke: found in the background, and kept on
//! disk.
//!
//! A [`LyricsWorker`] thread looks up the tracks it is asked for, one at a
//! time, in this order:
//!
//! 1. the file's lyrics tag ([`TagSource`]);
//! 2. a `.lrc` file beside it ([`SidecarSource`]);
//! 3. what was fetched before, from the cache;
//! 4. LRCLIB ([`web::Lrclib`]): first the one record for the artist, title,
//!    album and length, then a search by title and artist. A record marked
//!    instrumental is kept as "no lyrics". Only when `online` is on.
//!
//! The web requests go one after another, at most one a second
//! ([`web::SPACING`]), with a `User-Agent` of `jodeldiplom/<version>` and
//! nothing about the user; a `429` (or `503`) answer is waited out for as
//! long as its `Retry-After` says, within reason, and asked again.
//!
//! The cache is `lyrics/` in the data folder ([`crate::dirs::Data::lyrics`]),
//! a few files a track named
//! by a hash of its path, size and modification time (as the beat
//! analysis's cache is):
//!
//! - `{key}.lrc`: the text as fetched, after a `[source:...]` line, with
//!   when the source synced each line (LRCLIB's synced lyrics, an LRC
//!   file's times) as the scaffold the aligner checks and holds the lines
//!   near (`jodel_lyrics::scaffold`), never as the lines' own times;
//! - `{key}.none`: "nothing found" or "instrumental", with when; asked
//!   again after [`store::ASK_AGAIN`];
//! - `{key}.timed.lrc`: our own word times, as Enhanced LRC, with a header
//!   naming the source and the aligner, kept by
//!   [`store::Cache::store_timed`];
//! - `{key}.edited.lrc`: our word times of the lyrics as edited by hand,
//!   named by the edits they were made with ([`store::Cache::store_edited`]).
//!
//! A song edited by hand ([`edits`], kept apart as the user's work) has its
//! edits in place of all that: their text is its text (the one found only
//! lays its source's line times on it as the scaffold,
//! [`jodel_lyrics::scaffold::attach`]), timed as it is
//! ([`WordTimer::time_edited`]: the words moved by hand held where they
//! were set, the rest fitted around them), its words marked as edited.
//! [`LyricsWorker::edited`] keeps new edits and times the song with them,
//! delivering the song only once timed.
//!
//! The modules: [`web`] (the requests, LRCLIB), [`store`] (the cache),
//! [`edits`] (the edits kept), [`models`] (the models' files), [`timer`]
//! (the speech model's timing and the voice it times against).

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use jodel_lyrics::lrclib::{self, Record};
use jodel_lyrics::prepare::Others;
use jodel_lyrics::sources::{SidecarSource, TagSource};
use jodel_lyrics::{Found, Lyrics, LyricsSource, Maker, Source, Track};

pub mod edits;
pub mod models;
pub mod store;
pub mod timer;
pub mod web;

use edits::EditStore;
use store::{cache_key, now, Cache, Kept};
use web::{Http, Lrclib, Web};

/// How long a track whose look-up failed (the network, the service) is
/// not looked up on the web again.
const FAILED_PAUSE: Duration = Duration::from_secs(10 * 60);

/// Why a track's lyrics could not be looked up: the text kept for it is
/// of an older version ([`Kept::Outdated`]), to be fetched again, and
/// LRCLIB cannot be asked now.
pub const OUTDATED: &str =
    "the lyrics kept are to be fetched again, and LRCLIB cannot be asked now";

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
    /// The lyrics edited by hand, if any.
    pub edits: Option<EditStore>,
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
        self.look(track, now, wait, fetching).0
    }

    /// [`Finder::lookup_with`], and whether the answer is the song's edits
    /// ([`Finder::edited`]).
    fn look(
        &self,
        track: &Track,
        now: u64,
        wait: &dyn Fn(Duration) -> bool,
        fetching: &dyn Fn(bool),
    ) -> (Answer, bool) {
        if let Some(edits) = self.edits.as_ref().and_then(|e| e.load(&track.path)) {
            return (self.edited(track, &edits), true);
        }
        let answer = self.found(track, now, wait, fetching);
        (answer, false)
    }

    /// The lyrics of a song edited by hand as `edits`: their text, with the
    /// line times of the text found here (its tags, a file beside it, the
    /// cache; never fetched) as its scaffold, and our times of it when kept
    /// for these edits by the timer in use.
    fn edited(&self, track: &Track, edits: &Lyrics) -> Answer {
        let cache = self.cache.as_ref().zip(cache_key(&track.path));
        let local: [&dyn LyricsSource; 2] = [&TagSource, &SidecarSource];
        let found = local
            .iter()
            .find_map(|s| match s.find(track) {
                Ok(Found::Lyrics(l)) => Some(l),
                _ => None,
            })
            .or_else(|| match cache.as_ref().map(|(c, k)| c.text(k)) {
                Some(Kept::Current(l)) => Some(l),
                _ => None,
            });
        let mut text = edits.clone();
        if let Some(found) = &found {
            jodel_lyrics::scaffold::attach(&mut text, found);
        }
        let timed = cache
            .and_then(|(c, k)| c.edited(&k, edits))
            .filter(|t| still_current(t, self.timer.as_deref()));
        Answer::Found {
            lyrics: text,
            timed,
        }
    }

    /// [`Finder::lookup_with`] for a song not edited.
    fn found(
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
        let mut outdated = false;
        if let Some((c, k)) = cache {
            match c.text(k) {
                Kept::Current(lyrics) => return found(lyrics),
                Kept::Outdated => outdated = true,
                Kept::Missing => {}
            }
            match c.none(k, now) {
                Some(Found::Instrumental) => return Answer::Instrumental,
                Some(_) => return Answer::NotFound,
                None => {}
            }
        }
        let nameless = lrclib::get_url("", track).is_none();
        if !self.online || nameless {
            // A text kept by an older player waits until LRCLIB can be
            // asked for it again: none found, the song would be heard
            // instead, and that kept as its text.
            if outdated && !nameless {
                return Answer::Failed(OUTDATED.into());
            }
            return Answer::NotFound;
        }
        fetching(true);
        let fetched = self.lrclib.fetch(track, wait);
        fetching(false);
        match fetched {
            Ok(record) => {
                let answer = record.as_ref().map_or(Found::NotFound, Record::found);
                if let Some((c, k)) = cache {
                    // The words, and when LRCLIB synced the lines.
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
    })
}

/// Reads a track's details from its path.
pub type Probe = dyn Fn(&Path) -> Option<Track> + Send + Sync;

/// How a [`LyricsWorker`] works.
#[derive(Clone)]
pub struct Options {
    /// The cache folder (`lyrics` in the data folder:
    /// [`crate::dirs::Data::lyrics`]), if any.
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
    /// The folder of the lyrics edited by hand (`lyric-edits` in the data
    /// folder: [`crate::dirs::Data::lyric_edits`]), if any.
    pub edits: Option<PathBuf>,
}

/// Told when a file's lyrics are settled, and how (for the queue's icons).
pub type Done = dyn Fn(&Path, crate::core::LyricsFacts) + Send + Sync;

/// Told when work on a file starts (`true`) and ends.
pub type Working = dyn Fn(&Path, bool) + Send + Sync;

impl Options {
    /// The player's own, without a cache (the player gives its own):
    /// the web through [`Web`], LRCLIB at [`lrclib::BASE`].
    pub fn new(online: bool) -> Options {
        Options {
            cache: None,
            online,
            http: Arc::new(Web::new()),
            base: lrclib::BASE.into(),
            probe: Arc::new(track_of),
            timer: None,
            working: None,
            done: None,
            edits: None,
        }
    }
}

/// A song's words to time: the song, its text, and whether that is its
/// edits ([`WordTimer::time_edited`]).
type Job = (PathBuf, Lyrics, bool);

struct Wanted {
    queue: VecDeque<PathBuf>,
    /// The tracks last wanted (a timing for another is dropped).
    wanted: Vec<PathBuf>,
    /// Lyrics found, to be timed, in order.
    timing: VecDeque<Job>,
    /// Tracks to get into the cache after those, and their lyrics found
    /// to be timed (neither delivered).
    ahead: VecDeque<PathBuf>,
    timing_ahead: VecDeque<Job>,
    /// Songs edited by hand, their new edits to keep (`None`: forgotten),
    /// then the song timed with them ([`LyricsWorker::edited`]).
    edited: VecDeque<(PathBuf, Option<Lyrics>)>,
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
                edited: VecDeque::new(),
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
        // One LRCLIB for both threads, so they keep to its spacing together.
        let lrclib = Lrclib::new(Arc::clone(&options.http), options.base.clone());
        if let Some(timer) = options.timer.clone() {
            timer.set_online(options.online);
            let (shared, deliver) = (Arc::clone(&shared), Arc::clone(&deliver));
            let cache = options.cache.clone().map(Cache::new);
            let (working, done) = (Arc::clone(&working), Arc::clone(&done));
            let lrclib = lrclib.clone();
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
                lrclib,
                timer: options.timer.clone(),
                edits: options.edits.map(EditStore::new),
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
                // A song edited (kept first), else a track wanted, else one
                // to get ready ahead.
                let (path, ahead, edited) = {
                    let mut wanted = lock.lock().unwrap_or_else(|e| e.into_inner());
                    loop {
                        if wanted.stop {
                            return;
                        }
                        if let Some((path, edits)) = wanted.edited.pop_front() {
                            break (path, false, Some(edits));
                        }
                        if let Some(path) = wanted.queue.pop_front() {
                            finder.online = wanted.online;
                            break (path, false, None);
                        }
                        if let Some(path) = wanted.ahead.pop_front() {
                            if wanted.wanted.contains(&path) {
                                continue;
                            }
                            finder.online = wanted.online;
                            break (path, true, None);
                        }
                        wanted = ready.wait(wanted).unwrap_or_else(|e| e.into_inner());
                    }
                };
                // New edits kept, whole or not at all, before the song is
                // looked up (and so timed) with them.
                if let (Some(edits), Some(store)) = (&edited, &finder.edits) {
                    if let Err(e) = store.save(&path, edits.as_ref()) {
                        crate::trace::note(format_args!(
                            "lyrics: the edits of {} were not kept: {e}",
                            path.display()
                        ));
                    }
                }
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
                let (answer, is_edits) = asked.look(&track, now(), &wait, &|on| working(&path, on));
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
                            let job = (path.clone(), lyrics.clone(), is_edits);
                            // Edited by hand just now: timed first.
                            if edited.is_some() {
                                timing.push_front(job);
                            } else {
                                timing.push_back(job);
                            }
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
                    // Edited just now, not timed with them (to be, or nothing
                    // times the words): the song keeps the edits as applied
                    // to its times until then.
                    if edited.is_some() && matches!(answer, Answer::Found { timed: None, .. }) {
                        continue;
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
        wanted.timing.retain(|(p, ..)| paths.contains(p));
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
        wanted.timing_ahead.retain(|(p, ..)| paths.contains(p));
        wanted.ahead = paths.into();
        ready.notify_all();
    }

    /// Keeps `edits` as the song at `path`'s lyrics edited by hand (`None`:
    /// forgets them, its text as found again), then times the song with
    /// them, before anything else; delivered once timed (at once when a
    /// timing of these edits is kept, or when nothing times the words),
    /// for a track still wanted.
    pub fn edited(&self, path: PathBuf, edits: Option<Lyrics>) {
        let (lock, ready) = &*self.shared;
        let mut wanted = lock.lock().unwrap_or_else(|e| e.into_inner());
        wanted.edited.retain(|(p, _)| *p != path);
        wanted.timing.retain(|(p, ..)| *p != path);
        wanted.edited.push_back((path, edits));
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

/// Times the words of lyrics against a track's voice.
pub trait WordTimer: Send + Sync {
    /// Whether it can run now (its model is there, or can be fetched).
    fn ready(&self) -> bool;

    /// What its timings are made by ([`jodel_lyrics::Timing::by`]): a kept
    /// timing by anything else (an older aligner) is made again.
    fn maker(&self) -> Maker;

    /// `lyrics` with each word timed against the voice of `path`; when
    /// they do not fit what is sung, the best fitting of `others` (LRCLIB's
    /// other texts of the song) instead ([`jodel_lyrics::prepare`]), timed.
    fn time(&self, path: &Path, lyrics: &Lyrics, others: &Others) -> Result<Lyrics, String>;

    /// `lyrics`, a song's text as edited by hand, with each word timed
    /// against the voice of `path` as it is: never another text in its
    /// place, nor repeats put in; its words moved by hand
    /// ([`jodel_lyrics::Word::pinned`]) held where they were set, the rest
    /// fitted around them. By default as [`WordTimer::time`] times a text
    /// with no others.
    fn time_edited(&self, path: &Path, lyrics: &Lyrics) -> Result<Lyrics, String> {
        self.time(path, lyrics, &Vec::new)
    }

    /// The words sung in the voice of `path`, heard by a speech recogniser
    /// (for a track no source has lyrics for): the text, its lines cut
    /// where the recogniser's punctuation (or a line's length) says
    /// ([`jodel_lyrics::lines::transcript_lyrics`]), and the text timed.
    /// Its source is [`Source::Transcribed`]: no trusted source, so it may
    /// be wrong.
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

/// The timing thread: takes the lyrics found without our times, in order,
/// times them (a text from LRCLIB that does not fit the song swapped for
/// one of its others that does: [`jodel_lyrics::prepare::fitting`]), keeps
/// them in the cache and sends them on (for a track still wanted).
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
        let (path, lyrics, is_edits) = {
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
            .and_then(|(c, k)| {
                if is_edits {
                    c.edited(k, &lyrics)
                } else {
                    c.timed(k).filter(|t| t.source == lyrics.source)
                }
            })
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
            // Waits `pause` unless told to stop.
            let wait = |pause: Duration| {
                let wanted = lock.lock().unwrap_or_else(|e| e.into_inner());
                let (wanted, _) = ready
                    .wait_timeout_while(wanted, pause, |w| !w.stop)
                    .unwrap_or_else(|e| e.into_inner());
                !wanted.stop
            };
            probe(&path)
                .and_then(|track| lrclib.candidates(&track, &wait).ok())
                .unwrap_or_default()
        };
        let timed = if is_edits {
            timer.time_edited(&path, &lyrics)
        } else {
            timer.time(&path, &lyrics, &others)
        };
        working(&path, false);
        let timed = match timed {
            Ok(timed) => timed,
            // The text stays as it came; nothing to say, nothing more to do.
            Err(_) => {
                done_with(&path, crate::core::LyricsFacts::Untimed);
                continue;
            }
        };
        // Another text chosen, as its fit says: it is the song's from now
        // on. Not a text that fits and was timed with the repeats heard put
        // in ([`jodel_lyrics::prepare::text`]): it stays as found, and the
        // repeats are put in again, as heard then, whenever it is timed.
        let swapped = !is_edits && timed.timing.fit.is_some_and(|f| f.swapped);
        let lyrics = if swapped {
            let mut text = timed.clone();
            text.drop_times();
            text
        } else {
            lyrics
        };
        if let (Some(cache), Some(key)) = (cache, key) {
            if is_edits {
                let _ = cache.store_edited(&key, &lyrics, &timed);
            } else {
                if swapped {
                    let _ = cache.store_text(&key, &lyrics);
                }
                let _ = cache.store_timed(&key, &timed);
            }
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
#[cfg(test)]
#[expect(
    clippy::disallowed_methods,
    reason = "tests: stand-in tracks, sidecar and kept lyrics files in a temporary folder"
)]
mod tests;
