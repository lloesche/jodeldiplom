//! The lyrics' worker: what it is asked for ([`Wanted`]), the look-up
//! thread, and where each track's answer goes from there ([`route`]): to
//! be timed, to be heard, or settled.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use jodel_lyrics::Lyrics;

use super::finder::Finder;
use super::store::{now, Cache};
use super::timing::{time_words, Job, Sinks};
use super::web::Lrclib;
use super::{Answer, Options, WordTimer, FAILED_PAUSE};
use crate::core::prep::{Origin, Words};
use crate::core::SettledLyrics;
use crate::work::{Lane as Ask, Lanes, Queue};

/// Where a track no source has lyrics for stands in this run's hearing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Hearing {
    /// To be heard, or being heard.
    Coming,
    /// Heard (a hearing that failed kept nothing, and the track stays
    /// without lyrics until the next run).
    Heard,
}

/// What the worker is asked for, and has to do.
pub(super) struct Wanted {
    /// Tracks to look up: wanted now, and to get into the cache after those
    /// (not delivered).
    looking: Lanes<PathBuf>,
    /// The tracks last wanted (a timing for another is dropped).
    pub(super) wanted: Vec<PathBuf>,
    /// Lyrics found, to be timed, in order: for the tracks wanted, and for
    /// those ahead.
    pub(super) timing: Lanes<(PathBuf, Job)>,
    /// Songs edited by hand, their new edits to keep (`None`: forgotten),
    /// then the song timed with them ([`LyricsWorker::edited`]).
    edited: VecDeque<(PathBuf, Option<Lyrics>)>,
    /// Tracks no source has lyrics for, to be heard by the speech
    /// recogniser ([`WordTimer::transcribe`]), in order: wanted, and ahead.
    pub(super) hearing: Lanes<PathBuf>,
    /// The tracks heard this run, each once.
    pub(super) heard: HashMap<PathBuf, Hearing>,
    /// Whether LRCLIB is asked.
    pub(super) online: bool,
}

/// What comes of a track's answer once it is looked up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Route {
    /// Its words to be timed: first when it was edited by hand just now
    /// (and held until then), else last (delivered as found meanwhile).
    Time {
        /// Edited just now.
        edited: bool,
    },
    /// None found anywhere: heard from the voice instead (the answer comes
    /// from there), once a run.
    Hear,
    /// Settled (told so): delivered, but for edits made just now that
    /// nothing times, held (the song keeps the edits as applied to its
    /// times until then).
    Settle {
        /// Held.
        hold: bool,
    },
}

/// Where `answer` goes: timed when it was found without our times and the
/// timer can run (`ready`); heard when nothing was found, the timer can run
/// and the track was not heard this run (its [`Hearing`]); else settled.
/// `edited`: the song was edited by hand just now.
pub(super) fn route(answer: &Answer, ready: bool, edited: bool, heard: Option<Hearing>) -> Route {
    match answer {
        Answer::Found {
            lyrics,
            timed: None,
        } if ready && !lyrics.is_empty() => Route::Time { edited },
        Answer::NotFound if ready && heard != Some(Hearing::Heard) => Route::Hear,
        Answer::Found { timed: None, .. } => Route::Settle { hold: edited },
        _ => Route::Settle { hold: false },
    }
}

impl Route {
    /// Whether the answer is delivered now.
    fn delivers(self) -> bool {
        matches!(
            self,
            Route::Time { edited: false } | Route::Settle { hold: false }
        )
    }
}

/// Looks up lyrics in the background, one track at a time.
pub struct LyricsWorker {
    queue: Arc<Queue<Wanted>>,
    timer: Option<Arc<dyn WordTimer>>,
}

impl LyricsWorker {
    /// Starts the thread; each answer goes to `deliver` with its track's
    /// path.
    pub fn new(
        options: Options,
        deliver: impl FnMut(PathBuf, Answer) + Send + 'static,
    ) -> LyricsWorker {
        let queue = Arc::new(Queue::new(Wanted {
            looking: Lanes::default(),
            wanted: Vec::new(),
            timing: Lanes::default(),
            edited: VecDeque::new(),
            hearing: Lanes::default(),
            heard: HashMap::new(),
            online: options.online,
        }));
        let sinks = Arc::new(Sinks {
            deliver: Mutex::new(Box::new(deliver)),
            working: (options.working.clone()).unwrap_or_else(|| Arc::new(|_, _| {})),
            told: (options.told.clone()).unwrap_or_else(|| Arc::new(|_, _| {})),
            done: (options.done.clone()).unwrap_or_else(|| Arc::new(|_, _| {})),
        });
        // One LRCLIB for both threads, so they keep to its spacing together.
        let lrclib = Lrclib::new(Arc::clone(&options.http), options.base.clone());
        if let Some(timer) = options.timer.clone() {
            timer.set_online(options.online);
            let (queue, sinks) = (Arc::clone(&queue), Arc::clone(&sinks));
            let cache = options.cache.clone().map(Cache::new);
            let lrclib = lrclib.clone();
            let probe = Arc::clone(&options.probe);
            thread::spawn(move || {
                // Its GPU work gives way to the window's frames.
                jodel_gpu::pace::background();
                time_words(&queue, &*timer, cache.as_ref(), &lrclib, &*probe, &sinks);
            });
        }
        let kept_timer = options.timer.clone();
        let worker = Arc::clone(&queue);
        thread::spawn(move || {
            let queue = &*worker;
            let mut finder = Finder::of(&options, lrclib);
            let ready = || options.timer.as_ref().is_some_and(|t| t.ready());
            // Tracks the web failed for, and when.
            let mut failed: HashMap<PathBuf, Instant> = HashMap::new();
            // Waits `pause` unless told to stop.
            let wait = |pause: Duration| queue.pause(pause);
            loop {
                // A song edited (kept first), else a track wanted, else one
                // to get ready ahead.
                let next = queue.next(|wanted| {
                    if let Some((path, edits)) = wanted.edited.pop_front() {
                        return Some((path, Ask::Now, Some(edits)));
                    }
                    if let Some(path) = wanted.looking.lane(Ask::Now).pop_front() {
                        finder.online = wanted.online;
                        return Some((path, Ask::Now, None));
                    }
                    while let Some(path) = wanted.looking.lane(Ask::Ahead).pop_front() {
                        if !wanted.wanted.contains(&path) {
                            finder.online = wanted.online;
                            return Some((path, Ask::Ahead, None));
                        }
                    }
                    None
                });
                let Some((path, lane, edited)) = next else {
                    return;
                };
                let ahead = lane == Ask::Ahead;
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
                        sinks.deliver(path, Answer::Failed("the file cannot be read".into()));
                    }
                    continue;
                };
                let resting = failed
                    .get(&path)
                    .is_some_and(|at| at.elapsed() < FAILED_PAUSE);
                let mut asked = finder.clone();
                asked.online &= !resting;
                // A look-up on LRCLIB told as it begins; whether one was.
                let fetched = std::cell::Cell::new(false);
                let fetching = |on: bool| {
                    if on {
                        fetched.set(true);
                        (sinks.told)(&path, Words::Looking);
                    }
                    (sinks.working)(&path, on);
                };
                // The song's edits are timed as they are; a text found may
                // give way to a better fitting one.
                let (answer, job) = asked.lookup(&track, now(), &wait, &fetching);
                if matches!(answer, Answer::Failed(_)) {
                    failed.insert(path.clone(), Instant::now());
                } else if !resting {
                    failed.remove(&path);
                }
                if queue.stopped() {
                    return;
                }
                let route = queue.change(|wanted| {
                    let heard = wanted.heard.get(&path).copied();
                    let route = route(&answer, ready(), edited.is_some(), heard);
                    match route {
                        Route::Time { edited } => {
                            if let Answer::Found { lyrics, .. } = &answer {
                                let timing = wanted.timing.lane(lane);
                                let to_time = (path.clone(), job(lyrics.clone()));
                                if edited {
                                    timing.push_front(to_time);
                                } else {
                                    timing.push_back(to_time);
                                }
                            }
                        }
                        Route::Hear => {
                            let coming = wanted.heard.insert(path.clone(), Hearing::Coming);
                            if coming.is_none() {
                                wanted.hearing.lane(lane).push_back(path.clone());
                            } else if !ahead {
                                // Wanted now: heard before the songs ahead.
                                wanted.hearing.hurry(&path);
                            }
                        }
                        Route::Settle { .. } => (sinks.done)(&path, SettledLyrics::of(&answer)),
                    }
                    route
                });
                // Found by work now (fetched, or to be timed): told where.
                if let Answer::Found { lyrics, .. } = &answer {
                    if fetched.get() || matches!(route, Route::Time { .. }) {
                        (sinks.told)(&path, Words::Found(Origin::of(&lyrics.source)));
                    }
                }
                if route.delivers() && !ahead {
                    sinks.deliver(path, answer);
                }
            }
        });
        LyricsWorker {
            queue,
            timer: kept_timer,
        }
    }

    /// Looks up exactly `paths`, in this order (what was asked for before
    /// and not started yet is dropped): the playing track, then the next
    /// ones in the queue.
    pub fn want(&self, paths: Vec<PathBuf>) {
        self.queue.change(|wanted| {
            (wanted.timing.lane(Ask::Now)).retain(|(p, _)| paths.contains(p));
            wanted.wanted = paths.clone();
            wanted.looking.want(paths);
        });
    }

    /// Looks up `paths` ahead, in this order, after what is wanted, and
    /// times the words of what is found, into the cache only (nothing is
    /// delivered; what was asked for before and not started is dropped).
    pub fn prepare(&self, paths: Vec<PathBuf>) {
        self.queue.change(|wanted| {
            (wanted.timing.lane(Ask::Ahead)).retain(|(p, _)| paths.contains(p));
            wanted.looking.prepare(paths);
        });
    }

    /// Keeps `edits` as the song at `path`'s lyrics edited by hand (`None`:
    /// forgets them, its text as found again), then times the song with
    /// them, before anything else; delivered once timed (at once when a
    /// timing of these edits is kept, or when nothing times the words),
    /// for a track still wanted.
    pub fn edited(&self, path: PathBuf, edits: Option<Lyrics>) {
        self.queue.change(|wanted| {
            wanted.edited.retain(|(p, _)| *p != path);
            (wanted.timing.lane(Ask::Now)).retain(|(p, _)| *p != path);
            wanted.edited.push_back((path, edits));
        });
    }

    /// Turns the LRCLIB look-ups on or off, from the next track.
    pub fn set_online(&self, online: bool) {
        self.queue.lock().online = online;
        if let Some(timer) = &self.timer {
            timer.set_online(online);
        }
    }
}

impl Drop for LyricsWorker {
    fn drop(&mut self) {
        self.queue.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jodel_lyrics::Source;

    #[test]
    fn an_answer_is_timed_heard_or_settled() {
        let text = Lyrics::parse_text("Some words", Source::Lrclib);
        let found = |timed: Option<Lyrics>| Answer::Found {
            lyrics: text.clone(),
            timed,
        };
        let untimed = found(None);
        // Found without our times: timed when the timer can, first when
        // edited just now (and held), else delivered meanwhile.
        assert_eq!(
            route(&untimed, true, false, None),
            Route::Time { edited: false }
        );
        assert_eq!(
            route(&untimed, true, true, None),
            Route::Time { edited: true }
        );
        assert!(Route::Time { edited: false }.delivers());
        assert!(!Route::Time { edited: true }.delivers());
        // Nothing times them: settled, edits just made held.
        assert_eq!(
            route(&untimed, false, false, None),
            Route::Settle { hold: false }
        );
        assert_eq!(
            route(&untimed, false, true, None),
            Route::Settle { hold: true }
        );
        assert!(!Route::Settle { hold: true }.delivers());
        // Nothing to time: settled.
        let empty = Answer::Found {
            lyrics: Lyrics::parse_text("", Source::Lrclib),
            timed: None,
        };
        assert_eq!(
            route(&empty, true, false, None),
            Route::Settle { hold: false }
        );
        let timed = found(Some(text.clone()));
        assert_eq!(
            route(&timed, true, true, None),
            Route::Settle { hold: false }
        );
        // None found: heard, once a run.
        assert_eq!(route(&Answer::NotFound, true, false, None), Route::Hear);
        let coming = Some(Hearing::Coming);
        assert_eq!(route(&Answer::NotFound, true, false, coming), Route::Hear);
        assert_eq!(
            route(&Answer::NotFound, true, false, Some(Hearing::Heard)),
            Route::Settle { hold: false }
        );
        assert_eq!(
            route(&Answer::NotFound, false, false, None),
            Route::Settle { hold: false }
        );
        assert!(!Route::Hear.delivers());
        assert_eq!(
            route(&Answer::Instrumental, true, false, None),
            Route::Settle { hold: false }
        );
    }
}
