//! The timing thread: the words of the texts found without our times
//! timed, in order ([`time_words`]), and the songs no source has words for
//! heard from their voice instead ([`hear`]); each kept in the cache and
//! sent on for a track still wanted ([`Sinks::settle`]).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use jodel_lyrics::prepare::{Others, Text};
use jodel_lyrics::{Found, Lyrics, Source};

use super::finder::still_current;
use super::store::{cache_key, now, Cache};
use super::web::Lrclib;
use super::worker::{Hearing, Wanted};
use super::{Answer, Done, Probe, Told, WordTimer, Working, TOO_LITTLE_HEARD};
use crate::core::prep::{Origin, Preview, Words};
use crate::core::LyricsFacts;
use crate::work::Queue;

/// A song's words to time: a text found, or one edited by hand.
#[derive(Clone, Debug, PartialEq)]
pub enum Job {
    /// Found by a source: when it does not fit what is sung, one of
    /// LRCLIB's others may be timed in its place ([`Text::Found`]).
    Found(Lyrics),
    /// The song's edits by hand: timed as they are ([`Text::Edited`]).
    Edited(Lyrics),
}

impl Job {
    /// The text.
    fn text(&self) -> &Lyrics {
        match self {
            Job::Found(text) | Job::Edited(text) => text,
        }
    }

    /// Its timing as kept in `cache` under `key`, where it is.
    fn cached(&self, cache: &Cache, key: &str) -> Option<Lyrics> {
        match self {
            Job::Found(text) => cache.timed(key).filter(|t| t.source == text.source),
            Job::Edited(edits) => cache.edited(key, edits),
        }
    }

    /// Its words timed by `timer` against the voice of `path`; `others`
    /// the song's other texts, for a text found.
    fn time(&self, timer: &dyn WordTimer, path: &Path, others: &Others) -> Result<Lyrics, String> {
        match self {
            Job::Found(text) => timer.time(path, text, &Text::Found(others)),
            Job::Edited(edits) => timer.time(path, edits, &Text::Edited),
        }
    }

    /// The text the song has from now on, timed as `timed`, kept with that
    /// timing in `cache` (the cache and the song's key). Another text
    /// chosen as its fit says ([`jodel_lyrics::Fit::swapped`]) is the
    /// song's from now on; not one that fits and was timed with the
    /// repeats heard put in ([`jodel_lyrics::prepare::text`]): it stays as
    /// found, and the repeats are put in again, as heard then, whenever it
    /// is timed.
    fn store(self, cache: Option<(&Cache, &str)>, timed: &Lyrics) -> Lyrics {
        match self {
            Job::Edited(edits) => {
                if let Some((c, k)) = cache {
                    let _ = c.store_edited(k, &edits, timed);
                }
                edits
            }
            Job::Found(text) => {
                let swapped = timed.timing.fit.is_some_and(|f| f.swapped);
                let text = if swapped {
                    let mut text = timed.clone();
                    text.drop_times();
                    text
                } else {
                    text
                };
                if let Some((c, k)) = cache {
                    if swapped {
                        let _ = c.store_text(k, &text);
                    }
                    let _ = c.store_timed(k, timed);
                }
                text
            }
        }
    }
}

/// Where the worker's threads send what comes of their work.
pub(super) struct Sinks {
    /// Each answer, with its track's path.
    pub(super) deliver: Mutex<Box<Deliver>>,
    /// Told when work on a file starts and ends.
    pub(super) working: Arc<Working>,
    /// Told of each step of a file's work.
    pub(super) told: Arc<Told>,
    /// Told when a file's lyrics are settled.
    pub(super) done: Arc<Done>,
}

/// Takes each answer, with its track's path.
pub(super) type Deliver = dyn FnMut(PathBuf, Answer) + Send;

impl Sinks {
    /// `answer` for `path` sent on.
    pub(super) fn deliver(&self, path: PathBuf, answer: Answer) {
        (self.deliver.lock().unwrap_or_else(|e| e.into_inner()))(path, answer);
    }

    /// `path` settled as `facts` (told so), and `answer` delivered when the
    /// track is still wanted.
    fn settle(&self, queue: &Queue<Wanted>, path: PathBuf, answer: Answer, facts: LyricsFacts) {
        (self.done)(&path, facts);
        let still = !queue.stopped() && queue.lock().wanted.contains(&path);
        if still {
            self.deliver(path, answer);
        }
    }
}

/// The timing thread: takes the lyrics found without our times, in order,
/// times them (a text from LRCLIB that does not fit the song swapped for
/// one of its others that does: [`jodel_lyrics::prepare::fitting`]), keeps
/// them in the cache and sends them on (for a track still wanted).
pub(super) fn time_words(
    queue: &Queue<Wanted>,
    timer: &dyn WordTimer,
    cache: Option<&Cache>,
    lrclib: &Lrclib,
    probe: &Probe,
    sinks: &Sinks,
) {
    let (working, told) = (&*sinks.working, &*sinks.told);
    loop {
        // A timing first, else a track to hear.
        let next = queue.next(|wanted| {
            (wanted.timing.pop().map(|(job, _)| Ok(job)))
                .or_else(|| wanted.hearing.pop().map(|(path, _)| Err(path)))
        });
        let (path, job) = match next {
            Some(Ok(job)) => job,
            Some(Err(path)) => {
                hear(queue, timer, cache, sinks, path);
                continue;
            }
            None => return,
        };
        // Timed meanwhile (wanted and ahead both): not again.
        let key = cache_key(&path);
        let cache = cache.zip(key.as_deref());
        let kept = cache
            .and_then(|(c, k)| job.cached(c, k))
            .filter(|t| still_current(t, Some(timer)));
        if let Some(timed) = kept {
            let answer = Answer::Found {
                lyrics: job.text().clone(),
                timed: Some(timed),
            };
            sinks.settle(queue, path, answer, LyricsFacts::Timed);
            continue;
        }
        working(&path, true);
        told(&path, Words::Timing);
        // LRCLIB's other texts of the song, asked for only when this one
        // does not fit (and only of LRCLIB's: the file's own are kept).
        let others = || -> Vec<Lyrics> {
            let online = !queue.stopped() && queue.lock().online;
            if job.text().source != Source::Lrclib || !online {
                return Vec::new();
            }
            // Waits `pause` unless told to stop.
            let wait = |pause: Duration| queue.pause(pause);
            probe(&path)
                .and_then(|track| lrclib.candidates(&track, &wait).ok())
                .unwrap_or_default()
        };
        let timed = job.time(timer, &path, &others);
        working(&path, false);
        let timed = match timed {
            Ok(timed) => timed,
            // The text stays as it came; nothing to say, nothing more to do.
            Err(_) => {
                (sinks.done)(&path, LyricsFacts::Untimed);
                continue;
            }
        };
        let lyrics = job.store(cache, &timed);
        if let Some(preview) = Preview::of(&timed) {
            told(&path, Words::Timed(Arc::new(preview)));
        }
        let answer = Answer::Found {
            lyrics,
            timed: Some(timed),
        };
        sinks.settle(queue, path, answer, LyricsFacts::Timed);
    }
}

/// Hears the words sung in `path` ([`WordTimer::transcribe`]), for a track
/// no source has lyrics for: kept in the cache as the track's text (marked
/// as heard, not from a trusted source) and timing, and delivered when the
/// track is still wanted. Too little heard: kept as having no words. Else
/// (no recogniser here, or it failed): none found. Either way the track
/// is heard once a run ([`Hearing::Heard`]), however often it is asked for.
fn hear(
    queue: &Queue<Wanted>,
    timer: &dyn WordTimer,
    cache: Option<&Cache>,
    sinks: &Sinks,
    path: PathBuf,
) {
    let (working, told) = (&*sinks.working, &*sinks.told);
    working(&path, true);
    told(&path, Words::Hearing);
    let heard = timer.transcribe(&path);
    working(&path, false);
    let key = cache_key(&path);
    let cache = cache.zip(key.as_deref());
    let answer = match heard {
        Ok((text, timed)) => {
            if let Some((c, k)) = cache {
                let _ = c.store_text(k, &text);
                let _ = c.store_timed(k, &timed);
            }
            told(&path, Words::Found(Origin::Heard));
            if let Some(preview) = Preview::of(&timed) {
                told(&path, Words::Timed(Arc::new(preview)));
            }
            Answer::Found {
                lyrics: text,
                timed: Some(timed),
            }
        }
        Err(e) if e == TOO_LITTLE_HEARD => {
            if let Some((c, k)) = cache {
                let _ = c.store_none(k, &Found::Instrumental, now());
            }
            Answer::Instrumental
        }
        Err(_) => Answer::NotFound,
    };
    queue.lock().heard.insert(path.clone(), Hearing::Heard);
    let facts = LyricsFacts::of(&answer);
    sinks.settle(queue, path, answer, facts);
}
