//! A track's text looked up: its edits by hand, else the first source that
//! has it (the file's tags, a `.lrc` file beside it, the cache, LRCLIB),
//! with our times of it where they are kept and still stand.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use jodel_lyrics::lrclib::{self, Record};
use jodel_lyrics::sources::{SidecarSource, TagSource};
use jodel_lyrics::{Found, Lyrics, LyricsSource, Source, Track};

use super::edits::EditStore;
use super::store::{cache_key, Cache, Kept};
use super::timing::Job;
use super::web::Lrclib;
use super::{relined, Answer, Options, WordTimer, OUTDATED};

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
pub(super) fn still_current(timed: &Lyrics, timer: Option<&dyn WordTimer>) -> bool {
    match timer {
        Some(t) if t.ready() => timed.timing.by.as_ref() == Some(&t.maker()),
        _ => true,
    }
}

/// The text of `track` found here, never fetched: in its tags, in a file
/// beside it, else as kept in `cache` (the cache and the track's key).
fn local_text(track: &Track, cache: Option<(&Cache, &str)>) -> Kept {
    let local: [&dyn LyricsSource; 2] = [&TagSource, &SidecarSource];
    for source in local {
        if let Ok(Found::Lyrics(lyrics)) = source.find(track) {
            return Kept::Current(lyrics);
        }
    }
    cache.map_or(Kept::Missing, |(c, k)| c.text(k))
}

impl Finder {
    /// What the worker of `options` looks up with, asking LRCLIB through
    /// `lrclib` (shared with whatever else asks it, so they keep to its
    /// spacing together).
    pub fn of(options: &Options, lrclib: Lrclib) -> Finder {
        Finder {
            cache: options.cache.clone().map(Cache::new),
            online: options.online,
            lrclib,
            timer: options.timer.clone(),
            edits: options.edits.clone().map(EditStore::new),
        }
    }

    /// The lyrics of `track`: its edits by hand ([`Finder::edited`]), else
    /// from the first source that has them ([`Finder::found`]), saying to
    /// `fetching` when a web request starts (`true`) and ends; `now` is the
    /// time for the cache's records, `wait` waits out a busy service (saying
    /// `false` to give up). With what its words are timed as: edits as they
    /// are, a text found as found ([`Job`]).
    pub fn lookup(
        &self,
        track: &Track,
        now: u64,
        wait: &dyn Fn(Duration) -> bool,
        fetching: &dyn Fn(bool),
    ) -> (Answer, fn(Lyrics) -> Job) {
        match self.edits.as_ref().and_then(|e| e.load(&track.path)) {
            Some(edits) => (self.edited(track, &edits), Job::Edited),
            None => (self.found(track, now, wait, fetching), Job::Found),
        }
    }

    /// The lyrics of a song edited by hand as `edits`: their text, with the
    /// line times of the text found here ([`local_text`]) as its scaffold,
    /// and our times of it when kept for these edits by the timer in use.
    fn edited(&self, track: &Track, edits: &Lyrics) -> Answer {
        let key = cache_key(&track.path);
        let cache = self.cache.as_ref().zip(key.as_deref());
        let mut text = edits.clone();
        if let Kept::Current(found) = local_text(track, cache) {
            jodel_lyrics::scaffold::attach(&mut text, &found);
        }
        let timed = cache
            .and_then(|(c, k)| c.edited(k, edits))
            .filter(|t| still_current(t, self.timer.as_deref()));
        Answer::Found {
            lyrics: text,
            timed,
        }
    }

    /// The lyrics of a song not edited, from the first source that has
    /// them ([`local_text`], else LRCLIB), saying to `fetching` when a web
    /// request starts (`true`) and ends.
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
        let outdated = match local_text(track, cache) {
            Kept::Current(lyrics) => return found(lyrics),
            Kept::Outdated => true,
            Kept::Missing => false,
        };
        if let Some((c, k)) = cache {
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
