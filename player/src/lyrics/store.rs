//! The lyrics kept on disk: the text as found, "nothing found", and our
//! word times.

use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use jodel_lyrics::sources::read_text;
use jodel_lyrics::{Found, Lyrics, Source};

use crate::cache;

/// How long "nothing found" is believed before LRCLIB is asked again.
pub const ASK_AGAIN: Duration = Duration::from_secs(30 * 24 * 60 * 60);

/// The cache's name for `path`'s lyrics ([`key_of`]; a file retagged is
/// looked up afresh).
pub fn cache_key(path: &Path) -> Option<String> {
    Some(key_of(&crate::cache::identity(path)?))
}

/// The cache's name for the lyrics of the file `identity` names (its
/// path, size and modification time: [`crate::cache::identity`]).
pub fn key_of(identity: &str) -> String {
    crate::cache::name(identity)
}

/// Seconds since 1970.
pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The first line of a "nothing found" record.
const NONE_HEADER: &str = "jodel-lyrics none";

/// A fetched text as kept ([`Cache::text`]).
#[derive(Clone, Debug, PartialEq)]
pub enum Kept {
    /// Kept in this player's version.
    Current(Lyrics),
    /// Kept by an older player, without what this one keeps (LRCLIB's
    /// synced times): to be fetched again. The file stays until then, so
    /// the song is not taken for one without lyrics while that cannot be.
    Outdated,
    /// Not kept.
    Missing,
}

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
        durable::write_atomic(file, text.as_bytes())
    }

    /// The text of the file under `key` with `extension` (where it is now:
    /// while the data folder moves, not yet moved is where it was).
    fn read(&self, key: &str, extension: &str) -> Option<String> {
        Some(read_text(
            &crate::mover::read(&self.file(key, extension)).ok()?,
        ))
    }

    /// The fetched text kept under `key`: its words, and when its source
    /// synced its lines. One kept in another version is
    /// [`Kept::Outdated`], and left where it is.
    pub fn text(&self, key: &str) -> Kept {
        let Some(text) = self.read(key, "lrc") else {
            return Kept::Missing;
        };
        let Some(body) = cache::LYRICS_TEXT.body(&text) else {
            return Kept::Outdated;
        };
        let lyrics = Lyrics::parse_text(body, Source::Unknown);
        if lyrics.is_empty() {
            Kept::Missing
        } else {
            Kept::Current(lyrics)
        }
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
        let _ = crate::mover::remove(&self.file(key, "none"));
        Ok(())
    }

    /// The "nothing found" record under `key` ([`Found::NotFound`] or
    /// [`Found::Instrumental`]), if it is younger than [`ASK_AGAIN`] at
    /// `now` (seconds since 1970).
    pub fn none(&self, key: &str, now: u64) -> Option<Found> {
        let text = self.read(key, "none")?;
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
    /// `now`; a text kept there before (an outdated one) goes.
    pub fn store_none(&self, key: &str, found: &Found, now: u64) -> io::Result<()> {
        let kind = match found {
            Found::Instrumental => "instrumental",
            Found::NotFound => "not-found",
            Found::Lyrics(_) => return Ok(()),
        };
        self.write(
            &self.file(key, "none"),
            &cache::LYRICS_NONE.wrap(&format!("{NONE_HEADER}\n{kind} {now}\n")),
        )?;
        let _ = crate::mover::remove(&self.file(key, "lrc"));
        Ok(())
    }

    /// Our word times kept under `key`, when kept in this player's version;
    /// one of another version is deleted (it is timed again).
    pub fn timed(&self, key: &str) -> Option<Lyrics> {
        let text = self.read(key, "timed.lrc")?;
        let Some(body) = cache::LYRICS_TIMED.body(&text) else {
            let _ = crate::mover::remove(&self.file(key, "timed.lrc"));
            return None;
        };
        let lyrics = Lyrics::parse(body, Source::Unknown);
        (!lyrics.is_empty()).then_some(lyrics)
    }

    /// Keeps `lyrics` (timed by us) under `key`, as Enhanced LRC: the
    /// words' times only, not when their source synced the lines (the
    /// text keeps that).
    pub fn store_timed(&self, key: &str, lyrics: &Lyrics) -> io::Result<()> {
        let mut timed = lyrics.clone();
        for line in &mut timed.lines {
            line.synced = None;
        }
        self.write(
            &self.file(key, "timed.lrc"),
            &cache::LYRICS_TIMED.wrap(&timed.to_lrc()),
        )
    }
}
