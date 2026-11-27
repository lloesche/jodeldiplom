//! Lyrics of the tracks for karaoke: found in the background, and kept on
//! disk.
//!
//! A [`LyricsWorker`] thread looks up the tracks it is asked for, one at a
//! time, in this order:
//!
//! 1. the file's lyrics tag ([`jodel_lyrics::sources::TagSource`]);
//! 2. a `.lrc` file beside it ([`jodel_lyrics::sources::SidecarSource`]);
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
//! The cache is `lyrics/` in the data folder
//! ([`crate::store::dirs::Data::lyrics`]), a few files a track named
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
//! ([`Text::Edited`]: the words moved by hand held where they were set, the
//! rest fitted around them), its words marked as edited.
//! [`LyricsWorker::edited`] keeps new edits and times the song with them,
//! delivering the song only once timed.
//!
//! The modules: `finder` (a track's text looked up: [`Finder`]), `worker`
//! (the look-up thread, and where each answer goes from there),
//! `timing` (the timing and hearing thread), [`web`] (the requests,
//! LRCLIB), [`store`] (the cache), [`edits`] (the edits kept), [`models`]
//! (the models' files), [`timer`] (the speech model's timing and the voice
//! it times against).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use jodel_lyrics::prepare::Text;
use jodel_lyrics::{lrclib, Lyrics, Maker, Track};

pub mod edits;
mod finder;
pub mod models;
pub mod store;
pub mod timer;
mod timing;
pub mod web;
mod worker;

pub use finder::{track_of, Finder};
pub use worker::LyricsWorker;

use web::{Http, Web};

use crate::core::prep::Words;

/// How long a track whose look-up failed (the network, the service) is
/// not looked up on the web again.
const FAILED_PAUSE: Duration = Duration::from_secs(10 * 60);

/// Why a track's lyrics could not be looked up: the text kept for it is
/// of an older version ([`store::Kept::Outdated`]), to be fetched again,
/// and LRCLIB cannot be asked now.
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

/// Reads a track's details from its path.
pub type Probe = dyn Fn(&Path) -> Option<Track> + Send + Sync;

/// How a [`LyricsWorker`] works.
#[derive(Clone)]
pub struct Options {
    /// The cache folder (`lyrics` in the data folder:
    /// [`crate::store::dirs::Data::lyrics`]), if any.
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
    /// folder: [`crate::store::dirs::Data::lyric_edits`]), if any.
    pub edits: Option<PathBuf>,
    /// Told of each step of a file's work as it begins and what it finds
    /// ([`crate::core::prep::Words`]): a look-up on LRCLIB, the words found
    /// (fetched now, or to be timed), heard, timed.
    pub told: Option<Arc<Told>>,
}

/// Told when a file's lyrics are settled, and how (for the queue's icons).
pub type Done = dyn Fn(&Path, crate::core::LyricsFacts) + Send + Sync;

/// Told when work on a file starts (`true`) and ends.
pub type Working = dyn Fn(&Path, bool) + Send + Sync;

/// Told of a step of a file's work and what it finds.
pub type Told = dyn Fn(&Path, Words) + Send + Sync;

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
            told: None,
        }
    }
}

/// Times the words of lyrics against a track's voice.
pub trait WordTimer: Send + Sync {
    /// Whether it can run now (its model is there, or can be fetched).
    fn ready(&self) -> bool;

    /// What its timings are made by ([`jodel_lyrics::Timing::by`]): a kept
    /// timing by anything else (an older aligner) is made again.
    fn maker(&self) -> Maker;

    /// `lyrics` with each word timed against the voice of `path`, the text
    /// made as `text` lets it be ([`jodel_lyrics::prepare::text`]): a text
    /// found that does not fit what is sung gives way to the best fitting
    /// of LRCLIB's other texts of the song; a text edited by hand is timed
    /// as it is, its words moved by hand ([`jodel_lyrics::WordEdit::pin`])
    /// held where they were set, the rest fitted around them.
    fn time(&self, path: &Path, lyrics: &Lyrics, text: &Text) -> Result<Lyrics, String>;

    /// The words sung in the voice of `path`, heard by a speech recogniser
    /// (for a track no source has lyrics for): the text, its lines cut
    /// where the recogniser's punctuation (or a line's length) says
    /// ([`jodel_lyrics::lines::transcript_lyrics`]), and the text timed.
    /// Its source is [`jodel_lyrics::Source::Transcribed`]: no trusted source, so it may
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

/// Why a transcription gave nothing: the voice sings too few words.
pub const TOO_LITTLE_HEARD: &str = "too little was heard";
#[cfg(test)]
#[expect(
    clippy::disallowed_methods,
    reason = "tests: stand-in tracks, sidecar and kept lyrics files in a temporary folder"
)]
mod tests;
