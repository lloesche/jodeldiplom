//! The lyrics of the songs Karaoke is about: which are wanted, what is
//! known of each ([`SongLyrics`]), the speech model that times their words
//! ([`ModelState`]), and whether a song can be sung yet ([`not_singable`]).

use std::path::PathBuf;
use std::sync::Arc;

use jodel_lyrics::{Detail, Lyrics, Source};

use super::{in_mode, song_index};
use crate::core::{Effect, Side, State};
use crate::work::lyrics::Answer;

/// The warning shown with lyrics a speech recogniser heard ([`SongLyrics::heard`]).
pub const HEARD_WARNING: &str =
    "No trusted source has these lyrics: machine learning heard them from the voice, so they may be wrong.";

/// What is known of a song's lyrics.
#[derive(Clone, Debug, PartialEq)]
pub enum SongLyrics {
    /// Asked for.
    Pending,
    /// Found: the text as found (its words only), and as shown (with our
    /// word times once the speech model has them).
    Found {
        /// As found.
        lyrics: Arc<Lyrics>,
        /// As shown.
        shown: Arc<Lyrics>,
    },
    /// The song has no words.
    Instrumental,
    /// None were found.
    NotFound,
    /// The look-up failed (it is tried again later).
    Failed(String),
}

impl SongLyrics {
    /// From a look-up's answer.
    pub fn from_answer(answer: Answer) -> SongLyrics {
        match answer {
            Answer::Found { lyrics, timed } => {
                let shown = timed.unwrap_or_else(|| lyrics.clone());
                SongLyrics::Found {
                    lyrics: Arc::new(lyrics),
                    shown: Arc::new(shown),
                }
            }
            Answer::Instrumental => SongLyrics::Instrumental,
            Answer::NotFound => SongLyrics::NotFound,
            Answer::Failed(why) => SongLyrics::Failed(why),
        }
    }

    /// Whether its lyrics were heard from the voice by a speech recogniser,
    /// no trusted source having them ([`Source::Transcribed`]).
    pub fn heard(&self) -> bool {
        matches!(self, SongLyrics::Found { lyrics, .. } if matches!(lyrics.source, Source::Transcribed(_)))
    }

    /// In a few words, for the tab.
    pub fn describe(&self) -> String {
        match self {
            SongLyrics::Pending => "looking…".into(),
            SongLyrics::Found { lyrics, shown } => {
                let from = match lyrics.source {
                    Source::Tag => "from its tags",
                    Source::Sidecar => "from its .lrc file",
                    Source::Lrclib => "from LRCLIB",
                    Source::Transcribed(_) => {
                        "heard from the voice by machine learning (no trusted source)"
                    }
                    Source::Unknown => "found",
                };
                let timed = match (&shown.timing.by, shown.timing.detail) {
                    (Some(by), Detail::Words) => format!("words timed ({})", by.name),
                    _ => "not timed yet".into(),
                };
                // How well the text fit what the recogniser heard.
                let fit = match shown.timing.fit {
                    Some(f) if f.swapped => format!(
                        "; its first text did not fit the song, this one does ({} % heard)",
                        f.percent
                    ),
                    Some(f) if f.fits() => format!("; fits the song ({} % heard)", f.percent),
                    Some(f) => format!("; may not fit the song (only {} % heard)", f.percent),
                    None => String::new(),
                };
                format!("{from}, {timed}{fit}")
            }
            SongLyrics::Instrumental => "instrumental".into(),
            SongLyrics::NotFound => "none found".into(),
            SongLyrics::Failed(why) => format!("not looked up: {why}"),
        }
    }
}

/// The songs whose lyrics are wanted: while the tab or the mode shows, the one
/// playing (or held), then the next two in the queue (the lyrics editor's song
/// before them while it shows); while the Transition tab
/// shows, its tracks' ([`crate::core::transition::tab::lyrics_wanted`]).
pub fn lyrics_wanted(state: &State) -> Vec<PathBuf> {
    if let Some(paths) = crate::core::transition::tab::lyrics_wanted(state) {
        return paths;
    }
    // The lyrics editor's song first, its timing taken as it comes again.
    let editing = state
        .karaoke
        .editor
        .as_ref()
        .filter(|_| state.screen == crate::core::Screen::Lyrics)
        .map(|e| e.path.clone());
    if !(state.keys_on(Side::Karaoke) || in_mode(state) || editing.is_some()) {
        return Vec::new();
    }
    let mut out: Vec<PathBuf> = editing.into_iter().collect();
    if let Some(first) = song_index(state) {
        for path in state.queue.paths.iter().skip(first).take(3) {
            if !out.contains(path) {
                out.push(path.clone());
            }
        }
    }
    out
}

/// Asks for the lyrics wanted that are not known yet, when what is wanted
/// changed; forgets those no longer wanted.
pub(crate) fn want_lyrics(state: &mut State, fx: &mut Vec<Effect>) {
    let wanted = lyrics_wanted(state);
    if wanted == state.karaoke.lyrics_wanted {
        return;
    }
    let k = &mut state.karaoke;
    k.lyrics
        .retain(|p, l| wanted.contains(p) && !matches!(l, SongLyrics::Failed(_)));
    let ask: Vec<PathBuf> = wanted
        .iter()
        .filter(|p| !k.lyrics.contains_key(*p))
        .cloned()
        .collect();
    for path in &ask {
        k.lyrics.insert(path.clone(), SongLyrics::Pending);
    }
    k.lyrics_wanted = wanted;
    if !ask.is_empty() {
        fx.push(Effect::WantLyrics(ask));
    }
}

/// A song's lyrics arrived.
pub(crate) fn lyrics_arrived(state: &mut State, path: PathBuf, answer: Answer) {
    // Words timed by us: the speech model is here (fetched meanwhile).
    if let Answer::Found {
        timed: Some(timed), ..
    } = &answer
    {
        if timed.timing.detail == Detail::Words && timed.timing.by.is_some() {
            state.karaoke.model = ModelState::Ready;
        }
    }
    if state.karaoke.lyrics_wanted.contains(&path) {
        state
            .karaoke
            .lyrics
            .insert(path.clone(), SongLyrics::from_answer(answer));
        crate::core::lyric_edit::arrived(state, &path);
    }
}

/// The lyrics of the song at `index`, when known.
pub fn lyrics_of(state: &State, index: usize) -> Option<&SongLyrics> {
    state.karaoke.lyrics.get(state.queue.paths.get(index)?)
}

/// The speech model that times the words. Setup gets it before the player
/// starts ([`crate::setup`]); a player started without (tests) has none.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum ModelState {
    /// Not found, and nowhere to fetch it to: the words are shown, not
    /// timed.
    #[default]
    Missing,
    /// Not here yet: fetched when first needed (unless the lyrics are kept
    /// offline).
    Coming,
    /// There.
    Ready,
}

/// Why track `index` cannot be sung yet, if it cannot: its lyrics still
/// looked for, or their words still being timed (a big folder queued is
/// got ready a track at a time), or known to be missing (none found, an
/// instrumental, the look-up failed). Words that cannot be timed (no speech
/// model, or the timing failed) are sung as they are.
pub fn not_singable(state: &State, index: usize) -> Option<&'static str> {
    let path = state.queue.paths.get(index)?;
    let settled = state
        .ready
        .get(path)
        .is_some_and(|jobs| jobs.contains(&crate::core::Job::Lyrics));
    match lyrics_of(state, index) {
        None | Some(SongLyrics::Pending) => Some("still looking for its lyrics"),
        Some(SongLyrics::NotFound) => Some("no lyrics were found for it"),
        Some(SongLyrics::Instrumental) => Some("it is an instrumental"),
        Some(SongLyrics::Failed(_)) => Some("its lyrics could not be looked up"),
        Some(SongLyrics::Found { shown, .. }) => {
            let timed = shown.timing.detail == jodel_lyrics::Detail::Words;
            (!timed && !settled && state.karaoke.model != ModelState::Missing)
                .then_some("its words are still being timed")
        }
    }
}
