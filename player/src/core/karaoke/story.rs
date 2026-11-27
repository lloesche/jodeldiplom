//! The songs Karaoke mode wants a story of ([`stories_wanted`]): the ones
//! whose lyrics it wants, once their lyrics are settled, asked for
//! ([`Effect::WantStories`]) as that changes.

use std::path::PathBuf;
use std::sync::Arc;

use jodel_lyrics::Lyrics;

use super::{in_mode, lyrics_settled};
use crate::core::{Effect, State};

/// A song wanted for a story: its file, and its lyrics as shown.
#[derive(Clone, Debug)]
pub struct StorySong {
    /// Its file.
    pub path: PathBuf,
    /// Its lyrics as shown.
    pub lyrics: Arc<Lyrics>,
}

/// Two are equal as the same song with the very same lyrics (lyrics found
/// or timed again are other ones).
impl PartialEq for StorySong {
    fn eq(&self, other: &StorySong) -> bool {
        self.path == other.path && Arc::ptr_eq(&self.lyrics, &other.lyrics)
    }
}

/// The songs Karaoke mode wants a story of: those whose lyrics it wants
/// ([`super::lyrics_wanted`]: the one played or held, then the next
/// [`super::SONGS_AHEAD`]), in that order, each with its lyrics once
/// settled ([`lyrics_settled`]); none out of the mode.
pub fn stories_wanted(state: &State) -> Vec<StorySong> {
    if !in_mode(state) {
        return Vec::new();
    }
    (state.karaoke.lyrics_wanted.iter())
        .filter_map(|path| {
            Some(StorySong {
                path: path.clone(),
                lyrics: Arc::clone(lyrics_settled(state, path)?),
            })
        })
        .collect()
}

/// Asks for the stories wanted ([`stories_wanted`]) when they changed:
/// other songs, or other lyrics of one.
pub(crate) fn want_stories(state: &mut State, fx: &mut Vec<Effect>) {
    let wanted = stories_wanted(state);
    if wanted != state.karaoke.stories_wanted {
        state.karaoke.stories_wanted = wanted.clone();
        fx.push(Effect::WantStories(wanted));
    }
}
