//! Whether a song can be sung now ([`not_singable`]): karaoke needs a
//! graphics card, the song fully prepared (every job the queue's icons
//! show for it done: [`crate::core::still_to_do`]) and lyrics to sing. A
//! Next pressed in the mode on a song still being prepared waits for it
//! ([`start_when_ready`]).

use super::{in_mode, lyrics_of, song_index, SongLyrics};
use crate::core::{job_words, still_to_do, Effect, Gpu, Job, State};

/// Why a song cannot be sung now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NotSingable {
    /// No karaoke at all: no graphics card (or still looking for one).
    Gpu(String),
    /// Still being prepared: what is still to come for it, job by job in
    /// the words of the queue's icons ([`job_words`]). It can be sung as
    /// soon as that is done.
    Preparing(String),
    /// Nothing to sing: no lyrics found, an instrumental, the look-up
    /// failed.
    Nothing(&'static str),
}

impl NotSingable {
    /// Why, in a few words, for the tab and the status line.
    pub fn why(&self) -> String {
        match self {
            NotSingable::Gpu(why) => why.clone(),
            NotSingable::Preparing(what) => format!("still being prepared: {what}"),
            NotSingable::Nothing(why) => (*why).to_owned(),
        }
    }
}

/// Why there is no karaoke here, for any song, if there is none: no
/// graphics card (said with why none was found), or one still looked for.
pub fn no_karaoke(state: &State) -> Option<NotSingable> {
    match &state.gpu {
        Gpu::Present(_) => None,
        Gpu::Looking => Some(NotSingable::Gpu(
            "still looking for a graphics card (karaoke needs one)".into(),
        )),
        Gpu::Absent(why) => Some(NotSingable::Gpu(format!(
            "karaoke needs a graphics card: {why}"
        ))),
    }
}

/// Why track `index` cannot be sung now, if it cannot: without a graphics
/// card no song can (while one is still looked for, none yet); a song
/// with some of its preparation still to come waits for all of it (the
/// jobs the queue's icons show, in their words); once prepared, a song
/// with nothing to sing (no lyrics found, an instrumental, the look-up
/// failed) cannot be sung at all. Words that cannot be timed (no speech
/// model, or the timing failed) are sung as they are.
pub fn not_singable(state: &State, index: usize) -> Option<NotSingable> {
    if let Some(none) = no_karaoke(state) {
        return Some(none);
    }
    let path = state.queue.paths.get(index)?;
    let left = still_to_do(state, path);
    if !left.is_empty() {
        let jobs: Vec<&str> = left.iter().map(|&job| job_words(state, job)).collect();
        return Some(NotSingable::Preparing(jobs.join("; ")));
    }
    match lyrics_of(state, index) {
        // Done, but not here yet (read from the cache as they are wanted).
        None | Some(SongLyrics::Pending) => {
            Some(NotSingable::Preparing(job_words(state, Job::Lyrics).into()))
        }
        Some(SongLyrics::NotFound) => Some(NotSingable::Nothing("no lyrics were found for it")),
        Some(SongLyrics::Instrumental) => Some(NotSingable::Nothing("it is an instrumental")),
        Some(SongLyrics::Failed(_)) => {
            Some(NotSingable::Nothing("its lyrics could not be looked up"))
        }
        Some(SongLyrics::Found { .. }) => None,
    }
}

/// After every action: a Next waiting for its song
/// ([`super::KaraokeState::start_when_ready`]) starts it, with its
/// count-in, as soon as the song is prepared; it is called off once the
/// mode is left or is about another song, and refused (said why) if the
/// song turns out to have nothing to sing.
pub(crate) fn start_when_ready(state: &mut State, fx: &mut Vec<Effect>) {
    let Some(index) = state.karaoke.start_when_ready else {
        return;
    };
    if !in_mode(state) || song_index(state) != Some(index) {
        state.karaoke.start_when_ready = None;
        return;
    }
    match not_singable(state, index) {
        Some(NotSingable::Preparing(_)) => {}
        Some(_) => {
            state.karaoke.start_when_ready = None;
            super::refused(state, Some(index));
        }
        None => {
            state.karaoke.start_when_ready = None;
            super::start(state, fx);
        }
    }
}
