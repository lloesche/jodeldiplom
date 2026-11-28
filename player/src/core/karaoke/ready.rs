//! Whether a song can be sung now ([`not_singable`]): karaoke needs a
//! graphics card, the song fully prepared (every job the queue's icons
//! show for it done, [`needed_to_sing`]: its story only for the game,
//! whose backdrop it is) and lyrics to sing; where stories are made, the
//! game a story ([`NO_STORY`]).
//! A Next pressed in the mode on a song still being prepared waits for it
//! ([`start_when_ready`]).

use std::path::Path;

use super::{in_mode, lyrics_of, lyrics_wanted, song_index, SongLyrics};
use crate::core::{job_done, job_words, Effect, Gpu, Job, LyricsFacts, State, StoryFact};

/// Why a song with no story cannot be played in the game.
pub const NO_STORY: &str = "no story could be made for it (it can be sung with the game off)";

/// Why a song cannot be sung now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NotSingable {
    /// No karaoke at all: no graphics card (or still looking for one).
    Gpu(String),
    /// Still being prepared: what is still to come for it, job by job in
    /// the words of the queue's icons ([`job_words`]). It can be sung as
    /// soon as that is done.
    Preparing(String),
    /// Nothing to sing (no lyrics found, an instrumental, the look-up
    /// failed), or nothing to play the game over (no story: [`NO_STORY`]).
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
/// jobs the queue's icons show, in their words, with the step each is at;
/// its story only with the game on); once prepared, a song with nothing to
/// sing (no lyrics found, an instrumental, the look-up failed) cannot be
/// sung at all, nor one with no story in the game ([`NO_STORY`]). Words
/// that cannot be timed (no speech model, or the timing failed) are sung
/// as they are.
pub fn not_singable(state: &State, index: usize) -> Option<NotSingable> {
    if let Some(none) = no_karaoke(state) {
        return Some(none);
    }
    let path = state.queue.paths.get(index)?;
    let left: Vec<String> = (needed_to_sing(state))
        .filter(|&job| !job_done(state, path, job))
        .map(|job| job_words(state, path, job))
        .collect();
    if !left.is_empty() {
        return Some(NotSingable::Preparing(left.join("; ")));
    }
    match words(state, index, path) {
        Words::Coming => Some(NotSingable::Preparing(job_words(state, path, Job::Lyrics))),
        Words::Nothing(why) => Some(NotSingable::Nothing(why)),
        Words::Found => {
            let story = state.facts.get(path).and_then(|f| f.story.as_ref());
            let none = state.config.karaoke_game && story == Some(&StoryFact::NoStory);
            none.then_some(NotSingable::Nothing(NO_STORY))
        }
    }
}

/// What a prepared song's lyrics leave to sing ([`words`]).
enum Words {
    /// Settled, but not here yet: read from the cache as they are wanted.
    Coming,
    /// Nothing to sing, and why.
    Nothing(&'static str),
    /// Lyrics to sing.
    Found,
}

/// What the lyrics of track `index` (at `path`, its lyrics' job done)
/// leave to sing: as they are here, read in for showing; lyrics read in
/// for showing ([`lyrics_wanted`]: karaoke's songs, the lyrics editor's,
/// the Transition tab's) and not here yet are still to come. A song whose
/// lyrics nothing shows (further down the queue, or no karaoke shows) is
/// never read in, so it is as its lyrics were settled
/// ([`crate::core::QueueFacts::lyrics`]), not still being prepared for
/// ever.
fn words(state: &State, index: usize, path: &Path) -> Words {
    let how = match lyrics_of(state, index) {
        Some(SongLyrics::Found { .. }) => return Words::Found,
        Some(SongLyrics::NotFound) => LyricsFacts::NotFound,
        Some(SongLyrics::Instrumental) => LyricsFacts::Instrumental,
        Some(SongLyrics::Failed(_)) => LyricsFacts::Failed,
        None | Some(SongLyrics::Pending) => {
            let settled = state.facts.get(path).and_then(|f| f.lyrics);
            // The live list, never the one kept in `state.karaoke`: that
            // is brought up to date only after the action, so an action
            // that enters the mode and starts the song (the palette's
            // enter) would find the song not wanted yet and start it with
            // no lyrics read in.
            match settled {
                Some(l) if !lyrics_wanted(state).iter().any(|p| p == path) => l.how,
                _ => return Words::Coming,
            }
        }
    };
    match how {
        LyricsFacts::Timed | LyricsFacts::Untimed | LyricsFacts::Heard => Words::Found,
        LyricsFacts::NotFound => Words::Nothing("no lyrics were found for it"),
        LyricsFacts::Instrumental => Words::Nothing("it is an instrumental"),
        LyricsFacts::Failed => Words::Nothing("its lyrics could not be looked up"),
    }
}

/// The jobs a song must have done to be sung, of those the queue is got
/// ready by ([`State::preparing`]), in their order: all of them, but the
/// story only for the game (it is the game's backdrop; plain lyrics
/// karaoke sings over the blurred cover).
fn needed_to_sing(state: &State) -> impl Iterator<Item = Job> + '_ {
    let game = state.config.karaoke_game;
    (state.preparing.iter().copied()).filter(move |&job| game || job != Job::Story)
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
