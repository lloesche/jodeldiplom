//! The count-in and the title before it: a song START plays shows its
//! title and artist, then counts "3", "2", "1" a second each, ending as its
//! first word is sung ([`Countdown`]); a paused song goes on after a
//! count-in of its own ([`resume_count`]). Where the song has been heard
//! from without a jump decides whether one shows whole ([`counts_from`]).
//!
//! The clocks: the count-in of a song START plays goes by what is heard
//! ([`State::heard`], the frame's), a paused song's by the clock
//! ([`State::now_ms`]; the picture's by its frame's); whether a count-in is
//! over ([`super::tick`]) by the position the engine reports.

use jodel_playback::PlayState;

use super::{lyrics_of, SongLyrics};
use crate::core::State;

/// The numbers a count-in counts: "3", "2", "1", then the music.
pub const COUNT: usize = 3;
/// How long each number of a count-in shows, in seconds.
pub const COUNT_STEP: f64 = 1.0;
/// How long a song's title and artist show alone as it starts, in seconds,
/// before they fade out.
pub const TITLE_HOLD: f64 = 2.0;
/// How long they take to fade out, in seconds.
pub const TITLE_FADE: f64 = 1.0;
/// A count-in: the track, how long each number shows, and the second of
/// the track heard right after "1" (where the singing starts).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Countdown {
    /// The track.
    pub index: usize,
    /// How long each number shows, in seconds ([`COUNT_STEP`]).
    pub beat: f64,
    /// Seconds into the track heard right after "1": its first sung word,
    /// or its start.
    pub on: f64,
    /// Seconds from when the song starts to be heard (its lead's start)
    /// before the count may begin: the title's time, or none.
    pub title: f64,
}

impl Countdown {
    /// The count-in for track `index`: a number a second, then the track
    /// from its start.
    pub fn new(index: usize) -> Countdown {
        Countdown::before(index, 0.0)
    }

    /// The count-in for track `index` whose singing starts `on` seconds in:
    /// "1" ends there.
    pub fn before(index: usize, on: f64) -> Countdown {
        Countdown {
            index,
            beat: COUNT_STEP,
            on: on.max(0.0),
            title: 0.0,
        }
    }

    /// [`Countdown::before`], after the song's title has shown and faded
    /// ([`TITLE_HOLD`], [`TITLE_FADE`]): the music waits for it where the
    /// singing starts too soon.
    pub fn after_title(index: usize, on: f64) -> Countdown {
        Countdown {
            title: TITLE_HOLD + TITLE_FADE,
            ..Countdown::before(index, on)
        }
    }

    /// Seconds of silence before the track: the title's time and the
    /// count's beats before [`Countdown::on`], less what the track itself
    /// has before it.
    pub fn lead(&self) -> f64 {
        (self.title + COUNT as f64 * self.beat - self.on).max(0.0)
    }

    /// The number shown at `t` seconds into the track (negative in the
    /// silence before it) and how far through its beat, 0 to 1; `None`
    /// before the count and once it is over.
    pub fn at(&self, t: f64) -> Option<(usize, f64)> {
        count_at((self.on - t) / self.beat)
    }

    /// Whether the count is over at `t`.
    pub fn over(&self, t: f64) -> bool {
        t >= self.on
    }
}

/// The count-in before a paused song goes on at `now_ms`: the number shown
/// and how far through its second, if one is under way.
pub fn resume_count(state: &State, now_ms: u64) -> Option<(usize, f64)> {
    let at = state.karaoke.resume_at?;
    count_at(at.saturating_sub(now_ms) as f64 / 1000.0 / COUNT_STEP)
}

/// The number a count-in shows with `left` of its beats to come before it
/// ends, and how far through its beat, 0 to 1; `None` before the count
/// (more than [`COUNT`] to come) and once it is over.
fn count_at(left: f64) -> Option<(usize, f64)> {
    if left <= 0.0 || left > COUNT as f64 {
        return None;
    }
    let n = left.ceil();
    Some((n as usize, n - left))
}

/// The count-in for track `index`: after its title has shown and faded,
/// ending as its first word is sung, when its words are timed; else before
/// its start (the title stays, with what is known of the lyrics).
pub fn countdown_for(state: &State, index: usize) -> Countdown {
    match first_sung(state, index) {
        Some(on) => Countdown::after_title(index, on),
        None => Countdown::new(index),
    }
}

/// How much of track `index`'s title and artist shows at `t` (seconds into
/// it, as drawn), 0 to 1: all of it until it plays (held for START, or
/// waiting for its lead), then [`TITLE_HOLD`] from when it is first heard
/// (its lead's start, after START), fading out over [`TITLE_FADE`]; once,
/// whatever comes after (the count-in over, its record gone). None for a
/// song heard from further in (the mode entered with shift, a seek, a pause
/// ended).
pub fn title_shown(state: &State, index: usize, t: f64) -> f64 {
    let playing = state.playing.state == PlayState::Playing
        && state
            .playing
            .track
            .as_ref()
            .is_some_and(|tr| tr.index == index);
    let from = match state.karaoke.countdown {
        Some(c) if c.index == index && !playing => return 1.0,
        Some(c) if c.index == index => -c.lead(),
        _ if !playing && t < TITLE_HOLD => return 1.0,
        // Heard without a jump from there: its lead's start after START
        // (also once the count-in is over), else where it was joined.
        _ => counts_from(state, index),
    };
    if from > 0.05 {
        return 0.0;
    }
    (1.0 - (t - from - TITLE_HOLD) / TITLE_FADE).clamp(0.0, 1.0)
}

/// When the first word of track `index` is sung (seconds in), once its
/// words are timed.
pub fn first_sung(state: &State, index: usize) -> Option<f64> {
    let Some(SongLyrics::Found { shown, .. }) = lyrics_of(state, index) else {
        return None;
    };
    shown
        .lines
        .iter()
        .flat_map(|l| &l.words)
        .filter_map(|w| w.start)
        .reduce(f64::min)
}

/// Whether the song heard is in the silence before it (a count-in's lead):
/// at a negative time.
pub(super) fn in_lead(state: &State) -> bool {
    match (state.karaoke.countdown, &state.playing.track) {
        (Some(c), Some(t)) => t.index == c.index && state.heard() < 0.0,
        (Some(_), None) => true,
        _ => false,
    }
}

/// Notes where the song is heard from when it jumped (another song, a
/// seek, playing again after a pause): the count-ins ([`counts_from`])
/// only show whole after that.
pub(super) fn note_heard(state: &mut State) {
    let Some(index) = state.playing.track.as_ref().map(|t| t.index) else {
        state.karaoke.last_heard = None;
        return;
    };
    let at = state.heard();
    let playing = state.playing.state == PlayState::Playing;
    let jumped = match state.karaoke.last_heard {
        Some((i, was, then)) if i == index => {
            let went = (state.now_ms.saturating_sub(then)) as f64 / 1000.0;
            // Playing: about as far as the clock went. Not playing: nothing
            // is heard, so only where it plays from again counts.
            playing && (at - was - went).abs() > 1.0
        }
        _ => true,
    };
    if jumped || !playing {
        state.karaoke.counts_from = Some((index, at));
    }
    state.karaoke.last_heard = Some((index, at, state.now_ms));
}

/// Where the song playing (`index`) has been heard from without a jump: a
/// count-in shows only if it fits whole after it. Before anything is known,
/// from its start.
pub fn counts_from(state: &State, index: usize) -> f64 {
    match state.karaoke.counts_from {
        Some((i, at)) if i == index => at,
        _ => 0.0,
    }
}
