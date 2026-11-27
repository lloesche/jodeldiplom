//! A song heard to its end in Karaoke mode, the game on: how it went, kept
//! and shown ([`Shown`]: its reveal under way, then settled) while the mode
//! waits for Next (the next song held, or the queue played out), and its
//! best score raised where it was beaten. Every song played out has one,
//! however little of it was sung. The first Next during the reveal
//! completes it at once; the next goes on, as ever. How a song's scoring
//! comes to a result is [`SongResult`]'s: its medal and verdict and their
//! words (`verdict`); its reveal on the diploma, a timeline the stage
//! draws and the sounds play alike (`reveal`); the examiner's signature
//! on it ([`signature`]).
//!
//! The reveal goes by one clock, for what is drawn and what is heard
//! alike ([`clock_ms`]): the frame's.

use std::sync::Arc;

use super::in_mode;
use crate::core::State;

mod reveal;
pub mod signature;
mod verdict;

pub use reveal::{
    ease_out, ease_out_pace, signed, signing, unrolled, unrolling, Beat, Motion, Reveal, Shown,
    Side, DATE_FROM, DATING, DROP, GRADE_WRITE, SIGNING, STAMP_DOWN, STREAK_WRITE, UNROLL,
    WORDS_FILL,
};
pub use verdict::{verdict_words, Medal, SongResult, Verdict};

/// The reveal's clock: the time of the frame drawn last, or the tick's
/// where that is older (the core's clock moves only a tick at a time, a
/// frame's with every frame).
pub fn clock_ms(state: &State) -> u64 {
    state.drawn_ms.unwrap_or(0).max(state.now_ms)
}

/// The result shown in Karaoke mode now: the song before's, from its end
/// until Next, the next track heard, or the mode entered again.
pub fn result_shown(state: &State) -> Option<&Arc<Shown>> {
    state.karaoke.result.as_ref().filter(|_| in_mode(state))
}

/// The result shown now and its reveal's time ([`clock_ms`]).
pub fn result_now(state: &State) -> Option<(&Arc<Shown>, f64)> {
    result_shown(state).map(|s| (s, s.time(clock_ms(state))))
}

/// How the result shown moves now, if one shows.
pub fn result_motion(state: &State) -> Option<Motion> {
    result_shown(state).map(|s| s.motion(clock_ms(state)))
}

/// Next while the result's reveal is under way: it is completed at once
/// (everything shown, nothing played again), and that is all this Next
/// does. False where there is no reveal under way.
pub(super) fn skipped(state: &mut State) -> bool {
    let now = clock_ms(state);
    if !result_shown(state).is_some_and(|s| s.revealing(now)) {
        return false;
    }
    if let Some(shown) = state.karaoke.result.as_mut() {
        Arc::make_mut(shown).skip(now);
    }
    true
}

/// A track starts to be heard (Next's, or one a mix ran into): the result
/// of the song before is gone.
pub(super) fn track_started(state: &mut State) {
    state.karaoke.result = None;
}

/// The engine says track `index` was heard to where it ended: its own end,
/// or the end of a mix into the next one, cut short or not (never where it
/// was stopped or left for another). In the mode, the game on, how it went
/// (its points, words hit, the phrases' tiers, medal and best before) is
/// kept to show, and its best score raised where it was beaten; the next
/// track heard puts it away ([`track_started`]). Its singing is what
/// was scored of it: nothing where its notes were never made or sung
/// ([`crate::core::karaoke::result::Verdict::Silent`]).
pub(crate) fn ended(state: &mut State, index: usize) {
    if !in_mode(state) || !state.show().scored {
        return;
    }
    let Some(path) = state.queue.paths.get(index).cloned() else {
        return;
    };
    let singing = &mut state.karaoke.singing;
    let (tally, phrases, notes) = if singing.is_of(&path) {
        singing.take()
    } else {
        Default::default()
    };
    let key = path.display().to_string();
    let best = state.karaoke.bests.get(&key).map(|b| b.points);
    let title = state
        .details
        .get(&path)
        .and_then(|d| d.title.clone())
        .unwrap_or_else(|| crate::core::shown_title(state, &path));
    let result = SongResult::new(title, &tally, phrases, notes, best);
    if result.new_best() {
        let (points, hit, words) = (result.points, result.hit, result.words);
        state
            .karaoke
            .bests
            .insert(key, crate::store::scores::Best { points, hit, words });
        state.karaoke.bests_changed = true;
    }
    state.karaoke.results += 1;
    let shown = Shown::new(result, state.karaoke.results, clock_ms(state));
    state.karaoke.result = Some(Arc::new(shown));
}
