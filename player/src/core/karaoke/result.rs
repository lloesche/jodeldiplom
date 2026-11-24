//! A song sung to its end in Karaoke mode, the game on: how it went, kept
//! and shown ([`Shown`]: its reveal under way, then settled) while the next
//! song is held for START, and its best score raised where it was beaten.
//! The first START during the reveal completes it at once; the next goes
//! on, as ever.
//!
//! The reveal goes by one clock, for what is drawn and what is heard
//! alike ([`clock_ms`]): the frame's.

use std::path::Path;
use std::sync::Arc;

use super::in_mode;
use crate::core::singing::{Medal, Motion, Shown, SongResult};
use crate::core::State;

/// The reveal's clock: the time of the frame drawn last, or the tick's
/// where that is older (the core's clock moves only a tick at a time, a
/// frame's with every frame).
pub fn clock_ms(state: &State) -> u64 {
    state.drawn_ms.unwrap_or(0).max(state.now_ms)
}

/// The result shown in Karaoke mode now: the song before's, while the
/// next is held for START (none once a count-in or a song goes on).
pub fn result_shown(state: &State) -> Option<&Arc<Shown>> {
    let k = &state.karaoke;
    let waiting = k.held.is_some() && k.countdown.is_none() && k.resume_at.is_none();
    k.result.as_ref().filter(|_| waiting && in_mode(state))
}

/// The result shown now and its reveal's time ([`clock_ms`]).
pub fn result_now(state: &State) -> Option<(&Arc<Shown>, f64)> {
    result_shown(state).map(|s| (s, s.time(clock_ms(state))))
}

/// How the result shown moves now, if one shows.
pub fn result_motion(state: &State) -> Option<Motion> {
    result_shown(state).map(|s| s.motion(clock_ms(state)))
}

/// START while the result's reveal is under way: it is completed at once
/// (everything shown, nothing played again), and that is all this START
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

/// The song in the mode sung to its end, the game on: how it went (its
/// points, words hit, the phrases' tiers, medal and best before), kept to
/// show, and its best score raised where it was beaten.
pub(super) fn sung_to_the_end(state: &mut State) {
    if !state.config.karaoke_game {
        return;
    }
    // Its last words and phrases scored (heard on to past its end).
    state.karaoke.singing.finish();
    let singing = &state.karaoke.singing;
    let Some(path) = singing.song().map(Path::to_path_buf) else {
        return;
    };
    if singing.tally.scored_words == 0 {
        return;
    }
    let tally = singing.tally;
    let (points, hit, words) = (tally.score, tally.hit_words, tally.scored_words);
    let key = path.display().to_string();
    let best = state.karaoke.bests.get(&key).map(|b| b.points);
    let title = state
        .details
        .get(&path)
        .and_then(|d| d.title.clone())
        .unwrap_or_else(|| crate::core::shown_title(state, &path));
    // Only a song sung from its start (most of its words heard) wins a
    // medal or a best.
    let whole = u64::from(words) * 10 >= singing.words() as u64 * 9;
    let result = SongResult {
        title,
        points,
        hit,
        words,
        streak: tally.best_streak,
        phrases: singing.shone().copied().collect(),
        medal: whole.then(|| Medal::of(hit, words)).flatten(),
        best,
        whole,
    };
    if result.new_best() {
        state
            .karaoke
            .bests
            .insert(key, crate::scores::Best { points, hit, words });
        state.karaoke.bests_changed = true;
    }
    state.karaoke.results += 1;
    let shown = Shown::new(result, state.karaoke.results, clock_ms(state));
    state.karaoke.result = Some(Arc::new(shown));
    state.karaoke.singing.restart();
}
