//! Bar jumps: the Transition tab's digit keys move playback of the playing
//! track to the start of a bar some bars from the one heard, at once or
//! when the bar heard ends, without a gap. They act on playback, not on
//! the tab, so they are carried out with the player's own keys.

use jodel_playback::Command;
use jodel_rhythm::Analysis;

use crate::core::lanes::{rhythmed, Block};
use crate::core::transition::Status;
use crate::core::update::say_info;
use crate::core::{Effect, State};

/// The start of the bar `offset` bars from the one heard (0: its own
/// start) and the start of the next bar, in seconds of the playing track:
/// past the bars found, at the bar's length.
pub fn bar_jump(a: &Analysis, heard: f64, offset: isize) -> Option<(f64, f64)> {
    let starts: Vec<f64> = a
        .downbeats
        .iter()
        .filter_map(|&d| a.beats.get(d).copied())
        .collect();
    let n = starts.len();
    if n < 2 {
        return None;
    }
    let bar = (starts[n - 1] - starts[0]) / (n - 1) as f64;
    let at = |i: isize| -> f64 {
        if i < 0 {
            starts[0] + i as f64 * bar
        } else if i as usize >= n {
            starts[n - 1] + (i - (n as isize - 1)) as f64 * bar
        } else {
            starts[i as usize]
        }
    };
    // The bar heard (before the first bar found: a bar's length before it).
    let k = starts.partition_point(|&t| t <= heard) as isize - 1;
    let to = at(k + offset);
    (to >= 0.0 && to < a.duration).then(|| (to, at(k + 1)))
}

/// The playing track jumps to the start of the bar `offset` bars from the
/// one heard (0: its own start): at once, or (`at_end`) when the bar heard
/// ends. Without a gap: the engine keeps the bars around what is heard.
pub(crate) fn jump_bar(state: &mut State, offset: isize, at_end: bool, fx: &mut Vec<Effect>) {
    let (Some(index), Some(_)) = (state.queue.current, state.playing.track.as_ref()) else {
        return;
    };
    if state.transition.draft.status != Status::Draft {
        return say_info(state, "No bar jumps while a transition is set up");
    }
    let Some(r) = rhythmed(state, Block::Now).cloned() else {
        return say_info(state, "No bars yet: the track is being analysed");
    };
    let Some((to, end)) = bar_jump(&r.analysis, state.heard(), offset) else {
        return say_info(state, "No bar there");
    };
    let at = if at_end {
        jodel_playback::JumpAt::At(end)
    } else {
        jodel_playback::JumpAt::Now
    };
    fx.push(Effect::Engine(Command::Jump { index, to, at }));
}
