//! Karaoke mode's controls: the voice slider, the seek bar and the keys,
//! a bar over the picture's bottom (the window lays it out; the picture
//! never moves for them). They come up while the pointer is near the
//! bottom ([`in_reach`]) and stay while it is there, go [`LINGER_MS`] after
//! it leaves (off the window too), and fade in and out over [`FADE_MS`];
//! while a slider or the seek bar is held they show whatever the pointer
//! does. Nothing else brings them up (no key): START is the stage's own
//! pill, never in the bar. One clock for all of it, the frame's
//! ([`clock_ms`]): they are laid out (and take clicks) exactly while they
//! show at all.

use super::{clock_ms, in_mode};
use crate::core::State;

/// The share of the window's height, from the bottom, where the pointer
/// brings the controls up (never less than the bar and two lines:
/// [`crate::core::Fit::controls_reach`]).
const REACH: f32 = 0.15;
/// How long the controls stay after the pointer left the bottom, in ms.
pub const LINGER_MS: u64 = 1_500;
/// How long they take to fade in, or out, in ms.
pub const FADE_MS: u64 = 200;

/// Where the controls are by the pointer ([`State::now_ms`]'s clock).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Controls {
    /// Not up.
    #[default]
    Hidden,
    /// Up since `from` (fading in from then), the pointer near the bottom.
    Up {
        /// When they came up.
        from: u64,
    },
    /// Up since `from`, going from `until` (fading out from then).
    Going {
        /// When they came up.
        from: u64,
        /// When they start to go.
        until: u64,
    },
}

/// Whether the pointer, `from_bottom` of the window's height above its
/// bottom, is near enough the bottom to bring the controls up: within the
/// last [`REACH`] of the height, never less than the bar and two lines.
fn in_reach(state: &State, from_bottom: f32) -> bool {
    from_bottom < REACH.max(state.fit.controls_reach)
}

/// Whether a slider or the seek bar is held: the controls then show.
fn held(state: &State) -> bool {
    state.karaoke.dragging.is_some() || state.seek_drag.is_some()
}

/// How far a fade from `start` has come at `t` (ms), 0 to 1.
fn ramp(start: u64, t: u64) -> f64 {
    ((t as f64 - start as f64) / FADE_MS as f64).clamp(0.0, 1.0)
}

/// How much the controls show at `t` by the pointer alone, 0 to 1: faded
/// in from when they came up, out from when they go.
fn faded_at(controls: Controls, t: u64) -> f64 {
    match controls {
        Controls::Hidden => 0.0,
        Controls::Up { from } => ramp(from, t),
        Controls::Going { from, until } => ramp(from, t).min(1.0 - ramp(until, t)),
    }
}

/// How much the controls show in the frame drawn now, 0 (not at all) to 1.
pub fn fade(state: &State) -> f64 {
    if held(state) {
        1.0
    } else {
        faded_at(state.karaoke.controls, clock_ms(state))
    }
}

/// Whether the mode lays its controls out now (and they take clicks):
/// while they show at all, and while a slider or the seek bar is held.
pub fn shown(state: &State) -> bool {
    held(state) || fade(state) > 0.0
}

/// Whether the controls are fading in or out now (the frames then come at
/// the display's refresh).
pub fn fading(state: &State) -> bool {
    let t = clock_ms(state);
    let ramping = |start: u64| (start..start + FADE_MS).contains(&t);
    !held(state)
        && match state.karaoke.controls {
            Controls::Hidden => false,
            Controls::Up { from } => ramping(from),
            Controls::Going { from, until } => ramping(from) || ramping(until),
        }
}

/// When the controls start to go, while that is still to come: the fade
/// out then begins on time.
pub fn due(state: &State) -> Option<u64> {
    match state.karaoke.controls {
        Controls::Going { until, .. } if until > clock_ms(state) => Some(until),
        _ => None,
    }
}

/// The pointer moved over the mode (anywhere, however little), to
/// `from_bottom` of the window's height above its bottom: the time is
/// kept, for the pointer to show ([`super::pointer`]). Near the bottom the
/// controls come up (fading in from where a fade out had got to) and stay;
/// anywhere else those up go after [`LINGER_MS`].
pub(crate) fn pointer_moved(state: &mut State, from_bottom: f32) {
    if !in_mode(state) {
        return;
    }
    let t = clock_ms(state);
    state.karaoke.pointer_moved = t;
    let controls = state.karaoke.controls;
    state.karaoke.controls = match (in_reach(state, from_bottom), controls) {
        (true, Controls::Up { .. }) | (false, Controls::Hidden | Controls::Going { .. }) => {
            controls
        }
        (true, _) => {
            let shown = faded_at(controls, t) * FADE_MS as f64;
            Controls::Up {
                from: t - shown.round() as u64,
            }
        }
        (false, Controls::Up { from }) => Controls::Going {
            from,
            until: t + LINGER_MS,
        },
    };
}

/// The pointer left the window: controls up go after [`LINGER_MS`].
pub(crate) fn pointer_left(state: &mut State) {
    if let Controls::Up { from } = state.karaoke.controls {
        let until = clock_ms(state) + LINGER_MS;
        state.karaoke.controls = Controls::Going { from, until };
    }
}

/// How the fade splits between the stage's own progress (along the bottom,
/// where the bar comes) and the bar, so the two never show at once: the
/// progress is gone in the fade's first half, the bar shows in its second.
/// How far the progress has gone and how much the bar shows, at `fade`.
pub fn handover(fade: f64) -> (f64, f64) {
    ((fade * 2.0).min(1.0), (fade * 2.0 - 1.0).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::core::Screen;

    /// How far up the window the controls reach in the tests, as the
    /// window says (their bar and two lines).
    const BAR_REACH: f32 = 0.2;

    fn mode() -> State {
        let mut state = State::new("/m".into(), Config::default());
        state.fit.controls_reach = BAR_REACH;
        state.screen = Screen::Karaoke;
        state.now_ms = 10_000;
        state
    }

    fn at(state: &mut State, ms: u64) -> f64 {
        state.now_ms = ms;
        fade(state)
    }

    /// Near the bottom (the last 15 % of the window, never less than the
    /// bar and two lines) they fade in and stay however long the pointer
    /// rests; off it they linger, then fade out; back before they are
    /// gone, they fade in again from where they were, never popping. Laid
    /// out exactly while they show at all.
    #[test]
    fn the_controls_fade_in_stay_linger_and_fade_out() {
        let mut state = mode();
        assert!(in_reach(&state, BAR_REACH - 0.001) && !in_reach(&state, BAR_REACH));
        state.fit.controls_reach = 0.05;
        assert!(in_reach(&state, REACH - 0.001) && !in_reach(&state, REACH));
        state.fit.controls_reach = BAR_REACH;
        pointer_moved(&mut state, 0.5);
        assert_eq!(state.karaoke.controls, Controls::Hidden);
        pointer_moved(&mut state, 0.0);
        assert!(!shown(&state) && fading(&state));
        assert_eq!(at(&mut state, 10_000), 0.0);
        assert!((at(&mut state, 10_100) - 0.5).abs() < 1e-9 && shown(&state));
        assert_eq!(at(&mut state, 60_000), 1.0);
        assert!(!fading(&state) && due(&state).is_none());
        // Off the bottom: 1.5 s more, then out over 0.2 s.
        pointer_moved(&mut state, 0.97);
        assert_eq!(due(&state), Some(61_500));
        assert_eq!(at(&mut state, 61_499), 1.0);
        assert!((at(&mut state, 61_600) - 0.5).abs() < 1e-9);
        assert!(fading(&state));
        // Back half way out: in again from a half.
        pointer_moved(&mut state, 0.03);
        assert!((at(&mut state, 61_600) - 0.5).abs() < 1e-9);
        assert!((at(&mut state, 61_650) - 0.75).abs() < 1e-9);
        // Off the window: they go as well, and once gone take no clicks.
        pointer_left(&mut state);
        assert_eq!(due(&state), Some(63_150));
        assert!(shown(&state));
        assert_eq!(at(&mut state, 63_350), 0.0);
        assert!(!shown(&state) && !fading(&state) && due(&state).is_none());
    }

    /// A slider held keeps them up, wherever the pointer is.
    #[test]
    fn a_slider_held_keeps_them_up() {
        let mut state = mode();
        state.karaoke.dragging = Some(crate::core::sliders::Slider::Voice);
        assert!(shown(&state) && fade(&state) == 1.0);
        assert!(!fading(&state));
    }

    /// The stage's progress is gone before the bar shows: never both.
    #[test]
    fn the_progress_goes_before_the_bar_comes() {
        assert_eq!(handover(0.0), (0.0, 0.0));
        assert_eq!(handover(0.25), (0.5, 0.0));
        assert_eq!(handover(0.5), (1.0, 0.0));
        assert_eq!(handover(0.75), (1.0, 0.5));
        assert_eq!(handover(1.0), (1.0, 1.0));
        for k in 0..=100 {
            let (gone, bar) = handover(f64::from(k) / 100.0);
            assert!(gone == 1.0 || bar == 0.0, "{k}");
        }
    }
}
