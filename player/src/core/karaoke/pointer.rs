//! Karaoke mode's pointer, as in a video player: it hides once it has been
//! still for [`IDLE_MS`] while the mode shows, and shows at once on any
//! move, anywhere on the screen (the time of the last one is kept as the
//! controls hear it: [`super::controls::pointer_moved`]; the mode coming
//! counts as one). Out of the mode it always shows. Its own rule: the
//! controls' showing or not plays no part. One clock, the frame's
//! ([`clock_ms`]).

use super::{clock_ms, in_mode};
use crate::core::State;

/// How long the pointer stays still before it hides, in ms.
pub const IDLE_MS: u64 = 2_500;

/// Whether the pointer hides, the mode showing or not (`mode`) and the
/// pointer still for `still_ms`.
pub fn hides(mode: bool, still_ms: u64) -> bool {
    mode && still_ms >= IDLE_MS
}

/// How long the pointer has been still now, in ms.
fn still_ms(state: &State) -> u64 {
    clock_ms(state).saturating_sub(state.karaoke.pointer_moved)
}

/// Whether the pointer hides now ([`hides`]).
pub fn hidden(state: &State) -> bool {
    hides(in_mode(state), still_ms(state))
}

/// When the pointer is to hide, while that is still to come.
pub fn due(state: &State) -> Option<u64> {
    (in_mode(state) && !hidden(state)).then_some(state.karaoke.pointer_moved + IDLE_MS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::core::karaoke::controls::pointer_moved;
    use crate::core::mouse::{action, MouseButton, MouseInput, MouseKind};
    use crate::core::{Action, Screen};

    /// The rule alone: out of the mode never; in it, from [`IDLE_MS`]
    /// still.
    #[test]
    fn the_pointer_hides_after_it_was_still_in_the_mode_only() {
        assert!(!hides(true, 0));
        assert!(!hides(true, IDLE_MS - 1));
        assert!(hides(true, IDLE_MS));
        assert!(hides(true, 60_000));
        assert!(!hides(false, IDLE_MS));
        assert!(!hides(false, 60_000));
    }

    /// In the mode it hides 2.5 s after the last move, wherever that was
    /// (the controls up at the bottom or not); a move anywhere, a drag
    /// too, shows it at once; leaving the mode shows it, and coming back
    /// counts as a move.
    #[test]
    fn the_pointer_hides_when_still_and_shows_when_moved_or_out() {
        let mut state = State::new("/m".into(), Config::default(), (100, 40));
        state.screen = Screen::Karaoke;
        state.now_ms = 10_000;
        let rows = state.size.1;
        // Still since a move in the middle.
        pointer_moved(&mut state, rows / 2);
        assert_eq!(due(&state), Some(12_500));
        state.now_ms = 12_499;
        assert!(!hidden(&state));
        state.now_ms = 12_500;
        assert!(hidden(&state) && due(&state).is_none());
        // A move shows it at once; at the bottom, the controls up, it
        // still hides on time.
        pointer_moved(&mut state, rows - 1);
        assert!(!hidden(&state));
        state.now_ms = 15_000;
        assert!(hidden(&state) && super::super::controls::shown(&state));
        // A drag over the mode is a move too.
        let drag = MouseInput {
            kind: MouseKind::Drag(MouseButton::Left),
            col: 3,
            row: 4,
        };
        assert!(matches!(
            action(&state, drag),
            Some(Action::PointerMoved { row: 4 })
        ));
        // Out of the mode it shows, however long it was still.
        crate::core::karaoke::leave(&mut state, &mut Vec::new());
        state.now_ms = 60_000;
        assert!(!hidden(&state) && due(&state).is_none());
        // Back in: it shows as the mode comes, and hides 2.5 s on.
        crate::core::karaoke::enter(&mut state, &mut Vec::new(), false);
        assert!(!hidden(&state));
        assert_eq!(due(&state), Some(62_500));
        state.now_ms = 62_500;
        assert!(hidden(&state));
    }
}
