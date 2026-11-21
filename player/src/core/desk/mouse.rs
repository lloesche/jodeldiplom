//! The mouse on the desk: a fader's cap dragged up and down (held, so the
//! pointer may leave its track), the wheel over a strip, a double click
//! on a fader for 0 dB, and the buttons. The window knows the pointer to
//! the pixel, so a fader moves by less than a cell; near 0 dB it stays at
//! 0 dB, as in a desk's detent.

use ratatui::layout::Rect;

use super::{taper, Channel, DeskAction, DeskButton, StripKind};
use crate::core::layout::{self, Layout, Track};
use crate::core::mouse::{MouseButton, MouseKind, Target};
use crate::core::{Action, State};

/// Within this many rows of 0 dB a dragged cap is at 0 dB exactly: the
/// detent, which bit-perfect playback depends on landing in.
pub const DETENT_ROWS: f64 = 0.4;

/// The position (0: the bottom, 1: the top) with the pointer `y` cells
/// from the top: the middle of the track's first row is the top, of its
/// last row the bottom ([`Track`]). Within [`DETENT_ROWS`] of 0 dB it is
/// 0 dB's.
pub fn position_at(track: Rect, y: f64) -> f64 {
    let rows = Track::rows(track);
    let position = 1.0 - rows.share_at(y).unwrap_or(1.0);
    let unity = rows.centre(1.0 - taper::UNITY);
    if (y - unity).abs() < DETENT_ROWS {
        taper::UNITY
    } else {
        position
    }
}

/// Where the middle of the cap at `position` is, in cells from the top.
pub fn cap_y(track: Rect, position: f64) -> f64 {
    Track::rows(track).centre(1.0 - position)
}

/// The row the cap at `position` is drawn on.
pub fn cap_row(track: Rect, position: f64) -> u16 {
    Track::rows(track).cell(1.0 - position)
}

/// `strip`'s fader track as laid out now.
fn track(state: &State, strip: StripKind) -> Option<Rect> {
    let Layout::Main(m) = layout::compute(state) else {
        return None;
    };
    m.desk?.channel(Channel::Strip(strip))?.track
}

/// While a fader is held, what the pointer does to it, wherever it is
/// (`None`: none is held).
pub(crate) fn held(state: &State, kind: MouseKind, y: f64) -> Option<Option<Action>> {
    let drag = state.desk.dragging?;
    Some(match kind {
        MouseKind::Drag(MouseButton::Left) | MouseKind::Moved => track(state, drag.strip)
            .map(|t| Action::Desk(DeskAction::Slide(position_at(t, y + drag.offset)))),
        MouseKind::Up(MouseButton::Left) => Some(Action::Desk(DeskAction::Release)),
        _ => None,
    })
}

/// What `kind` does on a desk's `target`; `None` when the target is not
/// the desk's.
pub(crate) fn on(state: &State, kind: MouseKind, target: Target, y: f64) -> Option<Option<Action>> {
    use MouseKind::{Down, ScrollDown, ScrollUp};
    let steps = |kind| if kind == ScrollUp { 1 } else { -1 };
    let action = match (kind, target) {
        (Down(MouseButton::Left), Target::DeskFader(strip)) => {
            let t = track(state, strip)?;
            let at = taper::position_of(state.desk.setting(strip).db);
            // The cap taken where it is does not jump; elsewhere on the
            // track it comes to the pointer.
            let on_cap = y.floor() as u16 == cap_row(t, at);
            let offset = if on_cap { cap_y(t, at) - y } else { 0.0 };
            Some(DeskAction::Grab {
                strip,
                position: position_at(t, y + offset),
                offset,
            })
        }
        (ScrollUp | ScrollDown, Target::DeskFader(strip)) => {
            Some(DeskAction::Wheel(Channel::Strip(strip), steps(kind)))
        }
        (ScrollUp | ScrollDown, Target::DeskChannel(channel)) => {
            Some(DeskAction::Wheel(channel, steps(kind)))
        }
        (Down(MouseButton::Left), Target::DeskButton(button)) => Some(DeskAction::Press(button)),
        (Down(MouseButton::Left), Target::DeskChannel(channel)) => {
            Some(DeskAction::Select(channel))
        }
        (_, Target::DeskFader(_) | Target::DeskButton(_) | Target::DeskChannel(_)) => None,
        _ => return None,
    };
    Some(action.map(Action::Desk))
}

/// The desk's hit regions, most specific first. The echo canceller's
/// button is one only while a microphone is open (it is drawn only then).
pub(crate) fn regions(
    state: &State,
    desk: &layout::desk::DeskLayout,
    out: &mut Vec<(Rect, Target)>,
) {
    let mic = state.karaoke.mic.name.is_some();
    for c in &desk.channels {
        for &(rect, button) in &c.buttons {
            if button != DeskButton::Echo || mic {
                out.push((rect, Target::DeskButton(button)));
            }
        }
        if let (Some(t), Channel::Strip(strip)) = (c.track, c.channel) {
            out.push((t, Target::DeskFader(strip)));
        }
    }
    for c in &desk.channels {
        out.push((c.frame, Target::DeskChannel(c.channel)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cap_is_where_the_pointer_puts_it_with_a_detent_at_unity() {
        let t = Rect::new(5, 10, 3, 21);
        assert_eq!(position_at(t, 10.5), 1.0);
        assert_eq!(position_at(t, 30.5), 0.0);
        assert_eq!(position_at(t, 99.0), 0.0);
        for p in [0.0, 0.25, 0.5, 1.0] {
            assert!((position_at(t, cap_y(t, p)) - p).abs() < 1e-12);
        }
        assert_eq!(cap_row(t, 1.0), 10);
        assert_eq!(cap_row(t, taper::UNITY), 15);
        assert_eq!(cap_row(t, 0.0), 30);
        // Unity's middle is row 15's: within 0.4 rows of it, exactly 0 dB.
        let unity = cap_y(t, taper::UNITY);
        for dy in [-0.39, -0.1, 0.0, 0.2, 0.39] {
            assert_eq!(taper::db_at(position_at(t, unity + dy)), 0.0, "{dy}");
        }
        assert!(taper::db_at(position_at(t, unity + 0.41)) < 0.0);
        assert!(taper::db_at(position_at(t, unity - 0.41)) > 0.0);
    }
}
