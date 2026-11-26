//! The cell renderer's mouse: the pointer over the old canvas in cells,
//! and what the wheel does where nothing under it takes it
//! ([`scrolled`]). Every screen is the window's own widgets now: no cell
//! takes a click ([`action`]).

use crate::core::{Action, Screen, State};
use crate::view::layout::Grid;

/// A mouse button.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MouseButton {
    /// The main button.
    Left,
    /// The secondary button.
    Right,
    /// The wheel button.
    Middle,
}

/// What the mouse did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MouseKind {
    /// A button went down.
    Down(MouseButton),
    /// A button went up.
    Up(MouseButton),
    /// The pointer moved with a button held.
    Drag(MouseButton),
    /// The wheel turned up.
    ScrollUp,
    /// The wheel turned down.
    ScrollDown,
}

/// A mouse event at a cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MouseInput {
    /// What happened.
    pub kind: MouseKind,
    /// The cell column.
    pub col: u16,
    /// The cell row.
    pub row: u16,
}

/// What a mouse event on `grid` does: nothing, as no place on the cells
/// takes the pointer any more.
pub fn action(_state: &State, _grid: Grid, _input: MouseInput) -> Option<Action> {
    None
}

/// What a step of the wheel `up` (or down) over the screen does where
/// nothing under it takes it: the help's text or the list scroll by three
/// lines.
pub fn scrolled(state: &State, up: bool) -> Option<Action> {
    let step = if up { -3 } else { 3 };
    match state.screen {
        Screen::Help => Some(Action::HelpScroll(step)),
        Screen::Main if state.palette.is_none() => Some(Action::Scroll(step)),
        _ => None,
    }
}
