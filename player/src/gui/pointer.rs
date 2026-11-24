//! The pointer over the old canvas ([`super::embed`]) as the core's mouse
//! ([`crate::core::mouse`]): its cell, the button held, the links under
//! it; and the wheel, for the canvas and the Mixer tab's strips alike, by
//! whole steps ([`Wheel`]).
//!
//! The canvas's widget only forwards what the pointer does over it; this is
//! the one owner of what is held. A button is let go of wherever the
//! pointer is (the window hears every release, whichever widget it was
//! over), and a release with nothing held is dropped.

use iced::mouse::ScrollDelta;

use crate::app::App;
use crate::core::desk::Channel;
use crate::core::mouse::{self, MouseButton, MouseInput, MouseKind};
use crate::core::Action;

/// What the pointer did over the old canvas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PointerKind {
    /// It moved (or left the canvas, or went under something iced draws
    /// over it: `over` false).
    Moved,
    /// A button went down over it.
    Down(MouseButton),
    /// The wheel turned over it.
    Wheel(ScrollDelta),
}

/// The pointer over the old canvas: what, where in cells from its top left
/// (with the share of one), and whether it is over it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pointer {
    /// What.
    pub kind: PointerKind,
    /// Columns.
    pub x: f64,
    /// Rows.
    pub y: f64,
    /// Over the canvas (not off it, nor under the window's own widgets).
    pub over: bool,
}

impl Pointer {
    /// Its cell, within `size` cells.
    fn cell(&self, (cols, rows): (u16, u16)) -> (u16, u16) {
        (
            (self.x.max(0.0) as u16).min(cols.saturating_sub(1)),
            (self.y.max(0.0) as u16).min(rows.saturating_sub(1)),
        )
    }
}

/// Where the pointer was, and what it holds.
#[derive(Debug, Default)]
pub struct PointerState {
    /// Its cell.
    cell: (u16, u16),
    /// Its column, with the share of the cell (a Karaoke slider follows it
    /// to the pixel).
    x: f64,
    /// Its cell while it is over the canvas (links underline under it).
    inside: Option<(u16, u16)>,
    held: Option<MouseButton>,
}

impl PointerState {
    /// The pointer's cell, while it is over the canvas.
    pub fn inside(&self) -> Option<(u16, u16)> {
        self.inside
    }

    /// The pointer moved, or went down over the canvas (`shift` held: the
    /// shifted action); whether the frame is to be drawn again.
    pub fn take(&mut self, app: &mut App, pointer: Pointer, shift: bool) -> bool {
        let cell = pointer.cell(app.state.size);
        let x = pointer.x.min(f64::from(app.state.size.0));
        match pointer.kind {
            PointerKind::Moved => {
                let moved = x != self.x;
                self.x = x;
                let inside = pointer.over.then_some(cell);
                let mut dirty = inside != self.inside && self.hover(app, inside);
                // A Karaoke slider held follows the pointer within a cell.
                let sliding =
                    self.held == Some(MouseButton::Left) && app.state.karaoke.dragging.is_some();
                if cell == self.cell {
                    if moved && sliding {
                        dirty |= self.mouse(app, MouseKind::Drag(MouseButton::Left), shift);
                    }
                    return dirty;
                }
                self.cell = cell;
                if let Some(button) = self.held {
                    // A held button drags wherever the pointer is.
                    dirty |= self.mouse(app, MouseKind::Drag(button), shift);
                } else if pointer.over
                    && (crate::core::karaoke::in_mode(&app.state)
                        || crate::core::preview::held(&app.state))
                {
                    // Karaoke mode shows its controls while it moves; a
                    // preview selects the row under it.
                    dirty |= self.mouse(app, MouseKind::Moved, shift);
                }
                dirty
            }
            PointerKind::Down(button) => {
                (self.cell, self.x) = (cell, x);
                self.held = Some(button);
                self.mouse(app, MouseKind::Down(button), shift)
            }
            // The window turns the wheel into steps ([`PointerState::scroll`]).
            PointerKind::Wheel(_) => false,
        }
    }

    /// A button let go of, wherever the pointer is: dropped unless it is
    /// the one held.
    pub fn up(&mut self, app: &mut App, button: MouseButton, shift: bool) -> bool {
        self.release(button) && self.mouse(app, MouseKind::Up(button), shift)
    }

    /// `button` let go of: whether it was the one held.
    fn release(&mut self, button: MouseButton) -> bool {
        let held = self.held == Some(button);
        if held {
            self.held = None;
        }
        held
    }

    /// The wheel `steps` whole lines up (negative: down) at `pointer`.
    pub fn scroll(&mut self, app: &mut App, pointer: Pointer, steps: i32, shift: bool) -> bool {
        (self.cell, self.x) = (pointer.cell(app.state.size), pointer.x);
        let kind = if steps > 0 {
            MouseKind::ScrollUp
        } else {
            MouseKind::ScrollDown
        };
        (0..steps.unsigned_abs()).fold(false, |dirty, _| self.mouse(app, kind, shift) | dirty)
    }

    /// What `kind` at the pointer does, carried out (the clock first);
    /// whether it did anything.
    fn mouse(&self, app: &mut App, kind: MouseKind, shift: bool) -> bool {
        let input = MouseInput {
            kind,
            col: self.cell.0,
            row: self.cell.1,
        };
        let mut action = mouse::action_at(&app.state, input, self.x);
        if shift {
            action = action.map(mouse::shifted);
        }
        let Some(action) = action else {
            return false;
        };
        app.dispatch(Action::Tick(app.now_ms()));
        app.dispatch(action);
        true
    }

    /// Moves the pointer to `inside` (none: off the canvas); whether it
    /// entered or left a link.
    fn hover(&mut self, app: &App, inside: Option<(u16, u16)>) -> bool {
        let link = |at: Option<(u16, u16)>| {
            at.and_then(|(col, row)| match mouse::hit(&app.state, col, row) {
                Some(mouse::Target::Link(op)) => Some(op),
                _ => None,
            })
        };
        let before = link(self.inside);
        self.inside = inside;
        link(self.inside) != before
    }
}

/// What the wheel turns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelOn {
    /// The old canvas, line by line.
    Canvas,
    /// A strip of the Mixer tab: its fader, a step a line.
    Strip(Channel),
}

/// The wheel's turn not yet a whole step, for what it turns (a touchpad's
/// small pixel deltas add up; another target starts afresh).
#[derive(Debug, Default)]
pub struct Wheel {
    lines: f64,
    on: Option<WheelOn>,
}

impl Wheel {
    /// `delta` over `on`, a line being `line_px` pixels: the whole steps it
    /// makes now (up positive).
    pub fn turn(&mut self, on: WheelOn, delta: ScrollDelta, line_px: f32) -> i32 {
        if self.on != Some(on) {
            self.on = Some(on);
            self.lines = 0.0;
        }
        self.lines += match delta {
            ScrollDelta::Lines { y, .. } => f64::from(y),
            ScrollDelta::Pixels { y, .. } => f64::from(y / line_px.max(1.0)),
        };
        let whole = self.lines.trunc();
        self.lines -= whole;
        whole as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The button held is let go of once, by its own release, wherever it
    /// comes from; a release with nothing (or another button) held is
    /// dropped.
    #[test]
    fn only_the_button_held_is_let_go_of_and_only_once() {
        let mut pointer = PointerState::default();
        assert!(!pointer.release(MouseButton::Left));
        pointer.held = Some(MouseButton::Left);
        assert!(!pointer.release(MouseButton::Right));
        assert_eq!(pointer.held, Some(MouseButton::Left));
        assert!(pointer.release(MouseButton::Left));
        assert!(!pointer.release(MouseButton::Left));
        assert_eq!(pointer.held, None);
    }

    #[test]
    fn small_pixel_turns_add_up_to_a_step_and_another_target_starts_afresh() {
        let mut wheel = Wheel::default();
        let px = |y| ScrollDelta::Pixels { x: 0.0, y };
        let strip = WheelOn::Strip(Channel::Main);
        // Five touchpad deltas of 4 pixels, a line being 18: one step.
        let steps: Vec<i32> = (0..5).map(|_| wheel.turn(strip, px(4.0), 18.0)).collect();
        assert_eq!(steps, [0, 0, 0, 0, 1]);
        // What was left does not carry over to the canvas.
        assert_eq!(wheel.turn(WheelOn::Canvas, px(15.0), 18.0), 0);
        assert_eq!(wheel.turn(WheelOn::Canvas, px(4.0), 18.0), 1);
        // A mouse's notches are lines; down is negative.
        let lines = ScrollDelta::Lines { x: 0.0, y: -2.0 };
        assert_eq!(wheel.turn(WheelOn::Strip(Channel::Mic), lines, 18.0), -2);
    }
}
