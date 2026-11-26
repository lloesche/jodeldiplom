//! The pointer: what a widget's press holds ([`Grab`]), followed wherever
//! the pointer goes; the pointer over the old canvas ([`super::embed`]) as
//! the cell renderer's mouse ([`crate::view::mouse`]): its cell on the
//! canvas's grid, the button drawn in pixels under it (the stage's START
//! pill: [`crate::view::mouse::action_at`]'s `over`), the button held,
//! the links under it, and it going out of the window; and the wheel, for
//! the canvas and the widgets alike, by whole steps ([`Wheel`]).
//!
//! The widgets only say what the pointer does over them; this is the one
//! owner of what is held, a grab or the canvas's button, never both. A
//! button is let go of wherever the pointer is (the window hears every
//! release, whichever widget it was over), and a release with nothing held
//! is dropped.

use iced::mouse::ScrollDelta;
use iced::{Point, Rectangle};

use crate::app::App;
use crate::core::desk::Channel;
use crate::core::keymap::Op;
use crate::core::{Action, State};
use crate::view::layout::Grid;
use crate::view::mouse::{self, MouseButton, MouseInput, MouseKind, Target};

/// What a widget's press holds until its button is let go of: the
/// pointer's moves anywhere in the window go to it, and its release lands
/// it, each turned into the core's actions by what it holds. Its bounds
/// are in the window's points, as the widget was laid out when pressed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Grab {
    /// The seek bar's track: the share across it under the pointer is
    /// where a drag would go, sought on release.
    Seek(Rectangle),
    /// A queue row (its click told apart, [`Action::ClickRow`]): the
    /// track goes to the row under the pointer as it moves. `rows` is
    /// where the rows in view are, the first at its top, each `row`
    /// points high.
    QueueRow {
        /// The rows in view.
        rows: Rectangle,
        /// A row's height.
        row: f32,
    },
}

impl Grab {
    /// The button that holds it.
    fn button(self) -> MouseButton {
        match self {
            Grab::Seek(_) | Grab::QueueRow { .. } => MouseButton::Left,
        }
    }

    /// What the press at `at` does (a queue row's click is told by the
    /// row itself).
    fn pressed(self, at: Point, state: &State) -> Option<Action> {
        match self {
            Grab::Seek(_) => self.moved(at, state),
            Grab::QueueRow { .. } => None,
        }
    }

    /// What the pointer at `at` does while it holds.
    pub fn moved(self, at: Point, state: &State) -> Option<Action> {
        match self {
            Grab::Seek(track) => Some(Action::SeekDrag(share_across(track, at))),
            Grab::QueueRow { rows, row } => {
                if !rows.contains(at) || row <= 0.0 {
                    return None;
                }
                let index = state.queue.scroll + ((at.y - rows.y) / row) as usize;
                (index < state.queue.paths.len()).then_some(Action::QueueDrag(index))
            }
        }
    }

    /// What letting go at `at` does.
    fn released(self, at: Point) -> Action {
        match self {
            Grab::Seek(track) => Action::SeekTo(share_across(track, at)),
            Grab::QueueRow { .. } => Action::QueueDragEnd,
        }
    }

    /// Whether the core took the press (with nothing to seek, no drag
    /// starts; a row is held on the queue's side): it holds on only then.
    fn taken(self, state: &State) -> bool {
        match self {
            Grab::Seek(_) => state.seek_drag.is_some(),
            Grab::QueueRow { .. } => state.queue.dragging,
        }
    }
}

/// The share of `bounds`'s width left of `at` (0 to 1).
fn share_across(bounds: Rectangle, at: Point) -> f64 {
    if bounds.width <= 0.0 {
        return 0.0;
    }
    f64::from(((at.x - bounds.x) / bounds.width).clamp(0.0, 1.0))
}

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
    /// What the button drawn in pixels under it does (Karaoke mode's START
    /// pill), as the stage drew it last.
    pub on: Option<Op>,
}

impl Pointer {
    /// Its cell on `grid`.
    fn cell(&self, Grid { cols, rows, .. }: Grid) -> (u16, u16) {
        (
            (self.x.max(0.0) as u16).min(cols.saturating_sub(1)),
            (self.y.max(0.0) as u16).min(rows.saturating_sub(1)),
        )
    }
}

/// Where the pointer was, and what it holds.
#[derive(Debug, Default)]
pub struct PointerState {
    /// What a widget's press holds, and where the pointer was last seen
    /// while it does (in the window's points).
    grab: Option<(Grab, Point)>,
    /// Its cell.
    cell: (u16, u16),
    /// Its column, with the share of the cell (a Karaoke slider follows it
    /// to the pixel).
    x: f64,
    /// Its cell while it is over the canvas (links underline under it).
    inside: Option<(u16, u16)>,
    /// What the button drawn in pixels under it does.
    on: Option<Op>,
    held: Option<MouseButton>,
}

impl PointerState {
    /// Whether a widget's press holds the pointer.
    pub fn grabbed(&self) -> bool {
        self.grab.is_some()
    }

    /// A widget's press at `at` holds `grab` (unless the canvas holds a
    /// button already): what it does there carried out, and held on to if
    /// the core took it. Whether it did anything.
    pub fn grab(&mut self, app: &mut App, grab: Grab, at: Point) -> bool {
        if self.held.is_some() || self.grab.is_some() {
            return false;
        }
        if let Some(action) = grab.pressed(at, &app.state) {
            dispatch(app, action);
        }
        if grab.taken(&app.state) {
            self.grab = Some((grab, at));
        }
        true
    }

    /// The pointer moved to `at` in the window: what the grab held does
    /// there. Whether it did anything.
    pub fn moved(&mut self, app: &mut App, at: Point) -> bool {
        let Some((grab, last)) = &mut self.grab else {
            return false;
        };
        if *last == at {
            return false;
        }
        *last = at;
        let Some(action) = grab.moved(at, &app.state) else {
            return false;
        };
        dispatch(app, action);
        true
    }

    /// The pointer's cell, while it is over the canvas.
    pub fn inside(&self) -> Option<(u16, u16)> {
        self.inside
    }

    /// The pointer moved, or went down over the canvas laid out on `grid`
    /// (`shift` held: the shifted action); whether the frame is to be
    /// drawn again.
    pub fn take(&mut self, app: &mut App, grid: Grid, pointer: Pointer, shift: bool) -> bool {
        let cell = pointer.cell(grid);
        let x = pointer.x.min(f64::from(grid.cols));
        self.on = pointer.on;
        match pointer.kind {
            PointerKind::Moved => {
                let moved = x != self.x;
                self.x = x;
                let inside = pointer.over.then_some(cell);
                let out = inside.is_none() && self.inside.is_some();
                let mut dirty = inside != self.inside && self.hover(app, grid, inside);
                if out {
                    dirty |= self.mouse(app, grid, MouseKind::Out, shift);
                }
                // A Karaoke slider held follows the pointer within a cell.
                let sliding =
                    self.held == Some(MouseButton::Left) && app.state.karaoke.dragging.is_some();
                let karaoke = pointer.over && crate::core::karaoke::in_mode(&app.state);
                if cell == self.cell {
                    if moved && sliding {
                        dirty |= self.mouse(app, grid, MouseKind::Drag(MouseButton::Left), shift);
                    } else if karaoke && self.held.is_none() {
                        // Karaoke mode's pointer shows again for a move
                        // within a cell too; nothing on the canvas changes.
                        self.mouse(app, grid, MouseKind::Moved, shift);
                    }
                    return dirty;
                }
                self.cell = cell;
                if let Some(button) = self.held {
                    // A held button drags wherever the pointer is.
                    dirty |= self.mouse(app, grid, MouseKind::Drag(button), shift);
                } else if karaoke {
                    // Karaoke mode shows its pointer, and its controls near
                    // the bottom, while it moves.
                    dirty |= self.mouse(app, grid, MouseKind::Moved, shift);
                }
                dirty
            }
            // A grab owns the pointer until it is let go of.
            PointerKind::Down(_) if self.grab.is_some() => false,
            PointerKind::Down(button) => {
                (self.cell, self.x) = (cell, x);
                self.held = Some(button);
                self.mouse(app, grid, MouseKind::Down(button), shift)
            }
            // The window turns the wheel into steps ([`PointerState::scroll`]).
            PointerKind::Wheel(_) => false,
        }
    }

    /// A button let go of, wherever the pointer is: the grab's lands
    /// where the pointer was last seen; the canvas's goes to it; any other
    /// is dropped.
    pub fn up(&mut self, app: &mut App, grid: Grid, button: MouseButton, shift: bool) -> bool {
        match self.grab {
            Some((grab, at)) if grab.button() == button => {
                self.grab = None;
                dispatch(app, grab.released(at));
                true
            }
            Some(_) => false,
            None => self.release(button) && self.mouse(app, grid, MouseKind::Up(button), shift),
        }
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
    pub fn scroll(
        &mut self,
        app: &mut App,
        grid: Grid,
        pointer: Pointer,
        steps: i32,
        shift: bool,
    ) -> bool {
        (self.cell, self.x) = (pointer.cell(grid), pointer.x);
        self.on = pointer.on;
        let kind = if steps > 0 {
            MouseKind::ScrollUp
        } else {
            MouseKind::ScrollDown
        };
        (0..steps.unsigned_abs()).fold(false, |dirty, _| self.mouse(app, grid, kind, shift) | dirty)
    }

    /// What `kind` at the pointer does, carried out (the clock first);
    /// whether it did anything.
    fn mouse(&self, app: &mut App, grid: Grid, kind: MouseKind, shift: bool) -> bool {
        let input = MouseInput {
            kind,
            col: self.cell.0,
            row: self.cell.1,
        };
        let mut action = mouse::action_at(&app.state, grid, input, self.x, self.on);
        if shift {
            action = action.map(mouse::shifted);
        }
        let Some(action) = action else {
            return false;
        };
        dispatch(app, action);
        true
    }

    /// Moves the pointer to `inside` (none: off the canvas); whether it
    /// entered or left a link.
    fn hover(&mut self, app: &App, grid: Grid, inside: Option<(u16, u16)>) -> bool {
        let link = |at: Option<(u16, u16)>| {
            at.and_then(|(col, row)| match mouse::hit(&app.state, grid, col, row) {
                Some(Target::Link(op)) => Some(op),
                _ => None,
            })
        };
        let before = link(self.inside);
        self.inside = inside;
        link(self.inside) != before
    }
}

/// `action` carried out, the clock first.
fn dispatch(app: &mut App, action: Action) {
    app.dispatch(Action::Tick(app.now_ms()));
    app.dispatch(action);
}

/// What the wheel turns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelOn {
    /// The old canvas, line by line, and the widgets over it that scroll
    /// what it shows (the bottom bar).
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

    /// A seek grab reads the share across its track wherever the pointer
    /// is: before the track 0, past it 1; the drag as it moves, the seek
    /// as it is let go of.
    #[test]
    fn a_seek_grab_follows_the_pointer_anywhere() {
        let grab = Grab::Seek(Rectangle::new(
            Point::new(100.0, 500.0),
            iced::Size::new(400.0, 16.0),
        ));
        let at = |x, y| Point::new(x, y);
        let state = State::new("/m".into(), crate::config::Config::default());
        let moved = |p| grab.moved(p, &state);
        assert!(matches!(moved(at(300.0, 508.0)), Some(Action::SeekDrag(f)) if f == 0.5));
        assert!(matches!(moved(at(20.0, 0.0)), Some(Action::SeekDrag(f)) if f == 0.0));
        assert!(matches!(moved(at(900.0, 900.0)), Some(Action::SeekDrag(f)) if f == 1.0));
        assert!(matches!(grab.released(at(200.0, 10.0)), Action::SeekTo(f) if f == 0.25));
        assert_eq!(grab.button(), MouseButton::Left);
    }

    /// A queue row held goes to the row in view under the pointer, scrolled
    /// or not; off the rows, or past the queue's end, it stays; the
    /// release ends the drag.
    #[test]
    fn a_queue_row_grab_moves_the_track_to_the_row_under_the_pointer() {
        let grab = Grab::QueueRow {
            rows: Rectangle::new(Point::new(10.0, 100.0), iced::Size::new(300.0, 340.0)),
            row: 34.0,
        };
        let mut state = State::new("/m".into(), crate::config::Config::default());
        state.queue.paths = (0..30).map(|i| format!("/q/{i}.flac").into()).collect();
        let at = |x, y| Point::new(x, y);
        assert!(matches!(
            grab.moved(at(50.0, 100.0 + 34.0 * 3.5), &state),
            Some(Action::QueueDrag(3))
        ));
        state.queue.scroll = 12;
        assert!(matches!(
            grab.moved(at(50.0, 101.0), &state),
            Some(Action::QueueDrag(12))
        ));
        assert!(grab.moved(at(400.0, 120.0), &state).is_none());
        assert!(grab.moved(at(50.0, 90.0), &state).is_none());
        state.queue.paths.truncate(14);
        assert!(grab.moved(at(50.0, 100.0 + 34.0 * 5.0), &state).is_none());
        assert!(matches!(grab.released(at(0.0, 0.0)), Action::QueueDragEnd));
        assert!(grab.pressed(at(50.0, 120.0), &state).is_none());
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
