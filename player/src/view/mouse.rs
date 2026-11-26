//! The cell renderer's mouse: the named rects of its layout ([`layout`])
//! and what a click or wheel on each does, as the core's [`Action`]s. The
//! window reports the pointer in cells. The bottom bar, the list pane
//! with its tabs, the right column's text under the cover (or the summary
//! under the list), the Transition and Karaoke tabs, Karaoke mode and the
//! command palette are the window's own: their cells take no clicks here.

use ratatui::layout::{Position, Rect};

use crate::core::keymap::Op;
use crate::core::{Action, Screen, State};
use crate::view::layout::{self, Grid, Layout};

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

/// A named place on the screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// The cover (on Now playing, the whole screen).
    Cover,
}

/// Every hit region of the screen shown on `grid`, most specific first.
pub fn regions(state: &State, grid: Grid) -> Vec<(Rect, Target)> {
    let mut out = Vec::new();
    match layout::compute(state, grid) {
        Layout::TooSmall(_) => {}
        Layout::Main(m) => {
            if let Some(cover) = m.cover {
                out.push((cover, Target::Cover));
            }
        }
        // The window's own widgets take the pointer there.
        Layout::Native => {}
    }
    out
}

/// The target under a cell of `grid`.
pub fn hit(state: &State, grid: Grid, col: u16, row: u16) -> Option<Target> {
    let at = Position::new(col, row);
    regions(state, grid)
        .into_iter()
        .find(|(rect, _)| rect.contains(at))
        .map(|(_, target)| target)
}

/// What a mouse event on `grid` does, if anything.
pub fn action(state: &State, grid: Grid, input: MouseInput) -> Option<Action> {
    use MouseButton::Left;
    use MouseKind::*;
    let target = hit(state, grid, input.col, input.row)?;
    match (input.kind, target) {
        (ScrollUp | ScrollDown, _) => scrolled(state, input.kind == ScrollUp),
        (Down(Left), Target::Cover) => Some(Action::Op(Op::NowPlaying)),
        _ => None,
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::browser::{Entry, EntryKind};
    use crate::config::Config;

    /// The window the tests have.
    const GRID: Grid = Grid::new(120, 40);

    fn state() -> State {
        let mut s = State::new("/m".into(), Config::default());
        s.fit = layout::fit(&s, GRID);
        s.browser.loading = false;
        s.browser.entries = (0..50)
            .map(|i| Entry {
                name: format!("{i:02}.flac"),
                path: format!("/m/{i:02}.flac").into(),
                kind: EntryKind::File,
                relative: None,
            })
            .collect();
        s
    }

    fn down(col: u16, row: u16) -> MouseInput {
        MouseInput {
            kind: MouseKind::Down(MouseButton::Left),
            col,
            row,
        }
    }

    /// The cover takes a click; the list pane (its tabs, its rows) and the
    /// bottom bar are the window's: their cells take none here.
    #[test]
    fn the_cover_is_hit_and_the_list_and_the_bar_are_the_windows() {
        let s = state();
        let Layout::Main(m) = layout::compute(&s, GRID) else {
            panic!()
        };
        assert_eq!(hit(&s, GRID, m.rows.x + 2, m.rows.y), None);
        assert_eq!(hit(&s, GRID, m.list.x + 2, m.list.y), None);
        let bar = m.bottom.area;
        assert_eq!(hit(&s, GRID, bar.x + 2, bar.bottom() - 1), None);
        assert!(matches!(
            action(
                &s,
                GRID,
                down(m.cover.unwrap().x + 3, m.cover.unwrap().y + 1)
            ),
            Some(Action::Op(Op::NowPlaying))
        ));
    }
}
