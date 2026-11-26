//! The cell renderer's grid ([`Grid`]): the old canvas's size in cells.
//! No screen is laid out on it any more: the window lays every one out in
//! points ([`crate::gui::metrics`]).

use ratatui::layout::Rect;

/// The window as the cell renderer has it: its size in cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    /// Columns.
    pub cols: u16,
    /// Rows.
    pub rows: u16,
}

impl Grid {
    /// A grid of `cols` × `rows` cells.
    pub const fn new(cols: u16, rows: u16) -> Grid {
        Grid { cols, rows }
    }

    /// The whole grid.
    pub fn area(self) -> Rect {
        Rect::new(0, 0, self.cols, self.rows)
    }
}
