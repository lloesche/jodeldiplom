//! The cell renderer's layout: which panes each screen has on a grid of
//! cells ([`Grid`]) and how they collapse when it gets small; and what the
//! window reports of it to the core ([`fit`]).
//!
//! Main screen, wide:
//!
//! ```text
//! ┌ the list pane ([`MainLayout::list`]): ┐╭─────────────╮
//! │ its tabs on its top edge, the browser ││   cover     │
//! │ or the queue, drawn by the window; a  │╰─────────────╯
//! │ tab of its own (Transition, Karaoke,  │ the text under
//! │ the Mixer) in a box drawn here, the   │ it ([`MainLayout::info`]),
//! │ tabs over its top border              │ drawn by the window
//! └───────────────────────────────────────┘
//! ┌ the bottom bar ([`Bottom`]), drawn by the window ┐
//! │ the spectrum and its labels                      │
//! │ 0:12 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━───────── 4:01  │
//! │ ⏮ ⏸ ⏭ ⏹   message          settings  help      │
//! └──────────────────────────────────────────────────┘
//! ```
//!
//! Collapse rules (columns × rows):
//! - below 72 columns the right column goes; a two-row summary of the
//!   track sits above the spectrum instead,
//! - the spectrum takes 10 rows from 30 rows, 7 from 24, 4 from 18 and
//!   none below,
//! - the cover needs 4 rows; on a short screen it goes before the text,
//! - below 40 × 12 only a "too small" note is shown.

mod karaoke;
mod track;
mod transition;

pub use karaoke::{
    controls_bar, controls_share, KaraokeControls, KaraokeLayout, KaraokeTabLayout, SliderLayout,
    CONTROLS_ROWS, KARAOKE_ENTER, SLIDER_LABEL, SWITCH, SYNC_BUTTON,
};
pub use track::Track;
pub use transition::{
    TransitionBlockLayout, TransitionLayout, FIT_LYRICS, TRANSITION_LANE_LABELS,
    TRANSITION_LANE_ROWS,
};

use ratatui::layout::Rect;

use crate::core::{Fit, Screen, Side, State};

/// The smallest size anything but a note is drawn at.
pub const MIN_SIZE: (u16, u16) = (40, 12);

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

    /// Too small to draw anything but a note.
    fn too_small(self) -> bool {
        self.cols < MIN_SIZE.0 || self.rows < MIN_SIZE.1
    }
}

/// The bottom bar's place: the window draws the bar over these cells
/// (the spectrum's rows at its top, where it has them, then
/// [`BAR_ROWS`] of the seek bar and the buttons).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bottom {
    /// The whole bar.
    pub area: Rect,
    /// The spectrum's rows at its top, its labels' among them (0: none).
    pub spectrum: u16,
}

/// The rows of the bar under the spectrum: the seek bar and the buttons.
pub const BAR_ROWS: u16 = 3;

/// The right column's rows for the track's text under the cover (after a
/// free row), which the window draws: the way to the device may wrap. The
/// cover gives way to them.
const INFO_ROWS: u16 = 8;

/// The main screen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MainLayout {
    /// The list pane, its tabs on its top row (the window draws them, and
    /// the browser and the queue whole).
    pub list: Rect,
    /// Inside its border.
    pub rows: Rect,
    /// The right column.
    pub right: Option<Rect>,
    /// The cover, in the right column.
    pub cover: Option<Rect>,
    /// Under the cover in the right column: the track's text, which the
    /// window draws ([`crate::view::info`]).
    pub info: Option<Rect>,
    /// The two-row track summary when there is no right column, which the
    /// window draws.
    pub mini: Option<Rect>,
    /// The Transition tab, when it is the side shown (it then takes the
    /// whole width, with no right column and no spectrum).
    pub transition: Option<TransitionLayout>,
    /// The Karaoke tab, when it is the side shown (the whole width too).
    pub karaoke: Option<KaraokeTabLayout>,
    /// The bottom bar.
    pub bottom: Bottom,
}

/// The layout of the screen shown.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Too small to draw anything else.
    TooSmall(Rect),
    /// The main screen.
    Main(Box<MainLayout>),
    /// A screen the window lays out in iced's widgets (Now playing, the
    /// settings, the keys): nothing on the cells.
    Native,
    /// Karaoke mode.
    Karaoke(KaraokeLayout),
}

impl Layout {
    /// The bottom bar, on the main screen (the screens the window lays
    /// out have theirs in their own place).
    pub fn bottom(&self) -> Option<&Bottom> {
        match self {
            Layout::Main(m) => Some(&m.bottom),
            Layout::TooSmall(_) | Layout::Native | Layout::Karaoke(_) => None,
        }
    }
}

/// The layout for `state` on `grid`.
pub fn compute(state: &State, grid: Grid) -> Layout {
    let area = grid.area();
    match state.screen {
        Screen::NowPlaying | Screen::Settings | Screen::Help => Layout::Native,
        _ if grid.too_small() => Layout::TooSmall(area),
        Screen::Main => Layout::Main(Box::new(main(state, grid))),
        Screen::Karaoke => Layout::Karaoke(karaoke::mode(state, area)),
    }
}

/// What `grid` does to the keys and the pointer, for the core
/// ([`crate::core::Action::Fit`]): measured on the layouts the screens
/// have on it, whichever screen and tab show (the list keeps its rows
/// behind the other screens, the keys go round the desk's strips from any
/// tab), the desk's width at a scale of 1 ([`CELL_WIDTH`]; the window,
/// which knows its cells, gives the one drawn, [`desk_width`]). Nothing on
/// a grid too small to draw on. The list's page is its pane's rows inside
/// its border, a row a cell; the window, which draws the list in rows of
/// its own, counts those instead ([`crate::gui::screens::list::fit`]).
pub fn fit(state: &State, grid: Grid) -> Fit {
    if grid.too_small() {
        return Fit::default();
    }
    let area = grid.area();
    let list = main_list(grid);
    // The wide tabs' pane, inside its border.
    let wide = in_border(area);
    Fit {
        list_rows: list.rows.height,
        help_rows: help_rows(state, grid, CELL_HEIGHT),
        desk_width: desk_width(grid, CELL_WIDTH),
        lane_letters: transition::lane_width(wide),
        controls_reach: f32::from(CONTROLS_ROWS + 2) / f32::from(grid.rows),
        spectrum: compute(state, grid)
            .bottom()
            .is_some_and(|b| b.spectrum > 0),
    }
}

/// A cell's width in points at a scale of 1: the built-in Sarasa Fixed's
/// at [`crate::gui::font_px`]'s size there (Cascadia Mono's, without it,
/// is 9; the window gives the desk's width at its own cells).
pub const CELL_WIDTH: f32 = 8.0;

/// A cell's height in points at a scale of 1, as [`CELL_WIDTH`].
pub const CELL_HEIGHT: f32 = 19.0;

/// The help's lines on `grid`, a cell `cell` points high: the page over
/// the bottom bar, measured in points ([`crate::view::help::rows`]; the
/// window gives them at its own cells).
pub fn help_rows(state: &State, grid: Grid, cell: f32) -> u16 {
    let page = f32::from(grid.rows.saturating_sub(BAR_ROWS)) * cell;
    crate::view::help::rows(page, state)
}

/// The Mixer tab's desk's width in points on `grid`, a cell `cell` points
/// wide: the list pane's inside, the whole width (the window draws the
/// desk there, `gui::mixer`).
pub fn desk_width(grid: Grid, cell: f32) -> f32 {
    f32::from(in_border(grid.area()).width) * cell
}

/// The browser's and the queue's pane on `grid`, whichever side shows
/// (the list keeps its page behind the other tabs).
pub fn list_pane(grid: Grid) -> Rect {
    main_list(grid).list
}

/// `rect` inside its border.
fn in_border(rect: Rect) -> Rect {
    Rect::new(
        rect.x + 1,
        rect.y + 1,
        rect.width.saturating_sub(2),
        rect.height.saturating_sub(2),
    )
}

/// Spectrum rows on the main screen for a height.
pub fn spectrum_rows(rows: u16) -> u16 {
    match rows {
        30.. => 10,
        24..=29 => 7,
        18..=23 => 4,
        _ => 0,
    }
}

/// The right column's width for a width in columns (0: none): the same
/// whatever plays, so it stays put from track to track; the text in it
/// wraps.
pub fn right_width(cols: u16) -> u16 {
    match cols {
        120.. => (cols * 34 / 100).min(60),
        96..=119 => 34,
        72..=95 => 28,
        _ => 0,
    }
}

/// `area` with the bottom bar at its bottom, `spectrum` rows of it the
/// spectrum's (none under 3): the rest above it, and the bar.
fn bottom(area: Rect, spectrum: u16) -> (Rect, Bottom) {
    let spectrum = if spectrum >= 3 { spectrum } else { 0 };
    let used = (BAR_ROWS + spectrum).min(area.height);
    let top = Rect {
        height: area.height - used,
        ..area
    };
    let bar = Rect::new(area.x, area.y + top.height, area.width, used);
    (
        top,
        Bottom {
            area: bar,
            spectrum,
        },
    )
}

/// The main screen with a wide tab (Transition, Karaoke or Mixer): the list pane's
/// border over the whole width, no right column, no spectrum (the Karaoke
/// tab has its voice's lane there).
fn main_wide(state: &State, area: Rect) -> MainLayout {
    // The Karaoke tab has the voice's lane where the spectrum goes.
    let karaoke = state.side == Side::Karaoke;
    let strip = if karaoke {
        spectrum_rows(area.height)
    } else {
        0
    };
    let (top, mut bottom) = bottom(area, strip);
    let voice = (bottom.spectrum > 0).then(|| {
        let b = bottom.area;
        Rect::new(b.x + 1, b.y, b.width.saturating_sub(2), bottom.spectrum)
    });
    bottom.area.y += bottom.spectrum;
    bottom.area.height -= bottom.spectrum;
    bottom.spectrum = 0;
    let list = top;
    let inner = in_border(list);
    MainLayout {
        list,
        rows: inner,
        right: None,
        cover: None,
        info: None,
        mini: None,
        transition: (state.side == Side::Transition)
            .then(|| transition::compute(state, inner, list)),
        karaoke: karaoke.then(|| KaraokeTabLayout {
            voice,
            ..karaoke::tab(inner, state.karaoke.sync.as_ref())
        }),
        bottom,
    }
}

/// The main screen on `grid`: a wide tab's, or the browser's or the
/// queue's.
fn main(state: &State, grid: Grid) -> MainLayout {
    if state.side.wide() {
        main_wide(state, grid.area())
    } else {
        main_list(grid)
    }
}

/// The main screen with the browser or the queue (laid out so whatever
/// the side, for [`fit`]): the list pane, the right column (or the
/// track's summary under the list where there is none) and the spectrum.
fn main_list(grid: Grid) -> MainLayout {
    let area = grid.area();
    let (top, bottom) = bottom(area, spectrum_rows(area.height));
    let rw = right_width(grid.cols);
    let (left, right) = if rw > 0 {
        (
            Rect {
                width: top.width - rw,
                ..top
            },
            Some(Rect::new(top.x + top.width - rw, top.y, rw, top.height)),
        )
    } else {
        (top, None)
    };
    // Without a right column, a summary of the track sits under the list.
    let (list, mini) = if right.is_none() && left.height >= 10 {
        (
            Rect {
                height: left.height - 2,
                ..left
            },
            Some(Rect::new(
                left.x + 1,
                left.y + left.height - 2,
                left.width - 2,
                2,
            )),
        )
    } else {
        (left, None)
    };
    let (cover, info) = match right {
        Some(r) => {
            // The cover, a free row, then the text.
            let cover_h = (r.width / 2).min(r.height.saturating_sub(INFO_ROWS + 1));
            if cover_h >= 4 {
                (
                    Some(Rect::new(r.x, r.y, r.width, cover_h)),
                    Some(Rect::new(
                        r.x + 1,
                        r.y + cover_h + 1,
                        r.width - 1,
                        r.height - cover_h - 1,
                    )),
                )
            } else {
                (None, Some(Rect::new(r.x + 1, r.y, r.width - 1, r.height)))
            }
        }
        None => (None, None),
    };
    MainLayout {
        list,
        rows: in_border(list),
        right,
        cover,
        info,
        mini,
        transition: None,
        karaoke: None,
        bottom,
    }
}

/// The largest square (in cells of twice the height of their width)
/// centred in `area`.
fn square_in(area: Rect) -> Rect {
    let width = area.width.min(area.height.saturating_mul(2));
    let height = area.height.min(width.div_ceil(2));
    Rect::new(
        area.x + (area.width - width) / 2,
        area.y + (area.height - height) / 2,
        width,
        height,
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    /// A state, and the grid it is laid out on.
    fn state(cols: u16, rows: u16) -> (State, Grid) {
        (
            State::new("/music".into(), Config::default()),
            Grid::new(cols, rows),
        )
    }

    fn inside(inner: Rect, outer: Rect) -> bool {
        inner.x >= outer.x
            && inner.y >= outer.y
            && inner.right() <= outer.right()
            && inner.bottom() <= outer.bottom()
    }

    #[test]
    fn wide_main_screen_has_every_pane_inside_the_area() {
        let (s, grid) = state(160, 50);
        let Layout::Main(m) = compute(&s, grid) else {
            panic!()
        };
        let area = Rect::new(0, 0, 160, 50);
        let right = m.right.unwrap();
        let cover = m.cover.unwrap();
        assert!(inside(m.list, area) && inside(right, area) && inside(cover, right));
        assert!(!m.list.intersects(right));
        assert_eq!(m.list.width + right.width, 160);
        assert!(m.mini.is_none());
        // The cover is about square in pixels: twice as wide as tall.
        assert_eq!(cover.height, right.width / 2);
        // The bar under the list, the whole width: 10 rows of spectrum
        // and its own.
        let b = m.bottom;
        assert_eq!(b.spectrum, 10);
        assert_eq!(b.area, Rect::new(0, 50 - 10 - BAR_ROWS, 160, 10 + BAR_ROWS));
        assert_eq!(m.list.bottom(), b.area.y);
    }

    #[test]
    fn narrow_screens_drop_the_right_column_and_spectrum() {
        let (s, grid) = state(60, 20);
        let Layout::Main(m) = compute(&s, grid) else {
            panic!()
        };
        assert!(m.right.is_none() && m.cover.is_none());
        assert_eq!(m.mini.unwrap().height, 2);
        assert_eq!(m.bottom.spectrum, 4);
        let (s, grid) = state(60, 15);
        let Layout::Main(m) = compute(&s, grid) else {
            panic!()
        };
        assert_eq!(m.bottom.spectrum, 0);
        // The list, the two-row summary and the bar fill the height.
        assert_eq!(m.list.height + 2 + BAR_ROWS, 15);
        let at = |cols, rows| {
            let (s, grid) = state(cols, rows);
            compute(&s, grid)
        };
        assert!(matches!(at(39, 30), Layout::TooSmall(_)));
        assert!(matches!(at(80, 11), Layout::TooSmall(_)));
    }

    #[test]
    fn medium_sizes_keep_a_cover_when_there_is_room() {
        let (s, grid) = state(80, 24);
        let Layout::Main(m) = compute(&s, grid) else {
            panic!()
        };
        assert_eq!(m.right.unwrap().width, 28);
        let cover = m.cover.unwrap();
        assert!(cover.height >= 4 && m.info.unwrap().height >= 7);
        let (s, grid) = state(100, 20);
        let Layout::Main(m) = compute(&s, grid) else {
            panic!()
        };
        // 20 rows: 4 of spectrum and 3 of bar leave 13; the cover gets 4,
        // then a free row and 8 of text.
        assert_eq!(m.cover.unwrap().height, 4);
        assert_eq!(m.info.unwrap().y, m.cover.unwrap().bottom() + 1);
    }
}
