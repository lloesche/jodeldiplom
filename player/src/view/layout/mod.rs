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
//! │ tab of its own (Transition, Karaoke,  │ title, artist
//! │ the Mixer) in a box drawn here, the   │ ○ PCM
//! │ tabs over its top border              │ signal path
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
    KaraokeControls, KaraokeLayout, KaraokeTabLayout, SliderLayout, KARAOKE_ENTER, SLIDER_LABEL,
    SWITCH, SYNC_BUTTON,
};
pub use track::Track;
pub use transition::{
    TransitionBlockLayout, TransitionLayout, FIT_LYRICS, TRANSITION_LANE_LABELS,
    TRANSITION_LANE_ROWS,
};

use ratatui::layout::Rect;

use crate::core::{Fit, Screen, Side, State};
use crate::view::cells::{self, width};

/// The smallest size anything but a note is drawn at.
pub const MIN_SIZE: (u16, u16) = (40, 12);

/// The window as the cell renderer has it: its size in cells, and the
/// right column's width held while one track gives way to the next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    /// Columns.
    pub cols: u16,
    /// Rows.
    pub rows: u16,
    /// The widest right-column text seen since the size changed, in cells
    /// ([`Grid::hold`]): the column keeps this width, so it does not narrow
    /// and widen again between tracks.
    pub held: usize,
}

impl Grid {
    /// A grid of `cols` × `rows` cells, nothing held yet.
    pub const fn new(cols: u16, rows: u16) -> Grid {
        Grid {
            cols,
            rows,
            held: 0,
        }
    }

    /// The whole grid.
    pub fn area(self) -> Rect {
        Rect::new(0, 0, self.cols, self.rows)
    }

    /// The grid at `cols` × `rows`: what was held goes when the size
    /// changes.
    pub fn resize(&mut self, cols: u16, rows: u16) {
        if (cols, rows) != (self.cols, self.rows) {
            *self = Grid::new(cols, rows);
        }
    }

    /// Holds the right column's width for what `state` shows now.
    pub fn hold(&mut self, state: &State) {
        self.held = self.held.max(cells::info_width(state));
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
    /// Title, status light and signal path, in the right column.
    pub info: Option<Rect>,
    /// The two-row track summary when there is no right column.
    pub mini: Option<Rect>,
    /// The Transition tab, when it is the side shown (it then takes the
    /// whole width, with no right column and no spectrum).
    pub transition: Option<TransitionLayout>,
    /// The Karaoke tab, when it is the side shown (the whole width too).
    pub karaoke: Option<KaraokeTabLayout>,
    /// The bottom bar.
    pub bottom: Bottom,
}

/// The Now playing screen: the cover filling the screen and a caption
/// (title, artist · album, status light) at its bottom left. With a cover
/// the window draws it in `cover` and the caption over it; the placeholder
/// (no cover) goes in `cover_above`, which leaves the caption's rows free.
/// The caption is in the same place either way.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NowLayout {
    /// The whole screen (a click anywhere returns).
    pub screen: Rect,
    /// The cover, as large as the screen allows (square, in cells of
    /// twice the height of their width).
    pub cover: Rect,
    /// The cover with the caption's rows free under it.
    pub cover_above: Rect,
    /// The caption: under `cover_above` at its left edge, a row above the
    /// bottom (so over `cover`'s bottom left).
    pub caption: Rect,
}

/// One settings row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SettingsLine {
    /// The whole row.
    pub rect: Rect,
    /// The `‹` that steps back.
    pub less: Rect,
    /// The value.
    pub value: Rect,
    /// The `›` that steps on.
    pub more: Rect,
}

/// The settings screen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SettingsLayout {
    /// The bordered pane.
    pub pane: Rect,
    /// One line per [`crate::core::SettingsRow`].
    pub lines: Vec<SettingsLine>,
    /// The device list, or the data folder's list, when open.
    pub devices: Option<Rect>,
    /// Notes under the rows (the config file).
    pub notes: Rect,
    /// The bottom bar (no spectrum).
    pub bottom: Bottom,
}

/// The help screen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HelpLayout {
    /// The bordered pane.
    pub pane: Rect,
    /// The text inside it.
    pub text: Rect,
    /// The search line under the text, while the help is searched (as the
    /// browser's filter line).
    pub search: Option<Rect>,
    /// The bottom bar (no spectrum).
    pub bottom: Bottom,
}

/// The layout of the screen shown.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Too small to draw anything else.
    TooSmall(Rect),
    /// The main screen.
    Main(Box<MainLayout>),
    /// Now playing.
    NowPlaying(NowLayout),
    /// Settings.
    Settings(SettingsLayout),
    /// Help.
    Help(HelpLayout),
    /// Karaoke mode.
    Karaoke(KaraokeLayout),
}

impl Layout {
    /// The bottom bar, on every screen but the note and Now playing.
    pub fn bottom(&self) -> Option<&Bottom> {
        match self {
            Layout::TooSmall(_) => None,
            Layout::Main(m) => Some(&m.bottom),
            Layout::NowPlaying(_) => None,
            Layout::Settings(s) => Some(&s.bottom),
            Layout::Help(h) => Some(&h.bottom),
            Layout::Karaoke(_) => None,
        }
    }
}

/// The layout for `state` on `grid`.
pub fn compute(state: &State, grid: Grid) -> Layout {
    let area = grid.area();
    if grid.too_small() {
        return Layout::TooSmall(area);
    }
    match state.screen {
        Screen::Main => Layout::Main(Box::new(main(state, grid))),
        Screen::NowPlaying => Layout::NowPlaying(now_playing(state, area)),
        Screen::Settings => Layout::Settings(settings(state, area)),
        Screen::Help => Layout::Help(help(state, area)),
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
    let list = main_list(state, grid);
    let help = help(state, area);
    // The wide tabs' pane, inside its border.
    let wide = in_border(area);
    Fit {
        list_rows: list.rows.height,
        help_rows: help.text.height + u16::from(help.search.is_some()),
        desk_width: desk_width(grid, CELL_WIDTH),
        lane_letters: transition::lane_width(wide),
        controls_reach: f32::from(crate::core::karaoke::controls::ROWS + 2) / f32::from(grid.rows),
        spectrum: compute(state, grid)
            .bottom()
            .is_some_and(|b| b.spectrum > 0),
    }
}

/// A cell's width in points at a scale of 1: the built-in Sarasa Fixed's
/// at [`crate::gui::font_px`]'s size there (Cascadia Mono's, without it,
/// is 9; the window gives the desk's width at its own cells).
pub const CELL_WIDTH: f32 = 8.0;

/// The Mixer tab's desk's width in points on `grid`, a cell `cell` points
/// wide: the list pane's inside, the whole width (the window draws the
/// desk there, `gui::mixer`).
pub fn desk_width(grid: Grid, cell: f32) -> f32 {
    f32::from(in_border(grid.area()).width) * cell
}

/// The browser's and the queue's pane on `grid`, whichever side shows
/// (the list keeps its page behind the other tabs).
pub fn list_pane(state: &State, grid: Grid) -> Rect {
    main_list(state, grid).list
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

/// The right column's width on `grid` for the track shown: as wide as its
/// widest line needs (so the signal path does not wrap), or the widest the
/// grid holds ([`Grid::held`]), from [`right_width`] up to half the
/// screen; 0 (none) where [`right_width`] has none.
pub fn right_width_for(state: &State, grid: Grid) -> u16 {
    let cols = grid.cols;
    let base = right_width(cols);
    if base == 0 {
        return 0;
    }
    // The text is one cell in from the column's edge.
    let widest = cells::info_width(state).max(grid.held);
    let need = (widest + 1).min(usize::from(cols)) as u16;
    need.clamp(base, base.max(cols / 2))
}

/// The right column's least width for a width (0: none).
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
        main_list(state, grid)
    }
}

/// The main screen with the browser or the queue (laid out so whatever
/// the side, for [`fit`]): the list pane, the right column (or the
/// track's summary under the list where there is none) and the spectrum.
fn main_list(state: &State, grid: Grid) -> MainLayout {
    let area = grid.area();
    let (top, bottom) = bottom(area, spectrum_rows(area.height));
    let rw = right_width_for(state, grid);
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
            // The cover, a free row, then the text (at least 7 rows).
            let text_rows = cells::info_lines(state, usize::from(r.width.saturating_sub(1)))
                .len()
                .max(7) as u16;
            let cover_h = (r.width / 2).min(r.height.saturating_sub(text_rows + 1));
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

/// The widest a caption wraps to.
fn caption_wrap(cover_above: Rect, x: u16, area: Rect) -> usize {
    let room = area.right().saturating_sub(x + 2);
    usize::from(cover_above.width.max(24).min(room).max(1))
}

fn now_playing(state: &State, area: Rect) -> NowLayout {
    let cover = square_in(area);
    // The caption's rows and a free row each side of it come off the
    // cover above it; wrapping to that cover's width may take more rows.
    let mut rows = cells::caption_lines(state, usize::from(area.width.saturating_sub(4))).len();
    let (mut cover_above, mut lines);
    loop {
        let above = Rect {
            height: area.height.saturating_sub(rows as u16 + 2),
            ..area
        };
        cover_above = square_in(above);
        let x = cover_above.x.max(area.x + 2);
        lines = cells::caption_lines(state, caption_wrap(cover_above, x, area));
        if lines.len() <= rows || cover_above.height == 0 {
            break;
        }
        rows = lines.len();
    }
    let x = cover_above.x.max(area.x + 2);
    let height = (lines.len() as u16).min(area.height.saturating_sub(1));
    let width = lines
        .iter()
        .map(|l| l.iter().map(|s| width(&s.text)).sum::<usize>())
        .max()
        .unwrap_or(0) as u16;
    let caption = Rect::new(
        x,
        area.bottom().saturating_sub(1 + height),
        width.min(area.right().saturating_sub(x)),
        height,
    );
    NowLayout {
        screen: area,
        cover,
        cover_above,
        caption,
    }
}

/// The width of a settings label column.
pub const SETTINGS_LABEL: u16 = 24;
/// The widest a settings value is shown.
pub const SETTINGS_VALUE: u16 = 56;

fn settings(state: &State, area: Rect) -> SettingsLayout {
    let (top, bottom) = bottom(area, 0);
    let pane = top;
    let inner = Rect::new(
        pane.x + 2,
        pane.y + 2,
        pane.width.saturating_sub(4),
        pane.height.saturating_sub(3),
    );
    let rows = crate::core::SettingsRow::ALL.len() as u16;
    let notes_h = 3u16;
    // A blank line between rows when they fit (with the device list's
    // room when it is open), else none.
    let listing = state.settings.picking || state.settings.folders.is_some();
    let list_h = if listing { 8 } else { 0 };
    let step = if inner.height >= rows * 2 + notes_h + list_h {
        2
    } else {
        1
    };
    let mut lines = Vec::new();
    for (i, _) in crate::core::SettingsRow::ALL.iter().enumerate() {
        let y = inner.y + i as u16 * step;
        let rect = Rect::new(inner.x, y, inner.width, 1);
        let value_x = inner.x + SETTINGS_LABEL;
        let value_w = inner
            .width
            .saturating_sub(SETTINGS_LABEL + 4)
            .min(SETTINGS_VALUE);
        lines.push(SettingsLine {
            rect,
            less: Rect::new(value_x, y, 2, 1),
            value: Rect::new(value_x + 2, y, value_w, 1),
            more: Rect::new(value_x + 2 + value_w, y, 2, 1),
        });
    }
    let rows_end = inner.y + rows * step;
    let devices = listing.then(|| {
        Rect::new(
            inner.x + 2,
            rows_end,
            inner.width.saturating_sub(2),
            inner.bottom().saturating_sub(rows_end + notes_h),
        )
    });
    let notes = Rect::new(
        inner.x,
        inner.bottom().saturating_sub(notes_h),
        inner.width,
        notes_h.min(inner.height),
    );
    SettingsLayout {
        pane,
        lines,
        devices,
        notes,
        bottom,
    }
}

fn help(state: &State, area: Rect) -> HelpLayout {
    let (top, bottom) = bottom(area, 0);
    let inner = Rect::new(
        top.x + 2,
        top.y + 1,
        top.width.saturating_sub(4),
        top.height.saturating_sub(2),
    );
    let (text, search) = if state.help_search.is_some() {
        (
            Rect {
                height: inner.height.saturating_sub(1),
                ..inner
            },
            Some(Rect::new(
                inner.x,
                inner.bottom().saturating_sub(1),
                inner.width,
                1,
            )),
        )
    } else {
        (inner, None)
    };
    HelpLayout {
        pane: top,
        text,
        search,
        bottom,
    }
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
        let (s, grid) = state(160, 48);
        let Layout::Main(m) = compute(&s, grid) else {
            panic!()
        };
        let area = Rect::new(0, 0, 160, 48);
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
        assert_eq!(b.area, Rect::new(0, 48 - 10 - BAR_ROWS, 160, 10 + BAR_ROWS));
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
        let (s, grid) = state(100, 19);
        let Layout::Main(m) = compute(&s, grid) else {
            panic!()
        };
        // 19 rows: 4 of spectrum and 3 of bar leave 12; the cover gets 4,
        // then a free row and 7 of text.
        assert_eq!(m.cover.unwrap().height, 4);
        assert_eq!(m.info.unwrap().y, m.cover.unwrap().bottom() + 1);
    }

    #[test]
    fn now_playing_is_the_cover_with_a_caption() {
        let (mut s, grid) = state(160, 48);
        s.screen = Screen::NowPlaying;
        let Layout::NowPlaying(n) = compute(&s, grid) else {
            panic!()
        };
        let screen = Rect::new(0, 0, 160, 48);
        assert_eq!(n.screen, screen);
        // The cover is as tall as the screen, square and centred.
        assert_eq!(n.cover, Rect::new(32, 0, 96, 48));
        // The caption: under the smaller cover at its left edge, a row
        // above the bottom, and so over the big cover's bottom left.
        assert!(n.cover_above.bottom() < n.caption.y);
        assert_eq!(n.caption.bottom(), 47);
        assert_eq!(n.caption.x, n.cover_above.x);
        assert!(n.caption.height >= 1 && inside(n.caption, n.cover));
        // No bottom bar.
        assert!(compute(&s, grid).bottom().is_none());
        // Tall and narrow: the cover takes the width.
        let grid = Grid::new(70, 40);
        let Layout::NowPlaying(n) = compute(&s, grid) else {
            panic!()
        };
        assert_eq!(n.cover.width, 70);
        assert!(inside(n.cover, Rect::new(0, 0, 70, 40)));
        assert!(inside(n.caption, Rect::new(0, 0, 70, 40)));
        assert!(n.cover_above.bottom() < n.caption.y);
    }

    #[test]
    fn settings_rows_and_device_list() {
        let (mut s, grid) = state(100, 30);
        s.screen = Screen::Settings;
        s.settings.picking = true;
        let Layout::Settings(l) = compute(&s, grid) else {
            panic!()
        };
        let rows = crate::core::SettingsRow::ALL.len();
        assert_eq!(l.lines.len(), rows);
        let devices = l.devices.unwrap();
        assert!(devices.y > l.lines[rows - 1].rect.y && devices.height > 4);
        for line in &l.lines {
            assert!(line.less.right() <= line.value.x && line.value.right() <= line.more.x);
        }
        // Tall enough: a blank line between rows; short: none, all inside.
        for (height, step) in [(40, 2), (24, 1)] {
            let (mut s, grid) = state(100, height);
            s.screen = Screen::Settings;
            let Layout::Settings(l) = compute(&s, grid) else {
                panic!()
            };
            assert_eq!(l.lines[1].rect.y - l.lines[0].rect.y, step, "{height} rows");
            assert!(l.lines[rows - 1].rect.y < l.notes.y, "{height} rows");
        }
    }
}
