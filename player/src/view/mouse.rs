//! The cell renderer's mouse: the named rects of its layout ([`layout`])
//! and what a click, drag or wheel on each does, as the core's
//! [`Action`]s. The window reports the pointer in cells (and within its
//! cell, [`action_at`]). The bottom bar, the list pane with its tabs, the
//! right column's text under the cover (or the summary under the list),
//! the Transition tab and the command palette are the window's own: their
//! cells take no clicks here.

use ratatui::layout::{Position, Rect};

use crate::core::karaoke;
use crate::core::keymap::Op;
use crate::core::sliders::Slider;
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
    /// The pointer moved.
    Moved,
    /// The wheel turned up.
    ScrollUp,
    /// The wheel turned down.
    ScrollDown,
    /// The pointer went out of the window.
    Out,
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
    /// Karaoke mode's seek bar.
    Bar,
    /// A button.
    Button(Op),
    /// A Karaoke slider's row.
    KaraokeSlider(Slider),
}

/// The share of the window's height below the pointer in `row` of
/// `rows`, as the core hears where the pointer is
/// ([`Action::PointerMoved`]): under the row's bottom edge, so a row is in
/// a share from the bottom where any of it is.
pub fn from_bottom(rows: u16, row: u16) -> f32 {
    f32::from(rows.saturating_sub(row + 1)) / f32::from(rows.max(1))
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
            if let (Some(k), None) = (&m.karaoke, &state.palette) {
                for &(rect, op) in &k.panel_buttons {
                    out.push((rect, Target::Button(op)));
                }
                out.push((k.enter, Target::Button(Op::KaraokeEnter)));
                for s in &k.sliders {
                    out.push((s.row, Target::KaraokeSlider(s.slider)));
                }
                out.push((k.game, Target::Button(Op::KaraokeGame)));
                out.push((k.show_notes, Target::Button(Op::KaraokeNotes)));
                out.push((k.keep, Target::Button(Op::KaraokeKeep)));
                out.push((k.sync, Target::Button(Op::KaraokeSync)));
            }
        }
        // The window's own widgets take the pointer there.
        Layout::Native => {}
        Layout::Karaoke(k) => {
            if let Some(c) = &k.controls {
                for s in &c.sliders {
                    out.push((s.row, Target::KaraokeSlider(s.slider)));
                }
                out.push((c.seek, Target::Bar));
            }
        }
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

/// The share of Karaoke mode's seek bar at column `col`.
fn bar_fraction(state: &State, grid: Grid, col: u16) -> Option<f64> {
    let Layout::Karaoke(k) = layout::compute(state, grid) else {
        return None;
    };
    let bar = k.controls?.seek;
    if bar.width < 2 {
        return None;
    }
    let x = col.clamp(bar.x, bar.right() - 1) - bar.x;
    Some(f64::from(x) / f64::from(bar.width - 1))
}

/// The share of `slider`'s range with the pointer at `x` cells from the
/// left edge. The middle of the track's first cell is 0 and of its last
/// cell 1, where the knob is drawn for those shares.
fn slider_share(state: &State, grid: Grid, slider: Slider, x: f64) -> Option<f64> {
    let sliders = match layout::compute(state, grid) {
        Layout::Main(m) => m.karaoke?.sliders,
        Layout::Karaoke(k) => k.controls?.sliders,
        _ => return None,
    };
    let track = sliders.into_iter().find(|s| s.slider == slider)?.track;
    layout::Track::columns(track).share_at(x)
}

/// What a mouse event on `grid` does, if anything. The pointer is taken
/// to be in the middle of its cell, over the cells alone.
pub fn action(state: &State, grid: Grid, input: MouseInput) -> Option<Action> {
    action_at(state, grid, input, f64::from(input.col) + 0.5, None)
}

/// [`action`] with the pointer known within its cell and over what the
/// window draws in pixels: `x` is in cells from the left edge (the GUI
/// knows it to the pixel), and `over` what the button drawn in pixels
/// under the pointer does, if there is one (Karaoke mode's START pill, on
/// the stage), which is hit before the cells under it. Only a Karaoke
/// slider uses `x`: cell by cell, a 40-cell track moves 2.6 % a step
/// (5.1 % for the microphone's 200 %); pixel by pixel, every whole percent
/// is in reach.
pub fn action_at(
    state: &State,
    grid: Grid,
    input: MouseInput,
    x: f64,
    over: Option<Op>,
) -> Option<Action> {
    use MouseButton::Left;
    use MouseKind::*;
    // A drag on Karaoke mode's seek bar follows the pointer anywhere.
    if state.seek_drag.is_some() {
        return match input.kind {
            Drag(Left) | Moved => bar_fraction(state, grid, input.col).map(Action::SeekDrag),
            Up(Left) => bar_fraction(state, grid, input.col).map(Action::SeekTo),
            _ => None,
        };
    }
    // So does a Karaoke slider held.
    if state.karaoke.dragging.is_some() {
        return match input.kind {
            Drag(Left) | Moved => state
                .karaoke
                .dragging
                .and_then(|s| slider_share(state, grid, s, x).map(|share| (s, share)))
                .map(|(slider, share)| Action::KaraokeSlide { slider, share }),
            Up(Left) => Some(Action::KaraokeSlideEnd),
            _ => None,
        };
    }
    // While a sync shows, only its panel's buttons take a click.
    if state.karaoke.sync.is_some() {
        return match (input.kind, hit(state, grid, input.col, input.row)) {
            (Down(Left), Some(Target::Button(op @ (Op::SyncKeep | Op::SyncAgain | Op::Back)))) => {
                Some(Action::Op(op))
            }
            _ => None,
        };
    }
    // Moving over Karaoke mode (a button held or not) shows the pointer,
    // and over its bottom its controls; off the window they go.
    if karaoke::in_mode(state) {
        match input.kind {
            Moved | Drag(_) => {
                return Some(Action::PointerMoved {
                    from_bottom: from_bottom(grid.rows, input.row),
                })
            }
            Out => return Some(Action::PointerLeft),
            _ => {}
        }
    }
    let target = over
        .map(Target::Button)
        .or_else(|| hit(state, grid, input.col, input.row))?;
    match (input.kind, target) {
        (Down(Left), Target::KaraokeSlider(slider)) => {
            slider_share(state, grid, slider, x).map(|share| Action::KaraokeSlide { slider, share })
        }
        (ScrollUp | ScrollDown, Target::KaraokeSlider(slider)) => Some(Action::KaraokeStep {
            slider,
            by: if input.kind == ScrollUp { 1 } else { -1 },
        }),
        (ScrollUp | ScrollDown, _) => scrolled(state, input.kind == ScrollUp),
        (Down(Left), Target::Bar) => bar_fraction(state, grid, input.col).map(Action::SeekDrag),
        (Down(Left), Target::Cover) => Some(Action::Op(Op::NowPlaying)),
        (Down(Left), Target::Button(op)) => Some(Action::Op(op)),
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

/// What a click does with shift held: a few buttons have a second meaning
/// (Enter Karaoke Mode: enter where the song is); the rest are the same.
pub fn shifted(action: Action) -> Action {
    match action {
        Action::Op(Op::KaraokeEnter) => Action::Op(Op::KaraokeEnterHere),
        action => action,
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
