//! The Karaoke tab's layout in cells (where its cover and the voice's
//! lane go) and Karaoke mode's (the whole screen, and its controls' bar
//! while they show).

use ratatui::layout::Rect;

use crate::core::karaoke;
use crate::core::sliders::Slider;
use crate::core::State;

/// A Karaoke slider's cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliderLayout {
    /// Which.
    pub slider: Slider,
    /// Its row: the label, the track and the value.
    pub row: Rect,
    /// The track its knob moves along (a click or drag sets it).
    pub track: Rect,
    /// The value, right of the track.
    pub value: Rect,
}

/// The Karaoke tab: where its pictures go (the window draws the rest of
/// it in its own widgets over the pane, beside the cover).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KaraokeTabLayout {
    /// The playing track's cover, at the pane's left: drawn at its top
    /// left, half a cell down.
    pub cover: Rect,
    /// The voice's lane with the words on it, under the pane (where the
    /// other tabs have the spectrum).
    pub voice: Option<Rect>,
}

/// Karaoke mode: the whole screen is its picture, and its controls (while
/// they show) a bar over its bottom (the picture stays where it is).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KaraokeLayout {
    /// The whole screen.
    pub screen: Rect,
    /// The controls, while they show.
    pub controls: Option<KaraokeControls>,
}

/// Karaoke mode's controls.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KaraokeControls {
    /// The bar they are in ([`controls_bar`]).
    pub bar: Rect,
    /// The sliders.
    pub sliders: Vec<SliderLayout>,
    /// The time heard, the seek bar and the length, under the sliders.
    pub elapsed: Rect,
    /// The seek bar.
    pub seek: Rect,
    /// The length.
    pub total: Rect,
    /// The keys, in a line.
    pub hint: Rect,
}

/// How many rows Karaoke mode's controls' bar takes at the bottom.
pub const CONTROLS_ROWS: u16 = 5;

/// Karaoke mode's controls' bar on its `screen`: the bottom
/// [`CONTROLS_ROWS`] rows, whether it shows or not.
pub fn controls_bar(screen: Rect) -> Rect {
    let rows = CONTROLS_ROWS.min(screen.height);
    Rect::new(screen.x, screen.bottom() - rows, screen.width, rows)
}

/// How much of the `screen`'s height the controls' bar takes, 0 to 1
/// ([`crate::view::karaoke::KaraokePicture::bar`]).
pub fn controls_share(screen: Rect) -> f32 {
    f32::from(controls_bar(screen).height) / f32::from(screen.height.max(1))
}

/// A slider's label column, in cells.
const SLIDER_LABEL: u16 = 12;
/// A slider's value column (`100 %`, `−200 ms`), in cells.
const SLIDER_VALUE: u16 = 7;
/// The longest a slider's track gets, in cells.
const SLIDER_MOST: u16 = 40;

/// A slider in the row at `x`, `y`, `width` cells wide: its label, the
/// track (up to [`SLIDER_MOST`]) and the value after it.
fn slider_at(slider: Slider, x: u16, y: u16, width: u16) -> SliderLayout {
    let track = width
        .saturating_sub(SLIDER_LABEL + SLIDER_VALUE + 1)
        .min(SLIDER_MOST);
    let row = Rect::new(x, y, SLIDER_LABEL + track + 1 + SLIDER_VALUE, 1);
    SliderLayout {
        slider,
        row,
        track: Rect::new(x + SLIDER_LABEL, y, track, 1),
        value: Rect::new(x + SLIDER_LABEL + track + 1, y, SLIDER_VALUE, 1),
    }
}

/// The fewest columns the tab's text takes beside a large cover: in a
/// narrower pane the cover is small, the text under it.
const BESIDE: u16 = 80;
/// A small cover's width, in cells.
const SMALL: u16 = 16;

/// The Karaoke tab inside the list pane's border (`inner`), the voice's
/// lane under the pane in `voice`: the cover at the left, a third of the
/// pane wide at most, as tall as the pane allows with a row left at its
/// foot; in a pane too narrow for the text beside that, a small one.
pub(super) fn tab(inner: Rect, voice: Option<Rect>) -> KaraokeTabLayout {
    let tall = inner.height.saturating_sub(2);
    let large = (inner.width / 3).min(tall.saturating_mul(2));
    let (width, height) = if inner.width.saturating_sub(large + 4) >= BESIDE {
        (large, tall)
    } else {
        let width = SMALL.min(large);
        (width, (width / 2 + 1).min(tall))
    };
    KaraokeTabLayout {
        cover: Rect::new(inner.x + 1, inner.y, width, height),
        voice,
    }
}
/// Karaoke mode: the whole screen, and the controls' bar over its bottom
/// while they show (the sliders, the seek bar and the keys; START is the
/// stage's pill, above the bar). The game's settings are not here but on
/// the tab: they stay as they are from a song's start to its end.
pub(super) fn mode(state: &State, area: Rect) -> KaraokeLayout {
    let controls = karaoke::controls::shown(state).then(|| {
        let bar = controls_bar(area);
        let room = bar.width.saturating_sub(4);
        let sliders = Slider::MODE
            .iter()
            .enumerate()
            .map(|(k, &s)| slider_at(s, bar.x + 2, bar.y + 1 + k as u16, room))
            .collect();
        // The seek bar as on the other screens: the time, the bar, the
        // length.
        let seek_y = bar.y + 1 + Slider::MODE.len() as u16;
        let time_w = 8u16;
        let elapsed = Rect::new(bar.x + 1, seek_y, time_w.min(room), 1);
        let total = Rect::new(
            (bar.x + 2 + room).saturating_sub(time_w),
            seek_y,
            time_w.min(room),
            1,
        );
        let seek = Rect::new(
            elapsed.right() + 1,
            seek_y,
            total.x.saturating_sub(elapsed.right() + 2),
            1,
        );
        KaraokeControls {
            bar,
            sliders,
            elapsed,
            seek,
            total,
            hint: Rect::new(bar.x + 2, seek_y + 1, room, 1),
        }
    });
    KaraokeLayout {
        screen: area,
        controls,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The controls' bar is the bottom rows of the screen, however small,
    /// and its share of the height is theirs.
    #[test]
    fn the_bar_is_the_bottom_rows() {
        assert_eq!(
            controls_bar(Rect::new(0, 0, 80, 24)),
            Rect::new(0, 19, 80, 5)
        );
        assert_eq!(controls_bar(Rect::new(0, 0, 80, 3)), Rect::new(0, 0, 80, 3));
        assert_eq!(controls_share(Rect::new(0, 0, 80, 24)), 5.0 / 24.0);
        assert_eq!(controls_share(Rect::new(0, 0, 80, 3)), 1.0);
    }
}
