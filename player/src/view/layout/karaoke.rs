//! The Karaoke tab's layout in cells: where its cover and the voice's
//! lane go (the window draws the rest of it, and Karaoke mode, in its
//! own widgets).

use ratatui::layout::Rect;

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
