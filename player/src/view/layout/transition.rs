//! The Transition tab's layout in cells: a block for the playing track
//! and one for the next where there is room, each a header, the whole
//! track, the bar numbers and the lanes.

use ratatui::layout::Rect;

use crate::core::lanes::{self, Block, Lane};
use crate::core::transition::viewport;
use crate::core::State;
use crate::view::cells::width;

/// Cells left of a block's lanes for their labels.
pub const TRANSITION_LANE_LABELS: u16 = 7;
/// The most rows a lane of the Transition tab takes.
pub const TRANSITION_LANE_ROWS: u16 = 4;

/// One block of the Transition tab: a header line, the whole track in one row,
/// the bar numbers, and the lanes (all from [`TRANSITION_LANE_LABELS`] in, so their
/// columns are the same times).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransitionBlockLayout {
    /// The track it shows.
    pub block: Block,
    /// Title, tempo and bar.
    pub header: Rect,
    /// The whole track.
    pub overview: Rect,
    /// Bar numbers and the cursor.
    pub ruler: Rect,
    /// The lanes, top first, each with its label's cell (empty where that
    /// row is out of view) and the part of it in view.
    pub lanes: Vec<(Lane, Rect, Rect)>,
    /// The rows a lane takes, in view or not.
    pub lane_rows: u16,
    /// The rows of the first lane listed that are above the view (while
    /// the block slides out of it).
    pub top_cut: u16,
}

/// The Transition tab: the playing track's block, then the next one's when
/// there is room.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransitionLayout {
    /// The blocks, top first.
    pub blocks: Vec<TransitionBlockLayout>,
    /// The button that zooms to fit the lyrics ([`FIT_LYRICS`]), in the
    /// pane's bottom border.
    pub fit: Option<Rect>,
}

/// The lanes a Transition block shows: every stem's where its track has them
/// (or, before its analysis is in, where stems are asked for), else the whole
/// song's.
fn block_lanes(state: &State, block: Block) -> &'static [Lane] {
    let stems = match lanes::rhythmed(state, block) {
        Some(r) => r.waveform.as_ref().is_some_and(|w| w.stems.is_some()),
        None => state.rhythm_recipe.stems,
    };
    if stems {
        &Lane::ALL
    } else {
        &Lane::ALL[..1]
    }
}

/// The Transition tab's blocks in `inner`: each a header, an overview and a
/// ruler row and its lanes, as many rows a lane (up to [`TRANSITION_LANE_ROWS`]) as both
/// blocks fit with; the next track's block only where it fits.
pub(super) fn compute(state: &State, inner: Rect, list: Rect) -> TransitionLayout {
    let lanes = [
        block_lanes(state, Block::Now),
        block_lanes(state, Block::Next),
    ];
    let need = |which: usize, h: u16| 3 + lanes[which].len() as u16 * h;
    let rows = inner.height;
    // The tallest lanes (up to four rows) both blocks fit with.
    let both = (1..=TRANSITION_LANE_ROWS)
        .rev()
        .find(|&h| need(0, h) + 1 + need(1, h) <= rows);
    let plan: Vec<(usize, u16)> = match both {
        Some(h) => vec![(0, h), (1, h)],
        None if need(0, 1) <= rows => vec![(0, 1)],
        None => Vec::new(),
    };
    let mut order: Vec<(Block, &'static [Lane], u16)> = plan
        .into_iter()
        .map(|(which, h)| ([Block::Now, Block::Next][which], lanes[which], h))
        .collect();
    let (top, bottom) = (i32::from(inner.y), i32::from(inner.bottom()));
    let mut y = top;
    // After a hand-over, the track left slides up out of view and the
    // playing one up into its place, the next one in under it.
    if let (Some(done), Some(h)) = (viewport::slide(state), both) {
        let prev = block_lanes(state, Block::Prev);
        let pitch = 3 + prev.len() as i32 * i32::from(h) + 1;
        y -= (done * f64::from(pitch)).round() as i32;
        order.insert(0, (Block::Prev, prev, h));
    }
    let x = inner.x + TRANSITION_LANE_LABELS.min(inner.width);
    let w = lane_width(inner);
    let shown = |y: i32| y >= top && y < bottom;
    let row = |y: i32, x: u16, w: u16| {
        if shown(y) {
            Rect::new(x, y as u16, w, 1)
        } else {
            Rect::default()
        }
    };
    let mut blocks = Vec::new();
    for (block, block_lanes, h) in order {
        let header = row(y, inner.x, inner.width);
        let overview = row(y + 1, x, w);
        let ruler = row(y + 2, x, w);
        let first = y;
        y += 3;
        let mut placed = Vec::new();
        let mut top_cut = 0;
        for &lane in block_lanes {
            let (from, to) = (y.max(top), (y + i32::from(h)).min(bottom));
            if to > from {
                if placed.is_empty() {
                    top_cut = (from - y) as u16;
                }
                placed.push((
                    lane,
                    row(y, inner.x, TRANSITION_LANE_LABELS.min(inner.width)),
                    Rect::new(x, from as u16, w, (to - from) as u16),
                ));
            }
            y += i32::from(h);
        }
        y += 1;
        if first >= bottom || y <= top {
            continue;
        }
        blocks.push(TransitionBlockLayout {
            block,
            header,
            overview,
            ruler,
            lanes: placed,
            lane_rows: h,
            top_cut,
        });
    }
    // The zoom that fits the lyrics, at the right of the pane's bottom
    // border.
    let w = width(FIT_LYRICS) as u16;
    let fit = (list.width >= w + 24 && list.height >= 2)
        .then(|| Rect::new(list.right().saturating_sub(w + 2), list.bottom() - 1, w, 1));
    TransitionLayout { blocks, fit }
}

/// The button that zooms the Transition tab to fit the lyrics.
pub const FIT_LYRICS: &str = " z fit lyrics ";

/// The width of the lanes in the tab's pane `inner`, right of their
/// labels.
pub(super) fn lane_width(inner: Rect) -> u16 {
    inner.width.saturating_sub(TRANSITION_LANE_LABELS)
}
