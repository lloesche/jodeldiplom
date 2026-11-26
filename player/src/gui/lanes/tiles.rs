//! A lane in tiles ([`Strip`]): its columns painted in tiles of [`TILE`],
//! each once, on one grid for the lane (column `n` is `n × px` seconds into
//! the track), and the frame that says where the lane stands on them
//! ([`LaneFrame`]): the GPU (or the processor, without one) shows the
//! tiles from there, moved by the share of a column, the playhead over
//! them. A tile is painted again only when what is in it changes (the word
//! heard, a mark: the cursor, a selection, the cue, the fades), all of them
//! when the lane does (another track, zoom, size).

use std::ops::RangeInclusive;
use std::sync::Arc;

use super::paint::{self, Columns, Rows};
use super::{ruler, Pen};
use crate::gui::pixels::{pack, PixelRect, Target};
use crate::view::lanes::WavePlot;
use crate::view::transition::RulerPlot;

/// The width of a lane's tiles, in pixels.
pub const TILE: usize = 512;

/// What a lane shows: a waveform (and the part of its height that shows),
/// or a Transition block's bar numbers.
#[derive(Clone, Debug)]
pub enum Part {
    /// A lane's waveform.
    Wave(Box<WavePlot>, Rows),
    /// A block's bar numbers.
    Ruler(Box<RulerPlot>),
}

impl Part {
    /// The seconds shown, and the share of the way in the playhead stays
    /// where the lane follows it.
    fn span(&self) -> (f64, f64, Option<f64>) {
        match self {
            Part::Wave(p, _) => (p.start, p.end, p.follow),
            Part::Ruler(p) => (p.start, p.end, p.follow),
        }
    }

    /// Where the playhead is drawn over the lane (bar numbers have none).
    fn playhead(&self) -> Option<f64> {
        match self {
            Part::Wave(p, _) => p.playhead,
            Part::Ruler(_) => None,
        }
    }

    /// Whether it is a waveform's lane (drawn on a card) rather than bar
    /// numbers (on the ground).
    pub(super) fn is_wave(&self) -> bool {
        matches!(self, Part::Wave(..))
    }

    /// Whether `self` and `other` paint the same tiles but for their marks
    /// (another track, zoom, size, colours, bars: all of them afresh).
    fn same_ground(&self, other: &Part) -> bool {
        match (self, other) {
            (Part::Wave(a, ar), Part::Wave(b, br)) => {
                ar == br
                    && Arc::ptr_eq(&a.waveform, &b.waveform)
                    && match (&a.lyrics, &b.lyrics) {
                        (Some(x), Some(y)) => Arc::ptr_eq(x, y),
                        (None, None) => true,
                        _ => false,
                    }
                    && (a.lane, a.color, a.text_scale, a.lines, a.follow)
                        == (b.lane, b.color, b.text_scale, b.lines, b.follow)
                    && (a.deep, a.key) == (b.deep, b.key)
                    && a.downbeats == b.downbeats
                    && a.unclear == b.unclear
            }
            (Part::Ruler(a), Part::Ruler(b)) => a.bars == b.bars && a.follow == b.follow,
            _ => false,
        }
    }

    /// The times (from, to) where `self`'s marks differ from `other`'s.
    fn marks_changed(&self, other: &Part) -> Vec<(f64, f64)> {
        let mut out = Vec::new();
        let mut points = |x: Option<f64>, y: Option<f64>| {
            if x != y {
                out.extend(x.into_iter().chain(y).map(|t| (t, t)));
            }
        };
        match (self, other) {
            (Part::Wave(a, _), Part::Wave(b, _)) => {
                points(a.cursor, b.cursor);
                points(a.cue, b.cue);
                points(a.edit, b.edit);
                for (x, y) in [
                    (a.selection, b.selection),
                    (a.shown, b.shown),
                    (a.transition, b.transition),
                ] {
                    match (x, y) {
                        // A span moved: only between its old edges and
                        // its new ones.
                        (Some((a0, a1)), Some((b0, b1))) if x != y => {
                            out.push((a0.min(b0), a0.max(b0)));
                            out.push((a1.min(b1), a1.max(b1)));
                        }
                        _ if x != y => out.extend(x.into_iter().chain(y)),
                        _ => {}
                    }
                }
                if a.gain != b.gain {
                    out.extend(a.transition.into_iter().chain(b.transition));
                    for g in [&a.gain, &b.gain] {
                        if let (Some(first), Some(last)) = (g.first(), g.last()) {
                            out.push((first.0, last.0));
                        }
                    }
                }
            }
            (Part::Ruler(a), Part::Ruler(b)) => {
                points(a.cursor, b.cursor);
                points(a.cue, b.cue);
            }
            _ => {}
        }
        out
    }
}

/// Where a lane stands this frame, and the tiles to send over.
#[derive(Clone, Debug, PartialEq)]
pub struct LaneFrame {
    /// Where it goes in the window, in pixels.
    pub rect: PixelRect,
    /// How many tile slots there are (tile `t` is in slot `t` modulo
    /// them; its height is the rect's).
    pub slots: usize,
    /// The grid's column at its left edge, whole.
    pub offset: i64,
    /// And the share of the next one.
    pub frac: f32,
    /// The playhead's left edge across it, in pixels (fractional), and its
    /// width, where it shows.
    pub playhead: Option<(f32, usize)>,
    /// Tiles painted since the last frame, to send over (or, where the
    /// processor shows them, kept already): slot and pixels (`TILE` ×
    /// `rect.3`, rows of `0x00RRGGBB`).
    pub uploads: Vec<(usize, Arc<[u32]>)>,
    /// The lane is new (its slots hold nothing yet).
    pub fresh: bool,
}

/// One lane in tiles.
pub(super) struct Strip {
    /// What it shows, as last drawn.
    part: Part,
    /// Its pixels in the window, and its grid.
    rect: PixelRect,
    cols: Columns,
    /// The playhead's share of the way in where it follows it, and the
    /// span shown.
    lead: Option<f64>,
    span: f64,
    /// The tile each slot holds, if any; and its pixels, kept where the
    /// processor shows them.
    slots: Vec<Option<i64>>,
    kept: Vec<Option<Arc<[u32]>>>,
    /// Each word's times and the columns it takes (its outline in), in
    /// the order sung.
    words: Vec<(f64, f64, i64, i64)>,
    /// The word heard when the tiles were painted.
    heard: Option<usize>,
    /// Tiles to paint again (a mark in them changed).
    stale: Vec<i64>,
    /// No frame of it was made yet.
    fresh: bool,
}

/// The tiles a lane `w` pixels wide needs with column `offset` at its left
/// edge: those in view and the next one coming in.
fn tiles(offset: i64, w: usize) -> RangeInclusive<i64> {
    let first = offset.div_euclid(TILE as i64);
    let last = (offset + w as i64 + 1).div_euclid(TILE as i64) + 1;
    first..=last
}

/// How many tile slots a lane `w` pixels wide has: one more than it ever
/// shows at once, so no two tiles it needs share one.
fn slots(w: usize) -> usize {
    w.div_ceil(TILE) + 3
}

impl Strip {
    /// `part` in `rect`, painted with `pen` as frames ask for its tiles.
    pub(super) fn new(part: Part, rect: PixelRect, pen: &Pen) -> Strip {
        let (start, end, lead) = part.span();
        let cols = Columns::new(start, end, rect.2);
        let words = match &part {
            Part::Wave(plot, _) => word_columns(plot, cols, pen),
            Part::Ruler(_) => Vec::new(),
        };
        let n = slots(rect.2);
        Strip {
            part,
            rect,
            cols,
            lead,
            span: end - start,
            slots: vec![None; n],
            kept: vec![None; n],
            words,
            heard: None,
            stale: Vec::new(),
            fresh: true,
        }
    }

    /// Whether `part` in `rect` paints the same tiles as the lane, but for
    /// its marks.
    pub(super) fn same(&self, part: &Part, rect: PixelRect) -> bool {
        let (start, end, _) = part.span();
        rect == self.rect
            && Columns::new(start, end, rect.2).px == self.cols.px
            && self.part.same_ground(part)
    }

    /// The lane as `part` shows it now ([`Strip::same`] as it was): the
    /// tiles under the marks that moved painted again, a little either
    /// side (a mark's width, a bar number pushed along).
    pub(super) fn update(&mut self, part: Part, pen: &Pen) {
        let pad = (6 * pen.metrics.cell.0) as i64;
        let px = self.cols.px;
        for (t0, t1) in self.part.marks_changed(&part) {
            let c0 = (t0 / px).floor() as i64 - pad;
            let c1 = (t1 / px).floor() as i64 + pad;
            self.stale
                .extend(c0.div_euclid(TILE as i64)..=c1.div_euclid(TILE as i64));
        }
        self.stale.sort_unstable();
        self.stale.dedup();
        let (start, end, _) = part.span();
        self.cols = Columns::new(start, end, self.rect.2);
        self.span = end - start;
        self.part = part;
    }

    /// The lane's frame with `heard` seconds into the track: where it
    /// stands (following the playhead, or where its view is), and the tiles
    /// it needs painted (in view, and the next one coming in); kept on the
    /// processor (`keep`: it shows them) or handed over to the GPU.
    pub(super) fn frame(&mut self, pen: &mut Pen, heard: f64, keep: bool) -> LaneFrame {
        let (_, _, w, h) = self.rect;
        let px = self.cols.px;
        let (left, at) = match self.lead {
            Some(lead) => {
                let start = (heard - lead * self.span).max(0.0);
                (start / px, self.part.playhead().map(|_| heard))
            }
            None => (self.cols.n0 as f64, self.part.playhead()),
        };
        // On a whole column within a ten-thousandth of one (what the
        // seconds' rounding leaves), so a lane at rest stands still.
        let whole = left.round();
        let left = if (left - whole).abs() < 1e-4 {
            whole
        } else {
            left
        };
        let offset = left.floor() as i64;
        let frac = (left - offset as f64) as f32;
        // The word heard: its tiles again, and those of the one before; and
        // those a mark changed in.
        let heard_word = at.and_then(|at| {
            self.words
                .iter()
                .position(|&(a, b, _, _)| at >= a && at < b)
        });
        if heard_word != self.heard {
            for word in [self.heard, heard_word].into_iter().flatten() {
                let (_, _, c0, c1) = self.words[word];
                self.stale
                    .extend(c0.div_euclid(TILE as i64)..=c1.div_euclid(TILE as i64));
            }
            self.heard = heard_word;
        }
        let n = self.slots.len();
        let shown = tiles(offset, w);
        let (first, last) = (*shown.start(), *shown.end());
        let mut uploads = Vec::new();
        let mut drawn: Option<Part> = None;
        for tile in shown {
            let slot = tile.rem_euclid(n as i64) as usize;
            if self.slots[slot] == Some(tile) && !self.stale.contains(&tile) {
                continue;
            }
            // The words heard as at the playhead.
            let part = drawn.get_or_insert_with(|| match &self.part {
                Part::Wave(plot, rows) => {
                    let mut plot = plot.clone();
                    plot.playhead = at;
                    Part::Wave(plot, *rows)
                }
                other => other.clone(),
            });
            let pixels = paint_tile(pen, part, px, tile, h);
            self.slots[slot] = Some(tile);
            if keep {
                self.kept[slot] = Some(Arc::clone(&pixels));
            }
            uploads.push((slot, pixels));
        }
        // Stale tiles out of view now: painted when they come in.
        self.stale.retain(|t| *t < first || *t > last);
        for (slot, kept) in self.slots.iter_mut().zip(&mut self.kept) {
            if slot.is_some_and(|t| self.stale.contains(&t)) {
                *slot = None;
                *kept = None;
            }
        }
        // The playhead as wide as two lines, from the column it falls in.
        let line = pen.metrics.line();
        let playhead = at
            .map(|t| (t / px - left) as f32)
            .filter(|&x| x >= 0.0 && (x as usize) < w)
            .map(|x| (x, 2 * line));
        let fresh = std::mem::take(&mut self.fresh);
        LaneFrame {
            rect: self.rect,
            slots: n,
            offset,
            frac,
            playhead,
            uploads,
            fresh,
        }
    }

    /// The pixels of tile `tile`, where the processor keeps them.
    pub(super) fn kept(&self, tile: i64) -> Option<&[u32]> {
        let slot = tile.rem_euclid(self.slots.len() as i64) as usize;
        (self.slots[slot] == Some(tile))
            .then(|| self.kept[slot].as_deref())
            .flatten()
    }

    /// Whether it is a waveform's lane.
    pub(super) fn is_wave(&self) -> bool {
        self.part.is_wave()
    }

    /// The seconds a column of its grid.
    pub(super) fn px(&self) -> f64 {
        self.cols.px
    }

    /// Whether it follows the playhead.
    pub(super) fn follows(&self) -> bool {
        self.lead.is_some()
    }
}

/// Each word's times and the columns it takes on `plot`'s lane (its
/// outline in), as the words lay them out on the grid `cols`.
fn word_columns(plot: &WavePlot, cols: Columns, pen: &Pen) -> Vec<(f64, f64, i64, i64)> {
    let Some(lyrics) = plot.lyrics.as_deref() else {
        return Vec::new();
    };
    let text = usize::from(plot.text_scale.max(1));
    let cw = (pen.metrics.cell.0 * text) as f64;
    let margin = (pen.metrics.line() * text) as i64 + 1;
    let timed = crate::core::lanes::timed_words(lyrics);
    let x_of = |t: f64| (t / cols.px).floor();
    let mut words = Vec::new();
    crate::core::lanes::lay_out_words(
        &timed,
        &x_of,
        1.0 / cols.px,
        cw,
        plot.lines,
        &mut |k, at, wide, _| {
            let w = timed[k];
            if let (Some(a), Some(b)) = (w.start, w.end) {
                let c0 = at as i64 - margin;
                let c1 = at as i64 + (wide as f64 * cw) as i64 + margin;
                words.push((a, b, c0, c1));
            }
            true
        },
    );
    words
}

/// Tile `tile` of `part` on the grid of `px` seconds a column, `h` pixels
/// high: painted as the lane is, into pixels of its own.
pub(super) fn paint_tile(pen: &mut Pen, part: &Part, px: f64, tile: i64, h: usize) -> Arc<[u32]> {
    let cols = Columns {
        px,
        n0: tile * TILE as i64,
    };
    let ground = if part.is_wave() {
        super::GROUND
    } else {
        super::RULER
    };
    let mut pixels = vec![pack(ground); TILE * h];
    let mut target = Target {
        pixels: &mut pixels,
        width: TILE,
        height: h,
    };
    match part {
        Part::Wave(plot, rows) => {
            paint::paint(&mut target, pen, plot, cols, (0, 0, TILE, h), *rows)
        }
        Part::Ruler(plot) => ruler::paint(&mut target, pen, plot, cols, (0, 0, TILE, h)),
    }
    pixels.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A lane of any width (an 8K screen, a window across monitors: wider
    /// than the 16 slots there once were) has a slot for each tile it
    /// shows and the next one coming in, never two of them in one.
    #[test]
    fn a_lane_has_a_slot_for_every_tile_it_shows() {
        for w in [300usize, 2411, 7000, 8000, 20_000] {
            let n = slots(w) as i64;
            for offset in [0, 17, 511, 512, 100_003, -700] {
                let shown = tiles(offset, w);
                let count = shown.end() - shown.start() + 1;
                assert!(count < n, "{w} px from {offset}: {count} tiles");
                let wanted = (offset + w as i64 + 1).div_euclid(TILE as i64) + 2
                    - offset.div_euclid(TILE as i64);
                assert_eq!(count, wanted, "{w} px from {offset}");
                let mut used: Vec<i64> = shown.map(|t| t.rem_euclid(n)).collect();
                used.sort_unstable();
                used.dedup();
                assert_eq!(used.len() as i64, count, "{w} px from {offset}");
            }
        }
    }
}
