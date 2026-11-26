//! What follows the playhead as a layer the window scrolls by itself: the
//! lanes and bar numbers painted in tiles, each once, and the frame that
//! says where the layer stands and which tiles go over.

use std::ops::RangeInclusive;
use std::sync::Arc;

use super::lane::{self, Columns};
use super::{ruler, Canvas, Pen};
use crate::gui::pixels::{pack, PixelRect, Target};
use crate::view::lanes::WavePlot;
use crate::view::palette;
use crate::view::transition::RulerPlot;

/// The width of a layer's tiles, in pixels.
pub const TILE: usize = 512;

/// The most tile slots a layer has (the window's shader holds this many).
pub const MOST_SLOTS: usize = 16;

/// What follows the playhead together ([`WavePlot::follow`],
/// [`RulerPlot::follow`]): the Karaoke tab's voice lane, or the playing
/// track's block on the Transition tab (its bar numbers and lanes, stacked), on
/// one grid (column `n` is `n × px` seconds in). Painted in tiles of
/// [`TILE`] columns, each once, for the window to scroll through by itself:
/// a tile is painted again only when what is in it changes (the word heard,
/// a mark: the cursor, a selection, the cue, the fades), all of them when
/// the parts do (another track, zoom, size).
pub(super) struct Strip {
    /// The parts as last drawn (the lanes' playhead set per tile), each at
    /// its rows of pixels from the top.
    parts: Vec<Part>,
    /// Its pixels in the window, and the grid's seconds a column.
    rect: PixelRect,
    px: f64,
    /// The playhead's share of the way in, and the span shown.
    lead: f64,
    span: f64,
    /// Its tile slots: the tile each holds, if any.
    slots: Vec<Option<i64>>,
    /// Each word's times and the columns it takes (its outline in), in
    /// the order sung.
    words: Vec<(f64, f64, i64, i64)>,
    /// The word heard when the tiles were painted.
    heard: Option<usize>,
    /// Tiles to paint again (a mark in them changed).
    stale: Vec<i64>,
}

/// A part of a [`Strip`] and its rows of pixels: from, how many.
#[derive(Clone)]
pub(super) enum Part {
    Wave(Box<WavePlot>, usize, usize),
    Ruler(Box<RulerPlot>, usize, usize),
}

impl Part {
    /// Whether `self` and `other` paint the same tiles but for their marks
    /// (another track, zoom, size, colours, bars: all of them afresh).
    fn same_ground(&self, other: &Part) -> bool {
        match (self, other) {
            (Part::Wave(a, ay, ah), Part::Wave(b, by, bh)) => {
                (ay, ah) == (by, bh)
                    && Arc::ptr_eq(&a.waveform, &b.waveform)
                    && match (&a.lyrics, &b.lyrics) {
                        (Some(x), Some(y)) => Arc::ptr_eq(x, y),
                        (None, None) => true,
                        _ => false,
                    }
                    && (a.lane, a.color, a.rows, a.text_scale, a.lines, a.follow)
                        == (b.lane, b.color, b.rows, b.text_scale, b.lines, b.follow)
                    && (a.deep, a.key) == (b.deep, b.key)
                    && a.downbeats == b.downbeats
                    && a.unclear == b.unclear
            }
            (Part::Ruler(a, ay, ah), Part::Ruler(b, by, bh)) => {
                (ay, ah) == (by, bh) && a.bars == b.bars && a.follow == b.follow
            }
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
            (Part::Wave(a, ..), Part::Wave(b, ..)) => {
                points(a.cursor, b.cursor);
                points(a.cue, b.cue);
                points(a.edit, b.edit);
                for (x, y) in [
                    (a.selection, b.selection),
                    (a.shown, b.shown),
                    (a.transition, b.transition),
                ] {
                    if x != y {
                        out.extend(x.into_iter().chain(y));
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
            (Part::Ruler(a, ..), Part::Ruler(b, ..)) => {
                points(a.cursor, b.cursor);
                points(a.cue, b.cue);
            }
            _ => {}
        }
        out
    }
}

/// A layer's frame for the window ([`Canvas::layer_frame`]).
#[derive(Clone, Debug, PartialEq)]
pub struct LayerFrame {
    /// Where it goes in the window, in pixels.
    pub rect: PixelRect,
    /// Its tiles' width (their height is the rect's).
    pub tile: usize,
    /// How many tile slots there are.
    pub slots: usize,
    /// The grid's seconds a column.
    pub px: f64,
    /// The column at its left edge, whole.
    pub offset: i64,
    /// And the share of the next one.
    pub frac: f32,
    /// The playhead's left edge across it, in pixels (fractional), and its
    /// width, where it shows.
    pub playhead: Option<(f32, usize)>,
    /// The row of pixels it starts at (under the bar numbers).
    pub head_top: usize,
    /// Its colour.
    pub head: [u8; 3],
    /// The tile in each slot.
    pub resident: Vec<Option<i64>>,
    /// Tiles painted since the last frame, to send over: slot and pixels
    /// (`tile` × `rect.3`, rows of `0x00RRGGBB`).
    pub uploads: Vec<(usize, Vec<u32>)>,
}

/// The tiles a layer `w` pixels wide needs with column `offset` at its
/// left edge: those in view and the next one coming in, but never more
/// than its `slots` (a layer wider than its slots hold shows the first of
/// them; two tiles in one slot would take turns being uploaded).
fn tiles(offset: i64, w: usize, slots: usize) -> RangeInclusive<i64> {
    let first = offset.div_euclid(TILE as i64);
    let last = (offset + w as i64 + 1).div_euclid(TILE as i64) + 1;
    let last = last.min(first + slots as i64 - 1);
    debug_assert!(last - first < slots as i64, "more tiles than slots");
    first..=last
}

impl Canvas {
    /// The layer's frame with `heard` seconds into the track: where it
    /// stands, and the tiles it needs painted (in view, and the next one
    /// coming in). `None` where none shows.
    pub fn layer_frame(&mut self, heard: f64) -> Option<LayerFrame> {
        let strip = self.strip.as_ref()?;
        let (rect, px) = (strip.rect, strip.px);
        let w = rect.2;
        let start = (heard - strip.lead * strip.span).max(0.0);
        let left = start / px;
        let offset = left.floor() as i64;
        let frac = (left - offset as f64) as f32;
        // The word heard: its tiles again, and those of the one before; and
        // those a mark changed in.
        let heard_word = strip
            .words
            .iter()
            .position(|&(a, b, _, _)| heard >= a && heard < b);
        let slots_n = strip.slots.len();
        let mut stale = strip.stale.clone();
        if heard_word != strip.heard {
            for word in [strip.heard, heard_word].into_iter().flatten() {
                let (_, _, c0, c1) = strip.words[word];
                stale.extend(c0.div_euclid(TILE as i64)..=c1.div_euclid(TILE as i64));
            }
        }
        let tiles = tiles(offset, w, slots_n);
        let (first, last) = (*tiles.start(), *tiles.end());
        let head_top = strip
            .parts
            .iter()
            .find_map(|p| match p {
                Part::Wave(_, y, _) => Some(*y),
                Part::Ruler(..) => None,
            })
            .unwrap_or(0);
        let mut uploads = Vec::new();
        let parts: Vec<Part> = strip
            .parts
            .iter()
            .map(|p| match p {
                Part::Wave(plot, y, h) => {
                    let mut plot = plot.clone();
                    plot.playhead = Some(heard);
                    Part::Wave(plot, *y, *h)
                }
                other => other.clone(),
            })
            .collect();
        for tile in tiles {
            let slot = tile.rem_euclid(slots_n as i64) as usize;
            let held = self
                .strip
                .as_ref()
                .is_some_and(|s| s.slots[slot] == Some(tile));
            if held && !stale.contains(&tile) {
                continue;
            }
            let pixels = paint_tile(&mut self.pen(), &parts, px, tile, rect.3);
            if let Some(s) = self.strip.as_mut() {
                s.slots[slot] = Some(tile);
            }
            uploads.push((slot, pixels));
        }
        let strip = self.strip.as_mut()?;
        strip.heard = heard_word;
        // Stale tiles out of view now: painted when they come in.
        strip.stale.retain(|t| *t < first || *t > last);
        for slot in &mut strip.slots {
            if slot.is_some_and(|t| strip.stale.contains(&t)) {
                *slot = None;
            }
        }
        // The playhead as the lanes draw it: two lines wide, from the
        // column it falls in.
        let line = self.scale().round().max(1.0) as usize;
        let head_x = (heard / px - left) as f32;
        let strip = self.strip.as_ref()?;
        Some(LayerFrame {
            rect,
            tile: TILE,
            slots: slots_n,
            px,
            offset,
            frac,
            playhead: (head_x >= 0.0 && (head_x as usize) < w).then_some((head_x, 2 * line)),
            head_top,
            head: palette::ACCENT,
            resident: strip.slots.clone(),
            uploads,
        })
    }

    /// What follows the playhead this frame (`found`: each part and its
    /// pixels in the window, top to bottom) as the layer: kept where only
    /// its place moved, its tiles under a changed mark painted again, made
    /// afresh where the parts changed.
    pub(super) fn take_strip(&mut self, found: Vec<(Part, PixelRect)>) {
        let Some(&(_, first)) = found.first() else {
            self.strip = None;
            return;
        };
        let top = found.iter().map(|(_, r)| r.1).min().unwrap_or(first.1);
        let bottom = found
            .iter()
            .map(|(_, r)| r.1 + r.3)
            .max()
            .unwrap_or(first.1);
        let rect = (first.0, top, first.2, bottom - top);
        let (start, end, lead) = match &found[0].0 {
            Part::Wave(p, ..) => (p.start, p.end, p.follow.unwrap_or(0.0)),
            Part::Ruler(p, ..) => (p.start, p.end, p.follow.unwrap_or(0.0)),
        };
        let cols = Columns::new(start, end, rect.2);
        let parts: Vec<Part> = found
            .into_iter()
            .map(|(part, r)| match part {
                Part::Wave(p, ..) => Part::Wave(p, r.1 - top, r.3),
                Part::Ruler(p, ..) => Part::Ruler(p, r.1 - top, r.3),
            })
            .collect();
        let same = self.strip.as_ref().is_some_and(|s| {
            s.rect == rect
                && s.px == cols.px
                && s.parts.len() == parts.len()
                && s.parts.iter().zip(&parts).all(|(a, b)| a.same_ground(b))
        });
        if same {
            // The tiles under the marks that moved, a little either side
            // (a mark's width, a bar number pushed along by one).
            let pad = (6 * self.cell().0) as i64;
            if let Some(s) = self.strip.as_mut() {
                for (a, b) in s.parts.iter().zip(&parts) {
                    for (t0, t1) in a.marks_changed(b) {
                        let c0 = (t0 / s.px).floor() as i64 - pad;
                        let c1 = (t1 / s.px).floor() as i64 + pad;
                        s.stale
                            .extend(c0.div_euclid(TILE as i64)..=c1.div_euclid(TILE as i64));
                    }
                }
                s.stale.sort_unstable();
                s.stale.dedup();
                s.parts = parts;
                s.span = end - start;
            }
            return;
        }
        // Each word's columns, as the lane with words lays them out.
        let mut words = Vec::new();
        for part in &parts {
            let Part::Wave(plot, ..) = part else {
                continue;
            };
            let Some(lyrics) = plot.lyrics.as_deref() else {
                continue;
            };
            let text = usize::from(plot.text_scale.max(1));
            let cw = (self.cell().0 * text) as f64;
            let margin = (self.scale().round().max(1.0) as usize * text) as i64 + 1;
            let timed = crate::core::lanes::timed_words(lyrics);
            let x_of = |t: f64| (t / cols.px).floor();
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
            break;
        }
        let slots = (rect.2.div_ceil(TILE) + 3).min(MOST_SLOTS);
        self.strip = Some(Strip {
            parts,
            rect,
            px: cols.px,
            lead,
            span: end - start,
            slots: vec![None; slots],
            words,
            heard: None,
            stale: Vec::new(),
        });
    }
}

/// Tile `tile` of the layer's `parts` on the grid of `px` seconds a
/// column, `h` pixels high: painted as they would be (the playhead apart),
/// into pixels of its own.
fn paint_tile(pen: &mut Pen, parts: &[Part], px: f64, tile: i64, h: usize) -> Vec<u32> {
    let cols = Columns {
        px,
        n0: tile * TILE as i64,
    };
    let mut pixels = vec![pack(palette::BG); TILE * h];
    let mut target = Target {
        pixels: &mut pixels,
        width: TILE,
        height: h,
    };
    for part in parts {
        match part {
            Part::Wave(plot, y, ph) => {
                lane::paint(
                    &mut target,
                    pen,
                    plot,
                    cols,
                    (0, *y, TILE, *ph),
                    None,
                    false,
                );
            }
            Part::Ruler(plot, y, ph) => {
                ruler::paint(&mut target, pen, plot, cols, (0, *y, TILE, *ph));
            }
        }
    }
    pixels
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A layer as wide as its slots allow, or wider (8000 pixels: an 8K
    /// screen, a window across monitors), paints no more tiles than it has
    /// slots; one narrower paints those in view and the next one.
    #[test]
    fn a_layer_paints_no_more_tiles_than_it_has_slots() {
        for w in [300usize, 2411, 7000, 8000, 20_000] {
            let slots = (w.div_ceil(TILE) + 3).min(MOST_SLOTS);
            for offset in [0, 17, 511, 512, 100_003] {
                let shown = tiles(offset, w, slots);
                let n = shown.end() - shown.start() + 1;
                assert!(n <= slots as i64, "{w} px from {offset}: {n} tiles");
                let wanted = (offset + w as i64 + 1).div_euclid(TILE as i64) + 2
                    - offset.div_euclid(TILE as i64);
                assert_eq!(n, wanted.min(slots as i64), "{w} px from {offset}");
            }
        }
    }
}
