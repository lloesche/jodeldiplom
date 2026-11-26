//! The words along the voice's lane: laid out on the lane's grid of
//! columns, the first line centred on the lane and those under it after
//! it, each word painted whole with a dark edge so it reads over a loud
//! voice; the letters a cell each in the cells' glyphs, or larger in the
//! lyrics' face, a held sound a line between them.

use super::super::Pen;
use super::{Columns, WaveShape};
use crate::gui::big_text::{BigText, Face};
use crate::gui::pixels::{PixelRect, Target};
use crate::gui::raster::CellFont;
use crate::view::lanes::{self, WavePlot};
use crate::view::palette;

/// A word drawn on a Transition lane: its first column on the grid (absolute,
/// [`Columns::n0`] on), its characters, whether in the accent colour.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct WaveLabel {
    col: i64,
    text: String,
    heard: bool,
    /// Its line ([`crate::core::lanes::WordLines`]).
    row: usize,
}

/// A word's run on a lane: from, to (pixels), and its letters and their
/// size (none: a held sound's line).
type Piece = (f64, f64, Option<(String, f32)>);

/// A lane's words as laid out this frame.
pub(super) struct Words {
    /// The words that show.
    pub(super) labels: Vec<WaveLabel>,
    /// The first line's top in the lane's rect, where the words show.
    row: Option<usize>,
    /// A character's size in cells, a cell's width and height at it, and
    /// the lines.
    text: usize,
    cw: usize,
    tch: usize,
    lines: usize,
    /// A stroke's width at the window's scale.
    step: usize,
    /// The grid's first column, and the rect's width.
    n0: i64,
    w: usize,
}

impl Words {
    /// `plot`'s words on the grid `cols` of the lane `shape`: on the grid's
    /// columns (absolute: `n0` on), a character `text` cells wide and as
    /// many high (one, but on the Karaoke tab), centred on the lane, where
    /// that shows.
    pub(super) fn lay_out(plot: &WavePlot, cols: Columns, shape: &WaveShape, pen: &Pen) -> Words {
        let (_, _, w, h) = shape.pixels;
        let (cell_w, ch) = pen.cell();
        let (above, full) = (shape.above, shape.full);
        let text = shape.text;
        let cw = cell_w * text;
        let tch = ch * text;
        let step = pen.scale().round().max(1.0) as usize;
        // The first line centred on the lane, those under it after it (all
        // of them moved up as far as they need to fit).
        let lines = shape.rows;
        let middle = (full / 2)
            .saturating_sub(tch / 2)
            .min(full.saturating_sub(tch * lines));
        let row = (middle >= above && middle - above + tch <= h).then(|| middle - above);
        let labels: Vec<WaveLabel> = match (plot.lyrics.as_deref(), row) {
            (Some(lyrics), Some(_)) => {
                let words = crate::core::lanes::timed_words(lyrics);
                let x_of = |t: f64| (t / cols.px).floor();
                // A word just past either edge too: its outline reaches in.
                let reach = (step * text) as f64 + 1.0;
                let view = (cols.n0 as f64 - reach, (cols.n0 + w as i64) as f64 + reach);
                lanes::lane_labels(
                    &words,
                    &x_of,
                    1.0 / cols.px,
                    cw as f64,
                    plot.lines,
                    plot.playhead,
                    view,
                )
                .into_iter()
                .filter(|l| middle + (l.row + 1) * tch <= full)
                .map(|l| WaveLabel {
                    col: l.at as i64,
                    text: l.text,
                    heard: l.heard,
                    row: l.row,
                })
                .collect()
            }
            _ => Vec::new(),
        };
        Words {
            labels,
            row,
            text,
            cw,
            tch,
            lines,
            step,
            n0: cols.n0,
            w,
        }
    }

    /// The columns a label's pixels may touch (its outline and a glyph
    /// past its cell included), on this frame's grid.
    pub(super) fn span(&self, l: &WaveLabel) -> (i64, i64) {
        let margin = (self.step * self.text) as i64 + 1;
        let a = l.col - self.n0 - margin;
        let b = l.col - self.n0 + (l.text.chars().count() * self.cw) as i64 + margin;
        (a.clamp(0, self.w as i64), b.clamp(0, self.w as i64))
    }

    /// The words whose columns `mask` paints, into the lane's pixels
    /// `(x0, y0, w, h)` of `target`: each a dark edge first, then its
    /// letters and held sounds' lines.
    pub(super) fn paint(
        &self,
        target: &mut Target,
        pen: &mut Pen,
        mask: &[bool],
        (x0, y0, w, h): PixelRect,
    ) {
        let Some(row) = self.row else {
            return;
        };
        let (text, cw, tch, lines, step) = (self.text, self.cw, self.tch, self.lines, self.step);
        let big = pen.faces.get(Face::Lyrics);
        let font = &mut *pen.font;
        let text_px = pen.font_px * text as f32;
        let y = y0 + row;
        let clip = (x0, x0 + w);
        let edge = palette::BG;
        // The outline as wide as a stroke at the size drawn.
        let outline = step * text.div_ceil(2);
        let s = outline as f64;
        // The edges above and below only where they stay in the lane.
        let below = row + tch * lines + outline <= h;
        let up = row >= outline;
        // A held sound's line: as thick as a stroke, through the
        // middle of the cell.
        let thick = step.max(2) * text.div_ceil(2);
        let line_y = y + tch / 2 - thick / 2;
        // Letters in the cells' own glyphs, a cell each (one cell high);
        // larger, a run in the lyrics font.
        let put =
            |font: &mut CellFont, target: &mut Target, x: f64, y: usize, run: &str, fg: [u8; 3]| {
                font.text_at(target, x, y, run, fg, clip);
            };
        let clip_rect = (x0, y0, w, h);
        let draw_run = |big: &mut BigText,
                        target: &mut Target,
                        (x, y): (f64, usize),
                        run: &str,
                        size: f32,
                        fg: [u8; 3]| {
            big.draw(
                target,
                run,
                size,
                size,
                (x as f32, y as f32),
                f32::MAX,
                (fg, fg),
                1.0,
                clip_rect,
            );
        };
        for l in &self.labels {
            let (a, b) = self.span(l);
            if a >= b || !mask[a as usize] {
                continue;
            }
            let fg = if l.heard {
                palette::ACCENT
            } else {
                palette::TEXT
            };
            let x = (x0 as i64 + l.col - self.n0) as f64;
            // Its line's top, and its held sound's line.
            let y = y + l.row * tch;
            let line_y = line_y + l.row * tch;
            if y + tch > y0 + h {
                continue;
            }
            let below = below || l.row + 1 < lines;
            let up = up || l.row > 0;
            // The letters, and the runs of the line between them.
            let chars: Vec<char> = l.text.chars().collect();
            let mut runs: Vec<(usize, usize, bool)> = Vec::new();
            for (i, &c) in chars.iter().enumerate() {
                let held = c == lanes::HOLD;
                match runs.last_mut() {
                    Some(r) if r.2 == held => r.1 = i + 1,
                    _ => runs.push((i, i + 1, held)),
                }
            }
            // Each run where it goes: letters from their first cell, a
            // held sound's line between. In the cells' own glyphs, one
            // a cell; larger, in the lyrics font with its own spacing
            // (narrowed where a run would not fit its cells), the line
            // from just after the letters before it.
            let mut pieces: Vec<Piece> = Vec::new();
            let mut after: Option<f64> = None;
            for (n, &(i, j, held)) in runs.iter().enumerate() {
                let (slot0, slot1) = (x + (i * cw) as f64, x + (j * cw) as f64);
                if held {
                    let from = match (text, after) {
                        (1, _) | (_, None) => slot0,
                        (_, Some(end)) => end + s,
                    };
                    let to = if text > 1 && n + 1 < runs.len() {
                        slot1 - s
                    } else {
                        slot1
                    };
                    pieces.push((from, to.max(from), None));
                    continue;
                }
                let run: String = chars[i..j].iter().collect();
                if text == 1 {
                    pieces.push((slot0, slot1, Some((run, 0.0))));
                    continue;
                }
                let natural = f64::from(big.measure(&run, text_px));
                // Up to the next letters (into a held sound's line
                // between, which then starts later), else to the end of
                // the word's cells.
                let next = runs[n + 1..]
                    .iter()
                    .find(|r| !r.2)
                    .map_or(x + (chars.len() * cw) as f64, |r| x + (r.0 * cw) as f64 - s);
                let room = (next - slot0).max(slot1 - slot0);
                let size = if natural > room {
                    (text_px * (room / natural) as f32).floor()
                } else {
                    text_px
                };
                let width = natural.min(room);
                after = Some(slot0 + width);
                pieces.push((slot0, slot0 + width, Some((run, size))));
            }
            let px = |v: f64| (v.round().max(0.0) as usize).clamp(x0, x0 + w);
            // A dark edge first, so the words read over a loud voice.
            for (from, to, letters) in &pieces {
                let Some((run, size)) = letters else {
                    let top = line_y.saturating_sub(outline).max(y0);
                    let bottom = (line_y + thick + outline).min(y0 + h);
                    let left = px(*from).saturating_sub(outline).max(x0);
                    target.fill(left, top, (px(*to) + outline).min(x0 + w), bottom, edge);
                    continue;
                };
                for (dx, dy) in [(-s, 0.0), (s, 0.0), (0.0, -s), (0.0, s)] {
                    if (dy > 0.0 && !below) || (dy < 0.0 && !up) {
                        continue;
                    }
                    let row = (y as f64 + dy).max(0.0) as usize;
                    if text == 1 {
                        put(font, target, from + dx, row, run, edge);
                    } else {
                        draw_run(big, target, (from + dx, row), run, *size, edge);
                    }
                }
            }
            for (from, to, letters) in &pieces {
                match letters {
                    None => target.fill(px(*from), line_y, px(*to), line_y + thick, fg),
                    Some((run, _)) if text == 1 => put(font, target, *from, y, run, fg),
                    Some((run, size)) => draw_run(big, target, (*from, y), run, *size, fg),
                }
            }
        }
    }
}
