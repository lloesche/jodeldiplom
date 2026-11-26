//! The words along the voice's lane: laid out on the lane's grid of
//! columns, a letter a slot (the cells' width, times the words' size), the
//! first line centred on the lane and those under it after it; each word
//! drawn in the lyrics' face from its first slot, with a dark edge so it
//! reads over a loud voice, a held sound a line between its letters.

use super::paint::{Columns, Rows};
use super::Pen;
use crate::gui::big_text::{BigText, Face};
use crate::gui::pixels::{PixelRect, Target};
use crate::view::lanes::{self, WavePlot};
use crate::view::palette;

/// A word drawn on a lane: its first column on the grid (absolute,
/// [`Columns::n0`] on), its characters, whether in the accent colour.
#[derive(Clone, Debug, PartialEq)]
struct WaveLabel {
    col: i64,
    text: String,
    heard: bool,
    /// Its line ([`crate::core::lanes::WordLines`]).
    row: usize,
}

/// A word's run on a lane: from, to (pixels), and its letters and their
/// size (none: a held sound's line).
type Piece = (f64, f64, Option<(String, f32)>);

/// A lane's words as laid out.
pub(super) struct Words {
    /// The words that show.
    labels: Vec<WaveLabel>,
    /// The first line's top in the lane's rect, where the words show.
    row: Option<usize>,
    /// A character's size in cells, a slot's width and a line's height at
    /// it, and the lines.
    text: usize,
    cw: usize,
    tch: usize,
    lines: usize,
    /// A stroke's width at the window's scale.
    step: usize,
    /// The grid's first column.
    n0: i64,
}

impl Words {
    /// `plot`'s words on the grid `cols` of a lane `(w, h)` pixels showing
    /// `rows` of it: on the grid's columns (absolute: `n0` on), a character
    /// [`WavePlot::text_scale`] cells wide and as many high, centred on the
    /// lane, where that shows.
    pub(super) fn lay_out(
        plot: &WavePlot,
        cols: Columns,
        (w, h): (usize, usize),
        rows: Rows,
        pen: &Pen,
    ) -> Words {
        let (cell_w, ch) = pen.metrics.cell;
        let (above, full) = (rows.above, rows.full);
        let text = usize::from(plot.text_scale.max(1));
        let cw = cell_w * text;
        let tch = ch * text;
        let step = pen.metrics.line();
        // The first line centred on the lane, those under it after it (all
        // of them moved up as far as they need to fit).
        let lines = plot.lines.rows.max(1);
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
        }
    }

    /// The words into the lane's pixels `(x0, y0, w, h)` of `target`: each
    /// a dark edge first, then its letters and held sounds' lines.
    pub(super) fn paint(&self, target: &mut Target, pen: &mut Pen, (x0, y0, w, h): PixelRect) {
        let Some(row) = self.row else {
            return;
        };
        let (text, cw, tch, lines, step) = (self.text, self.cw, self.tch, self.lines, self.step);
        let big = pen.faces.get(Face::Lyrics);
        // A cell high, the letters fill their line, as the cells' glyphs
        // did; larger, at the font's size the cells count in.
        let text_px = match text {
            1 => tch as f32,
            _ => pen.metrics.font_px * text as f32,
        };
        let y = y0 + row;
        let edge = palette::BG;
        // The outline as wide as a stroke at the size drawn.
        let outline = step * text.div_ceil(2);
        let s = outline as f64;
        // The edges above and below only where they stay in the lane.
        let below = row + tch * lines + outline <= h;
        let up = row >= outline;
        // A held sound's line: as thick as a stroke, through the
        // middle of the line.
        let thick = step.max(2) * text.div_ceil(2);
        let line_y = y + tch / 2 - thick / 2;
        let clip = (x0, y0, w, h);
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
                clip,
            );
        };
        for l in &self.labels {
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
            // Each run where it goes: letters from their first slot, in
            // the lyrics' face with its own spacing (narrowed where a run
            // would not fit its slots), a held sound's line from just
            // after the letters before it.
            let mut pieces: Vec<Piece> = Vec::new();
            let mut after: Option<f64> = None;
            for (n, &(i, j, held)) in runs.iter().enumerate() {
                let (slot0, slot1) = (x + (i * cw) as f64, x + (j * cw) as f64);
                if held {
                    let from = after.map_or(slot0, |end| end + s);
                    let to = if n + 1 < runs.len() { slot1 - s } else { slot1 };
                    pieces.push((from, to.max(from), None));
                    continue;
                }
                let run: String = chars[i..j].iter().collect();
                let natural = f64::from(big.measure(&run, text_px));
                // Up to the next letters (into a held sound's line
                // between, which then starts later), else to the end of
                // the word's slots.
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
                    draw_run(big, target, (from + dx, row), run, *size, edge);
                }
            }
            for (from, to, letters) in &pieces {
                match letters {
                    None => target.fill(px(*from), line_y, px(*to), line_y + thick, fg),
                    Some((run, size)) => draw_run(big, target, (*from, y), run, *size, fg),
                }
            }
        }
    }
}
