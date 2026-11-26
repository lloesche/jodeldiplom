//! A Transition block's bar numbers in pixels: each starting at its bar's
//! line in the lanes below (to the pixel, so they move with the
//! waveforms), a tick where there is no room for it, the cue and the
//! cursor centred on theirs; kept in the pixmap where nothing in it
//! changed.

use ratatui::layout::Rect;

use super::lane::Columns;
use super::{Canvas, Pen};
use crate::gui::pixels::{PixelRect, Target};
use crate::view::transition::RulerPlot;
use crate::view::{lanes, palette};

/// A Transition block's bar numbers as the pixmap holds them.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct KeptRuler {
    /// Its cells.
    pub(super) area: Rect,
    cols: Columns,
    bars: Vec<(f64, usize)>,
    cursor: Option<f64>,
    cue: Option<f64>,
}

impl Canvas {
    /// `plot`'s bar numbers in its cells, on the lanes' grid of columns
    /// ([`Columns`]); kept where nothing in it changed. Returns whether a
    /// pixel changed.
    pub(super) fn paint_ruler(
        &mut self,
        plot: &RulerPlot,
        kept: Option<KeptRuler>,
    ) -> (bool, Option<KeptRuler>) {
        let rect = self.pixel_rect(plot.area);
        let (_, _, w, h) = rect;
        if w == 0 || h == 0 {
            return (false, None);
        }
        let cols = Columns::new(plot.start, plot.end, w);
        let now = KeptRuler {
            area: plot.area,
            cols,
            bars: plot.bars.clone(),
            cursor: plot.cursor,
            cue: plot.cue,
        };
        if kept.as_ref() == Some(&now) && !self.terminal.backend().damage.touches(plot.area) {
            return (false, Some(now));
        }
        let (mut target, mut pen) = self.pixmap_and_pen();
        paint(&mut target, &mut pen, plot, cols, rect);
        (true, Some(now))
    }
}

/// `plot`'s bar numbers into the pixels `(x0, y0, w, h)` of `target`, on
/// the grid `cols`. The numbers are laid out from the first bar given (for
/// a ruler following the playhead, the track's first), so they stand where
/// they are as the view moves.
pub(super) fn paint(
    target: &mut Target,
    pen: &mut Pen,
    plot: &RulerPlot,
    cols: Columns,
    (x0, y0, w, h): PixelRect,
) {
    let (cw, _) = pen.cell();
    let line = pen.scale().round().max(1.0) as usize;
    let mut px = target.sub((x0, y0, w, h));
    px.fill(palette::BG);
    let column = |t: f64| cols.of(t) as f64;
    let dim = palette::DIM;
    let mut ticks = Vec::new();
    let mut labels = Vec::new();
    let mut free = f64::NEG_INFINITY;
    let marks: Vec<f64> = [plot.cue, plot.cursor]
        .into_iter()
        .flatten()
        .map(column)
        .collect();
    for &(t, bar) in &plot.bars {
        let mut x = column(t);
        if x >= w as f64 {
            break;
        }
        // A mark on the bar goes before its number.
        if marks.iter().any(|&m| (m - x).abs() < cw as f64) {
            x += cw as f64 / 2.0 + line as f64;
        }
        let label = bar.to_string();
        if x < free {
            if x >= 0.0 {
                ticks.push(x as usize);
            }
            continue;
        }
        free = x + ((label.len() + 1) * cw) as f64;
        // Left of the view, but reaching into it: drawn, cut.
        if x + (label.len() * cw) as f64 > 0.0 {
            labels.push((x, label));
        }
    }
    for x in ticks {
        for dx in 0..line {
            for y in 0..h / 2 {
                px.set(x + dx, y, dim);
            }
        }
    }
    let clip = (x0, x0 + w);
    for (x, label) in labels {
        pen.font
            .text_at(target, x0 as f64 + x, y0, &label, dim, clip);
    }
    for (time, symbol, color) in [
        (plot.cue, "◆", lanes::CUE),
        (plot.cursor, "▼", palette::ACCENT),
    ] {
        if let Some(x) = time.map(column).filter(|&x| x >= 0.0 && x < w as f64) {
            let left = x0 as f64 + x - (cw / 2) as f64;
            pen.font.text_at(target, left, y0, symbol, color, clip);
        }
    }
}
