//! A Transition block's bar numbers in pixels: each bar a hairline, its
//! number starting at it (to the pixel, so they move with the waveforms)
//! where there is room, the cue a diamond and the cursor a small triangle
//! over theirs.

use super::paint::Columns;
use super::Pen;
use crate::gui::look;
use crate::gui::pixels::{PixelRect, Pixels, Target};
use crate::view::lanes;
use crate::view::transition::RulerPlot;

/// `plot`'s bar numbers into the pixels `(x0, y0, w, h)` of `target`, on
/// the grid `cols`. The numbers are laid out from the first bar given (the
/// track's first), so they stand where they are as the view moves.
pub(super) fn paint(
    target: &mut Target,
    pen: &mut Pen,
    plot: &RulerPlot,
    cols: Columns,
    (x0, y0, w, h): PixelRect,
) {
    let line = pen.metrics.line();
    let scale = pen.metrics.scale() as f32;
    // The numbers a step up the type scale from the text round them, to
    // read at a glance.
    let size = (look::size::TITLE * scale).round();
    let gap = (look::space::XS * scale).round() as f64;
    let dim = look::bytes(look::TEXT_DIM);
    let faint = look::bytes(look::TEXT_FAINT);
    target.sub((x0, y0, w, h)).fill(super::RULER);
    let column = |t: f64| cols.of(t) as f64;
    let big = pen.faces.get(super::NUMBERS);
    let marks: Vec<f64> = [plot.cue, plot.cursor]
        .into_iter()
        .flatten()
        .map(column)
        .collect();
    // The marks' size, and the hairlines' height from the bottom.
    let mark = (h as f64 * 0.45).round().max(4.0);
    let tick = (h / 3).max(2);
    let mut labels = Vec::new();
    let mut free = f64::NEG_INFINITY;
    for &(t, bar) in &plot.bars {
        let x = column(t);
        if x >= w as f64 {
            break;
        }
        if x + (line as f64) > 0.0 {
            let mut px = target.sub((x0, y0, w, h));
            for dx in 0..line {
                let at = x as i64 + dx as i64;
                if at >= 0 {
                    for y in h - tick..h {
                        px.set(at as usize, y, faint);
                    }
                }
            }
        }
        // A mark on the bar goes before its number.
        let mut at = x + (line as f64) + gap / 2.0;
        if marks.iter().any(|&m| (m - x).abs() < mark) {
            at += mark / 2.0;
        }
        let label = bar.to_string();
        let width = f64::from(big.measure(&label, size));
        if at < free {
            continue;
        }
        free = at + width + 2.0 * gap;
        // Left of the view, but reaching into it: drawn, cut.
        if at + width > 0.0 {
            labels.push((at, label));
        }
    }
    let (top, bottom) = big.ink_band("0123456789", size);
    let y = y0 as f32 + ((h as f32 - (bottom - top)) / 2.0 - top).round();
    for (x, label) in labels {
        big.draw(
            target,
            &label,
            size,
            size,
            ((x0 as f64 + x) as f32, y),
            f32::MAX,
            (dim, dim),
            1.0,
            (x0, y0, w, h),
        );
    }
    let mut px = target.sub((x0, y0, w, h));
    if let Some(x) = plot.cue.map(column) {
        shape(
            &mut px,
            x + line as f64 / 2.0,
            mark,
            Shape::Diamond,
            lanes::CUE,
        );
    }
    if let Some(x) = plot.cursor.map(column) {
        let c = look::bytes(look::ACCENT);
        shape(&mut px, x + line as f64 / 2.0, mark, Shape::Down, c);
    }
}

/// A mark's shape.
#[derive(Clone, Copy)]
enum Shape {
    /// A diamond (the cue).
    Diamond,
    /// A triangle pointing down (the cursor).
    Down,
}

/// `shape`, `size` pixels across, centred on `cx` from the top of `px`,
/// its edges smoothed (each pixel's share of it, four samples a side).
fn shape(px: &mut Pixels, cx: f64, size: f64, shape: Shape, c: [u8; 3]) {
    let half = size / 2.0;
    let inside = |x: f64, y: f64| match shape {
        Shape::Diamond => (x - cx).abs() + (y - half).abs() <= half,
        Shape::Down => {
            y >= 0.0 && y <= size * 0.8 && (x - cx).abs() <= half * (1.0 - y / (size * 0.8))
        }
    };
    let from = (cx - half - 1.0).floor().max(0.0) as usize;
    let to = (cx + half + 1.0).ceil().max(0.0) as usize;
    for y in 0..size.ceil() as usize + 1 {
        for x in from..to {
            let mut n = 0;
            for sy in 0..4 {
                for sx in 0..4 {
                    let (u, v) = (
                        x as f64 + (f64::from(sx) + 0.5) / 4.0,
                        y as f64 + (f64::from(sy) + 0.5) / 4.0,
                    );
                    n += usize::from(inside(u, v));
                }
            }
            if n > 0 {
                px.blend(x, y, c, n as f32 / 16.0);
            }
        }
    }
}
