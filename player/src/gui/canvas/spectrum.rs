//! The spectrum in pixels: the grid as single dots every few pixels, the
//! curve as round dots at a fixed spacing along it, both at the window's
//! scale.

use crate::gui::pixels::{PixelRect, Target};
use crate::view::palette;
use crate::view::plot::{self, Plot};

/// `plot`'s grid and curve into the pixels `rect` of `target`, at `scale`
/// pixels a point.
pub(super) fn paint(target: &mut Target, rect: PixelRect, scale: f64, plot: &Plot) {
    let (_, _, w, h) = rect;
    let spectrum = &plot.spectrum;
    if w == 0 || h == 0 || spectrum.is_empty() {
        return;
    }
    let grid_dot = scale.round().max(1.0) as usize;
    let grid_step = (4.0 * scale).round().max(3.0) as usize;
    let radius = scale.max(1.0);
    let spacing = 4.0 * scale.max(1.0);
    let mut px = target.sub(rect);
    px.fill(palette::BG);
    // The grid.
    let grid = palette::GRID;
    for &db in &plot.db_lines {
        let y = plot::unit_of(db, h);
        for x in (0..w).step_by(grid_step) {
            px.square(x, y, grid_dot, grid);
        }
    }
    let lines = plot.hz_lines.iter().map(|&hz| (hz, grid));
    let edges = plot.edges.iter().map(|&(hz, c)| (hz, plot::faint(c)));
    for (hz, color) in lines.chain(edges) {
        let x = spectrum.x_of(hz, w).floor();
        if (0.0..w as f64).contains(&x) {
            for y in (0..h).step_by(grid_step) {
                px.square(x as usize, y, grid_dot, color);
            }
        }
    }
    // The curve: one point per dot width, dots every `spacing` along it.
    let n = (w as f64 / (2.0 * radius)).round().max(2.0) as usize;
    let points: Vec<(f64, f64)> = spectrum
        .levels(n)
        .iter()
        .enumerate()
        .map(|(i, &db)| {
            let x = (i as f64 + 0.5) * w as f64 / n as f64;
            (x, plot::depth(db) * h as f64)
        })
        .collect();
    let floor = h as f64 - 0.5;
    let mut dot = |x: f64, y: f64| {
        if y < floor {
            let hz = spectrum.hz_at(x, w);
            let color = plot::zone_color(&plot.zones, hz);
            px.disc(x, y.clamp(radius, h as f64 - radius), radius, color);
        }
    };
    let mut next = 0.0;
    let mut walked = 0.0;
    for pair in points.windows(2) {
        let ((ax, ay), (bx, by)) = (pair[0], pair[1]);
        let length = (bx - ax).hypot(by - ay);
        while next <= walked + length {
            let t = (next - walked) / length;
            dot(ax + (bx - ax) * t, ay + (by - ay) * t);
            next += spacing;
        }
        walked += length;
    }
}
