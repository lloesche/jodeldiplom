//! The spectrum plot: a curve of the level per column, coloured by zone,
//! over a faint grid, with the level (dB) on the left and the frequency
//! under it.
//!
//! [`Plot::new`] lays it out in cells: where the curve goes, which grid
//! lines there are and the axis labels. The labels are text in cells; the
//! curve and the grid are painted by the window in pixels.
//!
//! Vertically, a level is at `depth × n` of `n` units from the top
//! ([`depth`]), the floor of [`RANGE_DB`] at `n`, just below the plot, so
//! silence draws nothing. The window's units are pixel rows and the
//! labels' cell rows, so a grid line lands in the row of its label.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};

use crate::core::text::width;
use crate::spectrum::{SpectrumFrame, LOW_HZ, RANGE_DB};
use crate::view::colors::{BG, DIM};
use crate::view::Zone;

/// The width of the level labels (`-120`, `0 dB`).
pub const DB_LABEL_W: u16 = 4;
/// The narrowest spectrum that gets level labels.
pub const DB_AXIS_MIN_W: u16 = 40;
/// The narrowest spectrum that gets frequency labels.
pub const HZ_AXIS_MIN_W: u16 = 20;

/// The frequency ticks: hertz, label and whether it is a decade (a
/// decade wins over the others when labels would overlap).
const HZ_TICKS: [(f64, &str, bool); 14] = [
    (20.0, "20 Hz", true),
    (50.0, "50", false),
    (100.0, "100", true),
    (200.0, "200", false),
    (500.0, "500", false),
    (1_000.0, "1k", true),
    (2_000.0, "2k", false),
    (5_000.0, "5k", false),
    (10_000.0, "10k", true),
    (20_000.0, "20k", false),
    (50_000.0, "50k", false),
    (100_000.0, "100k", true),
    (200_000.0, "200k", false),
    (500_000.0, "500k", false),
];

/// A label in cells.
#[derive(Clone, Debug, PartialEq)]
pub struct AxisLabel {
    /// Its first cell.
    pub x: u16,
    /// Its row.
    pub y: u16,
    /// The text.
    pub text: String,
    /// Its colour.
    pub color: Color,
}

/// A spectrum laid out for drawing.
#[derive(Clone, Debug, PartialEq)]
pub struct Plot {
    /// Where the curve and the grid go (right of the level labels).
    pub area: Rect,
    /// The spectrum.
    pub spectrum: SpectrumFrame,
    /// The zones colouring its frequency ranges.
    pub zones: Vec<Zone>,
    /// The levels of the horizontal grid lines, in dB.
    pub db_lines: Vec<f32>,
    /// The frequencies of the vertical grid lines (the labelled ticks).
    pub hz_lines: Vec<f64>,
    /// The zone boundaries inside the plot, with their zone's colour.
    pub edges: Vec<(f64, Color)>,
    /// The level labels, left of the plot.
    pub db_labels: Vec<AxisLabel>,
    /// The frequency labels, under it.
    pub hz_labels: Vec<AxisLabel>,
}

/// How far down a level is: 0 at the top of [`RANGE_DB`], 1 at its floor.
pub fn depth(db: f32) -> f64 {
    let (lo, hi) = RANGE_DB;
    f64::from(((hi - db) / (hi - lo)).clamp(0.0, 1.0))
}

/// The unit a level falls in, of `n` from the top: `n` (below the plot)
/// at the floor.
pub fn unit_of(db: f32, n: usize) -> usize {
    ((depth(db) * n as f64).floor() as usize).min(n)
}

/// The level ticks for a plot of `rows` rows: every 20, 40, 80 or 160 dB
/// from 0 dB down (the smallest step that leaves two rows between
/// labels), not the floor.
pub fn db_ticks(rows: u16) -> Vec<f32> {
    let range = RANGE_DB.1 - RANGE_DB.0;
    let step = [20.0, 40.0, 80.0]
        .into_iter()
        .find(|step| f32::from(rows) * step / range >= 2.0)
        .unwrap_or(range);
    let mut out = Vec::new();
    let mut db = RANGE_DB.1;
    while db > RANGE_DB.0 + 0.5 {
        out.push(db);
        db -= step;
    }
    out
}

fn db_label(db: f32) -> String {
    if db >= RANGE_DB.1 {
        format!("{db:.0} dB")
    } else {
        format!("{db:.0}")
    }
}

/// A zone edge's label: whole kilohertz, as `22k`.
fn edge_label(hz: f64) -> String {
    format!("{}k", (hz / 1000.0).floor())
}

/// The cell column a frequency falls in, of a plot `n` cells wide.
fn column_of(spectrum: &SpectrumFrame, hz: f64, n: u16) -> u16 {
    let x = spectrum.x_of(hz, usize::from(n)).floor();
    x.clamp(0.0, f64::from(n.saturating_sub(1))) as u16
}

impl Plot {
    /// Lays out `spectrum` in `area`, with the frequency labels in
    /// `labels` (the row under it) when there is one. An empty spectrum
    /// has no grid and no labels.
    pub fn new(
        area: Rect,
        labels: Option<Rect>,
        spectrum: &SpectrumFrame,
        zones: Vec<Zone>,
    ) -> Plot {
        let has_db_axis = area.width >= DB_AXIS_MIN_W && area.height > 0;
        let plot = if has_db_axis {
            Rect {
                x: area.x + DB_LABEL_W + 1,
                width: area.width - DB_LABEL_W - 1,
                ..area
            }
        } else {
            area
        };
        let mut out = Plot {
            area: plot,
            spectrum: spectrum.clone(),
            zones,
            db_lines: Vec::new(),
            hz_lines: Vec::new(),
            edges: Vec::new(),
            db_labels: Vec::new(),
            hz_labels: Vec::new(),
        };
        if spectrum.is_empty() || plot.width == 0 || plot.height == 0 {
            return out;
        }
        let nyquist = spectrum.nyquist();
        out.edges = out
            .zones
            .iter()
            .filter(|z| z.up_to_hz < nyquist)
            .map(|z| (z.up_to_hz, z.color))
            .collect();
        out.db_lines = db_ticks(plot.height);
        if has_db_axis {
            let rows = usize::from(plot.height);
            out.db_labels = out
                .db_lines
                .iter()
                .map(|&db| {
                    let text = db_label(db);
                    AxisLabel {
                        x: area.x + DB_LABEL_W - width(&text) as u16,
                        y: plot.y + unit_of(db, rows).min(rows - 1) as u16,
                        text,
                        color: DIM,
                    }
                })
                .collect();
        }
        if let Some(row) = labels.filter(|r| r.width >= HZ_AXIS_MIN_W && r.height > 0) {
            out.lay_out_hz_labels(row);
        }
        out
    }

    /// The frequency labels: the zone edges first (from the top, so the
    /// Nyquist frequency always shows), then the decades, then the rest,
    /// each where it does not touch one already placed.
    fn lay_out_hz_labels(&mut self, row: Rect) {
        let nyquist = self.spectrum.nyquist();
        // (hz, text, colour, rank, is a tick)
        let mut wanted: Vec<(f64, String, Color, u8, bool)> = self
            .zones
            .iter()
            .map(|z| {
                let hz = z.up_to_hz.min(nyquist);
                (hz, edge_label(hz), z.color, 0, false)
            })
            .collect();
        for (hz, text, decade) in HZ_TICKS {
            if hz < LOW_HZ || hz >= nyquist {
                continue;
            }
            if wanted.iter().any(|w| w.3 == 0 && (w.0 - hz).abs() < 1.0) {
                continue;
            }
            wanted.push((hz, text.to_owned(), DIM, if decade { 1 } else { 2 }, true));
        }
        wanted.sort_by(|a, b| {
            let by_hz = if a.3 == 0 {
                b.0.total_cmp(&a.0)
            } else {
                a.0.total_cmp(&b.0)
            };
            a.3.cmp(&b.3).then(by_hz)
        });
        let mut taken: Vec<(u16, u16)> = Vec::new();
        for (hz, text, color, _, tick) in wanted {
            let w = width(&text) as u16;
            if w > row.width {
                continue;
            }
            // The number is centred on its column (`20 Hz` starts there).
            let number = text.split(' ').next().map_or(w, |n| width(n) as u16);
            let column = self.area.x + column_of(&self.spectrum, hz, self.area.width);
            let start = column
                .saturating_sub(number.saturating_sub(1) / 2)
                .clamp(row.x, row.right() - w);
            let end = start + w;
            if taken.iter().any(|&(a, b)| start < b + 1 && a < end + 1) {
                continue;
            }
            taken.push((start, end));
            if tick {
                self.hz_lines.push(hz);
            }
            self.hz_labels.push(AxisLabel {
                x: start,
                y: row.y,
                text,
                color,
            });
        }
        self.hz_labels.sort_by_key(|l| l.x);
        self.hz_lines.sort_by(f64::total_cmp);
    }

    /// Writes the axis labels into `buf`.
    pub fn draw_labels(&self, buf: &mut Buffer) {
        for label in self.db_labels.iter().chain(&self.hz_labels) {
            buf.set_string(label.x, label.y, &label.text, Style::new().fg(label.color));
        }
    }
}

/// `color` faded towards the background: a zone boundary's colour.
pub fn faint(color: Color) -> Color {
    match (color, BG) {
        (Color::Rgb(r, g, b), Color::Rgb(br, bg, bb)) => {
            let m = |c: u8, d: u8| ((u16::from(c) + u16::from(d)) / 2) as u8;
            Color::Rgb(m(r, br), m(g, bg), m(b, bb))
        }
        _ => color,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_fit_the_height() {
        assert_eq!(db_ticks(9), vec![0.0, -40.0, -80.0, -120.0]);
        assert_eq!(db_ticks(6), vec![0.0, -80.0]);
        assert_eq!(db_ticks(17).len(), 8);
        assert_eq!(db_ticks(3), vec![0.0]);
        assert_eq!(db_label(0.0), "0 dB");
        assert_eq!(db_label(-120.0), "-120");
    }

    #[test]
    fn the_floor_is_below_the_plot() {
        assert_eq!(unit_of(RANGE_DB.0, 36), 36);
        assert_eq!(unit_of(-200.0, 36), 36);
        assert_eq!(unit_of(0.0, 36), 0);
        assert_eq!(unit_of(6.0, 36), 0);
        assert_eq!(unit_of(-80.0, 36), 18);
    }
}
