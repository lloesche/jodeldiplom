//! The spectrum plot both frontends draw: a dotted curve of the level per
//! column, coloured by zone, over a faint dotted grid, with the level (dB)
//! on the left and the frequency under it.
//!
//! [`Plot::new`] lays it out in cells: where the curve goes, which grid
//! lines there are and the axis labels. The labels are text in cells for
//! both frontends; the curve and the grid are drawn by the terminal in
//! braille dots ([`spectrum_cells`], 2 × 4 dots a cell) and by the window
//! in pixels.
//!
//! Vertically, a level is at `depth × n` of `n` units from the top
//! ([`depth`]), the floor of [`RANGE_DB`] at `n`, just below the plot, so
//! silence draws nothing. The terminal's units are dot rows, the window's
//! pixel rows and the labels' cell rows, so a grid line lands in the row
//! of its label in both.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};

use crate::core::text::width;
use crate::spectrum::{SpectrumFrame, LOW_HZ, RANGE_DB};
use crate::view::colors::{BG, DIM, GRID};
use crate::view::{zone_color, Zone};

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

/// The dot rows of a curve through one level per dot column, `n` dot
/// rows high: each column's dot, and the dots joining it to its
/// neighbours where they are more than one row apart (the upper column
/// takes the upper half of the gap), so a steep slope stays a line.
/// Dot rows at or below `n` (the floor) are left out.
pub fn curve_dots(levels: &[f32], n: usize) -> Vec<Vec<usize>> {
    let ys: Vec<usize> = levels.iter().map(|&db| unit_of(db, n)).collect();
    let mut dots: Vec<Vec<usize>> = ys.iter().map(|&y| vec![y]).collect();
    for c in 1..ys.len() {
        let (a, b) = (ys[c - 1], ys[c]);
        let (lo, hi) = (a.min(b), a.max(b));
        if hi - lo < 2 {
            continue;
        }
        let mid = (lo + hi) / 2;
        let (upper, lower) = if a < b { (c - 1, c) } else { (c, c - 1) };
        dots[upper].extend(lo + 1..=mid);
        dots[lower].extend(mid + 1..hi);
    }
    for column in &mut dots {
        column.retain(|&y| y < n);
        column.sort_unstable();
        column.dedup();
    }
    dots
}

/// The braille bit of the dot at `col` (0 or 1) and `row` (0 to 3) of a
/// cell.
pub fn braille_bit(col: usize, row: usize) -> u8 {
    const BITS: [[u8; 4]; 2] = [[0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80]];
    BITS[col][row]
}

/// The braille character with `bits` set.
pub fn braille(bits: u8) -> char {
    char::from_u32(0x2800 + u32::from(bits)).unwrap_or(' ')
}

/// The braille dots of a curve through `levels` (two per cell column) in
/// `rows` cell rows: `[row][column]` bits.
pub fn curve_cells(levels: &[f32], rows: usize) -> Vec<Vec<u8>> {
    let cols = levels.len().div_ceil(2);
    let mut cells = vec![vec![0u8; cols]; rows];
    for (c, column) in curve_dots(levels, rows * 4).iter().enumerate() {
        for &y in column {
            cells[y / 4][c / 2] |= braille_bit(c % 2, y % 4);
        }
    }
    cells
}

/// The spectrum drawn with braille dots: what the terminal shows. A cell
/// with curve dots shows only those, in its zone's colour; the others
/// show the grid.
pub fn spectrum_cells(buf: &mut Buffer, plot: &Plot) {
    let area = plot.area;
    if area.width == 0 || area.height == 0 || plot.spectrum.is_empty() {
        return;
    }
    let (w, h) = (usize::from(area.width), usize::from(area.height));
    let spectrum = &plot.spectrum;
    let curve = curve_cells(&spectrum.levels(w * 2), h);
    // The grid: bits and colour per cell (a zone boundary over a line).
    let mut grid = vec![vec![(0u8, GRID); w]; h];
    for &db in &plot.db_lines {
        let y = unit_of(db, h * 4);
        if y < h * 4 {
            for cell in &mut grid[y / 4] {
                cell.0 |= braille_bit(0, y % 4);
            }
        }
    }
    let lines = plot.hz_lines.iter().map(|&hz| (hz, GRID));
    let edges = plot.edges.iter().map(|&(hz, c)| (hz, faint(c)));
    for (hz, color) in lines.chain(edges) {
        let x = spectrum.x_of(hz, w * 2).floor();
        if !(0.0..(w * 2) as f64).contains(&x) {
            continue;
        }
        let x = x as usize;
        for y in (0..h * 4).step_by(2) {
            let cell = &mut grid[y / 4][x / 2];
            cell.0 |= braille_bit(x % 2, y % 4);
            if color != GRID {
                cell.1 = color;
            }
        }
    }
    for row in 0..h {
        for col in 0..w {
            let (bits, color) = if curve[row][col] != 0 {
                let hz = spectrum.hz_at(col as f64 + 0.5, w);
                (curve[row][col], zone_color(&plot.zones, hz))
            } else {
                grid[row][col]
            };
            if bits == 0 {
                continue;
            }
            let mut symbol = [0u8; 4];
            buf[(area.x + col as u16, area.y + row as u16)]
                .set_symbol(braille(bits).encode_utf8(&mut symbol))
                .set_style(Style::new().fg(color));
        }
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

    #[test]
    fn levels_land_on_their_braille_dots() {
        // Two rows: 8 dot rows of 20 dB. 0 dB is the top dot, -150 the
        // bottom one, -160 (the floor) nothing.
        let levels = [0.0, -20.0, -150.0, -160.0];
        // -20 (row 1) to -150 (row 7): the gap rows 2..=6 split between
        // the two columns, the upper half on the upper one.
        let dots = curve_dots(&levels, 8);
        assert_eq!(dots, vec![vec![0], vec![1, 2, 3, 4], vec![5, 6, 7], vec![]]);
        let cells = curve_cells(&levels, 2);
        // Top cell: the left column's row 0, the right column's rows 1 to 3.
        assert_eq!(cells[0][0], 0x01 | 0x10 | 0x20 | 0x80);
        assert_eq!(cells[0][1], 0);
        // Bottom cells: the right column's row 4, then rows 5 to 7.
        assert_eq!(cells[1][0], 0x08);
        assert_eq!(cells[1][1], 0x02 | 0x04 | 0x40);
        assert_eq!(braille(0x01 | 0x10), '⠑');
        assert_eq!(braille(0xff), '⣿');
    }

    #[test]
    fn a_steep_step_stays_continuous() {
        // Quiet, then a jump to full scale, then a fall to the floor.
        let mut levels = vec![-140.0; 6];
        levels.extend([0.0; 6]);
        levels.extend([-160.0; 4]);
        let rows = 3;
        let dots = curve_dots(&levels, rows * 4);
        // Every dot row between the two levels has a dot in the two
        // columns at the rise, and from the top down to the bottom row at
        // the fall.
        let rise: Vec<usize> = [&dots[5], &dots[6]]
            .into_iter()
            .flatten()
            .copied()
            .collect();
        let (low, high) = (unit_of(-140.0, 12), unit_of(0.0, 12));
        for y in high..=low {
            assert!(rise.contains(&y), "row {y} missing at the rise: {dots:?}");
        }
        let fall: Vec<usize> = [&dots[11], &dots[12]]
            .into_iter()
            .flatten()
            .copied()
            .collect();
        for y in 0..12 {
            assert!(fall.contains(&y), "row {y} missing at the fall: {dots:?}");
        }
        // Flat parts are one dot a column; the floor draws nothing.
        assert_eq!(dots[2].len(), 1);
        assert_eq!(dots[8], vec![0]);
        assert!(dots[14].is_empty());
        // In cells: the rise is the right column of cell 2 (rows 6 to 10)
        // and the left one of cell 3 (rows 0 to 5).
        let cells = curve_cells(&levels, rows);
        assert_eq!(cells[0][3], 0x01 | 0x02 | 0x04 | 0x40 | 0x08);
        assert_eq!(cells[1][2], 0x20 | 0x80);
        assert_eq!(cells[1][3], 0x01 | 0x02);
        assert_eq!(cells[2][2], 0x04 | 0x08 | 0x10 | 0x20);
    }
}
