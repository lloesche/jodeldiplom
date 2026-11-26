//! The spectrum plot's model: a curve of the level across the frequencies
//! over a faint grid, with the level (dB) on the left and the frequency
//! under it, the Nyquist frequency in the curve's colour.
//!
//! [`Plot::new`] lays it out in points for a size: where the curve goes,
//! which grid lines there are, and where each axis label is. The window
//! draws it ([`crate::gui::widgets::spectrum`]).
//!
//! Vertically, a level is at `depth × h` of the curve's height `h` from
//! its top ([`depth`]), the floor of [`RANGE_DB`] at `h`, its bottom
//! edge, so silence draws nothing.

use std::fmt;

use crate::spectrum::{SpectrumFrame, LOW_HZ, RANGE_DB};

/// The narrowest spectrum that gets level labels, in points.
pub const DB_AXIS_MIN_W: f32 = 320.0;
/// The narrowest spectrum that gets frequency labels, in points.
pub const HZ_AXIS_MIN_W: f32 = 160.0;
/// The width of the level labels (`-120`, `0 dB`), in figures.
const DB_LABEL_FIGURES: f32 = 4.0;

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

/// The labels' type as the window sets it, in points: the width of a
/// figure (the faces' figures are all one width, and the labels are
/// mostly figures) and a line's height.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Type {
    /// A figure's width.
    pub figure: f32,
    /// A line's height.
    pub line: f32,
}

impl Type {
    /// `text`'s width in this type.
    fn width(self, text: &str) -> f32 {
        text.chars().count() as f32 * self.figure
    }
}

/// What colour a label is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelTone {
    /// An axis's tick.
    Dim,
    /// The Nyquist frequency, in the curve's colour.
    Source,
}

/// An axis label.
#[derive(Clone, Debug, PartialEq)]
pub struct AxisLabel {
    /// Its left edge, from the plot's left.
    pub x: f32,
    /// The middle of its line, from the plot's top.
    pub y: f32,
    /// Its width.
    pub width: f32,
    /// The text.
    pub text: String,
    /// Its colour.
    pub tone: LabelTone,
}

/// A rectangle in points, from the plot's top left.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Area {
    /// Its left edge.
    pub x: f32,
    /// Its top edge.
    pub y: f32,
    /// Its width.
    pub width: f32,
    /// Its height.
    pub height: f32,
}

/// A spectrum laid out for drawing.
#[derive(Clone, Debug, PartialEq)]
pub struct Plot {
    /// Where the curve and the grid go (right of the level labels, above
    /// the frequency labels).
    pub area: Area,
    /// The spectrum.
    pub spectrum: SpectrumFrame,
    /// The levels of the horizontal grid lines, in dB.
    pub db_lines: Vec<f32>,
    /// The frequencies of the vertical grid lines (the labelled ticks).
    pub hz_lines: Vec<f64>,
    /// The level labels, left of the curve.
    pub db_labels: Vec<AxisLabel>,
    /// The frequency labels, under it.
    pub hz_labels: Vec<AxisLabel>,
}

/// How far down a level is: 0 at the top of [`RANGE_DB`], 1 at its floor.
pub fn depth(db: f32) -> f64 {
    let (lo, hi) = RANGE_DB;
    f64::from(((hi - db) / (hi - lo)).clamp(0.0, 1.0))
}

/// The level ticks for a curve `lines` label lines high: every 20, 40,
/// 80 or 160 dB from 0 dB down (the smallest step that leaves two lines
/// between labels), not the floor.
pub fn db_ticks(lines: f32) -> Vec<f32> {
    let range = RANGE_DB.1 - RANGE_DB.0;
    let step = [20.0, 40.0, 80.0]
        .into_iter()
        .find(|step| lines * step / range >= 2.0)
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

/// The Nyquist frequency's label: whole kilohertz, as `22k`.
fn nyquist_label(hz: f64) -> String {
    format!("{}k", (hz / 1000.0).floor())
}

impl Plot {
    /// Lays out `spectrum` in `width` × `height` points, its labels in
    /// `ty`: the level labels at the left where it is wide enough, the
    /// frequency labels along the bottom where it is wide and tall
    /// enough (three lines). An empty spectrum has no grid and no labels.
    pub fn new(width: f32, height: f32, spectrum: &SpectrumFrame, ty: Type) -> Plot {
        let has_hz_axis = width >= HZ_AXIS_MIN_W && height >= 3.0 * ty.line;
        let has_db_axis = width >= DB_AXIS_MIN_W && height > 0.0;
        let db_w = DB_LABEL_FIGURES * ty.figure;
        let x = if has_db_axis { db_w + ty.figure } else { 0.0 };
        let area = Area {
            x,
            y: 0.0,
            width: (width - x).max(0.0),
            height: if has_hz_axis {
                height - ty.line
            } else {
                height
            }
            .max(0.0),
        };
        let mut out = Plot {
            area,
            spectrum: spectrum.clone(),
            db_lines: Vec::new(),
            hz_lines: Vec::new(),
            db_labels: Vec::new(),
            hz_labels: Vec::new(),
        };
        if spectrum.is_empty() || area.width <= 0.0 || area.height <= 0.0 {
            return out;
        }
        out.db_lines = db_ticks(area.height / ty.line);
        if has_db_axis {
            let half = ty.line / 2.0;
            out.db_labels = out
                .db_lines
                .iter()
                .map(|&db| {
                    let text = db_label(db);
                    let w = ty.width(&text);
                    let y = depth(db) as f32 * area.height;
                    AxisLabel {
                        x: db_w - w,
                        y: y.clamp(half, (area.height - half).max(half)),
                        width: w,
                        text,
                        tone: LabelTone::Dim,
                    }
                })
                .collect();
        }
        if has_hz_axis {
            out.lay_out_hz_labels(width, height - ty.line / 2.0, ty);
        }
        out
    }

    /// The frequency labels, their lines' middle at `y`: the Nyquist
    /// frequency first (so it always shows), then the decades, then the
    /// rest, each where it does not touch one already placed (a figure
    /// apart), within `width`.
    fn lay_out_hz_labels(&mut self, width: f32, y: f32, ty: Type) {
        let nyquist = self.spectrum.nyquist();
        // (hz, text, tone, rank, is a tick)
        let mut wanted: Vec<(f64, String, LabelTone, u8, bool)> =
            vec![(nyquist, nyquist_label(nyquist), LabelTone::Source, 0, false)];
        for (hz, text, decade) in HZ_TICKS {
            if (LOW_HZ..nyquist).contains(&hz) {
                wanted.push((
                    hz,
                    text.to_owned(),
                    LabelTone::Dim,
                    if decade { 1 } else { 2 },
                    true,
                ));
            }
        }
        wanted.sort_by(|a, b| a.3.cmp(&b.3).then(a.0.total_cmp(&b.0)));
        let mut taken: Vec<(f32, f32)> = Vec::new();
        for (hz, text, tone, _, tick) in wanted {
            let w = ty.width(&text);
            if w > width {
                continue;
            }
            // The number is centred on its frequency (`20 Hz` starts there).
            let number = text.split(' ').next().map_or(w, |n| ty.width(n));
            let at = self.x_of(hz);
            let start = (at - number / 2.0).clamp(0.0, width - w);
            let end = start + w;
            if taken
                .iter()
                .any(|&(a, b)| start < b + ty.figure && a < end + ty.figure)
            {
                continue;
            }
            taken.push((start, end));
            if tick {
                self.hz_lines.push(hz);
            }
            self.hz_labels.push(AxisLabel {
                x: start,
                y,
                width: w,
                text,
                tone,
            });
        }
        self.hz_labels.sort_by(|a, b| a.x.total_cmp(&b.x));
        self.hz_lines.sort_by(f64::total_cmp);
    }

    /// Where `hz` is across the plot, from its left.
    pub fn x_of(&self, hz: f64) -> f32 {
        let w = f64::from(self.area.width);
        self.area.x + (self.spectrum.x_of(hz, 1) * w) as f32
    }

    /// Where the level `db` is down the plot, from its top.
    pub fn y_of(&self, db: f32) -> f32 {
        self.area.y + depth(db) as f32 * self.area.height
    }

    /// The curve: a point every `step` points across the curve's area (at
    /// the middle of each step), at its level. Silence is at the area's
    /// bottom edge.
    pub fn curve(&self, step: f32) -> Vec<(f32, f32)> {
        if self.spectrum.is_empty() || self.area.width <= 0.0 {
            return Vec::new();
        }
        let n = (self.area.width / step.max(0.5)).round().max(2.0) as usize;
        let w = self.area.width / n as f32;
        self.spectrum
            .levels(n)
            .iter()
            .enumerate()
            .map(|(i, &db)| (self.area.x + (i as f32 + 0.5) * w, self.y_of(db)))
            .collect()
    }
}

/// The labels, as a line: the levels, then the frequencies, the Nyquist
/// frequency's tone marked.
impl fmt::Display for Plot {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let texts = |labels: &[AxisLabel]| {
            labels
                .iter()
                .map(|l| match l.tone {
                    LabelTone::Dim => l.text.clone(),
                    LabelTone::Source => format!("[source] {}", l.text),
                })
                .collect::<Vec<_>>()
                .join(" · ")
        };
        write!(
            f,
            "[spectrum] {} | {}",
            texts(&self.db_labels),
            texts(&self.hz_labels)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_fit_the_height() {
        assert_eq!(db_ticks(9.0), vec![0.0, -40.0, -80.0, -120.0]);
        assert_eq!(db_ticks(6.0), vec![0.0, -80.0]);
        assert_eq!(db_ticks(17.0).len(), 8);
        assert_eq!(db_ticks(3.0), vec![0.0]);
        assert_eq!(db_label(0.0), "0 dB");
        assert_eq!(db_label(-120.0), "-120");
    }

    #[test]
    fn the_floor_is_the_bottom_edge() {
        assert_eq!(depth(RANGE_DB.0), 1.0);
        assert_eq!(depth(-200.0), 1.0);
        assert_eq!(depth(0.0), 0.0);
        assert_eq!(depth(6.0), 0.0);
        assert_eq!(depth(-80.0), 0.5);
    }
}
