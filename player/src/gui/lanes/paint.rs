//! A lane in pixels (a Transition lane, the Karaoke tab's voice lane): the
//! marks behind (the selection, the span the lanes show, the cue and the
//! cursor), the RMS filled and the peak outline over it, mirrored about the
//! middle, the bars as lines; on the voice's lane the notes sung and the
//! words across its middle ([`super::words`]), a held sound a line. The
//! playhead is not painted: it is drawn over the lane as it shows.

use std::sync::Arc;

use jodel_rhythm::waveform::{self, Melody, Waveform};

use super::keys::{self, Back, ColumnKey, MARK_CUE, MARK_CURSOR, MARK_EDIT};
use super::words::Words;
use super::Pen;
use crate::gui::pixels::{mix, pack, PixelRect, Pixels, Target};
use crate::gui::wgsl_math::{linear_to_srgb, srgb_to_linear};
use crate::view::lanes::{self, WavePlot};
use crate::view::palette;

/// How a picture's pixel columns map to time: column `x` is `n0 + x` on a
/// grid of `px` seconds from the start of the track. The grid is the same
/// from frame to frame while the span shown and the width are, so a view
/// that scrolls shows the same columns, moved by whole pixels (its start
/// is within a pixel of the plot's).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Columns {
    /// The seconds a column.
    pub(super) px: f64,
    /// The first column's place on the grid.
    pub(super) n0: i64,
}

impl Columns {
    /// The grid for `start..end` seconds across `w` columns.
    pub(super) fn new(start: f64, end: f64, w: usize) -> Columns {
        // Rounded, so a span that differs in its last bits from the last
        // frame's gives the same grid.
        let px = ((end - start) / w.max(1) as f64 * 1e12).round().max(1.0) / 1e12;
        Columns {
            px,
            n0: (start / px).round() as i64,
        }
    }

    /// The seconds at the left edge of column `x`.
    pub(super) fn time(&self, x: f64) -> f64 {
        (self.n0 as f64 + x) * self.px
    }

    /// The column `time` falls in (outside `0..w` where it is not shown).
    pub(super) fn of(&self, time: f64) -> i64 {
        (time / self.px).floor() as i64 - self.n0
    }
}

/// A lane's height: the rows of pixels a lane `full` high shows from
/// `above` down (all of them but while its block slides).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rows {
    /// The pixels of the lane above what shows.
    pub above: usize,
    /// The whole lane's height.
    pub full: usize,
}

/// The melodies the lanes last drew, worked out once for their waveform
/// (and lyrics).
#[derive(Default)]
pub(super) struct Melodies {
    /// The range of the last melody drawn, by its waveform.
    range: Option<(usize, Option<(u8, u8)>)>,
    /// The last melody drawn with what is not sung taken out, by its
    /// waveform and the lyrics whose words it was kept to.
    clean: Option<(usize, usize, Arc<Melody>)>,
}

impl Melodies {
    /// The range of `waveform`'s melody ([`Melody::range`]), kept from the
    /// last time it was asked for the same waveform.
    fn range(&mut self, waveform: &Arc<Waveform>) -> Option<(u8, u8)> {
        let id = Arc::as_ptr(waveform) as usize;
        match self.range {
            Some((kept, range)) if kept == id => range,
            _ => {
                let range = waveform.melody.as_ref().and_then(|m| m.range());
                self.range = Some((id, range));
                range
            }
        }
    }

    /// `waveform`'s melody with what is not sung taken out
    /// ([`Melody::cleaned`]): kept to the timed words of `lyrics` (all of
    /// it without), crumbs dropped; kept from the last time it was asked
    /// for the same waveform and lyrics.
    fn clean(
        &mut self,
        waveform: &Arc<Waveform>,
        lyrics: Option<&Arc<jodel_lyrics::Lyrics>>,
    ) -> Option<Arc<Melody>> {
        let melody = waveform.melody.as_ref()?;
        let id = Arc::as_ptr(waveform) as usize;
        let words = lyrics.map_or(0, |l| Arc::as_ptr(l) as usize);
        if let Some((w, l, m)) = &self.clean {
            if (*w, *l) == (id, words) {
                return Some(Arc::clone(m));
            }
        }
        let spans: Option<Vec<(f64, f64)>> = lyrics.map(|l| {
            l.lines
                .iter()
                .flat_map(|line| &line.words)
                .filter_map(|w| w.start.zip(w.end))
                .collect()
        });
        let spans = spans.filter(|s| !s.is_empty());
        let clean = Arc::new(melody.cleaned(spans.as_deref()));
        self.clean = Some((id, words, Arc::clone(&clean)));
        Some(clean)
    }
}

/// `plot`'s lane into the pixels `(x0, y0, w, h)` of `target`, on the grid
/// `cols`: the columns, then the words over them; `rows`: the part of the
/// lane they show. Nothing where the waveform has no levels for the lane.
pub(super) fn paint(
    target: &mut Target,
    pen: &mut Pen,
    plot: &WavePlot,
    cols: Columns,
    (x0, y0, w, h): PixelRect,
    rows: Rows,
) {
    if w == 0 || h == 0 || plot.levels().is_none() {
        return;
    }
    let line = pen.metrics.line();
    let full = rows.full.max(h + rows.above);
    let rows = Rows { full, ..rows };
    let keys = keys::keys(plot, cols, (w, h), line, (rows.above, full));
    // The voice's notes, on the lane that is read: each on its row of
    // the song's range (a semitone more each side), the scale's rows
    // lined across.
    let clean = if plot.deep {
        pen.melodies.clean(&plot.waveform, plot.lyrics.as_ref())
    } else {
        None
    };
    let notes = match &clean {
        Some(m) => pen
            .melodies
            .range(&plot.waveform)
            .map(|(lo, hi)| NoteRows::new(m, lo, hi, full, line, plot.key)),
        None => None,
    };
    let words = Words::lay_out(plot, cols, (w, h), rows, pen);
    let mut px = target.sub((x0, y0, w, h));
    paint_columns(&mut px, &keys, plot, cols, notes.as_ref(), (line, h, rows));
    words.paint(target, pen, (x0, y0, w, h));
}

/// The columns of a lane: the background its key gives, the RMS filled
/// and the peak outlined, the notes, the lines over them.
fn paint_columns(
    px: &mut Pixels,
    keys: &[ColumnKey],
    plot: &WavePlot,
    cols: Columns,
    notes: Option<&NoteRows>,
    (line, h, rows): (usize, usize, Rows),
) {
    let Some(levels) = plot.levels() else {
        return;
    };
    let (above, full) = (rows.above, rows.full);
    let ground = super::GROUND;
    let mid = full as f64 / 2.0;
    let curve = lanes::CURVE;
    for (x, &key) in keys.iter().enumerate() {
        let (t0, t1) = (cols.time(x as f64), cols.time(x as f64 + 1.0));
        let back = match key.back {
            Back::Selected => lanes::SELECT_BG,
            Back::Shown => lanes::SHOWN_BG,
            Back::Outside => lanes::OUTSIDE_BG,
            Back::Transition => lanes::TRANSITION_BG,
            Back::None => ground,
        };
        let color = if key.unclear {
            lanes::dimmed(plot.color)
        } else {
            plot.color
        };
        let color = lanes::faded(color, f32::from_bits(key.gain));
        let color = if key.back == Back::Outside {
            lanes::faded(color, lanes::OUTSIDE_GAIN)
        } else {
            color
        };
        // The RMS filled, the peak beyond it outlined; under words to be
        // read, the RMS deep and the peak bright.
        let (body, edge) = if plot.deep {
            (color, deep(color))
        } else {
            (lanes::dimmed(color), color)
        };
        let peak = f64::from(levels.peak_between(t0, t1)) * mid;
        let rms = f64::from(levels.rms_between(t0, t1)) * mid;
        let plain = (back, body, edge);
        let (back, body, edge) = (pack(back), pack(body), pack(edge));
        // The column in one pass: RMS filled, the peak outline, else
        // the background; the scale's lines across all of it.
        px.column(x, |y| {
            let d = ((y + above) as f64 + 0.5 - mid).abs();
            let (packed, c) = if d <= rms {
                (edge, plain.2)
            } else if d <= peak {
                (body, plain.1)
            } else {
                (back, plain.0)
            };
            match notes.map_or(0.0, |n| n.line_at(y + above)) {
                0.0 => packed,
                share => pack(mix(c, palette::TEXT, share)),
            }
        });
        // The note sung in the column's middle, a bar on its row with a
        // dark edge, over the voice.
        if let Some(n) = notes {
            let frame = ((t0 + t1) / 2.0 * waveform::FPS).floor();
            if let Some((top, bottom)) = (frame >= 0.0).then(|| n.bar(frame as usize)).flatten() {
                let bar = lanes::NOTE;
                for yy in top.saturating_sub(line)..bottom + line {
                    let Some(y) = yy.checked_sub(above) else {
                        continue;
                    };
                    if yy < top || yy >= bottom {
                        px.blend(x, y, palette::BG, 0.7);
                    } else {
                        px.blend(x, y, bar, 0.92);
                    }
                }
            }
        }
        // The lines over it, in their order: the bars, the cue, the
        // cursor, the bar being edited.
        let mut over = |c: [u8; 3], share: f32| {
            for y in 0..h {
                px.blend(x, y, c, share);
            }
        };
        for _ in 0..key.downbeats {
            over(palette::TEXT, 0.22);
        }
        for (bit, c, share) in [
            (MARK_CUE, lanes::CUE, 0.9),
            (MARK_CURSOR, palette::ACCENT, 0.55),
            (MARK_EDIT, lanes::CURVE, 0.8),
        ] {
            if key.marks & bit != 0 {
                over(c, share);
            }
        }
        // The transition's fade: a line from the bottom (off) to the
        // top (full), joined to the column before.
        if let Some((lo, hi)) = key.curve {
            for yy in lo..=hi {
                for dy in 0..line {
                    px.blend(x, (yy + dy).min(h - 1), curve, 0.9);
                }
            }
        }
    }
}

/// Where the voice's notes go on a lane `full` pixels high: a row a
/// semitone from a semitone over the range `hi` down to one under `lo`,
/// the lane's top and bottom sixteenth left free; the scale's rows of
/// `key` lined.
struct NoteRows<'a> {
    melody: &'a Melody,
    hi: u8,
    top: f64,
    row: f64,
    /// Each pixel row's line, as the share of the text's colour over the
    /// background (0 for none).
    lines: Vec<f32>,
    /// A bar's thickness.
    thick: usize,
}

impl<'a> NoteRows<'a> {
    fn new(
        melody: &'a Melody,
        lo: u8,
        hi: u8,
        full: usize,
        line: usize,
        key: Option<jodel_rhythm::key::Key>,
    ) -> NoteRows<'a> {
        let (lo, hi) = (lo.saturating_sub(1), hi.saturating_add(1).max(lo + 2));
        let top = full as f64 / 16.0;
        let row = (full as f64 - 2.0 * top) / f64::from(hi - lo + 1);
        let thick = ((row * 0.6).round() as usize).clamp(2 * line, 5 * line);
        let mut rows = NoteRows {
            melody,
            hi,
            top,
            row,
            lines: vec![0.0; full],
            thick,
        };
        if let Some(key) = key {
            let scale = key.scale();
            for n in lo..=hi {
                let class = n % 12;
                if !scale.contains(&class) {
                    continue;
                }
                let share = if class == key.tonic() { 0.2 } else { 0.08 };
                let y = rows.middle(n).round() as usize;
                for v in rows
                    .lines
                    .iter_mut()
                    .skip(y.saturating_sub(line / 2))
                    .take(line)
                {
                    *v = share;
                }
            }
        }
        rows
    }

    /// Note `n`'s row's middle, in pixels from the lane's top.
    fn middle(&self, n: u8) -> f64 {
        self.top + (f64::from(self.hi) - f64::from(n) + 0.5) * self.row
    }

    /// Pixel row `y`'s line.
    fn line_at(&self, y: usize) -> f32 {
        self.lines.get(y).copied().unwrap_or(0.0)
    }

    /// The bar of the note sung at `frame`, as pixel rows from the lane's
    /// top (from, to).
    fn bar(&self, frame: usize) -> Option<(usize, usize)> {
        let n = self.melody.sung(frame)?;
        let y = (self.middle(n) - self.thick as f64 / 2.0).round().max(0.0) as usize;
        Some((y, y + self.thick))
    }
}

/// `c` at an eighth of its light (in linear light): a deep shade of it,
/// near-white text on it reading at about 9 : 1.
fn deep(c: [u8; 3]) -> [u8; 3] {
    let lin = |v: u8| srgb_to_linear(f32::from(v) / 255.0);
    let enc = |v: f32| (linear_to_srgb(v.clamp(0.0, 1.0)) * 255.0).round() as u8;
    c.map(|v| enc(lin(v) * 0.12))
}
