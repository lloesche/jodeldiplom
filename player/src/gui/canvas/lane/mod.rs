//! A Transition lane in pixels (and the Karaoke tab's voice lane): the
//! marks behind (the selection, the span the lanes show, the cursor and the
//! cue), the RMS filled and the peak outline over it, mirrored about the
//! middle, the bars as lines, the playhead over all; on the voice's lane
//! the notes sung and the words across its middle ([`words`]), a held
//! sound a line.
//!
//! The lane's pixels stay from the last frame: a view that scrolled is
//! shifted by whole columns, and only the columns new to it or whose marks
//! changed are painted ([`mask`]), with the words over them; all of them
//! where the lane is new, moved, or a cell under it was redrawn. The words
//! stand on the columns' grid, so a word kept moves with the pixels; one
//! that changed (the word heard now, one coming in) is painted afresh with
//! the columns under it.

mod mask;
mod words;

use std::sync::Arc;

use jodel_rhythm::waveform::{self, Melody, Waveform};
use ratatui::layout::Rect;

use super::{Canvas, Pen};
use crate::core::layout::Lane;
use crate::gui::pixels::{mix, pack, PixelRect, Pixels, Target};
use crate::view::lanes::{self, WavePlot};
use crate::view::palette;
use mask::{Back, ColumnKey, MARK_CUE, MARK_CURSOR, MARK_EDIT, MARK_PLAYHEAD};
use words::{WaveLabel, Words};

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
    fn time(&self, x: f64) -> f64 {
        (self.n0 as f64 + x) * self.px
    }

    /// The column `time` falls in (outside `0..w` where it is not shown).
    pub(super) fn of(&self, time: f64) -> i64 {
        (time / self.px).floor() as i64 - self.n0
    }
}

/// What a lane's kept pixels were painted for, besides the columns.
#[derive(Clone, Copy, Debug, PartialEq)]
struct WaveShape {
    area: Rect,
    pixels: PixelRect,
    lane: Lane,
    color: [u8; 3],
    line: usize,
    above: usize,
    full: usize,
    px: f64,
    /// The words' size, in cells, and their lines.
    text: usize,
    rows: usize,
    /// The key the notes' lines stand on.
    key: Option<jodel_rhythm::key::Key>,
}

/// A Transition lane as the pixmap holds it.
pub(super) struct KeptWave {
    shape: WaveShape,
    waveform: Arc<Waveform>,
    n0: i64,
    keys: Vec<ColumnKey>,
    /// The words drawn over it.
    labels: Vec<WaveLabel>,
}

impl KeptWave {
    /// The cells it was painted in.
    pub(super) fn area(&self) -> Rect {
        self.shape.area
    }
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

impl Canvas {
    /// `plot`'s lane in its cells ([`paint`]), on its own grid of columns,
    /// the playhead drawn; `kept`: the lane as the pixmap holds it, kept
    /// where the cells under it were not redrawn. Returns the pixels that
    /// changed, if any.
    pub(super) fn paint_wave(
        &mut self,
        plot: &WavePlot,
        kept: Option<KeptWave>,
    ) -> (Option<PixelRect>, Option<KeptWave>) {
        let rect = self.pixel_rect(plot.area);
        let cols = Columns::new(plot.start, plot.end, rect.2);
        let damaged = self.terminal.backend().damage.touches(plot.area);
        let kept = kept.filter(|_| !damaged);
        let (mut target, mut pen) = self.pixmap_and_pen();
        paint(&mut target, &mut pen, plot, cols, rect, kept, true)
    }
}

/// The waveform's levels on `plot`'s lane, where it has them.
fn levels(plot: &WavePlot) -> Option<&waveform::Lane> {
    match plot.lane.stem() {
        None => Some(&plot.waveform.mix),
        Some(i) => plot.waveform.stems.as_ref().map(|s| &s[i]),
    }
}

/// `plot`'s lane into the pixels `rect` of `target`, on the grid `cols`:
/// only the columns that changed since `kept` (the lane as the target
/// holds it, if it does); `head`: the playhead drawn (not where something
/// else draws it over). Returns the pixels that changed, if any, and the
/// lane as the target holds it now.
pub(super) fn paint(
    target: &mut Target,
    pen: &mut Pen,
    plot: &WavePlot,
    cols: Columns,
    (x0, y0, w, h): PixelRect,
    kept: Option<KeptWave>,
    head: bool,
) -> (Option<PixelRect>, Option<KeptWave>) {
    if w == 0 || h == 0 || levels(plot).is_none() {
        return (None, None);
    }
    let line = pen.scale().round().max(1.0) as usize;
    // The whole lane's height, of which the rect shows the pixels from
    // `above` on (while its block slides).
    let ch = pen.cell().1;
    let above = usize::from(plot.rows.0) * ch;
    let full = (usize::from(plot.rows.1) * ch).max(h + above);
    let keys = mask::keys(plot, cols, w, h, line, above, full, head);
    let shape = WaveShape {
        area: plot.area,
        pixels: (x0, y0, w, h),
        lane: plot.lane,
        color: plot.color,
        line,
        above,
        full,
        px: cols.px,
        text: usize::from(plot.text_scale.max(1)),
        rows: plot.lines.rows.max(1),
        key: plot.key,
    };
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
    let words = Words::lay_out(plot, cols, &shape, pen);
    let mut px = target.sub((x0, y0, w, h));
    let (mask, shifted) = mask::to_paint(&mut px, kept.as_ref(), &shape, plot, cols, &keys, &words);
    let dirty: Vec<usize> = (0..w).filter(|&x| mask[x]).collect();
    let changed = match (shifted, dirty.first(), dirty.last()) {
        (true, _, _) => Some((x0, y0, w, h)),
        (false, Some(&a), Some(&b)) => Some((x0 + a, y0, b + 1 - a, h)),
        _ => None,
    };
    paint_columns(&mut px, &dirty, &keys, plot, cols, notes.as_ref(), &shape);
    // The words over the columns just painted (each is painted whole or
    // not at all).
    words.paint(target, pen, &mask, (x0, y0, w, h));
    let kept = KeptWave {
        shape,
        waveform: Arc::clone(&plot.waveform),
        n0: cols.n0,
        keys,
        labels: words.labels,
    };
    (changed, Some(kept))
}

/// The columns `dirty` of a lane: the background its key gives, the RMS
/// filled and the peak outlined, the notes, the lines over them.
fn paint_columns(
    px: &mut Pixels,
    dirty: &[usize],
    keys: &[ColumnKey],
    plot: &WavePlot,
    cols: Columns,
    notes: Option<&NoteRows>,
    shape: &WaveShape,
) {
    let Some(levels) = levels(plot) else {
        return;
    };
    let (line, above, full, h) = (shape.line, shape.above, shape.full, shape.pixels.3);
    let bg = palette::BG;
    let mid = full as f64 / 2.0;
    let curve = lanes::CURVE;
    for &x in dirty {
        let key = keys[x];
        let (t0, t1) = (cols.time(x as f64), cols.time(x as f64 + 1.0));
        let back = match key.back {
            Back::Selected => lanes::SELECT_BG,
            Back::Shown => lanes::SHOWN_BG,
            Back::Outside => lanes::OUTSIDE_BG,
            Back::Transition => lanes::TRANSITION_BG,
            Back::None => bg,
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
        // cursor, the bar being edited, the playhead.
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
            (MARK_PLAYHEAD, palette::ACCENT, 1.0),
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
    let lin = |v: u8| {
        let v = f32::from(v) / 255.0;
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    let enc = |v: f32| {
        let v = v.clamp(0.0, 1.0);
        let s = if v <= 0.003_130_8 {
            12.92 * v
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0).round() as u8
    };
    c.map(|v| enc(lin(v) * 0.12))
}
