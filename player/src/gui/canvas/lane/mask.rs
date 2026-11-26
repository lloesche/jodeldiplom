//! Which of a lane's columns to paint: each column's key (everything it
//! shows besides the waveform at its place), the kept pixels shifted to
//! this frame's grid, and the columns whose key or words changed, grown to
//! whole words.

use std::sync::Arc;

use super::words::Words;
use super::{Columns, KeptWave, WaveShape};
use crate::gui::pixels::Pixels;
use crate::view::lanes::WavePlot;

/// A Transition lane column's background.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Back {
    #[default]
    None,
    Selected,
    Shown,
    /// The overview outside the span shown: shaded.
    Outside,
    Transition,
}

pub(super) const MARK_CUE: u8 = 1;
pub(super) const MARK_CURSOR: u8 = 2;
pub(super) const MARK_EDIT: u8 = 4;
pub(super) const MARK_PLAYHEAD: u8 = 8;

/// Everything a Transition lane's pixel column shows besides the waveform at
/// its place on the grid: two columns with the same key at the same place are
/// the same pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct ColumnKey {
    pub(super) back: Back,
    pub(super) unclear: bool,
    /// The gain the waveform is faded by (its bits).
    pub(super) gain: u32,
    /// The bar lines through it.
    pub(super) downbeats: u8,
    /// The other lines through it (`MARK_*`).
    pub(super) marks: u8,
    /// The fade's line in it: from row to row.
    pub(super) curve: Option<(usize, usize)>,
}

/// The key of each of a lane's `w` pixel columns.
#[allow(clippy::too_many_arguments)]
pub(super) fn keys(
    plot: &WavePlot,
    cols: Columns,
    w: usize,
    h: usize,
    line: usize,
    above: usize,
    full: usize,
    head: bool,
) -> Vec<ColumnKey> {
    let mut keys = vec![ColumnKey::default(); w];
    // The stretches where the beat is unclear are in order: walked along.
    let mut unclear = plot.unclear.iter().peekable();
    for (x, key) in keys.iter_mut().enumerate() {
        let (t0, t1) = (cols.time(x as f64), cols.time(x as f64 + 1.0));
        let over = |r: Option<(f64, f64)>| r.is_some_and(|(a, b)| t1 > a && t0 < b);
        key.back = if over(plot.selection) {
            Back::Selected
        } else if over(plot.shown) {
            Back::Shown
        } else if over(plot.transition) {
            Back::Transition
        } else if plot.shown.is_some() {
            Back::Outside
        } else {
            Back::None
        };
        let middle = 0.5 * (t0 + t1);
        while unclear.next_if(|&&(_, b)| b <= middle).is_some() {}
        key.unclear = unclear
            .peek()
            .is_some_and(|&&(a, b)| middle >= a && middle < b);
        key.gain = plot.gain_at(middle).to_bits();
    }
    // A line `width` pixels wide from the column `t` falls in, where that
    // column is shown.
    let span = |t: f64, width: usize| {
        let c = cols.of(t);
        if c < 0 || c >= w as i64 {
            return 0..0;
        }
        c as usize..(c as usize + width).min(w)
    };
    for &d in &plot.downbeats {
        for x in span(d, line) {
            keys[x].downbeats = keys[x].downbeats.saturating_add(1);
        }
    }
    for (t, width, bit) in [
        (plot.cue, 2 * line, MARK_CUE),
        (plot.cursor, line, MARK_CURSOR),
        (plot.edit, 2 * line, MARK_EDIT),
        (plot.playhead.filter(|_| head), 2 * line, MARK_PLAYHEAD),
    ] {
        if let Some(t) = t {
            for x in span(t, width) {
                keys[x].marks |= bit;
            }
        }
    }
    if !plot.gain.is_empty() {
        // The fade's row at column `x` (-1: the one before the first, so
        // the line joins it as it would were it shown: a tile, a view
        // scrolled, draw it the same).
        let row_at = |x: f64| {
            let t = cols.time(x + 0.5);
            if !plot.transition.is_some_and(|(a, b)| t >= a && t <= b) {
                return None;
            }
            let g = f64::from(plot.gain_at(t).clamp(0.0, 1.0));
            (((1.0 - g) * (full - 1) as f64).round() as usize)
                .checked_sub(above)
                .filter(|&y| y < h)
        };
        let mut last: Option<usize> = row_at(-1.0);
        for (x, key) in keys.iter_mut().enumerate() {
            let Some(y) = row_at(x as f64) else {
                last = None;
                continue;
            };
            key.curve = Some(last.map_or((y, y), |p| (p.min(y), p.max(y))));
            last = Some(y);
        }
    }
    keys
}

/// The columns of the lane in `px` to paint (by column), and whether its
/// pixels all changed: `old`'s pixels shifted to this frame's grid where
/// they are the same lane's (`shape`, `plot`'s waveform, `cols`), then the
/// columns whose `keys` differ from `old`'s and those under a word gone or
/// new; all of them where there is nothing to keep. A word with a column
/// under it painted is painted whole (its glyphs blended once over clean
/// pixels), its columns with it.
pub(super) fn to_paint(
    px: &mut Pixels,
    old: Option<&KeptWave>,
    shape: &WaveShape,
    plot: &WavePlot,
    cols: Columns,
    keys: &[ColumnKey],
    words: &Words,
) -> (Vec<bool>, bool) {
    let w = keys.len();
    // The columns to paint, after the kept pixels are shifted (which
    // changes them all).
    let mut shifted = true;
    let mut mask: Vec<bool> = match old {
        Some(old)
            if old.shape == *shape
                && Arc::ptr_eq(&old.waveform, &plot.waveform)
                && (cols.n0 - old.n0).unsigned_abs() < w as u64 =>
        {
            let k = cols.n0 - old.n0;
            if k != 0 {
                px.shift(k);
            } else {
                shifted = false;
            }
            let mut mask: Vec<bool> = (0..w)
                .map(|x| {
                    let from = x as i64 + k;
                    from < 0 || from >= w as i64 || old.keys[from as usize] != keys[x]
                })
                .collect();
            // A word gone or changed: its old pixels go; a word new or
            // changed: painted. A word kept stays where the shift put
            // it.
            let mut mark = |(a, b): (i64, i64)| {
                for x in a..b {
                    mask[x as usize] = true;
                }
            };
            for l in old.labels.iter().filter(|l| !words.labels.contains(l)) {
                mark(words.span(l));
            }
            for l in words.labels.iter().filter(|l| !old.labels.contains(l)) {
                mark(words.span(l));
            }
            mask
        }
        _ => vec![true; w],
    };
    loop {
        let mut grew = false;
        for l in &words.labels {
            let (a, b) = words.span(l);
            let range = a as usize..b.max(a) as usize;
            let any = mask[range.clone()].iter().any(|&d| d);
            if any && !mask[range.clone()].iter().all(|&d| d) {
                mask[range].iter_mut().for_each(|d| *d = true);
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    (mask, shifted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::lanes::Lane;
    use crate::core::lanes::WordLines;
    use crate::view::palette;
    use jodel_rhythm::waveform::{self, Waveform};
    use ratatui::layout::Rect;

    /// The overview shades the track outside the span the lanes show; with
    /// the whole track shown, nothing.
    #[test]
    fn the_overview_shades_the_track_outside_the_span_shown() {
        // 100 s over 100 columns, the lanes showing 40 to 60 s.
        let frames = (100.0 * waveform::FPS) as usize;
        let lane = waveform::Lane {
            peak: vec![200; frames],
            rms: vec![150; frames],
        };
        let plot = WavePlot {
            area: Rect::new(0, 0, 100, 1),
            waveform: Arc::new(Waveform {
                mix: lane,
                stems: None,
                melody: None,
            }),
            lane: Lane::Song,
            start: 0.0,
            end: 100.0,
            color: palette::DIM,
            downbeats: Vec::new(),
            unclear: Vec::new(),
            selection: None,
            shown: Some((40.0, 60.0)),
            playhead: None,
            cursor: None,
            cue: None,
            transition: None,
            gain: Vec::new(),
            edit: None,
            rows: (0, 1),
            lyrics: None,
            text_scale: 1,
            lines: WordLines::ONE,
            follow: None,
            deep: false,
            key: None,
        };
        let keys = |plot: &WavePlot| {
            let cols = Columns::new(plot.start, plot.end, 100);
            keys(plot, cols, 100, 4, 1, 0, 4, false)
        };
        let shaded = keys(&plot);
        assert_eq!(shaded[50].back, Back::Shown);
        for x in [5, 30, 70, 95] {
            assert_eq!(shaded[x].back, Back::Outside, "column {x}");
        }
        // The whole track shown: nothing shaded.
        let whole = WavePlot {
            shown: None,
            ..plot
        };
        assert_ne!(keys(&whole)[5].back, Back::Outside);
    }
}
