//! What each of a lane's pixel columns shows besides the waveform at its
//! place: its background, its fade, the bars and the marks through it.

use super::paint::Columns;
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

/// The key of each of a lane's `w` pixel columns, `h` pixels of a lane
/// `full` high from `above` down, its lines `line` wide. The playhead is
/// not among the marks: it is drawn over the lane as it shows.
pub(super) fn keys(
    plot: &WavePlot,
    cols: Columns,
    (w, h): (usize, usize),
    line: usize,
    (above, full): (usize, usize),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::lanes::Lane;
    use crate::core::lanes::WordLines;
    use crate::view::palette;
    use jodel_rhythm::waveform::{self, Waveform};
    use std::sync::Arc;

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
            area: ratatui::layout::Rect::new(0, 0, 100, 1),
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
            keys(plot, cols, (100, 4), 1, (0, 4))
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
