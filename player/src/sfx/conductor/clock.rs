//! The music clock as the conductor plans by: the song's beat grid at a
//! moment ([`jodel_rhythm::grid::Grid`], from the beat analysis), put on
//! the stream's frames, and the sung lines from then on in those frames.
//! Where a sound lands is the grid's one rule ([`Grid::land`], as
//! [`When`] says), the same a look fired with it lands by.

use jodel_rhythm::grid::{Grid, When};
use jodel_rhythm::Analysis;

/// The beat grid and the singing about a moment, in stream frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Clock {
    /// The grid, its units frames.
    grid: Grid,
    /// The sung lines from about then on, each its first and its end
    /// frame, earliest first (none past the last known).
    lines: [Option<(u64, u64)>; Clock::LINES],
}

impl Clock {
    /// The sung lines a clock knows, from the one sung now (or next) on.
    pub const LINES: usize = 4;

    /// No music: the grid's free tempo at `rate` frames a second, unsure,
    /// nothing sung.
    pub(crate) fn free(rate: u32) -> Clock {
        Clock {
            grid: Grid::free().on(0.0, 0.0, f64::from(rate)),
            lines: [None; Clock::LINES],
        }
    }

    /// The grid of `a` at `t` seconds into the song ([`Analysis::grid_at`]),
    /// heard on stream frame `frame` at `rate` frames a second (the song's
    /// seconds and the stream's frames going on together), the sung lines
    /// from then on `lines`.
    pub fn from_analysis(
        a: &Analysis,
        t: f64,
        frame: u64,
        rate: u32,
        lines: [Option<(u64, u64)>; Clock::LINES],
    ) -> Clock {
        Clock {
            grid: a.grid_at(t).on(t, frame as f64, f64::from(rate)),
            lines,
        }
    }

    /// Frames a beat.
    pub(crate) fn beat_len(&self) -> f64 {
        self.grid.beat_len()
    }

    /// Beats a bar.
    pub(crate) fn bar(&self) -> u32 {
        self.grid.bar()
    }

    /// Beats a minute.
    pub(crate) fn tempo(&self) -> f64 {
        self.grid.tempo()
    }

    /// The first of `per_beat` steps a beat at or after `at`.
    pub(crate) fn next_step(&self, at: u64, per_beat: u32) -> u64 {
        self.grid.next_step(at as f64, per_beat).0.round() as u64
    }

    /// The frame a sound for `at` lands on, by the grid's rule.
    pub fn land(&self, at: u64, when: When) -> u64 {
        let mut sung = [(0.0, 0.0); Clock::LINES];
        let mut n = 0;
        for &(start, end) in self.lines.iter().flatten() {
            sung[n] = (start as f64, end as f64);
            n += 1;
        }
        self.grid.land(at as f64, when, &sung[..n]).round().max(0.0) as u64
    }

    /// Whether no known line is sung in the `length` frames from `from`.
    pub(crate) fn clear(&self, from: u64, length: u64) -> bool {
        let to = from + length;
        !self
            .lines
            .iter()
            .flatten()
            .any(|&(start, end)| start < to && end > from)
    }

    /// The frame of tick `k` (from 0) of a count-up of `of` ticks from
    /// `start`, in the grid's tempo: the first third on eighths, the next
    /// on sixteenths, the last on thirty-seconds, so it speeds up as it
    /// nears the total.
    pub(crate) fn tick(&self, start: u64, k: u32, of: u32) -> u64 {
        let third = of.div_ceil(3).max(1);
        let beats: f64 = (0..k.min(of))
            .map(|i| match i / third {
                0 => 0.5,
                1 => 0.25,
                _ => 0.125,
            })
            .sum();
        start + (beats * self.beat_len()).round() as u64
    }
}
