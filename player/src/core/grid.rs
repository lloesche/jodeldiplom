//! A beat grid fixed by hand: the beats and the bars, and the fixes.
//!
//! The analysis finds the beats and where the bars start; where it is
//! wrong, the Mix tab fixes it with these, each a pure function of the
//! grid. The fixed grid replaces the analysis's, and
//! [`Grid::apply`] gives the analysis with it (the tempo and summary
//! worked out again).

use jodel_rhythm::Analysis;

/// Beats to a bar where the fixes make bars afresh.
pub const BAR: usize = 4;

/// Beats (seconds, ascending) and the indices of the beats that start a
/// bar.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Grid {
    /// The beats, in seconds.
    pub beats: Vec<f64>,
    /// The beats that start a bar, ascending.
    pub downbeats: Vec<usize>,
}

impl Grid {
    /// The grid of `analysis`.
    pub fn of(analysis: &Analysis) -> Grid {
        Grid {
            beats: analysis.beats.clone(),
            downbeats: analysis.downbeats.clone(),
        }
    }

    /// `analysis` with this grid, its tempo and summary worked out again.
    pub fn apply(&self, analysis: &Analysis) -> Analysis {
        let mut a = analysis.clone();
        a.beats = self.beats.clone();
        a.downbeats = self.downbeats.clone();
        a.derive();
        a
    }

    /// Bars of [`BAR`] beats with beat `one` starting one, both ways.
    pub fn with_one_at(&self, one: usize) -> Grid {
        let n = self.beats.len();
        Grid {
            beats: self.beats.clone(),
            downbeats: (one % BAR..n).step_by(BAR).collect(),
        }
    }

    /// Every bar starting `by` beats later (earlier when negative). Bars
    /// keep their lengths; where the shift leaves room at the start, bars
    /// of the first one's length fill it.
    pub fn shifted(&self, by: isize) -> Grid {
        let n = self.beats.len() as isize;
        let mut downbeats: Vec<usize> = self
            .downbeats
            .iter()
            .map(|&d| d as isize + by)
            .filter(|&d| (0..n).contains(&d))
            .map(|d| d as usize)
            .collect();
        if let Some(&first) = downbeats.first() {
            let len = downbeats
                .get(1)
                .map_or(BAR, |&second| second - first)
                .max(1);
            let mut all: Vec<usize> = (1..=first / len).map(|k| first - k * len).collect();
            all.reverse();
            all.extend(downbeats);
            downbeats = all;
        }
        Grid {
            beats: self.beats.clone(),
            downbeats,
        }
    }

    /// Every beat `seconds` later (earlier when negative), none before 0.
    pub fn nudged(&self, seconds: f64) -> Grid {
        Grid {
            beats: self.beats.iter().map(|&b| (b + seconds).max(0.0)).collect(),
            downbeats: self.downbeats.clone(),
        }
    }

    /// Half the tempo: every other beat, those on the first bar's beat,
    /// in bars of [`BAR`] from there.
    pub fn halved(&self) -> Grid {
        let first = self.downbeats.first().copied().unwrap_or(0);
        let phase = first % 2;
        let beats: Vec<f64> = self.beats.iter().skip(phase).step_by(2).copied().collect();
        Grid {
            beats,
            downbeats: Vec::new(),
        }
        .with_one_at((first - phase) / 2)
    }

    /// Double the tempo: a beat between each two, in bars of [`BAR`] from
    /// the first bar's beat.
    pub fn doubled(&self) -> Grid {
        let first = self.downbeats.first().copied().unwrap_or(0);
        let mut beats = Vec::with_capacity(self.beats.len() * 2);
        for pair in self.beats.windows(2) {
            beats.push(pair[0]);
            beats.push(0.5 * (pair[0] + pair[1]));
        }
        beats.extend(self.beats.last());
        Grid {
            beats,
            downbeats: Vec::new(),
        }
        .with_one_at(2 * first)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 16 beats at 120 BPM from 1 s, bars of four from beat 1.
    fn grid() -> Grid {
        Grid {
            beats: (0..16).map(|i| 1.0 + 0.5 * f64::from(i)).collect(),
            downbeats: vec![1, 5, 9, 13],
        }
    }

    #[test]
    fn the_one_makes_bars_of_four_both_ways() {
        assert_eq!(grid().with_one_at(6).downbeats, vec![2, 6, 10, 14]);
        assert_eq!(grid().with_one_at(0).downbeats, vec![0, 4, 8, 12]);
    }

    #[test]
    fn shifting_keeps_the_bars_and_fills_the_start() {
        assert_eq!(grid().shifted(1).downbeats, vec![2, 6, 10, 14]);
        assert_eq!(grid().shifted(-1).downbeats, vec![0, 4, 8, 12]);
        // Three later: the first bar starts at 4, and one fits before it.
        assert_eq!(grid().shifted(3).downbeats, vec![0, 4, 8, 12]);
    }

    #[test]
    fn nudging_moves_every_beat() {
        let g = grid().nudged(-0.005);
        assert!((g.beats[0] - 0.995).abs() < 1e-12);
        assert_eq!(g.downbeats, grid().downbeats);
        assert_eq!(grid().nudged(-5.0).beats[0], 0.0);
    }

    #[test]
    fn half_and_double_tempo_keep_the_bar_start() {
        let half = grid().halved();
        // The beats on the first bar's beat (1, 3, 5, ...).
        assert_eq!(half.beats.len(), 8);
        assert!((half.beats[0] - 1.5).abs() < 1e-12);
        assert_eq!(half.downbeats, vec![0, 4]);
        let double = grid().doubled();
        assert_eq!(double.beats.len(), 31);
        assert!((double.beats[1] - 1.25).abs() < 1e-12);
        assert_eq!(double.downbeats, vec![2, 6, 10, 14, 18, 22, 26, 30]);
    }

    #[test]
    fn applying_works_the_tempo_out_again() {
        let analysis = Analysis {
            duration: 10.0,
            ..Analysis::default()
        };
        let a = grid().doubled().apply(&analysis);
        assert_eq!(a.beats.len(), 31);
        assert!((a.local_bpm[5] - 240.0).abs() < 1e-6);
    }
}
