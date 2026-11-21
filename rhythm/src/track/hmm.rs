//! A beat tracker that decodes tempo and beat position jointly.
//!
//! The hidden state is a beat interval (in frames, from a set of tempi
//! spaced evenly on a log scale) and the position within it. Each frame
//! the position advances by one; after the last position comes position 0
//! of any interval, the next beat. Moving from interval `a` to `b` has
//! probability proportional to `exp(−λ·|b/a − 1|)`, so tempo follows smooth
//! drift cheaply and jumps at a price that grows with their size.
//!
//! A state "is a beat" in the first sixteenth of its interval. There, the
//! activation `a` is the likelihood; elsewhere `(1 − a) / 15`, so a strong
//! onset away from the beat costs as much as a missing one on it. The
//! Viterbi path through these states gives the beats: its frames at
//! position 0, each moved to the activation's peak within the beat
//! region and refined between frames.
//!
//! Only position 0 of each interval has a choice of predecessor, so the
//! path is kept as one byte per interval per frame.

use super::{peak_in, Tracker};
use crate::activation::Curve;

/// The HMM tracker's settings.
#[derive(Clone, Copy, Debug)]
pub struct Hmm {
    /// The slowest tempo, in beats per minute.
    pub min_bpm: f64,
    /// The fastest tempo.
    pub max_bpm: f64,
    /// Tempi between them, spaced on a log scale (fewer where they round
    /// to the same interval).
    pub tempi: usize,
    /// How strongly tempo changes are held back.
    pub lambda: f64,
    /// The beat region is the first `1 / observation` of an interval.
    pub observation: f64,
}

impl Default for Hmm {
    fn default() -> Hmm {
        Hmm {
            min_bpm: 35.0,
            max_bpm: 260.0,
            tempi: 100,
            lambda: 100.0,
            observation: 16.0,
        }
    }
}

/// The smallest likelihood an observation may have.
const FLOOR: f32 = 0.02;

impl Tracker for Hmm {
    fn name(&self) -> &'static str {
        "hmm"
    }

    fn track(&self, curve: &Curve) -> Vec<f64> {
        let fps = curve.fps;
        let values = &curve.values;
        let n = values.len();
        if n < 2 || curve.is_silent() {
            return Vec::new();
        }
        let intervals = self.intervals(fps);
        let k_count = intervals.len();
        let offsets: Vec<usize> = intervals
            .iter()
            .scan(0, |sum, &l| {
                let at = *sum;
                *sum += l;
                Some(at)
            })
            .collect();
        let states: usize = intervals.iter().sum();
        let regions: Vec<usize> = intervals
            .iter()
            .map(|&l| ((l as f64 / self.observation).ceil() as usize).max(1))
            .collect();
        // log P(to | from), each row normalised.
        let mut transition = vec![0.0f32; k_count * k_count];
        for (from, &a) in intervals.iter().enumerate() {
            let row = &mut transition[from * k_count..(from + 1) * k_count];
            for (to, &b) in intervals.iter().enumerate() {
                row[to] = (-self.lambda * (b as f64 / a as f64 - 1.0).abs()) as f32;
            }
            let max = row.iter().copied().fold(f32::MIN, f32::max);
            let total: f32 = row.iter().map(|&v| (v - max).exp()).sum();
            let norm = max + total.ln();
            for v in row.iter_mut() {
                *v -= norm;
            }
        }
        let beat_log: Vec<f32> = values.iter().map(|&a| a.max(FLOOR).ln()).collect();
        let off_log: Vec<f32> = values
            .iter()
            .map(|&a| (((1.0 - a) / (self.observation as f32 - 1.0)).max(FLOOR / 15.0)).ln())
            .collect();

        let mut score = vec![0.0f32; states];
        let mut starts = vec![0.0f32; k_count];
        let mut back = vec![0u8; n * k_count];
        for t in 0..n {
            if t > 0 {
                for (to, start) in starts.iter_mut().enumerate() {
                    let mut best = f32::MIN;
                    let mut from_best = 0;
                    for from in 0..k_count {
                        let last = score[offsets[from] + intervals[from] - 1];
                        let v = last + transition[from * k_count + to];
                        if v > best {
                            best = v;
                            from_best = from;
                        }
                    }
                    *start = best;
                    back[t * k_count + to] = from_best as u8;
                }
                for k in 0..k_count {
                    let (o, l) = (offsets[k], intervals[k]);
                    score.copy_within(o..o + l - 1, o + 1);
                    score[o] = starts[k];
                }
            }
            let (on, off) = (beat_log[t], off_log[t]);
            for k in 0..k_count {
                let (o, l, r) = (offsets[k], intervals[k], regions[k]);
                for s in &mut score[o..o + r] {
                    *s += on;
                }
                for s in &mut score[o + r..o + l] {
                    *s += off;
                }
            }
            if t % 64 == 63 {
                let max = score.iter().copied().fold(f32::MIN, f32::max);
                for s in &mut score {
                    *s -= max;
                }
            }
        }

        // Back through the path from the best final state.
        let best = (0..states)
            .max_by(|&a, &b| score[a].total_cmp(&score[b]))
            .unwrap_or(0);
        let mut k = offsets.partition_point(|&o| o <= best) - 1;
        let mut p = best - offsets[k];
        let mut beats: Vec<(usize, usize)> = Vec::new();
        for t in (0..n).rev() {
            if p == 0 {
                beats.push((t, k));
                if t > 0 {
                    k = back[t * k_count + k] as usize;
                    p = intervals[k] - 1;
                }
            } else {
                p -= 1;
            }
        }
        beats.reverse();
        beats
            .into_iter()
            .filter_map(|(t, k)| peak_in(values, t, t + regions[k]))
            .map(|frame| frame / fps)
            .collect()
    }
}

impl Hmm {
    /// The beat intervals in frames at `fps`, ascending, without repeats.
    fn intervals(&self, fps: f64) -> Vec<usize> {
        let count = self.tempi.clamp(2, 255);
        let mut out: Vec<usize> = (0..count)
            .map(|i| {
                let bpm = self.min_bpm
                    * (self.max_bpm / self.min_bpm).powf(i as f64 / (count - 1) as f64);
                ((60.0 * fps / bpm).round() as usize).max(2)
            })
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pulses(fps: f64, seconds: f64, beats: &[f64]) -> Curve {
        let mut values = vec![0.0f32; (seconds * fps) as usize];
        for &b in beats {
            let f = (b * fps).round() as usize;
            if f < values.len() {
                values[f] = 1.0;
                if f + 1 < values.len() {
                    values[f + 1] = 0.4;
                }
            }
        }
        Curve { values, fps }
    }

    #[test]
    fn follows_a_pulse_train_and_a_tempo_jump() {
        let fps = 100.0;
        let mut truth: Vec<f64> = (0..40).map(|i| 1.0 + i as f64 * 0.5).collect(); // 120
        let last = *truth.last().unwrap();
        truth.extend((1..40).map(|i| last + i as f64 * 0.4)); // 150
        let seconds = truth.last().unwrap() + 1.0;
        let beats = Hmm::default().track(&pulses(fps, seconds, &truth));
        let f = crate::metrics::f_measure(&beats, &truth, 0.07);
        assert!(f > 0.97, "F {f}, {} beats", beats.len());
    }

    #[test]
    fn silence_has_no_beats() {
        let curve = Curve {
            values: vec![0.0; 1000],
            fps: 100.0,
        };
        assert!(Hmm::default().track(&curve).is_empty());
    }
}
