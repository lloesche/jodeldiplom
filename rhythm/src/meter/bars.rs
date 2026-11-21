//! Downbeats: which beats start a bar.
//!
//! Each beat gets an accent, scaled to zero mean and unit spread over the
//! track: the kick drum at the beat (it tends to land on the one) and the
//! chord change there (chords change at the bar line), plus the network's
//! downbeat probability when it ran, which counts for most. With stems,
//! the kick counts as far as the drums are present at the beat, after
//! the scaling (which would otherwise lift leakage into an absent drum
//! stem back up to full strength). A Viterbi
//! path over the states "position p in a bar of 3" and "position p in a
//! bar of 4" collects the accents of the beats it puts first in a bar;
//! the meter may change between bars at a price, so it changes only where
//! the music does.

use crate::activation::Curve;
use crate::cues::Cues;

/// The cost of changing the meter.
const SWITCH: f64 = 6.0;
/// The head start of four beats to the bar, the most common meter.
const FOUR_FIRST: f64 = 0.5;
/// How far from a beat its cues are read, in seconds.
const NEAR: f64 = 0.04;
/// The weight of the kick and the chord change against the network's
/// downbeat probability, when it ran.
const CUES_WITH_NETWORK: f64 = 0.25;

/// The indices of the beats that start a bar; empty for fewer than eight
/// beats. `network` is the network's downbeat probability, when it ran.
pub fn downbeats(beats: &[f64], cues: &Cues, network: Option<&Curve>) -> Vec<usize> {
    if beats.len() < 8 {
        return Vec::new();
    }
    let at = |curve: &Curve| -> Vec<f64> {
        let near: Vec<f64> = beats
            .iter()
            .map(|&b| f64::from(curve.max_near(b, NEAR)))
            .collect();
        standardized(&near)
    };
    let mut kick = at(&cues.kick);
    if let Some(present) = &cues.drums_present {
        for (k, &b) in kick.iter_mut().zip(beats) {
            *k *= f64::from(crate::presence::at(present, b));
        }
    }
    let change = at(&cues.change);
    let accent: Vec<f64> = match network {
        Some(curve) => at(curve)
            .iter()
            .zip(kick.iter().zip(&change))
            .map(|(n, (k, c))| n + CUES_WITH_NETWORK * (k + c))
            .collect(),
        None => kick.iter().zip(&change).map(|(k, c)| k + c).collect(),
    };
    bars(&accent)
}

/// The beats that start a bar, from each beat's accent.
fn bars(accent: &[f64]) -> Vec<usize> {
    let n = accent.len();
    // States: 0..3 are positions in a bar of 3, 3..7 in a bar of 4.
    let meter = |s: usize| if s < 3 { 3 } else { 4 };
    let position = |s: usize| if s < 3 { s } else { s - 3 };
    let first = |m: usize| if m == 3 { 0 } else { 3 };
    let mut score: Vec<f64> = (0..7)
        .map(|s| if meter(s) == 4 { FOUR_FIRST } else { 0.0 })
        .collect();
    let mut back = vec![[0usize; 7]; n];
    for (i, &a) in accent.iter().enumerate() {
        let mut next = [f64::MIN; 7];
        if i == 0 {
            next.copy_from_slice(&score);
        } else {
            for (s, &from) in score.iter().enumerate() {
                let (m, p) = (meter(s), position(s));
                let moves: &[(usize, f64)] = if p + 1 < m {
                    &[(s + 1, 0.0)]
                } else {
                    &[(first(m), 0.0), (first(7 - m), -SWITCH)]
                };
                for &(to, cost) in moves {
                    let v = from + cost;
                    if v > next[to] {
                        next[to] = v;
                        back[i][to] = s;
                    }
                }
            }
        }
        for (s, v) in next.iter_mut().enumerate() {
            if position(s) == 0 {
                *v += a;
            }
        }
        score = next.to_vec();
    }
    let mut s = (0..7)
        .max_by(|&a, &b| score[a].total_cmp(&score[b]))
        .unwrap_or(3);
    let mut out = Vec::new();
    for i in (0..n).rev() {
        if position(s) == 0 {
            out.push(i);
        }
        s = back[i][s];
    }
    out.reverse();
    out
}

/// `values` shifted and scaled to mean 0 and standard deviation 1 (all 0
/// when they do not vary).
fn standardized(values: &[f64]) -> Vec<f64> {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let spread = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
    if spread < 1e-12 {
        return vec![0.0; values.len()];
    }
    values.iter().map(|v| (v - mean) / spread).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accents_every_three_or_four_beats_make_the_bars() {
        let accent = |meter: usize| -> Vec<f64> {
            (0..48)
                .map(|i| if i % meter == 1 { 1.0 } else { 0.0 })
                .collect()
        };
        assert_eq!(bars(&accent(4)), (1..48).step_by(4).collect::<Vec<_>>());
        assert_eq!(bars(&accent(3)), (1..48).step_by(3).collect::<Vec<_>>());
    }
}
