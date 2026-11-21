//! Where a stem plays: how present it is and how steadily it pulses.
//!
//! A separator always gives every stem, even where the music has none:
//! the drums of a string quartet are the bow attacks that leaked into
//! them. A beat tracker finds a grid in anything, so the drum stem's
//! candidate may only vote where the drums are really there and keep a
//! pulse, and the meter trusts the drum and bass cues only where those
//! stems are there. Two curves say where, each `0..=1` frame by frame:
//!
//! - [`present`]: the stem's share of the music's energy over a few
//!   seconds, in decibels, through a logistic curve. Real drums or bass
//!   hold a large share (within 16 dB of the music on the tracks
//!   measured); leakage lies 35 dB or more below it;
//! - [`pulsing`]: how periodic the stem's onsets are around each frame:
//!   the highest peak of their autocorrelation at a beat's lag, against
//!   the onsets' own variance. A groove repeats; leakage and fills do
//!   not.
//!
//! Their product is the stem's [`say`]; the meter uses [`present`] alone.

use crate::activation::{moving_mean, Curve};

/// Seconds on each side over which a stem's share of the energy is taken.
const SHARE_RADIUS: f64 = 2.0;
/// The share, in decibels, at which a stem counts half as present.
const SHARE_MID_DB: f64 = -25.0;
/// How sharply presence rises with the share, in decibels.
const SHARE_WIDTH_DB: f64 = 3.0;
/// Seconds on each side of a frame whose onsets are autocorrelated.
const PULSE_RADIUS: f64 = 4.0;
/// Seconds between the frames the autocorrelation is worked out at (the
/// frames between are interpolated).
const PULSE_STEP: f64 = 0.5;
/// The lags a beat may have, in seconds (240 to 40 BPM).
const PULSE_LAGS: (f64, f64) = (0.25, 1.5);
/// The autocorrelation peak at which a stem counts as half pulsing.
const PULSE_MID: f64 = 0.4;
/// How sharply that rises with the peak.
const PULSE_WIDTH: f64 = 0.05;
/// The say at which a stem's candidate keeps its beats.
pub const KEEP: f32 = 0.5;

/// The logistic function of `(x − mid) / width`.
fn logistic(x: f64, mid: f64, width: f64) -> f32 {
    (1.0 / (1.0 + (-(x - mid) / width).exp())) as f32
}

/// Energy frame by frame: a stem's and the whole music's.
#[derive(Clone, Debug, Default)]
pub struct Levels {
    /// Frames per second.
    pub fps: f64,
    /// The drum stem's energy in each frame.
    pub drums: Vec<f32>,
    /// The bass stem's.
    pub bass: Vec<f32>,
    /// All stems' together (the music).
    pub total: Vec<f32>,
}

/// How present a stem with energy `stem` is in music with energy `total`
/// (both per frame, at `fps`): its share over two seconds on each
/// side, through a logistic curve in decibels.
pub fn present(stem: &[f32], total: &[f32], fps: f64) -> Curve {
    let radius = (SHARE_RADIUS * fps).round() as usize;
    let stem = moving_mean(stem, radius);
    let total = moving_mean(total, radius);
    let values = stem
        .iter()
        .zip(&total)
        .map(|(&s, &t)| {
            if t <= 0.0 || s <= 0.0 {
                return 0.0;
            }
            let db = 10.0 * f64::from(s / t).log10();
            logistic(db, SHARE_MID_DB, SHARE_WIDTH_DB)
        })
        .collect();
    Curve { values, fps }
}

/// The highest normalised autocorrelation of `values[lo..hi]` at the
/// lags `min..=max`, 0 where they do not vary.
fn periodicity(values: &[f32], lo: usize, hi: usize, min: usize, max: usize) -> f64 {
    let x = &values[lo..hi];
    if x.len() <= max {
        return 0.0;
    }
    let mean = x.iter().map(|&v| f64::from(v)).sum::<f64>() / x.len() as f64;
    let d: Vec<f64> = x.iter().map(|&v| f64::from(v) - mean).collect();
    let energy: f64 = d.iter().map(|v| v * v).sum();
    if energy <= 1e-9 {
        return 0.0;
    }
    (min..=max)
        .map(|lag| {
            let sum: f64 = d.iter().zip(&d[lag..]).map(|(a, b)| a * b).sum();
            // Scaled up for the products a lag leaves out.
            sum * d.len() as f64 / (d.len() - lag) as f64 / energy
        })
        .fold(0.0, f64::max)
}

/// How steadily `onsets` pulse around each frame: their autocorrelation
/// peak over four seconds on each side, at a beat's lags, through a
/// logistic curve.
pub fn pulsing(onsets: &Curve) -> Curve {
    let fps = onsets.fps;
    let n = onsets.values.len();
    let radius = (PULSE_RADIUS * fps).round() as usize;
    let step = ((PULSE_STEP * fps).round() as usize).max(1);
    let (min, max) = (
        (PULSE_LAGS.0 * fps).round() as usize,
        (PULSE_LAGS.1 * fps).round() as usize,
    );
    let points: Vec<(usize, f32)> = (0..n)
        .step_by(step)
        .chain(std::iter::once(n.saturating_sub(1)))
        .map(|t| {
            let (lo, hi) = (t.saturating_sub(radius), (t + radius + 1).min(n));
            let r = periodicity(&onsets.values, lo, hi, min, max);
            (t, logistic(r, PULSE_MID, PULSE_WIDTH))
        })
        .collect();
    let mut values = vec![0.0f32; n];
    for pair in points.windows(2) {
        let ((a, va), (b, vb)) = (pair[0], pair[1]);
        for (t, v) in values.iter_mut().enumerate().take(b + 1).skip(a) {
            let w = if b > a {
                (t - a) as f32 / (b - a) as f32
            } else {
                0.0
            };
            *v = va + (vb - va) * w;
        }
    }
    if let [(_, v)] = points[..] {
        values.fill(v);
    }
    Curve { values, fps }
}

/// A stem's say: how present it is times how steadily it pulses, frame by
/// frame (at `present`'s rate; `pulsing` is read at the nearest frame).
pub fn say(present: &Curve, pulsing: &Curve) -> Curve {
    let values = present
        .values
        .iter()
        .enumerate()
        .map(|(t, &p)| {
            let time = t as f64 / present.fps;
            let u = ((time * pulsing.fps).round() as usize).min(pulsing.values.len().max(1) - 1);
            p * pulsing.values.get(u).copied().unwrap_or(0.0)
        })
        .collect();
    Curve {
        values,
        fps: present.fps,
    }
}

/// The value of `curve` at `time` (the nearest frame; 0 outside it).
pub fn at(curve: &Curve, time: f64) -> f32 {
    if time < 0.0 {
        return 0.0;
    }
    let t = (time * curve.fps).round() as usize;
    curve.values.get(t).copied().unwrap_or(0.0)
}

/// The mean of `curve` over `start..=end` seconds (0 outside it).
pub fn mean_between(curve: &Curve, start: f64, end: f64) -> f64 {
    if curve.values.is_empty() {
        return 0.0;
    }
    let last = curve.values.len() - 1;
    let lo = (start * curve.fps).floor().clamp(0.0, last as f64) as usize;
    let hi = (end * curve.fps).ceil().clamp(0.0, last as f64) as usize;
    if hi < lo {
        return 0.0;
    }
    let v = &curve.values[lo..=hi];
    v.iter().map(|&x| f64::from(x)).sum::<f64>() / v.len() as f64
}

/// The stretches, in seconds, where `curve` is at least `level`.
pub fn spans(curve: &Curve, level: f32) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    let mut start = None;
    for (t, &v) in curve.values.iter().enumerate() {
        match (v >= level, start) {
            (true, None) => start = Some(t),
            (false, Some(s)) => {
                out.push((s as f64 / curve.fps, (t - 1) as f64 / curve.fps));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push((
            s as f64 / curve.fps,
            (curve.values.len() - 1) as f64 / curve.fps,
        ));
    }
    out
}

/// `curve` times `scale` frame by frame (`scale` read at the nearest
/// frame of its own rate).
pub fn scaled(curve: &Curve, scale: &Curve) -> Curve {
    let values = curve
        .values
        .iter()
        .enumerate()
        .map(|(t, &v)| v * at(scale, t as f64 / curve.fps))
        .collect();
    Curve {
        values,
        fps: curve.fps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FPS: f64 = 100.0;

    /// Onsets every `period` seconds for `seconds`.
    fn clicks(period: f64, seconds: f64) -> Curve {
        let n = (seconds * FPS) as usize;
        let mut values = vec![0.0; n];
        let mut t = 0.0;
        while t < seconds {
            let i = (t * FPS).round() as usize;
            if i < n {
                values[i] = 1.0;
            }
            t += period;
        }
        Curve { values, fps: FPS }
    }

    #[test]
    fn a_loud_stem_is_present_and_leakage_is_not() {
        let total = vec![1.0f32; 1000];
        let loud = present(&vec![0.3f32; 1000], &total, FPS);
        let leak = present(&vec![0.0001f32; 1000], &total, FPS);
        assert!(loud.values.iter().all(|&v| v > 0.95));
        assert!(leak.values.iter().all(|&v| v < 0.05));
    }

    #[test]
    fn steady_onsets_pulse_and_scattered_ones_do_not() {
        let steady = pulsing(&clicks(0.5, 20.0));
        assert!(steady.values[500..1500].iter().all(|&v| v > 0.9));
        // Onsets at scattered times: a fixed pseudo-random sequence.
        let mut values = vec![0.0f32; 2000];
        let mut x: u32 = 12345;
        for _ in 0..40 {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
            values[(x >> 8) as usize % 2000] = 1.0;
        }
        let scattered = pulsing(&Curve { values, fps: FPS });
        let mean = mean_between(&scattered, 5.0, 15.0);
        assert!(mean < 0.3, "{mean}");
    }

    #[test]
    fn spans_cover_the_runs_at_the_level() {
        let curve = Curve {
            values: vec![0.0, 0.6, 0.7, 0.2, 0.9, 0.9],
            fps: 1.0,
        };
        assert_eq!(spans(&curve, 0.5), vec![(1.0, 2.0), (4.0, 5.0)]);
    }
}
