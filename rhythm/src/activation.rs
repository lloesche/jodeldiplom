//! Activations: curves that rise where a beat is likely.
//!
//! A [`Flux`] turns a [`Mel`] spectrogram into a [`Curve`] in `0..=1`,
//! frame by frame. The network's beat probability ([`crate::net`]) is
//! another such curve, from its own spectrogram, and goes through the
//! same trackers and the same vote.

use std::borrow::Cow;
use std::collections::VecDeque;

use crate::frontend::{Mel, GAIN};

/// A curve sampled at a fixed frame rate.
#[derive(Clone, Debug, Default)]
pub struct Curve {
    /// One value per frame.
    pub values: Vec<f32>,
    /// Frames per second.
    pub fps: f64,
}

impl Curve {
    /// The largest value within `seconds` of `time`, 0 outside the curve.
    pub fn max_near(&self, time: f64, seconds: f64) -> f32 {
        if self.values.is_empty() {
            return 0.0;
        }
        let last = self.values.len() as f64 - 1.0;
        let lo = ((time - seconds) * self.fps).floor().clamp(0.0, last) as usize;
        let hi = ((time + seconds) * self.fps).ceil().clamp(0.0, last) as usize;
        self.values[lo..=hi].iter().copied().fold(0.0, f32::max)
    }

    /// Whether the curve is all (near) zero.
    pub fn is_silent(&self) -> bool {
        self.values.iter().all(|&v| v < 1e-6)
    }
}

/// Spectral flux over a range of bands: how much the log-mel spectrum
/// rose since a moment ago, summed over the bands.
///
/// Each band is compared with the largest of itself and its two
/// neighbours [`Flux::LAG`] frames earlier, so vibrato and glides that
/// move between neighbouring bands do not count as onsets. The sum loses
/// its slowly moving part (a one-second moving mean), and is scaled by the
/// largest value within three seconds, so quiet passages count as much as
/// loud ones; it is 0 where the music is silent.
///
/// The spectrum is compared as `ln(1 + gain·x)`. The frontend's gain of
/// 1000 makes onsets count alike however loud they are, which the beat
/// trackers want; the cues use 10, since they must tell instruments
/// apart by level (a snare's noise reaches the kick's bands too, only
/// weaker). What counts as silence scales with the gain.
#[derive(Clone, Copy, Debug)]
pub struct Flux {
    name: &'static str,
    low_hz: f64,
    high_hz: f64,
    gain: f32,
}

/// The gain the cues compare the spectrum at.
const CUE_GAIN: f32 = 10.0;

impl Flux {
    /// Frames between the compared spectra (20 ms at 100 frames per
    /// second).
    pub const LAG: usize = 2;
    /// Over all bands.
    pub const FULL: Flux = Flux {
        name: "flux",
        low_hz: 0.0,
        high_hz: f64::INFINITY,
        gain: GAIN,
    };
    /// Over the bands below 250 Hz: kick drums and bass.
    pub const LOW: Flux = Flux {
        name: "flux-low",
        low_hz: 0.0,
        high_hz: 250.0,
        gain: GAIN,
    };
    /// Kick drums, in the drums part: below 150 Hz.
    pub const KICK: Flux = Flux {
        name: "kick",
        low_hz: 0.0,
        high_hz: 150.0,
        gain: CUE_GAIN,
    };
    /// Snare drums and claps, in the drums part: 200 Hz to 5 kHz.
    pub const SNARE: Flux = Flux {
        name: "snare",
        low_hz: 200.0,
        high_hz: 5000.0,
        gain: CUE_GAIN,
    };
    /// Hi-hats and cymbals, in the drums part: above 6 kHz.
    pub const HATS: Flux = Flux {
        name: "hats",
        low_hz: 6000.0,
        high_hz: f64::INFINITY,
        gain: CUE_GAIN,
    };
    /// Bass notes, in the tonal part: below 250 Hz.
    pub const BASS: Flux = Flux {
        name: "bass",
        low_hz: 0.0,
        high_hz: 250.0,
        gain: CUE_GAIN,
    };

    /// The raw flux, before smoothing and scaling.
    pub fn raw(&self, mel: &Mel) -> Vec<f32> {
        let range = mel.bands_between(self.low_hz, self.high_hz);
        let frames = mel.frames();
        // The bands read: the range and a neighbour on each side.
        let first = range.start.saturating_sub(1);
        let width = (range.end + 1).min(mel.bands) - first;
        // Frame t's bands `first..first + width` start at t·stride + offset.
        let (data, stride, offset): (Cow<[f32]>, usize, usize) = if self.gain == GAIN {
            (Cow::Borrowed(&mel.data), mel.bands, first)
        } else {
            let scale = self.gain / GAIN;
            let owned = (0..frames)
                .flat_map(|t| &mel.frame(t)[first..first + width])
                .map(|&v| (scale * v.exp_m1()).ln_1p())
                .collect();
            (Cow::Owned(owned), width, 0)
        };
        let frame = |t: usize| &data[t * stride + offset..t * stride + offset + width];
        let mut out = vec![0.0f32; frames];
        for (t, o) in out.iter_mut().enumerate().skip(Self::LAG) {
            let now = frame(t);
            let before = frame(t - Self::LAG);
            let mut sum = 0.0;
            for b in range.clone().map(|b| b - first) {
                let lo = b.saturating_sub(1);
                let hi = (b + 1).min(width - 1);
                let reference = before[lo..=hi].iter().copied().fold(f32::MIN, f32::max);
                sum += (now[b] - reference).max(0.0);
            }
            *o = sum;
        }
        out
    }

    /// A short name for reports, such as `flux` or `flux-low`.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The activation of `mel`, at its frame rate.
    pub fn curve(&self, mel: &Mel) -> Curve {
        let bands = mel.bands_between(self.low_hz, self.high_hz).len().max(1);
        let silent = SILENT_PER_BAND * self.gain / GAIN * bands as f32;
        normalised(&self.raw(mel), mel.fps, silent)
    }
}

/// Below this (raw flux summed per band, at the frontend's gain) a frame
/// is treated as silent. Quiet sound reads in proportion to the gain.
const SILENT_PER_BAND: f32 = 0.002;

/// `raw` (at `fps`) as a curve in `0..=1`: less its one-second moving mean
/// (negative parts dropped), over the largest value within three seconds
/// (at least 5 % of the peak, and ten times `silent`); 0 throughout where
/// its peak is at most `silent`.
pub(crate) fn normalised(raw: &[f32], fps: f64, silent: f32) -> Curve {
    let mean = moving_mean(raw, (0.5 * fps).round() as usize);
    let lifted: Vec<f32> = raw
        .iter()
        .zip(&mean)
        .map(|(r, m)| (r - m).max(0.0))
        .collect();
    let peak = lifted.iter().copied().fold(0.0, f32::max);
    let floor = (0.05 * peak).max(10.0 * silent);
    let local = sliding_max(&lifted, (3.0 * fps).round() as usize);
    let values = lifted
        .iter()
        .zip(&local)
        .map(|(&v, &m)| {
            if peak <= silent {
                0.0
            } else {
                (v / m.max(floor)).min(1.0)
            }
        })
        .collect();
    Curve { values, fps }
}

/// The mean over `±radius` frames (fewer at the ends).
pub(crate) fn moving_mean(values: &[f32], radius: usize) -> Vec<f32> {
    let mut prefix = Vec::with_capacity(values.len() + 1);
    prefix.push(0.0f64);
    for &v in values {
        prefix.push(prefix.last().unwrap() + f64::from(v));
    }
    (0..values.len())
        .map(|i| {
            let lo = i.saturating_sub(radius);
            let hi = (i + radius + 1).min(values.len());
            ((prefix[hi] - prefix[lo]) / (hi - lo) as f64) as f32
        })
        .collect()
}

/// The maximum over `±radius` frames, in linear time.
pub(crate) fn sliding_max(values: &[f32], radius: usize) -> Vec<f32> {
    let n = values.len();
    let mut out = vec![0.0; n];
    let mut window: VecDeque<usize> = VecDeque::new();
    let mut next = 0;
    for (i, o) in out.iter_mut().enumerate() {
        let hi = (i + radius).min(n.saturating_sub(1));
        while next <= hi && next < n {
            while window.back().is_some_and(|&j| values[j] <= values[next]) {
                window.pop_back();
            }
            window.push_back(next);
            next += 1;
        }
        while window.front().is_some_and(|&j| j + radius < i) {
            window.pop_front();
        }
        *o = window.front().map_or(0.0, |&j| values[j]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_max_and_mean_match_the_direct_ones() {
        let values: Vec<f32> = (0..200).map(|i| ((i * 37) % 23) as f32).collect();
        for radius in [0usize, 1, 5, 40] {
            let max = sliding_max(&values, radius);
            let mean = moving_mean(&values, radius);
            for i in 0..values.len() {
                let lo = i.saturating_sub(radius);
                let hi = (i + radius).min(values.len() - 1);
                let want = values[lo..=hi].iter().copied().fold(f32::MIN, f32::max);
                assert_eq!(max[i], want);
                let avg = values[lo..=hi].iter().sum::<f32>() / (hi - lo + 1) as f32;
                assert!((mean[i] - avg).abs() < 1e-4);
            }
        }
    }
}
