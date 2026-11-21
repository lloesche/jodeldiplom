//! Predominant local pulse: beats from the locally best-fitting sinusoid.
//!
//! Around every hop (a tenth of a second) a Hann window of six seconds
//! measures how strongly the activation repeats at each tempo: the
//! magnitude of its Fourier coefficient there (a tempogram). The tempo
//! of each hop is chosen along a path that maximises the magnitudes while
//! paying for tempo changes in octaves, so the path does not flicker
//! between tempo levels that fit about as well. The windowed cosine of
//! that tempo, at the phase the coefficient gives, is added into a pulse
//! curve; the curve's positive peaks are the beats. The pulse follows
//! local tempo by construction, gradual or sudden, within about half a
//! window.

use std::f64::consts::PI;

use super::{refine, Tracker};
use crate::activation::{sliding_max, Curve};
use crate::fft::Complex;

/// The PLP tracker's settings.
#[derive(Clone, Copy, Debug)]
pub struct Plp {
    /// The slowest tempo, in beats per minute.
    pub min_bpm: f64,
    /// The fastest tempo.
    pub max_bpm: f64,
    /// Tempi between them, spaced on a log scale.
    pub tempi: usize,
    /// The analysis window, in seconds.
    pub window: f64,
    /// Seconds between analysis windows.
    pub hop: f64,
    /// The cost of a tempo change of one octave, against magnitudes
    /// scaled to 1 at each hop's best tempo.
    pub change_cost: f64,
}

impl Default for Plp {
    fn default() -> Plp {
        Plp {
            min_bpm: 35.0,
            max_bpm: 260.0,
            tempi: 120,
            window: 6.0,
            hop: 0.1,
            change_cost: 0.5,
        }
    }
}

impl Tracker for Plp {
    fn name(&self) -> &'static str {
        "plp"
    }

    fn track(&self, curve: &Curve) -> Vec<f64> {
        let fps = curve.fps;
        let values = &curve.values;
        let n = values.len();
        if n < 2 || curve.is_silent() {
            return Vec::new();
        }
        let size = ((self.window * fps).round() as usize).max(4) & !1;
        let half = size / 2;
        let hop = ((self.hop * fps).round() as usize).max(1);
        let window: Vec<f64> = (0..size)
            .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / size as f64).cos())
            .collect();
        let count = self.tempi.max(2);
        let tempi: Vec<f64> = (0..count)
            .map(|i| {
                self.min_bpm * (self.max_bpm / self.min_bpm).powf(i as f64 / (count - 1) as f64)
            })
            .collect();
        // Cycles per frame of each tempo.
        let omegas: Vec<f64> = tempi.iter().map(|bpm| bpm / 60.0 / fps).collect();
        // The window times the conjugate phasor, relative to the centre.
        let kernels: Vec<Vec<Complex>> = omegas
            .iter()
            .map(|&w| {
                (0..size)
                    .map(|r| {
                        let m = r as f64 - half as f64;
                        Complex::unit(-2.0 * PI * w * m).scale(window[r] as f32)
                    })
                    .collect()
            })
            .collect();

        let centres: Vec<usize> = (0..n).step_by(hop).collect();
        let mut coefficients = vec![Complex::default(); centres.len() * count];
        for (h, &c) in centres.iter().enumerate() {
            let lo = c.saturating_sub(half);
            let hi = (c + half).min(n);
            for (ti, kernel) in kernels.iter().enumerate() {
                let mut sum = Complex::default();
                for m in lo..hi {
                    let v = values[m];
                    if v != 0.0 {
                        sum = sum.add(kernel[m + half - c].scale(v));
                    }
                }
                coefficients[h * count + ti] = sum;
            }
        }

        // The tempo path.
        let octaves: Vec<f64> = tempi.iter().map(|t| t.log2()).collect();
        let mut score = vec![0.0f64; count];
        let mut back = vec![0u16; centres.len() * count];
        let mut next = vec![0.0f64; count];
        for h in 0..centres.len() {
            let row = &coefficients[h * count..(h + 1) * count];
            let peak = row.iter().map(|c| c.norm()).fold(0.0f32, f32::max);
            let scale = if peak > 0.0 {
                1.0 / f64::from(peak)
            } else {
                0.0
            };
            for to in 0..count {
                let (mut best, mut arg) = (f64::MIN, to);
                if h > 0 {
                    for from in 0..count {
                        let v =
                            score[from] - self.change_cost * (octaves[to] - octaves[from]).abs();
                        if v > best {
                            best = v;
                            arg = from;
                        }
                    }
                } else {
                    best = 0.0;
                }
                next[to] = best + f64::from(row[to].norm()) * scale;
                back[h * count + to] = arg as u16;
            }
            std::mem::swap(&mut score, &mut next);
        }
        let mut chosen = vec![0usize; centres.len()];
        let mut k = (0..count)
            .max_by(|&a, &b| score[a].total_cmp(&score[b]))
            .unwrap_or(0);
        for h in (0..centres.len()).rev() {
            chosen[h] = k;
            k = back[h * count + k] as usize;
        }

        // Add up the chosen sinusoids.
        let mut pulse = vec![0.0f32; n];
        let mut strength = vec![0.0f32; n];
        for (h, &c) in centres.iter().enumerate() {
            let ti = chosen[h];
            let coefficient = coefficients[h * count + ti];
            let magnitude = coefficient.norm();
            if magnitude <= 0.0 {
                continue;
            }
            let phase = f64::from(coefficient.im).atan2(f64::from(coefficient.re));
            let w = omegas[ti];
            let lo = c.saturating_sub(half);
            let hi = (c + half).min(n);
            for m in lo..hi {
                let x = m as f64 - c as f64;
                let k = window[m + half - c] * (2.0 * PI * w * x + phase).cos();
                pulse[m] += k as f32;
                strength[m] = strength[m].max(magnitude);
            }
        }
        for p in &mut pulse {
            *p = p.max(0.0);
        }
        // Peaks of the pulse where the music carries one.
        let local = sliding_max(&pulse, (2.0 * fps).round() as usize);
        let loudest = strength.iter().copied().fold(0.0, f32::max);
        let mut beats = Vec::new();
        for m in 1..n.saturating_sub(1) {
            if pulse[m] > pulse[m - 1]
                && pulse[m] >= pulse[m + 1]
                && pulse[m] > 0.1 * local[m]
                && strength[m] > 0.02 * loudest
            {
                beats.push(refine(&pulse, m) / fps);
            }
        }
        beats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peaks_sit_on_a_pulse_train_that_speeds_up() {
        let fps: f64 = 100.0;
        let mut truth: Vec<f64> = Vec::new();
        let mut t = 1.0;
        while t < 40.0 {
            truth.push(t);
            let bpm = 100.0 + 20.0 * (t / 40.0);
            t += 60.0 / bpm;
        }
        let mut values = vec![0.0f32; 4100];
        for &b in &truth {
            let f = (b * fps).round() as usize;
            values[f] = 1.0;
            values[f + 1] = 0.5;
        }
        let beats = Plp::default().track(&Curve { values, fps });
        let f = crate::metrics::f_measure(&beats, &truth, 0.07);
        assert!(f > 0.95, "F {f}");
    }
}
