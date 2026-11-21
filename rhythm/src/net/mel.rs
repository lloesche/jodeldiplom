//! The network's input: a log-mel spectrogram at 50 frames a second.
//!
//! Frames of 1024 samples (periodic Hann window) every 441 samples, the
//! first centred on sample 0 with the signal mirrored at both ends; the
//! magnitude spectrum divided by √1024; 128 triangular mel bands from
//! 30 Hz to 11 kHz on the Slaney mel scale, not area-normalised; then
//! ln(1 + 1000·x). This is the spectrogram the network was trained on.

use std::f64::consts::PI;

use crate::fft::RealFft;
use crate::SAMPLE_RATE;

/// Samples per frame.
pub const FRAME: usize = 1024;
/// Samples between frames.
pub const HOP: usize = 441;
/// Mel bands.
pub const BANDS: usize = 128;
/// Frames per second.
pub const FPS: f64 = SAMPLE_RATE as f64 / HOP as f64;
const LOW_HZ: f64 = 30.0;
const HIGH_HZ: f64 = 11_000.0;

/// The spectrogram: `frames` rows of [`BANDS`] values.
#[derive(Clone, Debug, Default)]
pub struct NetMel {
    /// How many frames.
    pub frames: usize,
    /// Frame-major values.
    pub data: Vec<f32>,
}

/// Takes the samples as they come and keeps only what the frames still
/// to be computed need.
pub struct NetFrontend {
    fft: RealFft,
    window: Vec<f32>,
    /// `(bin, weight)` per band.
    filters: Vec<Vec<(usize, f32)>>,
    /// The first samples, for the mirror at the start.
    head: Vec<f32>,
    /// Samples from `base` on.
    buffer: Vec<f32>,
    base: usize,
    next: usize,
    frame: Vec<f32>,
    spectrum: Vec<f32>,
    out: NetMel,
}

impl Default for NetFrontend {
    fn default() -> NetFrontend {
        NetFrontend::new()
    }
}

/// Hertz to mel on the Slaney scale: linear below 1 kHz, logarithmic
/// above.
fn hz_to_mel(hz: f64) -> f64 {
    let step = 6.4f64.ln() / 27.0;
    if hz >= 1000.0 {
        15.0 + (hz / 1000.0).ln() / step
    } else {
        hz * 3.0 / 200.0
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    let step = 6.4f64.ln() / 27.0;
    if mel >= 15.0 {
        1000.0 * (step * (mel - 15.0)).exp()
    } else {
        mel * 200.0 / 3.0
    }
}

/// The band filters: `(bin, weight)` pairs, each band a triangle from its
/// lower to its upper neighbour's centre, peaking at 1.
pub(crate) fn filters() -> Vec<Vec<(usize, f32)>> {
    let bins = FRAME / 2 + 1;
    let nyquist = f64::from(SAMPLE_RATE / 2);
    let (low, high) = (hz_to_mel(LOW_HZ), hz_to_mel(HIGH_HZ));
    let points: Vec<f64> = (0..BANDS + 2)
        .map(|i| mel_to_hz(low + (high - low) * i as f64 / (BANDS + 1) as f64))
        .collect();
    (0..BANDS)
        .map(|b| {
            let (lo, mid, hi) = (points[b], points[b + 1], points[b + 2]);
            (0..bins)
                .filter_map(|k| {
                    let hz = nyquist * k as f64 / (bins - 1) as f64;
                    let w = ((hz - lo) / (mid - lo)).min((hi - hz) / (hi - mid));
                    (w > 0.0).then_some((k, w as f32))
                })
                .collect()
        })
        .collect()
}

impl NetFrontend {
    /// A front end for one track.
    pub fn new() -> NetFrontend {
        NetFrontend {
            fft: RealFft::new(FRAME),
            window: (0..FRAME)
                .map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / FRAME as f64).cos()) as f32)
                .collect(),
            filters: filters(),
            head: Vec::with_capacity(FRAME / 2 + 1),
            buffer: Vec::new(),
            base: 0,
            next: 0,
            frame: vec![0.0; FRAME],
            spectrum: vec![0.0; FRAME / 2 + 1],
            out: NetMel::default(),
        }
    }

    /// Takes in more samples.
    pub fn push(&mut self, samples: &[f32]) {
        let room = (FRAME / 2 + 1).saturating_sub(self.head.len());
        self.head
            .extend_from_slice(&samples[..room.min(samples.len())]);
        self.buffer.extend_from_slice(samples);
        let total = self.base + self.buffer.len();
        while self.next * HOP + FRAME / 2 <= total {
            self.compute(None);
        }
        // Keep what the next frame (and a mirror at the end) may need.
        let keep_from = (self.next * HOP).saturating_sub(FRAME + FRAME / 2);
        if keep_from > self.base + FRAME * 8 {
            self.buffer.drain(..keep_from - self.base);
            self.base = keep_from;
        }
    }

    /// The spectrogram of everything pushed: `1 + samples / HOP` frames.
    pub fn finish(mut self) -> NetMel {
        let len = self.base + self.buffer.len();
        if len == 0 {
            return NetMel::default();
        }
        let frames = 1 + len / HOP;
        while self.next < frames {
            self.compute(Some(len));
        }
        self.out
    }

    /// Sample `i`, mirrored at the start and, once the length is known,
    /// at the end.
    fn sample(&self, i: isize, len: Option<usize>) -> f32 {
        let mut i = i.unsigned_abs();
        if let Some(len) = len {
            let last = len - 1;
            if i > last {
                // Mirror at the end; very short tracks bounce to the start.
                i = (2 * last).saturating_sub(i);
            }
        }
        if i < self.base {
            self.head.get(i).copied().unwrap_or(0.0)
        } else {
            self.buffer.get(i - self.base).copied().unwrap_or(0.0)
        }
    }

    fn compute(&mut self, len: Option<usize>) {
        let start = (self.next * HOP) as isize - (FRAME / 2) as isize;
        for j in 0..FRAME {
            self.frame[j] = self.sample(start + j as isize, len) * self.window[j];
        }
        self.fft.magnitudes(&self.frame, &mut self.spectrum);
        let norm = 1.0 / (FRAME as f32).sqrt();
        for band in &self.filters {
            let sum: f32 = band.iter().map(|&(k, w)| self.spectrum[k] * w).sum();
            self.out.data.push((1000.0 * sum * norm).ln_1p());
        }
        self.out.frames += 1;
        self.next += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bands_cover_thirty_hertz_to_eleven_kilohertz() {
        let f = filters();
        assert_eq!(f.len(), BANDS);
        let bin_hz = f64::from(SAMPLE_RATE) / FRAME as f64;
        let first = f[0].first().unwrap().0 as f64 * bin_hz;
        let last = f[BANDS - 1].last().unwrap().0 as f64 * bin_hz;
        assert!(first > 21.0 && first < 60.0, "{first}");
        assert!(last > 10_500.0 && last < 11_000.0, "{last}");
        for band in &f {
            assert!(band.iter().all(|&(_, w)| w > 0.0 && w <= 1.0));
        }
        assert!((hz_to_mel(mel_to_hz(30.0)) - 30.0).abs() < 1e-9);
    }

    #[test]
    fn frames_follow_the_length() {
        for len in [600usize, 22_050, 22_051, 50_000] {
            let mut fe = NetFrontend::new();
            let signal: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.05).sin()).collect();
            // In uneven pieces, as a decoder delivers them.
            for piece in signal.chunks(777) {
                fe.push(piece);
            }
            let mel = fe.finish();
            assert_eq!(mel.frames, 1 + len / HOP);
            assert_eq!(mel.data.len(), mel.frames * BANDS);
            let mut whole = NetFrontend::new();
            whole.push(&signal);
            assert_eq!(whole.finish().data, mel.data);
        }
    }
}
