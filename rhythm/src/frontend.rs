//! From mono samples to a log-compressed mel spectrogram, streaming.
//!
//! Frames are centred: frame `t` is the Hann-windowed block centred on
//! sample `t · hop` (the signal is padded with half a frame of silence at
//! both ends), so frame `t` describes time `t / fps`. Magnitudes are scaled
//! so that a full-scale sine reads 1, summed through triangular filters
//! spaced evenly on the mel scale, and compressed with `ln(1 + 1000·x)`,
//! which keeps silence at 0.

use std::f64::consts::PI;

use crate::fft::RealFft;

/// The gain of the compression: a magnitude `x` is `ln(1 + GAIN·x)`.
///
/// It is computed in two roundings, which differ by an ulp at times; each
/// stays the one its stage was tuned with:
/// - the front end's, `log_mel`: `(1 + GAIN·x).ln()`;
/// - the fast split's into parts ([`crate::parts`]), [`compress`]:
///   `(GAIN·x).ln_1p()`, undone by [`expand`].
pub(crate) const GAIN: f32 = 1000.0;

/// A magnitude as the front end's log-mel value.
fn log_mel(x: f32) -> f32 {
    (1.0 + GAIN * x).ln()
}

/// A magnitude as a log-mel value in the split's rounding (see [`GAIN`]).
pub(crate) fn compress(x: f32) -> f32 {
    (GAIN * x).ln_1p()
}

/// The magnitude a log-mel value stands for ([`compress`] undone).
pub(crate) fn expand(v: f32) -> f32 {
    v.exp_m1() / GAIN
}

/// How a [`Frontend`] frames and filters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontendConfig {
    /// The input's sample rate.
    pub sample_rate: u32,
    /// Frame (and FFT) length in samples, a power of two.
    pub frame: usize,
    /// Samples between frame centres.
    pub hop: usize,
    /// Mel bands.
    pub bands: usize,
    /// The lowest band's lower edge, in hertz.
    pub low_hz: f64,
    /// The highest band's upper edge, in hertz.
    pub high_hz: f64,
}

impl FrontendConfig {
    /// The front end of the spectral-flux methods: 1,024-sample frames
    /// every 220 samples at 22.05 kHz (about 100 frames per second), 128
    /// bands from 30 Hz to 10 kHz.
    pub const FLUX: FrontendConfig = FrontendConfig {
        sample_rate: crate::SAMPLE_RATE,
        frame: 1024,
        hop: 220,
        bands: 128,
        low_hz: 30.0,
        high_hz: 10_000.0,
    };

    /// Frames per second.
    pub fn fps(&self) -> f64 {
        f64::from(self.sample_rate) / self.hop as f64
    }
}

/// A mel spectrogram: `frames × bands` values, frame by frame.
#[derive(Clone, Debug, Default)]
pub struct Mel {
    /// Frames per second.
    pub fps: f64,
    /// Values per frame.
    pub bands: usize,
    /// Each band's centre frequency, in hertz.
    pub centers: Vec<f64>,
    /// The values, frame after frame.
    pub data: Vec<f32>,
}

impl Mel {
    /// The number of frames.
    pub fn frames(&self) -> usize {
        self.data.len().checked_div(self.bands).unwrap_or(0)
    }

    /// Frame `t`'s values.
    pub fn frame(&self, t: usize) -> &[f32] {
        &self.data[t * self.bands..(t + 1) * self.bands]
    }

    /// The bands whose centres lie in `low_hz..high_hz`.
    pub fn bands_between(&self, low_hz: f64, high_hz: f64) -> std::ops::Range<usize> {
        let start = self.centers.iter().take_while(|&&c| c < low_hz).count();
        let end = self.centers.iter().take_while(|&&c| c < high_hz).count();
        start..end.max(start)
    }
}

/// One triangular filter: its first bin and its weights from there.
struct Filter {
    first: usize,
    weights: Vec<f32>,
}

/// Turns samples into a [`Mel`] as they arrive.
pub struct Frontend {
    config: FrontendConfig,
    window: Vec<f32>,
    filters: Vec<Filter>,
    centers: Vec<f64>,
    fft: RealFft,
    /// Samples not yet framed, from `start` on.
    pending: Vec<f32>,
    start: usize,
    block: Vec<f32>,
    magnitudes: Vec<f32>,
    data: Vec<f32>,
}

fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}

impl Frontend {
    /// A front end with `config`.
    pub fn new(config: FrontendConfig) -> Frontend {
        let n = config.frame;
        let window: Vec<f32> = (0..n)
            .map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos()) as f32)
            .collect();
        let gain = 2.0 / window.iter().sum::<f32>();
        let window = window.iter().map(|w| w * gain).collect();
        let (filters, centers) = filterbank(&config);
        Frontend {
            config,
            window,
            filters,
            centers,
            fft: RealFft::new(n),
            pending: vec![0.0; n / 2],
            start: 0,
            block: vec![0.0; n],
            magnitudes: vec![0.0; n / 2 + 1],
            data: Vec::new(),
        }
    }

    /// Takes in more samples.
    pub fn push(&mut self, samples: &[f32]) {
        self.pending.extend_from_slice(samples);
        self.drain();
    }

    /// Frames the rest (padded with half a frame of silence) and returns
    /// the spectrogram.
    pub fn finish(mut self) -> Mel {
        let pad = vec![0.0; self.config.frame / 2];
        self.push(&pad);
        Mel {
            fps: self.config.fps(),
            bands: self.config.bands,
            centers: self.centers,
            data: self.data,
        }
    }

    fn drain(&mut self) {
        let (n, hop) = (self.config.frame, self.config.hop);
        while self.pending.len() - self.start >= n {
            let source = &self.pending[self.start..self.start + n];
            for ((b, &s), &w) in self.block.iter_mut().zip(source).zip(&self.window) {
                *b = s * w;
            }
            self.fft.magnitudes(&self.block, &mut self.magnitudes);
            for filter in &self.filters {
                let value: f32 = filter
                    .weights
                    .iter()
                    .zip(&self.magnitudes[filter.first..])
                    .map(|(w, m)| w * m)
                    .sum();
                self.data.push(log_mel(value));
            }
            self.start += hop;
        }
        if self.start >= 1 << 16 {
            self.pending.drain(..self.start);
            self.start = 0;
        }
    }
}

/// Triangular filters with peaks evenly spaced in mel between the edges.
/// A filter narrower than the bin spacing takes the bin nearest its
/// centre, so no band is empty.
fn filterbank(config: &FrontendConfig) -> (Vec<Filter>, Vec<f64>) {
    let bins = config.frame / 2 + 1;
    let bin_hz = f64::from(config.sample_rate) / config.frame as f64;
    let (low, high) = (hz_to_mel(config.low_hz), hz_to_mel(config.high_hz));
    let points: Vec<f64> = (0..config.bands + 2)
        .map(|i| mel_to_hz(low + (high - low) * i as f64 / (config.bands + 1) as f64))
        .collect();
    let mut filters = Vec::with_capacity(config.bands);
    let mut centers = Vec::with_capacity(config.bands);
    for b in 0..config.bands {
        let (lo, mid, hi) = (points[b], points[b + 1], points[b + 2]);
        let weights: Vec<(usize, f32)> = (0..bins)
            .filter_map(|k| {
                let f = k as f64 * bin_hz;
                let w = if f > lo && f <= mid {
                    (f - lo) / (mid - lo)
                } else if f > mid && f < hi {
                    (hi - f) / (hi - mid)
                } else {
                    0.0
                };
                (w > 0.0).then_some((k, w as f32))
            })
            .collect();
        let filter = match (weights.first(), weights.last()) {
            (Some(&(first, _)), Some(&(last, _))) => {
                let mut dense = vec![0.0; last - first + 1];
                for (k, w) in weights {
                    dense[k - first] = w;
                }
                Filter {
                    first,
                    weights: dense,
                }
            }
            _ => Filter {
                first: ((mid / bin_hz).round() as usize).min(bins - 1),
                weights: vec![1.0],
            },
        };
        filters.push(filter);
        centers.push(mid);
    }
    (filters, centers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sine_lights_the_band_at_its_frequency_and_frames_are_centred() {
        let config = FrontendConfig::FLUX;
        let sr = f64::from(config.sample_rate);
        // One second of 1 kHz at half scale, after one second of silence.
        let mut samples = vec![0.0f32; config.sample_rate as usize];
        samples.extend(
            (0..config.sample_rate as usize)
                .map(|i| (0.5 * (2.0 * PI * 1000.0 * i as f64 / sr).sin()) as f32),
        );
        let mut frontend = Frontend::new(config);
        // In uneven pieces, as a decoder would deliver them.
        for chunk in samples.chunks(777) {
            frontend.push(chunk);
        }
        let mel = frontend.finish();
        let expected = (samples.len() as f64 / config.hop as f64).floor() as usize + 1;
        assert!(
            mel.frames().abs_diff(expected) <= 1,
            "{} frames",
            mel.frames()
        );
        let t = (1.5 * mel.fps) as usize;
        let frame = mel.frame(t);
        let loudest = (0..mel.bands)
            .max_by(|&a, &b| frame[a].total_cmp(&frame[b]))
            .unwrap();
        let center = mel.centers[loudest];
        assert!(
            (center - 1000.0).abs() < 60.0,
            "loudest band at {center} Hz"
        );
        // Silence reads 0; the sine starts at 1 s, within one frame.
        assert!(mel
            .frame((0.5 * mel.fps) as usize)
            .iter()
            .all(|&v| v == 0.0));
        let first = (0..mel.frames())
            .find(|&t| mel.frame(t)[loudest] > 1.0)
            .unwrap();
        let start = first as f64 / mel.fps;
        assert!((start - 1.0).abs() < 0.03, "starts at {start} s");
    }

    #[test]
    fn every_band_has_weight() {
        let (filters, centers) = filterbank(&FrontendConfig::FLUX);
        assert_eq!(filters.len(), 128);
        assert!(filters.iter().all(|f| f.weights.iter().any(|&w| w > 0.0)));
        assert!(centers.windows(2).all(|c| c[0] < c[1]));
        assert!(centers[0] > 30.0 && centers[127] < 10_000.0);
    }
}
