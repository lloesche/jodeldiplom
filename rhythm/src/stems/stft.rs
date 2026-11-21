//! The separator's short-time Fourier transform and its inverse.
//!
//! As the network was trained with: 2,048-sample frames every 512
//! samples, a periodic Hann window, frames centred on their time (the
//! signal mirrored by half a frame at both ends, the edge sample not
//! repeated), no normalisation. The inverse adds the windowed inverse
//! frames back together and divides by the summed squared window.

use crate::fft::{Complex, RealFft};

/// Samples per frame.
pub const N_FFT: usize = 2048;
/// Samples between frames.
pub const HOP: usize = 512;
/// Bins per frame, `0..=N_FFT / 2`.
pub const BINS: usize = N_FFT / 2 + 1;

/// Transforms with their tables and buffers.
pub struct Stft {
    window: Vec<f32>,
    fft: RealFft,
    frame: Vec<f32>,
}

impl Default for Stft {
    fn default() -> Stft {
        Stft::new()
    }
}

/// Sample `j` of `x` mirrored by half a frame at both ends.
fn mirrored(x: &[f32], j: usize) -> f32 {
    let half = N_FFT / 2;
    let len = x.len() as isize;
    let mut i = j as isize - half as isize;
    if i < 0 {
        i = -i;
    }
    if i >= len {
        i = 2 * (len - 1) - i;
    }
    x[i.clamp(0, len - 1) as usize]
}

impl Stft {
    /// The tables.
    pub fn new() -> Stft {
        Stft {
            window: (0..N_FFT)
                .map(|i| {
                    let phase = 2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64;
                    (0.5 - 0.5 * phase.cos()) as f32
                })
                .collect(),
            fft: RealFft::new(N_FFT),
            frame: vec![0.0; N_FFT],
        }
    }

    /// The frames of a signal of `len` samples.
    pub fn frames(len: usize) -> usize {
        len / HOP + 1
    }

    /// The transform of `x` (more than half a frame long): frame after
    /// frame, [`BINS`] bins each.
    pub fn forward(&mut self, x: &[f32]) -> Vec<Complex> {
        let frames = Stft::frames(x.len());
        let mut out = vec![Complex::default(); frames * BINS];
        for (t, bins) in out.chunks_mut(BINS).enumerate() {
            for (n, (f, w)) in self.frame.iter_mut().zip(&self.window).enumerate() {
                *f = mirrored(x, t * HOP + n) * w;
            }
            self.fft.spectrum(&self.frame, bins);
        }
        out
    }

    /// The signal of `len` samples whose transform is `spectrum` (frames
    /// of [`BINS`] bins).
    pub fn inverse(&mut self, spectrum: &[Complex], len: usize) -> Vec<f32> {
        let frames = spectrum.len() / BINS;
        let padded = (frames - 1) * HOP + N_FFT;
        let mut sum = vec![0.0f32; padded];
        let mut envelope = vec![0.0f32; padded];
        for (t, bins) in spectrum.chunks(BINS).enumerate() {
            self.fft.inverse(bins, &mut self.frame);
            let start = t * HOP;
            for (n, (&v, &w)) in self.frame.iter().zip(&self.window).enumerate() {
                sum[start + n] += v * w;
                envelope[start + n] += w * w;
            }
        }
        let half = N_FFT / 2;
        (0..len)
            .map(|i| {
                let j = i + half;
                match (sum.get(j), envelope.get(j)) {
                    (Some(&s), Some(&e)) if e > 1e-11 => s / e,
                    _ => 0.0,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_inverse_gives_the_signal_back() {
        let x: Vec<f32> = (0..20_000)
            .map(|i| {
                let t = i as f32 / 44_100.0;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
                    + ((i * 7919) % 97) as f32 / 970.0
            })
            .collect();
        let mut stft = Stft::new();
        let spectrum = stft.forward(&x);
        assert_eq!(spectrum.len(), Stft::frames(x.len()) * BINS);
        let back = stft.inverse(&spectrum, x.len());
        for (i, (a, b)) in x.iter().zip(&back).enumerate() {
            assert!((a - b).abs() < 1e-4, "sample {i}: {a} vs {b}");
        }
    }

    #[test]
    fn a_sine_lands_in_its_bin() {
        // Bin 100 of 2048 at 44.1 kHz.
        let freq = 100.0 * 44_100.0 / N_FFT as f32;
        let x: Vec<f32> = (0..8192)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / 44_100.0).cos())
            .collect();
        let spectrum = Stft::new().forward(&x);
        let frame = &spectrum[4 * BINS..5 * BINS];
        let peak = (0..BINS)
            .max_by(|&a, &b| frame[a].norm().total_cmp(&frame[b].norm()))
            .unwrap();
        assert_eq!(peak, 100);
        // A periodic Hann window sums to N/2, so a full-scale cosine reads N/4.
        assert!((frame[100].norm() - N_FFT as f32 / 4.0).abs() < 1.0);
    }
}
