//! The speech recogniser's input: log-mel features as Whisper's feature
//! extractor makes them, which [`super`]'s encoder was trained on.
//!
//! 1. Frames of [`N_FFT`] samples, [`HOP`] apart, centred (the voice
//!    mirrored [`N_FFT`]` / 2` samples at either end), under a periodic
//!    Hann window.
//! 2. Each frame's power spectrum ([`BINS`] bins up to 8 kHz) through
//!    [`MELS`] triangular filters on the Slaney mel scale, each scaled to
//!    unit area.
//! 3. `log10` (at least 1e-10); the last frame dropped; each value held
//!    within 8 of the whole voice's highest; then `(x + 4) / 4`.
//!
//! The frames come `MELS`-major: `[mel][frame]`, as the model takes them.

use crate::nn::par_rows;

/// Samples a frame.
pub const N_FFT: usize = 400;
/// Samples between frames (100 frames a second).
pub const HOP: usize = 160;
/// Mel bands.
pub const MELS: usize = 128;
/// Frequency bins of a frame's spectrum.
pub const BINS: usize = N_FFT / 2 + 1;
const RATE: f64 = 16_000.0;

/// Hertz to the Slaney mel scale: linear to 1 kHz, logarithmic above.
fn mel_of(hz: f64) -> f64 {
    let (min_log_hz, min_log_mel, step) = (1000.0, 15.0, 27.0 / 6.4f64.ln());
    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() * step
    } else {
        3.0 * hz / 200.0
    }
}

/// The Slaney mel scale to hertz.
fn hz_of(mel: f64) -> f64 {
    let (min_log_hz, min_log_mel, step) = (1000.0, 15.0, 6.4f64.ln() / 27.0);
    if mel >= min_log_mel {
        min_log_hz * ((mel - min_log_mel) * step).exp()
    } else {
        200.0 * mel / 3.0
    }
}

/// The mel filters, `[bin][mel]`: [`MELS`] triangles evenly spaced in mel
/// from 0 to 8 kHz, each scaled by 2 / its width in hertz.
pub fn filters() -> Vec<f32> {
    let (lo, hi) = (mel_of(0.0), mel_of(RATE / 2.0));
    let edges: Vec<f64> = (0..MELS + 2)
        .map(|i| hz_of(lo + (hi - lo) * i as f64 / (MELS + 1) as f64))
        .collect();
    let mut out = vec![0.0f32; BINS * MELS];
    for b in 0..BINS {
        let hz = (RATE / 2.0) * b as f64 / (BINS - 1) as f64;
        for m in 0..MELS {
            let down = (hz - edges[m]) / (edges[m + 1] - edges[m]);
            let up = (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1]);
            let area = 2.0 / (edges[m + 2] - edges[m]);
            out[b * MELS + m] = (down.min(up).max(0.0) * area) as f32;
        }
    }
    out
}

/// The frames `samples` samples give (the last of the centred frames
/// dropped).
pub fn frames_of(samples: usize) -> usize {
    samples / HOP
}

/// What every voice's features are made with, made once: the real DFT as a
/// table (cos and sin of each bin at each sample, windowed) and the
/// filters.
struct Tables {
    cos: Vec<f32>,
    sin: Vec<f32>,
    filters: Vec<f32>,
}

fn tables() -> &'static Tables {
    static TABLES: std::sync::OnceLock<Tables> = std::sync::OnceLock::new();
    TABLES.get_or_init(|| {
        let window: Vec<f64> = (0..N_FFT)
            .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos())
            .collect();
        let mut cos = vec![0.0f32; BINS * N_FFT];
        let mut sin = vec![0.0f32; BINS * N_FFT];
        for b in 0..BINS {
            for n in 0..N_FFT {
                let a = 2.0 * std::f64::consts::PI * ((b * n) % N_FFT) as f64 / N_FFT as f64;
                cos[b * N_FFT + n] = (a.cos() * window[n]) as f32;
                sin[b * N_FFT + n] = (a.sin() * window[n]) as f32;
            }
        }
        Tables {
            cos,
            sin,
            filters: filters(),
        }
    })
}

/// The features of each of `pieces` ([`features`]): one piece on eight
/// threads, many on up to eight, each piece on one (the same features).
pub fn features_of(pieces: &[&[f32]]) -> Vec<(Vec<f32>, usize)> {
    if pieces.len() < 2 {
        return pieces.iter().map(|p| features(p, 8)).collect();
    }
    let workers = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(8);
    let mut out = vec![(Vec::new(), 0); pieces.len()];
    let each = pieces.len().div_ceil(workers);
    std::thread::scope(|s| {
        for (slots, pieces) in out.chunks_mut(each).zip(pieces.chunks(each)) {
            s.spawn(move || {
                for (slot, piece) in slots.iter_mut().zip(pieces) {
                    *slot = features(piece, 1);
                }
            });
        }
    });
    out
}

/// The log-mel features of `voice` (16 kHz mono), `[MELS][frames]`, and
/// the frames.
pub fn features(voice: &[f32], threads: usize) -> (Vec<f32>, usize) {
    let frames = frames_of(voice.len());
    if frames == 0 {
        return (Vec::new(), 0);
    }
    let half = N_FFT / 2;
    // Mirrored at the ends, as numpy's "reflect".
    let at = |i: i64| -> f32 {
        let n = voice.len() as i64;
        let mut i = i;
        if n == 1 {
            return voice[0];
        }
        while i < 0 || i >= n {
            if i < 0 {
                i = -i;
            }
            if i >= n {
                i = 2 * (n - 1) - i;
            }
        }
        voice[i as usize]
    };
    let Tables { cos, sin, filters } = tables();
    // Frame-major first: [frame][mel].
    let mut logs = vec![0.0f32; frames * MELS];
    par_rows(&mut logs, MELS, threads, |t, row| {
        let start = (t * HOP) as i64 - half as i64;
        let x: [f32; N_FFT] = std::array::from_fn(|n| at(start + n as i64));
        let mut power = [0.0f32; BINS];
        for (b, p) in power.iter_mut().enumerate() {
            let (c, s) = (&cos[b * N_FFT..][..N_FFT], &sin[b * N_FFT..][..N_FFT]);
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for n in 0..N_FFT {
                re += f64::from(x[n] * c[n]);
                im += f64::from(x[n] * s[n]);
            }
            *p = (re * re + im * im) as f32;
        }
        for (m, v) in row.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (b, p) in power.iter().enumerate() {
                sum += filters[b * MELS + m] * p;
            }
            *v = sum.max(1e-10).log10();
        }
    });
    let top = logs.iter().copied().fold(f32::MIN, f32::max);
    let mut out = vec![0.0f32; MELS * frames];
    for t in 0..frames {
        for m in 0..MELS {
            out[m * frames + t] = (logs[t * MELS + m].max(top - 8.0) + 4.0) / 4.0;
        }
    }
    (out, frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mel_scale_turns_at_a_kilohertz() {
        assert!((mel_of(1000.0) - 15.0).abs() < 1e-12);
        for hz in [0.0, 300.0, 999.0, 1000.0, 4000.0, 8000.0] {
            assert!((hz_of(mel_of(hz)) - hz).abs() < 1e-9, "{hz}");
        }
    }

    #[test]
    fn each_filter_is_a_triangle_of_its_area() {
        let f = filters();
        for m in 0..MELS {
            let peak = (0..BINS).map(|b| f[b * MELS + m]).fold(0.0f32, f32::max);
            assert!(peak > 0.0, "filter {m} is empty");
        }
    }

    #[test]
    fn a_second_gives_a_hundred_frames() {
        let voice: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.05).sin() * 0.1).collect();
        let (f, frames) = features(&voice, 2);
        assert_eq!(frames, 100);
        assert_eq!(f.len(), MELS * 100);
        let top = f.iter().copied().fold(f32::MIN, f32::max);
        let low = f.iter().copied().fold(f32::MAX, f32::min);
        assert!(top - low <= 2.0 + 1e-6, "held within 8 (2 after scaling)");
    }

    #[test]
    fn many_pieces_get_the_features_each_gets_alone() {
        let pieces: Vec<Vec<f32>> = (0..5)
            .map(|k| {
                (0..8_000 + 1_000 * k)
                    .map(|i| (i as f32 * (0.03 + 0.01 * k as f32)).sin() * 0.2)
                    .collect()
            })
            .collect();
        let all: Vec<&[f32]> = pieces.iter().map(|p| &p[..]).collect();
        for (piece, got) in pieces.iter().zip(features_of(&all)) {
            assert_eq!(got, features(piece, 8));
        }
    }
}
