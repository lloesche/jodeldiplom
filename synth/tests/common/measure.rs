//! Measures of a sound: its peak, RMS and largest step, its pitch (by
//! autocorrelation), its brightness (the spectral centroid, by a plain DFT),
//! and the DFT's own: a partial's tuning and ring time, where a spectral
//! envelope or a noise's spectrum peaks.

#![allow(dead_code)]

/// The largest magnitude.
pub fn peak(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |m, v| m.max(v.abs()))
}

/// The root mean square.
pub fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / x.len() as f64).sqrt() as f32
}

/// `x` in dB (full scale 1).
pub fn db(x: f32) -> f32 {
    20.0 * x.max(1e-10).log10()
}

/// The pitch of `x` at `rate`, Hz, from 50 Hz to 2 kHz: the first
/// autocorrelation peak within 10 % of the best, refined by a parabola.
pub fn pitch(x: &[f32], rate: f32) -> f32 {
    let lo = (rate / 2_000.0) as usize;
    let hi = (rate / 50.0) as usize;
    let n = x.len() - hi - 1;
    let corr = |lag: usize| -> f64 {
        let (mut s, mut a, mut b) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..n {
            let (u, v) = (f64::from(x[i]), f64::from(x[i + lag]));
            s += u * v;
            a += u * u;
            b += v * v;
        }
        s / (a * b).sqrt().max(1e-20)
    };
    let c: Vec<f64> = (lo - 1..=hi + 1).map(corr).collect();
    let best = c.iter().cloned().fold(f64::MIN, f64::max);
    let mut k = 1;
    while k + 1 < c.len() {
        if c[k] >= 0.9 * best && c[k] >= c[k - 1] && c[k] >= c[k + 1] {
            break;
        }
        k += 1;
    }
    let (a, b, g) = (c[k - 1], c[k], c[k + 1]);
    let shift = 0.5 * (a - g) / (a - 2.0 * b + g);
    rate / ((lo - 1 + k) as f64 + shift) as f32
}

/// The cents from `want` to `got`.
pub fn cents(got: f32, want: f32) -> f32 {
    1_200.0 * (got / want).log2()
}

/// The spectral centroid of `x` at `rate`, Hz: a Hann-windowed DFT of its
/// middle 1024 samples.
pub fn centroid(x: &[f32], rate: f32) -> f32 {
    const N: usize = 1_024;
    let from = x.len().saturating_sub(N) / 2;
    let frame = &x[from..(from + N).min(x.len())];
    let n = frame.len();
    let (mut num, mut den) = (0.0f64, 0.0f64);
    let cos: Vec<f64> = (0..n)
        .map(|i| (std::f64::consts::TAU * i as f64 / n as f64).cos())
        .collect();
    let sin: Vec<f64> = (0..n)
        .map(|i| (std::f64::consts::TAU * i as f64 / n as f64).sin())
        .collect();
    let w: Vec<f64> = (0..n)
        .map(|i| f64::from(frame[i]) * (0.5 - 0.5 * cos[i]))
        .collect();
    for k in 1..n / 2 {
        let (mut re, mut im) = (0.0, 0.0);
        for (i, v) in w.iter().enumerate() {
            let j = (k * i) % n;
            re += v * cos[j];
            im -= v * sin[j];
        }
        let mag = (re * re + im * im).sqrt();
        num += mag * k as f64;
        den += mag;
    }
    (num / den.max(1e-20) * f64::from(rate) / n as f64) as f32
}

/// The centroid every `step` seconds, of frames of 1024 samples.
pub fn centroids(x: &[f32], rate: f32, step: f32) -> Vec<f32> {
    let hop = (step * rate) as usize;
    let mut out = Vec::new();
    let mut at = 0;
    while at + 1_024 <= x.len() {
        let frame = &x[at..at + 1_024];
        out.push(if rms(frame) > 1e-4 {
            centroid(frame, rate)
        } else {
            0.0
        });
        at += hop;
    }
    out
}

/// Finite, audible and below `ceiling`; returns the peak.
pub fn sane(x: &[f32], ceiling: f32, what: &str) -> f32 {
    assert!(x.iter().all(|s| s.is_finite()), "{what}: not finite");
    let p = peak(x);
    assert!(p > 1e-3, "{what}: silent (peak {p})");
    assert!(p < ceiling, "{what}: peak {p}");
    p
}

/// The largest step between two samples.
pub fn biggest_jump(x: &[f32]) -> f32 {
    x.windows(2).fold(0.0f32, |m, w| m.max((w[1] - w[0]).abs()))
}

/// The Hann-windowed DFT magnitude of `x` at `f` Hz, at `rate`.
pub fn magnitude(x: &[f32], f: f64, rate: f32) -> f64 {
    let w = std::f64::consts::TAU * f / f64::from(rate);
    let n = x.len() as f64;
    let (mut re, mut im) = (0.0, 0.0);
    for (i, s) in x.iter().enumerate() {
        let window = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n).cos();
        let (sin, cos) = (w * i as f64).sin_cos();
        re += f64::from(*s) * window * cos;
        im += f64::from(*s) * window * sin;
    }
    (re * re + im * im).sqrt()
}

/// How many cents the spectral peak within 50 cents of `expected` Hz is
/// off it.
pub fn cents_off(x: &[f32], expected: f32, rate: f32) -> f64 {
    let at = |c: f64| magnitude(x, f64::from(expected) * (c / 1200.0).exp2(), rate).ln();
    let grid: Vec<f64> = (-50..=50).map(|c| at(f64::from(c))).collect();
    let i = (1..grid.len() - 1)
        .max_by(|&a, &b| grid[a].total_cmp(&grid[b]))
        .unwrap();
    let (a, b, c) = (grid[i - 1], grid[i], grid[i + 1]);
    i as f64 - 50.0 + 0.5 * (a - c) / (a - 2.0 * b + c)
}

/// The ring time to -60 dB of the partial at `f` Hz: its magnitude in a
/// 0.1 s window from 0.1 s against one `span` seconds later.
pub fn ring_time(x: &[f32], f: f32, span: f32, rate: f32) -> f32 {
    let window = |at: f32| {
        let start = (at * rate) as usize;
        magnitude(&x[start..start + (0.1 * rate) as usize], f64::from(f), rate)
    };
    let db = 20.0 * (window(0.1) / window(0.1 + span)).log10();
    60.0 * span / db as f32
}

/// The frequency of the strongest harmonic of `f0` within `lo .. hi` Hz,
/// refined by a parabola through it and its neighbours' log magnitudes:
/// where the spectral envelope peaks.
pub fn envelope_peak(x: &[f32], f0: f32, lo: f32, hi: f32, rate: f32) -> f32 {
    // A drift moves the harmonics a little: take each one's best nearby.
    let harmonic = |k: usize| {
        (-2..=2)
            .map(|d| magnitude(x, f64::from(k as f32 * f0 * (1.0 + 0.002 * d as f32)), rate))
            .fold(0.0, f64::max)
            .ln()
    };
    let ks = (lo / f0).ceil() as usize..=(hi / f0).floor() as usize;
    let k = ks
        .max_by(|&a, &b| harmonic(a).total_cmp(&harmonic(b)))
        .unwrap();
    let (a, b, c) = (harmonic(k - 1), harmonic(k), harmonic(k + 1));
    assert!(b >= a && b >= c, "no peak in {lo} .. {hi} Hz");
    (k as f64 + 0.5 * (a - c) / (a - 2.0 * b + c)) as f32 * f0
}

/// Where the averaged spectrum (Hann windows of 1024 samples, every 512)
/// peaks within `lo .. hi` Hz, refined by a parabola through the log power
/// of the peak bin and its neighbours.
pub fn noise_peak(x: &[f32], lo: f32, hi: f32, rate: f32) -> f32 {
    const N: usize = 1024;
    let bin = rate / N as f32;
    let power = |k: usize| {
        let segments: Vec<&[f32]> = x.windows(N).step_by(N / 2).collect();
        let sum: f64 = segments
            .iter()
            .map(|s| magnitude(s, f64::from(k as f32 * bin), rate).powi(2))
            .sum();
        (sum / segments.len() as f64).ln()
    };
    let ks = (lo / bin).ceil() as usize..=(hi / bin).floor() as usize;
    let k = ks.max_by(|&a, &b| power(a).total_cmp(&power(b))).unwrap();
    let (a, b, c) = (power(k - 1), power(k), power(k + 1));
    assert!(b >= a && b >= c, "no peak in {lo} .. {hi} Hz");
    (k as f64 + 0.5 * (a - c) / (a - 2.0 * b + c)) as f32 * bin
}
