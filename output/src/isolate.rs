//! The singer isolated: what the echo canceller leaves of the band, and the
//! room's own sounds (a fire, a fan, people talking far off), taken out of
//! the microphone so that only the voice is left.
//!
//! The canceller ([`crate::aec`]) takes out the echo it can predict from
//! what played. Through a laptop's speakers much of the band is not
//! predictable that way (their distortion, a room's tail), and the room
//! adds sounds no reference knows. A small network trained on singing
//! takes those out: each 10 ms it hears the last 20 ms and gives a gain
//! per frequency (0 to 1) and how likely a voice is there.
//!
//! # The network
//!
//! At 16 kHz, a Hann window of 320 samples every 160 (161 bins). Each
//! frame's log power (`log10(|X|² + 1e-10) / 5`) is taken less the running
//! mean of the frames' mean log power, decaying over two seconds: the input
//! is relative to the recent level, so a singer close to the microphone is
//! not mistaken for the band for being loud. Then a layer to 256, two GRU
//! layers of 256, and two heads: the 161 gains (sigmoid) and the voice's
//! presence (sigmoid). The gains multiply the spectrum, which is turned
//! back into samples by overlap-add. About 0.87 M weights, kept as f16 in
//! the binary; about 0.9 M multiply-adds a frame, a few per cent of a core.
//!
//! It was trained on our own material: voices separated from songs, under
//! what the canceller left of other songs' bands played through simulated
//! laptops into simulated rooms, and recorded noises, with losing the voice
//! weighed heavier than leaving some band. On 24 songs by artists it never
//! heard, a clean voice keeps its level (median −0.1 dB); the band the
//! canceller left is taken down 23 dB, an echo never cancelled 24 dB, a
//! fire 40 dB. A voice under that band, 5 dB louder than itself, loses
//! 1.8 dB (11.5 dB in its worst 5 %); it hears only the microphone, not
//! what played.
//!
//! # Streaming
//!
//! [`Isolator`] takes 16 kHz samples in any amounts and gives the isolated
//! voice [`DELAY`] samples late (and up to a hop more waiting for a frame
//! to fill). [`Downsampler`] brings the canceller's rate to 16 kHz.

use std::sync::OnceLock;

/// The network's rate.
pub const RATE: u32 = 16_000;
/// Frequency bins of a frame.
pub const BINS: usize = N_FFT / 2 + 1;
/// How late the isolated voice comes out, in samples at [`RATE`].
pub const DELAY: usize = HOP;

const N_FFT: usize = 320;
const HOP: usize = 160;

/// The weights, as `export.py` writes them (see [`Weights::parse`]).
static MODEL: &[u8] = include_bytes!("../models/voice-v6.bin");

struct Layer {
    w_ih: Vec<f32>,
    w_hh: Vec<f32>,
    b_ih: Vec<f32>,
    b_hh: Vec<f32>,
}

struct Weights {
    hidden: usize,
    level_decay: f64,
    inp_w: Vec<f32>,
    inp_b: Vec<f32>,
    layers: Vec<Layer>,
    mask_w: Vec<f32>,
    mask_b: Vec<f32>,
    voice_w: Vec<f32>,
    voice_b: f32,
}

/// An IEEE half as `f32`.
fn half(bits: u16) -> f32 {
    let exp = u32::from((bits >> 10) & 0x1f);
    let man = u32::from(bits & 0x3ff);
    let magnitude = match exp {
        0 => man as f32 * (1.0 / 16_777_216.0),
        31 if man == 0 => f32::INFINITY,
        31 => f32::NAN,
        _ => f32::from_bits(((exp + 112) << 23) | (man << 13)),
    };
    if bits & 0x8000 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

impl Weights {
    /// The model file: `b"JDLVOICE"`, then little-endian u32 version (2),
    /// bins, hidden, layers, inputs (1: the microphone alone), f32 level
    /// tau in seconds, u32 tensors; each tensor u32 ndim, u32 dims, its
    /// values as f16. The order: the input layer's weight and bias, each
    /// GRU layer's input and hidden weights and biases, the gains' weight
    /// and bias, the presence's.
    fn parse(bytes: &[u8]) -> Weights {
        assert_eq!(&bytes[..8], b"JDLVOICE", "not a voice model");
        let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let (version, bins, hidden, layers) = (word(8), word(12), word(16), word(20));
        let (inputs, tau) = (word(24), f32::from_bits(word(28)));
        let tensors = word(32) as usize;
        assert_eq!(version, 2, "voice model version");
        assert_eq!(bins as usize, BINS, "voice model bins");
        assert_eq!(inputs, 1, "voice model inputs (the microphone alone)");
        let (hidden, layers) = (hidden as usize, layers as usize);
        assert_eq!(tensors, 2 + 4 * layers + 4, "voice model tensors");
        let mut at = 36;
        let mut tensor = |shape: &[usize]| -> Vec<f32> {
            let ndim = word(at) as usize;
            at += 4;
            let dims: Vec<usize> = (0..ndim).map(|i| word(at + 4 * i) as usize).collect();
            at += 4 * ndim;
            assert_eq!(dims, shape, "voice model tensor shape");
            let n: usize = shape.iter().product();
            let v = bytes[at..at + 2 * n]
                .chunks_exact(2)
                .map(|b| half(u16::from_le_bytes([b[0], b[1]])))
                .collect();
            at += 2 * n;
            v
        };
        let inp_w = tensor(&[hidden, BINS]);
        let inp_b = tensor(&[hidden]);
        let layers = (0..layers)
            .map(|_| Layer {
                w_ih: tensor(&[3 * hidden, hidden]),
                w_hh: tensor(&[3 * hidden, hidden]),
                b_ih: tensor(&[3 * hidden]),
                b_hh: tensor(&[3 * hidden]),
            })
            .collect();
        let mask_w = tensor(&[BINS, hidden]);
        let mask_b = tensor(&[BINS]);
        let voice_w = tensor(&[1, hidden]);
        let voice_b = tensor(&[1])[0];
        assert_eq!(at, bytes.len(), "voice model length");
        Weights {
            hidden,
            level_decay: (-(HOP as f64) / f64::from(RATE) / f64::from(tau)).exp(),
            inp_w,
            inp_b,
            layers,
            mask_w,
            mask_b,
            voice_w,
            voice_b,
        }
    }

    fn get() -> &'static Weights {
        static WEIGHTS: OnceLock<Weights> = OnceLock::new();
        WEIGHTS.get_or_init(|| Weights::parse(MODEL))
    }
}

/// The analysis window and the transform's cosines and sines, bin by bin.
struct Tables {
    window: [f32; N_FFT],
    /// `window[n]² + window[n + HOP]²`: the overlap-add's sum of squares.
    envelope: [f32; HOP],
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Tables {
    fn get() -> &'static Tables {
        static TABLES: OnceLock<Tables> = OnceLock::new();
        TABLES.get_or_init(|| {
            let mut window = [0.0f32; N_FFT];
            for (n, w) in window.iter_mut().enumerate() {
                // Periodic Hann.
                *w = (0.5 - 0.5 * (std::f64::consts::TAU * n as f64 / N_FFT as f64).cos()) as f32;
            }
            let mut envelope = [0.0f32; HOP];
            for (n, e) in envelope.iter_mut().enumerate() {
                *e = window[n] * window[n] + window[n + HOP] * window[n + HOP];
            }
            let mut cos = vec![0.0f32; BINS * N_FFT];
            let mut sin = vec![0.0f32; BINS * N_FFT];
            for k in 0..BINS {
                for n in 0..N_FFT {
                    let a = std::f64::consts::TAU * ((k * n) % N_FFT) as f64 / N_FFT as f64;
                    cos[k * N_FFT + n] = a.cos() as f32;
                    sin[k * N_FFT + n] = a.sin() as f32;
                }
            }
            Tables {
                window,
                envelope,
                cos,
                sin,
            }
        })
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = ca
        .remainder()
        .iter()
        .zip(cb.remainder())
        .map(|(x, y)| x * y)
        .sum();
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// `out = w · x + b`, `w` row-major with `x.len()` columns.
fn affine(w: &[f32], b: &[f32], x: &[f32], out: &mut [f32]) {
    for ((o, row), bias) in out.iter_mut().zip(w.chunks_exact(x.len())).zip(b) {
        *o = dot(row, x) + bias;
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The isolating network, streaming at 16 kHz.
pub struct Isolator {
    weights: &'static Weights,
    tables: &'static Tables,
    /// The last `N_FFT` samples, the newest last.
    frame: [f32; N_FFT],
    /// Samples of the next hop so far.
    filled: usize,
    /// The running mean's sums (see the module's docs).
    level_sum: f64,
    level_weight: f64,
    /// Each GRU layer's state.
    state: Vec<Vec<f32>>,
    /// The overlap-add's sums of the frames so far.
    overlap: [f32; N_FFT],
    gains: [f32; BINS],
    presence: f32,
    // Scratch.
    re: [f32; BINS],
    im: [f32; BINS],
    feature: [f32; BINS],
    hidden: Vec<f32>,
    gi: Vec<f32>,
    gh: Vec<f32>,
}

impl Default for Isolator {
    fn default() -> Isolator {
        Isolator::new()
    }
}

impl Isolator {
    /// An isolator with the built-in network, its state empty.
    pub fn new() -> Isolator {
        let weights = Weights::get();
        let h = weights.hidden;
        let mut isolator = Isolator {
            weights,
            tables: Tables::get(),
            frame: [0.0; N_FFT],
            filled: 0,
            level_sum: 0.0,
            level_weight: 0.0,
            state: vec![vec![0.0; h]; weights.layers.len()],
            overlap: [0.0; N_FFT],
            gains: [0.0; BINS],
            presence: 0.0,
            re: [0.0; BINS],
            im: [0.0; BINS],
            feature: [0.0; BINS],
            hidden: vec![0.0; h],
            gi: vec![0.0; 3 * h],
            gh: vec![0.0; 3 * h],
        };
        isolator.reset();
        isolator
    }

    /// Forgets everything heard: a new stream starts.
    pub fn reset(&mut self) {
        // The first frame is centred on the first sample: half a window of
        // silence before it.
        self.frame = [0.0; N_FFT];
        self.filled = N_FFT - HOP;
        self.level_sum = 0.0;
        self.level_weight = 0.0;
        for s in &mut self.state {
            s.fill(0.0);
        }
        self.overlap = [0.0; N_FFT];
        self.gains = [0.0; BINS];
        self.presence = 0.0;
    }

    /// The last frame's gains, bin by bin (0 to 1).
    pub fn gains(&self) -> &[f32; BINS] {
        &self.gains
    }

    /// How likely the last frame holds a voice (0 to 1).
    pub fn presence(&self) -> f32 {
        self.presence
    }

    /// Takes `input` (16 kHz) and appends the isolated voice to `out`,
    /// [`DELAY`] samples late: the first [`DELAY`] samples out are the
    /// silence before the stream.
    pub fn process(&mut self, mut input: &[f32], out: &mut Vec<f32>) {
        while !input.is_empty() {
            let want = N_FFT - self.filled;
            let n = want.min(input.len());
            self.frame[self.filled..self.filled + n].copy_from_slice(&input[..n]);
            self.filled += n;
            input = &input[n..];
            if self.filled == N_FFT {
                self.step(out);
                self.frame.copy_within(HOP.., 0);
                self.filled = N_FFT - HOP;
            }
        }
    }

    /// One frame: the spectrum, the network, the gains applied, a hop out.
    fn step(&mut self, out: &mut Vec<f32>) {
        let t = self.tables;
        let w = self.weights;
        let mut windowed = [0.0f32; N_FFT];
        for ((o, x), win) in windowed.iter_mut().zip(&self.frame).zip(&t.window) {
            *o = x * win;
        }
        let mut level = 0.0f32;
        for k in 0..BINS {
            let row = k * N_FFT..(k + 1) * N_FFT;
            self.re[k] = dot(&windowed, &t.cos[row.clone()]);
            self.im[k] = -dot(&windowed, &t.sin[row]);
            let f = (self.re[k] * self.re[k] + self.im[k] * self.im[k] + 1e-10).log10() / 5.0;
            self.feature[k] = f;
            level += f;
        }
        self.level_sum = w.level_decay * self.level_sum + f64::from(level / BINS as f32);
        self.level_weight = w.level_decay * self.level_weight + 1.0;
        let mean = (self.level_sum / self.level_weight) as f32;
        for f in &mut self.feature {
            *f -= mean;
        }
        affine(&w.inp_w, &w.inp_b, &self.feature, &mut self.hidden);
        for v in &mut self.hidden {
            *v = v.max(0.0);
        }
        let h = w.hidden;
        for (layer, state) in w.layers.iter().zip(&mut self.state) {
            affine(&layer.w_ih, &layer.b_ih, &self.hidden, &mut self.gi);
            affine(&layer.w_hh, &layer.b_hh, state, &mut self.gh);
            for i in 0..h {
                let r = sigmoid(self.gi[i] + self.gh[i]);
                let z = sigmoid(self.gi[h + i] + self.gh[h + i]);
                let n = (self.gi[2 * h + i] + r * self.gh[2 * h + i]).tanh();
                state[i] = (1.0 - z) * n + z * state[i];
            }
            self.hidden.copy_from_slice(state);
        }
        affine(&w.mask_w, &w.mask_b, &self.hidden, &mut self.gains);
        for g in &mut self.gains {
            *g = sigmoid(*g);
        }
        self.presence = sigmoid(dot(&w.voice_w, &self.hidden) + w.voice_b);
        // Back to samples: the inverse of a one-sided spectrum (DC and the
        // top bin once, the others twice), windowed again and overlapped.
        let mut time = [0.0f32; N_FFT];
        for k in 0..BINS {
            let scale = if k == 0 || k == BINS - 1 { 1.0 } else { 2.0 } / N_FFT as f32;
            let (a, b) = (
                self.re[k] * self.gains[k] * scale,
                self.im[k] * self.gains[k] * scale,
            );
            let row = k * N_FFT..(k + 1) * N_FFT;
            for ((y, c), s) in time.iter_mut().zip(&t.cos[row.clone()]).zip(&t.sin[row]) {
                *y += a * c - b * s;
            }
        }
        for ((o, y), win) in self.overlap.iter_mut().zip(&time).zip(&t.window) {
            *o += y * win;
        }
        out.extend(
            self.overlap[..HOP]
                .iter()
                .zip(&t.envelope)
                .map(|(o, e)| o / e),
        );
        self.overlap.copy_within(HOP.., 0);
        self.overlap[N_FFT - HOP..].fill(0.0);
    }
}

/// Brings a rate down to [`RATE`]: a windowed sinc (Kaiser, β 5) cut off
/// at 8 kHz, 10 output samples to either side, the filter the network's
/// training material was resampled with (scipy's `resample_poly`). A
/// steeper one, cut off lower, changed what the network left in the
/// band-only stretches of real takes by up to 20 dB. It delays by those
/// 10 samples' span (0.6 ms).
pub struct Downsampler {
    /// Input samples per output sample.
    step: f64,
    /// The kernel's half width, in input samples.
    width: f64,
    /// The kernel from 0 to `width` in steps of `1 / PHASES` input samples.
    kernel: Vec<f32>,
    /// Input kept, from where the next output sample's kernel starts.
    kept: Vec<f32>,
    /// The next output sample's time, in input samples from `kept[0]`.
    next: f64,
}

const PHASES: f64 = 256.0;
const SIDE: f64 = 10.0;
const CUTOFF: f64 = 8_000.0;
const BETA: f64 = 5.0;

fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term) = (1.0, 1.0);
    for k in 1..50 {
        term *= (x / (2.0 * k as f64)).powi(2);
        sum += term;
        if term < 1e-12 * sum {
            break;
        }
    }
    sum
}

impl Downsampler {
    /// From `rate` (at least [`RATE`]).
    pub fn new(rate: u32) -> Downsampler {
        let rate = f64::from(rate.max(RATE));
        let step = rate / f64::from(RATE);
        let width = SIDE * step;
        let fc = 2.0 * CUTOFF / rate;
        let beta = BETA;
        let n = (width * PHASES).ceil() as usize + 2;
        let kernel = (0..n)
            .map(|i| {
                let t = i as f64 / PHASES;
                if t >= width {
                    return 0.0;
                }
                let x = std::f64::consts::PI * fc * t;
                let sinc = if x == 0.0 { 1.0 } else { x.sin() / x };
                let r = t / width;
                let window = bessel_i0(beta * (1.0 - r * r).sqrt()) / bessel_i0(beta);
                fc * sinc * window
            })
            .collect::<Vec<f64>>();
        // Unit gain at DC, as `firwin` scales it: the kernel's integral.
        let area = 2.0 * kernel.iter().sum::<f64>() / PHASES - kernel[0] / PHASES;
        let kernel = kernel.iter().map(|k| (k / area) as f32).collect();
        let pad = width.ceil() as usize + 1;
        Downsampler {
            step,
            width,
            kernel,
            kept: vec![0.0; pad],
            next: pad as f64,
        }
    }

    /// Takes `input` and appends what it makes at [`RATE`] to `out`.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        self.kept.extend_from_slice(input);
        let last = self.kept.len() as f64 - 1.0;
        while self.next + self.width <= last {
            let from = (self.next - self.width).ceil().max(0.0) as usize;
            let to = (self.next + self.width).floor() as usize;
            let mut acc = 0.0f32;
            for (i, &x) in self.kept[from..=to].iter().enumerate() {
                let t = ((from + i) as f64 - self.next).abs() * PHASES;
                let k = t as usize;
                let frac = (t - k as f64) as f32;
                let h = self.kernel[k] + (self.kernel[k + 1] - self.kernel[k]) * frac;
                acc += x * h;
            }
            out.push(acc);
            self.next += self.step;
        }
        // Drop what no later sample needs.
        let keep_from = (self.next - self.width).floor().max(0.0) as usize;
        if keep_from > 4096 {
            self.kept.drain(..keep_from);
            self.next -= keep_from as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The signal `export.py` made its reference from.
    fn test_signal() -> Vec<f32> {
        let mut state: u32 = 0x1234_5678;
        (0..64_000)
            .map(|i| {
                if i < 400 {
                    return 0.0;
                }
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let noise = f64::from(state >> 8) / 16_777_216.0 - 0.5;
                let t = f64::from(i) / 16_000.0;
                let a = if i < 32_000 { 0.1 } else { 0.4 };
                let tau = std::f64::consts::TAU;
                let tone = (tau * 220.0 * t).sin()
                    + 0.5 * (tau * 660.0 * t + 3.0 * (tau * 5.0 * t).sin()).sin();
                (a * tone + 0.02 * noise) as f32
            })
            .collect()
    }

    fn reference() -> Vec<f32> {
        include_bytes!("../tests/data/isolate-reference.bin")
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn halves_convert() {
        assert_eq!(half(0x3c00), 1.0);
        assert_eq!(half(0xc000), -2.0);
        assert_eq!(half(0x7bff), 65504.0);
        assert_eq!(half(0x0001), 2f32.powi(-24));
        assert_eq!(half(0x0000), 0.0);
        assert!(half(0x7c00).is_infinite() && half(0x7e00).is_nan());
    }

    #[test]
    fn matches_the_trained_network() {
        let x = test_signal();
        let r = reference();
        let (presence, rest) = r.split_at(400);
        let (gains, rest) = rest.split_at(25 * BINS);
        let (rms, rest) = rest.split_at(400);
        let (at_1s, at_3s) = rest.split_at(1600);
        // Frame by frame: each hop of input completes a frame.
        let mut iso = Isolator::new();
        let mut out = Vec::new();
        let (mut worst_p, mut worst_g) = (0.0f32, 0.0f32);
        for (t, hop) in x.chunks_exact(HOP).enumerate() {
            iso.process(hop, &mut out);
            worst_p = worst_p.max((iso.presence() - presence[t]).abs());
            if t % 16 == 0 {
                let want = &gains[t / 16 * BINS..(t / 16 + 1) * BINS];
                for (g, w) in iso.gains().iter().zip(want) {
                    worst_g = worst_g.max((g - w).abs());
                }
            }
        }
        // Measured 3e-5 and 1e-5 (f32 sums in another order).
        assert!(worst_p < 1e-4, "presence off by {worst_p}");
        assert!(worst_g < 1e-4, "gains off by {worst_g}");
        // The samples, fed in pieces that are not hops.
        let mut iso = Isolator::new();
        let mut pieces = Vec::new();
        for piece in x.chunks(128) {
            iso.process(piece, &mut pieces);
        }
        assert_eq!(pieces, out, "the pieces fed change nothing");
        let y = &out[DELAY..];
        let mut worst_rms = 0.0f64;
        for (b, want) in rms.iter().enumerate().take(398) {
            let block = &y[b * HOP..(b + 1) * HOP];
            let got =
                (block.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / HOP as f64).sqrt();
            worst_rms = worst_rms.max((got - f64::from(*want)).abs() / (f64::from(*want) + 1e-4));
        }
        assert!(
            worst_rms < 1e-3,
            "block rms off by {:.4} (relative)",
            worst_rms
        );
        let peak = at_3s.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        for (got, want) in y[16_000..17_600]
            .iter()
            .zip(at_1s)
            .chain(y[48_000..49_600].iter().zip(at_3s))
        {
            assert!(
                (got - want).abs() < 5e-4 * peak,
                "sample {got} against {want}"
            );
        }
    }

    #[test]
    fn a_steady_hiss_is_taken_out() {
        // A fan's hiss: white noise at -30 dBFS for 6 s.
        let mut state: u32 = 0x9e37_79b9;
        let x: Vec<f32> = (0..6 * 16_000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (f64::from(state >> 8) / 16_777_216.0 - 0.5) as f32 * 0.11
            })
            .collect();
        let mut iso = Isolator::new();
        let mut y = Vec::new();
        iso.process(&x, &mut y);
        let rms = |s: &[f32]| (s.iter().map(|v| v * v).sum::<f32>() / s.len() as f32).sqrt();
        let late = &y[2 * 16_000 + DELAY..6 * 16_000];
        let cut = 20.0 * (rms(late) / rms(&x[2 * 16_000..])).log10();
        // Measured: 65 dB taken out, presence 0.003.
        assert!(cut < -20.0, "the hiss kept at {cut:.1} dB");
        assert!(iso.presence() < 0.5, "a hiss called a voice");
    }

    #[test]
    fn downsampling_keeps_the_voice_band_and_folds_nothing_back() {
        for rate in [44_100u32, 48_000] {
            let tone = |f: f64| -> f32 {
                let x: Vec<f32> = (0..rate as usize)
                    .map(|i| (std::f64::consts::TAU * f * i as f64 / f64::from(rate)).sin() as f32)
                    .collect();
                let mut d = Downsampler::new(rate);
                let mut y = Vec::new();
                for piece in x.chunks(128) {
                    d.process(piece, &mut y);
                }
                assert!(
                    (y.len() as i64 - 16_000).unsigned_abs() < 40,
                    "{} samples",
                    y.len()
                );
                let y = &y[1000..15_000];
                (y.iter().map(|v| v * v).sum::<f32>() / y.len() as f32 * 2.0).sqrt()
            };
            for f in [200.0, 1_000.0, 3_000.0, 6_000.0] {
                let g = tone(f);
                assert!((g - 1.0).abs() < 0.01, "{rate} Hz: {f} Hz kept at {g}");
            }
            // 12 and 20 kHz would fold onto 4 kHz and under.
            for f in [12_000.0, 20_000.0] {
                let g = tone(f);
                assert!(g < 5e-3, "{rate} Hz: {f} Hz folds back at {g}");
            }
        }
    }
}
