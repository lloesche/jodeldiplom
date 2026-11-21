//! The pitch of a voice, frame by frame: our own inference of FCPE (Fast
//! Context-based Pitch Estimation, CNChTu; code and weights under the MIT
//! licence, see `models/LICENSE-fcpe`), its bundled convolution-only
//! model, the weights built in at about 6 bits a value (Q5_K fitted by
//! GPTQ, see [`crate::net::q4k`]; `examples/pitch_pack.rs` made them).
//!
//! The voice, at 16 kHz mono, becomes a log-mel spectrogram as the
//! authors' extractor makes it: frames of 1024 samples (periodic Hann
//! window) every 160 (100 a second), the signal mirrored by 432 samples at
//! each end and the frames not centred, the magnitude (with 1e-9 under the
//! root), 128 Slaney mel bands from 0 to 8 kHz normalised by their width,
//! the natural log of at least 1e-5; one frame more than whole hops, the
//! last repeated. The network:
//!
//! - a convolution of 3 frames to 512 channels, a group norm (4 groups,
//!   over the whole song), a leaky ReLU, another convolution of 3 frames;
//! - six times, added to what goes in: a layer norm, a pointwise
//!   convolution to 2048 channels, a GLU to 1024, a depthwise convolution
//!   of 31 frames, SiLU, a pointwise convolution back to 512;
//! - a layer norm, a linear map to 360 classes and their sigmoids.
//!
//! The classes are pitches 20 cents apart from 32.7 Hz (C1) to 1975.5 Hz
//! (B6). A frame's pitch is the mean of the nine classes around the
//! strongest, weighed by their sigmoids; where the strongest is at most
//! [`VOICED`], there is none. Only the group norm sees the whole song; the
//! rest sees 92 frames each side of a frame. So the group norm's
//! statistics come first, from the first convolution alone, and then the
//! song goes through in windows of [`WINDOW`] frames with [`HALO`] more at
//! each side: the same values as at once, in a few megabytes.

use std::f64::consts::PI;
use std::sync::OnceLock;

use crate::fft::{Complex, RealFft};
use crate::net::layers::{par_chunks, Linear};
use crate::net::weights::Weights;

/// The weights, built in (`examples/pitch_pack.rs`).
static WEIGHTS: &[u8] = include_bytes!("../models/pitch.bin");

/// The rate the network hears.
pub const RATE: u32 = 16_000;
/// Samples between frames (100 frames a second).
pub const HOP: usize = 160;
/// Frames a second.
pub const FPS: f64 = RATE as f64 / HOP as f64;
/// The spectrogram's mel bands.
pub const BANDS: usize = 128;
/// The pitch classes.
pub const CLASSES: usize = 360;
/// A frame whose strongest class is at most this has no pitch (the
/// authors' default).
pub const VOICED: f32 = 0.006;

const FRAME: usize = 1024;
/// Samples mirrored before the first frame (and after the last).
const PAD: usize = (FRAME - HOP) / 2;
const DIM: usize = 512;
const INNER: usize = 1024;
const KERNEL: usize = 31;
const LAYERS: usize = 6;
/// Frames worked on at a time.
const CHUNK: usize = 128;
/// Frames a window of the song gives.
pub const WINDOW: usize = 3000;
/// Frames heard beyond each side of a window: more than the network's
/// reach after the group norm (1 + 1 + 6·15).
pub const HALO: usize = 96;

fn threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

/// A voice's pitch, a value a frame ([`FPS`] a second).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pitch {
    /// The pitch in Hz, 0 where there is none.
    pub hz: Vec<f32>,
    /// The strongest class's sigmoid: how sure the network is there is a
    /// pitch, 0 to 1.
    pub confidence: Vec<f32>,
}

impl Pitch {
    /// Frame `i`'s pitch as a MIDI note number (69 is A4, 440 Hz), with
    /// its fraction; `None` where there is none.
    pub fn midi(&self, i: usize) -> Option<f32> {
        let hz = *self.hz.get(i)?;
        (hz > 0.0).then(|| 69.0 + 12.0 * (hz / 440.0).log2())
    }
}

/// Hertz to mel, on the Slaney scale.
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

/// librosa's mel filters for 16 kHz, 1024-point frames, 128 bands from 0
/// to 8 kHz, Slaney-normalised: `(bin, weight)` pairs a band.
fn bank() -> Vec<Vec<(usize, f32)>> {
    let bins = FRAME / 2 + 1;
    let top = hz_to_mel(8000.0);
    let points: Vec<f64> = (0..BANDS + 2)
        .map(|i| mel_to_hz(top * i as f64 / (BANDS + 1) as f64))
        .collect();
    (0..BANDS)
        .map(|b| {
            let (lo, mid, hi) = (points[b], points[b + 1], points[b + 2]);
            let norm = 2.0 / (hi - lo);
            (0..bins)
                .filter_map(|k| {
                    let hz = 8000.0 * k as f64 / (bins - 1) as f64;
                    let w = ((hz - lo) / (mid - lo)).min((hi - hz) / (hi - mid));
                    (w > 0.0).then_some((k, (w * norm) as f32))
                })
                .collect()
        })
        .collect()
}

/// The log-mel spectrogram of `samples` (16 kHz mono), frame-major
/// `[frames][BANDS]`, and its frames.
pub fn spectrogram(samples: &[f32]) -> (Vec<f32>, usize) {
    let len = samples.len();
    let frames = len / HOP + 1;
    let pad_right = PAD.max((FRAME - PAD).saturating_sub(len));
    let mirror = pad_right < len;
    // The frames the transform makes; the last is repeated past them.
    let made = ((len + PAD + pad_right).saturating_sub(FRAME) / HOP + 1).min(frames);
    let at = |i: isize| -> f32 {
        let n = len as isize;
        if (0..n).contains(&i) {
            samples[i as usize]
        } else if !mirror {
            0.0
        } else if i < 0 {
            samples[(-i) as usize]
        } else {
            samples[(2 * (n - 1) - i) as usize]
        }
    };
    let window: Vec<f32> = (0..FRAME)
        .map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / FRAME as f64).cos()) as f32)
        .collect();
    let bank = bank();
    let mut out = vec![0.0f32; frames * BANDS];
    par_chunks(&mut out, 64 * BANDS, threads(), |c, rows| {
        let mut fft = RealFft::new(FRAME);
        let mut frame = vec![0.0f32; FRAME];
        let mut spec = vec![Complex::default(); FRAME / 2 + 1];
        let mut mag = vec![0.0f32; FRAME / 2 + 1];
        for (r, row) in rows.chunks_exact_mut(BANDS).enumerate() {
            let f = (c * 64 + r).min(made - 1);
            let start = (f * HOP) as isize - PAD as isize;
            for (j, v) in frame.iter_mut().enumerate() {
                *v = at(start + j as isize) * window[j];
            }
            fft.spectrum(&frame, &mut spec);
            for (m, z) in mag.iter_mut().zip(&spec) {
                *m = (z.re * z.re + z.im * z.im + 1e-9).sqrt();
            }
            for (v, band) in row.iter_mut().zip(&bank) {
                let sum: f32 = band.iter().map(|&(k, w)| mag[k] * w).sum();
                *v = sum.max(1e-5).ln();
            }
        }
    });
    (out, frames)
}

/// Layer norm of each row of `x` (rows of `gamma.len()`), in place.
fn layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32]) {
    let n = gamma.len();
    for row in x.chunks_exact_mut(n) {
        let mean = row.iter().sum::<f32>() / n as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for ((v, g), b) in row.iter_mut().zip(gamma).zip(beta) {
            *v = (*v - mean) * inv * g + b;
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// What [`PitchNet::run_seeing`] shows the rows a weight matrix multiplies
/// to: the tensor's name and rows of its inputs.
pub type See<'a> = &'a (dyn Fn(&str, &[f32]) + Sync);

/// A convolution over 3 frames (zero beyond the song), as a linear map
/// of each frame and its neighbours.
struct Conv3 {
    name: String,
    inputs: usize,
    map: Linear,
}

impl Conv3 {
    fn load(w: &mut Weights, name: &str, inputs: usize, outputs: usize) -> Result<Conv3, String> {
        let weight = w.take(&format!("{name}.weight"), &[outputs, inputs, 3])?;
        let bias = w.take(&format!("{name}.bias"), &[outputs])?;
        // [out][in][k] to [out][k·in + in].
        let mut flat = vec![0.0f32; weight.len()];
        for o in 0..outputs {
            for c in 0..inputs {
                for k in 0..3 {
                    flat[o * 3 * inputs + k * inputs + c] = weight[(o * inputs + c) * 3 + k];
                }
            }
        }
        Ok(Conv3 {
            name: format!("{name}.weight"),
            inputs,
            map: Linear::new(flat, Some(bias), 3 * inputs, outputs),
        })
    }

    /// The convolution of `frames` frames of `x`; `see`, where given, is
    /// shown each frame with its neighbours, `[3][inputs]` a row.
    fn apply(&self, x: &[f32], frames: usize, see: Option<See>) -> Vec<f32> {
        let (cin, cout) = (self.inputs, self.map.outputs);
        let mut out = vec![0.0f32; frames * cout];
        par_chunks(&mut out, CHUNK * cout, threads(), |c, rows| {
            let first = c * CHUNK;
            let n = rows.len() / cout;
            let mut cols = vec![0.0f32; n * 3 * cin];
            for r in 0..n {
                for k in 0..3 {
                    let t = (first + r + k) as isize - 1;
                    if (0..frames as isize).contains(&t) {
                        let t = t as usize;
                        cols[(r * 3 + k) * cin..][..cin].copy_from_slice(&x[t * cin..][..cin]);
                    }
                }
            }
            if let Some(see) = see {
                see(&self.name, &cols);
            }
            rows.copy_from_slice(&self.map.apply(&cols, 1));
        });
        out
    }
}

/// One of the six layers.
struct Block {
    /// The names of `up`'s and `down`'s weights.
    names: [String; 2],
    gamma: Vec<f32>,
    beta: Vec<f32>,
    up: Linear,
    depth: Vec<f32>,
    depth_bias: Vec<f32>,
    down: Linear,
}

impl Block {
    fn load(w: &mut Weights, i: usize) -> Result<Block, String> {
        let p = format!("net.encoder_layers.{i}.conformer.net");
        let up = w.take(&format!("{p}.2.weight"), &[2 * INNER, DIM, 1])?;
        let up_bias = w.take(&format!("{p}.2.bias"), &[2 * INNER])?;
        let down = w.take(&format!("{p}.6.weight"), &[DIM, INNER, 1])?;
        let down_bias = w.take(&format!("{p}.6.bias"), &[DIM])?;
        Ok(Block {
            names: [format!("{p}.2.weight"), format!("{p}.6.weight")],
            gamma: w.take(&format!("{p}.0.weight"), &[DIM])?,
            beta: w.take(&format!("{p}.0.bias"), &[DIM])?,
            up: Linear::new(up, Some(up_bias), DIM, 2 * INNER),
            depth: w.take(&format!("{p}.4.conv.weight"), &[INNER, 1, KERNEL])?,
            depth_bias: w.take(&format!("{p}.4.conv.bias"), &[INNER])?,
            down: Linear::new(down, Some(down_bias), INNER, DIM),
        })
    }

    /// `x` and what the layer adds to it; `see`, where given, is shown
    /// what the pointwise convolutions multiply.
    fn apply(&self, x: &[f32], frames: usize, see: Option<See>) -> Vec<f32> {
        let half = KERNEL / 2;
        let mut next = x.to_vec();
        par_chunks(&mut next, CHUNK * DIM, threads(), |c, rows| {
            let first = c * CHUNK;
            let n = rows.len() / DIM;
            // The gated rows the depthwise convolution reads: the chunk and
            // `half` each side, zero beyond the song.
            let lo = first.saturating_sub(half);
            let hi = (first + n + half).min(frames);
            let mut normed = x[lo * DIM..hi * DIM].to_vec();
            layer_norm(&mut normed, &self.gamma, &self.beta);
            if let Some(see) = see {
                see(&self.names[0], &normed[(first - lo) * DIM..][..n * DIM]);
            }
            let wide = self.up.apply(&normed, 1);
            let span = n + 2 * half;
            let mut gated = vec![0.0f32; span * INNER];
            for (r, row) in wide.chunks_exact(2 * INNER).enumerate() {
                let at = (lo + r + half - first) * INNER;
                let g = &mut gated[at..at + INNER];
                for (k, v) in g.iter_mut().enumerate() {
                    *v = row[k] * sigmoid(row[INNER + k]);
                }
            }
            let mut mid = vec![0.0f32; n * INNER];
            for r in 0..n {
                let out = &mut mid[r * INNER..(r + 1) * INNER];
                out.copy_from_slice(&self.depth_bias);
                for j in 0..KERNEL {
                    let src = &gated[(r + j) * INNER..(r + j + 1) * INNER];
                    for (k, (o, s)) in out.iter_mut().zip(src).enumerate() {
                        *o += self.depth[k * KERNEL + j] * s;
                    }
                }
                for v in out.iter_mut() {
                    *v *= sigmoid(*v);
                }
            }
            if let Some(see) = see {
                see(&self.names[1], &mid);
            }
            for (o, d) in rows.iter_mut().zip(self.down.apply(&mid, 1)) {
                *o += d;
            }
        });
        next
    }
}

/// The network.
pub struct PitchNet {
    first: Conv3,
    group_gamma: Vec<f32>,
    group_beta: Vec<f32>,
    second: Conv3,
    blocks: Vec<Block>,
    gamma: Vec<f32>,
    beta: Vec<f32>,
    out: Linear,
    cents: Vec<f32>,
}

impl PitchNet {
    /// The network from its weights (either layout).
    pub fn load(mut w: Weights) -> Result<PitchNet, String> {
        let first = Conv3::load(&mut w, "input_stack.0", BANDS, DIM)?;
        let group_gamma = w.take("input_stack.1.weight", &[DIM])?;
        let group_beta = w.take("input_stack.1.bias", &[DIM])?;
        let second = Conv3::load(&mut w, "input_stack.3", DIM, DIM)?;
        let blocks = (0..LAYERS)
            .map(|i| Block::load(&mut w, i))
            .collect::<Result<Vec<_>, _>>()?;
        let gamma = w.take("norm.weight", &[DIM])?;
        let beta = w.take("norm.bias", &[DIM])?;
        let out = Linear::load(&mut w, "output_proj", DIM, CLASSES, true)?;
        let cents = w.take("cent_table", &[CLASSES])?;
        let left = w.left();
        if !left.is_empty() {
            return Err(format!("unused tensors: {left:?}"));
        }
        Ok(PitchNet {
            first,
            group_gamma,
            group_beta,
            second,
            blocks,
            gamma,
            beta,
            out,
            cents,
        })
    }

    /// The built-in network, loaded once.
    pub fn built_in() -> &'static PitchNet {
        static NET: OnceLock<PitchNet> = OnceLock::new();
        NET.get_or_init(|| {
            let w = Weights::parse(WEIGHTS).expect("the built-in pitch weights");
            PitchNet::load(w).expect("the built-in pitch network")
        })
    }

    /// The group norm's mean and inverse deviation for each of its four
    /// groups, over the whole song.
    fn group_stats(&self, mel: &[f32], frames: usize) -> [(f64, f64); 4] {
        let per = DIM / 4;
        let mut sums = [(0.0f64, 0.0f64); 4];
        let mut start = 0;
        while start < frames {
            let n = (frames - start).min(4 * WINDOW);
            // Each piece with a frame of its neighbours, so its edges are
            // as in the whole song.
            let lo = start.saturating_sub(1);
            let hi = (start + n + 1).min(frames);
            let x = self
                .first
                .apply(&mel[lo * BANDS..hi * BANDS], hi - lo, None);
            for row in x.chunks_exact(DIM).skip(start - lo).take(n) {
                for (g, sum) in sums.iter_mut().enumerate() {
                    for &v in &row[g * per..(g + 1) * per] {
                        sum.0 += f64::from(v);
                        sum.1 += f64::from(v) * f64::from(v);
                    }
                }
            }
            start += n;
        }
        let count = (frames.max(1) * per) as f64;
        sums.map(|(sum, squares)| {
            let mean = sum / count;
            let var = (squares / count - mean * mean).max(0.0);
            (mean, 1.0 / (var + 1e-5).sqrt())
        })
    }

    /// The sigmoids of `frames` frames of `mel`, the group norm's
    /// statistics given; each step's output kept in `steps`, where given,
    /// and `see`, where given, shown what each weight matrix multiplies.
    fn layers(
        &self,
        mel: &[f32],
        frames: usize,
        stats: &[(f64, f64); 4],
        mut steps: Option<&mut Vec<Vec<f32>>>,
        see: Option<See>,
    ) -> Vec<f32> {
        let mut x = self.first.apply(mel, frames, see);
        let per = DIM / 4;
        for row in x.chunks_exact_mut(DIM) {
            for (c, v) in row.iter_mut().enumerate() {
                let (mean, inv) = stats[c / per];
                let n = ((f64::from(*v) - mean) * inv) as f32;
                let n = n * self.group_gamma[c] + self.group_beta[c];
                *v = if n < 0.0 { 0.01 * n } else { n };
            }
        }
        x = self.second.apply(&x, frames, see);
        if let Some(s) = steps.as_deref_mut() {
            s.push(x.clone());
        }
        for block in &self.blocks {
            x = block.apply(&x, frames, see);
            if let Some(s) = steps.as_deref_mut() {
                s.push(x.clone());
            }
        }
        layer_norm(&mut x, &self.gamma, &self.beta);
        if let Some(see) = see {
            see("output_proj.weight", &x);
        }
        let mut y = self.out.apply(&x, threads());
        for v in &mut y {
            *v = sigmoid(*v);
        }
        y
    }

    /// The 360 sigmoids of each frame of `mel` (`[frames][BANDS]`), and,
    /// where `steps`, what the input stack and each layer gave
    /// (`[frames][512]` each; then the song goes through at once).
    pub fn run(&self, mel: &[f32], frames: usize, steps: bool) -> (Vec<f32>, Vec<Vec<f32>>) {
        let stats = self.group_stats(mel, frames);
        if steps {
            let mut kept = Vec::new();
            let y = self.layers(mel, frames, &stats, Some(&mut kept), None);
            return (y, kept);
        }
        let mut y = vec![0.0f32; frames * CLASSES];
        let mut start = 0;
        while start < frames {
            let n = (frames - start).min(WINDOW);
            let lo = start.saturating_sub(HALO);
            let hi = (start + n + HALO).min(frames);
            let part = self.layers(&mel[lo * BANDS..hi * BANDS], hi - lo, &stats, None, None);
            y[start * CLASSES..(start + n) * CLASSES]
                .copy_from_slice(&part[(start - lo) * CLASSES..][..n * CLASSES]);
            start += n;
        }
        (y, Vec::new())
    }

    /// The sigmoids [`PitchNet::run`] gives, the song at once, with `see`
    /// shown what each weight matrix multiplies (for the statistics a
    /// quantiser weighs its errors by): the tensor's name and rows of its
    /// inputs, a frame a row, in pieces of the song from several threads.
    /// The convolutions over 3 frames multiply a frame with its
    /// neighbours, a row `[3][inputs]` (the frame before first, zero
    /// beyond the song); the rest multiply a row of their input channels.
    pub fn run_seeing(&self, mel: &[f32], frames: usize, see: See) -> Vec<f32> {
        let stats = self.group_stats(mel, frames);
        self.layers(mel, frames, &stats, None, Some(see))
    }

    /// The pitch of each frame from its sigmoids.
    pub fn decode(&self, latent: &[f32]) -> Pitch {
        let mut p = Pitch::default();
        for row in latent.chunks_exact(CLASSES) {
            let (best, &top) = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap_or((0, &0.0));
            let (mut num, mut den) = (0.0f32, 0.0f32);
            for j in 0..9 {
                let k = (best + j).saturating_sub(4).min(CLASSES - 1);
                num += self.cents[k] * row[k];
                den += row[k];
            }
            let hz = if top > VOICED && den > 0.0 {
                10.0 * (num / den / 1200.0).exp2()
            } else {
                0.0
            };
            p.hz.push(hz);
            p.confidence.push(top);
        }
        p
    }
}

/// The pitch of a voice, `samples` its 16 kHz mono samples.
pub fn pitch_of(samples: &[f32]) -> Pitch {
    let net = PitchNet::built_in();
    let (mel, frames) = spectrogram(samples);
    let (latent, _) = net.run(&mel, frames, false);
    net.decode(&latent)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bands' weights sum as librosa's do: about 2 / width each, so a
    /// flat spectrum fills every band about equally per Hz.
    #[test]
    fn the_bank_covers_zero_to_eight_kilohertz() {
        let b = bank();
        assert_eq!(b.len(), BANDS);
        assert_eq!(b[0][0].0, 1);
        assert!(b[BANDS - 1].last().unwrap().0 <= FRAME / 2);
        for band in &b {
            assert!(!band.is_empty());
        }
    }

    #[test]
    fn frames_follow_the_length() {
        for len in [0usize, 100, 1024, 16_000, 16_123] {
            let (mel, frames) = spectrogram(&vec![0.1f32; len]);
            assert_eq!(frames, len / HOP + 1);
            assert_eq!(mel.len(), frames * BANDS);
        }
    }
}
