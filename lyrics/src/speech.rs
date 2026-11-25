//! The speech model on the CPU: `facebook/wav2vec2-base-960h` (Apache
//! 2.0), a CTC model of English letters, from its `model.safetensors`.
//!
//! The network, as the checkpoint has it:
//!
//! 1. Seven 1-D convolutions over the voice at 16 kHz (512 channels,
//!    kernels 10, 3, 3, 3, 3, 2, 2, strides 5, 2, 2, 2, 2, 2, 2, no
//!    biases), the first followed by a GroupNorm of one group a channel
//!    (over time), each by GELU: a frame every 320 samples (20 ms), seeing
//!    400.
//! 2. A LayerNorm and a projection from 512 to 768.
//! 3. A positional convolution (kernel 128, 16 groups of 48, its weight
//!    normalised per tap: `g · v / |v|`), GELU, added; a LayerNorm.
//! 4. Twelve transformer layers, the norm after each part: attention with
//!    12 heads of 64, then a feed-forward of 3,072 with GELU.
//! 5. A linear head to the 32 tokens ([`crate::tokens::WAV2VEC2_960H`]).
//!
//! The input is normalised to zero mean and unit variance first, as the
//! model's feature extractor does. Long voices go through in pieces of
//! [`PIECE_SECONDS`], each with [`CONTEXT_SECONDS`] of context either side
//! whose frames are dropped. Both normalisations over time, the input's
//! and the GroupNorm's, are measured once over the whole voice
//! ([`Levels`]), so a piece is heard as it would be in the whole song.
//!
//! [`Wav2Vec2`] is the reference for the GPU's version and runs where
//! there is no GPU: about a minute for a 4-minute song on 16 threads.

use std::path::Path;

use crate::aligner::Acoustic;
use crate::ctc::Emissions;
use crate::nn::{attend, gelu, matmul, Heads, LayerNorm, Linear};
use crate::{Maker, VOCALS_RATE};
use jodel_tensors::SafeTensors;

/// Seconds between the model's frames (one every 320 samples).
pub const FRAME_SECONDS: f64 = 0.02;
/// The samples one frame sees.
pub const RECEPTIVE: usize = 400;
/// Samples between frames.
pub const HOP: usize = 320;
/// The voice is run in pieces of this many seconds.
pub const PIECE_SECONDS: f64 = 30.0;
/// The context either side of a piece, whose frames are dropped.
pub const CONTEXT_SECONDS: f64 = 2.0;

/// The convolutions' kernels and strides.
pub const CONV: [(usize, usize); 7] = [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)];
const CHANNELS: usize = 512;
const DIM: usize = 768;
const HEADS: usize = 12;
const HEAD: usize = 64;
const FF: usize = 3072;
const LAYERS: usize = 12;
const POS_KERNEL: usize = 128;
const POS_GROUPS: usize = 16;
const TOKENS: usize = 32;
const EPS: f32 = 1e-5;

struct Layer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    norm: LayerNorm,
    up: Linear,
    down: Linear,
    final_norm: LayerNorm,
}

/// wav2vec2-base-960h on the CPU.
pub struct Wav2Vec2 {
    /// The first convolution, `[512][10]`, and its GroupNorm's scale and
    /// shift (a group a channel).
    conv0: Vec<f32>,
    group_weight: Vec<f32>,
    group_bias: Vec<f32>,
    /// The other six, as `[k·512][512]` (tap-major inputs).
    convs: Vec<Vec<f32>>,
    feature_norm: LayerNorm,
    projection: Linear,
    /// The positional convolution per group, `[128·48][48]`, and its bias.
    pos: Vec<Vec<f32>>,
    pos_bias: Vec<f32>,
    encoder_norm: LayerNorm,
    layers: Vec<Layer>,
    head: Linear,
    threads: usize,
}

impl Wav2Vec2 {
    /// The model in `model.safetensors` at `path`.
    pub fn open(path: &Path) -> Result<Wav2Vec2, String> {
        Wav2Vec2::load(&SafeTensors::open(path)?)
    }

    /// The model from its tensors.
    pub fn load(t: &SafeTensors) -> Result<Wav2Vec2, String> {
        let fe = "wav2vec2.feature_extractor.conv_layers";
        let conv0 = t.get(&format!("{fe}.0.conv.weight"), &[CHANNELS, 1, 10])?;
        let group_weight = t.get(&format!("{fe}.0.layer_norm.weight"), &[CHANNELS])?;
        let group_bias = t.get(&format!("{fe}.0.layer_norm.bias"), &[CHANNELS])?;
        let mut convs = Vec::new();
        for (l, &(k, _)) in CONV.iter().enumerate().skip(1) {
            let raw = t.get(&format!("{fe}.{l}.conv.weight"), &[CHANNELS, CHANNELS, k])?;
            // To [tap][input][output].
            let mut w = vec![0.0f32; raw.len()];
            for o in 0..CHANNELS {
                for i in 0..CHANNELS {
                    for j in 0..k {
                        w[(j * CHANNELS + i) * CHANNELS + o] = raw[(o * CHANNELS + i) * k + j];
                    }
                }
            }
            convs.push(w);
        }
        let feature_norm = LayerNorm::load(t, "wav2vec2.feature_projection.layer_norm", CHANNELS)?;
        let projection = Linear::load(t, "wav2vec2.feature_projection.projection", CHANNELS, DIM)?;
        let per = DIM / POS_GROUPS;
        let pc = "wav2vec2.encoder.pos_conv_embed.conv";
        let g = t.get(&format!("{pc}.weight_g"), &[1, 1, POS_KERNEL])?;
        let v = t.get(&format!("{pc}.weight_v"), &[DIM, per, POS_KERNEL])?;
        // The weight norm: each tap's weights over all outputs and inputs.
        let mut norms = vec![0.0f64; POS_KERNEL];
        for (n, x) in v.iter().enumerate() {
            norms[n % POS_KERNEL] += f64::from(*x) * f64::from(*x);
        }
        let mut pos = vec![vec![0.0f32; POS_KERNEL * per * per]; POS_GROUPS];
        for o in 0..DIM {
            let (group, ol) = (o / per, o % per);
            for i in 0..per {
                for j in 0..POS_KERNEL {
                    let w = f64::from(g[j]) * f64::from(v[(o * per + i) * POS_KERNEL + j])
                        / norms[j].sqrt();
                    pos[group][(j * per + i) * per + ol] = w as f32;
                }
            }
        }
        let pos_bias = t.get(&format!("{pc}.bias"), &[DIM])?;
        let encoder_norm = LayerNorm::load(t, "wav2vec2.encoder.layer_norm", DIM)?;
        let mut layers = Vec::new();
        for l in 0..LAYERS {
            let n = format!("wav2vec2.encoder.layers.{l}");
            layers.push(Layer {
                q: Linear::load(t, &format!("{n}.attention.q_proj"), DIM, DIM)?,
                k: Linear::load(t, &format!("{n}.attention.k_proj"), DIM, DIM)?,
                v: Linear::load(t, &format!("{n}.attention.v_proj"), DIM, DIM)?,
                o: Linear::load(t, &format!("{n}.attention.out_proj"), DIM, DIM)?,
                norm: LayerNorm::load(t, &format!("{n}.layer_norm"), DIM)?,
                up: Linear::load(t, &format!("{n}.feed_forward.intermediate_dense"), DIM, FF)?,
                down: Linear::load(t, &format!("{n}.feed_forward.output_dense"), FF, DIM)?,
                final_norm: LayerNorm::load(t, &format!("{n}.final_layer_norm"), DIM)?,
            });
        }
        let head = Linear::load(t, "lm_head", DIM, TOKENS)?;
        Ok(Wav2Vec2 {
            conv0,
            group_weight,
            group_bias,
            convs,
            feature_norm,
            projection,
            pos,
            pos_bias,
            encoder_norm,
            layers,
            head,
            threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
        })
    }

    /// Hands each tensor to `f`, by name, in the layouts this version
    /// computes with (a GPU's version takes them as they are):
    ///
    /// - `conv0` `[512][10]`; `gn.w`, `gn.b` (the GroupNorm);
    /// - `conv1` … `conv6` `[k·512][512]` (tap-major inputs);
    /// - `fp.ln.w`, `fp.ln.b`; `proj.w` `[512][768]`, `proj.b`;
    /// - `pos0` … `pos15` `[128·48][48]` (the weight norm applied), `pos.b`;
    /// - `enc.ln.w`, `enc.ln.b`;
    /// - per layer `l`: `{l}.q.w` … `{l}.o.w` `[768][768]` and their `.b`,
    ///   `{l}.ln.w`, `{l}.ln.b`, `{l}.up.w` `[768][3072]`, `{l}.up.b`,
    ///   `{l}.down.w` `[3072][768]`, `{l}.down.b`, `{l}.fln.w`, `{l}.fln.b`;
    /// - `head.w` `[768][32]`, `head.b`.
    pub fn visit(&self, f: &mut dyn FnMut(&str, &[f32])) {
        f("conv0", &self.conv0);
        f("gn.w", &self.group_weight);
        f("gn.b", &self.group_bias);
        for (l, w) in self.convs.iter().enumerate() {
            f(&format!("conv{}", l + 1), w);
        }
        f("fp.ln.w", &self.feature_norm.weight);
        f("fp.ln.b", &self.feature_norm.bias);
        let linear = |f: &mut dyn FnMut(&str, &[f32]), name: &str, l: &Linear| {
            f(&format!("{name}.w"), &l.weight);
            if let Some(b) = &l.bias {
                f(&format!("{name}.b"), b);
            }
        };
        let norm = |f: &mut dyn FnMut(&str, &[f32]), name: &str, n: &LayerNorm| {
            f(&format!("{name}.w"), &n.weight);
            f(&format!("{name}.b"), &n.bias);
        };
        linear(f, "proj", &self.projection);
        for (g, w) in self.pos.iter().enumerate() {
            f(&format!("pos{g}"), w);
        }
        f("pos.b", &self.pos_bias);
        norm(f, "enc.ln", &self.encoder_norm);
        for (i, l) in self.layers.iter().enumerate() {
            linear(f, &format!("{i}.q"), &l.q);
            linear(f, &format!("{i}.k"), &l.k);
            linear(f, &format!("{i}.v"), &l.v);
            linear(f, &format!("{i}.o"), &l.o);
            norm(f, &format!("{i}.ln"), &l.norm);
            linear(f, &format!("{i}.up"), &l.up);
            linear(f, &format!("{i}.down"), &l.down);
            norm(f, &format!("{i}.fln"), &l.final_norm);
        }
        linear(f, "head", &self.head);
    }

    /// Runs on at most `threads` threads (all the processor has, at first).
    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads.max(1);
    }

    /// The frames `samples` give.
    pub fn frames_of(samples: usize) -> usize {
        if samples < RECEPTIVE {
            0
        } else {
            (samples - RECEPTIVE) / HOP + 1
        }
    }

    /// The levels of a whole `voice` (at 16 kHz), for [`Wav2Vec2::logits_at`].
    pub fn levels(&self, voice: &[f32]) -> Levels {
        Levels::of(&self.conv0, voice)
    }

    /// The conv features (`frames × 512`, after the last GELU) of a voice
    /// normalised by `levels` ([`Levels::normalise`]), its first
    /// convolution's channels brought to `levels`' too.
    pub fn features(&self, x: &[f32], levels: &Levels) -> Vec<f32> {
        let (k0, s0) = CONV[0];
        let t0 = if x.len() < k0 {
            0
        } else {
            (x.len() - k0) / s0 + 1
        };
        // The first convolution, channels last.
        let mut h = vec![0.0f32; t0 * CHANNELS];
        for t in 0..t0 {
            let window = &x[t * s0..t * s0 + k0];
            let row = &mut h[t * CHANNELS..(t + 1) * CHANNELS];
            for (c, v) in row.iter_mut().enumerate() {
                let w = &self.conv0[c * k0..(c + 1) * k0];
                *v = window.iter().zip(w).map(|(a, b)| a * b).sum();
            }
        }
        // GroupNorm with a group a channel: over time, never lifting the
        // piece too far ([`Levels`]).
        let channels = levels.channels(x);
        for row in h.chunks_exact_mut(CHANNELS) {
            for (c, v) in row.iter_mut().enumerate() {
                let (mean, k) = (channels[2 * c], channels[2 * c + 1]);
                let (g, b) = (self.group_weight[c], self.group_bias[c]);
                *v = (*v - mean) * k * g + b;
            }
        }
        gelu(&mut h);
        let mut frames = t0;
        for (w, &(k, s)) in self.convs.iter().zip(&CONV[1..]) {
            let next = if frames < k { 0 } else { (frames - k) / s + 1 };
            // The windows side by side: a row of k·512 a frame.
            let mut patches = vec![0.0f32; next * k * CHANNELS];
            for t in 0..next {
                let from = t * s * CHANNELS;
                patches[t * k * CHANNELS..(t + 1) * k * CHANNELS]
                    .copy_from_slice(&h[from..from + k * CHANNELS]);
            }
            let mut out = vec![0.0f32; next * CHANNELS];
            matmul(w, k * CHANNELS, CHANNELS, &patches, &mut out, self.threads);
            gelu(&mut out);
            h = out;
            frames = next;
        }
        h
    }

    /// The hidden states entering the first layer (`frames × 768`) from
    /// the conv features.
    pub fn embed(&self, mut features: Vec<f32>) -> Vec<f32> {
        self.feature_norm.apply(&mut features);
        let mut h = self.projection.forward(&features, self.threads);
        let frames = h.len() / DIM;
        let per = DIM / POS_GROUPS;
        let pad = POS_KERNEL / 2;
        let mut pos = vec![0.0f32; frames * DIM];
        for g in 0..POS_GROUPS {
            // Each frame's 128 taps of this group's 48 channels (zeros
            // past the ends): output t sees inputs t - 64 .. t + 63.
            let mut patches = vec![0.0f32; frames * POS_KERNEL * per];
            for t in 0..frames {
                for j in 0..POS_KERNEL {
                    let src = t as i64 + j as i64 - pad as i64;
                    if src < 0 || src >= frames as i64 {
                        continue;
                    }
                    let from = src as usize * DIM + g * per;
                    let to = (t * POS_KERNEL + j) * per;
                    patches[to..to + per].copy_from_slice(&h[from..from + per]);
                }
            }
            let mut out = vec![0.0f32; frames * per];
            matmul(
                &self.pos[g],
                POS_KERNEL * per,
                per,
                &patches,
                &mut out,
                self.threads,
            );
            for t in 0..frames {
                pos[t * DIM + g * per..t * DIM + (g + 1) * per]
                    .copy_from_slice(&out[t * per..(t + 1) * per]);
            }
        }
        for row in pos.chunks_exact_mut(DIM) {
            for (v, b) in row.iter_mut().zip(&self.pos_bias) {
                *v += b;
            }
        }
        gelu(&mut pos);
        for (a, b) in h.iter_mut().zip(&pos) {
            *a += b;
        }
        self.encoder_norm.apply(&mut h);
        h
    }

    /// One transformer layer, in place.
    fn layer(&self, l: &Layer, h: &mut [f32]) {
        let frames = h.len() / DIM;
        let q = l.q.forward(h, self.threads);
        let k = l.k.forward(h, self.threads);
        let v = l.v.forward(h, self.threads);
        let heads = Heads {
            query: HEADS,
            kv: HEADS,
            width: HEAD,
        };
        let attended = attend(&q, &k, &v, heads, |_| 0..frames, self.threads);
        let o = l.o.forward(&attended, self.threads);
        for (a, b) in h.iter_mut().zip(&o) {
            *a += b;
        }
        l.norm.apply(h);
        let mut up = l.up.forward(h, self.threads);
        gelu(&mut up);
        let down = l.down.forward(&up, self.threads);
        for (a, b) in h.iter_mut().zip(&down) {
            *a += b;
        }
        l.final_norm.apply(h);
    }

    /// The hidden states after the last layer, from those entering the
    /// first.
    pub fn encode(&self, mut h: Vec<f32>) -> Vec<f32> {
        for l in &self.layers {
            self.layer(l, &mut h);
        }
        h
    }

    /// The logits (`frames × 32`) of a voice at 16 kHz heard on its own,
    /// in one piece: normalised at its own levels, then the whole network.
    pub fn logits(&self, voice: &[f32]) -> Vec<f32> {
        self.logits_at(voice, &self.levels(voice))
    }

    /// The logits (`frames × 32`) of a piece of a voice at 16 kHz, heard
    /// at the whole voice's `levels`.
    pub fn logits_at(&self, piece: &[f32], levels: &Levels) -> Vec<f32> {
        let x = levels.normalise(piece);
        let h = self.encode(self.embed(self.features(&x, levels)));
        self.head.forward(&h, self.threads)
    }
}

/// The first convolution's kernel and stride.
const K0: usize = CONV[0].0;
const S0: usize = CONV[0].1;

/// The sums of the first convolution's windows (`K0` samples, `S0`
/// apart), their samples' and their samples' products: what each channel's
/// mean and variance over those windows follow from, the convolution being
/// linear and without a bias.
#[derive(Clone, Debug, Default)]
struct Moments {
    count: f64,
    first: [f64; K0],
    second: [[f64; K0]; K0],
}

impl Moments {
    /// The windows of `x` that start in `from..to` (sample numbers, a
    /// window every `S0`).
    fn of(x: &[f32], from: usize, to: usize) -> Moments {
        let mut m = Moments::default();
        let mut start = from.div_ceil(S0) * S0;
        while start < to && start + K0 <= x.len() {
            let w: [f64; K0] = std::array::from_fn(|j| f64::from(x[start + j]));
            for i in 0..K0 {
                m.first[i] += w[i];
                for j in i..K0 {
                    m.second[i][j] += w[i] * w[j];
                }
            }
            m.count += 1.0;
            start += S0;
        }
        m
    }

    fn add(&mut self, other: &Moments) {
        self.count += other.count;
        for i in 0..K0 {
            self.first[i] += other.first[i];
            for j in i..K0 {
                self.second[i][j] += other.second[i][j];
            }
        }
    }

    /// The mean and variance over the windows of the channel of `weights`.
    fn channel(&self, weights: &[f32]) -> (f64, f64) {
        let w: [f64; K0] = std::array::from_fn(|j| f64::from(weights[j]));
        let count = self.count.max(1.0);
        let mean = (0..K0).map(|j| w[j] * self.first[j]).sum::<f64>() / count;
        let mut squares = 0.0;
        for i in 0..K0 {
            squares += w[i] * w[i] * self.second[i][i];
            for j in i + 1..K0 {
                squares += 2.0 * w[i] * w[j] * self.second[i][j];
            }
        }
        (mean, (squares / count - mean * mean).max(0.0))
    }

    /// The windows' mean power (their samples').
    fn power(&self) -> f64 {
        (0..K0).map(|i| self.second[i][i]).sum::<f64>() / (self.count.max(1.0) * K0 as f64)
    }
}

/// How far the model may lift a quiet piece: never more than this many dB
/// above where the song's singing is.
pub const LIFT_DB: f64 = 15.0;
/// The seconds of the song where singing is: those within this many dB
/// of its loud seconds (the 95th percentile).
pub const SUNG_DB: f64 = 20.0;

/// How loud a voice is, as the model measures it. The model brings what
/// it hears to one level twice: the voice to zero mean and unit variance
/// (its feature extractor), and each channel of its first convolution to
/// zero mean and unit variance over time (the GroupNorm, which undoes any
/// gain before it). Over each piece on its own, as the model was trained
/// (on utterances of some seconds), a quiet verse is heard as well as a
/// loud chorus, but a piece with no singing (an intro's leftovers of the
/// separation, 40 dB under the voice) is lifted to the level of speech
/// too, and the model hears letters in that noise. So each piece is
/// brought to its own level as before, but never lifted more than
/// [`LIFT_DB`] above where the whole song's singing is: a channel's
/// variance is taken as at least the song's singing's, that far down.
#[derive(Clone, Debug, PartialEq)]
pub struct Levels {
    /// The voice's mean.
    pub mean: f64,
    /// What scales the voice to unit variance.
    pub scale: f64,
    /// The first convolution, `[512][10]`.
    conv0: Vec<f32>,
    /// Each channel's least variance.
    floor: Vec<f64>,
}

impl Levels {
    /// The levels of `voice` (at 16 kHz) through the first convolution
    /// `conv0` (`[512][10]`).
    pub fn of(conv0: &[f32], voice: &[f32]) -> Levels {
        let n = voice.len().max(1) as f64;
        let mean = voice.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let var = voice
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / n;
        let scale = 1.0 / (var + 1e-7).sqrt();
        let mut levels = Levels {
            mean,
            scale,
            conv0: conv0.to_vec(),
            floor: Vec::new(),
        };
        // The song's seconds, and those with singing in them.
        let x = levels.normalise(voice);
        let second = VOCALS_RATE as usize;
        let seconds: Vec<Moments> = (0..x.len().div_ceil(second))
            .map(|s| Moments::of(&x, s * second, (s + 1) * second))
            .filter(|m| m.count > 0.0)
            .collect();
        let mut powers: Vec<f64> = seconds.iter().map(Moments::power).collect();
        powers.sort_by(f64::total_cmp);
        let loud = powers.get(powers.len() * 95 / 100).copied().unwrap_or(0.0);
        let sung_at = loud * 10f64.powf(-SUNG_DB / 10.0);
        let mut sung = Moments::default();
        for m in seconds.iter().filter(|m| m.power() >= sung_at) {
            sung.add(m);
        }
        let lift = 10f64.powf(-LIFT_DB / 10.0);
        levels.floor = conv0
            .chunks_exact(K0)
            .map(|w| sung.channel(w).1 * lift)
            .collect();
        levels
    }

    /// `piece` of the voice at zero mean and unit variance by the whole
    /// voice's.
    pub fn normalise(&self, piece: &[f32]) -> Vec<f32> {
        piece
            .iter()
            .map(|&v| ((f64::from(v) - self.mean) * self.scale) as f32)
            .collect()
    }

    /// Each channel of the first convolution over a piece `x` (as
    /// [`Levels::normalise`] gives it): its mean and what scales it to
    /// unit variance, side by side (`2 × 512`), its variance at least the
    /// channel's least.
    pub fn channels(&self, x: &[f32]) -> Vec<f32> {
        let m = Moments::of(x, 0, x.len());
        let mut out = Vec::with_capacity(2 * CHANNELS);
        for (w, &floor) in self.conv0.chunks_exact(K0).zip(&self.floor) {
            let (mean, var) = m.channel(w);
            out.push(mean as f32);
            out.push((1.0 / (var.max(floor) + f64::from(EPS)).sqrt()) as f32);
        }
        out
    }
}

/// Log-softmax of each row of `logits` (`tokens` a row), in place.
pub fn log_softmax(logits: &mut [f32], tokens: usize) {
    for row in logits.chunks_exact_mut(tokens) {
        let best = row.iter().copied().fold(f32::MIN, f32::max);
        let sum: f32 = row.iter().map(|&v| (v - best).exp()).sum();
        let log = best + sum.ln();
        for v in row.iter_mut() {
            *v -= log;
        }
    }
}

/// The emissions of a whole voice: pieces of [`PIECE_SECONDS`] with
/// [`CONTEXT_SECONDS`] either side, each piece's own frames kept, by
/// `logits` (the CPU's [`Wav2Vec2::logits`], or a GPU's). `None` when
/// `logits` gives none for a piece, or fewer frames than the piece has
/// ([`Wav2Vec2::frames_of`]).
pub fn emissions_in_pieces(
    voice: &[f32],
    mut logits: impl FnMut(&[f32]) -> Option<Vec<f32>>,
) -> Option<Emissions> {
    let rate = VOCALS_RATE as usize;
    let total = Wav2Vec2::frames_of(voice.len());
    let per_frame = HOP;
    let piece = (PIECE_SECONDS * rate as f64) as usize / per_frame;
    let context = (CONTEXT_SECONDS * rate as f64) as usize / per_frame;
    let mut out = Vec::with_capacity(total * TOKENS);
    let mut first = 0;
    while first < total {
        let last = (first + piece).min(total);
        // The piece's frames, with context either side.
        let from = first.saturating_sub(context);
        let to = (last + context).min(total);
        let start = from * per_frame;
        let end = ((to - 1) * per_frame + RECEPTIVE).min(voice.len());
        let mut l = logits(&voice[start..end])?;
        // A model gives a piece the frames of its convolutions
        // ([`Wav2Vec2::frames_of`]): every one of `from..to`.
        let (skip, keep) = (first - from, last - first);
        let got = l.len() / TOKENS;
        debug_assert!(got >= skip + keep, "{got} frames, {} asked", skip + keep);
        if got < skip + keep {
            return None;
        }
        log_softmax(&mut l, TOKENS);
        out.extend_from_slice(&l[skip * TOKENS..(skip + keep) * TOKENS]);
        first = last;
    }
    Some(Emissions {
        logp: out,
        tokens: TOKENS,
        frame_seconds: FRAME_SECONDS,
    })
}

/// The speech model's name and version, the same on the CPU and the GPU
/// (`jodel_gpu::GpuSpeech`): a change to what it hears raises the version,
/// and timings made before are made again. 2: a quiet piece lifted at most
/// [`LIFT_DB`].
pub fn maker() -> Maker {
    Maker::new("wav2vec2-base-960h", 2)
}

impl Acoustic for Wav2Vec2 {
    fn maker(&self) -> Maker {
        maker()
    }

    fn emissions(&self, vocals_16k: &[f32]) -> Option<Emissions> {
        let levels = self.levels(vocals_16k);
        emissions_in_pieces(vocals_16k, |piece| Some(self.logits_at(piece, &levels)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_frames_are_those_of_the_convolutions() {
        assert_eq!(Wav2Vec2::frames_of(399), 0);
        assert_eq!(Wav2Vec2::frames_of(400), 1);
        assert_eq!(Wav2Vec2::frames_of(16_000), 49);
        assert_eq!(Wav2Vec2::frames_of(48_000), 149);
        assert_eq!(Wav2Vec2::frames_of(192_000), 599);
        // The convolutions one after the other give as many (as
        // `Wav2Vec2::features` runs them), so a piece gets every frame
        // `emissions_in_pieces` asks of it.
        for samples in 0..20_000 {
            let frames = CONV.iter().fold(
                samples,
                |n, &(k, s)| if n < k { 0 } else { (n - k) / s + 1 },
            );
            assert_eq!(frames, Wav2Vec2::frames_of(samples), "{samples}");
        }
    }

    #[test]
    fn a_quiet_piece_is_lifted_no_further_than_the_singing_allows() {
        // Noise: 60 s loud, 20 s 10 dB under, 20 s 60 dB under (an
        // intro's leftovers of the separation). Two channels: the sample
        // and a difference of neighbours.
        let mut state = 7u64;
        let mut noise = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        };
        let rate = VOCALS_RATE as usize;
        let voice: Vec<f32> = (0..100 * rate)
            .map(|i| {
                let gain = match i / rate {
                    0..60 => 0.3,
                    60..80 => 0.3 * 10f32.powf(-0.5),
                    _ => 0.3e-3,
                };
                gain * noise()
            })
            .collect();
        let mut conv0 = vec![0.0f32; 2 * K0];
        conv0[0] = 1.0;
        conv0[K0] = 1.0;
        conv0[K0 + 1] = -1.0;
        let levels = Levels::of(&conv0, &voice);
        let x = levels.normalise(&voice);
        // A channel's variance over a stretch's windows, summed directly.
        let var = |c: usize, from: usize, to: usize| {
            let w = &conv0[c * K0..(c + 1) * K0];
            let out: Vec<f64> = (from..to)
                .step_by(S0)
                .filter(|s| s + K0 <= x.len())
                .map(|s| (0..K0).map(|j| f64::from(w[j] * x[s + j])).sum())
                .collect();
            let mean = out.iter().sum::<f64>() / out.len() as f64;
            out.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / out.len() as f64
        };
        let scale = |from: usize, to: usize| levels.channels(&x[from..to]);
        let near = |got: f32, want: f64| ((f64::from(got) - want) / want).abs() < 1e-3;
        // The loud piece and the one 10 dB under: each at its own level.
        for (from, to) in [(0, 30 * rate), (60 * rate, 80 * rate)] {
            let k = scale(from, to);
            for c in 0..2 {
                let want = 1.0 / (var(c, from, to) + f64::from(EPS)).sqrt();
                assert!(
                    near(k[2 * c + 1], want),
                    "{from} {c}: {} {want}",
                    k[2 * c + 1]
                );
            }
        }
        // The noise: lifted only to LIFT_DB under the singing (the first
        // 80 s, all within SUNG_DB of the loudest).
        let k = scale(80 * rate, 100 * rate);
        for c in 0..2 {
            let sung = (var(c, 0, 60 * rate) * 60.0 + var(c, 60 * rate, 80 * rate) * 20.0) / 80.0;
            let floor = sung * 10f64.powf(-LIFT_DB / 10.0);
            let want = 1.0 / (floor + f64::from(EPS)).sqrt();
            assert!(near(k[2 * c + 1], want), "{c}: {} {want}", k[2 * c + 1]);
        }
    }

    #[test]
    fn pieces_keep_their_own_frames() {
        // A fake model that marks each frame by its place in the whole
        // voice (the sample number its window starts at): the token of
        // that frame number, modulo the tokens.
        let voice: Vec<f32> = (0..16_000 * 70).map(|i| i as f32).collect();
        let e = emissions_in_pieces(&voice, |piece| {
            let frames = Wav2Vec2::frames_of(piece.len());
            let mut l = vec![-100.0f32; frames * TOKENS];
            for f in 0..frames {
                let frame = (piece[f * HOP] as usize) / HOP;
                l[f * TOKENS + frame % TOKENS] = 0.0;
            }
            Some(l)
        })
        .unwrap();
        assert_eq!(e.frames(), Wav2Vec2::frames_of(voice.len()));
        for f in 0..e.frames() {
            assert!(e.at(f, (f % TOKENS) as u32) > -1e-3, "frame {f}");
        }
    }
}
