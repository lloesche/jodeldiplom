//! A speech recogniser on the CPU: `Qwen/Qwen3-ASR-0.6B` (Apache 2.0),
//! from its `model.safetensors`. It writes what is sung, in any of 30
//! languages, as text: not a replacement for fetched lyrics (it gets about
//! a third of a song's words wrong), but enough to tell whether fetched
//! lyrics fit a recording and where their lines are.
//!
//! The network, as the checkpoint has it:
//!
//! 1. **Features** ([`mel`]): 128 log-mels, 100 frames a second.
//! 2. **The audio encoder**, on pieces of one second (100 frames; the last
//!    one zero-padded to that):
//!    - three 3×3 convolutions with stride 2 (480 channels, GELU each),
//!      13 steps a piece of 16 bands, flattened (7,680) and projected to
//!      896, plus sinusoidal positions 0–12 within the piece;
//!    - 18 transformer layers, the norm before each part (attention with
//!      14 heads of 64 within windows of 8 pieces, 104 steps; a
//!      feed-forward of 3,584 with GELU);
//!    - a LayerNorm, 896 → 896 with GELU, then 896 → 1,024.
//! 3. **The text decoder** (Qwen3's 0.6B): the prompt ([`text::prompt`])
//!    with the audio's steps in place of its audio tokens; 28 layers, the
//!    RMS norm before each part: attention with 16 query heads and 8
//!    key/value heads of 128 (each RMS-normed), rotary positions (base
//!    10⁶); a SwiGLU feed-forward of 3,072. Greedy decoding to its end
//!    token.
//!
//! [`Qwen3Asr`] is the reference for the GPU's version.

pub mod mel;
pub mod packed;
mod parts;
pub mod quant;
pub mod text;
mod weights;

pub use parts::{
    loop_start, parts_of, transcribe_parts, Part, Transcript, PART_LEAST, PART_SECONDS,
};

use std::convert::Infallible;
use std::path::Path;

use crate::nn::{attend, dot, gelu, halved, par_rows, rms_norm, Conv, Heads, LayerNorm, Linear};
use jodel_tensors::SafeTensors;

/// Channels of the encoder's convolutions.
pub const CONV: usize = 480;
/// Width of the encoder's steps.
pub const ENC_DIM: usize = 896;
/// The encoder's attention heads.
pub const ENC_HEADS: usize = 14;
/// Width of an encoder head.
pub const ENC_HEAD: usize = ENC_DIM / ENC_HEADS;
/// Width of the encoder's feed-forward.
pub const ENC_FF: usize = 3584;
/// The encoder's layers.
pub const ENC_LAYERS: usize = 18;
/// Frames of features a piece.
pub const PIECE: usize = 100;
/// Pieces an attention window of the encoder.
pub const WINDOW_PIECES: usize = 8;
/// The encoder's sinusoidal positions (more than a piece needs).
pub const MAX_POSITIONS: usize = 1500;
/// Width of the decoder's tokens.
pub const DIM: usize = 1024;
/// The decoder's query heads.
pub const HEADS: usize = 16;
/// The decoder's key and value heads.
pub const KV_HEADS: usize = 8;
/// Width of a decoder head.
pub const HEAD: usize = 128;
/// Width of the decoder's feed-forward.
pub const FF: usize = 3072;
/// The decoder's layers.
pub const LAYERS: usize = 28;
/// Tokens of the vocabulary.
pub const VOCAB: usize = 151_936;
/// The base of the decoder's rotary positions.
pub const ROPE_BASE: f64 = 1e6;

/// The bands of the encoder's steps, after the convolutions.
const BANDS: usize = piece_steps(mel::MELS);

/// The projection of the encoder's flattened steps as the checkpoint has it
/// (`[896][channel·16 + band]`), its inputs band-major instead, as the
/// steps are flattened: `[896][band·480 + channel]`.
fn band_major(raw: &[f32]) -> Vec<f32> {
    let mut rows = vec![0.0f32; raw.len()];
    for o in 0..ENC_DIM {
        for c in 0..CONV {
            for b in 0..BANDS {
                rows[o * CONV * BANDS + b * CONV + c] = raw[o * CONV * BANDS + c * BANDS + b];
            }
        }
    }
    rows
}

/// The encoder's steps for a piece of `frames` frames.
pub const fn piece_steps(frames: usize) -> usize {
    halved(halved(halved(frames)))
}

/// The encoder's steps for `frames` frames of features.
pub fn audio_steps(frames: usize) -> usize {
    (frames / PIECE) * piece_steps(PIECE) + piece_steps(frames % PIECE)
}

struct EncoderLayer {
    norm: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    final_norm: LayerNorm,
    up: Linear,
    down: Linear,
}

struct DecoderLayer {
    norm: Vec<f32>,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    post_norm: Vec<f32>,
    gate: Linear,
    up: Linear,
    down: Linear,
}

/// Qwen3-ASR-0.6B on the CPU.
pub struct Qwen3Asr {
    conv1: Conv,
    conv2: Conv,
    conv3: Conv,
    /// `[band·480 + channel][896]`: the flattened steps' projection.
    conv_out: Linear,
    positions: Vec<f32>,
    encoder: Vec<EncoderLayer>,
    post_norm: LayerNorm,
    proj1: Linear,
    proj2: Linear,
    /// `[token][1024]`.
    embed: Vec<f32>,
    decoder: Vec<DecoderLayer>,
    norm: Vec<f32>,
    /// `[token][1024]`.
    head: Vec<f32>,
    vocab: text::Vocab,
    threads: usize,
}

/// The decoder's keys and values so far, a layer each: `[position][8·128]`.
pub struct Cache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    len: usize,
}

impl Cache {
    fn new() -> Cache {
        Cache {
            keys: vec![Vec::new(); LAYERS],
            values: vec![Vec::new(); LAYERS],
            len: 0,
        }
    }
}

/// Sinusoidal positions, `[position][dim]`: sines then cosines of
/// timescales from 1 to 10,000.
pub fn sinusoids(length: usize, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let step = 10_000f64.ln() / (half - 1) as f64;
    let mut out = vec![0.0f32; length * dim];
    for t in 0..length {
        for i in 0..half {
            let a = t as f64 * (-step * i as f64).exp();
            out[t * dim + i] = a.sin() as f32;
            out[t * dim + half + i] = a.cos() as f32;
        }
    }
    out
}

impl Qwen3Asr {
    /// The model in the folder `dir`: `model.safetensors`, `vocab.json`,
    /// `tokenizer_config.json`.
    pub fn open(dir: &Path) -> Result<Qwen3Asr, String> {
        let t = SafeTensors::open(&dir.join("model.safetensors"))?;
        Qwen3Asr::load(&t, text::Vocab::open(dir)?)
    }

    /// The model from its tensors and vocabulary.
    pub fn load(t: &SafeTensors, vocab: text::Vocab) -> Result<Qwen3Asr, String> {
        let a = "thinker.audio_tower";
        let conv_out = Linear::from_rows(
            &band_major(&t.get(&format!("{a}.conv_out.weight"), &[ENC_DIM, CONV * BANDS])?),
            ENC_DIM,
            CONV * BANDS,
            None,
        );
        let mut encoder = Vec::new();
        for l in 0..ENC_LAYERS {
            let n = format!("{a}.layers.{l}");
            let linear = |name: &str, inputs, outputs| {
                Linear::load(t, &format!("{n}.{name}"), inputs, outputs)
            };
            encoder.push(EncoderLayer {
                norm: LayerNorm::load(t, &format!("{n}.self_attn_layer_norm"), ENC_DIM)?,
                q: linear("self_attn.q_proj", ENC_DIM, ENC_DIM)?,
                k: linear("self_attn.k_proj", ENC_DIM, ENC_DIM)?,
                v: linear("self_attn.v_proj", ENC_DIM, ENC_DIM)?,
                o: linear("self_attn.out_proj", ENC_DIM, ENC_DIM)?,
                final_norm: LayerNorm::load(t, &format!("{n}.final_layer_norm"), ENC_DIM)?,
                up: linear("fc1", ENC_DIM, ENC_FF)?,
                down: linear("fc2", ENC_FF, ENC_DIM)?,
            });
        }
        let m = "thinker.model";
        let mut decoder = Vec::new();
        for l in 0..LAYERS {
            let n = format!("{m}.layers.{l}");
            let linear = |name: &str, inputs, outputs| {
                Linear::load_unbiased(t, &format!("{n}.{name}"), inputs, outputs)
            };
            let weight = |name: &str, dim| t.get(&format!("{n}.{name}.weight"), &[dim]);
            decoder.push(DecoderLayer {
                norm: weight("input_layernorm", DIM)?,
                q: linear("self_attn.q_proj", DIM, HEADS * HEAD)?,
                k: linear("self_attn.k_proj", DIM, KV_HEADS * HEAD)?,
                v: linear("self_attn.v_proj", DIM, KV_HEADS * HEAD)?,
                o: linear("self_attn.o_proj", HEADS * HEAD, DIM)?,
                q_norm: weight("self_attn.q_norm", HEAD)?,
                k_norm: weight("self_attn.k_norm", HEAD)?,
                post_norm: weight("post_attention_layernorm", DIM)?,
                gate: linear("mlp.gate_proj", DIM, FF)?,
                up: linear("mlp.up_proj", DIM, FF)?,
                down: linear("mlp.down_proj", FF, DIM)?,
            });
        }
        Ok(Qwen3Asr {
            conv1: Conv::load(t, &format!("{a}.conv2d1"), 1, CONV)?,
            conv2: Conv::load(t, &format!("{a}.conv2d2"), CONV, CONV)?,
            conv3: Conv::load(t, &format!("{a}.conv2d3"), CONV, CONV)?,
            conv_out,
            positions: sinusoids(MAX_POSITIONS, ENC_DIM),
            encoder,
            post_norm: LayerNorm::load(t, &format!("{a}.ln_post"), ENC_DIM)?,
            proj1: Linear::load(t, &format!("{a}.proj1"), ENC_DIM, ENC_DIM)?,
            proj2: Linear::load(t, &format!("{a}.proj2"), ENC_DIM, DIM)?,
            embed: t.get(&format!("{m}.embed_tokens.weight"), &[VOCAB, DIM])?,
            decoder,
            norm: t.get(&format!("{m}.norm.weight"), &[DIM])?,
            head: t.get("thinker.lm_head.weight", &[VOCAB, DIM])?,
            vocab,
            threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
        })
    }

    /// Runs on at most `threads` threads (all the processor has, at first).
    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads.max(1);
    }

    /// The vocabulary.
    pub fn vocab(&self) -> &text::Vocab {
        &self.vocab
    }

    /// The log-mel features of a voice at 16 kHz: `[128][frames]`, frames.
    pub fn features(&self, voice: &[f32]) -> (Vec<f32>, usize) {
        mel::features(voice, self.threads)
    }

    /// The steps entering the encoder's first layer (`[steps][896]`):
    /// the convolutions over each piece, and the positions within it.
    pub fn embed_audio(&self, features: &[f32], frames: usize) -> Vec<f32> {
        let pieces = frames.div_ceil(PIECE);
        // Every piece padded to the longest (a whole piece, unless the
        // audio is shorter).
        let width = frames.min(PIECE);
        let per = piece_steps(width);
        let mut out = Vec::with_capacity(audio_steps(frames) * ENC_DIM);
        // A few pieces at a time keeps the patches small.
        for first in (0..pieces).step_by(8) {
            let batch = (pieces - first).min(8);
            // [piece][band][step][1].
            let mut x = vec![0.0f32; batch * mel::MELS * width];
            for p in 0..batch {
                let start = (first + p) * PIECE;
                let len = (frames - start).min(width);
                for b in 0..mel::MELS {
                    for s in 0..len {
                        x[(p * mel::MELS + b) * width + s] = features[b * frames + start + s];
                    }
                }
            }
            let (h, b1, s1) = self
                .conv1
                .forward(&x, (batch, mel::MELS, width), self.threads);
            let (h, b2, s2) = self.conv2.forward(&h, (batch, b1, s1), self.threads);
            let (h, b3, s3) = self.conv3.forward(&h, (batch, b2, s2), self.threads);
            debug_assert_eq!(s3, per);
            // Each step's bands side by side: [piece][step][band·480].
            let mut flat = vec![0.0f32; batch * s3 * b3 * CONV];
            for p in 0..batch {
                for s in 0..s3 {
                    for b in 0..b3 {
                        let from = ((p * b3 + b) * s3 + s) * CONV;
                        let to = ((p * s3 + s) * b3 + b) * CONV;
                        flat[to..to + CONV].copy_from_slice(&h[from..from + CONV]);
                    }
                }
            }
            let mut e = self.conv_out.forward(&flat, self.threads);
            for p in 0..batch {
                let start = (first + p) * PIECE;
                let keep = piece_steps((frames - start).min(PIECE));
                for s in 0..keep {
                    let row = &mut e[(p * s3 + s) * ENC_DIM..][..ENC_DIM];
                    for (v, pos) in row.iter_mut().zip(&self.positions[s * ENC_DIM..]) {
                        *v += pos;
                    }
                    out.extend_from_slice(row);
                }
            }
        }
        out
    }

    /// One encoder layer, in place; attention within windows of `window`
    /// steps.
    fn encoder_layer(&self, l: &EncoderLayer, h: &mut [f32], window: usize) {
        let steps = h.len() / ENC_DIM;
        let mut x = h.to_vec();
        l.norm.apply(&mut x);
        let q = l.q.forward(&x, self.threads);
        let k = l.k.forward(&x, self.threads);
        let v = l.v.forward(&x, self.threads);
        let heads = Heads {
            query: ENC_HEADS,
            kv: ENC_HEADS,
            width: ENC_HEAD,
        };
        let within = |t: usize| {
            let from = t / window * window;
            from..(from + window).min(steps)
        };
        let attended = attend(&q, &k, &v, heads, within, self.threads);
        let o = l.o.forward(&attended, self.threads);
        for (a, b) in h.iter_mut().zip(&o) {
            *a += b;
        }
        let mut x = h.to_vec();
        l.final_norm.apply(&mut x);
        let mut up = l.up.forward(&x, self.threads);
        gelu(&mut up);
        let down = l.down.forward(&up, self.threads);
        for (a, b) in h.iter_mut().zip(&down) {
            *a += b;
        }
    }

    /// The attention window, in steps, for `frames` frames.
    fn window(frames: usize) -> usize {
        piece_steps(frames.min(PIECE)) * WINDOW_PIECES
    }

    /// The first `layers` encoder layers over `h` (from [`Self::embed_audio`]).
    pub fn encoder_layers(&self, mut h: Vec<f32>, frames: usize, layers: usize) -> Vec<f32> {
        let window = Self::window(frames);
        for l in self.encoder.iter().take(layers) {
            self.encoder_layer(l, &mut h, window);
        }
        h
    }

    /// The audio's steps for the decoder (`[steps][1024]`) from the
    /// features.
    pub fn encode(&self, features: &[f32], frames: usize) -> Vec<f32> {
        let h = self.embed_audio(features, frames);
        let mut h = self.encoder_layers(h, frames, ENC_LAYERS);
        self.post_norm.apply(&mut h);
        let mut h = self.proj1.forward(&h, self.threads);
        gelu(&mut h);
        self.proj2.forward(&h, self.threads)
    }

    /// The prompt's embeddings with the audio's steps in place of its
    /// audio tokens.
    pub fn embed_prompt(&self, ids: &[u32], audio: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(ids.len() * DIM);
        let mut next = 0;
        for &id in ids {
            if id == text::AUDIO_PAD && (next + 1) * DIM <= audio.len() {
                out.extend_from_slice(&audio[next * DIM..(next + 1) * DIM]);
                next += 1;
            } else {
                out.extend_from_slice(&self.embed[id as usize * DIM..][..DIM]);
            }
        }
        out
    }

    /// Rotary positions on each `HEAD`-wide head of `x` (rows from
    /// position `start`): each half's pair turned by `position ·
    /// base^(-2i/128)`.
    fn rotate(x: &mut [f32], heads: usize, start: usize) {
        let half = HEAD / 2;
        let rows = x.len() / (heads * HEAD);
        for r in 0..rows {
            let p = (start + r) as f64;
            for i in 0..half {
                let a = p / ROPE_BASE.powf(2.0 * i as f64 / HEAD as f64);
                let (sin, cos) = (a.sin() as f32, a.cos() as f32);
                for h in 0..heads {
                    let v = &mut x[(r * heads + h) * HEAD..][..HEAD];
                    let (a0, a1) = (v[i], v[i + half]);
                    v[i] = a0 * cos - a1 * sin;
                    v[i + half] = a1 * cos + a0 * sin;
                }
            }
        }
    }

    /// One decoder layer over `h` (rows from the cache's length on), its
    /// keys and values added to the cache.
    fn decoder_layer(&self, li: usize, h: &mut [f32], cache: &mut Cache) {
        let l = &self.decoder[li];
        let start = cache.len;
        let mut x = h.to_vec();
        rms_norm(&mut x, &l.norm);
        let mut q = l.q.forward(&x, self.threads);
        let mut k = l.k.forward(&x, self.threads);
        let v = l.v.forward(&x, self.threads);
        rms_norm(&mut q, &l.q_norm);
        rms_norm(&mut k, &l.k_norm);
        Self::rotate(&mut q, HEADS, start);
        Self::rotate(&mut k, KV_HEADS, start);
        cache.keys[li].extend_from_slice(&k);
        cache.values[li].extend_from_slice(&v);
        let (keys, values) = (&cache.keys[li], &cache.values[li]);
        // Each row sees itself and every one before it.
        let causal = |r: usize| 0..start + r + 1;
        let attended = attend(
            &q,
            keys,
            values,
            Heads {
                query: HEADS,
                kv: KV_HEADS,
                width: HEAD,
            },
            causal,
            self.threads,
        );
        let o = l.o.forward(&attended, self.threads);
        for (a, b) in h.iter_mut().zip(&o) {
            *a += b;
        }
        let mut x = h.to_vec();
        rms_norm(&mut x, &l.post_norm);
        let mut gate = l.gate.forward(&x, self.threads);
        let up = l.up.forward(&x, self.threads);
        for (g, u) in gate.iter_mut().zip(&up) {
            *g = *g / (1.0 + (-*g).exp()) * u;
        }
        let down = l.down.forward(&gate, self.threads);
        for (a, b) in h.iter_mut().zip(&down) {
            *a += b;
        }
    }

    /// `x` (embeddings, rows from the cache's length on) through the
    /// decoder; the logits of its last row.
    pub fn decode(&self, mut x: Vec<f32>, cache: &mut Cache) -> Vec<f32> {
        let rows = x.len() / DIM;
        for li in 0..LAYERS {
            self.decoder_layer(li, &mut x, cache);
        }
        cache.len += rows;
        let mut last = x[(rows - 1) * DIM..].to_vec();
        rms_norm(&mut last, &self.norm);
        let mut logits = vec![0.0f32; VOCAB];
        par_rows(&mut logits, 1, self.threads, |t, v| {
            v[0] = dot(&last, &self.head[t * DIM..][..DIM]);
        });
        logits
    }

    /// A fresh cache.
    pub fn cache(&self) -> Cache {
        Cache::new()
    }

    /// The logits of the first token the model writes for `voice`.
    pub fn first_logits(&self, voice: &[f32]) -> Vec<f32> {
        let (features, frames) = self.features(voice);
        let audio = self.encode(&features, frames);
        let ids = text::prompt(audio.len() / DIM);
        self.decode(self.embed_prompt(&ids, &audio), &mut Cache::new())
    }

    /// The tokens written for one part of a voice, greedily, at most `max`,
    /// a loop cut short ([`loop_start`]).
    pub fn tokens(&self, voice: &[f32], max: usize) -> Vec<u32> {
        let (features, frames) = self.features(voice);
        if frames == 0 {
            return Vec::new();
        }
        let audio = self.encode(&features, frames);
        let prompt = text::prompt(audio.len() / DIM);
        let mut cache = Cache::new();
        let mut logits = self.decode(self.embed_prompt(&prompt, &audio), &mut cache);
        let mut ids = Vec::new();
        while ids.len() < max {
            let id = argmax(&logits);
            if text::is_end(id) {
                break;
            }
            ids.push(id);
            if let Some(keep) = loop_start(&ids) {
                ids.truncate(keep);
                break;
            }
            let x = self.embed[id as usize * DIM..][..DIM].to_vec();
            logits = self.decode(x, &mut cache);
        }
        ids
    }

    /// What is sung in `voice` (16 kHz mono): its parts ([`parts_of`])
    /// each heard on its own, greedily, at most `max` tokens each.
    pub fn transcribe(&self, voice: &[f32], max: usize) -> Transcript {
        let hear = |part: &[f32]| Ok::<_, Infallible>(self.tokens(part, max));
        let Ok(heard) = transcribe_parts(voice, &self.vocab, hear);
        heard
    }
}

/// The index of the largest value.
pub fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_second_is_thirteen_steps() {
        assert_eq!(piece_steps(100), 13);
        assert_eq!(audio_steps(2000), 260);
        assert_eq!(audio_steps(2050), 260 + 7);
        assert_eq!(audio_steps(1), 1);
    }

    #[test]
    fn the_positions_start_with_sines_of_zero() {
        let p = sinusoids(3, 8);
        assert_eq!(&p[..8], &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        assert!((p[8] - 1f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn rotation_keeps_lengths_and_does_nothing_at_zero() {
        let mut x: Vec<f32> = (0..2 * HEAD).map(|i| (i as f32 * 0.37).sin()).collect();
        let before = x.clone();
        Qwen3Asr::rotate(&mut x[..HEAD], 1, 0);
        assert_eq!(&x[..HEAD], &before[..HEAD]);
        Qwen3Asr::rotate(&mut x[HEAD..], 1, 7);
        let len = |v: &[f32]| v.iter().map(|a| a * a).sum::<f32>();
        assert!((len(&x[HEAD..]) - len(&before[HEAD..])).abs() < 1e-4);
    }
}
