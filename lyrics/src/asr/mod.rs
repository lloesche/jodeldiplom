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
pub mod quant;
pub mod text;

use std::path::Path;

use crate::speech::{dot, gelu, matmul, par_rows, SafeTensors};

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
const LN_EPS: f32 = 1e-5;
const RMS_EPS: f32 = 1e-6;

/// Steps `n` values of a convolution with stride 2 and padding 1 give.
fn halved(n: usize) -> usize {
    n.div_ceil(2)
}

/// The encoder's steps for a piece of `frames` frames.
pub fn piece_steps(frames: usize) -> usize {
    halved(halved(halved(frames)))
}

/// The encoder's steps for `frames` frames of features.
pub fn audio_steps(frames: usize) -> usize {
    (frames / PIECE) * piece_steps(PIECE) + piece_steps(frames % PIECE)
}

/// A linear layer, its weight transposed to `[input][output]`.
struct Linear {
    inputs: usize,
    outputs: usize,
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl Linear {
    fn from_rows(raw: &[f32], outputs: usize, inputs: usize, bias: Option<Vec<f32>>) -> Linear {
        let mut weight = vec![0.0f32; raw.len()];
        for o in 0..outputs {
            for i in 0..inputs {
                weight[i * outputs + o] = raw[o * inputs + i];
            }
        }
        Linear {
            inputs,
            outputs,
            weight,
            bias,
        }
    }

    fn load(
        t: &SafeTensors,
        name: &str,
        inputs: usize,
        outputs: usize,
        bias: bool,
    ) -> Result<Linear, String> {
        let raw = t.get(&format!("{name}.weight"), &[outputs, inputs])?;
        let bias = if bias {
            Some(t.get(&format!("{name}.bias"), &[outputs])?)
        } else {
            None
        };
        Ok(Linear::from_rows(&raw, outputs, inputs, bias))
    }

    fn forward(&self, x: &[f32], threads: usize) -> Vec<f32> {
        let rows = x.len() / self.inputs;
        let mut out = vec![0.0f32; rows * self.outputs];
        matmul(
            &self.weight,
            self.inputs,
            self.outputs,
            x,
            &mut out,
            threads,
        );
        if let Some(b) = &self.bias {
            for row in out.chunks_exact_mut(self.outputs) {
                for (o, bias) in row.iter_mut().zip(b) {
                    *o += bias;
                }
            }
        }
        out
    }
}

/// A LayerNorm.
struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl LayerNorm {
    fn load(t: &SafeTensors, name: &str, dim: usize) -> Result<LayerNorm, String> {
        Ok(LayerNorm {
            weight: t.get(&format!("{name}.weight"), &[dim])?,
            bias: t.get(&format!("{name}.bias"), &[dim])?,
        })
    }

    fn apply(&self, x: &mut [f32]) {
        let dim = self.weight.len();
        for row in x.chunks_exact_mut(dim) {
            let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / dim as f64;
            let var = row
                .iter()
                .map(|&v| (f64::from(v) - mean).powi(2))
                .sum::<f64>()
                / dim as f64;
            let k = 1.0 / (var + f64::from(LN_EPS)).sqrt();
            for (d, v) in row.iter_mut().enumerate() {
                *v = ((f64::from(*v) - mean) * k) as f32 * self.weight[d] + self.bias[d];
            }
        }
    }
}

/// Each `weight.len()`-wide row of `x` RMS-normed and scaled, in place.
fn rms_norm(x: &mut [f32], weight: &[f32]) {
    let dim = weight.len();
    for row in x.chunks_exact_mut(dim) {
        let mean = row
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            / dim as f64;
        let k = (1.0 / (mean + f64::from(RMS_EPS)).sqrt()) as f32;
        for (v, w) in row.iter_mut().zip(weight) {
            *v = *v * k * w;
        }
    }
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

/// A convolution's weight as `[tap·input][output]` (tap-major, the taps
/// row by row over 3 bands × 3 steps), and its bias.
struct Conv {
    inputs: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Conv {
    fn load(t: &SafeTensors, name: &str, inputs: usize) -> Result<Conv, String> {
        let raw = t.get(&format!("{name}.weight"), &[CONV, inputs, 3, 3])?;
        let mut weight = vec![0.0f32; raw.len()];
        for o in 0..CONV {
            for i in 0..inputs {
                for tap in 0..9 {
                    weight[(tap * inputs + i) * CONV + o] = raw[(o * inputs + i) * 9 + tap];
                }
            }
        }
        Ok(Conv {
            inputs,
            weight,
            bias: t.get(&format!("{name}.bias"), &[CONV])?,
        })
    }

    /// `x` (`[images][bands][steps][inputs]`) through the convolution
    /// (stride 2, padding 1) and GELU: `[images][bands/2][steps/2][480]`
    /// (rounded up), and its bands and steps.
    fn forward(
        &self,
        x: &[f32],
        (images, bands, steps): (usize, usize, usize),
        threads: usize,
    ) -> (Vec<f32>, usize, usize) {
        let (ob, os) = (halved(bands), halved(steps));
        let width = 9 * self.inputs;
        let rows = images * ob * os;
        let mut patches = vec![0.0f32; rows * width];
        par_rows(&mut patches, width, threads, |r, row| {
            let (img, rest) = (r / (ob * os), r % (ob * os));
            let (b, s) = (rest / os, rest % os);
            for kb in 0..3 {
                for ks in 0..3 {
                    let (ib, is) = ((2 * b + kb) as i64 - 1, (2 * s + ks) as i64 - 1);
                    if ib < 0 || is < 0 || ib >= bands as i64 || is >= steps as i64 {
                        continue;
                    }
                    let from = ((img * bands + ib as usize) * steps + is as usize) * self.inputs;
                    let to = (kb * 3 + ks) * self.inputs;
                    row[to..to + self.inputs].copy_from_slice(&x[from..from + self.inputs]);
                }
            }
        });
        let mut out = vec![0.0f32; rows * CONV];
        matmul(&self.weight, width, CONV, &patches, &mut out, threads);
        for row in out.chunks_exact_mut(CONV) {
            for (v, b) in row.iter_mut().zip(&self.bias) {
                *v += b;
            }
        }
        gelu(&mut out);
        (out, ob, os)
    }
}

/// What [`Qwen3Asr::visit`] hands each tensor to: its name, its values, and
/// its shape when it is a matrix.
pub type Visitor<'a> = dyn FnMut(&str, &[f32], Option<(usize, usize)>) + 'a;

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

/// What the model heard.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Transcript {
    /// The language it named ("English"), empty when none.
    pub language: String,
    /// The words.
    pub text: String,
    /// The tokens it wrote.
    pub ids: Vec<u32>,
    /// The parts it was heard in, and their words.
    pub parts: Vec<Part>,
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
        let read = |name: &str| {
            std::fs::read_to_string(dir.join(name)).map_err(|e| format!("{name}: {e}"))
        };
        let vocab = text::Vocab::parse(&read("vocab.json")?, &read("tokenizer_config.json")?)?;
        Qwen3Asr::load(&t, vocab)
    }

    /// The model from its tensors and vocabulary.
    pub fn load(t: &SafeTensors, vocab: text::Vocab) -> Result<Qwen3Asr, String> {
        let a = "thinker.audio_tower";
        let bands = piece_steps(mel::MELS);
        let raw = t.get(&format!("{a}.conv_out.weight"), &[ENC_DIM, CONV * bands])?;
        // From [896][channel·16 + band] to [band·480 + channel][896].
        let mut conv_out = vec![0.0f32; raw.len()];
        for o in 0..ENC_DIM {
            for c in 0..CONV {
                for b in 0..bands {
                    conv_out[(b * CONV + c) * ENC_DIM + o] = raw[o * CONV * bands + c * bands + b];
                }
            }
        }
        let mut encoder = Vec::new();
        for l in 0..ENC_LAYERS {
            let n = format!("{a}.layers.{l}");
            encoder.push(EncoderLayer {
                norm: LayerNorm::load(t, &format!("{n}.self_attn_layer_norm"), ENC_DIM)?,
                q: Linear::load(t, &format!("{n}.self_attn.q_proj"), ENC_DIM, ENC_DIM, true)?,
                k: Linear::load(t, &format!("{n}.self_attn.k_proj"), ENC_DIM, ENC_DIM, true)?,
                v: Linear::load(t, &format!("{n}.self_attn.v_proj"), ENC_DIM, ENC_DIM, true)?,
                o: Linear::load(
                    t,
                    &format!("{n}.self_attn.out_proj"),
                    ENC_DIM,
                    ENC_DIM,
                    true,
                )?,
                final_norm: LayerNorm::load(t, &format!("{n}.final_layer_norm"), ENC_DIM)?,
                up: Linear::load(t, &format!("{n}.fc1"), ENC_DIM, ENC_FF, true)?,
                down: Linear::load(t, &format!("{n}.fc2"), ENC_FF, ENC_DIM, true)?,
            });
        }
        let m = "thinker.model";
        let mut decoder = Vec::new();
        for l in 0..LAYERS {
            let n = format!("{m}.layers.{l}");
            decoder.push(DecoderLayer {
                norm: t.get(&format!("{n}.input_layernorm.weight"), &[DIM])?,
                q: Linear::load(
                    t,
                    &format!("{n}.self_attn.q_proj"),
                    DIM,
                    HEADS * HEAD,
                    false,
                )?,
                k: Linear::load(
                    t,
                    &format!("{n}.self_attn.k_proj"),
                    DIM,
                    KV_HEADS * HEAD,
                    false,
                )?,
                v: Linear::load(
                    t,
                    &format!("{n}.self_attn.v_proj"),
                    DIM,
                    KV_HEADS * HEAD,
                    false,
                )?,
                o: Linear::load(
                    t,
                    &format!("{n}.self_attn.o_proj"),
                    HEADS * HEAD,
                    DIM,
                    false,
                )?,
                q_norm: t.get(&format!("{n}.self_attn.q_norm.weight"), &[HEAD])?,
                k_norm: t.get(&format!("{n}.self_attn.k_norm.weight"), &[HEAD])?,
                post_norm: t.get(&format!("{n}.post_attention_layernorm.weight"), &[DIM])?,
                gate: Linear::load(t, &format!("{n}.mlp.gate_proj"), DIM, FF, false)?,
                up: Linear::load(t, &format!("{n}.mlp.up_proj"), DIM, FF, false)?,
                down: Linear::load(t, &format!("{n}.mlp.down_proj"), FF, DIM, false)?,
            });
        }
        Ok(Qwen3Asr {
            conv1: Conv::load(t, &format!("{a}.conv2d1"), 1)?,
            conv2: Conv::load(t, &format!("{a}.conv2d2"), CONV)?,
            conv3: Conv::load(t, &format!("{a}.conv2d3"), CONV)?,
            conv_out: Linear {
                inputs: CONV * bands,
                outputs: ENC_DIM,
                weight: conv_out,
                bias: None,
            },
            positions: sinusoids(MAX_POSITIONS, ENC_DIM),
            encoder,
            post_norm: LayerNorm::load(t, &format!("{a}.ln_post"), ENC_DIM)?,
            proj1: Linear::load(t, &format!("{a}.proj1"), ENC_DIM, ENC_DIM, true)?,
            proj2: Linear::load(t, &format!("{a}.proj2"), ENC_DIM, DIM, true)?,
            embed: t.get(&format!("{m}.embed_tokens.weight"), &[VOCAB, DIM])?,
            decoder,
            norm: t.get(&format!("{m}.norm.weight"), &[DIM])?,
            head: t.get("thinker.lm_head.weight", &[VOCAB, DIM])?,
            vocab,
            threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
        })
    }

    /// Hands each tensor to `f` by name, with its shape when it is a matrix
    /// (`(rows, columns)`: a row an output, its inputs along it), for a
    /// GPU's version:
    ///
    /// - `conv1.w` `[480][9]` (3 bands × 3 steps), `conv1.b`;
    /// - `conv2.w`, `conv3.w` `[480][9·480]` (tap-major inputs), their `.b`;
    /// - `conv_out.w` `[896][16·480]` (band-major inputs);
    /// - per encoder layer `l`: `enc{l}.ln.w`, `.b`; `enc{l}.qkv.w`
    ///   `[3·896][896]` (queries, keys, values), `enc{l}.qkv.b`;
    ///   `enc{l}.o.w`, `.b`; `enc{l}.fln.w`, `.b`; `enc{l}.up.w` `[3584][896]`,
    ///   `.b`; `enc{l}.down.w`, `.b`;
    /// - `post.w`, `post.b`; `proj1.w`, `.b`; `proj2.w` `[1024][896]`, `.b`;
    /// - `embed` and `head` `[151936][1024]`;
    /// - per decoder layer `l`: `dec{l}.norm`; `dec{l}.qkv.w` `[4096][1024]`
    ///   (2048 queries, 1024 keys, 1024 values); `dec{l}.o.w` `[1024][2048]`;
    ///   `dec{l}.q_norm`, `dec{l}.k_norm` (128); `dec{l}.post_norm`;
    ///   `dec{l}.gate_up.w` `[2·3072][1024]` (gates, then ups); `dec{l}.down.w`;
    /// - `norm`.
    pub fn visit(&self, f: &mut Visitor) {
        /// Stacked linear layers' weights as rows, and their biases.
        fn rows(parts: &[&Linear]) -> (Vec<f32>, usize, usize, Vec<f32>) {
            let inputs = parts[0].inputs;
            let outputs: usize = parts.iter().map(|l| l.outputs).sum();
            let mut w = Vec::with_capacity(inputs * outputs);
            let mut b = Vec::new();
            for l in parts {
                for o in 0..l.outputs {
                    w.extend((0..inputs).map(|i| l.weight[i * l.outputs + o]));
                }
                b.extend(l.bias.clone().unwrap_or_default());
            }
            (w, outputs, inputs, b)
        }
        let linear = |f: &mut Visitor, name: &str, parts: &[&Linear]| {
            let (w, o, i, b) = rows(parts);
            f(&format!("{name}.w"), &w, Some((o, i)));
            if !b.is_empty() {
                f(&format!("{name}.b"), &b, None);
            }
        };
        let conv = |f: &mut Visitor, name: &str, c: &Conv| {
            let width = 9 * c.inputs;
            let mut w = vec![0.0f32; width * CONV];
            for i in 0..width {
                for o in 0..CONV {
                    w[o * width + i] = c.weight[i * CONV + o];
                }
            }
            f(&format!("{name}.w"), &w, Some((CONV, width)));
            f(&format!("{name}.b"), &c.bias, None);
        };
        conv(f, "conv1", &self.conv1);
        conv(f, "conv2", &self.conv2);
        conv(f, "conv3", &self.conv3);
        linear(f, "conv_out", &[&self.conv_out]);
        let norm = |f: &mut Visitor, name: &str, n: &LayerNorm| {
            f(&format!("{name}.w"), &n.weight, None);
            f(&format!("{name}.b"), &n.bias, None);
        };
        for (i, l) in self.encoder.iter().enumerate() {
            norm(f, &format!("enc{i}.ln"), &l.norm);
            linear(f, &format!("enc{i}.qkv"), &[&l.q, &l.k, &l.v]);
            linear(f, &format!("enc{i}.o"), &[&l.o]);
            norm(f, &format!("enc{i}.fln"), &l.final_norm);
            linear(f, &format!("enc{i}.up"), &[&l.up]);
            linear(f, &format!("enc{i}.down"), &[&l.down]);
        }
        norm(f, "post", &self.post_norm);
        linear(f, "proj1", &[&self.proj1]);
        linear(f, "proj2", &[&self.proj2]);
        f("embed", &self.embed, Some((VOCAB, DIM)));
        f("head", &self.head, Some((VOCAB, DIM)));
        for (i, l) in self.decoder.iter().enumerate() {
            f(&format!("dec{i}.norm"), &l.norm, None);
            linear(f, &format!("dec{i}.qkv"), &[&l.q, &l.k, &l.v]);
            linear(f, &format!("dec{i}.o"), &[&l.o]);
            f(&format!("dec{i}.q_norm"), &l.q_norm, None);
            f(&format!("dec{i}.k_norm"), &l.k_norm, None);
            f(&format!("dec{i}.post_norm"), &l.post_norm, None);
            linear(f, &format!("dec{i}.gate_up"), &[&l.gate, &l.up]);
            linear(f, &format!("dec{i}.down"), &[&l.down]);
        }
        f("norm", &self.norm, None);
    }

    /// Rounds the large weights (every linear layer, the convolutions after
    /// the first, the embeddings and the head) as a quantised copy would
    /// hold them: groups of `group` along each output's inputs, `bits` a
    /// weight, a scale a group, and a minimum too unless `symmetric`. For
    /// measuring what quantising costs.
    pub fn round_weights(&mut self, bits: u32, group: usize, symmetric: bool) {
        let round = |w: &mut Vec<f32>, inputs: usize, outputs: usize, transposed: bool| {
            let mut rows = if transposed {
                let mut r = vec![0.0f32; w.len()];
                for i in 0..inputs {
                    for o in 0..outputs {
                        r[o * inputs + i] = w[i * outputs + o];
                    }
                }
                r
            } else {
                std::mem::take(w)
            };
            round_groups(&mut rows, bits, group, symmetric);
            if transposed {
                for i in 0..inputs {
                    for o in 0..outputs {
                        w[i * outputs + o] = rows[o * inputs + i];
                    }
                }
            } else {
                *w = rows;
            }
        };
        let mut linears: Vec<&mut Linear> =
            vec![&mut self.conv_out, &mut self.proj1, &mut self.proj2];
        for l in &mut self.encoder {
            linears.extend([
                &mut l.q,
                &mut l.k,
                &mut l.v,
                &mut l.o,
                &mut l.up,
                &mut l.down,
            ]);
        }
        for l in &mut self.decoder {
            linears.extend([
                &mut l.q,
                &mut l.k,
                &mut l.v,
                &mut l.o,
                &mut l.gate,
                &mut l.up,
                &mut l.down,
            ]);
        }
        for l in linears {
            let (i, o) = (l.inputs, l.outputs);
            round(&mut l.weight, i, o, true);
        }
        for c in [&mut self.conv2, &mut self.conv3] {
            let i = 9 * c.inputs;
            round(&mut c.weight, i, CONV, true);
        }
        round(&mut self.embed, DIM, VOCAB, false);
        round(&mut self.head, DIM, VOCAB, false);
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
        let mut attended = vec![0.0f32; steps * ENC_DIM];
        let scale = 1.0 / (ENC_HEAD as f32).sqrt();
        par_rows(&mut attended, ENC_DIM, self.threads, |t, row| {
            let from = t / window * window;
            let to = (from + window).min(steps);
            let mut scores = vec![0.0f32; to - from];
            for head in 0..ENC_HEADS {
                let qh = &q[t * ENC_DIM + head * ENC_HEAD..][..ENC_HEAD];
                let mut best = f32::MIN;
                for (u, s) in scores.iter_mut().enumerate() {
                    let kh = &k[(from + u) * ENC_DIM + head * ENC_HEAD..][..ENC_HEAD];
                    *s = dot(qh, kh) * scale;
                    best = best.max(*s);
                }
                let mut total = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - best).exp();
                    total += *s;
                }
                let out = &mut row[head * ENC_HEAD..(head + 1) * ENC_HEAD];
                for (u, &s) in scores.iter().enumerate() {
                    let vh = &v[(from + u) * ENC_DIM + head * ENC_HEAD..][..ENC_HEAD];
                    let w = s / total;
                    for (o, x) in out.iter_mut().zip(vh) {
                        *o += w * x;
                    }
                }
            }
        });
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
        let rows = h.len() / DIM;
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
        let kv = KV_HEADS * HEAD;
        let scale = 1.0 / (HEAD as f32).sqrt();
        let mut attended = vec![0.0f32; rows * HEADS * HEAD];
        par_rows(&mut attended, HEADS * HEAD, self.threads, |r, row| {
            let seen = start + r + 1;
            let mut scores = vec![0.0f32; seen];
            for head in 0..HEADS {
                let g = head / (HEADS / KV_HEADS);
                let qh = &q[(r * HEADS + head) * HEAD..][..HEAD];
                let mut best = f32::MIN;
                for (u, s) in scores.iter_mut().enumerate() {
                    *s = dot(qh, &keys[u * kv + g * HEAD..][..HEAD]) * scale;
                    best = best.max(*s);
                }
                let mut total = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - best).exp();
                    total += *s;
                }
                let out = &mut row[head * HEAD..(head + 1) * HEAD];
                for (u, &s) in scores.iter().enumerate() {
                    let w = s / total;
                    for (o, x) in out.iter_mut().zip(&values[u * kv + g * HEAD..][..HEAD]) {
                        *o += w * x;
                    }
                }
            }
        });
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
        let per = VOCAB.div_ceil(self.threads.max(1));
        std::thread::scope(|s| {
            for (c, chunk) in logits.chunks_mut(per).enumerate() {
                let (last, head) = (&last, &self.head);
                s.spawn(move || {
                    for (i, v) in chunk.iter_mut().enumerate() {
                        let t = c * per + i;
                        *v = dot(last, &head[t * DIM..][..DIM]);
                    }
                });
            }
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
        transcribe_parts(voice, &self.vocab, |part| Ok(self.tokens(part, max))).unwrap_or_default()
    }
}

/// The longest part of a voice heard on its own, in seconds: the decoder
/// keeps to what it hears in a part this long, where over minutes it may
/// loop or stop early.
pub const PART_SECONDS: f64 = 30.0;
/// A part ends at the quietest moment of the voice after this many
/// seconds of it.
pub const PART_LEAST: f64 = 20.0;

/// A stretch of a voice heard on its own, and what was heard in it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Part {
    /// Where it starts and ends, in seconds.
    pub start: f64,
    /// Where it ends.
    pub end: f64,
    /// The words heard in it.
    pub text: String,
}

/// Where `voice` (16 kHz) is cut into parts: sample offsets, from 0 to its
/// length, at most [`PART_SECONDS`] apart, each cut at the quietest 0.2 s
/// from [`PART_LEAST`] on (so a word is rarely cut).
pub fn parts_of(voice: &[f32]) -> Vec<usize> {
    let rate = f64::from(crate::VOCALS_RATE);
    let most = (PART_SECONDS * rate) as usize;
    let least = (PART_LEAST * rate) as usize;
    let window = (0.2 * rate) as usize;
    let step = window / 4;
    let mut cuts = vec![0];
    let mut at = 0;
    while voice.len() - at > most {
        let mut best = (f64::MAX, at + most);
        let mut w = at + least;
        while w + window <= at + most {
            let energy: f64 = voice[w..w + window]
                .iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum();
            if energy < best.0 {
                best = (energy, w + window / 2);
            }
            w += step;
        }
        at = best.1;
        cuts.push(at);
    }
    cuts.push(voice.len());
    cuts
}

/// Where the tokens have fallen into a loop, if they have: the last 32
/// all repeat a cycle of at most 4. Then the tokens to keep (up to two of
/// the cycles).
pub fn loop_start(ids: &[u32]) -> Option<usize> {
    const SPAN: usize = 32;
    if ids.len() < SPAN {
        return None;
    }
    let tail = &ids[ids.len() - SPAN..];
    (1..=4)
        .find(|&p| (p..SPAN).all(|i| tail[i] == tail[i - p]))
        .map(|p| ids.len() - SPAN + 2 * p)
}

/// The transcript of `voice` heard in parts ([`parts_of`]), `hear` giving
/// each part's tokens: the parts' words joined, the first language named.
pub fn transcribe_parts(
    voice: &[f32],
    vocab: &text::Vocab,
    mut hear: impl FnMut(&[f32]) -> Result<Vec<u32>, String>,
) -> Result<Transcript, String> {
    let rate = f64::from(crate::VOCALS_RATE);
    let cuts = parts_of(voice);
    let mut out = Transcript::default();
    for pair in cuts.windows(2) {
        let ids = hear(&voice[pair[0]..pair[1]])?;
        let (language, words) = text::split_language(&vocab.decode(&ids));
        if out.language.is_empty() || out.language == "None" {
            out.language = language;
        }
        if !words.is_empty() {
            if !out.text.is_empty() {
                out.text.push(' ');
            }
            out.text.push_str(&words);
        }
        out.parts.push(Part {
            start: pair[0] as f64 / rate,
            end: pair[1] as f64 / rate,
            text: words,
        });
        out.ids.extend(ids);
    }
    Ok(out)
}

/// Each `group` of `values` rounded to `bits` a value: a scale a group
/// (from the largest magnitude when `symmetric`, else from the range, with
/// its minimum).
fn round_groups(values: &mut [f32], bits: u32, group: usize, symmetric: bool) {
    if bits == 4 && group == quant::GROUP && values.len() % group == 0 {
        // Exactly as the 4-bit format holds them (scales as halves).
        let rows = values.len() / group;
        let q = quant::quantize(values, rows, group, symmetric);
        for (r, row) in values.chunks_exact_mut(group).enumerate() {
            quant::dequantize_row(&q, group, r, row);
        }
        return;
    }
    for g in values.chunks_mut(group) {
        if symmetric {
            let top = (1i32 << (bits - 1)) - 1;
            let peak = g.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            if peak == 0.0 {
                continue;
            }
            let scale = peak / top as f32;
            for v in g.iter_mut() {
                *v = (*v / scale).round().clamp(-(top as f32) - 1.0, top as f32) * scale;
            }
        } else {
            let top = ((1u32 << bits) - 1) as f32;
            let lo = g.iter().copied().fold(f32::MAX, f32::min);
            let hi = g.iter().copied().fold(f32::MIN, f32::max);
            if hi <= lo {
                continue;
            }
            let scale = (hi - lo) / top;
            for v in g.iter_mut() {
                *v = lo + ((*v - lo) / scale).round().clamp(0.0, top) * scale;
            }
        }
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
    fn a_long_voice_is_cut_where_it_is_quiet() {
        let rate = 16_000;
        // 70 s of a tone, silent at 24 s and 52 s.
        let mut voice: Vec<f32> = (0..70 * rate)
            .map(|i| (i as f32 * 0.03).sin() * 0.5)
            .collect();
        for q in [24, 52] {
            for v in &mut voice[q * rate..q * rate + rate / 2] {
                *v = 0.0;
            }
        }
        let cuts = parts_of(&voice);
        assert_eq!(cuts.len(), 4, "{cuts:?}");
        assert_eq!((cuts[0], cuts[3]), (0, voice.len()));
        for (cut, quiet) in cuts[1..3].iter().zip([24, 52]) {
            let at = *cut as f64 / f64::from(rate as u32);
            assert!(at >= quiet as f64 && at <= quiet as f64 + 0.5, "{at}");
        }
        assert_eq!(parts_of(&voice[..20 * rate]), vec![0, 20 * rate]);
    }

    #[test]
    fn a_loop_is_cut_after_two_turns() {
        let mut ids: Vec<u32> = (0..10).collect();
        assert_eq!(loop_start(&ids), None);
        for _ in 0..16 {
            ids.extend([7, 8]);
        }
        assert_eq!(loop_start(&ids), Some(10 + 4));
        let chorus: Vec<u32> = (0..40).map(|i| i % 9).collect();
        assert_eq!(loop_start(&chorus), None, "a longer cycle is a chorus");
    }

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
