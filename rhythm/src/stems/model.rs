//! Mini-BS-RoFormer on the CPU: from a chunk's STFT to the four stems'
//! masks.
//!
//! - **Band split**: each frame's bins (two channels, real and imaginary)
//!   are cut into 62 bands of 2 to 129 bins; each band goes through an
//!   RMSNorm and a layer to 256 features. The grid of frames × bands
//!   gets four more of each, zero but for the corner of 4 × 4, which
//!   holds learned register tokens.
//! - **8 layers**, each a transformer along the frames (one sequence per
//!   band), then one along the bands (one per frame): RMSNorm, attention
//!   with 8 query heads on 4 key/value heads and rotary positions (split
//!   halves; the padding is not turned), RMSNorm, and a gated GELU
//!   feed-forward step to 768 and back, each added back.
//! - An RMSNorm, then per stem and band a layer to the band's mask: a
//!   complex factor per bin and channel.

use super::stft::BINS;
use super::STEMS;
use crate::net::layers::{dot, exp_shifted, fast, gelu, par_chunks, Linear, Rotary, HEAD};
use crate::net::weights::Weights;

/// Bins per band, low to high.
pub const BAND_BINS: [usize; 62] = [
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 24, 24, 24, 24, 24, 24, 24, 24, 48, 48, 48, 48, 48,
    48, 48, 48, 128, 129,
];
/// Features per token.
pub const DIM: usize = 256;
/// The feed-forward step's width.
const HIDDEN: usize = 768;
/// Query heads, and the key/value heads they share.
const HEADS: usize = 8;
const KV_HEADS: usize = 4;
/// Layers (each a time and a band transformer).
const DEPTH: usize = 8;
/// Register frames and bands added to the grid.
const REGISTERS: usize = 4;
/// Values per frame: bins × channels × (real, imaginary).
pub(crate) const FRAME: usize = BINS * 2 * 2;
/// RMSNorm's epsilon.
const EPS: f32 = 1e-6;

/// RMS normalisation: each token divided by the root of its mean square
/// (plus [`EPS`]), times a weight per feature.
struct Norm {
    weight: Vec<f32>,
}

impl Norm {
    fn load(w: &mut Weights, name: &str, dim: usize) -> Result<Norm, String> {
        Ok(Norm {
            weight: w.take(&format!("{name}.weight"), &[dim])?,
        })
    }

    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let dim = self.weight.len();
        let mut y = x.to_vec();
        for token in y.chunks_mut(dim) {
            let k = 1.0 / (dot(token, token) / dim as f32 + EPS).sqrt();
            for (v, g) in token.iter_mut().zip(&self.weight) {
                *v *= k * g;
            }
        }
        y
    }
}

/// One transformer layer.
struct Block {
    attention_norm: Norm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    mlp_norm: Norm,
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Block {
    fn load(w: &mut Weights, name: &str) -> Result<Block, String> {
        let kv = KV_HEADS * HEAD;
        let linear = |w: &mut Weights, part: &str, inputs, outputs| {
            Linear::load(w, &format!("{name}.{part}"), inputs, outputs, false)
        };
        Ok(Block {
            attention_norm: Norm::load(w, &format!("{name}.input_layernorm"), DIM)?,
            q: linear(w, "self_attn.q_proj", DIM, DIM)?,
            k: linear(w, "self_attn.k_proj", DIM, kv)?,
            v: linear(w, "self_attn.v_proj", DIM, kv)?,
            o: linear(w, "self_attn.o_proj", DIM, DIM)?,
            mlp_norm: Norm::load(w, &format!("{name}.post_attention_layernorm"), DIM)?,
            gate: linear(w, "mlp.gate_proj", DIM, HIDDEN)?,
            up: linear(w, "mlp.up_proj", DIM, HIDDEN)?,
            down: linear(w, "mlp.down_proj", HIDDEN, DIM)?,
        })
    }

    /// The layer on `x`: sequences of `len` consecutive tokens, whose
    /// first `turned` positions are turned by `rotary`.
    fn apply(&self, x: &mut [f32], len: usize, turned: usize, rotary: &Rotary, threads: usize) {
        let attended = self.attend(&self.attention_norm.apply(x), len, turned, rotary, threads);
        for (a, b) in x.iter_mut().zip(attended) {
            *a += b;
        }
        let xn = self.mlp_norm.apply(x);
        let mut gate = self.gate.apply(&xn, threads);
        let up = self.up.apply(&xn, threads);
        par_chunks(&mut gate, 1 << 14, threads, |i, piece| {
            gelu(piece);
            for (g, u) in piece.iter_mut().zip(&up[i << 14..]) {
                *g *= u;
            }
        });
        for (a, b) in x.iter_mut().zip(self.down.apply(&gate, threads)) {
            *a += b;
        }
    }

    /// Attention over `xn` (normalised), through the output layer.
    fn attend(
        &self,
        xn: &[f32],
        len: usize,
        turned: usize,
        rotary: &Rotary,
        threads: usize,
    ) -> Vec<f32> {
        let tokens = xn.len() / DIM;
        let (q, k, v) = (
            self.q.apply(xn, threads),
            self.k.apply(xn, threads),
            self.v.apply(xn, threads),
        );
        // Queries as [seq][head][position][HEAD]; keys and values
        // transposed as [seq][kv head][HEAD][position], so the inner loops
        // run along positions.
        let mut queries = vec![0.0f32; tokens * DIM];
        par_chunks(&mut queries, len * DIM, threads, |seq, block| {
            for h in 0..HEADS {
                for p in 0..len {
                    let out = &mut block[(h * len + p) * HEAD..][..HEAD];
                    out.copy_from_slice(&q[(seq * len + p) * DIM + h * HEAD..][..HEAD]);
                    if p < turned {
                        rotary.rotate_halves(out, p);
                    }
                }
            }
        });
        let kv = KV_HEADS * HEAD;
        let pack = |source: &[f32], turn: bool| {
            let mut packed = vec![0.0f32; tokens * kv];
            par_chunks(&mut packed, len * kv, threads, |seq, block| {
                let mut buf = [0.0f32; HEAD];
                for h in 0..KV_HEADS {
                    let head = &mut block[h * HEAD * len..][..HEAD * len];
                    for p in 0..len {
                        buf.copy_from_slice(&source[(seq * len + p) * kv + h * HEAD..][..HEAD]);
                        if turn && p < turned {
                            rotary.rotate_halves(&mut buf, p);
                        }
                        for (d, b) in buf.iter().enumerate() {
                            head[d * len + p] = *b;
                        }
                    }
                }
            });
            packed
        };
        let (keys, values) = (pack(&k, true), pack(&v, false));
        // Each task: up to 64 queries of one sequence, all heads.
        let rows = 64.min(len);
        let mut out = vec![0.0f32; tokens * DIM];
        let mut tasks: Vec<(usize, usize, &mut [f32])> = Vec::new();
        for (seq, block) in out.chunks_mut(len * DIM).enumerate() {
            for (j, piece) in block.chunks_mut(rows * DIM).enumerate() {
                tasks.push((seq, j * rows, piece));
            }
        }
        par_chunks(&mut tasks, 1, threads, |_, task| {
            let (seq, first, ref mut piece) = task[0];
            let mut scores = vec![0.0f32; len];
            attend_grouped(
                &queries,
                &keys,
                &values,
                len,
                seq,
                first,
                piece,
                &mut scores,
            );
        });
        self.o.apply(&out, threads)
    }
}

fast! {
    /// Attention for queries `first..` of sequence `seq`, all heads, into
    /// `piece` (whole tokens); query head `h` uses key/value head
    /// `h / (HEADS / KV_HEADS)`. See [`Block::attend`] for the layouts.
    #[allow(clippy::too_many_arguments)]
    fn attend_grouped(
        queries: &[f32],
        keys: &[f32],
        values: &[f32],
        len: usize,
        seq: usize,
        first: usize,
        piece: &mut [f32],
        scores: &mut [f32],
    ) {
        let scale = 1.0 / (HEAD as f32).sqrt();
        let group = HEADS / KV_HEADS;
        for h in 0..HEADS {
            let kv_base = (seq * KV_HEADS + h / group) * HEAD * len;
            let k = &keys[kv_base..][..HEAD * len];
            let v = &values[kv_base..][..HEAD * len];
            let q_base = (seq * HEADS + h) * len * HEAD;
            for (r, token) in piece.chunks_mut(DIM).enumerate() {
                let query = &queries[q_base + (first + r) * HEAD..][..HEAD];
                scores.fill(0.0);
                for (qd, key_d) in query.iter().zip(k.chunks_exact(len)) {
                    let qd = qd * scale;
                    for (s, kd) in scores.iter_mut().zip(key_d) {
                        *s += qd * kd;
                    }
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum = exp_shifted(scores, max);
                for (o, value_d) in token[h * HEAD..][..HEAD].iter_mut().zip(v.chunks_exact(len)) {
                    *o = dot(scores, value_d) / sum;
                }
            }
        }
    }
}

/// The separator network.
pub(crate) struct Model {
    split: Vec<(Norm, Linear)>,
    /// Per layer: the time transformer, then the band transformer.
    layers: Vec<[Block; 2]>,
    /// [frame][band][DIM] for the 4 × 4 corner.
    registers: Vec<f32>,
    final_norm: Norm,
    /// Per stem, per band.
    masks: Vec<Vec<Linear>>,
}

/// Each band's first bin.
fn band_starts() -> Vec<usize> {
    let mut starts = Vec::with_capacity(BAND_BINS.len());
    let mut at = 0;
    for &b in &BAND_BINS {
        starts.push(at);
        at += b;
    }
    starts
}

impl Model {
    /// The network from `w`, which must hold exactly its weights.
    pub fn load(mut w: Weights) -> Result<Model, String> {
        let split = BAND_BINS
            .iter()
            .enumerate()
            .map(|(b, &bins)| {
                let name = format!("model.band_split.to_features.{b}");
                Ok((
                    Norm::load(&mut w, &format!("{name}.0"), 4 * bins)?,
                    Linear::load(&mut w, &format!("{name}.1"), 4 * bins, DIM, true)?,
                ))
            })
            .collect::<Result<_, String>>()?;
        let layers = (0..DEPTH)
            .map(|l| {
                let block = |w: &mut Weights, which| {
                    Block::load(w, &format!("model.layers.{l}.{which}.layers.0"))
                };
                Ok([block(&mut w, 0)?, block(&mut w, 1)?])
            })
            .collect::<Result<_, String>>()?;
        let registers = w.take("model.register_tokens", &[REGISTERS, REGISTERS, DIM])?;
        let final_norm = Norm::load(&mut w, "model.final_norm", DIM)?;
        let masks = (0..STEMS)
            .map(|s| {
                BAND_BINS
                    .iter()
                    .enumerate()
                    .map(|(b, &bins)| {
                        let name = format!("mask_estimators.{s}.to_freq_mlps.{b}");
                        Linear::load(&mut w, &name, DIM, 4 * bins, true)
                    })
                    .collect::<Result<_, String>>()
            })
            .collect::<Result<_, String>>()?;
        let left = w.left();
        if !left.is_empty() {
            return Err(format!("unused weights: {left:?}"));
        }
        Ok(Model {
            split,
            layers,
            registers,
            final_norm,
            masks,
        })
    }

    /// The masks of a chunk of `frames` frames of the mixture's STFT, each
    /// [`FRAME`] values laid out as bin, then channel, then real and
    /// imaginary part: [`STEMS`] blocks of `frames × FRAME` values in the
    /// same layout.
    pub fn masks(&self, frames: usize, stft: &[f32], threads: usize) -> Vec<f32> {
        debug_assert_eq!(stft.len(), frames * FRAME);
        let starts = band_starts();
        let bands = BAND_BINS.len();
        let (rows, cols) = (frames + REGISTERS, bands + REGISTERS);
        // The grid, [frame][band][DIM].
        let mut grid = vec![0.0f32; rows * cols * DIM];
        for (b, ((norm, layer), (&start, &bins))) in self
            .split
            .iter()
            .zip(starts.iter().zip(&BAND_BINS))
            .enumerate()
        {
            let input: Vec<f32> = (0..frames)
                .flat_map(|t| &stft[t * FRAME + 4 * start..][..4 * bins])
                .copied()
                .collect();
            let features = layer.apply(&norm.apply(&input), threads);
            for t in 0..frames {
                grid[(t * cols + b) * DIM..][..DIM].copy_from_slice(&features[t * DIM..][..DIM]);
            }
        }
        for i in 0..REGISTERS {
            for j in 0..REGISTERS {
                let at = ((frames + i) * cols + bands + j) * DIM;
                grid[at..at + DIM]
                    .copy_from_slice(&self.registers[(i * REGISTERS + j) * DIM..][..DIM]);
            }
        }
        let rotary = Rotary::new(frames.max(bands));
        let mut by_band = vec![0.0f32; grid.len()];
        for [time, band] in &self.layers {
            // Along the frames: one sequence per band.
            transpose(&grid, &mut by_band, rows, cols);
            time.apply(&mut by_band, rows, frames, &rotary, threads);
            transpose(&by_band, &mut grid, cols, rows);
            // Along the bands: one sequence per frame.
            band.apply(&mut grid, cols, bands, &rotary, threads);
        }
        // The real frames and bands, normalised, band by band.
        let mut out = vec![0.0f32; STEMS * frames * FRAME];
        for (b, (&start, &bins)) in starts.iter().zip(&BAND_BINS).enumerate() {
            let tokens: Vec<f32> = (0..frames)
                .flat_map(|t| &grid[(t * cols + b) * DIM..][..DIM])
                .copied()
                .collect();
            let tokens = self.final_norm.apply(&tokens);
            for (s, stem) in self.masks.iter().enumerate() {
                let mask = stem[b].apply(&tokens, threads);
                let block = &mut out[s * frames * FRAME..][..frames * FRAME];
                for t in 0..frames {
                    block[t * FRAME + 4 * start..][..4 * bins]
                        .copy_from_slice(&mask[t * 4 * bins..][..4 * bins]);
                }
            }
        }
        out
    }
}

/// `to` = `from` ([rows][cols] tokens of [`DIM`]) as [cols][rows].
fn transpose(from: &[f32], to: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        for c in 0..cols {
            to[(c * rows + r) * DIM..][..DIM].copy_from_slice(&from[(r * cols + c) * DIM..][..DIM]);
        }
    }
}
