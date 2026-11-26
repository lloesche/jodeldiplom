//! The networks' layers, on token-major `f32` buffers.

use jodel_tensors::fast;
use jodel_tensors::kernels::{par_chunks, transposed};

use super::weights::Weights;

/// The dot product, in eight lanes the compiler can vectorise.
#[inline(always)]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let (a8, b8) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ar, br) = (a8.remainder(), b8.remainder());
    for (x, y) in a8.zip(b8) {
        for k in 0..8 {
            lanes[k] += x[k] * y[k];
        }
    }
    let mut sum = (lanes[0] + lanes[4])
        + (lanes[1] + lanes[5])
        + (lanes[2] + lanes[6])
        + (lanes[3] + lanes[7]);
    for (x, y) in ar.iter().zip(br) {
        sum += x * y;
    }
    sum
}

/// Replaces each value `x` by `e^(x - max)` (all `x <= max`) and
/// returns their sum. The exponential is a polynomial the compiler can
/// vectorise, within about 2e-7 of the exact one.
#[inline(always)]
pub(crate) fn exp_shifted(values: &mut [f32], max: f32) -> f32 {
    let mut lanes = [0.0f32; 8];
    let mut chunks = values.chunks_exact_mut(8);
    for c in &mut chunks {
        for k in 0..8 {
            c[k] = exp_neg(c[k] - max);
            lanes[k] += c[k];
        }
    }
    let mut sum: f32 = lanes.iter().sum();
    for v in chunks.into_remainder() {
        *v = exp_neg(*v - max);
        sum += *v;
    }
    sum
}

/// `e^x` for `x <= 0`: `2^n · e^r` with `|r| <= ln 2 / 2` and `e^r` by a
/// degree-6 polynomial; `e^-87` below -87.
#[inline(always)]
fn exp_neg(x: f32) -> f32 {
    let x = x.max(-87.0);
    let n = (x * std::f32::consts::LOG2_E).round();
    // ln 2 in two parts, so r keeps its precision.
    let r = x - n * 0.693_145_75 - n * 1.428_606_8e-6;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (0.166_666_67
                    + r * (0.041_666_668 + r * (0.008_333_334 + r * 0.001_388_889)))));
    p * f32::from_bits(((n as i32 + 127) << 23) as u32)
}

/// GELU with the error function, which is taken as Abramowitz and
/// Stegun's 7.1.26 (within 1.5e-7).
#[inline(always)]
fn gelu_one(x: f32) -> f32 {
    let z = x.abs() * std::f32::consts::FRAC_1_SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_74 + t * (1.421_413_8 + t * (-1.453_152 + t * 1.061_405_4))));
    let erf = (1.0 - poly * exp_neg(-z * z)).copysign(x);
    0.5 * x * (1.0 + erf)
}

fast! {
    /// GELU, in place.
    pub(crate) fn gelu(values: &mut [f32]) {
        for v in values {
            *v = gelu_one(*v);
        }
    }
}

/// A fully connected layer: `out = W·x + b`.
pub struct Linear {
    /// The length of a row in.
    pub inputs: usize,
    /// The length of a row out.
    pub outputs: usize,
    /// Transposed: a row of `outputs` per input.
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

fast! {
    /// `out` (whole rows) = the rows of `x` through the layer: tiles of 4
    /// rows by 8 outputs kept in registers.
    fn linear_rows(
        w: &[f32],
        bias: Option<&[f32]>,
        inputs: usize,
        outputs: usize,
        x: &[f32],
        out: &mut [f32],
    ) {
        let rows = out.len() / outputs;
        let mut r = 0;
        while r < rows {
            let n = (rows - r).min(4);
            // Rows past the end repeat the last one; they are not stored.
            let row = |i: usize| &x[(r + i.min(n - 1)) * inputs..][..inputs];
            let (x0, x1, x2, x3) = (row(0), row(1), row(2), row(3));
            let mut o = 0;
            while o + 8 <= outputs {
                let mut acc = [[0.0f32; 8]; 4];
                for k in 0..inputs {
                    let wk: &[f32; 8] = w[k * outputs + o..][..8].try_into().unwrap();
                    let xs = [x0[k], x1[k], x2[k], x3[k]];
                    for (a, xv) in acc.iter_mut().zip(xs) {
                        for j in 0..8 {
                            a[j] += xv * wk[j];
                        }
                    }
                }
                for (i, a) in acc.iter().enumerate().take(n) {
                    out[(r + i) * outputs + o..][..8].copy_from_slice(a);
                }
                o += 8;
            }
            for o in o..outputs {
                for i in 0..n {
                    let xi = row(i);
                    let mut sum = 0.0f32;
                    for k in 0..inputs {
                        sum += xi[k] * w[k * outputs + o];
                    }
                    out[(r + i) * outputs + o] = sum;
                }
            }
            r += 4;
        }
        if let Some(b) = bias {
            for row in out.chunks_mut(outputs) {
                for (v, b) in row.iter_mut().zip(b) {
                    *v += b;
                }
            }
        }
    }
}

impl Linear {
    /// `NAME.weight` (`[outputs][inputs]`) of `w`, and `NAME.bias` if
    /// `bias`.
    pub fn load(
        w: &mut Weights,
        name: &str,
        inputs: usize,
        outputs: usize,
        bias: bool,
    ) -> Result<Linear, String> {
        let weight = w.take(&format!("{name}.weight"), &[outputs, inputs])?;
        let bias = if bias {
            Some(w.take(&format!("{name}.bias"), &[outputs])?)
        } else {
            None
        };
        Ok(Linear::new(weight, bias, inputs, outputs))
    }

    /// A layer from its weights, a row of `inputs` per output.
    pub fn new(weight: Vec<f32>, bias: Option<Vec<f32>>, inputs: usize, outputs: usize) -> Linear {
        Linear {
            inputs,
            outputs,
            weight: transposed(&weight, outputs, inputs),
            bias,
        }
    }

    /// Its weights for `f`, as it holds them: `NAME.w` (`[inputs][outputs]`)
    /// and, where it has one, `NAME.b`.
    pub fn visit(&self, name: &str, f: &mut dyn FnMut(&str, &[f32])) {
        f(&format!("{name}.w"), &self.weight);
        if let Some(b) = &self.bias {
            f(&format!("{name}.b"), b);
        }
    }

    /// The layer applied to every row of `x` (rows of `inputs`).
    pub fn apply(&self, x: &[f32], threads: usize) -> Vec<f32> {
        let rows = x.len() / self.inputs;
        let mut y = vec![0.0f32; rows * self.outputs];
        let block = 32;
        par_chunks(&mut y, block * self.outputs, threads, |i, out| {
            let first = i * block * self.inputs;
            let n = out.len() / self.outputs;
            linear_rows(
                &self.weight,
                self.bias.as_deref(),
                self.inputs,
                self.outputs,
                &x[first..][..n * self.inputs],
                out,
            );
        });
        y
    }
}

/// RMS normalisation: each token scaled to length `√dim`, times `gamma`.
pub(crate) struct RmsNorm {
    gamma: Vec<f32>,
}

impl RmsNorm {
    pub fn load(w: &mut Weights, name: &str, dim: usize) -> Result<RmsNorm, String> {
        Ok(RmsNorm {
            gamma: w.take(&format!("{name}.gamma"), &[dim])?,
        })
    }

    /// Its `gamma` for `f`, as `NAME`.
    pub fn visit(&self, name: &str, f: &mut dyn FnMut(&str, &[f32])) {
        f(name, &self.gamma);
    }

    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        let dim = self.gamma.len();
        let scale = (dim as f32).sqrt();
        let mut y = x.to_vec();
        for token in y.chunks_mut(dim) {
            let norm = dot(token, token).sqrt().max(1e-12);
            let k = scale / norm;
            for (v, g) in token.iter_mut().zip(&self.gamma) {
                *v *= k * g;
            }
        }
        y
    }
}

/// A per-channel scale and shift (a BatchNorm, folded).
pub(crate) struct Affine {
    pub scale: Vec<f32>,
    pub shift: Vec<f32>,
}

impl Affine {
    pub fn load(w: &mut Weights, name: &str, dim: usize) -> Result<Affine, String> {
        Ok(Affine {
            scale: w.take(&format!("{name}.scale"), &[dim])?,
            shift: w.take(&format!("{name}.shift"), &[dim])?,
        })
    }

    /// Its values for `f`: `NAME.scale` and `NAME.shift`.
    pub fn visit(&self, name: &str, f: &mut dyn FnMut(&str, &[f32])) {
        f(&format!("{name}.scale"), &self.scale);
        f(&format!("{name}.shift"), &self.shift);
    }

    /// Applied to channel-last tokens.
    pub fn apply(&self, x: &mut [f32]) {
        let dim = self.scale.len();
        for token in x.chunks_mut(dim) {
            for ((v, a), b) in token.iter_mut().zip(&self.scale).zip(&self.shift) {
                *v = *v * a + b;
            }
        }
    }
}

/// Rotary position tables for heads of [`HEAD`] dimensions.
pub(crate) struct Rotary {
    /// Per position, per pair: cosine and sine.
    table: Vec<(f32, f32)>,
}

/// The dimensions of an attention head.
pub const HEAD: usize = 32;

impl Rotary {
    pub fn new(positions: usize) -> Rotary {
        let pairs = HEAD / 2;
        let mut table = Vec::with_capacity(positions * pairs);
        for p in 0..positions {
            for i in 0..pairs {
                // Single precision like the original, so long sequences
                // turn by the same angles.
                let freq = 1.0f32 / 10_000f32.powf((2 * i) as f32 / HEAD as f32);
                let angle = p as f32 * freq;
                table.push((angle.cos(), angle.sin()));
            }
        }
        Rotary { table }
    }

    /// Rotates one head's vector for `position`, its halves paired: value
    /// `i` and value `i + HEAD/2` turn together by pair `i`'s angle.
    pub fn rotate_halves(&self, v: &mut [f32], position: usize) {
        let pairs = HEAD / 2;
        let row = &self.table[position * pairs..][..pairs];
        let (a, b) = v.split_at_mut(pairs);
        for ((x, y), &(c, s)) in a.iter_mut().zip(b.iter_mut()).zip(row) {
            let (p, q) = (*x, *y);
            *x = p * c - q * s;
            *y = q * c + p * s;
        }
    }

    /// Rotates one head's vector for `position`: each pair of neighbours
    /// turns by its own angle.
    fn rotate(&self, v: &mut [f32], position: usize) {
        let pairs = HEAD / 2;
        let row = &self.table[position * pairs..][..pairs];
        for (i, &(c, s)) in row.iter().enumerate() {
            let (a, b) = (v[2 * i], v[2 * i + 1]);
            v[2 * i] = a * c - b * s;
            v[2 * i + 1] = b * c + a * s;
        }
    }
}

/// Self-attention with rotary positions and a sigmoid gate per head.
pub(crate) struct Attention {
    norm: RmsNorm,
    qkv: Linear,
    gates: Linear,
    out: Linear,
    heads: usize,
}

impl Attention {
    pub fn load(w: &mut Weights, name: &str, dim: usize) -> Result<Attention, String> {
        let heads = dim / HEAD;
        Ok(Attention {
            norm: RmsNorm::load(w, &format!("{name}.norm"), dim)?,
            qkv: Linear::load(w, &format!("{name}.to_qkv"), dim, 3 * dim, false)?,
            gates: Linear::load(w, &format!("{name}.to_gates"), dim, heads, true)?,
            out: Linear::load(w, &format!("{name}.to_out.0"), dim, dim, false)?,
            heads,
        })
    }

    /// Its weights for `f`: `NAME.norm`, `NAME.qkv.w` (`[dim][3 × dim]`),
    /// `NAME.gates.w` (`[dim][heads]`) and `NAME.gates.b`, `NAME.out.w`
    /// (`[dim][dim]`).
    pub fn visit(&self, name: &str, f: &mut dyn FnMut(&str, &[f32])) {
        self.norm.visit(&format!("{name}.norm"), f);
        self.qkv.visit(&format!("{name}.qkv"), f);
        self.gates.visit(&format!("{name}.gates"), f);
        self.out.visit(&format!("{name}.out"), f);
    }

    /// Adds the attention's output to `x`: tokens of `dim`, in sequences
    /// of `len` consecutive tokens that attend only within themselves.
    pub fn add_to(&self, x: &mut [f32], len: usize, rotary: &Rotary, threads: usize) {
        let dim = self.heads * HEAD;
        let tokens = x.len() / dim;
        let seqs = tokens / len;
        let xn = self.norm.apply(x);
        let qkv = self.qkv.apply(&xn, threads);
        let mut gates = self.gates.apply(&xn, threads);
        for g in &mut gates {
            *g = 1.0 / (1.0 + (-*g).exp());
        }
        // Per sequence and head, rotated: queries as [position][HEAD],
        // keys and values transposed as [HEAD][position], so the inner
        // loops run along positions. [part][seq][head][..].
        let part = tokens * dim;
        let mut packed = vec![0.0f32; 3 * part];
        par_chunks(&mut packed, len * dim, threads, |i, block| {
            let (which, seq) = (i / seqs, i % seqs);
            let mut buf = [0.0f32; HEAD];
            for h in 0..self.heads {
                let head = &mut block[h * len * HEAD..][..len * HEAD];
                for p in 0..len {
                    let token = seq * len + p;
                    buf.copy_from_slice(&qkv[token * 3 * dim + which * dim + h * HEAD..][..HEAD]);
                    if which < 2 {
                        rotary.rotate(&mut buf, p);
                    }
                    if which == 0 {
                        head[p * HEAD..][..HEAD].copy_from_slice(&buf);
                    } else {
                        for (d, b) in buf.iter().enumerate() {
                            head[d * len + p] = *b;
                        }
                    }
                }
            }
        });
        let (q, rest) = packed.split_at(part);
        let (k, v) = rest.split_at(part);
        // Each task: up to 64 queries of one sequence, all heads.
        let rows = 64.min(len);
        let mut out = vec![0.0f32; part];
        let mut tasks: Vec<(usize, usize, &mut [f32])> = Vec::new();
        for (seq, block) in out.chunks_mut(len * dim).enumerate() {
            for (j, piece) in block.chunks_mut(rows * dim).enumerate() {
                tasks.push((seq, j * rows, piece));
            }
        }
        let heads = self.heads;
        par_chunks(&mut tasks, 1, threads, |_, task| {
            let (seq, first, ref mut piece) = task[0];
            let mut scores = vec![0.0f32; len];
            attend(q, k, v, &gates, heads, len, seq, first, piece, &mut scores);
        });
        let y = self.out.apply(&out, threads);
        for (a, b) in x.iter_mut().zip(y) {
            *a += b;
        }
    }
}

fast! {
    /// Attention for queries `first..` of sequence `seq`, all heads, into
    /// `piece` (whole tokens); see [`Attention::add_to`] for the layouts.
    #[allow(clippy::too_many_arguments)]
    fn attend(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gates: &[f32],
        heads: usize,
        len: usize,
        seq: usize,
        first: usize,
        piece: &mut [f32],
        scores: &mut [f32],
    ) {
        let dim = heads * HEAD;
        let scale = 1.0 / (HEAD as f32).sqrt();
        for h in 0..heads {
            let base = (seq * heads + h) * len * HEAD;
            let keys = &k[base..][..len * HEAD];
            let values = &v[base..][..len * HEAD];
            for (r, token_out) in piece.chunks_mut(dim).enumerate() {
                let p = first + r;
                let query = &q[base + p * HEAD..][..HEAD];
                scores.fill(0.0);
                for (qd, key_d) in query.iter().zip(keys.chunks_exact(len)) {
                    let qd = qd * scale;
                    for (s, kd) in scores.iter_mut().zip(key_d) {
                        *s += qd * kd;
                    }
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum = exp_shifted(scores, max);
                let gate = gates[(seq * len + p) * heads + h] / sum;
                for (o, value_d) in token_out[h * HEAD..][..HEAD]
                    .iter_mut()
                    .zip(values.chunks_exact(len))
                {
                    *o = dot(scores, value_d) * gate;
                }
            }
        }
    }
}

/// RMSNorm, a layer four times as wide, GELU, back to the width.
pub(crate) struct FeedForward {
    norm: RmsNorm,
    up: Linear,
    down: Linear,
}

impl FeedForward {
    pub fn load(w: &mut Weights, name: &str, dim: usize) -> Result<FeedForward, String> {
        Ok(FeedForward {
            norm: RmsNorm::load(w, &format!("{name}.net.0"), dim)?,
            up: Linear::load(w, &format!("{name}.net.1"), dim, 4 * dim, true)?,
            down: Linear::load(w, &format!("{name}.net.4"), 4 * dim, dim, true)?,
        })
    }

    /// Its weights for `f`: `NAME.norm`, `NAME.up.w` (`[dim][4 × dim]`) and
    /// `NAME.up.b`, `NAME.down.w` (`[4 × dim][dim]`) and `NAME.down.b`.
    pub fn visit(&self, name: &str, f: &mut dyn FnMut(&str, &[f32])) {
        self.norm.visit(&format!("{name}.norm"), f);
        self.up.visit(&format!("{name}.up"), f);
        self.down.visit(&format!("{name}.down"), f);
    }

    /// Adds the block's output to `x`.
    pub fn add_to(&self, x: &mut [f32], threads: usize) {
        let mut h = self.up.apply(&self.norm.apply(x), threads);
        par_chunks(&mut h, 1 << 14, threads, |_, piece| gelu(piece));
        let y = self.down.apply(&h, threads);
        for (a, b) in x.iter_mut().zip(y) {
            *a += b;
        }
    }
}

/// A 2-D convolution over (frequency, time) without bias: a kernel of
/// `kf` bands by 3 frames, `kf` bands apart (stride), frames padded by
/// one zero frame at each end.
pub(crate) struct Conv {
    kf: usize,
    inputs: usize,
    /// Weights as a layer over `[kf][3][inputs]` patches.
    layer: Linear,
}

impl Conv {
    pub fn load(
        w: &mut Weights,
        name: &str,
        inputs: usize,
        outputs: usize,
        kf: usize,
    ) -> Result<Conv, String> {
        let raw = w.take(&format!("{name}.weight"), &[outputs, inputs, kf, 3])?;
        // [out][in][kf][kt] → [out][kf][kt][in].
        let mut weight = vec![0.0f32; raw.len()];
        for o in 0..outputs {
            for i in 0..inputs {
                for f in 0..kf {
                    for t in 0..3 {
                        weight[((o * kf + f) * 3 + t) * inputs + i] =
                            raw[((o * inputs + i) * kf + f) * 3 + t];
                    }
                }
            }
        }
        Ok(Conv {
            kf,
            inputs,
            layer: Linear::new(weight, None, kf * 3 * inputs, outputs),
        })
    }

    /// Its weights for `f`: `NAME.w`, as a layer over its patches
    /// (`[kf × 3 × inputs][outputs]`, a patch's bands, then its frames,
    /// then its channels).
    pub fn visit(&self, name: &str, f: &mut dyn FnMut(&str, &[f32])) {
        self.layer.visit(name, f);
    }

    /// `x` is `[band][frame][channel]` with `bands` bands of `frames`
    /// frames; the result has `bands / kf` bands, same layout.
    pub fn apply(&self, x: &[f32], bands: usize, frames: usize, threads: usize) -> Vec<f32> {
        let c = self.inputs;
        let out_bands = bands / self.kf;
        let patch = self.kf * 3 * c;
        let mut patches = vec![0.0f32; out_bands * frames * patch];
        par_chunks(&mut patches, frames * patch, threads, |fo, block| {
            for t in 0..frames {
                let p = &mut block[t * patch..][..patch];
                for f in 0..self.kf {
                    let band = fo * self.kf + f;
                    for dt in 0..3 {
                        let dst = &mut p[(f * 3 + dt) * c..][..c];
                        match (t + dt).checked_sub(1).filter(|&s| s < frames) {
                            Some(s) => dst.copy_from_slice(&x[(band * frames + s) * c..][..c]),
                            None => dst.fill(0.0),
                        }
                    }
                }
            }
        });
        self.layer.apply(&patches, threads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_values() {
        let mut v = [1.0f32, -1.0, 3.0, 0.0, -8.0, 0.25];
        gelu(&mut v);
        let want = [
            0.841_344_7,
            -0.158_655_3,
            2.995_950_3,
            0.0,
            0.0,
            0.149_676_5,
        ];
        for (a, b) in v.iter().zip(want) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn the_layer_matches_a_plain_product() {
        for (inputs, outputs, rows) in [(5, 3, 7), (32, 96, 9), (64, 2, 33), (12, 16, 4)] {
            let weight: Vec<f32> = (0..inputs * outputs)
                .map(|i| ((i * 7) % 11) as f32 - 5.0)
                .collect();
            let bias: Vec<f32> = (0..outputs).map(|o| o as f32 * 0.5).collect();
            let x: Vec<f32> = (0..rows * inputs)
                .map(|i| ((i * 3) % 7) as f32 * 0.25)
                .collect();
            let layer = Linear::new(weight.clone(), Some(bias.clone()), inputs, outputs);
            let y = layer.apply(&x, 3);
            for r in 0..rows {
                for o in 0..outputs {
                    let want: f32 = (0..inputs)
                        .map(|i| x[r * inputs + i] * weight[o * inputs + i])
                        .sum::<f32>()
                        + bias[o];
                    assert!(
                        (y[r * outputs + o] - want).abs() < 1e-3,
                        "{inputs}x{outputs} row {r} out {o}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_fast_exponential() {
        let mut v: Vec<f32> = (0..190).map(|i| -(i as f32) * 0.45).collect();
        let want: Vec<f32> = v.iter().map(|x| x.exp()).collect();
        let sum = exp_shifted(&mut v, 0.0);
        for (a, b) in v.iter().zip(&want) {
            assert!((a - b).abs() <= 3e-7 * b + 1e-38, "{a} vs {b}");
        }
        assert!((sum - want.iter().sum::<f32>()).abs() < 1e-5);
    }

    #[test]
    fn dot_matches_a_plain_sum() {
        let a: Vec<f32> = (0..37).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..37).map(|i| 1.0 - i as f32 * 0.01).collect();
        let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - want).abs() < 1e-3);
    }
}
