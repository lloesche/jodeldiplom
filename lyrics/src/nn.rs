//! The building blocks of the speech models on the CPU ([`crate::speech`]
//! and [`crate::asr`]): linear layers, the LayerNorm, attention, GELU and
//! the matrix kernels they run on, which both use; the RMS norm and the
//! strided convolution, which only the recogniser uses.

use std::ops::Range;

use jodel_tensors::fast;
use jodel_tensors::kernels::{par_rows, transposed};
use jodel_tensors::SafeTensors;

/// The LayerNorm's epsilon (both models').
const LN_EPS: f32 = 1e-5;
/// The RMS norm's epsilon (the recogniser's).
const RMS_EPS: f32 = 1e-6;

/// A linear layer, its weight transposed to `[input][output]`.
pub(crate) struct Linear {
    pub(crate) inputs: usize,
    pub(crate) outputs: usize,
    pub(crate) weight: Vec<f32>,
    pub(crate) bias: Option<Vec<f32>>,
}

impl Linear {
    /// From a `[outputs][inputs]` weight.
    pub(crate) fn from_rows(
        raw: &[f32],
        outputs: usize,
        inputs: usize,
        bias: Option<Vec<f32>>,
    ) -> Linear {
        Linear {
            inputs,
            outputs,
            weight: transposed(raw, outputs, inputs),
            bias,
        }
    }

    /// `{name}.weight` (`[outputs][inputs]`) and `{name}.bias` of `t`.
    pub(crate) fn load(
        t: &SafeTensors,
        name: &str,
        inputs: usize,
        outputs: usize,
    ) -> Result<Linear, String> {
        let raw = t.get(&format!("{name}.weight"), &[outputs, inputs])?;
        let bias = t.get(&format!("{name}.bias"), &[outputs])?;
        Ok(Linear::from_rows(&raw, outputs, inputs, Some(bias)))
    }

    /// `{name}.weight` (`[outputs][inputs]`) of `t`, a layer without a bias.
    pub(crate) fn load_unbiased(
        t: &SafeTensors,
        name: &str,
        inputs: usize,
        outputs: usize,
    ) -> Result<Linear, String> {
        let raw = t.get(&format!("{name}.weight"), &[outputs, inputs])?;
        Ok(Linear::from_rows(&raw, outputs, inputs, None))
    }

    /// `x` (`rows × inputs`) through the layer.
    pub(crate) fn forward(&self, x: &[f32], threads: usize) -> Vec<f32> {
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

/// A LayerNorm's scale and shift.
pub(crate) struct LayerNorm {
    pub(crate) weight: Vec<f32>,
    pub(crate) bias: Vec<f32>,
}

impl LayerNorm {
    pub(crate) fn load(t: &SafeTensors, name: &str, dim: usize) -> Result<LayerNorm, String> {
        Ok(LayerNorm {
            weight: t.get(&format!("{name}.weight"), &[dim])?,
            bias: t.get(&format!("{name}.bias"), &[dim])?,
        })
    }

    /// Each row of `x` normalised, in place.
    pub(crate) fn apply(&self, x: &mut [f32]) {
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
pub(crate) fn rms_norm(x: &mut [f32], weight: &[f32]) {
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

/// Steps `n` values of a convolution with stride 2 and padding 1 give.
pub(crate) const fn halved(n: usize) -> usize {
    n.div_ceil(2)
}

/// A 3×3 convolution with stride 2 and padding 1, then GELU: its weight as
/// `[tap·input][output]` (tap-major, the taps row by row over 3 bands × 3
/// steps), and its bias.
pub(crate) struct Conv {
    pub(crate) inputs: usize,
    pub(crate) outputs: usize,
    pub(crate) weight: Vec<f32>,
    pub(crate) bias: Vec<f32>,
}

/// A 3×3 convolution's weight as a checkpoint has it
/// (`[outputs][inputs][3][3]`) in rows, an output each, its inputs
/// tap-major: `[outputs][tap·inputs + input]`.
pub(crate) fn tap_major(raw: &[f32], inputs: usize, outputs: usize) -> Vec<f32> {
    let mut rows = vec![0.0f32; raw.len()];
    for o in 0..outputs {
        for i in 0..inputs {
            for tap in 0..9 {
                rows[(o * 9 + tap) * inputs + i] = raw[(o * inputs + i) * 9 + tap];
            }
        }
    }
    rows
}

impl Conv {
    /// `{name}.weight` (`[outputs][inputs][3][3]`) and `{name}.bias` of `t`.
    pub(crate) fn load(
        t: &SafeTensors,
        name: &str,
        inputs: usize,
        outputs: usize,
    ) -> Result<Conv, String> {
        let raw = t.get(&format!("{name}.weight"), &[outputs, inputs, 3, 3])?;
        Ok(Conv {
            inputs,
            outputs,
            weight: transposed(&tap_major(&raw, inputs, outputs), outputs, 9 * inputs),
            bias: t.get(&format!("{name}.bias"), &[outputs])?,
        })
    }

    /// `x` (`[images][bands][steps][inputs]`) through the convolution and
    /// GELU: `[images][bands/2][steps/2][outputs]` (rounded up), and its
    /// bands and steps.
    pub(crate) fn forward(
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
        let mut out = vec![0.0f32; rows * self.outputs];
        matmul(
            &self.weight,
            width,
            self.outputs,
            &patches,
            &mut out,
            threads,
        );
        for row in out.chunks_exact_mut(self.outputs) {
            for (v, b) in row.iter_mut().zip(&self.bias) {
                *v += b;
            }
        }
        gelu(&mut out);
        (out, ob, os)
    }
}

/// The heads of an attention: `query` heads of `width`, and `kv` key and
/// value heads, each shared by a group of `query / kv` query heads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Heads {
    pub(crate) query: usize,
    pub(crate) kv: usize,
    pub(crate) width: usize,
}

/// Scaled dot-product attention: each row of `q` (`heads.query` heads)
/// over the rows of `k` and `v` (`heads.kv` heads each) that `keys_of`
/// gives for it. The attended rows, as wide as `q`'s.
pub(crate) fn attend(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: Heads,
    keys_of: impl Fn(usize) -> Range<usize> + Sync,
    threads: usize,
) -> Vec<f32> {
    debug_assert!(heads.query % heads.kv == 0, "{heads:?}");
    let head = heads.width;
    let (width, kv) = (heads.query * head, heads.kv * head);
    let group = heads.query / heads.kv;
    let mut attended = vec![0.0f32; q.len()];
    let scale = 1.0 / (head as f32).sqrt();
    par_rows(&mut attended, width, threads, |t, row| {
        let keys = keys_of(t);
        let mut scores = vec![0.0f32; keys.len()];
        for h in 0..heads.query {
            let g = h / group;
            let qh = &q[t * width + h * head..][..head];
            let mut best = f32::MIN;
            for (u, s) in scores.iter_mut().enumerate() {
                let kh = &k[(keys.start + u) * kv + g * head..][..head];
                *s = dot(qh, kh) * scale;
                best = best.max(*s);
            }
            let mut total = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - best).exp();
                total += *s;
            }
            let out = &mut row[h * head..(h + 1) * head];
            for (u, &s) in scores.iter().enumerate() {
                let vh = &v[(keys.start + u) * kv + g * head..][..head];
                let w = s / total;
                for (o, x) in out.iter_mut().zip(vh) {
                    *o += w * x;
                }
            }
        }
    });
    attended
}

/// The dot product of `a` and `b`, in eight lanes.
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
    let mut sum: f32 = lanes.iter().sum();
    for (x, y) in ar.iter().zip(br) {
        sum += x * y;
    }
    sum
}

/// `out` (`rows × n`) = `x` (`rows × k`) · `w` (`k × n`), rows spread over
/// the threads.
pub(crate) fn matmul(w: &[f32], k: usize, n: usize, x: &[f32], out: &mut [f32], threads: usize) {
    let rows = out.len() / n.max(1);
    if rows == 0 {
        return;
    }
    // Pieces of 16 rows, handed out in turn.
    let pieces = rows.div_ceil(16);
    let threads = threads.min(pieces).max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let chunks: Vec<std::sync::Mutex<&mut [f32]>> =
        out.chunks_mut(16 * n).map(std::sync::Mutex::new).collect();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let p = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if p >= pieces {
                    break;
                }
                let mut chunk = chunks[p].lock().unwrap();
                let r0 = p * 16;
                let r = chunk.len() / n;
                matmul_rows(w, k, n, &x[r0 * k..(r0 + r) * k], &mut chunk);
            });
        }
    });
}

fast! {
    /// Rows of `x` through `w`: tiles of 4 rows by 16 outputs kept in
    /// registers.
    fn matmul_rows(w: &[f32], k: usize, n: usize, x: &[f32], out: &mut [f32]) {
        let rows = out.len() / n;
        let mut r = 0;
        while r < rows {
            let m = (rows - r).min(4);
            let row = |i: usize| &x[(r + i.min(m - 1)) * k..][..k];
            let (x0, x1, x2, x3) = (row(0), row(1), row(2), row(3));
            let mut o = 0;
            while o + 16 <= n {
                let mut acc = [[0.0f32; 16]; 4];
                for kk in 0..k {
                    let wk: &[f32; 16] = w[kk * n + o..][..16].try_into().unwrap();
                    let xs = [x0[kk], x1[kk], x2[kk], x3[kk]];
                    for (a, xv) in acc.iter_mut().zip(xs) {
                        for j in 0..16 {
                            a[j] = xv.mul_add(wk[j], a[j]);
                        }
                    }
                }
                for (i, a) in acc.iter().enumerate().take(m) {
                    out[(r + i) * n + o..][..16].copy_from_slice(a);
                }
                o += 16;
            }
            for o in o..n {
                for i in 0..m {
                    let xi = row(i);
                    let mut sum = 0.0f32;
                    for kk in 0..k {
                        sum = xi[kk].mul_add(w[kk * n + o], sum);
                    }
                    out[(r + i) * n + o] = sum;
                }
            }
            r += 4;
        }
    }
}

/// GELU with the error function (Abramowitz and Stegun 7.1.26, within
/// 1.5e-7), in place.
pub(crate) fn gelu(values: &mut [f32]) {
    for v in values {
        let x = *v;
        let z = x.abs() * std::f32::consts::FRAC_1_SQRT_2;
        let t = 1.0 / (1.0 + 0.327_591_1 * z);
        let poly = t
            * (0.254_829_6
                + t * (-0.284_496_74 + t * (1.421_413_8 + t * (-1.453_152 + t * 1.061_405_4))));
        let erf = (1.0 - poly * (-z * z).exp()).copysign(x);
        *v = 0.5 * x * (1.0 + erf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_convolution_s_rows_are_tap_major() {
        // 2 outputs of 3 inputs, each weight numbered as the checkpoint
        // lays them out: [output][input][tap].
        let (inputs, outputs) = (3, 2);
        let raw: Vec<f32> = (0..outputs * inputs * 9).map(|v| v as f32).collect();
        let rows = tap_major(&raw, inputs, outputs);
        for o in 0..outputs {
            for i in 0..inputs {
                for tap in 0..9 {
                    let at = o * 9 * inputs + tap * inputs + i;
                    assert_eq!(rows[at], raw[(o * inputs + i) * 9 + tap]);
                }
            }
        }
    }

    #[test]
    fn matmul_matches_the_plain_sum() {
        let (rows, k, n) = (23, 37, 41);
        let x: Vec<f32> = (0..rows * k)
            .map(|i| ((i * 7 % 13) as f32 - 6.0) / 7.0)
            .collect();
        let w: Vec<f32> = (0..k * n)
            .map(|i| ((i * 5 % 11) as f32 - 5.0) / 9.0)
            .collect();
        let mut out = vec![0.0f32; rows * n];
        matmul(&w, k, n, &x, &mut out, 3);
        for r in 0..rows {
            for o in 0..n {
                let want: f32 = (0..k).map(|kk| x[r * k + kk] * w[kk * n + o]).sum();
                assert!((out[r * n + o] - want).abs() < 1e-4, "{r} {o}");
            }
        }
    }

    #[test]
    fn attention_weighs_the_keys_it_is_given() {
        // Two query heads sharing one key/value head, three rows; each row
        // sees itself and those before it.
        let heads = Heads {
            query: 2,
            kv: 1,
            width: 2,
        };
        let head = heads.width;
        let q = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let k = [4.0, 0.0, 0.0, 4.0, 0.0, 0.0];
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = attend(&q, &k, &v, heads, |t| 0..t + 1, 2);
        // The first row has only its own value.
        assert_eq!(&out[..4], &[1.0, 2.0, 1.0, 2.0]);
        // The last: by hand, a softmax of the scores over three keys.
        let s = 1.0 / 2f32.sqrt();
        for (h, scores) in [[4.0 * s, 0.0, 0.0], [0.0, 4.0 * s, 0.0]]
            .iter()
            .enumerate()
        {
            let e: Vec<f32> = scores.iter().map(|x| (x - 4.0 * s).exp()).collect();
            let total: f32 = e.iter().sum();
            for d in 0..head {
                let want: f32 = (0..3).map(|u| e[u] / total * v[u * head + d]).sum();
                let got = out[8 + h * head + d];
                assert!((got - want).abs() < 1e-6, "{h} {d}: {got} {want}");
            }
        }
    }
}
