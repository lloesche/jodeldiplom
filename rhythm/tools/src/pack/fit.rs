//! K blocks (Q4_K, Q5_K: `jodel_rhythm::nn::quant` reads them) fitted to
//! values.
//!
//! [`quantize`] fits a block by [`Fit`]: without importance, as GGML's
//! plain reference does (each run's scale and minimum by a search over 21
//! grids, a value weighted by its run's RMS plus its magnitude, then the
//! block's 6-bit scales from the largest); with it, as GGML does with an
//! importance matrix (a value weighted by its column's importance times
//! `√(σ² + x²)`, `σ²` twice the block's mean square, a search over 37
//! grids, the 6-bit scales fitted to the least error weighted by their
//! runs' weights). Refining then searches each run's 6-bit scale and
//! minimum near the fitted ones, and fits `d` and `dmin` to the levels by
//! least squares, while the weighted error falls.

use jodel_rhythm::nn::quant::{
    f16_to_f32, fifth_bit, Bits, BLOCK, D, DMIN, FIFTH_BITS, RUN, SCALES,
};

use super::f32_to_f16;

/// Runs a block holds.
const RUNS: usize = BLOCK / RUN;

/// How [`quantize`] fits a K block.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fit<'a> {
    /// How much each column's error counts, one a column of the rows (the
    /// mean square of what the column multiplies, say).
    pub importance: Option<&'a [f32]>,
    /// Whether to refine the fitted scales.
    pub refine: bool,
}

/// A K block before packing: the halves, each run's 6-bit scale and
/// minimum, each value's level.
pub struct KBlock {
    d: u16,
    dmin: u16,
    scales: [u8; RUNS],
    mins: [u8; RUNS],
    pub levels: [u8; BLOCK],
}

impl KBlock {
    /// Run `j`'s step and the amount below zero.
    pub fn run(&self, j: usize) -> (f32, f32) {
        (
            f16_to_f32(self.d) * f32::from(self.scales[j]),
            f16_to_f32(self.dmin) * f32::from(self.mins[j]),
        )
    }

    /// The levels of run `j` nearest `x`, against the scales as stored,
    /// and their weighted error.
    fn fill_run(&mut self, j: usize, x: &[f32], w: &[f32], top: u8) -> f32 {
        let (step, below) = self.run(j);
        let mut err = 0.0f32;
        for i in j * RUN..(j + 1) * RUN {
            let q = if step == 0.0 {
                0.0
            } else {
                ((x[i] + below) / step)
                    .round_ties_even()
                    .clamp(0.0, f32::from(top))
            };
            self.levels[i] = q as u8;
            let e = step * q - below - x[i];
            err += w[i] * e * e;
        }
        err
    }

    fn fill(&mut self, x: &[f32], w: &[f32], top: u8) -> f32 {
        (0..RUNS).map(|j| self.fill_run(j, x, w, top)).sum()
    }
}

/// A run's scale and minimum (as the amount below zero, so `x ≈ scale·l −
/// min`) and its levels in `levels`, by GGML's search: the grid from the
/// run's range, then `steps + 1` grids from `top + low` levels over the
/// range by `delta`, each refitted by weighted least squares; the least
/// weighted error wins.
fn fit_run(
    x: &[f32],
    w: &[f32],
    top: u8,
    (low, delta, steps): (f32, f32, usize),
    levels: &mut [u8],
) -> (f32, f32) {
    let top = f32::from(top);
    let (mut min, mut max) = (x[0], x[0]);
    let (mut sum_w, mut sum_x) = (0.0f32, 0.0f32);
    for (&v, &wi) in x.iter().zip(w) {
        min = min.min(v);
        max = max.max(v);
        sum_w += wi;
        sum_x += wi * v;
    }
    min = min.min(0.0);
    if max == min {
        levels.fill(0);
        return (0.0, -min);
    }
    let level =
        |iscale: f32, v: f32, min: f32| (iscale * (v - min)).round_ties_even().clamp(0.0, top);
    let mut iscale = top / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best = 0.0f32;
    for ((&v, &wi), l) in x.iter().zip(w).zip(levels.iter_mut()) {
        let q = level(iscale, v, min);
        *l = q as u8;
        let d = scale * q + min - v;
        best += wi * d * d;
    }
    let mut trial = [0u8; RUN];
    for step in 0..=steps {
        iscale = (low + delta * step as f32 + top) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0.0f32, 0.0f32, 0.0f32);
        for (i, (&v, &wi)) in x.iter().zip(w).enumerate() {
            let q = level(iscale, v, min);
            trial[i] = q as u8;
            sum_l += wi * q;
            sum_l2 += wi * q * q;
            sum_xl += wi * q * v;
        }
        let det = sum_w * sum_l2 - sum_l * sum_l;
        if det <= 0.0 {
            continue;
        }
        let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / det;
        let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / det;
        if this_min > 0.0 {
            this_min = 0.0;
            this_scale = sum_xl / sum_l2;
        }
        let mut err = 0.0f32;
        for (i, (&v, &wi)) in x.iter().zip(w).enumerate() {
            let d = this_scale * f32::from(trial[i]) + this_min - v;
            err += wi * d * d;
        }
        if err < best {
            levels.copy_from_slice(&trial[..x.len()]);
            best = err;
            scale = this_scale;
            min = this_min;
        }
    }
    (scale, -min)
}

/// The eight runs' 6-bit codes of `x` (non-negative) and their step: the
/// largest over 63.
fn codes_by_max(x: &[f32; RUNS]) -> (f32, [u8; RUNS]) {
    let top = x.iter().fold(0.0f32, |m, &s| m.max(s));
    let inv = if top > 0.0 { 63.0 / top } else { 0.0 };
    (
        top / 63.0,
        x.map(|v| ((inv * v).round_ties_even() as u8).min(63)),
    )
}

/// The eight runs' 6-bit codes of `x` (non-negative) and their step, by
/// GGML's weighted fit: nine steps near the largest over 63, then codes
/// moved one at a time while the error weighted by `w` falls, the step by
/// least squares.
fn codes_by_fit(x: &[f32; RUNS], w: &[f32; RUNS]) -> (f32, [u8; RUNS]) {
    const TOP: f32 = 63.0;
    let max = x.iter().fold(0.0f32, |m, &s| m.max(s));
    if max < 1e-15 {
        return (0.0, [0; RUNS]);
    }
    let code = |iscale: f32, v: f32| (iscale * v).round_ties_even().min(TOP);
    let err_at = |iscale: f32| -> f32 {
        let scale = 1.0 / iscale;
        x.iter()
            .zip(w)
            .map(|(&v, &wi)| {
                let d = v - scale * code(iscale, v);
                wi * d * d
            })
            .sum()
    };
    let mut iscale = TOP / max;
    let mut best = err_at(iscale);
    for is in -4i32..=4 {
        if is == 0 {
            continue;
        }
        let trial = (0.1 * is as f32 + TOP) / max;
        let err = err_at(trial);
        if err < best {
            best = err;
            iscale = trial;
        }
    }
    let mut codes = x.map(|v| code(iscale, v) as u8);
    let (mut sum_lx, mut sum_l2) = (0.0f32, 0.0f32);
    for ((&v, &wi), &l) in x.iter().zip(w).zip(&codes) {
        let l = f32::from(l);
        sum_lx += wi * v * l;
        sum_l2 += wi * l * l;
    }
    for _ in 0..5 {
        let mut changed = false;
        for i in 0..RUNS {
            let (v, wi, l) = (x[i], w[i], f32::from(codes[i]));
            let mut slx = sum_lx - wi * v * l;
            let mut sl2 = sum_l2 - wi * l * l;
            if slx > 0.0 && sl2 > 0.0 {
                let new = (v * sl2 / slx).round_ties_even().min(TOP);
                if new != l {
                    slx += wi * v * new;
                    sl2 += wi * new * new;
                    if slx * slx * sum_l2 > sum_lx * sum_lx * sl2 {
                        codes[i] = new as u8;
                        sum_lx = slx;
                        sum_l2 = sl2;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    let step = if sum_l2 > 0.0 { sum_lx / sum_l2 } else { 0.0 };
    (step, codes)
}

/// The weights of each value's error in block `x`, by `importance` (as
/// long) where given.
fn error_weights(x: &[f32], importance: Option<&[f32]>) -> [f32; BLOCK] {
    let mut w = [0.0f32; BLOCK];
    match importance {
        None => {
            for (wr, xr) in w.chunks_exact_mut(RUN).zip(x.chunks_exact(RUN)) {
                let rms = (xr.iter().map(|v| v * v).sum::<f32>() / RUN as f32).sqrt();
                for (wi, v) in wr.iter_mut().zip(xr) {
                    *wi = rms + v.abs();
                }
            }
        }
        Some(imp) => {
            let sigma2 = 2.0 * x.iter().map(|v| v * v).sum::<f32>() / BLOCK as f32;
            for ((wi, v), m) in w.iter_mut().zip(x).zip(imp) {
                *wi = m * (sigma2 + v * v).sqrt();
            }
        }
    }
    w
}

/// Block `x` fitted with levels of `bits`.
pub fn fit_block(x: &[f32], importance: Option<&[f32]>, bits: Bits, refine: bool) -> KBlock {
    debug_assert_eq!(x.len(), BLOCK);
    let top = bits.top();
    let w = error_weights(x, importance);
    let search = match (importance, bits) {
        (None, Bits::Four) => (-1.0, 0.1, 20),
        (None, Bits::Five) => (-0.5, 0.1, 15),
        (Some(_), _) => (-0.9, 0.05, 36),
    };
    let mut kb = KBlock {
        d: 0,
        dmin: 0,
        scales: [0; RUNS],
        mins: [0; RUNS],
        levels: [0; BLOCK],
    };
    let mut scales = [0.0f32; RUNS];
    let mut mins = [0.0f32; RUNS];
    let mut run_w = [0.0f32; RUNS];
    for j in 0..RUNS {
        let at = j * RUN..(j + 1) * RUN;
        run_w[j] = w[at.clone()].iter().sum();
        (scales[j], mins[j]) = fit_run(
            &x[at.clone()],
            &w[at.clone()],
            top,
            search,
            &mut kb.levels[at],
        );
    }
    let ((d, sc), (dmin, m)) = if importance.is_some() {
        (codes_by_fit(&scales, &run_w), codes_by_fit(&mins, &run_w))
    } else {
        (codes_by_max(&scales), codes_by_max(&mins))
    };
    (kb.d, kb.dmin, kb.scales, kb.mins) = (f32_to_f16(d), f32_to_f16(dmin), sc, m);
    let err = kb.fill(x, &w, top);
    if refine {
        refine_block(&mut kb, x, &w, top, err);
    }
    kb
}

/// Lowers `kb`'s weighted error (`err` now): each run's codes searched
/// within 2 of the present ones, then `d` and `dmin` fitted to the levels
/// by least squares, for as long as either helps.
fn refine_block(kb: &mut KBlock, x: &[f32], w: &[f32], top: u8, mut err: f32) {
    for _ in 0..4 {
        let before = err;
        for j in 0..RUNS {
            let (s0, m0) = (kb.scales[j], kb.mins[j]);
            let mut best = (kb.fill_run(j, x, w, top), s0, m0);
            for s in s0.saturating_sub(2)..=(s0 + 2).min(63) {
                for m in m0.saturating_sub(2)..=(m0 + 2).min(63) {
                    (kb.scales[j], kb.mins[j]) = (s, m);
                    let e = kb.fill_run(j, x, w, top);
                    if e < best.0 {
                        best = (e, s, m);
                    }
                }
            }
            (kb.scales[j], kb.mins[j]) = (best.1, best.2);
            kb.fill_run(j, x, w, top);
        }
        err = kb.fill(x, w, top);
        // x ≈ d·a − dmin·b, a the code times the level, b the minimum's code.
        let (mut aa, mut ab, mut bb, mut ax, mut bx) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for i in 0..BLOCK {
            let j = i / RUN;
            let a = f64::from(kb.scales[j]) * f64::from(kb.levels[i]);
            let b = f64::from(kb.mins[j]);
            let (wi, xi) = (f64::from(w[i]), f64::from(x[i]));
            aa += wi * a * a;
            ab += wi * a * b;
            bb += wi * b * b;
            ax += wi * a * xi;
            bx += wi * b * xi;
        }
        let det = aa * bb - ab * ab;
        if det > 0.0 {
            let d = (ax * bb - bx * ab) / det;
            let dmin = (ax * ab - bx * aa) / det;
            let old = (kb.d, kb.dmin, kb.levels);
            (kb.d, kb.dmin) = (f32_to_f16(d as f32), f32_to_f16(dmin as f32));
            let e = kb.fill(x, w, top);
            if e < err {
                err = e;
            } else {
                (kb.d, kb.dmin, kb.levels) = old;
            }
        }
        if err >= before * (1.0 - 1e-6) {
            break;
        }
    }
}

/// `kb`'s halves and codes into a block ([`D`], [`DMIN`], [`SCALES`]).
fn write_head(kb: &KBlock, out: &mut [u8]) {
    out[D].copy_from_slice(&kb.d.to_le_bytes());
    out[DMIN].copy_from_slice(&kb.dmin.to_le_bytes());
    let packed = &mut out[SCALES];
    packed.fill(0);
    for j in 0..RUNS {
        let (ls, lm) = (kb.scales[j], kb.mins[j]);
        if j < 4 {
            packed[j] = ls;
            packed[j + 4] = lm;
        } else {
            packed[j + 4] = (ls & 0xf) | ((lm & 0xf) << 4);
            packed[j - 4] |= (ls >> 4) << 6;
            packed[j] |= (lm >> 4) << 6;
        }
    }
}

/// The low nibbles of `levels` in Q4_K's order.
fn write_nibbles(levels: &[u8; BLOCK], qs: &mut [u8]) {
    for (c, chunk) in qs.chunks_exact_mut(32).enumerate() {
        for (l, b) in chunk.iter_mut().enumerate() {
            *b = (levels[c * 64 + l] & 0xf) | ((levels[c * 64 + 32 + l] & 0xf) << 4);
        }
    }
}

/// `kb` packed as a block of `bits`.
pub fn write_block(kb: &KBlock, bits: Bits, out: &mut [u8]) {
    write_head(kb, out);
    if bits == Bits::Five {
        let qh = &mut out[FIFTH_BITS];
        qh.fill(0);
        for (i, &l) in kb.levels.iter().enumerate() {
            let (at, bit) = fifth_bit(i);
            qh[at] |= (l >> 4) << bit;
        }
    }
    write_nibbles(&kb.levels, &mut out[bits.nibbles()]);
}

/// `values` (rows of `cols`, a multiple of [`BLOCK`]) packed as K blocks
/// of `bits` by `fit` (its importance one a column of the rows, where
/// given).
pub fn quantize(values: &[f32], cols: usize, bits: Bits, fit: Fit) -> Vec<u8> {
    assert_eq!(cols % BLOCK, 0, "a row of {cols} is not whole blocks");
    let bytes = bits.block_bytes();
    let blocks = values.len() / BLOCK;
    let mut out = vec![0u8; blocks * bytes];
    for (b, chunk) in out.chunks_exact_mut(bytes).enumerate() {
        let at = (b * BLOCK) % cols;
        let importance = fit.importance.map(|i| &i[at..at + BLOCK]);
        let kb = fit_block(
            &values[b * BLOCK..(b + 1) * BLOCK],
            importance,
            bits,
            fit.refine,
        );
        write_block(&kb, bits, chunk);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use jodel_rhythm::nn::quant::dequantize;

    fn smooth() -> Vec<f32> {
        (0..BLOCK * 4)
            .map(|i| {
                ((i as f32 * 0.37).sin() * 0.05 + (i % 7) as f32 * 0.003)
                    * if i < 512 { 1.0 } else { 4.0 }
            })
            .collect()
    }

    /// The largest error of each run against its range over `levels`.
    fn worst_step(x: &[f32], back: &[f32], levels: f32) -> f32 {
        x.chunks(RUN)
            .zip(back.chunks(RUN))
            .map(|(a, b)| {
                let range = a.iter().fold(0.0f32, |m, v| m.max(v.abs())) * 2.0;
                let worst = a
                    .iter()
                    .zip(b)
                    .fold(0.0f32, |m, (p, q)| m.max((p - q).abs()));
                worst / (range / levels)
            })
            .fold(0.0f32, f32::max)
    }

    /// Smooth values come back within a step of their run's grid, by
    /// every fit and either width.
    #[test]
    fn blocks_come_back_close() {
        let x = smooth();
        let imp: Vec<f32> = (0..512).map(|i| 1.0 + (i % 5) as f32).collect();
        for bits in [Bits::Four, Bits::Five] {
            for importance in [None, Some(imp.as_slice())] {
                for refine in [false, true] {
                    let fit = Fit { importance, refine };
                    let packed = quantize(&x, 512, bits, fit);
                    assert_eq!(packed.len(), 4 * bits.block_bytes());
                    let back = dequantize(&packed, bits);
                    let levels = f32::from(bits.top());
                    let worst = worst_step(&x, &back, levels);
                    assert!(worst <= 0.75, "{bits:?} {fit:?}: {worst}");
                }
            }
        }
    }

    /// Five bits come closer than four; refining never adds error.
    #[test]
    fn more_bits_and_refining_help() {
        let x = smooth();
        let sq = |bits, refine| -> f32 {
            let fit = Fit {
                importance: None,
                refine,
            };
            let back = dequantize(&quantize(&x, 512, bits, fit), bits);
            x.iter().zip(&back).map(|(a, b)| (a - b) * (a - b)).sum()
        };
        assert!(sq(Bits::Five, false) < sq(Bits::Four, false) / 2.0);
        assert!(sq(Bits::Four, true) <= sq(Bits::Four, false));
    }

    /// Zeros stay zeros, and a constant run is exact enough.
    #[test]
    fn plain_blocks() {
        for bits in [Bits::Four, Bits::Five] {
            let fit = Fit::default();
            let back = dequantize(&quantize(&[0.0; BLOCK], BLOCK, bits, fit), bits);
            assert!(back.iter().all(|&v| v == 0.0));
            let back = dequantize(&quantize(&[0.25; BLOCK], BLOCK, bits, fit), bits);
            assert!(
                back.iter().all(|&v| (v - 0.25).abs() < 1e-3),
                "{:?}",
                &back[..4]
            );
        }
    }
}
