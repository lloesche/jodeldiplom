//! Weights at fewer bits: three block layouts of the GGML file format.
//!
//! - Q4_K, about 4.5 bits a value. A row of a matrix (its length a
//!   multiple of [`BLOCK`]) is blocks of 256 values, each 144 bytes:
//!   `d` and `dmin`, IEEE halves (4 bytes); eight 6-bit scales and eight
//!   6-bit minimums, one each for the block's eight runs of 32, packed in
//!   12 bytes; 256 nibbles (128 bytes): for each 64 values, 32 bytes, the
//!   first 32 values in the low nibbles, the next 32 in the high. A value
//!   is `d·scale·q − dmin·min`, `q` its nibble.
//! - Q5_K, 5.5 bits a value: as Q4_K, with a fifth bit to each level, so
//!   `q` is 0 to 31. A block is 176 bytes: `d`, `dmin` and the scales as
//!   in Q4_K (16 bytes), then the fifth bits (32 bytes: bit `2c` of byte
//!   `l` for value `64c + l`, bit `2c + 1` for value `64c + 32 + l`), then
//!   the low nibbles as in Q4_K (128 bytes).
//! - Q8_0, 8.5 bits a value: a row (a multiple of [`Q8_BLOCK`]) is blocks
//!   of 32 values, each a half `d` and 32 signed bytes `q`, a value `d·q`.
//!
//! [`quantize`] fits a K block (Q4_K or Q5_K) by [`Fit`]: without
//! importance, as GGML's plain reference does (each run's scale and
//! minimum by a search over 21 grids, a value weighted by its run's RMS
//! plus its magnitude, then the block's 6-bit scales from the largest);
//! with it, as GGML does with an importance matrix (a value weighted by
//! its column's importance times `√(σ² + x²)`, `σ²` twice the block's mean
//! square, a search over 37 grids, the 6-bit scales fitted to the least
//! error weighted by their runs' weights). Refining then searches each
//! run's 6-bit scale and minimum near the fitted ones, and fits `d` and
//! `dmin` to the levels by least squares, while the weighted error falls.
//!
//! [`quantize_gptq`] packs K blocks by GPTQ instead: given the mean
//! product of each two inputs of a matrix, it rounds a row a column at a
//! time and moves each rounding error onto the columns not yet rounded,
//! so the row's output on those inputs changes least.

use super::weights::f16_to_f32;

/// Values a K block holds.
pub const BLOCK: usize = 256;
/// Bytes a Q4_K block takes.
pub const BLOCK_BYTES: usize = 144;
/// Bytes a Q5_K block takes.
pub const Q5_BLOCK_BYTES: usize = 176;
/// Values a Q8_0 block holds.
pub const Q8_BLOCK: usize = 32;
/// Bytes a Q8_0 block takes.
pub const Q8_BLOCK_BYTES: usize = 34;
/// Values a run (one scale and minimum) holds.
const RUN: usize = 32;
/// Runs a block holds.
const RUNS: usize = BLOCK / RUN;

/// Single precision to half, rounding to the nearest (ties to even).
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // Subnormal (or zero): the mantissa with its leading one, shifted.
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1 << shift) - 1);
        let mut v = m >> shift;
        if rest > half || (rest == half && v & 1 == 1) {
            v += 1;
        }
        return sign | v as u16;
    }
    let mut v = ((e as u32) << 10) | (mant >> 13);
    let rest = mant & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && v & 1 == 1) {
        v += 1; // may carry into the exponent, as it should
    }
    sign | v as u16
}

/// How [`quantize`] fits a K block.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fit<'a> {
    /// How much each column's error counts, one a column of the rows (the
    /// mean square of what the column multiplies, say).
    pub importance: Option<&'a [f32]>,
    /// Whether to refine the fitted scales.
    pub refine: bool,
}

/// The bits of a level of a K block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bits {
    /// Q4_K.
    Four,
    /// Q5_K.
    Five,
}

impl Bits {
    fn top(self) -> u8 {
        match self {
            Bits::Four => 15,
            Bits::Five => 31,
        }
    }

    /// Bytes a block takes.
    pub fn block_bytes(self) -> usize {
        match self {
            Bits::Four => BLOCK_BYTES,
            Bits::Five => Q5_BLOCK_BYTES,
        }
    }
}

/// A K block before packing: the halves, each run's 6-bit scale and
/// minimum, each value's level.
struct KBlock {
    d: u16,
    dmin: u16,
    scales: [u8; RUNS],
    mins: [u8; RUNS],
    levels: [u8; BLOCK],
}

impl KBlock {
    /// Run `j`'s step and the amount below zero.
    fn run(&self, j: usize) -> (f32, f32) {
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
fn fit_block(x: &[f32], importance: Option<&[f32]>, bits: Bits, refine: bool) -> KBlock {
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

/// The scale and minimum of run `j` from the 12 packed bytes.
fn scale_min(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// `kb`'s halves and codes into the first 16 bytes of a block.
fn write_head(kb: &KBlock, out: &mut [u8]) {
    out[0..2].copy_from_slice(&kb.d.to_le_bytes());
    out[2..4].copy_from_slice(&kb.dmin.to_le_bytes());
    let packed = &mut out[4..16];
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

/// One block of 256 values, packed with levels of `bits` by `fit`
/// (its importance as long as the block, where given).
pub fn quantize_block(x: &[f32], bits: Bits, fit: Fit, out: &mut [u8]) {
    write_block(&fit_block(x, fit.importance, bits, fit.refine), bits, out);
}

/// `kb` packed as a block of `bits`.
fn write_block(kb: &KBlock, bits: Bits, out: &mut [u8]) {
    write_head(kb, out);
    match bits {
        Bits::Four => write_nibbles(&kb.levels, &mut out[16..BLOCK_BYTES]),
        Bits::Five => {
            let qh = &mut out[16..48];
            qh.fill(0);
            for (i, &l) in kb.levels.iter().enumerate() {
                let (c, at, half) = (i / 64, i % 32, (i % 64) / 32);
                qh[at] |= (l >> 4) << (2 * c + half);
            }
            write_nibbles(&kb.levels, &mut out[48..Q5_BLOCK_BYTES]);
        }
    }
}

/// One block's 256 values from its bytes, its levels of `bits`.
pub fn dequantize_block(block: &[u8], bits: Bits, out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let packed = &block[4..16];
    let (qh, qs) = match bits {
        Bits::Four => (None, &block[16..BLOCK_BYTES]),
        Bits::Five => (Some(&block[16..48]), &block[48..Q5_BLOCK_BYTES]),
    };
    let high = |i: usize| -> u8 {
        qh.map_or(0, |h| {
            let (c, at, half) = (i / 64, i % 32, (i % 64) / 32);
            ((h[at] >> (2 * c + half)) & 1) << 4
        })
    };
    for (c, chunk) in qs.chunks_exact(32).enumerate() {
        let (s1, m1) = scale_min(2 * c, packed);
        let (s2, m2) = scale_min(2 * c + 1, packed);
        let (d1, b1) = (d * f32::from(s1), dmin * f32::from(m1));
        let (d2, b2) = (d * f32::from(s2), dmin * f32::from(m2));
        for (l, &q) in chunk.iter().enumerate() {
            let (i, k) = (c * 64 + l, c * 64 + 32 + l);
            out[i] = d1 * f32::from((q & 0xf) | high(i)) - b1;
            out[k] = d2 * f32::from((q >> 4) | high(k)) - b2;
        }
    }
}

/// `values` (rows of `cols`, a multiple of [`BLOCK`]) packed as K blocks
/// of `bits` by `fit`.
pub fn quantize(values: &[f32], cols: usize, bits: Bits, fit: Fit) -> Vec<u8> {
    assert_eq!(cols % BLOCK, 0, "a row of {cols} is not whole blocks");
    let bytes = bits.block_bytes();
    let blocks = values.len() / BLOCK;
    let mut out = vec![0u8; blocks * bytes];
    for (b, chunk) in out.chunks_exact_mut(bytes).enumerate() {
        let at = (b * BLOCK) % cols;
        let fit = Fit {
            importance: fit.importance.map(|i| &i[at..at + BLOCK]),
            ..fit
        };
        quantize_block(&values[b * BLOCK..(b + 1) * BLOCK], bits, fit, chunk);
    }
    out
}

/// The values of packed K blocks of `bits`.
pub fn dequantize(packed: &[u8], bits: Bits) -> Vec<f32> {
    let bytes = bits.block_bytes();
    let mut out = vec![0.0f32; packed.len() / bytes * BLOCK];
    for (block, o) in packed.chunks_exact(bytes).zip(out.chunks_exact_mut(BLOCK)) {
        dequantize_block(block, bits, o);
    }
    out
}

/// The lower Cholesky factor of `a` (`[n][n]`, symmetric, positive
/// definite), in place of its lower triangle; the upper triangle zeroed.
fn cholesky(a: &mut [f64], n: usize) -> Result<(), String> {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if d <= 0.0 {
            return Err(format!("not positive definite at {j}"));
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / d;
        }
        for k in j + 1..n {
            a[j * n + k] = 0.0;
        }
    }
    Ok(())
}

/// The upper Cholesky factor `U` of the inverse of `h` (`[n][n]`), so
/// `h⁻¹ = Uᵀ·U`: `h = L·Lᵀ`, `h⁻¹ = L⁻ᵀ·L⁻¹`, and that factored again.
fn inverse_factor(h: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let mut l = h.to_vec();
    cholesky(&mut l, n)?;
    // L⁻¹, lower, a row at a time: row i is (eᵢ − Σₖ L[i][k]·row k) / L[i][i].
    let mut inv = vec![0.0f64; n * n];
    for i in 0..n {
        let (done, row) = inv.split_at_mut(i * n);
        let row = &mut row[..n];
        row[i] = 1.0;
        for k in 0..i {
            let lik = l[i * n + k];
            for (r, &v) in row[..=k].iter_mut().zip(&done[k * n..k * n + k + 1]) {
                *r -= lik * v;
            }
        }
        let d = l[i * n + i];
        for r in &mut row[..=i] {
            *r /= d;
        }
    }
    // h⁻¹ = L⁻ᵀ·L⁻¹, the sum over rows k of L⁻¹ of each row's outer square.
    let mut hinv = vec![0.0f64; n * n];
    for k in 0..n {
        let row = &inv[k * n..k * n + k + 1];
        for (i, &a) in row.iter().enumerate() {
            for (h, &b) in hinv[i * n..i * n + k + 1].iter_mut().zip(row) {
                *h += a * b;
            }
        }
    }
    cholesky(&mut hinv, n)?;
    // The transpose of the lower factor.
    let mut u = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            u[j * n + i] = hinv[i * n + j];
        }
    }
    Ok(u)
}

/// `values` (rows of `cols`, a multiple of [`BLOCK`]) packed as K blocks
/// of `bits` by GPTQ (Frantar et al., 2022): a row a column at a time,
/// each column's rounding error spread over the columns after it so the
/// row's output on its inputs changes least; `gram` the mean product of
/// each two columns' inputs (`[cols][cols]`, damped by a hundredth of its
/// mean diagonal). Each block is fitted by `fit` (by the gram's diagonal
/// where it has no importance) to the row as it stands when the block's
/// first column comes.
pub fn quantize_gptq(
    values: &[f32],
    cols: usize,
    bits: Bits,
    gram: &[f32],
    fit: Fit,
) -> Result<Vec<u8>, String> {
    assert_eq!(cols % BLOCK, 0, "a row of {cols} is not whole blocks");
    assert_eq!(gram.len(), cols * cols, "a gram of {cols} by {cols}");
    let diag: Vec<f32> = (0..cols).map(|i| gram[i * cols + i]).collect();
    let mean = diag.iter().map(|&d| f64::from(d)).sum::<f64>() / cols as f64;
    let mut h: Vec<f64> = gram.iter().map(|&g| f64::from(g)).collect();
    for (i, &d) in diag.iter().enumerate() {
        if d <= 0.0 {
            // A column that never sees anything: kept apart from the rest.
            for j in 0..cols {
                h[i * cols + j] = 0.0;
                h[j * cols + i] = 0.0;
            }
            h[i * cols + i] = 1.0;
        }
        h[i * cols + i] += 0.01 * mean;
    }
    let u = inverse_factor(&h, cols)?;
    let importance = fit.importance.unwrap_or(&diag);
    let bytes = bits.block_bytes();
    let per_row = cols / BLOCK * bytes;
    let rows = values.len() / cols;
    let mut out = vec![0u8; rows * per_row];
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = rows.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (c, part) in out.chunks_mut(chunk * per_row).enumerate() {
            let u = &u;
            s.spawn(move || {
                for (r, packed) in part.chunks_exact_mut(per_row).enumerate() {
                    let row = c * chunk + r;
                    let x = &values[row * cols..(row + 1) * cols];
                    gptq_row(x, bits, importance, fit.refine, u, packed);
                }
            });
        }
    });
    Ok(out)
}

/// One row by GPTQ ([`quantize_gptq`]), `u` the upper Cholesky factor of
/// the inverse damped gram.
fn gptq_row(x: &[f32], bits: Bits, importance: &[f32], refine: bool, u: &[f64], out: &mut [u8]) {
    let cols = x.len();
    let top = f32::from(bits.top());
    let mut w: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    let bytes = bits.block_bytes();
    for (b, block) in out.chunks_exact_mut(bytes).enumerate() {
        let at = b * BLOCK;
        let now: Vec<f32> = w[at..at + BLOCK].iter().map(|&v| v as f32).collect();
        let mut kb = fit_block(&now, Some(&importance[at..at + BLOCK]), bits, refine);
        for i in at..at + BLOCK {
            let (step, below) = kb.run((i - at) / RUN);
            let q = if step == 0.0 {
                0.0
            } else {
                ((w[i] as f32 + below) / step)
                    .round_ties_even()
                    .clamp(0.0, top)
            };
            kb.levels[i - at] = q as u8;
            let err = (w[i] - f64::from(step * q - below)) / u[i * cols + i];
            let ui = &u[i * cols..(i + 1) * cols];
            for (wk, &uk) in w[i + 1..].iter_mut().zip(&ui[i + 1..]) {
                *wk -= err * uk;
            }
        }
        write_block(&kb, bits, block);
    }
}

/// `values` (a multiple of [`Q8_BLOCK`]) packed as Q8_0 blocks: each
/// block's step its largest magnitude over 127.
pub fn quantize_q8(values: &[f32]) -> Vec<u8> {
    assert_eq!(values.len() % Q8_BLOCK, 0, "not whole Q8_0 blocks");
    let mut out = Vec::with_capacity(values.len() / Q8_BLOCK * Q8_BLOCK_BYTES);
    for x in values.chunks_exact(Q8_BLOCK) {
        let top = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = f32_to_f16(top / 127.0);
        let step = f16_to_f32(d);
        out.extend(d.to_le_bytes());
        for &v in x {
            let q = if step > 0.0 {
                (v / step).round_ties_even().clamp(-127.0, 127.0)
            } else {
                0.0
            };
            out.push(q as i8 as u8);
        }
    }
    out
}

/// The values of packed Q8_0 blocks.
pub fn dequantize_q8(packed: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(packed.len() / Q8_BLOCK_BYTES * Q8_BLOCK);
    for block in packed.chunks_exact(Q8_BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        out.extend(block[2..].iter().map(|&q| d * f32::from(q as i8)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_round_to_nearest() {
        for x in [
            0.0f32, 1.0, -2.0, 0.333_333, 65504.0, 6.0e-8, 1.0e-5, -2.375,
        ] {
            let back = f16_to_f32(f32_to_f16(x));
            assert!(
                (back - x).abs() <= x.abs() / 1024.0 + 6.0e-8,
                "{x} → {back}"
            );
        }
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(1.0e6), 0x7c00);
    }

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

    /// A small positive definite matrix: `Uᵀ·U` is its inverse.
    #[test]
    fn the_inverse_factor_inverts() {
        let n = 5;
        let mut h = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                h[i * n + j] =
                    1.0 / (1.0 + (i as f64 - j as f64).abs()) + if i == j { 1.0 } else { 0.0 };
            }
        }
        let u = inverse_factor(&h, n).unwrap();
        for i in 0..n {
            for j in 0..i {
                assert_eq!(u[i * n + j], 0.0);
            }
        }
        for i in 0..n {
            for j in 0..n {
                // (Uᵀ·U·h)[i][j]
                let s: f64 = (0..n)
                    .map(|k| {
                        let utu: f64 = (0..n).map(|m| u[m * n + i] * u[m * n + k]).sum();
                        utu * h[k * n + j]
                    })
                    .sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((s - want).abs() < 1e-9, "({i}, {j}): {s}");
            }
        }
    }

    /// On inputs whose columns move together, GPTQ changes a row's output
    /// less than rounding each value alone.
    #[test]
    fn gptq_keeps_the_output_closer() {
        let cols = BLOCK;
        // Inputs: a shared signal plus each column's own.
        let inputs: Vec<Vec<f32>> = (0..400)
            .map(|t| {
                let common = (t as f32 * 0.7).sin();
                (0..cols)
                    .map(|c| common + 0.3 * ((t * 31 + c * 17) as f32 * 0.37).sin())
                    .collect()
            })
            .collect();
        let mut gram = vec![0.0f32; cols * cols];
        for x in &inputs {
            for i in 0..cols {
                for j in 0..cols {
                    gram[i * cols + j] += x[i] * x[j] / inputs.len() as f32;
                }
            }
        }
        let values: Vec<f32> = (0..2 * cols)
            .map(|i| ((i as f32) * 0.123).sin() * 0.1)
            .collect();
        let output_error = |back: &[f32]| -> f64 {
            let mut sum = 0.0f64;
            for (w, q) in values.chunks(cols).zip(back.chunks(cols)) {
                for x in &inputs {
                    let d: f32 = x
                        .iter()
                        .zip(w.iter().zip(q))
                        .map(|(xi, (a, b))| xi * (a - b))
                        .sum();
                    sum += f64::from(d * d);
                }
            }
            sum
        };
        let diag: Vec<f32> = (0..cols).map(|i| gram[i * cols + i]).collect();
        let fit = Fit {
            importance: Some(&diag),
            refine: false,
        };
        let plain = dequantize(&quantize(&values, cols, Bits::Four, fit), Bits::Four);
        let packed = quantize_gptq(&values, cols, Bits::Four, &gram, fit).unwrap();
        assert_eq!(packed.len(), 2 * BLOCK_BYTES);
        let gptq = dequantize(&packed, Bits::Four);
        let (p, g) = (output_error(&plain), output_error(&gptq));
        assert!(g < p * 0.8, "GPTQ {g} against {p}");
    }

    /// Q8_0 keeps each value within half a step of its block's.
    #[test]
    fn eight_bit_blocks() {
        let x = smooth();
        let packed = quantize_q8(&x);
        assert_eq!(packed.len(), x.len() / Q8_BLOCK * Q8_BLOCK_BYTES);
        let back = dequantize_q8(&packed);
        for (a, b) in x.chunks(Q8_BLOCK).zip(back.chunks(Q8_BLOCK)) {
            let step = a.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 127.0;
            for (p, q) in a.iter().zip(b) {
                assert!((p - q).abs() <= step * 0.51 + 1e-6, "{p} → {q}");
            }
        }
        assert!(dequantize_q8(&quantize_q8(&[0.0; 32]))
            .iter()
            .all(|&v| v == 0.0));
    }
}
