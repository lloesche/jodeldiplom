//! Weights at 4 bits: each row's values (an output's inputs) in groups of
//! [`GROUP`], each group as [`GROUP_WORDS`] u32s:
//!
//! - four of nibbles, value `j` of the group in word `j / 8`, bits
//!   `4·(j % 8)` up;
//! - one with the group's scale (low half) and offset (high half), each an
//!   IEEE half.
//!
//! A value is `offset + nibble · scale`. Symmetric rounding (the offset
//! −8 scales) and rounding to the group's range (the offset its minimum)
//! both fit; a GPU's kernel reads them the same way.

use jodel_tensors::f16_to_f32;

/// Values a group.
pub const GROUP: usize = 32;
/// u32s a group.
pub const GROUP_WORDS: usize = 5;

/// f32 to the nearest IEEE half (ties to even), as its bits.
pub fn f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if frac != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // Subnormal or zero.
        if e < -10 {
            return sign;
        }
        let m = frac | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1u32 << shift) - 1);
        let mut v = m >> shift;
        if rest > half || (rest == half && v & 1 == 1) {
            v += 1;
        }
        return sign | v as u16;
    }
    let mut v = ((e as u32) << 10) | (frac >> 13);
    let rest = frac & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && v & 1 == 1) {
        v += 1;
    }
    sign | v as u16
}

/// `values` (`rows × cols`, `cols` a multiple of [`GROUP`]) at 4 bits:
/// `rows × cols / GROUP` groups.
pub fn quantize(values: &[f32], rows: usize, cols: usize, symmetric: bool) -> Vec<u32> {
    assert_eq!(values.len(), rows * cols);
    assert_eq!(cols % GROUP, 0, "{cols} inputs are not whole groups");
    quantize_with(values, cols, symmetric, None)
}

/// [`quantize`] with each input column's rounding weighted by how much
/// signal it carries (`importance`, `cols` long: its mean square over
/// calibration audio, [`Statistics::mean_squares`]): the scale and offset
/// of a group fitted to least weighted squared error, so the columns that
/// matter most are rounded best.
pub fn quantize_weighted(values: &[f32], rows: usize, cols: usize, importance: &[f32]) -> Vec<u32> {
    assert_eq!(values.len(), rows * cols);
    assert_eq!(importance.len(), cols, "an importance a column");
    quantize_with(values, cols, false, Some(importance))
}

/// The groups of `values` shared out over the processor's threads.
fn quantize_with(
    values: &[f32],
    cols: usize,
    symmetric: bool,
    importance: Option<&[f32]>,
) -> Vec<u32> {
    let groups = values.len() / GROUP;
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(groups.div_ceil(4096))
        .max(1);
    let per = groups.div_ceil(threads) * GROUP;
    let parts: Vec<Vec<u32>> = std::thread::scope(|s| {
        let handles: Vec<_> = values
            .chunks(per)
            .enumerate()
            .map(|(i, part)| {
                s.spawn(move || {
                    quantize_groups(part, (i * per) % cols, cols, symmetric, importance)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    parts.concat()
}

/// Whole groups of `values` at 4 bits, the first at column `col` of rows
/// `cols` wide.
fn quantize_groups(
    values: &[f32],
    mut col: usize,
    cols: usize,
    symmetric: bool,
    importance: Option<&[f32]>,
) -> Vec<u32> {
    let mut out = Vec::with_capacity(values.len() / GROUP * GROUP_WORDS);
    let ones = [1.0f32; GROUP];
    for g in values.chunks_exact(GROUP) {
        let w = importance.map_or(&ones[..], |imp| &imp[col..col + GROUP]);
        col = (col + GROUP) % cols;
        let (scale, offset) = if symmetric {
            let peak = g.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let scale = f16_to_f32(f16_bits(peak / 7.0));
            (scale, -8.0 * scale)
        } else {
            fit(g, w)
        };
        let mut words = [0u32; GROUP_WORDS];
        for (j, &v) in g.iter().enumerate() {
            let q = if scale > 0.0 {
                ((v - offset) / scale).round().clamp(0.0, 15.0) as u32
            } else {
                0
            };
            words[j / 8] |= q << (4 * (j % 8));
        }
        words[4] = u32::from(f16_bits(scale)) | (u32::from(f16_bits(offset)) << 16);
        out.extend_from_slice(&words);
    }
    out
}

/// The squared error of `g` held as `offset + nibble · scale`, each value's
/// weighted by `w`.
fn error_of(g: &[f32], w: &[f32], scale: f32, offset: f32) -> f64 {
    g.iter()
        .zip(w)
        .map(|(&v, &w)| {
            let q = if scale > 0.0 {
                ((v - offset) / scale).round().clamp(0.0, 15.0)
            } else {
                0.0
            };
            f64::from(w) * f64::from(offset + q * scale - v).powi(2)
        })
        .sum()
}

/// A group's scale and offset (as halves) with the least squared error,
/// each value's weighted by `w`: the range's, or for each of a spread of
/// scales around it, the nibbles it gives and the scale and offset fitted
/// to them by weighted least squares.
fn fit(g: &[f32], w: &[f32]) -> (f32, f32) {
    let half = |x: f32| f16_to_f32(f16_bits(x));
    let lo = g.iter().copied().fold(f32::MAX, f32::min);
    let hi = g.iter().copied().fold(f32::MIN, f32::max);
    let offset = half(lo);
    let mut best = (half((hi - offset) / 15.0), offset);
    if hi <= lo {
        return best;
    }
    let mut least = error_of(g, w, best.0, best.1);
    let n: f64 = w.iter().map(|&w| f64::from(w)).sum();
    for step in 0..=20 {
        let inverse = (14.0 + 0.1 * step as f32) / (hi - lo);
        let (mut sq, mut sqq, mut sx, mut sqx) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (&v, &w) in g.iter().zip(w) {
            let (q, w) = (
                f64::from(((v - lo) * inverse).round().clamp(0.0, 15.0)),
                f64::from(w),
            );
            sq += w * q;
            sqq += w * q * q;
            sx += w * f64::from(v);
            sqx += w * q * f64::from(v);
        }
        let d = n * sqq - sq * sq;
        if d <= 0.0 {
            continue;
        }
        let scale = half(((n * sqx - sq * sx) / d) as f32);
        let offset = half(((sqq * sx - sq * sqx) / d) as f32);
        if scale <= 0.0 {
            continue;
        }
        let e = error_of(g, w, scale, offset);
        if e < least {
            least = e;
            best = (scale, offset);
        }
    }
    best
}

/// u32s a group at 8 bits: eight of bytes (value `j` in word `j / 4`, bits
/// `8·(j % 4)` up), then the scale and offset as at 4 bits.
pub const GROUP_WORDS_8: usize = 9;

/// `values` (`rows × cols`) at 8 bits, each group's scale and offset from
/// its range: `rows × cols / GROUP` groups of [`GROUP_WORDS_8`].
pub fn quantize8(values: &[f32], rows: usize, cols: usize) -> Vec<u32> {
    assert_eq!(values.len(), rows * cols);
    assert_eq!(cols % GROUP, 0, "{cols} inputs are not whole groups");
    let half = |x: f32| f16_to_f32(f16_bits(x));
    let mut out = Vec::with_capacity(rows * cols / GROUP * GROUP_WORDS_8);
    for g in values.chunks_exact(GROUP) {
        let lo = g.iter().copied().fold(f32::MAX, f32::min);
        let hi = g.iter().copied().fold(f32::MIN, f32::max);
        let offset = half(lo);
        let scale = half((hi - offset) / 255.0);
        let mut words = [0u32; GROUP_WORDS_8];
        for (j, &v) in g.iter().enumerate() {
            let q = if scale > 0.0 {
                ((v - offset) / scale).round().clamp(0.0, 255.0) as u32
            } else {
                0
            };
            words[j / 4] |= q << (8 * (j % 4));
        }
        words[8] = u32::from(f16_bits(scale)) | (u32::from(f16_bits(offset)) << 16);
        out.extend_from_slice(&words);
    }
    out
}

/// u32s a group at 16 bits: two halves a word, value `j` in word `j / 2`,
/// the low half first.
pub const GROUP_WORDS_16: usize = 16;

/// `values` (`rows × cols`) as halves, two a word (nothing lost for a
/// checkpoint kept as bf16 within the halves' range).
pub fn to_halves(values: &[f32], rows: usize, cols: usize) -> Vec<u32> {
    assert_eq!(values.len(), rows * cols);
    assert_eq!(cols % GROUP, 0, "{cols} inputs are not whole groups");
    values
        .chunks_exact(2)
        .map(|p| u32::from(f16_bits(p[0])) | (u32::from(f16_bits(p[1])) << 16))
        .collect()
}

/// The values of row `row` (`cols` wide) of an 8-bit matrix.
pub fn dequantize_row8(q: &[u32], cols: usize, row: usize, out: &mut [f32]) {
    let groups = cols / GROUP;
    for g in 0..groups {
        let words = &q[(row * groups + g) * GROUP_WORDS_8..][..GROUP_WORDS_8];
        let scale = f16_to_f32(words[8] as u16);
        let offset = f16_to_f32((words[8] >> 16) as u16);
        for j in 0..GROUP {
            let n = (words[j / 4] >> (8 * (j % 4))) & 0xff;
            out[g * GROUP + j] = offset + n as f32 * scale;
        }
    }
}

/// The values of row `row` (`cols` wide) of a quantised matrix.
pub fn dequantize_row(q: &[u32], cols: usize, row: usize, out: &mut [f32]) {
    let groups = cols / GROUP;
    for g in 0..groups {
        let words = &q[(row * groups + g) * GROUP_WORDS..][..GROUP_WORDS];
        let scale = f16_to_f32(words[4] as u16);
        let offset = f16_to_f32((words[4] >> 16) as u16);
        for j in 0..GROUP {
            let n = (words[j / 8] >> (4 * (j % 8))) & 0xf;
            out[g * GROUP + j] = offset + n as f32 * scale;
        }
    }
}

/// Calibration statistics: for each matrix (by [`super::Qwen3Asr::visit`]'s
/// name), how many input rows it was seen with and each input column's sum
/// of squares over them. The columns with the most energy are the ones
/// whose weights' rounding matters most ([`quantize_weighted`]). Sums of
/// any number of runs add up ([`Statistics::add`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Statistics {
    /// Each matrix: its name, its rows, its columns' sums of squares.
    pub matrices: Vec<(String, u64, Vec<f32>)>,
}

const STATS_MAGIC: &[u8; 8] = b"JDLSTAT1";

impl Statistics {
    /// `other`'s added in (a matrix new to these taken as it is).
    pub fn add(&mut self, other: &Statistics) {
        for (name, rows, sums) in &other.matrices {
            match self.matrices.iter_mut().find(|m| &m.0 == name) {
                Some(m) if m.2.len() == sums.len() => {
                    m.1 += rows;
                    for (a, b) in m.2.iter_mut().zip(sums) {
                        *a += b;
                    }
                }
                Some(_) => {}
                None => self.matrices.push((name.clone(), *rows, sums.clone())),
            }
        }
    }

    /// Matrix `name`'s columns' mean squares (`None` when not seen).
    pub fn mean_squares(&self, name: &str) -> Option<Vec<f32>> {
        let (_, rows, sums) = self.matrices.iter().find(|m| m.0 == name)?;
        (*rows > 0).then(|| sums.iter().map(|s| s / *rows as f32).collect())
    }

    /// As a file's bytes: `JDLSTAT1`, then each matrix's name (u16 length,
    /// UTF-8), rows (u64), columns (u32) and sums (f32), little-endian.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = STATS_MAGIC.to_vec();
        for (name, rows, sums) in &self.matrices {
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&rows.to_le_bytes());
            out.extend_from_slice(&(sums.len() as u32).to_le_bytes());
            for s in sums {
                out.extend_from_slice(&s.to_le_bytes());
            }
        }
        out
    }

    /// The statistics built in (`models/asr-calibration.stats`): the voice
    /// stems of 1000 songs of 100 artists heard at 16 bits. Weighting by
    /// them cuts the words the 4-bit recogniser hears otherwise than the
    /// 16-bit one from 19.2 % to 16.8 % (300 other songs); statistics from
    /// 100 songs did as well, so more would not help. On speech they were
    /// not fitted on (100 German radio plays, the whole mix) from 7.8 % to
    /// 6.5 %.
    pub fn built_in() -> Statistics {
        Statistics::from_bytes(include_bytes!("../../models/asr-calibration.stats"))
            .expect("the built-in calibration statistics")
    }

    /// From [`Statistics::to_bytes`]'s.
    pub fn from_bytes(data: &[u8]) -> Result<Statistics, String> {
        let broken = || "not calibration statistics, or broken ones".to_owned();
        if data.get(..8) != Some(&STATS_MAGIC[..]) {
            return Err(broken());
        }
        let mut at = 8;
        let mut take = |n: usize| -> Result<&[u8], String> {
            let s = data.get(at..at + n).ok_or_else(broken)?;
            at += n;
            Ok(s)
        };
        let mut matrices = Vec::new();
        while let Ok(len) = take(2) {
            let len = usize::from(u16::from_le_bytes([len[0], len[1]]));
            let name = String::from_utf8(take(len)?.to_vec()).map_err(|_| broken())?;
            let rows = u64::from_le_bytes(take(8)?.try_into().unwrap());
            let cols = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
            let sums = take(4 * cols)?
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            matrices.push((name, rows, sums));
        }
        Ok(Statistics { matrices })
    }
}

/// The bits a matrix is held at by a spec: `"4"` (or `"16"`, …) for all,
/// or a comma-separated list of `PREFIX:BITS` (the first whose prefix the
/// name starts with), 4 for the rest.
pub fn bits_by(spec: &str) -> impl Fn(&str) -> u32 + '_ {
    move |name| {
        let spec = spec.trim();
        if let Ok(all) = spec.parse::<u32>() {
            return all;
        }
        spec.split(',')
            .filter_map(|part| part.split_once(':'))
            .find(|(prefix, _)| !prefix.trim().is_empty() && name.starts_with(prefix.trim()))
            .and_then(|(_, bits)| bits.trim().parse().ok())
            .unwrap_or(4)
    }
}

/// Each `group` of `values` rounded to `bits` a value: a scale a group
/// (from the largest magnitude when `symmetric`, else from the range, with
/// its minimum).
pub fn round_groups(values: &mut [f32], bits: u32, group: usize, symmetric: bool) {
    if bits == 4 && group == GROUP && values.len() % group == 0 {
        // Exactly as the 4-bit format holds them (scales as halves).
        let rows = values.len() / group;
        let q = quantize(values, rows, group, symmetric);
        for (r, row) in values.chunks_exact_mut(group).enumerate() {
            dequantize_row(&q, group, r, row);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The packed file is the same on every machine (the player checks it
    /// against a checksum built in): each group is fitted on its own, so
    /// however the threads share the groups out, the words are the same.
    #[test]
    fn packing_does_not_depend_on_how_the_threads_share_it() {
        let (rows, cols) = (96, 256);
        let mut state = 0x2545_f491u32;
        let values: Vec<f32> = (0..rows * cols)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state >> 8) as f32 / 16_777_216.0 - 0.5
            })
            .collect();
        let importance: Vec<f32> = (0..cols).map(|c| 0.1 + (c % 7) as f32).collect();
        for imp in [None, Some(&importance[..])] {
            let whole = quantize_groups(&values, 0, cols, false, imp);
            for per_groups in [1usize, 3, 8, 50, 700] {
                let per = per_groups * GROUP;
                let split: Vec<u32> = values
                    .chunks(per)
                    .enumerate()
                    .flat_map(|(i, part)| quantize_groups(part, (i * per) % cols, cols, false, imp))
                    .collect();
                assert_eq!(split, whole, "split every {per_groups} groups");
            }
            assert_eq!(quantize_with(&values, cols, false, imp), whole);
        }
    }

    #[test]
    fn statistics_add_up_and_read_back() {
        let mut a = Statistics {
            matrices: vec![("x.w".into(), 2, vec![1.0, 4.0])],
        };
        let b = Statistics {
            matrices: vec![
                ("x.w".into(), 1, vec![2.0, 2.0]),
                ("y.w".into(), 5, vec![5.0]),
            ],
        };
        a.add(&b);
        assert_eq!(a.mean_squares("x.w"), Some(vec![1.0, 2.0]));
        assert_eq!(a.mean_squares("y.w"), Some(vec![1.0]));
        assert_eq!(Statistics::from_bytes(&a.to_bytes()), Ok(a));
        assert!(Statistics::from_bytes(b"nope").is_err());
    }

    #[test]
    fn bits_are_given_by_prefix() {
        let all = bits_by("16");
        assert_eq!(all("dec3.qkv.w"), 16);
        let some = bits_by("dec:8, enc:16");
        assert_eq!(some("dec3.qkv.w"), 8);
        assert_eq!(some("enc0.up.w"), 16);
        assert_eq!(some("embed"), 4);
    }

    #[test]
    fn halves_round_trip() {
        for x in [0.0f32, 1.0, -2.5, 0.1, 65504.0, 6.1e-5, 3.0e-7, -1e-3] {
            let back = f16_to_f32(f16_bits(x));
            assert!((back - x).abs() <= x.abs() * 1e-3 + 6e-8, "{x} -> {back}");
        }
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(-2.0), 0xc000);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
    }

    #[test]
    fn the_fitted_group_errs_no_more_than_its_range() {
        // A group with one outlier: the range wastes most steps on it.
        let mut g: Vec<f32> = (0..32)
            .map(|i| ((i * 13 % 32) as f32 - 16.0) / 100.0)
            .collect();
        g[5] = 1.5;
        let lo = g.iter().copied().fold(f32::MAX, f32::min);
        let hi = g.iter().copied().fold(f32::MIN, f32::max);
        let ones = [1.0; 32];
        let range = error_of(
            &g,
            &ones,
            f16_to_f32(f16_bits((hi - lo) / 15.0)),
            f16_to_f32(f16_bits(lo)),
        );
        let (s, o) = fit(&g, &ones);
        assert!(error_of(&g, &ones, s, o) <= range);
    }

    #[test]
    fn weighted_rounding_favours_the_columns_that_matter() {
        // Two rows of a group: the first column carries all the signal.
        let values: Vec<f32> = (0..64)
            .map(|i| ((i * 29 % 64) as f32 - 31.5) / 40.0)
            .collect();
        let mut importance = vec![1e-3f32; 32];
        importance[0] = 1.0;
        importance[1] = 1.0;
        let plain = quantize(&values, 2, 32, false);
        let weighted = quantize_weighted(&values, 2, 32, &importance);
        let err = |q: &[u32]| {
            let mut sum = 0.0f64;
            for row in 0..2 {
                let mut back = [0.0f32; 32];
                dequantize_row(q, 32, row, &mut back);
                for j in 0..32 {
                    let d = f64::from(back[j] - values[row * 32 + j]);
                    sum += f64::from(importance[j]) * d * d;
                }
            }
            sum
        };
        assert!(err(&weighted) <= err(&plain) + 1e-12);
        // All columns alike: the same as without weights.
        assert_eq!(quantize_weighted(&values, 2, 32, &[1.0; 32]), plain);
    }

    #[test]
    fn a_group_comes_back_within_half_a_step() {
        let values: Vec<f32> = (0..64)
            .map(|i| ((i * 37 % 64) as f32 - 31.0) / 50.0)
            .collect();
        for symmetric in [false, true] {
            let q = quantize(&values, 2, 32, symmetric);
            assert_eq!(q.len(), 2 * GROUP_WORDS);
            for row in 0..2 {
                let mut back = [0.0f32; 32];
                dequantize_row(&q, 32, row, &mut back);
                let g = &values[row * 32..][..32];
                // A step is at most a seventh of the largest magnitude.
                let step = g.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 7.0;
                for (a, b) in g.iter().zip(&back) {
                    assert!((a - b).abs() <= step * 0.501 + 1e-4, "{a} {b} {symmetric}");
                }
            }
        }
    }
}
