//! K blocks packed by GPTQ (Frantar et al., 2022): given the mean product
//! of each two inputs of a matrix, a row is rounded a column at a time and
//! each rounding error moved onto the columns not yet rounded, so the
//! row's output on those inputs changes least.

use jodel_rhythm::nn::quant::{Bits, BLOCK, RUN};

use super::fit::{fit_block, write_block, Fit};

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
/// of `bits` by GPTQ: a row a column at a time, each column's rounding
/// error spread over the columns after it so the row's output on its
/// inputs changes least; `gram` the mean product of each two columns'
/// inputs (`[cols][cols]`, damped by a hundredth of its mean diagonal).
/// Each block is fitted by `fit` (by the gram's diagonal where it has no
/// importance) to the row as it stands when the block's first column
/// comes.
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
    // A tool run on its own: every processor.
    let threads = jodel_tensors::kernels::processors();
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
    let highest = f32::from(bits.top());
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
                    .clamp(0.0, highest)
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

#[cfg(test)]
mod tests {
    use super::super::fit::quantize;
    use super::*;
    use jodel_rhythm::nn::quant::{dequantize, BLOCK_BYTES};

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
}
