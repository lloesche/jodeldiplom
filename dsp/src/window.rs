//! Windows: the periodic Hann window, and what a Kaiser-windowed sinc is
//! made of (the Bessel function I₀ and the sinc).

use std::f64::consts::PI;

/// Value `i` of the periodic Hann window of `n`: `0.5 − 0.5·cos(2πi/n)`.
pub fn hann_at(i: usize, n: usize) -> f64 {
    0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos()
}

/// The periodic Hann window of `n` ([`hann_at`]).
pub fn hann(n: usize) -> Vec<f64> {
    (0..n).map(|i| hann_at(i, n)).collect()
}

/// The zeroth-order modified Bessel function of the first kind, summed
/// until a term is below 1e-18 of the sum.
pub fn bessel_i0(x: f64) -> f64 {
    let quarter = x * x / 4.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut k = 1.0;
    while term > sum * 1e-18 {
        term *= quarter / (k * k);
        sum += term;
        k += 1.0;
    }
    sum
}

/// The normalised sinc: `sin(πx) / (πx)`, 1 at 0.
pub fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = PI * x;
        px.sin() / px
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bessel_matches_known_values() {
        for (x, want) in [
            (0.0, 1.0),
            (1.0, 1.266_065_877_752_008_4),
            (5.0, 27.239_871_823_604_45),
            (10.0, 2_815.716_628_466_254),
        ] {
            assert!((bessel_i0(x) - want).abs() < 1e-14 * want, "{x}");
        }
    }

    #[test]
    fn hann_is_periodic_and_peaks_in_the_middle() {
        let w = hann(8);
        assert_eq!(w[0], 0.0);
        assert!((w[4] - 1.0).abs() < 1e-15);
        assert!((w[1] - w[7]).abs() < 1e-15);
    }

    #[test]
    fn sinc_is_one_at_zero_and_zero_at_whole_numbers() {
        assert_eq!(sinc(0.0), 1.0);
        for k in 1..5 {
            assert!(sinc(f64::from(k)).abs() < 1e-15);
        }
    }
}
