//! Radix-2 FFTs of a power-of-two length, in `f32` or `f64`: a complex one
//! in place ([`Fft`]), and a real one ([`RealFft`]) computed as a complex
//! FFT of half the length plus one pass that separates (or, for the
//! inverse, joins) the even and odd samples' spectra.
//!
//! The tables are worked out in `f64` and rounded to the precision used,
//! so an `f32` transform and an `f64` one share their angles.

use std::f64::consts::PI;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A precision the transforms run in: `f32` or `f64`.
pub trait Float:
    Copy
    + Debug
    + Default
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// `x` rounded to this precision.
    fn of(x: f64) -> Self;
    /// The square root.
    fn sqrt(self) -> Self;
}

impl Float for f32 {
    fn of(x: f64) -> f32 {
        x as f32
    }

    fn sqrt(self) -> f32 {
        f32::sqrt(self)
    }
}

impl Float for f64 {
    fn of(x: f64) -> f64 {
        x
    }

    fn sqrt(self) -> f64 {
        f64::sqrt(self)
    }
}

/// A complex number.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex<T> {
    /// The real part.
    pub re: T,
    /// The imaginary part.
    pub im: T,
}

impl<T: Float> Complex<T> {
    /// `re + i·im`.
    pub fn new(re: T, im: T) -> Complex<T> {
        Complex { re, im }
    }

    /// `e^(i·angle)`.
    pub fn unit(angle: f64) -> Complex<T> {
        Complex::new(T::of(angle.cos()), T::of(angle.sin()))
    }

    /// Times the real `k`.
    pub fn scale(self, k: T) -> Complex<T> {
        Complex::new(self.re * k, self.im * k)
    }

    /// The conjugate.
    pub fn conj(self) -> Complex<T> {
        Complex::new(self.re, -self.im)
    }

    /// The magnitude.
    pub fn norm(self) -> T {
        (self.re * self.re + self.im * self.im).sqrt()
    }
}

impl<T: Float> Add for Complex<T> {
    type Output = Complex<T>;

    fn add(self, o: Complex<T>) -> Complex<T> {
        Complex::new(self.re + o.re, self.im + o.im)
    }
}

impl<T: Float> Sub for Complex<T> {
    type Output = Complex<T>;

    fn sub(self, o: Complex<T>) -> Complex<T> {
        Complex::new(self.re - o.re, self.im - o.im)
    }
}

impl<T: Float> Mul for Complex<T> {
    type Output = Complex<T>;

    fn mul(self, o: Complex<T>) -> Complex<T> {
        Complex::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

/// An in-place complex FFT of one length, with its tables.
#[derive(Clone, Debug)]
pub struct Fft<T> {
    /// `e^(-2πik/n)` for `k < n/2`.
    twiddles: Vec<Complex<T>>,
    /// The bit-reversed index of each index.
    reversed: Vec<u32>,
}

impl<T: Float> Fft<T> {
    /// Tables for length `n`, a power of two.
    pub fn new(n: usize) -> Fft<T> {
        assert!(n.is_power_of_two(), "FFT length {n}");
        let bits = n.trailing_zeros();
        Fft {
            twiddles: (0..n / 2)
                .map(|k| Complex::unit(-2.0 * PI * k as f64 / n as f64))
                .collect(),
            reversed: (0..n)
                .map(|i| {
                    if bits == 0 {
                        0
                    } else {
                        (i as u32).reverse_bits() >> (32 - bits)
                    }
                })
                .collect(),
        }
    }

    /// The length.
    pub fn len(&self) -> usize {
        self.reversed.len()
    }

    /// Whether the length is 0 (it never is).
    pub fn is_empty(&self) -> bool {
        self.reversed.is_empty()
    }

    /// `X[k] = Σ x[j]·e^(-2πijk/n)`, in place.
    pub fn forward(&self, data: &mut [Complex<T>]) {
        let n = data.len();
        debug_assert_eq!(n, self.reversed.len());
        for (i, &r) in self.reversed.iter().enumerate() {
            let r = r as usize;
            if i < r {
                data.swap(i, r);
            }
        }
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let stride = n / len;
            for block in data.chunks_exact_mut(len) {
                let (low, high) = block.split_at_mut(half);
                for (k, (a, b)) in low.iter_mut().zip(high.iter_mut()).enumerate() {
                    let t = *b * self.twiddles[k * stride];
                    let u = *a;
                    *a = u + t;
                    *b = u - t;
                }
            }
            len *= 2;
        }
    }

    /// `x[j] = (1/n)·Σ X[k]·e^(2πijk/n)`, in place.
    pub fn inverse(&self, data: &mut [Complex<T>]) {
        for z in data.iter_mut() {
            *z = z.conj();
        }
        self.forward(data);
        let scale = T::of(1.0) / T::of(data.len() as f64);
        for z in data.iter_mut() {
            *z = Complex::new(z.re * scale, -z.im * scale);
        }
    }
}

/// A real FFT of one length, with its tables.
pub struct RealFft<T> {
    n: usize,
    /// The complex FFT of half the length.
    half: Fft<T>,
    /// `e^(-2πik/n)` for the split pass, `k ≤ n/2`.
    split: Vec<Complex<T>>,
    scratch: Vec<Complex<T>>,
}

impl<T: Float> RealFft<T> {
    /// Tables for length `n`, a power of two of at least 4.
    pub fn new(n: usize) -> RealFft<T> {
        assert!(n.is_power_of_two() && n >= 4, "FFT length {n}");
        let m = n / 2;
        RealFft {
            n,
            half: Fft::new(m),
            split: (0..=m)
                .map(|k| Complex::unit(-2.0 * PI * k as f64 / n as f64))
                .collect(),
            scratch: vec![Complex::default(); m],
        }
    }

    /// Writes the magnitudes of bins `0..=n/2` of `input` (length `n`) to
    /// `out` (length `n/2 + 1`).
    pub fn magnitudes(&mut self, input: &[T], out: &mut [T]) {
        self.transform(input, |k, x| out[k] = x.norm());
    }

    /// Writes bins `0..=n/2` of `input` (length `n`) to `out` (length
    /// `n/2 + 1`): `X[k] = Σ x[j]·e^(-2πijk/n)`.
    pub fn spectrum(&mut self, input: &[T], out: &mut [Complex<T>]) {
        self.transform(input, |k, x| out[k] = x);
    }

    /// The forward transform, bin by bin to `put`.
    fn transform(&mut self, input: &[T], mut put: impl FnMut(usize, Complex<T>)) {
        let m = self.n / 2;
        debug_assert_eq!(input.len(), self.n);
        for (i, z) in self.scratch.iter_mut().enumerate() {
            *z = Complex::new(input[2 * i], input[2 * i + 1]);
        }
        self.half.forward(&mut self.scratch);
        let half = T::of(0.5);
        for k in 0..=m {
            let zk = self.scratch[k % m];
            let zm = self.scratch[(m - k) % m].conj();
            let even = (zk + zm).scale(half);
            let d = zk - zm;
            // (zk − zm) / (2i) = (d.im − i·d.re) / 2
            let odd = Complex::new(d.im, -d.re).scale(half);
            put(k, even + self.split[k] * odd);
        }
    }

    /// Writes the real signal (length `n`) whose bins `0..=n/2` are
    /// `spectrum` to `out`: `x[j] = (1/n)·Σ X[k]·e^(2πijk/n)` over all `n`
    /// bins, the upper half mirrored. The imaginary parts of bins 0 and
    /// `n/2` are ignored.
    pub fn inverse(&mut self, spectrum: &[Complex<T>], out: &mut [T]) {
        let m = self.n / 2;
        debug_assert_eq!(spectrum.len(), m + 1);
        debug_assert_eq!(out.len(), self.n);
        let edge = |k: usize| {
            let x = spectrum[k];
            if k == 0 || k == m {
                Complex::new(x.re, T::default())
            } else {
                x
            }
        };
        let half = T::of(0.5);
        // Join the even and odd samples' spectra (the forward pass run
        // backwards), as the conjugate, so the forward FFT inverts it.
        for k in 0..m {
            let (xk, xm) = (edge(k), edge(m - k).conj());
            let even = (xk + xm).scale(half);
            let odd = self.split[k].conj() * (xk - xm).scale(half);
            // Z = E + i·O
            let z = even + Complex::new(-odd.im, odd.re);
            self.scratch[k] = z.conj();
        }
        self.half.forward(&mut self.scratch);
        let scale = T::of(1.0) / T::of(m as f64);
        for (i, z) in self.scratch.iter().enumerate() {
            let z = z.conj().scale(scale);
            out[2 * i] = z.re;
            out[2 * i + 1] = z.im;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize) -> Vec<f32> {
        let mut seed = 0x1234_5678u32;
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    /// The magnitudes of bins `0..=n/2` of `input`, summed one by one.
    fn direct(input: &[f32]) -> Vec<f32> {
        let n = input.len();
        (0..=n / 2)
            .map(|k| {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for (i, &x) in input.iter().enumerate() {
                    let a = -2.0 * PI * (k * i) as f64 / n as f64;
                    re += f64::from(x) * a.cos();
                    im += f64::from(x) * a.sin();
                }
                (re * re + im * im).sqrt() as f32
            })
            .collect()
    }

    #[test]
    fn the_real_inverse_gives_the_signal_back() {
        for n in [4usize, 16, 2048] {
            let input = noise(n);
            let mut fft = RealFft::new(n);
            let mut bins = vec![Complex::default(); n / 2 + 1];
            fft.spectrum(&input, &mut bins);
            let mut back = vec![0.0; n];
            fft.inverse(&bins, &mut back);
            for (a, b) in input.iter().zip(&back) {
                assert!((a - b).abs() < 1e-5, "n {n}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn real_magnitudes_match_a_direct_dft() {
        for n in [4usize, 16, 256, 1024] {
            let input = noise(n);
            let mut fft = RealFft::new(n);
            let mut out = vec![0.0; n / 2 + 1];
            fft.magnitudes(&input, &mut out);
            let want = direct(&input);
            for (k, (a, b)) in out.iter().zip(&want).enumerate() {
                assert!(
                    (a - b).abs() < 1e-3 * (1.0 + b),
                    "n {n} bin {k}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn the_complex_fft_matches_a_direct_transform_and_inverts() {
        let n = 64;
        let x: Vec<f64> = (0..n).map(|i| ((i * 7 % 11) as f64 - 5.0) / 3.0).collect();
        let fft = Fft::new(n);
        let mut data: Vec<Complex<f64>> = x.iter().map(|&re| Complex::new(re, 0.0)).collect();
        fft.forward(&mut data);
        for (k, z) in data.iter().enumerate() {
            let (mut re, mut im) = (0.0, 0.0);
            for (j, &v) in x.iter().enumerate() {
                let a = -2.0 * PI * (j * k) as f64 / n as f64;
                re += v * a.cos();
                im += v * a.sin();
            }
            assert!((z.re - re).abs() < 1e-9 && (z.im - im).abs() < 1e-9, "{k}");
        }
        fft.inverse(&mut data);
        for (z, &v) in data.iter().zip(&x) {
            assert!((z.re - v).abs() < 1e-12 && z.im.abs() < 1e-12);
        }
    }

    #[test]
    fn the_real_fft_is_the_complex_one_of_its_samples() {
        let n = 32;
        let input = noise(n);
        let mut bins = vec![Complex::default(); n / 2 + 1];
        RealFft::new(n).spectrum(&input, &mut bins);
        let mut whole: Vec<Complex<f32>> = input.iter().map(|&re| Complex::new(re, 0.0)).collect();
        Fft::new(n).forward(&mut whole);
        for (a, b) in bins.iter().zip(&whole) {
            assert!((a.re - b.re).abs() < 1e-5 && (a.im - b.im).abs() < 1e-5);
        }
    }
}
