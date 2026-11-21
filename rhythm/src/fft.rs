//! A real FFT of a power-of-two length, and its inverse, computed as a
//! complex FFT of half that length plus one pass that separates (or, for
//! the inverse, joins) the even and odd samples' spectra.

use std::f64::consts::PI;

/// A complex number in `f32`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    pub fn new(re: f32, im: f32) -> Complex {
        Complex { re, im }
    }

    /// `e^(i·angle)`.
    pub fn unit(angle: f64) -> Complex {
        Complex::new(angle.cos() as f32, angle.sin() as f32)
    }

    pub fn add(self, o: Complex) -> Complex {
        Complex::new(self.re + o.re, self.im + o.im)
    }

    pub fn sub(self, o: Complex) -> Complex {
        Complex::new(self.re - o.re, self.im - o.im)
    }

    pub fn mul(self, o: Complex) -> Complex {
        Complex::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }

    pub fn scale(self, k: f32) -> Complex {
        Complex::new(self.re * k, self.im * k)
    }

    pub fn conj(self) -> Complex {
        Complex::new(self.re, -self.im)
    }

    pub fn norm(self) -> f32 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
}

/// A real FFT of one length, with its tables.
pub(crate) struct RealFft {
    n: usize,
    /// `e^(-2πik/m)` for the half-length complex FFT, `k < m/2`.
    twiddles: Vec<Complex>,
    /// `e^(-2πik/n)` for the split pass, `k ≤ n/2`.
    split: Vec<Complex>,
    /// The bit-reversed index of each position of the half-length FFT.
    reversed: Vec<u32>,
    scratch: Vec<Complex>,
}

impl RealFft {
    /// Tables for length `n`, a power of two of at least 4.
    pub fn new(n: usize) -> RealFft {
        assert!(n.is_power_of_two() && n >= 4, "FFT length {n}");
        let m = n / 2;
        let bits = m.trailing_zeros();
        RealFft {
            n,
            twiddles: (0..m / 2)
                .map(|k| Complex::unit(-2.0 * PI * k as f64 / m as f64))
                .collect(),
            split: (0..=m)
                .map(|k| Complex::unit(-2.0 * PI * k as f64 / n as f64))
                .collect(),
            reversed: (0..m)
                .map(|i| {
                    if bits == 0 {
                        0
                    } else {
                        ((i as u32).reverse_bits()) >> (32 - bits)
                    }
                })
                .collect(),
            scratch: vec![Complex::default(); m],
        }
    }

    /// Writes the magnitudes of bins `0..=n/2` of `input` (length `n`) to
    /// `out` (length `n/2 + 1`).
    pub fn magnitudes(&mut self, input: &[f32], out: &mut [f32]) {
        self.transform(input, |k, x| out[k] = x.norm());
    }

    /// Writes bins `0..=n/2` of `input` (length `n`) to `out` (length
    /// `n/2 + 1`): `X[k] = Σ x[j]·e^(-2πijk/n)`.
    pub fn spectrum(&mut self, input: &[f32], out: &mut [Complex]) {
        self.transform(input, |k, x| out[k] = x);
    }

    /// The forward transform, bin by bin to `put`.
    fn transform(&mut self, input: &[f32], mut put: impl FnMut(usize, Complex)) {
        let m = self.n / 2;
        debug_assert_eq!(input.len(), self.n);
        for (i, z) in self.scratch.iter_mut().enumerate() {
            *z = Complex::new(input[2 * i], input[2 * i + 1]);
        }
        fft(&mut self.scratch, &self.twiddles, &self.reversed);
        for k in 0..=m {
            let zk = self.scratch[k % m];
            let zm = self.scratch[(m - k) % m].conj();
            let even = zk.add(zm).scale(0.5);
            let d = zk.sub(zm);
            // (zk − zm) / (2i) = (d.im − i·d.re) / 2
            let odd = Complex::new(d.im, -d.re).scale(0.5);
            put(k, even.add(self.split[k].mul(odd)));
        }
    }

    /// Writes the real signal (length `n`) whose bins `0..=n/2` are
    /// `spectrum` to `out`: `x[j] = (1/n)·Σ X[k]·e^(2πijk/n)` over all `n`
    /// bins, the upper half mirrored. The imaginary parts of bins 0 and
    /// `n/2` are ignored.
    pub fn inverse(&mut self, spectrum: &[Complex], out: &mut [f32]) {
        let m = self.n / 2;
        debug_assert_eq!(spectrum.len(), m + 1);
        debug_assert_eq!(out.len(), self.n);
        let edge = |k: usize| {
            let x = spectrum[k];
            if k == 0 || k == m {
                Complex::new(x.re, 0.0)
            } else {
                x
            }
        };
        // Join the even and odd samples' spectra (the forward pass run
        // backwards), as the conjugate, so the forward FFT inverts it.
        for k in 0..m {
            let (xk, xm) = (edge(k), edge(m - k).conj());
            let even = xk.add(xm).scale(0.5);
            let odd = self.split[k].conj().mul(xk.sub(xm).scale(0.5));
            // Z = E + i·O
            let z = even.add(Complex::new(-odd.im, odd.re));
            self.scratch[k] = z.conj();
        }
        fft(&mut self.scratch, &self.twiddles, &self.reversed);
        let scale = 1.0 / m as f32;
        for (i, z) in self.scratch.iter().enumerate() {
            let z = z.conj().scale(scale);
            out[2 * i] = z.re;
            out[2 * i + 1] = z.im;
        }
    }
}

/// An in-place iterative radix-2 complex FFT.
fn fft(data: &mut [Complex], twiddles: &[Complex], reversed: &[u32]) {
    let m = data.len();
    for (i, &r) in reversed.iter().enumerate() {
        let j = r as usize;
        if i < j {
            data.swap(i, j);
        }
    }
    let mut size = 2;
    while size <= m {
        let half = size / 2;
        let step = m / size;
        for start in (0..m).step_by(size) {
            for k in 0..half {
                let w = twiddles[k * step];
                let a = data[start + k];
                let b = data[start + k + half].mul(w);
                data[start + k] = a.add(b);
                data[start + k + half] = a.sub(b);
            }
        }
        size *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn noise(n: usize) -> Vec<f32> {
        let mut seed = 0x1234_5678u32;
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    #[test]
    fn the_inverse_gives_the_signal_back() {
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
    fn magnitudes_match_a_direct_dft() {
        for n in [4usize, 16, 256, 1024] {
            let mut seed = 0x1234_5678u32;
            let input: Vec<f32> = (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
                })
                .collect();
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
}
