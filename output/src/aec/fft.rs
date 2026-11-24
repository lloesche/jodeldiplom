//! The transform the canceller works in: [`N`] points, two blocks.

use super::BLOCK;
use std::f32::consts::PI;

/// The transform's length: two blocks.
pub(super) const N: usize = 2 * BLOCK;
/// Its bins (0 to Nyquist).
pub(super) const BINS: usize = N / 2 + 1;

/// One frame's bins.
pub(super) type Bins = [C32; BINS];

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct C32 {
    pub(super) re: f32,
    pub(super) im: f32,
}

impl C32 {
    pub(super) fn mul(self, o: C32) -> C32 {
        C32 {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }

    /// conj(self) * o
    pub(super) fn conj_mul(self, o: C32) -> C32 {
        C32 {
            re: self.re * o.re + self.im * o.im,
            im: self.re * o.im - self.im * o.re,
        }
    }

    pub(super) fn norm(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    pub(super) fn scale(self, s: f32) -> C32 {
        C32 {
            re: self.re * s,
            im: self.im * s,
        }
    }

    pub(super) fn add(self, o: C32) -> C32 {
        C32 {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }

    pub(super) fn sub(self, o: C32) -> C32 {
        C32 {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }
}

/// A radix-2 transform of [`N`] points.
pub(super) struct Fft {
    twiddles: Vec<C32>,
    order: Vec<usize>,
}

impl Fft {
    pub(super) fn new() -> Fft {
        let bits = N.trailing_zeros();
        Fft {
            twiddles: (0..N / 2)
                .map(|k| {
                    let a = -2.0 * PI * k as f32 / N as f32;
                    C32 {
                        re: a.cos(),
                        im: a.sin(),
                    }
                })
                .collect(),
            order: (0..N)
                .map(|i| i.reverse_bits() >> (usize::BITS - bits))
                .collect(),
        }
    }

    /// In place; `inverse` without the 1/N.
    fn run(&self, x: &mut [C32; N], inverse: bool) {
        for (i, &j) in self.order.iter().enumerate() {
            if j > i {
                x.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= N {
            let stride = N / len;
            for start in (0..N).step_by(len) {
                for k in 0..len / 2 {
                    let mut w = self.twiddles[k * stride];
                    if inverse {
                        w.im = -w.im;
                    }
                    let a = x[start + k];
                    let b = x[start + k + len / 2].mul(w);
                    x[start + k] = a.add(b);
                    x[start + k + len / 2] = a.sub(b);
                }
            }
            len *= 2;
        }
    }

    /// The bins of `a` then `b` (each [`BLOCK`] samples, `a` older), as the
    /// transform of their concatenation.
    pub(super) fn real(&self, a: &[f32], b: &[f32]) -> Bins {
        let mut x = [C32::default(); N];
        for (i, v) in a.iter().chain(b).enumerate().take(N) {
            x[i].re = *v;
        }
        self.run(&mut x, false);
        let mut out = [C32::default(); BINS];
        out.copy_from_slice(&x[..BINS]);
        out
    }

    /// The real samples of the bins `bins` (and their mirror), scaled.
    pub(super) fn inverse(&self, bins: &Bins) -> [f32; N] {
        let mut x = [C32::default(); N];
        x[..BINS].copy_from_slice(bins);
        for k in 1..N / 2 {
            x[N - k] = C32 {
                re: bins[k].re,
                im: -bins[k].im,
            };
        }
        self.run(&mut x, true);
        let mut out = [0.0f32; N];
        for (o, v) in out.iter_mut().zip(x.iter()) {
            *o = v.re / N as f32;
        }
        out
    }
}
