//! One second-order section, a biquad (transposed direct form II, its
//! coefficients normalised: `a0` = 1), in `f32` or `f64` as its filter
//! needs: the loudness meter's K-weighting ([`crate::loudness`]) and the
//! pitch tracker's band ([`crate::yin`]).

use std::ops::{Add, Mul, Sub};

/// A biquad of samples `T`: its feed-forward coefficients `b`, its
/// feedback `a` (`a1`, `a2`), and its state.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Biquad<T> {
    b: [T; 3],
    a: [T; 2],
    z: [T; 2],
}

impl<T> Biquad<T>
where
    T: Copy + Default + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    /// The section of coefficients `b` and `a`, at rest.
    pub(crate) fn new(b: [T; 3], a: [T; 2]) -> Biquad<T> {
        Biquad {
            b,
            a,
            z: [T::default(); 2],
        }
    }

    /// Its coefficients, `b` then `a`, for a filter's test to hold to its
    /// table.
    #[cfg(test)]
    pub(crate) fn coefficients(&self) -> ([T; 3], [T; 2]) {
        (self.b, self.a)
    }

    /// `x` through it.
    pub(crate) fn run(&mut self, x: T) -> T {
        let y = self.b[0] * x + self.z[0];
        self.z[0] = self.b[1] * x - self.a[0] * y + self.z[1];
        self.z[1] = self.b[2] * x - self.a[1] * y;
        y
    }
}
