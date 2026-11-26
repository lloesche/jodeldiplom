//! The signal tools every audio crate here shares, each written once: the
//! FFTs ([`fft`]), the windows and the windowed sinc ([`window`]), the
//! Slaney mel scale ([`mel`]), xorshift ([`xorshift32`], [`xorshift64`]),
//! FNV-1a ([`fnv1a`], [`Fnv1a`]), WAV files ([`wav`]) and the 24-bit full
//! scale ([`FULL_SCALE`], [`PEAK`]).
//!
//! It depends on nothing, so the lowest crates (the output, the synth) can
//! use it too.

pub mod fft;
pub mod mel;
pub mod wav;
pub mod window;

/// The 24-bit sample that stands for 1.0: 2²³. A sample becomes a number
/// by dividing by it.
pub const FULL_SCALE: f64 = 8_388_608.0;

/// The largest 24-bit sample, 2²³ − 1. A number becomes a sample scaled
/// by it, so that 1.0 still fits.
pub const PEAK: f64 = 8_388_607.0;

/// The next number of xorshift32 state `s` (shifts 13, 17, 5; G.
/// Marsaglia, "Xorshift RNGs", Journal of Statistical Software, 2003).
/// A state not 0 never becomes 0.
#[inline]
pub fn xorshift32(s: &mut u32) -> u32 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    x
}

/// The next number of xorshift64 state `s` (shifts 13, 7, 17; Marsaglia
/// 2003). A state not 0 never becomes 0.
#[inline]
pub fn xorshift64(s: &mut u64) -> u64 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    x
}

/// FNV-1a, 64 bits, fed a piece at a time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fnv1a(pub u64);

impl Fnv1a {
    /// The offset basis.
    pub const BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    /// The prime.
    pub const PRIME: u64 = 0x0100_0000_01b3;

    /// A hash of nothing yet.
    pub fn new() -> Fnv1a {
        Fnv1a(Fnv1a::BASIS)
    }

    /// `bytes` hashed in.
    pub fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(Fnv1a::PRIME);
        }
    }

    /// The hash so far.
    pub fn finish(self) -> u64 {
        self.0
    }
}

impl Default for Fnv1a {
    fn default() -> Fnv1a {
        Fnv1a::new()
    }
}

/// FNV-1a, 64 bits, of `bytes`.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = Fnv1a::new();
    h.write(bytes);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_matches_its_published_values() {
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn xorshift_never_reaches_zero() {
        let (mut a, mut b) = (1u32, 1u64);
        for _ in 0..10_000 {
            assert_ne!(xorshift32(&mut a), 0);
            assert_ne!(xorshift64(&mut b), 0);
        }
    }

    #[test]
    fn the_full_scale_is_one_more_than_the_peak() {
        assert_eq!(FULL_SCALE, PEAK + 1.0);
        assert_eq!(FULL_SCALE as f32 as f64, FULL_SCALE);
        assert_eq!(PEAK as f32 as f64, PEAK);
    }
}
