//! Requantization to 24 bits with TPDF dither.

/// The dither generator's seed.
const SEED: u64 = 0x2545_f491_4f6c_dd1d;

/// The requantization to 24 bits: TPDF dither of ±1 LSB (the difference of
/// two uniform values from a seeded xorshift64* generator), rounding, and
/// saturation to the 24-bit range, counting the samples that clip.
///
/// The playback engine's resampler requantizes its output with one, its
/// source requantizes floating-point and wider-than-24-bit samples with
/// one (reseeded per packet, [`Requantizer::reseed`]), and the
/// [`crate::mixer`] its mix.
#[derive(Clone, Debug)]
pub struct Requantizer {
    rng: u64,
    clipped: u64,
}

impl Default for Requantizer {
    fn default() -> Self {
        Requantizer::new()
    }
}

impl Requantizer {
    /// A requantizer at the start of its sequence, with no clips counted.
    pub fn new() -> Requantizer {
        Requantizer {
            rng: SEED,
            clipped: 0,
        }
    }

    /// Back to the start of the sequence, with no clips counted.
    pub fn reset(&mut self) {
        self.rng = SEED;
        self.clipped = 0;
    }

    /// Restarts the dither sequence at a point derived from `key` (the
    /// clip count stays): the same key gives the same dither.
    pub fn reseed(&mut self, key: u64) {
        // SplitMix64 of the key; xorshift needs a non-zero state.
        let mut z = SEED ^ key.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        self.rng = if z == 0 { SEED } else { z };
    }

    /// Samples that clipped since the start (or the last reset).
    pub fn clipped(&self) -> u64 {
        self.clipped
    }

    /// `y` (in 24-bit units, full scale ±2²³) dithered, rounded and
    /// saturated to a 24-bit sample.
    pub fn quantize(&mut self, y: f64) -> i32 {
        self.quantize_to(y, 24)
    }

    /// `y` (in 24-bit units) dithered at one LSB of `bits` (16 to 24),
    /// rounded to it and saturated to its range: a 24-bit value whose low
    /// `24 - bits` bits are zero (a 16-bit device's sample, as
    /// [`crate::pack`] takes it).
    pub fn quantize_to(&mut self, y: f64, bits: u32) -> i32 {
        let shift = 24 - bits.clamp(16, 24);
        let lsb = f64::from(1u32 << shift);
        let max = f64::from((1i32 << (23 - shift)) - 1);
        let dither = self.uniform() - self.uniform();
        let q = (y / lsb + dither).round();
        let q = if q > max {
            self.clipped += 1;
            max
        } else if q < -max - 1.0 {
            self.clipped += 1;
            -max - 1.0
        } else {
            q
        };
        (q as i32) << shift
    }

    /// Moves the sequence on as if `samples` samples were quantized,
    /// without counting clips.
    pub fn skip(&mut self, samples: usize) {
        for _ in 0..samples {
            self.uniform();
            self.uniform();
        }
    }

    /// A uniform value in `[0, 1)` (xorshift64*).
    fn uniform(&mut self) -> f64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        (x.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requantizer_dithers_within_reach_and_reseeds_by_key() {
        let mut q = Requantizer::new();
        let input: Vec<f64> = (0..10_000)
            .map(|k| (k as f64 * 0.37).sin() * 1e6 + 0.25)
            .collect();
        let out: Vec<i32> = input.iter().map(|&y| q.quantize(y)).collect();
        let mut errors = 0.0;
        for (&o, &y) in out.iter().zip(&input) {
            let e = f64::from(o) - y;
            assert!(e.abs() < 1.5, "{e}");
            errors += e;
        }
        // TPDF: unbiased (the mean error is near zero, not -0.25).
        assert!((errors / input.len() as f64).abs() < 0.05);
        assert_eq!(q.clipped(), 0);
        // The same key gives the same dither; skipping equals quantizing.
        let mut a = Requantizer::new();
        let mut b = Requantizer::new();
        a.reseed(4_410);
        b.reseed(4_410);
        let first: Vec<i32> = input[..100].iter().map(|&y| a.quantize(y)).collect();
        b.skip(40);
        let rest: Vec<i32> = input[40..100].iter().map(|&y| b.quantize(y)).collect();
        assert_eq!(&first[40..], &rest[..]);
        let mut c = Requantizer::new();
        c.reseed(4_411);
        let other: Vec<i32> = input[..100].iter().map(|&y| c.quantize(y)).collect();
        assert_ne!(first, other);
        // Saturation, counted; reset clears the count.
        assert_eq!(c.quantize(9e6), 8_388_607);
        assert_eq!(c.quantize(-9e6), -8_388_608);
        assert_eq!(c.clipped(), 2);
        c.reset();
        assert_eq!(c.clipped(), 0);
    }

    #[test]
    fn requantizing_to_16_bits_dithers_at_its_lsb() {
        let mut q = Requantizer::new();
        let y = 1000.0 * 256.0 + 64.0; // a quarter of a 16-bit LSB over 1000
        let n = 20_000;
        let mut sum = 0.0;
        for _ in 0..n {
            let v = q.quantize_to(y, 16);
            assert_eq!(v & 0xFF, 0, "16-bit samples");
            assert!((v >> 8) >= 999 && (v >> 8) <= 1001);
            sum += f64::from(v);
        }
        // Unbiased: the mean is y, not 1000 × 256 (truncation's floor).
        assert!(
            (sum / f64::from(n) - y).abs() < 4.0,
            "{}",
            sum / f64::from(n)
        );
        assert_eq!(q.quantize_to(9e6, 16), 32_767 << 8);
        assert_eq!(q.quantize_to(-9e6, 16), -32_768 << 8);
        assert_eq!(q.clipped(), 2);
    }
}
