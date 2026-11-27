//! The sampler's numbers: the starting noise as PyTorch draws it on the
//! processor for the reference pipeline ([`noise`]) and the four steps'
//! times ([`schedule`]).

/// PyTorch's processor generator: MT19937 seeded as `manual_seed` seeds it.
struct Mt19937 {
    state: [u32; 624],
    at: usize,
}

impl Mt19937 {
    fn new(seed: u64) -> Mt19937 {
        let mut state = [0u32; 624];
        state[0] = seed as u32;
        for i in 1..624 {
            let prev = state[i - 1];
            state[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { state, at: 624 }
    }

    fn next(&mut self) -> u32 {
        if self.at >= 624 {
            for i in 0..624 {
                let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
                let mut v = self.state[(i + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= 0x9908_b0df;
                }
                self.state[i] = v;
            }
            self.at = 0;
        }
        let mut y = self.state[self.at];
        self.at += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }
}

/// `x` rounded to bf16 (to nearest, ties to even), as f32.
pub(crate) fn bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(rounded)
}

/// `count` values of a standard normal as `torch.randn(..., dtype=bfloat16,
/// generator=torch.Generator("cpu").manual_seed(seed))` draws them (as the
/// reference pipeline does for its latents): a uniform in bf16 for each
/// (the generator's 32 bits, their lowest 8 over 256), then Box–Muller over
/// each 16 (the first 8 the radii's, the next 8 the angles'), every step
/// rounded to bf16; a last part of fewer than 16 drawn again over the last
/// 16. (Its logarithms and cosines are the processor's: a value at a
/// rounding tie may come out a bf16 step apart.)
pub fn noise(seed: u64, count: usize) -> Vec<f32> {
    assert!(count >= 16, "PyTorch draws fewer than 16 otherwise");
    let mut rng = Mt19937::new(seed);
    let mut uniform = || ((rng.next() & 0xff) as f32) / 256.0;
    let mut data: Vec<f32> = (0..count).map(|_| uniform()).collect();
    let fill16 = |d: &mut [f32]| {
        for j in 0..8 {
            let u1 = bf16(1.0 - d[j]);
            let u2 = d[j + 8];
            let radius = bf16(bf16(-2.0 * bf16(u1.ln())).sqrt());
            let theta = bf16((2.0 * std::f64::consts::PI * f64::from(u2)) as f32);
            d[j] = bf16(radius * bf16(theta.cos()));
            d[j + 8] = bf16(radius * bf16(theta.sin()));
        }
    };
    for i in (0..count - 15).step_by(16) {
        fill16(&mut data[i..i + 16]);
    }
    if !count.is_multiple_of(16) {
        let last = count - 16;
        for v in &mut data[last..] {
            *v = uniform();
        }
        fill16(&mut data[last..]);
    }
    data
}

/// The four steps' times (σ from 1 down, the transformer's timestep σ ·
/// 1000) for an image of `tokens` latent tokens, as the reference
/// pipeline's scheduler makes them: σ = 1, 0.75, 0.5, 0.25 shifted by
/// exp(μ) / (exp(μ) + 1/σ - 1), μ fitted to the tokens and the steps
/// (`compute_empirical_mu`), in f32; then 0. Each step's σ and the next.
pub fn schedule(tokens: usize, steps: usize) -> Vec<(f32, f32)> {
    let mu = empirical_mu(tokens, steps);
    let e = mu.exp() as f32;
    let mut sigmas: Vec<f32> = (0..steps)
        .map(|i| {
            let s =
                (1.0 - (1.0 - 1.0 / steps as f64) * i as f64 / (steps - 1).max(1) as f64) as f32;
            e / (e + (1.0 / s - 1.0))
        })
        .collect();
    sigmas.push(0.0);
    sigmas.windows(2).map(|w| (w[0], w[1])).collect()
}

/// μ of the time shift for `tokens` latent tokens and `steps` steps.
fn empirical_mu(tokens: usize, steps: usize) -> f64 {
    let (a1, b1) = (8.738_095_24e-5, 1.898_333_33);
    let (a2, b2) = (0.000_169_27, 0.456_666_66);
    let l = tokens as f64;
    if tokens > 4300 {
        return a2 * l + b2;
    }
    let m200 = a2 * l + b2;
    let m10 = a1 * l + b1;
    let a = (m200 - m10) / 190.0;
    let b = m200 - 200.0 * a;
    a * steps as f64 + b
}

/// The timestep the transformer embeds for σ, as the reference pipeline's
/// bf16 run makes it: σ · 1000 to bf16, over 1000 in bf16, times 1000 in
/// bf16.
pub(crate) fn embedded_time(sigma: f32) -> f32 {
    let t = bf16(sigma * 1000.0);
    bf16(bf16(t / 1000.0) * 1000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generator's first outputs for seed 5489 (C++'s default seed),
    /// and the 10000th (the standard's check value).
    #[test]
    fn the_generator_is_mt19937() {
        let mut r = Mt19937::new(5489);
        assert_eq!(r.next(), 3_499_211_612);
        let mut r = Mt19937::new(5489);
        let mut last = 0;
        for _ in 0..10_000 {
            last = r.next();
        }
        assert_eq!(last, 4_123_659_995);
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(bf16(1.0), 1.0);
        assert_eq!(bf16(1.0 + 1.0 / 256.0), 1.0);
        assert_eq!(bf16(1.0 + 3.0 / 256.0), 1.0 + 4.0 / 256.0);
        assert_eq!(bf16(-957.3), -956.0);
    }

    /// The schedule of a 1344 × 768 picture (4032 tokens): σ falls from 1
    /// to 0 in four steps, shifted towards 1.
    #[test]
    fn the_schedule_shifts_towards_one() {
        let s = schedule(4032, 4);
        assert_eq!(s.len(), 4);
        assert_eq!(s[0].0, 1.0);
        assert_eq!(s[3].1, 0.0);
        for w in s.windows(2) {
            assert_eq!(w[0].1, w[1].0);
            assert!(w[1].0 < w[0].0);
        }
        assert!(s[3].0 > 0.25 && s[2].0 > 0.5);
    }
}
