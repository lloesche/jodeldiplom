//! What sparkles: a number that looks random (a seeded hash), the same
//! for the same seed (and a particle's own, [`rand`]), and a colour mixed.

/// White, sRGB.
pub const WHITE: [f32; 3] = [1.0, 1.0, 1.0];
/// A whole turn, radians.
pub const TAU: f32 = std::f32::consts::TAU;

/// A number from 0 to 1 that looks random, the same for the same `i`.
pub fn noise(i: u32) -> f32 {
    let mut x = i.wrapping_mul(0x9E37_79B9) ^ 0x85EB_CA6B;
    x ^= x >> 15;
    x = x.wrapping_mul(0x2C1B_3C6D);
    x ^= x >> 12;
    x = x.wrapping_mul(0x297A_2D39);
    x ^= x >> 15;
    (x >> 8) as f32 / (1u32 << 24) as f32
}

/// Number `j` of quad `q` of a burst of `seed` (24 bits): [`noise`], as
/// the particles' shader has it (`r`).
pub fn rand(seed: u32, q: u32, j: u32) -> f32 {
    noise((seed & 0xFF_FFFF).wrapping_mul(31).wrapping_add(q * 7 + j))
}
