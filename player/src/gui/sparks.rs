//! What sparkles: a number that looks random (a seeded hash), the same
//! for the same seed
//! (and a particle's own, [`rand`]), a colour mixed, and a four-pointed
//! sparkle drawn in pixels.

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

/// `a` towards `b` by `t`.
pub fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t)
}

/// A four-pointed sparkle at (`x`, `y`), its rays `size` long: a bright
/// core and two thin crossed rays, each pixel `put` with its cover times
/// `alpha`.
pub fn sparkle(put: &mut dyn FnMut(usize, usize, f32), x: f32, y: f32, size: f32, alpha: f32) {
    if alpha <= 0.01 || size < 0.5 {
        return;
    }
    let ray = (size * 0.09).max(0.9);
    for py in ((y - size).floor() as i64).max(0)..(y + size).ceil() as i64 {
        for px in ((x - size).floor() as i64).max(0)..(x + size).ceil() as i64 {
            let (dx, dy) = ((px as f32 + 0.5 - x).abs(), (py as f32 + 0.5 - y).abs());
            let core = (1.0 - (dx * dx + dy * dy).sqrt() / (size * 0.28)).clamp(0.0, 1.0);
            let across = (1.0 - dy / ray).clamp(0.0, 1.0) * (1.0 - dx / size).clamp(0.0, 1.0);
            let down = (1.0 - dx / ray).clamp(0.0, 1.0) * (1.0 - dy / size).clamp(0.0, 1.0);
            let cover = core.max(across.max(down) * 0.9);
            if cover > 0.0 {
                put(px as usize, py as usize, alpha * cover);
            }
        }
    }
}
