//! WGSL's types and builtins on the processor, and the helpers every pass
//! shares, written twice: here, and in WGSL ([`wgsl`], each `lk_` and the
//! same name). A pass's reference reads like its shader line for line, so
//! the two agree wherever they are held to each other (`tests/gpu.rs`,
//! the player's `shader_check`). The hash is integer arithmetic, the same bits on
//! both; the noise is built on it. The player's shaders' references are
//! written in the same builtins, with the shapes and the encoding of light
//! their own shaders share ([`round_box`], [`srgb_to_linear`],
//! [`linear_to_srgb`]): one home for them all.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// WGSL's `vec2<f32>`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V2 {
    /// Across.
    pub x: f32,
    /// Down.
    pub y: f32,
}

/// WGSL's `vec3<f32>`: a colour.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V3 {
    /// Red.
    pub x: f32,
    /// Green.
    pub y: f32,
    /// Blue.
    pub z: f32,
}

/// `vec2<f32>(x, y)`.
pub const fn v2(x: f32, y: f32) -> V2 {
    V2 { x, y }
}

/// `vec3<f32>(x, y, z)`.
pub const fn v3(x: f32, y: f32, z: f32) -> V3 {
    V3 { x, y, z }
}

/// `vec3<f32>(a)`.
pub const fn s3(a: f32) -> V3 {
    V3 { x: a, y: a, z: a }
}

macro_rules! ops {
    ($t:ident { $($f:ident),* }) => {
        impl Add for $t { type Output = $t; fn add(self, o: $t) -> $t { $t { $($f: self.$f + o.$f),* } } }
        impl Sub for $t { type Output = $t; fn sub(self, o: $t) -> $t { $t { $($f: self.$f - o.$f),* } } }
        impl Mul for $t { type Output = $t; fn mul(self, o: $t) -> $t { $t { $($f: self.$f * o.$f),* } } }
        impl Div for $t { type Output = $t; fn div(self, o: $t) -> $t { $t { $($f: self.$f / o.$f),* } } }
        impl Add<f32> for $t { type Output = $t; fn add(self, o: f32) -> $t { $t { $($f: self.$f + o),* } } }
        impl Sub<f32> for $t { type Output = $t; fn sub(self, o: f32) -> $t { $t { $($f: self.$f - o),* } } }
        impl Mul<f32> for $t { type Output = $t; fn mul(self, o: f32) -> $t { $t { $($f: self.$f * o),* } } }
        impl Div<f32> for $t { type Output = $t; fn div(self, o: f32) -> $t { $t { $($f: self.$f / o),* } } }
        impl Mul<$t> for f32 { type Output = $t; fn mul(self, o: $t) -> $t { $t { $($f: self * o.$f),* } } }
        impl Neg for $t { type Output = $t; fn neg(self) -> $t { $t { $($f: -self.$f),* } } }
        impl $t {
            /// Each part through `f`.
            pub fn map(self, f: impl Fn(f32) -> f32) -> $t { $t { $($f: f(self.$f)),* } }
            /// WGSL's `dot`.
            pub fn dot(self, o: $t) -> f32 { 0.0 $(+ self.$f * o.$f)* }
            /// WGSL's `length`.
            pub fn length(self) -> f32 { self.dot(self).sqrt() }
            /// WGSL's `floor`.
            pub fn floor(self) -> $t { self.map(f32::floor) }
            /// WGSL's `fract`.
            pub fn fract(self) -> $t { self.map(fract) }
            /// WGSL's `abs`.
            pub fn abs(self) -> $t { self.map(f32::abs) }
            /// WGSL's `max` with a number.
            pub fn max(self, m: f32) -> $t { self.map(|v| v.max(m)) }
            /// WGSL's `min` with a number.
            pub fn min(self, m: f32) -> $t { self.map(|v| v.min(m)) }
            /// WGSL's `clamp` between numbers.
            pub fn clamp(self, lo: f32, hi: f32) -> $t { self.map(|v| v.max(lo).min(hi)) }
            /// WGSL's `max` of two.
            pub fn vmax(self, o: $t) -> $t { $t { $($f: self.$f.max(o.$f)),* } }
            /// WGSL's `min` of two.
            pub fn vmin(self, o: $t) -> $t { $t { $($f: self.$f.min(o.$f)),* } }
        }
    };
}
ops!(V2 { x, y });
ops!(V3 { x, y, z });

impl V3 {
    /// `.xy`.
    pub fn xy(self) -> V2 {
        v2(self.x, self.y)
    }
}

impl From<[f32; 3]> for V3 {
    fn from([x, y, z]: [f32; 3]) -> V3 {
        v3(x, y, z)
    }
}

impl From<V3> for [f32; 3] {
    fn from(c: V3) -> [f32; 3] {
        [c.x, c.y, c.z]
    }
}

/// WGSL's `fract`.
pub fn fract(x: f32) -> f32 {
    x - x.floor()
}

/// WGSL's `clamp(x, 0.0, 1.0)`.
pub fn sat(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// WGSL's `mix` of numbers.
pub fn mix(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

/// WGSL's `mix` of colours by a number, the colours a [`V3`] or three
/// numbers alike.
pub fn mix3<C: Into<V3> + From<V3>>(a: C, b: C, t: f32) -> C {
    C::from(a.into() * (1.0 - t) + b.into() * t)
}

/// WGSL's `mix` of points by a number.
pub fn mix2(a: V2, b: V2, t: f32) -> V2 {
    a * (1.0 - t) + b * t
}

/// WGSL's `smoothstep`.
pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = sat((x - e0) / (e1 - e0));
    t * t * (3.0 - 2.0 * t)
}

/// WGSL's `step`: 1 where `x` is at least `edge`.
pub fn step(edge: f32, x: f32) -> f32 {
    if x >= edge {
        1.0
    } else {
        0.0
    }
}

/// A colour's luma (Rec. 709 weights, on the encoded values the passes
/// work in).
pub fn luma(c: V3) -> f32 {
    c.dot(v3(0.2126, 0.7152, 0.0722))
}

/// A colour's relative luminance in linear light (its encoded values
/// decoded as sRGB, none below 0; over 1 as the curve goes on): what the
/// eye's contrast against white and the screen's light are reckoned
/// from, the lyric band's ceiling and the guards' measures alike.
pub fn lum(c: V3) -> f32 {
    c.max(0.0)
        .map(srgb_to_linear)
        .dot(v3(0.2126, 0.7152, 0.0722))
}

/// An sRGB-encoded value (0 to 1) as linear light (`lk_decode` in WGSL).
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear light (0 to 1) encoded as sRGB: [`srgb_to_linear`]'s inverse
/// (`lk_encode` in WGSL).
pub fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// How far `p` is outside a box `half` its size round the origin, its
/// corners `r` round (negative inside): the player's shaders' `round_box`.
pub fn round_box(p: V2, half: V2, r: f32) -> f32 {
    let q = p.abs() - half + r;
    (q.x.max(0.0).hypot(q.y.max(0.0))) + q.x.max(q.y).min(0.0) - r
}

/// `c` as grey by `1 - s` (0: grey, 1: as it is, more: more vivid).
pub fn saturate(c: V3, s: f32) -> V3 {
    mix3(s3(luma(c)), c, s)
}

/// 32 bits mixed well (a low-bias integer hash).
pub fn mix32(x: u32) -> u32 {
    let mut h = x;
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    h
}

/// A number 0 to 1 for the lattice point `(x, y)` and the seed `s`.
pub fn hash2(x: i32, y: i32, s: u32) -> f32 {
    let h = mix32((x as u32) ^ mix32((y as u32) ^ mix32(s)));
    (h >> 8) as f32 / 16_777_216.0
}

/// WGSL's `i32(f)` of a floored number.
pub fn cell(f: f32) -> i32 {
    f as i32
}

/// Smooth value noise at `p`, 0 to 1, over the lattice of seed `s`.
pub fn vnoise(p: V2, s: u32) -> f32 {
    let i = p.floor();
    let f = p - i;
    let u = f * f * (s2(3.0) - f * 2.0);
    let (x, y) = (cell(i.x), cell(i.y));
    let a = hash2(x, y, s);
    let b = hash2(x + 1, y, s);
    let c = hash2(x, y + 1, s);
    let d = hash2(x + 1, y + 1, s);
    mix(mix(a, b, u.x), mix(c, d, u.x), u.y)
}

/// Three octaves of [`vnoise`], 0 to 1.
pub fn fbm(p: V2, s: u32) -> f32 {
    0.5 * vnoise(p, s)
        + 0.3 * vnoise(p * 2.03 + v2(17.0, 9.0), s + 1)
        + 0.2 * vnoise(p * 4.07 + v2(31.0, 5.0), s + 2)
}

/// `vec2<f32>(a)`.
pub const fn s2(a: f32) -> V2 {
    V2 { x: a, y: a }
}

/// A hue (turns), saturation and value as a colour.
pub fn hsv(h: f32, s: f32, v: f32) -> V3 {
    let k = (s3(fract(h)) + v3(0.0, 2.0 / 3.0, 1.0 / 3.0)).fract() * 6.0 - 3.0;
    let k = (k.abs() - 1.0).clamp(0.0, 1.0);
    mix3(s3(1.0), k, s) * v
}

/// `c`'s hue turned by `turns`, its luma kept (a turn of its chroma in
/// YIQ).
pub fn hue_turn(c: V3, turns: f32) -> V3 {
    let y = 0.299 * c.x + 0.587 * c.y + 0.114 * c.z;
    let i = 0.596 * c.x - 0.274 * c.y - 0.322 * c.z;
    let q = 0.211 * c.x - 0.523 * c.y + 0.312 * c.z;
    let a = turns * TAU;
    let (sa, ca) = (a.sin(), a.cos());
    let (i2, q2) = (i * ca - q * sa, i * sa + q * ca);
    v3(
        y + 0.956 * i2 + 0.621 * q2,
        y - 0.272 * i2 - 0.647 * q2,
        y - 1.106 * i2 + 1.703 * q2,
    )
}

/// The most light a drawn shape adds ([`add_light`]).
pub const LIGHT_CAP: f32 = 0.85;
/// Where the knee bends ([`knee`]).
pub const KNEE: f32 = 0.5;
/// How far over [`KNEE`] the knee rises at most.
pub const KNEE_ROOM: f32 = 0.2;

/// Light added to `c` as a screen adds it, `light` held under
/// [`LIGHT_CAP`]: a drawn shape over a bright picture lightens it, never
/// to white.
pub fn add_light(c: V3, light: V3) -> V3 {
    screen(c.clamp(0.0, 1.0), light.clamp(0.0, LIGHT_CAP))
}

/// A soft knee over [`KNEE`]: as it is below, then rising ever more
/// slowly to [`KNEE`] and [`KNEE_ROOM`] (an echo fed back piles up toward
/// a grey, never to white).
pub fn knee(x: f32) -> f32 {
    if x > KNEE {
        KNEE + KNEE_ROOM * (1.0 - (-(x - KNEE) / KNEE_ROOM).exp())
    } else {
        x
    }
}

/// A whole turn, radians.
pub const TAU: f32 = std::f32::consts::TAU;

/// The screen blend of `a` and `b`.
pub fn screen(a: V3, b: V3) -> V3 {
    s3(1.0) - (s3(1.0) - a) * (s3(1.0) - b)
}

/// `p` turned by `a` radians.
pub fn turn(p: V2, a: f32) -> V2 {
    let (s, c) = (a.sin(), a.cos());
    v2(c * p.x - s * p.y, s * p.x + c * p.y)
}

/// The helpers in WGSL, as [`self`] has them on the processor, their
/// numbers ([`LIGHT_CAP`], [`KNEE`], [`KNEE_ROOM`]) written in from here.
pub fn wgsl() -> String {
    format!(
        "const LK_LIGHT_CAP: f32 = {LIGHT_CAP:?};\nconst LK_KNEE: f32 = {KNEE:?};\nconst LK_KNEE_ROOM: f32 = {KNEE_ROOM:?};{HELPERS}"
    )
}

/// [`wgsl`]'s helpers.
const HELPERS: &str = r"
const LK_TAU: f32 = 6.2831853;
fn lk_sat(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn lk_luma(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); }
fn lk_encode(v: vec3<f32>) -> vec3<f32> {
    let x = max(v, vec3<f32>(0.0));
    return select(12.92 * x, 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055, x > vec3<f32>(0.0031308));
}
fn lk_decode(c: vec3<f32>) -> vec3<f32> {
    let x = max(c, vec3<f32>(0.0));
    return select(x / 12.92, pow((x + 0.055) / 1.055, vec3<f32>(2.4)), x > vec3<f32>(0.04045));
}
fn lk_lum(c: vec3<f32>) -> f32 { return dot(lk_decode(c), vec3<f32>(0.2126, 0.7152, 0.0722)); }
fn lk_saturate(c: vec3<f32>, s: f32) -> vec3<f32> { return mix(vec3<f32>(lk_luma(c)), c, s); }
fn lk_mix32(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 16u);
    h = h * 0x7feb352du;
    h = h ^ (h >> 15u);
    h = h * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return h;
}
fn lk_hash2(x: i32, y: i32, s: u32) -> f32 {
    let h = lk_mix32(bitcast<u32>(x) ^ lk_mix32(bitcast<u32>(y) ^ lk_mix32(s)));
    return f32(h >> 8u) / 16777216.0;
}
fn lk_vnoise(p: vec2<f32>, s: u32) -> f32 {
    let i = floor(p);
    let f = p - i;
    let u = f * f * (vec2<f32>(3.0) - f * 2.0);
    let x = i32(i.x);
    let y = i32(i.y);
    let a = lk_hash2(x, y, s);
    let b = lk_hash2(x + 1, y, s);
    let c = lk_hash2(x, y + 1, s);
    let d = lk_hash2(x + 1, y + 1, s);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
fn lk_fbm(p: vec2<f32>, s: u32) -> f32 {
    return 0.5 * lk_vnoise(p, s)
        + 0.3 * lk_vnoise(p * 2.03 + vec2<f32>(17.0, 9.0), s + 1u)
        + 0.2 * lk_vnoise(p * 4.07 + vec2<f32>(31.0, 5.0), s + 2u);
}
fn lk_hsv(h: f32, s: f32, v: f32) -> vec3<f32> {
    var k = fract(vec3<f32>(fract(h)) + vec3<f32>(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0;
    k = clamp(abs(k) - 1.0, vec3<f32>(0.0), vec3<f32>(1.0));
    return mix(vec3<f32>(1.0), k, s) * v;
}
fn lk_hue_turn(c: vec3<f32>, turns: f32) -> vec3<f32> {
    let y = 0.299 * c.x + 0.587 * c.y + 0.114 * c.z;
    let i = 0.596 * c.x - 0.274 * c.y - 0.322 * c.z;
    let q = 0.211 * c.x - 0.523 * c.y + 0.312 * c.z;
    let a = turns * LK_TAU;
    let sa = sin(a);
    let ca = cos(a);
    let i2 = i * ca - q * sa;
    let q2 = i * sa + q * ca;
    return vec3<f32>(y + 0.956 * i2 + 0.621 * q2, y - 0.272 * i2 - 0.647 * q2, y - 1.106 * i2 + 1.703 * q2);
}
fn lk_screen(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> { return vec3<f32>(1.0) - (vec3<f32>(1.0) - a) * (vec3<f32>(1.0) - b); }
fn lk_add_light(c: vec3<f32>, light: vec3<f32>) -> vec3<f32> {
    return lk_screen(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), clamp(light, vec3<f32>(0.0), vec3<f32>(LK_LIGHT_CAP)));
}
fn lk_knee(x: vec3<f32>) -> vec3<f32> {
    return select(x, LK_KNEE + LK_KNEE_ROOM * (1.0 - exp(-(x - LK_KNEE) / LK_KNEE_ROOM)), x > vec3<f32>(LK_KNEE));
}
fn lk_turn(p: vec2<f32>, a: f32) -> vec2<f32> {
    let s = sin(a);
    let c = cos(a);
    return vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
}
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_helpers_do_what_they_say() {
        assert_eq!(hsv(0.0, 1.0, 1.0), v3(1.0, 0.0, 0.0));
        let c = v3(0.3, 0.6, 0.2);
        let back = hue_turn(hue_turn(c, 0.3), -0.3);
        assert!((back - c).length() < 1e-3, "{back:?}");
        assert!((luma(hue_turn(c, 0.25)) - luma(c)).abs() < 0.05);
        let n: Vec<f32> = (0..100)
            .map(|i| vnoise(v2(i as f32 * 0.37, 1.3), 7))
            .collect();
        assert!(n.iter().all(|v| (0.0..=1.0).contains(v)));
        assert!(n.iter().any(|&v| v < 0.3) && n.iter().any(|&v| v > 0.7));
        assert_eq!(hash2(3, -4, 9), hash2(3, -4, 9));
        assert_ne!(hash2(3, -4, 9), hash2(3, -4, 10));
        assert_eq!(smoothstep(0.0, 1.0, 0.5), 0.5);
        assert!(round_box(v2(0.0, 0.0), v2(10.0, 5.0), 1.0) < -4.9);
        assert!((round_box(v2(13.0, 0.0), v2(10.0, 5.0), 1.0) - 3.0).abs() < 1e-5);
    }

    #[test]
    fn srgb_goes_there_and_back() {
        for i in 0..=255u8 {
            let c = f32::from(i) / 255.0;
            let back = linear_to_srgb(srgb_to_linear(c));
            assert!((back - c).abs() < 1e-5, "{i}: {back}");
        }
    }
}
