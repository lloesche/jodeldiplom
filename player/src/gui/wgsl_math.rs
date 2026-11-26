//! WGSL's builtins and the shaders' shared shapes on the processor: the
//! twins the shaders' references are written in ([`fract`], [`clamp01`],
//! [`smoothstep`], [`mix3`], [`round_box`]), and sRGB's encoding of light
//! ([`srgb_to_linear`], [`linear_to_srgb`]). One home for them all, so a
//! reference and its shader agree to the bit wherever they are used.

/// WGSL's `fract`: what `x` has past its floor.
pub fn fract(x: f32) -> f32 {
    x - x.floor()
}

/// `x` held to 0 to 1 (WGSL's `saturate`).
pub fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// WGSL's `smoothstep`: `x` from 0 below `e0` to 1 above `e1`, smoothly.
pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = clamp01((x - e0) / (e1 - e0));
    t * t * (3.0 - 2.0 * t)
}

/// WGSL's `mix` of two colours: `a` towards `b` by `t`.
pub fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t)
}

/// How far `p` is outside a box `half` its size round the origin, its
/// corners `r` round (negative inside): the shaders' `round_box`.
pub fn round_box(p: (f32, f32), half: (f32, f32), r: f32) -> f32 {
    let (qx, qy) = (p.0.abs() - half.0 + r, p.1.abs() - half.1 + r);
    (qx.max(0.0).hypot(qy.max(0.0))) + qx.max(qy).min(0.0) - r
}

/// An sRGB-encoded value (0 to 1) as linear light.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear light (0 to 1) encoded as sRGB: [`srgb_to_linear`]'s inverse.
pub fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_goes_there_and_back() {
        for i in 0..=255u8 {
            let c = f32::from(i) / 255.0;
            let back = linear_to_srgb(srgb_to_linear(c));
            assert!((back - c).abs() < 1e-5, "{i}: {back}");
        }
        assert_eq!(smoothstep(0.0, 1.0, 0.5), 0.5);
        assert!(round_box((0.0, 0.0), (10.0, 5.0), 1.0) < -4.9);
        assert!((round_box((13.0, 0.0), (10.0, 5.0), 1.0) - 3.0).abs() < 1e-5);
    }
}
