//! IEEE half precision (binary16) and bfloat16 to and from `f32`.

/// An IEEE half as f32: exact, a NaN keeping its payload.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let frac = u32::from(h & 0x3ff);
    let bits = match (exp, frac) {
        (0, 0) => sign,
        (0, f) => {
            // Subnormal: normalised, its leading one shifted out of the
            // fraction.
            let shift = f.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | ((f << shift) & 0x3ff) << 13
        }
        (31, f) => sign | 0x7f80_0000 | (f << 13),
        (e, f) => sign | ((e + 112) << 23) | (f << 13),
    };
    f32::from_bits(bits)
}

/// f32 to the nearest IEEE half (ties to even), as its bits: too large
/// becomes infinite, too small 0, a NaN a quiet NaN.
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if frac != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // Subnormal (or zero): the mantissa with its leading one, shifted.
        if e < -10 {
            return sign;
        }
        let m = frac | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1u32 << shift) - 1);
        let mut v = m >> shift;
        if rest > half || (rest == half && v & 1 == 1) {
            v += 1;
        }
        return sign | v as u16;
    }
    let mut v = ((e as u32) << 10) | (frac >> 13);
    let rest = frac & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && v & 1 == 1) {
        v += 1; // may carry into the exponent, as it should
    }
    sign | v as u16
}

/// A bfloat16 as f32: its bits are the f32's top half.
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_read_exactly() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x03ff), 1023.0 * 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
    }

    #[test]
    fn every_half_comes_back_through_f32() {
        for h in 0..=u16::MAX {
            let x = f16_to_f32(h);
            if x.is_nan() {
                assert!(f16_to_f32(f32_to_f16(x)).is_nan());
            } else {
                assert_eq!(f32_to_f16(x), h, "{h:#06x}");
            }
        }
    }

    #[test]
    fn rounding_is_to_the_nearest_even() {
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert_eq!(f32_to_f16(1.0e6), 0x7c00);
        assert_eq!(f32_to_f16(1.0e-9), 0);
        // Halfway between 1 and the next half: to the even one, 1.
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3c00);
        // Just past halfway: up.
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11) + 2f32.powi(-20)), 0x3c01);
        // The largest half's rounding carries into infinity.
        assert_eq!(f32_to_f16(65520.0), 0x7c00);
    }
}
