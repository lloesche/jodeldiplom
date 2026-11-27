//! The backdrop's shader pieces both its passes share, the ground's and
//! the scenes': the smooth cubic sampling's taps ([`bspline`], here for
//! the processor and in WGSL ([`BSPLINE`]), held to each other in
//! `super::super::shader_check`; the ground's stretch) and the passes'
//! triangle ([`SCREEN`]). The scenes' maths is their own
//! ([`super::scenes::look`]).

/// A smooth cubic (B-spline) read of a texture at `f`, the share of the
/// way from one texel to the next: the two bilinear reads' weights and
/// where they are, each from the first texel (`[g0, g1, h0, h1]`).
pub fn bspline(f: f32) -> [f32; 4] {
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    let w3 = f3 / 6.0;
    let (g0, g1) = (w0 + w1, w2 + w3);
    [g0, g1, w1 / g0 - 1.0, w3 / g1 + 1.0]
}

/// The backdrop's passes' triangle over their whole target (the scenes',
/// the ground's and the levels' stretch), its corners read as the
/// target's.
pub const SCREEN: &str = r"
struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> Out {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: Out;
    o.pos = vec4<f32>(p * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
    o.uv = vec2<f32>(p.x, 1.0 - p.y);
    return o;
}
";

/// [`bspline`] in WGSL.
pub const BSPLINE: &str = r"
fn bspline(f: f32) -> vec4<f32> {
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    let w3 = f3 / 6.0;
    let g0 = w0 + w1;
    let g1 = w2 + w3;
    return vec4<f32>(g0, g1, w1 / g0 - 1.0, w3 / g1 + 1.0);
}
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cubic_read_weighs_its_four_texels_as_a_b_spline() {
        for f in [0.0, 0.25, 0.5, 0.9] {
            let [g0, g1, h0, h1] = bspline(f);
            assert!((g0 + g1 - 1.0).abs() < 1e-5);
            // The reads fall between the texels either side of the two.
            assert!(
                (-1.0..=0.0).contains(&h0) && (1.0..=2.0).contains(&h1),
                "{f}"
            );
        }
        // Halfway: alike either side.
        let [g0, g1, h0, h1] = bspline(0.5);
        assert!((g0 - g1).abs() < 1e-6 && (h0 + h1 - 1.0).abs() < 1e-6);
    }
}
