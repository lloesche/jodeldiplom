//! The results screen's sticker and wax seal (what is stuck on and pressed
//! into the result diploma), on the GPU in the
//! stage's shader: a star sticker (a five-pointed foil star, each point's
//! two facets lit from the top left, a glint sweeping over it), a wax seal
//! (a blob of wax, its edge uneven and its lip domed, a die pressed into
//! it: a ring of beads and a star standing up in its middle, all lit from
//! the top left) with a sheen sweeping across, and a burst of soft rays
//! turning behind the diploma, light. Each is a signed distance and a tone
//! in the shape's own units (its size 1); the processor gives the GPU where
//! each is, its colours and where its glint is, nothing more.
//!
//! These functions are the shader's reference ([`wgsl`]), held to it on
//! the GPU (`crate::gui::shader_check`).

use std::f32::consts::{FRAC_1_SQRT_2, PI, TAU};

use crate::gui::wgsl_math::{clamp01, smoothstep};

/// The angle between two of a star's points.
const POINT: f32 = TAU / 5.0;
/// Where the light comes from: the top left (an angle from straight up,
/// clockwise).
const LIGHT: f32 = -PI / 4.0;

/// A line's ink or light `d` off it, `w` wide.
pub(super) fn line(d: f32, w: f32) -> f32 {
    (-(d / w) * (d / w)).exp()
}

fn sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// A five-pointed star's signed distance at `(x, y)` (up positive), its
/// points `r` out, its inner corners `rf` of the way.
pub fn star(x: f32, y: f32, r: f32, rf: f32) -> f32 {
    let k1 = (0.809_017, -0.587_785_25);
    let k2 = (-k1.0, k1.1);
    let (mut px, mut py) = (x.abs(), y);
    let d1 = 2.0 * (k1.0 * px + k1.1 * py).max(0.0);
    px -= d1 * k1.0;
    py -= d1 * k1.1;
    let d2 = 2.0 * (k2.0 * px + k2.1 * py).max(0.0);
    px -= d2 * k2.0;
    py -= d2 * k2.1;
    px = px.abs();
    py -= r;
    let (bx, by) = (rf * -k1.1, rf * k1.0 - 1.0);
    let h = ((px * bx + py * by) / (bx * bx + by * by)).clamp(0.0, r);
    let (qx, qy) = (px - bx * h, py - by * h);
    (qx * qx + qy * qy).sqrt() * sign(py * bx - px * by)
}

/// A star sticker at `(x, y)` (its points 1 out, one of them up, its
/// middle at 0, down positive): how far outside it (negative inside), its
/// tone (each point's two facets, the one facing the light brighter, its
/// ridges lighter, its valleys and its rim darker) and how far a glint at
/// `glint` (across it from its top left, -1.5 to 1.5) lights it.
pub fn sticker_at(x: f32, y: f32, glint: f32) -> [f32; 3] {
    let d = star(x, -y, 1.0, 0.5) - 0.05;
    let r = (x * x + y * y).sqrt();
    let a = x.atan2(-y);
    let k = (a / POINT + 0.5).floor();
    let off = a - k * POINT;
    let side = if off < 0.0 { -1.0 } else { 1.0 };
    let facing = k * POINT + side * 1.15;
    let mut tone = 0.8 + 0.32 * (facing - LIGHT).cos();
    tone += 0.3 * line(off * r, 0.03) * clamp01(r * 3.0);
    tone -= 0.15 * line((POINT * 0.5 - off.abs()) * r, 0.03);
    tone *= 1.0 - 0.35 * line(d + 0.02, 0.05);
    let sweep = line((x * 0.6 + y * 0.8) - glint, 0.22) * 0.75;
    [d, tone, sweep]
}

/// The wax's edge, out from its middle at the angle `a`: uneven, as wax
/// poured.
pub fn wax_edge(a: f32) -> f32 {
    1.0 - 0.05 * (0.5 + 0.5 * (7.0 * a + 0.7).sin())
        - 0.035 * (0.5 + 0.5 * (12.0 * a + 2.3).sin())
        - 0.02 * (0.5 + 0.5 * (19.0 * a + 4.1).sin())
}

/// How high the seal stands at `(x, y)` (its radius 1): its lip domed, the
/// die's disc pressed in, a ring of beads and a star standing up in it.
fn wax_height(x: f32, y: f32) -> f32 {
    let r = (x * x + y * y).sqrt();
    let a = y.atan2(x);
    let d = r - wax_edge(a);
    let mut h = smoothstep(0.0, 0.2, -d) * (0.55 + 0.1 * (1.0 - r * r));
    h -= 0.3 * (1.0 - smoothstep(0.64, 0.68, r));
    let step = TAU / 24.0;
    let q = (a - (a / step + 0.5).floor() * step) * r;
    let dr = r - 0.55;
    h += 0.1 * (-(dr * dr + q * q) / (0.028 * 0.028)).exp();
    h + 0.18 * (1.0 - smoothstep(-0.02, 0.02, star(x, -y, 0.36, 0.45)))
}

/// A wax seal at `(x, y)` (its radius 1, its middle at 0, down positive):
/// how far outside it (negative inside), its tone (0 its deep colour, 1
/// its light one: higher and facing the top left, lighter) and a sheen at
/// `sheen` (across it from its top left, -1.5 to 1.5).
pub fn seal_at(x: f32, y: f32, sheen: f32) -> [f32; 3] {
    let r = (x * x + y * y).sqrt();
    let d = r - wax_edge(y.atan2(x));
    let e = 0.01;
    let gx = (wax_height(x + e, y) - wax_height(x - e, y)) / (2.0 * e);
    let gy = (wax_height(x, y + e) - wax_height(x, y - e)) / (2.0 * e);
    let lit = 0.6 * gx + 0.8 * gy;
    let tone = (0.45 + 0.6 * wax_height(x, y) + 0.12 * lit).clamp(0.0, 1.35);
    let shine = line(x * FRAC_1_SQRT_2 + y * FRAC_1_SQRT_2 - sheen, 0.2) * 0.5;
    [d, tone, shine]
}

/// A burst of rays at `(x, y)` from its middle, `radius` its reach, `n`
/// of them, turned by `turn` (radians of each ray's cycle): its light, 0
/// to 1, falling off outwards.
pub fn rays_at(x: f32, y: f32, radius: f32, n: f32, turn: f32) -> f32 {
    let r = (x * x + y * y).sqrt();
    let a = if r < 0.5 { 0.0 } else { y.atan2(x) };
    let s = 0.5 + 0.5 * (n * a + turn).cos();
    let fall = (-r / radius.max(1.0) * 2.5).exp();
    (0.15 + 0.85 * s * s * s) * fall
}

/// The functions above in WGSL, and the shapes drawn with them: for the
/// stage's shader (which gives them `glow`, `over_plus`, `out_rgb` and
/// [`super::paper`]'s `turn_about`).
pub fn wgsl() -> String {
    format!("const POINT: f32 = {POINT:?};\nconst LIGHT: f32 = {LIGHT:?};\n{SHADER}")
}

const SHADER: &str = r"
fn line(d: f32, w: f32) -> f32 {
    return exp(-(d / w) * (d / w));
}

fn star(x: f32, y: f32, r: f32, rf: f32) -> f32 {
    let k1 = vec2<f32>(0.809017, -0.58778525);
    let k2 = vec2<f32>(-k1.x, k1.y);
    var p = vec2<f32>(abs(x), y);
    p = p - 2.0 * max(dot(k1, p), 0.0) * k1;
    p = p - 2.0 * max(dot(k2, p), 0.0) * k2;
    p.x = abs(p.x);
    p.y = p.y - r;
    let ba = vec2<f32>(rf * -k1.y, rf * k1.x - 1.0);
    let h = clamp(dot(p, ba) / dot(ba, ba), 0.0, r);
    return length(p - ba * h) * sign(p.y * ba.x - p.x * ba.y);
}

fn sticker_at(x: f32, y: f32, glint: f32) -> vec3<f32> {
    let d = star(x, -y, 1.0, 0.5) - 0.05;
    let r = sqrt(x * x + y * y);
    let a = atan2(x, -y);
    let k = floor(a / POINT + 0.5);
    let off = a - k * POINT;
    let side = select(1.0, -1.0, off < 0.0);
    let facing = k * POINT + side * 1.15;
    var tone = 0.8 + 0.32 * cos(facing - LIGHT);
    tone = tone + 0.3 * line(off * r, 0.03) * clamp(r * 3.0, 0.0, 1.0);
    tone = tone - 0.15 * line((POINT * 0.5 - abs(off)) * r, 0.03);
    tone = tone * (1.0 - 0.35 * line(d + 0.02, 0.05));
    let sweep = line((x * 0.6 + y * 0.8) - glint, 0.22) * 0.75;
    return vec3<f32>(d, tone, sweep);
}

fn wax_edge(a: f32) -> f32 {
    return 1.0 - 0.05 * (0.5 + 0.5 * sin(7.0 * a + 0.7))
        - 0.035 * (0.5 + 0.5 * sin(12.0 * a + 2.3))
        - 0.02 * (0.5 + 0.5 * sin(19.0 * a + 4.1));
}

fn wax_height(x: f32, y: f32) -> f32 {
    let r = sqrt(x * x + y * y);
    let a = atan2(y, x);
    let d = r - wax_edge(a);
    var h = smoothstep(0.0, 0.2, -d) * (0.55 + 0.1 * (1.0 - r * r));
    h = h - 0.3 * (1.0 - smoothstep(0.64, 0.68, r));
    let step = 6.2831855 / 24.0;
    let q = (a - floor(a / step + 0.5) * step) * r;
    let dr = r - 0.55;
    h = h + 0.1 * exp(-(dr * dr + q * q) / (0.028 * 0.028));
    return h + 0.18 * (1.0 - smoothstep(-0.02, 0.02, star(x, -y, 0.36, 0.45)));
}

fn seal_at(x: f32, y: f32, sheen: f32) -> vec3<f32> {
    let r = sqrt(x * x + y * y);
    let d = r - wax_edge(atan2(y, x));
    let e = 0.01;
    let gx = (wax_height(x + e, y) - wax_height(x - e, y)) / (2.0 * e);
    let gy = (wax_height(x, y + e) - wax_height(x, y - e)) / (2.0 * e);
    let lit = 0.6 * gx + 0.8 * gy;
    let tone = clamp(0.45 + 0.6 * wax_height(x, y) + 0.12 * lit, 0.0, 1.35);
    let shine = line(x * 0.70710677 + y * 0.70710677 - sheen, 0.2) * 0.5;
    return vec3<f32>(d, tone, shine);
}

fn rays_at(x: f32, y: f32, radius: f32, n: f32, turn: f32) -> f32 {
    let r = sqrt(x * x + y * y);
    var a = 0.0;
    if r >= 0.5 {
        a = atan2(y, x);
    }
    let s = 0.5 + 0.5 * cos(n * a + turn);
    let fall = exp(-r / max(radius, 1.0) * 2.5);
    return (0.15 + 0.85 * s * s * s) * fall;
}

// `col` covering `cover` of it, over a soft shadow `shade` round it, all
// `a` shown (premultiplied).
fn on_paper(col: vec3<f32>, cover: f32, shade: f32, a: f32) -> vec4<f32> {
    let s = shade * (1.0 - cover);
    return vec4<f32>(out_rgb(col) * cover * a, (cover + s) * a);
}

// What is stuck on and pressed into the diploma (each glowing round it,
// not over it, and casting a soft shadow `params.w` dark): a star sticker
// (`shape.xy` its middle, `shape.z` its size, turned `shape.w`, in
// `color`, glowing `params.y` in `color2`, its glint at `params.z`), a
// seal (`shape.xy` its middle, `shape.z` its radius; from `color2` deep
// to `color` light, glowing `params.y`, its sheen at `params.z`) and rays
// (round `shape.xy`, `shape.z` their reach, `shape.w` of them, turned
// `params.y`, in `color`: light).
fn seal_shape(o: Out, p: vec2<f32>, k: u32) -> vec4<f32> {
    let size = max(o.shape.z, 0.5);
    if k == K_RAYS {
        let q = p - o.shape.xy;
        let light = rays_at(q.x, q.y, o.shape.z, o.shape.w, o.params.y);
        return vec4<f32>(out_rgb(o.color.rgb) * light * o.color.a, 0.0);
    }
    if k == K_STAR {
        let l = (turn_about(p, o.shape.xy, -o.shape.w) - o.shape.xy) / size;
        let g = sticker_at(l.x, l.y, o.params.z);
        let d = g.x * size;
        let cover = clamp(0.5 - d, 0.0, 1.0);
        let col = min(o.color.rgb * g.y + vec3<f32>(g.z), vec3<f32>(1.0));
        let ls = l - vec2<f32>(0.05, 0.1);
        let ds = (star(ls.x, -ls.y, 1.0, 0.5) - 0.05) * size;
        let shade = o.params.w * exp(-max(ds, 0.0) / (size * 0.12));
        let halo = glow(d, o.params.y) * o.color2.a * (1.0 - cover);
        let lit = on_paper(col, cover, shade, o.color.a);
        return lit + vec4<f32>(out_rgb(o.color2.rgb * halo), 0.0);
    }
    let l = (p - o.shape.xy) / size;
    let m = seal_at(l.x, l.y, o.params.z);
    let d = m.x * size;
    let cover = clamp(0.5 - d, 0.0, 1.0);
    let col = min(mix(o.color2.rgb, o.color.rgb, m.y) + vec3<f32>(m.z), vec3<f32>(1.0));
    let ls = l - vec2<f32>(0.03, 0.08);
    let ds = (length(ls) - wax_edge(atan2(ls.y, ls.x))) * size;
    let shade = o.params.w * exp(-max(ds, 0.0) / (size * 0.07));
    let halo = glow(d, o.params.y) * o.color.a * 0.6 * (1.0 - cover);
    let lit = on_paper(col, cover, shade, o.color.a);
    return lit + vec4<f32>(out_rgb(o.color.rgb * halo), 0.0);
}
";

#[cfg(test)]
mod tests {
    use super::*;

    /// The sticker is a star, one point up; the facet facing the light is
    /// brighter than the one turned away; the glint lights where it is.
    #[test]
    fn the_sticker_is_a_lit_star() {
        assert!(sticker_at(0.0, -0.9, 9.0)[0] < 0.0);
        assert!(sticker_at(0.0, 0.9, 9.0)[0] > 0.0);
        assert!(sticker_at(0.0, 0.0, 9.0)[0] < -0.3);
        // The top point's left facet faces the top left.
        let left = sticker_at(-0.08, -0.5, 9.0)[1];
        let right = sticker_at(0.08, -0.5, 9.0)[1];
        assert!(left > right + 0.2, "{left} {right}");
        assert!(sticker_at(0.0, 0.0, 0.0)[2] > 0.7 && sticker_at(0.0, 0.0, 1.4)[2] < 0.01);
    }

    #[test]
    fn the_seal_and_its_star() {
        // Its edge uneven, near its radius.
        for k in 0..16 {
            let a = k as f32 * 0.4;
            let e = wax_edge(a);
            assert!((0.89..=1.0).contains(&e), "{e}");
            assert!(seal_at(e * a.cos() * 0.98, e * a.sin() * 0.98, 9.0)[0] < 0.0);
        }
        assert!(seal_at(1.05, 0.0, 9.0)[0] > 0.0);
        // The star's points out, its corners in.
        assert!(star(0.0, 0.56, 0.56, 0.45).abs() < 1e-4);
        assert!(star(0.0, 0.0, 0.56, 0.45) < -0.15);
        assert!(star(0.0, -0.4, 0.56, 0.45) > 0.0);
        // The star stands up out of the pressed disc; the lip lit at the
        // top left, in shade at the bottom right.
        assert!(seal_at(0.0, -0.05, 9.0)[1] > seal_at(0.3, 0.42, 9.0)[1]);
        let lip = |a: f32| seal_at(0.88 * a.cos(), 0.88 * a.sin(), 9.0)[1];
        assert!(lip(-2.2) > lip(0.9), "{} {}", lip(-2.2), lip(0.9));
    }

    #[test]
    fn the_rays_turn_and_fall_off() {
        let a = rays_at(100.0, 0.0, 300.0, 12.0, 0.0);
        let b = rays_at(100.0, 0.0, 300.0, 12.0, std::f32::consts::PI);
        assert!(a > 0.4 && b < a * 0.3, "{a} {b}");
        assert!(rays_at(600.0, 0.0, 300.0, 12.0, 0.0) < a * 0.3);
        assert!(rays_at(0.0, 0.0, 300.0, 12.0, 0.0).is_finite());
    }
}
