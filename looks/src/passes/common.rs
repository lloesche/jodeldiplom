//! The helpers the families share, each on the processor with its WGSL
//! beside it: a halftone screen's cells and dots, how much the picture
//! changes round a pixel, hatching, the light round the bright parts, the
//! ordered dither, falling things; all of them in every pass's shader
//! ([`wgsl`]).

use crate::math::*;
use crate::pass::{Ctx, Src};

/// The screen's cell `uv` is in, its dots `s` pixels apart at `a` turns:
/// the cell's middle (as a `uv`) and where in the cell it is (-0.5 to
/// 0.5).
pub(crate) fn screen_cell(u: &Ctx, uv: V2, s: f32, a: f32) -> (V2, V2) {
    let size = u.size2();
    let q = turn(uv * size, a * TAU) / s;
    let c = q.floor() + 0.5;
    (turn(c * s, -a * TAU) / size, q - c)
}

const SCREEN_CELL: &str = r"
fn lk_screen_cell(uv: vec2<f32>, s: f32, a: f32) -> vec4<f32> {
    let q = lk_turn(uv * u.size.xy, a * LK_TAU) / s;
    let c = floor(q) + 0.5;
    return vec4<f32>(lk_turn(c * s, -a * LK_TAU) / u.size.xy, q - c);
}
";

/// A dot `r` across (in cells: 0.71 fills one) at `local`, its edge a
/// pixel soft in cells `s` pixels wide.
pub(crate) fn dot_at(local: V2, r: f32, s: f32) -> f32 {
    let aa = 0.7 / s;
    1.0 - smoothstep(r - aa, r + aa, local.length())
}

const DOT: &str = r"
fn lk_dot(local: vec2<f32>, r: f32, s: f32) -> f32 {
    let aa = 0.7 / s;
    return 1.0 - smoothstep(r - aa, r + aa, length(local));
}
";

/// How much the picture changes across `w` pixels at `uv` (four reads).
pub(crate) fn edges(u: &Ctx, s: &dyn Src, uv: V2, w: f32) -> f32 {
    let d = u.px() * w;
    let gx = luma(s.src(uv + v2(d.x, 0.0))) - luma(s.src(uv - v2(d.x, 0.0)));
    let gy = luma(s.src(uv + v2(0.0, d.y))) - luma(s.src(uv - v2(0.0, d.y)));
    gx.abs() + gy.abs()
}

const EDGES: &str = r"
fn lk_edges(uv: vec2<f32>, w: f32) -> f32 {
    let d = u.size.zw * w;
    let gx = lk_luma(lk_src(uv + vec2<f32>(d.x, 0.0))) - lk_luma(lk_src(uv - vec2<f32>(d.x, 0.0)));
    let gy = lk_luma(lk_src(uv + vec2<f32>(0.0, d.y))) - lk_luma(lk_src(uv - vec2<f32>(0.0, d.y)));
    return abs(gx) + abs(gy);
}
";

/// Hatching's cover at `p` (pixels at 1080 lines) for darkness `dark`:
/// lines `spacing` apart at `angle` radians, wobbling, thicker the darker
/// past `start`.
pub(crate) fn hatch(p: V2, dark: f32, (spacing, angle, start): (f32, f32, f32)) -> f32 {
    let q = turn(p, angle);
    let t = (fract(q.y / spacing + 0.25 * vnoise(q * 0.04, 301)) - 0.5).abs() * 2.0;
    let w = sat((dark - start) * 1.6) * 0.7;
    let aa = 1.2 / spacing;
    (1.0 - smoothstep(w - aa, w + aa, t)) * step(0.001, w)
}

const HATCH: &str = r"
fn lk_hatch(p: vec2<f32>, dark: f32, spacing: f32, angle: f32, start: f32) -> f32 {
    let q = lk_turn(p, angle);
    let t = abs(fract(q.y / spacing + 0.25 * lk_vnoise(q * 0.04, 301u)) - 0.5) * 2.0;
    let w = lk_sat((dark - start) * 1.6) * 0.7;
    let aa = 1.2 / spacing;
    return (1.0 - smoothstep(w - aa, w + aa, t)) * step(0.001, w);
}
";

/// The light round the bright parts, from twelve reads in two rings
/// `radius` pixels out: what is over `threshold`, on average.
pub(crate) fn ring_light(u: &Ctx, s: &dyn Src, uv: V2, radius: f32, threshold: f32) -> V3 {
    let px = u.px();
    let mut acc = s3(0.0);
    for i in 0..12u32 {
        let a = i as f32 * (TAU / 12.0) + 0.3;
        let r = radius * (0.45 + 0.55 * (i % 2) as f32);
        let tap = s.src(uv + v2(a.cos(), a.sin()) * px * r);
        acc = acc + (tap - threshold).max(0.0);
    }
    acc / 12.0
}

const RING_LIGHT: &str = r"
fn lk_ring_light(uv: vec2<f32>, radius: f32, threshold: f32) -> vec3<f32> {
    var acc = vec3<f32>(0.0);
    for (var i = 0u; i < 12u; i = i + 1u) {
        let a = f32(i) * (LK_TAU / 12.0) + 0.3;
        let r = radius * (0.45 + 0.55 * f32(i % 2u));
        let tap = lk_src(uv + vec2<f32>(cos(a), sin(a)) * u.size.zw * r);
        acc = acc + max(tap - vec3<f32>(threshold), vec3<f32>(0.0));
    }
    return acc / 12.0;
}
";

/// The 4 by 4 ordered dither's threshold at cell `(x, y)`, 0 to 1.
pub(crate) fn bayer(x: i32, y: i32) -> f32 {
    const M: [f32; 16] = [
        0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0,
    ];
    (M[(y.rem_euclid(4) * 4 + x.rem_euclid(4)) as usize] + 0.5) / 16.0
}

const BAYER: &str = r"
fn lk_bayer(x: i32, y: i32) -> f32 {
    var m = array<f32, 16>(0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);
    let i = ((y % 4) + 4) % 4 * 4 + ((x % 4) + 4) % 4;
    return (m[i] + 0.5) / 16.0;
}
";

/// Falling things in `layers` layers of cells `cell` pixels at 1080
/// lines, falling `speed` cells a second and swaying: for each layer,
/// where in its cell the pixel is from the thing's middle (pixels), the
/// thing's seed and whether there is one (the share `amount`), nearest
/// first.
pub(crate) fn fall(
    u: &Ctx,
    uv: V2,
    (cellpx, speed, wind): (f32, f32, f32),
    layer: u32,
    amount: f32,
) -> (V2, f32, f32, f32) {
    let t = u.time[0];
    let p = u.px1080(uv);
    let fl = layer as f32;
    let cs = cellpx / (1.0 + 0.6 * fl);
    let qq = p / cs
        + v2(
            -t * wind * 0.3 * (1.0 + 0.5 * fl),
            -t * speed * (1.0 + 0.4 * fl),
        );
    let cv = qq.floor();
    let (x, y) = (cell(cv.x), cell(cv.y));
    let h = hash2(x, y, 201 + layer);
    let sway = (t * (0.6 + 0.5 * h) + h * TAU).sin() * 0.2;
    let at = v2(
        0.25 + 0.5 * hash2(x, y, 204 + layer) + sway,
        0.25 + 0.5 * hash2(x, y, 207 + layer),
    );
    let local = (qq - cv - at) * cs;
    (
        local,
        h,
        step(hash2(x, y, 210 + layer), amount),
        1.0 / (1.0 + 0.6 * fl),
    )
}

const FALL: &str = r"
fn lk_fall(uv: vec2<f32>, cellpx: f32, speed: f32, wind: f32, layer: u32, amount: f32) -> vec4<f32> {
    let t = u.time.x;
    let p = lk_px1080(uv);
    let fl = f32(layer);
    let cs = cellpx / (1.0 + 0.6 * fl);
    let qq = p / cs + vec2<f32>(-t * wind * 0.3 * (1.0 + 0.5 * fl), -t * speed * (1.0 + 0.4 * fl));
    let cv = floor(qq);
    let x = i32(cv.x);
    let y = i32(cv.y);
    let h = lk_hash2(x, y, 201u + layer);
    let sway = sin(t * (0.6 + 0.5 * h) + h * LK_TAU) * 0.2;
    let at = vec2<f32>(0.25 + 0.5 * lk_hash2(x, y, 204u + layer) + sway, 0.25 + 0.5 * lk_hash2(x, y, 207u + layer));
    let local = (qq - cv - at) * cs;
    return vec4<f32>(local, h, step(lk_hash2(x, y, 210u + layer), amount));
}
";

/// Every helper's WGSL above, as every pass's shader has them.
pub fn wgsl() -> String {
    [SCREEN_CELL, DOT, EDGES, HATCH, RING_LIGHT, BAYER, FALL].concat()
}
