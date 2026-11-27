//! Old screens and tapes: a tube's lines and mask, a tape's wobble and
//! bleed, pixels and their palettes, glitches, colour split apart.
//! What jitters does so on the song's time (a tape's lines at 30 a
//! second), and moves pixels, never the screen's light.

use crate::drive::Driver;
use crate::math::*;
use crate::pass::{knob, natural, Ctx, Family, Pass, Src, Unit};
use crate::passes::common::bayer;

const R: Family = Family::Retro;

fn crt(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let q = uv * 2.0 - 1.0;
    let bend = q * (1.0 + k[2] * 0.12 * q.dot(q));
    let at = bend * 0.5 + 0.5;
    let inside = smoothstep(0.0, 0.006, at.x.min(at.y).min((1.0 - at.x).min(1.0 - at.y)));
    let c = s.src(at.clamp(0.0, 1.0));
    let p = uv * u.size2();
    let period = 3.0 * u.scale();
    let line = 0.5 + 0.5 * (p.y * TAU / period).cos();
    let lum = luma(c);
    let scan = mix(1.0, 0.6 + 0.4 * line + 0.3 * lum * (1.0 - line), k[0]);
    let m = (p.x / u.scale()).floor() as i32;
    let ch = m.rem_euclid(3);
    let mask_rgb = v3(
        if ch == 0 { 1.0 } else { 0.7 },
        if ch == 1 { 1.0 } else { 0.7 },
        if ch == 2 { 1.0 } else { 0.7 },
    );
    let mask = mix3(s3(1.0), mask_rgb, k[1]) * (1.0 + 0.15 * k[1]);
    let d = u.px() * (2.5 * u.scale());
    let glow = (s.src(at + v2(d.x, 0.0))
        + s.src(at - v2(d.x, 0.0))
        + s.src(at + v2(0.0, d.y))
        + s.src(at - v2(0.0, d.y)))
        * 0.25;
    (c * scan * mask + glow * (k[3] * 0.25)) * inside
}

/// A cathode-ray tube.
pub const CRT: Pass = Pass {
    name: "crt",
    family: R,
    about: "an old tube screen: scan lines, the phosphor mask's stripes, a curved glass and its glow (the 80s, video games, synthwave)",
    knobs: &[
        knob("scanlines", "how strong the scan lines", (0.0, 1.0, 0.5), Unit::Share, (0.3, 0.15)),
        knob("mask", "how strong the colour mask's stripes", (0.0, 1.0, 0.4), Unit::Share, (0.2, 0.15)),
        knob("curve", "how curved the glass", (0.0, 1.0, 0.3), Unit::Share, (0.0, 0.1)),
        knob("glow", "how much the phosphor glows", (0.0, 1.0, 0.3), Unit::Share, (0.5, 0.1)),
    ],
    cost: 2.0,
    relief: 0.3,
    natural: &[natural("glow", Driver::Bass, 0.7), natural("scanlines", Driver::Drums, 0.5)],
    pairs: &["vhs", "pixelate", "palette", "teletext"],
    wgsl: r"
fn look_crt(uv: vec2<f32>) -> vec3<f32> {
    let q = uv * 2.0 - 1.0;
    let bend = q * (1.0 + lk_k(2u) * 0.12 * dot(q, q));
    let at = bend * 0.5 + 0.5;
    let inside = smoothstep(0.0, 0.006, min(min(at.x, at.y), min(1.0 - at.x, 1.0 - at.y)));
    let c = lk_src(clamp(at, vec2<f32>(0.0), vec2<f32>(1.0)));
    let p = uv * u.size.xy;
    let period = 3.0 * lk_scale();
    let line = 0.5 + 0.5 * cos(p.y * LK_TAU / period);
    let lum = lk_luma(c);
    let scan = mix(1.0, 0.6 + 0.4 * line + 0.3 * lum * (1.0 - line), lk_k(0u));
    let m = i32(floor(p.x / lk_scale()));
    let ch = ((m % 3) + 3) % 3;
    let mask_rgb = vec3<f32>(select(0.7, 1.0, ch == 0), select(0.7, 1.0, ch == 1), select(0.7, 1.0, ch == 2));
    let mask = mix(vec3<f32>(1.0), mask_rgb, lk_k(1u)) * (1.0 + 0.15 * lk_k(1u));
    let d = u.size.zw * (2.5 * lk_scale());
    let glow = (lk_src(at + vec2<f32>(d.x, 0.0)) + lk_src(at - vec2<f32>(d.x, 0.0)) + lk_src(at + vec2<f32>(0.0, d.y)) + lk_src(at - vec2<f32>(0.0, d.y))) * 0.25;
    return (c * scan * mask + glow * (lk_k(3u) * 0.25)) * inside;
}
",
    cpu: crt,
    ..Pass::BASE
};

fn yiq(c: V3) -> V3 {
    v3(
        0.299 * c.x + 0.587 * c.y + 0.114 * c.z,
        0.596 * c.x - 0.274 * c.y - 0.322 * c.z,
        0.211 * c.x - 0.523 * c.y + 0.312 * c.z,
    )
}

fn rgb(c: V3) -> V3 {
    v3(
        c.x + 0.956 * c.y + 0.621 * c.z,
        c.x - 0.272 * c.y - 0.647 * c.z,
        c.x - 1.106 * c.y + 1.703 * c.z,
    )
}

fn vhs(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let t = u.time[0];
    let p = u.px1080(uv);
    let f = (t * 30.0).max(0.0).floor() as u32;
    let band = fract(t * 0.07);
    let track = (1.0 - smoothstep(0.0, 0.05, (uv.y - band).abs())) * k[3];
    let wob = (vnoise(v2(p.y / 50.0, t * 2.0), 161) - 0.5) * 3.0 * k[0]
        + track * 12.0 * (vnoise(v2(p.y / 3.0, f as f32), 162) - 0.5);
    let at = uv + v2(wob * u.scale() * u.size[2], 0.0);
    let y = yiq(s.src(at));
    let step_x = 4.0 * k[1] * u.scale() * u.size[2];
    let mut ch = v2(y.y, y.z);
    for i in 1..4u32 {
        let o = yiq(s.src(at - v2(step_x * i as f32, 0.0)));
        ch = ch + v2(o.y, o.z);
    }
    let ch = mix2(v2(y.y, y.z), ch / 4.0, k[1]);
    let n = (hash2(cell(p.x / 2.0), cell(p.y), f) - 0.5) * 0.25 * k[2];
    let snow = step(
        1.0 - 0.02 * k[2] - track * 0.2,
        hash2(cell(p.x / 8.0), cell(p.y / 2.0), f + 1),
    );
    let o = rgb(v3(y.x + n, ch.x, ch.y)) + s3(snow * 0.5);
    o.max(0.0)
}

/// A worn video tape.
pub const VHS: Pass = Pass {
    name: "vhs",
    family: R,
    about: "a worn video tape: lines wobbling, the colour bleeding sideways, noise and a band of bad tracking rolling up (the 80s and 90s, camcorder, lo-fi)",
    knobs: &[
        knob("wobble", "how much the lines wobble", (0.0, 1.0, 0.4), Unit::Share, (0.5, 0.0)),
        knob("bleed", "how far the colour bleeds", (0.0, 1.0, 0.5), Unit::Share, (0.3, 0.01)),
        knob("noise", "how much noise and snow", (0.0, 1.0, 0.3), Unit::Share, (0.3, 0.04)),
        knob("tracking", "how bad the tracking band", (0.0, 1.0, 0.3), Unit::Share, (0.5, 0.04)),
    ],
    cost: 1.8,
    relief: 0.3,
    natural: &[natural("tracking", Driver::Snare, 0.9), natural("wobble", Driver::Bass, 0.7)],
    pairs: &["crt", "cross_process", "grain"],
    wgsl: r"
fn lk_yiq(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(0.299 * c.x + 0.587 * c.y + 0.114 * c.z, 0.596 * c.x - 0.274 * c.y - 0.322 * c.z, 0.211 * c.x - 0.523 * c.y + 0.312 * c.z);
}
fn lk_rgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(c.x + 0.956 * c.y + 0.621 * c.z, c.x - 0.272 * c.y - 0.647 * c.z, c.x - 1.106 * c.y + 1.703 * c.z);
}
fn look_vhs(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time.x;
    let p = lk_px1080(uv);
    let f = u32(floor(max(t * 30.0, 0.0)));
    let band = fract(t * 0.07);
    let track = (1.0 - smoothstep(0.0, 0.05, abs(uv.y - band))) * lk_k(3u);
    let wob = (lk_vnoise(vec2<f32>(p.y / 50.0, t * 2.0), 161u) - 0.5) * 3.0 * lk_k(0u) + track * 12.0 * (lk_vnoise(vec2<f32>(p.y / 3.0, f32(f)), 162u) - 0.5);
    let at = uv + vec2<f32>(wob * lk_scale() * u.size.z, 0.0);
    let y = lk_yiq(lk_src(at));
    let step_x = 4.0 * lk_k(1u) * lk_scale() * u.size.z;
    var ch = y.yz;
    for (var i = 1u; i < 4u; i = i + 1u) {
        let o = lk_yiq(lk_src(at - vec2<f32>(step_x * f32(i), 0.0)));
        ch = ch + o.yz;
    }
    let chm = mix(y.yz, ch / 4.0, lk_k(1u));
    let n = (lk_hash2(i32(p.x / 2.0), i32(p.y), f) - 0.5) * 0.25 * lk_k(2u);
    let snow = step(1.0 - 0.02 * lk_k(2u) - track * 0.2, lk_hash2(i32(p.x / 8.0), i32(p.y / 2.0), f + 1u));
    let o = lk_rgb(vec3<f32>(y.x + n, chm.x, chm.y)) + vec3<f32>(snow * 0.5);
    return max(o, vec3<f32>(0.0));
}
",
    cpu: vhs,
    ..Pass::BASE
};

fn pixelate(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let size = k[0] * u.scale();
    let cv = (uv * u.size2() / size).floor();
    let c = s.src((cv + 0.5) * size / u.size2());
    let n = k[1] - 1.0;
    let d = bayer(cell(cv.x), cell(cv.y));
    let th = mix(0.5, d, k[2]);
    c.clamp(0.0, 1.0).map(|x| (x * n + th).floor() / n)
}

/// Big pixels, few colours, ordered dither.
pub const PIXELATE: Pass = Pass {
    name: "pixelate",
    family: R,
    about: "big pixels in a few shades a channel, an ordered dither between them (8-bit and 16-bit games, chiptune)",
    knobs: &[
        knob("size", "the pixels' size", (2.0, 24.0, 6.0), Unit::Px, (5.0, 0.04)),
        knob("levels", "how many shades a channel", (2.0, 16.0, 6.0), Unit::Steps, (0.0, 0.14)),
        knob("dither", "none (0) to a full ordered dither (1)", (0.0, 1.0, 0.6), Unit::Share, (0.2, 0.02)),
    ],
    relief: 0.5,
    natural: &[natural("size", Driver::Kick, 0.7)],
    pairs: &["crt", "palette", "glitch"],
    wgsl: r"
fn look_pixelate(uv: vec2<f32>) -> vec3<f32> {
    let size = lk_k(0u) * lk_scale();
    let cv = floor(uv * u.size.xy / size);
    let c = lk_src((cv + 0.5) * size / u.size.xy);
    let n = lk_k(1u) - 1.0;
    let d = lk_bayer(i32(cv.x), i32(cv.y));
    let th = mix(0.5, d, lk_k(2u));
    return floor(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)) * n + th) / n;
}
",
    cpu: pixelate,
    ..Pass::BASE
};

/// The palettes: a handheld's four greens, a home computer's four (black,
/// cyan, magenta, white), an eight-colour micro's, the cover's own
/// (darkest to lightest with black and white).
fn palette_colour(which: i32, i: i32, u: &Ctx) -> V3 {
    let i = i.clamp(0, 7);
    match which {
        0 => [
            v3(0.06, 0.22, 0.06),
            v3(0.19, 0.38, 0.19),
            v3(0.55, 0.67, 0.06),
            v3(0.61, 0.74, 0.06),
        ][(i / 2) as usize],
        1 => [
            v3(0.0, 0.0, 0.0),
            v3(0.33, 1.0, 1.0),
            v3(1.0, 0.33, 1.0),
            v3(1.0, 1.0, 1.0),
        ][(i / 2) as usize],
        2 => [
            v3(0.0, 0.0, 0.0),
            v3(0.53, 0.0, 0.0),
            v3(0.0, 0.0, 0.67),
            v3(0.8, 0.27, 0.8),
            v3(0.0, 0.8, 0.33),
            v3(0.67, 1.0, 0.93),
            v3(0.93, 0.93, 0.47),
            v3(1.0, 1.0, 1.0),
        ][i as usize],
        _ => {
            let ramp = [s3(0.02), u.pal(2), u.pal(1), u.pal(0)];
            ramp[(i / 2) as usize]
        }
    }
}

fn palette(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let size = k[1] * u.scale();
    let cv = (uv * u.size2() / size).floor();
    let c = s.src((cv + 0.5) * size / u.size2());
    let which = k[0] as i32;
    let d = mix(0.5, bayer(cell(cv.x), cell(cv.y)), k[2]) - 0.5;
    if which == 2 {
        let mut best = 0;
        let mut dist = 99.0;
        let at = c + s3(d * 0.35);
        for i in 0..8 {
            let e = (palette_colour(2, i, u) - at).length();
            if e < dist {
                dist = e;
                best = i;
            }
        }
        palette_colour(2, best, u)
    } else {
        let l = sat(luma(c) + d * 0.3);
        let i = ((l * 3.999).floor() as i32) * 2;
        palette_colour(which, i, u)
    }
}

/// A limited palette.
pub const PALETTE: Pass = Pass {
    name: "palette",
    family: R,
    about: "an old machine's few colours, dithered: a handheld's four greens, a home computer's black-cyan-magenta-white, an eight-colour micro's, or the cover's own four (chiptune, video-game music)",
    knobs: &[
        knob("palette", "handheld greens (0), cyan-magenta (1), eight colours (2), the cover's (3)", (0.0, 3.0, 0.0), Unit::Steps, (0.0, 0.45)),
        knob("size", "the pixels' size", (2.0, 12.0, 4.0), Unit::Px, (3.0, 0.02)),
        knob("dither", "none (0) to a full ordered dither (1)", (0.0, 1.0, 0.5), Unit::Share, (0.2, 0.03)),
    ],
    relief: 0.5,
    natural: &[natural("size", Driver::Kick, 0.6)],
    pairs: &["crt", "pixelate"],
    wgsl: r"
fn lk_palette_colour(which: i32, i0: i32) -> vec3<f32> {
    let i = clamp(i0, 0, 7);
    var green = array<vec3<f32>, 4>(vec3<f32>(0.06, 0.22, 0.06), vec3<f32>(0.19, 0.38, 0.19), vec3<f32>(0.55, 0.67, 0.06), vec3<f32>(0.61, 0.74, 0.06));
    var cga = array<vec3<f32>, 4>(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.33, 1.0, 1.0), vec3<f32>(1.0, 0.33, 1.0), vec3<f32>(1.0, 1.0, 1.0));
    var eight = array<vec3<f32>, 8>(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.53, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.67), vec3<f32>(0.8, 0.27, 0.8), vec3<f32>(0.0, 0.8, 0.33), vec3<f32>(0.67, 1.0, 0.93), vec3<f32>(0.93, 0.93, 0.47), vec3<f32>(1.0, 1.0, 1.0));
    var cover = array<vec3<f32>, 4>(vec3<f32>(0.02), u.pal2.xyz, u.pal1.xyz, u.pal0.xyz);
    if (which == 0) { return green[i / 2]; }
    if (which == 1) { return cga[i / 2]; }
    if (which == 2) { return eight[i]; }
    return cover[i / 2];
}
fn look_palette(uv: vec2<f32>) -> vec3<f32> {
    let size = lk_k(1u) * lk_scale();
    let cv = floor(uv * u.size.xy / size);
    let c = lk_src((cv + 0.5) * size / u.size.xy);
    let which = i32(lk_k(0u));
    let d = mix(0.5, lk_bayer(i32(cv.x), i32(cv.y)), lk_k(2u)) - 0.5;
    if (which == 2) {
        var best = 0;
        var dist = 99.0;
        let at = c + vec3<f32>(d * 0.35);
        for (var i = 0; i < 8; i = i + 1) {
            let e = length(lk_palette_colour(2, i) - at);
            if (e < dist) {
                dist = e;
                best = i;
            }
        }
        return lk_palette_colour(2, best);
    }
    let l = lk_sat(lk_luma(c) + d * 0.3);
    let i = i32(floor(l * 3.999)) * 2;
    return lk_palette_colour(which, i);
}
",
    cpu: palette,
    ..Pass::BASE
};

fn glitch(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let p = u.px1080(uv);
    let g = (u.time[0] * 12.0).max(0.0).floor() as u32;
    let bs = v2(k[1], k[1] * 0.4);
    let cv = (p / bs).floor();
    let (x, y) = (cell(cv.x), cell(cv.y));
    let on = step(hash2(x, y, g), k[0] * 0.35);
    let shift = (hash2(x, y, g + 1) - 0.5) * k[1] * 3.0 * on;
    let at = uv + v2(shift * u.scale() * u.size[2], 0.0);
    let split = (k[2] * 6.0 * u.scale() * u.size[2]) * (0.3 + on);
    let r = s.src(at + v2(split, 0.0)).x;
    let gch = s.src(at).y;
    let b = s.src(at - v2(split, 0.0)).z;
    let o = v3(r, gch, b);
    let tint = mix3(s3(1.0), hsv(hash2(x, y, g + 2), 0.8, 1.2), on * 0.4 * k[0]);
    o * tint
}

/// Digital glitches: blocks torn sideways, colours split.
pub const GLITCH: Pass = Pass {
    name: "glitch",
    family: R,
    about: "digital glitches: blocks of the picture torn sideways and tinted, its colours split, new twelve times a second (Y2K, datamosh, industrial, dubstep drops)",
    knobs: &[
        knob("amount", "how many blocks tear", (0.0, 1.0, 0.3), Unit::Share, (0.6, 0.05)),
        knob("block", "the blocks' size", (8.0, 80.0, 32.0), Unit::Px, (16.0, 0.0)),
        knob("split", "how far the colours split", (0.0, 1.0, 0.4), Unit::Share, (0.5, 0.01)),
    ],
    cost: 1.5,
    relief: 0.6,
    natural: &[natural("amount", Driver::Snare, 1.0), natural("split", Driver::Kick, 0.6)],
    pairs: &["vhs", "chromatic", "xerox"],
    wgsl: r"
fn look_glitch(uv: vec2<f32>) -> vec3<f32> {
    let p = lk_px1080(uv);
    let g = u32(floor(max(u.time.x * 12.0, 0.0)));
    let bs = vec2<f32>(lk_k(1u), lk_k(1u) * 0.4);
    let cv = floor(p / bs);
    let x = i32(cv.x);
    let y = i32(cv.y);
    let on = step(lk_hash2(x, y, g), lk_k(0u) * 0.35);
    let shift = (lk_hash2(x, y, g + 1u) - 0.5) * lk_k(1u) * 3.0 * on;
    let at = uv + vec2<f32>(shift * lk_scale() * u.size.z, 0.0);
    let split = (lk_k(2u) * 6.0 * lk_scale() * u.size.z) * (0.3 + on);
    let r = lk_src(at + vec2<f32>(split, 0.0)).x;
    let gch = lk_src(at).y;
    let b = lk_src(at - vec2<f32>(split, 0.0)).z;
    let o = vec3<f32>(r, gch, b);
    let tint = mix(vec3<f32>(1.0), lk_hsv(lk_hash2(x, y, g + 2u), 0.8, 1.2), on * 0.4 * lk_k(0u));
    return o * tint;
}
",
    cpu: glitch,
    ..Pass::BASE
};

fn anaglyph(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let depth = luma(c) - 0.5;
    let off = (k[0] * u.scale() * u.size[2]) * (1.0 + k[2] * depth);
    let l = s.src(uv - v2(off, 0.0));
    let r = s.src(uv + v2(off, 0.0));
    let o = v3(luma(l), r.y, r.z);
    mix3(c, mix3(o, v3(l.x, r.y, r.z), 0.3), k[1])
}

/// Red and cyan 3D glasses' picture.
pub const ANAGLYPH: Pass = Pass {
    name: "anaglyph",
    family: R,
    about: "a red-and-cyan 3D picture, the two eyes' views apart (the 50s cinema, B-movies, sci-fi, fun)",
    knobs: &[
        knob("offset", "how far apart the eyes' views", (1.0, 20.0, 6.0), Unit::Px, (6.0, 0.01)),
        knob("amount", "how much of the 3D over the picture", (0.0, 1.0, 1.0), Unit::Share, (0.3, 0.05)),
        knob("depth", "how much brighter parts come forward", (0.0, 1.0, 0.5), Unit::Share, (0.2, 0.01)),
    ],
    relief: 0.4,
    natural: &[natural("offset", Driver::Kick, 0.8)],
    pairs: &["grain", "technicolor"],
    wgsl: r"
fn look_anaglyph(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let depth = lk_luma(c) - 0.5;
    let off = (lk_k(0u) * lk_scale() * u.size.z) * (1.0 + lk_k(2u) * depth);
    let l = lk_src(uv - vec2<f32>(off, 0.0));
    let r = lk_src(uv + vec2<f32>(off, 0.0));
    let o = vec3<f32>(lk_luma(l), r.y, r.z);
    return mix(c, mix(o, vec3<f32>(l.x, r.y, r.z), 0.3), lk_k(1u));
}
",
    cpu: anaglyph,
    ..Pass::BASE
};

fn chromatic(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let a = k[2] * TAU;
    let lin = v2(a.cos(), a.sin());
    let rad = (uv - 0.5) * 2.0;
    let dir = mix2(lin, rad, k[1]);
    let d = dir * (k[0] * 12.0 * u.scale()) * u.px();
    v3(s.src(uv + d).x, s.src(uv).y, s.src(uv - d).z)
}

/// The colours split apart.
pub const CHROMATIC: Pass = Pass {
    name: "chromatic",
    family: R,
    about: "the colours split apart, red one way and blue the other, as a cheap lens or a shaken screen does (psychedelia, glitch, drops)",
    knobs: &[
        knob("amount", "how far the colours split (up to 12 pixels)", (0.0, 1.0, 0.3), Unit::Share, (0.6, 0.01)),
        knob("radial", "the same way everywhere (0) to out from the middle (1)", (0.0, 1.0, 1.0), Unit::Share, (0.0, 0.0)),
        knob("angle", "the way they split, when not out from the middle", (0.0, 1.0, 0.0), Unit::Turns, (0.2, 0.0)),
    ],
    natural: &[natural("amount", Driver::Kick, 0.9)],
    pairs: &["glitch", "hue", "liquid"],
    wgsl: r"
fn look_chromatic(uv: vec2<f32>) -> vec3<f32> {
    let a = lk_k(2u) * LK_TAU;
    let lin = vec2<f32>(cos(a), sin(a));
    let rad = (uv - 0.5) * 2.0;
    let dir = mix(lin, rad, lk_k(1u));
    let d = dir * (lk_k(0u) * 12.0 * lk_scale()) * u.size.zw;
    return vec3<f32>(lk_src(uv + d).x, lk_src(uv).y, lk_src(uv - d).z);
}
",
    cpu: chromatic,
    ..Pass::BASE
};
