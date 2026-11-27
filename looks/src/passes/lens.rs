//! What a camera and a cinema add: the letterbox's black bars, bokeh's
//! soft balls of light, a zoom's blur rushing out from a point, confetti
//! falling in front of the lens.

use crate::drive::Driver;
use crate::math::*;
use crate::pass::{knob, natural, Ctx, Family, Pass, Src};
use crate::passes::atmos::fall;

fn letterbox(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let ar = u.size[0] / u.size[1];
    let shown = (ar / k[0]).min(1.0);
    let top = (1.0 - shown) * 0.5;
    let px = u.size[3];
    let inside = smoothstep(top - px, top + px, uv.y)
        * (1.0 - smoothstep(1.0 - top - px, 1.0 - top + px, uv.y));
    c * mix(1.0 - k[1], 1.0, inside)
}

/// A cinema's letterbox.
pub const LETTERBOX: Pass = Pass {
    name: "letterbox",
    family: Family::Film,
    about: "a cinema's black bars, the picture as wide as a film (ballads, epics, soundtracks, road movies)",
    knobs: &[
        knob("ratio", "the picture's width to its height (1.85 flat, 2.39 scope)", (1.85, 2.76, 2.39), "ratio", (0.0, 0.2)),
        knob("amount", "how black the bars", (0.0, 1.0, 1.0), "share", (0.0, 0.3)),
    ],
    pairs: &["split_tone", "grain", "halation"],
    wgsl: r"
fn look_letterbox(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let ar = u.size.x / u.size.y;
    let shown = min(ar / u.k0.x, 1.0);
    let top = (1.0 - shown) * 0.5;
    let px = u.size.w;
    let inside = smoothstep(top - px, top + px, uv.y) * (1.0 - smoothstep(1.0 - top - px, 1.0 - top + px, uv.y));
    return c * mix(1.0 - u.k0.y, 1.0, inside);
}
",
    cpu: letterbox,
    ..Pass::BASE
};

fn bokeh(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let mut acc = s3(0.0);
    for l in 0..2u32 {
        let (local, h, on, near) = fall(u, uv, (160.0, 0.03, 0.05), l, k[0] * 0.5);
        let r = (14.0 + 18.0 * h) * k[1] * near;
        let d = local.length();
        let disc = smoothstep(r, r * 0.85, d) * (0.6 + 0.4 * smoothstep(r * 0.6, r * 0.95, d));
        let col = mix3(hsv(k[2] + 0.1 * h, 0.5, 1.0), u.pal(0), k[3]);
        acc = acc + col * (disc * on * 0.25);
    }
    let dark = 1.0 - smoothstep(0.25, 0.7, luma(c));
    screen(c, acc * dark)
}

/// Bokeh: soft balls of light out of focus.
pub const BOKEH: Pass = Pass {
    name: "bokeh",
    family: Family::Film,
    about: "bokeh: soft balls of out-of-focus light drifting in the dark parts (Christmas, love songs, city nights, soul and R&B)",
    knobs: &[
        knob("amount", "how many lights", (0.0, 1.0, 0.5), "share", (0.4, 0.05)),
        knob("size", "how big", (0.5, 3.0, 1.0), "share", (0.8, 0.05)),
        knob("hue", "their colour", (0.0, 1.0, 0.1), "turns", (0.2, 0.0)),
        knob("cover", "the hue (0) or the cover's colour (1)", (0.0, 1.0, 0.0), "share", (0.0, 0.01)),
    ],
    cost: 1.3,
    relief: 0.5,
    cells: true,
    natural: &[natural("size", Driver::Vocals, 0.7), natural("amount", Driver::Chorus, 0.6)],
    pairs: &["glow", "diffusion", "white_balance"],
    wgsl: r"
fn look_bokeh(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    var acc = vec3<f32>(0.0);
    for (var l = 0u; l < 2u; l = l + 1u) {
        let f = lk_fall(uv, 160.0, 0.03, 0.05, l, u.k0.x * 0.5);
        let near = 1.0 / (1.0 + 0.6 * f32(l));
        let r = (14.0 + 18.0 * f.z) * u.k0.y * near;
        let d = length(f.xy);
        let disc = smoothstep(r, r * 0.85, d) * (0.6 + 0.4 * smoothstep(r * 0.6, r * 0.95, d));
        let col = mix(lk_hsv(u.k0.z + 0.1 * f.z, 0.5, 1.0), u.pal0.xyz, u.k0.w);
        acc = acc + col * (disc * f.w * 0.25);
    }
    let dark = 1.0 - smoothstep(0.25, 0.7, lk_luma(c));
    return lk_screen(c, acc * dark);
}
",
    cpu: bokeh,
    ..Pass::BASE
};

fn zoom_blur(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let dir = uv - v2(k[1], k[2]);
    let mut acc = s3(0.0);
    for i in 0..10u32 {
        acc = acc + s.src(uv - dir * (k[0] * 0.15 * i as f32 / 10.0));
    }
    acc / 10.0
}

/// A zoom's blur.
pub const ZOOM_BLUR: Pass = Pass {
    name: "zoom_blur",
    family: Family::Psychedelic,
    about: "a zoom's blur rushing out from a point, as if the camera lunged (drops, drum and bass, rock's big hits)",
    knobs: &[
        knob("amount", "how far it rushes", (0.0, 1.0, 0.3), "share", (0.6, 0.02)),
        knob("x", "the point's place across", (0.0, 1.0, 0.5), "share", (0.0, 0.0)),
        knob("y", "the point's place down", (0.0, 1.0, 0.45), "share", (0.0, 0.0)),
    ],
    cost: 3.0,
    relief: 0.6,
    natural: &[natural("amount", Driver::Kick, 0.9)],
    pairs: &["chromatic", "glitch", "feedback"],
    wgsl: r"
fn look_zoom_blur(uv: vec2<f32>) -> vec3<f32> {
    let dir = uv - vec2<f32>(u.k0.y, u.k0.z);
    var acc = vec3<f32>(0.0);
    for (var i = 0u; i < 10u; i = i + 1u) {
        acc = acc + lk_src(uv - dir * (u.k0.x * 0.15 * f32(i) / 10.0));
    }
    return acc / 10.0;
}
",
    cpu: zoom_blur,
    ..Pass::BASE
};

fn confetti(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let mut o = s.src(uv);
    for l in 0..3u32 {
        let (local, h, on, near) = fall(u, uv, (70.0, -0.3, k[2]), l, k[0] * 0.6);
        let spin = u.time[0] * (2.0 + 3.0 * h) + h * 11.0;
        let q = turn(local, spin);
        let size = k[1] * near.sqrt();
        let w = 4.0 * size;
        let hgt = 2.0 * size * (spin * 1.3).cos().abs() + 0.5;
        let inside = (1.0 - smoothstep(w - 0.7, w + 0.7, q.x.abs()))
            * (1.0 - smoothstep(hgt - 0.7, hgt + 0.7, q.y.abs()));
        let col = hsv(fract(h * 7.0), 0.8, 1.0);
        o = mix3(o, col, inside * on);
    }
    o
}

/// Confetti falling.
pub const CONFETTI: Pass = Pass {
    name: "confetti",
    family: Family::Atmosphere,
    about: "confetti fluttering down in every colour, thicker in the choruses (party, celebrations, disco, Schlager, weddings)",
    knobs: &[
        knob("amount", "how much confetti", (0.0, 1.0, 0.4), "share", (0.5, 0.04)),
        knob("size", "the pieces' size", (0.5, 3.0, 1.0), "share", (0.4, 0.03)),
        knob("wind", "blowing left (-) or right (+)", (-1.0, 1.0, 0.2), "share", (0.4, 0.0)),
    ],
    cost: 1.3,
    relief: 0.5,
    cells: true,
    natural: &[natural("amount", Driver::Chorus, 0.7), natural("wind", Driver::Bass, 0.5)],
    pairs: &["glow", "sparks", "saturation"],
    wgsl: r"
fn look_confetti(uv: vec2<f32>) -> vec3<f32> {
    var o = lk_src(uv);
    for (var l = 0u; l < 3u; l = l + 1u) {
        let f = lk_fall(uv, 70.0, -0.3, u.k0.z, l, u.k0.x * 0.6);
        let near = 1.0 / (1.0 + 0.6 * f32(l));
        let h = f.z;
        let spin = u.time.x * (2.0 + 3.0 * h) + h * 11.0;
        let q = lk_turn(f.xy, spin);
        let size = u.k0.y * sqrt(near);
        let w = 4.0 * size;
        let hgt = 2.0 * size * abs(cos(spin * 1.3)) + 0.5;
        let inside = (1.0 - smoothstep(w - 0.7, w + 0.7, abs(q.x))) * (1.0 - smoothstep(hgt - 0.7, hgt + 0.7, abs(q.y)));
        let col = lk_hsv(fract(h * 7.0), 0.8, 1.0);
        o = mix(o, col, inside * f.w);
    }
    return o;
}
",
    cpu: confetti,
    ..Pass::BASE
};
