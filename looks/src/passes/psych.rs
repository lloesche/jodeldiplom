//! Psychedelia: colour turning round the wheel, symmetry, the picture
//! flowing like liquid, a zoom's blur rushing out. The colour moves; the
//! light stays as it was.

use crate::drive::Driver;
use crate::math::*;
use crate::pass::{knob, natural, Ctx, Family, Pass, Src, Unit};

const Y: Family = Family::Psychedelic;

fn hue(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let turns = k[0] + k[2] * u.time[0] * 0.1 + k[1] * (uv.x * 0.7 + uv.y * 0.3);
    mix3(c, hue_turn(c, turns).max(0.0), k[3])
}

/// The colours turned round the wheel.
pub const HUE: Pass = Pass {
    name: "hue",
    family: Y,
    about: "the colours turned round the wheel: shifted, cycling slowly, or a rainbow across the screen, the light kept (psychedelia, a new hue for each section)",
    knobs: &[
        knob("shift", "how far round the wheel", (0.0, 1.0, 0.0), Unit::Turns, (1.0, 0.03)),
        knob("spread", "a rainbow across the screen", (0.0, 1.0, 0.0), Unit::Share, (0.4, 0.03)),
        knob("speed", "cycling backwards (-) or forwards (+), a turn in ten seconds at the most", (-1.0, 1.0, 0.0), Unit::Share, (0.3, 0.03)),
        knob("amount", "how much of the turned colour", (0.0, 1.0, 1.0), Unit::Share, (0.3, 0.03)),
    ],
    natural: &[natural("shift", Driver::Part, 1.0), natural("spread", Driver::Bass, 0.5)],
    pairs: &["kaleidoscope", "liquid", "saturation"],
    wgsl: r"
fn look_hue(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let turns = lk_k(0u) + lk_k(2u) * u.time.x * 0.1 + lk_k(1u) * (uv.x * 0.7 + uv.y * 0.3);
    return mix(c, max(lk_hue_turn(c, turns), vec3<f32>(0.0)), lk_k(3u));
}
",
    cpu: hue,
    ..Pass::BASE
};

/// `x` folded into 0 to 1 as a mirror repeats.
fn mirror_repeat(x: f32) -> f32 {
    1.0 - (fract(x * 0.5) * 2.0 - 1.0).abs()
}

fn kaleidoscope(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let p = u.aspect(uv);
    let r = p.length();
    let seg = TAU / k[0];
    let a = p.y.atan2(p.x) + k[1] * TAU;
    let a = (fract(a / seg) * seg - seg * 0.5).abs();
    let q = v2(a.cos(), a.sin()) * (r / k[2]);
    let at = q / v2(u.size[0] / u.size[1], 1.0) + 0.5;
    s.src(v2(mirror_repeat(at.x), mirror_repeat(at.y)))
}

/// A kaleidoscope.
pub const KALEIDOSCOPE: Pass = Pass {
    name: "kaleidoscope",
    family: Y,
    about: "a kaleidoscope: the picture folded into mirrored segments round the middle, turning (psychedelia, trance, instrumental breaks)",
    knobs: &[
        knob("segments", "how many mirrored segments", (2.0, 16.0, 6.0), Unit::Steps, (0.0, 0.17)),
        knob("turn", "how far turned", (0.0, 1.0, 0.0), Unit::Turns, (1.0, 0.03)),
        knob("zoom", "zoomed out (low) or in (high)", (0.5, 2.0, 1.0), Unit::Share, (0.4, 0.13)),
    ],
    relief: 0.5,
    natural: &[natural("turn", Driver::BarPhase, 0.25), natural("zoom", Driver::Bass, 0.6)],
    pairs: &["hue", "feedback", "liquid"],
    wgsl: r"
fn lk_mirror_repeat(x: f32) -> f32 { return 1.0 - abs(fract(x * 0.5) * 2.0 - 1.0); }
fn look_kaleidoscope(uv: vec2<f32>) -> vec3<f32> {
    let p = lk_aspect(uv);
    let r = length(p);
    let seg = LK_TAU / lk_k(0u);
    var a = atan2(p.y, p.x) + lk_k(1u) * LK_TAU;
    a = abs(fract(a / seg) * seg - seg * 0.5);
    let q = vec2<f32>(cos(a), sin(a)) * (r / lk_k(2u));
    let at = q / vec2<f32>(u.size.x / u.size.y, 1.0) + 0.5;
    return lk_src(vec2<f32>(lk_mirror_repeat(at.x), lk_mirror_repeat(at.y)));
}
",
    cpu: kaleidoscope,
    ..Pass::BASE
};

fn mirror(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let mode = k[0];
    let m = k[1];
    let x = if mode != 1.0 {
        m - (uv.x - m).abs()
    } else {
        uv.x
    };
    let y = if mode >= 1.0 {
        m - (uv.y - m).abs()
    } else {
        uv.y
    };
    s.src(v2(x, y).clamp(0.0, 1.0))
}

/// A mirror down or across the middle.
pub const MIRROR: Pass = Pass {
    name: "mirror",
    family: Y,
    about: "the picture mirrored left to right, top to bottom or four ways (symmetry, Rorschach, dance, the surreal)",
    knobs: &[
        knob("mode", "left onto right (0), top onto bottom (1), four ways (2)", (0.0, 2.0, 0.0), Unit::Steps, (0.0, 0.09)),
        knob("place", "where the mirror stands", (0.2, 0.8, 0.5), Unit::Share, (0.2, 0.05)),
    ],
    relief: 0.5,
    natural: &[natural("place", Driver::Other, 0.6)],
    pairs: &["kaleidoscope", "hue", "feedback"],
    wgsl: r"
fn look_mirror(uv: vec2<f32>) -> vec3<f32> {
    let mode = lk_k(0u);
    let m = lk_k(1u);
    let x = select(uv.x, m - abs(uv.x - m), mode != 1.0);
    let y = select(uv.y, m - abs(uv.y - m), mode >= 1.0);
    return lk_src(clamp(vec2<f32>(x, y), vec2<f32>(0.0), vec2<f32>(1.0)));
}
",
    cpu: mirror,
    ..Pass::BASE
};

fn liquid(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let p = u.aspect(uv) * (k[1] * 2.0) + s2(u.time[0] * k[2] * 0.2);
    let w = v2(fbm(p, 261) - 0.5, fbm(p + 5.2, 262) - 0.5) * (k[0] * 0.06);
    s.src(uv + w)
}

/// The picture flowing like liquid.
pub const LIQUID: Pass = Pass {
    name: "liquid",
    family: Y,
    about: "the picture flowing and warping like oil on water (psychedelia, drugs and dreams, the surreal, shoegaze)",
    knobs: &[
        knob("amount", "how far it flows", (0.0, 1.0, 0.4), Unit::Share, (0.5, 0.01)),
        knob("scale", "the swirls' size: big (low) to small (high)", (0.5, 4.0, 1.5), Unit::Share, (0.5, 0.0)),
        knob("speed", "how fast it flows", (0.0, 2.0, 0.5), Unit::Share, (0.6, 0.0)),
    ],
    cost: 1.2,
    relief: 0.5,
    natural: &[natural("amount", Driver::Bass, 0.8), natural("speed", Driver::Energy, 0.5)],
    pairs: &["hue", "chromatic", "feedback"],
    wgsl: r"
fn look_liquid(uv: vec2<f32>) -> vec3<f32> {
    let p = lk_aspect(uv) * (lk_k(1u) * 2.0) + vec2<f32>(u.time.x * lk_k(2u) * 0.2);
    let w = vec2<f32>(lk_fbm(p, 261u) - 0.5, lk_fbm(p + 5.2, 262u) - 0.5) * (lk_k(0u) * 0.06);
    return lk_src(uv + w);
}
",
    cpu: liquid,
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
    family: Y,
    about: "a zoom's blur rushing out from a point, as if the camera lunged (drops, drum and bass, rock's big hits)",
    knobs: &[
        knob("amount", "how far it rushes", (0.0, 1.0, 0.3), Unit::Share, (0.6, 0.02)),
        knob("x", "the point's place across", (0.0, 1.0, 0.5), Unit::Share, (0.0, 0.0)),
        knob("y", "the point's place down", (0.0, 1.0, 0.45), Unit::Share, (0.0, 0.0)),
    ],
    cost: 3.0,
    relief: 0.6,
    natural: &[natural("amount", Driver::Kick, 0.9)],
    pairs: &["chromatic", "glitch", "feedback"],
    wgsl: r"
fn look_zoom_blur(uv: vec2<f32>) -> vec3<f32> {
    let dir = uv - vec2<f32>(lk_k(1u), lk_k(2u));
    var acc = vec3<f32>(0.0);
    for (var i = 0u; i < 10u; i = i + 1u) {
        acc = acc + lk_src(uv - dir * (lk_k(0u) * 0.15 * f32(i) / 10.0));
    }
    return acc / 10.0;
}
",
    cpu: zoom_blur,
    ..Pass::BASE
};
