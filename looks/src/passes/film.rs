//! Film: its grain, its glow, its weave, its leaks, its age, the cinema's
//! letterbox and the lens's bokeh. What moves
//! with time moves at film's 24 frames a second on the song's own time
//! (the same frame at the same moment, however often it is drawn), and
//! keeps the screen's light even (grain is as bright as dark).

use crate::drive::Driver;
use crate::math::*;
use crate::pass::{knob, natural, Ctx, Family, Pass, Src};
use crate::passes::atmos::fall;

const F: Family = Family::Film;

/// The film frame at song time `t` (24 a second).
fn film_frame(t: f32) -> u32 {
    (t * 24.0).max(0.0).floor() as u32
}

fn grain(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let size = (k[1] * u.scale()).max(0.6);
    let f = film_frame(u.time[0]).wrapping_mul(7) + u.time[1] as u32;
    let p = uv * v2(u.size[0], u.size[1]) / size;
    let n = vnoise(p, f) - 0.5;
    let ch = v3(
        vnoise(p + v2(13.1, 7.7), f + 3) - 0.5,
        vnoise(p + v2(3.3, 19.9), f + 5) - 0.5,
        vnoise(p + v2(29.3, 1.1), f + 9) - 0.5,
    );
    let g = s3(n) + ch * k[2];
    let l = luma(c);
    let a = k[0] * k[0] * 0.5 * (1.0 - 0.6 * (2.0 * l - 1.0).abs());
    (c + g * a).max(0.0)
}

/// Film grain.
pub const GRAIN: Pass = Pass {
    name: "grain",
    family: F,
    about: "film grain: fine to coarse, mono or coloured, strongest in the midtones, new every film frame",
    knobs: &[
        knob("amount", "how much grain", (0.0, 1.0, 0.35), "share", (0.5, 0.02)),
        knob("size", "the grains' size", (0.6, 3.0, 1.2), "px", (1.0, 0.0)),
        knob("chroma", "mono (0) to coloured grain (1)", (0.0, 1.0, 0.15), "share", (0.2, 0.0)),
    ],
    natural: &[natural("amount", Driver::Bass, 0.8), natural("size", Driver::Drums, 0.6)],
    pairs: &["halation", "vignette", "curve", "gate_weave"],
    wgsl: r"
fn look_grain(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let size = max(u.k0.y * lk_scale(), 0.6);
    let f = u32(floor(max(u.time.x * 24.0, 0.0))) * 7u + u32(u.time.y);
    let p = uv * u.size.xy / size;
    let n = lk_vnoise(p, f) - 0.5;
    let ch = vec3<f32>(lk_vnoise(p + vec2<f32>(13.1, 7.7), f + 3u) - 0.5, lk_vnoise(p + vec2<f32>(3.3, 19.9), f + 5u) - 0.5, lk_vnoise(p + vec2<f32>(29.3, 1.1), f + 9u) - 0.5);
    let g = vec3<f32>(n) + ch * u.k0.z;
    let l = lk_luma(c);
    let a = u.k0.x * u.k0.x * 0.5 * (1.0 - 0.6 * abs(2.0 * l - 1.0));
    return max(c + g * a, vec3<f32>(0.0));
}
",
    cpu: grain,
    ..Pass::BASE
};

/// The light round the bright parts, from twelve reads in two rings
/// `radius` pixels out: what is over `threshold`, on average.
pub(crate) fn ring_light(u: &Ctx, s: &dyn Src, uv: V2, radius: f32, threshold: f32) -> V3 {
    let px = v2(u.size[2], u.size[3]);
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

fn halation(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let acc = ring_light(u, s, uv, k[1] * u.scale(), 0.55);
    let tint = mix3(v3(1.0, 0.5, 0.35), v3(1.0, 0.28, 0.12), k[2]);
    c + acc * tint * (k[0] * 2.2)
}

/// Halation: red light bleeding round the bright parts.
pub const HALATION: Pass = Pass {
    name: "halation",
    family: F,
    about: "halation: the red-orange glow film gets round bright lights, the 35 mm look at night",
    knobs: &[
        knob(
            "amount",
            "how strong the glow",
            (0.0, 1.0, 0.5),
            "share",
            (0.5, 0.06),
        ),
        knob(
            "radius",
            "how far it spreads",
            (2.0, 40.0, 12.0),
            "px",
            (10.0, 0.02),
        ),
        knob(
            "warmth",
            "pinkish (0) to deep red-orange (1)",
            (0.0, 1.0, 0.6),
            "share",
            (0.2, 0.0),
        ),
    ],
    cost: 3.0,
    natural: &[
        natural("amount", Driver::Bass, 0.8),
        natural("radius", Driver::Kick, 0.6),
    ],
    pairs: &["grain", "split_tone", "vignette"],
    wgsl: r"
fn look_halation(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let acc = lk_ring_light(uv, u.k0.y * lk_scale(), 0.55);
    let tint = mix(vec3<f32>(1.0, 0.5, 0.35), vec3<f32>(1.0, 0.28, 0.12), u.k0.z);
    return c + acc * tint * (u.k0.x * 2.2);
}
",
    cpu: halation,
    ..Pass::BASE
};

fn glow(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let acc = ring_light(u, s, uv, k[2] * u.scale(), k[1]);
    let tint = mix3(s3(1.0), hsv(k[3], 0.6, 1.0), k[4]);
    c + acc * tint * (k[0] * 2.5)
}

/// Bloom: the bright parts glowing, tinted.
pub const GLOW: Pass = Pass {
    name: "glow",
    family: F,
    about: "bloom: the bright parts glowing round, tinted if wanted (dreamy, neon, stage light)",
    knobs: &[
        knob(
            "amount",
            "how strong the glow",
            (0.0, 1.0, 0.4),
            "share",
            (0.5, 0.1),
        ),
        knob(
            "threshold",
            "how bright a part must be to glow",
            (0.3, 0.95, 0.65),
            "share",
            (0.1, 0.1),
        ),
        knob(
            "radius",
            "how far it spreads",
            (4.0, 60.0, 20.0),
            "px",
            (16.0, 0.03),
        ),
        knob(
            "hue",
            "the glow's tint",
            (0.0, 1.0, 0.1),
            "turns",
            (0.1, 0.0),
        ),
        knob(
            "tint",
            "white (0) to the hue (1)",
            (0.0, 1.0, 0.2),
            "share",
            (0.2, 0.0),
        ),
    ],
    cost: 3.0,
    natural: &[
        natural("amount", Driver::Vocals, 0.8),
        natural("radius", Driver::Bass, 0.6),
    ],
    pairs: &["diffusion", "neon_edges", "lift_gamma_gain"],
    wgsl: r"
fn look_glow(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let acc = lk_ring_light(uv, u.k0.z * lk_scale(), u.k0.y);
    let tint = mix(vec3<f32>(1.0), lk_hsv(u.k0.w, 0.6, 1.0), u.k1.x);
    return c + acc * tint * (u.k0.x * 2.5);
}
",
    cpu: glow,
    ..Pass::BASE
};

fn diffusion(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let px = v2(u.size[2], u.size[3]) * (k[1] * u.scale());
    let mut b = c;
    for i in 0..8u32 {
        let a = i as f32 * (TAU / 8.0);
        b = b + s.src(uv + v2(a.cos(), a.sin()) * px);
    }
    let b = b / 9.0;
    let o = mix3(c, c.vmax(b), k[0] * 0.8);
    o + k[0] * 0.035
}

/// Diffusion: a soft-focus filter's glow and lifted blacks.
pub const DIFFUSION: Pass = Pass {
    name: "diffusion",
    family: F,
    about: "a diffusion filter: soft focus, lights blooming gently, blacks a little lifted (dreamy, romantic, 70s)",
    knobs: &[
        knob("amount", "how soft", (0.0, 1.0, 0.4), "share", (0.4, 0.08)),
        knob("radius", "how far the softness reaches", (2.0, 30.0, 8.0), "px", (8.0, 0.02)),
    ],
    cost: 2.5,
    natural: &[natural("amount", Driver::Vocals, 0.7), natural("radius", Driver::Bass, 0.5)],
    pairs: &["glow", "curve", "grain"],
    wgsl: r"
fn look_diffusion(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let px = u.size.zw * (u.k0.y * lk_scale());
    var b = c;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let a = f32(i) * (LK_TAU / 8.0);
        b = b + lk_src(uv + vec2<f32>(cos(a), sin(a)) * px);
    }
    b = b / 9.0;
    let o = mix(c, max(c, b), u.k0.x * 0.8);
    return o + u.k0.x * 0.035;
}
",
    cpu: diffusion,
    ..Pass::BASE
};

fn gate_weave(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let t = u.time[0] * k[1];
    let w = k[0] * 2.5 * u.scale();
    let off = v2(
        (vnoise(v2(t, 1.7), 11) - 0.5) * 2.0 * w,
        (vnoise(v2(t, 5.3), 13) - 0.5) * 1.2 * w,
    );
    s.src(uv + off * v2(u.size[2], u.size[3]))
}

/// Gate weave: the frame drifting in the projector's gate.
pub const GATE_WEAVE: Pass = Pass {
    name: "gate_weave",
    family: F,
    about: "gate weave: the whole frame drifting a little, as film does in an old projector",
    knobs: &[
        knob(
            "amount",
            "how far it drifts (up to 2.5 pixels)",
            (0.0, 1.0, 0.4),
            "share",
            (0.6, 0.0),
        ),
        knob(
            "speed",
            "how fast it drifts",
            (0.2, 3.0, 1.0),
            "share",
            (0.5, 0.0),
        ),
    ],
    natural: &[natural("amount", Driver::BassNote, 0.9)],
    pairs: &["grain", "film_damage", "sepia"],
    wgsl: r"
fn look_gate_weave(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time.x * u.k0.y;
    let w = u.k0.x * 2.5 * lk_scale();
    let off = vec2<f32>((lk_vnoise(vec2<f32>(t, 1.7), 11u) - 0.5) * 2.0 * w, (lk_vnoise(vec2<f32>(t, 5.3), 13u) - 0.5) * 1.2 * w);
    return lk_src(uv + off * u.size.zw);
}
",
    cpu: gate_weave,
    ..Pass::BASE
};

/// A point's place on the screen in the shapes' units (the height 1, the
/// middle 0).
pub(crate) fn aspect(u: &Ctx, uv: V2) -> V2 {
    (uv - 0.5) * v2(u.size[0] / u.size[1], 1.0)
}

fn light_leak(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let p = aspect(u, uv);
    let ar = u.size[0] / u.size[1];
    let a = (k[2] + 0.03 * (u.time[0] * 0.21).sin()) * TAU;
    let at = v2(a.cos() * 0.5 * ar, a.sin() * 0.5) * 1.15;
    let d = (p - at).length() / (0.15 + 0.6 * k[3]);
    let g = (-d * d).exp();
    let col = hsv(
        k[1] + 0.04 * (vnoise(v2(u.time[0] * 0.2, 3.0), 21) - 0.5),
        0.75,
        1.0,
    );
    screen(c, col * (g * k[0] * 0.9))
}

/// A light leak: warm light spilling in at an edge.
pub const LIGHT_LEAK: Pass = Pass {
    name: "light_leak",
    family: F,
    about: "a light leak: warm light spilling in from an edge of the frame, drifting (summer, nostalgia, instant cameras)",
    knobs: &[
        knob("amount", "how bright the leak", (0.0, 1.0, 0.5), "share", (0.5, 0.15)),
        knob("hue", "its colour (orange, red, magenta ...)", (0.0, 1.0, 0.06), "turns", (0.1, 0.03)),
        knob("place", "where round the frame it comes in", (0.0, 1.0, 0.15), "turns", (0.3, 0.0)),
        knob("size", "how far into the frame it reaches", (0.0, 1.0, 0.5), "share", (0.3, 0.1)),
    ],
    natural: &[natural("amount", Driver::Downbeat, 0.7), natural("place", Driver::Part, 0.6)],
    pairs: &["grain", "polaroid", "cross_process"],
    wgsl: r"
fn look_light_leak(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let p = lk_aspect(uv);
    let ar = u.size.x / u.size.y;
    let a = (u.k0.z + 0.03 * sin(u.time.x * 0.21)) * LK_TAU;
    let at = vec2<f32>(cos(a) * 0.5 * ar, sin(a) * 0.5) * 1.15;
    let d = length(p - at) / (0.15 + 0.6 * u.k0.w);
    let g = exp(-d * d);
    let col = lk_hsv(u.k0.y + 0.04 * (lk_vnoise(vec2<f32>(u.time.x * 0.2, 3.0), 21u) - 0.5), 0.75, 1.0);
    return lk_screen(c, col * (g * u.k0.x * 0.9));
}
",
    cpu: light_leak,
    ..Pass::BASE
};

fn vignette(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let q = (uv - 0.5) * 2.0;
    let round = (q * v2(0.85, 1.0)).length() * 0.78;
    let frame = q.x.abs().max(q.y.abs());
    let d = mix(round, frame, k[2]);
    let v = smoothstep(0.95 - 0.55 * k[1] - 0.25, 1.05, d);
    c * (1.0 - k[0] * 0.85 * v)
}

/// A vignette: the corners darkened.
pub const VIGNETTE: Pass = Pass {
    name: "vignette",
    family: F,
    about: "a vignette: the edges darkened, round as a lens does or along the frame, soft or hard",
    knobs: &[
        knob(
            "amount",
            "how dark the edges",
            (0.0, 1.0, 0.4),
            "share",
            (0.4, 0.3),
        ),
        knob(
            "softness",
            "how far in the darkening starts",
            (0.0, 1.0, 0.5),
            "share",
            (0.3, 0.2),
        ),
        knob(
            "shape",
            "round (0) to the frame's shape (1)",
            (0.0, 1.0, 0.0),
            "share",
            (0.0, 0.1),
        ),
    ],
    natural: &[
        natural("amount", Driver::Quiet, 0.8),
        natural("softness", Driver::Kick, 0.6),
    ],
    pairs: &["grain", "curve", "sepia"],
    wgsl: r"
fn look_vignette(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let q = (uv - 0.5) * 2.0;
    let round = length(q * vec2<f32>(0.85, 1.0)) * 0.78;
    let frame = max(abs(q.x), abs(q.y));
    let d = mix(round, frame, u.k0.z);
    let v = smoothstep(0.95 - 0.55 * u.k0.y - 0.25, 1.05, d);
    return c * (1.0 - u.k0.x * 0.85 * v);
}
",
    cpu: vignette,
    ..Pass::BASE
};

fn polaroid(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let ar = u.size[0] / u.size[1];
    let side = 0.035 * k[0];
    let (bx, top, bottom) = (side / ar, side, side * 3.2);
    let inner = v2(
        (uv.x - bx) / (1.0 - 2.0 * bx),
        (uv.y - top) / (1.0 - top - bottom),
    );
    let c = s.src(inner.clamp(0.0, 1.0));
    let l = luma(c);
    let faded = mix3(c, c * 0.82 + v3(0.1, 0.12, 0.13), k[1]);
    let warm = faded * v3(1.0 + 0.08 * k[2], 1.0 + 0.02 * k[2], 1.0 - 0.08 * k[2]);
    let warm = mix3(
        warm,
        warm + v3(0.04, 0.02, -0.02) * k[1],
        smoothstep(0.5, 1.0, l),
    );
    let px = 1.5 * u.size[3];
    let ix = smoothstep(-px, px, inner.x) * (1.0 - smoothstep(1.0 - px, 1.0 + px, inner.x));
    let iy = smoothstep(-px, px, inner.y) * (1.0 - smoothstep(1.0 - px, 1.0 + px, inner.y));
    let paper =
        v3(0.93, 0.91, 0.86) * (0.97 + 0.03 * vnoise(uv * v2(u.size[0], u.size[1]) / 3.0, 31));
    mix3(paper, warm, ix * iy)
}

/// An instant photo: its white frame, its fade.
pub const POLAROID: Pass = Pass {
    name: "polaroid",
    family: F,
    about: "an instant photo: the picture in a white frame with a wide bottom, faded and warm (family, nostalgia, summer)",
    knobs: &[
        knob("frame", "how wide the white frame (0: none)", (0.0, 1.0, 0.6), "share", (0.0, 0.25)),
        knob("fade", "how faded, lifted and teal in the shadows", (0.0, 1.0, 0.5), "share", (0.3, 0.08)),
        knob("warmth", "cooler (-) or warmer (+)", (-1.0, 1.0, 0.3), "share", (0.5, 0.01)),
    ],
    cost: 1.2,
    natural: &[natural("warmth", Driver::Part, 0.6), natural("fade", Driver::Quiet, 0.5)],
    pairs: &["light_leak", "grain", "vignette"],
    wgsl: r"
fn look_polaroid(uv: vec2<f32>) -> vec3<f32> {
    let ar = u.size.x / u.size.y;
    let side = 0.035 * u.k0.x;
    let bx = side / ar;
    let top = side;
    let bottom = side * 3.2;
    let inner = vec2<f32>((uv.x - bx) / (1.0 - 2.0 * bx), (uv.y - top) / (1.0 - top - bottom));
    let c = lk_src(clamp(inner, vec2<f32>(0.0), vec2<f32>(1.0)));
    let l = lk_luma(c);
    let faded = mix(c, c * 0.82 + vec3<f32>(0.1, 0.12, 0.13), u.k0.y);
    var warm = faded * vec3<f32>(1.0 + 0.08 * u.k0.z, 1.0 + 0.02 * u.k0.z, 1.0 - 0.08 * u.k0.z);
    warm = mix(warm, warm + vec3<f32>(0.04, 0.02, -0.02) * u.k0.y, smoothstep(0.5, 1.0, l));
    let px = 1.5 * u.size.w;
    let ix = smoothstep(-px, px, inner.x) * (1.0 - smoothstep(1.0 - px, 1.0 + px, inner.x));
    let iy = smoothstep(-px, px, inner.y) * (1.0 - smoothstep(1.0 - px, 1.0 + px, inner.y));
    let paper = vec3<f32>(0.93, 0.91, 0.86) * (0.97 + 0.03 * lk_vnoise(uv * u.size.xy / 3.0, 31u));
    return mix(paper, warm, ix * iy);
}
",
    cpu: polaroid,
    ..Pass::BASE
};

fn film_damage(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let f = film_frame(u.time[0]);
    let px = u.px1080(uv);
    let cellv = (px / 36.0).floor();
    let h = hash2(cell(cellv.x), cell(cellv.y), f);
    let at = (cellv
        + v2(
            hash2(cell(cellv.x), cell(cellv.y), f + 1),
            hash2(cell(cellv.x), cell(cellv.y), f + 2),
        ))
        * 36.0;
    let r = 0.6 + 2.2 * hash2(cell(cellv.x), cell(cellv.y), f + 3);
    let speck = (1.0 - smoothstep(r - 0.7, r + 0.7, (px - at).length())) * step(h, k[0] * 0.06);
    let dark = step(0.35, hash2(cell(cellv.x), cell(cellv.y), f + 4));
    let col = (px.x / 140.0).floor();
    let sx = (col + hash2(cell(col), 0, f / 3 + 7)) * 140.0;
    let line = (1.0 - smoothstep(0.4, 1.2, (px.x - sx).abs()))
        * step(hash2(cell(col), 1, f / 3 + 7), k[0] * k[1] * 0.08);
    let damage = speck * mix(0.9, -0.9, dark) + line * 0.35;
    (c - s3(damage * (0.5 + 0.5 * k[1]))).clamp(0.0, 1.5)
}

/// Dust and scratches.
pub const FILM_DAMAGE: Pass = Pass {
    name: "film_damage",
    family: F,
    about: "dust specks and scratches, new each film frame: an old print, a silent film",
    knobs: &[
        knob(
            "amount",
            "how much dust",
            (0.0, 1.0, 0.4),
            "share",
            (0.5, 0.01),
        ),
        knob(
            "age",
            "how worn: scratches and darker dirt",
            (0.0, 1.0, 0.5),
            "share",
            (0.3, 0.01),
        ),
    ],
    cells: true,
    natural: &[
        natural("amount", Driver::Snare, 0.9),
        natural("age", Driver::Bass, 0.5),
    ],
    pairs: &["sepia", "gate_weave", "grain", "vignette"],
    wgsl: r"
fn look_film_damage(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let f = u32(floor(max(u.time.x * 24.0, 0.0)));
    let px = lk_px1080(uv);
    let cv = floor(px / 36.0);
    let cx = i32(cv.x);
    let cy = i32(cv.y);
    let h = lk_hash2(cx, cy, f);
    let at = (cv + vec2<f32>(lk_hash2(cx, cy, f + 1u), lk_hash2(cx, cy, f + 2u))) * 36.0;
    let r = 0.6 + 2.2 * lk_hash2(cx, cy, f + 3u);
    let speck = (1.0 - smoothstep(r - 0.7, r + 0.7, length(px - at))) * step(h, u.k0.x * 0.06);
    let dark = step(0.35, lk_hash2(cx, cy, f + 4u));
    let col = floor(px.x / 140.0);
    let sx = (col + lk_hash2(i32(col), 0, f / 3u + 7u)) * 140.0;
    let line = (1.0 - smoothstep(0.4, 1.2, abs(px.x - sx))) * step(lk_hash2(i32(col), 1, f / 3u + 7u), u.k0.x * u.k0.y * 0.08);
    let damage = speck * mix(0.9, -0.9, dark) + line * 0.35;
    return clamp(c - vec3<f32>(damage * (0.5 + 0.5 * u.k0.y)), vec3<f32>(0.0), vec3<f32>(1.5));
}
",
    cpu: film_damage,
    ..Pass::BASE
};

/// The film family's WGSL the other passes share.
pub const SHARED: &str = RING_LIGHT;

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
    family: F,
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
    family: F,
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
