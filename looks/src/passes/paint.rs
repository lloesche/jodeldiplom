//! Paint and drawing: the picture made of strokes, washes and grains of
//! a medium on its ground (canvas, paper), a pencil's sketch, chalk on a
//! board. Each reads the picture round a
//! pixel a few times; their sizes are pixels at 1080 lines.

use crate::drive::Driver;
use crate::math::*;
use crate::pass::{knob, natural, Ctx, Family, Pass, Src};
use crate::passes::print::edges;

const A: Family = Family::Paint;

/// Hatching's cover at `p` (pixels at 1080 lines) for darkness `dark`:
/// lines `spacing` apart at `angle` radians, wobbling, thicker the darker
/// past `start`.
fn hatch(p: V2, dark: f32, (spacing, angle, start): (f32, f32, f32)) -> f32 {
    let q = turn(p, angle);
    let t = (fract(q.y / spacing + 0.25 * vnoise(q * 0.04, 301)) - 0.5).abs() * 2.0;
    let w = sat((dark - start) * 1.6) * 0.7;
    let aa = 1.2 / spacing;
    (1.0 - smoothstep(w - aa, w + aa, t)) * step(0.001, w)
}

/// The paint family's WGSL the passes share.
pub const SHARED: &str = r"
fn lk_hatch(p: vec2<f32>, dark: f32, spacing: f32, angle: f32, start: f32) -> f32 {
    let q = lk_turn(p, angle);
    let t = abs(fract(q.y / spacing + 0.25 * lk_vnoise(q * 0.04, 301u)) - 0.5) * 2.0;
    let w = lk_sat((dark - start) * 1.6) * 0.7;
    let aa = 1.2 / spacing;
    return (1.0 - smoothstep(w - aa, w + aa, t)) * step(0.001, w);
}
";

fn kuwahara(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let r = k[0] * u.scale();
    let px = v2(u.size[2], u.size[3]);
    let mut acc = s3(0.0);
    let mut wsum = 0.0;
    for q in 0..4u32 {
        let sx = if q % 2 == 0 { -1.0 } else { 1.0 };
        let sy = if q / 2 == 0 { -1.0 } else { 1.0 };
        let mut m = s3(0.0);
        let mut m2 = 0.0;
        for t in 0..4u32 {
            let a = 0.3 + 0.7 * (t % 2) as f32;
            let b = 0.3 + 0.7 * (t / 2) as f32;
            let tap = s.src(uv + v2(sx * a, sy * b) * px * r);
            m = m + tap;
            m2 += luma(tap) * luma(tap);
        }
        m = m / 4.0;
        let var = (m2 / 4.0 - luma(m) * luma(m)).max(0.0);
        let w = 1.0 / (1e-4 + var * 60.0).powi(2);
        acc = acc + m * w;
        wsum += w;
    }
    mix3(c, acc / wsum, k[1])
}

/// A painterly smoothing that keeps edges.
pub const KUWAHARA: Pass = Pass {
    name: "kuwahara",
    family: A,
    about: "a painterly smear that keeps the edges: flat patches of colour like brushwork (oil, gouache, anime backgrounds)",
    knobs: &[
        knob("radius", "the brush's size", (2.0, 10.0, 4.0), "px", (3.0, 0.02)),
        knob("amount", "how much of the painting over the picture", (0.0, 1.0, 1.0), "share", (0.3, 0.02)),
    ],
    cost: 4.0,
    natural: &[natural("radius", Driver::Bass, 0.8)],
    pairs: &["canvas", "oil_paint", "saturation"],
    wgsl: r"
fn look_kuwahara(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let r = u.k0.x * lk_scale();
    var acc = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var q = 0u; q < 4u; q = q + 1u) {
        let sx = select(1.0, -1.0, q % 2u == 0u);
        let sy = select(1.0, -1.0, q / 2u == 0u);
        var m = vec3<f32>(0.0);
        var m2 = 0.0;
        for (var t = 0u; t < 4u; t = t + 1u) {
            let a = 0.3 + 0.7 * f32(t % 2u);
            let b = 0.3 + 0.7 * f32(t / 2u);
            let tap = lk_src(uv + vec2<f32>(sx * a, sy * b) * u.size.zw * r);
            m = m + tap;
            m2 = m2 + lk_luma(tap) * lk_luma(tap);
        }
        m = m / 4.0;
        let v = max(m2 / 4.0 - lk_luma(m) * lk_luma(m), 0.0);
        let w = 1.0 / pow(1e-4 + v * 60.0, 2.0);
        acc = acc + m * w;
        wsum = wsum + w;
    }
    return mix(c, acc / wsum, u.k0.y);
}
",
    cpu: kuwahara,
    ..Pass::BASE
};

fn watercolor(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let p = u.px1080(uv);
    let wob = v2(vnoise(p / 30.0, 81) - 0.5, vnoise(p / 30.0 + 7.0, 82) - 0.5) * (6.0 * k[1]);
    let at = uv + wob * u.scale() * v2(u.size[2], u.size[3]);
    let c = s.src(at);
    let pxs = v2(u.size[2], u.size[3]) * (6.0 * u.scale());
    let mut b = c;
    for i in 0..8u32 {
        let a = i as f32 * (TAU / 8.0) + 0.4;
        b = b + s.src(at + v2(a.cos(), a.sin()) * pxs);
    }
    let b = b / 9.0;
    let wash = mix3(c, b, k[0]);
    let paper = v3(0.97, 0.96, 0.93) * (0.94 + 0.06 * fbm(p / 9.0, 83));
    let pigment = mix3(s3(1.0), wash, 0.9);
    let edge = (c - b).length();
    let gran = 1.0 - k[3] * 0.25 * vnoise(p / 2.0, 84);
    let o = paper * pigment * (1.0 - k[1] * edge * 1.2) * mix(1.0, gran, 1.0 - luma(wash));
    mix3(s.src(uv), o.max(0.0), k[2])
}

/// Watercolour: washes, darker pigment at their edges, the paper's grain.
pub const WATERCOLOR: Pass = Pass {
    name: "watercolor",
    family: A,
    about: "watercolour: soft washes that bleed at wobbly edges, pigment pooling darker there, the paper showing through (folk, childhood, gentle songs)",
    knobs: &[
        knob("wash", "how far the colours wash together", (0.0, 1.0, 0.6), "share", (0.4, 0.02)),
        knob("bleed", "how wobbly the edges and how dark they pool", (0.0, 1.0, 0.5), "share", (0.4, 0.04)),
        knob("amount", "how much of the painting over the picture", (0.0, 1.0, 1.0), "share", (0.3, 0.15)),
        knob("granulation", "the pigment's grain in the darks", (0.0, 1.0, 0.4), "share", (0.2, 0.02)),
    ],
    cost: 2.5,
    relief: 0.2,
    natural: &[natural("bleed", Driver::Vocals, 0.7), natural("wash", Driver::Bass, 0.5)],
    pairs: &["canvas", "lift_gamma_gain", "saturation"],
    wgsl: r"
fn look_watercolor(uv: vec2<f32>) -> vec3<f32> {
    let p = lk_px1080(uv);
    let wob = vec2<f32>(lk_vnoise(p / 30.0, 81u) - 0.5, lk_vnoise(p / 30.0 + 7.0, 82u) - 0.5) * (6.0 * u.k0.y);
    let at = uv + wob * lk_scale() * u.size.zw;
    let c = lk_src(at);
    let pxs = u.size.zw * (6.0 * lk_scale());
    var b = c;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let a = f32(i) * (LK_TAU / 8.0) + 0.4;
        b = b + lk_src(at + vec2<f32>(cos(a), sin(a)) * pxs);
    }
    b = b / 9.0;
    let wash = mix(c, b, u.k0.x);
    let paper = vec3<f32>(0.97, 0.96, 0.93) * (0.94 + 0.06 * lk_fbm(p / 9.0, 83u));
    let pigment = mix(vec3<f32>(1.0), wash, 0.9);
    let edge = length(c - b);
    let gran = 1.0 - u.k0.w * 0.25 * lk_vnoise(p / 2.0, 84u);
    let o = paper * pigment * (1.0 - u.k0.y * edge * 1.2) * mix(1.0, gran, 1.0 - lk_luma(wash));
    return mix(lk_src(uv), max(o, vec3<f32>(0.0)), u.k0.z);
}
",
    cpu: watercolor,
    ..Pass::BASE
};

fn oil_paint(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let p = u.px1080(uv);
    let a = fbm(p / 90.0, 91) * TAU * 2.0;
    let dir = v2(a.cos(), a.sin()) * (k[0] * u.scale()) * v2(u.size[2], u.size[3]);
    let mut m = s3(0.0);
    for i in 0..5u32 {
        let t = (i as f32 - 2.0) * 0.5 * k[2];
        m = m + s.src(uv + dir * t);
    }
    let m = m / 5.0;
    let side = v2(-dir.y, dir.x);
    let stroke = vnoise(
        v2(
            (p.x * a.cos() + p.y * a.sin()) / (k[0] * 4.0),
            (p.y * a.cos() - p.x * a.sin()) / k[0],
        ),
        92,
    );
    let h1 = luma(s.src(uv + side)) + 0.15 * stroke;
    let h0 = luma(s.src(uv - side)) + 0.15 * vnoise(v2(stroke * 3.0, 1.0), 93);
    let lit = 1.0 + k[1] * (h1 - h0) * 2.5;
    mix3(s.src(uv), (m * lit).max(0.0), k[3])
}

/// Oil paint: strokes that follow a flow, their ridges catching the light.
pub const OIL_PAINT: Pass = Pass {
    name: "oil_paint",
    family: A,
    about: "oil paint: the colour smeared in strokes that follow a flow, their impasto ridges lit from one side (classic, ballads, folk portraits)",
    knobs: &[
        knob("brush", "the strokes' size", (2.0, 12.0, 5.0), "px", (2.0, 0.02)),
        knob("relief", "how much the ridges catch the light", (0.0, 1.0, 0.5), "share", (0.4, 0.03)),
        knob("smear", "how long the strokes", (0.0, 1.0, 0.6), "share", (0.4, 0.02)),
        knob("amount", "how much of the painting over the picture", (0.0, 1.0, 1.0), "share", (0.3, 0.03)),
    ],
    cost: 3.0,
    relief: 0.2,
    natural: &[natural("relief", Driver::Bass, 0.8), natural("smear", Driver::Vocals, 0.6)],
    pairs: &["kuwahara", "canvas", "vignette"],
    wgsl: r"
fn look_oil_paint(uv: vec2<f32>) -> vec3<f32> {
    let p = lk_px1080(uv);
    let a = lk_fbm(p / 90.0, 91u) * LK_TAU * 2.0;
    let dir = vec2<f32>(cos(a), sin(a)) * (u.k0.x * lk_scale()) * u.size.zw;
    var m = vec3<f32>(0.0);
    for (var i = 0u; i < 5u; i = i + 1u) {
        let t = (f32(i) - 2.0) * 0.5 * u.k0.z;
        m = m + lk_src(uv + dir * t);
    }
    m = m / 5.0;
    let side = vec2<f32>(-dir.y, dir.x);
    let stroke = lk_vnoise(vec2<f32>((p.x * cos(a) + p.y * sin(a)) / (u.k0.x * 4.0), (p.y * cos(a) - p.x * sin(a)) / u.k0.x), 92u);
    let h1 = lk_luma(lk_src(uv + side)) + 0.15 * stroke;
    let h0 = lk_luma(lk_src(uv - side)) + 0.15 * lk_vnoise(vec2<f32>(stroke * 3.0, 1.0), 93u);
    let lit = 1.0 + u.k0.y * (h1 - h0) * 2.5;
    return mix(lk_src(uv), max(m * lit, vec3<f32>(0.0)), u.k0.w);
}
",
    cpu: oil_paint,
    ..Pass::BASE
};

fn canvas(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let p = u.px1080(uv) / k[2];
    let weave =
        0.5 + 0.25 * ((p.x * 1.9).sin() + (p.y * 1.9).sin()) * (0.6 + 0.4 * vnoise(p / 6.0, 95));
    let fibre = fbm(v2(p.x / 3.0, p.y / 14.0), 96) * 0.6 + 0.4 * vnoise(p / 1.5, 97);
    let tex = mix(weave, fibre, k[0]);
    c * (1.0 - k[1] * 0.3 * (1.0 - tex)) + s3(k[1] * 0.04 * (tex - 0.5))
}

/// The ground under the picture: canvas or paper.
pub const CANVAS: Pass = Pass {
    name: "canvas",
    family: A,
    about: "the ground showing through: canvas weave or paper fibre over the whole picture, the finish of every painted or drawn look",
    knobs: &[
        knob("kind", "canvas weave (0) to paper fibre (1)", (0.0, 1.0, 0.0), "share", (0.0, 0.02)),
        knob("amount", "how strongly the ground shows", (0.0, 1.0, 0.4), "share", (0.4, 0.04)),
        knob("scale", "the weave's or fibre's size", (0.5, 3.0, 1.0), "share", (0.3, 0.01)),
    ],
    natural: &[natural("amount", Driver::Drums, 0.6)],
    pairs: &["oil_paint", "watercolor", "charcoal", "woodcut"],
    wgsl: r"
fn look_canvas(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let p = lk_px1080(uv) / u.k0.z;
    let weave = 0.5 + 0.25 * (sin(p.x * 1.9) + sin(p.y * 1.9)) * (0.6 + 0.4 * lk_vnoise(p / 6.0, 95u));
    let fibre = lk_fbm(vec2<f32>(p.x / 3.0, p.y / 14.0), 96u) * 0.6 + 0.4 * lk_vnoise(p / 1.5, 97u);
    let tex = mix(weave, fibre, u.k0.x);
    return c * (1.0 - u.k0.y * 0.3 * (1.0 - tex)) + vec3<f32>(u.k0.y * 0.04 * (tex - 0.5));
}
",
    cpu: canvas,
    ..Pass::BASE
};

fn charcoal(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let d = v2(u.size[2], u.size[3]) * (3.0 * u.scale());
    let blur = (luma(s.src(uv + d))
        + luma(s.src(uv - d))
        + luma(s.src(uv + v2(d.x, -d.y)))
        + luma(s.src(uv + v2(-d.x, d.y))))
        * 0.25;
    let l = mix(luma(c), blur, k[1]);
    let p = u.px1080(uv);
    let q = turn(p, 0.6);
    let stroke = vnoise(v2(q.x / 3.0, q.y / 18.0), 101);
    let tooth = vnoise(p / 1.7, 102) * k[2] + 0.5 * (1.0 - k[2]);
    let dark = (1.0 - l).powf(1.0 + k[3]);
    let deposit = smoothstep(tooth - 0.3, tooth + 0.3, dark + 0.25 * (stroke - 0.5));
    let o = mix3(v3(0.92, 0.9, 0.86), v3(0.1, 0.1, 0.11), deposit);
    mix3(c, o, k[0])
}

/// Charcoal on paper.
pub const CHARCOAL: Pass = Pass {
    name: "charcoal",
    family: A,
    about: "charcoal or chalk on paper: smudged darks in strokes, the paper's tooth catching the dust (blues, folk, grief, sketches)",
    knobs: &[
        knob("amount", "how much of the drawing over the picture", (0.0, 1.0, 1.0), "share", (0.3, 0.4)),
        knob("smudge", "how smudged", (0.0, 1.0, 0.4), "share", (0.4, 0.02)),
        knob("grain", "how much of the paper's tooth shows", (0.0, 1.0, 0.6), "share", (0.2, 0.03)),
        knob("contrast", "how dark the darks", (0.0, 1.0, 0.5), "share", (0.3, 0.25)),
    ],
    cost: 1.8,
    relief: 0.3,
    natural: &[natural("smudge", Driver::Bass, 0.7)],
    pairs: &["canvas", "vignette", "sepia"],
    wgsl: r"
fn look_charcoal(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let d = u.size.zw * (3.0 * lk_scale());
    let blur = (lk_luma(lk_src(uv + d)) + lk_luma(lk_src(uv - d)) + lk_luma(lk_src(uv + vec2<f32>(d.x, -d.y))) + lk_luma(lk_src(uv + vec2<f32>(-d.x, d.y)))) * 0.25;
    let l = mix(lk_luma(c), blur, u.k0.y);
    let p = lk_px1080(uv);
    let q = lk_turn(p, 0.6);
    let stroke = lk_vnoise(vec2<f32>(q.x / 3.0, q.y / 18.0), 101u);
    let tooth = lk_vnoise(p / 1.7, 102u) * u.k0.z + 0.5 * (1.0 - u.k0.z);
    let dark = pow(1.0 - l, 1.0 + u.k0.w);
    let deposit = smoothstep(tooth - 0.3, tooth + 0.3, dark + 0.25 * (stroke - 0.5));
    let o = mix(vec3<f32>(0.92, 0.9, 0.86), vec3<f32>(0.1, 0.1, 0.11), deposit);
    return mix(c, o, u.k0.x);
}
",
    cpu: charcoal,
    ..Pass::BASE
};

fn pastel(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let p = u.px1080(uv);
    let a = 0.9 + 0.6 * (vnoise(p / 120.0, 111) - 0.5);
    let q = turn(p, a);
    let stroke = vnoise(v2(q.x / (k[1] * 0.6), q.y / (k[1] * 3.0)), 112);
    let tooth = vnoise(p / 1.4, 113);
    let col = saturate(c, k[3]);
    let cover = smoothstep(
        0.15,
        0.6,
        (1.0 - luma(c)) * 0.6 + 0.35 + 0.3 * (stroke - 0.5) - k[2] * 0.35 * tooth,
    );
    let paper = v3(0.95, 0.93, 0.88);
    let o = mix3(paper, col * (0.9 + 0.2 * stroke), cover);
    mix3(c, o, k[0])
}

/// Wax crayon or soft pastel.
pub const PASTEL: Pass = Pass {
    name: "pastel",
    family: A,
    about: "wax crayon or soft pastel: colour laid in strokes, the paper's tooth showing white through it (children's songs, childhood, sunny pop)",
    knobs: &[
        knob("amount", "how much of the drawing over the picture", (0.0, 1.0, 1.0), "share", (0.3, 0.25)),
        knob("stroke", "the strokes' size", (2.0, 12.0, 6.0), "px", (3.0, 0.02)),
        knob("tooth", "how much paper shows through", (0.0, 1.0, 0.5), "share", (0.3, 0.1)),
        knob("vivid", "the colours' saturation", (0.5, 1.5, 1.1), "share", (0.2, 0.03)),
    ],
    cost: 1.2,
    relief: 0.3,
    natural: &[natural("stroke", Driver::Drums, 0.6)],
    pairs: &["canvas", "saturation", "marker"],
    wgsl: r"
fn look_pastel(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let p = lk_px1080(uv);
    let a = 0.9 + 0.6 * (lk_vnoise(p / 120.0, 111u) - 0.5);
    let q = lk_turn(p, a);
    let stroke = lk_vnoise(vec2<f32>(q.x / (u.k0.y * 0.6), q.y / (u.k0.y * 3.0)), 112u);
    let tooth = lk_vnoise(p / 1.4, 113u);
    let col = lk_saturate(c, u.k0.w);
    let cover = smoothstep(0.15, 0.6, (1.0 - lk_luma(c)) * 0.6 + 0.35 + 0.3 * (stroke - 0.5) - u.k0.z * 0.35 * tooth);
    let paper = vec3<f32>(0.95, 0.93, 0.88);
    let o = mix(paper, col * (0.9 + 0.2 * stroke), cover);
    return mix(c, o, u.k0.x);
}
",
    cpu: pastel,
    ..Pass::BASE
};

fn marker(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let n = k[1];
    let flat = saturate(c, 1.3).clamp(0.0, 1.0).map(|x| {
        let y = x * n;
        (y.floor() + smoothstep(0.4, 0.6, fract(y))) / n
    });
    let p = u.px1080(uv);
    let q = turn(p, 0.785);
    let band = fract(q.y / k[2] + 0.3 * vnoise(v2(q.x / 60.0, 0.5), 121));
    let overlap = smoothstep(0.8, 0.95, band) + 1.0 - smoothstep(0.0, 0.08, band);
    let ink = flat * (1.0 - 0.22 * overlap * (1.0 - luma(flat)));
    let e = smoothstep(0.08, 0.22, edges(u, s, uv, 1.5 * u.scale()));
    let o = mix3(ink, v3(0.05, 0.05, 0.08), e * k[3]);
    mix3(c, o, k[0])
}

/// Felt-tip markers.
pub const MARKER: Pass = Pass {
    name: "marker",
    family: A,
    about: "felt-tip markers: flat bold colours in streaky strokes, darker where they overlap, a pen outline (pop, children's songs, humour)",
    knobs: &[
        knob("amount", "how much of the drawing over the picture", (0.0, 1.0, 1.0), "share", (0.3, 0.1)),
        knob("steps", "how many tones each colour has", (2.0, 6.0, 3.0), "steps", (0.0, 0.08)),
        knob("stroke", "the strokes' width", (4.0, 20.0, 10.0), "px", (3.0, 0.02)),
        knob("outline", "how dark the pen outline", (0.0, 1.0, 0.6), "share", (0.4, 0.05)),
    ],
    cost: 1.5,
    relief: 0.3,
    natural: &[natural("outline", Driver::Kick, 0.7)],
    pairs: &["canvas", "pastel", "posterize_ink"],
    wgsl: r"
fn look_marker(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let n = u.k0.y;
    let y = clamp(lk_saturate(c, 1.3), vec3<f32>(0.0), vec3<f32>(1.0)) * n;
    let flat = (floor(y) + smoothstep(vec3<f32>(0.4), vec3<f32>(0.6), fract(y))) / n;
    let p = lk_px1080(uv);
    let q = lk_turn(p, 0.785);
    let band = fract(q.y / u.k0.z + 0.3 * lk_vnoise(vec2<f32>(q.x / 60.0, 0.5), 121u));
    let overlap = smoothstep(0.8, 0.95, band) + 1.0 - smoothstep(0.0, 0.08, band);
    let ink = flat * (1.0 - 0.22 * overlap * (1.0 - lk_luma(flat)));
    let e = smoothstep(0.08, 0.22, lk_edges(uv, 1.5 * lk_scale()));
    let o = mix(ink, vec3<f32>(0.05, 0.05, 0.08), e * u.k0.w);
    return mix(c, o, u.k0.x);
}
",
    cpu: marker,
    ..Pass::BASE
};

fn pencil(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let dark = 1.0 - luma(c);
    let p = u.px1080(uv);
    let e = smoothstep(0.03, 0.15, edges(u, s, uv, u.scale()));
    let one = hatch(p, dark, (k[2], 0.7, 0.35)) * k[1];
    let two = hatch(p, dark, (k[2], 2.2, 0.62)) * k[1];
    let strokes = (e * 0.9).max(one).max(two);
    let grain = 0.8 + 0.2 * vnoise(p / 1.2, 302);
    let lead = mix3(
        v3(0.95, 0.94, 0.91),
        v3(0.22, 0.22, 0.25),
        sat(strokes * (0.5 + 0.5 * k[3]) * grain),
    );
    mix3(c, lead, k[0])
}

/// A pencil sketch.
pub const PENCIL: Pass = Pass {
    name: "pencil",
    family: A,
    about: "a graphite pencil sketch: outlines, hatching in the shadows crossed in the darkest (singer-songwriters, folk, childhood, memories)",
    knobs: &[
        knob("amount", "how much of the sketch over the picture", (0.0, 1.0, 1.0), "share", (0.3, 0.57)),
        knob("hatch", "how much hatching in the shadows", (0.0, 1.0, 0.6), "share", (0.4, 0.34)),
        knob("spacing", "the hatching's spacing", (2.0, 8.0, 3.5), "px", (1.0, 0.03)),
        knob("pressure", "how dark the lines", (0.0, 1.0, 0.5), "share", (0.4, 0.13)),
    ],
    cost: 1.5,
    relief: 0.3,
    natural: &[natural("pressure", Driver::Vocals, 0.6), natural("hatch", Driver::Drums, 0.5)],
    pairs: &["canvas", "sepia", "vignette"],
    wgsl: r"
fn look_pencil(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let dark = 1.0 - lk_luma(c);
    let p = lk_px1080(uv);
    let e = smoothstep(0.03, 0.15, lk_edges(uv, lk_scale()));
    let one = lk_hatch(p, dark, u.k0.z, 0.7, 0.35) * u.k0.y;
    let two = lk_hatch(p, dark, u.k0.z, 2.2, 0.62) * u.k0.y;
    let strokes = max(max(e * 0.9, one), two);
    let grain = 0.8 + 0.2 * lk_vnoise(p / 1.2, 302u);
    let lead = mix(vec3<f32>(0.95, 0.94, 0.91), vec3<f32>(0.22, 0.22, 0.25), lk_sat(strokes * (0.5 + 0.5 * u.k0.w) * grain));
    return mix(c, lead, u.k0.x);
}
",
    cpu: pencil,
    ..Pass::BASE
};

fn chalkboard(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let l = luma(c);
    let p = u.px1080(uv);
    let e = smoothstep(0.04, 0.18, edges(u, s, uv, 1.2 * u.scale()));
    let board = hsv(k[1], 0.35, 0.16) * (0.9 + 0.1 * fbm(p / 40.0, 311));
    let q = turn(p, -0.5);
    let stroke = vnoise(v2(q.x / 3.0, q.y / 14.0), 312);
    let fill = smoothstep(0.35, 0.85, l + 0.25 * (stroke - 0.5)) * k[2];
    let dust = 0.6 + 0.4 * vnoise(p / 1.3, 313);
    let cover = sat(e * 1.2).max(fill) * dust;
    let chalk = mix3(s3(0.92), saturate(c, 0.6) * 1.1, 0.35);
    mix3(c, mix3(board, chalk, sat(cover)), k[0])
}

/// Chalk on a blackboard.
pub const CHALKBOARD: Pass = Pass {
    name: "chalkboard",
    family: A,
    about: "chalk on a dark board: the outlines and the lights in dusty chalk strokes (school days, children's songs, humour, nostalgia)",
    knobs: &[
        knob("amount", "how much of the board over the picture", (0.0, 1.0, 1.0), "share", (0.3, 0.45)),
        knob("board", "the board's hue (green by default, or slate)", (0.0, 1.0, 0.4), "turns", (0.0, 0.01)),
        knob("chalk", "outlines only (0) to the lights filled in chalk (1)", (0.0, 1.0, 0.6), "share", (0.4, 0.25)),
    ],
    cost: 1.6,
    relief: 0.3,
    natural: &[natural("chalk", Driver::Chorus, 0.6)],
    pairs: &["pastel", "vignette"],
    wgsl: r"
fn look_chalkboard(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let l = lk_luma(c);
    let p = lk_px1080(uv);
    let e = smoothstep(0.04, 0.18, lk_edges(uv, 1.2 * lk_scale()));
    let board = lk_hsv(u.k0.y, 0.35, 0.16) * (0.9 + 0.1 * lk_fbm(p / 40.0, 311u));
    let q = lk_turn(p, -0.5);
    let stroke = lk_vnoise(vec2<f32>(q.x / 3.0, q.y / 14.0), 312u);
    let fill = smoothstep(0.35, 0.85, l + 0.25 * (stroke - 0.5)) * u.k0.z;
    let dust = 0.6 + 0.4 * lk_vnoise(p / 1.3, 313u);
    let cover = max(lk_sat(e * 1.2), fill) * dust;
    let chalk = mix(vec3<f32>(0.92), lk_saturate(c, 0.6) * 1.1, 0.35);
    return mix(c, mix(board, chalk, lk_sat(cover)), u.k0.x);
}
",
    cpu: chalkboard,
    ..Pass::BASE
};
