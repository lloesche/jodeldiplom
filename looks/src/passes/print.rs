//! Print: the picture as ink on paper, in screens of dots, lines and
//! toner. Dots and lines are drawn with soft edges a pixel wide, so a
//! screen never shimmers; their sizes are in pixels at 1080 lines, so a
//! look looks the same on every screen.

use crate::drive::Driver;
use crate::math::*;
use crate::pass::{knob, natural, Ctx, Family, Pass, Src};

const P: Family = Family::Print;

/// The screen's cell `uv` is in, its dots `s` pixels apart at `a` turns:
/// the cell's middle (as a `uv`) and where in the cell it is (-0.5 to
/// 0.5).
pub(crate) fn screen_cell(u: &Ctx, uv: V2, s: f32, a: f32) -> (V2, V2) {
    let size = v2(u.size[0], u.size[1]);
    let q = turn(uv * size, a * TAU) / s;
    let c = q.floor() + 0.5;
    (turn(c * s, -a * TAU) / size, q - c)
}

/// A dot `r` across (in cells: 0.71 fills one) at `local`, its edge a
/// pixel soft in cells `s` pixels wide.
pub(crate) fn dot_at(local: V2, r: f32, s: f32) -> f32 {
    let aa = 0.7 / s;
    1.0 - smoothstep(r - aa, r + aa, local.length())
}

/// The print family's WGSL the passes share.
pub const SHARED: &str = r"
fn lk_screen_cell(uv: vec2<f32>, s: f32, a: f32) -> vec4<f32> {
    let q = lk_turn(uv * u.size.xy, a * LK_TAU) / s;
    let c = floor(q) + 0.5;
    return vec4<f32>(lk_turn(c * s, -a * LK_TAU) / u.size.xy, q - c);
}
fn lk_dot(local: vec2<f32>, r: f32, s: f32) -> f32 {
    let aa = 0.7 / s;
    return 1.0 - smoothstep(r - aa, r + aa, length(local));
}
fn lk_edges(uv: vec2<f32>, w: f32) -> f32 {
    let d = u.size.zw * w;
    let gx = lk_luma(lk_src(uv + vec2<f32>(d.x, 0.0))) - lk_luma(lk_src(uv - vec2<f32>(d.x, 0.0)));
    let gy = lk_luma(lk_src(uv + vec2<f32>(0.0, d.y))) - lk_luma(lk_src(uv - vec2<f32>(0.0, d.y)));
    return abs(gx) + abs(gy);
}
";

/// How much the picture changes across `w` pixels at `uv` (four reads).
pub(crate) fn edges(u: &Ctx, s: &dyn Src, uv: V2, w: f32) -> f32 {
    let d = v2(u.size[2], u.size[3]) * w;
    let gx = luma(s.src(uv + v2(d.x, 0.0))) - luma(s.src(uv - v2(d.x, 0.0)));
    let gy = luma(s.src(uv + v2(0.0, d.y))) - luma(s.src(uv - v2(0.0, d.y)));
    gx.abs() + gy.abs()
}

const PAPER: V3 = v3(0.96, 0.95, 0.92);

fn halftone(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let size = k[0] * u.scale();
    let (centre, local) = screen_cell(u, uv, size, k[1]);
    let col = s.src(centre);
    let cover = 1.0 - luma(col) * mix(1.0, 0.85, k[2]);
    let ink = dot_at(local, cover.max(0.0).sqrt() * 0.62, size);
    let paper = mix3(PAPER, v3(0.97, 0.92, 0.82), k[3]);
    let inkc = mix3(v3(0.07, 0.07, 0.09), saturate(col, 1.3) * 0.9, k[2]);
    mix3(c, mix3(paper, inkc, ink), k[4])
}

/// A halftone screen.
pub const HALFTONE: Pass = Pass {
    name: "halftone",
    family: P,
    about: "a halftone screen: the picture as dots on paper, black ink or the picture's own colours (newsprint, comics, posters)",
    knobs: &[
        knob("dot", "the dots' spacing", (3.0, 30.0, 8.0), "px", (6.0, 0.08)),
        knob("angle", "the screen's angle", (0.0, 1.0, 0.125), "turns", (0.1, 0.0)),
        knob("ink", "black ink (0) to the picture's colours (1)", (0.0, 1.0, 0.0), "share", (0.0, 0.16)),
        knob("paper", "white (0) to yellowed (1) paper", (0.0, 1.0, 0.3), "share", (0.0, 0.02)),
        knob("amount", "how much of the print over the picture", (0.0, 1.0, 1.0), "share", (0.3, 0.2)),
    ],
    cost: 1.5,
    relief: 0.5,
    cells: true,
    natural: &[natural("dot", Driver::Drums, 0.8), natural("angle", Driver::Part, 0.5)],
    pairs: &["duotone", "posterize_ink", "curve"],
    wgsl: r"
fn look_halftone(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let size = u.k0.x * lk_scale();
    let cell = lk_screen_cell(uv, size, u.k0.y);
    let col = lk_src(cell.xy);
    let cover = 1.0 - lk_luma(col) * mix(1.0, 0.85, u.k0.z);
    let ink = lk_dot(cell.zw, sqrt(max(cover, 0.0)) * 0.62, size);
    let paper = mix(vec3<f32>(0.96, 0.95, 0.92), vec3<f32>(0.97, 0.92, 0.82), u.k0.w);
    let inkc = mix(vec3<f32>(0.07, 0.07, 0.09), lk_saturate(col, 1.3) * 0.9, u.k0.z);
    return mix(c, mix(paper, inkc, ink), u.k1.x);
}
",
    cpu: halftone,
    ..Pass::BASE
};

const PLATES: [(f32, V3); 4] = [
    (0.042, v3(0.0, 0.62, 0.9)),
    (0.208, v3(0.9, 0.05, 0.55)),
    (0.0, v3(1.0, 0.92, 0.0)),
    (0.125, v3(0.12, 0.12, 0.14)),
];

fn halftone_cmyk(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let size = k[0] * u.scale();
    let mut o = PAPER;
    for (i, (angle, ink)) in PLATES.into_iter().enumerate() {
        let off = v2((i as f32 * 1.7).cos(), (i as f32 * 1.7).sin())
            * (k[2] * 4.0 * u.scale())
            * v2(u.size[2], u.size[3]);
        let (centre, local) = screen_cell(u, uv + off, size, angle + k[1]);
        let col = s.src(centre);
        let kk = 1.0 - col.x.max(col.y).max(col.z);
        let inv = 1.0 / (1.0 - kk + 1e-4);
        let v = [
            (1.0 - col.x - kk) * inv,
            (1.0 - col.y - kk) * inv,
            (1.0 - col.z - kk) * inv,
            kk,
        ][i];
        let cov = dot_at(local, sat(v).sqrt() * 0.62, size);
        o = o * mix3(s3(1.0), ink, cov);
    }
    mix3(c, o, k[3])
}

/// Four-colour process print.
pub const HALFTONE_CMYK: Pass = Pass {
    name: "halftone_cmyk",
    family: P,
    about: "four-colour process print: cyan, magenta, yellow and black screens at their angles, slightly off register if wanted (comics, magazines, pop art)",
    knobs: &[
        knob("dot", "the dots' spacing", (3.0, 24.0, 7.0), "px", (4.0, 0.03)),
        knob("angle", "the screens turned together", (0.0, 1.0, 0.0), "turns", (0.05, 0.0)),
        knob("registration", "how far the plates are off register (up to 4 pixels)", (0.0, 1.0, 0.2), "share", (0.6, 0.0)),
        knob("amount", "how much of the print over the picture", (0.0, 1.0, 1.0), "share", (0.3, 0.2)),
    ],
    cost: 2.5,
    relief: 0.5,
    cells: true,
    natural: &[natural("registration", Driver::Snare, 0.9), natural("dot", Driver::Kick, 0.6)],
    pairs: &["posterize_ink", "saturation", "ben_day"],
    wgsl: r"
fn look_halftone_cmyk(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let size = u.k0.x * lk_scale();
    var o = vec3<f32>(0.96, 0.95, 0.92);
    var angles = array<f32, 4>(0.042, 0.208, 0.0, 0.125);
    var inks = array<vec3<f32>, 4>(vec3<f32>(0.0, 0.62, 0.9), vec3<f32>(0.9, 0.05, 0.55), vec3<f32>(1.0, 0.92, 0.0), vec3<f32>(0.12, 0.12, 0.14));
    for (var i = 0u; i < 4u; i = i + 1u) {
        let fi = f32(i);
        let off = vec2<f32>(cos(fi * 1.7), sin(fi * 1.7)) * (u.k0.z * 4.0 * lk_scale()) * u.size.zw;
        let cell = lk_screen_cell(uv + off, size, angles[i] + u.k0.y);
        let col = lk_src(cell.xy);
        let kk = 1.0 - max(max(col.x, col.y), col.z);
        let inv = 1.0 / (1.0 - kk + 1e-4);
        let vs = vec4<f32>((1.0 - col.x - kk) * inv, (1.0 - col.y - kk) * inv, (1.0 - col.z - kk) * inv, kk);
        let cov = lk_dot(cell.zw, sqrt(lk_sat(vs[i])) * 0.62, size);
        o = o * mix(vec3<f32>(1.0), inks[i], cov);
    }
    return mix(c, o, u.k0.w);
}
",
    cpu: halftone_cmyk,
    ..Pass::BASE
};

fn ben_day(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let l = luma(c);
    let n = k[1];
    let vivid = saturate(c, 1.6)
        .clamp(0.0, 1.0)
        .map(|x| (x * n + 0.5).floor() / n);
    let size = k[0] * u.scale();
    let (_, local) = screen_cell(u, uv, size, 0.125);
    let dot = dot_at(local, 0.36, size);
    let mid = smoothstep(0.35, 0.45, l) * (1.0 - smoothstep(0.7, 0.8, l));
    let flat = mix3(vivid, mix3(v3(1.0, 0.98, 0.94), vivid, dot), mid);
    let e = smoothstep(0.08, 0.22, edges(u, s, uv, 1.5 * u.scale()));
    let o = mix3(flat, v3(0.04, 0.03, 0.05), e * k[2]);
    mix3(c, o, k[3])
}

/// Ben-Day dots: flat colours, dotted midtones, inked outlines.
pub const BEN_DAY: Pass = Pass {
    name: "ben_day",
    family: P,
    about: "pop-art comic print: flat bold colours, Ben-Day dots in the midtones, black outlines",
    knobs: &[
        knob(
            "dot",
            "the dots' spacing",
            (4.0, 20.0, 10.0),
            "px",
            (5.0, 0.02),
        ),
        knob(
            "levels",
            "how many steps each colour has",
            (2.0, 5.0, 3.0),
            "steps",
            (0.0, 0.1),
        ),
        knob(
            "outline",
            "how black the outlines",
            (0.0, 1.0, 0.7),
            "share",
            (0.3, 0.05),
        ),
        knob(
            "amount",
            "how much of the print over the picture",
            (0.0, 1.0, 1.0),
            "share",
            (0.3, 0.2),
        ),
    ],
    cost: 2.0,
    relief: 0.5,
    cells: true,
    natural: &[
        natural("dot", Driver::Kick, 0.8),
        natural("outline", Driver::Drums, 0.5),
    ],
    pairs: &["halftone_cmyk", "saturation", "duotone"],
    wgsl: r"
fn look_ben_day(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let l = lk_luma(c);
    let n = u.k0.y;
    let vivid = floor(clamp(lk_saturate(c, 1.6), vec3<f32>(0.0), vec3<f32>(1.0)) * n + 0.5) / n;
    let size = u.k0.x * lk_scale();
    let cell = lk_screen_cell(uv, size, 0.125);
    let dot = lk_dot(cell.zw, 0.36, size);
    let mid = smoothstep(0.35, 0.45, l) * (1.0 - smoothstep(0.7, 0.8, l));
    let flat = mix(vivid, mix(vec3<f32>(1.0, 0.98, 0.94), vivid, dot), mid);
    let e = smoothstep(0.08, 0.22, lk_edges(uv, 1.5 * lk_scale()));
    let o = mix(flat, vec3<f32>(0.04, 0.03, 0.05), e * u.k0.z);
    return mix(c, o, u.k0.w);
}
",
    cpu: ben_day,
    ..Pass::BASE
};

fn posterize_ink(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let l = luma(c);
    let x = l * k[0];
    let q = (x.floor() + smoothstep(0.42, 0.58, fract(x))) / k[0];
    let cel = (saturate(c, k[3]) * (q / l.max(0.02))).clamp(0.0, 1.2);
    let e = smoothstep(0.06, 0.2, edges(u, s, uv, k[2] * u.scale()));
    mix3(cel, v3(0.04, 0.03, 0.05), e * k[1])
}

/// Cel shading: tones in steps, ink round the edges.
pub const POSTERIZE_INK: Pass = Pass {
    name: "posterize_ink",
    family: P,
    about: "cel shading: the tones in a few flat steps and ink lines round the edges (comics, cartoons, anime)",
    knobs: &[
        knob("steps", "how many tones", (2.0, 8.0, 4.0), "steps", (0.0, 0.05)),
        knob("ink", "how dark the lines", (0.0, 1.0, 0.7), "share", (0.3, 0.05)),
        knob("width", "how wide the lines", (0.5, 4.0, 1.5), "px", (1.5, 0.03)),
        knob("vivid", "the colours' saturation", (0.0, 2.0, 1.2), "share", (0.5, 0.05)),
    ],
    cost: 1.5,
    relief: 0.3,
    natural: &[natural("width", Driver::Kick, 0.8), natural("vivid", Driver::Chorus, 0.6)],
    pairs: &["halftone", "ben_day", "saturation"],
    wgsl: r"
fn look_posterize_ink(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let l = lk_luma(c);
    let x = l * u.k0.x;
    let q = (floor(x) + smoothstep(0.42, 0.58, fract(x))) / u.k0.x;
    let cel = clamp(lk_saturate(c, u.k0.w) * (q / max(l, 0.02)), vec3<f32>(0.0), vec3<f32>(1.2));
    let e = smoothstep(0.06, 0.2, lk_edges(uv, u.k0.z * lk_scale()));
    return mix(cel, vec3<f32>(0.04, 0.03, 0.05), e * u.k0.y);
}
",
    cpu: posterize_ink,
    ..Pass::BASE
};

fn risograph(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let d = v2(1.0, 0.4) * (k[2] * 6.0 * u.scale()) * v2(u.size[2], u.size[3]);
    let a = s.src(uv + d);
    let b = s.src(uv - d);
    let px = u.px1080(uv);
    let da = sat(1.0 - (a.y * 0.6 + a.z * 0.4) + (vnoise(px / 1.5, 41) - 0.5) * k[3] * 0.5);
    let db = sat(1.0 - (b.x * 0.7 + b.y * 0.3) + (vnoise(px / 1.5, 43) - 0.5) * k[3] * 0.5);
    let o = v3(0.97, 0.95, 0.9)
        * mix3(s3(1.0), hsv(k[0], 0.85, 1.0), da * 0.9)
        * mix3(s3(1.0), hsv(k[1], 0.8, 0.85), db * 0.9);
    mix3(c, o, k[4])
}

/// A risograph: two fluorescent inks, a little off register, grainy.
pub const RISOGRAPH: Pass = Pass {
    name: "risograph",
    family: P,
    about: "a risograph print: two bright inks laid one over the other, slightly off register, grainy (zines, indie, lo-fi)",
    knobs: &[
        knob("first", "the first ink's hue (fluorescent pink by default)", (0.0, 1.0, 0.93), "turns", (0.05, 0.22)),
        knob("second", "the second ink's hue (blue by default)", (0.0, 1.0, 0.6), "turns", (0.05, 0.18)),
        knob("misregister", "how far the inks are apart (up to 6 pixels)", (0.0, 1.0, 0.35), "share", (0.6, 0.0)),
        knob("grain", "how grainy the ink", (0.0, 1.0, 0.5), "share", (0.4, 0.02)),
        knob("amount", "how much of the print over the picture", (0.0, 1.0, 1.0), "share", (0.3, 0.2)),
    ],
    cost: 2.0,
    relief: 0.4,
    natural: &[natural("misregister", Driver::BassNote, 0.9), natural("grain", Driver::Drums, 0.5)],
    pairs: &["grain", "halftone", "canvas"],
    wgsl: r"
fn look_risograph(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let d = vec2<f32>(1.0, 0.4) * (u.k0.z * 6.0 * lk_scale()) * u.size.zw;
    let a = lk_src(uv + d);
    let b = lk_src(uv - d);
    let px = lk_px1080(uv);
    let da = lk_sat(1.0 - (a.y * 0.6 + a.z * 0.4) + (lk_vnoise(px / 1.5, 41u) - 0.5) * u.k0.w * 0.5);
    let db = lk_sat(1.0 - (b.x * 0.7 + b.y * 0.3) + (lk_vnoise(px / 1.5, 43u) - 0.5) * u.k0.w * 0.5);
    let o = vec3<f32>(0.97, 0.95, 0.9) * mix(vec3<f32>(1.0), lk_hsv(u.k0.x, 0.85, 1.0), da * 0.9) * mix(vec3<f32>(1.0), lk_hsv(u.k0.y, 0.8, 0.85), db * 0.9);
    return mix(c, o, u.k1.x);
}
",
    cpu: risograph,
    ..Pass::BASE
};

fn xerox(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let px = u.px1080(uv);
    let l = luma(c);
    let col = (px.x / 3.0).floor();
    let streak = (hash2(cell(col), 3, 61) - 0.85).max(0.0) * 1.2 * k[1];
    let v = l + (vnoise(px / 1.3, 51) - 0.5) * k[1] * 0.25 - streak;
    let w = 0.2 * (1.0 - k[2]) + 0.015;
    let b = smoothstep(k[0] - w, k[0] + w, v);
    let o = mix3(v3(0.06, 0.06, 0.07), v3(0.94, 0.94, 0.91), b);
    mix3(c, o, k[3])
}

/// A photocopy: black and white, toner noise and streaks.
pub const XEROX: Pass = Pass {
    name: "xerox",
    family: P,
    about: "a photocopy of a photocopy: hard black and white, toner speckle and streaks (punk flyers, zines, hardcore)",
    knobs: &[
        knob("threshold", "where black turns white: more black (+) or more white (-)", (0.2, 0.8, 0.5), "share", (0.1, 0.6)),
        knob("toner", "how much speckle and streaking", (0.0, 1.0, 0.5), "share", (0.5, 0.05)),
        knob("contrast", "soft grey (0) to hard black and white (1)", (0.0, 1.0, 0.7), "share", (0.2, 0.05)),
        knob("amount", "how much of the copy over the picture", (0.0, 1.0, 1.0), "share", (0.3, 0.3)),
    ],
    relief: 0.4,
    natural: &[natural("toner", Driver::Snare, 0.9), natural("threshold", Driver::Bass, 0.4)],
    pairs: &["duotone", "film_damage", "glitch"],
    wgsl: r"
fn look_xerox(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let px = lk_px1080(uv);
    let l = lk_luma(c);
    let col = floor(px.x / 3.0);
    let streak = max(lk_hash2(i32(col), 3, 61u) - 0.85, 0.0) * 1.2 * u.k0.y;
    let v = l + (lk_vnoise(px / 1.3, 51u) - 0.5) * u.k0.y * 0.25 - streak;
    let w = 0.2 * (1.0 - u.k0.z) + 0.015;
    let b = smoothstep(u.k0.x - w, u.k0.x + w, v);
    let o = mix(vec3<f32>(0.06, 0.06, 0.07), vec3<f32>(0.94, 0.94, 0.91), b);
    return mix(c, o, u.k0.w);
}
",
    cpu: xerox,
    ..Pass::BASE
};

fn woodcut(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let spacing = k[0] * u.scale();
    let q = turn(uv * v2(u.size[0], u.size[1]), k[1] * TAU) / spacing;
    let l = sat(luma(c) + 0.25 * k[2]);
    let y = q.y + 0.35 * l + 0.15 * (q.x * 0.35).sin();
    let t = (fract(y) - 0.5).abs() * 2.0;
    let aa = 1.2 / spacing;
    let mut cov = 1.0 - smoothstep(1.0 - l - aa, 1.0 - l + aa, t);
    cov = cov.max(1.0 - smoothstep(0.08, 0.16, l));
    cov *= 1.0 - smoothstep(0.86, 0.94, l);
    let ink = mix3(v3(0.05, 0.04, 0.04), u.pal(0) * 0.45, k[3]);
    let o = mix3(v3(0.93, 0.89, 0.8), ink, cov);
    mix3(c, o, k[4])
}

/// A woodcut or linocut: carved lines that follow the form.
pub const WOODCUT: Pass = Pass {
    name: "woodcut",
    family: P,
    about: "a woodcut or linocut: the picture carved in black lines that swell in the dark and follow the forms (folk, doom and black metal, protest posters)",
    knobs: &[
        knob("lines", "the lines' spacing", (3.0, 20.0, 7.0), "px", (3.0, 0.02)),
        knob("angle", "the lines' angle", (0.0, 1.0, 0.08), "turns", (0.05, 0.0)),
        knob("threshold", "darker (-) or lighter (+) overall", (-1.0, 1.0, 0.0), "share", (0.4, 0.4)),
        knob("ink", "black ink (0) to the cover's colour, darkened (1)", (0.0, 1.0, 0.0), "share", (0.0, 0.13)),
        knob("amount", "how much of the print over the picture", (0.0, 1.0, 1.0), "share", (0.3, 0.3)),
    ],
    cost: 1.0,
    relief: 0.4,
    natural: &[natural("threshold", Driver::Bass, 0.6), natural("lines", Driver::Kick, 0.5)],
    pairs: &["duotone", "canvas", "vignette"],
    wgsl: r"
fn look_woodcut(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let spacing = u.k0.x * lk_scale();
    let q = lk_turn(uv * u.size.xy, u.k0.y * LK_TAU) / spacing;
    let l = lk_sat(lk_luma(c) + 0.25 * u.k0.z);
    let y = q.y + 0.35 * l + 0.15 * sin(q.x * 0.35);
    let t = abs(fract(y) - 0.5) * 2.0;
    let aa = 1.2 / spacing;
    var cov = 1.0 - smoothstep(1.0 - l - aa, 1.0 - l + aa, t);
    cov = max(cov, 1.0 - smoothstep(0.08, 0.16, l));
    cov = cov * (1.0 - smoothstep(0.86, 0.94, l));
    let ink = mix(vec3<f32>(0.05, 0.04, 0.04), u.pal0.xyz * 0.45, u.k0.w);
    let o = mix(vec3<f32>(0.93, 0.89, 0.8), ink, cov);
    return mix(c, o, u.k1.x);
}
",
    cpu: woodcut,
    ..Pass::BASE
};

fn engraving(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let spacing = k[0] * u.scale();
    let dark = 1.0 - luma(c);
    let aa = 1.0 / spacing;
    let mut cov: f32 = 0.0;
    for i in 0..3u32 {
        let fi = i as f32;
        let on = step(fi + 0.5, k[2]);
        let q = turn(uv * v2(u.size[0], u.size[1]), k[1] * TAU + fi * 1.05) / spacing;
        let t = (fract(q.y) - 0.5).abs() * 2.0;
        let th = (fi + 1.0) / (k[2] + 1.0);
        let w = sat((dark - th) * 2.5) * 0.6;
        cov = cov.max(on * (1.0 - smoothstep(w - aa, w + aa, t)) * step(0.001, w));
    }
    let o = mix3(v3(0.95, 0.92, 0.84), v3(0.12, 0.08, 0.06), cov);
    mix3(c, o, k[3])
}

/// An engraving: layers of cross-hatching.
pub const ENGRAVING: Pass = Pass {
    name: "engraving",
    family: P,
    about: "an engraving: the tones in fine hatching, crossed in the darks (banknotes, old books, classical, folk)",
    knobs: &[
        knob("spacing", "the lines' spacing", (2.0, 10.0, 4.0), "px", (2.0, 0.02)),
        knob("angle", "the first lines' angle", (0.0, 1.0, 0.125), "turns", (0.05, 0.0)),
        knob("layers", "how many directions of hatching", (1.0, 3.0, 2.0), "steps", (0.0, 0.1)),
        knob("amount", "how much of the engraving over the picture", (0.0, 1.0, 1.0), "share", (0.3, 0.3)),
    ],
    cost: 1.3,
    relief: 0.4,
    natural: &[natural("spacing", Driver::Bass, 0.6)],
    pairs: &["sepia", "canvas", "vignette"],
    wgsl: r"
fn look_engraving(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let spacing = u.k0.x * lk_scale();
    let dark = 1.0 - lk_luma(c);
    let aa = 1.0 / spacing;
    var cov = 0.0;
    for (var i = 0u; i < 3u; i = i + 1u) {
        let fi = f32(i);
        let on = step(fi + 0.5, u.k0.z);
        let q = lk_turn(uv * u.size.xy, u.k0.y * LK_TAU + fi * 1.05) / spacing;
        let t = abs(fract(q.y) - 0.5) * 2.0;
        let th = (fi + 1.0) / (u.k0.z + 1.0);
        let w = lk_sat((dark - th) * 2.5) * 0.6;
        cov = max(cov, on * (1.0 - smoothstep(w - aa, w + aa, t)) * step(0.001, w));
    }
    let o = mix(vec3<f32>(0.95, 0.92, 0.84), vec3<f32>(0.12, 0.08, 0.06), cov);
    return mix(c, o, u.k0.w);
}
",
    cpu: engraving,
    ..Pass::BASE
};

fn stipple(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let size = k[0] * u.scale() * 1.6;
    let px = uv * v2(u.size[0], u.size[1]);
    let cv = (px / size).floor();
    let (x, y) = (cell(cv.x), cell(cv.y));
    let centre = (cv + v2(0.3 + 0.4 * hash2(x, y, 71), 0.3 + 0.4 * hash2(x, y, 72))) * size;
    let col = s.src(centre / v2(u.size[0], u.size[1]));
    let d = sat((1.0 - luma(col) - 0.5) * (1.0 + 1.5 * k[1]) + 0.5);
    let on = step(hash2(x, y, 73), d);
    let r = size * 0.28;
    let cov = on * (1.0 - smoothstep(r - 0.7, r + 0.7, (px - centre).length()));
    let o = mix3(PAPER, v3(0.08, 0.08, 0.1), cov);
    mix3(c, o, k[2])
}

/// Stippling: dots, denser in the dark.
pub const STIPPLE: Pass = Pass {
    name: "stipple",
    family: P,
    about: "stippling: the picture in ink dots, denser where it is dark (pen-and-ink, old scientific plates)",
    knobs: &[
        knob("dot", "the dots' size", (2.0, 10.0, 4.0), "px", (2.5, 0.08)),
        knob("contrast", "how sharply dark and light part", (0.0, 1.0, 0.5), "share", (0.3, 0.1)),
        knob("amount", "how much of the stippling over the picture", (0.0, 1.0, 1.0), "share", (0.3, 0.3)),
    ],
    cost: 1.3,
    relief: 0.4,
    cells: true,
    natural: &[natural("dot", Driver::Drums, 0.7)],
    pairs: &["sepia", "canvas"],
    wgsl: r"
fn look_stipple(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let size = u.k0.x * lk_scale() * 1.6;
    let px = uv * u.size.xy;
    let cv = floor(px / size);
    let x = i32(cv.x);
    let y = i32(cv.y);
    let centre = (cv + vec2<f32>(0.3 + 0.4 * lk_hash2(x, y, 71u), 0.3 + 0.4 * lk_hash2(x, y, 72u))) * size;
    let col = lk_src(centre / u.size.xy);
    let d = lk_sat((1.0 - lk_luma(col) - 0.5) * (1.0 + 1.5 * u.k0.y) + 0.5);
    let on = step(lk_hash2(x, y, 73u), d);
    let r = size * 0.28;
    let cov = on * (1.0 - smoothstep(r - 0.7, r + 0.7, length(px - centre)));
    let o = mix(vec3<f32>(0.96, 0.95, 0.92), vec3<f32>(0.08, 0.08, 0.1), cov);
    return mix(c, o, u.k0.z);
}
",
    cpu: stipple,
    ..Pass::BASE
};
