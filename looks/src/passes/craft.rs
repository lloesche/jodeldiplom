//! Hand-made media: a pencil sketch, chalk on a board, a paper cutout,
//! an embossed relief. Each keeps the picture's forms (its edges, its
//! tones) and draws them in the medium's marks.

use crate::drive::Driver;
use crate::math::*;
use crate::pass::{knob, natural, Ctx, Family, Pass, Src};
use crate::passes::print::edges;

fn px1080(u: &Ctx, uv: V2) -> V2 {
    uv * v2(u.size[0], u.size[1]) / u.scale()
}

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

/// The craft family's WGSL the passes share.
pub const SHARED: &str = r"
fn lk_hatch(p: vec2<f32>, dark: f32, spacing: f32, angle: f32, start: f32) -> f32 {
    let q = lk_turn(p, angle);
    let t = abs(fract(q.y / spacing + 0.25 * lk_vnoise(q * 0.04, 301u)) - 0.5) * 2.0;
    let w = lk_sat((dark - start) * 1.6) * 0.7;
    let aa = 1.2 / spacing;
    return (1.0 - smoothstep(w - aa, w + aa, t)) * step(0.001, w);
}
";

fn pencil(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let dark = 1.0 - luma(c);
    let p = px1080(u, uv);
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
    family: Family::Paint,
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
    let p = uv * u.size.xy / lk_scale();
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
    let p = px1080(u, uv);
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
    family: Family::Paint,
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
    let p = uv * u.size.xy / lk_scale();
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

/// The paper layer a tone of `l` is cut into, of `n`, its edge ragged at
/// `p`.
fn layer_of(l: f32, n: f32, p: V2) -> f32 {
    ((l + (vnoise(p / 3.0, 321) - 0.5) * 0.04) * n)
        .floor()
        .clamp(0.0, n - 1.0)
}

fn cutout(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let n = k[0];
    let c = s.src(uv);
    let p = px1080(u, uv);
    let q = layer_of(luma(c), n, p);
    let off = v2(-4.0, -4.0) * u.scale() * v2(u.size[2], u.size[3]);
    let above = layer_of(luma(s.src(uv + off)), n, p + v2(-4.0, -4.0));
    let shade = 1.0 - k[1] * 0.45 * step(q + 0.5, above);
    let tone = (q + 0.5) / n;
    let col = (saturate(c, 1.3) * (tone / luma(c).max(0.05))).clamp(0.0, 1.0);
    let fibre = 0.94 + 0.06 * fbm(v2(p.x / 2.0, p.y / 9.0), 322) * k[2] + 0.03 * (1.0 - k[2]);
    mix3(c, col * shade * fibre, k[3])
}

/// A collage of cut paper.
pub const CUTOUT: Pass = Pass {
    name: "cutout",
    family: Family::Medium,
    about: "cut paper: the tones as a few layers of coloured paper, each casting a little shadow on the one under it (indie pop, children's songs, whimsy, animation)",
    knobs: &[
        knob("layers", "how many layers of paper", (2.0, 6.0, 4.0), "steps", (0.0, 0.08)),
        knob("shadow", "how deep the layers' shadows", (0.0, 1.0, 0.6), "share", (0.4, 0.06)),
        knob("paper", "how much the paper's fibre shows", (0.0, 1.0, 0.5), "share", (0.2, 0.02)),
        knob("amount", "how much of the collage over the picture", (0.0, 1.0, 1.0), "share", (0.3, 0.1)),
    ],
    cost: 1.3,
    relief: 0.4,
    cells: true,
    natural: &[natural("shadow", Driver::Kick, 0.7)],
    pairs: &["canvas", "saturation", "posterize_ink"],
    wgsl: r"
fn lk_layer_of(l: f32, n: f32, p: vec2<f32>) -> f32 {
    return clamp(floor((l + (lk_vnoise(p / 3.0, 321u) - 0.5) * 0.04) * n), 0.0, n - 1.0);
}
fn look_cutout(uv: vec2<f32>) -> vec3<f32> {
    let n = u.k0.x;
    let c = lk_src(uv);
    let p = uv * u.size.xy / lk_scale();
    let q = lk_layer_of(lk_luma(c), n, p);
    let off = vec2<f32>(-4.0, -4.0) * lk_scale() * u.size.zw;
    let above = lk_layer_of(lk_luma(lk_src(uv + off)), n, p + vec2<f32>(-4.0, -4.0));
    let shade = 1.0 - u.k0.y * 0.45 * step(q + 0.5, above);
    let tone = (q + 0.5) / n;
    let col = clamp(lk_saturate(c, 1.3) * (tone / max(lk_luma(c), 0.05)), vec3<f32>(0.0), vec3<f32>(1.0));
    let fibre = 0.94 + 0.06 * lk_fbm(vec2<f32>(p.x / 2.0, p.y / 9.0), 322u) * u.k0.z + 0.03 * (1.0 - u.k0.z);
    return mix(c, col * shade * fibre, u.k0.w);
}
",
    cpu: cutout,
    ..Pass::BASE
};

fn emboss(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let a = k[1] * TAU;
    let d = v2(a.cos(), a.sin()) * (1.5 * u.scale()) * v2(u.size[2], u.size[3]);
    let r = (luma(s.src(uv + d)) - luma(s.src(uv - d))) * k[3] * 4.0;
    let metal = mix3(s3(0.62), v3(0.85, 0.68, 0.35), k[2]);
    let lit = sat(r);
    let o = metal * (0.6 + (0.5 + r) * 0.8) + s3(lit * lit * lit * 0.5);
    mix3(c, o.max(0.0), k[0])
}

/// A relief embossed in metal or stone.
pub const EMBOSS: Pass = Pass {
    name: "emboss",
    family: Family::Medium,
    about: "the picture embossed as a relief in stone, silver or gold, lit from one side (metal, power metal, anthems, epic and classical)",
    knobs: &[
        knob("amount", "how much of the relief over the picture", (0.0, 1.0, 1.0), "share", (0.3, 0.3)),
        knob("light", "where the light comes from", (0.0, 1.0, 0.125), "turns", (0.2, 0.01)),
        knob("metal", "grey stone or silver (0) to gold (1)", (0.0, 1.0, 0.0), "share", (0.0, 0.13)),
        knob("depth", "how deep the relief", (0.0, 1.0, 0.5), "share", (0.5, 0.05)),
    ],
    cost: 1.3,
    relief: 0.4,
    natural: &[natural("depth", Driver::Bass, 0.8), natural("light", Driver::BarPhase, 0.5)],
    pairs: &["glow", "vignette", "bleach_bypass"],
    wgsl: r"
fn look_emboss(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let a = u.k0.y * LK_TAU;
    let d = vec2<f32>(cos(a), sin(a)) * (1.5 * lk_scale()) * u.size.zw;
    let r = (lk_luma(lk_src(uv + d)) - lk_luma(lk_src(uv - d))) * u.k0.w * 4.0;
    let metal = mix(vec3<f32>(0.62), vec3<f32>(0.85, 0.68, 0.35), u.k0.z);
    let lit = lk_sat(r);
    let o = metal * (0.6 + (0.5 + r) * 0.8) + vec3<f32>(lit * lit * lit * 0.5);
    return mix(c, max(o, vec3<f32>(0.0)), u.k0.x);
}
",
    cpu: emboss,
    ..Pass::BASE
};
