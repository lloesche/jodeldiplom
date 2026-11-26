//! Colour: the grades every look starts from. Each knob is even to the
//! eye across its range (a lift's tenth looks like the next tenth), and
//! none of them ends in a broken picture: at its ends the picture is
//! strongly graded, never black, white or banded.

use crate::drive::Driver;
use crate::math::*;
use crate::pass::{knob, natural, Ctx, Family, Pass, Src};

const G: Family = Family::Grade;

fn lift_gamma_gain(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let mut o = c * (1.0 + 0.3 * k[2]);
    o = o + (s3(1.0) - o) * (k[0] * mix(0.08, 0.15, step(0.0, k[0])));
    o.max(0.0).map(|v| v.powf((-0.6 * k[1]).exp2()))
}

/// Lift, gamma and gain: the blacks, the middle and the whites.
pub const LIFT_GAMMA_GAIN: Pass = Pass {
    name: "lift_gamma_gain",
    family: G,
    about: "the classic three-way grade: the blacks raised or crushed, the middle brighter or darker, the whites brighter or dimmer",
    knobs: &[
        knob("lift", "raises the blacks (+, a faded film) or crushes them (-, deep shadows)", (-1.0, 1.0, 0.0), "share", (0.3, 0.3)),
        knob("gamma", "the midtones brighter (+) or darker (-), the ends kept", (-1.0, 1.0, 0.0), "share", (0.3, 0.45)),
        knob("gain", "the whites brighter (+) or dimmer (-)", (-1.0, 1.0, 0.0), "share", (0.3, 0.45)),
    ],
    pairs: &["curve", "grain", "vignette"],
    wgsl: r"
fn look_lift_gamma_gain(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    var o = c * (1.0 + 0.3 * u.k0.z);
    o = o + (vec3<f32>(1.0) - o) * (u.k0.x * mix(0.08, 0.15, step(0.0, u.k0.x)));
    return pow(max(o, vec3<f32>(0.0)), vec3<f32>(exp2(-0.6 * u.k0.y)));
}
",
    cpu: lift_gamma_gain,
    ..Pass::BASE
};

fn white_balance(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let mut o = c * v3(1.0 + 0.16 * k[0], 1.0 + 0.02 * k[0], 1.0 - 0.16 * k[0]);
    o = o * v3(1.0 + 0.08 * k[1], 1.0 - 0.1 * k[1], 1.0 + 0.08 * k[1]);
    o * (luma(c) / luma(o).max(1e-4))
}

/// White balance: warmer or cooler, greener or pinker, the light kept.
pub const WHITE_BALANCE: Pass = Pass {
    name: "white_balance",
    family: G,
    about:
        "white balance and tint, the light kept: daylight to tungsten, a green or a magenta cast",
    knobs: &[
        knob(
            "warmth",
            "warmer, orange (+), or cooler, blue (-)",
            (-1.0, 1.0, 0.0),
            "share",
            (0.3, 0.02),
        ),
        knob(
            "tint",
            "magenta (+) or green (-)",
            (-1.0, 1.0, 0.0),
            "share",
            (0.3, 0.02),
        ),
    ],
    pairs: &["curve", "grain"],
    wgsl: r"
fn look_white_balance(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    var o = c * vec3<f32>(1.0 + 0.16 * u.k0.x, 1.0 + 0.02 * u.k0.x, 1.0 - 0.16 * u.k0.x);
    o = o * vec3<f32>(1.0 + 0.08 * u.k0.y, 1.0 - 0.1 * u.k0.y, 1.0 + 0.08 * u.k0.y);
    return o * (lk_luma(c) / max(lk_luma(o), 1e-4));
}
",
    cpu: white_balance,
    ..Pass::BASE
};

fn saturation(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let chroma = c.x.max(c.y).max(c.z) - c.x.min(c.y).min(c.z);
    let v = 1.0 + k[1] * (1.0 - chroma);
    saturate(c, k[0] * v).max(0.0)
}

/// Saturation and vibrance.
pub const SATURATION: Pass = Pass {
    name: "saturation",
    family: G,
    about: "how vivid the colours are: saturation for all of them, vibrance for the dull ones only (skin and sky kept)",
    knobs: &[
        knob("saturation", "0 grey, 1 as it is, 2 twice as vivid", (0.0, 2.0, 1.0), "share", (0.4, 0.05)),
        knob("vibrance", "the dull colours more vivid (+) or duller (-), the vivid ones kept", (-1.0, 1.0, 0.0), "share", (0.4, 0.05)),
    ],
    natural: &[natural("saturation", Driver::Chorus, 0.4)],
    pairs: &["curve", "duotone"],
    wgsl: r"
fn look_saturation(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let chroma = max(max(c.x, c.y), c.z) - min(min(c.x, c.y), c.z);
    let v = 1.0 + u.k0.y * (1.0 - chroma);
    return max(lk_saturate(c, u.k0.x * v), vec3<f32>(0.0));
}
",
    cpu: saturation,
    ..Pass::BASE
};

fn duotone(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let l = sat((luma(c) - 0.5) * (1.0 + 0.8 * k[4]) + 0.5);
    let dark = mix3(hsv(k[0], 0.8, 0.22), u.pal(2) * 0.6, k[3]);
    let light = mix3(hsv(k[1], 0.45, 1.0), u.pal(0), k[3]);
    mix3(c, mix3(dark, light, l), k[2])
}

/// Two colours: the shadows one, the lights the other.
pub const DUOTONE: Pass = Pass {
    name: "duotone",
    family: G,
    about: "the picture in two colours, dark to light (the cover's own, or two hues): posters, pop, moods",
    knobs: &[
        knob("dark", "the shadows' hue", (0.0, 1.0, 0.7), "turns", (0.1, 0.07)),
        knob("light", "the lights' hue", (0.0, 1.0, 0.1), "turns", (0.1, 0.08)),
        knob("amount", "how much of the two colours", (0.0, 1.0, 1.0), "share", (0.3, 0.3)),
        knob("cover", "from the two hues (0) to the cover's own colours (1)", (0.0, 1.0, 0.0), "share", (0.0, 0.3)),
        knob("contrast", "flatter (-) or harder (+) from dark to light", (-1.0, 1.0, 0.0), "share", (0.3, 0.1)),
    ],
    natural: &[natural("light", Driver::Part, 0.6)],
    pairs: &["grain", "halftone", "curve"],
    wgsl: r"
fn look_duotone(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let l = lk_sat((lk_luma(c) - 0.5) * (1.0 + 0.8 * u.k1.x) + 0.5);
    let dark = mix(lk_hsv(u.k0.x, 0.8, 0.22), u.pal2.xyz * 0.6, u.k0.w);
    let light = mix(lk_hsv(u.k0.y, 0.45, 1.0), u.pal0.xyz, u.k0.w);
    return mix(c, mix(dark, light, l), u.k0.z);
}
",
    cpu: duotone,
    ..Pass::BASE
};

fn tritone(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let l = luma(c);
    let a = hsv(k[0], 0.75, 0.18);
    let b = hsv(k[1], 0.7, 0.62);
    let w = hsv(k[2], 0.25, 1.0);
    let t = mix3(
        mix3(a, b, smoothstep(0.0, 0.5, l)),
        w,
        smoothstep(0.5, 1.0, l),
    );
    mix3(c, t, k[3])
}

/// Three colours: shadows, middle, lights.
pub const TRITONE: Pass = Pass {
    name: "tritone",
    family: G,
    about: "the picture in three colours, shadows, midtones and lights: screen prints, neon nights, sunsets",
    knobs: &[
        knob("dark", "the shadows' hue", (0.0, 1.0, 0.75), "turns", (0.1, 0.08)),
        knob("middle", "the midtones' hue", (0.0, 1.0, 0.95), "turns", (0.1, 0.14)),
        knob("light", "the lights' hue", (0.0, 1.0, 0.12), "turns", (0.1, 0.08)),
        knob("amount", "how much of the three colours", (0.0, 1.0, 1.0), "share", (0.3, 0.3)),
    ],
    natural: &[natural("middle", Driver::Part, 0.5)],
    pairs: &["grain", "halftone", "glow"],
    wgsl: r"
fn look_tritone(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let l = lk_luma(c);
    let a = lk_hsv(u.k0.x, 0.75, 0.18);
    let b = lk_hsv(u.k0.y, 0.7, 0.62);
    let w = lk_hsv(u.k0.z, 0.25, 1.0);
    let t = mix(mix(a, b, smoothstep(0.0, 0.5, l)), w, smoothstep(0.5, 1.0, l));
    return mix(c, t, u.k0.w);
}
",
    cpu: tritone,
    ..Pass::BASE
};

fn split_tone(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let l = luma(c);
    let b = 0.5 + 0.3 * k[3];
    let sh = (hsv(k[0], 1.0, 1.0) - 0.5) * (1.0 - smoothstep(0.0, b + 0.25, l));
    let hi = (hsv(k[1], 1.0, 1.0) - 0.5) * smoothstep(b - 0.25, 1.0, l);
    (c + (sh + hi) * (k[2] * 0.2)).max(0.0)
}

/// Shadows toward one hue, lights toward another.
pub const SPLIT_TONE: Pass = Pass {
    name: "split_tone",
    family: G,
    about: "the shadows tinted one way and the lights another (teal and orange, the film look)",
    knobs: &[
        knob(
            "shadows",
            "the shadows' hue",
            (0.0, 1.0, 0.52),
            "turns",
            (0.1, 0.03),
        ),
        knob(
            "lights",
            "the lights' hue",
            (0.0, 1.0, 0.08),
            "turns",
            (0.1, 0.03),
        ),
        knob(
            "amount",
            "how strongly tinted",
            (0.0, 1.0, 0.5),
            "share",
            (0.3, 0.05),
        ),
        knob(
            "balance",
            "where shadows end and lights begin: lower (-) or higher (+)",
            (-1.0, 1.0, 0.0),
            "share",
            (0.3, 0.02),
        ),
    ],
    pairs: &["curve", "grain", "halation"],
    wgsl: r"
fn look_split_tone(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let l = lk_luma(c);
    let b = 0.5 + 0.3 * u.k0.w;
    let sh = (lk_hsv(u.k0.x, 1.0, 1.0) - 0.5) * (1.0 - smoothstep(0.0, b + 0.25, l));
    let hi = (lk_hsv(u.k0.y, 1.0, 1.0) - 0.5) * smoothstep(b - 0.25, 1.0, l);
    return max(c + (sh + hi) * (u.k0.z * 0.2), vec3<f32>(0.0));
}
",
    cpu: split_tone,
    ..Pass::BASE
};

/// A channel's contrast about `p` by the power `e`, smoothly at both ends.
fn contrast(x: f32, p: f32, e: f32) -> f32 {
    let x = sat(x);
    if x > p {
        1.0 - (1.0 - p) * ((1.0 - x) / (1.0 - p)).powf(e)
    } else {
        p * (x / p).powf(e)
    }
}

fn curve(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let e = (1.6 * k[0]).exp2();
    let o = c.map(|x| contrast(x, k[1], e));
    mix3(o, o * 0.84 + 0.08, k[2])
}

/// An S curve: contrast about a pivot, and a matte fade.
pub const CURVE: Pass = Pass {
    name: "curve",
    family: G,
    about: "a tone curve: more or less contrast about a pivot, and the matte fade of printed film",
    knobs: &[
        knob(
            "contrast",
            "flatter (-) or punchier (+)",
            (-1.0, 1.0, 0.0),
            "share",
            (0.3, 0.2),
        ),
        knob(
            "pivot",
            "the tone the contrast turns about",
            (0.2, 0.8, 0.5),
            "share",
            (0.1, 0.3),
        ),
        knob(
            "fade",
            "the blacks lifted and the whites lowered, matte",
            (0.0, 1.0, 0.0),
            "share",
            (0.3, 0.08),
        ),
    ],
    pairs: &["split_tone", "grain"],
    wgsl: r"
fn lk_curve_channel(x0: f32, p: f32, e: f32) -> f32 {
    let x = lk_sat(x0);
    return select(p * pow(x / p, e), 1.0 - (1.0 - p) * pow((1.0 - x) / (1.0 - p), e), x > p);
}
fn look_curve(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let e = exp2(1.6 * u.k0.x);
    let o = vec3<f32>(lk_curve_channel(c.x, u.k0.y, e), lk_curve_channel(c.y, u.k0.y, e), lk_curve_channel(c.z, u.k0.y, e));
    return mix(o, o * 0.84 + 0.08, u.k0.z);
}
",
    cpu: curve,
    ..Pass::BASE
};

fn gradient_map(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let l = luma(c);
    let (a, b, w) = (u.pal(2) * 0.35, u.pal(1), u.pal(0));
    let m = mix3(
        mix3(a, b, smoothstep(0.0, 0.5, l)),
        mix3(w, s3(1.0), 0.3),
        smoothstep(0.5, 1.0, l),
    );
    mix3(c, m, k[0])
}

/// The tones mapped onto the cover's own colours.
pub const GRADIENT_MAP: Pass = Pass {
    name: "gradient_map",
    family: G,
    about: "the tones mapped onto the cover's three colours, dark to light: a picture made the cover's palette",
    knobs: &[knob("amount", "how much of the cover's palette", (0.0, 1.0, 0.7), "share", (0.3, 0.3))],
    pairs: &["grain", "halftone"],
    wgsl: r"
fn look_gradient_map(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let l = lk_luma(c);
    let a = u.pal2.xyz * 0.35;
    let b = u.pal1.xyz;
    let w = u.pal0.xyz;
    let m = mix(mix(a, b, smoothstep(0.0, 0.5, l)), mix(w, vec3<f32>(1.0), 0.3), smoothstep(0.5, 1.0, l));
    return mix(c, m, u.k0.x);
}
",
    cpu: gradient_map,
    ..Pass::BASE
};

fn technicolor(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let sep = v3(
        c.x - (c.y + c.z) * 0.5,
        c.y - (c.x + c.z) * 0.5,
        c.z - (c.x + c.y) * 0.5,
    );
    let t = c + sep * (0.5 + 0.7 * k[1]);
    let t = t * (luma(c) / luma(t).max(1e-3)).min(1.5);
    mix3(c, t.max(0.0), k[0])
}

/// The three-strip look: dyes that separate the colours.
pub const TECHNICOLOR: Pass = Pass {
    name: "technicolor",
    family: G,
    about: "the three-strip 1950s colour: each primary separated and rich, reds deep, skies cyan",
    knobs: &[
        knob(
            "amount",
            "how much of the look",
            (0.0, 1.0, 0.8),
            "share",
            (0.3, 0.05),
        ),
        knob(
            "punch",
            "how far the dyes separate",
            (0.0, 1.0, 0.5),
            "share",
            (0.3, 0.05),
        ),
    ],
    pairs: &["grain", "halation", "curve"],
    wgsl: r"
fn look_technicolor(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let sep = vec3<f32>(c.x - (c.y + c.z) * 0.5, c.y - (c.x + c.z) * 0.5, c.z - (c.x + c.y) * 0.5);
    var t = c + sep * (0.5 + 0.7 * u.k0.y);
    t = t * min(lk_luma(c) / max(lk_luma(t), 1e-3), 1.5);
    return mix(c, max(t, vec3<f32>(0.0)), u.k0.x);
}
",
    cpu: technicolor,
    ..Pass::BASE
};

fn bleach_bypass(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let l = luma(c);
    let over = |x: f32| {
        if l > 0.5 {
            1.0 - 2.0 * (1.0 - x) * (1.0 - l)
        } else {
            2.0 * x * l
        }
    };
    let o = mix3(c, c.map(over), 0.5 + 0.5 * k[1]);
    mix3(c, saturate(o, 0.55), k[0])
}

/// The silver kept in the print: harsh, desaturated, metallic.
pub const BLEACH_BYPASS: Pass = Pass {
    name: "bleach_bypass",
    family: G,
    about: "the silver left in the print: harsh contrast, washed-out colour, metal and war films",
    knobs: &[
        knob(
            "amount",
            "how much of the look",
            (0.0, 1.0, 0.7),
            "share",
            (0.3, 0.1),
        ),
        knob(
            "contrast",
            "how harsh",
            (0.0, 1.0, 0.5),
            "share",
            (0.3, 0.1),
        ),
    ],
    pairs: &["grain", "vignette"],
    wgsl: r"
fn look_bleach_bypass(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let l = lk_luma(c);
    let hi = vec3<f32>(1.0) - 2.0 * (vec3<f32>(1.0) - c) * (1.0 - l);
    let lo = 2.0 * c * l;
    let over = select(lo, hi, l > 0.5);
    let o = mix(c, over, 0.5 + 0.5 * u.k0.y);
    return mix(c, lk_saturate(o, 0.55), u.k0.x);
}
",
    cpu: bleach_bypass,
    ..Pass::BASE
};

fn cross_process(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let x = v3(
        smoothstep(0.05, 0.95, c.x),
        c.y * 0.88 + 0.07,
        c.z * 0.55 + 0.22,
    );
    mix3(c, x, k[0])
}

/// Slide film in the negative's chemistry: yellow lights, cyan shadows.
pub const CROSS_PROCESS: Pass = Pass {
    name: "cross_process",
    family: G,
    about: "slide film in the wrong chemistry: yellow-green lights, blue shadows, hard reds (90s indie, lo-fi)",
    knobs: &[knob("amount", "how much of the look", (0.0, 1.0, 0.7), "share", (0.3, 0.1))],
    pairs: &["grain", "vignette", "light_leak"],
    wgsl: r"
fn look_cross_process(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let x = vec3<f32>(smoothstep(0.05, 0.95, c.x), c.y * 0.88 + 0.07, c.z * 0.55 + 0.22);
    return mix(c, x, u.k0.x);
}
",
    cpu: cross_process,
    ..Pass::BASE
};

fn infrared(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let ir = v3(
        c.y * 0.9 + c.x * 0.25,
        c.x * 0.65 + c.z * 0.2,
        c.z * 0.85 + c.y * 0.1,
    );
    let mono = s3((c.y * 1.4 + c.x * 0.3 - c.z * 0.3).clamp(0.0, 1.0));
    mix3(c, mix3(ir, mono, k[1]), k[0])
}

/// Infrared film: foliage pink or white.
pub const INFRARED: Pass = Pass {
    name: "infrared",
    family: G,
    about: "infrared film: greens turn pink and red, skies deep (or in black and white, foliage glowing white)",
    knobs: &[
        knob("amount", "how much of the look", (0.0, 1.0, 0.8), "share", (0.3, 0.1)),
        knob("mono", "colour infrared (0) or black-and-white infrared (1)", (0.0, 1.0, 0.0), "share", (0.0, 0.1)),
    ],
    pairs: &["glow", "grain"],
    wgsl: r"
fn look_infrared(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let ir = vec3<f32>(c.y * 0.9 + c.x * 0.25, c.x * 0.65 + c.z * 0.2, c.z * 0.85 + c.y * 0.1);
    let mono = vec3<f32>(clamp(c.y * 1.4 + c.x * 0.3 - c.z * 0.3, 0.0, 1.0));
    return mix(c, mix(ir, mono, u.k0.y), u.k0.x);
}
",
    cpu: infrared,
    ..Pass::BASE
};

fn sepia(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let l = smoothstep(0.02, 0.98, luma(c));
    let dark = mix3(v3(0.16, 0.09, 0.04), v3(0.08, 0.09, 0.1), sat(-k[1]));
    let light = mix3(v3(0.98, 0.9, 0.74), v3(0.9, 0.92, 0.94), sat(-k[1]));
    let t = mix3(dark, light, l);
    let t = mix3(t, t * 0.8 + 0.12, k[2]);
    mix3(c, t, k[0])
}

/// An old photograph: warm brown, or a daguerreotype's cool silver.
pub const SEPIA: Pass = Pass {
    name: "sepia",
    family: G,
    about: "an old photograph: warm sepia brown or a daguerreotype's cool silver, faded with age",
    knobs: &[
        knob(
            "amount",
            "how much of the look",
            (0.0, 1.0, 0.9),
            "share",
            (0.3, 0.1),
        ),
        knob(
            "tone",
            "a daguerreotype's silver (-1) to warm sepia (+1)",
            (-1.0, 1.0, 1.0),
            "share",
            (0.2, 0.02),
        ),
        knob(
            "fade",
            "how faded with age",
            (0.0, 1.0, 0.3),
            "share",
            (0.3, 0.08),
        ),
    ],
    natural: &[natural("fade", Driver::Progress, 0.5)],
    pairs: &["film_damage", "vignette", "gate_weave"],
    wgsl: r"
fn look_sepia(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let l = smoothstep(0.02, 0.98, lk_luma(c));
    let dark = mix(vec3<f32>(0.16, 0.09, 0.04), vec3<f32>(0.08, 0.09, 0.1), lk_sat(-u.k0.y));
    let light = mix(vec3<f32>(0.98, 0.9, 0.74), vec3<f32>(0.9, 0.92, 0.94), lk_sat(-u.k0.y));
    var t = mix(dark, light, l);
    t = mix(t, t * 0.8 + 0.12, u.k0.z);
    return mix(c, t, u.k0.x);
}
",
    cpu: sepia,
    ..Pass::BASE
};
