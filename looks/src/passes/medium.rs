//! The picture turned into another medium: glass and lead, tiles, thread
//! on cloth, a blueprint, a heat camera, night vision, an X-ray, chrome,
//! neon tubes, teletext's blocks (shapes only: never readable text).

use crate::drive::Driver;
use crate::math::*;
use crate::pass::{knob, natural, Ctx, Family, Pass, Src};
use crate::passes::print::edges;

const M: Family = Family::Medium;

fn size2(u: &Ctx) -> V2 {
    v2(u.size[0], u.size[1])
}

fn stained_glass(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let size = k[0] * u.scale();
    let p = uv * size2(u) / size;
    let base = p.floor();
    let mut d1 = 9.0f32;
    let mut d2 = 9.0f32;
    let mut best = v2(0.0, 0.0);
    for j in 0..9i32 {
        let o = v2((j % 3 - 1) as f32, (j / 3 - 1) as f32);
        let cv = base + o;
        let (x, y) = (cell(cv.x), cell(cv.y));
        let at = cv + v2(0.15 + 0.7 * hash2(x, y, 131), 0.15 + 0.7 * hash2(x, y, 132));
        let d = (p - at).length();
        if d < d1 {
            d2 = d1;
            d1 = d;
            best = at;
        } else if d < d2 {
            d2 = d;
        }
    }
    let glass = saturate(s.src(best * size / size2(u)), 1.4);
    let lit = glass * (0.75 + 0.5 * k[2] * (1.0 - (d1 * 1.6).min(1.0)));
    let lead_w = k[1] * u.scale() / size;
    let aa = 1.0 / size;
    let lead = 1.0 - smoothstep(lead_w - aa, lead_w + aa, (d2 - d1) * 0.5);
    mix3(lit, v3(0.05, 0.05, 0.06), lead)
}

/// Leaded glass.
pub const STAINED_GLASS: Pass = Pass {
    name: "stained_glass",
    family: M,
    about: "leaded glass: the picture in pieces of coloured glass between dark lead lines, lit from behind (ballads, folk, fantasy)",
    knobs: &[
        knob("cell", "the pieces' size", (12.0, 80.0, 36.0), "px", (12.0, 0.02)),
        knob("lead", "the lead lines' width", (0.5, 6.0, 2.5), "px", (1.0, 0.08)),
        knob("glow", "how much the light shines through each piece's middle", (0.0, 1.0, 0.5), "share", (0.5, 0.1)),
    ],
    cost: 2.0,
    relief: 0.5,
    cells: true,
    natural: &[natural("glow", Driver::Vocals, 0.8), natural("cell", Driver::Bass, 0.5)],
    pairs: &["glow", "vignette"],
    wgsl: r"
fn look_stained_glass(uv: vec2<f32>) -> vec3<f32> {
    let size = u.k0.x * lk_scale();
    let p = uv * u.size.xy / size;
    let base = floor(p);
    var d1 = 9.0;
    var d2 = 9.0;
    var best = vec2<f32>(0.0);
    for (var j = 0; j < 9; j = j + 1) {
        let o = vec2<f32>(f32(j % 3 - 1), f32(j / 3 - 1));
        let cv = base + o;
        let x = i32(cv.x);
        let y = i32(cv.y);
        let at = cv + vec2<f32>(0.15 + 0.7 * lk_hash2(x, y, 131u), 0.15 + 0.7 * lk_hash2(x, y, 132u));
        let d = length(p - at);
        if (d < d1) {
            d2 = d1;
            d1 = d;
            best = at;
        } else if (d < d2) {
            d2 = d;
        }
    }
    let glass = lk_saturate(lk_src(best * size / u.size.xy), 1.4);
    let lit = glass * (0.75 + 0.5 * u.k0.z * (1.0 - min(d1 * 1.6, 1.0)));
    let lead_w = u.k0.y * lk_scale() / size;
    let aa = 1.0 / size;
    let lead = 1.0 - smoothstep(lead_w - aa, lead_w + aa, (d2 - d1) * 0.5);
    return mix(lit, vec3<f32>(0.05, 0.05, 0.06), lead);
}
",
    cpu: stained_glass,
    ..Pass::BASE
};

fn mosaic(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let size = k[0] * u.scale();
    let p = uv * size2(u) / size;
    let cv = p.floor();
    let (x, y) = (cell(cv.x), cell(cv.y));
    let f = p - cv;
    let col = s.src((cv + 0.5) * size / size2(u));
    let tile = col * (1.0 + k[2] * 0.2 * (hash2(x, y, 141) - 0.5));
    let g = k[1] * 0.12 + 0.5 / size;
    let edge = f.x.min(f.y).min((1.0 - f.x).min(1.0 - f.y));
    let aa = 1.0 / size;
    let grout = 1.0 - smoothstep(g - aa, g + aa, edge);
    let bevel = 1.0 + 0.12 * (0.5 - f.y) * (1.0 - grout);
    mix3(
        tile * bevel,
        v3(0.32, 0.3, 0.28),
        grout * k[1].clamp(0.3, 1.0),
    )
}

/// Mosaic tiles.
pub const MOSAIC: Pass = Pass {
    name: "mosaic",
    family: M,
    about: "a mosaic: the picture in square tiles of one colour each, grout between them (Mediterranean, ancient, summer)",
    knobs: &[
        knob("tile", "the tiles' size", (6.0, 40.0, 14.0), "px", (6.0, 0.02)),
        knob("grout", "how wide and how visible the grout", (0.0, 1.0, 0.4), "share", (0.2, 0.1)),
        knob("variety", "how much each tile's shade varies", (0.0, 1.0, 0.3), "share", (0.2, 0.01)),
    ],
    cost: 1.0,
    relief: 0.5,
    cells: true,
    natural: &[natural("tile", Driver::Kick, 0.7)],
    pairs: &["saturation", "vignette"],
    wgsl: r"
fn look_mosaic(uv: vec2<f32>) -> vec3<f32> {
    let size = u.k0.x * lk_scale();
    let p = uv * u.size.xy / size;
    let cv = floor(p);
    let x = i32(cv.x);
    let y = i32(cv.y);
    let f = p - cv;
    let col = lk_src((cv + 0.5) * size / u.size.xy);
    let tile = col * (1.0 + u.k0.z * 0.2 * (lk_hash2(x, y, 141u) - 0.5));
    let g = u.k0.y * 0.12 + 0.5 / size;
    let edge = min(min(f.x, f.y), min(1.0 - f.x, 1.0 - f.y));
    let aa = 1.0 / size;
    let grout = 1.0 - smoothstep(g - aa, g + aa, edge);
    let bevel = 1.0 + 0.12 * (0.5 - f.y) * (1.0 - grout);
    return mix(tile * bevel, vec3<f32>(0.32, 0.3, 0.28), grout * max(min(u.k0.y, 1.0), 0.3));
}
",
    cpu: mosaic,
    ..Pass::BASE
};

fn cross_stitch(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let size = k[0] * u.scale();
    let p = uv * size2(u) / size;
    let cv = p.floor();
    let f = p - cv;
    let col = s.src((cv + 0.5) * size / size2(u));
    let n = k[1] - 1.0;
    let thread = col.clamp(0.0, 1.0).map(|x| (x * n + 0.5).floor() / n);
    let d = (f.x - f.y).abs().min((f.x + f.y - 1.0).abs()) * std::f32::consts::FRAC_1_SQRT_2;
    let aa = 0.7 / size;
    let w = 0.16;
    let on = 1.0 - smoothstep(w - aa, w + aa, d);
    let shade = 0.82 + 0.3 * (1.0 - d / w).max(0.0);
    let holes = (fract(p.x * 2.0) - 0.5)
        .abs()
        .min((fract(p.y * 2.0) - 0.5).abs());
    let fabric = v3(0.93, 0.91, 0.85) * (0.9 + 0.1 * smoothstep(0.0, 0.2, holes));
    let stitched = 1.0 - smoothstep(0.86, 0.92, luma(col)) * k[2];
    mix3(fabric, thread * shade, on * stitched)
}

/// Cross-stitch on cloth.
pub const CROSS_STITCH: Pass = Pass {
    name: "cross_stitch",
    family: M,
    about: "cross-stitch: the picture in little X stitches of a few thread colours on cloth (folk, family, cosy, Volksmusik)",
    knobs: &[
        knob("stitch", "the stitches' size", (4.0, 20.0, 8.0), "px", (3.0, 0.09)),
        knob("levels", "how many shades of red, of green and of blue the threads come in", (2.0, 5.0, 3.0), "steps", (0.0, 0.11)),
        knob("fabric", "how much bare cloth is left in the lights", (0.0, 1.0, 0.6), "share", (0.2, 0.1)),
    ],
    cost: 1.0,
    relief: 0.5,
    cells: true,
    natural: &[natural("stitch", Driver::Kick, 0.6)],
    pairs: &["canvas", "vignette"],
    wgsl: r"
fn look_cross_stitch(uv: vec2<f32>) -> vec3<f32> {
    let size = u.k0.x * lk_scale();
    let p = uv * u.size.xy / size;
    let cv = floor(p);
    let f = p - cv;
    let col = lk_src((cv + 0.5) * size / u.size.xy);
    let n = u.k0.y - 1.0;
    let thread = floor(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)) * n + 0.5) / n;
    let d = min(abs(f.x - f.y), abs(f.x + f.y - 1.0)) * 0.70710677;
    let aa = 0.7 / size;
    let w = 0.16;
    let on = 1.0 - smoothstep(w - aa, w + aa, d);
    let shade = 0.82 + 0.3 * max(1.0 - d / w, 0.0);
    let holes = min(abs(fract(p.x * 2.0) - 0.5), abs(fract(p.y * 2.0) - 0.5));
    let fabric = vec3<f32>(0.93, 0.91, 0.85) * (0.9 + 0.1 * smoothstep(0.0, 0.2, holes));
    let stitched = 1.0 - smoothstep(0.86, 0.92, lk_luma(col)) * u.k0.z;
    return mix(fabric, thread * shade, on * stitched);
}
",
    cpu: cross_stitch,
    ..Pass::BASE
};

fn cyanotype(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let l = luma(c);
    let p = uv * size2(u) / u.scale();
    let print = mix3(
        v3(0.04, 0.17, 0.4),
        v3(0.86, 0.91, 0.96),
        l.max(0.0).powf(1.2),
    ) * (0.95 + 0.05 * fbm(p / 7.0, 151));
    let e = smoothstep(0.05, 0.18, edges(u, s, uv, 1.2 * u.scale()));
    let gx = (fract(p.x / 40.0) - 0.5).abs();
    let gy = (fract(p.y / 40.0) - 0.5).abs();
    let grid = (1.0 - smoothstep(0.47, 0.49, gx.max(gy))) * k[2];
    let plan = v3(0.08, 0.25, 0.55) * (0.9 + 0.1 * l) + s3(0.85) * (e * 0.9 + grid * 0.25);
    mix3(c, mix3(print, plan, k[0]), k[1])
}

/// A cyanotype print or a blueprint.
pub const CYANOTYPE: Pass = Pass {
    name: "cyanotype",
    family: M,
    about: "a cyanotype's Prussian blue print, or a blueprint's white lines and grid on blue (science, the sea, dreams, plans)",
    knobs: &[
        knob("plan", "a cyanotype print (0) to a blueprint's lines (1)", (0.0, 1.0, 0.0), "share", (0.0, 0.3)),
        knob("amount", "how much of the medium over the picture", (0.0, 1.0, 1.0), "share", (0.3, 0.2)),
        knob("grid", "how much of a blueprint's grid", (0.0, 1.0, 0.3), "share", (0.4, 0.03)),
    ],
    cost: 1.6,
    relief: 0.3,
    natural: &[natural("grid", Driver::Drums, 0.6)],
    pairs: &["canvas", "vignette"],
    wgsl: r"
fn look_cyanotype(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let l = lk_luma(c);
    let p = uv * u.size.xy / lk_scale();
    let print = mix(vec3<f32>(0.04, 0.17, 0.4), vec3<f32>(0.86, 0.91, 0.96), pow(max(l, 0.0), 1.2)) * (0.95 + 0.05 * lk_fbm(p / 7.0, 151u));
    let e = smoothstep(0.05, 0.18, lk_edges(uv, 1.2 * lk_scale()));
    let gx = abs(fract(p.x / 40.0) - 0.5);
    let gy = abs(fract(p.y / 40.0) - 0.5);
    let grid = (1.0 - smoothstep(0.47, 0.49, max(gx, gy))) * u.k0.z;
    let plan = vec3<f32>(0.08, 0.25, 0.55) * (0.9 + 0.1 * l) + vec3<f32>(0.85) * (e * 0.9 + grid * 0.25);
    return mix(c, mix(print, plan, u.k0.x), u.k0.y);
}
",
    cpu: cyanotype,
    ..Pass::BASE
};

/// A heat camera's iron palette at heat `h` (0 to 1).
fn iron(h: f32) -> V3 {
    let a = mix3(
        v3(0.0, 0.0, 0.05),
        v3(0.35, 0.0, 0.55),
        smoothstep(0.0, 0.25, h),
    );
    let b = mix3(a, v3(0.85, 0.1, 0.25), smoothstep(0.2, 0.5, h));
    let c = mix3(b, v3(1.0, 0.6, 0.0), smoothstep(0.45, 0.75, h));
    mix3(c, v3(1.0, 1.0, 0.85), smoothstep(0.75, 1.0, h))
}

fn thermal(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let p = uv * size2(u) / u.scale();
    let f = (u.time[0] * 15.0).max(0.0).floor() as u32;
    let h = sat(luma(c) * 1.1 - 0.05 + (vnoise(p / 2.0, f) - 0.5) * 0.1 * k[2]);
    let o = mix3(iron(h), hsv(0.7 - 0.75 * h, 0.9, 0.35 + 0.65 * h), k[1]);
    mix3(c, o, k[0])
}

/// A heat camera.
pub const THERMAL: Pass = Pass {
    name: "thermal",
    family: M,
    about: "a heat camera: the brightness read as heat, in an iron palette or a rainbow (techno, sci-fi, war, horror, sweat and dance)",
    knobs: &[
        knob("amount", "how much of the camera over the picture", (0.0, 1.0, 1.0), "share", (0.3, 0.2)),
        knob("palette", "iron, black to white-hot (0), to a rainbow (1)", (0.0, 1.0, 0.0), "share", (0.0, 0.1)),
        knob("noise", "the sensor's noise", (0.0, 1.0, 0.3), "share", (0.5, 0.01)),
    ],
    natural: &[natural("noise", Driver::High, 0.8)],
    pairs: &["crt", "glitch"],
    wgsl: r"
fn lk_iron(h: f32) -> vec3<f32> {
    let a = mix(vec3<f32>(0.0, 0.0, 0.05), vec3<f32>(0.35, 0.0, 0.55), smoothstep(0.0, 0.25, h));
    let b = mix(a, vec3<f32>(0.85, 0.1, 0.25), smoothstep(0.2, 0.5, h));
    let c = mix(b, vec3<f32>(1.0, 0.6, 0.0), smoothstep(0.45, 0.75, h));
    return mix(c, vec3<f32>(1.0, 1.0, 0.85), smoothstep(0.75, 1.0, h));
}
fn look_thermal(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let p = uv * u.size.xy / lk_scale();
    let f = u32(floor(max(u.time.x * 15.0, 0.0)));
    let h = lk_sat(lk_luma(c) * 1.1 - 0.05 + (lk_vnoise(p / 2.0, f) - 0.5) * 0.1 * u.k0.z);
    let o = mix(lk_iron(h), lk_hsv(0.7 - 0.75 * h, 0.9, 0.35 + 0.65 * h), u.k0.y);
    return mix(c, o, u.k0.x);
}
",
    cpu: thermal,
    ..Pass::BASE
};

fn night_vision(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let p = uv * size2(u) / u.scale();
    let f = (u.time[0] * 24.0).max(0.0).floor() as u32;
    let l = sat(luma(c) * 1.6).max(0.0).powf(0.8);
    let n = (vnoise(p / 1.5, f + 7) - 0.5) * 0.35 * k[1];
    let scan = 1.0 - k[2] * 0.25 * (0.5 + 0.5 * (p.y * 2.2).sin());
    let q = (uv - 0.5) * v2(u.size[0] / u.size[1], 1.0);
    let tube = 1.0 - smoothstep(0.42, 0.62, q.length());
    let o = v3(0.25, 1.0, 0.35) * sat(l + n) * scan * (0.25 + 0.75 * tube);
    mix3(c, o, k[0])
}

/// Night vision goggles.
pub const NIGHT_VISION: Pass = Pass {
    name: "night_vision",
    family: M,
    about: "night vision: a green phosphor picture, noisy and lined, in a round tube (night, war, spies, horror)",
    knobs: &[
        knob("amount", "how much of the goggles over the picture", (0.0, 1.0, 1.0), "share", (0.3, 0.2)),
        knob("noise", "the tube's noise", (0.0, 1.0, 0.4), "share", (0.5, 0.02)),
        knob("lines", "how strong the scan lines", (0.0, 1.0, 0.3), "share", (0.2, 0.05)),
    ],
    natural: &[natural("noise", Driver::Drums, 0.8)],
    pairs: &["vignette", "glitch"],
    wgsl: r"
fn look_night_vision(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let p = uv * u.size.xy / lk_scale();
    let f = u32(floor(max(u.time.x * 24.0, 0.0)));
    let l = pow(max(lk_sat(lk_luma(c) * 1.6), 0.0), 0.8);
    let n = (lk_vnoise(p / 1.5, f + 7u) - 0.5) * 0.35 * u.k0.y;
    let scan = 1.0 - u.k0.z * 0.25 * (0.5 + 0.5 * sin(p.y * 2.2));
    let q = lk_aspect(uv);
    let tube = 1.0 - smoothstep(0.42, 0.62, length(q));
    let o = vec3<f32>(0.25, 1.0, 0.35) * lk_sat(l + n) * scan * (0.25 + 0.75 * tube);
    return mix(c, o, u.k0.x);
}
",
    cpu: night_vision,
    ..Pass::BASE
};

fn negative(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let inv = s3(1.0) - c;
    let l = 1.0 - luma(c);
    let xray = mix3(
        v3(0.0, 0.02, 0.06),
        v3(0.78, 0.9, 1.0),
        l.max(0.0).powf(1.3 + k[2]),
    );
    mix3(c, mix3(inv, xray, k[1]), k[0])
}

/// A negative or an X-ray.
pub const NEGATIVE: Pass = Pass {
    name: "negative",
    family: M,
    about: "a film negative, or an X-ray's cold glow (horror, Halloween, the surreal, sci-fi)",
    knobs: &[
        knob(
            "amount",
            "how much of the negative over the picture",
            (0.0, 1.0, 1.0),
            "share",
            (0.2, 0.6),
        ),
        knob(
            "xray",
            "a colour negative (0) to an X-ray (1)",
            (0.0, 1.0, 0.5),
            "share",
            (0.4, 0.3),
        ),
        knob(
            "contrast",
            "how dark the X-ray's dark parts",
            (0.0, 1.0, 0.4),
            "share",
            (0.2, 0.1),
        ),
    ],
    natural: &[natural("xray", Driver::Part, 0.6)],
    pairs: &["glow", "grain"],
    wgsl: r"
fn look_negative(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let inv = vec3<f32>(1.0) - c;
    let l = 1.0 - lk_luma(c);
    let xray = mix(vec3<f32>(0.0, 0.02, 0.06), vec3<f32>(0.78, 0.9, 1.0), pow(max(l, 0.0), 1.3 + u.k0.z));
    return mix(c, mix(inv, xray, u.k0.y), u.k0.x);
}
",
    cpu: negative,
    ..Pass::BASE
};

fn solarize(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let th = k[0];
    let sol = c.map(|x| {
        if x > th {
            (1.0 - x) * th / (1.0 - th)
        } else {
            x
        }
    });
    let l = luma(c);
    let chrome =
        s3(0.5 + 0.5 * (l * TAU * 2.5 + 0.5).cos()) * v3(0.92, 0.96, 1.0) + v3(0.02, 0.03, 0.06);
    mix3(c, mix3(sol, chrome, k[2]), k[1])
}

/// Solarisation, or chrome.
pub const SOLARIZE: Pass = Pass {
    name: "solarize",
    family: M,
    about: "solarisation, the darkroom's turned tones, or liquid chrome's bands (psychedelia, Y2K chrome, glam)",
    knobs: &[
        knob("threshold", "where the tones turn back", (0.3, 0.9, 0.55), "share", (0.15, 0.2)),
        knob("amount", "how much of the effect over the picture", (0.0, 1.0, 1.0), "share", (0.3, 0.2)),
        knob("chrome", "solarised (0) to chrome bands (1)", (0.0, 1.0, 0.0), "share", (0.0, 0.2)),
    ],
    natural: &[natural("threshold", Driver::Bass, 0.6)],
    pairs: &["hue", "glow"],
    wgsl: r"
fn look_solarize(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let th = u.k0.x;
    let sol = select(c, (vec3<f32>(1.0) - c) * th / (1.0 - th), c > vec3<f32>(th));
    let l = lk_luma(c);
    let chrome = vec3<f32>(0.5 + 0.5 * cos(l * LK_TAU * 2.5 + 0.5)) * vec3<f32>(0.92, 0.96, 1.0) + vec3<f32>(0.02, 0.03, 0.06);
    return mix(c, mix(sol, chrome, u.k0.z), u.k0.y);
}
",
    cpu: solarize,
    ..Pass::BASE
};

fn neon_edges(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let w = k[0] * u.scale();
    let e = smoothstep(0.04, 0.2, edges(u, s, uv, w));
    let wide = smoothstep(0.04, 0.25, edges(u, s, uv, w * 3.0));
    let hue = k[1] + k[4] * u.time[0] * 0.1 + k[4] * 0.3 * uv.x;
    let neon = hsv(hue, 0.75, 1.0);
    c * (1.0 - k[3] * 0.7) + neon * (e * 1.2 + wide * 0.5 * k[2])
}

/// Neon tubes along the edges.
pub const NEON_EDGES: Pass = Pass {
    name: "neon_edges",
    family: M,
    about: "the picture's edges as glowing neon tubes over a darkened picture (synthwave, the 80s, night and the city, club music)",
    knobs: &[
        knob("width", "the tubes' width", (0.5, 4.0, 1.5), "px", (1.0, 0.05)),
        knob("hue", "the neon's colour", (0.0, 1.0, 0.85), "turns", (0.3, 0.02)),
        knob("glow", "how far the glow spreads", (0.0, 1.0, 0.6), "share", (0.5, 0.06)),
        knob("dark", "how dark the picture under the neon", (0.0, 1.0, 0.6), "share", (0.2, 0.45)),
        knob("cycle", "still (0) to a rainbow running through the colours (1)", (0.0, 1.0, 0.0), "share", (0.3, 0.01)),
    ],
    cost: 2.0,
    relief: 0.3,
    natural: &[natural("glow", Driver::Kick, 0.8), natural("hue", Driver::Part, 0.6)],
    pairs: &["glow", "crt", "tritone"],
    wgsl: r"
fn look_neon_edges(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let w = u.k0.x * lk_scale();
    let e = smoothstep(0.04, 0.2, lk_edges(uv, w));
    let wide = smoothstep(0.04, 0.25, lk_edges(uv, w * 3.0));
    let hue = u.k0.y + u.k1.x * u.time.x * 0.1 + u.k1.x * 0.3 * uv.x;
    let neon = lk_hsv(hue, 0.75, 1.0);
    return c * (1.0 - u.k0.w * 0.7) + neon * (e * 1.2 + wide * 0.5 * u.k0.z);
}
",
    cpu: neon_edges,
    ..Pass::BASE
};

/// Teletext's eight colours: each of the picture's channels on or off.
fn teletext_colour(c: V3) -> V3 {
    c.map(|x| step(0.45, x))
}

fn teletext(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let cw = k[0] * u.scale();
    let csize = v2(cw, cw * 1.5);
    let p = uv * size2(u) / csize;
    let cv = p.floor();
    let f = p - cv;
    let sub = v2((f.x * 2.0).floor(), (f.y * 3.0).floor());
    let mut sum = 0.0;
    let mut mine = 0.0;
    let mut bright = s3(0.0);
    for j in 0..6u32 {
        let sx = (j % 2) as f32;
        let sy = (j / 2) as f32;
        let at = (cv + v2((sx + 0.5) / 2.0, (sy + 0.5) / 3.0)) * csize / size2(u);
        let tap = s.src(at);
        let l = luma(tap);
        sum += l;
        bright = bright.vmax(tap);
        if sx == sub.x && sy == sub.y {
            mine = l;
        }
    }
    let on = step((sum / 6.0).max(0.12), mine + 0.02);
    let fg = mix3(teletext_colour(saturate(bright, 1.5)), u.pal(0), k[1]).vmax(s3(0.15));
    let o = fg * on;
    mix3(c, o, k[2])
}

/// Teletext's block mosaics.
pub const TELETEXT: Pass = Pass {
    name: "teletext",
    family: M,
    about: "teletext's block graphics: each character cell of six blocks lit or dark in one of eight bright colours on black (no letters: shapes only) (the 80s, chiptune, retro TV)",
    knobs: &[
        knob("cell", "a cell's width", (6.0, 24.0, 12.0), "px", (5.0, 0.03)),
        knob("cover", "teletext's eight colours (0) to the cover's colour (1)", (0.0, 1.0, 0.0), "share", (0.0, 0.14)),
        knob("amount", "how much of the blocks over the picture", (0.0, 1.0, 1.0), "share", (0.3, 0.3)),
    ],
    cost: 2.0,
    relief: 0.5,
    cells: true,
    natural: &[natural("cell", Driver::Kick, 0.6)],
    pairs: &["crt", "pixelate"],
    wgsl: r"
fn look_teletext(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let cw = u.k0.x * lk_scale();
    let csize = vec2<f32>(cw, cw * 1.5);
    let p = uv * u.size.xy / csize;
    let cv = floor(p);
    let f = p - cv;
    let sub = vec2<f32>(floor(f.x * 2.0), floor(f.y * 3.0));
    var sum = 0.0;
    var mine = 0.0;
    var bright = vec3<f32>(0.0);
    for (var j = 0u; j < 6u; j = j + 1u) {
        let sx = f32(j % 2u);
        let sy = f32(j / 2u);
        let at = (cv + vec2<f32>((sx + 0.5) / 2.0, (sy + 0.5) / 3.0)) * csize / u.size.xy;
        let tap = lk_src(at);
        let l = lk_luma(tap);
        sum = sum + l;
        bright = max(bright, tap);
        if (sx == sub.x && sy == sub.y) {
            mine = l;
        }
    }
    let on = step(max(sum / 6.0, 0.12), mine + 0.02);
    let tc = step(vec3<f32>(0.45), lk_saturate(bright, 1.5));
    let fg = max(mix(tc, u.pal0.xyz, u.k0.y), vec3<f32>(0.15));
    let o = fg * on;
    return mix(c, o, u.k0.z);
}
",
    cpu: teletext,
    ..Pass::BASE
};
