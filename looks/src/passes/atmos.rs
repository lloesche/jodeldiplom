//! Weather and light in the air: haze and fog, rays and flares, rain,
//! snow, petals and dust (each seeded, in layers of cells that fall on the
//! song's time, confetti too), heat haze, water, candlelight, an aurora, stars. Light
//! that flickers does so slowly and a little (the candle's under a tenth
//! of its pool's light, a few times a second at most).

use crate::drive::Driver;
use crate::math::*;
use crate::pass::{knob, natural, Ctx, Family, Pass, Src, Unit};
use crate::passes::common::fall;

const W: Family = Family::Atmosphere;

fn haze(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let w = smoothstep(k[2], k[2] - k[3], uv.y);
    let col = hsv(k[1], 0.25, 0.85);
    mix3(c, col, k[0] * 0.75 * w * (0.6 + 0.4 * (1.0 - luma(c))))
}

/// Haze toward the far parts.
pub const HAZE: Pass = Pass {
    name: "haze",
    family: W,
    about: "haze: the picture fading into the air's colour toward the top and the distance (mountains, summer heat, dreams, longing)",
    knobs: &[
        knob("amount", "how thick the haze", (0.0, 1.0, 0.4), Unit::Share, (0.3, 0.2)),
        knob("hue", "the air's colour", (0.0, 1.0, 0.58), Unit::Turns, (0.1, 0.02)),
        knob("horizon", "where the haze starts, up from the bottom", (0.0, 1.0, 0.6), Unit::Share, (0.1, 0.1)),
        knob("height", "how far up it thickens", (0.1, 1.0, 0.5), Unit::Share, (0.2, 0.1)),
    ],
    natural: &[natural("amount", Driver::Quiet, 0.6), natural("height", Driver::Vocals, 0.5)],
    pairs: &["god_rays", "split_tone", "glow"],
    wgsl: r"
fn look_haze(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let w = smoothstep(lk_k(2u), lk_k(2u) - lk_k(3u), uv.y);
    let col = lk_hsv(lk_k(1u), 0.25, 0.85);
    return mix(c, col, lk_k(0u) * 0.75 * w * (0.6 + 0.4 * (1.0 - lk_luma(c))));
}
",
    cpu: haze,
    ..Pass::BASE
};

fn god_rays(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let light = v2(k[1], k[2]);
    let dir = (light - uv) * k[3];
    let mut acc = s3(0.0);
    let mut w = 1.0;
    for i in 0..16u32 {
        let t = i as f32 / 16.0;
        let tap = s.src(uv + dir * t);
        acc = acc + (tap - 0.62).max(0.0) * w;
        w *= 0.93;
    }
    c + acc * (k[0] * 0.35) * v3(1.0, 0.92, 0.75)
}

/// Rays of light from a bright point.
pub const GOD_RAYS: Pass = Pass {
    name: "god_rays",
    family: W,
    about: "rays of light streaming from a bright point through the picture (gospel, anthems, sunrise, faith, choruses)",
    knobs: &[
        knob("amount", "how bright the rays", (0.0, 1.0, 0.5), Unit::Share, (0.5, 0.15)),
        knob("x", "the light's place across", (0.0, 1.0, 0.5), Unit::Share, (0.1, 0.0)),
        knob("y", "the light's place down", (0.0, 1.0, 0.25), Unit::Share, (0.1, 0.0)),
        knob("length", "how long the rays", (0.1, 1.0, 0.5), Unit::Share, (0.3, 0.05)),
    ],
    cost: 4.0,
    natural: &[natural("amount", Driver::Chorus, 0.7), natural("length", Driver::Bass, 0.6)],
    pairs: &["haze", "glow", "lens_flare"],
    wgsl: r"
fn look_god_rays(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let light = vec2<f32>(lk_k(1u), lk_k(2u));
    let dir = (light - uv) * lk_k(3u);
    var acc = vec3<f32>(0.0);
    var w = 1.0;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let t = f32(i) / 16.0;
        let tap = lk_src(uv + dir * t);
        acc = acc + max(tap - vec3<f32>(0.62), vec3<f32>(0.0)) * w;
        w = w * 0.93;
    }
    return c + acc * (lk_k(0u) * 0.35) * vec3<f32>(1.0, 0.92, 0.75);
}
",
    cpu: god_rays,
    ..Pass::BASE
};

fn fog(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let t = u.time[0];
    let p = u.aspect(uv) * (k[1] * 2.0) + v2(t * k[2] * 0.05, t * 0.01);
    let f = smoothstep(0.3, 0.8, fbm(p, 191));
    let w = mix(1.0, smoothstep(0.2, 1.0, uv.y), k[4]);
    mix3(c, hsv(k[3], 0.12, 0.8), k[0] * 0.8 * f * w)
}

/// Fog drifting.
pub const FOG: Pass = Pass {
    name: "fog",
    family: W,
    about: "fog drifting across in slow banks, lying low if wanted (horror, Halloween, dub smoke, grief, mountains)",
    knobs: &[
        knob("amount", "how thick the fog", (0.0, 1.0, 0.4), Unit::Share, (0.3, 0.25)),
        knob("scale", "the banks' size: big (low) to small (high)", (0.5, 4.0, 1.5), Unit::Share, (0.5, 0.01)),
        knob("speed", "how fast it drifts", (0.0, 2.0, 0.5), Unit::Share, (0.8, 0.0)),
        knob("hue", "its colour", (0.0, 1.0, 0.6), Unit::Turns, (0.1, 0.01)),
        knob("low", "everywhere (0) to lying low (1)", (0.0, 1.0, 0.5), Unit::Share, (0.2, 0.1)),
    ],
    cost: 1.2,
    natural: &[natural("amount", Driver::Bass, 0.6), natural("speed", Driver::Energy, 0.6)],
    pairs: &["vignette", "haze", "night_vision"],
    wgsl: r"
fn look_fog(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let t = u.time.x;
    let p = lk_aspect(uv) * (lk_k(1u) * 2.0) + vec2<f32>(t * lk_k(2u) * 0.05, t * 0.01);
    let f = smoothstep(0.3, 0.8, lk_fbm(p, 191u));
    let w = mix(1.0, smoothstep(0.2, 1.0, uv.y), lk_k(4u));
    return mix(c, lk_hsv(lk_k(3u), 0.12, 0.8), lk_k(0u) * 0.8 * f * w);
}
",
    cpu: fog,
    ..Pass::BASE
};

fn rain(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let t = u.time[0];
    let p = u.px1080(uv);
    let slant = k[1] * 0.3;
    let q = v2(p.x + p.y * slant, p.y);
    let mut acc = 0.0;
    for l in 0..3u32 {
        let fl = l as f32;
        let cs = v2(14.0, 90.0) * (1.0 + 0.5 * fl);
        let qq = q / cs + v2(0.0, -t * k[3] * (3.0 + 2.0 * fl));
        let cv = qq.floor();
        let f = qq - cv;
        let (x, y) = (cell(cv.x), cell(cv.y));
        let on = step(hash2(x, y, 171 + l), k[0] * 0.5);
        let x0 = 0.2 + 0.6 * hash2(x, y, 174 + l);
        let len = 0.3 + 0.6 * k[2];
        let y0 = hash2(x, y, 177 + l) * (1.0 - len);
        let dx = ((f.x - x0) * cs.x).abs();
        let streak = (1.0 - smoothstep(0.4, 1.4 + 0.5 * fl, dx))
            * smoothstep(y0, y0 + 0.05, f.y)
            * (1.0 - smoothstep(y0 + len - 0.05, y0 + len, f.y));
        acc += streak * on * (0.25 + 0.1 * fl);
    }
    c * (1.0 - 0.08 * k[0]) + v3(0.75, 0.8, 0.85) * (acc * k[0].sqrt())
}

/// Rain falling.
pub const RAIN: Pass = Pass {
    name: "rain",
    family: W,
    about: "rain falling in streaks, near and far, slanting with the wind (heartbreak, grief, the city at night, blues)",
    knobs: &[
        knob("amount", "how heavy the rain", (0.0, 1.0, 0.5), Unit::Share, (0.5, 0.06)),
        knob("slant", "the wind's slant, left (-) or right (+)", (-1.0, 1.0, 0.2), Unit::Share, (0.4, 0.0)),
        knob("length", "how long the streaks", (0.0, 1.0, 0.5), Unit::Share, (0.2, 0.02)),
        knob("speed", "how fast it falls", (0.5, 2.0, 1.0), Unit::Share, (0.3, 0.0)),
    ],
    cost: 1.5,
    relief: 0.5,
    natural: &[natural("amount", Driver::Chorus, 0.8), natural("slant", Driver::Bass, 0.5)],
    pairs: &["fog", "split_tone", "glow"],
    wgsl: r"
fn look_rain(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let t = u.time.x;
    let p = lk_px1080(uv);
    let slant = lk_k(1u) * 0.3;
    let q = vec2<f32>(p.x + p.y * slant, p.y);
    var acc = 0.0;
    for (var l = 0u; l < 3u; l = l + 1u) {
        let fl = f32(l);
        let cs = vec2<f32>(14.0, 90.0) * (1.0 + 0.5 * fl);
        let qq = q / cs + vec2<f32>(0.0, -t * lk_k(3u) * (3.0 + 2.0 * fl));
        let cv = floor(qq);
        let f = qq - cv;
        let x = i32(cv.x);
        let y = i32(cv.y);
        let on = step(lk_hash2(x, y, 171u + l), lk_k(0u) * 0.5);
        let x0 = 0.2 + 0.6 * lk_hash2(x, y, 174u + l);
        let len = 0.3 + 0.6 * lk_k(2u);
        let y0 = lk_hash2(x, y, 177u + l) * (1.0 - len);
        let dx = abs((f.x - x0) * cs.x);
        let streak = (1.0 - smoothstep(0.4, 1.4 + 0.5 * fl, dx)) * smoothstep(y0, y0 + 0.05, f.y) * (1.0 - smoothstep(y0 + len - 0.05, y0 + len, f.y));
        acc = acc + streak * on * (0.25 + 0.1 * fl);
    }
    return c * (1.0 - 0.08 * lk_k(0u)) + vec3<f32>(0.75, 0.8, 0.85) * (acc * sqrt(lk_k(0u)));
}
",
    cpu: rain,
    ..Pass::BASE
};

fn snow(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let mut acc = 0.0;
    for l in 0..3u32 {
        let (local, h, on, near) = fall(u, uv, (60.0, -0.25 * k[3], k[2]), l, k[0] * 0.6);
        let r = (1.0 + 1.5 * h) * k[1] * near.sqrt() * 2.0;
        let flake = 1.0 - smoothstep(r * 0.3, r, local.length());
        acc += flake * on * (0.5 + 0.5 * near);
    }
    c + s3(acc * 0.85)
}

/// Snow falling.
pub const SNOW: Pass = Pass {
    name: "snow",
    family: W,
    about: "snow falling softly in layers, near flakes bigger, drifting with the wind (Christmas, winter, cosy songs, Schlager)",
    knobs: &[
        knob("amount", "how heavy the snow", (0.0, 1.0, 0.5), Unit::Share, (0.5, 0.08)),
        knob("size", "the flakes' size", (0.5, 3.0, 1.0), Unit::Share, (0.5, 0.05)),
        knob("wind", "blowing left (-) or right (+)", (-1.0, 1.0, 0.2), Unit::Share, (0.5, 0.0)),
        knob("speed", "how fast it falls", (0.3, 2.0, 1.0), Unit::Share, (0.3, 0.0)),
    ],
    cost: 1.3,
    relief: 0.5,
    natural: &[natural("amount", Driver::Chorus, 0.7), natural("wind", Driver::Bass, 0.5)],
    pairs: &["glow", "white_balance", "vignette"],
    wgsl: r"
fn look_snow(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    var acc = 0.0;
    for (var l = 0u; l < 3u; l = l + 1u) {
        let f = lk_fall(uv, 60.0, -0.25 * lk_k(3u), lk_k(2u), l, lk_k(0u) * 0.6);
        let near = 1.0 / (1.0 + 0.6 * f32(l));
        let r = (1.0 + 1.5 * f.z) * lk_k(1u) * sqrt(near) * 2.0;
        let flake = 1.0 - smoothstep(r * 0.3, r, length(f.xy));
        acc = acc + flake * f.w * (0.5 + 0.5 * near);
    }
    return c + vec3<f32>(acc * 0.85);
}
",
    cpu: snow,
    ..Pass::BASE
};

fn petals(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let mut o = s.src(uv);
    for l in 0..2u32 {
        let (local, h, on, near) = fall(u, uv, (110.0, -0.18, k[3]), l, k[0] * 0.5);
        let spin = u.time[0] * (0.8 + h) + h * 9.0;
        let q = turn(local, spin);
        let a = 7.0 * k[2] * near.sqrt() * (0.6 + 0.4 * spin.cos().abs());
        let b = 4.0 * k[2] * near.sqrt();
        let d = (q / v2(a, b)).length();
        let leaf = (1.0 - smoothstep(0.8, 1.0, d)) * on;
        let pink = v3(1.0, 0.72, 0.82) * (0.9 + 0.1 * h);
        let autumn = hsv(0.03 + 0.08 * h, 0.85, 0.8);
        let col = mix3(pink, autumn, k[1]) * (0.8 + 0.2 * (1.0 - d));
        o = mix3(o, col, leaf * 0.92);
    }
    o
}

/// Petals or leaves falling.
pub const PETALS: Pass = Pass {
    name: "petals",
    family: W,
    about: "blossom petals or autumn leaves tumbling down (love, spring, K-pop and J-pop ballads, autumn, nostalgia)",
    knobs: &[
        knob("amount", "how many fall", (0.0, 1.0, 0.4), Unit::Share, (0.5, 0.06)),
        knob("kind", "pink blossom (0) to autumn leaves (1)", (0.0, 1.0, 0.0), Unit::Share, (0.0, 0.02)),
        knob("size", "their size", (0.5, 3.0, 1.0), Unit::Share, (0.4, 0.05)),
        knob("wind", "blowing left (-) or right (+)", (-1.0, 1.0, 0.3), Unit::Share, (0.5, 0.0)),
    ],
    cost: 1.2,
    relief: 0.5,
    natural: &[natural("amount", Driver::Chorus, 0.7), natural("wind", Driver::Other, 0.6)],
    pairs: &["glow", "diffusion", "split_tone"],
    wgsl: r"
fn look_petals(uv: vec2<f32>) -> vec3<f32> {
    var o = lk_src(uv);
    for (var l = 0u; l < 2u; l = l + 1u) {
        let f = lk_fall(uv, 110.0, -0.18, lk_k(3u), l, lk_k(0u) * 0.5);
        let near = 1.0 / (1.0 + 0.6 * f32(l));
        let h = f.z;
        let spin = u.time.x * (0.8 + h) + h * 9.0;
        let q = lk_turn(f.xy, spin);
        let a = 7.0 * lk_k(2u) * sqrt(near) * (0.6 + 0.4 * abs(cos(spin)));
        let b = 4.0 * lk_k(2u) * sqrt(near);
        let d = length(q / vec2<f32>(a, b));
        let leaf = (1.0 - smoothstep(0.8, 1.0, d)) * f.w;
        let pink = vec3<f32>(1.0, 0.72, 0.82) * (0.9 + 0.1 * h);
        let autumn = lk_hsv(0.03 + 0.08 * h, 0.85, 0.8);
        let col = mix(pink, autumn, lk_k(1u)) * (0.8 + 0.2 * (1.0 - d));
        o = mix(o, col, leaf * 0.92);
    }
    return o;
}
",
    cpu: petals,
    ..Pass::BASE
};

fn dust(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let mut acc = 0.0;
    for l in 0..2u32 {
        let (local, h, on, near) = fall(u, uv, (50.0, 0.02, 0.1), l, k[0] * 0.5);
        let r = (0.8 + 1.4 * h) * k[1] * near.sqrt();
        acc += (1.0 - smoothstep(0.0, r, local.length())) * on;
    }
    let lit = 0.3 + 1.2 * luma(c) * k[2] + (1.0 - k[2]) * 0.4;
    c + v3(1.0, 0.92, 0.78) * (acc * 0.7 * lit)
}

/// Dust motes floating in the light.
pub const DUST: Pass = Pass {
    name: "dust",
    family: W,
    about: "dust motes floating slowly, glinting where the light is (old rooms, memories, folk, attics, quiet songs)",
    knobs: &[
        knob("amount", "how many motes", (0.0, 1.0, 0.4), Unit::Share, (0.5, 0.03)),
        knob("size", "their size", (0.5, 3.0, 1.0), Unit::Share, (0.6, 0.02)),
        knob("lit", "everywhere alike (0) to glinting only in the light (1)", (0.0, 1.0, 0.6), Unit::Share, (0.0, 0.02)),
    ],
    cost: 1.0,
    relief: 0.5,
    natural: &[natural("amount", Driver::Other, 0.7), natural("size", Driver::Bass, 0.5)],
    pairs: &["god_rays", "sepia", "film_damage"],
    wgsl: r"
fn look_dust(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    var acc = 0.0;
    for (var l = 0u; l < 2u; l = l + 1u) {
        let f = lk_fall(uv, 50.0, 0.02, 0.1, l, lk_k(0u) * 0.5);
        let near = 1.0 / (1.0 + 0.6 * f32(l));
        let r = (0.8 + 1.4 * f.z) * lk_k(1u) * sqrt(near);
        acc = acc + (1.0 - smoothstep(0.0, r, length(f.xy))) * f.w;
    }
    let lit = 0.3 + 1.2 * lk_luma(c) * lk_k(2u) + (1.0 - lk_k(2u)) * 0.4;
    return c + vec3<f32>(1.0, 0.92, 0.78) * (acc * 0.7 * lit);
}
",
    cpu: dust,
    ..Pass::BASE
};

fn heat_haze(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let p = u.px1080(uv) / k[1];
    let n = vnoise(v2(p.x / 40.0, p.y / 25.0 + u.time[0] * 1.5), 221) - 0.5;
    let w = mix(1.0, uv.y, k[2]);
    s.src(uv + v2(n * 0.4, n) * (k[0] * 4.0 * u.scale() * w) * u.px())
}

/// Heat shimmering.
pub const HEAT_HAZE: Pass = Pass {
    name: "heat_haze",
    family: W,
    about: "heat shimmering the picture, more near the ground (desert, summer, road trips, Americana, flamenco)",
    knobs: &[
        knob("amount", "how strong the shimmer", (0.0, 1.0, 0.4), Unit::Share, (0.5, 0.0)),
        knob("scale", "the ripples' size", (0.5, 3.0, 1.0), Unit::Share, (0.3, 0.0)),
        knob("ground", "everywhere (0) to near the ground (1)", (0.0, 1.0, 0.6), Unit::Share, (0.0, 0.0)),
    ],
    natural: &[natural("amount", Driver::Energy, 0.7)],
    pairs: &["haze", "white_balance", "grain"],
    wgsl: r"
fn look_heat_haze(uv: vec2<f32>) -> vec3<f32> {
    let p = lk_px1080(uv) / lk_k(1u);
    let n = lk_vnoise(vec2<f32>(p.x / 40.0, p.y / 25.0 + u.time.x * 1.5), 221u) - 0.5;
    let w = mix(1.0, uv.y, lk_k(2u));
    return lk_src(uv + vec2<f32>(n * 0.4, n) * (lk_k(0u) * 4.0 * lk_scale() * w) * u.size.zw);
}
",
    cpu: heat_haze,
    ..Pass::BASE
};

fn underwater(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let t = u.time[0];
    let wob = v2((uv.y * 30.0 + t * 1.5).sin(), (uv.x * 25.0 + t * 1.2).cos()) * (0.003 * k[0]);
    let c = s.src(uv + wob);
    let d = k[2];
    let absorbed = c * v3(1.0 - 0.6 * d, 1.0 - 0.25 * d, 1.0) + v3(0.0, 0.04, 0.08) * d;
    let p = u.px1080(uv) / 80.0;
    let n1 = vnoise(p + v2(t * 0.3, 0.0), 231);
    let n2 = vnoise(p * 1.3 + v2(5.0, -t * 0.25), 232);
    let ca = sat(1.0 - (n1 - n2).abs() * 2.5).powf(4.0) * (1.0 - uv.y * 0.6);
    let o = absorbed + v3(0.6, 0.9, 1.0) * (ca * k[1] * 0.35);
    mix3(s.src(uv), o, k[0])
}

/// Under water.
pub const UNDERWATER: Pass = Pass {
    name: "underwater",
    family: W,
    about: "under water: the reds absorbed, the picture wavering, caustic light dancing from above (the sea, dreams, ambient, dub)",
    knobs: &[
        knob("amount", "how much under water", (0.0, 1.0, 0.7), Unit::Share, (0.3, 0.1)),
        knob("caustics", "how bright the dancing light", (0.0, 1.0, 0.5), Unit::Share, (0.5, 0.08)),
        knob("depth", "how deep: how much red is lost", (0.0, 1.0, 0.5), Unit::Share, (0.3, 0.1)),
    ],
    cost: 1.3,
    natural: &[natural("caustics", Driver::Other, 0.7), natural("depth", Driver::Bass, 0.5)],
    pairs: &["god_rays", "liquid", "glow"],
    wgsl: r"
fn look_underwater(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time.x;
    let wob = vec2<f32>(sin(uv.y * 30.0 + t * 1.5), cos(uv.x * 25.0 + t * 1.2)) * (0.003 * lk_k(0u));
    let c = lk_src(uv + wob);
    let d = lk_k(2u);
    let absorbed = c * vec3<f32>(1.0 - 0.6 * d, 1.0 - 0.25 * d, 1.0) + vec3<f32>(0.0, 0.04, 0.08) * d;
    let p = lk_px1080(uv) / 80.0;
    let n1 = lk_vnoise(p + vec2<f32>(t * 0.3, 0.0), 231u);
    let n2 = lk_vnoise(p * 1.3 + vec2<f32>(5.0, -t * 0.25), 232u);
    let ca = pow(lk_sat(1.0 - abs(n1 - n2) * 2.5), 4.0) * (1.0 - uv.y * 0.6);
    let o = absorbed + vec3<f32>(0.6, 0.9, 1.0) * (ca * lk_k(1u) * 0.35);
    return mix(lk_src(uv), o, lk_k(0u));
}
",
    cpu: underwater,
    ..Pass::BASE
};

fn candle(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let t = u.time[0];
    let d = (u.aspect(uv) - u.aspect(v2(k[2], k[3]))).length() / k[4];
    let pool = (-d * d * 2.0).exp();
    let flicker =
        0.88 + 0.08 * vnoise(v2(t * 2.2, 0.5), 241) + 0.04 * vnoise(v2(t * 4.1 + 3.0, 0.5), 242);
    let warm = mix3(v3(1.0, 0.85, 0.6), v3(1.0, 0.62, 0.3), k[1]);
    let a = k[0];
    c * mix(1.0 - 0.45 * a, 1.0, pool * flicker) * mix3(s3(1.0), warm, a * pool)
        + warm * (pool * flicker * 0.12 * a)
}

/// Candlelight.
pub const CANDLE: Pass = Pass {
    name: "candle",
    family: W,
    about: "candlelight: a warm pool of gently flickering light, the rest in shadow (Christmas, chanson, intimate ballads, faith)",
    knobs: &[
        knob("amount", "how much candlelight and shadow", (0.0, 1.0, 0.5), Unit::Share, (0.3, 0.3)),
        knob("warmth", "pale gold (0) to deep orange (1)", (0.0, 1.0, 0.7), Unit::Share, (0.2, 0.02)),
        knob("x", "the candle's place across", (0.0, 1.0, 0.5), Unit::Share, (0.0, 0.0)),
        knob("y", "the candle's place down", (0.0, 1.0, 0.65), Unit::Share, (0.0, 0.0)),
        knob("reach", "how far its light reaches", (0.2, 1.0, 0.6), Unit::Share, (0.25, 0.15)),
    ],
    natural: &[natural("reach", Driver::Vocals, 0.6)],
    pairs: &["glow", "grain", "vignette"],
    wgsl: r"
fn look_candle(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let t = u.time.x;
    let d = length(lk_aspect(uv) - lk_aspect(vec2<f32>(lk_k(2u), lk_k(3u)))) / lk_k(4u);
    let pool = exp(-d * d * 2.0);
    let flicker = 0.88 + 0.08 * lk_vnoise(vec2<f32>(t * 2.2, 0.5), 241u) + 0.04 * lk_vnoise(vec2<f32>(t * 4.1 + 3.0, 0.5), 242u);
    let warm = mix(vec3<f32>(1.0, 0.85, 0.6), vec3<f32>(1.0, 0.62, 0.3), lk_k(1u));
    let a = lk_k(0u);
    return c * mix(1.0 - 0.45 * a, 1.0, pool * flicker) * mix(vec3<f32>(1.0), warm, a * pool) + warm * (pool * flicker * 0.12 * a);
}
",
    cpu: candle,
    ..Pass::BASE
};

fn aurora(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let t = u.time[0] * k[3];
    let x = u.aspect(uv).x;
    let mut acc = s3(0.0);
    for i in 0..2u32 {
        let fi = i as f32;
        let cy = 0.15
            + 0.08 * fi
            + 0.05 * (x * 2.5 + t * 0.15 + fi * 2.0).sin()
            + 0.04 * (vnoise(v2(x * 1.5 + t * 0.05, fi), 181 + i) - 0.5);
        let d = cy - uv.y;
        let glow = (-(d.max(0.0)) * (10.0 - 6.0 * k[2])).exp() * (-(-d).max(0.0) * 60.0).exp();
        let rays = 0.6 + 0.4 * vnoise(v2(x * 25.0, t * 0.3), 183 + i);
        acc = acc + hsv(k[1] + 0.08 * fi, 0.7, 1.0) * (glow * rays);
    }
    let dark = 1.0 - smoothstep(0.2, 0.6, luma(c));
    c + acc * (k[0] * dark * 0.8)
}

/// Northern lights.
pub const AURORA: Pass = Pass {
    name: "aurora",
    family: W,
    about: "northern lights: glowing curtains rippling in the sky, over the dark parts (winter, space, ambient, wonder)",
    knobs: &[
        knob("amount", "how bright the curtains", (0.0, 1.0, 0.5), Unit::Share, (0.5, 0.1)),
        knob("hue", "their colour (green by default)", (0.0, 1.0, 0.38), Unit::Turns, (0.2, 0.01)),
        knob("height", "how far down they reach", (0.0, 1.0, 0.35), Unit::Share, (0.2, 0.05)),
        knob("speed", "how fast they ripple", (0.0, 2.0, 0.5), Unit::Share, (0.6, 0.0)),
    ],
    cost: 1.2,
    natural: &[natural("amount", Driver::Other, 0.6), natural("hue", Driver::Part, 0.5), natural("speed", Driver::Energy, 0.6)],
    pairs: &["starfield", "glow", "white_balance"],
    wgsl: r"
fn look_aurora(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let t = u.time.x * lk_k(3u);
    let x = lk_aspect(uv).x;
    var acc = vec3<f32>(0.0);
    for (var i = 0u; i < 2u; i = i + 1u) {
        let fi = f32(i);
        let cy = 0.15 + 0.08 * fi + 0.05 * sin(x * 2.5 + t * 0.15 + fi * 2.0) + 0.04 * (lk_vnoise(vec2<f32>(x * 1.5 + t * 0.05, fi), 181u + i) - 0.5);
        let d = cy - uv.y;
        let glow = exp(-max(d, 0.0) * (10.0 - 6.0 * lk_k(2u))) * exp(-max(-d, 0.0) * 60.0);
        let rays = 0.6 + 0.4 * lk_vnoise(vec2<f32>(x * 25.0, t * 0.3), 183u + i);
        acc = acc + lk_hsv(lk_k(1u) + 0.08 * fi, 0.7, 1.0) * (glow * rays);
    }
    let dark = 1.0 - smoothstep(0.2, 0.6, lk_luma(c));
    return c + acc * (lk_k(0u) * dark * 0.8);
}
",
    cpu: aurora,
    ..Pass::BASE
};

fn starfield(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let p = u.px1080(uv) / 28.0;
    let cv = p.floor();
    let (x, y) = (cell(cv.x), cell(cv.y));
    let on = step(hash2(x, y, 251), k[0] * 0.5);
    let at = cv + v2(0.2 + 0.6 * hash2(x, y, 252), 0.2 + 0.6 * hash2(x, y, 253));
    let h = hash2(x, y, 254);
    let r = (0.6 + 1.2 * hash2(x, y, 255)) * k[2];
    let tw = mix(
        1.0,
        0.6 + 0.4 * (u.time[0] * (1.5 + 2.0 * h) + h * TAU).sin(),
        k[1],
    );
    let d = (p - at).length() * 28.0;
    let star = (1.0 - smoothstep(0.0, r, d)) * on * tw * (0.5 + 0.5 * h);
    let dark = 1.0 - smoothstep(0.15, 0.45, luma(c));
    c + v3(0.9, 0.93, 1.0) * (star * dark)
}

/// Stars twinkling in the dark parts.
pub const STARFIELD: Pass = Pass {
    name: "starfield",
    family: W,
    about: "stars twinkling slowly in the dark parts of the picture (space and stars, night, lullabies, dreams)",
    knobs: &[
        knob("amount", "how many stars", (0.0, 1.0, 0.5), Unit::Share, (0.4, 0.02)),
        knob("twinkle", "how much they twinkle (slowly)", (0.0, 1.0, 0.5), Unit::Share, (0.4, 0.01)),
        knob("size", "their size", (0.5, 3.0, 1.0), Unit::Share, (0.6, 0.02)),
    ],
    relief: 0.5,
    natural: &[natural("twinkle", Driver::High, 0.8), natural("size", Driver::Bass, 0.5)],
    pairs: &["aurora", "glow", "night_vision"],
    wgsl: r"
fn look_starfield(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let p = lk_px1080(uv) / 28.0;
    let cv = floor(p);
    let x = i32(cv.x);
    let y = i32(cv.y);
    let on = step(lk_hash2(x, y, 251u), lk_k(0u) * 0.5);
    let at = cv + vec2<f32>(0.2 + 0.6 * lk_hash2(x, y, 252u), 0.2 + 0.6 * lk_hash2(x, y, 253u));
    let h = lk_hash2(x, y, 254u);
    let r = (0.6 + 1.2 * lk_hash2(x, y, 255u)) * lk_k(2u);
    let tw = mix(1.0, 0.6 + 0.4 * sin(u.time.x * (1.5 + 2.0 * h) + h * LK_TAU), lk_k(1u));
    let d = length(p - at) * 28.0;
    let star = (1.0 - smoothstep(0.0, r, d)) * on * tw * (0.5 + 0.5 * h);
    let dark = 1.0 - smoothstep(0.15, 0.45, lk_luma(c));
    return c + vec3<f32>(0.9, 0.93, 1.0) * (star * dark);
}
",
    cpu: starfield,
    ..Pass::BASE
};

const GHOSTS: [(f32, f32); 4] = [(0.5, 0.05), (0.8, 0.03), (1.2, 0.08), (1.6, 0.04)];

fn lens_flare(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let p = u.aspect(uv);
    let l = u.aspect(v2(k[1], k[2]));
    let glow = (-(p - l).length() * 8.0).exp() * 0.6;
    let streak = (-(p.y - l.y).abs() * 120.0).exp() * (-(p.x - l.x).abs() * 2.5).exp() * 0.5;
    let warm = hsv(k[3], 0.5, 1.0);
    let mut ghosts = s3(0.0);
    for (i, (f, r)) in GHOSTS.into_iter().enumerate() {
        let at = l * (1.0 - 2.0 * f);
        let d = (p - at).length();
        let g = smoothstep(r, r * 0.6, d) * 0.25;
        ghosts = ghosts + hsv(k[3] + 0.1 * i as f32, 0.6, 1.0) * g;
    }
    screen(c, (warm * (glow + streak) + ghosts) * k[0])
}

/// A lens's flare from a light.
pub const LENS_FLARE: Pass = Pass {
    name: "lens_flare",
    family: W,
    about: "a lens flare: a light's glow and streak, its ghosts strung across the frame (summer, road trips, the 80s, stadium rock)",
    knobs: &[
        knob("amount", "how strong the flare", (0.0, 1.0, 0.5), Unit::Share, (0.5, 0.15)),
        knob("x", "the light's place across", (0.0, 1.0, 0.75), Unit::Share, (0.1, 0.0)),
        knob("y", "the light's place down", (0.0, 1.0, 0.25), Unit::Share, (0.1, 0.0)),
        knob("hue", "its colour", (0.0, 1.0, 0.08), Unit::Turns, (0.1, 0.0)),
    ],
    natural: &[natural("amount", Driver::Downbeat, 0.7)],
    pairs: &["light_leak", "god_rays", "glow"],
    wgsl: r"
fn look_lens_flare(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let p = lk_aspect(uv);
    let l = lk_aspect(vec2<f32>(lk_k(1u), lk_k(2u)));
    let glow = exp(-length(p - l) * 8.0) * 0.6;
    let streak = exp(-abs(p.y - l.y) * 120.0) * exp(-abs(p.x - l.x) * 2.5) * 0.5;
    let warm = lk_hsv(lk_k(3u), 0.5, 1.0);
    var fs = array<f32, 4>(0.5, 0.8, 1.2, 1.6);
    var rs = array<f32, 4>(0.05, 0.03, 0.08, 0.04);
    var ghosts = vec3<f32>(0.0);
    for (var i = 0u; i < 4u; i = i + 1u) {
        let at = l * (1.0 - 2.0 * fs[i]);
        let d = length(p - at);
        let g = smoothstep(rs[i], rs[i] * 0.6, d) * 0.25;
        ghosts = ghosts + lk_hsv(lk_k(3u) + 0.1 * f32(i), 0.6, 1.0) * g;
    }
    return lk_screen(c, (warm * (glow + streak) + ghosts) * lk_k(0u));
}
",
    cpu: lens_flare,
    ..Pass::BASE
};

fn tilt_shift(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let dist = (uv.y - k[0]).abs();
    let r = k[2] * 8.0 * u.scale() * smoothstep(k[1] * 0.5, k[1] * 0.5 + 0.25, dist);
    let px = u.px();
    let mut acc = s.src(uv);
    for i in 0..8u32 {
        let a = i as f32 * 2.399_963;
        let rr = ((i as f32 + 0.5) / 8.0).sqrt() * r;
        acc = acc + s.src(uv + v2(a.cos(), a.sin()) * px * rr);
    }
    saturate(acc / 9.0, 1.0 + 0.15 * k[2])
}

/// A tilted lens's narrow focus: the world as a model.
pub const TILT_SHIFT: Pass = Pass {
    name: "tilt_shift",
    family: W,
    about: "a tilted lens: only a band in focus, above and below it blurred, the world looking like a model (city pop, childhood, toy-like whimsy)",
    knobs: &[
        knob("focus", "where the sharp band is, down the frame", (0.0, 1.0, 0.55), Unit::Share, (0.15, 0.0)),
        knob("band", "how tall the sharp band", (0.05, 0.6, 0.2), Unit::Share, (0.1, 0.0)),
        knob("blur", "how blurred the rest", (0.0, 1.0, 0.6), Unit::Share, (0.4, 0.02)),
    ],
    cost: 2.5,
    natural: &[natural("blur", Driver::Bass, 0.6), natural("focus", Driver::Vocals, 0.4)],
    pairs: &["saturation", "curve"],
    wgsl: r"
fn look_tilt_shift(uv: vec2<f32>) -> vec3<f32> {
    let dist = abs(uv.y - lk_k(0u));
    let r = lk_k(2u) * 8.0 * lk_scale() * smoothstep(lk_k(1u) * 0.5, lk_k(1u) * 0.5 + 0.25, dist);
    var acc = lk_src(uv);
    for (var i = 0u; i < 8u; i = i + 1u) {
        let a = f32(i) * 2.399963;
        let rr = sqrt((f32(i) + 0.5) / 8.0) * r;
        acc = acc + lk_src(uv + vec2<f32>(cos(a), sin(a)) * u.size.zw * rr);
    }
    return lk_saturate(acc / 9.0, 1.0 + 0.15 * lk_k(2u));
}
",
    cpu: tilt_shift,
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
    family: W,
    about: "confetti fluttering down in every colour, thicker in the choruses (party, celebrations, disco, Schlager, weddings)",
    knobs: &[
        knob("amount", "how much confetti", (0.0, 1.0, 0.4), Unit::Share, (0.5, 0.04)),
        knob("size", "the pieces' size", (0.5, 3.0, 1.0), Unit::Share, (0.4, 0.03)),
        knob("wind", "blowing left (-) or right (+)", (-1.0, 1.0, 0.2), Unit::Share, (0.4, 0.0)),
    ],
    cost: 1.3,
    relief: 0.5,
    natural: &[natural("amount", Driver::Chorus, 0.7), natural("wind", Driver::Bass, 0.5)],
    pairs: &["glow", "sparks", "saturation"],
    wgsl: r"
fn look_confetti(uv: vec2<f32>) -> vec3<f32> {
    var o = lk_src(uv);
    for (var l = 0u; l < 3u; l = l + 1u) {
        let f = lk_fall(uv, 70.0, -0.3, lk_k(2u), l, lk_k(0u) * 0.6);
        let near = 1.0 / (1.0 + 0.6 * f32(l));
        let h = f.z;
        let spin = u.time.x * (2.0 + 3.0 * h) + h * 11.0;
        let q = lk_turn(f.xy, spin);
        let size = lk_k(1u) * sqrt(near);
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
