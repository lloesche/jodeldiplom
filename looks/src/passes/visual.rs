//! The visualizer: the music drawn (a scope's line, a ring and bars of
//! the stems' spectrum, rings on the beat, sparks from the hits, a
//! Lissajous curve) in the look's colours, and the frame fed back into
//! itself (zoomed, turned, swirled, flowing, fading: [`FEEDBACK`]), over
//! whatever the backdrop draws: faint over a story's picture, the whole
//! screen where there is none. Every shape is worked out from the song's
//! curves at the time heard ([`crate::drive::Feed`]), so it lands where
//! the music is; the feedback forgets the frame before on a seek or a new
//! look, and fades by the second, not the frame.

use crate::drive::{Driver, Feed, Signals};
use crate::math::*;
use crate::pass::{knob, natural, Ctx, Family, Pass, Src, DATA};
use crate::passes::film::aspect;

const V: Family = Family::Visualizer;

/// The stems' levels (bass, drums, other, vocals) at `t`: the song's
/// curves where there are some, else the frame's drivers.
fn levels(feed: &Feed, t: f64) -> [f32; 4] {
    let s: Signals = feed.curves.map_or(*feed.signals, |c| c.at(t.max(0.0)));
    [
        s.get(Driver::Bass),
        s.get(Driver::Drums),
        s.get(Driver::Other),
        s.get(Driver::Vocals),
    ]
}

/// Data from 64 numbers.
fn pack(v: &[f32; 64]) -> [[f32; 4]; DATA] {
    std::array::from_fn(|i| [v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]])
}

/// The scope's line: the last two seconds of the stems, as a wave each
/// at its own pitch, left (two seconds ago) to right (now), -1 to 1.
fn scope_data(feed: &Feed) -> [[f32; 4]; DATA] {
    let mut v = [0.0; 64];
    let t = feed.t;
    for (i, out) in v.iter_mut().enumerate() {
        let at = t - (63 - i) as f64 * (2.0 / 63.0);
        let [bass, drums, other, vocals] = levels(feed, at);
        let x = i as f32;
        let ph = t as f32;
        let wave = bass * (x * 0.25 + ph * 3.0).sin()
            + vocals * 0.8 * (x * 0.62 + ph * 5.0).sin()
            + other * 0.5 * (x * 1.1 - ph * 2.0).sin()
            + drums * 0.6 * (hash2(i as i32, (t * 30.0) as i32, 281) - 0.5);
        *out = (wave / 1.6).clamp(-1.0, 1.0);
    }
    pack(&v)
}

/// The stems' spectrum: 32 bins, low to high, each 0 to 1 (the bass low,
/// the voice in the middle, the rest broad, the drums' kicks low and
/// their bright hits high), with a little life of its own, saturating
/// softly.
fn spectrum_data(feed: &Feed) -> [[f32; 4]; DATA] {
    let s = feed.signals;
    let [bass, drums, other, vocals] = levels(feed, feed.t);
    let (kick, high) = (s.get(Driver::Kick), s.get(Driver::High));
    let mut v = [0.0; 64];
    for (j, out) in v.iter_mut().take(32).enumerate() {
        let x = j as f32;
        let b = bass * (-x / 4.0).exp();
        let vo = vocals * (-((x - 12.0) / 5.0).powi(2)).exp();
        let o = other * 0.8 * (-((x - 18.0) / 8.0).powi(2)).exp();
        let d =
            drums * 0.3 + kick * 0.6 * (-x / 3.0).exp() + high * 0.7 * smoothstep(18.0, 28.0, x);
        let life = 0.15
            * vnoise(v2(x * 0.7, feed.t as f32 * 4.0), 283)
            * (bass + drums + other + vocals)
            * 0.25;
        // Softly saturating: a loud chorus fills the bins most of the way,
        // never all of them to the top.
        let sum = (b + vo + o + d * 0.6) * 0.45 + life;
        *out = (1.0 - (-1.6 * sum).exp()).clamp(0.0, 1.0);
    }
    pack(&v)
}

/// The beat: where it is in the beat, which beat of the bar, how many in
/// a bar.
fn beat_data(feed: &Feed) -> [[f32; 4]; DATA] {
    let mut v = [0.0; 64];
    v[0] = feed.signals.get(Driver::BeatPhase);
    if let Some(c) = feed.curves {
        let i = c.beats().partition_point(|&b| b <= feed.t);
        let bar = c.bars().partition_point(|&b| b <= feed.t);
        let first = bar
            .checked_sub(1)
            .and_then(|k| c.bars().get(k))
            .map_or(0, |&b| c.beats().partition_point(|&x| x < b));
        v[1] = i.saturating_sub(1).saturating_sub(first) as f32;
    }
    v[2] = 4.0;
    pack(&v)
}

/// The sparks: the last eight hits' ages (seconds) and seeds.
fn spark_data(feed: &Feed) -> [[f32; 4]; DATA] {
    let mut v = [0.0; 64];
    for e in 0..8 {
        v[2 * e] = 99.0;
    }
    if let Some(c) = feed.curves {
        for (e, (j, at)) in c.hits_before(feed.t, 8).enumerate() {
            v[2 * e] = (feed.t - at) as f32;
            v[2 * e + 1] = (j % 65536) as f32;
        }
    } else {
        // No song: a hit on each of the drivers' beats.
        let phase = feed.signals.get(Driver::BeatPhase);
        v[0] = phase * 0.5;
        v[1] = (feed.t * 2.0).floor() as f32 % 65536.0;
    }
    pack(&v)
}

/// The curve: its two frequencies (from the key), its phase (with the
/// tempo), how wide (the bass) and tall (the voice).
fn lissajous_data(feed: &Feed) -> [[f32; 4]; DATA] {
    const RATIOS: [(f32, f32); 6] = [
        (1.0, 2.0),
        (2.0, 3.0),
        (3.0, 4.0),
        (3.0, 5.0),
        (4.0, 5.0),
        (5.0, 6.0),
    ];
    let s = feed.signals;
    let key = (s.get(Driver::KeyHue) * 12.0).round() as usize % RATIOS.len();
    let (a, b) = RATIOS[key];
    let [bass, _, _, vocals] = levels(feed, feed.t);
    let mut v = [0.0; 64];
    v[0] = a;
    v[1] = b;
    v[2] = (feed.t * 0.3 * f64::from(s.get(Driver::Tempo).max(0.5))) as f32 % TAU;
    v[3] = 0.5 + 0.5 * bass;
    v[4] = 0.5 + 0.5 * vocals;
    pack(&v)
}

/// The middle the round shapes are drawn about: a little above the
/// screen's, clear of the lyrics.
const ROUND_MIDDLE: V2 = v2(0.0, -0.08);

fn feedback(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let (t, dt, forget) = (u.time[0], u.time[2], u.time[3]);
    let p = aspect(u, uv);
    let ang = (k[2] * 1.5 + k[3] * 2.0 * (1.0 - p.length().min(1.0))) * dt;
    let mut q = turn(p * (-k[1] * 0.6 * dt).exp(), ang);
    let fl = v2(
        vnoise(p * 3.0 + s2(t * 0.2), 271) - 0.5,
        vnoise(p * 3.0 + s2(7.0 - t * 0.2), 272) - 0.5,
    );
    q = q + fl * (k[4] * 0.25 * dt);
    let puv = q / v2(u.size[0] / u.size[1], 1.0) + 0.5;
    let echo = hue_turn(s.prev(puv), k[6] * 0.3 * dt).max(0.0).map(knee);
    let keep = mix(0.01, 0.9, k[0]).powf(dt) * (1.0 - forget);
    let new = s.src(uv);
    (new * k[5]).vmax(echo * keep)
}

/// The frame fed back into itself.
pub const FEEDBACK: Pass = Pass {
    name: "feedback",
    family: V,
    about: "the frame before fed back under this one: zoomed, turned, swirled and flowing, fading by the second, the new picture over it (trails and echoes; with little of the scene, a full visualizer)",
    knobs: &[
        knob("trail", "how long the echoes last: a moment (0) to many seconds (1)", (0.0, 1.0, 0.6), "share", (0.3, 0.1)),
        knob("zoom", "the echoes shrinking away (-) or growing out (+)", (-1.0, 1.0, 0.2), "share", (0.6, 0.02)),
        knob("turn", "the echoes turning one way (-) or the other (+)", (-1.0, 1.0, 0.0), "share", (0.6, 0.0)),
        knob("swirl", "the echoes swirling, more in the middle", (0.0, 1.0, 0.0), "share", (0.6, 0.0)),
        knob("flow", "the echoes flowing in a noise's currents", (0.0, 1.0, 0.2), "share", (0.5, 0.0)),
        knob("scene", "how much of the new picture shows over the echoes", (0.0, 1.0, 0.85), "share", (0.4, 0.45)),
        knob("hue", "the echoes' colour turning as they fade", (-1.0, 1.0, 0.0), "share", (0.4, 0.0)),
    ],
    cost: 2.0,
    relief: 0.6,
    history: true,
    natural: &[
        natural("zoom", Driver::Bass, 0.6),
        natural("turn", Driver::Snare, 0.6),
        natural("swirl", Driver::Chorus, 0.5),
        natural("scene", Driver::Instrumental, -0.6),
    ],
    pairs: &["kaleidoscope", "scope", "spectrum_ring", "hue"],
    wgsl: r"
fn look_feedback(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time.x;
    let dt = u.time.z;
    let forget = u.time.w;
    let p = lk_aspect(uv);
    let ang = (u.k0.z * 1.5 + u.k0.w * 2.0 * (1.0 - min(length(p), 1.0))) * dt;
    var q = lk_turn(p * exp(-u.k0.y * 0.6 * dt), ang);
    let fl = vec2<f32>(lk_vnoise(p * 3.0 + vec2<f32>(t * 0.2), 271u) - 0.5, lk_vnoise(p * 3.0 + vec2<f32>(7.0 - t * 0.2), 272u) - 0.5);
    q = q + fl * (u.k1.x * 0.25 * dt);
    let puv = q / vec2<f32>(u.size.x / u.size.y, 1.0) + 0.5;
    let echo = lk_knee(max(lk_hue_turn(lk_prev(puv), u.k1.z * 0.3 * dt), vec3<f32>(0.0)));
    let keep = pow(mix(0.01, 0.9, u.k0.x), dt) * (1.0 - forget);
    let fresh = lk_src(uv);
    return max(fresh * u.k1.y, echo * keep);
}
",
    cpu: feedback,
    ..Pass::BASE
};

/// A shape's colour: a hue, or the cover's own colour.
fn shape_colour(u: &Ctx, hue: f32, cover: f32) -> V3 {
    mix3(hsv(hue, 0.6, 1.0), u.pal(0), cover)
}

fn scope(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let x = uv.x * 63.0;
    let i = x.floor().clamp(0.0, 63.0);
    let f = x - i;
    let a = u.d(i as usize);
    let b = u.d((i as usize + 1).min(63));
    let v = mix(a, b, f);
    let y = k[4] + v * k[0] * 0.2;
    let dist = (uv.y - y).abs() * u.size[1];
    let th = k[1] * u.scale();
    let core = 1.0 - smoothstep(th * 0.5, th * 0.5 + 1.0, dist);
    let g = (-dist / (th * 4.0)).exp() * k[3];
    add_light(c, shape_colour(u, k[2], k[5]) * (core * 0.75 + g * 0.35))
}

/// An oscilloscope's line.
pub const SCOPE: Pass = Pass {
    name: "scope",
    family: V,
    about: "an oscilloscope's glowing line across the screen, the stems' last two seconds as a wave (electronic, synth-pop, instrumental breaks)",
    knobs: &[
        knob("height", "how tall the wave", (0.0, 1.0, 0.4), "share", (0.4, 0.02)),
        knob("thick", "the line's width", (1.0, 8.0, 2.5), "px", (2.0, 0.02)),
        knob("hue", "its colour", (0.0, 1.0, 0.5), "turns", (0.2, 0.0)),
        knob("glow", "how much it glows", (0.0, 1.0, 0.5), "share", (0.4, 0.03)),
        knob("y", "where it is, down the screen", (0.0, 1.0, 0.35), "share", (0.0, 0.03)),
        knob("cover", "the hue (0) or the cover's colour (1)", (0.0, 1.0, 0.0), "share", (0.0, 0.0)),
    ],
    relief: 0.7,
    natural: &[natural("glow", Driver::Level, 0.5)],
    pairs: &["feedback", "crt", "glow"],
    wgsl: r"
fn lk_shape_colour(hue: f32, cover: f32) -> vec3<f32> { return mix(lk_hsv(hue, 0.6, 1.0), u.pal0.xyz, cover); }
fn look_scope(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let x = uv.x * 63.0;
    let i = clamp(floor(x), 0.0, 63.0);
    let f = x - i;
    let a = lk_d(u32(i));
    let b = lk_d(min(u32(i) + 1u, 63u));
    let v = mix(a, b, f);
    let y = u.k1.x + v * u.k0.x * 0.2;
    let dist = abs(uv.y - y) * u.size.y;
    let th = u.k0.y * lk_scale();
    let core = 1.0 - smoothstep(th * 0.5, th * 0.5 + 1.0, dist);
    let g = exp(-dist / (th * 4.0)) * u.k0.w;
    return lk_add_light(c, lk_shape_colour(u.k0.z, u.k1.y) * (core * 0.75 + g * 0.35));
}
",
    cpu: scope,
    data: Some(scope_data),
    ..Pass::BASE
};

fn spectrum_ring(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let p = aspect(u, uv) - ROUND_MIDDLE;
    let r = p.length();
    let a = p.y.atan2(p.x) / TAU + 0.5;
    let m = (fract(a + 0.25) * 2.0 - 1.0).abs();
    let j = (m * 31.999).floor();
    let v = u.d(j as usize);
    let gap = fract(m * 32.0);
    let bar = smoothstep(0.1, 0.2, gap) * (1.0 - smoothstep(0.8, 0.9, gap));
    let px = 1.5 * u.size[3];
    let r0 = k[0];
    let r1 = r0 + v * k[1] * 0.22 + 0.004;
    let inside = smoothstep(r0 - px, r0, r) * (1.0 - smoothstep(r1, r1 + px, r)) * bar;
    let g = (-(r - r0).abs() * 40.0).exp() * k[4] * 0.3;
    add_light(
        c,
        hsv(k[2] + k[3] * m * 0.5, 0.7, 1.0) * (inside * 0.6 + g * 0.6),
    )
}

/// A ring of the spectrum.
pub const SPECTRUM_RING: Pass = Pass {
    name: "spectrum_ring",
    family: V,
    about: "a ring of the stems' spectrum, its bars standing out from a circle round the middle, mirrored (EDM, house, trance, hip-hop visualizers)",
    knobs: &[
        knob("radius", "the ring's size", (0.1, 0.45, 0.25), "share", (0.08, 0.02)),
        knob("length", "how far the bars reach", (0.0, 1.0, 0.5), "share", (0.4, 0.03)),
        knob("hue", "its colour", (0.0, 1.0, 0.6), "turns", (0.2, 0.03)),
        knob("spread", "one colour (0) to a rainbow round the ring (1)", (0.0, 1.0, 0.5), "share", (0.2, 0.0)),
        knob("glow", "how much the circle glows", (0.0, 1.0, 0.4), "share", (0.4, 0.03)),
    ],
    relief: 0.7,
    cells: true,
    natural: &[natural("radius", Driver::Kick, 0.4), natural("hue", Driver::Part, 0.5)],
    pairs: &["feedback", "beat_rings", "kaleidoscope"],
    wgsl: r"
fn look_spectrum_ring(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let p = lk_aspect(uv) - vec2<f32>(0.0, -0.08);
    let r = length(p);
    let a = atan2(p.y, p.x) / LK_TAU + 0.5;
    let m = abs(fract(a + 0.25) * 2.0 - 1.0);
    let j = floor(m * 31.999);
    let v = lk_d(u32(j));
    let gap = fract(m * 32.0);
    let bar = smoothstep(0.1, 0.2, gap) * (1.0 - smoothstep(0.8, 0.9, gap));
    let px = 1.5 * u.size.w;
    let r0 = u.k0.x;
    let r1 = r0 + v * u.k0.y * 0.22 + 0.004;
    let inside = smoothstep(r0 - px, r0, r) * (1.0 - smoothstep(r1, r1 + px, r)) * bar;
    let g = exp(-abs(r - r0) * 40.0) * u.k1.x * 0.3;
    return lk_add_light(c, lk_hsv(u.k0.z + u.k0.w * m * 0.5, 0.7, 1.0) * (inside * 0.6 + g * 0.6));
}
",
    cpu: spectrum_ring,
    data: Some(spectrum_data),
    ..Pass::BASE
};

fn bars(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let n = k[0];
    let x = uv.x * n;
    let j = x.floor();
    let bin = (j / n * 31.99).floor();
    let v = u.d(bin as usize);
    let gap = fract(x);
    let bar = smoothstep(0.08, 0.14, gap) * (1.0 - smoothstep(0.86, 0.92, gap));
    let h = v * k[1] * 0.3;
    let px = 1.5 * u.size[3];
    let inside = (1.0 - smoothstep(h, h + px, uv.y)) * bar * step(0.002, h);
    let shade = 0.6 + 0.4 * (1.0 - uv.y / h.max(1e-3)).max(0.0);
    add_light(
        c,
        hsv(k[2] + k[3] * uv.x * 0.5, 0.75, 1.0) * (inside * shade * 0.65),
    )
}

/// Spectrum bars from the top edge.
pub const BARS: Pass = Pass {
    name: "bars",
    family: V,
    about: "a spectrum's bars hanging from the top of the screen, clear of the lyrics (pop, disco, club, chiptune)",
    knobs: &[
        knob("count", "how many bars", (8.0, 64.0, 32.0), "steps", (0.0, 0.02)),
        knob("height", "how far down they reach", (0.0, 1.0, 0.4), "share", (0.4, 0.04)),
        knob("hue", "their colour", (0.0, 1.0, 0.55), "turns", (0.2, 0.02)),
        knob("spread", "one colour (0) to a rainbow across (1)", (0.0, 1.0, 0.5), "share", (0.2, 0.0)),
    ],
    relief: 0.7,
    cells: true,
    natural: &[natural("hue", Driver::Part, 0.5)],
    pairs: &["pixelate", "crt", "feedback"],
    wgsl: r"
fn look_bars(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let n = u.k0.x;
    let x = uv.x * n;
    let j = floor(x);
    let bin = floor(j / n * 31.99);
    let v = lk_d(u32(bin));
    let gap = fract(x);
    let bar = smoothstep(0.08, 0.14, gap) * (1.0 - smoothstep(0.86, 0.92, gap));
    let h = v * u.k0.y * 0.3;
    let px = 1.5 * u.size.w;
    let inside = (1.0 - smoothstep(h, h + px, uv.y)) * bar * step(0.002, h);
    let shade = 0.6 + 0.4 * max(1.0 - uv.y / max(h, 1e-3), 0.0);
    return lk_add_light(c, lk_hsv(u.k0.z + u.k0.w * uv.x * 0.5, 0.75, 1.0) * (inside * shade * 0.65));
}
",
    cpu: bars,
    data: Some(spectrum_data),
    ..Pass::BASE
};

fn beat_rings(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let r = (aspect(u, uv) - ROUND_MIDDLE).length();
    let w = k[3] * u.scale() * u.size[3];
    let mut acc = 0.0;
    for i in 0..4u32 {
        let fi = i as f32;
        let age = u.d(0) + fi;
        let rad = age * k[1] * 0.16;
        let ring = (-((r - rad) / w).powi(2)).exp();
        let idx = u.d(1) - fi;
        let down = step((idx - (idx / u.d(2)).floor() * u.d(2)).abs(), 0.01);
        acc += ring * (-age * 0.8).exp() * (1.0 + down);
    }
    add_light(c, hsv(k[2], 0.6, 1.0) * (acc * k[0] * 0.5))
}

/// Rings on the beat.
pub const BEAT_RINGS: Pass = Pass {
    name: "beat_rings",
    family: V,
    about: "rings sent out from the middle on every beat, the bar's first brighter (dance, house, techno, disco)",
    knobs: &[
        knob("amount", "how bright the rings", (0.0, 1.0, 0.6), "share", (0.4, 0.05)),
        knob("speed", "how fast they spread", (0.5, 2.0, 1.0), "share", (0.4, 0.0)),
        knob("hue", "their colour", (0.0, 1.0, 0.1), "turns", (0.2, 0.0)),
        knob("width", "how wide a ring", (1.0, 10.0, 3.0), "px", (3.0, 0.02)),
    ],
    relief: 0.7,
    natural: &[natural("amount", Driver::Energy, 0.5)],
    pairs: &["spectrum_ring", "feedback", "glow"],
    wgsl: r"
fn look_beat_rings(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let r = length(lk_aspect(uv) - vec2<f32>(0.0, -0.08));
    let w = u.k0.w * lk_scale() * u.size.w;
    var acc = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let fi = f32(i);
        let age = lk_d(0u) + fi;
        let rad = age * u.k0.y * 0.16;
        let ring = exp(-pow((r - rad) / w, 2.0));
        let idx = lk_d(1u) - fi;
        let down = step(abs(idx - floor(idx / lk_d(2u)) * lk_d(2u)), 0.01);
        acc = acc + ring * exp(-age * 0.8) * (1.0 + down);
    }
    return lk_add_light(c, lk_hsv(u.k0.z, 0.6, 1.0) * (acc * u.k0.x * 0.5));
}
",
    cpu: beat_rings,
    data: Some(beat_data),
    ..Pass::BASE
};

fn sparks(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let p = aspect(u, uv);
    let ar = u.size[0] / u.size[1];
    let mut acc = s3(0.0);
    for e in 0..8usize {
        let age = u.d(2 * e);
        let seed = u.d(2 * e + 1) as u32;
        let fade = (-age * 2.5).exp() * step(age, 1.5);
        let origin = v2(
            (0.15 + 0.7 * hash2(seed as i32, 1, 291) - 0.5) * ar,
            0.12 + 0.4 * hash2(seed as i32, 2, 291) - 0.5,
        );
        let col = hsv(k[2] + 0.08 * hash2(seed as i32, 3, 291), 0.6, 1.0);
        let mut burst = 0.0;
        for n in 0..10i32 {
            let a = hash2(seed as i32, 10 + n, 292) * TAU;
            let speed = (0.2 + 0.5 * hash2(seed as i32, 30 + n, 292)) * k[3] * 0.6;
            let at = origin
                + v2(a.cos(), a.sin()) * (speed * age * (1.0 - 0.35 * age))
                + v2(0.0, 0.15 * age * age);
            let d = (p - at).length() * u.size[1];
            let r = (2.0 + 2.0 * hash2(seed as i32, 50 + n, 292)) * k[1] * u.scale();
            burst += 1.0 - smoothstep(r * 0.3, r, d);
        }
        acc = acc + col * (burst * fade);
    }
    add_light(c, acc * (k[0] * 0.7))
}

/// Sparks flying from the hits.
pub const SPARKS: Pass = Pass {
    name: "sparks",
    family: V,
    about: "a burst of sparks flying from somewhere above the lyrics on each kick, snare and bass note, falling as they fade (rock, party, fireworks, celebrations)",
    knobs: &[
        knob("amount", "how bright the sparks", (0.0, 1.0, 0.6), "share", (0.4, 0.04)),
        knob("size", "their size", (0.5, 3.0, 1.0), "share", (0.5, 0.02)),
        knob("hue", "their colour", (0.0, 1.0, 0.08), "turns", (0.2, 0.0)),
        knob("spread", "how far they fly", (0.0, 1.0, 0.5), "share", (0.4, 0.0)),
    ],
    cost: 2.0,
    relief: 0.7,
    natural: &[natural("amount", Driver::Energy, 0.4)],
    pairs: &["glow", "feedback", "beat_rings"],
    wgsl: r"
fn look_sparks(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let p = lk_aspect(uv);
    let ar = u.size.x / u.size.y;
    var acc = vec3<f32>(0.0);
    for (var e = 0u; e < 8u; e = e + 1u) {
        let age = lk_d(2u * e);
        let seed = i32(u32(lk_d(2u * e + 1u)));
        let fade = exp(-age * 2.5) * step(age, 1.5);
        let origin = vec2<f32>((0.15 + 0.7 * lk_hash2(seed, 1, 291u) - 0.5) * ar, 0.12 + 0.4 * lk_hash2(seed, 2, 291u) - 0.5);
        let col = lk_hsv(u.k0.z + 0.08 * lk_hash2(seed, 3, 291u), 0.6, 1.0);
        var burst = 0.0;
        for (var n = 0; n < 10; n = n + 1) {
            let a = lk_hash2(seed, 10 + n, 292u) * LK_TAU;
            let speed = (0.2 + 0.5 * lk_hash2(seed, 30 + n, 292u)) * u.k0.w * 0.6;
            let at = origin + vec2<f32>(cos(a), sin(a)) * (speed * age * (1.0 - 0.35 * age)) + vec2<f32>(0.0, 0.15 * age * age);
            let d = length(p - at) * u.size.y;
            let r = (2.0 + 2.0 * lk_hash2(seed, 50 + n, 292u)) * u.k0.y * lk_scale();
            burst = burst + 1.0 - smoothstep(r * 0.3, r, d);
        }
        acc = acc + col * (burst * fade);
    }
    return lk_add_light(c, acc * (u.k0.x * 0.7));
}
",
    cpu: sparks,
    data: Some(spark_data),
    ..Pass::BASE
};

fn lissajous(u: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    let k = &u.k;
    let c = s.src(uv);
    let p = aspect(u, uv) - ROUND_MIDDLE;
    let (a, b, ph) = (u.d(0), u.d(1), u.d(2));
    let (ax, ay) = (u.d(3) * k[0], u.d(4) * k[0]);
    let curve = |i: f32| {
        let s0 = i / 64.0 * TAU;
        v2(ax * (a * s0 + ph).sin(), ay * (b * s0).sin())
    };
    let mut dmin: f32 = 9.0;
    let mut p0 = curve(0.0);
    for i in 1..65u32 {
        let p1 = curve(i as f32);
        let e = p1 - p0;
        let h = sat((p - p0).dot(e) / e.dot(e).max(1e-8));
        dmin = dmin.min((p - p0 - e * h).length());
        p0 = p1;
    }
    let dpx = dmin * u.size[1];
    let th = k[1] * u.scale();
    let core = 1.0 - smoothstep(th * 0.5, th * 0.5 + 1.0, dpx);
    let g = (-dpx / (th * 5.0)).exp() * k[3];
    add_light(c, hsv(k[2], 0.55, 1.0) * (core * 0.75 + g * 0.35))
}

/// A Lissajous curve.
pub const LISSAJOUS: Pass = Pass {
    name: "lissajous",
    family: V,
    about: "a Lissajous figure turning in the middle, its loops from the key, its width the bass and height the voice (ambient, electronic, space, jazz)",
    knobs: &[
        knob("size", "how big", (0.1, 0.45, 0.3), "share", (0.08, 0.02)),
        knob("thick", "the line's width", (1.0, 6.0, 2.0), "px", (1.5, 0.02)),
        knob("hue", "its colour", (0.0, 1.0, 0.45), "turns", (0.2, 0.0)),
        knob("glow", "how much it glows", (0.0, 1.0, 0.5), "share", (0.4, 0.03)),
    ],
    cost: 2.5,
    relief: 0.7,
    natural: &[natural("glow", Driver::Vocals, 0.5)],
    pairs: &["feedback", "scope", "starfield"],
    wgsl: r"
fn lk_lissajous_at(i: f32, a: f32, b: f32, ph: f32, ax: f32, ay: f32) -> vec2<f32> {
    let s0 = i / 64.0 * LK_TAU;
    return vec2<f32>(ax * sin(a * s0 + ph), ay * sin(b * s0));
}
fn look_lissajous(uv: vec2<f32>) -> vec3<f32> {
    let c = lk_src(uv);
    let p = lk_aspect(uv) - vec2<f32>(0.0, -0.08);
    let a = lk_d(0u);
    let b = lk_d(1u);
    let ph = lk_d(2u);
    let ax = lk_d(3u) * u.k0.x;
    let ay = lk_d(4u) * u.k0.x;
    var dmin = 9.0;
    var p0 = lk_lissajous_at(0.0, a, b, ph, ax, ay);
    for (var i = 1u; i < 65u; i = i + 1u) {
        let p1 = lk_lissajous_at(f32(i), a, b, ph, ax, ay);
        let e = p1 - p0;
        let h = lk_sat(dot(p - p0, e) / max(dot(e, e), 1e-8));
        dmin = min(dmin, length(p - p0 - e * h));
        p0 = p1;
    }
    let dpx = dmin * u.size.y;
    let th = u.k0.y * lk_scale();
    let core = 1.0 - smoothstep(th * 0.5, th * 0.5 + 1.0, dpx);
    let g = exp(-dpx / (th * 5.0)) * u.k0.w;
    return lk_add_light(c, lk_hsv(u.k0.z, 0.55, 1.0) * (core * 0.75 + g * 0.35));
}
",
    cpu: lissajous,
    data: Some(lissajous_data),
    ..Pass::BASE
};
