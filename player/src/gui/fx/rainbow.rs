//! The streak's rainbow: while the singing shines, rainbow bands fan out
//! from the playhead behind the pills, as many as the tier's row has,
//! brighter and faster the higher, lighting the whole lane as much as the
//! row says ([`bands`]); a swirl turns round the singer's arrow
//! ([`swirl`]); in the picture, a rainbow edge on the bar under the word
//! sung ([`rim`]); the shouts (the multiplier, the verdict) in rainbow
//! ([`text_tint`], its lyric treatment). Lost, the bands collapse into the
//! playhead and the colour drains, as the shine's clock says
//! ([`crate::gui::shine::shine_at`]). It eases by that clock itself, as
//! strong as its rows list it where the glow comes from or goes to
//! ([`super::Live::listed`]): its light, and its text's tint, by as much.
//!
//! These functions are the shader's reference ([`wgsl`]), held to it on
//! the GPU ([`REFERENCE`]).

use super::{At, Fx, Layer, Live, Lyric, Out, Reference, Slot, Spec, Text, Treat};
use crate::gui::shine::{hue, shine_at, Glow, SHOWS, WHOLE};
use crate::gui::sparks::TAU;
use crate::gui::stage::gpu::{Instance, Shape};
use crate::gui::stage::LaneBox;
use crate::gui::wgsl_math::{mix3, round_box, smoothstep};
use crate::tiers::{held, wgsl_array, TIERS};

/// A streak's rainbow bands fanning out from (`shape[0]`, `shape[1]`)
/// over a lane `shape[2]`, `shape[3]` half wide and high, within the lane
/// (`uv`: its centre and half size); `params[1..]` its shine (level, from,
/// since); `color[3]` how much dimmer than whole it is (0: whole, as
/// strong as listed at 1): light, added ([`bands_of`]).
pub const BANDS: f32 = Fx::Rainbow.kind(0);
/// A streak's swirl round (`shape[0]`, `shape[1]`), `shape[2]` out; its
/// shine as [`BANDS`]' ([`swirl_of`]).
pub const SWIRL: f32 = Fx::Rainbow.kind(1);
/// A streak's rainbow edge round a bar at `shape` (centre, half size);
/// its shine as [`BANDS`]' ([`rim_of`]).
pub const RIM: f32 = Fx::Rainbow.kind(2);
/// How many bands fan out with no tier (as the shine collapses).
const NO_BANDS: f32 = 6.0;

/// The rainbow: its bands under the tubes and its swirl round the arrow
/// while the singing shines, the shouts in it.
pub const SPEC: Spec = Spec {
    name: "rainbow",
    ident: "rainbow",
    at: &[Layer::Under, Layer::Arrow],
    draw,
    kinds: &["bands", "swirl", "rim"],
    wgsl: Some(wgsl),
    lyric: Some(Lyric {
        on: Text::Shout,
        strain,
    }),
    reference: Some(REFERENCE),
    ..Spec::NONE
};

fn draw(at: &At, live: &Live, _: &mut Slot, out: &mut Out) {
    let (glow, s) = ((live.glow, live.listed), live.clock);
    let shape = match *at {
        At::Under { head, lane } => bands_of(head, lane, glow, s),
        At::Arrow { at, radius } => swirl_of(at, radius, glow, s),
        _ => None,
    };
    out.shapes.extend(shape);
}

/// The shouts in the rainbow while the singing shines, eased as its glow,
/// as strong as listed (the glow's level and where it came from scaled by
/// it: [`crate::gui::shine::shine_at`] goes with them).
fn strain(live: &Live) -> Option<[f32; 3]> {
    let [level, from, since] = live.glow.triple();
    let s = live.listed;
    (live.glow.shows(live.clock as f32)).then_some([level * s, from * s, since])
}

/// How many bands `s` fans out: those of the tier whose glow is nearest
/// its level (or how strong it showed as it changed, the more; the higher
/// tier's halfway between), [`NO_BANDS`] nearer none. A whole number: no
/// seam where the angle turns.
fn count(s: &Glow) -> f32 {
    let x = s.level.max(s.from);
    let mut n = NO_BANDS;
    let mut below = 0.0;
    for t in &TIERS {
        if x >= (below + t.glow) * 0.5 {
            n = t.bands;
        }
        below = t.glow;
    }
    n
}

fn scale(c: [f32; 3], k: f32) -> [f32; 3] {
    c.map(|v| v * k)
}

/// The bands' light at `p` (pixels), fanning out from `origin` over a lane
/// `size` (half its width and height), `s` at `time`: rays turning round
/// the playhead, more as it is higher, reaching further, the whole lane
/// glowing at the top; collapsing into the playhead, drained, when lost.
pub fn bands(p: (f32, f32), origin: (f32, f32), size: (f32, f32), s: &Glow, time: f32) -> [f32; 3] {
    let (k, c) = shine_at(s, time);
    if k <= SHOWS {
        return [0.0; 3];
    }
    let (dx, dy) = (p.0 - origin.0, p.1 - origin.1);
    let (r, a) = ((dx * dx + dy * dy).sqrt(), dy.atan2(dx));
    let n = count(s);
    let speed = 0.35 + 1.2 * k;
    let w = 0.5 + 0.5 * (n * a - time * speed * TAU).cos();
    let ray = (w * w) * (w * w) * (w * w);
    let reach = (size.0 * (0.1 + 0.6 * k) * (1.0 - c)).max(1.0);
    let fall = (-r / reach).exp();
    let col = mix3(hue(a / TAU + time * 0.12 + r * 0.0015), [0.85; 3], c);
    let lane = 0.16 * held(k, |t| t.lane_light);
    let glow = hue(p.0 * 0.0015 - time * 0.2);
    let fade = 1.0 - c * c;
    let light = scale(col, ray * fall * (0.1 + 0.48 * k));
    [0, 1, 2].map(|i| (light[i] + glow[i] * lane) * fade)
}

/// The swirl's light at `p` round the singer's arrow at `centre`, `radius`
/// out, `s` at `time`: three arms turning, faster as it is higher.
pub fn swirl(p: (f32, f32), centre: (f32, f32), radius: f32, s: &Glow, time: f32) -> [f32; 3] {
    let (k, c) = shine_at(s, time);
    if k <= SHOWS {
        return [0.0; 3];
    }
    let (dx, dy) = (p.0 - centre.0, p.1 - centre.1);
    let (r, a) = ((dx * dx + dy * dy).sqrt(), dy.atan2(dx));
    let radius = radius.max(1.0);
    let off = (r - radius * (1.0 - 0.5 * c)) / (radius * 0.3);
    let ring = (-off * off).exp();
    let spin = 1.2 + 3.2 * k;
    let w = 0.5 + 0.5 * (3.0 * a - time * spin * TAU + r / radius * 2.0).sin();
    let col = mix3(hue(a / TAU + time * 0.25), [0.85; 3], c);
    scale(
        col,
        ring * w * w * w * (0.25 + 0.8 * k) * (k / WHOLE).min(1.0) * (1.0 - c),
    )
}

/// The rainbow edge's light `d` pixels outside a bar `half_y` half high
/// (negative inside), `x` across, `s` at `time`: colours running along it.
pub fn rim(d: f32, x: f32, half_y: f32, s: &Glow, time: f32) -> [f32; 3] {
    let (k, c) = shine_at(s, time);
    if k <= SHOWS {
        return [0.0; 3];
    }
    let w = (half_y * 0.9).max(1.5);
    let e = (-(d / w) * (d / w)).exp();
    let col = mix3(hue(x * 0.012 - time * 0.6), [0.85; 3], c);
    scale(col, e * (0.45 + 0.8 * k) * (k / WHOLE).min(1.0) * (1.0 - c))
}

/// The rainbow edge's light at `p` round a bar at `c`, `half` its size
/// (its corners a little round), `s` at `time` ([`rim`]).
pub fn rim_at(p: (f32, f32), c: (f32, f32), half: (f32, f32), s: &Glow, time: f32) -> [f32; 3] {
    let d = round_box((p.0 - c.0, p.1 - c.1), half, half.0.min(half.1) * 0.3);
    rim(d, p.0, half.1, s, time)
}

/// How much of the bands shows at `p` in a lane at `c`, `half` its size:
/// all of it well inside, fading to none at its edge.
pub fn lane_mask(p: (f32, f32), c: (f32, f32), half: (f32, f32)) -> f32 {
    let d = round_box((p.0 - c.0, p.1 - c.1), half, half.0.min(half.1));
    1.0 - smoothstep(-half.1 * 0.5, 0.0, d)
}

/// Text's colour `col` at `x` (pixels across) in `s`'s rainbow at `time`:
/// the rainbow running along it as strongly as `s` shows (the shader's
/// `lyric_rainbow`, given that strength).
pub fn text_tint(col: [f32; 3], x: f32, s: &Glow, time: f32) -> [f32; 3] {
    let k = shine_at(s, time).0;
    mix3(col, hue(x * 0.01 - time * 0.6), 0.85 * (k / WHOLE).min(1.0))
}

/// The bands fanning out from `origin` over a lane at `centre`, `half`
/// its size, as `s` says, while anything of them shows at `time`.
pub fn bands_of(
    origin: (f32, f32),
    lane: LaneBox,
    (s, strength): (Glow, f32),
    time: f64,
) -> Option<Instance> {
    let (centre, half) = (lane.middle, lane.half);
    s.shows(time as f32).then(|| Instance {
        quad: [
            centre.0 - half.0,
            centre.1 - half.1,
            centre.0 + half.0,
            centre.1 + half.1,
        ],
        shape: [origin.0, origin.1, half.0, half.1],
        uv: [centre.0, centre.1, half.0, half.1],
        params: [BANDS, s.level, s.from, s.since],
        color: dimmed(strength),
        ..Instance::default()
    })
}

/// The swirl round `centre`, `radius` out, as `s` says, while anything of
/// it shows at `time`.
pub fn swirl_of(
    centre: (f32, f32),
    radius: f32,
    (s, strength): (Glow, f32),
    time: f64,
) -> Option<Instance> {
    let m = radius * 1.8 + 2.0;
    s.shows(time as f32).then(|| Instance {
        quad: [centre.0 - m, centre.1 - m, centre.0 + m, centre.1 + m],
        shape: [centre.0, centre.1, radius, 0.0],
        params: [SWIRL, s.level, s.from, s.since],
        color: dimmed(strength),
        ..Instance::default()
    })
}

/// The rainbow edge round a bar at `centre`, `half` its size, as `s` says,
/// while anything of it shows at `time` (the picture's: not a tier's).
pub fn rim_of(
    centre: (f32, f32),
    half: (f32, f32),
    (s, strength): (Glow, f32),
    time: f64,
) -> Option<Instance> {
    let m = (half.1 * 0.9).max(1.5) * 3.0 + 1.0;
    let (mx, my) = (half.0 + m, half.1 + m);
    s.shows(time as f32).then(|| Instance {
        quad: [centre.0 - mx, centre.1 - my, centre.0 + mx, centre.1 + my],
        shape: [centre.0, centre.1, half.0, half.1],
        params: [RIM, s.level, s.from, s.since],
        color: dimmed(strength),
        ..Instance::default()
    })
}

/// How the rainbow's shapes keep how strong they are (`strength`, 0 to 1):
/// how much dimmer than whole, in `color[3]`.
fn dimmed(strength: f32) -> [f32; 4] {
    [0.0, 0.0, 0.0, 1.0 - strength]
}

/// The shine an instance of the bands, a swirl, a rim or text in the
/// rainbow was made with, as the shader reads it.
pub fn shine_of(i: &Instance) -> Option<Glow> {
    let k = i.params[0];
    if [BANDS, SWIRL, RIM].contains(&k) {
        let [_, level, from, since] = i.params;
        return Some(Glow { level, from, since });
    }
    let (fx, [level, from, since]) = Treat::of(i).filter(|_| i.is(Shape::Text))?;
    (fx == Fx::Rainbow).then_some(Glow { level, from, since })
}

/// The rainbow's maths in WGSL, its stage shapes (`rainbow_shape`) and its
/// treatment (`lyric_rainbow`), with the rows' band counts and lane light
/// they read: for the stage's shader (which gives them `hue`, `shine_at`,
/// `held`, `round_box`, `out_rgb` and `TIERS`).
pub fn wgsl() -> String {
    format!(
        "
const NO_BANDS: f32 = {NO_BANDS:?};
const BANDS: array<f32, TIERS> = {};
const LANE_LIGHTS: array<f32, TIERS> = {};
{SHADER}",
        wgsl_array(TIERS.iter().map(|t| t.bands)),
        wgsl_array(TIERS.iter().map(|t| t.lane_light)),
    )
}

const SHADER: &str = r"
// The bands of the tier whose glow is nearest `x` (the higher halfway).
fn band_count(x: f32) -> f32 {
    var n = NO_BANDS;
    var below = 0.0;
    for (var i = 0u; i < TIERS; i = i + 1u) {
        if x >= (below + GLOWS[i]) * 0.5 {
            n = BANDS[i];
        }
        below = GLOWS[i];
    }
    return n;
}

fn rim_at(p: vec2<f32>, c: vec2<f32>, half: vec2<f32>, s: vec3<f32>, time: f32) -> vec3<f32> {
    let d = round_box(p - c, half, min(half.x, half.y) * 0.3);
    return rim(d, p.x, half.y, s, time);
}

fn lane_mask(p: vec2<f32>, c: vec2<f32>, half: vec2<f32>) -> f32 {
    let d = round_box(p - c, half, min(half.x, half.y));
    return 1.0 - smoothstep(-half.y * 0.5, 0.0, d);
}

fn bands(p: vec2<f32>, origin: vec2<f32>, size: vec2<f32>, s: vec3<f32>, time: f32) -> vec3<f32> {
    let sc = shine_at(s, time);
    let k = sc.x;
    let c = sc.y;
    if k <= SHOWS {
        return vec3<f32>(0.0);
    }
    let d = p - origin;
    let r = length(d);
    let a = atan2(d.y, d.x);
    let n = band_count(max(s.x, s.y));
    let speed = 0.35 + 1.2 * k;
    let w = 0.5 + 0.5 * cos(n * a - time * speed * TAU);
    let ray = (w * w) * (w * w) * (w * w);
    let reach = max(size.x * (0.1 + 0.6 * k) * (1.0 - c), 1.0);
    let fall = exp(-r / reach);
    let col = mix(hue(a / TAU + time * 0.12 + r * 0.0015), vec3<f32>(0.85), c);
    let lane = 0.16 * held(k, LANE_LIGHTS);
    let glow = hue(p.x * 0.0015 - time * 0.2);
    let fade = 1.0 - c * c;
    return (col * ray * fall * (0.1 + 0.48 * k) + glow * lane) * fade;
}

fn swirl(p: vec2<f32>, centre: vec2<f32>, radius0: f32, s: vec3<f32>, time: f32) -> vec3<f32> {
    let sc = shine_at(s, time);
    let k = sc.x;
    let c = sc.y;
    if k <= SHOWS {
        return vec3<f32>(0.0);
    }
    let d = p - centre;
    let r = length(d);
    let a = atan2(d.y, d.x);
    let radius = max(radius0, 1.0);
    let off = (r - radius * (1.0 - 0.5 * c)) / (radius * 0.3);
    let ring = exp(-off * off);
    let spin = 1.2 + 3.2 * k;
    let w = 0.5 + 0.5 * sin(3.0 * a - time * spin * TAU + r / radius * 2.0);
    let col = mix(hue(a / TAU + time * 0.25), vec3<f32>(0.85), c);
    return col * ring * w * w * w * (0.25 + 0.8 * k) * min(k / WHOLE, 1.0) * (1.0 - c);
}

fn rim(d: f32, x: f32, half_y: f32, s: vec3<f32>, time: f32) -> vec3<f32> {
    let sc = shine_at(s, time);
    let k = sc.x;
    let c = sc.y;
    if k <= SHOWS {
        return vec3<f32>(0.0);
    }
    let w = max(half_y * 0.9, 1.5);
    let e = exp(-(d / w) * (d / w));
    let col = mix(hue(x * 0.012 - time * 0.6), vec3<f32>(0.85), c);
    return col * e * (0.45 + 0.8 * k) * min(k / WHOLE, 1.0) * (1.0 - c);
}

// The rainbow's shapes: its bands (within the lane, softly), a swirl, a
// rainbow edge; light, added. `params.yzw` the shine, `color.a` how much
// dimmer than whole.
fn rainbow_shape(o: Out, p: vec2<f32>, k: u32) -> vec4<f32> {
    let s = o.params.yzw;
    let strength = 1.0 - o.color.a;
    if k == K_BANDS {
        let inside = lane_mask(p, o.uv.xy, o.uv.zw);
        let l = bands(p, o.shape.xy, o.shape.zw, s, screen.time);
        return vec4<f32>(out_rgb(l) * inside * strength, 0.0);
    }
    if k == K_SWIRL {
        return vec4<f32>(out_rgb(swirl(p, o.shape.xy, o.shape.z, s, screen.time)) * strength, 0.0);
    }
    return vec4<f32>(out_rgb(rim_at(p, o.shape.xy, o.shape.zw, s, screen.time)) * strength, 0.0);
}

// Text in the rainbow, running along it as strongly as it shines.
fn lyric_rainbow(col: vec3<f32>, a: f32, px: vec2<f32>, box: vec4<f32>, time: f32, strength: f32, look: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(mix(col, hue(px.x * 0.01 - time * 0.6), 0.85 * min(strength / WHOLE, 1.0)), a);
}
";

/// The shine's maths at points across the lane, round the arrow and along
/// a bar, at levels coming in, held, eased down a level and lost, through
/// its rise and collapse: the shader's `shine_at`, [`lane_mask`], `hue`,
/// [`bands`], [`swirl`], [`rim`], [`rim_at`] and the text's tint
/// ([`text_tint`], `lyric_rainbow`) as the processor's.
pub const REFERENCE: Reference = Reference {
    samples,
    ins: 4,
    outs: 7,
    body: "
    let a = ins[i * 4u];
    let b = ins[i * 4u + 1u];
    let c = ins[i * 4u + 2u];
    let h = ins[i * 4u + 3u];
    let s = a.xyz;
    let bar = vec2<f32>(c.z * 3.0, c.z * 0.2);
    outs[i * 7u] = vec4<f32>(shine_at(s, a.w), lane_mask(b.xy, b.zw, c.xy), 0.0);
    outs[i * 7u + 1u] = vec4<f32>(hue(h.x), 0.0);
    outs[i * 7u + 2u] = vec4<f32>(bands(b.xy, b.zw, c.xy, s, a.w), 0.0);
    outs[i * 7u + 3u] = vec4<f32>(swirl(b.xy, b.zw, c.z, s, a.w), 0.0);
    outs[i * 7u + 4u] = vec4<f32>(rim(c.w, b.x, c.z * 0.2, s, a.w), 0.0);
    outs[i * 7u + 5u] = vec4<f32>(rim_at(b.xy, b.zw, bar, s, a.w), 0.0);
    let k = shine_at(s, a.w).x;
    outs[i * 7u + 6u] = vec4<f32>(lyric_rainbow(h.yzw, 1.0, b.xy, vec4<f32>(0.0), a.w, k, vec4<f32>(0.0)).xyz, 0.0);",
    cpu,
    near: (2e-3, 2e-3),
    shows: |outs| outs[2].iter().sum::<f32>() > 0.01,
    least: 100,
};

fn samples() -> Vec<[f32; 4]> {
    let shines = [
        (0.25, 0.0),
        (0.5, 0.25),
        (0.75, 0.5),
        (0.75, 0.75),
        (0.5, 0.75),
        (0.25, 0.6),
        (0.0, 0.25),
        (0.0, 0.75),
    ];
    (0..480)
        .flat_map(|k| {
            let f = k as f32;
            let (level, from) = shines[k % shines.len()];
            let since = 5.0;
            let time = since + (f * 0.37).fract() * 0.9 - 0.05;
            let p = (
                100.0 + (f * 0.61).fract() * 700.0,
                60.0 + (f * 0.29).fract() * 200.0,
            );
            let d = (f * 0.43).fract() * 12.0 - 4.0;
            let tint = [(f * 0.11).fract(), (f * 0.23).fract(), (f * 0.31).fract()];
            [
                [level, from, since, time],
                [p.0, p.1, 300.0, 160.0],
                [700.0, 90.0, 24.0, d],
                [(f * 0.17).fract() * 3.0 - 1.0, tint[0], tint[1], tint[2]],
            ]
        })
        .collect()
}

fn cpu(s: &[[f32; 4]]) -> Vec<[f32; 4]> {
    let [a, b, c, h] = [s[0], s[1], s[2], s[3]];
    let g = Glow {
        level: a[0],
        from: a[1],
        since: a[2],
    };
    let (t, p, o) = (a[3], (b[0], b[1]), (b[2], b[3]));
    let v = |c: [f32; 3]| [c[0], c[1], c[2], 0.0];
    let (strength, collapse) = shine_at(&g, t);
    vec![
        [strength, collapse, lane_mask(p, o, (c[0], c[1])), 0.0],
        v(hue(h[0])),
        v(bands(p, o, (c[0], c[1]), &g, t)),
        v(swirl(p, o, c[2], &g, t)),
        v(rim(c[3], b[0], c[2] * 0.2, &g, t)),
        v(rim_at(p, o, (c[2] * 3.0, c[2] * 0.2), &g, t)),
        v(text_tint([h[1], h[2], h[3]], b[0], &g, t)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shine(level: f32, from: f32) -> Glow {
        Glow {
            level,
            from,
            since: 10.0,
        }
    }

    /// The first tier's shine a soft shimmer, Amazing's the whole lane
    /// alight: more light the higher, more bands round the playhead,
    /// reaching further, turning faster; none at 0.
    #[test]
    fn the_bands_grow_with_the_level() {
        let origin = (300.0, 200.0);
        let size = (800.0, 120.0);
        let light = |level: f32, p: (f32, f32), t: f32| {
            let c = bands(p, origin, size, &shine(level, level), 10.0 + t);
            c[0] + c[1] + c[2]
        };
        // Round the playhead, at a radius: the total light, the peaks.
        let ring = |level: f32, radius: f32| {
            let points = (0..720).map(|i| {
                let a = i as f32 / 720.0 * TAU;
                (origin.0 + radius * a.cos(), origin.1 + radius * a.sin())
            });
            let lights: Vec<f32> = points.map(|p| light(level, p, 1.0)).collect();
            let sum: f32 = lights.iter().sum();
            let peaks = lights
                .iter()
                .enumerate()
                .filter(|&(i, &v)| v > lights[(i + 719) % 720] && v >= lights[(i + 1) % 720])
                .filter(|&(_, &v)| v > 0.01)
                .count();
            (sum, peaks)
        };
        assert_eq!(ring(0.0, 60.0).0, 0.0);
        let (one, three) = (ring(0.25, 60.0), ring(0.75, 60.0));
        assert!(three.0 > 2.0 * one.0, "{one:?} {three:?}");
        assert!(three.1 > one.1, "more bands: {one:?} {three:?}");
        // Further out at the top, relatively.
        let reach = |level: f32| ring(level, 300.0).0 / ring(level, 60.0).0;
        assert!(reach(0.75) > reach(0.25));
        // The top lights the whole lane; the first not.
        let far = (origin.0 + 700.0, origin.1 + 50.0);
        assert!(light(0.75, far, 1.0) > 0.1 && light(0.25, far, 1.0) < 0.02);
        // They turn: the light at a point changes, faster at the top.
        let p = (origin.0 + 60.0, origin.1);
        let change = |level: f32| (light(level, p, 1.0) - light(level, p, 1.03)).abs();
        assert!(change(0.25) > 0.0 && change(0.75) > 0.0);
    }

    /// Lost, the shine collapses into the playhead (reaching less, drained
    /// of colour) and is gone; the swirl and the rim with it.
    #[test]
    fn a_lost_streak_collapses_into_the_playhead() {
        let lost = shine(0.0, 0.75);
        let origin = (300.0, 200.0);
        let at = |p: (f32, f32), t: f32| {
            let c = bands(p, origin, (800.0, 120.0), &lost, 10.0 + t);
            c[0] + c[1] + c[2]
        };
        let near = (origin.0 + 20.0, origin.1);
        let far = (origin.0 + 200.0, origin.1);
        let share = |t: f32| {
            let ring = |r: f32| -> f32 {
                (0..360)
                    .map(|i| {
                        let a = i as f32 / 360.0 * TAU;
                        at((origin.0 + r * a.cos(), origin.1 + r * a.sin()), t)
                    })
                    .sum()
            };
            ring(200.0) / ring(20.0).max(1e-9)
        };
        assert!(share(0.3) < share(0.05), "pulled in");
        assert!(at(near, 0.45) < at(near, 0.05) && at(far, 0.45) < at(far, 0.05));
        assert!(at(near, crate::gui::shine::COLLAPSE + 0.01) == 0.0, "gone");
        // Drained: the colour greyer as it goes.
        let spread = |c: [f32; 3]| {
            let (lo, hi) = (c[0].min(c[1]).min(c[2]), c[0].max(c[1]).max(c[2]));
            (hi - lo) / hi.max(1e-9)
        };
        let colour = |t: f32| {
            (0..90)
                .map(|i| {
                    bands(
                        near,
                        origin,
                        (800.0, 120.0),
                        &lost,
                        10.0 + t + i as f32 * 1e-3,
                    )
                })
                .fold([0.0; 3], |m, c| [m[0] + c[0], m[1] + c[1], m[2] + c[2]])
        };
        assert!(spread(colour(0.4)) < spread(colour(0.02)));
        let gone = 10.0 + crate::gui::shine::COLLAPSE;
        assert_eq!(swirl(near, origin, 30.0, &lost, gone), [0.0; 3]);
        assert_eq!(rim(0.0, 10.0, 3.0, &lost, gone), [0.0; 3]);
    }

    /// Down a level (Amazing's shine to Great's) the bands, swirl and rim
    /// all still there.
    #[test]
    fn a_drop_of_a_level_still_shines() {
        let drop = shine(0.5, 0.75);
        let sum = |c: [f32; 3]| c.iter().sum::<f32>();
        let lit = (0..40).map(|i| {
            let t = 12.0 + i as f32 * 0.013;
            sum(bands(
                (360.0, 200.0),
                (300.0, 200.0),
                (800.0, 120.0),
                &drop,
                t,
            ))
        });
        assert!(lit.fold(0.0, f32::max) > 0.01);
        let round = (0..36).map(|i| {
            let a = i as f32 / 36.0 * TAU;
            let p = (300.0 + 30.0 * a.cos(), 200.0 + 30.0 * a.sin());
            sum(swirl(p, (300.0, 200.0), 30.0, &drop, 12.0))
        });
        assert!(round.fold(0.0, f32::max) > 0.01);
        assert!(sum(rim(0.0, 10.0, 3.0, &drop, 12.0)) > 0.1);
    }

    /// Text in the rainbow knows it; text of no treatment, and shapes of
    /// other kinds, not.
    #[test]
    fn the_shine_an_instance_was_made_with_is_read_back() {
        let g = shine(0.5, 0.25);
        let mut text = Instance {
            params: [Shape::Text.code(), 0.0, 0.0, 0.0],
            ..Instance::default()
        };
        assert_eq!(shine_of(&text), None);
        Treat {
            code: Fx::Rainbow.treatment_code(),
            strain: g.triple(),
        }
        .apply(&mut text);
        assert_eq!(shine_of(&text), Some(g));
        let lane = LaneBox {
            middle: (3.0, 4.0),
            half: (5.0, 6.0),
        };
        let bands = bands_of((1.0, 2.0), lane, (g, 1.0), 10.5).unwrap();
        assert_eq!(shine_of(&bands), Some(g));
        assert_eq!(shine_of(&Instance::default()), None);
    }
}
