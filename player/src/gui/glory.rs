//! The shine's looks on the GPU, in the stage's shader: the meter (a
//! glass capsule its fill springs to, coloured by the tier it reaches,
//! the tiers' marks along it lighting up as it passes them), the ring of
//! the time left round the countdown (draining as the song goes, in the
//! rainbow while the singing shines), the marks of the phrases along the
//! progress bar (each popping as its tier is decided), and the tiers'
//! light, drawn where it blooms ([`super::bloom`]): the glow round the
//! lane and the flames blazing off its edges, as strong as the tiers' rows
//! hold them.
//!
//! The processor gives each a few numbers (where, its fill and when it
//! changed, the tier's shine); everything that moves moves by the time in
//! the shader. These functions are the shader's reference ([`wgsl`]), held
//! to it on the GPU (`super::shader_check`). The meter's marks and fills
//! come from the tier table ([`crate::tiers::TIERS`]), the shader's
//! written from the same arrays ([`MARKS`], [`FILLS`]).

use super::rainbow::{hue, shine_at, Glow, WHOLE};
use super::sparks::noise;
use crate::core::shine::Tier;
use crate::tiers::{wgsl_array, Paint, TIERS};

/// Where along the meter each tier's mark is, Good first: evenly,
/// whatever the shine each takes, the top's at its end and the stretch
/// below the first two steps long.
pub const MARKS: [f32; Tier::COUNT] = {
    let mut marks = [0.0; Tier::COUNT];
    let mut i = 0;
    while i < Tier::COUNT {
        marks[i] = (i + 2) as f32 / (Tier::COUNT + 1) as f32;
        i += 1;
    }
    marks
};
/// How far below its mark the fill takes on a tier's colour: all of it
/// at the mark.
const EDGE: f32 = 0.02;
/// The fill below the first mark: cool.
const COOL: [f32; 3] = [0.35, 0.6, 1.0];

/// Each tier's bar as a share of the top's ([`Tier::bar`]), Good first.
fn bars() -> [f32; Tier::COUNT] {
    Tier::ALL.map(|t| (t.bar() / Tier::TOP.bar()) as f32)
}

/// Each tier's fill ([`crate::tiers::TierSpec::fill`]) as the shader
/// takes it, Good first: its colour, and last 1 for the rainbow (its
/// colour then unused), else 0.
pub const FILLS: [[f32; 4]; Tier::COUNT] = {
    let mut fills = [[0.0; 4]; Tier::COUNT];
    let mut i = 0;
    while i < Tier::COUNT {
        fills[i] = match TIERS[i].fill {
            Paint::Rgb([r, g, b]) => [r, g, b, 0.0],
            Paint::Rainbow => [0.0, 0.0, 0.0, 1.0],
        };
        i += 1;
    }
    fills
};

fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = clamp01((x - e0) / (e1 - e0));
    t * t * (3.0 - 2.0 * t)
}

fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t)
}

/// Where along the meter a shine of `of_top` (its share of its bar for
/// the top tier) shows: through the tiers' marks ([`MARKS`]) at their bars
/// (a share of the top's, [`bars`]), linear between, full at the top.
pub fn position(of_top: f32) -> f32 {
    let mut from = (0.0, 0.0);
    for (b, m) in bars().into_iter().zip(MARKS) {
        if of_top <= b {
            return from.1 + (m - from.1) * clamp01((of_top - from.0) / (b - from.0));
        }
        from = (b, m);
    }
    1.0
}

/// The meter's fill at `time`, springing from `from` to `to` since
/// `since`: over it a little, back, settled.
pub fn meter_fill(from: f32, to: f32, since: f32, time: f32) -> f32 {
    let t = (time - since).max(0.0);
    to + (from - to) * (-5.0 * t).exp() * (11.0 * t).cos()
}

/// The colour of the meter's fill at `u` along it (0 to 1), the fill at
/// `level`, at `time`: all of it in the tier the level reaches (cool below
/// the first, then each tier's fill, its colour or the rainbow running
/// along it, from [`EDGE`] below its mark), deeper at its start.
pub fn fill_color(level: f32, u: f32, time: f32) -> [f32; 3] {
    let rainbow = hue(u * 1.5 - time * 0.4);
    let mut c = COOL;
    for (m, [r, g, b, bow]) in MARKS.into_iter().zip(FILLS) {
        let fill = if bow > 0.5 { rainbow } else { [r, g, b] };
        c = mix3(c, fill, smoothstep(m - EDGE, m, level));
    }
    let deep = 0.7 + 0.3 * clamp01(u / level.max(0.05));
    c.map(|v| v * deep)
}

/// The meter at `u` along it (0 to 1) and `v` across (-1 its top, 1 its
/// bottom), filled to `fill`, at `time`: its colour and how much it covers
/// (inside the capsule; the shader cuts its edge).
pub fn meter_at(u: f32, v: f32, fill: f32, time: f32) -> [f32; 4] {
    // The glass: dark, lighter at its top.
    let glass = 0.06 + 0.1 * clamp01(-v);
    let mut rgb = [glass, glass, glass * 1.6];
    let mut a = 0.55_f32;
    let filled = 1.0 - smoothstep(fill - 0.004, fill + 0.004, u);
    if filled > 0.0 {
        let c = fill_color(fill, u, time);
        // Lit as glass: its top bright, a sheen running along it.
        let shade = 1.15 - 0.35 * clamp01(v * 0.5 + 0.5) + 0.35 * clamp01(-v - 0.3);
        let at = (time * 0.6).fract() * 1.6 - 0.3;
        let sheen = (-((u - at) * 9.0).powi(2)).exp() * 0.45;
        let lit = mix3(c.map(|x| x * shade), [1.0; 3], sheen);
        rgb = mix3(rgb, lit, filled);
        a = a + (1.0 - a) * filled;
    }
    // The front: a bright edge where the fill ends.
    let front = (-((u - fill) * 60.0).powi(2)).exp() * clamp01(fill * 8.0);
    rgb = mix3(rgb, [1.0; 3], front * 0.8);
    // The tiers' marks: thin lines, bright once passed.
    for m in MARKS {
        let line = (-((u - m) * 160.0).powi(2)).exp() * (1.0 - v.abs() * 0.3);
        let passed = if fill >= m - 0.002 { 1.0 } else { 0.35 };
        rgb = mix3(rgb, [1.0; 3], line * passed * 0.9);
        a = a.max(line * passed);
    }
    [rgb[0], rgb[1], rgb[2], a]
}

/// The ring of the time left at `p` (pixels), round `c`, `radius` to its
/// middle, `thick` wide, the share `left` of the song still to come, the
/// singing's `shine` at `time`: its colour (straight) and how much it
/// covers. Its track faint; its arc from the top clockwise, gold to coral
/// (in the rainbow, turning, as strongly as it shines), its head bright.
pub fn ring_at(
    p: (f32, f32),
    c: (f32, f32),
    (radius, thick): (f32, f32),
    left: f32,
    shine: &Glow,
    time: f32,
) -> [f32; 4] {
    let (dx, dy) = (p.0 - c.0, p.1 - c.1);
    let r = (dx * dx + dy * dy).sqrt();
    // Round from the top, clockwise, 0 to 1.
    let u = (dx.atan2(-dy) / std::f32::consts::TAU + 1.0).fract();
    let band = clamp01(0.5 - ((r - radius).abs() - thick * 0.5));
    let on = 1.0 - smoothstep(left - 0.003, left + 0.003, u);
    let (k, _) = shine_at(shine, time);
    let warm = mix3([1.0, 0.82, 0.3], [1.0, 0.42, 0.38], u);
    let bow = hue(u - time * 0.25);
    let arc = mix3(warm, bow, clamp01(k / WHOLE));
    let track = [1.0, 1.0, 1.0];
    let rgb = mix3(track, arc, on);
    let a = band * (0.14 + 0.81 * on);
    // Its head: a bright point where the arc ends, glowing.
    let hx = c.0 + radius * (left * std::f32::consts::TAU).sin();
    let hy = c.1 - radius * (left * std::f32::consts::TAU).cos();
    let hd = ((p.0 - hx).powi(2) + (p.1 - hy).powi(2)).sqrt();
    let head = (-(hd / (thick * 0.9)).powi(2)).exp() * smoothstep(0.0, 0.01, left);
    let rgb = mix3(rgb, [1.0; 3], head);
    [rgb[0], rgb[1], rgb[2], a.max(head)]
}

/// A phrase's mark on the progress bar `age` seconds after its tier was
/// decided: how much larger it is (popping out and back) and how bright
/// (a flash fading).
pub fn mark_pop(age: f32) -> (f32, f32) {
    if age < 0.0 {
        return (1.0, 0.0);
    }
    (
        1.0 + 1.2 * (-7.0 * age).exp() * (14.0 * age).cos().max(-0.3),
        (-5.0 * age).exp(),
    )
}

/// A tier's name over the meter (and its badge) `age` seconds after its
/// phrase was decided, showing `life` seconds: how much larger it is
/// (popping out and back) and how much of it shows (fading out over its
/// last [`LABEL_FADE`]); none before or after.
pub fn label_pop(age: f32, life: f32) -> (f32, f32) {
    if age < 0.0 || age >= life {
        return (1.0, 0.0);
    }
    (
        1.0 + 0.6 * (-age * 9.0).exp() * (age * 18.0).cos(),
        1.0 - clamp01((age - (life - LABEL_FADE)) / LABEL_FADE),
    )
}

/// How long a tier's name fades out, in seconds.
const LABEL_FADE: f32 = 0.4;

/// How bright the glow round the lane is `d` pixels outside it (negative
/// inside), `wide` its reach, at `time`, `k` its strength: soft, breathing.
pub fn glow_at(d: f32, wide: f32, time: f32, k: f32) -> f32 {
    let t = d.max(0.0) / wide.max(1.0);
    let inside = (d.min(0.0) / (wide * 0.5).max(1.0)).exp();
    (-t * t * 2.5).exp() * inside * (0.78 + 0.22 * (time * 2.6).sin()) * k
}

/// A smooth number that looks random over the plane, 0 to 1: value noise
/// on a lattice of [`noise`].
pub(super) fn vnoise(x: f32, y: f32) -> f32 {
    let (ix, iy) = (x.floor(), y.floor());
    let (fx, fy) = (x - ix, y - iy);
    let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
    let at = |i: f32, j: f32| {
        let (i, j) = (i as i32 as u32, j as i32 as u32);
        noise(i.wrapping_mul(73_856_093) ^ j.wrapping_mul(19_349_663))
    };
    let top = at(ix, iy) + (at(ix + 1.0, iy) - at(ix, iy)) * sx;
    let bottom = at(ix, iy + 1.0) + (at(ix + 1.0, iy + 1.0) - at(ix, iy + 1.0)) * sx;
    top + (bottom - top) * sy
}

/// The flames `d` pixels off the lane's edge (outwards; negative:
/// into the lane) at `x` along it, `high` their height, at `time`, `k`
/// their strength: their light (bright, for the bloom), licking up and
/// flickering, white at their root to red at their tips.
pub fn blaze(x: f32, d: f32, high: f32, time: f32, k: f32) -> [f32; 3] {
    let h = high.max(1.0);
    let n = 0.65 * vnoise(x / h * 4.0, d / h * 1.2 + time * 2.6)
        + 0.35 * vnoise(x / h * 9.0 + 7.0, d / h * 2.4 + time * 4.1);
    let reach = h * (0.08 + 1.4 * n * n);
    let flame = if d >= 0.0 {
        clamp01(1.0 - d / reach).powf(1.3)
    } else {
        0.5 * (d / (h * 0.06)).exp()
    };
    let heat = flame * (0.45 + 0.85 * n);
    let red = [1.0, 0.18, 0.04];
    let orange = [1.0, 0.55, 0.1];
    let yellow = [1.0, 0.9, 0.45];
    let c = mix3(red, orange, smoothstep(0.15, 0.5, heat));
    let c = mix3(c, yellow, smoothstep(0.5, 0.85, heat));
    let c = mix3(c, [1.0; 3], smoothstep(0.85, 1.1, heat));
    c.map(|v| v * heat * 1.1 * k)
}

/// The functions above in WGSL, and the stage's shapes drawn with them:
/// for the stage's shader (which gives them `hue`, `shine_at`, `glow`,
/// `round_box`, `over_plus` and `TIERS`). The tiers' marks and fills are
/// written in from [`MARKS`] and [`FILLS`].
pub fn wgsl() -> String {
    let marks = wgsl_array(MARKS);
    let fills: Vec<String> = FILLS
        .iter()
        .map(|[r, g, b, a]| format!("vec4<f32>({r:?}, {g:?}, {b:?}, {a:?})"))
        .collect();
    format!(
        "const MARKS: array<f32, TIERS> = {marks};
const FILLS: array<vec4<f32>, TIERS> = array<vec4<f32>, TIERS>({});
const COOL: vec3<f32> = vec3<f32>({:?}, {:?}, {:?});
const EDGE: f32 = {EDGE:?};
const LABEL_FADE: f32 = {LABEL_FADE:?};
{SHADER}",
        fills.join(", "),
        COOL[0],
        COOL[1],
        COOL[2]
    )
}

const SHADER: &str = r"
fn meter_fill(a: f32, b: f32, since: f32, time: f32) -> f32 {
    let t = max(time - since, 0.0);
    return b + (a - b) * exp(-5.0 * t) * cos(11.0 * t);
}

fn fill_color(level: f32, u: f32, time: f32) -> vec3<f32> {
    let rainbow = hue(u * 1.5 - time * 0.4);
    var c = COOL;
    for (var i = 0u; i < TIERS; i = i + 1u) {
        let f = FILLS[i];
        var fill = f.rgb;
        if f.w > 0.5 {
            fill = rainbow;
        }
        c = mix(c, fill, smoothstep(MARKS[i] - EDGE, MARKS[i], level));
    }
    return c * (0.7 + 0.3 * clamp(u / max(level, 0.05), 0.0, 1.0));
}

fn meter_at(u: f32, v: f32, fill: f32, time: f32) -> vec4<f32> {
    let glass = 0.06 + 0.1 * clamp(-v, 0.0, 1.0);
    var rgb = vec3<f32>(glass, glass, glass * 1.6);
    var a = 0.55;
    let filled = 1.0 - smoothstep(fill - 0.004, fill + 0.004, u);
    if filled > 0.0 {
        let c = fill_color(fill, u, time);
        let shade = 1.15 - 0.35 * clamp(v * 0.5 + 0.5, 0.0, 1.0) + 0.35 * clamp(-v - 0.3, 0.0, 1.0);
        let at = fract(time * 0.6) * 1.6 - 0.3;
        let sheen = exp(-pow((u - at) * 9.0, 2.0)) * 0.45;
        let lit = mix(c * shade, vec3<f32>(1.0), sheen);
        rgb = mix(rgb, lit, filled);
        a = a + (1.0 - a) * filled;
    }
    let front = exp(-pow((u - fill) * 60.0, 2.0)) * clamp(fill * 8.0, 0.0, 1.0);
    rgb = mix(rgb, vec3<f32>(1.0), front * 0.8);
    for (var i = 0u; i < TIERS; i = i + 1u) {
        let m = MARKS[i];
        let line = exp(-pow((u - m) * 160.0, 2.0)) * (1.0 - abs(v) * 0.3);
        var passed = 0.35;
        if fill >= m - 0.002 {
            passed = 1.0;
        }
        rgb = mix(rgb, vec3<f32>(1.0), line * passed * 0.9);
        a = max(a, line * passed);
    }
    return vec4<f32>(rgb, a);
}

fn ring_at(p: vec2<f32>, c: vec2<f32>, size: vec2<f32>, left: f32, s: vec3<f32>, time: f32) -> vec4<f32> {
    let radius = size.x;
    let thick = size.y;
    let dp = p - c;
    let r = length(dp);
    let u = fract(atan2(dp.x, -dp.y) / TAU + 1.0);
    let band = clamp(0.5 - (abs(r - radius) - thick * 0.5), 0.0, 1.0);
    let on = 1.0 - smoothstep(left - 0.003, left + 0.003, u);
    let k = shine_at(s, time).x;
    let warm = mix(vec3<f32>(1.0, 0.82, 0.3), vec3<f32>(1.0, 0.42, 0.38), u);
    let arc = mix(warm, hue(u - time * 0.25), clamp(k / WHOLE, 0.0, 1.0));
    var rgb = mix(vec3<f32>(1.0), arc, on);
    let a = band * (0.14 + 0.81 * on);
    let h = c + radius * vec2<f32>(sin(left * TAU), -cos(left * TAU));
    let head = exp(-pow(length(p - h) / (thick * 0.9), 2.0)) * smoothstep(0.0, 0.01, left);
    rgb = mix(rgb, vec3<f32>(1.0), head);
    return vec4<f32>(rgb, max(a, head));
}

fn mark_pop(age: f32) -> vec2<f32> {
    if age < 0.0 {
        return vec2<f32>(1.0, 0.0);
    }
    return vec2<f32>(1.0 + 1.2 * exp(-7.0 * age) * max(cos(14.0 * age), -0.3), exp(-5.0 * age));
}

fn label_pop(age: f32, life: f32) -> vec2<f32> {
    if age < 0.0 || age >= life {
        return vec2<f32>(1.0, 0.0);
    }
    return vec2<f32>(
        1.0 + 0.6 * exp(-age * 9.0) * cos(age * 18.0),
        1.0 - clamp((age - (life - LABEL_FADE)) / LABEL_FADE, 0.0, 1.0),
    );
}

fn glow_at(d: f32, wide: f32, time: f32, k: f32) -> f32 {
    let t = max(d, 0.0) / max(wide, 1.0);
    let inside = exp(min(d, 0.0) / max(wide * 0.5, 1.0));
    return exp(-t * t * 2.5) * inside * (0.78 + 0.22 * sin(time * 2.6)) * k;
}

fn lattice(i: f32, j: f32) -> f32 {
    let a = bitcast<u32>(i32(i)) * 73856093u;
    let b = bitcast<u32>(i32(j)) * 19349663u;
    var x = ((a ^ b) * 0x9E3779B9u) ^ 0x85EBCA6Bu;
    x = x ^ (x >> 15u);
    x = x * 0x2C1B3C6Du;
    x = x ^ (x >> 12u);
    x = x * 0x297A2D39u;
    x = x ^ (x >> 15u);
    return f32(x >> 8u) / 16777216.0;
}

fn vnoise(x: f32, y: f32) -> f32 {
    let ix = floor(x);
    let iy = floor(y);
    let fx = x - ix;
    let fy = y - iy;
    let sx = fx * fx * (3.0 - 2.0 * fx);
    let sy = fy * fy * (3.0 - 2.0 * fy);
    let top = lattice(ix, iy) + (lattice(ix + 1.0, iy) - lattice(ix, iy)) * sx;
    let bottom = lattice(ix, iy + 1.0) + (lattice(ix + 1.0, iy + 1.0) - lattice(ix, iy + 1.0)) * sx;
    return top + (bottom - top) * sy;
}

fn blaze(x: f32, d: f32, high: f32, time: f32, k: f32) -> vec3<f32> {
    let h = max(high, 1.0);
    let n = 0.65 * vnoise(x / h * 4.0, d / h * 1.2 + time * 2.6)
        + 0.35 * vnoise(x / h * 9.0 + 7.0, d / h * 2.4 + time * 4.1);
    let reach = h * (0.08 + 1.4 * n * n);
    var flame = 0.5 * exp(d / (h * 0.06));
    if d >= 0.0 {
        flame = pow(clamp(1.0 - d / reach, 0.0, 1.0), 1.3);
    }
    let heat = flame * (0.45 + 0.85 * n);
    var c = mix(vec3<f32>(1.0, 0.18, 0.04), vec3<f32>(1.0, 0.55, 0.1), smoothstep(0.15, 0.5, heat));
    c = mix(c, vec3<f32>(1.0, 0.9, 0.45), smoothstep(0.5, 0.85, heat));
    c = mix(c, vec3<f32>(1.0), smoothstep(0.85, 1.1, heat));
    return c * heat * 1.1 * k;
}

// The shine's shapes: the meter (`shape` its middle and half size,
// `params.yzw` its fill from, to and since), the ring (`shape.xy` its
// middle, `shape.zw` its radius and thickness, `params.y` the share left,
// `strain.xyz` the shine), a phrase's mark (a disc at `shape.xy`,
// `shape.z` across, in `color`, `params.y` when its tier came), the
// glow round a lane (`shape`, in `color`, `params.y` its reach, `params.z`
// its strength) and the flames (along `shape.x` to `shape.z` at
// `shape.y`, outward `shape.w` (1 down, -1 up), `params.y` high,
// `params.z` strong).
fn glory(o: Out, p: vec2<f32>, k: u32) -> vec4<f32> {
    if k == K_METER {
        let c = o.shape.xy;
        let half = o.shape.zw;
        let d = round_box(p - c, half, half.y);
        let cover = clamp(0.5 - d, 0.0, 1.0);
        let u = (p.x - (c.x - half.x)) / max(2.0 * half.x, 1.0);
        let v = (p.y - c.y) / max(half.y, 1.0);
        let fill = meter_fill(o.params.y, o.params.z, o.params.w, screen.time);
        let m = meter_at(u, v, fill, screen.time);
        let g = glow(d, half.y * 1.2) * clamp(fill, 0.0, 1.0) * 0.5 * (1.0 - cover);
        return over_plus(m.rgb, cover * m.a * o.color.a, fill_color(fill, fill, screen.time) * g * o.color.a);
    }
    if k == K_RING {
        let m = ring_at(p, o.shape.xy, o.shape.zw, o.params.y, o.strain.xyz, screen.time);
        return over_plus(m.rgb, m.a * o.color.a, vec3<f32>(0.0));
    }
    if k == K_MARK {
        let pop = mark_pop(screen.time - o.params.y);
        let r = o.shape.z * pop.x;
        let d = length(p - o.shape.xy) - r;
        let cover = clamp(0.5 - d, 0.0, 1.0);
        let col = mix(o.color.rgb, vec3<f32>(1.0), pop.y * 0.7);
        let g = glow(d, r * (2.0 + 3.0 * pop.y)) * (0.35 + 0.65 * pop.y);
        return over_plus(col, cover * o.color.a, o.color.rgb * g * o.color.a);
    }
    if k == K_GLOW {
        let d = round_box(p - o.shape.xy, o.shape.zw, o.shape.w);
        let l = glow_at(d, o.params.y, screen.time, o.params.z);
        return vec4<f32>(out_rgb(o.color.rgb) * l, 0.0);
    }
    // The flames.
    let d = (p.y - o.shape.y) * o.shape.w;
    let along = smoothstep(o.shape.x, o.shape.x + o.params.y, p.x)
        * (1.0 - smoothstep(o.shape.z - o.params.y, o.shape.z, p.x));
    let l = blaze(p.x, d, o.params.y, screen.time, o.params.z) * along;
    return vec4<f32>(out_rgb(l), 0.0);
}
";
