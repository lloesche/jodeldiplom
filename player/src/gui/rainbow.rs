//! A streak's shine on the GPU: while the singing shines (a [`Glow`]
//! level above 0: the glow of the tier the last phrase reached, 0 to 1,
//! [`crate::tiers::TierSpec::glow`]), rainbow bands fan out from the
//! playhead behind the pills, as many as the tier's row has, brighter and
//! faster the higher, lighting the whole lane as much as the row says
//! ([`bands`]); a swirl turns around the singer's arrow ([`swirl`]); in
//! the picture, a rainbow edge on the bar under the word sung ([`rim`]);
//! the multiplier's number in rainbow. Each step up throws a burst from
//! the playhead: stars, notes and rainbow bits that spin, arc down and fade
//! ([`step_up`]). Lost, the bands collapse into the playhead in
//! [`COLLAPSE`] and the colour drains ([`shine_at`]).
//!
//! The processor gives only the level (when it changed, and from what)
//! and a record a step up; everything moves by the time in the shaders.
//! These functions are their reference ([`stage_wgsl`],
//! [`particles_wgsl`]), held to them on the GPU (`super::shader_check`).

use super::particles::{Look, Spray, Sprite};
use super::sparks::{rand, TAU};
use super::stage_gpu::{flags, kind, Instance};
use crate::core::shine::Tier;
use crate::tiers::{held, TIERS};

/// How long a level takes to come in, in seconds.
pub const RISE: f32 = 0.25;
/// How long the shine takes to collapse into the playhead when lost.
pub const COLLAPSE: f32 = 0.5;
/// How long a step up's burst lasts, at most, in seconds.
pub const STEP_LIFE: f32 = 1.4;
/// The least strength that shows.
const SHOWS: f32 = 0.00025;
/// How strong the shine is once the swirl, the rim, the text's tint and
/// the time left's ring are whole in the rainbow (they come in as it
/// grows to it): the first tier's glow.
pub const WHOLE: f32 = TIERS[0].glow;
/// How many bands fan out with no tier (as the shine collapses).
const NO_BANDS: f32 = 6.0;

/// A streak's shine as the shaders take it: its level (0: none, up to the
/// top's 1), how strong it showed when it changed (its level before, or
/// part way there) and when that was (seconds, the singing's clock).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Glow {
    /// Its level.
    pub level: f32,
    /// How strong it showed as it changed.
    pub from: f32,
    /// When it changed.
    pub since: f32,
}

impl Glow {
    /// How strong it is at `time`, and how far it has collapsed
    /// ([`shine_at`]).
    pub fn at(&self, time: f32) -> (f32, f32) {
        shine_at(self, time)
    }

    /// Whether anything of it shows at `time`.
    pub fn shows(&self, time: f32) -> bool {
        self.at(time).0 > SHOWS
    }
}

fn fract(x: f32) -> f32 {
    x - x.floor()
}

fn smooth(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// WGSL's `smoothstep`.
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    smooth((x - e0) / (e1 - e0))
}

/// How far `p` is outside a box `half` its size round the origin, its
/// corners `r` round (negative inside): the shaders' `round_box`.
pub fn round_box(p: (f32, f32), half: (f32, f32), r: f32) -> f32 {
    let (qx, qy) = (p.0.abs() - half.0 + r, p.1.abs() - half.1 + r);
    (qx.max(0.0).hypot(qy.max(0.0))) + qx.max(qy).min(0.0) - r
}

/// A colour of the rainbow, `h` round it (a whole turn every 1).
pub fn hue(h: f32) -> [f32; 3] {
    let h = fract(h) * 6.0;
    [
        ((h - 3.0).abs() - 1.0).clamp(0.0, 1.0),
        (2.0 - (h - 2.0).abs()).clamp(0.0, 1.0),
        (2.0 - (h - 4.0).abs()).clamp(0.0, 1.0),
    ]
}

/// How strong `s` is at `time`, and how far it has collapsed (0 to 1):
/// easing over [`RISE`] from how strong it showed to its level, up or down
/// a level alike; lost (its level 0), collapsing over [`COLLAPSE`].
pub fn shine_at(s: &Glow, time: f32) -> (f32, f32) {
    let t = (time - s.since).max(0.0);
    if s.level <= 0.0 && s.from > 0.0 {
        let c = (t / COLLAPSE).clamp(0.0, 1.0);
        (s.from * (1.0 - c), c)
    } else {
        (s.from + (s.level - s.from) * smooth(t / RISE), 0.0)
    }
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

fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t)
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
/// the rainbow running along it as strongly as `s` shows.
pub fn text_tint(col: [f32; 3], x: f32, s: &Glow, time: f32) -> [f32; 3] {
    let k = shine_at(s, time).0;
    mix3(col, hue(x * 0.01 - time * 0.6), 0.85 * (k / WHOLE).min(1.0))
}

/// The bands fanning out from `origin` over a lane at `centre`, `half`
/// its size, as `s` says, while anything of them shows at `time`.
pub fn bands_of(
    origin: (f32, f32),
    (centre, half): ((f32, f32), (f32, f32)),
    s: &Glow,
    time: f64,
) -> Option<Instance> {
    s.shows(time as f32).then(|| Instance {
        quad: [
            centre.0 - half.0,
            centre.1 - half.1,
            centre.0 + half.0,
            centre.1 + half.1,
        ],
        shape: [origin.0, origin.1, half.0, half.1],
        uv: [centre.0, centre.1, half.0, half.1],
        params: [kind::BANDS, s.level, s.from, s.since],
        ..Instance::default()
    })
}

/// The swirl round `centre`, `radius` out, as `s` says, while anything of
/// it shows at `time`.
pub fn swirl_of(centre: (f32, f32), radius: f32, s: &Glow, time: f64) -> Option<Instance> {
    let m = radius * 1.8 + 2.0;
    s.shows(time as f32).then(|| Instance {
        quad: [centre.0 - m, centre.1 - m, centre.0 + m, centre.1 + m],
        shape: [centre.0, centre.1, radius, 0.0],
        params: [kind::SWIRL, s.level, s.from, s.since],
        ..Instance::default()
    })
}

/// The rainbow edge round a bar at `centre`, `half` its size, as `s` says,
/// while anything of it shows at `time`.
pub fn rim_of(centre: (f32, f32), half: (f32, f32), s: &Glow, time: f64) -> Option<Instance> {
    let m = (half.1 * 0.9).max(1.5) * 3.0 + 1.0;
    let (mx, my) = (half.0 + m, half.1 + m);
    s.shows(time as f32).then(|| Instance {
        quad: [centre.0 - mx, centre.1 - my, centre.0 + mx, centre.1 + my],
        shape: [centre.0, centre.1, half.0, half.1],
        params: [kind::RIM, s.level, s.from, s.since],
        ..Instance::default()
    })
}

/// `text` (an instance of text) in `s`'s rainbow ([`text_tint`]).
pub fn in_rainbow(text: &mut Instance, s: &Glow) {
    text.strain = [s.level, s.from, s.since, flags::RAINBOW as f32];
}

/// The shine an instance of the bands, a swirl, a rim or text in rainbow
/// was made with, as the shader reads it.
pub fn shine_of(i: &Instance) -> Option<Glow> {
    let k = i.params[0];
    if [kind::BANDS, kind::SWIRL, kind::RIM].contains(&k) {
        let [_, level, from, since] = i.params;
        return Some(Glow { level, from, since });
    }
    let rainbow = k == kind::TEXT && i.strain[3] as u32 & flags::RAINBOW != 0;
    rainbow.then_some(Glow {
        level: i.strain[0],
        from: i.strain[1],
        since: i.strain[2],
    })
}

/// Piece `q` of step up `s` `age` seconds on: a star, a note or a rainbow
/// bit, shot out (mostly up) from where it went up, spinning, arcing down,
/// its colour running round the rainbow, fading in its own time.
pub fn step_up(s: &Spray, q: u32, age: f32) -> Option<Sprite> {
    let r = |j: u32| rand(s.seed, q, j);
    let life = STEP_LIFE * (0.6 + 0.4 * r(4));
    let t = age / life;
    if t >= 1.0 {
        return None;
    }
    let a = -TAU / 4.0 + (r(0) - 0.5) * TAU * 0.4;
    let speed = s.size * (3.0 + 6.0 * r(1));
    let g = s.size * 5.0;
    let (look, big) = match q % 3 {
        0 => (Look::Sparkle, 0.25 + 0.25 * r(5)),
        1 => (Look::Note, 0.35 + 0.2 * r(5)),
        _ => (Look::Bit, 0.12 + 0.1 * r(5)),
    };
    let size = s.size * big * (1.0 - 0.3 * t);
    let half = match look {
        Look::Bit => (size, size * 0.45),
        _ => (size, size),
    };
    Some(Sprite {
        at: (
            s.at.0 + a.cos() * speed * age + s.drift * age,
            s.at.1 + a.sin() * speed * age + 0.5 * g * age * age,
        ),
        half,
        alpha: (1.0 - t).powf(1.2),
        rgb: hue(r(6) + age * 0.8),
        look,
        turn: r(2) * TAU + (r(3) - 0.5) * 10.0 * age,
    })
}

/// [`hue`], [`shine_at`], [`bands`], [`swirl`], [`rim`], [`rim_at`],
/// [`lane_mask`] and [`text_tint`] in WGSL (and `TAU`, which the stage's
/// shader has not otherwise), with [`held`] and the rows' values they
/// read (and `TIERS`, how many there are, for the rest of the shader):
/// for the stage's shader, which gives them `round_box`.
pub fn stage_wgsl() -> String {
    let row = |value: fn(&crate::tiers::TierSpec) -> f32| {
        crate::tiers::wgsl_array(TIERS.iter().map(value))
    };
    format!(
        "
const TAU: f32 = {TAU:?};
const RISE: f32 = {RISE:?};
const COLLAPSE: f32 = {COLLAPSE:?};
const SHOWS: f32 = {SHOWS:?};
const WHOLE: f32 = {WHOLE:?};
const NO_BANDS: f32 = {NO_BANDS:?};
const TIERS: u32 = {}u;
const GLOWS: array<f32, TIERS> = {};
const BANDS: array<f32, TIERS> = {};
const LANE_LIGHTS: array<f32, TIERS> = {};
const RAINBOW: u32 = {}u;
{HUE}{SHINE}",
        Tier::COUNT,
        row(|t| t.glow),
        row(|t| t.bands),
        row(|t| t.lane_light),
        flags::RAINBOW
    )
}

/// [`hue`] and [`step_up`] in WGSL: for the particles' shader.
pub fn particles_wgsl() -> String {
    format!(
        "
const STEP_LIFE: f32 = {STEP_LIFE:?};
const LOOK_NOTE: u32 = {}u;
const LOOK_BIT: u32 = {}u;
{HUE}{STEP}",
        Look::Note.code(),
        Look::Bit.code()
    )
}

const HUE: &str = r"
fn hue(h0: f32) -> vec3<f32> {
    let h = fract(h0) * 6.0;
    return vec3<f32>(
        clamp(abs(h - 3.0) - 1.0, 0.0, 1.0),
        clamp(2.0 - abs(h - 2.0), 0.0, 1.0),
        clamp(2.0 - abs(h - 4.0), 0.0, 1.0),
    );
}
";

const SHINE: &str = r"
// How strong a shine (level, from, since) is at `time`, and how far it
// has collapsed.
fn shine_at(s: vec3<f32>, time: f32) -> vec2<f32> {
    let t = max(time - s.z, 0.0);
    if s.x <= 0.0 && s.y > 0.0 {
        let c = clamp(t / COLLAPSE, 0.0, 1.0);
        return vec2<f32>(s.y * (1.0 - c), c);
    }
    return vec2<f32>(s.y + (s.x - s.y) * smoothstep(0.0, 1.0, t / RISE), 0.0);
}

// A held look at glow `k`: the straight line through none at 0 and each
// row's value at its glow, the top's from there on.
fn held(k: f32, values: array<f32, TIERS>) -> f32 {
    var a = vec2<f32>(0.0, 0.0);
    for (var i = 0u; i < TIERS; i = i + 1u) {
        let b = vec2<f32>(GLOWS[i], values[i]);
        if k < b.x {
            return a.y + (b.y - a.y) * ((k - a.x) / (b.x - a.x));
        }
        a = b;
    }
    return a.y;
}

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

fn text_tint(col: vec3<f32>, x: f32, s: vec3<f32>, time: f32) -> vec3<f32> {
    let k = shine_at(s, time).x;
    return mix(col, hue(x * 0.01 - time * 0.6), 0.85 * min(k / WHOLE, 1.0));
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
";

const STEP: &str = r"
fn step_up(at: vec4<f32>, size: f32, seed: u32, q: u32, age: f32, drift: f32) -> Sp {
    var o: Sp;
    let life = STEP_LIFE * (0.6 + 0.4 * r(seed, q, 4u));
    let t = age / life;
    if t >= 1.0 {
        return o;
    }
    let a = -TAU / 4.0 + (r(seed, q, 0u) - 0.5) * TAU * 0.4;
    let speed = size * (3.0 + 6.0 * r(seed, q, 1u));
    let g = size * 5.0;
    var look = LOOK_BIT;
    var big = 0.12 + 0.1 * r(seed, q, 5u);
    if q % 3u == 0u {
        look = LOOK_SPARKLE;
        big = 0.25 + 0.25 * r(seed, q, 5u);
    } else if q % 3u == 1u {
        look = LOOK_NOTE;
        big = 0.35 + 0.2 * r(seed, q, 5u);
    }
    let each = size * big * (1.0 - 0.3 * t);
    o.half = vec2<f32>(each);
    if look == LOOK_BIT {
        o.half = vec2<f32>(each, each * 0.45);
    }
    o.at = vec2<f32>(
        at.x + cos(a) * speed * age + drift,
        at.y + sin(a) * speed * age + 0.5 * g * age * age,
    );
    o.alpha = pow(1.0 - t, 1.2);
    o.rgb = hue(r(seed, q, 6u) + age * 0.8);
    o.look = look;
    o.turn = r(seed, q, 2u) * TAU + (r(seed, q, 3u) - 0.5) * 10.0 * age;
    o.shows = 1u;
    return o;
}

// A note: a slanted head and its stem with a flag, turned, over what is
// under it, a little glow.
fn draw_note(p: vec2<f32>, c: vec2<f32>, half: vec2<f32>, turn: f32, look: vec4<f32>) -> vec4<f32> {
    let s = sin(turn);
    let co = cos(turn);
    let q0 = (p - c) / max(half.x, 0.5);
    let l = vec2<f32>(co * q0.x + s * q0.y, -s * q0.x + co * q0.y);
    // The head, an ellipse low left, tilted.
    let hp = l - vec2<f32>(-0.25, 0.45);
    let ht = vec2<f32>(0.9 * hp.x + 0.45 * hp.y, -0.45 * hp.x + 0.9 * hp.y);
    let head = length(ht / vec2<f32>(0.36, 0.24)) - 1.0;
    // The stem up its right, and a flag off its top.
    let stem = max(abs(l.x - 0.08) - 0.07, abs(l.y + 0.1) - 0.6);
    let fl = l - vec2<f32>(0.3, -0.45);
    let flag = max(abs(fl.x + 0.3 * fl.y) - 0.07, abs(fl.y) - 0.25);
    let d = min(head * 0.3, min(stem, flag)) * max(half.x, 0.5);
    let cover = clamp(0.5 - d, 0.0, 1.0);
    let glow = exp(-max(d, 0.0) / max(half.x * 0.25, 1.0)) * 0.3;
    return vec4<f32>(out_rgb(look.rgb) * (cover + glow) * look.a, cover * look.a);
}

// A rainbow bit: a small bar, turned, over what is under it, lit.
fn draw_bit(p: vec2<f32>, c: vec2<f32>, half: vec2<f32>, turn: f32, look: vec4<f32>) -> vec4<f32> {
    let s = sin(turn);
    let co = cos(turn);
    let q0 = p - c;
    let l = vec2<f32>(co * q0.x + s * q0.y, -s * q0.x + co * q0.y);
    let d = round_box(l, half, min(half.x, half.y) * 0.5);
    let cover = clamp(0.5 - d, 0.0, 1.0);
    let glow = exp(-max(d, 0.0) / max(half.y, 1.0)) * 0.4;
    return vec4<f32>(out_rgb(look.rgb) * (cover + glow) * look.a, cover * look.a);
}
";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::particles::Kind;

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

    /// A new level comes in over [`RISE`]; lost, the shine collapses into
    /// the playhead over [`COLLAPSE`] (reaching less, drained of colour)
    /// and is gone.
    #[test]
    fn a_lost_streak_collapses_into_the_playhead() {
        let rise = shine(0.5, 0.25);
        assert_eq!(shine_at(&rise, 10.0).0, 0.25);
        assert!((shine_at(&rise, 10.0 + RISE).0 - 0.5).abs() < 1e-6);
        assert_eq!(shine_at(&rise, 99.0), (0.5, 0.0));
        let lost = shine(0.0, 0.75);
        let (k, c) = shine_at(&lost, 10.25);
        assert!((c - 0.5).abs() < 1e-6 && (k - 0.375).abs() < 1e-6);
        assert_eq!(shine_at(&lost, 10.0 + COLLAPSE), (0.0, 1.0));
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
        assert!(at(near, COLLAPSE + 0.01) == 0.0, "gone");
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
        // The swirl and the rim go with it.
        assert_eq!(swirl(near, origin, 30.0, &lost, 10.0 + COLLAPSE), [0.0; 3]);
        assert_eq!(rim(0.0, 10.0, 3.0, &lost, 10.0 + COLLAPSE), [0.0; 3]);
    }

    /// A step up's pieces: stars, notes and rainbow bits, shot out from
    /// where it went up (mostly up), arcing down, spinning, fading, their
    /// colours running round the rainbow; gone after their life.
    #[test]
    fn a_step_up_bursts_and_its_pieces_arc_and_fade() {
        let s = Spray {
            at: (400.0, 300.0),
            half: (0.0, 0.0),
            start: 10.0,
            drift: 0.0,
            size: 40.0,
            seed: 9,
            rgb: [1.0; 3],
            kind: Kind::StepUp,
        };
        let mut looks = std::collections::HashSet::new();
        let mut up = 0;
        for q in 0..super::super::particles::QUADS {
            let at = |t: f32| step_up(&s, q, t);
            let (a, b, c) = (at(0.05).unwrap(), at(0.25).unwrap(), at(0.45).unwrap());
            looks.insert(a.look.code());
            // Out from where it went up, the faster pieces further.
            assert!((b.at.0 - 400.0).abs() >= (a.at.0 - 400.0).abs());
            // Arcing: falling faster each step.
            let (v1, v2) = (b.at.1 - a.at.1, c.at.1 - b.at.1);
            assert!(v2 > v1, "{q}");
            up += usize::from(a.at.1 < 300.0);
            assert!(c.alpha < a.alpha && (c.turn - a.turn).abs() > 0.0);
            assert_ne!(a.rgb, c.rgb, "its colour runs");
            assert!(at(STEP_LIFE).is_none());
        }
        assert_eq!(looks.len(), 3, "stars, notes and bits");
        assert!(
            up > super::super::particles::QUADS as usize * 2 / 3,
            "mostly up: {up}"
        );
    }

    /// Down a tier (Amazing's shine to Great's) it eases down over [`RISE`]
    /// and shines on at Great's: no collapse, its bands, swirl and rim all
    /// there.
    #[test]
    fn a_drop_of_a_level_still_shines() {
        let drop = shine(0.5, 0.75);
        assert_eq!(drop.at(10.0), (0.75, 0.0));
        let half = drop.at(10.0 + RISE / 2.0);
        assert!(half.0 < 0.75 && half.0 > 0.5 && half.1 == 0.0, "{half:?}");
        assert_eq!(drop.at(99.0), (0.5, 0.0));
        let p = (360.0, 200.0);
        let sum = |c: [f32; 3]| c.iter().sum::<f32>();
        let lit = (0..40).map(|i| {
            let t = 12.0 + i as f32 * 0.013;
            sum(bands(p, (300.0, 200.0), (800.0, 120.0), &drop, t))
        });
        assert!(lit.fold(0.0, f32::max) > 0.01);
        let round = (0..36).map(|i| {
            let a = i as f32 / 36.0 * TAU;
            let p = (300.0 + 30.0 * a.cos(), 200.0 + 30.0 * a.sin());
            sum(swirl(p, (300.0, 200.0), 30.0, &drop, 12.0))
        });
        assert!(round.fold(0.0, f32::max) > 0.01);
        assert!(sum(rim(0.0, 10.0, 3.0, &drop, 12.0)) > 0.1);
        assert!(drop.shows(99.0));
    }

    #[test]
    fn the_rainbow_goes_round() {
        let near = |a: [f32; 3], b: [f32; 3]| (0..3).all(|i| (a[i] - b[i]).abs() < 1e-4);
        assert!(near(hue(0.0), [1.0, 0.0, 0.0]));
        assert!(near(hue(1.0 / 3.0), [0.0, 1.0, 0.0]));
        assert!(near(hue(2.0 / 3.0), [0.0, 0.0, 1.0]));
        assert!(near(hue(-1.0 / 3.0), hue(2.0 / 3.0)));
    }
}
