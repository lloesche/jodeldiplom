//! Gold coins on the GPU, for the rungs that pay out: a fountain of them
//! shot up out of the playhead and falling back ([`coin_fountain`], a
//! rung reached), coins raining down the whole screen ([`coin_rain`], the
//! top rung) and a coin or two flicked up out of a word as it is hit
//! ([`coin_flick`], while a high rung is held). Each is one record
//! ([`Spray`]) on the particles' pass ([`crate::gui::particles`]); every
//! coin is worked out in its vertex shader in closed form from the seed
//! and the age, and drawn solid ([`Look::Gold`]: a coin spinning on its
//! edge, its rim and thickness showing as it turns, a star struck on its
//! face, a glint as it faces round), over the frame rather than into the
//! light, so it reads as metal on any cover.
//!
//! The lyric line stays readable: a fountain's coins fall back only as
//! far as the playhead and fade there (the lyrics are under the lane), a
//! flick's hop up off its word and fade on the way down, and the rain's
//! coins cross the lyrics falling fast (a few hundredths of a second
//! over a line).
//!
//! Three effects: the fountain fired at a rung or a phrase, the rain fired
//! likewise, the flick held (a coin off each word hit while it is). Each
//! draws from the playhead's slot, over the stage ([`super::Layer::Playhead`]).
//!
//! [`fountain`], [`rain`] and [`flick`] are the shader's maths on the
//! processor, held to it on the GPU (`crate::gui::shader_check`, from their
//! motions' probes: [`crate::gui::particles::MOTIONS`]).

use super::{At, Layer, Live, Out, Slot, Spec};
use crate::gui::particles::{Kind, Look, Spray, Sprite};
use crate::gui::sparks::{mix, rand, TAU, WHITE};

/// A fountain of coins out of the playhead.
pub const FOUNTAIN: Spec = Spec {
    name: "coin fountain",
    ident: "fountain",
    at: &[Layer::Playhead],
    draw: draw_fountain,
    ..Spec::NONE
};

/// Coins raining down the screen.
pub const RAIN: Spec = Spec {
    name: "coin rain",
    ident: "rain",
    at: &[Layer::Playhead],
    draw: draw_rain,
    ..Spec::NONE
};

/// A coin flicked out of each word hit.
pub const FLICK: Spec = Spec {
    name: "coin flick",
    ident: "flick",
    at: &[Layer::Playhead],
    draw: draw_flick,
    ..Spec::NONE
};

/// A seed of `clock` (the singing's time) for an effect's sprays, `salt`
/// apart from another's.
fn seed(clock: f64, salt: u32) -> u32 {
    (clock * 1000.0) as u32 ^ salt
}

fn draw_fountain(at: &At, live: &Live, _: &mut Slot, out: &mut Out) {
    if let (&At::Playhead { tip, pill, .. }, true) = (at, live.fired) {
        let size = pill * 0.3 * live.strength;
        let s = coin_fountain(tip, size, live.clock, seed(live.clock, 0xF0_4417));
        out.sprays.push(s);
    }
}

fn draw_rain(at: &At, live: &Live, _: &mut Slot, out: &mut Out) {
    if let (&At::Playhead { pill, area, .. }, true) = (at, live.fired) {
        let (w, h) = super::far_corner(area);
        let seed = seed(live.clock, 0x4A_1700);
        out.sprays.extend(coin_rain(
            w,
            h,
            pill * 0.4 * live.strength,
            live.clock,
            seed,
        ));
    }
}

fn draw_flick(at: &At, live: &Live, _: &mut Slot, out: &mut Out) {
    if let (&At::Playhead { word, pill, .. }, true) = (at, live.strength > 0.0) {
        let size = pill * 0.25 * live.listed;
        let flick = |w| coin_flick(w, size, live.clock, seed(live.clock, 0xF1_1C00));
        out.sprays.extend(word.map(flick));
    }
}

/// The coins' gold, sRGB.
pub const GOLD: [f32; 3] = [1.0, 0.76, 0.26];
/// How long a fountain lasts, in seconds.
pub const FOUNTAIN_LIFE: f32 = 2.4;
/// How long a spray of the rain lasts, in seconds.
pub const RAIN_LIFE: f32 = 3.4;
/// How long a flick lasts, in seconds.
pub const FLICK_LIFE: f32 = 0.6;
/// The sprays a rain is made of (each
/// [`QUADS`](crate::gui::particles::QUADS) coins).
pub const RAIN_SPRAYS: u32 = 3;
/// A fountain's coins (its first quads; the rest are sparkles at its
/// mouth).
const FOUNTAIN_COINS: u32 = 28;
/// How long a rain's coins keep starting, in seconds.
const RAIN_SPREAD: f32 = 1.2;
/// How much of its speed a rain's coin keeps bouncing off the floor.
const BOUNCE: f32 = 0.32;
/// How long a rain's coin bounces before it is gone, in seconds.
const BOUNCED: f32 = 0.45;

/// A fountain of coins out of `at` (the playhead), each `size` pixels in
/// radius, at `clock` (the singing's time), scattered by `seed`.
pub fn coin_fountain(at: (f32, f32), size: f32, clock: f64, seed: u32) -> Spray {
    Spray {
        at,
        half: (0.0, 0.0),
        start: clock as f32,
        drift: 0.0,
        size,
        seed,
        rgb: GOLD,
        kind: Kind::CoinFountain,
    }
}

/// Coins raining down a screen `width` × `height` from `clock`, across
/// its width, each `size` pixels in radius, bouncing a little off its
/// bottom: [`RAIN_SPRAYS`] sprays, each with a seed of its own.
pub fn coin_rain(width: f32, height: f32, size: f32, clock: f64, seed: u32) -> Vec<Spray> {
    (0..RAIN_SPRAYS)
        .map(|k| Spray {
            at: (width / 2.0, -size * 2.0),
            half: (width / 2.0, height),
            start: clock as f32,
            drift: 0.0,
            size,
            seed: seed.wrapping_add(k.wrapping_mul(0x9E37)),
            rgb: GOLD,
            kind: Kind::CoinRain,
        })
        .collect()
}

/// A coin or two flicked up out of a word hit at `at`, each `size` pixels
/// in radius, at `clock`.
pub fn coin_flick(at: (f32, f32), size: f32, clock: f64, seed: u32) -> Spray {
    Spray {
        at,
        half: (0.0, 0.0),
        start: clock as f32,
        drift: 0.0,
        size,
        seed,
        rgb: GOLD,
        kind: Kind::CoinFlick,
    }
}

/// Quad `q` of a fountain `s`, `age` seconds on: the first
/// [`FOUNTAIN_COINS`] coins shot up and out (each a moment after the
/// last), arcing back down under gravity, spinning, fading as they come
/// back down to the playhead; the rest sparkles flashing at its mouth.
pub fn fountain(s: &Spray, q: u32, age: f32) -> Option<Sprite> {
    let r = |j: u32| rand(s.seed, q, j);
    if q >= FOUNTAIN_COINS {
        let u = age / 0.4;
        if u >= 1.0 {
            return None;
        }
        let ray = s.size * (1.6 + 1.2 * r(0)) * (1.0 - 0.6 * u);
        return Some(Sprite {
            at: (
                s.at.0 + (r(1) - 0.5) * s.size * 2.5,
                s.at.1 - s.size * (0.5 + 1.5 * r(2)) * u,
            ),
            half: (ray, ray),
            alpha: (1.0 - u) * (1.0 - u),
            rgb: mix(s.rgb, WHITE, 0.7),
            look: Look::Sparkle,
            turn: r(3) * TAU,
        });
    }
    let t = age - r(0) * 0.3;
    let a = -TAU / 4.0 + (r(1) - 0.5) * 2.0;
    let v = s.size * (18.0 + 22.0 * r(2));
    let g = s.size * 45.0;
    let (vx, vy) = (a.cos() * v, a.sin() * v);
    // Back down to the playhead's height, and a little past it.
    let life = -2.0 * vy / g * 1.03;
    if t < 0.0 || t >= life {
        return None;
    }
    let grow = (t / 0.08).min(1.0);
    let radius = s.size * (0.85 + 0.3 * r(3)) * (0.4 + 0.6 * grow);
    Some(Sprite {
        at: (s.at.0 + vx * t, s.at.1 + vy * t + 0.5 * g * t * t),
        half: (radius, radius),
        alpha: ((life - t) / (0.3 * life)).min(1.0),
        rgb: mix(s.rgb, WHITE, 0.2 * r(4)),
        look: Look::Gold,
        turn: r(5) * TAU + (7.0 + 9.0 * r(6)) * t,
    })
}

/// Quad `q` of a spray of the rain `s`, `age` seconds on: a coin in at
/// its own moment over the screen's width (`band`), falling faster and
/// faster, swaying, spinning, bouncing once off the screen's bottom and
/// fading.
pub fn rain(s: &Spray, q: u32, age: f32) -> Option<Sprite> {
    let r = |j: u32| rand(s.seed, q, j);
    let t = age - r(0) * RAIN_SPREAD;
    if t < 0.0 {
        return None;
    }
    let radius = s.size * (0.75 + 0.35 * r(1));
    let v0 = s.size * (6.0 + 6.0 * r(2));
    let g = s.size * 36.0;
    let floor = s.screen_height() - radius;
    let hit = ((v0 * v0 + 2.0 * g * (floor - s.at.1)).max(0.0).sqrt() - v0) / g;
    let (y, alpha) = if t < hit {
        (s.at.1 + v0 * t + 0.5 * g * t * t, 1.0)
    } else {
        let b = t - hit;
        if b >= BOUNCED {
            return None;
        }
        let up = BOUNCE * (v0 + g * hit);
        (
            (floor - up * b + 0.5 * g * b * b).min(floor),
            1.0 - b / BOUNCED,
        )
    };
    let x = s.at.0
        + (r(3) * 2.0 - 1.0) * s.band()
        + s.size * 1.2 * (t * (2.0 + r(4)) + r(5) * TAU).sin();
    Some(Sprite {
        at: (x, y),
        half: (radius, radius),
        alpha,
        rgb: mix(s.rgb, WHITE, 0.2 * r(6)),
        look: Look::Gold,
        turn: r(7) * TAU + (5.0 + 8.0 * r(8)) * t,
    })
}

/// Quad `q` of a flick `s`, `age` seconds on: a coin (and every other
/// time a second) hopping up and off to a side, spinning fast, fading on
/// its way down; a few sparkles where it left.
pub fn flick(s: &Spray, q: u32, age: f32) -> Option<Sprite> {
    let r = |j: u32| rand(s.seed, q, j);
    let u = age / FLICK_LIFE;
    if q >= 2 {
        let u = age / 0.3;
        if q >= 5 || u >= 1.0 {
            return None;
        }
        let ray = s.size * (1.4 + 0.8 * r(0)) * (1.0 - 0.5 * u);
        return Some(Sprite {
            at: (
                s.at.0 + (r(1) - 0.5) * s.size * 2.0,
                s.at.1 - s.size * (0.3 + 1.2 * r(2)) * u,
            ),
            half: (ray, ray),
            alpha: 1.0 - u,
            rgb: mix(s.rgb, WHITE, 0.7),
            look: Look::Sparkle,
            turn: r(3) * TAU,
        });
    }
    if q == 1 && rand(s.seed, 0, 9) >= 0.5 {
        return None;
    }
    let side = if q == 0 { -1.0 } else { 1.0 } * if rand(s.seed, 0, 8) < 0.5 { 1.0 } else { -1.0 };
    let a = -TAU / 4.0 + side * (0.25 + 0.3 * r(1));
    let v = s.size * (16.0 + 6.0 * r(2));
    let g = s.size * 50.0;
    let radius = s.size * (0.9 + 0.2 * r(3)) * (0.5 + 0.5 * (age / 0.06).min(1.0));
    Some(Sprite {
        at: (
            s.at.0 + a.cos() * v * age,
            s.at.1 + a.sin() * v * age + 0.5 * g * age * age,
        ),
        half: (radius, radius),
        alpha: ((1.0 - u) / 0.4).min(1.0),
        rgb: s.rgb,
        look: Look::Gold,
        turn: r(4) * TAU + 16.0 * age,
    })
}

/// [`fountain`] in WGSL: for the particles' shader (which gives it `r`,
/// `Sp` and the looks' numbers).
pub fn fountain_wgsl() -> String {
    format!("\nconst FOUNTAIN_COINS: u32 = {FOUNTAIN_COINS}u;\n{FOUNTAIN_WGSL}")
}

/// [`rain`] in WGSL.
pub fn rain_wgsl() -> String {
    format!(
        "\nconst RAIN_SPREAD: f32 = {RAIN_SPREAD:?};\nconst BOUNCE: f32 = {BOUNCE:?};\nconst BOUNCED: f32 = {BOUNCED:?};\n{RAIN_WGSL}"
    )
}

/// [`flick`] in WGSL.
pub fn flick_wgsl() -> String {
    format!("\nconst FLICK_LIFE: f32 = {FLICK_LIFE:?};\n{FLICK_WGSL}")
}

const FOUNTAIN_WGSL: &str = r"
fn coin_fountain(at: vec4<f32>, rgb: vec3<f32>, size: f32, seed: u32, q: u32, age: f32) -> Sp {
    var o: Sp;
    if q >= FOUNTAIN_COINS {
        let u = age / 0.4;
        if u >= 1.0 {
            return o;
        }
        let ray = size * (1.6 + 1.2 * r(seed, q, 0u)) * (1.0 - 0.6 * u);
        o.at = vec2<f32>(
            at.x + (r(seed, q, 1u) - 0.5) * size * 2.5,
            at.y - size * (0.5 + 1.5 * r(seed, q, 2u)) * u,
        );
        o.half = vec2<f32>(ray);
        o.alpha = (1.0 - u) * (1.0 - u);
        o.rgb = mix(rgb, vec3<f32>(1.0), 0.7);
        o.look = LOOK_SPARKLE;
        o.turn = r(seed, q, 3u) * TAU;
        o.shows = 1u;
        return o;
    }
    let t = age - r(seed, q, 0u) * 0.3;
    let a = -TAU / 4.0 + (r(seed, q, 1u) - 0.5) * 2.0;
    let v = size * (18.0 + 22.0 * r(seed, q, 2u));
    let g = size * 45.0;
    let vx = cos(a) * v;
    let vy = sin(a) * v;
    let life = -2.0 * vy / g * 1.03;
    if t < 0.0 || t >= life {
        return o;
    }
    let grow = min(t / 0.08, 1.0);
    let radius = size * (0.85 + 0.3 * r(seed, q, 3u)) * (0.4 + 0.6 * grow);
    o.at = vec2<f32>(at.x + vx * t, at.y + vy * t + 0.5 * g * t * t);
    o.half = vec2<f32>(radius);
    o.alpha = min((life - t) / (0.3 * life), 1.0);
    o.rgb = mix(rgb, vec3<f32>(1.0), 0.2 * r(seed, q, 4u));
    o.look = LOOK_GOLD;
    o.turn = r(seed, q, 5u) * TAU + (7.0 + 9.0 * r(seed, q, 6u)) * t;
    o.shows = 1u;
    return o;
}

";

const RAIN_WGSL: &str = r"
fn coin_rain(at: vec4<f32>, rgb: vec3<f32>, size: f32, seed: u32, q: u32, age: f32) -> Sp {
    var o: Sp;
    let t = age - r(seed, q, 0u) * RAIN_SPREAD;
    if t < 0.0 {
        return o;
    }
    let radius = size * (0.75 + 0.35 * r(seed, q, 1u));
    let v0 = size * (6.0 + 6.0 * r(seed, q, 2u));
    let g = size * 36.0;
    let floor_y = at.w - radius;
    let hit = (sqrt(max(v0 * v0 + 2.0 * g * (floor_y - at.y), 0.0)) - v0) / g;
    var y = at.y + v0 * t + 0.5 * g * t * t;
    var alpha = 1.0;
    if t >= hit {
        let b = t - hit;
        if b >= BOUNCED {
            return o;
        }
        let up = BOUNCE * (v0 + g * hit);
        y = min(floor_y - up * b + 0.5 * g * b * b, floor_y);
        alpha = 1.0 - b / BOUNCED;
    }
    let x = at.x + (r(seed, q, 3u) * 2.0 - 1.0) * at.z
        + size * 1.2 * sin(t * (2.0 + r(seed, q, 4u)) + r(seed, q, 5u) * TAU);
    o.at = vec2<f32>(x, y);
    o.half = vec2<f32>(radius);
    o.alpha = alpha;
    o.rgb = mix(rgb, vec3<f32>(1.0), 0.2 * r(seed, q, 6u));
    o.look = LOOK_GOLD;
    o.turn = r(seed, q, 7u) * TAU + (5.0 + 8.0 * r(seed, q, 8u)) * t;
    o.shows = 1u;
    return o;
}

";

const FLICK_WGSL: &str = r"
fn coin_flick(at: vec4<f32>, rgb: vec3<f32>, size: f32, seed: u32, q: u32, age: f32) -> Sp {
    var o: Sp;
    let u = age / FLICK_LIFE;
    if q >= 2u {
        let w = age / 0.3;
        if q >= 5u || w >= 1.0 {
            return o;
        }
        let ray = size * (1.4 + 0.8 * r(seed, q, 0u)) * (1.0 - 0.5 * w);
        o.at = vec2<f32>(
            at.x + (r(seed, q, 1u) - 0.5) * size * 2.0,
            at.y - size * (0.3 + 1.2 * r(seed, q, 2u)) * w,
        );
        o.half = vec2<f32>(ray);
        o.alpha = 1.0 - w;
        o.rgb = mix(rgb, vec3<f32>(1.0), 0.7);
        o.look = LOOK_SPARKLE;
        o.turn = r(seed, q, 3u) * TAU;
        o.shows = 1u;
        return o;
    }
    if q == 1u && r(seed, 0u, 9u) >= 0.5 {
        return o;
    }
    var side = 1.0;
    if q == 0u {
        side = -1.0;
    }
    if r(seed, 0u, 8u) >= 0.5 {
        side = -side;
    }
    let a = -TAU / 4.0 + side * (0.25 + 0.3 * r(seed, q, 1u));
    let v = size * (16.0 + 6.0 * r(seed, q, 2u));
    let g = size * 50.0;
    let radius = size * (0.9 + 0.2 * r(seed, q, 3u)) * (0.5 + 0.5 * min(age / 0.06, 1.0));
    o.at = vec2<f32>(at.x + cos(a) * v * age, at.y + sin(a) * v * age + 0.5 * g * age * age);
    o.half = vec2<f32>(radius);
    o.alpha = min((1.0 - u) / 0.4, 1.0);
    o.rgb = rgb;
    o.look = LOOK_GOLD;
    o.turn = r(seed, q, 4u) * TAU + 16.0 * age;
    o.shows = 1u;
    return o;
}

";

/// How a gold coin is drawn, in WGSL (its look's line, with how far its
/// quad reaches past its radius for its glint:
/// [`crate::gui::particles::LOOKS`]).
pub const DRAW_GOLD: &str = r"
// A solid gold coin spinning about its upright (`turn`): its face an
// ellipse its width the spin's cosine, its milled edge showing beside it
// as it turns, a raised rim and a star struck on the face lit from the
// top left, a sheen sweeping over it, a dark outline so it reads on any
// cover, and a glint flashing off it as it faces round. Over, the glint
// added.
fn draw_gold(p: vec2<f32>, c: vec2<f32>, half: vec2<f32>, turn: f32, look: vec4<f32>) -> vec4<f32> {
    let s = max(half.x, 1.0);
    let ct = cos(turn);
    let st = sin(turn);
    let w = max(abs(ct), 0.06);
    let l = (p - c) / s;
    let th = 0.16 * abs(st);
    var side = 1.0;
    if st * ct < 0.0 {
        side = -1.0;
    }
    let ef = vec2<f32>((l.x + side * th * 0.5) / w, l.y);
    let eb = vec2<f32>((l.x - side * th * 0.5) / w, l.y);
    let rf = length(ef);
    let df = (rf - 1.0) * s * w;
    let db = (length(eb) - 1.0) * s * w;
    let band = max(abs(l.y) - 1.0, abs(l.x) - th * 0.5) * s;
    let dc = min(df, min(db, band));
    let cover = clamp(0.5 - dc, 0.0, 1.0);
    let face = clamp(0.5 - df, 0.0, 1.0);
    // The face: lit from the top left, its rim raised, its star struck.
    let n = ef / max(rf, 1e-3);
    let shade = 0.8 + 0.32 * (-ef.y * 0.75 - ef.x * 0.4);
    let ring = smoothstep(0.72, 0.78, rf) * (1.0 - smoothstep(0.86, 0.92, rf));
    let bevel = dot(n, vec2<f32>(-0.45, -0.89));
    let an = atan2(ef.y, ef.x);
    let star_r = 0.4 + 0.15 * cos(5.0 * an + 1.5708);
    let star = clamp((star_r - rf) * 14.0, 0.0, 1.0);
    let sheen = exp(-pow((ef.x * 0.6 + ef.y * 0.8) + 1.6 * sin(turn * 0.5), 2.0) * 6.0) * 0.3;
    var face_col = look.rgb * shade * (1.0 + 0.5 * ring * bevel + 0.22 * star - 0.12 * smoothstep(0.92, 1.0, rf));
    face_col = face_col * (0.7 + 0.4 * abs(ct)) + vec3<f32>(1.0, 0.92, 0.7) * sheen;
    if ct < 0.0 {
        face_col = face_col * vec3<f32>(1.0, 0.92, 0.82);
    }
    // The edge: darker, milled.
    let edge_col = look.rgb * vec3<f32>(0.62, 0.48, 0.28) * (0.85 + 0.15 * sin(l.y * s * 1.3));
    var col = mix(edge_col, face_col, face);
    col = col * (1.0 - 0.45 * smoothstep(-2.2, 0.0, dc));
    // The glint, as it faces round.
    let flash = pow(max(cos(2.0 * turn - 0.5), 0.0), 24.0);
    let g = p - c - vec2<f32>(-0.32 * w, -0.38) * s;
    let thin = max(s * 0.06, 0.8);
    let ray = exp(-g.y * g.y / (thin * thin)) * clamp(1.0 - abs(g.x) / (s * 1.7), 0.0, 1.0)
        + exp(-g.x * g.x / (thin * thin)) * clamp(1.0 - abs(g.y) / (s * 1.7), 0.0, 1.0);
    let glint = flash * (ray * 0.9 + exp(-dot(g, g) / (s * s * 0.04)));
    let light = vec3<f32>(1.0, 0.95, 0.8) * glint;
    let a = cover * look.a;
    return vec4<f32>(out_rgb(col) * a + out_rgb(light) * look.a, a);
}
";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::particles::{sprite, QUADS};

    const CLOCK: f64 = 10.0;

    fn shown(s: &Spray, t: f32) -> Vec<Sprite> {
        (0..QUADS)
            .filter_map(|q| sprite(s, q, CLOCK as f32 + t))
            .collect()
    }

    fn coins(s: &Spray, t: f32) -> Vec<Sprite> {
        shown(s, t)
            .into_iter()
            .filter(|p| p.look == Look::Gold)
            .collect()
    }

    /// A fountain: about thirty coins shot up out of the playhead, up and
    /// out, back down to it and no further, spinning; none before its
    /// start or after its life.
    #[test]
    fn a_fountain_shoots_coins_up_and_they_fall_back() {
        let s = coin_fountain((800.0, 700.0), 30.0, CLOCK, 77);
        assert!(shown(&s, -0.01).is_empty() && shown(&s, FOUNTAIN_LIFE).is_empty());
        assert!(coins(&s, 0.45).len() + 4 >= FOUNTAIN_COINS as usize);
        let up = coins(&s, 0.5);
        assert!(up.iter().all(|p| p.at.1 < 700.0), "up");
        let high = up.iter().map(|p| 700.0 - p.at.1).fold(0.0, f32::max);
        assert!(high > 30.0 * 6.0, "high: {high}");
        let wide = up
            .iter()
            .map(|p| (p.at.0 - 800.0).abs())
            .fold(0.0, f32::max);
        assert!(wide > 30.0 * 4.0, "out: {wide}");
        // Never far below the playhead (the lyrics are under the lane).
        for k in 0..240 {
            for p in shown(&s, k as f32 * 0.01) {
                assert!(p.at.1 < 700.0 + 30.0 * 3.0, "{p:?}");
            }
        }
        // Spinning.
        let (a, b) = (sprite(&s, 3, 10.4).unwrap(), sprite(&s, 3, 10.5).unwrap());
        assert!((b.turn - a.turn).abs() > 0.5);
        // Falling back: later, the coins are lower and going.
        let late = coins(&s, 1.6);
        assert!(late.len() < up.len());
    }

    /// The rain: coins across the whole width, falling, over in a few
    /// seconds, bouncing on the screen's bottom, never below it.
    #[test]
    fn the_rain_falls_across_the_screen() {
        let (w, h) = (1600.0, 1000.0);
        let rain = coin_rain(w, h, 20.0, CLOCK, 5);
        assert_eq!(rain.len(), RAIN_SPRAYS as usize);
        let all = |t: f32| -> Vec<Sprite> { rain.iter().flat_map(|s| coins(s, t)).collect() };
        let at = all(1.0);
        assert!(at.len() > 40, "{}", at.len());
        let (left, right) = (
            at.iter().filter(|p| p.at.0 < w * 0.3).count(),
            at.iter().filter(|p| p.at.0 > w * 0.7).count(),
        );
        assert!(left > 5 && right > 5, "{left} {right}");
        for k in 0..340 {
            for p in all(k as f32 * 0.01) {
                assert!(p.at.1 <= h - p.half.1 + 1e-3, "{p:?}");
            }
        }
        // Falling: a coin lower a moment on (before it lands).
        let s = &rain[0];
        let (a, b) = (sprite(s, 4, 11.3), sprite(s, 4, 11.35));
        if let (Some(a), Some(b)) = (a, b) {
            assert!(b.at.1 > a.at.1 || b.alpha < 1.0);
        }
        assert!(all(RAIN_LIFE).is_empty());
    }

    /// A flick: one or two small coins hopping up off the word, gone in
    /// well under a second.
    #[test]
    fn a_flick_hops_a_coin_or_two_up() {
        for seed in 0..20 {
            let s = coin_flick((500.0, 400.0), 12.0, CLOCK, seed);
            let n = coins(&s, 0.1).len();
            assert!((1..=2).contains(&n), "{n}");
            for p in coins(&s, 0.2) {
                assert!(p.at.1 < 400.0, "up: {p:?}");
            }
            assert!(shown(&s, FLICK_LIFE).is_empty());
        }
    }
}
