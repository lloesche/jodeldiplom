//! The top tier's fanfare on the GPU: angels descending while the singing
//! is Diva ([`Kind::Angels`]: a flight of them drifting down the screen,
//! swaying, wings beating, star dust trailing), and a jackpot when a
//! phrase is Diva ([`Kind::Jackpot`]: a fountain of spinning gold coins
//! and stars shot up from the meter, arcing and raining down). Each is one
//! record ([`Spray`]); every quad is worked out in the particles' vertex
//! shader in closed form from its seed and age, drawn into the light that
//! blooms ([`super::bloom`]).
//!
//! [`angel`] and [`coin`] are the shader's maths on the processor, held
//! to it on the GPU (`super::shader_check`).

use super::particles::{Kind, Look, Spray, Sprite};
use super::rainbow::hue;
use super::sparks::{mix, rand, TAU};

/// How long a flight of angels takes to come down, in seconds.
pub const ANGELS_LIFE: f32 = 6.0;
/// The angels in a flight (its first quads; the rest are their dust).
pub const ANGELS: u32 = 5;
/// How long a jackpot lasts, in seconds.
pub const JACKPOT_LIFE: f32 = 2.4;

/// A flight of angels over a screen `width` × `height` from `clock` (the
/// singing's time), scattered by `seed`, each `size` tall.
pub fn angels(width: f32, height: f32, size: f32, clock: f64, seed: u32) -> Spray {
    Spray {
        at: (width / 2.0, -size),
        half: (width * 0.45, height),
        start: clock as f32,
        drift: 0.0,
        size,
        seed,
        rgb: [1.0, 0.95, 0.8],
        kind: Kind::Angels,
    }
}

/// A jackpot from `at` (the meter), its coins `size` across, at `clock`.
pub fn jackpot(at: (f32, f32), size: f32, clock: f64, seed: u32) -> Spray {
    Spray {
        at,
        half: (0.0, 0.0),
        start: clock as f32,
        drift: 0.0,
        size,
        seed,
        rgb: [1.0, 0.8, 0.3],
        kind: Kind::Jackpot,
    }
}

/// Quad `q` of a flight of angels `s`, `age` seconds on: the first
/// [`ANGELS`] each an angel (coming in at its own time, drifting down and
/// swaying, its wings beating: `turn` their phase), the others dust off
/// them (falling slower, twinkling).
pub fn angel(s: &Spray, q: u32, age: f32) -> Option<Sprite> {
    let r = |j: u32| rand(s.seed, q, j);
    let k = q % ANGELS;
    let a = |j: u32| rand(s.seed, k, j);
    // Its angel's way: in at its time, down over the screen.
    let delay = a(0) * 1.4;
    let t = (age - delay) / (ANGELS_LIFE - 1.4);
    if !(0.0..1.0).contains(&t) {
        return None;
    }
    let x0 = s.at.0 + (a(1) * 2.0 - 1.0) * s.half.0;
    let fall = s.half.1 * (0.45 + 0.35 * a(2));
    let sway = s.size * (0.6 + 0.6 * a(3));
    let ax = x0 + sway * (age * (0.9 + 0.5 * a(4)) + a(5) * TAU).sin();
    let ay = s.at.1 + fall * t + s.size * 0.4 * (age * 1.7 + a(6) * TAU).sin() * 0.3;
    let fade = (t * 6.0).min(1.0) * ((1.0 - t) * 3.0).min(1.0);
    if q < ANGELS {
        return Some(Sprite {
            at: (ax, ay),
            half: (s.size * 0.5, s.size * 0.5),
            alpha: fade,
            rgb: s.rgb,
            look: Look::Angel,
            turn: age * (7.0 + 2.0 * a(7)),
        });
    }
    // Dust: left behind along its way, drifting down, fading.
    let born = r(0) * 0.9;
    let d = (age - delay - born * (ANGELS_LIFE - 1.4)).max(0.0);
    let life = 0.9 + 0.8 * r(1);
    if d <= 0.0 || d >= life {
        return None;
    }
    let tb = born;
    let bx =
        x0 + sway * ((delay + tb * (ANGELS_LIFE - 1.4)) * (0.9 + 0.5 * a(4)) + a(5) * TAU).sin();
    let by = s.at.1 + fall * tb + s.size * 0.4;
    let u = d / life;
    let ray = s.size * (0.08 + 0.08 * r(2)) * (1.0 - 0.5 * u);
    Some(Sprite {
        at: (bx + (r(3) - 0.5) * s.size * 0.8, by + s.size * 0.5 * d),
        half: (ray, ray),
        alpha: (1.0 - u) * (0.6 + 0.4 * (d * 20.0 + r(4) * TAU).sin()) * fade.max(0.3),
        rgb: mix(s.rgb, hue(r(5)), 0.25),
        look: Look::Sparkle,
        turn: r(6) * TAU,
    })
}

/// Quad `q` of a jackpot `s`, `age` seconds on: shot up and out from the
/// meter, arcing down under gravity, slowing a little; coins spinning
/// (`turn`: the spin, their width its cosine), every fourth quad a star.
pub fn coin(s: &Spray, q: u32, age: f32) -> Option<Sprite> {
    let r = |j: u32| rand(s.seed, q, j);
    let delay = r(0) * 0.25;
    let t = age - delay;
    let life = JACKPOT_LIFE - 0.25 - 0.5 * r(1);
    if t < 0.0 || t >= life {
        return None;
    }
    let a = -TAU / 4.0 + (r(2) - 0.5) * TAU * 0.55;
    let speed = s.size * (9.0 + 9.0 * r(3));
    let g = s.size * 22.0;
    let drag = (1.0 - (-0.8 * t).exp()) / 0.8;
    let star = q % 4 == 3;
    let size = s.size
        * if star {
            0.45 + 0.3 * r(4)
        } else {
            0.35 + 0.2 * r(4)
        };
    let u = t / life;
    Some(Sprite {
        at: (
            s.at.0 + a.cos() * speed * drag,
            s.at.1 + a.sin() * speed * drag + 0.5 * g * t * t,
        ),
        half: (size, size),
        alpha: ((1.0 - u) * 4.0).min(1.0),
        rgb: if star {
            mix(s.rgb, [1.0; 3], 0.6)
        } else {
            s.rgb
        },
        look: if star { Look::Sparkle } else { Look::Coin },
        turn: r(5) * TAU + (6.0 + 8.0 * r(6)) * t,
    })
}

/// [`angel`] and [`coin`] in WGSL, and how an angel and a coin are drawn:
/// for the particles' shader (which gives them `r`, `hue`, `Sp`).
pub fn wgsl() -> String {
    format!(
        "
const ANGELS_LIFE: f32 = {ANGELS_LIFE:?};
const ANGELS: u32 = {ANGELS}u;
const JACKPOT_LIFE: f32 = {JACKPOT_LIFE:?};
const LOOK_ANGEL: u32 = {}u;
const LOOK_COIN: u32 = {}u;
{SHADER}",
        Look::Angel.code(),
        Look::Coin.code()
    )
}

const SHADER: &str = r"
fn angel(at: vec4<f32>, rgb: vec3<f32>, size: f32, seed: u32, q: u32, age: f32) -> Sp {
    var o: Sp;
    let k = q % ANGELS;
    let delay = r(seed, k, 0u) * 1.4;
    let t = (age - delay) / (ANGELS_LIFE - 1.4);
    if t < 0.0 || t >= 1.0 {
        return o;
    }
    let x0 = at.x + (r(seed, k, 1u) * 2.0 - 1.0) * at.z;
    let fall = at.w * (0.45 + 0.35 * r(seed, k, 2u));
    let sway = size * (0.6 + 0.6 * r(seed, k, 3u));
    let ax = x0 + sway * sin(age * (0.9 + 0.5 * r(seed, k, 4u)) + r(seed, k, 5u) * TAU);
    let ay = at.y + fall * t + size * 0.4 * sin(age * 1.7 + r(seed, k, 6u) * TAU) * 0.3;
    let fade = min(t * 6.0, 1.0) * min((1.0 - t) * 3.0, 1.0);
    if q < ANGELS {
        o.at = vec2<f32>(ax, ay);
        o.half = vec2<f32>(size * 0.5);
        o.alpha = fade;
        o.rgb = rgb;
        o.look = LOOK_ANGEL;
        o.turn = age * (7.0 + 2.0 * r(seed, k, 7u));
        o.shows = 1u;
        return o;
    }
    let born = r(seed, q, 0u) * 0.9;
    let d = max(age - delay - born * (ANGELS_LIFE - 1.4), 0.0);
    let life = 0.9 + 0.8 * r(seed, q, 1u);
    if d <= 0.0 || d >= life {
        return o;
    }
    let bx = x0 + sway * sin((delay + born * (ANGELS_LIFE - 1.4)) * (0.9 + 0.5 * r(seed, k, 4u)) + r(seed, k, 5u) * TAU);
    let by = at.y + fall * born + size * 0.4;
    let u = d / life;
    let ray = size * (0.08 + 0.08 * r(seed, q, 2u)) * (1.0 - 0.5 * u);
    o.at = vec2<f32>(bx + (r(seed, q, 3u) - 0.5) * size * 0.8, by + size * 0.5 * d);
    o.half = vec2<f32>(ray);
    o.alpha = (1.0 - u) * (0.6 + 0.4 * sin(d * 20.0 + r(seed, q, 4u) * TAU)) * max(fade, 0.3);
    o.rgb = mix(rgb, hue(r(seed, q, 5u)), 0.25);
    o.look = LOOK_SPARKLE;
    o.turn = r(seed, q, 6u) * TAU;
    o.shows = 1u;
    return o;
}

fn coin(at: vec4<f32>, rgb: vec3<f32>, size0: f32, seed: u32, q: u32, age: f32) -> Sp {
    var o: Sp;
    let delay = r(seed, q, 0u) * 0.25;
    let t = age - delay;
    let life = JACKPOT_LIFE - 0.25 - 0.5 * r(seed, q, 1u);
    if t < 0.0 || t >= life {
        return o;
    }
    let a = -TAU / 4.0 + (r(seed, q, 2u) - 0.5) * TAU * 0.55;
    let speed = size0 * (9.0 + 9.0 * r(seed, q, 3u));
    let g = size0 * 22.0;
    let drag = (1.0 - exp(-0.8 * t)) / 0.8;
    let star = q % 4u == 3u;
    var size = size0 * (0.35 + 0.2 * r(seed, q, 4u));
    o.rgb = rgb;
    o.look = LOOK_COIN;
    if star {
        size = size0 * (0.45 + 0.3 * r(seed, q, 4u));
        o.rgb = mix(rgb, vec3<f32>(1.0), 0.6);
        o.look = LOOK_SPARKLE;
    }
    let u = t / life;
    o.at = vec2<f32>(at.x + cos(a) * speed * drag, at.y + sin(a) * speed * drag + 0.5 * g * t * t);
    o.half = vec2<f32>(size);
    o.alpha = min((1.0 - u) * 4.0, 1.0);
    o.turn = r(seed, q, 5u) * TAU + (6.0 + 8.0 * r(seed, q, 6u)) * t;
    o.shows = 1u;
    return o;
}

// An angel: a halo over its head, a robe below, two wings beating
// (`turn` their phase), glowing; light.
fn draw_angel(p: vec2<f32>, c: vec2<f32>, half: vec2<f32>, turn: f32, look: vec4<f32>) -> vec4<f32> {
    let s = max(half.x, 1.0);
    let l = (p - c) / s;
    let head = length(l - vec2<f32>(0.0, -0.35)) - 0.16;
    let halo = abs(length((l - vec2<f32>(0.0, -0.62)) * vec2<f32>(1.0, 3.2)) - 0.2) - 0.035;
    // The robe: a rounded triangle from the shoulders to the hem.
    let rb = l - vec2<f32>(0.0, 0.25);
    let robe = max(abs(rb.x) - (0.1 + 0.32 * clamp(rb.y + 0.45, 0.0, 1.0)), abs(rb.y) - 0.42);
    // The wings: ellipses off the shoulders, lifting and falling.
    let beat = 0.45 + 0.35 * sin(turn);
    let wx = abs(l.x) - 0.38;
    let wy = l.y + 0.05;
    let ca = cos(beat);
    let sa = sin(beat);
    let w = vec2<f32>(ca * wx + sa * wy, -sa * wx + ca * wy);
    let wing = length(w / vec2<f32>(0.42, 0.17)) - 1.0;
    // Each part its own brightness, so the figure reads through the bloom:
    // the head brightest, the robe, the wings softer and feathered at
    // their tips.
    let feather = 0.75 + 0.25 * sin(atan2(w.y, w.x) * 9.0);
    let parts = clamp(0.5 - head * s, 0.0, 1.0) * 0.9
        + clamp(0.5 - robe * s, 0.0, 1.0) * 0.6
        + clamp(0.5 - wing * 0.16 * s, 0.0, 1.0) * 0.38 * feather;
    let body = min(min(head, robe), wing * 0.16);
    let ring = clamp(0.5 - halo * s, 0.0, 1.0);
    let glow = exp(-max(min(body, halo), 0.0) * 7.0) * 0.18;
    let gold = vec3<f32>(1.0, 0.8, 0.32);
    let light = look.rgb * (min(parts, 0.9) + glow) + gold * ring * 1.3;
    return vec4<f32>(out_rgb(light) * look.a, 0.0);
}

// A coin, spinning (`turn`): its width the spin's cosine, gold with a
// bright rim and a star struck on it, its face lit as it turns; light.
fn draw_coin(p: vec2<f32>, c: vec2<f32>, half: vec2<f32>, turn: f32, look: vec4<f32>) -> vec4<f32> {
    let s = max(half.x, 1.0);
    let squash = max(abs(cos(turn)), 0.08);
    let l = (p - c) / s;
    let e = vec2<f32>(l.x / squash, l.y);
    let d = (length(e) - 1.0) * s * squash;
    let cover = clamp(0.5 - d, 0.0, 1.0);
    let rim = clamp(1.0 - abs(length(e) - 0.82) * 8.0, 0.0, 1.0);
    // A five-pointed star on its face.
    let an = atan2(e.y, e.x);
    let star_r = 0.38 + 0.18 * cos(5.0 * an);
    let star = clamp((star_r - length(e)) * 10.0, 0.0, 1.0);
    let face = 0.75 + 0.5 * sin(turn * 2.0);
    let col = look.rgb * (face + 0.6 * rim + 0.5 * star) * vec3<f32>(1.0, 0.92, 0.7);
    let glow = exp(-max(d, 0.0) / (s * 0.3)) * 0.15;
    return vec4<f32>(out_rgb(col * (cover * 0.9 + glow)) * look.a, 0.0);
}
";
