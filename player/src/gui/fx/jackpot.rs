//! The jackpot: a phrase at the top throws a fountain of spinning gold
//! coins and stars up from the meter, arcing and raining down, and a
//! little one from its mark on the song's progress ([`Kind::Jackpot`], one
//! record each, worked out in the particles' vertex shader, drawn into
//! the light that blooms: [`crate::gui::bloom`]). The diploma throws them
//! too.
//!
//! [`coin`] is the shader's maths on the processor ([`wgsl`]), held to it
//! on the GPU (`crate::gui::shader_check`).

use super::{At, Layer, Live, Out, Slot, Spec};
use crate::gui::particles::{Kind, Look, Spray, Sprite};
use crate::gui::sparks::{mix, rand, TAU};

/// How long a jackpot lasts, in seconds.
pub const JACKPOT_LIFE: f32 = 2.4;

/// A jackpot from the meter and from the phrase's mark.
pub const SPEC: Spec = Spec {
    name: "jackpot",
    ident: "jackpot",
    at: &[Layer::Meter, Layer::Mark],
    draw,
    ..Spec::NONE
};

fn draw(at: &At, live: &Live, _: &mut Slot, out: &mut Out) {
    if !live.fired {
        return;
    }
    let s = live.strength;
    let spray = match *at {
        At::Meter { at, size } => jackpot(at, size * s, live.clock, (live.clock * 997.0) as u32),
        At::Mark { at, size, line } => jackpot(at, size * s, live.clock, line),
        _ => return,
    };
    out.sprays.push(spray);
}

/// A jackpot from `at`, its coins `size` across, at `clock`, scattered by
/// `seed`.
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

/// Quad `q` of a jackpot `s`, `age` seconds on: shot up and out, arcing
/// down under gravity, slowing a little; coins spinning (`turn`: the spin,
/// their width its cosine), every fourth quad a star.
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

/// [`coin`] in WGSL: for the particles' shader (which gives it `r`, `Sp`
/// and the looks' numbers).
pub fn wgsl() -> String {
    format!("\nconst JACKPOT_LIFE: f32 = {JACKPOT_LIFE:?};\n{MOTION}")
}

const MOTION: &str = r"
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

";

/// How a coin of light is drawn, in WGSL (its look's line,
/// [`crate::gui::particles::LOOKS`]).
pub const DRAW_COIN: &str = r"
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
