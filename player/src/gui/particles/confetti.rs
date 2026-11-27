//! Confetti on the GPU, for a new best ([`Kind::Confetti`]): a shower of
//! paper bits falling from over the top of the screen, each in its own
//! time, swaying as it falls and fluttering (turning over: its width the
//! cosine of its flip), in the rainbow's colours tinted by the screen's.
//! One record ([`Spray`]) a shower; every bit is worked out in the
//! particles' vertex shader in closed form from its seed and age, drawn as
//! a rainbow bit is ([`crate::gui::fx::step_up`]'s).
//!
//! [`piece`] is the shader's maths on the processor, held to it on the GPU
//! (`crate::gui::shader_check`).

use crate::gui::particles::{Kind, Look, Spray, Sprite};
use crate::gui::shine::hue;
use crate::gui::sparks::{rand, TAU};
use jodel_looks::math::mix3;

/// How long a shower lasts, in seconds.
pub const CONFETTI_LIFE: f32 = 4.0;

/// A shower over a screen `width` × `height` (its top at `top`) from
/// `clock`, its bits `size` long, tinted `rgb`, scattered by `seed`.
pub fn shower(
    (left, top): (f32, f32),
    (width, height): (f32, f32),
    size: f32,
    clock: f64,
    (rgb, seed): ([f32; 3], u32),
) -> Spray {
    Spray {
        at: (left + width / 2.0, top),
        half: (width / 2.0, height),
        start: clock as f32,
        drift: 0.0,
        size,
        seed,
        rgb,
        kind: Kind::Confetti,
    }
}

/// Bit `q` of shower `s`, `age` seconds on: in at its own time from over
/// the top, falling to past the bottom, swaying, fluttering, turning.
pub fn piece(s: &Spray, q: u32, age: f32) -> Option<Sprite> {
    let r = |j: u32| rand(s.seed, q, j);
    let delay = r(0) * 0.8;
    let t = age - delay;
    let life = CONFETTI_LIFE - 0.8 - 0.6 * r(1);
    if t < 0.0 || t >= life {
        return None;
    }
    let x0 = s.at.0 + (r(2) * 2.0 - 1.0) * s.band();
    let y0 = s.at.1 - s.size * (1.0 + 4.0 * r(3));
    let fall = s.screen_height() * (0.32 + 0.22 * r(4));
    let sway = s.size * (1.0 + 1.5 * r(5));
    let flip = (t * (5.0 + 6.0 * r(6)) + r(7) * TAU).cos().abs();
    let long = s.size * (0.4 + 0.3 * r(8));
    Some(Sprite {
        at: (
            x0 + sway * (t * (1.5 + 1.5 * r(9)) + r(10) * TAU).sin(),
            y0 + fall * t,
        ),
        half: (long, long * (0.08 + 0.5 * flip)),
        alpha: ((life - t) * 2.0).min(1.0) * (0.75 + 0.25 * flip),
        rgb: mix3(hue(r(11)), s.rgb, 0.3),
        look: Look::Bit,
        turn: r(12) * TAU + (r(13) - 0.5) * 4.0 * t,
    })
}

/// [`piece`] in WGSL: for the particles' shader (which gives it `r`,
/// `hue`, `Sp` and `LOOK_BIT`).
pub fn wgsl() -> String {
    format!("\nconst CONFETTI_LIFE: f32 = {CONFETTI_LIFE:?};\n{SHADER}")
}

const SHADER: &str = r"
// `at.z` its band across, `at.w` the screen's height (`Spray::band`,
// `Spray::screen_height`).
fn confetti(at: vec4<f32>, rgb: vec3<f32>, size: f32, seed: u32, q: u32, age: f32) -> Sp {
    var o: Sp;
    let delay = r(seed, q, 0u) * 0.8;
    let t = age - delay;
    let life = CONFETTI_LIFE - 0.8 - 0.6 * r(seed, q, 1u);
    if t < 0.0 || t >= life {
        return o;
    }
    let x0 = at.x + (r(seed, q, 2u) * 2.0 - 1.0) * at.z;
    let y0 = at.y - size * (1.0 + 4.0 * r(seed, q, 3u));
    let fall = at.w * (0.32 + 0.22 * r(seed, q, 4u));
    let sway = size * (1.0 + 1.5 * r(seed, q, 5u));
    let flip = abs(cos(t * (5.0 + 6.0 * r(seed, q, 6u)) + r(seed, q, 7u) * TAU));
    let long = size * (0.4 + 0.3 * r(seed, q, 8u));
    o.at = vec2<f32>(
        x0 + sway * sin(t * (1.5 + 1.5 * r(seed, q, 9u)) + r(seed, q, 10u) * TAU),
        y0 + fall * t,
    );
    o.half = vec2<f32>(long, long * (0.08 + 0.5 * flip));
    o.alpha = min((life - t) * 2.0, 1.0) * (0.75 + 0.25 * flip);
    o.rgb = mix(hue(r(seed, q, 11u)), rgb, 0.3);
    o.look = LOOK_BIT;
    o.turn = r(seed, q, 12u) * TAU + (r(seed, q, 13u) - 0.5) * 4.0 * t;
    o.shows = 1u;
    return o;
}
";

#[cfg(test)]
mod tests {
    use super::*;

    /// The bits come in over the top, fall down the screen and are gone
    /// before the shower ends; each its own way.
    #[test]
    fn confetti_falls_from_the_top() {
        let s = shower((0.0, 0.0), (1600.0, 900.0), 20.0, 10.0, ([1.0; 3], 7));
        let at = |q: u32, age: f32| piece(&s, q, age);
        let early: Vec<f32> = (0..32).filter_map(|q| at(q, 0.9)).map(|p| p.at.1).collect();
        let late: Vec<f32> = (0..32).filter_map(|q| at(q, 2.5)).map(|p| p.at.1).collect();
        assert!(early.len() > 16 && late.len() > 16);
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        assert!(mean(&early) < 300.0 && mean(&late) > mean(&early) + 300.0);
        assert!((0..32).all(|q| at(q, CONFETTI_LIFE).is_none()));
        let xs: Vec<f32> = (0..32).filter_map(|q| at(q, 1.5)).map(|p| p.at.0).collect();
        assert!(xs.iter().any(|&x| x < 400.0) && xs.iter().any(|&x| x > 1200.0));
    }
}
