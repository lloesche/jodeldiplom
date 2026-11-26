//! A step up: each tier a phrase takes the singing up, a burst from the
//! playhead of stars, notes and rainbow bits that spin, arc down and fade
//! (one record a burst, [`Kind::StepUp`], worked out in the particles'
//! vertex shader); the bar under the word sung of the lyrics the stage
//! sets in the middle throws a smaller one ([`crate::gui::stage::lyrics`]).
//!
//! [`step_up`] is the shader's maths on the processor ([`wgsl`]), held to
//! it on the GPU (`crate::gui::shader_check`).

use super::{At, Layer, Live, Out, Slot, Spec};
use crate::gui::particles::{Kind, Look, Spray, Sprite};
use crate::gui::rainbow::hue;
use crate::gui::sparks::{rand, TAU};

/// How long a step up's burst lasts, at most, in seconds.
pub const STEP_LIFE: f32 = 1.4;

/// A burst from the playhead for each tier the phrase rose.
pub const SPEC: Spec = Spec {
    name: "step up",
    ident: "step_up",
    at: &[Layer::Playhead],
    draw,
    ..Spec::NONE
};

fn draw(at: &At, live: &Live, _: &mut Slot, out: &mut Out) {
    let &At::Playhead { tip, pill, pps, .. } = at else {
        return;
    };
    if !live.fired {
        return;
    }
    let clock = live.clock;
    let seed = |k: u32| (clock * 1000.0) as u32 + 7919 * k;
    let size = pill * live.strength;
    let ups = (0..live.rose).map(|k| burst(tip, size, clock, pps, seed(k)));
    out.sprays.extend(ups);
}

/// A step up's burst from `at`, `size` its pieces' scale, at `clock` (the
/// singing's time), going by at `pps` (a lane's), its pieces scattered by
/// `seed`.
pub fn burst(at: (f32, f32), size: f32, clock: f64, pps: f32, seed: u32) -> Spray {
    Spray {
        at,
        half: (0.0, 0.0),
        start: clock as f32,
        drift: -pps,
        size,
        seed,
        rgb: [1.0; 3],
        kind: Kind::StepUp,
    }
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

/// [`step_up`] in WGSL: for the particles' shader (which gives it `r`,
/// `hue`, `Sp` and the looks' numbers).
pub fn wgsl() -> String {
    format!("\nconst STEP_LIFE: f32 = {STEP_LIFE:?};\n{MOTION}")
}

const MOTION: &str = r"
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

";

/// How a note is drawn, in WGSL (its look's line,
/// [`crate::gui::particles::LOOKS`]).
pub const DRAW_NOTE: &str = r"
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
";

/// How a rainbow bit is drawn, in WGSL.
pub const DRAW_BIT: &str = r"
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
    use crate::gui::particles::QUADS;

    /// A step up's pieces: stars, notes and rainbow bits, shot out from
    /// where it went up (mostly up), arcing down, spinning, fading, their
    /// colours running round the rainbow; gone after their life.
    #[test]
    fn a_step_up_bursts_and_its_pieces_arc_and_fade() {
        let s = burst((400.0, 300.0), 40.0, 10.0, 0.0, 9);
        let mut looks = std::collections::HashSet::new();
        let mut up = 0;
        for q in 0..QUADS {
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
        assert!(up > QUADS as usize * 2 / 3, "mostly up: {up}");
    }
}
