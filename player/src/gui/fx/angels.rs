//! Angels descending while the singing is at the top: every while a
//! flight of them drifts down the screen, swaying, wings beating, star
//! dust trailing ([`Kind::Angels`], one record a flight, worked out in the
//! particles' vertex shader, drawn into the light that blooms:
//! [`crate::gui::bloom`]). A flight comes as soon as the singing is to be
//! at the top (its glow's aim, not yet its eased strength), then one every
//! [`ANGELS_EVERY`]; sought back before the last, the next comes at once.
//!
//! [`angel`] is the shader's maths on the processor ([`wgsl`]), held to it
//! on the GPU (`crate::gui::shader_check`).

use super::{At, Layer, Live, Out, Slot, Spec};
use crate::gui::particles::{Kind, Look, Spray, Sprite};
use crate::gui::rainbow::hue;
use crate::gui::sparks::{mix, rand, TAU};

/// How long a flight of angels takes to come down, in seconds.
pub const ANGELS_LIFE: f32 = 6.0;
/// The angels in a flight (its first quads; the rest are their dust).
pub const ANGELS: u32 = 5;
/// How often a new flight comes while the singing is at the top.
const ANGELS_EVERY: f32 = 2.2;

/// A flight of angels every while at the top.
pub const SPEC: Spec = Spec {
    name: "angels",
    ident: "angels",
    at: &[Layer::Sky],
    draw,
    ..Spec::NONE
};

fn draw(at: &At, live: &Live, slot: &mut Slot, out: &mut Out) {
    let &At::Sky { area: (_, _, w, h) } = at else {
        return;
    };
    let clock = live.clock as f32;
    if live.aim > 0.0 && clock - slot.last >= ANGELS_EVERY {
        slot.last = clock;
        let seed = (live.clock * 1000.0) as u32 ^ 0x5EED;
        out.sprays.push(flight(
            w,
            h * 0.75,
            h * 0.12 * live.listed,
            live.clock,
            seed,
        ));
    }
}

/// A flight of angels over a screen `width` × `height` from `clock` (the
/// singing's time), scattered by `seed`, each `size` tall.
pub fn flight(width: f32, height: f32, size: f32, clock: f64, seed: u32) -> Spray {
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
    let x0 = s.at.0 + (a(1) * 2.0 - 1.0) * s.band();
    let fall = s.screen_height() * (0.45 + 0.35 * a(2));
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

/// [`angel`] in WGSL: for the particles' shader (which gives it `r`,
/// `hue`, `Sp` and the looks' numbers).
pub fn wgsl() -> String {
    format!("\nconst ANGELS_LIFE: f32 = {ANGELS_LIFE:?};\nconst ANGELS: u32 = {ANGELS}u;\n{MOTION}")
}

const MOTION: &str = r"
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

";

/// How an angel is drawn, in WGSL (its look's line,
/// [`crate::gui::particles::LOOKS`]).
pub const DRAW_ANGEL: &str = r"
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
";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::fx::{Board, Fx, Music};
    use crate::gui::rainbow::Glow;
    use crate::gui::streak::Step;
    use crate::tiers::Tier;

    /// At the top a flight comes at once, then one every while (none in
    /// between); below it none; sought back, the next at once.
    #[test]
    fn a_flight_comes_every_while_at_the_top_and_at_once_after_a_seek() {
        let mut board = Board::default();
        let top = Step {
            tier: Some(Tier::TOP),
            glow: Glow {
                level: 1.0,
                from: 1.0,
                since: 0.0,
            },
            ..Step::default()
        };
        let below = Step {
            tier: Some(Tier::ALL[0]),
            glow: Glow {
                level: Tier::ALL[0].spec().glow,
                from: 0.0,
                since: 0.0,
            },
            ..Step::default()
        };
        let mut flights = |step: &Step, clock: f64| {
            let frame = board.frame(step, &Music::NEUTRAL, clock);
            let mut sprays = Vec::new();
            let sky = At::Sky {
                area: (0.0, 0.0, 1600.0, 900.0),
            };
            board.sprays(&sky, &frame, &mut sprays);
            assert!(sprays.iter().all(|s| s.kind == Kind::Angels));
            sprays.len()
        };
        assert_eq!(flights(&below, 3.0), 0, "below the top");
        assert_eq!(flights(&top, 3.0), 1, "at once");
        assert_eq!(flights(&top, 4.0), 0);
        assert_eq!(flights(&top, 3.0 + f64::from(ANGELS_EVERY) + 0.05), 1);
        assert_eq!(flights(&top, 6.0), 0);
        assert_eq!(flights(&top, 2.0), 1, "sought back: at once");
        assert!(Fx::Angels.spec().at.contains(&Layer::Sky));
    }
}
