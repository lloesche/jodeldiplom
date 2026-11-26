//! The liquid a word's bubble fills with as it is sung (its fill
//! shader), on the GPU: its
//! front (where the singing fills it) a living surface, a meniscus and a
//! few waves moving along it, sloshing a moment when the fill jumps (the
//! sweep to full); a shimmer moving through it; and the bubble under
//! pressure in the last stretch to full (a slight bulge and a wobble),
//! then overfilling as it bursts (bulging to 1 + [`GROW`]). The processor
//! gives only a bubble's level, pressure, wobble phase, when its fill
//! jumped and when it bursts ([`crate::gui::stage::gpu::Strain`]); all of it
//! moves by the time in the stage's shader, for the stage's bubbles and
//! the picture's alike. These functions are the reference of the shader's
//! ([`wgsl`]), held to it on the GPU (`crate::gui::shader_check`).

/// How much larger a bubble bulges as it overfills, a share of its size.
pub const GROW: f32 = 0.3;
/// How much larger it bulges at its most under pressure.
pub const BULGE: f32 = 0.07;
/// How much it wobbles on top of that.
pub const WOBBLE: f32 = 0.035;
/// How fast it wobbles, radians a second.
const WOBBLE_RATE: f32 = 23.0;

/// A bubble under `pressure` (0 to 1) at `time`, `phase` its own: how much
/// larger across and down it shows (each 1 to 1 + [`BULGE`] +
/// [`WOBBLE`]: never smaller than it is), the two wobbling in turn.
pub fn bulge(pressure: f32, time: f32, phase: f32) -> (f32, f32) {
    let p = pressure.clamp(0.0, 1.0);
    let w = WOBBLE * p;
    let a = (time * WOBBLE_RATE + phase).sin();
    (
        1.0 + BULGE * p + w * (0.5 + 0.5 * a),
        1.0 + BULGE * p + w * (0.5 - 0.5 * a),
    )
}

/// A bubble `s` (0 to 1) through its overfilling: how much larger it
/// shows, from about where its pressure left it to 1 + [`GROW`], quickly
/// at first.
pub fn swell(s: f32) -> f32 {
    let s = s.clamp(0.0, 1.0);
    let eased = 1.0 - (1.0 - s) * (1.0 - s);
    let least = 1.0 + BULGE + WOBBLE * 0.5;
    least + (1.0 + GROW - least) * eased
}

/// How much more its front moves `since` seconds after its fill jumped
/// (none before): a slosh dying away, from 2.5 times.
pub fn slosh(since: f32) -> f32 {
    if since < 0.0 {
        0.0
    } else {
        2.5 * (-since * 5.0).exp()
    }
}

/// How far ahead of its level the liquid's front is at `y` (-1 at the
/// bubble's top to 1 at its bottom) at `time`, its waves `amp` pixels, its
/// fill last jumped at `jumped`: a meniscus ahead in the middle and two
/// waves moving along it, from -0.5 to 1 times `amp` (sloshing, up to 3.5
/// times that).
pub fn front(y: f32, time: f32, amp: f32, jumped: f32) -> f32 {
    let a = amp * (1.0 + slosh(time - jumped));
    let y = y.clamp(-1.0, 1.0);
    a * (0.5 * (1.0 - y * y)
        + 0.3 * (y * 2.6 + time * 7.0).sin()
        + 0.2 * (y * 5.1 - time * 11.0).sin())
}

/// The shimmer in the liquid at (`x`, `y`) (pixels) at `time`: 0 to 1, a
/// light moving through it.
pub fn shimmer(x: f32, y: f32, time: f32) -> f32 {
    let a = 0.5 + 0.5 * (x * 0.09 - time * 3.1 + y * 0.35).sin();
    let b = 0.5 + 0.5 * (x * 0.05 + time * 1.7 - y * 0.2).sin();
    (a * b).powi(3)
}

/// How high a bubble's front's waves go, it `half_y` half high.
pub fn amplitude(half_y: f32) -> f32 {
    (half_y * 0.35).max(1.0)
}

/// A bubble of `seed`'s wobble's own phase, radians.
pub fn phase(seed: u32) -> f32 {
    ((seed & 0xFF_FFFF) as f32 * 0.618).fract() * std::f32::consts::TAU
}

/// [`bulge`], [`swell`], [`slosh`], [`front`], [`shimmer`] and
/// [`amplitude`] in WGSL, and `strained`: how a bubble's strain shows at a
/// moment (larger across and down; 0 once it has torn).
pub fn wgsl() -> String {
    format!(
        "
const SWELL: f32 = {:?};
const GROW: f32 = {GROW:?};
const BULGE: f32 = {BULGE:?};
const WOBBLE: f32 = {WOBBLE:?};
const WOBBLE_RATE: f32 = {WOBBLE_RATE:?};
const POPPING: u32 = {}u;
const FRONT: u32 = {}u;
{SHADER}",
        crate::view::bubbles::SWELL as f32,
        crate::gui::stage::gpu::flags::POPPING,
        crate::gui::stage::gpu::flags::FRONT,
    )
}

const SHADER: &str = r"
fn bulge(pressure: f32, time: f32, phase: f32) -> vec2<f32> {
    let p = clamp(pressure, 0.0, 1.0);
    let w = WOBBLE * p;
    let a = sin(time * WOBBLE_RATE + phase);
    return vec2<f32>(1.0 + BULGE * p + w * (0.5 + 0.5 * a), 1.0 + BULGE * p + w * (0.5 - 0.5 * a));
}

fn swell(s0: f32) -> f32 {
    let s = clamp(s0, 0.0, 1.0);
    let eased = 1.0 - (1.0 - s) * (1.0 - s);
    let least = 1.0 + BULGE + WOBBLE * 0.5;
    return least + (1.0 + GROW - least) * eased;
}

fn slosh(since: f32) -> f32 {
    if since < 0.0 {
        return 0.0;
    }
    return 2.5 * exp(-since * 5.0);
}

fn front(y0: f32, time: f32, amp: f32, jumped: f32) -> f32 {
    let a = amp * (1.0 + slosh(time - jumped));
    let y = clamp(y0, -1.0, 1.0);
    return a * (0.5 * (1.0 - y * y) + 0.3 * sin(y * 2.6 + time * 7.0) + 0.2 * sin(y * 5.1 - time * 11.0));
}

fn shimmer(x: f32, y: f32, time: f32) -> f32 {
    let a = 0.5 + 0.5 * sin(x * 0.09 - time * 3.1 + y * 0.35);
    let b = 0.5 + 0.5 * sin(x * 0.05 + time * 1.7 - y * 0.2);
    let ab = a * b;
    return ab * ab * ab;
}

fn amplitude(half_y: f32) -> f32 {
    return max(half_y * 0.35, 1.0);
}

// How a bubble strains at `time` (`strain`: its pressure, when it bursts,
// its phase, its flags): how much larger across and down, and 1 while it
// shows (0 once it has torn).
fn strained(strain: vec4<f32>, time: f32) -> vec3<f32> {
    let flags = u32(strain.w);
    if (flags & POPPING) != 0u {
        let s = (time - strain.y) / SWELL;
        if s >= 1.0 {
            return vec3<f32>(0.0);
        }
        return vec3<f32>(swell(s), swell(s), 1.0);
    }
    return vec3<f32>(bulge(strain.x, time, strain.z), 1.0);
}
";

#[cfg(test)]
mod tests {
    use super::*;

    /// Under pressure a bubble bulges a little and wobbles, never smaller
    /// than it is; without, it is as it is.
    #[test]
    fn a_bubble_under_pressure_bulges_and_wobbles() {
        assert_eq!(bulge(0.0, 3.0, 1.0), (1.0, 1.0));
        let mut seen = Vec::new();
        for k in 0..60 {
            let (x, y) = bulge(1.0, k as f32 / 60.0, 0.4);
            assert!((1.0..=1.0 + BULGE + WOBBLE + 1e-6).contains(&x), "{x}");
            assert!((1.0..=1.0 + BULGE + WOBBLE + 1e-6).contains(&y), "{y}");
            seen.push(x - y);
        }
        let (lo, hi) = seen
            .iter()
            .fold((0.0f32, 0.0f32), |m, &v| (m.0.min(v), m.1.max(v)));
        assert!(lo < -0.02 && hi > 0.02, "it wobbles: {lo} {hi}");
        // Overfilling: from about there to 1.3, quickly at first.
        assert!((swell(0.0) - (1.0 + BULGE + WOBBLE * 0.5)).abs() < 1e-6);
        assert!((swell(1.0) - 1.3).abs() < 1e-6);
        assert!(swell(0.5) - swell(0.0) > swell(1.0) - swell(0.5));
    }

    /// The front moves with the time, within its waves; a jump sloshes it
    /// more, a moment.
    #[test]
    fn the_front_ripples_and_sloshes_after_a_jump() {
        let amp = 2.0;
        let at = |y: f32, t: f32, jumped: f32| front(y, t, amp, jumped);
        let calm: Vec<f32> = (0..30).map(|k| at(0.3, k as f32 / 30.0, -9.0)).collect();
        assert!(calm.iter().all(|f| (-0.5 * amp..=amp).contains(f)));
        let spread = calm.iter().fold(f32::MIN, |m, &v| m.max(v))
            - calm.iter().fold(f32::MAX, |m, &v| m.min(v));
        assert!(spread > 0.5 * amp, "it moves: {spread}");
        // Along it too.
        assert!((at(-0.8, 0.2, -9.0) - at(0.4, 0.2, -9.0)).abs() > 0.05);
        assert!(slosh(0.02) > 2.0 && slosh(0.6) < 0.2 && slosh(-0.1) == 0.0);
        let s = (0..50).map(|k| shimmer(k as f32 * 3.0, 2.0, 0.5));
        let (lo, hi) = s.fold((1.0f32, 0.0f32), |m, v| (m.0.min(v), m.1.max(v)));
        assert!(lo >= 0.0 && hi <= 1.0 && hi - lo > 0.2);
    }
}
