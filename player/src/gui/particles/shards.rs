//! A bubble bursting on the GPU, a particle effect
//! ([`crate::gui::particles`]), all of it at once,
//! once it has overfilled (its overfilling is the bubble's own, drawn with
//! it: [`crate::gui::stage::shapes::liquid`]): it tears, its liquid splashing
//! out as [`DROPS`] drops of a few sizes in its colour that arc out and fall,
//! stretched along their way, fading; [`SHARDS`] torn bits of its skin fly off,
//! spinning a little; a wet sheen fades where it was ([`AFTERGLOW`]); then
//! nothing. [`sheen`], [`shard`] and [`drop`] are the processor's maths,
//! the reference of the shader's ([`wgsl`]), held to it on the GPU
//! (`crate::gui::shader_check`).

use crate::gui::particles::{Look, Spray, Sprite, BUBBLE_SPARKS};
use crate::gui::sparks::{rand, TAU, WHITE};
use crate::gui::stage::shapes::liquid::GROW;
use crate::view::bubbles::{BURST, SWELL};
use jodel_looks::math::mix3;

/// How long the wet sheen a bubble leaves where it was fades once it
/// tears, in seconds.
pub const AFTERGLOW: f32 = 0.2;
/// The torn bits of skin a bubble throws.
pub const SHARDS: u32 = 6;
/// The drops of its liquid.
pub const DROPS: u32 = 12;
/// The first of its quads that is a bit of skin.
pub const FIRST_SHARD: u32 = 1 + BUBBLE_SPARKS;
/// The first that is a drop.
pub const FIRST_DROP: u32 = FIRST_SHARD + SHARDS;

/// The sheen bubble `s` leaves `age` seconds into its burst, once it has
/// torn: where it was, as large as it overfilled to, fading; none before or
/// after.
pub fn sheen(s: &Spray, age: f32) -> Option<Sprite> {
    let g = (age - SWELL as f32) / AFTERGLOW;
    if !(0.0..1.0).contains(&g) {
        return None;
    }
    let big = 1.0 + GROW;
    Some(Sprite {
        at: (s.at.0 + s.drift * age, s.at.1),
        half: (s.half.0 * big, s.half.1 * big),
        alpha: 0.3 * (1.0 - g) * (1.0 - g),
        rgb: mix3(s.rgb, WHITE, 0.4),
        look: Look::Sheen,
        turn: 0.0,
    })
}

/// Bit `q` of bubble `s`'s torn skin `age` seconds into its burst: from
/// its place on the swollen skin, out the way it faces, slowing, turning a
/// little, shrinking and fading; none before it tears or after.
pub fn shard(s: &Spray, q: u32, age: f32) -> Option<Sprite> {
    let (swell, life) = (SWELL as f32, BURST as f32);
    if age < swell || age >= life {
        return None;
    }
    let t = (age - swell) / (life - swell);
    let r = |j: u32| rand(s.seed, q, j);
    let k = (q - FIRST_SHARD) as f32;
    let theta = (k + 0.5 + 0.6 * (r(0) - 0.5)) / SHARDS as f32 * TAU;
    let (dx, dy) = (theta.cos(), theta.sin());
    let (x, y, hw, hh) = (s.at.0 + s.drift * age, s.at.1, s.half.0, s.half.1);
    let big = 1.0 + GROW;
    let fly = s.size * (0.6 + 0.5 * r(1)) * (1.0 - (1.0 - t).powf(3.5));
    let rim = (4.0 * (hw - hh).max(0.0) + TAU * hh) * big;
    let len = (rim / SHARDS as f32 * 0.7).clamp(s.size * 0.08, s.size * 0.35) * (1.0 - 0.35 * t);
    let least = (s.size * 0.025).max(1.2);
    let thick = (hh * big * 0.2).clamp(least, (s.size * 0.05).max(least)) * (1.0 - 0.3 * t);
    Some(Sprite {
        at: (x + dx * (hw * big + fly), y + dy * (hh * big + fly)),
        half: (len / 2.0, thick),
        alpha: 0.85 * (1.0 - t).powf(1.4),
        rgb: mix3(s.rgb, WHITE, 0.75),
        look: Look::Shard,
        turn: theta + TAU / 4.0 + (r(2) - 0.5) * 3.0 * t,
    })
}

/// Drop `q` of bubble `s`'s liquid `age` seconds into its burst: from
/// inside the swollen bubble, out and up, falling (an arc), stretched
/// along its way, fading; of a few sizes; none before it tears or after.
pub fn drop(s: &Spray, q: u32, age: f32) -> Option<Sprite> {
    let (swell, life) = (SWELL as f32, BURST as f32);
    if age < swell || age >= life {
        return None;
    }
    let ta = age - swell;
    let t = ta / (life - swell);
    let r = |j: u32| rand(s.seed, q, j);
    let (x, y, hw, hh) = (s.at.0 + s.drift * age, s.at.1, s.half.0, s.half.1);
    let big = 1.0 + GROW;
    let (u, v) = (r(0) * 2.0 - 1.0, r(1) * 2.0 - 1.0);
    let (sx, sy) = (x + u * hw * big * 0.85, y + v * hh * big * 0.6);
    let speed = s.size * (2.2 + 1.8 * r(4));
    let vx = (u + (r(2) - 0.5) * 0.8) * speed;
    let vy = (-(0.5 + 0.7 * r(3)) + v * 0.3) * speed;
    let g = s.size * 12.0;
    let (wx, wy) = (vx, vy + g * ta);
    let sort = r(5);
    let share = if sort < 0.5 {
        0.05
    } else if sort < 0.85 {
        0.08
    } else {
        0.12
    };
    let rad = (s.size * share * (1.0 - 0.3 * t)).max(1.2);
    let stretch = 1.0 + ((wx * wx + wy * wy).sqrt() / (s.size * 4.0)).min(1.2);
    Some(Sprite {
        at: (sx + vx * ta, sy + vy * ta + 0.5 * g * ta * ta),
        half: (rad * stretch, rad),
        alpha: (1.0 - t).powf(0.9),
        rgb: mix3(s.rgb, WHITE, 0.25 * (1.0 - t)),
        look: Look::Drop,
        turn: wy.atan2(wx),
    })
}

/// [`sheen`], [`shard`] and [`drop`] in WGSL: for the particles' shader
/// (a bubble's burst, [`crate::gui::particles::MOTIONS`]).
pub fn wgsl() -> String {
    format!(
        "
const SWELL: f32 = {:?};
const BURST: f32 = {:?};
const GROW: f32 = {GROW:?};
const AFTERGLOW: f32 = {AFTERGLOW:?};
const SHARDS: f32 = {:?};
const FIRST_SHARD: u32 = {FIRST_SHARD}u;
const FIRST_DROP: u32 = {FIRST_DROP}u;
{MOTION}",
        SWELL as f32, BURST as f32, SHARDS as f32
    )
}

const MOTION: &str = r"
fn sheen(at: vec4<f32>, rgb: vec3<f32>, age: f32, drift: f32) -> Sp {
    var o: Sp;
    let g = (age - SWELL) / AFTERGLOW;
    if g < 0.0 || g >= 1.0 {
        return o;
    }
    o.at = vec2<f32>(at.x + drift, at.y);
    o.half = at.zw * (1.0 + GROW);
    o.alpha = 0.3 * (1.0 - g) * (1.0 - g);
    o.rgb = mix(rgb, vec3<f32>(1.0), 0.4);
    o.look = LOOK_SHEEN;
    o.shows = 1u;
    return o;
}

fn shard(at: vec4<f32>, rgb: vec3<f32>, size: f32, seed: u32, q: u32, age: f32, drift: f32) -> Sp {
    var o: Sp;
    if age < SWELL || age >= BURST {
        return o;
    }
    let t = (age - SWELL) / (BURST - SWELL);
    let k = f32(q - FIRST_SHARD);
    let theta = (k + 0.5 + 0.6 * (r(seed, q, 0u) - 0.5)) / SHARDS * TAU;
    let d = vec2<f32>(cos(theta), sin(theta));
    let big = 1.0 + GROW;
    let fly = size * (0.6 + 0.5 * r(seed, q, 1u)) * (1.0 - pow(1.0 - t, 3.5));
    let rim = (4.0 * max(at.z - at.w, 0.0) + TAU * at.w) * big;
    let len = clamp(rim / SHARDS * 0.7, size * 0.08, size * 0.35) * (1.0 - 0.35 * t);
    let least = max(size * 0.025, 1.2);
    let thick = clamp(at.w * big * 0.2, least, max(size * 0.05, least)) * (1.0 - 0.3 * t);
    o.at = vec2<f32>(at.x + drift, at.y) + d * (at.zw * big + vec2<f32>(fly));
    o.half = vec2<f32>(len / 2.0, thick);
    o.alpha = 0.85 * pow(1.0 - t, 1.4);
    o.rgb = mix(rgb, vec3<f32>(1.0), 0.75);
    o.look = LOOK_SHARD;
    o.turn = theta + TAU / 4.0 + (r(seed, q, 2u) - 0.5) * 3.0 * t;
    o.shows = 1u;
    return o;
}

fn drop(at: vec4<f32>, rgb: vec3<f32>, size: f32, seed: u32, q: u32, age: f32, drift: f32) -> Sp {
    var o: Sp;
    if age < SWELL || age >= BURST {
        return o;
    }
    let ta = age - SWELL;
    let t = ta / (BURST - SWELL);
    let big = 1.0 + GROW;
    let u = r(seed, q, 0u) * 2.0 - 1.0;
    let v = r(seed, q, 1u) * 2.0 - 1.0;
    let s0 = vec2<f32>(at.x + drift + u * at.z * big * 0.85, at.y + v * at.w * big * 0.6);
    let speed = size * (2.2 + 1.8 * r(seed, q, 4u));
    let vel = vec2<f32>(
        (u + (r(seed, q, 2u) - 0.5) * 0.8) * speed,
        (-(0.5 + 0.7 * r(seed, q, 3u)) + v * 0.3) * speed,
    );
    let g = size * 12.0;
    let w = vec2<f32>(vel.x, vel.y + g * ta);
    let sort = r(seed, q, 5u);
    var share = 0.12;
    if sort < 0.5 {
        share = 0.05;
    } else if sort < 0.85 {
        share = 0.08;
    }
    let rad = max(size * share * (1.0 - 0.3 * t), 1.2);
    let stretch = 1.0 + min(length(w) / (size * 4.0), 1.2);
    o.at = s0 + vel * ta + vec2<f32>(0.0, 0.5 * g * ta * ta);
    o.half = vec2<f32>(rad * stretch, rad);
    o.alpha = pow(1.0 - t, 0.9);
    o.rgb = mix(rgb, vec3<f32>(1.0), 0.25 * (1.0 - t));
    o.look = LOOK_DROP;
    o.turn = atan2(w.y, w.x);
    o.shows = 1u;
    return o;
}

";

/// How a sheen, a bit of skin and a drop are drawn, in WGSL (their looks'
/// lines, [`crate::gui::particles::LOOKS`]).
pub const DRAW_SHEEN: &str = r"
// The wet sheen a bubble leaves: a soft light where it was, added.
fn draw_sheen(p: vec2<f32>, c: vec2<f32>, half: vec2<f32>, turn: f32, look: vec4<f32>) -> vec4<f32> {
    let d = round_box(p - c, half, min(half.x, half.y));
    let inside = clamp(0.5 - d, 0.0, 1.0);
    let s = max(d, 0.0) / (half.y * 0.8 + 1.0);
    let soft = exp(-s * s * 3.0) * (1.0 - smoothstep(0.75, 1.0, s));
    return vec4<f32>(out_rgb(look.rgb) * (inside * 0.6 + soft * 0.4) * look.a, 0.0);
}
";

/// [`DRAW_SHEEN`]'s for a bit of skin.
pub const DRAW_SHARD: &str = r"
// A bit of skin: an arc (bulging out, -y turned), half.x half its length,
// half.y half its thickness, over what is under it, a little glow.
fn draw_shard(p: vec2<f32>, c: vec2<f32>, half: vec2<f32>, turn: f32, look: vec4<f32>) -> vec4<f32> {
    let s = sin(turn);
    let co = cos(turn);
    let q0 = p - c;
    let l = vec2<f32>(co * q0.x + s * q0.y, -s * q0.x + co * q0.y);
    let radius = max(half.x * 1.2, 1.0);
    let rc = l - vec2<f32>(0.0, radius);
    let along = atan2(rc.x, -rc.y);
    let d = max(abs(length(rc) - radius) - half.y, (abs(along) - half.x / radius) * radius);
    let cover = clamp(0.5 - d, 0.0, 1.0);
    let g = max(d, 0.0) / (half.y * 2.0 + 1.0);
    let glow = exp(-g * g * 2.0) * 0.3;
    let rgb = out_rgb(look.rgb);
    return vec4<f32>(rgb * (cover + glow) * look.a, cover * look.a);
}
";

/// [`DRAW_SHEEN`]'s for a drop.
pub const DRAW_DROP: &str = r"
// A drop: an ellipse along its way (`turn`), lit at its front, over what
// is under it.
fn draw_drop(p: vec2<f32>, c: vec2<f32>, half: vec2<f32>, turn: f32, look: vec4<f32>) -> vec4<f32> {
    let s = sin(turn);
    let co = cos(turn);
    let q0 = p - c;
    let l = vec2<f32>(co * q0.x + s * q0.y, -s * q0.x + co * q0.y);
    let e = length(l / max(half, vec2<f32>(0.5)));
    let d = (e - 1.0) * half.y;
    let cover = clamp(0.5 - d, 0.0, 1.0);
    let shine = clamp(1.0 - length(l / max(half, vec2<f32>(0.5)) - vec2<f32>(0.35, -0.35)) * 1.6, 0.0, 1.0);
    let rgb = out_rgb(min(look.rgb * (0.9 + 0.3 * (1.0 - e)) + vec3<f32>(shine * 0.5), vec3<f32>(1.0)));
    return vec4<f32>(rgb * cover * look.a, cover * look.a);
}
";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::particles::{sprite, Kind};

    fn bubble() -> Spray {
        Spray {
            at: (400.0, 300.0),
            half: (30.0, 4.0),
            start: 10.0,
            drift: 0.0,
            size: 60.0,
            seed: 7,
            rgb: [1.0, 0.75, 0.24],
            kind: Kind::Burst,
        }
    }

    /// Torn, a bubble leaves only a wet sheen where it was, as large as it
    /// overfilled to, fading, gone by [`AFTERGLOW`] after; none while it
    /// overfills (the bubble itself shows then).
    #[test]
    fn a_bubble_leaves_a_fading_sheen() {
        let s = bubble();
        let at = |t: f32| sprite(&s, 0, 10.0 + t);
        assert!(at(0.02).is_none(), "the bubble itself while it overfills");
        let (a, b) = (
            at(SWELL as f32 + 0.02).unwrap(),
            at(SWELL as f32 + 0.15).unwrap(),
        );
        assert!(a.look == Look::Sheen && b.look == Look::Sheen);
        assert!((a.half.0 / 30.0 - 1.3).abs() < 1e-4);
        assert!(b.alpha < a.alpha && a.alpha < 0.35);
        assert!(at(SWELL as f32 + AFTERGLOW + 0.01).is_none());
    }

    /// The bits of skin: none while it overfills; then each flies out from
    /// the bubble, slowing, and fades; none after the burst's life; every
    /// way round.
    #[test]
    fn the_bits_of_skin_fly_out_and_fade() {
        let s = bubble();
        let at = |q: u32, t: f32| sprite(&s, q, 10.0 + t);
        let away = |p: &Sprite| (p.at.0 - 400.0).hypot(p.at.1 - 300.0);
        let mut ways = Vec::new();
        for q in FIRST_SHARD..FIRST_DROP {
            assert!(at(q, 0.05).is_none(), "not while it overfills");
            assert!(at(q, BURST as f32).is_none(), "gone after");
            let (a, b, c) = (
                at(q, 0.1).unwrap(),
                at(q, 0.25).unwrap(),
                at(q, 0.45).unwrap(),
            );
            assert!(a.look == Look::Shard);
            assert!(away(&a) < away(&b) && away(&b) < away(&c), "out: {q}");
            let step = |p: &Sprite, n: &Sprite| (n.at.0 - p.at.0).hypot(n.at.1 - p.at.1);
            assert!(step(&b, &c) < step(&a, &b), "slowing: {q}");
            assert!(c.alpha < 0.3 * a.alpha, "fading: {q}");
            assert!((c.turn - a.turn).abs() < 1.6, "a little spin: {q}");
            ways.push((c.at.1 - 300.0).atan2(c.at.0 - 400.0));
        }
        ways.sort_by(f32::total_cmp);
        let widest = ways
            .windows(2)
            .map(|w| w[1] - w[0])
            .chain([ways[0] + TAU - ways[ways.len() - 1]])
            .fold(0.0, f32::max);
        assert!(widest < 2.4, "{ways:?}");
    }

    /// The drops: none while it overfills; then each goes out from the
    /// bubble and falls (its way turning down), stretched along its way, of
    /// a few sizes, fading; none after the burst's life.
    #[test]
    fn the_drops_arc_out_fall_and_fade() {
        let s = bubble();
        let at = |q: u32, t: f32| sprite(&s, q, 10.0 + t);
        let mut sizes = Vec::new();
        for q in FIRST_DROP..FIRST_DROP + DROPS {
            assert!(at(q, 0.05).is_none() && at(q, BURST as f32).is_none());
            let ps: Vec<Sprite> = [0.1, 0.2, 0.3, 0.4, 0.49]
                .iter()
                .map(|&t| at(q, t).unwrap())
                .collect();
            assert!(ps.iter().all(|p| p.look == Look::Drop));
            // Falling: each step lower against the one before.
            let dy: Vec<f32> = ps.windows(2).map(|w| w[1].at.1 - w[0].at.1).collect();
            assert!(dy.windows(2).all(|w| w[1] > w[0]), "falls: {q} {dy:?}");
            // Out across, the way it went.
            let dx: Vec<f32> = ps.windows(2).map(|w| w[1].at.0 - w[0].at.0).collect();
            assert!(dx.iter().all(|d| d.signum() == dx[0].signum()), "{q}");
            for p in &ps {
                assert!(p.half.0 > p.half.1, "stretched: {q}");
            }
            // Along its way: turned down more as it falls.
            assert!(ps[4].turn.sin() > ps[0].turn.sin(), "{q}");
            assert!(ps[4].alpha < 0.25 * ps[0].alpha, "fading: {q}");
            sizes.push((ps[0].half.1 * 100.0).round() as u32);
        }
        sizes.sort_unstable();
        sizes.dedup();
        assert!(sizes.len() >= 2, "a few sizes: {sizes:?}");
    }
}
