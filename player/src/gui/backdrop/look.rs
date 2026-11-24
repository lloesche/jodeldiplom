//! The backdrop's look as numbers: the pieces of its shader that are maths
//! alone, each here twice, for the processor and in WGSL ([`wgsl`]), held
//! to each other in `super::super::shader_check`: the card's rounded edge
//! ([`rounded_box`], [`card_edge`]) and the light and shadow falling off
//! round it ([`falloff`]), the smooth cubic sampling's taps ([`bspline`]),
//! how a layer moves with its depth ([`shift`]), the corners' shade
//! ([`vignette`]), the rim's light ([`rim_light`]), the light sweeping
//! across the plate ([`sweep`]), the palette's slow drift ([`drift`]) and
//! the rays from behind the subject ([`ray_fan`]). And where the card and
//! the blur under it sit in the window ([`card`], [`fill`]), how softly
//! the card is read for its size ([`softness`]) and the levels the cover is
//! read blurred at ([`blurs`]), which only the processor works out.

/// The card's side at most, a share of the window's height (and, for a
/// tall window, of its width).
pub const CARD: f32 = 0.84;
/// The card's side at most, a share of the window's width.
pub const CARD_ACROSS: f32 = 0.9;
/// The most a cover's pixel is enlarged on the card: a small one (an MP3's
/// 300 pixels) still most of a 5K screen's height, read softly; past this
/// it would only be softer, not larger.
pub const CAP: f32 = 6.0;
/// The card's corners' radius, a share of its shorter side.
pub const RADIUS: f32 = 0.022;
/// How wide the card's edge fades into the blur, a share of its shorter
/// side.
pub const EDGE: f32 = 0.006;
/// The blur under the card: the cover read about this many texels across.
pub const BLUR_TEXELS: f32 = 12.0;
/// How far the card's shadow and the light round it reach, a share of its
/// shorter side.
pub const REACH: f32 = 0.09;
/// The rim's light at rest.
pub const RIM_REST: f32 = 0.18;
/// The rim's light on a drum hit.
pub const RIM_HIT: f32 = 2.2;
/// How much darker the corners are.
pub const VIGNETTE: f32 = 0.5;
/// How long the light sweeping across the plate takes, in seconds.
pub const SWEEP_PERIOD: f32 = 17.0;
/// How wide the sweeping light is, a share of the card.
pub const SWEEP_WIDTH: f32 = 0.22;
/// How long the palette takes to drift through its colours, in seconds.
pub const DRIFT_PERIOD: f32 = 45.0;

/// `x` from 0 below `a` to 1 above `b`, smoothly (as WGSL's `smoothstep`).
pub fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// How far (`dx`, `dy`) from a rounded box's middle is outside it (less
/// than 0 inside), the box `hx` × `hy` from its middle to its sides, its
/// corners rounded by `r`.
pub fn rounded_box(dx: f32, dy: f32, hx: f32, hy: f32, r: f32) -> f32 {
    let qx = dx.abs() - hx + r;
    let qy = dy.abs() - hy + r;
    let out = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt();
    out + qx.max(qy).min(0.0) - r
}

/// How much of the card shows `d` from its edge (less than 0 inside), its
/// shorter side `side` long: all of it inside, fading over [`EDGE`] to
/// none at the edge.
pub fn card_edge(d: f32, side: f32) -> f32 {
    1.0 - smoothstep(-EDGE * side, 0.0, d)
}

/// Light (or shade) falling off `d` outside the card (all of it inside),
/// over `reach`: smoothly, as a soft light does.
pub fn falloff(d: f32, reach: f32) -> f32 {
    let x = d.max(0.0) / reach.max(1e-4);
    1.0 / (1.0 + x * x * 3.0)
}

/// A smooth cubic (B-spline) read of a texture at `f`, the share of the
/// way from one texel to the next: the two bilinear reads' weights and
/// where they are, each from the first texel (`[g0, g1, h0, h1]`).
pub fn bspline(f: f32) -> [f32; 4] {
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    let w3 = f3 / 6.0;
    let (g0, g1) = (w0 + w1, w2 + w3);
    [g0, g1, w1 / g0 - 1.0, w3 / g1 + 1.0]
}

/// Where a layer is read for the cover's point `uv` at nearness `near`:
/// moved against the camera's drift `cam` (cover units) the nearer it is
/// (the far moving the other way), and breathed in round `focus` by
/// `breath` (a share), the nearer the more.
pub fn shift(uv: [f32; 2], near: f32, cam: [f32; 2], breath: f32, focus: [f32; 2]) -> [f32; 2] {
    let k = (near - 0.5) * 2.0;
    [0, 1].map(|i| uv[i] - cam[i] * k - (uv[i] - focus[i]) * breath * near)
}

/// How much of the light shows at the window's point (`x`, `y`, shares):
/// all of it in the middle, less towards the corners.
pub fn vignette(x: f32, y: f32) -> f32 {
    let (dx, dy) = (x - 0.5, y - 0.5);
    let d = (dx * dx + dy * dy).sqrt() * 2.0;
    1.0 - VIGNETTE * smoothstep(0.35, 1.4, d)
}

/// The rim's light where its mask is `rim` (0 to 1, the edge brightest), at
/// the cover's point (`u`, `v`) and time `t`, its drum hit at `pulse` (0 to
/// 1): a shimmer at rest, bright and flickering a little on a hit.
pub fn rim_light(rim: f32, pulse: f32, u: f32, v: f32, t: f32) -> f32 {
    let shimmer = 0.75 + 0.25 * (u * 23.0 + v * 17.0 + t * 1.3).sin();
    let flicker = 1.0 - 0.3 * pulse * (0.5 + 0.5 * (u * 131.0 - v * 97.0 + t * 17.0).sin());
    rim.max(0.0).powf(1.5) * (RIM_REST * shimmer + RIM_HIT * pulse * flicker)
}

/// The light sweeping across the card's point (`u`, `v`, the cover's
/// units) at time `t`: a broad band, slanting, coming round every
/// [`SWEEP_PERIOD`] (0 to 1).
pub fn sweep(u: f32, v: f32, t: f32) -> f32 {
    let along = u * 0.82 + v * 0.57;
    let span = 0.82 + 0.57 + 1.0;
    let pos = (t / SWEEP_PERIOD).fract() * span - 0.5;
    let d = (along - pos) / SWEEP_WIDTH;
    (-d * d).exp()
}

/// The palette's three colours' shares at time `t`, drifting from one to
/// the next over [`DRIFT_PERIOD`] (they add up to 1).
pub fn drift(t: f32) -> [f32; 3] {
    let phase = t / DRIFT_PERIOD * std::f32::consts::TAU;
    let w = [0.0f32, 1.0, 2.0].map(|k| {
        let c = 0.5 + 0.5 * (phase - k * std::f32::consts::TAU / 3.0).cos();
        c * c + 0.02
    });
    let sum: f32 = w.iter().sum();
    w.map(|x| x / sum)
}

/// The rays fanning out from the light behind the subject, at (`dx`, `dy`)
/// from it (shares of the height) at time `t`, reaching about `reach`:
/// two sets of beams turning slowly against each other, fading out with
/// the distance (0 to 1).
pub fn ray_fan(dx: f32, dy: f32, t: f32, reach: f32) -> f32 {
    let d = (dx * dx + dy * dy).sqrt();
    let angle = dy.atan2(dx);
    let beam = |n: f32, turn: f32| (0.5 + 0.5 * (angle * n + turn).sin()).powi(8);
    let beams = beam(9.0, t * 0.15).max(0.7 * beam(14.0, 1.7 - t * 0.11));
    beams * (-d / (reach.max(0.01) * 2.5)).exp()
}

/// Where the card is in a `window` (pixels) for a `cover` of pixels: its
/// middle and its half width and height, shares of the window's. The whole
/// cover, its aspect kept, as large as [`CARD`] and [`CARD_ACROSS`] let it
/// be, but no cover pixel more than [`CAP`] window pixels across.
pub fn card(window: (f32, f32), cover: (f32, f32)) -> [f32; 4] {
    let (w, h) = (window.0.max(1.0), window.1.max(1.0));
    let (cw, ch) = (cover.0.max(1.0), cover.1.max(1.0));
    let room = (CARD * w.min(h)).min(CARD_ACROSS * w);
    let scale = (room / cw.max(ch)).min(CAP);
    [0.5, 0.5, cw * scale / (2.0 * w), ch * scale / (2.0 * h)]
}

/// How softly the card is read, its cover's pixel `scale` window pixels
/// across: as it is up to a little enlarged, smoothly (cubic) when much
/// enlarged (no pixel nor block of the cover's coding shows), softer still
/// the more a small cover is (`[smooth, softer]`, each 0 to 1); and the level a
/// smaller card reads (each level half the last).
pub fn softness(scale: f32) -> ([f32; 2], f32) {
    let smooth = smoothstep(1.2, 2.2, scale);
    let softer = smoothstep(2.4, 5.0, scale) * 0.35;
    ([smooth, softer], (1.0 / scale.max(1e-3)).log2().max(0.0))
}

/// The levels a `cover` (pixels) with `levels` of them is read blurred at:
/// the blur under the card (about [`BLUR_TEXELS`] across), the glow round
/// the subject (twice as fine) and its shadow and the rim's glow (four
/// times), each at least the first.
pub fn blurs(cover: (u32, u32), levels: u32) -> [f32; 3] {
    let long = cover.0.max(cover.1).max(1) as f32;
    let last = levels.max(1) as f32 - 1.0;
    let blur = (long / BLUR_TEXELS).log2().floor().clamp(0.0, last);
    [blur, (blur - 1.0).max(0.0), (blur - 2.0).max(0.0)]
}

/// Where the blur under the card is read for the window: a point (shares
/// across and down) is `fit[0..2]` × it + `fit[2..4]` in the cover's
/// units. The cover fills the `window` (its aspect kept, the rest cut off,
/// centred), `zoom` times larger still (room for the drift).
pub fn fill(window: (f32, f32), cover: (f32, f32), zoom: f32) -> [f32; 4] {
    let wa = window.0 / window.1.max(1.0);
    let ca = cover.0 / cover.1.max(1.0);
    let (sx, sy) = if wa > ca {
        (1.0, ca / wa)
    } else {
        (wa / ca, 1.0)
    };
    let (sx, sy) = (sx / zoom, sy / zoom);
    [sx, sy, (1.0 - sx) / 2.0, (1.0 - sy) / 2.0]
}

/// The functions above in WGSL, their constants with them.
pub fn wgsl() -> String {
    format!(
        "const RADIUS: f32 = {RADIUS:?};
const EDGE: f32 = {EDGE:?};
const REACH: f32 = {REACH:?};
const RIM_REST: f32 = {RIM_REST:?};
const RIM_HIT: f32 = {RIM_HIT:?};
const VIGNETTE: f32 = {VIGNETTE:?};
const SWEEP_PERIOD: f32 = {SWEEP_PERIOD:?};
const SWEEP_WIDTH: f32 = {SWEEP_WIDTH:?};
const DRIFT_PERIOD: f32 = {DRIFT_PERIOD:?};
const TAU_: f32 = 6.2831855;
{WGSL}"
    )
}

const WGSL: &str = r"
fn rounded_box(dx: f32, dy: f32, hx: f32, hy: f32, r: f32) -> f32 {
    let q = vec2<f32>(abs(dx) - hx + r, abs(dy) - hy + r);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

fn card_edge(d: f32, side: f32) -> f32 {
    return 1.0 - smoothstep(-EDGE * side, 0.0, d);
}

fn falloff(d: f32, reach: f32) -> f32 {
    let x = max(d, 0.0) / max(reach, 1e-4);
    return 1.0 / (1.0 + x * x * 3.0);
}

fn bspline(f: f32) -> vec4<f32> {
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    let w3 = f3 / 6.0;
    let g0 = w0 + w1;
    let g1 = w2 + w3;
    return vec4<f32>(g0, g1, w1 / g0 - 1.0, w3 / g1 + 1.0);
}

fn shift(uv: vec2<f32>, near: f32, cam: vec2<f32>, breath: f32, focus: vec2<f32>) -> vec2<f32> {
    return uv - cam * ((near - 0.5) * 2.0) - (uv - focus) * (breath * near);
}

fn vignette(x: f32, y: f32) -> f32 {
    let d = length(vec2<f32>(x - 0.5, y - 0.5)) * 2.0;
    return 1.0 - VIGNETTE * smoothstep(0.35, 1.4, d);
}

fn rim_light(rim: f32, pulse: f32, u: f32, v: f32, t: f32) -> f32 {
    let shimmer = 0.75 + 0.25 * sin(u * 23.0 + v * 17.0 + t * 1.3);
    let flicker = 1.0 - 0.3 * pulse * (0.5 + 0.5 * sin(u * 131.0 - v * 97.0 + t * 17.0));
    return pow(max(rim, 0.0), 1.5) * (RIM_REST * shimmer + RIM_HIT * pulse * flicker);
}

fn sweep(u: f32, v: f32, t: f32) -> f32 {
    let along = u * 0.82 + v * 0.57;
    let span = 0.82 + 0.57 + 1.0;
    let pos = fract(t / SWEEP_PERIOD) * span - 0.5;
    let d = (along - pos) / SWEEP_WIDTH;
    return exp(-d * d);
}

fn drift(t: f32) -> vec3<f32> {
    let phase = t / DRIFT_PERIOD * TAU_;
    let c = vec3<f32>(0.5) + 0.5 * cos(vec3<f32>(phase) - vec3<f32>(0.0, 1.0, 2.0) * (TAU_ / 3.0));
    let w = c * c + vec3<f32>(0.02);
    return w / (w.x + w.y + w.z);
}

fn ray_beam(angle: f32, n: f32, turn: f32) -> f32 {
    let b = 0.5 + 0.5 * sin(angle * n + turn);
    let b2 = b * b;
    let b4 = b2 * b2;
    return b4 * b4;
}

fn ray_fan(dx: f32, dy: f32, t: f32, reach: f32) -> f32 {
    let d = sqrt(dx * dx + dy * dy);
    let angle = atan2(dy, dx);
    let beams = max(ray_beam(angle, 9.0, t * 0.15), 0.7 * ray_beam(angle, 14.0, 1.7 - t * 0.11));
    return beams * exp(-d / (max(reach, 0.01) * 2.5));
}
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_card_is_the_whole_cover_as_large_as_it_looks_good() {
        // A wide window: as tall as it may be, square, in the middle.
        let c = card((1600.0, 900.0), (1024.0, 1024.0));
        assert_eq!((c[0], c[1]), (0.5, 0.5));
        assert!((c[3] - CARD / 2.0).abs() < 1e-5, "{c:?}");
        assert!((c[2] * 1600.0 - c[3] * 900.0).abs() < 1e-2, "{c:?}");
        // A tall one: as wide as it may be.
        let c = card((900.0, 1600.0), (1024.0, 1024.0));
        assert!((c[2] - CARD / 2.0).abs() < 1e-5, "{c:?}");
        // A small cover on a large screen: no more than CAP times its size.
        let c = card((5120.0, 2880.0), (300.0, 300.0));
        assert!((c[3] * 2.0 * 2880.0 - 300.0 * CAP).abs() < 0.1, "{c:?}");
        // Its aspect kept.
        let c = card((1600.0, 900.0), (1000.0, 500.0));
        assert!((c[2] * 1600.0 - 2.0 * c[3] * 900.0).abs() < 1e-2, "{c:?}");
    }

    #[test]
    fn a_card_much_enlarged_is_read_smoothly_and_a_small_one_from_its_levels() {
        assert_eq!(softness(1.0), ([0.0, 0.0], 0.0));
        let ([smooth, softer], level) = softness(2.5);
        assert!(smooth == 1.0 && softer > 0.0 && softer < 0.1 && level == 0.0);
        assert_eq!(softness(CAP).0, [1.0, 0.35]);
        assert!((softness(0.25).1 - 2.0).abs() < 1e-5);
    }

    #[test]
    fn the_blur_is_a_dozen_texels_across_whatever_the_cover_s_size() {
        assert_eq!(blurs((1024, 1024), 9), [6.0, 5.0, 4.0]);
        assert_eq!(blurs((300, 200), 9), [4.0, 3.0, 2.0]);
        // A tiny cover: its own pixels; never past its last level.
        assert_eq!(blurs((8, 8), 4), [0.0, 0.0, 0.0]);
        assert_eq!(blurs((4096, 4096), 9), [8.0, 7.0, 6.0]);
    }

    #[test]
    fn the_card_s_edge_is_rounded_and_fades_into_the_blur() {
        // Inside, on the edge, outside.
        assert!(rounded_box(0.0, 0.0, 10.0, 5.0, 1.0) < -4.9);
        assert!(rounded_box(10.0, 0.0, 10.0, 5.0, 1.0).abs() < 1e-5);
        assert!((rounded_box(13.0, 0.0, 10.0, 5.0, 1.0) - 3.0).abs() < 1e-5);
        // The corner rounded: its tip is outside.
        assert!(rounded_box(10.0, 5.0, 10.0, 5.0, 1.0) > 0.4);
        assert_eq!(card_edge(-100.0, 1000.0), 1.0);
        assert_eq!(card_edge(0.0, 1000.0), 0.0);
        let half = card_edge(-EDGE * 500.0, 1000.0);
        assert!(half > 0.3 && half < 0.7, "{half}");
        assert_eq!(falloff(-3.0, 10.0), 1.0);
        assert!(falloff(10.0, 10.0) < 0.3 && falloff(2.0, 10.0) > 0.85);
    }

    #[test]
    fn the_cubic_read_weighs_its_four_texels_as_a_b_spline() {
        for f in [0.0, 0.25, 0.5, 0.9] {
            let [g0, g1, h0, h1] = bspline(f);
            assert!((g0 + g1 - 1.0).abs() < 1e-5);
            // The reads fall between the texels either side of the two.
            assert!(
                (-1.0..=0.0).contains(&h0) && (1.0..=2.0).contains(&h1),
                "{f}"
            );
        }
        // Halfway: alike either side.
        let [g0, g1, h0, h1] = bspline(0.5);
        assert!((g0 - g1).abs() < 1e-6 && (h0 + h1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn near_layers_move_with_the_camera_and_far_ones_against_it() {
        let (cam, focus) = ([0.01, 0.0], [0.5, 0.5]);
        let near = shift([0.3, 0.3], 1.0, cam, 0.0, focus);
        let far = shift([0.3, 0.3], 0.0, cam, 0.0, focus);
        assert!(near[0] < 0.3 && far[0] > 0.3);
        // Breathing in reads nearer the focus: larger on screen.
        let b = shift([0.3, 0.3], 1.0, [0.0; 2], 0.01, focus);
        assert!(b[0] > 0.3 && b[0] - 0.3 < 0.003);
    }

    #[test]
    fn the_rim_shimmers_at_rest_and_lights_on_a_hit() {
        let rest = rim_light(1.0, 0.0, 0.3, 0.4, 2.0);
        let hit = rim_light(1.0, 1.0, 0.3, 0.4, 2.0);
        assert!(rest > 0.08 && rest < 0.25, "{rest}");
        assert!(hit > 1.5, "{hit}");
        assert_eq!(rim_light(0.0, 1.0, 0.3, 0.4, 2.0), 0.0);
    }

    #[test]
    fn the_sweep_crosses_the_card_and_comes_round() {
        let seen: Vec<f32> = (0..170).map(|s| sweep(0.5, 0.5, s as f32 * 0.1)).collect();
        assert!(seen.iter().any(|&s| s > 0.95));
        assert!(seen.iter().any(|&s| s < 0.05));
        assert!((sweep(0.5, 0.5, 3.0) - sweep(0.5, 0.5, 3.0 + SWEEP_PERIOD)).abs() < 1e-3);
    }

    #[test]
    fn the_palette_drifts_through_its_colours() {
        for t in [0.0, 7.0, 20.0, 33.0] {
            let w = drift(t);
            assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
        assert!(drift(0.0)[0] > 0.6);
        assert!(drift(DRIFT_PERIOD / 3.0)[1] > 0.6);
    }

    #[test]
    fn the_blur_fills_wide_and_tall_windows() {
        let f = fill((1600.0, 900.0), (1.0, 1.0), 1.0);
        assert_eq!(f[0], 1.0);
        assert!((f[1] - 0.5625).abs() < 1e-4 && (f[3] - 0.21875).abs() < 1e-4);
        let f = fill((800.0, 1600.0), (1.0, 1.0), 1.25);
        assert!((f[0] - 0.4).abs() < 1e-4 && (f[1] - 0.8).abs() < 1e-4);
        assert!((f[2] - 0.3).abs() < 1e-4 && (f[3] - 0.1).abs() < 1e-4);
    }

    #[test]
    fn the_rays_are_beams_fading_out() {
        let round: Vec<f32> = (0..360)
            .map(|k| {
                let a = (k as f32).to_radians();
                ray_fan(a.cos() * 0.1, a.sin() * 0.1, 3.0, 0.2)
            })
            .collect();
        let most = round.iter().copied().fold(0.0, f32::max);
        let least = round.iter().copied().fold(1.0, f32::min);
        assert!(most > 0.6 && least < 0.01, "{most} {least}");
        assert!(ray_fan(0.0, 0.9, 3.0, 0.2) < 0.2 * ray_fan(0.0, 0.05, 3.0, 0.2));
    }

    #[test]
    fn the_corners_are_darker() {
        assert_eq!(vignette(0.5, 0.5), 1.0);
        assert!(vignette(0.0, 0.0) < 0.7);
    }
}
