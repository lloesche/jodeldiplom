//! The backdrop's look as numbers: the pieces of its shader that are maths
//! alone, each here twice, for the processor and in WGSL ([`wgsl`]), held
//! to each other in `super::super::shader_check`: how calm it is under the
//! text ([`calm_at`], [`settle`]), how a layer moves with its depth
//! ([`shift`]), how bright it is shown ([`exposure`], [`vignette`]), the
//! rim's light ([`rim_light`]), the light sweeping across the plate
//! ([`sweep`]), the palette's slow drift ([`drift`]) and the rays from
//! behind the subject ([`ray_fan`]). And where the cover sits in the
//! window ([`fit`]), which only the processor works out.

/// How far the calm under the text reaches past it, a share of the height.
pub const FEATHER: f32 = 0.12;
/// The light the backdrop is shown at (linear: about half the cover's
/// brightness as the eye sees it).
pub const EXPOSE: f32 = 0.2;
/// How much darker it is under the text.
pub const CALM_DIM: f32 = 0.45;
/// How much the brights are held down under the text.
pub const CALM_KNEE: f32 = 7.0;
/// How much darker the corners are.
pub const VIGNETTE: f32 = 0.38;
/// The rim's light at rest.
pub const RIM_REST: f32 = 0.08;
/// The rim's light on a drum hit.
pub const RIM_HIT: f32 = 0.8;
/// How long the light sweeping across the plate takes, in seconds.
pub const SWEEP_PERIOD: f32 = 26.0;
/// How wide the sweeping light is, a share of the height.
pub const SWEEP_WIDTH: f32 = 0.38;
/// How long the palette takes to drift through its colours, in seconds.
pub const DRIFT_PERIOD: f32 = 45.0;

/// `x` from 0 below `a` to 1 above `b`, smoothly (as WGSL's `smoothstep`).
pub fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// How calm the backdrop is at `y` (a share of the height, down): 1 where
/// the text is (`text`: its top and bottom; at -1, none), 0 away from it,
/// fading over [`FEATHER`].
pub fn calm_at(y: f32, text: [f32; 2]) -> f32 {
    smoothstep(text[0] - FEATHER, text[0], y) * (1.0 - smoothstep(text[1], text[1] + FEATHER, y))
}

/// Where a layer is read for the cover's point `uv` at nearness `near`:
/// moved against the camera's drift `cam` (cover units) the nearer it is
/// (the far moving the other way), and breathed in round `focus` by
/// `breath` (a share), the nearer the more.
pub fn shift(uv: [f32; 2], near: f32, cam: [f32; 2], breath: f32, focus: [f32; 2]) -> [f32; 2] {
    let k = (near - 0.5) * 2.0;
    [0, 1].map(|i| uv[i] - cam[i] * k - (uv[i] - focus[i]) * breath * near)
}

/// The light the backdrop is shown at where it is `calm` (0 to 1), lifted
/// by `lift` (a section, the singer's shine).
pub fn exposure(calm: f32, lift: f32) -> f32 {
    EXPOSE * (1.0 - CALM_DIM * calm) * (1.0 + lift)
}

/// Light `x` (linear, already exposed) where it is `calm`: its brights held
/// down under the text (a bright subject must not shout over the lyrics),
/// the darks as they are.
pub fn settle(x: f32, calm: f32) -> f32 {
    x / (1.0 + CALM_KNEE * calm * x.max(0.0))
}

/// How much of the light shows at the window's point (`x`, `y`, shares):
/// all of it in the middle, less towards the corners.
pub fn vignette(x: f32, y: f32) -> f32 {
    let (dx, dy) = (x - 0.5, y - 0.5);
    let d = (dx * dx + dy * dy).sqrt() * 2.0;
    1.0 - VIGNETTE * smoothstep(0.55, 1.45, d)
}

/// The rim's light where its mask is `rim` (0 to 1, the edge brightest), at
/// the cover's point (`u`, `v`) and time `t`, its drum hit at `pulse` (0 to
/// 1): a faint shimmer at rest, bright and flickering a little on a hit.
pub fn rim_light(rim: f32, pulse: f32, u: f32, v: f32, t: f32) -> f32 {
    let shimmer = 0.8 + 0.2 * (u * 37.0 + v * 23.0 + t * 1.3).sin();
    let flicker = 1.0 - 0.3 * pulse * (0.5 + 0.5 * (u * 131.0 - v * 97.0 + t * 17.0).sin());
    rim.max(0.0).powf(1.5) * (RIM_REST * shimmer + RIM_HIT * pulse * flicker)
}

/// The light sweeping across the window's point (`x`, `y`, shares) at time
/// `t`, the window `aspect` wide for its height: a broad band, slanting,
/// coming round every [`SWEEP_PERIOD`] (0 to 1).
pub fn sweep(x: f32, y: f32, t: f32, aspect: f32) -> f32 {
    let along = x * aspect * 0.82 + y * 0.57;
    let span = aspect * 0.82 + 0.57 + 1.2;
    let pos = (t / SWEEP_PERIOD).fract() * span - 0.6;
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

/// Where the cover is read for the window: a point (shares across and
/// down) is `fit[0..2]` × it + `fit[2..4]` in the cover's units. The cover
/// fills the `window` (its aspect kept, the rest cut off), `zoom` times
/// larger still (room for the drift), the cut centred on `focus` (the
/// cover's units) as far as the cover reaches.
pub fn fit(window: (f32, f32), cover: (f32, f32), focus: (f32, f32), zoom: f32) -> [f32; 4] {
    let wa = window.0 / window.1.max(1.0);
    let ca = cover.0 / cover.1.max(1.0);
    let (sx, sy) = if wa > ca {
        (1.0, ca / wa)
    } else {
        (wa / ca, 1.0)
    };
    let (sx, sy) = (sx / zoom, sy / zoom);
    let ox = (focus.0 - sx / 2.0).clamp(0.0, 1.0 - sx);
    let oy = (focus.1 - sy / 2.0).clamp(0.0, 1.0 - sy);
    [sx, sy, ox, oy]
}

/// The functions above in WGSL, their constants with them.
pub fn wgsl() -> String {
    format!(
        "const FEATHER: f32 = {FEATHER:?};
const EXPOSE: f32 = {EXPOSE:?};
const CALM_DIM: f32 = {CALM_DIM:?};
const VIGNETTE: f32 = {VIGNETTE:?};
const CALM_KNEE: f32 = {CALM_KNEE:?};
const RIM_REST: f32 = {RIM_REST:?};
const RIM_HIT: f32 = {RIM_HIT:?};
const SWEEP_PERIOD: f32 = {SWEEP_PERIOD:?};
const SWEEP_WIDTH: f32 = {SWEEP_WIDTH:?};
const DRIFT_PERIOD: f32 = {DRIFT_PERIOD:?};
const TAU_: f32 = 6.2831855;
{WGSL}"
    )
}

const WGSL: &str = r"
fn calm_at(y: f32, text: vec2<f32>) -> f32 {
    return smoothstep(text.x - FEATHER, text.x, y) * (1.0 - smoothstep(text.y, text.y + FEATHER, y));
}

fn shift(uv: vec2<f32>, near: f32, cam: vec2<f32>, breath: f32, focus: vec2<f32>) -> vec2<f32> {
    return uv - cam * ((near - 0.5) * 2.0) - (uv - focus) * (breath * near);
}

fn exposure(calm: f32, lift: f32) -> f32 {
    return EXPOSE * (1.0 - CALM_DIM * calm) * (1.0 + lift);
}

fn settle(x: vec3<f32>, calm: f32) -> vec3<f32> {
    return x / (vec3<f32>(1.0) + CALM_KNEE * calm * max(x, vec3<f32>(0.0)));
}

fn vignette(x: f32, y: f32) -> f32 {
    let d = length(vec2<f32>(x - 0.5, y - 0.5)) * 2.0;
    return 1.0 - VIGNETTE * smoothstep(0.55, 1.45, d);
}

fn rim_light(rim: f32, pulse: f32, u: f32, v: f32, t: f32) -> f32 {
    let shimmer = 0.8 + 0.2 * sin(u * 37.0 + v * 23.0 + t * 1.3);
    let flicker = 1.0 - 0.3 * pulse * (0.5 + 0.5 * sin(u * 131.0 - v * 97.0 + t * 17.0));
    return pow(max(rim, 0.0), 1.5) * (RIM_REST * shimmer + RIM_HIT * pulse * flicker);
}

fn sweep(x: f32, y: f32, t: f32, aspect: f32) -> f32 {
    let along = x * aspect * 0.82 + y * 0.57;
    let span = aspect * 0.82 + 0.57 + 1.2;
    let pos = fract(t / SWEEP_PERIOD) * span - 0.6;
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
    fn it_is_calm_under_the_text_and_free_away_from_it() {
        let text = [0.2, 0.8];
        assert_eq!(calm_at(0.3, text), 1.0);
        assert_eq!(calm_at(0.75, text), 1.0);
        assert_eq!(calm_at(0.05, text), 0.0);
        assert_eq!(calm_at(0.95, text), 0.0);
        let edge = calm_at(0.2 - FEATHER / 2.0, text);
        assert!(edge > 0.3 && edge < 0.7, "{edge}");
        // At -1: none.
        assert_eq!(calm_at(0.01, [-1.0, -1.0]), 0.0);
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
    fn the_rim_rests_faintly_and_lights_on_a_hit() {
        let rest = rim_light(1.0, 0.0, 0.3, 0.4, 2.0);
        let hit = rim_light(1.0, 1.0, 0.3, 0.4, 2.0);
        assert!(rest > 0.05 && rest < 0.11, "{rest}");
        assert!(hit > 0.5, "{hit}");
        assert_eq!(rim_light(0.0, 1.0, 0.3, 0.4, 2.0), 0.0);
    }

    #[test]
    fn the_sweep_crosses_the_window_and_comes_round() {
        let seen: Vec<f32> = (0..26)
            .map(|s| sweep(0.5, 0.5, s as f32, 16.0 / 9.0))
            .collect();
        assert!(seen.iter().any(|&s| s > 0.95));
        assert!(seen.iter().any(|&s| s < 0.05));
        assert!(
            (sweep(0.5, 0.5, 3.0, 1.5) - sweep(0.5, 0.5, 3.0 + SWEEP_PERIOD, 1.5)).abs() < 1e-3
        );
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
    fn the_cover_fills_wide_and_tall_windows_round_its_subject() {
        // Wide: all across, a band of it down, round the subject's top.
        let f = fit((1600.0, 900.0), (1.0, 1.0), (0.5, 0.3), 1.0);
        assert_eq!(f[0], 1.0);
        assert!((f[1] - 0.5625).abs() < 1e-4);
        assert!((f[3] - (0.3 - 0.28125)).abs() < 1e-4);
        // Tall: all down, the subject to the right kept in.
        let f = fit((800.0, 1400.0), (1.0, 1.0), (0.85, 0.5), 1.05);
        assert!(f[1] < 1.0 && f[0] < 0.6);
        assert!(f[2] <= 0.85 && f[2] + f[0] >= 0.85);
        assert!(f[2] + f[0] <= 1.0);
        // Never past the cover's edges.
        let f = fit((1600.0, 900.0), (1.0, 1.0), (0.5, 0.98), 1.0);
        assert!((f[3] + f[1] - 1.0).abs() < 1e-5);
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
    fn the_corners_are_darker_and_the_text_calmer() {
        assert_eq!(vignette(0.5, 0.5), 1.0);
        assert!(vignette(0.0, 0.0) < 0.7);
        assert!(exposure(1.0, 0.0) < exposure(0.0, 0.0));
        assert!(exposure(0.0, 0.3) > exposure(0.0, 0.0));
        // Under the text the brights come down more than the darks.
        assert!(settle(0.3, 1.0) < 0.3 * 0.5 && settle(0.01, 1.0) > 0.009);
        assert_eq!(settle(0.3, 0.0), 0.3);
    }
}
