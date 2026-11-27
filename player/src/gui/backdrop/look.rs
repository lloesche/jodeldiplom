//! The backdrop's look as numbers: the pieces of its shader that are maths
//! alone, each here twice, for the processor and in WGSL ([`wgsl`]), held
//! to each other in `super::super::shader_check`: the card's rounded edge
//! ([`card_edge`], round its distance from the box:
//! [`jodel_looks::math::round_box`]) and the light and shadow falling off
//! round it ([`falloff`]), the smooth cubic sampling's taps ([`bspline`]),
//! how a layer moves with its depth ([`shift`]), the corners' shade
//! ([`vignette`]), the light sweeping across the plate ([`sweep`]), the
//! palette's slow drift ([`drift`]), the spotlight rays from behind the
//! subject ([`ray_fan`]), the subject pushed out by the beat round its
//! middle ([`kicked`]) and how blurred the plate round it is for its depth
//! ([`defocus`]). And where the card and
//! the blur under it sit in the window ([`card`], [`fill`]), how softly
//! the card is read for its size ([`softness`]) and the levels the cover is
//! read blurred at ([`blurs`], [`defocus_level`]), which only the processor
//! works out.

use jodel_looks::math::smoothstep;

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
/// How much darker the corners are.
pub const VIGNETTE: f32 = 0.5;
/// How long the light sweeping across the plate takes, in seconds.
pub const SWEEP_PERIOD: f32 = 17.0;
/// How wide the sweeping light is, a share of the card.
pub const SWEEP_WIDTH: f32 = 0.22;
/// How long the palette takes to drift through its colours, in seconds.
pub const DRIFT_PERIOD: f32 = 45.0;
/// How fast the spotlight rays turn, radians a second: sweeping round about every ten seconds.
pub const RAYS_TURN: f32 = 0.6;
/// The most the plate round the subject is blurred, a share of the cover's
/// longer side (about the blur's width).
pub const DEFOCUS: f32 = 0.035;
/// The share of the most blur the plate has where it is as near as the
/// subject (everything round the subject is blurred; more farther back).
pub const DEFOCUS_BASE: f32 = 0.5;
/// How much farther back than the subject (the depth's nearness) the plate
/// is blurred the most.
pub const DEFOCUS_SPAN: f32 = 0.3;

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

/// The spotlight rays fanning out from behind the subject, at (`dx`, `dy`)
/// from its middle (shares of the height) at time `t`, reaching about
/// `reach`: two sets of broad, soft beams sweeping round against each
/// other ([`RAYS_TURN`]), fading out with the distance (0 to 1).
pub fn ray_fan(dx: f32, dy: f32, t: f32, reach: f32) -> f32 {
    let d = (dx * dx + dy * dy).sqrt();
    let angle = dy.atan2(dx);
    let beam = |n: f32, turn: f32| (0.5 + 0.5 * (angle * n + turn).sin()).powi(4);
    let beams = 0.65 * beam(6.0, t * RAYS_TURN) + 0.35 * beam(9.0, 1.7 - t * 1.6 * RAYS_TURN);
    beams * (-d / (reach.max(0.01) * 7.0)).exp()
}

/// Where the subject is read for the cover's point `uv` grown by `grow` (a
/// share of its size) round its `centre`: as a speaker's cone pushes out.
pub fn kicked(uv: [f32; 2], centre: [f32; 2], grow: f32) -> [f32; 2] {
    [0, 1].map(|i| centre[i] + (uv[i] - centre[i]) / (1.0 + grow))
}

/// How blurred the plate is where its nearness is `near`, the subject's
/// `focus` (a share of the most blur, [`DEFOCUS`]): everything round the
/// subject [`DEFOCUS_BASE`] of it, the farther back the more, all of it
/// [`DEFOCUS_SPAN`] behind.
pub fn defocus(near: f32, focus: f32) -> f32 {
    DEFOCUS_BASE + (1.0 - DEFOCUS_BASE) * smoothstep(0.0, DEFOCUS_SPAN, focus - near)
}

/// The level a `cover` (pixels) with `levels` of them is read at blurred
/// the most ([`DEFOCUS`]; a level between two is read from both).
pub fn defocus_level(cover: (u32, u32), levels: u32) -> f32 {
    let long = cover.0.max(cover.1).max(1) as f32;
    (DEFOCUS * long)
        .log2()
        .clamp(0.0, levels.max(1) as f32 - 1.0)
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
/// the blur under the card (about [`BLUR_TEXELS`] across) and the subject's
/// shadow (four times as fine), each at least the first.
pub fn blurs(cover: (u32, u32), levels: u32) -> [f32; 2] {
    let long = cover.0.max(cover.1).max(1) as f32;
    let last = levels.max(1) as f32 - 1.0;
    let blur = (long / BLUR_TEXELS).log2().floor().clamp(0.0, last);
    [blur, (blur - 2.0).max(0.0)]
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
const VIGNETTE: f32 = {VIGNETTE:?};
const SWEEP_PERIOD: f32 = {SWEEP_PERIOD:?};
const SWEEP_WIDTH: f32 = {SWEEP_WIDTH:?};
const DRIFT_PERIOD: f32 = {DRIFT_PERIOD:?};
const RAYS_TURN: f32 = {RAYS_TURN:?};
const DEFOCUS_BASE: f32 = {DEFOCUS_BASE:?};
const DEFOCUS_SPAN: f32 = {DEFOCUS_SPAN:?};
const TAU_: f32 = 6.2831855;
{WGSL}"
    )
}

/// The backdrop's passes' triangle over their whole target (the covers',
/// the scenes', the stretch's), its corners read as the target's.
pub const SCREEN: &str = r"
struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> Out {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: Out;
    o.pos = vec4<f32>(p * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
    o.uv = vec2<f32>(p.x, 1.0 - p.y);
    return o;
}
";

/// How the covers' and the scenes' light is shown (with the light's tone,
/// [`crate::gui::bloom::TONE`], and `LINEAR` before it): `finish` tones
/// it, grades it by the theme (the colours mostly taken out, tinted; 0:
/// none) and dithers it by the pixel; `luma` and `hash` with it.
pub const FINISH: &str = r"
// How much of its colour a graded colour keeps.
const GRADE_KEPT: f32 = 0.4;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn finish(col: vec3<f32>, grade: vec4<f32>, pos: vec2<f32>) -> vec3<f32> {
    var seen = tone(col, LINEAR);
    let graded = mix(vec3<f32>(luma(seen)), seen, GRADE_KEPT) * grade.rgb;
    seen = mix(seen, graded, grade.w);
    return seen + vec3<f32>((hash(pos) - 0.5) / 255.0);
}
";

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
    return b2 * b2;
}

fn ray_fan(dx: f32, dy: f32, t: f32, reach: f32) -> f32 {
    let d = sqrt(dx * dx + dy * dy);
    let angle = atan2(dy, dx);
    let beams = 0.65 * ray_beam(angle, 6.0, t * RAYS_TURN)
        + 0.35 * ray_beam(angle, 9.0, 1.7 - t * 1.6 * RAYS_TURN);
    return beams * exp(-d / (max(reach, 0.01) * 7.0));
}

fn kicked(uv: vec2<f32>, centre: vec2<f32>, grow: f32) -> vec2<f32> {
    return centre + (uv - centre) / (1.0 + grow);
}

fn defocus(near: f32, focus: f32) -> f32 {
    return DEFOCUS_BASE + (1.0 - DEFOCUS_BASE) * smoothstep(0.0, DEFOCUS_SPAN, focus - near);
}
";

#[cfg(test)]
mod tests {
    use super::*;
    use jodel_looks::math::{round_box, v2};

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
        assert_eq!(blurs((1024, 1024), 9), [6.0, 4.0]);
        assert_eq!(blurs((300, 200), 9), [4.0, 2.0]);
        // A tiny cover: its own pixels; never past its last level.
        assert_eq!(blurs((8, 8), 4), [0.0, 0.0]);
        assert_eq!(blurs((4096, 4096), 9), [8.0, 6.0]);
    }

    /// Everything round the subject blurred, as near as it is half the most,
    /// the farther back the more, all of it a span behind; nearer than it,
    /// no less than half (only the subject is sharp).
    #[test]
    fn the_plate_is_blurred_the_more_the_farther_back_it_is() {
        assert_eq!(defocus(0.8, 0.8), DEFOCUS_BASE);
        assert_eq!(defocus(0.95, 0.8), DEFOCUS_BASE);
        assert_eq!(defocus(0.8 - DEFOCUS_SPAN, 0.8), 1.0);
        let mut last = 0.0;
        for k in 0..=10 {
            let d = defocus(0.8 - k as f32 * 0.04, 0.8);
            assert!(d >= last && (DEFOCUS_BASE..=1.0).contains(&d), "{k}: {d}");
            last = d;
        }
        // The most blur about a thirtieth of the cover across, whatever its
        // size; never past its last level.
        assert!((defocus_level((1000, 1000), 9) - 35f32.log2()).abs() < 1e-5);
        assert!((defocus_level((600, 300), 9) - 21f32.log2()).abs() < 1e-5);
        assert_eq!(defocus_level((100_000, 100_000), 9), 8.0);
        assert_eq!(defocus_level((20, 20), 5), 0.0);
    }

    /// The beat grows the subject round its middle: the middle stays, a
    /// point off it is read nearer it (so shows farther out), by the share.
    #[test]
    fn the_subject_grows_round_its_middle() {
        let c = [0.6, 0.4];
        assert_eq!(kicked(c, c, 0.03), c);
        let k = kicked([0.9, 0.4], c, 0.03);
        assert!((k[0] - (0.6 + 0.3 / 1.03)).abs() < 1e-6 && k[1] == 0.4);
        let still = kicked([0.1, 0.2], c, 0.0);
        assert!((still[0] - 0.1).abs() < 1e-6 && (still[1] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn the_card_s_edge_is_rounded_and_fades_into_the_blur() {
        // Inside, on the edge, outside.
        let half = v2(10.0, 5.0);
        assert!(round_box(v2(0.0, 0.0), half, 1.0) < -4.9);
        assert!(round_box(v2(10.0, 0.0), half, 1.0).abs() < 1e-5);
        assert!((round_box(v2(13.0, 0.0), half, 1.0) - 3.0).abs() < 1e-5);
        // The corner rounded: its tip is outside.
        assert!(round_box(v2(10.0, 5.0), half, 1.0) > 0.4);
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
    fn the_rays_are_soft_beams_fading_out_and_sweeping_round() {
        let round = |t: f32| -> Vec<f32> {
            (0..360)
                .map(|k| {
                    let a = (k as f32).to_radians();
                    ray_fan(a.cos() * 0.1, a.sin() * 0.1, t, 0.2)
                })
                .collect()
        };
        let now = round(3.0);
        let most = now.iter().copied().fold(0.0, f32::max);
        let least = now.iter().copied().fold(1.0, f32::min);
        assert!(most > 0.6 && least < 0.15 * most, "{most} {least}");
        // Soft: a beam is broad, a good share of the way round lit.
        let lit = now.iter().filter(|&&r| r > 0.3 * most).count();
        assert!(lit > 60 && lit < 240, "{lit}");
        // Sweeping: a twentieth of a second on, a little; a second on, a
        // lot.
        let moved = |t: f32| -> f32 {
            let then = round(t);
            now.iter()
                .zip(&then)
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / 360.0
        };
        assert!(
            moved(3.05) < 0.05 && moved(4.0) > 0.08,
            "{} {}",
            moved(3.05),
            moved(4.0)
        );
        assert!(ray_fan(0.0, 3.0, 3.0, 0.2) < 0.2 * ray_fan(0.0, 0.05, 3.0, 0.2));
    }

    #[test]
    fn the_corners_are_darker() {
        assert_eq!(vignette(0.5, 0.5), 1.0);
        assert!(vignette(0.0, 0.0) < 0.7);
    }
}
