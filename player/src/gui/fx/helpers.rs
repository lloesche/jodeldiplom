//! What the effects' maths share: the noise their looks move by (the
//! stage's own [`lattice`] and [`vnoise`], which the paper reads too),
//! fire's colours (the theme's [`Ramp`]), the one fire every burning
//! effect draws ([`fire`]: the lane's flames, the marquee's bulbs on fire,
//! the screen's edges), where a line's letters sit in their quad, and how
//! fast a look moves; on the processor here, and in WGSL ([`wgsl`]:
//! `fx_heat`, `fx_fire`, `fx_fire_light`, `fx_letter_up`, `fx_speed`, with
//! the screen passes' own, [`super::screen_pass::WGSL`]) once in a shader
//! that carries any effect's code ([`crate::gui::stage::gpu::shader`]).

pub(super) use crate::gui::stage::shapes::glory::{lattice, vnoise};
pub(super) use jodel_looks::math::{mix3, sat, smoothstep};

use crate::theme::Ramp;

/// Fire's colour at `heat` (0 to 1) in `fire`'s colours: its first at
/// none, easing to each next by the heats [`Ramp`] names, white hot (as
/// made) at 1.
pub(super) fn heat(k: f32, fire: &Ramp) -> [f32; 3] {
    let [c0, c1, c2, c3, c4] = fire.0;
    let c = mix3(c0, c1, smoothstep(0.0, 0.3, k));
    let c = mix3(c, c2, smoothstep(0.25, 0.55, k));
    let c = mix3(c, c3, smoothstep(0.5, 0.8, k));
    mix3(c, c4, smoothstep(0.8, 1.0, k))
}

/// Fire's heat at `a` along the line it burns from and `d` out from it
/// (both in its flames' heights; `d` negative behind the line) at `t`
/// seconds: tongues of flame licking out from the line, rising and
/// flickering, swaying more towards their tips, ragged at their edges;
/// about 1 at their roots (more in the hottest), falling to none at their
/// tips, the tallest some 1.45 heights out; behind the line a little of
/// it, gone a few tenths of a height back. Its light: [`fire_light`]. An
/// effect chooses only where its line is, how high its flames are (taller
/// on a beat: the height `d` is in grown, `a`'s kept, so the tongues stay
/// where they are) and how strong.
pub(super) fn fire(a: f32, d: f32, t: f32) -> f32 {
    let rise = t * 2.4;
    let sway = (vnoise(a * 1.1 + 3.3, d * 0.7 - t * 0.9) - 0.5) * (0.2 + d.max(0.0)) * 0.9;
    let x = a + sway;
    let n = 0.55 * vnoise(x * 4.0, d * 1.4 - rise)
        + 0.3 * vnoise(x * 8.5 + 7.0, d * 2.8 - rise * 1.6)
        + 0.15 * vnoise(x * 17.0 + 3.0, d * 5.6 - rise * 2.5);
    let reach = 0.08 + 1.4 * n * n;
    let flame = if d >= 0.0 {
        (1.0 - d / reach).clamp(0.0, 1.0).powf(1.3)
    } else {
        0.5 * (d / 0.06).exp()
    };
    flame * (0.45 + 0.85 * n)
}

/// How far out [`fire`] reaches at most, in its flames' heights.
pub(super) const TALLEST_FLAME: f32 = 1.48;

/// Fire's light at heat `k` ([`fire`]): its colour ([`heat`], in `fire`'s
/// colours) as bright as it is hot, the white-hot roots brightest, the
/// red tips dim.
pub(super) fn fire_light(k: f32, fire: &Ramp) -> [f32; 3] {
    heat(k, fire).map(|c| c * k * 1.1)
}

/// Where a line's letters sit in its text's quad, as the stage lays text
/// out (its line's top at the quad's top, [`crate::gui::big_text::COVER_PAD`]
/// round it): their capitals' tops and their feet (the baseline), shares
/// of the way down. A face's size is its ascent and descent together, and
/// the quad is 1.3 sizes high: for the game's line face, the baseline
/// about 0.57 of the way down, the capitals' tops about 0.18.
const CAPS: f32 = 0.18;
const FEET: f32 = 0.57;

/// How far up a line's letters `y` (pixels) is in its text's quad `bx`
/// (left, top, right, bottom): 0 at their feet, 1 at their capitals' tops,
/// below 0 under them.
pub(super) fn letter_up(y: f32, bx: [f32; 4]) -> f32 {
    let high = (bx[3] - bx[1]).max(1.0);
    (FEET - (y - bx[1]) / high) / (FEET - CAPS)
}

/// How fast a look moves: its look's `y` (1, as fast as it was made;
/// none or less, taken as 1).
pub(super) fn speed(p: [f32; 4]) -> f32 {
    if p[1] > 0.0 {
        p[1]
    } else {
        1.0
    }
}

/// [`heat`], [`fire`], [`fire_light`], [`letter_up`] and [`speed`] in
/// WGSL, and the screen passes' scaffolding ([`super::screen_pass::WGSL`]).
pub fn wgsl() -> String {
    let screen_pass = super::screen_pass::WGSL;
    format!(
        "
const FX_CAPS: f32 = {CAPS:?};
const FX_FEET: f32 = {FEET:?};
const FX_TALLEST_FLAME: f32 = {TALLEST_FLAME:?};
{HELPERS}{screen_pass}"
    )
}

const HELPERS: &str = r"
fn fx_heat(k: f32, fire: array<vec4<f32>, FIRE_STOPS>) -> vec3<f32> {
    var c = mix(fire[0].rgb, fire[1].rgb, smoothstep(0.0, 0.3, k));
    c = mix(c, fire[2].rgb, smoothstep(0.25, 0.55, k));
    c = mix(c, fire[3].rgb, smoothstep(0.5, 0.8, k));
    return mix(c, fire[4].rgb, smoothstep(0.8, 1.0, k));
}

fn fx_fire(a: f32, d: f32, t: f32) -> f32 {
    let rise = t * 2.4;
    let sway = (vnoise(a * 1.1 + 3.3, d * 0.7 - t * 0.9) - 0.5) * (0.2 + max(d, 0.0)) * 0.9;
    let x = a + sway;
    let n = 0.55 * vnoise(x * 4.0, d * 1.4 - rise)
        + 0.3 * vnoise(x * 8.5 + 7.0, d * 2.8 - rise * 1.6)
        + 0.15 * vnoise(x * 17.0 + 3.0, d * 5.6 - rise * 2.5);
    let reach = 0.08 + 1.4 * n * n;
    var flame = 0.5 * exp(min(d, 0.0) / 0.06);
    if d >= 0.0 {
        flame = pow(clamp(1.0 - d / reach, 0.0, 1.0), 1.3);
    }
    return flame * (0.45 + 0.85 * n);
}

fn fx_fire_light(k: f32, fire: array<vec4<f32>, FIRE_STOPS>) -> vec3<f32> {
    return fx_heat(k, fire) * (k * 1.1);
}

fn fx_letter_up(y: f32, box: vec4<f32>) -> f32 {
    let high = max(box.w - box.y, 1.0);
    return (FX_FEET - (y - box.y) / high) / (FX_FEET - FX_CAPS);
}

fn fx_speed(p: vec4<f32>) -> f32 {
    return select(1.0, p.y, p.y > 0.0);
}
";
