//! What the effects' maths share: the noise their looks move by (the
//! stage's own [`lattice`] and [`vnoise`], which the flames and the paper
//! read too), fire's colours, where a line's letters sit in their quad,
//! how fast a look moves; on the processor here, and in WGSL ([`WGSL`]:
//! `fx_heat`, `fx_letter_up`, `fx_speed`) once in a shader that carries
//! any effect's code ([`crate::gui::stage_gpu::shader`]).

pub(super) use crate::gui::glory::{lattice, vnoise};
pub(super) use crate::gui::rainbow::{clamp01, mix3, smoothstep};

/// Fire's colour at `heat` (0 to 1): deep red, red, orange, yellow, white
/// hot; `ghost` (0 to 1) of the way to a ghostly green, the same heat in
/// greens.
pub(super) fn heat(k: f32, ghost: f32) -> [f32; 3] {
    let c = mix3(
        [0.45, 0.03, 0.01],
        [1.0, 0.2, 0.02],
        smoothstep(0.0, 0.3, k),
    );
    let c = mix3(c, [1.0, 0.55, 0.08], smoothstep(0.25, 0.55, k));
    let c = mix3(c, [1.0, 0.88, 0.35], smoothstep(0.5, 0.8, k));
    let c = mix3(c, [1.0, 0.98, 0.85], smoothstep(0.8, 1.0, k));
    let green = [c[2] * 0.8 + c[1] * 0.25, c[0], c[1] * 0.45 + c[2] * 0.5];
    mix3(c, green, clamp01(ghost))
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

/// [`heat`], [`letter_up`] and [`speed`] in WGSL.
pub fn wgsl() -> String {
    format!(
        "
const FX_CAPS: f32 = {CAPS:?};
const FX_FEET: f32 = {FEET:?};
{HELPERS}"
    )
}

const HELPERS: &str = r"
fn fx_heat(k: f32, ghost: f32) -> vec3<f32> {
    var c = mix(vec3<f32>(0.45, 0.03, 0.01), vec3<f32>(1.0, 0.2, 0.02), smoothstep(0.0, 0.3, k));
    c = mix(c, vec3<f32>(1.0, 0.55, 0.08), smoothstep(0.25, 0.55, k));
    c = mix(c, vec3<f32>(1.0, 0.88, 0.35), smoothstep(0.5, 0.8, k));
    c = mix(c, vec3<f32>(1.0, 0.98, 0.85), smoothstep(0.8, 1.0, k));
    let green = vec3<f32>(c.z * 0.8 + c.y * 0.25, c.x, c.y * 0.45 + c.z * 0.5);
    return mix(c, green, clamp(ghost, 0.0, 1.0));
}

fn fx_letter_up(y: f32, box: vec4<f32>) -> f32 {
    let high = max(box.w - box.y, 1.0);
    return (FX_FEET - (y - box.y) / high) / (FX_FEET - FX_CAPS);
}

fn fx_speed(p: vec4<f32>) -> f32 {
    return select(1.0, p.y, p.y > 0.0);
}
";
