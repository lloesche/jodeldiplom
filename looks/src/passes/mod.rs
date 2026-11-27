//! The library of passes, family by family, each a [`Pass`] with its WGSL
//! and its maths on the processor side by side, and the helpers the
//! families share ([`common`]).

pub mod atmos;
pub mod common;
pub mod film;
pub mod grade;
pub mod medium;
pub mod paint;
pub mod print;
pub mod psych;
pub mod retro;
pub mod visual;

use crate::pass::Pass;

/// Every pass, family by family.
pub static PASSES: &[&Pass] = &[
    &grade::LIFT_GAMMA_GAIN,
    &grade::WHITE_BALANCE,
    &grade::SATURATION,
    &grade::DUOTONE,
    &grade::TRITONE,
    &grade::SPLIT_TONE,
    &grade::CURVE,
    &grade::GRADIENT_MAP,
    &grade::TECHNICOLOR,
    &grade::BLEACH_BYPASS,
    &grade::CROSS_PROCESS,
    &grade::INFRARED,
    &grade::SEPIA,
    &film::GRAIN,
    &film::HALATION,
    &film::GLOW,
    &film::DIFFUSION,
    &film::GATE_WEAVE,
    &film::LIGHT_LEAK,
    &film::VIGNETTE,
    &film::POLAROID,
    &film::FILM_DAMAGE,
    &film::LETTERBOX,
    &film::BOKEH,
    &print::HALFTONE,
    &print::HALFTONE_CMYK,
    &print::BEN_DAY,
    &print::POSTERIZE_INK,
    &print::RISOGRAPH,
    &print::XEROX,
    &print::WOODCUT,
    &print::ENGRAVING,
    &print::STIPPLE,
    &paint::KUWAHARA,
    &paint::WATERCOLOR,
    &paint::OIL_PAINT,
    &paint::CANVAS,
    &paint::CHARCOAL,
    &paint::PASTEL,
    &paint::MARKER,
    &paint::PENCIL,
    &paint::CHALKBOARD,
    &medium::STAINED_GLASS,
    &medium::MOSAIC,
    &medium::CROSS_STITCH,
    &medium::CYANOTYPE,
    &medium::THERMAL,
    &medium::NIGHT_VISION,
    &medium::NEGATIVE,
    &medium::SOLARIZE,
    &medium::NEON_EDGES,
    &medium::TELETEXT,
    &medium::CUTOUT,
    &medium::EMBOSS,
    &retro::CRT,
    &retro::VHS,
    &retro::PIXELATE,
    &retro::PALETTE,
    &retro::GLITCH,
    &retro::ANAGLYPH,
    &retro::CHROMATIC,
    &atmos::HAZE,
    &atmos::GOD_RAYS,
    &atmos::FOG,
    &atmos::RAIN,
    &atmos::SNOW,
    &atmos::PETALS,
    &atmos::DUST,
    &atmos::HEAT_HAZE,
    &atmos::UNDERWATER,
    &atmos::CANDLE,
    &atmos::AURORA,
    &atmos::STARFIELD,
    &atmos::LENS_FLARE,
    &atmos::TILT_SHIFT,
    &atmos::CONFETTI,
    &psych::HUE,
    &psych::KALEIDOSCOPE,
    &psych::MIRROR,
    &psych::LIQUID,
    &psych::ZOOM_BLUR,
    &visual::FEEDBACK,
    &visual::SCOPE,
    &visual::SPECTRUM_RING,
    &visual::BARS,
    &visual::BEAT_RINGS,
    &visual::SPARKS,
    &visual::LISSAJOUS,
];

/// The pass named `name`.
pub fn named(name: &str) -> Option<&'static Pass> {
    PASSES.iter().copied().find(|p| p.name == name)
}
