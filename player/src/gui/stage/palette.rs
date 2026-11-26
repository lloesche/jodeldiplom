//! The stage's own colours (sRGB, 0 to 1, as [`super::Instance`] takes
//! them): what its lane, its arrow, its text and its diploma are drawn in,
//! and the gold of the words scored that the bubbles' bursts take too
//! ([`crate::gui::pop`]).

/// sRGB colours, 0 to 1.
pub type Rgb = [f32; 3];

/// White: the text, the playhead, the glints.
pub const WHITE: Rgb = [1.0, 1.0, 1.0];
/// The lane's glass, dark and a little cool.
pub const SMOKE: Rgb = [0.02, 0.025, 0.05];
/// Gold at its lightest: a gold shape's top, its glow.
pub const GOLD_TOP: Rgb = [1.0, 0.93, 0.6];
/// Gold: what was hit, the line sung, the multiplier.
pub const GOLD: Rgb = [1.0, 0.72, 0.2];
/// Gold at its deepest: a gold shape's bottom.
pub const GOLD_DEEP: Rgb = [0.98, 0.55, 0.12];
/// Coral: the arrow off the notes.
pub const CORAL: Rgb = [0.95, 0.43, 0.31];
/// The warm light of what is active.
pub const ACTIVE: Rgb = [1.0, 0.86, 0.5];

/// `c` with its alpha.
pub fn rgba(c: Rgb, a: f32) -> [f32; 4] {
    [c[0], c[1], c[2], a]
}
