//! What each tier's row of the tier table ([`crate::tiers::TIERS`]) plays
//! on top of the conductor's sounds: a small list of layers for the
//! streak at its rung ([`Streak`]) and one for a phrase's chime reaching
//! it ([`Chime`]), together a row's [`Layers`]. How big a sound is beyond
//! that (how many notes, how loud, how long) follows from where its tier
//! is among the rows, its share of the way to the top
//! ([`rung_share`], [`tier_share`]); so a new tier is one row more.

use crate::tiers::Tier;

/// A layer the streak plays at a tier's rung.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Streak {
    /// A glint an octave over each word's pop.
    Glint,
    /// The stinger's hit doubled an octave down, a bell struck on it.
    Octave,
    /// The stinger's bell tremolo and the register's ka-ching.
    Register,
    /// A coin's clink after each word.
    Clinks,
    /// The streak lost from it: the reels running down (else two soft
    /// notes falling).
    Reels,
    /// A brass chord held under the stinger.
    Brass,
    /// A riser leading into it from the word before.
    Riser,
    /// The shimmer pad held while the streak is at it.
    Shimmer,
    /// Glitter over its stinger and after each word.
    Glitter,
    /// Its payout on a downbeat after its stinger.
    Payout,
}

/// A layer of a phrase's chime reaching a tier (and of the diploma's
/// points at it), over the bells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chime {
    /// Plucks running up the chord.
    Plucks,
    /// A short brass hit.
    BrassHit,
    /// A "ta-daa" in brass (in place of the short hit).
    TaDaa,
    /// A choir's chord, the low tonic and glitter.
    Choir,
}

/// A tier's layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layers {
    /// The streak's at its rung.
    pub streak: &'static [Streak],
    /// A phrase's chime's.
    pub chime: &'static [Chime],
}

/// The rungs: rung `r` (1 up) is reached at `RUNGS[r - 1]` words in a
/// row, the tiers' streaks.
pub(super) fn rungs() -> [u32; Tier::COUNT] {
    Tier::ALL.map(|t| t.spec().streak)
}

/// The layers at rung `r` (0: under the first, none).
pub(super) fn at_rung(r: u32) -> &'static [Streak] {
    (r as usize)
        .checked_sub(1)
        .and_then(|k| Tier::ALL.get(k))
        .map_or(&[], |t| t.spec().layers.streak)
}

/// Whether rung `r` plays `layer`.
pub(super) fn rung_has(r: u32, layer: Streak) -> bool {
    at_rung(r).contains(&layer)
}

/// Whether a phrase's chime at `tier` (none: under the first) plays
/// `layer`.
pub(super) fn chime_has(tier: Option<Tier>, layer: Chime) -> bool {
    tier.is_some_and(|t| t.spec().layers.chime.contains(&layer))
}

/// How far rung `r` is from the first rung playing `layer` to the top, 0
/// to 1 (none: it plays none).
pub(super) fn reach(r: u32, layer: Streak) -> Option<f32> {
    if !rung_has(r, layer) {
        return None;
    }
    let first = (1..=Tier::COUNT as u32).find(|&k| rung_has(k, layer))?;
    let top = Tier::COUNT as u32;
    Some(match top - first {
        0 => 1.0,
        span => (r - first) as f32 / span as f32,
    })
}

/// Rung `r`'s share of the way to the top (1 up to the rungs).
pub(super) fn rung_share(r: u32) -> f32 {
    (r as f32 / Tier::COUNT as f32).clamp(0.0, 1.0)
}

/// Tier `t`'s (none: under the first, 0).
pub(super) fn tier_share(t: Option<Tier>) -> f32 {
    t.map_or(0.0, |t| (t.index() + 1) as f32 / Tier::COUNT as f32)
}

/// The rung a tier's is (1 up).
pub(super) fn rung_of(t: Tier) -> u32 {
    t.index() as u32 + 1
}
