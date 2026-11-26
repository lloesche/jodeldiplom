//! What each tier's row of the tier table ([`crate::tiers::TIERS`]) plays
//! on top of the conductor's sounds: a small list of layers for the
//! streak at its rung ([`Streak`]) and one for a phrase's chime reaching
//! it ([`Chime`], its brass the chords each layer names: [`Hit`]),
//! together a row's [`Layers`]. How big a sound is beyond
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
    /// Its multiplier called out, sung ("times thir-ty-two!"), from its
    /// stinger's hit on (where its slam lands: [`super::call`]).
    Call,
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
    /// The tier's name called out, sung ("di-va!"), from the first beat
    /// of a bar with room for it before a sung line on (where a phrase's
    /// slam lands, with the jackpot: [`super::call`]); a phrase's only,
    /// not the diploma's points' (no slam there).
    Call,
}

/// A brass chord of a chime's layer ([`Chime::brass`]): where it starts
/// against the chime's hit, how long it holds, how loud, on how many of
/// the chord's tones.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Hit {
    /// Beats before the chime's hit it starts (left out where that is
    /// before the moment it celebrates: no room for it).
    pub(super) before: f64,
    /// Beats held.
    pub(super) length: f64,
    /// Its velocity.
    pub(super) velocity: f32,
    /// The chord's tones it plays, from the tonic up.
    pub(super) tones: i32,
}

/// The short brass hit: on the beat, half a beat long.
const BRASS_HIT: [Hit; 1] = [Hit {
    before: 0.0,
    length: 0.5,
    velocity: 0.55,
    tones: 3,
}];

/// The "ta-daa": a short hit a sixteenth before the beat, the long one on
/// it, a fuller chord held a beat and a half.
const TA_DAA: [Hit; 2] = [
    Hit {
        before: 0.25,
        length: 0.2,
        velocity: 0.5,
        tones: 3,
    },
    Hit {
        before: 0.0,
        length: 1.5,
        velocity: 0.65,
        tones: 4,
    },
];

impl Chime {
    /// Its brass chords (none: it plays no brass).
    pub(super) fn brass(self) -> &'static [Hit] {
        match self {
            Chime::BrassHit => &BRASS_HIT,
            Chime::TaDaa => &TA_DAA,
            Chime::Plucks | Chime::Choir | Chime::Call => &[],
        }
    }
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

/// The brass chords of a phrase's chime at `tier` (none: under the
/// first), its layers' in their order.
pub(super) fn chime_brass(tier: Option<Tier>) -> impl Iterator<Item = &'static Hit> {
    let layers = tier.map_or(&[][..], |t| t.spec().layers.chime);
    layers.iter().flat_map(|c| c.brass())
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
