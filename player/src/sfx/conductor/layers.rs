//! What each tier's row of the tier table ([`crate::tiers::TIERS`]) plays
//! on top of the conductor's sounds: the layers the streak takes on at its
//! rung ([`Streak`], [`Layers::adds`]: a rung plays its own and every rung's
//! below, [`at_rung`]) and those of a phrase's chime reaching it
//! ([`Chime`], its brass the chords each layer names: [`Hit`]), together a
//! row's [`Layers`]. How big a sound is beyond that (how many notes, how
//! loud, how long) follows from where its tier is among the rows, its
//! share of the way to the top ([`share`]); so a new tier is one row more.

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

impl Streak {
    /// Every layer, in their order.
    pub const ALL: [Streak; 11] = [
        Streak::Glint,
        Streak::Octave,
        Streak::Register,
        Streak::Clinks,
        Streak::Reels,
        Streak::Brass,
        Streak::Riser,
        Streak::Shimmer,
        Streak::Glitter,
        Streak::Payout,
        Streak::Call,
    ];

    /// The first tier whose rung plays it (none: no row adds it).
    pub fn first_tier(self) -> Option<Tier> {
        (Tier::ALL.into_iter()).find(|t| t.spec().layers.adds.contains(&self))
    }

    /// Its name, as the Synth tab shows it.
    pub fn name(self) -> &'static str {
        match self {
            Streak::Glint => "glint",
            Streak::Octave => "octave",
            Streak::Register => "register",
            Streak::Clinks => "clinks",
            Streak::Reels => "reels",
            Streak::Brass => "brass",
            Streak::Riser => "riser",
            Streak::Shimmer => "shimmer",
            Streak::Glitter => "glitter",
            Streak::Payout => "payout",
            Streak::Call => "call",
        }
    }
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

impl Chime {
    /// Its name, as the Synth tab shows it.
    pub fn name(self) -> &'static str {
        match self {
            Chime::Plucks => "plucks",
            Chime::BrassHit => "brass hit",
            Chime::TaDaa => "ta-daa",
            Chime::Choir => "choir",
            Chime::Call => "call",
        }
    }
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

/// A set of the streak's layers ([`Streak`]), one bit a layer: as cheap
/// to ask whether it holds one as to read a bit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreakSet(u16);

// Every layer has its bit.
const _: () = assert!(Streak::ALL.len() <= 16);

impl StreakSet {
    /// No layers.
    pub const NONE: StreakSet = StreakSet(0);

    /// The set of `layers`.
    pub const fn of(layers: &[Streak]) -> StreakSet {
        let mut bits = 0;
        let mut i = 0;
        while i < layers.len() {
            bits |= 1 << layers[i] as u16;
            i += 1;
        }
        StreakSet(bits)
    }

    /// This set and `other`'s layers.
    pub const fn with(self, other: StreakSet) -> StreakSet {
        StreakSet(self.0 | other.0)
    }

    /// Whether it holds `layer`.
    pub const fn contains(self, layer: Streak) -> bool {
        self.0 & (1 << layer as u16) != 0
    }

    /// Its layers, in their order.
    pub fn iter(self) -> impl Iterator<Item = Streak> {
        Streak::ALL.into_iter().filter(move |&l| self.contains(l))
    }
}

/// A tier's layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layers {
    /// The streak's its rung adds to the rung's below (it plays those and
    /// these: [`at_rung`]).
    pub adds: &'static [Streak],
    /// A phrase's chime's.
    pub chime: &'static [Chime],
}

/// The rungs: a tier's is reached at its streak, words in a row, Good's
/// first.
pub(super) fn rungs() -> [u32; Tier::COUNT] {
    Tier::ALL.map(|t| t.spec().streak)
}

/// The streak's layers at `rung` (none: under the first, none): what
/// every row up to it adds.
pub fn at_rung(rung: Option<Tier>) -> StreakSet {
    let up_to = rung.map_or(0, |t| t.index() + 1);
    (Tier::ALL[..up_to].iter()).fold(StreakSet::NONE, |s, t| {
        s.with(StreakSet::of(t.spec().layers.adds))
    })
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

/// How far `rung` is from the first rung playing `layer` to the top, 0
/// to 1 (none: it plays none).
pub(super) fn reach(rung: Tier, layer: Streak) -> Option<f32> {
    let first = layer.first_tier().filter(|&f| f <= rung)?;
    Some(match Tier::TOP.index() - first.index() {
        0 => 1.0,
        span => (rung.index() - first.index()) as f32 / span as f32,
    })
}

/// `tier`'s share of the way to the top (none: under the first, 0; the
/// top's 1): how big its sounds are, a phrase's chime at it or its rung's.
pub(super) fn share(tier: Option<Tier>) -> f32 {
    tier.map_or(0.0, |t| (t.index() + 1) as f32 / Tier::COUNT as f32)
}
