//! The shine: each phrase (a sung line) of the singing measured
//! ([`measure`]) and held against the original singer's same phrase (the
//! song's vocals measured the same way, its [`Yardstick`]: [`reference`]),
//! scored as a share of it when the phrase ends, and the tier it reaches
//! ([`Tier`]).
//!
//! A phrase's shine is its coverage times its quality ([`shine`]). The
//! coverage: its words sung as the bubbles count them, each against what
//! the original has on it, softened (`(c + 0.1) / 1.1`). The quality: the
//! measures' shares of the original's, each capped and weighted; a measure
//! the original has none of is left out (the others weigh more), one the
//! singer has none of where the original has it counts 0. A phrase the
//! original is heard too little on (under two words counted, or under
//! [`VOICED`] of its note time) has no shine: it is sung for its bubbles
//! only; nor has one its everyday hearings reach too little on
//! ([`reference::PAR_LEAST`]).
//!
//! The score is the original's own: its everyday hearings (another rate,
//! the delay off, octave errors, the pitch lost now and then) reach the
//! top tier's bar ([`Tier::TOP`], Diva) on every phrase, since each
//! phrase's bars are set from the least they reach ([`Par`],
//! [`Tier::of`]).

pub mod measure;
pub mod reference;
mod text;

pub use measure::{Measured, Yardstick};
pub use reference::{Par, Phrase, Reference};

pub use crate::tiers::Tier;

/// How late the frames come against the song's notes, the frame clock's
/// own (one frame): taken off the frames the shine measures, the
/// original's and the singer's alike.
pub const FRAME_LATE: f64 = 0.01;

/// A frame heard for the shine: when in the song (seconds, [`FRAME_LATE`]
/// taken off), its pitch (hundredths of a MIDI note; none: not sung), and
/// how far the singer sang from the song up to it (octaves aside; none
/// before any frame on a note).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    /// When.
    pub t: f64,
    /// Its pitch.
    pub cents: Option<f32>,
    /// The singer's key then.
    pub key: Option<f64>,
}

/// The fewest words of a phrase the original must be counted on for it to
/// shine.
pub const WORDS: usize = 2;
/// The least share of its note time the original must be voiced on.
pub const VOICED: f64 = 0.15;
/// The coverage's softening.
const COVERAGE_K: f64 = 0.10;
/// The cents added to the singer's and the original's errors (pitch off,
/// spread): a near perfect original is not a yardstick no one reaches.
const ERROR_C: f64 = 10.0;
/// A score's softening (timing, sustain, range): `(m + k) / (r + k)`.
const SCORE_K: f64 = 0.15;

/// The quality measures' weights.
const PITCH: f64 = 0.45;
const TIMING: f64 = 0.20;
const STEADY: f64 = 0.15;
const SUSTAIN: f64 = 0.10;
const RANGE: f64 = 0.10;
/// The most a measure's share counts (over 1: better than the original
/// counts a little); timing and sustain at most the original's.
const CAP: f64 = 1.10;
const CAP_TIMING: f64 = 1.0;
const CAP_SUSTAIN: f64 = 1.0;

/// A phrase's shine: its value (the original about 1), the coverage's
/// share and each quality measure's (none: not measured, the original
/// having none of it).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Shine {
    /// The shine.
    pub value: f64,
    /// The coverage's share.
    pub coverage: f64,
    /// How near the notes: the original's error over the singer's.
    pub pitch: Option<f64>,
    /// The voicing's lag against the original's.
    pub timing: Option<f64>,
    /// The spread on held notes: the original's over the singer's.
    pub steady: Option<f64>,
    /// The long notes held, each against the original's.
    pub sustain: Option<f64>,
    /// The top and bottom reached.
    pub range: Option<f64>,
}

/// Whether the original's phrase `y` is heard enough to hold a singer to.
pub fn shines(y: &Yardstick) -> bool {
    y.counted() >= WORDS && y.voiced >= VOICED
}

/// Phrase `p` (as sung) against `y`, the original's same phrase.
pub fn shine(p: &Measured, y: &Yardstick) -> Shine {
    let coverage = ((p.coverage + COVERAGE_K) / (1.0 + COVERAGE_K)).min(1.0);
    let error =
        |m: Option<f64>, r: f64| m.map_or(0.0, |m| ((r + ERROR_C) / (m + ERROR_C)).min(CAP));
    let score = |m: f64, r: f64, cap: f64| ((m + SCORE_K) / (r + SCORE_K)).min(cap);
    let mut out = Shine {
        coverage,
        pitch: y.off.map(|r| error(p.off, r)),
        timing: y
            .times()
            .then(|| score(p.timing.unwrap_or(0.0), 1.0, CAP_TIMING)),
        steady: y.spread.map(|r| error(p.spread, r)),
        sustain: (!y.held.is_empty()).then(|| {
            let each = p
                .held
                .iter()
                .zip(&y.held)
                .map(|(m, r)| score(*m, *r, CAP_SUSTAIN));
            each.sum::<f64>() / y.held.len() as f64
        }),
        range: y.range.map(|r| score(p.range.unwrap_or(0.0), r, CAP)),
        ..Shine::default()
    };
    let parts = [
        (out.pitch, PITCH),
        (out.timing, TIMING),
        (out.steady, STEADY),
        (out.sustain, SUSTAIN),
        (out.range, RANGE),
    ];
    let (sum, weights) = parts
        .iter()
        .filter_map(|&(share, w)| Some((share? * w, w)))
        .fold((0.0, 0.0), |(s, ws), (x, w)| (s + x, ws + w));
    let quality = if weights > 0.0 { sum / weights } else { 1.0 };
    out.value = coverage * quality;
    out
}

/// The tiers a phrase may reach ([`crate::tiers::TIERS`]), each bar a
/// share of what the original reaches.
impl Tier {
    /// The tier `shine` reaches on a phrase whose original reaches `par`
    /// (its everyday hearings' least): the highest whose bars it clears,
    /// each bar times the original's shine where it is under 1, a tier's
    /// pitch bar (where it has one) times the original's pitch.
    pub fn of(shine: &Shine, par: Par) -> Option<Tier> {
        let scale = par.scale();
        let clears = |t: Tier| {
            let pitch_ok = t
                .spec()
                .pitch_bar
                .is_none_or(|bar| shine.pitch.is_none_or(|p| p >= bar * par.pitch_scale()));
            shine.value >= t.bar() * scale - 1e-9 && pitch_ok
        };
        Tier::ALL.into_iter().rev().find(|&t| clears(t))
    }
}

/// A phrase as it shone, scored when it ended: its line, when it ended
/// (seconds into the song), its shine (each against its own bar for the
/// top tier, 1: the top's) and the tier it reached.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shone {
    /// Its line (among the sung lines).
    pub line: usize,
    /// When it ended.
    pub end: f64,
    /// Its shine.
    pub value: f32,
    /// The top tier's bar for it (its shine at which it is the top, the
    /// pitch aside).
    pub top: f32,
    /// The tier reached, if any.
    pub tier: Option<Tier>,
}

impl Shone {
    /// `line`'s phrase ending at `end` shone `shine`, its original at `par`.
    pub fn new(line: usize, end: f64, shine: &Shine, par: Par) -> Shone {
        Shone {
            line,
            end,
            value: shine.value as f32,
            top: (Tier::TOP.bar() * par.scale()) as f32,
            tier: Tier::of(shine, par),
        }
    }

    /// Its shine against the top tier's bar (1: the top's; the meter's
    /// fill).
    pub fn of_top(&self) -> f32 {
        self.value / self.top.max(1e-3)
    }
}

#[cfg(test)]
mod tests;
