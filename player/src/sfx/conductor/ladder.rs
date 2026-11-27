//! The per-word pitch ladder (the combo's rising pitch): each word hit in
//! a row pops a step higher
//! up the song's pentatonic, and the word that reaches a rung lands on
//! the tonic an octave over where the climb began, the resolution the
//! rung is (Peggle's rising peg hits, a scale in key, as described in its
//! patent US 8,128,476 and by its audio team). The climb from one rung to
//! the next always spans that octave, however many words it takes: rungs
//! a few words apart climb a step a word, rungs many words apart a step
//! every few words (so the register never runs away); past the top rung
//! it climbs on in octaves the length of the last. Which rung a word
//! reaches is the game's ([`crate::core::singing::rung`]).

use super::key::DEGREES;

/// The degree over the climb's tonic (1 to [`DEGREES`], the octave: a
/// rung's word) of the word making a run of `streak` (1: the first of a
/// run) when the rungs are reached at `rungs` words in a row (strictly
/// increasing, at least one).
pub fn degree(streak: u32, rungs: &[u32]) -> i32 {
    let streak = streak.max(1);
    let top = rungs.last().copied().unwrap_or(1).max(1);
    let (from, length) = match rungs.iter().position(|&r| r >= streak) {
        Some(0) => (0, rungs[0]),
        Some(k) => (rungs[k - 1], rungs[k] - rungs[k - 1]),
        None => {
            let last = match rungs.len() {
                0 | 1 => top,
                n => top - rungs[n - 2],
            }
            .max(1);
            (top + last * ((streak - top - 1) / last), last)
        }
    };
    let j = streak - from;
    let degree = (DEGREES as f32 * j as f32 / length.max(1) as f32).round() as i32;
    degree.clamp(1, DEGREES)
}
