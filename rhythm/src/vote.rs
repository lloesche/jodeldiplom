//! The vote: the grid the candidates agree on, and how sure they are.
//!
//! Every candidate (an activation tracked by a tracker) is compared with
//! every other by their [`agreement`]: the F-measure beyond chance at the
//! tempo level where the two agree best, so a grid at twice the tempo of
//! another still agrees on where the beats are. The candidate that agrees
//! most with the rest, each counted by its weight, wins, and that mean
//! agreement is the confidence. Candidates that fail differently
//! disagree, so a low confidence flags music where the beat is unclear.
//!
//! A stem's candidate ([`Candidate::say`]) votes only where its stem
//! plays: it is compared with the others over those stretches alone, and
//! counts by how much of the track they cover and how surely the stem
//! plays there. It judges the others but does not win, since its beats
//! cover only part of the track. Where no stem plays, the vote is that of
//! the other candidates alone.
//!
//! Which level of the winner's pulse is the beat is the meter's to decide
//! ([`crate::meter`]). Once it has, [`regions`] cuts the beats into
//! stretches of 16 with the candidates' agreement in each, so later steps
//! can tell where the beat is clear.

use crate::activation::Curve;
use crate::metrics::{agreement, between};
use crate::presence::{self, KEEP};

/// The beat-matching tolerance, in seconds.
pub const TOLERANCE: f64 = 0.07;
/// Beats per region.
pub const REGION_BEATS: usize = 16;

/// One tracker's beats on one activation.
#[derive(Clone, Debug)]
pub struct Candidate {
    /// Its name, as `activation/tracker`.
    pub name: String,
    /// The beats, in seconds.
    pub beats: Vec<f64>,
    /// Its say in the vote and the meter.
    pub weight: f64,
    /// Where it may vote, frame by frame in `0..=1`: a stem's say
    /// ([`crate::presence::say`]); it has beats only where that is at
    /// least [`KEEP`]. `None`: everywhere.
    pub say: Option<Curve>,
}

impl Candidate {
    /// A candidate that votes everywhere.
    pub fn new(name: String, beats: Vec<f64>, weight: f64) -> Candidate {
        Candidate {
            name,
            beats,
            weight,
            say: None,
        }
    }

    /// Whether it votes only where its stem plays.
    pub fn is_partial(&self) -> bool {
        self.say.is_some()
    }

    /// Its mean say over `start..=end`: 1 for one that votes everywhere.
    pub fn say_between(&self, start: f64, end: f64) -> f64 {
        self.say
            .as_ref()
            .map_or(1.0, |say| presence::mean_between(say, start, end))
    }

    /// The stretches it votes on, `None` for everywhere.
    pub(crate) fn spans(&self) -> Option<Vec<(f64, f64)>> {
        self.say.as_ref().map(|say| presence::spans(say, KEEP))
    }

    /// How much of the track it votes on, by its say there: 1 for one that
    /// votes everywhere.
    fn coverage(&self) -> f64 {
        let Some(say) = &self.say else {
            return 1.0;
        };
        if say.values.is_empty() {
            return 0.0;
        }
        let kept: f64 = say
            .values
            .iter()
            .filter(|&&v| v >= KEEP)
            .map(|&v| f64::from(v))
            .sum();
        kept / say.values.len() as f64
    }
}

/// The beats of `beats` within `spans` (each widened by [`TOLERANCE`]),
/// each once and in order however close the spans lie.
pub(crate) fn within(beats: &[f64], spans: &[(f64, f64)]) -> Vec<f64> {
    beats
        .iter()
        .copied()
        .filter(|&b| {
            spans
                .iter()
                .any(|&(start, end)| b >= start - TOLERANCE && b <= end + TOLERANCE)
        })
        .collect()
}

/// A stretch of the result and how much the candidates agree there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Region {
    /// Its first beat, in seconds.
    pub start: f64,
    /// Its last beat.
    pub end: f64,
    /// The other candidates' agreement there (mean by weight), 0 to 1.
    pub confidence: f64,
}

/// What the vote decided.
#[derive(Clone, Debug)]
pub struct Outcome {
    /// The winning candidate's index.
    pub chosen: usize,
    /// The winner's agreement with the others (mean by weight), 0 to 1.
    pub confidence: f64,
    /// Each candidate's agreement with the others (mean by weight).
    pub agreements: Vec<f64>,
}

/// The candidates with at least four beats.
fn usable(candidates: &[Candidate]) -> Vec<usize> {
    (0..candidates.len())
        .filter(|&i| candidates[i].beats.len() >= 4)
        .collect()
}

/// Votes among `candidates`; `None` when no candidate has four beats.
/// Ties go to the earlier candidate. Only candidates that vote everywhere
/// win, unless there are none.
pub fn vote(candidates: &[Candidate]) -> Option<Outcome> {
    let usable = usable(candidates);
    let spans: Vec<Option<Vec<(f64, f64)>>> = candidates.iter().map(Candidate::spans).collect();
    let coverage: Vec<f64> = candidates.iter().map(Candidate::coverage).collect();
    let mut agreements = vec![0.0; candidates.len()];
    for &i in &usable {
        let (mut sum, mut weights) = (0.0, 0.0);
        for &j in usable.iter().filter(|&&j| j != i) {
            let (a, b) = (&candidates[i].beats, &candidates[j].beats);
            // A stem's candidate is compared where it votes, and counts by
            // how much of the track that is; two stems are not compared.
            let (agreed, w) = match (&spans[i], &spans[j]) {
                (None, None) => (agreement(a, b, TOLERANCE), 1.0),
                (None, Some(theirs)) => (agreement(&within(a, theirs), b, TOLERANCE), coverage[j]),
                (Some(ours), None) => (agreement(a, &within(b, ours), TOLERANCE), coverage[i]),
                (Some(_), Some(_)) => continue,
            };
            let w = w * candidates[j].weight;
            sum += w * agreed;
            weights += w;
        }
        if weights > 0.0 {
            agreements[i] = sum / weights;
        }
    }
    let full: Vec<usize> = usable
        .iter()
        .copied()
        .filter(|&i| !candidates[i].is_partial())
        .collect();
    let pool = if full.is_empty() { &usable } else { &full };
    let &first = pool.first()?;
    let chosen = pool.iter().copied().fold(first, |best, i| {
        if agreements[i] > agreements[best] {
            i
        } else {
            best
        }
    });
    Some(Outcome {
        chosen,
        confidence: agreements[chosen],
        agreements,
    })
}

/// `beats` in regions of [`REGION_BEATS`], each with the agreement there
/// of the candidates other than `chosen` with the chosen one's grid
/// (mean by weight; a stem's candidate also by its say there). The beats
/// are a grouping of that grid, so this is how sure the candidates are of
/// where they lie, whichever grouping the meter took.
pub fn regions(beats: &[f64], candidates: &[Candidate], chosen: usize) -> Vec<Region> {
    let others: Vec<usize> = usable(candidates)
        .into_iter()
        .filter(|&j| j != chosen)
        .collect();
    let grid = &candidates[chosen].beats;
    let spans: Vec<Option<Vec<(f64, f64)>>> = candidates.iter().map(Candidate::spans).collect();
    beats
        .chunks(REGION_BEATS)
        .filter(|chunk| chunk.len() >= 4)
        .map(|chunk| {
            let (start, end) = (chunk[0], chunk[chunk.len() - 1]);
            let ours = between(grid, start - TOLERANCE, end + TOLERANCE);
            let (mut sum, mut weights) = (0.0, 0.0);
            for &j in &others {
                let c = &candidates[j];
                let w = c.weight * c.say_between(start, end);
                if w <= 0.0 {
                    continue;
                }
                let theirs = between(&c.beats, start - TOLERANCE, end + TOLERANCE);
                // A stem's candidate against the grid where it votes.
                let agreed = match &spans[j] {
                    Some(spans) => agreement(&within(ours, spans), theirs, TOLERANCE),
                    None => agreement(ours, theirs, TOLERANCE),
                };
                sum += w * agreed;
                weights += w;
            }
            Region {
                start,
                end,
                confidence: if weights > 0.0 { sum / weights } else { 0.0 },
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(bpm: f64, phase: f64, count: usize) -> Vec<f64> {
        (0..count).map(|i| phase + i as f64 * 60.0 / bpm).collect()
    }

    fn candidate(name: &str, beats: Vec<f64>) -> Candidate {
        Candidate::new(name.into(), beats, 1.0)
    }

    /// A stem's candidate with `beats` that plays from `from` to `to`
    /// seconds of a track of `seconds`.
    fn partial(name: &str, beats: &[f64], from: f64, to: f64, seconds: f64) -> Candidate {
        let fps = 100.0;
        let values = (0..(seconds * fps) as usize)
            .map(|t| {
                let time = t as f64 / fps;
                if time >= from && time <= to {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        Candidate {
            say: Some(Curve { values, fps }),
            ..candidate(name, between(beats, from, to).to_vec())
        }
    }

    #[test]
    fn the_majority_wins_and_sets_the_confidence() {
        let truth = grid(120.0, 0.0, 64);
        let candidates = [
            candidate("a", truth.clone()),
            candidate("b", truth.iter().map(|b| b + 0.01).collect()),
            candidate("c", grid(97.0, 0.13, 52)),
        ];
        let outcome = vote(&candidates).unwrap();
        assert!(outcome.chosen < 2);
        assert!(
            outcome.confidence > 0.4 && outcome.confidence < 0.8,
            "{}",
            outcome.confidence
        );
        let regions = regions(&truth, &candidates, outcome.chosen);
        assert_eq!(regions.len(), 4);
        assert!(vote(&[candidate("a", vec![1.0, 2.0])]).is_none());
    }

    #[test]
    fn heavier_candidates_count_for_more() {
        let pairs = |weight: f64| {
            let mut candidates = vec![
                candidate("a", grid(120.0, 0.0, 64)),
                candidate("b", grid(120.0, 0.0, 64)),
                candidate("c", grid(97.0, 0.13, 52)),
                candidate("d", grid(97.0, 0.13, 52)),
            ];
            candidates[2].weight = weight;
            candidates[3].weight = weight;
            vote(&candidates).unwrap().chosen
        };
        // Two pairs that agree within each pair: the heavier pair wins.
        assert!(pairs(0.2) < 2);
        assert!(pairs(3.0) >= 2);
    }

    #[test]
    fn a_stem_judges_where_it_plays_and_does_not_win() {
        let truth = grid(120.0, 0.0, 64); // 32 s
        let wrong = grid(97.0, 0.13, 52);
        // Two full candidates disagree: a tie, so the earlier one wins.
        let tie = [
            candidate("wrong", wrong.clone()),
            candidate("right", truth.clone()),
        ];
        assert_eq!(vote(&tie).unwrap().chosen, 0);
        // A stem that plays from 8 s on, on the right grid, breaks the tie
        // and does not win itself.
        let mut with_stem = tie.to_vec();
        with_stem.push(partial("drums", &truth, 8.0, 32.0, 32.0));
        let outcome = vote(&with_stem).unwrap();
        assert_eq!(outcome.chosen, 1);
        // A stem that never plays changes nothing.
        let mut silent = tie.to_vec();
        silent.push(partial("drums", &truth, 40.0, 50.0, 32.0));
        let outcome = vote(&silent).unwrap();
        assert_eq!(outcome.chosen, 0);
        assert_eq!(outcome.agreements[..2], vote(&tie).unwrap().agreements[..]);
    }

    #[test]
    fn a_stem_counts_by_how_much_it_plays() {
        let truth = grid(120.0, 0.0, 64);
        let wrong = grid(97.0, 0.13, 52);
        // A full candidate on the wrong grid against a stem on the right one
        // that plays briefly: the full candidate's support grows with the
        // stem's coverage.
        let support = |to: f64| {
            let c = [
                candidate("right", truth.clone()),
                candidate("wrong", wrong.clone()),
                partial("drums", &truth, 0.0, to, 32.0),
            ];
            vote(&c).unwrap().agreements[0]
        };
        assert!(support(32.0) > support(8.0));
    }

    #[test]
    fn close_spans_keep_each_beat_once() {
        let beats = [9.9, 10.0, 10.02, 10.1, 20.0];
        let spans = [(9.0, 10.0), (10.05, 11.0)];
        assert_eq!(within(&beats, &spans), vec![9.9, 10.0, 10.02, 10.1]);
    }

    #[test]
    fn a_region_the_drums_enter_is_judged_where_they_play() {
        let truth = grid(120.0, 0.0, 64);
        let candidates = [
            candidate("a", truth.clone()),
            // Drums from 4 s on: the first region's second half.
            partial("drums", &truth, 4.0, 32.0, 32.0),
        ];
        let regions = regions(&truth, &candidates, 0);
        assert!(regions[0].confidence > 0.99, "{:?}", regions[0]);
    }
}
