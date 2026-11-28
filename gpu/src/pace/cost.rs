//! What a dispatch costs ([`Cost`]: its estimated GPU time, made with it
//! by its plan) and how a submission is sized from the costs ([`Sizing`]):
//! about a piece ([`PIECE`]) of GPU time, each kind of estimate
//! ([`Bound`]) corrected on its own by the times the submissions took.

#[cfg(test)]
use std::ops::Range;
use std::time::Duration;

use super::PIECE;

/// A piece's time, in seconds.
const PIECE_SECS: f64 = PIECE.as_secs_f64();

// The GPU time a dispatch takes however little it does (starting it, the
// barrier after it), and the bytes a second the GPUs' kernels that move
// values reach, in seconds and bytes a second: estimates, not measured,
// each a little under the chip's own (the 5080's 960 GB/s; the M4 Max's
// 532 GB/s read and 464 written). The pace corrects them as it goes.
pub(crate) const LAUNCH: f64 = 10e-6;
pub(crate) const RTX_5080_BANDWIDTH: f64 = 800e9;
pub(crate) const M4_MAX_BANDWIDTH: f64 = 400e9;

/// What a dispatch's estimate is reckoned from, each kind corrected on its
/// own: one kind's estimates may be off another way than another's (the
/// rates of the multiplies are measured, the bandwidths and the start of
/// a dispatch guessed), and a correction learned on one would size the
/// other wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Bound {
    /// Its arithmetic: a multiply's or an attention's FLOPs at its rate.
    Work,
    /// Its values moved and its start: a norm's, a copy's, a one-row
    /// multiply's weights (and a dispatch no plan estimates).
    Traffic,
}

impl Bound {
    const ALL: [Bound; 2] = [Bound::Work, Bound::Traffic];

    fn at(self) -> usize {
        self as usize
    }
}

/// A dispatch's estimated GPU time, in seconds, and what it is reckoned
/// from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Cost {
    pub(crate) seconds: f64,
    pub(crate) bound: Bound,
}

impl Cost {
    /// A dispatch of a network that makes no estimate: a piece, the most a
    /// dispatch should take, so its submissions are counted in dispatches
    /// as they were before estimates (one, one, two, four, then towards a
    /// piece of GPU time).
    pub(crate) const UNESTIMATED: Cost = Cost {
        seconds: PIECE_SECS,
        bound: Bound::Traffic,
    };

    /// `flops` at `rate` FLOP/s, and [`LAUNCH`].
    pub(crate) fn work(flops: f64, rate: f64) -> Cost {
        Cost {
            seconds: LAUNCH + flops / rate,
            bound: Bound::Work,
        }
    }

    /// `bytes` at `bandwidth` bytes a second, and [`LAUNCH`].
    pub(crate) fn traffic(bytes: f64, bandwidth: f64) -> Cost {
        Cost {
            seconds: LAUNCH + bytes / bandwidth,
            bound: Bound::Traffic,
        }
    }
}

/// The estimated seconds of a submission's dispatches, and how many, of
/// each [`Bound`] (or a run's).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Spent {
    seconds: [f64; 2],
    dispatches: [usize; 2],
}

impl Spent {
    /// `costs`' together.
    pub(crate) fn of(costs: &[Cost]) -> Spent {
        let mut spent = Spent::default();
        for c in costs {
            spent.seconds[c.bound.at()] += c.seconds;
            spent.dispatches[c.bound.at()] += 1;
        }
        spent
    }

    /// This and `other` together (a submission and the one before it).
    pub(crate) fn and(self, other: Spent) -> Spent {
        Spent {
            seconds: [0, 1].map(|b| self.seconds[b] + other.seconds[b]),
            dispatches: [0, 1].map(|b| self.dispatches[b] + other.dispatches[b]),
        }
    }
}

/// Halfway from `now` towards `ideal`: how a size is learned, smoothly.
pub(super) fn toward(now: f64, ideal: f64) -> f64 {
    now * 0.5 + ideal * 0.5
}

/// Slack for the sums of estimates against a size (their rounding).
const SLACK: f64 = 1e-9;

/// How large a submission is, learned per run: for each [`Bound`], the
/// estimated seconds of its dispatches a piece of GPU time takes (a
/// piece's time over the GPU time they took over their estimate), none
/// until one of its submissions is learned from. A submission takes as
/// many dispatches as fit a piece by those, one whole step at least.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Sizing {
    sizes: [Option<f64>; 2],
}

impl Sizing {
    /// Where the submission from dispatch `start` of a run whose dispatches
    /// cost `costs` ends, and what it holds: whole steps of `whole`
    /// dispatches (`Recorded::whole`), as many as fit a piece by the sizes
    /// (one step at least; one alone where a [`Bound`] of it has none yet:
    /// a run's first submission is one dispatch). Whole steps of more than
    /// one (the recogniser's decoder steps, never estimated) are taken
    /// while the submission is short of the piece, as many as it was
    /// counted in, rounded up, as before.
    pub(crate) fn end(&self, start: usize, whole: usize, costs: &[Cost]) -> (usize, Spent) {
        let len = costs.len();
        let (mut end, mut spent, mut filled) = (start, Spent::default(), 0.0);
        while end < len {
            let next = (end + whole).min(len);
            let step = Spent::of(&costs[end..next]);
            // The share of a piece the step takes; unsized, none fits beside it.
            let share: f64 = Bound::ALL
                .iter()
                .filter(|b| step.dispatches[b.at()] > 0)
                .map(|b| match self.sizes[b.at()] {
                    Some(size) => step.seconds[b.at()] / size,
                    None => f64::INFINITY,
                })
                .sum();
            if end > start {
                let fits = filled + share <= 1.0 + SLACK;
                let short = whole > 1 && filled < 1.0 - SLACK;
                if !(fits || short) {
                    break;
                }
            }
            (end, spent, filled) = (next, spent.and(step), filled + share);
        }
        (end, spent)
    }

    /// A submission that held `spent` (of a run that holds `all`) took
    /// `took`: the size of the [`Bound`] that took the most of it learned
    /// ([`learned`]), its part of the time as the sizes reckon it.
    pub(crate) fn learn(&mut self, spent: Spent, all: Spent, took: Duration) {
        // Each bound's part of the time as the sizes reckon it (one not
        // sized yet as its estimates say).
        let reckoned = Bound::ALL.map(|b| {
            let size = self.sizes[b.at()].unwrap_or(PIECE_SECS);
            spent.seconds[b.at()] / size
        });
        let whole: f64 = reckoned.iter().sum();
        let Some(most) = (Bound::ALL.into_iter())
            .filter(|b| spent.dispatches[b.at()] > 0)
            .max_by(|a, b| reckoned[a.at()].total_cmp(&reckoned[b.at()]))
        else {
            return;
        };
        let b = most.at();
        let part = if reckoned[b] >= whole {
            took
        } else {
            took.mul_f64(reckoned[b] / whole)
        };
        let estimated = (spent.seconds[b], spent.dispatches[b]);
        let run = all.seconds[b].max(spent.seconds[b]);
        self.sizes[b] = Some(learned(self.sizes[b], estimated, run, part));
    }
}

/// The size learned from `size` (none: one of the submission's dispatches,
/// as a run's first submission is) where `n` dispatches estimated at `seconds` (of a
/// run of `run`) took `took`: halfway towards what would take a piece
/// ([`toward`]), in whole dispatches of theirs, at most twice the more of
/// the size and theirs, one of them at least and the run at most. Where
/// every dispatch is estimated alike (a network that makes no estimate:
/// [`Cost::UNESTIMATED`]) this is the count of dispatches as it was learned
/// before estimates, the same sizes.
///
/// The cap bounds how fast the estimates of a submission grow, not its
/// time: a submission after one whose estimates were too high, of
/// dispatches whose estimates are too low, takes twice a piece times how
/// far their errors are apart. Their bounds are corrected apart for that
/// ([`Bound`]), and the plans' estimates kept honest; a dispatch that runs
/// far slower than its estimate (a model's first use) adds its excess to
/// its submission, on top of up to two pieces.
fn learned(size: Option<f64>, (seconds, n): (f64, usize), run: f64, took: Duration) -> f64 {
    let one = seconds / n as f64;
    let size = size.unwrap_or(one);
    let ideal = seconds * PIECE_SECS / took.as_secs_f64().max(1e-5);
    let next = (toward(size, ideal) / one).round() * one;
    next.min(2.0 * size.max(seconds)).clamp(one, run.max(one))
}

/// For checks: the submissions a new background pace makes of a run whose
/// dispatches cost `costs` (sized for each, learned from each one's time
/// once the next is made, as a run does), each taking `took` (each
/// dispatch's GPU time, in seconds, by its index) and 150 µs more (its
/// submission's own): each one's time, in seconds, and its dispatches.
#[cfg(test)]
pub(crate) fn submissions(costs: &[Cost], took: impl Fn(usize) -> f64) -> Vec<(f64, Range<usize>)> {
    let mut sizing = Sizing::default();
    let all = Spent::of(costs);
    let (mut at, mut made, mut running) = (0, Vec::new(), None);
    while at < costs.len() {
        let (end, spent) = sizing.end(at, 1, costs);
        let time = 150e-6 + (at..end).map(&took).sum::<f64>();
        made.push((time, at..end));
        // As the run does: the one before is settled once this one is
        // made.
        if let Some((s, t)) = running.replace((spent, time)) {
            sizing.learn(s, all, Duration::from_secs_f64(t));
        }
        at = end;
    }
    made
}

/// For checks: a run whose dispatches cost `costs` (of `what`) paced
/// ([`submissions`]) with each taking its estimate, and again with them
/// off in unlike ways (the multiplies' as estimated; the start of a
/// dispatch twice its estimate, its values moved three times theirs): no
/// submission takes more than two pieces, or than its one dispatch where
/// that is longer. The first submissions' times and the longest are
/// printed (`--nocapture`).
#[cfg(test)]
pub(crate) fn within_two_pieces(what: &str, costs: &[Cost]) {
    let exact = |i: usize| costs[i].seconds;
    let skewed = |i: usize| skew(costs[i]);
    for (how, took) in [
        ("as estimated", &exact as &dyn Fn(usize) -> f64),
        ("skewed", &skewed),
    ] {
        let made = submissions(costs, took);
        let ms: Vec<String> = (made.iter().take(8))
            .map(|(t, _)| format!("{:.1}", t * 1e3))
            .collect();
        let most = made.iter().map(|m| m.0).fold(0.0, f64::max);
        eprintln!(
            "{what}, {how}: {} submissions, the first {} ms, the longest {:.1} ms",
            made.len(),
            ms.join(" "),
            most * 1e3
        );
        for (i, (t, ops)) in made.iter().enumerate() {
            let own = ops.clone().map(took).fold(0.0, f64::max) + 150e-6;
            let bound = if ops.len() == 1 {
                own
            } else {
                2.0 * PIECE_SECS
            };
            assert!(
                *t <= bound + 1e-12,
                "{what}, {how}: submission {i} ({} dispatches) {:.2} ms",
                ops.len(),
                t * 1e3
            );
        }
    }
}

/// For checks: `cost`'s time skewed as [`within_two_pieces`] says.
#[cfg(test)]
fn skew(cost: Cost) -> f64 {
    match cost.bound {
        Bound::Work => cost.seconds,
        Bound::Traffic => 2.0 * LAUNCH + 3.0 * (cost.seconds - LAUNCH).max(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The size as it was learned before estimates, in dispatches: towards
    /// a piece, at most twice the more of `per` and `n`, within the run.
    fn counted(per: usize, n: usize, all: usize, took: Duration) -> usize {
        let ideal = n as f64 * PIECE_SECS / took.as_secs_f64().max(1e-5);
        let next = toward(per as f64, ideal).round() as usize;
        next.min(2 * per.max(n)).clamp(1, all.max(1))
    }

    /// A network that makes no estimate is sized as it was before
    /// estimates, submission for submission, in single dispatches and in
    /// whole steps of several, however long its submissions take: one, one
    /// (made before the first is learned from), two, four, then towards a
    /// piece, the count rounded to the nearest.
    #[test]
    fn a_network_without_estimates_is_sized_as_before() {
        let mut noise = 12345u64;
        let mut next = || {
            noise = noise
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (noise >> 33) as f64 / (1u64 << 31) as f64
        };
        for whole in [1, 3, 5] {
            for (len, us) in [(400, 300.0), (37, 2000.0), (1000, 40.0)] {
                let costs = vec![Cost::UNESTIMATED; len];
                let all = Spent::of(&costs);
                let (mut sizing, mut per) = (Sizing::default(), 1usize);
                let (mut at, mut running) = (0, None);
                let mut first = Vec::new();
                while at < len {
                    let (end, spent) = sizing.end(at, whole, &costs);
                    let want = (at + per.next_multiple_of(whole)).min(len);
                    assert_eq!(end, want, "whole {whole}, {len} of {us} µs, from {at}");
                    first.push(end - at);
                    let took =
                        Duration::from_secs_f64((end - at) as f64 * us * 1e-6 * (0.5 + next()));
                    if let Some((s, n, t)) = running.replace((spent, end - at, took)) {
                        sizing.learn(s, all, t);
                        per = counted(per, n, len, t);
                    }
                    at = end;
                }
                if whole == 1 && len == 400 {
                    assert_eq!(first[..4], [1, 1, 2, 4], "{first:?}");
                }
            }
        }
    }

    /// The size goes halfway towards a piece of GPU time, both ways, in
    /// whole dispatches of the submission's, at most twice the last; none
    /// learned is one of them.
    #[test]
    fn the_size_goes_halfway_towards_a_piece() {
        let ms = Duration::from_millis;
        let piece = PIECE_SECS;
        let near = |got: f64, want: f64| assert!((got - want).abs() < 1e-12, "{got} {want}");
        // A piece's worth that took twice that: halfway to half a piece,
        // to the nearest of its dispatches.
        near(learned(Some(piece), (piece, 10), 1.0, ms(10)), 0.8 * piece);
        // That took a fifth: towards five, at most twice.
        near(learned(Some(piece), (piece, 10), 1.0, ms(1)), 2.0 * piece);
        // In whole dispatches of the submission's (a tenth of a piece each).
        near(learned(Some(piece), (piece, 10), 1.0, ms(7)), 0.9 * piece);
        // None learned: one of its dispatches.
        near(learned(None, (piece / 4.0, 1), 1.0, ms(10)), 0.25 * piece);
        // Never past the run; halfway down, never under one dispatch.
        near(
            learned(Some(piece), (piece, 10), 1.2 * piece, ms(1)),
            1.2 * piece,
        );
        near(
            learned(Some(piece), (piece, 10), 1.0, Duration::from_secs(10)),
            0.5 * piece,
        );
        near(
            learned(Some(0.1 * piece), (piece, 10), 1.0, Duration::from_secs(10)),
            0.1 * piece,
        );
    }

    /// A step of a bound not sized yet goes alone; sized, the sizes are
    /// filled by each bound's share of a piece.
    #[test]
    fn a_submission_fills_a_piece_by_each_bounds_size() {
        let work = Cost::work(1e9, 1e12); // 1.01 ms
        let traffic = Cost::traffic(0.0, 1.0); // 10 µs
        let costs = [work, work, traffic, traffic, work, traffic];
        let mut sizing = Sizing::default();
        assert_eq!(sizing.end(0, 1, &costs).0, 1);
        sizing.sizes[Bound::Work.at()] = Some(2.0 * work.seconds);
        // Two of the work fill it; the traffic is not sized yet.
        assert_eq!(sizing.end(0, 1, &costs).0, 2);
        assert_eq!(sizing.end(2, 1, &costs).0, 3);
        sizing.sizes[Bound::Traffic.at()] = Some(10.0 * traffic.seconds);
        // One work (half) and two of the traffic (a fifth) fit.
        assert_eq!(sizing.end(1, 1, &costs).0, 4);
        let (end, spent) = sizing.end(0, 1, &costs);
        assert_eq!((end, spent.dispatches), (2, [2, 0]));
    }

    /// Estimates off in unlike ways, as each pace learns them: the
    /// multiplies' as estimated, the norms' and copies' three times
    /// theirs, the start of a dispatch twice its estimate. Each bound
    /// corrected on its own, no submission of more than one dispatch takes
    /// more than two pieces past the first two (made before either is
    /// learned from), and one of one takes its own. With one multiply ten
    /// times its own besides (a first use), wherever it is: its submission
    /// takes its excess on top of that bound (nothing sizes it before its
    /// time is known), and every other submission keeps to it.
    #[test]
    fn skewed_estimates_keep_a_submission_within_two_pieces() {
        // A run of a picture's kind: slices of multiplies between runs of
        // norms, a few long ones of each.
        let mut costs = Vec::new();
        for i in 0..600 {
            costs.push(Cost::work(
                match i % 7 {
                    0 => 4.5e9,
                    1 => 0.3e9,
                    _ => 2e9,
                },
                1e12,
            ));
            for _ in 0..(i % 5) * 3 {
                costs.push(Cost::traffic(((i % 11) as f64 + 1.0) * 2e6, 400e9));
            }
        }
        let piece = PIECE_SECS;
        let multiplies = (0..costs.len()).filter(|&i| costs[i].bound == Bound::Work);
        for slow in [None].into_iter().chain(multiplies.map(Some)) {
            let took = |i: usize| {
                let t = skew(costs[i]);
                if Some(i) == slow {
                    10.0 * t
                } else {
                    t
                }
            };
            let made = submissions(&costs, took);
            for (i, (t, ops)) in made.iter().enumerate().skip(2) {
                let own = ops.clone().map(took).fold(0.0, f64::max);
                let excess = slow
                    .filter(|s| ops.contains(s))
                    .map_or(0.0, |s| took(s) - skew(costs[s]));
                let bound = if ops.len() == 1 {
                    own + 150e-6
                } else {
                    2.0 * piece + excess
                };
                assert!(
                    *t <= bound + 1e-12,
                    "slow {slow:?}: submission {i} ({} dispatches): {:.2} ms",
                    ops.len(),
                    t * 1e3
                );
            }
        }
    }
}
