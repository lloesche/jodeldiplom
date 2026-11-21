//! The meter: which grouping of the pulse is the beat, and the bars.
//!
//! The vote gives a grid the candidates agree on, but not which level of
//! the music's pulse a listener counts: trackers and the network alike
//! follow the steadiest pulse, often the eighth notes. So the meter puts
//! a pulse between each two beats of that grid and considers beats of 1,
//! 2, 4, 6 or 8 pulses ([`Grouping`]: ×2, ×1, ×½, ×⅓ or ×¼ the grid), each
//! at every phase. Around each beat of the grid, every grouping is scored
//! over the pulses nearby from what a listener hears ([`Score`]):
//!
//! - **backbeat**: the snare on every second beat: how much more snare
//!   one set of alternate beats has than the other. It is strongest at
//!   the beat a listener counts; counted twice as fast, the snare comes
//!   only every fourth beat, so the difference halves, and counted half
//!   as fast, it is on every beat or on none. (The kick is left out: in
//!   hip-hop it often falls between the beats);
//! - **accent**: kick, snare and bass at the beats rather than at the
//!   pulses between them, which rules out beats too slow to have all of
//!   them;
//! - **change**: chords change on beats, not between them;
//! - **candidates**: the share (by weight, and a stem's candidate by its
//!   say there, over the beats where it votes) of the candidates that
//!   track this grouping's beats;
//! - **preference**: a mild one for tempi near 110 BPM, for near ties.
//!
//! With stems, the cues already count only where their stem is present
//! ([`Cues::new`]).
//!
//! A Viterbi path through the groupings, paying for each change, picks
//! one per beat of the grid, so the level changes only where the music
//! does (a half-time bridge). The bars then come from [`downbeats`].

mod bars;

pub use bars::downbeats;

use crate::activation::Curve;
use crate::cues::Cues;
use crate::metrics::{between, f_measure};
use crate::vote::{within, Candidate, TOLERANCE};

/// Pulses on each side of a beat whose cues score its grouping: 24 beats
/// of the grid.
const RADIUS: usize = 48;
/// The longest interval of the grid, in its usual intervals, whose
/// pulses may still be beats.
const FILL: usize = 4;

/// How much each part of a [`Score`] counts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Weights {
    /// The backbeat's weight.
    pub backbeat: f64,
    /// The accents' weight.
    pub accent: f64,
    /// The chord changes' weight.
    pub change: f64,
    /// The candidates' weight.
    pub candidates: f64,
    /// The tempo preference's weight.
    pub preference: f64,
    /// The tempo the preference is centred on, in BPM.
    pub center_bpm: f64,
    /// The price of changing the grouping from one beat to the next.
    pub switch: f64,
}

impl Weights {
    /// Fitted with the cues from the fast split of the mix.
    pub const MIX: Weights = Weights {
        backbeat: 1.0,
        accent: 3.0,
        change: 0.0,
        candidates: 1.0,
        preference: 0.6,
        center_bpm: 110.0,
        switch: 32.0,
    };

    /// Fitted with the cues from stems: their chord changes are clean
    /// enough to count for most.
    pub const STEMS: Weights = Weights {
        backbeat: 1.0,
        accent: 0.5,
        change: 4.0,
        candidates: 0.5,
        preference: 1.0,
        center_bpm: 120.0,
        switch: 32.0,
    };

    /// The weights fitted for cues from stems (`true`) or from the mix.
    pub fn for_parts(stems: bool) -> Weights {
        if stems {
            Weights::STEMS
        } else {
            Weights::MIX
        }
    }
}

impl Default for Weights {
    /// The weights for the mix's cues.
    fn default() -> Weights {
        Weights::MIX
    }
}

/// A beat as a grouping of the pulse (the vote's grid with a pulse
/// between each two of its beats): every `pulses` pulses, from pulse
/// `phase`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Grouping {
    /// Pulses per beat: 1, 2, 4, 6 or 8.
    pub pulses: usize,
    /// The first beat's pulse, below `pulses`.
    pub phase: usize,
}

impl Grouping {
    /// The groupings' sizes.
    pub const SIZES: [usize; 5] = [1, 2, 4, 6, 8];

    /// The vote's grid as it is.
    pub const GRID: Grouping = Grouping {
        pulses: 2,
        phase: 0,
    };

    /// The grouping of `pulses` from `phase`, if `pulses` is one of
    /// [`Grouping::SIZES`] and `phase` is below it.
    pub fn new(pulses: usize, phase: usize) -> Option<Grouping> {
        (Self::SIZES.contains(&pulses) && phase < pulses).then_some(Grouping { pulses, phase })
    }

    /// Every grouping at every phase.
    pub fn all() -> Vec<Grouping> {
        Self::SIZES
            .iter()
            .flat_map(|&pulses| (0..pulses).map(move |phase| Grouping { pulses, phase }))
            .collect()
    }

    /// The beat's tempo relative to the grid's: 2, 1, ½, ⅓ or ¼.
    pub fn ratio(self) -> f64 {
        2.0 / self.pulses as f64
    }

    /// The ratio as people read it: `×2`, `×1`, `×½`, `×⅓` or `×¼`.
    pub fn label(self) -> &'static str {
        match self.pulses {
            1 => "×2",
            2 => "×1",
            4 => "×½",
            6 => "×⅓",
            8 => "×¼",
            _ => "×?",
        }
    }

    /// Whether `pulse` is a beat.
    pub fn is_beat(self, pulse: usize) -> bool {
        pulse % self.pulses == self.phase
    }
}

/// A stretch of the beats with one grouping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Span {
    /// Its first beat, in seconds.
    pub start: f64,
    /// Its grouping.
    pub grouping: Grouping,
}

/// How well a grouping fits the music around a beat, part by part, each
/// about 0 to 1.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Score {
    /// The snare's difference between alternate beats.
    pub backbeat: f64,
    /// Kick, snare and bass at the beats, less between them.
    pub accent: f64,
    /// Chord changes at the beats, less between them.
    pub change: f64,
    /// The share (by weight) of the candidates that track these beats.
    pub candidates: f64,
    /// The tempo preference.
    pub preference: f64,
}

impl Score {
    /// The weighted sum.
    pub fn total(&self, w: &Weights) -> f64 {
        w.backbeat * self.backbeat
            + w.accent * self.accent
            + w.change * self.change
            + w.candidates * self.candidates
            + w.preference * self.preference
    }
}

/// The pulse and the cues at each pulse.
///
/// Each interval of the grid holds two pulses (its beat and the midpoint).
/// A longer interval holds as many as fit at the grid's usual spacing, so
/// the pulses keep counting through it. Up to [`FILL`] usual intervals
/// (a tracker that skipped beats, or followed half the tempo for a while)
/// those pulses may be beats; a longer gap is a stretch without a beat,
/// and they only keep the count.
struct Pulses {
    times: Vec<f64>,
    /// Whether the pulse may be a beat (not one counting through a gap).
    real: Vec<bool>,
    /// Each grid beat's own pulse; its interval runs to the next one's.
    first: Vec<usize>,
    /// Snare and claps.
    snare: Vec<f64>,
    /// Kick, snare and bass.
    accent: Vec<f64>,
    change: Vec<f64>,
}

/// The median interval of `beats`.
fn median_interval(beats: &[f64]) -> f64 {
    let mut gaps: Vec<f64> = beats.windows(2).map(|w| w[1] - w[0]).collect();
    gaps.sort_by(f64::total_cmp);
    gaps.get(gaps.len() / 2).copied().unwrap_or(0.5)
}

impl Pulses {
    fn new(grid: &[f64], cues: &Cues) -> Pulses {
        let usual = median_interval(grid);
        let (mut times, mut real, mut first) = (Vec::new(), Vec::new(), Vec::new());
        for (i, &b) in grid.iter().enumerate() {
            first.push(times.len());
            let Some(&next) = grid.get(i + 1) else {
                times.push(b);
                real.push(true);
                break;
            };
            let steps = ((next - b) / usual).round().max(1.0) as usize;
            let count = 2 * steps;
            for k in 0..count {
                times.push(b + (next - b) * k as f64 / count as f64);
                real.push(steps <= FILL || k == 0);
            }
        }
        // Within a third of the pulse, at most 35 ms.
        let near = (usual / 6.0).min(0.035);
        let at = |curve: &Curve| -> Vec<f64> {
            times
                .iter()
                .map(|&t| f64::from(curve.max_near(t, near)))
                .collect()
        };
        let (kick, snare, bass) = (at(&cues.kick), at(&cues.snare), at(&cues.bass));
        Pulses {
            accent: (0..times.len())
                .map(|i| (kick[i] + snare[i] + bass[i]) / 3.0)
                .collect(),
            change: at(&cues.change),
            snare,
            times,
            real,
            first,
        }
    }

    /// The pulses around grid beat `i`.
    fn window(&self, i: usize) -> (usize, usize) {
        let mid = self.first[i];
        (
            mid.saturating_sub(RADIUS),
            (mid + RADIUS + 1).min(self.times.len()),
        )
    }
}

/// The mean of `values`, 0 when empty.
fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, count) = values.fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

/// The preference for tempi near `center`, 1 there and falling with the
/// distance in octaves.
fn preference(bpm: f64, center: f64) -> f64 {
    let octaves = (bpm / center).log2();
    (-0.5 * octaves * octaves).exp()
}

/// `grouping`'s score over pulses `lo..hi`.
fn score(
    grouping: Grouping,
    lo: usize,
    hi: usize,
    pulses: &Pulses,
    candidates: &[Candidate],
    spans: &[Option<Vec<(f64, f64)>>],
    weights: &Weights,
) -> Score {
    let window = lo..hi;
    let beats: Vec<usize> = window.clone().filter(|&j| grouping.is_beat(j)).collect();
    if beats.len() < 4 {
        return Score::default();
    }
    let others = || window.clone().filter(|&j| !grouping.is_beat(j));
    let parity = |j: usize| (j / grouping.pulses) % 2;
    let side = |p: usize| {
        mean(
            beats
                .iter()
                .filter(|&&j| parity(j) == p)
                .map(|&j| pulses.snare[j]),
        )
    };
    let contrast = |cue: &[f64]| {
        if grouping.pulses == 1 {
            return 0.0;
        }
        mean(beats.iter().map(|&j| cue[j])) - mean(others().map(|j| cue[j]))
    };
    let times: Vec<f64> = beats.iter().map(|&j| pulses.times[j]).collect();
    let (start, end) = (times[0] - TOLERANCE, times[times.len() - 1] + TOLERANCE);
    let (mut support, mut total) = (0.0, 0.0);
    for (c, spans) in candidates.iter().zip(spans) {
        let w = c.weight * c.say_between(start, end);
        if w <= 0.0 {
            continue;
        }
        let theirs = between(&c.beats, start, end);
        // A stem's candidate against these beats where it votes.
        let f = match spans {
            Some(spans) => f_measure(theirs, &within(&times, spans), TOLERANCE),
            None => f_measure(theirs, &times, TOLERANCE),
        };
        support += w * ((f - 0.6) / 0.4).clamp(0.0, 1.0);
        total += w;
    }
    let seconds = (times[times.len() - 1] - times[0]) / (times.len() - 1) as f64;
    Score {
        backbeat: (side(0) - side(1)).abs(),
        accent: contrast(&pulses.accent),
        change: contrast(&pulses.change),
        candidates: if total > 0.0 { support / total } else { 0.0 },
        preference: preference(60.0 / seconds, weights.center_bpm),
    }
}

/// Every grouping's score around each grid beat.
fn scores(pulses: &Pulses, candidates: &[Candidate], weights: &Weights) -> Vec<Vec<Score>> {
    let states = Grouping::all();
    let spans: Vec<Option<Vec<(f64, f64)>>> = candidates.iter().map(Candidate::spans).collect();
    (0..pulses.first.len())
        .map(|i| {
            let (lo, hi) = pulses.window(i);
            states
                .iter()
                .map(|&g| score(g, lo, hi, pulses, candidates, &spans, weights))
                .collect()
        })
        .collect()
}

/// The beats: the groupings of `grid` (the vote's winner) a listener
/// would count, from `cues` and `candidates`, and the spans of each.
/// Fewer than four grid beats are kept as they are.
pub fn beats(
    grid: &[f64],
    candidates: &[Candidate],
    cues: &Cues,
    weights: &Weights,
) -> (Vec<f64>, Vec<Span>) {
    if grid.len() < 4 {
        let spans = grid
            .first()
            .map(|&start| Span {
                start,
                grouping: Grouping::GRID,
            })
            .into_iter()
            .collect();
        return (grid.to_vec(), spans);
    }
    let pulses = Pulses::new(grid, cues);
    let totals: Vec<Vec<f64>> = scores(&pulses, candidates, weights)
        .iter()
        .map(|row| row.iter().map(|s| s.total(weights)).collect())
        .collect();
    let path = choose(&pulses, &totals, weights.switch);
    emit(&pulses, &path)
}

/// The pulses of grid beat `i`'s interval.
fn interval(pulses: &Pulses, i: usize) -> std::ops::Range<usize> {
    let end = pulses
        .first
        .get(i + 1)
        .copied()
        .unwrap_or(pulses.times.len());
    pulses.first[i]..end
}

/// The first pulse of grid beat `i`'s interval that is a beat of both
/// `a` and `b`: where the grouping may change from one to the other, so
/// the beats keep their spacing.
fn shared(pulses: &Pulses, i: usize, a: Grouping, b: Grouping) -> Option<usize> {
    interval(pulses, i).find(|&j| a.is_beat(j) && b.is_beat(j))
}

/// The grouping at each grid beat with the best total of `totals`
/// (each grid beat's total per grouping, in [`Grouping::all`]'s order),
/// paying `switch` for each change. A change at grid beat `i` needs a
/// pulse in its interval that both groupings have as a beat.
fn choose(pulses: &Pulses, totals: &[Vec<f64>], switch: f64) -> Vec<Grouping> {
    let states = Grouping::all();
    let (n, k) = (totals.len(), states.len());
    let mut total = totals[0].clone();
    let mut back = vec![vec![0usize; k]; n];
    for i in 1..n {
        let mut next = vec![0.0; k];
        for to in 0..k {
            let (mut from, mut best) = (to, total[to]);
            for (s, &v) in total.iter().enumerate() {
                if s != to
                    && v - switch > best
                    && shared(pulses, i, states[s], states[to]).is_some()
                {
                    (from, best) = (s, v - switch);
                }
            }
            back[i][to] = from;
            next[to] = best + totals[i][to];
        }
        total = next;
    }
    let mut s = (0..k).fold(0, |b, s| if total[s] > total[b] { s } else { b });
    let mut path = vec![Grouping::GRID; n];
    for i in (0..n).rev() {
        path[i] = states[s];
        s = back[i][s];
    }
    path
}

/// The beats of `path` and its spans. Where the grouping changes at grid
/// beat `i`, the pulses of its interval before the shared beat keep the
/// old grouping.
fn emit(pulses: &Pulses, path: &[Grouping]) -> (Vec<f64>, Vec<Span>) {
    let mut out = Vec::new();
    let mut spans: Vec<Span> = Vec::new();
    for (i, &grouping) in path.iter().enumerate() {
        let before = if i > 0 { path[i - 1] } else { grouping };
        let cut = if before == grouping {
            pulses.first[i]
        } else {
            shared(pulses, i, before, grouping).unwrap_or(pulses.first[i])
        };
        for j in interval(pulses, i) {
            let g = if j < cut { before } else { grouping };
            if pulses.real[j] && g.is_beat(j) {
                let t = pulses.times[j];
                if spans.last().is_none_or(|s| s.grouping != g) {
                    spans.push(Span {
                        start: t,
                        grouping: g,
                    });
                }
                out.push(t);
            }
        }
    }
    (out, spans)
}

/// Each grouping's score averaged over the track, for reports: what each
/// cue said about each level.
pub fn explain(
    grid: &[f64],
    candidates: &[Candidate],
    cues: &Cues,
    weights: &Weights,
) -> Vec<(Grouping, Score)> {
    if grid.len() < 4 {
        return Vec::new();
    }
    let rows = scores(&Pulses::new(grid, cues), candidates, weights);
    Grouping::all()
        .into_iter()
        .enumerate()
        .map(|(s, g)| {
            let avg = |f: fn(&Score) -> f64| mean(rows.iter().map(|row| f(&row[s])));
            let score = Score {
                backbeat: avg(|s| s.backbeat),
                accent: avg(|s| s.accent),
                change: avg(|s| s.change),
                candidates: avg(|s| s.candidates),
                preference: avg(|s| s.preference),
            };
            (g, score)
        })
        .collect()
}

/// The spans' groupings in words: one label when the grouping is the
/// same throughout, else each with its share of the time, the largest
/// first; empty for the grid as it is.
pub fn describe(spans: &[Span], end: f64) -> String {
    let mut shares: Vec<(&'static str, f64)> = Vec::new();
    for (i, s) in spans.iter().enumerate() {
        let until = spans.get(i + 1).map_or(end, |n| n.start);
        let label = s.grouping.label();
        match shares.iter_mut().find(|(l, _)| *l == label) {
            Some((_, t)) => *t += until - s.start,
            None => shares.push((label, until - s.start)),
        }
    }
    shares.sort_by(|a, b| b.1.total_cmp(&a.1));
    let whole: f64 = shares.iter().map(|s| s.1).sum();
    match shares.as_slice() {
        [] => String::new(),
        [(label, _)] if *label == "×1" => String::new(),
        [(label, _)] => (*label).to_owned(),
        _ => shares
            .iter()
            .map(|(l, t)| format!("{l} {:.0} %", 100.0 * t / whole.max(1e-9)))
            .collect::<Vec<_>>()
            .join(", "),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(n: usize, interval: f64) -> Vec<f64> {
        (0..n).map(|i| 1.0 + i as f64 * interval).collect()
    }

    /// Totals that want `want(i)` at grid beat `i` and nothing else.
    fn wanting(n: usize, want: impl Fn(usize) -> Grouping) -> Vec<Vec<f64>> {
        let states = Grouping::all();
        (0..n)
            .map(|i| {
                states
                    .iter()
                    .map(|&s| if s == want(i) { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn groupings_are_checked_and_named() {
        assert_eq!(
            Grouping::new(6, 5),
            Some(Grouping {
                pulses: 6,
                phase: 5
            })
        );
        assert_eq!(Grouping::new(6, 6), None);
        assert_eq!(Grouping::new(3, 0), None);
        assert_eq!(Grouping::all().len(), 21);
        let third = Grouping::new(6, 0).unwrap();
        assert_eq!(third.label(), "×⅓");
        assert!((third.ratio() - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn a_change_of_grouping_keeps_the_beats_spaced() {
        let g = grid(64, 0.5);
        let pulses = Pulses::new(&g, &Cues::default());
        let third = Grouping::new(6, 0).unwrap();
        let totals = wanting(g.len(), |i| if i < 32 { Grouping::GRID } else { third });
        let path = choose(&pulses, &totals, 2.0);
        assert_eq!((path[10], path[50]), (Grouping::GRID, third));
        let (beats, spans) = emit(&pulses, &path);
        assert_eq!(spans.len(), 2);
        // Every interval is one grid beat or three, none in between.
        for w in beats.windows(2) {
            let d = w[1] - w[0];
            assert!((d - 0.5).abs() < 1e-9 || (d - 1.5).abs() < 1e-9, "{d}");
        }
    }

    #[test]
    fn an_off_beat_phase_can_be_reached_mid_track() {
        let g = grid(64, 0.5);
        let pulses = Pulses::new(&g, &Cues::default());
        let double = Grouping::new(1, 0).unwrap();
        let off = Grouping::new(2, 1).unwrap();
        let totals = wanting(g.len(), |i| if i < 32 { double } else { off });
        let path = choose(&pulses, &totals, 2.0);
        assert_eq!((path[10], path[50]), (double, off));
        let (beats, _) = emit(&pulses, &path);
        // After the change the beats are the grid's midpoints.
        assert!(beats
            .iter()
            .any(|&b| (b - (1.0 + 50.0 * 0.5 + 0.25)).abs() < 1e-9));
        assert!(!beats.iter().any(|&b| (b - (1.0 + 50.0 * 0.5)).abs() < 1e-9));
    }

    #[test]
    fn short_gaps_hold_beats_and_long_ones_only_count() {
        let mut g: Vec<f64> = (0..20).map(|i| f64::from(i) * 0.5).collect();
        // The tracker skipped two beats, then the music stopped for 10 s.
        g.retain(|&t| t != 5.0 && t != 5.5);
        let last = g[g.len() - 1];
        g.extend((1..=10).map(|i| last + 10.0 + f64::from(i) * 0.5));
        let pulses = Pulses::new(&g, &Cues::default());
        let (beats, _) = emit(&pulses, &vec![Grouping::GRID; g.len()]);
        assert!(beats.iter().any(|&b| (b - 5.0).abs() < 1e-9));
        assert!(!beats.iter().any(|&b| b > last + 0.1 && b < last + 10.4));
        assert!(beats.iter().any(|&b| (b - (last + 10.5)).abs() < 1e-9));
    }

    #[test]
    fn the_backbeat_is_strongest_at_the_counted_beat() {
        // Eighth notes at 84 BPM, the snare on 2 and 4 (every fourth one).
        let g = grid(96, 60.0 / 168.0);
        let mut snare = Curve {
            values: vec![0.0; 6000],
            fps: 100.0,
        };
        for i in (1..96).step_by(4) {
            snare.values[(g[i] * 100.0).round() as usize] = 1.0;
        }
        let cues = Cues {
            snare,
            ..Cues::default()
        };
        let pulses = Pulses::new(&g, &cues);
        let weights = Weights::default();
        let backbeat = |pulses_per_beat, phase| {
            let grouping = Grouping::new(pulses_per_beat, phase).unwrap();
            score(grouping, 40, 140, &pulses, &[], &[], &weights).backbeat
        };
        assert!(backbeat(4, 2) > 0.9);
        assert!((backbeat(2, 0) - 0.5).abs() < 0.1);
        assert!(backbeat(8, 2) < 0.1);
    }

    #[test]
    fn spans_in_words() {
        let span = |start, pulses| Span {
            start,
            grouping: Grouping::new(pulses, 0).unwrap(),
        };
        assert_eq!(describe(&[span(0.0, 2)], 100.0), "");
        assert_eq!(describe(&[span(0.0, 6)], 100.0), "×⅓");
        assert_eq!(
            describe(&[span(0.0, 2), span(75.0, 4)], 100.0),
            "×1 75 %, ×½ 25 %"
        );
    }
}
