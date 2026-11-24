//! A phrase heard, measured: the original singer's, once, as the
//! [`Yardstick`] a take is held to ([`yardstick`]); a take's against it
//! ([`phrase`]: a [`Measured`], scored by [`super::shine`]).
//!
//! Each frame with a pitch is held against the best of the song's notes
//! within [`SLACK`] of it, in the singer's key (as the bubbles follow it),
//! so a delay a little off or a singer a little early or late does not
//! count against the pitch: the timing measures that. On the frames placed
//! on the phrase's notes:
//!
//! - each word's frames within the bubbles' [`TOLERANCE`] of its notes,
//!   those sung while it is (the coverage: a word against what the
//!   original has on it, so a word left out is not made up for by its
//!   neighbours' frames held against it);
//! - how far off the notes the pitch is (the median, in the singer's key
//!   or the song's, whichever is nearer);
//! - the spread on held notes (drift and jitter, a vibrato let off);
//! - the voicing about the phrase, a frame each 10 ms, against the
//!   original's: the timing, by the lag it fits best at;
//! - each long note's second half held;
//! - the phrase's top and bottom notes reached.

use super::Frame;
use crate::core::singing::judge::{fold, TOLERANCE};
use crate::core::singing::{LineNotes, Notes};

/// How far from a frame (s) its note may be.
pub const SLACK: f64 = 0.06;
/// Where around the frame its note is looked for, in [`SLACK`]s.
const SLACK_STEPS: [f64; 5] = [0.0, -0.5, 0.5, -1.0, 1.0];
/// How far around the phrase its voicing is looked at (s): the phrase is
/// measured once the frames reach this far past its end.
pub const MASK_AROUND: f64 = 0.35;
/// Steadiness: notes at least this long (s) are held notes.
const HELD: f64 = 0.4;
/// The fewest voiced frames in a row on a held note looked at.
const HELD_RUN: usize = 30;
/// The widest gap (s) a held note's voiced run bridges: a frame or two
/// lost to the pitch tracker does not end it.
const RUN_GAP: f64 = 0.035;
/// The fewest frames on held notes the steadiness is told from.
const STEADY_LEAST: usize = 50;
/// The drift: the pitch averaged over this many frames (about one cycle of
/// a 4.8 Hz vibrato, so the vibrato averages out).
const DRIFT_FRAMES: usize = 21;
/// Vibrato: the lags (frames) of 4 to 7.7 Hz.
const VIB_LAGS: (usize, usize) = (13, 25);
/// The least autocorrelation that is a vibrato.
const VIB_PERIODIC: f64 = 0.3;
/// The widest vibrato let off (RMS cents: about ±100 cents peak).
const VIB_WIDEST: f64 = 71.0;
/// Timing: the lags (s) the phrase's voicing is tried at.
const LAG_MOST: f64 = 0.30;
/// The fewest frames of the original's voicing a phrase's timing is told
/// from (fewer: no timing to hold a singer to), and the least fit of the
/// singer's to it (less: not in time at all).
const LAG_LEAST: usize = 20;
const LAG_FIT: f64 = 0.2;
/// Timing: this close to the original (s) scores fully, this far nothing.
const TIMING_FULL: f64 = 0.06;
const TIMING_NONE: f64 = 0.25;
/// Sustain: notes at least this long (s), their second half in bins this
/// long, each held where a frame near the note is in it.
const LONG: f64 = 0.5;
const SUSTAIN_BIN: f64 = 0.1;
/// Range: a phrase spanning at least this many cents has a top and a
/// bottom to reach, each with this share of its frames within
/// [`RANGE_NEAR`]; only notes at least [`RANGE_LEAST`] long count.
const RANGE_SPAN: f64 = 300.0;
const RANGE_SHARE: f64 = 0.15;
const RANGE_NEAR: f64 = 100.0;
const RANGE_LEAST: f64 = 0.15;
/// Words the original has fewer frames on are not counted (too little
/// heard of the original to hold a singer to).
pub const WORD_LEAST: f64 = 3.0;
/// A word asks this share of the original's frames on it.
const WORD_OF_ORIGINAL: f64 = 0.5;

/// The original's phrase as a take is held to it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Yardstick {
    /// Each word's frames within [`TOLERANCE`] of its notes, sung while it
    /// is.
    pub word_hits: Vec<f64>,
    /// The share of its note time with a pitch on its notes.
    pub voiced: f64,
    /// How far off the notes its pitch is, the median (cents).
    pub off: Option<f64>,
    /// The spread on held notes (cents), where it has enough of them.
    pub spread: Option<f64>,
    /// Each long note's second half held (0 to 1).
    pub held: Vec<f64>,
    /// Its top and bottom reached, averaged (0 to 1), where it has them.
    pub range: Option<f64>,
    /// The voicing from [`MASK_AROUND`] before its first note to as far
    /// after its last, a frame each 10 ms.
    pub mask: Vec<bool>,
}

impl Yardstick {
    /// The words the coverage counts: those it is heard on.
    pub fn counted(&self) -> usize {
        self.word_hits.iter().filter(|&&h| h >= WORD_LEAST).count()
    }

    /// Whether its voicing has a timing to hold a singer to: enough of it,
    /// and not all of the time.
    pub fn times(&self) -> bool {
        let voiced = self.mask.iter().filter(|&&v| v).count();
        voiced >= LAG_LEAST && voiced < self.mask.len()
    }
}

/// A take's phrase measured against the original's [`Yardstick`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Measured {
    /// Each counted word's frames on it against what it asks, averaged
    /// (0 to 1).
    pub coverage: f64,
    /// How far off the notes its pitch is, the median (cents), where it has
    /// a pitch on them.
    pub off: Option<f64>,
    /// The spread on held notes (cents), where it has enough of them.
    pub spread: Option<f64>,
    /// The timing score (0 to 1) where the original has one: 0 where its
    /// voicing fits the original's nowhere.
    pub timing: Option<f64>,
    /// Each long note's second half held (0 to 1).
    pub held: Vec<f64>,
    /// Its top and bottom reached (0 to 1), where it has them.
    pub range: Option<f64>,
}

/// What a hearing of a phrase has: its words' frames near their notes and
/// what each asks, and the measures.
struct Hearing {
    word_hits: Vec<f64>,
    voiced: f64,
    off: Option<f64>,
    spread: Option<f64>,
    held: Vec<f64>,
    range: Option<f64>,
    mask: Vec<bool>,
}

/// Line `l` of `notes` heard on `frames` (in time, one piece of singing);
/// none where the frames do not reach over all of it (and [`MASK_AROUND`]
/// each side), or it has no notes.
fn hearing(notes: &Notes, l: usize, frames: &[Frame]) -> Option<Hearing> {
    let ln = notes.lines.get(l)?;
    let (first, last) = (ln.notes.first()?, ln.notes.last()?);
    let (start, end) = (first.start, last.end);
    let (from, to) = (frames.first()?.t, frames.last()?.t);
    if start - MASK_AROUND < from || end + MASK_AROUND > to {
        return None;
    }
    let frame_at = |t: f64| frames.partition_point(|f| f.t < t);
    let on = placed(
        notes,
        l,
        &frames[frame_at(start - SLACK)..frame_at(end + SLACK)],
    );
    let mut words: Vec<usize> = ln.notes.iter().map(|n| n.word).collect();
    words.dedup();
    let word_hits = (words.iter())
        .map(|&w| {
            let sung_in = |f: &&Placed| {
                notes
                    .word_at(f.t)
                    .is_some_and(|((line, word), _)| (line, word) == (l, w))
            };
            (on.iter().filter(sung_in))
                .filter(|f| f.free.abs() <= TOLERANCE)
                .count() as f64
        })
        .collect();
    let note_frames: f64 = ln.notes.iter().map(|n| frames_of(n.end - n.start)).sum();
    let off = (!on.is_empty()).then(|| {
        let mut abs: Vec<f64> = on.iter().map(|f| f.free.abs().min(f.keyed.abs())).collect();
        abs.sort_by(f64::total_cmp);
        abs[abs.len() / 2]
    });
    let around = &frames[frame_at(start - MASK_AROUND)..frame_at(end + MASK_AROUND)];
    Some(Hearing {
        word_hits,
        voiced: (on.len() as f64 / note_frames).min(1.0),
        off,
        spread: spread(ln, &on),
        held: held(ln, &on),
        range: range(ln, &on),
        mask: mask_of(around, start, end),
    })
}

/// Line `l` of `notes` as the original sings it on `frames`: its
/// yardstick; none where the frames do not reach over it.
pub fn yardstick(notes: &Notes, l: usize, frames: &[Frame]) -> Option<Yardstick> {
    let h = hearing(notes, l, frames)?;
    Some(Yardstick {
        word_hits: h.word_hits,
        voiced: h.voiced,
        off: h.off,
        spread: h.spread,
        held: h.held,
        range: h.range,
        mask: h.mask,
    })
}

/// Line `l` of `notes` as sung on `frames`, against the original's
/// yardstick `y`; none where the frames do not reach over it.
///
/// The coverage: each word the original is counted on, its frames near
/// its notes (within the bubbles' [`TOLERANCE`]) sung while it is, against
/// half the original's ([`WORD_OF_ORIGINAL`]). The timing: the lag the voicing fits the
/// original's best at, where the original has one to tell (0 where it fits
/// nowhere).
pub fn phrase(notes: &Notes, l: usize, frames: &[Frame], y: &Yardstick) -> Option<Measured> {
    let h = hearing(notes, l, frames)?;
    let (mut credit, mut counted) = (0.0, 0usize);
    for (hit, theirs) in h.word_hits.iter().zip(&y.word_hits) {
        if *theirs < WORD_LEAST {
            continue;
        }
        credit += (hit / (WORD_OF_ORIGINAL * theirs).max(1.0)).min(1.0);
        counted += 1;
    }
    let timing = y.times().then(|| {
        lag_of(&h.mask, &y.mask)
            .filter(|&(_, fit)| fit >= LAG_FIT)
            .map_or(0.0, |(lag, _)| timing_score(lag))
    });
    Some(Measured {
        coverage: credit / counted.max(1) as f64,
        off: h.off,
        spread: h.spread,
        timing,
        held: h.held,
        range: h.range,
    })
}

/// A frame with a pitch, placed on a note: its time, pitch, the note
/// (line, index) it is held against, and how far off that note it is, in
/// the singer's key and in the song's (octaves aside).
#[derive(Clone, Copy, Debug)]
struct Placed {
    t: f64,
    cents: f64,
    line: usize,
    note: usize,
    free: f64,
    keyed: f64,
}

/// A note `seconds` long, in frames (at least one).
fn frames_of(seconds: f64) -> f64 {
    (seconds * 100.0).max(1.0)
}

/// `frames` (those about line `l`) with a pitch and a key, each placed on
/// the best of the notes within [`SLACK`] of it; those on line `l`'s.
fn placed(notes: &Notes, l: usize, frames: &[Frame]) -> Vec<Placed> {
    let mut out = Vec::new();
    for f in frames {
        let (Some(c), Some(key)) = (f.cents, f.key) else {
            continue;
        };
        let c = f64::from(c);
        let mut best: Option<Placed> = None;
        for d in SLACK_STEPS {
            let Some((line, k)) = notes.note_at(f.t + d * SLACK) else {
                continue;
            };
            let note = notes.lines[line].notes[k].cents;
            let free = fold(c - key - note);
            if best.is_none_or(|b| free.abs() < b.free.abs()) {
                best = Some(Placed {
                    t: f.t,
                    cents: c,
                    line,
                    note: k,
                    free,
                    keyed: fold(c - note),
                });
            }
        }
        out.extend(best.filter(|b| b.line == l));
    }
    out
}

/// The spread on held notes: each voiced run on one at least [`HELD_RUN`]
/// long, weighted by its length; none with fewer than [`STEADY_LEAST`]
/// frames on them.
fn spread(ln: &LineNotes, on: &[Placed]) -> Option<f64> {
    let (mut sum, mut n) = (0.0, 0usize);
    for (k, note) in ln
        .notes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.end - n.start >= HELD)
    {
        let mut runs: Vec<Vec<f64>> = vec![Vec::new()];
        let mut last_t = f64::NEG_INFINITY;
        for f in on
            .iter()
            .filter(|f| f.note == k && f.t >= note.start + 0.05 && f.t < note.end - 0.03)
        {
            if f.t - last_t > RUN_GAP {
                runs.push(Vec::new());
            }
            runs.last_mut()?.push(f.cents);
            last_t = f.t;
        }
        for r in runs.iter().filter(|r| r.len() >= HELD_RUN) {
            sum += spread_of(r) * r.len() as f64;
            n += r.len();
        }
    }
    (n >= STEADY_LEAST).then(|| sum / n as f64)
}

/// One voiced run's spread (cents): its drift (the pitch averaged over
/// [`DRIFT_FRAMES`]) and its jitter about that, a vibrato's periodic part
/// let off.
fn spread_of(run: &[f64]) -> f64 {
    let mut sorted = run.to_vec();
    sorted.sort_by(f64::total_cmp);
    let med = sorted[sorted.len() / 2];
    let x: Vec<f64> = run.iter().map(|&c| med + fold(c - med)).collect();
    let n = x.len();
    let half = DRIFT_FRAMES / 2;
    let drift: Vec<f64> = (0..n)
        .map(|i| {
            let (a, b) = (i.saturating_sub(half), (i + half + 1).min(n));
            x[a..b].iter().sum::<f64>() / (b - a) as f64
        })
        .collect();
    let mean = drift.iter().sum::<f64>() / n as f64;
    let wobble = (drift.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    let r: Vec<f64> = x.iter().zip(&drift).map(|(a, b)| a - b).collect();
    let power = r.iter().map(|v| v * v).sum::<f64>();
    let rms = (power / n as f64).sqrt();
    let mut periodic = 0.0f64;
    if power > 0.0 {
        for lag in VIB_LAGS.0..=VIB_LAGS.1.min(n / 2) {
            let c: f64 = (0..n - lag).map(|i| r[i] * r[i + lag]).sum::<f64>() / power;
            periodic = periodic.max(c);
        }
    }
    let vibrato = periodic >= VIB_PERIODIC && rms <= VIB_WIDEST && rms > 8.0;
    let jitter = if vibrato {
        rms * (1.0 - periodic).max(0.0).sqrt()
    } else {
        rms
    };
    wobble.hypot(jitter)
}

/// The voicing of `frames` from [`MASK_AROUND`] before `start` to as far
/// after `end`, a frame each 10 ms.
fn mask_of(frames: &[Frame], start: f64, end: f64) -> Vec<bool> {
    let n = ((end - start + 2.0 * MASK_AROUND) * 100.0).round() as usize;
    let mut out = vec![false; n];
    for f in frames.iter().filter(|f| f.cents.is_some()) {
        let i = ((f.t - (start - MASK_AROUND)) * 100.0).round();
        if i >= 0.0 && (i as usize) < n {
            out[i as usize] = true;
        }
    }
    out
}

/// The lag (s) the singer's voicing best matches the original's at
/// (positive: late), and how well (their correlation); none where either
/// is all one or the other.
fn lag_of(singer: &[bool], original: &[bool]) -> Option<(f64, f64)> {
    let n = singer.len().min(original.len());
    if n < 20 {
        return None;
    }
    let mean = |m: &[bool]| m.iter().filter(|&&v| v).count() as f64 / m.len() as f64;
    let (ms, mo) = (mean(&singer[..n]), mean(&original[..n]));
    if ms <= 0.0 || mo <= 0.0 || mo >= 1.0 || ms >= 1.0 {
        return None;
    }
    let most = (LAG_MOST * 100.0) as isize;
    let mut best: Option<(isize, f64)> = None;
    for lag in -most..=most {
        let mut sum = 0.0;
        for i in 0..n as isize {
            let j = i + lag;
            if j < 0 || j >= n as isize {
                continue;
            }
            let s = f64::from(u8::from(singer[j as usize])) - ms;
            let o = f64::from(u8::from(original[i as usize])) - mo;
            sum += s * o;
        }
        // Ties to the smaller lag.
        let better = best
            .is_none_or(|(bl, bs)| sum > bs + 1e-9 || (sum > bs - 1e-9 && lag.abs() < bl.abs()));
        if better {
            best = Some((lag, sum));
        }
    }
    let (lag, sum) = best?;
    let norm = (n as f64 * ms * (1.0 - ms) * n as f64 * mo * (1.0 - mo)).sqrt();
    Some((lag as f64 / 100.0, sum / norm.max(1e-9)))
}

fn timing_score(lag: f64) -> f64 {
    ((TIMING_NONE - lag.abs()) / (TIMING_NONE - TIMING_FULL)).clamp(0.0, 1.0)
}

/// Each long note's second half held: in bins of [`SUSTAIN_BIN`], each
/// held where a frame near the note (within [`TOLERANCE`], whichever note
/// it is placed on) is in it.
fn held(ln: &LineNotes, on: &[Placed]) -> Vec<f64> {
    let notes = &ln.notes;
    let mut out = Vec::new();
    for n in notes.iter().filter(|n| n.end - n.start >= LONG) {
        let from = n.start + 0.5 * (n.end - n.start);
        let bins = ((n.end + SLACK - from) / SUSTAIN_BIN).ceil().max(1.0) as usize;
        let mut got = vec![false; bins];
        for f in on
            .iter()
            .filter(|f| f.t >= from && f.t <= n.end + SLACK)
            .filter(|f| fold(f.free + notes[f.note].cents - n.cents).abs() <= TOLERANCE)
        {
            let b = ((f.t - from) / SUSTAIN_BIN) as usize;
            if b < bins {
                got[b] = true;
            }
        }
        out.push(got.iter().filter(|&&g| g).count() as f64 / bins as f64);
    }
    out
}

/// The phrase's top and bottom notes reached (each the best of the notes
/// there, a share of [`RANGE_SHARE`] of its frames near it), averaged;
/// none where it spans less than [`RANGE_SPAN`].
fn range(ln: &LineNotes, on: &[Placed]) -> Option<f64> {
    let notes = &ln.notes;
    let long: Vec<usize> = (0..notes.len())
        .filter(|&k| notes[k].end - notes[k].start >= RANGE_LEAST)
        .collect();
    let top = long
        .iter()
        .map(|&k| notes[k].cents)
        .fold(f64::NEG_INFINITY, f64::max);
    let bottom = long
        .iter()
        .map(|&k| notes[k].cents)
        .fold(f64::INFINITY, f64::min);
    if top - bottom < RANGE_SPAN {
        return None;
    }
    let reached = |want: &dyn Fn(f64) -> bool| {
        long.iter()
            .copied()
            .filter(|&k| want(notes[k].cents))
            .map(|k| {
                let n = &notes[k];
                let near = on
                    .iter()
                    .filter(|f| f.t >= n.start - SLACK && f.t <= n.end + SLACK)
                    .filter(|f| fold(f.free + notes[f.note].cents - n.cents).abs() <= RANGE_NEAR)
                    .count() as f64;
                (near / frames_of(n.end - n.start) / RANGE_SHARE).min(1.0)
            })
            .fold(0.0, f64::max)
    };
    let hi = reached(&|c| c >= top - 50.0);
    let lo = reached(&|c| c <= bottom + 50.0);
    Some((hi + lo) / 2.0)
}
