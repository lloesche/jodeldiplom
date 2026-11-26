//! The word bubbles of Karaoke mode, its note targets (one under each
//! word sung, through
//! its notes): how the singing fills them, and when one is full. Pure,
//! from what [`Singing`] heard; the window sweeps a bubble full to full
//! and bursts it ([`crate::view::bubbles`]).
//!
//! A bubble fills where a frame of the singer's pitch hit one of its
//! notes (as [`Singing`] judged it; a gap of frames without a pitch, a
//! consonant, under [`BRIDGE`] filled over). It is full ([`Bubble::full_at`]):
//!
//! - as soon as its word counts as the scoring counts it ([`credit`]):
//!   its frames that hit against all the frames it will have had (those
//!   heard on it, and those to come until it ends), a little more to be
//!   sure; a word the scoring counts that never got there, full as it
//!   ends;
//! - once its fill covers [`FULL`] of it;
//! - rescued: a single word missed, the next word (once swept full) shoots
//!   it full as its shot lands ([`super::singing::Rescue`],
//!   [`Singing::rescue_of`]).
//!
//! A bubble full is a word the scoring counts hit: one full by its fill
//! has been hit more than the scoring asks (pinned by the tuning: a
//! compile-time check beside [`FULL`]), so a bubble never bursts, rescues
//! the word before it, and is then scored missed.
//!
//! Full, a bubble sweeps to full ([`SWEEP`]: its word won,
//! [`Bubble::swept_at`]) and bursts once its word is over too
//! ([`Bubble::bursts_at`]): a word still sung is held full until its end
//! passes the playhead, and the longer it was held the bigger it bursts
//! ([`burst_size`]).
//!
//! Each time is fixed by the frames up to it, so a bubble is full once.

use std::ops::Range;

use jodel_rhythm::yin::HOP;

use super::singing::judge::{credit, SHARE};
use super::singing::Singing;

/// The share of a bubble its fill covers for it to be full.
pub const FULL: f64 = 0.85;
/// The longest gap between frames that hit that is filled over, in
/// seconds.
pub const BRIDGE: f64 = 0.06;

// A bubble full by its fill is a word hit: each frame that hit fills at
// most its own frame and a gap bridged, so a word filled to FULL had at
// least FULL / (HOP + BRIDGE) of its 1 / HOP frames a second hit, more
// than the scoring's share of them (its frames up to the delay's slew
// more).
const _: () = assert!(FULL * HOP >= SHARE * (HOP + BRIDGE) * (1.0 + super::lag::SLEW));

/// A word's bubble as the singing filled it.
#[derive(Clone, Debug, PartialEq)]
pub struct Bubble {
    /// The word's index on its line.
    pub word: usize,
    /// Its notes' indices on the line.
    pub notes: Range<usize>,
    /// When it is sung: from, to (seconds into the song).
    pub time: (f64, f64),
    /// Where it is filled as sung: from, to (seconds into the song).
    pub spans: Vec<(f64, f64)>,
    /// When it filled whole (seconds into the song), if it did.
    pub full_at: Option<f64>,
}

impl Bubble {
    /// When it is swept full (seconds into the song), once full: its word
    /// won.
    pub fn swept_at(&self) -> Option<f64> {
        self.full_at.map(|f| f + SWEEP)
    }

    /// When it bursts (seconds into the song), once full: swept full and
    /// its word over.
    pub fn bursts_at(&self) -> Option<f64> {
        self.swept_at().map(|s| s.max(self.time.1))
    }

    /// How long it is held full before it bursts, in seconds: from its
    /// sweep to full to its word's end (0: not full, or its word over by
    /// then).
    pub fn held(&self) -> f64 {
        self.swept_at().map_or(0.0, |s| (self.time.1 - s).max(0.0))
    }

    /// How much of `from` to `to` (seconds into the song) its fill covers,
    /// in seconds.
    pub fn covered(&self, (from, to): (f64, f64)) -> f64 {
        let each = |&(a, z): &(f64, f64)| (z.min(to) - a.max(from)).max(0.0);
        self.spans.iter().map(each).sum()
    }
}

/// A line's bubbles: one a word with notes, in order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Bubbles {
    /// The bubbles.
    pub words: Vec<Bubble>,
}

impl Bubbles {
    /// Word `word`'s bubble.
    pub fn word(&self, word: usize) -> Option<&Bubble> {
        self.words.iter().find(|w| w.word == word)
    }

    /// The bubbles full.
    pub fn words_full(&self) -> usize {
        self.words.iter().filter(|w| w.full_at.is_some()).count()
    }
}

/// Line `l`'s bubbles as `singing` heard them up to `upto` seconds into
/// the song; none for a line it has no notes of.
pub fn line(singing: &Singing, l: usize, upto: f64) -> Bubbles {
    let Some(notes) = singing.notes.lines.get(l) else {
        return Bubbles::default();
    };
    let (Some(first), Some(last)) = (notes.notes.first(), notes.notes.last()) else {
        return Bubbles::default();
    };
    let mut out = Bubbles::default();
    let mut word_of = Vec::with_capacity(notes.notes.len());
    for (k, n) in notes.notes.iter().enumerate() {
        match out.words.last_mut() {
            Some(w) if w.word == n.word => {
                w.notes.end = k + 1;
                w.time.1 = w.time.1.max(n.end);
            }
            _ => out.words.push(Bubble {
                word: n.word,
                notes: k..k + 1,
                time: (n.start, n.end),
                spans: Vec::new(),
                full_at: None,
            }),
        }
        word_of.push(out.words.len() - 1);
    }
    // The frames each word will have had, as the scoring counts them: those
    // heard on it, and those to come (a frame each HOP) until it ends.
    let heard_to = singing.last_heard().unwrap_or(f64::NEG_INFINITY);
    let expected: Vec<f64> = out
        .words
        .iter()
        .map(|w| {
            let (start, end) = w.time;
            let to_come = if heard_to < start {
                ((end - start) / HOP - 1e-6).ceil()
            } else {
                ((end - heard_to) / HOP - 1e-6).ceil() - 1.0
            };
            f64::from(singing.noted(l, w.word)) + to_come.max(0.0)
        })
        .collect();
    let counts = |frames: f64, hit: f64| credit(frames, hit) >= 1.0;
    // How much of each bubble the fill covers (seconds), the frames that
    // hit each word.
    let mut covered = vec![0.0; out.words.len()];
    let mut hits = vec![0.0; out.words.len()];
    for h in singing.heard_on(l, first.start, upto.min(last.end)) {
        if !h.hit {
            continue;
        }
        let k = notes.notes.partition_point(|n| n.start <= h.t);
        let Some(k) = k.checked_sub(1).filter(|&k| h.t < notes.notes[k].end) else {
            continue;
        };
        let w = word_of[k];
        let b = &mut out.words[w];
        let to = (h.t + HOP).min(b.time.1);
        match b.spans.last_mut() {
            Some(s) if h.t - s.1 < BRIDGE => {
                if to > s.1 {
                    covered[w] += to - s.1;
                    s.1 = to;
                }
            }
            _ => {
                covered[w] += to - h.t;
                b.spans.push((h.t, to));
            }
        }
        if b.full_at.is_none() && covered[w] >= FULL * (b.time.1 - b.time.0) {
            b.full_at = Some(h.t);
        }
        // A word counts as soon as it is hit for its share of a little
        // more than it will have had (the delay slewing packs up to its
        // slew more frames into a second): full.
        hits[w] += 1.0;
        if counts(expected[w] * (1.0 + super::lag::SLEW) + 1.0, hits[w]) {
            b.full_at.get_or_insert(h.t);
        }
    }
    for (w, b) in out.words.iter_mut().enumerate() {
        // A word over that the scoring counts and that never got there
        // early: full as it ends.
        let end = b.time.1;
        if heard_to >= end && counts(expected[w], hits[w]) {
            b.full_at.get_or_insert(end);
        }
        // Missed, then shot full by the next word swept full.
        if let (None, Some(r)) = (b.full_at, singing.rescue_of(l, b.word)) {
            b.full_at = Some(r.lands());
        }
    }
    out
}

/// The words of `singing`'s song full by `upto` (their bubbles), and the
/// words with notes.
pub fn words_full(singing: &Singing, upto: f64) -> (usize, usize) {
    (0..singing.notes.lines.len())
        .map(|l| line(singing, l, upto))
        .fold((0, 0), |(n, all), b| {
            (n + b.words_full(), all + b.words.len())
        })
}

/// How long a bubble full takes to sweep to full before it bursts, in
/// seconds of the song.
pub const SWEEP: f64 = 0.2;

/// How long of its burst a bubble overfills (bulging) before it tears and
/// its liquid splashes out, in seconds.
pub const SWELL: f64 = 0.08;

/// A bubble held full up to this long (seconds) bursts as any.
pub const HELD_SHORT: f64 = 0.3;
/// One held full this long or longer bursts the biggest.
pub const HELD_LONG: f64 = 2.0;

/// How big a bubble held full for `held` seconds bursts, 0 to 1: 0 up to
/// [`HELD_SHORT`] (as any), growing evenly to 1 at [`HELD_LONG`].
pub fn burst_size(held: f64) -> f64 {
    ((held - HELD_SHORT) / (HELD_LONG - HELD_SHORT)).clamp(0.0, 1.0)
}

/// How long a rescuing shot flies from the bubble that fires it to the
/// one it rescues, in seconds.
pub const FLIGHT: f64 = 0.32;

#[cfg(test)]
mod tests;
