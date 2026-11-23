//! The note bubbles of Karaoke mode (a capsule under each note the words
//! are sung on): how the singing fills them, and when one is full. Pure,
//! from what [`Singing`] heard; the frontends sweep a bubble full to full
//! and burst it ([`crate::view::bubbles`]).
//!
//! A bubble fills where a frame of the singer's pitch hit its note (fully
//! or half, as [`Singing`] judged it; a gap of frames without a pitch, a
//! consonant, under [`BRIDGE`] filled over, on every level, as the stage
//! always did). It is full ([`Bubble::full_at`]):
//!
//! - on [`Level::Easy`], as soon as its word counts: all the word's
//!   bubbles at once, the part already sung and the part to come. A word
//!   counts as the scoring counts it ([`Level::credit`]): its frames that
//!   hit against all the frames it will have had (those heard on it, and
//!   those to come until it ends), a little more to be sure; a word the
//!   scoring counts that never got there, full as it ends;
//! - on any level, once its fill covers [`FULL`] of it.
//!
//! A word is full once all its bubbles are ([`Word::full_at`]). Each time
//! is fixed by the frames up to it, so a bubble is full once.

use std::ops::Range;

use jodel_rhythm::yin::HOP;

use super::singing::{Hit, Level, Singing};

/// The share of a bubble its fill covers for it to be full.
pub const FULL: f64 = 0.85;
/// The longest gap between frames that hit that is filled over, in
/// seconds.
pub const BRIDGE: f64 = 0.06;

/// A note's bubble as the singing filled it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Bubble {
    /// Where it is filled as sung: from, to (seconds into the song), and
    /// whether fully hit (else half).
    pub spans: Vec<(f64, f64, bool)>,
    /// When it filled whole (seconds into the song), if it did.
    pub full_at: Option<f64>,
    /// Whether it filled with its word (on easy, its word counting): it
    /// sweeps full across the word, else across itself.
    pub by_word: bool,
}

/// A word's bubbles.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    /// The word's index on its line.
    pub word: usize,
    /// Its notes' (and bubbles') indices.
    pub notes: Range<usize>,
    /// When it is sung: from, to (seconds into the song).
    pub time: (f64, f64),
    /// When the last of its bubbles filled, once all are full.
    pub full_at: Option<f64>,
}

/// A line's bubbles: one a note, in order, and its words.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Bubbles {
    /// The bubbles, as the line's notes.
    pub notes: Vec<Bubble>,
    /// The words, in order.
    pub words: Vec<Word>,
}

impl Bubbles {
    /// Word `word` of the line.
    pub fn word(&self, word: usize) -> Option<&Word> {
        self.words.iter().find(|w| w.word == word)
    }

    /// The words full.
    pub fn words_full(&self) -> usize {
        self.words.iter().filter(|w| w.full_at.is_some()).count()
    }
}

/// Line `l`'s bubbles as `singing` heard them up to `upto` seconds into
/// the song; none for a line it has no notes of.
pub fn line(singing: &Singing, l: usize, upto: f64) -> Bubbles {
    let Some(notes) = singing.lines.get(l) else {
        return Bubbles::default();
    };
    let (Some(first), Some(last)) = (notes.notes.first(), notes.notes.last()) else {
        return Bubbles::default();
    };
    let level = singing.level;
    let mut out = Bubbles {
        notes: vec![Bubble::default(); notes.notes.len()],
        words: Vec::new(),
    };
    let mut word_of = Vec::with_capacity(notes.notes.len());
    for (k, n) in notes.notes.iter().enumerate() {
        match out.words.last_mut() {
            Some(w) if w.word == n.word => {
                w.notes.end = k + 1;
                w.time.1 = w.time.1.max(n.end);
            }
            _ => out.words.push(Word {
                word: n.word,
                notes: k..k + 1,
                time: (n.start, n.end),
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
    let counts = |frames: f64, hit: f64| level.credit(frames, hit, 0.0).hit() == Some(Hit::Full);
    // How much of each bubble the fill covers (seconds), the frames that
    // hit each word.
    let mut covered = vec![0.0; out.notes.len()];
    let mut hits = vec![0.0; out.words.len()];
    for h in singing.heard_on(l, first.start, upto.min(last.end)) {
        let Some(hit) = h.hit else { continue };
        let k = notes.notes.partition_point(|n| n.start <= h.t);
        let Some(k) = k.checked_sub(1).filter(|&k| h.t < notes.notes[k].end) else {
            continue;
        };
        let (full, n) = (hit == Hit::Full, &notes.notes[k]);
        let b = &mut out.notes[k];
        let to = (h.t + HOP).min(n.end);
        match b.spans.last_mut() {
            Some(s) if s.2 == full && h.t - s.1 < BRIDGE => {
                if to > s.1 {
                    covered[k] += to - s.1;
                    s.1 = to;
                }
            }
            _ => {
                covered[k] += to - h.t;
                b.spans.push((h.t, to, full));
            }
        }
        if b.full_at.is_none() && covered[k] >= FULL * (n.end - n.start) {
            b.full_at = Some(h.t);
        }
        // On easy a word counts as soon as it is hit for its share of a
        // little more than it will have had (the delay slewing packs up to
        // its slew more frames into a second): all its bubbles full.
        let w = word_of[k];
        if level == Level::Easy && full {
            hits[w] += 1.0;
            if counts(expected[w] * (1.0 + super::lag::SLEW) + 1.0, hits[w]) {
                fill_word(&mut out, w, h.t);
            }
        }
    }
    // A word over that the scoring counts and that never got there early:
    // full as it ends.
    for w in 0..out.words.len() {
        let end = out.words[w].time.1;
        if level == Level::Easy && heard_to >= end && counts(expected[w], hits[w]) {
            fill_word(&mut out, w, end);
        }
        let word = &mut out.words[w];
        word.full_at = out.notes[word.notes.clone()]
            .iter()
            .map(|b| b.full_at)
            .try_fold(f64::NEG_INFINITY, |m, f| f.map(|f| m.max(f)));
    }
    out
}

/// Word `w`'s bubbles not yet full, full with it at `t`.
fn fill_word(out: &mut Bubbles, w: usize, t: f64) {
    for b in &mut out.notes[out.words[w].notes.clone()] {
        if b.full_at.is_none() {
            b.full_at = Some(t);
            b.by_word = true;
        }
    }
}

/// The words of `singing`'s song full by `upto` (all their bubbles), and
/// the words with notes.
pub fn words_full(singing: &Singing, upto: f64) -> (usize, usize) {
    (0..singing.lines.len())
        .map(|l| line(singing, l, upto))
        .fold((0, 0), |(n, all), b| {
            (n + b.words_full(), all + b.words.len())
        })
}

#[cfg(test)]
mod tests;
