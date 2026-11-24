//! The word bubbles of the line being sung as the window shows them
//! ([`crate::core::bubbles`]): each fills with liquid as it is sung, each
//! of its notes' parts to a level (how much of that note was hit), poured
//! in from the part's start; a bubble full sweeps its level up to full
//! ([`SWEEP`], front to back along the word), straining in the last
//! stretch ([`pressure`]), then bursts, all of it at once ([`BURST`]: it
//! overfills, bulging for [`SWELL`], tears, its liquid splashing out, and
//! is gone). A bubble never full keeps its liquid at its level and never
//! strains or bursts. One bubble as the window shows it: [`Liquid`]
//! ([`Liquid::of`]).
//!
//! The time it goes by is the singing's own: the song's time as heard less
//! how late the voice comes (the frames of a moment come that much after
//! it is heard), so the sweep and the burst are seen whole, that much
//! later; a paused song holds them. Nothing is kept between frames: a
//! bubble no longer full (sought back) shows again as it is.

use super::karaoke::LyricLine;
use crate::core::bubbles::{self, Bubble};
use crate::core::singing::{Note, Singing};

pub use crate::core::bubbles::{FLIGHT, SWEEP, SWELL};
/// How long it bursts after that: it overfills, tears, and its liquid
/// splashes out and fades.
pub const BURST: f64 = 0.5;
/// The level from which a bubble strains: where it is full.
pub const STRAIN: f64 = bubbles::FULL;

/// How far the sweep of a bubble full at `full_at` is at `now`, 0 to 1 (0:
/// not full).
pub fn swept(full_at: Option<f64>, now: f64) -> f64 {
    full_at.map_or(0.0, |f| ((now - f) / SWEEP).clamp(0.0, 1.0))
}

/// How much a bubble filled to `level` (0 to 1) strains: none below
/// [`STRAIN`], all of it full.
pub fn pressure(level: f64) -> f64 {
    let t = ((level - STRAIN) / (1.0 - STRAIN)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The liquid in the part of bubble `b` over a note sung over `note`
/// (from, to) at `now`: its level (0 to 1: how much of it was hit, and the
/// sweep along the word as far as it has got), and when its fill jumped
/// (the sweep reaching it; none: -∞).
pub fn level(b: &Bubble, note: (f64, f64), now: f64) -> (f64, f64) {
    let long = (note.1 - note.0).max(1e-6);
    let base = (b.covered(note) / long).min(1.0);
    let Some(f) = b.full_at else {
        return (base, f64::NEG_INFINITY);
    };
    let wide = (b.time.1 - b.time.0).max(1e-6);
    let reach = b.time.0 + swept(Some(f), now) * wide;
    let swept = ((reach - note.0) / long).clamp(0.0, 1.0);
    let level = base + (1.0 - base) * swept;
    let jumped = f + SWEEP * ((note.0 - b.time.0) / wide).clamp(0.0, 1.0);
    (level, jumped)
}

/// A note's part of a word's bubble at a moment: which note's, under which
/// characters (from, to) and when sung (from, to, seconds), how high (as
/// the notes' heights), its liquid's level and when that last jumped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Part {
    /// Its note, as the line's.
    pub note: usize,
    /// Where it starts, in characters of the line.
    pub from: f32,
    /// Where it ends.
    pub to: f32,
    /// When its note is sung: from, to (seconds into the song).
    pub time: (f64, f64),
    /// How high: -1 to 1.
    pub height: f32,
    /// How full, 0 to 1.
    pub level: f64,
    /// When its fill jumped (the sweep reaching it), the singing's time;
    /// -∞: never.
    pub jumped: f64,
}

/// A word's bubble at a moment, as every frontend shows it: which word's,
/// its notes' parts, when it is sung (from, to, seconds), how full as a
/// whole, when it bursts (once swept full) and how long since, and a
/// number of its own (the same each frame) its wobble and splash are
/// scattered by.
#[derive(Clone, Debug, PartialEq)]
pub struct Liquid {
    /// Its word, as the line's.
    pub word: usize,
    /// Its notes' parts, in order.
    pub parts: Vec<Part>,
    /// When it is sung: from, to (seconds into the song).
    pub time: (f64, f64),
    /// How full, 0 to 1 (its parts' levels over their time).
    pub level: f64,
    /// When it bursts, the singing's time, if it is full.
    pub pops_at: Option<f64>,
    /// How long it has been bursting, once it has begun to (at most
    /// [`BURST`]: over, it stays the same from then on).
    pub age: Option<f64>,
    /// Its own number (24 bits, as the GPU takes it).
    pub seed: u32,
}

impl Liquid {
    /// Bubble `b` of line `line` (its notes `notes`, the line's), as the
    /// singing filled it, at `now` (the singing's time).
    pub fn of(b: &Bubble, line: usize, notes: &[Note], now: f64) -> Liquid {
        let parts: Vec<Part> = (b.notes.clone())
            .filter_map(|k| {
                let n = notes.get(k)?;
                let (level, jumped) = level(b, (n.start, n.end), now);
                Some(Part {
                    note: k,
                    from: n.from,
                    to: n.to,
                    time: (n.start, n.end),
                    height: n.height,
                    level,
                    jumped,
                })
            })
            .collect();
        let long = |p: &Part| (p.time.1 - p.time.0).max(1e-6);
        let time: f64 = parts.iter().map(long).sum();
        let filled: f64 = parts.iter().map(|p| p.level * long(p)).sum();
        let pops_at = b.full_at.map(|f| f + SWEEP);
        Liquid {
            word: b.word,
            parts,
            time: b.time,
            level: filled / time.max(1e-9),
            pops_at,
            age: pops_at.filter(|&p| now >= p).map(|p| (now - p).min(BURST)),
            seed: seed(line, b.word),
        }
    }

    /// How much of its word is filled, in seconds of it.
    pub fn filled(&self) -> f64 {
        let each = |p: &Part| p.level * (p.time.1 - p.time.0);
        self.parts.iter().map(each).sum()
    }

    /// Whether its note `k` is among its parts.
    pub fn has(&self, k: usize) -> bool {
        self.parts.iter().any(|p| p.note == k)
    }

    /// How much it strains ([`pressure`]).
    pub fn pressure(&self) -> f64 {
        pressure(self.level)
    }

    /// How far through its overfilling it is, 0 to 1, while it overfills.
    pub fn swelling(&self) -> Option<f64> {
        self.age.filter(|&a| a < SWELL).map(|a| a / SWELL)
    }

    /// Whether it has torn: overfilled, it is not drawn from then on.
    pub fn torn(&self) -> bool {
        self.age.is_some_and(|a| a >= SWELL)
    }

    /// How far through its burst it is, 0 to 1, while it bursts.
    pub fn bursting(&self) -> Option<f64> {
        self.age.filter(|&a| a < BURST).map(|a| a / BURST)
    }

    /// Whether its burst has begun (it is no longer a bubble to paint).
    pub fn begun(&self) -> bool {
        self.age.is_some()
    }
}

/// Bubble `word` of line `line`'s own number, the same every frame: what
/// its wobble and splash are scattered by (24 bits, as the GPU takes it).
pub fn seed(line: usize, word: usize) -> u32 {
    ((line as u32).wrapping_mul(7919) ^ (word as u32).wrapping_mul(104_729)) & 0xFF_FFFF
}

/// Line `i`'s bubbles as `singing` filled them, at `clock` (the singing's
/// time: the song's as heard less how late the voice comes), into `l`:
/// how far each note's part is filled (in characters, from its start),
/// and those with liquid in them, bursting or burst.
pub fn fill(singing: &Singing, i: usize, clock: f64, l: &mut LyricLine) {
    let Some(notes) = singing.notes.lines.get(i) else {
        return;
    };
    for b in &bubbles::line(singing, i, f64::INFINITY).words {
        let b = Liquid::of(b, i, &notes.notes, clock);
        if b.level <= 0.0 && !b.begun() {
            continue;
        }
        for p in &b.parts {
            l.hits
                .push((p.from, p.from + p.level as f32 * (p.to - p.from)));
        }
        l.liquid.push(b);
    }
    // Side by side as one.
    l.hits.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut merged: Vec<(f32, f32)> = Vec::with_capacity(l.hits.len());
    for h in l.hits.drain(..) {
        match merged.last_mut() {
            Some(m) if h.0 <= m.1 + 0.05 => m.1 = m.1.max(h.1),
            _ => merged.push(h),
        }
    }
    merged.retain(|h| h.1 > h.0);
    l.hits = merged;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A note of word 0 under characters `from` to `to`, sung `time`.
    fn note(from: f32, to: f32, time: (f64, f64), height: f32) -> Note {
        Note {
            from,
            to,
            height,
            start: time.0,
            end: time.1,
            cents: 6000.0,
            word: 0,
        }
    }

    /// A word's parts fill to the share of each hit; full, its level sweeps
    /// up to full front to back along the word; it strains only in the
    /// last stretch; it bursts once swept and stays burst (gone); one not
    /// full neither.
    #[test]
    fn a_word_fills_part_by_part_sweeps_full_then_bursts() {
        let notes = [
            note(0.0, 2.0, (0.0, 0.5), -0.5),
            note(2.0, 4.0, (0.5, 1.0), 0.5),
        ];
        let full = Bubble {
            word: 0,
            notes: 0..2,
            time: (0.0, 1.0),
            spans: vec![(0.0, 0.1), (0.2, 0.3)],
            full_at: Some(0.3),
        };
        let at = |k: usize, now: f64| level(&full, [(0.0, 0.5), (0.5, 1.0)][k], now);
        // Before its sweep: each part its share hit.
        let (l, j) = at(0, 0.29);
        assert!((l - 0.4).abs() < 1e-9, "{l}");
        assert!(j == 0.3, "{j}");
        assert_eq!(at(1, 0.29).0, 0.0);
        // Half way through the sweep: the first part full, the second not
        // reached yet.
        let mid = 0.3 + SWEEP / 2.0;
        assert!((at(0, mid).0 - 1.0).abs() < 1e-9 && at(1, mid).0 < 1e-9);
        assert!((at(1, 0.3 + SWEEP * 0.75).0 - 0.5).abs() < 1e-9);
        assert!((at(1, 0.3 + SWEEP / 2.0 + 0.001).1 - mid).abs() < 1e-9);
        // The whole: its parts over their time.
        let b = |now: f64| Liquid::of(&full, 0, &notes, now);
        assert!((b(0.29).level - 0.2).abs() < 1e-9 && b(0.29).parts.len() == 2);
        assert!((b(0.29).filled() - 0.2).abs() < 1e-9);
        // Strains only in its last stretch.
        assert_eq!(pressure(0.6), 0.0);
        assert!(pressure(0.95) > 0.5 && pressure(1.0) == 1.0);
        // Its burst: from once swept full, overfilling, then torn (not
        // drawn), then over, staying so.
        let pops = 0.3 + SWEEP;
        assert_eq!(b(0.1).pops_at, Some(pops));
        assert!(!b(pops - 0.01).begun() && b(pops - 0.01).bursting().is_none());
        assert!(b(pops + 0.04).swelling().is_some() && !b(pops + 0.04).torn());
        let mid = b(pops + 0.25);
        assert!(mid.swelling().is_none() && mid.torn());
        assert!((mid.bursting().unwrap() - 0.5).abs() < 1e-9);
        let over = b(pops + BURST + 9.0);
        assert!(over.torn() && over.bursting().is_none());
        assert_eq!(over, b(pops + BURST + 1.0), "the same once over");
        // Not full: its level, never more, never bursting.
        let never = Bubble {
            spans: vec![(0.5, 0.8)],
            full_at: None,
            ..full
        };
        let n = Liquid::of(&never, 0, &notes, 9.0);
        assert!((n.parts[1].level - 0.6).abs() < 1e-9 && n.parts[0].level == 0.0);
        assert!(n.parts[1].jumped == f64::NEG_INFINITY);
        assert!(n.pops_at.is_none() && !n.begun() && n.pressure() == 0.0);
    }
}
