//! The note bubbles of the line being sung as the frontends show them
//! ([`crate::core::bubbles`]): each fills with liquid as it is sung, to a
//! level (how much of it was hit, fully or half), poured in from its
//! start; a bubble full sweeps its level up to full ([`SWEEP`], across its
//! word where it filled with it, else across itself), straining in the
//! last stretch ([`pressure`]), then bursts, all of it at once, on its own
//! ([`BURST`]: it overfills, bulging for [`SWELL`], tears, its liquid
//! splashing out, and is gone). A bubble never full keeps its liquid at
//! its level and never strains or bursts. One bubble as every frontend
//! shows it: [`Liquid`] ([`Liquid::of`]); the notes' row in cells, where
//! a bubble bursting or burst is blank.
//!
//! The time it goes by is the singing's own: the song's time as heard less
//! how late the voice comes (the frames of a moment come that much after
//! it is heard), so the sweep and the burst are seen whole, that much
//! later; a paused song holds them. Nothing is kept between frames: a
//! bubble no longer full (sought back) shows again as it is.

use ratatui::buffer::Buffer;

use super::karaoke::{LyricLine, HALF_HIT, HIT, NOTE_AHEAD, SINGER};
use crate::core::bubbles::{self, Bubbles};
use crate::core::singing::{Note, Singing};
use crate::core::text::width;

/// How long a bubble full takes to sweep to full, in seconds of the song.
pub const SWEEP: f64 = 0.2;
/// How long it bursts after that: it overfills, tears, and its liquid
/// splashes out and fades.
pub const BURST: f64 = 0.5;
/// How long of its burst it overfills (bulging) before it tears, in
/// seconds: from then on it is not drawn.
pub const SWELL: f64 = 0.08;
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

/// The liquid in bubble `k` of `bubbles` (its note sung over `note`: from,
/// to) at `now`: its level (0 to 1: how much of it was hit, and the sweep
/// as far as it has got, across its word where it filled with it), the
/// share of that fully hit (else half), and when its fill jumped (the
/// sweep reaching it; none: -∞).
pub fn level(bubbles: &Bubbles, k: usize, note: (f64, f64), now: f64) -> (f64, f64, f64) {
    let b = &bubbles.notes[k];
    let long = (note.1 - note.0).max(1e-6);
    let (mut full, mut half) = (0.0, 0.0);
    for &(a, z, f) in &b.spans {
        let len = (z.min(note.1) - a.max(note.0)).max(0.0);
        if f {
            full += len;
        } else {
            half += len;
        }
    }
    let base = ((full + half) / long).min(1.0);
    let across = match bubbles.words.iter().find(|w| w.notes.contains(&k)) {
        Some(w) if b.by_word => w.time,
        _ => note,
    };
    let Some(f) = b.full_at else {
        return (
            base,
            if base > 0.0 { full / long / base } else { 0.0 },
            f64::NEG_INFINITY,
        );
    };
    let wide = (across.1 - across.0).max(1e-6);
    let reach = across.0 + swept(Some(f), now) * wide;
    let swept = ((reach - note.0) / long).clamp(0.0, 1.0);
    let level = base + (1.0 - base) * swept;
    let share = ((full / long + level - base) / level.max(1e-6)).min(1.0);
    let jumped = f + SWEEP * ((note.0 - across.0) / wide).clamp(0.0, 1.0);
    (level, share, jumped)
}

/// A note's bubble at a moment, as every frontend shows it: which note's,
/// under which characters (from, to) and when sung (from, to, seconds),
/// how high (as the notes' heights), its liquid (its level and the share
/// fully hit) and when that last jumped, when it bursts (once swept full)
/// and how long since, and a number of its own (the same each frame) its
/// wobble and splash are scattered by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Liquid {
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
    /// The share of that fully hit.
    pub full: f64,
    /// When its fill jumped (the sweep reaching it), the singing's time;
    /// -∞: never.
    pub jumped: f64,
    /// When it bursts, the singing's time, if it is full.
    pub pops_at: Option<f64>,
    /// How long it has been bursting, once it has begun to (at most
    /// [`BURST`]: over, it stays the same from then on).
    pub age: Option<f64>,
    /// Its own number (24 bits, as the GPU takes it).
    pub seed: u32,
}

impl Liquid {
    /// Note `k` (`n`) of line `line`'s bubble, its `bubbles` as the singing
    /// filled them, at `now` (the singing's time).
    pub fn of(bubbles: &Bubbles, line: usize, k: usize, n: &Note, now: f64) -> Liquid {
        let (level, full, jumped) = level(bubbles, k, (n.start, n.end), now);
        let pops_at = bubbles.notes[k].full_at.map(|f| f + SWEEP);
        Liquid {
            note: k,
            from: n.from,
            to: n.to,
            time: (n.start, n.end),
            height: n.height,
            level,
            full,
            jumped,
            pops_at,
            age: pops_at.filter(|&p| now >= p).map(|p| (now - p).min(BURST)),
            seed: seed(line, k),
        }
    }

    /// How much it strains ([`pressure`]).
    pub fn pressure(&self) -> f64 {
        pressure(self.level)
    }

    /// Whether it is mostly fully hit (else half).
    pub fn fully(&self) -> bool {
        self.full >= 0.5
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

/// Bubble `note` of line `line`'s own number, the same every frame: what
/// its wobble and splash are scattered by (24 bits, as the GPU takes it).
pub fn seed(line: usize, note: usize) -> u32 {
    ((line as u32).wrapping_mul(7919) ^ (note as u32).wrapping_mul(104_729)) & 0xFF_FFFF
}

/// Line `i`'s bubbles as `singing` filled them, at `clock` (the singing's
/// time: the song's as heard less how late the voice comes), into `l`:
/// how far each is filled, fully or half (in characters, from its start),
/// and those with liquid in them, bursting or burst.
pub fn fill(singing: &Singing, i: usize, clock: f64, l: &mut LyricLine) {
    let Some(notes) = singing.lines.get(i) else {
        return;
    };
    let bubbles = bubbles::line(singing, i, f64::INFINITY);
    for (k, n) in notes.notes.iter().enumerate() {
        let b = Liquid::of(&bubbles, i, k, n, clock);
        if b.level <= 0.0 && !b.begun() {
            continue;
        }
        let to = n.from + b.level as f32 * (n.to - n.from);
        l.hits.push((n.from, to, b.fully()));
        l.liquid.push(b);
    }
    // Side by side as one; what is full shows full over what is half.
    l.hits
        .sort_by(|a, b| a.0.total_cmp(&b.0).then(b.2.cmp(&a.2)));
    let mut merged: Vec<(f32, f32, bool)> = Vec::with_capacity(l.hits.len());
    for h in l.hits.drain(..) {
        match merged.last_mut() {
            Some(m) if m.2 == h.2 && h.0 <= m.1 + 0.05 => m.1 = m.1.max(h.1),
            Some(m) if m.2 && !h.2 && h.1 <= m.1 => {}
            _ => merged.push(h),
        }
    }
    l.hits = merged;
}

/// A note's mark in the terminal, lowest to highest: scan lines at five
/// heights of a cell.
const NOTE_GLYPHS: [&str; 5] = ["⎽", "⎼", "─", "⎻", "⎺"];

/// `line`'s notes in a terminal, in the row at (`x`, `y`) of `buf` under
/// its text (`chars`, the first `split` sung): under the letters sung on
/// them (each character's cells), at five heights; those sung already
/// brighter, filled gold as far as their liquid goes; those bursting or
/// burst, blank; the singer now (`↑` `↓` `●`).
pub fn notes_row(
    buf: &mut Buffer,
    line: &LyricLine,
    (x, y): (u16, u16),
    chars: &[char],
    split: usize,
) {
    let mut cell_of = Vec::with_capacity(chars.len() + 1);
    let mut at = 0u16;
    for c in chars {
        cell_of.push(at);
        at += width(&c.to_string()) as u16;
    }
    cell_of.push(at);
    let last = chars.len();
    // The cells under characters `from` to `to`.
    let cells = |from: f32, to: f32| {
        let a = (from.floor().max(0.0) as usize).min(last);
        let b = (to.ceil().max(0.0) as usize).clamp(a + 1, last.max(a + 1));
        (cell_of[a], cell_of[b.min(last)].max(cell_of[a] + 1))
    };
    for (k, note) in line.notes.iter().enumerate() {
        let level = (note.height * 2.0).round().clamp(-2.0, 2.0) as i32;
        let glyph = NOTE_GLYPHS[(level + 2) as usize];
        let (c0, c1) = cells(note.from, note.to);
        if line.liquid.iter().any(|b| b.note == k && b.begun()) {
            continue;
        }
        for c in c0..c1 {
            let ch = cell_of.iter().rposition(|&k| k <= c).unwrap_or(0);
            let mid = ch as f32 + 0.5;
            let color = match line.hits.iter().find(|h| mid >= h.0 && mid < h.1) {
                Some(&(_, _, true)) => HIT,
                Some(_) => HALF_HIT,
                None if ch < split => super::lanes::NOTE,
                None => NOTE_AHEAD,
            };
            buf[(x + c, y)].set_symbol(glyph).set_fg(color);
        }
    }
    if let Some((at, _, way)) = line.voice {
        let ch = (at.floor().max(0.0) as usize).min(last.saturating_sub(1));
        let glyph = match way {
            1 => "↑",
            -1 => "↓",
            _ => "●",
        };
        buf[(x + cell_of[ch], y)].set_symbol(glyph).set_fg(SINGER);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::bubbles::Bubble;

    /// A bubble fills to the share of it hit; full, its level sweeps up to
    /// full across its word (when it filled with it), the share fully hit
    /// growing with it; it strains only in the last stretch; it bursts once
    /// swept and stays burst (gone); one not full neither.
    #[test]
    fn a_bubble_fills_to_its_level_sweeps_full_then_bursts() {
        let bubbles = Bubbles {
            notes: vec![
                Bubble {
                    spans: vec![(0.0, 0.1, true), (0.2, 0.3, false)],
                    full_at: Some(0.3),
                    by_word: true,
                },
                Bubble {
                    full_at: Some(0.3),
                    by_word: true,
                    ..Bubble::default()
                },
                Bubble {
                    spans: vec![(1.0, 1.6, true)],
                    ..Bubble::default()
                },
            ],
            words: vec![
                bubbles::Word {
                    word: 0,
                    notes: 0..2,
                    time: (0.0, 1.0),
                    full_at: Some(0.3),
                },
                bubbles::Word {
                    word: 1,
                    notes: 2..3,
                    time: (1.0, 2.0),
                    full_at: None,
                },
            ],
        };
        let at = |k: usize, now: f64| {
            let note = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0)][k];
            level(&bubbles, k, note, now)
        };
        // Before its sweep: its share hit, half of that fully.
        let (l, f, j) = at(0, 0.29);
        assert!((l - 0.4).abs() < 1e-9 && (f - 0.5).abs() < 1e-9, "{l} {f}");
        assert!(j == 0.3, "{j}");
        // Half way through the sweep: the first note full, the second not
        // reached yet.
        let mid = 0.3 + SWEEP / 2.0;
        assert!((at(0, mid).0 - 1.0).abs() < 1e-9 && at(1, mid).0 < 1e-9);
        assert!((at(1, 0.3 + SWEEP * 0.75).0 - 0.5).abs() < 1e-9);
        assert!((at(1, 0.3 + SWEEP / 2.0 + 0.001).2 - mid).abs() < 1e-9);
        assert!(at(0, mid).1 > 0.79, "fully hit as swept");
        // Strains only in its last stretch.
        assert_eq!(pressure(0.6), 0.0);
        assert!(pressure(0.95) > 0.5 && pressure(1.0) == 1.0);
        // Not full: its level, never more.
        let (l, f, j) = at(2, 9.0);
        assert!((l - 0.6).abs() < 1e-9 && (f - 1.0).abs() < 1e-9 && j == f64::NEG_INFINITY);
        // Its burst: from once swept full, overfilling, then torn (not
        // drawn), then over, staying so; none for one not full.
        let note = crate::core::singing::Note {
            from: 0.0,
            to: 2.0,
            height: 0.0,
            start: 0.0,
            end: 0.5,
            cents: 6000.0,
            word: 0,
        };
        let b = |now: f64| Liquid::of(&bubbles, 0, 0, &note, now);
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
        let never = Liquid::of(&bubbles, 0, 2, &note, 9.0);
        assert!(never.pops_at.is_none() && !never.begun() && never.pressure() == 0.0);
    }
}
