//! The note bubbles of the line being sung as the frontends show them
//! ([`crate::core::bubbles`]): a bubble full sweeps to full ([`SWEEP`],
//! across its word where it filled with it, else across itself), then
//! bursts ([`BURST`]); their fill in characters of the line, those
//! bursting, and the notes' row in a terminal, the burst's flourish around
//! a bubble there (✦ ✧ ·, fading).
//!
//! The time it goes by is the singing's own: the song's time as heard less
//! how late the voice comes (the frames of a moment come that much after
//! it is heard), so the sweep and the burst are seen whole, that much
//! later; a paused song holds them. Nothing is kept between frames.

use ratatui::buffer::Buffer;
use ratatui::style::{Color, Modifier};

use super::karaoke::{LyricLine, HALF_HIT, HIT, NOTE_AHEAD, SINGER};
use crate::core::bubbles::{self, Bubble, Bubbles};
use crate::core::singing::{Note, Singing};
use crate::core::text::width;

/// How long a bubble full takes to sweep to full, in seconds of the song.
pub const SWEEP: f64 = 0.2;
/// How long it bursts after that.
pub const BURST: f64 = 0.5;
/// How long of a burst its bubble pops (larger, brighter), a share of it.
pub const POP: f32 = 0.35;

/// How far the sweep of a bubble full at `full_at` is at `now`, 0 to 1 (0:
/// not full).
pub fn swept(full_at: Option<f64>, now: f64) -> f64 {
    full_at.map_or(0.0, |f| ((now - f) / SWEEP).clamp(0.0, 1.0))
}

/// How far through its burst a bubble full at `full_at` is at `now`, 0 to
/// 1; none before or after the burst.
pub fn burst(full_at: Option<f64>, now: f64) -> Option<f64> {
    let age = (now - full_at? - SWEEP) / BURST;
    (0.0..1.0).contains(&age).then_some(age)
}

/// Where bubble `k` of `bubbles` (its note sung over `note`: from, to)
/// shows filled at `now`: its spans, and the sweep (fully hit) as far as
/// it has got, across its word where it filled with it.
pub fn shown(bubbles: &Bubbles, k: usize, note: (f64, f64), now: f64) -> Vec<(f64, f64, bool)> {
    let b: &Bubble = &bubbles.notes[k];
    let mut out = b.spans.clone();
    let p = swept(b.full_at, now);
    let across = match bubbles.words.iter().find(|w| w.notes.contains(&k)) {
        Some(w) if b.by_word => w.time,
        _ => note,
    };
    let (from, to) = (
        across.0.max(note.0),
        (across.0 + p * (across.1 - across.0)).min(note.1),
    );
    if p > 0.0 && to > from {
        out.push((from, to, true));
    }
    out
}

/// A bubble bursting: under which characters (from, to), how high (as
/// the notes' heights), how far through its burst (0 to 1) and when it
/// started (the singing's time), and a number of its own (the same each
/// frame) its sparkles are scattered by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Burst {
    /// Where it starts, in characters of the line.
    pub from: f32,
    /// Where it ends.
    pub to: f32,
    /// How high: -1 to 1.
    pub height: f32,
    /// How far through its burst, 0 to 1.
    pub age: f32,
    /// When it started, in seconds of the singing's time.
    pub start: f64,
    /// Its own number.
    pub seed: u32,
}

/// Bubble `note` of line `line`'s own number, the same every frame: what
/// its sparkles are scattered by (24 bits, as the GPU takes it).
pub fn seed(line: usize, note: usize) -> u32 {
    ((line as u32).wrapping_mul(7919) ^ (note as u32).wrapping_mul(104_729)) & 0xFF_FFFF
}

/// Where on note `n`'s characters `t` seconds into the song is.
fn char_of(n: &Note, t: f64) -> f32 {
    let share = ((t - n.start) / (n.end - n.start).max(1e-3)).clamp(0.0, 1.0) as f32;
    n.from + share * (n.to - n.from)
}

/// Line `i`'s bubbles as `singing` filled them, at `clock` (the singing's
/// time: the song's as heard less how late the voice comes), into `l`:
/// where they show filled, fully or half (in characters), and those
/// bursting.
pub fn fill(singing: &Singing, i: usize, clock: f64, l: &mut LyricLine) {
    let Some(notes) = singing.lines.get(i) else {
        return;
    };
    let bubbles = bubbles::line(singing, i, f64::INFINITY);
    for (k, n) in notes.notes.iter().enumerate() {
        for (a, z, full) in shown(&bubbles, k, (n.start, n.end), clock) {
            l.hits.push((char_of(n, a), char_of(n, z), full));
        }
        let full_at = bubbles.notes[k].full_at;
        if let (Some(age), Some(f)) = (burst(full_at, clock), full_at) {
            l.bursts.push(Burst {
                from: n.from,
                to: n.to,
                height: n.height,
                age: age as f32,
                start: f + SWEEP,
                seed: seed(i, k),
            });
        }
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
/// brighter, those hit gold; those bursting brighter still, a flourish
/// either side; the singer now (`↑` `↓` `●`).
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
    for note in &line.notes {
        let level = (note.height * 2.0).round().clamp(-2.0, 2.0) as i32;
        let glyph = NOTE_GLYPHS[(level + 2) as usize];
        let (c0, c1) = cells(note.from, note.to);
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
    for b in &line.bursts {
        let (c0, c1) = cells(b.from, b.to);
        flourish_cells(buf, (x + c0, x + c1), y, b.age);
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

/// The glyphs of a flourish, in turn as it fades.
pub const FLOURISH: [&str; 3] = ["✦", "✧", "·"];

/// The flourish's glyph and colour at `age` (0 to 1 through the burst):
/// ✦ white, then ✧ gold, then · fading.
pub fn flourish_at(age: f32) -> (&'static str, Color) {
    let k = ((age.clamp(0.0, 0.999) * FLOURISH.len() as f32) as usize).min(FLOURISH.len() - 1);
    let fade = 1.0 - age.clamp(0.0, 1.0);
    let Color::Rgb(r, g, b) = HIT else {
        return (FLOURISH[k], HIT);
    };
    // From white to gold, then darker.
    let white = (1.0 - age * 2.0).clamp(0.0, 1.0);
    let mix = |c: u8| {
        let lit = f32::from(c) + (255.0 - f32::from(c)) * white;
        (lit * (0.35 + 0.65 * fade)).round() as u8
    };
    (FLOURISH[k], Color::Rgb(mix(r), mix(g), mix(b)))
}

/// A burst in a terminal over a bubble whose cells are `x0..x1` in row `y`
/// of `buf`: the bubble brighter while it pops ([`POP`]), and a flourish
/// just either side of it (where those cells are blank).
pub fn flourish_cells(buf: &mut Buffer, (x0, x1): (u16, u16), y: u16, age: f32) {
    let area = buf.area;
    if y < area.y || y >= area.bottom() {
        return;
    }
    if age < POP {
        for x in x0..x1.min(area.right()) {
            buf[(x, y)]
                .set_fg(Color::Rgb(255, 246, 214))
                .set_style(Modifier::BOLD);
        }
    }
    let (glyph, color) = flourish_at(age);
    for x in [x0.checked_sub(1), Some(x1)].into_iter().flatten() {
        if x < area.x || x >= area.right() {
            continue;
        }
        let cell = &mut buf[(x, y)];
        if cell.symbol().trim().is_empty() || FLOURISH.contains(&cell.symbol()) {
            cell.set_symbol(glyph).set_fg(color);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_flourish_turns_from_a_white_star_to_a_fading_dot() {
        let (a, ca) = flourish_at(0.05);
        let (b, _) = flourish_at(0.5);
        let (c, cc) = flourish_at(0.95);
        assert_eq!((a, b, c), ("✦", "✧", "·"));
        let lum = |c: Color| match c {
            Color::Rgb(r, g, b) => u32::from(r) + u32::from(g) + u32::from(b),
            _ => 0,
        };
        assert!(lum(ca) > lum(cc), "{ca:?} {cc:?}");
        for g in FLOURISH {
            assert_eq!(width(g), 1, "{g}");
        }
    }

    /// A bubble full sweeps to full across its word (when it filled with
    /// it), then bursts for [`BURST`]; one not full neither.
    #[test]
    fn a_bubble_full_sweeps_then_bursts() {
        let bubbles = Bubbles {
            notes: vec![
                Bubble {
                    spans: vec![(0.0, 0.1, true)],
                    full_at: Some(0.1),
                    by_word: true,
                },
                Bubble {
                    full_at: Some(0.1),
                    by_word: true,
                    ..Bubble::default()
                },
                Bubble::default(),
            ],
            words: vec![
                bubbles::Word {
                    word: 0,
                    notes: 0..2,
                    time: (0.0, 1.0),
                    full_at: Some(0.1),
                },
                bubbles::Word {
                    word: 1,
                    notes: 2..3,
                    time: (1.0, 2.0),
                    full_at: None,
                },
            ],
        };
        // Half way through the sweep: the word's first half swept, its
        // second note not yet.
        let mid = 0.1 + SWEEP / 2.0;
        assert!(shown(&bubbles, 0, (0.0, 0.5), mid).contains(&(0.0, 0.5, true)));
        assert_eq!(shown(&bubbles, 1, (0.5, 1.0), mid), vec![]);
        assert!(shown(&bubbles, 1, (0.5, 1.0), 0.5).contains(&(0.5, 1.0, true)));
        assert_eq!(burst(Some(0.1), 0.1 + SWEEP - 0.01), None);
        assert!(burst(Some(0.1), 0.1 + SWEEP + 0.1).is_some());
        assert_eq!(burst(Some(0.1), 0.1 + SWEEP + BURST + 0.01), None);
        assert_eq!(shown(&bubbles, 2, (1.0, 2.0), 9.0), vec![]);
        assert_eq!(burst(None, 9.0), None);
    }
}
