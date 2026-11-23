//! The note bubbles on the GPU's stage ([`super::stage`]): a word's pill
//! filled as its bubbles are ([`crate::core::bubbles`],
//! [`crate::view::bubbles`]), popping while one of them bursts (the
//! stage's shader grows and lights it by the burst's age), and the
//! sparkles of those bursting, a [`Spray`] each for the GPU
//! ([`super::particles`]), made afresh each frame from the bubbles as the
//! picture's are: one per burst, none after a seek back before it. All of
//! it goes by the singing's clock.

use std::collections::HashMap;

use super::particles::{Kind, Spray};
use super::stage_gpu::Instance;
use crate::core::bubbles::{self, Bubbles};
use crate::core::singing::{Note, Singing};
use crate::view::bubbles::{burst, seed, shown, SWEEP};

/// Gold, as the stage's.
const GOLD: [f32; 3] = [1.0, 0.72, 0.2];

/// A bubble of a pill bursting: its index on its line, when its note is
/// sung (from, to), and when its burst started.
pub type Bursting = (usize, (f64, f64), f64);

/// A word's pill on the stage at a moment.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pill {
    /// Where it shows filled: from, to (seconds), whether fully.
    pub spans: Vec<(f64, f64, bool)>,
    /// When the burst of one of its bubbles bursting now started (the
    /// latest), if one is.
    pub burst: Option<f64>,
    /// Its bubbles bursting now, each throwing its sparkles.
    pub bursting: Vec<Bursting>,
}

/// Word `word`'s pill of a line with `notes`, its `bubbles` as the
/// singing filled them, at `clock` (the singing's time).
pub fn pill(notes: &[Note], bubbles: &Bubbles, word: usize, clock: f64) -> Pill {
    let mut out = Pill::default();
    let Some(w) = bubbles.word(word) else {
        return out;
    };
    for k in w.notes.clone() {
        let Some(n) = notes.get(k) else { continue };
        out.spans.extend(shown(bubbles, k, (n.start, n.end), clock));
        let full_at = bubbles.notes[k].full_at;
        if let (Some(f), Some(_)) = (full_at, burst(full_at, clock)) {
            let start = f + SWEEP;
            out.burst = Some(out.burst.map_or(start, |b: f64| b.max(start)));
            out.bursting.push((k, (n.start, n.end), start));
        }
    }
    out.spans.sort_by(|a, b| a.0.total_cmp(&b.0));
    out
}

/// The pills of the stage's words in a frame: each line's bubbles made
/// once.
pub struct Pills<'a> {
    singing: &'a Singing,
    clock: f64,
    lines: HashMap<usize, Bubbles>,
}

impl<'a> Pills<'a> {
    /// For `singing` at `clock` (the singing's time).
    pub fn new(singing: &'a Singing, clock: f64) -> Pills<'a> {
        Pills {
            singing,
            clock,
            lines: HashMap::new(),
        }
    }

    /// Line `l`'s word `word`'s pill ([`pill`]), with the game on (`game`;
    /// off, nothing filled).
    pub fn pill(&mut self, game: bool, l: usize, word: usize) -> Pill {
        let (Some(notes), true) = (self.singing.lines.get(l), game) else {
            return Pill::default();
        };
        let singing = self.singing;
        let b = self
            .lines
            .entry(l)
            .or_insert_with(|| bubbles::line(singing, l, f64::INFINITY));
        pill(&notes.notes, b, word, self.clock)
    }
}

/// A pill bursting on the stage, `px` high: its body, its edge and its
/// glow at their most (the shader dims them as the burst goes).
pub fn lit(px: f32) -> ([f32; 4], [f32; 4], f32) {
    ([1.0, 0.93, 0.6, 0.3], [1.0, 1.0, 1.0, 0.95], px * 1.2)
}

/// `pill` (an instance of the stage's, `px` high) bursting since `start`:
/// the shader grows it by its pop, its quad made room for it.
pub fn bursting(pill: &mut Instance, start: f64, px: f32) {
    let more = (pill.shape[2] + px) * 0.28 + px;
    pill.quad = [
        pill.quad[0] - more,
        pill.quad[1] - more,
        pill.quad[2] + more,
        pill.quad[3] + more,
    ];
    pill.uv = [start as f32, 1.0, 0.0, 0.0];
}

/// The sparkles of bubble `k` of line `l` on the stage, from `x` to `x1`
/// (pixels across at `clock`) at `y`, its pill `px` high, its burst from
/// `start` (the singing's time), the lane going across at `pps`.
#[allow(clippy::too_many_arguments)]
pub fn sparkles(
    l: usize,
    k: usize,
    (x0, x1, y): (f32, f32, f32),
    px: f32,
    start: f64,
    clock: f64,
    pps: f32,
) -> Spray {
    // Where it was as the burst started: it has gone by with the lane since.
    let back = pps * (clock - start) as f32;
    Spray {
        at: ((x0 + x1) / 2.0 + back, y),
        half: ((x1 - x0).abs() / 2.0, px / 2.0),
        start: start as f32,
        drift: -pps,
        // Larger than a pill: they fly off it, over the lane.
        size: px * 2.0,
        seed: seed(l, k),
        rgb: GOLD,
        kind: Kind::Sparkles,
    }
}

/// A word ending at `end` scored at (`x`, `y`) at `clock`, the stage's
/// pills `px` high, the lane going across at `pps`: its sparks (scattered
/// by its end); further as the multiplier goes `up`.
pub fn scored((x, y): (f32, f32), px: f32, clock: f64, pps: f32, up: bool, end: f64) -> Spray {
    Spray {
        at: (x, y),
        half: (0.0, 0.0),
        start: clock as f32,
        drift: -pps,
        size: px * 0.65,
        seed: (end * 1000.0) as u32,
        rgb: GOLD,
        kind: if up { Kind::ScoredUp } else { Kind::Scored },
    }
}

impl<'a> super::stage::Song<'a> {
    /// The song of `singing` (its `sung` lines, `length` long) with the
    /// game on and the voice taken as it comes (no delay): for a stage
    /// made outside the player.
    pub fn new(
        singing: &'a Singing,
        sung: Vec<&'a jodel_lyrics::Line>,
        length: Option<f64>,
    ) -> Self {
        super::stage::Song {
            singing,
            sung,
            game: true,
            length,
            late: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::singing::Level;
    use crate::view::bubbles::BURST;

    /// "la" a second on C4 then G4 (two notes), sung for its first 15 % on
    /// easy, heard `late` seconds late: frame by frame at 60 a second, as
    /// the stage goes by the singing's time (the song's less `late`). Each
    /// bubble bursts once (one start each, for its burst only), the pill
    /// pops for the burst, the sweep is seen from its start; sought back
    /// before a burst, it shows none, and on again, the same.
    #[test]
    fn the_stage_bursts_each_bubble_once_however_late_the_voice() {
        for late in [0.0, 0.264] {
            let mut m = jodel_rhythm::waveform::Melody {
                cents: vec![0; 200],
                sure: vec![255; 200],
            };
            for (i, c) in m.cents.iter_mut().enumerate() {
                *c = if i < 50 { 6000 } else { 6700 };
            }
            let lyrics = std::sync::Arc::new(jodel_lyrics::Lyrics {
                lines: vec![jodel_lyrics::Line {
                    words: vec![jodel_lyrics::Word {
                        start: Some(0.0),
                        end: Some(1.0),
                        confidence: 0.9,
                        ..jodel_lyrics::Word::new("lala")
                    }],
                    ..jodel_lyrics::Line::default()
                }],
                source: jodel_lyrics::Source::Unknown,
                timing: jodel_lyrics::Timing::default(),
            });
            let mut s = Singing::default();
            s.prepare(std::path::Path::new("a"), &lyrics, &m, 1, Level::Easy);
            let (mut starts, mut popped, mut swept) = (Vec::new(), Vec::new(), Vec::new());
            let mut heard = 0;
            // On to 1.2 s, back to 0.25 s (before the burst), on again.
            let frames = (0..72).chain(15..90);
            for f in frames {
                let now = f64::from(f) / 60.0;
                let clock = now - late;
                // The frames come `late` after the song is heard.
                if (f64::from(heard) / 100.0) > clock + 0.02 {
                    heard = (clock * 100.0).max(0.0).floor() as u32;
                }
                while f64::from(heard) / 100.0 <= clock {
                    let t = f64::from(heard) / 100.0;
                    s.hear(t, (t < 0.15).then_some(6000.0));
                    heard += 1;
                }
                let b = bubbles::line(&s, 0, f64::INFINITY);
                let p = pill(&s.lines[0].notes, &b, 0, clock);
                for &(k, _, start) in &p.bursting {
                    assert!((start..start + BURST).contains(&clock), "late {late}");
                    if !starts.contains(&(k, start.to_bits())) {
                        starts.push((k, start.to_bits()));
                    }
                }
                if p.burst.is_some() {
                    popped.push(clock);
                }
                swept.push(p.spans.iter().map(|s| s.1 - s.0).sum::<f64>());
            }
            assert_eq!(starts.len(), 2, "late {late}: one burst a bubble");
            assert!(!popped.is_empty());
            // The sweep seen part way: more than what was sung, less than
            // the whole word.
            assert!(swept.iter().any(|&f| f > 0.2 && f < 0.9), "late {late}");
        }
    }
}
