//! The note bubbles on the GPU's stage ([`super::stage`]): a word's pill
//! is its notes' bubbles in a row along it ([`Liquid`], one a note, as
//! every frontend has them), each holding its liquid to its level, its
//! front a living surface and the liquid shimmering, straining in the last
//! stretch to full, overfilling as it bursts, the moment it is full, on
//! its own, then not drawn (all in the stage's shader:
//! [`Instance::bubble`], [`Instance::liquid`]); its drops, bits of skin,
//! sheen and sparkles fly off from its place as particles (a [`Spray`]
//! each, [`super::particles`]). The word's other bubbles stay with their
//! liquid until they fill and burst in turn; a bubble never full keeps its
//! liquid at its level. Made afresh each frame from the bubbles as the
//! picture's are: one burst a bubble, none after a seek back before it
//! (the bubble then shows again as it is). All of it goes by the singing's
//! clock.

use std::collections::HashMap;

use super::particles::{Kind, Spray};
use super::stage_gpu::{Instance, Pour, Skin, Strain, Tint};
use crate::core::bubbles::{self, Bubbles};
use crate::core::singing::Singing;
use crate::view::bubbles::Liquid;

/// Gold, as the stage's.
const GOLD: [f32; 3] = [1.0, 0.72, 0.2];

/// A bubble's strain on the GPU, as its [`Liquid`] has it.
pub fn strain(b: &Liquid) -> Strain {
    Strain {
        pressure: b.pressure() as f32,
        phase: super::liquid::phase(b.seed),
        popping: b
            .pops_at
            .filter(|_| b.swelling().is_some())
            .map(|p| p as f32),
    }
}

/// A word's pill on the stage at a moment: its bubbles in a row (none:
/// the pill whole, as without the game).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pill {
    /// Its bubbles, one a note, in order.
    pub bubbles: Vec<Liquid>,
}

impl Pill {
    /// How much of its word is filled, in seconds of it.
    pub fn filled(&self) -> f64 {
        let each = |b: &Liquid| b.level * (b.time.1 - b.time.0);
        self.bubbles.iter().map(each).sum()
    }

    /// Where its bubble `i` is across its pill from `c - half` to
    /// `c + half` (pixels; `across` giving where on the lane a time of the
    /// song is), `px` high: a hair apart from the next.
    fn span(
        &self,
        i: usize,
        (c, half): (f32, f32),
        across: &dyn Fn(f64) -> f32,
        px: f32,
    ) -> (f32, f32) {
        let gap = px * 0.04;
        let b = &self.bubbles[i];
        let x0 = if i == 0 {
            c - half
        } else {
            across(b.time.0) + gap
        };
        let last = i + 1 == self.bubbles.len();
        let x1 = if last {
            c + half
        } else {
            across(b.time.1) - gap
        };
        (x0, x1.max(x0 + 1.0))
    }

    /// Its bursts for the GPU, of line `l`: each bubble bursting, from its
    /// place on the pill at `c`, `half`, `px` high (`across` giving where
    /// on the lane a time of the song is), at `clock`, the lane going
    /// across at `pps`.
    pub fn sprays(
        &self,
        across: &dyn Fn(f64) -> f32,
        (c, half, px): ((f32, f32), (f32, f32), f32),
        (clock, pps): (f64, f32),
    ) -> Vec<Spray> {
        let bursting = self.bubbles.iter().enumerate();
        let bursting = bursting.filter(|(_, b)| b.bursting().is_some());
        bursting
            .filter_map(|(i, b)| {
                let start = b.pops_at?;
                let (x0, x1) = self.span(i, (c.0, half.0), across, px);
                // Where it was as it burst: it has gone by with the lane since.
                let back = pps * (clock - start) as f32;
                Some(Spray {
                    at: ((x0 + x1) / 2.0 + back, c.1),
                    half: ((x1 - x0) / 2.0, half.1),
                    start: start as f32,
                    drift: -pps,
                    // Larger than a pill: they fly off it, over the lane.
                    size: px * 2.0,
                    seed: b.seed,
                    rgb: GOLD,
                    kind: Kind::Burst,
                })
            })
            .collect()
    }

    /// Draws the pill at `c`, `half` (`px` high) into `out` in its `skin`:
    /// each bubble not torn, straining and overfilling as it does, and its
    /// liquid in `tint`; without bubbles, the pill whole. How much of
    /// the word is filled, in seconds of it.
    pub fn draw(
        &self,
        out: &mut Vec<Instance>,
        (c, half): ((f32, f32), (f32, f32)),
        (skin, tint): (&Skin, &Tint),
        across: &dyn Fn(f64) -> f32,
        px: f32,
    ) -> f64 {
        if self.bubbles.is_empty() {
            out.push(Instance::bubble(c, half, 0.0, skin, &Strain::default()));
            return 0.0;
        }
        for (i, b) in self.bubbles.iter().enumerate() {
            if b.torn() {
                continue;
            }
            let (x0, x1) = self.span(i, (c.0, half.0), across, px);
            let (bc, bh) = (((x0 + x1) / 2.0, c.1), ((x1 - x0) / 2.0, half.1));
            let strain = strain(b);
            let skin = if strain.popping.is_some() {
                lit(px)
            } else {
                *skin
            };
            out.push(Instance::bubble(bc, bh, 0.0, &skin, &strain));
            let pour = Pour {
                span: (x0, x0 + (x1 - x0) * b.level as f32),
                front: b.level < 1.0,
                jumped: b.jumped.max(-1e6) as f32,
                tint: *tint,
            };
            out.extend(Instance::liquid(bc, bh, 0.0, &pour, &strain));
        }
        self.filled()
    }
}

/// Word `word`'s pill of line `l`, its `bubbles` as the singing filled
/// them (its notes `notes`), at `clock` (the singing's time).
pub fn pill(
    notes: &[crate::core::singing::Note],
    bubbles: &Bubbles,
    (l, word): (usize, usize),
    clock: f64,
) -> Pill {
    let Some(w) = bubbles.word(word) else {
        return Pill::default();
    };
    let each = w.notes.clone().filter_map(|k| Some((k, notes.get(k)?)));
    Pill {
        bubbles: each
            .map(|(k, n)| Liquid::of(bubbles, l, k, n, clock))
            .collect(),
    }
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
    /// off, the pill whole).
    pub fn pill(&mut self, game: bool, l: usize, word: usize) -> Pill {
        let (Some(notes), true) = (self.singing.notes.lines.get(l), game) else {
            return Pill::default();
        };
        let singing = self.singing;
        let b = self
            .lines
            .entry(l)
            .or_insert_with(|| bubbles::line(singing, l, f64::INFINITY));
        pill(&notes.notes, b, (l, word), self.clock)
    }
}

/// A bubble overfilling on the stage, `px` high: its skin thin (its liquid
/// shows through), its edge white, glowing.
pub fn lit(px: f32) -> Skin {
    Skin {
        body: [1.0, 0.95, 0.8, 0.2],
        edge: [1.0, 1.0, 1.0, 0.95],
        glow: px * 0.9,
        outline: (px * 0.05).max(1.2),
    }
}

/// A word ending at `end` scored at (`x`, `y`) at `clock`, the stage's
/// pills `px` high, the lane going across at `pps`: its sparks (scattered
/// by its end).
pub fn scored((x, y): (f32, f32), px: f32, clock: f64, pps: f32, end: f64) -> Spray {
    Spray {
        at: (x, y),
        half: (0.0, 0.0),
        start: clock as f32,
        drift: -pps,
        size: px * 0.65,
        seed: (end * 1000.0) as u32,
        rgb: GOLD,
        kind: Kind::Scored,
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
    use crate::view::bubbles::{BURST, SWELL};

    /// "lala" a second, its first half on C4, its second on G4 (two notes,
    /// two bubbles), sung as `sung` says at each moment (cents, or none),
    /// heard `late` seconds late: frame by frame at 60 a second, as the
    /// stage goes by the singing's time (the song's less `late`), on to
    /// 1.2 s, back to 0.25 s, on again. Each frame's pill.
    fn frames(sung: impl Fn(f64) -> Option<f32>, late: f64) -> Vec<(f64, Pill)> {
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
        s.prepare(
            std::path::Path::new("a"),
            &lyrics,
            &crate::core::singing::melodic(&m),
        );
        let mut out = Vec::new();
        let mut heard = 0;
        for f in (0..72).chain(15..90) {
            let now = f64::from(f) / 60.0;
            let clock = now - late;
            // The frames come `late` after the song is heard.
            if (f64::from(heard) / 100.0) > clock + 0.02 {
                heard = (clock * 100.0).max(0.0).floor() as u32;
            }
            while f64::from(heard) / 100.0 <= clock {
                let t = f64::from(heard) / 100.0;
                s.hear(t, sung(t));
                heard += 1;
            }
            let b = bubbles::line(&s, 0, f64::INFINITY);
            out.push((clock, pill(&s.notes.lines[0].notes, &b, (0, 0), clock)));
        }
        out
    }

    /// The times bubble `i` burst at in `frames`, each once.
    fn bursts(frames: &[(f64, Pill)], i: usize) -> Vec<f64> {
        let mut out: Vec<f64> = Vec::new();
        for (clock, p) in frames {
            let b = &p.bubbles[i];
            if b.bursting().is_some() {
                let start = b.pops_at.unwrap();
                assert!((start..start + BURST).contains(clock));
                assert_eq!(b.swelling().is_some(), *clock - start < SWELL);
                if !out.contains(&start) {
                    out.push(start);
                }
            }
        }
        out
    }

    /// However late the voice: the word counts and both its
    /// bubbles sweep full, each bursting once on its own, torn after its
    /// overfilling; sought back before, each shows again, and bursts again
    /// on the way.
    #[test]
    fn each_bubble_of_a_word_bursts_once_however_late_the_voice() {
        for late in [0.0, 0.264] {
            let frames = frames(|t| (t < 0.15).then_some(6000.0), late);
            assert!(frames.iter().all(|f| f.1.bubbles.len() == 2));
            for i in 0..2 {
                let starts = bursts(&frames, i);
                assert_eq!(starts.len(), 1, "late {late}: bubble {i} bursts once");
                let start = starts[0];
                for (clock, p) in &frames {
                    assert_eq!(p.bubbles[i].torn(), *clock - start >= SWELL, "late {late}");
                }
                let first = frames.iter().position(|f| f.1.bubbles[i].torn()).unwrap();
                let back = frames[first..].iter().find(|f| f.0 < start).unwrap();
                assert!(!back.1.bubbles[i].begun());
            }
            assert_eq!(frames[0].1.bubbles[0].pressure(), 0.0);
            // The sweep seen part way: more than what was sung, less than
            // the whole word.
            let filled = frames.iter().map(|(_, p)| p.filled());
            assert!(filled.clone().any(|f| f > 0.2 && f < 0.9), "late {late}");
        }
    }

    /// The word sung 5 % (under the tenth it asks, its first bubble a tenth
    /// full): no burst, no strain, its liquid kept, both drawn all the way on.
    #[test]
    fn a_pill_partly_filled_never_pops() {
        let frames = frames(|t| (t < 0.05).then_some(6000.0), 0.0);
        let mut filled = 0.0f64;
        for (_, p) in &frames {
            for b in &p.bubbles {
                assert!(!b.begun() && b.pops_at.is_none(), "{b:?}");
                assert_eq!(b.pressure(), 0.0, "it never strains");
            }
            filled = filled.max(p.filled());
        }
        assert!(filled > 0.03 && filled < 0.08, "{filled}");
        assert!(
            frames.last().unwrap().1.bubbles[0].level > 0.08,
            "its liquid kept"
        );
    }
}
