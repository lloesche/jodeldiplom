//! The word bubbles on the GPU's stage ([`super::stage`]): a word's bubble
//! is one tube through its notes ([`Snake`]), its liquid poured along it
//! note by note ([`Liquid`], as every frontend has it), its front a living
//! surface and the liquid shimmering, straining in the last stretch to
//! full, overfilling as it bursts, all of it at once, then not drawn (all
//! in the stage's shader, [`super::snake`]); its drops, bits of skin,
//! sheen and sparkles fly off all along it as particles (a [`Spray`] a
//! stretch of it, [`super::particles`]). Held full (its word won and still
//! sung) it pumps, thicker as it grows and glowing more, and bursts the
//! bigger for it: more sprays, flying further. A bubble never full keeps its
//! liquid at its level. Made afresh each frame from the bubbles as the
//! picture's are: one burst a bubble, none after a seek back before it
//! (the bubble then shows again as it is). All of it goes by the singing's
//! clock.

use std::collections::HashMap;

use super::particles::{Kind, Spray};
use super::snake::Snake;
use super::stage_gpu::{Instance, Pour, Skin, Strain, Tint};
use crate::core::bubbles::{self, Bubbles};
use crate::core::singing::Singing;
use crate::view::bubbles::Liquid;

/// Gold, as the stage's.
const GOLD: [f32; 3] = [1.0, 0.72, 0.2];
/// The least length of a stretch of a tube that bursts as one spray, in
/// its radii.
const STRETCH: f32 = 5.0;

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

/// Where a word's tube is on the stage: the tube, its ends across (its
/// round ends included), and where across a time of its singing is.
#[derive(Clone, Copy)]
pub struct Laid<'a> {
    /// The tube.
    pub snake: &'a Snake,
    /// Its left and right.
    pub ends: (f32, f32),
    /// Where across a time of the song is along it.
    pub across: &'a dyn Fn(f64) -> f32,
}

/// Where part `i` of bubble `b` is across its tube laid as `at`: from its
/// note's start to its end, the first from the tube's left, the last to
/// its right.
fn span(b: &Liquid, i: usize, at: &Laid) -> (f32, f32) {
    let p = &b.parts[i];
    let x0 = if i == 0 {
        at.ends.0
    } else {
        (at.across)(p.time.0)
    };
    let last = i + 1 == b.parts.len();
    let x1 = if last {
        at.ends.1
    } else {
        (at.across)(p.time.1)
    };
    (x0, x1.max(x0 + 1.0))
}

/// How far across bubble `b`, laid as `at`, its liquid reaches unbroken
/// from its left: its parts full one after another, up to the first that
/// is not, as far as that one is filled.
pub fn front(b: &Liquid, at: &Laid) -> f32 {
    let mut to = at.ends.0;
    for (i, p) in b.parts.iter().enumerate() {
        let (x0, x1) = span(b, i, at);
        to = x0 + (x1 - x0) * p.level.clamp(0.0, 1.0) as f32;
        if p.level < 1.0 {
            break;
        }
    }
    to
}

/// How much further and larger the bits of a bubble burst the biggest
/// fly, a share of any's ([`Liquid::size`]).
pub const BIGGER: f32 = 0.6;

/// Bubble `b`'s bursts for the GPU, laid as `at`, `px` high, at `clock`,
/// the lane going across at `pps`: while it bursts, one spray a stretch of
/// its tube (at least [`STRETCH`] radii long), from where that was as it
/// burst; held full long ([`Liquid::size`]), as large as it grew and up to
/// two a stretch, flying [`BIGGER`] further.
pub fn sprays(b: &Liquid, at: &Laid, px: f32, (clock, pps): (f64, f32)) -> Vec<Spray> {
    let Some(start) = b.pops_at.filter(|_| b.bursting().is_some()) else {
        return Vec::new();
    };
    let r = at.snake.radius;
    let (x0, x1) = at.ends;
    let n = (((x1 - x0) / (STRETCH * r)).floor() as usize).clamp(1, 5);
    let r = r * b.pump() as f32;
    let wide = (x1 - x0) / n as f32;
    let size = b.size() as f32;
    let each = 1 + usize::from(size >= 0.5);
    // Where it was as it burst: it has gone by with the lane since.
    let back = pps * (clock - start) as f32;
    (0..n * each)
        .map(|k| {
            let x = x0 + wide * ((k / each) as f32 + 0.5);
            Spray {
                at: (x + back, at.snake.y_at(x)),
                half: (wide / 2.0, r),
                start: start as f32,
                drift: -pps,
                // Larger than a tube: they fly off it, over the lane.
                size: px * 2.0 * (1.0 + BIGGER * size),
                seed: b.seed.wrapping_add(k as u32 * 7919) & 0xFF_FFFF,
                rgb: GOLD,
                kind: Kind::Burst,
            }
        })
        .collect()
}

/// Draws the tube laid as `at` with its bubble `liquid` (none: the game
/// off, the tube empty) into `out` (its points into `paths`) in its `skin`,
/// `px` high: straining and overfilling as it does, gone once torn, each
/// part's liquid in `tint`.
pub fn draw(
    liquid: Option<&Liquid>,
    (out, paths): (&mut Vec<Instance>, &mut Vec<[f32; 4]>),
    at: &Laid,
    (skin, tint): (&Skin, &Tint),
    px: f32,
) {
    let Some(b) = liquid else {
        let first = at.snake.put(paths);
        out.push(at.snake.skin(first, skin, &Strain::default()));
        return;
    };
    if b.torn() {
        return;
    }
    // Held full, it pumps: thicker as it grows, glowing more.
    let pumped;
    let snake = if b.pump() > 1.0 {
        pumped = Snake {
            radius: at.snake.radius * b.pump() as f32,
            ..at.snake.clone()
        };
        &pumped
    } else {
        at.snake
    };
    let first = snake.put(paths);
    let strain = strain(b);
    let skin = if strain.popping.is_some() {
        lit(px, b.size() as f32)
    } else {
        pumping(skin, b.glow() as f32, px)
    };
    out.push(snake.skin(first, &skin, &strain));
    let tint = Tint {
        glow: tint.glow * (1.0 + b.glow() as f32),
        ..*tint
    };
    for (i, p) in b.parts.iter().enumerate() {
        let (x0, x1) = span(b, i, at);
        let pour = Pour {
            span: (x0, x0 + (x1 - x0) * p.level as f32),
            front: p.level < 1.0,
            jumped: p.jumped.max(-1e6) as f32,
            tint,
        };
        out.extend(snake.liquid(first, &pour, &strain));
    }
}

/// A bubble's `skin` held full, `glow` of the way to its most (0 to 1),
/// `px` high: its edge lighter and its glow reaching further.
fn pumping(skin: &Skin, glow: f32, px: f32) -> Skin {
    let edge = skin.edge;
    let light = |c: f32| c + (1.0 - c) * 0.5 * glow;
    Skin {
        edge: [light(edge[0]), light(edge[1]), light(edge[2]), edge[3]],
        glow: skin.glow + px * 0.4 * glow,
        ..*skin
    }
}

/// The bubbles of the stage's words in a frame: each line's bubbles made
/// once.
pub struct Tubes<'a> {
    singing: &'a Singing,
    clock: f64,
    lines: HashMap<usize, Bubbles>,
}

impl<'a> Tubes<'a> {
    /// For `singing` at `clock` (the singing's time).
    pub fn new(singing: &'a Singing, clock: f64) -> Tubes<'a> {
        Tubes {
            singing,
            clock,
            lines: HashMap::new(),
        }
    }

    /// Line `l`'s word `word`'s bubble as the singing filled it, with the
    /// game on (`game`; off, none).
    pub fn liquid(&mut self, game: bool, l: usize, word: usize) -> Option<Liquid> {
        let notes = self.singing.notes.lines.get(l).filter(|_| game)?;
        let singing = self.singing;
        let b = self
            .lines
            .entry(l)
            .or_insert_with(|| bubbles::line(singing, l, f64::INFINITY));
        let b = b.word(word)?;
        Some(Liquid::of(b, l, &notes.notes, self.clock))
    }
}

/// A bubble overfilling on the stage, `px` high, bursting `size` big (0
/// as any to 1, [`Liquid::size`]): its skin thin (its liquid shows
/// through), its edge white, glowing ([`BIGGER`] further the biggest).
pub fn lit(px: f32, size: f32) -> Skin {
    Skin {
        body: [1.0, 0.95, 0.8, 0.2],
        edge: [1.0, 1.0, 1.0, 0.95],
        glow: px * 0.9 * (1.0 + BIGGER * size),
        outline: (px * 0.05).max(1.2),
    }
}

/// A word ending at `end` scored at (`x`, `y`) at `clock`, the stage's
/// tubes `px` high, the lane going across at `pps`: its sparks (scattered
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
            demo: false,
            heard: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::bubbles::{BURST, SWELL};

    /// "lala" a second, its first half on C4, its second on G4 (two notes,
    /// one bubble), sung as `sung` says at each moment (cents, or none),
    /// heard `late` seconds late: frame by frame at 60 a second, as the
    /// stage goes by the singing's time (the song's less `late`), on to
    /// 1.7 s (past its end, where it bursts), back to 0.83 s, on again.
    /// Each frame's bubble.
    fn frames(sung: impl Fn(f64) -> Option<f32>, late: f64) -> Vec<(f64, Liquid)> {
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
        for f in (0..102).chain(50..120) {
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
            let b = b.word(0).expect("its bubble");
            out.push((clock, Liquid::of(b, 0, &s.notes.lines[0].notes, clock)));
        }
        out
    }

    /// However late the voice: the word counts and its one bubble (over
    /// both notes) sweeps full and bursts once, torn after its
    /// overfilling; sought back before, it shows again, and bursts again
    /// on the way.
    #[test]
    fn a_word_s_bubble_bursts_once_however_late_the_voice() {
        for late in [0.0, 0.264] {
            let frames = frames(|t| (t < 0.15).then_some(6000.0), late);
            let liquid = |f: &(f64, Liquid)| f.1.clone();
            assert!(frames.iter().all(|f| liquid(f).parts.len() == 2));
            let mut starts: Vec<f64> = Vec::new();
            for f in &frames {
                let b = liquid(f);
                if b.bursting().is_some() {
                    let start = b.pops_at.unwrap();
                    assert!((start..start + BURST).contains(&f.0));
                    assert_eq!(b.swelling().is_some(), f.0 - start < SWELL);
                    if !starts.contains(&start) {
                        starts.push(start);
                    }
                }
            }
            assert_eq!(starts.len(), 1, "late {late}: it bursts once");
            let start = starts[0];
            for f in &frames {
                assert_eq!(liquid(f).torn(), f.0 - start >= SWELL, "late {late}");
            }
            let first = frames.iter().position(|f| liquid(f).torn()).unwrap();
            let back = frames[first..].iter().find(|f| f.0 < start).unwrap();
            assert!(!liquid(back).begun());
            assert_eq!(liquid(&frames[0]).pressure(), 0.0);
            // The sweep seen part way: more than what was sung, less than
            // the whole word.
            let filled = frames.iter().map(|(_, p)| p.filled());
            assert!(filled.clone().any(|f| f > 0.2 && f < 0.9), "late {late}");
        }
    }

    /// The word sung 5 % (under the tenth it asks, its first part a tenth
    /// full): no burst, no strain, its liquid kept.
    #[test]
    fn a_bubble_partly_filled_never_pops() {
        let frames = frames(|t| (t < 0.05).then_some(6000.0), 0.0);
        let mut filled = 0.0f64;
        for (_, b) in &frames {
            assert!(!b.begun() && b.pops_at.is_none(), "{b:?}");
            assert_eq!(b.pressure(), 0.0, "it never strains");
            filled = filled.max(b.filled());
        }
        assert!(filled > 0.03 && filled < 0.08, "{filled}");
        let last = &frames.last().unwrap().1;
        assert!(last.parts[0].level > 0.08, "its liquid kept");
    }

    /// A bubble bursting throws a spray a stretch of its tube, each where
    /// that stretch was, up and down with it.
    #[test]
    fn a_tube_bursts_all_along_itself() {
        let frames = frames(|t| (t < 0.15).then_some(6000.0), 0.0);
        let (clock, b) = frames.iter().find(|f| f.1.bursting().is_some()).unwrap();
        let snake = Snake::through(
            &[(0.0, 100.0, 80.0), (100.0, 200.0, 40.0)],
            (0.0, 200.0),
            10.0,
        );
        let across = |s: f64| (s * 200.0) as f32;
        let at = Laid {
            snake: &snake,
            ends: (0.0, 200.0),
            across: &across,
        };
        let sprays = sprays(b, &at, 20.0, (*clock, 0.0));
        assert_eq!(sprays.len(), 4);
        assert!(sprays[0].at.1 > 70.0 && sprays[3].at.1 < 50.0, "{sprays:?}");
        assert!(sprays.windows(2).all(|w| w[0].seed != w[1].seed));
    }

    /// A word held full long pumps its tube thicker and glowing more, and
    /// bursts the bigger: twice the sprays, as large as it grew, flying
    /// further, its overfilling glowing further; one held a moment bursts
    /// as any.
    #[test]
    fn a_tube_held_full_long_pumps_and_bursts_bigger() {
        let notes = [crate::core::singing::Note {
            from: 0.0,
            to: 4.0,
            height: 0.0,
            start: 0.0,
            end: 2.6,
            cents: 6000.0,
            word: 0,
        }];
        let held = |end: f64| crate::core::bubbles::Bubble {
            word: 0,
            notes: 0..1,
            time: (0.0, end),
            spans: vec![(0.0, 0.1)],
            full_at: Some(0.1),
        };
        let snake = Snake::through(&[(0.0, 200.0, 50.0)], (0.0, 200.0), 10.0);
        let across = |s: f64| (s * 80.0) as f32;
        let at = Laid {
            snake: &snake,
            ends: (0.0, 200.0),
            across: &across,
        };
        let burst = |end: f64| {
            let b = Liquid::of(&held(end), 0, &notes, end + 0.1);
            sprays(&b, &at, 20.0, (end + 0.1, 0.0))
        };
        let (short, long) = (burst(0.6), burst(2.6));
        assert_eq!((short.len(), long.len()), (4, 8));
        assert!(short.iter().all(|s| s.size == 40.0 && s.half.1 > 10.0));
        assert!(long.iter().all(|s| s.size > 60.0 && s.half.1 > 11.0));
        // Pumping: drawn thicker and glowing more than as it was swept.
        let drawn = |now: f64| {
            let b = Liquid::of(&held(2.6), 0, &notes, now);
            let (mut out, mut paths) = (Vec::new(), Vec::new());
            let skin = Skin {
                body: [0.0; 4],
                edge: [1.0, 0.7, 0.2, 1.0],
                glow: 6.0,
                outline: 1.2,
            };
            let tint = Tint {
                top: [1.0; 4],
                bottom: [1.0; 4],
                glow: 7.0,
            };
            draw(Some(&b), (&mut out, &mut paths), &at, (&skin, &tint), 20.0);
            (out[0].shape[2], out[0].params[1])
        };
        let (swept, pumped) = (drawn(0.31), drawn(2.0));
        assert!(pumped.0 > swept.0 * 1.08 && pumped.1 > swept.1 + 4.0);
        assert!(lit(20.0, 1.0).glow > lit(20.0, 0.0).glow * 1.5);
    }
}
