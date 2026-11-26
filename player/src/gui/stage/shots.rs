//! The rescuing shots on the stage ([`crate::core::singing::Rescue`]): a
//! glowing star flying from the bubble that fired it to the word missed
//! before it, in an arc, a tail of light behind it and sparkles off it;
//! where it lands, a flash and a burst of sparks (the word then fills and
//! bursts as any bubble, [`super::super::pop`]). A word missed long gone
//! off the lane's left: the shot flies off the edge after it. Both words
//! go by with the lane, the shot's way with them. Made afresh each frame
//! from the singing's rescues, by the singing's clock.

use super::super::particles::{Kind, Spray};
use super::super::stage_gpu::{Instance, Shape};
use super::words::Words;
use super::{disc, rgba, segment, Lane, Song, GOLD, GOLD_TOP};
use crate::core::bubbles::FLIGHT;
use crate::core::singing::Rescue;

/// How long after it lands a shot's flash shows, in seconds.
const FLASH: f64 = 0.18;
/// The pieces of a shot's tail.
const TAIL: usize = 7;

/// What a frame of shots draws: over the stage, as light, and the bursts
/// where they land.
#[derive(Default)]
pub(super) struct Shots {
    pub out: Vec<Instance>,
    pub lights: Vec<Instance>,
    pub sprays: Vec<Spray>,
}

/// Where line `l`'s word `w` is on the lane: its tube's middle where it is
/// laid, else where its sung time and height put it (off the lane, gone
/// by or to come).
fn place(song: &Song, lane: &Lane, words: &Words, (l, w): (usize, usize)) -> Option<(f32, f32)> {
    if let Some(word) = words.get(l, w) {
        return Some(word.centre());
    }
    let notes = song.singing.notes.lines.get(l)?;
    let of: Vec<usize> = (0..notes.notes.len())
        .filter(|&k| notes.notes[k].word == w)
        .collect();
    let (&a, &z) = (of.first()?, of.last()?);
    let mid = (notes.notes[a].start + notes.notes[z].end) / 2.0;
    Some((lane.x(mid), lane.y(notes.word_height(a..z + 1))))
}

/// Where shot `r` is `f` of its way (0 fired, 1 landed) from `from` to
/// `to`: an arc over the straight way, higher the further it goes.
fn along(from: (f32, f32), to: (f32, f32), f: f32, px: f32) -> (f32, f32) {
    let (dx, dy) = (to.0 - from.0, to.1 - from.1);
    let arc = px * 0.9 + (0.18 * (dx * dx + dy * dy).sqrt()).min(px * 3.0);
    let f = f.clamp(0.0, 1.0);
    (from.0 + dx * f, from.1 + dy * f - arc * 4.0 * f * (1.0 - f))
}

/// The shots of `song` in flight (or just landed) at `clock`, the words
/// `words` laid on `lane`.
pub(super) fn shots(song: &Song, lane: &Lane, words: &Words, clock: f64) -> Shots {
    let mut out = Shots::default();
    if !song.game {
        return out;
    }
    let p = lane.pill;
    let mut rescues: Vec<((usize, usize), Rescue)> = (song.singing.rescues())
        .filter(|(_, r)| clock >= r.fired && clock < r.lands() + FLASH)
        .map(|(w, r)| (w, *r))
        .collect();
    rescues.sort_by(|a, b| a.1.fired.total_cmp(&b.1.fired));
    for (to, r) in rescues {
        let (Some(a), Some(b)) = (
            place(song, lane, words, r.from),
            place(song, lane, words, to),
        ) else {
            continue;
        };
        let f = ((clock - r.fired) / FLIGHT) as f32;
        if f < 1.0 {
            flying(&mut out, (a, b), f, (p, clock as f32));
        } else {
            landed(&mut out, b, ((clock - r.lands()) / FLASH) as f32, p);
            // Its sparks, from where it landed (gone by with the lane since).
            let back = lane.pps * (clock - r.lands()) as f32;
            out.sprays.push(Spray {
                at: (b.0 + back, b.1),
                half: (0.0, 0.0),
                start: r.lands() as f32,
                drift: -lane.pps,
                size: p * 0.8,
                seed: (r.lands() * 1000.0) as u32,
                rgb: GOLD,
                kind: Kind::Scored,
            });
        }
    }
    out
}

/// A shot `f` of its way from `ends.0` to `ends.1`, the tubes `p` thick,
/// at `time`: its tail, a glowing head and a sparkle turning on it.
fn flying(out: &mut Shots, ends: ((f32, f32), (f32, f32)), f: f32, (p, time): (f32, f32)) {
    let (a, b) = ends;
    let head = along(a, b, f, p);
    // The tail: pieces back along its way, thinner and fainter.
    for k in 0..TAIL {
        let (f0, f1) = (f - 0.045 * (k + 1) as f32, f - 0.045 * k as f32);
        if f0 < 0.0 {
            break;
        }
        let left = 1.0 - k as f32 / TAIL as f32;
        out.out.push(segment(
            along(a, b, f0, p),
            along(a, b, f1, p),
            p * 0.11 * left,
            rgba(GOLD_TOP, 0.9 * left),
            rgba(GOLD, 0.45 * left),
            p * 0.35,
        ));
    }
    let core = disc(
        head,
        p * 0.22,
        rgba([1.0, 0.98, 0.9], 1.0),
        rgba(GOLD, 0.8),
        p * 1.2,
    );
    out.out.push(core);
    out.lights.push(disc(
        head,
        p * 0.12,
        rgba(GOLD_TOP, 0.9),
        rgba(GOLD, 0.5),
        p * 0.6,
    ));
    let size = p * 0.85;
    let quad = [head.0 - size, head.1 - size, head.0 + size, head.1 + size];
    let spark = Instance::of(Shape::Spark, quad, [head.0, head.1, size, time * 9.0]);
    out.lights
        .push(spark.colors(rgba([1.0, 0.97, 0.85], 0.9), [0.0; 4]));
}

/// A shot landed at `at`, `g` (0 to 1) through its flash, the tubes `p`
/// thick: a ring of light opening and fading.
fn landed(out: &mut Shots, at: (f32, f32), g: f32, p: f32) {
    let g = g.clamp(0.0, 1.0);
    let fade = (1.0 - g) * (1.0 - g);
    let r = p * (0.3 + 1.1 * g);
    out.lights.push(disc(
        at,
        r,
        rgba(GOLD_TOP, 0.0),
        rgba(GOLD_TOP, 0.9 * fade),
        p * 0.8,
    ));
    out.out.push(disc(
        at,
        p * 0.25 * (1.0 - g),
        rgba([1.0; 3], fade),
        rgba(GOLD, 0.6 * fade),
        p,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A shot leaves where it is fired and lands on what it rescues,
    /// rising over the way between.
    #[test]
    fn a_shot_arcs_from_its_bubble_onto_the_word_missed() {
        let (a, b) = ((500.0, 300.0), (200.0, 320.0));
        assert_eq!(along(a, b, 0.0, 40.0), a);
        assert_eq!(along(a, b, 1.0, 40.0), b);
        let mid = along(a, b, 0.5, 40.0);
        assert!((mid.0 - 350.0).abs() < 1e-3 && mid.1 < 270.0, "{mid:?}");
    }
}
