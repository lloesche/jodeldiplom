//! The diploma's entries, written in on their ruled lines as their beats
//! come: the points counting up in gold ink (bumping at each tick, popping
//! as they land, sparks flying), the words hit and the best streak written
//! by hand (sparks as the words are in, stars and notes for the streak);
//! then a new best, a red rubber stamp slapped on at an angle (confetti
//! raining), or the best before, calmly, its words on the points' line.
//! The rubber stamp ([`Stage::rubber_stamp`]) goes right of the longest
//! value where there is room for it, else by the heading's right corner.

use super::super::super::confetti::shower;
use super::super::super::stage_gpu::{kind, Instance};
use super::super::{disc, rgba, Face, Rgb, Stage, GOLD, WHITE};
use super::sheet::{ENTRIES, LINE_FROM, LINE_TO};
use super::{bump, burst, clamp01, fade, Drawn, Pen, ResultShown, Room, Written, PRINT};
use crate::core::singing::{Beat, WORDS_FILL};
use crate::gui::particles::Kind;
use crate::view::karaoke::points_text;

/// How long the best streak takes to write.
const STREAK_WRITE: f32 = 0.45;
/// How long a new best's words take to write, after its stamp.
const BEST_WRITE: f32 = 0.5;
/// A new best's stamp: its words and red.
const NEW_BEST: (&str, Rgb) = ("NEW BEST!", [0.84, 0.12, 0.14]);
/// How a rubber stamp is turned (radians), and its words' size and its
/// frame's room round them (in the sheet's units).
const STAMP_TURN: f32 = -0.2;
const STAMP_PX: f32 = 5.2;
const STAMP_PAD: (f32, f32) = (2.4, 4.4);
/// The least share of its size a stamp is shrunk to beside the entries
/// before it goes by the heading instead.
const STAMP_LEAST: f32 = 0.62;

impl Room {
    /// Where an entry's value is written: its left on the line `i`, its
    /// capitals' middle `px` high text sitting on it.
    fn entry(&self, i: usize, px: f32) -> (f32, f32) {
        let (x, y) = self.at(LINE_FROM + 0.02, ENTRIES[i]);
        (x, y + 2.0 * self.s - 0.38 * px)
    }

    /// The size of the handwritten entries.
    fn entry_px(&self) -> f32 {
        (4.6 * self.s).round().max(12.0)
    }

    /// The size of the points, `figures` of them: as large as fits a third
    /// of the sheet.
    fn points_px(&self, figures: usize) -> f32 {
        let px = (7.5 * self.s).round().max(18.0);
        px.min((self.across(0.34) / (figures.max(4) as f32 * 0.62)).round())
    }
}

impl Stage {
    /// The points, counting up (each tick a bump) in gold ink on their
    /// line, popping as they land with a burst of sparks.
    pub(super) fn points(
        &mut self,
        d: &mut Drawn,
        room: &Room,
        shown: &ResultShown,
        words: &Written,
    ) {
        let (t, reveal) = (shown.t, &shown.shown.reveal);
        let (from, to) = reveal.count();
        let shows = fade(Some(t - from + 0.15), 0.15);
        if shows <= 0.0 {
            return;
        }
        let last_tick = reveal
            .beats()
            .iter()
            .rev()
            .find(|(at, b)| matches!(b, Beat::Tick(..)) && *at <= t)
            .map(|(at, _)| t - at);
        let landed = reveal.age(Beat::Points, t);
        let scale = match landed {
            Some(_) => bump(landed, 0.22),
            None => bump(last_tick, 0.05),
        };
        let counting;
        let value = match landed {
            Some(_) => words.points.as_str(),
            None => {
                counting = points_text(reveal.points_at(t));
                counting.as_str()
            }
        };
        let px = room.points_px(words.points.chars().count());
        let at = room.entry(0, px);
        let color = rgba(room.inks.gold, shows);
        let w = self.digits(&mut d.out, value, (px, scale), at, color);
        let middle = (at.0 + w / 2.0, at.1);
        // A glow round them as they land.
        let glow = landed.map_or(0.0, |a| (-(a as f32) * 2.5).exp());
        let lit = rgba(GOLD, 0.25 * glow);
        d.lights.push(disc(middle, 0.0, [0.0; 4], lit, px * 1.3));
        let spray = burst(middle, to, px * 0.5, (23, GOLD), Kind::Scored);
        d.sprays.push(spray);
    }

    /// `value` in the numbers' face, figure by figure (each kept once in
    /// the atlas, whatever the count), from `at` (its left, its capitals'
    /// middle); its width.
    fn digits(
        &mut self,
        out: &mut Vec<Instance>,
        value: &str,
        (px, scale): (f32, f32),
        at: (f32, f32),
        color: [f32; 4],
    ) -> f32 {
        let mut left = at.0;
        let mut buf = [0u8; 4];
        for ch in value.chars() {
            let text = (Face::SHOUT, &*ch.encode_utf8(&mut buf));
            let c = ((left, at.1), 0.0);
            left += self.written(out, text, (px, scale), c, color, Pen::FLAT);
        }
        left - at.0
    }

    /// Text written in by hand on entry `i`'s line from `age` seconds ago
    /// over `over` seconds, in the pen's ink, the nib at its front while
    /// it writes; where it ends.
    fn entry_written(
        &mut self,
        d: &mut Drawn,
        room: &Room,
        (i, text): (usize, &str),
        (age, over): (f32, f32),
    ) -> (f32, f32) {
        let px = room.entry_px();
        let at = room.entry(i, px);
        let share = clamp01(age / over);
        let pen = rgba(room.inks.pen, 0.95);
        let text = (Face::NUMBER, text);
        let w = self.written(
            &mut d.out,
            text,
            (px, 1.0),
            (at, 0.0),
            pen,
            Pen::writing(share),
        );
        if share < 1.0 {
            let nib = (at.0 + w * share, at.1 + 0.3 * px);
            d.out.push(disc(nib, 0.3 * room.s, pen, [0.0; 4], 0.0));
        }
        (at.0 + w, at.1)
    }

    /// The words hit, written in as their beat comes, sparks as they are.
    pub(super) fn hits(
        &mut self,
        d: &mut Drawn,
        room: &Room,
        shown: &ResultShown,
        words: &Written,
    ) {
        let (t, r) = (shown.t, &shown.shown.result);
        let Some(a) = shown.shown.reveal.age(Beat::Words, t) else {
            return;
        };
        let fill = WORDS_FILL as f32;
        let end = self.entry_written(d, room, (1, &words.hits), (a as f32, fill));
        if r.hit > 0 {
            let at = t - a + WORDS_FILL;
            let spray = burst(end, at, 3.0 * room.s, (31, GOLD), Kind::Scored);
            d.sprays.push(spray);
        }
    }

    /// The best streak, written in as its beat comes, with a burst of
    /// stars and notes.
    pub(super) fn streak(
        &mut self,
        d: &mut Drawn,
        room: &Room,
        shown: &ResultShown,
        words: &Written,
    ) {
        let t = shown.t;
        let Some(a) = shown.shown.reveal.age(Beat::Streak, t) else {
            return;
        };
        let age = (a as f32, STREAK_WRITE);
        let end = self.entry_written(d, room, (2, &words.streak), age);
        let at = room.entry(2, 0.0);
        let middle = ((at.0 + end.0) / 2.0, at.1 - 2.0 * room.s);
        let spray = burst(middle, t - a, 5.0 * room.s, (41, WHITE), Kind::StepUp);
        d.sprays.push(spray);
    }

    /// A new best: NEW BEST! in a red rubber stamp slapped on (confetti
    /// raining), its words written on the points' line; or the best
    /// before, calmly, there.
    pub(super) fn best(
        &mut self,
        d: &mut Drawn,
        room: &Room,
        shown: &ResultShown,
        words: &Written,
    ) {
        let (t, reveal) = (shown.t, &shown.shown.reveal);
        let new = reveal.age(Beat::NewBest, t).map(|a| a as f32);
        let calm = reveal.find(|b| matches!(b, Beat::Best(_)));
        match (new, calm) {
            (Some(a), _) => {
                self.rubber_stamp(d, room, words, NEW_BEST, (t, a));
                let share = clamp01((a - 0.3) / BEST_WRITE);
                let pen = (Face::NUMBER, rgba(room.inks.pen, 0.9));
                self.best_line(d, room, words, pen, Pen::writing(share));
                let start = t - f64::from(a);
                for k in 0..5u32 {
                    d.sprays.push(shower(
                        (room.x, room.y),
                        (room.w, room.h),
                        1.8 * room.u,
                        start + 0.12 * f64::from(k),
                        ([1.0, 0.9, 0.7], 101 + k),
                    ));
                }
            }
            (None, Some((at, _))) if t >= at => {
                let ink = rgba(room.inks.ink, 0.75 * fade(Some(t - at), 0.5));
                self.best_line(d, room, words, (PRINT, ink), Pen::FLAT);
            }
            (None, _) => {}
        }
    }

    /// The words under the best (`words`'), small, on the points' line:
    /// right-aligned to its end where they fit after the points, else
    /// under its end; in `face` and `color`, put down with `pen`.
    fn best_line(
        &mut self,
        d: &mut Drawn,
        room: &Room,
        words: &Written,
        (face, color): (Face, [f32; 4]),
        pen: Pen,
    ) {
        let s = room.s;
        let px = (2.8 * s).round().max(9.0);
        let pp = room.points_px(words.points.chars().count());
        let value_end = room.entry(0, pp).0 + self.measure(Face::SHOUT, &words.points, pp);
        let (end, line) = room.at(LINE_TO, ENTRIES[0]);
        let line = line + 2.0 * s;
        let w = self.measure(face, &words.best, px);
        let fits = end - w > value_end + 1.5 * s;
        let y = if fits {
            line - 0.9 * px
        } else {
            line + 0.9 * px
        };
        let text = (face, words.best.as_str());
        self.written(&mut d.out, text, (px, 1.0), ((end, y), 1.0), color, pen);
    }

    /// A rubber stamp of `text` in `ink` slapped on `age` seconds before
    /// `t` (the reveal's time; down from large in a tenth of a second, a
    /// thump), turned, its frame worn
    /// and its words specked where the ink did not take: right of the
    /// entries' longest value, shrunk to fit there down to
    /// [`STAMP_LEAST`] of its size, else by the heading's right corner.
    pub(super) fn rubber_stamp(
        &mut self,
        d: &mut Drawn,
        room: &Room,
        words: &Written,
        (text, ink): (&str, Rgb),
        (t, age): (f64, f32),
    ) {
        let s = room.s;
        let ((cx, cy), k) = self.stamp_place(room, words, text);
        let (scale, shows) = if age < 0.1 {
            (1.9 - 0.9 * (age / 0.1) * (age / 0.1), clamp01(age / 0.04))
        } else {
            let k = age - 0.1;
            (1.0 + 0.07 * (-k * 12.0).exp() * (k * 34.0).cos(), 1.0)
        };
        let px = (STAMP_PX * s * k).round().max(9.0);
        let tw = self.measure(Face::SHOUT, text, px);
        let color = rgba(ink, 0.9 * shows);
        let c = (cx + room.shake.0, cy + room.shake.1);
        let pen = Pen {
            turn: STAMP_TURN,
            ..Pen::FLAT
        };
        self.written(
            &mut d.top,
            (Face::SHOUT, text),
            (px, scale),
            (c, 0.5),
            color,
            pen,
        );
        let half = (
            (tw / 2.0 + STAMP_PAD.0 * s * k) * scale,
            STAMP_PAD.1 * s * k * scale,
        );
        let reach = half.0.hypot(half.1) + 2.0;
        d.top.push(Instance {
            quad: [c.0 - reach, c.1 - reach, c.0 + reach, c.1 + reach],
            shape: [c.0, c.1, half.0, half.1],
            color,
            color2: rgba(room.inks.paper, 0.8 * shows),
            params: [kind::STAMP, STAMP_TURN, s * k, 0.0],
            ..Instance::default()
        });
        let start = t - f64::from(age);
        for (n, side) in [-1.0f32, 1.0].into_iter().enumerate() {
            let at = (c.0 + side * half.0, c.1);
            let spray = burst(at, start, px * 0.9, (51 + n as u32, WHITE), Kind::StepUp);
            d.sprays.push(spray);
        }
    }

    /// Where a rubber stamp of `text` goes (its middle, not shaken) and
    /// the share of its size it is drawn at ([`Stage::rubber_stamp`]).
    fn stamp_place(&mut self, room: &Room, words: &Written, text: &str) -> ((f32, f32), f32) {
        let s = room.s;
        let tw = self.measure(Face::SHOUT, text, (STAMP_PX * s).round().max(9.0));
        // Its width turned.
        let (sin, cos) = STAMP_TURN.abs().sin_cos();
        let wide = (tw + 2.0 * STAMP_PAD.0 * s) * cos + 2.0 * STAMP_PAD.1 * s * sin;
        let (ep, pp) = (
            room.entry_px(),
            room.points_px(words.points.chars().count()),
        );
        let longest = self
            .measure(Face::SHOUT, &words.points, pp)
            .max(self.measure(Face::NUMBER, &words.hits, ep))
            .max(self.measure(Face::NUMBER, &words.streak, ep));
        let unshaken = |fx: f32, fy: f32| {
            let (x, y) = room.at(fx, fy);
            (x - room.shake.0, y - room.shake.1)
        };
        let from = unshaken(LINE_FROM + 0.02, 0.0).0 + longest + 2.0 * s;
        let to = unshaken(0.94, 0.0).0;
        let k = ((to - from) / wide).min(1.0);
        if k >= STAMP_LEAST {
            let x = unshaken(0.79, 0.0)
                .0
                .min(to - wide * k / 2.0)
                .max(from + wide * k / 2.0);
            return ((x, unshaken(0.0, 0.54).1), k);
        }
        // By the heading's right corner, over the border a little.
        let (from, to) = (unshaken(0.7, 0.0).0, unshaken(0.97, 0.0).0);
        let k = ((to - from) / wide).min(1.0);
        ((to - wide * k / 2.0, unshaken(0.0, 0.205).1), k)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{result, stage};
    use super::super::Tint;
    use super::*;
    use crate::core::singing::{Medal, Shown, Verdict};

    /// A new best's stamp keeps off the entries' values, however long, on
    /// a wide sheet and an upright one: right of the longest where there
    /// is room, else by the heading, above them; in the sheet either way.
    #[test]
    fn the_stamp_keeps_off_the_entries() {
        let mut r = result(Some(Medal::Gold), true, Some(9_850));
        (r.points, r.hit, r.words, r.streak) = (124_850, 187, 243, 112);
        let shown = Shown::new(r, 1, 0);
        let look = super::super::seal::look(Verdict::Medal(Medal::Gold), &Tint::STAGE);
        for (w, h) in [
            (1600.0, 900.0),
            (1024.0, 768.0),
            (640.0, 1080.0),
            (500.0, 900.0),
        ] {
            let mut stage = stage();
            let words = stage.words_of(&shown);
            let area = (0.0, 0.0, w, h);
            let line = super::super::Line::above(area, h * 0.9);
            let room = Room::new(area, line, &Tint::STAGE, look);
            let ((cx, cy), k) = stage.stamp_place(&room, &words, NEW_BEST.0);
            let s = room.s;
            let px = (STAMP_PX * s * k).round().max(9.0);
            let tw = stage.measure(Face::SHOUT, NEW_BEST.0, px);
            let (sin, cos) = STAMP_TURN.abs().sin_cos();
            let half = (tw / 2.0 + STAMP_PAD.0 * s * k, STAMP_PAD.1 * s * k);
            let (wide, high) = (half.0 * cos + half.1 * sin, half.0 * sin + half.1 * cos);
            let (sx, sy, sw, _) = room.sheet;
            assert!(cx - wide > sx && cx + wide < sx + sw, "{w}×{h}");
            let (ep, pp) = (
                room.entry_px(),
                room.points_px(words.points.chars().count()),
            );
            let ends = [
                room.entry(0, pp).0 + stage.measure(Face::SHOUT, &words.points, pp),
                room.entry(1, ep).0 + stage.measure(Face::NUMBER, &words.hits, ep),
                room.entry(2, ep).0 + stage.measure(Face::NUMBER, &words.streak, ep),
            ];
            let beside = cx - wide > ends.into_iter().fold(0.0, f32::max);
            let above = cy + high < room.entry(0, pp).1 - pp * 0.5;
            assert!(beside || above, "{w}×{h}: {cx} {cy} {wide} {ends:?}");
            assert!(k >= 0.5 && cy > sy, "{w}×{h}: {k}");
        }
    }
}
