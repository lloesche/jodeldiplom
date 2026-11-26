//! The score, top right ([`Stage::score`]): the points rolling in as on an
//! odometer, the multiplier's rung under them. Each column of figures is
//! a drum turning forward from the figure it showed to the one it shows
//! next, the figure going out sliding up out of its slot as the one coming
//! in slides up into it; a column whose figure stays stays still. On a big
//! jump the columns under the jump's top spin whole laps more (up to
//! [`LAPS`]), a blur of copies, and the number glows (mixed towards
//! [`HOT`]) and swells a little, more for a big jump, while it counts.
//!
//! Each change rolls in on its own, eased out, over [`ROLL`] (longer for a
//! bigger jump, up to [`ROLL_LONG`]), alongside those still rolling: so the
//! counter is never more than that behind the score, never past it, and
//! each drum only ever turns forward. A score lower than the one shown (a
//! seek back, another song) or one just come is shown at once. The counter
//! is stepped by the frames' own time, apart from the core's state: what
//! it counts to is the core's score.

use super::super::big_text::{BigText, Face, Faces};
use super::super::pixels::{mix, PixelRect};
use super::super::stage_gpu::Instance;
use super::{clip, Stage};
use crate::view::karaoke::{KaraokePicture, Score, HIT};
use crate::view::palette;

/// How long points added take to roll in, in seconds: a word's.
const ROLL: f32 = 0.35;
/// How long the biggest jumps take ([`BIG`] points and more).
const ROLL_LONG: f32 = 0.6;
/// A jump of no more than this many points is a plain one...
const PLAIN: f64 = 100.0;
/// ... and one of this many or more the biggest.
const BIG: f64 = 10_000.0;
/// The most whole laps a column under a jump's top spins, on the biggest.
const LAPS: f32 = 3.0;
/// How long the glow takes to come while the number counts, and to go
/// once it stops, in seconds.
const GLOW_IN: f32 = 0.05;
const GLOW_OUT: f32 = 0.3;
/// How long a column's motion is smeared over, as a camera's shutter
/// open for a frame, in seconds.
const SHUTTER: f64 = 1.0 / 90.0;
/// The most copies a smeared column is drawn in.
const COPIES: usize = 6;
/// What the number glows towards.
const HOT: [u8; 3] = [255, 248, 225];
/// The figures, whose ink the slots are made round.
const FIGURES: &str = "0123456789";

/// How many of its figures' sizes wide the score's corner is: room for a
/// score of nine figures and for its rung's name under it.
const SCORE_WIDE: f32 = 6.5;

/// Points added: from what to what, how long ago and how long it rolls
/// for, in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Change {
    from: u32,
    to: u32,
    age: f32,
    length: f32,
}

impl Change {
    /// How far it has rolled in, 0 to 1 (eased out: quick, then
    /// settling), and how fast it rolls now (a second).
    fn through(&self) -> (f64, f64) {
        let u = f64::from((self.age / self.length).clamp(0.0, 1.0));
        let left = 1.0 - u;
        (
            1.0 - left * left * left,
            3.0 * left * left / f64::from(self.length),
        )
    }

    /// How many turns column `k` (0 the units) makes in it: forward to
    /// its new figure, and under the jump's top as many whole laps more as
    /// the jump is big (never more than counting up one by one would make).
    fn turns(&self, k: u32) -> u64 {
        let (from, to) = (column(self.from, k), column(self.to, k));
        let forward = (to % 10 + 10 - from % 10) % 10;
        let jump = self.to - self.from;
        let laps = if k + 1 < figures(jump) {
            ((bigness(jump) * LAPS).floor() as u64).min((to - from) / 10)
        } else {
            0
        };
        forward + 10 * laps
    }
}

/// Column `k` of `points` (0 the units) as a count of its turns: the
/// points over ten to the `k`.
fn column(points: u32, k: u32) -> u64 {
    u64::from(points) / 10u64.pow(k)
}

/// How many figures `points` has.
fn figures(points: u32) -> u32 {
    points.checked_ilog10().map_or(1, |l| l + 1)
}

/// How big a jump of `points` is: 0 ([`PLAIN`] and fewer) to 1 ([`BIG`]
/// and more).
fn bigness(points: u32) -> f32 {
    let p = f64::from(points.max(1)).log10();
    ((p - PLAIN.log10()) / (BIG.log10() - PLAIN.log10())).clamp(0.0, 1.0) as f32
}

/// The counter: the points at rest under the changes still rolling in
/// (none while no score shows), those changes, and its glow.
#[derive(Clone, Debug, Default)]
pub(super) struct Odometer {
    base: Option<u32>,
    changes: Vec<Change>,
    /// How much the number glows, 0 to 1, and how big the jumps it glows
    /// for are (0 to 1, [`bigness`]).
    glow: f32,
    big: f32,
}

impl Odometer {
    /// The points it counts to: the last change's, else those at rest.
    fn target(&self) -> Option<u32> {
        self.changes.last().map(|c| c.to).or(self.base)
    }

    /// `dt` seconds on, the score `points` now (none: none shows): points
    /// added start rolling in, a lower score or one just come is shown at
    /// once.
    pub fn step(&mut self, points: Option<u32>, dt: f32) {
        let dt = dt.max(0.0);
        for change in &mut self.changes {
            change.age += dt;
        }
        // Those rolled in, from the first: at rest under the rest. (A
        // column's laps are never more than counting up one by one makes,
        // so each drum is then where it was or whole turns on.)
        while self.changes.first().is_some_and(|c| c.age >= c.length) {
            self.base = Some(self.changes.remove(0).to);
        }
        match (self.target(), points) {
            (_, None) => *self = Odometer::default(),
            (Some(was), Some(now)) if now == was => {}
            (Some(was), Some(now)) if now > was => {
                let big = bigness(now - was);
                self.big = if self.moving() {
                    self.big.max(big)
                } else {
                    big
                };
                self.changes.push(Change {
                    from: was,
                    to: now,
                    age: 0.0,
                    length: ROLL + (ROLL_LONG - ROLL) * big,
                });
            }
            (_, Some(now)) => {
                *self = Odometer {
                    base: Some(now),
                    ..Odometer::default()
                }
            }
        }
        self.glow = if self.changes.is_empty() {
            (self.glow - dt / GLOW_OUT).max(0.0)
        } else {
            (self.glow + dt / GLOW_IN).min(1.0)
        };
        if !self.moving() {
            self.big = 0.0;
        }
    }

    /// Whether it still moves: points rolling in, or its glow going.
    pub fn moving(&self) -> bool {
        !self.changes.is_empty() || self.glow > 0.0
    }

    /// What it shows (nought where no score was stepped to).
    pub fn reading(&self) -> Reading {
        let base = self.base.unwrap_or(0);
        let columns = (0..figures(self.target().unwrap_or(0)))
            .map(|k| {
                let at = (column(base, k) as f64, 0.0);
                self.changes.iter().fold(at, |(at, speed), c| {
                    let (through, fast) = c.through();
                    let turns = c.turns(k) as f64;
                    (at + turns * through, speed + turns * fast)
                })
            })
            .collect();
        Reading {
            columns,
            glow: self.glow,
            big: self.big,
        }
    }
}

/// What the counter shows: each column's drum from the units up, where it
/// is in turns (its figure the whole turns' last digit, nought a blank
/// where it never turned; between two figures while it turns) and how fast
/// it turns, a second; how much the number glows and how big its jump is,
/// 0 to 1.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Reading {
    columns: Vec<(f64, f64)>,
    glow: f32,
    big: f32,
}

/// A figure of the counter as it is drawn.
#[derive(Clone, Debug, PartialEq)]
struct Figure {
    /// The figure.
    pub text: String,
    /// The size its glyphs are made at.
    pub base: f32,
    /// The size it is drawn at.
    pub px: f32,
    /// Its line's top-left.
    pub at: (f32, f32),
    /// How much of it shows.
    pub alpha: f32,
    /// The slot it is kept to (pixels): its column's, a turn of the drum
    /// high.
    pub slot: PixelRect,
}

/// The score as it is set top right.
#[derive(Clone, Debug, PartialEq)]
struct Placed {
    /// The rung's name under the points: its text, size and line's
    /// top-left (none at ×1).
    pub rung: Option<(String, f32, (f32, f32))>,
    /// The counter's figures.
    pub figures: Vec<Figure>,
    /// Their colour, the glow mixed in.
    pub color: [u8; 3],
}

/// The score's sizes in the window's `area` (left, top, width, height):
/// its figures', the margin it keeps from the corner, and its rung's
/// under it. Its figures a share of the line sung's, smaller where the
/// corner would come down past the HUD's band over the lane
/// ([`super::Lane::hud`]: over the marquee bulbs).
pub fn sizes((x, y, w, h): (f32, f32, f32, f32)) -> (f32, f32, f32) {
    let sizes = |px: f32| (px, px * 0.8, (px * 0.45).round().max(10.0));
    let (_, bottom) = super::Lane::new((x, y, w, h), 0.0).hud((x, y, w, h));
    let mut px = (super::lyrics::big_of(h) * 0.42).round().max(12.0);
    while px > 12.0 && corner_bottom(y, sizes(px)) > bottom {
        px -= 1.0;
    }
    sizes(px)
}

/// The score's corner's bottom under the window's top `y`, at its sizes.
fn corner_bottom(y: f32, (px, margin, small): (f32, f32, f32)) -> f32 {
    y + margin * 0.6 + px * 1.25 + small * 1.5
}

/// The corner the score takes top right in the window's `area` (left,
/// top, width, height), its rung under it: left, top, right, bottom. The
/// effects over the frame keep it clear (the edges on fire), as they do
/// the lyric band.
pub fn corner((x, y, w, h): (f32, f32, f32, f32)) -> [f32; 4] {
    let sizes = sizes((x, y, w, h));
    let (px, margin, _) = sizes;
    [
        x + w - margin - px * SCORE_WIDE,
        y,
        x + w,
        corner_bottom(y, sizes),
    ]
}

/// `score` as `reading` shows it, set top right in the window's `area`
/// (left, top, width, height), in `faces` (its figures in
/// [`Face::NUMBER`], its rung in [`Face::SHOUT`]).
fn placed(
    faces: &mut Faces,
    score: Score,
    reading: &Reading,
    (x, y, w, h): (f32, f32, f32, f32),
) -> Placed {
    let (px, margin, small) = sizes((x, y, w, h));
    // Under it the multiplier's rung, named as the badge names it.
    let rung = score.rung_label().map(|label| {
        let lw = faces.get(Face::SHOUT).measure(&label, small);
        let at = (x + w - margin - lw, y + margin * 0.6 + px * 1.25);
        (label, small, at)
    });
    let right_top = (x + w - margin, y + margin * 0.6);
    Placed {
        rung,
        figures: figures_of(faces.get(Face::NUMBER), reading, right_top, px),
        color: mix(HIT, HOT, 0.6 * reading.glow),
    }
}

/// The figures `reading` is drawn as at `px` (its glyphs in `text`, the
/// score's face), its right edge at `right` and its top at `top`: in each
/// column the figure going out and the one coming in (in several copies
/// along its motion where it spins fast), each kept to its column's slot.
/// At rest it is the points as [`points_text`] writes them, set as that
/// text is.
///
/// [`points_text`]: crate::view::karaoke::points_text
fn figures_of(
    text: &mut BigText,
    reading: &Reading,
    (right, top): (f32, f32),
    px: f32,
) -> Vec<Figure> {
    let advance = text.measure("0", px);
    let apart = text.measure("\u{2009}", px);
    let n = reading.columns.len();
    // Thousands apart, as `points_text` puts them: after columns 3, 6...
    let apart_after = |k: usize| k > 0 && k.is_multiple_of(3);
    // As wide as the text: its advances added up as it adds them.
    let mut width = 0.0;
    for k in (0..n).rev() {
        width += advance;
        if apart_after(k) {
            width += apart;
        }
    }
    // The slots: from a little above the figures' ink to a little below,
    // a turn as tall as a slot. The number swells from its right edge and
    // the middle of its ink.
    let (ink_top, ink_bottom) = text.ink_band(FIGURES, px);
    let room = 0.1 * px;
    let middle = (ink_top + ink_bottom) / 2.0;
    let pop = 1.0 + reading.glow * (0.04 + 0.1 * reading.big);
    let size = px * pop;
    let y = top + middle * (1.0 - pop);
    let slot_top = y + (ink_top - room) * pop;
    let pitch = (ink_bottom - ink_top + 2.0 * room) * pop;
    let mut out = Vec::new();
    let mut pen = right - width;
    for k in (0..n).rev() {
        let x = pen + (pen - right) * (pop - 1.0);
        let side = 0.06 * size;
        let slot: PixelRect = (
            (x - side).max(0.0).floor() as usize,
            slot_top.max(0.0).floor() as usize,
            (advance * pop + 2.0 * side).ceil() as usize,
            pitch.ceil() as usize,
        );
        let (at, speed) = reading.columns[k];
        // Smeared over the shutter where it spins fast: copies trailing
        // behind it (below: it turns upwards), together as strong as one.
        let smear = (speed * SHUTTER).min(0.95);
        let copies = if smear < 0.06 {
            1
        } else {
            (1 + (smear * 10.0).ceil() as usize).min(COPIES)
        };
        let alpha = 1.0 - 0.05f32.powf(1.0 / copies as f32);
        let alpha = if copies == 1 { 1.0 } else { alpha };
        for j in 0..copies {
            let back = if copies == 1 {
                0.0
            } else {
                smear * j as f64 / (copies - 1) as f64
            };
            let at = (at - back).max(0.0);
            let whole = at.floor();
            let phase = (at - whole) as f32;
            let shown = whole as u64;
            let mut figure_at = |digit: u64, off: f32| {
                out.push(Figure {
                    text: (digit % 10).to_string(),
                    base: px,
                    px: size,
                    at: (x, y + off * pitch),
                    // Fading away from the middle of its slot, as on a
                    // drum curving away.
                    alpha: alpha * (1.0 - off * off),
                    slot,
                });
            };
            // The units always; a higher column's nought only under a
            // figure (a leading one is blank).
            if k == 0 || shown > 0 {
                figure_at(shown, -phase);
            }
            if phase > 1e-3 {
                figure_at(shown + 1, 1.0 - phase);
            }
        }
        pen += advance;
        if apart_after(k) {
            pen += apart;
        }
    }
    out
}

/// A colour of the window's (sRGB bytes) at `alpha`, as the stage takes it.
pub(super) fn rgba8(c: [u8; 3], alpha: f32) -> [f32; 4] {
    [
        f32::from(c[0]) / 255.0,
        f32::from(c[1]) / 255.0,
        f32::from(c[2]) / 255.0,
        alpha,
    ]
}

impl Stage {
    /// Karaoke mode just come: its score shows at once, rolling from
    /// nothing.
    pub fn entered(&mut self) {
        self.odometer.step(None, 0.0);
    }

    /// Whether what the stage draws moves by itself, so frames must keep
    /// coming while nothing else moves: the score rolling up.
    pub fn moving(&self) -> bool {
        self.odometer.moving()
    }

    /// `picture`'s score into `out`, its counter stepped `dt` seconds on
    /// (none: no score, the counter at rest on nothing), set top right in
    /// the window's `area`.
    pub(super) fn score(
        &mut self,
        out: &mut Vec<Instance>,
        picture: &KaraokePicture,
        area: (f32, f32, f32, f32),
        dt: f32,
    ) {
        self.odometer.step(picture.score.map(|s| s.points), dt);
        let Some(score) = picture.score else {
            return;
        };
        let placed = placed(&mut self.faces, score, &self.odometer.reading(), area);
        if let Some((label, px, at)) = &placed.rung {
            let dim = rgba8(palette::DIM, 1.0);
            let whole = (Face::SHOUT, label.as_str());
            self.set_text(out, whole, (*px, *px), *at, f32::INFINITY, (dim, dim));
        }
        for f in &placed.figures {
            let color = rgba8(placed.color, f.alpha);
            let from = out.len();
            let figure = (Face::NUMBER, f.text.as_str());
            self.set_text(
                out,
                figure,
                (f.base, f.px),
                f.at,
                f32::INFINITY,
                (color, color),
            );
            let (sx, sy, sw, sh) = f.slot;
            let slot = [sx as f32, sy as f32, (sx + sw) as f32, (sy + sh) as f32];
            clip(out, from, slot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::stage_gpu::kind;

    const FRAME: f32 = 1.0 / 60.0;

    /// The frame at `secs` (a frame a [`FRAME`], the first at 0).
    fn frame_at(secs: f32) -> usize {
        (secs / FRAME).round() as usize
    }

    impl Reading {
        /// `points` standing still.
        fn at_rest(points: u32) -> Reading {
            Reading {
                columns: (0..figures(points))
                    .map(|k| (column(points, k) as f64, 0.0))
                    .collect(),
                glow: 0.0,
                big: 0.0,
            }
        }

        /// The figures it shows (those going out), as a number.
        fn value(&self) -> u64 {
            self.columns
                .iter()
                .enumerate()
                .map(|(k, &(at, _))| (at.floor() as u64 % 10) * 10u64.pow(k as u32))
                .sum()
        }
    }

    /// The score and its rung (a long score, the top rung's name) lie in
    /// the score's corner, which the effects over the frame keep clear.
    #[test]
    fn the_score_lies_in_its_corner() {
        let area = (0.0, 0.0, 1600.0, 900.0);
        let picture = KaraokePicture {
            bar: 0.1,
            lines: Vec::new(),
            count: None,
            cue: None,
            controls: None,
            start: None,
            before: Vec::new(),
            moved: 1.0,
            score: Some(Score {
                points: 987_654_321,
                streak: 60,
            }),
            result: None,
        };
        let mut stage = Stage::default();
        let mut out = Vec::new();
        // At rest on it.
        for _ in 0..200 {
            out.clear();
            stage.score(&mut out, &picture, area, 0.05);
        }
        let corner = corner(area);
        assert!(out.len() > 9, "{}", out.len());
        for i in &out {
            let [x0, y0, x1, y1] = i.quad;
            assert!(x0 >= corner[0] && x1 <= corner[2], "{:?}", i.quad);
            assert!(y0 >= corner[1] && y1 <= corner[3], "{:?}", i.quad);
        }
    }

    /// The score at each of `steps` (from when, in seconds; each on a
    /// frame), frame by frame: the counter after each frame, and its
    /// reading.
    fn run(steps: &[(f32, u32)], frames: usize) -> Vec<(Odometer, Reading)> {
        let mut odometer = Odometer::default();
        (0..frames)
            .map(|f| {
                let target = steps
                    .iter()
                    .take_while(|(at, _)| frame_at(*at) <= f)
                    .last()
                    .map(|s| s.1);
                odometer.step(target, FRAME);
                (odometer.clone(), odometer.reading())
            })
            .collect()
    }

    /// Whether each drum of `next` is where `last`'s is or further on.
    fn forward(last: &Reading, next: &Reading) -> bool {
        last.columns
            .iter()
            .zip(&next.columns)
            .all(|(a, b)| b.0 >= a.0)
    }

    /// From 12 400 to 186 000 at once (a jackpot): each drum turns forward
    /// to its figure, the low ones spinning laps (fast), the top one a
    /// single turn (slowly), never past where they stop; the score shows
    /// within [`ROLL_LONG`], at rest after its glow.
    #[test]
    fn a_jump_rolls_up_to_the_score_and_never_past_it() {
        let jumped = 0.1;
        let frames = run(&[(0.0, 12_400), (jumped, 186_000)], 120);
        assert_eq!(frames[0].1, Reading::at_rest(12_400));
        let jump = Change {
            from: 12_400,
            to: 186_000,
            age: 0.0,
            length: ROLL_LONG,
        };
        let ends = |k: u32| {
            let from = column(12_400, k);
            (from as f64, (from + jump.turns(k)) as f64)
        };
        let after: Vec<&Reading> = frames[frame_at(jumped)..].iter().map(|f| &f.1).collect();
        for (f, pair) in after.windows(2).enumerate() {
            assert!(forward(pair[0], pair[1]), "frame {f}: {pair:?}");
        }
        // Until it lands, never past where each drum stops.
        for r in &after[..frame_at(ROLL_LONG)] {
            for (k, c) in r.columns.iter().enumerate() {
                assert!(c.0 <= ends(k as u32).1, "{r:?}");
            }
        }
        // Mid-roll each drum is between its two ends, and the number
        // glows.
        let mid = &frames[frame_at(jumped + 0.15)].1;
        for (k, c) in mid.columns.iter().enumerate() {
            let (from, to) = ends(k as u32);
            assert!(c.0 > from && c.0 < to, "column {k}: {c:?}");
        }
        assert!(mid.glow > 0.9 && mid.big > 0.9, "{mid:?}");
        assert!(
            mid.columns[0].1 > 50.0 && mid.columns[5].1 < 10.0,
            "{mid:?}"
        );
        let landed = frame_at(jumped + ROLL_LONG) + 1;
        assert_eq!(frames[landed].1.value(), 186_000);
        let rest = frame_at(jumped + ROLL_LONG + GLOW_OUT) + 4;
        assert_eq!(frames[rest].1, Reading::at_rest(186_000));
        assert!(!frames[rest].0.moving());
    }

    /// A word's points at ×4: only the hundreds turn, from 4 to 8; the
    /// figures that stay stay still.
    #[test]
    fn a_word_turns_only_the_figures_that_change() {
        let frames = run(&[(0.0, 12_400), (0.1, 12_800)], 40);
        let mid = &frames[frame_at(0.2)].1;
        assert_eq!(mid.columns[0], (12_400.0, 0.0));
        assert_eq!(mid.columns[1], (1_240.0, 0.0));
        assert!(
            mid.columns[2].0 > 124.0 && mid.columns[2].0 < 128.0,
            "{mid:?}"
        );
        assert_eq!(mid.columns[3], (12.0, 0.0));
        assert_eq!(frames[39].1.value(), 12_800);
    }

    /// Words scored in quick succession (bigger and bigger, one every
    /// 0.2 s, faster than each rolls in): the changes roll together, none
    /// still rolling [`ROLL_LONG`] after it came, each drum only forward;
    /// the score shown within [`ROLL_LONG`] of the last, at rest in the end.
    #[test]
    fn its_lag_is_bounded_while_points_keep_coming() {
        let steps: Vec<(f32, u32)> = (0..40)
            .map(|k| (k as f32 * 0.2, 1_000 + k * (100 + 97 * k)))
            .collect();
        let frames = run(&steps, 600);
        let last = steps[39];
        for (f, pair) in frames.windows(2).enumerate() {
            let (odometer, reading) = &pair[1];
            assert!(
                odometer.changes.iter().all(|c| c.age < ROLL_LONG),
                "frame {f}"
            );
            if pair[0].1.columns.len() == reading.columns.len() {
                assert!(forward(&pair[0].1, reading), "frame {f}: {pair:?}");
            }
            if f > frame_at(last.0 + ROLL_LONG) {
                assert_eq!(reading.value(), u64::from(last.1), "frame {f}");
            }
        }
        assert_eq!(frames.last().unwrap().1, Reading::at_rest(last.1));
    }

    /// A lower score (sought back), a score gone and come again (another
    /// song), and the first: each shown at once, with nothing moving.
    #[test]
    fn a_seek_back_or_a_new_song_snaps() {
        let mut odometer = Odometer::default();
        odometer.step(Some(5_000), FRAME);
        assert_eq!(odometer.reading(), Reading::at_rest(5_000));
        assert!(!odometer.moving());
        odometer.step(Some(9_000), FRAME);
        assert!(odometer.moving());
        // Mid-roll, sought back.
        odometer.step(Some(4_200), FRAME);
        assert_eq!(odometer.reading(), Reading::at_rest(4_200));
        assert!(!odometer.moving());
        odometer.step(Some(9_000), FRAME);
        odometer.step(None, FRAME);
        assert!(!odometer.moving());
        odometer.step(Some(300), FRAME);
        assert_eq!(odometer.reading(), Reading::at_rest(300));
    }

    /// The picture with the score `points` and nothing else.
    fn scored(points: u32) -> KaraokePicture {
        KaraokePicture {
            bar: 3.0 / 24.0,
            lines: Vec::new(),
            count: None,
            cue: None,
            controls: None,
            start: None,
            before: Vec::new(),
            moved: 1.0,
            score: Some(Score { points, streak: 0 }),
            result: None,
        }
    }

    /// At rest the counter draws the points as the text `points_text`
    /// writes, laid where the score goes, on whole pixels and at its own
    /// size; mid-roll it draws something else, each figure only inside its
    /// slot (nothing above or below the figures' band).
    #[test]
    fn it_draws_the_points_at_rest_and_rolls_inside_its_slots() {
        let area = (0.0, 0.0, 640.0, 360.0);
        let mut stage = Stage::default();
        let frame = |stage: &mut Stage, points: u32| {
            let mut out = Vec::new();
            stage.score(&mut out, &scored(points), area, FRAME);
            out
        };
        let rest = frame(&mut stage, 12_400);
        let (px, margin, _) = sizes(area);
        let text = stage.faces.get(Face::NUMBER);
        let points = crate::view::karaoke::points_text(12_400);
        let tw = text.measure(&points, px);
        // As many runs as figures, side by side from where the text
        // starts to where it ends, each at its size, unturned and whole.
        assert_eq!(rest.len(), 5, "{rest:?}");
        let pad = crate::gui::big_text::COVER_PAD as f32;
        let left = (640.0 - margin - tw).floor() - pad;
        assert!((rest[0].quad[0] - left).abs() <= 1.0, "{:?}", rest[0].quad);
        for i in &rest {
            assert_eq!(i.params[0], kind::TEXT);
            assert_eq!(i.quad[0], i.quad[0].round(), "on whole pixels");
            let (w, h) = (i.quad[2] - i.quad[0], i.quad[3] - i.quad[1]);
            assert!(
                (w - (i.uv[2] - i.uv[0])).abs() < 1e-3 && (h - (i.uv[3] - i.uv[1])).abs() < 1e-3
            );
        }
        assert!(rest.last().unwrap().quad[2] <= 640.0 - margin + pad + 1.0);
        // Mid-roll, frame by frame.
        let mut rolling = rest.clone();
        for _ in 1..10 {
            rolling = frame(&mut stage, 186_000);
        }
        assert!(stage.moving());
        assert_ne!(rolling, rest);
        // The slots, swollen by the glow at most this much about the
        // ink's middle.
        let text = stage.faces.get(Face::NUMBER);
        let (ink_top, ink_bottom) = text.ink_band(FIGURES, px);
        let middle = margin * 0.6 + (ink_top + ink_bottom) / 2.0;
        let reach = ((ink_bottom - ink_top) / 2.0 + 0.1 * px) * 1.15;
        let (lo, hi) = (middle - reach - 1.0, middle + reach + 1.0);
        for i in &rolling {
            assert!(
                i.quad[1] >= lo.floor() && i.quad[3] <= hi.ceil(),
                "a figure at {:?} outside {lo}..{hi}",
                i.quad
            );
        }
    }
}
