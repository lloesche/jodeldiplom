//! The score's counter, top right: points added roll in as on an
//! odometer. Each column of figures is a drum turning forward from the
//! figure it showed to the one it shows next, the figure going out sliding
//! up out of its cell as the one coming in slides up into it; a column
//! whose figure stays stays still. On a big jump the columns under the
//! jump's top spin whole laps more (up to [`LAPS`]), a blur, and the number
//! glows (and swells a little, more for a big jump) while it counts.
//!
//! Each change rolls in on its own, eased out, over [`ROLL`] (longer for a
//! bigger jump, up to [`ROLL_LONG`]), alongside those still rolling: so the
//! counter is never more than that behind the score, never past it, and
//! each drum only ever turns forward. A score lower than the one shown (a
//! seek back, another song) or one just come is shown at once. The counter
//! is stepped by the frames' own time, apart from the core's state: what
//! it counts to is the core's score.
//!
//! The glow is the figures' colour mixed towards [`HOT`] here, in the
//! pixmap; it may later come from the effects library on the GPU instead.

use super::TextOp;
use crate::gui::big_text::BigText;
use crate::gui::raster::{mix, PixelRect};

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
/// The figures, whose ink the cells are made round.
const FIGURES: &str = "0123456789";

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
pub struct Odometer {
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
    pub(super) fn reading(&self) -> Reading {
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

/// The pieces of text `reading` is drawn as at `px` (its glyphs in `text`,
/// the score's face), its right edge at `right` and its top at `top`, in
/// `color`: in each column the figure going out and the one coming in (in
/// several copies along its motion where it spins fast), each kept to its
/// column's cell. At rest it is the points as [`points_text`] writes them,
/// drawn as that text is.
///
/// [`points_text`]: crate::view::karaoke::points_text
pub(super) fn ops(
    text: &mut BigText,
    reading: &Reading,
    (right, top): (f32, f32),
    px: f32,
    color: [u8; 3],
    out: &mut Vec<TextOp>,
) {
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
    // The cells: from a little above the figures' ink to a little below,
    // a turn as tall as a cell. The number swells from its right edge and
    // the middle of its ink.
    let (ink_top, ink_bottom) = text.ink_band(FIGURES, px);
    let room = 0.1 * px;
    let middle = (ink_top + ink_bottom) / 2.0;
    let pop = 1.0 + reading.glow * (0.04 + 0.1 * reading.big);
    let size = px * pop;
    let y = top + middle * (1.0 - pop);
    let cell_top = y + (ink_top - room) * pop;
    let pitch = (ink_bottom - ink_top + 2.0 * room) * pop;
    let color = mix(color, HOT, 0.6 * reading.glow);
    let mut pen = right - width;
    for k in (0..n).rev() {
        let x = pen + (pen - right) * (pop - 1.0);
        let side = 0.06 * size;
        let cell: PixelRect = (
            (x - side).max(0.0).floor() as usize,
            cell_top.max(0.0).floor() as usize,
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
                out.push(TextOp {
                    text: (digit % 10).to_string(),
                    face: text.face(),
                    base: px,
                    px: size,
                    at: (x, y + off * pitch),
                    fill_x: f32::MAX,
                    colors: (color, color),
                    // Fading away from the middle of its cell, as on a drum
                    // curving away.
                    alpha: alpha * (1.0 - off * off),
                    sung: None,
                    clip: Some(cell),
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
}

#[cfg(test)]
mod tests {
    use super::super::{paint, Backdrop, Layer};
    use super::*;
    use crate::gui::big_text::{Face, Faces};
    use crate::gui::raster::Target;
    use crate::view::karaoke::{KaraokePicture, Score};
    use std::time::{Duration, Instant};

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
            area: ratatui::layout::Rect::new(0, 0, 80, 24),
            bar: ratatui::layout::Rect::new(0, 21, 80, 3),
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
    /// writes, pixel for pixel; mid-roll it draws only inside its cells
    /// (nothing above or below the figures' band), something different.
    #[test]
    fn it_draws_the_points_at_rest_and_rolls_inside_its_cells() {
        let (w, h) = (640, 360);
        let backdrop = Backdrop::new(None, w, h);
        let frame = |layer: &mut Layer, points: u32| {
            let mut pixels = vec![0u32; w * h];
            let mut target = Target {
                pixels: &mut pixels,
                width: w,
                height: h,
            };
            let area = (0, 0, w, h);
            let mut faces = Faces::default();
            paint(
                &mut target,
                &mut faces,
                layer,
                &backdrop,
                &scored(points),
                area,
                None,
                true,
            );
            pixels
        };
        let t0 = Instant::now();
        let at = |f: usize| t0 + Duration::from_secs_f32(f as f32 * FRAME);
        let mut layer = Layer::default();
        layer.at(at(0));
        let rest = frame(&mut layer, 12_400);
        // The text itself, drawn where the score's is.
        let mut faces = Faces::default();
        let text = faces.get(Face::NUMBER);
        let (px, margin, _) = super::super::score_sizes((0.0, 0.0, w as f32, h as f32));
        let points = crate::view::karaoke::points_text(12_400);
        let tw = text.measure_from(&points, px, px);
        let mut plain = backdrop.pixels.clone();
        let mut target = Target {
            pixels: &mut plain,
            width: w,
            height: h,
        };
        let hit = super::super::rgb(crate::view::karaoke::HIT);
        text.draw(
            &mut target,
            &points,
            px,
            px,
            (w as f32 - margin - tw, margin * 0.6),
            f32::MAX,
            (hit, hit),
            1.0,
            (0, 0, w, h),
        );
        assert_eq!(rest, plain);
        // Mid-roll, painted frame by frame.
        let mut rolling = rest.clone();
        for f in 1..10 {
            layer.at(at(f));
            rolling = frame(&mut layer, 186_000);
        }
        assert!(layer.odometer.moving());
        assert_ne!(rolling, rest);
        // The cells, swollen by the glow at most this much about the
        // ink's middle.
        let (ink_top, ink_bottom) = text.ink_band(FIGURES, px);
        let middle = margin * 0.6 + (ink_top + ink_bottom) / 2.0;
        let reach = ((ink_bottom - ink_top) / 2.0 + 0.1 * px) * 1.15;
        let (lo, hi) = (middle - reach - 1.0, middle + reach + 1.0);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if rolling[i] != backdrop.pixels[i] {
                    assert!(
                        (y as f32) >= lo.floor() && (y as f32) <= hi.ceil(),
                        "a pixel at {x}, {y} outside {lo}..{hi}"
                    );
                }
            }
        }
    }
}
