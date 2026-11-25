//! The result screen, one more scene of the stage: a diploma (headed
//! "Jodeldiplom") handed out for the song before, filled in beat by beat
//! ([`crate::core::singing::Reveal`]) over the backdrop (the cover's
//! layers, lit by the music), in Karaoke mode's faces, with the stage's
//! light and bursts. The sheet unrolls, its border tinted, its heading and
//! the song's title (landing on its beat) certified as yodelled
//! ([`sheet`]); the entries are written in by hand as their beats come:
//! the points counting up (bumping at each tick, popping as they land),
//! the words hit, the best streak ([`entries`]); each phrase's star
//! sticker pops on, in song order, in its tier's colour (Diva's shining,
//! [`stars`]); the grade is written in and the seal stamped down with a
//! shake, a thump, sparkles and rays behind the sheet, or a star stands
//! for it ([`seal`], whose [`seal::look`] is how each verdict looks);
//! then a new best, a rubber stamp slapped on with confetti, or the best
//! before, calmly; at the end the examiner signs and dates it. The song
//! START plays and START under it throughout. It names nobody: it
//! certifies the song.
//!
//! Everything is a pure function of the result, its reveal's time
//! ([`ResultShown::t`]) and the day the result was first drawn on (read
//! once a result, [`Kept`]): each frame is made afresh from them, the
//! bursts included (each from its beat's time, on the reveal's clock).
//! The diploma is tinted with the cover's palette where its layers have
//! one ([`Tint::of`]), else the stage's own colours. Once settled it
//! twinkles a while, the twinkle calming down as the reveal's time comes
//! to its still point, where it stops ([`crate::core::singing::Motion`]).

use std::collections::HashMap;
use std::sync::Arc;

use super::super::particles::{Kind, Spray};
use super::super::stage_gpu::{kind, Instance, StageFrame};
use super::start::{self, Line};
use super::{disc, pill, rgba, Face, Rgb, Stage, COVER_PAD};
use crate::core::shine::Tier;
use crate::core::singing::{Beat, Shown};
use crate::view::karaoke::{
    best_words, date_words, points_text, KaraokePicture, ResultShown, Start,
};

mod entries;
mod seal;
mod sheet;
mod stars;
mod tint;

use seal::Look;
use sheet::Inks;
pub use tint::Tint;

/// The diploma's printed words: Nunito, lighter than the lyrics'.
const PRINT: Face = Face::Words(650);

/// What the scene draws: over the backdrop, over everything, light (to
/// bloom), and bursts (and those that are light).
#[derive(Default)]
struct Drawn {
    out: Vec<Instance>,
    top: Vec<Instance>,
    lights: Vec<Instance>,
    sprays: Vec<Spray>,
    light_sprays: Vec<Spray>,
}

/// What the result screen keeps between frames: the words of the result
/// shown, made once a result (its date read then, so it never turns over
/// on screen at midnight), and text fitted to a width (forgotten with each
/// new result); and where the day comes from (the system; a test's own).
pub(super) struct Kept {
    written: Option<Written>,
    fitted: HashMap<String, Vec<(Fit, Arc<str>, f32)>>,
    today: fn() -> Option<(i32, u32, u32)>,
}

impl Default for Kept {
    fn default() -> Kept {
        Kept {
            written: None,
            fitted: HashMap::new(),
            today: crate::calendar::today,
        }
    }
}

/// Text fitted: its face, its size and the width it was fitted to (a
/// quarter pixel's steps, whole pixels).
type Fit = (Face, u32, u32);

/// A result's words as the diploma writes them, made once a result: which
/// result, its points, words hit and best streak, the words under its
/// best, the tally's (by tier, the best first) and its date.
struct Written {
    id: u64,
    points: String,
    hits: String,
    streak: String,
    best: String,
    tally: Vec<(Tier, String)>,
    date: Option<String>,
}

impl Written {
    /// `shown`'s words, dated by `today`.
    fn of(shown: &Shown, today: fn() -> Option<(i32, u32, u32)>) -> Written {
        let r = &shown.result;
        let counts = r.tier_counts();
        let tally = Tier::ALL
            .iter()
            .rev()
            .filter(|tier| counts[tier.index()] > 0)
            .map(|&tier| (tier, format!("{} {}", counts[tier.index()], tier.name())))
            .collect();
        Written {
            id: shown.id,
            points: points_text(r.points),
            hits: format!("{} of {}", r.hit, r.words),
            streak: format!("{} in a row", r.streak),
            best: best_words(r),
            tally,
            date: today().map(|(y, m, d)| date_words(y, m, d)),
        }
    }
}

/// Where the scene is and what it is drawn in: the window's area and a
/// hundredth of its height; the sheet's place (left, top, width, height)
/// and its unit (a hundredth of its height, or of its width a little
/// grown where it stands upright); the shake now (the seal stamped down);
/// START's line under it (above the controls' bar); how much it twinkles
/// (1, calming to 0 as it goes still); its inks; and how its verdict
/// looks.
#[derive(Clone, Copy, Debug)]
struct Room {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    u: f32,
    sheet: (f32, f32, f32, f32),
    s: f32,
    shake: (f32, f32),
    start: Line,
    calm: f32,
    inks: Inks,
    look: Look,
}

impl Room {
    /// The room in the window's area `(x, y, w, h)`, START's line at
    /// `start`, for a verdict that looks `look`: the sheet as large as
    /// fits over that line, landscape where the window is wide, upright
    /// where it is tall.
    fn new((x, y, w, h): (f32, f32, f32, f32), start: Line, tint: &Tint, look: Look) -> Room {
        let u = h / 100.0;
        let top = y + 3.0 * u;
        let high = (start.y - 4.4 * u - top).max(10.0);
        let sw = (w * 0.94).min(high * 1.5);
        let sh = high.min(sw * 1.7);
        let sheet = (x + (w - sw) / 2.0, top + (high - sh) / 2.0, sw, sh);
        Room {
            x,
            y,
            w,
            h,
            u,
            sheet,
            s: (sw / 0.85).min(sh) / 100.0,
            shake: (0.0, 0.0),
            start,
            calm: 1.0,
            inks: Inks::of(tint),
            look,
        }
    }

    /// The point `fx` of the sheet's width and `fy` of its height in,
    /// shaken.
    fn at(&self, fx: f32, fy: f32) -> (f32, f32) {
        let (sx, sy, sw, sh) = self.sheet;
        (sx + sw * fx + self.shake.0, sy + sh * fy + self.shake.1)
    }

    /// A share of the sheet's width.
    fn across(&self, share: f32) -> f32 {
        self.sheet.2 * share
    }
}

fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Eased out, past its end a little and back (`x` 0 to 1).
fn back(x: f32) -> f32 {
    let x = clamp01(x) - 1.0;
    1.0 + 2.7 * x * x * x + 1.7 * x * x
}

/// Something landing `age` seconds ago: its scale (growing past its size
/// and back) and how much of it shows; nothing before it lands.
fn land(age: Option<f64>) -> (f32, f32) {
    match age {
        Some(a) => (back(a as f32 / 0.32), clamp01(a as f32 / 0.12)),
        None => (0.0, 0.0),
    }
}

/// How much shows of something fading in over `over` seconds from `age`
/// seconds ago.
fn fade(age: Option<f64>, over: f32) -> f32 {
    age.map_or(0.0, |a| clamp01(a as f32 / over))
}

/// A bump `age` seconds after something hit: 1 and more, settling fast.
fn bump(age: Option<f64>, by: f32) -> f32 {
    age.map_or(1.0, |a| {
        1.0 + by * (-(a as f32) * 9.0).exp() * (a as f32 * 22.0).cos()
    })
}

/// A burst of `kind` at `at` from `start` (the reveal's clock), `size`
/// large, in `rgb`, scattered by `seed`.
fn burst(at: (f32, f32), start: f64, size: f32, (seed, rgb): (u32, Rgb), kind: Kind) -> Spray {
    Spray {
        at,
        half: (0.0, 0.0),
        start: start as f32,
        drift: 0.0,
        size,
        seed,
        rgb,
        kind,
    }
}

/// How text is put down: how much of it is written yet (from its left; 1
/// all of it), how dark a soft shadow under it is (0 none), and how far it
/// is turned round its anchor (radians).
#[derive(Clone, Copy, Debug)]
pub(super) struct Pen {
    share: f32,
    shadow: f32,
    turn: f32,
}

impl Pen {
    /// All of it, flat on the page.
    const FLAT: Pen = Pen {
        share: 1.0,
        shadow: 0.0,
        turn: 0.0,
    };
    /// All of it, a soft shadow under it.
    pub(super) const SHADED: Pen = Pen {
        shadow: 1.0,
        ..Pen::FLAT
    };

    /// Written as far as `share`.
    fn writing(share: f32) -> Pen {
        Pen { share, ..Pen::FLAT }
    }
}

impl Stage {
    /// Text centred across on `c.0`, its capitals' middle on `c.1`
    /// (`anchor` 0 puts its left there, 1 its right), `px` drawn at
    /// `scale`, in `color`, put down with `pen` (turned round `c`); its
    /// width drawn.
    pub(super) fn written(
        &mut self,
        out: &mut Vec<Instance>,
        (face, text): (Face, &str),
        (px, scale): (f32, f32),
        (c, anchor): ((f32, f32), f32),
        color: [f32; 4],
        pen: Pen,
    ) -> f32 {
        if scale <= 0.01 || color[3] <= 0.004 || pen.share <= 0.0 {
            return 0.0;
        }
        let w = self.measure(face, text, px) * scale;
        let top = c.1 - (self.cap_middle(face, px) - COVER_PAD as f32) * scale;
        let left = c.0 - w * anchor;
        let split = if pen.share >= 1.0 {
            f32::INFINITY
        } else {
            left + w * pen.share
        };
        let turned = |i: Option<&mut Instance>| {
            if let Some(i) = i.filter(|_| pen.turn != 0.0) {
                i.params[2] = pen.turn;
                (i.shape[0], i.shape[1]) = c;
            }
        };
        if pen.shadow > 0.0 {
            let shade = (super::Shade::DROP, color[3] * pen.shadow);
            let from = self.shade(out, (face, text), (px, scale), ((left, top), split), shade);
            out[from..].iter_mut().for_each(|i| turned(Some(i)));
        }
        let none = [color[0], color[1], color[2], 0.0];
        let at = (left, top);
        turned(self.text(out, face, text, (px, scale), at, split, (none, color)));
        w
    }

    /// The words of the result `shown`, made once a result (and the
    /// fitted text of the result before forgotten).
    fn words_of(&mut self, shown: &Shown) -> Written {
        match self.kept.written.take() {
            Some(w) if w.id == shown.id => w,
            _ => {
                self.kept.fitted.clear();
                Written::of(shown, self.kept.today)
            }
        }
    }

    /// The result screen of `picture` (none without a result) in the
    /// window's `area` (pixels), above its controls' bar, in `tint`.
    pub fn result(
        &mut self,
        picture: &KaraokePicture,
        tint: &Tint,
        (x, y, w, h): (f32, f32, f32, f32),
    ) -> Option<StageFrame> {
        let shown = picture.result.as_ref()?;
        let t = shown.t;
        let reveal = &shown.shown.reveal;
        let r = &shown.shown.result;
        // The verdict's beat, and how long ago it landed (before: less
        // than none).
        let verdict = reveal
            .find(|b| matches!(b, Beat::Verdict(_)))
            .map(|(at, _)| (t - at) as f32);
        let look = seal::look(r.verdict(), tint);
        let start = Line::of(picture, (x, y, w, h));
        let mut room = Room::new((x, y, w, h), start, tint, look);
        room.shake = seal::shake(&look, verdict, room.s);
        room.calm = clamp01((reveal.still() - t) as f32 / 2.0);
        let words = self.words_of(&shown.shown);
        let mut d = Drawn::default();
        // Over the backdrop: a little darker, a light behind the sheet, and
        // the rays a medal sends out behind it.
        d.out.push(pill(
            (x + w / 2.0, y + h / 2.0),
            (w / 2.0 + 2.0, h / 2.0 + 2.0),
            [0.0, 0.0, 0.02, 0.35],
            [0.0; 4],
            0.0,
            0.0,
            1.0,
        ));
        let lit = fade(Some(t - 0.2), 1.0);
        let (_, _, sw, sh) = room.sheet;
        d.out.push(disc(
            room.at(0.5, 0.5),
            0.0,
            [0.0; 4],
            rgba(tint.glow, 0.16 * lit),
            sw.max(sh) * 0.62,
        ));
        seal::rays(&mut d, &room, verdict, t);
        self.sheet(&mut d, &room, shown);
        self.points(&mut d, &room, shown, &words);
        self.hits(&mut d, &room, shown, &words);
        self.streak(&mut d, &room, shown, &words);
        self.stars(&mut d, &room, shown, &words);
        self.verdict(&mut d, &room, shown, tint, (verdict, &words));
        self.best(&mut d, &room, shown, &words);
        self.signed(&mut d, &room, shown, &words);
        self.next(&mut d, &room, (shown, picture.start), tint);
        self.kept.written = Some(words);
        Some(StageFrame {
            instances: d.out,
            top: d.top,
            uploads: std::mem::take(&mut self.atlas.uploads),
            paths: Vec::new(),
            sprays: d.sprays,
            clock: t as f32,
            lights: d.lights,
            light_sprays: d.light_sprays,
            ..StageFrame::default()
        })
    }

    /// `text` in `face` at `px` fitted to `width`: as it is where it fits,
    /// else smaller (down to a little over half), else cut short with an
    /// ellipsis; and the size it is drawn at. Kept, so a frame fits each
    /// text once a result.
    pub(super) fn fitted(
        &mut self,
        face: Face,
        text: &str,
        px: f32,
        width: f32,
    ) -> (Arc<str>, f32) {
        let fit = (face, (px * 4.0).round() as u32, width.round() as u32);
        let kept = self.kept.fitted.get(text).into_iter().flatten();
        if let Some((_, out, px)) = kept.into_iter().find(|k| k.0 == fit) {
            return (out.clone(), *px);
        }
        let (out, px) = self.fit(face, text, px, width);
        let entry = self.kept.fitted.entry(text.to_owned()).or_default();
        entry.push((fit, out.clone(), px));
        (out, px)
    }

    /// [`Stage::fitted`], made: cut a character at a time, in place.
    fn fit(&mut self, face: Face, text: &str, px: f32, width: f32) -> (Arc<str>, f32) {
        let full = self.measure(face, text, px).max(1.0);
        if full <= width {
            return (text.into(), px);
        }
        let px = (px * (width / full).clamp(0.6, 1.0)).round().max(8.0);
        let mut out = text.to_owned();
        while self.measure(face, &out, px) > width {
            if out.ends_with('…') {
                out.pop();
            }
            if out.pop().is_none() {
                break;
            }
            out.truncate(out.trim_end().len());
            out.push('…');
        }
        (out.into(), px)
    }

    /// The song START plays and START's pill ([`super::start`]), under
    /// the sheet, while the mode waits for START (`start`): there
    /// throughout (fading in as the sheet unrolls; greyed unless the song
    /// can be sung), START pulsing once the reveal is over.
    fn next(
        &mut self,
        d: &mut Drawn,
        room: &Room,
        (shown, start): (&ResultShown, Option<Start>),
        tint: &Tint,
    ) {
        let Some(start) = start else {
            return;
        };
        let (t, reveal) = (shown.t, &shown.shown.reveal);
        let shows = fade(Some(t - 0.3), 0.5);
        let over = reveal.age(Beat::End, t);
        let look = start::Look {
            shows,
            pulse: over.map_or(0.0, |a| 0.5 - 0.5 * (a as f32 * 4.0).cos()) * room.calm,
            lit: 0.55 + 0.45 * fade(over, 0.3),
            ready: start.ready,
        };
        let words = shown.next_words();
        let soft = rgba(tint.soft, 0.85 * shows);
        self.start_pill(&mut d.out, &room.start, Some((&words, soft)), look);
    }
}

/// An instance of `k` round `c`, reaching `reach` each way.
fn shape(k: f32, c: (f32, f32), reach: f32, shape: [f32; 4]) -> Instance {
    Instance {
        quad: [c.0 - reach, c.1 - reach, c.0 + reach, c.1 + reach],
        shape,
        params: [k, 0.0, 0.0, 0.0],
        ..Instance::default()
    }
}

/// A sparkle (light) at `c`, its rays `size` long, turned `turn`.
fn sparkle(c: (f32, f32), size: f32, turn: f32, color: [f32; 4]) -> Instance {
    Instance {
        color,
        ..shape(kind::SPARK, c, size + 1.0, [c.0, c.1, size, turn])
    }
}

/// A star sticker at `c`, `size` out to its points, turned `turn`, in
/// `color`, glowing `glow` far in `halo`, its glint at `glint`.
fn sticker(
    c: (f32, f32),
    (size, turn): (f32, f32),
    color: [f32; 4],
    (glow, halo): (f32, [f32; 4]),
    glint: f32,
) -> Instance {
    let reach = size * 1.15 + glow + 1.0;
    Instance {
        color,
        color2: halo,
        params: [kind::STAR, glow, glint, 0.35 * color[3]],
        ..shape(kind::STAR, c, reach, [c.0, c.1, size, turn])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::shine::Shone;
    use crate::core::singing::{Medal, SongResult};
    use ratatui::layout::Rect;

    pub(super) fn result(medal: Option<Medal>, whole: bool, best: Option<u32>) -> SongResult {
        let phrase = |tier| Shone {
            line: 0,
            end: 1.0,
            value: 0.9,
            top: 0.9,
            tier,
        };
        SongResult {
            title: "A Song".into(),
            points: 12_400,
            hit: 40,
            words: 45,
            streak: 17,
            phrases: [
                Some(Tier::TOP),
                None,
                Some(Tier::named("Good")),
                Some(Tier::named("Amazing")),
            ]
            .map(phrase)
            .to_vec(),
            medal,
            best,
            whole,
            demo: false,
        }
    }

    /// A stage dated 24 November 2026.
    pub(super) fn stage() -> Stage {
        let mut stage = Stage::default();
        stage.kept.today = || Some((2026, 11, 24));
        stage
    }

    /// The picture of a result `id` at `t`, waiting for START.
    fn picture(r: &SongResult, id: u64, t: f64) -> KaraokePicture {
        KaraokePicture {
            area: Rect::new(0, 0, 100, 30),
            bar: Rect::new(0, 25, 100, 5),
            lines: Vec::new(),
            count: None,
            cue: None,
            controls: None,
            start: Some(Start { ready: true }),
            before: Vec::new(),
            moved: 1.0,
            score: None,
            result: Some(ResultShown {
                shown: Arc::new(Shown::new(r.clone(), id, 0)),
                t,
                next: "Next Song".into(),
                again: false,
            }),
        }
    }

    /// The scene at `t` of a result, in a window `size`.
    pub(super) fn frame_in(r: &SongResult, t: f64, size: (f32, f32)) -> StageFrame {
        let area = (0.0, 0.0, size.0, size.1);
        let frame = stage().result(&picture(r, 1, t), &Tint::STAGE, area);
        frame.expect("a result screen")
    }

    /// The scene at `t` in a window of 1600 × 900.
    pub(super) fn frame_at(r: &SongResult, t: f64) -> StageFrame {
        frame_in(r, t, (1600.0, 900.0))
    }

    pub(super) fn count(f: &StageFrame, k: f32) -> usize {
        f.instances
            .iter()
            .chain(&f.top)
            .filter(|i| i.params[0] == k)
            .count()
    }

    /// Everything of a frame inside a window `size`, and finite.
    fn inside(f: &StageFrame, size: (f32, f32), what: &str) {
        for i in f.instances.iter().chain(&f.top) {
            let [x0, y0, x1, y1] = i.quad;
            assert!(
                x0 < size.0 + 10.0 && x1 > -10.0 && y0 < size.1 + 10.0 && y1 > -10.0,
                "{what}: {i:?}"
            );
            let numbers = i.quad.iter().chain(&i.shape).chain(&i.params[2..]);
            assert!(numbers.into_iter().all(|v| v.is_finite()), "{i:?}");
        }
    }

    /// A title too long for the window is cut short, not run off it; one
    /// that fits is kept as it is.
    #[test]
    fn a_long_title_is_cut_short() {
        let mut stage = Stage::default();
        let long = "A very long title ".repeat(12);
        let (text, px) = stage.fitted(Face::LINE, &long, 54.0, 800.0);
        assert!(text.ends_with('…') && stage.measure(Face::LINE, &text, px) <= 800.0);
        assert!(px >= 54.0 * 0.6 - 1.0);
        assert_eq!(stage.fitted(Face::LINE, &long, 54.0, 800.0), (text, px));
        assert_eq!(
            stage.fitted(Face::LINE, "Short", 54.0, 800.0),
            ("Short".into(), 54.0)
        );
    }

    /// Everything lands inside the window, wide or tall; at first the
    /// sheet alone (unrolling), more shows as the reveal goes on; at its
    /// end the seal, a star a phrase, the stamp and the signature.
    #[test]
    fn the_diploma_fills_in_and_stays_in_the_window() {
        let r = result(Some(Medal::Gold), true, Some(9_000));
        let reveal = crate::core::singing::Reveal::of(&r);
        for size in [(1600.0, 900.0), (600.0, 1000.0), (1024.0, 768.0)] {
            let counts: Vec<usize> = [0.0, 1.0, 3.0, reveal.end() + 2.0]
                .iter()
                .map(|&t| {
                    let f = frame_in(&r, t, size);
                    inside(&f, size, &format!("{t}"));
                    f.instances.len() + f.top.len()
                })
                .collect();
            assert!(
                counts.windows(2).all(|c| c[0] < c[1]),
                "{size:?}: {counts:?}"
            );
            let last = frame_in(&r, reveal.end() + 2.0, size);
            assert_eq!(count(&last, kind::PAPER), 1);
            assert_eq!(count(&last, kind::SEAL), 1);
            assert_eq!(count(&last, kind::STAMP), 1);
            assert_eq!(count(&last, kind::SIGNATURE), 1);
            // A sticker a phrase, the tally's three and the heading's two.
            assert_eq!(count(&last, kind::STAR), 4 + 3 + 2);
            assert!(last.sprays.iter().any(|s| s.kind == Kind::Confetti));
        }
    }

    /// The heading ("Jodeldiplom") fits between its two stars, and they on
    /// the sheet, however wide or tall the window: the stars clear of its
    /// words either side, inside the sheet's border.
    #[test]
    fn the_heading_fits_between_its_stars() {
        let r = result(Some(Medal::Gold), true, None);
        for size in [
            (1600.0, 900.0),
            (900.0, 1400.0),
            (500.0, 900.0),
            (1024.0, 768.0),
        ] {
            let f = frame_in(&r, 3.0, size);
            let all = || f.instances.iter().chain(&f.top);
            let paper = all().find(|i| i.params[0] == kind::PAPER).unwrap();
            let [cx, cy, hx, hy] = paper.shape;
            // The room the frame was drawn in, and the heading's size there.
            let area = (0.0, 0.0, size.0, size.1);
            let line = Line::of(&picture(&r, 1, 3.0), area);
            let look = seal::look(r.verdict(), &Tint::STAGE);
            let room = Room::new(area, line, &Tint::STAGE, look);
            assert_eq!(room.sheet.2, 2.0 * hx, "{size:?}: the same sheet");
            let mut stage = stage();
            let heading = crate::view::karaoke::DIPLOMA.heading;
            assert_eq!(heading, "Jodeldiplom");
            let hp = sheet::heading_px(&mut stage, &room);
            assert!(hp >= 16.0, "{size:?}: {hp}");
            let half = stage.measure(Face::SHOUT, heading, hp) / 2.0;
            assert!(
                half <= room.across(sheet::HEADING_WIDE) / 2.0 + 0.5,
                "whole"
            );
            let row = cy - hy + 0.135 * 2.0 * hy;
            let stars: Vec<[f32; 4]> = all()
                .filter(|i| i.params[0] == kind::STAR && (i.shape[1] - row).abs() < 1.0)
                .map(|i| i.shape)
                .collect();
            assert_eq!(stars.len(), 2, "{size:?}");
            for [x, _, reach, _] in stars {
                let out = (x - cx).abs();
                assert!(out - reach * 1.15 > half, "{size:?}: {out} {reach} {half}");
                assert!(
                    out + reach * 1.15 < hx * 0.93,
                    "{size:?}: {out} {reach} {hx}"
                );
            }
        }
    }

    /// No medal: a star where the seal goes, no confetti without a new
    /// best; not sung whole: no seal, no new best, however high.
    #[test]
    fn no_medal_and_not_whole_draw_a_star() {
        for r in [result(None, true, Some(50_000)), result(None, false, None)] {
            let reveal = crate::core::singing::Reveal::of(&r);
            let f = frame_at(&r, reveal.end() + 2.0);
            assert_eq!(count(&f, kind::SEAL), 0);
            assert_eq!(count(&f, kind::STAMP), 0);
            assert_eq!(count(&f, kind::STAR), 4 + 3 + 2 + 1);
            assert!(!f.sprays.iter().any(|s| s.kind == Kind::Confetti));
        }
    }

    /// Nothing scored (no points, no words, no phrases): the diploma all
    /// the same, in the window and finite throughout: a sleepy star where
    /// the seal goes and a "Zzz" stamp, no stickers, no seal, no confetti.
    #[test]
    fn a_silent_result_draws_a_star_and_no_gems() {
        let r = SongResult {
            points: 0,
            hit: 0,
            words: 0,
            streak: 0,
            phrases: Vec::new(),
            ..result(None, false, None)
        };
        let reveal = crate::core::singing::Reveal::of(&r);
        for t in [0.0, 0.6, 1.0, 2.0, reveal.end() + 2.0] {
            inside(&frame_at(&r, t), (1600.0, 900.0), &format!("{t}"));
        }
        let f = frame_at(&r, reveal.end() + 2.0);
        // The heading's two stars and the sleepy one.
        assert_eq!((count(&f, kind::STAR), count(&f, kind::SEAL)), (2 + 1, 0));
        assert_eq!(count(&f, kind::STAMP), 1);
        assert!(!f.sprays.iter().any(|s| s.kind == Kind::Confetti));
    }

    /// The day is read once a result: the same however often it is drawn
    /// (even past midnight), the next result's its own.
    #[test]
    fn the_date_is_read_once_a_result() {
        let mut stage = stage();
        let r = result(Some(Medal::Gold), true, None);
        let area = (0.0, 0.0, 1600.0, 900.0);
        let date = |stage: &Stage| stage.kept.written.as_ref().and_then(|w| w.date.clone());
        stage.result(&picture(&r, 1, 1.0), &Tint::STAGE, area);
        assert_eq!(date(&stage).as_deref(), Some("24 November 2026"));
        stage.kept.today = || Some((2026, 11, 25));
        stage.result(&picture(&r, 1, 9.0), &Tint::STAGE, area);
        assert_eq!(date(&stage).as_deref(), Some("24 November 2026"));
        stage.result(&picture(&r, 2, 1.0), &Tint::STAGE, area);
        assert_eq!(date(&stage).as_deref(), Some("25 November 2026"));
    }
}
