//! The result screen, one more scene of the stage: how the song before
//! was sung, revealed beat by beat ([`crate::core::singing::Reveal`]) over
//! the blurred cover, in Karaoke mode's faces, with the stage's light and
//! bursts. The title drops in; the points count up (bumping at each tick,
//! popping as they land); the words hit fill their bar; the best streak
//! pops in its badge; each phrase's gem pops in, in song order, in its
//! tier's colour (Diva's shining); the medal is stamped down with a shake,
//! a shock ring, sparkles and its rays ([`medal`]), or a star stands for
//! it; then a new best with confetti, or the best before, calmly; the next
//! song and START at the bottom throughout.
//!
//! Everything is a pure function of the result and its reveal's time
//! ([`ResultShown::t`]): each frame is made afresh from them, the bursts
//! included (each from its beat's time, on the reveal's clock). The
//! screen is tinted with the cover's palette where its layers have one
//! ([`Tint::of`]), else the stage's own colours. Once settled it twinkles
//! a while, the twinkle calming down as the reveal's time comes to its
//! still point, where it stops ([`crate::core::singing::Motion`]).

use super::super::particles::{Kind, Spray};
use super::super::rainbow::{in_rainbow, Glow};
use super::super::stage_gpu::{kind, Instance, StageFrame};
use super::{
    disc, fill, pill, rgba, Face, Rgb, Stage, COVER_PAD, GOLD, GOLD_DEEP, GOLD_TOP, WHITE,
};
use crate::core::singing::{Beat, SongResult, WORDS_FILL};
use crate::view::karaoke::{best_words, points_text, KaraokePicture, ResultShown};

mod medal;
mod tint;

pub use tint::Tint;

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

/// Where the scene is: the window's area, a hundredth of its height, the
/// shake now (the medal stamped down), the room above the controls, and
/// how much it twinkles (1, calming to 0 as it goes still).
#[derive(Clone, Copy, Debug)]
struct Room {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    u: f32,
    shake: (f32, f32),
    bottom: f32,
    calm: f32,
}

impl Room {
    /// The point `sx` of the width and `sy` of the height in, shaken.
    fn at(&self, sx: f32, sy: f32) -> (f32, f32) {
        (
            self.x + self.w * sx + self.shake.0,
            self.y + self.h * sy + self.shake.1,
        )
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

/// Eased out, slowing to its end.
fn ease_out(x: f32) -> f32 {
    1.0 - (1.0 - clamp01(x)).powi(3)
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

impl Stage {
    /// Text centred across on `c.0`, its capitals' middle on `c.1`
    /// (`anchor` 0 puts its left there, 1 its right), `px` drawn at
    /// `scale`, in `color`, with a soft shadow under it where `shadow`;
    /// its width drawn.
    #[allow(clippy::too_many_arguments)]
    fn words(
        &mut self,
        out: &mut Vec<Instance>,
        (face, text): (Face, &str),
        (px, scale): (f32, f32),
        c: (f32, f32),
        anchor: f32,
        color: [f32; 4],
        shadow: bool,
    ) -> f32 {
        if scale <= 0.01 || color[3] <= 0.004 {
            return 0.0;
        }
        let w = self.measure(face, text, px) * scale;
        let top = c.1 - (self.cap_middle(face, px) - COVER_PAD as f32) * scale;
        let left = c.0 - w * anchor;
        if shadow {
            let dark = [0.02, 0.01, 0.05, 0.55 * color[3]];
            let drop = (px * 0.05).max(1.0) * scale;
            self.text(
                out,
                face,
                text,
                (px, scale),
                (left, top + drop),
                f32::NEG_INFINITY,
                (dark, dark),
            );
        }
        self.text(
            out,
            face,
            text,
            (px, scale),
            (left, top),
            f32::NEG_INFINITY,
            (color, color),
        );
        w
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
        let u = h / 100.0;
        let rows = f32::from(picture.area.height.max(1));
        let bottom = picture.controls.map_or(y + h, |c| {
            y + h * f32::from(c.y.saturating_sub(picture.area.y)) / rows
        });
        let room = Room {
            x,
            y,
            w,
            h,
            u,
            shake: medal::shake(r.verdict(), verdict, u),
            bottom,
            calm: clamp01((reveal.still() - t) as f32 / 2.0),
        };
        let mut d = Drawn::default();
        // Over the cover: a little darker, a card for the numbers, a light
        // behind the medal and the points.
        d.out.push(pill(
            (x + w / 2.0, y + h / 2.0),
            (w / 2.0 + 2.0, h / 2.0 + 2.0),
            [0.0, 0.0, 0.02, 0.3],
            [0.0; 4],
            0.0,
            0.0,
            1.0,
        ));
        let card = fade(Some(t), 0.35);
        let (cx, cy) = room.at(0.5, 0.42);
        d.out.push(pill(
            (cx, cy),
            ((w * 0.45).min(h * 0.95), h * 0.295),
            rgba(tint.card, 0.5 * card),
            rgba(WHITE, 0.1 * card),
            0.0,
            1.5,
            4.0 * u,
        ));
        let lit = fade(Some(t - 0.2), 1.0);
        d.out.push(disc(
            room.at(0.73, 0.3),
            0.0,
            [0.0; 4],
            rgba(tint.glow, 0.22 * lit),
            42.0 * u,
        ));
        d.out.push(disc(
            room.at(0.34, 0.27),
            0.0,
            [0.0; 4],
            rgba(GOLD, 0.1 * lit),
            30.0 * u,
        ));
        self.title(&mut d, &room, r, (reveal.when(Beat::Title), t));
        self.points(&mut d, &room, shown);
        self.hits(&mut d, &room, r, reveal.age(Beat::Words, t), t);
        self.streak(&mut d, &room, r, reveal.age(Beat::Streak, t), t);
        self.gems(&mut d, &room, shown);
        self.verdict(&mut d, &room, shown, tint, verdict);
        self.best(&mut d, &room, shown);
        self.next(&mut d, &room, shown, tint);
        Some(StageFrame {
            instances: d.out,
            top: d.top,
            uploads: std::mem::take(&mut self.atlas.uploads),
            paths: Vec::new(),
            sprays: d.sprays,
            clock: t as f32,
            lights: d.lights,
            light_sprays: d.light_sprays,
        })
    }

    /// `text` in `face` at `px` fitted to `width`: as it is where it fits,
    /// else smaller (down to a little over half), else cut short with an
    /// ellipsis; and the size it is drawn at.
    fn fitted(&mut self, face: Face, text: &str, px: f32, width: f32) -> (String, f32) {
        let full = self.measure(face, text, px).max(1.0);
        let px = (px * (width / full).clamp(0.6, 1.0)).round().max(8.0);
        let mut chars: Vec<char> = text.chars().collect();
        let mut out = text.to_owned();
        while self.measure(face, &out, px) > width && !chars.is_empty() {
            chars.pop();
            let kept: String = chars.iter().collect();
            out = format!("{}…", kept.trim_end());
        }
        (out, px)
    }

    /// The song's title, dropping in from above and bouncing, a sparkle off
    /// each end as it lands.
    fn title(&mut self, d: &mut Drawn, room: &Room, r: &SongResult, (at, t): (Option<f64>, f64)) {
        let Some(at) = at.filter(|&at| t >= at) else {
            return;
        };
        let a = t - at;
        let u = room.u;
        let px = (6.0 * u).round().max(14.0);
        let (title, px) = self.fitted(Face::LINE, &r.title, px, room.w * 0.84);
        let fall = 1.0 - ease_out(a as f32 / 0.45);
        let wobble = (-(a as f32) * 7.0).exp() * (a as f32 * 16.0).sin() * 0.6 * u;
        let (cx, cy) = room.at(0.5, 0.075);
        let cy = cy - 6.0 * u * fall * fall + wobble;
        let alpha = clamp01(a as f32 / 0.15);
        let tw = self.words(
            &mut d.out,
            (Face::LINE, &title),
            (px, 1.0),
            (cx, cy),
            0.5,
            rgba(WHITE, alpha),
            true,
        );
        // A sparkle off each end as it lands.
        for (k, side) in [-1.0f32, 1.0].into_iter().enumerate() {
            d.light_sprays.push(Spray {
                at: (cx + side * (tw / 2.0 + px * 0.3), cy),
                half: (0.0, 0.0),
                start: (at + 0.4) as f32,
                drift: -side * px * 0.3,
                size: px * 0.45,
                seed: 11 + k as u32,
                rgb: GOLD_TOP,
                kind: Kind::Scored,
            });
        }
    }

    /// The points, counting up (each tick a bump), popping as they land
    /// with a burst of sparks; "POINTS" over them.
    fn points(&mut self, d: &mut Drawn, room: &Room, shown: &ResultShown) {
        let (t, reveal) = (shown.t, &shown.shown.reveal);
        let (from, to) = reveal.count();
        let shows = fade(Some(t - from + 0.15), 0.15);
        if shows <= 0.0 {
            return;
        }
        let u = room.u;
        let (cx, cy) = room.at(0.34, 0.27);
        let label = (Face::SHOUT, "POINTS");
        let soft = rgba(GOLD_TOP, 0.75 * shows);
        let small = (2.6 * u).round().max(9.0);
        self.words(
            &mut d.out,
            label,
            (small, 1.0),
            (cx, cy - 9.5 * u),
            0.5,
            soft,
            false,
        );
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
        let value = points_text(reveal.points_at(t));
        let px = (13.0 * u).round().max(20.0);
        let px = px.min((room.w * 0.36 / (value.chars().count().max(4) as f32 * 0.6)).round());
        // A glow behind them as they land.
        let glow = landed.map_or(0.0, |a| (-(a as f32) * 2.5).exp());
        d.lights.push(disc(
            (cx, cy),
            0.0,
            [0.0; 4],
            rgba(GOLD, 0.35 * glow),
            px * 1.5,
        ));
        self.digits(&mut d.out, &value, (px, scale), (cx, cy), shows);
        d.light_sprays.push(Spray {
            at: (cx, cy),
            half: (0.0, 0.0),
            start: to as f32,
            drift: px * 0.3,
            size: px * 0.5,
            seed: 23,
            rgb: GOLD,
            kind: Kind::Scored,
        });
    }

    /// `value` in the numbers' face, figure by figure (each kept once in
    /// the atlas, whatever the count), centred on `c`, gold.
    fn digits(
        &mut self,
        out: &mut Vec<Instance>,
        value: &str,
        (px, scale): (f32, f32),
        c: (f32, f32),
        alpha: f32,
    ) {
        let widths: Vec<f32> = value
            .chars()
            .map(|ch| self.measure(Face::SHOUT, &ch.to_string(), px) * scale)
            .collect();
        let mut left = c.0 - widths.iter().sum::<f32>() / 2.0;
        for (ch, cw) in value.chars().zip(widths) {
            let gold = rgba(GOLD_TOP, alpha);
            let text = (Face::SHOUT, ch.to_string());
            self.words(
                out,
                (text.0, &text.1),
                (px, scale),
                (left, c.1),
                0.0,
                gold,
                true,
            );
            left += cw;
        }
    }

    /// The words hit: their bar filling to its share, the count rising
    /// with it, a burst at its front as it is full.
    fn hits(&mut self, d: &mut Drawn, room: &Room, r: &SongResult, age: Option<f64>, t: f64) {
        let Some(a) = age else { return };
        let u = room.u;
        let (cx, cy) = room.at(0.34, 0.41);
        let half = ((room.w * 0.17).min(36.0 * u), 2.3 * u);
        let (grow, shows) = land(Some(a));
        let share = r.hit as f32 / r.words.max(1) as f32;
        let filled = share * ease_out(a as f32 / WORDS_FILL as f32);
        let scale = 0.9 + 0.1 * grow;
        let half = (half.0 * scale, half.1 * scale);
        d.out.push(pill(
            (cx, cy),
            half,
            [0.02, 0.02, 0.06, 0.55 * shows],
            rgba(WHITE, 0.25 * shows),
            0.0,
            1.5,
            0.0,
        ));
        let inner = (half.0 - 3.0, half.1 - 3.0);
        let front = cx - inner.0 + 2.0 * inner.0 * filled;
        if let Some(f) = fill(
            (cx, cy),
            inner,
            (cx - inner.0, front),
            rgba(GOLD_TOP, shows),
            rgba(GOLD_DEEP, shows),
            inner.1 * 0.8,
        ) {
            d.out.push(f);
        }
        let small = (2.4 * u).round().max(9.0);
        let soft = rgba(WHITE, 0.7 * shows);
        let (lx, rx) = (cx - half.0, cx + half.0);
        let top = cy - half.1 - 2.4 * u;
        self.words(
            &mut d.out,
            (Face::SHOUT, "WORDS"),
            (small, 1.0),
            (lx, top),
            0.0,
            soft,
            false,
        );
        let count = format!(
            "{} of {}",
            (r.hit as f32 * filled / share.max(1e-6)).round() as u32,
            r.words
        );
        let bright = rgba(WHITE, 0.95 * shows);
        self.words(
            &mut d.out,
            (Face::NUMBER, &count),
            (small * 1.15, 1.0),
            (rx, top),
            1.0,
            bright,
            false,
        );
        // The front glinting as it moves; a burst as it is full.
        if (a as f32) < WORDS_FILL as f32 + 0.2 && filled > 0.0 {
            let k = 1.0 - clamp01((a as f32 - WORDS_FILL as f32) / 0.2);
            d.lights.push(disc(
                (front, cy),
                half.1 * 0.3,
                rgba(WHITE, k),
                rgba(GOLD, 0.6 * k),
                half.1 * 2.0,
            ));
        }
        if r.hit > 0 {
            d.light_sprays.push(Spray {
                at: (front.max(cx - inner.0), cy),
                half: (0.0, 0.0),
                start: (t - a + WORDS_FILL) as f32,
                drift: half.1,
                size: half.1 * 1.6,
                seed: 31,
                rgb: GOLD,
                kind: Kind::Scored,
            });
        }
    }

    /// The best streak: a badge popping in with a burst of stars and notes.
    fn streak(&mut self, d: &mut Drawn, room: &Room, r: &SongResult, age: Option<f64>, t: f64) {
        let Some(a) = age else { return };
        let u = room.u;
        let (cx, cy) = room.at(0.34, 0.52);
        let (scale, shows) = land(Some(a));
        let px = (3.6 * u).round().max(10.0);
        let text = format!("{} in a row", r.streak);
        let tw = self.measure(Face::NUMBER, &text, px);
        let half = ((tw / 2.0 + px * 0.8) * scale, px * 0.85 * scale);
        let small = (2.2 * u).round().max(8.0);
        let soft = rgba(WHITE, 0.7 * shows);
        let label = (Face::SHOUT, "BEST STREAK");
        self.words(
            &mut d.out,
            label,
            (small, 1.0),
            (cx, cy - half.1 - 2.2 * u),
            0.5,
            soft,
            false,
        );
        d.out.push(pill(
            (cx, cy),
            half,
            rgba(CORAL_DEEP, 0.95 * shows),
            rgba([1.0, 0.62, 0.45], 0.8 * shows),
            px * 0.5,
            2.0,
            0.0,
        ));
        let white = rgba(WHITE, shows);
        self.words(
            &mut d.out,
            (Face::NUMBER, &text),
            (px, scale),
            (cx, cy),
            0.5,
            white,
            true,
        );
        d.sprays.push(Spray {
            at: (cx, cy - half.1),
            half: (0.0, 0.0),
            start: (t - a) as f32,
            drift: 0.0,
            size: px * 1.1,
            seed: 41,
            rgb: WHITE,
            kind: Kind::StepUp,
        });
    }

    /// A new best (popping in the rainbow, confetti raining, its words
    /// under it), or the best before (calmly).
    fn best(&mut self, d: &mut Drawn, room: &Room, shown: &ResultShown) {
        let (t, reveal, r) = (shown.t, &shown.shown.reveal, &shown.shown.result);
        let u = room.u;
        let below = best_words(r);
        if let Some(a) = reveal.age(Beat::NewBest, t) {
            let (cx, cy) = room.at(0.5, 0.755);
            let (scale, shows) = land(Some(a));
            let px = (7.0 * u).round().max(16.0);
            let tilt = (-(a as f32) * 4.0).exp() * (a as f32 * 12.0).sin() * 0.8 * u;
            let banner = (Face::SHOUT, "NEW BEST!");
            let at = (cx, cy + tilt);
            let gold = rgba(GOLD_TOP, shows);
            let start = d.top.len();
            self.words(&mut d.top, banner, (px, scale * 1.08), at, 0.5, gold, true);
            let drawn = d.top.len() > start;
            if let Some(text) = d.top.last_mut().filter(|_| drawn) {
                let shine = Glow {
                    level: 1.0,
                    from: 1.0,
                    since: 0.0,
                };
                in_rainbow(text, &shine);
            }
            let small = (2.6 * u).round().max(9.0);
            let soft = rgba(WHITE, 0.85 * fade(Some(a - 0.25), 0.3));
            self.words(
                &mut d.top,
                (Face::NUMBER, &below),
                (small, 1.0),
                (cx, cy + px * 0.72),
                0.5,
                soft,
                true,
            );
            let start = (t - a) as f32;
            for k in 0..5u32 {
                d.sprays.push(super::super::confetti::shower(
                    (room.x, room.y),
                    (room.w, room.h),
                    1.8 * u,
                    f64::from(start + 0.12 * k as f32),
                    ([1.0, 0.9, 0.7], 101 + k),
                ));
            }
            for (k, side) in [-1.0f32, 1.0].into_iter().enumerate() {
                d.sprays.push(Spray {
                    at: (cx + side * px * 2.6, cy),
                    half: (0.0, 0.0),
                    start,
                    drift: 0.0,
                    size: px * 0.9,
                    seed: 51 + k as u32,
                    rgb: WHITE,
                    kind: Kind::StepUp,
                });
            }
        } else if let Some((at, _)) = reveal.find(|b| matches!(b, Beat::Best(_))) {
            let shows = fade(Some(t - at), 0.5);
            let px = (3.0 * u).round().max(10.0);
            let soft = rgba(WHITE, 0.72 * shows);
            self.words(
                &mut d.out,
                (Face::NUMBER, &below),
                (px, 1.0),
                room.at(0.5, 0.775),
                0.5,
                soft,
                true,
            );
        }
    }

    /// The song coming next and START, at the bottom (above the controls'
    /// bar): there throughout, START pulsing once the reveal is over.
    fn next(&mut self, d: &mut Drawn, room: &Room, shown: &ResultShown, tint: &Tint) {
        let (t, reveal) = (shown.t, &shown.shown.reveal);
        let u = room.u;
        let px = (2.9 * u).round().max(10.0);
        let cy = (room.y + room.h * 0.915).min(room.bottom - 3.2 * u);
        let shows = fade(Some(t - 0.3), 0.5);
        let start = (Face::SHOUT, "START");
        let sw = self.measure(start.0, start.1, px);
        let room_for = room.w * 0.92 - sw - px * 2.4;
        let (text, _) = self.fitted(Face::NUMBER, &shown.next_words(), px, room_for);
        let tw = self.measure(Face::NUMBER, &text, px);
        let gap = px * 1.4;
        let left = room.x + room.w / 2.0 - (tw + gap + sw + px) / 2.0;
        let soft = rgba(tint.soft, 0.85 * shows);
        self.words(
            &mut d.out,
            (Face::NUMBER, &text),
            (px, 1.0),
            (left, cy),
            0.0,
            soft,
            true,
        );
        let over = reveal.age(Beat::End, t);
        let pulse = over.map_or(0.0, |a| 0.5 - 0.5 * (a as f32 * 4.0).cos()) * room.calm;
        let scale = 1.0 + 0.06 * pulse;
        let c = (left + tw + gap + (sw + px) / 2.0, cy);
        let lit = 0.55 + 0.45 * fade(over, 0.3);
        d.out.push(pill(
            c,
            ((sw / 2.0 + px * 0.5) * scale, px * 0.8 * scale),
            rgba(GOLD_DEEP, 0.9 * shows * lit),
            rgba(GOLD_TOP, (0.5 + 0.5 * pulse) * shows),
            0.0,
            1.5,
            0.0,
        ));
        let white = rgba(WHITE, shows);
        self.words(&mut d.out, start, (px, scale), c, 0.5, white, true);
    }
}

/// The streak's badge: a deep coral.
const CORAL_DEEP: Rgb = [0.86, 0.32, 0.3];

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::shine::{Shone, Tier};
    use crate::core::singing::Medal;
    use crate::core::singing::Shown;
    use ratatui::layout::Rect;

    pub(super) fn result(medal: Option<Medal>, whole: bool, best: Option<u32>) -> SongResult {
        let phrase = |tier| Shone {
            line: 0,
            end: 1.0,
            value: 0.9,
            diva: 0.9,
            tier,
        };
        SongResult {
            title: "A Song".into(),
            points: 12_400,
            hit: 40,
            words: 45,
            streak: 17,
            phrases: [
                Some(Tier::Diva),
                None,
                Some(Tier::Good),
                Some(Tier::Amazing),
            ]
            .map(phrase)
            .to_vec(),
            medal,
            best,
            whole,
        }
    }

    /// The scene at `t` of a result, in a window of 1600 × 900 (its
    /// controls' bar the bottom 3 of its 30 rows).
    pub(super) fn frame_at(r: &SongResult, t: f64) -> StageFrame {
        let picture = KaraokePicture {
            area: Rect::new(0, 0, 100, 30),
            lines: Vec::new(),
            count: None,
            cue: None,
            controls: Some(Rect::new(0, 27, 100, 3)),
            start: None,
            before: Vec::new(),
            moved: 1.0,
            score: None,
            result: Some(ResultShown {
                shown: std::sync::Arc::new(Shown::new(r.clone(), 1, 0)),
                t,
                next: "Next Song".into(),
                again: false,
            }),
        };
        let area = (0.0, 0.0, 1600.0, 900.0);
        let frame = Stage::default().result(&picture, &Tint::STAGE, area);
        frame.expect("a result screen")
    }

    /// A title too long for the window is cut short, not run off it.
    #[test]
    fn a_long_title_is_cut_short() {
        let mut stage = Stage::default();
        let long = "A very long title ".repeat(12);
        let (text, px) = stage.fitted(Face::LINE, &long, 54.0, 800.0);
        assert!(text.ends_with('…') && stage.measure(Face::LINE, &text, px) <= 800.0);
        assert!(px >= 54.0 * 0.6 - 1.0);
        assert_eq!(
            stage.fitted(Face::LINE, "Short", 54.0, 800.0),
            ("Short".into(), 54.0)
        );
    }

    /// Everything lands inside the window; before the title nothing of the
    /// scene but its backdrop; more shows as the reveal goes on.
    #[test]
    fn the_scene_grows_as_it_is_revealed_and_stays_in_the_window() {
        let r = result(Some(Medal::Gold), true, Some(9_000));
        let reveal = crate::core::singing::Reveal::of(&r);
        let counts: Vec<usize> = [0.0, 1.0, 3.0, reveal.end() + 2.0]
            .iter()
            .map(|&t| {
                let f = frame_at(&r, t);
                for i in f.instances.iter().chain(&f.top) {
                    let [x0, y0, x1, y1] = i.quad;
                    assert!(
                        x0 < 1610.0 && x1 > -10.0 && y0 < 910.0 && y1 > -10.0,
                        "{t}: {i:?}"
                    );
                    assert!(
                        i.quad.iter().chain(&i.shape).all(|v| v.is_finite()),
                        "{i:?}"
                    );
                }
                f.instances.len() + f.top.len()
            })
            .collect();
        assert!(counts.windows(2).all(|c| c[0] < c[1]), "{counts:?}");
        let last = frame_at(&r, reveal.end() + 2.0);
        let medals = last
            .instances
            .iter()
            .filter(|i| i.params[0] == kind::MEDAL)
            .count();
        let gems = last
            .instances
            .iter()
            .filter(|i| i.params[0] == kind::GEM)
            .count();
        assert_eq!((medals, gems), (1, 4 + 3));
        assert!(last.sprays.iter().any(|s| s.kind == Kind::Confetti));
    }

    /// No medal: a star badge, no confetti without a new best; not sung
    /// whole: no medal, no new best, however high.
    #[test]
    fn no_medal_and_not_whole_draw_a_star() {
        for r in [result(None, true, Some(50_000)), result(None, false, None)] {
            let reveal = crate::core::singing::Reveal::of(&r);
            let f = frame_at(&r, reveal.end() + 2.0);
            let star = f
                .instances
                .iter()
                .find(|i| i.params[0] == kind::MEDAL)
                .expect("a star");
            assert_eq!(star.shape[3], 1.0);
            assert!(!f.sprays.iter().any(|s| s.kind == Kind::Confetti));
        }
    }

    /// Nothing scored (no points, no words, no phrases): the scene draws
    /// all the same, in the window and finite throughout, a star for its
    /// wink at the microphone, no gems and no confetti.
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
            let f = frame_at(&r, t);
            for i in f.instances.iter().chain(&f.top) {
                let [x0, y0, x1, y1] = i.quad;
                assert!(x0 < 1610.0 && x1 > -10.0 && y0 < 910.0 && y1 > -10.0);
                assert!(i.quad.iter().chain(&i.shape).all(|v| v.is_finite()));
            }
        }
        let f = frame_at(&r, reveal.end() + 2.0);
        let kinds = |k| f.instances.iter().filter(|i| i.params[0] == k).count();
        assert_eq!((kinds(kind::MEDAL), kinds(kind::GEM)), (1, 0));
        assert!(!f.sprays.iter().any(|s| s.kind == Kind::Confetti));
    }
}
