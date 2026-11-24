//! The diploma's sheet: cream paper unrolling from its top (a roll lying
//! across under its front, [`crate::gui::paper::paper_at`]), its ornate
//! border in the tint's colours; printed on it as it shows, the heading
//! (flanked by two gold stars popping on), the lines round the song's
//! title (which lands on its beat, a sparkle off each end), its entries'
//! names and ruled lines to write on (a form waiting to be filled in), and
//! under the signature's and the date's lines what they are; at the end
//! the examiner signs it in one stroke and writes the day.

use super::super::super::paper::signature_point;
use super::super::super::stage_gpu::{kind, Instance};
use super::super::{disc, rgba, segment, Face, Rgb, Stage, GOLD, GOLD_TOP};
use super::{
    burst, clamp01, ease_out, fade, land, sticker, Drawn, Pen, ResultShown, Room, Tint, Written,
    PRINT,
};
use crate::core::singing::Beat;
use crate::gui::particles::Kind;
use crate::view::karaoke::DIPLOMA;

/// How long the sheet takes to unroll, in seconds.
const UNROLL: f32 = 0.55;
/// How long the signature takes to write, and the date after it.
const SIGNING: f32 = 0.9;
const DATING: f32 = 0.6;
/// Where the entries' lines are down the sheet, and where they run across
/// (their names right of the first).
pub(super) const ENTRIES: [f32; 3] = [0.43, 0.505, 0.575];
pub(super) const LINE_FROM: f32 = 0.42;
pub(super) const LINE_TO: f32 = 0.8;
/// Where the signature and the date are written, down the sheet, and
/// across.
const SIGNED_Y: f32 = 0.87;
const SIGNATURE_X: f32 = 0.25;
const DATE_X: f32 = 0.56;

/// The diploma's colours, from the tint: its paper, its printed ink, its
/// heading's, the pen's (what is written in), the gold ink (the points),
/// and its border's two.
#[derive(Clone, Copy, Debug)]
pub(super) struct Inks {
    pub(super) paper: Rgb,
    pub(super) ink: Rgb,
    pub(super) heading: Rgb,
    pub(super) pen: Rgb,
    pub(super) gold: Rgb,
    pub(super) border: (Rgb, Rgb),
}

fn mix(a: Rgb, b: Rgb, t: f32) -> Rgb {
    [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t)
}

impl Inks {
    /// The inks in `tint`.
    pub(super) fn of(tint: &Tint) -> Inks {
        let deep = |c: Rgb, k: f32| c.map(|v| v * k);
        Inks {
            paper: mix([0.99, 0.96, 0.87], tint.glow, 0.05),
            ink: tint.ink,
            heading: mix(deep(tint.ribbon.0, 0.7), tint.ink, 0.3),
            pen: mix(deep(tint.ribbon.1, 0.45), [0.08, 0.12, 0.42], 0.55),
            gold: [0.74, 0.44, 0.04],
            border: (
                mix(deep(tint.ribbon.0, 0.75), tint.ink, 0.25),
                mix(deep(tint.ribbon.1, 0.75), tint.ink, 0.25),
            ),
        }
    }
}

impl Stage {
    /// The sheet unrolling (its first [`UNROLL`] seconds) and what is
    /// printed on it, each line once the front has passed it; the title
    /// landing on its beat.
    pub(super) fn sheet(&mut self, d: &mut Drawn, room: &Room, shown: &ResultShown) {
        let t = shown.t;
        let (_, _, sw, sh) = room.sheet;
        let s = room.s;
        let inks = room.inks;
        let open = ease_out(t as f32 / UNROLL);
        let c = room.at(0.5, 0.5);
        let half = (sw / 2.0, sh / 2.0);
        let front = -half.1 + sh * open;
        let reach = 4.0 * s;
        d.out.push(Instance {
            quad: [
                c.0 - half.0 - reach,
                c.1 - half.1 - reach,
                c.0 + half.0 + reach,
                c.1 + half.1 + reach + 2.0 * s,
            ],
            shape: [c.0, c.1, half.0, half.1],
            color: rgba(inks.paper, 1.0),
            color2: rgba(inks.border.0, 0.85),
            params: [kind::PAPER, front, s, 0.0],
            uv: [inks.border.1[0], inks.border.1[1], inks.border.1[2], 0.7],
            strain: [0.0; 4],
        });
        // What shows of something printed `fy` down the sheet.
        let printed = |fy: f32| clamp01((open - fy) * sh / (3.0 * s));
        let ink = |a: f32| rgba(inks.ink, a);
        // The heading, two gold stars popping on beside it.
        let hp = (10.5 * s).round().max(16.0);
        let (heading, hp) = self.fitted(Face::SHOUT, DIPLOMA.heading, hp, room.across(0.62));
        let at = room.at(0.5, 0.135);
        let head = rgba(inks.heading, printed(0.135));
        let text = (Face::SHOUT, &*heading);
        let hw = self.written(&mut d.out, text, (hp, 1.0), (at, 0.5), head, Pen::FLAT);
        let popped = t - f64::from(UNROLL) * 0.5;
        for (k, side) in [-1.0f32, 1.0].into_iter().enumerate() {
            let age = Some(popped - k as f64 * 0.08).filter(|&a| a >= 0.0);
            let (scale, shows) = land(age);
            let star_at = (at.0 + side * (hw / 2.0 + hp * 0.55), at.1);
            let size = (hp * 0.36 * scale, side * 0.25);
            let halo = (0.0, [0.0; 4]);
            let glint = -1.6 + fade(age, 0.7) * 3.4;
            d.out
                .push(sticker(star_at, size, rgba(GOLD, shows), halo, glint));
        }
        // The lines round the title (the one under it as the verdict has
        // it: yodelled duly, or very quietly).
        let px = (3.0 * s).round().max(10.0);
        let width = room.across(0.84);
        for (text, fy) in [(DIPLOMA.certify, 0.225), (room.look.yodelled, 0.36)] {
            let (text, px) = self.fitted(PRINT, text, px, width);
            let a = 0.78 * printed(fy);
            let text = (PRINT, &*text);
            let at = (room.at(0.5, fy), 0.5);
            self.written(&mut d.out, text, (px, 1.0), at, ink(a), Pen::FLAT);
        }
        self.title(d, room, shown);
        // The entries' names and lines.
        let lp = (3.1 * s).round().max(10.0);
        let thin = (0.09 * s).max(0.6);
        for (name, fy) in DIPLOMA.entries.into_iter().zip(ENTRIES) {
            let a = printed(fy);
            let at = (room.at(LINE_FROM - 0.025, fy), 1.0);
            self.written(
                &mut d.out,
                (PRINT, name),
                (lp, 1.0),
                at,
                ink(0.82 * a),
                Pen::FLAT,
            );
            let base = fy + 2.0 * s / sh;
            let (from, to) = (room.at(LINE_FROM, base), room.at(LINE_TO, base));
            d.out
                .push(segment(from, to, thin, ink(0.42 * a), [0.0; 4], 0.0));
        }
        // Under the signature's line and the date's, what they are.
        let cp = (2.1 * s).round().max(8.0);
        for (text, fx, wide) in [
            (DIPLOMA.signed.0, SIGNATURE_X, 0.13),
            (DIPLOMA.signed.1, DATE_X, 0.12),
        ] {
            let a = printed(SIGNED_Y + 0.05);
            let base = SIGNED_Y + 0.03;
            let (from, to) = (room.at(fx - wide, base), room.at(fx + wide, base));
            d.out
                .push(segment(from, to, thin, ink(0.5 * a), [0.0; 4], 0.0));
            let (text, cp) = self.fitted(PRINT, text, cp, room.across(wide * 2.0));
            let at = (room.at(fx, base + 2.4 * s / sh), 0.5);
            let text = (PRINT, &*text);
            self.written(&mut d.out, text, (cp, 1.0), at, ink(0.7 * a), Pen::FLAT);
        }
    }

    /// The song's title, landing on its beat (growing past its size and
    /// back), a sparkle off each end as it lands.
    fn title(&mut self, d: &mut Drawn, room: &Room, shown: &ResultShown) {
        let (t, r) = (shown.t, &shown.shown.result);
        let Some(at) = shown.shown.reveal.when(Beat::Title).filter(|&at| t >= at) else {
            return;
        };
        let s = room.s;
        let px = (6.0 * s).round().max(14.0);
        let (title, px) = self.fitted(Face::LINE, &r.title, px, room.across(0.8));
        let (scale, shows) = land(Some(t - at));
        let c = room.at(0.5, 0.293);
        let ink = rgba(room.inks.ink, shows);
        let text = (Face::LINE, &*title);
        let tw = self.written(&mut d.out, text, (px, scale), (c, 0.5), ink, Pen::FLAT);
        for (k, side) in [-1.0f32, 1.0].into_iter().enumerate() {
            let end = (c.0 + side * (tw / 2.0 + px * 0.4), c.1);
            let seed = (11 + k as u32, GOLD_TOP);
            d.sprays
                .push(burst(end, at + 0.3, px * 0.45, seed, Kind::Scored));
        }
    }

    /// At the end, the examiner's signature written in one stroke, then
    /// the day the result was first drawn on written beside it (`words`'
    /// date; none where the system has none).
    pub(super) fn signed(
        &mut self,
        d: &mut Drawn,
        room: &Room,
        shown: &ResultShown,
        words: &Written,
    ) {
        let (t, reveal) = (shown.t, &shown.shown.reveal);
        let Some(a) = reveal.age(Beat::End, t).map(|a| a as f32) else {
            return;
        };
        let s = room.s;
        let pen = rgba(room.inks.pen, 0.92);
        let drawn = ease_out(a / SIGNING).min(1.0);
        let size = (12.0 * s).min(room.across(0.13));
        let c = room.at(SIGNATURE_X, SIGNED_Y - 0.004);
        let reach = size * 1.3;
        d.out.push(Instance {
            quad: [
                c.0 - reach,
                c.1 - size * 0.7,
                c.0 + reach,
                c.1 + size * 0.55,
            ],
            shape: [c.0, c.1, size, 0.0],
            color: pen,
            params: [kind::SIGNATURE, drawn, (0.2 * s).max(0.8), 0.0],
            ..Instance::default()
        });
        if drawn < 1.0 {
            // The pen's nib, where it is.
            let (nx, ny) = signature_point(drawn);
            let nib = (c.0 + nx * size, c.1 + ny * size);
            d.out.push(disc(nib, 0.35 * s, pen, [0.0; 4], 0.0));
        }
        let Some(date) = &words.date else {
            return;
        };
        let px = (3.2 * s).round().max(10.0);
        let (date, px) = self.fitted(Face::NUMBER, date, px, room.across(0.28));
        let share = clamp01((a - SIGNING * 0.4) / DATING);
        let at = (room.at(DATE_X, SIGNED_Y), 0.5);
        let text = (Face::NUMBER, &*date);
        self.written(&mut d.out, text, (px, 1.0), at, pen, Pen::writing(share));
    }
}
