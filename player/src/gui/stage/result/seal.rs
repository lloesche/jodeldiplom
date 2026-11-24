//! The diploma's verdict ([`Verdict`]), each looking as [`look`] says:
//! the grade written in under the stars (the medal as a grade, in its
//! metal's colour; a kind line without one; else the verdict's own name),
//! its remark under it; and, for a medal, a wax seal in its metal stamped
//! down into the sheet's corner on its beat (falling from large for the
//! last [`DROP`] before it; the sheet shaking, a ring of dust going out,
//! sparkles, coins for Gold), its ribbons hanging under it and its name on
//! a banner across it, rays turning out from behind the sheet, a sheen
//! sweeping over it now and then while the screen twinkles. Without a
//! medal a star sticker pops on where the seal would go (a sleepy one,
//! with a "Zzz" rubber stamp, where nothing was scored). Until then the
//! seal's place is marked, faintly.

use super::super::super::stage_gpu::kind;
use super::super::{disc, pill, rgba, segment, Face, Rgb, Stage, GOLD, GOLD_TOP, WHITE};
use super::{
    burst, clamp01, ease_out, fade, land, shape, sparkle, sticker, Drawn, Pen, ResultShown, Room,
    Tint, Written, PRINT,
};
use crate::core::singing::{Medal, Verdict, DROP};
use crate::gui::particles::Kind;
use crate::view::karaoke::{verdict_words, DIPLOMA};

/// How long a seal falls, as the scene takes it.
const FALL: f32 = DROP as f32;
/// Where the seal goes (its middle, shares of the sheet's width and
/// height), and the grade and its remark (down the sheet).
const SEAL_AT: (f32, f32) = (0.865, 0.8);
const GRADE_Y: f32 = 0.765;
const REMARK_Y: f32 = 0.815;
/// How long the grade takes to write.
const GRADE_WRITE: f32 = 0.6;

/// How a verdict looks on the diploma.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Look {
    /// The medal whose seal is stamped down (none: a star sticker stands
    /// for it).
    pub(super) medal: Option<Medal>,
    /// The seal's wax or the star's colour, light and deep.
    light: Rgb,
    deep: Rgb,
    /// How strong the rays behind the sheet are (0: none).
    rays: f32,
    /// How hard the sheet shakes as the seal is stamped down, in its units.
    shake: f32,
    /// The grade's ink (none: the pen's) and its size, in units.
    ink: Option<Rgb>,
    grade_px: f32,
    /// The grade's own words (none: the verdict's name).
    grade: Option<&'static str>,
    /// The line under the song's title.
    pub(super) yodelled: &'static str,
    /// A rubber stamp slapped on with the verdict: its words and ink.
    stamp: Option<(&'static str, Rgb)>,
}

/// How `verdict` looks in `tint`: a medal's metal, rays and Latin; a
/// cheer's star in the tint's light; a calm blue star for a song not sung
/// from its start; a sleepy lilac one, "Zzz" stamped beside it and the
/// song yodelled very quietly, where nothing was scored.
pub(super) fn look(verdict: Verdict, tint: &Tint) -> Look {
    let medal = |m, light, deep, (rays, shake), ink, grade| Look {
        medal: Some(m),
        light,
        deep,
        rays,
        shake,
        ink: Some(ink),
        grade_px: 6.0,
        grade: Some(grade),
        yodelled: DIPLOMA.yodelled.0,
        stamp: None,
    };
    let star = |light: Rgb, grade| Look {
        medal: None,
        light,
        deep: light.map(|k| k * 0.45),
        rays: 0.0,
        shake: 0.0,
        ink: None,
        grade_px: 4.6,
        grade,
        yodelled: DIPLOMA.yodelled.0,
        stamp: None,
    };
    match verdict {
        Verdict::Medal(Medal::Gold) => medal(
            Medal::Gold,
            [1.0, 0.86, 0.36],
            [0.62, 0.36, 0.03],
            (0.6, 1.1),
            [0.7, 0.42, 0.03],
            "summa cum laude",
        ),
        Verdict::Medal(Medal::Silver) => medal(
            Medal::Silver,
            [0.96, 0.97, 1.0],
            [0.42, 0.46, 0.58],
            (0.45, 0.85),
            [0.36, 0.4, 0.52],
            "magna cum laude",
        ),
        Verdict::Medal(Medal::Bronze) => medal(
            Medal::Bronze,
            [1.0, 0.72, 0.46],
            [0.48, 0.22, 0.08],
            (0.35, 0.7),
            [0.55, 0.27, 0.1],
            "cum laude",
        ),
        Verdict::Cheer => star(tint.glow, Some("participated with great enthusiasm")),
        Verdict::Whole => star([0.5, 0.68, 1.0], None),
        Verdict::Silent => Look {
            yodelled: DIPLOMA.yodelled.1,
            stamp: Some((DIPLOMA.asleep, [0.42, 0.38, 0.6])),
            ..star([0.72, 0.66, 0.86], None)
        },
    }
}

/// The sheet's shake `hit` seconds after the verdict's beat (before it:
/// less than none), `s` its unit: a seal stamped down, the bigger the
/// medal the harder; a star none.
pub(super) fn shake(look: &Look, hit: Option<f32>, s: f32) -> (f32, f32) {
    let Some(a) = hit.filter(|a| (0.0..0.45).contains(a)) else {
        return (0.0, 0.0);
    };
    let k = look.shake * s * (-a * 10.0).exp();
    (k * (a * 95.0).sin(), 0.6 * k * (a * 80.0).cos())
}

/// The rays turning out from behind the sheet once a medal is stamped
/// (`hit` seconds after its beat), `t` the reveal's time: light.
pub(super) fn rays(d: &mut Drawn, room: &Room, hit: Option<f32>, t: f64) {
    let look = &room.look;
    let Some(hit) = hit.filter(|&h| h >= 0.0 && look.rays > 0.0) else {
        return;
    };
    let (_, _, sw, sh) = room.sheet;
    let c = room.at(0.5, 0.5);
    let reach = sw.max(sh) * 0.62;
    let mut rays = shape(kind::RAYS, c, reach * 1.6, [c.0, c.1, reach, 16.0]);
    rays.params[1] = t as f32 * 0.3;
    rays.color = rgba(look.light, look.rays * fade(Some(f64::from(hit)), 0.6));
    d.out.push(rays);
}

impl Stage {
    /// The verdict, `hit` seconds after its beat (before it: less than
    /// none): the seal's place marked while it waits; then the grade
    /// written in, and the seal stamped down or a star popping on (and its
    /// rubber stamp slapped on, if it has one).
    pub(super) fn verdict(
        &mut self,
        d: &mut Drawn,
        room: &Room,
        shown: &ResultShown,
        tint: &Tint,
        (hit, words): (Option<f32>, &Written),
    ) {
        let t = shown.t;
        let s = room.s;
        let look = room.look;
        let radius = (8.5 * s).min(room.across(0.11));
        let c = room.at(SEAL_AT.0, SEAL_AT.1);
        // A seal shows from its fall.
        let shown_from = if look.medal.is_some() { -FALL } else { 0.0 };
        let waiting = hit.map_or(1.0, |h| 1.0 - clamp01((h - shown_from) / 0.12));
        self.slot(d, room, (c, radius), t, waiting);
        let Some(hit) = hit.filter(|&h| h >= shown_from) else {
            return;
        };
        self.grade(d, room, shown, hit);
        match look.medal {
            Some(m) => self.seal(d, room, shown, tint, (c, radius), (m, hit)),
            None => star(d, room, (c, radius), look.light, (t, hit)),
        }
        if let Some(stamp) = look.stamp.filter(|_| hit >= 0.0) {
            self.rubber_stamp(d, room, words, stamp, (t, hit));
        }
    }

    /// The seal's place while it waits: a faint ring, "L.S." in it,
    /// `shows` of it.
    fn slot(
        &mut self,
        d: &mut Drawn,
        room: &Room,
        (c, radius): ((f32, f32), f32),
        t: f64,
        shows: f32,
    ) {
        let shows = 0.3 * shows * fade(Some(t - 0.45), 0.6);
        if shows <= 0.0 {
            return;
        }
        let ink = rgba(room.inks.ink, shows);
        let ring = radius * 0.8;
        d.out
            .push(pill(c, (ring, ring), [0.0; 4], ink, 0.0, 1.2, 0.0));
        let px = (2.6 * room.s).round().max(9.0);
        let ls = (PRINT, DIPLOMA.seal);
        self.written(&mut d.out, ls, (px, 1.0), (c, 0.5), ink, Pen::FLAT);
    }

    /// The grade written in from its beat (`hit` seconds ago) in its ink,
    /// the pen's nib at its front, its remark fading in under it.
    fn grade(&mut self, d: &mut Drawn, room: &Room, shown: &ResultShown, hit: f32) {
        let r = &shown.shown.result;
        let (s, look) = (room.s, room.look);
        let (name, remark) = verdict_words(r);
        let grade = look.grade.unwrap_or(name);
        let px = (look.grade_px * s).round().max(12.0);
        let (grade, px) = self.fitted(Face::NUMBER, grade, px, room.across(0.6));
        let share = clamp01(hit / GRADE_WRITE);
        let c = room.at(0.5, GRADE_Y);
        let pen = rgba(look.ink.unwrap_or(room.inks.pen), 0.95);
        let text = (Face::NUMBER, &*grade);
        let w = self.written(
            &mut d.out,
            text,
            (px, 1.0),
            (c, 0.5),
            pen,
            Pen::writing(share),
        );
        if (0.0..1.0).contains(&share) {
            let nib = (c.0 - w / 2.0 + w * share, c.1 + 0.3 * px);
            d.out.push(disc(nib, 0.32 * s, pen, [0.0; 4], 0.0));
        }
        if look.medal == Some(Medal::Gold) {
            let lit = rgba(GOLD, 0.25 * fade(Some(f64::from(hit)), 0.4));
            d.lights.push(disc(c, 0.0, [0.0; 4], lit, px * 1.6));
        }
        let rp = (2.8 * s).round().max(9.0);
        let (remark, rp) = self.fitted(PRINT, remark, rp, room.across(0.6));
        let soft = rgba(room.inks.ink, 0.8 * fade(Some(f64::from(hit) - 0.45), 0.4));
        let at = room.at(0.5, REMARK_Y);
        let text = (PRINT, &*remark);
        self.written(&mut d.out, text, (rp, 1.0), (at, 0.5), soft, Pen::FLAT);
    }

    /// Medal `m`'s seal at `c` (`radius` its size), `hit` seconds after its
    /// beat (falling for the last [`FALL`] before it, stamped down on it):
    /// its ribbons, the wax, its name's banner across it, sparkles as it
    /// lands and twinkling round its rim.
    fn seal(
        &mut self,
        d: &mut Drawn,
        room: &Room,
        shown: &ResultShown,
        tint: &Tint,
        (c, radius): ((f32, f32), f32),
        (m, hit): (Medal, f32),
    ) {
        let t = shown.t;
        let (light, deep) = (room.look.light, room.look.deep);
        let scale = if hit < 0.0 {
            let falling = clamp01((hit + FALL) / FALL);
            2.2 - 1.2 * falling * falling
        } else {
            1.0 + 0.1 * (-hit * 10.0).exp() * (hit * 28.0).cos()
        };
        let shows = clamp01((hit + FALL) / 0.08);
        let r_now = radius * scale;
        ribbons(d, (c, r_now), tint, shows);
        // Its sheen sweeping over it as it lands, then every few seconds
        // while the screen twinkles.
        let sheen = match (hit < 1.0, room.calm > 0.5) {
            (true, _) => -1.6 + hit.max(0.0) * 3.2,
            (false, true) => -1.6 + ((hit + 1.0) % 3.4) * 2.4,
            (false, false) => 9.0,
        };
        let reach = r_now * 1.15 + radius * 0.2 + 2.0;
        let mut wax = shape(kind::SEAL, c, reach, [c.0, c.1, r_now, 0.0]);
        wax.color = rgba(light, shows);
        wax.color2 = rgba(deep, shows);
        wax.params = [kind::SEAL, radius * 0.15, sheen, 0.45 * shows];
        d.out.push(wax);
        // Its name on a banner across it (fitted at its size, drawn at its
        // scale).
        let band = (c.0, c.1 + 0.62 * r_now);
        let half = (0.82 * r_now, 0.2 * r_now);
        let cloth = tint.ribbon.0.map(|k| k * 0.8);
        let edge = rgba(tint.ribbon.0.map(|k| (k * 1.2).min(1.0)), shows);
        d.out.push(pill(
            band,
            half,
            rgba(cloth, shows),
            edge,
            0.0,
            1.2,
            half.1 * 0.4,
        ));
        let (name, _) = verdict_words(&shown.shown.result);
        let px = (0.27 * radius).round().max(8.0);
        let (name, px) = self.fitted(Face::SHOUT, name, px, 0.82 * radius * 1.7);
        let white = rgba(WHITE, shows);
        let pen = Pen {
            shadow: 0.6,
            ..Pen::FLAT
        };
        let text = (Face::SHOUT, &*name);
        self.written(&mut d.out, text, (px, scale), (band, 0.5), white, pen);
        let stamp = t - f64::from(hit);
        d.sprays
            .push(burst(c, stamp, radius * 0.7, (301, light), Kind::Scored));
        stamped(d, room, (c, radius), m, (stamp, hit));
        twinkle(d, room, (c, r_now), (t, hit));
    }
}

/// Without a medal, a star sticker at `c` (in `light`, smaller than a
/// seal) popping on `hit` seconds after its beat, sparkles as it lands
/// and twinkling round it.
fn star(
    d: &mut Drawn,
    room: &Room,
    (c, radius): ((f32, f32), f32),
    light: Rgb,
    (t, hit): (f64, f32),
) {
    let (scale, shows) = land(Some(f64::from(hit)));
    let size = radius * 0.72;
    let glint = match (hit < 1.0, room.calm > 0.5) {
        (true, _) => -1.6 + hit * 3.4,
        (false, true) => -1.6 + ((hit + 1.0) % 3.4) * 2.4,
        (false, false) => 9.0,
    };
    let halo = (size * 0.5, rgba(light, 0.3 * shows));
    d.out.push(sticker(
        c,
        (size * scale, -0.12),
        rgba(light, shows),
        halo,
        glint,
    ));
    let at = t - f64::from(hit);
    d.sprays
        .push(burst(c, at, size, (301, light), Kind::Scored));
    twinkle(d, room, (c, size), (t, hit));
}

/// A few sparkles twinkling round `c` (`r` out), light, from `hit`
/// seconds after the verdict, while the screen twinkles.
fn twinkle(d: &mut Drawn, room: &Room, (c, r): ((f32, f32), f32), (t, hit): (f64, f32)) {
    let after = fade(Some(f64::from(hit)), 0.3);
    for k in 0..3 {
        let turn = k as f32 * 2.1 + 0.6;
        let tw = 0.5 + 0.5 * (t as f32 * 3.0 + k as f32 * 1.7).sin();
        let at = (c.0 + turn.cos() * r * 0.95, c.1 + turn.sin() * r * 0.95);
        let shine = rgba(GOLD_TOP, tw * after * room.calm);
        d.lights.push(sparkle(at, r * 0.3 * tw, t as f32, shine));
    }
}

/// The seal's ribbons at `c` (it `r` across): two tails hanging from
/// behind it, in the tint's two colours, a stripe down each.
fn ribbons(d: &mut Drawn, (c, r): ((f32, f32), f32), tint: &Tint, shows: f32) {
    for (side, col) in [(-1.0f32, tint.ribbon.0), (1.0, tint.ribbon.1)] {
        let from = (c.0 + side * 0.12 * r, c.1 + 0.2 * r);
        let to = (c.0 + side * 0.62 * r, c.1 + 1.45 * r);
        let edge = rgba(col.map(|k| k * 0.6), shows);
        d.out.push(segment(from, to, 0.24 * r, edge, [0.0; 4], 0.0));
        d.out
            .push(segment(from, to, 0.19 * r, rgba(col, shows), [0.0; 4], 0.0));
        let stripe = rgba(WHITE, 0.35 * shows);
        d.out
            .push(segment(from, to, 0.03 * r, stripe, [0.0; 4], 0.0));
    }
}

/// Medal `m`'s seal at `c` stamped down at `stamp` (the reveal's clock:
/// its beat), `hit` seconds ago: a ring of dust going out over the sheet;
/// coins for Gold, stars and notes for Gold and Silver.
fn stamped(
    d: &mut Drawn,
    room: &Room,
    (c, radius): ((f32, f32), f32),
    m: Medal,
    (stamp, hit): (f64, f32),
) {
    if (0.0..0.55).contains(&hit) {
        let k = hit / 0.55;
        let ring = radius * (1.05 + 1.0 * ease_out(k));
        let line = rgba(room.inks.ink, 0.4 * (1.0 - k) * (1.0 - k));
        let width = radius * 0.05 * (1.0 - k) + 1.0;
        d.out
            .push(pill(c, (ring, ring), [0.0; 4], line, 0.0, width, 0.0));
    }
    if m == Medal::Gold {
        let coins = burst(
            c,
            stamp,
            2.2 * room.u,
            (307, [1.0, 0.8, 0.3]),
            Kind::Jackpot,
        );
        d.light_sprays.push(coins);
    }
    if matches!(m, Medal::Gold | Medal::Silver) {
        let top = (c.0, c.1 - radius);
        d.sprays
            .push(burst(top, stamp, radius * 0.5, (311, WHITE), Kind::StepUp));
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{frame_at, result};
    use super::*;
    use crate::core::singing::Beat;

    const GOLD_BEAT: Beat = Beat::Verdict(Verdict::Medal(Medal::Gold));

    /// The seal falls in large before its beat and is stamped down on it
    /// at its size, the sheet shaking only from then, and not for a star.
    #[test]
    fn the_seal_is_stamped_down_on_its_beat() {
        let r = result(Some(Medal::Gold), true, None);
        let reveal = crate::core::singing::Reveal::of(&r);
        let at = reveal.when(GOLD_BEAT).unwrap();
        let radius = |t: f64| {
            let f = frame_at(&r, t);
            f.instances
                .iter()
                .find(|i| i.params[0] == kind::SEAL)
                .map(|i| i.shape[2])
        };
        assert_eq!(radius(at - DROP - 0.05), None);
        let falling = radius(at - DROP * 0.8).unwrap();
        let (stamped, landed) = (radius(at).unwrap(), radius(at + 2.0).unwrap());
        assert!(falling > landed * 1.5, "{falling} {landed}");
        assert!(
            (stamped - landed).abs() < landed * 0.11 && landed > 40.0,
            "{landed}"
        );
        let gold = look(Verdict::Medal(Medal::Gold), &Tint::STAGE);
        assert_eq!(shake(&gold, Some(-0.1), 9.0), (0.0, 0.0));
        let (dx, dy) = shake(&gold, Some(0.02), 9.0);
        assert!(dx.abs() + dy.abs() > 0.5);
        assert_eq!(shake(&gold, Some(3.0), 9.0), (0.0, 0.0));
        for v in [Verdict::Cheer, Verdict::Whole, Verdict::Silent] {
            assert_eq!(shake(&look(v, &Tint::STAGE), Some(0.02), 9.0), (0.0, 0.0));
        }
    }

    /// Each verdict's look: a seal, rays and Latin for a medal; a star
    /// for the rest; the song yodelled very quietly, a "Zzz" stamp and the
    /// verdict's own name where nothing was scored.
    #[test]
    fn each_verdict_looks_its_own() {
        let of = |v| look(v, &Tint::STAGE);
        let gold = of(Verdict::Medal(Medal::Gold));
        assert_eq!(gold.grade, Some("summa cum laude"));
        assert!(gold.medal.is_some() && gold.rays > 0.0 && gold.stamp.is_none());
        assert_eq!(of(Verdict::Medal(Medal::Bronze)).grade, Some("cum laude"));
        for v in [Verdict::Cheer, Verdict::Whole, Verdict::Silent] {
            let l = of(v);
            assert!(
                l.medal.is_none() && l.rays == 0.0 && l.shake == 0.0,
                "{v:?}"
            );
        }
        let silent = of(Verdict::Silent);
        assert_eq!((silent.grade, silent.yodelled), (None, DIPLOMA.yodelled.1));
        assert!(silent.yodelled.contains("quietly") && silent.stamp.is_some());
        assert_eq!(of(Verdict::Cheer).yodelled, DIPLOMA.yodelled.0);
    }
}
