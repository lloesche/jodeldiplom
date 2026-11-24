//! The result screen's trophies: each phrase's gem popping in, in its
//! tier's colour (Diva's prismatic, glinting and twinkling), the tiers
//! counted under them; and the verdict ([`Verdict`]): a medal falling from
//! above and stamped down on its beat (the screen shaking, a shock ring
//! going out, sparkles, a jackpot for Gold), its ribbon over it and its
//! rays turning behind it, its name and a word under it; or, without one,
//! a star that cheers the singer on, popping up on its beat.

use super::super::super::rainbow::hue;
use super::super::meter::tier_color;
use super::super::{disc, pill, rgba, segment, Face, Rgb, Stage, GOLD, GOLD_TOP, WHITE};
use super::{clamp01, ease_out, fade, land, shape, sparkle, Drawn, ResultShown, Room, Tint};
use crate::core::shine::Tier;
use crate::core::singing::{Beat, Medal, Verdict, DROP};
use crate::gui::particles::{Kind, Spray};
use crate::gui::stage_gpu::{kind, Instance};
use crate::view::karaoke::verdict_words;

/// How long a medal falls, as the scene takes it.
const FALL: f32 = DROP as f32;

/// The screen's shake `hit` seconds after the verdict's beat (before it:
/// less than none): a medal stamped down, the bigger the medal the harder;
/// a star none.
pub(super) fn shake(verdict: Verdict, hit: Option<f32>, u: f32) -> (f32, f32) {
    let amp = match verdict {
        Verdict::Medal(Medal::Gold) => 0.9,
        Verdict::Medal(Medal::Silver) => 0.7,
        Verdict::Medal(Medal::Bronze) => 0.55,
        Verdict::Cheer | Verdict::Whole => 0.0,
    } * u;
    let Some(a) = hit.filter(|a| (0.0..0.45).contains(a)) else {
        return (0.0, 0.0);
    };
    let k = amp * (-a * 10.0).exp();
    (k * (a * 95.0).sin(), 0.6 * k * (a * 80.0).cos())
}

/// The verdict's look: its light and deep colours (a medal's metal, a
/// star's from the tint, a calm blue for a song not sung whole) and how
/// strong its rays are.
fn look(verdict: Verdict, tint: &Tint) -> (Rgb, Rgb, f32) {
    match verdict {
        Verdict::Medal(Medal::Gold) => ([1.0, 0.86, 0.36], [0.7, 0.4, 0.04], 0.6),
        Verdict::Medal(Medal::Silver) => ([0.96, 0.97, 1.0], [0.46, 0.5, 0.62], 0.45),
        Verdict::Medal(Medal::Bronze) => ([1.0, 0.72, 0.46], [0.52, 0.26, 0.1], 0.35),
        Verdict::Cheer => (tint.soft, tint.glow.map(|k| k * 0.45), 0.3),
        Verdict::Whole => ([0.72, 0.84, 1.0], [0.25, 0.35, 0.65], 0.18),
    }
}

/// Where `n` gems go in `room`: their size, and each one's middle, in rows
/// centred under the numbers, in a band [`GEM_BAND`] of the height high.
/// At most three rows, the gems smaller the more there are, down to a
/// third of a unit across each way; past that (hundreds of phrases in a
/// narrow window) as many rows as they take.
fn gem_places(n: usize, room: &Room) -> (f32, Vec<(f32, f32)>) {
    let mut size = 1.9 * room.u;
    let smallest = 0.35 * room.u;
    let width = room.w * 0.78;
    let per = |size: f32| ((width / (size * 2.6)).floor() as usize).max(1);
    let high = |size: f32| n.div_ceil(per(size)) as f32 * size * 2.3;
    while (n.div_ceil(per(size)) > 3 || high(size) > room.h * GEM_BAND) && size > smallest {
        size = (size * 0.9).max(smallest);
    }
    let rows = n.div_ceil(per(size)).max(1);
    let in_row = n.div_ceil(rows).max(1);
    let (cx, cy) = room.at(0.5, 0.605);
    let places = (0..n)
        .map(|i| {
            let (row, j) = (i / in_row, i % in_row);
            let here = in_row.min(n - row * in_row);
            let x = cx + (j as f32 - (here as f32 - 1.0) / 2.0) * size * 2.6;
            let y = cy + (row as f32 - (rows as f32 - 1.0) / 2.0) * size * 2.3;
            (x, y)
        })
        .collect();
    (size, places)
}

/// How high the gems' rows may reach, a share of the height.
const GEM_BAND: f32 = 0.09;

/// A gem at `c`, `size` across each way, in `color`, glowing `glow` far in
/// `halo`, its glint at `glint`.
fn gem(
    c: (f32, f32),
    size: f32,
    color: [f32; 4],
    (glow, halo): (f32, [f32; 4]),
    glint: f32,
) -> Instance {
    let reach = size * 1.1 + glow + 1.0;
    Instance {
        color,
        color2: halo,
        params: [kind::GEM, glow, glint, 0.0],
        ..shape(kind::GEM, c, reach, [c.0, c.1, size, 0.0])
    }
}

/// A tier's gem colour at `t` (Diva's running through the rainbow a
/// little, `i` its phase).
fn gem_color(tier: Option<Tier>, t: f64, i: usize) -> Rgb {
    let base = tier_color(tier);
    if tier == Some(Tier::Diva) {
        let h = hue(t as f32 * 0.25 + i as f32 * 0.13);
        return [0, 1, 2].map(|k| base[k] + (h[k] - base[k]) * 0.35);
    }
    base
}

impl Stage {
    /// The phrases' gems, popping in one by one in song order (each
    /// glinting as it lands, Amazing and Diva glinting now and then while
    /// the screen twinkles, Diva twinkling and glowing too), and, once all
    /// are in, how many reached each tier under them.
    pub(super) fn gems(&mut self, d: &mut Drawn, room: &Room, shown: &ResultShown) {
        let (t, reveal, r) = (shown.t, &shown.shown.reveal, &shown.shown.result);
        let (size, places) = gem_places(r.phrases.len(), room);
        for (i, (p, &c)) in r.phrases.iter().zip(&places).enumerate() {
            let age = reveal.age(Beat::Gem(i, p.tier), t);
            let (scale, shows) = land(age);
            let Some(a) = age.map(|a| a as f32) else {
                continue;
            };
            let rich = matches!(p.tier, Some(Tier::Amazing | Tier::Diva));
            let again = (a + i as f32 * 0.37) % 2.4;
            let glint = match (a < 0.7, rich && room.calm > 0.5) {
                (true, _) => -1.6 + a * 5.0,
                (false, true) => -1.6 + again * 3.0,
                (false, false) => 9.0,
            };
            let (plain, s) = match p.tier {
                None => (0.7, size * 0.8),
                Some(_) => (1.0, size),
            };
            let col = gem_color(p.tier, t, i);
            let halo = if rich { 0.55 } else { 0.25 };
            let glow = (s * 0.8, rgba(col, halo * shows));
            d.out
                .push(gem(c, s * scale, rgba(col, plain * shows), glow, glint));
            if p.tier == Some(Tier::Diva) {
                let light = rgba(col, 0.22 * shows);
                d.lights.push(disc(c, 0.0, [0.0; 4], light, s * 2.0));
                for (k, (ox, oy)) in [(0.75, -0.55), (-0.7, 0.35)].into_iter().enumerate() {
                    let tw = 0.5 + 0.5 * (t as f32 * 5.0 + i as f32 + k as f32 * 2.1).sin();
                    let at = (c.0 + ox * s, c.1 + oy * s);
                    let ray = s * (0.5 + 0.5 * tw) * scale;
                    let shine = rgba(WHITE, tw * shows * room.calm);
                    d.out.push(sparkle(at, ray, t as f32 * 1.5, shine));
                }
                d.light_sprays.push(Spray {
                    at: c,
                    half: (0.0, 0.0),
                    start: (t - f64::from(a)) as f32,
                    drift: s * 0.6,
                    size: s * 1.4,
                    seed: 200 + i as u32,
                    rgb: [1.0, 0.95, 1.0],
                    kind: Kind::Scored,
                });
            }
        }
        let bottom = places.iter().map(|p| p.1).fold(0.0, f32::max);
        self.tally(d, room, shown, (size, bottom + size * 1.3));
    }

    /// How many phrases reached each tier, the best first, under the gems
    /// (`size` across, the last row's bottom at `below`), once all are in.
    fn tally(
        &mut self,
        d: &mut Drawn,
        room: &Room,
        shown: &ResultShown,
        (size, below): (f32, f32),
    ) {
        let (t, reveal, r) = (shown.t, &shown.shown.reveal, &shown.shown.result);
        let Some(last) = r.phrases.len().checked_sub(1) else {
            return;
        };
        let done = reveal
            .find(|b| matches!(b, Beat::Gem(i, _) if *i == last))
            .map(|(at, _)| t - at - 0.25);
        let shows = fade(done, 0.4);
        if shows <= 0.0 {
            return;
        }
        let counts = r.tier_counts();
        let tiers: Vec<(Tier, u32)> = Tier::ALL
            .iter()
            .rev()
            .map(|&tier| (tier, counts[tier as usize]))
            .filter(|(_, n)| *n > 0)
            .collect();
        let u = room.u;
        let px = (2.3 * u).round().max(8.0);
        let chip = size.max(0.8 * u).min(1.1 * u);
        let texts: Vec<String> = tiers
            .iter()
            .map(|(tier, n)| format!("{n} {}", tier.name()))
            .collect();
        let widths: Vec<f32> = texts
            .iter()
            .map(|s| self.measure(Face::NUMBER, s, px))
            .collect();
        let each = chip * 2.6 + px * 1.2;
        let total = widths.iter().sum::<f32>() + each * tiers.len() as f32;
        let cy = below + 2.6 * u;
        let mut left = room.x + room.w / 2.0 - total / 2.0;
        for (((tier, _), text), tw) in tiers.iter().zip(&texts).zip(widths) {
            let col = gem_color(Some(*tier), t, 0);
            let halo = (chip * 0.6, rgba(col, 0.3 * shows));
            d.out
                .push(gem((left + chip, cy), chip, rgba(col, shows), halo, -9.0));
            let soft = rgba(WHITE, 0.8 * shows);
            let at = (left + chip * 2.6, cy);
            self.words(
                &mut d.out,
                (Face::NUMBER, text),
                (px, 1.0),
                at,
                0.0,
                soft,
                false,
            );
            left += tw + each;
        }
    }

    /// The verdict, `hit` seconds after its beat (before it: less than
    /// none): its slot while it waits; a medal falling for the last
    /// [`DROP`] before its beat and stamped down on it (a shock ring,
    /// sparkles, a jackpot for Gold), a star popping up on its beat; its
    /// ribbon, its rays turning behind it, its sheen sweeping over it now
    /// and then while the screen twinkles, its words under it.
    pub(super) fn verdict(
        &mut self,
        d: &mut Drawn,
        room: &Room,
        shown: &ResultShown,
        tint: &Tint,
        hit: Option<f32>,
    ) {
        let (t, r) = (shown.t, &shown.shown.result);
        let u = room.u;
        let radius = (12.5 * u).min(room.w * 0.1);
        let c = room.at(0.73, 0.3);
        let verdict = r.verdict();
        let medal = match verdict {
            Verdict::Medal(m) => Some(m),
            Verdict::Cheer | Verdict::Whole => None,
        };
        let shown_from = if medal.is_some() { -FALL } else { 0.0 };
        let Some(hit) = hit.filter(|&h| h >= shown_from) else {
            self.slot(d, (c, radius), u, t);
            return;
        };
        let (light, deep, strength) = look(verdict, tint);
        let scale = match medal {
            None => land(Some(f64::from(hit))).0,
            Some(_) if hit < 0.0 => {
                let falling = clamp01((hit + FALL) / FALL);
                2.2 - 1.2 * falling * falling
            }
            Some(_) => 1.0 + 0.1 * (-hit * 10.0).exp() * (hit * 28.0).cos(),
        };
        let shows = clamp01((hit - shown_from) / 0.08);
        let after = |k: f32| fade(Some(f64::from(hit)), k);
        // Its rays turning behind it, and its light.
        let reach = radius * 3.2;
        let mut rays = shape(kind::RAYS, c, reach * 1.6, [c.0, c.1, reach, 14.0]);
        rays.params[1] = t as f32 * 0.5;
        rays.color = rgba(light, strength * after(0.6));
        d.out.push(rays);
        let strong = if medal.is_some() { 0.5 } else { 0.15 };
        let flash = 0.08 + strong * (-hit.max(0.0) * 5.0).exp();
        let lit = rgba(light, flash * after(0.05));
        d.lights.push(disc(c, 0.0, [0.0; 4], lit, radius * 1.5));
        let r_now = radius * scale;
        if medal.is_some() {
            ribbon(d, (c, r_now), tint, shows);
        }
        // It, its sheen sweeping over it as it lands, then every few
        // seconds while the screen twinkles.
        let sheen = match (hit < 1.0, room.calm > 0.5) {
            (true, _) => -1.6 + hit.max(0.0) * 3.2,
            (false, true) => -1.6 + ((hit + 1.0) % 3.4) * 2.4,
            (false, false) => 9.0,
        };
        let badge = if medal.is_some() { 0.0 } else { 1.0 };
        let reach = r_now * 1.05 + radius * 0.3 + 2.0;
        let mut m = shape(kind::MEDAL, c, reach, [c.0, c.1, r_now, badge]);
        m.color = rgba(light, shows);
        m.color2 = rgba(deep, shows);
        m.params[1] = radius * 0.3;
        m.params[2] = sheen;
        d.out.push(m);
        // Sparkles as it lands; a medal stamped down, more.
        let stamp = t as f32 - hit;
        d.light_sprays.push(Spray {
            at: c,
            half: (0.0, 0.0),
            start: stamp,
            drift: radius * 0.2,
            size: radius * 0.7,
            seed: 301,
            rgb: light,
            kind: Kind::Scored,
        });
        if let Some(m) = medal {
            stamped(d, (c, radius), m, (stamp, hit), u);
        }
        // A few sparkles twinkling round its rim.
        for k in 0..3 {
            let turn = k as f32 * 2.1 + 0.6;
            let tw = 0.5 + 0.5 * (t as f32 * 3.0 + k as f32 * 1.7).sin();
            let at = (
                c.0 + turn.cos() * r_now * 0.95,
                c.1 + turn.sin() * r_now * 0.95,
            );
            let shine = rgba(GOLD_TOP, tw * after(0.3) * room.calm);
            d.out.push(sparkle(at, radius * 0.28 * tw, t as f32, shine));
        }
        self.naming(d, room, shown, (c, radius), (tint, light), hit);
    }

    /// The medal's slot while it is waiting, faint, a question in it.
    fn slot(&mut self, d: &mut Drawn, (c, radius): ((f32, f32), f32), u: f32, t: f64) {
        let shows = 0.3 * fade(Some(t - 0.4), 0.6);
        let ring = rgba(WHITE, shows);
        d.out
            .push(pill(c, (radius, radius), [0.0; 4], ring, 0.0, 2.0, 0.0));
        let q = (Face::SHOUT, "?");
        let px = ((7.0 * u).round(), 1.0);
        self.words(&mut d.out, q, px, c, 0.5, ring, false);
    }

    /// The verdict's words under it (`light` its colour) at `c`: its name
    /// popping up `hit` seconds after its beat, the line under it after.
    fn naming(
        &mut self,
        d: &mut Drawn,
        room: &Room,
        shown: &ResultShown,
        (c, radius): ((f32, f32), f32),
        (tint, light): (&Tint, Rgb),
        hit: f32,
    ) {
        let u = room.u;
        let r = &shown.shown.result;
        let medal = matches!(r.verdict(), Verdict::Medal(_));
        let after = |k: f32| fade(Some(f64::from(hit)), k);
        let (name, line) = verdict_words(r);
        let (pop, seen) = land(Some(f64::from(hit) - 0.05).filter(|&x| x >= 0.0));
        let px = (if medal { 6.5 } else { 5.0 } * u).round().max(12.0);
        let ny = c.1 + radius + 5.5 * u;
        let width = room.w * 0.4;
        let (name, px) = self.fitted(Face::SHOUT, name, px, width);
        let name_color = rgba(if medal { light } else { WHITE }, seen);
        self.words(
            &mut d.out,
            (Face::SHOUT, &name),
            (px, pop),
            (c.0, ny),
            0.5,
            name_color,
            true,
        );
        let small = (2.6 * u).round().max(9.0);
        let (line, small) = self.fitted(Face::NUMBER, line, small, width);
        let soft = rgba(
            if medal { WHITE } else { tint.soft },
            0.85 * after(0.5) * seen,
        );
        let ly = ny + px * 0.55 + 3.2 * u;
        self.words(
            &mut d.out,
            (Face::NUMBER, &line),
            (small, 1.0),
            (c.0, ly),
            0.5,
            soft,
            true,
        );
        if r.verdict() == Verdict::Medal(Medal::Gold) {
            // Gold's name lit too.
            let lit = rgba(GOLD, 0.3 * seen);
            d.lights.push(disc((c.0, ny), 0.0, [0.0; 4], lit, px * 1.4));
        }
    }
}

/// The medal's ribbon at `c` (it `r` across): two straps from over its
/// top down behind it, in the tint's two colours, a stripe down each.
fn ribbon(d: &mut Drawn, (c, r): ((f32, f32), f32), tint: &Tint, shows: f32) {
    for (side, col) in [(-1.0f32, tint.ribbon.0), (1.0, tint.ribbon.1)] {
        let from = (c.0 + side * 0.22 * r, c.1 - 0.4 * r);
        let to = (c.0 + side * 0.66 * r, c.1 - 1.22 * r);
        let edge = rgba(col.map(|k| k * 0.6), shows);
        d.out.push(segment(to, from, 0.28 * r, edge, [0.0; 4], 0.0));
        d.out
            .push(segment(to, from, 0.23 * r, rgba(col, shows), [0.0; 4], 0.0));
        let stripe = rgba(WHITE, 0.3 * shows);
        d.out
            .push(segment(to, from, 0.035 * r, stripe, [0.0; 4], 0.0));
    }
}

/// Medal `m` at `c` stamped down at `stamp` (the reveal's clock: its beat),
/// `hit` seconds ago: a shock ring going out; a jackpot for Gold, stars and
/// notes for Gold and Silver.
fn stamped(
    d: &mut Drawn,
    (c, radius): ((f32, f32), f32),
    m: Medal,
    (stamp, hit): (f32, f32),
    u: f32,
) {
    if (0.0..0.55).contains(&hit) {
        let k = hit / 0.55;
        let ring = radius * (1.05 + 1.1 * ease_out(k));
        let line = rgba(WHITE, 0.7 * (1.0 - k) * (1.0 - k));
        let width = radius * 0.06 * (1.0 - k) + 1.0;
        d.out.push(pill(
            c,
            (ring, ring),
            [0.0; 4],
            line,
            radius * 0.2,
            width,
            0.0,
        ));
    }
    let burst = |at, size, seed, kind, rgb| Spray {
        at,
        half: (0.0, 0.0),
        start: stamp,
        drift: 0.0,
        size,
        seed,
        rgb,
        kind,
    };
    if m == Medal::Gold {
        let coins = burst(c, 2.2 * u, 307, Kind::Jackpot, [1.0, 0.8, 0.3]);
        d.light_sprays.push(coins);
    }
    if matches!(m, Medal::Gold | Medal::Silver) {
        let top = (c.0, c.1 - radius);
        d.sprays
            .push(burst(top, radius * 0.5, 311, Kind::StepUp, WHITE));
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{frame_at, result};
    use super::*;

    const GOLD_BEAT: Beat = Beat::Verdict(Verdict::Medal(Medal::Gold));

    /// The medal falls in large before its beat and is stamped down on it
    /// at its size, the screen shaking only from then, and not for a star.
    #[test]
    fn the_medal_is_stamped_down_on_its_beat() {
        let r = result(Some(Medal::Gold), true, None);
        let reveal = crate::core::singing::Reveal::of(&r);
        let at = reveal.when(GOLD_BEAT).unwrap();
        let radius = |t: f64| {
            let f = frame_at(&r, t);
            f.instances
                .iter()
                .find(|i| i.params[0] == kind::MEDAL)
                .map(|i| i.shape[2])
        };
        assert_eq!(radius(at - DROP - 0.05), None);
        let falling = radius(at - DROP * 0.8).unwrap();
        let (stamped, landed) = (radius(at).unwrap(), radius(at + 2.0).unwrap());
        assert!(falling > landed * 1.5, "{falling} {landed}");
        assert!((stamped - landed).abs() < landed * 0.11 && (landed - 112.5).abs() < 0.5);
        let gold = Verdict::Medal(Medal::Gold);
        assert_eq!(shake(gold, Some(-0.1), 9.0), (0.0, 0.0));
        let (dx, dy) = shake(gold, Some(0.02), 9.0);
        assert!(dx.abs() + dy.abs() > 0.5);
        assert_eq!(shake(gold, Some(3.0), 9.0), (0.0, 0.0));
        assert_eq!(shake(Verdict::Cheer, Some(0.02), 9.0), (0.0, 0.0));
    }

    fn room(w: f32, h: f32) -> Room {
        Room {
            x: 0.0,
            y: 0.0,
            w,
            h,
            u: h / 100.0,
            shake: (0.0, 0.0),
            bottom: h * 0.92,
            calm: 1.0,
        }
    }

    /// Many phrases fit in at most three rows, inside the window; in a
    /// narrow (portrait) window hundreds take more rows, still inside it
    /// and clear of the words above and below.
    #[test]
    fn many_gems_fit() {
        let rows = |places: &[(f32, f32)]| {
            places
                .iter()
                .map(|p| p.1 as i32)
                .collect::<std::collections::BTreeSet<_>>()
                .len()
        };
        let wide = room(1600.0, 900.0);
        for n in [1, 12, 40, 120, 200] {
            let (size, places) = gem_places(n, &wide);
            assert_eq!(places.len(), n);
            assert!(rows(&places) <= 3, "{n}");
            assert!(places
                .iter()
                .all(|p| p.0 - size > 100.0 && p.0 + size < 1500.0));
        }
        let tall = room(500.0, 900.0);
        for n in [12, 40, 200] {
            let (size, places) = gem_places(n, &tall);
            assert!(
                places
                    .iter()
                    .all(|p| p.0 - size > 0.0 && p.0 + size < 500.0),
                "{n}"
            );
            let ys = places.iter().map(|p| p.1);
            let (top, bottom) = ys.fold((f32::MAX, f32::MIN), |(a, b), y| (a.min(y), b.max(y)));
            assert!(
                top - size > 900.0 * 0.55 && bottom + size < 900.0 * 0.66,
                "{n}"
            );
            if n <= 40 {
                assert!(rows(&places) <= 3, "{n}");
            }
        }
    }
}
