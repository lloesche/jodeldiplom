//! The diploma's star stickers: one a phrase, popping on in song order in
//! a row near its foot, each a little askew, in its tier's colour (a plain
//! grey one for a phrase that reached none; Amazing and Diva glinting now
//! and then while the screen twinkles, Diva's prismatic, twinkling and
//! glowing too); once all are on, how many reached each tier under them.

use super::super::super::rainbow::hue;
use super::super::super::sparks::noise;
use super::super::meter::tier_color;
use super::super::{disc, rgba, Rgb, Stage, WHITE};
use super::{burst, fade, land, sparkle, sticker, Drawn, Pen, ResultShown, Room, Written, PRINT};
use crate::core::shine::Tier;
use crate::core::singing::Beat;
use crate::gui::particles::Kind;

/// How far down the sheet the stickers' rows are centred, how much of
/// its width and height they may take.
const ROW_Y: f32 = 0.655;
const ROW_WIDE: f32 = 0.56;
const ROW_BAND: f32 = 0.075;
/// A sticker for a phrase that reached no tier.
const PLAIN: Rgb = [0.74, 0.72, 0.68];

/// Where `n` stickers go: their size, in rows (at most three, the
/// stickers smaller the more there are, down to a third of a unit out each
/// way; past that, hundreds of phrases on a narrow sheet, as many rows as
/// they take) of `in_row`, centred on `middle`.
#[derive(Clone, Copy, Debug)]
struct Places {
    n: usize,
    size: f32,
    rows: usize,
    in_row: usize,
    middle: (f32, f32),
}

impl Places {
    /// Where `n` stickers go in `room`.
    fn of(n: usize, room: &Room) -> Places {
        let mut size = 2.1 * room.s;
        let smallest = 0.35 * room.s;
        let width = room.across(ROW_WIDE);
        let band = room.sheet.3 * ROW_BAND;
        let per = |size: f32| ((width / (size * 2.5)).floor() as usize).max(1);
        let high = |size: f32| n.div_ceil(per(size)) as f32 * size * 2.3;
        while (n.div_ceil(per(size)) > 3 || high(size) > band) && size > smallest {
            size = (size * 0.9).max(smallest);
        }
        let rows = n.div_ceil(per(size)).max(1);
        Places {
            n,
            size,
            rows,
            in_row: n.div_ceil(rows).max(1),
            middle: room.at(0.5, ROW_Y),
        }
    }

    /// Sticker `i`'s middle.
    fn at(&self, i: usize) -> (f32, f32) {
        let (row, j) = (i / self.in_row, i % self.in_row);
        let here = self.in_row.min(self.n - row * self.in_row);
        let x = self.middle.0 + (j as f32 - (here as f32 - 1.0) / 2.0) * self.size * 2.5;
        let y = self.middle.1 + self.row(row);
        (x, y)
    }

    /// How far row `row` is down from their middle.
    fn row(&self, row: usize) -> f32 {
        (row as f32 - (self.rows as f32 - 1.0) / 2.0) * self.size * 2.3
    }

    /// The last row's middle, down.
    fn bottom(&self) -> f32 {
        self.middle.1 + self.row(self.rows - 1)
    }
}

/// A tier's sticker colour at `t` (Diva's running through the rainbow a
/// little, `i` its phase).
fn star_color(tier: Option<Tier>, t: f64, i: usize) -> Rgb {
    let Some(tier) = tier else {
        return PLAIN;
    };
    let base = tier_color(Some(tier));
    if tier == Tier::Diva {
        let h = hue(t as f32 * 0.25 + i as f32 * 0.13);
        return [0, 1, 2].map(|k| base[k] + (h[k] - base[k]) * 0.35);
    }
    base
}

impl Stage {
    /// The phrases' stickers, popping on one by one in song order (each
    /// glinting as it lands, Amazing and Diva glinting now and then while
    /// the screen twinkles, Diva twinkling and glowing too), and, once all
    /// are on, how many reached each tier under them (`words`' tally).
    pub(super) fn stars(
        &mut self,
        d: &mut Drawn,
        room: &Room,
        shown: &ResultShown,
        words: &Written,
    ) {
        let (t, reveal, r) = (shown.t, &shown.shown.reveal, &shown.shown.result);
        let places = Places::of(r.phrases.len(), room);
        let size = places.size;
        for (i, p) in r.phrases.iter().enumerate() {
            let age = reveal.age(Beat::Gem(i, p.tier), t);
            let (scale, shows) = land(age);
            let Some(a) = age.map(|a| a as f32) else {
                continue;
            };
            let c = places.at(i);
            let rich = matches!(p.tier, Some(Tier::Amazing | Tier::Diva));
            let again = (a + i as f32 * 0.37) % 2.4;
            let glint = match (a < 0.7, rich && room.calm > 0.5) {
                (true, _) => -1.6 + a * 5.0,
                (false, true) => -1.6 + again * 3.0,
                (false, false) => 9.0,
            };
            let (plain, s) = match p.tier {
                None => (0.55, size * 0.75),
                Some(_) => (1.0, size),
            };
            let turn = (noise(i as u32 * 7 + 3) - 0.5) * 0.6;
            let col = star_color(p.tier, t, i);
            let halo = if p.tier == Some(Tier::Diva) { 0.5 } else { 0.0 };
            let glow = (s * 0.7, rgba(col, halo * shows));
            let ink = rgba(col, plain * shows);
            d.out.push(sticker(c, (s * scale, turn), ink, glow, glint));
            if p.tier == Some(Tier::Diva) {
                let light = rgba(col, 0.12 * shows);
                d.lights.push(disc(c, 0.0, [0.0; 4], light, s * 1.6));
                for (k, (ox, oy)) in [(0.85, -0.6), (-0.8, 0.45)].into_iter().enumerate() {
                    let tw = 0.5 + 0.5 * (t as f32 * 5.0 + i as f32 + k as f32 * 2.1).sin();
                    let at = (c.0 + ox * s, c.1 + oy * s);
                    let ray = s * (0.5 + 0.5 * tw) * scale;
                    let shine = rgba(WHITE, tw * shows * room.calm);
                    d.lights.push(sparkle(at, ray, t as f32 * 1.5, shine));
                }
                let seed = (200 + i as u32, col);
                let at = t - f64::from(a);
                d.sprays.push(burst(c, at, s * 1.2, seed, Kind::StepUp));
            }
        }
        if !r.phrases.is_empty() {
            self.tally(d, room, shown, (words, places.bottom() + size * 1.2));
        }
    }

    /// How many phrases reached each tier (`words`' tally, the best
    /// first), under the stickers (the last row's bottom at `below`), once
    /// all are on.
    fn tally(
        &mut self,
        d: &mut Drawn,
        room: &Room,
        shown: &ResultShown,
        (words, below): (&Written, f32),
    ) {
        let (t, reveal, r) = (shown.t, &shown.shown.reveal, &shown.shown.result);
        let last = r.phrases.len().saturating_sub(1);
        let done = reveal
            .find(|b| matches!(b, Beat::Gem(i, _) if *i == last))
            .map(|(at, _)| t - at - 0.25);
        let shows = fade(done, 0.4);
        if shows <= 0.0 {
            return;
        }
        let s = room.s;
        let px = (2.4 * s).round().max(8.0);
        let chip = 1.05 * s;
        let each = chip * 2.6 + px * 1.2;
        let mut total = 0.0;
        for (_, text) in &words.tally {
            total += self.measure(PRINT, text, px) + each;
        }
        let cy = below + 2.2 * s;
        let mut left = room.at(0.5, 0.0).0 - total / 2.0;
        let ink = rgba(room.inks.ink, 0.8 * shows);
        for (tier, text) in &words.tally {
            let col = star_color(Some(*tier), t, 0);
            let none = (0.0, [0.0; 4]);
            let chip_at = (left + chip, cy);
            d.out
                .push(sticker(chip_at, (chip, 0.0), rgba(col, shows), none, 9.0));
            let at = ((left + chip * 2.6, cy), 0.0);
            let tw = self.written(&mut d.out, (PRINT, text), (px, 1.0), at, ink, Pen::FLAT);
            left += tw + each;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn room(w: f32, h: f32) -> Room {
        let tint = &super::super::Tint::STAGE;
        let look = super::super::seal::look(crate::core::singing::Verdict::Cheer, tint);
        Room::new((0.0, 0.0, w, h), h * 0.9, tint, look)
    }

    /// Many phrases fit in at most three rows, inside the sheet's middle;
    /// on a narrow (upright) sheet hundreds take more rows, still inside
    /// it and clear of the entries above and the grade below.
    #[test]
    fn many_stars_fit() {
        for (w, h, most) in [(1600.0, 900.0, 200), (500.0, 900.0, 40)] {
            let room = room(w, h);
            let (sx, sy, sw, sh) = room.sheet;
            for n in [1, 12, 40, 120, 200] {
                let p = Places::of(n, &room);
                let size = p.size;
                let all: Vec<(f32, f32)> = (0..n).map(|i| p.at(i)).collect();
                assert!(
                    all.iter()
                        .all(|c| c.0 - size > sx + sw * 0.2 && c.0 + size < sx + sw * 0.8),
                    "{w}: {n}"
                );
                let (top, bottom) = all
                    .iter()
                    .fold((f32::MAX, f32::MIN), |(a, b), c| (a.min(c.1), b.max(c.1)));
                assert!(
                    top - size > sy + sh * 0.6 && bottom + size < sy + sh * 0.71,
                    "{w}: {n}"
                );
                assert!((bottom - p.bottom()).abs() < 1e-3);
                if n <= most {
                    assert!(p.rows <= 3, "{w}: {n}");
                }
            }
        }
    }
}
