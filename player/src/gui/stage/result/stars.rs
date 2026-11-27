//! The diploma's star stickers: one a phrase, popping on in song order in
//! a row near its foot, each a little askew, in its tier's colour (a plain
//! grey one for a phrase that reached none) and looking as its tier's row
//! says ([`crate::tiers::Sticker`]: glinting now and then while the screen
//! twinkles, prismatic, glowing in a halo); once all are on, how many
//! reached each tier under them, in a row (two, should one not fit across
//! the sheet).

use super::super::super::shine::hue;
use super::super::super::sparks::noise;
use super::super::meter::tier_color;
use super::super::{disc, rgba, Rgb, Stage, WHITE};
use super::{burst, fade, land, sparkle, sticker, Pen, ResultShown, Room, Written, PRINT};
use crate::core::karaoke::result::Beat;
use crate::core::shine::Tier;
use crate::gui::particles::Kind;
use crate::gui::stage::gpu::StageFrame;
use crate::theme::Theme;

/// How far down the sheet the stickers' rows are centred, how much of
/// its width and height they may take.
const ROW_Y: f32 = 0.655;
const ROW_WIDE: f32 = 0.56;
const ROW_BAND: f32 = 0.075;
/// How much of the sheet's width the tally may take in a row (inside its
/// border); wider, it takes two.
const TALLY_WIDE: f32 = 0.9;
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

/// A tier's sticker colour in `theme` at `t` (a prismatic one's running
/// through the rainbow a little, `i` its phase).
fn star_color(theme: &Theme, tier: Option<Tier>, t: f64, i: usize) -> Rgb {
    let Some(tier) = tier else {
        return PLAIN;
    };
    let base = tier_color(theme, Some(tier));
    if tier.spec().prismatic {
        let h = hue(t as f32 * 0.25 + i as f32 * 0.13);
        return [0, 1, 2].map(|k| base[k] + (h[k] - base[k]) * 0.35);
    }
    base
}

impl Stage {
    /// The phrases' stickers, popping on one by one in song order (each
    /// glinting as it lands, and as its tier's row says: glinting now and
    /// then while the screen twinkles, prismatic, a halo with its light,
    /// twinkles and a burst), and, once all are on, how many reached each
    /// tier under them (`words`' tally); each in its tier's colour in
    /// `theme`.
    pub(super) fn stars(
        &mut self,
        d: &mut StageFrame,
        room: &Room,
        shown: &ResultShown,
        (words, theme): (&Written, &Theme),
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
            let look = p.tier.map(|t| t.spec().sticker);
            let rich = look.is_some_and(|l| l.glints);
            let halo = look.map_or(0.0, |l| l.halo);
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
            let col = star_color(theme, p.tier, t, i);
            let glow = (s * 0.7, rgba(col, halo * shows));
            let ink = rgba(col, plain * shows);
            d.instances
                .push(sticker(c, (s * scale, turn), ink, glow, glint));
            if halo > 0.0 {
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
            let below = places.bottom() + size * 1.2;
            self.tally(d, room, shown, (words, below, theme));
        }
    }

    /// How many phrases reached each tier (`words`' tally, the best
    /// first), under the stickers (the last row's bottom at `below`), once
    /// all are on, each in its tier's colour in `theme`: in a row across
    /// the sheet's middle, or two where one would not fit ([`tally_rows`]).
    fn tally(
        &mut self,
        d: &mut StageFrame,
        room: &Room,
        shown: &ResultShown,
        (words, below, theme): (&Written, f32, &Theme),
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
        let wide: Vec<f32> = (words.tally.iter())
            .map(|(_, text)| self.measure(PRINT, text, px) + each)
            .collect();
        let cy = below + 2.2 * s;
        let ink = rgba(room.inks.ink, 0.8 * shows);
        for (row, items) in tally_rows(&wide, room.across(TALLY_WIDE)).enumerate() {
            let total = wide[items.clone()].iter().fold(0.0, |a, w| a + w);
            let cy = cy + row as f32 * px * 1.6;
            let mut left = room.at(0.5, 0.0).0 - total / 2.0;
            for (tier, text) in &words.tally[items] {
                let col = star_color(theme, Some(*tier), t, 0);
                let none = (0.0, [0.0; 4]);
                let chip_at = (left + chip, cy);
                d.instances
                    .push(sticker(chip_at, (chip, 0.0), rgba(col, shows), none, 9.0));
                let at = ((left + chip * 2.6, cy), 0.0);
                let tw = self.written(
                    &mut d.instances,
                    (PRINT, text),
                    (px, 1.0),
                    at,
                    ink,
                    Pen::FLAT,
                );
                left += tw + each;
            }
        }
    }
}

/// The tally's rows: its entries `wide` across each, in one row if they
/// fit in `room`, else in two (the first the longer by one where they are
/// odd).
fn tally_rows(wide: &[f32], room: f32) -> impl Iterator<Item = std::ops::Range<usize>> {
    let n = wide.len();
    let fits = wide.iter().sum::<f32>() <= room;
    let split = if fits || n < 2 { n } else { n.div_ceil(2) };
    [0..split, split..n].into_iter().filter(|r| !r.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn room(w: f32, h: f32) -> Room {
        let tint = &super::super::Tint::STAGE;
        let look = super::super::seal::look(crate::core::karaoke::result::Verdict::Cheer, tint);
        let area = (0.0, 0.0, w, h);
        super::super::tests::room_in(&mut Stage::default(), area, look)
    }

    /// The tally in a row while it fits, else in two, the first the
    /// longer; none without entries.
    #[test]
    fn the_tally_wraps_only_when_too_wide() {
        let rows = |wide: &[f32], room| -> Vec<(usize, usize)> {
            tally_rows(wide, room).map(|r| (r.start, r.end)).collect()
        };
        assert_eq!(rows(&[10.0; 4], 40.0), [(0, 4)]);
        assert_eq!(rows(&[10.0; 4], 39.0), [(0, 2), (2, 4)]);
        assert_eq!(rows(&[10.0; 5], 30.0), [(0, 3), (3, 5)]);
        assert_eq!(rows(&[50.0], 30.0), [(0, 1)]);
        assert!(rows(&[], 30.0).is_empty());
    }

    /// The tally of every tier fits in a row on a wide sheet, in two on
    /// the upright ones the diploma is drawn on, each row inside the
    /// sheet.
    #[test]
    fn every_tier_s_tally_fits_in_a_row_or_two() {
        let mut stage = Stage::default();
        for (w, h, most) in [(1600.0, 900.0, 1), (900.0, 1400.0, 2), (500.0, 900.0, 2)] {
            let room = room(w, h);
            let px = (2.4 * room.s).round().max(8.0);
            let each = 1.05 * room.s * 2.6 + px * 1.2;
            let wide: Vec<f32> = Tier::ALL
                .iter()
                .rev()
                .map(|t| stage.measure(PRINT, &format!("99 {}", t.name()), px) + each)
                .collect();
            let rows: Vec<_> = tally_rows(&wide, room.across(TALLY_WIDE)).collect();
            assert!(rows.len() <= most, "{w}: {rows:?}");
            for r in rows {
                let across = wide[r].iter().sum::<f32>();
                assert!(across <= room.sheet.2 * 0.95, "{w}: {across}");
            }
        }
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
