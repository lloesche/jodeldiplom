//! The shine on the stage ([`super::Stage::shine`]): the meter at the top
//! (filling as the phrase is sung, its words hit, then springing to the
//! shine the phrase reached, its tier's name popping under it), the
//! tiers' looks over the lane (Good the rainbow streak itself, Great a
//! glow round the lane, Amazing flames blazing off its edges, Diva angels
//! descending and a jackpot of coins from the meter), and the song's
//! progress along the bottom: its sung stretches, the part sung coloured
//! phrase by phrase by the tier each reached, a mark a phrase popping as
//! its tier is decided (a little jackpot on Diva), and the time left in a
//! ring that drains as the song goes, in the rainbow while the singing
//! shines. The processor gives the GPU only where each is and when it
//! changed ([`crate::gui::glory`], [`crate::gui::fanfare`]).

use super::{rgba, Lane, Stage, BREAK, GOLD, PROGRESS, WHITE};
use crate::core::shine::{Shone, Tier};
use crate::core::singing::{Cue, Singing};
use crate::gui::fanfare;
use crate::gui::glory::{meter_fill, position, MARKS};
use crate::gui::particles::Spray;
use crate::gui::rainbow::Glow;
use crate::gui::stage_gpu::{kind, Instance};
use crate::gui::streak::Step;

/// How long a tier's name shows under the meter, in seconds.
const VERDICT: f32 = 1.8;
/// How often a new flight of angels comes while the singing is Diva.
const ANGELS_EVERY: f32 = 2.2;

/// A tier's colour (sRGB); none's a cool grey.
pub fn tier_color(tier: Option<Tier>) -> [f32; 3] {
    match tier {
        None => [0.55, 0.6, 0.7],
        Some(Tier::Good) => [0.35, 0.92, 0.62],
        Some(Tier::Great) => [1.0, 0.78, 0.22],
        Some(Tier::Amazing) => [1.0, 0.35, 0.6],
        Some(Tier::Diva) => [0.96, 0.93, 1.0],
    }
}

/// The meter between frames: where its fill springs from and to, and
/// since when (the singing's clock); the last phrase shown, and when;
/// when the last flight of angels came.
#[derive(Debug, Default)]
pub struct Meter {
    from: f32,
    to: f32,
    since: f32,
    shown: Option<(Shone, f32)>,
    angels: f32,
}

impl Meter {
    /// The fill to spring to `to` at `clock`, from where it shows now.
    fn aim(&mut self, to: f32, clock: f32) {
        if (to - self.to).abs() > 1e-4 {
            self.from = meter_fill(self.from, self.to, self.since, clock);
            self.to = to;
            self.since = clock;
        }
    }
}

/// How the line being sung at `clock` (the singing's time) is going: its
/// words hit so far over all its words (a rough share of what it will
/// shine, for the meter while it is sung).
fn sung_so_far(singing: &Singing, clock: f64) -> Option<f32> {
    let line = singing.notes.lines.iter().position(|l| {
        l.notes.first().is_some_and(|n| n.start <= clock)
            && l.notes.last().is_some_and(|n| n.end >= clock)
    })?;
    let words = {
        let mut w: Vec<usize> = singing.notes.lines[line]
            .notes
            .iter()
            .map(|n| n.word)
            .collect();
        w.dedup();
        w.len().max(1)
    };
    let start = singing.notes.lines[line].notes.first()?.start;
    let hit = singing
        .moments()
        .iter()
        .rev()
        .take_while(|m| m.at >= start)
        .filter(|m| matches!(m.cue, Cue::Word { end, hit: true, .. } if end >= start))
        .count();
    Some(hit as f32 / words as f32)
}

/// What the shine needs of the stage: the area, the lane and its box, the
/// pills' height, the song's time and the singing's clock.
pub struct Where {
    pub area: (f32, f32, f32, f32),
    pub lane: Lane,
    pub lane_box: ((f32, f32), (f32, f32)),
    pub now: f64,
    pub clock: f64,
}

/// What the shine draws: over the stage, over everything (the light
/// too), as light, and its bursts.
#[derive(Default)]
pub struct Drawn {
    pub out: Vec<Instance>,
    pub top: Vec<Instance>,
    pub lights: Vec<Instance>,
    pub light_sprays: Vec<Spray>,
}

/// A shape of `k` over `quad` with its numbers.
fn shape(k: f32, quad: [f32; 4], shape: [f32; 4], color: [f32; 4], params: [f32; 3]) -> Instance {
    Instance {
        quad,
        shape,
        color,
        color2: [0.0; 4],
        params: [k, params[0], params[1], params[2]],
        uv: [0.0; 4],
        strain: [0.0; 4],
    }
}

impl Stage {
    /// The shine for `singing` (the game on: `game`) at `at`, its streak's
    /// `step` this frame, into `d`; the song `length` seconds long.
    pub(super) fn shine(
        &mut self,
        d: &mut Drawn,
        (singing, game, length): (&Singing, bool, Option<f64>),
        at: &Where,
        step: &Step,
    ) {
        let (_, _, w, h) = at.area;
        let clock = at.clock as f32;
        let p = at.lane.pill;
        let (k, _) = step.glow.at(clock);
        let level = step.glow.level;
        if game && singing.reference().is_some() {
            self.meter_and_verdict(d, singing, at, step);
        }
        // Great: a glow round the lane; Amazing: flames off its edges.
        let ((cx, cy), (hx, hy)) = at.lane_box;
        let great = (k - 1.0).clamp(0.0, 1.0);
        if great > 0.01 {
            let reach = p * 1.6;
            d.lights.push(shape(
                kind::GLOW,
                [
                    cx - hx - reach,
                    cy - hy - reach,
                    cx + hx + reach,
                    cy + hy + reach,
                ],
                [cx, cy, hx, hy],
                rgba([1.0, 0.74, 0.3], 1.0),
                [reach, 0.12 * great, 0.0],
            ));
        }
        let amazing = (k - 2.0).clamp(0.0, 1.0);
        if amazing > 0.01 {
            let high = p * 1.5;
            for (edge, out) in [(cy - hy, -1.0), (cy + hy, 1.0)] {
                let (y0, y1) = if out < 0.0 {
                    (edge - high * 1.4, edge + high * 0.3)
                } else {
                    (edge - high * 0.3, edge + high * 1.4)
                };
                d.lights.push(shape(
                    kind::BLAZE,
                    [cx - hx, y0, cx + hx, y1],
                    [cx - hx, edge, cx + hx, out],
                    [1.0; 4],
                    [high, amazing, 0.0],
                ));
            }
        }
        // Diva: a flight of angels every while (sought back before the
        // last, as if it came a while ago).
        if clock < self.meter.angels {
            self.meter.angels = clock - ANGELS_EVERY;
        }
        if level >= 4.0 && clock - self.meter.angels >= ANGELS_EVERY {
            self.meter.angels = clock;
            let seed = (at.clock * 1000.0) as u32 ^ 0x5EED;
            d.light_sprays
                .push(fanfare::angels(w, h * 0.75, h * 0.12, at.clock, seed));
        }
        self.progress(d, (singing, length), at, (&step.glow, step.phrase));
    }

    /// The meter, its fill, and the tier's name under it as a phrase is
    /// decided (a jackpot from it on Diva).
    fn meter_and_verdict(&mut self, d: &mut Drawn, singing: &Singing, at: &Where, step: &Step) {
        let (x, y, w, h) = at.area;
        let clock = at.clock as f32;
        let (mx, my) = (x + w * 0.5, y + h * 0.075);
        let half = (w * 0.17, (h * 0.016).max(5.0));
        if let Some(p) = step.phrase {
            self.meter.shown = Some((p, clock));
            // Within the tier reached (a shine past a bar with the pitch short
            // of Diva's shows short of it).
            let next = p.tier.map_or(0, |t| t as usize + 1);
            let cap = MARKS.get(next).map_or(1.0, |m| m - 0.025);
            self.meter.aim(position(p.of_diva()).min(cap), clock);
            if p.tier == Some(Tier::Diva) {
                let seed = (at.clock * 997.0) as u32;
                d.light_sprays.push(fanfare::jackpot(
                    (mx + half.0, my),
                    half.1 * 2.2,
                    at.clock,
                    seed,
                ));
            }
        }
        let verdict_on = self.meter.shown.is_some_and(|(_, t)| clock - t < VERDICT);
        if !verdict_on {
            if let Some(live) = sung_so_far(singing, at.clock) {
                self.meter.aim(position(live), clock);
            }
        }
        let m = &self.meter;
        let pad = half.1 * 1.6;
        d.out.push(shape(
            kind::METER,
            [
                mx - half.0 - pad,
                my - half.1 - pad,
                mx + half.0 + pad,
                my + half.1 + pad,
            ],
            [mx, my, half.0, half.1],
            [1.0; 4],
            [m.from, m.to, m.since],
        ));
        // The tier's name, popping and fading.
        let Some((p, t)) = self.meter.shown else {
            return;
        };
        let age = clock - t;
        if age >= VERDICT {
            return;
        }
        let Some(tier) = p.tier else {
            return;
        };
        let label = format!("{}!", tier.name());
        let px = (h * 0.07).round().clamp(18.0, 140.0);
        let tw = self.text.measure(&label, px);
        // Over the meter, its middle on the meter's.
        let ty = my - self.cap_middle(px) + super::COVER_PAD as f32;
        let c = tier_color(Some(tier));
        // On a badge of its own over the meter, ringed in the tier's colour,
        // over the light (its bloom would wash it out); both popping round
        // the meter's middle and fading as the shader times them
        // ([`crate::gui::glory::label_pop`]).
        let half = (tw / 2.0 + px * 0.45, px * 0.62);
        let m = half.0.max(half.1) * 2.0;
        d.top.push(Instance {
            quad: [mx - m, my - m, mx + m, my + m],
            shape: [mx, my, half.0, half.1],
            color: [0.04, 0.03, 0.09, 0.82],
            color2: rgba(c, 0.9),
            params: [kind::BADGE, t, VERDICT, (px * 0.06).max(1.5)],
            uv: [0.0; 4],
            strain: [0.0; 4],
        });
        let text = self.text(
            &mut d.top,
            &label,
            px,
            1.0,
            (mx - tw / 2.0, ty),
            f32::NEG_INFINITY,
            rgba(c, 1.0),
            rgba(c, 1.0),
        );
        if let Some(text) = text {
            text.shape = [mx, my, t, VERDICT];
            crate::gui::rainbow::in_rainbow(text, &step.glow);
        }
    }

    /// The song's progress along the bottom: its sung stretches, the part
    /// sung coloured by the tiers its phrases reached, a mark a phrase, the
    /// time left in its ring.
    fn progress(
        &mut self,
        d: &mut Drawn,
        (singing, length): (&Singing, Option<f64>),
        at: &Where,
        (shine, fresh): (&Glow, Option<Shone>),
    ) {
        let Some(length) = length else {
            return;
        };
        let (x, y, w, h) = at.area;
        let now = at.now;
        let (bx0, bx1) = (x + w * 0.06, x + w * 0.88);
        let by = y + h * PROGRESS;
        let bh = (h * 0.0045).max(2.5);
        let bx = |t: f64| bx0 + (bx1 - bx0) * (t / length).clamp(0.0, 1.0) as f32;
        let bar = |c, half, color| super::pill(c, half, color, [0.0; 4], 0.0, 0.0, 0.0);
        d.out.push(bar(
            ((bx0 + bx1) / 2.0, by),
            ((bx1 - bx0) / 2.0, bh * 0.6),
            rgba(WHITE, 0.12),
        ));
        let mut stretches: Vec<(f64, f64)> = Vec::new();
        for n in singing.notes.lines.iter().flat_map(|l| l.notes.iter()) {
            match stretches.last_mut() {
                Some(s) if n.start - s.1 < BREAK => s.1 = s.1.max(n.end),
                _ => stretches.push((n.start, n.end)),
            }
        }
        stretches.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (a, b) in &stretches {
            let (xa, xb) = (bx(*a), bx(*b));
            d.out.push(bar(
                ((xa + xb) / 2.0, by),
                (((xb - xa) / 2.0).max(bh), bh),
                rgba(WHITE, 0.3),
            ));
        }
        // The part sung: gold up to now, each phrase that shone in its
        // tier's colour over it.
        let at_now = bx(now);
        d.out.extend(super::fill(
            ((bx0 + bx1) / 2.0, by),
            ((bx1 - bx0) / 2.0, bh),
            (bx0, at_now),
            rgba(super::GOLD_TOP, 0.95),
            rgba(super::GOLD_DEEP, 0.8),
            bh * 3.0,
        ));
        let decided: Vec<(Shone, f64)> = singing
            .moments()
            .iter()
            .filter_map(|m| match m.cue {
                Cue::Phrase(p) => Some((p, m.at)),
                _ => None,
            })
            .collect();
        for (l, line) in singing.notes.lines.iter().enumerate() {
            let (Some(first), Some(last)) = (line.notes.first(), line.notes.last()) else {
                continue;
            };
            let Some((p, when)) = decided.iter().find(|(p, _)| p.line == l) else {
                continue;
            };
            let (xa, xb) = (bx(first.start), bx(last.end));
            let c = tier_color(p.tier);
            d.out.push(bar(
                ((xa + xb) / 2.0, by),
                (((xb - xa) / 2.0).max(bh), bh * 1.15),
                rgba(c, 0.95),
            ));
            let r = bh * if p.tier == Some(Tier::Diva) { 2.4 } else { 1.8 };
            let m = r * 6.0;
            d.out.push(shape(
                kind::MARK,
                [xb - m, by - m, xb + m, by + m],
                [xb, by, r, 0.0],
                rgba(c, 1.0),
                [*when as f32, 0.0, 0.0],
            ));
        }
        // A little jackpot from a phrase's mark the moment it is Diva.
        if let Some(p) = fresh.filter(|p| p.tier == Some(Tier::Diva)) {
            d.light_sprays.push(fanfare::jackpot(
                (bx(p.end), by),
                bh * 3.0,
                at.clock,
                p.line as u32,
            ));
        }
        // The playhead's point on it.
        d.out.push(super::disc(
            (at_now, by),
            bh * 1.7,
            rgba([1.0, 0.98, 0.9], 1.0),
            rgba(GOLD, 0.6),
            bh * 5.0,
        ));
        // The time left, in its ring.
        let left = (length - now.max(0.0)).max(0.0);
        let secs = left.ceil() as u64;
        let label = format!("{}:{:02}", secs / 60, secs % 60);
        let px = (h * 0.024).round().clamp(10.0, 44.0);
        let radius = px * 1.75;
        let rc = (bx1 + w * 0.025 + radius, by);
        let thick = (px * 0.22).max(2.5);
        let mut ring = shape(
            kind::RING,
            [
                rc.0 - radius - thick * 3.0,
                rc.1 - radius - thick * 3.0,
                rc.0 + radius + thick * 3.0,
                rc.1 + radius + thick * 3.0,
            ],
            [rc.0, rc.1, radius, thick],
            [1.0; 4],
            [(left / length).clamp(0.0, 1.0) as f32, 0.0, 0.0],
        );
        ring.strain = [shine.level, shine.from, shine.since, 0.0];
        d.out.push(ring);
        let tw = self.text.measure(&label, px);
        let ty = rc.1 - self.cap_middle(px) + super::COVER_PAD as f32;
        self.text(
            &mut d.out,
            &label,
            px,
            1.0,
            (rc.0 - tw / 2.0, ty),
            f32::NEG_INFINITY,
            rgba(WHITE, 0.9),
            rgba(WHITE, 0.9),
        );
    }
}
