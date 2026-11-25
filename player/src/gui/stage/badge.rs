//! The multiplier's badge over the playhead: its × (popping as it goes up
//! a rung, in the shouts' treatment, the rainbow while the singing
//! shines), and before it on a dark pill of their own (to read over the
//! flames) the rung's name in its tier's colour and the streak: "Amazing
//! 42 in a row ×16". Right of the playhead is the meter's, over the lane
//! the bulbs': the pill keeps left of the ×, made smaller where the
//! screen's edge is near, never past it. It is laid out by the widest it
//! can be (the top rung's ×, the longest name, [`WIDEST`]), so nothing
//! moves or changes size as the rung or the streak does.

use super::{disc, pill, rgba, Lane, Stage, GOLD, GOLD_TOP, WHITE};
use crate::core::singing::{multiplier, Singing};
use crate::gui::big_text::{Face, COVER_PAD};
use crate::gui::fx::{self, Text};
use crate::gui::stage_gpu::Instance;
use crate::tiers::Tier;

/// The widest streak the pill makes room for.
const WIDEST: &str = "888 in a row";
/// How long the badge takes to come in or go, about (seconds).
const EASE: f32 = 0.12;

/// The multiplier's badge as shown: the rung (the last reached, kept
/// while it fades), how much it shows, how long since it went up, and
/// the widths it is laid out by.
#[derive(Debug, Default)]
pub(super) struct Badge {
    rung: Option<Tier>,
    shown: f32,
    since: f32,
    widths: Option<Widths>,
}

/// The badge as a frame shows it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Showing {
    /// The rung.
    tier: Tier,
    /// How much it shows, 0 to 1.
    alpha: f32,
    /// How much bigger the × is than at rest (popping as it goes up).
    pop: f32,
}

impl Badge {
    /// A frame `dt` on, the multiplier at `rung` (none: ×1): it eases in
    /// while there is one (popping when the rung changes), out while
    /// there is none (as the last rung); none once gone.
    fn step(&mut self, rung: Option<Tier>, dt: f32) -> Option<Showing> {
        if let Some(t) = rung {
            if self.rung != Some(t) {
                self.since = 0.0;
            }
            self.rung = Some(t);
        }
        let target = if rung.is_some() { 1.0 } else { 0.0 };
        self.shown += (target - self.shown) * (1.0 - (-dt / EASE).exp());
        self.since += dt;
        let tier = self.rung?;
        (self.shown > 0.01).then(|| Showing {
            tier,
            alpha: self.shown,
            pop: 1.0 + 0.45 * (-self.since * 7.0).exp(),
        })
    }
}

/// What the badge is laid out by at a unit `p`, measured once a size:
/// the widest × (the top rung's) and the widest the pill's text can be
/// (the longest name, the gap, [`WIDEST`]), at their sizes in full.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Widths {
    p: f32,
    label: f32,
    pill: f32,
}

/// Where the pill's text goes: its sizes (the name's, the streak's) and
/// the right end of the pill's text.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Place {
    named: f32,
    small: f32,
    right: f32,
}

/// The full sizes of the name and the streak, at a unit `p`.
fn sizes(p: f32) -> (f32, f32) {
    (p * 0.5, p * 0.36)
}

/// The pill's text placed before the × at `head` (laid out by `w`), the
/// stage's left edge at `left`: in full where it fits, else smaller (to
/// 0.4 of it), in whole pixels.
fn place(head: f32, left: f32, w: &Widths) -> Place {
    let (named, small) = sizes(w.p);
    let right = head - w.label / 2.0 - w.p * 0.3;
    let room = right - left - small;
    let k = (room / w.pill).clamp(0.4, 1.0);
    Place {
        named: (named * k).round().max(8.0),
        small: (small * k).round().max(7.0),
        right,
    }
}

/// Where the pill's text `wide` across starts, padded `pad` either side:
/// just before `right`, never past the stage's `left` edge.
fn text_left(right: f32, wide: f32, pad: f32, left: f32) -> f32 {
    (right - pad - wide).max(left + 2.0 * pad)
}

impl Stage {
    /// The widths the badge is laid out by at `p` ([`Widths`]), measured
    /// once a size.
    fn badge_widths(&mut self, p: f32) -> Widths {
        if let Some(w) = self.badge.widths.filter(|w| w.p == p) {
            return w;
        }
        let (named, small) = sizes(p);
        let top = format!("×{}", multiplier(u32::MAX));
        let label = self.measure(Face::SHOUT, &top, (p * 1.05).round());
        let name = (Tier::ALL.into_iter())
            .map(|t| self.measure(Face::SHOUT, t.name(), named))
            .fold(0.0, f32::max);
        let pill = name + small * 0.6 + self.measure(Face::NUMBER, WIDEST, small);
        let w = Widths { p, label, pill };
        self.badge.widths = Some(w);
        w
    }

    /// The multiplier over the playhead of `lane` (the stage's left edge
    /// at `left`), `p` a unit of the lane's, `dt` the frame's step: its ×
    /// into `out`, the rung's name and the streak of `singing` (with the
    /// game on, `game`) into `top`, over the light.
    pub(super) fn draw_badge(
        &mut self,
        (out, top): (&mut Vec<Instance>, &mut Vec<Instance>),
        (singing, game): (&Singing, bool),
        (lane, left): (&Lane, f32),
        (p, dt): (f32, f32),
        effects: &fx::Frame,
    ) {
        let rung = if game { singing.rung() } else { None };
        let Some(Showing { tier, alpha, pop }) = self.badge.step(rung, dt) else {
            return;
        };
        let widths = self.badge_widths(p);
        let px = (p * 1.05).round();
        let label = format!("×{}", tier.multiplier());
        let tw = self.measure(Face::SHOUT, &label, px) * pop;
        let cy = lane.band.0 - p * 1.55;
        // Its glow, stronger up the rungs.
        let up = (tier.index() + 1) as f32 / Tier::COUNT as f32;
        out.push(disc(
            (lane.head, cy),
            0.0,
            [0.0; 4],
            rgba(GOLD, (0.35 + 0.5 * up) * alpha * pop),
            p * 1.4 * pop,
        ));
        let gold = rgba(GOLD_TOP, alpha);
        let at = (lane.head - tw / 2.0, cy - px * 0.62 * pop);
        let number = self.text(
            out,
            Face::SHOUT,
            &label,
            (px, pop),
            at,
            f32::NEG_INFINITY,
            (gold, gold),
        );
        if let (Some(n), Some(t)) = (number, fx::treatment(Text::Shout, effects)) {
            t.apply(n);
        }
        let Place {
            named,
            small,
            right,
        } = place(lane.head, left, &widths);
        let streak = (game && singing.tally.streak > 0)
            .then(|| format!("{} in a row", singing.tally.streak));
        let (gap, pad) = (small * 0.6, small * 0.45);
        let nw = self.measure(Face::SHOUT, tier.name(), named);
        let sw = (streak.as_ref()).map_or(0.0, |s| gap + self.measure(Face::NUMBER, s, small));
        let x = text_left(right, nw + sw, pad, left);
        top.push(pill(
            (x + (nw + sw) / 2.0, cy),
            ((nw + sw) / 2.0 + pad, named * 0.62),
            [0.04, 0.03, 0.09, 0.6 * alpha],
            [0.0; 4],
            0.0,
            0.0,
            0.0,
        ));
        let ink = rgba(tier.spec().color, alpha);
        let ny = cy - self.cap_middle(Face::SHOUT, named) + COVER_PAD as f32;
        let at = (x, ny);
        self.text(
            top,
            Face::SHOUT,
            tier.name(),
            (named, 1.0),
            at,
            f32::NEG_INFINITY,
            (ink, ink),
        );
        if let Some(streak) = streak {
            let white = rgba(WHITE, 0.8 * alpha);
            let ty = cy - self.cap_middle(Face::NUMBER, small) + COVER_PAD as f32;
            let at = (x + nw + gap, ty);
            self.text(
                top,
                Face::NUMBER,
                &streak,
                (small, 1.0),
                at,
                f32::NEG_INFINITY,
                (white, white),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// It comes in with a rung, popping; pops again a rung up; with none
    /// it fades as the rung it was, and is gone.
    #[test]
    fn the_badge_comes_pops_and_goes_with_the_rung() {
        let mut b = Badge::default();
        assert_eq!(b.step(None, 0.016), None);
        let (first, second) = (Tier::ALL[0], Tier::ALL[1]);
        let came = b.step(Some(first), 0.016).unwrap();
        assert_eq!(came.tier, first);
        assert!(came.alpha < 0.5 && came.pop > 1.4);
        for _ in 0..60 {
            b.step(Some(first), 0.016);
        }
        let held = b.step(Some(first), 0.016).unwrap();
        assert!(held.alpha > 0.99 && held.pop < 1.01);
        let up = b.step(Some(second), 0.016).unwrap();
        assert_eq!(up.tier, second);
        assert!(up.pop > 1.4, "popping a rung up");
        let fading = b.step(None, 0.05).unwrap();
        assert_eq!(fading.tier, second);
        assert!(fading.alpha < 1.0);
        let gone = (0..60).map(|_| b.step(None, 0.016)).last().unwrap();
        assert_eq!(gone, None);
    }

    /// Laid out in full where there is room, the pill's text ending before
    /// the widest ×; smaller where the edge is near; never past it.
    #[test]
    fn the_pill_fits_before_the_times_and_inside_the_stage() {
        let w = Widths {
            p: 50.0,
            label: 80.0,
            pill: 250.0,
        };
        let wide = place(450.0, 0.0, &w);
        assert_eq!((wide.named, wide.small), (25.0, 18.0));
        assert!(wide.right < 450.0 - 40.0);
        let narrow = place(250.0, 0.0, &w);
        assert!(narrow.named < wide.named && narrow.small < wide.small);
        assert_eq!(narrow.right, wide.right - 200.0);
        // As wide as it may be, it ends before the × and starts in the
        // stage.
        let x = text_left(narrow.right, 150.0, 4.0, 0.0);
        assert!(x >= 8.0 && x + 150.0 <= narrow.right);
        // Wider than the room left: kept inside the stage.
        assert_eq!(text_left(100.0, 300.0, 4.0, 10.0), 18.0);
    }
}
