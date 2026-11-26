//! The shine on the stage ([`super::Stage::shine`]): the meter at the top
//! (filling as the phrase is sung, its words hit, then springing to the
//! shine the phrase reached, its tier's name popping under it, in what the
//! shouts are in, unless an effect the phrase fired shouts it itself:
//! [`crate::gui::fx::Frame::shouts_verdict`]), in the HUD's band over the
//! lane right of the multiplier's badge and left of the score's corner
//! (made smaller where they leave it less room: [`Stage::meter_spot`],
//! whose box the effects keep clear),
//! the tiers' effects round the lane and over the screen
//! (each row's glow round the lane and flames off its edges, the angels,
//! the jackpot from the meter as a phrase reaches the top: their slots,
//! [`crate::gui::fx::At`]), and the song's progress along the bottom: its
//! sung stretches, the part sung coloured phrase by phrase by the tier
//! each reached, a mark a phrase (its tier's size) popping as its tier is
//! decided (a little jackpot at the top), and the time left in a ring that
//! drains as the song goes, in the rainbow while the singing shines;
//! sinking away before the controls' bar fades in where it is (its seek
//! bar and times stand for it then), so the two never show at once. The
//! processor gives the GPU only where each is and when it changed
//! ([`crate::gui::glory`], [`crate::gui::fx`]).

use super::{rgba, Face, Lane, Stage, BREAK, GOLD, PROGRESS, WHITE};
use crate::core::shine::{Shone, Tier};
use crate::core::singing::{Cue, Singing};
use crate::gui::fx::{self, At, Frame, Text};
use crate::gui::glory::{meter_fill, position, MARKS};
use crate::gui::particles::Spray;
use crate::gui::rainbow::Glow;
use crate::gui::stage_gpu::{kind, Instance};
use crate::gui::streak::Step;

/// How long a tier's name shows under the meter, in seconds.
const VERDICT: f32 = 1.8;
/// How far the progress sinks as the controls come up, a share of the
/// height.
const SINK: f32 = 0.03;
/// A phrase with no tier's colour: a cool grey.
const NONE: [f32; 3] = [0.55, 0.6, 0.7];
/// How large the mark of a phrase with no tier is, in the bar's heights.
const NONE_MARK: f32 = 1.8;

/// A tier's colour (sRGB; [`crate::tiers::TierSpec::color`]); none's a
/// cool grey.
pub fn tier_color(tier: Option<Tier>) -> [f32; 3] {
    tier.map_or(NONE, |t| t.spec().color)
}

/// The meter between frames: where its fill springs from and to, and
/// since when (the singing's clock); the last phrase shown, and when;
/// whether an effect it fired shouts its verdict (none shown then); where
/// it stands in the area it was last placed in.
#[derive(Debug, Default)]
pub struct Meter {
    from: f32,
    to: f32,
    since: f32,
    shown: Option<(Shone, f32)>,
    shouted: bool,
    placed: Option<((f32, f32, f32, f32), Spot)>,
}

/// Where the meter stands on a stage: its middle, half its capsule's
/// size, the room round the capsule (its glow), its verdict's size, and
/// the box it all keeps to (left, top, right, bottom: the capsule and the
/// room round it, the widest tier's verdict on its pill at rest).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Spot {
    at: (f32, f32),
    half: (f32, f32),
    pad: f32,
    px: f32,
    /// The box.
    pub(super) rect: [f32; 4],
}

/// The room round the meter's capsule (its glow), in its half heights.
const PAD: f32 = 1.6;

/// The verdict a phrase at `tier` shows.
fn verdict(tier: Tier) -> String {
    format!("{}!", tier.name())
}

/// Half the verdict's pill at rest, its text `tw` wide at `px`.
fn verdict_half(tw: f32, px: f32) -> (f32, f32) {
    (tw / 2.0 + px * 0.45, px * 0.62)
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
        .filter(|m| match m.cue {
            Cue::Word { end, hit, .. } => hit && end >= start,
            Cue::Shot { to, .. } => to.0 == line,
            _ => false,
        })
        .count();
    Some(hit as f32 / words as f32)
}

/// What the shine needs of the stage: the area, the lane, the
/// pills' height, the song's time and the singing's clock, and how much
/// the progress has gone for the controls' bar (0 to 1,
/// [`crate::core::karaoke::controls::handover`]).
pub struct Where {
    pub area: (f32, f32, f32, f32),
    pub lane: Lane,
    pub now: f64,
    pub clock: f64,
    pub gone: f32,
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
    /// Where the meter stands on a stage in the window's `area` over
    /// `lane` ([`Spot`]): across its middle (or, where the multiplier's
    /// badge and the score's corner leave too little room there, in the
    /// middle of what they leave), in the middle of the HUD's band
    /// ([`Lane::hud`]); its capsule a third of the width long, its verdict
    /// 7 % of the height, each smaller where the band is lower than it or
    /// the room between the badge and the score's corner narrower. Worked
    /// out once an area.
    pub(super) fn meter_spot(&mut self, area: (f32, f32, f32, f32), lane: &Lane) -> Spot {
        if let Some((_, spot)) = self.meter.placed.filter(|(a, _)| *a == area) {
            return spot;
        }
        let (x, _, w, h) = area;
        let (top, bottom) = lane.hud(area);
        let tall = bottom - top;
        let badge = self.badge_box(lane, area);
        let score = crate::gui::karaoke::score_corner(area);
        let air = h * super::AIR;
        let hy = (h * 0.016).max(5.0).min(tall / (2.0 * (1.0 + PAD)));
        let pad = hy * PAD;
        let mut px = (h * 0.07).clamp(18.0, 140.0).min(tall / 1.24).floor();
        let widest = |stage: &mut Stage, px: f32| {
            (Tier::ALL.into_iter())
                .map(|t| stage.measure(Face::SHOUT, &verdict(t), px))
                .fold(0.0, f32::max)
        };
        let mut v = verdict_half(widest(self, px), px);
        // Across the middle where it has the room, else in the middle of
        // what the badge and the score's corner leave; half that room
        // across, short of them.
        let (from, to) = (badge[2] + air, score[0] - air);
        let want = (w * 0.17 + pad).max(v.0).min((to - from) / 2.0);
        let mx = if from + want <= to - want {
            (x + w * 0.5).clamp(from + want, to - want)
        } else {
            (from + to) / 2.0
        };
        let my = (top + bottom) / 2.0;
        let room = (mx - from).min(to - mx).max(1.0);
        let hx = (w * 0.17).min(room - pad).max(hy);
        while v.0 > room && px > 1.0 {
            px = (px * room / v.0).floor().min(px - 1.0).max(1.0);
            v = verdict_half(widest(self, px), px);
        }
        let (rx, ry) = ((hx + pad).max(v.0), (hy + pad).max(v.1));
        let spot = Spot {
            at: (mx, my),
            half: (hx, hy),
            pad,
            px,
            rect: [mx - rx, my - ry, mx + rx, my + ry],
        };
        self.meter.placed = Some((area, spot));
        spot
    }

    /// The shine for `singing` (the game on: `game`) at `at`, its streak's
    /// `step` this frame and the `effects` as strong as it holds them, into
    /// `d`; the song `length` seconds long.
    pub(super) fn shine(
        &mut self,
        d: &mut Drawn,
        (singing, game, length): (&Singing, bool, Option<f64>),
        at: &Where,
        (step, effects): (&Step, &Frame),
    ) {
        if game && singing.reference().is_some() {
            self.meter_and_verdict(d, singing, at, (step, effects));
        }
        // The effects' light round the lane (the rows' glow and flames),
        // then over the screen (the angels).
        let lane = Stage::lane_at(at.area, &at.lane, effects);
        self.effects_at(&lane, effects, (&mut d.lights, &mut Vec::new()));
        let sky = At::Sky { area: at.area };
        self.effects_at(&sky, effects, (&mut Vec::new(), &mut d.light_sprays));
        self.progress(d, (singing, length), at, (&step.glow, step.phrase), effects);
    }
    /// The meter, its fill, and the tier's name under it as a phrase is
    /// decided (the effects' bursts from it: a jackpot at the top tier).
    fn meter_and_verdict(
        &mut self,
        d: &mut Drawn,
        singing: &Singing,
        at: &Where,
        (step, effects): (&Step, &Frame),
    ) {
        let clock = at.clock as f32;
        let Spot {
            at: (mx, my),
            half,
            pad,
            px,
            ..
        } = self.meter_spot(at.area, &at.lane);
        if let Some(p) = step.phrase {
            self.meter.shown = Some((p, clock));
            self.meter.shouted = effects.shouts_verdict();
            // Within the tier reached (a shine past a bar with the pitch short
            // of the next's shows short of it).
            let next = p.tier.map_or(0, |t| t.index() + 1);
            let cap = MARKS.get(next).map_or(1.0, |m| m - 0.025);
            self.meter.aim(position(p.of_top()).min(cap), clock);
            let meter = At::Meter {
                at: (mx + half.0, my),
                size: half.1 * 2.2,
            };
            self.effects_at(&meter, effects, (&mut Vec::new(), &mut d.light_sprays));
        }
        let verdict_on = self.meter.shown.is_some_and(|(_, t)| clock - t < VERDICT);
        if !verdict_on {
            if let Some(live) = sung_so_far(singing, at.clock) {
                self.meter.aim(position(live), clock);
            }
        }
        // Over everything (the bursts falling through it, the light's bloom),
        // so it always reads.
        let m = &self.meter;
        d.top.push(shape(
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
        // The tier's name, popping and fading (unless shouted).
        let Some((p, t)) = self.meter.shown.filter(|_| !self.meter.shouted) else {
            return;
        };
        let age = clock - t;
        if age >= VERDICT {
            return;
        }
        let Some(tier) = p.tier else {
            return;
        };
        let label = verdict(tier);
        let tw = self.measure(Face::SHOUT, &label, px);
        // Over the meter, its middle on the meter's.
        let ty = my - self.cap_middle(Face::SHOUT, px) + super::COVER_PAD as f32;
        let c = tier_color(Some(tier));
        // On a badge of its own over the meter, ringed in the tier's colour,
        // over the light (its bloom would wash it out); both popping round
        // the meter's middle and fading as the shader times them
        // ([`crate::gui::glory::label_pop`]).
        let half = verdict_half(tw, px);
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
            Face::SHOUT,
            &label,
            (px, 1.0),
            (mx - tw / 2.0, ty),
            f32::NEG_INFINITY,
            (rgba(c, 1.0), rgba(c, 1.0)),
        );
        if let Some(text) = text {
            text.shape = [mx, my, t, VERDICT];
            if let Some(treat) = fx::treatment(Text::Shout, effects) {
                treat.apply(text);
            }
        }
    }

    /// The song's progress along the bottom: its sung stretches, the part
    /// sung coloured by the tiers its phrases reached, a mark a phrase, the
    /// time left in its ring; sunk and faded as far as it has gone for the
    /// controls (none once gone); the effects' bursts from a phrase's mark
    /// the moment it is decided (a little jackpot at the top).
    fn progress(
        &mut self,
        d: &mut Drawn,
        (singing, length): (&Singing, Option<f64>),
        at: &Where,
        (shine, fresh): (&Glow, Option<Shone>),
        effects: &Frame,
    ) {
        let Some(length) = length.filter(|_| at.gone < 1.0) else {
            return;
        };
        let (x, y, w, h) = at.area;
        let now = at.now;
        // The ring at the right, inside the window however narrow (the bar
        // ends short of it).
        let px = (h * 0.024).round().clamp(10.0, 44.0);
        let (radius, thick) = (px * 1.75, (px * 0.22).max(2.5));
        let ring_x = (x + w * 0.905 + radius).min(x + w - radius - thick * 3.0);
        let (bx0, bx1) = (
            x + w * 0.06,
            (x + w * 0.88).min(ring_x - radius - w * 0.025),
        );
        let by = y + h * (PROGRESS + SINK * at.gone);
        let first = d.out.len();
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
            let r = bh * p.tier.map_or(NONE_MARK, |t| t.spec().mark);
            let m = r * 6.0;
            d.out.push(shape(
                kind::MARK,
                [xb - m, by - m, xb + m, by + m],
                [xb, by, r, 0.0],
                rgba(c, 1.0),
                [*when as f32, 0.0, 0.0],
            ));
        }
        // The effects' bursts from a phrase's mark the moment it is
        // decided.
        if let Some(p) = fresh.filter(|_| at.gone < 0.5) {
            let mark = At::Mark {
                at: (bx(p.end), by),
                size: bh * 3.0,
                line: p.line as u32,
            };
            self.effects_at(&mark, effects, (&mut Vec::new(), &mut d.light_sprays));
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
        let rc = (ring_x, by);
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
        let tw = self.measure(Face::NUMBER, &label, px);
        let ty = rc.1 - self.cap_middle(Face::NUMBER, px) + super::COVER_PAD as f32;
        self.text(
            &mut d.out,
            Face::NUMBER,
            &label,
            (px, 1.0),
            (rc.0 - tw / 2.0, ty),
            f32::NEG_INFINITY,
            (rgba(WHITE, 0.9), rgba(WHITE, 0.9)),
        );
        let shows = 1.0 - at.gone;
        for i in &mut d.out[first..] {
            i.color[3] *= shows;
            i.color2[3] *= shows;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::karaoke::controls::handover;
    use crate::core::singing::Singing;
    use crate::gui::stage::Song;
    use crate::gui::stage_gpu::StageFrame;
    use crate::view::karaoke::KaraokePicture;

    /// A song of two words a second each from 1 s, its singing ready.
    fn singing() -> (Singing, std::sync::Arc<jodel_lyrics::Lyrics>) {
        let word = |text: &str, s: f64| jodel_lyrics::Word {
            start: Some(s),
            end: Some(s + 1.0),
            confidence: 0.9,
            ..jodel_lyrics::Word::new(text)
        };
        let line = jodel_lyrics::Line {
            words: vec![word("la", 1.0), word("lo", 2.0)],
            ..jodel_lyrics::Line::default()
        };
        let lyrics = std::sync::Arc::new(jodel_lyrics::Lyrics {
            lines: vec![line],
            source: jodel_lyrics::Source::Unknown,
            timing: jodel_lyrics::Timing::default(),
        });
        let mut melody = jodel_rhythm::waveform::Melody {
            cents: vec![0; 400],
            sure: vec![0; 400],
        };
        for i in 100..300 {
            melody.cents[i] = 6000;
            melody.sure[i] = 255;
        }
        let mut singing = Singing::default();
        let notes = crate::core::singing::melodic(&melody);
        singing.prepare(std::path::Path::new("a"), &lyrics, &notes);
        (singing, lyrics)
    }

    /// The stage at 1.2 s into the song in a window `w` × `h` (24 rows),
    /// the controls faded in as far as `fade`: its shapes and those over
    /// everything.
    fn frame(song: &Song, size: (f32, f32), fade: Option<f64>) -> Vec<Instance> {
        let f = stage_frame(song, size, fade);
        f.instances.iter().chain(&f.top).copied().collect()
    }

    /// The stage at 1.2 s into the song in a window `w` × `h` (24 rows),
    /// the controls faded in as far as `fade`.
    fn stage_frame(song: &Song, (w, h): (f32, f32), fade: Option<f64>) -> StageFrame {
        let picture = KaraokePicture {
            bar: 5.0 / 24.0,
            lines: Vec::new(),
            count: None,
            cue: None,
            controls: fade,
            start: None,
            before: Vec::new(),
            moved: 1.0,
            score: None,
            result: None,
        };
        Stage::default().frame(song, &picture, (0.0, 0.0, w, h), 1.2, 1.0 / 60.0)
    }

    /// With no tier, nothing of the effects library shows: none of its
    /// shapes, no screen pass, no text in a treatment, none of its bursts.
    #[test]
    fn nothing_of_the_library_shows_at_no_tier() {
        use crate::gui::fx::{Fx, Treat};
        let (singing, lyrics) = singing();
        let song = Song::new(&singing, lyrics.lines.iter().collect(), Some(60.0));
        let f = stage_frame(&song, (800.0, 450.0), None);
        let mut all = f.instances.iter().chain(&f.top).chain(&f.lights);
        assert!(all.all(|i| Fx::of_kind(i.params[0]).is_none() && Treat::of(i).is_none()));
        assert!(f.screen.is_empty() && f.light_sprays.is_empty());
        assert!(f.instances.iter().any(|i| i.params[0] == kind::TEXT));
    }

    /// The song's progress along the stage's bottom and the controls' bar
    /// never show at once: in the fade's first half the progress (its
    /// bar, marks and ring) sinks and fades, gone at its middle, when the
    /// bar has not begun to show; back as the bar goes. Nothing else on the
    /// stage moves for it (the lane and the lines stay where they are).
    #[test]
    fn the_progress_gives_way_to_the_controls_bar() {
        let (singing, lyrics) = singing();
        let song = Song::new(&singing, lyrics.lines.iter().collect(), Some(60.0));
        let (w, h) = (800.0, 450.0);
        let bar_top = h * 19.0 / 24.0;
        let in_bar = |i: &Instance| i.quad[3] > bar_top && i.color[3] > 0.0;
        let low = |i: &Instance| (i.quad[1] + i.quad[3]) / 2.0 > h * 0.85;
        let alone = frame(&song, (w, h), None);
        let hidden = frame(&song, (w, h), Some(0.0));
        assert!(alone.iter().any(in_bar), "the progress, low on the stage");
        assert_eq!(alone, hidden, "a bar that does not show changes nothing");
        // A quarter in: the progress half gone, the bar not yet there.
        let quarter = frame(&song, (w, h), Some(0.25));
        let alpha = |f: &[Instance]| f.iter().filter(|i| low(i)).map(|i| i.color[3]).sum::<f32>();
        assert!(alpha(&quarter) < alpha(&hidden) * 0.6 && alpha(&quarter) > 0.0);
        assert_eq!(handover(0.25).1, 0.0);
        // At the middle and after: the progress gone; only then the bar.
        for fade in [0.5, 0.75, 1.0] {
            let f = frame(&song, (w, h), Some(fade));
            assert!(!f.iter().any(in_bar), "{fade}: nothing where the bar is");
        }
        assert_eq!(handover(0.5), (1.0, 0.0));
        // The rest exactly as it was.
        let rest =
            |f: &[Instance]| -> Vec<Instance> { f.iter().filter(|i| !low(i)).copied().collect() };
        assert_eq!(rest(&frame(&song, (w, h), Some(1.0))), rest(&hidden));
    }

    /// A phrase whose row fires an effect that shouts its verdict itself
    /// (the word slam) shows no verdict under the meter: one at a time. A
    /// row that does not, its verdict.
    #[test]
    fn a_row_that_shouts_owns_the_verdict() {
        use crate::gui::fx::{Effects, Fx, Music, Use};
        static SLAM: [Use; 1] = [Use::of(Fx::Slam, 1.0)];
        let (singing, _) = singing();
        let slams = Effects {
            on_phrase: &SLAM,
            ..Effects::NONE
        };
        let tier = Tier::TOP;
        for (effects, shown) in [(&slams, false), (&Effects::NONE, true)] {
            let rows: Vec<(f32, &Effects)> = (Tier::ALL.into_iter())
                .map(|t| (1.0, if t == tier { effects } else { &Effects::NONE }))
                .collect();
            let phrase = Shone {
                line: 0,
                end: 1.0,
                value: 1.0,
                top: 0.9,
                tier: Some(tier),
            };
            let step = Step {
                phrase: Some(phrase),
                tier: Some(tier),
                ..Step::default()
            };
            let mut stage = Stage::default();
            let frame = (stage.board).frame_of(rows.iter().copied(), &step, &Music::NEUTRAL, 1.2);
            assert_eq!(frame.shouts_verdict(), !shown);
            let area = (0.0, 0.0, 1600.0, 900.0);
            let lane = Lane::new(area, 1.2);
            let at = Where {
                area,
                lane,
                now: 1.2,
                clock: 1.2,
                gone: 0.0,
            };
            let mut d = Drawn::default();
            stage.meter_and_verdict(&mut d, &singing, &at, (&step, &frame));
            let verdict = d.top.iter().any(|i| i.params[0] == kind::BADGE);
            assert_eq!(verdict, shown);
            assert!(d.top.iter().any(|i| i.params[0] == kind::METER));
        }
    }

    /// In a tall, narrow window the ring of the time left stays inside it,
    /// the bar ending short of the ring.
    #[test]
    fn the_ring_stays_inside_a_tall_window() {
        let (singing, lyrics) = singing();
        let song = Song::new(&singing, lyrics.lines.iter().collect(), Some(60.0));
        let (w, h) = (900.0, 1400.0);
        let f = frame(&song, (w, h), None);
        let ring = f
            .iter()
            .find(|i| i.params[0] == kind::RING)
            .expect("the ring");
        let [cx, _, radius, thick] = ring.shape;
        assert!(cx + radius + thick <= w, "{cx} {radius} {thick}");
        // The bar's track: what starts at its left, low down.
        let track = |i: &&Instance| i.quad[1] > h * 0.85 && i.quad[0] < w * 0.1;
        let bar_end = f
            .iter()
            .filter(track)
            .map(|i| i.quad[2])
            .fold(0.0, f32::max);
        assert!(bar_end < cx - radius, "{bar_end} {cx}");
    }
}
