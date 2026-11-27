//! A story's scene moving with the beat, from the music's clock (the shared
//! grid, [`Music`]): a small push inwards of the camera on each beat, like
//! a speaker's cone, and a stronger kick on each bar's first with a soft
//! light on the layers' edges ([`Beat`], shaped here from where the music
//! is in the beat and the bar). The singer's
//! tier makes it all stronger (its row's [`crate::tiers::TierSpec::backdrop_beat`],
//! held through the rows), and a phrase landing flares it for a moment
//! ([`Pulse::of`]). The beat fades only where the grid is really unsure
//! (as [`Music::sure`] has it, as every effect's): a little unsure, it
//! still beats.
//!
//! The strengths are look parameters, the constants here.

use crate::gui::fx::Music;
use jodel_looks::math::smoothstep;

/// How far the camera pushes in on each beat, a share of the picture.
pub const BEAT: f32 = 0.009;
/// How much further on a bar's first beat.
pub const DOWNBEAT: f32 = 0.006;
/// How much more the top tier's row moves it all (the row's `backdrop_beat`
/// of 1: twice as far, the edges twice as bright).
pub const TIER: f32 = 1.0;
/// How bright the edges light on a bar's first beat.
pub const RIM: f32 = 0.9;
/// The share of the edges' light without a tier (the row's adds the rest).
pub const RIM_BASE: f32 = 0.35;
/// A phrase landing: how far the camera pushes in more, and how much the
/// edges light more, at the flare's height.
pub const FLARE: (f32, f32) = (0.012, 0.8);
/// How long a push takes to rise, in seconds: quick, but not a jump.
pub const ATTACK: f32 = 0.02;
/// How fast a beat's push falls back, then a bar's: to about a third in
/// this many seconds.
pub const FALL: (f32, f32) = (0.11, 0.28);

/// The beat's pushes now, each 0 to 1: a beat's and a bar's first's, as
/// sure as the grid lets them be.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Beat {
    /// The push of the beat just passed.
    pub beat: f32,
    /// The push of the bar's first beat just passed.
    pub downbeat: f32,
}

impl Beat {
    /// The pushes in `music`: each a cone's push ([`cone`]) from the beat
    /// (and the bar's first) just passed, faded where the grid is unsure
    /// ([`Music::sure`]).
    pub fn of(music: &Music) -> Beat {
        let sure = music.sure;
        let len = music.beat_length();
        let since_beat = music.beat_phase * len;
        let since_bar = music.bar_phase * music.bar_beats.max(1.0) * len;
        Beat {
            beat: sure * cone(since_beat, FALL.0),
            downbeat: sure * cone(since_bar, FALL.1),
        }
    }

    /// Whether either push still shows.
    pub fn moving(self) -> bool {
        self.beat > 0.003 || self.downbeat > 0.003
    }
}

/// A push `since` seconds after it: rising smoothly over [`ATTACK`], then
/// falling back by `fall` (1 at its height).
pub fn cone(since: f32, fall: f32) -> f32 {
    if since < ATTACK {
        smoothstep(0.0, ATTACK, since.max(0.0))
    } else {
        (-(since - ATTACK) / fall.max(1e-4)).exp()
    }
}

/// What the beat does to a scene in a frame: how far the camera pushes in
/// (a share of the picture) and how bright the layers' edges light.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pulse {
    /// How far the camera pushes in.
    pub grow: f32,
    /// How bright the edges light.
    pub rim: f32,
}

impl Pulse {
    /// The pulse of `beat`, the singer's tier's row's `tier` (held: 0
    /// none, 1 the top's) and a phrase landing's `flare` (0 to 1).
    pub fn of(beat: Beat, tier: f32, flare: f32) -> Pulse {
        let k = 1.0 + TIER * tier;
        let flare = flare * (0.35 + 0.65 * tier);
        Pulse {
            grow: (BEAT * beat.beat + DOWNBEAT * beat.downbeat) * k + FLARE.0 * flare,
            rim: RIM * beat.downbeat * (RIM_BASE + (1.0 - RIM_BASE) * tier) + FLARE.1 * flare,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiers::{held, Tier};

    /// The beat of a steady grid at 120 a minute, four to the bar, `sure`
    /// as the analysis is, `beats` in.
    fn at(beats: f64, confidence: f64) -> Beat {
        Beat::of(&Music::steady(beats, 120.0, 4, confidence, 0.0))
    }

    #[test]
    fn a_push_rises_quickly_and_falls_back() {
        assert_eq!(cone(0.0, 0.1), 0.0);
        assert!((cone(ATTACK, 0.1) - 1.0).abs() < 1e-6);
        assert!(cone(ATTACK / 2.0, 0.1) > 0.3);
        let later = cone(ATTACK + 0.1, 0.1);
        assert!((later - (-1.0f32).exp()).abs() < 1e-5, "{later}");
        assert!(cone(0.45, 0.1) < 0.02);
    }

    /// On the grid: just after each beat a push, none just before the next;
    /// a bar's first pushes more and longer.
    #[test]
    fn the_beat_pushes_on_the_grid_and_the_bar_s_first_more() {
        // 30 ms after the bar's first (the 9th beat, 8 in: a bar's first).
        let on = at(8.06, 0.9);
        assert!(on.beat > 0.8 && on.downbeat > 0.9, "{on:?}");
        // Just before the next beat: next to nothing.
        let off = at(8.95, 0.9);
        assert!(off.beat < 0.03, "{off:?}");
        // A beat that is not a bar's first: its own push, the bar's long gone.
        let second = at(9.06, 0.9);
        assert!(second.beat > 0.8 && second.downbeat < 0.2, "{second:?}");
        let g = |b: Beat| BEAT * b.beat + DOWNBEAT * b.downbeat;
        let (on, second) = (Pulse::of(on, 0.0, 0.0), Pulse::of(second, 0.0, 0.0));
        assert!(on.grow > second.grow * 1.4 && on.rim > 3.0 * second.rim);
        assert!((on.grow - g(at(8.06, 0.9))).abs() < 1e-6);
        // About 1 to 3 % at most.
        let top = Pulse::of(at(8.04, 0.9), 1.0, 0.0);
        assert!(top.grow > 0.025 && top.grow < 0.04, "{top:?}");
    }

    /// The beat fades only where the grid is really unsure: a little
    /// unsure, it beats as much; with no grid at all, not at all.
    #[test]
    fn the_beat_fades_only_where_the_grid_is_really_unsure() {
        let sure = at(8.03, 0.9).beat;
        let middling = at(8.03, 0.5).beat;
        assert!(middling > 0.95 * sure, "{middling} {sure}");
        assert!(at(8.03, 0.1).beat < 0.01);
        assert_eq!(Beat::of(&Music::NEUTRAL), Beat::default());
        assert!(!Beat::default().moving() && at(8.03, 0.9).moving());
    }

    /// Through the rows: the lowest's pulse a little more than none, the
    /// top's the strongest, each row's at least the one's below; a phrase
    /// landing flares it, more at the top.
    #[test]
    fn the_pulse_grows_with_the_tiers() {
        let beat = at(8.03, 0.9);
        let of = |t: Tier| Pulse::of(beat, held(t.spec().glow, |r| r.backdrop_beat), 0.0);
        let none = Pulse::of(beat, 0.0, 0.0);
        let rows: Vec<Pulse> = Tier::ALL.into_iter().map(of).collect();
        assert!(rows[0].grow > none.grow && rows[0].grow < 1.3 * none.grow);
        for w in rows.windows(2) {
            assert!(w[1].grow > w[0].grow && w[1].rim > w[0].rim, "{w:?}");
        }
        let top = rows[Tier::TOP.index()];
        assert!(
            top.grow > 1.9 * none.grow && top.rim > 2.5 * none.rim,
            "{top:?}"
        );
        let quiet = Beat::default();
        let flared = |tier: f32| Pulse::of(quiet, tier, 1.0);
        assert!(flared(0.0).grow > 0.0 && flared(1.0).grow > 2.0 * flared(0.0).grow);
        assert!(flared(1.0).rim > 0.5 && Pulse::of(quiet, 1.0, 0.0).rim == 0.0);
    }
}
