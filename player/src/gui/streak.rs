//! The singing's glow followed frame by frame ([`super::rainbow`]): the
//! tier the last phrase reached and its glow (0 off, up to the top's 1:
//! [`crate::tiers::TierSpec::glow`]), when it last changed and how strong
//! it showed then ([`Glow`]); read from the singing's moments
//! ([`crate::core::singing::Singing::moments_since`]): each word scored
//! hit (its sparks), each phrase decided, with the tiers the glow went up
//! with it (a step up each: a burst from the playhead,
//! [`super::fx::step_up`]), and the multiplier's rung reached or lost. The
//! effects ease and fire from it ([`super::fx::Board`]). The stage has
//! one: with its lane, for the lane's effects; without, for the bar under
//! the word sung of the lyrics set in the middle, which gets a rainbow edge
//! and a smaller burst a step up ([`super::stage::lyrics`]). Only the level,
//! its change and a record a step up go to the GPU.

use super::particles::Spray;
use super::rainbow::Glow;
use crate::core::shine::{Shone, Tier};
use crate::core::singing::{Cue, Since, Singing};

/// The tier `singing` glows at (none with the game off): the one its last
/// phrase reached.
pub fn tier(singing: &Singing, game: bool) -> Option<Tier> {
    singing.last_phrase().and_then(|p| p.tier).filter(|_| game)
}

/// The level `singing` glows at (none with the game off): the glow of the
/// tier its last phrase reached (the top's 1), 0 for none.
pub fn level(singing: &Singing, game: bool) -> f32 {
    glow_of(tier(singing, game))
}

/// `tier`'s glow, 0 for none.
fn glow_of(tier: Option<Tier>) -> f32 {
    tier.map_or(0.0, |t| t.spec().glow)
}

/// How many tiers up from `was` `now` is (none down or the same).
fn tiers_up(was: Option<Tier>, now: Option<Tier>) -> u32 {
    let rank = |t: Option<Tier>| t.map_or(0, |t| t.index() as u32 + 1);
    rank(now).saturating_sub(rank(was))
}

/// A frame of the glow: the tier it is at and the glow, the end of the
/// word just scored hit (the last, once), the phrase just decided (the
/// last, once), the tiers the glow went up with it (a step up each), the
/// multiplier's rung (none: ×1) and its change this frame (from, to), once.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Step {
    /// The tier the glow is at (or going to).
    pub tier: Option<Tier>,
    /// The glow.
    pub glow: Glow,
    /// When the word just scored hit ended, if one was.
    pub word: Option<f64>,
    /// The phrase just decided, if one was.
    pub phrase: Option<Shone>,
    /// The steps up with it.
    pub ups: u32,
    /// The multiplier's rung.
    pub at_rung: Option<Tier>,
    /// The rung reached or lost this frame: from, to.
    pub rung: Option<(Option<Tier>, Option<Tier>)>,
}

impl Step {
    /// The word's sparks, where one was just scored hit, from where it
    /// ended (`at` its end): the pills `px` high at `clock` (the singing's
    /// time), the lane going across at `pps`.
    pub fn word_spray(
        &self,
        at: impl Fn(f64) -> (f32, f32),
        (px, clock, pps): (f32, f64, f32),
    ) -> Option<Spray> {
        self.word
            .map(|end| super::pop::scored(at(end), px, clock, pps, end))
    }
}

/// The glow followed frame by frame: the tier it is at, the glow, and the
/// multiplier's rung.
#[derive(Debug, Default)]
pub struct Streak {
    tier: Option<Tier>,
    glow: Glow,
    rung: Option<Tier>,
}

impl Streak {
    /// The frame of `singing`'s glow (the game on, `game`) at `clock` (the
    /// singing's time), its moments new since the frame before `since`. A
    /// new tier's glow eases in from how strong the glow showed (or, lost,
    /// collapses); the moments taken back (a seek, the song begun again)
    /// snap it to its tier's, and what came with them throws nothing. The
    /// rung is the multiplier's (none with the game off), changed with
    /// the streak's moments (a rung reached or lost) or without (a seek).
    pub fn step(&mut self, singing: &Singing, game: bool, since: &Since, clock: f64) -> Step {
        let tier = tier(singing, game);
        let level = glow_of(tier);
        let t = clock as f32;
        let was = self.tier;
        self.tier = tier;
        let at_rung = if game { singing.rung() } else { None };
        let from = std::mem::replace(&mut self.rung, at_rung);
        if since.sought {
            self.glow = Glow {
                level,
                from: level,
                since: t - 10.0,
            };
            return Step {
                tier,
                glow: self.glow,
                at_rung,
                ..Step::default()
            };
        }
        if tier != was {
            let from = self.glow.at(t).0;
            self.glow = Glow {
                level,
                from,
                since: t,
            };
        }
        if !game {
            return Step {
                tier,
                glow: self.glow,
                at_rung,
                ..Step::default()
            };
        }
        let new = since.moments.iter().rev();
        let phrase = new.clone().find_map(|m| match m.cue {
            Cue::Phrase(p) => Some(p),
            _ => None,
        });
        let word = new.clone().find_map(|m| match m.cue {
            Cue::Word { end, hit: true, .. } => Some(end),
            _ => None,
        });
        let climbed = new
            .clone()
            .any(|m| matches!(m.cue, Cue::StreakUp(_) | Cue::StreakLost(_)));
        Step {
            tier,
            glow: self.glow,
            word,
            phrase,
            ups: if phrase.is_some() {
                tiers_up(was, tier)
            } else {
                0
            },
            at_rung,
            rung: (climbed && from != at_rung).then_some((from, at_rung)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::singing::Moment;

    fn since(moments: &[Moment], sought: bool) -> Since<'_> {
        Since { moments, sought }
    }

    fn phrase(end: f64, tier: Option<crate::core::shine::Tier>) -> Moment {
        Moment {
            at: end,
            decided: end,
            cue: Cue::Phrase(Shone {
                line: 0,
                end,
                value: 0.8,
                top: 0.9,
                tier,
            }),
        }
    }

    /// With no phrase decided, no glow; a seek snaps it; the game off,
    /// none.
    #[test]
    fn the_glow_eases_in_and_a_seek_snaps_it() {
        let singing = Singing::default();
        let mut s = Streak::default();
        assert_eq!(
            s.step(&singing, true, &since(&[], false), 1.0).glow.level,
            0.0
        );
        assert_eq!(level(&singing, false), 0.0);
        let snapped = s.step(&singing, true, &since(&[], true), 2.0).glow;
        assert_eq!((snapped.level, snapped.from), (0.0, 0.0));
        assert!(!snapped.shows(2.0));
    }

    /// A phrase decided: as many steps up as the glow went up, once; a
    /// seek with it, none.
    #[test]
    fn a_step_up_for_each_level_with_the_phrase_only() {
        let singing = Singing::default();
        let mut s = Streak::default();
        let m = [phrase(1.0, Some(Tier::named("Great")))];
        // Its level as the singing has it (none here): no step up.
        let step = s.step(&singing, true, &since(&m, false), 1.1);
        assert_eq!(step.phrase.map(|p| p.end), Some(1.0));
        assert_eq!(step.ups, 0);
        let sought = s.step(&singing, true, &since(&m, true), 1.2);
        assert_eq!((sought.phrase, sought.ups), (None, 0));
    }

    /// A step up for each tier the glow goes up, none going down or
    /// staying.
    #[test]
    fn a_step_up_for_each_tier_passed() {
        let t = |i: usize| Some(Tier::ALL[i]);
        assert_eq!(tiers_up(None, t(0)), 1);
        assert_eq!(tiers_up(None, t(2)), 3);
        assert_eq!(tiers_up(t(0), t(2)), 2);
        assert_eq!(tiers_up(t(1), t(1)), 0);
        assert_eq!(tiers_up(t(2), t(0)), 0);
        assert_eq!(tiers_up(t(0), None), 0);
    }

    /// A rung reached with its moment: the change, once; held after; a
    /// seek to another rung snaps there with no change; the game off, none.
    #[test]
    fn a_rung_reached_comes_with_its_moment_and_a_seek_snaps_it() {
        let mut singing = Singing::default();
        let mut s = Streak::default();
        let first = Tier::ALL[0];
        singing.tally.streak = first.spec().streak;
        let up = [Moment {
            at: 1.0,
            decided: 1.0,
            cue: Cue::StreakUp(first),
        }];
        let step = s.step(&singing, true, &since(&up, false), 1.0);
        let reached = Some((None, Some(first)));
        assert_eq!((step.at_rung, step.rung), (Some(first), reached));
        let held = s.step(&singing, true, &since(&[], false), 1.1);
        assert_eq!((held.at_rung, held.rung), (Some(first), None));
        singing.tally.streak = 0;
        let sought = s.step(&singing, true, &since(&up, true), 1.2);
        assert_eq!((sought.at_rung, sought.rung), (None, None));
        singing.tally.streak = Tier::TOP.spec().streak;
        let off = s.step(&singing, false, &since(&up, false), 1.3);
        assert_eq!(off.at_rung, None);
    }
}
