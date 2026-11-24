//! The singing's glow followed frame by frame ([`super::rainbow`]): its
//! level (0 off, then the tier the last phrase reached, 1 Good up to
//! [`TOP`], Diva; [`crate::core::shine::Tier`]), when it last changed and
//! how strong it showed then ([`Glow`]); read from the singing's moments
//! ([`crate::core::singing::Singing::moments_since`]): each word scored
//! hit (its sparks) and each phrase decided, with the levels the glow went
//! up with it (a step up each: a burst from the playhead). The stage has
//! one; the window has another for its picture ([`PictureShine`]), whose
//! bar under the word sung gets a rainbow edge and a smaller burst a step
//! up. Only the level, its change and a record a step up go to the GPU.

use super::notes::Over;
use super::particles::{Kind, Spray};
use super::rainbow::{rim_of, Glow};
use super::stage_gpu::Instance;
use crate::core::shine::Shone;
use crate::core::singing::{Cue, Cursor, Since, Singing};

/// The highest level: Diva.
pub const TOP: u32 = 4;

/// The level `singing` glows at (none with the game off): the tier its
/// last phrase reached (1 Good to [`TOP`] Diva), 0 for none.
pub fn level(singing: &Singing, game: bool) -> u32 {
    let tier = singing.last_phrase().and_then(|p| p.tier);
    match tier {
        Some(t) if game => t as u32 + 1,
        _ => 0,
    }
}

/// A step up's burst from `at` (the playhead), `size` its pieces' scale,
/// at `clock` (the singing's time), going by at `pps` (a lane's), its
/// pieces scattered by `seed`.
pub fn step_up(at: (f32, f32), size: f32, clock: f64, pps: f32, seed: u32) -> Spray {
    Spray {
        at,
        half: (0.0, 0.0),
        start: clock as f32,
        drift: -pps,
        size,
        seed,
        rgb: [1.0; 3],
        kind: Kind::StepUp,
    }
}

/// A frame of the glow: the glow, the end of the word just scored hit (the
/// last, once), the phrase just decided (the last, once), and the levels
/// the glow went up with it (a step up each).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Step {
    /// The glow.
    pub glow: Glow,
    /// When the word just scored hit ended, if one was.
    pub word: Option<f64>,
    /// The phrase just decided, if one was.
    pub phrase: Option<Shone>,
    /// The steps up with it.
    pub ups: u32,
}

impl Step {
    /// Its bursts on the stage: the word's sparks from where it hit
    /// (`at` its end), a step up's from `tip` (the playhead) each step; the
    /// pills `px` high at `clock` (the singing's time), the lane going
    /// across at `pps`.
    pub fn sprays(
        &self,
        at: impl Fn(f64) -> (f32, f32),
        tip: (f32, f32),
        (px, clock, pps): (f32, f64, f32),
    ) -> Vec<Spray> {
        let word = self
            .word
            .map(|end| super::pop::scored(at(end), px, clock, pps, end));
        let seed = |k: u32| (clock * 1000.0) as u32 + 7919 * k;
        let ups = (0..self.ups).map(|k| step_up(tip, px, clock, pps, seed(k)));
        word.into_iter().chain(ups).collect()
    }
}

/// The glow followed frame by frame.
#[derive(Debug, Default)]
pub struct Streak {
    glow: Glow,
}

impl Streak {
    /// The frame of `singing`'s glow (the game on, `game`) at `clock` (the
    /// singing's time), its moments new since the frame before `since`. A
    /// new level eases in from how strong the glow showed (or, lost,
    /// collapses); the moments taken back (a seek, the song begun again)
    /// snap it to its level, and what came with them throws nothing.
    pub fn step(&mut self, singing: &Singing, game: bool, since: &Since, clock: f64) -> Step {
        let level = level(singing, game);
        let t = clock as f32;
        let was = self.glow.level;
        if since.sought {
            self.glow = Glow {
                level: level as f32,
                from: level as f32,
                since: t - 10.0,
            };
            return Step {
                glow: self.glow,
                ..Step::default()
            };
        }
        if level as f32 != self.glow.level {
            let from = self.glow.at(t).0;
            self.glow = Glow {
                level: level as f32,
                from,
                since: t,
            };
        }
        if !game {
            return Step {
                glow: self.glow,
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
        Step {
            glow: self.glow,
            word,
            phrase,
            ups: if phrase.is_some() {
                level.saturating_sub(was as u32)
            } else {
                0
            },
        }
    }
}

/// The song as the picture's glow takes it: its singing, and whether the
/// game is on.
#[derive(Clone, Copy)]
pub struct Sung<'a> {
    /// The singing.
    pub singing: &'a Singing,
    /// Whether the game is on.
    pub game: bool,
}

/// The picture's glow: a rainbow edge on the bar under the word sung while
/// it shows, and a step up's burst from it, smaller than the stage's.
#[derive(Debug, Default)]
pub struct PictureShine {
    streak: Streak,
    cursor: Cursor,
    /// Its step ups' bursts showing.
    ups: Vec<Spray>,
}

impl PictureShine {
    /// What the GPU draws over the picture (`over`) for `song`'s glow at
    /// `clock` (the singing's time): the bar's rainbow edge, while it
    /// shows, and the bursts.
    pub fn frame(
        &mut self,
        over: Option<&Over>,
        song: Sung,
        clock: f64,
    ) -> (Option<Instance>, &[Spray]) {
        let since = song.singing.moments_since(&mut self.cursor);
        let step = self.streak.step(song.singing, song.game, &since, clock);
        let rim = over.and_then(|o| o.rim);
        if let Some((c, half)) = rim {
            let seed = |k: u32| (clock * 1000.0) as u32 + 7919 * k;
            let ups = (0..step.ups).map(|k| step_up(c, half.1 * 6.0, clock, 0.0, seed(k)));
            self.ups.extend(ups);
        }
        self.ups.retain(|s| s.alive(clock));
        let edge = rim.and_then(|(c, half)| rim_of(c, half, &step.glow, clock));
        (edge, &self.ups)
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
                diva: 0.9,
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
        assert_eq!(level(&singing, false), 0);
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
        let m = [phrase(1.0, Some(crate::core::shine::Tier::Great))];
        // Its level as the singing has it (none here): no step up.
        let step = s.step(&singing, true, &since(&m, false), 1.1);
        assert_eq!(step.phrase.map(|p| p.end), Some(1.0));
        assert_eq!(step.ups, 0);
        let sought = s.step(&singing, true, &since(&m, true), 1.2);
        assert_eq!((sought.phrase, sought.ups), (None, 0));
    }
}
