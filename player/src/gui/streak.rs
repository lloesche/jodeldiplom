//! A streak followed frame by frame for its shine ([`super::rainbow`]): its
//! level (an intensity, 0 off, 1 the first, up to [`TOP`]: the
//! multiplier's for now, ×2 the first; a shine meter's tier later), when
//! it last changed and how strong it showed then ([`Shine`]), and the
//! words scored, each once, with the levels the shine went up since the
//! word before (a step up each: a burst from the playhead). The stage has
//! one; the window has another for its picture ([`PictureShine`]), whose
//! bar under the word sung gets a rainbow edge and a smaller burst a step
//! up. Only the level, its change and a record a step up go to the GPU.

use super::notes::Over;
use super::particles::{Kind, Spray};
use super::rainbow::{rim_of, Shine};
use super::stage_gpu::Instance;
use crate::core::singing::{multiplier, Hit, Scored, Singing};

/// The highest level.
pub const TOP: u32 = 3;

/// The level `singing`'s streak shines at (none with the game off): 0 at
/// ×1, 1 at ×2, up to [`TOP`] at ×4.
pub fn level(singing: &Singing, game: bool) -> u32 {
    if game {
        (multiplier(singing.tally.streak) - 1).min(TOP)
    } else {
        0
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

/// A frame of a streak: its shine, the word just scored fully hit (once
/// a word), and the levels the shine went up since the word before it was
/// seen (a step up each).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Step {
    /// Its shine.
    pub shine: Shine,
    /// The word just scored fully hit (the game on), if one was.
    pub word: Option<Scored>,
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
            .map(|w| super::pop::scored(at(w.end), px, clock, pps, w.end));
        let seed = |k: u32| (clock * 1000.0) as u32 + 7919 * k;
        let ups = (0..self.ups).map(|k| step_up(tip, px, clock, pps, seed(k)));
        word.into_iter().chain(ups).collect()
    }
}

/// A streak followed frame by frame.
#[derive(Debug, Default)]
pub struct Streak {
    shine: Shine,
    /// The last word scored, by its end.
    scored: Option<f64>,
    /// The shine's level when the last word scored was seen.
    seen: f32,
    /// The song's time at the frame before.
    before: Option<f64>,
}

impl Streak {
    /// The frame of `singing`'s streak (the game on, `game`) at `now`
    /// seconds into the song and `clock` (the singing's time). A new level
    /// eases in from how strong the shine showed (or, lost, collapses); a
    /// seek (the song's time going back) snaps it to its level; one back a
    /// second or more (or a song begun again) takes the words afresh.
    pub fn step(&mut self, singing: &Singing, game: bool, (now, clock): (f64, f64)) -> Step {
        let level = level(singing, game) as f32;
        let t = clock as f32;
        let back = self
            .before
            .replace(now)
            .map(|b| b - now)
            .filter(|&d| d > 0.0);
        if back.is_some() {
            self.shine = Shine {
                level,
                from: level,
                since: t - 10.0,
            };
        } else if level != self.shine.level {
            let from = self.shine.at(t).0;
            self.shine = Shine {
                level,
                from,
                since: t,
            };
        }
        if back.is_some_and(|d| d >= 1.0) || singing.heard.is_empty() {
            self.scored = None;
            self.seen = level;
        }
        let last = singing.tally.last_scored;
        let new = last.filter(|s| self.scored.is_none_or(|e| s.end > e));
        let mut ups = 0;
        if let Some(s) = new {
            self.scored = Some(s.end);
            ups = (level - self.seen).max(0.0) as u32;
            self.seen = level;
        }
        Step {
            shine: self.shine,
            word: new.filter(|s| game && s.hit == Some(Hit::Full)),
            ups,
        }
    }
}

/// The song as the picture's shine takes it: its singing, and whether the
/// game is on.
#[derive(Clone, Copy)]
pub struct Sung<'a> {
    /// The singing.
    pub singing: &'a Singing,
    /// Whether the game is on.
    pub game: bool,
}

/// A moment: seconds into the song, and the singing's time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Moment {
    /// Seconds into the song.
    pub now: f64,
    /// The singing's time.
    pub clock: f64,
}

/// The picture's streak: a rainbow edge on the bar under the word sung
/// while it shines, and a step up's burst from it, smaller than the
/// stage's.
#[derive(Debug, Default)]
pub struct PictureShine {
    streak: Streak,
    /// Its step ups' bursts showing.
    ups: Vec<Spray>,
}

impl PictureShine {
    /// What the GPU draws over the picture (`over`) for `song`'s streak at
    /// `at`: the bar's rainbow edge, while it shows, and the bursts.
    pub fn frame(
        &mut self,
        over: Option<&Over>,
        song: Sung,
        at: Moment,
    ) -> (Option<Instance>, &[Spray]) {
        let step = self
            .streak
            .step(song.singing, song.game, (at.now, at.clock));
        let rim = over.and_then(|o| o.rim);
        if let Some((c, half)) = rim {
            let seed = |k: u32| (at.clock * 1000.0) as u32 + 7919 * k;
            let ups = (0..step.ups).map(|k| step_up(c, half.1 * 6.0, at.clock, 0.0, seed(k)));
            self.ups.extend(ups);
        }
        self.ups.retain(|s| s.alive(at.clock));
        let edge = rim.and_then(|(c, half)| rim_of(c, half, &step.shine, at.clock));
        (edge, &self.ups)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::singing::Level;

    /// A song with its tally set: `streak` words in a row, the last scored
    /// ending at `end`.
    fn set(singing: &mut Singing, streak: u32, end: f64) {
        singing.tally.streak = streak;
        singing.tally.last_scored = Some(Scored {
            end,
            hit: Some(Hit::Full),
            points: 10,
            streak,
        });
    }

    /// A singing with a frame heard (a song under way).
    fn song() -> Singing {
        let mut s = Singing::default();
        s.level = Level::Easy;
        s.heard.push(crate::core::singing::Heard {
            t: 0.0,
            cents: 6000.0,
            off: None,
            hit: None,
            line: None,
        });
        s
    }

    /// The shine changes with the level, easing from how strong it shows;
    /// a seek back snaps it to its level.
    #[test]
    fn the_shine_follows_the_level() {
        let (mut s, mut singing) = (Streak::default(), song());
        set(&mut singing, 0, 0.5);
        assert_eq!(s.step(&singing, true, (1.0, 1.0)).shine.level, 0.0);
        set(&mut singing, 5, 1.5);
        let up = s.step(&singing, true, (2.0, 2.0)).shine;
        assert_eq!((up.level, up.from, up.since), (1.0, 0.0, 2.0));
        assert_eq!(
            s.step(&singing, true, (3.0, 3.0)).shine,
            up,
            "the same while it holds"
        );
        set(&mut singing, 0, 3.5);
        let lost = s.step(&singing, true, (4.0, 4.0)).shine;
        assert_eq!((lost.level, lost.from, lost.since), (0.0, 1.0, 4.0));
        let back = s.step(&singing, true, (3.9, 3.9)).shine;
        assert_eq!((back.level, back.from), (0.0, 0.0), "snapped");
        assert!(!back.shows(3.9));
    }

    /// Down a level (3 to 2), it still shines, at 2, once eased there; a
    /// new level mid-collapse comes in from how strong it showed, no jump.
    #[test]
    fn a_drop_eases_down_and_a_level_mid_collapse_comes_in_smoothly() {
        let (mut s, mut singing) = (Streak::default(), song());
        set(&mut singing, 20, 0.5);
        s.step(&singing, true, (1.0, 1.0));
        set(&mut singing, 10, 1.5);
        let drop = s.step(&singing, true, (3.0, 3.0)).shine;
        assert_eq!((drop.level, drop.from), (2.0, 3.0));
        assert_eq!(drop.at(3.0 + 1.0), (2.0, 0.0), "shining at 2");
        assert!(drop.shows(9.0));
        set(&mut singing, 0, 3.5);
        let lost = s.step(&singing, true, (4.0, 4.0)).shine;
        let mid = 4.0 + super::super::rainbow::COLLAPSE / 2.0;
        let before = lost.at(mid);
        set(&mut singing, 5, 4.1);
        let back = s
            .step(&singing, true, (f64::from(mid), f64::from(mid)))
            .shine;
        let after = back.at(mid);
        assert!((after.0 - before.0).abs() < 1e-5, "{before:?} {after:?}");
        assert_eq!(after.1, 0.0, "no longer collapsing");
        assert_eq!(back.at(mid + 1.0), (1.0, 0.0));
    }

    /// Several words scored in a frame: a step up each level the shine went
    /// up since the word before was seen; one that keeps the level, none.
    #[test]
    fn a_step_up_for_each_level_however_many_words_a_frame() {
        let (mut s, mut singing) = (Streak::default(), song());
        set(&mut singing, 3, 0.5);
        assert_eq!(s.step(&singing, true, (1.0, 1.0)).ups, 0);
        set(&mut singing, 6, 1.4);
        let one = s.step(&singing, true, (1.5, 1.5));
        assert_eq!((one.ups, one.word.map(|w| w.end)), (1, Some(1.4)));
        assert_eq!(s.step(&singing, true, (1.6, 1.6)).ups, 0, "once a word");
        set(&mut singing, 7, 1.7);
        assert_eq!(s.step(&singing, true, (1.8, 1.8)).ups, 0, "the level kept");
        set(&mut singing, 21, 2.5);
        assert_eq!(
            s.step(&singing, true, (2.6, 2.6)).ups,
            2,
            "×2 to ×4 at once"
        );
    }

    /// Sought back under a second, the words heard again score no step up
    /// again; begun again (sought back further), they do.
    #[test]
    fn a_step_up_once_across_a_seek_back_and_again_after_a_restart() {
        let (mut s, mut singing) = (Streak::default(), song());
        set(&mut singing, 4, 1.0);
        s.step(&singing, true, (1.0, 1.0));
        set(&mut singing, 5, 1.5);
        assert_eq!(s.step(&singing, true, (1.6, 1.6)).ups, 1);
        // Back to 1.2 s: the tally as it was, then the word again.
        set(&mut singing, 4, 1.0);
        assert_eq!(s.step(&singing, true, (1.2, 1.2)).ups, 0);
        set(&mut singing, 5, 1.5);
        assert_eq!(s.step(&singing, true, (1.6, 1.6)).ups, 0, "no ghost");
        // Begun again: from the start, the streak built up again.
        set(&mut singing, 0, 0.0);
        s.step(&singing, true, (0.1, 0.1));
        set(&mut singing, 4, 1.0);
        s.step(&singing, true, (1.0, 1.0));
        set(&mut singing, 5, 1.5);
        assert_eq!(s.step(&singing, true, (1.6, 1.6)).ups, 1, "again");
    }
}
