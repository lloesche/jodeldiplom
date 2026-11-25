//! The game's moments as the sounds take them: read from the singing's one
//! stream of them ([`Singing::moments_since`], through a [`Cursor`]), each
//! made the sound it goes with ([`Cued`]) at the singing's time it happens
//! (or at once, for one happening now: a word scored, the streak, a
//! phrase's tier). On top of the stream: bubbles bursting together pop one
//! after another, a ripple up the line ([`RIPPLE`], one splash for them
//! all), and one seen long after it happened ([`STALE`]: the first frames
//! after a stall) stays silent. The stream going back (a seek, the song
//! sung again) is told ([`Frame::sought`]): what was cued and not heard
//! yet is of another place in the song.

use std::path::PathBuf;

use super::sounds::{Sound, STEPS};
use crate::core::singing::{Cue, Cursor, Moment, Singing};

/// A sound cued: which, its gain and pan (-1 left to 1 right), and when
/// (the singing's time; `None`: at once).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cued {
    /// The sound.
    pub sound: Sound,
    /// Its gain.
    pub gain: f32,
    /// Its pan.
    pub pan: f32,
    /// When.
    pub at: Option<f64>,
}

/// What a frame of the game brought: its sounds, and whether what was
/// cued before is off.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Frame {
    /// The new sounds.
    pub cues: Vec<Cued>,
    /// The cues not heard yet are of another place in the song.
    pub sought: bool,
}

/// How long after its time a moment still sounds (seen late: the first
/// frame after a pause or a stall); older ones pass silently.
pub const STALE: f64 = 0.25;
/// How far apart bubbles bursting together pop: a ripple up the line.
pub const RIPPLE: f64 = 0.035;
/// The most bubbles bursting together that pop.
pub const RIPPLE_MOST: usize = 6;
/// A moment this little ahead of the singing's time sounds at once (the
/// desk is ahead of what is heard by more than that).
const NOW: f64 = 0.02;

/// The game's moments read frame by frame.
#[derive(Debug, Default)]
pub struct Moments {
    /// The song read, and where in its stream.
    song: Option<PathBuf>,
    cursor: Cursor,
}

/// The scale step of a note `height` high (-1 to 1).
pub fn step(height: f32) -> usize {
    let h = f64::from(height.clamp(-1.0, 1.0));
    ((h + 1.0) / 2.0 * (STEPS - 1) as f64).round() as usize
}

/// The pan of a moment `across` its line (0 its start, 1 its end).
fn pan(across: f32) -> f32 {
    (across.clamp(0.0, 1.0) - 0.5) * 0.8
}

/// The sound `cue` goes with, and its pan (none for a word scored: its
/// bubbles and the streak have theirs).
pub fn sound_of(cue: &Cue) -> Option<(Sound, f32)> {
    Some(match *cue {
        Cue::Pop(b) => (Sound::Pop(step(b.height)), pan(b.across)),
        Cue::Splash(b) => (Sound::Sparkle(step(b.height)), pan(b.across)),
        Cue::Shot { across, .. } => (Sound::Zap, pan(across)),
        Cue::StreakUp(level) => (Sound::StepUp(level), 0.0),
        Cue::StreakLost => (Sound::Break, 0.0),
        Cue::Phrase(shone) => (Sound::Tier(shone.tier?), 0.0),
        Cue::Word { .. } => return None,
    })
}

impl Moments {
    /// Forgets the song (the game off, karaoke left): the next frame reads
    /// it from then on.
    pub fn reset(&mut self) {
        *self = Moments::default();
    }

    /// The frame of `singing`'s game at `clock` (the singing's time).
    pub fn frame(&mut self, singing: &Singing, clock: f64) -> Frame {
        let song = singing.song().map(PathBuf::from);
        if song != self.song {
            *self = Moments {
                song,
                ..Moments::default()
            };
        }
        let since = singing.moments_since(&mut self.cursor);
        Frame {
            cues: cues(since.moments, clock),
            sought: since.sought,
        }
    }
}

/// The sounds of `moments` (read together) at `clock`.
fn cues(moments: &[Moment], clock: f64) -> Vec<Cued> {
    let mut out = Vec::new();
    for m in moments {
        let Some((sound, pan)) = sound_of(&m.cue) else {
            continue;
        };
        let mut at = m.at;
        if let Cue::Pop(b) | Cue::Splash(b) = m.cue {
            // Of the bubbles bursting together, the first few pop, one
            // after another; one splash, on the last of them.
            let (j, n) = b.burst;
            let splash = matches!(m.cue, Cue::Splash(_));
            if j >= RIPPLE_MOST || (splash && j + 1 != n.min(RIPPLE_MOST)) {
                continue;
            }
            at += RIPPLE * j as f64;
        }
        if at < clock - STALE {
            continue;
        }
        out.push(Cued {
            sound,
            gain: 1.0,
            pan,
            at: (at > clock + NOW).then_some(at),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Word `word`'s bubble bursting at `at` as `burst` (its place among
    /// those bursting together, and their count).
    fn pop(at: f64, word: usize, height: f32, burst: (usize, usize)) -> [Moment; 2] {
        let b = crate::core::singing::Burst {
            line: 0,
            word,
            height,
            across: 0.25,
            burst,
        };
        [
            Moment {
                at,
                decided: at - 0.2,
                cue: Cue::Pop(b),
            },
            Moment {
                at: at + crate::core::bubbles::SWELL,
                decided: at - 0.2,
                cue: Cue::Splash(b),
            },
        ]
    }

    /// A shot fired zaps as it is fired; the word it rescues pops as it
    /// lands, as any.
    #[test]
    fn a_shot_zaps_and_its_word_pops_as_it_lands() {
        let shot = Moment {
            at: 2.0,
            decided: 1.9,
            cue: Cue::Shot {
                from: (0, 3),
                to: (0, 2),
                across: 0.9,
            },
        };
        let lands = 2.0 + crate::core::bubbles::FLIGHT + crate::core::bubbles::SWEEP;
        let mut m = vec![shot];
        m.extend(pop(lands, 2, 0.0, (0, 1)));
        let c = cues(&m, 1.9);
        assert_eq!((c[0].sound, c[0].at), (Sound::Zap, Some(2.0)));
        assert!(c[0].pan > 0.15, "from the right of the line");
        assert_eq!((c[1].sound, c[1].at), (Sound::Pop(4), Some(lands)));
    }

    #[test]
    fn a_pop_sounds_on_its_time_and_splashes_as_it_tears() {
        let m = pop(1.2, 0, -1.0, (0, 1));
        let c = cues(&m, 1.0);
        assert_eq!(c.len(), 2);
        assert_eq!((c[0].sound, c[0].at), (Sound::Pop(0), Some(1.2)));
        assert!(c[0].pan < -0.15, "on the left of the line");
        let swell = crate::core::bubbles::SWELL;
        assert_eq!(
            (c[1].sound, c[1].at),
            (Sound::Sparkle(0), Some(1.2 + swell))
        );
        // Due now: at once; long gone: silent.
        assert_eq!(cues(&m, 1.21)[0].at, None);
        assert!(cues(&m, 1.2 + swell + STALE + 0.01).is_empty());
    }

    #[test]
    fn bubbles_bursting_together_ripple_up_with_one_splash() {
        let m: Vec<Moment> = (0..8).flat_map(|k| pop(2.0, k, 1.0, (k, 8))).collect();
        let c = cues(&m, 1.5);
        let pops: Vec<f64> = c
            .iter()
            .filter(|c| matches!(c.sound, Sound::Pop(_)))
            .map(|c| c.at.unwrap())
            .collect();
        assert_eq!(pops.len(), RIPPLE_MOST);
        for (j, at) in pops.iter().enumerate() {
            assert!((at - (2.0 + RIPPLE * j as f64)).abs() < 1e-9);
        }
        let splashes: Vec<&Cued> = c
            .iter()
            .filter(|c| matches!(c.sound, Sound::Sparkle(_)))
            .collect();
        assert_eq!(splashes.len(), 1);
        let last = 2.0 + RIPPLE * (RIPPLE_MOST - 1) as f64 + crate::core::bubbles::SWELL;
        assert!((splashes[0].at.unwrap() - last).abs() < 1e-9);
    }

    #[test]
    fn the_streak_and_a_phrase_rated_sound_at_once() {
        use crate::core::shine::{Shone, Tier};
        let at = |cue| Moment {
            at: 3.0,
            decided: 3.0,
            cue,
        };
        let phrase = |tier| {
            Cue::Phrase(Shone {
                line: 0,
                end: 2.9,
                value: 0.9,
                top: 0.9,
                tier,
            })
        };
        let m = [
            at(Cue::Word {
                end: 2.9,
                hit: true,
                streak: 5,
            }),
            at(Cue::StreakUp(1)),
            at(Cue::StreakLost),
            at(phrase(Some(Tier::TOP))),
            at(phrase(None)),
        ];
        let c: Vec<(Sound, Option<f64>)> = cues(&m, 3.0).iter().map(|c| (c.sound, c.at)).collect();
        assert_eq!(
            c,
            [
                (Sound::StepUp(1), None),
                (Sound::Break, None),
                (Sound::Tier(Tier::TOP), None),
            ]
        );
    }

    /// The last phrase's tier, decided as the song ends (its frames all
    /// in once the song is over): read from the singing's stream as any,
    /// it sounds at once.
    #[test]
    fn the_last_phrase_of_a_song_sounds_its_tier() {
        use crate::core::shine::{Shone, Tier};
        let mut singing = Singing::default();
        let mut moments = Moments::default();
        assert!(moments.frame(&singing, 60.0).cues.is_empty());
        singing.decided(
            60.2,
            Shone {
                line: 9,
                end: 59.8,
                value: 0.8,
                top: 0.9,
                tier: Some(Tier::named("Amazing")),
            },
        );
        let f = moments.frame(&singing, 60.2);
        assert!(!f.sought);
        let c: Vec<(Sound, Option<f64>)> = f.cues.iter().map(|c| (c.sound, c.at)).collect();
        assert_eq!(c, [(Sound::Tier(Tier::named("Amazing")), None)]);
    }
}
