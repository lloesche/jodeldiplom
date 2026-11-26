//! The game's moments as the sounds take them: read from the singing's one
//! stream of them ([`Singing::moments_since`], through a [`Cursor`]), each
//! made the conductor's event it goes with ([`Cued`], [`Event`]) at the
//! singing's time it happens (or at once, for one happening now: the
//! streak lost, a phrase's tier). A bubble bursting is a word's pop up the
//! ladder, a shot fired its zap, the streak lost from a rung its wind-down,
//! a phrase rated its chime (and the jackpot where its tier's row fires
//! one, [`crate::gui::fx::Fx::Jackpot`]).
//!
//! A pop is told the words in a row with it ([`Event::Word`]), as the
//! singing counts them as it bursts ([`crate::core::singing::Burst::run`]:
//! a bubble bursts as its word ends, once won, mostly before the word is
//! scored), and how big it bursts (held full long, bigger:
//! [`crate::core::singing::Burst::size`]). A rung's stinger goes with the
//! pop that reaches it ([`super::conductor`]), so a word held full long
//! pumps on to its end and bursts with its rung. Bubbles bursting together
//! pop one after another, a ripple up the line
//! ([`RIPPLE`]); one seen long after it happened ([`STALE`]: the first
//! frames after a stall) stays silent, and so does a bubble's splash (the
//! pop's glint is it). The stream going back (a seek, the song sung again)
//! is told ([`Frame::sought`]): what was cued and not heard yet is of
//! another place in the song.

use std::path::PathBuf;

use super::conductor::Event;
use crate::core::singing::{Cue, Cursor, Moment, Singing};
use crate::gui::fx::{strength, Fx};
use crate::tiers::Tier;

/// An event cued: what, and when (the singing's time; `None`: at once).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cued {
    /// The event.
    pub event: Event,
    /// When.
    pub at: Option<f64>,
}

/// What a frame of the game brought: its events, and whether what was
/// cued before is off.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Frame {
    /// The new events.
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

/// The pan of a moment `across` its line (0 its start, 1 its end).
fn pan(across: f32) -> f32 {
    (across.clamp(0.0, 1.0) - 0.5) * 0.8
}

/// Whether a phrase reaching `tier` hits the jackpot (its row fires one).
pub fn jackpot(tier: Tier) -> bool {
    strength(tier.spec().fx.on_phrase, Fx::Jackpot) > 0.0
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
        let mut cues = Vec::new();
        for m in since.moments {
            cue(m, clock, &mut cues);
        }
        Frame {
            cues,
            sought: since.sought,
        }
    }
}

/// The events of `m` at `clock`, into `out`.
fn cue(m: &Moment, clock: f64, out: &mut Vec<Cued>) {
    {
        let mut at = m.at;
        let mut push = |event: Event, at: f64| {
            if at >= clock - STALE {
                let at = (at > clock + NOW).then_some(at);
                out.push(Cued { event, at });
            }
        };
        match m.cue {
            Cue::Pop(b) => {
                // Of the bubbles bursting together, the first few pop, one
                // after another.
                let (j, _) = b.burst;
                if j >= RIPPLE_MOST {
                    return;
                }
                at += RIPPLE * j as f64;
                let pan = pan(b.across);
                let size = b.size() as f32;
                let streak = b.run;
                push(Event::Word { streak, pan, size }, at);
            }
            Cue::Shot { across, .. } => push(Event::Zap { pan: pan(across) }, at),
            Cue::StreakLost(from) => push(Event::Lost(from), at),
            Cue::Phrase(shone) => {
                let Some(tier) = shone.tier else {
                    return;
                };
                push(Event::Phrase(tier), at);
                if jackpot(tier) {
                    push(Event::Jackpot, at);
                }
            }
            Cue::Word { .. } | Cue::Splash(_) | Cue::StreakUp(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::shine::Shone;

    /// Word `word`'s bubble bursting at `at` as `burst` (its place among
    /// those bursting together, and their count).
    fn pop(at: f64, word: usize, burst: (usize, usize)) -> [Moment; 2] {
        let b = crate::core::singing::Burst {
            line: 0,
            word,
            height: 0.0,
            across: 0.25,
            burst,
            run: word as u32 + 1,
            held: 0.0,
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

    fn cues(moments: &[Moment], clock: f64) -> Vec<Cued> {
        let mut out = Vec::new();
        for moment in moments {
            cue(moment, clock, &mut out);
        }
        out
    }

    fn streaks(c: &[Cued]) -> Vec<u32> {
        c.iter()
            .filter_map(|c| match c.event {
                Event::Word { streak, .. } => Some(streak),
                _ => None,
            })
            .collect()
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
        m.extend(pop(lands, 2, (0, 1)));
        let c = cues(&m, 1.9);
        assert_eq!(c.len(), 2, "no splash: {c:?}");
        assert!(matches!(c[0].event, Event::Zap { pan } if pan > 0.15));
        assert_eq!(c[0].at, Some(2.0));
        assert!(matches!(c[1].event, Event::Word { streak: 3, .. }));
        assert_eq!(c[1].at, Some(lands));
    }

    #[test]
    fn a_pop_sounds_on_its_time_on_the_left_of_the_line() {
        let m = pop(1.2, 0, (0, 1));
        let c = cues(&m, 1.0);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].at, Some(1.2));
        assert!(matches!(c[0].event, Event::Word { pan, .. } if pan < -0.15));
        // Due now: at once; long gone: silent.
        assert_eq!(cues(&m, 1.21)[0].at, None);
        assert!(cues(&m, 1.2 + STALE + 0.01).is_empty());
    }

    #[test]
    fn bubbles_bursting_together_ripple_up_the_ladder() {
        let m: Vec<Moment> = (0..8).flat_map(|k| pop(2.0, k, (k, 8))).collect();
        let c = cues(&m, 1.5);
        assert_eq!(c.len(), RIPPLE_MOST);
        for (j, cue) in c.iter().enumerate() {
            assert!((cue.at.unwrap() - (2.0 + RIPPLE * j as f64)).abs() < 1e-9);
        }
        assert_eq!(streaks(&c), [1, 2, 3, 4, 5, 6]);
    }

    /// Each pop climbs the run the singing counts as it bursts: a word
    /// missed alone and rescued across the first rung (the fifth of
    /// fourteen) never takes the ladder back, and the rung is reached once.
    #[test]
    fn a_rescue_across_a_rung_never_takes_the_ladder_back() {
        let mut s = crate::core::singing::tests::fourteen(&[4]);
        s.finish();
        let c = cues(s.moments(), 0.0);
        let runs = streaks(&c);
        assert!(
            c.iter().any(|c| matches!(c.event, Event::Zap { .. })),
            "a rescue"
        );
        // Up a step a word, the rescued one too, through the first rung, in
        // the order they are heard (each word held full to its end, the
        // one rescued bursting as its shot lands, before its shooter).
        assert_eq!(runs, (1..=14).collect::<Vec<u32>>());
        let mut heard: Vec<(f64, u32)> = (c.iter())
            .filter_map(|c| match c.event {
                Event::Word { streak, .. } => Some((c.at.unwrap(), streak)),
                _ => None,
            })
            .collect();
        heard.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert!(heard.windows(2).all(|w| w[1].1 == w[0].1 + 1), "{heard:?}");
        let first = Tier::ALL[0].spec().streak;
        assert!(runs.iter().filter(|&&r| r == first).count() == 1);
        assert!(!c.iter().any(|c| matches!(c.event, Event::Lost(_))));
    }

    /// Sung live, a long word won early pops as it ends (its sound cued a
    /// sweep ahead, as any), the bigger for being held full so long; the
    /// short words before and after it pop as any, the ladder climbing a
    /// step a word.
    #[test]
    fn a_word_held_full_pops_as_it_ends_and_bigger() {
        let mut s = crate::core::singing::tests::a_long_word();
        let mut moments = Moments::default();
        let mut words = Vec::new();
        for i in 0..500 {
            let t = f64::from(i) / 100.0;
            let c = s.word_at(t).map(|(_, c)| c as f32);
            s.hear(t, c);
            for cue in moments.frame(&s, t).cues {
                if let Event::Word { streak, size, .. } = cue.event {
                    words.push((t, cue.at, streak, size));
                }
            }
        }
        let [first, (told, at, streak, size), last] = words[..] else {
            panic!("three pops: {words:?}");
        };
        assert_eq!((at, streak, size), (Some(3.6), 2, 1.0));
        assert!((3.6 - crate::core::bubbles::SWEEP - 0.01..3.6).contains(&told));
        assert_eq!((first.1, first.2, first.3), (Some(0.5), 1, 0.0));
        assert_eq!((last.1, last.2, last.3), (Some(4.2), 3, 0.0));
    }

    #[test]
    fn the_streak_lost_and_a_phrase_rated_sound_at_once_the_top_with_its_jackpot() {
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
                end: 2.0,
                hit: true,
                streak: 5,
            }),
            at(Cue::StreakUp(Tier::ALL[0])),
            at(Cue::StreakLost(Tier::ALL[0])),
            at(phrase(Some(Tier::named("Great")))),
            at(phrase(Some(Tier::TOP))),
            at(phrase(None)),
        ];
        let c: Vec<(Event, Option<f64>)> = cues(&m, 3.0).iter().map(|c| (c.event, c.at)).collect();
        assert_eq!(
            c,
            [
                (Event::Lost(Tier::ALL[0]), None),
                (Event::Phrase(Tier::named("Great")), None),
                (Event::Phrase(Tier::TOP), None),
                (Event::Jackpot, None),
            ]
        );
        assert!(Tier::ALL[..Tier::COUNT - 1].iter().all(|&t| !jackpot(t)));
    }

    /// The last phrase's tier, decided as the song ends (its frames all
    /// in once the song is over): read from the singing's stream as any,
    /// it sounds at once.
    #[test]
    fn the_last_phrase_of_a_song_sounds_its_tier() {
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
        let c: Vec<(Event, Option<f64>)> = f.cues.iter().map(|c| (c.event, c.at)).collect();
        assert_eq!(c, [(Event::Phrase(Tier::named("Amazing")), None)]);
    }
}
