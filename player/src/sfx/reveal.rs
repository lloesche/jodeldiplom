//! The result's reveal as the sounds take it: each beat of it
//! ([`Beat`], the same beats the stage draws, at the same clock:
//! [`crate::core::karaoke::result_now`]) made the sounds it lands with
//! ([`sounds_of`]), played at once on the frame that reaches it; what the
//! stage draws a while from a beat (the sheet unrolling, an entry written
//! in, a stamp coming down, the signature) its sound follows inside
//! ([`super::diploma`]). A frame reads the beats since the frame before;
//! one long gone ([`STALE`]: the first frame after a stall) stays silent,
//! and so does everything a skip jumped over (the reveal completed at
//! once plays nothing again). Gems coming closer together than
//! [`PLINK_EVERY`] plink fewer than one each: one for a few, in the best
//! tier among them.
//!
//! Each sound is panned to where it lands on the sheet: the heading's
//! stars either side, the seal, the stamps and the date right of the
//! middle, the signature left of it.

use super::diploma::{Diploma, Entry};
use super::moments::{Cued, STALE};
use super::sounds::{Sound, LEVELS};
use crate::core::shine::Tier;
use crate::core::singing::{Beat, Shown, Side, Verdict};

/// The closest two gems' plinks come.
pub const PLINK_EVERY: f64 = 0.035;

/// Where things land across the sheet, as pans: the heading's stars (the
/// right one's; the left one's the other way), the seal and the stamps,
/// the signature and the date.
const STARS: f32 = 0.4;
const SEAL: f32 = 0.35;
const SIGNATURE: f32 = -0.3;
const DATE: f32 = 0.05;

/// A sound a beat lands with: which, its gain and its pan.
pub type Heard = (Sound, f32, f32);

/// How many gems share a plink in `shown`'s reveal (1: each its own).
fn plink_stride(shown: &Shown) -> usize {
    (PLINK_EVERY / shown.reveal.gem_every().max(1e-6))
        .ceil()
        .max(1.0) as usize
}

/// The sounds `beat` of `shown`'s reveal lands with (three at most), each
/// with its gain and pan; none for what lands silently.
pub fn sounds_of(beat: Beat, shown: &Shown) -> impl Iterator<Item = Heard> {
    let steps = super::cheers::TICK_STEPS;
    let r = &shown.result;
    let paper = |d: Diploma, pan: f32| Some((Sound::Diploma(d), 1.0, pan));
    let one = |h: Option<Heard>| [h, None, None];
    let heard: [Option<Heard>; 3] = match beat {
        Beat::Paper => one(paper(Diploma::Unroll, 0.0)),
        Beat::Heading(side) => {
            let pan = match side {
                Side::Left => -STARS,
                Side::Right => STARS,
            };
            one(paper(Diploma::Star(side), pan))
        }
        Beat::Title => one(paper(Diploma::Title, 0.0)),
        Beat::Tick(k) => {
            let step = (k as usize * steps) / (shown.reveal.ticks() as usize).max(1);
            one(Some((Sound::Tick(step.min(steps - 1)), 1.0, 0.0)))
        }
        // Nothing counted: the nought is just written.
        Beat::Points if r.points == 0 => one(None),
        Beat::Points => one(paper(Diploma::Ding, 0.0)),
        // No word hit: the pen writes it in, nothing pops.
        Beat::Words if r.hit == 0 => one(paper(Diploma::Scribbled(Entry::Words), 0.0)),
        Beat::Words => one(paper(Diploma::Written(Entry::Words), 0.0)),
        Beat::Streak if r.streak == 0 => one(paper(Diploma::Scribbled(Entry::Streak), 0.0)),
        Beat::Streak => {
            let level = crate::core::singing::multiplier(r.streak) - 1;
            let up = (Sound::StepUp(level.clamp(1, LEVELS)), 0.55, 0.0);
            [Some(up), paper(Diploma::Written(Entry::Streak), 0.0), None]
        }
        Beat::Gem(i, _) => one(gem(i, shown)),
        Beat::Verdict(Verdict::Medal(m)) => one(Some((Sound::Medal(m), 1.0, 0.5 * SEAL))),
        // A star sticker pressed on where the seal goes, and the verdict's
        // own little tune: kind, whatever it is.
        Beat::Verdict(Verdict::Cheer) => [
            paper(Diploma::Sticker, SEAL),
            paper(Diploma::Cheer, 0.0),
            None,
        ],
        Beat::Verdict(Verdict::Whole) => [
            paper(Diploma::Sticker, SEAL),
            paper(Diploma::Whole, 0.0),
            None,
        ],
        // A sleepy star, a "Zzz" stamped beside it, a lullaby.
        Beat::Verdict(Verdict::Silent) => [
            Some((Sound::Diploma(Diploma::Sticker), 0.7, SEAL)),
            Some((Sound::Diploma(Diploma::Stamp), 0.6, SEAL)),
            paper(Diploma::Snooze, 0.0),
        ],
        Beat::NewBest => one(Some((Sound::NewBest, 1.0, 0.5 * SEAL))),
        // The best before shown calmly, or the word that the demo key's
        // singing is no best: a soft tap of the pen, never a stamp.
        Beat::Best(_) => one(paper(Diploma::Tap, SEAL)),
        Beat::End => [
            paper(Diploma::Signature, SIGNATURE),
            paper(Diploma::Date, DATE),
            None,
        ],
    };
    heard.into_iter().flatten()
}

/// Gem `i`'s plink in `shown`'s reveal (none where it shares one before
/// it): in the best tier among the gems it stands for, panned across.
fn gem(i: usize, shown: &Shown) -> Option<Heard> {
    let r = &shown.result;
    let stride = plink_stride(shown);
    if !i.is_multiple_of(stride) {
        return None;
    }
    let tier: Option<Tier> = r.phrases[i..(i + stride).min(r.phrases.len())]
        .iter()
        .filter_map(|p| p.tier)
        .max();
    let n = r.phrases.len().max(2) - 1;
    let across = i.min(n) as f32 / n as f32;
    Some((Sound::Gem(tier), 1.0, (across - 0.5) * 0.8))
}

/// The reveal's sounds followed frame by frame: the result read (its id,
/// and how far a skip put it), and where in its reveal.
#[derive(Debug, Default)]
pub struct RevealSounds {
    seen: Option<(u64, f64)>,
    last: f64,
}

impl RevealSounds {
    /// The sounds of a frame with `now` (the result shown and its reveal's
    /// time; none: none shows, and the next one is read from its start).
    pub fn frame(&mut self, now: Option<(&Shown, f64)>) -> Vec<Cued> {
        let Some((shown, t)) = now else {
            self.seen = None;
            return Vec::new();
        };
        let key = (shown.id, shown.ahead);
        match self.seen {
            // Another result: from its start.
            Some((id, _)) if id != key.0 => self.last = f64::NEG_INFINITY,
            None => self.last = f64::NEG_INFINITY,
            // Skipped: what it jumped over stays silent.
            Some((_, ahead)) if ahead != key.1 => self.last = t,
            Some(_) => {}
        }
        self.seen = Some(key);
        let cues = shown
            .reveal
            .between(self.last, t)
            .filter(|&(at, _)| at >= t - STALE)
            .flat_map(|(_, beat)| sounds_of(beat, shown))
            .map(|(sound, gain, pan)| Cued {
                sound,
                gain,
                pan,
                at: None,
            })
            .collect();
        self.last = self.last.max(t);
        cues
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::shine::Shone;
    use crate::core::singing::{Medal, SongResult};

    fn result(tiers: &[Option<Tier>]) -> SongResult {
        let phrase = |&tier: &Option<Tier>| Shone {
            line: 0,
            end: 1.0,
            value: 0.9,
            diva: 0.9,
            tier,
        };
        SongResult {
            title: "Song".into(),
            points: 8_800,
            hit: 44,
            words: 45,
            streak: 30,
            phrases: tiers.iter().map(phrase).collect(),
            medal: Some(Medal::Gold),
            best: Some(5_000),
            whole: true,
            demo: false,
        }
    }

    fn shown() -> Shown {
        let tiers = [Some(Tier::Diva), None, Some(Tier::Great), Some(Tier::Diva)];
        Shown::new(result(&tiers), 1, 1_000)
    }

    fn sounds(beat: Beat, shown: &Shown) -> Vec<Sound> {
        sounds_of(beat, shown).map(|h| h.0).collect()
    }

    fn heard(beat: Beat, shown: &Shown) -> Vec<Heard> {
        sounds_of(beat, shown).collect()
    }

    /// Frames at 60 a second from when the result shows to past its end:
    /// each beat's sounds once, on the first frame at or past it.
    #[test]
    fn each_beat_sounds_once_on_its_frame() {
        let s = shown();
        let mut reveal = RevealSounds::default();
        let mut heard: Vec<(f64, Sound)> = Vec::new();
        for f in 0..(60.0 * (s.reveal.end() + 2.0)) as u64 {
            let t = s.time(1_000 + f * 1000 / 60);
            heard.extend(reveal.frame(Some((&s, t))).iter().map(|c| (t, c.sound)));
        }
        let due: Vec<(f64, Sound)> = s
            .reveal
            .beats()
            .iter()
            .flat_map(|&(at, b)| sounds(b, &s).into_iter().map(move |x| (at, x)))
            .collect();
        assert_eq!(heard.len(), due.len(), "{heard:?}");
        for ((t, sound), (at, want)) in heard.iter().zip(&due) {
            assert_eq!(sound, want);
            assert!(
                *t >= *at && *t < at + 1.0 / 60.0 + 1e-3,
                "{want:?} at {at}: {t}"
            );
        }
        let has = |x: Sound| heard.iter().any(|(_, s)| *s == x);
        assert!(has(Sound::Medal(Medal::Gold)) && has(Sound::NewBest));
        assert!(has(Sound::Gem(Some(Tier::Diva))) && has(Sound::Gem(None)));
        assert!(heard.iter().any(|(_, s)| matches!(s, Sound::Tick(_))));
    }

    /// Every moment of the diploma has its sound, in the order drawn: the
    /// paper unrolling, the heading's stars (left, then right), the title,
    /// the ticks climbing the scale to the points' ding, each entry
    /// written in and landing, the seal, the stamp, the signature and the
    /// date.
    #[test]
    fn each_diploma_moment_has_its_sound() {
        let s = shown();
        let d = Sound::Diploma;
        assert_eq!(sounds(Beat::Paper, &s), [d(Diploma::Unroll)]);
        let stars: Vec<Heard> = [Side::Left, Side::Right]
            .into_iter()
            .flat_map(|side| heard(Beat::Heading(side), &s))
            .collect();
        assert_eq!(
            (stars[0].0, stars[1].0),
            (d(Diploma::Star(Side::Left)), d(Diploma::Star(Side::Right)))
        );
        assert!(stars[0].2 < 0.0 && stars[1].2 > 0.0, "{stars:?}");
        assert_eq!(sounds(Beat::Title, &s), [d(Diploma::Title)]);
        let steps: Vec<usize> = (0..s.reveal.ticks())
            .flat_map(|k| sounds(Beat::Tick(k), &s))
            .map(|x| match x {
                Sound::Tick(step) => step,
                other => panic!("{other:?}"),
            })
            .collect();
        assert!(steps.windows(2).all(|w| w[0] <= w[1]), "{steps:?}");
        let top = super::super::cheers::TICK_STEPS - 1;
        assert_eq!((steps[0], *steps.last().unwrap()), (0, top));
        assert_eq!(sounds(Beat::Points, &s), [d(Diploma::Ding)]);
        assert_eq!(sounds(Beat::Words, &s), [d(Diploma::Written(Entry::Words))]);
        assert_eq!(
            sounds(Beat::Streak, &s),
            [Sound::StepUp(3), d(Diploma::Written(Entry::Streak))]
        );
        let seal = heard(Beat::Verdict(Verdict::Medal(Medal::Gold)), &s);
        assert_eq!(seal[0].0, Sound::Medal(Medal::Gold));
        assert!(seal[0].2 > 0.0, "the seal's on the right");
        assert_eq!(sounds(Beat::NewBest, &s), [Sound::NewBest]);
        assert_eq!(sounds(Beat::Best(9), &s), [d(Diploma::Tap)]);
        let end = heard(Beat::End, &s);
        assert_eq!(
            (end[0].0, end[1].0),
            (d(Diploma::Signature), d(Diploma::Date))
        );
        assert!(end[0].2 < end[1].2, "signed left of the date");
        // Each verdict without a medal: a sticker and its own tune; where
        // nothing was scored, a stamp too, and the pen alone for entries
        // with nothing to cheer.
        let mut r = s.result.clone();
        r.medal = None;
        let cheer = Shown::new(r.clone(), 2, 0);
        assert_eq!(
            sounds(Beat::Verdict(Verdict::Cheer), &cheer),
            [d(Diploma::Sticker), d(Diploma::Cheer)]
        );
        assert_eq!(
            sounds(Beat::Verdict(Verdict::Whole), &cheer),
            [d(Diploma::Sticker), d(Diploma::Whole)]
        );
        (r.points, r.hit, r.streak) = (0, 0, 0);
        let silent = Shown::new(r, 3, 0);
        assert_eq!(
            sounds(Beat::Verdict(Verdict::Silent), &silent),
            [d(Diploma::Sticker), d(Diploma::Stamp), d(Diploma::Snooze)]
        );
        assert!(sounds(Beat::Points, &silent).is_empty());
        assert_eq!(
            sounds(Beat::Words, &silent),
            [d(Diploma::Scribbled(Entry::Words))]
        );
        assert_eq!(
            sounds(Beat::Streak, &silent),
            [d(Diploma::Scribbled(Entry::Streak))]
        );
        // The demo key sang it: no new best, no stamp, its best line (that
        // it is none) a soft tap.
        let demo = Shown::new(
            SongResult {
                demo: true,
                best: None,
                ..s.result.clone()
            },
            4,
            0,
        );
        let heard: Vec<Sound> = demo
            .reveal
            .beats()
            .iter()
            .flat_map(|&(_, b)| sounds(b, &demo))
            .collect();
        assert!(demo.reveal.when(Beat::Best(0)).is_some());
        assert!(!heard.contains(&Sound::NewBest) && !heard.contains(&d(Diploma::Stamp)));
        assert_eq!(sounds(Beat::Best(0), &demo), [d(Diploma::Tap)]);
    }

    /// Hundreds of gems: fewer plinks than gems, never closer than
    /// PLINK_EVERY, each in the best tier among those it stands for.
    #[test]
    fn dense_gems_plink_fewer() {
        let mut tiers = vec![None; 200];
        tiers[7] = Some(Tier::Diva);
        let s = Shown::new(result(&tiers), 1, 0);
        let plinks: Vec<(f64, Sound)> = s
            .reveal
            .beats()
            .iter()
            .filter(|(_, b)| matches!(b, Beat::Gem(..)))
            .flat_map(|&(at, b)| sounds(b, &s).into_iter().map(move |x| (at, x)))
            .collect();
        assert!(plinks.len() < 60 && plinks.len() > 20, "{}", plinks.len());
        assert!(plinks
            .windows(2)
            .all(|w| w[1].0 - w[0].0 >= PLINK_EVERY - 1e-9));
        assert!(plinks
            .iter()
            .any(|(_, s)| *s == Sound::Gem(Some(Tier::Diva))));
    }

    /// Skipped halfway: nothing it jumped over sounds, then or after; a
    /// stall long past a beat is silent too; another result (even one
    /// shown on the same millisecond) is read from its start.
    #[test]
    fn a_skip_plays_nothing_again() {
        let mut s = shown();
        let mut reveal = RevealSounds::default();
        let mut heard = 0;
        let half = (s.reveal.end() * 500.0) as u64;
        for now in (1_000..1_000 + half).step_by(16) {
            heard += reveal.frame(Some((&s, s.time(now)))).len();
        }
        assert!(heard > 0);
        s.skip(1_000 + half);
        for now in (1_000 + half..1_000 + half + 20_000).step_by(16) {
            assert!(reveal.frame(Some((&s, s.time(now)))).is_empty(), "{now}");
        }
        // Stalled: the first frame long after plays nothing stale.
        let s = shown();
        let mut reveal = RevealSounds::default();
        let t = s.time(1_000 + 3_000);
        let late = reveal.frame(Some((&s, t)));
        let fresh: usize = s
            .reveal
            .between(t - STALE, t)
            .map(|(_, b)| sounds_of(b, &s).count())
            .sum();
        assert_eq!(late.len(), fresh);
        // Another result shown on the same millisecond: from its start.
        let again = Shown::new(s.result.clone(), 2, 1_000);
        assert!(!reveal.frame(Some((&again, 0.2))).is_empty());
        // Gone, then shown again: from its start.
        assert!(reveal.frame(None).is_empty());
        assert!(!reveal.frame(Some((&s, 0.2))).is_empty());
    }
}
