//! The result's reveal as the sounds take them: each beat of it
//! ([`Beat`], the same beats the stage draws, at the same clock:
//! [`crate::core::karaoke::result_now`]) made the sounds it lands with
//! ([`sounds_of`]), played by the game's synth at once on the frame that
//! reaches it ([`Diploma`]): the diploma's paper, pen, stamps and tunes
//! ([`Sheet`]: what the stage draws a while from a beat, the sheet
//! unrolling, an entry written in, a stamp coming down, the signature, its
//! sound follows inside), and its music (the count-up's ticks, the points
//! all counted, the gems, the best streak's rung). A frame reads the beats
//! since the frame
//! before; one long gone ([`STALE`]: the first frame after a stall) stays
//! silent, and so does everything a skip jumped over (the reveal completed
//! at once plays nothing again). Gems coming closer together than
//! [`PLINK_EVERY`] plink fewer than one each: one for a few, in the best
//! tier among them.
//!
//! The sounds are panned to where they land on the sheet: the heading's
//! stars either side, the gems across it, the seal, the stamps and the
//! date right of the middle, the signature left of it.

use super::conductor::{Diploma, Entry, Sheet};
use super::moments::STALE;
use crate::core::karaoke::result::{Beat, Shown, Side, Verdict};
use crate::core::shine::Tier;

/// The closest two gems' plinks come.
pub const PLINK_EVERY: f64 = 0.035;

/// Where things land across the sheet, as pans: the heading's stars (the
/// right one's; the left one's the other way), the seal and the stamps,
/// the signature and the date.
const STARS: f32 = 0.4;
const SEAL: f32 = 0.35;
const SIGNATURE: f32 = -0.3;
const DATE: f32 = 0.05;

/// How many gems share a plink in `shown`'s reveal (1: each its own).
fn plink_stride(shown: &Shown) -> usize {
    (PLINK_EVERY / shown.reveal.gem_every().max(1e-6))
        .ceil()
        .max(1.0) as usize
}

/// The sounds `beat` of `shown`'s reveal lands with (three at most); none
/// for what lands silently.
pub fn sounds_of(beat: Beat, shown: &Shown) -> impl Iterator<Item = Diploma> {
    let r = &shown.result;
    let on = |sound: Sheet, gain: f32, pan: f32| Some(Diploma::Sheet { sound, gain, pan });
    let paper = |sound: Sheet, pan: f32| on(sound, 1.0, pan);
    let one = |h: Option<Diploma>| [h, None, None];
    let heard: [Option<Diploma>; 3] = match beat {
        Beat::Paper => one(paper(Sheet::Unroll, 0.0)),
        Beat::Heading(side) => {
            let pan = match side {
                Side::Left => -STARS,
                Side::Right => STARS,
            };
            one(paper(Sheet::Star(side), pan))
        }
        Beat::Title => one(paper(Sheet::Title, 0.0)),
        Beat::Tick(k) => {
            let of = shown.reveal.ticks();
            one(Some(Diploma::Tick { k, of }))
        }
        // Nothing counted: the nought is just written.
        Beat::Points if r.points == 0 => one(None),
        Beat::Points => {
            let best = r.phrases.iter().filter_map(|p| p.tier).max();
            one(Some(Diploma::Points(best)))
        }
        // No word hit: the pen writes it in, nothing pops.
        Beat::Words if r.hit == 0 => one(paper(Sheet::Scribbled(Entry::Words), 0.0)),
        Beat::Words => one(paper(Sheet::Written(Entry::Words), 0.0)),
        Beat::Streak if r.streak == 0 => one(paper(Sheet::Scribbled(Entry::Streak), 0.0)),
        Beat::Streak => {
            // The stinger of the rung it reached (under the first, only
            // the pen).
            let rung = crate::core::singing::rung(r.streak);
            let up = rung.map(Diploma::Rung);
            [up, paper(Sheet::Written(Entry::Streak), 0.0), None]
        }
        Beat::Gem(i, _) => one(gem(i, shown)),
        Beat::Verdict(Verdict::Medal(m)) => one(paper(Sheet::Medal(m), 0.5 * SEAL)),
        // A star sticker pressed on where the seal goes, and the verdict's
        // own little tune: kind, whatever it is.
        Beat::Verdict(Verdict::Cheer) => {
            [paper(Sheet::Sticker, SEAL), paper(Sheet::Cheer, 0.0), None]
        }
        Beat::Verdict(Verdict::Whole) => {
            [paper(Sheet::Sticker, SEAL), paper(Sheet::Whole, 0.0), None]
        }
        // A sleepy star, a "Zzz" stamped beside it, a lullaby.
        Beat::Verdict(Verdict::Silent) => [
            on(Sheet::Sticker, 0.7, SEAL),
            on(Sheet::Stamp, 0.6, SEAL),
            paper(Sheet::Snooze, 0.0),
        ],
        Beat::NewBest => one(paper(Sheet::NewBest, 0.5 * SEAL)),
        // The best before shown calmly, or the word that the demo key's
        // singing is no best: a soft tap of the pen, never a stamp.
        Beat::Best(_) => one(paper(Sheet::Tap, SEAL)),
        Beat::End => [
            paper(Sheet::Signature, SIGNATURE),
            paper(Sheet::Date, DATE),
            None,
        ],
    };
    heard.into_iter().flatten()
}

/// Gem `i`'s plink in `shown`'s reveal (none where it shares one before
/// it): in the best tier among the gems it stands for, panned across.
fn gem(i: usize, shown: &Shown) -> Option<Diploma> {
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
    let pan = (across - 0.5) * 0.8;
    Some(Diploma::Gem { tier, pan })
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
    pub fn frame(&mut self, now: Option<(&Shown, f64)>) -> Vec<Diploma> {
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
        let heard = shown
            .reveal
            .between(self.last, t)
            .filter(|&(at, _)| at >= t - STALE)
            .flat_map(|(_, beat)| sounds_of(beat, shown))
            .collect();
        self.last = self.last.max(t);
        heard
    }

    /// Whether a result is being revealed (since the last frame).
    pub fn showing(&self) -> bool {
        self.seen.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::karaoke::result::{Medal, SongResult};
    use crate::core::shine::Shone;

    fn result(tiers: &[Option<Tier>]) -> SongResult {
        let phrase = |&tier: &Option<Tier>| Shone {
            line: 0,
            end: 1.0,
            value: 0.9,
            top: 0.9,
            tier,
        };
        SongResult {
            title: "Song".into(),
            points: 8_800,
            hit: 44,
            words: 45,
            streak: Tier::named("Awesome").spec().streak,
            phrases: tiers.iter().map(phrase).collect(),
            medal: Some(Medal::Gold),
            best: Some(5_000),
            whole: true,
            demo: false,
        }
    }

    fn shown() -> Shown {
        let tiers = [
            Some(Tier::TOP),
            None,
            Some(Tier::named("Great")),
            Some(Tier::TOP),
        ];
        Shown::new(result(&tiers), 1, 1_000)
    }

    /// The sheet's sounds of `beat`.
    fn sounds(beat: Beat, shown: &Shown) -> Vec<Sheet> {
        sounds_of(beat, shown)
            .filter_map(|h| match h {
                Diploma::Sheet { sound, .. } => Some(sound),
                _ => None,
            })
            .collect()
    }

    /// The music of `beat` (the rest).
    fn live(beat: Beat, shown: &Shown) -> Vec<Diploma> {
        sounds_of(beat, shown)
            .filter(|h| !matches!(h, Diploma::Sheet { .. }))
            .collect()
    }

    /// The pans of the sheet's sounds of `beat`.
    fn pans(beat: Beat, shown: &Shown) -> Vec<f32> {
        sounds_of(beat, shown)
            .filter_map(|h| match h {
                Diploma::Sheet { pan, .. } => Some(pan),
                _ => None,
            })
            .collect()
    }

    /// `sound` at its level, at `pan`.
    fn sheet(sound: Sheet, pan: f32) -> Diploma {
        Diploma::Sheet {
            sound,
            gain: 1.0,
            pan,
        }
    }

    /// Frames at 60 a second from when the result shows to past its end:
    /// each beat's sounds once, on the first frame at or past it.
    #[test]
    fn each_beat_sounds_once_on_its_frame() {
        let s = shown();
        let mut reveal = RevealSounds::default();
        let mut heard: Vec<(f64, Diploma)> = Vec::new();
        for f in 0..(60.0 * (s.reveal.end() + 2.0)) as u64 {
            let t = s.time(1_000 + f * 1000 / 60);
            heard.extend(reveal.frame(Some((&s, t))).iter().map(|&h| (t, h)));
        }
        assert!(reveal.showing());
        let due: Vec<(f64, Diploma)> = s
            .reveal
            .beats()
            .iter()
            .flat_map(|&(at, b)| sounds_of(b, &s).map(move |x| (at, x)))
            .collect();
        assert_eq!(heard.len(), due.len(), "{heard:?}");
        for ((t, sound), (at, want)) in heard.iter().zip(&due) {
            assert_eq!(sound, want);
            assert!(
                *t >= *at && *t < at + 1.0 / 60.0 + 1e-3,
                "{want:?} at {at}: {t}"
            );
        }
        let has = |x: Diploma| heard.iter().any(|(_, h)| *h == x);
        assert!(has(sheet(Sheet::Medal(Medal::Gold), 0.5 * SEAL)));
        assert!(has(sheet(Sheet::NewBest, 0.5 * SEAL)));
        let gem = |t: Option<Tier>| {
            heard
                .iter()
                .any(|(_, h)| matches!(h, Diploma::Gem { tier, .. } if *tier == t))
        };
        assert!(gem(Some(Tier::TOP)) && gem(None));
        // Across the sheet, the first on the left, the last on the right.
        let pans: Vec<f32> = (heard.iter())
            .filter_map(|(_, h)| match h {
                Diploma::Gem { pan, .. } => Some(*pan),
                _ => None,
            })
            .collect();
        assert!(pans[0] < 0.0 && *pans.last().unwrap() > 0.0, "{pans:?}");
        assert!(heard.iter().any(|(_, h)| matches!(h, Diploma::Tick { .. })));
        assert!(reveal.frame(None).is_empty() && !reveal.showing());
    }

    /// Every moment of the diploma has its sound, in the order drawn: the
    /// paper unrolling, the heading's stars (left, then right), the title,
    /// the ticks climbing to the points' chime in the best tier reached,
    /// each entry written in and landing (the streak's with its rung), the
    /// seal, the stamp, the signature and the date.
    #[test]
    fn each_diploma_moment_has_its_sound() {
        let s = shown();
        assert_eq!(sounds(Beat::Paper, &s), [Sheet::Unroll]);
        let stars: Vec<Sheet> = [Side::Left, Side::Right]
            .into_iter()
            .flat_map(|side| sounds(Beat::Heading(side), &s))
            .collect();
        assert_eq!(stars, [Sheet::Star(Side::Left), Sheet::Star(Side::Right)]);
        assert!(pans(Beat::Heading(Side::Left), &s)[0] < 0.0);
        assert!(pans(Beat::Heading(Side::Right), &s)[0] > 0.0);
        assert_eq!(sounds(Beat::Title, &s), [Sheet::Title]);
        let of = s.reveal.ticks();
        let ticks: Vec<Diploma> = (0..of).flat_map(|k| live(Beat::Tick(k), &s)).collect();
        let want: Vec<Diploma> = (0..of).map(|k| Diploma::Tick { k, of }).collect();
        assert_eq!(ticks, want);
        assert_eq!(live(Beat::Points, &s), [Diploma::Points(Some(Tier::TOP))]);
        assert_eq!(sounds(Beat::Words, &s), [Sheet::Written(Entry::Words)]);
        assert_eq!(
            live(Beat::Streak, &s),
            [Diploma::Rung(Tier::named("Awesome"))]
        );
        assert_eq!(sounds(Beat::Streak, &s), [Sheet::Written(Entry::Streak)]);
        let seal = Beat::Verdict(Verdict::Medal(Medal::Gold));
        assert_eq!(sounds(seal, &s), [Sheet::Medal(Medal::Gold)]);
        assert!(pans(seal, &s)[0] > 0.0, "the seal's on the right");
        assert_eq!(sounds(Beat::NewBest, &s), [Sheet::NewBest]);
        assert_eq!(sounds(Beat::Best(9), &s), [Sheet::Tap]);
        assert_eq!(sounds(Beat::End, &s), [Sheet::Signature, Sheet::Date]);
        let end = pans(Beat::End, &s);
        assert!(end[0] < end[1], "signed left of the date");
        // Each verdict without a medal: a sticker and its own tune; where
        // nothing was scored, a stamp too, and the pen alone for entries
        // with nothing to cheer.
        let mut r = s.result.clone();
        r.medal = None;
        let cheer = Shown::new(r.clone(), 2, 0);
        assert_eq!(
            sounds(Beat::Verdict(Verdict::Cheer), &cheer),
            [Sheet::Sticker, Sheet::Cheer]
        );
        assert_eq!(
            sounds(Beat::Verdict(Verdict::Whole), &cheer),
            [Sheet::Sticker, Sheet::Whole]
        );
        (r.points, r.hit, r.streak) = (0, 0, 0);
        r.phrases.clear();
        let silent = Shown::new(r, 3, 0);
        assert_eq!(
            sounds(Beat::Verdict(Verdict::Silent), &silent),
            [Sheet::Sticker, Sheet::Stamp, Sheet::Snooze]
        );
        assert_eq!(sounds_of(Beat::Points, &silent).count(), 0);
        assert_eq!(
            sounds(Beat::Words, &silent),
            [Sheet::Scribbled(Entry::Words)]
        );
        assert_eq!(
            sounds(Beat::Streak, &silent),
            [Sheet::Scribbled(Entry::Streak)]
        );
        assert!(live(Beat::Streak, &silent).is_empty());
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
        let heard: Vec<Sheet> = demo
            .reveal
            .beats()
            .iter()
            .flat_map(|&(_, b)| sounds(b, &demo))
            .collect();
        assert!(demo.reveal.when(Beat::Best(0)).is_some());
        assert!(!heard.contains(&Sheet::NewBest) && !heard.contains(&Sheet::Stamp));
        assert_eq!(sounds(Beat::Best(0), &demo), [Sheet::Tap]);
    }

    /// Hundreds of gems: fewer plinks than gems, never closer than
    /// PLINK_EVERY, each in the best tier among those it stands for.
    #[test]
    fn dense_gems_plink_fewer() {
        let mut tiers = vec![None; 200];
        tiers[7] = Some(Tier::TOP);
        let s = Shown::new(result(&tiers), 1, 0);
        let plinks: Vec<(f64, Diploma)> = s
            .reveal
            .beats()
            .iter()
            .filter(|(_, b)| matches!(b, Beat::Gem(..)))
            .flat_map(|&(at, b)| live(b, &s).into_iter().map(move |x| (at, x)))
            .collect();
        assert!(plinks.len() < 60 && plinks.len() > 20, "{}", plinks.len());
        assert!(plinks
            .windows(2)
            .all(|w| w[1].0 - w[0].0 >= PLINK_EVERY - 1e-9));
        assert!(plinks.iter().any(|(_, g)| matches!(
            g,
            Diploma::Gem {
                tier: Some(Tier::TOP),
                ..
            }
        )));
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
