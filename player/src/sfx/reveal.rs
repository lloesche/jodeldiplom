//! The result's reveal as the sounds take it: each beat of it
//! ([`Beat`], the same beats the stage draws, at the same clock:
//! [`crate::core::karaoke::result_now`]) made the sound it lands with,
//! played at once on the frame that reaches it. A frame reads the beats
//! since the frame before; one long gone ([`STALE`]: the first frame after
//! a stall) stays silent, and so does everything a skip jumped over (the
//! reveal completed at once plays nothing again). Gems coming closer
//! together than [`PLINK_EVERY`] plink fewer than one each: one for a few,
//! in the best tier among them.

use super::moments::{Cued, STALE};
use super::sounds::{Sound, LEVELS};
use crate::core::shine::Tier;
use crate::core::singing::{Beat, Shown, Verdict};

/// The closest two gems' plinks come.
pub const PLINK_EVERY: f64 = 0.035;

/// How many gems share a plink in `shown`'s reveal (1: each its own).
fn plink_stride(shown: &Shown) -> usize {
    (PLINK_EVERY / shown.reveal.gem_every().max(1e-6))
        .ceil()
        .max(1.0) as usize
}

/// The sound `beat` of `shown`'s reveal lands with, and its gain and pan;
/// none for what lands silently.
pub fn sound_of(beat: Beat, shown: &Shown) -> Option<(Sound, f32, f32)> {
    let steps = super::cheers::TICK_STEPS;
    let r = &shown.result;
    Some(match beat {
        Beat::Title => (Sound::Pop(4), 0.8, 0.0),
        Beat::Tick(k) => {
            let step = (k as usize * steps) / (shown.reveal.ticks() as usize).max(1);
            (Sound::Tick(step.min(steps - 1)), 1.0, 0.0)
        }
        // Nothing to cheer for nothing counted, or no word hit.
        Beat::Points if r.points == 0 => return None,
        Beat::Points => (Sound::Sparkle(8), 1.0, 0.0),
        Beat::Words if r.hit == 0 => return None,
        Beat::Words => (Sound::StepUp(1), 0.8, 0.0),
        Beat::Streak => {
            let level = crate::core::singing::multiplier(r.streak) - 1;
            (Sound::StepUp(level.clamp(1, LEVELS)), 0.8, 0.0)
        }
        Beat::Gem(i, _) => {
            let stride = plink_stride(shown);
            if i % stride != 0 {
                return None;
            }
            let tier: Option<Tier> = r.phrases[i..(i + stride).min(r.phrases.len())]
                .iter()
                .filter_map(|p| p.tier)
                .max();
            let n = r.phrases.len().max(2) - 1;
            let across = i.min(n) as f32 / n as f32;
            (Sound::Gem(tier), 1.0, (across - 0.5) * 0.8)
        }
        Beat::Verdict(Verdict::Medal(m)) => (Sound::Medal(m), 1.0, 0.0),
        Beat::Verdict(Verdict::Cheer) => (Sound::Tier(Tier::Good), 1.0, 0.0),
        Beat::Verdict(Verdict::Whole) => (Sound::Tier(Tier::Good), 0.7, 0.0),
        // A cheeky little "womp" for a silent song.
        Beat::Verdict(Verdict::Silent) => (Sound::Break, 0.6, 0.0),
        Beat::NewBest => (Sound::NewBest, 1.0, 0.0),
        Beat::Best(_) => (Sound::Pop(2), 0.5, 0.0),
        Beat::End => return None,
    })
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
            .filter_map(|(_, beat)| sound_of(beat, shown))
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
        }
    }

    fn shown() -> Shown {
        let tiers = [Some(Tier::Diva), None, Some(Tier::Great), Some(Tier::Diva)];
        Shown::new(result(&tiers), 1, 1_000)
    }

    /// Frames at 60 a second from when the result shows to past its end:
    /// each beat's sound once, on the first frame at or past it.
    #[test]
    fn each_beat_sounds_once_on_its_frame() {
        let s = shown();
        let mut sounds = RevealSounds::default();
        let mut heard: Vec<(f64, Sound)> = Vec::new();
        for f in 0..(60.0 * (s.reveal.end() + 2.0)) as u64 {
            let t = s.time(1_000 + f * 1000 / 60);
            heard.extend(sounds.frame(Some((&s, t))).iter().map(|c| (t, c.sound)));
        }
        let sounded: Vec<(f64, Beat)> = s
            .reveal
            .beats()
            .iter()
            .copied()
            .filter(|&(_, b)| sound_of(b, &s).is_some())
            .collect();
        assert_eq!(heard.len(), sounded.len(), "{heard:?}");
        for ((t, sound), (at, beat)) in heard.iter().zip(&sounded) {
            assert_eq!(*sound, sound_of(*beat, &s).unwrap().0, "{beat:?}");
            assert!(
                *t >= *at && *t < at + 1.0 / 60.0 + 1e-3,
                "{beat:?} at {at}: {t}"
            );
        }
        let has = |x: Sound| heard.iter().any(|(_, s)| *s == x);
        assert!(has(Sound::Medal(Medal::Gold)) && has(Sound::NewBest));
        assert!(has(Sound::Gem(Some(Tier::Diva))) && has(Sound::Gem(None)));
        assert!(heard.iter().any(|(_, s)| matches!(s, Sound::Tick(_))));
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
            .filter_map(|&(at, b)| sound_of(b, &s).map(|x| (at, x.0)))
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
        let mut sounds = RevealSounds::default();
        let mut heard = 0;
        let half = (s.reveal.end() * 500.0) as u64;
        for now in (1_000..1_000 + half).step_by(16) {
            heard += sounds.frame(Some((&s, s.time(now)))).len();
        }
        assert!(heard > 0);
        s.skip(1_000 + half);
        for now in (1_000 + half..1_000 + half + 20_000).step_by(16) {
            assert!(sounds.frame(Some((&s, s.time(now)))).is_empty(), "{now}");
        }
        // Stalled: the first frame long after plays nothing stale.
        let s = shown();
        let mut sounds = RevealSounds::default();
        let t = s.time(1_000 + 3_000);
        let late = sounds.frame(Some((&s, t)));
        let fresh = s
            .reveal
            .between(t - STALE, t)
            .filter(|&(_, b)| sound_of(b, &s).is_some());
        assert_eq!(late.len(), fresh.count());
        // Another result shown on the same millisecond: from its start.
        let again = Shown::new(s.result.clone(), 2, 1_000);
        assert!(!sounds.frame(Some((&again, 0.2))).is_empty());
        // Gone, then shown again: from its start.
        assert!(sounds.frame(None).is_empty());
        assert!(!sounds.frame(Some((&s, 0.2))).is_empty());
    }
}
