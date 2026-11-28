//! The demo key, an autoplay ([`Demo`]): while it is held in Karaoke
//! mode, the game
//! hears a perfect singer in place of the microphone, to show it off
//! without singing. Each of its frames (a [`HOP`] apart, as the pitch
//! tracker's) is the note the bubbles are made of at its time, a few cents
//! of slow wobble on it so the arrow lives, and nothing where no note is
//! due. They come at the song's time as heard, for no voice of it comes
//! late (while it is held the game's delay is none:
//! [`crate::core::karaoke::KaraokeState::sung_late`]; the arrow takes up each
//! note as it reaches the playhead), and go through [`Singing::hear`] as
//! the microphone's do:
//! the bubbles, the streak, the shine and the result follow as for
//! singing. Let go of, it stops at once: no frame of it after, and what it
//! sang stands, the microphone heard from where it stopped
//! ([`Demo::microphone`]: its voice comes late, so its first frames are
//! of moments the demo sang).
//!
//! A song it sang in is marked ([`super::Tally::demo`], kept with the
//! scoring: a song started over is no longer marked): its result shows,
//! never as a best.

use jodel_rhythm::yin::HOP;

use super::Singing;

/// How far the wobble goes either way, in hundredths of a semitone.
const WOBBLE: f64 = 6.0;
/// How often it goes up and down, a second.
const WOBBLE_HZ: f64 = 5.0;
/// The longest the demo goes on from its last frame, in seconds: from
/// further on (a seek, the window held up) it starts again where the song
/// is, never singing what was not heard.
const GAP: f64 = 0.5;

/// The demo key: held or not, where its singing got to, and, let go of,
/// where it had sung to.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Demo {
    held: bool,
    /// When its last frame was, in the singing's time.
    last: Option<f64>,
    /// Let go of: the latest it sang to, until the microphone takes over
    /// from there ([`Demo::follow`]).
    sang_to: Option<f64>,
}

impl Demo {
    /// Whether it is held.
    pub fn held(&self) -> bool {
        self.held
    }

    /// The key went down (or repeats): the demo sings from now.
    pub fn hold(&mut self) {
        self.held = true;
    }

    /// The key let go of: the demo stops at once, what it sang stands (to
    /// the latest it sang to, also when it was let go of and held again
    /// within a frame, singing nothing between).
    pub fn let_go(&mut self) {
        self.held = false;
        self.sang_to = match (self.last.take(), self.sang_to) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
    }

    /// Nothing of what it sang waits for the microphone any more: the mode
    /// left, another song.
    pub fn forget_sang(&mut self) {
        self.sang_to = None;
    }

    /// Once a frame, the song heard at `heard` seconds, the microphone's
    /// voice coming `late`: the microphone has taken over once its frames
    /// are past where the demo sang to, or the song's time went back from
    /// there (a seek, another song).
    pub fn follow(&mut self, heard: f64, late: f64) {
        if let Some(to) = self.sang_to {
            if heard - late > to || heard < to - HOP {
                self.sang_to = None;
            }
        }
    }

    /// The microphone's frame caught at `at` seconds into the song, its
    /// voice coming `late`, at `cents`: heard by `singing` at the singing's
    /// time (`at` less `late`) and by the delay's `estimate`; not while
    /// that time is at or before where the demo, just let go of, sang to
    /// (the voice comes late, so the microphone's first frames after the
    /// key are of moments the demo sang: they would take its singing back).
    pub fn microphone(
        &self,
        (singing, estimate): (&mut Singing, &mut crate::core::lag::estimate::Estimator),
        (at, late): (f64, f64),
        cents: Option<f32>,
    ) {
        let t = at - late;
        if self.sang_to.is_none_or(|to| t > to + 1e-9) {
            singing.hear(t, cents);
            estimate.hear(singing, at, cents);
        }
    }

    /// While held, the perfect singer's frames after its last up to `to`
    /// (the singing's time: the song's, with the demo held) heard by
    /// `singing`, the song marked as the
    /// demo's; nothing while not.
    pub fn sing(&mut self, singing: &mut Singing, to: f64) {
        if !self.held {
            return;
        }
        let mut t = match self.last {
            Some(last) if to >= last && to - last <= GAP => last + HOP,
            _ => to,
        };
        while t <= to + 1e-9 {
            singing.hear(t, perfect(singing, t));
            singing.tally.demo = true;
            self.last = Some(t);
            t += HOP;
        }
    }
}

/// The perfect singer at `t`: the note due then, wobbling a little; none
/// off the notes.
fn perfect(singing: &Singing, t: f64) -> Option<f32> {
    let (l, k) = singing.notes.note_at(t)?;
    let note = singing.notes.lines[l].notes[k].cents;
    let wobble = WOBBLE * (std::f64::consts::TAU * WOBBLE_HZ * t).sin();
    Some((note + wobble) as f32)
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use jodel_rhythm::waveform::Melody;

    use super::*;
    use crate::core::singing::{melodic, Cue};

    /// How many words the demo song has: the first rung's.
    fn words() -> u32 {
        crate::tiers::Tier::ALL[0].spec().streak
    }

    /// A song of one line of [`words`] words, a second each from 1 s, up
    /// and down a third; its singing ready.
    fn a_line() -> Singing {
        let frames = 100 * (words() as usize + 3);
        let mut m = Melody {
            cents: vec![0; frames],
            sure: vec![0; frames],
        };
        for i in 100..frames - 200 {
            m.cents[i] = [6000, 6200, 6400, 6200, 6000, 6400][(i / 100 - 1) % 6];
            m.sure[i] = 255;
        }
        let words = (0..words())
            .map(|k| jodel_lyrics::Word {
                start: Some(1.0 + f64::from(k)),
                end: Some(2.0 + f64::from(k)),
                confidence: 0.9,
                ..jodel_lyrics::Word::new("la")
            })
            .collect();
        let lyrics = Arc::new(jodel_lyrics::Lyrics {
            lines: vec![jodel_lyrics::Line {
                words,
                ..jodel_lyrics::Line::default()
            }],
            source: jodel_lyrics::Source::Unknown,
            timing: jodel_lyrics::Timing::default(),
        });
        let mut singing = Singing::default();
        singing.prepare(Path::new("a"), &lyrics, &melodic(&m));
        singing
    }

    /// The demo sung up to `to` seconds, as drawn frames ask (every 16 ms).
    fn frames(demo: &mut Demo, singing: &mut Singing, from: f64, to: f64) {
        let mut t = from;
        while t <= to {
            demo.sing(singing, t);
            t += 0.016;
        }
    }

    #[test]
    fn held_through_a_phrase_every_word_bursts_and_the_multiplier_goes_up() {
        let mut singing = a_line();
        let mut demo = Demo::default();
        demo.hold();
        let n = words();
        frames(&mut demo, &mut singing, 0.0, f64::from(n + 2));
        let popped: Vec<(usize, usize)> = singing
            .moments()
            .iter()
            .filter_map(|m| match m.cue {
                Cue::Pop(b) => Some((b.line, b.word)),
                _ => None,
            })
            .collect();
        assert_eq!(popped, (0..n as usize).map(|w| (0, w)).collect::<Vec<_>>());
        let t = singing.tally;
        assert_eq!((t.hit_words, t.scored_words, t.streak), (n, n, n));
        let good = crate::tiers::Tier::ALL[0];
        assert_eq!(singing.rung(), Some(good));
        assert!(singing
            .moments()
            .iter()
            .any(|m| m.cue == Cue::StreakUp(good)));
        assert!(t.demo, "the song is marked as the demo's");
        // On its notes, a few cents off at most (from the key it is heard
        // in, its own wobble in it): the arrow alive on them.
        let on: Vec<f64> = singing.heard.iter().filter_map(|h| h.off).collect();
        let most = on.iter().fold(0.0f64, |m, o| m.max(o.abs()));
        assert!(
            on.len() > 500 && most <= 2.0 * WOBBLE,
            "{} {most}",
            on.len()
        );
        assert!(on.iter().any(|o| o.abs() > 1.0), "it wobbles");
        assert!(singing.heard.iter().all(|h| h.hit));
    }

    #[test]
    fn let_go_it_stops_at_once() {
        let mut singing = a_line();
        let mut demo = Demo::default();
        demo.hold();
        frames(&mut demo, &mut singing, 0.0, 2.5);
        demo.let_go();
        assert!(!demo.held());
        let heard = singing.heard.len();
        let last = singing.last_heard().unwrap();
        assert!((last - 2.5).abs() < 0.02, "{last}");
        frames(&mut demo, &mut singing, 2.5, 5.0);
        assert_eq!(singing.heard.len(), heard, "nothing heard after");
        assert_eq!(singing.last_heard(), Some(last));
        // Held again later: from there, nothing in between sung.
        demo.hold();
        demo.sing(&mut singing, 5.0);
        assert!(singing.heard[heard..].iter().all(|h| h.t >= 5.0 - 1e-9));
    }

    /// Let go of just after a word was scored (and held and let go of
    /// again within a frame), the microphone's frames coming 150 ms late
    /// (silent, the singer just let go of a key) through the path
    /// MicSession::sing takes: what the demo sang stands (the word hit,
    /// the streak, no moment taken back), the microphone heard from where
    /// it stopped; the song sought back, it is heard from there at once.
    #[test]
    fn let_go_what_it_sang_stands_and_the_microphone_goes_on_after_it() {
        use crate::core::lag::estimate::Estimator;
        let mut singing = a_line();
        let mut estimate = Estimator::default();
        let mut demo = Demo::default();
        demo.hold();
        // Let go of just after the first word was scored (0.3 s after its
        // end), so a take-back would cut that moment.
        frames(&mut demo, &mut singing, 0.0, 2.35);
        demo.let_go();
        // Held and let go of again within a frame, singing nothing: where
        // it sang to stands.
        demo.hold();
        demo.let_go();
        let popped = |s: &Singing| {
            (s.moments().iter()).any(|m| matches!(m.cue, Cue::Pop(b) if (b.line, b.word) == (0, 0)))
        };
        assert!(popped(&singing), "the first word burst");
        let (tally, last) = (singing.tally, singing.last_heard().unwrap());
        assert_eq!((tally.hit_words, tally.scored_words), (1, 1), "{tally:?}");
        let late = 0.15;
        // The demo's frames the microphone's first come over, all hit.
        let overlap = |s: &Singing| -> Vec<(f64, bool)> {
            (s.heard.iter())
                .filter(|h| h.t > last - late - 1e-9 && h.t <= last + 1e-9)
                .map(|h| (h.t, h.hit))
                .collect()
        };
        let sang = overlap(&singing);
        assert!(sang.len() >= 14 && sang.iter().all(|h| h.1), "{sang:?}");
        let mut cursor = crate::core::singing::Cursor::default();
        let _ = singing.moments_since(&mut cursor);
        // The microphone as MicSession::sing hears it, silent: a frame
        // caught as the song is heard at `at`, each window frame followed.
        let mut at = last;
        while at <= 2.9 {
            demo.follow(at, late);
            demo.microphone((&mut singing, &mut estimate), (at, late), None);
            at += HOP;
        }
        assert!(
            !singing.moments_since(&mut cursor).sought,
            "nothing taken back"
        );
        assert_eq!(overlap(&singing), sang, "the demo's frames stand");
        // The first word scored hit, the streak as it was.
        let now = singing.tally;
        assert_eq!(
            (now.hit_words, now.scored_words, now.streak),
            (tally.hit_words, tally.scored_words, tally.streak)
        );
        assert!(popped(&singing), "the word stays hit");
        assert!(singing.last_heard().unwrap() > 2.7, "the microphone heard");
        // Held and let go of again, then the song sought back: the
        // microphone heard from there at once.
        demo.hold();
        frames(&mut demo, &mut singing, 2.9, 3.2);
        demo.let_go();
        demo.follow(1.0, late);
        demo.microphone((&mut singing, &mut estimate), (1.0, late), None);
        assert_eq!(
            singing.last_heard(),
            Some(1.0 - late),
            "a seek back is heard"
        );
    }

    #[test]
    fn nothing_is_sung_off_the_notes_and_nothing_without_the_key() {
        let mut singing = a_line();
        let mut demo = Demo::default();
        frames(&mut demo, &mut singing, 0.0, 3.0);
        assert_eq!(singing.last_heard(), None);
        assert!(!singing.tally.demo);
        // Held before the first word: frames, none voiced.
        demo.hold();
        frames(&mut demo, &mut singing, 0.0, 0.9);
        assert!(singing.last_heard().is_some() && singing.heard.is_empty());
    }
}
