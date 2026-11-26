//! The demo key ([`Demo`]): while it is held in Karaoke mode, the game
//! hears a perfect singer in place of the microphone, to show it off
//! without singing. Each of its frames (a [`HOP`] apart, as the pitch
//! tracker's) is the note the bubbles are made of at its time, a few cents
//! of slow wobble on it so the arrow lives, and nothing where no note is
//! due. They come at the singing's own time (the song heard less how late
//! the voice comes: [`crate::core::lag::sung_clock`]), as a singer's on
//! time would, and go through [`Singing::hear`] as the microphone's do:
//! the bubbles, the streak, the shine and the result follow as for
//! singing. Let go of, it stops at once: no frame of it after.
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

/// The demo key: held or not, and where its singing got to.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Demo {
    held: bool,
    /// When its last frame was, in the singing's time.
    last: Option<f64>,
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

    /// The key let go of: the demo stops at once.
    pub fn let_go(&mut self) {
        self.held = false;
        self.last = None;
    }

    /// While held, the perfect singer's frames after its last up to `to`
    /// (the singing's time) heard by `singing`, the song marked as the
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
