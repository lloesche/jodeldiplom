//! What the last frames heard did to the singing, kept a while
//! ([`UNDO_FOR`]) so that a short seek back (the delay nudged mid-song,
//! the song sought back a little) takes it back and hears them again,
//! rather than counting them twice ([`super::Singing::hear`]).

use std::collections::VecDeque;

use super::judge::KeyStep;
use super::{Hit, Tally};

/// How far back what the frames did is kept, in seconds: more than a seek
/// back that is taken back rather than started over.
pub const UNDO_FOR: f64 = 1.5;

/// What a frame did: when it was, the word it was on and how it hit it,
/// the sung key as it moved it, and the scoring before it.
#[derive(Clone, Debug)]
pub struct Undo {
    pub t: f64,
    pub word: Option<((usize, usize), Option<Hit>)>,
    pub key: Option<KeyStep>,
    pub tally: Tally,
}

/// The frames' undos, in time.
#[derive(Clone, Debug, Default)]
pub struct Log(VecDeque<Undo>);

impl Log {
    /// What a frame did, kept; those older than [`UNDO_FOR`] before it let
    /// go.
    pub fn push(&mut self, undo: Undo) {
        let t = undo.t;
        self.0.push_back(undo);
        while self.0.front().is_some_and(|u| u.t < t - UNDO_FOR) {
            self.0.pop_front();
        }
    }

    /// Nothing kept.
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// What the frames from `from` on did, taken out of the log, the latest
    /// first (to be taken back in that order); none (at once) where the
    /// last frame was before it, as it is but for a seek back.
    pub fn take_from(&mut self, from: f64) -> Vec<Undo> {
        if self.0.back().is_none_or(|u| u.t < from) {
            return Vec::new();
        }
        let first = self.0.partition_point(|u| u.t < from);
        self.0.drain(first..).rev().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use std::path::Path;

    /// Ten words a second each, on C4, E4, G4, D4 and A4 in turn.
    fn song() -> Singing {
        const PITCH: [u16; 5] = [6000, 6400, 6700, 6200, 6900];
        let mut m = Melody {
            cents: vec![0; 1100],
            sure: vec![0; 1100],
        };
        for i in 0..1000 {
            m.cents[i] = PITCH[(i / 100) % 5];
            m.sure[i] = 255;
        }
        let words = (0..10)
            .map(|k| jodel_lyrics::Word {
                start: Some(k as f64),
                end: Some(k as f64 + 1.0),
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
        let mut s = Singing::default();
        s.prepare(Path::new("a"), &lyrics, &m, 1, Level::Medium);
        s
    }

    /// The pitch sung at `t` on pass `p`: the melody, off by a pass's own
    /// amount on some frames, none on others (each pass sung otherwise).
    fn sung(s: &Singing, t: f64, p: usize) -> Option<f32> {
        let i = (t * 100.0).round() as usize;
        let off = [0.0, 220.0, -260.0, 140.0, 40.0][p % 5];
        let quiet = (i * (p + 3)).is_multiple_of(7);
        s.word_at(t)
            .map(|(_, c)| (c + off) as f32)
            .filter(|_| !quiet)
    }

    /// `passes` (from, to, in seconds) heard one after another, each frame
    /// every 10 ms: each from before the last frame (sought back a little,
    /// the delay nudged), heard again sung otherwise. The same as the
    /// frames as they ended up heard straight through: the scoring, the
    /// counts, the frames kept and the sung key.
    fn same_as_straight(passes: &[(f64, f64)]) {
        let mut s = song();
        // The pass each frame was last heard on.
        let mut last: Vec<Option<usize>> = vec![None; 1100];
        for (p, &(from, to)) in passes.iter().enumerate() {
            let (a, b) = (
                (from * 100.0).round() as usize,
                (to * 100.0).round() as usize,
            );
            for (i, slot) in last.iter_mut().enumerate().take(b + 1).skip(a) {
                let t = i as f64 / 100.0;
                s.hear(t, sung(&s, t, p));
                *slot = Some(p);
            }
            for slot in last.iter_mut().skip(b + 1) {
                *slot = None;
            }
        }
        let mut straight = song();
        for (i, p) in last.iter().enumerate() {
            if let Some(p) = p {
                let t = i as f64 / 100.0;
                straight.hear(t, sung(&straight, t, *p));
            }
        }
        assert_eq!(s.tally, straight.tally, "{passes:?}");
        assert_eq!(s.frames, straight.frames, "{passes:?}");
        assert_eq!(s.heard, straight.heard, "{passes:?}");
        assert_eq!(s.sung_key, straight.sung_key, "{passes:?}");
    }

    #[test]
    fn short_seeks_back_in_a_row_are_as_heard_straight() {
        same_as_straight(&[(0.0, 2.0), (1.6, 2.2), (1.9, 2.4), (2.3, 3.5), (3.1, 6.0)]);
    }

    /// The delay nudged mid-song: the frames' times step back by the
    /// nudge (up to its most, 400 ms), and on again.
    #[test]
    fn a_delay_nudged_mid_song_is_as_heard_straight() {
        same_as_straight(&[(0.0, 4.2), (3.8, 5.0), (4.95, 7.4), (7.0, 9.5)]);
        // Over three hundred frames, the sung key letting go of the oldest:
        // taken back too.
        same_as_straight(&[(0.0, 8.0), (7.2, 9.0)]);
    }
}
