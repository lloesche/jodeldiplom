//! How the singing is judged: a frame against its note (in the singer's
//! own key where the level allows, [`SungKey`]; [`Level::judge`]), and a
//! word by how much of its note time was hit ([`Level::credit`]). The
//! scoring and the delay estimated from the singing
//! ([`crate::core::lag::estimate`]) both judge by these.

use std::collections::VecDeque;

use super::{Hit, Level};

/// `d` (hundredths of a semitone) brought within half an octave of 0, a
/// whole number of octaves taken off.
fn fold(d: f64) -> f64 {
    (d + 600.0).rem_euclid(1200.0) - 600.0
}

/// The frames of the singer's pitch the key they sing in is told from (the
/// last 3 s of them on a note).
pub const KEY_FRAMES: usize = 300;

/// How far the singer sings from the song, octaves aside: followed frame
/// by frame on the notes, the median of the last [`KEY_FRAMES`] (each
/// taken near the median before, so a frame an octave out does not pull).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SungKey {
    /// The last frames' distances, in the order sung, and sorted.
    recent: VecDeque<f64>,
    sorted: Vec<f64>,
    /// Their median.
    median: Option<f64>,
}

impl SungKey {
    /// A frame sung at `cents` on a note at `note` (hundredths of a MIDI
    /// note); what it did, to take it back ([`SungKey::undo`]).
    pub fn hear(&mut self, cents: f64, note: f64) -> KeyStep {
        let mut step = KeyStep {
            dropped: None,
            median: self.median,
        };
        let from = fold(cents - note);
        let near = self.median.map_or(from, |k| k + fold(from - k));
        self.recent.push_back(near);
        self.insert(near);
        if self.recent.len() > KEY_FRAMES {
            step.dropped = self.recent.pop_front();
            if let Some(old) = step.dropped {
                let at = self.sorted.partition_point(|v| v.total_cmp(&old).is_lt());
                self.sorted.remove(at);
            }
        }
        self.median = Some(self.sorted[self.sorted.len() / 2]);
        step
    }

    fn insert(&mut self, v: f64) {
        let at = self.sorted.partition_point(|s| s.total_cmp(&v).is_lt());
        self.sorted.insert(at, v);
    }

    /// The last frame heard taken back: `step`, what it did.
    pub fn undo(&mut self, step: KeyStep) {
        if let Some(v) = self.recent.pop_back() {
            let at = self.sorted.partition_point(|s| s.total_cmp(&v).is_lt());
            self.sorted.remove(at);
        }
        if let Some(old) = step.dropped {
            self.recent.push_front(old);
            self.insert(old);
        }
        self.median = step.median;
    }

    /// How far a frame sung at `cents` is from its note at `note`, in
    /// hundredths of a semitone, at `level`: in the singer's own key where
    /// the level lets them sing in it, octaves aside.
    pub fn off(&self, level: Level, cents: f64, note: f64) -> f64 {
        let shift = if level.any_key() {
            self.median.unwrap_or(0.0)
        } else {
            0.0
        };
        fold(cents - shift - note)
    }

    /// Nothing heard yet.
    pub fn clear(&mut self) {
        *self = SungKey::default();
    }
}

/// What a frame did to the sung key: the frame it let go (the oldest), and
/// the median before it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeyStep {
    dropped: Option<f64>,
    median: Option<f64>,
}

/// What a word got of what its level asks: its frames that hit, and those
/// with the half hits, each as a share of the frames it needs to hit (1:
/// enough).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Credit {
    full: f64,
    with_half: f64,
}

impl Credit {
    /// How the word scores: hit when its hits alone are enough, half hit
    /// when they are with its half hits, else missed.
    pub fn hit(self) -> Option<Hit> {
        if self.full >= 1.0 {
            Some(Hit::Full)
        } else if self.with_half >= 1.0 {
            Some(Hit::Half)
        } else {
            None
        }
    }

    /// The same made smooth, 0 to 1: a word hit is 1, one half hit a half
    /// (as their points), one on its way there as far as it got.
    pub fn smooth(self) -> f64 {
        0.5 * (self.full.min(1.0) + self.with_half.min(1.0))
    }
}

impl Level {
    /// The share of a word's note time to hit for it to count.
    pub fn share(self) -> f64 {
        match self {
            Level::Easy => 0.10,
            Level::Medium => 0.25,
            Level::Hard => 0.40,
        }
    }

    /// How a frame `off` hundredths of a semitone from its note hits it
    /// (none: it misses).
    pub fn judge(self, off: f64) -> Option<Hit> {
        if off.abs() <= self.tolerance() {
            Some(Hit::Full)
        } else if self.near().is_some_and(|near| off.abs() <= near) {
            Some(Hit::Half)
        } else {
            None
        }
    }

    /// What a word got: `noted` frames on its notes, `hit` of them hit and
    /// `half` half hit, against the level's [`share`](Level::share).
    pub fn credit(self, noted: f64, hit: f64, half: f64) -> Credit {
        let needed = self.share() * noted;
        if needed <= 0.0 {
            return Credit {
                full: 0.0,
                with_half: 0.0,
            };
        }
        Credit {
            full: hit / needed,
            with_half: (hit + half) / needed,
        }
    }
}
