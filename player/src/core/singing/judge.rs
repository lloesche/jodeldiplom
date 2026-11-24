//! How the singing is judged: a frame against its note, in the singer's
//! own key ([`SungKey`]), hits within [`TOLERANCE`]; a word scores when its
//! frames that hit are [`SHARE`] of its note time ([`credit`]).

use std::collections::VecDeque;

/// How far from a note a frame may be and still hit, in hundredths of a
/// semitone: the shape of the melody, in any key, within 3 semitones.
pub const TOLERANCE: f64 = 300.0;

/// The share of a word's note time to hit for it to count: a tenth (a
/// pitch is heard on only about half of a word sung well, its consonants
/// and breaths having none).
pub const SHARE: f64 = 0.10;

/// `d` (hundredths of a semitone) brought within half an octave of 0, a
/// whole number of octaves taken off.
pub fn fold(d: f64) -> f64 {
    (d + 600.0).rem_euclid(1200.0) - 600.0
}

/// Whether a frame `off` hundredths of a semitone from its note hits it.
pub fn hits(off: f64) -> bool {
    off.abs() <= TOLERANCE
}

/// What a word got of what it needs: its `hit` frames as a share of the
/// [`SHARE`] of its `noted` frames it needs (1: enough; 0 with none
/// noted).
pub fn credit(noted: f64, hit: f64) -> f64 {
    let needed = SHARE * noted;
    if needed <= 0.0 {
        0.0
    } else {
        hit / needed
    }
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

    /// How far the singer sings from the song now (none before a frame
    /// on a note), in hundredths of a semitone.
    pub fn median(&self) -> Option<f64> {
        self.median
    }

    /// How far a frame sung at `cents` is from its note at `note`, in
    /// hundredths of a semitone: in the singer's own key, octaves aside.
    pub fn off(&self, cents: f64, note: f64) -> f64 {
        fold(cents - self.median.unwrap_or(0.0) - note)
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
