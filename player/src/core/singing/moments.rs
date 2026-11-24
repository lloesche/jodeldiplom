//! The singing's moments: one stream of what happened, as plain data for
//! whatever follows the game (the sounds, the stage, the result). Each
//! word scored, a bubble bursting and its liquid splashing out, the
//! streak up a multiplier or lost, each phrase's tier decided; in the
//! order they were decided, each at the time of the singing it happens
//! ([`Moment::at`]: a bubble bursts a sweep after it is full) and the frame
//! that decided it ([`Moment::decided`]). Sought back, those decided from
//! there on are taken back (and decided again as the frames come again);
//! the singing started over, all go. A reader follows the stream with a
//! [`Cursor`] ([`super::Singing::moments_since`]), told when what it read
//! was taken back.

use crate::core::shine::Shone;

/// What happened.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cue {
    /// A word scored: when it ended, whether it was hit, the streak after.
    Word {
        /// When it ended, seconds into the song.
        end: f64,
        /// Whether it was hit.
        hit: bool,
        /// The streak after it.
        streak: u32,
    },
    /// A note's bubble bursts (full, swept to full): its line and note,
    /// how high its note is (-1 to 1) and where along its line it is (0
    /// its start, 1 its end), and the bubbles bursting with it (full on
    /// the same frame): its place among them, in their order on the lines,
    /// and how many there are.
    Pop {
        /// Its line.
        line: usize,
        /// Its note on the line.
        note: usize,
        /// How high.
        height: f32,
        /// Where along the line.
        across: f32,
        /// Its place among the bubbles bursting together, and their count.
        burst: (usize, usize),
    },
    /// Its liquid splashing out as it tears, a bulge after it bursts
    /// ([`crate::core::bubbles::SWELL`]); as [`Cue::Pop`].
    Splash {
        /// Its line.
        line: usize,
        /// Its note on the line.
        note: usize,
        /// How high.
        height: f32,
        /// Where along the line.
        across: f32,
        /// Its place among the bubbles bursting together, and their count.
        burst: (usize, usize),
    },
    /// The streak up to a level (1: ×2, up to 3: ×4).
    StreakUp(u32),
    /// The streak lost, from a level of 1 or more.
    StreakLost,
    /// A phrase's shine and tier decided.
    Phrase(Shone),
}

/// A moment: when it happens (the singing's time, seconds), the frame
/// that decided it (the singing's time too), and what.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Moment {
    /// When it happens.
    pub at: f64,
    /// When it was decided.
    pub decided: f64,
    /// What.
    pub cue: Cue,
}

/// Where a reader is in the stream: what it read up to, of which run of
/// it (a run ends where moments it may have read were taken back).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cursor {
    run: Option<u64>,
    at: usize,
}

/// What a reader gets: the moments new to it, and whether what it read
/// before was (in part) taken back (sought back, started over; or the
/// first read: it starts from now).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Since<'a> {
    /// The new moments.
    pub moments: &'a [Moment],
    /// Whether the stream went back since the last read.
    pub sought: bool,
}

/// The stream as [`super::Singing`] keeps it.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Log {
    moments: Vec<Moment>,
    /// The run (one more each time moments are taken back), and the last
    /// cuts: the run each began, and how many moments were kept then.
    run: u64,
    cuts: std::collections::VecDeque<(u64, usize)>,
}

/// The cuts remembered: a reader further behind starts over from none.
const CUTS: usize = 64;

impl Log {
    pub(super) fn push(&mut self, at: f64, decided: f64, cue: Cue) {
        self.moments.push(Moment { at, decided, cue });
    }

    pub(super) fn all(&self) -> &[Moment] {
        &self.moments
    }

    fn cut(&mut self, keep: usize) {
        self.moments.truncate(keep);
        self.run += 1;
        self.cuts.push_back((self.run, keep));
        if self.cuts.len() > CUTS {
            self.cuts.pop_front();
        }
    }

    /// All of it gone (the singing started over).
    pub(super) fn clear(&mut self) {
        self.cut(0);
    }

    /// Those decided at `from` or later taken back.
    pub(super) fn take_back(&mut self, from: f64) {
        let keep = self.moments.partition_point(|m| m.decided < from);
        if keep < self.moments.len() {
            self.cut(keep);
        }
    }

    /// The moments new to `cursor`, which moves past them.
    pub(super) fn since(&self, cursor: &mut Cursor) -> Since<'_> {
        let sought = cursor.run != Some(self.run);
        if sought {
            // A first read starts from now; one of an earlier run from the
            // least kept by the cuts since (none remembered: from none).
            cursor.at = match cursor.run {
                None => self.moments.len(),
                Some(r) => {
                    let kept = self.cuts.iter().filter(|c| c.0 > r).map(|c| c.1).min();
                    let known = self.cuts.front().is_some_and(|c| c.0 <= r + 1);
                    if known {
                        cursor.at.min(kept.unwrap_or(cursor.at))
                    } else {
                        0
                    }
                }
            };
            cursor.run = Some(self.run);
        }
        let from = cursor.at.min(self.moments.len());
        cursor.at = self.moments.len();
        Since {
            moments: &self.moments[from..],
            sought,
        }
    }
}
