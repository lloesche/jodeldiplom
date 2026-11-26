//! The singing's moments (the game's event stream): one stream of what
//! happened, as plain data for
//! whatever follows the game (the sounds, the stage, the result). Each
//! word scored, a bubble bursting and its liquid splashing out, a shot
//! rescuing a word missed, the streak up a multiplier or lost, each
//! phrase's tier decided; in the order they were decided, each at the
//! time of the singing it happens
//! ([`Moment::at`]: a bubble bursts a sweep after it is full, or as its
//! word ends if that is later) and the frame
//! that decided it ([`Moment::decided`]). Sought back, those decided from
//! there on are taken back (and decided again as the frames come again);
//! the singing started over, all go. A reader follows the stream with a
//! [`Cursor`] ([`super::Singing::moments_since`]), told when what it read
//! was taken back.

use super::bursts::Index;
use crate::core::shine::{Shone, Tier};

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
    /// A word's bubble bursts (full, swept to full, its word over: held
    /// full until then, [`Burst::held`]): its line and word, how high its
    /// notes are (-1 to 1, their mean over their time) and where along its
    /// line it is (0 its start, 1 its end), the bubbles bursting with it
    /// (at the same time): its place among them, in their order on the
    /// lines, and how many there are. One a shot rescued has that shot
    /// before it ([`Cue::Shot`] to it).
    Pop(Burst),
    /// Its liquid splashing out as it tears, a bulge after it bursts
    /// ([`crate::core::bubbles::SWELL`]); as [`Cue::Pop`].
    Splash(Burst),
    /// A shot fired by a bubble bursting (`from`: its line and word) at
    /// the single word missed before it (`to`), rescuing it: it lands
    /// [`crate::core::bubbles::FLIGHT`] after it is fired (the moment's
    /// time), the word filling and bursting as if sung; `across` where the
    /// bubble firing it is along its line (0 to 1).
    Shot {
        /// The bubble firing it: line, word.
        from: (usize, usize),
        /// The bubble it rescues: line, word.
        to: (usize, usize),
        /// Where the bubble firing it is along its line.
        across: f32,
    },
    /// The streak up to a rung of the multiplier, the rung named after its
    /// tier ([`crate::core::singing::rung`]).
    StreakUp(Tier),
    /// The streak lost, from the rung it was at.
    StreakLost(Tier),
    /// A phrase's shine and tier decided.
    Phrase(Shone),
}

/// A bubble bursting, as [`Cue::Pop`] and [`Cue::Splash`] tell it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Burst {
    /// Its line.
    pub line: usize,
    /// Its word on the line.
    pub word: usize,
    /// How high.
    pub height: f32,
    /// Where along the line.
    pub across: f32,
    /// Its place among the bubbles bursting together, and their count.
    pub burst: (usize, usize),
    /// The words hit in a row with it as it bursts, in the order they
    /// burst: the streak as scored, and the words won and not scored yet,
    /// those bursting by then (it among them), less the words rescued
    /// bursting after it (what a pop climbs a ladder by).
    pub run: u32,
    /// How long it was held full before it burst, in seconds (0: its word
    /// over as it was swept full).
    pub held: f64,
}

impl Burst {
    /// How big it bursts, 0 (as any) to 1 ([`crate::core::bubbles::burst_size`]).
    pub fn size(&self) -> f64 {
        crate::core::bubbles::burst_size(self.held)
    }
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

/// The stream as [`super::Singing`] keeps it, and what it tells of the
/// bubbles ([`Index`]): the one place either is written.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Log {
    moments: Vec<Moment>,
    index: Index,
    /// The run (one more each time moments are taken back), and the last
    /// cuts: the run each began, and how many moments were kept then.
    run: u64,
    cuts: std::collections::VecDeque<(u64, usize)>,
}

/// The cuts remembered: a reader further behind starts over from none.
const CUTS: usize = 64;

impl Log {
    pub(super) fn push(&mut self, at: f64, decided: f64, cue: Cue) {
        let moment = Moment { at, decided, cue };
        self.index.add(&moment);
        self.moments.push(moment);
    }

    pub(super) fn all(&self) -> &[Moment] {
        &self.moments
    }

    /// What its moments tell of the bubbles.
    pub(super) fn index(&self) -> &Index {
        &self.index
    }

    fn cut(&mut self, keep: usize) {
        self.moments.truncate(keep);
        self.index = Index::default();
        for m in &self.moments {
            self.index.add(m);
        }
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
