//! The bubbles bursting, as moments ([`Cue::Pop`], [`Cue::Splash`]), and
//! the shots that rescue a single word missed ([`Cue::Shot`], [`Rescue`]).
//!
//! A word scored missed while the streak goes on waits
//! ([`super::tally::Run::Waiting`]). It is rescued only by the very next
//! of the song's words (by when they end) bursting, not itself rescued
//! ([`Singing::shooter`]): as that bubble bursts, or as the word missed is
//! scored if it burst already. The shot is fired then (a moment, at the
//! burst or the scoring, whichever is later), the word scored as if sung,
//! the streak going on. It lands [`FLIGHT`] later ([`Rescue::lands`]): the
//! word fills ([`crate::core::bubbles`] has it full then) and bursts as any
//! bubble. A word passed unheard (sought past) ends the wait: nothing is
//! rescued over it ([`super::tally::Tally::score_word`]).
//!
//! What the moments tell of the bubbles (those burst, the words rescued)
//! is indexed from the moments themselves ([`Index`]), as they are
//! logged.

use std::collections::HashMap;
use std::ops::Range;

use super::moments::{Burst, Cue, Moment};
use super::tally::Run;
use super::{Singing, SCORED_AFTER};
use crate::core::bubbles::{self, FLIGHT, SWEEP, SWELL};

/// A word missed and rescued by a shot: the bubble that fired it (its
/// line and word), and when it was fired (the singing's time).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rescue {
    /// The bubble that fired it.
    pub from: (usize, usize),
    /// When it was fired.
    pub fired: f64,
}

impl Rescue {
    /// When its shot lands (the singing's time).
    pub fn lands(&self) -> f64 {
        self.fired + FLIGHT
    }
}

/// What the moments tell of the bubbles, by line and word: those burst
/// (when each bursts, and how) and the words rescued. Made only from the
/// moments ([`Index::add`]), as they are logged and again after a seek
/// back.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Index {
    pub popped: HashMap<(usize, usize), (f64, Burst)>,
    pub rescues: HashMap<(usize, usize), Rescue>,
}

impl Index {
    /// Moment `m` logged.
    pub(super) fn add(&mut self, m: &Moment) {
        match m.cue {
            Cue::Pop(b) => {
                self.popped.insert((b.line, b.word), (m.at, b));
            }
            Cue::Shot { from, to, .. } => {
                let fired = m.at;
                self.rescues.insert(to, Rescue { from, fired });
            }
            _ => {}
        }
    }
}

impl Singing {
    /// The bubbles of the lines about `t` (and those shots rescued) whose
    /// time to burst the frames up to `t` fixed: each a moment as it
    /// bursts and as it splashes out, the bubbles full on the same frame
    /// bursting together; each rescuing the word waiting before it, if any
    /// (and that word then bursting in turn).
    pub(super) fn pops(&mut self, t: f64) {
        loop {
            let new = self.bursting(t);
            if new.is_empty() {
                return;
            }
            let mut k = 0;
            while k < new.len() {
                let together = new[k..].iter().take_while(|b| b.0 == new[k].0).count();
                for (j, &(full, b)) in new[k..k + together].iter().enumerate() {
                    let at = full + SWEEP;
                    let b = Burst {
                        burst: (j, together),
                        ..b
                    };
                    self.log.push(at, t, Cue::Pop(b));
                    self.log.push(at + SWELL, t, Cue::Splash(b));
                    self.rescue_waiting(t);
                }
                k += together;
            }
        }
    }

    /// The bubbles full and not burst yet, of the lines about `t` and
    /// those with a word rescued: when each filled, and its burst (its
    /// place among those bursting together to be told), in order.
    fn bursting(&self, t: f64) -> Vec<(f64, Burst)> {
        let index = self.log.index();
        let rescued: Vec<usize> = (index.rescues.keys())
            .filter(|w| !index.popped.contains_key(w))
            .map(|w| w.0)
            .collect();
        let mut new = Vec::new();
        for (l, ln) in self.notes.lines.iter().enumerate() {
            let (Some(first), Some(last)) = (ln.notes.first(), ln.notes.last()) else {
                continue;
            };
            let near = first.start <= t && last.end + SCORED_AFTER + 0.1 >= t;
            if !near && !rescued.contains(&l) {
                continue;
            }
            for b in bubbles::line(self, l, f64::INFINITY).words {
                let Some(full) = b.full_at else { continue };
                if index.popped.contains_key(&(l, b.word)) {
                    continue;
                }
                let burst = Burst {
                    line: l,
                    word: b.word,
                    height: ln.word_height(b.notes.clone()),
                    across: self.across(l, b.notes),
                    burst: (0, 1),
                };
                new.push((full, burst));
            }
        }
        new.sort_by(|a, b| {
            (a.0.total_cmp(&b.0))
                .then(a.1.line.cmp(&b.1.line))
                .then(a.1.word.cmp(&b.1.word))
        });
        new
    }

    /// Where the notes `notes` of line `l` are along it: 0 its start, 1 its
    /// end.
    fn across(&self, l: usize, notes: Range<usize>) -> f32 {
        let Some(ln) = self.notes.lines.get(l) else {
            return 0.5;
        };
        let line_end = ln.notes.last().map_or(1.0, |n| n.to.max(1.0));
        let last = notes.end.checked_sub(1).and_then(|k| ln.notes.get(k));
        let (Some(first), Some(last)) = (ln.notes.get(notes.start), last) else {
            return 0.5;
        };
        (0.5 * (first.from + last.to) / line_end).clamp(0.0, 1.0)
    }

    /// The bubble that rescues the word waiting at `waiting` (its place
    /// among the song's words), if there is one yet: the very next word's,
    /// burst, not itself rescued; when it bursts, and its burst.
    fn shooter(&self, waiting: usize) -> Option<(f64, Burst)> {
        let &(l, w, _) = self.notes.words().get(waiting + 1)?;
        let index = self.log.index();
        let popped = *index.popped.get(&(l, w))?;
        (!index.rescues.contains_key(&(l, w))).then_some(popped)
    }

    /// The word waiting, if any, rescued by its shooter if it has one yet
    /// (seen at `t`): scored as if sung (a rung up of the streak with it a
    /// moment too, as the shot is fired), the shot a moment, fired as the
    /// shooter bursts or now, whichever is later.
    pub(super) fn rescue_waiting(&mut self, t: f64) {
        let Run::Waiting(waiting) = self.tally.run else {
            return;
        };
        let (Some(&(l, w, _)), Some((at, from))) =
            (self.notes.words().get(waiting), self.shooter(waiting))
        else {
            return;
        };
        let fired = at.max(t);
        let before = self.rung();
        self.tally.rescue();
        let shot = Cue::Shot {
            from: (from.line, from.word),
            to: (l, w),
            across: from.across,
        };
        self.log.push(fired, t, shot);
        if let Some(up) = self.rung().filter(|&now| Some(now) > before) {
            self.log.push(fired, t, Cue::StreakUp(up));
        }
    }
}
