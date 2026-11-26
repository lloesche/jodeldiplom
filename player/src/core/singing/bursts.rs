//! The bubbles bursting, as moments ([`Cue::Pop`], [`Cue::Splash`]), and
//! the shots that rescue a single word missed ([`Cue::Shot`], [`Rescue`]).
//!
//! A bubble full bursts once it is swept full and its word is over
//! ([`crate::core::bubbles::Bubble::bursts_at`]): a word won while still
//! sung is held full until it ends. Its burst is decided as soon as it is
//! full, or, held, a sweep before it bursts: whatever follows it hears of
//! it as far ahead as of any, and the words bursting before it are known
//! by then (its run, [`Burst::run`]).
//!
//! A word scored missed while the streak goes on waits
//! ([`super::tally::Run::Waiting`]). It is rescued only by the very next
//! of the song's words (by when they end) being won (swept full, held or
//! burst), not itself rescued ([`Singing::shooter`]): as that bubble is
//! swept full, or as the word missed is scored if it was already. The shot
//! is fired then (a moment, at the sweep or the scoring, whichever is
//! later), the word scored as if sung, the streak going on. It lands
//! [`FLIGHT`] later ([`Rescue::lands`]): the word fills
//! ([`crate::core::bubbles`] has it full then) and bursts as any bubble. A
//! word passed unheard (sought past) ends the wait: nothing is rescued
//! over it ([`super::tally::Tally::score_word`]).
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

/// A bubble full and not burst yet: when it bursts, and its burst (its
/// run and its place among those bursting together to be told).
#[derive(Clone, Copy, Debug)]
struct Full {
    at: f64,
    burst: Burst,
}

impl Full {
    /// Whether its burst is decided at `t` (the singing's time): at once,
    /// or held full, a sweep before it bursts.
    fn due(&self, t: f64) -> bool {
        self.burst.held <= 0.0 || t >= self.at - SWEEP
    }
}

impl Singing {
    /// The word waiting rescued if its shooter is won, then the bubbles of
    /// the lines about `t` (and those shots rescued) whose burst the
    /// frames up to `t` decide: each a moment as it bursts and as it
    /// splashes out, those bursting at the same time together.
    pub(super) fn pops(&mut self, t: f64) {
        self.rescue_waiting(t);
        let full = self.bursting(t);
        let due: Vec<Full> = full.iter().filter(|f| f.due(t)).copied().collect();
        let mut k = 0;
        while k < due.len() {
            let together = due[k..].iter().take_while(|f| f.at == due[k].at).count();
            for (j, f) in due[k..k + together].iter().enumerate() {
                let b = Burst {
                    burst: (j, together),
                    run: self.run_with(f, &full),
                    ..f.burst
                };
                self.log.push(f.at, t, Cue::Pop(b));
                self.log.push(f.at + SWELL, t, Cue::Splash(b));
            }
            k += together;
        }
    }

    /// The bubbles full and not burst yet, of the lines about `t` and
    /// those with a word rescued: when each bursts, and its burst (its run
    /// and its place among those bursting together to be told), in order.
    fn bursting(&self, t: f64) -> Vec<Full> {
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
                let Some(at) = b.bursts_at() else { continue };
                if index.popped.contains_key(&(l, b.word)) {
                    continue;
                }
                let burst = Burst {
                    line: l,
                    word: b.word,
                    height: ln.word_height(b.notes.clone()),
                    across: self.across(l, b.notes.clone()),
                    burst: (0, 1),
                    run: 0,
                    held: b.held(),
                };
                new.push(Full { at, burst });
            }
        }
        new.sort_by(|a, b| {
            (a.at.total_cmp(&b.at))
                .then(a.burst.line.cmp(&b.burst.line))
                .then(a.burst.word.cmp(&b.burst.word))
        });
        new
    }

    /// The words hit in a row with bubble `x` as it bursts, in the order
    /// they burst (`full`: those full and not burst yet): the streak as
    /// scored, and the words won and not scored yet bursting by then, it
    /// among them; less the words rescued (in the streak) bursting after
    /// it.
    fn run_with(&self, x: &Full, full: &[Full]) -> u32 {
        let index = self.log.index();
        let me = (x.at, x.burst.line, x.burst.word);
        let bursts = |w: (usize, usize)| {
            (index.popped.get(&w).map(|p| p.0)).or_else(|| {
                (full.iter())
                    .find(|f| (f.burst.line, f.burst.word) == w)
                    .map(|f| f.at)
            })
        };
        let by_then = |w: (usize, usize)| bursts(w).is_some_and(|at| (at, w.0, w.1) <= me);
        let words = self.notes.words();
        let ahead = (words.get(self.tally.next_word..).unwrap_or(&[]).iter())
            .filter(|&&(l, w, _)| by_then((l, w)))
            .count();
        let after = (index.rescues.keys()).filter(|&&w| !by_then(w)).count();
        (self.tally.streak + ahead as u32).saturating_sub(after as u32)
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
    /// won (swept full: held or burst), not itself rescued; its line and
    /// word, when it was swept full, and where along its line it is.
    fn shooter(&self, waiting: usize) -> Option<((usize, usize), f64, f32)> {
        let &(l, w, _) = self.notes.words().get(waiting + 1)?;
        let index = self.log.index();
        if index.rescues.contains_key(&(l, w)) {
            return None;
        }
        if let Some(&(at, b)) = index.popped.get(&(l, w)) {
            return Some(((l, w), at - b.held, b.across));
        }
        let line = bubbles::line(self, l, f64::INFINITY);
        let b = line.word(w)?;
        Some(((l, w), b.swept_at()?, self.across(l, b.notes.clone())))
    }

    /// The word waiting, if any, rescued by its shooter if it has one yet
    /// (seen at `t`): scored as if sung (a rung up of the streak with it a
    /// moment too, as the shot is fired), the shot a moment, fired as the
    /// shooter is swept full or now, whichever is later.
    pub(super) fn rescue_waiting(&mut self, t: f64) {
        let Run::Waiting(waiting) = self.tally.run else {
            return;
        };
        let (Some(&(l, w, _)), Some((from, won, across))) =
            (self.notes.words().get(waiting), self.shooter(waiting))
        else {
            return;
        };
        let fired = won.max(t);
        let before = self.rung();
        self.tally.rescue();
        let shot = Cue::Shot {
            from,
            to: (l, w),
            across,
        };
        self.log.push(fired, t, shot);
        if let Some(up) = self.rung().filter(|&now| Some(now) > before) {
            self.log.push(fired, t, Cue::StreakUp(up));
        }
    }
}
