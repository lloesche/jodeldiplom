//! The control word's timeline ([`jodel_output::lanes::Parts`]): what was
//! written where, and what the next frames say, given what parts there are.
//!
//! The one promise it keeps: once frames have said [`Parts::Present`], the
//! parts never stop at once. They end with [`Parts::Ending`] for a mixer's
//! switch's length (`e` frames, [`jodel_output::lanes::ending_frames`]),
//! with the parts valid all that time, and only then say [`Parts::None`]
//! (unless the stream has a gap: a flush, a new output). For that the parts
//! of the `e` frames after what is written are promised
//! ([`Timeline::promised`]): the buffer keeps them, whatever else changes.
//! On a track's last frames the parts may run to its end with nothing
//! after: the next track's carry on, or nothing follows. Should the next
//! track have none after all on its first frame (one that did not open was
//! skipped, its stems failed after the last look), the word goes straight
//! to [`Parts::None`] there: on the frame main itself cuts to a new track,
//! where a mixer snaps to main ([`jodel_output::lanes`] allows it).
//!
//! A change of word is logged at its frame; frames the output gives back
//! take their changes back with them ([`Timeline::given_back`]).

use std::collections::VecDeque;

use jodel_output::lanes::Parts;

/// The word in force before a frame, and since when for an end.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum State {
    Absent,
    Present,
    /// Ending from this frame on (for `e` frames from it).
    Ending(u64),
}

/// The parts there are from the frame written next.
#[derive(Clone, Copy, Debug)]
pub(super) struct Supply {
    /// They are valid from the frame written next up to this one.
    pub(super) there: u64,
    /// More are being made: absent ones may come.
    pub(super) coming: bool,
    /// Frames of them needed ahead before absent ones are said to be there.
    pub(super) hold: u64,
    /// They end with the track and nothing more is needed of them after
    /// (the next track's follow, or nothing does): once they reach main's
    /// end they may be said to be there up to it, with no switch's length
    /// after.
    pub(super) through: bool,
    /// The output frame after main's last, when all of it is made.
    pub(super) main_end: Option<u64>,
}

impl Supply {
    /// Whether the parts reach the track's end, with nothing more needed,
    /// for a stretch of main up to output frame `end`.
    fn reaches_the_end(self, end: u64) -> bool {
        self.through
            && self
                .main_end
                .is_some_and(|main| self.there >= main && end <= main)
    }
}

/// The words of a stretch of frames: the word in force at its first, and
/// the changes within it, in order.
#[derive(Clone, Debug, Default)]
pub(super) struct Plan {
    base: Option<Parts>,
    changes: Vec<(u64, Parts)>,
}

impl Plan {
    /// The word of frame `o` of the stretch.
    pub(super) fn word(&self, o: u64) -> Parts {
        self.changes
            .iter()
            .rev()
            .find(|(f, _)| *f <= o)
            .map_or(self.base.unwrap_or(Parts::None), |&(_, w)| w)
    }
}

/// The words written, as their changes (frame, word), and the frame after
/// the last written.
#[derive(Debug)]
pub(super) struct Timeline {
    log: VecDeque<(u64, Parts)>,
    written: u64,
    /// A mixer's switch's length, in frames.
    e: u64,
}

impl Timeline {
    pub(super) fn new(e: u64) -> Timeline {
        Timeline {
            log: VecDeque::new(),
            written: 0,
            e,
        }
    }

    /// The frame after the last written.
    pub(super) fn written(&self) -> u64 {
        self.written
    }

    /// The word in force before frame `from`.
    pub(super) fn state(&self, from: u64) -> State {
        match self.log.iter().rev().find(|(f, _)| *f < from) {
            Some(&(_, Parts::Present)) => State::Present,
            Some(&(f, Parts::Ending)) if from < f + self.e => State::Ending(f),
            _ => State::Absent,
        }
    }

    /// The frame from which the words before frame `from` have said the
    /// parts are absent, when they do (from the stream's start: 0); `None`
    /// while they are there or ending.
    pub(super) fn quiet_since(&self, from: u64) -> Option<u64> {
        match self.log.iter().rev().find(|(f, _)| *f < from) {
            None => Some(0),
            Some(&(f, Parts::None)) => Some(f),
            Some(&(f, Parts::Ending)) if from >= f + self.e => Some(f + self.e),
            _ => None,
        }
    }

    /// The frame up to which parts must stay valid past `from` (the frame
    /// written next): a switch's length while they are there, the rest of
    /// an end under way.
    pub(super) fn promised(&self, from: u64) -> u64 {
        match self.state(from) {
            State::Present => from + self.e,
            State::Ending(f) => f + self.e,
            State::Absent => from,
        }
    }

    /// The words of `n` frames from `from` (the frame written next), given
    /// `supply`.
    pub(super) fn plan(&self, from: u64, n: u64, supply: Supply) -> Plan {
        let e = self.e;
        let end = from + n;
        let state = self.state(from);
        let mut plan = Plan {
            base: Some(match state {
                State::Absent => Parts::None,
                State::Present => Parts::Present,
                State::Ending(_) => Parts::Ending,
            }),
            changes: Vec::new(),
        };
        let mut present = state == State::Present;
        if state == State::Absent
            && supply.coming
            && (supply.there >= from + supply.hold.max(n + e) || supply.reaches_the_end(end))
        {
            plan.changes.push((from, Parts::Present));
            present = true;
        }
        let mut ending = match state {
            State::Ending(f) => Some(f),
            _ => None,
        };
        if present {
            let enough = supply.there >= end + e || supply.reaches_the_end(end);
            if !enough {
                let f = from.max(supply.there.saturating_sub(e));
                plan.changes.push((f, Parts::Ending));
                ending = Some(f);
            }
        }
        if let Some(f) = ending {
            if f + e < end {
                plan.changes.push(((f + e).max(from), Parts::None));
            }
        }
        plan
    }

    /// `took` frames of `plan`'s stretch from `from` were written.
    pub(super) fn commit(&mut self, plan: &Plan, from: u64, took: u64) {
        let end = from + took;
        if took == 0 {
            return;
        }
        let mut last = self.log.back().map(|&(_, w)| w);
        for &(f, w) in plan.changes.iter().filter(|(f, _)| *f < end) {
            if last != Some(w) {
                self.log.push_back((f, w));
                last = Some(w);
            }
        }
        self.written = end;
    }

    /// The output gave back what was written from `from` on: so are the
    /// changes there.
    pub(super) fn given_back(&mut self, from: u64) {
        while self.log.back().is_some_and(|(f, _)| *f >= from) {
            self.log.pop_back();
        }
        self.written = self.written.min(from);
    }

    /// A gap in the stream (a flush, a new output): nothing written before
    /// it binds what comes after.
    pub(super) fn clear(&mut self, at: u64) {
        self.log.clear();
        self.written = at;
    }

    /// Forgets the changes before `before` but the one in force there
    /// (what [`Timeline::state`] of `before` reads).
    pub(super) fn trim(&mut self, before: u64) {
        while self.log.len() > 1 && self.log[1].0 < before {
            self.log.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const E: u64 = 10;

    fn supply(there: u64, coming: bool) -> Supply {
        Supply {
            there,
            coming,
            hold: 40,
            through: false,
            main_end: None,
        }
    }

    /// Writes `n` frames from `from` with `supply`, taking `took` of them;
    /// returns their words.
    fn write(t: &mut Timeline, from: u64, n: u64, took: u64, s: Supply) -> Vec<Parts> {
        let plan = t.plan(from, n, s);
        t.commit(&plan, from, took);
        (from..from + took).map(|o| plan.word(o)).collect()
    }

    /// Whether `words` keep the promise: no `None` right after `Present`,
    /// and at least `E` frames of `Ending` before a `None` after one.
    fn kept(words: &[Parts]) -> Result<(), String> {
        let mut run = 0;
        for (i, w) in words.iter().enumerate() {
            match (i.checked_sub(1).map(|j| words[j]), w) {
                (Some(Parts::Present), Parts::None) => {
                    return Err(format!("None after Present at {i}"))
                }
                (Some(Parts::Ending), Parts::None) if run < E => {
                    return Err(format!("None after {run} frames of Ending at {i}"))
                }
                _ => {}
            }
            run = if *w == Parts::Ending { run + 1 } else { 0 };
        }
        Ok(())
    }

    #[test]
    fn parts_come_once_far_enough_ahead_and_end_over_a_switch() {
        let mut t = Timeline::new(E);
        let mut words = Vec::new();
        // Too few ahead: none.
        words.extend(write(&mut t, 0, 8, 8, supply(30, true)));
        assert!(words.iter().all(|&w| w == Parts::None));
        // Far enough: there.
        words.extend(write(&mut t, 8, 8, 8, supply(60, true)));
        assert_eq!(words[8], Parts::Present);
        assert_eq!(t.promised(16), 16 + E);
        // They run out at 30 (promised to 26): end from 20, none from 30.
        words.extend(write(&mut t, 16, 8, 8, supply(30, true)));
        words.extend(write(&mut t, 24, 8, 8, supply(30, true)));
        assert_eq!(words[19], Parts::Present);
        assert_eq!(words[20], Parts::Ending);
        assert_eq!(words[30], Parts::None);
        kept(&words).unwrap();
        // Not again until far enough ahead again.
        words.extend(write(&mut t, 32, 8, 8, supply(60, true)));
        assert_eq!(words[32], Parts::None);
        words.extend(write(&mut t, 40, 8, 8, supply(90, true)));
        assert_eq!(words[40], Parts::Present);
        kept(&words).unwrap();
    }

    #[test]
    fn a_track_end_may_run_the_parts_to_its_last_frame() {
        let mut t = Timeline::new(E);
        write(&mut t, 0, 8, 8, supply(100, true));
        let through = Supply {
            through: true,
            main_end: Some(20),
            ..supply(20, true)
        };
        let words = write(&mut t, 8, 12, 12, through);
        assert!(words.iter().all(|&w| w == Parts::Present));
        // Main goes on past the parts: they end before.
        let mut t = Timeline::new(E);
        write(&mut t, 0, 8, 8, supply(100, true));
        let longer = Supply {
            main_end: Some(24),
            ..through
        };
        let words = write(&mut t, 8, 6, 6, longer);
        let words = [words, write(&mut t, 14, 10, 10, longer)].concat();
        assert_eq!((words[1], words[2]), (Parts::Present, Parts::Ending));
        assert_eq!(words[12], Parts::None);
        // The parts as far as main is, but main's end not known yet: not
        // to the end.
        let mut t = Timeline::new(E);
        write(&mut t, 0, 8, 8, supply(100, true));
        let unknown = Supply {
            main_end: None,
            ..through
        };
        let words = write(&mut t, 8, 12, 12, unknown);
        assert_eq!(words[2], Parts::Ending);
        // A change kept by a trim is still the one in force.
        let mut t = Timeline::new(E);
        write(&mut t, 0, 8, 8, supply(100, true));
        write(&mut t, 8, 8, 8, supply(100, true));
        t.trim(0);
        t.given_back(0);
        assert_eq!(t.state(0), State::Absent);
        let mut t = Timeline::new(E);
        write(&mut t, 0, 8, 8, supply(100, true));
        write(&mut t, 8, 8, 8, supply(14, false));
        assert_eq!(t.state(9), State::Ending(8));
        t.trim(8);
        assert_eq!(t.state(8), State::Present);
    }

    #[test]
    fn frames_given_back_take_their_words_back() {
        // During an end: given back to before it, the parts are there again.
        let mut t = Timeline::new(E);
        write(&mut t, 0, 8, 8, supply(100, true));
        write(&mut t, 8, 8, 8, supply(21, false));
        assert_eq!(t.state(16), State::Ending(11));
        t.given_back(10);
        assert_eq!(t.state(10), State::Present);
        assert_eq!(t.written(), 10);
        // After an end: given back into it, it goes on.
        let mut t = Timeline::new(E);
        write(&mut t, 0, 8, 8, supply(100, true));
        write(&mut t, 8, 40, 40, supply(18, false));
        assert_eq!(t.state(48), State::Absent);
        t.given_back(12);
        assert_eq!(t.state(12), State::Ending(8));
        assert_eq!(t.promised(12), 8 + E);
        // Given back to before the parts came: absent.
        t.given_back(0);
        assert_eq!(t.state(0), State::Absent);
    }

    #[test]
    fn the_parts_are_absent_since_their_end_or_the_start() {
        let mut t = Timeline::new(E);
        write(&mut t, 0, 8, 8, supply(30, true));
        assert_eq!(t.quiet_since(8), Some(0));
        write(&mut t, 8, 8, 8, supply(60, true));
        assert_eq!(t.quiet_since(16), None);
        // They run out at 30: ending from 20, absent from 30.
        write(&mut t, 16, 20, 20, supply(30, true));
        assert_eq!((t.quiet_since(25), t.quiet_since(36)), (None, Some(30)));
        write(&mut t, 36, 8, 8, supply(30, true));
        assert_eq!(t.quiet_since(44), Some(30));
    }

    /// Any supply that honours the promise, written in any pieces, with
    /// frames given back and gaps: the words keep it.
    #[test]
    fn the_promise_holds_whatever_comes() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = |m: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % m.max(1)
        };
        let (mut present, mut ended) = (0, 0);
        for _ in 0..200 {
            let mut t = Timeline::new(E);
            let mut words: Vec<Parts> = Vec::new();
            let mut there = 0u64;
            for _ in 0..200 {
                let from = words.len() as u64;
                if next(20) == 0 && from > 0 {
                    // Given back.
                    let back = next(from.min(30)) + 1;
                    t.given_back(from - back);
                    words.truncate((from - back) as usize);
                    continue;
                }
                if next(50) == 0 {
                    // A gap: what follows starts afresh, so check the rest
                    // on its own.
                    kept(&words).unwrap();
                    words.clear();
                    t = Timeline::new(E);
                    there = 0;
                    continue;
                }
                let from = words.len() as u64;
                // The buffer keeps what is promised; beyond, anything.
                there = there.max(t.promised(from)).max(from);
                match next(4) {
                    0 => {}
                    1 => there += next(60),
                    _ => there = t.promised(from).max(from) + next(30),
                }
                let s = Supply {
                    there,
                    coming: next(5) != 0,
                    hold: 40,
                    through: next(30) == 0,
                    main_end: Some(there),
                };
                let n = next(16) + 1;
                let took = next(n + 1);
                let plan = t.plan(from, n, s);
                for o in from..from + took {
                    let w = plan.word(o);
                    assert!(w == Parts::None || o < there, "a word with no parts at {o}");
                }
                t.commit(&plan, from, took);
                words.extend((from..from + took).map(|o| plan.word(o)));
                present += words.iter().filter(|&&w| w == Parts::Present).count();
                ended += words
                    .windows(2)
                    .filter(|w| (w[0], w[1]) == (Parts::Ending, Parts::None))
                    .count();
                if s.through && from + took >= there {
                    // The track ended here: the next one's first frame may
                    // go straight to None (main cuts there).
                    kept(&words).unwrap();
                    words.clear();
                    t = Timeline::new(E);
                    there = 0;
                }
                kept(&words).unwrap_or_else(|e| panic!("{e}: {words:?}"));
            }
        }
        // It went through all of it, many times.
        assert!(
            present > 10_000 && ended > 100,
            "{present} present, {ended} ends"
        );
    }
}
