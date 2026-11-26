//! What is said as it is heard: cues ([`Cue`]) that wait for the output
//! clock to reach their frame, kept in output-frame order by the one queue
//! that holds them ([`CueQueue`]): the engine's, or a track's own while
//! its cues are held (decoded ahead of being heard).

use std::collections::VecDeque;

use crate::error::Error;
use crate::transition::TransitionState;

use super::api::TrackStart;
use super::track::Heard;

/// Something to send when the output clock reaches a frame.
pub(super) enum CueEvent {
    Started(Box<TrackStart>, Heard),
    Clipped {
        index: usize,
        samples: u64,
    },
    Ended {
        index: usize,
    },
    Skipped {
        index: usize,
        error: Error,
    },
    QueueEnded,
    /// Holding at the end: this track is the next one.
    Held {
        index: usize,
    },
    /// A transition's progress, as heard.
    Transition(TransitionState),
    /// The track heard is now mapped by this (it went onto a map, or it
    /// jumped).
    Heard(Heard),
    /// A jump within the track is heard.
    Jumped {
        index: usize,
        seconds: f64,
    },
}

impl CueEvent {
    /// Its track's index moved by `map` (kept where that track is gone).
    fn reindex(&mut self, map: &dyn Fn(usize) -> Option<usize>) {
        let keep = |i: &mut usize| *i = map(*i).unwrap_or(*i);
        match self {
            CueEvent::Started(start, heard) => {
                keep(&mut start.index);
                keep(&mut heard.index);
            }
            CueEvent::Heard(heard) => keep(&mut heard.index),
            CueEvent::Clipped { index, .. }
            | CueEvent::Ended { index }
            | CueEvent::Skipped { index, .. }
            | CueEvent::Jumped { index, .. }
            | CueEvent::Held { index } => keep(index),
            CueEvent::QueueEnded | CueEvent::Transition(_) => {}
        }
    }

    /// Whether it is track `index`'s clip count.
    pub(super) fn clipped_of(&self, index: usize) -> bool {
        matches!(self, CueEvent::Clipped { index: i, .. } if *i == index)
    }
}

/// An event and the output frame it is heard at.
pub(super) struct Cue {
    pub(super) at: u64,
    pub(super) event: CueEvent,
}

/// Cues in the order they are heard: each one's frame no earlier than the
/// one's before it.
#[derive(Default)]
pub(super) struct CueQueue(VecDeque<Cue>);

impl CueQueue {
    /// Queues `event` at output frame `at`, or at the last cue's frame when
    /// that is later. Returns where it went.
    pub(super) fn push(&mut self, at: u64, event: CueEvent) -> u64 {
        let at = at.max(self.0.back().map_or(0, |c| c.at));
        self.0.push_back(Cue { at, event });
        at
    }

    /// The first cue, taken off, when the clock at `position` has reached
    /// it.
    pub(super) fn pop_due(&mut self, position: u64) -> Option<Cue> {
        match self.0.front() {
            Some(c) if c.at <= position => self.0.pop_front(),
            _ => None,
        }
    }

    /// The cues from output frame `from` on that `stale` says are not heard
    /// (what was queued for frames given back, or past a cut) go.
    pub(super) fn retract(&mut self, from: u64, stale: impl Fn(&CueEvent) -> bool) {
        self.0.retain(|c| c.at < from || !stale(&c.event));
    }

    /// The cues before output frame `from` go (they were for frames not
    /// heard).
    pub(super) fn drop_before(&mut self, from: u64) {
        self.0.retain(|c| c.at >= from);
    }

    /// Every cue `by` frames earlier (the first frame at the earliest): as
    /// counted on another output whose frame 0 is this one's frame `by`.
    pub(super) fn rebase(&mut self, by: u64) {
        for cue in &mut self.0 {
            cue.at = cue.at.saturating_sub(by);
        }
    }

    /// No cue later than output frame `end` (nothing is heard after it).
    pub(super) fn clamp_to(&mut self, end: u64) {
        for cue in &mut self.0 {
            cue.at = cue.at.min(end);
        }
    }

    /// Queues every cue of `held`, counted from output frame `offset`.
    pub(super) fn release(&mut self, held: &mut CueQueue, offset: u64) {
        for cue in held.0.drain(..) {
            self.push(cue.at + offset, cue.event);
        }
    }

    /// Where the first cue that `first` matches is in the queue.
    pub(super) fn position(&self, first: impl Fn(&CueEvent) -> bool) -> Option<usize> {
        self.0.iter().position(|c| first(&c.event))
    }

    /// The cues from the one at `position` on go; the output frame of that
    /// one.
    pub(super) fn cut_at(&mut self, position: usize) -> u64 {
        let at = self.0[position].at;
        self.0.truncate(position);
        at
    }

    /// Every track index the cues hold moved by `map`.
    pub(super) fn reindex(&mut self, map: &dyn Fn(usize) -> Option<usize>) {
        for cue in &mut self.0 {
            cue.event.reindex(map);
        }
    }

    /// The frame of the last cue, if any.
    pub(super) fn last_at(&self) -> Option<u64> {
        self.0.back().map(|c| c.at)
    }

    pub(super) fn len(&self) -> usize {
        self.0.len()
    }

    pub(super) fn clear(&mut self) {
        self.0.clear();
    }
}
