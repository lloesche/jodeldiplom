//! Changes to the queue while it plays ([`crate::Command::Remove`],
//! [`crate::Command::Move`]): every index the engine holds (the track
//! playing and the one waiting, a switch, a transition, the cues still to
//! be heard) is moved along with its track, so playback goes on as it
//! was. The track being heard, removed, gives way to the one after it; a
//! track decoded ahead of it and not heard yet goes without a break in it.

use super::api::Event;
use super::track::{Cue, CueEvent};
use super::Engine;
use crate::error::Error;

/// What a track removed is to playback.
enum Removed {
    /// Decoded ahead of the track heard and not heard yet: its start is
    /// queued (the cue at this place in the queue of cues; `None`: it
    /// waits for the output to play out, nothing of it written).
    Ahead { start: Option<usize> },
    /// Heard, or about to be (playback is not under way to it from a
    /// track heard).
    Heard,
    /// Anything else: only the indices move.
    Other,
}

impl Engine {
    /// [`crate::Command::Remove`].
    pub(super) fn remove_track(&mut self, index: usize) {
        if index >= self.queue.len() {
            // Nothing to do; the queue is said all the same (a frontend may
            // wait for it).
            return self.emit_queue();
        }
        match self.removed(index) {
            Removed::Ahead { start } => self.remove_ahead(index, start),
            Removed::Heard => {
                // What is heard (or about to be) goes: the track after it
                // plays, from its start, as the next track key would. Only
                // a track heard ends; the jump ends any other heard.
                self.queue.remove(index);
                if self.heard.as_ref().is_some_and(|h| h.index == index) {
                    self.emit(Event::TrackEnded {
                        index,
                        finished: false,
                    });
                    self.heard = None;
                }
                self.emit_queue();
                if index < self.queue.len() {
                    self.paused = false;
                    self.jump(index, 0.0);
                } else {
                    self.stop(true);
                }
            }
            Removed::Other => {
                self.queue.remove(index);
                self.reindex(&gone(index));
                self.emit_queue();
            }
        }
    }

    /// What track `index` is to playback, from the timeline: ahead when a
    /// track other than it is heard, no transition is under way, and its
    /// start is queued or it is the track waiting.
    fn removed(&self, index: usize) -> Removed {
        let heard = self.heard.as_ref().map(|h| h.index);
        let start = self
            .cues
            .iter()
            .position(|c| matches!(&c.event, CueEvent::Started(s, _) if s.index == index));
        let waiting = self.next.as_ref().is_some_and(|t| t.index == index);
        let other_heard = heard.is_some_and(|h| h != index);
        if other_heard && !self.transition_running() && (start.is_some() || waiting) {
            return Removed::Ahead { start };
        }
        let playing = self.head.playing.as_ref().map(|t| t.index);
        let waiting = self.next.as_ref().map(|t| t.index);
        if [heard, playing, waiting].contains(&Some(index)) {
            Removed::Heard
        } else {
            Removed::Other
        }
    }

    /// Removes track `index`, decoded ahead ([`Removed::Ahead`]): the queue
    /// is said first. Then what was written from its start on is taken
    /// back, and the track after it follows the one heard from there, as
    /// if it had been next all along (held there with
    /// [`crate::Command::HoldAtEnd`]); the heard one plays to its end. The
    /// output gives back all but what the device keeps (a few of its
    /// buffers, for one on a mixer's input): what it keeps of the removed
    /// track is heard after the heard one's end, the next track after it.
    fn remove_ahead(&mut self, index: usize, start: Option<usize>) {
        self.queue.remove(index);
        self.reindex(&gone(index));
        self.emit_queue();
        // Every cue from its start on is its own or a later track's.
        let from = start.map(|i| {
            let at = self.cues[i].at;
            self.cues.truncate(i);
            at
        });
        let mut end = 0;
        if let Some(out) = self.head.out.as_mut() {
            let written = out.output.written();
            if let Some(from) = from {
                if let Err(e) = out.output.retract(written.saturating_sub(from)) {
                    return self.fail(None, e.into());
                }
            }
            end = out.output.written();
        }
        self.drop_underway();
        self.finishing = false;
        self.tail = None;
        if let Err(error) = self.after_end(index, end) {
            self.fail(None, error);
        }
    }

    /// What follows a track whose last frame is written by output frame
    /// `end`: with [`crate::Command::HoldAtEnd`], track `next` waits to be
    /// started (playback stops once the track is heard to its end);
    /// otherwise playback goes on with it.
    pub(super) fn after_end(&mut self, next: usize, end: u64) -> Result<(), Error> {
        if self.hold && next < self.queue.len() {
            self.cues.push_back(Cue {
                at: end,
                event: CueEvent::Held { index: next },
            });
            self.tail = None;
            self.finishing = true;
            return Ok(());
        }
        self.advance(next, end)
    }

    /// [`crate::Command::Move`].
    pub(super) fn move_track(&mut self, from: usize, to: usize) {
        let len = self.queue.len();
        if from >= len || to >= len || from == to {
            return self.emit_queue();
        }
        let path = self.queue.remove(from);
        self.queue.insert(to, path);
        self.reindex(&|i| {
            Some(if i == from {
                to
            } else if from < to && i > from && i <= to {
                i - 1
            } else if to < from && i >= to && i < from {
                i + 1
            } else {
                i
            })
        });
        self.emit_queue();
    }

    /// Moves every queue index the engine holds by `map` (`None`: that
    /// track is gone; nothing it held can stay).
    fn reindex(&mut self, map: &dyn Fn(usize) -> Option<usize>) {
        let keep = |i: usize| map(i).unwrap_or(i);
        for track in [self.head.playing.as_mut(), self.next.as_mut()]
            .into_iter()
            .flatten()
        {
            track.index = keep(track.index);
            for cue in &mut track.held {
                cue.event.reindex(map);
            }
        }
        if let Some(switch) = &mut self.head.switch {
            switch.track.index = keep(switch.track.index);
        }
        if let Some(heard) = &mut self.heard {
            heard.index = keep(heard.index);
        }
        self.tail = self.tail.and_then(map);
        for cue in &mut self.cues {
            cue.event.reindex(map);
        }
        self.reindex_jumps(map);
        self.reindex_transition(map);
    }
}

/// The queue indices once track `index` is removed (`None`: that track).
fn gone(index: usize) -> impl Fn(usize) -> Option<usize> {
    move |i| match i.cmp(&index) {
        std::cmp::Ordering::Less => Some(i),
        std::cmp::Ordering::Equal => None,
        std::cmp::Ordering::Greater => Some(i - 1),
    }
}

impl CueEvent {
    /// Its track's index moved by `map` (kept where that track is gone).
    pub(super) fn reindex(&mut self, map: &dyn Fn(usize) -> Option<usize>) {
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
}
