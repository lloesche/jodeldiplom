//! Changes to the queue while it plays ([`crate::Command::Remove`],
//! [`crate::Command::Move`]): every index the engine holds (the track
//! playing and the one waiting, a switch, a transition, the cues still to
//! be heard) is moved along with its track, so playback goes on as it
//! was. The track being heard, removed, gives way to the one after it.

use super::api::Event;
use super::track::CueEvent;
use super::Engine;

impl Engine {
    /// [`crate::Command::Remove`].
    pub(super) fn remove_track(&mut self, index: usize) {
        if index >= self.queue.len() {
            // Nothing to do; the queue is said all the same (a frontend may
            // wait for it).
            return self.emit_queue();
        }
        let heard = self.heard.as_ref().map(|h| h.index);
        let playing = self.playing.as_ref().map(|t| t.index);
        let waiting = self.next.as_ref().map(|t| t.index);
        if [heard, playing, waiting].contains(&Some(index)) {
            // What is heard (or about to be) goes: the track after it
            // plays, from its start, as the next track key would.
            self.queue.remove(index);
            self.emit(Event::TrackEnded {
                index,
                finished: false,
            });
            self.heard = None;
            self.emit_queue();
            if index < self.queue.len() {
                self.paused = false;
                self.jump(index, 0.0);
            } else {
                self.stop(true);
            }
            return;
        }
        self.queue.remove(index);
        self.reindex(&|i| match i.cmp(&index) {
            std::cmp::Ordering::Less => Some(i),
            std::cmp::Ordering::Equal => None,
            std::cmp::Ordering::Greater => Some(i - 1),
        });
        self.emit_queue();
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
        for track in [self.playing.as_mut(), self.next.as_mut()]
            .into_iter()
            .flatten()
        {
            track.index = keep(track.index);
            for cue in &mut track.held {
                cue.event.reindex(map);
            }
        }
        if let Some(switch) = &mut self.switch {
            switch.track.index = keep(switch.track.index);
        }
        if let Some(heard) = &mut self.heard {
            heard.index = keep(heard.index);
        }
        self.stem_gains.reindex(&keep);
        self.tail = self.tail.and_then(map);
        for cue in &mut self.cues {
            cue.event.reindex(map);
        }
        self.reindex_jumps(map);
        self.reindex_transition(map);
    }
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
}
