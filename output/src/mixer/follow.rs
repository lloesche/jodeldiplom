//! A desk link: an input that follows another ([`super::Fader::follow`]).
//! The follower's stream frame 0 lands on the leader's stream frame
//! `origin`, and from there the follower moves with the leader's stream,
//! never with the device's frames:
//!
//! - **Lockstep:** each chunk, the follower fills exactly the chunk's
//!   frames its leader gave (from the chunk's first frame; a leader short
//!   of frames pads the rest), from the frame the leader is at `origin`
//!   on: until then it gives nothing, and in the chunk the leader gets
//!   there its frames start inside the chunk.
//! - **Held:** while the leader gives nothing (paused, short, flushed) the
//!   follower gives nothing either and loses nothing; a follower of a
//!   leader that does not run does not keep the device running, and is
//!   paused when its leader is (its ramps end).
//! - **Released:** a leader at its end (drained: its ring empty, nothing
//!   more expected) lets its followers go from the frame after its last:
//!   each plays on by itself, its map kept.
//! - **Late frames dropped:** the frame due at each of those chunk frames
//!   is an absolute stream frame (the leader's there less `origin`), so a
//!   follower behind it (it ran short) drops its frames up to it first,
//!   counted ([`crate::Stats::dropped`], [`super::Fader::dropped`])
//!   and played in no device frames. Its ring's read side is what counts,
//!   so a retract that writes the same frames again changes nothing. A
//!   follower ahead of it (linked after it gave frames) gives its frames
//!   from the chunk frame they are due at.
//! - **Its clock:** the follower's map of stream frames onto the device is
//!   its leader's, `origin` on, from the moment it is linked:
//!   [`super::Fader::at_stream`] is exact before it has been mixed.
//! - **Offline:** on a device that plays what it is given at once, no
//!   period that takes a leader past `origin` is mixed before its follower
//!   has begun (given a frame since it was linked), so the render is the
//!   same however the writers are scheduled (the same periods, the
//!   follower's frames in the same places), and nothing is dropped. A
//!   follower that has begun holds nothing up: paused, it is held.
//! - **Unlinked** by a flush or a close of either output (a new one opened
//!   on either input closes the old), and by [`super::Fader::unfollow`].
//!   A leader follows no one, and a follower leads no one.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::chunk::Slot;
use super::input::{Feed, Given, Link, Span};
use super::mic::Port;
use super::run::Inputs;
use crate::error::Error;

/// A follower's link, on its stream.
pub(crate) struct Follow {
    /// The leader's input, and the output open on it that is followed.
    pub(super) slot: u64,
    pub(super) leader: Arc<Link>,
    /// The leader's stream frame the follower's frame 0 lands on.
    pub(super) origin: u64,
    /// The follower has given frames since it was linked.
    pub(super) gave: bool,
    /// The follower's own map before it was linked (epoch, offset), kept
    /// for an unlink before it gives a frame.
    own: Option<(u64, u64)>,
}

impl Link {
    /// The output has run to its end: drained (nothing more is expected)
    /// with its ring empty, while it runs.
    fn ended(&self) -> bool {
        self.running.load(Ordering::SeqCst) && self.shared.draining() && self.shared.ring.len() == 0
    }
}

impl Feed {
    /// It runs, and is not held by a leader that does not.
    pub(crate) fn live(&self) -> bool {
        self.link.running.load(Ordering::SeqCst)
            && self
                .follow
                .as_ref()
                .is_none_or(|f| f.leader.running.load(Ordering::SeqCst))
    }

    /// It is paused, or follows a leader that is.
    pub(crate) fn paused(&self) -> bool {
        self.link.paused.load(Ordering::SeqCst)
            || self
                .follow
                .as_ref()
                .is_some_and(|f| f.leader.paused.load(Ordering::SeqCst))
    }
}

/// The frames of a chunk of `n` a follower of `origin` fills, from what
/// its leader gave there (`None`: nothing, so it holds); with the leader
/// `ended`, on to the chunk's end (from after the leader's last frame, if
/// it had not begun).
fn span(origin: u64, given: Option<Given>, n: usize, ended: bool) -> Span {
    let locked = given.and_then(|g| {
        let (start, end) = (g.frame.max(origin), g.frame + g.got);
        (end > start).then_some(Span {
            from: g.at + (start - g.frame) as usize,
            to: g.at + g.got as usize,
            due: Some(start - origin),
        })
    });
    match (locked, ended) {
        (Some(span), false) => span,
        (None, false) => Span::NONE,
        (Some(span), true) => Span { to: n, ..span },
        (None, true) => Span {
            from: given.map_or(0, |g| g.at + g.got as usize),
            to: n,
            due: None,
        },
    }
}

impl Slot {
    /// Its output follows no one any more: it keeps the map it had (its
    /// own frames' from before it was linked, or none) unless it gave
    /// frames since.
    fn unfollow(&mut self) {
        let Some(f) = self.feed.as_mut().and_then(|s| s.follow.take()) else {
            return;
        };
        match (f.gave, f.own) {
            (true, _) => {}
            (false, Some((epoch, offset))) => self.map_offset(epoch, offset),
            (false, None) => self.unmap(),
        }
    }

    /// Its leader ended: it plays on by itself, its map kept (its own
    /// frames', `own`, when it gave some in that chunk).
    fn release(&mut self, own: Option<u64>, epoch: u64) {
        if let Some(stream) = self.feed.as_mut() {
            stream.follow = None;
        }
        if let Some(offset) = own {
            self.map_offset(epoch, offset);
        }
    }

    /// While it follows, its map is its leader's (`mapped`), `origin` on.
    fn map_after(&mut self, mapped: Option<(u64, u64)>, origin: u64) {
        if let Some((epoch, offset)) = mapped {
            self.map_offset(epoch, offset.wrapping_add(origin));
        }
    }
}

impl Inputs {
    /// The output open on input `id` follows the one open on input
    /// `leader`'s from its frame `origin` (`None`: it follows no one).
    pub(super) fn follow(&mut self, id: u64, to: Option<(u64, u64)>) -> Result<(), Error> {
        let Some((leader, origin)) = to else {
            if let Some(slot) = self.slots.get_mut(&id) {
                slot.unfollow();
            }
            return Ok(());
        };
        let invalid = |why: &str| Err(Error::InvalidArgument(why.into()));
        if leader == id {
            return invalid("an input cannot follow itself");
        }
        let (Some(slot), Some(lead)) = (self.slots.get(&id), self.slots.get(&leader)) else {
            return Err(Error::Closed);
        };
        let (Some(own), Some(theirs)) = (&slot.feed, &lead.feed) else {
            return invalid("a link needs an output open on both inputs");
        };
        if theirs.follow.is_some() {
            return invalid("the leader follows another input");
        }
        let output = own.link.id;
        let leads = |s: &Feed| s.follow.as_ref().is_some_and(|f| f.leader.id == output);
        if self.feeds().any(leads) {
            return invalid("the input leads another");
        }
        let follow = Follow {
            slot: leader,
            leader: theirs.link.clone(),
            origin,
            gave: false,
            own: slot.mapped(),
        };
        let mapped = lead.mapped();
        let slot = self.slots.get_mut(&id).expect("checked");
        if let Some(stream) = slot.feed.as_mut() {
            stream.follow = Some(follow);
        }
        slot.map_after(mapped, origin);
        Ok(())
    }

    /// Output `id` was flushed or closed: its link ends, and so do those of
    /// the outputs that follow it.
    pub(super) fn unlink(&mut self, id: u64) {
        for slot in self.slots.values_mut() {
            let linked = slot.feed.as_ref().is_some_and(|s| {
                s.follow
                    .as_ref()
                    .is_some_and(|f| s.link.id == id || f.leader.id == id)
            });
            if linked {
                slot.unfollow();
            }
        }
    }

    /// On a device that plays what it is given at once: the frames every
    /// leader may still be mixed before its follower begins (`None`: no
    /// limit). A period that would go past it waits whole, so the periods
    /// are the same as with the follower running from the start.
    pub(super) fn offline_room(&self) -> Option<u64> {
        self.feeds()
            .filter(|s| !s.link.running.load(Ordering::SeqCst))
            .filter_map(|s| s.follow.as_ref())
            .filter(|f| !f.gave)
            .map(|f| f.origin.saturating_sub(f.leader.shared.head()))
            .min()
    }

    /// Takes each follower's frames of the chunk of `n` frames from device
    /// frame `first` of the device's opening `epoch`, in lockstep with what
    /// its leader gave there (the leaders are taken first; `port` notes
    /// where a tapped one's went), and lets go of those whose leader ended.
    pub(super) fn take_followers(&mut self, n: usize, first: u64, port: &Port, epoch: u64) {
        let mut next = self.slots.keys().next().copied();
        while let Some(id) = next {
            next = self.slots.range(id + 1..).next().map(|(&k, _)| k);
            let stream = self.slots[&id].feed.as_ref();
            let Some(f) = stream.and_then(|s| s.follow.as_ref()) else {
                continue;
            };
            let origin = f.origin;
            let ended = f.leader.ended();
            let lead = self.slots.get(&f.slot);
            let mapped = lead.and_then(Slot::mapped);
            let theirs = lead.and_then(|s| s.feed.as_ref());
            debug_assert!(
                theirs.is_some_and(|s| Arc::ptr_eq(&s.link, &f.leader)),
                "a link ends with its leader's output"
            );
            let given = theirs.and_then(Feed::given);
            let slot = self.slots.get_mut(&id).expect("listed");
            let own = slot.take(n, first, span(origin, given, n, ended), port);
            if ended {
                slot.release(own, epoch);
                continue;
            }
            slot.map_after(mapped, origin);
            if let Some(f) = slot.feed.as_mut().and_then(|s| s.follow.as_mut()) {
                f.gave |= own.is_some();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn given(at: usize, frame: u64, got: u64) -> Option<Given> {
        Some(Given { at, frame, got })
    }

    fn span_of(from: usize, to: usize, due: Option<u64>) -> Span {
        Span { from, to, due }
    }

    #[test]
    fn a_follower_fills_what_its_leader_gave_from_origin_on() {
        // Before the origin, and a leader that gave nothing: nothing.
        assert_eq!(span(1_000, given(0, 0, 480), 480, false), Span::NONE);
        assert_eq!(span(1_000, given(0, 520, 480), 480, false), Span::NONE);
        assert_eq!(span(1_000, None, 480, false), Span::NONE);
        assert_eq!(span(0, given(0, 7, 0), 480, false), Span::NONE);
        // The origin inside the chunk: from there, frame 0 due.
        let crossing = span(1_000, given(0, 960, 480), 480, false);
        assert_eq!(crossing, span_of(40, 480, Some(0)));
        // Past it, a leader short of frames: the same frames.
        let short = span(1_000, given(0, 2_000, 100), 480, false);
        assert_eq!(short, span_of(0, 100, Some(1_000)));
    }

    #[test]
    fn a_leader_at_its_end_lets_its_follower_go_on() {
        // Locked: on to the chunk's end, from the frame due.
        let last = span(1_000, given(0, 2_000, 100), 480, true);
        assert_eq!(last, span_of(0, 480, Some(1_000)));
        // Not begun: from after the leader's last frame.
        assert_eq!(
            span(1_000, given(0, 0, 100), 480, true),
            span_of(100, 480, None)
        );
        assert_eq!(span(1_000, None, 480, true), span_of(0, 480, None));
    }
}
