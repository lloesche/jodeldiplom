//! What moves the playing track while it plays: a jump or a move onto a
//! map waiting for its frame ([`Move`], one at a time), and the second
//! reading of the track a move to another place or route may need: opened
//! a little before the frame it is to meet the track at, decoded ahead on
//! turns of its own while the track plays on, its cues held, and met there
//! crossfaded. A route switch's new route ([`super::switch`]) and a jump's
//! target the track does not hold ([`super::jump`]) are such readings; they
//! share their opening, their turns and their decoding here, and meet the
//! track each their own way (a route on the segment's own line, a jump on
//! a new segment at its target).

use std::time::{Duration, Instant};

use crate::error::Error;
use crate::rate::Route;

use super::along::PendingAlong;
use super::drop::drop_later;
use super::jump::{JumpFill, PendingJump};
use super::track::{Progress, StartAt, Track};
use super::{Engine, SWITCH_TURN};

/// The playing track's move to another place, or onto a map, waiting for
/// its frame: at most one.
pub(super) enum Move {
    /// None.
    Idle,
    /// A [`super::Command::Jump`] waiting for its time, its target in what
    /// the track holds.
    JumpAt(PendingJump),
    /// A jump whose target is read again ([`JumpFill`]): met at once, or
    /// at its time.
    JumpRead(Box<JumpFill>),
    /// A move onto a map waiting for its frame.
    MapAt(PendingAlong),
}

impl Move {
    /// `new` in place of what was under way (a reading dropped in the
    /// background). Whether a jump was waiting for its time.
    pub(super) fn set(&mut self, new: Move) -> bool {
        match std::mem::replace(self, new) {
            Move::JumpAt(_) => true,
            Move::JumpRead(fill) => {
                let timed = fill.at.is_some();
                drop_later(fill.track);
                timed
            }
            Move::Idle | Move::MapAt(_) => false,
        }
    }

    /// A jump under way goes (a move onto a map stays). Whether one was
    /// waiting for its time.
    pub(super) fn drop_jump(&mut self) -> bool {
        match self {
            Move::JumpAt(_) | Move::JumpRead(_) => self.set(Move::Idle),
            Move::Idle | Move::MapAt(_) => false,
        }
    }

    /// Whether a jump is under way.
    pub(super) fn jumping(&self) -> bool {
        matches!(self, Move::JumpAt(_) | Move::JumpRead(_))
    }

    /// The move onto a map waiting for its frame, if one is.
    pub(super) fn map_at(&mut self) -> Option<&mut PendingAlong> {
        match self {
            Move::MapAt(pending) => Some(pending),
            _ => None,
        }
    }

    /// The move onto a map waiting for its frame, taken off, if one is.
    pub(super) fn take_map(&mut self) -> Option<PendingAlong> {
        match std::mem::replace(self, Move::Idle) {
            Move::MapAt(pending) => Some(pending),
            other => {
                *self = other;
                None
            }
        }
    }

    /// The jump's reading under way, if one is.
    pub(super) fn reading(&self) -> Option<&JumpFill> {
        match self {
            Move::JumpRead(fill) => Some(fill),
            _ => None,
        }
    }
}

/// What a turn of a reading did.
pub(super) enum Ahead {
    /// It decoded some.
    Busy,
    /// It has decoded all there is.
    Done,
    /// It failed (an error that costs only the track): it holds what it
    /// decoded before.
    Broke(Error),
}

impl Engine {
    /// A second reading of the playing track (`index`) on `route` from
    /// track frame `frame`, decoded ahead of being heard: its cues held.
    pub(super) fn open_reading(
        &mut self,
        index: usize,
        frame: u64,
        route: Route,
    ) -> Result<Track, Error> {
        let mut track = self.open_track(index, StartAt::Frame(frame), Some(route))?;
        track.pending = true;
        Ok(track)
    }

    /// Whether a reading has the thread: paused, the output holding at
    /// least half its buffer, or it took nothing of the last write a
    /// moment ago ([`SWITCH_TURN`]).
    pub(super) fn reading_turn(&self) -> bool {
        let Some(out) = &self.head.out else {
            return false;
        };
        let buffered = out.output.written().saturating_sub(out.clock.position());
        let ring = out.info.ring_frames;
        self.paused
            || (ring > 0 && buffered.saturating_mul(2) >= ring)
            || self
                .head
                .stalled
                .is_some_and(|at: Instant| at.elapsed() < SWITCH_TURN)
    }

    /// One piece of `reading` decoded ahead.
    pub(super) fn read_ahead(&mut self, reading: &mut Track) -> Result<Ahead, Error> {
        match self.produce(reading, Duration::ZERO) {
            Ok(Progress::Busy) => Ok(Ahead::Busy),
            Ok(Progress::Complete) => Ok(Ahead::Done),
            Err(error) if error.is_skippable() => Ok(Ahead::Broke(error)),
            Err(error) => Err(error),
        }
    }
}
