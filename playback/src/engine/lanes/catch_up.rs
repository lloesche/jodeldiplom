//! The parts caught up behind the write head: when the output holds frames
//! written saying the parts are absent while parts are being made for them
//! (the first touch of a stem's strip, say, with [`super::MakeStems`] going
//! from `Never` to `Kept`, whose parts start on the frame heard), once the
//! parts are made far enough past what is written, the output gives back
//! what it has not played ([`jodel_output::Output::retract`], as a jump
//! does) and the same main is written again from there with its parts. So
//! the touch is heard about as soon as the parts are made (a stem cache: a
//! few milliseconds; a separation: its first chunk), not after the
//! output's ring has played out. Parts that ran out and are made again (a
//! separator that fell behind) are caught up the same way.
//!
//! For that the playing track keeps what the output may give back behind
//! what it wrote, also without a jump window ([`Engine::window_frames`]).
//! Main is written again exactly as it was: only where the buffer holds it
//! as written ([`Track::rewritable_from`]: after a jump's crossfade), on
//! the track's plain route, with nothing else under way that counts the
//! frames written (a route switch, a transition, a jump waiting). Only
//! frames whose words all said the parts are absent are written again, and
//! the words given back go with them
//! ([`super::control::Timeline::given_back`]), so the promise is kept.

use jodel_output::lanes::ending_frames;

use super::stream::Line;
use super::{StemLanes, AHEAD};
use crate::engine::track::Track;
use crate::engine::Engine;
use crate::error::Error;

impl StemLanes {
    /// Before a write of `line` from output frame `from`: the frame to give
    /// the output back to, so what follows is written again with its parts.
    /// From the first of the frames before `from` that all said they are
    /// absent (no sooner than `floor`, where the frames may be written
    /// again, nor than the parts there are), once the parts reach as far
    /// past `from` as they must for the frames written next to say they
    /// are there: as far ahead as absent parts must be made before they
    /// are ([`AHEAD`]), and at least a mixer's switch and `margin` (the
    /// output may give back less than asked, so it is reckoned from
    /// `from`).
    pub(super) fn rewind_to(&self, line: &Line, from: u64, floor: u64, margin: u64) -> Option<u64> {
        self.stream.as_ref().filter(|s| s.line.same(line))?;
        let quiet = self.timeline.quiet_since(from)?;
        let at = quiet.max(floor).max(self.buffer.start());
        let hold = (AHEAD * f64::from(self.rate)).round() as u64;
        let need = from + hold.max(margin + ending_frames(self.rate));
        (at < from && self.buffer.there(at) >= need).then_some(at)
    }
}

impl Engine {
    /// The first output frame `track` may write again, when there is one
    /// before what is written: the next one heard, within what its buffer
    /// holds as written, after a route's rise. `None` offline (the writing
    /// waits for the parts there), with a route switch, a transition or a
    /// jump under way (they count on the frames written), off the plain
    /// route, and with nothing to write again.
    pub(super) fn rewrite_floor(&self, track: &Track) -> Option<u64> {
        let busy = self.head.switch.is_some()
            || self.transition.is_some()
            || self.head.pending_jump.is_some()
            || self.head.jump_fill.is_some();
        if self.offline || busy || !track.plain() {
            return None;
        }
        let out = self.head.out.as_ref().filter(|o| o.output.lanes() > 1)?;
        let ch = usize::from(track.route.channels);
        let written = out.output.written();
        let held = (track.out_at / ch) as u64;
        let floor = out
            .clock
            .position()
            .max(written.saturating_sub(held))
            .max(track.rewritable_from)
            .max(track.seg_out + track.parts_rise);
        (floor < written).then_some(floor)
    }

    /// Before a write of lanes: once the parts for frames written without
    /// them are made ([`StemLanes::rewind_to`]), the output gives back
    /// what it has not played from there, and `track` writes it again.
    pub(super) fn rewrite_lanes(&mut self, track: &mut Track) -> Result<(), Error> {
        let (Some(floor), Some(out)) = (self.rewrite_floor(track), self.head.out.as_ref()) else {
            return Ok(());
        };
        let from = out.output.written();
        let margin = self.lane_margin(out.info.format.rate);
        let Some(to) = self
            .head
            .lanes
            .rewind_to(&Line::of(track), from, floor, margin)
        else {
            return Ok(());
        };
        let out = self.head.out.as_mut().ok_or(Error::Stopped)?;
        let n = out.output.retract(from - to)?;
        track.out_at -= n as usize * usize::from(track.route.channels);
        Ok(())
    }
}
