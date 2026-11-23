//! A head read along a time map ([`crate::Command::Along`]): its stems are
//! read along the map as its parts ([`super::lanes`]), and its main is
//! their sum, so the two stay on the same frames for a mixer's switch
//! between them.
//!
//! Asked for, the move waits for the head to be written up to its frame
//! (it writes no further), and the run of its stems there starts at once
//! unless one has them already. There the head moves onto the map as onto
//! a segment joined to what played: main crossfaded once from its route's
//! over a switch's fade, the parts from what was there as a jump's are.
//! Without a separator (or stems that failed) the whole mix is the other
//! instruments' part; on a device, stems separated too slowly to be heard
//! on time give way to the whole mix too. While it waits or reads along
//! the map nothing moves the head to another route (new settings wait), and
//! no transition is set up. Once the map runs straight again (and a stretch
//! is steered back onto it), the head goes back onto its own route at the
//! same format, a switch met where the map puts it ([`Engine::reroute`],
//! with the settings then): from there it plays the file bit for bit, as
//! it would have. A switch back that does not come about is asked again a
//! second later.

use std::sync::Arc;
use std::time::Duration;

use crate::error::Error;
use crate::transition::TempoMode;
use crate::warp::TimeMap;

use super::api::Event;
use super::lanes::next_of;
use super::track::{
    crossfade, frames_of, AlongMap, Back, Cue, CueEvent, HeardWarp, Progress, Track,
};
use super::{Engine, WRITE_WAIT};

/// Seconds after its map runs straight that a head goes back onto its own
/// route.
const STRAIGHT: f64 = 0.3;

/// Seconds of output after which a switch back that did not come about is
/// asked again.
const RETRY: f64 = 1.0;

/// A [`crate::Command::Along`] waiting for the playing head to be written
/// up to its frame.
pub(super) struct PendingAlong {
    /// The head's segment it was given in (another one drops it).
    segment: u64,
    map: AlongMap,
    from: u64,
}

impl Engine {
    /// Whether a head is read along a map, or waits to be.
    pub(super) fn along_under_way(&self) -> bool {
        self.along_at.is_some() || self.playing.as_ref().is_some_and(|t| t.along.is_some())
    }

    /// Whether a head read along a map (or waiting to be) holds its route:
    /// any but the switch back it asked for waits.
    pub(super) fn along_holds_route(&self) -> bool {
        let on_map = self.playing.as_ref().and_then(|t| t.along.as_ref());
        self.along_at.is_some() || on_map.is_some_and(|a| a.back != Back::Asked)
    }

    /// [`crate::Command::Along`]: the playing track is read along `map`
    /// from output frame `from` (the frame written next, if that is
    /// later), by `tempo`; it writes no further until it is. Refused (with
    /// a notice) with nothing playing, for a track of more than two
    /// channels or off its own route (a transition's), off an output of
    /// lanes, while a transition is set up, a count-in plays or a route
    /// switch is meeting the track; a route switch not meeting it yet is
    /// dropped, and a jump waiting (with a notice).
    pub(super) fn read_along(&mut self, map: Arc<TimeMap>, from: u64, tempo: TempoMode) {
        let refuse = |why: &str| format!("This track cannot be read along a time map: {why}");
        let (Some(track), Some(out)) = (self.playing.as_ref(), self.out.as_ref()) else {
            let message = "Nothing is playing to be read along a time map".into();
            return self.notice(None, message);
        };
        let index = track.index;
        let why = if track.route.channels > 2 {
            Some(format!(
                "it has {} channels, and stems are made of two at most",
                track.route.channels
            ))
        } else if !track.plain() && track.along.is_none() {
            Some("it is not on its own route".into())
        } else if out.output.lanes() <= 1 {
            Some("the output takes no stems beside it".into())
        } else if self.transition.is_some() {
            Some("a transition is set up".into())
        } else if self.lead > 0 {
            Some("its count-in is playing".into())
        } else if self.switch.as_ref().is_some_and(|s| s.spliced) {
            Some("its move to another route is meeting it".into())
        } else {
            None
        };
        if let Some(why) = why {
            return self.notice(Some(index), refuse(&why));
        }
        let heard = Arc::new(HeardWarp {
            map,
            origin: 0,
            a0: 0.0,
            out_rate: out.info.format.rate,
        });
        self.lanes.prepare_along(track, &heard, from);
        self.along_at = Some(PendingAlong {
            segment: track.segment.id,
            map: AlongMap { heard, tempo },
            from,
        });
        self.cancel_switch();
        self.drop_jump_fill();
        if self.pending_jump.take().is_some() {
            let message =
                "The jump waiting for its time is dropped: the track is read along a time map";
            self.notice(Some(index), message.into());
        }
    }

    /// The sample index the playing `track` writes no further than for a
    /// move onto a map waiting for its frame.
    pub(super) fn along_limit(&self, track: &Track) -> Option<usize> {
        let pending = self.along_at.as_ref()?;
        let written = self.out.as_ref()?.output.written();
        let ch = usize::from(track.route.channels);
        let to = pending.from.saturating_sub(written) as usize * ch;
        Some(track.out_at + to)
    }

    /// Moves `track` onto `map` from the frame written next, its route's
    /// output made through a switch's fade first (to crossfade from).
    fn begin_along(&mut self, track: &mut Track, map: AlongMap) -> Result<(), Error> {
        let out = self.out.as_ref().ok_or(Error::Stopped)?;
        let (at, rate) = (out.output.written(), out.info.format.rate);
        let fade = frames_of(self.switch_fade, rate) as usize;
        while track.unwritten() < fade {
            if let Progress::Complete = self.produce(track, WRITE_WAIT)? {
                break;
            }
        }
        track.go_along(map, at, fade);
        // What its route queued past there is not heard; its position is
        // the map's from there.
        let index = track.index;
        self.cues.retain(|c| {
            c.at < at || !matches!(&c.event, CueEvent::Clipped { index: i, .. } if *i == index)
        });
        let at = at.max(self.cues.back().map_or(0, |c| c.at));
        self.cues.push_back(Cue {
            at,
            event: CueEvent::Heard(track.heard()),
        });
        Ok(())
    }

    /// One piece of a head read along a map: its parts made a piece past
    /// what main needs (waiting up to `wait` for its stems), and main their
    /// sum up to a seam's margin before their end (so the parts are there
    /// past any frame written), to their end once they are all made; its
    /// first piece a switch's fade long at least, the route's main
    /// crossfaded into it.
    pub(super) fn step_along(
        &mut self,
        track: &mut Track,
        wait: Duration,
    ) -> Result<Progress, Error> {
        let out = self.out.as_ref().ok_or(Error::Stopped)?;
        let (written, rate) = (out.output.written(), out.info.format.rate);
        let ch = usize::from(track.route.channels).max(1);
        let index = track.index;
        let margin = self.lane_margin(rate);
        let main_end = track.seg_out + track.emitted;
        let fade = track
            .along
            .as_ref()
            .and_then(|a| a.tail.as_ref())
            .map_or(0, |t| (t.len() / ch) as u64);
        let ask = self.lane_ask(track, written, 0);
        let next = next_of(&self.queue, track, self.hold, self.transition.is_none());
        let target = main_end + margin + fade + 1;
        let making = match self.lanes.make_along(&ask.want(track, next), target, wait) {
            Ok(making) => making,
            Err(reason) => {
                self.emit(Event::PartsStopped { index, reason });
                Default::default()
            }
        };
        let (end, done) = self.lanes.along_made();
        let to = if done {
            end
        } else {
            end.saturating_sub(margin)
        };
        // On a device stems that are not separated in time would leave it
        // with nothing to play: the whole mix from here.
        let short = to < main_end + fade.max(1);
        if !self.offline && short && making.waiting && self.lanes.along_separating() {
            self.lanes.along_whole(track, written);
            let reason = "the stems were not separated in time: the whole mix plays".into();
            self.emit(Event::PartsStopped { index, reason });
            return Ok(Progress::Busy);
        }
        if to <= main_end || (short && !done) {
            if done {
                track.source_done = true;
                return Ok(Progress::Complete);
            }
            return Ok(Progress::Busy);
        }
        let before = track.out_buf.len();
        track.mix_clips += self.lanes.sum_parts(main_end, to, ch, &mut track.out_buf);
        let clipped = self.lanes.along_clipped();
        if let Some(along) = track.along.as_mut() {
            // The route's main crossfaded into the sum, once.
            if let Some(tail) = along.tail.take() {
                let n = tail.len().min(track.out_buf.len() - before);
                crossfade(&tail[..n], &mut track.out_buf[before..before + n], ch);
            }
            // The parts' own clips (counted from their latest start).
            along.clips = along.clips.min(clipped);
            track.mix_clips += clipped - along.clips;
            along.clips = clipped;
        }
        self.after_output(track, before);
        Ok(Progress::Busy)
    }

    /// A move onto a map whose frame is the one written next is made (one
    /// given for another segment than the head's, or once a transition is
    /// set up, is dropped; one waits while a route switch is meeting the
    /// head); the head read along a map goes back onto its own route once
    /// the map has run straight a while there and a stretch is steered
    /// back onto it (it is at its own tempo then): a switch at the same
    /// format, met where the map puts it, asked again a second later when
    /// it does not come about.
    pub(super) fn along_turn(&mut self) -> Result<(), Error> {
        let Some(out) = self.out.as_ref() else {
            return Ok(());
        };
        let (written, rate) = (out.output.written(), out.info.format.rate);
        let retry = written + (RETRY * f64::from(rate)) as u64;
        if let Some(pending) = self.along_at.take() {
            let Some(mut track) = self.playing.take() else {
                self.lanes.forget_ahead();
                return Ok(());
            };
            let meeting = self.switch.as_ref().is_some_and(|s| s.spliced);
            let result = if track.segment.id != pending.segment || self.transition.is_some() {
                self.lanes.forget_ahead();
                Ok(())
            } else if written < pending.from || meeting {
                self.along_at = Some(pending);
                Ok(())
            } else {
                self.cancel_switch();
                self.begin_along(&mut track, pending.map)
            };
            self.playing = Some(track);
            return result;
        }
        let switching = self.switch.is_some();
        let steady = self.lanes.along_offset().abs() < 1.0;
        let Some(along) = self.playing.as_mut().and_then(|t| t.along.as_mut()) else {
            return Ok(());
        };
        match along.back {
            Back::Asked if !switching => along.back = Back::Retry(retry),
            Back::Retry(at) if written >= at => along.back = Back::Reading,
            _ => {}
        }
        let heard = &along.map.heard;
        let a = heard.a0
            + written.saturating_sub(heard.origin) as f64 / f64::from(heard.out_rate.max(1));
        let straight = a >= heard.map.settled_at() + STRAIGHT && steady;
        if along.back != Back::Reading || !straight {
            return Ok(());
        }
        along.back = Back::Asked;
        self.reroute();
        if self.switch.is_none() {
            if let Some(along) = self.playing.as_mut().and_then(|t| t.along.as_mut()) {
                along.back = Back::Retry(retry);
            }
        }
        Ok(())
    }
}
