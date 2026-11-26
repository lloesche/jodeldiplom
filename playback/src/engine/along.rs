//! A head read along a time map, time-warped (a transition's tracks;
//! `Command::Along`
//! in tests): its stems are read along the map as its parts
//! ([`super::lanes`]), and its main is their sum, so the two stay on the
//! same frames for a mixer's switch between them.
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
//! is steered back onto it), or the head is told to leave it
//! ([`Engine::leave_map`]: a transition called off that had it glide), the
//! head goes back onto its own route at the same format, a switch met where
//! the map puts it ([`Engine::reroute`], with the settings then): from
//! there it plays the file bit for bit, as it would have. A switch back
//! that does not come about is asked again a second later.
//!
//! A transition reads its tracks along maps too ([`Engine::read_along_at`]): the
//! playing one along its glide, the next one, on a head of its own, from
//! its first frame ([`Engine::start_along`]).

use std::sync::Arc;
use std::time::Duration;

use crate::error::Error;
use crate::transition::TempoMode;
#[cfg(feature = "test-support")]
use crate::warp::TimeMap;

use super::api::Event;
use super::cues::CueEvent;
use super::lanes::next_of;
use super::moving::Move;
use super::policy::{Action, Refusal};
use super::track::{crossfade, frames_of, AlongMap, Back, HeardFrom, HeardWarp, Progress, Track};
use super::{Engine, WRITE_WAIT};

/// Seconds after its map runs straight that a head goes back onto its own
/// route.
const STRAIGHT: f64 = 0.3;

/// Seconds of output after which a switch back that did not come about is
/// asked again.
const RETRY: f64 = 1.0;

/// Seconds of a head's silence still to be heard (a transition's next
/// track's before its window) under which, on a device, stems not
/// separated in time give way to the whole mix.
const SOON: f64 = 0.25;

/// A move onto a map waiting for the playing head to be written up to its
/// frame.
pub(super) struct PendingAlong {
    /// The head's segment it was given in (another one drops it).
    segment: u64,
    map: AlongMap,
    from: u64,
}

impl Engine {
    /// `Command::Along` (feature `test-support`): the playing track is read
    /// along `map` (the output's seconds to the track's) from output frame
    /// `from` (the frame written next, if that is later), by `tempo`; it
    /// writes no further until it is. Refused (with a notice) with nothing
    /// playing, while a transition is set up, and where
    /// [`Engine::read_along_at`] refuses.
    #[cfg(feature = "test-support")]
    pub(super) fn read_along(&mut self, map: Arc<TimeMap>, from: u64, tempo: TempoMode) {
        let refuse = |why: &str| format!("This track cannot be read along a time map: {why}");
        let (Some(track), Some(out)) = (self.head.playing.as_ref(), self.head.out.as_ref()) else {
            let message = "Nothing is playing to be read along a time map".into();
            return self.notice(None, message);
        };
        let index = track.index;
        let heard = HeardWarp {
            map,
            origin: 0,
            a0: 0.0,
            out_rate: out.info.format.rate,
        };
        let why = if self.transition.is_some() {
            Err(Refusal::TransitionSetUp.reason())
        } else {
            self.read_along_at(heard, from, tempo)
        };
        if let Err(why) = why {
            self.notice(Some(index), refuse(&why));
        }
    }

    /// A move onto a map waiting for its frame takes `key` there; whether
    /// one is waiting.
    pub(super) fn set_pending_key(&mut self, key: i8) -> bool {
        let Some(pending) = self.head.moving.map_at() else {
            return false;
        };
        pending.map.key = key;
        true
    }

    /// The playing track is read along a map, as `heard` puts it, from
    /// output frame `from` (the frame written next, if that is later), by
    /// `tempo`; it writes no further until it is. Why not, for a track of
    /// more than two channels or off its own route (not along a map), off
    /// an output of lanes, while a count-in plays or a route switch is
    /// meeting the track; a route switch not meeting it yet is dropped, and
    /// a jump waiting (with a notice).
    pub(super) fn read_along_at(
        &mut self,
        heard: HeardWarp,
        from: u64,
        tempo: TempoMode,
    ) -> Result<(), String> {
        let track = self.head.playing.as_ref().ok_or("nothing is playing")?;
        self.allows(Action::Map).map_err(Refusal::reason)?;
        let index = track.index;
        let heard = Arc::new(heard);
        self.head.lanes.prepare_along(track, &heard, from);
        let pending = PendingAlong {
            segment: track.segment.id,
            map: AlongMap {
                heard,
                tempo,
                key: self.key,
            },
            from,
        };
        let timed = self.head.moving.set(Move::MapAt(pending));
        self.cancel_switch();
        if timed {
            let message =
                "The jump waiting for its time is dropped: the track is read along a time map";
            self.notice(Some(index), message.into());
        }
        Ok(())
    }

    /// A track nothing of whose segment is written yet (a transition's next
    /// track, on a head of its own, from its first frame; a track installed
    /// in a key) is read along `map` from output frame `at`, its segment's
    /// first, silent until `heard_from` says it is heard (its stems may be
    /// waited for until that is near): its stems' run starts now.
    pub(super) fn start_along(
        &mut self,
        track: &mut Track,
        map: AlongMap,
        at: u64,
        heard_from: HeardFrom,
    ) {
        self.head.lanes.prepare_along(track, &map.heard, at);
        track.go_along(map, at, 0);
        if let Some(along) = track.along.as_mut() {
            along.heard_from = Some(heard_from);
        }
    }

    /// The playing head leaves its map as soon as it can (a transition
    /// called off that had it glide): it goes back onto its own route from
    /// where the map has it, as once a map runs straight.
    pub(super) fn leave_map(&mut self) {
        if self.head.moving.map_at().is_some() {
            self.head.moving = Move::Idle;
        }
        if let Some(along) = self.head.playing.as_mut().and_then(|t| t.along.as_mut()) {
            along.leave = true;
        }
    }

    /// The sample index the playing `track` writes no further than for a
    /// move onto a map waiting for its frame.
    pub(super) fn along_limit(&self, track: &Track) -> Option<usize> {
        let Move::MapAt(pending) = &self.head.moving else {
            return None;
        };
        let written = self.head.out.as_ref()?.output.written();
        let ch = usize::from(track.route.channels);
        let to = pending.from.saturating_sub(written) as usize * ch;
        Some(track.out_at + to)
    }

    /// Moves `track` onto `map` from the frame written next, joined to what
    /// played: what it made from there (its route's, or along its map as it
    /// was) made through a switch's fade first, to crossfade from.
    pub(super) fn join_along(&mut self, track: &mut Track, map: AlongMap) -> Result<(), Error> {
        let out = self.head.out.as_ref().ok_or(Error::Stopped)?;
        let (at, rate) = (out.output.written(), out.info.format.rate);
        let fade = frames_of(self.switch_fade, rate) as usize;
        while track.unwritten() < fade {
            if let Progress::Complete = self.produce(track, WRITE_WAIT)? {
                break;
            }
        }
        track.go_along(map, at, fade);
        // What it queued past there is not heard; its position is the
        // map's from there.
        self.requeue_heard(track, at);
        Ok(())
    }

    /// `track` is heard as it is now from output frame `at`: what was
    /// queued for it from there (its clips, how it was heard) goes.
    pub(super) fn requeue_heard(&mut self, track: &Track, at: u64) {
        let index = track.index;
        self.cues.retract(at, |event| match event {
            CueEvent::Heard(heard) => heard.index == index,
            event => event.clipped_of(index),
        });
        self.cues.push(at, CueEvent::Heard(track.heard()));
    }

    /// A transition's next track on a device, waiting for its parts, has
    /// frames the desk plays past meanwhile: they are not heard (written
    /// then, they are late and dropped), so silence stands for them from
    /// `main_end` (where its main is made to) up to `unplayed`, made at
    /// once and written as the playing track plays; its parts, or the whole
    /// mix, start after them, with no stretch of late frames to get through
    /// first. Returns where its main is made to now.
    fn fill_unheard(&mut self, track: &mut Track, main_end: u64, unplayed: u64) -> u64 {
        if unplayed <= main_end {
            return main_end;
        }
        let ch = usize::from(track.route.channels).max(1);
        let before = track.out_buf.len();
        track
            .out_buf
            .resize(before + (unplayed - main_end) as usize * ch, 0);
        if let Some(along) = track.along.as_mut() {
            along.tail = None;
        }
        self.after_output(track, before);
        unplayed
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
        let out = self.head.out.as_ref().ok_or(Error::Stopped)?;
        let (written, rate) = (out.output.written(), out.info.format.rate);
        let ch = usize::from(track.route.channels).max(1);
        let index = track.index;
        let margin = self.lane_margin(rate);
        let main_end = track.line.seg_out + track.emitted;
        let fade = track
            .along
            .as_ref()
            .and_then(|a| a.tail.as_ref())
            .map_or(0, |t| (t.len() / ch) as u64);
        let ask = self.lane_ask(track, written, 0);
        let next = next_of(&self.queue, track, self.hold, self.transition.is_none());
        let target = main_end + margin + fade + 1;
        let making = match self
            .head
            .lanes
            .make_along(&ask.want(track, next), target, wait)
        {
            Ok(making) => making,
            Err(reason) => {
                self.emit(Event::PartsStopped { index, reason });
                Default::default()
            }
        };
        let (end, done) = self.head.lanes.along_made();
        let to = if done {
            end
        } else {
            end.saturating_sub(margin)
        };
        // A head's own wait to be heard (a count-in) is over once it is.
        if let Some(along) = track.along.as_mut() {
            along.heard_from.take_if(|h| h.heard_now());
        }
        let heard_from = track.along.as_ref().and_then(|a| a.heard_from.as_ref());
        let (pending, silent) = (
            heard_from.is_some(),
            heard_from.map_or(0, HeardFrom::silent),
        );
        let unplayed = heard_from.map_or(0, HeardFrom::unplayed);
        let main_end = if !self.offline && to <= main_end && !done {
            self.fill_unheard(track, main_end, unplayed)
        } else {
            main_end
        };
        // On a device stems that are not separated in time would leave it
        // with nothing to play: the whole mix from here (unless a while of
        // its silence is still to be heard first: a transition's next
        // track's frames late then are dropped on the desk, in time), from
        // where its main is made to for a next track (the silence above).
        let short = to < main_end + fade.max(1);
        let soon = (silent as f64) < SOON * f64::from(rate);
        if !self.offline && short && soon && making.waiting && self.head.lanes.along_separating() {
            let from = if pending { main_end } else { written };
            // (What follows the track is the writes' business: `before`.)
            let want = ask.want(track, super::lanes::Next::Nothing);
            self.head.lanes.along_whole(&want, from);
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
        track.mix_clips += self
            .head
            .lanes
            .sum_parts(main_end, to, ch, &mut track.out_buf);
        let clipped = self.head.lanes.along_clipped();
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
    /// given for another segment than the head's is dropped; one waits
    /// while a route switch is meeting the head); the head read along a map
    /// goes back onto its own route once the map has run straight a while
    /// there and a stretch is steered back onto it (it is at its own tempo
    /// then), or once it is told to leave it: a switch at the same format,
    /// met where the map puts it, asked again a second later when it does
    /// not come about.
    pub(super) fn along_turn(&mut self) -> Result<(), Error> {
        let Some(out) = self.head.out.as_ref() else {
            return Ok(());
        };
        let (written, rate) = (out.output.written(), out.info.format.rate);
        let retry = written + (RETRY * f64::from(rate)) as u64;
        if let Some(pending) = self.head.moving.take_map() {
            let meeting = self.head.switch.as_ref().is_some_and(|s| s.spliced);
            let turn = self.with_playing(|e, track| {
                if track.segment.id != pending.segment {
                    return None;
                }
                Some(if written < pending.from || meeting {
                    e.head.moving = Move::MapAt(pending);
                    Ok(())
                } else {
                    e.cancel_switch();
                    e.join_along(track, pending.map)
                })
            });
            return match turn {
                Some(Some(result)) => result,
                // Given for another segment: the key it was to take goes
                // on the head as it is.
                Some(None) => {
                    self.head.lanes.forget_ahead();
                    self.key_now()
                }
                None => {
                    self.head.lanes.forget_ahead();
                    Ok(())
                }
            };
        }
        let switching = self.head.switch.is_some();
        let steady = self.head.lanes.along_offset().abs() < 1.0;
        let Some(along) = self.head.playing.as_mut().and_then(|t| t.along.as_mut()) else {
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
        // (In a key of its own it stays on the map.)
        let settled = a >= heard.map.settled_at() + STRAIGHT && steady && along.map.key == 0;
        let straight = along.leave || settled;
        if along.back != Back::Reading || !straight {
            return Ok(());
        }
        along.back = Back::Asked;
        self.reroute();
        if self.head.switch.is_none() {
            if let Some(along) = self.head.playing.as_mut().and_then(|t| t.along.as_mut()) {
                along.back = Back::Retry(retry);
            }
        }
        Ok(())
    }
}
