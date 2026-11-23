//! A head read along a time map ([`crate::Command::Along`]): its stems are
//! read along the map as its parts ([`Stream`] on a line with a map), and
//! its main is their sum, so the two stay on the same frames for a mixer's
//! switch between them.
//!
//! The head moves onto the map at the frame it is told (the frame written
//! next at the earliest) as onto a segment joined to what played: main
//! crossfaded from its route's over a switch's fade, the parts from what
//! was there as a jump's are. Its stems come from the run kept when that
//! has them there, so a separation under way goes on; without a separator
//! (or stems that failed) the whole mix is the other instruments' part.
//! Once the map runs straight again, the head goes back onto its own route
//! at the same format, a switch met where the map puts it
//! ([`Engine::reroute`]): from there it plays the file bit for bit, as it
//! would have.

use std::sync::Arc;
use std::time::Duration;

use crate::error::Error;
use crate::stems::{StemRun, Unkept, RATE as STEM_RATE};
use crate::transition::TempoMode;
use crate::warp::TimeMap;

use super::super::api::Event;
use super::super::track::{
    fade_weight, frames_of, Along, AlongMap, Cue, CueEvent, HeardWarp, Progress, SegStart, Track,
};
use super::super::{Engine, PREROLL_RESAMPLE, WRITE_WAIT};
use super::stream::{Line, Made, Stream};
use super::{next_of, StemLanes, Stopped, Want};

/// Seconds of stems before the map's first read position (the reader's
/// interpolation reaches back into them).
const MARGIN: f64 = 0.05;

/// Seconds after its map runs straight that a head goes back onto its own
/// route.
const STRAIGHT: f64 = 0.3;

/// The 24-bit range main is summed into.
const TOP: i64 = (1 << 23) - 1;

impl StemLanes {
    /// Starts the parts of a head read along a map (`line` has one) from
    /// its segment's first frame (or the frame written next, should that be
    /// later): a resampler's run-up before it, into what is there,
    /// crossfaded as main is; from the run kept when it has the stems
    /// there, else from a new one that gets those not kept as `unkept`
    /// says.
    pub(super) fn start_along(&mut self, line: Line, want: &Want, unkept: Unkept) {
        let track = want.track;
        let from = want.from.max(line.seg_out);
        let fade = match line.begun {
            SegStart::Joined { fade } => fade,
            _ => 0,
        };
        let promised = self.timeline.promised(want.from);
        self.buffer
            .restart(from, promised.max(from + fade as u64), fade);
        let first = from as i64 - (PREROLL_RESAMPLE * f64::from(line.out)).round() as i64;
        let (_, read) = line.along_at(first).expect("a line along a map");
        let read = (read - MARGIN).max(0.0);
        let rate = f64::from(line.rate);
        // The run frame of the track's second `read`.
        let frame_of = |run: &StemRun| {
            ((read - run.start as f64 / rate) * f64::from(STEM_RATE)).floor() as i64
        };
        let reach = |run: &StemRun| (run.kept_end() + want.keep.1.max(u64::from(STEM_RATE))) as i64;
        let usable = |run: &StemRun| {
            let r = frame_of(run);
            let fine = run.path == track.info.path && !run.failed() && !run.missing();
            (fine && r >= run.kept_from() as i64 && r <= reach(run)).then_some(r as u64)
        };
        let mut found = None;
        for slot in [&mut self.run, &mut self.prestart] {
            if let Some(r) = slot.as_ref().and_then(usable) {
                found = slot.take().map(|run| (run, r));
                break;
            }
        }
        let (mut run, r) = found.unwrap_or_else(|| {
            // Stems that failed for it are not asked of the store again.
            let store = self.store.clone().filter(|_| !self.is_off(track));
            let start = (read * rate).floor() as u64;
            (
                StemRun::start(track.info.path.clone(), start, unkept, store),
                0,
            )
        });
        run.seek(r);
        run.set_keep(want.keep.0, want.keep.1);
        let read_from = run.start as f64 / rate + r as f64 / f64::from(STEM_RATE);
        let channels = usize::from(track.route.channels);
        self.stream = Some(Stream::new(line, first, channels, read_from));
        self.run = Some(run);
    }

    /// The parts of a head read along a map made until they reach output
    /// frame `target`, the first piece waiting up to `wait` for its stems;
    /// on the line of `want.track`, started for it when it is not the one
    /// made. Why its stems failed, when they just did (the next piece reads
    /// the whole mix).
    fn make_along(&mut self, want: &Want, target: u64, wait: Duration) -> Result<(), String> {
        let line = Line::of(want.track);
        if !self.stream.as_ref().is_some_and(|s| s.line.same(&line)) {
            let unkept = self.unkept_for(want).unwrap_or(Unkept::Whole);
            self.start(line, want, unkept);
        }
        let (Some(stream), Some(run)) = (self.stream.as_mut(), self.run.as_mut()) else {
            return Ok(());
        };
        run.set_keep(want.keep.0, want.keep.1);
        let buffer = &mut self.buffer;
        let mut wait = wait;
        let stopped = loop {
            if buffer.end() >= target {
                return Ok(());
            }
            match stream.make(run, wait, &mut |frame, parts| buffer.push(frame, parts)) {
                Made::Frames => wait = Duration::ZERO,
                Made::Waiting | Made::Done => return Ok(()),
                Made::Failed(why) => break Stopped::Failed(why),
                Made::Missing => break Stopped::Missing,
            }
        };
        self.give_up(want.track, want.from);
        match stopped {
            Stopped::Failed(why) => Err(why),
            Stopped::Missing => Ok(()),
        }
    }

    /// Whether the parts made are all the head's (its stems have ended).
    fn along_done(&self) -> bool {
        self.stream.as_ref().is_some_and(|s| s.done)
    }

    /// Appends the sum of the parts there for output frames `from..to` to
    /// `main` (whole frames of `channels`), within 24 bits; returns the
    /// samples that had to be clipped to it.
    fn sum_parts(&mut self, from: u64, to: u64, channels: usize, main: &mut Vec<i32>) -> u64 {
        let mut clipped = 0;
        let mut frame = Vec::with_capacity(self.buffer.width());
        main.reserve((to - from) as usize * channels);
        for o in from..to {
            frame.clear();
            self.buffer.copy(o, &mut frame);
            for c in 0..channels {
                let sum: i64 = frame
                    .iter()
                    .skip(c)
                    .step_by(channels)
                    .map(|&s| i64::from(s))
                    .sum();
                let kept = sum.clamp(-TOP - 1, TOP);
                clipped += u64::from(kept != sum);
                main.push(kept as i32);
            }
        }
        clipped
    }
}

/// A [`crate::Command::Along`] waiting for the playing head to be written
/// up to its frame.
pub(in crate::engine) struct PendingAlong {
    /// The head's segment it was given in (another one drops it).
    segment: u64,
    map: AlongMap,
    from: u64,
}

impl Engine {
    /// [`crate::Command::Along`]: the playing track is read along `map`
    /// from output frame `from` (the frame written next, if that is
    /// later), by `tempo`; it writes no further until it is. Refused (with
    /// a notice) for a track of more than two channels, off an output of
    /// lanes, and while a transition is set up or a count-in plays; a
    /// route switch or a jump under way is dropped.
    pub(in crate::engine) fn read_along(&mut self, map: Arc<TimeMap>, from: u64, tempo: TempoMode) {
        let (Some(track), Some(out)) = (self.playing.as_ref(), self.out.as_ref()) else {
            return;
        };
        let why = if track.route.channels > 2 {
            Some(format!(
                "it has {} channels, and stems are made of two at most",
                track.route.channels
            ))
        } else if out.output.lanes() <= 1 {
            Some("the output takes no stems beside it".into())
        } else if self.transition.is_some() {
            Some("a transition is set up".into())
        } else {
            (self.lead > 0).then(|| "its count-in is playing".into())
        };
        if let Some(why) = why {
            let message = format!("This track cannot be read along a time map: {why}");
            return self.notice(Some(track.index), message);
        }
        let heard = Arc::new(HeardWarp {
            map,
            origin: 0,
            a0: 0.0,
            out_rate: out.info.format.rate,
        });
        self.along_at = Some(PendingAlong {
            segment: track.segment.id,
            map: AlongMap { heard, tempo },
            from,
        });
        self.cancel_switch();
        self.pending_jump = None;
        self.drop_jump_fill();
    }

    /// The sample index the playing `track` writes no further than for a
    /// move onto a map waiting for its frame.
    pub(in crate::engine) fn along_limit(&self, track: &Track) -> Option<usize> {
        let pending = self.along_at.as_ref()?;
        let written = self.out.as_ref()?.output.written();
        let ch = usize::from(track.route.channels);
        let to = pending.from.saturating_sub(written) as usize * ch;
        Some(track.out_at + to)
    }

    /// Moves `track` onto `map` from the frame written next: what its
    /// route made from there is cut, kept a switch's fade long to crossfade
    /// the parts' sum in from, and a joined segment begins there, heard
    /// along the map.
    fn begin_along(&mut self, track: &mut Track, map: AlongMap) -> Result<(), Error> {
        let out = self.out.as_ref().ok_or(Error::Stopped)?;
        let (at, rate) = (out.output.written(), out.info.format.rate);
        let ch = usize::from(track.route.channels);
        let fade = frames_of(self.switch_fade, rate);
        // Its route's output through the crossfade.
        while track.unwritten() < fade as usize {
            if let Progress::Complete = self.produce(track, WRITE_WAIT)? {
                break;
            }
        }
        let tail = track.out_buf[track.out_at..].to_vec();
        track.out_buf.truncate(track.out_at);
        let seconds = map.heard.seconds_at(at);
        let frame = (seconds * f64::from(track.info.rate)).round() as u64;
        let fade = (tail.len() / ch.max(1)).min(fade as usize);
        track.begin_segment(at, frame, SegStart::Joined { fade });
        track.emitted = 0;
        track.resampler = None;
        track.source_done = false;
        track.ahead_done = false;
        track.restore = None;
        (track.parts_cut, track.parts_rise) = (None, 0);
        track.along = Some(Along {
            map,
            tail: tail[..fade * ch].to_vec(),
            returning: false,
        });
        // What its route queued past there is not heard; its position is
        // the map's from there.
        let index = track.index;
        self.cues.retain(|c| {
            c.at < at || !matches!(&c.event, CueEvent::Clipped { index: i, .. } if *i == index)
        });
        let heard = self.heard_along(track).expect("read along a map");
        let at = at.max(self.cues.back().map_or(0, |c| c.at));
        self.cues.push_back(Cue {
            at,
            event: CueEvent::Heard(heard),
        });
        Ok(())
    }

    /// One piece of a head read along a map: its parts made a piece past
    /// what main needs (waiting up to `wait` for its stems), and main their
    /// sum up to a seam's margin before their end (so the parts are there
    /// past any frame written), to their end once they are all made.
    pub(in crate::engine) fn step_along(
        &mut self,
        track: &mut Track,
        wait: Duration,
    ) -> Result<Progress, Error> {
        let out = self.out.as_ref().ok_or(Error::Stopped)?;
        let (written, rate) = (out.output.written(), out.info.format.rate);
        let ch = usize::from(track.route.channels);
        let margin = self.lane_margin(rate);
        let main_end = track.seg_out + track.emitted;
        let ask = self.lane_ask(track, written, 0);
        let next = next_of(&self.queue, track, self.hold, self.transition.is_none());
        let target = main_end + margin + 1;
        if let Err(reason) = self.lanes.make_along(&ask.want(track, next), target, wait) {
            let index = track.index;
            self.emit(Event::PartsStopped { index, reason });
        }
        let end = self.lanes.buffer.end();
        let done = self.lanes.along_done();
        let to = if done {
            end
        } else {
            end.saturating_sub(margin)
        };
        if to <= main_end {
            if done {
                track.source_done = true;
                return Ok(Progress::Complete);
            }
            return Ok(Progress::Busy);
        }
        let before = track.out_buf.len();
        track.mix_clips += self.lanes.sum_parts(main_end, to, ch, &mut track.out_buf);
        // The route's main crossfaded into the sum.
        if let Some(along) = track.along.as_ref() {
            let fade = along.tail.len() / ch.max(1);
            let into = (main_end - track.seg_out) as usize;
            for (i, frame) in track.out_buf[before..].chunks_exact_mut(ch).enumerate() {
                let k = into + i;
                if k >= fade {
                    break;
                }
                let w = fade_weight(k, fade);
                for (c, s) in frame.iter_mut().enumerate() {
                    let old = f64::from(along.tail[k * ch + c]);
                    *s = (old * (1.0 - w) + f64::from(*s) * w).round() as i32;
                }
            }
        }
        self.after_output(track, before);
        Ok(Progress::Busy)
    }

    /// A move onto a map whose frame is the one written next is made (one
    /// given for another segment than the head's is dropped); the head
    /// read along a map goes back onto its own route once the map has run
    /// straight a while there (it is at its own tempo then): a switch at
    /// the same format, met where the map puts it.
    pub(in crate::engine) fn along_turn(&mut self) -> Result<(), Error> {
        let Some(out) = self.out.as_ref() else {
            return Ok(());
        };
        let written = out.output.written();
        if let Some(pending) = self.along_at.take() {
            let Some(mut track) = self.playing.take() else {
                return Ok(());
            };
            let result = if track.segment.id != pending.segment {
                Ok(())
            } else if written < pending.from {
                self.along_at = Some(pending);
                Ok(())
            } else {
                self.begin_along(&mut track, pending.map)
            };
            self.playing = Some(track);
            return result;
        }
        let Some(along) = self
            .playing
            .as_mut()
            .and_then(|t| t.along.as_mut())
            .filter(|a| !a.returning)
        else {
            return Ok(());
        };
        let heard = &along.map.heard;
        let a = heard.a0
            + written.saturating_sub(heard.origin) as f64 / f64::from(heard.out_rate.max(1));
        if self.switch.is_some() || a < heard.map.settled_at() + STRAIGHT {
            return Ok(());
        }
        along.returning = true;
        self.reroute();
        Ok(())
    }
}
