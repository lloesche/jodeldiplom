//! Jumps within the playing track ([`crate::Command::Jump`]): to the start
//! of a bar, say, heard at once or at a time of the track, without a gap.
//!
//! With a jump window ([`crate::Command::JumpWindow`]) the playing track
//! keeps what it wrote for a while (its buffer is a stretch of the track,
//! in order, from a little behind what is heard to what it decoded) and
//! decodes ahead while the output is full. A jump inside that stretch only
//! moves where the buffer is written from: the output first gives back
//! what it has not played yet ([`jodel_output::Output::retract`]; the device
//! keeps playing what it holds, a few of its buffers), then the samples
//! from the target follow, their first [`crate::PlayerOptions::switch_fade`]
//! crossfaded from what would have played next, and the output clock maps
//! to the track from there on (the segment's output frame moves; its track
//! frame and everything decoded stay). A jump at a time of the track
//! writes up to that frame and jumps there; one asked for after that frame
//! was written takes it back first, and one too late for that lands the
//! same distance after the target, so the beat goes on. A target outside
//! the stretch (or on a route that keeps none) is read again
//! ([`JumpFill`]) while the track plays on, and takes over once it is
//! ready: later, never with a gap.
//!
//! No window while a transition is set up (it cuts the playing track's
//! output at its window's end), nor for a track read along a map (until it
//! is back on its own route, a jump there reads the target again). The
//! stems as parts of the output follow a jump on their own
//! ([`super::lanes`]).

use std::time::{Duration, Instant};

use super::track::{crossfade, frames_of, grid, Cue, CueEvent, Progress, SegStart, Track};
use super::{Engine, WRITE_WAIT};
use crate::error::Error;
use crate::stems::RATE as STEM_RATE;

/// When a jump is heard ([`crate::Command::Jump`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JumpAt {
    /// As soon as the output can (what it holds beyond a few device
    /// buffers is given back).
    Now,
    /// When the track reaches this many seconds (a bar's end, say).
    At(f64),
}

/// A jump waiting for its time.
#[derive(Clone, Copy, Debug)]
pub(super) struct PendingJump {
    index: usize,
    /// The track frame it is heard at.
    at: u64,
    /// The track frame it goes to.
    to: u64,
    /// The target is read again ([`JumpFill`]): it is met once ready.
    read: bool,
}

/// A reading of the playing track at a jump's target, for a target the
/// track does not hold: it decodes there (from a little before, as a
/// switch's new route does), and once it has a seam's worth from the
/// target it takes over from the track at the frame being written (or at
/// the jump's time, or as late after the target as it is late), without a
/// gap: a jump outside the window is later, never broken.
pub(super) struct JumpFill {
    track: Track,
    /// The track frame it goes to.
    to: u64,
    /// When it is heard: at once, or at this track frame.
    at: Option<u64>,
    /// Output frames before `to` still to drop.
    skip: u64,
    /// Its output from `to` on, in `track.out_buf`, and how many frames
    /// it must have before it can take over.
    need: u64,
    /// Turns alternate between it and the playing track.
    other: bool,
}

impl JumpFill {
    fn frames(&self) -> u64 {
        (self.track.out_buf.len() / usize::from(self.track.route.channels)) as u64
    }

    /// Whether it has `late` frames past its target and a seam after them
    /// (or all there is).
    fn ready(&self, late: u64) -> bool {
        self.skip == 0 && (self.frames() >= late + self.need || self.track.ahead_done)
    }
}

/// How long a jump may decode on the thread to reach its target (the
/// output plays from what it holds meanwhile).
const REACH: Duration = Duration::from_millis(150);

/// The fewest written frames a trim of the jump window drops at once.
const MIN_TRIM: usize = 4096;

impl Engine {
    /// Whether `track` keeps a jump window: there is one, and the track is
    /// on its own route (not read along a time map), nor is a transition
    /// set up.
    pub(super) fn windowed(&self, track: &Track) -> bool {
        let (back, ahead) = self.jump_window;
        (back > 0.0 || ahead > 0.0) && track.plain_route() && self.transition.is_none()
    }

    /// How many frames `track` keeps behind what it wrote, and decodes
    /// ahead of it: none without a window ([`Engine::windowed`]), but what
    /// the output may give back on a device's output of lanes, so the stem
    /// lanes may write it again with their parts ([`super::lanes`]; not
    /// while a transition is set up, which counts on the frames written,
    /// nor off the plain route). A jump back within that is met from it, as
    /// within a window.
    pub(super) fn window_frames(&self, track: &Track) -> (usize, usize) {
        let ring = self.head.out.as_ref().map_or(0, |o| o.info.ring_frames) as usize;
        if !self.windowed(track) {
            let lanes = self.head.out.as_ref().is_some_and(|o| o.output.lanes() > 1)
                && self.head.lanes.possible()
                && !self.offline
                && self.transition.is_none()
                && track.plain_route();
            return (if lanes { ring } else { 0 }, 0);
        }
        let (back, ahead) = self.jump_window;
        let rate = track.route.output_rate;
        let seconds = |s: f64| frames_of(Duration::from_secs_f64(s.max(0.0)), rate) as usize;
        // What the output may give back is kept too.
        (seconds(back) + ring, seconds(ahead))
    }

    /// How many frames (at the separator's rate) the run of `track`'s stem
    /// parts keeps behind the frame it hands on next, and takes ahead of
    /// it: the jump window, and what the output holds besides (and a
    /// second).
    pub(super) fn stem_keep(&self, track: &Track) -> (u64, u64) {
        if !self.windowed(track) {
            return (0, 0);
        }
        let (back, ahead) = self.jump_window;
        let held = self.head.out.as_ref().map_or(0.0, |o| {
            o.info.ring_frames as f64 / f64::from(o.info.format.rate.max(1))
        });
        let frames = |s: f64| (s.max(0.0) * f64::from(STEM_RATE)).round() as u64;
        (frames(back + held + 1.0), frames(ahead))
    }

    /// Drops what `track` wrote beyond the `back` frames it keeps (all of
    /// it with none kept: [`Engine::window_frames`]).
    pub(super) fn trim_written(track: &mut Track, back: usize) {
        let ch = usize::from(track.route.channels);
        let done = track.out_at / ch;
        // Trimmed in steps of a quarter of what is kept (at least
        // [`MIN_TRIM`]): each trim moves all that is kept, so seldom.
        let step = (back / 4).max(MIN_TRIM);
        if back == 0 || done > back + step {
            let drop = done.saturating_sub(back);
            track.out_buf.drain(..drop * ch);
            track.out_at -= drop * ch;
            if let Some((at, _)) = &mut track.restore {
                match at.checked_sub(drop) {
                    Some(left) => *at = left,
                    None => track.restore = None,
                }
            }
        }
    }

    /// Puts back the samples a jump's crossfade replaced once they are
    /// written (or before another jump), so the buffer holds the track.
    pub(super) fn restore_faded(track: &mut Track, force: bool) {
        let ch = usize::from(track.route.channels);
        let Some((at, saved)) = &track.restore else {
            return;
        };
        if force || track.out_at >= (at * ch + saved.len()) {
            let from = at * ch;
            if from + saved.len() <= track.out_buf.len() {
                track.out_buf[from..from + saved.len()].copy_from_slice(saved);
            }
            track.restore = None;
        }
    }

    /// The sample index `track` writes no further than for a jump waiting
    /// for its time, if one is (and, read again, has its target ready: one
    /// that is not lets the track play on, and lands late).
    pub(super) fn jump_limit(&self, track: &Track) -> Option<usize> {
        let pending = self.head.pending_jump.filter(|p| p.index == track.index)?;
        if pending.read && !self.head.jump_fill.as_ref().is_some_and(|f| f.ready(0)) {
            return None;
        }
        let ch = usize::from(track.route.channels);
        let at = stream_index(track, pending.at);
        Some(at.max(0) as usize * ch).filter(|&limit| limit >= track.out_at)
    }

    /// Whether the playing track holds track frame `to` now or will once
    /// it has decoded its window ahead.
    fn holds(&self, to: u64) -> bool {
        let Some(track) = self.head.playing.as_ref() else {
            return false;
        };
        let (_, ahead) = self.window_frames(track);
        let z = stream_frame(track, to);
        z >= buffer_start(track) && z < track.emitted as i64 + ahead as i64
    }

    /// [`crate::Command::Jump`]: the playing track goes to `to` seconds,
    /// now or at `at`.
    pub(super) fn jump_within(&mut self, index: usize, to: f64, at: JumpAt) {
        let Some(track) = self.head.playing.as_ref().filter(|t| t.index == index) else {
            return;
        };
        if self.transition.is_some() {
            return self.notice(Some(index), "No jumps while a transition is set up".into());
        }
        if track.along.is_some() {
            let message = "No jumps while the track is read along a time map";
            return self.notice(Some(index), message.into());
        }
        let rate = track.info.rate;
        let (a, _) = grid(&track.route);
        let (a, seg) = (a as i64, track.seg_frame as i64);
        // On the route's grid from the segment's track frame.
        let snap = |seconds: f64| {
            let frame = (seconds.max(0.0) * f64::from(rate)).round() as i64;
            let mut snapped = seg + ((frame - seg) as f64 / a as f64).round() as i64 * a;
            while snapped < 0 {
                snapped += a;
            }
            snapped as u64
        };
        let to_frame = snap(to);
        let at_frame = match at {
            JumpAt::At(seconds) => Some(snap(seconds)),
            JumpAt::Now => None,
        };
        self.drop_jump_fill();
        match at_frame {
            None => {
                self.head.pending_jump = None;
                self.jump_now(to_frame);
            }
            Some(at) => {
                // Not in what the track holds: read there now, met then.
                let read = !self.holds(to_frame);
                self.head.pending_jump = Some(PendingJump {
                    index,
                    at,
                    to: to_frame,
                    read,
                });
                if read {
                    self.start_jump_fill(to_frame, Some(at));
                }
            }
        }
    }

    /// A jump waiting for its time: once the track is written up to it
    /// (or past it: taken back, or landing as late after the target).
    /// Returns whether it took the turn.
    pub(super) fn jump_turn(&mut self) -> Result<bool, Error> {
        let Some(pending) = self.head.pending_jump else {
            return Ok(false);
        };
        let Some(track) = self.head.playing.as_ref() else {
            return Ok(false);
        };
        if track.index != pending.index {
            self.head.pending_jump = None;
            return Ok(false);
        }
        if pending.read {
            // Its reading meets the track ([`Engine::jump_fill_turn`]).
            return Ok(false);
        }
        let ch = usize::from(track.route.channels);
        let (a, b) = grid(&track.route);
        let at = stream_index(track, pending.at);
        if at < 0 {
            // Before what the track holds: too late to place; at once.
            self.head.pending_jump = None;
            self.jump_now(pending.to);
            return Ok(true);
        }
        let done = (track.out_at / ch) as i64;
        if at > done {
            return Ok(false);
        }
        if !self.reach(pending.to)? {
            // Not there after all: read there, met as late as it lands.
            self.head.pending_jump = Some(PendingJump {
                read: true,
                ..pending
            });
            self.start_jump_fill(pending.to, Some(pending.at));
            return Ok(true);
        }
        self.head.pending_jump = None;
        let late = (done - at) as u64;
        // Written past it: what the output has not played goes back.
        let back = self.take_back(late)?;
        let late = late - back;
        let target = pending.to + late / b * a;
        if !self.jump_to(target, late % b)? {
            self.start_jump_fill(target, None);
        }
        Ok(true)
    }

    /// Jumps to track frame `to` as soon as the output can: from what the
    /// track holds, or else from a reading of it there.
    fn jump_now(&mut self, to: u64) {
        let result = (|| {
            if !self.reach(to)? {
                return Ok(false);
            }
            // Everything the output holds beyond what it must keep.
            self.take_back(u64::MAX)?;
            self.jump_to(to, 0)
        })();
        match result {
            Ok(true) => {}
            Ok(false) => self.start_jump_fill(to, None),
            Err(error) => self.fail(None, error),
        }
    }

    /// Decodes the playing track until it holds `to` and a seam after it
    /// (for at most [`REACH`]); whether it does.
    fn reach(&mut self, to: u64) -> Result<bool, Error> {
        let Some(mut track) = self.head.playing.take() else {
            return Ok(false);
        };
        let fade = frames_of(self.switch_fade, track.route.output_rate);
        let started = Instant::now();
        let z = stream_frame(&track, to);
        let result = loop {
            if track.emitted as i64 > z + fade as i64 || started.elapsed() > REACH {
                break Ok(track.emitted as i64 > z);
            }
            match self.produce(&mut track, WRITE_WAIT) {
                Ok(Progress::Busy) => {}
                Ok(Progress::Complete) => {
                    track.ahead_done = true;
                    break Ok(track.emitted as i64 > z);
                }
                Err(error) if error.is_skippable() => break Ok(false),
                Err(error) => break Err(error),
            }
            if track.failure.is_some() {
                break Ok(false);
            }
        };
        let held = z >= buffer_start(&track);
        self.head.playing = Some(track);
        Ok(result? && held)
    }

    /// Takes back up to `frames` of what the output has not played (as
    /// much as the track still holds); returns how many.
    fn take_back(&mut self, frames: u64) -> Result<u64, Error> {
        let (Some(out), Some(track)) = (self.head.out.as_mut(), self.head.playing.as_mut()) else {
            return Ok(0);
        };
        let ch = usize::from(track.route.channels);
        let held = (track.out_at / ch) as u64;
        let n = out.output.retract(frames.min(held))?;
        track.out_at -= n as usize * ch;
        let written = out.output.written();
        let index = track.index;
        // What was queued for the frames given back is not heard.
        self.cues.retain(|c| {
            c.at < written
                || !matches!(
                    &c.event,
                    CueEvent::Clipped { index: i, .. } if *i == index
                )
        });
        Ok(n)
    }

    /// The jump itself, at the frame being written: the playing track goes
    /// on from track frame `to` (and `extra` output frames after it), its
    /// first frames crossfaded from what would have played next. False
    /// when the track does not hold the target.
    fn jump_to(&mut self, to: u64, extra: u64) -> Result<bool, Error> {
        let Some(out) = &self.head.out else {
            return Ok(false);
        };
        let written = out.output.written();
        let fade = frames_of(self.switch_fade, out.info.format.rate);
        let Some(track) = self.head.playing.as_mut() else {
            return Ok(false);
        };
        let ch = usize::from(track.route.channels);
        let frames = track.out_buf.len() / ch;
        let start = buffer_start(track);
        let z = stream_frame(track, to);
        if z < start || z + extra as i64 >= track.emitted as i64 {
            return Ok(false);
        }
        // `extra` frames after the target are heard at the frame being
        // written.
        let Some(seg_out) = written.checked_sub(extra) else {
            return Ok(false);
        };
        Self::restore_faded(track, true);
        let anchor = (z - start) as usize;
        let target = anchor + extra as usize;
        let here = track.out_at / ch;
        let next: Vec<i32> =
            track.out_buf[here * ch..(here + fade as usize).min(frames) * ch].to_vec();
        let n = (next.len() / ch).min(frames - target);
        let span = target * ch..(target + n) * ch;
        track.restore = Some((target, track.out_buf[span.clone()].to_vec()));
        crossfade(&next[..n * ch], &mut track.out_buf[span], ch);
        track.out_at = target * ch;
        // What is written from here on is the buffer as it stands, once
        // the crossfade is put back.
        track.rewritable_from = written + n as u64;
        // The segment starts at the target now (as a seek's would): the
        // buffer and the decoder go on as they are.
        track.begin_segment(seg_out, to, SegStart::Joined { fade: n });
        track.emitted = (frames - anchor) as u64;
        jumped(&mut self.cues, track, written);
        self.after_jump();
        Ok(true)
    }

    /// After a jump: a switch under way decoded for the old place, so it
    /// starts again.
    fn after_jump(&mut self) {
        if self.head.switch.is_some() {
            self.cancel_switch();
            self.reroute();
        }
    }
}

/// Queues what a jump of `track` heard from output frame `written` says:
/// the clock's new map of the track, and the jump.
fn jumped(cues: &mut std::collections::VecDeque<Cue>, track: &Track, written: u64) {
    let heard = super::Heard {
        index: track.index,
        info: track.info.clone(),
        route: track.route.clone(),
        seg_out: track.seg_out,
        seg_frame: track.seg_frame,
        warp: None,
    };
    let seconds = heard.seconds_at(written);
    let index = track.index;
    let last = cues.back().map_or(0, |c| c.at);
    cues.push_back(Cue {
        at: written.max(last),
        event: CueEvent::Heard(heard),
    });
    cues.push_back(Cue {
        at: written.max(last),
        event: CueEvent::Jumped { index, seconds },
    });
}

/// The stream frame of track frame `frame`: output frames from the
/// segment's track frame (negative before it: what the buffer kept from
/// before a jump).
fn stream_frame(track: &Track, frame: u64) -> i64 {
    let (a, b) = grid(&track.route);
    (frame as i64 - track.seg_frame as i64).div_euclid(a as i64) * b as i64
}

/// The stream frame of the buffer's first frame.
fn buffer_start(track: &Track) -> i64 {
    let ch = usize::from(track.route.channels);
    track.emitted as i64 - (track.out_buf.len() / ch) as i64
}

/// The frame index in `track`'s buffer of track frame `frame` (negative:
/// before it).
fn stream_index(track: &Track, frame: u64) -> i64 {
    stream_frame(track, frame) - buffer_start(track)
}

impl Engine {
    /// Reads the playing track at track frame `to` for a jump heard at
    /// once (`at` none) or at track frame `at` ([`JumpFill`]).
    pub(super) fn start_jump_fill(&mut self, to: u64, at: Option<u64>) {
        let Some(live) = self.head.playing.as_ref() else {
            return;
        };
        let (index, route, rate) = (live.index, live.route.clone(), live.info.rate);
        let (a, b) = grid(&route);
        let back = super::preroll(&route, rate).div_ceil(a).min(to / a);
        let fade = frames_of(self.switch_fade, route.output_rate);
        match self.prepare_at(index, to - back * a, Some(route)) {
            Ok(mut track) => {
                track.pending = true;
                self.head.jump_fill = Some(Box::new(JumpFill {
                    track,
                    to,
                    at,
                    skip: back * b,
                    need: fade.max(1),
                    other: false,
                }));
            }
            Err(error) => {
                self.head.pending_jump = None;
                self.notice(
                    Some(index),
                    format!("The jump did not happen: the track did not open again ({error})"),
                );
            }
        }
    }

    /// Drops a reading for a jump (and the jump waiting on it).
    pub(super) fn drop_jump_fill(&mut self) {
        if let Some(fill) = self.head.jump_fill.take() {
            super::drop_later(fill.track);
            if self.head.pending_jump.is_some_and(|p| p.read) {
                self.head.pending_jump = None;
            }
        }
    }

    /// The reading's turn: it decodes on the turns a switch's new route
    /// has, and takes over once it is ready and the jump's time has come.
    /// Returns whether it took the turn.
    pub(super) fn jump_fill_turn(&mut self) -> Result<bool, Error> {
        let Some(fill) = self.head.jump_fill.as_ref() else {
            return Ok(false);
        };
        let Some(live) = self
            .head
            .playing
            .as_ref()
            .filter(|t| t.index == fill.track.index)
        else {
            self.drop_jump_fill();
            return Ok(false);
        };
        let ch = usize::from(live.route.channels);
        let done = (live.out_at / ch) as i64;
        // How late it would land if it took over now.
        let late = match fill.at {
            None => 0,
            Some(at) => {
                let at = stream_index(live, at);
                if at > done && fill.ready(0) {
                    // The track writes up to it first.
                    return Ok(false);
                }
                (done - at).max(0) as u64
            }
        };
        if !fill.ready(late) {
            // Its turns: those a switch's new route would have, and every
            // other one besides (a jump is waited for).
            let mut fill = self.head.jump_fill.take().expect("checked");
            fill.other = !fill.other;
            if !fill.other && !self.reading_turn() {
                self.head.jump_fill = Some(fill);
                return Ok(false);
            }
            match self.produce(&mut fill.track, Duration::ZERO) {
                Ok(Progress::Busy) => {}
                Ok(Progress::Complete) => fill.track.ahead_done = true,
                Err(error) if error.is_skippable() => {
                    let index = fill.track.index;
                    super::drop_later(fill.track);
                    self.head.pending_jump = None;
                    self.notice(Some(index), format!("The jump did not happen: {error}"));
                    return Ok(true);
                }
                Err(error) => return Err(error),
            }
            fill.track.held.clear();
            let made = fill.frames();
            let drop = fill.skip.min(made);
            fill.track.out_buf.drain(..drop as usize * ch);
            fill.skip -= drop;
            self.head.jump_fill = Some(fill);
            return Ok(true);
        }
        // Ready: the output gives back what it has not played (as late as
        // the jump is, or all it can), and the reading takes over.
        let back = self.take_back(if fill_at_once(self) { u64::MAX } else { late })?;
        let late = late.saturating_sub(back);
        self.meet_jump_fill(late)?;
        Ok(true)
    }

    /// Whether a reading has the thread: paused, the output holding at
    /// least half its buffer, or it took nothing of the last write.
    fn reading_turn(&self) -> bool {
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
                .is_some_and(|at| at.elapsed() < super::SWITCH_TURN)
    }

    /// The reading takes over at the frame being written, `late` frames
    /// after its target, crossfaded from what the track would have played.
    fn meet_jump_fill(&mut self, late: u64) -> Result<(), Error> {
        let (Some(fill), Some(mut live)) = (self.head.jump_fill.take(), self.head.playing.take())
        else {
            return Ok(());
        };
        self.head.pending_jump = None;
        let written = self.head.out.as_ref().map_or(0, |o| o.output.written());
        let ch = usize::from(live.route.channels);
        let fade = frames_of(self.switch_fade, live.route.output_rate) as usize;
        // What the track would have played next, a seam's worth.
        while live.unwritten() < fade && !live.ahead_done && live.failure.is_none() {
            match self.produce(&mut live, WRITE_WAIT) {
                Ok(Progress::Busy) => {}
                Ok(Progress::Complete) => live.ahead_done = true,
                Err(error) if error.is_skippable() => break,
                Err(error) => {
                    self.head.playing = Some(live);
                    return Err(error);
                }
            }
        }
        Self::restore_faded(&mut live, true);
        let next: Vec<i32> = live.out_buf[live.out_at..]
            .iter()
            .take(fade * ch)
            .copied()
            .collect();
        let mut track = fill.track;
        let late = (late as usize).min(track.out_buf.len() / ch);
        track.out_buf.drain(..late * ch);
        let n = next.len().min(track.out_buf.len());
        crossfade(&next[..n], &mut track.out_buf[..n], ch);
        track.out_at = 0;
        let joined = SegStart::Joined { fade: n / ch };
        track.begin_segment(written - late as u64, fill.to, joined);
        track.emitted = (late + track.out_buf.len() / ch) as u64;
        track.pending = false;
        track.held.clear();
        track.clip_base = track.clipped();
        track.clips = None;
        track.restore = None;
        jumped(&mut self.cues, &track, written);
        super::drop_later(live);
        self.head.playing = Some(track);
        self.after_jump();
        Ok(())
    }

    /// The queue was reordered ([`super::queue`]): a jump waiting, and a
    /// reading for one, follow their track (gone, they go).
    pub(super) fn reindex_jumps(&mut self, map: &dyn Fn(usize) -> Option<usize>) {
        if let Some(p) = &mut self.head.pending_jump {
            match map(p.index) {
                Some(i) => p.index = i,
                None => self.head.pending_jump = None,
            }
        }
        if let Some(fill) = &mut self.head.jump_fill {
            match map(fill.track.index) {
                Some(i) => fill.track.index = i,
                None => self.drop_jump_fill(),
            }
        }
    }
}

/// Whether the reading under way is for a jump heard at once.
fn fill_at_once(engine: &Engine) -> bool {
    engine
        .head
        .jump_fill
        .as_ref()
        .is_some_and(|f| f.at.is_none())
}
