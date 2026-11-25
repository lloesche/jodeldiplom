//! Moving the playing track to a new route without a break (new
//! settings, a head back off its map): the new route, at the open
//! output's format, reads ahead while the old one plays, and the two meet
//! at an exact frame, crossfaded.

use std::collections::VecDeque;
use std::time::Duration;

use crate::error::Error;
use crate::rate::Route;

use super::drop::drop_later;
use super::track::{
    crossfade, frames_of, gcd, grid, preroll, rescale, Cue, CueEvent, Progress, SegStart, Splice,
    StartAt, Switch, Track,
};
use super::{Engine, SWITCH_TURN, WRITE_WAIT};

impl Engine {
    /// Moves the track being played to its route with the settings' headroom
    /// (at the format it plays at): a [`Switch`], decoded ahead and met at
    /// an exact frame. The track waiting for the output to play out takes
    /// its new route at once (it is not heard yet).
    pub(super) fn reroute(&mut self) {
        // A head read along a map (or waiting to be: a transition's too)
        // holds its route: its end routes it again. While a transition is
        // prepared or mixed, its playing track is where the next one is
        // linked to: only a way back off a map moves it.
        let leaving = self
            .head
            .playing
            .as_ref()
            .is_some_and(|t| t.along.is_some());
        if self.along_holds_route() || (self.transition_holds_route() && !leaving) {
            return;
        }
        if self.head.switch.as_ref().is_some_and(|s| s.spliced) {
            // The old route ends at the frame the new one begins: once
            // the new one is playing, it is routed again.
            self.head.reroute_later = true;
            return;
        }
        let Some(track) = &self.head.playing else {
            return self.reroute_next();
        };
        if self.head.out.is_none() {
            return;
        }
        let (index, current) = (track.index, track.route.clone());
        // A track read along a map (a head's, a transition's) is not on its
        // route as the settings make it once that is over, though the route
        // is the same: it moves to a fresh one.
        let leftover = !track.plain_route();
        let route = current.clone().with_headroom(self.settings.headroom);
        if route == current && !leftover {
            self.cancel_switch();
            return;
        }
        if self
            .head
            .switch
            .as_ref()
            .is_some_and(|s| s.track.route == route)
        {
            return;
        }
        self.cancel_switch();
        if let Err(error) = self.begin_switch(route) {
            self.notice(Some(index), not_reopened(&error, &current));
        }
    }

    /// [`Engine::reroute`] for the track waiting for the output to play
    /// out (to open again at the target's format): it takes its new route
    /// now, opened again where it waits.
    fn reroute_next(&mut self) {
        let Some(next) = &self.next else {
            return;
        };
        let (index, frame, current) = (next.index, next.seg_frame, next.route.clone());
        let route = current.clone().with_headroom(self.settings.headroom);
        if route == current {
            return;
        }
        match self.open_track(index, StartAt::Frame(frame), Some(route)) {
            Ok(track) => self.next = Some(track),
            Err(error) => self.notice(Some(index), not_reopened(&error, &current)),
        }
    }

    /// Starts the playing track's move to `route`: the track opens on it a
    /// little before the frame being written ([`preroll`]), on the grid
    /// both routes meet exactly on, and decodes ahead from there.
    pub(super) fn begin_switch(&mut self, route: Route) -> Result<(), Error> {
        let out = self.head.out.as_ref().ok_or(Error::Stopped)?;
        let written = out.output.written();
        let old = self.head.playing.as_ref().ok_or(Error::Stopped)?;
        let (a_old, _) = grid(&old.route);
        let (a_new, _) = grid(&route);
        let step = a_old / gcd(a_old, a_new) * a_new;
        // A track read along a map (a head's, a transition's) is where its
        // map, as heard, puts it.
        let along = self.heard_along(old).map(|h| h.frame_at(written));
        let at = along.unwrap_or_else(|| {
            old.seg_frame
                + rescale(
                    written.saturating_sub(old.seg_out),
                    old.route.output_rate,
                    old.route.input_rate,
                )
        });
        let want = at.saturating_sub(preroll(&route, old.info.rate));
        // On the grid from the old segment's start.
        let first = old.seg_frame % step;
        let start = if want >= first {
            first + (want - first) / step * step
        } else {
            first
        };
        let index = old.index;
        let mut track = self.open_track(index, StartAt::Frame(start), Some(route))?;
        track.pending = true;
        self.head.switch = Some(Switch {
            track,
            grid: step,
            retried: false,
            done: false,
            spliced: false,
        });
        Ok(())
    }

    /// Drops the switch under way, if any (in the background: closing its
    /// file and freeing its buffers can take a while).
    pub(super) fn cancel_switch(&mut self) {
        if let Some(switch) = self.head.switch.take() {
            drop_later(switch.track);
        }
    }

    /// The switch's turn, when it has one: the new route decodes ahead
    /// while the output holds at least half its buffer, is paused, or took
    /// nothing of the last write ([`SWITCH_TURN`]), and meets the old one
    /// once it has caught up; once the old one is written up to where they
    /// meet, the new one takes over. Returns whether it took the turn
    /// (otherwise the old route writes or decodes).
    pub(super) fn work_switch(&mut self) -> Result<bool, Error> {
        let Some(switch) = &self.head.switch else {
            return Ok(false);
        };
        if switch.spliced {
            if self.head.playing.as_ref().map_or(0, Track::unwritten) > 0 {
                return Ok(false);
            }
            self.hand_over();
            return Ok(true);
        }
        let Some(out) = &self.head.out else {
            return Ok(false);
        };
        let buffered = out.output.written().saturating_sub(out.clock.position());
        let ring = out.info.ring_frames;
        let turn = self.paused
            || self.offline
            || (ring > 0 && buffered.saturating_mul(2) >= ring)
            || self
                .head
                .stalled
                .is_some_and(|at| at.elapsed() < SWITCH_TURN);
        if !turn {
            return Ok(false);
        }
        let mut switch = self.head.switch.take().expect("checked");
        if !switch.done {
            match self.produce(&mut switch.track, Duration::ZERO) {
                Ok(Progress::Busy) => {}
                Ok(Progress::Complete) => switch.done = true,
                Err(error) if error.is_skippable() => {
                    // What it decoded before the error plays, then the
                    // error, as on its own.
                    self.broke(&mut switch.track, error);
                    switch.done = true;
                }
                Err(error) => return Err(error),
            }
        }
        self.head.switch = Some(switch);
        self.try_splice()?;
        Ok(true)
    }

    /// The switch's new route failed before it was heard (`why`): it
    /// decodes once more from its start (a decoder may refuse to start
    /// mid-stream), or else the track keeps its route.
    fn switch_failed(&mut self, switch: Switch, why: &str) {
        let index = switch.track.index;
        let route = switch.track.route.clone();
        let start = switch.track.seg_frame;
        let first = start % switch.grid;
        drop_later(switch.track);
        if !switch.retried && start > first {
            if let Ok(mut track) =
                self.open_track(index, StartAt::Frame(first), Some(route.clone()))
            {
                track.pending = true;
                self.head.switch = Some(Switch {
                    track,
                    retried: true,
                    done: false,
                    spliced: false,
                    ..switch
                });
                return;
            }
        }
        let current = self
            .head
            .playing
            .as_ref()
            .map_or_else(String::new, |t| t.route.describe());
        let message = format!(
            "{} failed on this track ({why}); keeping {current} for this track",
            route.describe()
        );
        self.notice(Some(index), message);
    }

    /// Meets the old route with the switch's new one if it has caught up
    /// ([`Engine::splice`]); drops the switch when the track ends first.
    pub(super) fn try_splice(&mut self) -> Result<(), Error> {
        let Some(mut switch) = self.head.switch.take() else {
            return Ok(());
        };
        let Some(mut old) = self.head.playing.take() else {
            return Ok(());
        };
        let result = self.splice(&mut old, &mut switch);
        self.head.playing = Some(old);
        match result? {
            Splice::Wait | Splice::Done => self.head.switch = Some(switch),
            Splice::Cancel => {
                // A new route that failed before where they would meet.
                match switch.track.failure.take() {
                    Some(error) => self.switch_failed(switch, &error.to_string()),
                    None => drop_later(switch.track),
                }
            }
        }
        Ok(())
    }

    /// Where the old route and the new one meet: at the first track frame
    /// on the grid ([`Switch::grid`]) heard at or after the frame being
    /// written, so that output frame `at` on the old route and frame `j` of
    /// the new one are the same track frame exactly. Once the new route has
    /// a crossfade's worth of output from there, the old one is cut at
    /// `at` and the two crossfade from there.
    fn splice(&mut self, old: &mut Track, switch: &mut Switch) -> Result<Splice, Error> {
        /// Output frames a pending track keeps before what it may still
        /// meet the old route at, before it drops them.
        const KEEP: u64 = 65_536;
        let Some(out) = &self.head.out else {
            return Ok(Splice::Wait);
        };
        let written = out.output.written();
        let fade = frames_of(self.switch_fade, old.route.output_rate);
        let (a_old, b_old) = grid(&old.route);
        let (a_new, b_new) = grid(&switch.track.route);
        let step_out = switch.grid / a_old * b_old;
        let start = switch.track.seg_frame;
        // An old track read along a map (a head along its own, or along a
        // transition's map) has no grid with the
        // output: it meets the new one on the first frame of the new
        // route's grid its map reaches at or after output frame `from`, at
        // the output frame the map puts it on (to half a frame; the
        // crossfade takes the rest).
        let along = self.heard_along(old);
        // The grid point at or after output frame `from`, as (track frame,
        // output frame), and the new route's output frame for it.
        let meet = |from: u64| {
            if let Some(h) = &along {
                let warp = h.warp.as_ref().expect("heard along a map");
                let rate = f64::from(old.info.rate.max(1));
                let at = |frame: u64| warp.output_frame_at(frame as f64 / rate);
                // The first grid frame the map puts at or after `from` (one
                // a hair before it is the frame before it, played already).
                let from = from.max(old.seg_out);
                let mut frame =
                    start + h.frame_at(from).saturating_sub(start).div_ceil(a_new) * a_new;
                while at(frame) < from {
                    frame += a_new;
                }
                let j = (frame - start) / a_new * b_new;
                return (frame, at(frame), j);
            }
            let n = from.saturating_sub(old.seg_out).div_ceil(step_out);
            let frame = old.seg_frame + n * switch.grid;
            let j = frame.saturating_sub(start) / a_new * b_new;
            (frame, old.seg_out + n * step_out, j)
        };
        let (frame, at, j) = meet(written);
        if old.info.frames.is_some_and(|total| frame >= total) {
            return Ok(Splice::Cancel);
        }
        let new = &mut switch.track;
        let channels = usize::from(new.route.channels);
        let base = new.emitted - (new.out_buf.len() / channels) as u64;
        if (new.emitted < j + fade || new.emitted <= j) && !switch.done {
            // Not caught up: drop what is before the earliest frame it can
            // still meet the old route at (it may not have made that much
            // yet).
            let drop = j.min(new.emitted);
            if drop > base + KEEP {
                new.out_buf.drain(..(drop - base) as usize * channels);
            }
            return Ok(Splice::Wait);
        }
        if new.emitted <= j {
            // The new route ends before it.
            return Ok(Splice::Cancel);
        }
        let cross = fade.min(new.emitted - j);
        // The old route's output up to there (and through the crossfade).
        while old.seg_out + old.emitted < at + cross {
            if old.failure.is_some() {
                return Ok(Splice::Cancel);
            }
            match self.produce(old, WRITE_WAIT) {
                Ok(Progress::Busy) => {}
                Ok(Progress::Complete) => return Ok(Splice::Cancel),
                Err(error) if error.is_skippable() => {
                    self.broke(old, error);
                    return Ok(Splice::Cancel);
                }
                Err(error) => return Err(error),
            }
        }
        // The old route stops at `at`.
        let old_channels = usize::from(old.route.channels);
        let keep = old.out_at + (at - written) as usize * old_channels;
        let tail = old.out_buf[keep..keep + cross as usize * old_channels].to_vec();
        old.out_buf.truncate(keep);
        let index = old.index;
        self.cues.retain(|c| {
            c.at < at
                || !matches!(
                    &c.event,
                    CueEvent::Clipped { index: i, .. } if *i == index
                ) && !matches!(&c.event, CueEvent::Started(s, _) if s.index == index)
        });
        // The new one starts there, crossfaded from the old one.
        let new = &mut switch.track;
        new.out_buf.drain(..(j - base) as usize * channels);
        new.out_at = 0;
        crossfade(&tail, &mut new.out_buf[..tail.len()], channels);
        new.emitted -= j;
        new.pending = false;
        new.clip_base = new.clipped();
        new.clips = None;
        // Its cues from there.
        let mut held = VecDeque::new();
        for cue in std::mem::take(&mut new.held) {
            match cue.event {
                CueEvent::Clipped { .. } => {}
                event => held.push_back(Cue {
                    at: cue.at.saturating_sub(j),
                    event,
                }),
            }
        }
        new.held = held;
        // The same segment, met on its line: the stem parts go on, or (at
        // another gain) start again crossfaded as main is.
        let joined = SegStart::Joined {
            fade: cross as usize,
        };
        new.continue_segment(old.segment, at, frame, joined);
        let cue = self.started(new);
        self.cues.push_back(cue);
        self.release(new);
        switch.spliced = true;
        Ok(Splice::Done)
    }

    /// The old route is written up to where the new one begins: the new
    /// one plays on.
    fn hand_over(&mut self) {
        let Some(switch) = self.head.switch.take() else {
            return;
        };
        // The old track, written to where the new one begins.
        if let Some(old) = self.head.playing.take() {
            drop_later(old);
        }
        self.head.playing = Some(switch.track);
        if std::mem::take(&mut self.head.reroute_later) {
            self.reroute();
        }
    }
}

/// The notice for a track whose new route did not open (`error`): it
/// keeps `current`.
fn not_reopened(error: &Error, current: &Route) -> String {
    format!(
        "Could not reopen this track: {error}; keeping {} for this track",
        current.describe()
    )
}
