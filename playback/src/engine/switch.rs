//! Moving the playing track to a new route without a break (new
//! settings, a stems change): the new route reads ahead while the old one
//! plays, and the two meet at an exact frame, crossfaded at the same
//! output format or through a reopened output at another.

use std::collections::VecDeque;
use std::time::Duration;

use crate::error::Error;
use crate::rate::{Route, Stage};

use super::api::Event;
use super::track::{
    crossfade, drop_later, fade, frames_of, gcd, grid, preroll, rescale, Cue, CueEvent, Progress,
    Splice, Switch, Track,
};
use super::{Engine, SWITCH_PREFILL, SWITCH_TURN, WRITE_WAIT};

impl Engine {
    /// Moves the track being played to the route the settings choose for
    /// it: a [`Switch`], decoded ahead and met at an exact frame. The track
    /// waiting for the output to play out takes its new route at once (it
    /// is not heard yet).
    pub(super) fn reroute(&mut self) {
        // A transition under way has its tracks where it needs them; its
        // end routes them again.
        if self.transition_running() {
            return;
        }
        if self.switch.as_ref().is_some_and(|s| s.spliced) {
            // The old route ends at the frame the new one begins: once
            // the new one is playing, it is routed again.
            self.reroute_later = true;
            return;
        }
        let Some(track) = &self.playing else {
            return self.reroute_next();
        };
        let (index, info, current) = (track.index, track.info.clone(), track.route.clone());
        // A track a transition made (read along its time map, its stems
        // faded by it, or on its stand-in separator) is not on its route as
        // the settings make it once the transition is over, though the
        // route is the same: it moves to a fresh one.
        let leftover = self.transition.is_none()
            && (track.warp.is_some() || track.gains.is_some() || track.separator.is_some());
        let keeping = |why: String| format!("{why}; keeping {} for this track", current.describe());
        let route = match self.route_for(&info) {
            Ok(route) => route,
            Err(why) => {
                self.cancel_switch();
                return self.notice(Some(index), keeping(why));
            }
        };
        if route == current && !leftover {
            self.cancel_switch();
            return;
        }
        if self.switch.as_ref().is_some_and(|s| s.track.route == route) {
            return;
        }
        self.cancel_switch();
        if let Err(error) = self.begin_switch(route) {
            let message = keeping(format!("Could not reopen this track: {error}"));
            self.notice(Some(index), message);
        }
    }

    /// [`Engine::reroute`] for the track waiting for the output to play
    /// out: it takes its new route now, and when that fits the open output,
    /// it follows what was written without a break.
    fn reroute_next(&mut self) {
        let Some(next) = &self.next else {
            return;
        };
        let (index, frame, info, current, fallback) = (
            next.index,
            next.seg_frame,
            next.info.clone(),
            next.route.clone(),
            next.fallback.clone(),
        );
        let keeping = |why: String| format!("{why}; keeping {} for this track", current.describe());
        let route = match self.route_for(&info) {
            Ok(route) if route == current => return,
            Ok(route) => route,
            Err(why) => return self.notice(Some(index), keeping(why)),
        };
        let mut track = match self.prepare_at(index, frame, Some(route)) {
            Ok(track) => track,
            Err(error) => {
                let message = keeping(format!("Could not reopen this track: {error}"));
                return self.notice(Some(index), message);
            }
        };
        // A switch's new route waits for the old one, faded out, to play
        // out: it opens the output again rather than start unfaded after
        // it, and can go back to the old route.
        let switched = fallback.is_some();
        track.fallback = fallback;
        let open = self
            .out
            .as_ref()
            .map(|o| (o.info.format, o.output.written()));
        match open {
            Some((format, end)) if !switched && track.route.fits(&format) => {
                // Nothing was written after the end the track waits at.
                track.seg_out = end;
                let cue = self.started(&track);
                self.cues.push_back(cue);
                self.next = None;
                self.finishing = false;
                self.playing = Some(track);
            }
            _ => self.next = Some(track),
        }
    }

    /// Starts the playing track's move to `route`: the track opens on it a
    /// little before the frame being written ([`preroll`]), on the grid
    /// both routes meet exactly on, and decodes ahead from there.
    pub(super) fn begin_switch(&mut self, route: Route) -> Result<(), Error> {
        let out = self.out.as_ref().ok_or(Error::Stopped)?;
        let written = out.output.written();
        let fits = route.fits(&out.info.format);
        let old = self.playing.as_ref().ok_or(Error::Stopped)?;
        let (a_old, _) = grid(&old.route);
        let (a_new, _) = grid(&route);
        let step = a_old / gcd(a_old, a_new) * a_new;
        // A track read along a map (the playing one along a transition's
        // glide) is where its map, as heard, puts it.
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
        let mut track = self.prepare_at(index, start, Some(route))?;
        track.pending = true;
        self.switch = Some(Switch {
            track,
            grid: step,
            fits,
            retried: false,
            done: false,
            spliced: false,
        });
        Ok(())
    }

    /// Drops the switch under way, if any (in the background: closing its
    /// file and freeing its buffers can take a while).
    pub(super) fn cancel_switch(&mut self) {
        if let Some(mut switch) = self.switch.take() {
            // The stems it took over are kept for the next one.
            let file = self.playing.as_ref().is_some_and(|t| {
                t.route.stage != Stage::Stems && t.info.path == switch.track.info.path
            });
            if file && self.shadow.is_none() {
                self.shadow = switch.track.stems.take();
            }
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
        let Some(switch) = &self.switch else {
            return Ok(false);
        };
        if switch.spliced {
            if self.playing.as_ref().map_or(0, Track::unwritten) > 0 {
                return Ok(false);
            }
            self.hand_over();
            return Ok(true);
        }
        let Some(out) = &self.out else {
            return Ok(false);
        };
        let buffered = out.output.written().saturating_sub(out.clock.position());
        let ring = out.info.ring_frames;
        let turn = self.paused
            || self.offline
            || (ring > 0 && buffered.saturating_mul(2) >= ring)
            || self.stalled.is_some_and(|at| at.elapsed() < SWITCH_TURN);
        if !turn {
            return Ok(false);
        }
        let mut switch = self.switch.take().expect("checked");
        let before = switch.track.emitted;
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
        if let Some(reason) = switch.track.stems_error.take() {
            // The stems did not start: the track plays on as it is, and
            // every track does until the mix changes.
            self.stems_failed = true;
            self.prestart = None;
            let index = switch.track.index;
            self.emit(Event::StemsStopped { index, reason });
            return Ok(true);
        }
        let waiting = switch.track.route.stage == Stage::Stems
            && switch.track.emitted == before
            && !switch.done;
        self.switch = Some(switch);
        self.try_splice()?;
        Ok(!waiting)
    }

    /// The switch's new route failed before it was heard (`why`): it
    /// decodes once more from the start of the track (a decoder may refuse
    /// to start mid-stream), or else the track keeps its route.
    fn switch_failed(&mut self, switch: Switch, why: &str) {
        let index = switch.track.index;
        let route = switch.track.route.clone();
        let start = switch.track.seg_frame;
        let first = start % switch.grid;
        drop_later(switch.track);
        if !switch.retried && start > first {
            if let Ok(mut track) = self.prepare_at(index, first, Some(route.clone())) {
                track.pending = true;
                self.switch = Some(Switch {
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
        let Some(mut switch) = self.switch.take() else {
            return Ok(());
        };
        let Some(mut old) = self.playing.take() else {
            return Ok(());
        };
        let result = self.splice(&mut old, &mut switch);
        self.playing = Some(old);
        match result? {
            Splice::Wait | Splice::Done => self.switch = Some(switch),
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
    /// written (and, at another format, a fade's length after it), so that
    /// output frame `at` on the old route and frame `j` of the new one are
    /// the same track frame exactly. Once the new route has its output
    /// from there (a crossfade's worth, or at another format enough to
    /// start the new output), the old one is cut at `at`; at the same
    /// format the two crossfade from there, at another the old one fades
    /// out before it and the new one fades in.
    fn splice(&mut self, old: &mut Track, switch: &mut Switch) -> Result<Splice, Error> {
        /// Output frames a pending track keeps before what it may still
        /// meet the old route at, before it drops them.
        const KEEP: u64 = 65_536;
        let Some(out) = &self.out else {
            return Ok(Splice::Wait);
        };
        let written = out.output.written();
        let new_rate = switch.track.route.output_rate;
        let fade_old = frames_of(self.switch_fade, old.route.output_rate);
        let fade_new = frames_of(self.switch_fade, new_rate);
        let (a_old, b_old) = grid(&old.route);
        let (a_new, b_new) = grid(&switch.track.route);
        let step_out = switch.grid / a_old * b_old;
        let start = switch.track.seg_frame;
        // An old track read along a map (the playing one along a
        // transition's glide, called off) has no grid with the output: it
        // meets the new one at output frame `from` itself, where its map
        // puts it (on the new route's grid; the crossfade takes the rest).
        let along = self.heard_along(old).cloned();
        // The grid point at or after output frame `from`, as (track frame,
        // output frame), and the new route's output frame for it.
        let meet = |from: u64| {
            if let Some(h) = &along {
                let frame = start + h.frame_at(from).saturating_sub(start) / a_new * a_new;
                let j = (frame - start) / a_new * b_new;
                return (frame, from.max(old.seg_out), j);
            }
            let n = from.saturating_sub(old.seg_out).div_ceil(step_out);
            let frame = old.seg_frame + n * switch.grid;
            let j = frame.saturating_sub(start) / a_new * b_new;
            (frame, old.seg_out + n * step_out, j)
        };
        let guard = if switch.fits { 0 } else { fade_old };
        let (frame, at, j) = meet(written + guard);
        if old.info.frames.is_some_and(|total| frame >= total) {
            return Ok(Splice::Cancel);
        }
        let new = &mut switch.track;
        let channels = usize::from(new.route.channels);
        let base = new.emitted - (new.out_buf.len() / channels) as u64;
        let lead = if switch.fits {
            fade_new
        } else {
            frames_of(SWITCH_PREFILL, new_rate).max(fade_new)
        };
        if (new.emitted < j + lead || new.emitted <= j) && !switch.done {
            // Not caught up: drop what is before the earliest frame it can
            // still meet the old route at.
            // (It may not have made that much yet.)
            let (_, _, earliest) = meet(written);
            let drop = earliest.min(new.emitted);
            if drop > base + KEEP {
                new.out_buf.drain(..(drop - base) as usize * channels);
            }
            return Ok(Splice::Wait);
        }
        if new.emitted <= j {
            // The new route ends before it.
            return Ok(Splice::Cancel);
        }
        let ahead = new.emitted - j;
        let cross = if switch.fits { fade_old.min(ahead) } else { 0 };
        // The old route's output up to there (and through the crossfade).
        while old.seg_out + old.emitted < at + cross {
            if old.failure.is_some() || old.stems_error.is_some() {
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
        if !switch.fits {
            let from = keep - fade_old as usize * old_channels;
            fade(&mut old.out_buf[from..], old_channels, false);
        }
        let index = old.index;
        self.cues.retain(|c| {
            c.at < at
                || !matches!(
                    &c.event,
                    CueEvent::Clipped { index: i, .. } if *i == index
                ) && !matches!(&c.event, CueEvent::Started(s, _) if s.index == index)
        });
        // The new one starts there.
        let new = &mut switch.track;
        new.out_buf.drain(..(j - base) as usize * channels);
        new.out_at = 0;
        if switch.fits {
            crossfade(&tail, &mut new.out_buf[..tail.len()], channels);
        } else {
            let rise = fade_new.min(ahead) as usize * channels;
            fade(&mut new.out_buf[..rise], channels, true);
        }
        new.emitted -= j;
        new.seg_frame = frame;
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
        if switch.fits {
            new.seg_out = at;
            let cue = self.started(new);
            self.cues.push_back(cue);
            self.release(new);
        } else {
            new.seg_out = 0;
            new.fallback = Some(old.route.clone());
        }
        switch.spliced = true;
        Ok(Splice::Done)
    }

    /// The old route is written up to where the new one begins: the new
    /// one plays on (or, at another format, waits for the output to play
    /// out and open again).
    fn hand_over(&mut self) {
        let Some(switch) = self.switch.take() else {
            return;
        };
        // The old track, written to where the new one begins (its stems
        // kept for the track on its file again).
        if let Some(mut old) = self.playing.take() {
            if old.index == switch.track.index && switch.track.route.stage != Stage::Stems {
                self.shadow = old.stems.take();
            }
            drop_later(old);
        }
        if switch.fits {
            self.playing = Some(switch.track);
        } else {
            self.next = Some(switch.track);
            self.finishing = true;
        }
        if std::mem::take(&mut self.reroute_later) {
            self.reroute();
        }
    }

    /// The stems of `track` stopped (`why`): it goes on, on its usual
    /// route, from the frame after what was written, with nothing flushed
    /// (in the same output when it fits, else after this one plays out);
    /// the stems rest until the mix changes.
    pub(super) fn stems_stopped(&mut self, track: Track, why: &str) {
        self.cancel_switch();
        let index = track.index;
        let channels = usize::from(track.route.channels);
        let unwritten = (track.out_buf.len() - track.out_at) / channels;
        let written = track.emitted.saturating_sub(unwritten as u64);
        let frame =
            track.seg_frame + rescale(written, track.route.output_rate, track.route.input_rate);
        let rate = f64::from(track.info.rate.max(1));
        drop(track);
        self.stems_failed = true;
        self.prestart = None;
        self.shadow = None;
        self.emit(Event::StemsStopped {
            index,
            reason: why.to_owned(),
        });
        let end = self.out.as_ref().map_or(0, |o| o.output.written());
        match self.prepare(index, frame as f64 / rate) {
            Ok(mut track) => {
                let fits = self
                    .out
                    .as_ref()
                    .is_some_and(|o| track.route.fits(&o.info.format));
                if fits {
                    track.seg_out = end;
                    let cue = self.started(&track);
                    self.cues.push_back(cue);
                    self.playing = Some(track);
                } else {
                    self.next = Some(track);
                    self.finishing = true;
                }
            }
            Err(error) if error.is_skippable() => {
                self.cues.push_back(Cue {
                    at: end,
                    event: CueEvent::Skipped { index, error },
                });
                if let Err(error) = self.advance(index + 1, end) {
                    self.fail(None, error);
                }
            }
            Err(error) => self.fail(Some(index), error),
        }
    }
}
