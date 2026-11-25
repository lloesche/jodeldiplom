//! Opening and starting tracks: the output opened at a route's format, a
//! track installed as the one played, jumps to a track and seeks within
//! it, the next track run into gaplessly or through a reopen, stopping,
//! and the end of the queue.

use std::time::Instant;

use jodel_output::{OpenRequest, Output, StreamFormat};

use crate::error::Error;
use crate::rate::{choose_route, FileFacts, Route};
use crate::source::Source;

use super::api::{Event, PlayState, TrackStart};
use super::track::{Cue, CueEvent, OpenOutput, SegStart, StartAt, Track};
use super::{Engine, OPEN_BUSY_STEP, OPEN_BUSY_WAIT};

impl Engine {
    /// Opens track `index` at `seconds` and chooses its route.
    pub(super) fn prepare(&mut self, index: usize, seconds: f64) -> Result<Track, Error> {
        self.prepare_with(index, StartAt::Seconds(seconds), None)
    }

    /// Opens track `index` at `start` on `route`, or on the route the
    /// settings choose for it.
    fn prepare_with(
        &mut self,
        index: usize,
        start: StartAt,
        route: Option<Route>,
    ) -> Result<Track, Error> {
        let mut track = self.open_track(index, route)?;
        let frame = match start {
            StartAt::Seconds(seconds) => {
                (seconds.max(0.0) * f64::from(track.info.rate)).round() as u64
            }
            StartAt::Share(share) => track
                .info
                .frames
                .map_or(0, |n| (n as f64 * share).round() as u64),
        };
        track.start_at(frame)?;
        Ok(track)
    }

    /// [`Engine::prepare_with`] at track frame `frame`.
    pub(super) fn prepare_at(
        &mut self,
        index: usize,
        frame: u64,
        route: Option<Route>,
    ) -> Result<Track, Error> {
        let mut track = self.open_track(index, route)?;
        track.start_at(frame)?;
        Ok(track)
    }

    /// Opens track `index` at its start on `route`, or on the route the
    /// settings choose for it.
    fn open_track(&mut self, index: usize, route: Option<Route>) -> Result<Track, Error> {
        let path = self.queue[index].clone();
        let source = Source::open(&path)?;
        let info = source.info().clone();
        let route = match route {
            Some(route) => route,
            None => {
                let caps = self.target_caps()?;
                choose_route(&FileFacts::from(&*info), &caps, &self.settings)?
            }
        };
        Track::new(index, info, route, source, self.segments.clone())
    }

    /// Opens an output, trying again while the device is busy, for up to
    /// [`OPEN_BUSY_WAIT`].
    fn open_target(&mut self, request: &OpenRequest) -> Result<Box<dyn Output>, Error> {
        let deadline = Instant::now() + OPEN_BUSY_WAIT;
        loop {
            match self.head.target.open(request) {
                Err(jodel_output::Error::DeviceBusy) if Instant::now() < deadline => {
                    std::thread::sleep(OPEN_BUSY_STEP);
                }
                result => return Ok(result?),
            }
        }
    }

    /// Opens an output at `route`'s format (with lanes where the target
    /// takes them), paused when `paused`, in place of the one open.
    pub(super) fn open_output(&mut self, route: &Route, paused: bool) -> Result<(), Error> {
        self.close_output();
        let mut request = route.open_request();
        // Main and the four stems, where the target takes them (with no
        // separator nor stems kept, a head read along a map has the whole
        // mix as a stem).
        let lanes = 1 + crate::stems::STEMS as u16;
        if self.head.target.lanes() >= lanes {
            request = request.lanes(lanes);
        }
        let mut output = self.open_target(&request)?;
        if paused {
            output.pause()?;
        }
        let info = output.info().clone();
        let clock = output.clock();
        self.head.lanes.open(info.format);
        self.head.out = Some(OpenOutput {
            output,
            info,
            clock,
        });
        Ok(())
    }

    pub(super) fn close_output(&mut self) {
        if let Some(mut out) = self.head.out.take() {
            let _ = out.output.close();
        }
        self.head.lanes.flushed(0);
    }

    pub(super) fn started(&self, track: &Track) -> Cue {
        let out = self.head.out.as_ref().expect("an output is open");
        let format: StreamFormat = out.info.format;
        let bit_perfect = out.info.exclusive && track.route.lossless && track.route.fits(&format);
        let start = TrackStart {
            index: track.index,
            head: self.head.id,
            info: track.info.clone(),
            route: track.route.clone(),
            output: out.info.clone(),
            device: self.head.target.name(),
            bit_perfect,
            output_frame: track.seg_out,
            from_frame: track.seg_frame,
        };
        Cue {
            at: track.seg_out,
            event: CueEvent::Started(Box::new(start), track.heard()),
        }
    }

    /// Makes `track` the one being played now: reuses the output when it
    /// fits (flushing what it holds; paused, it plays again unless the
    /// player stays paused), otherwise opens one.
    fn install(&mut self, mut track: Track) -> Result<(), Error> {
        let fits = self
            .head
            .out
            .as_ref()
            .is_some_and(|o| track.route.fits(&o.info.format));
        if fits {
            let out = self.head.out.as_mut().expect("checked");
            out.output.flush()?;
            if !self.paused && out.output.is_paused() {
                out.output.resume()?;
            }
            self.head.lanes.flushed(out.output.written());
        } else {
            self.open_output(&track.route, self.paused)?;
        }
        let out = self.head.out.as_ref().expect("an output is open");
        let written = out.output.written();
        // A count-in's silence first: the track is heard after it, and
        // starts to be as it begins.
        let rate = f64::from(out.info.format.rate);
        self.head.lead = (std::mem::take(&mut self.lead_in) * rate).round() as u64;
        track.begin_segment(written + self.head.lead, track.seg_frame, SegStart::Cold);
        let mut cue = self.started(&track);
        cue.at = written;
        self.cues.push_back(cue);
        self.release(&mut track);
        self.head.playing = Some(track);
        self.track_installed();
        Ok(())
    }

    /// Queues `track`'s held cues, from its `seg_out`.
    pub(super) fn release(&mut self, track: &mut Track) {
        for mut cue in track.held.drain(..) {
            cue.at = (cue.at + track.seg_out).max(self.cues.back().map_or(0, |c| c.at));
            self.cues.push_back(cue);
        }
    }

    /// Plays track `index` at `seconds` now, skipping tracks that fail to
    /// open or route.
    pub(super) fn jump(&mut self, index: usize, seconds: f64) {
        self.jump_on(index, StartAt::Seconds(seconds), None);
    }

    /// [`Engine::jump`] to `start` of the track, on `route` when one is
    /// given.
    pub(super) fn jump_on(&mut self, index: usize, start: StartAt, route: Option<Route>) {
        let previous = self.heard.as_ref().map(|h| h.index);
        self.clear_timeline();
        if let Some(previous) = previous {
            if previous != index {
                self.emit(Event::TrackEnded {
                    index: previous,
                    finished: false,
                });
                self.heard = None;
            }
        }
        let mut index = index;
        let mut start = start;
        let mut route = route;
        loop {
            if index >= self.queue.len() {
                self.close_output();
                if self.heard.take().is_some() || previous.is_some() {
                    self.emit(Event::QueueEnded);
                }
                self.paused = false;
                self.emit(Event::State(PlayState::Stopped));
                return;
            }
            match self.prepare_with(index, start, route.take()) {
                Ok(track) => {
                    if let Err(error) = self.install(track) {
                        return self.fail(Some(index), error);
                    }
                    self.fire_cues();
                    self.emit_play_state();
                    return;
                }
                Err(error) if error.is_skippable() => {
                    self.emit(Event::Error {
                        index: Some(index),
                        error,
                        skipped: true,
                    });
                    index += 1;
                    start = StartAt::Seconds(0.0);
                }
                Err(error) => return self.fail(Some(index), error),
            }
        }
    }

    pub(super) fn seek(&mut self, seconds: f64) {
        // A count-in under way ends: the track plays from where it went.
        self.head.lead = 0;
        self.head.pending_jump = None;
        self.drop_jump_fill();
        let Some(heard) = self.heard.clone() else {
            return;
        };
        // A switch under way, or one waiting for the output to open again,
        // has the route the track is going to: the seek's flush hides the
        // change.
        let switching = self
            .head
            .switch
            .take()
            .filter(|s| s.track.index == heard.index)
            .map(|s| s.track.route)
            .or_else(|| {
                self.next
                    .as_ref()
                    .filter(|t| t.index == heard.index)
                    .map(|t| t.route.clone())
            });
        // A transition is off (the output it handed over from closes); a
        // track read along a map starts afresh on its own route.
        self.abandon_transition("a seek");
        let transitioned = self
            .head
            .playing
            .as_ref()
            .is_some_and(|t| t.along.is_some());
        if transitioned {
            self.jump_on(heard.index, StartAt::Seconds(seconds), None);
            if let Some(h) = &self.heard {
                let at = h.seconds_at(h.seg_out);
                self.emit(Event::Seeked {
                    index: h.index,
                    seconds: at,
                });
            }
            return;
        }
        let in_place = switching.is_none()
            && self
                .head
                .playing
                .as_ref()
                .is_some_and(|t| t.index == heard.index);
        if !in_place {
            // The track keeps the route it is heard on, or goes on to the
            // one it is switching to.
            let route = switching.unwrap_or_else(|| heard.route.clone());
            self.jump_on(heard.index, StartAt::Seconds(seconds), Some(route));
            if let Some(h) = &self.heard {
                let at = h.seconds_at(h.seg_out);
                self.emit(Event::Seeked {
                    index: h.index,
                    seconds: at,
                });
            }
            return;
        }
        let frame = (seconds.max(0.0) * f64::from(heard.info.rate)).round() as u64;
        let result = (|| {
            let out = self.head.out.as_mut().ok_or(Error::Stopped)?;
            out.output.flush()?;
            let written = out.output.written();
            let track = self.head.playing.as_mut().expect("checked");
            track.restart_at(frame)?;
            track.begin_segment(written, track.seg_frame, SegStart::Cold);
            self.head.lanes.flushed(written);
            Ok::<_, Error>((written, track.seg_frame))
        })();
        match result {
            Ok((seg_out, seg_frame)) => {
                self.cues.clear();
                self.next = None;
                self.finishing = false;
                self.tail = None;
                let heard = self.heard.as_mut().expect("checked");
                heard.seg_out = seg_out;
                heard.seg_frame = seg_frame;
                let (index, at) = (heard.index, heard.seconds_at(seg_out));
                self.emit(Event::Seeked { index, seconds: at });
                self.report_position(true);
            }
            Err(error) if error.is_skippable() => {
                self.emit(Event::Error {
                    index: Some(heard.index),
                    error,
                    skipped: true,
                });
                self.jump(heard.index + 1, 0.0);
            }
            Err(error) => self.fail(Some(heard.index), error),
        }
    }

    /// Stops playback and closes the output.
    pub(super) fn stop(&mut self, announce: bool) {
        let was_active = self.head.out.is_some() || self.head.playing.is_some();
        self.every_lanes().for_each(|l| l.stop());
        self.clear_timeline();
        self.close_output();
        self.paused = false;
        if let Some(heard) = self.heard.take() {
            if announce {
                self.emit(Event::TrackEnded {
                    index: heard.index,
                    finished: false,
                });
            }
        }
        if announce && was_active {
            self.emit(Event::State(PlayState::Stopped));
        }
    }

    pub(super) fn fail(&mut self, index: Option<usize>, error: Error) {
        self.emit(Event::Error {
            index,
            error,
            skipped: false,
        });
        self.stop(true);
    }

    /// Installs the track that waited for the output to play out. One that
    /// switched to another format goes back to its old route, at the same
    /// frame, when its output does not open (a notice says so).
    pub(super) fn install_next(&mut self, mut track: Track) -> Result<(), Error> {
        let fallback = track.fallback.take();
        let (index, frame, route) = (track.index, track.seg_frame, track.route.clone());
        let Err(error) = self.install(track) else {
            return Ok(());
        };
        let Some(back) = fallback else {
            return Err(error);
        };
        let again = self
            .prepare_at(index, frame, Some(back.clone()))
            .and_then(|track| self.install(track));
        if again.is_err() {
            return Err(error);
        }
        self.fire_cues();
        let message = format!(
            "{} did not open for {}: {error}; keeping {} for this track",
            self.head.target.name(),
            route.describe(),
            back.describe()
        );
        self.notice(Some(index), message);
        Ok(())
    }

    /// `track` is fully written: queue its end and go on with the next.
    pub(super) fn finish(&mut self, track: Track) -> Result<(), Error> {
        // Inside a transition's window the next track takes over.
        let Some(track) = self.transition_takes_over(track) else {
            return Ok(());
        };
        // A switch of this track has nothing left to switch.
        self.cancel_switch();
        let end = self.head.out.as_ref().map_or(0, |o| o.output.written());
        // A cue mapped past the last output frame is never heard after the
        // end; it applies at the end at the latest.
        for cue in self.cues.iter_mut() {
            cue.at = cue.at.min(end);
        }
        self.cues.push_back(Cue {
            at: end,
            event: CueEvent::Ended { index: track.index },
        });
        // Holding: the next track waits to be started; playback stops
        // once this one is heard to its end.
        if self.hold && track.index + 1 < self.queue.len() {
            self.cues.push_back(Cue {
                at: end,
                event: CueEvent::Held {
                    index: track.index + 1,
                },
            });
            self.tail = None;
            self.finishing = true;
            return Ok(());
        }
        self.advance(track.index + 1, end)
    }

    /// Goes on with the first track from `index` that opens, at output
    /// frame `end`: gaplessly when it fits the open output.
    pub(super) fn advance(&mut self, mut index: usize, end: u64) -> Result<(), Error> {
        loop {
            if index >= self.queue.len() {
                self.cues.push_back(Cue {
                    at: end,
                    event: CueEvent::QueueEnded,
                });
                self.tail = index.checked_sub(1);
                self.finishing = true;
                return Ok(());
            }
            match self.prepare(index, 0.0) {
                Ok(mut track) => {
                    let fits = self
                        .head
                        .out
                        .as_ref()
                        .is_some_and(|o| track.route.fits(&o.info.format));
                    if fits {
                        track.begin_segment(end, track.seg_frame, SegStart::Cold);
                        let cue = self.started(&track);
                        self.cues.push_back(cue);
                        self.head.playing = Some(track);
                        self.track_installed();
                    } else {
                        self.next = Some(track);
                        self.finishing = true;
                    }
                    return Ok(());
                }
                Err(error) if error.is_skippable() => {
                    self.cues.push_back(Cue {
                        at: end,
                        event: CueEvent::Skipped { index, error },
                    });
                    index += 1;
                }
                Err(error) => return Err(error),
            }
        }
    }
}
