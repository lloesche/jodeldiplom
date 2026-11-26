//! Making a track's output and writing it: a plain route decoded (and
//! resampled), a head read along a map ([`super::along`]), and a count-in's
//! silence before a track.

use std::time::{Duration, Instant};

use jodel_output::lanes::lane_width;

use crate::error::Error;

use super::cues::CueEvent;
use super::track::{Progress, Track};
use super::{Engine, SILENCE_FRAMES, WRITE_WAIT};

impl Engine {
    /// Writes some of a count-in's silence ([`Engine::lead`]); meanwhile the
    /// track's stem parts are made from its first frame, and while the
    /// output is full the track decodes ahead behind it, so it is ready
    /// when the silence ends.
    pub(super) fn write_lead(&mut self) -> Result<(), Error> {
        let out = self.head.out.as_mut().ok_or(Error::Stopped)?;
        // An output of lanes gets silence on every lane, and no parts.
        let width = lane_width(out.info.format.channels, out.output.lanes());
        let frames = self.head.lead.min(SILENCE_FRAMES as u64) as usize;
        // One zeroed buffer, grown to the widest write once.
        let samples = frames * width;
        if self.silence.len() < samples {
            self.silence.resize(samples, 0);
        }
        let took = out
            .output
            .write(&self.silence[..samples], Some(WRITE_WAIT))?;
        self.head.lead -= took as u64;
        self.with_playing(|e, track| e.lanes_ahead(track));
        if took == 0 {
            self.head.stalled = Some(Instant::now());
            let ahead = self.with_playing(|e, track| {
                // Ahead by a few seconds at most: the rest as it plays.
                let ahead = u64::from(track.route.output_rate) * 3;
                if (track.unwritten() as u64) < ahead {
                    e.produce(track, Duration::ZERO).map(|_| ())
                } else {
                    Ok(())
                }
            });
            ahead.unwrap_or(Ok(()))?;
        }
        Ok(())
    }

    /// How long a write of `track`, or a wait for its stems, may block: a
    /// moment, but not at all for a track decoded ahead of being heard on
    /// a device (a transition's next track: the playing one is not to wait
    /// for it).
    pub(super) fn wait_for(&self, track: &Track) -> Duration {
        if track.pending && !self.offline {
            Duration::ZERO
        } else {
            WRITE_WAIT
        }
    }

    /// Writes `track`'s output from `out_at` up to sample `limit` and
    /// returns the frames the output took: as it is, or on an output of
    /// lanes with its stem parts beside it ([`Engine::write_lanes`]).
    pub(super) fn write_track(&mut self, track: &mut Track, limit: usize) -> Result<usize, Error> {
        let wait = self.wait_for(track);
        let out = self.head.out.as_mut().ok_or(Error::Stopped)?;
        if out.output.lanes() > 1 {
            return self.write_lanes(track, limit, wait);
        }
        let channels = usize::from(out.info.format.channels);
        let frames = out
            .output
            .write(&track.out_buf[track.out_at..limit], Some(wait))?;
        track.out_at += frames * channels;
        Ok(frames)
    }

    /// `track` failed mid-way (`error`, one that costs only the track):
    /// what it decoded plays (with the resampler's tail), then the error is
    /// reported and the next track goes on.
    pub(super) fn broke(&mut self, track: &mut Track, error: Error) {
        track.source_done = true;
        if track.resampler.as_ref().is_some_and(|r| !r.is_finished()) {
            let before = track.out_buf.len();
            if let Some(resampler) = &mut track.resampler {
                resampler.finish(&mut track.out_buf);
            }
            self.after_output(track, before);
        }
        track.failure = Some(error);
    }

    pub(super) fn step(&mut self, track: &mut Track) -> Result<Progress, Error> {
        let (back, ahead) = self.window_frames(track);
        if track.out_at < track.out_buf.len() {
            // A jump or a move onto a map waiting for its frame, or a
            // transition's window end: written no further than it.
            let limit = [
                self.jump_limit(track),
                self.along_limit(track),
                self.transition_limit(track),
            ]
            .into_iter()
            .flatten()
            .min()
            .map_or(track.out_buf.len(), |l| l.min(track.out_buf.len()));
            let mut frames = 0;
            if track.out_at < limit {
                frames = self.write_track(track, limit)?;
                self.head.stalled = (frames == 0).then(Instant::now);
                Self::restore_faded(track, false);
            }
            // The output is full: the window's stretch ahead is decoded.
            if frames == 0 && ahead > 0 && !track.ahead_done && track.unwritten() < ahead {
                if let Progress::Complete = self.produce(track, Duration::ZERO)? {
                    track.ahead_done = true;
                }
            }
            return Ok(Progress::Busy);
        }
        // Written: what the window keeps stays, the rest goes.
        Self::trim_written(track, back);
        // One decoded ahead of being heard (a transition's next track) has
        // its failure said by what it was for.
        if let Some(error) = track.failure.take_if(|_| track.pending) {
            return Err(error);
        }
        if let Some(error) = track.failure.take() {
            let end = self.head.out.as_ref().map_or(0, |o| o.output.written());
            let index = track.index;
            self.cues.push(end, CueEvent::Skipped { index, error });
            return Ok(Progress::Complete);
        }
        let wait = self.wait_for(track);
        self.produce(track, wait)
    }

    /// Decodes one piece of `track` and appends its output to `out_buf`
    /// (a head read along a map waits up to `wait` for its stems).
    pub(super) fn produce(&mut self, track: &mut Track, wait: Duration) -> Result<Progress, Error> {
        if track.along.is_some() {
            return self.step_along(track, wait);
        }
        self.step_plain(track)
    }

    /// One read of plain PCM: resampled into the output buffer when the
    /// route has a resampler (unquantized when the file's samples are not
    /// 24-bit integers: lossy, floating-point, 32-bit, so that they are
    /// requantized once), appended to it as read otherwise. Once the source has ended the
    /// resampler is flushed, then the track is complete.
    fn step_plain(&mut self, track: &mut Track) -> Result<Progress, Error> {
        let before = track.out_buf.len();
        if !track.source_done {
            let frames = match &mut track.resampler {
                Some(resampler) if !track.info.fits_24_bits() => {
                    track.wide_buf.clear();
                    let frames = track.source.read_unquantized(&mut track.wide_buf)?;
                    resampler.process_float(&track.wide_buf, &mut track.out_buf);
                    frames
                }
                Some(resampler) => {
                    track.read_buf.clear();
                    let frames = track.source.read(&mut track.read_buf)?;
                    resampler.process(&track.read_buf, &mut track.out_buf);
                    frames
                }
                None => track.source.read(&mut track.out_buf)?,
            };
            if frames > 0 {
                self.after_output(track, before);
                return Ok(Progress::Busy);
            }
            track.source_done = true;
        }
        match &mut track.resampler {
            Some(resampler) if !resampler.is_finished() => {
                resampler.finish(&mut track.out_buf);
                self.after_output(track, before);
                Ok(Progress::Busy)
            }
            _ => Ok(Progress::Complete),
        }
    }

    /// Counts the output frames just made and queues a clip count when the
    /// source's requantization or the resampler clipped (at most four a
    /// second of output).
    pub(super) fn after_output(&mut self, track: &mut Track, before: usize) {
        let start = track.line.seg_out + track.emitted;
        let made = (track.out_buf.len() - before) / usize::from(track.route.channels);
        track.emitted += made as u64;
        let clipped = track.clipped().saturating_sub(track.clip_base);
        let spacing = u64::from(track.route.output_rate / 4);
        let due = match track.clips {
            None => clipped > 0,
            Some((count, at)) => clipped > count && start >= at + spacing,
        };
        if due {
            let event = CueEvent::Clipped {
                index: track.index,
                samples: clipped,
            };
            let at = self.push_cue(track, start, event);
            track.clips = Some((clipped, at));
        }
    }
}
