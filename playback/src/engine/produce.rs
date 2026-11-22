//! Making a track's output and writing it: a plain route decoded (and
//! resampled), a stems route mixed from its stem run, and a count-in's
//! silence before a track.

use std::time::{Duration, Instant};

use crate::error::Error;
use crate::rate::Stage;
use crate::resample::Resampler;
use crate::stems::{Pull, StemRun};

use super::track::{Cue, CueEvent, Progress, Track};
use super::{Engine, STEM_FRAMES, WRITE_WAIT};

/// Drops what `track` just made (from sample `before` on) while it has
/// frames to discard (a stems route's preroll after a jump).
fn discard_made(track: &mut Track, before: usize) {
    if track.discard == 0 {
        return;
    }
    let channels = usize::from(track.route.channels);
    let made = ((track.out_buf.len() - before) / channels) as u64;
    let drop = track.discard.min(made);
    track
        .out_buf
        .drain(before..before + drop as usize * channels);
    track.discard -= drop;
}

/// The output frame of `resampler`'s stream that a stems route ends on.
/// The stems come at the separator's rate, so their end is only known to
/// a frame of that rate; resampled to the output's, they end up to a few
/// frames off the track's end (at 192 kHz up to two short or over, by the
/// frame the stems start at). Where the track's stated length agrees with
/// them within that rounding, the stream ends on its stated end, where the
/// track's other routes end (a file a few frames shorter or longer than it
/// says is padded or cut to it); past that, where its stems end. They end
/// there too for a track read along a time map, one with no stated length,
/// or a segment that starts past it (and a route at the separator's rate
/// has no resampler: it ends where its stems do).
fn stems_end(track: &Track, resampler: &Resampler) -> u64 {
    let design = resampler.design();
    let (from, to) = (design.from_rate(), design.to_rate());
    let stems = crate::resample::output_frames(from, to, resampler.input_frames());
    if track.warp.is_some() {
        return stems;
    }
    let Some(left) = track
        .info
        .frames
        .and_then(|total| total.checked_sub(track.seg_frame))
    else {
        return stems;
    };
    let heard = crate::resample::output_frames(track.info.rate.max(1), to, left);
    // The stream counts the frames before the segment's start too (a
    // switch's run-up, a jump's preroll), and those of them still to be
    // dropped.
    let Some(end) = (heard + resampler.output_frames() + track.discard).checked_sub(track.emitted)
    else {
        return stems;
    };
    // Where the stems end and where they start (a run taken up mid-way)
    // are each within half a frame of the separator's: one of its frames
    // at the output's rate. The two ends rounded to output frames add one.
    let slack = u64::from(to.div_ceil(from)) + 1;
    if end.abs_diff(stems) <= slack {
        end
    } else {
        stems
    }
}

impl Engine {
    /// Writes some of a count-in's silence ([`Engine::lead`]); while the
    /// output is full, the track decodes ahead behind it, so it is ready
    /// (its stems separated) when the silence ends.
    pub(super) fn write_lead(&mut self) -> Result<(), Error> {
        let out = self.out.as_mut().ok_or(Error::Stopped)?;
        let channels = usize::from(out.info.format.channels);
        let frames = self.lead.min(STEM_FRAMES as u64) as usize;
        let silence = vec![0i32; frames * channels];
        let took = out.output.write(&silence, Some(WRITE_WAIT))?;
        self.lead -= took as u64;
        if took == 0 {
            self.stalled = Some(Instant::now());
            if let Some(mut track) = self.playing.take() {
                // Ahead by a few seconds at most: the rest as it plays.
                let ahead = u64::from(track.route.output_rate) * 3;
                let result = if (track.unwritten() as u64) < ahead {
                    self.produce(&mut track, Duration::ZERO).map(|_| ())
                } else {
                    Ok(())
                };
                self.playing = Some(track);
                result?;
            }
        }
        Ok(())
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
            // A jump waiting for its time: written no further than it.
            let limit = self
                .jump_limit(track)
                .map_or(track.out_buf.len(), |l| l.min(track.out_buf.len()));
            let mut frames = 0;
            if track.out_at < limit {
                let out = self.out.as_mut().ok_or(Error::Stopped)?;
                let channels = usize::from(out.info.format.channels);
                frames = out
                    .output
                    .write(&track.out_buf[track.out_at..limit], Some(WRITE_WAIT))?;
                self.stalled = (frames == 0).then(Instant::now);
                track.out_at += frames * channels;
                Self::restore_faded(track, false);
            }
            // The output is full: the window's stretch ahead is decoded (a
            // stems route takes what its stems have separated).
            if frames == 0 && ahead > 0 && !track.ahead_done && track.unwritten() < ahead {
                if let Progress::Complete = self.produce(track, Duration::ZERO)? {
                    track.ahead_done = true;
                }
            }
            if frames == 0 {
                if let Some(run) = track.stems.as_mut() {
                    run.fetch(0, Duration::ZERO);
                }
            }
            return Ok(Progress::Busy);
        }
        // Written: what the window keeps stays, the rest goes.
        Self::trim_written(track, back);
        if let Some(error) = track.failure.take() {
            let end = self.out.as_ref().map_or(0, |o| o.output.written());
            self.cues.push_back(Cue {
                at: end,
                event: CueEvent::Skipped {
                    index: track.index,
                    error,
                },
            });
            return Ok(Progress::Complete);
        }
        self.produce(track, WRITE_WAIT)
    }

    /// Decodes one piece of `track` and appends its output to `out_buf`
    /// (a stem mix waits up to `wait` for its stems).
    pub(super) fn produce(&mut self, track: &mut Track, wait: Duration) -> Result<Progress, Error> {
        if track.route.stage == Stage::Stems {
            return self.step_stems(track, wait);
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

    /// One piece of a [`Stage::Stems`] route: the stems that are on, summed,
    /// resampled or requantized into the output buffer (waiting up to
    /// `wait` for the stems).
    pub(super) fn step_stems(
        &mut self,
        track: &mut Track,
        wait: Duration,
    ) -> Result<Progress, Error> {
        let Some(separator) = track.separator.or(self.separator) else {
            track.stems_error = Some("no separator".into());
            return Ok(Progress::Busy);
        };
        let before = track.out_buf.len();
        if track.stems_done {
            if let Some(mut resampler) = track.resampler.take_if(|r| !r.is_finished()) {
                let end = stems_end(track, &resampler);
                resampler.finish_at(end, &mut track.out_buf);
                track.resampler = Some(resampler);
                discard_made(track, before);
                self.after_output(track, before);
                return Ok(Progress::Busy);
            }
            return Ok(Progress::Complete);
        }
        let keep = self.stem_keep(track);
        let run = match track.stems.take() {
            Some(run) => run,
            None => match self.prestart.take() {
                Some(run) if run.path == track.info.path && run.start == track.seg_frame => run,
                other => {
                    self.prestart = other;
                    match self.take_shadow(track) {
                        Some(run) => run,
                        None => StemRun::start(
                            track.info.path.clone(),
                            track.seg_frame,
                            separator,
                            self.stem_store.clone(),
                        ),
                    }
                }
            },
        };
        let run = track.stems.insert(run);
        run.set_keep(keep.0, keep.1);
        track.wide_buf.clear();
        let origin = run.start as f64 / f64::from(track.info.rate.max(1));
        let gains = track.gains.as_ref().map(|g| (g, origin));
        // A change at a time of this track: the stems before it until its
        // frame of the run, the track's time at the change's output frame
        // as it is heard now (a track decoded ahead for a switch is heard
        // on the playing one's timeline).
        let (mix, change) = match self.stems_change {
            Some(c) if c.index == track.index => {
                let timeline =
                    |t: &Track| (t.seg_frame, t.info.rate, t.seg_out, t.route.output_rate);
                let (frame, rate, seg_out, out_rate) = match self.playing.as_ref() {
                    Some(playing) if track.pending => timeline(playing),
                    _ => (
                        track.seg_frame,
                        track.info.rate,
                        track.seg_out,
                        track.route.output_rate,
                    ),
                };
                let seconds = frame as f64 / f64::from(rate.max(1))
                    + (c.out as f64 - seg_out as f64) / f64::from(out_rate.max(1));
                let at = ((seconds - origin) * f64::from(crate::stems::RATE))
                    .round()
                    .max(0.0) as u64;
                (c.before, Some((at, self.settings.stems)))
            }
            _ => (self.settings.stems, None),
        };
        let pull = run.mix(STEM_FRAMES, mix, change, gains, &mut track.wide_buf, wait);
        let read_all = run.read_all();
        if let Some(warp) = track.warp.as_mut() {
            match pull {
                Pull::Mixed => warp.read(&mut track.wide_buf),
                // What the reader holds comes out before the end.
                Pull::End => {
                    warp.end();
                    warp.read(&mut track.wide_buf);
                }
                _ => {}
            }
            if matches!(pull, Pull::End) && !track.wide_buf.is_empty() {
                match track.resampler.as_mut() {
                    Some(resampler) => resampler.process_float(&track.wide_buf, &mut track.out_buf),
                    None => {
                        let quantize = &mut track.requantizer;
                        track
                            .out_buf
                            .extend(track.wide_buf.iter().map(|&y| quantize.quantize(y)));
                    }
                }
                self.after_output(track, before);
                track.stems_done = true;
                return Ok(Progress::Busy);
            }
        }
        match pull {
            Pull::Mixed => {
                match track.resampler.as_mut() {
                    Some(resampler) => resampler.process_float(&track.wide_buf, &mut track.out_buf),
                    None => {
                        let quantize = &mut track.requantizer;
                        track
                            .out_buf
                            .extend(track.wide_buf.iter().map(|&y| quantize.quantize(y)));
                    }
                }
                discard_made(track, before);
                self.after_output(track, before);
            }
            Pull::Waiting => {}
            Pull::End => track.stems_done = true,
            Pull::Failed(why) => track.stems_error = Some(why),
        }
        // The next track's stems start once this file is read.
        let next = self.queue.get(track.index + 1);
        let stale = self.prestart.as_ref().is_none_or(|p| Some(&p.path) != next);
        if read_all && stale {
            let store = self.stem_store.clone();
            self.prestart = next.map(|path| StemRun::start(path.clone(), 0, separator, store));
        }
        Ok(Progress::Busy)
    }

    /// Counts the output frames just made and queues a clip count when the
    /// source's requantization or the resampler clipped (at most four a
    /// second of output).
    fn after_output(&mut self, track: &mut Track, before: usize) {
        let start = track.seg_out + track.emitted;
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
