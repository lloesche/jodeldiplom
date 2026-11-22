//! A track as the engine plays it ([`Track`]): its source, route and
//! buffers; what is heard of it ([`Heard`]); a route switch under way
//! ([`Switch`]); the cues that wait for the output clock ([`Cue`]); and the
//! small helpers on frames and fades they share.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use jodel_output::{Clock, Output, StreamInfo};

use crate::error::Error;
use crate::rate::{Route, Stage};
use crate::resample::{Requantizer, Resampler};
use crate::source::{Source, TrackInfo};
use crate::stems::{Gains, SeparatorSource, StemRun};
use crate::transition::TransitionState;
use crate::warp::TimeMap;

use super::api::TrackStart;
use super::warping::Warping;
use super::{PREROLL_RESAMPLE, PREROLL_STEMS};

/// Where in a track playback starts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum StartAt {
    /// Seconds in.
    Seconds(f64),
    /// A share of its length (its start when the length is not known).
    Share(f64),
}

/// The track being heard: what maps the output clock to the track.
#[derive(Clone)]
pub(super) struct Heard {
    pub(super) index: usize,
    pub(super) info: Arc<TrackInfo>,
    pub(super) route: Route,
    /// The output frame of the segment's first frame.
    pub(super) seg_out: u64,
    /// The track frame at the segment's start.
    pub(super) seg_frame: u64,
    /// The track is read along a time map (a transition's next track):
    /// its position comes from that, not from a straight line.
    pub(super) warp: Option<Arc<HeardWarp>>,
}

/// How a track read along a time map is heard: its output frame `origin`
/// is the map's time `a0`, and output frames go at `out_rate`.
#[derive(Clone)]
pub(super) struct HeardWarp {
    pub(super) map: Arc<TimeMap>,
    pub(super) origin: u64,
    pub(super) a0: f64,
    pub(super) out_rate: u32,
}

impl HeardWarp {
    /// The track's seconds at `output_frame`.
    pub(super) fn seconds_at(&self, output_frame: u64) -> f64 {
        let since = output_frame.saturating_sub(self.origin) as f64;
        self.map
            .at(self.a0 + since / f64::from(self.out_rate.max(1)))
            .max(0.0)
    }
}

/// `frames` at `from` Hz in frames at `to` Hz, rounded down.
pub(super) fn rescale(frames: u64, from: u32, to: u32) -> u64 {
    (u128::from(frames) * u128::from(to) / u128::from(from.max(1))) as u64
}

pub(super) fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

/// A route's smallest exact step: `a` track frames are `b` output frames
/// (a track frame that many from the segment's start is heard at an exact
/// output frame).
pub(super) fn grid(route: &Route) -> (u64, u64) {
    let (input, output) = (u64::from(route.input_rate), u64::from(route.output_rate));
    let g = gcd(input, output);
    (input / g, output / g)
}

/// `duration` in frames at `rate`.
pub(super) fn frames_of(duration: Duration, rate: u32) -> u64 {
    (duration.as_secs_f64() * f64::from(rate)).round() as u64
}

/// How many track frames before the switch `route` starts decoding.
pub(super) fn preroll(route: &Route, rate: u32) -> u64 {
    let mut seconds: f64 = 0.0;
    if route.stage == Stage::Stems {
        seconds = seconds.max(PREROLL_STEMS);
    }
    if route.resample.is_some() {
        seconds = seconds.max(PREROLL_RESAMPLE);
    }
    (seconds * f64::from(rate)).round() as u64
}

/// Drops `value` on a thread of its own: a track's file closes there (on a
/// network share that can take a while) and its buffers are freed, while
/// the playback thread goes on writing.
pub(super) fn drop_later<T: Send + 'static>(value: T) {
    // If no thread starts, it is dropped here after all.
    let _ = std::thread::Builder::new()
        .name("jodel-drop".into())
        .spawn(move || drop(value));
}

/// The weight of the new side `i` frames into a fade of `frames`: a raised
/// cosine from 0 to 1 (the two sides sum to one).
fn fade_weight(i: usize, frames: usize) -> f64 {
    0.5 - 0.5 * (std::f64::consts::PI * (i as f64 + 0.5) / frames as f64).cos()
}

/// Crossfades from `old` into `new` (interleaved, the same length).
pub(super) fn crossfade(old: &[i32], new: &mut [i32], channels: usize) {
    let frames = old.len() / channels;
    for (i, (from, to)) in old
        .chunks_exact(channels)
        .zip(new.chunks_exact_mut(channels))
        .enumerate()
    {
        let w = fade_weight(i, frames);
        for (a, b) in from.iter().zip(to) {
            *b = (f64::from(*a) * (1.0 - w) + f64::from(*b) * w).round() as i32;
        }
    }
}

/// Fades `samples` (interleaved) in, or out.
pub(super) fn fade(samples: &mut [i32], channels: usize, rising: bool) {
    let frames = samples.len() / channels;
    for (i, frame) in samples.chunks_exact_mut(channels).enumerate() {
        let w = fade_weight(i, frames);
        let w = if rising { w } else { 1.0 - w };
        for s in frame {
            *s = (f64::from(*s) * w).round() as i32;
        }
    }
}

impl Heard {
    pub(super) fn frame_at(&self, output_frame: u64) -> u64 {
        if let Some(w) = &self.warp {
            let frame = (w.seconds_at(output_frame) * f64::from(self.info.rate)) as u64;
            return self.info.frames.map_or(frame, |total| frame.min(total));
        }
        let since = output_frame.saturating_sub(self.seg_out);
        let frame = self.seg_frame + rescale(since, self.route.output_rate, self.route.input_rate);
        self.info.frames.map_or(frame, |total| frame.min(total))
    }

    pub(super) fn seconds_at(&self, output_frame: u64) -> f64 {
        if let Some(w) = &self.warp {
            let seconds = w.seconds_at(output_frame);
            return match self.info.duration() {
                Some(d) => seconds.min(d.as_secs_f64()),
                None => seconds,
            };
        }
        // Before the segment only in a count-in's silence: negative there.
        let since = output_frame as f64 - self.seg_out as f64;
        let seconds = self.seg_frame as f64 / f64::from(self.info.rate)
            + since / f64::from(self.route.output_rate);
        match self.info.duration() {
            Some(d) => seconds.min(d.as_secs_f64()),
            None => seconds,
        }
    }
}

/// Something to send when the output clock reaches a frame.
pub(super) enum CueEvent {
    Started(Box<TrackStart>, Heard),
    Clipped {
        index: usize,
        samples: u64,
    },
    Ended {
        index: usize,
    },
    Skipped {
        index: usize,
        error: Error,
    },
    QueueEnded,
    /// Holding at the end: this track is the next one.
    Held {
        index: usize,
    },
    /// A transition's progress, as heard.
    Transition(TransitionState),
    /// The track heard is now mapped by this (its timeline was made
    /// straight again after a transition, or it jumped).
    Heard(Heard),
    /// A jump within the track is heard.
    Jumped {
        index: usize,
        seconds: f64,
    },
}

pub(super) struct Cue {
    pub(super) at: u64,
    pub(super) event: CueEvent,
}

/// A track being decoded and written.
pub(super) struct Track {
    pub(super) index: usize,
    pub(super) info: Arc<TrackInfo>,
    pub(super) route: Route,
    pub(super) source: Source,
    /// The rate conversion after the source, when the route has one.
    pub(super) resampler: Option<Resampler>,
    pub(super) source_done: bool,
    /// The source's samples on their way into the resampler.
    pub(super) read_buf: Vec<i32>,
    /// The source's unquantized samples on their way into the resampler.
    pub(super) wide_buf: Vec<f64>,
    /// Output samples not yet written, from `out_at`.
    pub(super) out_buf: Vec<i32>,
    pub(super) out_at: usize,
    /// The output frame of this segment's first frame.
    pub(super) seg_out: u64,
    /// The track frame where the segment starts.
    pub(super) seg_frame: u64,
    /// Output frames still to drop before the segment starts (on a stems
    /// route, output frames after a jump's preroll).
    pub(super) discard: u64,
    /// Output frames made in this segment (after the discard).
    pub(super) emitted: u64,
    /// The clip count last queued, and the output frame it was queued at.
    pub(super) clips: Option<(u64, u64)>,
    /// The error that ended the track early, reported once what was
    /// decoded before it is written.
    pub(super) failure: Option<Error>,
    /// On a [`Stage::Stems`] route: the separation, started when the track
    /// is first stepped (or taken over from [`super::Engine::prestart`]).
    pub(super) stems: Option<StemRun>,
    /// Every stem has been taken.
    pub(super) stems_done: bool,
    /// Why the separation stopped: the track goes back to its usual route.
    pub(super) stems_error: Option<String>,
    /// The requantization of a stem mix the device takes at its own rate.
    pub(super) requantizer: Requantizer,
    /// Decoded ahead for a switch, not written yet: its output frames
    /// count from 0 at `seg_frame`, and its cues are held.
    pub(super) pending: bool,
    /// Cues not queued yet: a pending track's, and once it has met the old
    /// route, those to queue when it is installed (counted from
    /// `seg_out`).
    pub(super) held: VecDeque<Cue>,
    /// Clips made before the segment heard (in a switch's run-up).
    pub(super) clip_base: u64,
    /// The route it switched from, to go back to if its output does not
    /// open.
    pub(super) fallback: Option<Route>,
    /// A transition's fades of its stems, in place of the settings' mix.
    pub(super) gains: Option<Gains>,
    /// A transition's next track: its stems read along the time map.
    pub(super) warp: Option<Box<Warping>>,
    /// The separator its stems use, in place of the player's (a
    /// transition without one).
    pub(super) separator: Option<SeparatorSource>,
    /// Samples clipped when a transition added the next track to this
    /// one's output (past the soft limit's bend, so none in practice).
    pub(super) mix_clips: u64,
    /// The samples a jump's crossfade replaced, from this frame of
    /// `out_buf`, put back once written ([`super::jump`]).
    pub(super) restore: Option<(usize, Vec<i32>)>,
    /// Decoding ahead found the end of the track.
    pub(super) ahead_done: bool,
}

/// A switch of the track being played to another route, under way.
pub(super) struct Switch {
    /// The track on the new route, decoding ahead from `seg_frame`.
    pub(super) track: Track,
    /// The track frames from the old segment's start that the two routes
    /// can meet at are multiples of this.
    pub(super) grid: u64,
    /// The new route plays at the open output's format.
    pub(super) fits: bool,
    /// Decoded from the start once already.
    pub(super) retried: bool,
    /// The new route has played up to the end; it can meet the old one
    /// with what it has.
    pub(super) done: bool,
    /// The routes have met: the old track's output ends where the new
    /// one's begins.
    pub(super) spliced: bool,
}

/// What a switch is waiting for.
pub(super) enum Splice {
    Wait,
    Done,
    Cancel,
}

impl Track {
    pub(super) fn new(
        index: usize,
        info: Arc<TrackInfo>,
        route: Route,
        mut source: Source,
    ) -> Result<Track, Error> {
        let resampler = route.resample.map(|r| {
            let mut resampler = Resampler::new(r.from, r.to, usize::from(route.channels));
            // The headroom scales the samples before the one
            // requantization.
            resampler.set_gain(r.headroom.gain());
            resampler
        });
        if resampler.is_none() {
            // The one requantization is the source's (a lossy or
            // floating-point file): the headroom goes there.
            source.set_gain(route.headroom.gain());
        }
        Ok(Track {
            index,
            info,
            route,
            source,
            resampler,
            source_done: false,
            read_buf: Vec::new(),
            wide_buf: Vec::new(),
            out_buf: Vec::new(),
            out_at: 0,
            seg_out: 0,
            seg_frame: 0,
            discard: 0,
            emitted: 0,
            clips: None,
            failure: None,
            stems: None,
            stems_done: false,
            stems_error: None,
            requantizer: Requantizer::new(),
            pending: false,
            held: VecDeque::new(),
            gains: None,
            warp: None,
            separator: None,
            mix_clips: 0,
            clip_base: 0,
            fallback: None,
            restore: None,
            ahead_done: false,
        })
    }

    /// The track's output frames waiting to be written.
    pub(super) fn unwritten(&self) -> usize {
        (self.out_buf.len() - self.out_at) / usize::from(self.route.channels)
    }

    /// Samples clipped by the source's requantization, the resampler and
    /// the stem mix's requantization, since the track opened.
    pub(super) fn clipped(&self) -> u64 {
        self.source.clipped()
            + self.resampler.as_ref().map_or(0, |r| r.clipped())
            + self.requantizer.clipped()
            + self.mix_clips
    }

    /// Restarts the track at `frame`: the source moves there and the
    /// segment starts again.
    pub(super) fn restart_at(&mut self, frame: u64) -> Result<(), Error> {
        let frame = self.info.frames.map_or(frame, |total| frame.min(total));
        self.restore = None;
        self.ahead_done = false;
        if self.route.stage == Stage::Stems {
            // A new separation starts there when the track is next stepped.
            self.stems = None;
            self.stems_done = false;
            self.stems_error = None;
            self.requantizer.reset();
        } else {
            if self.source.position() != frame {
                self.source.seek(frame)?;
            }
            self.source_done = false;
        }
        if let Some(resampler) = &mut self.resampler {
            resampler.reset();
        }
        self.out_buf.clear();
        self.out_at = 0;
        self.seg_frame = frame;
        self.discard = 0;
        self.emitted = 0;
        self.clips = None;
        self.failure = None;
        Ok(())
    }

    /// Moves a track just opened to `frame` (nothing to do at 0).
    pub(super) fn start_at(&mut self, frame: u64) -> Result<(), Error> {
        if frame > 0 {
            self.restart_at(frame)?;
        }
        Ok(())
    }
}

pub(super) struct OpenOutput {
    pub(super) output: Box<dyn Output>,
    pub(super) info: StreamInfo,
    pub(super) clock: Clock,
}

/// What one unit of work on the current track did.
pub(super) enum Progress {
    Busy,
    Complete,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_steps_are_the_smallest_exact_ones() {
        assert_eq!(gcd(44_100, 48_000), 300);
        for (input, output, want) in [
            (44_100u64, 88_200u64, (1, 2)),
            (44_100, 48_000, (147, 160)),
            (96_000, 44_100, (320, 147)),
            (48_000, 48_000, (1, 1)),
        ] {
            let g = gcd(input, output);
            let (a, b) = (input / g, output / g);
            assert_eq!((a, b), want);
            // `a` track frames are exactly `b` output frames, and no fewer.
            assert_eq!(a * output, b * input);
            for n in 1..a {
                assert_ne!(n * output % input, 0);
            }
        }
    }

    #[test]
    fn fades_sum_to_one_and_end_where_they_should() {
        let frames = 64;
        for i in 0..frames {
            let sum = fade_weight(i, frames) + fade_weight(frames - 1 - i, frames);
            assert!((sum - 1.0).abs() < 1e-12);
        }
        assert!(fade_weight(0, frames) < 0.001);
        assert!(fade_weight(frames - 1, frames) > 0.999);
        let old = vec![1_000; 2 * frames];
        let mut new = vec![-1_000; 2 * frames];
        crossfade(&old, &mut new, 2);
        assert!(new[0] > 990 && new[2 * frames - 1] < -990);
        assert!(new.windows(2).all(|w| w[1] <= w[0]));
    }
}
