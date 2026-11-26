//! A track as the engine plays it ([`Track`]): its source, route and
//! buffers; what is heard of it ([`Heard`]); a route switch under way
//! ([`Switch`]); and the small helpers on frames and fades they share.

use std::sync::Arc;
use std::time::Duration;

use jodel_output::{Clock, Output, StreamInfo};

use crate::error::Error;
use crate::rate::Route;
use crate::resample::Resampler;
use crate::source::{Source, TrackInfo};
use crate::transition::TempoMode;
use crate::warp::TimeMap;

use super::cues::CueQueue;
use super::segment::{SegStart, Segment, SegmentLine, Segments};
use super::PREROLL_RESAMPLE;

/// Where in a track playback starts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum StartAt {
    /// A track frame.
    Frame(u64),
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
    /// Where its segment is on the output.
    pub(super) line: SegmentLine,
    /// The track is read along a time map (a head's, a transition's
    /// tracks'): its position comes from that, not from a straight line.
    pub(super) warp: Option<Arc<HeardWarp>>,
    /// The key it is heard in, in semitones ([`crate::Command::Key`]).
    pub(super) key: i8,
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

/// A map a head's stems are read along, and how ([`Along`]): by `tempo`,
/// `key` semitones apart (the drums kept in theirs).
#[derive(Clone)]
pub(super) struct AlongMap {
    pub(super) heard: Arc<HeardWarp>,
    pub(super) tempo: TempoMode,
    pub(super) key: i8,
}

/// A head read along a time map (a transition's, a key's) on an output
/// of lanes: its stems are read along the map as its parts, and its main
/// is their sum ([`super::along`]), from output frame `seg_out` of its
/// segment.
pub(super) struct Along {
    pub(super) map: AlongMap,
    /// Main of the route it was on, from the segment's first frame on,
    /// until the sum is crossfaded in from it.
    pub(super) tail: Option<Vec<i32>>,
    /// Its way back onto its own route, once the map runs straight.
    pub(super) back: Back,
    /// It leaves the map as soon as it can, straight or not
    /// ([`super::Engine::leave_map`]).
    pub(super) leave: bool,
    /// For a transition's next track (silent before its window), or a
    /// track installed in a key: when it is heard. Its stems may be waited
    /// for until then is near.
    pub(super) heard_from: Option<HeardFrom>,
    /// The parts' clips counted into the track's so far.
    pub(super) clips: u64,
}

/// When a head read along a map is heard, on the playing head's clock:
/// that head's frame this one's frame 0 is linked to on the desk
/// (`origin`: a transition's next track; none for the playing head itself,
/// a track installed in a key behind a count-in), and its frame from which
/// this one is heard (`at`).
pub(super) struct HeardFrom {
    pub(super) clock: Clock,
    pub(super) origin: Option<u64>,
    pub(super) at: u64,
}

impl HeardFrom {
    /// The playing head's frames still to be heard before this one is.
    pub(super) fn silent(&self) -> u64 {
        self.at.saturating_sub(self.clock.position())
    }

    /// A linked head's first frame not played yet: the desk has played
    /// past those before it, so they are not heard whatever they hold
    /// (written now, they are late and dropped). None for a head on its own
    /// clock, which plays all it is given.
    pub(super) fn unplayed(&self) -> u64 {
        self.origin
            .map_or(0, |origin| self.clock.position().saturating_sub(origin))
    }

    /// Whether it is the playing head's own wait, and over: heard now.
    pub(super) fn heard_now(&self) -> bool {
        self.origin.is_none() && self.silent() == 0
    }
}

/// Where a head read along a map stands with its way back onto its own
/// route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Back {
    /// On the map: nothing may move it (a route switch waits for it).
    Reading,
    /// The switch back is asked for (under way while there is one).
    Asked,
    /// The switch back did not come about: asked again once output frame
    /// this is written.
    Retry(u64),
}

impl HeardWarp {
    /// The track's seconds at `output_frame`.
    pub(super) fn seconds_at(&self, output_frame: u64) -> f64 {
        let since = output_frame.saturating_sub(self.origin) as f64;
        self.map
            .at(self.a0 + since / f64::from(self.out_rate.max(1)))
            .max(0.0)
    }

    /// The output frame nearest where the track is at `seconds`.
    pub(super) fn output_frame_at(&self, seconds: f64) -> u64 {
        let since = (self.map.inverse(seconds) - self.a0) * f64::from(self.out_rate.max(1));
        (self.origin as f64 + since).round().max(0.0) as u64
    }
}

/// `duration` in frames at `rate`.
pub(super) fn frames_of(duration: Duration, rate: u32) -> u64 {
    (duration.as_secs_f64() * f64::from(rate)).round() as u64
}

/// How many track frames before the switch `route` starts decoding.
pub(super) fn preroll(route: &Route, rate: u32) -> u64 {
    let seconds = if route.resample.is_some() {
        PREROLL_RESAMPLE
    } else {
        0.0
    };
    (seconds * f64::from(rate)).round() as u64
}

/// The weight of the new side `i` frames into a fade of `frames`: a raised
/// cosine from 0 to 1 (the two sides sum to one).
pub(super) fn fade_weight(i: usize, frames: usize) -> f64 {
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

impl HeardWarp {
    /// The same, its output frames counted `by` fewer (`by` is frame 0 of
    /// another output that follows this one).
    fn rebased(&self, by: u64) -> HeardWarp {
        let back = self.origin.min(by);
        let early = (by - back) as f64 / f64::from(self.out_rate.max(1));
        HeardWarp {
            map: Arc::clone(&self.map),
            origin: self.origin - back,
            a0: self.a0 + early,
            out_rate: self.out_rate,
        }
    }
}

impl Heard {
    /// The same, its output frames counted `by` fewer: as heard on an
    /// output whose frame 0 is this one's frame `by` (the head a
    /// transition hands over to).
    pub(super) fn rebased(&self, by: u64) -> Heard {
        Heard {
            line: self.line.rebased(by),
            warp: self.warp.as_ref().map(|w| Arc::new(w.rebased(by))),
            ..self.clone()
        }
    }

    pub(super) fn frame_at(&self, output_frame: u64) -> u64 {
        if let Some(w) = &self.warp {
            // Rounded: a map's arithmetic is a hair off whole frames.
            let frame = (w.seconds_at(output_frame) * f64::from(self.info.rate)).round() as u64;
            return self.info.frames.map_or(frame, |total| frame.min(total));
        }
        let frame = self.line.frame_at(output_frame);
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
        let seconds = self.line.seconds_at(output_frame);
        match self.info.duration() {
            Some(d) => seconds.min(d.as_secs_f64()),
            None => seconds,
        }
    }
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
    /// The output frame of this segment's first frame, and the track frame
    /// it is: set by [`Track::begin_segment`] and
    /// Where its segment is on the output: set by [`Track::begin_segment`]
    /// and [`Track::continue_segment`] only.
    pub(super) line: SegmentLine,
    /// Output frames made in this segment.
    pub(super) emitted: u64,
    /// The segment's id and how it began, and where ids come from.
    pub(super) segment: Segment,
    segments: Segments,
    /// The clip count last queued, and the output frame it was queued at.
    pub(super) clips: Option<(u64, u64)>,
    /// The error that ended the track early, reported once what was
    /// decoded before it is written.
    pub(super) failure: Option<Error>,
    /// Decoded ahead for a switch, not written yet: its output frames
    /// count from 0 at `seg_frame`, and its cues are held.
    pub(super) pending: bool,
    /// Cues not queued yet: a pending track's, and once it has met the old
    /// route, those to queue when it is installed (counted from
    /// `seg_out`).
    pub(super) held: CueQueue,
    /// Clips made before the segment heard (in a switch's run-up).
    pub(super) clip_base: u64,
    /// Read along a time map as stem parts, main their sum.
    pub(super) along: Option<Along>,
    /// Samples clipped where a head read along a map summed its parts
    /// into main.
    pub(super) mix_clips: u64,
    /// The samples a jump's crossfade replaced, from this frame of
    /// `out_buf`, put back once written ([`super::jump`]).
    pub(super) restore: Option<(usize, Vec<i32>)>,
    /// The first output frame from which what `out_buf` holds before
    /// `out_at` is what was written there (before it: the track before a
    /// jump's target, and the samples the jump's crossfade replaced), so
    /// the stem lanes may write it again ([`super::lanes`]).
    pub(super) rewritable_from: u64,
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
    /// Decoded from its start once already.
    pub(super) retried: bool,
    /// The new route has played up to the end; it can meet the old one
    /// with what it has.
    pub(super) done: bool,
    /// The routes have met: the old track's output ends where the new
    /// one's begins.
    pub(super) spliced: bool,
    /// The settings changed while it was being handed over: the track is
    /// routed again once it has been.
    pub(super) again: bool,
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
        segments: Segments,
    ) -> Result<Track, Error> {
        // A file the device does not take in its channels is read as
        // stereo.
        if route.channels != info.channels {
            source.make_stereo();
        }
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
        let line = SegmentLine::new(&route, info.rate);
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
            line,
            emitted: 0,
            segment: Segment {
                id: segments.next(),
                begun: SegStart::Cold,
            },
            segments,
            clips: None,
            failure: None,
            pending: false,
            held: CueQueue::default(),
            along: None,
            mix_clips: 0,
            clip_base: 0,
            restore: None,
            rewritable_from: 0,
            ahead_done: false,
        })
    }

    /// The track's output frames waiting to be written.
    pub(super) fn unwritten(&self) -> usize {
        (self.out_buf.len() - self.out_at) / usize::from(self.route.channels)
    }

    /// Samples clipped by the source's requantization, the resampler and a
    /// head's sum of its parts, since the track opened.
    pub(super) fn clipped(&self) -> u64 {
        self.source.clipped() + self.resampler.as_ref().map_or(0, |r| r.clipped()) + self.mix_clips
    }

    /// The track's output starts a new segment: output frame `seg_out` is
    /// track frame `seg_frame`, begun as `begun`.
    pub(super) fn begin_segment(&mut self, seg_out: u64, seg_frame: u64, begun: SegStart) {
        let id = self.segments.next();
        self.set_segment(seg_out, seg_frame, Segment { id, begun });
    }

    /// The track's output goes on in `old`'s segment (another route met on
    /// its line) from output frame `seg_out`, track frame `seg_frame`.
    pub(super) fn continue_segment(
        &mut self,
        old: Segment,
        seg_out: u64,
        seg_frame: u64,
        begun: SegStart,
    ) {
        self.set_segment(seg_out, seg_frame, Segment { id: old.id, begun });
    }

    fn set_segment(&mut self, seg_out: u64, seg_frame: u64, segment: Segment) {
        self.line = self.line.at(seg_out, seg_frame);
        self.segment = segment;
    }

    /// Whether the track plays its plain route as the settings make it:
    /// the file (resampled perhaps), not read along a map, nor decoded
    /// ahead for a switch.
    pub(super) fn plain(&self) -> bool {
        self.plain_route() && !self.pending
    }

    /// Whether its route is its plain one: the file (resampled perhaps),
    /// not read along a map (decoded ahead for a switch or not).
    pub(super) fn plain_route(&self) -> bool {
        self.along.is_none()
    }

    /// Whether it plays as it is heard: its plain route, or along a map
    /// (not a route decoded ahead for a switch). Such a track may have
    /// stem parts, and be read along a map.
    pub(super) fn as_heard(&self) -> bool {
        !self.pending || self.along.is_some()
    }

    /// The key it plays in, in semitones (0 off a map).
    pub(super) fn key(&self) -> i8 {
        self.along.as_ref().map_or(0, |a| a.map.key)
    }

    /// Where its map puts the track, from `seg_out` on, when it is read
    /// along one ([`Along`]).
    pub(super) fn heard_warp(&self) -> Option<Arc<HeardWarp>> {
        self.along.as_ref().map(|a| Arc::clone(&a.map.heard))
    }

    /// Whether everything of the track (or of what it was told to play) is
    /// made: its output ends at `seg_out + emitted`.
    pub(super) fn made_all(&self) -> bool {
        self.source_done && self.resampler.as_ref().is_none_or(Resampler::is_finished)
    }

    /// The output frame after its output's last, once all of it is made
    /// ([`Track::made_all`]).
    pub(super) fn main_end(&self) -> Option<u64> {
        self.made_all().then_some(self.line.seg_out + self.emitted)
    }

    /// Restarts the track at `frame`: the source moves there and the
    /// segment starts again.
    pub(super) fn restart_at(&mut self, frame: u64) -> Result<(), Error> {
        let frame = self.info.frames.map_or(frame, |total| frame.min(total));
        self.restore = None;
        self.ahead_done = false;
        if self.source.position() != frame {
            self.source.seek(frame)?;
        }
        self.source_done = false;
        if let Some(resampler) = &mut self.resampler {
            resampler.reset();
        }
        self.out_buf.clear();
        self.out_at = 0;
        self.rewritable_from = 0;
        self.begin_segment(self.line.seg_out, frame, SegStart::Cold);
        self.emitted = 0;
        self.clips = None;
        self.failure = None;
        Ok(())
    }

    /// The track goes on along `map` from output frame `at`, the frame
    /// written next: what its route made from there is cut and kept as the
    /// tail its parts' sum crossfades in from (a switch's `fade` long at
    /// most), and a segment joined to what played begins there, at the
    /// track frame the map puts there. Its route makes nothing more, and
    /// its clips count afresh.
    pub(super) fn go_along(&mut self, map: AlongMap, at: u64, fade: usize) {
        let ch = usize::from(self.route.channels).max(1);
        let tail: Vec<i32> = self.out_buf.drain(self.out_at..).collect();
        let fade = (tail.len() / ch).min(fade);
        let frame = (map.heard.seconds_at(at) * f64::from(self.info.rate)).round() as u64;
        self.begin_segment(at, frame, SegStart::Joined { fade });
        self.emitted = 0;
        self.resampler = None;
        self.source_done = false;
        self.ahead_done = false;
        self.restore = None;
        self.rewritable_from = at;
        self.clip_base = self.clipped();
        self.clips = None;
        self.along = Some(Along {
            map,
            tail: Some(tail[..fade * ch].to_vec()),
            back: Back::Reading,
            leave: false,
            heard_from: None,
            clips: 0,
        });
    }

    /// How the track is heard from its segment's start.
    pub(super) fn heard(&self) -> Heard {
        Heard {
            index: self.index,
            info: self.info.clone(),
            route: self.route.clone(),
            line: self.line,
            warp: self.heard_warp(),
            key: self.key(),
        }
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
