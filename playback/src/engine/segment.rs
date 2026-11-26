//! A track's segments on its output: where each starts and how it began
//! ([`Segment`]), and the line that maps the output's frames to the
//! track's along it ([`SegmentLine`]), the one place the engine works out
//! where the listener is in a track.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::rate::Route;
use crate::resample::gcd;

/// How a track's segment begins on the output (what its stem parts start
/// like, [`super::lanes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SegStart {
    /// From silence at its frame: a new track, a seek (main's resampler
    /// starts there too).
    Cold,
    /// Out of what played before, main crossfaded from it over `fade`
    /// frames: a jump, a route met in a switch, a move onto a map
    /// (over none for a transition's next track, read along its map from
    /// its first frame).
    Joined { fade: usize },
}

/// A segment of a track's output: its id, and how it began (where it is:
/// the track's [`SegmentLine`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Segment {
    pub(super) id: u64,
    pub(super) begun: SegStart,
}

/// An engine's count of the segments its tracks began, for their ids.
#[derive(Clone, Debug, Default)]
pub(super) struct Segments(Arc<AtomicU64>);

impl Segments {
    pub(super) fn next(&self) -> u64 {
        self.0.fetch_add(1, Ordering::Relaxed)
    }
}

/// `frames` at `from` Hz in frames at `to` Hz, rounded down.
pub(super) fn rescale(frames: u64, from: u32, to: u32) -> u64 {
    (u128::from(frames) * u128::from(to) / u128::from(from.max(1))) as u64
}

/// The line a track's segment follows on its output: output frame
/// `seg_out` is track frame `seg_frame`, and from there the two go on at
/// their rates (the route's input and output rates; the track's own, for
/// its seconds).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SegmentLine {
    /// The output frame of the segment's first frame.
    pub(super) seg_out: u64,
    /// The track frame at the segment's start.
    pub(super) seg_frame: u64,
    /// Track frames a second, as the route reads them.
    pub(super) input_rate: u32,
    /// Output frames a second.
    pub(super) output_rate: u32,
    /// The track's own frames a second.
    pub(super) track_rate: u32,
}

impl SegmentLine {
    /// The line of `route` (a track at `track_rate`) from output frame 0,
    /// track frame 0.
    pub(super) fn new(route: &Route, track_rate: u32) -> SegmentLine {
        SegmentLine {
            seg_out: 0,
            seg_frame: 0,
            input_rate: route.input_rate.max(1),
            output_rate: route.output_rate.max(1),
            track_rate: track_rate.max(1),
        }
    }

    /// The same rates, output frame `seg_out` track frame `seg_frame`.
    pub(super) fn at(self, seg_out: u64, seg_frame: u64) -> SegmentLine {
        SegmentLine {
            seg_out,
            seg_frame,
            ..self
        }
    }

    /// The track frame heard at output frame `out`, rounded down (the
    /// segment's first before its start).
    pub(super) fn frame_at(&self, out: u64) -> u64 {
        let since = out.saturating_sub(self.seg_out);
        self.seg_frame + rescale(since, self.output_rate, self.input_rate)
    }

    /// The output frame track frame `frame` is heard at, rounded down (the
    /// segment's first, for a frame before it).
    pub(super) fn out_at(&self, frame: u64) -> u64 {
        let since = frame.saturating_sub(self.seg_frame);
        self.seg_out + rescale(since, self.input_rate, self.output_rate)
    }

    /// The track's seconds at output frame `out` (negative before the
    /// segment: only in a count-in's silence).
    pub(super) fn seconds_at(&self, out: u64) -> f64 {
        let since = out as f64 - self.seg_out as f64;
        self.seg_frame as f64 / f64::from(self.track_rate) + since / f64::from(self.output_rate)
    }

    /// The route's smallest exact step: `a` track frames are `b` output
    /// frames (a track frame that many from the segment's start is heard at
    /// an exact output frame).
    pub(super) fn grid(&self) -> (u64, u64) {
        let (input, output) = (u64::from(self.input_rate), u64::from(self.output_rate));
        let g = gcd(input, output);
        (input / g, output / g)
    }

    /// The output frames from the segment's start of track frame `frame`,
    /// on the grid: whole steps, rounded down (negative before the start).
    pub(super) fn stream_frame(&self, frame: u64) -> i64 {
        let (a, b) = self.grid();
        (frame as i64 - self.seg_frame as i64).div_euclid(a as i64) * b as i64
    }

    /// The same, its output frames counted `by` fewer: as heard on an
    /// output whose frame 0 is this one's frame `by` (the head a transition
    /// hands over to).
    pub(super) fn rebased(&self, by: u64) -> SegmentLine {
        let back = self.seg_out.min(by);
        let since = rescale(by - back, self.output_rate, self.input_rate);
        self.at(self.seg_out - back, self.seg_frame + since)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(input: u32, output: u32) -> SegmentLine {
        SegmentLine {
            seg_out: 0,
            seg_frame: 0,
            input_rate: input,
            output_rate: output,
            track_rate: input,
        }
    }

    #[test]
    fn grid_steps_are_the_smallest_exact_ones() {
        assert_eq!(gcd(44_100, 48_000), 300);
        for (input, output, want) in [
            (44_100u32, 88_200u32, (1, 2)),
            (44_100, 48_000, (147, 160)),
            (96_000, 44_100, (320, 147)),
            (48_000, 48_000, (1, 1)),
        ] {
            let (a, b) = line(input, output).grid();
            assert_eq!((a, b), want);
            // `a` track frames are exactly `b` output frames, and no fewer.
            assert_eq!(a * u64::from(output), b * u64::from(input));
            for n in 1..a {
                assert_ne!(n * u64::from(output) % u64::from(input), 0);
            }
        }
    }

    #[test]
    fn the_line_maps_both_ways_on_its_grid() {
        let l = line(44_100, 48_000).at(1_000, 44_100);
        assert_eq!(l.frame_at(1_000), 44_100);
        assert_eq!(l.frame_at(1_000 + 160), 44_100 + 147);
        assert_eq!(l.out_at(44_100 + 147), 1_000 + 160);
        assert_eq!(l.frame_at(0), 44_100, "the segment's first before it");
        assert!((l.seconds_at(1_000) - 1.0).abs() < 1e-12);
        assert!((l.seconds_at(1_000 + 48_000) - 2.0).abs() < 1e-12);
        assert!(l.seconds_at(0) < 1.0, "negative time before it");
        assert_eq!(l.stream_frame(44_100 + 147), 160);
        assert_eq!(l.stream_frame(44_100 - 1), -160);
        let r = l.rebased(1_000 + 160);
        assert_eq!((r.seg_out, r.seg_frame), (0, 44_100 + 147));
        assert_eq!(r.frame_at(160), l.frame_at(1_000 + 320));
    }
}
