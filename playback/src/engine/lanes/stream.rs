//! A stem run's stems at the output's rate, on the timeline of the track
//! they belong to: where a run frame falls among the output frames
//! ([`Line`], [`Grid`]), and the stream that takes the run's stems,
//! resamples them to the output's rate and lays them out in the track's
//! channels ([`Stream`]).
//!
//! The output follows a track along a line: output frame `seg_out` is track
//! frame `seg_frame`, and from there the two go on at their rates. A run
//! started at track frame `start` has its frames at the separator's rate
//! from there. A stream that starts at an output frame where a run frame
//! falls exactly, resampled with the same filter as the track's own
//! resampler, gives parts on the very instants of the output's frames: the
//! parts add up to what the track plays, frame for frame, as closely as the
//! separator's stems add up to the song.
//!
//! A head read along a time map ([`crate::engine::track::Along`]) has its
//! line follow the map instead: the stems are read along it at the
//! separator's rate ([`Warping`]), every channel at the same instants,
//! before they are resampled to the output's rate, so its main (their sum)
//! and its parts stay on the same frames.

use std::sync::Arc;

use crate::engine::segment::SegStart;
use crate::engine::track::{AlongMap, Track};
use crate::resample::{gcd, output_frames, Requantizer, Resampler};
use crate::stems::{lane, Pull, Stem, StemRun, RATE as STEM_RATE, STEMS};
use crate::warping::{Reading, Warping};

/// Frames of stems taken from a run at a time.
const PIECE: usize = 4096;

/// The line the output follows a track along.
#[derive(Clone)]
pub(super) struct Line {
    /// The track's segment ([`Track::segment`]), and how it began.
    pub(super) segment: u64,
    pub(super) begun: SegStart,
    /// The track's rate, and the output's.
    pub(super) rate: u32,
    pub(super) out: u32,
    /// Output frame `seg_out` is track frame `seg_frame`.
    pub(super) seg_out: u64,
    pub(super) seg_frame: u64,
    /// The output frame after the track's last, when its length is known.
    pub(super) end: Option<u64>,
    /// The gain the track's samples are heard at (its route's headroom).
    pub(super) gain: f64,
    /// The map the track is read along from `seg_out` on, if it is.
    pub(super) along: Option<AlongMap>,
}

impl Line {
    /// The line `track` is heard along.
    pub(super) fn of(track: &Track) -> Line {
        let (rate, out) = (track.info.rate.max(1), track.route.output_rate.max(1));
        let along = track.along.as_ref().map(|a| a.map.clone());
        // Read along a map, it ends where its stems do.
        let end = track
            .info
            .frames
            .filter(|_| along.is_none())
            .and_then(|total| total.checked_sub(track.line.seg_frame))
            .map(|left| track.line.seg_out + output_frames(rate, out, left));
        let route = &track.route;
        let gain = route
            .resample
            .map_or(route.headroom.gain(), |r| r.headroom.gain());
        Line {
            segment: track.segment.id,
            begun: track.segment.begun,
            rate,
            out,
            seg_out: track.line.seg_out,
            seg_frame: track.line.seg_frame,
            end,
            gain,
            along,
        }
    }

    /// Whether `other` is this line: the same segment at the same rates
    /// and gain, read along the same map if along one, its start elsewhere
    /// along it (a route met at the same format goes on from a frame of
    /// it).
    pub(super) fn same(&self, other: &Line) -> bool {
        let along = (i128::from(other.seg_out) - i128::from(self.seg_out)) * i128::from(self.rate)
            == (i128::from(other.seg_frame) - i128::from(self.seg_frame)) * i128::from(self.out);
        let map = match (&self.along, &other.along) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                Arc::ptr_eq(&a.heard, &b.heard) && a.tempo == b.tempo && a.key == b.key
            }
            _ => false,
        };
        self.segment == other.segment
            && self.rate == other.rate
            && self.out == other.out
            && self.gain == other.gain
            && along
            && map
    }

    /// The map's time at output frame `o`, and the track's seconds it reads
    /// there (read along a map only).
    pub(super) fn along_at(&self, o: i64) -> Option<(f64, f64)> {
        let heard = &self.along.as_ref()?.heard;
        let a = heard.a0 + (o - heard.origin as i64) as f64 / f64::from(self.out);
        Some((a, heard.map.at(a)))
    }

    /// The track frame heard at output frame `o`, rounded down (0 before
    /// the track's start).
    pub(super) fn frame_at(&self, o: i128) -> u64 {
        let since = o - i128::from(self.seg_out);
        let frame = i128::from(self.seg_frame)
            + (since * i128::from(self.rate)).div_euclid(i128::from(self.out));
        frame.max(0) as u64
    }
}

/// The output frames at which frames of a run started at track frame
/// `start` fall exactly: output frame `o` is run frame `r`, and so on every
/// `period` output frames, `step` run frames apart.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Grid {
    o: i128,
    r: i128,
    period: i128,
    step: i128,
}

/// `a` to the power −1 modulo `m` (`a` and `m` coprime).
fn inverse(a: i128, m: i128) -> i128 {
    let (mut r0, mut r1) = (a.rem_euclid(m), m);
    let (mut s0, mut s1) = (1i128, 0i128);
    while r1 != 0 {
        let q = r0 / r1;
        (r0, r1) = (r1, r0 - q * r1);
        (s0, s1) = (s1, s0 - q * s1);
    }
    s0.rem_euclid(m)
}

impl Grid {
    /// Where a run started at track frame `start` meets `line`, if a run
    /// frame falls on an output frame at all (it does when the run starts
    /// a whole number of the separator's frames from the segment's track
    /// frame, as [`run_start`] has it).
    ///
    /// Output frame `seg_out + u` is the run frame
    /// `(Δ·R·out + u·R·rate) / (rate·out)` (Δ the segment's track frame
    /// from the run's start, R the separator's rate): a whole one where the
    /// numerator is a multiple of `rate·out`, a congruence in `u`.
    pub(super) fn of(line: &Line, start: u64) -> Option<Grid> {
        let (r, rate, out) = (
            i128::from(STEM_RATE),
            i128::from(line.rate),
            i128::from(line.out),
        );
        let delta = i128::from(line.seg_frame) - i128::from(start);
        let h = i128::from(gcd(u64::from(STEM_RATE), u64::from(line.out)));
        let g = rate * h;
        let c = delta * r * out;
        if c % g != 0 {
            return None;
        }
        let (a, m) = (r / h, out / h);
        let u = if m == 1 {
            0
        } else {
            ((-c / g).rem_euclid(m) * inverse(a, m)).rem_euclid(m)
        };
        Some(Grid {
            o: i128::from(line.seg_out) + u,
            r: (c + u * r * rate) / (rate * out),
            period: m,
            step: a,
        })
    }

    /// The last output frame on the grid at or before `x`, and its run
    /// frame.
    pub(super) fn at_or_before(&self, x: i128) -> (i128, i128) {
        let k = (x - self.o).div_euclid(self.period);
        (self.o + k * self.period, self.r + k * self.step)
    }

    /// The first output frame on the grid whose run frame is at least
    /// `r`, and that run frame.
    pub(super) fn first_at_run(&self, r: i128) -> (i128, i128) {
        let k = (r - self.r + self.step - 1).div_euclid(self.step);
        (self.o + k * self.period, self.r + k * self.step)
    }
}

/// The track frame a run for `line` starts at, at or before track frame
/// `want`: a whole number of the separator's frames from the segment's
/// track frame (so its frames fall on output frames, [`Grid::of`]), and
/// not before the track's start.
pub(super) fn run_start(line: &Line, want: u64) -> u64 {
    let step = i128::from(line.rate) / i128::from(gcd(u64::from(line.rate), u64::from(STEM_RATE)));
    let seg = i128::from(line.seg_frame);
    let mut start = seg + (i128::from(want) - seg).div_euclid(step) * step;
    if start < 0 {
        start += (-start + step - 1) / step * step;
    }
    start as u64
}

/// What [`Stream::make`] did.
pub(super) enum Made {
    /// Frames were made (perhaps none past what is dropped).
    Frames,
    /// The run has nothing yet.
    Waiting,
    /// The stream has made everything up to the track's end.
    Done,
    /// The run failed.
    Failed(String),
    /// The track's stems are not kept, and the run separates none.
    Missing,
}

/// The parts of one track at the output's rate, from a run: the stems
/// taken in order, made the track's channels (a mono file's stems are
/// averaged, a file of more than two channels has them on its front pair
/// and nothing on the rest), read along its map when the line has one,
/// scaled by the track's headroom as main is, resampled to the output's
/// rate (requantized to 24 bits) and cut or padded to the track's end. A
/// run of the whole mix has only its other instruments' stem made, the
/// rest exactly silent.
pub(super) struct Stream {
    pub(super) line: Line,
    /// The output frame of the stream's first frame (before the output's
    /// first, for a run-up near it).
    pub(super) first: i64,
    /// Frames made since then.
    made: u64,
    /// The track's channels, and the stems' (one or two each).
    channels: usize,
    pairs: usize,
    /// Read along the line's map: the track's seconds at the run's first
    /// frame.
    read_from: Option<f64>,
    /// How the stems are made, once the run says what it serves.
    pass: Option<Pass>,
    requantizer: Requantizer,
    taken: Vec<f64>,
    feed: Vec<f64>,
    out: Vec<i32>,
    laid: Vec<i32>,
    pub(super) done: bool,
}

/// How a stream makes its stems: from which lane on (the other
/// instruments' alone for a run of the whole mix), read along the map,
/// resampled.
struct Pass {
    from: usize,
    warping: Option<Warping>,
    resampler: Option<Resampler>,
}

/// The other instruments' lane ([`crate::stems::LANES`]' last).
const OTHER: usize = STEMS - 1;

/// The channels of the drums' lane (they keep their key) among those made
/// from lane `from` on, `pairs` a lane: none when they are not made.
fn drums(from: usize, pairs: usize) -> Vec<usize> {
    let l = lane(Stem::Drums);
    if l < from {
        return Vec::new();
    }
    ((l - from) * pairs..(l - from + 1) * pairs).collect()
}

impl Stream {
    /// A stream on `line` whose first frame is heard at output frame
    /// `first`, for a track of `channels` channels.
    pub(super) fn new(line: Line, first: i64, channels: usize) -> Stream {
        Stream {
            line,
            first,
            made: 0,
            channels,
            pairs: channels.clamp(1, 2),
            read_from: None,
            pass: None,
            requantizer: Requantizer::new(),
            taken: Vec::new(),
            feed: Vec::new(),
            out: Vec::new(),
            laid: Vec::new(),
            done: false,
        }
    }

    /// The same, read along the line's map from a run whose first frame is
    /// the track's seconds `read_from`.
    pub(super) fn along(line: Line, first: i64, channels: usize, read_from: f64) -> Stream {
        Stream {
            read_from: Some(read_from),
            ..Stream::new(line, first, channels)
        }
    }

    /// Frames the stream has in all: to the track's end, when known.
    fn length(&self) -> Option<u64> {
        self.line
            .end
            .map(|end| (end as i64 - self.first).max(0) as u64)
    }

    /// The first lane made.
    fn from(&self) -> usize {
        self.pass.as_ref().map_or(0, |p| p.from)
    }

    /// Samples the stems are made in, a frame.
    fn width(&self) -> usize {
        (STEMS - self.from()) * self.pairs
    }

    /// How the stems are made, set up for what `run` serves.
    fn pass(&mut self, run: &StemRun) -> &mut Pass {
        let pairs = self.pairs;
        let (line, first, read_from) = (&self.line, self.first, self.read_from);
        self.pass.get_or_insert_with(|| {
            let from = if run.whole() { OTHER } else { 0 };
            let channels = (STEMS - from) * pairs;
            let resampler = (line.out != STEM_RATE).then(|| {
                let mut r = Resampler::new(STEM_RATE, line.out, channels);
                r.set_gain(line.gain);
                r
            });
            let warping = line
                .along_at(first)
                .zip(line.along.as_ref())
                .zip(read_from)
                .map(|(((a, _), along), read_from)| {
                    Warping::new(Reading {
                        map: Arc::clone(&along.heard.map),
                        a0: a,
                        b_origin: read_from,
                        tempo: along.tempo,
                        channels,
                        key: along.key,
                        kept: drums(from, pairs),
                    })
                });
            Pass {
                from,
                warping,
                resampler,
            }
        })
    }

    /// Samples its resampling or requantization clipped so far.
    pub(super) fn clipped(&self) -> u64 {
        let resampler = self.pass.as_ref().and_then(|p| p.resampler.as_ref());
        self.requantizer.clipped() + resampler.map_or(0, Resampler::clipped)
    }

    /// How far (in frames at the stems' rate) what it reads is off its map.
    pub(super) fn offset(&self) -> f64 {
        let warping = self.pass.as_ref().and_then(|p| p.warping.as_ref());
        warping.map_or(0.0, Warping::offset)
    }

    /// Makes the next frames from `run` (waiting up to `wait` for its
    /// stems) and hands them to `put` with the output frame of the first,
    /// as samples of [`STEMS`] × the track's channels a frame, in lane
    /// order.
    pub(super) fn make(
        &mut self,
        run: &mut StemRun,
        wait: std::time::Duration,
        put: &mut dyn FnMut(i64, &[i32]),
    ) -> Made {
        if self.done {
            return Made::Done;
        }
        self.taken.clear();
        self.out.clear();
        match run.take(PIECE, &mut self.taken, wait) {
            Pull::Frames => {
                let from = self.pass(run).from;
                self.feed.clear();
                for frame in self.taken.chunks_exact(2 * STEMS) {
                    for stem in frame[2 * from..].chunks_exact(2) {
                        if self.pairs == 1 {
                            self.feed.push(0.5 * (stem[0] + stem[1]));
                        } else {
                            self.feed.extend_from_slice(stem);
                        }
                    }
                }
                if let Some(w) = self.pass.as_mut().and_then(|p| p.warping.as_mut()) {
                    w.read(&mut self.feed);
                }
                self.put_feed();
            }
            Pull::End => {
                // What the map's reader holds comes out before the end.
                self.feed.clear();
                self.pass(run);
                if let Some(w) = self.pass.as_mut().and_then(|p| p.warping.as_mut()) {
                    w.end();
                    w.read(&mut self.feed);
                }
                self.put_feed();
                // The stems end with the track: the stream is cut or
                // padded to its end.
                let (length, width) = (self.length(), self.width());
                let resampler = self.pass.as_mut().and_then(|p| p.resampler.as_mut());
                match (resampler, length) {
                    (Some(r), Some(n)) => r.finish_at(n, &mut self.out),
                    (Some(r), None) => r.finish(&mut self.out),
                    (None, Some(n)) => {
                        let short = n.saturating_sub(self.made) as usize;
                        self.out.resize(short * width, 0);
                    }
                    (None, None) => {}
                }
                self.done = true;
            }
            Pull::Waiting => return Made::Waiting,
            Pull::Failed(why) => return Made::Failed(why),
            Pull::Missing => return Made::Missing,
        }
        let mut frames = self.out.len() / self.width();
        if let Some(n) = self.length() {
            let left = n.saturating_sub(self.made) as usize;
            if frames >= left {
                frames = left;
                self.done = true;
            }
        }
        self.lay(frames);
        put(self.first + self.made as i64, &self.laid);
        self.made += frames as u64;
        Made::Frames
    }

    /// The stems in `feed` resampled (or requantized) onto `out`.
    fn put_feed(&mut self) {
        match self.pass.as_mut().and_then(|p| p.resampler.as_mut()) {
            Some(r) => r.process_float(&self.feed, &mut self.out),
            None => {
                let (q, gain) = (&mut self.requantizer, self.line.gain);
                self.out
                    .extend(self.feed.iter().map(|&y| q.quantize(y * gain)));
            }
        }
    }

    /// Lays the first `frames` made out in the track's channels, the lanes
    /// not made silent.
    fn lay(&mut self, frames: usize) {
        let (width, from) = (self.width(), self.from());
        self.laid.clear();
        if self.channels == self.pairs && from == 0 {
            self.laid.extend_from_slice(&self.out[..frames * width]);
            return;
        }
        self.laid.resize(frames * STEMS * self.channels, 0);
        for (made, to) in self
            .out
            .chunks_exact(width)
            .take(frames)
            .zip(self.laid.chunks_exact_mut(STEMS * self.channels))
        {
            let lanes = to[from * self.channels..].chunks_exact_mut(self.channels);
            for (stem, part) in made.chunks_exact(self.pairs).zip(lanes) {
                part[..self.pairs].copy_from_slice(stem);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(rate: u32, out: u32, seg_out: u64, seg_frame: u64) -> Line {
        Line {
            segment: 0,
            begun: SegStart::Cold,
            rate,
            out,
            seg_out,
            seg_frame,
            end: None,
            gain: 1.0,
            along: None,
        }
    }

    /// The run frame at output frame `o`, as a fraction.
    fn exact(line: &Line, start: u64, o: i128) -> f64 {
        let t = line.seg_frame as f64 / f64::from(line.rate)
            + (o - i128::from(line.seg_out)) as f64 / f64::from(line.out);
        (t - start as f64 / f64::from(line.rate)) * f64::from(STEM_RATE)
    }

    #[test]
    fn run_frames_fall_on_output_frames_on_the_grid() {
        for (rate, out, seg_out, seg_frame) in [
            (44_100, 44_100, 5, 100),
            (44_100, 48_000, 1_000, 12_345),
            (44_100, 192_000, 77, 3),
            (48_000, 48_000, 10, 4_000),
            (96_000, 44_100, 500, 96_007),
            (192_000, 192_000, 0, 1_234_567),
        ] {
            let l = line(rate, out, seg_out, seg_frame);
            let start = run_start(&l, seg_frame.saturating_sub(u64::from(rate) / 3));
            assert!(start <= seg_frame);
            let grid = Grid::of(&l, start).expect("a grid");
            for x in [0i128, 999, 123_456] {
                let x = x + i128::from(seg_out);
                let (o, r) = grid.at_or_before(x);
                assert!(o <= x && o > x - grid.period, "{o} {x}");
                let want = exact(&l, start, o);
                assert!(
                    (want - r as f64).abs() < 1e-6,
                    "{rate}→{out}: {want} vs {r}"
                );
                let (o2, r2) = grid.first_at_run(r + 1);
                assert!(r2 > r && o2 > o);
                assert!((exact(&l, start, o2) - r2 as f64).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn a_run_from_the_segments_frame_meets_it_at_its_start() {
        let l = line(44_100, 48_000, 2_000, 9_999);
        let grid = Grid::of(&l, 9_999).unwrap();
        assert_eq!(grid.at_or_before(2_000), (2_000, 0));
        // A 44.1 kHz file's segment start is on any run's grid.
        let grid = Grid::of(&l, 1_234).unwrap();
        assert_eq!(grid.at_or_before(2_000).0, 2_000);
        // Off the separator's frames: no output frame is a run frame.
        let l = line(96_000, 44_100, 0, 1);
        assert_eq!(Grid::of(&l, 0), None);
    }

    #[test]
    fn a_line_is_the_same_further_along_it() {
        let a = line(44_100, 48_000, 1_000, 0);
        let mut b = a.clone();
        (b.seg_out, b.seg_frame) = (1_160, 147);
        assert!(a.same(&b));
        b.seg_frame = 148;
        assert!(!a.same(&b));
        let mut c = a.clone();
        c.gain = 0.5;
        assert!(!a.same(&c));
        assert_eq!(a.frame_at(1_160), 147);
        assert_eq!(a.frame_at(0), 0);
    }
}
