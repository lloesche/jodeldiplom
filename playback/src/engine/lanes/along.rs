//! The parts of a head read along a time map ([`super::super::along`]):
//! its stems read along the map ([`Stream`] on a line with a map), from the
//! run that has them there (one under way goes on), else from one started
//! when the move was asked for; and main, their sum.

use std::path::Path;
use std::time::Duration;

use crate::stems::{StemRun, Unkept, RATE as STEM_RATE};

use super::super::track::{HeardWarp, SegStart, Track};
use super::super::PREROLL_RESAMPLE;
use super::stream::{Line, Stream};
use super::{Making, StemLanes, Stopped, Want};

/// Seconds of stems before the map's first read position (the reader's
/// interpolation reaches back into them).
const MARGIN: f64 = 0.05;

/// The 24-bit range main is summed into.
const TOP: i64 = (1 << 23) - 1;

/// The run frame a run has the track's second `read` at, when it is a run
/// of `path` that may still have it (kept, or within `reach` frames of what
/// it kept).
fn usable(path: &Path, read: f64, rate: f64, reach: u64) -> impl Fn(&StemRun) -> Option<u64> + '_ {
    move |run: &StemRun| {
        let r = ((read - run.start as f64 / rate) * f64::from(STEM_RATE)).floor() as i64;
        let fine = run.path == path && !run.failed() && !run.missing();
        let reach = (run.kept_end() + reach.max(u64::from(STEM_RATE))) as i64;
        (fine && r >= run.kept_from() as i64 && r <= reach).then_some(r as u64)
    }
}

/// The output frame a stream read along a map from output frame `from`
/// starts at (a resampler's run-up before it), at `out` Hz.
fn first_of(from: u64, out: u32) -> i64 {
    from as i64 - (PREROLL_RESAMPLE * f64::from(out)).round() as i64
}

impl StemLanes {
    /// Starts the parts of a head read along a map (`line` has one) from
    /// its segment's first frame (or the frame written next, should that be
    /// later): a resampler's run-up before it, into what is there,
    /// crossfaded as main is; from the run that has the stems there, else
    /// from a new one that gets those not kept as `unkept` says.
    pub(super) fn start_along(&mut self, line: Line, want: &Want, unkept: Unkept) {
        let track = want.track;
        let from = want.from.max(line.seg_out);
        let fade = match line.begun {
            SegStart::Joined { fade } => fade,
            _ => 0,
        };
        let promised = self.timeline.promised(want.from);
        self.buffer
            .restart(from, promised.max(from + fade as u64), fade);
        let first = first_of(from, line.out);
        let (_, read) = line.along_at(first).expect("a line along a map");
        let read = (read - MARGIN).max(0.0);
        let rate = f64::from(line.rate);
        let path = &track.info.path;
        let found = self.take_run(usable(path, read, rate, want.keep.1));
        let (mut run, r) = found.unwrap_or_else(|| {
            let start = (read * rate).floor() as u64;
            (self.run_for(track, start, unkept), 0)
        });
        run.seek(r);
        run.set_keep(want.keep.0, want.keep.1);
        let read_from = run.start as f64 / rate + r as f64 / f64::from(STEM_RATE);
        let channels = usize::from(track.route.channels);
        self.stream = Some(Stream::along(line, first, channels, read_from));
        self.run = Some(run);
    }

    /// A new run of `track`'s stems from track frame `start`, the stems not
    /// kept got as `unkept` says (not asked of the store again where they
    /// failed for it).
    fn run_for(&self, track: &Track, start: u64, unkept: Unkept) -> StemRun {
        let store = self.store.clone().filter(|_| !self.is_off(track));
        StemRun::start(track.info.path.clone(), start, unkept, store)
    }

    /// A move of `track` onto a map (heard as `heard`) from output frame
    /// `from` was asked for: unless a run has its stems there already, one
    /// starts for them now, so they are there when it is made.
    pub(in crate::engine) fn prepare_along(&mut self, track: &Track, heard: &HeardWarp, from: u64) {
        let first = first_of(from, heard.out_rate).max(0) as u64;
        let read = (heard.seconds_at(first) - MARGIN).max(0.0);
        let rate = f64::from(track.info.rate.max(1));
        let usable = usable(&track.info.path, read, rate, 0);
        let there = [&self.run, &self.prestart, &self.ahead]
            .into_iter()
            .any(|slot| slot.as_ref().and_then(&usable).is_some());
        if !there {
            let start = (read * rate).floor() as u64;
            self.ahead = Some(self.run_for(track, start, self.unkept_along(track)));
        }
    }

    /// A move onto a map that was asked for is not made: the run started
    /// for it goes.
    pub(in crate::engine) fn forget_ahead(&mut self) {
        self.ahead = None;
    }

    /// The parts of a head read along a map made until they reach output
    /// frame `target`, the first piece waiting up to `wait` for its stems;
    /// on the line of `want.track`, started for it when it is not the one
    /// made. What was made, or why its stems failed, when they just did
    /// (the next piece reads the whole mix).
    pub(in crate::engine) fn make_along(
        &mut self,
        want: &Want,
        target: u64,
        wait: Duration,
    ) -> Result<Making, String> {
        let line = Line::of(want.track);
        if !self.stream.as_ref().is_some_and(|s| s.line.same(&line)) {
            self.start(line, want, self.unkept_along(want.track));
        }
        match self.make_to(want, target, (wait, usize::MAX)) {
            Ok(making) => Ok(making),
            Err(stopped) => {
                self.give_up(want.track, want.from);
                match stopped {
                    Stopped::Failed(why) => Err(why),
                    Stopped::Missing => Ok(Making::default()),
                }
            }
        }
    }

    /// The head's stems are separated too slowly to be heard on time: it
    /// takes the whole mix from output frame `from` (the parts made stay
    /// as far as they are promised).
    pub(in crate::engine) fn along_whole(&mut self, track: &Track, from: u64) {
        self.give_up(track, from);
    }

    /// Where the head's parts made end, and whether they are all its own
    /// (its stems have ended).
    pub(in crate::engine) fn along_made(&self) -> (u64, bool) {
        let done = self.stream.as_ref().is_some_and(|s| s.done);
        (self.buffer.end(), done)
    }

    /// Whether the head's stems are being separated (so they may fall
    /// behind what is heard).
    pub(in crate::engine) fn along_separating(&self) -> bool {
        self.run.as_ref().is_some_and(StemRun::separating)
    }

    /// Samples the head's parts clipped so far (from their latest start).
    pub(in crate::engine) fn along_clipped(&self) -> u64 {
        self.stream.as_ref().map_or(0, Stream::clipped)
    }

    /// How far (in frames at the stems' rate) the head's parts are off its
    /// map (a stretch steered back onto it).
    pub(in crate::engine) fn along_offset(&self) -> f64 {
        self.stream.as_ref().map_or(0.0, Stream::offset)
    }

    /// Appends the sum of the parts made for output frames `from..to` to
    /// `main` (whole frames of `channels`), within 24 bits, as they were
    /// made (before a new start's crossfade from what was there, which main
    /// makes itself); returns the samples that had to be clipped to it.
    pub(in crate::engine) fn sum_parts(
        &self,
        from: u64,
        to: u64,
        channels: usize,
        main: &mut Vec<i32>,
    ) -> u64 {
        let mut clipped = 0;
        let mut frame = Vec::with_capacity(self.buffer.width());
        main.reserve((to - from) as usize * channels);
        for o in from..to {
            frame.clear();
            self.buffer.copy_fresh(o, &mut frame);
            for c in 0..channels {
                let sum: i64 = frame
                    .iter()
                    .skip(c)
                    .step_by(channels)
                    .map(|&s| i64::from(s))
                    .sum();
                let kept = sum.clamp(-TOP - 1, TOP);
                clipped += u64::from(kept != sum);
                main.push(kept as i32);
            }
        }
        clipped
    }
}
