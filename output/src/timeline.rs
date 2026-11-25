//! Mapping the device clock to the stream position.
//!
//! The device plays *device frames*: the stream's frames in order, with
//! silence inserted wherever the ring ran dry. The backend tells the
//! timeline what it handed the device ([`Timeline::submit`]) and, as often
//! as the device reports it, how many device frames have reached the DAC
//! ([`Timeline::report`], or [`Timeline::report_delay`] for APIs that report
//! the frames still queued). The position is stream frames played: device
//! frames played, minus the silence among them, plus the frames that were
//! played or discarded before the device timeline last restarted (or
//! passed over: [`Timeline::skip`]), minus the frames still on their way
//! through a device further down (a mixer's input:
//! [`Timeline::set_downstream`]).
//!
//! The [`Timeline`] has one writer at a time (see `shared.rs`): the
//! consumer (the device thread), or the handle while it keeps the consumer
//! out. Nothing in it locks, and its storage is made once. After each
//! change it publishes what a reader needs ([`Snapshot`]): the stream
//! frames played as of its last report, and the runs of silence and stream
//! frames handed to the device after it. A reader on any thread works the
//! position out from that ([`Published::position`]), without a lock, and
//! the device thread never waits for one.

use crate::seqlock::{Packed, SeqLock};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Silence ranges the timeline keeps apart, not yet played. A device
/// reports every buffer, and only what it has not played yet is kept, so
/// a handful are; past this many the newest takes the next one in (its
/// silence counted before the stream frames between them, which only
/// delays those until the device has played the two).
const PADS: usize = 64;

/// The runs after the last report a reader can play into: each silence
/// then the stream frames after it.
const RUNS: usize = 4;

#[derive(Clone, Copy, Debug)]
struct Anchor {
    device: u64,
    at: Instant,
    extrapolate: bool,
}

/// Silence ranges `[start, end)` in device frames, oldest first: a ring of
/// [`PADS`] made once.
#[derive(Debug)]
struct Pads {
    ranges: [(u64, u64); PADS],
    first: usize,
    len: usize,
}

impl Pads {
    fn new() -> Pads {
        Pads {
            ranges: [(0, 0); PADS],
            first: 0,
            len: 0,
        }
    }

    fn iter(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        (0..self.len).map(|k| self.ranges[(self.first + k) % PADS])
    }

    fn front(&self) -> Option<(u64, u64)> {
        (self.len > 0).then(|| self.ranges[self.first])
    }

    fn pop_front(&mut self) {
        self.first = (self.first + 1) % PADS;
        self.len -= 1;
    }

    /// Adds the silence `[start, end)`, after every range kept: onto the
    /// last when it follows it, or (all taken) into the last.
    fn push(&mut self, start: u64, end: u64) {
        if self.len > 0 {
            let last = &mut self.ranges[(self.first + self.len - 1) % PADS];
            if last.1 == start {
                last.1 = end;
                return;
            }
            if self.len == PADS {
                last.1 += end - start;
                return;
            }
        }
        self.ranges[(self.first + self.len) % PADS] = (start, end);
        self.len += 1;
    }

    fn clear(&mut self) {
        self.len = 0;
    }
}

#[derive(Debug)]
pub(crate) struct Timeline {
    /// Stream frames played or discarded before this device timeline, and
    /// those passed over.
    base: u64,
    /// Device frames handed to the device.
    submitted: u64,
    /// Stream frames among them.
    stream_submitted: u64,
    /// Silence not yet fully played (by the last report).
    pads: Pads,
    /// Silence frames in ranges already pruned from `pads`.
    pruned_silence: u64,
    anchor: Option<Anchor>,
    /// Stream frames counted as played above that a device further down
    /// (the mixer's) has not played yet.
    downstream: u64,
    /// How many times the timeline has restarted.
    epoch: u64,
}

impl Timeline {
    pub(crate) fn new() -> Timeline {
        Timeline {
            base: 0,
            submitted: 0,
            stream_submitted: 0,
            pads: Pads::new(),
            pruned_silence: 0,
            anchor: None,
            downstream: 0,
            epoch: 0,
        }
    }

    /// The device took `stream` stream frames followed by `silence` frames
    /// of silence.
    pub(crate) fn submit(&mut self, stream: u64, silence: u64) {
        self.submitted += stream;
        self.stream_submitted += stream;
        if silence > 0 {
            let start = self.submitted;
            self.pads.push(start, start + silence);
            self.submitted += silence;
        }
    }

    /// `frames` stream frames are passed over, in no device frames: they
    /// count as played at once (a mixer's input, whose device is the mixer,
    /// holds them back as downstream until the device is there).
    pub(crate) fn skip(&mut self, frames: u64) {
        self.base += frames;
    }

    /// `device` device frames had reached the DAC at `at`. With
    /// `extrapolate`, the position advances at the nominal rate from there
    /// until the next report (never past what was submitted).
    pub(crate) fn report(&mut self, device: u64, at: Instant, extrapolate: bool) {
        let device = device.min(self.submitted);
        self.anchor = Some(Anchor {
            device,
            at,
            extrapolate,
        });
        while let Some((start, end)) = self.pads.front() {
            if end > device {
                break;
            }
            self.pruned_silence += end - start;
            self.pads.pop_front();
        }
    }

    /// `delay` submitted device frames had not reached the DAC yet at `at`.
    pub(crate) fn report_delay(&mut self, delay: u64, at: Instant, extrapolate: bool) {
        self.report(self.submitted.saturating_sub(delay), at, extrapolate);
    }

    /// Stops extrapolating from the last report (the device has stopped).
    pub(crate) fn freeze(&mut self) {
        if let Some(a) = &mut self.anchor {
            a.extrapolate = false;
        }
    }

    /// Restarts the device timeline at zero with `base` stream frames
    /// already played or discarded (after a device reset); nothing is
    /// downstream any more.
    pub(crate) fn restart(&mut self, base: u64) {
        self.base = base;
        self.submitted = 0;
        self.stream_submitted = 0;
        self.pads.clear();
        self.pruned_silence = 0;
        self.anchor = None;
        self.downstream = 0;
        self.epoch += 1;
    }

    /// How many times the timeline has restarted.
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch
    }

    /// `frames` more stream frames went downstream (counted as played
    /// here, not heard yet).
    pub(crate) fn add_downstream(&mut self, frames: u64) {
        self.downstream += frames;
    }

    /// Only `frames` of what went downstream since restart `epoch` are not
    /// played yet (nothing changes after a later restart; the count never
    /// grows here, so the position never goes back).
    pub(crate) fn set_downstream(&mut self, epoch: u64, frames: u64) {
        if epoch == self.epoch {
            self.downstream = self.downstream.min(frames);
        }
    }

    /// Stream frames handed to the device, including those before the
    /// last restart: the position once the device has played everything.
    #[cfg_attr(windows, allow(dead_code))]
    pub(crate) fn stream_submitted_total(&self) -> u64 {
        self.base + self.stream_submitted
    }

    /// Publishes the timeline as it stands for the readers of `to`.
    pub(crate) fn publish(&self, to: &Published) {
        let (device, at, extrapolate) = match self.anchor {
            Some(a) => (a.device, Some(a.at), a.extrapolate),
            None => (0, None, false),
        };
        // The silence before the anchor, and after it the runs: silence,
        // then the stream frames up to the next silence (or the end).
        let mut silence = self.pruned_silence;
        let mut runs = [(0, 0); RUNS];
        let mut run = 0;
        let mut cursor = device;
        for (start, end) in self.pads.iter() {
            if start < device {
                silence += end.min(device) - start;
            }
            if end <= device {
                continue;
            }
            let start = start.max(device);
            if start > cursor {
                // The stream frames before this silence close the run.
                runs[run].1 = start - cursor;
                run += 1;
                if run == RUNS {
                    break;
                }
            }
            runs[run].0 = end - start;
            cursor = end;
        }
        if run < RUNS {
            runs[run].1 = self.submitted - cursor;
        }
        // (A report that went back may leave pruned silence after it.)
        let stream = device.saturating_sub(silence);
        to.snapshot.store(&Snapshot {
            played: self.base + stream.min(self.stream_submitted),
            downstream: self.downstream,
            at: at.map_or(0, |at| to.since_origin(at)),
            extrapolate: extrapolate && at.is_some(),
            runs,
        });
    }
}

/// What the timeline publishes for its readers.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Snapshot {
    /// Stream frames played as of the last report, the base counted in.
    played: u64,
    /// Of them, those still downstream.
    downstream: u64,
    /// The report's time, in nanoseconds from [`Published::origin`]
    /// (before it, below 0).
    at: i64,
    /// The position advances from the report at the nominal rate.
    extrapolate: bool,
    /// After the report: silence, then stream frames, [`RUNS`] times.
    runs: [(u64, u64); RUNS],
}

impl Packed for Snapshot {
    const WORDS: usize = 4 + 2 * RUNS;

    fn pack(&self, words: &mut [u64]) {
        words[0] = self.played;
        words[1] = self.downstream;
        words[2] = self.at as u64;
        words[3] = u64::from(self.extrapolate);
        for (pair, &(silence, run)) in words[4..].chunks_exact_mut(2).zip(&self.runs) {
            pair[0] = silence;
            pair[1] = run;
        }
    }

    fn unpack(words: &[u64]) -> Snapshot {
        let mut runs = [(0, 0); RUNS];
        for (r, pair) in runs.iter_mut().zip(words[4..].chunks_exact(2)) {
            *r = (pair[0], pair[1]);
        }
        Snapshot {
            played: words[0],
            downstream: words[1],
            at: words[2] as i64,
            extrapolate: words[3] != 0,
            runs,
        }
    }
}

/// What the timeline publishes, and the position read from it: never
/// back, whatever the reports say.
pub(crate) struct Published {
    rate: u32,
    origin: Instant,
    snapshot: SeqLock<Snapshot>,
    /// The highest position read so far.
    floor: AtomicU64,
}

impl Published {
    pub(crate) fn new(rate: u32) -> Published {
        Published {
            rate,
            origin: Instant::now(),
            snapshot: SeqLock::default(),
            floor: AtomicU64::new(0),
        }
    }

    /// Nanoseconds from the origin to `t` (before it, below 0).
    fn since_origin(&self, t: Instant) -> i64 {
        match t.checked_duration_since(self.origin) {
            Some(d) => d.as_nanos() as i64,
            None => -(self.origin.duration_since(t).as_nanos() as i64),
        }
    }

    /// Stream frames played (or discarded) as of `now`.
    pub(crate) fn position(&self, now: Instant) -> u64 {
        let s = self.snapshot.load();
        let mut stream = s.played;
        if s.extrapolate {
            let elapsed = self.since_origin(now).saturating_sub(s.at).max(0);
            let mut ahead = (elapsed as u128 * u128::from(self.rate) / 1_000_000_000) as u64;
            for (silence, run) in s.runs {
                if ahead <= silence {
                    break;
                }
                ahead -= silence;
                let played = ahead.min(run);
                stream += played;
                ahead -= played;
            }
        }
        let position = stream.saturating_sub(s.downstream);
        self.floor
            .fetch_max(position, Ordering::AcqRel)
            .max(position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// A timeline at `rate`, and what it publishes.
    struct Seen {
        t: Timeline,
        p: Published,
    }

    impl Seen {
        fn new(rate: u32) -> Seen {
            Seen {
                t: Timeline::new(),
                p: Published::new(rate),
            }
        }

        fn change(&mut self, f: impl FnOnce(&mut Timeline)) {
            f(&mut self.t);
            self.t.publish(&self.p);
        }

        fn position(&self, now: Instant) -> u64 {
            self.p.position(now)
        }
    }

    #[test]
    fn plays_stream_frames_minus_silence() {
        let mut s = Seen::new(48_000);
        let t0 = Instant::now();
        assert_eq!(s.position(t0), 0);
        // Prefill: 100 frames, then 50 stream + 50 silence (an underrun),
        // then 100 stream frames.
        s.change(|t| {
            t.submit(100, 0);
            t.submit(50, 50);
            t.submit(100, 0);
        });
        for (device, stream) in [
            (0, 0),
            (100, 100),
            (150, 150),
            (175, 150),
            (200, 150),
            (260, 210),
            (300, 250),
        ] {
            s.change(|t| t.report(device, t0, false));
            assert_eq!(s.position(t0), stream, "device {device}");
        }
    }

    #[test]
    fn extrapolates_through_silence_and_the_stream_after_it() {
        let mut s = Seen::new(1000);
        let t0 = Instant::now();
        // 100 stream, 50 silence, 30 stream, 20 silence, 100 stream.
        s.change(|t| {
            t.submit(100, 50);
            t.submit(30, 20);
            t.submit(100, 0);
            t.report(80, t0, true);
        });
        let at = |s: &Seen, ms| s.position(t0 + Duration::from_millis(ms));
        for (ms, stream) in [
            (0, 80),
            (20, 100),
            (60, 100),
            (80, 110),
            (110, 130),
            (120, 130),
            (130, 140),
            (150, 160),
            (5_000, 230),
        ] {
            assert_eq!(at(&s, ms), stream, "{ms} ms");
        }
        // From inside a silence on.
        let mut s = Seen::new(1000);
        let t1 = Instant::now();
        s.change(|t| {
            t.submit(100, 50);
            t.submit(30, 20);
            t.submit(100, 0);
            t.report(120, t1, true);
        });
        for (ms, stream) in [(0, 100), (30, 100), (40, 110), (70, 130), (95, 145)] {
            let position = s.position(t1 + Duration::from_millis(ms));
            assert_eq!(position, stream, "{ms} ms");
        }
    }

    #[test]
    fn never_runs_ahead_of_what_was_submitted_or_back() {
        let mut s = Seen::new(1000);
        let t0 = Instant::now();
        s.change(|t| {
            t.submit(100, 0);
            t.report(10, t0, true);
        });
        assert_eq!(s.position(t0 + Duration::from_millis(50)), 60);
        assert_eq!(s.position(t0 + Duration::from_secs(5)), 100);
        // An older report cannot move the position back.
        s.change(|t| t.report(20, t0, false));
        assert_eq!(s.position(t0), 100);
    }

    #[test]
    fn trailing_silence_is_not_counted() {
        let mut s = Seen::new(1000);
        let t0 = Instant::now();
        s.change(|t| {
            t.submit(30, 70);
            t.report_delay(0, t0, false);
        });
        assert_eq!(s.position(t0), 30);
        s.change(|t| {
            t.submit(0, 100);
            t.submit(10, 0);
            t.report(205, t0, false);
        });
        assert_eq!(s.position(t0), 35);
    }

    #[test]
    fn delay_reports_count_back_from_submitted() {
        let mut s = Seen::new(1000);
        let t0 = Instant::now();
        s.change(|t| {
            t.submit(500, 0);
            t.report_delay(120, t0, false);
        });
        assert_eq!(s.position(t0), 380);
    }

    #[test]
    fn freeze_stops_extrapolation() {
        let mut s = Seen::new(1000);
        let t0 = Instant::now();
        s.change(|t| {
            t.submit(1000, 0);
            t.report(100, t0, true);
            t.freeze();
        });
        assert_eq!(s.position(t0 + Duration::from_millis(300)), 100);
    }

    #[test]
    fn restart_keeps_the_base() {
        let mut s = Seen::new(1000);
        let t0 = Instant::now();
        s.change(|t| {
            t.submit(100, 0);
            t.report(40, t0, false);
        });
        assert_eq!(s.position(t0), 40);
        assert_eq!(s.t.stream_submitted_total(), 100);
        s.change(|t| t.restart(100));
        assert_eq!(s.position(t0), 100);
        s.change(|t| {
            t.submit(20, 0);
            t.report(5, t0, false);
        });
        assert_eq!(s.position(t0), 105);
        assert_eq!(s.t.stream_submitted_total(), 120);
    }

    #[test]
    fn frames_passed_over_count_as_played() {
        let mut s = Seen::new(1000);
        let t0 = Instant::now();
        s.change(|t| {
            t.submit(100, 0);
            t.report(60, t0, false);
            t.skip(40);
        });
        assert_eq!(s.position(t0), 100);
        s.change(|t| {
            t.submit(50, 0);
            t.report(150, t0, false);
        });
        assert_eq!(s.position(t0), 190);
        assert_eq!(s.t.stream_submitted_total(), 190);
    }

    #[test]
    fn frames_downstream_are_not_played_yet() {
        let mut s = Seen::new(1000);
        let t0 = Instant::now();
        s.change(|t| {
            t.submit(100, 0);
            t.report_delay(0, t0, false);
            t.add_downstream(100);
        });
        assert_eq!(s.position(t0), 0);
        s.change(|t| t.set_downstream(0, 30));
        assert_eq!(s.position(t0), 70);
        // It never grows back, and a restart clears it.
        s.change(|t| t.set_downstream(0, 50));
        assert_eq!(s.position(t0), 70);
        s.change(|t| t.restart(100));
        assert_eq!(s.t.epoch(), 1);
        assert_eq!(s.position(t0), 100);
        s.change(|t| {
            t.set_downstream(0, 0);
            t.submit(10, 0);
            t.report_delay(0, t0, false);
            t.add_downstream(10);
            t.set_downstream(0, 0);
        });
        assert_eq!(s.position(t0), 100);
        s.change(|t| t.set_downstream(1, 4));
        assert_eq!(s.position(t0), 106);
    }

    #[test]
    fn underruns_past_its_room_lose_no_silence() {
        let mut t = Timeline::new();
        let t0 = Instant::now();
        let n = 10 * PADS as u64;
        for _ in 0..3 {
            // An underrun a buffer, none played: full, the last range takes
            // the rest in, and nothing silent is lost.
            for _ in 0..n {
                t.submit(10, 5);
            }
            assert_eq!(t.pads.len, PADS);
            let silence: u64 = t.pads.iter().map(|(s, e)| e - s).sum();
            assert_eq!(silence, 5 * n);
            t.restart(0);
            assert_eq!(t.pads.len, 0);
        }
        for _ in 0..n {
            t.submit(10, 5);
        }
        t.report_delay(0, t0, false);
        assert_eq!(t.pads.len, 0);
        let p = Published::new(1000);
        t.publish(&p);
        assert_eq!(p.position(t0), 10 * n);
    }
}
