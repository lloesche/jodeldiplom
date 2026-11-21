//! Mapping the device clock to the stream position.
//!
//! The device plays *device frames*: the stream's frames in order, with
//! silence inserted wherever the ring ran dry. The backend tells the
//! timeline what it handed the device ([`Timeline::submit`]) and, as often
//! as the device reports it, how many device frames have reached the DAC
//! ([`Timeline::report`], or [`Timeline::report_delay`] for APIs that report
//! the frames still queued). [`Timeline::position`] turns that into stream
//! frames played: device frames played, minus the silence among them, plus
//! the frames that were played or discarded before the device timeline last
//! restarted, minus the frames still on their way through a device further
//! down (a mixer's input: [`Timeline::set_downstream`]).

use std::collections::VecDeque;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
struct Anchor {
    device: u64,
    at: Instant,
    extrapolate: bool,
}

#[derive(Debug)]
pub(crate) struct Timeline {
    rate: u32,
    /// Stream frames played or discarded before this device timeline.
    base: u64,
    /// Device frames handed to the device.
    submitted: u64,
    /// Stream frames among them.
    stream_submitted: u64,
    /// Silence ranges `[start, end)` in device frames not yet fully played.
    pads: VecDeque<(u64, u64)>,
    /// Silence frames in ranges already pruned from `pads`.
    pruned_silence: u64,
    anchor: Option<Anchor>,
    /// Device frames played as of the last position query; never decreases.
    played: u64,
    /// Stream frames counted as played above that a device further down
    /// (the mixer's) has not played yet.
    downstream: u64,
    /// How many times the timeline has restarted.
    epoch: u64,
}

impl Timeline {
    pub(crate) fn new(rate: u32) -> Timeline {
        Timeline::starting_at(rate, 0)
    }

    fn starting_at(rate: u32, base: u64) -> Timeline {
        Timeline {
            rate,
            base,
            submitted: 0,
            stream_submitted: 0,
            pads: VecDeque::new(),
            pruned_silence: 0,
            anchor: None,
            played: 0,
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
            let end = start + silence;
            match self.pads.back_mut() {
                Some(last) if last.1 == start => last.1 = end,
                _ => self.pads.push_back((start, end)),
            }
            self.submitted = end;
        }
    }

    /// `device` device frames had reached the DAC at `at`. With
    /// `extrapolate`, the position advances at the nominal rate from there
    /// until the next report (never past what was submitted).
    pub(crate) fn report(&mut self, device: u64, at: Instant, extrapolate: bool) {
        self.anchor = Some(Anchor {
            device: device.min(self.submitted),
            at,
            extrapolate,
        });
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

    fn device_played(&self, now: Instant) -> u64 {
        let Some(a) = self.anchor else {
            return self.played;
        };
        let mut d = a.device;
        if a.extrapolate {
            let elapsed = now.saturating_duration_since(a.at).as_nanos();
            d = d.saturating_add((elapsed * u128::from(self.rate) / 1_000_000_000) as u64);
        }
        d.min(self.submitted).max(self.played)
    }

    /// Stream frames played (or discarded) as of `now`.
    pub(crate) fn position(&mut self, now: Instant) -> u64 {
        let d = self.device_played(now);
        self.played = d;
        while let Some(&(start, end)) = self.pads.front() {
            if end <= d {
                self.pruned_silence += end - start;
                self.pads.pop_front();
            } else {
                break;
            }
        }
        let partial: u64 = self
            .pads
            .iter()
            .take_while(|(start, _)| *start < d)
            .map(|(start, end)| (*end).min(d) - start)
            .sum();
        let stream = d - self.pruned_silence - partial;
        (self.base + stream.min(self.stream_submitted)).saturating_sub(self.downstream)
    }

    /// Restarts the device timeline at zero with `base` stream frames
    /// already played or discarded (after a device reset); nothing is
    /// downstream any more.
    pub(crate) fn restart(&mut self, base: u64) {
        let epoch = self.epoch + 1;
        *self = Timeline::starting_at(self.rate, base);
        self.epoch = epoch;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn plays_stream_frames_minus_silence() {
        let t0 = Instant::now();
        let mut t = Timeline::new(48_000);
        assert_eq!(t.position(t0), 0);
        // Prefill: 100 frames, then 50 stream + 50 silence (an underrun),
        // then 100 stream frames.
        t.submit(100, 0);
        t.submit(50, 50);
        t.submit(100, 0);
        for (device, stream) in [
            (0, 0),
            (100, 100),
            (150, 150),
            (175, 150),
            (200, 150),
            (260, 210),
            (300, 250),
        ] {
            t.report(device, t0, false);
            assert_eq!(t.position(t0), stream, "device {device}");
        }
    }

    #[test]
    fn never_runs_ahead_of_what_was_submitted_or_back() {
        let t0 = Instant::now();
        let mut t = Timeline::new(1000);
        t.submit(100, 0);
        t.report(10, t0, true);
        assert_eq!(t.position(t0 + Duration::from_millis(50)), 60);
        assert_eq!(t.position(t0 + Duration::from_secs(5)), 100);
        // An older report cannot move the position back.
        t.report(20, t0, false);
        assert_eq!(t.position(t0), 100);
    }

    #[test]
    fn trailing_silence_is_not_counted() {
        let t0 = Instant::now();
        let mut t = Timeline::new(1000);
        t.submit(30, 70);
        t.report_delay(0, t0, false);
        assert_eq!(t.position(t0), 30);
        t.submit(0, 100);
        t.submit(10, 0);
        t.report(205, t0, false);
        assert_eq!(t.position(t0), 35);
    }

    #[test]
    fn delay_reports_count_back_from_submitted() {
        let t0 = Instant::now();
        let mut t = Timeline::new(1000);
        t.submit(500, 0);
        t.report_delay(120, t0, false);
        assert_eq!(t.position(t0), 380);
    }

    #[test]
    fn freeze_stops_extrapolation() {
        let t0 = Instant::now();
        let mut t = Timeline::new(1000);
        t.submit(1000, 0);
        t.report(100, t0, true);
        t.freeze();
        assert_eq!(t.position(t0 + Duration::from_millis(300)), 100);
    }

    #[test]
    fn restart_keeps_the_base() {
        let t0 = Instant::now();
        let mut t = Timeline::new(1000);
        t.submit(100, 0);
        t.report(40, t0, false);
        assert_eq!(t.position(t0), 40);
        assert_eq!(t.stream_submitted_total(), 100);
        t.restart(100);
        assert_eq!(t.position(t0), 100);
        t.submit(20, 0);
        t.report(5, t0, false);
        assert_eq!(t.position(t0), 105);
        assert_eq!(t.stream_submitted_total(), 120);
    }

    #[test]
    fn frames_downstream_are_not_played_yet() {
        let t0 = Instant::now();
        let mut t = Timeline::new(1000);
        t.submit(100, 0);
        t.report_delay(0, t0, false);
        t.add_downstream(100);
        assert_eq!(t.position(t0), 0);
        t.set_downstream(0, 30);
        assert_eq!(t.position(t0), 70);
        // It never grows back, and a restart clears it.
        t.set_downstream(0, 50);
        assert_eq!(t.position(t0), 70);
        t.restart(100);
        assert_eq!(t.epoch(), 1);
        assert_eq!(t.position(t0), 100);
        t.set_downstream(0, 0);
        t.submit(10, 0);
        t.report_delay(0, t0, false);
        t.add_downstream(10);
        t.set_downstream(0, 0);
        assert_eq!(t.position(t0), 100);
        t.set_downstream(1, 4);
        assert_eq!(t.position(t0), 106);
    }
}
