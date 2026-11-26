//! The desk's device as the mixer thread runs it: one opening of it, its
//! mix, and the frames of it mixed, given and written.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use super::chunk::HeardLane;
use super::mix::{Made, Mix};
use super::strip::channel_peaks;
use super::{MixShared, Monitor};
use crate::error::Error;
use crate::format::StreamFormat;
use crate::output::{Clock, Output, StreamInfo};
use crate::shared::Shared;

/// How long the device may take to play out what it holds before it is
/// flushed.
const DRAIN_WAIT: Duration = Duration::from_millis(500);

/// One opening of the device, and what has gone to it, its frames counted
/// from the opening's first.
pub(super) struct Device {
    output: Box<dyn Output>,
    pub(super) format: StreamFormat,
    pub(super) info: StreamInfo,
    /// Its stream, as the microphone's port taps it.
    pub(super) taps: Arc<Shared>,
    /// Paused by the mixer (nothing to play).
    paused: bool,
    mix: Mix,
    /// The next frame to mix, and the frames given it (written or in
    /// `out`).
    pub(super) frame: u64,
    given: u64,
    /// Frames for it, and how many of their samples it has taken.
    out: Vec<i32>,
    sent: usize,
    /// The last frames went out as an input wrote them, or as silence.
    pub(super) untouched: bool,
}

impl Device {
    pub(super) fn new(output: Box<dyn Output>) -> Device {
        let info = output.info().clone();
        Device {
            format: info.format,
            mix: Mix::new(info.format),
            taps: output.clock().shared(),
            info,
            output,
            paused: false,
            frame: 0,
            given: 0,
            out: Vec::new(),
            sent: 0,
            untouched: true,
        }
    }

    pub(super) fn channels(&self) -> usize {
        usize::from(self.format.channels)
    }

    /// Everything mixed has been written.
    pub(super) fn written(&self) -> bool {
        self.sent == self.out.len()
    }

    /// Room for chunks of `frames` frames.
    pub(super) fn reserve(&mut self, frames: usize) {
        self.mix.reserve(frames);
        let out = 2 * frames * self.channels();
        self.out.reserve(out.saturating_sub(self.out.len()));
    }

    /// Mixes the chunk of `n` frames that the lanes `heard` give into the
    /// frames for the device: one exact lane heard at exactly 1 (or none)
    /// as written, else the lanes summed.
    pub(super) fn mix<'a, I>(&mut self, n: usize, heard: impl Fn() -> I)
    where
        I: Iterator<Item = HeardLane<'a>>,
    {
        let ch = self.channels();
        let (count, alone) = heard().fold((0, None), |(k, _), l| (k + 1, Some(l)));
        let untouched = match (count, alone) {
            (0, _) => true,
            (1, Some(l)) => l.exact && l.gains.flat() == Some(1.0),
            _ => false,
        };
        self.out.clear();
        self.sent = 0;
        if untouched && !self.mix.at_rest() {
            // What the limiter held back goes first, so these frames can
            // go as they are.
            self.mix.end(true, &mut self.out);
        }
        let made = if untouched && self.mix.at_rest() {
            match alone {
                Some(l) if count == 1 => {
                    self.out.extend_from_slice(l.samples);
                    Made::Passthrough
                }
                _ => {
                    self.out.resize(self.out.len() + n * ch, 0);
                    Made::Silence
                }
            }
        } else {
            self.mix.begin(n * ch);
            for l in heard() {
                self.mix.add(l.samples, l.gains);
            }
            self.mix.end(untouched, &mut self.out);
            Made::Mixed
        };
        self.untouched = made != Made::Mixed;
        self.frame += n as u64;
    }

    /// Gives out what the limiter holds back (all of it written).
    pub(super) fn give_held(&mut self, monitor: &mut Option<Box<dyn Monitor>>, shared: &MixShared) {
        self.out.clear();
        self.sent = 0;
        self.mix.end(true, &mut self.out);
        self.gave(monitor, shared);
    }

    /// What the frames for the device hold goes to it: the monitor sees
    /// it, the master's meters and limiter flag take it in.
    pub(super) fn gave(&mut self, monitor: &mut Option<Box<dyn Monitor>>, shared: &MixShared) {
        let ch = self.channels();
        if let Some(m) = monitor {
            m.mixed(&self.out, ch, self.given);
        }
        let peaks = channel_peaks(&self.out, ch, |s| f64::from(s.unsigned_abs()));
        shared.peaks.raise(&peaks);
        if self.mix.take_limited() {
            shared.limited.store(true, Ordering::Relaxed);
        }
        self.given += (self.out.len() / ch) as u64;
    }

    /// Writes some of what is mixed, waiting up to `wait` for room.
    pub(super) fn write(&mut self, wait: Duration) -> Result<(), Error> {
        let n = self.output.write(&self.out[self.sent..], Some(wait))?;
        self.sent += n * self.channels();
        Ok(())
    }

    /// Its clock (its position counts the device frames heard).
    pub(super) fn clock(&self) -> Clock {
        self.output.clock()
    }

    /// The device frames it has played.
    pub(super) fn position(&self) -> Result<u64, Error> {
        self.output.position()
    }

    /// What the limiter holds back, in frames.
    pub(super) fn held(&self) -> usize {
        self.mix.held()
    }

    /// Samples clipped on it.
    pub(super) fn clipped(&self) -> u64 {
        self.mix.clipped()
    }

    /// Writes what is mixed and what the limiter holds back, and then
    /// drops what it holds (a flush of the one input heard).
    pub(super) fn flush(
        &mut self,
        monitor: &mut Option<Box<dyn Monitor>>,
        shared: &MixShared,
    ) -> Result<(), Error> {
        self.write_all(monitor, shared)?;
        self.output.flush()
    }

    /// Closes it; returns the samples clipped on it.
    pub(super) fn close(mut self) -> u64 {
        let _ = self.output.close();
        self.mix.clipped()
    }

    /// Writes what is mixed and the limiter holds back, waiting as long as
    /// it takes.
    fn write_all(
        &mut self,
        monitor: &mut Option<Box<dyn Monitor>>,
        shared: &MixShared,
    ) -> Result<(), Error> {
        self.set_paused(false)?;
        loop {
            if self.written() {
                if self.mix.held() == 0 {
                    return Ok(());
                }
                self.give_held(monitor, shared);
                continue;
            }
            let n = self
                .output
                .write(&self.out[self.sent..], Some(DRAIN_WAIT))?;
            if n == 0 {
                return Err(Error::Timeout("the device took nothing"));
            }
            self.sent += n * self.channels();
        }
    }

    /// Pauses it (nothing to play) or resumes it.
    pub(super) fn set_paused(&mut self, paused: bool) -> Result<(), Error> {
        if self.paused == paused {
            return Ok(());
        }
        self.paused = paused;
        if paused {
            self.output.pause()
        } else {
            self.output.resume()
        }
    }
}
