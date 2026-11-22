//! The output interface every backend and test output implements.

use crate::caps::Capabilities;
use crate::error::Error;
use crate::format::{SampleFormat, StreamFormat};
use crate::lanes::lane_format;
use crate::pack::{pack, PackMode, PackReport};
use crate::shared::Shared;
use crate::tap::Tap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// What to open: the stream's rate and channel count, the formats to try
/// and the buffering around the device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenRequest {
    /// Frames per second. The device is opened at exactly this rate or not
    /// at all.
    pub rate: u32,
    /// Samples per frame.
    pub channels: u16,
    /// Formats to try, most wanted first; the first one the device accepts
    /// exclusively is used. Defaults to [`SampleFormat::PREFERENCE`].
    pub formats: Vec<SampleFormat>,
    /// The device period to ask for (WASAPI buffer duration, ALSA period,
    /// Core Audio IO buffer). The device may round it, and never goes below
    /// its minimum. Defaults to 10 ms.
    pub period: Duration,
    /// The ring between the writer and the device. Defaults to 500 ms.
    pub buffer: Duration,
    /// How much must be queued before the device starts on its own.
    /// [`Output::drain`] and a full ring start it earlier. Defaults to
    /// 100 ms.
    pub prefill: Duration,
    /// What [`Output::write`] does with samples the format cannot hold.
    /// Defaults to [`PackMode::Saturate`] (clamped and counted in
    /// [`Stats`]).
    pub pack_mode: PackMode,
    /// The lanes written side by side (see [`crate::lanes`]): 1, a plain
    /// stream (the default), or what the target offers
    /// ([`OutputTarget::lanes`]). `channels` are each lane's.
    pub lanes: u16,
}

impl OpenRequest {
    /// A request for `rate` and `channels` with the defaults.
    pub fn new(rate: u32, channels: u16) -> Self {
        OpenRequest {
            rate,
            channels,
            formats: SampleFormat::PREFERENCE.to_vec(),
            period: Duration::from_millis(10),
            buffer: Duration::from_millis(500),
            prefill: Duration::from_millis(100),
            pack_mode: PackMode::Saturate,
            lanes: 1,
        }
    }

    /// Sets the formats to try, in order.
    pub fn formats(mut self, formats: &[SampleFormat]) -> Self {
        self.formats = formats.to_vec();
        self
    }

    /// Sets the device period.
    pub fn period(mut self, period: Duration) -> Self {
        self.period = period;
        self
    }

    /// Sets the ring length.
    pub fn buffer(mut self, buffer: Duration) -> Self {
        self.buffer = buffer;
        self
    }

    /// Sets the start threshold.
    pub fn prefill(mut self, prefill: Duration) -> Self {
        self.prefill = prefill;
        self
    }

    /// Sets the pack mode.
    pub fn pack_mode(mut self, mode: PackMode) -> Self {
        self.pack_mode = mode;
        self
    }

    /// Sets the lanes (see [`OpenRequest::lanes`]).
    pub fn lanes(mut self, lanes: u16) -> Self {
        self.lanes = lanes;
        self
    }

    /// Whether a target offering `offered` lanes ([`OutputTarget::lanes`])
    /// can open this request: [`Error::InvalidArgument`] for more lanes than
    /// it offers. Every target checks it as it opens.
    pub fn check_lanes(&self, offered: u16) -> Result<(), Error> {
        if self.lanes > offered.max(1) {
            return Err(Error::InvalidArgument(format!(
                "{} lanes asked of a target that takes {}",
                self.lanes,
                offered.max(1)
            )));
        }
        Ok(())
    }

    pub(crate) fn validate(&self) -> Result<(), Error> {
        if self.rate == 0 || self.channels == 0 {
            return Err(Error::InvalidArgument(
                "rate and channels must be non-zero".into(),
            ));
        }
        if self.formats.is_empty() {
            return Err(Error::InvalidArgument("no sample formats to try".into()));
        }
        if self.lanes == 0 {
            return Err(Error::InvalidArgument(
                "a stream has one lane or more".into(),
            ));
        }
        Ok(())
    }

    /// The ring length in frames for a device buffer of `device_frames`:
    /// at least two device buffers.
    pub(crate) fn ring_frames(&self, format: &StreamFormat, device_frames: u64) -> usize {
        format
            .duration_to_frames(self.buffer)
            .max(2 * device_frames)
            .max(1) as usize
    }
}

/// What an open stream runs at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamInfo {
    /// The rate, channels and the format the device accepted.
    pub format: StreamFormat,
    /// Which backend plays it.
    pub backend: &'static str,
    /// Whether the device is held exclusively (bit-perfect path). Test
    /// outputs say `false`.
    pub exclusive: bool,
    /// The device buffer (one period) in frames.
    pub device_buffer_frames: u64,
    /// The latency the device adds after its buffer, as it reports it.
    pub device_latency: Duration,
    /// The ring length in frames.
    pub ring_frames: u64,
    /// The device plays in real time, on a clock of its own: it takes
    /// frames as it plays them, ready or not. A file or a test's memory
    /// takes what it is given at once (or when told), and says `false`.
    pub realtime: bool,
    /// The device's IO boundary takes 32-bit float (Core Audio, when its
    /// virtual format is not integer). The samples are converted exactly:
    /// every 24-bit value is a float the device turns back into the same
    /// integer.
    pub float_io: bool,
}

/// Counters since the output was opened.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Times the ring ran dry while the device was playing and more was
    /// expected (each run of padded buffers counts once).
    pub underruns: u64,
    /// Frames of silence inserted for those underruns.
    pub underrun_frames: u64,
    /// Times the device itself ran dry (an xrun the driver reported).
    pub device_underruns: u64,
    /// Samples clamped because they were outside 24 bits.
    pub clipped: u64,
    /// Samples whose low bits were dropped for 16-bit output.
    pub truncated: u64,
}

/// A device stream, or a test stand-in for one.
///
/// The shape is push-in, pull-out: [`Output::write`] packs samples into a
/// ring buffer in the device format, and the backend drains the ring from
/// its own thread whenever the device asks for a buffer (the WASAPI event,
/// the ALSA period, the Core Audio IOProc). When the ring is empty the
/// device gets silence and the gap is counted in [`Stats`]. The device
/// starts by itself once [`OpenRequest::prefill`] is queued.
///
/// Samples are the engine's `i32` values (24-bit, see [`crate::pack`]),
/// interleaved, `channels` per frame.
///
/// Positions count stream frames: [`Output::position`] is how many of the
/// frames written have been played by the device (or discarded by
/// [`Output::flush`]), from the device clock. It never decreases;
/// [`Output::written`] goes back by what [`Output::retract`] takes back.
///
/// The handle is `Send`; [`Output::clock`] gives a cheap clone for other
/// threads that only read the position and counters.
pub trait Output: Send {
    /// What the stream runs at.
    fn info(&self) -> &StreamInfo;

    /// The stream format (`info().format`).
    fn format(&self) -> StreamFormat {
        self.info().format
    }

    /// The lanes each frame written holds (see [`crate::lanes`]): 1 for a
    /// plain stream, whose frames are [`Output::format`]'s channels.
    fn lanes(&self) -> u16 {
        1
    }

    /// Queues `samples` (whole frames: [`Output::format`]'s channels each,
    /// or [`crate::lanes::lane_width`] of them for a stream of more than
    /// one lane). Blocks while the ring is full, up to
    /// `timeout` (`None`: until everything is queued), and returns the
    /// frames queued. It returns early, with what fitted, while paused.
    ///
    /// In [`PackMode::Exact`] a sample that does not fit fails the call
    /// before anything is queued.
    fn write(&mut self, samples: &[i32], timeout: Option<Duration>) -> Result<usize, Error>;

    /// Starts the device if it has not started, and waits up to `timeout`
    /// (`None`: no limit) until every written frame has been played.
    /// Returns whether it got there. The device keeps running (on silence,
    /// which is not counted as underrun) until the next write.
    fn drain(&mut self, timeout: Option<Duration>) -> Result<bool, Error>;

    /// Stream frames played, from the device clock.
    fn position(&self) -> Result<u64, Error>;

    /// Stream frames accepted by [`Output::write`].
    fn written(&self) -> u64;

    /// How long until a frame written now is heard: the frames written but
    /// not yet played, as time.
    fn latency(&self) -> Duration;

    /// Frames written that the device has not taken yet: what its ring
    /// holds, so how long the device can go on without more. What it has
    /// taken into a buffer of its own, or holds as latency, is not counted
    /// (it is in [`Output::latency`]).
    fn queued(&self) -> u64;

    /// Stops the device; queued frames stay queued.
    fn pause(&mut self) -> Result<(), Error>;

    /// Restarts the device after [`Output::pause`].
    fn resume(&mut self) -> Result<(), Error>;

    /// Whether the output is paused.
    fn is_paused(&self) -> bool;

    /// Discards every frame not yet played (for a seek). Afterwards
    /// `position() == written()`. A running device starts again once the
    /// prefill is queued; a paused one stays paused.
    fn flush(&mut self) -> Result<(), Error>;

    /// Takes back up to `frames` of the frames written last that the device
    /// has not taken yet, keeping enough queued that it never runs dry and
    /// is never stopped (for a jump heard at once, without the gap of
    /// [`Output::flush`]). Afterwards [`Output::written`] is that much
    /// smaller. Returns how many frames it took back (none, for outputs
    /// that cannot).
    fn retract(&mut self, frames: u64) -> Result<u64, Error> {
        let _ = frames;
        Ok(0)
    }

    /// Sets the [`Tap`] the device thread shows every buffer it hands the
    /// device (a microphone's echo canceller taking what plays, see
    /// [`crate::input`]), or removes it with `None`. A tap only sees: the
    /// device gets exactly the bytes written, tap or not. The device thread
    /// takes the new tap at its next buffer, without a lock; the old one is
    /// dropped on this thread before the call returns. An output keeps its
    /// tap until it is closed, so after reopening one, set it again.
    ///
    /// Outputs that run no taps say [`Error::Unsupported`].
    fn set_tap(&mut self, tap: Option<Box<dyn Tap>>) -> Result<(), Error> {
        drop(tap);
        Err(Error::Unsupported("this output runs no taps".into()))
    }

    /// Counters since open.
    fn stats(&self) -> Stats;

    /// A read-only view for other threads.
    fn clock(&self) -> Clock;

    /// Stops and releases the device. Later calls return [`Error::Closed`]
    /// (except the read-only ones). Dropping the output closes it too.
    fn close(&mut self) -> Result<(), Error>;
}

/// Something outputs are opened on: a device, a [`crate::mixer`]'s input,
/// a test's memory.
pub trait OutputTarget: Send {
    /// A name for people (the device name).
    fn name(&self) -> String;

    /// What it accepts exclusively.
    fn capabilities(&mut self) -> Result<Capabilities, Error>;

    /// Opens an output for `request`.
    fn open(&mut self, request: &OpenRequest) -> Result<Box<dyn Output>, Error>;

    /// The most lanes an output opened on it takes ([`OpenRequest::lanes`]):
    /// 1 unless it says more.
    fn lanes(&mut self) -> u16 {
        1
    }
}

/// A read-only, cloneable view of an output's clock and counters, for
/// threads other than the writer.
#[derive(Clone)]
pub struct Clock {
    shared: Arc<Shared>,
    format: StreamFormat,
}

impl Clock {
    /// Stream frames played (see [`Output::position`]).
    pub fn position(&self) -> u64 {
        self.shared.position()
    }

    /// Stream frames written.
    pub fn written(&self) -> u64 {
        self.shared.written()
    }

    /// The stream format (as [`Output::format`]).
    pub fn format(&self) -> StreamFormat {
        self.format
    }

    /// Counters since open.
    pub fn stats(&self) -> Stats {
        self.shared.stats()
    }

    /// The error that failed the device, if one did.
    pub fn failure(&self) -> Option<Error> {
        self.shared.failure()
    }
}

/// What a backend does to its device; the handle logic around it is shared.
pub(crate) trait Driver: Send {
    /// Starts (or resumes) the device pulling from the ring.
    fn start(&mut self) -> Result<(), Error>;
    /// Stops the device. Queued frames stay queued; frames the device
    /// cannot keep are passed to [`Shared::skip_device_queue`].
    fn pause(&mut self) -> Result<(), Error>;
    /// Stops the device, calls [`Shared::discard_all`] and resets the
    /// device's own position.
    fn flush(&mut self) -> Result<(), Error>;
    /// Stops and releases the device.
    fn close(&mut self) -> Result<(), Error>;
    /// Frames were just queued.
    fn after_write(&mut self) -> Result<(), Error> {
        Ok(())
    }
    /// Whether the device pulls buffers on its own while started (a test
    /// device may only play when told to).
    fn pulls(&self) -> bool {
        true
    }
}

/// The handle logic shared by every output: packing, backpressure, start
/// threshold, pause state, drain and close. Frames are written and packed
/// in the ring's format (`shared.format`): the device's, or a lane
/// stream's ([`crate::lanes`]).
pub(crate) struct Core {
    pub(crate) shared: Arc<Shared>,
    driver: Box<dyn Driver>,
    info: StreamInfo,
    /// The lanes of each frame (1 unless the ring is a lane stream's).
    lanes: u16,
    mode: PackMode,
    prefill: u64,
    running: bool,
    paused: bool,
    started: bool,
    closed: bool,
}

impl Core {
    pub(crate) fn new(
        shared: Arc<Shared>,
        driver: Box<dyn Driver>,
        info: StreamInfo,
        request: &OpenRequest,
    ) -> Core {
        let prefill = info
            .format
            .duration_to_frames(request.prefill)
            .max(1)
            .min(shared.ring.capacity());
        // The opener checked the lanes against what it offers
        // ([`OpenRequest::check_lanes`]) and made the ring for them.
        let lanes = request.lanes;
        debug_assert_eq!(shared.format, lane_format(info.format, lanes));
        Core {
            shared,
            driver,
            info,
            lanes,
            mode: request.pack_mode,
            prefill,
            running: false,
            paused: false,
            started: false,
            closed: false,
        }
    }

    pub(crate) fn set_prefill(&mut self, frames: u64) {
        self.prefill = frames.min(self.shared.ring.capacity());
    }

    /// Sets the latency a simulated device says it adds.
    pub(crate) fn set_device_latency(&mut self, latency: Duration) {
        self.info.device_latency = latency;
    }

    fn usable(&self) -> Result<(), Error> {
        if self.closed {
            return Err(Error::Closed);
        }
        self.shared.check()
    }

    fn start(&mut self) -> Result<(), Error> {
        if !self.running {
            self.driver.start()?;
            self.running = true;
            self.started = true;
        }
        Ok(())
    }

    fn maybe_start(&mut self) -> Result<(), Error> {
        if !self.running && !self.paused && self.shared.ring.len() >= self.prefill {
            self.start()?;
        }
        Ok(())
    }

    fn push(&mut self, samples: &[i32]) -> Result<(), Error> {
        let ch = usize::from(self.shared.format.channels);
        let sample = self.shared.format.sample;
        let frames = (samples.len() / ch) as u64;
        let mode = self.mode;
        let report = self.shared.ring.push_with(frames, |a, b| {
            let split = a.len() / sample.bytes();
            let mut report = PackReport::default();
            report.add(pack(&samples[..split], sample, mode, a)?);
            report.add(pack(&samples[split..], sample, mode, b)?);
            Ok::<_, Error>(report)
        })?;
        self.shared
            .note_written(frames, report.clipped, report.truncated);
        Ok(())
    }
}

impl Output for Core {
    fn info(&self) -> &StreamInfo {
        &self.info
    }

    fn lanes(&self) -> u16 {
        self.lanes
    }

    fn write(&mut self, samples: &[i32], timeout: Option<Duration>) -> Result<usize, Error> {
        self.usable()?;
        let ch = usize::from(self.shared.format.channels);
        if samples.len() % ch != 0 {
            return Err(Error::InvalidArgument(format!(
                "{} samples is not a whole number of {ch}-channel frames",
                samples.len()
            )));
        }
        if self.mode == PackMode::Exact {
            // Check everything first so a failing call queues nothing: a
            // lane stream's main against the device's format (it may reach
            // the device as written), every sample against the ring's.
            crate::pack::check(samples, self.shared.format.sample)?;
            if self.lanes > 1 {
                let main = usize::from(self.info.format.channels);
                for (f, frame) in samples.chunks_exact(ch).enumerate() {
                    crate::pack::check(&frame[..main], self.info.format.sample)
                        .map_err(|e| e.at(f * ch))?;
                }
            }
        }
        let total = samples.len() / ch;
        let deadline = timeout.map(|t| Instant::now() + t);
        let mut done = 0usize;
        loop {
            self.shared.check()?;
            let n = (self.shared.ring.free() as usize).min(total - done);
            if n > 0 {
                self.push(&samples[done * ch..(done + n) * ch])?;
                done += n;
                self.driver.after_write()?;
                self.maybe_start()?;
            }
            if done == total {
                break;
            }
            if !self.running && !self.paused {
                // The ring is full but below the prefill: start anyway.
                self.start()?;
            }
            if self.paused {
                break;
            }
            let wait = match deadline {
                Some(d) => {
                    let now = Instant::now();
                    if now >= d {
                        break;
                    }
                    d - now
                }
                None => Duration::from_millis(5),
            };
            self.shared.wait_for_space(wait);
        }
        Ok(done)
    }

    fn drain(&mut self, timeout: Option<Duration>) -> Result<bool, Error> {
        self.usable()?;
        self.shared.expect_empty();
        let target = self.shared.written();
        if self.shared.position() >= target {
            return Ok(true);
        }
        if self.paused {
            return Ok(false);
        }
        self.start()?;
        let deadline = timeout.map(|t| Instant::now() + t);
        loop {
            self.shared.check()?;
            if self.shared.position() >= target {
                return Ok(true);
            }
            if let Some(d) = deadline {
                if Instant::now() >= d {
                    return Ok(false);
                }
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn position(&self) -> Result<u64, Error> {
        self.shared.check()?;
        Ok(self.shared.position())
    }

    fn written(&self) -> u64 {
        self.shared.written()
    }

    fn latency(&self) -> Duration {
        let pending = self.shared.written().saturating_sub(self.shared.position());
        self.info.format.frames_to_duration(pending)
    }

    fn queued(&self) -> u64 {
        self.shared.ring.len()
    }

    fn pause(&mut self) -> Result<(), Error> {
        self.usable()?;
        if self.paused {
            return Ok(());
        }
        if self.running {
            self.driver.pause()?;
            self.running = false;
        }
        self.paused = true;
        Ok(())
    }

    fn resume(&mut self) -> Result<(), Error> {
        self.usable()?;
        if !self.paused {
            return Ok(());
        }
        self.paused = false;
        if self.started {
            // The device may still hold frames even when the ring is empty.
            self.start()
        } else {
            self.maybe_start()
        }
    }

    fn is_paused(&self) -> bool {
        self.paused
    }

    fn flush(&mut self) -> Result<(), Error> {
        self.usable()?;
        self.driver.flush()?;
        self.running = false;
        self.started = false;
        Ok(())
    }

    fn retract(&mut self, frames: u64) -> Result<u64, Error> {
        self.usable()?;
        // The device copies one buffer at a time; three stay queued.
        let keep = 3 * self.info.device_buffer_frames.max(64);
        let n = self.shared.ring.retract(frames, keep);
        self.shared.unwrite(n);
        Ok(n)
    }

    fn set_tap(&mut self, tap: Option<Box<dyn Tap>>) -> Result<(), Error> {
        self.usable()?;
        // A device that pulls takes it within a buffer or two.
        let wait = if self.running && self.driver.pulls() {
            let period = self
                .info
                .format
                .frames_to_duration(self.info.device_buffer_frames);
            (4 * period).clamp(Duration::from_millis(50), Duration::from_millis(500))
        } else {
            Duration::ZERO
        };
        self.shared.set_tap(tap, wait);
        Ok(())
    }

    fn stats(&self) -> Stats {
        self.shared.stats()
    }

    fn clock(&self) -> Clock {
        Clock {
            shared: self.shared.clone(),
            format: self.info.format,
        }
    }

    fn close(&mut self) -> Result<(), Error> {
        if self.closed {
            return Err(Error::Closed);
        }
        self.closed = true;
        self.running = false;
        self.driver.close()
    }
}

impl Drop for Core {
    fn drop(&mut self) {
        if !self.closed {
            let _ = self.close();
        }
    }
}

/// Implements [`Output`] for a wrapper by forwarding to its field `$field`.
macro_rules! delegate_output {
    ($ty:ty, $field:ident) => {
        impl $crate::output::Output for $ty {
            $crate::output::forward_output!($field);
            fn set_tap(&mut self, tap: Option<Box<dyn $crate::Tap>>) -> Result<(), $crate::Error> {
                $crate::output::Output::set_tap(&mut self.$field, tap)
            }
        }
    };
}
pub(crate) use delegate_output;

/// The methods of [`Output`] but [`Output::set_tap`], forwarding to the
/// field `$field`, for an `impl Output` that does its own taps.
macro_rules! forward_output {
    ($field:ident) => {
        fn info(&self) -> &$crate::output::StreamInfo {
            $crate::output::Output::info(&self.$field)
        }
        fn lanes(&self) -> u16 {
            $crate::output::Output::lanes(&self.$field)
        }
        fn write(
            &mut self,
            samples: &[i32],
            timeout: Option<std::time::Duration>,
        ) -> Result<usize, $crate::Error> {
            $crate::output::Output::write(&mut self.$field, samples, timeout)
        }
        fn drain(&mut self, timeout: Option<std::time::Duration>) -> Result<bool, $crate::Error> {
            $crate::output::Output::drain(&mut self.$field, timeout)
        }
        fn position(&self) -> Result<u64, $crate::Error> {
            $crate::output::Output::position(&self.$field)
        }
        fn written(&self) -> u64 {
            $crate::output::Output::written(&self.$field)
        }
        fn latency(&self) -> std::time::Duration {
            $crate::output::Output::latency(&self.$field)
        }
        fn queued(&self) -> u64 {
            $crate::output::Output::queued(&self.$field)
        }
        fn pause(&mut self) -> Result<(), $crate::Error> {
            $crate::output::Output::pause(&mut self.$field)
        }
        fn resume(&mut self) -> Result<(), $crate::Error> {
            $crate::output::Output::resume(&mut self.$field)
        }
        fn is_paused(&self) -> bool {
            $crate::output::Output::is_paused(&self.$field)
        }
        fn flush(&mut self) -> Result<(), $crate::Error> {
            $crate::output::Output::flush(&mut self.$field)
        }
        fn retract(&mut self, frames: u64) -> Result<u64, $crate::Error> {
            $crate::output::Output::retract(&mut self.$field, frames)
        }
        fn stats(&self) -> $crate::output::Stats {
            $crate::output::Output::stats(&self.$field)
        }
        fn clock(&self) -> $crate::output::Clock {
            $crate::output::Output::clock(&self.$field)
        }
        fn close(&mut self) -> Result<(), $crate::Error> {
            $crate::output::Output::close(&mut self.$field)
        }
    };
}
pub(crate) use forward_output;
