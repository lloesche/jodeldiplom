//! Outputs without a device, for tests: [`FileOutput`] writes the packed
//! device bytes to a sink, [`NullOutput`] discards them. Both have the same
//! API as a device output, with a simulated clock.

use crate::caps::Capabilities;
use crate::error::Error;
use crate::format::StreamFormat;
use crate::lanes::lane_format;
use crate::output::{delegate_output, Core, Driver, OpenRequest, StreamInfo};
use crate::shared::Shared;
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// How a simulated device consumes frames.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SimClock {
    /// Every frame is played the moment it is written: writes never block,
    /// `position() == written()` after each write, and nothing is ever
    /// padded. For capturing exactly what an engine writes.
    #[default]
    Instant,
    /// Frames are played only when the test calls
    /// [`FileOutput::advance`]: a full ring blocks the writer, an empty one
    /// pads with silence and counts an underrun, exactly as a device would.
    Manual,
    /// A thread plays one period (the request's `period`) every period of
    /// wall-clock time, like a device.
    Realtime,
    /// A thread plays what is written, no faster than wall-clock time (a
    /// period every period at most, from the first frame), but only what
    /// is written: an empty ring waits for the writer, padded with nothing
    /// and no underrun, and the time it waited is not made up in a burst.
    /// It does not play in real time ([`StreamInfo::realtime`] is false: a
    /// mixer on it waits for every input), yet its writers run at about a
    /// device's pace: for tests whose frames must line up exactly while
    /// players run as they do on a device.
    Paced,
}

/// The sink of [`FileOutput::to_memory`]: the bytes the simulated device
/// played, readable while the output is open.
#[derive(Clone, Default)]
pub struct MemorySink(Arc<Mutex<Vec<u8>>>);

impl MemorySink {
    /// A copy of the bytes so far.
    pub fn bytes(&self) -> Vec<u8> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// How many bytes so far.
    pub fn len(&self) -> usize {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Whether nothing has been played yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Write for MemorySink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

type Sink = Arc<Mutex<Box<dyn Write + Send>>>;

/// The simulated device: pulls from the ring through the same path as the
/// real backends and writes what it plays to the sink.
struct SimDevice {
    shared: Arc<Shared>,
    sink: Sink,
    running: AtomicBool,
    stop: AtomicBool,
}

impl SimDevice {
    /// Plays up to `frames` device frames (at most what is queued when
    /// `only_queued`). Returns the stream frames among them.
    fn play(&self, frames: u64, only_queued: bool) -> u64 {
        let fb = self.shared.format.frame_bytes();
        let mut left = if only_queued {
            frames.min(self.shared.ring.len())
        } else {
            frames
        };
        let mut chunk = vec![0u8; 4096 * fb];
        let mut stream = 0;
        let mut sink = self.sink.lock().unwrap_or_else(|e| e.into_inner());
        while left > 0 {
            let n = left.min(4096) as usize;
            let buf = &mut chunk[..n * fb];
            let Some(got) = self.shared.render_or_skip(buf) else {
                break;
            };
            stream += got;
            left -= n as u64;
            if let Err(e) = sink.write_all(buf) {
                self.shared.fail(e.into());
                break;
            }
            // The simulated DAC has no latency: what was submitted is played.
            self.shared.report_delay(0, Instant::now(), false);
        }
        stream
    }
}

struct SimDriver {
    clock: SimClock,
    device: Arc<SimDevice>,
    period: u64,
    thread: Option<JoinHandle<()>>,
}

impl SimDriver {
    fn spawn_realtime(&mut self) {
        let device = self.device.clone();
        let period_frames = self.period;
        let period = device.shared.format.frames_to_duration(period_frames);
        self.thread = Some(std::thread::spawn(move || {
            let mut next = Instant::now();
            while !device.stop.load(Ordering::SeqCst) {
                next += period;
                if device.running.load(Ordering::SeqCst) {
                    device.play(period_frames, false);
                } else {
                    next = Instant::now() + period;
                }
                let now = Instant::now();
                if next > now {
                    std::thread::sleep(next - now);
                }
            }
        }));
    }

    /// [`SimClock::Paced`]: up to a period of what is queued each period;
    /// with nothing queued it looks again in a millisecond, and plays on
    /// from then (no time owed).
    fn spawn_paced(&mut self) {
        let device = self.device.clone();
        let period_frames = self.period;
        let format = device.shared.format;
        let period = format.frames_to_duration(period_frames);
        self.thread = Some(std::thread::spawn(move || {
            let mut next = Instant::now();
            while !device.stop.load(Ordering::SeqCst) {
                let played = if device.running.load(Ordering::SeqCst) {
                    device.play(period_frames, true)
                } else {
                    0
                };
                let now = Instant::now();
                if played == 0 {
                    next = now;
                    std::thread::sleep(Duration::from_millis(1));
                    continue;
                }
                // Put off, it makes up a period at most.
                next += format.frames_to_duration(played);
                if let Some(behind) = now.checked_sub(period) {
                    next = next.max(behind);
                }
                if next > now {
                    std::thread::sleep(next - now);
                }
            }
        }));
    }
}

impl Driver for SimDriver {
    fn start(&mut self) -> Result<(), Error> {
        self.device.running.store(true, Ordering::SeqCst);
        if self.clock == SimClock::Instant {
            self.device.play(u64::MAX, true);
        }
        Ok(())
    }

    fn pause(&mut self) -> Result<(), Error> {
        self.device.running.store(false, Ordering::SeqCst);
        Ok(())
    }

    fn flush(&mut self) -> Result<(), Error> {
        self.device.running.store(false, Ordering::SeqCst);
        self.device.shared.discard_all();
        Ok(())
    }

    fn close(&mut self) -> Result<(), Error> {
        self.device.running.store(false, Ordering::SeqCst);
        self.device.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        let mut sink = self.device.sink.lock().unwrap_or_else(|e| e.into_inner());
        sink.flush().map_err(Error::from)?;
        // The stream is over: the sink is let go of (a file takes its name
        // then, [`FileOutput::to_file`]).
        drop(std::mem::replace(&mut *sink, Box::new(io::sink())));
        Ok(())
    }

    fn pulls(&self) -> bool {
        matches!(self.clock, SimClock::Realtime | SimClock::Paced)
    }

    fn after_write(&mut self) -> Result<(), Error> {
        if self.clock == SimClock::Instant && self.device.running.load(Ordering::SeqCst) {
            self.device.play(u64::MAX, true);
        }
        Ok(())
    }
}

/// An output that writes the packed device bytes (exactly what a DAC would
/// receive, including any silence padding) to a file, memory or any
/// writer, with a simulated clock.
///
/// The bytes are raw interleaved PCM in the stream format's layout
/// ([`Output::format`](crate::Output::format)), with no header; a lane
/// stream's are its ring's ([`FileOutput::open_lanes`]).
pub struct FileOutput {
    core: Core,
    device: Arc<SimDevice>,
    clock: SimClock,
}

impl FileOutput {
    /// An output in `format` writing to `sink`, with the default request
    /// settings for buffering.
    pub fn new(
        sink: impl Write + Send + 'static,
        format: StreamFormat,
        clock: SimClock,
    ) -> FileOutput {
        let request = OpenRequest::new(format.rate, format.channels).formats(&[format.sample]);
        Self::build(Box::new(sink), format, &request, clock)
    }

    /// Opens like a device that accepts exactly `accepts`: the first format
    /// in `request.formats` accepted at the request's rate and channels.
    /// A plain stream, as a device takes: more than one lane is
    /// [`Error::InvalidArgument`] (see [`FileOutput::open_lanes`]).
    pub fn open(
        sink: impl Write + Send + 'static,
        accepts: &Capabilities,
        request: &OpenRequest,
        clock: SimClock,
    ) -> Result<FileOutput, Error> {
        request.check_lanes(1)?;
        Self::open_lanes(sink, accepts, request, clock)
    }

    /// Opens as [`FileOutput::open`] does, taking any number of lanes
    /// ([`OpenRequest::lanes`]), for a test that captures what a writer of
    /// lanes writes. Its device format ([`crate::Output::format`]) is the
    /// one chosen, what main is written for; but what it plays to the sink
    /// is the lane stream as written: the ring's bytes,
    /// [`crate::lanes::LANE_SAMPLE`] at [`crate::lanes::lane_width`]
    /// channels a frame. With one lane it is [`FileOutput::open`].
    pub fn open_lanes(
        sink: impl Write + Send + 'static,
        accepts: &Capabilities,
        request: &OpenRequest,
        clock: SimClock,
    ) -> Result<FileOutput, Error> {
        request.validate()?;
        let format = accepts
            .choose(request.rate, request.channels, &request.formats)
            .ok_or(Error::FormatNotAccepted {
                rate: request.rate,
                channels: request.channels,
            })?;
        Ok(Self::build(Box::new(sink), format, request, clock))
    }

    /// An output writing to a new file at `path`, under a temporary name
    /// that becomes `path` when the output is done with it (dropped): a
    /// file at `path` is always a whole stream ([`durable::AtomicFile`]).
    pub fn to_file(
        path: impl AsRef<Path>,
        format: StreamFormat,
        clock: SimClock,
    ) -> Result<FileOutput, Error> {
        let file = CommittedOnDrop(Some(durable::AtomicFile::create(path.as_ref())?));
        Ok(Self::new(file, format, clock))
    }

    /// An output writing to memory, and the handle to read it.
    pub fn to_memory(format: StreamFormat, clock: SimClock) -> (FileOutput, MemorySink) {
        let sink = MemorySink::default();
        (Self::new(sink.clone(), format, clock), sink)
    }

    fn build(
        sink: Box<dyn Write + Send>,
        format: StreamFormat,
        request: &OpenRequest,
        clock: SimClock,
    ) -> FileOutput {
        let period = format.duration_to_frames(request.period).max(1);
        let ring_frames = request.ring_frames(&format, period);
        let ring = lane_format(format, request.lanes);
        let shared = Arc::new(Shared::new(ring, ring_frames));
        let device = Arc::new(SimDevice {
            shared: shared.clone(),
            sink: Arc::new(Mutex::new(sink)),
            running: AtomicBool::new(false),
            stop: AtomicBool::new(false),
        });
        let mut driver = SimDriver {
            clock,
            device: device.clone(),
            period,
            thread: None,
        };
        match clock {
            SimClock::Realtime => driver.spawn_realtime(),
            SimClock::Paced => driver.spawn_paced(),
            SimClock::Instant | SimClock::Manual => {}
        }
        let info = StreamInfo {
            format,
            backend: "file",
            exclusive: false,
            device_buffer_frames: period,
            device_latency: Duration::ZERO,
            ring_frames: shared.ring.capacity(),
            realtime: clock == SimClock::Realtime,
            float_io: false,
        };
        let mut core = Core::new(shared, Box::new(driver), info, request);
        if matches!(clock, SimClock::Instant | SimClock::Paced) {
            core.set_prefill(1);
        }
        FileOutput {
            core,
            device,
            clock,
        }
    }

    /// [`SimClock::Manual`] only: lets `frames` device frames of time pass.
    /// A running device plays them (padding with silence when the ring runs
    /// dry); a stopped or paused one plays nothing. Returns the stream
    /// frames played.
    pub fn advance(&mut self, frames: u64) -> Result<u64, Error> {
        if self.clock != SimClock::Manual {
            return Err(Error::InvalidArgument(
                "advance needs SimClock::Manual".into(),
            ));
        }
        self.core.shared.check()?;
        if !self.device.running.load(Ordering::SeqCst) {
            return Ok(0);
        }
        let played = self.device.play(frames, false);
        self.core.shared.check()?;
        Ok(played)
    }
}

delegate_output!(FileOutput, core);

/// An output that discards what it plays: a [`FileOutput`] over
/// [`std::io::sink`].
pub struct NullOutput {
    inner: FileOutput,
}

impl NullOutput {
    /// A null output in `format`.
    pub fn new(format: StreamFormat, clock: SimClock) -> NullOutput {
        NullOutput {
            inner: FileOutput::new(io::sink(), format, clock),
        }
    }

    /// See [`FileOutput::advance`].
    pub fn advance(&mut self, frames: u64) -> Result<u64, Error> {
        self.inner.advance(frames)
    }
}

delegate_output!(NullOutput, inner);

/// A file the output writes into, taking its name when the output lets go
/// of it (a crash leaves only its temporary).
struct CommittedOnDrop(Option<durable::AtomicFile>);

impl Write for CommittedOnDrop {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .as_mut()
            .expect("written before it is dropped")
            .write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0
            .as_mut()
            .expect("written before it is dropped")
            .flush()
    }
}

impl Drop for CommittedOnDrop {
    fn drop(&mut self) {
        if let Some(file) = self.0.take() {
            let _ = file.commit();
        }
    }
}
