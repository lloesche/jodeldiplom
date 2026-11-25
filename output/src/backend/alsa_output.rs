//! Linux: ALSA's `default` PCM, shared.
//!
//! The one device is `default`, which goes to the sound server where one
//! runs (PipeWire or PulseAudio, through their ALSA plugins) and to ALSA's
//! own mixer (`dmix`) where none does: the sink is the system's default,
//! chosen in its sound settings. No `hw:` device is opened, so every other
//! application plays beside us.
//!
//! Its one format ([`mix_format`]) is stereo 32-bit float at the server's
//! rate: ALSA cannot ask the server for its rate, so it is [`SERVER_RATE`]
//! (what PipeWire, PulseAudio and `dmix` run at unless set otherwise), or
//! the nearest the PCM has; a server at another rate converts.
//!
//! The stream runs on its own thread with the PCM in non-blocking mode: it
//! waits for a period of room, renders one period from the ring, writes it
//! and reports `snd_pcm_delay` as the position (the servers' plugins count
//! their own latency in it). Pause is `snd_pcm_pause` where the PCM
//! supports it; otherwise the device queue is dropped (those frames count
//! as played) and refilled on resume.

use super::thread::{Command, Request, ThreadDriver};
use crate::device::Device;
use crate::error::Error;
use crate::format::{SampleFormat, StreamFormat};
use crate::output::{Core, OpenRequest, StreamInfo};
use crate::shared::Shared;
use alsa::pcm::{Access, Format, Frames, HwParams, State, PCM};
use alsa::{Direction, ValueOr};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub(crate) const BACKEND: &str = "ALSA default (shared)";

/// The rate asked of a shared PCM.
pub(super) const SERVER_RATE: u32 = 48_000;

/// The one PCM: the system's default.
pub(super) const DEFAULT: &str = "default";

const ENOENT: i32 = 2;
const EAGAIN: i32 = 11;
const EBUSY: i32 = 16;
const ENODEV: i32 = 19;
const EPIPE: i32 = 32;
const ESTRPIPE: i32 = 86;

pub(super) fn map_error(call: &'static str, e: alsa::Error) -> Error {
    match e.errno() {
        EBUSY => Error::DeviceBusy,
        ENODEV => Error::DeviceLost,
        errno => Error::Backend {
            call,
            code: -i64::from(errno),
            message: e.to_string(),
        },
    }
}

/// Opens the PCM `id` for `direction`, non-blocking (so a device held by
/// another application fails at once, and a thread sees a stop within its
/// wait).
pub(super) fn open_pcm(id: &str, direction: Direction) -> Result<PCM, Error> {
    PCM::new(id, direction, true).map_err(|e| match e.errno() {
        ENOENT | ENODEV => Error::DeviceNotFound(id.into()),
        _ => map_error("snd_pcm_open", e),
    })
}

pub(crate) fn devices() -> Result<Vec<Device>, Error> {
    Ok(vec![Device {
        id: DEFAULT.into(),
        name: "Default output (the system's)".into(),
        is_default: true,
    }])
}

/// The PCM's one format: stereo float at [`SERVER_RATE`], or the nearest
/// rate it has.
fn format_of(pcm: &PCM) -> Result<StreamFormat, Error> {
    let err = |call| move |e| map_error(call, e);
    let hwp = HwParams::any(pcm).map_err(err("snd_pcm_hw_params_any"))?;
    hwp.set_access(Access::RWInterleaved)
        .map_err(err("set_access"))?;
    hwp.set_channels(2).map_err(err("set_channels"))?;
    hwp.set_format(Format::FloatLE)
        .map_err(|_| Error::Unsupported("the PCM takes no 32-bit float".into()))?;
    let rate = hwp
        .set_rate_near(SERVER_RATE, ValueOr::Nearest)
        .map_err(err("set_rate_near"))?;
    Ok(StreamFormat::new(rate, 2, SampleFormat::F32))
}

pub(crate) fn mix_format(id: &str) -> Result<StreamFormat, Error> {
    format_of(&open_pcm(id, Direction::Playback)?)
}

struct Config {
    format: StreamFormat,
    period: u64,
    buffer: u64,
    can_pause: bool,
}

fn configure(pcm: &PCM, request: &OpenRequest) -> Result<Config, Error> {
    let format = format_of(pcm)?;
    if !request.takes(format) {
        return Err(Error::FormatNotAccepted {
            rate: request.rate,
            channels: request.channels,
        });
    }
    let err = |call| move |e| map_error(call, e);
    let period = format.duration_to_frames(request.period).max(16);
    {
        let hwp = HwParams::any(pcm).map_err(err("snd_pcm_hw_params_any"))?;
        hwp.set_access(Access::RWInterleaved)
            .map_err(err("set_access"))?;
        hwp.set_channels(u32::from(format.channels))
            .map_err(err("set_channels"))?;
        hwp.set_format(Format::FloatLE).map_err(err("set_format"))?;
        hwp.set_rate(format.rate, ValueOr::Nearest)
            .map_err(err("set_rate"))?;
        hwp.set_period_size_near(period as Frames, ValueOr::Nearest)
            .map_err(err("set_period_size_near"))?;
        hwp.set_buffer_size_near(4 * period as Frames)
            .map_err(err("set_buffer_size_near"))?;
        pcm.hw_params(&hwp).map_err(err("snd_pcm_hw_params"))?;
    }
    let current = pcm
        .hw_params_current()
        .map_err(err("snd_pcm_hw_params_current"))?;
    let rate = current.get_rate().map_err(err("get_rate"))?;
    if rate != format.rate {
        return Err(Error::FormatNotAccepted {
            rate: format.rate,
            channels: format.channels,
        });
    }
    let period = current
        .get_period_size()
        .map_err(err("get_period_size"))?
        .max(1) as u64;
    let buffer = current
        .get_buffer_size()
        .map_err(err("get_buffer_size"))?
        .max(1) as u64;
    let can_pause = current.can_pause();
    drop(current);
    let swp = pcm
        .sw_params_current()
        .map_err(err("snd_pcm_sw_params_current"))?;
    swp.set_start_threshold(buffer as Frames)
        .map_err(err("set_start_threshold"))?;
    swp.set_avail_min(period as Frames)
        .map_err(err("set_avail_min"))?;
    pcm.sw_params(&swp).map_err(err("snd_pcm_sw_params"))?;
    Ok(Config {
        format,
        period,
        buffer,
        can_pause,
    })
}

struct Stream {
    pcm: PCM,
    config: Config,
    scratch: Vec<u8>,
}

impl Stream {
    /// Writes all of `scratch` (one period), waiting for room as needed.
    fn write_period(&mut self, shared: &Shared) -> Result<(), Error> {
        shared.render(&mut self.scratch);
        let fb = self.config.format.frame_bytes();
        let mut done = 0;
        while done < self.scratch.len() {
            let io = self.pcm.io_bytes();
            match io.writei(&self.scratch[done..]) {
                Ok(frames) => done += frames * fb,
                Err(e) if e.errno() == EAGAIN => {
                    drop(io);
                    let _ = self.pcm.wait(Some(self.wait_ms()));
                }
                Err(e) => {
                    drop(io);
                    self.recover(shared, e)?;
                }
            }
        }
        Ok(())
    }

    fn recover(&self, shared: &Shared, e: alsa::Error) -> Result<(), Error> {
        if e.errno() == EPIPE || e.errno() == ESTRPIPE {
            shared.note_device_underrun();
        }
        self.pcm
            .try_recover(e, true)
            .map_err(|e| map_error("snd_pcm_recover", e))
    }

    fn wait_ms(&self) -> u32 {
        let ms = self.config.period * 2000 / u64::from(self.config.format.rate);
        ms.clamp(1, 1000) as u32
    }

    fn report(&self, shared: &Shared) {
        if let Ok(delay) = self.pcm.delay() {
            let running = self.pcm.state() == State::Running;
            shared.report_delay(delay.max(0) as u64, Instant::now(), running);
        }
    }

    /// Fills the device buffer and starts it.
    fn start(&mut self, shared: &Shared) -> Result<(), Error> {
        if self.pcm.state() == State::Paused {
            return self
                .pcm
                .pause(false)
                .map_err(|e| map_error("snd_pcm_pause", e));
        }
        let mut filled = 0;
        while filled + self.config.period <= self.config.buffer {
            self.write_period(shared)?;
            filled += self.config.period;
        }
        if self.pcm.state() == State::Prepared {
            self.pcm
                .start()
                .map_err(|e| map_error("snd_pcm_start", e))?;
        }
        self.report(shared);
        Ok(())
    }

    fn feed(&mut self, shared: &Shared) -> Result<(), Error> {
        match self.pcm.avail_update() {
            Ok(avail) if avail as u64 >= self.config.period => {
                self.write_period(shared)?;
            }
            Ok(_) => {
                let _ = self.pcm.wait(Some(self.wait_ms()));
            }
            Err(e) => self.recover(shared, e)?,
        }
        self.report(shared);
        Ok(())
    }

    fn pause(&mut self, shared: &Shared) -> Result<(), Error> {
        if self.config.can_pause {
            self.pcm
                .pause(true)
                .map_err(|e| map_error("snd_pcm_pause", e))?;
            self.report(shared);
            shared.freeze();
        } else {
            let _ = self.pcm.drop();
            shared.skip_device_queue();
            self.pcm
                .prepare()
                .map_err(|e| map_error("snd_pcm_prepare", e))?;
        }
        Ok(())
    }

    fn flush(&mut self, shared: &Shared) -> Result<(), Error> {
        let _ = self.pcm.drop();
        shared.discard_all();
        self.pcm
            .prepare()
            .map_err(|e| map_error("snd_pcm_prepare", e))
    }
}

fn run(mut stream: Stream, shared: Arc<Shared>, commands: Receiver<Request>) {
    let mut running = false;
    loop {
        let request = if running {
            match commands.try_recv() {
                Ok(r) => Some(r),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => break,
            }
        } else {
            match commands.recv() {
                Ok(r) => Some(r),
                Err(_) => break,
            }
        };
        let Some(request) = request else {
            if let Err(e) = stream.feed(&shared) {
                shared.fail(e);
                let _ = stream.pcm.drop();
                running = false;
            }
            continue;
        };
        let command = request.command;
        let result = match command {
            Command::Start if running => Ok(()),
            Command::Start => shared
                .check()
                .and_then(|_| stream.start(&shared))
                .map(|_| running = true),
            Command::Pause if !running => Ok(()),
            Command::Pause => {
                running = false;
                stream.pause(&shared)
            }
            Command::Flush => {
                running = false;
                stream.flush(&shared)
            }
            Command::Close => {
                running = false;
                let _ = stream.pcm.drop();
                shared.freeze();
                Ok(())
            }
        };
        request.reply(result);
        if command == Command::Close {
            break;
        }
    }
}

/// Opens the PCM `id`; the stream runs on its own thread.
pub(crate) fn open(id: &str, request: &OpenRequest) -> Result<Core, Error> {
    request.validate()?;
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let id_owned = id.to_owned();
    let request_owned = request.clone();
    let mut driver = ThreadDriver::spawn("jodel-output-alsa", Box::new(|| {}), move |commands| {
        let opened = open_pcm(&id_owned, Direction::Playback)
            .and_then(|pcm| configure(&pcm, &request_owned).map(|c| (pcm, c)));
        match opened {
            Ok((pcm, config)) => {
                let shared = Arc::new(Shared::new(
                    config.format,
                    request_owned.ring_frames(&config.format, config.buffer),
                ));
                let info = StreamInfo {
                    format: config.format,
                    backend: BACKEND,
                    device_buffer_frames: config.period,
                    ring_frames: shared.ring.capacity(),
                    realtime: true,
                };
                let scratch = vec![0; config.period as usize * config.format.frame_bytes()];
                let _ = ready_tx.send(Ok((info, shared.clone())));
                run(
                    Stream {
                        pcm,
                        config,
                        scratch,
                    },
                    shared,
                    commands,
                );
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
                while let Ok(r) = commands.recv() {
                    let done = r.command == Command::Close;
                    r.reply(Ok(()));
                    if done {
                        break;
                    }
                }
            }
        }
    })?;
    let opened = ready_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap_or(Err(Error::Timeout("opening the device")));
    match opened {
        Ok((info, shared)) => Ok(Core::new(shared, Box::new(driver), info, request)),
        Err(e) => {
            use crate::output::Driver;
            let _ = driver.close();
            Err(e)
        }
    }
}
