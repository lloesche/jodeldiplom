//! Linux: ALSA capture from `default`, as the output plays to it
//! ([`super::alsa_output`]): the sound server's default source, chosen in
//! the system's sound settings (ALSA's own `dsnoop` where no server runs).
//! No `plughw:` or `hw:` device is opened, so the microphone stays the
//! other applications' too (a call's, say). A stream runs at the server's
//! rate (48 kHz, or the nearest), in 32-bit float where the PCM takes it,
//! 2 channels or fewer, with a period of about 5 ms. A thread of its own
//! reads it period by period into the microphone ring.

use super::alsa_output::{map_error, open_pcm, DEFAULT, SERVER_RATE};
use super::thread::spawn_opened;
use crate::device::Device;
use crate::error::Error;
use crate::input::{Input, InputInfo, MicClock, Mono, Opened, Producer, Wire, CAPTURE_PERIOD};
use alsa::pcm::{Access, Format, Frames, HwParams, PCM};
use alsa::{Direction, ValueOr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

pub(crate) const INPUT_BACKEND: &str = "ALSA default (shared)";

const EAGAIN: i32 = 11;

pub(crate) fn input_devices() -> Result<Vec<Device>, Error> {
    Ok(vec![Device {
        id: DEFAULT.into(),
        name: "Default input (the system's)".into(),
        is_default: true,
    }])
}

struct Config {
    wire: Wire,
    rate: u32,
    channels: u16,
    period: usize,
}

fn configure(pcm: &PCM) -> Result<Config, Error> {
    let err = |call| move |e| map_error(call, e);
    let hwp = HwParams::any(pcm).map_err(err("snd_pcm_hw_params_any"))?;
    hwp.set_access(Access::RWInterleaved)
        .map_err(err("set_access"))?;
    let wire = [
        (Format::FloatLE, Wire::F32),
        (Format::S32LE, Wire::I32),
        (Format::S16LE, Wire::I16),
        (Format::S243LE, Wire::I24),
        (Format::S24LE, Wire::I24Low),
    ]
    .into_iter()
    .find(|&(f, _)| hwp.set_format(f).is_ok())
    .map(|(_, w)| w)
    .ok_or_else(|| Error::Unsupported("the input takes no PCM format we read".into()))?;
    let channels = hwp.set_channels_near(2).map_err(err("set_channels_near"))?;
    let rate = hwp
        .set_rate_near(SERVER_RATE, ValueOr::Nearest)
        .map_err(err("set_rate_near"))?;
    let period = (u64::from(rate) * CAPTURE_PERIOD.as_micros() as u64 / 1_000_000).max(16);
    hwp.set_period_size_near(period as Frames, ValueOr::Nearest)
        .map_err(err("set_period_size_near"))?;
    hwp.set_buffer_size_near(8 * period as Frames)
        .map_err(err("set_buffer_size_near"))?;
    pcm.hw_params(&hwp).map_err(err("snd_pcm_hw_params"))?;
    let current = pcm
        .hw_params_current()
        .map_err(err("snd_pcm_hw_params_current"))?;
    let rate = current.get_rate().map_err(err("get_rate"))?;
    let channels = current.get_channels().unwrap_or(channels);
    let period = current
        .get_period_size()
        .map_err(err("get_period_size"))?
        .max(1) as usize;
    Ok(Config {
        wire,
        rate,
        channels: channels.clamp(1, u32::from(u16::MAX)) as u16,
        period,
    })
}

fn run(pcm: PCM, config: Config, mut producer: Producer, stop: Arc<AtomicBool>) {
    let shared = producer.shared().clone();
    let channels = usize::from(config.channels);
    let fb = channels * config.wire.bytes();
    let mut buf = vec![0u8; config.period * fb];
    let result = (|| -> Result<(), Error> {
        pcm.prepare().map_err(|e| map_error("snd_pcm_prepare", e))?;
        pcm.start().map_err(|e| map_error("snd_pcm_start", e))?;
        while !stop.load(Ordering::Acquire) {
            let io = pcm.io_bytes();
            match io.readi(&mut buf) {
                Ok(0) => {}
                Ok(frames) => producer.push(&buf[..frames * fb], config.wire, channels, frames),
                Err(e) if e.errno() == EAGAIN => {
                    drop(io);
                    let _ = pcm.wait(Some(100));
                }
                Err(e) => {
                    drop(io);
                    // An overrun (nobody read for a while): start again.
                    pcm.try_recover(e, true)
                        .map_err(|e| map_error("snd_pcm_recover", e))?;
                    let _ = pcm.start();
                }
            }
        }
        Ok(())
    })();
    if let Err(e) = result {
        shared.fail(e);
    }
    let _ = pcm.drop();
}

/// The running capture: stopping it is dropping it.
struct Running {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for Running {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

pub(crate) fn open_input(id: &str, mono: Mono) -> Result<Input, Error> {
    let stop = Arc::new(AtomicBool::new(false));
    let stopped = stop.clone();
    let id = id.to_owned();
    let ((mut info, opened), thread) = spawn_opened(
        "jodel-input-alsa",
        "opening the input",
        move || {
            let pcm = open_pcm(&id, Direction::Capture)?;
            let config = configure(&pcm)?;
            let period = config.period as u32;
            let (producer, opened) = Opened::new(config.rate, period, mono, MicClock::system());
            let info = InputInfo {
                name: id.clone(),
                id,
                backend: INPUT_BACKEND,
                rate: config.rate,
                channels: config.channels,
                period_frames: period,
                mono,
            };
            Ok(((info, opened), (pcm, config, producer)))
        },
        move |(pcm, config, producer)| run(pcm, config, producer, stopped),
    )?;
    let running = Running {
        stop,
        thread: Some(thread),
    };
    if let Some(d) = input_devices()
        .ok()
        .and_then(|all| all.into_iter().find(|d| d.id == info.id))
    {
        info.name = d.name;
    }
    Ok(Input::new(info, opened, Box::new(running)))
}
