//! Linux: ALSA capture.
//!
//! A microphone needs no bit-perfect path, so inputs go through ALSA's
//! plugins: `default` (the sound server, where there is one) and every
//! capture device as `plughw:CARD=<id>,DEV=<n>`. A stream opens with
//! ALSA's resampling off where the device allows it, so it runs at a rate
//! the hardware has (48 kHz, or the nearest), in 32-bit float where the
//! plugin converts, 2 channels or fewer, with a period of about 5 ms. A
//! thread of its own reads it period by period into the microphone ring.

use crate::device::Device;
use crate::error::Error;
use crate::input::{Input, InputInfo, MicShared, Mono, Producer, Wire, CAPTURE_PERIOD};
use alsa::pcm::{Access, Format, Frames, HwParams, PCM};
use alsa::{Direction, ValueOr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::JoinHandle;
use std::time::Duration;

pub(crate) const INPUT_BACKEND: &str = "ALSA";

const ENOENT: i32 = 2;
const EAGAIN: i32 = 11;
const EBUSY: i32 = 16;
const ENODEV: i32 = 19;

fn map_error(call: &'static str, e: alsa::Error) -> Error {
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

pub(crate) fn input_devices() -> Result<Vec<Device>, Error> {
    let mut out = vec![Device {
        id: "default".into(),
        name: "Default input".into(),
        is_default: true,
    }];
    for card in alsa::card::Iter::new() {
        let card = card.map_err(|e| map_error("snd_card_next", e))?;
        let Ok(ctl) = alsa::Ctl::from_card(&card, false) else {
            continue;
        };
        let Ok(info) = ctl.card_info() else {
            continue;
        };
        let card_id = info.get_id().unwrap_or_default().to_owned();
        let card_name = info.get_name().unwrap_or_default().to_owned();
        for dev in alsa::ctl::DeviceIter::new(&ctl) {
            let Ok(pcm) = ctl.pcm_info(dev as u32, 0, Direction::Capture) else {
                continue;
            };
            let pcm_name = pcm.get_name().unwrap_or_default();
            out.push(Device {
                id: format!("plughw:CARD={card_id},DEV={dev}"),
                name: if pcm_name.is_empty() {
                    card_name.clone()
                } else {
                    format!("{card_name}: {pcm_name}")
                },
                is_default: false,
            });
        }
    }
    Ok(out)
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
    // The hardware's own rate where the plugin lets us ask for it.
    let _ = hwp.set_rate_resample(false);
    let rate = hwp
        .set_rate_near(48_000, ValueOr::Nearest)
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
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let id_owned = id.to_owned();
    let stop_thread = stop.clone();
    let thread = std::thread::Builder::new()
        .name("jodel-input-alsa".into())
        .spawn(move || {
            // Non-blocking, so the thread sees a stop within its wait.
            let opened = PCM::new(&id_owned, Direction::Capture, true)
                .map_err(|e| match e.errno() {
                    ENOENT | ENODEV => Error::DeviceNotFound(id_owned.clone()),
                    _ => map_error("snd_pcm_open", e),
                })
                .and_then(|pcm| configure(&pcm).map(|c| (pcm, c)));
            match opened {
                Ok((pcm, config)) => {
                    let shared = MicShared::new(config.rate, config.period as u32, false);
                    let info = InputInfo {
                        id: id_owned.clone(),
                        name: id_owned.clone(),
                        backend: INPUT_BACKEND,
                        rate: config.rate,
                        channels: config.channels,
                        period_frames: config.period as u32,
                        mono,
                    };
                    let _ = ready_tx.send(Ok((info, shared.clone())));
                    run(pcm, config, Producer::new(shared, mono), stop_thread);
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                }
            }
        })?;
    let running = Running {
        stop,
        thread: Some(thread),
    };
    let opened = ready_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap_or(Err(Error::Timeout("opening the input")));
    let (mut info, shared) = opened?;
    if let Some(d) = input_devices()
        .ok()
        .and_then(|all| all.into_iter().find(|d| d.id == info.id))
    {
        info.name = d.name;
    }
    Ok(Input::new(info, shared, Box::new(running)))
}
