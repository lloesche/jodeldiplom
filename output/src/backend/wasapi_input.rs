//! Windows: WASAPI shared-mode capture, event-driven.
//!
//! Inputs are the active capture endpoints. A stream opens at the
//! endpoint's mix format (its own rate and channels, usually 32-bit float)
//! on a thread of its own, COM in the multithreaded apartment and MMCSS
//! "Pro Audio". The period is about 5 ms where the audio engine allows a
//! shorter one than its default (`IAudioClient3`), else the engine's
//! default (10 ms). Every event drains the packets queued into the
//! microphone ring.

use super::thread::spawn_opened;
use super::wasapi::{
    activate, endpoints, enumerator, find, friendly_name, map_error, Com, Event, Mmcss,
};
use crate::device::Device;
use crate::error::Error;
use crate::input::{Input, InputInfo, MicClock, Mono, Opened, Producer, Wire, CAPTURE_PERIOD};
use std::ptr;
use std::sync::Arc;
use std::thread::JoinHandle;
use windows::core::Interface;
use windows::Win32::Foundation::{WAIT_OBJECT_0, WAIT_TIMEOUT};
use windows::Win32::Media::Audio::{
    eCapture, IAudioCaptureClient, IAudioClient, IAudioClient3, AUDCLNT_BUFFERFLAGS_SILENT,
    AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_EVENTCALLBACK, WAVEFORMATEX,
    WAVEFORMATEXTENSIBLE, WAVE_FORMAT_PCM,
};
use windows::Win32::Media::KernelStreaming::{KSDATAFORMAT_SUBTYPE_PCM, WAVE_FORMAT_EXTENSIBLE};
use windows::Win32::Media::Multimedia::{KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, WAVE_FORMAT_IEEE_FLOAT};
use windows::Win32::System::Com::CoTaskMemFree;
use windows::Win32::System::Threading::WaitForMultipleObjects;

pub(crate) const INPUT_BACKEND: &str = "WASAPI (shared)";

pub(crate) fn input_devices() -> Result<Vec<Device>, Error> {
    endpoints(eCapture)
}

/// The sample layout of a mix format, if we read it.
fn wire_of(p: *const WAVEFORMATEX) -> Option<Wire> {
    // SAFETY: `p` is a mix format from GetMixFormat, read unaligned (the
    // structs are packed); the extensible part only when it says it has one.
    let (w, float, pcm) = unsafe {
        let w: WAVEFORMATEX = ptr::read_unaligned(p);
        let tag = u32::from(w.wFormatTag);
        if tag == WAVE_FORMAT_EXTENSIBLE && w.cbSize >= 22 {
            let x: WAVEFORMATEXTENSIBLE = ptr::read_unaligned(p as *const WAVEFORMATEXTENSIBLE);
            let sub = x.SubFormat;
            (
                w,
                sub == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT,
                sub == KSDATAFORMAT_SUBTYPE_PCM,
            )
        } else {
            (w, tag == WAVE_FORMAT_IEEE_FLOAT, tag == WAVE_FORMAT_PCM)
        }
    };
    let bits = w.wBitsPerSample;
    let wire = match (float, pcm, bits) {
        (true, _, 32) => Wire::F32,
        (_, true, 16) => Wire::I16,
        (_, true, 24) => Wire::I24,
        // 24 valid bits sit high in a 32-bit WASAPI container.
        (_, true, 32) => Wire::I32,
        _ => return None,
    };
    let block = usize::from(w.nBlockAlign);
    (w.nChannels > 0 && block == usize::from(w.nChannels) * wire.bytes()).then_some(wire)
}

/// A capture stream, before its thread runs it.
struct Capture {
    client: IAudioClient,
    capture: IAudioCaptureClient,
    event: Event,
    wire: Wire,
    rate: u32,
    channels: u16,
    period: u32,
}

/// Initialises `client` with a period near [`CAPTURE_PERIOD`] through
/// `IAudioClient3`; returns the period in frames.
fn low_latency(
    client: &IAudioClient,
    mix: *const WAVEFORMATEX,
    rate: u32,
) -> windows::core::Result<u32> {
    let c3: IAudioClient3 = client.cast()?;
    let (mut default, mut fundamental, mut min, mut max) = (0u32, 0u32, 0u32, 0u32);
    // SAFETY: out parameters on the stack; `mix` outlives the calls.
    unsafe {
        c3.GetSharedModeEnginePeriod(mix, &mut default, &mut fundamental, &mut min, &mut max)?;
        let want = (u64::from(rate) * CAPTURE_PERIOD.as_micros() as u64 / 1_000_000) as u32;
        let fundamental = fundamental.max(1);
        let period = (want / fundamental * fundamental).clamp(min.max(fundamental), max.max(min));
        c3.InitializeSharedAudioStream(AUDCLNT_STREAMFLAGS_EVENTCALLBACK, period, mix, None)?;
        Ok(period)
    }
}

fn open_capture(id: &str) -> Result<(Capture, String), Error> {
    let dev = find(&enumerator()?, id)?;
    let name = friendly_name(&dev).unwrap_or_else(|| id.to_owned());
    let client = activate(&dev)?;
    // SAFETY: the mix format is a CoTaskMem block, freed below.
    let mix = unsafe { client.GetMixFormat() }.map_err(|e| map_error("GetMixFormat", e))?;
    let result = (|| {
        let wire = wire_of(mix).ok_or_else(|| {
            Error::Unsupported("the input's mix format is not PCM or 32-bit float".into())
        })?;
        // SAFETY: read as in `wire_of`.
        let w: WAVEFORMATEX = unsafe { ptr::read_unaligned(mix) };
        let (rate, channels) = (w.nSamplesPerSec, w.nChannels);
        let (client, period) = match low_latency(&client, mix, rate) {
            Ok(period) => (client, period),
            Err(_) => {
                // The engine's default period, on a fresh client.
                let client = activate(&dev)?;
                let mut default = 0i64;
                // SAFETY: COM calls on the client owned here; `mix` outlives them.
                unsafe {
                    client
                        .Initialize(
                            AUDCLNT_SHAREMODE_SHARED,
                            AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                            0,
                            0,
                            mix,
                            None,
                        )
                        .map_err(|e| map_error("IAudioClient::Initialize", e))?;
                    let _ = client.GetDevicePeriod(Some(&mut default), None);
                }
                let period = (default.max(0) as u64 * u64::from(rate) / 10_000_000) as u32;
                (client, period.max(1))
            }
        };
        let event = Event::new()?;
        // SAFETY: COM calls on the initialised client.
        let capture = unsafe {
            client
                .SetEventHandle(event.handle())
                .map_err(|e| map_error("SetEventHandle", e))?;
            client
                .GetService::<IAudioCaptureClient>()
                .map_err(|e| map_error("GetService(capture)", e))?
        };
        Ok(Capture {
            client,
            capture,
            event,
            wire,
            rate,
            channels,
            period,
        })
    })();
    // SAFETY: frees the mix format once, after its last use.
    unsafe { CoTaskMemFree(Some(mix as *const _)) };
    result.map(|c| (c, name))
}

/// The capture thread: drains every packet on each event until `stop`.
fn run(cap: Capture, mut producer: Producer, stop: &Event) {
    let _mmcss = Mmcss::enter();
    let shared = producer.shared().clone();
    let channels = usize::from(cap.channels);
    let fb = channels * cap.wire.bytes();
    // SAFETY: COM calls on the stream this thread owns; the buffer from
    // GetBuffer is valid until ReleaseBuffer.
    let result = unsafe {
        (|| -> Result<(), Error> {
            cap.client
                .Start()
                .map_err(|e| map_error("IAudioClient::Start", e))?;
            let handles = [cap.event.handle(), stop.handle()];
            loop {
                let woke = WaitForMultipleObjects(&handles, false, 2000);
                if woke.0 == WAIT_OBJECT_0.0 + 1 {
                    return Ok(());
                }
                if woke == WAIT_TIMEOUT {
                    return Err(Error::Timeout("the input stopped delivering"));
                }
                if woke != WAIT_OBJECT_0 {
                    return Err(Error::Backend {
                        call: "WaitForMultipleObjects",
                        code: i64::from(woke.0),
                        message: String::new(),
                    });
                }
                while cap
                    .capture
                    .GetNextPacketSize()
                    .map_err(|e| map_error("GetNextPacketSize", e))?
                    > 0
                {
                    let mut data = ptr::null_mut();
                    let mut frames = 0u32;
                    let mut flags = 0u32;
                    cap.capture
                        .GetBuffer(&mut data, &mut frames, &mut flags, None, None)
                        .map_err(|e| map_error("IAudioCaptureClient::GetBuffer", e))?;
                    let n = frames as usize;
                    if flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 != 0 || data.is_null() {
                        producer.push_silence(n);
                    } else {
                        let bytes = std::slice::from_raw_parts(data, n * fb);
                        producer.push(bytes, cap.wire, channels, n);
                    }
                    cap.capture
                        .ReleaseBuffer(frames)
                        .map_err(|e| map_error("IAudioCaptureClient::ReleaseBuffer", e))?;
                }
            }
        })()
    };
    if let Err(e) = result {
        shared.fail(e);
    }
    // SAFETY: as above.
    let _ = unsafe { cap.client.Stop() };
}

/// The running capture: stopping it is dropping it.
struct Running {
    stop: Arc<Event>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for Running {
    fn drop(&mut self) {
        self.stop.set();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

pub(crate) fn open_input(id: &str, mono: Mono) -> Result<Input, Error> {
    let stop = Arc::new(Event::new()?);
    let stopped = stop.clone();
    let id = id.to_owned();
    let ((info, opened), thread) = spawn_opened(
        "jodel-input-wasapi",
        "opening the input",
        move || {
            let com = Com::init()?;
            let (cap, name) = open_capture(&id)?;
            let (producer, opened) = Opened::new(cap.rate, cap.period, mono, MicClock::system());
            let info = InputInfo {
                id,
                name,
                backend: INPUT_BACKEND,
                rate: cap.rate,
                channels: cap.channels,
                period_frames: cap.period,
                mono,
            };
            Ok(((info, opened), (com, cap, producer)))
        },
        move |(com, cap, producer)| {
            run(cap, producer, &stopped);
            // COM goes last, after the stream.
            drop(com);
        },
    )?;
    let running = Running {
        stop,
        thread: Some(thread),
    };
    Ok(Input::new(info, opened, Box::new(running)))
}
