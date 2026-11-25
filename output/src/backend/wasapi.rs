//! Windows: WASAPI in shared mode, event-driven.
//!
//! Device ids are the endpoint ids from `IMMDevice::GetId`; names are the
//! endpoints' friendly names. A device is shared with every other
//! application. Its one format ([`mix_format`]) is the audio engine's mix
//! format (`GetMixFormat`): the rate and sample format set for the device
//! in the system's sound settings (usually 32-bit float at 48 kHz), in
//! stereo, which the engine maps to its own channels
//! (`AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM`; at the mix rate it converts
//! nothing else).
//!
//! The stream lives on its own thread: COM in the multithreaded apartment,
//! `Initialize(SHARED, EVENTCALLBACK)` with a buffer of two engine periods
//! (so a late wake-up is not a gap), MMCSS "Pro Audio", the buffer filled
//! before `Start`, then on every event as much as the engine has taken
//! (`GetBufferSize` less `GetCurrentPadding`). The position comes from
//! `IAudioClock` after every buffer, at the performance-counter time it
//! stamped it: what the endpoint plays, after what the engine and the
//! driver hold (about 25 ms on the developer's machine). Pause is `Stop`
//! (the engine keeps the buffer and the clock); flush is `Stop` + `Reset`.
//! A change of the device's format in the sound settings, or the device
//! going away, invalidates the stream: [`Error::DeviceLost`].

use super::thread::{Command, Request, ThreadDriver};
use crate::device::Device;
use crate::error::Error;
use crate::format::{SampleFormat, StreamFormat};
use crate::output::{Core, OpenRequest, StreamInfo};
use crate::shared::Shared;
use std::ptr;
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::time::{Duration, Instant};
use windows::core::{HRESULT, HSTRING, PWSTR};
use windows::Win32::Devices::FunctionDiscovery::PKEY_Device_FriendlyName;
use windows::Win32::Foundation::{
    CloseHandle, ERROR_NOT_FOUND, HANDLE, RPC_E_CHANGED_MODE, WAIT_OBJECT_0, WAIT_TIMEOUT,
};
use windows::Win32::Media::Audio::{
    eConsole, eRender, EDataFlow, IAudioClient, IAudioClock, IAudioRenderClient, IMMDevice,
    IMMDeviceEnumerator, MMDeviceEnumerator, AUDCLNT_E_DEVICE_INVALIDATED, AUDCLNT_E_DEVICE_IN_USE,
    AUDCLNT_E_RESOURCES_INVALIDATED, AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM,
    AUDCLNT_STREAMFLAGS_EVENTCALLBACK, DEVICE_STATE_ACTIVE, WAVEFORMATEX, WAVEFORMATEXTENSIBLE,
    WAVE_FORMAT_PCM,
};
use windows::Win32::Media::KernelStreaming::{KSDATAFORMAT_SUBTYPE_PCM, WAVE_FORMAT_EXTENSIBLE};
use windows::Win32::Media::Multimedia::{KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, WAVE_FORMAT_IEEE_FLOAT};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoTaskMemFree, CoUninitialize, CLSCTX_ALL,
    COINIT_MULTITHREADED, STGM_READ,
};
use windows::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
use windows::Win32::System::Threading::{
    AvRevertMmThreadCharacteristics, AvSetMmThreadCharacteristicsW, CreateEventW, SetEvent,
    WaitForMultipleObjects, INFINITE,
};

pub(crate) const BACKEND: &str = "WASAPI (shared)";

/// COM for the current thread, balanced on drop. A thread already in a
/// single-threaded apartment keeps it.
pub(super) struct Com(bool);

impl Com {
    pub(super) fn init() -> Result<Com, Error> {
        // SAFETY: plain COM initialisation for this thread.
        let hr = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
        if hr.is_ok() {
            Ok(Com(true))
        } else if hr == RPC_E_CHANGED_MODE {
            Ok(Com(false))
        } else {
            Err(backend_error("CoInitializeEx", hr, ""))
        }
    }
}

impl Drop for Com {
    fn drop(&mut self) {
        if self.0 {
            // SAFETY: balances the successful CoInitializeEx above.
            unsafe { CoUninitialize() };
        }
    }
}

/// A kernel event handle that the handle thread signals and the device
/// thread waits on.
#[derive(Clone, Copy)]
pub(super) struct Event(pub(super) HANDLE);
// SAFETY: event handles are process-wide kernel objects, usable from any thread.
unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Event {
    pub(super) fn new() -> Result<Event, Error> {
        // SAFETY: an unnamed auto-reset event, initially unset.
        unsafe { CreateEventW(None, false, false, None) }
            .map(Event)
            .map_err(|e| map_error("CreateEventW", e))
    }

    pub(super) fn set(self) {
        // SAFETY: the handle is open until `close`.
        let _ = unsafe { SetEvent(self.0) };
    }

    pub(super) fn close(self) {
        // SAFETY: called once, when nothing waits on the handle any more.
        let _ = unsafe { CloseHandle(self.0) };
    }
}

fn backend_error(call: &'static str, hr: HRESULT, message: &str) -> Error {
    Error::Backend {
        call,
        code: i64::from(hr.0),
        message: message.into(),
    }
}

pub(super) fn map_error(call: &'static str, e: windows::core::Error) -> Error {
    let hr = e.code();
    if hr == AUDCLNT_E_DEVICE_IN_USE {
        // Another application holds it in exclusive mode.
        Error::DeviceBusy
    } else if hr == AUDCLNT_E_DEVICE_INVALIDATED || hr == AUDCLNT_E_RESOURCES_INVALIDATED {
        Error::DeviceLost
    } else {
        backend_error(call, hr, &e.message())
    }
}

pub(super) fn enumerator() -> Result<IMMDeviceEnumerator, Error> {
    // SAFETY: COM is initialised on this thread by the caller.
    unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL) }
        .map_err(|e| map_error("CoCreateInstance", e))
}

fn device_id(device: &IMMDevice) -> Result<String, Error> {
    // SAFETY: GetId returns a CoTaskMem string that we free after copying.
    unsafe {
        let p: PWSTR = device
            .GetId()
            .map_err(|e| map_error("IMMDevice::GetId", e))?;
        let s = p.to_string().unwrap_or_default();
        CoTaskMemFree(Some(p.0 as *const _));
        Ok(s)
    }
}

pub(super) fn friendly_name(device: &IMMDevice) -> Option<String> {
    // SAFETY: reads one string property from the endpoint's store.
    unsafe {
        let store = device.OpenPropertyStore(STGM_READ).ok()?;
        let value = store.GetValue(&PKEY_Device_FriendlyName).ok()?;
        let s = value.to_string();
        (!s.is_empty()).then_some(s)
    }
}

pub(super) fn find(enumerator: &IMMDeviceEnumerator, id: &str) -> Result<IMMDevice, Error> {
    // SAFETY: looks the endpoint up by its id string.
    unsafe { enumerator.GetDevice(&HSTRING::from(id)) }.map_err(|e| {
        if e.code() == ERROR_NOT_FOUND.to_hresult() {
            Error::DeviceNotFound(id.into())
        } else {
            map_error("IMMDeviceEnumerator::GetDevice", e)
        }
    })
}

pub(super) fn activate(device: &IMMDevice) -> Result<IAudioClient, Error> {
    // SAFETY: activates a fresh audio client on the endpoint.
    unsafe { device.Activate::<IAudioClient>(CLSCTX_ALL, None) }
        .map_err(|e| map_error("IMMDevice::Activate", e))
}

pub(crate) fn devices() -> Result<Vec<Device>, Error> {
    endpoints(eRender)
}

/// The active endpoints of `flow` (render or capture).
pub(super) fn endpoints(flow: EDataFlow) -> Result<Vec<Device>, Error> {
    let _com = Com::init()?;
    let en = enumerator()?;
    // SAFETY: COM calls on objects owned here.
    unsafe {
        let default_id = en
            .GetDefaultAudioEndpoint(flow, eConsole)
            .ok()
            .and_then(|d| device_id(&d).ok());
        let all = en
            .EnumAudioEndpoints(flow, DEVICE_STATE_ACTIVE)
            .map_err(|e| map_error("EnumAudioEndpoints", e))?;
        let n = all.GetCount().map_err(|e| map_error("GetCount", e))?;
        let mut out = Vec::with_capacity(n as usize);
        for i in 0..n {
            let dev = all.Item(i).map_err(|e| map_error("Item", e))?;
            let id = device_id(&dev)?;
            let name = friendly_name(&dev).unwrap_or_else(|| id.clone());
            out.push(Device {
                is_default: default_id.as_deref() == Some(id.as_str()),
                id,
                name,
            });
        }
        Ok(out)
    }
}

/// The sample format of a container of `bits` with `valid` bits, float or
/// integer, if it is one of ours.
fn sample_of(float: bool, bits: u16, valid: u16) -> Option<SampleFormat> {
    match (float, bits, valid) {
        (true, 32, _) => Some(SampleFormat::F32),
        (false, 16, 16) => Some(SampleFormat::S16),
        (false, 24, 24) => Some(SampleFormat::S24Packed),
        // Valid bits are left-justified in a WASAPI container.
        (false, 32, 24) => Some(SampleFormat::S24In32),
        (false, 32, 32) => Some(SampleFormat::S32),
        _ => None,
    }
}

/// What a device takes: the engine's mix format's rate and sample format,
/// in stereo (the engine maps stereo to its own channels).
fn mix_of(client: &IAudioClient) -> Result<StreamFormat, Error> {
    // SAFETY: GetMixFormat returns a CoTaskMem WAVEFORMATEX(-TENSIBLE),
    // read unaligned (the structs are packed), the extensible part only
    // where it says it has one, and freed.
    let (w, float, valid) = unsafe {
        let p = client
            .GetMixFormat()
            .map_err(|e| map_error("GetMixFormat", e))?;
        let w: WAVEFORMATEX = ptr::read_unaligned(p);
        let tag = u32::from(w.wFormatTag);
        let read = if tag == WAVE_FORMAT_EXTENSIBLE && w.cbSize >= 22 {
            let x: WAVEFORMATEXTENSIBLE = ptr::read_unaligned(p as *const WAVEFORMATEXTENSIBLE);
            let sub = x.SubFormat;
            let float = sub == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
            let pcm = sub == KSDATAFORMAT_SUBTYPE_PCM;
            (float || pcm).then_some((float, x.Samples.wValidBitsPerSample))
        } else {
            let float = tag == WAVE_FORMAT_IEEE_FLOAT;
            (float || tag == WAVE_FORMAT_PCM).then_some((float, w.wBitsPerSample))
        };
        CoTaskMemFree(Some(p as *const _));
        let (float, valid) = read.ok_or_else(unreadable)?;
        (w, float, valid)
    };
    let sample = sample_of(float, w.wBitsPerSample, valid).ok_or_else(unreadable)?;
    if w.nChannels == 0 || w.nSamplesPerSec == 0 {
        return Err(unreadable());
    }
    Ok(StreamFormat::new(w.nSamplesPerSec, 2, sample))
}

fn unreadable() -> Error {
    Error::Unsupported("the device's mix format is not PCM or 32-bit float".into())
}

/// The wave format for `f`, stereo (front left and right).
fn descriptor(f: StreamFormat) -> WAVEFORMATEXTENSIBLE {
    let container = (f.sample.bytes() * 8) as u16;
    let (valid, sub) = match f.sample {
        SampleFormat::F32 => (32, KSDATAFORMAT_SUBTYPE_IEEE_FLOAT),
        other => (other.valid_bits() as u16, KSDATAFORMAT_SUBTYPE_PCM),
    };
    let block = f.channels * container / 8;
    let mut w = WAVEFORMATEXTENSIBLE::default();
    w.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE as u16;
    w.Format.nChannels = f.channels;
    w.Format.nSamplesPerSec = f.rate;
    w.Format.wBitsPerSample = container;
    w.Format.nBlockAlign = block;
    w.Format.nAvgBytesPerSec = f.rate * u32::from(block);
    w.Format.cbSize = 22;
    w.Samples.wValidBitsPerSample = valid;
    w.dwChannelMask = 0x3;
    w.SubFormat = sub;
    w
}

pub(crate) fn mix_format(id: &str) -> Result<StreamFormat, Error> {
    let _com = Com::init()?;
    let dev = find(&enumerator()?, id)?;
    mix_of(&activate(&dev)?)
}

/// 100 ns units as frames at `rate`.
fn hns_to_frames(hns: i64, rate: u32) -> u64 {
    (hns.max(0) as u64 * u64::from(rate)).div_ceil(10_000_000)
}

/// The performance counter now, in 100 ns units (as `IAudioClock` stamps
/// its positions).
fn qpc_now() -> u64 {
    let (mut count, mut freq) = (0i64, 0i64);
    // SAFETY: two out parameters on the stack; both calls cannot fail on
    // Windows XP and later.
    unsafe {
        let _ = QueryPerformanceCounter(&mut count);
        let _ = QueryPerformanceFrequency(&mut freq);
    }
    (i128::from(count.max(0)) * 10_000_000 / i128::from(freq.max(1))) as u64
}

/// Everything the device thread owns.
struct Stream {
    client: IAudioClient,
    render: IAudioRenderClient,
    clock: IAudioClock,
    clock_hz: u64,
    event: Event,
    /// The engine buffer, in frames.
    frames: u32,
    format: StreamFormat,
}

/// The stream and the engine's period in frames.
fn open_stream(id: &str, request: &OpenRequest) -> Result<(Stream, u64), Error> {
    let dev = find(&enumerator()?, id)?;
    let client = activate(&dev)?;
    let format = mix_of(&client)?;
    if !request.takes(format) {
        return Err(Error::FormatNotAccepted {
            rate: request.rate,
            channels: request.channels,
        });
    }
    let w = descriptor(format);
    let event = Event::new()?;
    // SAFETY: COM calls on the client owned here; `w` outlives them.
    let setup = unsafe {
        (|| -> Result<(Stream, u64), Error> {
            let mut period = 0i64;
            client
                .GetDevicePeriod(Some(&mut period), None)
                .map_err(|e| map_error("GetDevicePeriod", e))?;
            let asked = (request.period.as_nanos() / 100).min(i64::MAX as u128) as i64;
            let period = period.max(asked);
            client
                .Initialize(
                    AUDCLNT_SHAREMODE_SHARED,
                    AUDCLNT_STREAMFLAGS_EVENTCALLBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM,
                    2 * period,
                    0,
                    &w as *const _ as *const WAVEFORMATEX,
                    None,
                )
                .map_err(|e| map_error("IAudioClient::Initialize", e))?;
            client
                .SetEventHandle(event.0)
                .map_err(|e| map_error("SetEventHandle", e))?;
            let frames = client
                .GetBufferSize()
                .map_err(|e| map_error("GetBufferSize", e))?;
            let render: IAudioRenderClient = client
                .GetService()
                .map_err(|e| map_error("GetService(render)", e))?;
            let clock: IAudioClock = client
                .GetService()
                .map_err(|e| map_error("GetService(clock)", e))?;
            let clock_hz = clock
                .GetFrequency()
                .map_err(|e| map_error("IAudioClock::GetFrequency", e))?;
            let period = hns_to_frames(period, format.rate).clamp(1, u64::from(frames));
            Ok((
                Stream {
                    client: client.clone(),
                    render,
                    clock,
                    clock_hz: clock_hz.max(1),
                    event,
                    frames,
                    format,
                },
                period,
            ))
        })()
    };
    if setup.is_err() {
        event.close();
    }
    setup
}

impl Stream {
    /// Hands the engine what it has room for, from the ring.
    fn fill(&self, shared: &Shared) -> Result<(), Error> {
        // SAFETY: GetBuffer returns `room` frames of engine memory, valid
        // until ReleaseBuffer.
        unsafe {
            let queued = self
                .client
                .GetCurrentPadding()
                .map_err(|e| map_error("GetCurrentPadding", e))?;
            let room = self.frames.saturating_sub(queued);
            if room == 0 {
                return Ok(());
            }
            let data = self
                .render
                .GetBuffer(room)
                .map_err(|e| map_error("GetBuffer", e))?;
            let bytes = room as usize * self.format.frame_bytes();
            shared.render(std::slice::from_raw_parts_mut(data, bytes));
            self.render
                .ReleaseBuffer(room, 0)
                .map_err(|e| map_error("ReleaseBuffer", e))
        }
    }

    /// Publishes the device clock's position, at the time the clock
    /// stamped it.
    fn report(&self, shared: &Shared, running: bool) {
        let (mut pos, mut stamped) = (0u64, 0u64);
        // SAFETY: a clock read on the thread that owns the clock.
        if unsafe { self.clock.GetPosition(&mut pos, Some(&mut stamped)) }.is_ok() {
            let frames = u128::from(pos) * u128::from(self.format.rate) / u128::from(self.clock_hz);
            let ago = Duration::from_nanos(qpc_now().saturating_sub(stamped) * 100);
            let at = Instant::now().checked_sub(ago).unwrap_or_else(Instant::now);
            shared.report(frames as u64, at, running);
        }
    }
}

/// Raises the thread to MMCSS "Pro Audio" for its lifetime, if it can.
pub(crate) struct Mmcss(Option<HANDLE>);

impl Mmcss {
    pub(crate) fn enter() -> Mmcss {
        let mut index = 0u32;
        // SAFETY: registers this thread with an MMCSS task by name.
        Mmcss(
            unsafe { AvSetMmThreadCharacteristicsW(windows::core::w!("Pro Audio"), &mut index) }
                .ok(),
        )
    }
}

impl Drop for Mmcss {
    fn drop(&mut self) {
        if let Some(h) = self.0 {
            // SAFETY: reverts the registration made in `enter`.
            let _ = unsafe { AvRevertMmThreadCharacteristics(h) };
        }
    }
}

/// The device thread: owns the stream until `Close`.
fn run(stream: Stream, shared: Arc<Shared>, commands: Receiver<Request>, wake: Event) {
    let _mmcss = Mmcss::enter();
    let handles = [stream.event.0, wake.0];
    let mut running = false;
    // The engine buffer holds a prefill that has not been reset.
    let mut primed = false;
    loop {
        // SAFETY: both handles stay open while this loop runs.
        let woke = unsafe {
            WaitForMultipleObjects(&handles, false, if running { 2000 } else { INFINITE })
        };
        if woke == WAIT_OBJECT_0 {
            if running {
                if let Err(e) = stream.fill(&shared) {
                    shared.fail(e);
                    // SAFETY: stops a stream that can no longer be fed.
                    let _ = unsafe { stream.client.Stop() };
                    running = false;
                    continue;
                }
                stream.report(&shared, true);
            }
            continue;
        }
        if woke == WAIT_TIMEOUT {
            // A running stream is signalled every engine period.
            shared.fail(Error::Timeout("the device stopped asking for data"));
            // SAFETY: as above.
            let _ = unsafe { stream.client.Stop() };
            running = false;
            continue;
        }
        if woke.0 != WAIT_OBJECT_0.0 + 1 {
            shared.fail(Error::Backend {
                call: "WaitForMultipleObjects",
                code: i64::from(woke.0),
                message: String::new(),
            });
            running = false;
        }
        while let Ok(request) = commands.try_recv() {
            let command = request.command;
            // SAFETY: COM calls on the stream this thread owns.
            let result = unsafe {
                match command {
                    Command::Start if running => Ok(()),
                    Command::Start => (|| {
                        shared.check()?;
                        if !primed {
                            stream.fill(&shared)?;
                            primed = true;
                        }
                        stream
                            .client
                            .Start()
                            .map_err(|e| map_error("IAudioClient::Start", e))?;
                        running = true;
                        Ok(())
                    })(),
                    Command::Pause => {
                        let r = if running {
                            stream
                                .client
                                .Stop()
                                .map_err(|e| map_error("IAudioClient::Stop", e))
                        } else {
                            Ok(())
                        };
                        running = false;
                        stream.report(&shared, false);
                        r
                    }
                    Command::Flush => {
                        if running {
                            let _ = stream.client.Stop();
                        }
                        running = false;
                        shared.discard_all();
                        primed = false;
                        stream
                            .client
                            .Reset()
                            .map_err(|e| map_error("IAudioClient::Reset", e))
                    }
                    Command::Close => {
                        if running {
                            let _ = stream.client.Stop();
                        }
                        running = false;
                        shared.freeze();
                        Ok(())
                    }
                }
            };
            request.reply(result);
            if command == Command::Close {
                stream.event.close();
                return;
            }
        }
    }
}

/// Opens `id` shared; the stream runs on its own thread.
pub(crate) fn open(id: &str, request: &OpenRequest) -> Result<Core, Error> {
    request.validate()?;
    let wake = Event::new()?;
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let id_owned = id.to_owned();
    let request_owned = request.clone();
    let driver = ThreadDriver::spawn(
        "jodel-output-wasapi",
        Box::new(move || wake.set()),
        move |commands| {
            let com = match Com::init() {
                Ok(c) => c,
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            match open_stream(&id_owned, &request_owned) {
                Ok((stream, period)) => {
                    let shared = Arc::new(Shared::new(
                        stream.format,
                        request_owned.ring_frames(&stream.format, u64::from(stream.frames)),
                    ));
                    let info = StreamInfo {
                        format: stream.format,
                        backend: BACKEND,
                        device_buffer_frames: period,
                        ring_frames: shared.ring.capacity(),
                        realtime: true,
                    };
                    let _ = ready_tx.send(Ok((info, shared.clone())));
                    run(stream, shared, commands, wake);
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    // Answer the handle's close.
                    while let Ok(r) = commands.recv() {
                        let done = r.command == Command::Close;
                        r.reply(Ok(()));
                        if done {
                            break;
                        }
                    }
                }
            }
            drop(com);
        },
    );
    let mut driver = match driver {
        Ok(d) => d,
        Err(e) => {
            wake.close();
            return Err(e);
        }
    };
    let opened = ready_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap_or(Err(Error::Timeout("opening the device")));
    match opened {
        Ok((info, shared)) => Ok(Core::new(
            shared,
            Box::new(WasapiDriver {
                inner: driver,
                wake,
            }),
            info,
            request,
        )),
        Err(e) => {
            use crate::output::Driver;
            let _ = driver.close();
            wake.close();
            Err(e)
        }
    }
}

/// The thread driver plus the wake event it owns.
struct WasapiDriver {
    inner: ThreadDriver,
    wake: Event,
}

impl crate::output::Driver for WasapiDriver {
    fn start(&mut self) -> Result<(), Error> {
        self.inner.start()
    }
    fn pause(&mut self) -> Result<(), Error> {
        self.inner.pause()
    }
    fn flush(&mut self) -> Result<(), Error> {
        self.inner.flush()
    }
    fn close(&mut self) -> Result<(), Error> {
        let r = self.inner.close();
        if r.is_ok() {
            self.wake.close();
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fields that matter, copied out of the packed struct.
    fn fields(w: &WAVEFORMATEXTENSIBLE) -> (u16, u16, u16, u16, u32, u32, bool) {
        let f = w.Format;
        // SAFETY: every variant of the samples union is a u16.
        let valid = unsafe { w.Samples.wValidBitsPerSample };
        let mask = w.dwChannelMask;
        let sub = w.SubFormat;
        (
            f.wFormatTag,
            f.wBitsPerSample,
            valid,
            f.nBlockAlign,
            f.nAvgBytesPerSec,
            mask,
            sub == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT,
        )
    }

    #[test]
    fn descriptors_follow_the_container_rules() {
        let f = |s| StreamFormat::new(48_000, 2, s);
        assert_eq!(
            fields(&descriptor(f(SampleFormat::F32))),
            (0xFFFE, 32, 32, 8, 48_000 * 8, 0x3, true)
        );
        assert_eq!(
            fields(&descriptor(f(SampleFormat::S24In32))),
            (0xFFFE, 32, 24, 8, 48_000 * 8, 0x3, false)
        );
        assert_eq!(
            fields(&descriptor(f(SampleFormat::S16))),
            (0xFFFE, 16, 16, 4, 48_000 * 4, 0x3, false)
        );
    }

    #[test]
    fn mix_formats_map_to_sample_formats() {
        assert_eq!(sample_of(true, 32, 32), Some(SampleFormat::F32));
        assert_eq!(sample_of(false, 32, 24), Some(SampleFormat::S24In32));
        assert_eq!(sample_of(false, 32, 32), Some(SampleFormat::S32));
        assert_eq!(sample_of(false, 24, 24), Some(SampleFormat::S24Packed));
        assert_eq!(sample_of(false, 16, 16), Some(SampleFormat::S16));
        assert_eq!(sample_of(true, 64, 64), None);
        assert_eq!(sample_of(false, 32, 20), None);
        // 10 ms of 100 ns units at 48 kHz.
        assert_eq!(hns_to_frames(100_000, 48_000), 480);
    }
}
