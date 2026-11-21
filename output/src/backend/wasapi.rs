//! Windows: WASAPI in exclusive mode, event-driven.
//!
//! Device ids are the endpoint ids from `IMMDevice::GetId`; names are the
//! endpoints' friendly names. Formats are asked with
//! `IsFormatSupported(EXCLUSIVE)` as `WAVEFORMATEXTENSIBLE` and, for one or
//! two channels, also as a plain `WAVEFORMATEX` (some drivers accept only
//! one of the two). WASAPI has no low-aligned 24-in-32 container, so
//! [`SampleFormat::S24In32Low`] is never accepted here.
//!
//! The stream lives on its own thread: COM in the multithreaded apartment,
//! `Initialize(EXCLUSIVE, EVENTCALLBACK)` with the buffer-alignment retry,
//! MMCSS "Pro Audio", one buffer prefilled before `Start`, then one device
//! buffer per event. The position comes from `IAudioClock` after every
//! buffer. Pause is `Stop` (the device keeps its buffer and its clock);
//! flush is `Stop` + `Reset`.

use super::thread::{Command, Request, ThreadDriver};
use crate::caps::{candidates, Capabilities, PROBE_RATES};
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
    CloseHandle, ERROR_NOT_FOUND, HANDLE, RPC_E_CHANGED_MODE, S_OK, WAIT_OBJECT_0, WAIT_TIMEOUT,
};
use windows::Win32::Media::Audio::{
    eConsole, eRender, EDataFlow, IAudioClient, IAudioClock, IAudioRenderClient, IMMDevice,
    IMMDeviceEnumerator, MMDeviceEnumerator, AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED,
    AUDCLNT_E_DEVICE_INVALIDATED, AUDCLNT_E_DEVICE_IN_USE, AUDCLNT_E_EXCLUSIVE_MODE_NOT_ALLOWED,
    AUDCLNT_E_RESOURCES_INVALIDATED, AUDCLNT_E_UNSUPPORTED_FORMAT, AUDCLNT_SHAREMODE_EXCLUSIVE,
    AUDCLNT_STREAMFLAGS_EVENTCALLBACK, DEVICE_STATE_ACTIVE, WAVEFORMATEX, WAVEFORMATEXTENSIBLE,
    WAVE_FORMAT_PCM,
};
use windows::Win32::Media::KernelStreaming::{
    KSDATAFORMAT_SUBTYPE_PCM, SPEAKER_FRONT_CENTER, WAVE_FORMAT_EXTENSIBLE,
};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoTaskMemFree, CoUninitialize, CLSCTX_ALL,
    COINIT_MULTITHREADED, STGM_READ,
};
use windows::Win32::System::Threading::{
    AvRevertMmThreadCharacteristics, AvSetMmThreadCharacteristicsW, CreateEventW, SetEvent,
    WaitForMultipleObjects, INFINITE,
};

pub(crate) const BACKEND: &str = "WASAPI exclusive";

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
    map_hresult(call, e.code(), &e.message())
}

fn map_hresult(call: &'static str, hr: HRESULT, message: &str) -> Error {
    if hr == AUDCLNT_E_DEVICE_IN_USE {
        Error::DeviceBusy
    } else if hr == AUDCLNT_E_EXCLUSIVE_MODE_NOT_ALLOWED {
        Error::ExclusiveNotAllowed
    } else if hr == AUDCLNT_E_DEVICE_INVALIDATED || hr == AUDCLNT_E_RESOURCES_INVALIDATED {
        Error::DeviceLost
    } else {
        backend_error(call, hr, message)
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

/// The endpoint's own channel count and mask, from its mix format.
fn native_layout(client: &IAudioClient) -> (u16, u32) {
    // SAFETY: GetMixFormat returns a CoTaskMem WAVEFORMATEX(-TENSIBLE),
    // read unaligned (the struct is packed) and freed.
    unsafe {
        match client.GetMixFormat() {
            Ok(p) => {
                let w: WAVEFORMATEX = ptr::read_unaligned(p);
                let mask = if u32::from(w.wFormatTag) == WAVE_FORMAT_EXTENSIBLE && w.cbSize >= 22 {
                    let x: WAVEFORMATEXTENSIBLE =
                        ptr::read_unaligned(p as *const WAVEFORMATEXTENSIBLE);
                    x.dwChannelMask
                } else {
                    0
                };
                CoTaskMemFree(Some(p as *const _));
                (w.nChannels, mask)
            }
            Err(_) => (2, 0x3),
        }
    }
}

fn channel_mask(channels: u16, native: (u16, u32)) -> u32 {
    match channels {
        1 => SPEAKER_FRONT_CENTER,
        2 => 0x3, // front left | front right
        n if n == native.0 => native.1,
        _ => 0,
    }
}

/// The wave format for `f`, as WAVEFORMATEXTENSIBLE or (when `extensible`
/// is false) a plain WAVEFORMATEX, if WASAPI can describe it that way.
fn descriptor(f: StreamFormat, mask: u32, extensible: bool) -> Option<WAVEFORMATEXTENSIBLE> {
    let container = (f.sample.bytes() * 8) as u16;
    let valid = match f.sample {
        // Valid bits are left-justified in a WASAPI container.
        SampleFormat::S24In32Low => return None,
        other => other.valid_bits() as u16,
    };
    if !extensible && (container != valid || f.channels > 2) {
        return None;
    }
    let block = f.channels * container / 8;
    let mut w = WAVEFORMATEXTENSIBLE::default();
    w.Format.wFormatTag = if extensible {
        WAVE_FORMAT_EXTENSIBLE as u16
    } else {
        WAVE_FORMAT_PCM as u16
    };
    w.Format.nChannels = f.channels;
    w.Format.nSamplesPerSec = f.rate;
    w.Format.wBitsPerSample = container;
    w.Format.nBlockAlign = block;
    w.Format.nAvgBytesPerSec = f.rate * u32::from(block);
    w.Format.cbSize = if extensible { 22 } else { 0 };
    w.Samples.wValidBitsPerSample = valid;
    w.dwChannelMask = mask;
    w.SubFormat = KSDATAFORMAT_SUBTYPE_PCM;
    Some(w)
}

/// Asks the driver whether it takes `f` exclusively, in either descriptor.
/// Returns the descriptor it took, or the last refusal code.
fn accepted(
    client: &IAudioClient,
    f: StreamFormat,
    mask: u32,
) -> Result<WAVEFORMATEXTENSIBLE, HRESULT> {
    let mut last = AUDCLNT_E_UNSUPPORTED_FORMAT;
    for extensible in [true, false] {
        let Some(w) = descriptor(f, mask, extensible) else {
            continue;
        };
        // SAFETY: `w` outlives the call; no closest match is asked for.
        let hr = unsafe {
            client.IsFormatSupported(
                AUDCLNT_SHAREMODE_EXCLUSIVE,
                &w as *const _ as *const WAVEFORMATEX,
                None,
            )
        };
        if hr == S_OK {
            return Ok(w);
        }
        last = hr;
    }
    Err(last)
}

/// A refusal that means the whole device, not the format, is unavailable.
fn device_refusal(hr: HRESULT) -> Option<Error> {
    (hr == AUDCLNT_E_EXCLUSIVE_MODE_NOT_ALLOWED
        || hr == AUDCLNT_E_DEVICE_IN_USE
        || hr == AUDCLNT_E_DEVICE_INVALIDATED)
        .then(|| map_hresult("IsFormatSupported", hr, ""))
}

pub(crate) fn capabilities(id: &str) -> Result<Capabilities, Error> {
    let _com = Com::init()?;
    let dev = find(&enumerator()?, id)?;
    let client = activate(&dev)?;
    let native = native_layout(&client);
    let mut layouts = vec![(2u16, 0x3u32)];
    if native.0 != 2 && native.0 > 0 {
        layouts.push(native);
    }
    let mut caps = Capabilities::new();
    let mut refusal = None;
    for (channels, mask) in layouts {
        for rate in PROBE_RATES {
            for sample in SampleFormat::ALL {
                let f = StreamFormat::new(rate, channels, sample);
                match accepted(&client, f, mask) {
                    Ok(_) => caps.insert(f),
                    Err(hr) => refusal = refusal.or(device_refusal(hr)),
                }
            }
        }
    }
    match refusal {
        Some(e) if caps.is_empty() => Err(e),
        _ => Ok(caps),
    }
}

/// Converts frames at `rate` to 100 ns units, rounded as the alignment
/// retry in the WASAPI documentation does.
fn frames_to_hns(frames: u32, rate: u32) -> i64 {
    (10_000.0 * 1000.0 / f64::from(rate) * f64::from(frames) + 0.5) as i64
}

/// `Initialize(EXCLUSIVE, EVENTCALLBACK)` with the alignment retry: on
/// `AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED`, read the aligned size from the
/// failed client, release it and initialise a fresh one with that size.
fn initialize(
    device: &IMMDevice,
    w: &WAVEFORMATEXTENSIBLE,
    period: Duration,
) -> Result<IAudioClient, Error> {
    let client = activate(device)?;
    let fmt = w as *const _ as *const WAVEFORMATEX;
    let rate = w.Format.nSamplesPerSec;
    let (mut default_period, mut minimum) = (0i64, 0i64);
    // SAFETY: COM calls on a client owned here; `w` outlives them.
    unsafe {
        client
            .GetDevicePeriod(Some(&mut default_period), Some(&mut minimum))
            .map_err(|e| map_error("GetDevicePeriod", e))?;
        let hns = (period.as_nanos() / 100).min(i64::MAX as u128) as i64;
        let hns = hns.max(minimum);
        match client.Initialize(
            AUDCLNT_SHAREMODE_EXCLUSIVE,
            AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
            hns,
            hns,
            fmt,
            None,
        ) {
            Ok(()) => Ok(client),
            Err(e) if e.code() == AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED => {
                let frames = client
                    .GetBufferSize()
                    .map_err(|e| map_error("GetBufferSize", e))?;
                drop(client);
                let hns = frames_to_hns(frames, rate);
                let client = activate(device)?;
                client
                    .Initialize(
                        AUDCLNT_SHAREMODE_EXCLUSIVE,
                        AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                        hns,
                        hns,
                        fmt,
                        None,
                    )
                    .map_err(|e| map_error("IAudioClient::Initialize", e))?;
                Ok(client)
            }
            Err(e) => Err(map_error("IAudioClient::Initialize", e)),
        }
    }
}

/// Everything the device thread owns.
struct Stream {
    client: IAudioClient,
    render: IAudioRenderClient,
    clock: IAudioClock,
    clock_hz: u64,
    event: Event,
    frames: u32,
    format: StreamFormat,
}

fn open_stream(id: &str, request: &OpenRequest) -> Result<(Stream, Duration), Error> {
    let dev = find(&enumerator()?, id)?;
    let probe = activate(&dev)?;
    let native = native_layout(&probe);
    let mask = channel_mask(request.channels, native);
    let mut refusal = None;
    let mut chosen = None;
    for sample in candidates(&request.formats) {
        let f = StreamFormat::new(request.rate, request.channels, sample);
        let w = match accepted(&probe, f, mask) {
            Ok(w) => w,
            Err(hr) => {
                refusal = refusal.or(device_refusal(hr));
                continue;
            }
        };
        match initialize(&dev, &w, request.period) {
            Ok(client) => {
                chosen = Some((client, f));
                break;
            }
            Err(Error::Backend { code, .. })
                if code == i64::from(AUDCLNT_E_UNSUPPORTED_FORMAT.0) =>
            {
                continue
            }
            Err(e) => return Err(e),
        }
    }
    drop(probe);
    let Some((client, format)) = chosen else {
        return Err(refusal.unwrap_or(Error::FormatNotAccepted {
            rate: request.rate,
            channels: request.channels,
        }));
    };
    let event = Event::new()?;
    // SAFETY: COM calls on the initialised client owned here.
    let setup = unsafe {
        (|| -> Result<(Stream, Duration), Error> {
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
            let latency = client.GetStreamLatency().unwrap_or(0).max(0) as u64;
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
                Duration::from_nanos(latency * 100),
            ))
        })()
    };
    if setup.is_err() {
        event.close();
    }
    setup
}

impl Stream {
    /// Hands the device one buffer from the ring.
    fn fill(&self, shared: &Shared) -> Result<(), Error> {
        let bytes = self.frames as usize * self.format.frame_bytes();
        // SAFETY: GetBuffer returns `frames` frames of device memory, valid
        // until ReleaseBuffer.
        unsafe {
            let data = self
                .render
                .GetBuffer(self.frames)
                .map_err(|e| map_error("GetBuffer", e))?;
            shared.render(std::slice::from_raw_parts_mut(data, bytes));
            self.render
                .ReleaseBuffer(self.frames, 0)
                .map_err(|e| map_error("ReleaseBuffer", e))
        }
    }

    /// Publishes the device clock's position.
    fn report(&self, shared: &Shared, running: bool) {
        let mut pos = 0u64;
        // SAFETY: a clock read on the thread that owns the clock.
        if unsafe { self.clock.GetPosition(&mut pos, None) }.is_ok() {
            let frames = u128::from(pos) * u128::from(self.format.rate) / u128::from(self.clock_hz);
            shared.report(frames as u64, Instant::now(), running);
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
    // The device buffer holds a prefill that has not been reset.
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
            // A running exclusive stream signals every period.
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

/// Opens `id` exclusively; the stream runs on its own thread.
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
                Ok((stream, latency)) => {
                    let shared = Arc::new(Shared::new(
                        stream.format,
                        request_owned.ring_frames(&stream.format, u64::from(stream.frames)),
                    ));
                    let info = StreamInfo {
                        format: stream.format,
                        backend: BACKEND,
                        exclusive: true,
                        device_buffer_frames: u64::from(stream.frames),
                        device_latency: latency,
                        ring_frames: shared.ring.capacity(),
                        realtime: true,
                        float_io: false,
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
    fn fields(w: &WAVEFORMATEXTENSIBLE) -> (u16, u16, u16, u16, u32, u32) {
        let f = w.Format;
        // SAFETY: every variant of the samples union is a u16.
        let valid = unsafe { w.Samples.wValidBitsPerSample };
        let mask = w.dwChannelMask;
        (
            f.wFormatTag,
            f.wBitsPerSample,
            valid,
            f.nBlockAlign,
            f.nAvgBytesPerSec,
            mask,
        )
    }

    #[test]
    fn descriptors_follow_the_container_rules() {
        let f = |s| StreamFormat::new(96_000, 2, s);
        let w = descriptor(f(SampleFormat::S24In32), 0x3, true).unwrap();
        assert_eq!(fields(&w), (0xFFFE, 32, 24, 8, 96_000 * 8, 0x3));
        assert!(descriptor(f(SampleFormat::S24In32), 0x3, false).is_none());
        assert!(descriptor(f(SampleFormat::S24In32Low), 0x3, true).is_none());
        let w = descriptor(f(SampleFormat::S24Packed), 0x3, false).unwrap();
        assert_eq!(fields(&w), (1, 24, 24, 6, 96_000 * 6, 0x3));
        let eight = StreamFormat::new(48_000, 8, SampleFormat::S16);
        assert!(descriptor(eight, 0x63f, false).is_none());
        assert_eq!(
            fields(&descriptor(eight, 0x63f, true).unwrap()),
            (0xFFFE, 16, 16, 16, 48_000 * 16, 0x63f)
        );
    }

    #[test]
    fn alignment_retry_duration() {
        // 480 frames at 48 kHz is 10 ms.
        assert_eq!(frames_to_hns(480, 48_000), 100_000);
        assert_eq!(frames_to_hns(441, 44_100), 100_000);
    }
}
