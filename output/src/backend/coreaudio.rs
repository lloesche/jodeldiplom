//! macOS: the Core Audio HAL directly, in hog mode.
//!
//! Device ids are Core Audio device UIDs; names are the devices' names.
//! What a device accepts is what its output streams list in
//! `kAudioStreamPropertyAvailablePhysicalFormats`: interleaved, signed,
//! little-endian integer PCM, mapped to [`SampleFormat`] by bits per
//! channel, bytes per channel and the "aligned high" flag. A 32-bit float
//! physical format (all that built-in speakers and many virtual devices
//! list) counts as [`SampleFormat::S24In32`]: every 24-bit value is a
//! float exactly, so the stream reaches the device unchanged, converted at
//! the IO boundary (step 4). An integer format is preferred where a device
//! lists both. Listing them does not touch the device.
//!
//! Opening, in order:
//!
//! 1. take hog mode (`kAudioDevicePropertyHogMode` = our pid; refused if
//!    another process holds it);
//! 2. turn mixing off where the device lets us
//!    (`kAudioDevicePropertySupportsMixing` = 0);
//! 3. set the chosen physical format on the output stream; it carries the
//!    nominal rate. Wait (up to 2 s) until the device reports it;
//! 4. read the stream's virtual format, the format the IOProc exchanges.
//!    If it is the same integer layout, the ring's bytes go to the device
//!    as they are; if it is 32-bit float, each sample is converted exactly
//!    at the IO boundary (every 24-bit value is a float the HAL turns back
//!    into the same integer) and [`StreamInfo::float_io`] says so;
//! 5. ask for an IO buffer of one period and register our IOProc.
//!
//! Closing first has the IOProc deliver silence until the device has taken
//! enough of it to fill what it holds (two IO buffers, its latency and
//! safety offset, and 20 ms; at most [`SETTLE_WAIT`]; a paused device runs
//! again for it), then undoes the
//! rest in reverse: stop, destroy the IOProc, restore the physical format,
//! turn mixing back on, release hog mode. Stopped at once, a device kept
//! our last samples in its buffers, and some (USB interfaces) played them
//! again read as the restored format: a burst of noise.
//!
//! The IOProc fills the device buffer from the ring. The position is the
//! frames delivered minus how far ahead of "now" the buffer's output time
//! is, minus the device and stream latency.

use crate::caps::{candidates, Capabilities, PROBE_RATES};
use crate::device::{Device, Volume};
use crate::error::Error;
use crate::format::{SampleFormat, StreamFormat};
use crate::output::{Core, Driver, OpenRequest, StreamInfo};
use crate::pack::unpack_sample;
use crate::shared::Shared;
use objc2_core_audio::{
    kAudioDevicePropertyBufferFrameSize, kAudioDevicePropertyDeviceIsAlive,
    kAudioDevicePropertyDeviceUID, kAudioDevicePropertyHogMode, kAudioDevicePropertyLatency,
    kAudioDevicePropertySafetyOffset, kAudioDevicePropertyStreams,
    kAudioDevicePropertySupportsMixing, kAudioHardwarePropertyDefaultOutputDevice,
    kAudioHardwarePropertyDevices, kAudioObjectPropertyElementMain, kAudioObjectPropertyName,
    kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyScopeOutput, kAudioObjectSystemObject,
    kAudioStreamPropertyAvailablePhysicalFormats, kAudioStreamPropertyLatency,
    kAudioStreamPropertyPhysicalFormat, kAudioStreamPropertyVirtualFormat,
    AudioDeviceCreateIOProcID, AudioDeviceDestroyIOProcID, AudioDeviceIOProcID, AudioDeviceStart,
    AudioDeviceStop, AudioObjectGetPropertyData, AudioObjectGetPropertyDataSize,
    AudioObjectHasProperty, AudioObjectID, AudioObjectIsPropertySettable,
    AudioObjectPropertyAddress, AudioObjectSetPropertyData, AudioStreamRangedDescription,
};
use objc2_core_audio_types::{
    kAudioFormatFlagIsAlignedHigh, kAudioFormatFlagIsBigEndian, kAudioFormatFlagIsFloat,
    kAudioFormatFlagIsNonInterleaved, kAudioFormatFlagIsSignedInteger, kAudioFormatLinearPCM,
    AudioBuffer, AudioBufferList, AudioStreamBasicDescription, AudioTimeStamp,
};
use objc2_core_foundation::{CFRetained, CFString};
use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::mem::size_of;
use std::ptr::{null, NonNull};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub(crate) const BACKEND: &str = "Core Audio HAL (hog mode)";

pub(super) const GLOBAL: u32 = kAudioObjectPropertyScopeGlobal;
const OUTPUT: u32 = kAudioObjectPropertyScopeOutput;

pub(super) fn addr(selector: u32, scope: u32) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMain,
    }
}

pub(super) fn status_error(call: &'static str, status: i32) -> Error {
    Error::Backend {
        call,
        code: i64::from(status),
        message: String::new(),
    }
}

/// Reads a fixed-size property.
pub(super) fn get<T: Copy>(
    object: AudioObjectID,
    address: AudioObjectPropertyAddress,
    call: &'static str,
) -> Result<T, Error> {
    // SAFETY: `value` is a plain-data out buffer of the size we pass.
    unsafe {
        let mut value: T = std::mem::zeroed();
        let mut size = size_of::<T>() as u32;
        let status = AudioObjectGetPropertyData(
            object,
            NonNull::from(&address),
            0,
            null(),
            NonNull::from(&mut size),
            NonNull::from(&mut value).cast(),
        );
        if status == 0 {
            Ok(value)
        } else {
            Err(status_error(call, status))
        }
    }
}

/// Reads an array property.
pub(super) fn get_vec<T: Copy>(
    object: AudioObjectID,
    address: AudioObjectPropertyAddress,
) -> Vec<T> {
    // SAFETY: the buffer is sized from the property's reported size.
    unsafe {
        let mut size = 0u32;
        if AudioObjectGetPropertyDataSize(
            object,
            NonNull::from(&address),
            0,
            null(),
            NonNull::from(&mut size),
        ) != 0
            || (size as usize) < size_of::<T>()
        {
            return Vec::new();
        }
        let mut values: Vec<T> = (0..size as usize / size_of::<T>())
            .map(|_| std::mem::zeroed())
            .collect();
        let mut size = (values.len() * size_of::<T>()) as u32;
        let status = AudioObjectGetPropertyData(
            object,
            NonNull::from(&address),
            0,
            null(),
            NonNull::from(&mut size),
            NonNull::new_unchecked(values.as_mut_ptr().cast()),
        );
        if status != 0 {
            return Vec::new();
        }
        values.truncate(size as usize / size_of::<T>());
        values
    }
}

pub(super) fn set<T: Copy>(
    object: AudioObjectID,
    address: AudioObjectPropertyAddress,
    value: T,
    call: &'static str,
) -> Result<(), Error> {
    // SAFETY: passes a plain-data value of its own size.
    let status = unsafe {
        AudioObjectSetPropertyData(
            object,
            NonNull::from(&address),
            0,
            null(),
            size_of::<T>() as u32,
            NonNull::from(&value).cast(),
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(status_error(call, status))
    }
}

fn settable(object: AudioObjectID, address: AudioObjectPropertyAddress) -> bool {
    let mut flag: u8 = 0;
    // SAFETY: a one-byte Boolean out parameter.
    unsafe {
        AudioObjectIsPropertySettable(object, NonNull::from(&address), NonNull::from(&mut flag))
            == 0
            && flag != 0
    }
}

/// Reads a CFString property and releases it.
pub(super) fn get_string(object: AudioObjectID, selector: u32) -> Option<String> {
    let p: *const CFString =
        get(object, addr(selector, GLOBAL), "AudioObjectGetPropertyData").ok()?;
    // SAFETY: string properties are returned with +1 retain count.
    let s = unsafe { CFRetained::from_raw(NonNull::new(p as *mut CFString)?) };
    Some(s.to_string())
}

fn output_streams(device: AudioObjectID) -> Vec<AudioObjectID> {
    get_vec(device, addr(kAudioDevicePropertyStreams, OUTPUT))
}

pub(super) fn system_devices() -> Vec<AudioObjectID> {
    get_vec(
        kAudioObjectSystemObject as AudioObjectID,
        addr(kAudioHardwarePropertyDevices, GLOBAL),
    )
}

pub(crate) fn devices() -> Result<Vec<Device>, Error> {
    let system: AudioObjectID = get(
        kAudioObjectSystemObject as AudioObjectID,
        addr(kAudioHardwarePropertyDefaultOutputDevice, GLOBAL),
        "default output device",
    )
    .unwrap_or(0);
    let listed: Vec<(AudioObjectID, String)> = system_devices()
        .into_iter()
        .filter(|&d| !output_streams(d).is_empty())
        .filter_map(|d| Some((d, get_string(d, kAudioDevicePropertyDeviceUID)?)))
        .collect();
    let objects: Vec<AudioObjectID> = listed.iter().map(|&(d, _)| d).collect();
    let pid = std::process::id() as i32;
    let default = default_output(system, &objects, |d| hog_owner(d) == Some(pid));
    Ok(listed
        .into_iter()
        .map(|(d, id)| {
            let name = get_string(d, kAudioObjectPropertyName).unwrap_or_else(|| id.clone());
            Device {
                id,
                name,
                is_default: Some(d) == default,
            }
        })
        .collect())
}

/// The process that holds `device` in hog mode (`None`: no one, or it
/// cannot be read).
fn hog_owner(device: AudioObjectID) -> Option<i32> {
    get::<i32>(
        device,
        addr(kAudioDevicePropertyHogMode, GLOBAL),
        "get hog mode",
    )
    .ok()
    .filter(|&pid| pid != -1)
}

/// The output device to mark as the default among `listed`: the system's
/// default output where it is one of them, else the one this process holds
/// in hog mode (`ours`). While this process hogs the default device, the
/// system's default output can name none of the devices, and the device
/// the player plays on is the default it chose.
fn default_output(
    system: AudioObjectID,
    listed: &[AudioObjectID],
    ours: impl Fn(AudioObjectID) -> bool,
) -> Option<AudioObjectID> {
    if listed.contains(&system) {
        return Some(system);
    }
    listed.iter().copied().find(|&d| ours(d))
}

fn find(uid: &str) -> Result<AudioObjectID, Error> {
    system_devices()
        .into_iter()
        .find(|&d| {
            !output_streams(d).is_empty()
                && get_string(d, kAudioDevicePropertyDeviceUID).as_deref() == Some(uid)
        })
        .ok_or_else(|| Error::DeviceNotFound(uid.into()))
}

/// The sample format of an interleaved little-endian signed integer
/// description, if it is one of ours.
pub(super) fn sample_format(d: &AudioStreamBasicDescription) -> Option<SampleFormat> {
    let flags = d.mFormatFlags;
    if d.mFormatID != kAudioFormatLinearPCM
        || flags & kAudioFormatFlagIsFloat != 0
        || flags & kAudioFormatFlagIsBigEndian != 0
        || flags & kAudioFormatFlagIsNonInterleaved != 0
        || flags & kAudioFormatFlagIsSignedInteger == 0
        || d.mChannelsPerFrame == 0
    {
        return None;
    }
    let bytes = d.mBytesPerFrame / d.mChannelsPerFrame;
    match (d.mBitsPerChannel, bytes) {
        (16, 2) => Some(SampleFormat::S16),
        (24, 3) => Some(SampleFormat::S24Packed),
        (24, 4) if flags & kAudioFormatFlagIsAlignedHigh != 0 => Some(SampleFormat::S24In32),
        (24, 4) => Some(SampleFormat::S24In32Low),
        (32, 4) => Some(SampleFormat::S32),
        _ => None,
    }
}

pub(super) fn is_float32(d: &AudioStreamBasicDescription) -> bool {
    d.mFormatID == kAudioFormatLinearPCM
        && d.mFormatFlags & kAudioFormatFlagIsFloat != 0
        && d.mFormatFlags & (kAudioFormatFlagIsBigEndian | kAudioFormatFlagIsNonInterleaved) == 0
        && d.mBitsPerChannel == 32
}

/// The sample format a physical format carries: its integer layout, or
/// 24 bits for 32-bit float (see the module notes).
fn physical_sample(d: &AudioStreamBasicDescription) -> Option<SampleFormat> {
    sample_format(d).or_else(|| is_float32(d).then_some(SampleFormat::S24In32))
}

/// The rates a ranged description covers among the probed ones (and its
/// own rate, if it names one).
fn rates_of(r: &AudioStreamRangedDescription) -> Vec<u32> {
    let (lo, hi) = (r.mSampleRateRange.mMinimum, r.mSampleRateRange.mMaximum);
    let mut rates: Vec<u32> = PROBE_RATES
        .into_iter()
        .filter(|&rate| f64::from(rate) >= lo && f64::from(rate) <= hi)
        .collect();
    let own = r.mFormat.mSampleRate;
    if own > 0.0 && own.fract() == 0.0 && !rates.contains(&(own as u32)) {
        rates.push(own as u32);
    }
    rates
}

pub(crate) fn capabilities(uid: &str) -> Result<Capabilities, Error> {
    let device = find(uid)?;
    let mut caps = Capabilities::new();
    for stream in output_streams(device) {
        for r in get_vec::<AudioStreamRangedDescription>(
            stream,
            addr(kAudioStreamPropertyAvailablePhysicalFormats, GLOBAL),
        ) {
            if let Some(sample) = physical_sample(&r.mFormat) {
                for rate in rates_of(&r) {
                    caps.insert(StreamFormat::new(
                        rate,
                        r.mFormat.mChannelsPerFrame as u16,
                        sample,
                    ));
                }
            }
        }
    }
    Ok(caps)
}

/// `kAudioDevicePropertyVolumeScalar` ('volm'): a volume control's level,
/// 0 to 1, as the system's slider shows it.
const VOLUME_SCALAR: u32 = u32::from_be_bytes(*b"volm");
/// `kAudioDevicePropertyVolumeDecibels` ('vold'): its gain in dB.
const VOLUME_DB: u32 = u32::from_be_bytes(*b"vold");
/// `kAudioDevicePropertyMute` ('mute').
const MUTE: u32 = u32::from_be_bytes(*b"mute");

/// A control of the output: element 0 is the master, 1… the channels.
fn control(selector: u32, element: u32) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: OUTPUT,
        mElement: element,
    }
}

fn has(object: AudioObjectID, address: AudioObjectPropertyAddress) -> bool {
    // SAFETY: asks whether the object has the property.
    unsafe { AudioObjectHasProperty(object, NonNull::from(&address)) }
}

/// The output channels across the device's output streams.
fn output_channels(device: AudioObjectID) -> u32 {
    output_streams(device)
        .into_iter()
        .filter_map(|stream| {
            get::<AudioStreamBasicDescription>(
                stream,
                addr(kAudioStreamPropertyVirtualFormat, GLOBAL),
                "get virtual format",
            )
            .ok()
        })
        .map(|d| d.mChannelsPerFrame)
        .sum()
}

/// The volume controls this process can set: the master where there is
/// one, else each channel's.
fn volume_elements(device: AudioObjectID) -> Vec<u32> {
    let usable = |element: u32| {
        let address = control(VOLUME_SCALAR, element);
        has(device, address) && settable(device, address)
    };
    if usable(0) {
        return vec![0];
    }
    (1..=output_channels(device))
        .filter(|&e| usable(e))
        .collect()
}

/// The volume as `elements` have it (their mean level).
fn read_volume(device: AudioObjectID, elements: &[u32]) -> Option<Volume> {
    let levels: Vec<f32> = elements
        .iter()
        .filter_map(|&e| get::<f32>(device, control(VOLUME_SCALAR, e), "get volume").ok())
        .collect();
    if levels.is_empty() {
        return None;
    }
    let level = levels.iter().sum::<f32>() / levels.len() as f32;
    let db = get::<f32>(device, control(VOLUME_DB, elements[0]), "get volume dB").ok();
    let muted = (0..=output_channels(device))
        .find(|&e| has(device, control(MUTE, e)))
        .and_then(|e| get::<u32>(device, control(MUTE, e), "get mute").ok())
        .map(|m| m != 0);
    Some(Volume { level, db, muted })
}

pub(crate) fn volume(uid: &str) -> Result<Option<Volume>, Error> {
    let device = find(uid)?;
    Ok(read_volume(device, &volume_elements(device)))
}

pub(crate) fn set_volume(uid: &str, level: f32) -> Result<Volume, Error> {
    let device = find(uid)?;
    let elements = volume_elements(device);
    if elements.is_empty() {
        return Err(Error::Unsupported(
            "the device has no volume control of its own".into(),
        ));
    }
    for &e in &elements {
        set(device, control(VOLUME_SCALAR, e), level, "set volume")?;
    }
    if level > 0.0 {
        for e in 0..=output_channels(device) {
            let mute = control(MUTE, e);
            if has(device, mute)
                && settable(device, mute)
                && get::<u32>(device, mute, "get mute").is_ok_and(|m| m != 0)
            {
                set(device, mute, 0u32, "unmute")?;
            }
        }
    }
    read_volume(device, &elements).ok_or(Error::Unsupported(
        "the device's volume could not be read back".into(),
    ))
}

/// How the IOProc hands samples to the device.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Transport {
    /// The ring's bytes as they are.
    Direct,
    /// Converted to 32-bit float.
    Float32,
}

/// What the IOProc reads; owned by the driver, freed after the IOProc is
/// destroyed.
struct IoContext {
    shared: Arc<Shared>,
    transport: Transport,
    /// Index of our stream among the device's output streams (= the
    /// buffer index in the IOProc's buffer list).
    stream_index: usize,
    /// Device and stream latency, in frames.
    latency: f64,
    /// Device frames delivered since the last restart.
    delivered: AtomicU64,
    /// Scratch for the float conversion; only the IOProc touches it.
    scratch: UnsafeCell<Vec<u8>>,
    /// Closing: the IOProc delivers only silence.
    silence: AtomicBool,
    /// Device frames of that silence delivered.
    silent: AtomicU64,
    /// Device frames of silence to deliver before the device stops.
    settle: u64,
    /// Bytes of one device frame in the IOProc's buffer.
    device_frame_bytes: usize,
}

/// The longest a close waits for the device to take its silence.
const SETTLE_WAIT: Duration = Duration::from_millis(300);

// SAFETY: `scratch` is used only from the IOProc, which the HAL calls on
// one thread at a time; everything else is atomic or internally synchronised.
unsafe impl Sync for IoContext {}

impl IoContext {
    /// Fills one device buffer. Returns the device frames delivered, or
    /// `None` when the handle had the consumer suspended (the buffer is
    /// silence and nothing counts).
    fn fill(&self, out: &mut [u8]) -> Option<u64> {
        let format = self.shared.format;
        match self.transport {
            Transport::Direct => {
                self.shared.render_or_skip(out)?;
                Some((out.len() / format.frame_bytes()) as u64)
            }
            Transport::Float32 => {
                // SAFETY: see the Sync impl.
                let scratch = unsafe { &mut *self.scratch.get() };
                let ch = usize::from(format.channels);
                let frames = out.len() / (4 * ch);
                let per_chunk = scratch.len() / format.frame_bytes();
                let mut done = 0;
                let mut skipped = false;
                while done < frames {
                    let n = per_chunk.min(frames - done);
                    let packed = &mut scratch[..n * format.frame_bytes()];
                    skipped |= self.shared.render_or_skip(packed).is_none();
                    let floats = &mut out[done * 4 * ch..(done + n) * 4 * ch];
                    for (sample, f) in packed
                        .chunks_exact(format.sample.bytes())
                        .zip(floats.chunks_exact_mut(4))
                    {
                        let v = unpack_sample(sample, format.sample) as f32 / 8_388_608.0;
                        f.copy_from_slice(&v.to_le_bytes());
                    }
                    done += n;
                }
                (!skipped).then_some(frames as u64)
            }
        }
    }
}

unsafe extern "C-unwind" fn io_proc(
    _device: AudioObjectID,
    now: NonNull<AudioTimeStamp>,
    _input: NonNull<AudioBufferList>,
    _input_time: NonNull<AudioTimeStamp>,
    output: NonNull<AudioBufferList>,
    output_time: NonNull<AudioTimeStamp>,
    client: *mut c_void,
) -> i32 {
    let ctx = &*(client as *const IoContext);
    let list = output.as_ptr();
    let count = (*list).mNumberBuffers as usize;
    let buffers = std::ptr::addr_of_mut!((*list).mBuffers) as *mut AudioBuffer;
    if ctx.silence.load(Ordering::Acquire) {
        // Closing: silence, counted, so the device holds none of ours.
        let mut frames = 0;
        for i in 0..count {
            let b = &mut *buffers.add(i);
            if b.mData.is_null() {
                continue;
            }
            std::slice::from_raw_parts_mut(b.mData as *mut u8, b.mDataByteSize as usize).fill(0);
            if i == ctx.stream_index {
                frames = b.mDataByteSize as usize / ctx.device_frame_bytes.max(1);
            }
        }
        ctx.silent.fetch_add(frames as u64, Ordering::Release);
        return 0;
    }
    let mut frames = Some(0);
    for i in 0..count {
        let b = &mut *buffers.add(i);
        if b.mData.is_null() {
            continue;
        }
        let bytes = std::slice::from_raw_parts_mut(b.mData as *mut u8, b.mDataByteSize as usize);
        if i == ctx.stream_index {
            frames = ctx.fill(bytes);
        } else {
            bytes.fill(0);
        }
    }
    let Some(frames) = frames else {
        return 0;
    };
    let delivered = ctx.delivered.load(Ordering::Relaxed);
    // The first frame of this buffer reaches the hardware at the output
    // time and the DAC `latency` frames later.
    let ahead = (*output_time.as_ptr()).mSampleTime - (*now.as_ptr()).mSampleTime;
    let played = delivered as f64 - ahead.max(0.0) - ctx.latency;
    ctx.shared
        .report(played.max(0.0) as u64, Instant::now(), true);
    ctx.delivered.store(delivered + frames, Ordering::Relaxed);
    0
}

struct Opened {
    device: AudioObjectID,
    stream: AudioObjectID,
    took_hog: bool,
    mixing: Option<u32>,
    original: Option<AudioStreamBasicDescription>,
}

impl Opened {
    /// Undoes what open changed, in reverse.
    fn restore(&self) {
        if let Some(original) = self.original {
            let _ = set(
                self.stream,
                addr(kAudioStreamPropertyPhysicalFormat, GLOBAL),
                original,
                "restore format",
            );
        }
        if let Some(mixing) = self.mixing {
            let _ = set(
                self.device,
                addr(kAudioDevicePropertySupportsMixing, GLOBAL),
                mixing,
                "restore mixing",
            );
        }
        if self.took_hog {
            let _ = set(
                self.device,
                addr(kAudioDevicePropertyHogMode, GLOBAL),
                -1i32,
                "release hog mode",
            );
        }
    }
}

fn same_format(a: &AudioStreamBasicDescription, b: &AudioStreamBasicDescription) -> bool {
    a.mSampleRate == b.mSampleRate
        && a.mFormatID == b.mFormatID
        && a.mBitsPerChannel == b.mBitsPerChannel
        && a.mBytesPerFrame == b.mBytesPerFrame
        && a.mChannelsPerFrame == b.mChannelsPerFrame
        && sample_format(a) == sample_format(b)
        && is_float32(a) == is_float32(b)
}

/// Picks (stream index, stream, physical description) for the request:
/// the first format the request prefers that a stream takes, as integer
/// samples where a stream lists them, else as float.
fn choose(
    device: AudioObjectID,
    request: &OpenRequest,
) -> Option<(
    usize,
    AudioObjectID,
    AudioStreamBasicDescription,
    StreamFormat,
)> {
    let streams = output_streams(device);
    let lists: Vec<Vec<AudioStreamRangedDescription>> = streams
        .iter()
        .map(|&stream| {
            get_vec(
                stream,
                addr(kAudioStreamPropertyAvailablePhysicalFormats, GLOBAL),
            )
        })
        .collect();
    let matches = |sample: SampleFormat, float: bool, d: &AudioStreamBasicDescription| {
        let carries = if float {
            is_float32(d) && physical_sample(d) == Some(sample)
        } else {
            sample_format(d) == Some(sample)
        };
        carries && d.mChannelsPerFrame == u32::from(request.channels)
    };
    for float in [false, true] {
        for sample in candidates(&request.formats) {
            for (index, (&stream, list)) in streams.iter().zip(&lists).enumerate() {
                for r in list {
                    if !matches(sample, float, &r.mFormat) || !rates_of(r).contains(&request.rate) {
                        continue;
                    }
                    let mut d = r.mFormat;
                    d.mSampleRate = f64::from(request.rate);
                    return Some((
                        index,
                        stream,
                        d,
                        StreamFormat::new(request.rate, request.channels, sample),
                    ));
                }
            }
        }
    }
    None
}

fn configure(
    uid: &str,
    request: &OpenRequest,
) -> Result<(Opened, IoContext, StreamInfo, u64), Error> {
    let device = find(uid)?;
    if get::<u32>(
        device,
        addr(kAudioDevicePropertyDeviceIsAlive, GLOBAL),
        "alive",
    )
    .unwrap_or(1)
        == 0
    {
        return Err(Error::DeviceLost);
    }
    let (stream_index, stream, physical, format) =
        choose(device, request).ok_or(Error::FormatNotAccepted {
            rate: request.rate,
            channels: request.channels,
        })?;
    let mut opened = Opened {
        device,
        stream,
        took_hog: false,
        mixing: None,
        original: None,
    };
    let result = (|| {
        // 1. Hog mode. Setting the property toggles it for the calling
        // process (the value passed is ignored): it takes a free device and
        // releases one we hold, and leaves another owner's alone.
        let pid = std::process::id() as i32;
        let hog = addr(kAudioDevicePropertyHogMode, GLOBAL);
        let owner: i32 = get(device, hog, "get hog mode")?;
        if owner != -1 && owner != pid {
            return Err(Error::DeviceBusy);
        }
        if owner != pid {
            set(device, hog, pid, "set hog mode")?;
            if get::<i32>(device, hog, "get hog mode")? != pid {
                return Err(Error::DeviceBusy);
            }
            opened.took_hog = true;
        }
        // 2. Mixing off.
        let mixing = addr(kAudioDevicePropertySupportsMixing, GLOBAL);
        if settable(device, mixing) {
            if let Ok(old) = get::<u32>(device, mixing, "get mixing") {
                if old != 0 {
                    set(device, mixing, 0u32, "set mixing")?;
                    opened.mixing = Some(old);
                }
            }
        }
        // 3. Physical format and rate.
        let pf = addr(kAudioStreamPropertyPhysicalFormat, GLOBAL);
        let original: AudioStreamBasicDescription = get(stream, pf, "get physical format")?;
        if !same_format(&original, &physical) {
            opened.original = Some(original);
            set(stream, pf, physical, "set physical format")?;
            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                let now: AudioStreamBasicDescription = get(stream, pf, "get physical format")?;
                if same_format(&now, &physical) {
                    break;
                }
                if Instant::now() >= deadline {
                    return Err(Error::Timeout(
                        "the device did not switch to the physical format",
                    ));
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
        // 4. The IOProc's format.
        let virtual_format: AudioStreamBasicDescription = get(
            stream,
            addr(kAudioStreamPropertyVirtualFormat, GLOBAL),
            "get virtual format",
        )?;
        let transport = if virtual_format.mChannelsPerFrame != u32::from(format.channels) {
            return Err(Error::Unsupported(format!(
                "the IO format has {} channels, the stream {}",
                virtual_format.mChannelsPerFrame, format.channels
            )));
        } else if sample_format(&virtual_format) == Some(format.sample) {
            Transport::Direct
        } else if is_float32(&virtual_format) {
            Transport::Float32
        } else {
            return Err(Error::Unsupported(
                "the device's IO format is neither the physical format nor float".into(),
            ));
        };
        // 5. One period of IO buffer.
        let period = format.duration_to_frames(request.period).max(16) as u32;
        let bfs = addr(kAudioDevicePropertyBufferFrameSize, GLOBAL);
        let _ = set(device, bfs, period, "set buffer frame size");
        let buffer = get::<u32>(device, bfs, "get buffer frame size").unwrap_or(period);
        let device_latency =
            get::<u32>(device, addr(kAudioDevicePropertyLatency, OUTPUT), "latency").unwrap_or(0);
        let stream_latency = get::<u32>(
            stream,
            addr(kAudioStreamPropertyLatency, GLOBAL),
            "stream latency",
        )
        .unwrap_or(0);
        let safety = get::<u32>(
            device,
            addr(kAudioDevicePropertySafetyOffset, OUTPUT),
            "safety offset",
        )
        .unwrap_or(0);
        let latency = u64::from(device_latency) + u64::from(stream_latency);
        let shared = Arc::new(Shared::new(
            format,
            request.ring_frames(&format, u64::from(buffer)),
        ));
        let info = StreamInfo {
            format,
            backend: BACKEND,
            exclusive: true,
            device_buffer_frames: u64::from(buffer),
            device_latency: format.frames_to_duration(latency + u64::from(safety)),
            ring_frames: shared.ring.capacity(),
            realtime: true,
            float_io: transport == Transport::Float32,
        };
        let ctx = IoContext {
            shared,
            transport,
            stream_index,
            latency: latency as f64,
            delivered: AtomicU64::new(0),
            scratch: UnsafeCell::new(vec![0; 4096 * format.frame_bytes()]),
            silence: AtomicBool::new(false),
            silent: AtomicU64::new(0),
            settle: 2 * u64::from(buffer)
                + latency
                + u64::from(safety)
                + u64::from(format.rate / 50),
            device_frame_bytes: match transport {
                Transport::Direct => format.frame_bytes(),
                Transport::Float32 => 4 * usize::from(format.channels),
            },
        };
        Ok((ctx, info, u64::from(buffer)))
    })();
    match result {
        Ok((ctx, info, buffer)) => Ok((opened, ctx, info, buffer)),
        Err(e) => {
            opened.restore();
            Err(e)
        }
    }
}

struct HalDriver {
    opened: Opened,
    proc_id: AudioDeviceIOProcID,
    ctx: *mut IoContext,
    running: bool,
    closed: bool,
}

// SAFETY: the HAL calls used here are thread-safe; `ctx` is only freed in
// `close`, after the IOProc is destroyed.
unsafe impl Send for HalDriver {}

impl HalDriver {
    fn ctx(&self) -> &IoContext {
        // SAFETY: valid until `close`.
        unsafe { &*self.ctx }
    }

    fn stop(&mut self) -> Result<(), Error> {
        if self.running {
            // SAFETY: stops our IOProc on our device.
            let status = unsafe { AudioDeviceStop(self.opened.device, self.proc_id) };
            self.running = false;
            if status != 0 {
                return Err(status_error("AudioDeviceStop", status));
            }
        }
        Ok(())
    }
}

impl Driver for HalDriver {
    fn start(&mut self) -> Result<(), Error> {
        if !self.running {
            // SAFETY: starts our IOProc on our device.
            let status = unsafe { AudioDeviceStart(self.opened.device, self.proc_id) };
            if status != 0 {
                return Err(status_error("AudioDeviceStart", status));
            }
            self.running = true;
        }
        Ok(())
    }

    fn pause(&mut self) -> Result<(), Error> {
        self.stop()?;
        // What the HAL had taken is not played after a stop; the ring keeps
        // the rest, and the delivered count restarts with the timeline.
        let ctx = self.ctx();
        ctx.shared.suspend();
        ctx.shared.skip_device_queue();
        ctx.delivered.store(0, Ordering::Relaxed);
        ctx.shared.unsuspend();
        Ok(())
    }

    fn flush(&mut self) -> Result<(), Error> {
        self.stop()?;
        let ctx = self.ctx();
        ctx.shared.discard_all();
        ctx.delivered.store(0, Ordering::Relaxed);
        Ok(())
    }

    fn close(&mut self) -> Result<(), Error> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        // Silence through the device's buffers before it stops and its
        // format changes back, also after a pause (the device stopped with
        // our samples in it): it runs again for that, silent.
        self.ctx().silence.store(true, Ordering::Release);
        if self.start().is_ok() {
            let ctx = self.ctx();
            let deadline = Instant::now() + SETTLE_WAIT;
            while ctx.silent.load(Ordering::Acquire) < ctx.settle && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(2));
            }
        }
        let stopped = self.stop();
        self.ctx().shared.freeze();
        // SAFETY: destroys our IOProc; after this the HAL no longer calls it,
        // so the context can be freed.
        unsafe {
            AudioDeviceDestroyIOProcID(self.opened.device, self.proc_id);
            drop(Box::from_raw(self.ctx));
        }
        self.opened.restore();
        stopped
    }
}

impl Drop for HalDriver {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// Opens the device with UID `uid` in hog mode.
pub(crate) fn open(uid: &str, request: &OpenRequest) -> Result<Core, Error> {
    request.validate()?;
    let (opened, ctx, info, _buffer) = configure(uid, request)?;
    let shared = ctx.shared.clone();
    let ctx = Box::into_raw(Box::new(ctx));
    let mut proc_id: AudioDeviceIOProcID = None;
    // SAFETY: registers `io_proc` with a context that lives until close.
    let status = unsafe {
        AudioDeviceCreateIOProcID(
            opened.device,
            Some(io_proc),
            ctx.cast(),
            NonNull::from(&mut proc_id),
        )
    };
    if status != 0 || proc_id.is_none() {
        // SAFETY: the IOProc was not registered, so nothing else holds ctx.
        unsafe { drop(Box::from_raw(ctx)) };
        opened.restore();
        return Err(status_error("AudioDeviceCreateIOProcID", status));
    }
    let driver = HalDriver {
        opened,
        proc_id,
        ctx,
        running: false,
        closed: false,
    };
    Ok(Core::new(shared, Box::new(driver), info, request))
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_core_audio_types::kAudioFormatFlagIsPacked;

    fn asbd(bits: u32, bytes: u32, flags: u32) -> AudioStreamBasicDescription {
        AudioStreamBasicDescription {
            mSampleRate: 96_000.0,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: flags,
            mBytesPerPacket: bytes * 2,
            mFramesPerPacket: 1,
            mBytesPerFrame: bytes * 2,
            mChannelsPerFrame: 2,
            mBitsPerChannel: bits,
            mReserved: 0,
        }
    }

    #[test]
    fn physical_formats_map_to_sample_formats() {
        let int = kAudioFormatFlagIsSignedInteger;
        assert_eq!(
            sample_format(&asbd(16, 2, int | kAudioFormatFlagIsPacked)),
            Some(SampleFormat::S16)
        );
        assert_eq!(
            sample_format(&asbd(24, 3, int | kAudioFormatFlagIsPacked)),
            Some(SampleFormat::S24Packed)
        );
        assert_eq!(
            sample_format(&asbd(24, 4, int | kAudioFormatFlagIsAlignedHigh)),
            Some(SampleFormat::S24In32)
        );
        assert_eq!(
            sample_format(&asbd(24, 4, int)),
            Some(SampleFormat::S24In32Low)
        );
        assert_eq!(
            sample_format(&asbd(32, 4, int | kAudioFormatFlagIsPacked)),
            Some(SampleFormat::S32)
        );
        assert_eq!(sample_format(&asbd(32, 4, kAudioFormatFlagIsFloat)), None);
        assert_eq!(
            sample_format(&asbd(24, 3, int | kAudioFormatFlagIsBigEndian)),
            None
        );
        assert!(is_float32(&asbd(
            32,
            4,
            kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked
        )));
    }

    #[test]
    fn float_physical_formats_carry_24_bits() {
        let float = asbd(32, 4, kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked);
        assert_eq!(physical_sample(&float), Some(SampleFormat::S24In32));
        let int = asbd(
            32,
            4,
            kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked,
        );
        assert_eq!(physical_sample(&int), Some(SampleFormat::S32));
        assert!(!same_format(
            &float,
            &asbd(
                24,
                4,
                kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsAlignedHigh
            )
        ));
        let double = asbd(64, 8, kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked);
        assert_eq!(physical_sample(&double), None);
    }

    #[test]
    fn the_default_is_the_systems_else_the_device_we_hog() {
        let listed = [40, 52, 77];
        // The system's default output, whatever we hog.
        assert_eq!(default_output(52, &listed, |d| d == 77), Some(52));
        // None while we hog the default device: the hogged one.
        assert_eq!(default_output(0, &listed, |d| d == 77), Some(77));
        assert_eq!(default_output(99, &listed, |d| d == 40), Some(40));
        // Nothing hogged and no default: none.
        assert_eq!(default_output(0, &listed, |_| false), None);
    }
}
