//! macOS: Core Audio HAL input, our own IOProc.
//!
//! Inputs are the devices with input streams; ids are their UIDs. A stream
//! opens on the device's first input stream at its virtual format (32-bit
//! float as a rule, integer when mixing is off), interleaved or one buffer
//! per channel, at the device's nominal rate. The IOProc mixes each
//! buffer down into the microphone ring.
//!
//! The device is shared, as the output's is: nothing about it is changed
//! but this process's IO buffer, set to about 5 ms (on an audio interface
//! that has both, the output of ours on it then runs with that buffer
//! too). A listener follows the device: when its nominal rate changes (in
//! Audio MIDI Setup) it tells the tap, which resamples from the new one;
//! when the device is gone (a USB microphone unplugged: it is no longer
//! alive, or no longer an object the HAL knows) or its stream's layout is
//! no longer the one the IOProc reads, the capture fails with
//! [`Error::DeviceLost`] and the IOProc queues nothing more. A query that
//! fails otherwise tells nothing, and the change is let pass. A device
//! another process holds in hog mode cannot be opened.
//!
//! The process needs the system's permission to use the microphone; without
//! it the HAL delivers silence.

use super::coreaudio_hal::{
    addr, get, get_string, get_vec, hogged, set, status_error, system_devices, Listener, GLOBAL,
};
use crate::device::Device;
use crate::error::Error;
use crate::input::{
    Input, InputInfo, MicClock, MicShared, Mono, Opened, Producer, Wire, CAPTURE_PERIOD,
};
use objc2_core_audio::{
    kAudioDevicePropertyBufferFrameSize, kAudioDevicePropertyDeviceIsAlive,
    kAudioDevicePropertyDeviceUID, kAudioDevicePropertyNominalSampleRate,
    kAudioDevicePropertyStreams, kAudioHardwareBadDeviceError, kAudioHardwareBadObjectError,
    kAudioHardwarePropertyDefaultInputDevice, kAudioObjectPropertyName,
    kAudioObjectPropertyScopeInput, kAudioObjectSystemObject, kAudioStreamPropertyVirtualFormat,
    AudioDeviceCreateIOProcID, AudioDeviceDestroyIOProcID, AudioDeviceIOProcID, AudioDeviceStart,
    AudioDeviceStop, AudioObjectID, AudioObjectPropertyAddress,
};
use objc2_core_audio_types::{
    kAudioFormatFlagIsAlignedHigh, kAudioFormatFlagIsBigEndian, kAudioFormatFlagIsFloat,
    kAudioFormatFlagIsNonInterleaved, kAudioFormatFlagIsSignedInteger, kAudioFormatLinearPCM,
    AudioBuffer, AudioBufferList, AudioStreamBasicDescription, AudioTimeStamp,
};
use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

pub(crate) const INPUT_BACKEND: &str = "Core Audio (shared)";

const INPUT: u32 = kAudioObjectPropertyScopeInput;

/// The most channel buffers of a non-interleaved stream that are read.
const MAX_PLANES: usize = 8;

fn input_streams(device: AudioObjectID) -> Vec<AudioObjectID> {
    get_vec(device, addr(kAudioDevicePropertyStreams, INPUT))
}

pub(crate) fn input_devices() -> Result<Vec<Device>, Error> {
    let default: AudioObjectID = get(
        kAudioObjectSystemObject as AudioObjectID,
        addr(kAudioHardwarePropertyDefaultInputDevice, GLOBAL),
        "default input device",
    )
    .unwrap_or(0);
    Ok(system_devices()
        .into_iter()
        .filter(|&d| !input_streams(d).is_empty())
        .filter_map(|d| {
            let id = get_string(d, kAudioDevicePropertyDeviceUID)?;
            let name = get_string(d, kAudioObjectPropertyName).unwrap_or_else(|| id.clone());
            Some(Device {
                id,
                name,
                is_default: d == default,
            })
        })
        .collect())
}

fn find(uid: &str) -> Result<AudioObjectID, Error> {
    system_devices()
        .into_iter()
        .find(|&d| {
            !input_streams(d).is_empty()
                && get_string(d, kAudioDevicePropertyDeviceUID).as_deref() == Some(uid)
        })
        .ok_or_else(|| Error::DeviceNotFound(uid.into()))
}

/// The layout of an input stream's virtual format, and whether it comes
/// one buffer per channel.
fn input_wire(d: &AudioStreamBasicDescription) -> Option<(Wire, bool)> {
    let flags = d.mFormatFlags;
    if d.mFormatID != kAudioFormatLinearPCM
        || flags & kAudioFormatFlagIsBigEndian != 0
        || d.mChannelsPerFrame == 0
    {
        return None;
    }
    let planar = flags & kAudioFormatFlagIsNonInterleaved != 0;
    let bytes = if planar {
        d.mBytesPerFrame
    } else {
        d.mBytesPerFrame / d.mChannelsPerFrame
    };
    let wire = if flags & kAudioFormatFlagIsFloat != 0 {
        (d.mBitsPerChannel == 32 && bytes == 4).then_some(Wire::F32)?
    } else if flags & kAudioFormatFlagIsSignedInteger != 0 {
        match (d.mBitsPerChannel, bytes) {
            (16, 2) => Wire::I16,
            (24, 3) => Wire::I24,
            (24, 4) if flags & kAudioFormatFlagIsAlignedHigh != 0 => Wire::I32,
            (24, 4) => Wire::I24Low,
            (32, 4) => Wire::I32,
            _ => return None,
        }
    } else {
        return None;
    };
    Some((wire, planar))
}

/// What the IOProc and the listener read; freed after both are gone.
struct Ctx {
    /// Only the IOProc touches it (the HAL calls it on one thread at a time).
    producer: UnsafeCell<Producer>,
    shared: Arc<MicShared>,
    device: AudioObjectID,
    stream: AudioObjectID,
    wire: Wire,
    planar: bool,
    /// Held through [`Ctx::follow`]: the opener's look and the listener's
    /// come one after the other, so a rate read earlier is never stored
    /// over one read later.
    following: Mutex<()>,
}

// SAFETY: see `producer`; everything else is immutable or synchronised.
unsafe impl Sync for Ctx {}

/// What a look at the device tells.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Look {
    /// Still there, in the layout the IOProc reads.
    Fine,
    /// Gone, or in another layout: the capture is lost.
    Lost,
    /// A query failed for another reason: nothing is known.
    Unknown,
}

/// Whether a failed query says its object is gone.
fn gone(error: &Error) -> bool {
    matches!(error, Error::Backend { code, .. }
        if *code == i64::from(kAudioHardwareBadObjectError)
            || *code == i64::from(kAudioHardwareBadDeviceError))
}

impl Ctx {
    fn look(&self) -> Look {
        let alive = get::<u32>(
            self.device,
            addr(kAudioDevicePropertyDeviceIsAlive, GLOBAL),
            "alive",
        );
        match alive {
            Ok(0) => return Look::Lost,
            Ok(_) => {}
            Err(e) if gone(&e) => return Look::Lost,
            Err(_) => return Look::Unknown,
        }
        let format = get::<AudioStreamBasicDescription>(
            self.stream,
            addr(kAudioStreamPropertyVirtualFormat, GLOBAL),
            "input virtual format",
        );
        match format {
            Ok(f) if input_wire(&f) == Some((self.wire, self.planar)) => Look::Fine,
            Ok(_) => Look::Lost,
            Err(_) => Look::Unknown,
        }
    }

    /// Follows the device: a capture whose device is gone, or whose stream
    /// no longer has the layout the IOProc reads, fails; a new nominal rate
    /// is told to the tap; when the look tells nothing, nothing changes.
    fn follow(&self) -> Look {
        let _one_at_a_time = self.following.lock().unwrap_or_else(|e| e.into_inner());
        let look = self.look();
        match look {
            Look::Lost => self.shared.fail(Error::DeviceLost),
            Look::Fine => {
                if let Ok(rate) = get::<f64>(
                    self.device,
                    addr(kAudioDevicePropertyNominalSampleRate, GLOBAL),
                    "nominal rate",
                ) {
                    if rate >= 1.0 {
                        self.shared.set_rate(rate.round() as u32);
                    }
                }
            }
            Look::Unknown => {}
        }
        look
    }
}

unsafe extern "C-unwind" fn input_proc(
    _device: AudioObjectID,
    _now: NonNull<AudioTimeStamp>,
    input: NonNull<AudioBufferList>,
    _input_time: NonNull<AudioTimeStamp>,
    _output: NonNull<AudioBufferList>,
    _output_time: NonNull<AudioTimeStamp>,
    client: *mut c_void,
) -> i32 {
    let ctx = &*(client as *const Ctx);
    if ctx.shared.failed() {
        return 0;
    }
    let producer = &mut *ctx.producer.get();
    let list = input.as_ptr();
    let count = (*list).mNumberBuffers as usize;
    let buffers = std::ptr::addr_of!((*list).mBuffers) as *const AudioBuffer;
    let sb = ctx.wire.bytes();
    if ctx.planar {
        let n = count.min(MAX_PLANES);
        let mut planes: [&[u8]; MAX_PLANES] = [&[]; MAX_PLANES];
        for (i, plane) in planes.iter_mut().enumerate().take(n) {
            let b = &*buffers.add(i);
            if !b.mData.is_null() {
                *plane = std::slice::from_raw_parts(b.mData as *const u8, b.mDataByteSize as usize);
            }
        }
        if n > 0 {
            let frames = planes[0].len() / sb;
            producer.push_planar(&planes[..n], ctx.wire, frames);
        }
    } else if count > 0 {
        let b = &*buffers;
        let channels = (b.mNumberChannels as usize).max(1);
        let frames = b.mDataByteSize as usize / (sb * channels);
        if b.mData.is_null() {
            producer.push_silence(frames);
        } else {
            let bytes = std::slice::from_raw_parts(b.mData as *const u8, b.mDataByteSize as usize);
            producer.push(bytes, ctx.wire, channels, frames);
        }
    }
    0
}

/// A property the capture depends on changed ([`Ctx::follow`]).
unsafe extern "C-unwind" fn changed(
    _object: AudioObjectID,
    _count: u32,
    _addresses: NonNull<AudioObjectPropertyAddress>,
    client: *mut c_void,
) -> i32 {
    (*(client as *const Ctx)).follow();
    0
}

/// The running capture: stopping it is dropping it.
struct Running {
    device: AudioObjectID,
    proc_id: AudioDeviceIOProcID,
    ctx: *mut Ctx,
    /// Removed before the context is freed.
    listener: Option<Listener>,
}

// SAFETY: the HAL calls used here are thread-safe; `ctx` is freed only in
// drop, after the IOProc and the listener are gone.
unsafe impl Send for Running {}

impl Drop for Running {
    fn drop(&mut self) {
        self.listener = None;
        // SAFETY: undoes what `open_input` registered, then frees the
        // context nothing calls with any more.
        unsafe {
            AudioDeviceStop(self.device, self.proc_id);
            AudioDeviceDestroyIOProcID(self.device, self.proc_id);
            drop(Box::from_raw(self.ctx));
        }
    }
}

pub(crate) fn open_input(uid: &str, mono: Mono) -> Result<Input, Error> {
    let device = find(uid)?;
    let stream = *input_streams(device)
        .first()
        .ok_or_else(|| Error::DeviceNotFound(uid.into()))?;
    let format: AudioStreamBasicDescription = get(
        stream,
        addr(kAudioStreamPropertyVirtualFormat, GLOBAL),
        "input virtual format",
    )?;
    let (wire, planar) = input_wire(&format).ok_or_else(|| {
        Error::Unsupported("the input's format is not PCM or 32-bit float".into())
    })?;
    let rate = get::<f64>(
        device,
        addr(kAudioDevicePropertyNominalSampleRate, GLOBAL),
        "nominal rate",
    )
    .unwrap_or(format.mSampleRate)
    .round() as u32;
    if rate == 0 {
        return Err(Error::Unsupported("the input reports no rate".into()));
    }
    // A device in another process's hog mode is not ours to use.
    if hogged(device) {
        return Err(Error::DeviceBusy);
    }
    let bfs = addr(kAudioDevicePropertyBufferFrameSize, GLOBAL);
    let want = (u64::from(rate) * CAPTURE_PERIOD.as_micros() as u64 / 1_000_000).max(16) as u32;
    let _ = set(device, bfs, want, "set buffer frame size");
    let period = get::<u32>(device, bfs, "get buffer frame size").unwrap_or(want);
    let (producer, opened) = Opened::new(rate, period, mono, MicClock::system());
    let ctx = Box::into_raw(Box::new(Ctx {
        shared: Arc::clone(producer.shared()),
        producer: UnsafeCell::new(producer),
        device,
        stream,
        wire,
        planar,
        following: Mutex::new(()),
    }));
    let mut proc_id: AudioDeviceIOProcID = None;
    // SAFETY: registers `input_proc` with a context that lives until the
    // `Running` guard drops.
    let status = unsafe {
        AudioDeviceCreateIOProcID(
            device,
            Some(input_proc),
            ctx.cast(),
            NonNull::from(&mut proc_id),
        )
    };
    if status != 0 || proc_id.is_none() {
        // SAFETY: nothing was registered with the context.
        unsafe { drop(Box::from_raw(ctx)) };
        return Err(status_error("AudioDeviceCreateIOProcID", status));
    }
    let listener = Listener::add(
        &[
            (device, addr(kAudioDevicePropertyNominalSampleRate, GLOBAL)),
            (device, addr(kAudioDevicePropertyDeviceIsAlive, GLOBAL)),
            (stream, addr(kAudioStreamPropertyVirtualFormat, GLOBAL)),
        ],
        Some(changed),
        ctx.cast(),
    );
    let running = Running {
        device,
        proc_id,
        ctx,
        listener: Some(listener),
    };
    // SAFETY: starts our IOProc on the device.
    let status = unsafe { AudioDeviceStart(device, proc_id) };
    if status != 0 {
        return Err(status_error("AudioDeviceStart", status));
    }
    // What changed before the listener was there: a device lost already
    // is not opened (`running`, dropped, stops it).
    // SAFETY: the context lives until `running` drops.
    if unsafe { (*ctx).follow() } == Look::Lost {
        return Err(Error::DeviceLost);
    }
    let info = InputInfo {
        id: uid.to_owned(),
        name: get_string(device, kAudioObjectPropertyName).unwrap_or_else(|| uid.to_owned()),
        backend: INPUT_BACKEND,
        rate,
        channels: format.mChannelsPerFrame.min(u32::from(u16::MAX)) as u16,
        period_frames: period,
        mono,
    };
    Ok(Input::new(info, opened, Box::new(running)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_core_audio_types::kAudioFormatFlagIsPacked;

    fn asbd(
        bits: u32,
        bytes_per_frame: u32,
        channels: u32,
        flags: u32,
    ) -> AudioStreamBasicDescription {
        AudioStreamBasicDescription {
            mSampleRate: 48_000.0,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: flags,
            mBytesPerPacket: bytes_per_frame,
            mFramesPerPacket: 1,
            mBytesPerFrame: bytes_per_frame,
            mChannelsPerFrame: channels,
            mBitsPerChannel: bits,
            mReserved: 0,
        }
    }

    #[test]
    fn input_formats_map_to_wires() {
        let float = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
        let int = kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked;
        assert_eq!(input_wire(&asbd(32, 8, 2, float)), Some((Wire::F32, false)));
        assert_eq!(
            input_wire(&asbd(32, 4, 2, float | kAudioFormatFlagIsNonInterleaved)),
            Some((Wire::F32, true))
        );
        assert_eq!(input_wire(&asbd(16, 2, 1, int)), Some((Wire::I16, false)));
        assert_eq!(input_wire(&asbd(24, 6, 2, int)), Some((Wire::I24, false)));
        assert_eq!(
            input_wire(&asbd(
                24,
                8,
                2,
                kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsAlignedHigh
            )),
            Some((Wire::I32, false))
        );
        assert_eq!(
            input_wire(&asbd(24, 8, 2, kAudioFormatFlagIsSignedInteger)),
            Some((Wire::I24Low, false))
        );
        assert_eq!(input_wire(&asbd(64, 16, 2, float)), None);
        assert_eq!(
            input_wire(&asbd(16, 4, 2, int | kAudioFormatFlagIsBigEndian)),
            None
        );
    }
}
