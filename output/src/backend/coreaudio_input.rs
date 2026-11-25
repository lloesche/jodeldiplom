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
//! too). When its nominal rate changes (in Audio MIDI Setup), a listener
//! tells the tap, which resamples from the new one. A device another
//! process holds in hog mode cannot be opened.
//!
//! The process needs the system's permission to use the microphone; without
//! it the HAL delivers silence.

use super::coreaudio_hal::{
    addr, get, get_string, get_vec, hogged, set, status_error, system_devices, Listener, GLOBAL,
};
use crate::device::Device;
use crate::error::Error;
use crate::input::{Input, InputInfo, MicShared, Mono, Producer, Wire, CAPTURE_PERIOD};
use objc2_core_audio::{
    kAudioDevicePropertyBufferFrameSize, kAudioDevicePropertyDeviceUID,
    kAudioDevicePropertyNominalSampleRate, kAudioDevicePropertyStreams,
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
use std::sync::Arc;

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

/// What the IOProc and the rate listener read; freed after both are gone.
struct Ctx {
    /// Only the IOProc touches it (the HAL calls it on one thread at a time).
    producer: UnsafeCell<Producer>,
    shared: Arc<MicShared>,
    wire: Wire,
    planar: bool,
    channels: usize,
}

// SAFETY: see `producer`; everything else is immutable or synchronised.
unsafe impl Sync for Ctx {}

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
    let producer = &mut *ctx.producer.get();
    let list = input.as_ptr();
    let count = (*list).mNumberBuffers as usize;
    let buffers = std::ptr::addr_of!((*list).mBuffers) as *const AudioBuffer;
    let sb = ctx.wire.bytes();
    if ctx.planar {
        let n = ctx.channels.min(count).min(MAX_PLANES);
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

unsafe extern "C-unwind" fn rate_changed(
    device: AudioObjectID,
    _count: u32,
    _addresses: NonNull<AudioObjectPropertyAddress>,
    client: *mut c_void,
) -> i32 {
    let ctx = &*(client as *const Ctx);
    if let Ok(rate) = get::<f64>(
        device,
        addr(kAudioDevicePropertyNominalSampleRate, GLOBAL),
        "nominal rate",
    ) {
        if rate >= 1.0 {
            ctx.shared.set_rate(rate.round() as u32);
        }
    }
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
    let shared = MicShared::new(rate, period, false);
    let channels = format.mChannelsPerFrame as usize;
    let ctx = Box::into_raw(Box::new(Ctx {
        producer: UnsafeCell::new(Producer::new(shared.clone(), mono)),
        shared: shared.clone(),
        wire,
        planar,
        channels,
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
        &[(device, addr(kAudioDevicePropertyNominalSampleRate, GLOBAL))],
        Some(rate_changed),
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
    let info = InputInfo {
        id: uid.to_owned(),
        name: get_string(device, kAudioObjectPropertyName).unwrap_or_else(|| uid.to_owned()),
        backend: INPUT_BACKEND,
        rate,
        channels: format.mChannelsPerFrame.min(u32::from(u16::MAX)) as u16,
        period_frames: period,
        mono,
    };
    Ok(Input::new(info, shared, Box::new(running)))
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
