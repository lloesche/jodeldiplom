//! macOS: the Core Audio HAL directly, shared, with our own IOProc.
//!
//! Device ids are Core Audio device UIDs; names are the devices' names.
//! A device is shared with every other application: the HAL mixes our
//! IOProc's buffers with theirs. Nothing about the device is changed: no
//! hog mode, mixing left on, its physical format and nominal rate as the
//! system (Audio MIDI Setup) has them. Its one format ([`mix_format`]) is
//! stereo in the IO (virtual) format of its first output stream of two
//! channels or more, the format an IOProc exchanges (32-bit float on a
//! mixable device, [`SampleFormat::F32`]), at that format's rate; on a
//! stream of more channels the stream plays on the first two and the rest
//! get silence. Reading it does not touch the device.
//!
//! An IOProc on the HAL device rather than the AUHAL output unit: it needs
//! no new dependency (the unit is in AudioToolbox), runs shared as well as
//! the unit does, and gives the stream's own output time stamps, which the
//! position below is made of. The unit's extras (its own format and rate
//! conversion, default-device following) are not wanted: the stream is in
//! the device's own format, and the player follows the default device
//! itself.
//!
//! Opening: refuse a device another process holds in hog mode
//! ([`Error::DeviceBusy`]), ask for an IO buffer of one period (a setting
//! of this process only, which the microphone's IO buffer on the same
//! device replaces; the stream info keeps the one it opened with), register
//! our IOProc, and listen to the device's nominal rate, its being alive and
//! the stream's IO format: anything changed there (another application, or
//! Audio MIDI Setup) fails the stream with [`Error::DeviceLost`], so it is
//! opened again in the device's new format. The IOProc fills the device
//! buffer from the ring. The position ([`played`]) is the frames delivered
//! minus how far ahead of "now" the buffer's output time is, minus the
//! device and stream latency.

use super::coreaudio_hal::{
    addr, get, get_string, hogged, output_streams, set, status_error, system_devices, Listener,
    GLOBAL, OUTPUT,
};
use crate::device::Device;
use crate::error::Error;
use crate::format::{SampleFormat, StreamFormat};
use crate::output::{Core, Driver, OpenRequest, StreamInfo};
use crate::shared::Shared;
use objc2_core_audio::{
    kAudioDevicePropertyBufferFrameSize, kAudioDevicePropertyDeviceIsAlive,
    kAudioDevicePropertyDeviceUID, kAudioDevicePropertyLatency,
    kAudioDevicePropertyNominalSampleRate, kAudioHardwarePropertyDefaultOutputDevice,
    kAudioObjectPropertyName, kAudioObjectSystemObject, kAudioStreamPropertyLatency,
    kAudioStreamPropertyVirtualFormat, AudioDeviceCreateIOProcID, AudioDeviceDestroyIOProcID,
    AudioDeviceIOProcID, AudioDeviceStart, AudioDeviceStop, AudioObjectID,
    AudioObjectPropertyAddress,
};
use objc2_core_audio_types::{
    kAudioFormatFlagIsAlignedHigh, kAudioFormatFlagIsBigEndian, kAudioFormatFlagIsFloat,
    kAudioFormatFlagIsNonInterleaved, kAudioFormatFlagIsSignedInteger, kAudioFormatLinearPCM,
    AudioBuffer, AudioBufferList, AudioStreamBasicDescription, AudioTimeStamp,
};
use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub(crate) const BACKEND: &str = "Core Audio (shared)";

/// The stereo frames a wider stream's buffer is rendered by at a time.
const SPREAD_FRAMES: usize = 1024;

pub(crate) fn devices() -> Result<Vec<Device>, Error> {
    let default: AudioObjectID = get(
        kAudioObjectSystemObject as AudioObjectID,
        addr(kAudioHardwarePropertyDefaultOutputDevice, GLOBAL),
        "default output device",
    )
    .unwrap_or(0);
    Ok(system_devices()
        .into_iter()
        .filter(|&d| !output_streams(d).is_empty())
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
            !output_streams(d).is_empty()
                && get_string(d, kAudioDevicePropertyDeviceUID).as_deref() == Some(uid)
        })
        .ok_or_else(|| Error::DeviceNotFound(uid.into()))
}

/// The sample format of an interleaved little-endian description, 32-bit
/// float or signed integer, if it is one of ours.
fn sample_format(d: &AudioStreamBasicDescription) -> Option<SampleFormat> {
    let flags = d.mFormatFlags;
    if d.mFormatID != kAudioFormatLinearPCM
        || flags & kAudioFormatFlagIsBigEndian != 0
        || flags & kAudioFormatFlagIsNonInterleaved != 0
        || d.mChannelsPerFrame == 0
    {
        return None;
    }
    let bytes = d.mBytesPerFrame / d.mChannelsPerFrame;
    if flags & kAudioFormatFlagIsFloat != 0 {
        return (d.mBitsPerChannel == 32 && bytes == 4).then_some(SampleFormat::F32);
    }
    if flags & kAudioFormatFlagIsSignedInteger == 0 {
        return None;
    }
    match (d.mBitsPerChannel, bytes) {
        (16, 2) => Some(SampleFormat::S16),
        (24, 3) => Some(SampleFormat::S24Packed),
        (24, 4) if flags & kAudioFormatFlagIsAlignedHigh != 0 => Some(SampleFormat::S24In32),
        (24, 4) => Some(SampleFormat::S24In32Low),
        (32, 4) => Some(SampleFormat::S32),
        _ => None,
    }
}

/// The stereo format an IO format `d` plays: its rate (whole Hz) and sample
/// format, where it has two channels or more.
fn stereo_of(d: &AudioStreamBasicDescription) -> Option<StreamFormat> {
    let rate = d.mSampleRate.round();
    (d.mChannelsPerFrame >= 2 && rate >= 1.0)
        .then(|| sample_format(d))
        .flatten()
        .map(|sample| StreamFormat::new(rate as u32, 2, sample))
}

/// Whether two IO formats are the same for a stream opened at the first.
fn same_io(a: &AudioStreamBasicDescription, b: &AudioStreamBasicDescription) -> bool {
    a.mSampleRate == b.mSampleRate
        && a.mFormatID == b.mFormatID
        && a.mFormatFlags == b.mFormatFlags
        && a.mBytesPerFrame == b.mBytesPerFrame
        && a.mChannelsPerFrame == b.mChannelsPerFrame
        && a.mBitsPerChannel == b.mBitsPerChannel
}

/// The output stream we play on: its index among the device's output
/// streams (= the buffer index in the IOProc's buffer list), the stream,
/// its IO format, and the stereo format it plays.
struct IoStream {
    index: usize,
    stream: AudioObjectID,
    io: AudioStreamBasicDescription,
    format: StreamFormat,
}

fn io_format(stream: AudioObjectID) -> Result<AudioStreamBasicDescription, Error> {
    get(
        stream,
        addr(kAudioStreamPropertyVirtualFormat, GLOBAL),
        "get virtual format",
    )
}

/// The device's first output stream of two channels or more in a format
/// of ours.
fn io_stream(device: AudioObjectID) -> Result<IoStream, Error> {
    output_streams(device)
        .into_iter()
        .enumerate()
        .find_map(|(index, stream)| {
            let io = io_format(stream).ok()?;
            let format = stereo_of(&io)?;
            Some(IoStream {
                index,
                stream,
                io,
                format,
            })
        })
        .ok_or_else(|| Error::Unsupported("the device has no stereo stream we play".into()))
}

pub(crate) fn mix_format(uid: &str) -> Result<StreamFormat, Error> {
    Ok(io_stream(find(uid)?)?.format)
}

/// Stream frames played as of an IOProc call: `delivered` frames were
/// handed over before this buffer, whose first frame reaches the hardware
/// `ahead` frames from now, and the DAC `latency` frames after that.
fn played(delivered: u64, ahead: f64, latency: f64) -> u64 {
    (delivered as f64 - ahead.max(0.0) - latency).max(0.0) as u64
}

/// What the IOProc and the listener read; owned by the driver, freed after
/// both are gone.
struct IoContext {
    shared: Arc<Shared>,
    /// Our stream's buffer index in the IOProc's buffer list.
    stream_index: usize,
    /// The stream's channels (2 or more; ours are the first two).
    channels: usize,
    /// Device and stream latency, in frames.
    latency: f64,
    /// Device frames delivered since the last restart.
    delivered: AtomicU64,
    /// A wider stream's stereo frames before they are spread; only the
    /// IOProc touches it.
    scratch: UnsafeCell<Vec<u8>>,
    /// What the stream was opened at, for the listener to compare.
    device: AudioObjectID,
    stream: AudioObjectID,
    io: AudioStreamBasicDescription,
}

// SAFETY: `scratch` is used only from the IOProc, which the HAL calls on one
// thread at a time; everything else is atomic or internally synchronised.
unsafe impl Sync for IoContext {}

impl IoContext {
    /// Fills our stream's buffer: the ring's stereo frames, on a wider
    /// stream in its first two channels with silence in the rest (a chunk
    /// at a time). Returns its frames, every one delivered: those rendered
    /// while the handle kept the consumer out are silence, which the
    /// timeline counts at the next render that goes through.
    fn fill(&self, out: &mut [u8]) -> u64 {
        let fb = self.shared.format.frame_bytes();
        if self.channels == 2 {
            self.shared.render_or_skip(out);
            return (out.len() / fb) as u64;
        }
        let wide = fb / 2 * self.channels;
        let frames = out.len() / wide;
        // SAFETY: see the Sync impl.
        let scratch = unsafe { &mut *self.scratch.get() };
        for (start, chunk) in out.chunks_mut(SPREAD_FRAMES * wide).enumerate() {
            let n = (frames - start * SPREAD_FRAMES).min(SPREAD_FRAMES);
            let stereo = &mut scratch[..n * fb];
            self.shared.render_or_skip(stereo);
            for (frame, pair) in chunk.chunks_exact_mut(wide).zip(stereo.chunks_exact(fb)) {
                frame[..fb].copy_from_slice(pair);
                frame[fb..].fill(0);
            }
        }
        frames as u64
    }

    /// Whether the device still runs the stream's format.
    fn unchanged(&self) -> bool {
        let alive = get::<u32>(
            self.device,
            addr(kAudioDevicePropertyDeviceIsAlive, GLOBAL),
            "alive",
        )
        .is_ok_and(|a| a != 0);
        let rate = get::<f64>(
            self.device,
            addr(kAudioDevicePropertyNominalSampleRate, GLOBAL),
            "nominal rate",
        );
        alive
            && rate.is_ok_and(|r| r.round() as u32 == self.shared.format.rate)
            && io_format(self.stream).is_ok_and(|io| same_io(&io, &self.io))
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
    let mut frames = 0;
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
    let delivered = ctx.delivered.load(Ordering::Relaxed);
    let ahead = (*output_time.as_ptr()).mSampleTime - (*now.as_ptr()).mSampleTime;
    // `now` is the time of this call: the clock is read at it.
    ctx.shared
        .report(played(delivered, ahead, ctx.latency), Instant::now(), true);
    ctx.delivered.store(delivered + frames, Ordering::Relaxed);
    0
}

/// A property the stream depends on changed: it fails, unless the device
/// still runs its format.
unsafe extern "C-unwind" fn changed(
    _object: AudioObjectID,
    _count: u32,
    _addresses: NonNull<AudioObjectPropertyAddress>,
    client: *mut c_void,
) -> i32 {
    let ctx = &*(client as *const IoContext);
    if !ctx.unchanged() {
        ctx.shared.fail(Error::DeviceLost);
    }
    0
}

fn configure(uid: &str, request: &OpenRequest) -> Result<(IoContext, StreamInfo), Error> {
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
    if hogged(device) {
        return Err(Error::DeviceBusy);
    }
    let s = io_stream(device)?;
    if !request.takes(s.format) {
        return Err(Error::FormatNotAccepted {
            rate: request.rate,
            channels: request.channels,
        });
    }
    let format = s.format;
    // One period of IO buffer, for this process only.
    let period = format.duration_to_frames(request.period).max(16) as u32;
    let bfs = addr(kAudioDevicePropertyBufferFrameSize, GLOBAL);
    let _ = set(device, bfs, period, "set buffer frame size");
    let buffer = get::<u32>(device, bfs, "get buffer frame size").unwrap_or(period);
    let device_latency =
        get::<u32>(device, addr(kAudioDevicePropertyLatency, OUTPUT), "latency").unwrap_or(0);
    let stream_latency = get::<u32>(
        s.stream,
        addr(kAudioStreamPropertyLatency, GLOBAL),
        "stream latency",
    )
    .unwrap_or(0);
    let shared = Arc::new(Shared::new(
        format,
        request.ring_frames(&format, u64::from(buffer)),
    ));
    let info = StreamInfo {
        format,
        backend: BACKEND,
        device_buffer_frames: u64::from(buffer),
        ring_frames: shared.ring.capacity(),
        realtime: true,
    };
    let channels = s.io.mChannelsPerFrame as usize;
    let ctx = IoContext {
        shared,
        stream_index: s.index,
        channels,
        latency: f64::from(device_latency) + f64::from(stream_latency),
        delivered: AtomicU64::new(0),
        scratch: UnsafeCell::new(if channels == 2 {
            Vec::new()
        } else {
            vec![0; SPREAD_FRAMES * format.frame_bytes()]
        }),
        device,
        stream: s.stream,
        io: s.io,
    };
    Ok((ctx, info))
}

struct HalDriver {
    device: AudioObjectID,
    proc_id: AudioDeviceIOProcID,
    ctx: *mut IoContext,
    /// Removed in `close`, before the context is freed.
    listener: Option<Listener>,
    running: bool,
    closed: bool,
}

// SAFETY: the HAL calls used here are thread-safe; `ctx` is only freed in
// `close`, after the IOProc and the listener are gone.
unsafe impl Send for HalDriver {}

impl HalDriver {
    fn ctx(&self) -> &IoContext {
        // SAFETY: valid until `close`.
        unsafe { &*self.ctx }
    }

    fn stop(&mut self) -> Result<(), Error> {
        if self.running {
            // SAFETY: stops our IOProc on our device.
            let status = unsafe { AudioDeviceStop(self.device, self.proc_id) };
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
            let status = unsafe { AudioDeviceStart(self.device, self.proc_id) };
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
        self.listener = None;
        let stopped = self.stop();
        self.ctx().shared.freeze();
        // SAFETY: destroys our IOProc; after this the HAL no longer calls it
        // (nor, the listener removed, anything else), so the context can be
        // freed.
        unsafe {
            AudioDeviceDestroyIOProcID(self.device, self.proc_id);
            drop(Box::from_raw(self.ctx));
        }
        stopped
    }
}

impl Drop for HalDriver {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// Opens the device with UID `uid` shared.
pub(crate) fn open(uid: &str, request: &OpenRequest) -> Result<Core, Error> {
    request.validate()?;
    let (ctx, info) = configure(uid, request)?;
    let (device, stream) = (ctx.device, ctx.stream);
    let shared = ctx.shared.clone();
    let ctx = Box::into_raw(Box::new(ctx));
    let mut proc_id: AudioDeviceIOProcID = None;
    // SAFETY: registers `io_proc` with a context that lives until close.
    let status = unsafe {
        AudioDeviceCreateIOProcID(
            device,
            Some(io_proc),
            ctx.cast(),
            NonNull::from(&mut proc_id),
        )
    };
    if status != 0 || proc_id.is_none() {
        // SAFETY: the IOProc was not registered, so nothing else holds ctx.
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
    let driver = HalDriver {
        device,
        proc_id,
        ctx,
        listener: Some(listener),
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
            mSampleRate: 48_000.0,
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
    fn io_formats_map_to_sample_formats() {
        let int = kAudioFormatFlagIsSignedInteger;
        let float = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
        assert_eq!(sample_format(&asbd(32, 4, float)), Some(SampleFormat::F32));
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
        assert_eq!(sample_format(&asbd(64, 8, float)), None);
        assert_eq!(
            sample_format(&asbd(32, 4, float | kAudioFormatFlagIsNonInterleaved)),
            None
        );
        assert_eq!(
            sample_format(&asbd(24, 3, int | kAudioFormatFlagIsBigEndian)),
            None
        );
    }

    /// Stereo at the IO format's own rate, on a stream of two channels or
    /// more; a changed rate or layout is another format.
    #[test]
    fn a_stream_plays_stereo_at_its_io_rate() {
        let float = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
        let mut eight = asbd(32, 4, float);
        eight.mChannelsPerFrame = 8;
        eight.mBytesPerFrame = 32;
        eight.mSampleRate = 44_100.0;
        assert_eq!(
            stereo_of(&eight),
            Some(StreamFormat::new(44_100, 2, SampleFormat::F32))
        );
        let mut mono = asbd(32, 4, float);
        mono.mChannelsPerFrame = 1;
        mono.mBytesPerFrame = 4;
        assert_eq!(stereo_of(&mono), None);
        let stereo = asbd(32, 4, float);
        assert!(same_io(&stereo, &stereo));
        let mut faster = stereo;
        faster.mSampleRate = 96_000.0;
        assert!(!same_io(&stereo, &faster));
        assert!(!same_io(&stereo, &eight));
    }

    /// A buffer of a stream wider than stereo is rendered a chunk at a
    /// time: with the consumer kept out from a chunk on, the whole buffer
    /// is delivered, the chunks kept out counted as silence, so the
    /// position stays on the device's frames.
    #[test]
    fn a_wide_buffer_kept_out_midway_is_delivered_whole() {
        use crate::tap::Tap;

        /// Keeps the consumer out from the buffer after the one it sees.
        struct HoldsOut(Arc<Shared>);
        impl Tap for HoldsOut {
            fn begin(&mut self, _: usize, _: u64, _: usize, _: StreamFormat) {
                self.0.hold_out();
            }
            fn see(&mut self, _: &[i32], _: StreamFormat) {}
        }

        let format = StreamFormat::new(48_000, 2, SampleFormat::F32);
        let shared = Arc::new(Shared::new(format, 8192));
        let frames = 3000;
        shared.ring.push_with(frames, |a, b| {
            a.fill(0);
            b.fill(0);
        });
        shared.note_written(frames, 0, 0);
        let tap = HoldsOut(shared.clone());
        shared.set_tap(Some(Box::new(tap)));
        let float = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
        let mut io = asbd(32, 4, float);
        io.mChannelsPerFrame = 4;
        io.mBytesPerFrame = 16;
        let ctx = IoContext {
            shared: shared.clone(),
            stream_index: 0,
            channels: 4,
            latency: 0.0,
            delivered: AtomicU64::new(0),
            scratch: UnsafeCell::new(vec![0; SPREAD_FRAMES * format.frame_bytes()]),
            device: 0,
            stream: 0,
            io,
        };
        let mut out = vec![0u8; frames as usize * 16];
        assert_eq!(ctx.fill(&mut out), frames);
        let rest = frames - SPREAD_FRAMES as u64;
        assert_eq!(shared.ring.len(), rest);
        // Let go (the tap holds the stream), and play the rest: after the
        // silence the device played meanwhile.
        shared.unsuspend();
        shared.set_tap(None);
        let mut out = vec![0u8; rest as usize * 16];
        assert_eq!(ctx.fill(&mut out), rest);
        let now = std::time::Instant::now();
        shared.report(frames, now, false);
        assert_eq!(shared.position(), SPREAD_FRAMES as u64);
        shared.report(frames + rest, now, false);
        assert_eq!(shared.position(), frames);
    }

    /// What was delivered, less what is still ahead of the hardware and
    /// the DAC's latency; never below zero.
    #[test]
    fn the_position_is_what_reached_the_dac() {
        assert_eq!(played(48_000, 512.0, 40.0), 47_448);
        // An output time behind now counts as now.
        assert_eq!(played(1_000, -30.0, 0.0), 1_000);
        assert_eq!(played(100, 512.0, 40.0), 0);
        assert_eq!(played(0, 0.0, 0.0), 0);
    }
}
