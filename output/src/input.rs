//! Microphone input: the singer heard, never played.
//!
//! Nothing of the microphone reaches the output. It is listened to, to
//! measure the singing: its voice, the music's echo taken out, is kept
//! block by block with where in the stream it was sung.
//!
//! To take the echo out, the canceller needs what played at the moment the
//! microphone heard it. An engine writes far ahead of the DAC (up to the
//! output's ring, 500 ms by default), so that is taken where the device
//! takes its buffers: a [`crate::Tap`] in the device thread sees each one
//! (and cannot change it), and lines it up with the microphone's samples of
//! that moment. The canceller itself runs on a thread of its own.
//!
//! # Use
//!
//! 1. [`input_devices`] lists the capture devices; [`open_input`] opens one
//!    in shared mode at its own rate, with a period of about 5 ms where the
//!    system allows it. Capture runs on its own thread from then on, mixed
//!    down to mono ([`Mono`]) into a lock-free ring of `f32`.
//! 2. [`Input::tap`] makes the [`MicTap`] to hand to
//!    [`Output::set_tap`](crate::Output::set_tap). In the device thread it
//!    resamples the voice to the output's rate and queues it, with what
//!    plays, for the cleaner: a thread that takes the music's echo out
//!    ([`crate::aec`]).
//! 3. [`Input::take_voice`] gives the voice cleaned, block by block
//!    ([`VoiceBlock`]); [`Input::voice_peak`] is a meter of the singer
//!    isolated from it ([`crate::isolate`]: the band the canceller missed
//!    and the room's sounds taken out too) and [`Input::peak`] one of the
//!    voice as captured; [`Input::status`] tells
//!    how the two clocks are held together.
//!
//! Capture stops when the [`Input`] and every tap made from it are dropped.
//!
//! # Two clocks
//!
//! The microphone and the DAC run on different crystals. The tap's
//! resampler (cubic, varispeed) runs at the nominal ratio of the two rates
//! corrected by at most ±0.2 %: a control loop holds the ring's fill at
//! about two capture periods. Measured at the moment a device buffer is
//! made, the fill would rise and fall with where the last capture packet
//! fell; the time since that packet is counted in, so the loop sees the
//! fill as if the voice arrived continuously.
//!
//! When the ring runs dry (the capture stalled) the voice drops out and
//! starts again once two periods are queued. When the fill is held more
//! than a period over its target (a burst of capture, as devices deliver
//! when they start, or the device thread stalled), the excess is skipped
//! at once rather than drained at the loop's limit. Both are counted in
//! [`MicStatus`].
//!
//! # Tests
//!
//! [`Input::feed`] makes an input fed by the caller instead of a device, on
//! a simulated clock, its cleaner run by the caller ([`MicFeed::clean`]):
//! with [`crate::FileOutput`] the whole path runs without hardware.

use crate::aec::{EchoCanceller, BLOCK};
use crate::backend;
use crate::device::Device;
use crate::error::Error;
use crate::format::StreamFormat;
use crate::isolate::{Downsampler, Isolator};
use crate::tap::{Tap, TAP_CHUNK};
use std::cell::UnsafeCell;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The capture backend this build uses: `"WASAPI shared"`, `"Core Audio
/// HAL"`, `"ALSA"`, or `"none"`.
pub const INPUT_BACKEND: &str = backend::INPUT_BACKEND;

/// The capture period asked for.
pub const CAPTURE_PERIOD: Duration = Duration::from_millis(5);

/// The furthest the resampler's ratio moves from the nominal one.
pub const MAX_ADJUST: f64 = 0.002;

/// How a multichannel input becomes one voice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mono {
    /// One channel as it is, the mean of two, the first of more.
    #[default]
    Auto,
    /// The first channel.
    First,
    /// The mean of the first two channels (the first, with one).
    Mean,
    /// This channel (from 0); the first, if there is no such channel.
    Channel(u16),
}

impl Mono {
    fn pick(self, channels: usize) -> Pick {
        match self {
            Mono::Auto if channels == 2 => Pick::Mean,
            Mono::Mean if channels >= 2 => Pick::Mean,
            Mono::Channel(c) if usize::from(c) < channels => Pick::One(usize::from(c)),
            _ => Pick::One(0),
        }
    }
}

#[derive(Clone, Copy)]
enum Pick {
    One(usize),
    Mean,
}

/// The capture devices that are present now. `Device::id` is what
/// [`open_input`] takes.
pub fn input_devices() -> Result<Vec<Device>, Error> {
    backend::input_devices()
}

/// The system's default capture device.
pub fn default_input() -> Result<Device, Error> {
    input_devices()?
        .into_iter()
        .find(|d| d.is_default)
        .ok_or(Error::NoDevice)
}

/// Finds a capture device by `query`: `"default"`, an exact id, an exact
/// name, or a case-insensitive part of exactly one device's name.
pub fn find_input(query: &str) -> Result<Device, Error> {
    if query.eq_ignore_ascii_case("default") {
        return default_input();
    }
    crate::device::pick(input_devices()?, query)
}

/// Opens the capture device `id` (shared, at its own rate) and starts
/// capturing, mixed down as [`Mono::Auto`].
pub fn open_input(id: &str) -> Result<Input, Error> {
    open_input_with(id, Mono::Auto)
}

/// [`open_input`] with the mixdown given.
pub fn open_input_with(id: &str, mono: Mono) -> Result<Input, Error> {
    backend::open_input(id, mono)
}

/// What an open input runs at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputInfo {
    /// The device's id.
    pub id: String,
    /// The device's name.
    pub name: String,
    /// Which backend captures it.
    pub backend: &'static str,
    /// The rate it captures at, as opened ([`Input::rate`] follows a change).
    pub rate: u32,
    /// The device's channels.
    pub channels: u16,
    /// The capture period the system gave, in frames.
    pub period_frames: u32,
    /// How the channels become one voice.
    pub mono: Mono,
}

/// How the microphone path is doing; see [`Input::status`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MicStatus {
    /// The capture rate.
    pub rate: u32,
    /// The capture period the system gave, in frames.
    pub period_frames: u32,
    /// The largest capture packet seen, in frames.
    pub packet_frames: u32,
    /// The ring's fill as the control loop sees it (smoothed), in capture
    /// frames.
    pub fill: f64,
    /// The fill it holds: two capture periods (of the larger of the
    /// period and the largest packet).
    pub target: u32,
    /// The correction to the nominal ratio of the two clocks: positive when
    /// the microphone runs fast against the DAC. Within ±[`MAX_ADJUST`].
    pub adjust: f64,
    /// Whether a tap is consuming the voice now.
    pub active: bool,
    /// Times the ring ran dry and the voice dropped out.
    pub underruns: u64,
    /// Capture frames lost because the ring was full (nothing consumed
    /// them).
    pub overruns: u64,
    /// Times the fill was held more than a period over its target and the
    /// excess was skipped.
    pub resyncs: u64,
    /// Blocks lost on the way to the cleaner (its thread fell behind).
    pub blocks_dropped: u64,
    /// Cleaned blocks lost because nothing took them ([`Input::take_voice`]).
    pub voice_dropped: u64,
}

impl MicStatus {
    /// The fill as time.
    pub fn buffered(&self) -> Duration {
        if self.rate == 0 {
            return Duration::ZERO;
        }
        Duration::from_secs_f64(self.fill.max(0.0) / f64::from(self.rate))
    }
}

// ---------------------------------------------------------------------------
// The ring

/// A single-producer, single-consumer ring of `f32`.
pub(crate) struct FloatRing {
    buf: Box<[UnsafeCell<f32>]>,
    capacity: u64,
    head: AtomicU64,
    tail: AtomicU64,
}

// SAFETY: as `crate::ring::Ring`: the producer writes only the free region,
// the consumer reads only the filled one, and the Release stores of `head`
// and `tail` publish them.
unsafe impl Sync for FloatRing {}

impl FloatRing {
    pub(crate) fn new(capacity: usize) -> FloatRing {
        let capacity = capacity.max(1);
        FloatRing {
            buf: (0..capacity).map(|_| UnsafeCell::new(0.0)).collect(),
            capacity: capacity as u64,
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
        }
    }

    pub(crate) fn len(&self) -> usize {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        (head - tail) as usize
    }

    fn slot(&self, i: u64) -> *mut f32 {
        self.buf[(i % self.capacity) as usize].get()
    }

    /// Producer only: the room left.
    pub(crate) fn free(&self) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        (self.capacity - (head - tail)) as usize
    }

    /// Producer only: queues what fits of `samples`; returns how many.
    pub(crate) fn push(&self, samples: &[f32]) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        let n = samples.len().min((self.capacity - (head - tail)) as usize);
        for (k, &s) in samples[..n].iter().enumerate() {
            // SAFETY: the slot is in the free region.
            unsafe { *self.slot(head + k as u64) = s };
        }
        self.head.store(head + n as u64, Ordering::Release);
        n
    }

    /// Consumer only: takes up to `out.len()` samples; returns how many.
    pub(crate) fn pop(&self, out: &mut [f32]) -> usize {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        let n = out.len().min((head - tail) as usize);
        for (k, o) in out[..n].iter_mut().enumerate() {
            // SAFETY: the slot is in the filled region.
            *o = unsafe { *self.slot(tail + k as u64) };
        }
        self.tail.store(tail + n as u64, Ordering::Release);
        n
    }

    /// Consumer only: drops up to `n` of the oldest samples.
    pub(crate) fn skip(&self, n: usize) -> usize {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        let n = n.min((head - tail) as usize);
        self.tail.store(tail + n as u64, Ordering::Release);
        n
    }
}

// ---------------------------------------------------------------------------
// What the capture side and the insert share

fn f32_bits(v: f32) -> u32 {
    v.to_bits()
}

pub(crate) struct MicShared {
    ring: FloatRing,
    rate: AtomicU32,
    period: AtomicU32,
    packet: AtomicU32,
    epoch: Instant,
    /// The clock is the caller's ([`MicFeed::set_time`]), not the system's.
    simulated: bool,
    sim_now: AtomicU64,
    /// When the last packet was queued, in ns from `epoch`, plus one (0:
    /// never).
    pushed_at: AtomicU64,
    /// When the first packet was queued, the same way.
    first_at: AtomicU64,
    peak: AtomicU32,
    /// Anything but exact zeros has come in.
    signal: AtomicBool,
    /// The echo canceller is on ([`Input::set_echo_cancel`]).
    echo_on: AtomicBool,
    /// What the tap queues for the cleaner, block by block (see
    /// [`Packet`]): the microphone and what plays, side by side.
    blocks: FloatRing,
    /// Blocks the tap had no room for.
    blocks_dropped: AtomicU64,
    /// The voice the cleaner gives, block by block ([`VoiceBlock`]), for
    /// [`Input::take_voice`]; blocks it had no room for.
    voice: FloatRing,
    voice_dropped: AtomicU64,
    /// The voice's peak after the canceller, as `f32` bits, falling like
    /// `peak`; and when it was last set (0: never), in ns from `epoch`.
    voice_peak: AtomicU32,
    voice_at: AtomicU64,
    /// The canceller's status: the echo removed in dB by both stages and
    /// by the filter (`f32` bits), whether it has learnt the echo's way,
    /// and the echo's delay in seconds (`f32` bits; negative: not known).
    echo_removed: AtomicU32,
    echo_filter: AtomicU32,
    echo_learnt: AtomicBool,
    echo_delay: AtomicU32,
    /// A recording of the canceller under way ([`Input::record_echo`]):
    /// its blocks (the microphone, what plays, the voice it gives:
    /// interleaved) queue here for the thread that writes them; the rate
    /// the canceller runs at.
    recording: AtomicBool,
    record: FloatRing,
    echo_rate: AtomicU32,
    /// Tells the cleaner's thread to stop.
    stop: AtomicBool,
    fill: AtomicU64,
    target: AtomicU32,
    adjust: AtomicU64,
    underruns: AtomicU64,
    overruns: AtomicU64,
    resyncs: AtomicU64,
    /// An insert holds the consumer side of the ring.
    claimed: AtomicBool,
    failed: AtomicBool,
    failure: Mutex<Option<Error>>,
}

impl MicShared {
    pub(crate) fn new(rate: u32, period: u32, simulated: bool) -> Arc<MicShared> {
        // A second of voice: far more than the loop ever holds.
        let capacity = (rate as usize).max(4096);
        Arc::new(MicShared {
            ring: FloatRing::new(capacity),
            rate: AtomicU32::new(rate),
            period: AtomicU32::new(period.max(1)),
            packet: AtomicU32::new(0),
            epoch: Instant::now(),
            simulated,
            sim_now: AtomicU64::new(0),
            pushed_at: AtomicU64::new(0),
            first_at: AtomicU64::new(0),
            peak: AtomicU32::new(0),
            signal: AtomicBool::new(false),
            echo_on: AtomicBool::new(true),
            blocks: FloatRing::new(BLOCK_QUEUE * Packet::LEN),
            blocks_dropped: AtomicU64::new(0),
            voice: FloatRing::new(VOICE_QUEUE * VoiceBlock::LEN),
            voice_dropped: AtomicU64::new(0),
            voice_peak: AtomicU32::new(0),
            voice_at: AtomicU64::new(0),
            echo_removed: AtomicU32::new(0),
            echo_filter: AtomicU32::new(0),
            echo_learnt: AtomicBool::new(false),
            echo_delay: AtomicU32::new(f32_bits(-1.0)),
            recording: AtomicBool::new(false),
            record: FloatRing::new(RECORD_QUEUE),
            echo_rate: AtomicU32::new(0),
            stop: AtomicBool::new(false),
            fill: AtomicU64::new(0),
            target: AtomicU32::new(0),
            adjust: AtomicU64::new(0),
            underruns: AtomicU64::new(0),
            overruns: AtomicU64::new(0),
            resyncs: AtomicU64::new(0),
            claimed: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            failure: Mutex::new(None),
        })
    }

    fn now_ns(&self) -> u64 {
        if self.simulated {
            self.sim_now.load(Ordering::Acquire)
        } else {
            self.epoch.elapsed().as_nanos() as u64
        }
    }

    /// The capture rate changed under the stream (macOS: the device's
    /// nominal rate, when it is also the output).
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(crate) fn set_rate(&self, rate: u32) {
        self.rate.store(rate, Ordering::Release);
    }

    /// The capture failed; capture stops and [`Input::failure`] says why.
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    pub(crate) fn fail(&self, error: Error) {
        let mut f = self.failure.lock().unwrap_or_else(|e| e.into_inner());
        if f.is_none() {
            *f = Some(error);
        }
        self.failed.store(true, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// The capture side

/// A sample layout capture devices deliver, interleaved, little-endian.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Wire {
    F32,
    I16,
    /// 24 bits in three bytes.
    I24,
    /// A 32-bit word, the value in its top bits (S32, and 24-in-32 aligned
    /// high).
    I32,
    /// 24 bits in the low three bytes of a 32-bit word.
    #[cfg_attr(windows, allow(dead_code))]
    I24Low,
}

impl Wire {
    pub(crate) fn bytes(self) -> usize {
        match self {
            Wire::I16 => 2,
            Wire::I24 => 3,
            Wire::F32 | Wire::I32 | Wire::I24Low => 4,
        }
    }

    fn read(self, b: &[u8]) -> f32 {
        match self {
            Wire::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            Wire::I16 => f32::from(i16::from_le_bytes([b[0], b[1]])) / 32_768.0,
            Wire::I24 => i32::from_le_bytes([0, b[0], b[1], b[2]]) as f32 / 2_147_483_648.0,
            Wire::I32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.0,
            Wire::I24Low => i32::from_le_bytes([0, b[0], b[1], b[2]]) as f32 / 2_147_483_648.0,
        }
    }
}

/// The capture thread's end: mixes packets down to mono and queues them.
pub(crate) struct Producer {
    shared: Arc<MicShared>,
    mono: Mono,
    scratch: Vec<f32>,
}

impl Producer {
    pub(crate) fn new(shared: Arc<MicShared>, mono: Mono) -> Producer {
        Producer {
            shared,
            mono,
            scratch: vec![0.0; 4096],
        }
    }

    /// Queues `frames` frames of interleaved `channels`-channel `wire`
    /// samples.
    pub(crate) fn push(&mut self, data: &[u8], wire: Wire, channels: usize, frames: usize) {
        let sb = wire.bytes();
        let fb = sb * channels.max(1);
        let frames = frames.min(data.len() / fb);
        let pick = self.mono.pick(channels);
        self.push_with(frames, |f| {
            let frame = &data[f * fb..];
            match pick {
                Pick::One(c) => wire.read(&frame[c * sb..]),
                Pick::Mean => 0.5 * (wire.read(frame) + wire.read(&frame[sb..])),
            }
        });
    }

    /// Queues `frames` frames from one buffer per channel (non-interleaved).
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(crate) fn push_planar(&mut self, planes: &[&[u8]], wire: Wire, frames: usize) {
        let sb = wire.bytes();
        if planes.is_empty() {
            return;
        }
        let frames = planes
            .iter()
            .map(|p| p.len() / sb)
            .min()
            .unwrap_or(0)
            .min(frames);
        let pick = self.mono.pick(planes.len());
        self.push_with(frames, |f| match pick {
            Pick::One(c) => wire.read(&planes[c][f * sb..]),
            Pick::Mean => 0.5 * (wire.read(&planes[0][f * sb..]) + wire.read(&planes[1][f * sb..])),
        });
    }

    /// Queues `frames` frames of silence (a packet the system marked silent).
    #[cfg_attr(target_os = "linux", allow(dead_code))]
    pub(crate) fn push_silence(&mut self, frames: usize) {
        self.push_with(frames, |_| 0.0);
    }

    fn push_with(&mut self, frames: usize, sample: impl Fn(usize) -> f32) {
        let s = &self.shared;
        let mut peak = 0.0f32;
        let mut done = 0;
        while done < frames {
            let n = (frames - done).min(self.scratch.len());
            for (k, o) in self.scratch[..n].iter_mut().enumerate() {
                let v = sample(done + k);
                // A NaN or an infinity from a broken driver is silence.
                *o = if v.is_finite() { v } else { 0.0 };
                peak = peak.max(o.abs());
            }
            let queued = s.ring.push(&self.scratch[..n]);
            if queued < n {
                s.overruns.fetch_add((n - queued) as u64, Ordering::Relaxed);
            }
            done += n;
        }
        let now = s.now_ns() + 1;
        s.pushed_at.store(now, Ordering::Release);
        if s.first_at.load(Ordering::Relaxed) == 0 {
            s.first_at.store(now, Ordering::Relaxed);
        }
        s.packet.fetch_max(frames as u32, Ordering::Relaxed);
        if peak > 0.0 {
            s.signal.store(true, Ordering::Relaxed);
        }
        // The meter falls by about 30 dB a second.
        let rate = s.rate.load(Ordering::Relaxed).max(1);
        let decay = (-(frames as f32) / rate as f32 / 0.3).exp();
        let old = f32::from_bits(s.peak.load(Ordering::Relaxed));
        s.peak
            .store(f32_bits(peak.max(old * decay)), Ordering::Relaxed);
    }

    pub(crate) fn shared(&self) -> &Arc<MicShared> {
        &self.shared
    }
}

// ---------------------------------------------------------------------------
// The handle

struct InputInner {
    shared: Arc<MicShared>,
    info: InputInfo,
    /// The backend's stream; dropping it stops capture.
    stream: Mutex<Option<Box<dyn Send>>>,
    /// The cleaner's thread (none for a [`MicFeed`], whose caller runs it).
    cleaner: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Drop for InputInner {
    fn drop(&mut self) {
        let stream = self
            .stream
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        drop(stream);
        self.shared.stop.store(true, Ordering::Release);
        let cleaner = self
            .cleaner
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(thread) = cleaner {
            let _ = thread.join();
        }
    }
}

/// An open capture device (or a [`MicFeed`]). Cheap to clone; capture stops
/// when the last clone and every [`MicTap`] made from it are dropped.
#[derive(Clone)]
pub struct Input {
    inner: Arc<InputInner>,
}

impl Input {
    pub(crate) fn new(info: InputInfo, shared: Arc<MicShared>, stream: Box<dyn Send>) -> Input {
        let cleaner = if shared.simulated {
            None
        } else {
            let shared = Arc::clone(&shared);
            std::thread::Builder::new()
                .name("jodel-mic-clean".into())
                .spawn(move || {
                    let mut cleaner = Cleaner::new();
                    while !shared.stop.load(Ordering::Acquire) {
                        if !cleaner.run(&shared) {
                            std::thread::sleep(Duration::from_millis(5));
                        }
                    }
                })
                .ok()
        };
        Input {
            inner: Arc::new(InputInner {
                shared,
                info,
                stream: Mutex::new(Some(stream)),
                cleaner: Mutex::new(cleaner),
            }),
        }
    }

    /// An input the caller feeds ([`MicFeed::push`]) instead of a device,
    /// at `rate` with packets of `period_frames`, on the caller's clock
    /// ([`MicFeed::set_time`]). For tests and simulations.
    pub fn feed(rate: u32, period_frames: u32) -> (Input, MicFeed) {
        let shared = MicShared::new(rate, period_frames, true);
        let info = InputInfo {
            id: "feed".into(),
            name: "feed".into(),
            backend: "feed",
            rate,
            channels: 1,
            period_frames,
            mono: Mono::Auto,
        };
        let feed = MicFeed {
            producer: Producer::new(shared.clone(), Mono::Auto),
            cleaner: Cleaner::new(),
        };
        (Input::new(info, shared, Box::new(())), feed)
    }

    fn shared(&self) -> &MicShared {
        &self.inner.shared
    }

    /// What it was opened at.
    pub fn info(&self) -> &InputInfo {
        &self.inner.info
    }

    /// The capture rate now.
    pub fn rate(&self) -> u32 {
        self.shared().rate.load(Ordering::Acquire)
    }

    /// The voice's recent peak as captured: 0.0 to 1.0 (full scale),
    /// falling by about 30 dB a second.
    pub fn peak(&self) -> f32 {
        f32::from_bits(self.shared().peak.load(Ordering::Relaxed))
    }

    /// Turns the echo canceller on or off ([`crate::aec`]): on (as it
    /// opens), what the output plays is taken out of the voice the cleaner
    /// gives; off, the voice is the microphone as it is. Any thread.
    pub fn set_echo_cancel(&self, on: bool) {
        self.shared().echo_on.store(on, Ordering::Relaxed);
    }

    /// Whether the echo canceller is on.
    pub fn echo_cancel(&self) -> bool {
        self.shared().echo_on.load(Ordering::Relaxed)
    }

    /// The recent peak of the singer isolated: the music's echo taken out,
    /// then what the canceller left of the band and the room's sounds
    /// ([`crate::isolate`]), 11 to 21 ms late. It falls like
    /// [`Input::peak`]; `None` while the cleaner has given
    /// none lately (no output tapped: nothing plays to take out, and
    /// [`Input::peak`] is then the voice).
    pub fn voice_peak(&self) -> Option<f32> {
        let s = self.shared();
        let at = s.voice_at.load(Ordering::Relaxed);
        let fresh = at != 0 && s.now_ns().saturating_sub(at - 1) < 500_000_000;
        fresh.then(|| f32::from_bits(s.voice_peak.load(Ordering::Relaxed)))
    }

    /// Takes the voice cleaned since the last call into `out` (appended),
    /// oldest first. The cleaner keeps the last few seconds; what nobody
    /// takes in time is dropped ([`MicStatus::voice_dropped`]).
    pub fn take_voice(&self, out: &mut Vec<VoiceBlock>) {
        let ring = &self.shared().voice;
        let mut packet = [0.0f32; VoiceBlock::LEN];
        while ring.len() >= VoiceBlock::LEN {
            ring.pop(&mut packet);
            out.push(VoiceBlock::read(&packet));
        }
    }

    /// How the echo canceller is doing: the echo it removes in dB, and
    /// whether it has learnt the echo's way (see [`crate::aec::EchoStatus`]).
    pub fn echo_status(&self) -> crate::aec::EchoStatus {
        let s = self.shared();
        crate::aec::EchoStatus {
            removed_db: f32::from_bits(s.echo_removed.load(Ordering::Relaxed)),
            filter_db: f32::from_bits(s.echo_filter.load(Ordering::Relaxed)),
            learnt: s.echo_learnt.load(Ordering::Relaxed),
            delay: Some(f32::from_bits(s.echo_delay.load(Ordering::Relaxed))).filter(|d| *d >= 0.0),
        }
    }

    /// Records what the echo canceller works with for `seconds` (of its
    /// own time) to `path`, a WAV file of three channels at its rate,
    /// 32-bit float ([`wav_header`]): the microphone as it hears it, what plays (the music
    /// going to the speakers), and the voice it gives. For tuning it to a
    /// room. The cleaner queues its blocks; a thread of its own writes
    /// them. The handle says when it is done and how it went. It stops
    /// early, with what it has, when no block comes for a few seconds (no
    /// output tapped). One recording at a time.
    pub fn record_echo(&self, path: &Path, seconds: f64) -> Result<EchoRecording, String> {
        let shared = Arc::clone(&self.inner.shared);
        if shared.recording.swap(true, Ordering::AcqRel) {
            return Err("a recording is under way".into());
        }
        let path = path.to_owned();
        let thread = std::thread::Builder::new()
            .name("jodel-echo-record".into())
            .spawn(move || {
                let result = write_recording(&shared, &path, seconds);
                shared.recording.store(false, Ordering::Release);
                result.map(|()| path)
            })
            .map_err(|e| {
                self.shared().recording.store(false, Ordering::Release);
                e.to_string()
            })?;
        Ok(EchoRecording {
            thread: Some(thread),
        })
    }

    /// Whether capture has run for at least `after` and every sample was
    /// exactly zero. A real microphone always has some noise, so this is
    /// a muted device or, on macOS, a process the system has not allowed
    /// to use the microphone (the HAL then delivers silence, no error).
    pub fn only_silence(&self, after: Duration) -> bool {
        let s = self.shared();
        let first = s.first_at.load(Ordering::Relaxed);
        first != 0
            && !s.signal.load(Ordering::Relaxed)
            && s.now_ns().saturating_sub(first - 1) >= after.as_nanos() as u64
    }

    /// How the path is doing.
    pub fn status(&self) -> MicStatus {
        let s = self.shared();
        MicStatus {
            rate: s.rate.load(Ordering::Acquire),
            period_frames: s.period.load(Ordering::Relaxed),
            packet_frames: s.packet.load(Ordering::Relaxed),
            fill: f64::from_bits(s.fill.load(Ordering::Relaxed)),
            target: s.target.load(Ordering::Relaxed),
            adjust: f64::from_bits(s.adjust.load(Ordering::Relaxed)),
            active: s.claimed.load(Ordering::Acquire),
            underruns: s.underruns.load(Ordering::Relaxed),
            overruns: s.overruns.load(Ordering::Relaxed),
            resyncs: s.resyncs.load(Ordering::Relaxed),
            blocks_dropped: s.blocks_dropped.load(Ordering::Relaxed),
            voice_dropped: s.voice_dropped.load(Ordering::Relaxed),
        }
    }

    /// The error that stopped capture, if one did.
    pub fn failure(&self) -> Option<Error> {
        let s = self.shared();
        if s.failed.load(Ordering::Acquire) {
            s.failure.lock().unwrap_or_else(|e| e.into_inner()).clone()
        } else {
            None
        }
    }

    /// A tap that lines this input up with what an output plays, for the
    /// cleaner (see [`MicTap`]). It keeps the capture running while it
    /// lives.
    pub fn tap(&self) -> MicTap {
        MicTap::new(self.clone())
    }
}

impl std::fmt::Debug for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Input").field("info", self.info()).finish()
    }
}

/// The caller's end of an [`Input::feed`].
pub struct MicFeed {
    producer: Producer,
    /// The cleaner, run when the caller says ([`MicFeed::clean`]).
    cleaner: Cleaner,
}

impl MicFeed {
    /// Queues mono samples (full scale ±1) as one capture packet, at the
    /// time last set. Returns how many fitted (the ring holds a second).
    pub fn push(&mut self, samples: &[f32]) -> usize {
        let before = self.producer.shared().overruns.load(Ordering::Relaxed);
        let bytes: &[u8] = bytemuck_f32(samples);
        self.producer.push(bytes, Wire::F32, 1, samples.len());
        let lost = self.producer.shared().overruns.load(Ordering::Relaxed) - before;
        samples.len() - lost as usize
    }

    /// Sets the simulated clock that both the feed and the tap read.
    pub fn set_time(&self, t: Duration) {
        self.producer
            .shared()
            .sim_now
            .store(t.as_nanos() as u64, Ordering::Release);
    }

    /// Runs the cleaner over every block the tap has queued (a device's
    /// input has a thread of its own for it).
    pub fn clean(&mut self) {
        while self.cleaner.run(self.producer.shared()) {}
    }
}

/// The bytes of `f32` samples, as the wire reads them (little-endian
/// hosts; elsewhere each sample is converted).
fn bytemuck_f32(samples: &[f32]) -> &[u8] {
    #[cfg(target_endian = "little")]
    {
        // SAFETY: f32 has no padding and any byte is a valid u8.
        unsafe {
            std::slice::from_raw_parts(samples.as_ptr().cast(), std::mem::size_of_val(samples))
        }
    }
    #[cfg(not(target_endian = "little"))]
    {
        compile_error!("jodel-output assumes a little-endian host")
    }
}

/// A recording of the echo canceller ([`Input::record_echo`]).
pub struct EchoRecording {
    thread: Option<std::thread::JoinHandle<Result<PathBuf, String>>>,
}

impl EchoRecording {
    /// The file written, once done (`Some` once, then `None`).
    pub fn done(&mut self) -> Option<Result<PathBuf, String>> {
        if !self.thread.as_ref()?.is_finished() {
            return None;
        }
        let thread = self.thread.take()?;
        Some(
            thread
                .join()
                .unwrap_or_else(|_| Err("the recording stopped".into())),
        )
    }
}

/// The bytes of [`wav_header`].
const WAV_HEADER: usize = 80;

/// A WAV header for `frames` of `channels` channels of 32-bit float at
/// `rate`, as players everywhere read one (macOS's Quick Look among them):
/// WAVE_FORMAT_EXTENSIBLE, saying which speaker each channel is for (front
/// left, front right, centre, then on), and a `fact` chunk with the frame
/// count, which a WAV file of floats should have. The plain 16-byte header
/// of float data on three channels was refused by some.
fn wav_header(channels: u16, rate: u32, frames: u64) -> Vec<u8> {
    /// KSDATAFORMAT_SUBTYPE_IEEE_FLOAT.
    const FLOAT: [u8; 16] = [
        0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B,
        0x71,
    ];
    let block = u32::from(channels) * 4;
    let data = (frames * u64::from(block)).min(u64::from(u32::MAX - WAV_HEADER as u32)) as u32;
    let mask: u32 = if channels >= 32 {
        u32::MAX
    } else {
        (1u32 << channels) - 1
    };
    let mut h = Vec::with_capacity(WAV_HEADER);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&(WAV_HEADER as u32 - 8 + data).to_le_bytes());
    h.extend_from_slice(b"WAVE");
    h.extend_from_slice(b"fmt ");
    h.extend_from_slice(&40u32.to_le_bytes());
    h.extend_from_slice(&0xFFFEu16.to_le_bytes());
    h.extend_from_slice(&channels.to_le_bytes());
    h.extend_from_slice(&rate.to_le_bytes());
    h.extend_from_slice(&(rate * block).to_le_bytes());
    h.extend_from_slice(&(block as u16).to_le_bytes());
    h.extend_from_slice(&32u16.to_le_bytes());
    h.extend_from_slice(&22u16.to_le_bytes());
    h.extend_from_slice(&32u16.to_le_bytes());
    h.extend_from_slice(&mask.to_le_bytes());
    h.extend_from_slice(&FLOAT);
    h.extend_from_slice(b"fact");
    h.extend_from_slice(&4u32.to_le_bytes());
    h.extend_from_slice(&(frames.min(u64::from(u32::MAX)) as u32).to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&data.to_le_bytes());
    debug_assert_eq!(h.len(), WAV_HEADER);
    h
}

/// Writes the canceller's queued blocks to `path` as they come, for
/// `seconds` of them (or until none comes for three seconds), under a
/// temporary name that becomes `path` only once the recording is whole
/// ([`durable::AtomicFile`]).
fn write_recording(shared: &MicShared, path: &Path, seconds: f64) -> Result<(), String> {
    use std::io::{Seek, SeekFrom, Write};
    let fail = |e: std::io::Error| format!("{}: {e}", path.display());
    let ring = &shared.record;
    // What was queued before is not this recording's.
    ring.skip(ring.len());
    let mut file = durable::AtomicFile::create(path).map_err(fail)?;
    file.write_all(&[0u8; WAV_HEADER]).map_err(fail)?;
    let mut buf = vec![0.0f32; RECORD_CHANNELS * 4096];
    let mut bytes = Vec::with_capacity(buf.len() * 4);
    let mut frames = 0u64;
    let mut rate = 0u32;
    let mut last = Instant::now();
    loop {
        let n = ring.pop(&mut buf);
        if n == 0 {
            if last.elapsed() > Duration::from_secs(3) {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
            continue;
        }
        last = Instant::now();
        if rate == 0 {
            rate = shared.echo_rate.load(Ordering::Relaxed);
        }
        bytes.clear();
        for v in &buf[..n] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        file.write_all(&bytes).map_err(fail)?;
        frames += (n / RECORD_CHANNELS) as u64;
        if rate > 0 && frames as f64 >= seconds * f64::from(rate) {
            break;
        }
    }
    if frames == 0 {
        // Dropped: the temporary goes, nothing takes the name.
        return Err("nothing came from the echo canceller (is music playing?)".into());
    }
    // The header, now the length is known; then the name.
    file.seek(SeekFrom::Start(0)).map_err(fail)?;
    file.write_all(&wav_header(RECORD_CHANNELS as u16, rate, frames))
        .map_err(fail)?;
    file.commit().map_err(fail)
}

// ---------------------------------------------------------------------------
// The tap

/// The loop's proportional gain, per second of fill error.
const KP: f64 = 0.35;
/// The loop's integral gain, per second² (with `KP`: 0.25 rad/s, damping
/// 0.7, so a clock 0.1 % off settles in about 20 s, within 2 ms of the
/// target on the way).
const KI: f64 = 0.0625;
/// The time constant of the fill's smoothing, in seconds: well inside the
/// loop, and long enough to quiet the jitter of when threads wake.
const SMOOTHING: f64 = 0.5;
/// The most input samples one output sample may need (a ratio of 16).
const MAX_RATIO: usize = 16;
/// How long both sides run before the voice starts: devices deliver and
/// take in bursts as they start.
const WARM_UP: Duration = Duration::from_millis(100);

/// The canceller's rate: about this, the output's divided by a whole number.
const ECHO_RATE: u32 = 48_000;
/// Blocks the tap may queue for the cleaner (about 2.7 s at 48 kHz).
const BLOCK_QUEUE: usize = 1024;
/// Cleaned blocks kept for [`Input::take_voice`] (about 11 s at 48 kHz).
const VOICE_QUEUE: usize = 4096;

/// A recording of the canceller's channels ([`Input::record_echo`]).
const RECORD_CHANNELS: usize = 3;
/// What a recording of the canceller queues at most: two seconds of its
/// channels at its highest rate.
const RECORD_QUEUE: usize = RECORD_CHANNELS * 56_000 * 2;

/// A whole number under 2⁶⁴ as three `f32`s that hold it exactly (24, 24
/// and 16 bits), so it goes through a ring of samples unchanged.
fn put_u64(v: u64, out: &mut [f32]) {
    out[0] = (v & 0xFF_FFFF) as f32;
    out[1] = ((v >> 24) & 0xFF_FFFF) as f32;
    out[2] = (v >> 48) as f32;
}

fn get_u64(v: &[f32]) -> u64 {
    (v[0] as u64) | ((v[1] as u64) << 24) | ((v[2] as u64) << 48)
}

/// One block the tap queues for the cleaner: where it is, then the
/// microphone's and what plays, [`BLOCK`] samples each at the canceller's
/// rate.
struct Packet {
    /// The stream frame the block's first sample was handed to the device
    /// at (as [`crate::Output::position`] counts them).
    stream_frame: u64,
    /// The canceller's rate the block is at.
    rate: u32,
    /// The block follows the one before without a gap.
    continues: bool,
    mic: [f32; BLOCK],
    reference: [f32; BLOCK],
}

impl Packet {
    /// Its length in the ring.
    const LEN: usize = 5 + 2 * BLOCK;

    fn write(&self, out: &mut [f32; Packet::LEN]) {
        put_u64(self.stream_frame, &mut out[..3]);
        out[3] = self.rate as f32;
        out[4] = if self.continues { 1.0 } else { 0.0 };
        out[5..5 + BLOCK].copy_from_slice(&self.mic);
        out[5 + BLOCK..].copy_from_slice(&self.reference);
    }

    fn read(v: &[f32; Packet::LEN]) -> Packet {
        let mut p = Packet {
            stream_frame: get_u64(&v[..3]),
            rate: v[3] as u32,
            continues: v[4] != 0.0,
            mic: [0.0; BLOCK],
            reference: [0.0; BLOCK],
        };
        p.mic.copy_from_slice(&v[5..5 + BLOCK]);
        p.reference.copy_from_slice(&v[5 + BLOCK..]);
        p
    }
}

/// A block of the voice the cleaner gives ([`Input::take_voice`]).
#[derive(Clone, Debug, PartialEq)]
pub struct VoiceBlock {
    /// The stream frame (of the output it was heard with, as
    /// [`crate::Output::position`] counts them) the music of the moment
    /// its first sample was sung went to the device at.
    pub stream_frame: u64,
    /// The block's rate (the canceller's: the output's divided by a whole
    /// number, about 48 kHz).
    pub rate: u32,
    /// The block follows the one before without a gap.
    pub continues: bool,
    /// The voice, mono, full scale ±1: [`BLOCK`] samples.
    pub samples: Vec<f32>,
}

impl VoiceBlock {
    /// Its length in the ring.
    const LEN: usize = 5 + BLOCK;

    fn write(stream_frame: u64, rate: u32, continues: bool, samples: &[f32], out: &mut [f32]) {
        put_u64(stream_frame, &mut out[..3]);
        out[3] = rate as f32;
        out[4] = if continues { 1.0 } else { 0.0 };
        out[5..5 + BLOCK].copy_from_slice(samples);
    }

    fn read(v: &[f32]) -> VoiceBlock {
        VoiceBlock {
            stream_frame: get_u64(&v[..3]),
            rate: v[3] as u32,
            continues: v[4] != 0.0,
            samples: v[5..5 + BLOCK].to_vec(),
        }
    }
}

/// Lines an [`Input`] up with what an output plays, in its device thread:
/// the voice resampled to the output's rate, and beside it what the
/// buffer plays (its channels' mean), both brought down to the canceller's
/// rate (the output's divided by a whole number) and queued, block by
/// block, for the cleaner. It only reads the buffers. Hand it to
/// [`Output::set_tap`](crate::Output::set_tap).
///
/// Only one tap of an input consumes it at a time: another one (on an
/// output just reopened while the old one still runs) queues nothing until
/// the first is dropped.
pub struct MicTap {
    input: Input,
    claimed: bool,
    /// The current buffer gets voice.
    live: bool,
    steer: Steer,
    /// Voice at the output's rate, one chunk's worth.
    voice: Vec<f32>,
    /// Capture samples taken from the ring for one chunk.
    taken: Vec<f32>,
    /// The output's rate divided by this is the canceller's; 0: none fits
    /// (nothing is queued).
    factor: usize,
    /// The output rate it was set up for.
    rate: u32,
    /// The buffer's first stream frame, its stream frames, and the frame
    /// of it the next sample seen is.
    buffer_frame: u64,
    buffer_stream: usize,
    at: usize,
    /// Sums of the samples of the canceller's sample being made, and how
    /// many so far.
    mic_sum: f32,
    reference_sum: f32,
    phase: usize,
    /// The block being filled, and how far.
    packet: Box<Packet>,
    filled: usize,
    /// The next block follows the last one queued without a gap.
    continues: bool,
}

impl MicTap {
    fn new(input: Input) -> MicTap {
        MicTap {
            input,
            claimed: false,
            live: false,
            steer: Steer::new(),
            voice: vec![0.0; TAP_CHUNK],
            taken: vec![0.0; TAP_CHUNK * MAX_RATIO + 4],
            factor: 0,
            rate: 0,
            buffer_frame: 0,
            buffer_stream: 0,
            at: 0,
            mic_sum: 0.0,
            reference_sum: 0.0,
            phase: 0,
            packet: Box::new(Packet {
                stream_frame: 0,
                rate: 0,
                continues: false,
                mic: [0.0; BLOCK],
                reference: [0.0; BLOCK],
            }),
            filled: 0,
            continues: false,
        }
    }

    /// The input it lines up.
    pub fn input(&self) -> &Input {
        &self.input
    }

    fn claim(&mut self) -> bool {
        if !self.claimed {
            self.claimed = self
                .input
                .shared()
                .claimed
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok();
        }
        self.claimed
    }

    /// Set up for an output at `rate`.
    fn set_rate(&mut self, rate: u32) {
        if self.rate == rate {
            return;
        }
        self.rate = rate;
        let factor = (f64::from(rate) / f64::from(ECHO_RATE)).round().max(1.0) as usize;
        let work = rate as usize / factor;
        self.factor = if (16_000..=56_000).contains(&work) {
            factor
        } else {
            0
        };
        self.start_over();
    }

    /// Starts a new block: the next one queued does not follow the last.
    fn start_over(&mut self) {
        self.mic_sum = 0.0;
        self.reference_sum = 0.0;
        self.phase = 0;
        self.filled = 0;
        self.continues = false;
    }

    /// The stream frame the buffer's frame `at` is.
    fn frame_at(&self, at: usize) -> u64 {
        self.buffer_frame + at.min(self.buffer_stream) as u64
    }
}

impl Drop for MicTap {
    fn drop(&mut self) {
        if self.claimed {
            self.input.shared().claimed.store(false, Ordering::Release);
        }
    }
}

impl Tap for MicTap {
    fn begin(
        &mut self,
        frames: usize,
        stream_frame: u64,
        stream_frames: usize,
        format: StreamFormat,
    ) {
        let was = self.live;
        self.live = self.claim()
            && self
                .steer
                .begin(&self.input.inner.shared, format.rate, frames);
        self.set_rate(format.rate);
        self.buffer_frame = stream_frame;
        self.buffer_stream = stream_frames;
        self.at = 0;
        if self.live && !was {
            // The voice was off (starting, or the capture stalled): a new
            // run of blocks.
            self.start_over();
        }
    }

    fn see(&mut self, samples: &[i32], format: StreamFormat) {
        let ch = usize::from(format.channels).max(1);
        let frames = (samples.len() / ch).min(self.voice.len());
        if frames == 0 {
            return;
        }
        if !self.live || self.factor == 0 {
            self.at += frames;
            return;
        }
        let shared = &*self.input.inner.shared;
        let voice = &mut self.voice[..frames];
        self.steer.render(shared, voice, &mut self.taken);
        let scale = 1.0 / (8_388_608.0 * ch as f32);
        let rate = self.rate / self.factor as u32;
        for (frame, &mic) in samples.chunks_exact(ch).zip(self.voice[..frames].iter()) {
            if self.phase == 0 && self.filled == 0 {
                self.packet.stream_frame = self.frame_at(self.at);
            }
            self.at += 1;
            self.mic_sum += mic;
            self.reference_sum += frame.iter().map(|&s| s as f32).sum::<f32>() * scale;
            self.phase += 1;
            if self.phase < self.factor {
                continue;
            }
            let n = self.factor as f32;
            self.packet.mic[self.filled] = self.mic_sum / n;
            self.packet.reference[self.filled] = self.reference_sum / n;
            self.mic_sum = 0.0;
            self.reference_sum = 0.0;
            self.phase = 0;
            self.filled += 1;
            if self.filled == BLOCK {
                self.filled = 0;
                self.packet.rate = rate;
                self.packet.continues = self.continues;
                let mut raw = [0.0f32; Packet::LEN];
                self.packet.write(&mut raw);
                if shared.blocks.free() >= Packet::LEN {
                    shared.blocks.push(&raw);
                    self.continues = true;
                } else {
                    shared.blocks_dropped.fetch_add(1, Ordering::Relaxed);
                    self.continues = false;
                }
            }
        }
    }
}

/// Takes the music's echo out of the blocks a [`MicTap`] queues, off the
/// device thread: on the cleaner's thread of a device's [`Input`], or when
/// a [`MicFeed`]'s caller says.
struct Cleaner {
    canceller: Box<EchoCanceller>,
    /// The rate the canceller is set up for (0: none yet).
    rate: u32,
    /// The canceller's output brought to the isolator's rate.
    down: Downsampler,
    /// The singer isolated from what the canceller left ([`crate::isolate`]),
    /// for the meter.
    isolator: Box<Isolator>,
    /// Scratch: the block at 16 kHz, and isolated.
    at_16k: Vec<f32>,
    isolated: Vec<f32>,
    /// The meter's peak.
    peak: f32,
}

impl Cleaner {
    fn new() -> Cleaner {
        Cleaner {
            canceller: Box::new(EchoCanceller::new(ECHO_RATE)),
            rate: 0,
            down: Downsampler::new(ECHO_RATE),
            isolator: Box::new(Isolator::new()),
            at_16k: Vec::with_capacity(BLOCK),
            isolated: Vec::with_capacity(2 * BLOCK),
            peak: 0.0,
        }
    }

    /// Cleans one queued block; whether there was one.
    fn run(&mut self, shared: &MicShared) -> bool {
        if shared.blocks.len() < Packet::LEN {
            return false;
        }
        let mut raw = [0.0f32; Packet::LEN];
        shared.blocks.pop(&mut raw);
        let packet = Packet::read(&raw);
        if packet.rate != self.rate {
            // A new output: the echo's way is new.
            self.rate = packet.rate;
            self.canceller.reset();
            self.canceller.set_rate(packet.rate);
            self.down = Downsampler::new(packet.rate);
            self.isolator.reset();
            shared.echo_rate.store(packet.rate, Ordering::Relaxed);
        }
        let mut out = packet.mic;
        if shared.echo_on.load(Ordering::Relaxed) {
            self.canceller
                .process(&packet.mic, &packet.reference, &mut out);
        }
        let (frame, continues) = (packet.stream_frame, packet.continues);
        // The meter: the singer isolated from what the canceller left (the
        // band it missed, the room), falling as the capture's. A frame
        // comes every 10 ms, so a block may bring none: it only falls.
        self.down.process(&out, &mut self.at_16k);
        self.isolator.process(&self.at_16k, &mut self.isolated);
        self.at_16k.clear();
        let block_peak = self.isolated.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        self.isolated.clear();
        let decay = (-(BLOCK as f32) / packet.rate.max(1) as f32 / 0.3).exp();
        self.peak = block_peak.max(self.peak * decay);
        shared
            .voice_peak
            .store(f32_bits(self.peak), Ordering::Relaxed);
        shared
            .voice_at
            .store(shared.now_ns() + 1, Ordering::Relaxed);
        let status = self.canceller.status();
        shared
            .echo_removed
            .store(f32_bits(status.removed_db), Ordering::Relaxed);
        shared
            .echo_filter
            .store(f32_bits(status.filter_db), Ordering::Relaxed);
        shared.echo_learnt.store(status.learnt, Ordering::Relaxed);
        shared
            .echo_delay
            .store(f32_bits(status.delay.unwrap_or(-1.0)), Ordering::Relaxed);
        // The voice kept.
        let mut voice = [0.0f32; VoiceBlock::LEN];
        VoiceBlock::write(frame, packet.rate, continues, &out, &mut voice);
        if shared.voice.free() >= VoiceBlock::LEN {
            shared.voice.push(&voice);
        } else {
            shared.voice_dropped.fetch_add(1, Ordering::Relaxed);
        }
        if shared.recording.load(Ordering::Relaxed) {
            // The block as the canceller had it, whole or not at all.
            let mut frame = [0.0f32; RECORD_CHANNELS * BLOCK];
            for (i, f) in frame.chunks_exact_mut(RECORD_CHANNELS).enumerate() {
                f.copy_from_slice(&[packet.mic[i], packet.reference[i], out[i]]);
            }
            if shared.record.free() >= frame.len() {
                shared.record.push(&frame);
            }
        }
        true
    }
}

/// The varispeed resampler and the loop that steers it.
struct Steer {
    /// The rates it was set up for (capture, output); 0 before the first
    /// buffer.
    rates: (u32, u32),
    nominal: f64,
    adjust: f64,
    integral: f64,
    /// The smoothed fill, in capture frames; `None` until measured.
    smoothed: Option<f64>,
    /// Running (else waiting for the ring to fill to its target).
    primed: bool,
    /// Output frames seen since the rates were set, up to the warm-up.
    seen: u64,
    /// The input position of the next output sample, in 32.32 fixed
    /// point, between `history[1]` and `history[2]`.
    position: u64,
    step: u64,
    /// The four input samples around the position, oldest first.
    history: [f32; 4],
}

impl Steer {
    fn new() -> Steer {
        Steer {
            rates: (0, 0),
            nominal: 1.0,
            adjust: 0.0,
            integral: 0.0,
            smoothed: None,
            primed: false,
            seen: 0,
            position: 0,
            step: 1 << 32,
            history: [0.0; 4],
        }
    }

    fn set_step(&mut self) {
        let ratio = self.nominal * (1.0 + self.adjust);
        self.step = (ratio * 4_294_967_296.0).round() as u64;
    }

    /// Capture samples `frames` output frames consume from here.
    fn need(&self, frames: usize) -> usize {
        ((self.position + frames as u64 * self.step) >> 32) as usize
    }

    /// Before a buffer of `frames` at `out_rate`: steers the ratio on the
    /// fill this buffer will leave, and says whether the ring holds the
    /// whole buffer's voice (else the buffer gets none).
    fn begin(&mut self, shared: &MicShared, out_rate: u32, frames: usize) -> bool {
        let in_rate = shared.rate.load(Ordering::Acquire);
        if in_rate == 0 || out_rate == 0 {
            return false;
        }
        if self.rates != (in_rate, out_rate) {
            *self = Steer::new();
            self.rates = (in_rate, out_rate);
            self.nominal = f64::from(in_rate) / f64::from(out_rate);
            self.set_step();
        }
        if self.nominal * (1.0 + 2.0 * MAX_ADJUST) >= MAX_RATIO as f64 {
            return false;
        }
        let packet = shared.packet.load(Ordering::Relaxed);
        let period = shared.period.load(Ordering::Relaxed).max(packet).max(1);
        let target = 2 * period as usize;
        shared.target.store(target as u32, Ordering::Relaxed);
        let ring = &shared.ring;
        let queued = ring.len();
        let arriving = self.arriving(shared, packet);
        // What to keep queued for the fill to be at its target after this
        // buffer.
        let keep = (target + self.need(frames)) as f64 - arriving;
        let keep = (keep.round().max(0.0) as usize).max(self.need(frames));
        if !self.primed {
            let warm = |rate: u32| rate as u64 * WARM_UP.as_millis() as u64 / 1000;
            let first_at = shared.first_at.load(Ordering::Relaxed);
            let captured = if first_at == 0 {
                u64::MAX
            } else {
                shared.now_ns().saturating_sub(first_at - 1)
            };
            let output_warm = self.seen >= warm(out_rate);
            self.seen = (self.seen + frames as u64).min(warm(out_rate));
            if !output_warm || captured < WARM_UP.as_nanos() as u64 || queued < keep {
                return false;
            }
            ring.skip(queued - keep);
            self.primed = true;
            self.smoothed = None;
            return true;
        }
        let need = self.need(frames);
        if queued < need {
            // The capture stalled: no voice until the target is queued
            // again. The loop holds its estimate of the clocks meanwhile.
            shared.underruns.fetch_add(1, Ordering::Relaxed);
            self.primed = false;
            self.history = [0.0; 4];
            self.position = 0;
            return false;
        }
        if queued > 4 * target + 2 * need {
            // Far too much queued (the device thread stalled): skip to the
            // target.
            ring.skip(queued - keep);
            shared.resyncs.fetch_add(1, Ordering::Relaxed);
            self.smoothed = None;
        } else {
            let fill = queued as f64 - need as f64 + arriving;
            self.steer(shared, frames, target, fill);
            // Held well over the target (a burst of capture, as devices
            // deliver when they start): skip the excess rather than drain
            // it at the loop's limit for seconds.
            let over = self.smoothed.unwrap_or(fill) - target as f64;
            let excess = (fill - target as f64).round();
            if over > f64::from(period) && excess >= 1.0 {
                ring.skip(excess as usize);
                self.smoothed = Some(fill - excess);
                shared.resyncs.fetch_add(1, Ordering::Relaxed);
            }
        }
        // A faster ratio may need a sample more than is queued.
        ring.len() >= self.need(frames)
    }

    /// Fills `voice` with the capture resampled; the buffer's
    /// [`Steer::begin`] made sure the ring holds it.
    fn render(&mut self, shared: &MicShared, voice: &mut [f32], taken: &mut [f32]) {
        let need = self.need(voice.len()).min(taken.len());
        let got = shared.ring.pop(&mut taken[..need]);
        let mut next = taken[..got].iter();
        for out in voice.iter_mut() {
            let t = (self.position & 0xFFFF_FFFF) as f32 / 4_294_967_296.0;
            *out = catmull_rom(&self.history, t);
            self.position = (self.position & 0xFFFF_FFFF) + self.step;
            for _ in 0..(self.position >> 32) {
                let x = next.next().copied().unwrap_or(0.0);
                self.history = [self.history[1], self.history[2], self.history[3], x];
            }
        }
        self.position &= 0xFFFF_FFFF;
    }

    /// What to add to the queue's length for the fill as if the voice
    /// arrived continuously: what has come in since the last packet, minus
    /// half a packet (so the fill means what is queued on average).
    fn arriving(&self, shared: &MicShared, packet: u32) -> f64 {
        let pushed_at = shared.pushed_at.load(Ordering::Acquire);
        let since = if pushed_at == 0 {
            0.0
        } else {
            let ns = shared.now_ns().saturating_sub(pushed_at - 1);
            (ns as f64 * 1e-9 * f64::from(self.rates.0)).min(f64::from(packet))
        };
        since - f64::from(packet) / 2.0
    }

    /// Moves the ratio on `fill`, what a buffer of `frames` will leave
    /// queued (as [`Steer::arriving`] counts it).
    fn steer(&mut self, shared: &MicShared, frames: usize, target: usize, fill: f64) {
        let (in_rate, out_rate) = self.rates;
        let fin = f64::from(in_rate);
        let dt = frames as f64 / f64::from(out_rate);
        let smoothed = match self.smoothed {
            None => fill,
            Some(s) => s + (fill - s) * (dt / SMOOTHING).min(1.0),
        };
        self.smoothed = Some(smoothed);
        let error = (smoothed - target as f64) / fin;
        // The integral is the estimate of the clocks' offset: it does not
        // wind up while the ratio is held at its limit.
        let integral = self.integral + error * dt;
        let wanted = KP * error + KI * integral;
        if wanted.abs() <= MAX_ADJUST || (wanted > 0.0) != (error > 0.0) {
            self.integral = integral;
        }
        self.adjust = (KP * error + KI * self.integral).clamp(-MAX_ADJUST, MAX_ADJUST);
        self.set_step();
        shared.fill.store(smoothed.to_bits(), Ordering::Relaxed);
        shared
            .adjust
            .store(self.adjust.to_bits(), Ordering::Relaxed);
    }
}

/// Catmull-Rom between `h[1]` and `h[2]` at `t` in [0, 1).
fn catmull_rom(h: &[f32; 4], t: f32) -> f32 {
    let [a, b, c, d] = *h;
    let c1 = 0.5 * (c - a);
    let c2 = a - 2.5 * b + 2.0 * c - 0.5 * d;
    let c3 = 0.5 * (d - a) + 1.5 * (b - c);
    ((c3 * t + c2) * t + c1) * t + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_recording_has_the_header_players_read() {
        let h = wav_header(3, 44_100, 1000);
        assert_eq!(h.len(), WAV_HEADER);
        let u16_at = |i: usize| u16::from_le_bytes([h[i], h[i + 1]]);
        let u32_at = |i: usize| u32::from_le_bytes([h[i], h[i + 1], h[i + 2], h[i + 3]]);
        assert_eq!(&h[0..4], b"RIFF");
        // The file's length less 8: the header's 72 and the samples'.
        assert_eq!(u32_at(4), 72 + 1000 * 12);
        assert_eq!(&h[8..16], b"WAVEfmt ");
        assert_eq!(u32_at(16), 40);
        assert_eq!(u16_at(20), 0xFFFE);
        assert_eq!((u16_at(22), u32_at(24)), (3, 44_100));
        assert_eq!((u32_at(28), u16_at(32), u16_at(34)), (44_100 * 12, 12, 32));
        assert_eq!((u16_at(36), u16_at(38)), (22, 32));
        // Front left, front right, centre.
        assert_eq!(u32_at(40), 0b111);
        // The float sub-format, then the frames in `fact`, then the data.
        assert_eq!(&h[44..48], &[3, 0, 0, 0]);
        assert_eq!(&h[56..60], &[0x00, 0x38, 0x9B, 0x71]);
        assert_eq!(&h[60..64], b"fact");
        assert_eq!((u32_at(64), u32_at(68)), (4, 1000));
        assert_eq!(&h[72..76], b"data");
        assert_eq!(u32_at(76), 1000 * 12);
        // A file of it, for a reader of its own (`JODEL_WAV_SAMPLE`).
        if let Some(out) = std::env::var_os("JODEL_WAV_SAMPLE") {
            let mut bytes = h.clone();
            for i in 0..3000u32 {
                bytes.extend_from_slice(&((i % 3) as f32 * 0.25).to_le_bytes());
            }
            std::fs::write(out, bytes).unwrap();
        }
    }

    #[test]
    fn ring_wraps_and_counts() {
        let ring = FloatRing::new(5);
        assert_eq!(ring.push(&[1.0, 2.0, 3.0, 4.0]), 4);
        let mut out = [0.0; 3];
        assert_eq!(ring.pop(&mut out), 3);
        assert_eq!(out, [1.0, 2.0, 3.0]);
        assert_eq!(ring.push(&[5.0, 6.0, 7.0, 8.0, 9.0]), 4);
        assert_eq!(ring.len(), 5);
        assert_eq!(ring.skip(2), 2);
        let mut out = [0.0; 8];
        assert_eq!(ring.pop(&mut out), 3);
        assert_eq!(out[..3], [6.0, 7.0, 8.0]);
    }

    #[test]
    fn wires_read_full_scale() {
        assert_eq!(Wire::I16.read(&i16::MIN.to_le_bytes()), -1.0);
        assert_eq!(Wire::I24.read(&[0, 0, 0x80]), -1.0);
        assert_eq!(Wire::I24.read(&[0, 0, 0x40]), 0.5);
        assert_eq!(Wire::I32.read(&(1i32 << 30).to_le_bytes()), 0.5);
        assert_eq!(Wire::I24Low.read(&[0, 0, 0x40, 0x00]), 0.5);
        assert_eq!(Wire::I24Low.read(&[0, 0, 0xC0, 0xFF]), -0.5);
        assert_eq!(Wire::F32.read(&0.25f32.to_le_bytes()), 0.25);
    }

    #[test]
    fn mono_mixdowns() {
        let shared = MicShared::new(48_000, 240, true);
        let frames: Vec<u8> = [0.5f32, -0.25, 0.1, 1.0, 0.0, 0.3]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut out = [0.0; 4];
        for (mono, channels, want) in [
            (Mono::Auto, 2, vec![0.125, 0.55, 0.15]),
            (Mono::Auto, 3, vec![0.5, 1.0]),
            (Mono::First, 2, vec![0.5, 0.1, 0.0]),
            (Mono::Channel(1), 2, vec![-0.25, 1.0, 0.3]),
            (Mono::Channel(7), 2, vec![0.5, 0.1, 0.0]),
            (Mono::Mean, 3, vec![0.125, 0.5]),
        ] {
            let mut p = Producer::new(shared.clone(), mono);
            p.push(&frames, Wire::F32, channels, 6 / channels);
            let n = shared.ring.pop(&mut out);
            assert_eq!(n, want.len(), "{mono:?} {channels}");
            for (a, b) in out[..n].iter().zip(&want) {
                assert!((a - b).abs() < 1e-6, "{mono:?} {channels}: {a} vs {b}");
            }
        }
        let mut p = Producer::new(shared.clone(), Mono::Auto);
        let planes: [&[u8]; 2] = [&frames[..8], &frames[8..16]];
        p.push_planar(&planes, Wire::F32, 2);
        assert_eq!(shared.ring.pop(&mut out), 2);
        assert!((out[0] - 0.3).abs() < 1e-6 && (out[1] - 0.375).abs() < 1e-6);
    }

    #[test]
    fn catmull_rom_is_exact_on_its_points_and_on_quadratics() {
        let f = |x: f32| 0.1 * x * x * x - 0.3 * x * x + 0.2 * x + 0.05;
        let h = [f(-1.0), f(0.0), f(1.0), f(2.0)];
        assert_eq!(catmull_rom(&h, 0.0), h[1]);
        // Catmull-Rom reproduces quadratics exactly.
        let q = |x: f32| 0.4 * x * x - 0.2 * x + 0.1;
        let hq = [q(-1.0), q(0.0), q(1.0), q(2.0)];
        for t in [0.25f32, 0.5, 0.75] {
            assert!((catmull_rom(&hq, t) - q(t)).abs() < 1e-6);
        }
    }

    #[test]
    fn same_rates_pass_the_voice_through_unchanged() {
        let shared = MicShared::new(48_000, 240, true);
        let mut steer = Steer::new();
        let ramp: Vec<f32> = (0..48_000).map(|i| (i as f32 * 0.01).sin()).collect();
        shared.ring.push(&ramp);
        shared.packet.store(240, Ordering::Relaxed);
        let mut taken = vec![0.0; 4096 * MAX_RATIO];
        let mut voice = vec![0.0; 480];
        // The output's warm-up first, consuming nothing.
        for _ in 0..10 {
            assert!(!steer.begin(&shared, 48_000, 480));
        }
        assert!(steer.begin(&shared, 48_000, 480));
        steer.render(&shared, &mut voice, &mut taken);
        // The first buffer runs at the nominal ratio, three samples late
        // (the interpolator's history), from the target (and half a packet,
        // with none arriving) before the end.
        let start = 48_000 - (480 + 480 + 120);
        for k in 3..480 {
            assert_eq!(voice[k], ramp[start + k - 3], "{k}");
        }
    }
}
