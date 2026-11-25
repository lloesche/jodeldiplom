//! The capture side: what all sides share ([`MicShared`], grouped by the
//! side that writes it), an input as a backend opens it ([`Opened`]), and
//! the end of the capture thread that mixes packets down to mono
//! ([`Producer`]).

use super::clean::{Cleaner, Voice, VOICE_QUEUE};
use super::record::{RecordBlock, RECORD_QUEUE};
use super::tap::{Packet, BLOCK_QUEUE};
use super::{Mono, Pick};
use crate::error::Error;
use crate::spsc::{self, Spsc};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

/// The clock an input's times are on.
pub(crate) enum MicClock {
    /// The system's, from when the input opened.
    System(Instant),
    /// The caller's ([`super::MicFeed::set_time`]), in ns.
    Caller(AtomicU64),
}

impl MicClock {
    pub(crate) fn system() -> MicClock {
        MicClock::System(Instant::now())
    }

    pub(super) fn caller() -> MicClock {
        MicClock::Caller(AtomicU64::new(0))
    }
}

/// What the capture writes (its thread, and on macOS the rate listener):
/// the ring of mono samples, and how they come.
pub(super) struct Capture {
    /// The voice, mono; the clock bridge takes the consumer.
    pub(super) ring: Arc<Spsc<f32>>,
    pub(super) rate: AtomicU32,
    pub(super) period: AtomicU32,
    /// The largest packet seen, in frames.
    pub(super) packet: AtomicU32,
    /// When the last packet was queued, in ns on the input's clock, plus
    /// one (0: never).
    pub(super) pushed_at: AtomicU64,
    /// When the first packet was queued, the same way.
    pub(super) first_at: AtomicU64,
    /// The recent peak, as `f32` bits.
    pub(super) peak: AtomicU32,
    /// Anything but exact zeros has come in.
    pub(super) signal: AtomicBool,
    /// Frames lost to a full ring while the bridge read it.
    pub(super) overruns: AtomicU64,
}

/// What the tap writes: how the clock bridge runs, and the blocks it
/// queues for the cleaner.
pub(super) struct Bridge {
    /// The blocks for the cleaner; the consuming tap takes the producer.
    pub(super) blocks: Arc<Spsc<Packet>>,
    /// The bridge reads the ring: it is primed.
    pub(super) reading: AtomicBool,
    /// The smoothed fill and the correction, as `f64` bits.
    pub(super) fill: AtomicU64,
    pub(super) target: AtomicU32,
    pub(super) adjust: AtomicU64,
    pub(super) underruns: AtomicU64,
    pub(super) resyncs: AtomicU64,
    /// Blocks the cleaner had no room for.
    pub(super) blocks_dropped: AtomicU64,
}

/// What the cleaner writes: the voice it gives, its meter, the canceller's
/// status.
pub(super) struct Cleaned {
    /// The voice, block by block, for [`super::Input::take_voice`], which
    /// takes the consumer; blocks it had no room for.
    pub(super) voice: Arc<Spsc<Voice>>,
    pub(super) voice_dropped: AtomicU64,
    /// The voice's peak after the canceller, as `f32` bits; and when it was
    /// last set (0: never), as `Capture::pushed_at`.
    pub(super) voice_peak: AtomicU32,
    pub(super) voice_at: AtomicU64,
    /// The echo removed in dB by both stages and by the filter (`f32`
    /// bits), whether it has learnt the echo's way, and the echo's delay in
    /// seconds (`f32` bits; negative: not known).
    pub(super) echo_removed: AtomicU32,
    pub(super) echo_filter: AtomicU32,
    pub(super) echo_learnt: AtomicBool,
    pub(super) echo_delay: AtomicU32,
    /// The rate the canceller runs at (0: none yet).
    pub(super) echo_rate: AtomicU32,
    /// What it works with, for a recording ([`super::Input::record_echo`]):
    /// one is under way while someone holds the consumer.
    pub(super) record: Arc<Spsc<RecordBlock>>,
}

pub(crate) struct MicShared {
    pub(super) clock: MicClock,
    pub(super) capture: Capture,
    pub(super) bridge: Bridge,
    pub(super) cleaned: Cleaned,
    /// The echo canceller is on ([`super::Input::set_echo_cancel`]).
    pub(super) echo_on: AtomicBool,
    /// Why capture stopped.
    failure: OnceLock<Error>,
}

impl MicShared {
    pub(super) fn now_ns(&self) -> u64 {
        match &self.clock {
            MicClock::System(epoch) => epoch.elapsed().as_nanos() as u64,
            MicClock::Caller(now) => now.load(Ordering::Acquire),
        }
    }

    /// Sets the caller's clock to `t` (the system's is not set).
    pub(super) fn set_time(&self, t: Duration) {
        if let MicClock::Caller(now) = &self.clock {
            now.store(t.as_nanos() as u64, Ordering::Release);
        }
    }

    /// The capture rate changed under the stream (macOS: the device's
    /// nominal rate, when it is also the output).
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(crate) fn set_rate(&self, rate: u32) {
        self.capture.rate.store(rate, Ordering::Release);
    }

    /// The capture failed; capture stops and [`super::Input::failure`] says
    /// why (the first failure).
    pub(crate) fn fail(&self, error: Error) {
        let _ = self.failure.set(error);
    }

    pub(super) fn failure(&self) -> Option<Error> {
        self.failure.get().cloned()
    }

    /// Whether the capture failed.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(crate) fn failed(&self) -> bool {
        self.failure.get().is_some()
    }
}

/// An input as a backend opens it: what becomes the [`super::Input`]
/// ([`super::Input::new`]), beside the capture thread's [`Producer`].
pub(crate) struct Opened {
    pub(super) shared: Arc<MicShared>,
    pub(super) cleaner: Cleaner,
}

/// Why claiming an end of a queue just made cannot fail.
const FRESH: &str = "a new queue's ends are free";

impl Opened {
    /// An input capturing at `rate` in packets of about `period` frames,
    /// mixed down as `mono`, on `clock`.
    pub(crate) fn new(rate: u32, period: u32, mono: Mono, clock: MicClock) -> (Producer, Opened) {
        // A second of voice: far more than the loop ever holds.
        let ring = Spsc::new((rate as usize).max(4096));
        let blocks = Spsc::new(BLOCK_QUEUE);
        let voice = Spsc::new(VOICE_QUEUE);
        let record = Spsc::new(RECORD_QUEUE);
        // The ends that stay with one owner are claimed as the queues are
        // made; the others by whoever comes (a tap, a taker, a recording).
        let capture_end = ring.producer().expect(FRESH);
        let cleaner = Cleaner::new(
            blocks.consumer().expect(FRESH),
            voice.producer().expect(FRESH),
            record.producer().expect(FRESH),
        );
        let shared = Arc::new(MicShared {
            clock,
            capture: Capture {
                ring,
                rate: AtomicU32::new(rate),
                period: AtomicU32::new(period.max(1)),
                packet: AtomicU32::new(0),
                pushed_at: AtomicU64::new(0),
                first_at: AtomicU64::new(0),
                peak: AtomicU32::new(0),
                signal: AtomicBool::new(false),
                overruns: AtomicU64::new(0),
            },
            bridge: Bridge {
                blocks,
                reading: AtomicBool::new(false),
                fill: AtomicU64::new(0),
                target: AtomicU32::new(0),
                adjust: AtomicU64::new(0),
                underruns: AtomicU64::new(0),
                resyncs: AtomicU64::new(0),
                blocks_dropped: AtomicU64::new(0),
            },
            cleaned: Cleaned {
                voice,
                voice_dropped: AtomicU64::new(0),
                voice_peak: AtomicU32::new(0),
                voice_at: AtomicU64::new(0),
                echo_removed: AtomicU32::new(0),
                echo_filter: AtomicU32::new(0),
                echo_learnt: AtomicBool::new(false),
                echo_delay: AtomicU32::new((-1.0f32).to_bits()),
                echo_rate: AtomicU32::new(0),
                record,
            },
            echo_on: AtomicBool::new(true),
            failure: OnceLock::new(),
        });
        let producer = Producer {
            shared: Arc::clone(&shared),
            ring: capture_end,
            mono,
            scratch: vec![0.0; 4096],
        };
        (producer, Opened { shared, cleaner })
    }
}

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
    ring: spsc::Producer<f32>,
    mono: Mono,
    scratch: Vec<f32>,
}

impl Producer {
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
        let c = &s.capture;
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
            let queued = self.ring.push_slice(&self.scratch[..n]);
            // Lost only while the bridge reads the voice: before it does
            // (the microphone opened while the output is still being set
            // up, or warming up), the ring fills with the first second and
            // drops what comes after; the bridge starts from the newest of
            // what it holds (a couple of capture periods, then the voice as
            // it comes).
            if queued < n && s.bridge.reading.load(Ordering::Acquire) {
                c.overruns.fetch_add((n - queued) as u64, Ordering::Relaxed);
            }
            done += n;
        }
        let now = s.now_ns() + 1;
        c.pushed_at.store(now, Ordering::Release);
        if c.first_at.load(Ordering::Relaxed) == 0 {
            c.first_at.store(now, Ordering::Relaxed);
        }
        c.packet.fetch_max(frames as u32, Ordering::Relaxed);
        if peak > 0.0 {
            c.signal.store(true, Ordering::Relaxed);
        }
        // The meter falls by about 30 dB a second.
        let rate = c.rate.load(Ordering::Relaxed).max(1);
        let decay = (-(frames as f32) / rate as f32 / 0.3).exp();
        let old = f32::from_bits(c.peak.load(Ordering::Relaxed));
        c.peak
            .store(peak.max(old * decay).to_bits(), Ordering::Relaxed);
    }

    /// The room left in the ring.
    pub(super) fn free(&self) -> usize {
        self.ring.free()
    }

    pub(crate) fn shared(&self) -> &Arc<MicShared> {
        &self.shared
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            let (mut p, opened) = Opened::new(48_000, 240, mono, MicClock::caller());
            let mut ring = opened.shared.capture.ring.consumer().unwrap();
            p.push(&frames, Wire::F32, channels, 6 / channels);
            let n = ring.pop_slice(&mut out);
            assert_eq!(n, want.len(), "{mono:?} {channels}");
            for (a, b) in out[..n].iter().zip(&want) {
                assert!((a - b).abs() < 1e-6, "{mono:?} {channels}: {a} vs {b}");
            }
        }
        let (mut p, opened) = Opened::new(48_000, 240, Mono::Auto, MicClock::caller());
        let mut ring = opened.shared.capture.ring.consumer().unwrap();
        let planes: [&[u8]; 2] = [&frames[..8], &frames[8..16]];
        p.push_planar(&planes, Wire::F32, 2);
        assert_eq!(ring.pop_slice(&mut out), 2);
        assert!((out[0] - 0.3).abs() < 1e-6 && (out[1] - 0.375).abs() < 1e-6);
    }
}
