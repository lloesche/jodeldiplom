//! The capture side: the ring of `f32` the capture thread fills, what it
//! shares with the tap and the cleaner ([`MicShared`]), and the end of the
//! capture thread that mixes packets down to mono ([`Producer`]).

use super::clean::{VoiceBlock, VOICE_QUEUE};
use super::record::RECORD_QUEUE;
use super::tap::{Packet, BLOCK_QUEUE};
use super::{Mono, Pick};
use crate::error::Error;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

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

pub(super) fn f32_bits(v: f32) -> u32 {
    v.to_bits()
}

pub(crate) struct MicShared {
    pub(super) ring: FloatRing,
    pub(super) rate: AtomicU32,
    pub(super) period: AtomicU32,
    pub(super) packet: AtomicU32,
    pub(super) epoch: Instant,
    /// Fed by the caller ([`MicFeed`]): no capture thread, and its cleaner
    /// run by the caller.
    pub(super) simulated: bool,
    /// The clock is the caller's ([`MicFeed::set_time`]), not the system's.
    pub(super) caller_clock: bool,
    pub(super) sim_now: AtomicU64,
    /// When the last packet was queued, in ns from `epoch`, plus one (0:
    /// never).
    pub(super) pushed_at: AtomicU64,
    /// When the first packet was queued, the same way.
    pub(super) first_at: AtomicU64,
    pub(super) peak: AtomicU32,
    /// Anything but exact zeros has come in.
    pub(super) signal: AtomicBool,
    /// The echo canceller is on ([`Input::set_echo_cancel`]).
    pub(super) echo_on: AtomicBool,
    /// What the tap queues for the cleaner, block by block (see
    /// [`Packet`]): the microphone and what plays, side by side.
    pub(super) blocks: FloatRing,
    /// Blocks the tap had no room for.
    pub(super) blocks_dropped: AtomicU64,
    /// The voice the cleaner gives, block by block ([`VoiceBlock`]), for
    /// [`Input::take_voice`]; blocks it had no room for.
    pub(super) voice: FloatRing,
    pub(super) voice_dropped: AtomicU64,
    /// The voice's peak after the canceller, as `f32` bits, falling like
    /// `peak`; and when it was last set (0: never), in ns from `epoch`.
    pub(super) voice_peak: AtomicU32,
    pub(super) voice_at: AtomicU64,
    /// The canceller's status: the echo removed in dB by both stages and
    /// by the filter (`f32` bits), whether it has learnt the echo's way,
    /// and the echo's delay in seconds (`f32` bits; negative: not known).
    pub(super) echo_removed: AtomicU32,
    pub(super) echo_filter: AtomicU32,
    pub(super) echo_learnt: AtomicBool,
    pub(super) echo_delay: AtomicU32,
    /// A recording of the canceller under way ([`Input::record_echo`]):
    /// its blocks (the microphone, what plays, the voice it gives:
    /// interleaved) queue here for the thread that writes them; the rate
    /// the canceller runs at.
    pub(super) recording: AtomicBool,
    pub(super) record: FloatRing,
    pub(super) echo_rate: AtomicU32,
    /// Tells the cleaner's thread to stop.
    pub(super) stop: AtomicBool,
    pub(super) fill: AtomicU64,
    pub(super) target: AtomicU32,
    pub(super) adjust: AtomicU64,
    pub(super) underruns: AtomicU64,
    pub(super) overruns: AtomicU64,
    pub(super) resyncs: AtomicU64,
    /// An insert holds the consumer side of the ring.
    pub(super) claimed: AtomicBool,
    pub(super) failed: AtomicBool,
    pub(super) failure: Mutex<Option<Error>>,
}

impl MicShared {
    pub(crate) fn new(rate: u32, period: u32, simulated: bool) -> Arc<MicShared> {
        Self::build(rate, period, simulated, simulated)
    }

    /// Fed by the caller, on the system's clock.
    pub(super) fn fed_in_real_time(rate: u32, period: u32) -> Arc<MicShared> {
        Self::build(rate, period, true, false)
    }

    fn build(rate: u32, period: u32, simulated: bool, caller_clock: bool) -> Arc<MicShared> {
        // A second of voice: far more than the loop ever holds.
        let capacity = (rate as usize).max(4096);
        Arc::new(MicShared {
            ring: FloatRing::new(capacity),
            rate: AtomicU32::new(rate),
            period: AtomicU32::new(period.max(1)),
            packet: AtomicU32::new(0),
            epoch: Instant::now(),
            simulated,
            caller_clock,
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

    pub(super) fn now_ns(&self) -> u64 {
        if self.caller_clock {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
