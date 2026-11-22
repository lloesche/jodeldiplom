//! The cleaner, off the device thread: the echo canceller with what played
//! as its sidechain, then the singer isolated for the meter; the voice it
//! gives is queued as [`VoiceBlock`]s.

use super::capture::{f32_bits, MicShared};
use super::record::RECORD_CHANNELS;
use super::tap::{get_u64, put_u64, Packet, ECHO_RATE};
use crate::aec::{EchoCanceller, BLOCK};
use crate::isolate::{Downsampler, Isolator};
use std::sync::atomic::Ordering;

/// Cleaned blocks kept for [`Input::take_voice`] (about 11 s at 48 kHz).
pub(super) const VOICE_QUEUE: usize = 4096;

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
    pub(super) const LEN: usize = 5 + BLOCK;

    fn write(stream_frame: u64, rate: u32, continues: bool, samples: &[f32], out: &mut [f32]) {
        put_u64(stream_frame, &mut out[..3]);
        out[3] = rate as f32;
        out[4] = if continues { 1.0 } else { 0.0 };
        out[5..5 + BLOCK].copy_from_slice(samples);
    }

    pub(super) fn read(v: &[f32]) -> VoiceBlock {
        VoiceBlock {
            stream_frame: get_u64(&v[..3]),
            rate: v[3] as u32,
            continues: v[4] != 0.0,
            samples: v[5..5 + BLOCK].to_vec(),
        }
    }
}

/// Takes the music's echo out of the blocks a [`MicTap`] queues, off the
/// device thread: on the cleaner's thread of a device's [`Input`], or when
/// a [`MicFeed`]'s caller says.
pub(super) struct Cleaner {
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
    /// The last block was lost to a full queue: the next one queued does
    /// not follow the one before it.
    dropped: bool,
}

impl Cleaner {
    pub(super) fn new() -> Cleaner {
        Cleaner {
            canceller: Box::new(EchoCanceller::new(ECHO_RATE)),
            rate: 0,
            down: Downsampler::new(ECHO_RATE),
            isolator: Box::new(Isolator::new()),
            at_16k: Vec::with_capacity(BLOCK),
            isolated: Vec::with_capacity(2 * BLOCK),
            peak: 0.0,
            dropped: false,
        }
    }

    /// Cleans one queued block; whether there was one.
    pub(super) fn run(&mut self, shared: &MicShared) -> bool {
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
        // The voice kept. After a block lost, the next does not follow
        // the one before it (which may have started a run: whoever reads
        // the voice must see the gap).
        let mut voice = [0.0f32; VoiceBlock::LEN];
        let continues = continues && !self.dropped;
        VoiceBlock::write(frame, packet.rate, continues, &out, &mut voice);
        self.dropped = shared.voice.free() < VoiceBlock::LEN;
        if self.dropped {
            shared.voice_dropped.fetch_add(1, Ordering::Relaxed);
        } else {
            shared.voice.push(&voice);
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
