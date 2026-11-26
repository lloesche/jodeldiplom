//! The cleaner (the microphone's processing chain), off the device
//! thread: the echo canceller with what played
//! as its sidechain, the leftover of the echo suppressed after it, then the
//! singer isolated for the meter; the voice it gives (suppressed: as the
//! game hears it) is queued block by block for [`super::Input::take_voice`].

use super::capture::MicShared;
use super::record::{RecordBlock, RECORD_CHANNELS};
use super::tap::{Packet, Stamp, ECHO_RATE};
use crate::aec::{EchoCanceller, Suppressor, BLOCK};
use crate::isolate::{Downsampler, Isolator};
use crate::spsc;
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

/// A block of the voice as it is queued: [`VoiceBlock`] without the heap.
#[derive(Clone, Copy)]
pub(super) struct Voice {
    stamp: Stamp,
    samples: [f32; BLOCK],
}

impl Voice {
    pub(super) fn block(&self) -> VoiceBlock {
        VoiceBlock {
            stream_frame: self.stamp.stream_frame,
            rate: self.stamp.rate,
            continues: self.stamp.continues,
            samples: self.samples.to_vec(),
        }
    }
}

/// A block the suppressor holds (it gives each out one block late): what
/// went in, and what came out of the canceller.
#[derive(Clone, Copy)]
struct Held {
    packet: Packet,
    cancelled: [f32; BLOCK],
}

/// Takes the music's echo out of the blocks a [`MicTap`] queues, off the
/// device thread: on the cleaner's thread of a device's [`Input`], or when
/// a [`MicFeed`]'s caller says.
pub(super) struct Cleaner {
    /// The tap's blocks in; the voice out; what a recording keeps.
    blocks: spsc::Consumer<Packet>,
    voice: spsc::Producer<Voice>,
    record: spsc::Producer<RecordBlock>,
    canceller: Box<EchoCanceller>,
    /// What the canceller's filter leaves of the echo, taken down
    /// ([`Suppressor`]); the block it holds (what comes out of the
    /// suppressor is that block's): the one record of its delay.
    suppressor: Box<Suppressor>,
    held: Option<Held>,
    /// The rate the canceller is set up for (0: none yet).
    rate: u32,
    /// The voice brought to the isolator's rate.
    down: Downsampler,
    /// The singer isolated from the voice ([`crate::isolate`]), for the
    /// meter.
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
    pub(super) fn new(
        blocks: spsc::Consumer<Packet>,
        voice: spsc::Producer<Voice>,
        record: spsc::Producer<RecordBlock>,
    ) -> Cleaner {
        Cleaner {
            blocks,
            voice,
            record,
            canceller: Box::new(EchoCanceller::new(ECHO_RATE)),
            suppressor: Box::new(Suppressor::new(ECHO_RATE)),
            held: None,
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
        let Some(packet) = self.blocks.pop() else {
            return false;
        };
        let stamp = packet.stamp;
        if stamp.rate != self.rate {
            // A new output: the echo's way is new.
            self.held = None;
            self.rate = stamp.rate;
            self.canceller.reset();
            self.canceller.set_rate(stamp.rate);
            self.suppressor.reset();
            self.suppressor.set_rate(stamp.rate);
            self.down = Downsampler::new(stamp.rate);
            self.isolator.reset();
            shared
                .cleaned
                .echo_rate
                .store(stamp.rate, Ordering::Relaxed);
        } else if !stamp.continues {
            // A gap: the block held is dropped (under 3 ms; given out with
            // the next run's first, it would come after a hand-over to a
            // new Lead, told in the old output's frames).
            self.held = None;
            self.suppressor.restart();
        }
        let mut cancelled = packet.mic;
        let mut voice = [0.0f32; BLOCK];
        if shared.echo_on.load(Ordering::Relaxed) {
            let echo =
                self.canceller
                    .process(&packet.mic, &packet.left, &packet.right, &mut cancelled);
            self.suppressor.process(&cancelled, &echo, &mut voice);
        } else {
            self.suppressor.pass(&cancelled, &mut voice);
        }
        let status = self.canceller.status();
        let c = &shared.cleaned;
        c.echo_removed
            .store(status.removed_db.to_bits(), Ordering::Relaxed);
        c.echo_filter
            .store(status.filter_db.to_bits(), Ordering::Relaxed);
        c.echo_learnt.store(status.learnt, Ordering::Relaxed);
        c.echo_delay
            .store(status.delay.unwrap_or(-1.0).to_bits(), Ordering::Relaxed);
        let now = Held { packet, cancelled };
        // What the suppressor gave is the block held's (none after a gap).
        if let Some(before) = self.held.replace(now) {
            self.give(shared, &before, &voice);
        }
        true
    }

    /// The voice of the block `held`: to the meter, the queue, and a
    /// recording under way.
    fn give(&mut self, shared: &MicShared, held: &Held, voice: &[f32; BLOCK]) {
        let c = &shared.cleaned;
        let stamp = held.packet.stamp;
        // The meter: the singer isolated from the voice (the band the
        // canceller missed, the room), falling as the capture's. A frame
        // comes every 10 ms, so a block may bring none: it only falls.
        self.down.process(voice, &mut self.at_16k);
        self.isolator.process(&self.at_16k, &mut self.isolated);
        self.at_16k.clear();
        let block_peak = self.isolated.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        self.isolated.clear();
        let decay = (-(BLOCK as f32) / stamp.rate.max(1) as f32 / 0.3).exp();
        self.peak = block_peak.max(self.peak * decay);
        c.voice_peak.store(self.peak.to_bits(), Ordering::Relaxed);
        c.voice_at.store(shared.now_ns() + 1, Ordering::Relaxed);
        // The voice kept. After a block lost, the next does not follow
        // the one before it (which may have started a run: whoever reads
        // the voice must see the gap).
        let kept = Voice {
            stamp: Stamp {
                continues: stamp.continues && !self.dropped,
                ..stamp
            },
            samples: *voice,
        };
        self.dropped = !self.voice.push(kept);
        if self.dropped {
            c.voice_dropped.fetch_add(1, Ordering::Relaxed);
        }
        if shared.cleaned.record.has_consumer() {
            // The block as the canceller had it, whole or not at all.
            let p = &held.packet;
            let mut block: RecordBlock = [[0.0; RECORD_CHANNELS]; BLOCK];
            for (i, f) in block.iter_mut().enumerate() {
                *f = [p.mic[i], p.left[i], p.right[i], held.cancelled[i], voice[i]];
            }
            self.record.push(block);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::capture::{MicClock, Opened};
    use super::super::Mono;
    use super::*;

    const RATE: u32 = 48_000;

    /// An input's shared state and its cleaner, the tap's end of the
    /// cleaner's queue, and the voice's end.
    fn cleaner() -> (Opened, spsc::Producer<Packet>, spsc::Consumer<Voice>) {
        let (_, opened) = Opened::new(RATE, 240, Mono::Auto, MicClock::caller());
        let blocks = opened.shared.bridge.blocks.producer().unwrap();
        let voice = opened.shared.cleaned.voice.consumer().unwrap();
        (opened, blocks, voice)
    }

    /// Queues a block stamped `stream_frame`, following the one before or
    /// not, nothing playing, the microphone silent but for a click at
    /// sample `click` of it.
    fn queue(
        blocks: &mut spsc::Producer<Packet>,
        stream_frame: u64,
        continues: bool,
        click: Option<usize>,
    ) {
        let mut mic = [0.0f32; BLOCK];
        if let Some(i) = click {
            mic[i] = 0.5;
        }
        assert!(blocks.push(Packet {
            stamp: Stamp {
                stream_frame,
                rate: RATE,
                continues,
            },
            mic,
            left: [0.0; BLOCK],
            right: [0.0; BLOCK],
        }));
    }

    /// Cleans all queued; the voice blocks given.
    fn clean(opened: &mut Opened, voice: &mut spsc::Consumer<Voice>) -> Vec<VoiceBlock> {
        while opened.cleaner.run(&opened.shared) {}
        std::iter::from_fn(|| voice.pop())
            .map(|v| v.block())
            .collect()
    }

    /// A click in block 4 comes out in the block stamped as block 4, at the
    /// same sample, the canceller on or off; everything else is silent; the
    /// newest block is held (the suppressor gives each one block late).
    #[test]
    fn a_click_comes_out_in_the_block_it_was_heard_in() {
        for echo in [true, false] {
            let (mut opened, mut blocks, mut voice) = cleaner();
            opened.shared.echo_on.store(echo, Ordering::Relaxed);
            for k in 0..10u64 {
                let click = (k == 4).then_some(37);
                queue(&mut blocks, k * BLOCK as u64, k > 0, click);
            }
            let given = clean(&mut opened, &mut voice);
            let frames: Vec<u64> = given.iter().map(|v| v.stream_frame).collect();
            let want: Vec<u64> = (0..9).map(|k| k * BLOCK as u64).collect();
            assert_eq!(frames, want, "echo {echo}");
            for v in &given {
                for (i, s) in v.samples.iter().enumerate() {
                    let click = v.stream_frame == 4 * BLOCK as u64 && i == 37;
                    let expected = if click { 0.5 } else { 0.0 };
                    assert!(
                        (s - expected).abs() < 1e-5,
                        "echo {echo}: block at {} sample {i}: {s}",
                        v.stream_frame
                    );
                }
            }
        }
    }

    /// A gap: the block held when it comes is dropped; the new run's first
    /// block is given only once the one after it comes, not following the
    /// one before; a click in the new run comes out where it was heard.
    #[test]
    fn at_a_gap_the_block_held_is_dropped() {
        let (mut opened, mut blocks, mut voice) = cleaner();
        let block = BLOCK as u64;
        for k in 0..5u64 {
            queue(&mut blocks, k * block, k > 0, None);
        }
        queue(&mut blocks, 100 * block, false, None);
        let given = clean(&mut opened, &mut voice);
        let frames: Vec<u64> = given.iter().map(|v| v.stream_frame).collect();
        assert_eq!(frames, [0, block, 2 * block, 3 * block]);
        queue(&mut blocks, 101 * block, true, Some(5));
        queue(&mut blocks, 102 * block, true, None);
        let given = clean(&mut opened, &mut voice);
        let frames: Vec<u64> = given.iter().map(|v| v.stream_frame).collect();
        assert_eq!(frames, [100 * block, 101 * block]);
        assert!(!given[0].continues && given[1].continues);
        assert!((given[1].samples[5] - 0.5).abs() < 1e-5);
        let rest: f32 = given
            .iter()
            .flat_map(|v| v.samples.iter())
            .map(|s| s.abs())
            .sum();
        assert!((rest - 0.5).abs() < 1e-4, "{rest}");
    }
}
