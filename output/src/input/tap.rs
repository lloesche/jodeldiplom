//! The tap in the device thread: the voice brought over the bridge to the
//! output's rate, lined up with what the device plays (the echo
//! reference), and queued block by block for the cleaner.

use super::bridge::{Steer, MAX_RATIO};
use super::capture::MicShared;
use super::Input;
use crate::aec::BLOCK;
use crate::format::StreamFormat;
use crate::spsc;
use crate::tap::{Tap, TAP_CHUNK};
use std::sync::atomic::Ordering;

/// The canceller's rate: about this, the output's divided by a whole number.
pub(super) const ECHO_RATE: u32 = 48_000;
/// Blocks the tap may queue for the cleaner (about 2.7 s at 48 kHz).
pub(super) const BLOCK_QUEUE: usize = 1024;

/// Where a block is in the stream.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Stamp {
    /// The stream frame the block's first sample was handed to the device
    /// at (as [`crate::Output::position`] counts them).
    pub(super) stream_frame: u64,
    /// The canceller's rate the block is at.
    pub(super) rate: u32,
    /// The block follows the one before without a gap.
    pub(super) continues: bool,
}

/// One block the tap queues for the cleaner: where it is, and the
/// microphone's and what plays (left and right), [`BLOCK`] samples each at
/// the canceller's rate.
#[derive(Clone, Copy)]
pub(super) struct Packet {
    pub(super) stamp: Stamp,
    pub(super) mic: [f32; BLOCK],
    pub(super) left: [f32; BLOCK],
    pub(super) right: [f32; BLOCK],
}

/// The ends of the input's queues the consuming tap holds: the capture
/// ring it reads, the cleaner's queue it fills.
struct Ends {
    ring: spsc::Consumer<f32>,
    blocks: spsc::Producer<Packet>,
}

impl Ends {
    /// Both, unless another tap holds them.
    fn take(shared: &MicShared) -> Option<Ends> {
        Some(Ends {
            ring: shared.capture.ring.consumer()?,
            blocks: shared.bridge.blocks.producer()?,
        })
    }
}

/// Lines an [`Input`] up with what an output plays, in its device thread:
/// the voice resampled to the output's rate, and beside it what the
/// buffer plays, left and right ([`MicTap::sides`]), all brought down to
/// the canceller's rate (the output's divided by a whole number) and
/// queued, block by block, for the cleaner. It only reads the buffers. Hand it to
/// [`Output::set_tap`](crate::Output::set_tap).
///
/// Only one tap of an input consumes it at a time: another one (on an
/// output just reopened while the old one still runs) queues nothing until
/// the first is dropped.
pub struct MicTap {
    /// The queues' ends, once this tap consumes the input.
    ends: Option<Ends>,
    input: Input,
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
    left_sum: f32,
    right_sum: f32,
    phase: usize,
    /// The block being filled, and how far.
    packet: Box<Packet>,
    filled: usize,
    /// The next block follows the last one queued without a gap.
    continues: bool,
}

impl MicTap {
    pub(super) fn new(input: Input) -> MicTap {
        MicTap {
            ends: None,
            input,
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
            left_sum: 0.0,
            right_sum: 0.0,
            phase: 0,
            packet: Box::new(Packet {
                stamp: Stamp::default(),
                mic: [0.0; BLOCK],
                left: [0.0; BLOCK],
                right: [0.0; BLOCK],
            }),
            filled: 0,
            continues: false,
        }
    }

    /// The input it lines up.
    pub fn input(&self) -> &Input {
        &self.input
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
        self.left_sum = 0.0;
        self.right_sum = 0.0;
        self.phase = 0;
        self.filled = 0;
        self.continues = false;
    }
}

impl MicTap {
    /// What a frame of 24-bit values plays left and right, full scale ±1:
    /// of a mono output its channel on both; of a stereo one its two; of
    /// more, the first two, every channel after them on both halves alike
    /// (a centre, a subwoofer, the rear: the canceller hears them as what
    /// both sides play). Their mean is the frame's mean in each case.
    fn sides(frame: &[i32]) -> (f32, f32) {
        let ch = frame.len().max(1);
        let unit = 1.0 / 8_388_608.0;
        match frame {
            [] => (0.0, 0.0),
            [one] => (*one as f32 * unit, *one as f32 * unit),
            [l, r, rest @ ..] => {
                let shared: f32 = rest.iter().map(|&s| s as f32).sum::<f32>();
                let scale = 2.0 * unit / ch as f32;
                (
                    (*l as f32 + shared / 2.0) * scale,
                    (*r as f32 + shared / 2.0) * scale,
                )
            }
        }
    }
}

impl Drop for MicTap {
    fn drop(&mut self) {
        // The bridge stops reading before the ends go back.
        if self.ends.is_some() {
            self.input
                .shared()
                .bridge
                .reading
                .store(false, Ordering::Release);
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
        let shared = &*self.input.inner.shared;
        if self.ends.is_none() {
            self.ends = Ends::take(shared);
        }
        self.live = match &mut self.ends {
            Some(ends) => self
                .steer
                .begin(shared, &mut ends.ring, format.rate, frames),
            None => false,
        };
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
        let ends = match &mut self.ends {
            Some(ends) if self.live && self.factor != 0 => ends,
            _ => {
                self.at += frames;
                return;
            }
        };
        let shared = &*self.input.inner.shared;
        self.steer
            .render(&mut ends.ring, &mut self.voice[..frames], &mut self.taken);
        let rate = self.rate / self.factor as u32;
        for (frame, &mic) in samples.chunks_exact(ch).zip(self.voice[..frames].iter()) {
            if self.phase == 0 && self.filled == 0 {
                self.packet.stamp.stream_frame =
                    self.buffer_frame + self.at.min(self.buffer_stream) as u64;
            }
            self.at += 1;
            self.mic_sum += mic;
            let (left, right) = MicTap::sides(frame);
            self.left_sum += left;
            self.right_sum += right;
            self.phase += 1;
            if self.phase < self.factor {
                continue;
            }
            let n = self.factor as f32;
            self.packet.mic[self.filled] = self.mic_sum / n;
            self.packet.left[self.filled] = self.left_sum / n;
            self.packet.right[self.filled] = self.right_sum / n;
            self.mic_sum = 0.0;
            self.left_sum = 0.0;
            self.right_sum = 0.0;
            self.phase = 0;
            self.filled += 1;
            if self.filled == BLOCK {
                self.filled = 0;
                self.packet.stamp.rate = rate;
                self.packet.stamp.continues = self.continues;
                self.continues = ends.blocks.push(*self.packet);
                if !self.continues {
                    shared.bridge.blocks_dropped.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
}
