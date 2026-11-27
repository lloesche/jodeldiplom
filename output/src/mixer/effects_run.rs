//! The effects input on the mixer thread ([`super::effects`]): its live
//! synth's frames (`live.rs`, which the mixer tells of the device opening,
//! an output flushed, a synth come) summed into a chunk's samples for the
//! mix, at its strip's gains. Nothing here allocates or locks as it mixes: the
//! chunk's samples and gains have their room, for a period of up to
//! [`RESERVED_CHANNELS`] channels at the device's rate or the synth's,
//! from when the input is made, a synth comes or the device opens
//! (`Run::fit_effects`, which also gives the mix and the strip's meter
//! theirs).

use super::chunk::{HeardLane, Slot};
use super::envelope::Envelope;
use super::live::LiveRun;
use super::strips::Strips;
use jodel_dsp::PEAK;

/// The channels the effects' buffers have room for from the start.
pub(super) const RESERVED_CHANNELS: usize = 8;

/// The effects input as the mixer thread runs it.
pub(crate) struct EffectsRun {
    /// Its strip.
    pub(super) strip: u64,
    /// Its live synth, the commands sent it, and what it tells back.
    pub(super) live: LiveRun,
    /// This chunk's sum (the device's channels), when it sounded.
    samples: Vec<i32>,
    sounded: bool,
    /// The gains it is heard at this chunk (its strip's path), and whether
    /// they let it through.
    gains: Envelope,
    audible: bool,
}

impl EffectsRun {
    pub(super) fn new(strip: u64, live: LiveRun) -> EffectsRun {
        EffectsRun {
            strip,
            live,
            samples: Vec::new(),
            sounded: false,
            gains: Envelope::new(1.0),
            audible: false,
        }
    }

    /// Room for chunks of `frames` frames of up to [`RESERVED_CHANNELS`]
    /// channels, so mixing them never allocates.
    pub(super) fn reserve(&mut self, frames: usize) {
        let samples = frames * RESERVED_CHANNELS;
        self.samples
            .reserve(samples.saturating_sub(self.samples.len()));
        self.gains.reserve(frames);
        self.live.reserve(frames);
    }

    /// Takes the gains it is heard at in the chunk just summed
    /// ([`EffectsRun::chunk`]): its strip's path (`strips`).
    pub(super) fn take_gains(&mut self, strips: &Strips) {
        self.audible = false;
        if let (true, Some(path)) = (self.sounded, strips.path_of(self.strip)) {
            self.gains.assign(path);
            self.audible = self.gains.flat() != Some(0.0);
        }
    }

    /// What it gives the chunk, when it is heard: its sum at its strip's
    /// gains (never an exact lane).
    pub(super) fn lane(&self) -> Option<HeardLane<'_>> {
        self.audible.then(|| HeardLane {
            strip: self.strip,
            samples: &self.samples,
            gains: self.gains.as_gains(),
            exact: false,
        })
    }

    /// The `n` frames from device frame `first` of its opening `epoch`,
    /// the device at `rate` with `ch` channels, the inputs as they gave
    /// their frames (`slots`): the commands come go to the synth, and the
    /// synth is summed into [`EffectsRun::samples`]
    /// ([`EffectsRun::sounded`]: anything was).
    pub(super) fn chunk(
        &mut self,
        (epoch, first, n): (u64, u64, usize),
        ch: usize,
        rate: u32,
        slots: &std::collections::BTreeMap<u64, Slot>,
    ) {
        self.live.take((epoch, first), rate, slots);
        self.samples.clear();
        self.samples.resize(n * ch, 0);
        self.sounded = self.live.sum(n, ch, rate, &mut self.samples);
    }
}

/// A sample at full scale 1.0 as a 24-bit word.
pub(super) fn to_word(x: f32) -> i32 {
    (x.clamp(-1.0, 1.0) * PEAK as f32).round() as i32
}
