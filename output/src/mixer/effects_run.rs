//! The effects input on the mixer thread ([`super::effects`]): its live
//! synth's frames (`live.rs`) summed into a chunk's samples for the mix,
//! at its strip's gains. Nothing here allocates or locks as it mixes: the
//! chunk's samples and gains have their room, for a period of up to
//! [`RESERVED_CHANNELS`] channels at the device's rate or the synth's,
//! from when the input is made, a synth comes or the device opens
//! (`Run::fit_effects`, which also gives the mix and the strip's meter
//! theirs).

use std::sync::atomic::Ordering;
use std::sync::Arc;

use jodel_synth::Synth;

use super::chunk::{HeardLane, Slot};
use super::effects::Counts;
use super::envelope::Envelope;
use super::live::{LiveRun, Sent};
use super::strips::Strips;
use crate::spsc::Consumer;

/// Full scale of the 24-bit samples mixed.
const FULL: f32 = 8_388_607.0;

/// The channels the effects' buffers have room for from the start.
pub(super) const RESERVED_CHANNELS: usize = 8;

/// The effects input as the mixer thread runs it.
pub(crate) struct EffectsRun {
    /// Its strip.
    pub(super) strip: u64,
    pub(super) counts: Arc<Counts>,
    /// Its live synth, and the commands sent it.
    live: LiveRun,
    /// This chunk's sum (the device's channels), when it sounded.
    samples: Vec<i32>,
    sounded: bool,
    /// The gains it is heard at this chunk (its strip's path), and whether
    /// they let it through.
    gains: Envelope,
    audible: bool,
}

impl EffectsRun {
    pub(super) fn new(strip: u64, counts: Arc<Counts>, live: Consumer<Sent>) -> EffectsRun {
        EffectsRun {
            strip,
            counts,
            live: LiveRun::new(live),
            samples: Vec::new(),
            sounded: false,
            gains: Envelope::new(1.0),
            audible: false,
        }
    }

    /// The rate of its synth (`None`: none yet).
    pub(super) fn rate(&self) -> Option<u32> {
        self.live.rate()
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

    /// `synth` played live from now on; the one before, to be let go of
    /// elsewhere.
    pub(super) fn set_live(&mut self, synth: Option<Box<Synth>>) -> Option<Box<Synth>> {
        let rate = synth.as_ref().map_or(0, |s| s.rate().round() as u64);
        self.counts.live_rate.store(rate, Ordering::Relaxed);
        self.live.set(synth, &self.counts)
    }

    /// Its live synth, taken (to be let go of elsewhere).
    pub(super) fn take_live(&mut self) -> Option<Box<Synth>> {
        self.counts.live_rate.store(0, Ordering::Relaxed);
        self.live.take_synth(&self.counts)
    }

    /// The device opened (again): what waits in its synth is called off.
    pub(super) fn opened(&mut self) {
        self.live.opened(&self.counts);
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

    /// Whether it may be heard for now (the synth busy).
    pub(super) fn busy(&self) -> bool {
        self.live.busy()
    }

    /// Output `id` was flushed: what the synth waits to play on its frames
    /// is called off.
    pub(super) fn flushed(&mut self, id: u64) {
        self.live.flushed(id, &self.counts);
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
        self.live.take((epoch, first), rate, slots, &self.counts);
        self.samples.clear();
        self.samples.resize(n * ch, 0);
        self.sounded = self.live.sum(n, ch, rate, &mut self.samples);
    }
}

/// A sample at full scale 1.0 as a 24-bit word.
pub(super) fn to_word(x: f32) -> i32 {
    (x.clamp(-1.0, 1.0) * FULL).round() as i32
}
