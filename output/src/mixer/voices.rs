//! The effects input on the mixer thread ([`super::effects`]): the cues
//! taken from its queue, each sounding on a voice from the chunk it is
//! taken in; a chunk's voices, and its live synth's frames (`live.rs`),
//! summed into samples for the mix. Nothing here allocates or locks as it
//! mixes: the lists have their room from the start; the chunk's samples
//! and gains theirs, for a period of up to [`RESERVED_CHANNELS`] channels
//! at the device's rate, the bank's or the synth's, from when the input is
//! made, a bank or a synth comes or the device opens (`Run::fit_effects`,
//! which also gives the mix and the strip's meter theirs).

use std::sync::atomic::Ordering;
use std::sync::Arc;

use jodel_synth::Synth;

use super::chunk::{HeardLane, Slot};
use super::effects::{CueQueue, EffectCue, SoundBank, VOICES};
use super::envelope::Envelope;
use super::live::{LiveRun, Sent};
use super::strips::Strips;
use crate::spsc::Consumer;

/// Full scale of the 24-bit samples mixed.
const FULL: f32 = 8_388_607.0;

/// The channels the effects' buffers have room for from the start.
pub(super) const RESERVED_CHANNELS: usize = 8;

/// A sound sounding: how far into it, at what gain on each side.
#[derive(Clone, Copy, Debug)]
struct Voice {
    sound: usize,
    at: usize,
    gains: (f32, f32),
}

/// The effects input as the mixer thread runs it.
pub(crate) struct EffectsRun {
    /// Its strip.
    pub(super) strip: u64,
    pub(super) queue: Arc<CueQueue>,
    bank: Option<Arc<SoundBank>>,
    voices: Vec<Voice>,
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

/// The gains of each side at `pan` (-1 to 1): the near side whole.
fn balance(pan: f32, gain: f32) -> (f32, f32) {
    let p = if pan.is_finite() {
        pan.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    (gain * (1.0 - p).min(1.0), gain * (1.0 + p).min(1.0))
}

impl EffectsRun {
    pub(super) fn new(strip: u64, queue: Arc<CueQueue>, live: Consumer<Sent>) -> EffectsRun {
        EffectsRun {
            strip,
            queue,
            bank: None,
            voices: Vec::with_capacity(VOICES),
            live: LiveRun::new(live),
            samples: Vec::new(),
            sounded: false,
            gains: Envelope::new(1.0),
            audible: false,
        }
    }

    /// The highest rate of its bank and its synth (`None`: neither yet).
    pub(super) fn rate(&self) -> Option<u32> {
        let bank = self.bank.as_ref().map(|b| b.rate());
        bank.max(self.live.rate())
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
        self.queue.live_rate.store(rate, Ordering::Relaxed);
        self.live.set(synth, &self.queue)
    }

    /// Its live synth, taken (to be let go of elsewhere).
    pub(super) fn take_live(&mut self) -> Option<Box<Synth>> {
        self.queue.live_rate.store(0, Ordering::Relaxed);
        self.live.take_synth(&self.queue)
    }

    /// The device opened (again): what waits in its synth is called off.
    pub(super) fn opened(&mut self) {
        self.live.opened(&self.queue);
    }

    /// The sounds from now on. The bank it held goes with this message
    /// (its handle still holds it).
    pub(super) fn load(&mut self, bank: Arc<SoundBank>) {
        self.queue
            .rate
            .store(u64::from(bank.rate()), Ordering::Relaxed);
        self.bank = Some(bank);
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

    /// Whether it may be heard for now (a voice sounding, or the synth
    /// busy).
    pub(super) fn busy(&self) -> bool {
        !self.voices.is_empty() || self.live.busy()
    }

    /// Output `id` was flushed: what the synth waits to play on its frames
    /// is called off.
    pub(super) fn flushed(&mut self, id: u64) {
        self.live.flushed(id, &self.queue);
    }

    /// The `n` frames from device frame `first` of its opening `epoch`, the device at `rate` with
    /// `ch` channels, the inputs as they gave their frames (`slots`): the
    /// cues come start, the commands come go to the synth, the voices
    /// sounding and the synth are summed into [`EffectsRun::samples`]
    /// ([`EffectsRun::sounded`]: anything was).
    pub(super) fn chunk(
        &mut self,
        (epoch, first, n): (u64, u64, usize),
        ch: usize,
        rate: u32,
        slots: &std::collections::BTreeMap<u64, Slot>,
    ) {
        self.live.take((epoch, first), rate, slots, &self.queue);
        self.sounded = false;
        // (A count more on the bank held, for this chunk: nothing freed.)
        let bank = self.bank.clone().filter(|b| b.rate() == rate);
        while let Some(cue) = self.queue.pop() {
            match bank.as_ref().filter(|b| b.sound(cue.sound).is_some()) {
                Some(_) => {
                    self.start(cue);
                    self.queue.played.fetch_add(1, Ordering::Relaxed);
                }
                None => self.queue.miss(),
            }
        }
        self.samples.clear();
        self.samples.resize(n * ch, 0);
        if let Some(bank) = bank {
            self.sum(&bank, n, ch);
        } else {
            // Not heard at this rate: what sounds ends.
            self.voices.clear();
        }
        let live = self.live.sum(n, ch, rate, &mut self.samples);
        self.sounded |= live;
        self.queue
            .sounding
            .store(self.voices.len() as u64, Ordering::Relaxed);
    }

    /// `cue` sounds from this chunk's first frame, on a voice of its own
    /// (with all of them taken, on the one nearest its end).
    fn start(&mut self, cue: EffectCue) {
        let voice = Voice {
            sound: cue.sound,
            at: 0,
            gains: balance(cue.pan, if cue.gain.is_finite() { cue.gain } else { 0.0 }),
        };
        if self.voices.len() < VOICES {
            self.voices.push(voice);
            return;
        }
        let bank = self.bank.as_ref();
        let left = |v: &Voice| {
            let len = bank.and_then(|b| b.sound(v.sound)).map_or(0, <[f32]>::len) / 2;
            len.saturating_sub(v.at)
        };
        if let Some(k) = (0..self.voices.len()).min_by_key(|&k| left(&self.voices[k])) {
            self.voices[k] = voice;
        }
    }

    /// The voices summed into `n` frames of `ch` channels; those that end
    /// go.
    fn sum(&mut self, bank: &SoundBank, n: usize, ch: usize) {
        if self.voices.is_empty() {
            return;
        }
        let mut k = 0;
        while k < self.voices.len() {
            let v = &mut self.voices[k];
            let sound = bank.sound(v.sound).unwrap_or(&[]);
            let frames = sound.len() / 2;
            let (gl, gr) = v.gains;
            // (Past its end, should another bank have a shorter sound in
            // its place: none left.)
            let at = v.at.min(frames);
            let take = (frames - at).min(n);
            let frames_in = sound[2 * at..2 * (at + take)].chunks_exact(2);
            let frames_out = self.samples[..take * ch].chunks_exact_mut(ch);
            for (s, out) in frames_in.zip(frames_out) {
                let (l, r) = (s[0] * gl, s[1] * gr);
                if ch == 1 {
                    out[0] += to_word(0.5 * (l + r));
                } else {
                    out[0] += to_word(l);
                    out[1] += to_word(r);
                }
            }
            v.at = at + take;
            if v.at >= frames {
                self.voices.swap_remove(k);
            } else {
                k += 1;
            }
        }
        self.sounded = true;
    }
}

/// A sample at full scale 1.0 as a 24-bit word.
pub(super) fn to_word(x: f32) -> i32 {
    (x.clamp(-1.0, 1.0) * FULL).round() as i32
}
