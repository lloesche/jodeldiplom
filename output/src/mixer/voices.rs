//! The effects input on the mixer thread ([`super::effects`]): the cues
//! taken from its queue, each waiting for its frame, then sounding on a
//! voice; a chunk's voices summed into samples for the mix. Nothing here
//! allocates or locks as it mixes: the lists have their room from the
//! start; the chunk's samples and gains theirs, for a period of up to
//! [`RESERVED_CHANNELS`] channels at the device's rate or the bank's, from
//! when the input is made, a bank comes or the device opens
//! (`Run::fit_effects`, which also gives the mix and the strip's meter
//! theirs).

use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::chunk::{HeardLane, Slot};
use super::effects::{CueQueue, EffectCue, SoundBank, Taken, LATE, VOICES, WAITING};
use super::envelope::Envelope;
use super::strips::Strips;

/// Full scale of the 24-bit samples mixed.
const FULL: f32 = 8_388_607.0;

/// The channels the effects' buffers have room for from the start.
pub(super) const RESERVED_CHANNELS: usize = 8;

/// A cue waiting for its frame: the input and the output on it it was
/// bound to as it was taken, and its frame of that output.
#[derive(Clone, Copy, Debug)]
struct Waiting {
    cue: EffectCue,
    on: Option<(u64, u64, u64)>,
    cancels: u64,
}

/// A sound sounding: from which of the chunk's frames (this chunk only:
/// 0 after), how far into it, at what gain on each side.
#[derive(Clone, Copy, Debug)]
struct Voice {
    sound: usize,
    from: usize,
    at: usize,
    gains: (f32, f32),
}

/// The effects input as the mixer thread runs it.
pub(crate) struct EffectsRun {
    /// Its strip.
    pub(super) strip: u64,
    pub(super) queue: Arc<CueQueue>,
    bank: Option<Arc<SoundBank>>,
    waiting: Vec<Waiting>,
    voices: Vec<Voice>,
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
    pub(super) fn new(strip: u64, queue: Arc<CueQueue>) -> EffectsRun {
        EffectsRun {
            strip,
            queue,
            bank: None,
            waiting: Vec::with_capacity(WAITING),
            voices: Vec::with_capacity(VOICES),
            samples: Vec::new(),
            sounded: false,
            gains: Envelope::new(1.0),
            audible: false,
        }
    }

    /// The rate of its bank (`None`: none yet).
    pub(super) fn rate(&self) -> Option<u32> {
        self.bank.as_ref().map(|b| b.rate())
    }

    /// Room for chunks of `frames` frames of up to [`RESERVED_CHANNELS`]
    /// channels, so mixing them never allocates.
    pub(super) fn reserve(&mut self, frames: usize) {
        let samples = frames * RESERVED_CHANNELS;
        self.samples
            .reserve(samples.saturating_sub(self.samples.len()));
        self.gains.reserve(frames);
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

    /// Whether it may be heard for now (a voice sounding or a cue waiting).
    pub(super) fn busy(&self) -> bool {
        !self.voices.is_empty() || !self.waiting.is_empty()
    }

    /// The cues come since the last chunk, each bound to the output open
    /// on its input (one with none is dropped), those cancelled dropped.
    fn take_cues(&mut self, slots: &std::collections::BTreeMap<u64, Slot>) {
        let cancels = self.queue.cancels.load(Ordering::Acquire);
        self.waiting.retain(|w| {
            let keep = w.cancels >= cancels;
            if !keep {
                self.queue.miss();
            }
            keep
        });
        while let Some(Taken {
            cue,
            on,
            cancels: c,
        }) = self.queue.pop()
        {
            let on = match on {
                None => None,
                Some((slot, frame)) => {
                    let stream = slots.get(&slot).and_then(|s| s.stream.as_ref());
                    match stream {
                        Some(st) => Some((slot, st.link.id, frame)),
                        None => {
                            self.queue.miss();
                            continue;
                        }
                    }
                }
            };
            if c < cancels || self.waiting.len() == WAITING {
                self.queue.miss();
                continue;
            }
            self.waiting.push(Waiting {
                cue,
                on,
                cancels: c,
            });
        }
    }

    /// The cues waiting for output `id` are dropped (it was flushed).
    pub(super) fn flushed(&mut self, id: u64) {
        let queue = &self.queue;
        self.waiting.retain(|w| {
            let keep = w.on.is_none_or(|(_, output, _)| output != id);
            if !keep {
                queue.miss();
            }
            keep
        });
    }

    /// The `n` frames from device frame `first`, the device at `rate` with
    /// `ch` channels, the inputs as they gave their frames (`slots`): the
    /// cues due in them start, the voices sounding are summed into
    /// [`EffectsRun::samples`] ([`EffectsRun::sounded`]: anything was).
    pub(super) fn chunk(
        &mut self,
        n: usize,
        ch: usize,
        rate: u32,
        slots: &std::collections::BTreeMap<u64, Slot>,
    ) {
        self.take_cues(slots);
        self.sounded = false;
        let late = (LATE.as_secs_f64() * f64::from(rate)) as u64;
        // (A count more on the bank held, for this chunk: nothing freed.)
        let bank = self.bank.clone().filter(|b| b.rate() == rate);
        let mut k = 0;
        while k < self.waiting.len() {
            let w = self.waiting[k];
            // Where in the chunk it starts (`None`: not yet), or dropped.
            let start: Result<Option<usize>, ()> = match w.on {
                None => Ok(Some(0)),
                Some((slot, output, frame)) => {
                    let stream = slots.get(&slot).and_then(|s| s.stream.as_ref());
                    match stream.filter(|st| st.link.id == output) {
                        None => Err(()),
                        Some(st) => match st.given() {
                            None => Ok(None),
                            Some(g) if frame >= g.frame + g.got => Ok(None),
                            Some(g) if frame >= g.frame => {
                                Ok(Some(g.at + (frame - g.frame) as usize))
                            }
                            Some(g) if g.frame - frame <= late => Ok(Some(g.at)),
                            Some(_) => Err(()),
                        },
                    }
                }
            };
            match start {
                Ok(None) => {
                    k += 1;
                    continue;
                }
                Ok(Some(from)) => {
                    self.waiting.swap_remove(k);
                    match bank.as_ref().filter(|b| b.sound(w.cue.sound).is_some()) {
                        Some(_) => {
                            self.start(w.cue, from);
                            self.queue.played.fetch_add(1, Ordering::Relaxed);
                        }
                        None => self.queue.miss(),
                    }
                }
                Err(()) => {
                    self.waiting.swap_remove(k);
                    self.queue.miss();
                }
            }
        }
        if let Some(bank) = bank {
            self.sum(&bank, n, ch);
        } else {
            // Not heard at this rate: what sounds ends.
            self.voices.clear();
        }
        self.queue
            .sounding
            .store(self.voices.len() as u64, Ordering::Relaxed);
    }

    /// `cue` sounds from frame `from` of this chunk, on a voice of its own
    /// (with all of them taken, on the one nearest its end).
    fn start(&mut self, cue: EffectCue, from: usize) {
        let voice = Voice {
            sound: cue.sound,
            from,
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
        self.samples.clear();
        self.samples.resize(n * ch, 0);
        let mut k = 0;
        while k < self.voices.len() {
            let v = &mut self.voices[k];
            let sound = bank.sound(v.sound).unwrap_or(&[]);
            let frames = sound.len() / 2;
            let (gl, gr) = v.gains;
            let from = v.from.min(n);
            // (Past its end, should another bank have a shorter sound in
            // its place: none left.)
            let at = v.at.min(frames);
            let take = (frames - at).min(n - from);
            let frames_in = sound[2 * at..2 * (at + take)].chunks_exact(2);
            let frames_out = self.samples[from * ch..(from + take) * ch].chunks_exact_mut(ch);
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
            v.from = 0;
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
fn to_word(x: f32) -> i32 {
    (x.clamp(-1.0, 1.0) * FULL).round() as i32
}
