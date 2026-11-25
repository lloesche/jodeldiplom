//! The desk's effects input ([`Mixer::effects`]): a game's sounds on a
//! strip of their own, summed with the inputs after their gains (so the
//! microphone's tap on the device hears them, as the song). They are of
//! two kinds:
//!
//! - Sounds made ahead, in a [`SoundBank`] at one rate, made off the mixer
//!   thread ([`Effects::load`]); a bank at another rate than the device's
//!   is not heard (the mixer does not resample), so the caller makes one
//!   again when the device opens at another. A sound is cued for now
//!   ([`Effects::play`]) through a queue of fixed size the mixer thread
//!   reads each period without a lock or an allocation, and starts on the
//!   next period mixed; it sounds on a voice of its own until it ends
//!   ([`VOICES`] at once: a new one takes the place of the one nearest its
//!   end).
//! - A synth played live ([`Effects::live`]): a planner sends it timed
//!   notes ([`jodel_synth::Command`], [`Effects::commands`]) on the stream
//!   frames of the output open on a fader's input (so each lands on its
//!   moment of the song, inside the period it falls in), or on the desk's
//!   own frames ([`super::Timebase::frame`]: a sound after the song, when
//!   no output is open); the mixer thread puts them on the synth's frames
//!   as it takes them, and renders the synth with the bank's voices (see
//!   `live.rs`: one late by more than [`LATE`] is dropped, a flush of the
//!   output calls off what waits on its frames).
//!
//! The desk's device plays all along in real time (see [`Mixer`]), so a
//! cue for now sounds on the next period mixed whatever the outputs do (a
//! song over, paused, or the next held back). On a device that plays only
//! what it is given it waits for an output to run; while the device is
//! down (not there, or failed) cues pushed wait in the queue for the
//! mixer thread's next message (an output opened), and sound once it is
//! open again.
//!
//! While no voice sounds and the synth rests, the effects input adds
//! nothing, and what goes to the device is as it was without it, bit for
//! bit.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use jodel_synth::{Command, Commands, Synth};

use super::input::ask;
use super::live::{On, Sent, COMMANDS};
use super::{Fader, Mixer, Msg, Strip};
use crate::spsc::{Producer, Spsc};

/// The most voices sounding at once.
pub const VOICES: usize = 24;
/// The cues the queue to the mixer thread holds.
const QUEUE: usize = 64;
/// How late a live synth's note may start, past its frame.
pub const LATE: Duration = Duration::from_millis(80);

/// Sounds at one rate, each stereo, interleaved, at full scale 1.0.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SoundBank {
    rate: u32,
    sounds: Vec<Box<[f32]>>,
}

impl SoundBank {
    /// An empty bank at `rate`.
    pub fn new(rate: u32) -> SoundBank {
        SoundBank {
            rate,
            sounds: Vec::new(),
        }
    }

    /// Adds a sound (stereo frames, interleaved: an odd sample at the end
    /// is left out); returns its index.
    pub fn add(&mut self, stereo: Vec<f32>) -> usize {
        let mut stereo = stereo;
        stereo.truncate(stereo.len() / 2 * 2);
        self.sounds.push(stereo.into_boxed_slice());
        self.sounds.len() - 1
    }

    /// Its rate.
    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// How many sounds it has.
    pub fn len(&self) -> usize {
        self.sounds.len()
    }

    /// Whether it has none.
    pub fn is_empty(&self) -> bool {
        self.sounds.is_empty()
    }

    /// Sound `k`'s samples (stereo, interleaved), if it has one.
    pub fn sound(&self, k: usize) -> Option<&[f32]> {
        self.sounds.get(k).map(|s| &**s)
    }
}

/// A sound to play: which of the bank, at what gain (linear), where
/// between the sides (-1 left to 1 right; 0 as made).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EffectCue {
    /// Its index in the bank.
    pub sound: usize,
    /// Its gain.
    pub gain: f32,
    /// Its pan.
    pub pan: f32,
}

impl EffectCue {
    /// Sound `sound` as made.
    pub fn new(sound: usize) -> EffectCue {
        EffectCue {
            sound,
            gain: 1.0,
            pan: 0.0,
        }
    }
}

/// What the effects input did, since it was made.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EffectsView {
    /// Cues that started sounding.
    pub played: u64,
    /// Cues and commands dropped: the queue full, a sound the bank has
    /// not, a note too late or its output gone, the synth's list of those
    /// waiting full, a synth at another rate, or called off by a flush.
    pub missed: u64,
    /// Voices sounding after the last frames mixed.
    pub sounding: u64,
    /// The rate of the bank in use (0: none).
    pub rate: u32,
    /// Commands the live synth took.
    pub commands: u64,
    /// The rate of the live synth (0: none).
    pub live: u32,
}

/// A cue in the queue: its words (see [`CueQueue::push`]).
type Words = [AtomicU64; 2];

/// The cues on their way to the mixer thread, and what it tells back. One
/// side pushes (the handle, under its lock), the mixer thread pops: no
/// lock on its side, nothing allocated.
pub(crate) struct CueQueue {
    slots: Box<[Words]>,
    /// Cues ever pushed (the producer's), and popped (the mixer thread's).
    head: AtomicU64,
    tail: AtomicU64,
    pub(crate) played: AtomicU64,
    pub(crate) missed: AtomicU64,
    pub(crate) sounding: AtomicU64,
    pub(crate) rate: AtomicU64,
    pub(crate) commands: AtomicU64,
    pub(crate) live_rate: AtomicU64,
}

impl CueQueue {
    fn new() -> CueQueue {
        CueQueue {
            slots: (0..QUEUE).map(|_| Default::default()).collect(),
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            played: AtomicU64::new(0),
            missed: AtomicU64::new(0),
            sounding: AtomicU64::new(0),
            rate: AtomicU64::new(0),
            commands: AtomicU64::new(0),
            live_rate: AtomicU64::new(0),
        }
    }

    /// Pushes a cue (one producer at a time); false when full.
    fn push(&self, cue: EffectCue) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        if head - self.tail.load(Ordering::Acquire) >= QUEUE as u64 {
            return false;
        }
        let w = &self.slots[(head % QUEUE as u64) as usize];
        // Word 0: the sound; 1: the gain and the pan (f32 bits each).
        let gp = u64::from(cue.gain.to_bits()) | (u64::from(cue.pan.to_bits()) << 32);
        w[0].store(cue.sound as u64, Ordering::Relaxed);
        w[1].store(gp, Ordering::Relaxed);
        self.head.store(head + 1, Ordering::SeqCst);
        true
    }

    /// The mixer thread: the next cue, if one came.
    pub(crate) fn pop(&self) -> Option<EffectCue> {
        let tail = self.tail.load(Ordering::Relaxed);
        if tail == self.head.load(Ordering::Acquire) {
            return None;
        }
        let w = &self.slots[(tail % QUEUE as u64) as usize];
        let (sound, gp) = (w[0].load(Ordering::Relaxed), w[1].load(Ordering::Relaxed));
        self.tail.store(tail + 1, Ordering::Release);
        Some(EffectCue {
            sound: sound as usize,
            gain: f32::from_bits(gp as u32),
            pan: f32::from_bits((gp >> 32) as u32),
        })
    }

    /// One more cue dropped.
    pub(crate) fn miss(&self) {
        self.missed.fetch_add(1, Ordering::Relaxed);
    }
}

/// The desk's effects input, on its strip ([`Mixer::effects`]). Dropping
/// it takes the input off the desk (the voices sounding stop).
pub struct Effects {
    tx: Sender<Msg>,
    queue: Arc<CueQueue>,
    /// One producer at a time (the handle may be shared between threads).
    pushing: Mutex<()>,
    /// The banks sent, kept until the mixer thread let go of them, so that
    /// none is freed there.
    banks: Mutex<Vec<Arc<SoundBank>>>,
    /// The commands' way to the live synth (one sender at a time).
    live: Mutex<Producer<Sent>>,
    strip: Strip,
}

/// Where a planner's commands go to the live synth ([`Effects::commands`]):
/// each on the clock it was made for. It holds the queue's producer lock
/// while it lives (another sender waits for it): keep it for one planning
/// step, as long as an event or a clock takes, never across a wait.
pub struct LiveCommands<'a> {
    tx: MutexGuard<'a, Producer<Sent>>,
    on: On,
    queue: &'a CueQueue,
}

impl Commands for LiveCommands<'_> {
    fn send(&mut self, c: Command) -> bool {
        let sent = self.tx.push(Sent {
            command: c,
            on: self.on,
        });
        if !sent {
            self.queue.miss();
        }
        sent
    }
}

impl Mixer {
    /// The effects input on `strip` (one of this mixer's), in place of any
    /// made before: its level, mute and solo are the strip's (a strip for
    /// it alone is the user's fader for the effects).
    pub fn effects(&self, strip: &Strip) -> Effects {
        assert!(
            Arc::ptr_eq(&strip.handle.shared, &self.shared),
            "a strip of another mixer"
        );
        let queue = Arc::new(CueQueue::new());
        let live = Spsc::new(COMMANDS);
        let (tx, rx) = (live.producer(), live.consumer());
        let (tx, rx) = (tx.expect("a new queue"), rx.expect("a new queue"));
        let _ = self.tx.send(Msg::Effects(strip.id(), queue.clone(), rx));
        Effects {
            tx: self.tx.clone(),
            queue,
            pushing: Mutex::new(()),
            banks: Mutex::new(Vec::new()),
            live: Mutex::new(tx),
            strip: strip.clone(),
        }
    }
}

impl Effects {
    /// The sounds played from now on (cues waiting keep their indices): a
    /// bank, or one shared (kept by the caller for another time, as it
    /// was). The mixer reserves what mixing them needs as it takes it.
    pub fn load(&self, bank: impl Into<Arc<SoundBank>>) {
        let bank = bank.into();
        let mut banks = self.banks.lock().unwrap_or_else(|e| e.into_inner());
        // Those the mixer thread no longer holds go here, not there.
        banks.retain(|b| Arc::strong_count(b) > 1);
        banks.push(bank.clone());
        let _ = self.tx.send(Msg::EffectsBank(bank));
    }

    /// `synth` played live from now on, at `gain` (linear) on the strip,
    /// in place of any before (which is let go of off the mixer thread);
    /// heard while the device runs at its rate. Its commands come through
    /// [`Effects::commands`].
    pub fn live(&self, synth: Synth, gain: f32) {
        let _ = self.tx.send(Msg::EffectsLive(Box::new(synth), gain));
    }

    /// Where commands go to the live synth, their frames the stream frames
    /// of the output open on `fader`'s input, or with none the desk's own
    /// ([`super::Timebase::frame`], of the device's opening now: one sent
    /// before it opens again is dropped). A command not taken (the queue
    /// full) is dropped and counted.
    pub fn commands(&self, fader: Option<&Fader>) -> LiveCommands<'_> {
        LiveCommands {
            tx: self.live.lock().unwrap_or_else(|e| e.into_inner()),
            on: match fader {
                Some(f) => On::Input(f.slot_id()),
                None => On::Desk(self.strip.handle.shared.epoch.load(Ordering::Acquire)),
            },
            queue: &self.queue,
        }
    }

    /// Plays `cue` on the next period mixed; false when the queue is full
    /// (the cue is dropped).
    pub fn play(&self, cue: EffectCue) -> bool {
        let _one = self.pushing.lock().unwrap_or_else(|e| e.into_inner());
        let pushed = self.queue.push(cue);
        if !pushed {
            self.queue.miss();
        }
        pushed
    }

    /// What it did, as of the last frames mixed.
    pub fn view(&self) -> EffectsView {
        let q = &self.queue;
        EffectsView {
            played: q.played.load(Ordering::Relaxed),
            missed: q.missed.load(Ordering::Relaxed),
            sounding: q.sounding.load(Ordering::Relaxed),
            rate: q.rate.load(Ordering::Relaxed) as u32,
            commands: q.commands.load(Ordering::Relaxed),
            live: q.live_rate.load(Ordering::Relaxed) as u32,
        }
    }

    /// Its strip.
    pub fn strip(&self) -> &Strip {
        &self.strip
    }
}

impl Drop for Effects {
    /// Takes the input off the desk (not another made since): what the
    /// mixer thread ran of it, its bank and its synth among it, comes back
    /// and is let go of here, not there.
    fn drop(&mut self) {
        let queue = self.queue.clone();
        let run = ask(&self.tx, |done| Msg::EffectsGone { queue, done });
        drop(run);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cues_go_through_the_queue_as_they_were_given() {
        let q = CueQueue::new();
        let cue = EffectCue {
            sound: 7,
            gain: 0.5,
            pan: -0.25,
        };
        assert!(q.push(cue));
        assert!(q.push(EffectCue::new(1)));
        assert_eq!(q.pop(), Some(cue));
        assert_eq!(q.pop(), Some(EffectCue::new(1)));
        assert!(q.pop().is_none());
        // Full: refused, nothing overwritten.
        for _ in 0..QUEUE {
            assert!(q.push(EffectCue::new(0)));
        }
        assert!(!q.push(EffectCue::new(9)));
        assert_eq!(q.pop().unwrap().sound, 0);
    }
}
