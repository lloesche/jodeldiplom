//! The desk's effects input ([`Mixer::effects`]): short sounds a game plays
//! on its own strip, each started on a frame of a player's stream so it
//! lands on its moment, summed with the inputs after their gains (so the
//! microphone's tap on the device hears them, as the song).
//!
//! The sounds come in a [`SoundBank`] at one rate, made off the mixer
//! thread ([`Effects::load`]); a bank at another rate than the device's is
//! not heard (the mixer does not resample), so the caller makes one again
//! when the device opens at another. A sound is cued ([`Effects::play`],
//! [`Effects::play_at`]) through a queue of fixed size the mixer thread
//! reads each period without a lock or an allocation; it sounds on a voice
//! of its own until it ends ([`VOICES`] at once: a new one takes the place
//! of the one nearest its end).
//!
//! A cue on a stream frame ([`Effects::play_at`]) is bound to the output
//! open on that fader's input when the mixer takes it: it starts on that
//! very frame, inside the period it falls in, as the output's frames go
//! to the device. One whose frame went by already (it came late) starts
//! at once if it is at most [`LATE`] behind, else it is dropped
//! ([`EffectsView::missed`]); so is one whose output is flushed, closed
//! or replaced first, and every one waiting at [`Effects::cancel`]. The
//! frames are the output's as they are written: a player that takes back
//! frames it wrote and writes others in their place (a jump) gives the
//! same numbers to other music, so it cancels what it cued on them. A cue
//! for now starts on the next period mixed.
//!
//! The desk's device plays all along in real time (see [`Mixer`]), so a
//! cue for now sounds on the next period mixed whatever the outputs do (a
//! song over, paused, or the next held back). On a device that plays only
//! what it is given it waits for an output to run; while the device is
//! down (not there, or failed) cues pushed wait in the queue for the
//! mixer thread's next message (an output opened), and sound once it is
//! open again.
//!
//! While no voice sounds, the effects input adds nothing, and what goes to
//! the device is as it was without it, bit for bit.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::input::ask;
use super::{Fader, Mixer, Msg, Strip};

/// The most voices sounding at once.
pub const VOICES: usize = 24;
/// The most cues waiting for their frames on the mixer thread.
pub const WAITING: usize = 64;
/// The cues the queue to the mixer thread holds.
const QUEUE: usize = 64;
/// How late a cue may start, past its frame.
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
    /// Cues dropped: too late, their output gone, the queue or the waiting
    /// list full, a sound the bank has not, or cancelled.
    pub missed: u64,
    /// Voices sounding after the last frames mixed.
    pub sounding: u64,
    /// The rate of the bank in use (0: none).
    pub rate: u32,
}

/// A cue in the queue: its words (see [`CueQueue::push`]).
type Words = [AtomicU64; 4];

/// The cues on their way to the mixer thread, and what it tells back. One
/// side pushes (the handle, under its lock), the mixer thread pops: no
/// lock on its side, nothing allocated.
pub(crate) struct CueQueue {
    slots: Box<[Words]>,
    /// Cues ever pushed (the producer's), and popped (the mixer thread's).
    head: AtomicU64,
    tail: AtomicU64,
    /// [`Effects::cancel`]s so far.
    pub(crate) cancels: AtomicU64,
    pub(crate) played: AtomicU64,
    pub(crate) missed: AtomicU64,
    pub(crate) sounding: AtomicU64,
    pub(crate) rate: AtomicU64,
}

/// A cue as the mixer thread takes it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Taken {
    pub(crate) cue: EffectCue,
    /// The input whose output's frame it starts on (`None`: now), and
    /// that frame.
    pub(crate) on: Option<(u64, u64)>,
    /// The cancels there had been when it was cued.
    pub(crate) cancels: u64,
}

impl CueQueue {
    fn new() -> CueQueue {
        CueQueue {
            slots: (0..QUEUE).map(|_| Default::default()).collect(),
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            cancels: AtomicU64::new(0),
            played: AtomicU64::new(0),
            missed: AtomicU64::new(0),
            sounding: AtomicU64::new(0),
            rate: AtomicU64::new(0),
        }
    }

    /// Pushes a cue (one producer at a time); false when full.
    fn push(&self, cue: EffectCue, on: Option<(u64, u64)>) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        if head - self.tail.load(Ordering::Acquire) >= QUEUE as u64 {
            return false;
        }
        let w = &self.slots[(head % QUEUE as u64) as usize];
        let cancels = self.cancels.load(Ordering::Acquire);
        // Word 0: the sound; 1: the input + 1 (0: now); 2: its frame; 3:
        // the gain and the pan (f32 bits each), the cancels in the next.
        let (slot, frame) = on.map_or((0, 0), |(s, f)| (s + 1, f));
        let gp = u64::from(cue.gain.to_bits()) | (u64::from(cue.pan.to_bits()) << 32);
        w[0].store(cue.sound as u64 | (cancels << 32), Ordering::Relaxed);
        w[1].store(slot, Ordering::Relaxed);
        w[2].store(frame, Ordering::Relaxed);
        w[3].store(gp, Ordering::Relaxed);
        self.head.store(head + 1, Ordering::SeqCst);
        true
    }

    /// The mixer thread: the next cue, if one came.
    pub(crate) fn pop(&self) -> Option<Taken> {
        let tail = self.tail.load(Ordering::Relaxed);
        if tail == self.head.load(Ordering::Acquire) {
            return None;
        }
        let w = &self.slots[(tail % QUEUE as u64) as usize];
        let (w0, slot, frame, gp) = (
            w[0].load(Ordering::Relaxed),
            w[1].load(Ordering::Relaxed),
            w[2].load(Ordering::Relaxed),
            w[3].load(Ordering::Relaxed),
        );
        self.tail.store(tail + 1, Ordering::Release);
        Some(Taken {
            cue: EffectCue {
                sound: (w0 & 0xFFFF_FFFF) as usize,
                gain: f32::from_bits(gp as u32),
                pan: f32::from_bits((gp >> 32) as u32),
            },
            on: (slot != 0).then(|| (slot - 1, frame)),
            cancels: w0 >> 32,
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
    strip: Strip,
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
        let _ = self.tx.send(Msg::Effects(strip.id(), queue.clone()));
        Effects {
            tx: self.tx.clone(),
            queue,
            pushing: Mutex::new(()),
            banks: Mutex::new(Vec::new()),
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

    /// Plays `cue` on the next period mixed; false when the queue is full
    /// (the cue is dropped).
    pub fn play(&self, cue: EffectCue) -> bool {
        self.push(cue, None)
    }

    /// Plays `cue` from frame `stream_frame` of the output open on
    /// `fader`'s input (see the module's docs); false when the queue is
    /// full.
    pub fn play_at(&self, cue: EffectCue, fader: &Fader, stream_frame: u64) -> bool {
        self.push(cue, Some((fader.slot_id(), stream_frame)))
    }

    fn push(&self, cue: EffectCue, on: Option<(u64, u64)>) -> bool {
        let _one = self.pushing.lock().unwrap_or_else(|e| e.into_inner());
        let pushed = self.queue.push(cue, on);
        if !pushed {
            self.queue.miss();
        }
        pushed
    }

    /// Drops every cue not sounding yet (a seek: their frames are of
    /// another place in the song). The voices sounding play on.
    pub fn cancel(&self) {
        self.queue.cancels.fetch_add(1, Ordering::AcqRel);
    }

    /// What it did, as of the last frames mixed.
    pub fn view(&self) -> EffectsView {
        let q = &self.queue;
        EffectsView {
            played: q.played.load(Ordering::Relaxed),
            missed: q.missed.load(Ordering::Relaxed),
            sounding: q.sounding.load(Ordering::Relaxed),
            rate: q.rate.load(Ordering::Relaxed) as u32,
        }
    }

    /// Its strip.
    pub fn strip(&self) -> &Strip {
        &self.strip
    }
}

impl Drop for Effects {
    /// Takes the input off the desk (not another made since): what the
    /// mixer thread ran of it, its bank among it, comes back and is let go
    /// of here, not there.
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
        assert!(q.push(cue, Some((3, 123_456_789_012))));
        q.cancels.store(2, Ordering::Relaxed);
        assert!(q.push(EffectCue::new(1), None));
        let a = q.pop().unwrap();
        assert_eq!(a.cue, cue);
        assert_eq!(a.on, Some((3, 123_456_789_012)));
        assert_eq!(a.cancels, 0);
        let b = q.pop().unwrap();
        assert_eq!((b.on, b.cancels), (None, 2));
        assert!(q.pop().is_none());
        // Full: refused, nothing overwritten.
        for _ in 0..QUEUE {
            assert!(q.push(EffectCue::new(0), None));
        }
        assert!(!q.push(EffectCue::new(9), None));
        assert_eq!(q.pop().unwrap().cue.sound, 0);
    }
}
