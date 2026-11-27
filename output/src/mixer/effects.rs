//! The desk's effects input ([`Mixer::effects`]): a game's sounds on a
//! strip of their own, summed with the inputs after their gains (so the
//! microphone's tap on the device hears them, as the song). They come from
//! a synth played live ([`Effects::live`]): a planner sends it timed notes
//! ([`jodel_synth::Command`], [`Effects::commands`]) on the stream frames
//! of the output open on a fader's input (so each lands on its moment of
//! the song, inside the period it falls in), or on the desk's own frames
//! ([`super::Timebase::frame`]: a sound after the song, when no output is
//! open); the mixer thread puts them on the synth's frames as it takes
//! them, and renders the synth (see `live.rs`: one late by more than
//! [`LATE`] is dropped, a flush of the output calls off what waits on its
//! frames). A patch changed as it plays goes by a queue of its own, taken
//! first each chunk ([`LiveCommands::patch`]). The synth is heard at the
//! device's rate: the caller makes one again when the device opens at
//! another.
//!
//! The desk's device plays all along in real time (see [`Mixer`]), so a
//! sound on the desk's frames plays whatever the outputs do (a song over,
//! paused, or the next held back). On a device that plays only what it is
//! given it waits for an output to run.
//!
//! While the synth rests, the effects input adds nothing, and what goes to
//! the device is as it was without it, bit for bit.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use jodel_synth::{Command, Commands, Patch, PatchId, Synth};

use super::input::ask;
use super::live::{Changed, On, Sent, COMMANDS, PATCHES};
use super::{Fader, Mixer, Msg, Strip};
use crate::spsc::{Consumer, Producer, Spsc};

/// How late a live synth's note may start, past its frame.
pub const LATE: Duration = Duration::from_millis(80);

/// What the effects input did, since it was made.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EffectsView {
    /// Commands dropped: the queue full, a note too late or its output
    /// gone, the synth's list of those waiting full, a synth at another
    /// rate, or called off by a flush.
    pub missed: u64,
    /// Commands the live synth took.
    pub commands: u64,
    /// The rate of the live synth (0: none).
    pub live: u32,
}

/// What the mixer thread tells back of the effects input: what it took and
/// what it dropped, and its synth's rate.
#[derive(Debug, Default)]
pub(crate) struct Counts {
    pub(crate) missed: AtomicU64,
    pub(crate) commands: AtomicU64,
    pub(crate) live_rate: AtomicU64,
}

impl Counts {
    /// One more command dropped.
    pub(crate) fn miss(&self) {
        self.missed.fetch_add(1, Ordering::Relaxed);
    }
}

/// The desk's effects input, on its strip ([`Mixer::effects`]). Dropping
/// it takes the input off the desk (what sounds stops).
pub struct Effects {
    tx: Sender<Msg>,
    counts: Arc<Counts>,
    /// The commands' and the patches' ways to the live synth (one sender
    /// at a time).
    live: Mutex<Senders>,
    strip: Strip,
}

/// The two queues to the live synth: its commands, and the patches
/// changed.
struct Senders {
    commands: Producer<Sent>,
    patches: Producer<Changed>,
}

/// Where a planner's commands go to the live synth ([`Effects::commands`]):
/// each on the clock it was made for. It holds the queue's producer lock
/// while it lives (another sender waits for it): keep it for one planning
/// step, as long as an event or a clock takes, never across a wait.
pub struct LiveCommands<'a> {
    tx: MutexGuard<'a, Senders>,
    on: On,
    counts: &'a Counts,
}

impl LiveCommands<'_> {
    /// `patch` is `to` from the next chunk the mixer takes on (an edit:
    /// [`Synth::set_patch`]), for the notes waiting in the synth already
    /// too; whether it was taken (not when its queue is full: counted).
    pub fn patch(&mut self, patch: PatchId, to: Patch) -> bool {
        let taken = self.tx.patches.push((patch, to));
        if !taken {
            self.counts.miss();
        }
        taken
    }
}

impl Commands for LiveCommands<'_> {
    fn send(&mut self, c: Command) -> bool {
        let sent = self.tx.commands.push(Sent {
            command: c,
            on: self.on,
        });
        if !sent {
            self.counts.miss();
        }
        sent
    }
}

/// A new queue of `capacity` values, both its ends.
fn ends<T: Copy>(capacity: usize) -> (Producer<T>, Consumer<T>) {
    let q = Spsc::new(capacity);
    (
        q.producer().expect("a new queue"),
        q.consumer().expect("a new queue"),
    )
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
        let counts = Arc::new(Counts::default());
        let (commands, rx) = ends(COMMANDS);
        let (patches, patches_rx) = ends(PATCHES);
        let _ = self
            .tx
            .send(Msg::Effects(strip.id(), counts.clone(), rx, patches_rx));
        Effects {
            tx: self.tx.clone(),
            counts,
            live: Mutex::new(Senders { commands, patches }),
            strip: strip.clone(),
        }
    }
}

impl Effects {
    /// `synth` played live from now on, as it renders (its level is its
    /// own bus's, under its limiter's ceiling), in place of any before
    /// (which is let go of off the mixer thread); heard while the device
    /// runs at its rate. Its commands come through [`Effects::commands`].
    pub fn live(&self, synth: Synth) {
        let _ = self.tx.send(Msg::EffectsLive(Box::new(synth)));
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
            counts: &self.counts,
        }
    }

    /// What it did, as of the last frames mixed.
    pub fn view(&self) -> EffectsView {
        let c = &self.counts;
        EffectsView {
            missed: c.missed.load(Ordering::Relaxed),
            commands: c.commands.load(Ordering::Relaxed),
            live: c.live_rate.load(Ordering::Relaxed) as u32,
        }
    }

    /// Its strip.
    pub fn strip(&self) -> &Strip {
        &self.strip
    }
}

impl Drop for Effects {
    /// Takes the input off the desk (not another made since): what the
    /// mixer thread ran of it, its synth among it, comes back and is let
    /// go of here, not there.
    fn drop(&mut self) {
        let counts = self.counts.clone();
        let run = ask(&self.tx, |done| Msg::EffectsGone { counts, done });
        drop(run);
    }
}
