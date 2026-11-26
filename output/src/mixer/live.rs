//! The effects input's live synth on the mixer thread ([`super::Effects::live`]):
//! the commands a planner sent ([`jodel_synth::Command`], through a queue
//! of fixed size), their frames put on the synth's own as they are taken,
//! and the synth rendered a chunk at a time into a buffer of its own,
//! summed into the chunk (`effects_run.rs`).
//!
//! A command's frame is of the clock it was sent on ([`On`]): the desk's
//! own frames (the mixer's [`super::Timebase::frame`], of the device's
//! opening it was sent in), or the stream frames of the output open on an
//! input (a fader's). As the chunk from desk frame `first` is mixed, desk
//! frame `f` is the synth's `synth.frame() + (f - first)`; an output's
//! frame `f`, with the frames `g` it gave in this chunk, `synth.frame() +
//! g.at + (f - g.frame)`. A note whose frame went by more than [`LATE`]
//! ago is dropped, one late by less starts at once (a release always goes,
//! at once if late); one for an output that gave no frames yet waits for
//! its first (as long as that output is the one open there: one closed or
//! replaced first drops it). One more than [`AHEAD`] ahead is dropped (no
//! sound is planned that far: it is of a clock gone wrong), and so is one
//! on the desk's frames of an opening before this one. An output flushed
//! (a seek) calls off what waits in the synth on its frames (all that
//! waits: [`Command::Cancel`] of every tag), which are of another place in
//! the song now; what sounds rings out. The device opened again does the
//! same (and what was sent on the desk's frames before it is dropped, by
//! its epoch, as it is taken). What was put on the frames of
//! one closed plays out where it was put (a sound after the song's last
//! frame, the device playing on); a player that gives frames' numbers to
//! other music (a jump) calls off what it planned itself.
//!
//! A note is put on the synth's frames as it is taken, by where the
//! output's frames go this chunk: should the output fall short of frames
//! later (an underrun), its song is heard that much later and the note
//! does not wait for it. A note is due a beat or two ahead at most, so the
//! notes stay in step with each other and, but for such a gap, with the
//! song; holding each until its own chunk would cost a list of waiting
//! notes as long as the synth's own, for a case the device's buffer
//! keeps rare.
//!
//! The synth renders while anything sounds or waits in it, and on for
//! [`TAIL`] after (its echoes and its hall dying away); then it rests, and
//! the effects input adds nothing for it. It is heard only at the device's
//! rate: one made at another renders nothing, and its commands are
//! dropped. Nothing here allocates: the queue, the list of those waiting
//! and the render buffer have their room from the start (the buffer's for
//! a period: [`LiveRun::reserve`]).

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;

use jodel_synth::{Command, Synth, Tag};

use super::chunk::Slot;
use super::effects::{Counts, LATE};
use crate::spsc::Consumer;

/// The commands the queue to the mixer thread holds: several game events'
/// worth (the most one sends is the planner's to say, a few hundred).
pub const COMMANDS: usize = 2_048;
/// The commands waiting for their output's first frames.
const HELD: usize = 512;
/// The outputs commands were put on the synth's frames by (their ids, the
/// latest), remembered so one flushed is told.
const BOUND: usize = 4;
/// How long the synth renders on once nothing sounds or waits in it,
/// seconds: past the longest echo and the hall's tail.
pub const TAIL: f64 = 2.5;
/// The furthest ahead a command may be, seconds: past the longest sound a
/// game plans (a payout of two slow bars after its stinger).
pub const AHEAD: f64 = 15.0;
/// A sample this small is silence (under half a 24-bit step).
const SILENT: f32 = 0.5 / 8_388_607.0;

/// The clock a command's frames are on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum On {
    /// The desk's own frames, of the device's opening (its epoch) `.0`.
    Desk(u64),
    /// The stream frames of the output open on input `.0`.
    Input(u64),
}

/// A command on its way to the mixer thread.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sent {
    pub(crate) command: Command,
    pub(crate) on: On,
}

/// A command waiting for the first frames of output `output` on input
/// `slot`.
#[derive(Clone, Copy, Debug)]
struct Held {
    command: Command,
    slot: u64,
    output: u64,
}

/// Where a command goes on the synth's frames: so many frames from its
/// next, kept until its output gives frames, or dropped.
enum Place {
    At(u64),
    Hold(u64, u64),
    Drop,
}

/// The chunk commands are placed in: the device's opening, its first desk
/// frame, and how late (and how far ahead) a frame may be, in frames.
#[derive(Clone, Copy)]
struct Chunk {
    epoch: u64,
    first: u64,
    late: u64,
    ahead: u64,
}

/// The live synth as the mixer thread runs it.
pub(crate) struct LiveRun {
    rx: Consumer<Sent>,
    synth: Option<Box<Synth>>,
    held: Vec<Held>,
    bound: [Option<u64>; BOUND],
    /// Frames rendered since it was last busy or heard (past its tail: at
    /// rest).
    quiet: u64,
    /// The device's rate as last mixed at (0: none yet).
    device_rate: u32,
    /// The last chunk rendered, stereo, interleaved.
    scratch: Vec<f32>,
}

/// A command's frame, if it has one.
fn frame_of(c: &Command) -> Option<u64> {
    match *c {
        Command::Note { frame, .. } | Command::Release { frame, .. } => Some(frame),
        Command::Cancel { .. }
        | Command::Tempo { .. }
        | Command::Level { .. }
        | Command::Patch { .. } => None,
    }
}

/// `c` on frame `f`.
fn at_frame(c: Command, f: u64) -> Command {
    match c {
        Command::Note {
            patch,
            note,
            length,
            tag,
            ..
        } => Command::Note {
            frame: f,
            patch,
            note,
            length,
            tag,
        },
        Command::Release {
            patch, note, tag, ..
        } => Command::Release {
            frame: f,
            patch,
            note,
            tag,
        },
        other => other,
    }
}

/// Whether a cancel of `tag` calls `c` off.
fn called_off(c: &Command, tag: Option<Tag>) -> bool {
    let of = match *c {
        Command::Note { tag, .. } | Command::Release { tag, .. } => tag,
        _ => return false,
    };
    match tag {
        None => true,
        Some(t) => t != Tag::NONE && t == of,
    }
}

impl LiveRun {
    pub(super) fn new(rx: Consumer<Sent>) -> LiveRun {
        LiveRun {
            rx,
            synth: None,
            held: Vec::with_capacity(HELD),
            bound: [None; BOUND],
            quiet: u64::MAX,
            device_rate: 0,
            scratch: Vec::new(),
        }
    }

    /// `synth` played from now on; the one before, to be let go of
    /// elsewhere. What waited for the one before is dropped (counted).
    pub(super) fn set(&mut self, synth: Option<Box<Synth>>, counts: &Counts) -> Option<Box<Synth>> {
        self.drop_held(counts);
        self.bound = [None; BOUND];
        self.quiet = u64::MAX;
        std::mem::replace(&mut self.synth, synth)
    }

    /// Its synth, taken (to be let go of elsewhere).
    pub(super) fn take_synth(&mut self, counts: &Counts) -> Option<Box<Synth>> {
        self.set(None, counts)
    }

    /// The rate of its synth (`None`: none).
    pub(super) fn rate(&self) -> Option<u32> {
        self.synth.as_ref().map(|s| s.rate().round() as u32)
    }

    /// Room to render chunks of `frames` frames.
    pub(super) fn reserve(&mut self, frames: usize) {
        let samples = 2 * frames;
        self.scratch
            .reserve(samples.saturating_sub(self.scratch.len()));
    }

    /// Whether it renders a chunk at `rate`: its synth is of that rate, and
    /// something sounds or waits in it, or its tail is not over.
    fn rendering(&self, rate: u32) -> bool {
        let Some(synth) = self.synth.as_ref().filter(|_| self.rate() == Some(rate)) else {
            return false;
        };
        !synth.idle() || self.quiet < (TAIL * f64::from(rate)) as u64
    }

    /// Whether it may be heard for now: it renders (its tail too), or
    /// commands wait for their output.
    pub(super) fn busy(&self) -> bool {
        self.rendering(self.device_rate) || !self.held.is_empty()
    }

    /// The commands waiting for their output's frames dropped (counted).
    fn drop_held(&mut self, counts: &Counts) {
        for _ in self.held.drain(..) {
            counts.miss();
        }
    }

    /// Everything waiting called off (its frames are of another place).
    fn cancel_all(&mut self, counts: &Counts) {
        self.drop_held(counts);
        if let Some(s) = &mut self.synth {
            s.command(Command::Cancel { tag: None });
        }
    }

    /// Output `id` was flushed: what was put on its frames is called off.
    pub(super) fn flushed(&mut self, id: u64, counts: &Counts) {
        let ours = self.bound.contains(&Some(id)) || self.held.iter().any(|h| h.output == id);
        if ours {
            self.bound = [None; BOUND];
            self.cancel_all(counts);
        }
    }

    /// The device opened (again): the desk's frames start over; what waits
    /// is called off. What is still on its way on the desk's frames of an
    /// opening before is dropped (counted) as it is taken: by its epoch, so
    /// that a command sent for this opening as it opens is not (draining
    /// the queue here would drop those too).
    pub(super) fn opened(&mut self, counts: &Counts) {
        self.bound = [None; BOUND];
        self.cancel_all(counts);
    }

    /// The commands come since the last chunk, put on the synth's frames,
    /// for the chunk from desk frame `first` of the device's opening
    /// `epoch` at `rate`, the inputs as they gave their frames (`slots`).
    pub(super) fn take(
        &mut self,
        (epoch, first): (u64, u64),
        rate: u32,
        slots: &BTreeMap<u64, Slot>,
        counts: &Counts,
    ) {
        self.device_rate = rate;
        if self.rate() != Some(rate) {
            // Not heard at this rate: nothing it is sent is.
            while let Some(sent) = self.rx.pop() {
                if frame_of(&sent.command).is_some() {
                    counts.miss();
                }
            }
            self.drop_held(counts);
            return;
        }
        let chunk = Chunk {
            epoch,
            first,
            late: (LATE.as_secs_f64() * f64::from(rate)) as u64,
            ahead: (AHEAD * f64::from(rate)) as u64,
        };
        let mut k = 0;
        while k < self.held.len() {
            let h = self.held[k];
            match place(h.command, On::Input(h.slot), chunk, slots) {
                Place::Hold(_, output) if output == h.output => k += 1,
                Place::Hold(..) | Place::Drop => {
                    self.held.swap_remove(k);
                    counts.miss();
                }
                Place::At(offset) => {
                    self.held.swap_remove(k);
                    self.bind(h.slot, slots);
                    self.carry_out(h.command, offset, counts);
                }
            }
        }
        while let Some(Sent { command, on }) = self.rx.pop() {
            if frame_of(&command).is_none() {
                if let Command::Cancel { tag } = command {
                    let before = self.held.len();
                    self.held.retain(|h| !called_off(&h.command, tag));
                    for _ in self.held.len()..before {
                        counts.miss();
                    }
                }
                self.carry_out(command, 0, counts);
                continue;
            }
            match place(command, on, chunk, slots) {
                Place::At(offset) => {
                    if let On::Input(slot) = on {
                        self.bind(slot, slots);
                    }
                    self.carry_out(command, offset, counts);
                }
                Place::Hold(slot, output) if self.held.len() < HELD => {
                    self.held.push(Held {
                        command,
                        slot,
                        output,
                    });
                }
                Place::Hold(..) | Place::Drop => counts.miss(),
            }
        }
    }

    /// `command` given the synth `offset` frames from its next.
    fn carry_out(&mut self, command: Command, offset: u64, counts: &Counts) {
        let Some(synth) = &mut self.synth else {
            counts.miss();
            return;
        };
        let command = match frame_of(&command) {
            Some(_) => at_frame(command, synth.frame() + offset),
            None => command,
        };
        if synth.command(command) {
            counts.commands.fetch_add(1, Ordering::Relaxed);
        } else {
            counts.miss();
        }
    }

    /// Input `slot`'s output remembered as one commands were put on.
    fn bind(&mut self, slot: u64, slots: &BTreeMap<u64, Slot>) {
        let Some(output) = slots
            .get(&slot)
            .and_then(|s| s.feed.as_ref())
            .map(|st| st.link.id)
        else {
            return;
        };
        if self.bound.contains(&Some(output)) {
            return;
        }
        match self.bound.iter_mut().find(|b| b.is_none()) {
            Some(free) => *free = Some(output),
            None => {
                self.bound.rotate_left(1);
                self.bound[BOUND - 1] = Some(output);
            }
        }
    }

    /// The synth's next `n` frames, at `ch` channels, added into `samples`
    /// (its room for them made, zeroed where nothing else sounds), when it
    /// renders at the device's `rate`; whether anything of it was heard.
    pub(super) fn sum(&mut self, n: usize, ch: usize, rate: u32, samples: &mut [i32]) -> bool {
        if !self.rendering(rate) {
            return false;
        }
        let Some(synth) = &mut self.synth else {
            return false;
        };
        self.scratch.clear();
        self.scratch.resize(2 * n, 0.0);
        synth.render_to(&mut self.scratch);
        let mut heard = false;
        for (s, out) in self
            .scratch
            .chunks_exact(2)
            .zip(samples.chunks_exact_mut(ch))
        {
            let (l, r) = (s[0], s[1]);
            if l.abs() < SILENT && r.abs() < SILENT {
                continue;
            }
            heard = true;
            if ch == 1 {
                out[0] += super::effects_run::to_word(0.5 * (l + r));
            } else {
                out[0] += super::effects_run::to_word(l);
                out[1] += super::effects_run::to_word(r);
            }
        }
        if heard || !synth.idle() {
            self.quiet = 0;
        } else {
            self.quiet = self.quiet.saturating_add(n as u64);
        }
        heard
    }
}

/// Where `command`, on `on`, goes in `chunk`: frames from the synth's next
/// (one gone by at most its `late` at once), held for its output's first
/// frames, or dropped (a note too late, one too far ahead, an input with
/// no output open, the desk's frames of another opening).
fn place(command: Command, on: On, chunk: Chunk, slots: &BTreeMap<u64, Slot>) -> Place {
    let Some(f) = frame_of(&command) else {
        return Place::At(0);
    };
    let note = matches!(command, Command::Note { .. });
    let (base, origin) = match on {
        On::Desk(epoch) if epoch != chunk.epoch => return Place::Drop,
        On::Desk(_) => (0, chunk.first),
        On::Input(slot) => {
            let Some(stream) = slots.get(&slot).and_then(|s| s.feed.as_ref()) else {
                return Place::Drop;
            };
            let Some(g) = stream.given() else {
                return Place::Hold(slot, stream.link.id);
            };
            (g.at as u64, g.frame)
        }
    };
    if f >= origin {
        let offset = base + (f - origin);
        if offset > chunk.ahead {
            Place::Drop
        } else {
            Place::At(offset)
        }
    } else if note && origin - f > chunk.late {
        Place::Drop
    } else {
        Place::At(base)
    }
}
