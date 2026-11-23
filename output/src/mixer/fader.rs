//! An input's handles: its [`InputTarget`], where one player opens its
//! outputs, and its [`Fader`]s, the controllers' gains: the input's own
//! and, for an input with parts ([`Mixer::input_lanes`]), one per part.

use std::sync::atomic::{fence, AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::Duration;

use super::desk::unpack_format;
use super::input::ask;
use super::ramp::GainId;
use super::{Mixer, Msg, Ramp, Role, Start, Strip};
use crate::caps::Capabilities;
use crate::error::Error;
use crate::output::{OpenRequest, Output, OutputTarget};

/// What the mixer thread publishes about a fader's gain.
pub(crate) struct FaderState {
    /// The gain at the last frame mixed (f64 bits).
    pub(crate) gain: AtomicU64,
    /// How many ramps the mixer had taken when it last had none running or
    /// waiting.
    pub(crate) settled: AtomicU64,
}

/// What the mixer thread publishes about a lane.
pub(crate) struct LaneState {
    pub(crate) strip: u64,
    /// The gain it was heard at, at the last frame mixed (f64 bits).
    pub(crate) gain: AtomicU64,
    pub(crate) heard: AtomicBool,
}

/// What the mixer thread publishes about the switch of an input with
/// parts.
#[derive(Default)]
pub(crate) struct PartsState {
    /// The control word of the last frame mixed.
    pub(crate) control: AtomicU8,
    /// The parts' share at the last frame mixed (f64 bits).
    pub(crate) share: AtomicU64,
}

/// What the mixer thread publishes about an input.
pub(crate) struct SlotState {
    pub(crate) role: Role,
    /// Its faders: its own, then one per part.
    pub(crate) faders: Box<[FaderState]>,
    /// Its lanes: main, then the parts.
    pub(crate) lanes: Box<[LaneState]>,
    pub(crate) parts: Option<PartsState>,
    /// A lane of it was heard in the last chunk mixed.
    pub(crate) heard: AtomicBool,
    /// Its output's format (see [`super::desk::pack_format`]; 0: none),
    /// the frames queued in its ring, the ring's capacity, and the late
    /// frames it dropped following another.
    pub(crate) format: AtomicU64,
    pub(crate) queued: AtomicU64,
    pub(crate) capacity: AtomicU64,
    pub(crate) dropped: AtomicU64,
    /// Where its output's frames went last on the device.
    pub(crate) mapped: Mapped,
}

/// The device epoch (0: none, not mapped) and the device frame less the
/// stream frame where an input's output's frames went last, written by the
/// mixer thread and read by any: a sequence lock (the count is odd while a
/// write is under way, so a reader that sees the same even count before
/// and after has both of one write).
#[derive(Default)]
pub(crate) struct Mapped {
    seq: AtomicU64,
    epoch: AtomicU64,
    offset: AtomicU64,
}

impl Mapped {
    /// The mixer thread: frames of epoch `epoch` go `offset` device frames
    /// on from their stream frames (epoch 0: not mapped).
    pub(crate) fn store(&self, epoch: u64, offset: u64) {
        let seq = self.seq.load(Ordering::Relaxed);
        self.seq.store(seq + 1, Ordering::Relaxed);
        fence(Ordering::Release);
        self.epoch.store(epoch, Ordering::Relaxed);
        self.offset.store(offset, Ordering::Relaxed);
        self.seq.store(seq + 2, Ordering::Release);
    }

    /// The epoch and the offset of one write (`None`: not mapped).
    pub(crate) fn load(&self) -> Option<(u64, u64)> {
        loop {
            let seq = self.seq.load(Ordering::Acquire);
            if seq % 2 == 1 {
                std::hint::spin_loop();
                continue;
            }
            let epoch = self.epoch.load(Ordering::Relaxed);
            let offset = self.offset.load(Ordering::Relaxed);
            fence(Ordering::Acquire);
            if self.seq.load(Ordering::Relaxed) == seq {
                return (epoch != 0).then_some((epoch, offset));
            }
        }
    }
}

impl SlotState {
    fn new(role: Role, gain: f64, strips: &[u64]) -> SlotState {
        let fader = |g: f64| FaderState {
            gain: AtomicU64::new(g.to_bits()),
            settled: AtomicU64::new(0),
        };
        SlotState {
            role,
            faders: (0..strips.len())
                .map(|k| fader(if k == 0 { gain } else { 1.0 }))
                .collect(),
            lanes: strips
                .iter()
                .map(|&strip| LaneState {
                    strip,
                    gain: AtomicU64::new(0),
                    heard: AtomicBool::new(false),
                })
                .collect(),
            parts: (strips.len() > 1).then(PartsState::default),
            heard: AtomicBool::new(false),
            format: AtomicU64::new(0),
            queued: AtomicU64::new(0),
            capacity: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            mapped: Mapped::default(),
        }
    }
}

impl Mixer {
    /// A new input on `strip` (one of this mixer's) at `gain`: where one
    /// player opens its outputs, and its fader. Its outputs are plain
    /// streams.
    pub fn input(&self, strip: &Strip, role: Role, gain: f64) -> (InputTarget, Fader) {
        let (target, fader, _) = self.input_lanes(strip, &[], role, gain);
        (target, fader)
    }

    /// A new input whose outputs may be lane streams ([`crate::lanes`]):
    /// main on `main`, part k on `parts[k - 1]` (all of this mixer's; every
    /// part's strip a child of main's, so main's group level and mute act on
    /// the parts as on main, and a part soloed is told from main by its
    /// gate), at `gain`. Returns where one player opens its outputs, the input's
    /// fader (over every lane: a duck), and one fader per part (the
    /// controllers' gain of that part). The input offers
    /// [`OutputTarget::lanes`] = 1 + the parts; an output opened with one
    /// lane plays main alone.
    pub fn input_lanes(
        &self,
        main: &Strip,
        parts: &[&Strip],
        role: Role,
        gain: f64,
    ) -> (InputTarget, Fader, Vec<Fader>) {
        let strips: Vec<Strip> = [main]
            .into_iter()
            .chain(parts.iter().copied())
            .cloned()
            .collect();
        for s in &strips {
            assert!(
                Arc::ptr_eq(&s.handle.shared, &self.shared),
                "a strip of another mixer"
            );
        }
        for p in parts {
            assert_eq!(
                p.parent(),
                Some(main.id()),
                "a part's strip is a child of main's"
            );
        }
        let ids: Vec<u64> = strips.iter().map(Strip::id).collect();
        let id = self.shared.ids.fetch_add(1, Ordering::Relaxed);
        let state = Arc::new(SlotState::new(role, gain, &ids));
        if let Some(entry) = self.shared.strips().get_mut(&main.handle.id) {
            entry.inputs.insert(id, state.clone());
        }
        let _ = self.tx.send(Msg::Slot {
            id,
            lanes: ids,
            gain,
            state: state.clone(),
        });
        let slot = Arc::new(SlotHandle {
            id,
            tx: self.tx.clone(),
            state,
            ramps: (0..strips.len()).map(|_| AtomicU64::new(0)).collect(),
            name: self.name.clone(),
            closed: AtomicBool::new(false),
            strips,
        });
        let fader = |lane: u16| Fader {
            slot: slot.clone(),
            lane,
        };
        let parts = (1..=parts.len() as u16).map(fader).collect();
        (InputTarget { slot: slot.clone() }, fader(0), parts)
    }

    /// An equal-power crossfade over `over`: `into` rises to 1 from `start`
    /// (as [`Fader::set`] would), and every one of `out` falls to 0 from
    /// the very frame it starts at. Should `into` get another ramp before
    /// it starts, `out` keep their gains.
    pub fn crossfade(&self, into: &Fader, start: Start, out: &[&Fader], over: Duration) {
        for f in out.iter().chain([&into]) {
            f.ramps().fetch_add(1, Ordering::AcqRel);
        }
        let _ = self.tx.send(Msg::Crossfade {
            into: into.id(),
            start,
            out: out.iter().map(|f| f.id()).collect(),
            over,
        });
    }
}

/// What an input's target and faders share.
struct SlotHandle {
    id: u64,
    tx: Sender<Msg>,
    state: Arc<SlotState>,
    /// Ramps set on each fader.
    ramps: Box<[AtomicU64]>,
    name: String,
    /// Forgotten already ([`Fader::close`]).
    closed: AtomicBool,
    /// Its lanes' strips, main's first, kept while it lives.
    strips: Vec<Strip>,
}

impl SlotHandle {
    /// Has the mixer forget the input, once.
    fn forget(&self) {
        if self.close() {
            let _ = self.tx.send(Msg::Forget(self.id));
        }
    }

    /// Marks the input closed and takes it off its strip; whether it was
    /// open until now.
    fn close(&self) -> bool {
        if self.closed.swap(true, Ordering::AcqRel) {
            return false;
        }
        let main = &self.strips[0].handle;
        if let Some(entry) = main.shared.strips().get_mut(&main.id) {
            entry.inputs.remove(&self.id);
        }
        true
    }
}

impl Drop for SlotHandle {
    fn drop(&mut self) {
        self.forget();
    }
}

/// Where one player opens its outputs on an input of a [`Mixer`] (one at a
/// time: opening another closes the one before).
pub struct InputTarget {
    slot: Arc<SlotHandle>,
}

impl OutputTarget for InputTarget {
    fn name(&self) -> String {
        self.slot.name.clone()
    }

    /// What the input takes, fixed when first asked: a Lead the device's
    /// capabilities; a guest exactly the device's format while it is open,
    /// else the device's capabilities.
    fn capabilities(&mut self) -> Result<Capabilities, Error> {
        let slot = self.slot.id;
        ask(&self.slot.tx, |done| Msg::Capabilities { slot, done })?
    }

    /// Opens an output of one lane, or of as many as the input has
    /// ([`OutputTarget::lanes`]; anything else is
    /// [`Error::InvalidArgument`]).
    fn open(&mut self, request: &OpenRequest) -> Result<Box<dyn Output>, Error> {
        let (slot, request) = (self.slot.id, request.clone());
        ask(&self.slot.tx, |done| Msg::Open {
            slot,
            request,
            done,
        })?
    }

    /// Main and the parts: 1 for a plain input.
    fn lanes(&mut self) -> u16 {
        self.slot.strips.len() as u16
    }
}

/// A controllers' gain on an input: the input's own (over all of it), or
/// one of its parts' ([`Mixer::input_lanes`]); its strips' are the user's.
/// Clones are the same fader.
#[derive(Clone)]
pub struct Fader {
    slot: Arc<SlotHandle>,
    /// 0: the input's own; k: part k's.
    lane: u16,
}

impl Fader {
    fn id(&self) -> GainId {
        GainId::new(self.slot.id, self.lane)
    }

    fn ramps(&self) -> &AtomicU64 {
        &self.slot.ramps[usize::from(self.lane)]
    }

    /// Changes the gain (see [`Ramp`]), in place of a ramp still waiting
    /// to start. Never blocks.
    pub fn set(&self, ramp: Ramp) {
        self.ramps().fetch_add(1, Ordering::AcqRel);
        let _ = self.slot.tx.send(Msg::Ramp {
            gain: self.id(),
            ramp,
        });
    }

    /// The gain at the last frame mixed.
    pub fn gain(&self) -> f64 {
        let state = &self.slot.state.faders[usize::from(self.lane)];
        f64::from_bits(state.gain.load(Ordering::Acquire))
    }

    /// The late frames the output open on its input dropped to keep up
    /// with the one it follows ([`Fader::follow`]), since it was opened
    /// (as its [`crate::Stats::dropped`]), as of the last frames mixed; 0
    /// with none open.
    pub fn dropped(&self) -> u64 {
        self.slot.state.dropped.load(Ordering::Relaxed)
    }

    /// Whether the last ramp set has run to its end (one waiting for its
    /// start has not).
    pub fn settled(&self) -> bool {
        let state = &self.slot.state.faders[usize::from(self.lane)];
        state.settled.load(Ordering::Acquire) == self.ramps().load(Ordering::Acquire)
    }

    /// When frame `stream_frame` of its input's output is mixed, as that
    /// output plays now: a [`Start::At`] for a ramp, or a strip's change,
    /// to land on that very frame. It is the input's, whichever of its
    /// faders is asked: where its output's last frames mixed went on the
    /// device; for an output that follows another ([`Fader::follow`]), its
    /// leader's at `stream_frame + origin` from the moment it is linked,
    /// mixed or not. With nothing of the output mixed since it was opened
    /// or flushed in this opening of the device (nor of its leader's),
    /// [`Start::Now`].
    pub fn at_stream(&self, stream_frame: u64) -> Start {
        let state = &self.slot.state;
        let rate = unpack_format(state.format.load(Ordering::Relaxed)).map_or(0, |f| f.rate);
        let now = self.slot.strips[0]
            .handle
            .shared
            .epoch
            .load(Ordering::Acquire);
        match state.mapped.load() {
            Some((epoch, offset)) if rate > 0 && epoch == now => {
                let frame = stream_frame.wrapping_add(offset);
                Start::At {
                    epoch,
                    time: Duration::from_secs_f64(frame as f64 / f64::from(rate)),
                }
            }
            _ => Start::Now,
        }
    }

    /// Has the output open on this fader's input follow the one open on
    /// `leader`'s input (a fader of another input of the same mixer): its
    /// stream frame 0 lands on the leader's stream frame `origin`, and from
    /// there it moves with the leader's stream, frame for frame (see the
    /// module's docs). Set it once the output is open, before it is
    /// written: what it gave before is not taken back. The link holds
    /// until either output is flushed or closed (a new one opened on
    /// either input closes the old), or [`Fader::unfollow`]; another link
    /// takes its place. Waits for the mixer thread (a device period,
    /// about); then [`Fader::at_stream`] is the leader's, `origin` on.
    ///
    /// [`Error::InvalidArgument`] when either input has no output open, an
    /// input would follow itself, the leader follows another, or this
    /// input leads one (no chains); [`Error::Closed`] when either input is
    /// gone.
    pub fn follow(&self, leader: &Fader, origin: u64) -> Result<(), Error> {
        let same = Arc::ptr_eq(
            &self.slot.strips[0].handle.shared,
            &leader.slot.strips[0].handle.shared,
        );
        assert!(same, "a leader of another mixer");
        let (slot, leader) = (self.slot.id, Some((leader.slot.id, origin)));
        ask(&self.slot.tx, |done| Msg::Follow {
            slot,
            leader,
            done: Some(done),
        })?
    }

    /// The output on this fader's input follows no one any more: it plays
    /// on from where it is, as any other. Never blocks.
    pub fn unfollow(&self) {
        let _ = self.slot.tx.send(Msg::Follow {
            slot: self.slot.id,
            leader: None,
            done: None,
        });
    }

    /// Shuts the input now, while its target may still be held (by a
    /// player on its way out): its output closes ([`Error::Closed`]), and
    /// so does the device when that was the last, and every later open or
    /// question on its [`InputTarget`] fails with [`Error::Closed`], so it
    /// can never take the device again. Ramps set after it do nothing; the
    /// gains stay at their last values.
    pub fn close(&self) {
        self.slot.forget();
    }

    /// Shuts the input as [`Fader::close`] does, but only while nothing of
    /// it has been heard: silent, with a ramp waiting to start (a fade in
    /// waiting for its first audio). That ramp never comes then, and those
    /// set to start with it ([`Mixer::crossfade`]) keep their gains. Once
    /// its ramp is under way, if only for a frame, nothing changes: it is
    /// heard, and what fell as it rose is down. The mixer thread decides,
    /// on the frame it is at, so this waits for its answer (a device
    /// period, about); returns whether the input was shut (an input shut
    /// already counts).
    pub fn withdraw(&self) -> bool {
        if self.slot.closed.load(Ordering::Acquire) {
            return true;
        }
        let gain = self.id();
        let shut = ask(&self.slot.tx, |done| Msg::Withdraw { gain, done }).unwrap_or(true);
        if shut {
            self.slot.close();
        }
        shut
    }
}
