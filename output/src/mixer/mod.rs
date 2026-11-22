//! A mixer in front of a device: a desk. It owns the device, any number of
//! writers play into its inputs on the user's strips, and their gains are
//! curves on one clock.
//!
//! [`Mixer::open`] takes an [`OutputTarget`] (the device, or a test's
//! memory) and runs one thread, `jodel-mixer`, which holds the device output.
//! [`Mixer::strip`] makes a [`Strip`], a channel of the desk, and
//! [`Mixer::strip_in`] one in a group (another strip);
//! [`Mixer::input`] plugs an input into one: an [`InputTarget`], where one
//! player opens its outputs as on a device, and a [`Fader`], its gain, which
//! any number of hands may hold. The input goes when its target and every
//! fader are dropped, or at once with [`Fader::close`]. Each output opened is a full [`Output`] in the
//! device's format (pause, flush, retract, drain, its clock) with the ring
//! the player writes; the mixer drains every running one a device period
//! at a time, sums them and writes the device, as fast as the device takes
//! frames: in real time on a real device, as fast as the inputs are
//! written on one that plays what it is given at once (a file, a test's
//! memory).
//!
//! - **Two gains:** an input is heard at its [`Fader`]'s gain (the
//!   controllers': a preview's crossfade, a transition) times its strip's
//!   path (the user's: [`Strip::set_level`], [`Strip::set_mute`],
//!   [`Strip::set_solo`], solo in place). A strip's changes are heard over
//!   [`STRIP_RAMP`], a linear ramp, so they never click. A strip starts at
//!   0 dB and is kept nowhere.
//! - **Groups:** a strip may sit in a group, which may have inputs of its
//!   own. The path's gain is the product, over the strip and every group
//!   above it, of the level (0 while muted), times the solo gate: 1 while
//!   nothing is soloed anywhere, or the strip or a group above it is;
//!   else 0. So soloing a child silences its siblings, its group's own
//!   inputs and every other strip, and soloing a group keeps its children.
//!   A strip's own inputs are heard through one ramp to its level, mute and
//!   gate together (a strip at the top of the desk has only that one), its
//!   children through a second one to its level and mute, so a mute and a
//!   solo set together never multiply into a bump.
//! - **Scheduled changes:** [`Strip::set_level_at`] and
//!   [`Strip::set_mute_at`] take a [`Start`] as a fader's ramp does; one
//!   for a time starts its ramp on that very desk frame, inside the period
//!   mixed. Each setting has one change waiting at most: a new one takes
//!   its place ([`Strip::pending_mute`] and [`Strip::pending_level`] show
//!   it, so a caller can toggle against what is meant).
//! - **Bit for bit:** when one input is heard (it supplied frames, at a
//!   gain other than 0), at gain exactly 1 with no ramp running, its
//!   samples reach the device untouched ([`MasterView::passthrough`] says
//!   whether the last frames did). Otherwise the inputs are summed in
//!   `f64` with their gains, limited only where the sum could and does go
//!   over full scale (a look-ahead gain envelope), and requantized once
//!   to the device's bits with TPDF dither ([`crate::Requantizer`]); clips
//!   are counted ([`MasterView::clipped`]).
//! - **Meters:** the mixer thread measures each strip's subtree after its
//!   gains (its inputs and its children's, summed) and what goes to the
//!   device, the peak of each channel
//!   up to [`METER_CHANNELS`]; [`Mixer::meters`] takes them (the peaks
//!   since the last look). [`Mixer::desk`] shows everything else, and
//!   looking changes nothing ([`DeskView`]).
//! - **The clock:** a [`Timebase`] per device opening (its epoch): the
//!   rate and the frames mixed. A gain change is a [`Ramp`] that starts
//!   [`Start::Now`], at a time of an epoch, or at the first audio of what
//!   an input loads next; [`Mixer::crossfade`] fades inputs out on the
//!   very frame another one fades in.
//! - **Short device buffer:** the device holds [`MixerOptions::buffer`]
//!   (40 ms); the inputs keep the long rings that ride out decoding
//!   hiccups, so a gain change is heard within it. An input's position
//!   ([`Output::position`]) leaves out what the device has not played yet,
//!   so it is what is heard.
//! - **Formats:** the mixer does not resample. The [`Role::Lead`] input
//!   owns the device's format: when it opens at another one, every other
//!   output is closed ([`Error::Closed`]) and the device opens again at
//!   it. A [`Role::Guest`] never displaces one: it is offered exactly the
//!   device's format while the device is open, so its player resamples;
//!   only with nothing open does it open the device at its own.
//! - **The microphone's port:** [`Mixer::set_mic`] gives the desk a
//!   microphone. The mixer taps the device for it while a Lead output is
//!   open (again for each one, and each opening of the device), so its echo
//!   canceller hears the whole mix as its sidechain, the stream frames told
//!   in the Lead's own; the chain behind is the microphone's
//!   ([`crate::input`]: the clock bridge, the echo canceller, the singer
//!   isolated, the voice queued for its takers). The microphone has no
//!   strip and is no input: nothing of it can reach the device. The
//!   outputs on inputs run no taps of their own.
//! - **Monitor:** a [`Monitor`] sees every frame as it goes to the device
//!   (a spectrum of what is heard).

mod desk;
mod envelope;
mod input;
mod mic;
mod mix;
mod ramp;
mod run;
mod strip;
mod strips;

pub use desk::{DeskView, InputView, MasterView, Meters, MicPort, MicView, StripView};
pub use ramp::{Curve, Ramp, Start};
pub use strip::{Strip, METER_CHANNELS, STRIP_RAMP};

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::caps::Capabilities;
use crate::error::Error;
use crate::input::Input;
use crate::output::{Clock, OpenRequest, Output, OutputTarget, StreamInfo};
use input::ask;
use strip::{Change, Entry, Peaks, StripState};

/// How a [`Mixer`] runs its device.
#[derive(Clone, Debug)]
pub struct MixerOptions {
    /// The device period, and how much is mixed at a time. Defaults to
    /// 10 ms.
    pub period: Duration,
    /// The device's ring: how far ahead of what is heard the mix is
    /// written. Defaults to 40 ms.
    pub buffer: Duration,
    /// The device's capabilities, when the caller has probed them already
    /// (otherwise the mixer asks the target once, when first asked).
    pub capabilities: Option<Capabilities>,
}

impl Default for MixerOptions {
    fn default() -> Self {
        MixerOptions {
            period: Duration::from_millis(10),
            buffer: Duration::from_millis(40),
            capabilities: None,
        }
    }
}

/// What an input is to the device's format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// The main player: owns the device's format.
    Lead,
    /// Another player (a preview): plays at the device's format.
    Guest,
}

/// The mixer's clock: one per opening of the device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Timebase {
    /// Which opening of the device (from 1; 0: none yet). A time of another
    /// epoch never comes.
    pub epoch: u64,
    /// The device's rate (0: none open).
    pub rate: u32,
    /// The next frame to be mixed, from the epoch's first.
    pub frame: u64,
    /// The frame being heard.
    pub heard: u64,
}

impl Timebase {
    /// Frame `frame` of this epoch, as a [`Start::At`].
    pub fn at(&self, frame: u64) -> Start {
        let rate = f64::from(self.rate.max(1));
        Start::At {
            epoch: self.epoch,
            time: Duration::from_secs_f64(frame as f64 / rate),
        }
    }
}

/// Sees every frame the mixer gives the device, on its thread: it must not
/// block.
pub trait Monitor: Send {
    /// The device opened: what it runs at, and its clock (its position
    /// counts the device frames heard).
    fn opened(&mut self, info: &StreamInfo, clock: Clock);
    /// `samples` (interleaved, `channels` per frame) go to the device from
    /// device frame `first_frame`.
    fn mixed(&mut self, samples: &[i32], channels: usize, first_frame: u64);
    /// The device closed.
    fn closed(&mut self);
}

/// What the handles ask of the mixer thread.
pub(crate) enum Msg {
    Strip {
        id: u64,
        state: Arc<StripState>,
    },
    /// A change of a strip's level or mute, from `start`: the handles'
    /// `set`th of that setting.
    StripSet {
        id: u64,
        change: Change,
        start: Start,
        set: u64,
    },
    /// A strip soloed or not, now.
    StripSolo {
        id: u64,
        on: bool,
    },
    StripGone(u64),
    Slot {
        id: u64,
        strip: u64,
        gain: f64,
        state: Arc<SlotState>,
    },
    Forget(u64),
    Capabilities {
        slot: u64,
        done: Sender<Result<Capabilities, Error>>,
    },
    Open {
        slot: u64,
        request: OpenRequest,
        done: Sender<Result<Box<dyn Output>, Error>>,
    },
    Closed(u64),
    Flush {
        stream: u64,
        done: Sender<Result<(), Error>>,
    },
    Ramp {
        slot: u64,
        ramp: Ramp,
    },
    Crossfade {
        into: u64,
        start: Start,
        out: Vec<u64>,
        over: Duration,
    },
    Mic(Option<Input>),
    Monitor(Option<Box<dyn Monitor>>),
    /// An output started or paused: something to mix, or to settle.
    Wake,
    Quit,
}

/// What the mixer thread publishes, and what the handles keep for a look
/// at the desk. The clock is behind a lock taken every turn, for an
/// instant; what the device runs at and the microphone's port behind
/// locks of their own, taken when they change.
#[derive(Default)]
pub(crate) struct MixShared {
    timebase: Mutex<Timebase>,
    /// What the device runs at (`None`: closed).
    device: Mutex<Option<StreamInfo>>,
    mic: Mutex<MicView>,
    /// Since the start, over every opening of the device.
    clipped: AtomicU64,
    passthrough: AtomicBool,
    /// The limiter turned the gain down since the last look.
    limited: AtomicBool,
    /// What goes to the device.
    peaks: Peaks,
    /// The ids of strips, inputs and outputs given out.
    ids: AtomicU64,
    /// The strips and their inputs, by id (the handles keep it).
    strips: Mutex<BTreeMap<u64, Entry>>,
}

/// `m` locked, whether or not a thread panicked holding it.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl MixShared {
    pub(crate) fn timebase(&self) -> MutexGuard<'_, Timebase> {
        lock(&self.timebase)
    }

    pub(crate) fn device(&self) -> MutexGuard<'_, Option<StreamInfo>> {
        lock(&self.device)
    }

    pub(crate) fn mic(&self) -> MutexGuard<'_, MicView> {
        lock(&self.mic)
    }

    pub(crate) fn strips(&self) -> MutexGuard<'_, BTreeMap<u64, Entry>> {
        lock(&self.strips)
    }
}

/// What the mixer thread publishes about an input.
pub(crate) struct SlotState {
    pub(crate) role: Role,
    /// Its controllers' gain at the last frame mixed (f64 bits).
    pub(crate) gain: AtomicU64,
    /// How many ramps the mixer had taken when it last had none running
    /// or waiting.
    pub(crate) settled: AtomicU64,
    /// It was heard in the last chunk mixed.
    pub(crate) heard: AtomicBool,
    /// Its output's format (see [`desk::pack_format`]; 0: none), the frames
    /// queued in its ring, and the ring's capacity.
    pub(crate) format: AtomicU64,
    pub(crate) queued: AtomicU64,
    pub(crate) capacity: AtomicU64,
}

impl SlotState {
    fn new(role: Role, gain: f64) -> SlotState {
        SlotState {
            role,
            gain: AtomicU64::new(gain.to_bits()),
            settled: AtomicU64::new(0),
            heard: AtomicBool::new(false),
            format: AtomicU64::new(0),
            queued: AtomicU64::new(0),
            capacity: AtomicU64::new(0),
        }
    }
}

/// The mixer: owns a device, mixes its inputs into it. Dropping it closes
/// the device and every input's output ([`Error::Closed`]).
pub struct Mixer {
    tx: Sender<Msg>,
    shared: Arc<MixShared>,
    name: String,
    thread: Option<JoinHandle<()>>,
}

impl Mixer {
    /// Starts the mixer on `target`. The device opens when the first input
    /// opens an output, and closes with the last.
    pub fn open(target: Box<dyn OutputTarget>, options: MixerOptions) -> Mixer {
        let (tx, rx) = mpsc::channel();
        let shared = Arc::new(MixShared::default());
        let name = target.name();
        let run = run::Run::new(target, options, tx.clone(), shared.clone());
        let thread = std::thread::Builder::new()
            .name("jodel-mixer".into())
            .spawn(move || run.run(rx))
            .expect("spawn the mixer thread");
        Mixer {
            tx,
            shared,
            name,
            thread: Some(thread),
        }
    }

    /// A new strip named `name` at the top of the desk, after those made
    /// before.
    pub fn strip(&self, name: &str) -> Strip {
        let id = self.shared.ids.fetch_add(1, Ordering::Relaxed);
        Strip::new(id, self.tx.clone(), self.shared.clone(), name, None)
    }

    /// A new strip named `name` in the group `parent` (one of this
    /// mixer's), after those made before: its inputs are heard through
    /// its own gain and then the group's, as through a VCA (see
    /// [`Strip`]). The group keeps inputs of its own besides.
    pub fn strip_in(&self, parent: &Strip, name: &str) -> Strip {
        assert!(
            Arc::ptr_eq(&parent.handle.shared, &self.shared),
            "a strip of another mixer"
        );
        let id = self.shared.ids.fetch_add(1, Ordering::Relaxed);
        let parent = Some(parent.clone());
        Strip::new(id, self.tx.clone(), self.shared.clone(), name, parent)
    }

    /// A new input on `strip` (one of this mixer's) at `gain`: where one
    /// player opens its outputs, and its fader.
    pub fn input(&self, strip: &Strip, role: Role, gain: f64) -> (InputTarget, Fader) {
        assert!(
            Arc::ptr_eq(&strip.handle.shared, &self.shared),
            "a strip of another mixer"
        );
        let id = self.shared.ids.fetch_add(1, Ordering::Relaxed);
        let state = Arc::new(SlotState::new(role, gain));
        if let Some(entry) = self.shared.strips().get_mut(&strip.handle.id) {
            entry.inputs.insert(id, state.clone());
        }
        let _ = self.tx.send(Msg::Slot {
            id,
            strip: strip.handle.id,
            gain,
            state: state.clone(),
        });
        let slot = Arc::new(SlotHandle {
            id,
            tx: self.tx.clone(),
            state,
            ramps: AtomicU64::new(0),
            name: self.name.clone(),
            closed: AtomicBool::new(false),
            strip: strip.clone(),
        });
        (InputTarget { slot: slot.clone() }, Fader { slot })
    }

    /// An equal-power crossfade over `over`: `into` rises to 1 from `start`
    /// (as [`Fader::set`] would), and every one of `out` falls to 0 from
    /// the very frame it starts at. Should `into` get another ramp before
    /// it starts, `out` keep their gains.
    pub fn crossfade(&self, into: &Fader, start: Start, out: &[&Fader], over: Duration) {
        for f in out.iter().chain([&into]) {
            f.slot.ramps.fetch_add(1, Ordering::AcqRel);
        }
        let _ = self.tx.send(Msg::Crossfade {
            into: into.slot.id,
            start,
            out: out.iter().map(|f| f.slot.id).collect(),
            over,
        });
    }

    /// The clock, as of the last frames mixed: what a [`Ramp`] is timed
    /// on (a cheaper look than [`Mixer::desk`], which has it too).
    pub fn timebase(&self) -> Timebase {
        *self.shared.timebase()
    }

    /// The microphone listened to (`None`: none). The mixer taps the device
    /// for it while a Lead output is open (see the module's docs); how it
    /// stands is [`Mixer::mic`]. Nothing of it is ever played.
    pub fn set_mic(&self, mic: Option<Input>) {
        let _ = self.tx.send(Msg::Mic(mic));
    }

    /// Sets the [`Monitor`] that sees every frame (`None`: none).
    pub fn set_monitor(&self, monitor: Option<Box<dyn Monitor>>) {
        let _ = self.tx.send(Msg::Monitor(monitor));
    }
}

impl Drop for Mixer {
    fn drop(&mut self) {
        let _ = self.tx.send(Msg::Quit);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// What an input's target and faders share.
struct SlotHandle {
    id: u64,
    tx: Sender<Msg>,
    state: Arc<SlotState>,
    /// Ramps set on it.
    ramps: AtomicU64,
    name: String,
    /// Forgotten already ([`Fader::close`]).
    closed: AtomicBool,
    /// Its strip, kept while it lives.
    strip: Strip,
}

impl SlotHandle {
    /// Has the mixer forget the input, once.
    fn forget(&self) {
        if !self.closed.swap(true, Ordering::AcqRel) {
            let shared = &self.strip.handle.shared;
            if let Some(entry) = shared.strips().get_mut(&self.strip.handle.id) {
                entry.inputs.remove(&self.id);
            }
            let _ = self.tx.send(Msg::Forget(self.id));
        }
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

    fn open(&mut self, request: &OpenRequest) -> Result<Box<dyn Output>, Error> {
        let (slot, request) = (self.slot.id, request.clone());
        ask(&self.slot.tx, |done| Msg::Open {
            slot,
            request,
            done,
        })?
    }
}

/// An input's gain, the controllers' (its strip's is the user's). Clones
/// are the same fader.
#[derive(Clone)]
pub struct Fader {
    slot: Arc<SlotHandle>,
}

impl Fader {
    /// Changes the gain (see [`Ramp`]), in place of a ramp still waiting
    /// to start. Never blocks.
    pub fn set(&self, ramp: Ramp) {
        self.slot.ramps.fetch_add(1, Ordering::AcqRel);
        let _ = self.slot.tx.send(Msg::Ramp {
            slot: self.slot.id,
            ramp,
        });
    }

    /// The gain at the last frame mixed.
    pub fn gain(&self) -> f64 {
        f64::from_bits(self.slot.state.gain.load(Ordering::Acquire))
    }

    /// Whether the last ramp set has run to its end (one waiting for its
    /// start has not).
    pub fn settled(&self) -> bool {
        self.slot.state.settled.load(Ordering::Acquire) == self.slot.ramps.load(Ordering::Acquire)
    }

    /// Shuts the input now, while its target may still be held (by a
    /// player on its way out): its output closes ([`Error::Closed`]), and
    /// so does the device when that was the last, and every later open or
    /// question on its [`InputTarget`] fails with [`Error::Closed`], so it
    /// can never take the device again. Ramps set after it do nothing; the
    /// gain stays at its last value.
    pub fn close(&self) {
        self.slot.forget();
    }
}
