//! A mixer in front of a device: it owns the device, any number of writers
//! play into its inputs, and their gains are curves on one clock.
//!
//! [`Mixer::open`] takes an [`OutputTarget`] (the device, or a test's
//! memory) and runs one thread, `jodel-mixer`, which holds the device output.
//! [`Mixer::input`] makes an input: an [`InputTarget`], where one player
//! opens its outputs as on a device, and a [`Fader`], its gain, which any
//! number of hands may hold. The input goes when its target and every
//! fader are dropped, or at once with [`Fader::close`]. Each output opened is a full [`Output`] in the
//! device's format (pause, flush, retract, drain, its clock) with the ring
//! the player writes; the mixer drains every running one a device period
//! at a time, sums them and writes the device, as fast as the device takes
//! frames: in real time on a real device, as fast as the inputs are
//! written on one that plays what it is given at once (a file, a test's
//! memory).
//!
//! - **Bit for bit:** when one input is heard (it supplied frames, at a
//!   gain other than 0), at gain exactly 1 with no ramp running, its
//!   samples reach the device untouched ([`Mixer::passthrough`] says
//!   whether the last frames did). Otherwise the inputs are summed in
//!   `f64` with their gains, limited only where the sum could and does go
//!   over full scale (a look-ahead gain envelope), and requantized once
//!   to the device's bits with TPDF dither ([`crate::Requantizer`]); clips
//!   are counted ([`Mixer::clipped`]).
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
//! - **Taps:** a Lead output's [`Output::set_tap`] goes to the device, so a
//!   microphone's echo canceller hears the whole mix; the stream frames it
//!   is told are the Lead output's.
//! - **Monitor:** a [`Monitor`] sees every frame as it goes to the device
//!   (a spectrum of what is heard).

mod input;
mod mix;
mod ramp;
mod run;

pub use ramp::{Curve, Ramp, Start};

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::caps::Capabilities;
use crate::error::Error;
use crate::output::{Clock, OpenRequest, Output, OutputTarget, StreamInfo};
use crate::tap::Tap;
use input::ask;

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
    Slot {
        id: u64,
        role: Role,
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
    Tap {
        stream: u64,
        tap: Option<Box<dyn Tap>>,
        done: Sender<Result<(), Error>>,
    },
    Monitor(Option<Box<dyn Monitor>>),
    /// An output started or paused: something to mix, or to settle.
    Wake,
    Quit,
}

/// What the mixer thread publishes.
#[derive(Default)]
pub(crate) struct MixShared {
    timebase: Mutex<Timebase>,
    clipped: AtomicU64,
    passthrough: AtomicBool,
    /// The ids of inputs and outputs given out.
    ids: AtomicU64,
}

/// What the mixer thread publishes about an input.
#[derive(Default)]
pub(crate) struct SlotState {
    /// Its gain at the last frame mixed (f64 bits).
    gain: AtomicU64,
    /// How many ramps the mixer had taken when it last had none running
    /// or waiting.
    settled: AtomicU64,
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

    /// A new input at `gain`: where one player opens its outputs, and its
    /// fader.
    pub fn input(&self, role: Role, gain: f64) -> (InputTarget, Fader) {
        let id = self.shared.ids.fetch_add(1, Ordering::Relaxed);
        let state = Arc::new(SlotState::default());
        state.gain.store(gain.to_bits(), Ordering::Relaxed);
        let _ = self.tx.send(Msg::Slot {
            id,
            role,
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

    /// The clock, as of the last frames mixed.
    pub fn timebase(&self) -> Timebase {
        *self
            .shared
            .timebase
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Samples that clipped when a mix was requantized, since the start.
    pub fn clipped(&self) -> u64 {
        self.shared.clipped.load(Ordering::Relaxed)
    }

    /// Whether the last frames went to the device as an input wrote them
    /// (or were silence): nothing summed, limited or dithered.
    pub fn passthrough(&self) -> bool {
        self.shared.passthrough.load(Ordering::Relaxed)
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
}

impl SlotHandle {
    /// Has the mixer forget the input, once.
    fn forget(&self) {
        if !self.closed.swap(true, Ordering::AcqRel) {
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

/// An input's gain. Clones are the same fader.
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
