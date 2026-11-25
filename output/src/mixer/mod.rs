//! A mixer in front of a device: a desk. It owns the device, any number of
//! writers play into its inputs on the user's strips, and their gains are
//! curves on one clock.
//!
//! [`Mixer::open`] takes an [`OutputTarget`] (the device, or a test's
//! memory) and runs one thread, `jodel-mixer`, which opens the device at
//! once at the desk's format ([`desk_format`]: stereo, at the system's mix
//! rate) and holds it open until the mixer is dropped. In real time it
//! plays all along: silence where nothing sounds, never paused, never
//! closed between songs. A device that is not there as the mixer starts,
//! or that fails, is asked again what it takes, and opened, by the next
//! output opened (a failure's is another epoch). Another device is another
//! mixer.
//! [`Mixer::strip`] makes a [`Strip`], a channel of the desk, and
//! [`Mixer::strip_in`] one in a group (another strip);
//! [`Mixer::input`] plugs an input into one: an [`InputTarget`], where one
//! player opens its outputs as on a device, and a [`Fader`], its gain, which
//! any number of hands may hold. The input goes when its target and every
//! fader are dropped, or at once with [`Fader::close`] (or with
//! [`Fader::withdraw`], only while nothing of it has been heard). Each
//! output opened is a full [`Output`] in the desk's format (pause,
//! flush, retract, drain, its clock) with the ring the player writes; the
//! mixer drains every running one a device period at a time, sums them and
//! writes the device, as fast as the device takes frames: in real time on
//! a real device, as fast as the inputs are written on one that plays what
//! it is given at once (a file, a test's memory).
//!
//! - **Why the mixer does not wait for a late input:** the input's own ring
//!   (half a second, the player's) is the buffer that rides out hiccups,
//!   so an input short of a period is far behind, and a short wait would
//!   not save it; the device's headroom, its short ring, must stay full,
//!   or every input glitches when the mixer thread wakes late. Waiting
//!   while the device had frames to spare was tried, and underran on
//!   macOS.
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
//! - **Links:** [`Fader::follow`] has an input's output follow another
//!   input's: its frame 0 lands on the leader's stream frame `origin`
//!   (inside a period), and from there it moves with the leader's stream,
//!   frame for frame, never with the device's. While the leader gives
//!   nothing (paused, short of frames) the follower holds, and loses
//!   nothing; a follower behind (short of frames itself) drops its late
//!   frames to catch up ([`Fader::dropped`]), so the mixer never waits
//!   for it. A flush or a close of either side ends the link (see
//!   `follow.rs`).
//! - **Lane inputs:** [`Mixer::input_lanes`] makes an input whose outputs
//!   may be lane streams ([`crate::lanes`]): main on one strip, each part
//!   on its own, with the input's fader over all of them and a fader per
//!   part. Lanes are the units heard. The input switches between main and
//!   the parts over [`SWITCH_FADE`] (see `parts.rs`): to the parts when
//!   they are touched and the control word says they are there, back to
//!   main, bit for bit, once nothing is touched and every part's ramps
//!   have settled; across a gap in the stream (a seek, a resume, an
//!   underrun) it keeps what it knew and is back at its aim at once.
//!   [`Fader::at_stream`] times a change on a stream frame of the input's
//!   output.
//! - **Automation:** [`Fader::automate`] gives an input gain curves on the
//!   stream frames of its output ([`Automation`], [`GainCurve`]): its own,
//!   main's and each part's, heard on top of every fader and strip, moving
//!   only as its stream does (a pad or a pause holds them), and replaced
//!   whole, smoothed from the gains heard. A part's curve away from 1 has
//!   the input heard on its parts; once every curve has ended at exactly
//!   1 the input is back on main, bit for bit (see `automation.rs`).
//! - **Bit for bit:** when one exact lane is heard (main, or a plain
//!   input's; it supplied frames, at a gain other than 0), at gain exactly
//!   1 with no ramp running, its samples reach the device untouched
//!   ([`MasterView::passthrough`] says whether the last frames did); a
//!   part never does. Otherwise the lanes are summed in
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
//! - **One format:** the mixer does not resample. Every input is offered
//!   the desk's one format ([`InputTarget`]'s capabilities), so each
//!   player resamples to it (songs, stems, the preview); an output asked
//!   at another is refused ([`Error::FormatNotAccepted`]). A song at the
//!   desk's rate plays bit for bit (see "Bit for bit"). An input's
//!   [`Role`] says which is the main player, the one the microphone's
//!   frames are told in; [`Fader::set_role`] changes it as it plays (the
//!   incoming song of a transition taking the lead).
//! - **The microphone's port:** [`Mixer::set_mic`] gives the desk a
//!   microphone. The mixer taps the device for it while a Lead output is
//!   open (again for each one), so its echo canceller hears the whole mix
//!   as its sidechain (at the one rate all along: what it has learnt of
//!   the room is kept from song to song), the stream frames told
//!   in the Lead's own; the chain behind is the microphone's
//!   ([`crate::input`]: the clock bridge, the echo canceller, the singer
//!   isolated, the voice queued for its takers). The microphone has no
//!   strip and is no input: nothing of it can reach the device. The
//!   outputs on inputs run no taps of their own.
//! - **Effects:** [`Mixer::effects`] gives the desk an effects input on a
//!   strip of its own: short sounds (a game's) from a bank at the
//!   device's rate, each started on a frame of an input's stream, summed
//!   with the inputs after the strip's gain; while none sounds, nothing
//!   of it reaches the device (see `effects.rs`); one for now sounds on
//!   the next period, a song playing or not. It is in what the
//!   microphone's tap hears, as everything summed is.
//! - **Monitor:** a [`Monitor`] sees every frame as it goes to the device
//!   (a spectrum of what is heard).

mod aside;
mod automation;
mod chunk;
mod desk;
mod effects;
mod envelope;
mod fader;
mod follow;
mod input;
mod mic;
mod mix;
mod parts;
mod ramp;
mod rate;
mod run;
mod strip;
mod strips;
mod voices;

pub use crate::lanes::SWITCH_FADE;
pub use automation::{Automation, AutomationView, GainCurve, CUT_FADE};
pub use desk::{
    DeskView, InputView, LaneView, MasterView, Meters, MicPort, MicView, PartsView, StripView,
};
pub use effects::{EffectCue, Effects, EffectsView, SoundBank, LATE as EFFECT_LATE, VOICES};
pub use fader::{Fader, InputTarget};
pub use ramp::{equal_power, Curve, Ramp, Start};
pub use rate::{desk_format, DESK_CHANNELS};
pub use strip::{Strip, METER_CHANNELS, STRIP_RAMP};

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender, SyncSender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::caps::Capabilities;
use crate::error::Error;
use crate::format::StreamFormat;
use crate::input::Input;
use crate::output::{Clock, OutputTarget, StreamInfo};
use crate::seqlock::{Packed, SeqLock};
use fader::SlotState;
use ramp::GainId;
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
    /// (otherwise the mixer asks the target, as it opens).
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

/// What an input is to the desk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// The main player: the microphone's frames are told in its.
    Lead,
    /// Another player (a preview).
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

impl Packed for Timebase {
    const WORDS: usize = 4;

    fn pack(&self, words: &mut [u64]) {
        words.copy_from_slice(&[self.epoch, u64::from(self.rate), self.frame, self.heard]);
    }

    fn unpack(words: &[u64]) -> Timebase {
        Timebase {
            epoch: words[0],
            rate: words[1] as u32,
            frame: words[2],
            heard: words[3],
        }
    }
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
    /// An input on the strips `lanes` (main's first).
    Slot {
        id: u64,
        lanes: Vec<u64>,
        gain: f64,
        state: Arc<SlotState>,
    },
    Forget(u64),
    /// Forget the input of fader `gain` if it is silent with a ramp waiting
    /// to start; says whether it was.
    Withdraw {
        gain: GainId,
        done: SyncSender<bool>,
    },
    Capabilities {
        slot: u64,
        done: SyncSender<Result<Capabilities, Error>>,
    },
    /// An output on input `slot`, made by its opener ([`input::open`]).
    Open {
        slot: u64,
        stream: input::Stream,
        done: SyncSender<Result<input::Opened, Error>>,
    },
    Closed(u64),
    Flush {
        stream: u64,
        done: SyncSender<Result<(), Error>>,
    },
    /// The output on input `slot` follows the one on input `leader.0` from
    /// its frame `leader.1` (`None`: no one); `done` hears whether it
    /// does.
    Follow {
        slot: u64,
        leader: Option<(u64, u64)>,
        done: Option<SyncSender<Result<(), Error>>>,
    },
    Ramp {
        gain: GainId,
        ramp: Ramp,
    },
    /// Input `slot` is heard through `automation` (`None`: none).
    Automate {
        slot: u64,
        automation: Option<Box<automation::Automation>>,
    },
    /// Input `slot` is made a `role`; `done` hears whether it is.
    Role {
        slot: u64,
        role: Role,
        done: SyncSender<Result<(), Error>>,
    },
    Crossfade {
        into: GainId,
        start: Start,
        out: Vec<GainId>,
        over: Duration,
    },
    Mic(Option<Input>),
    /// The effects input on strip `.0`, its cues through `.1`, in place of
    /// any before.
    Effects(u64, Arc<effects::CueQueue>),
    /// The effects input of `queue` gone: what the mixer thread ran of it
    /// goes back in `done` (`None`: another one runs now).
    EffectsGone {
        queue: Arc<effects::CueQueue>,
        done: SyncSender<Option<Box<voices::EffectsRun>>>,
    },
    /// The effects input's sounds from now on.
    EffectsBank(Arc<effects::SoundBank>),
    Monitor(Option<Box<dyn Monitor>>),
    /// An output started or paused: something to mix, or to settle.
    Wake,
    Quit,
}

/// What the mixer thread publishes, and what the handles keep for a look
/// at the desk. The clock, written every turn, is read without a lock;
/// what the device runs at behind a lock taken when it opens or closes,
/// the microphone's port behind one its helper takes.
#[derive(Default)]
pub(crate) struct MixShared {
    timebase: SeqLock<Timebase>,
    /// The device period, how much is mixed at a time
    /// ([`MixerOptions::period`]), and the device's ring
    /// ([`MixerOptions::buffer`]).
    pub(crate) period: Duration,
    pub(crate) buffer: Duration,
    /// Jobs the mixer thread had for its helper past the room it keeps for
    /// them (see `aside.rs`).
    pub(crate) overflows: AtomicU64,
    /// What the device runs at (`None`: closed).
    device: Mutex<Option<StreamInfo>>,
    /// The desk's format (`None`: the device takes no stereo stream, or
    /// is not there).
    format: Mutex<Option<StreamFormat>>,
    mic: Mutex<MicView>,
    /// Since the start, over every opening of the device.
    clipped: AtomicU64,
    passthrough: AtomicBool,
    /// The limiter turned the gain down since the last look.
    limited: AtomicBool,
    /// What goes to the device.
    peaks: Peaks,
    /// The device's opening (its epoch), as [`Timebase::epoch`], for the
    /// handles to read without the clock's lock.
    pub(crate) epoch: AtomicU64,
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
    pub(crate) fn timebase(&self) -> Timebase {
        self.timebase.load()
    }

    /// The mixer thread: the clock is `t` now.
    pub(crate) fn set_timebase(&self, t: Timebase) {
        self.timebase.store(&t);
    }

    pub(crate) fn device(&self) -> MutexGuard<'_, Option<StreamInfo>> {
        lock(&self.device)
    }

    pub(crate) fn format(&self) -> MutexGuard<'_, Option<StreamFormat>> {
        lock(&self.format)
    }

    pub(crate) fn mic(&self) -> MutexGuard<'_, MicView> {
        lock(&self.mic)
    }

    pub(crate) fn strips(&self) -> MutexGuard<'_, BTreeMap<u64, Entry>> {
        lock(&self.strips)
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
    /// Starts the mixer on `target`, the device opened at once at the
    /// desk's format, for as long as the mixer runs.
    pub fn open(mut target: Box<dyn OutputTarget>, options: MixerOptions) -> Mixer {
        let (tx, rx) = mpsc::channel();
        let shared = Arc::new(MixShared {
            period: options.period,
            buffer: options.buffer,
            ..MixShared::default()
        });
        let name = target.name();
        let caps = options.capabilities.clone();
        let caps = caps.or_else(|| target.capabilities().ok());
        let format = caps.and_then(|c| desk_format(&c));
        *shared.format() = format;
        let run = run::Run::new(target, format, options, shared.clone());
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

    /// The desk's format: what the device runs at, and what every output on
    /// an input plays at (`None`: the device takes no stereo stream, or is
    /// not there: the next output opened asks it again).
    pub fn format(&self) -> Option<StreamFormat> {
        *self.shared.format()
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

    /// The clock, as of the last frames mixed: what a [`Ramp`] is timed
    /// on (a cheaper look than [`Mixer::desk`], which has it too).
    pub fn timebase(&self) -> Timebase {
        self.shared.timebase()
    }

    /// The microphone listened to (`None`: none). The mixer taps the device
    /// for it while a Lead output is open (see the module's docs); how it
    /// stands is [`Mixer::mic`]. Nothing of it is ever played.
    pub fn set_mic(&self, mic: Option<Input>) {
        let _ = self.tx.send(Msg::Mic(mic));
    }

    /// Times the mixer thread had more to hand its helper thread than the
    /// room it keeps for that, each one an allocation on the mixer thread:
    /// 0 unless something is wrong.
    pub fn helper_overflows(&self) -> u64 {
        self.shared.overflows.load(Ordering::Relaxed)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FileOutput, OpenRequest, Output, SampleFormat, SimClock};

    /// A device in real time at 48 kHz, stereo.
    struct Sim;

    impl OutputTarget for Sim {
        fn name(&self) -> String {
            "sim".into()
        }

        fn capabilities(&mut self) -> Result<Capabilities, Error> {
            let format = StreamFormat::new(48_000, 2, SampleFormat::S32);
            Ok([format].into_iter().collect())
        }

        fn open(&mut self, request: &OpenRequest) -> Result<Box<dyn Output>, Error> {
            let caps = self.capabilities()?;
            let out = FileOutput::open(std::io::sink(), &caps, request, SimClock::Realtime)?;
            Ok(Box::new(out))
        }
    }

    /// An input plays one output at a time, and a request it refuses
    /// (another rate, other channels, lanes it has not, no format at all)
    /// leaves the one it plays playing.
    #[test]
    fn a_request_refused_leaves_the_output_playing() {
        let mixer = Mixer::open(Box::new(Sim), MixerOptions::default());
        let strip = mixer.strip("Player");
        let (mut target, _fader) = mixer.input(&strip, Role::Lead, 1.0);
        let request = OpenRequest::new(48_000, 2).formats(&[SampleFormat::S32]);
        let mut out = target.open(&request).unwrap();
        let tenth = vec![1_000; 2 * 4_800];
        out.write(&tenth, None).unwrap();
        let refused = [
            OpenRequest::new(44_100, 2).formats(&[SampleFormat::S32]),
            OpenRequest::new(48_000, 1).formats(&[SampleFormat::S32]),
            request.clone().lanes(2),
            request.clone().formats(&[]),
        ];
        for bad in &refused {
            assert!(target.open(bad).is_err(), "{bad:?}");
        }
        out.write(&tenth, None).unwrap();
        assert!(out.drain(Some(Duration::from_secs(10))).unwrap());
        assert_eq!(out.position().unwrap(), 9_600);
        // The next one opened takes its place.
        let mut next = target.open(&request).unwrap();
        assert!(matches!(out.write(&tenth, None), Err(Error::Closed)));
        next.write(&tenth, None).unwrap();
        assert!(next.drain(Some(Duration::from_secs(10))).unwrap());
    }
}
