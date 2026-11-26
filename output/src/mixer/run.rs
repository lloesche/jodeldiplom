//! The mixer's thread: it holds the device, carries out what the handles
//! ask, and mixes a period at a time as the device takes frames.
//!
//! What it runs is in three parts besides the strips: the [`Inputs`]
//! (their slots, and the ramps that start in the chunk mixed), the
//! [`Device`] (one opening of it, see `device.rs`), and the [`Ports`]
//! (the microphone's, the effects input, the monitor).

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::Arc;
use std::time::Duration;

use super::aside::{Aside, Gone};
use super::chunk::Slot;
use super::desk::pack_format;
use super::device::Device;
use super::effects_run::{EffectsRun, RESERVED_CHANNELS};
use super::input::{Feed, Opened};
use super::mic::Port;
use super::parts::LaneRun;
use super::ramp::{GainId, Waiting, When};
use super::strips::Strips;
use super::{
    desk_format, Curve, MixShared, MixerOptions, Monitor, Msg, Ramp, Role, Start, Timebase,
};
use crate::backend::AudioPriority;
use crate::caps::Capabilities;
use crate::error::Error;
use crate::format::StreamFormat;
use crate::lanes::lane_format;
use crate::output::{OpenRequest, OutputTarget};

/// How long the thread waits for its inputs, with outputs running on a
/// device that plays only what it is given.
const INPUT_WAIT: Duration = Duration::from_millis(1);

/// The inputs, as the mixer thread runs them.
#[derive(Default)]
pub(super) struct Inputs {
    pub(super) slots: BTreeMap<u64, Slot>,
    /// The faders' gains whose waiting ramps start in this chunk, and at
    /// what frame.
    pub(super) starts: Vec<(GainId, u64)>,
    /// The id of the newest output opened (0: none): a strip's change for
    /// the first audio waits for one newer.
    newest_output: u64,
}

impl Inputs {
    /// The outputs open on the inputs.
    pub(super) fn feeds(&self) -> impl Iterator<Item = &Feed> + '_ {
        self.slots.values().filter_map(|s| s.feed.as_ref())
    }

    /// The outputs running, but for followers held by their leaders.
    fn running(&self) -> impl Iterator<Item = &Feed> + '_ {
        self.feeds().filter(|s| s.live())
    }

    /// The newest Lead output open.
    fn lead(&self) -> Option<u64> {
        self.slots
            .values()
            .filter(|s| s.state.role() == Role::Lead)
            .filter_map(|s| s.feed.as_ref().map(|st| st.link.id))
            .max()
    }

    /// The lane of fader `id`.
    fn lane(&mut self, id: GainId) -> Option<&mut LaneRun> {
        let slot = self.slots.get_mut(&id.slot)?;
        slot.lanes.get_mut(usize::from(id.lane))
    }

    /// A ramp set on each of `gains` has been taken (whether it comes or
    /// not).
    fn taken(&mut self, gains: &[GainId]) {
        for &id in gains {
            if let Some(lane) = self.lane(id) {
                lane.ramps += 1;
            }
        }
    }

    /// Fader `gain` waits for `ramp` (`None`: for nothing), in place of
    /// the ramp it waited for: the gains that were to start with that one
    /// keep theirs.
    fn wait_for(&mut self, gain: GainId, ramp: Option<Waiting>) {
        let Some(lane) = self.lane(gain) else {
            return;
        };
        let had = lane.gain.next().is_some();
        match ramp {
            Some(r) => lane.gain.set(r),
            None => lane.gain.drop_next(),
        }
        if had {
            let lanes = self.slots.values_mut().flat_map(|s| s.lanes.iter_mut());
            for other in lanes.filter(|l| l.gain.next() == Some(When::With(gain))) {
                other.gain.drop_next();
            }
        }
    }

    /// The device opened again: the ramps waiting for a frame never come
    /// (it was the last epoch's), those under way end at once (they were
    /// on the last epoch's frames), and, nothing being mixed on it yet,
    /// one due now (set while the device was closed or opening) is in
    /// force at once, but for one a crossfade's other side waits to start
    /// with.
    fn new_epoch(&mut self) {
        let ids: Vec<GainId> = self
            .slots
            .iter()
            .flat_map(|(&id, s)| {
                let lanes = s.lanes.iter().enumerate();
                lanes
                    .filter(|(_, l)| matches!(l.gain.next(), Some(When::Frame(_))))
                    .map(move |(k, _)| GainId::new(id, k as u16))
            })
            .collect();
        for id in ids {
            self.wait_for(id, None);
        }
        for slot in self.slots.values_mut() {
            for lane in &mut slot.lanes {
                lane.gain.new_epoch();
            }
            if let Some(switch) = &mut slot.switch {
                switch.new_epoch();
            }
        }
        let led: Vec<GainId> = self
            .slots
            .values()
            .flat_map(|s| s.lanes.iter())
            .filter_map(|l| match l.gain.next() {
                Some(When::With(lead)) => Some(lead),
                _ => None,
            })
            .collect();
        let lanes = self.slots.values_mut().flat_map(|s| s.lanes.iter_mut());
        for lane in lanes.filter(|l| l.gain.next() == Some(When::Now)) {
            if !led.contains(&lane.id) {
                lane.gain.settle();
            }
        }
    }
}

/// What the desk has besides its strips, inputs and device.
pub(super) struct Ports {
    /// The microphone's port.
    pub(super) mic: Port,
    /// The effects input.
    pub(super) effects: Option<EffectsRun>,
    monitor: Option<Box<dyn Monitor>>,
}

pub(super) struct Run {
    target: Box<dyn OutputTarget>,
    options: MixerOptions,
    /// The desk's format: what the device is opened at, and every output
    /// on an input plays at (`None`: the device has none to offer).
    format: Option<StreamFormat>,
    shared: Arc<MixShared>,
    strips: Strips,
    inputs: Inputs,
    device: Option<Device>,
    ports: Ports,
    /// The device's openings.
    epoch: u64,
    /// Samples clipped on the openings of the device before this one.
    clipped: u64,
    /// The helper thread, last: dropped (joined) once the rest is.
    aside: Aside,
}

impl Run {
    /// The mixer's thread, made on the caller's (with the microphone's
    /// port and its helper thread).
    pub(super) fn new(
        target: Box<dyn OutputTarget>,
        format: Option<StreamFormat>,
        options: MixerOptions,
        shared: Arc<MixShared>,
    ) -> Run {
        Run {
            target,
            format,
            options,
            ports: Ports {
                mic: Port::new(),
                effects: None,
                monitor: None,
            },
            aside: Aside::new(shared.clone()),
            shared,
            strips: Strips::default(),
            inputs: Inputs::default(),
            device: None,
            epoch: 0,
            clipped: 0,
        }
    }

    pub(super) fn run(mut self, rx: Receiver<Msg>) {
        let _priority = AudioPriority::enter(self.options.period);
        // The desk's device, open from the start to the end (where it
        // cannot be opened now, the first output opened tries again, and
        // says why it cannot).
        let _ = self.open_desk();
        let mut busy = false;
        loop {
            // Nothing to play (no device, or one that plays only what it is
            // given, with no output running): nothing to do until word
            // comes.
            let got = if !self.playing() {
                rx.recv().map_err(|_| RecvTimeoutError::Disconnected)
            } else if busy {
                rx.recv_timeout(Duration::ZERO)
            } else {
                rx.recv_timeout(INPUT_WAIT)
            };
            let mut quit = false;
            match got {
                Ok(msg) => {
                    quit = !self.handle(msg);
                    while !quit {
                        match rx.try_recv() {
                            Ok(msg) => quit = !self.handle(msg),
                            Err(_) => break,
                        }
                    }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => quit = true,
            }
            if quit {
                break;
            }
            busy = self.turn();
            self.report();
        }
        self.close_feeds(Error::Closed);
        self.close_device();
        // The microphone goes too; the helper is joined as the run ends.
        self.ports.mic.set(&mut self.aside, None);
    }

    /// The desk's format: as decided, or (none yet: the device was not
    /// there, or it failed) asked of the device now ([`desk_format`]),
    /// for every handle to see ([`super::Mixer::format`]).
    fn desk(&mut self) -> Result<StreamFormat, Error> {
        if self.format.is_none() {
            let caps = self.target.capabilities()?;
            self.set_format(desk_format(&caps));
        }
        self.format.ok_or(Error::NoDevice)
    }

    fn set_format(&mut self, format: Option<StreamFormat>) {
        self.format = format;
        *self.shared.format() = format;
    }

    /// Carries out `msg`; false: quit.
    fn handle(&mut self, msg: Msg) -> bool {
        match msg {
            Msg::Strip { id, state } => self.strips.add(id, state),
            Msg::StripSet {
                id,
                change,
                start,
                set,
            } => {
                let when = self.when(start, self.inputs.newest_output);
                self.strips.set(id, change, when, set);
            }
            Msg::StripSolo { id, on } => self.strips.solo(id, on),
            Msg::StripGone(id) => self.strips.remove(id),
            Msg::Slot {
                id,
                lanes,
                gain,
                state,
            } => {
                let slot = Slot::new(id, state, &lanes, gain);
                self.inputs.slots.insert(id, slot);
            }
            Msg::Forget(id) => self.forget(id),
            Msg::Withdraw { gain, done } => {
                let silent = self
                    .inputs
                    .lane(gain)
                    .is_some_and(|l| l.gain.next().is_some() && l.gain.unheard());
                if silent {
                    self.forget(gain.slot);
                }
                let _ = done.send(silent);
            }
            Msg::Capabilities { slot, done } => {
                let _ = done.send(self.capabilities(slot));
            }
            Msg::Open { slot, feed, done } => {
                let _ = done.send(self.open(slot, feed));
            }
            Msg::Closed(feed) => self.closed(feed),
            Msg::Flush { feed, done } => {
                let _ = done.send(self.flush(feed));
            }
            Msg::Follow { slot, leader, done } => {
                let followed = self.inputs.follow(slot, leader);
                if let Some(done) = done {
                    let _ = done.send(followed);
                }
            }
            Msg::Ramp { gain, ramp } => {
                self.inputs.taken(&[gain]);
                if let Some(waiting) = self.waiting(gain, ramp) {
                    self.inputs.wait_for(gain, Some(waiting));
                }
            }
            Msg::Automate { slot, automation } => {
                // The set replaced is freed here, as messages are.
                if let Some(slot) = self.inputs.slots.get_mut(&slot) {
                    slot.automation.set(automation);
                }
            }
            Msg::Role { slot, role, done } => {
                let _ = done.send(self.set_role(slot, role));
            }
            Msg::Crossfade {
                into,
                start,
                out,
                over,
            } => {
                self.inputs.taken(&out);
                self.inputs.taken(&[into]);
                if let Some(rise) = self.waiting(into, Ramp::fade(1.0, over, start)) {
                    self.inputs.wait_for(into, Some(rise));
                    let fall = Waiting {
                        when: When::With(into),
                        to: 0.0,
                        over,
                        curve: Curve::EqualPower,
                    };
                    for gain in out {
                        self.inputs.wait_for(gain, Some(fall));
                    }
                }
            }
            Msg::Mic(input) => {
                self.ports.mic.set(&mut self.aside, input);
                self.fit_mic();
            }
            Msg::Effects(strip, counts, live) => {
                // One made before goes with this message (its synth goes
                // aside).
                let run = EffectsRun::new(strip, counts, live);
                if let Some(mut old) = self.ports.effects.replace(run) {
                    if let Some(synth) = old.take_live() {
                        self.aside.drop_later(Gone::Synth(synth));
                    }
                }
                self.fit_effects();
            }
            Msg::EffectsGone { counts, done } => {
                let effects = &mut self.ports.effects;
                let ours = effects
                    .as_ref()
                    .is_some_and(|fx| Arc::ptr_eq(&fx.counts, &counts));
                let run = if ours { effects.take() } else { None };
                let _ = done.send(run.map(Box::new));
            }
            Msg::EffectsLive(synth) => {
                let old = match &mut self.ports.effects {
                    Some(fx) => fx.set_live(Some(synth)),
                    None => Some(synth),
                };
                if let Some(old) = old {
                    self.aside.drop_later(Gone::Synth(old));
                }
                self.fit_effects();
            }
            Msg::Monitor(monitor) => {
                let ports = &mut self.ports;
                if let Some(old) = &mut ports.monitor {
                    if self.device.is_some() {
                        old.closed();
                    }
                }
                ports.monitor = monitor;
                if let (Some(m), Some(d)) = (&mut ports.monitor, &self.device) {
                    m.opened(&d.info, d.clock());
                }
            }
            Msg::Wake => {}
            Msg::Quit => return false,
        }
        true
    }

    /// Room for the effects' sounds, so the first of them allocates
    /// nothing: their sum and gains, the synth's frames, the mix, the
    /// device's frames and their strip's meter, for a period at the
    /// device's rate or their synth's (the higher), of up to
    /// [`RESERVED_CHANNELS`] channels.
    fn fit_effects(&mut self) {
        let Some(fx) = &mut self.ports.effects else {
            return;
        };
        let device = self.device.as_ref().map_or(0, |d| d.format.rate);
        let rate = device.max(fx.rate().unwrap_or(0));
        let frames = (self.options.period.as_secs_f64() * f64::from(rate)).ceil() as usize;
        fx.reserve(frames);
        self.strips
            .reserve_meter(fx.strip, frames * RESERVED_CHANNELS);
        if let Some(d) = &mut self.device {
            d.reserve(frames);
        }
    }

    /// Whether there is something to mix: on a device that plays in real
    /// time, always (it is never paused: silence where nothing sounds);
    /// on one that plays only what it is given, while an output runs.
    fn playing(&self) -> bool {
        self.device.as_ref().is_some_and(|d| d.info.realtime)
            || self.inputs.running().next().is_some()
    }

    /// Forgets input `id`: its waiting ramps never come (nor those to start
    /// with them), its output is closed.
    fn forget(&mut self, id: u64) {
        let lanes = self.inputs.slots.get(&id).map_or(0, |s| s.lanes.len());
        for k in 0..lanes {
            self.inputs.wait_for(GainId::new(id, k as u16), None);
        }
        if let Some(feed) = self.inputs.slots.remove(&id).and_then(|s| s.feed) {
            self.retire(feed, Error::Closed);
        }
    }

    /// The microphone's tap on the device, for the newest Lead output (or
    /// taken off, with none); called after every change of the outputs or
    /// the device, it does nothing when none matters to it.
    fn fit_mic(&mut self) {
        let device = self.device.as_ref().map(|d| &d.taps);
        let lead = self.inputs.lead();
        self.ports
            .mic
            .fit(&mut self.aside, device, self.epoch, lead);
    }

    /// `start` as it is waited for, `after` the output a first audio must
    /// come after: `None` when it never comes (a time of an epoch gone).
    fn when(&self, start: Start, after: u64) -> Option<When> {
        Some(match start {
            Start::Now => When::Now,
            Start::FirstAudio => When::FirstAudio { after },
            Start::At { epoch, time } => {
                let rate = self.device.as_ref()?.format.rate;
                if epoch != self.epoch {
                    return None;
                }
                When::Frame((time.as_secs_f64() * f64::from(rate)).round() as u64)
            }
        })
    }

    /// `ramp` for fader `gain`, as it waits: `None` when it never comes.
    fn waiting(&self, gain: GainId, ramp: Ramp) -> Option<Waiting> {
        let opened = self.inputs.slots.get(&gain.slot).map_or(0, |s| s.opened);
        Some(Waiting {
            when: self.when(ramp.start, opened)?,
            to: ramp.to,
            over: ramp.over,
            curve: ramp.curve,
        })
    }

    /// Input `id` is made a `role`: the microphone fitted again; refused
    /// for a Lead with an output open when no other Lead has one.
    fn set_role(&mut self, id: u64, role: Role) -> Result<(), Error> {
        let slots = &mut self.inputs.slots;
        let slot = slots.get(&id).ok_or(Error::Closed)?;
        if slot.state.role() == role {
            return Ok(());
        }
        let lead = |s: &Slot| s.state.role() == Role::Lead && s.feed.is_some();
        let others = slots.iter().any(|(&k, s)| k != id && lead(s));
        if lead(slot) && !others {
            return Err(Error::InvalidArgument(
                "the only Lead output open cannot be a guest".into(),
            ));
        }
        let slot = slots.get_mut(&id).expect("checked");
        slot.state.set_role(role);
        self.fit_mic();
        Ok(())
    }

    /// What an input's outputs may open at: the desk's one format,
    /// whatever the input.
    fn capabilities(&mut self, slot: u64) -> Result<Capabilities, Error> {
        if !self.inputs.slots.contains_key(&slot) {
            return Err(Error::Closed);
        }
        let format = self.desk()?;
        Ok([format].into_iter().collect())
    }

    /// Takes in `feed`, an output its opener made for input `slot_id`
    /// ([`super::input::open`]), in place of the one the input plays (one
    /// at a time), once it can play: its input is there, it is in the
    /// desk's format, and the device is open at it (or, not there or
    /// failed, opened now). Otherwise the output on the input plays on.
    fn open(&mut self, slot_id: u64, feed: Feed) -> Result<Opened, Error> {
        let admitted = self.admit(slot_id, &feed);
        if !matches!(admitted, Ok(Opened::Yes { .. })) {
            self.retire(feed, Error::Closed);
            return admitted;
        }
        let old = self
            .inputs
            .slots
            .get_mut(&slot_id)
            .and_then(Slot::drop_feed);
        if let Some(old) = old {
            self.retire(old, Error::Closed);
        }
        // What was set while it had none is in force from its first frame.
        self.settle_idle();
        let id = feed.link.id;
        self.inputs.newest_output = id;
        let format = self.device.as_ref().expect("admitted").format;
        let slot = self.inputs.slots.get_mut(&slot_id).expect("admitted");
        slot.opened = id;
        let s = &slot.state;
        s.format.store(pack_format(Some(format)), Ordering::Relaxed);
        s.capacity
            .store(feed.link.shared.ring.capacity(), Ordering::Relaxed);
        slot.feed = Some(feed);
        self.fit_mic();
        admitted
    }

    /// Whether `feed` can play on input `slot_id` (see [`Self::open`]).
    fn admit(&mut self, slot_id: u64, feed: &Feed) -> Result<Opened, Error> {
        if !self.inputs.slots.contains_key(&slot_id) {
            return Err(Error::Closed);
        }
        let format = self.desk()?;
        if feed.link.shared.format != lane_format(format, feed.lanes) {
            return Ok(Opened::Desk(format));
        }
        if self.device.is_none() {
            self.open_desk()?;
        }
        let device = self.device.as_ref().expect("the device is open");
        Ok(Opened::Yes {
            backend: device.info.backend,
            realtime: device.info.realtime,
        })
    }

    /// Opens the device at the desk's format: a new epoch, its clock from
    /// 0, and times of the last one never come.
    fn open_desk(&mut self) -> Result<(), Error> {
        let format = self.desk()?;
        let request = OpenRequest::new(format.rate, format.channels)
            .formats(&[format.sample])
            .period(self.options.period)
            .buffer(self.options.buffer)
            .prefill(self.options.buffer / 2);
        let device = Device::new(self.target.open(&request)?);
        if let Some(m) = &mut self.ports.monitor {
            m.opened(&device.info, device.clock());
        }
        self.epoch += 1;
        self.shared.epoch.store(self.epoch, Ordering::Release);
        self.inputs.new_epoch();
        self.strips.new_epoch();
        *self.shared.device() = Some(device.info.clone());
        self.device = Some(device);
        // The desk's frames start over: what its synth waits to play on
        // them is called off.
        if let Some(fx) = &mut self.ports.effects {
            fx.opened();
        }
        self.fit_effects();
        Ok(())
    }

    /// Closes the device.
    fn close_device(&mut self) {
        let Some(device) = self.device.take() else {
            return;
        };
        self.clipped += device.close();
        *self.shared.device() = None;
        self.fit_mic();
        if let Some(m) = &mut self.ports.monitor {
            m.closed();
        }
    }

    /// Closes every input's output with `error`.
    fn close_feeds(&mut self, error: Error) {
        let mut next = self.inputs.slots.keys().next().copied();
        while let Some(id) = next {
            let slots = &mut self.inputs.slots;
            next = slots.range(id + 1..).next().map(|(&k, _)| k);
            let slot = slots.get_mut(&id).expect("listed");
            if let Some(feed) = slot.drop_feed() {
                self.retire(feed, error.clone());
            }
        }
        self.fit_mic();
    }

    /// Output `id` closed (or was let go of). The device stays open.
    fn closed(&mut self, id: u64) {
        let playing = |s: &&mut Slot| s.feed.as_ref().is_some_and(|st| st.link.id == id);
        let slot = self.inputs.slots.values_mut().find(playing);
        if let Some(feed) = slot.and_then(Slot::drop_feed) {
            self.retire(feed, Error::Closed);
        }
    }

    /// Output `feed` goes: it fails with `error` (its handle hears it),
    /// the outputs following it follow no one, the microphone is fitted
    /// again, and it is let go of on the helper thread.
    fn retire(&mut self, feed: Feed, error: Error) {
        let id = feed.link.id;
        feed.link.running.store(false, Ordering::SeqCst);
        feed.link.shared.fail(error);
        self.aside.drop_later(Gone::Feed(feed));
        self.inputs.unlink(id);
        self.fit_mic();
    }

    /// Output `id` was flushed (a seek): its link and those of the outputs
    /// following it end; unless another input is heard, the device drops
    /// what it holds too, as it would without a mixer. What is mixed goes
    /// to it first, so its frames keep their places.
    fn flush(&mut self, id: u64) -> Result<(), Error> {
        self.inputs.unlink(id);
        if let Some(fx) = &mut self.ports.effects {
            fx.flushed(id);
        }
        // Its frames restart: there is no stream frame to time a change on
        // until they are mixed again.
        let slots = &mut self.inputs.slots;
        let flushed = |s: &&mut Slot| s.feed.as_ref().is_some_and(|st| st.link.id == id);
        if let Some(slot) = slots.values_mut().find(flushed) {
            slot.unmap();
        }
        let others = slots
            .values()
            .any(|s| s.heard && s.feed.as_ref().is_some_and(|st| st.link.id != id));
        let Some(device) = self.device.as_mut().filter(|_| !others) else {
            return Ok(());
        };
        let flushed = device.flush(&mut self.ports.monitor, &self.shared);
        if let Err(e) = &flushed {
            self.device_failed(e.clone());
        }
        flushed
    }

    /// The device failed: so does every input's output; the next one
    /// opened asks the device again what it takes, and opens it again
    /// (another epoch).
    fn device_failed(&mut self, error: Error) {
        self.close_feeds(error);
        self.close_device();
        self.set_format(None);
    }

    /// Writes what is mixed, or mixes the next period: when the device
    /// plays in real time, as it takes frames (an output short of them is
    /// silence for the rest); otherwise once every running output holds a
    /// period, or is draining, and never a period past the origin of a
    /// follower not running yet. A device that plays only what it is given
    /// pauses with nothing running (a follower held by its leader does not
    /// count); one in real time plays on, silence where nothing sounds.
    /// Returns whether there was work.
    fn turn(&mut self) -> bool {
        let playing = self.playing();
        let Some(device) = &mut self.device else {
            return false;
        };
        if let Err(e) = device.set_paused(!playing) {
            self.device_failed(e);
            return false;
        }
        if !playing {
            return false;
        }
        if device.written() {
            let realtime = device.info.realtime;
            let period = device.format.duration_to_frames(self.options.period).max(1);
            let (mut full, mut lacking, mut most) = (true, false, 0);
            for s in self.inputs.running() {
                let queued = s.link.shared.ring.len();
                most = most.max(queued);
                if queued < period {
                    full = false;
                    lacking |= !s.link.shared.draining();
                }
            }
            let n = if full || realtime {
                period
            } else if lacking {
                return false;
            } else if most > 0 {
                most.min(period)
            } else if device.held() > 0 {
                // Every output is drained: what the limiter holds back
                // goes too.
                device.give_held(&mut self.ports.monitor, &self.shared);
                return self.write();
            } else {
                return false;
            };
            if !realtime && self.inputs.offline_room().is_some_and(|room| n > room) {
                return false;
            }
            if !self.mix_chunk(n as usize) {
                return false;
            }
        }
        self.write()
    }

    /// Writes some of what is mixed (waiting up to a period for room).
    fn write(&mut self) -> bool {
        let Some(device) = &mut self.device else {
            return false;
        };
        match device.write(self.options.period) {
            Ok(()) => true,
            Err(e) => {
                self.device_failed(e);
                false
            }
        }
    }

    /// Mixes the next `n` frames for the device; returns whether it did
    /// (see [`Inputs::take`] for when it does not).
    fn mix_chunk(&mut self, n: usize) -> bool {
        let Run {
            inputs,
            device,
            ports,
            strips,
            shared,
            epoch,
            ..
        } = self;
        let device = device.as_mut().expect("the device is open");
        let (ch, rate) = (device.channels(), device.format.rate);
        let chunk = (*epoch, device.frame, n);
        if !inputs.take(chunk, ch, device.info.realtime, &ports.mic) {
            return false;
        }
        inputs.fill(chunk, rate, strips);
        // The effects: the commands come go to the synth, which is summed,
        // at their strip's path.
        if let Some(fx) = &mut ports.effects {
            fx.chunk(chunk, ch, rate, &inputs.slots);
            fx.take_gains(strips);
        }
        let effects = ports.effects.as_ref().and_then(EffectsRun::lane);
        // Each strip's meter: its subtree's heard lanes at those gains.
        let fx_meter = effects.map(|l| (l.strip, l.samples, l.gains, true));
        let lanes = inputs.slots.values().flat_map(Slot::metered_lanes);
        strips.meter(ch, lanes.chain(fx_meter));
        // Every lane heard: the inputs', the effects'.
        let slots = &inputs.slots;
        device.mix(n, || {
            slots.values().flat_map(Slot::heard_lanes).chain(effects)
        });
        device.gave(&mut ports.monitor, shared);
        true
    }

    /// Nothing of an idle input will be heard for now: its ramps end, and
    /// so do those of a strip with only idle inputs (what is set before an
    /// output's first audio is in force from its first frame).
    fn settle_idle(&mut self) {
        let slots = &mut self.inputs.slots;
        for slot in slots.values_mut().filter(|s| s.idle()) {
            for lane in &mut slot.lanes {
                lane.gain.settle();
            }
            if let Some(switch) = &mut slot.switch {
                switch.settle();
            }
        }
        let busy = slots.values().filter(|s| !s.idle());
        let busy = busy.flat_map(|s| s.lanes.iter().map(|l| l.strip));
        let effects = self.ports.effects.as_ref().filter(|fx| fx.busy());
        self.strips
            .settle_idle(busy.chain(effects.map(|fx| fx.strip)));
    }

    /// Publishes the clock, what each output has on the device still, and
    /// each input's state; settles the ramps of inputs and strips with
    /// nothing to mix.
    fn report(&mut self) {
        let mut heard = 0;
        if let Some(device) = &self.device {
            match device.position() {
                Ok(played) => {
                    heard = played;
                    for slot in self.inputs.slots.values_mut() {
                        if let Some(feed) = &mut slot.feed {
                            feed.played(played);
                        }
                    }
                }
                Err(e) => self.device_failed(e),
            }
        }
        self.settle_idle();
        for slot in self.inputs.slots.values() {
            slot.publish();
        }
        let device = self.device.as_ref();
        self.shared.set_timebase(Timebase {
            epoch: self.epoch,
            rate: device.map_or(0, |d| d.format.rate),
            frame: device.map_or(0, |d| d.frame),
            heard,
        });
        // What its helper had no room for yet goes now.
        self.aside.flush();
        if let Some(d) = device {
            self.shared
                .passthrough
                .store(d.untouched, Ordering::Relaxed);
        }
        let clipped = self.clipped + device.map_or(0, Device::clipped);
        self.shared.clipped.store(clipped, Ordering::Relaxed);
    }
}
