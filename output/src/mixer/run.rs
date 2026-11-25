//! The mixer's thread: it holds the device, carries out what the handles
//! ask, and mixes a period at a time as the device takes frames.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::Duration;

use super::chunk::Slot;
use super::desk::pack_format;
use super::input::{InputDriver, InputOutput, Link, Stream};
use super::mic::Port;
use super::mix::Mix;
use super::parts::LaneRun;
use super::ramp::{GainId, Waiting, When};
use super::strip::channel_peaks;
use super::strips::Strips;
use super::voices::{EffectsRun, RESERVED_CHANNELS};
use super::{
    desk_format, Curve, MixShared, MixerOptions, Monitor, Msg, Ramp, Role, Start, Timebase,
};
use crate::backend::AudioPriority;
use crate::caps::Capabilities;
use crate::error::Error;
use crate::format::StreamFormat;
use crate::lanes::lane_format;
use crate::output::{Core, OpenRequest, Output, OutputTarget, StreamInfo};
use crate::shared::Shared;

/// How long the device may take to play out what it holds before it is
/// flushed.
const DRAIN_WAIT: Duration = Duration::from_millis(500);
/// How long the thread waits for its inputs, with outputs running on a
/// device that plays only what it is given.
const INPUT_WAIT: Duration = Duration::from_millis(1);

pub(super) struct Device {
    pub(super) output: Box<dyn Output>,
    pub(super) format: StreamFormat,
    pub(super) info: StreamInfo,
    /// Paused by the mixer (nothing to play).
    paused: bool,
    pub(super) mix: Mix,
}

pub(super) struct Run {
    target: Box<dyn OutputTarget>,
    options: MixerOptions,
    /// The desk's format: what the device is opened at, and every output
    /// on an input plays at (`None`: the device has none to offer).
    format: Option<StreamFormat>,
    tx: Sender<Msg>,
    shared: Arc<MixShared>,
    pub(super) strips: Strips,
    pub(super) slots: BTreeMap<u64, Slot>,
    pub(super) device: Option<Device>,
    /// The microphone's port.
    pub(super) port: Port,
    /// The effects input.
    pub(super) effects: Option<EffectsRun>,
    monitor: Option<Box<dyn Monitor>>,
    /// The device's openings.
    pub(super) epoch: u64,
    /// Samples clipped on the openings of the device before this one.
    clipped: u64,
    /// The next frame to mix, and the frames given the device (written or
    /// in `out`), from the epoch's first.
    pub(super) frame: u64,
    given: u64,
    /// Frames for the device, and how many of their samples it has taken.
    pub(super) out: Vec<i32>,
    pub(super) sent: usize,
    /// The last frames went out as an input wrote them, or as silence.
    pub(super) untouched: bool,
    /// The faders' gains whose waiting ramps start in this chunk, and at
    /// what frame.
    pub(super) starts: Vec<(GainId, u64)>,
    /// The id of the newest output opened (0: none): a strip's change for
    /// the first audio waits for one newer.
    newest_output: u64,
}

impl Run {
    pub(super) fn new(
        target: Box<dyn OutputTarget>,
        format: Option<StreamFormat>,
        options: MixerOptions,
        tx: Sender<Msg>,
        shared: Arc<MixShared>,
    ) -> Run {
        Run {
            target,
            format,
            options,
            tx,
            shared,
            strips: Strips::default(),
            slots: BTreeMap::new(),
            device: None,
            port: Port::default(),
            effects: None,
            monitor: None,
            epoch: 0,
            clipped: 0,
            frame: 0,
            given: 0,
            out: Vec::new(),
            sent: 0,
            untouched: true,
            starts: Vec::new(),
            newest_output: 0,
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
        self.close_streams(Error::Closed);
        self.close_device();
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
                let when = self.when(start, self.newest_output);
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
                self.slots.insert(id, Slot::new(id, state, &lanes, gain));
            }
            Msg::Forget(id) => self.forget(id),
            Msg::Withdraw { gain, done } => {
                let silent = self
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
            Msg::Open {
                slot,
                request,
                done,
            } => {
                let _ = done.send(self.open(slot, &request));
            }
            Msg::Closed(stream) => self.closed(stream),
            Msg::Flush { stream, done } => {
                let _ = done.send(self.flush(stream));
            }
            Msg::Follow { slot, leader, done } => {
                let followed = self.follow(slot, leader);
                if let Some(done) = done {
                    let _ = done.send(followed);
                }
            }
            Msg::Ramp { gain, ramp } => {
                self.taken(&[gain]);
                if let Some(waiting) = self.waiting(gain, ramp) {
                    self.wait_for(gain, Some(waiting));
                }
            }
            Msg::Automate { slot, automation } => {
                // The set replaced is freed here, as messages are.
                if let Some(slot) = self.slots.get_mut(&slot) {
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
                self.taken(&out);
                self.taken(&[into]);
                if let Some(rise) = self.waiting(into, Ramp::fade(1.0, over, start)) {
                    self.wait_for(into, Some(rise));
                    let fall = Waiting {
                        when: When::With(into),
                        to: 0.0,
                        over,
                        curve: Curve::EqualPower,
                    };
                    for gain in out {
                        self.wait_for(gain, Some(fall));
                    }
                }
            }
            Msg::Mic(input) => {
                self.port.set(input);
                self.fit_mic();
            }
            Msg::Effects(strip, queue) => {
                // One made before goes with this message (its handle, if it
                // lives, still holds its bank).
                self.effects = Some(EffectsRun::new(strip, queue));
                self.fit_effects();
            }
            Msg::EffectsGone { queue, done } => {
                let ours = self
                    .effects
                    .as_ref()
                    .is_some_and(|fx| Arc::ptr_eq(&fx.queue, &queue));
                let run = if ours { self.effects.take() } else { None };
                let _ = done.send(run.map(Box::new));
            }
            Msg::EffectsBank(bank) => {
                if let Some(fx) = &mut self.effects {
                    fx.load(bank);
                }
                self.fit_effects();
            }
            Msg::Monitor(monitor) => {
                if let Some(old) = &mut self.monitor {
                    if self.device.is_some() {
                        old.closed();
                    }
                }
                self.monitor = monitor;
                if let (Some(m), Some(d)) = (&mut self.monitor, &self.device) {
                    m.opened(&d.info, d.output.clock());
                }
            }
            Msg::Wake => {}
            Msg::Quit => return false,
        }
        true
    }

    /// Room for the effects' sounds, so the first of them allocates
    /// nothing: their sum and gains, the mix, the device's frames and their
    /// strip's meter, for a period at the device's rate or their bank's
    /// (the higher), of up to [`RESERVED_CHANNELS`] channels.
    fn fit_effects(&mut self) {
        let Some(fx) = &mut self.effects else {
            return;
        };
        let device = self.device.as_ref().map_or(0, |d| d.format.rate);
        let rate = device.max(fx.rate().unwrap_or(0));
        let frames = (self.options.period.as_secs_f64() * f64::from(rate)).ceil() as usize;
        fx.reserve(frames);
        self.strips
            .reserve_meter(fx.strip, frames * RESERVED_CHANNELS);
        if let Some(d) = &mut self.device {
            d.mix.reserve(frames);
            let out = 2 * frames * usize::from(d.format.channels);
            self.out.reserve(out.saturating_sub(self.out.len()));
        }
    }

    /// Whether there is something to mix: on a device that plays in real
    /// time, always (it is never paused: silence where nothing sounds);
    /// on one that plays only what it is given, while an output runs.
    fn playing(&self) -> bool {
        self.device.as_ref().is_some_and(|d| d.info.realtime) || self.running().next().is_some()
    }

    /// Forgets input `id`: its waiting ramps never come (nor those to start
    /// with them), its output is closed.
    fn forget(&mut self, id: u64) {
        let lanes = self.slots.get(&id).map_or(0, |s| s.lanes.len());
        for k in 0..lanes {
            self.wait_for(GainId::new(id, k as u16), None);
        }
        if let Some(stream) = self.slots.remove(&id).and_then(|s| s.stream) {
            stream.link.shared.fail(Error::Closed);
            self.closed(stream.link.id);
        }
    }

    /// The microphone's tap on the device, for the newest Lead output (or
    /// taken off, with none); called after every change of the streams or
    /// the device, it does nothing when none matters to it.
    fn fit_mic(&mut self) {
        let lead = self
            .slots
            .values()
            .filter(|s| s.state.role() == Role::Lead)
            .filter_map(|s| s.stream.as_ref().map(|st| st.link.id))
            .max();
        let output = self.device.as_mut().map(|d| &mut d.output);
        self.port.fit(output, self.epoch, lead, &self.shared);
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

    /// The lane of fader `id`.
    fn lane(&mut self, id: GainId) -> Option<&mut LaneRun> {
        let slot = self.slots.get_mut(&id.slot)?;
        slot.lanes.get_mut(usize::from(id.lane))
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
        let opened = self.slots.get(&gain.slot).map_or(0, |s| s.opened);
        Some(Waiting {
            when: self.when(ramp.start, opened)?,
            to: ramp.to,
            over: ramp.over,
            curve: ramp.curve,
        })
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

    pub(super) fn streams(&self) -> impl Iterator<Item = &Stream> + '_ {
        self.slots.values().filter_map(|s| s.stream.as_ref())
    }

    /// The outputs running, but for followers held by their leaders.
    fn running(&self) -> impl Iterator<Item = &Stream> + '_ {
        self.streams().filter(|s| s.live())
    }

    /// Input `id` is made a `role`: the microphone fitted again; refused
    /// for a Lead with an output open when no other Lead has one.
    fn set_role(&mut self, id: u64, role: Role) -> Result<(), Error> {
        let slot = self.slots.get(&id).ok_or(Error::Closed)?;
        if slot.state.role() == role {
            return Ok(());
        }
        let lead = |s: &Slot| s.state.role() == Role::Lead && s.stream.is_some();
        let others = self.slots.iter().any(|(&k, s)| k != id && lead(s));
        if lead(slot) && !others {
            return Err(Error::InvalidArgument(
                "the only Lead output open cannot be a guest".into(),
            ));
        }
        let slot = self.slots.get_mut(&id).expect("checked");
        slot.state.set_role(role);
        self.fit_mic();
        Ok(())
    }

    /// What an input's outputs may open at: the desk's one format,
    /// whatever the input.
    fn capabilities(&mut self, slot: u64) -> Result<Capabilities, Error> {
        if !self.slots.contains_key(&slot) {
            return Err(Error::Closed);
        }
        let format = self.desk()?;
        Ok([format].into_iter().collect())
    }

    fn open(&mut self, slot_id: u64, request: &OpenRequest) -> Result<Box<dyn Output>, Error> {
        request.validate()?;
        let slot = self.slots.get(&slot_id).ok_or(Error::Closed)?;
        let lanes = slot.lanes.len() as u16;
        if request.lanes != 1 && request.lanes != lanes {
            return Err(Error::InvalidArgument(format!(
                "an input of {lanes} lanes opens 1 or {lanes}, not {}",
                request.lanes
            )));
        }
        // An input plays one output at a time.
        if let Some(old) = self.slots.get_mut(&slot_id).and_then(Slot::drop_stream) {
            old.link.shared.fail(Error::Closed);
            self.closed(old.link.id);
        }
        // Every output plays at the desk's format (its player resamples to
        // it); the device is open at it, or (it was not there, or failed)
        // opened now.
        let format = self.desk()?;
        if !request.takes(format) {
            return Err(Error::FormatNotAccepted {
                rate: request.rate,
                channels: request.channels,
            });
        }
        if self.device.is_none() {
            self.open_desk()?;
        }
        // What was set while it had none is in force from its first frame.
        self.settle_idle();
        let device = self.device.as_ref().expect("the device is open");
        let format = device.format;
        let chunk = format.duration_to_frames(self.options.period).max(1);
        let ring = lane_format(format, request.lanes);
        let shared = Arc::new(Shared::new(ring, request.ring_frames(&format, chunk)));
        let id = self.shared.ids.fetch_add(1, Ordering::Relaxed);
        self.newest_output = id;
        let link = Arc::new(Link {
            id,
            shared: shared.clone(),
            running: AtomicBool::new(false),
            paused: AtomicBool::new(false),
        });
        let info = StreamInfo {
            format,
            backend: device.info.backend,
            device_buffer_frames: chunk,
            ring_frames: shared.ring.capacity(),
            realtime: device.info.realtime,
        };
        let driver = InputDriver {
            link: link.clone(),
            tx: self.tx.clone(),
        };
        let shared_capacity = shared.ring.capacity();
        let core = Core::new(shared, Box::new(driver), info, request);
        let slot = self.slots.get_mut(&slot_id).expect("checked");
        slot.opened = id;
        let s = &slot.state;
        s.format.store(pack_format(Some(format)), Ordering::Relaxed);
        s.capacity.store(shared_capacity, Ordering::Relaxed);
        slot.stream = Some(Stream::new(link, request.lanes));
        self.fit_mic();
        Ok(Box::new(InputOutput { core }))
    }

    /// Opens the device at the desk's format.
    fn open_desk(&mut self) -> Result<(), Error> {
        let format = self.desk()?;
        let request = OpenRequest::new(format.rate, format.channels)
            .formats(&[format.sample])
            .period(self.options.period)
            .buffer(self.options.buffer)
            .prefill(self.options.buffer / 2);
        let output = self.target.open(&request)?;
        let info = output.info().clone();
        if let Some(m) = &mut self.monitor {
            m.opened(&info, output.clock());
        }
        // A new epoch: its clock starts at 0, and times of the last one
        // never come.
        self.epoch += 1;
        self.shared.epoch.store(self.epoch, Ordering::Release);
        self.frame = 0;
        self.given = 0;
        self.out.clear();
        self.sent = 0;
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
        // Ramps under way were on the last epoch's frames.
        for slot in self.slots.values_mut() {
            for lane in &mut slot.lanes {
                lane.gain.new_epoch();
            }
            if let Some(switch) = &mut slot.switch {
                switch.new_epoch();
            }
        }
        // Nothing is mixed on it yet: a fader's ramp due now (set while the
        // device was closed or opening) is in force at once, but for one a
        // crossfade's other side waits to start with.
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
        self.strips.new_epoch();
        *self.shared.device() = Some(info.clone());
        self.device = Some(Device {
            format: info.format,
            mix: Mix::new(info.format),
            info,
            output,
            paused: false,
        });
        self.fit_effects();
        Ok(())
    }

    /// Gives out what the limiter holds back, into `self.out` (all of
    /// it written).
    fn give_held(&mut self) {
        let Some(device) = &mut self.device else {
            return;
        };
        let ch = usize::from(device.format.channels);
        self.out.clear();
        self.sent = 0;
        device.mix.end(true, &mut self.out);
        self.gave(ch);
    }

    /// What `self.out` holds goes to the device: the monitor sees it, the
    /// master's meters and limiter flag take it in.
    pub(super) fn gave(&mut self, ch: usize) {
        if let Some(m) = &mut self.monitor {
            m.mixed(&self.out, ch, self.given);
        }
        let peaks = channel_peaks(&self.out, ch, |s| f64::from(s.unsigned_abs()));
        self.shared.peaks.raise(&peaks);
        if self.device.as_mut().is_some_and(|d| d.mix.take_limited()) {
            self.shared.limited.store(true, Ordering::Relaxed);
        }
        self.given += (self.out.len() / ch) as u64;
    }

    /// Writes what is mixed and the limiter holds back, waiting as long as
    /// it takes (a drain, a flush).
    fn write_all(&mut self) -> Result<(), Error> {
        let Some(device) = &mut self.device else {
            return Ok(());
        };
        if device.paused {
            device.output.resume()?;
            device.paused = false;
        }
        loop {
            let Some(device) = &mut self.device else {
                return Ok(());
            };
            if self.sent == self.out.len() {
                if device.mix.held() == 0 {
                    return Ok(());
                }
                self.give_held();
                continue;
            }
            let ch = usize::from(device.format.channels);
            let n = device
                .output
                .write(&self.out[self.sent..], Some(DRAIN_WAIT))?;
            if n == 0 {
                return Err(Error::Timeout("the device took nothing"));
            }
            self.sent += n * ch;
        }
    }

    /// Closes the device.
    fn close_device(&mut self) {
        let Some(mut device) = self.device.take() else {
            return;
        };
        self.out.clear();
        self.sent = 0;
        let _ = device.output.close();
        self.clipped += device.mix.clipped();
        *self.shared.device() = None;
        self.fit_mic();
        if let Some(m) = &mut self.monitor {
            m.closed();
        }
    }

    /// Closes every input's output with `error`.
    fn close_streams(&mut self, error: Error) {
        for slot in self.slots.values_mut() {
            if let Some(stream) = slot.drop_stream() {
                stream.link.running.store(false, Ordering::SeqCst);
                stream.link.shared.fail(error.clone());
            }
        }
        self.fit_mic();
    }

    /// Output `id` closed (or was let go of): the outputs following it
    /// follow no one. The device stays open.
    fn closed(&mut self, id: u64) {
        for slot in self.slots.values_mut() {
            if slot.stream.as_ref().is_some_and(|s| s.link.id == id) {
                slot.drop_stream();
            }
        }
        self.unlink(id);
        self.fit_mic();
    }

    /// Output `id` was flushed (a seek): its link and those of the outputs
    /// following it end; unless another input is heard, the device drops
    /// what it holds too, as it would without a mixer. What is mixed goes
    /// to it first, so its frames keep their places.
    fn flush(&mut self, id: u64) -> Result<(), Error> {
        self.unlink(id);
        if let Some(fx) = &mut self.effects {
            fx.flushed(id);
        }
        // Its frames restart: there is no stream frame to time a change on
        // until they are mixed again.
        let flushed = self.slots.values_mut();
        let mut flushed = flushed.filter(|s| s.stream.as_ref().is_some_and(|st| st.link.id == id));
        if let Some(slot) = flushed.next() {
            slot.unmap();
        }
        let others = self
            .slots
            .values()
            .any(|s| s.heard && s.stream.as_ref().is_some_and(|st| st.link.id != id));
        if others || self.device.is_none() {
            return Ok(());
        }
        let flushed = self
            .write_all()
            .and_then(|()| self.device.as_mut().expect("checked").output.flush());
        if let Err(e) = &flushed {
            self.device_failed(e.clone());
        }
        flushed
    }

    /// The device failed: so does every input's output; the next one
    /// opened asks the device again what it takes, and opens it again
    /// (another epoch).
    fn device_failed(&mut self, error: Error) {
        self.close_streams(error);
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
        if device.paused == playing {
            let toggled = if playing {
                device.output.resume()
            } else {
                device.output.pause()
            };
            device.paused = !playing;
            if let Err(e) = toggled {
                self.device_failed(e);
                return false;
            }
        }
        if !playing {
            return false;
        }
        let realtime = device.info.realtime;
        let held = device.mix.held();
        let period = device.format.duration_to_frames(self.options.period).max(1);
        if self.sent == self.out.len() {
            let (mut full, mut lacking, mut most) = (true, false, 0);
            for s in self.running() {
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
            } else if held > 0 {
                // Every output is drained: what the limiter holds back
                // goes too.
                self.give_held();
                return self.write();
            } else {
                return false;
            };
            if !realtime && self.offline_room().is_some_and(|room| n > room) {
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
        let ch = usize::from(device.format.channels);
        match device
            .output
            .write(&self.out[self.sent..], Some(self.options.period))
        {
            Ok(n) => {
                self.sent += n * ch;
                true
            }
            Err(e) => {
                self.device_failed(e);
                false
            }
        }
    }

    /// Nothing of an idle input will be heard for now: its ramps end, and
    /// so do those of a strip with only idle inputs (what is set before an
    /// output's first audio is in force from its first frame).
    fn settle_idle(&mut self) {
        for slot in self.slots.values_mut().filter(|s| s.idle()) {
            for lane in &mut slot.lanes {
                lane.gain.settle();
            }
            if let Some(switch) = &mut slot.switch {
                switch.settle();
            }
        }
        let busy = self.slots.values().filter(|s| !s.idle());
        let busy = busy.flat_map(|s| s.lanes.iter().map(|l| l.strip));
        let effects = self.effects.as_ref().filter(|fx| fx.busy());
        self.strips
            .settle_idle(busy.chain(effects.map(|fx| fx.strip)));
    }

    /// Publishes the clock, what each output has on the device still, and
    /// each input's state; settles the ramps of inputs and strips with
    /// nothing to mix.
    fn report(&mut self) {
        let mut failed = None;
        let mut heard = 0;
        if let Some(device) = &self.device {
            match device.output.position() {
                Ok(played) => {
                    heard = played;
                    for stream in self.slots.values_mut().filter_map(|s| s.stream.as_mut()) {
                        stream.played(played);
                    }
                }
                Err(e) => failed = Some(e),
            }
        }
        if let Some(e) = failed {
            self.device_failed(e);
        }
        self.settle_idle();
        for slot in self.slots.values_mut() {
            slot.publish();
        }
        *self.shared.timebase() = Timebase {
            epoch: self.epoch,
            rate: self.device.as_ref().map_or(0, |d| d.format.rate),
            frame: self.frame,
            heard,
        };
        self.shared
            .passthrough
            .store(self.untouched, Ordering::Relaxed);
        let clipped = self.clipped + self.device.as_ref().map_or(0, |d| d.mix.clipped());
        self.shared.clipped.store(clipped, Ordering::Relaxed);
    }
}
