//! The mixer's thread: it holds the device, carries out what the handles
//! ask, and mixes a period at a time as the device takes frames.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::Duration;

use super::desk::pack_format;
use super::input::{InputDriver, InputOutput, Link, Stream};
use super::mic::Port;
use super::mix::{Gains, Made, Mix};
use super::ramp::{resolve_starts, start_of, Gain, Waiting, When};
use super::strip::{channel_peaks, Strips};
use super::{Curve, MixShared, MixerOptions, Monitor, Msg, Ramp, Role, SlotState, Start, Timebase};
use crate::backend::AudioPriority;
use crate::caps::Capabilities;
use crate::error::Error;
use crate::format::StreamFormat;
use crate::output::{Core, OpenRequest, Output, OutputTarget, StreamInfo};
use crate::shared::Shared;

/// How long the device may take to play out what it holds before it is
/// opened again at another format, or flushed.
const DRAIN_WAIT: Duration = Duration::from_millis(500);
/// How long the thread waits for its inputs, with outputs running on a
/// device that plays only what it is given.
const INPUT_WAIT: Duration = Duration::from_millis(1);

struct Slot {
    role: Role,
    /// The strip it is on.
    strip: u64,
    state: Arc<SlotState>,
    /// Its controllers' gain.
    gain: Gain,
    /// Ramps taken.
    ramps: u64,
    /// Outputs opened on it.
    opened: u64,
    /// What it was offered, once asked.
    caps: Option<Capabilities>,
    /// This chunk's gains through its strip's (what it is heard at), or
    /// the one for all of it.
    gains: Vec<f64>,
    flat: Option<f64>,
    /// It was heard in the last chunk: it supplied frames at a gain other
    /// than 0.
    heard: bool,
    stream: Option<Stream>,
}

impl Slot {
    /// The gains it is heard at this chunk.
    fn heard_at(&self) -> Gains<'_> {
        match self.flat {
            Some(g) => Gains::Flat(g),
            None => Gains::Each(&self.gains),
        }
    }

    /// Nothing of it will be heard for now: no output, or paused (one held
    /// back for a flush or a prefill comes back).
    fn idle(&self) -> bool {
        self.stream
            .as_ref()
            .is_none_or(|s| s.link.paused.load(Ordering::SeqCst))
    }

    /// Its output is gone (if it had one), and so is its format on the
    /// desk.
    fn drop_stream(&mut self) -> Option<Stream> {
        self.state.format.store(0, Ordering::Relaxed);
        self.state.capacity.store(0, Ordering::Relaxed);
        self.state.queued.store(0, Ordering::Relaxed);
        self.stream.take()
    }

    /// Publishes its gain, whether it settled and was heard, and its ring's
    /// fill.
    fn publish(&self) {
        let s = &self.state;
        s.gain.store(self.gain.last().to_bits(), Ordering::Release);
        if !self.gain.moving() {
            s.settled.store(self.ramps, Ordering::Release);
        }
        s.heard.store(self.heard && !self.idle(), Ordering::Relaxed);
        let queued = self
            .stream
            .as_ref()
            .map_or(0, |st| st.link.shared.ring.len());
        s.queued.store(queued, Ordering::Relaxed);
    }
}

struct Device {
    output: Box<dyn Output>,
    format: StreamFormat,
    info: StreamInfo,
    /// Paused by the mixer (nothing to play).
    paused: bool,
    mix: Mix,
}

pub(super) struct Run {
    target: Box<dyn OutputTarget>,
    options: MixerOptions,
    /// The device's capabilities, once asked.
    caps: Option<Capabilities>,
    tx: Sender<Msg>,
    shared: Arc<MixShared>,
    strips: Strips,
    slots: BTreeMap<u64, Slot>,
    device: Option<Device>,
    /// The microphone's port.
    port: Port,
    monitor: Option<Box<dyn Monitor>>,
    /// The device's openings.
    epoch: u64,
    /// Samples clipped on the openings of the device before this one.
    clipped: u64,
    /// The next frame to mix, and the frames given the device (written or
    /// in `out`), from the epoch's first.
    frame: u64,
    given: u64,
    /// Frames for the device, and how many of their samples it has taken.
    out: Vec<i32>,
    sent: usize,
    /// The last frames went out as an input wrote them, or as silence.
    untouched: bool,
    /// The inputs whose waiting ramps start in this chunk, and at what
    /// frame.
    starts: Vec<(u64, u64)>,
}

/// Whether the device's `format` is one `request` asks for.
fn accepts(request: &OpenRequest, format: StreamFormat) -> bool {
    format.rate == request.rate
        && format.channels == request.channels
        && request.formats.contains(&format.sample)
}

impl Run {
    pub(super) fn new(
        target: Box<dyn OutputTarget>,
        options: MixerOptions,
        tx: Sender<Msg>,
        shared: Arc<MixShared>,
    ) -> Run {
        Run {
            target,
            caps: options.capabilities.clone(),
            options,
            tx,
            shared,
            strips: Strips::default(),
            slots: BTreeMap::new(),
            device: None,
            port: Port::default(),
            monitor: None,
            epoch: 0,
            clipped: 0,
            frame: 0,
            given: 0,
            out: Vec::new(),
            sent: 0,
            untouched: true,
            starts: Vec::new(),
        }
    }

    pub(super) fn run(mut self, rx: Receiver<Msg>) {
        let _priority = AudioPriority::enter();
        let mut busy = false;
        loop {
            // Nothing running: nothing to do until word comes.
            let got = if self.running().next().is_none() {
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
        self.close_device(false);
    }

    /// Carries out `msg`; false: quit.
    fn handle(&mut self, msg: Msg) -> bool {
        match msg {
            Msg::Strip { id, state } => self.strips.add(id, state),
            Msg::StripChanged => self.strips.changed(),
            Msg::StripGone(id) => self.strips.remove(id),
            Msg::Slot {
                id,
                strip,
                gain,
                state,
            } => {
                self.slots.insert(
                    id,
                    Slot {
                        role: state.role,
                        strip,
                        state,
                        gain: Gain::new(gain),
                        ramps: 0,
                        opened: 0,
                        caps: None,
                        gains: Vec::new(),
                        flat: Some(gain),
                        heard: false,
                        stream: None,
                    },
                );
            }
            Msg::Forget(id) => {
                self.wait_for(id, None);
                if let Some(stream) = self.slots.remove(&id).and_then(|s| s.stream) {
                    stream.link.shared.fail(Error::Closed);
                    self.closed(stream.link.id);
                }
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
            Msg::Ramp { slot, ramp } => {
                self.taken(&[slot]);
                if let Some(waiting) = self.waiting(slot, ramp) {
                    self.wait_for(slot, Some(waiting));
                }
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
                    for slot in out {
                        self.wait_for(slot, Some(fall));
                    }
                }
            }
            Msg::Mic(input) => {
                self.port.set(input);
                self.fit_mic();
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

    /// The microphone's tap on the device, for the newest Lead output (or
    /// taken off, with none); called after every change of the streams or
    /// the device, it does nothing when none matters to it.
    fn fit_mic(&mut self) {
        let lead = self
            .slots
            .values()
            .filter(|s| s.role == Role::Lead)
            .filter_map(|s| s.stream.as_ref().map(|st| st.link.id))
            .max();
        let output = self.device.as_mut().map(|d| &mut d.output);
        self.port.fit(output, self.epoch, lead, &self.shared);
    }

    /// A ramp set on each of `slots` has been taken (whether it comes or
    /// not).
    fn taken(&mut self, slots: &[u64]) {
        for id in slots {
            if let Some(s) = self.slots.get_mut(id) {
                s.ramps += 1;
            }
        }
    }

    /// `ramp` for input `slot`, as it waits: `None` when it never comes (a
    /// time of an epoch gone).
    fn waiting(&self, slot: u64, ramp: Ramp) -> Option<Waiting> {
        let when = match ramp.start {
            Start::Now => When::Now,
            Start::FirstAudio => When::FirstAudio {
                after: self.slots.get(&slot).map_or(0, |s| s.opened),
            },
            Start::At { epoch, time } => {
                let rate = self.device.as_ref()?.format.rate;
                if epoch != self.epoch {
                    return None;
                }
                When::Frame((time.as_secs_f64() * f64::from(rate)).round() as u64)
            }
        };
        Some(Waiting {
            when,
            to: ramp.to,
            over: ramp.over,
            curve: ramp.curve,
        })
    }

    /// Input `slot` waits for `ramp` (`None`: for nothing), in place of the
    /// ramp it waited for: the inputs that were to start with that one
    /// keep their gains.
    fn wait_for(&mut self, slot: u64, ramp: Option<Waiting>) {
        let Some(s) = self.slots.get_mut(&slot) else {
            return;
        };
        let had = s.gain.next().is_some();
        match ramp {
            Some(r) => s.gain.set(r),
            None => s.gain.drop_next(),
        }
        if had {
            for other in self.slots.values_mut() {
                if other.gain.next() == Some(When::With(slot)) {
                    other.gain.drop_next();
                }
            }
        }
    }

    fn streams(&self) -> impl Iterator<Item = &Stream> + '_ {
        self.slots.values().filter_map(|s| s.stream.as_ref())
    }

    fn running(&self) -> impl Iterator<Item = &Stream> + '_ {
        self.streams()
            .filter(|s| s.link.running.load(Ordering::SeqCst))
    }

    fn capabilities(&mut self, slot: u64) -> Result<Capabilities, Error> {
        let Some(s) = self.slots.get(&slot) else {
            return Err(Error::Closed);
        };
        if let Some(caps) = &s.caps {
            return Ok(caps.clone());
        }
        let caps = match (&self.device, s.role) {
            (Some(d), Role::Guest) => [d.format].into_iter().collect(),
            _ => match &self.caps {
                Some(caps) => caps.clone(),
                None => {
                    let caps = self.target.capabilities()?;
                    self.caps = Some(caps.clone());
                    caps
                }
            },
        };
        if let Some(s) = self.slots.get_mut(&slot) {
            s.caps = Some(caps.clone());
        }
        Ok(caps)
    }

    fn open(&mut self, slot_id: u64, request: &OpenRequest) -> Result<Box<dyn Output>, Error> {
        request.validate()?;
        let role = self.slots.get(&slot_id).ok_or(Error::Closed)?.role;
        // An input plays one output at a time.
        if let Some(old) = self.slots.get_mut(&slot_id).and_then(Slot::drop_stream) {
            old.link.shared.fail(Error::Closed);
            self.closed(old.link.id);
        }
        let fits = self
            .device
            .as_ref()
            .is_some_and(|d| accepts(request, d.format));
        if !fits {
            if self.device.is_some() {
                // Only the Lead opens the device again; a guest takes it
                // as it is.
                if role == Role::Guest {
                    return Err(Error::FormatNotAccepted {
                        rate: request.rate,
                        channels: request.channels,
                    });
                }
                self.close_streams(Error::Closed);
                self.close_device(true);
            }
            self.open_device(request)?;
        }
        let device = self.device.as_ref().expect("the device is open");
        let format = device.format;
        let chunk = format.duration_to_frames(self.options.period).max(1);
        let shared = Arc::new(Shared::new(format, request.ring_frames(&format, chunk)));
        let id = self.shared.ids.fetch_add(1, Ordering::Relaxed);
        let link = Arc::new(Link {
            id,
            shared: shared.clone(),
            running: AtomicBool::new(false),
            paused: AtomicBool::new(false),
        });
        let info = StreamInfo {
            format,
            backend: device.info.backend,
            exclusive: device.info.exclusive,
            device_buffer_frames: chunk,
            device_latency: device.info.device_latency
                + format.frames_to_duration(device.info.ring_frames),
            ring_frames: shared.ring.capacity(),
            realtime: device.info.realtime,
            float_io: device.info.float_io,
        };
        let driver = InputDriver {
            link: link.clone(),
            tx: self.tx.clone(),
        };
        let shared_capacity = shared.ring.capacity();
        let core = Core::new(shared, Box::new(driver), info, request);
        let slot = self.slots.get_mut(&slot_id).expect("checked");
        slot.opened += 1;
        let s = &slot.state;
        s.format.store(pack_format(Some(format)), Ordering::Relaxed);
        s.capacity.store(shared_capacity, Ordering::Relaxed);
        slot.stream = Some(Stream::new(link, slot.opened));
        self.fit_mic();
        Ok(Box::new(InputOutput { core }))
    }

    fn open_device(&mut self, request: &OpenRequest) -> Result<(), Error> {
        let request = request
            .clone()
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
        self.frame = 0;
        self.given = 0;
        self.out.clear();
        self.sent = 0;
        let ids: Vec<u64> = self
            .slots
            .iter()
            .filter(|(_, s)| matches!(s.gain.next(), Some(When::Frame(_))))
            .map(|(&id, _)| id)
            .collect();
        for id in ids {
            self.wait_for(id, None);
        }
        // Ramps under way were on the last epoch's frames.
        for slot in self.slots.values_mut() {
            slot.gain.new_epoch();
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
    fn gave(&mut self, ch: usize) {
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

    /// Closes the device; with `drain`, after it has played what was
    /// mixed.
    fn close_device(&mut self, drain: bool) {
        if drain && self.write_all().is_ok() {
            if let Some(d) = &mut self.device {
                let _ = d.output.drain(Some(DRAIN_WAIT));
            }
        }
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

    /// Output `id` closed (or was let go of): the device closes with the
    /// last one.
    fn closed(&mut self, id: u64) {
        for slot in self.slots.values_mut() {
            if slot.stream.as_ref().is_some_and(|s| s.link.id == id) {
                slot.drop_stream();
            }
        }
        self.fit_mic();
        if self.streams().next().is_none() {
            self.close_device(false);
        }
    }

    /// Output `id` was flushed (a seek): unless another input is heard,
    /// the device drops what it holds too, as it would without a mixer.
    /// What is mixed goes to it first, so its frames keep their places.
    fn flush(&mut self, id: u64) -> Result<(), Error> {
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

    /// The device failed: so does every input's output.
    fn device_failed(&mut self, error: Error) {
        self.close_streams(error);
        self.close_device(false);
    }

    /// Writes what is mixed, or mixes the next period: when the device
    /// plays in real time, as it takes frames (an output short of them is
    /// silence for the rest); otherwise once every running output holds a
    /// period, or is draining. With nothing running, the device pauses.
    /// Returns whether there was work.
    fn turn(&mut self) -> bool {
        let playing = self.running().next().is_some();
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
            self.mix_chunk(n as usize);
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

    /// Mixes the next `n` frames into `self.out`.
    fn mix_chunk(&mut self, n: usize) {
        let device = self.device.as_mut().expect("the device is open");
        let format = device.format;
        let ch = usize::from(format.channels);
        let first = self.frame;
        let end = first + n as u64;
        // What each output gives.
        for stream in self.slots.values_mut().filter_map(|s| s.stream.as_mut()) {
            if let Some((lead, got)) = stream.take(n, format, first) {
                if let Some(map) = self.port.map_for(stream.link.id) {
                    map.push(first, lead, got);
                }
            }
        }
        // Where the waiting ramps start.
        let waiting = self.slots.iter().filter_map(|(&id, slot)| {
            let audio = slot
                .stream
                .as_ref()
                .filter(|s| s.got > 0)
                .map(|s| s.ordinal);
            slot.gain.next().map(|when| (id, when, audio))
        });
        resolve_starts(first, end, waiting, &mut self.starts);
        // The strips' gains; each input's through its strip's (what it is
        // heard at), and who is heard.
        self.strips.fill(first, n, format.rate);
        let (mut heard, mut alone) = (0, None);
        for (&id, slot) in self.slots.iter_mut() {
            let start = start_of(&self.starts, id);
            slot.gains.resize(n, 0.0);
            let fader = slot.gain.fill(first, start, format.rate, &mut slot.gains);
            slot.flat = self.strips.through(slot.strip, fader, &mut slot.gains);
            slot.heard = slot.stream.as_ref().is_some_and(|s| s.taken && s.got > 0)
                && slot.flat != Some(0.0);
            if slot.heard {
                heard += 1;
                alone = Some(id);
            }
        }
        // Each strip's meter: its heard inputs summed at those gains.
        let inputs = self.slots.values().filter(|s| s.heard).filter_map(|s| {
            let stream = s.stream.as_ref()?;
            Some((s.strip, &stream.samples[..], s.heard_at()))
        });
        self.strips.meter(ch, inputs);
        // One heard at 1 (or none): as written; else summed.
        let untouched = match (heard, alone) {
            (0, _) => true,
            (1, Some(id)) => self.slots[&id].flat == Some(1.0),
            _ => false,
        };
        self.out.clear();
        self.sent = 0;
        if untouched && !device.mix.at_rest() {
            // What the limiter held back goes first, so these frames can
            // go as they are.
            device.mix.end(true, &mut self.out);
        }
        let made = if untouched && device.mix.at_rest() {
            match alone.and_then(|id| self.slots[&id].stream.as_ref()) {
                Some(s) if heard == 1 => {
                    self.out.extend_from_slice(&s.samples);
                    Made::Passthrough
                }
                _ => {
                    self.out.resize(self.out.len() + n * ch, 0);
                    Made::Silence
                }
            }
        } else {
            device.mix.begin(n * ch);
            for slot in self.slots.values().filter(|s| s.heard) {
                let stream = slot.stream.as_ref().expect("heard");
                device.mix.add(&stream.samples, slot.heard_at());
            }
            device.mix.end(untouched, &mut self.out);
            Made::Mixed
        };
        self.untouched = made != Made::Mixed;
        self.gave(ch);
        self.frame = end;
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
        // Nothing of an idle input will be heard for now: its ramps end, and
        // so do those of a strip with only idle inputs.
        for slot in self.slots.values_mut() {
            if slot.idle() {
                slot.gain.settle();
            }
            slot.publish();
        }
        let busy = self.slots.values().filter(|s| !s.idle()).map(|s| s.strip);
        self.strips.settle_idle(busy);
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
