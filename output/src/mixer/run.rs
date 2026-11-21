//! The mixer's thread: it holds the device, carries out what the handles
//! ask, and mixes a period at a time as the device takes frames.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::Duration;

use super::input::{FrameMap, InputDriver, InputOutput, Link, MappedTap};
use super::mix::{Gains, Made, Mix};
use super::ramp::{Gain, Waiting, When};
use super::{Curve, MixShared, MixerOptions, Monitor, Msg, Ramp, Role, SlotState, Start, Timebase};
use crate::backend::AudioPriority;
use crate::caps::Capabilities;
use crate::error::Error;
use crate::format::StreamFormat;
use crate::output::{Core, OpenRequest, Output, OutputTarget, StreamInfo};
use crate::pack::unpack_sample;
use crate::shared::Shared;
use crate::tap::Tap;

/// How long the device may take to play out what it holds before it is
/// opened again at another format, or flushed.
const DRAIN_WAIT: Duration = Duration::from_millis(500);
/// How long the thread waits for its inputs, with outputs running on a
/// device that plays only what it is given.
const INPUT_WAIT: Duration = Duration::from_millis(1);

/// An output open on an input.
struct Stream {
    link: Arc<Link>,
    /// Which output of its input it is (from 1).
    ordinal: u64,
    /// This chunk: taken from (it was running), and the frames it gave.
    taken: bool,
    got: u64,
    bytes: Vec<u8>,
    samples: Vec<i32>,
    /// Its chunks on the device not all played yet: the device frame each
    /// starts at and how many of its frames it has.
    sent: VecDeque<(u64, u64)>,
    /// The timeline epoch (restarts by flush) of what was taken last.
    epoch: u64,
}

struct Slot {
    role: Role,
    state: Arc<SlotState>,
    gain: Gain,
    /// Ramps taken.
    ramps: u64,
    /// Outputs opened on it.
    opened: u64,
    /// What it was offered, once asked.
    caps: Option<Capabilities>,
    /// This chunk's gains, or the one for all of it.
    gains: Vec<f64>,
    flat: Option<f64>,
    /// It was heard in the last chunk: it supplied frames at a gain other
    /// than 0.
    heard: bool,
    stream: Option<Stream>,
}

struct Device {
    output: Box<dyn Output>,
    format: StreamFormat,
    info: StreamInfo,
    /// Paused by the mixer (nothing to play).
    paused: bool,
    /// The Lead output whose tap it runs, and its frame map.
    tap: Option<(u64, Arc<FrameMap>)>,
    mix: Mix,
}

pub(super) struct Run {
    target: Box<dyn OutputTarget>,
    options: MixerOptions,
    /// The device's capabilities, once asked.
    caps: Option<Capabilities>,
    tx: Sender<Msg>,
    shared: Arc<MixShared>,
    slots: BTreeMap<u64, Slot>,
    device: Option<Device>,
    monitor: Option<Box<dyn Monitor>>,
    /// The device's openings.
    epoch: u64,
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
            slots: BTreeMap::new(),
            device: None,
            monitor: None,
            epoch: 0,
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
            Msg::Slot {
                id,
                role,
                gain,
                state,
            } => {
                self.slots.insert(
                    id,
                    Slot {
                        role,
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
            Msg::Tap { stream, tap, done } => {
                let _ = done.send(self.set_tap(stream, tap));
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
        if let Some(old) = self.slots.get_mut(&slot_id).and_then(|s| s.stream.take()) {
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
        let core = Core::new(shared, Box::new(driver), info, request);
        let slot = self.slots.get_mut(&slot_id).expect("checked");
        slot.opened += 1;
        slot.stream = Some(Stream {
            link: link.clone(),
            ordinal: slot.opened,
            taken: false,
            got: 0,
            bytes: Vec::new(),
            samples: Vec::new(),
            sent: VecDeque::new(),
            epoch: 0,
        });
        Ok(Box::new(InputOutput {
            core,
            link,
            tx: self.tx.clone(),
            role,
        }))
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
        self.device = Some(Device {
            format: info.format,
            mix: Mix::new(info.format),
            info,
            output,
            paused: false,
            tap: None,
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
        if let Some(m) = &mut self.monitor {
            m.mixed(&self.out, ch, self.given);
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
        if let Some(m) = &mut self.monitor {
            m.closed();
        }
    }

    /// Closes every input's output with `error`.
    fn close_streams(&mut self, error: Error) {
        for slot in self.slots.values_mut() {
            if let Some(stream) = slot.stream.take() {
                stream.link.running.store(false, Ordering::SeqCst);
                stream.link.shared.fail(error.clone());
            }
        }
        if let Some(d) = &mut self.device {
            if d.tap.take().is_some() {
                let _ = d.output.set_tap(None);
            }
        }
    }

    /// Output `id` closed (or was let go of): the device closes with the
    /// last one.
    fn closed(&mut self, id: u64) {
        for slot in self.slots.values_mut() {
            if slot.stream.as_ref().is_some_and(|s| s.link.id == id) {
                slot.stream = None;
            }
        }
        if let Some(d) = &mut self.device {
            if d.tap.as_ref().is_some_and(|(t, _)| *t == id) {
                d.tap = None;
                let _ = d.output.set_tap(None);
            }
        }
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

    fn set_tap(&mut self, stream: u64, tap: Option<Box<dyn Tap>>) -> Result<(), Error> {
        let device = self.device.as_mut().ok_or(Error::Closed)?;
        match tap {
            Some(tap) => {
                let map = Arc::new(FrameMap::new());
                let mapped = MappedTap::new(tap, map.clone());
                device.output.set_tap(Some(Box::new(mapped)))?;
                device.tap = Some((stream, map));
            }
            None => {
                device.tap = None;
                device.output.set_tap(None)?;
            }
        }
        Ok(())
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
        let sb = format.sample.bytes();
        let first = self.frame;
        let end = first + n as u64;
        // What each output gives.
        for slot in self.slots.values_mut() {
            let Some(stream) = &mut slot.stream else {
                continue;
            };
            stream.taken = stream.link.running.load(Ordering::SeqCst);
            stream.got = 0;
            if !stream.taken {
                continue;
            }
            stream.bytes.resize(n * format.frame_bytes(), 0);
            // None: a flush keeps the mixer out; silence.
            if let Some((lead, got, epoch)) = stream.link.shared.render_input(&mut stream.bytes) {
                if epoch != stream.epoch {
                    stream.sent.clear();
                    stream.epoch = epoch;
                }
                if got > 0 {
                    stream.sent.push_back((first, got));
                }
                if let Some((_, map)) = device.tap.as_ref().filter(|(t, _)| *t == stream.link.id) {
                    map.push(first, lead, got);
                }
                stream.got = got;
            }
            stream.samples.resize(n * ch, 0);
            for (s, b) in stream.samples.iter_mut().zip(stream.bytes.chunks_exact(sb)) {
                *s = unpack_sample(b, format.sample);
            }
        }
        // Where the waiting ramps start: their own frame, the first audio
        // of an output opened since, or with the ramp they follow.
        self.starts.clear();
        for (&id, slot) in &self.slots {
            let audio = |after: u64| {
                slot.stream
                    .as_ref()
                    .is_some_and(|s| s.got > 0 && s.ordinal > after)
            };
            let start = match slot.gain.next() {
                Some(When::Now) => Some(first),
                Some(When::Frame(f)) if f < end => Some(f.max(first)),
                Some(When::FirstAudio { after }) if audio(after) => Some(first),
                _ => None,
            };
            if let Some(at) = start {
                self.starts.push((id, at));
            }
        }
        let led = self.starts.len();
        for (&id, slot) in &self.slots {
            if let Some(When::With(lead)) = slot.gain.next() {
                let at = self.starts[..led]
                    .iter()
                    .find(|(i, _)| *i == lead)
                    .map(|&(_, at)| at);
                if let Some(at) = at {
                    self.starts.push((id, at));
                }
            }
        }
        // The gains, and who is heard.
        let (mut heard, mut alone) = (0, None);
        for (&id, slot) in self.slots.iter_mut() {
            let start = self
                .starts
                .iter()
                .find(|(i, _)| *i == id)
                .map(|&(_, at)| at);
            slot.gains.resize(n, 0.0);
            slot.flat = slot.gain.fill(first, start, format.rate, &mut slot.gains);
            slot.heard = slot.stream.as_ref().is_some_and(|s| s.taken && s.got > 0)
                && slot.flat != Some(0.0);
            if slot.heard {
                heard += 1;
                alone = Some(id);
            }
        }
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
                let gain = match slot.flat {
                    Some(g) => Gains::Flat(g),
                    None => Gains::Each(&slot.gains),
                };
                device.mix.add(&stream.samples, gain);
            }
            device.mix.end(untouched, &mut self.out);
            Made::Mixed
        };
        self.untouched = made != Made::Mixed;
        if let Some(m) = &mut self.monitor {
            m.mixed(&self.out, ch, self.given);
        }
        self.given += (self.out.len() / ch) as u64;
        self.frame = end;
    }

    /// Publishes the clock, what each output has on the device still, and
    /// each input's gain; settles the ramps of inputs with nothing to mix.
    fn report(&mut self) {
        let mut failed = None;
        let mut heard = 0;
        if let Some(device) = &self.device {
            match device.output.position() {
                Ok(played) => {
                    heard = played;
                    for stream in self.slots.values_mut().filter_map(|s| s.stream.as_mut()) {
                        while stream.sent.front().is_some_and(|&(d, n)| d + n <= played) {
                            stream.sent.pop_front();
                        }
                        let unplayed: u64 = stream
                            .sent
                            .iter()
                            .map(|&(d, n)| n - played.saturating_sub(d).min(n))
                            .sum();
                        stream.link.shared.set_downstream(stream.epoch, unplayed);
                    }
                }
                Err(e) => failed = Some(e),
            }
        }
        if let Some(e) = failed {
            self.device_failed(e);
        }
        for slot in self.slots.values_mut() {
            // Nothing of it will be heard for now: no output, or paused
            // (one held back for a flush or a prefill comes back).
            let idle = slot
                .stream
                .as_ref()
                .is_none_or(|s| s.link.paused.load(Ordering::SeqCst));
            if idle {
                slot.gain.settle();
            }
            let s = &slot.state;
            s.gain.store(slot.gain.last().to_bits(), Ordering::Release);
            if !slot.gain.moving() {
                s.settled.store(slot.ramps, Ordering::Release);
            }
        }
        let timebase = Timebase {
            epoch: self.epoch,
            rate: self.device.as_ref().map_or(0, |d| d.format.rate),
            frame: self.frame,
            heard,
        };
        *self
            .shared
            .timebase
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = timebase;
        self.shared
            .passthrough
            .store(self.untouched, Ordering::Relaxed);
        if let Some(d) = &self.device {
            self.shared
                .clipped
                .store(d.mix.clipped(), Ordering::Relaxed);
        }
    }
}
