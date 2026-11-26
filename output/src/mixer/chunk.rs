//! Mixing a chunk: each input as the mixer thread runs it (its lanes, and
//! the switch of one with parts), what each output gives (a follower's in
//! lockstep with its leader's, see `follow.rs`) split into its lanes, and
//! every lane's gains (its automation's on top, see `automation.rs`), for
//! the meters and the device's frames (`run.rs`).

use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::automation::AutomationRun;
use super::fader::Mapped;
use super::input::{Feed, Span};
use super::mic::Port;
use super::mix::Gains;
use super::parts::{split, touch, LaneRun, Switch};
use super::ramp::{resolve_starts, start_of, Audio, GainId};
use super::run::Inputs;
use super::strips::Strips;
use super::SlotState;
use crate::lanes::Parts;

/// A chunk mixed: the device's opening (its epoch), its first frame there,
/// and how many.
pub(super) type Chunk = (u64, u64, usize);

/// A lane heard in a chunk: its strip, its samples, the gains it is heard
/// at, and whether it may go to the device as written (an exact lane, at
/// exactly 1, alone).
#[derive(Clone, Copy)]
pub(crate) struct HeardLane<'a> {
    pub(super) strip: u64,
    pub(super) samples: &'a [i32],
    pub(super) gains: Gains<'a>,
    pub(super) exact: bool,
}

/// What an input's output gives, as its lanes are mixed: the one answer
/// to whether its frames carry parts ([`Slot::shape`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Shape {
    /// A plain input: main, its one lane.
    Plain,
    /// An input with parts, its output a lane stream (or none: a gap in
    /// one): main and the parts, with a control word a frame.
    Lanes,
    /// An input with parts, a plain output on it: main alone, with no
    /// parts on any frame.
    Main,
}

/// An input as the mixer thread runs it.
pub(super) struct Slot {
    /// What it publishes, and its role (the mixer thread sets it).
    pub(super) state: Arc<SlotState>,
    /// Its lanes, main first: one for a plain input.
    pub(super) lanes: Vec<LaneRun>,
    /// An input with parts: its switch between main and them.
    pub(super) switch: Option<Switch>,
    /// Its automation.
    pub(super) automation: AutomationRun,
    /// The id of the last output opened on it (0: none): a fade for the
    /// first audio waits for one newer.
    pub(super) opened: u64,
    /// A lane of it was heard in the last chunk.
    pub(super) heard: bool,
    pub(super) feed: Option<Feed>,
    /// The epoch and the device frame less the stream frame of the last
    /// frames it gave, as published for [`super::Fader::at_stream`].
    mapped: (u64, u64),
}

impl Slot {
    /// Input `id` on the strips `lanes` (main's first) at `gain`, its state
    /// `state`.
    pub(super) fn new(id: u64, state: Arc<SlotState>, lanes: &[u64], gain: f64) -> Slot {
        let runs = lanes
            .iter()
            .enumerate()
            .map(|(k, &strip)| {
                let gain = if k == 0 { gain } else { 1.0 };
                LaneRun::new(GainId::new(id, k as u16), strip, gain)
            })
            .collect();
        Slot {
            state,
            lanes: runs,
            switch: (lanes.len() > 1).then(Switch::new),
            automation: AutomationRun::new(lanes.len() - 1),
            opened: 0,
            heard: false,
            feed: None,
            mapped: (0, 0),
        }
    }

    /// Nothing of it will be heard for now: no output, or paused, or
    /// following a leader paused (one held back for a flush or a prefill
    /// comes back).
    pub(super) fn idle(&self) -> bool {
        self.feed.as_ref().is_none_or(Feed::paused)
    }

    /// Its output is gone (if it had one), and so are its format on the
    /// desk, the map of its frames and the automation given for it (one
    /// given with no output open waits for the next).
    pub(super) fn drop_feed(&mut self) -> Option<Feed> {
        self.unmap();
        self.state.format.store(0, Ordering::Relaxed);
        self.state.capacity.store(0, Ordering::Relaxed);
        self.state.queued.store(0, Ordering::Relaxed);
        self.state.dropped.store(0, Ordering::Relaxed);
        self.state.next.store(0, Ordering::Relaxed);
        let feed = self.feed.take();
        if feed.is_some() {
            self.automation.reset();
        }
        feed
    }

    /// What its output gives (see [`Shape`]).
    pub(super) fn shape(&self) -> Shape {
        match (&self.switch, &self.feed) {
            (None, _) => Shape::Plain,
            (Some(_), Some(feed)) if feed.lanes == 1 => Shape::Main,
            (Some(_), _) => Shape::Lanes,
        }
    }

    /// The lanes its output supplied this chunk: main, or every one.
    fn supplied(&self) -> usize {
        match self.feed.as_ref().filter(|f| f.got > 0) {
            None => 0,
            Some(_) if self.shape() == Shape::Lanes => self.lanes.len(),
            Some(_) => 1,
        }
    }

    /// Lane `k`'s samples this chunk: a lane stream's split into its lanes,
    /// a plain stream's as given.
    fn samples(&self, k: usize) -> &[i32] {
        match &self.feed {
            Some(feed) if self.shape() != Shape::Lanes => &feed.samples,
            _ => &self.lanes[k].samples,
        }
    }

    /// Its heard lanes: strip, samples, the gains they are heard at.
    pub(super) fn heard_lanes(&self) -> impl Iterator<Item = HeardLane<'_>> + '_ {
        self.lanes
            .iter()
            .enumerate()
            .filter(|(_, l)| l.heard)
            .map(|(k, l)| HeardLane {
                strip: l.strip,
                samples: self.samples(k),
                gains: l.gains.as_gains(),
                exact: l.exact,
            })
    }

    /// Its lanes metered: the heard ones at the gains they are heard at,
    /// into their strips' subtrees; a part present but not heard at what
    /// it would be heard at switched in, into its own strip only (its
    /// group meters main, heard in its place).
    pub(super) fn metered_lanes(
        &self,
    ) -> impl Iterator<Item = (u64, &[i32], Gains<'_>, bool)> + '_ {
        self.lanes
            .iter()
            .enumerate()
            .filter_map(|(k, l)| match (l.heard, l.metering) {
                (true, _) => Some((l.strip, self.samples(k), l.gains.as_gains(), true)),
                (false, true) => Some((l.strip, self.samples(k), l.meter.as_gains(), false)),
                _ => None,
            })
    }

    /// Its frames' map is gone (a flush, a new output, none): until frames
    /// of it are mixed again, there is no stream frame to time a change on.
    pub(super) fn unmap(&mut self) {
        self.mapped = (0, 0);
        self.state.mapped.store(&Mapped::default());
    }

    /// Its stream frames go `offset` device frames on in epoch `epoch`:
    /// published when that moves the map.
    pub(super) fn map_offset(&mut self, epoch: u64, offset: u64) {
        let mapped = (epoch, offset);
        if mapped != self.mapped {
            self.mapped = mapped;
            let (epoch, offset) = mapped;
            self.state.mapped.store(&Mapped { epoch, offset });
        }
    }

    /// The epoch and the offset of its map (`None`: not mapped).
    pub(super) fn mapped(&self) -> Option<(u64, u64)> {
        (self.mapped.0 != 0).then_some(self.mapped)
    }

    /// Takes its output's frames of `span` in the chunk of `n` frames from
    /// device frame `first` (see [`Feed::take`]), and notes where they
    /// went in the microphone's map, when `port` taps the device for that
    /// output. Returns how many device frames on from their stream frames
    /// they went, when it gave any (its own map, for the caller to keep:
    /// a follower's is its leader's).
    pub(super) fn take(&mut self, n: usize, first: u64, span: Span, port: &Port) -> Option<u64> {
        let feed = self.feed.as_mut()?;
        if !feed.take(n, first, span) {
            return None;
        }
        let (device, frame, got) = (first + feed.at as u64, feed.frame, feed.got);
        if let Some(map) = port.map_for(feed.link.id) {
            map.push(device, frame, got);
        }
        (got > 0).then(|| device.wrapping_sub(frame))
    }

    /// Its lanes' gains for the `n` frames from `first` (`starts` the
    /// faders' ramps that start in them): each fader's, the switch's,
    /// their strips' paths, its automation's; and which lanes are heard.
    fn fill(&mut self, first: u64, n: usize, rate: u32, strips: &Strips, starts: &[(GainId, u64)]) {
        let given = self.feed.as_ref().and_then(Feed::given);
        self.automation.fill(first, n, rate, given);
        let auto = &self.automation;
        // Each fader times its lane's curve (main's fader the input's own,
        // over every lane as the fader is).
        for (k, lane) in self.lanes.iter_mut().enumerate() {
            let start = start_of(starts, lane.id);
            let LaneRun { gain, gains, .. } = lane;
            gains.fill(gain, first, n, start, rate);
            gains.times(auto.lane(k));
        }
        let supplied = given.is_some();
        let shape = self.shape();
        if let Some(switch) = &mut self.switch {
            // How the parts stand, their faders' ramps started.
            let touched = touch(&self.lanes, strips, starts).merge(auto.touch());
            switch.fill(first, n, rate, shape == Shape::Lanes, touched);
            let wide = supplied && shape == Shape::Lanes;
            fill_parts(&mut self.lanes, switch, strips, n, wide);
        }
        let main = &mut self.lanes[0];
        main.gains.times(auto.main());
        // A lane input's main is gated as its parts stand: a part soloed
        // keeps it while it is heard in their place.
        let path = match shape {
            Shape::Plain => strips.path(main.strip),
            Shape::Lanes | Shape::Main => strips.lead_path(main.strip),
        };
        main.gains.times(path);
        main.heard = supplied && main.gains.flat() != Some(0.0);
        main.last = main.gains.last(n);
        self.heard = self.lanes.iter().any(|l| l.heard);
    }

    /// Publishes its faders' gains and whether they settled, its lanes'
    /// gains and whether they were heard, its switch, and its ring's fill.
    pub(super) fn publish(&self) {
        let s = &self.state;
        let idle = self.idle();
        for ((lane, fader), view) in self.lanes.iter().zip(&*s.faders).zip(&*s.lanes) {
            fader
                .gain
                .store(lane.gain.last().to_bits(), Ordering::Release);
            if !lane.gain.moving() {
                fader.settled.store(lane.ramps, Ordering::Release);
            }
            view.gain.store(lane.last.to_bits(), Ordering::Relaxed);
            view.heard.store(lane.heard && !idle, Ordering::Relaxed);
        }
        if let (Some(switch), Some(parts)) = (&self.switch, &s.parts) {
            parts
                .control
                .store(switch.control.word() as u8, Ordering::Relaxed);
            parts.share.store(switch.last.to_bits(), Ordering::Relaxed);
        }
        s.heard.store(self.heard && !idle, Ordering::Relaxed);
        let auto = &s.automation;
        let view = self.automation.view();
        auto.set.store(view.is_some(), Ordering::Relaxed);
        if let Some((gains, settled)) = view {
            for (a, g) in auto.gains.iter().zip(gains) {
                a.store(g.to_bits(), Ordering::Relaxed);
            }
            auto.settled.store(settled, Ordering::Relaxed);
        }
        if let Some(st) = &self.feed {
            s.queued.store(st.link.shared.ring.len(), Ordering::Relaxed);
            s.dropped.store(st.link.shared.dropped(), Ordering::Relaxed);
            s.next.store(st.link.shared.head(), Ordering::Relaxed);
        }
    }
}

/// The parts' gains of an input with parts (`lanes`, main first) for a
/// chunk of `n` frames, through its `switch`: each part's fader times the
/// input's (main's), the switch's share and the part's strip's path, heard
/// where its output supplied the parts (`wide`) and that is not 0; main's
/// fader times the switch's rest.
fn fill_parts(lanes: &mut [LaneRun], switch: &Switch, strips: &Strips, n: usize, wide: bool) {
    let (main, parts) = lanes.split_at_mut(1);
    let main = &mut main[0];
    let present = wide && switch.control != Parts::None;
    for part in parts.iter_mut() {
        let path = strips.path(part.strip);
        part.gains.times(&main.gains);
        part.metering = present;
        if present {
            part.meter.assign(&part.gains).times(path);
        }
        part.gains.times(&switch.share).times(path);
        part.heard = wide && part.gains.flat() != Some(0.0);
        part.last = part.gains.last(n);
    }
    main.gains.times(&switch.rest);
}

impl Inputs {
    /// Takes what each output gives of the chunk of `n` frames from device
    /// frame `first` of epoch `epoch` (`ch` channels a frame), in its lanes:
    /// the leaders' and the plain ones' first, then the followers' in
    /// lockstep with them; and resolves where the faders' waiting ramps
    /// start in it. Returns whether the chunk is to be mixed: on a device
    /// that plays only what it is given (not `realtime`), not when no
    /// output gave a frame to it. The outputs it was sized for have stopped
    /// since then (closed, paused or flushed between the look at their
    /// rings and the take), and mixing it would add a period of silence
    /// the device then plays as if written.
    pub(super) fn take(&mut self, chunk: Chunk, ch: usize, realtime: bool, port: &Port) -> bool {
        let (epoch, first, n) = chunk;
        let end = first + n as u64;
        let leads = |s: &&mut Slot| s.feed.as_ref().is_some_and(|st| st.follow.is_none());
        for slot in self.slots.values_mut().filter(leads) {
            if let Some(offset) = slot.take(n, first, Span::whole(n), port) {
                slot.map_offset(epoch, offset);
            }
        }
        self.take_followers(n, first, port, epoch);
        // The lanes of each lane stream, and the words of the frames it
        // gave (none when it gave none: a gap).
        for slot in self.slots.values_mut() {
            let lanes = slot.shape() == Shape::Lanes;
            let Some(switch) = slot.switch.as_mut() else {
                continue;
            };
            switch.words.clear();
            switch.from = 0;
            if let Some(feed) = slot.feed.as_ref().filter(|s| s.taken && lanes) {
                let (at, got) = (feed.at, feed.got as usize);
                split(
                    &feed.samples,
                    ch,
                    at,
                    got,
                    &mut slot.lanes,
                    &mut switch.words,
                );
                switch.from = at;
            }
        }
        let gave = |s: &Slot| s.feed.as_ref().is_some_and(|st| st.got > 0);
        if !realtime && !self.slots.values().any(gave) {
            return false;
        }
        // Where the faders' waiting ramps start (one for the first audio on
        // the frame that audio starts at).
        let waiting = self.slots.values().flat_map(|slot| {
            let audio = slot
                .feed
                .as_ref()
                .filter(|s| s.got > 0)
                .map(|s| s.audio(first));
            slot.lanes.iter().filter_map(move |lane| {
                let when = lane.gain.next()?;
                Some((lane.id, when, audio))
            })
        });
        resolve_starts(first, end, waiting, &mut self.starts);
        true
    }

    /// The strips' gains along their paths for the chunk taken at `rate`;
    /// each lane's through its strip's (what it is heard at), and who is
    /// heard.
    pub(super) fn fill(&mut self, chunk: Chunk, rate: u32, strips: &mut Strips) {
        let (_, first, n) = chunk;
        let audio = self.slots.values().flat_map(|s| {
            let audio = s.feed.as_ref().map(|st| st.audio(first));
            let lanes = s.lanes[..s.supplied()].iter();
            lanes.filter_map(move |l| Some((l.strip, audio?)))
        });
        strips.fill(first, n, rate, audio);
        for slot in self.slots.values_mut() {
            slot.fill(first, n, rate, strips, &self.starts);
        }
        strips.chunk_done();
    }
}

impl Feed {
    /// Its audio in the chunk from device frame `first` taken last.
    fn audio(&self, first: u64) -> Audio {
        Audio {
            output: self.link.id,
            at: first + self.at as u64,
        }
    }
}
