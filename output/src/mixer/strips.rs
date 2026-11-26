//! The strips as the mixer thread runs them. Each has three gains on the
//! clock: `heard`, what its own inputs are heard at (its level, 0 while
//! muted, times its solo gate: 1 while nothing is soloed, or it or a group
//! it sits in is; else 0), `vca`, what it puts on its children (its level,
//! 0 while muted), and `lead`, what the main of a lane input on it is heard
//! at: as `heard`, but with its gate open also while one of its children's
//! is, since a part soloed stands for main while main is heard in the
//! parts' place (once the parts are heard, each part's own gate applies).
//! Each is one ramp, so a mute and a solo set together never multiply into
//! a bump. An input is heard at its strip's path: the strip's `heard` (a
//! lane input's main: its `lead`) times the `vca` of every group above it;
//! a strip at the top of the desk is heard at its own alone. A change
//! set for a frame waits for it and starts both ramps on that very frame,
//! inside the chunk being filled. A strip meters its subtree: its inputs
//! and its children's, summed after their gains.

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::envelope::Envelope;
use super::mix::Gains;
use super::ramp::{Aimed, Audio, When};
use super::strip::{channel_peaks, Change, StripState, STRIP_RAMP};

/// One of a strip's gains: on the clock, aimed where the settings say,
/// and this chunk's gains.
struct Factor {
    aimed: Aimed,
    env: Envelope,
}

impl Factor {
    fn new(at: f64) -> Factor {
        Factor {
            aimed: Aimed::new(at),
            env: Envelope::new(at),
        }
    }

    /// Ramps to `to` over [`STRIP_RAMP`], from the next frame filled;
    /// returns whether that is a change.
    fn aim(&mut self, to: f64) -> bool {
        self.aimed.aim(to, STRIP_RAMP)
    }

    /// The gains of frames `first..first + n` at `rate`.
    fn fill(&mut self, first: u64, n: usize, rate: u32) {
        let gain = &mut self.aimed.gain;
        let start = (gain.next() == Some(When::Now)).then_some(first);
        self.env.fill(gain, first, n, start, rate);
    }

    /// A ramp under way, or due now, ends at once ([`super::ramp::Gain::settle`]).
    fn settle(&mut self) {
        self.aimed.gain.settle();
    }

    /// Whether it is changing, or a ramp waits.
    fn moving(&self) -> bool {
        self.aimed.gain.moving()
    }
}

/// A strip as the mixer thread runs it.
struct StripRun {
    state: Arc<StripState>,
    /// The groups it sits in, nearest first.
    ancestors: Vec<u64>,
    /// The settings in force, and the solo gate they give (with the
    /// others' solos).
    level: f64,
    mute: bool,
    solo: bool,
    gate: f64,
    /// The gate of a lane input's main on it: open while its own is, or
    /// one of its children's.
    lead_gate: f64,
    /// A change of the level, and one of the mute, waiting for its start.
    next_level: Option<(When, f64)>,
    next_mute: Option<(When, bool)>,
    /// The newest set of the level, and of the mute, taken.
    level_set: u64,
    mute_set: u64,
    heard: Factor,
    vca: Factor,
    lead: Factor,
    /// This chunk's gains along its path: what its inputs are heard at,
    /// and a lane input's main.
    path: Envelope,
    lead_path: Envelope,
    /// This chunk: where the audio its level's and its mute's changes wait
    /// for starts, of the outputs on inputs of its subtree.
    arrived: (Option<u64>, Option<u64>),
    /// Inputs of its subtree may be heard.
    busy: bool,
    /// Its subtree's heard inputs summed this chunk, for the meter (and
    /// whether any was).
    sum: Vec<f64>,
    metered: bool,
    /// The earliest frame its targets changed at since the last chunk
    /// ([`Strips::chunk_done`]; at or before the chunk's first: from its
    /// start).
    changed: Option<u64>,
}

impl StripRun {
    fn own_target(&self) -> f64 {
        if self.mute {
            0.0
        } else {
            self.level
        }
    }

    /// Both gains ramp to what the settings give now, from frame `at` (0:
    /// from the next chunk's first), which is noted when either changes.
    fn aim(&mut self, at: u64) {
        let own = self.own_target();
        let heard = self.heard.aim(own * self.gate);
        let lead = self.lead.aim(own * self.lead_gate);
        if self.vca.aim(own) || heard || lead {
            self.changed = Some(self.changed.map_or(at, |c| c.min(at)));
        }
    }

    /// The level in force and the one waiting go to its handles, unless a
    /// newer set of it is on its way (its handle shows that one).
    fn publish_level(&self) {
        if self.state.level_sets.load(Ordering::Acquire) == self.level_set {
            let pending = self.next_level.map(|c| c.1);
            self.state.show_level(Some(self.level), pending);
        }
    }

    /// The mute in force and the one waiting, as [`StripRun::publish_level`].
    fn publish_mute(&self) {
        if self.state.mute_sets.load(Ordering::Acquire) == self.mute_set {
            let pending = self.next_mute.map(|c| c.1);
            self.state.show_mute(Some(self.mute), pending);
        }
    }

    /// Its gains for frames `first..first + n`: the changes waiting that
    /// come in them are taken on their frames, both gains cut there.
    fn fill(&mut self, first: u64, n: usize, rate: u32) {
        let end = first + n as u64;
        let (to_level, to_mute) = self.arrived;
        let (mut at, mut cut) = (first, false);
        loop {
            let level = self.next_level.and_then(|c| c.0.due(at, end, to_level));
            let mute = self.next_mute.and_then(|c| c.0.due(at, end, to_mute));
            let Some(f) = level.into_iter().chain(mute).min() else {
                break;
            };
            if !cut {
                self.heard.env.cut(n);
                self.vca.env.cut(n);
                self.lead.env.cut(n);
                cut = true;
            }
            self.fill_part(first, at, f, rate);
            if level == Some(f) {
                self.level = self.next_level.take().expect("due").1;
                self.publish_level();
            }
            if mute == Some(f) {
                self.mute = self.next_mute.take().expect("due").1;
                self.publish_mute();
            }
            self.aim(f);
            at = f;
        }
        if cut {
            self.fill_part(first, at, end, rate);
            self.heard.env.join();
            self.vca.env.join();
            self.lead.env.join();
        } else {
            self.heard.fill(first, n, rate);
            self.vca.fill(first, n, rate);
            self.lead.fill(first, n, rate);
        }
    }

    /// Both gains for device frames `from..to` of the chunk from `first`.
    fn fill_part(&mut self, first: u64, from: u64, to: u64, rate: u32) {
        let (from, to) = ((from - first) as usize, (to - first) as usize);
        for Factor { aimed, env } in [&mut self.heard, &mut self.vca, &mut self.lead] {
            env.fill_part(&mut aimed.gain, first, from, to, rate);
        }
    }
}

/// The solo gate of a strip soloed or not (`solo`) in the groups
/// `ancestors`: open while nothing is soloed, or it or a group it sits in
/// is.
fn gate(strips: &BTreeMap<u64, StripRun>, solo: bool, ancestors: &[u64], any_solo: bool) -> f64 {
    let soloed = |a: &u64| strips.get(a).is_some_and(|a| a.solo);
    if !any_solo || solo || ancestors.iter().any(soloed) {
        1.0
    } else {
        0.0
    }
}

/// How a strip stands, as a lane input's switch between main and its parts
/// looks at it ([`Strips::look`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Look {
    /// Its own gain's target: its level, or 0 while muted.
    pub(super) own: f64,
    /// Its solo gate.
    pub(super) gate: f64,
    /// The gate of a lane input's main on it.
    pub(super) lead_gate: f64,
    /// The earliest frame either target changed at since the last chunk
    /// (`None`: neither did; at or before the chunk's first: from its
    /// start, as [`super::parts`] counts it).
    pub(super) changed: Option<u64>,
    /// A ramp of it is under way or waits.
    pub(super) moving: bool,
}

/// The strips, as the mixer thread runs them, by id: a group's id is below
/// its children's (it was made first).
#[derive(Default)]
pub(super) struct Strips(BTreeMap<u64, StripRun>);

impl Strips {
    /// The strip after `id` (the first with `None`), by id.
    fn after(&self, id: Option<u64>) -> Option<u64> {
        let from = id.map_or(0, |i| i + 1);
        self.0.range(from..).next().map(|(&k, _)| k)
    }

    /// Of `audio`, that of the inputs on strip `id` or a strip inside it.
    fn audio_of<'a>(
        &'a self,
        id: u64,
        audio: impl Iterator<Item = (u64, Audio)> + 'a,
    ) -> impl Iterator<Item = Audio> + 'a {
        let inside = move |strip: u64| {
            strip == id
                || self
                    .0
                    .get(&strip)
                    .is_some_and(|s| s.ancestors.contains(&id))
        };
        audio
            .filter(move |&(strip, _)| inside(strip))
            .map(|(_, a)| a)
    }

    /// `f` on strip `id` and on every group it sits in.
    fn each_up(&mut self, id: u64, mut f: impl FnMut(&mut StripRun)) {
        let Some(s) = self.0.get_mut(&id) else {
            return;
        };
        f(s);
        // Each group is looked up by index again: holding the strip's
        // list while borrowing the groups would mean copying it, which
        // allocates.
        for k in 0..s.ancestors.len() {
            let a = self.0[&id].ancestors[k];
            if let Some(s) = self.0.get_mut(&a) {
                f(s);
            }
        }
    }

    /// Every gate to what the solos give now, and the gains ramp to it.
    fn aim_gates(&mut self) {
        let any_solo = self.0.values().any(|s| s.solo);
        let mut next = self.after(None);
        while let Some(id) = next {
            let s = &self.0[&id];
            let to = gate(&self.0, s.solo, &s.ancestors, any_solo);
            let s = self.0.get_mut(&id).expect("listed");
            s.gate = to;
            s.lead_gate = to;
            next = self.after(Some(id));
        }
        // A child's gate open opens its group's for a lane input's main.
        let open: Vec<u64> = self
            .0
            .values()
            .filter(|s| s.gate > 0.0)
            .filter_map(|s| s.ancestors.first().copied())
            .collect();
        for id in open {
            if let Some(s) = self.0.get_mut(&id) {
                s.lead_gate = 1.0;
            }
        }
        for s in self.0.values_mut() {
            s.aim(0);
        }
    }

    /// Strip `id` made (in its group `state.parent`, made before it), as
    /// every strip starts: at 0 dB, unmuted, not soloed; silent at once
    /// while another strip is soloed outside its groups.
    pub(super) fn add(&mut self, id: u64, state: Arc<StripState>) {
        let mut ancestors = Vec::new();
        if let Some(p) = state.parent {
            let group = self.0.get(&p).expect("a strip's group is made before it");
            ancestors.push(p);
            ancestors.extend_from_slice(&group.ancestors);
        }
        let any_solo = self.0.values().any(|s| s.solo);
        let gate = gate(&self.0, false, &ancestors, any_solo);
        let strip = StripRun {
            state,
            ancestors,
            level: 1.0,
            mute: false,
            solo: false,
            gate,
            lead_gate: gate,
            next_level: None,
            next_mute: None,
            level_set: 0,
            mute_set: 0,
            heard: Factor::new(gate),
            vca: Factor::new(1.0),
            lead: Factor::new(gate),
            path: Envelope::new(gate),
            lead_path: Envelope::new(gate),
            arrived: (None, None),
            busy: false,
            sum: Vec::new(),
            metered: false,
            changed: None,
        };
        self.0.insert(id, strip);
    }

    /// A change of strip `id`'s level or mute, from `when` (`None`: never;
    /// it changes nothing), the handles' `set`th of that setting. One now
    /// is in force at once, in place of a change of the same setting still
    /// waiting; one later waits, in place of that one.
    pub(super) fn set(&mut self, id: u64, change: Change, when: Option<When>, set: u64) {
        let Some(s) = self.0.get_mut(&id) else {
            return;
        };
        match change {
            Change::Level(_) => s.level_set = set,
            Change::Mute(_) => s.mute_set = set,
        }
        let now = when == Some(When::Now);
        match (change, when) {
            (Change::Level(g), Some(_)) if now => (s.level, s.next_level) = (g, None),
            (Change::Mute(on), Some(_)) if now => (s.mute, s.next_mute) = (on, None),
            (Change::Level(g), Some(w)) => s.next_level = Some((w, g)),
            (Change::Mute(on), Some(w)) => s.next_mute = Some((w, on)),
            // (Its handle showed it waiting: it shows what waits again.)
            (_, None) => {}
        }
        match change {
            Change::Level(_) => s.publish_level(),
            Change::Mute(_) => s.publish_mute(),
        }
        if now {
            s.aim(0);
        }
    }

    /// Strip `id` soloed (or not), now: every gate follows.
    pub(super) fn solo(&mut self, id: u64, on: bool) {
        let Some(s) = self.0.get_mut(&id) else {
            return;
        };
        s.solo = on;
        self.aim_gates();
    }

    /// Strip `id` gone (its children went before it; its solo goes with
    /// it).
    pub(super) fn remove(&mut self, id: u64) {
        self.0.remove(&id);
        self.aim_gates();
    }

    /// The device opened again: nothing has been mixed on it yet, so a
    /// ramp under way or due now ends at once (a mute set before the first
    /// audio is in force from its first frame), and changes waiting for a
    /// frame of the last epoch never come.
    pub(super) fn new_epoch(&mut self) {
        for s in self.0.values_mut() {
            s.heard.settle();
            s.vca.settle();
            s.lead.settle();
            if matches!(s.next_level, Some((When::Frame(_), _))) {
                s.next_level = None;
                s.publish_level();
            }
            if matches!(s.next_mute, Some((When::Frame(_), _))) {
                s.next_mute = None;
                s.publish_mute();
            }
        }
    }

    /// Their gains along their paths for frames `first..first + n` at
    /// `rate`. `audio` gives each output that supplied audio in them: the
    /// strip of its input, and its audio.
    pub(super) fn fill(
        &mut self,
        first: u64,
        n: usize,
        rate: u32,
        audio: impl Iterator<Item = (u64, Audio)> + Clone,
    ) {
        let mut next = self.after(None);
        while let Some(id) = next {
            let s = &self.0[&id];
            let arrival =
                |w: Option<When>| w.and_then(|w| w.arrival(self.audio_of(id, audio.clone())));
            let arrived = (
                arrival(s.next_level.map(|c| c.0)),
                arrival(s.next_mute.map(|c| c.0)),
            );
            self.0.get_mut(&id).expect("listed").arrived = arrived;
            next = self.after(Some(id));
        }
        for s in self.0.values_mut() {
            s.fill(first, n, rate);
        }
        let mut next = self.after(None);
        while let Some(id) = next {
            let run = self.0.get_mut(&id).expect("listed");
            let mut path = std::mem::take(&mut run.path);
            let mut lead = std::mem::take(&mut run.lead_path);
            let s = &self.0[&id];
            let groups = s.ancestors.iter().filter_map(|a| self.0.get(a));
            groups.fold(path.assign(&s.heard.env), |p, g| p.times(&g.vca.env));
            let groups = s.ancestors.iter().filter_map(|a| self.0.get(a));
            groups.fold(lead.assign(&s.lead.env), |p, g| p.times(&g.vca.env));
            let run = self.0.get_mut(&id).expect("listed");
            (run.path, run.lead_path) = (path, lead);
            next = self.after(Some(id));
        }
    }

    /// How strip `id` stands for a lane's switch: see [`Look`].
    pub(super) fn look(&self, id: u64) -> Look {
        let s = self.0.get(&id).expect("a lane's strip lives");
        Look {
            own: s.own_target(),
            gate: s.gate,
            lead_gate: s.lead_gate,
            changed: s.changed,
            moving: s.heard.moving() || s.vca.moving() || s.lead.moving(),
        }
    }

    /// The chunk is mixed: the changes noted for it are done with.
    pub(super) fn chunk_done(&mut self) {
        for s in self.0.values_mut() {
            s.changed = None;
        }
    }

    /// Strip `id`'s gains along its path this chunk.
    pub(super) fn path(&self, id: u64) -> &Envelope {
        &self.0.get(&id).expect("an input's strip lives").path
    }

    /// Strip `id`'s gains along its path this chunk, if it lives.
    pub(super) fn path_of(&self, id: u64) -> Option<&Envelope> {
        self.0.get(&id).map(|s| &s.path)
    }

    /// The gains along strip `id`'s path of a lane input's main on it this
    /// chunk (its `lead` for its own `heard`).
    pub(super) fn lead_path(&self, id: u64) -> &Envelope {
        &self.0.get(&id).expect("an input's strip lives").lead_path
    }

    /// Meters each strip: the heard inputs of its subtree (`inputs`: the
    /// strip, samples `channels` per frame, the gains they are heard at,
    /// and whether they count in the groups above it too) summed, then the
    /// peak of each channel.
    pub(super) fn meter<'a>(
        &mut self,
        channels: usize,
        inputs: impl Iterator<Item = (u64, &'a [i32], Gains<'a>, bool)>,
    ) {
        for s in self.0.values_mut() {
            s.metered = false;
        }
        for (strip, samples, gain, up) in inputs {
            let add = |s: &mut StripRun| {
                if !s.metered {
                    s.sum.clear();
                    s.sum.resize(samples.len(), 0.0);
                    s.metered = true;
                }
                gain.accumulate(&mut s.sum, samples, channels);
            };
            if up {
                self.each_up(strip, add);
            } else if let Some(s) = self.0.get_mut(&strip) {
                add(s);
            }
        }
        for s in self.0.values().filter(|s| s.metered) {
            let peaks = channel_peaks(&s.sum, channels, f64::abs);
            s.state.peaks.raise(&peaks);
        }
    }

    /// Room in strip `id`'s meter, and its groups', for chunks of `samples`
    /// samples, so metering them never allocates.
    pub(super) fn reserve_meter(&mut self, id: u64, samples: usize) {
        self.each_up(id, |s| s.sum.reserve(samples.saturating_sub(s.sum.len())));
    }

    /// Strips none of whose subtree's inputs will be heard for now
    /// (`busy`: the strips of those that may be) settle: a ramp ends at
    /// once (a change waiting for its frame or for audio still waits).
    pub(super) fn settle_idle(&mut self, busy: impl Iterator<Item = u64>) {
        for s in self.0.values_mut() {
            s.busy = false;
        }
        for id in busy {
            self.each_up(id, |s| s.busy = true);
        }
        for s in self.0.values_mut().filter(|s| !s.busy) {
            s.heard.settle();
            s.vca.settle();
            s.lead.settle();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;

    /// Strips `ids` with their parents.
    fn desk(strips: &[(u64, Option<u64>)]) -> Strips {
        let mut d = Strips::default();
        for &(id, parent) in strips {
            d.add(id, Arc::new(StripState::new("s", parent)));
        }
        d
    }

    /// Each strip's path gain at its targets (ramps settled).
    fn settled(d: &mut Strips) -> Vec<f64> {
        d.settle_idle(std::iter::empty());
        d.fill(0, 4, RATE, std::iter::empty());
        d.0.values().map(|s| s.path.flat().expect("flat")).collect()
    }

    /// Strip `id`'s path gains for `n` frames from `first`.
    fn path(d: &mut Strips, id: u64, first: u64, n: usize) -> Vec<f64> {
        d.fill(first, n, RATE, std::iter::empty());
        (0..n).map(|k| d.path(id).at(k)).collect()
    }

    #[test]
    fn the_path_multiplies_the_groups_and_solos_follow_it() {
        // 1 a group of 2 and 3; 4 at the top; 5 in 3 (a group in a group).
        let mut d = desk(&[
            (1, None),
            (2, Some(1)),
            (3, Some(1)),
            (4, None),
            (5, Some(3)),
        ]);
        assert_eq!(d.0[&5].ancestors, [3, 1]);
        d.set(1, Change::Level(0.5), Some(When::Now), 0);
        d.set(3, Change::Level(0.5), Some(When::Now), 0);
        assert_eq!(settled(&mut d), [0.5, 0.5, 0.25, 1.0, 0.25]);
        // A child soloed: its siblings, its group's own inputs and the
        // others go; its children stay; the group's level still counts.
        d.solo(3, true);
        assert_eq!(settled(&mut d), [0.0, 0.0, 0.25, 0.0, 0.25]);
        assert!(d.0[&3].solo);
        // The group soloed too: all of it.
        d.solo(1, true);
        assert_eq!(settled(&mut d), [0.5, 0.5, 0.25, 0.0, 0.25]);
        d.solo(3, false);
        d.solo(1, false);
        d.set(1, Change::Mute(true), Some(When::Now), 0);
        assert_eq!(settled(&mut d), [0.0, 0.0, 0.0, 1.0, 0.0]);
        // A strip made while another is soloed starts silent.
        d.solo(4, true);
        d.add(6, Arc::new(StripState::new("late", None)));
        assert_eq!(d.0[&6].heard.aimed.to(), 0.0);
        // A soloed strip gone: the others back.
        d.remove(4);
        assert_eq!(settled(&mut d), [0.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn a_solo_released_and_a_mute_set_together_never_blip() {
        let mut d = desk(&[(1, None), (2, None)]);
        d.solo(2, true);
        settled(&mut d);
        // Silent behind the other's solo; the solo released and the strip
        // muted in one batch: it stays silent all through.
        d.solo(2, false);
        d.set(1, Change::Mute(true), Some(When::Now), 0);
        assert!(path(&mut d, 1, 0, 1_500).iter().all(|&g| g == 0.0));
        // 5 ms apart (240 frames): up from the solo's release, then down
        // from where it got to, each a straight line.
        d.set(1, Change::Mute(false), Some(When::Now), 0);
        d.solo(2, true);
        settled(&mut d);
        d.solo(2, false);
        let up = path(&mut d, 1, 2_000, 240);
        d.set(1, Change::Mute(true), Some(When::Now), 0);
        let down = path(&mut d, 1, 2_240, 1_500);
        assert!(up.windows(2).all(|w| w[1] >= w[0]) && up[0] == 0.0);
        assert!(down.windows(2).all(|w| w[1] <= w[0]));
        assert!((down[0] - 0.25).abs() < 1e-12 && down[1_499] == 0.0);
    }

    #[test]
    fn a_change_for_a_frame_ramps_from_that_frame() {
        let mut d = desk(&[(1, None), (2, Some(1))]);
        d.set(1, Change::Mute(true), Some(When::Frame(130)), 0);
        d.set(2, Change::Level(0.5), Some(When::Frame(150)), 0);
        assert_eq!(d.0[&1].state.pending_mute(), Some(true));
        assert_eq!(d.0[&2].state.pending_level(), Some(0.5));
        d.fill(0, 100, RATE, std::iter::empty());
        assert_eq!(d.0[&2].path.flat(), Some(1.0));
        assert!(!d.0[&1].state.mute());
        let path = path(&mut d, 2, 100, 100);
        assert_eq!(path[..30], [1.0; 30]);
        // The group's mute from 130 (20 ms: 960 frames), the child's level
        // from 150, multiplied.
        let mute = |f: usize| 1.0 - (f - 130) as f64 / 960.0;
        for f in 130..200 {
            let level = if f < 150 {
                1.0
            } else {
                1.0 - 0.5 * (f - 150) as f64 / 960.0
            };
            assert!((path[f - 100] - mute(f) * level).abs() < 1e-12, "{f}");
        }
        assert!(d.0[&1].state.mute() && d.0[&2].state.level() == 0.5);
        assert_eq!(d.0[&1].state.pending_mute(), None);
        // A later change takes the place of one waiting; a change now
        // drops one waiting.
        d.set(2, Change::Level(0.25), Some(When::Frame(5_000)), 0);
        d.set(2, Change::Level(1.0), Some(When::Frame(6_000)), 0);
        assert_eq!(d.0[&2].next_level, Some((When::Frame(6_000), 1.0)));
        d.set(2, Change::Level(0.75), Some(When::Now), 0);
        assert_eq!((d.0[&2].next_level, d.0[&2].level), (None, 0.75));
        assert_eq!(d.0[&2].state.pending_level(), None);
        // One of an epoch gone never comes.
        d.set(1, Change::Mute(false), Some(When::Frame(9_000)), 0);
        d.new_epoch();
        assert_eq!(d.0[&1].next_mute, None);
        assert_eq!(d.0[&1].state.pending_mute(), None);
        // One whose start never comes: what its handle showed waiting is
        // put back as the mixer has it.
        d.0[&1].state.show_mute(Some(true), Some(false));
        d.set(1, Change::Mute(false), None, 0);
        assert_eq!(d.0[&1].state.pending_mute(), None);
    }

    #[test]
    fn an_older_set_never_shows_over_a_newer_one() {
        let mut d = desk(&[(1, None)]);
        // The handles made two sets and show the second; the first taken
        // leaves what they show, the second shows what the mixer has.
        let state = d.0[&1].state.clone();
        state.level_sets.store(2, Ordering::Relaxed);
        state.show_level(Some(0.25), None);
        d.set(1, Change::Level(0.5), Some(When::Now), 1);
        assert_eq!(state.level(), 0.25);
        d.set(1, Change::Level(0.25), Some(When::Frame(100)), 2);
        assert_eq!((state.level(), state.pending_level()), (0.5, Some(0.25)));
    }

    #[test]
    fn first_audio_is_an_output_of_the_subtree_opened_after() {
        let mut d = desk(&[(1, None), (2, Some(1)), (3, None)]);
        d.set(
            1,
            Change::Mute(true),
            Some(When::FirstAudio { after: 10 }),
            0,
        );
        // An older output, and a newer one on another strip: not yet.
        let audio = |output, at| Audio { output, at };
        d.fill(
            0,
            10,
            RATE,
            [(2, audio(10, 0)), (3, audio(11, 0))].into_iter(),
        );
        assert!(d.0[&1].next_mute.is_some());
        // Newer ones on its children: from the earliest of them.
        d.fill(
            10,
            10,
            RATE,
            [(2, audio(13, 17)), (2, audio(12, 14))].into_iter(),
        );
        assert!(d.0[&1].next_mute.is_none() && d.0[&1].mute);
        assert_eq!(d.0[&1].heard.env.at(4), 1.0);
        assert!(d.0[&1].heard.env.at(5) < 1.0);
    }

    #[test]
    fn a_change_to_where_it_is_keeps_the_path_flat() {
        let mut d = desk(&[(1, None)]);
        d.set(1, Change::Mute(false), Some(When::Frame(30)), 0);
        d.fill(0, 100, RATE, std::iter::empty());
        assert_eq!(d.0[&1].path.flat(), Some(1.0));
    }
}
