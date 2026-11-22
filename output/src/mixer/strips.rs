//! The strips as the mixer thread runs them. Each has two gains on the
//! clock: `heard`, what its own inputs are heard at (its level, 0 while
//! muted, times its solo gate: 1 while nothing is soloed, or it or a group
//! it sits in is; else 0), and `vca`, what it puts on its children (its
//! level, 0 while muted). Each is one ramp, so a mute and a solo set
//! together never multiply into a bump. An input is heard at its strip's
//! path: the strip's `heard` times the `vca` of every group above it; a
//! strip at the top of the desk is heard at its `heard` alone. A change
//! set for a frame waits for it and starts both ramps on that very frame,
//! inside the chunk being filled. A strip meters its subtree: its inputs
//! and its children's, summed after their gains.

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::envelope::Envelope;
use super::mix::Gains;
use super::ramp::{Gain, Waiting, When};
use super::strip::{channel_peaks, Change, StripState, STRIP_RAMP};
use super::Curve;

/// One of a strip's gains: on the clock, what the settings aim it at, and
/// this chunk's gains.
struct Factor {
    gain: Gain,
    target: f64,
    env: Envelope,
}

impl Factor {
    fn new(at: f64) -> Factor {
        Factor {
            gain: Gain::new(at),
            target: at,
            env: Envelope::new(at),
        }
    }

    /// Ramps to `to` over [`STRIP_RAMP`], from the next frame filled;
    /// returns whether that is a change.
    fn aim(&mut self, to: f64) -> bool {
        let changed = to != self.target;
        if changed {
            self.target = to;
            self.gain.set(Waiting {
                when: When::Now,
                to,
                over: STRIP_RAMP,
                curve: Curve::Linear,
            });
        }
        changed
    }

    /// The gains of frames `first..first + n` at `rate`.
    fn fill(&mut self, first: u64, n: usize, rate: u32) {
        let start = (self.gain.next() == Some(When::Now)).then_some(first);
        self.env.fill(&mut self.gain, first, n, start, rate);
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
    /// A change of the level, and one of the mute, waiting for its start.
    next_level: Option<(When, f64)>,
    next_mute: Option<(When, bool)>,
    /// The newest set of the level, and of the mute, taken.
    level_set: u64,
    mute_set: u64,
    heard: Factor,
    vca: Factor,
    /// This chunk's gains along its path: what its inputs are heard at.
    path: Envelope,
    /// This chunk: the newest output that supplied audio on an input of
    /// its subtree.
    audio: Option<u64>,
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
        if self.vca.aim(own) || heard {
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
        let audio = self.audio;
        let due = |w: Option<When>, at: u64| w.and_then(|w| w.due(at, end, audio));
        let (mut at, mut cut) = (first, false);
        loop {
            let level = due(self.next_level.map(|c| c.0), at);
            let mute = due(self.next_mute.map(|c| c.0), at);
            let Some(f) = level.into_iter().chain(mute).min() else {
                break;
            };
            if !cut {
                self.heard.env.cut(n);
                self.vca.env.cut(n);
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
        } else {
            self.heard.fill(first, n, rate);
            self.vca.fill(first, n, rate);
        }
    }

    /// Both gains for device frames `from..to` of the chunk from `first`.
    fn fill_part(&mut self, first: u64, from: u64, to: u64, rate: u32) {
        let (from, to) = ((from - first) as usize, (to - first) as usize);
        for Factor { gain, env, .. } in [&mut self.heard, &mut self.vca] {
            env.fill_part(gain, first, from, to, rate);
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
            s.aim(0);
            next = self.after(Some(id));
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
            next_level: None,
            next_mute: None,
            level_set: 0,
            mute_set: 0,
            heard: Factor::new(gate),
            vca: Factor::new(1.0),
            path: Envelope::new(gate),
            audio: None,
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
            s.heard.gain.settle();
            s.vca.gain.settle();
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
    /// strip of its input, and its id.
    pub(super) fn fill(
        &mut self,
        first: u64,
        n: usize,
        rate: u32,
        audio: impl Iterator<Item = (u64, u64)>,
    ) {
        for s in self.0.values_mut() {
            s.audio = None;
        }
        for (strip, output) in audio {
            self.each_up(strip, |s| s.audio = s.audio.max(Some(output)));
        }
        for s in self.0.values_mut() {
            s.fill(first, n, rate);
        }
        let mut next = self.after(None);
        while let Some(id) = next {
            let mut path = std::mem::take(&mut self.0.get_mut(&id).expect("listed").path);
            let s = &self.0[&id];
            let groups = s.ancestors.iter().filter_map(|a| self.0.get(a));
            groups.fold(path.assign(&s.heard.env), |p, g| p.times(&g.vca.env));
            self.0.get_mut(&id).expect("listed").path = path;
            next = self.after(Some(id));
        }
    }

    /// How strip `id` stands for a lane's switch: see [`Look`].
    pub(super) fn look(&self, id: u64) -> Look {
        let s = self.0.get(&id).expect("a lane's strip lives");
        Look {
            own: s.own_target(),
            gate: s.gate,
            changed: s.changed,
            moving: s.heard.gain.moving() || s.vca.gain.moving(),
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
            s.heard.gain.settle();
            s.vca.gain.settle();
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
        assert_eq!(d.0[&6].heard.target, 0.0);
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
        d.fill(0, 10, RATE, [(2, 10), (3, 11)].into_iter());
        assert!(d.0[&1].next_mute.is_some());
        // A newer one on its child.
        d.fill(10, 10, RATE, [(2, 12)].into_iter());
        assert!(d.0[&1].next_mute.is_none() && d.0[&1].mute);
    }

    #[test]
    fn a_change_to_where_it_is_keeps_the_path_flat() {
        let mut d = desk(&[(1, None)]);
        d.set(1, Change::Mute(false), Some(When::Frame(30)), 0);
        d.fill(0, 100, RATE, std::iter::empty());
        assert_eq!(d.0[&1].path.flat(), Some(1.0));
    }
}
