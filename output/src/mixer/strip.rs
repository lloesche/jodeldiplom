//! Strips: the user's channels of the desk, as their handles see them. Each
//! input plugs into one; a strip may sit in another, its group, and the
//! gains along that path (each strip's level, muted or not, and the solo
//! gate) multiply the gain of every input on it. The mixer thread runs them
//! in [`super::strips`].

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::Duration;

use jodel_dsp::FULL_SCALE;

use super::{MixShared, Msg, SlotState, Start};

/// How long a strip's level, mute or solo takes to be heard: a linear ramp,
/// so it never clicks.
pub const STRIP_RAMP: Duration = Duration::from_millis(20);

/// The most channels metered; a device's channels past these are not.
pub const METER_CHANNELS: usize = 8;

/// The peak of each channel of `samples` (`channels` per frame), each
/// sample's size given by `magnitude`.
pub(super) fn channel_peaks<T: Copy>(
    samples: &[T],
    channels: usize,
    magnitude: impl Fn(T) -> f64,
) -> [f64; METER_CHANNELS] {
    let mut peaks = [0.0f64; METER_CHANNELS];
    for frame in samples.chunks_exact(channels.max(1)) {
        for (p, &s) in peaks.iter_mut().zip(frame) {
            *p = p.max(magnitude(s));
        }
    }
    peaks
}

/// Peaks per channel, as `f32` bits (linear, 1.0: full scale), raised by the
/// mixer thread and taken by [`super::Mixer::meters`]: the peak since the
/// last look. A non-negative `f32`'s bits order as the `u32` does, so
/// raising is one `fetch_max`.
#[derive(Default)]
pub(crate) struct Peaks([AtomicU32; METER_CHANNELS]);

impl Peaks {
    /// Raises each channel to `peaks` (in the mixer's samples).
    pub(crate) fn raise(&self, peaks: &[f64; METER_CHANNELS]) {
        for (a, &p) in self.0.iter().zip(peaks) {
            if p > 0.0 {
                a.fetch_max(((p / FULL_SCALE) as f32).to_bits(), Ordering::Relaxed);
            }
        }
    }

    /// The peaks of the first `channels` since the last look, starting the
    /// next.
    pub(crate) fn take(&self, channels: usize) -> Vec<f32> {
        self.0
            .iter()
            .map(|a| f32::from_bits(a.swap(0, Ordering::Relaxed)))
            .take(channels)
            .collect()
    }
}

/// A change of a strip's level or mute, on its way to the mixer thread (a
/// solo is always now: [`Msg::StripSolo`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Change {
    Level(f64),
    Mute(bool),
}

/// The bits of no level waiting (a NaN; a level set is never one).
const NO_LEVEL: u64 = u64::MAX;

/// What a strip's handles and the mixer thread share.
pub(crate) struct StripState {
    pub(crate) name: String,
    /// Its group.
    pub(crate) parent: Option<u64>,
    /// The settings in force: the level (`f64` bits), muted, soloed; and
    /// the changes waiting for their start: a level ([`NO_LEVEL`]: none),
    /// a mute (0: none, 1: off, 2: on). A handle shows what it sets at
    /// once; the mixer thread shows a setting as it has it (a change
    /// landed, one gone, one that never comes) only once it has taken the
    /// newest set of it, so it never shows an older one over a newer.
    pub(crate) level: AtomicU64,
    pub(crate) mute: AtomicBool,
    pub(crate) solo: AtomicBool,
    pending_level: AtomicU64,
    pending_mute: AtomicU8,
    /// The sets of the level, and of the mute, the handles have made.
    pub(crate) level_sets: AtomicU64,
    pub(crate) mute_sets: AtomicU64,
    /// Post-fader peaks of the strip's subtree summed: its inputs and its
    /// children's.
    pub(crate) peaks: Peaks,
}

impl StripState {
    pub(crate) fn new(name: &str, parent: Option<u64>) -> StripState {
        StripState {
            name: name.to_owned(),
            parent,
            level: AtomicU64::new(1f64.to_bits()),
            mute: AtomicBool::new(false),
            solo: AtomicBool::new(false),
            pending_level: AtomicU64::new(NO_LEVEL),
            pending_mute: AtomicU8::new(0),
            level_sets: AtomicU64::new(0),
            mute_sets: AtomicU64::new(0),
            peaks: Peaks::default(),
        }
    }

    pub(crate) fn level(&self) -> f64 {
        f64::from_bits(self.level.load(Ordering::Relaxed))
    }

    pub(crate) fn mute(&self) -> bool {
        self.mute.load(Ordering::Relaxed)
    }

    pub(crate) fn solo(&self) -> bool {
        self.solo.load(Ordering::Relaxed)
    }

    pub(crate) fn pending_level(&self) -> Option<f64> {
        let bits = self.pending_level.load(Ordering::Relaxed);
        (bits != NO_LEVEL).then(|| f64::from_bits(bits))
    }

    pub(crate) fn pending_mute(&self) -> Option<bool> {
        match self.pending_mute.load(Ordering::Relaxed) {
            0 => None,
            m => Some(m == 2),
        }
    }

    /// Shows the level in force (`None`: as it is) and the one waiting.
    pub(crate) fn show_level(&self, level: Option<f64>, pending: Option<f64>) {
        if let Some(level) = level {
            self.level.store(level.to_bits(), Ordering::Relaxed);
        }
        let pending = pending.map_or(NO_LEVEL, f64::to_bits);
        self.pending_level.store(pending, Ordering::Relaxed);
    }

    /// Shows the mute in force (`None`: as it is) and the one waiting.
    pub(crate) fn show_mute(&self, on: Option<bool>, pending: Option<bool>) {
        if let Some(on) = on {
            self.mute.store(on, Ordering::Relaxed);
        }
        let pending = pending.map_or(0, |on| 1 + u8::from(on));
        self.pending_mute.store(pending, Ordering::Relaxed);
    }
}

/// The inputs plugged into a strip, for a look at the desk (kept by the
/// handles; the mixer thread never touches it).
pub(crate) struct Entry {
    pub(crate) state: Arc<StripState>,
    pub(crate) inputs: BTreeMap<u64, Arc<SlotState>>,
}

/// What a strip's clones, its children and its inputs share.
pub(crate) struct StripHandle {
    pub(crate) id: u64,
    tx: Sender<Msg>,
    pub(crate) shared: Arc<MixShared>,
    state: Arc<StripState>,
    /// Its group, kept while it lives (so it goes after its children).
    _parent: Option<Strip>,
}

impl Drop for StripHandle {
    fn drop(&mut self) {
        self.shared.strips().remove(&self.id);
        let _ = self.tx.send(Msg::StripGone(self.id));
    }
}

/// A channel of the desk ([`crate::mixer::Mixer::strip`]): the user's gain
/// over the inputs plugged into it. It starts at 0 dB, unmuted, not soloed.
/// A strip made in another ([`crate::mixer::Mixer::strip_in`]) is a child
/// of that group: the group's level and mute act on it as a VCA does, on
/// top of its own. Clones are the same strip; it lives while a clone, a
/// child or an input on it does, so an input is never without its strip,
/// nor a child without its group.
#[derive(Clone)]
pub struct Strip {
    pub(crate) handle: Arc<StripHandle>,
}

impl Strip {
    pub(crate) fn new(
        id: u64,
        tx: Sender<Msg>,
        shared: Arc<MixShared>,
        name: &str,
        parent: Option<Strip>,
    ) -> Strip {
        let state = Arc::new(StripState::new(name, parent.as_ref().map(Strip::id)));
        shared.strips().insert(
            id,
            Entry {
                state: state.clone(),
                inputs: BTreeMap::new(),
            },
        );
        let _ = tx.send(Msg::Strip {
            id,
            state: state.clone(),
        });
        Strip {
            handle: Arc::new(StripHandle {
                id,
                tx,
                shared,
                state,
                _parent: parent,
            }),
        }
    }

    /// Which strip of its mixer it is (as [`super::StripView::id`]).
    pub fn id(&self) -> u64 {
        self.handle.id
    }

    /// Its name.
    pub fn name(&self) -> &str {
        &self.handle.state.name
    }

    /// The group it sits in, by id (`None`: at the top of the desk).
    pub fn parent(&self) -> Option<u64> {
        self.handle.state.parent
    }

    /// Sets the level, a linear gain (1: 0 dB; a level that is not a
    /// number, or below 0, is 0), heard over [`STRIP_RAMP`] from the next
    /// frame mixed.
    pub fn set_level(&self, gain: f64) {
        self.set_level_at(gain, Start::Now);
    }

    /// Sets the level as [`Strip::set_level`] does, from `start`: its
    /// ramp starts on that very frame. It takes the place of a change of
    /// the level still waiting for its start; a start that never comes
    /// (a time of an epoch gone) changes nothing. [`Start::FirstAudio`]
    /// is the first audio of an output opened after it on an input of
    /// this strip or of its children.
    pub fn set_level_at(&self, gain: f64, start: Start) {
        let gain = if gain.is_finite() { gain.max(0.0) } else { 0.0 };
        let s = &self.handle.state;
        let set = s.level_sets.fetch_add(1, Ordering::AcqRel) + 1;
        match start {
            Start::Now => s.show_level(Some(gain), None),
            _ => s.show_level(None, Some(gain)),
        }
        self.send(Change::Level(gain), start, set);
    }

    /// Mutes it (or not), over [`STRIP_RAMP`] from the next frame mixed.
    pub fn set_mute(&self, on: bool) {
        self.set_mute_at(on, Start::Now);
    }

    /// Mutes it (or not) from `start`, as [`Strip::set_level_at`] sets
    /// the level: in place of a mute still waiting, on that very frame.
    pub fn set_mute_at(&self, on: bool, start: Start) {
        let s = &self.handle.state;
        let set = s.mute_sets.fetch_add(1, Ordering::AcqRel) + 1;
        match start {
            Start::Now => s.show_mute(Some(on), None),
            _ => s.show_mute(None, Some(on)),
        }
        self.send(Change::Mute(on), start, set);
    }

    /// Solos it in place (or not), over [`STRIP_RAMP`] from the next frame
    /// mixed: while any strip is soloed, a strip is heard only when it or
    /// a group it sits in is. Soloing a child silences its siblings, its
    /// group's own inputs and every other strip (it is still heard through
    /// its group's level and mute); soloing a group keeps all its
    /// children.
    pub fn set_solo(&self, on: bool) {
        self.handle.state.solo.store(on, Ordering::Relaxed);
        let _ = self.handle.tx.send(Msg::StripSolo {
            id: self.handle.id,
            on,
        });
    }

    /// The level in force: one set now at once, one set for later once its
    /// frame is mixed.
    pub fn level(&self) -> f64 {
        self.handle.state.level()
    }

    /// Whether it is muted, in force as [`Strip::level`] is.
    pub fn mute(&self) -> bool {
        self.handle.state.mute()
    }

    /// Whether it is soloed.
    pub fn solo(&self) -> bool {
        self.handle.state.solo()
    }

    /// The level set for later and waiting for its start (`None`: none
    /// waits; one whose start never comes goes once the mixer has taken
    /// it). What the level is meant to be is this, else [`Strip::level`].
    pub fn pending_level(&self) -> Option<f64> {
        self.handle.state.pending_level()
    }

    /// The mute waiting for its start, as [`Strip::pending_level`]: what a
    /// toggle goes against is this, else [`Strip::mute`].
    pub fn pending_mute(&self) -> Option<bool> {
        self.handle.state.pending_mute()
    }

    fn send(&self, change: Change, start: Start, set: u64) {
        let _ = self.handle.tx.send(Msg::StripSet {
            id: self.handle.id,
            change,
            start,
            set,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peaks_rise_and_are_taken() {
        let peaks = Peaks::default();
        let size = |s: i32| f64::from(s.unsigned_abs());
        peaks.raise(&channel_peaks(
            &[4_194_304, -8_388_608, -1_000, 2_000],
            2,
            size,
        ));
        peaks.raise(&channel_peaks(&[0, 1], 2, size));
        assert_eq!(peaks.take(2), [0.5, 1.0]);
        assert_eq!(peaks.take(2), [0.0, 0.0]);
    }
}
