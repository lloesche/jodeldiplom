//! Strips: the user's channels of the desk. Each input plugs into one; the
//! strip's gain (its level, muted or not, soloed or not) multiplies the
//! gain of every input on it, and it is metered after that gain, its inputs
//! summed.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::Duration;

use super::mix::Gains;
use super::ramp::{Gain, Waiting, When};
use super::{Curve, MixShared, Msg, SlotState};

/// How long a strip's level, mute or solo takes to be heard: a linear ramp,
/// so it never clicks.
pub const STRIP_RAMP: Duration = Duration::from_millis(20);

/// The most channels metered; a device's channels past these are not.
pub const METER_CHANNELS: usize = 8;

/// Digital full scale of the 24-bit samples the mixer works in (the device's
/// full scale, whatever its bits).
const FULL_SCALE: f64 = 8_388_608.0;

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

/// What a strip's handles and the mixer thread share.
pub(crate) struct StripState {
    pub(crate) name: String,
    /// What the user set: the level (`f64` bits), muted, soloed. The mixer
    /// thread reads them when told they changed.
    pub(crate) level: AtomicU64,
    pub(crate) mute: AtomicBool,
    pub(crate) solo: AtomicBool,
    /// Post-fader peaks of the strip's inputs summed.
    pub(crate) peaks: Peaks,
}

impl StripState {
    pub(crate) fn new(name: &str) -> StripState {
        StripState {
            name: name.to_owned(),
            level: AtomicU64::new(1f64.to_bits()),
            mute: AtomicBool::new(false),
            solo: AtomicBool::new(false),
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
}

/// The inputs plugged into a strip, for a look at the desk (kept by the
/// handles; the mixer thread never touches it).
pub(crate) struct Entry {
    pub(crate) state: Arc<StripState>,
    pub(crate) inputs: BTreeMap<u64, Arc<SlotState>>,
}

/// What a strip's clones and its inputs share.
pub(crate) struct StripHandle {
    pub(crate) id: u64,
    tx: Sender<Msg>,
    pub(crate) shared: Arc<MixShared>,
    state: Arc<StripState>,
}

impl Drop for StripHandle {
    fn drop(&mut self) {
        self.shared.strips().remove(&self.id);
        let _ = self.tx.send(Msg::StripGone(self.id));
    }
}

/// A channel of the desk ([`crate::mixer::Mixer::strip`]): the user's gain
/// over the inputs plugged into it. It starts at 0 dB, unmuted, not soloed.
/// Clones are the same strip; it lives while a clone or an input on it
/// does (an input holds its strip), so an input is never without one.
#[derive(Clone)]
pub struct Strip {
    pub(crate) handle: Arc<StripHandle>,
}

impl Strip {
    pub(crate) fn new(id: u64, tx: Sender<Msg>, shared: Arc<MixShared>, name: &str) -> Strip {
        let state = Arc::new(StripState::new(name));
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

    /// Sets the level, a linear gain (1: 0 dB; a level that is not a
    /// number, or below 0, is 0), heard over [`STRIP_RAMP`].
    pub fn set_level(&self, gain: f64) {
        let gain = if gain.is_finite() { gain.max(0.0) } else { 0.0 };
        let s = &self.handle.state;
        s.level.store(gain.to_bits(), Ordering::Relaxed);
        self.changed();
    }

    /// Mutes it (or not), over [`STRIP_RAMP`].
    pub fn set_mute(&self, on: bool) {
        self.handle.state.mute.store(on, Ordering::Relaxed);
        self.changed();
    }

    /// Solos it in place (or not), over [`STRIP_RAMP`]: while any strip is
    /// soloed, every strip that is not is silent.
    pub fn set_solo(&self, on: bool) {
        self.handle.state.solo.store(on, Ordering::Relaxed);
        self.changed();
    }

    /// The level set.
    pub fn level(&self) -> f64 {
        self.handle.state.level()
    }

    /// Whether it is muted.
    pub fn mute(&self) -> bool {
        self.handle.state.mute()
    }

    /// Whether it is soloed.
    pub fn solo(&self) -> bool {
        self.handle.state.solo()
    }

    /// The mixer reads the settings again (the latest, whoever set them).
    fn changed(&self) {
        let _ = self.handle.tx.send(Msg::StripChanged);
    }
}

/// A strip as the mixer thread runs it.
struct StripRun {
    state: Arc<StripState>,
    /// What its settings make its gain, and the gain on its way there.
    target: f64,
    gain: Gain,
    /// This chunk's gains, or the one for all of it.
    gains: Vec<f64>,
    flat: Option<f64>,
    /// Its inputs summed, for the meter.
    sum: Vec<f64>,
}

impl StripRun {
    /// The gain its settings give it, with `any_solo` (a strip soloed).
    fn target(&self, any_solo: bool) -> f64 {
        let s = &self.state;
        if s.mute() || (any_solo && !s.solo()) {
            0.0
        } else {
            s.level()
        }
    }

    /// Ramps the gain to what the settings give now.
    fn aim(&mut self, any_solo: bool) {
        let to = self.target(any_solo);
        if to != self.target {
            self.target = to;
            self.gain.set(Waiting {
                when: When::Now,
                to,
                over: STRIP_RAMP,
                curve: Curve::Linear,
            });
        }
    }
}

/// The strips, as the mixer thread runs them.
#[derive(Default)]
pub(super) struct Strips(BTreeMap<u64, StripRun>);

impl Strips {
    fn any_solo(&self) -> bool {
        self.0.values().any(|s| s.state.solo())
    }

    /// Strip `id` made: at what its settings give it, at once.
    pub(super) fn add(&mut self, id: u64, state: Arc<StripState>) {
        let mut strip = StripRun {
            state,
            target: 0.0,
            gain: Gain::new(0.0),
            gains: Vec::new(),
            flat: None,
            sum: Vec::new(),
        };
        strip.target = strip.target(self.any_solo());
        strip.gain = Gain::new(strip.target);
        self.0.insert(id, strip);
    }

    /// A strip's settings changed: every strip ramps to what they give now
    /// (a solo silences the others).
    pub(super) fn changed(&mut self) {
        let any_solo = self.any_solo();
        for s in self.0.values_mut() {
            s.aim(any_solo);
        }
    }

    /// Strip `id` gone (its solo with it).
    pub(super) fn remove(&mut self, id: u64) {
        self.0.remove(&id);
        self.changed();
    }

    /// The device opened again: ramps under way end at once.
    pub(super) fn new_epoch(&mut self) {
        for s in self.0.values_mut() {
            s.gain.new_epoch();
        }
    }

    /// Their gains for frames `first..first + n` at `rate`.
    pub(super) fn fill(&mut self, first: u64, n: usize, rate: u32) {
        for s in self.0.values_mut() {
            let start = (s.gain.next() == Some(When::Now)).then_some(first);
            s.gains.resize(n, 0.0);
            s.flat = s.gain.fill(first, start, rate, &mut s.gains);
        }
    }

    /// An input's gains on strip `id` (`flat`, or one per frame in
    /// `gains`) through the strip's: the gain it is heard at, flat when
    /// both are.
    pub(super) fn through(&self, id: u64, flat: Option<f64>, gains: &mut [f64]) -> Option<f64> {
        let strip = self.0.get(&id).expect("an input's strip lives");
        match (flat, strip.flat) {
            (Some(a), Some(b)) => Some(a * b),
            (Some(a), None) => {
                for (g, s) in gains.iter_mut().zip(&strip.gains) {
                    *g = a * s;
                }
                None
            }
            (None, Some(b)) => {
                for g in gains.iter_mut() {
                    *g *= b;
                }
                None
            }
            (None, None) => {
                for (g, s) in gains.iter_mut().zip(&strip.gains) {
                    *g *= s;
                }
                None
            }
        }
    }

    /// Meters each strip: the heard inputs on it (`inputs`: strip, samples
    /// `channels` per frame, the gains they are heard at) summed, then the
    /// peak of each channel.
    pub(super) fn meter<'a>(
        &mut self,
        channels: usize,
        inputs: impl Iterator<Item = (u64, &'a [i32], Gains<'a>)> + Clone,
    ) {
        for (&id, strip) in self.0.iter_mut() {
            let mut any = false;
            for (_, samples, gain) in inputs.clone().filter(|i| i.0 == id) {
                if !any {
                    strip.sum.clear();
                    strip.sum.resize(samples.len(), 0.0);
                    any = true;
                }
                gain.accumulate(&mut strip.sum, samples, channels);
            }
            if any {
                let peaks = channel_peaks(&strip.sum, channels, f64::abs);
                strip.state.peaks.raise(&peaks);
            }
        }
    }

    /// Strips none of whose inputs will be heard for now (`busy`: the
    /// strips of those that may be) settle: a ramp ends at once.
    pub(super) fn settle_idle(&mut self, busy: impl Iterator<Item = u64> + Clone) {
        for (&id, strip) in self.0.iter_mut() {
            if !busy.clone().any(|b| b == id) {
                strip.gain.settle();
            }
        }
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

    #[test]
    fn mute_and_solo_silence() {
        let state = Arc::new(StripState::new("a"));
        let mut a = Strips::default();
        a.add(1, state.clone());
        state.level.store(0.5f64.to_bits(), Ordering::Relaxed);
        a.changed();
        let target = |s: &Strips, id: u64| s.0[&id].target;
        assert_eq!(target(&a, 1), 0.5);
        state.mute.store(true, Ordering::Relaxed);
        state.solo.store(true, Ordering::Relaxed);
        a.changed();
        assert_eq!(target(&a, 1), 0.0);
        // Another strip soloed silences this one; its own solo brings it
        // back.
        let mut b = Strips::default();
        let (c, other) = (
            Arc::new(StripState::new("c")),
            Arc::new(StripState::new("d")),
        );
        other.solo.store(true, Ordering::Relaxed);
        b.add(1, c.clone());
        b.add(2, other);
        b.changed();
        assert_eq!((target(&b, 1), target(&b, 2)), (0.0, 1.0));
        c.solo.store(true, Ordering::Relaxed);
        b.changed();
        assert_eq!(target(&b, 1), 1.0);
    }
}
