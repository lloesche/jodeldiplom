//! The stems a transition mixes inside the engine: each stem's gain as a
//! mixer plays them ([`crate::Command::StemMask`]), each change of them at
//! a time of one of its tracks (a bar's first beat).

use super::transition::seconds_written;
use super::Engine;
use crate::stems::{Stem, LANES, STEMS};
use crate::transition::Side;

/// Gains in [`LANES`]'s order put in [`Stem::ALL`]'s, as a transition's
/// fades are.
fn stem_order(lanes: [f64; STEMS]) -> [f64; STEMS] {
    Stem::ALL.map(|s| {
        let k = LANES.iter().position(|&l| l == s);
        lanes[k.expect("every stem has a lane")]
    })
}

/// The stems a transition's tracks play, as a mixer plays them
/// ([`crate::Command::StemMask`]): `base`, then each change from its time of
/// one of the two tracks, in [`Stem::ALL`]'s order.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct StemsMask {
    base: [f64; STEMS],
    changes: Vec<(Side, f64, [f64; STEMS])>,
}

impl StemsMask {
    /// The gains in A at its time `a` (a change at a time of B's is in A
    /// already).
    pub(super) fn for_a(&self, a: f64) -> [f64; STEMS] {
        let mut g = self.base;
        for &(side, at, gains) in &self.changes {
            if side == Side::Into || a >= at {
                g = gains;
            }
        }
        g
    }

    /// The gains in B at its time `b`, which is A's time `a()`.
    pub(super) fn for_b(&self, b: f64, a: impl Fn() -> f64) -> [f64; STEMS] {
        let mut g = self.base;
        for &(side, at, gains) in &self.changes {
            let reached = match side {
                Side::Into => b >= at,
                Side::From => a() >= at,
            };
            if reached {
                g = gains;
            }
        }
        g
    }
}

/// A change of the mixer's stem gains at a time of a track
/// ([`crate::Command::StemMask`] with a time), in [`LANES`]'s order.
#[derive(Clone, Copy, Debug)]
pub(super) struct MaskChange {
    /// The track.
    index: usize,
    /// Seconds into it.
    at: f64,
    /// The gains from then.
    gains: [f64; STEMS],
}

/// The mixer's stem gains as the engine keeps them: in force now (before
/// the first change to come), and the changes to come, in order of time.
#[derive(Clone, Debug)]
pub(super) struct StemGains {
    base: [f64; STEMS],
    changes: Vec<MaskChange>,
}

impl Default for StemGains {
    fn default() -> Self {
        StemGains {
            base: [1.0; STEMS],
            changes: Vec::new(),
        }
    }
}

impl StemGains {
    /// The changes to come follow their tracks through the queue (`keep`:
    /// each track's new index).
    pub(super) fn reindex(&mut self, keep: &dyn Fn(usize) -> usize) {
        for c in &mut self.changes {
            c.index = keep(c.index);
        }
    }
}

impl Engine {
    /// Seconds into the playing track at the frame written next, and its
    /// index: what a change at a time of it is still ahead of.
    fn written_time(&self) -> Option<(usize, f64)> {
        let track = self.playing.as_ref()?;
        let written = self.out.as_ref().map_or(0, |o| o.output.written());
        Some((track.index, seconds_written(track, written)))
    }

    /// [`crate::Command::StemMask`] (in [`LANES`]'s order): at once, or a
    /// change at `at` seconds into the playing track, after those to come
    /// before it (the ones from then on give way to it); a transition under
    /// way takes them into its fades, at their times. Another mix outside a
    /// transition has the next one try stems that failed again.
    pub(super) fn set_stem_mask(&mut self, gains: [f64; STEMS], at: Option<f64>) {
        let gains = gains.map(|g| g.max(0.0));
        self.expire_mask_changes();
        let index = self.playing.as_ref().map(|t| t.index);
        let m = &mut self.stem_gains;
        let changed = match (at, index) {
            (Some(at), Some(index)) => {
                m.changes.retain(|c| c.index == index && c.at < at);
                let last = m.changes.last().map_or(m.base, |c| c.gains);
                let changed = last != gains;
                if changed {
                    m.changes.push(MaskChange { index, at, gains });
                }
                changed
            }
            _ => {
                let changed = m.base != gains || !m.changes.is_empty();
                *m = StemGains {
                    base: gains,
                    changes: Vec::new(),
                };
                changed
            }
        };
        if changed && !self.transition_running() {
            self.stems_failed = false;
        }
        self.retune_transition();
    }

    /// Changes whose time has been written (or whose track is no longer
    /// played) are in force: they go into the base.
    fn expire_mask_changes(&mut self) {
        let now = self.written_time();
        let m = &mut self.stem_gains;
        while let Some(c) = m.changes.first().copied() {
            let ahead = now.is_some_and(|(index, t)| c.index == index && c.at > t);
            if ahead {
                break;
            }
            m.base = c.gains;
            m.changes.remove(0);
        }
    }

    /// Whether the stems are heard as they are, now and with every change
    /// still to come: every gain one.
    pub(super) fn mask_is_unity(&self) -> bool {
        let now = self.written_time();
        let m = &self.stem_gains;
        let unity = |g: &[f64; STEMS]| g.iter().all(|&g| g == 1.0);
        let ahead = |c: &&MaskChange| now.is_some_and(|(i, t)| c.index == i && c.at > t);
        let latest = m.changes.iter().rev().find(|c| !ahead(c));
        unity(&latest.map_or(m.base, |c| c.gains))
            && m.changes.iter().filter(ahead).all(|c| unity(&c.gains))
    }

    /// The stems a transition's tracks play, as a mixer plays them (changes
    /// still to come at times of its tracks, from then); a whole track
    /// ([`super::track::Track::whole`]) takes none.
    pub(super) fn transition_mask(&self) -> StemsMask {
        let gains = stem_order;
        let side = |index: usize| {
            let (from, into) = self.transition.as_ref()?.tracks();
            if index == from {
                Some(Side::From)
            } else if index == into {
                Some(Side::Into)
            } else {
                None
            }
        };
        let m = &self.stem_gains;
        let mut mask = StemsMask {
            base: gains(m.base),
            changes: Vec::new(),
        };
        for c in &m.changes {
            match side(c.index) {
                Some(side) => mask.changes.push((side, c.at, gains(c.gains))),
                // A change of another track's time is in force here.
                None => mask.base = gains(c.gains),
            }
        }
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mixers_changes_land_each_at_its_time_on_either_track() {
        let (one, two, three) = ([1.0; STEMS], [0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
        let mask = StemsMask {
            base: one,
            changes: vec![(Side::From, 6.0, two), (Side::From, 8.0, three)],
        };
        assert_eq!(mask.for_a(5.9), one);
        assert_eq!(mask.for_a(6.0), two);
        assert_eq!(mask.for_a(9.0), three);
        // B along A's time.
        assert_eq!(mask.for_b(100.0, || 7.0), two);
        // One at a time of B's: in A already.
        let mask = StemsMask {
            base: one,
            changes: vec![(Side::Into, 2.0, two)],
        };
        assert_eq!(mask.for_a(0.0), two);
        assert_eq!(mask.for_b(1.0, || 0.0), one);
        assert_eq!(mask.for_b(2.0, || 0.0), two);
        // The lanes' order into the stems'.
        let lanes = [0.1, 0.2, 0.3, 0.4];
        let stems = stem_order(lanes);
        for (k, s) in Stem::ALL.iter().enumerate() {
            let lane = LANES.iter().position(|l| l == s).unwrap();
            assert_eq!(stems[k], lanes[lane]);
        }
    }

    /// The queue reordered: the changes still to come follow their tracks.
    #[test]
    fn changes_to_come_follow_their_tracks_through_the_queue() {
        let change = |index| MaskChange {
            index,
            at: 4.0,
            gains: [0.0, 1.0, 1.0, 1.0],
        };
        let mut gains = StemGains {
            base: [1.0; STEMS],
            changes: vec![change(2), change(3)],
        };
        // Track 0 moved after 3: 1 to 3 move up one.
        gains.reindex(&|i| match i {
            0 => 3,
            i => i - 1,
        });
        let indices: Vec<usize> = gains.changes.iter().map(|c| c.index).collect();
        assert_eq!(indices, [1, 2]);
        assert_eq!(gains.base, [1.0; STEMS]);
    }
}
