//! The beat grid at a moment of a track ([`Analysis::grid_at`]), and the
//! one rule a hit fired then lands on it by ([`Grid::land`], as [`When`]
//! says), so what is heard and what is seen land on the same frame.
//!
//! A grid is a straight line of beats through the analysis's beat at or
//! before the moment (the first, before it), at the local tempo there;
//! its bars as the downbeats around that beat have them (the bar it is in,
//! else the one before or after; [`BAR`] beats where there are none, and
//! counted back from the first downbeat before it); as sure as the
//! analysis's region there. Made anew as the track goes on, it follows the
//! tempo where it drifts. It is in seconds; [`Grid::on`] puts it on
//! another clock (a stream's frames), as exactly.

use crate::Analysis;

/// The tempo where there is no grid, beats a minute.
pub const FREE_TEMPO: f64 = 120.0;
/// The beats of a bar where the grid has no bars.
pub const BAR: u32 = 4;
/// How sure the grid must be (its region's confidence) for a hit to wait
/// for it: below, a hit lands at once.
pub const SURE: f32 = 0.5;

/// Where a hit lands ([`Grid::land`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum When {
    /// At once.
    Now,
    /// On the next beat.
    Beat,
    /// On a bar's first beat, with `length` of room after it before a
    /// sung line.
    Downbeat {
        /// How long it sounds (or shows), in the grid's units.
        length: f64,
    },
}

/// The beat grid about a moment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid {
    /// A beat's time (any beat of the grid).
    beat: f64,
    /// The time a beat lasts.
    beat_len: f64,
    /// Beats a minute (whatever the units: kept as it was heard).
    tempo: f64,
    /// Which beat of its bar that beat is (0: the bar's first).
    in_bar: u32,
    /// Beats a bar.
    bar: u32,
    /// How sure the analysis is there, 0 to 1.
    sure: f32,
}

impl Default for Grid {
    fn default() -> Grid {
        Grid::free()
    }
}

impl Analysis {
    /// The grid `t` seconds into the track (see the module).
    pub fn grid_at(&self, t: f64) -> Grid {
        let tempo = self
            .bpm_at(t)
            .or(self.summary.map(|s| s.bpm))
            .unwrap_or(FREE_TEMPO);
        let sure = self.region_at(t).map_or(0.0, |r| r.confidence as f32);
        if self.beats.is_empty() {
            return Grid::new(t, tempo, 0, BAR, sure);
        }
        let i = self.beat_at(t).unwrap_or(0);
        let downs = &self.downbeats;
        let d = downs.partition_point(|&j| j <= i);
        let length = |k: usize| (downs.get(k + 1)?.checked_sub(*downs.get(k)?)).filter(|&n| n > 0);
        let bar_of = |k: Option<usize>| k.and_then(length).unwrap_or(BAR as usize);
        let (in_bar, bar) = match d.checked_sub(1) {
            Some(k) => {
                let bar = length(k).unwrap_or_else(|| bar_of(k.checked_sub(1)));
                ((i - downs[k]) % bar, bar)
            }
            None => match downs.first() {
                Some(&first) => {
                    let bar = bar_of(Some(0));
                    ((bar - (first - i) % bar) % bar, bar)
                }
                None => (i % BAR as usize, BAR as usize),
            },
        };
        Grid::new(self.beats[i], tempo, in_bar as u32, bar as u32, sure)
    }
}

impl Grid {
    /// A grid through a beat at `beat` seconds, at `tempo` beats a minute,
    /// that beat the `in_bar`-th of a bar of `bar` beats, as sure as
    /// `sure`. What makes no sense is taken as no grid: a tempo not above
    /// 0 (or not a number) the free tempo, a bar of no beats [`BAR`]'s, a
    /// beat not a number at 0, a sureness not a number unsure.
    pub(crate) fn new(beat: f64, tempo: f64, in_bar: u32, bar: u32, sure: f32) -> Grid {
        let tempo = if tempo.is_finite() && tempo > 0.0 {
            tempo
        } else {
            FREE_TEMPO
        };
        let bar = if bar == 0 { BAR } else { bar };
        Grid {
            beat: if beat.is_finite() { beat } else { 0.0 },
            beat_len: 60.0 / tempo,
            tempo,
            in_bar: in_bar % bar,
            bar,
            sure: if sure.is_nan() {
                0.0
            } else {
                sure.clamp(0.0, 1.0)
            },
        }
    }

    /// No grid: the free tempo from 0, unsure.
    pub fn free() -> Grid {
        Grid::new(0.0, FREE_TEMPO, 0, BAR, 0.0)
    }

    /// A steady grid at `tempo` beats a minute in bars of `bar` beats, a
    /// bar's first beat at 0, sure (a tempo chosen by hand: the Synth
    /// tab's).
    pub fn steady(tempo: f64, bar: u32) -> Grid {
        Grid::new(0.0, tempo, 0, bar, 1.0)
    }

    /// The same grid on another clock, where `t` seconds is `at` and a
    /// second is `per_second` of its units (a stream at its rate).
    pub fn on(&self, t: f64, at: f64, per_second: f64) -> Grid {
        let per_second = if per_second.is_finite() && per_second > 0.0 {
            per_second
        } else {
            1.0
        };
        Grid {
            beat: at + (self.beat - t) * per_second,
            beat_len: self.beat_len * per_second,
            ..*self
        }
    }

    /// How long a beat lasts.
    pub fn beat_len(&self) -> f64 {
        self.beat_len
    }

    /// Beats a minute.
    pub fn tempo(&self) -> f64 {
        self.tempo
    }

    /// Beats a bar.
    pub fn bar(&self) -> u32 {
        self.bar
    }

    /// Whether it is sure enough to wait for ([`SURE`]).
    pub fn is_sure(&self) -> bool {
        self.sure >= SURE
    }

    /// The first of `per_beat` steps a beat (1 beats, 4 sixteenths) at or
    /// after `at`, and its number from the grid's beat.
    pub fn next_step(&self, at: f64, per_beat: u32) -> (f64, i64) {
        let step = self.beat_len / f64::from(per_beat.max(1));
        let k = ((at - self.beat) / step - 1e-9).ceil();
        ((self.beat + k * step).max(at), k as i64)
    }

    /// The first bar's first beat at or after `at`.
    pub fn next_downbeat(&self, at: f64) -> f64 {
        let (_, k) = self.next_step(at, 1);
        let bar = i64::from(self.bar);
        let k = k + (bar - (i64::from(self.in_bar) + k).rem_euclid(bar)) % bar;
        (self.beat + k as f64 * self.beat_len).max(at)
    }

    /// Where a hit fired at `at` lands, `sung` the spans sung from about
    /// then on (start, end), as `when` says:
    ///
    /// - [`When::Now`]: at `at`.
    /// - [`When::Beat`]: on the next beat; at `at` if that is more than
    ///   half a beat away, or the grid is unsure. (A sound plays its
    ///   pickup at `at` and its hit there.)
    /// - [`When::Downbeat`]: on the first bar's first beat from `at` with
    ///   `length` of room before any sung span, so it never lands over the
    ///   singing (past the last span known there is room); at `at` where
    ///   the grid is unsure.
    pub fn land(&self, at: f64, when: When, sung: &[(f64, f64)]) -> f64 {
        if !self.is_sure() {
            return at;
        }
        match when {
            When::Now => at,
            When::Beat => {
                let (next, _) = self.next_step(at, 1);
                if next - at > 0.5 * self.beat_len {
                    at
                } else {
                    next
                }
            }
            When::Downbeat { length } => {
                let mut d = self.next_downbeat(at);
                loop {
                    let free = room(d, length, sung);
                    if free <= d {
                        return d;
                    }
                    d = self.next_downbeat(free);
                }
            }
        }
    }
}

/// The first time from `from` with `length` after it in no span of `sung`.
fn room(from: f64, length: f64, sung: &[(f64, f64)]) -> f64 {
    let mut f = from;
    while let Some(&(_, end)) = sung
        .iter()
        .find(|&&(start, end)| start < f + length && end > f)
    {
        f = end;
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Region;

    /// Beats every half second from 1 s, a bar every `bar` beats from beat
    /// `first`, sure from `sure` seconds on.
    fn analysis(bar: usize, first: usize, sure: f64) -> Analysis {
        let beats: Vec<f64> = (0..64).map(|k| 1.0 + 0.5 * k as f64).collect();
        Analysis {
            downbeats: (first..64).step_by(bar).collect(),
            local_bpm: vec![120.0; beats.len()],
            regions: vec![
                Region {
                    start: 0.0,
                    end: sure,
                    confidence: 0.1,
                },
                Region {
                    start: sure,
                    end: 40.0,
                    confidence: 0.9,
                },
            ],
            beats,
            ..Analysis::default()
        }
    }

    #[test]
    fn a_hit_lands_on_the_next_beat_if_at_most_half_a_beat_away() {
        let g = analysis(4, 0, 0.0).grid_at(5.25);
        assert_eq!(g.land(5.25, When::Beat, &[]), 5.5, "a quarter beat: waits");
        assert_eq!(g.land(5.25, When::Now, &[]), 5.25);
        assert_eq!(g.land(5.24, When::Beat, &[]), 5.24, "more than half: now");
        assert_eq!(g.land(5.5, When::Beat, &[]), 5.5, "on the beat");
        assert_eq!(g.tempo(), 120.0);
        // Before the first beat, the grid runs back from it.
        let early = analysis(4, 0, 0.0).grid_at(0.1);
        assert_eq!(early.land(0.4, When::Beat, &[]), 0.5);
    }

    #[test]
    fn an_unsure_grid_lands_everything_now() {
        let g = analysis(4, 0, 10.0).grid_at(5.25);
        assert!(!g.is_sure());
        assert_eq!(g.land(5.25, When::Beat, &[]), 5.25);
        assert_eq!(g.land(5.25, When::Downbeat { length: 1.0 }, &[]), 5.25);
    }

    #[test]
    fn a_downbeat_in_its_own_bars_with_room_before_the_singing() {
        // 4/4 from beat 0 (1 s): beat 8 (5 s) is a bar's first.
        let g = analysis(4, 0, 0.0).grid_at(5.25);
        let bar = |length| When::Downbeat { length };
        assert_eq!(g.land(5.25, bar(1.0), &[]), 7.0);
        // Sung over that one: the bar after it in the gap; a gap too short
        // for it: on; sung on and on: after the last span.
        assert_eq!(g.land(5.25, bar(1.0), &[(6.5, 8.0)]), 9.0);
        assert_eq!(g.land(5.25, bar(1.5), &[(6.5, 8.0), (10.0, 12.0)]), 13.0);
        assert_eq!(g.land(5.25, bar(1.0), &[(5.0, 20.0), (20.0, 21.0)]), 21.0);
        // 3/4 from beat 1 (1.5 s): beat 8 is the second of its bar.
        let waltz = analysis(3, 1, 0.0).grid_at(5.25);
        assert_eq!(waltz.bar(), 3);
        assert_eq!(waltz.land(5.25, bar(0.5), &[]), 6.0);
        // Before the first downbeat (beat 0 of the waltz): its bar counted
        // back from there, so beat 1 is the first.
        let early = analysis(3, 1, 0.0).grid_at(1.1);
        assert_eq!(early.land(1.1, bar(0.5), &[]), 1.5);
        // In 5/4 bars at the end (no next downbeat): the bar before's.
        let mut odd = analysis(5, 0, 0.0);
        odd.downbeats = vec![0, 5, 10];
        let late = odd.grid_at(7.0);
        assert_eq!(late.bar(), 5);
        assert_eq!(late.land(7.0, bar(0.5), &[]), 8.5);
    }

    #[test]
    fn on_another_clock_it_lands_on_the_same_moments() {
        let g = analysis(4, 0, 0.0).grid_at(5.25);
        // 48 000 frames a second, 5.25 s at frame 100 000.
        let f = g.on(5.25, 100_000.0, 48_000.0);
        assert_eq!(f.beat_len(), 24_000.0);
        assert_eq!(f.tempo(), 120.0);
        assert_eq!(f.land(100_000.0, When::Beat, &[]), 112_000.0);
        let sung = [(190_000.0, 200_000.0)];
        let d = f.land(100_000.0, When::Downbeat { length: 48_000.0 }, &sung);
        assert_eq!(d, 100_000.0 + 3.75 * 48_000.0);
    }

    #[test]
    fn a_grid_that_makes_no_sense_is_the_free_one() {
        for tempo in [0.0, -3.0, f64::NAN, f64::INFINITY] {
            let g = Grid::new(f64::NAN, tempo, 7, 0, f32::NAN);
            assert_eq!(
                (g.beat_len(), g.bar(), g.is_sure()),
                (0.5, 4, false),
                "{tempo}"
            );
        }
        let none = Analysis::default().grid_at(3.0);
        assert_eq!((none.beat_len(), none.is_sure()), (0.5, false));
        assert_eq!(Grid::free().land(1.3, When::Beat, &[]), 1.3);
    }
}
