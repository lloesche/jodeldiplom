//! Transitions: the playing track and the next one heard together for a
//! stretch of bars, each stem of each on its own fade, the next one's beats
//! laid onto the playing one's, then the next one plays on alone.
//!
//! A [`TransitionPlan`] says where (the window, in the playing track's
//! seconds, and where the next track is at its start), which beats meet
//! (pairs of beats, one of each track), how each stem fades
//! ([`Envelope`]s over the window) and how the tempo is matched
//! ([`TempoMode`]). The engine plays it on a desk
//! ([`crate::Command::Transition`], [`crate::Player::on_desk`]) and reports
//! how it goes ([`TransitionState`]).

use crate::stems::{Stem, STEMS};

/// Seconds of the playing track before the window at which the next
/// track's output starts: its frame 0 is mixed there (silent until the
/// window), and from the hand-over on the output frames the player reports
/// are its, counted from there.
pub const NEXT_LEAD: f64 = 1.0;

/// A point of an [`Envelope`]: the gain at a share of the window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    /// Where, as a share of the window (0 its start, 1 its end).
    pub at: f64,
    /// The gain there, 0 to 1.
    pub gain: f64,
    /// The gain changes to this one at once (a cut) rather than fading
    /// from the point before.
    pub cut: bool,
}

/// A stem's gain over the window: points joined by equal-power fades (so
/// two stems crossing over keep the loudness) or cuts; before the first
/// point and after the last, their gains.
#[derive(Clone, Debug, PartialEq)]
pub struct Envelope {
    /// The points, by `at`.
    pub points: Vec<Point>,
}

impl Envelope {
    /// The same gain throughout.
    pub fn flat(gain: f64) -> Envelope {
        Envelope {
            points: vec![Point {
                at: 0.0,
                gain,
                cut: false,
            }],
        }
    }

    /// A fade from `from` at `start` to `to` at `end` (shares of the
    /// window).
    pub fn fade(from: f64, to: f64, start: f64, end: f64) -> Envelope {
        Envelope {
            points: vec![
                Point {
                    at: start,
                    gain: from,
                    cut: false,
                },
                Point {
                    at: end,
                    gain: to,
                    cut: false,
                },
            ],
        }
    }

    /// A cut from `from` to `to` at `at`.
    pub fn cut(from: f64, to: f64, at: f64) -> Envelope {
        Envelope {
            points: vec![
                Point {
                    at: 0.0,
                    gain: from,
                    cut: false,
                },
                Point {
                    at,
                    gain: to,
                    cut: true,
                },
            ],
        }
    }

    /// The gain at `at` (a share of the window).
    pub fn gain(&self, at: f64) -> f64 {
        let p = &self.points;
        let Some(first) = p.first() else {
            return 1.0;
        };
        if at <= first.at {
            return first.gain;
        }
        let i = p.partition_point(|q| q.at <= at);
        let Some(next) = p.get(i) else {
            return p[p.len() - 1].gain;
        };
        let prev = &p[i - 1];
        if next.cut || next.at <= prev.at {
            return prev.gain;
        }
        // Equal power, as the desk's: the squares move linearly.
        let u = (at - prev.at) / (next.at - prev.at);
        jodel_output::mixer::equal_power(prev.gain, next.gain, u)
    }
}

/// How the next track's tempo meets the playing one's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TempoMode {
    /// It plays faster or slower, its pitch with it.
    #[default]
    Varispeed,
    /// Its tempo changes and its pitch stays.
    Stretch,
}

/// Which track of a transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// The track playing when it starts.
    From,
    /// The next one, which plays on after it.
    Into,
}

/// A transition from the playing track into the next.
#[derive(Clone, Debug, PartialEq)]
pub struct TransitionPlan {
    /// The playing track's queue index.
    pub from: usize,
    /// The next track's (`from + 1`).
    pub into: usize,
    /// Seconds into the playing track where the window starts.
    pub start: f64,
    /// The window's length in the playing track's seconds (whole bars).
    pub length: f64,
    /// Seconds into the next track at the window's start.
    pub entry: f64,
    /// Beats that meet: a time of the playing track, a time of the next
    /// track, from the window's start on (the next track's beats past the
    /// playing track's end paired with its beats continued at its tempo).
    pub beats: Vec<(f64, f64)>,
    /// Each stem's gain over the window: the playing track's four, then
    /// the next track's, in [`Stem::ALL`] order.
    pub lanes: [Envelope; 2 * STEMS],
    /// How the tempi meet.
    pub tempo: TempoMode,
    /// Seconds after the window over which the next track goes back to
    /// its own tempo (what is left of it, with `glide`).
    pub settle: f64,
    /// The tempo glides across the window from the playing track's to the
    /// next one's: the playing track is sped up or slowed down as well
    /// (in `tempo`'s way), so the next track reaches its own tempo at the
    /// window's end. Otherwise the playing track keeps its own and the
    /// next one goes back to its own over `settle` after the window.
    pub glide: bool,
}

impl TransitionPlan {
    /// Each stem's gain of one side at `at` (a share of the window), in
    /// [`Stem::ALL`] order.
    pub fn gains(&self, side: Side, at: f64) -> [f64; STEMS] {
        let base = match side {
            Side::From => 0,
            Side::Into => STEMS,
        };
        std::array::from_fn(|s| self.lanes[base + s].gain(at))
    }

    /// The lane of `stem` on `side`.
    pub fn lane(&self, side: Side, stem: Stem) -> &Envelope {
        let s = Stem::ALL.iter().position(|&x| x == stem).unwrap_or(0);
        match side {
            Side::From => &self.lanes[s],
            Side::Into => &self.lanes[STEMS + s],
        }
    }

    /// The window's share at `seconds` of the playing track.
    pub fn share(&self, seconds: f64) -> f64 {
        if self.length <= 0.0 {
            return if seconds < self.start { 0.0 } else { 1.0 };
        }
        (seconds - self.start) / self.length
    }

    /// How the playing track's time goes along the output's through the
    /// window, with `map` (from [`TransitionPlan::time_map`]): with
    /// `glide`, its speed ends where the next track, locked to it, plays
    /// at its own there (the inverse of the map's speed over the window's
    /// last quarter, up to its last paired beat); otherwise none.
    pub fn glide_of(&self, map: &crate::warp::TimeMap) -> crate::warp::Glide {
        let end = if self.glide && self.beats.len() >= 2 && self.length > 0.0 {
            let last = self.beats[self.beats.len() - 1].0;
            let to = (self.start + self.length).min(last);
            let span = (self.length / 4.0).min(to - self.beats[0].0);
            let speed = if span > 0.0 {
                (map.at(to) - map.at(to - span)) / span
            } else {
                0.0
            };
            if speed > 0.0 {
                1.0 / speed
            } else {
                1.0 / map.typical_speed()
            }
        } else {
            1.0
        };
        crate::warp::Glide::new(self.start, self.length, end)
    }

    /// The map from the playing track's time to the next one's that the
    /// next track is read along: through the paired beats, or, with fewer
    /// than two, at its own speed from where the window meets it.
    pub fn time_map(&self) -> crate::warp::TimeMap {
        if self.beats.len() >= 2 {
            crate::warp::TimeMap::new(&self.beats, self.settle)
        } else {
            crate::warp::TimeMap::new(&[(self.start, self.entry)], self.settle)
        }
    }
}

/// How a transition goes.
#[derive(Clone, Debug, PartialEq)]
pub enum TransitionState {
    /// Taken: it starts when the playing track reaches the window.
    Armed,
    /// The window started to be heard at this output frame of the playing
    /// track's head.
    Started {
        /// The head ([`crate::TrackStart::head`]) whose output frames
        /// these are.
        head: usize,
        /// The output frame.
        output_frame: u64,
        /// That head's output frame the next track's head's frame 0 is
        /// linked to: from the hand-over on, the positions told count the
        /// next track's head's frames, these fewer than the playing
        /// head's.
        origin: u64,
    },
    /// The next track has taken over and is back at its own tempo.
    Finished,
    /// It will not happen (any more).
    Cancelled {
        /// Why, for the user.
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelopes_fade_at_equal_power_and_cut() {
        let down = Envelope::fade(1.0, 0.0, 0.25, 0.75);
        let up = Envelope::fade(0.0, 1.0, 0.25, 0.75);
        assert_eq!(down.gain(0.0), 1.0);
        assert_eq!(down.gain(1.0), 0.0);
        for k in 0..=10 {
            let at = 0.25 + 0.05 * f64::from(k);
            let power = down.gain(at).powi(2) + up.gain(at).powi(2);
            assert!((power - 1.0).abs() < 1e-12, "{at}: {power}");
        }
        let cut = Envelope::cut(1.0, 0.0, 0.5);
        assert_eq!(cut.gain(0.49), 1.0);
        assert_eq!(cut.gain(0.5), 0.0);
        assert_eq!(Envelope::flat(0.5).gain(0.3), 0.5);
    }
}
