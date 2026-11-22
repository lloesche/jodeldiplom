//! When the mixer mixes the next period: once every running output holds
//! one. On a device in real time an output short of one (lagging) is
//! waited for while the device's ring ([`Output::queued`]: not what the
//! device has taken into a buffer of its own) holds more than its low
//! water ([`low_water`]: what the device takes in one go, a second pull
//! and [`SCHEDULING_MARGIN`]). The wait is on events, not on predicted
//! times: the device signals each take from its ring
//! ([`Output::set_pull_signal`]) and the output each write, and the thread
//! looks again at once; a timer only backstops a device that stops taking
//! frames (the wait's slack at its start and a period). Then the period is
//! mixed with what the output has, silence for the rest, and the output
//! is late: not waited for again until it holds a period, so one that
//! stalls keeps no one on the low water. [`next_step`] is the decision,
//! without the clock; [`Run::turn`] carries it out.
//!
//! [`Output::queued`]: crate::Output::queued
//! [`Output::set_pull_signal`]: crate::Output::set_pull_signal

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use super::input::Stream;
use super::run::Run;

/// On a device in real time, what is kept in its ring beyond what the
/// device takes in one go while the mix waits for a lagging input, at the
/// least: the time the mixer thread may take to wake and mix a period (a
/// device pull, when that is longer).
pub(super) const SCHEDULING_MARGIN: Duration = Duration::from_millis(5);

/// The frames a device's ring keeps at the least while the mix waits for a
/// lagging input: the most the device has taken in one go (`burst`, at
/// least two pulls of `pull`: one, and one more while the mixer wakes)
/// and a margin (`margin`, at least a pull), never more than the ring.
pub(super) fn low_water(pull: u64, burst: u64, margin: u64, ring: u64) -> u64 {
    let pull = pull.max(1);
    (burst.max(2 * pull) + margin.max(pull)).min(ring)
}

/// What the thread does after a turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Turn {
    /// More to do at once.
    Busy,
    /// Nothing until word comes.
    Idle,
    /// Waiting for a lagging input: until then, or until it writes.
    Wait(Instant),
}

/// What the next period is, as [`next_step`] decides it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Step {
    /// Mix `frames`; `give_up`: the lagging outputs are late now.
    Mix { frames: u64, give_up: bool },
    /// Wait for a lagging output: the device's ring has this many frames
    /// beyond its low water.
    Wait(u64),
    /// Every output is drained: what the limiter holds back goes.
    GiveHeld,
    /// Nothing to mix for now.
    Idle,
}

/// What [`next_step`] looks at.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Look {
    /// The frames of a period.
    pub(super) period: u64,
    /// Every running output holds a period.
    pub(super) full: bool,
    /// One is lagging ([`lagging`]).
    pub(super) lagging: bool,
    /// The device plays in real time.
    pub(super) realtime: bool,
    /// The most frames an output holds.
    pub(super) most: u64,
    /// Frames the limiter holds back.
    pub(super) held: u64,
    /// On a device in real time with an output lagging: the frames its
    /// ring holds beyond its low water.
    pub(super) slack: u64,
    /// The wait's backstop deadline has passed.
    pub(super) deadline_passed: bool,
}

/// The next period: once every running output holds one. On a device in
/// real time, while one lags, a wait as long as the device's ring holds
/// more than its low water and the deadline has not passed; then the
/// period with what there is, the lagging outputs given up on. On any
/// other device a wait for as long as it takes, a drained output's last
/// frames, and what the limiter holds back once all are drained.
pub(super) fn next_step(look: Look) -> Step {
    let mix = |frames, give_up| Step::Mix { frames, give_up };
    if look.full {
        mix(look.period, false)
    } else if look.realtime {
        if !look.lagging {
            mix(look.period, false)
        } else if look.slack > 0 && !look.deadline_passed {
            Step::Wait(look.slack)
        } else {
            mix(look.period, true)
        }
    } else if look.lagging {
        Step::Idle
    } else if look.most > 0 {
        mix(look.most.min(look.period), false)
    } else if look.held > 0 {
        Step::GiveHeld
    } else {
        Step::Idle
    }
}

/// Whether `stream` lags: running, short of a period, not draining (nor
/// still to write its first frame), and not late.
pub(super) fn lagging(stream: &Stream, period: u64) -> bool {
    stream.link.running.load(Ordering::SeqCst)
        && !stream.late
        && stream.link.shared.ring.len() < period
        && !stream.link.shared.draining()
}

impl Run {
    /// Writes what is mixed, or mixes the next period as [`next_step`]
    /// says. With nothing running, the device pauses.
    pub(super) fn turn(&mut self) -> Turn {
        let playing = self.running().next().is_some();
        let Some(device) = &mut self.device else {
            return Turn::Idle;
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
                return Turn::Idle;
            }
        }
        if !playing {
            device.deadline = None;
            return Turn::Idle;
        }
        let period = device.format.duration_to_frames(self.options.period).max(1);
        if self.sent == self.out.len() {
            let (mut full, mut lag, mut most) = (true, false, 0);
            for s in self.slots.values_mut().filter_map(|s| s.stream.as_mut()) {
                if !s.link.running.load(Ordering::SeqCst) {
                    continue;
                }
                let queued = s.link.shared.ring.len();
                most = most.max(queued);
                if queued >= period {
                    s.late = false;
                } else {
                    full = false;
                    lag |= lagging(s, period);
                }
            }
            let device = self.device.as_mut().expect("checked");
            let realtime = device.info.realtime;
            let slack = if realtime && lag {
                // What the device took since the last look of this wait:
                // more than one pull at a time, the low water holds that.
                let fill = device.output.queued();
                if let Some(last) = device.last_fill {
                    device.burst = device.burst.max(last.saturating_sub(fill));
                }
                device.last_fill = Some(fill);
                let low = low_water(
                    device.info.device_buffer_frames,
                    device.burst,
                    device.format.duration_to_frames(SCHEDULING_MARGIN),
                    device.info.ring_frames,
                );
                fill.saturating_sub(low)
            } else {
                0
            };
            let now = Instant::now();
            let look = Look {
                period,
                full,
                lagging: lag,
                realtime,
                most,
                held: device.mix.held() as u64,
                slack,
                deadline_passed: device.deadline.is_some_and(|d| now >= d),
            };
            match next_step(look) {
                Step::Wait(slack) => {
                    let slack = device.format.frames_to_duration(slack);
                    let tick = self.options.period;
                    let deadline = *device.deadline.get_or_insert(now + slack + tick);
                    // Woken by the device's next take or the output's
                    // write; the timer is the backstop (and word for the
                    // thread waits no longer than a period).
                    return Turn::Wait(deadline.min(now + tick));
                }
                Step::Mix { frames, give_up } => {
                    device.deadline = None;
                    if device.last_fill.take().is_some() {
                        // What it took at a time long ago counts for less.
                        device.burst -= device.burst / 8;
                    }
                    if give_up {
                        for s in self.slots.values_mut().filter_map(|s| s.stream.as_mut()) {
                            if lagging(s, period) {
                                s.late = true;
                            }
                        }
                    }
                    self.mix_chunk(frames as usize);
                }
                Step::GiveHeld => {
                    self.give_held();
                    return if self.write() { Turn::Busy } else { Turn::Idle };
                }
                Step::Idle => return Turn::Idle,
            }
        }
        if self.write() {
            Turn::Busy
        } else {
            Turn::Idle
        }
    }

    /// Sleeps until `until`, or until a lagging output writes: armed first,
    /// then a look at whether one still lags (the write's wake and this
    /// look cannot both miss each other: [`super::wake::Waker::arm`]).
    pub(super) fn wait(&self, until: Instant) {
        let Some(device) = &self.device else {
            return;
        };
        let period = device.format.duration_to_frames(self.options.period).max(1);
        self.wake.arm();
        if self.streams().any(|s| lagging(s, period)) {
            self.wake.sleep_until(until);
        }
        self.wake.disarm();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn look() -> Look {
        Look {
            period: 480,
            ..Look::default()
        }
    }

    fn mix(frames: u64, give_up: bool) -> Step {
        Step::Mix { frames, give_up }
    }

    #[test]
    fn a_period_is_mixed_once_every_output_holds_one() {
        let full = Look {
            full: true,
            realtime: true,
            ..look()
        };
        assert_eq!(next_step(full), mix(480, false));
        assert_eq!(
            next_step(Look {
                full: true,
                ..look()
            }),
            mix(480, false)
        );
    }

    #[test]
    fn in_real_time_a_lagging_output_is_waited_for_down_to_the_low_water() {
        let lagging = Look {
            realtime: true,
            lagging: true,
            slack: 300,
            ..look()
        };
        assert_eq!(next_step(lagging), Step::Wait(300));
        // Down to it, or past the deadline: mixed as it is, given up on.
        assert_eq!(
            next_step(Look {
                slack: 0,
                ..lagging
            }),
            mix(480, true)
        );
        let passed = Look {
            deadline_passed: true,
            ..lagging
        };
        assert_eq!(next_step(passed), mix(480, true));
        // None lagging (late, draining, loading): mixed at once.
        let none = Look {
            lagging: false,
            ..lagging
        };
        assert_eq!(next_step(none), mix(480, false));
    }

    #[test]
    fn the_low_water_holds_two_pulls_and_a_margin_or_the_largest_take() {
        // 10 ms pulls at 48 kHz, a 5 ms margin: two pulls and a pull.
        assert_eq!(low_water(480, 0, 240, 9_600), 1_440);
        // A device that took four periods at once keeps those.
        assert_eq!(low_water(480, 1_920, 240, 9_600), 2_400);
        // A large margin counts as it is; never more than the ring.
        assert_eq!(low_water(480, 0, 960, 9_600), 1_920);
        assert_eq!(low_water(480, 4_800, 240, 1_920), 1_920);
    }

    #[test]
    fn a_device_that_plays_what_it_is_given_waits_for_every_output() {
        let lagging = Look {
            lagging: true,
            most: 100,
            ..look()
        };
        assert_eq!(next_step(lagging), Step::Idle);
        // Short but draining: what there is.
        assert_eq!(
            next_step(Look {
                most: 100,
                ..look()
            }),
            mix(100, false)
        );
        assert_eq!(next_step(Look { held: 64, ..look() }), Step::GiveHeld);
        assert_eq!(next_step(look()), Step::Idle);
    }
}
