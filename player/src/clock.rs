//! The app's clock: what it times by (its ticks, the frames' clock, the
//! time since a position was told, the result's reveal). The wall clock,
//! or a stepped one ([`AppClock::stepped`]) that a test moves on by hand,
//! as it lets a simulated device play ([`jodel_output::SimTime`]), so the
//! two go exactly as far together, however busy the machine.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The app's clock; the wall clock by default.
#[derive(Clone, Debug, Default)]
pub struct AppClock(Option<Arc<Stepped>>);

/// A stepped clock: where it started, and the nanoseconds it was moved on.
#[derive(Debug)]
struct Stepped {
    from: Instant,
    nanos: AtomicU64,
}

impl AppClock {
    /// A clock that stands still but where it is moved on
    /// ([`AppClock::step`]), from now.
    pub fn stepped() -> AppClock {
        AppClock(Some(Arc::new(Stepped {
            from: Instant::now(),
            nanos: AtomicU64::new(0),
        })))
    }

    /// The time now.
    pub fn now(&self) -> Instant {
        match &self.0 {
            None => Instant::now(),
            Some(s) => s.from + Duration::from_nanos(s.nanos.load(Ordering::Acquire)),
        }
    }

    /// The time since `then` (none, where `then` is later).
    pub fn since(&self, then: Instant) -> Duration {
        self.now().saturating_duration_since(then)
    }

    /// A stepped clock moved on by `by`, for every copy of it.
    ///
    /// # Panics
    /// On the wall clock: it moves by itself.
    pub fn step(&self, by: Duration) {
        let s = self.0.as_ref().expect("only a stepped clock is stepped");
        let nanos = u64::try_from(by.as_nanos()).unwrap_or(u64::MAX);
        s.nanos.fetch_add(nanos, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stepped clock moves only when stepped, by as much, in every copy.
    #[test]
    fn a_stepped_clock_moves_only_when_stepped() {
        let clock = AppClock::stepped();
        let copy = clock.clone();
        let start = clock.now();
        assert_eq!(clock.now(), start);
        copy.step(Duration::from_millis(16));
        assert_eq!(clock.since(start), Duration::from_millis(16));
        assert_eq!(
            copy.since(clock.now() + Duration::from_secs(1)),
            Duration::ZERO
        );
    }
}
