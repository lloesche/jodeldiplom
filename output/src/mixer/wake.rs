//! The mixer thread's sleep while it waits for a lagging input: cut short
//! when the device takes frames from its ring ([`PullSignal`], on the
//! device's real-time thread) or an input writes, while it is armed
//! ([`Waker::wake_if_armed`]: from any thread, without allocating or
//! locking: an event set on Windows, the thread parker's kernel wake
//! elsewhere); otherwise until a backstop deadline, on a
//! high-resolution waitable timer on Windows (where the default wait
//! rounds to the 15.6 ms system tick) and the thread parker elsewhere.
//!
//! The handshake: the sleeper arms, then looks at the rings it waits on,
//! and sleeps only if one still lags; a writer puts its frames in the
//! ring, then looks at the arming. Both sides fence (`SeqCst`) between
//! their store and their load, so at least one sees the other: the writer
//! wakes the sleeper, or the sleeper sees the frames and does not sleep.

use std::sync::atomic::{fence, AtomicBool, Ordering};
use std::time::Instant;

use crate::output::PullSignal;

/// Wakes the mixer thread from [`Waker::sleep_until`].
pub(crate) struct Waker {
    /// The sleeper is about to sleep, or sleeps: a write wakes it.
    armed: AtomicBool,
    /// Woken since the last sleep began (a wake before the sleep ends the
    /// next one at once).
    woken: AtomicBool,
    os: imp::Os,
}

impl Waker {
    /// A waker for the thread that calls this, the one that will sleep.
    pub(crate) fn for_current_thread() -> Waker {
        Waker {
            armed: AtomicBool::new(false),
            woken: AtomicBool::new(false),
            os: imp::Os::new(),
        }
    }

    /// The sleeper, before it looks a last time at what it waits for: a
    /// write from now on wakes it.
    pub(crate) fn arm(&self) {
        self.armed.store(true, Ordering::SeqCst);
        fence(Ordering::SeqCst);
    }

    /// The sleeper, awake again.
    pub(crate) fn disarm(&self) {
        self.armed.store(false, Ordering::SeqCst);
    }

    /// A writer, its frames in the ring: wakes the sleeper if it is armed.
    pub(crate) fn wake_if_armed(&self) {
        fence(Ordering::SeqCst);
        if self.armed.load(Ordering::SeqCst) {
            self.wake();
        }
    }

    /// Ends the sleeper's sleep (or its next one).
    pub(crate) fn wake(&self) {
        if !self.woken.swap(true, Ordering::AcqRel) {
            self.os.signal();
        }
    }

    /// Sleeps until `deadline` or a wake, whichever comes first.
    pub(crate) fn sleep_until(&self, deadline: Instant) {
        if !self.woken.swap(false, Ordering::AcqRel) {
            self.os.wait_until(deadline);
        }
        self.woken.store(false, Ordering::Release);
    }
}

/// The device took frames: a sleeper waiting on its ring looks again.
impl PullSignal for Waker {
    fn pulled(&self) {
        self.wake_if_armed();
    }
}

#[cfg(windows)]
mod imp {
    use std::time::Instant;

    use windows::core::PCWSTR;
    use windows::Win32::Foundation::{CloseHandle, HANDLE};
    use windows::Win32::System::Threading::{
        CreateEventW, CreateWaitableTimerExW, SetEvent, SetWaitableTimer, WaitForMultipleObjects,
        CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, INFINITE, TIMER_ALL_ACCESS,
    };

    /// An auto-reset event the waker sets, and a high-resolution timer
    /// for the deadline. Windows before 10 version 1803 has no
    /// high-resolution timer: a plain one then, its due times rounded to
    /// the system tick (a wait may run that much long; the ring's low
    /// water holds a few milliseconds only, so a device that old may run
    /// dry while an input lags). Without a timer at all: a 1 ms sleep.
    pub(super) struct Os {
        event: Option<HANDLE>,
        timer: Option<HANDLE>,
    }

    // SAFETY: kernel handles are process-wide, usable from any thread; the
    // timer is only set and waited on by the sleeping thread.
    unsafe impl Send for Os {}
    unsafe impl Sync for Os {}

    impl Os {
        pub(super) fn new() -> Os {
            // SAFETY: an unnamed auto-reset event, initially unset.
            let event = unsafe { CreateEventW(None, false, false, None) }.ok();
            let timer = |flags| {
                // SAFETY: an unnamed timer; the handle is ours to close.
                unsafe { CreateWaitableTimerExW(None, PCWSTR::null(), flags, TIMER_ALL_ACCESS.0) }
                    .ok()
            };
            let timer = timer(CREATE_WAITABLE_TIMER_HIGH_RESOLUTION).or_else(|| timer(0));
            Os { event, timer }
        }

        pub(super) fn signal(&self) {
            if let Some(event) = self.event {
                // SAFETY: the handle is open while `self` lives.
                let _ = unsafe { SetEvent(event) };
            }
        }

        pub(super) fn wait_until(&self, deadline: Instant) {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return;
            }
            let (Some(event), Some(timer)) = (self.event, self.timer) else {
                std::thread::sleep(left.min(std::time::Duration::from_millis(1)));
                return;
            };
            // Relative, in 100 ns units (negative), at least one.
            let due = -((left.as_nanos() / 100).max(1) as i64);
            // SAFETY: the timer is open; no completion routine.
            if unsafe { SetWaitableTimer(timer, &due, 0, None, None, false) }.is_err() {
                std::thread::sleep(left);
                return;
            }
            // SAFETY: both handles are open while `self` lives.
            let _ = unsafe { WaitForMultipleObjects(&[event, timer], false, INFINITE) };
        }
    }

    impl Drop for Os {
        fn drop(&mut self) {
            for handle in [self.event, self.timer].into_iter().flatten() {
                // SAFETY: closed once, with nothing waiting on it any more.
                let _ = unsafe { CloseHandle(handle) };
            }
        }
    }
}

#[cfg(not(windows))]
mod imp {
    use std::thread::Thread;
    use std::time::Instant;

    /// The sleeping thread, parked with a timeout (to the nanosecond on
    /// Linux and macOS). Only it may sleep: a park is its own.
    pub(super) struct Os {
        thread: Thread,
    }

    impl Os {
        pub(super) fn new() -> Os {
            Os {
                thread: std::thread::current(),
            }
        }

        pub(super) fn signal(&self) {
            self.thread.unpark();
        }

        pub(super) fn wait_until(&self, deadline: Instant) {
            debug_assert_eq!(
                std::thread::current().id(),
                self.thread.id(),
                "only the waker's own thread sleeps on it"
            );
            // A spurious unpark ends it early: the caller looks again.
            let left = deadline.saturating_duration_since(Instant::now());
            if !left.is_zero() {
                std::thread::park_timeout(left);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn it_sleeps_to_its_deadline_and_a_wake_cuts_it_short() {
        let waker = Arc::new(Waker::for_current_thread());
        let start = Instant::now();
        waker.sleep_until(start + Duration::from_millis(30));
        let slept = start.elapsed();
        assert!(slept >= Duration::from_millis(29), "{slept:?}");
        // Woken from another thread a little into a long sleep.
        let other = waker.clone();
        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            other.wake();
        });
        let start = Instant::now();
        waker.sleep_until(start + Duration::from_secs(5));
        assert!(start.elapsed() < Duration::from_secs(2));
        t.join().unwrap();
        // A wake before the sleep ends the next one at once.
        waker.wake();
        let start = Instant::now();
        waker.sleep_until(start + Duration::from_secs(5));
        assert!(start.elapsed() < Duration::from_secs(1));
    }
}
