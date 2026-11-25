//! The mixer's thread raised for audio work ([`AudioPriority`]), for as
//! long as it holds it: it fills the device's short ring (40 ms) a period
//! at a time, so it must run when its period comes however busy the rest
//! of the process is (the window drawing a heavy frame, a network on the
//! GPU, a slow present). On Windows MMCSS "Pro Audio"; on macOS a real-time
//! thread (Mach's time-constraint policy: some of each period, within it,
//! as Core Audio's own threads run), which also runs on a performance core
//! in Low Power Mode, where an ordinary thread can wait tens of
//! milliseconds behind the window's; elsewhere nothing.

use std::time::Duration;

/// The share of a period the mixer may take, and must have, each period.
#[cfg(target_os = "macos")]
const COMPUTATION: f64 = 0.5;

/// Held while the thread does audio work.
pub(crate) struct AudioPriority {
    #[cfg(windows)]
    _mmcss: super::wasapi::Mmcss,
    #[cfg(target_os = "macos")]
    thread: Option<mach::Thread>,
}

impl AudioPriority {
    /// Raises the calling thread, which does its work every `period`.
    #[cfg_attr(not(target_os = "macos"), allow(unused_variables))]
    pub(crate) fn enter(period: Duration) -> AudioPriority {
        AudioPriority {
            #[cfg(windows)]
            _mmcss: super::wasapi::Mmcss::enter(),
            #[cfg(target_os = "macos")]
            thread: mach::real_time(period, COMPUTATION),
        }
    }
}

#[cfg(target_os = "macos")]
impl Drop for AudioPriority {
    fn drop(&mut self) {
        if let Some(thread) = self.thread {
            mach::standard(thread);
        }
    }
}

#[cfg(target_os = "macos")]
mod mach {
    use std::time::Duration;

    /// A Mach thread port.
    pub(super) type Thread = u32;

    /// `thread_time_constraint_policy_data_t`, in Mach absolute time.
    #[repr(C)]
    #[derive(Default)]
    pub(super) struct TimeConstraint {
        pub(super) period: u32,
        pub(super) computation: u32,
        pub(super) constraint: u32,
        pub(super) preemptible: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct Timebase {
        numer: u32,
        denom: u32,
    }

    const THREAD_STANDARD_POLICY: u32 = 1;
    pub(super) const THREAD_TIME_CONSTRAINT_POLICY: u32 = 2;
    pub(super) const TIME_CONSTRAINT_COUNT: u32 = 4;

    unsafe extern "C" {
        fn pthread_self() -> usize;
        fn pthread_mach_thread_np(thread: usize) -> Thread;
        fn mach_timebase_info(info: *mut Timebase) -> i32;
        fn thread_policy_set(thread: Thread, flavor: u32, info: *const u32, count: u32) -> i32;
    }

    /// The calling thread's port (not a new right: nothing to release).
    pub(super) fn this_thread() -> Thread {
        // SAFETY: both only read the calling thread's identity.
        unsafe { pthread_mach_thread_np(pthread_self()) }
    }

    /// `d` in Mach absolute time.
    fn absolute(d: Duration) -> u32 {
        let mut tb = Timebase::default();
        // SAFETY: fills the struct given.
        if unsafe { mach_timebase_info(&mut tb) } != 0 || tb.numer == 0 {
            return 0;
        }
        let ticks = d.as_nanos() * u128::from(tb.denom) / u128::from(tb.numer);
        u32::try_from(ticks).unwrap_or(u32::MAX)
    }

    /// The calling thread made real time: `share` of every `period` of
    /// processor time, within the period. Its port where that took.
    pub(super) fn real_time(period: Duration, share: f64) -> Option<Thread> {
        let thread = this_thread();
        let policy = TimeConstraint {
            period: absolute(period),
            computation: absolute(period.mul_f64(share)),
            constraint: absolute(period),
            preemptible: 1,
        };
        if policy.period == 0 {
            return None;
        }
        // SAFETY: the policy is the flavour's struct, of its count of
        // 32-bit words, for the calling thread.
        let set = unsafe {
            thread_policy_set(
                thread,
                THREAD_TIME_CONSTRAINT_POLICY,
                (&raw const policy).cast(),
                TIME_CONSTRAINT_COUNT,
            )
        };
        (set == 0).then_some(thread)
    }

    /// `thread` back to the ordinary policy.
    pub(super) fn standard(thread: Thread) {
        // SAFETY: the standard policy has no data.
        let _ = unsafe { thread_policy_set(thread, THREAD_STANDARD_POLICY, std::ptr::null(), 0) };
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    unsafe extern "C" {
        fn thread_policy_get(
            thread: mach::Thread,
            flavor: u32,
            info: *mut u32,
            count: *mut u32,
            get_default: *mut u32,
        ) -> i32;
    }

    /// The calling thread's time-constraint policy, if it has one.
    fn policy() -> Option<mach::TimeConstraint> {
        let mut p = mach::TimeConstraint::default();
        let mut count = mach::TIME_CONSTRAINT_COUNT;
        let mut default = 0u32;
        // SAFETY: the struct and its count, for the calling thread.
        let got = unsafe {
            thread_policy_get(
                mach::this_thread(),
                mach::THREAD_TIME_CONSTRAINT_POLICY,
                (&raw mut p).cast(),
                &mut count,
                &mut default,
            )
        };
        (got == 0 && default == 0).then_some(p)
    }

    #[test]
    fn the_mixer_s_thread_is_real_time_while_it_holds_its_priority() {
        std::thread::spawn(|| {
            assert!(policy().is_none(), "an ordinary thread to begin with");
            let held = AudioPriority::enter(Duration::from_millis(10));
            let p = policy().expect("real time");
            assert!(p.computation > 0 && p.computation < p.period);
            assert_eq!(p.constraint, p.period);
            drop(held);
            assert!(policy().is_none(), "ordinary again");
        })
        .join()
        .unwrap();
    }
}
