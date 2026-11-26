//! A network the process keeps on a device of its own ([`Kept`]): set up
//! when first needed, and set up again when its device is lost (a driver
//! reset, a GPU unplugged or switched). A setup that failed stands for
//! [`RETRY`]; the next use after that tries again.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use crate::Turned;

/// How long a failed setup stands before the next use tries again.
pub(crate) const RETRY: Duration = Duration::from_secs(30);

/// A network on a device: whether the device was lost
/// ([`super::Program::lost`]).
pub(crate) trait OnDevice {
    fn lost(&self) -> bool;
}

/// What sets a network up: the network on its device, and the GPU's name.
pub(crate) type SetUp<T> = Box<dyn Fn() -> Result<(T, String), String> + Send + Sync>;

/// The network now, and what the last setup gave.
struct Slot<T> {
    on: Option<Turned<T>>,
    /// When the last setup failed, and why.
    failed: Option<(Instant, String)>,
}

/// A network set up by `set_up` when first needed and kept, set up again
/// when its device is lost, and run one use at a time ([`Kept::run`]).
pub(crate) struct Kept<T> {
    set_up: SetUp<T>,
    /// How long a failed setup stands ([`RETRY`]).
    retry: Duration,
    slot: Mutex<Slot<T>>,
    /// Whether the network is set up, read without waiting for a use.
    up: AtomicBool,
    /// The GPU's name, from the first setup that worked (the process's GPU
    /// is chosen once), read without waiting for a use.
    name: OnceLock<String>,
}

impl<T: OnDevice> Kept<T> {
    /// The network `set_up` makes, not set up yet.
    pub(crate) fn new(set_up: SetUp<T>) -> Kept<T> {
        Kept {
            set_up,
            retry: RETRY,
            slot: Mutex::new(Slot {
                on: None,
                failed: None,
            }),
            up: AtomicBool::new(false),
            name: OnceLock::new(),
        }
    }

    /// The network set up now; why not.
    pub(crate) fn set_up_now(set_up: SetUp<T>) -> Result<Kept<T>, String> {
        let kept = Kept::new(set_up);
        kept.ready(&mut kept.lock())?;
        Ok(kept)
    }

    /// The slot; a use that panicked leaves no network behind (the next
    /// use sets one up).
    fn lock(&self) -> MutexGuard<'_, Slot<T>> {
        self.slot.lock().unwrap_or_else(|e| {
            let mut slot = e.into_inner();
            slot.on = None;
            self.up.store(false, Ordering::Release);
            self.slot.clear_poison();
            slot
        })
    }

    /// Sets the network up where it is not, or where its device was lost;
    /// a failed setup stands for `retry`.
    fn ready(&self, slot: &mut Slot<T>) -> Result<(), String> {
        if slot.on.as_ref().is_some_and(|on| on.lost()) {
            slot.on = None;
        }
        if slot.on.is_some() {
            return Ok(());
        }
        self.up.store(false, Ordering::Release);
        if let Some((at, why)) = &slot.failed {
            if at.elapsed() < self.retry {
                return Err(why.clone());
            }
        }
        let made = std::panic::catch_unwind(std::panic::AssertUnwindSafe(&self.set_up))
            .unwrap_or_else(|_| Err("setting it up panicked".into()));
        match made {
            Ok((on, name)) => {
                slot.on = Some(Turned::new(on));
                let _ = self.name.set(name);
                slot.failed = None;
                self.up.store(true, Ordering::Release);
                Ok(())
            }
            Err(why) => {
                slot.failed = Some((Instant::now(), why.clone()));
                Err(why)
            }
        }
    }

    /// Whether the network is set up (set up now where it is not and may
    /// be). Once it is, this waits for no use under way.
    pub(crate) fn usable(&self) -> bool {
        self.up.load(Ordering::Acquire) || self.ready(&mut self.lock()).is_ok()
    }

    /// `work` on the network, set up first where it is not; where its
    /// device is lost meanwhile, once more on a network set up again.
    pub(crate) fn run<R>(
        &self,
        mut work: impl FnMut(&mut T) -> Result<R, String>,
    ) -> Result<R, String> {
        let mut slot = self.lock();
        self.ready(&mut slot)?;
        let on = slot.on.as_mut().expect("set up");
        let done = work(on);
        if done.is_ok() || !on.lost() {
            return done;
        }
        slot.on = None;
        self.ready(&mut slot)?;
        work(slot.on.as_mut().expect("set up"))
    }

    /// The GPU's name, once set up.
    pub(crate) fn name(&self) -> String {
        self.name.get().cloned().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    /// A network whose device is lost when `lost` says.
    struct Fake {
        lost: Arc<AtomicBool>,
    }

    impl OnDevice for Fake {
        fn lost(&self) -> bool {
            self.lost.load(Ordering::Acquire)
        }
    }

    /// A kept fake whose setups are counted, failing while `fail` says;
    /// a failed setup standing for `retry`.
    fn fake(
        retry: Duration,
        fail: Arc<AtomicBool>,
    ) -> (Kept<Fake>, Arc<AtomicUsize>, Arc<AtomicBool>) {
        let setups = Arc::new(AtomicUsize::new(0));
        let lost = Arc::new(AtomicBool::new(false));
        let (count, flag) = (Arc::clone(&setups), Arc::clone(&lost));
        let kept = Kept {
            retry,
            ..Kept::new(Box::new(move || {
                count.fetch_add(1, Ordering::SeqCst);
                if fail.load(Ordering::SeqCst) {
                    return Err("no GPU".into());
                }
                flag.store(false, Ordering::SeqCst);
                Ok((
                    Fake {
                        lost: Arc::clone(&flag),
                    },
                    "a GPU".into(),
                ))
            }))
        };
        (kept, setups, lost)
    }

    #[test]
    fn a_device_lost_is_set_up_again_and_the_work_done_on_it() {
        let (kept, setups, lost) = fake(RETRY, Arc::new(AtomicBool::new(false)));
        assert_eq!(kept.run(|_| Ok(1)), Ok(1));
        assert_eq!(kept.name(), "a GPU");
        // Lost between two uses: the next sets it up again.
        lost.store(true, Ordering::SeqCst);
        assert_eq!(kept.run(|on| Ok(on.lost())), Ok(false));
        assert_eq!(setups.load(Ordering::SeqCst), 2);
        // Lost during a use: done again on a new one.
        let mut tries = 0;
        let done = kept.run(|on| {
            tries += 1;
            if tries == 1 {
                on.lost.store(true, Ordering::SeqCst);
                return Err("lost".to_owned());
            }
            Ok(tries)
        });
        assert_eq!(done, Ok(2));
        assert_eq!(setups.load(Ordering::SeqCst), 3);
        // An error on a device that is fine is the work's own.
        assert_eq!(
            kept.run(|_| Err::<(), _>("no".to_owned())),
            Err("no".into())
        );
        assert_eq!(setups.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn a_failed_setup_stands_for_a_while_then_is_tried_again() {
        let fail = Arc::new(AtomicBool::new(true));
        let (kept, setups, _) = fake(RETRY, Arc::clone(&fail));
        assert!(!kept.usable());
        assert!(!kept.usable());
        assert_eq!(kept.run(|_| Ok(())), Err("no GPU".into()));
        assert_eq!(setups.load(Ordering::SeqCst), 1, "it stands");
        let (kept, setups, _) = fake(Duration::ZERO, Arc::clone(&fail));
        assert!(!kept.usable());
        fail.store(false, Ordering::SeqCst);
        assert!(kept.usable());
        assert!(kept.usable());
        assert_eq!(setups.load(Ordering::SeqCst), 2, "tried again, then kept");
    }

    #[test]
    fn a_use_that_panicked_leaves_a_new_setup_behind() {
        let (kept, setups, _) = fake(RETRY, Arc::new(AtomicBool::new(false)));
        let kept = Arc::new(kept);
        let other = Arc::clone(&kept);
        let panicked =
            std::thread::spawn(move || other.run(|_| -> Result<(), String> { panic!("a bug") }))
                .join();
        assert!(panicked.is_err());
        assert_eq!(kept.run(|_| Ok(7)), Ok(7));
        assert_eq!(setups.load(Ordering::SeqCst), 2);
    }
}
