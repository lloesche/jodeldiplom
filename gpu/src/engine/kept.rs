//! A network kept on a device of its own ([`Kept`]): set up when first
//! needed, and set up again when its device is lost (a driver reset, a
//! GPU unplugged or switched) or after it was let go. A setup that failed
//! stands a while ([`RETRY`] at first, doubled after each failure up to
//! [`MOST_RETRY`]); the next use after that tries again. It is a
//! [`Resident`]: what it takes on the GPU is said before it is set up (its
//! footprint, worked out from its shapes once they can be read; until
//! then [`UNKNOWN`]), and a scheduler may let it go.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use crate::{Footprint, Resident, Turned};

/// How long a first failed setup stands before the next use tries again.
pub(crate) const RETRY: Duration = Duration::from_secs(30);
/// The longest a failed setup stands (each failure doubles the wait).
pub(crate) const MOST_RETRY: Duration = Duration::from_secs(30 * 60);

/// The footprint of a network whose shapes cannot be read yet (its files
/// not there yet): the most there is, so a scheduler loads nothing beside
/// it.
pub const UNKNOWN: Footprint = Footprint {
    resident: u64::MAX,
    working: 0,
};

/// A network on a device: whether the device was lost.
pub trait OnDevice {
    /// Whether its device was lost: it runs nothing again.
    fn lost(&self) -> bool;
}

/// What sets a network up: the network on its device, and the GPU's name.
pub type SetUp<T> = Box<dyn Fn() -> Result<(T, String), String> + Send + Sync>;

/// How long a failed setup stands: `first`, else twice as long as the
/// failure `before` it, up to [`MOST_RETRY`].
pub(super) fn stands(before: Option<Duration>, first: Duration) -> Duration {
    before.map_or(first, |before| (before * 2).min(MOST_RETRY))
}

/// What works a network's footprint out: `None` while its shapes cannot
/// be read yet.
type Plan = Box<dyn Fn() -> Option<Footprint> + Send + Sync>;

/// The network now, and what the last setup gave.
struct Slot<T> {
    on: Option<Turned<T>>,
    /// When the last setup failed, why, and how long that stands.
    failed: Option<(Instant, String, Duration)>,
}

/// A network set up by `set_up` when first needed and kept, set up again
/// when its device is lost or after it was let go, and run one use at a
/// time ([`Kept::run`]).
pub struct Kept<T> {
    set_up: SetUp<T>,
    /// Its footprint, worked out when first asked and kept once it could
    /// be ([`Resident::footprint`]).
    plan: Plan,
    planned: OnceLock<Footprint>,
    /// Held while it is worked out: one asking meanwhile waits for that
    /// answer rather than work it out again.
    planning: Mutex<()>,
    /// How long a first failed setup stands ([`RETRY`]).
    retry: Duration,
    slot: Mutex<Slot<T>>,
    /// Whether the network is set up, read without waiting for a use.
    up: AtomicBool,
    /// The GPU's name, from the first setup that worked (the process's GPU
    /// is chosen once), read without waiting for a use.
    name: OnceLock<String>,
}

impl<T: OnDevice> Kept<T> {
    /// The network `set_up` makes, not set up yet, taking what `footprint`
    /// works out on the GPU (`None` while it cannot: asked again later).
    pub fn new(
        footprint: impl Fn() -> Option<Footprint> + Send + Sync + 'static,
        set_up: SetUp<T>,
    ) -> Kept<T> {
        Kept {
            set_up,
            plan: Box::new(footprint),
            planned: OnceLock::new(),
            planning: Mutex::new(()),
            retry: RETRY,
            slot: Mutex::new(Slot {
                on: None,
                failed: None,
            }),
            up: AtomicBool::new(false),
            name: OnceLock::new(),
        }
    }

    /// [`Kept::new`], set up now; why not.
    pub fn set_up_now(
        footprint: impl Fn() -> Option<Footprint> + Send + Sync + 'static,
        set_up: SetUp<T>,
    ) -> Result<Kept<T>, String> {
        let kept = Kept::new(footprint, set_up);
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
    /// a failed setup stands a while: `retry`, doubled after each failure
    /// in a row, up to [`MOST_RETRY`].
    fn ready(&self, slot: &mut Slot<T>) -> Result<(), String> {
        if slot.on.as_ref().is_some_and(|on| on.lost()) {
            slot.on = None;
        }
        if slot.on.is_some() {
            return Ok(());
        }
        self.up.store(false, Ordering::Release);
        if let Some((at, why, stands)) = &slot.failed {
            if at.elapsed() < *stands {
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
                let before = slot.failed.as_ref().map(|&(_, _, before)| before);
                let stands = stands(before, self.retry);
                slot.failed = Some((Instant::now(), why.clone(), stands));
                Err(why)
            }
        }
    }

    /// `work` on the network where it is set up (and its device not
    /// lost); `None` where it is not: nothing is set up for it.
    pub fn with_set_up<R>(&self, work: impl FnOnce(&mut T) -> R) -> Option<R> {
        let mut slot = self.lock();
        let on = slot.on.as_mut().filter(|on| !on.lost())?;
        Some(work(on))
    }

    /// Whether the network is set up (set up now where it is not and may
    /// be). Once it is, this waits for no use under way.
    pub fn usable(&self) -> bool {
        self.up.load(Ordering::Acquire) || self.ready(&mut self.lock()).is_ok()
    }

    /// `work` on the network, set up first where it is not; where its
    /// device is lost meanwhile, once more on a network set up again.
    pub fn run<R>(&self, mut work: impl FnMut(&mut T) -> Result<R, String>) -> Result<R, String> {
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
    pub fn name(&self) -> String {
        self.name.get().cloned().unwrap_or_default()
    }
}

impl<T: OnDevice + Send> Resident for Kept<T> {
    /// Worked out once its shapes can be read, and kept then; until then
    /// [`UNKNOWN`]. One asking while it is worked out waits for that.
    fn footprint(&self) -> Footprint {
        if let Some(known) = self.planned.get() {
            return *known;
        }
        let _planning = self.planning.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(known) = self.planned.get() {
            return *known;
        }
        match (self.plan)() {
            Some(known) => *self.planned.get_or_init(|| known),
            None => UNKNOWN,
        }
    }

    /// The network goes, its device with it (after a use under way).
    fn release(&self) {
        let mut slot = self.lock();
        slot.on = None;
        self.up.store(false, Ordering::Release);
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
            ..Kept::new(
                || Some(Footprint::default()),
                Box::new(move || {
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
                }),
            )
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
    fn a_failed_setup_stands_longer_each_time_up_to_half_an_hour() {
        let mut waits = vec![stands(None, RETRY)];
        for _ in 0..8 {
            waits.push(stands(waits.last().copied(), RETRY));
        }
        let minutes: Vec<u64> = waits.iter().map(|w| w.as_secs() / 30).collect();
        assert_eq!(minutes, [1, 2, 4, 8, 16, 32, 60, 60, 60]);
        // A failure stands the first wait; a setup that works forgets it.
        let fail = Arc::new(AtomicBool::new(true));
        let (kept, _, _) = fake(Duration::ZERO, Arc::clone(&fail));
        assert!(!kept.usable());
        assert_eq!(
            kept.lock().failed.as_ref().map(|f| f.2),
            Some(Duration::ZERO)
        );
        fail.store(false, Ordering::SeqCst);
        assert!(kept.usable());
        assert!(kept.lock().failed.is_none());
    }

    #[test]
    fn what_is_set_up_is_worked_on_and_nothing_is_set_up_for_it() {
        let (kept, setups, _) = fake(RETRY, Arc::new(AtomicBool::new(false)));
        assert_eq!(kept.with_set_up(|_| 1), None);
        assert_eq!(setups.load(Ordering::SeqCst), 0, "not set up for it");
        assert!(kept.usable());
        assert_eq!(kept.with_set_up(|_| 2), Some(2));
    }

    #[test]
    fn a_footprint_not_known_yet_is_asked_again_and_kept_once_known() {
        let known = Arc::new(AtomicBool::new(false));
        let asked = Arc::clone(&known);
        let footprint = Footprint {
            resident: 1,
            working: 2,
        };
        let kept: Kept<Fake> = Kept::new(
            move || asked.load(Ordering::SeqCst).then_some(footprint),
            Box::new(|| Err("no GPU".into())),
        );
        assert_eq!(kept.footprint(), UNKNOWN);
        known.store(true, Ordering::SeqCst);
        assert_eq!(kept.footprint(), footprint);
        known.store(false, Ordering::SeqCst);
        assert_eq!(kept.footprint(), footprint, "kept once known");
    }

    #[test]
    fn a_network_let_go_is_set_up_again_on_its_next_use() {
        let (kept, setups, _) = fake(RETRY, Arc::new(AtomicBool::new(false)));
        assert_eq!(kept.run(|_| Ok(1)), Ok(1));
        kept.release();
        assert!(!kept.up.load(Ordering::Acquire));
        kept.release();
        assert_eq!(setups.load(Ordering::SeqCst), 1, "nothing set up to let go");
        assert_eq!(kept.run(|_| Ok(2)), Ok(2));
        assert_eq!(setups.load(Ordering::SeqCst), 2);
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
