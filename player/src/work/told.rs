//! For tests of the workers that tell how they stand through a
//! [`Notify`]: waiting on what they tell, not on the clock.

use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::work::outbox::Notify;

/// How long a wait goes without a word from the worker before it is taken
/// for a hang: far longer than any step takes on a loaded machine, so that
/// only a worker that stopped telling fails by it.
const HANG: Duration = Duration::from_secs(60);

/// Counts the calls of the [`Notify`] it makes ([`Told::notify`]) and wakes
/// whoever waits for the next one.
#[derive(Default)]
pub struct Told {
    count: Mutex<u64>,
    more: Condvar,
}

impl Told {
    pub fn new() -> Arc<Told> {
        Arc::default()
    }

    /// The [`Notify`] to give the worker.
    pub fn notify(self: &Arc<Self>) -> Notify {
        let told = Arc::clone(self);
        Arc::new(move || {
            *told.count.lock().unwrap() += 1;
            told.more.notify_all();
        })
    }

    /// How many times it was told so far.
    pub fn count(&self) -> u64 {
        *self.count.lock().unwrap()
    }

    /// What `look` finds, looked at now and then after each word; `what`
    /// names it when the worker goes [`HANG`] without one.
    pub fn until<T>(&self, what: &str, mut look: impl FnMut() -> Option<T>) -> T {
        let mut count = self.count.lock().unwrap();
        loop {
            if let Some(found) = look() {
                return found;
            }
            let seen = *count;
            let (next, _) = self
                .more
                .wait_timeout_while(count, HANG, |c| *c == seen)
                .unwrap();
            assert!(*next > seen, "no word in {HANG:?}: {what}");
            count = next;
        }
    }
}
