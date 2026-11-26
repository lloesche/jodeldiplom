//! What the background workers are built of: a [`Queue`] of work their
//! threads wait on, the two [`Lanes`] a worker is asked in (wanted now,
//! and to get ready ahead), and a [`Job`] that runs once on a thread of
//! its own, tells how it stands, and waits for a retry after a failure.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use crate::outbox::Notify;

/// Work for background threads to wait on: what is to be done (`W`), under
/// a lock, and a stop. Every change wakes every thread waiting, and a stop
/// ends each at its next wait. A lock left poisoned by a thread that
/// panicked is taken over as it is.
pub struct Queue<W> {
    work: Mutex<W>,
    ready: Condvar,
    stop: AtomicBool,
}

/// What [`Queue::next_within`] found.
pub enum Next<T> {
    /// Something to do.
    Do(T),
    /// Nothing, for as long as it waited.
    Idle,
    /// Stopped.
    Stop,
}

impl<W> Queue<W> {
    /// A queue starting with `work`.
    pub fn new(work: W) -> Queue<W> {
        Queue {
            work: Mutex::new(work),
            ready: Condvar::new(),
            stop: AtomicBool::new(false),
        }
    }

    /// The work, locked.
    pub fn lock(&self) -> MutexGuard<'_, W> {
        self.work.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Changes the work and wakes every thread waiting.
    pub fn change<R>(&self, change: impl FnOnce(&mut W) -> R) -> R {
        let made = change(&mut self.lock());
        self.ready.notify_all();
        made
    }

    /// The next thing to do, as `take` takes it from the work, waited for;
    /// `None` once stopped.
    pub fn next<T>(&self, take: impl FnMut(&mut W) -> Option<T>) -> Option<T> {
        match self.next_within(Duration::MAX, take) {
            Next::Do(found) => Some(found),
            Next::Idle | Next::Stop => None,
        }
    }

    /// [`Queue::next`], waiting at most `wait`.
    pub fn next_within<T>(
        &self,
        wait: Duration,
        mut take: impl FnMut(&mut W) -> Option<T>,
    ) -> Next<T> {
        let until = Instant::now().checked_add(wait);
        let mut work = self.lock();
        loop {
            if self.stopped() {
                return Next::Stop;
            }
            if let Some(found) = take(&mut work) {
                return Next::Do(found);
            }
            work = match until {
                None => self
                    .ready
                    .wait(work)
                    .unwrap_or_else(PoisonError::into_inner),
                Some(until) => {
                    let left = until.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        return Next::Idle;
                    }
                    self.ready
                        .wait_timeout(work, left)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0
                }
            };
        }
    }

    /// Waits `pause` unless stopped meanwhile: whether it goes on.
    pub fn pause(&self, pause: Duration) -> bool {
        let work = self.lock();
        let _ = self
            .ready
            .wait_timeout_while(work, pause, |_| !self.stopped())
            .unwrap_or_else(PoisonError::into_inner);
        !self.stopped()
    }

    /// Whether it was stopped.
    pub fn stopped(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }

    /// Stops it: every thread ends at its next wait (one at work finishes
    /// what it is at first).
    pub fn stop(&self) {
        let _work = self.lock();
        self.stop.store(true, Ordering::SeqCst);
        self.ready.notify_all();
    }
}

/// Which of a worker's [`Lanes`] a thing came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    /// Wanted now: answered.
    Now,
    /// To get ready ahead, after what is wanted: kept, not answered.
    Ahead,
}

/// What a worker is asked for, in two lanes: what is wanted now, and what
/// to get ready ahead after it. Each is replaced whole when asked anew, and
/// counts how often it was ([`Lanes::asked`]), so that what was found for
/// an older ask is dropped.
pub struct Lanes<T> {
    now: VecDeque<T>,
    ahead: VecDeque<T>,
    asked: [u64; 2],
}

impl<T> Default for Lanes<T> {
    fn default() -> Lanes<T> {
        Lanes {
            now: VecDeque::new(),
            ahead: VecDeque::new(),
            asked: [0; 2],
        }
    }
}

impl<T> Lanes<T> {
    /// Wants exactly `items` now, in this order (what was wanted before
    /// and not taken yet is dropped).
    pub fn want(&mut self, items: impl IntoIterator<Item = T>) {
        self.ask(Lane::Now, items);
    }

    /// Gets `items` ready ahead, in this order, after what is wanted (what
    /// was asked for before and not taken yet is dropped).
    pub fn prepare(&mut self, items: impl IntoIterator<Item = T>) {
        self.ask(Lane::Ahead, items);
    }

    fn ask(&mut self, lane: Lane, items: impl IntoIterator<Item = T>) {
        *self.lane(lane) = items.into_iter().collect();
        self.asked[lane as usize] += 1;
    }

    /// How often `lane` was asked anew.
    pub fn asked(&self, lane: Lane) -> u64 {
        self.asked[lane as usize]
    }

    /// The things waiting in `lane`, in order.
    pub fn lane(&mut self, lane: Lane) -> &mut VecDeque<T> {
        match lane {
            Lane::Now => &mut self.now,
            Lane::Ahead => &mut self.ahead,
        }
    }

    /// The next thing to do: the first wanted, else the first ahead.
    pub fn pop(&mut self) -> Option<(T, Lane)> {
        let lane = if self.now.is_empty() {
            Lane::Ahead
        } else {
            Lane::Now
        };
        self.lane(lane).pop_front().map(|t| (t, lane))
    }
}

/// A job on a thread of its own, run once: how it stands (`S`, read by the
/// screen), told through a [`Notify`] as it changes, and a retry the job
/// waits for after a failure.
pub struct Job<S> {
    status: Mutex<S>,
    /// Set by [`Job::retry`], taken by the job waiting after a failure.
    again: Mutex<bool>,
    told: Condvar,
    notify: Notify,
}

impl<S: Clone> Job<S> {
    /// A job standing at `status`, telling `notify`.
    pub fn new(status: S, notify: Notify) -> Arc<Job<S>> {
        Arc::new(Job {
            status: Mutex::new(status),
            again: Mutex::new(false),
            told: Condvar::new(),
            notify,
        })
    }

    /// Where it stands now.
    pub fn status(&self) -> S {
        self.status
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// After a failure: the job goes on.
    pub fn retry(&self) {
        *self.again.lock().unwrap_or_else(PoisonError::into_inner) = true;
        self.told.notify_all();
    }

    /// The job's side: waits for [`Job::retry`] (one asked for already
    /// counts).
    pub fn wait_for_retry(&self) {
        let mut again = self.again.lock().unwrap_or_else(PoisonError::into_inner);
        while !*again {
            again = self
                .told
                .wait(again)
                .unwrap_or_else(PoisonError::into_inner);
        }
        *again = false;
    }

    /// The job's side: what changes its status, telling at most once
    /// `every`.
    pub fn teller(&self, every: Duration) -> Teller<'_, S> {
        Teller {
            job: self,
            every,
            last: Instant::now(),
        }
    }
}

/// Changes a [`Job`]'s status, telling the screen at most once a while
/// (every change is told when forced: a step's end).
pub struct Teller<'a, S> {
    job: &'a Job<S>,
    every: Duration,
    last: Instant,
}

impl<S> Teller<'_, S> {
    /// Changes the status by `change`, and tells it when `force`d or once
    /// the while is up.
    pub fn set(&mut self, force: bool, change: impl FnOnce(&mut S)) {
        change(
            &mut self
                .job
                .status
                .lock()
                .unwrap_or_else(PoisonError::into_inner),
        );
        if force || self.last.elapsed() >= self.every {
            self.last = Instant::now();
            (self.job.notify)();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn a_stop_ends_every_thread_waiting() {
        let queue = Arc::new(Queue::new(VecDeque::from([7])));
        assert_eq!(queue.next(VecDeque::pop_front), Some(7));
        let threads: Vec<_> = (0..3)
            .map(|_| {
                let queue = Arc::clone(&queue);
                thread::spawn(move || queue.next(VecDeque::pop_front))
            })
            .collect();
        queue.stop();
        for thread in threads {
            assert_eq!(thread.join().unwrap(), None);
        }
    }

    #[test]
    fn a_wait_within_a_while_comes_back_idle() {
        let queue = Queue::new(None::<u32>);
        assert!(matches!(
            queue.next_within(Duration::from_millis(1), Option::take),
            Next::Idle
        ));
        queue.change(|w| *w = Some(1));
        assert!(matches!(
            queue.next_within(Duration::from_millis(1), Option::take),
            Next::Do(1)
        ));
        queue.stop();
        assert!(matches!(
            queue.next_within(Duration::from_millis(1), Option::take),
            Next::Stop
        ));
        assert!(!queue.pause(Duration::from_secs(60)), "a stop ends a pause");
    }

    #[test]
    fn lanes_give_what_is_wanted_first_and_count_their_asks() {
        let mut lanes = Lanes::default();
        lanes.prepare([1, 2]);
        lanes.want([3]);
        assert_eq!(lanes.pop(), Some((3, Lane::Now)));
        assert_eq!(lanes.pop(), Some((1, Lane::Ahead)));
        lanes.prepare([4]);
        assert_eq!((lanes.asked(Lane::Now), lanes.asked(Lane::Ahead)), (1, 2));
        assert_eq!(lanes.pop(), Some((4, Lane::Ahead)));
        assert_eq!(lanes.pop(), None);
    }

    #[test]
    fn a_job_waits_for_its_retry_and_tells_each_forced_change() {
        let told = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let notify: Notify = {
            let told = Arc::clone(&told);
            Arc::new(move || {
                told.fetch_add(1, Ordering::SeqCst);
            })
        };
        let job = Job::new(0u32, notify);
        let mut teller = job.teller(Duration::from_secs(3600));
        teller.set(false, |s| *s = 1);
        teller.set(true, |s| *s = 2);
        assert_eq!((job.status(), told.load(Ordering::SeqCst)), (2, 1));
        job.retry();
        job.wait_for_retry();
    }
}
