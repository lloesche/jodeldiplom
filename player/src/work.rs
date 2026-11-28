//! The player's background work: the workers, each on threads of its own
//! (never the audio's, never holding up play), and what they are built
//! of.
//!
//! - The workers: the beats and tempo of the tracks ([`rhythm`]), their
//!   lyrics ([`lyrics`]), their karaoke stories planned, drawn and their
//!   scenes taken apart ([`story`]), the song playing's scenes made ready
//!   for the karaoke background ([`scenes`]; the picture tools in
//!   [`picture`]), the cover art decoded ([`art`]), each song's loudness
//!   ([`loudness`]) and the original singer's phrases ([`shine`]) as the
//!   karaoke sings it, and the files the setup fetches ([`download`]).
//!   Each keeps what it makes in the data folder. Their work on the GPU
//!   goes by model, each model's jobs in a queue of its own ([`stages`]),
//!   a song's voice taken from its kept stems or separated there
//!   ([`voice`]).
//! - What they are built of: a [`Queue`] of work their threads wait on,
//!   the two [`Lanes`] a worker is asked in (wanted now, and to get ready
//!   ahead), a [`Job`] that runs once on a thread of its own, tells how it
//!   stands, and waits for a retry after a failure, and a thread doing the
//!   last thing asked of it ([`Latest`]: the shine's, the loudness's);
//!   their results come back to the app through the [`outbox`].

pub mod art;
pub mod download;
pub mod loudness;
pub mod lyrics;
pub mod outbox;
pub mod picture;
pub mod rhythm;
pub mod scenes;
pub mod shine;
pub mod stages;
pub mod story;
pub mod story_models;
#[cfg(test)]
pub(crate) mod told;
pub mod voice;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use outbox::Notify;

/// A song's file name, for the trace.
pub fn song_name(path: &std::path::Path) -> std::borrow::Cow<'_, str> {
    path.file_name().unwrap_or_default().to_string_lossy()
}

/// A thread doing the last thing asked of it, one at a time: what is
/// asked while it works waits, and of all asked meanwhile only the last is
/// done ([`Latest::want`], each ask once in a row); what it finds is
/// taken as it comes ([`Latest::take`]), `notify` told as each is (the
/// shine's references, the songs' loudness).
pub struct Latest<Ask, Found> {
    tx: Sender<Ask>,
    rx: Receiver<Found>,
    /// What was last asked for.
    asked: Option<Ask>,
}

impl<Ask: Clone + Send + 'static, Found: Send + 'static> Latest<Ask, Found> {
    /// A worker on a thread of its own: `start` run on it first (setting
    /// the thread up), giving the work done for each ask; `notify` told as
    /// each is done.
    pub fn new<W: FnMut(Ask) -> Found>(
        start: impl FnOnce() -> W + Send + 'static,
        notify: Notify,
    ) -> Latest<Ask, Found> {
        let (tx, wants) = mpsc::channel::<Ask>();
        let (found, rx) = mpsc::channel();
        thread::spawn(move || {
            let mut work = start();
            while let Ok(mut want) = wants.recv() {
                // Only the last asked for.
                while let Ok(later) = wants.try_recv() {
                    want = later;
                }
                if found.send(work(want)).is_err() {
                    return;
                }
                notify();
            }
        });
        Latest {
            tx,
            rx,
            asked: None,
        }
    }

    /// The ask `ask` makes wanted, unless `is` says the last asked for is
    /// it (so a want made every frame asks once, making nothing).
    pub fn want(&mut self, is: impl FnOnce(&Ask) -> bool, ask: impl FnOnce() -> Ask) {
        if self.asked.as_ref().is_some_and(is) {
            return;
        }
        let ask = ask();
        self.asked = Some(ask.clone());
        let _ = self.tx.send(ask);
    }

    /// What was found since the last call, if anything was.
    pub fn take(&self) -> Option<Found> {
        self.rx.try_recv().ok()
    }
}

/// Work for background threads to wait on: what is to be done (`W`), under
/// a lock, and a stop. Every change wakes every thread waiting, and a stop
/// ends each at its next wait. A lock left poisoned by a thread that
/// panicked is taken over as it is.
pub struct Queue<W> {
    work: Mutex<W>,
    ready: Condvar,
    stop: AtomicBool,
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
    pub fn next<T>(&self, mut take: impl FnMut(&mut W) -> Option<T>) -> Option<T> {
        self.next_by(|work| take(work).ok_or(None))
    }

    /// [`Queue::next`], `take` saying when to ask it again where there is
    /// nothing to take (`Err`): at every change, and at the time given too
    /// (asked again at once where that time has passed, so a `take` gives
    /// a time to come); `None` once stopped.
    pub fn next_by<T>(
        &self,
        mut take: impl FnMut(&mut W) -> Result<T, Option<Instant>>,
    ) -> Option<T> {
        let mut work = self.lock();
        loop {
            if self.stopped() {
                return None;
            }
            let until = match take(&mut work) {
                Ok(found) => return Some(found),
                Err(until) => until,
            };
            let now = Instant::now();
            work = match until {
                Some(until) if until > now => {
                    let (work, _) = self
                        .ready
                        .wait_timeout(work, until - now)
                        .unwrap_or_else(PoisonError::into_inner);
                    work
                }
                Some(_) => work,
                None => self
                    .ready
                    .wait(work)
                    .unwrap_or_else(PoisonError::into_inner),
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

    /// `item`, waiting ahead, wanted now: moved to the end of what is
    /// wanted, before what waits ahead. Nothing where it does not wait
    /// ahead.
    pub fn hurry(&mut self, item: &T)
    where
        T: PartialEq,
    {
        if let Some(k) = self.ahead.iter().position(|t| t == item) {
            if let Some(t) = self.ahead.remove(k) {
                self.now.push_back(t);
            }
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

    /// A wait until a time ends there (asked again no sooner); a change
    /// before ends it early, the time not come yet.
    #[test]
    fn a_wait_until_a_time_ends_there_or_at_a_change_before() {
        let queue = Arc::new(Queue::new(0u32));
        let due = Instant::now() + Duration::from_millis(50);
        let took = queue.next_by(|_| (Instant::now() >= due).then_some(()).ok_or(Some(due)));
        assert!(took.is_some() && Instant::now() >= due);
        let far = Instant::now() + Duration::from_secs(600);
        let asked = Arc::clone(&queue);
        let changer = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            asked.change(|w| *w = 1);
        });
        let early = queue.next_by(|w| match *w {
            1 => Ok(Instant::now() >= far),
            _ => Err(Some(far)),
        });
        assert_eq!(early, Some(false), "a change before: not due");
        changer.join().unwrap();
    }

    #[test]
    fn a_stop_ends_a_pause_and_what_is_there_is_taken_until_then() {
        let queue = Queue::new(Some(1u32));
        assert_eq!(queue.next(Option::take), Some(1));
        queue.change(|w| *w = Some(2));
        queue.stop();
        assert_eq!(queue.next(Option::take), None);
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

    /// One waiting ahead, wanted now: on to the end of the wanted, before
    /// what waits ahead; one that waits nowhere ahead stays as it is.
    #[test]
    fn one_hurried_goes_from_ahead_to_the_end_of_the_wanted() {
        let mut lanes = Lanes::default();
        lanes.prepare([1, 2, 3]);
        lanes.want([4]);
        lanes.hurry(&2);
        lanes.hurry(&4);
        lanes.hurry(&9);
        let order: Vec<(i32, Lane)> = std::iter::from_fn(|| lanes.pop()).collect();
        assert_eq!(
            order,
            [
                (4, Lane::Now),
                (2, Lane::Now),
                (1, Lane::Ahead),
                (3, Lane::Ahead)
            ]
        );
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

    /// Asked while it works, it does only the last ask after; an ask the
    /// same as the last is asked once; each found told.
    #[test]
    fn latest_does_only_the_last_asked_meanwhile() {
        let (gate, held) = mpsc::channel::<()>();
        let (begun, working) = mpsc::channel::<()>();
        let (told, heard) = mpsc::channel::<()>();
        let notify: Notify = Arc::new(move || {
            let _ = told.send(());
        });
        let mut latest = Latest::new(
            move || {
                move |ask: u32| {
                    if ask == 1 {
                        begun.send(()).unwrap();
                        held.recv().unwrap();
                    }
                    ask * 10
                }
            },
            notify,
        );
        let wait = Duration::from_secs(10);
        latest.want(|&a| a == 1, || 1);
        working.recv_timeout(wait).unwrap();
        latest.want(|&a| a == 2, || 2);
        latest.want(|&a| a == 3, || 3);
        latest.want(|&a| a == 3, || unreachable!("asked once in a row"));
        gate.send(()).unwrap();
        heard.recv_timeout(wait).unwrap();
        heard.recv_timeout(wait).unwrap();
        let found: Vec<u32> = std::iter::from_fn(|| latest.take()).collect();
        assert_eq!(found, [10, 30]);
    }
}
