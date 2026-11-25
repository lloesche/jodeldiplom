//! The mixer's helper thread, `jodel-mixer-aside`: it does what the mixer
//! thread must not, in the order it was handed over. It sets the
//! microphone's tap on the device (waiting there for the device thread to
//! take it, see `mic.rs`) and drops what the mixer thread lets go of (a
//! microphone, whose last handle waits for its cleaner; an output's
//! stream, whose ring is freed; the effects' synth replaced).
//!
//! Jobs go over a queue made at the start. One that finds it full waits in
//! a list of the mixer thread's, also made at the start, and goes on a
//! later turn ([`Aside::flush`]); so the mixer thread never drops what it
//! hands over, and allocates only past both (counted:
//! [`super::Mixer::helper_overflows`]). Handing a job over takes the
//! queue's waker lock for an instant when the helper sleeps on it.

use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::JoinHandle;

use super::desk::MicView;
use super::input::Stream;
use super::mic::Fit;
use super::MixShared;
use crate::input::Input;
use crate::shared::Shared;

/// Jobs the queue holds, and those the mixer thread keeps waiting.
const JOBS: usize = 64;
const PENDING: usize = 64;

/// What the mixer thread lets go of.
pub(super) enum Gone {
    Input(Input),
    Stream(Stream),
    /// A device's stream, once tapped.
    Device(Arc<Shared>),
    /// The effects input's live synth, replaced.
    Synth(Box<jodel_synth::Synth>),
    #[cfg(test)]
    Probe(tests::Probe),
}

pub(super) enum Job {
    DropLater(Gone),
    Tap(Fit),
}

/// The helper thread and the way to it; dropping it hands over what
/// waits, lets the helper finish and joins it.
pub(super) struct Aside {
    jobs: Option<SyncSender<Job>>,
    /// Jobs that found the queue full, oldest first.
    pending: VecDeque<Job>,
    thread: Option<JoinHandle<()>>,
    shared: Arc<MixShared>,
}

impl Aside {
    /// The helper started (on the caller's thread: this allocates);
    /// `shared` is where it says how the microphone's port stands.
    pub(super) fn new(shared: Arc<MixShared>) -> Aside {
        Aside::with_room(shared, JOBS, PENDING)
    }

    fn with_room(shared: Arc<MixShared>, jobs: usize, pending: usize) -> Aside {
        let (tx, rx) = mpsc::sync_channel(jobs);
        let theirs = shared.clone();
        let thread = std::thread::Builder::new()
            .name("jodel-mixer-aside".into())
            .spawn(move || run(rx, &theirs))
            .expect("spawn the mixer's helper thread");
        Aside {
            jobs: Some(tx),
            pending: VecDeque::with_capacity(pending),
            thread: Some(thread),
            shared,
        }
    }

    /// Lets go of `gone` on the helper thread.
    pub(super) fn drop_later(&mut self, gone: Gone) {
        self.send(Job::DropLater(gone));
    }

    /// Fits the device's tap as `fit` says, on the helper thread.
    pub(super) fn tap(&mut self, fit: Fit) {
        self.send(Job::Tap(fit));
    }

    /// Hands over what waits, in order, as far as the queue takes it.
    pub(super) fn flush(&mut self) {
        while let Some(job) = self.pending.pop_front() {
            if let Some(job) = self.try_send(job) {
                self.pending.push_front(job);
                return;
            }
        }
    }

    fn send(&mut self, job: Job) {
        self.flush();
        let job = if self.pending.is_empty() {
            match self.try_send(job) {
                None => return,
                Some(job) => job,
            }
        } else {
            job
        };
        if self.pending.len() == self.pending.capacity() {
            // It grows: an allocation here, never meant to happen.
            self.shared.overflows.fetch_add(1, Ordering::Relaxed);
        }
        self.pending.push_back(job);
    }

    /// Hands `job` over, or gives it back with the queue full.
    fn try_send(&self, job: Job) -> Option<Job> {
        let Some(jobs) = &self.jobs else {
            return Some(job);
        };
        match jobs.try_send(job) {
            Ok(()) => None,
            Err(TrySendError::Full(job)) => Some(job),
            Err(TrySendError::Disconnected(job)) => {
                debug_assert!(false, "the mixer's helper thread is gone");
                // Nowhere else to go.
                drop(job);
                None
            }
        }
    }
}

impl Drop for Aside {
    /// The mixer thread at its end, where waiting is fine.
    fn drop(&mut self) {
        if let Some(jobs) = self.jobs.take() {
            for job in self.pending.drain(..) {
                let _ = jobs.send(job);
            }
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// The helper thread: the jobs in turn, until the mixer's side is gone.
fn run(jobs: Receiver<Job>, shared: &MixShared) {
    let mut view = MicView::default();
    while let Ok(job) = jobs.recv() {
        match job {
            Job::Tap(fit) => fit.run(&mut view, shared),
            // The last handle of a microphone stops its capture and waits
            // for its cleaner; a stream's frees its ring.
            Job::DropLater(gone) => match gone {
                Gone::Input(input) => drop(input),
                Gone::Stream(stream) => drop(stream),
                Gone::Device(device) => drop(device),
                Gone::Synth(synth) => drop(synth),
                #[cfg(test)]
                Gone::Probe(probe) => drop(probe),
            },
        }
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use std::sync::mpsc::Sender;
    use std::sync::Mutex;

    /// Each probe dropped, and the thread it was dropped on.
    type Dropped = Arc<Mutex<Vec<(usize, Option<String>)>>>;

    /// Tells where it was dropped; the first one dropped waits for word.
    pub(crate) struct Probe {
        k: usize,
        started: Option<Sender<()>>,
        gate: Option<Receiver<()>>,
        dropped: Dropped,
    }

    impl Drop for Probe {
        fn drop(&mut self) {
            if let Some(started) = self.started.take() {
                let _ = started.send(());
            }
            if let Some(gate) = self.gate.take() {
                let _ = gate.recv();
            }
            let name = std::thread::current().name().map(str::to_owned);
            self.dropped.lock().unwrap().push((self.k, name));
        }
    }

    /// The helper held up: jobs fill its queue, then the mixer thread's
    /// list, then (counted) more; none is dropped on the mixer thread, and
    /// all are done in order once the helper goes on.
    #[test]
    fn a_full_queue_keeps_jobs_in_order_on_the_mixer_side() {
        let shared = Arc::new(MixShared::default());
        let mut aside = Aside::with_room(shared.clone(), 2, 4);
        let dropped = Arc::new(Mutex::new(Vec::new()));
        let (started, started_rx) = mpsc::channel();
        let (release, gate) = mpsc::channel();
        let probe = |k: usize| Probe {
            k,
            started: None,
            gate: None,
            dropped: dropped.clone(),
        };
        aside.drop_later(Gone::Probe(Probe {
            k: 0,
            started: Some(started),
            gate: Some(gate),
            dropped: dropped.clone(),
        }));
        started_rx.recv().unwrap();
        // The queue's two, the list's room, and one past it.
        let room = aside.pending.capacity();
        let n = 2 + room + 1;
        for k in 1..=n {
            aside.drop_later(Gone::Probe(probe(k)));
        }
        assert_eq!(shared.overflows.load(Ordering::Relaxed), 1);
        assert_eq!(aside.pending.len(), room + 1);
        assert!(dropped.lock().unwrap().is_empty());
        release.send(()).unwrap();
        while !aside.pending.is_empty() {
            aside.flush();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        drop(aside);
        let dropped = dropped.lock().unwrap();
        let order: Vec<usize> = dropped.iter().map(|d| d.0).collect();
        assert_eq!(order, (0..=n).collect::<Vec<_>>());
        let helper = Some("jodel-mixer-aside".to_owned());
        assert!(dropped.iter().all(|d| d.1 == helper), "{dropped:?}");
    }
}
