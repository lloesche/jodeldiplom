//! A store's writer: a thread of its own that does the store's jobs in
//! the order asked, one at a time, and says each one that fails with
//! [`Action::Failed`] ([`Writer`]).

use std::sync::mpsc::{self, Sender};
use std::thread;

use crate::core::Action;
use crate::work::outbox::Outbox;

/// Does a store's jobs (`J`) on a thread of its own: in the order asked,
/// one at a time. A job that fails is said to the outbox; the thread
/// stops once the app is gone, or the writer is.
pub struct Writer<J> {
    jobs: Sender<J>,
}

impl<J: Send + 'static> Writer<J> {
    /// A writer doing each job with `run`, which gives what a failed job
    /// says to `outbox`.
    pub fn new(
        outbox: Outbox,
        mut run: impl FnMut(J) -> Result<(), String> + Send + 'static,
    ) -> Writer<J> {
        let (jobs, rx) = mpsc::channel::<J>();
        thread::spawn(move || {
            for job in rx {
                if let Err(text) = run(job) {
                    if !outbox.send(Action::Failed(text)) {
                        return;
                    }
                }
            }
        });
        Writer { jobs }
    }

    /// Asks for `job`, done after those asked before it.
    pub fn send(&self, job: J) {
        let _ = self.jobs.send(job);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    /// The jobs are done in the order asked; those that fail are said, in
    /// that order, and the others are not.
    #[test]
    fn jobs_are_done_in_order_and_failures_said() {
        let (tx, rx) = mpsc::channel();
        let writer = Writer::new(Outbox::new(tx, Arc::new(|| {})), |n: u32| {
            if n.is_multiple_of(2) {
                Ok(())
            } else {
                Err(format!("job {n}"))
            }
        });
        for n in 0..6 {
            writer.send(n);
        }
        drop(writer);
        let said: Vec<String> = rx
            .iter()
            .map(|a| match a {
                Action::Failed(text) => text,
                _ => panic!("only failures are said"),
            })
            .collect();
        assert_eq!(said, ["job 1", "job 3", "job 5"]);
    }
}
