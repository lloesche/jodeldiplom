//! Models found gone or damaged while the player runs ([`Repair`]).
//!
//! At its start setup takes a model by its files' lengths and their stamps
//! alone (a stat each, no hashing): a file changed since it was checked,
//! or of another length, is checked or got again then. While the player
//! runs, a model that cannot be read (its file gone, or not read as its
//! format) is told to the repair ([`Unreadable`]); one worker, on a thread
//! of its own, takes the models told one at a time (so the stamps and the
//! disk's room are one writer's): it hashes the model's files against
//! their built-in checksums, and gets again only those wrong or missing,
//! as setup gets them, into the data folder where it is now (a move under
//! way followed: [`crate::store::mover::current`]). The user is told in a
//! line as that begins and as it ends (a download's progress is not shown
//! on the way), and the model's loader tries it again at once
//! ([`Unreadable::retries`]). A model whole was let down by something
//! else (the GPU, say): nothing is said, and it is not looked at again
//! this run while it stays as it was. One that could not be got again
//! rests [`REST`] before it is tried again (its files not hashed again:
//! this run's stamps hold), one whose source sends another file is given
//! up until the next start ([`super::Failure::lasting`]).
//!
//! A model's file is read mapped while it loads (`jodel_tensors::Bytes`),
//! and repaired by a rename over it (`durable`), never written in place:
//! a mapping goes on reading the file it mapped. A file that is short is
//! caught before any load maps it (setup's check of every file's length
//! at the start, then the reader's of its header's places). A disk read
//! error under a load's mapping, though (a bad sector, a drive unplugged
//! or a network folder gone), or on Linux and macOS a file truncated in
//! place by another program, ends the process (`EXCEPTION_IN_PAGE_ERROR`,
//! `SIGBUS`): it is not repairable while the player runs.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use super::{
    attempt, capital, check_need, forget, look, Failure, Lookup, Need, Paths, Places, Status, Work,
};
use crate::store::mover;
use crate::work::Job;
use jodel_gpu::{Kept, OnDevice};

/// What tells the repair of a model that cannot be read, by its path in
/// the store ([`Need::file`]); [`Unreadable::none`] tells nothing (a
/// player without setup: the examples, the tests).
#[derive(Clone, Default)]
pub struct Unreadable(Option<Arc<Repair>>);

impl Unreadable {
    /// One that tells nothing.
    pub fn none() -> Unreadable {
        Unreadable(None)
    }

    /// `made` (the model of `file` read and set up), the repair told of
    /// `file` where it failed.
    pub fn report<T>(&self, file: &str, made: Result<T, String>) -> Result<T, String> {
        if made.is_err() {
            self.told(file);
        }
        made
    }

    /// Whether the model of `file` is at `path`; the repair told of it
    /// where it is gone.
    pub fn present(&self, file: &str, path: &Path) -> bool {
        let there = path.is_file();
        if !there {
            self.told(file);
        }
        there
    }

    /// `kept`, the loader of the model of `file`, tried again at once each
    /// time the model is got again ([`Kept::retry_now`]), not after its
    /// failed setup's while. The repair only pokes it (it holds it weakly):
    /// the loader holds what tells the repair, never the other way round.
    pub fn retries<T: OnDevice + Send + 'static>(&self, file: &str, kept: &Arc<Kept<T>>) {
        let kept = Arc::downgrade(kept);
        self.when_got(file, move || {
            if let Some(kept) = kept.upgrade() {
                kept.retry_now();
            }
        });
    }

    /// `again` run each time the model of `file` is got again.
    fn when_got(&self, file: &str, again: impl Fn() + Send + Sync + 'static) {
        if let Some(repair) = &self.0 {
            let mut got = locked(&repair.shared.got);
            got.entry(file.to_owned())
                .or_default()
                .push(Box::new(again));
        }
    }

    fn told(&self, file: &str) {
        if let Some(repair) = &self.0 {
            repair.report(file);
        }
    }
}

/// What the user is told: a line for the status bar, as information or
/// (`Err`) as a failure.
pub type Tell = Box<dyn Fn(Result<String, String>) + Send + Sync>;

/// How long a model that could not be got again rests before it is tried
/// again (each failed read of it meanwhile asks nothing).
pub const REST: Duration = Duration::from_secs(5 * 60);

/// Where a model told stands this run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum State {
    /// Told the worker, or being checked.
    Busy,
    /// Its files whole (found so, or got again): not looked at again while
    /// they stay as they are (stamped).
    Whole,
    /// Its source sends another file: not tried again this run.
    GivenUp,
    /// It could not be got again then: tried again [`REST`] after, its
    /// files not hashed again.
    Failed(Instant),
}

/// The models setup got, checked and got again while the player runs:
/// what tells it ([`Repair::unreadable`]), and its one worker.
pub struct Repair {
    shared: Arc<Shared>,
    queue: Sender<(String, bool)>,
}

/// What the worker runs again with run after run of repairs.
type Again = Box<dyn Fn() + Send + Sync>;

/// What the repair and its worker share.
struct Shared {
    needs: Vec<Need>,
    /// Where setup looked and put the models at the start (a move since
    /// followed at each repair).
    places: Places,
    work: Arc<dyn Work>,
    /// Every model's path as setup found it, its loaders read it from.
    in_use: Paths,
    tell: Mutex<Option<Tell>>,
    states: Mutex<HashMap<String, State>>,
    got: Mutex<HashMap<String, Vec<Again>>>,
}

fn locked<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Repair {
    /// The models `needs`, as setup got them into `places` (each found at
    /// its path in `in_use`), got again through `work`; its worker started.
    pub fn new(
        needs: Vec<Need>,
        places: Places,
        work: Arc<dyn Work>,
        in_use: &Paths,
    ) -> Arc<Repair> {
        let (queue, told) = mpsc::channel::<(String, bool)>();
        let shared = Arc::new(Shared {
            needs,
            places,
            work,
            in_use: in_use.clone(),
            tell: Mutex::new(None),
            states: Mutex::new(HashMap::new()),
            got: Mutex::new(HashMap::new()),
        });
        let worker = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("repair".into())
            .spawn(move || {
                for (file, fresh) in told {
                    // A repair that panics fails that model alone; the
                    // worker goes on with the next.
                    let mended = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        worker.mend(&file, fresh)
                    }));
                    let state = mended.unwrap_or_else(|_| {
                        crate::trace::note(format_args!("setup: the repair of {file} panicked"));
                        State::Failed(Instant::now())
                    });
                    locked(&worker.states).insert(file, state);
                }
            })
            .expect("a thread for the repair");
        Arc::new(Repair { shared, queue })
    }

    /// What tells the user from now on (until then, the trace alone).
    pub fn tell_to(&self, tell: Tell) {
        *locked(&self.shared.tell) = Some(tell);
    }

    /// What tells it of a model that cannot be read.
    pub fn unreadable(self: &Arc<Self>) -> Unreadable {
        Unreadable(Some(Arc::clone(self)))
    }

    /// The model of `file` told: given to the worker, unless it is with it
    /// already, given up, resting, or whole and unchanged since (nothing
    /// asked of the disk but where it was found whole).
    fn report(&self, file: &str) {
        let shared = &self.shared;
        let Some(need) = shared.need(file) else {
            return;
        };
        let was = locked(&shared.states).get(file).copied();
        let fresh = match was {
            Some(State::Busy | State::GivenUp) => return,
            Some(State::Failed(at)) if at.elapsed() < REST => return,
            Some(State::Failed(_)) => false,
            // Read from the disk with no lock held.
            Some(State::Whole) if shared.checked(need) => return,
            Some(State::Whole) | None => true,
        };
        let mut states = locked(&shared.states);
        // Another report came between the two looks: it is that one's.
        if states.get(file).copied() != was {
            return;
        }
        states.insert(file.to_owned(), State::Busy);
        drop(states);
        if self.queue.send((file.to_owned(), fresh)).is_err() {
            locked(&shared.states).remove(file);
        }
    }

    /// Where the model of `file` stands (tests).
    #[cfg(test)]
    pub(super) fn state(&self, file: &str) -> Option<State> {
        locked(&self.shared.states).get(file).copied()
    }
}

impl Shared {
    fn need(&self, file: &str) -> Option<&Need> {
        self.needs.iter().find(|n| n.file == file)
    }

    /// Where the models are looked for and put now: a data folder moving,
    /// where it goes.
    fn places_now(&self) -> Places {
        Places {
            look: self.places.look.iter().map(|d| mover::current(d)).collect(),
            store: mover::current(&self.places.store),
        }
    }

    /// Whether `need` is whole and stamped where it is found now.
    fn checked(&self, need: &Need) -> bool {
        matches!(look(need, &self.places_now()), Lookup::Checked(_))
    }

    fn say(&self, line: Result<String, String>) {
        let (Ok(text) | Err(text)) = &line;
        crate::trace::note(format_args!("setup: {text}"));
        if let Some(tell) = locked(&self.tell).as_ref() {
            tell(line);
        }
    }

    /// The model of `file` checked (`fresh`: its stamps forgotten and
    /// every file hashed, else only what this run did not find right got),
    /// those of its files wrong or missing got again, the user told; where
    /// it stands then.
    fn mend(&self, file: &str, fresh: bool) -> State {
        let Some(need) = self.need(file) else {
            return State::Whole;
        };
        let places = self.places_now();
        let title = capital(&need.title);
        match self.got_again(need, &places, fresh) {
            Ok(false) => {
                crate::trace::note(format_args!(
                    "setup: {title} could not be set up, but its files are whole"
                ));
                State::Whole
            }
            Ok(true) => {
                let used = self.in_use.iter().find(|(f, _)| f == file);
                let elsewhere =
                    used.is_some_and(|(_, p)| !mover::current(p).starts_with(&places.store));
                if elsewhere {
                    // A copy outside the data folder (beside the executable)
                    // is the one its loader reads this run.
                    self.say(Ok(format!(
                        "{title} is got again into {}: it is used from the next start.",
                        places.store.display()
                    )));
                } else {
                    self.say(Ok(format!("{title} is got again and checked.")));
                    for again in locked(&self.got).get(file).into_iter().flatten() {
                        again();
                    }
                }
                State::Whole
            }
            Err(Failure {
                what,
                advice,
                lasting,
            }) => {
                let mut line = format!("{title} could not be got again: {what} {advice}");
                if lasting {
                    line += " It is not tried again until the player starts again.";
                }
                self.say(Err(line));
                if lasting {
                    State::GivenUp
                } else {
                    State::Failed(Instant::now())
                }
            }
        }
    }

    /// Whether `need` had to be got again into `places` (and was): where
    /// `fresh`, its stamps forgotten and every file hashed first (whole, it
    /// is not got); then those of its files not right got.
    fn got_again(&self, need: &Need, places: &Places, fresh: bool) -> Result<bool, Failure> {
        let job = Job::new(Status::default(), Arc::new(|| {}));
        let mut teller = job.teller(Duration::from_secs(1));
        if fresh {
            let every: Vec<PathBuf> = (places.look.iter())
                .flat_map(|d| need.expected(&need.at(d)).into_iter().map(|(p, _)| p))
                .collect();
            forget(&places.store, &every)?;
            let whole = match look(need, places) {
                Lookup::Unchecked(at) | Lookup::Checked(at) => {
                    check_need(need, &at, places, &mut teller)?
                }
                Lookup::Missing => false,
            };
            if whole {
                return Ok(false);
            }
        } else if matches!(look(need, places), Lookup::Checked(_)) {
            return Ok(false);
        }
        self.say(Ok(format!(
            "{} is missing or damaged: getting it again.",
            capital(&need.title)
        )));
        attempt(std::slice::from_ref(need), places, &*self.work, &mut teller)?;
        Ok(true)
    }
}
