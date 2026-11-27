//! The GPU's stages: a song's preparation goes by model, not by song.
//!
//! Each model the background work runs has a queue of its own (a
//! [`Stage`]): the separator of the stems in the background, the beat
//! network, the speech recogniser, the speech model that times the
//! words, and the cover's depth and cut-out. A job waits in the queues of
//! the models it needs, first in first out (the song playing next goes
//! before no other), one job a model at a time, and runs on the thread
//! that queued it once its turn comes ([`run`]).
//!
//! The models stay on the card as its memory allows ([`plan`]): the
//! budget is read through the backend ([`jodel_gpu::memory`]), and a
//! model is loaded beside the others while what is loaded, by each
//! model's footprint ([`jodel_gpu::Resident`], worked out from its shapes
//! before it is loaded), fits the budget less a margin. The plan is made
//! before loading, never found out by failing (Windows pages video memory
//! out silently, and everything crawls). Where the next model would not
//! fit, the others not in use are let go (the one used longest ago
//! first); one at work, or with a job queued before, is never let go: a
//! model drains what was queued before another takes its room, and a job
//! queued meanwhile waits until then. Once every queue has been empty for
//! a minute ([`plan::IDLE`]), every model is let go, on any card: the
//! memory goes back to the game and the other programs.
//!
//! Singing takes the GPU first: while a song is sung ([`sing`]) nothing
//! new is loaded but for that song's own jobs (a thread working for it
//! says so: [`for_the_song_sung`], or a job: [`run_for_the_song_sung`]):
//! a model's upload, and the memory it takes, would come out of the
//! game's frames. The jobs of the models
//! already loaded give way to the window's frames as all background work
//! does ([`jodel_gpu::pace`]).
//!
//! One thread plans ([`Stages::global`]'s): it reads the budget, makes the
//! plan, lets models go with no lock held, and only then starts the jobs
//! granted. Queuing a job, ending one and singing only tell it.
//!
//! # A stage's job
//!
//! - **The models it needs:** its stages, each one model ([`run`]). The
//!   models register themselves where they are made ([`register`]), each
//!   a [`jodel_gpu::Resident`]: its footprint, and its release. A stage
//!   nobody registered plans as taking nothing.
//! - **Its input and output:** what the job's closure takes and gives
//!   back; it runs on the caller's thread (a thread marked for background
//!   work: [`jodel_gpu::pace::background`]).
//! - **A model is loaded inside the job** (on its first use there), never
//!   outside a turn, so that nothing goes on the card the plan did not
//!   make room for.
//! - **One song's whole work for the model** (every chunk, every piece): a
//!   model is let go only between jobs.
//! - **Never one turn inside another:** a job that queues another job on
//!   its own thread panics ([`run`]); on a card where the two do not fit,
//!   neither would ever run. (A job waiting for another thread's job is
//!   not caught: do not.) A song's next stage is queued once this one is
//!   done.

pub mod plan;
#[cfg(test)]
mod tests;

use std::cell::Cell;
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Once, OnceLock, PoisonError};
use std::time::Duration;

use jodel_gpu::memory::Budget;
use jodel_gpu::{Footprint, Resident};

use crate::clock::AppClock;
use plan::Plan;

/// A model's queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    /// The separator of the stems in the background (an analysis's, the
    /// lyrics' voice, the singer's phrases), not the track playing's.
    Stems,
    /// The beat network of an analysis.
    Beats,
    /// The speech recogniser hearing the voice: the words checked against
    /// it, or heard where none were found.
    Hearing,
    /// The speech model timing the words against the voice.
    Timing,
    /// The cover's depth and cut-out.
    Cover,
}

impl Stage {
    /// Every stage.
    pub const ALL: [Stage; 5] = [
        Stage::Stems,
        Stage::Beats,
        Stage::Hearing,
        Stage::Timing,
        Stage::Cover,
    ];
}

thread_local! {
    /// The turns this thread holds ([`run`]).
    static TURNS: Cell<usize> = const { Cell::new(0) };
    /// Whether this thread works for the song being sung
    /// ([`for_the_song_sung`]).
    static SUNG: Cell<bool> = const { Cell::new(false) };
}

/// Runs `job`, which needs the models of `stages`, on this thread once its
/// turn comes in the process's stages ([`Stages::global`]), `what` naming
/// it in the trace; what it gives back.
///
/// # Panics
/// Inside another job's turn: a turn inside a turn could wait for ever.
pub fn run<T>(stages: &[Stage], what: &str, job: impl FnOnce() -> T) -> T {
    Stages::global().run(stages, what, job)
}

/// [`run`], the job for the song being sung (it loads its models while
/// the song is sung), whatever its thread does otherwise.
pub fn run_for_the_song_sung<T>(stages: &[Stage], what: &str, job: impl FnOnce() -> T) -> T {
    Stages::global().run_as(stages, what, true, job)
}

/// `model` is `stage`'s in the process's stages from now on (in place of
/// any registered before).
pub fn register(stage: Stage, model: Arc<dyn Resident>) {
    Stages::global().register(stage, model);
}

/// A song is sung in the process until what this gives goes ([`Singing`]).
pub fn sing() -> Singing<'static> {
    Stages::global().sing()
}

/// Marks the calling thread as working for the song being sung: its jobs
/// load their models while it is sung.
pub fn for_the_song_sung() {
    SUNG.set(true);
}

/// The plan, the models it plans for, and what the planner is told.
struct Inner {
    plan: Plan,
    models: [Option<Arc<dyn Resident>>; Stage::ALL.len()],
    /// The songs being sung ([`Stages::sing`]).
    singers: usize,
    /// Something changed since the planner last planned.
    dirty: bool,
    /// The tickets whose jobs start (their models' room made).
    started: Vec<u64>,
}

/// The GPU's stages (see the module's doc): their queues, and the models
/// on the card. The process has one ([`Stages::global`]); a test makes
/// its own on a stepped clock and a budget of its own, and plans it by
/// hand (`plan_once`).
pub struct Stages {
    inner: Mutex<Inner>,
    changed: Condvar,
    clock: AppClock,
    memory: Box<dyn Fn() -> Option<Budget> + Send + Sync>,
}

impl Stages {
    /// Stages timed by `clock`, the card's budget read by `memory`, with no
    /// planner yet.
    pub fn new(
        clock: AppClock,
        memory: impl Fn() -> Option<Budget> + Send + Sync + 'static,
    ) -> Stages {
        Stages {
            inner: Mutex::new(Inner {
                plan: Plan::default(),
                models: Default::default(),
                singers: 0,
                dirty: false,
                started: Vec::new(),
            }),
            changed: Condvar::new(),
            clock,
            memory: Box::new(memory),
        }
    }

    /// The process's stages: the card's budget read through its backend,
    /// on the wall clock, planned by a thread of their own (the planner:
    /// `plan_once` each time it is told of a change, or once the
    /// idle deadline is due).
    ///
    /// # Panics
    /// Where the planner's thread cannot be made (then on every call, so
    /// that no job waits for a planner that is not there).
    pub fn global() -> &'static Stages {
        static STAGES: OnceLock<Stages> = OnceLock::new();
        static PLANNER: Once = Once::new();
        let stages =
            STAGES.get_or_init(|| Stages::new(AppClock::default(), jodel_gpu::memory::budget));
        PLANNER.call_once(|| {
            std::thread::Builder::new()
                .name("gpu stages".into())
                .spawn(move || stages.plan_for_ever())
                .expect("a thread for the GPU's stages");
        });
        stages
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `model` is `stage`'s from now on, in place of one registered
    /// before: the plan neither counts nor lets go of that one any more
    /// (its holder does, as it is dropped). The app registers each once.
    ///
    /// Its footprint is worked out at once on a thread of its own (a
    /// model's may take a second or two: weights packed, checkpoints read),
    /// so the first plan that asks for it finds it known, or ready soon.
    pub fn register(&self, stage: Stage, model: Arc<dyn Resident>) {
        self.lock().models[stage as usize] = Some(Arc::clone(&model));
        let warmed = std::thread::Builder::new()
            .name("gpu footprint".into())
            .spawn(move || {
                if std::panic::catch_unwind(AssertUnwindSafe(|| model.footprint())).is_err() {
                    crate::trace::note(format_args!("stages: {stage:?}'s footprint panicked"));
                }
            });
        if warmed.is_err() {
            crate::trace::note(format_args!(
                "stages: {stage:?}'s footprint left to the plan"
            ));
        }
    }

    /// [`run`] in these stages.
    pub fn run<T>(&self, stages: &[Stage], what: &str, job: impl FnOnce() -> T) -> T {
        self.run_as(stages, what, SUNG.get(), job)
    }

    /// [`run`], for the song being sung where `sung`.
    fn run_as<T>(&self, stages: &[Stage], what: &str, sung: bool, job: impl FnOnce() -> T) -> T {
        assert!(
            TURNS.get() == 0,
            "{what}: a turn on the GPU inside another job's turn"
        );
        let queued = self.clock.now();
        let mut inner = self.lock();
        let ticket = inner.plan.queue(stages, sung);
        self.tell(&mut inner);
        loop {
            if let Some(at) = inner.started.iter().position(|&t| t == ticket) {
                inner.started.swap_remove(at);
                break;
            }
            inner = self
                .changed
                .wait(inner)
                .unwrap_or_else(PoisonError::into_inner);
        }
        drop(inner);
        TURNS.set(TURNS.get() + 1);
        let _turn = Turn {
            stages: self,
            ticket,
        };
        let waited = self.clock.since(queued);
        if waited >= Duration::from_secs(1) {
            crate::trace::note(format_args!(
                "stages: {stages:?} for {what} waited {:.1} s for its turn",
                waited.as_secs_f64()
            ));
        }
        job()
    }

    /// Someone sings until what this gives goes: while anyone does,
    /// nothing new goes on the card but for the jobs for the song.
    pub fn sing(&self) -> Singing<'_> {
        let mut inner = self.lock();
        inner.singers += 1;
        if inner.singers == 1 {
            inner.plan.sing(true);
            self.tell(&mut inner);
        }
        Singing { stages: self }
    }

    /// Whether `stage`'s model may be on the card now.
    #[cfg(test)]
    fn loaded(&self, stage: Stage) -> bool {
        self.lock().plan.loaded(stage)
    }

    /// The planner told of a change.
    fn tell(&self, inner: &mut Inner) {
        inner.dirty = true;
        self.changed.notify_all();
    }

    /// The plan made once: every model's footprint and the card's budget
    /// read, the plan stepped, the models it lets go released with no lock
    /// held, and only then the jobs it grants started. Only the planner
    /// plans (and a test, by hand): two could let a model go that the
    /// other's grant is loading. A model whose footprint panics is told in
    /// the trace and taken as not known ([`jodel_gpu::UNKNOWN`]: it runs
    /// alone); one whose release panics, as let go. A stage nobody
    /// registered takes nothing.
    fn plan_once(&self) {
        let models = {
            let mut inner = self.lock();
            inner.dirty = false;
            inner.models.clone()
        };
        let declared: Vec<Footprint> = (Stage::ALL.into_iter().zip(models.each_ref()))
            .map(|(stage, m)| {
                m.as_ref().map_or(Footprint::default(), |m| {
                    std::panic::catch_unwind(AssertUnwindSafe(|| m.footprint())).unwrap_or_else(
                        |_| {
                            crate::trace::note(format_args!(
                                "stages: {stage:?}'s footprint panicked"
                            ));
                            jodel_gpu::UNKNOWN
                        },
                    )
                })
            })
            .collect();
        let budget = (self.memory)();
        let step = {
            let mut inner = self.lock();
            for (stage, footprint) in Stage::ALL.into_iter().zip(declared) {
                inner.plan.declare(stage, footprint);
            }
            inner.plan.step(budget, self.clock.now())
        };
        // From the step to the grants, nothing that can panic but the
        // releases (caught): a grant lost there would wait for ever.
        for stage in step.released {
            if let Some(model) = &models[stage as usize] {
                crate::trace::note(format_args!("stages: {stage:?}'s model let go"));
                if std::panic::catch_unwind(AssertUnwindSafe(|| model.release())).is_err() {
                    crate::trace::note(format_args!("stages: {stage:?}'s release panicked"));
                }
            }
        }
        let mut inner = self.lock();
        inner.started.extend(step.granted);
        self.changed.notify_all();
    }

    /// The planner's loop: plans whenever told of a change, or once the
    /// idle deadline is due.
    fn plan_for_ever(&self) {
        loop {
            let mut inner = self.lock();
            loop {
                let now = self.clock.now();
                let due = inner.plan.deadline();
                if inner.dirty || due.is_some_and(|d| now >= d) {
                    break;
                }
                inner = match due {
                    Some(d) => {
                        self.changed
                            .wait_timeout(inner, d - now)
                            .unwrap_or_else(PoisonError::into_inner)
                            .0
                    }
                    None => self
                        .changed
                        .wait(inner)
                        .unwrap_or_else(PoisonError::into_inner),
                };
            }
            drop(inner);
            // A bug in the plan stops no work for good: told, and planned
            // again at the next change.
            if std::panic::catch_unwind(AssertUnwindSafe(|| self.plan_once())).is_err() {
                crate::trace::note("stages: the plan panicked");
            }
        }
    }
}

/// A song being sung ([`Stages::sing`]), until it goes.
pub struct Singing<'a> {
    stages: &'a Stages,
}

impl Drop for Singing<'_> {
    fn drop(&mut self) {
        let stages = self.stages;
        let mut inner = stages.lock();
        inner.singers -= 1;
        if inner.singers == 0 {
            inner.plan.sing(false);
            stages.tell(&mut inner);
        }
    }
}

/// A job's turn: its end (the job done, or panicked) lets the next go.
struct Turn<'a> {
    stages: &'a Stages,
    ticket: u64,
}

impl Drop for Turn<'_> {
    fn drop(&mut self) {
        TURNS.set(TURNS.get() - 1);
        let stages = self.stages;
        let mut inner = stages.lock();
        inner.plan.done(self.ticket, stages.clock.now());
        stages.tell(&mut inner);
    }
}
