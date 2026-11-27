//! Where setup stands, as the setup screen reads it ([`crate::view::setup`]):
//! its steps, the one under way and its progress, why it stopped.

use std::path::PathBuf;

/// What a step's progress counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    /// Bytes (downloaded, or hashed).
    Bytes,
    /// Tensors packed.
    Tensors,
}

/// Where a step stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepState {
    /// Still to come.
    ToDo,
    /// Under way.
    Working,
    /// Done.
    Done,
}

/// A step, as the screen shows it.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    /// What it does ("Downloading the word-timing model (378 MB)").
    pub title: String,
    /// Why ("It hears where each sung word falls…").
    pub why: String,
    /// Where it stands.
    pub state: StepState,
    /// One of the player's own start ([`super::Then`]), not one that gets
    /// a file.
    pub start: bool,
    /// Its share of the whole, as a rough time.
    pub(super) weight: f64,
}

impl Step {
    /// A step that gets a file, saying `title` and `why`, in `state`, of
    /// the weight of a byte downloaded.
    pub(crate) fn new(title: String, why: String, state: StepState) -> Step {
        Step {
            title,
            why,
            state,
            start: false,
            weight: 1.0,
        }
    }
}

/// Why setup stopped, and what to do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    /// What went wrong.
    pub what: String,
    /// What to do about it.
    pub advice: String,
}

/// A failure: `what` went wrong, `advice` says what to do.
pub(super) fn failure(what: impl Into<String>, advice: impl Into<String>) -> Failure {
    Failure {
        what: what.into(),
        advice: advice.into(),
    }
}

/// A size as the setup screen says it, to the nearest: "1.88 GB",
/// "378 MB", "13 KB" (at least 1 KB).
pub fn size(bytes: u64) -> String {
    let b = bytes as f64;
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", b / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.0} MB", b / 1e6)
    } else {
        format!("{:.0} KB", (b / 1e3).max(1.0))
    }
}

/// Every needed model's path as setup found or got it, with the model's
/// path in the store ([`super::Need::file`]), in the manifest's order.
pub type Paths = Vec<(String, PathBuf)>;

/// Where setup stands, for the screen.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Status {
    /// The graphics card is chosen and the files looked for: the steps
    /// that get them are not known yet.
    pub looking: bool,
    /// Its steps, in order: those that get the files, then the player's
    /// own start's.
    pub steps: Vec<Step>,
    /// The step under way.
    pub current: Option<usize>,
    /// The step's progress: so far, in all (0: not known), counting what.
    pub done: u64,
    /// See [`Status::done`].
    pub total: u64,
    /// See [`Status::done`].
    pub unit: Option<Unit>,
    /// A download's speed, bytes a second, once it can be told.
    pub rate: Option<f64>,
    /// Why it stopped: it waits for [`super::Setup::retry`].
    pub failure: Option<Failure>,
    /// Every needed model's path, in the manifest's order: the player can
    /// start.
    pub finished: Option<Paths>,
}

impl Status {
    /// Whether any step gets a file (a download, a check of one found).
    pub fn gets_files(&self) -> bool {
        self.steps.iter().any(|s| !s.start)
    }

    /// Whether only the player's own start is still to come: nothing waits
    /// for the network or the person (the files looked for counts).
    pub fn starting(&self) -> bool {
        self.failure.is_none()
            && self.finished.is_none()
            && self
                .steps
                .iter()
                .all(|s| s.start || s.state == StepState::Done)
    }

    /// The share of the current step done (0 to 1).
    pub fn step_share(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.done as f64 / self.total as f64).clamp(0.0, 1.0)
        }
    }

    /// The share of the whole done (0 to 1).
    pub fn overall(&self) -> f64 {
        let all: f64 = self.steps.iter().map(|s| s.weight).sum();
        if all <= 0.0 {
            return if self.finished.is_some() { 1.0 } else { 0.0 };
        }
        let done: f64 = self
            .steps
            .iter()
            .enumerate()
            .map(|(i, s)| match s.state {
                StepState::Done => s.weight,
                StepState::Working if Some(i) == self.current => s.weight * self.step_share(),
                _ => 0.0,
            })
            .sum();
        (done / all).clamp(0.0, 1.0)
    }

    /// Step `current` begun (under way), or none: its progress not known
    /// yet.
    pub(super) fn begin(&mut self, current: Option<usize>) {
        self.current = current;
        if let Some(step) = current.and_then(|k| self.steps.get_mut(k)) {
            step.state = StepState::Working;
        }
        self.done = 0;
        self.total = 0;
        self.unit = None;
        self.rate = None;
    }
}
