//! The plan the stages keep to ([`Plan`]): each stage's queue, which
//! models are on the card, and what to let go, decided on the processor
//! from the models' footprints and the card's budget, the time given (no
//! threads, no clock of its own).

use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};

use jodel_gpu::memory::Budget;
use jodel_gpu::Footprint;

use super::Stage;

/// How long every queue stays empty (nothing waiting, nothing running)
/// before every model is let go.
pub const IDLE: Duration = Duration::from_secs(60);

/// What is kept free of the budget for what grows on the card beside the
/// stages' models without asking them: the track playing's separator
/// (0.84 GB at its peak, while it loads, as measured with the budget) and
/// the window's pictures.
pub const MARGIN: u64 = 1 << 30;

/// A stage as the plan keeps it.
#[derive(Debug, Default)]
struct Slot {
    /// The tickets waiting, first in first out.
    waiting: VecDeque<u64>,
    /// The ticket whose job runs now.
    running: Option<u64>,
    /// What its model takes.
    footprint: Footprint,
    /// Whether its model may be on the card (its weights counted).
    loaded: bool,
    /// When a job of it last started or ended.
    used: Option<Instant>,
}

impl Slot {
    /// What it holds on the card, as planned: its weights while loaded, and
    /// its working buffers while a job runs.
    fn held(&self) -> u64 {
        let weights = if self.loaded {
            self.footprint.resident
        } else {
            0
        };
        let working = if self.running.is_some() {
            self.footprint.working
        } else {
            0
        };
        weights.saturating_add(working)
    }

    /// Nothing runs, and nothing queued before `ticket` waits: it may be
    /// let go to make room for `ticket`.
    fn unused_before(&self, ticket: u64) -> bool {
        self.running.is_none() && self.waiting.iter().all(|&t| t > ticket)
    }
}

/// A job queued: the stages it needs (one model each), whether it is for
/// the song being sung, and whether it runs.
#[derive(Debug)]
struct Ticket {
    stages: Vec<Stage>,
    sung: bool,
    started: bool,
}

/// What a step decided: the stages whose models go (to be carried out
/// before the jobs granted start), and the tickets whose jobs start.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Step {
    /// The stages whose models go.
    pub released: Vec<Stage>,
    /// The tickets whose jobs start.
    pub granted: Vec<u64>,
}

/// The stages' queues and the card's memory as planned.
///
/// - A job runs once it is first in each of its stages' queues and none
///   of them runs another; jobs start in the order they were queued, and
///   once one waits for room on the card, no job queued after it starts
///   before it (so a stage that keeps being asked for cannot hold another
///   off for ever).
/// - A model is loaded beside the others while what is loaded fits the
///   budget less [`MARGIN`]; where the next would not fit, the models not
///   in use go (the one used longest ago first), never one at work or
///   with a job queued before: a model drains what was queued before the
///   swap was asked for, and a job queued meanwhile waits until then. A
///   model left alone on the card goes on even where it does not fit
///   (there is no better plan); so does one whose footprint is not known
///   yet (said as the most there is: `jodel_gpu::Kept`'s), which nothing
///   is loaded beside.
/// - Weights count while a model is loaded, its working buffers while a
///   job of it runs.
/// - While a song is sung nothing new goes on the card, but for the jobs
///   for that song; a job held back by it holds no queue's front, so the
///   song's own jobs behind it go on.
/// - Every model goes once every queue has been empty for [`IDLE`].
#[derive(Debug, Default)]
pub struct Plan {
    slots: [Slot; Stage::ALL.len()],
    tickets: BTreeMap<u64, Ticket>,
    /// The last ticket given.
    last: u64,
    singing: bool,
    /// Since when every queue is empty.
    idle_since: Option<Instant>,
    /// What the process held on the card while none of the stages' models
    /// were loaded (the window, the track playing's stems), read after the
    /// releases planned before were carried out: the least counted beside
    /// them, where a model planned is not on the card yet.
    floor: u64,
}

impl Plan {
    fn slot(&mut self, stage: Stage) -> &mut Slot {
        &mut self.slots[stage as usize]
    }

    /// `stage`'s model takes `footprint`.
    pub fn declare(&mut self, stage: Stage, footprint: Footprint) {
        self.slot(stage).footprint = footprint;
    }

    /// A job needing `stages` queued (`sung`: for the song being sung): its
    /// ticket.
    pub fn queue(&mut self, stages: &[Stage], sung: bool) -> u64 {
        self.last += 1;
        let ticket = self.last;
        for &stage in stages {
            self.slot(stage).waiting.push_back(ticket);
        }
        let stages = stages.to_vec();
        self.tickets.insert(
            ticket,
            Ticket {
                stages,
                sung,
                started: false,
            },
        );
        self.idle_since = None;
        ticket
    }

    /// `ticket`'s job ended at `now`.
    pub fn done(&mut self, ticket: u64, now: Instant) {
        let Some(t) = self.tickets.remove(&ticket) else {
            return;
        };
        debug_assert!(t.started, "only a job running ends");
        for stage in t.stages {
            let slot = self.slot(stage);
            slot.running = None;
            slot.used = Some(now);
        }
        if self.tickets.is_empty() {
            self.idle_since = Some(now);
        }
    }

    /// Someone sings now, or no more.
    pub fn sing(&mut self, on: bool) {
        self.singing = on;
    }

    /// Whether `stage`'s model may be on the card now.
    #[cfg(test)]
    pub fn loaded(&self, stage: Stage) -> bool {
        self.slots[stage as usize].loaded
    }

    /// The jobs waiting for `stage`.
    #[cfg(test)]
    pub fn waiting(&self, stage: Stage) -> usize {
        self.slots[stage as usize].waiting.len()
    }

    /// When every model is let go, if nothing is queued before: a while
    /// after the last job, while some model is loaded.
    pub fn deadline(&self) -> Option<Instant> {
        let loaded = self.slots.iter().any(|s| s.loaded);
        self.idle_since.filter(|_| loaded).map(|since| since + IDLE)
    }

    /// What every stage holds on the card, as planned.
    pub fn held(&self) -> u64 {
        (self.slots.iter().map(Slot::held)).fold(0, u64::saturating_add)
    }

    /// The plan at `now` with the card's `budget` as told, read after the
    /// last step's releases were carried out (none: nothing to plan by,
    /// every model fits).
    pub fn step(&mut self, budget: Option<Budget>, now: Instant) -> Step {
        let mut step = Step::default();
        let held = self.held();
        if let (0, Some(b)) = (held, budget) {
            self.floor = b.usage;
        }
        // What the stages' models may hold in all: the budget less the
        // margin and what else the process holds (what is told beyond the
        // plan, at least the floor).
        let room = budget.map(|b| {
            let other = b.usage.saturating_sub(held).max(self.floor);
            b.budget.saturating_sub(MARGIN).saturating_sub(other)
        });
        if self.deadline().is_some_and(|due| now >= due) {
            for stage in Stage::ALL {
                let slot = self.slot(stage);
                if slot.loaded {
                    slot.loaded = false;
                    step.released.push(stage);
                }
            }
        }
        let waiting: Vec<u64> = (self.tickets.iter())
            .filter(|(_, t)| !t.started)
            .map(|(&k, _)| k)
            .collect();
        // The jobs singing holds back: they hold no queue's front.
        let mut barred: Vec<u64> = Vec::new();
        for ticket in waiting {
            let stages = self.tickets[&ticket].stages.clone();
            let first = stages.iter().all(|&s| {
                let slot = &self.slots[s as usize];
                let front = slot.waiting.iter().find(|t| !barred.contains(t));
                slot.running.is_none() && front == Some(&ticket)
            });
            if !first {
                continue;
            }
            let loads = stages.iter().any(|&s| !self.slots[s as usize].loaded);
            if loads && self.singing && !self.tickets[&ticket].sung {
                barred.push(ticket);
                continue;
            }
            let need = (stages.iter().map(|&s| &self.slots[s as usize]))
                .map(|s| {
                    let weights = if s.loaded { 0 } else { s.footprint.resident };
                    s.footprint.working.saturating_add(weights)
                })
                .fold(0, u64::saturating_add);
            match self.room_for(ticket, &stages, need, room) {
                Some(released) => step.released.extend(released),
                // Nothing queued after a job waiting for room starts.
                None => break,
            }
            for &stage in &stages {
                let slot = self.slot(stage);
                slot.loaded = true;
                // Not always the front: a job held back by singing may be
                // before it.
                slot.waiting.retain(|&t| t != ticket);
                slot.running = Some(ticket);
                slot.used = Some(now);
            }
            if let Some(t) = self.tickets.get_mut(&ticket) {
                t.started = true;
            }
            step.granted.push(ticket);
        }
        step
    }

    /// Room for `need` more bytes for `ticket`'s `stages` within `room`:
    /// the stages let go to make it (none where it fits as it is), carried
    /// out in the plan; `None` where it cannot be made yet. The models not
    /// in use before `ticket` go, the one used longest ago first; `stages`
    /// left alone on the card go on even where they do not fit.
    fn room_for(
        &mut self,
        ticket: u64,
        stages: &[Stage],
        need: u64,
        room: Option<u64>,
    ) -> Option<Vec<Stage>> {
        let Some(room) = room else {
            return Some(Vec::new());
        };
        let mut held = self.held();
        let fits = |held: u64| held.saturating_add(need) <= room;
        let mut unused: Vec<Stage> = (Stage::ALL.into_iter())
            .filter(|s| !stages.contains(s))
            .filter(|&s| self.slots[s as usize].loaded)
            .filter(|&s| self.slots[s as usize].unused_before(ticket))
            .collect();
        unused.sort_by_key(|&s| self.slots[s as usize].used);
        let mut released = Vec::new();
        for s in unused {
            if fits(held) {
                break;
            }
            held = held.saturating_sub(self.slots[s as usize].held());
            released.push(s);
        }
        let own =
            (stages.iter().map(|&s| self.slots[s as usize].held())).fold(0, u64::saturating_add);
        if !fits(held) && held != own {
            return None;
        }
        for &s in &released {
            self.slot(s).loaded = false;
        }
        Some(released)
    }
}
