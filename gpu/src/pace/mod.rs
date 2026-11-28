//! Sharing the GPU with the window: the networks' work time-sliced.
//!
//! A network run as one submission keeps the GPU to itself until it is
//! done: a 6 s chunk of stems, a 30 s piece of speech, a tenth of a second
//! or more on a laptop's GPU, and every frame the window wants meanwhile
//! waits. So every run here goes in short submissions ([`dispatch`]), a
//! few milliseconds of GPU time apiece, and the window's frames go in
//! between. The next submission is made while the last one runs, so the
//! GPU does not wait for the processor between them. The recogniser's
//! decoder steps are never split: their submissions are whole steps, one
//! at least, so a step costs one round trip at most (the language model's
//! steps, longer, may be split). A model's weights go up the same way
//! ([`Uploading`]).
//!
//! Background work gives way to the window: the beat network, the speech
//! models, the story's language and picture models, and
//! [`crate::GpuSeparator::background`] (the analyses' stems, the lyrics'
//! voice). The window tells the pace of each of its ticks ([`tick`]):
//! whether it came late (its frame, or getting it, waited for the GPU, as
//! Metal's `nextDrawable` does while the GPU is busy with other work) and
//! whether something on it moves smoothly. While something moves, and
//! until [`HOLD`] after the window was last starved of the GPU (its ticks
//! late [`STARVING`] times within a second while background work ran),
//! background work keeps to a share of the time ([`duty`]): it has one
//! submission on the GPU at most, waiting for each before it makes the
//! next, so a frame of the window never queues behind more than one, and
//! rests after it (before its next submission, its processor work done
//! meanwhile), so the GPU is idle for the window's frames and the
//! compositor's. Each late tick past the [`STARVING`]th of a second cuts
//! the share, a second of ticks on time gives some back. Where the
//! window's ticks come on time (the 5080's: the system shares its GPU
//! between the window and the work) nothing is given way for lateness.
//! Otherwise background work runs flat out. The stems of the track playing
//! ([`crate::GpuSeparator::shared`]), which have to keep up with it, never
//! rest, and background work does not run beside them: while they have
//! work on the GPU, it waits (at most [`LONGEST_WAIT`] at a time), so the
//! GPU has one network on it at once, as when they shared a separator.
//!
//! [`background`] marks a thread whose work is background work, and
//! [`crate::GpuSeparator::for_this_thread`] picks the background separator
//! there. Each network keeps a [`Tally`] of what its GPU work cost so far:
//! its submissions, the times it waited for the GPU, and how long it gave
//! way. All of this is the process's one pace, which every network's work
//! keeps to. The submissions themselves are made in [`submit`].

mod submit;

use std::cell::Cell;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

pub(crate) use submit::{dispatch, dispatch_left, finish, Ran, Track, Uploading};

thread_local! {
    static BACKGROUND: Cell<bool> = const { Cell::new(false) };
}

/// Marks the calling thread as one doing background work (see the
/// module's doc).
pub fn background() {
    BACKGROUND.with(|b| b.set(true));
}

/// Whether the calling thread's GPU work is background work.
pub fn is_background() -> bool {
    BACKGROUND.with(Cell::get)
}

/// The share of the time background work keeps to while it gives way, at
/// most.
pub const MOST: f32 = 0.75;
/// And at least.
pub const LEAST: f32 = 0.1;
/// Where it starts.
const START: f32 = 0.5;
/// Background work gives way this long after the window's last tick of
/// something moving smoothly.
const MOVING: Duration = Duration::from_millis(500);
/// This many late ticks within a second, while background work runs,
/// mean the window is starved of the GPU (each one past it cuts the
/// share): a starved window's ticks are nearly all late (on the M4 Max its
/// main thread waited for nine frames in ten), while the 5080's few late
/// ones come one or two at a time.
pub const STARVING: u64 = 4;
/// And background work gives way until this long after the window was
/// last starved, or a tick came late while it gave way: long enough that
/// it does not flap (given way, the ticks come on time; each time it stops,
/// the next few late ticks start it again).
pub const HOLD: Duration = Duration::from_secs(10);
/// Background work counts as running this long after its last submission:
/// late ticks then say it held the window up.
const WORKING: Duration = Duration::from_millis(250);
/// How long one submission should take on the GPU (and a picture model's
/// dispatch at most: `image::plan`).
pub(crate) const PIECE: Duration = Duration::from_millis(5);
/// The dispatches a run's first submission takes where nothing is learned
/// yet: one, no longer than a piece (`image::plan` cuts the long ones),
/// the next sized from its time towards [`PIECE`], at most twice the last
/// (`submit::learned`). An eighth of the run, as it was, put some 3 s of
/// klein's first picture on the M4 Max in one submission (Gemma's first
/// words 0.8 s), and the window's frames waited behind it.
pub(crate) const FIRST: usize = 1;
/// The longest rest after one.
const LONGEST_REST: Duration = Duration::from_millis(60);
/// The longest background work waits for other work at a time.
pub const LONGEST_WAIT: Duration = Duration::from_secs(1);

/// What a network's GPU work cost so far, for traces: the difference of
/// two ([`Tally::since`]) is what was done between them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    /// Submissions made ([`dispatch`]).
    pub submissions: u64,
    /// Times it waited for the GPU to finish work: its round trips; and
    /// how long they took.
    pub waits: u64,
    pub waited: Duration,
    /// The processor's time recording the submissions and making them.
    pub recorded: Duration,
    /// Rests taken to give way to the window, and their total.
    pub rests: u64,
    pub rested: Duration,
    /// How long it waited for the work that does not rest.
    pub held: Duration,
}

impl Tally {
    /// What was added since `earlier` (an earlier tally of the same
    /// network).
    pub fn since(&self, earlier: &Tally) -> Tally {
        Tally {
            submissions: self.submissions.saturating_sub(earlier.submissions),
            waits: self.waits.saturating_sub(earlier.waits),
            waited: self.waited.saturating_sub(earlier.waited),
            recorded: self.recorded.saturating_sub(earlier.recorded),
            rests: self.rests.saturating_sub(earlier.rests),
            rested: self.rested.saturating_sub(earlier.rested),
            held: self.held.saturating_sub(earlier.held),
        }
    }

    /// This and `other` together (two runs, or two networks' work).
    pub fn and(&self, other: &Tally) -> Tally {
        Tally {
            submissions: self.submissions + other.submissions,
            waits: self.waits + other.waits,
            waited: self.waited + other.waited,
            recorded: self.recorded + other.recorded,
            rests: self.rests + other.rests,
            rested: self.rested + other.rested,
            held: self.held + other.held,
        }
    }
}

impl std::fmt::Display for Tally {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} submissions ({:.3} s recording them), {} waits for the GPU ({:.3} s), {} rests ({:.1} s), {:.1} s behind the track playing",
            self.submissions,
            self.recorded.as_secs_f64(),
            self.waits,
            self.waited.as_secs_f64(),
            self.rests,
            self.rested.as_secs_f64(),
            self.held.as_secs_f64()
        )
    }
}

/// A network's [`Tally`] as it grows: shared with whoever asks for it
/// ([`Counts::tally`] never waits for the network's work), its times in µs.
#[derive(Clone, Debug, Default)]
pub(crate) struct Counts(Arc<[AtomicU64; 7]>);

impl Counts {
    const SUBMISSIONS: usize = 0;
    const WAITS: usize = 1;
    const RESTS: usize = 2;
    const RESTED: usize = 3;
    const HELD: usize = 4;
    const WAITED: usize = 5;
    const RECORDED: usize = 6;

    fn add(&self, which: usize, n: u64) {
        self.0[which].fetch_add(n, Ordering::Relaxed);
    }

    fn add_time(&self, which: usize, d: Duration) {
        self.add(which, d.as_micros() as u64);
    }

    /// The tally so far.
    pub(crate) fn tally(&self) -> Tally {
        let n = |which: usize| self.0[which].load(Ordering::Relaxed);
        Tally {
            submissions: n(Counts::SUBMISSIONS),
            waits: n(Counts::WAITS),
            waited: Duration::from_micros(n(Counts::WAITED)),
            recorded: Duration::from_micros(n(Counts::RECORDED)),
            rests: n(Counts::RESTS),
            rested: Duration::from_micros(n(Counts::RESTED)),
            held: Duration::from_micros(n(Counts::HELD)),
        }
    }
}

/// A pace that work keeps to: the window's ticks and the share of the time
/// they leave background work, the runs of work that does not rest under
/// way, and what the trace is told. The process has one
/// ([`Pace::global`]); a test makes its own.
#[derive(Debug)]
pub(crate) struct Pace {
    epoch: Instant,
    /// The window's last tick of something moving smoothly, in µs from
    /// `epoch`, plus one (0: none yet).
    moving_at: AtomicU64,
    /// The late ticks counted since the second they are counted in began
    /// (the same way), and when the window was last starved: the
    /// [`STARVING`]th late tick of a second, or a late one while it gave
    /// way.
    lates: AtomicU64,
    lates_from: AtomicU64,
    starved_at: AtomicU64,
    /// Background work's last submission, the same way.
    worked_at: AtomicU64,
    /// When the share last climbed, the same way.
    climbed_at: AtomicU64,
    /// The share, as `f32` bits.
    share: AtomicU32,
    /// Runs of work that does not rest under way now.
    live: AtomicU32,
    /// For the trace: late ticks while background work ran, those that
    /// cut the share (the window starved), how long it was held giving way
    /// for that (µs), rests taken and their total (µs), and the time
    /// background work waited for other work (µs).
    late: AtomicU64,
    starved: AtomicU64,
    starved_for: AtomicU64,
    rests: AtomicU64,
    rested: AtomicU64,
    waited: AtomicU64,
}

impl Pace {
    fn new() -> Pace {
        Pace {
            epoch: Instant::now(),
            moving_at: AtomicU64::new(0),
            lates: AtomicU64::new(0),
            lates_from: AtomicU64::new(0),
            starved_at: AtomicU64::new(0),
            worked_at: AtomicU64::new(0),
            climbed_at: AtomicU64::new(0),
            share: AtomicU32::new(START.to_bits()),
            live: AtomicU32::new(0),
            late: AtomicU64::new(0),
            starved: AtomicU64::new(0),
            starved_for: AtomicU64::new(0),
            rests: AtomicU64::new(0),
            rested: AtomicU64::new(0),
            waited: AtomicU64::new(0),
        }
    }

    /// The process's pace: the window's ticks say how it goes ([`tick`]),
    /// and every network's work keeps to it.
    fn global() -> &'static Pace {
        static PACE: OnceLock<Pace> = OnceLock::new();
        PACE.get_or_init(Pace::new)
    }

    fn now(&self) -> u64 {
        self.epoch.elapsed().as_micros() as u64 + 1
    }

    fn since(&self, at: &AtomicU64) -> Option<Duration> {
        match at.load(Ordering::Relaxed) {
            0 => None,
            t => Some(Duration::from_micros(self.now().saturating_sub(t))),
        }
    }

    /// Whether `at` is less than `long` ago.
    fn within(&self, at: &AtomicU64, long: Duration) -> bool {
        self.since(at).is_some_and(|d| d < long)
    }

    /// The share background work keeps to while it gives way.
    fn share(&self) -> f32 {
        f32::from_bits(self.share.load(Ordering::Relaxed))
    }

    fn set_share(&self, d: f32) {
        self.share
            .store(d.clamp(LEAST, MOST).to_bits(), Ordering::Relaxed);
    }

    /// See [`tick`].
    fn tick(&self, late: bool, smooth: bool) {
        let now = self.now();
        if smooth {
            self.moving_at.store(now, Ordering::Relaxed);
        }
        // Late while no background work ran (the window's own work, a file
        // being read): not what giving way would help.
        if !self.within(&self.worked_at, WORKING) {
            return;
        }
        if !late {
            if !self.within(&self.climbed_at, Duration::from_secs(1)) {
                self.set_share(self.share() + 0.05);
                self.climbed_at.store(now, Ordering::Relaxed);
            }
            return;
        }
        self.late.fetch_add(1, Ordering::Relaxed);
        if !self.within(&self.lates_from, Duration::from_secs(1)) {
            self.lates_from.store(now, Ordering::Relaxed);
            self.lates.store(0, Ordering::Relaxed);
        }
        let lates = self.lates.fetch_add(1, Ordering::Relaxed) + 1;
        let held = self.within(&self.starved_at, HOLD);
        if lates < STARVING && !held {
            return;
        }
        // Starved: held giving way from now (the time added counted).
        let until = |at: u64| at + HOLD.as_micros() as u64;
        let before = match self.starved_at.swap(now, Ordering::Relaxed) {
            0 => now,
            at => until(at).max(now),
        };
        self.starved_for
            .fetch_add(until(now) - before, Ordering::Relaxed);
        if lates >= STARVING {
            self.starved.fetch_add(1, Ordering::Relaxed);
            self.set_share(self.share() * 0.6);
            self.climbed_at.store(now, Ordering::Relaxed);
        }
    }

    /// See [`duty`].
    fn duty(&self) -> f32 {
        if self.within(&self.moving_at, MOVING) || self.within(&self.starved_at, HOLD) {
            self.share()
        } else {
            1.0
        }
    }

    /// See [`report`].
    fn report(&self) -> Report {
        let us = |at: &AtomicU64| Duration::from_micros(at.swap(0, Ordering::Relaxed));
        Report {
            late: self.late.swap(0, Ordering::Relaxed),
            starved: self.starved.swap(0, Ordering::Relaxed),
            starved_for: us(&self.starved_for),
            rests: self.rests.swap(0, Ordering::Relaxed),
            rested: us(&self.rested),
            waited: us(&self.waited),
            duty: self.duty(),
        }
    }

    /// Background work just ran on the GPU (late ticks now say it held the
    /// window up: [`tick`]).
    fn worked(&self) {
        self.worked_at.store(self.now(), Ordering::Relaxed);
    }
}

/// Counts a run of work that does not rest while it lives.
struct Live(&'static Pace);

impl Live {
    fn start(pace: &'static Pace) -> Live {
        pace.live.fetch_add(1, Ordering::AcqRel);
        Live(pace)
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        self.0.live.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Background work, before a submission: the rest it owes for the one
/// before (while it still gives way), then waits while other work is on
/// the GPU (at most [`LONGEST_WAIT`]); both counted.
fn give_way(pieces: &mut Pieces) {
    let p = pieces.pace;
    let owed = std::mem::take(&mut pieces.owed);
    if !owed.is_zero() && p.duty() < 1.0 {
        p.rests.fetch_add(1, Ordering::Relaxed);
        p.rested
            .fetch_add(owed.as_micros() as u64, Ordering::Relaxed);
        pieces.counts.add(Counts::RESTS, 1);
        pieces.counts.add_time(Counts::RESTED, owed);
        std::thread::sleep(owed);
    }
    let started = Instant::now();
    while p.live.load(Ordering::Acquire) > 0 && started.elapsed() < LONGEST_WAIT {
        std::thread::sleep(Duration::from_millis(1));
    }
    let waited = started.elapsed();
    if waited > Duration::from_micros(500) {
        p.waited
            .fetch_add(waited.as_micros() as u64, Ordering::Relaxed);
        pieces.counts.add_time(Counts::HELD, waited);
    }
}

/// A tick of the window: `late`, it came a quarter of a frame or more
/// after it was due (getting its frame waited for the GPU, or its last
/// frame was not through the GPU yet), `smooth`, something on it moves
/// smoothly (at the display's refresh). Background work gives way while
/// something moves, and while the window is starved: [`STARVING`] late
/// ticks within a second while background work runs (a model's weights
/// going up included; a tick late while none ran says nothing), each
/// cutting the share, and it holds that until [`HOLD`] after the last
/// late tick. A second of ticks on time, while background work runs, gives
/// some of the share back.
pub fn tick(late: bool, smooth: bool) {
    Pace::global().tick(late, smooth);
}

/// The share of the time background work keeps to now: [`MOST`] down to
/// [`LEAST`] while the window moves or is starved of the GPU ([`tick`]), 1
/// otherwise. Below 1, each of its submissions is waited for before the
/// next is made.
pub fn duty() -> f32 {
    Pace::global().duty()
}

/// How background work has given way since the last call, for a trace.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Report {
    /// The window's ticks late while it ran, those that found the window
    /// starved (each cut its share), and how long that held it giving way.
    pub late: u64,
    pub starved: u64,
    pub starved_for: Duration,
    /// Rests it took, and their total.
    pub rests: u64,
    pub rested: Duration,
    /// How long it waited for the work that does not rest.
    pub waited: Duration,
    /// Its share of the time now.
    pub duty: f32,
}

/// [`Report`] since the last call.
pub fn report() -> Report {
    Pace::global().report()
}

/// After a submission of background work that took `took`: the rest that
/// keeps the work to its share `duty`, owed until its next submission
/// ([`give_way`]).
fn owe(pieces: &mut Pieces, took: Duration, duty: f32) {
    if duty >= 1.0 {
        return;
    }
    let rest = took.mul_f32((1.0 - duty) / duty).min(LONGEST_REST);
    if rest > Duration::from_micros(200) {
        pieces.owed = rest;
    }
}

/// How many dispatches a submission takes, learned per run: about
/// [`PIECE`] of GPU time; whether the work is background work, which
/// rests, and the rest it owes; what its runs cost so far; and the pace it
/// keeps to (the process's).
#[derive(Clone, Debug)]
pub(crate) struct Pieces {
    /// [`FIRST`] at first, then learned from each submission's time.
    per: usize,
    rests: bool,
    owed: Duration,
    counts: Counts,
    pace: &'static Pace,
}

impl Pieces {
    /// For work that does not rest (the stems of the track playing).
    pub(crate) fn foreground() -> Pieces {
        Pieces {
            per: FIRST,
            rests: false,
            owed: Duration::ZERO,
            counts: Counts::default(),
            pace: Pace::global(),
        }
    }

    /// For background work.
    pub(crate) fn resting() -> Pieces {
        Pieces {
            rests: true,
            ..Pieces::foreground()
        }
    }

    /// For the calling thread's work ([`is_background`]).
    pub(crate) fn for_this_thread() -> Pieces {
        if is_background() {
            Pieces::resting()
        } else {
            Pieces::foreground()
        }
    }

    /// What its runs cost, as it grows.
    pub(crate) fn counts(&self) -> Counts {
        self.counts.clone()
    }

    /// What its runs cost so far.
    pub(crate) fn tally(&self) -> Tally {
        self.counts.tally()
    }

    /// Where the submission from dispatch `start` of a run of `len` ends:
    /// as many as it takes now, in whole steps of `whole` dispatches
    /// (`Recorded::whole`).
    pub(crate) fn end(&self, start: usize, whole: usize, len: usize) -> usize {
        (start + self.per.next_multiple_of(whole)).min(len)
    }

    /// `n` dispatches (of a run of `all`) took `took`: how many the next
    /// submissions take, learned from it ([`submit::learned`]).
    pub(crate) fn learn(&mut self, n: usize, all: usize, took: Duration) {
        self.per = submit::learned(self.per, n, all, took);
    }

    /// Before a submission: background work gives way ([`give_way`]).
    fn before(&mut self) {
        if self.rests {
            give_way(self);
        }
    }

    /// After one that took `took`: background work counts as running, and
    /// owes the rest that keeps it to its share now ([`owe`]).
    fn after(&mut self, took: Duration) {
        if self.rests {
            self.pace.worked();
            let duty = self.pace.duty();
            owe(self, took, duty);
        }
    }

    /// The share it keeps to now: background work's, 1 for work that does
    /// not rest.
    fn duty(&self) -> f32 {
        if self.rests {
            self.pace.duty()
        } else {
            1.0
        }
    }
}

/// What a run of a decoding model is, each kind paced on its own (how
/// many dispatches make a submission of a few milliseconds is learned
/// for each: a prompt's are heavier than a step's).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    /// The prompt (the recogniser's encoder with it).
    Prompt,
    /// A batch of steps.
    Steps,
}

/// The pacing of each [`Phase`] of a decoding model's runs (the
/// recogniser's, the language model's), both background work.
pub(crate) struct Paces {
    pub(crate) prompt: Pieces,
    pub(crate) steps: Pieces,
}

impl Paces {
    /// Both resting.
    pub(crate) fn resting() -> Paces {
        Paces {
            prompt: Pieces::resting(),
            steps: Pieces::resting(),
        }
    }

    /// The pacing of `phase`'s runs.
    pub(crate) fn of(&mut self, phase: Phase) -> &mut Pieces {
        match phase {
            Phase::Prompt => &mut self.prompt,
            Phase::Steps => &mut self.steps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `pieces` on a pace of their own, so that no other test's work (nor
    /// the window's ticks) moves it, its clock started a while ago (so
    /// that "long ago" is on it).
    pub(super) fn alone(pieces: Pieces) -> (&'static Pace, Pieces) {
        let mut pace = Pace::new();
        pace.epoch -= Duration::from_secs(60);
        let pace: &'static Pace = Box::leak(Box::new(pace));
        (pace, Pieces { pace, ..pieces })
    }

    /// Late ticks while no background work runs, or fewer than
    /// [`STARVING`] in a second, give nothing way; that many while it runs
    /// cut the share and hold background work to it, as while something
    /// moves; a late tick while it gives way holds it longer, and [`HOLD`]
    /// after the last it runs flat out again. The trace is told.
    #[test]
    fn background_work_gives_way_to_a_window_starved_of_the_gpu() {
        let (p, _) = alone(Pieces::resting());
        let ago = |d: Duration| p.now() - d.as_micros() as u64;
        assert_eq!(p.duty(), 1.0, "ticks on time");
        for _ in 0..2 * STARVING {
            p.tick(true, false);
        }
        assert_eq!(p.duty(), 1.0, "no background work ran");
        p.worked();
        for _ in 1..STARVING {
            p.tick(true, false);
        }
        assert_eq!(p.duty(), 1.0, "fewer than STARVING: hiccups");
        // As many more, but in the next second: still hiccups.
        p.lates_from
            .store(ago(Duration::from_secs(2)), Ordering::Relaxed);
        for _ in 1..STARVING {
            p.tick(true, false);
        }
        assert_eq!(p.duty(), 1.0);
        p.tick(true, false);
        assert!((p.duty() - START * 0.6).abs() < 1e-6, "{}", p.duty());
        // Each late tick past it cuts again, down to the least.
        for _ in 0..20 {
            p.tick(true, false);
        }
        assert_eq!(p.duty(), LEAST);
        // Given way, a late tick alone holds it: it does not flap.
        p.starved_at.store(ago(HOLD / 2), Ordering::Relaxed);
        p.lates_from.store(ago(HOLD / 2), Ordering::Relaxed);
        p.tick(true, false);
        assert!(
            p.within(&p.starved_at, Duration::from_secs(1)),
            "held from now"
        );
        p.starved_at
            .store(ago(HOLD - Duration::from_secs(1)), Ordering::Relaxed);
        assert_eq!(p.duty(), LEAST);
        p.starved_at
            .store(ago(HOLD + Duration::from_secs(1)), Ordering::Relaxed);
        assert_eq!(p.duty(), 1.0, "HOLD after the last late tick");
        let report = p.report();
        assert_eq!(report.late, (STARVING - 1) * 2 + 1 + 20 + 1);
        assert_eq!(report.starved, 21);
        assert!(report.starved_for >= HOLD, "{report:?}");
    }

    #[test]
    fn the_share_falls_on_late_ticks_and_climbs_back() {
        let (p, mut pieces) = alone(Pieces::resting());
        p.set_share(0.5);
        // No background work: late ticks change nothing.
        p.tick(true, true);
        assert_eq!(p.share(), 0.5);
        // With it: a few late ticks in a second cut the share, down to the
        // least.
        p.worked();
        for _ in 0..STARVING {
            p.tick(true, true);
        }
        assert!((p.share() - 0.3).abs() < 1e-6, "{}", p.share());
        for _ in 0..20 {
            p.tick(true, true);
        }
        assert_eq!(p.share(), LEAST);
        // Something moves: background work keeps to the share.
        assert_eq!(p.duty(), LEAST);
        // A tick on time, with no change for a second, gives some back.
        p.climbed_at.store(0, Ordering::Relaxed);
        p.tick(false, true);
        assert!(p.share() > LEAST);
        // The rest owed keeps to the share: about 0.85 / 0.15 of the work,
        // taken before the next submission.
        owe(&mut pieces, Duration::from_millis(4), p.duty());
        let started = Instant::now();
        pieces.before();
        let rested = started.elapsed();
        assert!(rested >= Duration::from_millis(12), "{rested:?}");
        let tally = pieces.counts().tally();
        assert_eq!(tally.rests, 1);
        assert!(tally.rested >= Duration::from_millis(12), "{tally:?}");
        // Taken once.
        let started = Instant::now();
        pieces.before();
        assert!(started.elapsed() < Duration::from_millis(10));
        // While nothing moves and the window is not starved, none.
        owe(&mut pieces, Duration::from_millis(20), p.duty());
        p.moving_at.store(0, Ordering::Relaxed);
        p.starved_at.store(0, Ordering::Relaxed);
        assert_eq!(p.duty(), 1.0);
        let started = Instant::now();
        pieces.before();
        assert!(started.elapsed() < Duration::from_millis(10));
        assert_eq!(pieces.counts().tally().rests, 1);
        // What the trace is told: the late ticks and the rest.
        let report = p.report();
        assert_eq!((report.late, report.rests), (STARVING + 20, 1));
    }

    /// Background work waits while work that does not rest is under way,
    /// and goes on once it ends.
    #[test]
    fn background_work_waits_for_the_work_that_does_not_rest() {
        let (pace, mut pieces) = alone(Pieces::resting());
        let started = Instant::now();
        pieces.before();
        assert!(
            started.elapsed() < Duration::from_millis(50),
            "nothing to wait for"
        );
        let live = Live::start(pace);
        let ended = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(80));
            drop(live);
        });
        let started = Instant::now();
        pieces.before();
        let waited = started.elapsed();
        ended.join().unwrap();
        assert!(waited >= Duration::from_millis(60), "{waited:?}");
        assert!(waited < LONGEST_WAIT, "{waited:?}");
        let held = pieces.counts().tally().held;
        assert!(held >= Duration::from_millis(60), "{held:?}");
        assert_eq!(pace.live.load(Ordering::Acquire), 0);
    }
}
