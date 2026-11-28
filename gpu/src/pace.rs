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
//! steps, longer, may be split).
//!
//! Background work gives way while the window draws at the display's
//! refresh, something on it moving smoothly ([`frame`] says so, each such
//! frame; calm frames, a few dozen a second at most, leave the GPU plenty
//! of time and say nothing): the beat network, the speech
//! models, and [`crate::GpuSeparator::background`] (the analyses' stems,
//! the lyrics' voice). It then waits for each of its submissions and rests
//! after it (before its next submission, its processor work done
//! meanwhile), keeping to a share of the time ([`duty`]) that falls when
//! the window's frames come late and climbs back while they are on time;
//! otherwise it runs flat out. The stems of the track playing
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
//! keeps to.

use std::cell::Cell;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use crate::engine::{Kernels, Program, Recorded};
use crate::profile::Stamps;

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
/// Background work gives way this long after the window's last frame at
/// the display's refresh.
const MOVING: Duration = Duration::from_millis(500);
/// How long one submission should take on the GPU (and a picture model's
/// dispatch at most: `image::plan`).
pub(crate) const PIECE: Duration = Duration::from_millis(5);
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

/// A pace that work keeps to: the window's frames and the share of the
/// time they leave background work, the runs of work that does not rest
/// under way, and what the trace is told. The process has one
/// ([`Pace::global`]); a test makes its own.
#[derive(Debug)]
pub(crate) struct Pace {
    epoch: Instant,
    /// The window's last frame at the refresh, in µs from `epoch`, plus one
    /// (0: none yet).
    frame_at: AtomicU64,
    /// Background work's last submission, the same way.
    worked_at: AtomicU64,
    /// When the share last climbed, the same way.
    climbed_at: AtomicU64,
    /// The share, as `f32` bits.
    share: AtomicU32,
    /// Runs of work that does not rest under way now.
    live: AtomicU32,
    /// Late frames while background work ran, and rests taken, and their
    /// total in µs (for the trace).
    late: AtomicU64,
    rests: AtomicU64,
    rested: AtomicU64,
    /// Time background work waited for other work, in µs.
    waited: AtomicU64,
}

impl Pace {
    fn new() -> Pace {
        Pace {
            epoch: Instant::now(),
            frame_at: AtomicU64::new(0),
            worked_at: AtomicU64::new(0),
            climbed_at: AtomicU64::new(0),
            share: AtomicU32::new(START.to_bits()),
            live: AtomicU32::new(0),
            late: AtomicU64::new(0),
            rests: AtomicU64::new(0),
            rested: AtomicU64::new(0),
            waited: AtomicU64::new(0),
        }
    }

    /// The process's pace: the window's frames say how it goes
    /// ([`frame`]), and every network's work keeps to it.
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

    /// The share background work keeps to while the window draws at the
    /// refresh.
    fn share(&self) -> f32 {
        f32::from_bits(self.share.load(Ordering::Relaxed))
    }

    fn set_share(&self, d: f32) {
        self.share
            .store(d.clamp(LEAST, MOST).to_bits(), Ordering::Relaxed);
    }

    /// See [`frame`].
    fn frame(&self, late: bool) {
        let now = self.now();
        self.frame_at.store(now, Ordering::Relaxed);
        let working = self
            .since(&self.worked_at)
            .is_some_and(|d| d < Duration::from_millis(250));
        if !working {
            return;
        }
        if late {
            self.late.fetch_add(1, Ordering::Relaxed);
            self.set_share(self.share() * 0.6);
            self.climbed_at.store(now, Ordering::Relaxed);
        } else if self
            .since(&self.climbed_at)
            .is_none_or(|d| d >= Duration::from_secs(1))
        {
            self.set_share(self.share() + 0.05);
            self.climbed_at.store(now, Ordering::Relaxed);
        }
    }

    /// See [`duty`].
    fn duty(&self) -> f32 {
        if self.since(&self.frame_at).is_some_and(|d| d < MOVING) {
            self.share()
        } else {
            1.0
        }
    }

    /// See [`report`].
    fn report(&self) -> Report {
        Report {
            late: self.late.swap(0, Ordering::Relaxed),
            rests: self.rests.swap(0, Ordering::Relaxed),
            rested: Duration::from_micros(self.rested.swap(0, Ordering::Relaxed)),
            waited: Duration::from_micros(self.waited.swap(0, Ordering::Relaxed)),
            duty: self.duty(),
        }
    }

    /// Background work just ran on the GPU (late frames now cut its share:
    /// [`frame`]).
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
/// before (while the window still draws at the refresh), then waits while
/// other work is on the GPU (at most [`LONGEST_WAIT`]); both counted.
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

/// The window drew a frame at the display's refresh, something on it
/// moving smoothly (calm frames say nothing); `late`: its last frame was
/// not through the GPU yet (it missed its turn, the GPU busy with other
/// work). A late frame while background work runs cuts that work's share;
/// a second of frames on time gives some back.
pub fn frame(late: bool) {
    Pace::global().frame(late);
}

/// The share of the time background work keeps to now: [`MOST`] down to
/// [`LEAST`] while the window draws at the display's refresh, 1 while it
/// does not.
pub fn duty() -> f32 {
    Pace::global().duty()
}

/// How background work has given way since the last call, for a trace.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Report {
    /// Frames late while it ran.
    pub late: u64,
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

/// A device's submissions as its runs' timing needs them (a program's,
/// [`Program::track`]): each numbered as it is made, from 1; the run's
/// last left running, while no run was made after it and it is not
/// finished; the highest number waited for; and when the last run left
/// running whose time was its own was found done.
#[derive(Default)]
pub(crate) struct Track {
    submitted: AtomicU64,
    left: AtomicU64,
    waited: AtomicU64,
    found: Mutex<Option<Instant>>,
}

impl Track {
    /// The number of the submission just made.
    fn number(&self) -> u64 {
        self.submitted.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// The submission `number`, its run's last, is left running.
    fn leave(&self, number: u64) {
        self.left.store(number, Ordering::Release);
    }

    /// A run starts: whether a run left running before it may still run
    /// (its first wait then waits for that too, and says nothing of its own
    /// time).
    fn take_left(&self) -> bool {
        self.left.swap(0, Ordering::AcqRel) != 0
    }

    /// The submission `number` is waited for.
    fn waiting(&self, number: u64) {
        self.waited.fetch_max(number, Ordering::AcqRel);
    }

    /// The run left running as `number` is finished: the next run's first
    /// wait is its own.
    fn finishing(&self, number: u64) {
        let _ = self
            .left
            .compare_exchange(number, 0, Ordering::AcqRel, Ordering::Acquire);
    }

    /// Where the time of the run left running as `number`, its clock
    /// started at `from` by the run, is its own (no submission made after
    /// it was waited for yet, else that wait outlasted it, and its own wait
    /// ends at once): when its clock starts, `from` or when the run left
    /// running before it was found done, if later (a decoding model makes
    /// its next run while the last one runs, and finishes that one after);
    /// else `None`.
    fn start(&self, number: u64, from: Instant) -> Option<Instant> {
        if self.waited.load(Ordering::Acquire) >= number {
            return None;
        }
        let found = *self.found.lock().unwrap_or_else(PoisonError::into_inner);
        Some(found.map_or(from, |f| f.max(from)))
    }

    /// A run left running, its time its own, was found done `at`.
    fn found_done(&self, at: Instant) {
        *self.found.lock().unwrap_or_else(PoisonError::into_inner) = Some(at);
    }
}

/// A run's submissions made ([`dispatch_left`]): its last, and where that
/// was left running, how to time it ([`finish`]).
pub(crate) struct Ran {
    pub(crate) last: wgpu::SubmissionIndex,
    pub(crate) left: Option<Left>,
}

/// How to time a run whose last submission was left running: that one's
/// number ([`Track`]), its dispatches (with those of the one before it
/// where that was still running), the run's, and when the GPU could start
/// them as the run saw it (when they were made, or when the one before
/// them was done, if later).
pub(crate) struct Left {
    number: u64,
    n: usize,
    all: usize,
    from: Instant,
}

/// Waits until the GPU has done `last`, the submission that
/// [`dispatch_left`] left running on `program` (timed as `left` says),
/// counted in `pieces`' tally (those it ran with). Where its time is its
/// own ([`Track::start`]), it sizes the next submissions, and when it was
/// found done starts the next run's clock.
pub(crate) fn finish<K: Kernels>(
    pieces: &mut Pieces,
    program: &Program<K>,
    left: Left,
    last: &wgpu::SubmissionIndex,
) -> Result<(), String> {
    let track = &program.track;
    track.finishing(left.number);
    let start = track.start(left.number, left.from);
    wait(&pieces.counts, program, last.clone(), left.number)?;
    if let Some(from) = start {
        let now = Instant::now();
        track.found_done(now);
        pieces.per = learned(pieces.per, left.n, left.all, now - from);
    }
    Ok(())
}

/// Towards [`PIECE`] a submission, smoothly, from `per` dispatches, where
/// `n` (of a run of `all`) took `took` (at least one, none where there are
/// none).
fn learned(per: usize, n: usize, all: usize, took: Duration) -> usize {
    let ideal = n as f64 * PIECE.as_secs_f64() / took.as_secs_f64().max(1e-5);
    let next = (per as f64 * 0.5 + ideal * 0.5).round() as usize;
    next.clamp(1, all.max(1))
}

/// Waits until the GPU has done the submission `index` (numbered `number`)
/// of `program`, counted in `counts` and in its [`Track`].
fn wait<K>(
    counts: &Counts,
    program: &Program<K>,
    index: wgpu::SubmissionIndex,
    number: u64,
) -> Result<(), String> {
    counts.add(Counts::WAITS, 1);
    program.track.waiting(number);
    let started = Instant::now();
    let done = program
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        })
        .map(|_| ())
        .map_err(|e| e.to_string());
    counts.add_time(Counts::WAITED, started.elapsed());
    done
}

/// How many dispatches a submission takes, learned per run: about
/// [`PIECE`] of GPU time; whether the work is background work, which
/// rests, and the rest it owes; what its runs cost so far; and the pace it
/// keeps to (the process's).
#[derive(Clone, Debug)]
pub(crate) struct Pieces {
    /// Unknown yet (0): an eighth of a run's dispatches at first.
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
            per: 0,
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

    /// What its runs cost, as it grows.
    pub(crate) fn counts(&self) -> Counts {
        self.counts.clone()
    }

    /// What its runs cost so far.
    pub(crate) fn tally(&self) -> Tally {
        self.counts.tally()
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

/// Runs `program`'s `recorded` dispatches in submissions of
/// about [`PIECE`] each (of whole steps where they are the recogniser's
/// decoder steps, `Recorded::whole`: such a step is never split, so it is
/// one submission where it takes longer than that), `prelude` before the
/// first and `finish` after the last (one submission of the two where
/// `ops` is empty), and waits until they are done (the caller then reads
/// their result:
/// [`crate::device::read_back`]). Each submission is made while the one
/// before runs, which is waited for then; but while background work
/// (`pieces.rests`) gives way to the window ([`duty`] below 1), each is
/// waited for at once, and the rest it owes is taken before the next
/// ([`give_way`]).
pub(crate) fn dispatch<K: Kernels, B>(
    program: &Program<K>,
    pieces: &mut Pieces,
    recorded: &Recorded<K, B>,
    (prelude, finish): (
        impl FnOnce(&mut wgpu::CommandEncoder),
        impl FnOnce(&mut wgpu::CommandEncoder),
    ),
) -> Result<(), String> {
    submissions(program, pieces, recorded, (prelude, finish), false).map(|_| ())
}

/// [`dispatch`] with no prelude, returning while its last submission may
/// still run, for [`finish`]: the caller does more meanwhile. The [`Ran`]
/// holds that last submission, after which what it wrote can be read
/// without waiting for work sent later ([`crate::device::read_after`]),
/// and what was left running (none where it was waited for: given way,
/// or profiled).
pub(crate) fn dispatch_left<K: Kernels, B>(
    program: &Program<K>,
    pieces: &mut Pieces,
    recorded: &Recorded<K, B>,
    finish: impl FnOnce(&mut wgpu::CommandEncoder),
) -> Result<Ran, String> {
    let prelude = |_: &mut wgpu::CommandEncoder| {};
    submissions(program, pieces, recorded, (prelude, finish), true)
}

/// What becomes of a submission once it is made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Left running: the run's last, nothing waiting for it (nor for the
    /// one before, where that still runs).
    Leave,
    /// Waited for at once (with the one before): the run's last, or work
    /// giving way to the window, or profiled.
    Wait,
    /// Running while the next is made: the one before is waited for.
    Pipeline,
}

impl Mode {
    /// The mode of a submission, the run's `last` or not, of a run that
    /// may `leave` its last running, at the share `duty`, `profiled` or
    /// not.
    fn of(last: bool, leave: bool, duty: f32, profiled: bool) -> Mode {
        let alone = duty < 1.0 || profiled;
        match (last, alone) {
            (true, false) if leave => Mode::Leave,
            (true, _) | (_, true) => Mode::Wait,
            (false, false) => Mode::Pipeline,
        }
    }
}

/// A run's submissions as they go: the one still running (its index, its
/// number, its dispatches and when it was made), when the one before it
/// was done, and whether its first wait may also wait for work left
/// running before (so it does not say how long this run's work takes).
struct Timing {
    running: Option<(wgpu::SubmissionIndex, u64, usize, Instant)>,
    done_at: Option<Instant>,
    after_left: bool,
}

impl Timing {
    /// A submission of `n` dispatches made at `made`, taken with the one
    /// before where that still runs: their dispatches together, and when
    /// the first of them was made.
    fn with_running(&mut self, n: usize, made: Instant) -> (usize, Instant) {
        match self.running.take() {
            Some((_, _, before, at)) => (before + n, at),
            None => (n, made),
        }
    }

    /// The GPU time of submissions made at `at`, done now: from then, or
    /// from when the one before them was done, if later.
    fn took(&mut self, at: Instant) -> Duration {
        let now = Instant::now();
        let took = now - self.done_at.map_or(at, |d| d.max(at));
        self.done_at = Some(now);
        took
    }
}

/// [`dispatch`]'s and [`dispatch_left`]'s submissions: where it may
/// `leave` the last running, the run's [`Left`] when it does.
fn submissions<K: Kernels, B>(
    program: &Program<K>,
    pieces: &mut Pieces,
    recorded: &Recorded<K, B>,
    (prelude, finish): (
        impl FnOnce(&mut wgpu::CommandEncoder),
        impl FnOnce(&mut wgpu::CommandEncoder),
    ),
    leave: bool,
) -> Result<Ran, String> {
    let (device, queue, label) = (&program.device, &program.queue, program.label());
    let (groups, ops) = (recorded.groups(), recorded.dispatches());
    if pieces.per == 0 {
        pieces.per = ops.len().div_ceil(8).max(1);
    }
    let _live = (!pieces.rests).then(|| Live::start(pieces.pace));
    let (mut prelude, mut finish) = (Some(prelude), Some(finish));
    let mut start = 0;
    let mut timing = Timing {
        running: None,
        done_at: None,
        after_left: program.track.take_left(),
    };
    // Whole steps a submission, where the dispatches are the recogniser's
    // decoder steps (`Recorded::whole`).
    let whole = recorded.whole;
    loop {
        let end = (start + pieces.per.next_multiple_of(whole)).min(ops.len());
        let last = end == ops.len();
        let recording = Instant::now();
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        if let Some(prelude) = prelude.take() {
            prelude(&mut encoder);
        }
        let stamps = Stamps::new(
            device,
            label,
            ops[start..end].iter().map(|d| (d.kernel.entry(), d.groups)),
        );
        program.encode(&mut encoder, groups, &ops[start..end], stamps.as_ref());
        if last {
            if let Some(finish) = finish.take() {
                finish(&mut encoder);
            }
        }
        let commands = encoder.finish();
        let recorded = recording.elapsed();
        if pieces.rests {
            give_way(pieces);
        }
        pieces.counts.add(Counts::SUBMISSIONS, 1);
        let made = Instant::now();
        let index = queue.submit([commands]);
        let number = program.track.number();
        // Where it is the run's last, what the run reads back is read once
        // it is done.
        let written = last.then(|| index.clone());
        pieces
            .counts
            .add_time(Counts::RECORDED, recorded + made.elapsed());
        let n = end - start;
        let duty = if pieces.rests {
            pieces.pace.duty()
        } else {
            1.0
        };
        // The submission to wait for now: this one with the one before, or
        // the one before while this one runs; none where this one is left
        // running.
        let waited = match Mode::of(last, leave, duty, stamps.is_some()) {
            Mode::Leave => {
                program.track.leave(number);
                let (n, at) = timing.with_running(n, made);
                let left = Left {
                    number,
                    n,
                    all: ops.len(),
                    from: timing.done_at.map_or(at, |d| d.max(at)),
                };
                return Ok(Ran {
                    last: index,
                    left: Some(left),
                });
            }
            Mode::Wait => {
                let (n, at) = timing.with_running(n, made);
                Some((index, number, n, at))
            }
            Mode::Pipeline => timing.running.replace((index, number, n, made)),
        };
        if let Some((index, number, n, at)) = waited {
            wait(&pieces.counts, program, index, number)?;
            if let Some(stamps) = stamps {
                stamps.add(device, queue)?;
            }
            let took = timing.took(at);
            // Towards PIECE a submission, smoothly (at least one, none
            // where there are none).
            if !std::mem::take(&mut timing.after_left) {
                pieces.per = learned(pieces.per, n, ops.len(), took);
            }
            if pieces.rests {
                pieces.pace.worked();
                owe(pieces, took, duty);
            }
        }
        if let Some(last) = written {
            return Ok(Ran { last, left: None });
        }
        start = end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `pieces` on a pace of their own, so that no other test's work (nor
    /// the window's frames) moves it.
    fn alone(pieces: Pieces) -> (&'static Pace, Pieces) {
        let pace: &'static Pace = Box::leak(Box::new(Pace::new()));
        (pace, Pieces { pace, ..pieces })
    }

    /// No kernels: a program for dispatches of none.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum NoKernel {}

    impl Kernels for NoKernel {
        const ALL: &'static [NoKernel] = &[];
        const PARAM_BYTES: u64 = 48;
        const EXTRA: &'static [crate::engine::Access] = &[];
        fn entry(self) -> &'static str {
            match self {}
        }
    }

    /// A submission's size goes halfway towards [`PIECE`] of GPU time, both
    /// ways, within the run.
    #[test]
    fn the_size_goes_halfway_towards_a_piece() {
        let ms = Duration::from_millis;
        // 100 dispatches in 10 ms: 50 would take PIECE.
        assert_eq!(learned(100, 100, 400, ms(10)), 75);
        // In 2 ms: 250 would.
        assert_eq!(learned(100, 100, 400, ms(2)), 175);
        // Never past the run, never none.
        assert_eq!(learned(400, 400, 400, ms(1)), 400);
        assert_eq!(learned(1, 1, 400, Duration::from_secs(10)), 1);
    }

    /// A run left running, finished after the next run was made (as a
    /// decoding model does), times itself only where no submission made
    /// after it was waited for first, its clock from the later of its own
    /// start and when the run left running before it was found done.
    #[test]
    fn a_left_run_is_timed_only_where_its_wait_is_its_own() {
        let track = Track::default();
        let t0 = Instant::now();
        let later = |ms| t0 + Duration::from_millis(ms);
        // A run of one submission, left running.
        let a = track.number();
        track.leave(a);
        // The next run, one submission too, made before `a` is finished:
        // its first wait would have waited for `a`, and it waits for none.
        assert!(track.take_left());
        let b = track.number();
        track.leave(b);
        track.finishing(a);
        assert_eq!(track.start(a, t0), Some(t0), "nothing found done before");
        track.waiting(a);
        track.found_done(later(7));
        // `b` started on the GPU once `a` was done, not when it was made.
        track.finishing(b);
        assert_eq!(track.start(b, later(1)), Some(later(7)));
        // A start later than that is its own.
        assert_eq!(track.start(b, later(9)), Some(later(9)));
        // A run whose next run waited inside itself before it was
        // finished: that wait outlasted it, its own says nothing.
        assert!(!track.take_left(), "b was finished");
        let c = track.number();
        track.leave(c);
        assert!(track.take_left());
        let d0 = track.number();
        let d1 = track.number();
        track.waiting(d0);
        track.leave(d1);
        track.finishing(c);
        assert_eq!(track.start(c, later(10)), None);
    }

    #[test]
    fn the_share_falls_on_late_frames_and_climbs_back() {
        let (p, mut pieces) = alone(Pieces::resting());
        p.set_share(0.5);
        // No background work: late frames change nothing.
        p.worked_at.store(0, Ordering::Relaxed);
        p.frame(true);
        assert_eq!(p.share(), 0.5);
        // With it: a late frame cuts the share, down to the least.
        p.worked_at.store(p.now(), Ordering::Relaxed);
        p.frame(true);
        assert!((p.share() - 0.3).abs() < 1e-6, "{}", p.share());
        for _ in 0..20 {
            p.frame(true);
        }
        assert_eq!(p.share(), LEAST);
        // The window draws at the refresh: background work keeps to the
        // share.
        assert_eq!(p.duty(), LEAST);
        // A frame on time, with no change for a second, gives some back.
        p.climbed_at.store(0, Ordering::Relaxed);
        p.frame(false);
        assert!(p.share() > LEAST);
        // The rest owed keeps to the share: about 0.85 / 0.15 of the work,
        // taken before the next submission.
        owe(&mut pieces, Duration::from_millis(4), p.duty());
        let started = Instant::now();
        give_way(&mut pieces);
        let rested = started.elapsed();
        assert!(rested >= Duration::from_millis(12), "{rested:?}");
        let tally = pieces.counts().tally();
        assert_eq!(tally.rests, 1);
        assert!(tally.rested >= Duration::from_millis(12), "{tally:?}");
        // Taken once.
        let started = Instant::now();
        give_way(&mut pieces);
        assert!(started.elapsed() < Duration::from_millis(10));
        // While the window does not draw at the refresh, none.
        owe(&mut pieces, Duration::from_millis(20), p.duty());
        p.frame_at.store(0, Ordering::Relaxed);
        assert_eq!(p.duty(), 1.0);
        let started = Instant::now();
        give_way(&mut pieces);
        assert!(started.elapsed() < Duration::from_millis(10));
        assert_eq!(pieces.counts().tally().rests, 1);
        // What the trace is told: the late frames and the rest.
        let report = p.report();
        assert_eq!((report.late, report.rests), (21, 1));
    }

    /// Background work waits while work that does not rest is under way,
    /// and goes on once it ends.
    #[test]
    fn background_work_waits_for_the_work_that_does_not_rest() {
        let (pace, mut pieces) = alone(Pieces::resting());
        let started = Instant::now();
        give_way(&mut pieces);
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
        give_way(&mut pieces);
        let waited = started.elapsed();
        ended.join().unwrap();
        assert!(waited >= Duration::from_millis(60), "{waited:?}");
        assert!(waited < LONGEST_WAIT, "{waited:?}");
        let held = pieces.counts().tally().held;
        assert!(held >= Duration::from_millis(60), "{held:?}");
        assert_eq!(pace.live.load(Ordering::Acquire), 0);
    }

    /// An empty list of dispatches is one submission of the prelude and the
    /// finish, for work that rests and work that does not; skipped (with a
    /// note) where there is no usable GPU.
    #[test]
    fn an_empty_list_of_dispatches_runs_its_prelude_and_finish() {
        let program = crate::device::open("jodel-gpu pace", |device, queue, limits| {
            Ok(Program::<NoKernel>::new(
                device,
                queue,
                limits,
                "jodel-gpu pace",
                "",
            ))
        });
        let Ok((program, _)) = program else {
            eprintln!("no usable GPU: skipped");
            return;
        };
        let program = crate::Turned::new(program);
        let values = program.storage("jodel-gpu pace values", 4).unwrap();
        let staging = program.staging(4);
        for (_, mut pieces) in [alone(Pieces::resting()), alone(Pieces::foreground())] {
            program
                .queue
                .write_buffer(&values, 0, &crate::engine::bytes(&[1.0f32; 4]));
            dispatch(
                &program,
                &mut pieces,
                &program.recording::<()>(0),
                (
                    |encoder: &mut wgpu::CommandEncoder| encoder.clear_buffer(&values, 0, None),
                    |encoder: &mut wgpu::CommandEncoder| {
                        encoder.copy_buffer_to_buffer(&values, 0, &staging, 0, 16)
                    },
                ),
            )
            .unwrap();
            let read = crate::device::read_back(&program.device, &staging, 4).unwrap();
            assert_eq!(read, [0.0; 4], "the prelude, then the finish");
            assert_eq!(pieces.tally().submissions, 1);
            assert_eq!(pieces.per, 1);
            assert_eq!(pieces.pace.live.load(Ordering::Acquire), 0);
        }
    }
}
