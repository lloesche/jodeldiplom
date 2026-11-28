//! The submissions of the pace's work: a run's dispatches in pieces of
//! about [`PIECE`] ([`dispatch`], [`dispatch_left`] and [`finish`], timed
//! on each device's [`Track`]), and a model's weights going up in batches
//! of as long ([`Uploading`]), each giving way as [`super`] says.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use super::{Counts, Live, Pieces, PIECE};
use crate::engine::{Kernels, Program, Recorded};
use crate::profile::Stamps;

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

/// Halfway from `now` towards `ideal`: how a size is learned, smoothly.
fn toward(now: f64, ideal: f64) -> f64 {
    now * 0.5 + ideal * 0.5
}

/// What `amount` that took `took` would be in a piece's time ([`PIECE`]).
fn in_a_piece(amount: f64, took: Duration) -> f64 {
    amount * PIECE.as_secs_f64() / took.as_secs_f64().max(1e-5)
}

/// Towards [`PIECE`] a submission ([`toward`]), from `per` dispatches,
/// where `n` (of a run of `all`) took `took` (at least one, none where
/// there are none).
fn learned(per: usize, n: usize, all: usize, took: Duration) -> usize {
    let next = toward(per as f64, in_a_piece(n as f64, took)).round() as usize;
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

/// A batch of weights going up ([`Uploading`]): the first's bytes, and
/// the fewest and the most (wgpu stages each write until it is sent, so a
/// larger batch holds more staging at once).
const UPLOAD_FIRST: u64 = 32 << 20;
const UPLOAD_LEAST: u64 = 4 << 20;
const UPLOAD_MOST: u64 = 256 << 20;

/// A model's weights going up, paced as other work is: written in batches
/// of about a piece ([`PIECE`]) of the GPU's time copying them, their size
/// learned as they go ([`UPLOAD_FIRST`] to start, [`UPLOAD_LEAST`] to
/// [`UPLOAD_MOST`]), each sent and waited for. A background thread's give
/// way before each as its runs' submissions do, and count as its work
/// running, so a window starved while they go up is told
/// ([`super::tick`]).
pub(crate) struct Uploading {
    pieces: Pieces,
    batch: u64,
    /// The fewest and the most bytes a batch takes.
    bounds: (u64, u64),
    /// The bytes written since the last batch was sent.
    staged: u64,
}

impl Uploading {
    /// For the calling thread's work.
    pub(crate) fn new() -> Uploading {
        Uploading {
            pieces: Pieces::for_this_thread(),
            batch: UPLOAD_FIRST,
            bounds: (UPLOAD_LEAST, UPLOAD_MOST),
            staged: 0,
        }
    }

    /// For checks: batches of `batch` bytes, always.
    #[cfg(test)]
    pub(crate) fn of(batch: u64) -> Uploading {
        Uploading {
            batch,
            bounds: (batch, batch),
            ..Uploading::new()
        }
    }

    /// The bytes the batch has room for still, whole words (one at least).
    pub(crate) fn room(&self) -> u64 {
        (self.batch.saturating_sub(self.staged) & !3).max(4)
    }

    /// `n` bytes written to `program`'s queue: the batch sent where it is
    /// full.
    pub(crate) fn wrote<K: Kernels>(&mut self, program: &Program<K>, n: u64) -> Result<(), String> {
        self.staged += n;
        if self.staged >= self.batch {
            self.send(program, None)?;
        }
        Ok(())
    }

    /// The last batch sent and waited for.
    pub(crate) fn finish<K: Kernels>(&mut self, program: &Program<K>) -> Result<(), String> {
        self.send(program, None)
    }

    /// `buffer` zeroed on `program`'s GPU, a batch at a time. wgpu zeroes a
    /// buffer before its first use where nothing wrote it, all of it in
    /// the submission that first uses it (a plan's buffers, gigabytes of
    /// them, in one go); zeroed here first, they are paced as the weights
    /// are.
    pub(crate) fn clear<K: Kernels>(
        &mut self,
        program: &Program<K>,
        buffer: &wgpu::Buffer,
    ) -> Result<(), String> {
        let size = buffer.size();
        let mut at = 0;
        while at < size {
            let n = self.room().min(size - at);
            let mut encoder = program
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encoder.clear_buffer(buffer, at, Some(n));
            self.staged += n;
            self.send(program, Some(encoder.finish()))?;
            at += n;
        }
        Ok(())
    }

    /// The bytes written (and `commands`) sent and waited for, after giving
    /// way; the next batch's size learned from how long it took.
    fn send<K: Kernels>(
        &mut self,
        program: &Program<K>,
        commands: Option<wgpu::CommandBuffer>,
    ) -> Result<(), String> {
        self.pieces.before();
        let made = Instant::now();
        program.queue.submit(commands);
        program.wait()?;
        let took = made.elapsed();
        self.batch = next_batch(self.batch, self.staged, took, self.bounds);
        self.staged = 0;
        self.pieces.after(took);
        Ok(())
    }
}

/// The next batch's bytes, from `batch` where `staged` bytes took `took`:
/// towards a piece's time ([`toward`]), within `(least, most)`.
fn next_batch(batch: u64, staged: u64, took: Duration, (least, most): (u64, u64)) -> u64 {
    (toward(batch as f64, in_a_piece(staged as f64, took)) as u64).clamp(least, most)
}

/// Runs `program`'s `recorded` dispatches in submissions of about
/// [`PIECE`] each (of whole steps where they are the recogniser's decoder
/// steps, `Recorded::whole`: such a step is never split, so it is one
/// submission where it takes longer than that), `prelude` before the
/// first and `finish` after the last (one submission of the two where
/// `ops` is empty), and waits until they are done (the caller then reads
/// their result: [`crate::device::read_back`]). Each submission is made
/// while the one before runs, which is waited for then; but while
/// background work (`pieces.rests`) gives way to the window
/// ([`super::duty`] below 1), the one before is waited for before the next
/// is made (that one recorded meanwhile), so one is on the GPU at most,
/// and the rest it owes is taken before the next ([`super::give_way`]).
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
    /// Waited for at once (with the one before): the run's last, or
    /// profiled.
    Wait,
    /// Running while the next is made: waited for then (before the next is
    /// made, while giving way; after, otherwise).
    Pipeline,
}

impl Mode {
    /// The mode of a submission, the run's `last` or not, of a run that
    /// may `leave` its last running, `giving_way` or not, `profiled` or
    /// not.
    fn of(last: bool, leave: bool, giving_way: bool, profiled: bool) -> Mode {
        match (last, giving_way) {
            _ if profiled => Mode::Wait,
            (true, false) if leave => Mode::Leave,
            (true, _) => Mode::Wait,
            (false, _) => Mode::Pipeline,
        }
    }
}

/// A submission made: its index, its number, its dispatches (with those of
/// the one before it where that still ran), and when they were made.
type Made = (wgpu::SubmissionIndex, u64, usize, Instant);

/// A run's submissions as they go: the one still running, when the one
/// before it was done, and whether its first wait may also wait for work
/// left running before (so it does not say how long this run's work
/// takes).
struct Timing {
    running: Option<Made>,
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
        // Giving way: the one before, recorded after, is waited for now,
        // before this one is made.
        let giving_way = pieces.duty() < 1.0;
        if giving_way {
            if let Some(before) = timing.running.take() {
                settle(program, pieces, &mut timing, before, ops.len(), None)?;
            }
        }
        pieces.before();
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
        // The submission to wait for now: this one with the one before, or
        // the one before while this one runs; none where this one is left
        // running.
        let waited = match Mode::of(last, leave, giving_way, stamps.is_some()) {
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
        if let Some(made) = waited {
            settle(program, pieces, &mut timing, made, ops.len(), stamps)?;
        }
        if let Some(last) = written {
            return Ok(Ran { last, left: None });
        }
        start = end;
    }
}

/// A submission of a run of `all` dispatches, `made`, waited for, `stamps`
/// (where it is profiled) read: the next submissions' size learned from
/// it, and, for background work, the rest it owes ([`Pieces::after`]).
fn settle<K: Kernels>(
    program: &Program<K>,
    pieces: &mut Pieces,
    timing: &mut Timing,
    (index, number, n, at): Made,
    all: usize,
    stamps: Option<Stamps>,
) -> Result<(), String> {
    wait(&pieces.counts, program, index, number)?;
    if let Some(stamps) = stamps {
        stamps.add(&program.device, &program.queue)?;
    }
    let took = timing.took(at);
    // Towards PIECE a submission, smoothly (at least one, none where there
    // are none).
    if !std::mem::take(&mut timing.after_left) {
        pieces.per = learned(pieces.per, n, all, took);
    }
    pieces.after(took);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::alone;
    use super::*;

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

    /// A batch's size goes towards a piece's time, never past its bounds,
    /// however fast or slow a batch went.
    #[test]
    fn a_batch_stays_within_its_bounds() {
        let bounds = (UPLOAD_LEAST, UPLOAD_MOST);
        let ns = Duration::from_nanos(1);
        assert_eq!(
            next_batch(UPLOAD_FIRST, UPLOAD_MOST, ns, bounds),
            UPLOAD_MOST
        );
        let long = Duration::from_secs(10);
        assert_eq!(next_batch(UPLOAD_LEAST, 4, long, bounds), UPLOAD_LEAST);
        // 32 MB in 10 ms: 16 MB would take a piece; halfway there.
        let ms = Duration::from_millis(10);
        let next = next_batch(UPLOAD_FIRST, UPLOAD_FIRST, ms, bounds);
        assert!(next.abs_diff(24 << 20) <= 1, "{next}");
    }

    /// Giving way, a submission is made only once the one before is done
    /// (no run left running); otherwise the next is made while it runs,
    /// and a run's last is left running where it may be. Profiled, each is
    /// waited for at once.
    #[test]
    fn giving_way_has_one_submission_on_the_gpu_at_most() {
        assert_eq!(Mode::of(false, true, false, false), Mode::Pipeline);
        assert_eq!(Mode::of(true, true, false, false), Mode::Leave);
        assert_eq!(Mode::of(true, false, false, false), Mode::Wait);
        assert_eq!(Mode::of(false, true, true, false), Mode::Pipeline);
        assert_eq!(Mode::of(true, true, true, false), Mode::Wait);
        assert_eq!(Mode::of(false, true, false, true), Mode::Wait);
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
