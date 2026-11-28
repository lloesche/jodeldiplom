//! The submissions of the pace's work: a run's dispatches in pieces of
//! about [`PIECE`] ([`dispatch`], [`dispatch_left`] and [`finish`], timed
//! on each device's [`Track`]), and a model's weights going up in batches
//! of as long ([`Uploading`]), each giving way as [`super`] says.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Receiver;
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
        pieces.learn(left.n, left.all, now - from);
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
/// there are none), at most twice the more of `per` and `n` (a submission
/// of whole steps, or one taken with the one before, takes more than
/// `per`; a run's short last one fewer): dispatches cost unlike amounts (a
/// norm some microseconds, a multiply's slice a piece), so a submission of
/// cheap ones says little of the next, and one twice as long as the last
/// is the most a wrong guess costs.
pub(super) fn learned(per: usize, n: usize, all: usize, took: Duration) -> usize {
    let next = toward(per as f64, in_a_piece(n as f64, took)).round() as usize;
    next.min(2 * per.max(n)).clamp(1, all.max(1))
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
/// the fewest and the most (a batch is staged whole until it is sent, so a
/// larger one holds more staging at once).
const UPLOAD_FIRST: u64 = 32 << 20;
const UPLOAD_LEAST: u64 = 4 << 20;
const UPLOAD_MOST: u64 = 256 << 20;

/// A model's weights going up, paced as other work is: written in batches
/// of about a piece ([`PIECE`]) of the GPU's time copying them, their size
/// learned as they go ([`UPLOAD_FIRST`] to start, [`UPLOAD_LEAST`] to
/// [`UPLOAD_MOST`]), each sent and waited for. A background thread's give
/// way before each as its runs' submissions do, and count as its work
/// running, so a window starved while they go up is told
/// ([`super::tick`]). The writes wait for their batch in one staging
/// buffer, kept from batch to batch ([`Staging`]).
pub(crate) struct Uploading {
    pieces: Pieces,
    batch: u64,
    /// The fewest and the most bytes a batch takes.
    bounds: (u64, u64),
    /// The bytes the batch holds so far, written or zeroed: what its time
    /// is learned from.
    batched: u64,
    /// Where the batch's next write goes in the staging: the batch's
    /// writes lie in it one after another.
    cursor: u64,
    /// The bytes still to be written, as the upload was told (whole
    /// words): the staging is no larger than they need.
    left: u64,
    staging: Option<Staging>,
    /// The batch's writes, copied from the staging when it is sent.
    copies: Vec<StagedWrite>,
}

/// One write of a batch: its bytes' place in the staging (`from`), its
/// buffer (`to`), where in that (`at`), and how many (`len`).
struct StagedWrite {
    from: u64,
    to: wgpu::Buffer,
    at: u64,
    len: u64,
}

/// Where a batch's writes wait to be copied to their buffers
/// ([`Uploading`]): a buffer the processor writes, the batch's writes one
/// after another in it, as large as a batch or what is left to write,
/// whichever is less (a power of two: made again only where the batches
/// outgrow it), kept from batch to batch of an upload. `wgpu` zeroes a buffer mapped as it is made on the processor
/// (besides the system's zeroing of its fresh pages), so it is made no
/// larger than it must be. `wgpu`'s own writes stage each in a new buffer:
/// fresh memory for every batch, some 13 GB of it for the story's models.
/// Unmapped while a batch is copied from it, and mapped again as that
/// batch is waited for. It goes with its upload: kept by its program, it
/// would hold up to [`UPLOAD_MOST`] for the program's life that no
/// footprint counts (the beat network's 99 MB took 386).
struct Staging {
    buffer: wgpu::Buffer,
    /// What tells that it is mapped again, where it is being mapped (it is
    /// mapped otherwise).
    mapping: Option<Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

impl Staging {
    /// One of `size` bytes on `program`'s device, mapped.
    fn new<K>(program: &Program<K>, size: u64) -> Staging {
        Staging {
            buffer: program.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jodel-gpu upload"),
                size,
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            }),
            mapping: None,
        }
    }

    /// The buffer, mapped: where it was being mapped, once it is (as the
    /// batch copied from it was waited for, mostly; else waited for now).
    fn ready<K: Kernels>(&mut self, program: &Program<K>) -> Result<&wgpu::Buffer, String> {
        if let Some(rx) = self.mapping.take() {
            let mapped = match rx.try_recv() {
                Ok(mapped) => mapped,
                Err(_) => {
                    program.wait()?;
                    rx.recv().map_err(|e| e.to_string())?
                }
            };
            mapped.map_err(|e| e.to_string())?;
        }
        Ok(&self.buffer)
    }
}

impl Uploading {
    /// For the calling thread's work, `bytes` to be written in all.
    pub(crate) fn new(bytes: u64) -> Uploading {
        Uploading {
            pieces: Pieces::for_this_thread(),
            batch: UPLOAD_FIRST,
            bounds: (UPLOAD_LEAST, UPLOAD_MOST),
            batched: 0,
            cursor: 0,
            left: bytes,
            staging: None,
            copies: Vec::new(),
        }
    }

    /// For checks: batches of `least` bytes to start, learned up to `most`;
    /// `bytes` to be written.
    #[cfg(test)]
    pub(crate) fn within(least: u64, most: u64, bytes: u64) -> Uploading {
        Uploading {
            batch: least,
            bounds: (least, most),
            ..Uploading::new(bytes)
        }
    }

    /// For checks: batches of `bytes` from the next one on, always.
    #[cfg(test)]
    pub(crate) fn batches_of(&mut self, bytes: u64) {
        (self.batch, self.bounds) = (bytes, (bytes, bytes));
    }

    /// For checks: the bytes of the staging now (none before a write).
    #[cfg(test)]
    pub(crate) fn staging_bytes(&self) -> u64 {
        self.staging.as_ref().map_or(0, |s| s.buffer.size())
    }

    /// The bytes the batch has room for still, whole words (one at least).
    pub(crate) fn room(&self) -> u64 {
        (self.batch.saturating_sub(self.cursor) & !3).max(4)
    }

    /// `data` (whole words, no more than [`Uploading::room`]) to go to
    /// `buffer` from its byte `at` with the batch: staged now, the batch
    /// sent where it is full.
    pub(crate) fn write<K: Kernels>(
        &mut self,
        program: &Program<K>,
        buffer: &wgpu::Buffer,
        at: u64,
        data: &[u8],
    ) -> Result<(), String> {
        let len = data.len() as u64;
        assert!(len.is_multiple_of(4), "a write of {len} bytes");
        // A batch's first write makes the staging as large as the batch or
        // what is left, whichever is less (this write at least), where the
        // one there is smaller.
        if self.cursor == 0 {
            let size = self.batch.min(self.left).max(len).next_power_of_two();
            if self.staging.as_ref().is_none_or(|s| s.buffer.size() < size) {
                self.staging = Some(Staging::new(program, size));
            }
        }
        let staging = (self.staging.as_mut()).expect("made at the batch's first write");
        let from = self.cursor;
        let staged = staging.ready(program)?;
        // The staging holds the bytes written and no more: past it, the
        // upload was told fewer than it writes.
        assert!(
            from + len <= staged.size(),
            "an upload told fewer bytes than it writes"
        );
        // A mapping starts on 8 bytes: from the one at or below the write.
        let start = from & !7;
        staged
            .get_mapped_range_mut(start..from + len)
            .map_err(|e| e.to_string())?
            .slice((from - start) as usize..)
            .copy_from_slice(data);
        self.copies.push(StagedWrite {
            from,
            to: buffer.clone(),
            at,
            len,
        });
        self.cursor = from + len;
        self.batched += len;
        self.left = self.left.saturating_sub(len);
        // Full: no room for another word.
        if self.cursor + 4 > self.batch {
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
            self.send(program, Some((buffer, at, n)))?;
            at += n;
        }
        Ok(())
    }

    /// The batch's writes copied from the staging (and `clear`'s bytes of
    /// its buffer zeroed after them, counted in the batch: from where, how
    /// many) sent and waited for, after giving way, the staging mapped again
    /// meanwhile; the next batch's size learned from how long it took.
    fn send<K: Kernels>(
        &mut self,
        program: &Program<K>,
        clear: Option<(&wgpu::Buffer, u64, u64)>,
    ) -> Result<(), String> {
        let mut encoder = program
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let copied = !self.copies.is_empty();
        if copied {
            let staging = self
                .staging
                .as_ref()
                .expect("made at the batch's first write");
            staging.buffer.unmap();
            for c in self.copies.drain(..) {
                encoder.copy_buffer_to_buffer(&staging.buffer, c.from, &c.to, c.at, c.len);
            }
        }
        if let Some((buffer, at, n)) = clear {
            encoder.clear_buffer(buffer, at, Some(n));
            self.batched += n;
        }
        self.pieces.before();
        let made = Instant::now();
        program.queue.submit([encoder.finish()]);
        if copied {
            // Mapped once the GPU has copied from it: the wait below covers
            // it.
            let staging = self
                .staging
                .as_mut()
                .expect("made at the batch's first write");
            let (tx, rx) = std::sync::mpsc::channel();
            staging
                .buffer
                .map_async(wgpu::MapMode::Write, .., move |r| {
                    let _ = tx.send(r);
                });
            staging.mapping = Some(rx);
        }
        program.wait()?;
        let took = made.elapsed();
        self.batch = next_batch(self.batch, self.batched, took, self.bounds);
        self.batched = 0;
        self.cursor = 0;
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
        let end = pieces.end(start, whole, ops.len());
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
        pieces.learn(n, all, took);
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
