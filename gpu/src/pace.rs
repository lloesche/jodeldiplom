//! Sharing the GPU with the window.
//!
//! A network run as one submission keeps the GPU to itself until it is
//! done: a 6 s chunk of stems, a 30 s piece of speech, a tenth of a second
//! or more on a laptop's GPU, and every frame the window wants meanwhile
//! waits. So every run here goes in short submissions ([`dispatch`]), a
//! few milliseconds of GPU time apiece, and the window's frames go in
//! between. The next submission is made while the last one runs, so the
//! GPU does not wait for the processor between them.
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
//! [`background`] marks a thread whose work is background work, for the
//! caller to pick the background separator there. Each network keeps a
//! [`Tally`] of what its GPU work cost so far: its submissions, the times
//! it waited for the GPU, and how long it gave way.

use std::cell::Cell;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

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
/// How long one submission should take on the GPU.
const PIECE: Duration = Duration::from_millis(5);
/// The longest rest after one.
const LONGEST_REST: Duration = Duration::from_millis(60);
/// The longest background work waits for other work at a time.
pub const LONGEST_WAIT: Duration = Duration::from_secs(1);

/// Runs of work that does not rest under way now.
static LIVE: AtomicU32 = AtomicU32::new(0);

/// What a network's GPU work cost so far, for traces: the difference of
/// two ([`Tally::since`]) is what was done between them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    /// Submissions made ([`dispatch`]).
    pub submissions: u64,
    /// Times it waited for the GPU to finish work: its round trips.
    pub waits: u64,
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
            "{} submissions, {} waits for the GPU, {} rests ({:.1} s), {:.1} s behind the track playing",
            self.submissions,
            self.waits,
            self.rests,
            self.rested.as_secs_f64(),
            self.held.as_secs_f64()
        )
    }
}

/// A network's [`Tally`] as it grows: shared with whoever asks for it
/// ([`Counts::tally`] never waits for the network's work), its times in µs.
#[derive(Clone, Debug, Default)]
pub(crate) struct Counts(Arc<[AtomicU64; 5]>);

impl Counts {
    const SUBMISSIONS: usize = 0;
    const WAITS: usize = 1;
    const RESTS: usize = 2;
    const RESTED: usize = 3;
    const HELD: usize = 4;

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
            rests: n(Counts::RESTS),
            rested: Duration::from_micros(n(Counts::RESTED)),
            held: Duration::from_micros(n(Counts::HELD)),
        }
    }
}

/// Counts a run of work that does not rest while it lives.
struct Live;

impl Live {
    fn start() -> Live {
        LIVE.fetch_add(1, Ordering::AcqRel);
        Live
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        LIVE.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Background work, before a submission: the rest it owes for the one
/// before (while the window still draws at the refresh), then waits while
/// other work is on the GPU (at most [`LONGEST_WAIT`]); both counted.
fn give_way(pieces: &mut Pieces) {
    let owed = std::mem::take(&mut pieces.owed);
    if !owed.is_zero() && duty() < 1.0 {
        let p = pace();
        p.rests.fetch_add(1, Ordering::Relaxed);
        p.rested
            .fetch_add(owed.as_micros() as u64, Ordering::Relaxed);
        pieces.counts.add(Counts::RESTS, 1);
        pieces.counts.add_time(Counts::RESTED, owed);
        std::thread::sleep(owed);
    }
    let started = Instant::now();
    while LIVE.load(Ordering::Acquire) > 0 && started.elapsed() < LONGEST_WAIT {
        std::thread::sleep(Duration::from_millis(1));
    }
    let waited = started.elapsed();
    if waited > Duration::from_micros(500) {
        pace()
            .waited
            .fetch_add(waited.as_micros() as u64, Ordering::Relaxed);
        pieces.counts.add_time(Counts::HELD, waited);
    }
}

struct Pace {
    epoch: Instant,
    /// The window's last frame at the refresh, in µs from `epoch`, plus one
    /// (0: none yet).
    frame_at: AtomicU64,
    /// Background work's last submission, the same way.
    worked_at: AtomicU64,
    /// When the share last climbed, the same way.
    climbed_at: AtomicU64,
    /// The share, as `f32` bits.
    duty: AtomicU32,
    /// Late frames while background work ran, and rests taken, and their
    /// total in µs (for the trace).
    late: AtomicU64,
    rests: AtomicU64,
    rested: AtomicU64,
    /// Time background work waited for other work, in µs.
    waited: AtomicU64,
}

fn pace() -> &'static Pace {
    static PACE: OnceLock<Pace> = OnceLock::new();
    PACE.get_or_init(|| Pace {
        epoch: Instant::now(),
        frame_at: AtomicU64::new(0),
        worked_at: AtomicU64::new(0),
        climbed_at: AtomicU64::new(0),
        duty: AtomicU32::new(START.to_bits()),
        late: AtomicU64::new(0),
        rests: AtomicU64::new(0),
        rested: AtomicU64::new(0),
        waited: AtomicU64::new(0),
    })
}

impl Pace {
    fn now(&self) -> u64 {
        self.epoch.elapsed().as_micros() as u64 + 1
    }

    fn since(&self, at: &AtomicU64) -> Option<Duration> {
        match at.load(Ordering::Relaxed) {
            0 => None,
            t => Some(Duration::from_micros(self.now().saturating_sub(t))),
        }
    }

    fn duty(&self) -> f32 {
        f32::from_bits(self.duty.load(Ordering::Relaxed))
    }

    fn set_duty(&self, d: f32) {
        self.duty
            .store(d.clamp(LEAST, MOST).to_bits(), Ordering::Relaxed);
    }
}

/// The window drew a frame at the display's refresh, something on it
/// moving smoothly (calm frames say nothing); `late`: its last frame was
/// not through the GPU yet (it missed its turn, the GPU busy with other
/// work). A late frame while background work runs cuts that work's share;
/// a second of frames on time gives some back.
pub fn frame(late: bool) {
    let p = pace();
    let now = p.now();
    p.frame_at.store(now, Ordering::Relaxed);
    let working = p
        .since(&p.worked_at)
        .is_some_and(|d| d < Duration::from_millis(250));
    if !working {
        return;
    }
    if late {
        p.late.fetch_add(1, Ordering::Relaxed);
        p.set_duty(p.duty() * 0.6);
        p.climbed_at.store(now, Ordering::Relaxed);
    } else if p
        .since(&p.climbed_at)
        .is_none_or(|d| d >= Duration::from_secs(1))
    {
        p.set_duty(p.duty() + 0.05);
        p.climbed_at.store(now, Ordering::Relaxed);
    }
}

/// The share of the time background work keeps to now: [`MOST`] down to
/// [`LEAST`] while the window draws at the display's refresh, 1 while it
/// does not.
pub fn duty() -> f32 {
    let p = pace();
    if p.since(&p.frame_at).is_some_and(|d| d < MOVING) {
        p.duty()
    } else {
        1.0
    }
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
    let p = pace();
    Report {
        late: p.late.swap(0, Ordering::Relaxed),
        rests: p.rests.swap(0, Ordering::Relaxed),
        rested: Duration::from_micros(p.rested.swap(0, Ordering::Relaxed)),
        waited: Duration::from_micros(p.waited.swap(0, Ordering::Relaxed)),
        duty: duty(),
    }
}

/// Background work just ran on the GPU (late frames now cut its share:
/// [`frame`]).
fn worked() {
    let p = pace();
    p.worked_at.store(p.now(), Ordering::Relaxed);
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

/// Waits until the GPU has done the submission `index`, counted in
/// `counts`.
fn wait(
    counts: &Counts,
    device: &wgpu::Device,
    index: wgpu::SubmissionIndex,
) -> Result<(), String> {
    counts.add(Counts::WAITS, 1);
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        })
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// How many dispatches a submission takes, learned per run: about
/// [`PIECE`] of GPU time; whether the work is background work, which
/// rests, and the rest it owes; and what its runs cost so far.
#[derive(Clone, Debug, Default)]
pub(crate) struct Pieces {
    /// Unknown yet (0): an eighth of a run's dispatches at first.
    per: usize,
    pub(crate) rests: bool,
    owed: Duration,
    counts: Counts,
}

impl Pieces {
    /// For background work.
    pub(crate) fn resting() -> Pieces {
        Pieces {
            rests: true,
            ..Pieces::default()
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

/// One recorded dispatch: the kernel, the bind group, its parameters'
/// offset, the workgroups.
pub(crate) type Op<K> = (K, usize, u64, [u32; 3]);

/// Runs `ops` in submissions of about [`PIECE`] each, `prelude` before the
/// first and `finish` after the last, and waits until they are done (the
/// caller then reads their result: [`crate::device::read_back`]). Each
/// submission is made while the one before runs, which is waited for then;
/// but while background work (`pieces.rests`) gives way to the window
/// ([`duty`] below 1), each is waited for at once, and the rest it owes is
/// taken before the next ([`give_way`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch<K: Copy + Eq + Hash>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    pieces: &mut Pieces,
    pipelines: &HashMap<K, wgpu::ComputePipeline>,
    groups: &[wgpu::BindGroup],
    ops: &[Op<K>],
    prelude: impl FnOnce(&mut wgpu::CommandEncoder),
    finish: impl FnOnce(&mut wgpu::CommandEncoder),
) -> Result<(), String> {
    if pieces.per == 0 {
        pieces.per = ops.len().div_ceil(8).max(1);
    }
    let _live = (!pieces.rests).then(Live::start);
    let (mut prelude, mut finish) = (Some(prelude), Some(finish));
    let mut start = 0;
    // The submission still running (its index, its dispatches and when it
    // was made), and when the one before it was done.
    let mut running: Option<(wgpu::SubmissionIndex, usize, Instant)> = None;
    let mut done_at: Option<Instant> = None;
    loop {
        let end = (start + pieces.per).min(ops.len());
        let last = end == ops.len();
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        if let Some(prelude) = prelude.take() {
            prelude(&mut encoder);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            for &(kernel, group, offset, [x, y, z]) in &ops[start..end] {
                pass.set_pipeline(&pipelines[&kernel]);
                pass.set_bind_group(0, &groups[group], &[offset as u32]);
                pass.dispatch_workgroups(x, y, z);
            }
        }
        if last {
            if let Some(finish) = finish.take() {
                finish(&mut encoder);
            }
        }
        if pieces.rests {
            give_way(pieces);
        }
        pieces.counts.add(Counts::SUBMISSIONS, 1);
        let made = Instant::now();
        let index = queue.submit([encoder.finish()]);
        let n = end - start;
        let duty = if pieces.rests { duty() } else { 1.0 };
        // The last, or giving way: this one waited for now (the one before
        // with it); else the one before, this one running meanwhile.
        let waited = if last || duty < 1.0 {
            Some(match running.take() {
                Some((_, before, at)) => (index, before + n, at),
                None => (index, n, made),
            })
        } else {
            running.replace((index, n, made))
        };
        if let Some((index, n, at)) = waited {
            wait(&pieces.counts, device, index)?;
            let now = Instant::now();
            // Their GPU time: from when they were made, or when the one
            // before them was done, if later.
            let took = now - done_at.map_or(at, |d| d.max(at));
            done_at = Some(now);
            // Towards PIECE a submission, smoothly.
            let ideal = n as f64 * PIECE.as_secs_f64() / took.as_secs_f64().max(1e-5);
            let next = (pieces.per as f64 * 0.5 + ideal * 0.5).round() as usize;
            pieces.per = next.clamp(1, ops.len());
            if pieces.rests {
                worked();
                owe(pieces, took, duty);
            }
        }
        if last {
            return Ok(());
        }
        start = end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_share_falls_on_late_frames_and_climbs_back() {
        let p = pace();
        p.set_duty(0.5);
        // No background work: late frames change nothing.
        p.worked_at.store(0, Ordering::Relaxed);
        frame(true);
        assert_eq!(p.duty(), 0.5);
        // With it: a late frame cuts the share, down to the least.
        p.worked_at.store(p.now(), Ordering::Relaxed);
        frame(true);
        assert!((p.duty() - 0.3).abs() < 1e-6, "{}", p.duty());
        for _ in 0..20 {
            frame(true);
        }
        assert_eq!(p.duty(), LEAST);
        // The window draws at the refresh: background work keeps to the
        // share.
        assert_eq!(duty(), LEAST);
        // A frame on time, with no change for a second, gives some back.
        p.climbed_at.store(0, Ordering::Relaxed);
        frame(false);
        assert!(p.duty() > LEAST);
        // The rest owed keeps to the share: about 0.85 / 0.15 of the work,
        // taken before the next submission.
        let mut pieces = Pieces::resting();
        owe(&mut pieces, Duration::from_millis(4), duty());
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
        owe(&mut pieces, Duration::from_millis(20), duty());
        p.frame_at.store(0, Ordering::Relaxed);
        assert_eq!(duty(), 1.0);
        let started = Instant::now();
        give_way(&mut pieces);
        assert!(started.elapsed() < Duration::from_millis(10));
        assert_eq!(pieces.counts().tally().rests, 1);
    }

    /// Background work waits while work that does not rest is under way,
    /// and goes on once it ends.
    #[test]
    fn background_work_waits_for_the_work_that_does_not_rest() {
        let mut pieces = Pieces::resting();
        let started = Instant::now();
        give_way(&mut pieces);
        assert!(
            started.elapsed() < Duration::from_millis(50),
            "nothing to wait for"
        );
        let live = Live::start();
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
    }
}
