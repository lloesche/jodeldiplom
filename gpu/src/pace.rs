//! Sharing the GPU with the window.
//!
//! A network run as one submission keeps the GPU to itself until it is
//! done: a 6 s chunk of stems, a 30 s piece of speech, a tenth of a second
//! or more on a laptop's GPU, and every frame the window wants meanwhile
//! waits. So every run here goes in short submissions ([`dispatch`]), each
//! waited for, a few milliseconds of GPU time apiece, and the window's
//! frames go in between.
//!
//! Background work also rests between its submissions while the window is
//! moving ([`frame`] says so, every frame): the beat network, the speech
//! model, and [`crate::GpuSeparator::background`] (the analyses' stems,
//! the lyrics' voice). It keeps to a share of the time ([`duty`]) that
//! falls when the window's frames come late and climbs back while they are
//! on time; while nothing moves on screen it runs flat out. The stems of
//! the track playing ([`crate::GpuSeparator::shared`]), which have to keep
//! up with it, never rest, and background work does not run beside them:
//! while they have work on the GPU, it waits (at most [`LONGEST_WAIT`] at a
//! time), so the GPU has one network on it at once, as when they shared a
//! separator.
//!
//! [`background`] marks a thread whose work is background work, for the
//! caller to pick the background separator there.

use std::cell::Cell;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::OnceLock;
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

/// The share of the time background work keeps to while the window moves,
/// at most.
pub const MOST: f32 = 0.75;
/// And at least.
pub const LEAST: f32 = 0.1;
/// Where it starts.
const START: f32 = 0.5;
/// The window counts as moving this long after its last frame.
const MOVING: Duration = Duration::from_millis(500);
/// How long one submission should take on the GPU.
const PIECE: Duration = Duration::from_millis(5);
/// The longest rest after one.
const LONGEST_REST: Duration = Duration::from_millis(60);
/// The longest background work waits for other work at a time.
pub const LONGEST_WAIT: Duration = Duration::from_secs(1);

/// Runs of work that does not rest under way now.
static LIVE: AtomicU32 = AtomicU32::new(0);
thread_local! {
    /// Submissions [`dispatch`] has made on this thread, for traces.
    static SUBMITTED: Cell<u64> = const { Cell::new(0) };
}

/// The submissions of work the calling thread has made through this
/// module so far, for traces.
pub(crate) fn submissions() -> u64 {
    SUBMITTED.with(Cell::get)
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

/// Background work, before a submission: waits while other work is on the
/// GPU (at most [`LONGEST_WAIT`]).
fn give_way() {
    let started = Instant::now();
    while LIVE.load(Ordering::Acquire) > 0 && started.elapsed() < LONGEST_WAIT {
        std::thread::sleep(Duration::from_millis(1));
    }
    let waited = started.elapsed();
    if waited > Duration::from_micros(500) {
        pace()
            .waited
            .fetch_add(waited.as_micros() as u64, Ordering::Relaxed);
    }
}

struct Pace {
    epoch: Instant,
    /// The window's last frame, in µs from `epoch`, plus one (0: none yet).
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

/// The window drew a frame while something on it moves; `late`: its last
/// frame was not through the GPU yet (it missed its turn, the GPU busy
/// with other work). A late frame while background work
/// runs cuts that work's share; a second of frames on time gives some back.
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
/// [`LEAST`] while the window moves, 1 while it does not.
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

/// After a submission of background work that took `took`: rests so the
/// work keeps to its share.
fn rest(took: Duration) {
    let p = pace();
    p.worked_at.store(p.now(), Ordering::Relaxed);
    let duty = duty();
    if duty >= 1.0 {
        return;
    }
    let rest = took.mul_f32((1.0 - duty) / duty).min(LONGEST_REST);
    if rest > Duration::from_micros(200) {
        p.rests.fetch_add(1, Ordering::Relaxed);
        p.rested
            .fetch_add(rest.as_micros() as u64, Ordering::Relaxed);
        std::thread::sleep(rest);
    }
}

/// How many dispatches a submission takes, learned per run: about
/// [`PIECE`] of GPU time.
/// And whether the work is background work, which rests.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Pieces {
    /// Unknown yet (0): an eighth of a run's dispatches at first.
    per: usize,
    pub(crate) rests: bool,
}

impl Pieces {
    /// For background work.
    pub(crate) fn resting() -> Pieces {
        Pieces {
            per: 0,
            rests: true,
        }
    }
}

/// One recorded dispatch: the kernel, the bind group, its parameters'
/// offset, the workgroups.
pub(crate) type Op<K> = (K, usize, u64, [u32; 3]);

/// Runs `ops` in submissions of about [`PIECE`] each, each waited for (and
/// rested after, for background work: `pieces.rests`): `prelude` goes
/// before the first,
/// `finish` after the last (that one is submitted, not waited for: the
/// caller maps its result and waits).
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
    let mut prelude = Some(prelude);
    let mut start = 0;
    loop {
        if pieces.rests {
            give_way();
        }
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
        SUBMITTED.with(|s| s.set(s.get() + 1));
        if last {
            finish(&mut encoder);
            queue.submit([encoder.finish()]);
            return Ok(());
        }
        let started = Instant::now();
        queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| e.to_string())?;
        let took = started.elapsed();
        // Towards PIECE a submission, smoothly.
        let n = end - start;
        let ideal = n as f64 * PIECE.as_secs_f64() / took.as_secs_f64().max(1e-5);
        let next = (n as f64 * 0.5 + ideal * 0.5).round() as usize;
        pieces.per = next.clamp(1, ops.len());
        if pieces.rests {
            rest(took);
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
        // The window moves: background work keeps to the share.
        assert_eq!(duty(), LEAST);
        // A frame on time, with no change for a second, gives some back.
        p.climbed_at.store(0, Ordering::Relaxed);
        frame(false);
        assert!(p.duty() > LEAST);
        // A rest keeps to the share: about 0.85 / 0.15 of the work.
        let started = Instant::now();
        rest(Duration::from_millis(4));
        let rested = started.elapsed();
        assert!(rested >= Duration::from_millis(12), "{rested:?}");
        // While the window is still, none.
        p.frame_at.store(0, Ordering::Relaxed);
        let started = Instant::now();
        rest(Duration::from_millis(20));
        assert!(started.elapsed() < Duration::from_millis(10));
    }

    /// Background work waits while work that does not rest is under way,
    /// and goes on once it ends.
    #[test]
    fn background_work_waits_for_the_work_that_does_not_rest() {
        let started = Instant::now();
        give_way();
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
        give_way();
        let waited = started.elapsed();
        ended.join().unwrap();
        assert!(waited >= Duration::from_millis(60), "{waited:?}");
        assert!(waited < LONGEST_WAIT, "{waited:?}");
    }
}
