//! The state an output handle shares with its device thread.
//!
//! The handle packs samples into the [`Ring`]; the backend's device thread
//! (or IOProc) calls [`Shared::render`] for every device buffer, which
//! copies queued frames out and pads with silence when the ring runs dry.
//! Position, statistics and failures are published here for any thread to
//! read. A [`Tap`] set on the output sees every buffer here too, and
//! cannot change it.
//!
//! On the device thread's way nothing locks, allocates or waits, but for
//! two things: a failure (kept behind a lock, its message made as the
//! device fails) and waking a writer that waits for room (a condition
//! variable's notify, a system call with no lock). The timeline has one
//! writer at a time: the consumer, inside its render ([`Shared::consume`]),
//! or the handle, which keeps the consumer out meanwhile and takes a lock
//! of its own against other handle-side writers ([`Shared::keep_out`]).
//! Readers take what it publishes ([`Published`]).

use crate::error::Error;
use crate::format::StreamFormat;
use crate::output::Stats;
use crate::ring::Ring;
use crate::tap::{Tap, TapSlot};
use crate::timeline::{Published, Timeline};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// The longest [`Shared::set_tap`] waits for the consumer to take a tap
/// while it renders, before putting it in place itself.
const TAP_WAIT: Duration = Duration::from_secs(1);

/// The timeline: reached only through [`Shared::consume`] and
/// [`Shared::keep_out`], one writer at a time.
struct Owned(UnsafeCell<Timeline>);

// SAFETY: one side at a time reaches the timeline, as above; readers go
// through what it publishes.
unsafe impl Sync for Owned {}

pub(crate) struct Shared {
    pub(crate) format: StreamFormat,
    pub(crate) ring: Ring,
    timeline: Owned,
    /// What the timeline publishes: the position, for any thread.
    clock: Published,
    /// Held by a handle-side writer of the timeline ([`Self::keep_out`]);
    /// the consumer never takes it.
    handle: Mutex<()>,
    /// An empty ring is expected (after a drain, before the next write), so
    /// padding is not an underrun.
    expect_empty: AtomicBool,
    /// The consumer padded its last buffer: an underrun is counted once per
    /// run of padded buffers with no stream frames between them.
    in_underrun: AtomicBool,
    /// Above 0 while the handle needs the consumer to keep out of the ring
    /// (suspensions may nest).
    suspend: AtomicU32,
    /// Set while the consumer is inside the ring.
    in_render: AtomicBool,
    /// Device frames the consumer filled with silence while kept out: the
    /// timeline is told of them at the next render that goes through.
    skipped: AtomicU64,
    /// Buffers the consumer was asked for.
    renders: AtomicU64,
    /// The device pulls buffers on its own now (it runs, and is not one
    /// that plays only when told), and its period in frames.
    pulling: AtomicBool,
    period: AtomicU64,
    underruns: AtomicU64,
    underrun_frames: AtomicU64,
    device_underruns: AtomicU64,
    clipped: AtomicU64,
    truncated: AtomicU64,
    /// Late frames a mixer's input dropped to keep up with the one it
    /// follows ([`Self::skip_input`]).
    dropped: AtomicU64,
    written: AtomicU64,
    /// The handle closed the output.
    closed: AtomicBool,
    failed: AtomicBool,
    failure: Mutex<Option<Error>>,
    writer_waiting: AtomicBool,
    space_lock: Mutex<()>,
    space: Condvar,
    tap: TapSlot,
    /// Stream frames copied out of the ring so far (as the position counts
    /// them: frames discarded count as copied). Only the consumer moves
    /// it, but for [`Self::discard_all`].
    rendered: AtomicU64,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Shared {
    pub(crate) fn new(format: StreamFormat, ring_frames: usize) -> Shared {
        Shared {
            format,
            ring: Ring::new(ring_frames, format.frame_bytes()),
            timeline: Owned(UnsafeCell::new(Timeline::new())),
            clock: Published::new(format.rate),
            handle: Mutex::new(()),
            expect_empty: AtomicBool::new(true),
            in_underrun: AtomicBool::new(false),
            suspend: AtomicU32::new(0),
            in_render: AtomicBool::new(false),
            skipped: AtomicU64::new(0),
            renders: AtomicU64::new(0),
            pulling: AtomicBool::new(false),
            period: AtomicU64::new(0),
            underruns: AtomicU64::new(0),
            underrun_frames: AtomicU64::new(0),
            device_underruns: AtomicU64::new(0),
            clipped: AtomicU64::new(0),
            truncated: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            written: AtomicU64::new(0),
            closed: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            failure: Mutex::new(None),
            writer_waiting: AtomicBool::new(false),
            space_lock: Mutex::new(()),
            space: Condvar::new(),
            tap: TapSlot::new(),
            rendered: AtomicU64::new(0),
        }
    }

    // ---- consumer side (device thread) ----

    /// Fills `out` (whole frames) from the ring and pads the rest with
    /// silence. Returns the stream frames copied, or `None` when the handle
    /// has suspended the consumer: then `out` is all silence, which the
    /// timeline counts at the next render that goes through.
    pub(crate) fn render_or_skip(&self, out: &mut [u8]) -> Option<u64> {
        self.render_with(out, |t, got, missing| t.submit(got, missing))
            .map(|(_, got, ())| got)
    }

    /// [`Self::render_or_skip`] for a mixer's input ([`crate::mixer`]): the
    /// frames count as played at once and as gone downstream, until the
    /// mixer says the device has played them ([`Self::set_downstream`]).
    /// Returns the first stream frame copied, how many, and the timeline's
    /// epoch they belong to.
    pub(crate) fn render_input(&self, out: &mut [u8]) -> Option<(u64, u64, u64)> {
        self.render_with(out, |t, got, missing| {
            t.submit(got, missing);
            t.report_delay(0, Instant::now(), false);
            t.add_downstream(got);
            t.epoch()
        })
    }

    /// For a mixer's input that follows another: drops up to `frames` of
    /// the oldest frames queued (late ones), counted as played at once in
    /// no device frames, and as gone downstream (as the frames of
    /// [`Self::render_input`]). Returns how many, and the timeline's epoch
    /// they belong to; `None` while the handle keeps the consumer out.
    pub(crate) fn skip_input(&self, frames: u64) -> Option<(u64, u64)> {
        let (dropped, epoch) = self.consume(|t| {
            let n = self.ring.skip(frames);
            self.rendered.fetch_add(n, Ordering::Relaxed);
            self.dropped.fetch_add(n, Ordering::Relaxed);
            t.skip(n);
            t.add_downstream(n);
            (n, t.epoch())
        })?;
        if dropped > 0 && self.writer_waiting.load(Ordering::SeqCst) {
            self.space.notify_all();
        }
        Some((dropped, epoch))
    }

    /// The consumer: the stream frame it takes next (frames discarded and
    /// dropped count as taken).
    pub(crate) fn head(&self) -> u64 {
        self.rendered.load(Ordering::Relaxed)
    }

    /// Of what went downstream since the timeline's restart `epoch`, only
    /// `frames` are not played yet (nothing, while the handle keeps the
    /// consumer out: it is restarting the timeline).
    pub(crate) fn set_downstream(&self, epoch: u64, frames: u64) {
        self.consume(|t| t.set_downstream(epoch, frames));
    }

    /// The consumer changes the timeline with `f`, inside the ring, and
    /// publishes it (`None`, without running `f`, while the handle keeps
    /// the consumer out). With [`Self::keep_out`], the timeline's only
    /// writer.
    fn consume<R>(&self, f: impl FnOnce(&mut Timeline) -> R) -> Option<R> {
        if self.suspend.load(Ordering::SeqCst) != 0 {
            return None;
        }
        self.in_render.store(true, Ordering::SeqCst);
        let r = (self.suspend.load(Ordering::SeqCst) == 0).then(|| {
            // SAFETY: the consumer is inside the ring and the handle keeps
            // out of the timeline until it has left (`suspend`).
            let t = unsafe { &mut *self.timeline.0.get() };
            let r = f(t);
            t.publish(&self.clock);
            r
        });
        self.in_render.store(false, Ordering::SeqCst);
        r
    }

    /// The handle changes the timeline with `f` and publishes it, the
    /// consumer kept out and other handle-side writers waiting.
    fn keep_out<R>(&self, f: impl FnOnce(&mut Timeline) -> R) -> R {
        let _writer = lock(&self.handle);
        self.suspend();
        // SAFETY: the consumer is out of the ring until `unsuspend`, and
        // the lock keeps every other handle-side writer out.
        let t = unsafe { &mut *self.timeline.0.get() };
        let r = f(t);
        t.publish(&self.clock);
        self.unsuspend();
        r
    }

    /// The render: `note` records it in the timeline (after the silence
    /// of the buffers skipped since the last one). Returns the first
    /// stream frame copied, how many, and what `note` gave.
    fn render_with<R>(
        &self,
        out: &mut [u8],
        note: impl FnOnce(&mut Timeline, u64, u64) -> R,
    ) -> Option<(u64, u64, R)> {
        let frames = (out.len() / self.format.frame_bytes()) as u64;
        self.renders.fetch_add(1, Ordering::Release);
        let Some(rendered) = self.consume(|t| self.copy_out(t, out, note)) else {
            out.fill(0);
            self.skipped.fetch_add(frames, Ordering::Relaxed);
            return None;
        };
        if rendered.1 > 0 && self.writer_waiting.load(Ordering::SeqCst) {
            self.space.notify_all();
        }
        Some(rendered)
    }

    /// [`Self::render_with`] inside the ring.
    fn copy_out<R>(
        &self,
        t: &mut Timeline,
        out: &mut [u8],
        note: impl FnOnce(&mut Timeline, u64, u64) -> R,
    ) -> (u64, u64, R) {
        let fb = self.format.frame_bytes();
        let want = (out.len() / fb) as u64;
        let got = self.ring.pop_into(out);
        out[got as usize * fb..].fill(0);
        let first = self.rendered.fetch_add(got, Ordering::Relaxed);
        // SAFETY: inside the render, as the slot requires. The tap only
        // reads `out`.
        unsafe { self.tap.run(out, first, got as usize, self.format) };
        let missing = want - got;
        if missing > 0 && !self.expect_empty.load(Ordering::Acquire) {
            // A buffer with no stream frames after a padded one continues
            // the same underrun; any stream frame ends it.
            let continuing = got == 0 && self.in_underrun.load(Ordering::Relaxed);
            if !continuing {
                self.underruns.fetch_add(1, Ordering::Relaxed);
            }
            self.in_underrun.store(true, Ordering::Relaxed);
            self.underrun_frames.fetch_add(missing, Ordering::Relaxed);
        } else {
            self.in_underrun.store(false, Ordering::Relaxed);
        }
        // What played while the consumer was kept out came first.
        let skipped = self.skipped.swap(0, Ordering::Relaxed);
        if skipped > 0 {
            t.submit(0, skipped);
        }
        (first, got, note(t, got, missing))
    }

    /// [`Self::render_or_skip`] for device buffers that must be filled
    /// either way.
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    pub(crate) fn render(&self, out: &mut [u8]) -> u64 {
        self.render_or_skip(out).unwrap_or(0)
    }

    /// The device reports `device` device frames played at `at` (not
    /// counted while the handle keeps the consumer out).
    #[cfg_attr(target_os = "linux", allow(dead_code))]
    pub(crate) fn report(&self, device: u64, at: Instant, extrapolate: bool) {
        self.consume(|t| t.report(device, at, extrapolate));
    }

    /// The device reports `delay` submitted frames not yet played at `at`
    /// (not counted while the handle keeps the consumer out).
    pub(crate) fn report_delay(&self, delay: u64, at: Instant, extrapolate: bool) {
        self.consume(|t| t.report_delay(delay, at, extrapolate));
    }

    /// The device has stopped: the position holds until the next report.
    pub(crate) fn freeze(&self) {
        self.keep_out(Timeline::freeze);
    }

    /// The device itself ran dry (an xrun the driver reported).
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub(crate) fn note_device_underrun(&self) {
        self.device_underruns.fetch_add(1, Ordering::Relaxed);
    }

    /// The device failed; every later call on the handle returns `error`.
    pub(crate) fn fail(&self, error: Error) {
        let mut f = lock(&self.failure);
        if f.is_none() {
            *f = Some(error);
        }
        self.failed.store(true, Ordering::SeqCst);
        drop(f);
        self.space.notify_all();
    }

    // ---- handle side ----

    /// Keeps the consumer out of the ring until [`Self::unsuspend`].
    pub(crate) fn suspend(&self) {
        self.suspend.fetch_add(1, Ordering::SeqCst);
        while self.in_render.load(Ordering::SeqCst) {
            std::hint::spin_loop();
            std::thread::yield_now();
        }
    }

    pub(crate) fn unsuspend(&self) {
        self.suspend.fetch_sub(1, Ordering::SeqCst);
    }

    /// [`Self::suspend`] from inside the render (a test's tap): the
    /// consumer is kept out from its next render on.
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn hold_out(&self) {
        self.suspend.fetch_add(1, Ordering::SeqCst);
    }

    /// Discards everything queued and restarts the device timeline with
    /// every written frame counted as played. Call with the device stopped
    /// (or the consumer suspended) and before the device's own reset.
    pub(crate) fn discard_all(&self) {
        self.keep_out(|t| {
            self.ring.clear();
            let written = self.written();
            t.restart(written);
            self.skipped.store(0, Ordering::Relaxed);
            self.rendered.store(written, Ordering::Relaxed);
            self.in_underrun.store(false, Ordering::Relaxed);
            self.expect_empty.store(true, Ordering::Release);
        });
    }

    /// The device dropped what it had queued but the ring keeps its frames
    /// (a stop that cannot resume in place): the dropped frames count as
    /// played and the device timeline restarts at zero.
    #[cfg_attr(windows, allow(dead_code))]
    pub(crate) fn skip_device_queue(&self) {
        self.keep_out(|t| {
            t.restart(t.stream_submitted_total());
            self.skipped.store(0, Ordering::Relaxed);
        });
    }

    /// Frames are about to go into the ring: from before they can be seen
    /// there, an empty ring is not what is expected (a mixer looking
    /// between the two would take them for a drain's last).
    pub(crate) fn writing(&self) {
        self.expect_empty.store(false, Ordering::Release);
    }

    pub(crate) fn note_written(&self, frames: u64, clipped: u64, truncated: u64) {
        if clipped > 0 {
            self.clipped.fetch_add(clipped, Ordering::Relaxed);
        }
        if truncated > 0 {
            self.truncated.fetch_add(truncated, Ordering::Relaxed);
        }
        self.expect_empty.store(false, Ordering::Release);
        self.written.fetch_add(frames, Ordering::Release);
    }

    /// Frames written were taken back from the ring ([`Ring::retract`]).
    pub(crate) fn unwrite(&self, frames: u64) {
        self.written.fetch_sub(frames, Ordering::Release);
    }

    pub(crate) fn expect_empty(&self) {
        self.expect_empty.store(true, Ordering::Release);
    }

    /// Whether an empty ring is expected (the writer is draining it, or
    /// has not written yet): what it holds is all there is for now.
    pub(crate) fn draining(&self) -> bool {
        self.expect_empty.load(Ordering::Acquire)
    }

    /// Waits up to `timeout` for the consumer to free space.
    pub(crate) fn wait_for_space(&self, timeout: Duration) {
        let guard = lock(&self.space_lock);
        self.writer_waiting.store(true, Ordering::SeqCst);
        if self.ring.free() == 0 && !self.failed.load(Ordering::SeqCst) {
            // Bounded, so a wake-up that raced the flag costs at most this.
            let _ = self
                .space
                .wait_timeout(guard, timeout.min(Duration::from_millis(5)));
        }
        self.writer_waiting.store(false, Ordering::SeqCst);
    }

    /// Replaces the tap; the old one is dropped here before this returns.
    /// A device that pulls buffers on its own takes the new one at its next
    /// buffer: this waits for that while renders keep coming, and puts the
    /// tap in place itself, with the consumer kept out, once they have
    /// stopped for a few periods ([`Self::quiet`]). On a device that does
    /// not pull (stopped, or one that plays only when told) it does so at
    /// once. One caller at a time.
    pub(crate) fn set_tap(&self, tap: Option<Box<dyn Tap>>) {
        self.tap.collect();
        self.tap.post(tap, self.format.channels);
        if self.pulling.load(Ordering::Acquire) && self.taken_while_rendering() {
            return;
        }
        match self.tap.unpost() {
            Some(node) => {
                self.suspend();
                // SAFETY: the consumer is kept out.
                let old = unsafe { self.tap.install(node) };
                self.unsuspend();
                drop(old);
            }
            // The consumer took it after all and is handing the old one back.
            None => {
                while !self.tap.collect() {
                    std::thread::yield_now();
                }
            }
        }
    }

    /// Waits while the consumer has a posted tap to take, as long as its
    /// renders keep coming (at most [`TAP_WAIT`]); whether it took it (the
    /// old one dropped here).
    fn taken_while_rendering(&self) -> bool {
        let started = Instant::now();
        let (mut renders, mut moved) = (self.renders.load(Ordering::Acquire), started);
        loop {
            if self.tap.collect() {
                return true;
            }
            let now = Instant::now();
            let seen = self.renders.load(Ordering::Acquire);
            if seen != renders {
                (renders, moved) = (seen, now);
            } else if now - moved >= self.quiet() || now - started >= TAP_WAIT {
                return false;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// How long renders may stop before a device that pulls is taken for
    /// stopped: a few of its periods, and no less than a timer's slack on
    /// a busy system (a device thread woken that late still runs).
    fn quiet(&self) -> Duration {
        let period = self.period.load(Ordering::Relaxed);
        let period = self.format.frames_to_duration(period);
        (4 * period).clamp(Duration::from_millis(50), Duration::from_millis(500))
    }

    /// The device runs and pulls buffers on its own (or not).
    pub(crate) fn set_pulling(&self, pulling: bool) {
        self.pulling.store(pulling, Ordering::Release);
    }

    /// The device's period, in frames.
    pub(crate) fn set_period(&self, frames: u64) {
        self.period.store(frames, Ordering::Relaxed);
    }

    /// The handle closed the output; whether it was open.
    pub(crate) fn close(&self) -> bool {
        !self.closed.swap(true, Ordering::SeqCst)
    }

    // ---- any thread ----

    pub(crate) fn failure(&self) -> Option<Error> {
        if self.failed.load(Ordering::SeqCst) {
            lock(&self.failure).clone()
        } else {
            None
        }
    }

    /// The device's failure, if it failed.
    pub(crate) fn check(&self) -> Result<(), Error> {
        match self.failure() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// [`Error::Closed`] once the handle closed it, else [`Self::check`].
    pub(crate) fn usable(&self) -> Result<(), Error> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(Error::Closed);
        }
        self.check()
    }

    pub(crate) fn written(&self) -> u64 {
        self.written.load(Ordering::Acquire)
    }

    /// [`Stats::dropped`], cheaply.
    pub(crate) fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    pub(crate) fn position(&self) -> u64 {
        self.clock.position(Instant::now())
    }

    pub(crate) fn stats(&self) -> Stats {
        Stats {
            underruns: self.underruns.load(Ordering::Relaxed),
            underrun_frames: self.underrun_frames.load(Ordering::Relaxed),
            device_underruns: self.device_underruns.load(Ordering::Relaxed),
            clipped: self.clipped.load(Ordering::Relaxed),
            truncated: self.truncated.load(Ordering::Relaxed),
            dropped: self.dropped.load(Ordering::Relaxed),
        }
    }
}
