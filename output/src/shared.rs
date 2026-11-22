//! The state an output handle shares with its device thread.
//!
//! The handle packs samples into the [`Ring`]; the backend's device thread
//! (or IOProc) calls [`Shared::render`] for every device buffer, which
//! copies queued frames out and pads with silence when the ring runs dry.
//! Position, statistics and failures are published here for any thread to
//! read. A [`Tap`] set on the output sees every buffer here too, and
//! cannot change it.

use crate::error::Error;
use crate::format::StreamFormat;
use crate::output::Stats;
use crate::ring::Ring;
use crate::tap::{Tap, TapSlot};
use crate::timeline::Timeline;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

pub(crate) struct Shared {
    pub(crate) format: StreamFormat,
    pub(crate) ring: Ring,
    timeline: Mutex<Timeline>,
    /// An empty ring is expected (after a drain, before the next write), so
    /// padding is not an underrun.
    expect_empty: AtomicBool,
    /// The consumer padded its last buffer: an underrun is counted once per
    /// run of padded buffers with no stream frames between them.
    in_underrun: AtomicBool,
    /// Set while the handle needs the consumer to keep out of the ring.
    suspend: AtomicBool,
    /// Set while the consumer is inside the ring.
    in_render: AtomicBool,
    underruns: AtomicU64,
    underrun_frames: AtomicU64,
    device_underruns: AtomicU64,
    clipped: AtomicU64,
    truncated: AtomicU64,
    written: AtomicU64,
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
            timeline: Mutex::new(Timeline::new(format.rate)),
            expect_empty: AtomicBool::new(true),
            in_underrun: AtomicBool::new(false),
            suspend: AtomicBool::new(false),
            in_render: AtomicBool::new(false),
            underruns: AtomicU64::new(0),
            underrun_frames: AtomicU64::new(0),
            device_underruns: AtomicU64::new(0),
            clipped: AtomicU64::new(0),
            truncated: AtomicU64::new(0),
            written: AtomicU64::new(0),
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
    /// has suspended the consumer, in which case `out` is all silence and
    /// nothing is recorded.
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

    /// Of what went downstream since the timeline's restart `epoch`, only
    /// `frames` are not played yet.
    pub(crate) fn set_downstream(&self, epoch: u64, frames: u64) {
        lock(&self.timeline).set_downstream(epoch, frames);
    }

    /// The render: `note` records it in the timeline. Returns the first
    /// stream frame copied, how many, and what `note` gave.
    fn render_with<R>(
        &self,
        out: &mut [u8],
        note: impl FnOnce(&mut Timeline, u64, u64) -> R,
    ) -> Option<(u64, u64, R)> {
        if self.suspend.load(Ordering::SeqCst) {
            out.fill(0);
            return None;
        }
        self.in_render.store(true, Ordering::SeqCst);
        if self.suspend.load(Ordering::SeqCst) {
            self.in_render.store(false, Ordering::SeqCst);
            out.fill(0);
            return None;
        }
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
        let noted = note(&mut lock(&self.timeline), got, missing);
        self.in_render.store(false, Ordering::SeqCst);
        if got > 0 && self.writer_waiting.load(Ordering::SeqCst) {
            self.space.notify_all();
        }
        Some((first, got, noted))
    }

    /// [`Self::render_or_skip`] for device buffers that must be filled
    /// either way.
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    pub(crate) fn render(&self, out: &mut [u8]) -> u64 {
        self.render_or_skip(out).unwrap_or(0)
    }

    /// The device reports `device` device frames played at `at`.
    #[cfg_attr(target_os = "linux", allow(dead_code))]
    pub(crate) fn report(&self, device: u64, at: Instant, extrapolate: bool) {
        lock(&self.timeline).report(device, at, extrapolate);
    }

    /// The device reports `delay` submitted frames not yet played at `at`.
    pub(crate) fn report_delay(&self, delay: u64, at: Instant, extrapolate: bool) {
        lock(&self.timeline).report_delay(delay, at, extrapolate);
    }

    /// The device has stopped: the position holds until the next report.
    pub(crate) fn freeze(&self) {
        lock(&self.timeline).freeze();
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
        self.suspend.store(true, Ordering::SeqCst);
        while self.in_render.load(Ordering::SeqCst) {
            std::hint::spin_loop();
            std::thread::yield_now();
        }
    }

    pub(crate) fn unsuspend(&self) {
        self.suspend.store(false, Ordering::SeqCst);
    }

    /// Discards everything queued and restarts the device timeline with
    /// every written frame counted as played. Call with the device stopped
    /// (or the consumer suspended) and before the device's own reset.
    pub(crate) fn discard_all(&self) {
        self.suspend();
        self.ring.clear();
        lock(&self.timeline).restart(self.written());
        self.rendered.store(self.written(), Ordering::Relaxed);
        self.in_underrun.store(false, Ordering::Relaxed);
        self.expect_empty.store(true, Ordering::Release);
        self.unsuspend();
    }

    /// The device dropped what it had queued but the ring keeps its frames
    /// (a stop that cannot resume in place): the dropped frames count as
    /// played and the device timeline restarts at zero.
    #[cfg_attr(windows, allow(dead_code))]
    pub(crate) fn skip_device_queue(&self) {
        let mut t = lock(&self.timeline);
        let base = t.stream_submitted_total();
        t.restart(base);
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

    /// Replaces the tap. While the consumer renders, it takes the new one
    /// at its next buffer (waiting up to `wait` for that); otherwise it is
    /// installed with the consumer kept out. Either way the old tap is
    /// dropped here before this returns.
    pub(crate) fn set_tap(&self, tap: Option<Box<dyn Tap>>, wait: Duration) {
        self.tap.collect();
        self.tap.post(tap, self.format.channels);
        let deadline = Instant::now() + wait;
        while Instant::now() < deadline {
            if self.tap.collect() {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
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

    // ---- any thread ----

    pub(crate) fn failure(&self) -> Option<Error> {
        if self.failed.load(Ordering::SeqCst) {
            lock(&self.failure).clone()
        } else {
            None
        }
    }

    pub(crate) fn check(&self) -> Result<(), Error> {
        match self.failure() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    pub(crate) fn written(&self) -> u64 {
        self.written.load(Ordering::Acquire)
    }

    pub(crate) fn position(&self) -> u64 {
        lock(&self.timeline).position(Instant::now())
    }

    pub(crate) fn stats(&self) -> Stats {
        Stats {
            underruns: self.underruns.load(Ordering::Relaxed),
            underrun_frames: self.underrun_frames.load(Ordering::Relaxed),
            device_underruns: self.device_underruns.load(Ordering::Relaxed),
            clipped: self.clipped.load(Ordering::Relaxed),
            truncated: self.truncated.load(Ordering::Relaxed),
        }
    }
}
