//! A single-producer, single-consumer ring of packed frames.
//!
//! The writer (the engine's thread, through the output handle) packs
//! samples straight into the ring; the device thread copies frames out of it
//! into the device buffer. Neither side locks, and neither ever waits for
//! the other. Counters are in whole frames, so the wrap point always falls
//! on a frame boundary.
//!
//! The producer may also take back frames it pushed that are not taken yet
//! ([`Ring::retract`]), so `head` can go down while the consumer copies.
//! The two keep apart through a claim the consumer posts before each move
//! and a limit the producer posts before lowering `head`; why that holds is
//! told at [`Ring::retract`].

use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) struct Ring {
    buf: Box<[std::cell::UnsafeCell<u8>]>,
    frame_bytes: usize,
    capacity: u64,
    /// Frames ever pushed, less those taken back; only the producer stores
    /// it.
    head: AtomicU64,
    /// Frames ever popped, dropped or cleared; the consumer stores it (and
    /// [`Ring::clear`], while the consumer is kept out).
    tail: AtomicU64,
    /// The frame the consumer's move under way may reach (posted before it
    /// looks at `head`), and where its last move ended between moves. Only
    /// the consumer stores it.
    claim: AtomicU64,
    /// While the producer takes frames back, the head it leaves at the
    /// least: the consumer moves no further. [`NO_LIMIT`] otherwise. Only
    /// the producer stores it.
    limit: AtomicU64,
    #[cfg(test)]
    hooks: hooks::Hooks,
}

const NO_LIMIT: u64 = u64::MAX;

/// The points inside a move of either side where a test can step in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Stage {
    /// The producer has worked out how far to take back, not posted yet.
    Planned,
    /// The producer has posted its limit and read the claim, and is about to
    /// lower `head`.
    Announced,
    /// The consumer has read `head` and is about to copy and move `tail`.
    Moving,
}

// SAFETY: the producer only writes bytes in the free region and the consumer
// only reads bytes in the filled region; the Release stores of `head` and
// `tail` publish those bytes to the other side, which loads with Acquire. A
// retract leaves `head` above every frame the consumer may be reading
// ([`Ring::retract`]), so the frames it hands back to the producer are never
// among them.
unsafe impl Sync for Ring {}
unsafe impl Send for Ring {}

/// Frames from `tail` to `head`. A head behind the tail is a broken ring:
/// the count would wrap to nearly 2^64, a push would then pass its space
/// check and write past the buffer. It stops here, in release builds too.
fn queued(tail: u64, head: u64) -> u64 {
    assert!(head >= tail, "ring head {head} behind its tail {tail}");
    head - tail
}

impl Ring {
    pub(crate) fn new(capacity_frames: usize, frame_bytes: usize) -> Ring {
        let capacity_frames = capacity_frames.max(1);
        let buf = (0..capacity_frames * frame_bytes)
            .map(|_| std::cell::UnsafeCell::new(0))
            .collect();
        Ring {
            buf,
            frame_bytes,
            capacity: capacity_frames as u64,
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            claim: AtomicU64::new(0),
            limit: AtomicU64::new(NO_LIMIT),
            #[cfg(test)]
            hooks: hooks::Hooks::default(),
        }
    }

    pub(crate) fn capacity(&self) -> u64 {
        self.capacity
    }

    pub(crate) fn len(&self) -> u64 {
        // The tail first: a head read after it is at least that tail.
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        queued(tail, head)
    }

    /// Producer or consumer only: a third thread may read a tail older than
    /// the head it reads next and count more than the capacity (its `len`
    /// stays right).
    pub(crate) fn free(&self) -> u64 {
        let len = self.len();
        assert!(
            len <= self.capacity,
            "ring holds {len} frames, more than its {}",
            self.capacity
        );
        self.capacity - len
    }

    fn base(&self) -> *mut u8 {
        // UnsafeCell<u8> has the layout of u8 and permits writes through a
        // shared reference.
        self.buf.as_ptr() as *mut u8
    }

    /// The one or two spans (start and length in frames, then the length
    /// from 0) that `frames` frames from counter `at` take in the buffer.
    /// They lie inside it and apart, or this panics.
    fn spans(&self, at: u64, frames: u64) -> (usize, usize, usize) {
        let cap = self.capacity as usize;
        let start = (at % self.capacity) as usize;
        let first = (frames.min(self.capacity) as usize).min(cap - start);
        let second = frames as usize - first;
        assert!(
            start + first <= cap && second <= start,
            "{frames} frames at {at} do not fit a ring of {cap}"
        );
        (start, first, second)
    }

    /// Producer only: makes room for `frames` frames (at most `free()`) and
    /// hands `fill` the space as one or two byte slices, in order.
    pub(crate) fn push_with<R>(
        &self,
        frames: u64,
        fill: impl FnOnce(&mut [u8], &mut [u8]) -> R,
    ) -> R {
        assert!(frames <= self.free(), "ring overfilled");
        let head = self.head.load(Ordering::Relaxed);
        let (start, first, second) = self.spans(head, frames);
        let fb = self.frame_bytes;
        // SAFETY: [start, start + first) and [0, second) lie in the buffer,
        // apart (`spans`), and in the free region, which the consumer does
        // not read until `head` moves.
        let (a, b) = unsafe {
            (
                std::slice::from_raw_parts_mut(self.base().add(start * fb), first * fb),
                std::slice::from_raw_parts_mut(self.base(), second * fb),
            )
        };
        let r = fill(a, b);
        self.head.store(head + frames, Ordering::Release);
        r
    }

    /// Consumer only: copies up to `out.len() / frame_bytes` frames into
    /// `out` and returns how many it copied.
    pub(crate) fn pop_into(&self, out: &mut [u8]) -> u64 {
        let fb = self.frame_bytes;
        self.advance((out.len() / fb) as u64, |tail, frames| {
            let (start, first, second) = self.spans(tail, frames);
            // SAFETY: both ranges lie in the buffer (`spans`) and in the
            // filled region below the head `advance` may go to, which the
            // producer does not write until `tail` moves, a retract
            // included; `out` holds `frames` frames.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.base().add(start * fb),
                    out.as_mut_ptr(),
                    first * fb,
                );
                std::ptr::copy_nonoverlapping(
                    self.base(),
                    out.as_mut_ptr().add(first * fb),
                    second * fb,
                );
            }
        })
    }

    /// Consumer only: drops up to `frames` of the oldest frames queued and
    /// returns how many it dropped. While the producer takes frames back,
    /// it drops no further than the head the producer leaves.
    pub(crate) fn skip(&self, frames: u64) -> u64 {
        self.advance(frames, |_, _| {})
    }

    /// Consumer only: moves `tail` on by up to `most` frames, as far as the
    /// producer lets it, first handing `take` the old tail and how many it
    /// passes over (to copy them). Returns how many. Never waits.
    fn advance(&self, most: u64, take: impl FnOnce(u64, u64)) -> u64 {
        let tail = self.tail.load(Ordering::Relaxed);
        // C1, then C2 and C3 (see `retract`).
        self.claim
            .store(tail.saturating_add(most), Ordering::SeqCst);
        let limit = self.limit.load(Ordering::SeqCst);
        let head = self.head.load(Ordering::Acquire);
        let n = queued(tail, head).min(most).min(limit.saturating_sub(tail));
        self.at(Stage::Moving);
        take(tail, n);
        self.tail.store(tail + n, Ordering::Release);
        // C4: between moves the claim is where the consumer is (a claim
        // left at `tail + most` would hold back a later retract for no
        // reason).
        self.claim.store(tail + n, Ordering::Release);
        n
    }

    /// Producer only: takes back up to `frames` of the frames pushed last,
    /// leaving at least `keep` queued, and returns how many it took back.
    /// Never waits.
    ///
    /// The consumer may be moving `tail` meanwhile. It posts how far a move
    /// may go before it looks at `head`, and the producer posts how low it
    /// will take `head` before it looks at that claim:
    ///
    /// - consumer: C1 store `claim` = tail + most (SeqCst); C2 load `limit`
    ///   (SeqCst); C3 load `head` (Acquire); move to at most the least of
    ///   the three; C4 store `claim` = where it ended (Release).
    /// - producer: plan `plan` from a tail read; P0 store `limit` = plan
    ///   (SeqCst); P1 load `claim` (SeqCst); P2 store `head` = max(plan,
    ///   min(claim, old head)) (Release); P3 store `limit` = none (Release).
    ///
    /// The SeqCst operations C1, C2, P0, P1 fall in one total order. Each of
    /// `claim` and `limit` has one writer, so every older store to it happens
    /// before the next SeqCst one: a SeqCst load reads the store just before
    /// it in that order or a later one, never an older. Take a move of the
    /// consumer's against a retract:
    ///
    /// - C1 before P1: P1 reads this claim or a later one, and every later
    ///   claim is at least where this move ends (C4 posts that end, and the
    ///   next move's claim starts from it). So the new head is at least
    ///   where the move ends, unless the move read a head that the retract
    ///   or a later push stored (C3), which already bounds it.
    /// - P1 before C1: then P0 comes before C2, so C2 reads `plan` (and the
    ///   move stops at or below it, which the new head is not below) or
    ///   the "none" of P3 or later. That store is Release and C2 Acquire,
    ///   so P2 is seen by C3, which then reads the new head or later.
    ///
    /// Either way the consumer neither reads frames the producer takes back
    /// (and writes again) nor moves `tail` past the new head. A retract
    /// during a move takes back nothing past `tail + most` (the whole
    /// request, even when fewer frames were queued): safe, and at worst it
    /// takes back less than it could. A move that begins after P0 waits for
    /// nothing; it stops at `plan`.
    pub(crate) fn retract(&self, frames: u64, keep: u64) -> u64 {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        let plan = head - frames.min(queued(tail, head).saturating_sub(keep));
        if plan == head {
            return 0;
        }
        self.at(Stage::Planned);
        self.limit.store(plan, Ordering::SeqCst);
        let claim = self.claim.load(Ordering::SeqCst);
        let new_head = plan.max(claim.min(head));
        self.at(Stage::Announced);
        self.head.store(new_head, Ordering::Release);
        self.limit.store(NO_LIMIT, Ordering::Release);
        head - new_head
    }

    /// Drops everything queued. Only while neither side is active (the
    /// output quiesces the consumer first; the producer is the caller).
    pub(crate) fn clear(&self) {
        let head = self.head.load(Ordering::Acquire);
        self.tail.store(head, Ordering::Release);
    }

    #[cfg(not(test))]
    #[inline(always)]
    fn at(&self, _stage: Stage) {}

    /// Lets a test step in at `stage` ([`hooks::Hooks`]).
    #[cfg(test)]
    fn at(&self, stage: Stage) {
        self.hooks.at(self, stage);
    }
}

#[cfg(test)]
mod hooks {
    use super::{Ring, Stage};
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::sync::Mutex;

    type Once = Box<dyn FnOnce(&Ring) + Send>;

    /// Where a test steps into a ring's moves.
    #[derive(Default)]
    pub(super) struct Hooks {
        /// Times either side yields at each of its stages.
        pub(super) stall: AtomicU32,
        armed: AtomicBool,
        once: Mutex<Option<(Stage, Once)>>,
    }

    impl Hooks {
        /// Runs `f` once, the next time a side reaches `stage`, on its
        /// thread (`f` may move the ring as the other side).
        pub(super) fn once(&self, stage: Stage, f: impl FnOnce(&Ring) + Send + 'static) {
            *self.once.lock().unwrap() = Some((stage, Box::new(f)));
            self.armed.store(true, Ordering::SeqCst);
        }

        pub(super) fn at(&self, ring: &Ring, stage: Stage) {
            for _ in 0..self.stall.load(Ordering::Relaxed) {
                std::thread::yield_now();
            }
            if !self.armed.load(Ordering::SeqCst) {
                return;
            }
            let hook = {
                let mut once = self.once.lock().unwrap();
                match once.take() {
                    Some((s, f)) if s == stage => Some(f),
                    other => {
                        *once = other;
                        None
                    }
                }
            };
            if let Some(f) = hook {
                self.armed.store(false, Ordering::SeqCst);
                f(ring);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    fn push(ring: &Ring, bytes: &[u8]) {
        let frames = (bytes.len() / ring.frame_bytes) as u64;
        ring.push_with(frames, |a, b| {
            a.copy_from_slice(&bytes[..a.len()]);
            b.copy_from_slice(&bytes[a.len()..]);
        });
    }

    #[test]
    fn wraps_on_frame_boundaries() {
        let ring = Ring::new(4, 3);
        push(&ring, &[1, 1, 1, 2, 2, 2, 3, 3, 3]);
        let mut out = [0; 6];
        assert_eq!(ring.pop_into(&mut out), 2);
        assert_eq!(out, [1, 1, 1, 2, 2, 2]);
        assert_eq!(ring.free(), 3);
        // Three more frames: one at the end, two at the start.
        ring.push_with(3, |a, b| {
            assert_eq!((a.len(), b.len()), (3, 6));
            a.fill(4);
            b[..3].fill(5);
            b[3..].fill(6);
        });
        assert_eq!(ring.len(), 4);
        let mut out = [0; 15];
        assert_eq!(ring.pop_into(&mut out), 4);
        assert_eq!(out[..12], [3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]);
        assert_eq!(ring.pop_into(&mut out), 0);
    }

    #[test]
    fn clear_drops_everything() {
        let ring = Ring::new(8, 2);
        push(&ring, &[1; 10]);
        ring.clear();
        assert_eq!(ring.len(), 0);
        assert_eq!(ring.free(), 8);
    }

    #[test]
    #[should_panic(expected = "ring overfilled")]
    fn a_push_past_the_capacity_panics() {
        let ring = Ring::new(4, 1);
        push(&ring, &[1, 2, 3]);
        push(&ring, &[4, 5]);
    }

    #[test]
    fn retract_takes_back_the_newest_frames_and_keeps_some() {
        let ring = Ring::new(8, 1);
        push(&ring, &[1, 2, 3, 4, 5, 6]);
        assert_eq!(ring.retract(10, 2), 4);
        assert_eq!(ring.len(), 2);
        push(&ring, &[7, 8]);
        let mut out = [0; 8];
        assert_eq!(ring.pop_into(&mut out), 4);
        assert_eq!(out[..4], [1, 2, 7, 8]);
        // Nothing beyond what is kept.
        push(&ring, &[9]);
        assert_eq!(ring.retract(5, 1), 0);
        assert_eq!(ring.len(), 1);
    }

    #[test]
    fn skip_drops_the_oldest_frames_queued() {
        let ring = Ring::new(8, 1);
        push(&ring, &[1, 2, 3, 4, 5]);
        assert_eq!(ring.skip(3), 3);
        let mut out = [0; 8];
        assert_eq!(ring.pop_into(&mut out), 2);
        assert_eq!(out[..2], [4, 5]);
        push(&ring, &[6]);
        assert_eq!(ring.skip(10), 1);
        assert_eq!(ring.len(), 0);
        assert_eq!(ring.skip(u64::MAX), 0);
    }

    /// The race two reviews found: the producer reads the tail, the consumer
    /// then copies more than the frames kept, and only then is the head
    /// lowered. It must not go below the tail.
    #[test]
    fn a_consumer_that_moves_on_before_the_retract_posts_keeps_its_frames() {
        let ring = Ring::new(16, 1);
        push(&ring, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let copied = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = copied.clone();
        ring.hooks.once(Stage::Planned, move |ring| {
            let mut out = [0; 8];
            let n = ring.pop_into(&mut out) as usize;
            seen.lock().unwrap().extend_from_slice(&out[..n]);
        });
        // Planned from tail 0: keep 2, so take back 8 (down to 2).
        assert_eq!(ring.retract(10, 2), 2);
        assert_eq!(*copied.lock().unwrap(), [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(ring.len(), 0);
        assert_eq!(ring.free(), 16);
        push(&ring, &[11, 12]);
        let mut out = [0; 4];
        assert_eq!(ring.pop_into(&mut out), 2);
        assert_eq!(out[..2], [11, 12]);
    }

    /// A consumer that begins once the retract has posted its limit stops
    /// there, copying or dropping.
    #[test]
    fn a_consumer_that_sees_the_retract_posted_stops_at_its_plan() {
        let ring = Ring::new(16, 1);
        push(&ring, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let copied = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = copied.clone();
        ring.hooks.once(Stage::Announced, move |ring| {
            let mut out = [0; 8];
            let n = ring.pop_into(&mut out) as usize;
            seen.lock().unwrap().extend_from_slice(&out[..n]);
        });
        assert_eq!(ring.retract(10, 2), 8);
        assert_eq!(*copied.lock().unwrap(), [1, 2]);
        assert_eq!(ring.len(), 0);

        push(&ring, &[3, 4, 5, 6, 7, 8]);
        let dropped = Arc::new(AtomicU64::new(0));
        let got = dropped.clone();
        ring.hooks.once(Stage::Announced, move |ring| {
            got.store(ring.skip(10), Ordering::SeqCst);
        });
        assert_eq!(ring.retract(10, 1), 5);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        assert_eq!(ring.len(), 0);
    }

    /// A retract (and the frames written again after it) while the consumer
    /// copies leaves every frame it copies.
    #[test]
    fn a_retract_during_a_copy_leaves_the_frames_being_copied() {
        let ring = Ring::new(16, 1);
        push(&ring, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let back = Arc::new(AtomicU64::new(u64::MAX));
        let got = back.clone();
        ring.hooks.once(Stage::Moving, move |ring| {
            let n = ring.retract(10, 2);
            got.store(n, Ordering::SeqCst);
            push(ring, &vec![99; n as usize]);
        });
        let mut out = [0; 8];
        assert_eq!(ring.pop_into(&mut out), 8);
        assert_eq!(out, [1, 2, 3, 4, 5, 6, 7, 8]);
        // Planned down to 2, but the copy claimed up to 8.
        assert_eq!(back.load(Ordering::SeqCst), 2);
        assert_eq!(ring.len(), 2);
        assert_eq!(ring.pop_into(&mut out), 2);
        assert_eq!(out[..2], [99, 99]);
        // Its claim is cleared once it is done: a retract now takes all
        // but what it keeps.
        push(&ring, &[9, 10, 11, 12]);
        assert_eq!(ring.retract(10, 1), 3);
        assert_eq!(ring.len(), 1);
    }

    /// How [`pushes_retracts_copies_and_drops`] runs.
    struct Race {
        /// Times each side yields at each stage inside its moves.
        stall: u32,
        /// The producer fills the ring up to this many frames queued.
        depth: u64,
        /// Every so often it takes back up to `back` frames, keeping `keep`.
        back: u64,
        keep: u64,
        /// Frames numbered, all told.
        total: u32,
    }

    /// The producer pushes numbered frames and now and then takes some
    /// back and writes the same numbers again; the consumer copies (13
    /// frames a move) and drops.
    fn pushes_retracts_copies_and_drops(race: Race) {
        let Race {
            stall,
            depth,
            back,
            keep,
            total,
        } = race;
        let ring = Arc::new(Ring::new(64, 4));
        ring.hooks.stall.store(stall, Ordering::Relaxed);
        let done = Arc::new(AtomicBool::new(false));
        let producer = {
            let (ring, done) = (ring.clone(), done.clone());
            std::thread::spawn(move || {
                let (mut next, mut k, mut taken) = (0u32, 0u64, 0u64);
                while next < total {
                    k += 1;
                    if k % 7 == 0 {
                        // The same frames are written again after.
                        let n = ring.retract(back, keep);
                        next -= n as u32;
                        taken += n;
                    }
                    let room = depth.saturating_sub(ring.len()).min(ring.free());
                    let n = room.min(u64::from(total - next)).min(11);
                    if n == 0 {
                        std::thread::yield_now();
                        continue;
                    }
                    ring.push_with(n, |a, b| {
                        for chunk in a.chunks_exact_mut(4).chain(b.chunks_exact_mut(4)) {
                            chunk.copy_from_slice(&next.to_le_bytes());
                            next += 1;
                        }
                    });
                }
                done.store(true, Ordering::SeqCst);
                taken
            })
        };
        let (mut last, mut k, mut dropped, mut copied) = (None, 0u64, 0, 0u64);
        let mut out = [0u8; 4 * 13];
        loop {
            k += 1;
            // A head behind the tail panics in `len` itself.
            assert!(ring.len() <= ring.capacity());
            if k % 5 == 0 {
                dropped += ring.skip(9);
            }
            let finished = done.load(Ordering::SeqCst);
            let n = ring.pop_into(&mut out) as usize;
            copied += n as u64;
            for chunk in out[..n * 4].chunks_exact(4) {
                let v = u32::from_le_bytes(chunk.try_into().unwrap());
                assert!(last.is_none_or(|l| v > l), "{v} after {last:?}");
                last = Some(v);
            }
            if finished && ring.len() == 0 {
                break;
            }
            if n == 0 {
                std::thread::yield_now();
            }
        }
        let taken = producer.join().unwrap();
        assert!(dropped > 0 && copied > 0);
        assert!(taken > 0, "no retract took anything back");
        assert!(last.is_none_or(|l| l < total));
    }

    #[test]
    fn drops_and_retracts_on_two_threads_keep_the_order() {
        pushes_retracts_copies_and_drops(Race {
            stall: 0,
            depth: 64,
            back: 5,
            keep: 16,
            total: 200_000,
        });
    }

    /// The ring holds little more than a copy, the producer takes back all
    /// but 1 frame (less than a copy), and both sides stall inside their
    /// moves: retracts land on copies under way and copies on retracts
    /// posted. Without the claim and the limit the head falls behind the
    /// tail.
    #[test]
    fn copies_racing_retracts_that_keep_less_than_a_copy_keep_the_order() {
        pushes_retracts_copies_and_drops(Race {
            stall: 3,
            depth: 20,
            back: 64,
            keep: 1,
            total: 50_000,
        });
    }

    #[test]
    fn threads_see_every_byte_in_order() {
        let ring = Arc::new(Ring::new(37, 4));
        let total = 50_000u32;
        let producer = {
            let ring = ring.clone();
            std::thread::spawn(move || {
                let mut next = 0u32;
                while next < total {
                    let n = ring.free().min(u64::from(total - next)).min(11);
                    if n == 0 {
                        std::thread::yield_now();
                        continue;
                    }
                    ring.push_with(n, |a, b| {
                        for chunk in a.chunks_exact_mut(4).chain(b.chunks_exact_mut(4)) {
                            chunk.copy_from_slice(&next.to_le_bytes());
                            next += 1;
                        }
                    });
                }
            })
        };
        let mut expected = 0u32;
        let mut out = [0u8; 4 * 13];
        while expected < total {
            let n = ring.pop_into(&mut out) as usize;
            for chunk in out[..n * 4].chunks_exact(4) {
                assert_eq!(u32::from_le_bytes(chunk.try_into().unwrap()), expected);
                expected += 1;
            }
            if n == 0 {
                std::thread::yield_now();
            }
        }
        producer.join().unwrap();
    }
}
