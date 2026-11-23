//! A single-producer, single-consumer ring of packed frames.
//!
//! The writer (the engine's thread, through the output handle) packs
//! samples straight into the ring; the device thread copies frames out of it
//! into the device buffer. Neither side locks. Counters are in whole frames,
//! so the wrap point always falls on a frame boundary.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub(crate) struct Ring {
    buf: Box<[UnsafeCell<u8>]>,
    frame_bytes: usize,
    capacity: u64,
    /// Frames ever pushed; only the producer stores it.
    head: AtomicU64,
    /// Frames ever popped, dropped or cleared; only the consumer stores it.
    tail: AtomicU64,
    /// Set while the consumer drops frames ([`Ring::skip`]), and while the
    /// producer takes frames back ([`Ring::retract`]): each side sets its
    /// own and then looks at the other's, so the two never overlap (a
    /// drop may pass the frames a retract keeps queued).
    skipping: AtomicBool,
    retracting: AtomicBool,
}

// SAFETY: the producer only writes bytes in the free region and the consumer
// only reads bytes in the filled region; the Release stores of `head` and
// `tail` publish those bytes to the other side, which loads with Acquire.
unsafe impl Sync for Ring {}
unsafe impl Send for Ring {}

impl Ring {
    pub(crate) fn new(capacity_frames: usize, frame_bytes: usize) -> Ring {
        let capacity_frames = capacity_frames.max(1);
        let buf = (0..capacity_frames * frame_bytes)
            .map(|_| UnsafeCell::new(0))
            .collect();
        Ring {
            buf,
            frame_bytes,
            capacity: capacity_frames as u64,
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            skipping: AtomicBool::new(false),
            retracting: AtomicBool::new(false),
        }
    }

    pub(crate) fn capacity(&self) -> u64 {
        self.capacity
    }

    pub(crate) fn len(&self) -> u64 {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        head - tail
    }

    pub(crate) fn free(&self) -> u64 {
        self.capacity - self.len()
    }

    fn base(&self) -> *mut u8 {
        // UnsafeCell<u8> has the layout of u8 and permits writes through a
        // shared reference.
        self.buf.as_ptr() as *mut u8
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
        let start = (head % self.capacity) as usize;
        let first = (frames as usize).min(self.capacity as usize - start);
        let second = frames as usize - first;
        let fb = self.frame_bytes;
        // SAFETY: [start, start + first) and [0, second) lie in the free
        // region, which the consumer does not read until `head` moves.
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
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        let frames = (head - tail).min((out.len() / fb) as u64);
        let start = (tail % self.capacity) as usize;
        let first = (frames as usize).min(self.capacity as usize - start);
        let second = frames as usize - first;
        // SAFETY: both ranges lie in the filled region, which the producer
        // does not write until `tail` moves.
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
        self.tail.store(tail + frames, Ordering::Release);
        frames
    }

    /// Consumer only: drops up to `frames` of the oldest frames queued and
    /// returns how many it dropped: none while the producer takes frames
    /// back (the consumer never waits: a follower refused gives nothing in
    /// that chunk, still late, and drops them in a later one).
    pub(crate) fn skip(&self, frames: u64) -> u64 {
        self.skipping.store(true, Ordering::SeqCst);
        let mut n = 0;
        if !self.retracting.load(Ordering::SeqCst) {
            let tail = self.tail.load(Ordering::Relaxed);
            let head = self.head.load(Ordering::Acquire);
            n = (head - tail).min(frames);
            self.tail.store(tail + n, Ordering::Release);
        }
        self.skipping.store(false, Ordering::SeqCst);
        n
    }

    /// Producer only: takes back up to `frames` of the frames pushed last,
    /// leaving at least `keep` queued, and returns how many it took back.
    /// The consumer may be copying from the oldest frames meanwhile: with
    /// `keep` at least the most it copies at a time, the frames taken back
    /// (which the producer writes again) are never among them. A drop of
    /// the consumer's under way is waited for (it takes an instant).
    pub(crate) fn retract(&self, frames: u64, keep: u64) -> u64 {
        self.retracting.store(true, Ordering::SeqCst);
        while self.skipping.load(Ordering::SeqCst) {
            std::hint::spin_loop();
            std::thread::yield_now();
        }
        let head = self.head.load(Ordering::Relaxed);
        // In the one order of `retracting`'s store: a consumer that saw it
        // unset after a fence of its own has its copies seen here.
        let tail = self.tail.load(Ordering::SeqCst);
        let n = frames.min((head - tail).saturating_sub(keep));
        self.head.store(head - n, Ordering::Release);
        self.retracting.store(false, Ordering::SeqCst);
        n
    }

    /// Drops everything queued. Only while neither side is active (the
    /// output quiesces the consumer first; the producer is the caller).
    pub(crate) fn clear(&self) {
        let head = self.head.load(Ordering::Acquire);
        self.tail.store(head, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::fence;
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
    }

    #[test]
    fn drops_and_retracts_on_two_threads_keep_the_order() {
        let ring = Arc::new(Ring::new(64, 4));
        let total = 200_000u32;
        let done = Arc::new(AtomicBool::new(false));
        let producer = {
            let (ring, done) = (ring.clone(), done.clone());
            std::thread::spawn(move || {
                let (mut next, mut k) = (0u32, 0u64);
                while next < total {
                    k += 1;
                    if k % 7 == 0 {
                        // The same frames are written again after.
                        next -= ring.retract(5, 16) as u32;
                    }
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
                done.store(true, Ordering::SeqCst);
            })
        };
        let (mut last, mut k, mut dropped) = (None, 0u64, 0);
        let mut out = [0u8; 4 * 13];
        loop {
            k += 1;
            // A head behind the tail would not even get here (the length
            // is a subtraction that cannot go below 0).
            assert!(ring.len() <= ring.capacity());
            if k % 5 == 0 {
                dropped += ring.skip(9);
            }
            let finished = done.load(Ordering::SeqCst);
            // A real consumer copies a period at a time, periods apart, so
            // within a retract it copies no more than the frames kept; this
            // one, as fast as it can go, waits each retract out instead (a
            // drop is the ring's own to keep apart from one). The fence has
            // its last copy seen by a retract it does not see begin (a store
            // and a later load are otherwise reordered, and it would copy
            // again past the tail the retract kept frames after).
            fence(Ordering::SeqCst);
            if ring.retracting.load(Ordering::SeqCst) {
                std::thread::yield_now();
                continue;
            }
            let n = ring.pop_into(&mut out) as usize;
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
        producer.join().unwrap();
        assert!(dropped > 0);
        assert!(last.is_none_or(|l| l < total));
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
