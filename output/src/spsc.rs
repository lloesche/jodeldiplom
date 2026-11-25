//! A single-producer, single-consumer queue of `Copy` values, lock-free and
//! allocation-free after it is made.
//!
//! Each end is a value: a [`Producer`] or a [`Consumer`], claimed by one
//! owner at a time ([`Spsc::producer`], [`Spsc::consumer`]) and given back
//! when it is dropped, for the next owner to claim. An end that stays with
//! one owner is claimed by it as the queue is made. Claiming is one
//! compare-and-swap, so a real-time thread may claim; nothing waits.
//!
//! Positions count every value ever pushed or popped (`u64`: they never
//! wrap), so the queue's length is `head - tail` and a full queue is told
//! from an empty one without a spare slot.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// A value on a cache line of its own (128 bytes: two lines where the
/// prefetcher pulls pairs, one on Apple silicon), so the two ends' stores
/// do not contend.
#[repr(align(128))]
struct Line<T>(T);

impl<T> Deref for Line<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

/// The queue both ends share.
pub(crate) struct Spsc<T> {
    slots: Box<[UnsafeCell<MaybeUninit<T>>]>,
    /// Values ever pushed; only the producer stores it.
    head: Line<AtomicU64>,
    /// Values ever popped or skipped; only the consumer stores it.
    tail: Line<AtomicU64>,
    /// A [`Producer`] exists; a [`Consumer`] exists.
    producing: AtomicBool,
    consuming: AtomicBool,
}

// SAFETY: there is at most one `Producer` and one `Consumer` at a time (each
// is claimed by a compare-and-swap of its flag and given back when
// dropped), and only they touch the slots: the producer writes only slots
// in the free region `[head, tail + capacity)`, the consumer reads only
// slots in the filled region `[tail, head)`. The producer's Release store
// of `head` publishes the values it wrote to the consumer's Acquire load;
// the consumer's Release store of `tail` hands the slots it read back to
// the producer's Acquire load. An end given back (Release) and claimed
// again (Acquire) carries its owner's last position to the next owner.
// `T: Copy` has no drop, so values left in the slots need none.
unsafe impl<T: Send> Sync for Spsc<T> {}

impl<T: Copy> Spsc<T> {
    /// A queue of `capacity` values (at least one), neither end claimed.
    pub(crate) fn new(capacity: usize) -> Arc<Spsc<T>> {
        Arc::new(Spsc {
            slots: (0..capacity.max(1))
                .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
                .collect(),
            head: Line(AtomicU64::new(0)),
            tail: Line(AtomicU64::new(0)),
            producing: AtomicBool::new(false),
            consuming: AtomicBool::new(false),
        })
    }

    /// Claims the producer, unless another owner has it.
    pub(crate) fn producer(self: &Arc<Self>) -> Option<Producer<T>> {
        claim(&self.producing).then(|| Producer {
            queue: Arc::clone(self),
        })
    }

    /// Claims the consumer, unless another owner has it.
    pub(crate) fn consumer(self: &Arc<Self>) -> Option<Consumer<T>> {
        claim(&self.consuming).then(|| Consumer {
            queue: Arc::clone(self),
        })
    }

    /// Whether someone holds the consumer now.
    pub(crate) fn has_consumer(&self) -> bool {
        self.consuming.load(Ordering::Acquire)
    }

    fn capacity(&self) -> u64 {
        self.slots.len() as u64
    }

    fn slot(&self, position: u64) -> *mut MaybeUninit<T> {
        self.slots[(position % self.capacity()) as usize].get()
    }

    /// `(head, tail)` as the producer sees them: `head` its own (`Relaxed`:
    /// only it stores it), `tail` the consumer's.
    fn producer_view(&self) -> (u64, u64) {
        let head = self.head.load(Ordering::Relaxed);
        self.checked(head, self.tail.load(Ordering::Acquire))
    }

    /// `(head, tail)` as the consumer sees them: `tail` its own.
    fn consumer_view(&self) -> (u64, u64) {
        let tail = self.tail.load(Ordering::Relaxed);
        self.checked(self.head.load(Ordering::Acquire), tail)
    }

    fn checked(&self, head: u64, tail: u64) -> (u64, u64) {
        debug_assert!(tail <= head && head - tail <= self.capacity());
        (head, tail)
    }
}

fn claim(flag: &AtomicBool) -> bool {
    flag.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_ok()
}

/// The end that pushes; one at a time.
pub(crate) struct Producer<T: Copy> {
    queue: Arc<Spsc<T>>,
}

impl<T: Copy> Producer<T> {
    /// The room left.
    pub(crate) fn free(&self) -> usize {
        let (head, tail) = self.queue.producer_view();
        (self.queue.capacity() - (head - tail)) as usize
    }

    /// Queues `value`; `false` (and nothing queued) when the queue is full.
    pub(crate) fn push(&mut self, value: T) -> bool {
        self.push_slice(std::slice::from_ref(&value)) == 1
    }

    /// Queues what fits of `values`, in order; returns how many.
    pub(crate) fn push_slice(&mut self, values: &[T]) -> usize {
        let q = &*self.queue;
        let (head, tail) = q.producer_view();
        let n = values.len().min((q.capacity() - (head - tail)) as usize);
        for (k, &v) in values[..n].iter().enumerate() {
            // SAFETY: the slot is in the free region, which only this end
            // touches (see `Spsc`'s Sync).
            unsafe { (*q.slot(head + k as u64)).write(v) };
        }
        q.head.store(head + n as u64, Ordering::Release);
        n
    }
}

impl<T: Copy> Drop for Producer<T> {
    fn drop(&mut self) {
        self.queue.producing.store(false, Ordering::Release);
    }
}

/// The end that pops; one at a time.
pub(crate) struct Consumer<T: Copy> {
    queue: Arc<Spsc<T>>,
}

impl<T: Copy> Consumer<T> {
    /// The values queued.
    pub(crate) fn len(&self) -> usize {
        let (head, tail) = self.queue.consumer_view();
        (head - tail) as usize
    }

    /// Takes the oldest value.
    pub(crate) fn pop(&mut self) -> Option<T> {
        let mut value = None;
        self.take(1, |_, v| value = Some(v));
        value
    }

    /// Takes up to `out.len()` of the oldest values; returns how many.
    pub(crate) fn pop_slice(&mut self, out: &mut [T]) -> usize {
        self.take(out.len(), |k, v| out[k] = v)
    }

    /// Drops up to `n` of the oldest values; returns how many.
    pub(crate) fn skip(&mut self, n: usize) -> usize {
        self.take(n, |_, _| ())
    }

    /// Hands up to `most` of the oldest values to `each`, with their index;
    /// returns how many.
    fn take(&mut self, most: usize, mut each: impl FnMut(usize, T)) -> usize {
        let q = &*self.queue;
        let (head, tail) = q.consumer_view();
        let n = most.min((head - tail) as usize);
        for k in 0..n {
            // SAFETY: the slot is in the filled region: the producer wrote
            // it before the `head` just loaded, and only this end reads it.
            each(k, unsafe { (*q.slot(tail + k as u64)).assume_init_read() });
        }
        q.tail.store(tail + n as u64, Ordering::Release);
        n
    }
}

impl<T: Copy> Drop for Consumer<T> {
    fn drop(&mut self) {
        self.queue.consuming.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_wrap_in_order_and_are_counted() {
        let queue = Spsc::new(5);
        let (mut tx, mut rx) = (queue.producer().unwrap(), queue.consumer().unwrap());
        assert_eq!(tx.push_slice(&[1.0f32, 2.0, 3.0, 4.0]), 4);
        let mut out = [0.0; 3];
        assert_eq!(rx.pop_slice(&mut out), 3);
        assert_eq!(out, [1.0, 2.0, 3.0]);
        assert_eq!(tx.push_slice(&[5.0, 6.0, 7.0, 8.0, 9.0]), 4);
        assert_eq!((tx.free(), rx.len()), (0, 5));
        assert!(!tx.push(10.0));
        assert_eq!(rx.skip(2), 2);
        assert_eq!(rx.pop(), Some(6.0));
        let mut out = [0.0; 8];
        assert_eq!(rx.pop_slice(&mut out), 2);
        assert_eq!(out[..2], [7.0, 8.0]);
        assert_eq!((rx.pop(), rx.skip(3)), (None, 0));
    }

    #[test]
    fn each_end_has_one_owner_at_a_time() {
        let queue = Spsc::<u32>::new(4);
        let mut tx = queue.producer().unwrap();
        assert!(queue.producer().is_none());
        assert!(!queue.has_consumer());
        let mut rx = queue.consumer().unwrap();
        assert!(queue.has_consumer() && queue.consumer().is_none());
        assert!(tx.push(1) && tx.push(2));
        assert_eq!(rx.pop(), Some(1));
        drop(rx);
        // The next owner goes on where the last one stopped.
        let mut rx = queue.consumer().unwrap();
        assert_eq!(rx.pop(), Some(2));
        drop(tx);
        let mut tx = queue.producer().unwrap();
        assert!(tx.push(3));
        assert_eq!(rx.pop(), Some(3));
    }

    /// A value much larger than a word, as the mic path queues: whole, in
    /// order, or not at all.
    #[test]
    fn large_values_cross_threads_whole() {
        const N: u64 = if cfg!(miri) { 200 } else { 50_000 };
        let queue = Spsc::<(u64, [u64; 16])>::new(7);
        let (mut tx, mut rx) = (queue.producer().unwrap(), queue.consumer().unwrap());
        let producer = std::thread::spawn(move || {
            let mut k = 0;
            while k < N {
                if tx.push((k, [k; 16])) {
                    k += 1;
                } else {
                    std::thread::yield_now();
                }
            }
        });
        let mut want = 0;
        while want < N {
            match rx.pop() {
                Some((k, body)) => {
                    assert_eq!((k, body), (want, [want; 16]));
                    want += 1;
                }
                None => std::thread::yield_now(),
            }
        }
        producer.join().unwrap();
    }

    /// The consumer passed between threads while the producer runs: every
    /// value is taken once, and each taker sees its own in order.
    #[test]
    fn an_end_handed_over_keeps_its_place() {
        const N: u64 = if cfg!(miri) { 300 } else { 100_000 };
        let queue = Spsc::<u64>::new(16);
        let mut tx = queue.producer().unwrap();
        let producer = std::thread::spawn(move || {
            let mut k = 0;
            while k < N {
                if tx.push(k) {
                    k += 1;
                } else {
                    std::thread::yield_now();
                }
            }
        });
        let takers: Vec<_> = (0..3)
            .map(|_| {
                let queue = Arc::clone(&queue);
                std::thread::spawn(move || {
                    let mut seen = Vec::new();
                    while queue.tail.load(Ordering::Acquire) < N {
                        if let Some(mut rx) = queue.consumer() {
                            let mut buf = [0u64; 5];
                            let n = rx.pop_slice(&mut buf);
                            seen.extend_from_slice(&buf[..n]);
                        }
                        std::thread::yield_now();
                    }
                    seen
                })
            })
            .collect();
        producer.join().unwrap();
        let mut all = Vec::new();
        for taker in takers {
            let seen = taker.join().unwrap();
            assert!(seen.windows(2).all(|w| w[0] < w[1]));
            all.extend(seen);
        }
        all.sort_unstable();
        assert_eq!(all, (0..N).collect::<Vec<_>>());
    }
}
