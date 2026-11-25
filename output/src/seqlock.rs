//! A value written by one thread at a time and read by any, whole, with no
//! lock: a sequence lock over the value's words.
//!
//! The writer bumps the count to odd, stores the words and bumps it back
//! to even; a reader that sees the same even count before and after its
//! loads has all of one write. The writer never waits for a reader. It is
//! a real-time thread (the device's, the mixer's) whose write is a dozen
//! stores, so a reader retries a few times at most; one that keeps seeing
//! a write under way (its writer preempted in the middle) yields, so the
//! writer can finish.

use std::marker::PhantomData;
use std::sync::atomic::{fence, AtomicU64, Ordering};

/// The most words a value takes.
const MAX_WORDS: usize = 12;

/// Spins a reader makes before it yields between tries.
const SPINS: u32 = 64;

/// A value a [`SeqLock`] holds: `WORDS` words, at most [`MAX_WORDS`].
pub(crate) trait Packed: Copy {
    const WORDS: usize;
    /// Its words, into `words` (`WORDS` long).
    fn pack(&self, words: &mut [u64]);
    /// The value of `words` (`WORDS` long).
    fn unpack(words: &[u64]) -> Self;
}

pub(crate) struct SeqLock<T: Packed> {
    seq: AtomicU64,
    words: [AtomicU64; MAX_WORDS],
    value: PhantomData<T>,
}

impl<T: Packed + Default> Default for SeqLock<T> {
    fn default() -> Self {
        SeqLock::new(T::default())
    }
}

impl<T: Packed> SeqLock<T> {
    pub(crate) fn new(value: T) -> SeqLock<T> {
        const { assert!(T::WORDS <= MAX_WORDS) };
        let lock = SeqLock {
            seq: AtomicU64::new(0),
            words: std::array::from_fn(|_| AtomicU64::new(0)),
            value: PhantomData,
        };
        lock.store(&value);
        lock
    }

    /// The writer (one at a time): `value` from now on.
    pub(crate) fn store(&self, value: &T) {
        let mut words = [0; MAX_WORDS];
        value.pack(&mut words[..T::WORDS]);
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        debug_assert!(seq % 2 == 0, "two writers at once");
        fence(Ordering::Release);
        for (w, &v) in self.words.iter().zip(&words[..T::WORDS]) {
            w.store(v, Ordering::Relaxed);
        }
        self.seq.store(seq + 2, Ordering::Release);
    }

    /// The value of one write.
    pub(crate) fn load(&self) -> T {
        let mut words = [0; MAX_WORDS];
        let mut tries = 0u32;
        loop {
            let seq = self.seq.load(Ordering::Acquire);
            if seq % 2 == 0 {
                for (v, w) in words.iter_mut().zip(&self.words[..T::WORDS]) {
                    *v = w.load(Ordering::Relaxed);
                }
                fence(Ordering::Acquire);
                if self.seq.load(Ordering::Relaxed) == seq {
                    return T::unpack(&words[..T::WORDS]);
                }
            }
            tries += 1;
            if tries < SPINS {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    /// Every word the same: a value read whole has them all alike.
    #[derive(Clone, Copy, Default)]
    struct Alike([u64; MAX_WORDS]);

    impl Packed for Alike {
        const WORDS: usize = MAX_WORDS;
        fn pack(&self, words: &mut [u64]) {
            words.copy_from_slice(&self.0);
        }
        fn unpack(words: &[u64]) -> Self {
            let mut v = [0; MAX_WORDS];
            v.copy_from_slice(words);
            Alike(v)
        }
    }

    #[test]
    fn readers_never_see_half_a_write() {
        let lock = Arc::new(SeqLock::<Alike>::default());
        let done = Arc::new(AtomicBool::new(false));
        let reads = Arc::new(AtomicU64::new(0));
        let readers: Vec<_> = (0..3)
            .map(|_| {
                let (lock, done, reads) = (lock.clone(), done.clone(), reads.clone());
                std::thread::spawn(move || {
                    while !done.load(Ordering::Relaxed) {
                        let v = lock.load().0;
                        assert!(v.iter().all(|&w| w == v[0]), "{v:?}");
                        reads.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();
        // Writes all the while the readers read, many times over.
        let mut k = 0u64;
        while reads.load(Ordering::Relaxed) < 300_000 || k < 100_000 {
            k += 1;
            lock.store(&Alike([k; MAX_WORDS]));
        }
        done.store(true, Ordering::Relaxed);
        for r in readers {
            r.join().unwrap();
        }
        assert_eq!(lock.load().0, [k; MAX_WORDS]);
    }
}
