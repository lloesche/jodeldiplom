//! What the CPU networks' kernels share: work shared out over threads
//! ([`par_chunks`], [`par_rows`]), how many threads to take
//! ([`threads`], [`processors`]), a build of a kernel for AVX2 and FMA
//! picked at run time ([`fast!`](crate::fast)), a weight transposed
//! ([`transposed`]) and the logistic function ([`sigmoid`]).

use std::sync::Mutex;

/// How many threads a network that runs beside playback uses: half the
/// processors, so playback and the interface keep theirs.
pub fn threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| (n.get() / 2).max(1))
        .unwrap_or(1)
}

/// The processors, 4 where the system does not say: what a job that has
/// the machine to itself takes.
pub fn processors() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

/// Runs `work` on every `chunk`-sized piece of `data` (the last may be
/// shorter) on up to `threads` threads; `work` gets the piece's index.
/// The pieces are handed out in turn, so a slow one holds up only its
/// thread.
pub fn par_chunks<T: Send>(
    data: &mut [T],
    chunk: usize,
    threads: usize,
    work: impl Fn(usize, &mut [T]) + Sync,
) {
    let pieces = data.len().div_ceil(chunk.max(1));
    let threads = threads.min(pieces).max(1);
    if threads == 1 {
        for (i, piece) in data.chunks_mut(chunk.max(1)).enumerate() {
            work(i, piece);
        }
        return;
    }
    let queue = Mutex::new(data.chunks_mut(chunk.max(1)).enumerate());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let next = queue.lock().unwrap().next();
                match next {
                    Some((i, piece)) => work(i, piece),
                    None => break,
                }
            });
        }
    });
}

/// Runs `work` on each `width`-sized row of `data` on up to `threads`
/// threads, each thread a run of rows of its own; `work` gets the row's
/// index.
pub fn par_rows<T: Send>(
    data: &mut [T],
    width: usize,
    threads: usize,
    work: impl Fn(usize, &mut [T]) + Sync,
) {
    let width = width.max(1);
    let rows = data.len() / width;
    let per = rows.div_ceil(threads.min(rows).max(1)).max(1);
    par_chunks(&mut data[..rows * width], per * width, threads, |c, run| {
        for (r, row) in run.chunks_exact_mut(width).enumerate() {
            work(c * per + r, row);
        }
    });
}

/// `m` (`rows × columns`) transposed: `columns × rows`. A linear layer's
/// weight as a checkpoint keeps it (`[outputs][inputs]`) becomes
/// `[inputs][outputs]`, as the kernels read it.
pub fn transposed(m: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    assert_eq!(m.len(), rows * columns);
    let mut out = vec![0.0f32; m.len()];
    for r in 0..rows {
        for c in 0..columns {
            out[c * rows + r] = m[r * columns + c];
        }
    }
    out
}

/// The logistic function.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Defines `$name` to run its body compiled for AVX2 and FMA when the
/// processor has them, else as built. Rust never fuses a multiply and an
/// add on its own, so the sums are the same either way; a kernel that
/// asks for `mul_add` gets the one rounding it asks for in both.
#[macro_export]
macro_rules! fast {
    ($(#[$doc:meta])* $vis:vis fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $body:block) => {
        $(#[$doc])*
        $vis fn $name($($arg: $ty),*) {
            #[inline(always)]
            #[allow(clippy::too_many_arguments)]
            fn imp($($arg: $ty),*) $body
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx2,fma")]
                #[allow(clippy::too_many_arguments)]
                unsafe fn avx2($($arg: $ty),*) {
                    imp($($arg),*)
                }
                if std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    // SAFETY: the processor has AVX2 and FMA.
                    return unsafe { avx2($($arg),*) };
                }
            }
            imp($($arg),*)
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_row_is_worked_once_with_its_index() {
        for (rows, threads) in [(0usize, 3usize), (1, 4), (7, 3), (64, 8), (5, 1)] {
            let mut data = vec![0usize; rows * 3 + 2];
            par_rows(&mut data, 3, threads, |r, row| {
                for v in row.iter_mut() {
                    *v += r + 1;
                }
            });
            for (i, v) in data.iter().enumerate() {
                let want = if i < rows * 3 { i / 3 + 1 } else { 0 };
                assert_eq!(*v, want, "{rows} rows on {threads}");
            }
        }
    }

    #[test]
    fn every_chunk_is_worked_once_with_its_index() {
        let mut data = vec![0usize; 103];
        par_chunks(&mut data, 10, 4, |c, piece| {
            for v in piece.iter_mut() {
                *v += c + 1;
            }
        });
        for (i, v) in data.iter().enumerate() {
            assert_eq!(*v, i / 10 + 1);
        }
    }

    #[test]
    fn a_transpose_swaps_rows_and_columns() {
        let m = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(transposed(&m, 2, 3), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    crate::fast! {
        /// A sum made by the macro.
        fn summed(x: &[f32], out: &mut f32) {
            *out = x.iter().sum();
        }
    }

    #[test]
    fn a_fast_kernel_runs() {
        let mut out = 0.0;
        summed(&[1.0, 2.0, 3.5], &mut out);
        assert_eq!(out, 6.5);
    }
}
