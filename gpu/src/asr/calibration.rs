//! Calibration: while it is on, each matrix multiply's input columns'
//! sums of squares are added up (`qa_colsq` before it) and its rows
//! counted, for the 4-bit packing to weigh its columns by. Its runs go
//! through the kernels one at a time (the fused ones take their inputs
//! normed or put together on the way, where they cannot be counted) and
//! hear one piece at a time (each row counted is one heard).

use std::collections::HashMap;

use super::ops::{fused, Op};
use super::{Buf, GpuAsr, Kernel};
use crate::engine::{params, spread};

/// The statistics' bookkeeping.
pub(super) struct Stats {
    /// Whether they are collected now.
    pub(super) on: bool,
    /// Each matrix by where it is: its name, where its sums go, its
    /// columns.
    at: HashMap<usize, (String, usize, usize)>,
    /// The rows each matrix's input has had.
    rows_seen: HashMap<usize, u64>,
    /// The sums' f32s, all matrices'.
    pub(super) len: usize,
}

impl Stats {
    /// For the matrices `(name, where, columns)`.
    pub(super) fn new(matrices: &[(String, usize, usize)]) -> Stats {
        let mut len = 0;
        let at = matrices
            .iter()
            .map(|(name, w, cols)| {
                len += cols;
                (*w, (name.clone(), len - cols, *cols))
            })
            .collect();
        Stats {
            on: false,
            at,
            rows_seen: HashMap::new(),
            len,
        }
    }

    /// `ops` with each matrix multiply's input columns' sums of squares
    /// added to the statistics first, its rows counted.
    pub(super) fn counted(&mut self, ops: &[Op]) -> Vec<Op> {
        let mut out = Vec::with_capacity(ops.len() * 2);
        for op in ops {
            debug_assert!(!fused(op), "a fused multiply's input cannot be counted");
            if matches!(op.kernel, Kernel::Matmul | Kernel::Rows) {
                let [m, _, k, a_off, lda, w] = [0, 1, 2, 3, 4, 5].map(|i| op.params[i]);
                if let Some(&(_, s_off, _)) = self.at.get(&w) {
                    *self.rows_seen.entry(w).or_default() += m as u64;
                    out.push(Op {
                        kernel: Kernel::ColSq,
                        buffers: [op.buffers[0], Buf::Small, Buf::Stats, Buf::Spare],
                        params: params(&[m, k, a_off, lda, s_off]),
                        groups: spread(k),
                    });
                }
            }
            out.push(op.clone());
        }
        out
    }
}

impl GpuAsr {
    /// Starts (afresh) or stops collecting calibration statistics: each
    /// matrix's input columns' sums of squares over what is heard.
    pub fn calibrate(&self, on: bool) -> Result<(), String> {
        self.using(|inner| {
            if on {
                let zeros = vec![0.0f32; inner.stats.len];
                inner.program.queue.write_buffer(
                    &inner.buffers[&Buf::Stats],
                    0,
                    &crate::engine::bytes(&zeros),
                );
                inner.stats.rows_seen.clear();
            }
            inner.stats.on = on;
            Ok(())
        })
    }

    /// The statistics collected since [`GpuAsr::calibrate`]: each matrix's
    /// name, the rows its input had, and its columns' sums of squares.
    pub fn statistics(&self) -> Result<Vec<(String, u64, Vec<f32>)>, String> {
        self.using(|inner| {
            let len = inner.stats.len;
            let sums = inner.run(&[], Some((Buf::Stats, 0, len)), crate::pace::Phase::Steps)?;
            let stats = &inner.stats;
            let mut out: Vec<(String, u64, Vec<f32>)> = stats
                .at
                .iter()
                .map(|(w, (name, at, cols))| {
                    let rows = stats.rows_seen.get(w).copied().unwrap_or(0);
                    (name.clone(), rows, sums[*at..*at + *cols].to_vec())
                })
                .collect();
            out.sort_by(|a, b| a.0.cmp(&b.0));
            Ok(out)
        })
    }
}
