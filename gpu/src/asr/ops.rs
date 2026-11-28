//! The recogniser's dispatches as they are planned: a kernel, its
//! buffers, its numbers and its workgroups, and the helpers that plan the
//! matrix multiplies (the engine's [`Mul`]: the tiled `qa_matmul` for many
//! rows, `qa_rows` for a few, its rows normed or put together on the way
//! as [`Before`] says) and the kernels with a workgroup or an invocation a
//! row.

use super::pack::Matrix;
use super::{Buf, Kernel, VOCAB};
pub(super) use crate::engine::After;
use crate::engine::{self, params, spread, ARGMAX, PARAM_WORDS, SWIGLU};

/// Rounds of eight columns a workgroup of the head (`qa_rows`).
const HEAD_ROUNDS: usize = 8;
/// The head's workgroups a row, each with its largest value.
pub(super) const BLOCKS: usize = VOCAB.div_ceil(8 * HEAD_ROUNDS);
/// Workgroups enough to keep the GPU busy: `qa_rows` takes more columns a
/// workgroup while it keeps as many.
const ENOUGH_WORKGROUPS: usize = 128;
/// The most values of a row `qa_rows` holds in workgroup memory at once:
/// a matrix with more inputs takes one round of columns.
const STAGED: usize = 1024;

/// One dispatch: its buffers src, aux, dst, and the second dst of a kernel
/// that writes two.
pub(super) type Op = engine::Op<Kernel, [Buf; 4], PARAM_WORDS>;

/// A matrix multiply of the recogniser's: its 4-bit, 8-bit or half
/// matrices.
pub(super) type Mul = engine::Mul<Buf, Matrix>;

/// `qa_rows`'s flags for what [`Before`] does.
const NORM: usize = 32;
const JOIN: usize = 256;

/// What the rows are before the matrix (`qa_rows`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Before {
    /// As they are.
    Plain,
    /// Each RMS-normed by the weights at this offset.
    Norm(usize),
    /// `qa_dec_attend`'s chunks of each head (this many), put together.
    Join { chunks: usize },
}

impl engine::Matrix for Matrix {
    fn at(self) -> usize {
        self.at
    }

    fn flag(self) -> usize {
        self.format.flag()
    }
}

/// The most inputs `qa_matmul` takes in one part (with more, they are cut
/// in parts: [`matmul_cut`]).
const PART: usize = 1280;

/// `mul` as planned: through `qa_rows` when each piece has one row of it
/// (`per_piece`, as one row of one piece always goes, so a piece's sums do
/// not depend on the pieces with it), else the tiled `qa_matmul`.
pub(super) fn matmul(ops: &mut Vec<Op>, mul: &Mul, per_piece: usize) {
    tiled(ops, mul, per_piece, None);
}

/// [`matmul`], a long `k` (past [`PART`]) cut in parts that workgroups
/// take side by side (a matrix with few rows has few tiles, and a tile
/// through all of k keeps a workgroup long), their sums written to
/// `parts` (room for as many rows of `n` as there are parts, the
/// matrix's rows each) and added up in their order by `qa_split_sum`. The
/// parts follow from `k` alone, so a row's sums do not depend on the rows
/// with it.
pub(super) fn matmul_cut(ops: &mut Vec<Op>, mul: &Mul, per_piece: usize, parts: Buf) {
    tiled(ops, mul, per_piece, Some(parts));
}

fn tiled(ops: &mut Vec<Op>, mul: &Mul, per_piece: usize, parts: Option<Buf>) {
    if per_piece == 1 {
        rows(ops, mul, Before::Plain);
        return;
    }
    assert!(
        mul.flags() & (SWIGLU | ARGMAX) == 0,
        "qa_matmul only writes, adds or takes GELU"
    );
    let groups = mul.k / 32;
    let cut = parts.filter(|_| mul.k > PART).map(|buf| {
        let count = mul.k.div_ceil(PART);
        (buf, groups.div_ceil(count), count)
    });
    let mut numbers = mul.numbers().to_vec();
    let tiles = [mul.n.div_ceil(64) as u32, mul.m.div_ceil(128) as u32];
    let Some((buf, per, count)) = cut else {
        ops.push(Op {
            kernel: Kernel::Matmul,
            buffers: [mul.from.0, Buf::Small, mul.to.0, Buf::Spare],
            params: params(&numbers),
            groups: [tiles[0], tiles[1], 1],
            cost: crate::pace::Cost::UNESTIMATED,
        });
        return;
    };
    // The parts as many as the groups make (the last may be shorter).
    let count = groups.div_ceil(per).min(count);
    numbers.extend([per, count]);
    ops.push(Op {
        kernel: Kernel::Matmul,
        buffers: [mul.from.0, Buf::Small, mul.to.0, buf],
        params: params(&numbers),
        groups: [tiles[0], tiles[1], count as u32],
        cost: crate::pace::Cost::UNESTIMATED,
    });
    ops.push(Op {
        kernel: Kernel::SplitSum,
        buffers: [buf, Buf::Small, mul.to.0, Buf::Spare],
        params: params(&numbers),
        groups: spread(mul.m * mul.n / 4),
        cost: crate::pace::Cost::UNESTIMATED,
    });
}

/// `mul` through `qa_rows`, its rows as `before` says: four rows a
/// workgroup, its columns eight at a time, in as many rounds a workgroup as
/// still leave the GPU enough workgroups (the head's always
/// [`HEAD_ROUNDS`]: its workgroups are [`BLOCKS`]).
pub(super) fn rows(ops: &mut Vec<Op>, mul: &Mul, before: Before) {
    let (before_flag, norm, chunks) = match before {
        Before::Plain => (0, 0, 0),
        Before::Norm(at) => (NORM, at, 0),
        Before::Join { chunks } => (JOIN, 0, chunks),
    };
    let more = match mul.after {
        After::Argmax { stride } => stride,
        _ => chunks,
    };
    let flags = mul.flags() | before_flag;
    let (m, n, k) = (mul.m, mul.n, mul.k);
    // Eight columns a round, four outputs of gates and ups.
    let blocks = |rounds: usize| {
        if flags & SWIGLU != 0 {
            (n / 2).div_ceil(4 * rounds)
        } else {
            n.div_ceil(8 * rounds)
        }
    };
    let row_blocks = m.div_ceil(4);
    let rounds = if flags & ARGMAX != 0 {
        // More than a round only with the rows in workgroup memory at once.
        assert!(k <= STAGED, "the head takes at most {STAGED} inputs");
        HEAD_ROUNDS
    } else if k > STAGED {
        1
    } else {
        [8, 4, 2]
            .into_iter()
            .find(|&r| blocks(r) * row_blocks >= ENOUGH_WORKGROUPS)
            .unwrap_or(1)
    };
    let mut values = mul.numbers().to_vec();
    values[9] = flags | (rounds << 16);
    values.extend([norm, more]);
    let blocks = blocks(rounds);
    assert!(blocks <= 65_535, "{n} columns");
    let dst2 = if flags & ARGMAX != 0 {
        Buf::Best
    } else {
        Buf::Spare
    };
    ops.push(Op {
        kernel: Kernel::Rows,
        buffers: [mul.from.0, Buf::Small, mul.to.0, dst2],
        params: params(&values),
        groups: [row_blocks as u32, blocks as u32, 1],
        cost: crate::pace::Cost::UNESTIMATED,
    });
}

/// Whether `op` takes its matrix's input other than as it lies in memory
/// (normed or put together on the way), which calibrating cannot count.
pub(super) fn fused(op: &Op) -> bool {
    op.kernel == Kernel::Rows && op.params[9] & (NORM | JOIN) != 0
}

/// A kernel with a workgroup a row.
pub(super) fn per_row(
    ops: &mut Vec<Op>,
    kernel: Kernel,
    buffers: [Buf; 3],
    rows: usize,
    values: &[usize],
) {
    let x = rows.clamp(1, 65_535);
    ops.push(Op {
        kernel,
        buffers: [buffers[0], buffers[1], buffers[2], Buf::Spare],
        params: params(values),
        groups: [x as u32, rows.div_ceil(x).max(1) as u32, 1],
        cost: crate::pace::Cost::UNESTIMATED,
    });
}

/// An elementwise or per-row kernel over `count` invocations.
pub(super) fn each(
    ops: &mut Vec<Op>,
    kernel: Kernel,
    buffers: [Buf; 3],
    count: usize,
    values: &[usize],
) {
    ops.push(Op {
        kernel,
        buffers: [buffers[0], buffers[1], buffers[2], Buf::Spare],
        params: params(values),
        groups: spread(count),
        cost: crate::pace::Cost::UNESTIMATED,
    });
}
