//! The recogniser's dispatches as they are planned: a kernel, its
//! buffers, its numbers and its workgroups, and the helpers that plan the
//! matrix multiplies ([`Mul`]: the tiled one for many rows, `qa_rows` for a
//! few) and the kernels with a workgroup or an invocation a row.

use super::pack::Matrix;
use super::{spread, Buf, Kernel, VOCAB};

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

/// One dispatch.
#[derive(Clone)]
pub(super) struct Op {
    pub(super) kernel: Kernel,
    /// src, aux, dst, and the second dst of a kernel that writes two.
    pub(super) buffers: [Buf; 4],
    pub(super) params: [u32; 12],
    pub(super) groups: [u32; 3],
}

pub(super) fn params(values: &[usize]) -> [u32; 12] {
    let mut p = [0u32; 12];
    for (slot, &v) in p.iter_mut().zip(values) {
        *slot = v as u32;
    }
    p
}

/// The kernels' flags (`qa_matmul`'s and `qa_rows`'s).
const BIAS: usize = 1;
const ADD: usize = 2;
const GELU: usize = 4;
const NORM: usize = 32;
const SWIGLU: usize = 64;
const ARGMAX: usize = 128;
const JOIN: usize = 256;

/// What the rows are before the matrix (`qa_rows` only but `Plain`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Before {
    /// As they are.
    Plain,
    /// Each RMS-normed by the weights at this offset.
    Norm(usize),
    /// `qa_dec_attend`'s chunks of each head (this many), put together.
    Join { chunks: usize },
}

/// What becomes of the products (after the bias).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum After {
    /// Written.
    Store,
    /// Added to what is there.
    Add,
    /// GELU, then written.
    Gelu,
    /// The columns are gates and ups (the first half and the rest): each
    /// pair written as silu(gate) · up (`qa_rows` only).
    SwiGlu,
    /// Written, and each row's largest of a workgroup's with its column to
    /// `Best`, `stride` apart a row (`qa_rows` only, the head).
    Argmax { stride: usize },
}

/// A matrix multiply: `to`'s rows (`m × n`, from its offset, its stride
/// apart) (+)= `from`'s rows (`k` values from its offset, its stride
/// apart) times `w` (+ the bias at `bias`), as `before` and `after` say.
pub(super) struct Mul {
    pub(super) from: (Buf, usize, usize),
    pub(super) to: (Buf, usize, usize),
    pub(super) m: usize,
    pub(super) n: usize,
    pub(super) k: usize,
    pub(super) w: Matrix,
    pub(super) bias: Option<usize>,
    pub(super) before: Before,
    pub(super) after: After,
}

impl Mul {
    /// `m` rows of `k` values from the start of `from` through `w` (`n`
    /// columns) to the start of `to`, the rows as they are.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dense(
        from: Buf,
        to: Buf,
        (m, n, k): (usize, usize, usize),
        w: Matrix,
        bias: Option<usize>,
        after: After,
    ) -> Mul {
        Mul {
            from: (from, 0, k),
            to: (to, 0, n),
            m,
            n,
            k,
            w,
            bias,
            before: Before::Plain,
            after,
        }
    }

    /// The flags and the values for `before` and `after` (the norm's
    /// weights, then the chunks or the stride).
    fn flags(&self) -> (usize, usize, usize) {
        let mut flags = self.w.format.flag();
        if self.bias.is_some() {
            flags |= BIAS;
        }
        let (norm, more) = match self.before {
            Before::Plain => (0, 0),
            Before::Norm(at) => {
                flags |= NORM;
                (at, 0)
            }
            Before::Join { chunks } => {
                flags |= JOIN;
                (0, chunks)
            }
        };
        let more = match self.after {
            After::Store => more,
            After::Add => {
                flags |= ADD;
                more
            }
            After::Gelu => {
                flags |= GELU;
                more
            }
            After::SwiGlu => {
                flags |= SWIGLU;
                more
            }
            After::Argmax { stride } => {
                flags |= ARGMAX;
                stride
            }
        };
        (flags, norm, more)
    }
}

/// `mul` as planned: through `qa_rows` when each piece has one row of it
/// (`per_piece`, as one row of one piece always goes, so a piece's sums do
/// not depend on the pieces with it), else the tiled `qa_matmul`.
pub(super) fn matmul(ops: &mut Vec<Op>, mul: &Mul, per_piece: usize) {
    if per_piece == 1 {
        rows(ops, mul);
        return;
    }
    assert_eq!(
        mul.before,
        Before::Plain,
        "qa_matmul takes the rows as they are"
    );
    let (flags, _, _) = mul.flags();
    assert!(
        flags & (SWIGLU | ARGMAX) == 0,
        "qa_matmul only writes, adds or takes GELU"
    );
    let (src, a_off, lda) = mul.from;
    let (dst, c_off, ldc) = mul.to;
    let values = [
        mul.m,
        mul.n,
        mul.k,
        a_off,
        lda,
        mul.w.at,
        c_off,
        ldc,
        mul.bias.unwrap_or(0),
        flags,
    ];
    ops.push(Op {
        kernel: Kernel::Matmul,
        buffers: [src, Buf::Small, dst, Buf::Spare],
        params: params(&values),
        groups: [mul.n.div_ceil(64) as u32, mul.m.div_ceil(128) as u32, 1],
    });
}

/// `mul` through `qa_rows`: four rows a workgroup, its columns eight at a
/// time, in as many rounds a workgroup as still leave the GPU enough
/// workgroups (the head's always [`HEAD_ROUNDS`]: its workgroups are
/// [`BLOCKS`]).
pub(super) fn rows(ops: &mut Vec<Op>, mul: &Mul) {
    let (flags, norm, more) = mul.flags();
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
    let (src, a_off, lda) = mul.from;
    let (dst, c_off, ldc) = mul.to;
    let values = [
        m,
        n,
        k,
        a_off,
        lda,
        mul.w.at,
        c_off,
        ldc,
        mul.bias.unwrap_or(0),
        flags | (rounds << 16),
        norm,
        more,
    ];
    let blocks = blocks(rounds);
    assert!(blocks <= 65_535, "{n} columns");
    let dst2 = if flags & ARGMAX != 0 {
        Buf::Best
    } else {
        Buf::Spare
    };
    ops.push(Op {
        kernel: Kernel::Rows,
        buffers: [src, Buf::Small, dst, dst2],
        params: params(&values),
        groups: [row_blocks as u32, blocks as u32, 1],
    });
}

/// Whether `op` takes its matrix's input other than as it lies in memory
/// (normed or put together on the way), which calibrating cannot count.
pub(super) fn fused(op: &Op) -> bool {
    op.kernel == Kernel::Rows && op.params[9] as usize & (NORM | JOIN) != 0
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
    });
}
