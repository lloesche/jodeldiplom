//! A matrix multiply as it is planned ([`Mul`]): the rows it reads, the
//! rows it writes, the weights, and what becomes of the products. Every
//! network's matrix kernels take the same first ten numbers
//! ([`Mul::numbers`]): the separator's tiled one (`kernels.wgsl`, which
//! the beat network and the speech model use too, [`Mul::tiled`]) and the
//! recogniser's (`asr/ops.rs`).

use super::{params, Op};

/// The kernels' flags: the bias added, the products added to what is
/// there, GELU taken, the gated products, the head's largest.
const BIAS: usize = 1;
const ADD: usize = 2;
const GELU: usize = 4;
pub(crate) const SWIGLU: usize = 64;
pub(crate) const ARGMAX: usize = 128;
const GEGLU: usize = 512;

/// What becomes of the products (after the bias). The tiled kernel writes
/// or adds them, or takes the GELU-gated products; the recogniser's also
/// take GELU, and its `qa_rows` the SiLU-gated products and the head's
/// largest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum After {
    /// Written.
    Store,
    /// Added to what is there.
    Add,
    /// GELU, then written.
    Gelu,
    /// The columns are gates and ups (the first half and the rest): each
    /// pair written as silu(gate) · up.
    SwiGlu,
    /// The columns are gates and ups in pairs (gate `j` at `2j`, its up
    /// next to it): each pair written as gelu(gate) · up at column `j`.
    GeGlu,
    /// Written, and each row's largest of a workgroup's with its column,
    /// `stride` apart a row.
    Argmax { stride: usize },
}

impl After {
    fn flag(self) -> usize {
        match self {
            After::Store => 0,
            After::Add => ADD,
            After::Gelu => GELU,
            After::SwiGlu => SWIGLU,
            After::GeGlu => GEGLU,
            After::Argmax { .. } => ARGMAX,
        }
    }
}

/// Where a matrix's weights are: their offset in the weights buffer, and
/// the kernels' flag for how they are held (0: f32s, `[in][out]`).
pub(crate) trait Matrix: Copy {
    fn at(self) -> usize;
    fn flag(self) -> usize;
}

/// f32 weights at this offset.
impl Matrix for usize {
    fn at(self) -> usize {
        self
    }

    fn flag(self) -> usize {
        0
    }
}

/// A matrix multiply: `to`'s rows (`m × n`, from its offset, its stride
/// apart) (+)= `from`'s rows (`k` values from its offset, its stride
/// apart) times `w` (+ the bias at `bias`), as `after` says. `B` names
/// the buffers.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Mul<B, W = usize> {
    pub(crate) from: (B, usize, usize),
    pub(crate) to: (B, usize, usize),
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) w: W,
    pub(crate) bias: Option<usize>,
    pub(crate) after: After,
}

impl<B: Copy, W: Matrix> Mul<B, W> {
    /// `m` rows of `k` values from the start of `from` through `w` (`n`
    /// columns) to the start of `to`.
    pub(crate) fn dense(
        from: B,
        to: B,
        (m, n, k): (usize, usize, usize),
        w: W,
        bias: Option<usize>,
        after: After,
    ) -> Mul<B, W> {
        Mul {
            from: (from, 0, k),
            to: (to, 0, n),
            m,
            n,
            k,
            w,
            bias,
            after,
        }
    }

    /// Its flags: the bias, what comes after, how the weights are held.
    pub(crate) fn flags(&self) -> usize {
        let bias = if self.bias.is_some() { BIAS } else { 0 };
        self.w.flag() | bias | self.after.flag()
    }

    /// The numbers every matrix kernel takes first: `m`, `n`, `k`, `from`'s
    /// offset and stride, the weights' offset, `to`'s offset and stride,
    /// the bias's offset, the flags.
    pub(crate) fn numbers(&self) -> [usize; 10] {
        let (_, a_off, lda) = self.from;
        let (_, c_off, ldc) = self.to;
        [
            self.m,
            self.n,
            self.k,
            a_off,
            lda,
            self.w.at(),
            c_off,
            ldc,
            self.bias.unwrap_or(0),
            self.flags(),
        ]
    }

    /// The tiled matrix multiply (`matmul` of `kernels.wgsl`, tiles of 128
    /// rows × 64 columns) as `kernel`, the weights in `weights`; it writes,
    /// adds or takes the GELU-gated products.
    pub(crate) fn tiled<K>(&self, kernel: K, weights: B) -> Op<K, [B; 3], 12> {
        assert!(
            matches!(self.after, After::Store | After::Add | After::GeGlu),
            "the tiled matrix multiply writes, adds or gates"
        );
        Op {
            kernel,
            buffers: [self.from.0, weights, self.to.0],
            params: params(&self.numbers()),
            groups: [self.n.div_ceil(64) as u32, self.m.div_ceil(128) as u32, 1],
        }
    }
}
