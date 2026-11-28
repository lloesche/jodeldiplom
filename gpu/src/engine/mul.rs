//! A matrix multiply as it is planned ([`Mul`]): the rows it reads, the
//! rows it writes, the weights, and what becomes of the products. Every
//! network's matrix kernels take the same first ten numbers
//! ([`Mul::numbers`]): the shared tiled one (`shared.wgsl`'s `matmul`, the
//! separator's, the beat network's and the speech model's: [`Mul::tiled`],
//! what it does with the products a [`Tiled`]) and the recogniser's
//! (`asr/ops.rs`, an [`After`]).

use super::{params, ModelOp};

/// The kernels' flags: the bias added, the products added to what is
/// there, GELU taken, the gated products, the head's largest.
pub(crate) const BIAS: usize = 1;
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
    /// Its flag in a matrix kernel's numbers (the kernels' sources take
    /// them from here: `engine::wgsl_consts`).
    pub(crate) fn flag(self) -> usize {
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

/// What the shared tiled matrix multiply does with the products (after
/// the bias): those of [`After`] it has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tiled {
    /// Written.
    Store,
    /// Added to what is there.
    Add,
    /// The columns are gates and ups in pairs: [`After::GeGlu`].
    GeGlu,
}

impl From<Tiled> for After {
    fn from(t: Tiled) -> After {
        match t {
            Tiled::Store => After::Store,
            Tiled::Add => After::Add,
            Tiled::GeGlu => After::GeGlu,
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

/// The names of the numbers every matrix kernel takes first
/// ([`Mul::numbers`]), as the kernels read them ([`super::wgsl_numbers`]).
pub(crate) const MUL_NUMBERS: [&str; 10] =
    ["m", "n", "k", "a", "lda", "w", "c", "ldc", "bias", "flags"];

/// A matrix multiply: `to`'s rows (`m × n`, from its offset, its stride
/// apart) (+)= `from`'s rows (`k` values from its offset, its stride
/// apart) times `w` (+ the bias at `bias`), as `after` says (`A`: what
/// its kernel does with the products). `B` names the buffers.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Mul<B, W = usize, A = After> {
    pub(crate) from: (B, usize, usize),
    pub(crate) to: (B, usize, usize),
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) w: W,
    pub(crate) bias: Option<usize>,
    pub(crate) after: A,
}

impl<B: Copy, W: Matrix, A: Copy + Into<After>> Mul<B, W, A> {
    /// `m` rows of `k` values from the start of `from` through `w` (`n`
    /// columns) to the start of `to`.
    pub(crate) fn dense(
        from: B,
        to: B,
        (m, n, k): (usize, usize, usize),
        w: W,
        bias: Option<usize>,
        after: A,
    ) -> Mul<B, W, A> {
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
        self.w.flag() | bias | self.after.into().flag()
    }

    /// The numbers every matrix kernel takes first ([`MUL_NUMBERS`]): `m`,
    /// `n`, `k`, `from`'s offset and stride, the weights' offset, `to`'s
    /// offset and stride, the bias's offset, the flags.
    pub(crate) fn numbers(&self) -> [usize; MUL_NUMBERS.len()] {
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
}

/// A matrix multiply through the shared tiled kernel ([`Mul::tiled`]),
/// its weights f32s.
pub(crate) type TiledMul<B> = Mul<B, usize, Tiled>;

impl<B: Copy> TiledMul<B> {
    /// The shared tiled matrix multiply (`matmul` of `shared.wgsl`, tiles
    /// of 128 rows × 64 columns) as `kernel`, the weights in `weights`.
    pub(crate) fn tiled<K>(&self, kernel: K, weights: B) -> ModelOp<K, B> {
        ModelOp {
            kernel,
            buffers: [self.from.0, weights, self.to.0],
            params: params(&self.numbers()),
            groups: [self.n.div_ceil(64) as u32, self.m.div_ceil(128) as u32, 1],
            cost: crate::pace::Cost::UNESTIMATED,
        }
    }
}
