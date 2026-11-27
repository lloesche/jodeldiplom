//! A plan of dispatches ([`Plan`]): its buffers (slots, each as large as
//! the largest use made of it) and its dispatches, each written by a
//! helper that knows its kernel's numbers. No dispatch reads and writes
//! the same slot (wgpu binds a buffer one way a dispatch); a dispatch that
//! adds to what is there reads it through its output.

use super::Kernel;
use crate::engine::{self, params, spread, After, Matrix, Mul, MUL_NUMBERS};

/// The matrix kernels' own flags (`tail.wgsl`'s `MX_*`): A as a
/// convolution's patches, the products added times a gate per column.
pub(crate) const MX_CONV: usize = 1;
pub(crate) const MX_GATED: usize = 2;
/// How W's values are held (`MM_*` beside the engine's flags: FP8 none).
pub(crate) const MM_BF16: usize = 4096;
pub(crate) const MM_F32: usize = 8192;
/// A convolution's input group-normed, then SiLU (`GN_*`).
pub(crate) const GN_NORM: usize = 1;
pub(crate) const GN_SILU: usize = 2;

/// Where a matrix multiply's numbers after the engine's ten are: the FP8
/// scales' offset and how many a row group has, the gates', W's row
/// stride; a convolution's input rows, columns and channels, the output's
/// columns, the kernel's size, stride and padding, the doubling; the group
/// norm's statistics, weights, biases, channels a group and flags; the
/// kernel's own flags.
const MM_SCALES: usize = MUL_NUMBERS.len();
const MM_PER: usize = 11;
const MM_GATES: usize = 12;
const MM_LDW: usize = 13;
const CV_H: usize = 14;
const GN_AT: usize = 22;
const MX_FLAGS: usize = 27;

/// Those places by the names the kernels read them by (each a function of
/// the source).
pub(crate) const MM_NUMBERS: [(&str, usize); 18] = [
    ("mm_scales", MM_SCALES),
    ("mm_per", MM_PER),
    ("mm_gates", MM_GATES),
    ("mm_ldw", MM_LDW),
    ("cv_h", CV_H),
    ("cv_w", CV_H + 1),
    ("cv_c", CV_H + 2),
    ("cv_ow", CV_H + 3),
    ("cv_size", CV_H + 4),
    ("cv_stride", CV_H + 5),
    ("cv_pad", CV_H + 6),
    ("cv_up", CV_H + 7),
    ("gn_at", GN_AT),
    ("gn_weight", GN_AT + 1),
    ("gn_bias", GN_AT + 2),
    ("gn_group", GN_AT + 3),
    ("gn_flags", GN_AT + 4),
    ("mx_flags", MX_FLAGS),
];

/// The buffers a dispatch binds: a part's matrices, its f32 tensors, and
/// the plan's slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Buf {
    W(u8),
    Small,
    Slot(u8),
}

/// One dispatch: its buffers (src, the matrices, dst, the f32 tensors,
/// src2), its numbers, its workgroups.
pub(crate) type Op = engine::Op<Kernel, [Buf; 5], 32>;

/// Where a dispatch's buffers bind the one it writes (the others it
/// reads).
const DST: usize = 2;

/// Whether `bufs` bind `b` to read.
fn reads(bufs: &[Buf; 5], b: Buf) -> bool {
    bufs.iter().enumerate().any(|(k, &x)| k != DST && x == b)
}

/// A tensor's place: its slot and its first f32 there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct At {
    pub(crate) slot: u8,
    pub(crate) off: usize,
}

impl At {
    pub(crate) fn new(slot: u8, off: usize) -> At {
        At { slot, off }
    }

    /// `n` f32s further on.
    pub(crate) fn plus(self, n: usize) -> At {
        At {
            off: self.off + n,
            ..self
        }
    }

    pub(crate) fn buf(self) -> Buf {
        Buf::Slot(self.slot)
    }
}

/// How a matrix's values are held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Format {
    /// FP8 (e4m3), four a word, times a scale per 64 rows and 128 values
    /// of a row (in `small`).
    Fp8 {
        scales: usize,
        per_row_group: usize,
    },
    Bf16,
    F32,
}

/// A matrix in a part's buffers (or in a slot): `n` rows (its outputs) of
/// `k` values, `ldw` apart (a multiple of 32, zeros after `k`), from the
/// `at`-th vector of four words; its bias (in `small`), if any.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Mat {
    pub(crate) buf: Buf,
    pub(crate) at: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) ldw: usize,
    pub(crate) format: Format,
    pub(crate) bias: Option<usize>,
}

impl Mat {
    /// `rows` of `k` f32s in a slot (lda a multiple of 32) as a matrix.
    pub(crate) fn of_slot(at: At, n: usize, k: usize, ldw: usize) -> Mat {
        assert!(at.off.is_multiple_of(4) && ldw.is_multiple_of(32));
        Mat {
            buf: at.buf(),
            at: at.off / 4,
            n,
            k,
            ldw,
            format: Format::F32,
            bias: None,
        }
    }
}

/// The engine's flags for how a matrix's values are held (`tail.wgsl`'s
/// `MM_BF16`, `MM_F32`; FP8 none).
impl Matrix for Mat {
    fn at(self) -> usize {
        self.at
    }

    fn flag(self) -> usize {
        match self.format {
            Format::Fp8 { .. } => 0,
            Format::Bf16 => MM_BF16,
            Format::F32 => MM_F32,
        }
    }
}

/// What becomes of a matrix multiply's products (after the bias).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tail {
    Store,
    /// Added to what is there.
    Add,
    /// Added times a gate per column (from this place).
    Gated(At),
    /// Each tile's 128 columns as 64 gates and their ups: silu(gate) · up
    /// written at column tile/2 + j.
    SwiGlu,
}

/// A group norm taken (with SiLU where asked) on a convolution's input as
/// it is read: its statistics' place (mean and inverse spread a group,
/// from `gn_stats`), its weights and biases in `small`, its channels a
/// group.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Norm {
    pub(crate) stats: At,
    pub(crate) weight: usize,
    pub(crate) bias: usize,
    pub(crate) group: usize,
    pub(crate) silu: bool,
}

/// A convolution: its input (`h × w × cin`, channels side by side, a
/// multiple of 4), its kernel's size, stride, padding before and after
/// (zeros), and whether the input is doubled first (nearest).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Conv {
    pub(crate) input: At,
    pub(crate) h: usize,
    pub(crate) w: usize,
    pub(crate) cin: usize,
    pub(crate) size: usize,
    pub(crate) stride: usize,
    pub(crate) pad: usize,
    pub(crate) trail: usize,
    pub(crate) up: bool,
}

impl Conv {
    /// The output's rows and columns.
    pub(crate) fn out(&self) -> (usize, usize) {
        let (h, w) = if self.up {
            (2 * self.h, 2 * self.w)
        } else {
            (self.h, self.w)
        };
        let o = |x: usize| (x + self.pad + self.trail - self.size) / self.stride + 1;
        (o(h), o(w))
    }
}

/// A plan: its slots' sizes (in f32s), its dispatches, and whether the
/// tensor-core kernels are used (only where the GPU has them).
#[derive(Clone, Debug, Default)]
pub(crate) struct Plan {
    pub(crate) slots: Vec<usize>,
    pub(crate) ops: Vec<Op>,
    pub(crate) fast: bool,
    /// Slots taken over by another use, and from which dispatch.
    handovers: Vec<(u8, usize)>,
    /// The dispatches that read the slot they write (adding into it, or
    /// in place).
    in_place: Vec<usize>,
}

impl Plan {
    pub(crate) fn new(fast: bool) -> Plan {
        Plan {
            fast,
            ..Plan::default()
        }
    }

    /// A slot of its own of `len` f32s (grown where a later use needs more).
    pub(crate) fn slot(&mut self, len: usize) -> u8 {
        self.slots.push(len);
        u8::try_from(self.slots.len() - 1).expect("at most 256 slots")
    }

    /// Slot `slot` (handed out by [`Plan::slot`]) holds at least `len`
    /// f32s.
    pub(crate) fn need(&mut self, slot: u8, len: usize) {
        let held = &mut self.slots[usize::from(slot)];
        *held = (*held).max(len);
    }

    /// Slot `slot` taken over from here by another use (a part that takes
    /// another's buffers in turns, or a result written where its input
    /// was): what it held is not read again, so its first use from here
    /// writes it ([`Plan::check`]).
    pub(crate) fn hand_over(&mut self, slot: u8) {
        self.handovers.push((slot, self.ops.len()));
    }

    /// Where a slot taken over is read before the new use writes it: the
    /// slot and the dispatch, else `None`.
    pub(crate) fn check(&self) -> Option<(u8, usize)> {
        self.handovers.iter().find_map(|&(slot, from)| {
            let b = Buf::Slot(slot);
            let (i, op) = self.ops[from..]
                .iter()
                .enumerate()
                .find(|(_, op)| op.buffers.contains(&b))?;
            // It holds `b`: where it does not read it, it writes it.
            let read = reads(&op.buffers, b) || self.in_place.contains(&(from + i));
            read.then_some((slot, from + i))
        })
    }

    /// The bytes its slots take.
    pub(crate) fn bytes(&self) -> u64 {
        self.slots.iter().map(|&n| (n.max(1) * 4) as u64).sum()
    }

    fn push(&mut self, kernel: Kernel, bufs: [Buf; 5], values: &[usize], groups: [u32; 3]) {
        let out = bufs[DST];
        debug_assert!(
            !matches!(out, Buf::W(_) | Buf::Small),
            "{kernel:?} writes a model's buffer"
        );
        debug_assert!(
            !reads(&bufs, out),
            "{kernel:?} binds the slot it writes to read ({out:?})"
        );
        self.ops.push(Op {
            kernel,
            buffers: bufs,
            params: params(values),
            groups,
        });
    }

    /// The dispatch just pushed reads the slot it writes too.
    fn reads_its_output(&mut self) {
        self.in_place.push(self.ops.len() - 1);
    }

    /// A matrix multiply's numbers: the engine's [`Mul`]'s ten (`m` rows
    /// of A at `a`, lda apart, through `w` into `c`, ldc apart, `tail`'s
    /// bias and flags), then the FP8 scales, the gates, W's row stride and
    /// the kernel's own flags; with the second input bound (the gates).
    fn mm_numbers(
        &self,
        (a, lda): (At, usize),
        m: usize,
        w: &Mat,
        (c, ldc): (At, usize),
        tail: Tail,
    ) -> ([usize; 32], Buf) {
        let (scales, per) = match w.format {
            Format::Fp8 {
                scales,
                per_row_group,
            } => (scales, per_row_group),
            Format::Bf16 | Format::F32 => (0, 0),
        };
        let after = match tail {
            Tail::Store => After::Store,
            Tail::Add | Tail::Gated(_) => After::Add,
            Tail::SwiGlu => {
                assert_eq!(w.n % 128, 0, "SwiGLU's pairs in tiles of 128");
                After::SwiGlu
            }
        };
        let mul = Mul {
            from: (a.buf(), a.off, lda),
            to: (c.buf(), c.off, ldc),
            m,
            n: w.n,
            k: w.k,
            w: *w,
            bias: w.bias,
            after,
        };
        let mut numbers = [0; 32];
        numbers[..MM_SCALES].copy_from_slice(&mul.numbers());
        numbers[MM_SCALES] = scales;
        numbers[MM_PER] = per;
        numbers[MM_LDW] = w.ldw;
        let mut src2 = Buf::Small;
        if let Tail::Gated(g) = tail {
            numbers[MM_GATES] = g.off;
            numbers[MX_FLAGS] = MX_GATED;
            src2 = g.buf();
        }
        (numbers, src2)
    }

    fn mm_kernel(&self) -> Kernel {
        if self.fast {
            Kernel::Gemm16
        } else {
            Kernel::Gemm
        }
    }

    /// C (`m` rows at `c`, ldc apart) = A (at `a`, lda apart) · Wᵀ, then
    /// `tail`. A's rows are f32 (lda and their place multiples of 4; k a
    /// multiple of 4).
    pub(crate) fn gemm(&mut self, a: (At, usize), m: usize, w: &Mat, c: (At, usize), tail: Tail) {
        assert!(a.0.off.is_multiple_of(4) && a.1.is_multiple_of(4) && w.k.is_multiple_of(4));
        let (numbers, src2) = self.mm_numbers(a, m, w, c, tail);
        let kernel = self.mm_kernel();
        self.push(
            kernel,
            [a.0.buf(), w.buf, c.0.buf(), Buf::Small, src2],
            &numbers,
            [w.n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1],
        );
        if matches!(tail, Tail::Add | Tail::Gated(_)) {
            self.reads_its_output();
        }
    }

    /// The convolution `conv` through `w` (its rows the output channels,
    /// each `size² · cin` values as tap, then channel) into `out`
    /// (channels side by side), `norm` taken on its input first.
    pub(crate) fn conv(&mut self, conv: Conv, w: &Mat, out: At, tail: Tail, norm: Option<Norm>) {
        assert_eq!(conv.cin % 4, 0);
        assert_eq!(w.k, conv.size * conv.size * conv.cin);
        assert!(matches!(tail, Tail::Store | Tail::Add));
        let (oh, ow) = conv.out();
        let m = oh * ow;
        let (mut numbers, mut src2) = self.mm_numbers((conv.input, 0), m, w, (out, w.n), tail);
        numbers[MX_FLAGS] |= MX_CONV;
        numbers[CV_H..CV_H + 8].copy_from_slice(&[
            conv.h,
            conv.w,
            conv.cin,
            ow,
            conv.size,
            conv.stride,
            conv.pad,
            usize::from(conv.up),
        ]);
        if let Some(n) = norm {
            assert_eq!(n.group % 4, 0);
            numbers[GN_AT..GN_AT + 5].copy_from_slice(&[
                n.stats.off,
                n.weight,
                n.bias,
                n.group,
                GN_NORM | if n.silu { GN_SILU } else { 0 },
            ]);
            src2 = n.stats.buf();
        }
        let kernel = self.mm_kernel();
        self.push(
            kernel,
            [conv.input.buf(), w.buf, out.buf(), Buf::Small, src2],
            &numbers,
            [w.n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1],
        );
        if tail == Tail::Add {
            self.reads_its_output();
        }
    }

    /// RMSNorm of `rows` rows of `d` (from `x`, its stride; to `y`, its
    /// stride), the weights at `weight` in `small`.
    pub(crate) fn rmsnorm(
        &mut self,
        rows: usize,
        d: usize,
        (x, xs): (At, usize),
        (y, ys): (At, usize),
        weight: usize,
        eps: f32,
    ) {
        self.push(
            Kernel::RmsNorm,
            [x.buf(), Buf::Small, y.buf(), Buf::Small, Buf::Small],
            &[
                rows,
                d,
                x.off,
                xs,
                y.off,
                ys,
                weight,
                eps.to_bits() as usize,
            ],
            [rows as u32, 1, 1],
        );
    }

    /// LayerNorm (no weights) of `rows` rows of `d`, then (1 + scale) · x +
    /// shift (`shift`, `scale`: places in the same slot).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn layernorm_mod(
        &mut self,
        rows: usize,
        d: usize,
        (x, xs): (At, usize),
        (y, ys): (At, usize),
        shift: At,
        scale: At,
        eps: f32,
    ) {
        assert_eq!(shift.slot, scale.slot);
        self.push(
            Kernel::LayerNormMod,
            [x.buf(), Buf::Small, y.buf(), Buf::Small, shift.buf()],
            &[
                rows,
                d,
                x.off,
                xs,
                y.off,
                ys,
                shift.off,
                scale.off,
                eps.to_bits() as usize,
            ],
            [rows as u32, 1, 1],
        );
    }

    /// `tokens` tokens' `heads` heads of 128 (from `x`, its stride) into the
    /// heads' layout at `out` (`[head][total][128]`, from token `first`):
    /// RMSNorm where `norm` gives its weights, then the rotation `rope`
    /// (with its table: per token, 64 pairs' cosine and sine).
    pub(crate) fn qk_prep(
        &mut self,
        (tokens, heads): (usize, usize),
        (x, xs): (At, usize),
        (out, total, first): (At, usize, usize),
        norm: Option<usize>,
        rope: Option<(Rope, Buf, usize)>,
        eps: f32,
    ) {
        let (mode, table, tbuf) = match rope {
            None => (0, 0, Buf::Small),
            Some((Rope::Pairs, b, at)) => (1, at, b),
            Some((Rope::Halves, b, at)) => (2, at, b),
        };
        self.push(
            Kernel::QkPrep,
            [x.buf(), Buf::Small, out.buf(), Buf::Small, tbuf],
            &[
                tokens,
                heads,
                x.off,
                xs,
                out.off,
                total,
                first,
                norm.unwrap_or(0xffff_ffff),
                mode,
                table,
                eps.to_bits() as usize,
            ],
            [tokens as u32, heads as u32, 1],
        );
    }

    /// `tokens` tokens' queries, keys and values (`q_heads`, `kv_heads` and
    /// `kv_heads` heads of 128 side by side, from `x`, its stride) into the
    /// heads' layout at `q`, `k` and `v` (`[head][total][128]`, from token
    /// `first`): the queries and keys normed by their weights (`norms`, in
    /// `small`) and turned by `rope`, the values as they are.
    pub(crate) fn heads(
        &mut self,
        (tokens, q_heads, kv_heads): (usize, usize, usize),
        (x, xs): (At, usize),
        ((q, k, v), total, first): ((At, At, At), usize, usize),
        (q_norm, k_norm): (usize, usize),
        rope: Option<(Rope, Buf, usize)>,
        eps: f32,
    ) {
        let kx = x.plus(q_heads * 128);
        let vx = kx.plus(kv_heads * 128);
        let rows = |n| (tokens, n);
        self.qk_prep(
            rows(q_heads),
            (x, xs),
            (q, total, first),
            Some(q_norm),
            rope,
            eps,
        );
        self.qk_prep(
            rows(kv_heads),
            (kx, xs),
            (k, total, first),
            Some(k_norm),
            rope,
            eps,
        );
        self.qk_prep(rows(kv_heads), (vx, xs), (v, total, first), None, None, eps);
    }

    /// Attention of `heads` query heads over `kv_heads` (heads of 128,
    /// laid out `[head][token][128]` at `q`, `k`, `v`, all in one slot):
    /// `queries` queries, `keys` keys (those from `valid` on left out),
    /// causal where asked; the output token-major at `out` (stride `os`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention(
        &mut self,
        (queries, keys, valid): (usize, usize, usize),
        (heads, kv_heads): (usize, usize),
        (q, k, v): (At, At, At),
        (out, os): (At, usize),
        causal: bool,
    ) {
        assert!(q.slot == k.slot && k.slot == v.slot);
        let scale = 1.0f32 / 128f32.sqrt();
        // 128 queries a workgroup on tensor cores, else 64.
        let (kernel, per) = if self.fast {
            (Kernel::Attention16, 128)
        } else {
            (Kernel::Attention, 64)
        };
        self.push(
            kernel,
            [q.buf(), Buf::Small, out.buf(), Buf::Small, Buf::Small],
            &[
                queries,
                keys,
                heads,
                heads / kv_heads,
                q.off,
                k.off,
                v.off,
                out.off,
                os,
                scale.to_bits() as usize,
                usize::from(causal),
                valid,
            ],
            [queries.div_ceil(per) as u32, heads as u32, 1],
        );
    }

    /// x (`count` values at `x`) += dt · d (at `d`).
    pub(crate) fn euler(&mut self, count: usize, x: At, d: At, dt: f32) {
        self.push(
            Kernel::Euler,
            [d.buf(), Buf::Small, x.buf(), Buf::Small, Buf::Small],
            &[count, x.off, d.off, dt.to_bits() as usize],
            spread(count),
        );
        self.reads_its_output();
    }

    /// y = silu(x), `count` values.
    pub(crate) fn silu(&mut self, count: usize, x: At, y: At) {
        self.push(
            Kernel::Act,
            [x.buf(), Buf::Small, y.buf(), Buf::Small, Buf::Small],
            &[count, x.off, y.off],
            spread(count),
        );
    }

    /// `rows` rows of `width` (multiples of 4) copied from `x` (its stride)
    /// to `y` (its stride).
    pub(crate) fn copy(
        &mut self,
        rows: usize,
        width: usize,
        (x, xs): (At, usize),
        (y, ys): (At, usize),
    ) {
        assert!(
            width.is_multiple_of(4)
                && x.off % 4 == 0
                && y.off % 4 == 0
                && xs % 4 == 0
                && ys % 4 == 0
        );
        self.push(
            Kernel::Copy,
            [x.buf(), Buf::Small, y.buf(), Buf::Small, Buf::Small],
            &[rows, width, x.off, xs, y.off, ys],
            spread(rows * width / 4),
        );
    }

    /// The autoencoder's latents (`2h × 2w`, `channels` a pixel, the first
    /// 32 taken) at `x` as `h × w` tokens of 128 at `y`, normalized by the
    /// batch norm (its means and spreads in `small`).
    pub(crate) fn patchify(
        &mut self,
        (h, w, channels): (usize, usize, usize),
        x: At,
        y: At,
        (mean, spread_at): (usize, usize),
    ) {
        self.push(
            Kernel::Patchify,
            [x.buf(), Buf::Small, y.buf(), Buf::Small, Buf::Small],
            &[h, w, x.off, channels, y.off, mean, spread_at],
            spread(h * w * 128),
        );
    }

    /// The reverse: `h × w` tokens at `x` to latents `2h × 2w × 32` at `y`.
    pub(crate) fn unpatchify(
        &mut self,
        (h, w): (usize, usize),
        x: At,
        y: At,
        (mean, spread_at): (usize, usize),
    ) {
        self.push(
            Kernel::Unpatchify,
            [x.buf(), Buf::Small, y.buf(), Buf::Small, Buf::Small],
            &[h, w, x.off, y.off, mean, spread_at],
            spread(h * w * 128),
        );
    }

    /// The group norm's statistics of `pixels × channels` at `x` (32
    /// groups): partial sums at `part` (in another slot than `stats`'),
    /// then each group's mean and inverse spread at `stats`.
    pub(crate) fn gn_stats(
        &mut self,
        (pixels, channels): (usize, usize),
        x: At,
        part: At,
        stats: At,
        eps: f32,
    ) {
        assert!(channels % 128 == 0 || channels == 64 || channels == 32);
        let chunk = 1024;
        let chunks = pixels.div_ceil(chunk);
        self.need(part.slot, part.off + chunks * 96);
        self.push(
            Kernel::GnStats,
            [x.buf(), Buf::Small, part.buf(), Buf::Small, Buf::Small],
            &[pixels, channels, x.off, channels / 32, chunk, part.off],
            [chunks as u32, 1, 1],
        );
        self.push(
            Kernel::GnReduce,
            [part.buf(), Buf::Small, stats.buf(), Buf::Small, Buf::Small],
            &[chunks, part.off, stats.off, eps.to_bits() as usize],
            [1, 1, 1],
        );
    }

    /// A softmax over each of `rows` rows of `n` at `x` (its stride), in
    /// place, the values times `scale` first.
    pub(crate) fn softmax(&mut self, rows: usize, n: usize, (x, xs): (At, usize), scale: f32) {
        self.push(
            Kernel::Softmax,
            [Buf::Small, Buf::Small, x.buf(), Buf::Small, Buf::Small],
            &[rows, n, x.off, xs, scale.to_bits() as usize],
            [rows as u32, 1, 1],
        );
        self.reads_its_output();
    }

    /// y (its stride) = the transpose of `rows × cols` at `x` (its stride).
    pub(crate) fn transpose(
        &mut self,
        (rows, cols): (usize, usize),
        (x, xs): (At, usize),
        (y, ys): (At, usize),
    ) {
        self.push(
            Kernel::Transpose,
            [x.buf(), Buf::Small, y.buf(), Buf::Small, Buf::Small],
            &[rows, cols, x.off, xs, y.off, ys],
            [cols.div_ceil(16) as u32, rows.div_ceil(16) as u32, 1],
        );
    }
}

/// How a head's values are paired for the rotary positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rope {
    /// Neighbours (2i, 2i + 1): FLUX.2's.
    Pairs,
    /// Halves (i, i + 64): Qwen's.
    Halves,
}
