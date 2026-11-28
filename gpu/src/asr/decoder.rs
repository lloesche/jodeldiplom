//! The decoder's dispatches for pieces heard together, a sequence each:
//! the prompts all at once ([`Inner::prompt_ops`]), then a token a step
//! for each ([`Inner::step_ops`]).
//!
//! A step is five dispatches a layer: the row normed through its queries,
//! key and value (`qa_rows`); those normed and turned, into the cache and
//! attending in chunks of 128 positions (`qa_dec_attend`, a workgroup for
//! each row, key/value head and chunk); the chunks put together through
//! the output matrix, added to the row; the row normed through gate and
//! up to their product; and that through the down matrix, added to the
//! row. Then the last norm, the head (each workgroup's largest logit with
//! it), and each row's next token with its embedding: 143 dispatches a
//! step, where the kernels one at a time took 340. Each sums as those did
//! (a matrix's columns lane by lane, then in a tree; the attention in
//! chunks, then put together), so a step's tokens are theirs, whatever the
//! rows with it. While calibrating, which counts the matrices' inputs, a
//! step goes through the prompt's kernels instead ([`Inner::unfused_ops`]).

use jodel_lyrics::asr::{DIM, FF, HEAD, HEADS, KV_HEADS, VOCAB};

use super::ops::{each, matmul, matmul_cut, per_row, rows, After, Before, Mul, Op, BLOCKS};
use super::{Buf, Inner, Kernel, Shape, KV, PARTS, QKV_DEC};
use crate::engine::params;

impl Inner {
    /// The decoder over the prompts of `shape`'s pieces (`shape.rows` rows
    /// each in `X`), their keys and values into the cache, then each one's
    /// first token ([`Inner::next_ops`]).
    pub(super) fn prompt_ops(&self, ops: &mut Vec<Op>, shape: &Shape) {
        let r = shape.rows;
        self.unfused_ops(ops, shape, 0, r);
        self.next_ops(ops, shape, (r - 1, r), r);
    }

    /// One step for each of `shape`'s pieces: the token at `pos` (its
    /// embedding in `X`, a row each) through the decoder, the next at
    /// `pos + 1`.
    pub(super) fn step_ops(&self, ops: &mut Vec<Op>, shape: &Shape, pos: usize) {
        if self.stats.on {
            self.unfused_ops(ops, shape, pos, 1);
        } else {
            self.fused_ops(ops, shape, pos);
        }
        self.next_ops(ops, shape, (0, 1), pos + 1);
    }

    /// The decoder's layers for a step, fused (see the module's doc).
    pub(super) fn fused_ops(&self, ops: &mut Vec<Op>, shape: &Shape, pos: usize) {
        let b = shape.pieces;
        let seq = shape.cache();
        let parts = b * seq;
        let chunks = (pos + 1).div_ceil(128);
        for (l, layer) in self.tensors.dec.iter().enumerate() {
            rows(
                ops,
                &Mul::dense(
                    Buf::X,
                    Buf::QkvD,
                    (b, QKV_DEC, DIM),
                    layer.qkv,
                    None,
                    After::Store,
                ),
                Before::Norm(layer.norm),
            );
            ops.push(Op {
                kernel: Kernel::DecAttend,
                buffers: [Buf::QkvD, Buf::Small, Buf::Cache, Buf::Spare],
                params: params(&[
                    pos,
                    l * shape.positions * KV,
                    seq,
                    shape.positions,
                    0,
                    0,
                    layer.q_norm,
                    layer.k_norm,
                    self.tensors.rope,
                    parts,
                ]),
                groups: [b as u32, KV_HEADS as u32, chunks as u32],
                cost: crate::pace::Cost::UNESTIMATED,
            });
            rows(
                ops,
                &Mul {
                    from: (Buf::Cache, parts, PARTS),
                    ..Mul::dense(
                        Buf::Cache,
                        Buf::X,
                        (b, DIM, HEADS * HEAD),
                        layer.o,
                        None,
                        After::Add,
                    )
                },
                Before::Join { chunks },
            );
            rows(
                ops,
                &Mul {
                    to: (Buf::Act, 0, FF),
                    ..Mul::dense(
                        Buf::X,
                        Buf::Act,
                        (b, 2 * FF, DIM),
                        layer.gate_up,
                        None,
                        After::SwiGlu,
                    )
                },
                Before::Norm(layer.post_norm),
            );
            rows(
                ops,
                &Mul::dense(Buf::Act, Buf::X, (b, DIM, FF), layer.down, None, After::Add),
                Before::Plain,
            );
        }
    }

    /// The decoder's layers through the kernels one at a time, for `n`
    /// rows a piece of `shape`'s at positions from `start` (the prompts:
    /// from 0; a step: one): each matrix's input written out (what
    /// calibrating counts), the keys and values into the cache.
    pub(super) fn unfused_ops(&self, ops: &mut Vec<Op>, shape: &Shape, start: usize, n: usize) {
        let b = shape.pieces;
        let rows = b * n;
        let seq = shape.cache();
        let rope = self.tensors.rope;
        for (l, layer) in self.tensors.dec.iter().enumerate() {
            per_row(
                ops,
                Kernel::RmsNorm,
                [Buf::X, Buf::Small, Buf::Xn],
                rows,
                &[rows, DIM, layer.norm, 0, DIM],
            );
            matmul(
                ops,
                &Mul::dense(
                    Buf::Xn,
                    Buf::QkvD,
                    (rows, QKV_DEC, DIM),
                    layer.qkv,
                    None,
                    After::Store,
                ),
                n,
            );
            each(
                ops,
                Kernel::HeadNormRope,
                [Buf::Small, Buf::Small, Buf::QkvD],
                rows * HEADS,
                &[rows, QKV_DEC, 0, HEADS, layer.q_norm, start, rope, n],
            );
            each(
                ops,
                Kernel::HeadNormRope,
                [Buf::Small, Buf::Small, Buf::QkvD],
                rows * KV_HEADS,
                &[
                    rows,
                    QKV_DEC,
                    HEADS * HEAD,
                    KV_HEADS,
                    layer.k_norm,
                    start,
                    rope,
                    n,
                ],
            );
            // The keys as qa_dec_attention keeps them (each of a key's
            // values the positions' side by side), then the values.
            let (cache, positions) = (l * shape.positions * KV, shape.positions);
            let width = KV / 2;
            each(
                ops,
                Kernel::Copy,
                [Buf::QkvD, Buf::Small, Buf::Cache],
                rows * width,
                &[
                    rows,
                    width,
                    HEADS * HEAD,
                    QKV_DEC,
                    cache + start,
                    1,
                    n,
                    seq,
                    positions,
                ],
            );
            each(
                ops,
                Kernel::Copy,
                [Buf::QkvD, Buf::Small, Buf::Cache],
                rows * width,
                &[
                    rows,
                    width,
                    HEADS * HEAD + width,
                    QKV_DEC,
                    cache + positions * width + start * width,
                    width,
                    n,
                    seq,
                ],
            );
            ops.push(Op {
                kernel: Kernel::DecAttention,
                buffers: [Buf::QkvD, Buf::Cache, Buf::AttnD, Buf::Spare],
                params: params(&[n, start, cache, seq, positions]),
                groups: [n.div_ceil(64) as u32, HEADS as u32, b as u32],
                cost: crate::pace::Cost::UNESTIMATED,
            });
            // Long matrices cut in parts (their sums in `Gu`, unused
            // here).
            matmul_cut(
                ops,
                &Mul::dense(
                    Buf::AttnD,
                    Buf::X,
                    (rows, DIM, HEADS * HEAD),
                    layer.o,
                    None,
                    After::Add,
                ),
                n,
                Buf::Gu,
            );
            per_row(
                ops,
                Kernel::RmsNorm,
                [Buf::X, Buf::Small, Buf::Xn],
                rows,
                &[rows, DIM, layer.post_norm, 0, DIM],
            );
            matmul(
                ops,
                &Mul::dense(
                    Buf::Xn,
                    Buf::Gu,
                    (rows, 2 * FF, DIM),
                    layer.gate_up,
                    None,
                    After::Store,
                ),
                n,
            );
            each(
                ops,
                Kernel::SwiGlu,
                [Buf::Gu, Buf::Small, Buf::Act],
                rows * FF,
                &[rows],
            );
            matmul_cut(
                ops,
                &Mul::dense(
                    Buf::Act,
                    Buf::X,
                    (rows, DIM, FF),
                    layer.down,
                    None,
                    After::Add,
                ),
                n,
                Buf::Gu,
            );
        }
    }

    /// Each piece's next token from its row `last` of `X` (`stride` rows
    /// a piece): the last norm, the head, and the token (to its place
    /// `place` in `Tokens`, its embedding to `X`'s first rows, a row a
    /// piece).
    pub(super) fn next_ops(
        &self,
        ops: &mut Vec<Op>,
        shape: &Shape,
        (last, stride): (usize, usize),
        place: usize,
    ) {
        let b = shape.pieces;
        per_row(
            ops,
            Kernel::RmsNorm,
            [Buf::X, Buf::Small, Buf::Last],
            b,
            &[b, DIM, self.tensors.norm, last * DIM, stride * DIM],
        );
        rows(
            ops,
            &Mul::dense(
                Buf::Last,
                Buf::Logits,
                (b, VOCAB, DIM),
                self.tensors.head,
                None,
                After::Argmax { stride: 2 * BLOCKS },
            ),
            Before::Plain,
        );
        ops.push(Op {
            kernel: Kernel::ArgmaxEmbed,
            buffers: [Buf::Best, Buf::Small, Buf::Tokens, Buf::X],
            params: params(&[
                BLOCKS,
                place,
                shape.positions,
                self.tensors.embed.at,
                self.tensors.embed.format.bits(),
            ]),
            groups: [b as u32, 1, 1],
            cost: crate::pace::Cost::UNESTIMATED,
        });
    }
}
