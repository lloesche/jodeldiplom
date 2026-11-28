//! The encoder's dispatches for pieces of the same length: their seconds
//! through the convolutions a few at a time, then the layers over all
//! their steps, each piece's attention within its own windows.

use jodel_lyrics::asr::{
    mel, piece_steps, CONV, DIM, ENC_DIM, ENC_FF, ENC_HEADS, PIECE, WINDOW_PIECES,
};

use super::ops::{each, matmul, matmul_cut, per_row, After, Mul, Op};
use super::{Buf, Inner, Kernel, Shape, BANDS, CONV_BATCH, PATCH, QKV_ENC};
use crate::engine::params;

impl Inner {
    /// The features of `shape`'s pieces (each `[MELS][frames]`) as pieces
    /// of a second, each piece's last padded with zeros (as all are, to the
    /// longest: a whole second unless the pieces are shorter).
    pub(super) fn upload(&self, features: &[&[f32]], shape: &Shape) {
        let (frames, width, images) = (shape.frames, shape.width, shape.images);
        let mut x = vec![0.0f32; features.len() * images * mel::MELS * width];
        for (k, f) in features.iter().enumerate() {
            for p in 0..images {
                let start = p * PIECE;
                let len = (frames - start).min(width);
                for b in 0..mel::MELS {
                    let row = &mut x[((k * images + p) * mel::MELS + b) * width..][..len];
                    row.copy_from_slice(&f[b * frames + start..][..len]);
                }
            }
        }
        self.program
            .queue
            .write_buffer(&self.buffers[&Buf::Pieces], 0, &crate::engine::bytes(&x));
    }

    /// The encoder's dispatches for `shape`'s pieces (their features
    /// already in `Pieces`): each piece's steps in `Audio`, one after the
    /// other.
    pub(super) fn encoder_ops(&self, shape: &Shape) -> Vec<Op> {
        let t = &self.tensors;
        let mut ops = Vec::new();
        let (frames, width, b) = (shape.frames, shape.width, shape.pieces);
        let images = b * shape.images;
        let (s1, s2, s3) = (
            width.div_ceil(2),
            width.div_ceil(2).div_ceil(2),
            piece_steps(width),
        );
        let steps = shape.steps;
        let rows = b * steps;
        let keep = piece_steps(frames - (shape.images - 1) * PIECE);
        for first in (0..images).step_by(CONV_BATCH) {
            let n = (images - first).min(CONV_BATCH);
            each(
                &mut ops,
                Kernel::Conv1,
                [Buf::Pieces, Buf::Small, Buf::Conv1],
                n * 64 * s1 * CONV,
                &[n, width, s1, t.conv1.0, t.conv1.1, first],
            );
            each(
                &mut ops,
                Kernel::Patches,
                [Buf::Conv1, Buf::Small, Buf::Patches],
                n * 32 * s2 * PATCH,
                &[n, 64, s1, 32, s2],
            );
            let (w, bias) = t.conv2;
            matmul(
                &mut ops,
                &Mul::dense(
                    Buf::Patches,
                    Buf::Conv2,
                    (n * 32 * s2, CONV, PATCH),
                    w,
                    Some(bias),
                    After::Gelu,
                ),
                shape.images * 32 * s2,
            );
            each(
                &mut ops,
                Kernel::Patches,
                [Buf::Conv2, Buf::Small, Buf::Patches],
                n * BANDS * s3 * PATCH,
                &[n, 32, s2, BANDS, s3],
            );
            let (w, bias) = t.conv3;
            matmul(
                &mut ops,
                &Mul::dense(
                    Buf::Patches,
                    Buf::Conv3,
                    (n * BANDS * s3, CONV, PATCH),
                    w,
                    Some(bias),
                    After::Gelu,
                ),
                shape.images * BANDS * s3,
            );
            each(
                &mut ops,
                Kernel::Flatten,
                [Buf::Conv3, Buf::Small, Buf::Flat],
                n * s3 * BANDS * CONV,
                &[n, s3],
            );
            // Cut in parts (its sums in `Conv1`, read before).
            matmul_cut(
                &mut ops,
                &Mul::dense(
                    Buf::Flat,
                    Buf::ConvOut,
                    (n * s3, ENC_DIM, BANDS * CONV),
                    t.conv_out,
                    None,
                    After::Store,
                ),
                steps,
                Buf::Conv1,
            );
            each(
                &mut ops,
                Kernel::Place,
                [Buf::ConvOut, Buf::Small, Buf::Hidden],
                n * s3 * ENC_DIM,
                &[n, s3, keep, first, t.positions, shape.images, steps],
            );
        }
        let window = s3 * WINDOW_PIECES;
        for layer in &t.enc {
            per_row(
                &mut ops,
                Kernel::LayerNorm,
                [Buf::Hidden, Buf::Small, Buf::Normed],
                rows,
                &[rows, ENC_DIM, layer.ln.0, layer.ln.1],
            );
            let (w, bias) = layer.qkv;
            matmul(
                &mut ops,
                &Mul::dense(
                    Buf::Normed,
                    Buf::Qkv,
                    (rows, QKV_ENC, ENC_DIM),
                    w,
                    Some(bias),
                    After::Store,
                ),
                steps,
            );
            ops.push(Op {
                kernel: Kernel::EncAttention,
                buffers: [Buf::Qkv, Buf::Small, Buf::Attn, Buf::Spare],
                params: params(&[steps, window]),
                groups: [steps.div_ceil(64) as u32, ENC_HEADS as u32, b as u32],
                cost: crate::pace::Cost::UNESTIMATED,
            });
            let (w, bias) = layer.o;
            matmul(
                &mut ops,
                &Mul::dense(
                    Buf::Attn,
                    Buf::Hidden,
                    (rows, ENC_DIM, ENC_DIM),
                    w,
                    Some(bias),
                    After::Add,
                ),
                steps,
            );
            per_row(
                &mut ops,
                Kernel::LayerNorm,
                [Buf::Hidden, Buf::Small, Buf::Normed],
                rows,
                &[rows, ENC_DIM, layer.fln.0, layer.fln.1],
            );
            let (w, bias) = layer.up;
            matmul(
                &mut ops,
                &Mul::dense(
                    Buf::Normed,
                    Buf::Up,
                    (rows, ENC_FF, ENC_DIM),
                    w,
                    Some(bias),
                    After::Gelu,
                ),
                steps,
            );
            // Cut in parts (their sums in `Qkv`, read before).
            let (w, bias) = layer.down;
            matmul_cut(
                &mut ops,
                &Mul::dense(
                    Buf::Up,
                    Buf::Hidden,
                    (rows, ENC_DIM, ENC_FF),
                    w,
                    Some(bias),
                    After::Add,
                ),
                steps,
                Buf::Qkv,
            );
        }
        per_row(
            &mut ops,
            Kernel::LayerNorm,
            [Buf::Hidden, Buf::Small, Buf::Normed],
            rows,
            &[rows, ENC_DIM, t.post.0, t.post.1],
        );
        let (w, bias) = t.proj1;
        matmul(
            &mut ops,
            &Mul::dense(
                Buf::Normed,
                Buf::Attn,
                (rows, ENC_DIM, ENC_DIM),
                w,
                Some(bias),
                After::Gelu,
            ),
            steps,
        );
        let (w, bias) = t.proj2;
        matmul(
            &mut ops,
            &Mul::dense(
                Buf::Attn,
                Buf::Audio,
                (rows, DIM, ENC_DIM),
                w,
                Some(bias),
                After::Store,
            ),
            steps,
        );
        ops
    }
}
