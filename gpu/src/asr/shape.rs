//! What the pieces of one run make: their seconds through the
//! convolutions, the encoder's steps, the prompt's rows, the positions
//! each one's cache holds, and the room each buffer needs for them.

use std::collections::HashMap;

use jodel_lyrics::asr::{
    audio_steps, mel, piece_steps, text, CONV, DIM, ENC_DIM, ENC_FF, FF, HEAD, HEADS, LAYERS,
    PIECE, VOCAB,
};
use jodel_lyrics::VOCALS_RATE;

use super::ops::BLOCKS;
use super::{
    Buf, BANDS, CONV_BATCH, KV, MAX_NEW, MAX_SECONDS, NEW_AT_ONCE, PARTS, PATCH, QKV_DEC, QKV_ENC,
    STEPS_A_READ,
};

/// The pieces of one run: how many, each `frames` frames long, and what
/// that makes of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Shape {
    pub(super) pieces: usize,
    pub(super) frames: usize,
    /// Each piece's seconds through the convolutions, and their frames.
    pub(super) images: usize,
    pub(super) width: usize,
    /// The encoder's steps a piece.
    pub(super) steps: usize,
    /// The prompt's rows a piece.
    pub(super) rows: usize,
    /// The positions each piece's cache holds: its prompt, room for
    /// [`MAX_NEW`] tokens alone ([`NEW_AT_ONCE`] together), and a read of
    /// steps more.
    pub(super) positions: usize,
}

impl Shape {
    pub(super) fn new(pieces: usize, frames: usize) -> Shape {
        Shape::with_room(pieces, frames, NEW_AT_ONCE)
    }

    /// [`Shape::new`] with room for `together` tokens a piece heard with
    /// others.
    pub(super) fn with_room(pieces: usize, frames: usize, together: usize) -> Shape {
        let steps = audio_steps(frames);
        let rows = text::prompt(steps).len();
        let new = if pieces == 1 { MAX_NEW } else { together };
        Shape {
            pieces,
            frames,
            images: frames.div_ceil(PIECE),
            width: frames.min(PIECE),
            steps,
            rows,
            positions: rows + new + read(pieces),
        }
    }

    /// f32s (u32s for the weights) each buffer needs for these pieces.
    pub(super) fn needs(&self, buf: Buf) -> usize {
        let (b, s, r) = (self.pieces, self.steps, self.rows);
        let (s1, s2, s3) = (
            self.width.div_ceil(2),
            self.width.div_ceil(2).div_ceil(2),
            piece_steps(self.width),
        );
        match buf {
            Buf::Pieces => b * self.images * mel::MELS * self.width,
            Buf::Conv1 => CONV_BATCH * 64 * s1 * CONV,
            Buf::Patches => CONV_BATCH * 32 * s2 * PATCH,
            Buf::Conv2 => CONV_BATCH * 32 * s2 * CONV,
            Buf::Conv3 => CONV_BATCH * BANDS * s3 * CONV,
            Buf::Flat => CONV_BATCH * s3 * BANDS * CONV,
            Buf::ConvOut => CONV_BATCH * s3 * ENC_DIM,
            Buf::Hidden | Buf::Normed | Buf::Attn => b * s * ENC_DIM,
            Buf::Qkv => b * s * QKV_ENC,
            Buf::Up => b * s * ENC_FF,
            Buf::Audio => b * s * DIM,
            Buf::X | Buf::Xn => b * r * DIM,
            Buf::QkvD => b * r * QKV_DEC,
            Buf::AttnD => b * r * HEADS * HEAD,
            Buf::Gu => b * r * 2 * FF,
            Buf::Act => b * r * FF,
            Buf::Cache => b * (LAYERS * self.positions * KV + PARTS),
            Buf::Last => b * DIM,
            Buf::Logits => b * VOCAB,
            Buf::Best => b * BLOCKS * 2,
            Buf::Tokens => b * self.positions,
            Buf::Weights | Buf::Small | Buf::Stats | Buf::Spare => 0,
        }
    }

    /// A piece's keys and values, in f32s.
    pub(super) fn cache(&self) -> usize {
        LAYERS * self.positions * KV
    }
}

/// The steps decoded between reads for `pieces` heard together: half as
/// many for more than one (the steps past the last one's end are each a
/// step for all).
pub(super) fn read(pieces: usize) -> usize {
    if pieces == 1 {
        STEPS_A_READ
    } else {
        STEPS_A_READ / 2
    }
}

/// The buffers whose sizes follow the pieces heard.
pub(super) const SIZED: [Buf; 24] = [
    Buf::Pieces,
    Buf::Conv1,
    Buf::Patches,
    Buf::Conv2,
    Buf::Conv3,
    Buf::Flat,
    Buf::ConvOut,
    Buf::Hidden,
    Buf::Normed,
    Buf::Qkv,
    Buf::Attn,
    Buf::Up,
    Buf::Audio,
    Buf::X,
    Buf::Xn,
    Buf::QkvD,
    Buf::AttnD,
    Buf::Gu,
    Buf::Act,
    Buf::Cache,
    Buf::Last,
    Buf::Logits,
    Buf::Best,
    Buf::Tokens,
];

/// The frames of a part of [`MAX_SECONDS`], the longest heard.
pub(super) fn longest() -> usize {
    mel::frames_of(MAX_SECONDS * VOCALS_RATE as usize)
}

/// The frames of a glimpse ([`jodel_lyrics::glimpses::GLIMPSE`]), the
/// pieces the buffers hold [`super::MOST_AT_ONCE`] of.
pub(super) fn glimpse() -> usize {
    mel::frames_of((jodel_lyrics::glimpses::GLIMPSE * f64::from(VOCALS_RATE)) as usize)
}

/// The room the buffers have: each sized one's f32s (for a part of
/// [`MAX_SECONDS`] alone and the most glimpses together), and how many
/// glimpses that is.
pub(super) struct Room {
    of: HashMap<Buf, usize>,
    /// The f32s read back at most.
    pub(super) staging: usize,
    /// The most pieces of a glimpse's length heard together.
    pub(super) most: usize,
}

impl Room {
    /// The room for as many glimpses together (at most `most`) as keep
    /// their keys and values in `cache_bytes`, reading back at most
    /// `read` f32s besides.
    pub(super) fn new(most: usize, cache_bytes: u64, read: usize) -> Room {
        let most = (1..=most)
            .rev()
            .find(|&b| (Shape::new(b, glimpse()).needs(Buf::Cache) * 4) as u64 <= cache_bytes)
            .unwrap_or(1);
        let alone = Shape::new(1, longest());
        let together = Shape::new(most, glimpse());
        let of: HashMap<Buf, usize> = SIZED
            .iter()
            .map(|&b| (b, alone.needs(b).max(together.needs(b))))
            .collect();
        let staging = of[&Buf::Tokens].max(VOCAB).max(of[&Buf::Audio]).max(read);
        Room { of, staging, most }
    }

    /// The sized buffers and their f32s.
    pub(super) fn buffers(&self) -> impl Iterator<Item = (Buf, usize)> + '_ {
        self.of.iter().map(|(&b, &n)| (b, n))
    }

    /// Whether `shape`'s pieces fit.
    pub(super) fn fits(&self, shape: &Shape) -> bool {
        SIZED.iter().all(|&buf| shape.needs(buf) <= self.of[&buf])
            && shape.needs(Buf::Tokens) <= self.staging
    }

    /// How many pieces of `frames` frames are heard together: as many as
    /// fit, at most [`Room::most`].
    pub(super) fn at_once(&self, frames: usize) -> usize {
        (1..=self.most)
            .rev()
            .find(|&b| self.fits(&Shape::new(b, frames)))
            .unwrap_or(1)
    }
}
