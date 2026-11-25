//! What the canceller keeps of the reference: its bins block by block, as
//! far back as the window may reach, as played and whitened.

use super::fft::{Bins, BINS, C32};

/// The reference's channels the filters learn from: what both speakers
/// play alike (the mean of the two), and what they play apart (half their
/// difference). A laptop's microphone sits nearer one speaker than the
/// other, so it hears each its own way; from the mean alone, what the two
/// play apart (a song's wide guitars, its reverb) cannot be predicted at
/// all. The two are far less alike than left and right are, so each learns
/// at its own pace; of a song in mono the second is silent and its filter
/// never moves.
pub(super) const CHANNELS: usize = 2;

/// One block's bins, a set per channel.
pub(super) type Frame = [Bins; CHANNELS];

/// A frame of nothing.
pub(super) const SILENT: Frame = [[C32 { re: 0.0, im: 0.0 }; BINS]; CHANNELS];

/// The reference's bins, block by block, as far back as the window may
/// reach: a ring.
pub(super) struct History {
    spectra: Vec<Frame>,
    /// The newest.
    head: usize,
    /// Blocks pushed since it was cleared (at most its length).
    filled: usize,
}

impl History {
    pub(super) fn new(len: usize) -> History {
        History {
            spectra: vec![SILENT; len],
            head: 0,
            filled: 0,
        }
    }

    /// Blocks it holds.
    pub(super) fn len(&self) -> usize {
        self.spectra.len()
    }

    pub(super) fn push(&mut self, x: Frame) {
        self.head = (self.head + 1) % self.spectra.len();
        self.spectra[self.head] = x;
        self.filled = (self.filled + 1).min(self.spectra.len());
    }

    /// Blocks pushed since it was cleared (at most its length).
    pub(super) fn filled(&self) -> usize {
        self.filled
    }

    /// The bins `age` blocks back (0: the newest).
    pub(super) fn at(&self, age: usize) -> &Frame {
        let len = self.spectra.len();
        &self.spectra[(self.head + len - age % len) % len]
    }

    pub(super) fn clear(&mut self) {
        self.spectra.fill(SILENT);
        self.filled = 0;
    }
}

/// How fast the whitener follows each frequency's steady part (per block).
const WHITE_FOLLOW: f32 = 0.02;
/// How much of a steady tone it may take out (about 30 dB).
const WHITE_MOST: f32 = 0.97;

/// One channel of the reference whitened for learning: in each frequency,
/// what of this frame the frame two blocks before (which shares no sample
/// with it) predicts, taken out. A steady tone turns the same way every
/// block, so it goes; noise and the music's changes stay. The filters step
/// along the whitened reference, against the error they left as they are,
/// and are normalised by the whitened reference's power: a tone no longer
/// outweighs the rest of its frequency band. Along the tone itself, a
/// filter fits that one note, and the rest of the band, where a voice heard
/// back may be, takes many seconds to learn; whitened, all of it learns at
/// once. (Whitening the error too would match the reference only if the
/// filter held still between the frames.)
pub(super) struct Whitener {
    /// Each bin's predictor: the smoothed product of a frame with the one
    /// two before, and the earlier one's power.
    cross: Bins,
    earlier: [f32; BINS],
}

impl Whitener {
    pub(super) fn new() -> Whitener {
        Whitener {
            cross: [C32::default(); BINS],
            earlier: [0.0; BINS],
        }
    }

    /// Bin `k`'s predictor (at most [`WHITE_MOST`] of the earlier frame).
    fn predictor(&self, k: usize) -> C32 {
        let p = self.earlier[k];
        if p <= 1e-20 {
            return C32::default();
        }
        let c = self.cross[k].scale(1.0 / p);
        let size = c.norm().sqrt();
        if size > WHITE_MOST {
            c.scale(WHITE_MOST / size)
        } else {
            c
        }
    }

    /// The channel's frame `x` whitened, `before` the frame two blocks
    /// earlier; the predictors learn from them first.
    pub(super) fn reference(&mut self, x: &Bins, before: &Bins) -> Bins {
        let mut out = [C32::default(); BINS];
        for k in 0..BINS {
            let g = before[k].conj_mul(x[k]);
            self.cross[k] = self.cross[k].add(g.sub(self.cross[k]).scale(WHITE_FOLLOW));
            self.earlier[k] += (before[k].norm() - self.earlier[k]) * WHITE_FOLLOW;
            out[k] = x[k].sub(self.predictor(k).mul(before[k]));
        }
        out
    }
}
