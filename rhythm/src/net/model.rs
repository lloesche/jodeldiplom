//! The Beat This! network (small): spectrogram frames in, a beat and a
//! downbeat logit per frame out.
//!
//! - Stem: per-band scale and shift; a convolution of 4 bands by 3
//!   frames, 4 bands apart, to 32 channels; scale and shift; GELU.
//! - Three blocks, at 32, 64 and 128 channels: attention and a
//!   feed-forward step across the bands of each frame, then across the
//!   frames of each band; a convolution of 2 bands by 3 frames, 2 bands
//!   apart, doubling the channels; scale and shift; GELU. The bands go
//!   32 → 16 → 8 → 4.
//! - Each frame's 256 channels × 4 bands, channel-major, to 128 values.
//! - Six transformer layers of 128 (attention, feed-forward), a final
//!   RMS normalisation, and a layer to two logits: the downbeat, and
//!   the beat as beat plus downbeat.

use super::layers::{
    gelu, par_chunks, Affine, Attention, Conv, FeedForward, Linear, RmsNorm, Rotary,
};
use super::mel::BANDS;
use super::weights::Weights;

/// The widest sequence the rotary tables cover (frames per chunk).
pub(crate) const MAX_FRAMES: usize = 1500;

struct Block {
    attn_f: Attention,
    ff_f: FeedForward,
    attn_t: Attention,
    ff_t: FeedForward,
    conv: Conv,
    norm: Affine,
}

/// The network with its weights.
pub(crate) struct Model {
    bands_in: Affine,
    stem: Conv,
    stem_norm: Affine,
    blocks: Vec<Block>,
    project: Linear,
    layers: Vec<(Attention, FeedForward)>,
    norm: RmsNorm,
    head: Linear,
    rotary: Rotary,
}

const STEM: usize = 32;
const DIM: usize = 128;
const LAYERS: usize = 6;

impl Model {
    /// The model from a weights file's tensors; every tensor must be used.
    pub fn load(mut w: Weights) -> Result<Model, String> {
        let w = &mut w;
        let mut blocks = Vec::new();
        let mut c = STEM;
        for b in 0..3 {
            let n = format!("frontend.blocks.{b}");
            blocks.push(Block {
                attn_f: Attention::load(w, &format!("{n}.partial.attnF"), c)?,
                ff_f: FeedForward::load(w, &format!("{n}.partial.ffF"), c)?,
                attn_t: Attention::load(w, &format!("{n}.partial.attnT"), c)?,
                ff_t: FeedForward::load(w, &format!("{n}.partial.ffT"), c)?,
                conv: Conv::load(w, &format!("{n}.conv2d"), c, 2 * c, 2)?,
                norm: Affine::load(w, &format!("{n}.norm"), 2 * c)?,
            });
            c *= 2;
        }
        let bands_out = BANDS / 4 / 8;
        let mut layers = Vec::new();
        for l in 0..LAYERS {
            let n = format!("transformer_blocks.layers.{l}");
            layers.push((
                Attention::load(w, &format!("{n}.0"), DIM)?,
                FeedForward::load(w, &format!("{n}.1"), DIM)?,
            ));
        }
        let model = Model {
            bands_in: Affine::load(w, "frontend.stem.bn1d", BANDS)?,
            stem: Conv::load(w, "frontend.stem.conv2d", 1, STEM, 4)?,
            stem_norm: Affine::load(w, "frontend.stem.bn2d", STEM)?,
            blocks,
            project: Linear::load(w, "frontend.linear", c * bands_out, DIM, true)?,
            layers,
            norm: RmsNorm::load(w, "transformer_blocks.norm", DIM)?,
            head: Linear::load(w, "task_heads.beat_downbeat_lin", DIM, 2, true)?,
            rotary: Rotary::new(MAX_FRAMES),
        };
        let left = w.left();
        if !left.is_empty() {
            return Err(format!("unused tensors: {left:?}"));
        }
        Ok(model)
    }

    /// The beat and downbeat logits of `frames` spectrogram frames
    /// (frame-major, [`BANDS`] each), at most [`MAX_FRAMES`].
    pub fn forward(&self, spect: &[f32], frames: usize, threads: usize) -> (Vec<f32>, Vec<f32>) {
        assert!(frames <= MAX_FRAMES && spect.len() == frames * BANDS);
        // [band][frame][channel], one channel, each band scaled.
        let mut x = vec![0.0f32; BANDS * frames];
        for t in 0..frames {
            for f in 0..BANDS {
                let v = spect[t * BANDS + f];
                x[f * frames + t] = v * self.bands_in.scale[f] + self.bands_in.shift[f];
            }
        }
        let mut x = self.stem.apply(&x, BANDS, frames, threads);
        let mut bands = BANDS / 4;
        self.stem_norm.apply(&mut x);
        par_chunks(&mut x, 1 << 14, threads, |_, p| gelu(p));
        let mut c = STEM;
        for block in &self.blocks {
            // Across the bands of each frame: [frame][band][channel].
            let mut y = transpose(&x, bands, frames, c);
            block.attn_f.add_to(&mut y, bands, &self.rotary, threads);
            block.ff_f.add_to(&mut y, threads);
            // Across the frames of each band: back to [band][frame][channel].
            let mut x2 = transpose(&y, frames, bands, c);
            block.attn_t.add_to(&mut x2, frames, &self.rotary, threads);
            block.ff_t.add_to(&mut x2, threads);
            x = block.conv.apply(&x2, bands, frames, threads);
            bands /= 2;
            c *= 2;
            block.norm.apply(&mut x);
            par_chunks(&mut x, 1 << 14, threads, |_, p| gelu(p));
        }
        // Each frame: channel-major (c, band).
        let mut flat = vec![0.0f32; frames * c * bands];
        for f in 0..bands {
            for t in 0..frames {
                let src = &x[(f * frames + t) * c..][..c];
                let row = &mut flat[t * c * bands..][..c * bands];
                for (ch, v) in src.iter().enumerate() {
                    row[ch * bands + f] = *v;
                }
            }
        }
        let mut h = self.project.apply(&flat, threads);
        for (attn, ff) in &self.layers {
            attn.add_to(&mut h, frames, &self.rotary, threads);
            ff.add_to(&mut h, threads);
        }
        let h = self.norm.apply(&h);
        let out = self.head.apply(&h, threads);
        let downbeat: Vec<f32> = out.chunks(2).map(|o| o[1]).collect();
        let beat = out.chunks(2).map(|o| o[0] + o[1]).collect();
        (beat, downbeat)
    }
}

/// `[a][b][c]` to `[b][a][c]`.
fn transpose(x: &[f32], a: usize, b: usize, c: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    for i in 0..a {
        for j in 0..b {
            y[(j * a + i) * c..][..c].copy_from_slice(&x[(i * b + j) * c..][..c]);
        }
    }
    y
}
