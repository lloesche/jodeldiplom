//! A cover taken apart, once per cover, for the karaoke background to move
//! and light cheaply in real time: its subject (an RGBA layer with a soft
//! matte, its colours freed of the old background at the edges), the clean
//! plate behind it (the subject removed and the hole filled), the depth of
//! both, the subject's rim, and the cover's palette ([`CoverLayers`]).
//!
//! The subject's matte is BEN2 Base's (MIT) and the depth Depth Anything 3
//! Small's (Apache-2.0), both on the GPU (`jodel_gpu::vision`); the rest is
//! here (`split.rs`, the fill in `fill.rs`). It is a step of a song's
//! preparation (`worker.rs`, beside the analysis and the lyrics), kept in
//! two levels (`store.rs`): the cover's result once, keyed by its pixels
//! (and an album's re-encoded copies found by a perceptual hash), and a
//! small entry per song that links to it. A machine without a usable GPU
//! or without the weights gets no layers (the background keeps its blur).

mod fill;
mod hash;
mod split;
mod store;
mod worker;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use image::{GrayImage, RgbImage, RgbaImage};

pub use hash::{album_keys, perceptual, pixel_hash, Perceptual};
pub use split::{hole, split, Parts};
pub use store::{from_bytes, to_bytes, CoverStore, Kept, SongCover, Splitter};
pub use worker::{CoverWorker, GpuSplitter};

/// What the shared results are made by: the two models and the way they
/// are taken apart. A result made otherwise is made again.
pub const VERSION: &str = "da3-small ben2-base split-1";

/// The depth model's weights: Depth Anything 3 Small by ByteDance Seed
/// (Apache-2.0), its checkpoint as published.
pub const DEPTH_FILE: &str = "da3-small.safetensors";
/// Where it is downloaded from, at a fixed revision.
pub const DEPTH_URL: &str = "https://huggingface.co/depth-anything/DA3-SMALL/resolve/e08cab65ca0ec38e7826075418411ab90cab4da3/model.safetensors";
/// Its length in bytes.
pub const DEPTH_BYTES: u64 = 137_248_940;
/// Its SHA-256.
pub const DEPTH_SHA256: &str = "364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf";

/// The matte model's weights: BEN2 Base by Prama LLC (MIT), its checkpoint
/// as published.
pub const MATTE_FILE: &str = "ben2-base.safetensors";
/// Where it is downloaded from, at a fixed revision.
pub const MATTE_URL: &str = "https://huggingface.co/PramaLLC/BEN2/resolve/e48a20765fb421d19dcdb0bf3cc61e802ca5ec8f/model.safetensors";
/// Its length in bytes.
pub const MATTE_BYTES: u64 = 380_577_976;
/// Its SHA-256.
pub const MATTE_SHA256: &str = "ea8b7907176a09667c86343dc7d00de6a6d871076cb90bb5f753618fd6fb3ebb";

/// A depth map: nearness in 0..=65535 (larger nearer: the parallax a pixel
/// moves by), `width × height`, rows top to bottom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NearMap {
    /// Its size.
    pub width: u32,
    /// Its size.
    pub height: u32,
    /// The nearness of each pixel.
    pub near: Vec<u16>,
}

impl NearMap {
    /// The nearness at (u, v) in 0..1 of the picture, in 0..1 (nearest
    /// sample).
    pub fn at(&self, u: f32, v: f32) -> f32 {
        let x = ((u * self.width as f32) as u32).min(self.width - 1);
        let y = ((v * self.height as f32) as u32).min(self.height - 1);
        f32::from(self.near[(y * self.width + x) as usize]) / 65535.0
    }
}

/// A colour of the cover's palette and its share of the cover.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Swatch {
    /// The colour.
    pub rgb: [u8; 3],
    /// Its share of the cover's pixels, 0..=1.
    pub share: f32,
}

/// A cover taken apart: plain data for the background's shader.
#[derive(Clone, Debug, PartialEq)]
pub struct CoverLayers {
    /// The cover with its subject removed and the hole filled (the cover
    /// itself where it has no clear subject).
    pub plate: RgbImage,
    /// The subject: its colours (not premultiplied, freed of the old
    /// background at its edges) and its soft matte; `None` where the cover
    /// has no clear subject.
    pub subject: Option<RgbaImage>,
    /// The cover's nearness as it is.
    pub depth: NearMap,
    /// The plate's nearness: under the subject filled from around it.
    pub plate_depth: NearMap,
    /// The subject's rim, for its outline to shimmer: 255 at its edge,
    /// fading inwards; without a subject, the depth's edges.
    pub outline: GrayImage,
    /// The cover's main colours, the largest share first.
    pub palette: Vec<Swatch>,
}

impl CoverLayers {
    /// Its size: the plate's.
    pub fn size(&self) -> (u32, u32) {
        self.plate.dimensions()
    }

    /// Its bytes in memory, roughly.
    pub fn bytes(&self) -> usize {
        self.plate.len()
            + self.subject.as_ref().map_or(0, |s| s.len())
            + 2 * (self.depth.near.len() + self.plate_depth.near.len())
            + self.outline.len()
    }
}

/// The layers as the app holds them: shared, with the cover they are of.
pub type Layers = Arc<CoverLayers>;

/// The covers' models' weights where setup found them
/// ([`crate::setup::Found`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoverModels {
    /// Depth Anything 3 Small's ([`DEPTH_FILE`]).
    pub depth: PathBuf,
    /// BEN2 Base's ([`MATTE_FILE`]).
    pub matte: PathBuf,
}

/// A song's key in the covers' store: as the lyrics' cache keys a song (its
/// path, size and modification time: a file retagged is looked at afresh),
/// or its path's hash where it cannot be read.
pub fn song_key(path: &Path) -> String {
    crate::lyrics::store::cache_key(path)
        .unwrap_or_else(|| format!("{:016x}", crate::rhythm::hash(&path.display().to_string())))
}
