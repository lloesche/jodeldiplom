//! The two models a story's scenes are taken apart with on the GPU
//! ([`GpuNets`]): Depth Anything 3 Small's depth (Apache-2.0) and BEN2
//! Base's matte (MIT), `jodel_gpu::vision`, their weights where setup put
//! them ([`SceneModels`]). They are one job of the GPU's stages
//! ([`Stage::Scenes`]): set up on the first picture of a turn, let go as
//! the stages plan.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use jodel_gpu::vision::prep::Rgb;
use jodel_gpu::vision::{Depth, Matte};
use jodel_gpu::{Footprint, GpuDepth, GpuMatte, Kept, OnDevice, Resident};

use super::SceneNets;
use crate::setup::Unreadable;
use crate::work::stages::{self, Stage};

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

/// The scenes' models' weights where setup found them
/// ([`crate::setup::Found`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SceneModels {
    /// Depth Anything 3 Small's ([`DEPTH_FILE`]).
    pub depth: PathBuf,
    /// BEN2 Base's ([`MATTE_FILE`]).
    pub matte: PathBuf,
}

/// The two models on the GPU.
struct Vision {
    depth: GpuDepth,
    matte: GpuMatte,
}

impl OnDevice for Vision {
    fn lost(&self) -> bool {
        self.depth.lost() || self.matte.lost()
    }
}

/// The checkpoint at `path`'s bytes.
fn read(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

/// What the two models take on the GPU, from their checkpoints' shapes
/// (read for this, once a process, as the GPU's stages first ask): both
/// weights while set up, and the larger of their working buffers while a
/// picture is taken apart (one runs at a time, and lets its buffers go
/// after); `None` while a checkpoint is not there (nothing read then).
fn footprint(files: &SceneModels) -> Option<Footprint> {
    if !files.depth.is_file() || !files.matte.is_file() {
        return None;
    }
    let depth = GpuDepth::footprint(read(&files.depth).ok()?).ok()?;
    let matte = GpuMatte::footprint(read(&files.matte).ok()?).ok()?;
    Some(Footprint {
        resident: depth.resident + matte.resident,
        working: depth.working.max(matte.working),
    })
}

/// The scenes' two models on the GPU, registered for the scenes' stage.
pub struct GpuNets {
    vision: Arc<Kept<Vision>>,
}

impl GpuNets {
    /// The models of the weights in `files`, set up on their first
    /// picture; one that cannot be read or set up told to `unreadable`
    /// (checked, and got again where it is gone or damaged; then tried
    /// again at once).
    pub fn new(files: SceneModels, unreadable: Unreadable) -> GpuNets {
        let (plan, load) = (files.clone(), files);
        let told = unreadable.clone();
        let vision = Arc::new(Kept::new(
            move || footprint(&plan),
            Box::new(move || {
                let depth = told.report(DEPTH_FILE, read(&load.depth).and_then(GpuDepth::new))?;
                let matte = told.report(MATTE_FILE, read(&load.matte).and_then(GpuMatte::new))?;
                let name = depth.name();
                Ok((Vision { depth, matte }, name))
            }),
        ));
        unreadable.retries(DEPTH_FILE, &vision);
        unreadable.retries(MATTE_FILE, &vision);
        stages::register(Stage::Scenes, Arc::clone(&vision) as Arc<dyn Resident>);
        GpuNets { vision }
    }

    /// The bytes the models keep on the GPU between pictures (their
    /// weights), and at most while one is taken apart (the weights and
    /// the larger model's working buffers: one runs at a time); none where
    /// they are not set up (nothing is set up for this).
    pub fn gpu_bytes(&self) -> (u64, u64) {
        let sizes = (self.vision).with_set_up(|v| (v.depth.gpu_sizes(), v.matte.gpu_sizes()));
        match sizes {
            Some(((dw, dp), (mw, mp))) => (dw + mw, dw + mw + dp.max(mp)),
            None => (0, 0),
        }
    }
}

/// Each model's working buffers let go once it has run (its weights stay);
/// set up first where they are not (and again where their GPU was lost).
/// Only inside the scenes' turn.
impl SceneNets for GpuNets {
    fn depth(&self, picture: &Rgb) -> Result<Depth, String> {
        self.vision.run(|v| {
            let found = v.depth.depth(picture);
            v.depth.release();
            found
        })
    }

    fn matte(&self, picture: &Rgb) -> Result<Matte, String> {
        self.vision.run(|v| {
            let cut = v.matte.matte(picture);
            v.matte.release();
            cut
        })
    }
}
