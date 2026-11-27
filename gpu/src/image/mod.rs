//! Pictures on the GPU: FLUX.2 [klein] 4B (Black Forest Labs, Apache-2.0)
//! from its prompt and optional reference pictures, as the karaoke
//! story's scenes. Three parts, each loaded and freed on its own:
//!
//! - the text encoder ([`TextEncoder`], `encoder.rs`): Qwen3-4B from
//!   Qwen's FP8 release, its prompt in the chat template of the model's
//!   tokenizer (`text.rs`), padded to 512 tokens; the hidden states after
//!   layers 9, 18 and 27 side by side are the prompt's embeddings
//!   ([`Embeds`]), so only those 27 layers are loaded;
//! - the transformer and the autoencoder ([`Painter`], `transformer.rs`,
//!   `vae.rs`): BFL's FP8 transformer (5 double-stream and 20
//!   single-stream blocks, 24 heads of 128) in four steps of the flow
//!   sampler (`sampling.rs`: the schedule's shift, the noise drawn as
//!   PyTorch draws it on the processor), the reference pictures encoded
//!   by the autoencoder and taken in as more tokens, and the latents
//!   decoded by it.
//!
//! The FP8 matrices stay FP8 on the GPU (four values a word), turned to
//! numbers inside the matrix multiply as it reads them, times the
//! checkpoint's scales (one per matrix, or one per 128 × 128 block); the
//! bf16 ones stay bf16. Activations are f32. Where the GPU has cooperative
//! matrices of 16 × 16 × 16 in f16 with f32 sums (tensor cores: Vulkan on
//! NVIDIA, for one), the large matrix multiplies and the transformer's
//! attention run on them ([`Precision::Fast`]), their inputs rounded to f16;
//! else, and with [`Precision::Full`], on f32 multiply-adds.
//!
//! The kernels (`prelude.wgsl`, `../common.wgsl`, then `tail.wgsl`,
//! `gemm.wgsl`, `ops.wgsl`, `attention.wgsl`, and on such a GPU
//! `gemm_coop.wgsl` and `attention_coop.wgsl`): the matrix multiply in
//! tiles of 128 × 128 with its tail fused (a bias, added to what is there,
//! times a gate per column, or SwiGLU's pairs), reading a convolution's
//! patches where they lie (the group norm and SiLU before it taken on the
//! way); flash attention (an online softmax over tiles of keys, the
//! scores never stored); RMSNorm, LayerNorm with the modulation, the
//! heads' norm and rotary positions, the group norm's statistics, the
//! latents' patches, a softmax and a transpose. A picture is one plan of
//! dispatches (`plan.rs`) run in short submissions that give way to the
//! window ([`crate::pace`]); its buffers are kept for the next picture of
//! the same size.

mod encoder;
mod gpu;
mod painter;
mod plan;
mod sampling;
#[cfg(test)]
mod tests;
mod text;
mod transformer;
mod vae;
mod weights;

use crate::engine::{Access, Kernels};

pub use encoder::{Embeds, TextEncoder};
pub use painter::{Painter, Part, Picture, Reference, STEPS};
pub use sampling::{noise, schedule};
pub use text::Tokenizer;

/// How the large matrix multiplies and the attention reckon.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Precision {
    /// On tensor cores where the GPU has them (inputs rounded to f16,
    /// sums in f32), else as [`Precision::Full`].
    Fast,
    /// f32 throughout.
    Full,
}

/// What a part takes on the GPU, in bytes: its weights while loaded, and
/// its buffers while it works (kept until the next size, or freed).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Footprint {
    pub resident: u64,
    pub working: u64,
}

impl Footprint {
    /// Both.
    pub fn total(&self) -> u64 {
        self.resident + self.working
    }
}

/// The kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Kernel {
    Gemm,
    /// The matrix multiply on tensor cores (where the GPU has none, the
    /// f32 one).
    Gemm16,
    RmsNorm,
    LayerNormMod,
    QkPrep,
    Attention,
    /// Attention on tensor cores (where the GPU has none, the f32 one).
    Attention16,
    Euler,
    Act,
    Copy,
    Patchify,
    Unpatchify,
    GnStats,
    GnReduce,
    Softmax,
    Transpose,
}

impl Kernels for Kernel {
    const ALL: &'static [Kernel] = &[
        Kernel::Gemm,
        Kernel::Gemm16,
        Kernel::RmsNorm,
        Kernel::LayerNormMod,
        Kernel::QkPrep,
        Kernel::Attention,
        Kernel::Attention16,
        Kernel::Euler,
        Kernel::Act,
        Kernel::Copy,
        Kernel::Patchify,
        Kernel::Unpatchify,
        Kernel::GnStats,
        Kernel::GnReduce,
        Kernel::Softmax,
        Kernel::Transpose,
    ];
    const PARAM_BYTES: u64 = 128;
    /// After the parameters: the f32 tensors, a second input.
    const EXTRA: &'static [Access] = &[Access::Read, Access::Read];

    fn entry(self) -> &'static str {
        match self {
            Kernel::Gemm => "gemm",
            Kernel::Gemm16 => "gemm16",
            Kernel::RmsNorm => "rmsnorm",
            Kernel::LayerNormMod => "layernorm_mod",
            Kernel::QkPrep => "qk_prep",
            Kernel::Attention => "attention",
            Kernel::Attention16 => "attention16",
            Kernel::Euler => "euler",
            Kernel::Act => "act",
            Kernel::Copy => "copy",
            Kernel::Patchify => "patchify",
            Kernel::Unpatchify => "unpatchify",
            Kernel::GnStats => "gn_stats",
            Kernel::GnReduce => "gn_reduce",
            Kernel::Softmax => "softmax",
            Kernel::Transpose => "transpose",
        }
    }
}

/// Where the GPU has no tensor cores, the tensor-core kernels' entry points
/// are the f32 kernels (the pipelines are made for every kernel; plans
/// there use the f32 ones).
const PLAIN_FAST: &str = "
@compute @workgroup_size(256)
fn gemm16(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    gemm_f32(wg, li);
}

@compute @workgroup_size(128)
fn attention16(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    attention_f32(wg, li);
}
";

/// The kernels' source, with the tensor-core kernels where `coop` (the
/// device has cooperative matrices and f16).
pub(crate) fn source(coop: bool) -> String {
    let mut parts = Vec::new();
    if coop {
        parts.push("enable f16;\nenable wgpu_cooperative_matrix;\n");
    }
    parts.extend([
        include_str!("prelude.wgsl"),
        include_str!("../common.wgsl"),
        include_str!("tail.wgsl"),
        include_str!("gemm.wgsl"),
        include_str!("ops.wgsl"),
        include_str!("attention.wgsl"),
    ]);
    if coop {
        parts.extend([
            include_str!("gemm_coop.wgsl"),
            include_str!("attention_coop.wgsl"),
        ]);
    } else {
        parts.push(PLAIN_FAST);
    }
    parts.join("\n")
}
