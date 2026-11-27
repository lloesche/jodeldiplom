//! Pictures on the GPU: FLUX.2 [klein] 4B (Black Forest Labs, Apache-2.0)
//! from its prompt and optional reference pictures, as the karaoke
//! story's scenes. Two parts, each loaded and freed on its own (each a
//! [`crate::Resident`]):
//!
//! - the text encoder ([`TextEncoder`], `encoder.rs`): Qwen3-4B from
//!   Qwen's FP8 release, its prompt in the chat template of the model's
//!   tokenizer (`text.rs`), padded to 512 tokens; the hidden states after
//!   layers 9, 18 and 27 side by side are the prompt's embeddings
//!   ([`Embeds`]), so only those 27 layers are loaded;
//! - the transformer and the autoencoder ([`Drawer`], `transformer.rs`,
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
//! matrices in f16 with f32 sums, of 16 × 16 × 16 (tensor cores: Vulkan on
//! NVIDIA, for one) or of 8 × 8 × 8 (Metal on Apple's GPUs), the large
//! matrix multiplies run on them ([`crate::Precision::Fast`]), their inputs
//! rounded to f16, and on 16 × 16 the transformer's attention too; else,
//! and with [`crate::Precision::Full`], on f32 multiply-adds.
//!
//! The kernels (the constants they share with the plan, `prelude.wgsl`,
//! `../common.wgsl`, then `tail.wgsl`, `gemm.wgsl`, `ops.wgsl`,
//! `attention.wgsl`, and on such a GPU `coop.wgsl`, its tiles' layer
//! `coop16.wgsl` or `coop8.wgsl`, `gemm_coop.wgsl`, and on 16 × 16
//! `attention_coop.wgsl`): the matrix multiply in tiles of 128 × 128 with
//! its tail fused (a bias, added to what is there, times a gate per column,
//! or SwiGLU's pairs), reading a convolution's patches where they lie (the
//! group norm and SiLU before it taken on the way); flash attention (an
//! online softmax over tiles of keys, the scores never stored); RMSNorm,
//! LayerNorm with the modulation, the heads' norm and rotary positions,
//! the group norm's statistics, the latents' patches, a softmax and a
//! transpose. A picture is one plan of dispatches (`plan.rs`) run in short
//! submissions that give way to the window ([`crate::pace`]); its buffers
//! are kept for the next picture of the same size.

mod drawer;
mod encoder;
mod gpu;
mod plan;
mod sampling;
#[cfg(test)]
mod tests;
mod text;
mod transformer;
mod vae;
mod weights;

use crate::engine::{self, Access, After, Kernels};
use gpu::Tiles;

pub use drawer::{Drawer, Part, Picture, Reference, PIPELINE_FILES, STEPS, TRANSFORMER_FILES};
pub use encoder::{Embeds, TextEncoder, ENCODER_FILES};
pub use sampling::{noise, schedule};
pub use text::Tokenizer;

/// The kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Kernel {
    Gemm,
    /// The matrix multiply on tensor cores (only where the GPU has them).
    GemmCoop,
    RmsNorm,
    LayerNormMod,
    QkPrep,
    Attention,
    /// Attention on tensor cores (only on 16 × 16 ones:
    /// [`Tiles::attention`]).
    AttentionCoop,
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
        Kernel::GemmCoop,
        Kernel::RmsNorm,
        Kernel::LayerNormMod,
        Kernel::QkPrep,
        Kernel::Attention,
        Kernel::AttentionCoop,
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
            Kernel::GemmCoop => "gemm_coop",
            Kernel::RmsNorm => "rmsnorm",
            Kernel::LayerNormMod => "layernorm_mod",
            Kernel::QkPrep => "qk_prep",
            Kernel::Attention => "attention",
            Kernel::AttentionCoop => "attention_coop",
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

impl Kernel {
    /// The kernels a device runs whose tensor-core kernels take `coop`'s
    /// tiles: the tensor-core multiply wherever there are tiles, the
    /// tensor-core attention where they take it too ([`Tiles::attention`]).
    pub(crate) fn on(coop: Option<Tiles>) -> Vec<Kernel> {
        <Kernel as Kernels>::ALL
            .iter()
            .copied()
            .filter(|k| match k {
                Kernel::GemmCoop => coop.is_some(),
                Kernel::AttentionCoop => coop.is_some_and(Tiles::attention),
                _ => true,
            })
            .collect()
    }
}

/// The numbers the kernels share with the plan, declared for them: the
/// matrix multiply's flags (the engine's, how W is held, its own, the
/// group norm's) and its numbers by name (the engine's ten as `mm_m()`
/// and on, then the picture models' own).
fn constants() -> String {
    let mut out = engine::wgsl_consts(&[
        ("MM_BIAS", engine::BIAS),
        ("MM_ADD", After::Add.flag()),
        ("MM_SWIGLU", After::SwiGlu.flag()),
        ("MM_BF16", plan::MM_BF16),
        ("MM_F32", plan::MM_F32),
        ("MX_CONV", plan::MX_CONV),
        ("MX_GATED", plan::MX_GATED),
        ("GN_NORM", plan::GN_NORM),
        ("GN_SILU", plan::GN_SILU),
    ]);
    let engine_ten = (0..)
        .zip(engine::MUL_NUMBERS)
        .map(|(i, n)| (format!("mm_{n}"), i));
    out += &engine::wgsl_numbers("n", engine_ten);
    out += &engine::wgsl_numbers("n", plan::MM_NUMBERS);
    out
}

/// The kernels' source, with the tensor-core kernels on `coop`'s tiles
/// where the device has cooperative matrices and f16 (the attention only
/// where the tiles take it: [`Kernel::on`]).
pub(crate) fn source(coop: Option<Tiles>) -> String {
    let mut parts = Vec::new();
    if coop.is_some() {
        parts.push("enable f16;\nenable wgpu_cooperative_matrix;\n".to_owned());
    }
    parts.extend(
        [
            include_str!("prelude.wgsl"),
            &constants(),
            include_str!("../common.wgsl"),
            include_str!("tail.wgsl"),
            include_str!("gemm.wgsl"),
            include_str!("ops.wgsl"),
            include_str!("attention.wgsl"),
        ]
        .map(str::to_owned),
    );
    if let Some(tiles) = coop {
        parts.push(include_str!("coop.wgsl").to_owned());
        parts.push(tiles.layer().to_owned());
        parts.push(tiles.kernel(include_str!("gemm_coop.wgsl")));
        if tiles.attention() {
            parts.push(tiles.kernel(include_str!("attention_coop.wgsl")));
        }
    }
    parts.join("\n")
}

impl Tiles {
    /// The kernels' layer for these tiles.
    fn layer(self) -> &'static str {
        match self {
            Tiles::Sixteen => include_str!("coop16.wgsl"),
            Tiles::Eight => include_str!("coop8.wgsl"),
        }
    }

    /// A tensor-core kernel's source on these tiles: on 16 × 16 its
    /// products (`cm_mma`) are the cooperative matrices' own
    /// (`coopMultiplyAdd`, the same arguments), written in place.
    fn kernel(self, source: &str) -> String {
        match self {
            Tiles::Sixteen => source.replace("cm_mma(", "coopMultiplyAdd("),
            Tiles::Eight => source.to_owned(),
        }
    }
}
