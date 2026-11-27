//! Karaoke story mode's models: where setup ([`crate::setup`]) puts them
//! in the store, where they come from (each repository at a fixed
//! revision) and their files' lengths and SHA-256s. All four are
//! Apache-2.0.
//!
//! In the store (`models` in the data folder), under `story/`, each model
//! is kept as its repository publishes it, the files at the same paths
//! (those of a repository that the player reads):
//!
//! ```text
//! story/gemma-4-E4B-it-Q4_K_M.gguf            the language model
//! story/flux2-klein-4b-fp8/                    the image model (its transformer)
//!     flux-2-klein-4b-fp8.safetensors
//! story/flux2-klein-4b/                        its decoder and settings
//!     model_index.json
//!     scheduler/scheduler_config.json
//!     transformer/config.json
//!     vae/config.json
//!     vae/diffusion_pytorch_model.safetensors
//! story/qwen3-4b-fp8/                          its text encoder, with the tokenizer
//!     config.json
//!     model.safetensors.index.json
//!     model-00001-of-00002.safetensors
//!     model-00002-of-00002.safetensors
//!     merges.txt
//!     tokenizer_config.json
//!     vocab.json
//! ```
//!
//! The image model's prompts are read with the encoder's tokenizer
//! (`jodel_gpu::image`'s reads `vocab.json`, `merges.txt` and the added
//! tokens of `tokenizer_config.json`): FLUX.2 [klein]'s own `tokenizer/` is
//! the same (the same files, its chat template the one in
//! `tokenizer_config.json`).

use std::path::PathBuf;

/// The language model: Gemma 4 E4B-it by Google DeepMind (Apache-2.0),
/// quantised by Unsloth to Q4_K_M (`unsloth/gemma-4-E4B-it-GGUF`), as
/// published.
pub const LANGUAGE_FILE: &str = "story/gemma-4-E4B-it-Q4_K_M.gguf";
/// Where it is downloaded from, at a fixed revision.
pub const LANGUAGE_URL: &str = "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/bfc15c382204943c3a8fff0c750b94ae2364d7a3/gemma-4-E4B-it-Q4_K_M.gguf";
/// Its length in bytes.
pub const LANGUAGE_BYTES: u64 = 4_977_171_584;
/// Its SHA-256.
pub const LANGUAGE_SHA256: &str =
    "85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87";

/// The image model's folder: FLUX.2 [klein] 4B's transformer at 8 bits,
/// Black Forest Labs' own FP8 release (`black-forest-labs/FLUX.2-klein-4b-fp8`,
/// Apache-2.0).
pub const TRANSFORMER_DIR: &str = "story/flux2-klein-4b-fp8";
/// Its repository at a fixed revision.
pub(crate) const TRANSFORMER_BASE: &str = "https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/5b4408e59397a4a37ccb46afe426d8ed86379441";
/// Its files: path, length, SHA-256.
pub(crate) const TRANSFORMER_FILES: [(&str, u64, &str); 1] = [(
    "flux-2-klein-4b-fp8.safetensors",
    4_070_624_520,
    "97ed34fe0567e436200f2faee3939b88f2b5d99f8af2a4dc16532c4245c0ccb6",
)];

/// The image model's decoder (its VAE) and settings: from FLUX.2 [klein]
/// 4B's own repository (`black-forest-labs/FLUX.2-klein-4B`, Apache-2.0),
/// its pipeline's, scheduler's, transformer's and VAE's settings and the
/// VAE's weights.
pub const PIPELINE_DIR: &str = "story/flux2-klein-4b";
/// Its repository at a fixed revision.
pub(crate) const PIPELINE_BASE: &str = "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/e7b7dc27f91deacad38e78976d1f2b499d76a294";
/// Its files: path, length, SHA-256.
pub(crate) const PIPELINE_FILES: [(&str, u64, &str); 5] = [
    (
        "vae/diffusion_pytorch_model.safetensors",
        168_120_878,
        "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04",
    ),
    (
        "vae/config.json",
        821,
        "0d6dfb69ae95a5e2ac9836284bbb63d8b38ce67b25ba2dff380752b2a10ab948",
    ),
    (
        "transformer/config.json",
        541,
        "09733c74a3da6d17dd0a0472a091a8950c7c6935889c32c16cc800ede05029de",
    ),
    (
        "scheduler/scheduler_config.json",
        486,
        "067afb012cef64553a763447d1efd93daeffcc0123ca7e25b09f8de20b90762e",
    ),
    (
        "model_index.json",
        446,
        "51a76cb1cf3ed37423a1128c79c22faee8e6fbe7f5aaeb737f0a258930dbaac0",
    ),
];

/// The image model's text encoder: Qwen3-4B at 8 bits, Qwen's own FP8
/// release (`Qwen/Qwen3-4B-FP8`, Apache-2.0), with its tokenizer.
pub const ENCODER_DIR: &str = "story/qwen3-4b-fp8";
/// Its repository at a fixed revision.
pub(crate) const ENCODER_BASE: &str =
    "https://huggingface.co/Qwen/Qwen3-4B-FP8/resolve/96b30dc13593a244a5e59e84687309f53c375cfa";
/// Its files: path, length, SHA-256.
pub(crate) const ENCODER_FILES: [(&str, u64, &str); 7] = [
    (
        "model-00001-of-00002.safetensors",
        4_412_140_424,
        "30b1bebf7b81f39f280b09c74f90feed00835a73145955a8d2fb69543b8371fb",
    ),
    (
        "model-00002-of-00002.safetensors",
        777_912_448,
        "79f82dbc594a0237e24982800665eb0ddbe8c7bbdf3ac49c2a0005682aca15dc",
    ),
    (
        "model.safetensors.index.json",
        55_847,
        "be5d78efd19058e26d5363a322f148b16ade6dcd85a6f335139595df0e4eb64b",
    ),
    (
        "config.json",
        894,
        "68fff9378aacf280608bbe1fbf594432a7e1eaeee12d738d53ed55024da1fc29",
    ),
    (
        "vocab.json",
        2_776_833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
    (
        "merges.txt",
        1_671_853,
        "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    ),
    (
        "tokenizer_config.json",
        9_732,
        "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    ),
];

/// Karaoke story mode's models where setup found them
/// ([`crate::setup::Found`]): the language model's file, and each image
/// model's folder (its files at the paths in this module's layout). Read
/// a file through [`crate::store::mover::locate`]: while the data folder
/// moves, one not yet moved is read where it was.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoryModels {
    /// The language model's file ([`LANGUAGE_FILE`]).
    pub language: PathBuf,
    /// The image model's folder ([`TRANSFORMER_DIR`]).
    pub transformer: PathBuf,
    /// Its decoder's and settings' folder ([`PIPELINE_DIR`]).
    pub pipeline: PathBuf,
    /// Its text encoder's folder ([`ENCODER_DIR`]).
    pub encoder: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A folder's files: path, length, SHA-256.
    type Table = &'static [(&'static str, u64, &'static str)];

    /// The files `jodel_gpu::image` opens in each folder, as it names them.
    const READ: [(&str, Table, &[&str]); 3] = [
        (
            TRANSFORMER_DIR,
            &TRANSFORMER_FILES,
            jodel_gpu::image::TRANSFORMER_FILES,
        ),
        (
            PIPELINE_DIR,
            &PIPELINE_FILES,
            jodel_gpu::image::PIPELINE_FILES,
        ),
        (ENCODER_DIR, &ENCODER_FILES, jodel_gpu::image::ENCODER_FILES),
    ];

    #[test]
    fn every_file_the_image_engine_opens_is_downloaded() {
        for (dir, files, read) in READ {
            for name in read {
                assert!(
                    files.iter().any(|(file, _, _)| file == name),
                    "{dir}/{name} is read but not downloaded"
                );
            }
        }
    }
}
