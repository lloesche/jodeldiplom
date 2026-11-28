//! How the story's models reckon on this GPU ([`Precision`]), as their
//! callers ask.

/// How a model reckons on this GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Precision {
    /// As fast as the GPU does well: the picture models' large matrix
    /// multiplies and attention on tensor cores (their inputs rounded to
    /// f16) where it has them; the language model's quantised matrices
    /// taking their rows at 8 bits where it multiplies bytes four at a time
    /// natively (not Metal, where that is emulated). Elsewhere as
    /// [`Precision::Full`].
    Fast,
    /// f32 throughout.
    Full,
}
