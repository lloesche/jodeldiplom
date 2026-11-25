//! What the crate's networks share: the layers they are built of (plain
//! loops written so the compiler vectorises them, an AVX2 build of the hot
//! ones picked at run time), the weights file they are read from, its
//! blocks of fewer bits, and how many threads they take.
//!
//! The beat network ([`crate::net`]), the stem separator
//! ([`crate::stems`]), the key ([`crate::key`]) and the pitch
//! ([`crate::pitch`]) all run on these.

pub(crate) mod layers;
pub mod quant;
pub mod weights;

/// How many threads a network uses: half the processors, so playback and
/// the interface keep theirs. Every analysis that runs while the player
/// plays takes this many.
pub fn threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| (n.get() / 2).max(1))
        .unwrap_or(1)
}
