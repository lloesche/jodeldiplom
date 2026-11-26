//! What the crate's networks share: the layers they are built of (plain
//! loops written so the compiler vectorises them, an AVX2 build of the hot
//! ones picked at run time through `jodel_tensors::fast!`), the weights
//! file they are read from and its blocks of fewer bits. How many threads
//! they take is `jodel_tensors::kernels::threads`.
//!
//! The beat network ([`crate::net`]), the stem separator
//! ([`crate::stems`]), the key ([`crate::key`]) and the pitch
//! ([`crate::pitch`]) all run on these, and so do the player's small
//! detectors for a cover's lettering and faces (through [`layers::Linear`]).

pub mod layers;
pub mod quant;
pub mod weights;
