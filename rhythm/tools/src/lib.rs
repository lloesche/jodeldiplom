//! The tools behind `jodel-rhythm`'s built-in pitch weights
//! (`rhythm/models/pitch.bin`), kept out of the library the player builds:
//!
//! - [`pack`]: a packed weights file written, its matrices fitted to K
//!   blocks (by GPTQ for the built-in file), which `jodel_rhythm::nn`
//!   reads;
//! - `pitch_pack` (a binary): the network's full precision weights packed;
//! - `pitch_quant` (a binary): packed weights measured on voices, and the
//!   statistics the packing weighs its errors by gathered.
//!
//! The Python converters beside this crate (`rhythm/tools/*.py`) make the
//! full precision files from the authors' checkpoints.

pub mod pack;
