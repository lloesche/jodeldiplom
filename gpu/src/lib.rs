//! The player's networks on the GPU, through `wgpu` (Vulkan, DirectX 12 or
//! Metal), each giving what its CPU version gives to within rounding:
//!
//! - the stem separator of [`jodel_rhythm::stems`] ([`GpuSeparator`],
//!   `separator.rs`);
//! - the beat network of [`jodel_rhythm::net`] ([`GpuNetwork`], `net.rs`);
//! - the speech model of [`jodel_lyrics::speech`] ([`GpuSpeech`],
//!   `speech.rs`) and the speech recogniser of [`jodel_lyrics::asr`]
//!   ([`GpuAsr`], `asr/`);
//! - the cover's depth and cut-out ([`GpuDepth`], [`GpuMatte`], `vision/`).
//!
//! Each is a program of kernels on one bind group layout, its dispatches
//! planned on the processor and run in short submissions that share the
//! GPU with the window ([`pace`]); what they share is in `engine/`.
//!
//! The GPU is chosen once a process ([`chosen`], `device.rs`): one `wgpu`
//! instance, the adapters on every backend listed once, the best one kept
//! (and the same card through DirectX 12, where the best is on another
//! backend), and every model's device made on it. Listing the adapters
//! starts every GPU's driver, an unused integrated one's too, so the
//! choice does not list them again for each model; a failed choice (no
//! GPU) is kept too. Setups and teardowns take turns ([`setting_up`]).
//! [`report`] says what the GPU side finds on this machine (`diag.rs`).
//!
//! ```no_run
//! use jodel_gpu::GpuSeparator;
//! use jodel_rhythm::stems;
//!
//! if let Some(gpu) = GpuSeparator::shared() {
//!     let (left, right) = (vec![0.0; 44_100], vec![0.0; 44_100]);
//!     let parts = stems::parts(gpu, &left, &right).unwrap();
//! }
//! ```

mod asr;
mod device;
mod diag;
mod engine;
mod net;
pub mod pace;
mod packer;
mod separator;
mod speech;
pub mod vision;

pub use asr::{GpuAsr, Hearing, MAX_NEW, MAX_SECONDS};
pub use device::{card, chosen, gpu_name, setting_up, Chosen, Turn, Turned};
pub use diag::report;
pub use net::GpuNetwork;
pub use separator::GpuSeparator;
pub use speech::GpuSpeech;
pub use vision::{GpuDepth, GpuMatte};
