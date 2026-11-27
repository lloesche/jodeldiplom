//! The player's networks on the GPU, through `wgpu` (Vulkan, DirectX 12 or
//! Metal), each giving what its CPU version gives to within rounding:
//!
//! - the stem separator of [`jodel_rhythm::stems`] ([`GpuSeparator`],
//!   `separator.rs`);
//! - the beat network of [`jodel_rhythm::net`] ([`GpuNetwork`], `net.rs`);
//! - the speech model of [`jodel_lyrics::speech`] ([`GpuSpeech`],
//!   `speech.rs`) and the speech recogniser of [`jodel_lyrics::asr`]
//!   ([`GpuAsr`], `asr/`);
//! - the story scenes' depth and cut-out ([`GpuDepth`], [`GpuMatte`], `vision/`);
//! - the karaoke story mode's language model, Gemma 4 E4B's text decoder
//!   ([`Gemma`], `llm/`);
//! - the karaoke story's pictures, FLUX.2 [klein] 4B ([`image`]: its text
//!   encoder and its transformer with the autoencoder, each loaded and
//!   freed on its own).
//!
//! Each is a program of kernels on one bind group layout, its dispatches
//! planned on the processor (every matrix multiply as the engine's `Mul`)
//! and run in short submissions that share the GPU with the window
//! ([`pace`]); what they share is in `engine/`. The separator and the beat
//! network pack their weights from the CPU models as those hold them.
//!
//! A device can be lost (a driver reset, a GPU unplugged or switched).
//! Every network is held as a [`Kept`]: set up again on its next use, and
//! a setup that failed tried again a while later. The process keeps the
//! separators ([`GpuSeparator::shared`] and [`GpuSeparator::background`])
//! and the beat network ([`GpuNetwork::shared`]) so; the caller keeps the
//! others, each saying when its device is lost ([`OnDevice`]:
//! [`GpuSpeech`], [`GpuAsr`], [`GpuDepth`], [`GpuMatte`], [`Gemma`], the
//! picture model's parts). The devices the vision networks and the
//! picture model's parts share are set up again the same way.
//!
//! The GPU is chosen once a process ([`chosen`], `device.rs`): one `wgpu`
//! instance, the adapters on every backend listed once, the best one kept
//! (and the same card through DirectX 12, where the best is on another
//! backend), and every model's device made on it. Listing the adapters
//! starts every GPU's driver, an unused integrated one's too, so the
//! choice does not list them again for each model; a failed choice (no
//! GPU) is kept too. Setups and teardowns take turns ([`setting_up`]).
//! [`report`] says what the GPU side finds on this machine (`diag.rs`),
//! and [`memory::budget`] what the card's memory allows this process now.
//! Each network the process keeps is a [`Resident`] ([`Kept`]): what it
//! takes on the GPU is worked out from its shapes before it is set up
//! (its [`Footprint`]), and the player's scheduler may let it go.
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
pub mod image;
pub mod llm;
pub mod memory;
mod net;
pub mod pace;
mod packer;
mod precision;
pub mod profile;
mod resident;
mod separator;
mod speech;
pub mod vision;

pub use asr::{GpuAsr, Hearing, MAX_NEW, MAX_SECONDS};
pub use device::{card, chosen, gpu_name, setting_up, Chosen, Turn, Turned};
pub use diag::report;
pub use engine::{Kept, OnDevice, SetUp, UNKNOWN};
pub use llm::Gemma;
pub use net::GpuNetwork;
pub use precision::Precision;
pub use resident::{Footprint, Resident};
pub use separator::GpuSeparator;
pub use speech::GpuSpeech;
pub use vision::{GpuDepth, GpuMatte};
