//! The playback engine between files and [`jodel_output`]: FLAC, WAV,
//! AIFF, CAF and ALAC files to bit-perfect device output, and MP3 (MP2,
//! MP1) and Ogg Vorbis files decoded to 24 bits, gapless.
//!
//! # Layers
//!
//! - [`source`]: opens files with symphonia, streams 24-bit PCM
//!   (requantizing lossy, floating-point and 32-bit samples once, with
//!   dither), seeks to an exact frame, and reads tags and the cover.
//! - [`rate`]: the pure rate policy, [`rate::choose_route`]: from the file,
//!   the device's exclusive [`jodel_output::Capabilities`] and the
//!   [`Settings`] to a [`Route`] (output rate, channels, format), or a [`NoRoute`] that says why nothing fits. A bit-perfect
//!   route always wins; when there is none, the route can gain a
//!   [`Resample`] step (unless [`Settings::resample`] is off).
//! - [`resample`]: the [`Resampler`], a rational polyphase FIR in `f64`
//!   (Kaiser-windowed sinc, ≥ 160 dB stopband, linear phase, delay
//!   compensated, exact lengths), requantized once to 24 bits with TPDF
//!   dither.
//! - [`Player`] (module [`engine`]): commands in, [`Event`]s out, one decode
//!   thread. Seek, pause, gapless track changes at the same output format,
//!   a clean reopen when the format changes, resampling when the route
//!   asks for it, and events timed to the output clock.
//! - [`stems`]: stem playback. With a [`StemMix`] that leaves a stem out,
//!   a track plays on a [`Stage::Stems`] route: separated into drums, bass,
//!   vocals and other in a thread of its own as it plays, and the stems
//!   that are on summed (never bit-perfect).
//! - [`SpectrumTap`]: a lock-free ring of the most recent frames a
//!   mixer's device plays, tagged with their frame index, so a UI can
//!   analyse what is heard now.
//! - [`target`]: where the engine plays: a device ([`DeviceTarget`]),
//!   nothing ([`NullTarget`]) or memory ([`MemoryTarget`], for tests).
//!
//! # Use
//!
//! ```no_run
//! use jodel_playback::{DeviceTarget, Event, Player, PlayerOptions};
//!
//! let target = DeviceTarget::find("default")?;
//! let (player, events) = Player::new(Box::new(target), PlayerOptions::default());
//! player.load(vec!["track.flac".into()])?;
//! player.play(0)?;
//! for event in events {
//!     match event {
//!         Event::TrackStarted(start) => println!("{}", start.route.describe()),
//!         Event::QueueEnded => break,
//!         _ => {}
//!     }
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![warn(missing_docs)]

pub mod engine;
mod error;
mod headers;
pub mod rate;
pub mod resample;
pub mod source;
pub mod stem_file;
pub mod stems;
pub mod stretch;
mod tap;
pub mod target;
pub mod transition;
pub mod warp;

pub use engine::{
    Command, Event, JumpAt, PlayState, Player, PlayerOptions, Position, TrackStart, OPEN_BUSY_WAIT,
};
pub use error::Error;
pub use rate::{
    choose_route, choose_route_near, FileFacts, Headroom, NoRoute, Resample, Route, Settings,
    Stage, LOSSY_HEADROOM,
};
pub use resample::{FilterDesign, Requantizer, Resampler};
pub use source::{
    probe, probe_header, Codec, Container, Cover, CoverOrigin, Source, Tags, TrackInfo,
};
pub use stem_file::{StemFile, StemStore, StemWriter};
pub use stems::{SeparatorSource, Stem, StemMix, StereoFeed};
pub use tap::{SpectrumTap, TapCursor, TapFrame, TapMiss, TapWindow};
pub use target::{
    any_capabilities, Capture, Captures, DeviceTarget, MemoryTarget, NullTarget, OutputTarget,
};
pub use transition::{Envelope, Point, TempoMode, TransitionPlan, TransitionState};
