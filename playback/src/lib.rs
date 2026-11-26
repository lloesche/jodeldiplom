//! The playback engine between files and [`jodel_output`]: FLAC, WAV,
//! AIFF, CAF and ALAC files to device output, and AAC-LC (in
//! MP4), MP3 (MP2, MP1) and Ogg Vorbis files decoded to 24 bits, gapless.
//!
//! # Layers
//!
//! - [`source`]: opens files with symphonia, streams 24-bit PCM
//!   (requantizing lossy, floating-point and 32-bit samples once, with
//!   dither), seeks to an exact frame, and reads tags and the cover.
//! - [`rate`]: the pure rate policy, [`rate::choose_route`]: from the file,
//!   the one format its target plays at and the [`Settings`] to a
//!   [`Route`], or a [`NoRoute`] that says why nothing fits. A file at the
//!   target's rate plays as it is (lossless where it can be); at another
//!   the route gains a [`Resample`] step to the target's rate.
//! - [`resample`]: the [`Resampler`], a rational polyphase FIR in `f64`
//!   (Kaiser-windowed sinc, ≥ 160 dB stopband, linear phase, delay
//!   compensated, exact lengths), requantized once to 24 bits with TPDF
//!   dither.
//! - [`Player`] (module [`engine`]): commands in, [`Event`]s out, one decode
//!   thread. Seek, pause, gapless track changes, a clean reopen when the
//!   target's format changes, resampling when the route asks for it, and
//!   events timed to the output clock. On a desk with two
//!   heads ([`Player::on_desk`]) it plays transitions: the next track on
//!   the second head, linked to the playing one, both automated by the
//!   plan ([`transition`]).
//! - [`stems`]: a track's stems, separated into drums, bass, vocals and
//!   other in a thread of its own as it plays (or read where they are
//!   kept, [`stem_file`]): handed on beside the file as parts of an output
//!   of lanes ([`MakeStems`]), for a mixer to play on its strips (read
//!   along a time map for a head that is: a transition's tracks).
//! - [`desk`]: a player on a mixer's desk ([`StemDesk`]): its group strip,
//!   a strip for each stem inside it, and the lane input it plays on (two,
//!   for a player with two heads).
//! - [`SpectrumTap`]: a lock-free ring of the most recent frames a
//!   mixer's device plays, tagged with their frame index, so a UI can
//!   analyse what is heard now.
//! - [`target`]: where the engine plays: a device ([`DeviceTarget`], for a
//!   mixer to open once), nothing ([`NullTarget`]) or memory
//!   ([`MemoryTarget`], for tests).
//!
//! # Use
//!
//! The device is opened once, by a mixer; the player plays on one of its
//! inputs.
//!
//! ```no_run
//! use jodel_output::mixer::{Mixer, MixerOptions, Role};
//! use jodel_playback::{DeviceTarget, Event, Player, PlayerOptions};
//!
//! let device = DeviceTarget::find("default")?;
//! let mixer = Mixer::open(Box::new(device), MixerOptions::default());
//! let (input, _fader) = mixer.input(&mixer.strip("Player"), Role::Lead, 1.0);
//! let (player, events) = Player::new(Box::new(input), PlayerOptions::default());
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

pub mod desk;
pub mod engine;
mod error;
mod headers;
pub mod rate;
pub mod remix;
pub mod resample;
mod shift;
pub mod source;
pub mod stem_file;
pub mod stems;
pub mod stretch;
mod tap;
pub mod target;
pub mod transition;
pub mod warp;
mod warping;

pub use desk::StemDesk;
pub use engine::{
    Command, Event, Head, HeadFaders, JumpAt, MakeStems, PlayState, Player, PlayerOptions,
    Position, TrackStart,
};
pub use error::Error;
pub use rate::{
    choose_route, FileFacts, Headroom, NoRoute, Resample, Route, Settings, LOSSY_HEADROOM,
};
pub use resample::{FilterDesign, Requantizer, Resampler};
pub use shift::{ratio_of, KEY_RANGE};
pub use source::{
    probe, probe_header, Codec, Container, Cover, CoverOrigin, Source, Tags, TrackInfo,
};
pub use stem_file::{StemFile, StemStore, StemWriter};
pub use stems::{SeparatorSource, Stem, StereoFeed};
pub use tap::{SpectrumTap, TapCursor, TapFrame, TapMiss, TapWindow};
pub use target::{
    any_capabilities, Capture, Captures, DeviceTarget, MemoryTarget, NullTarget, OutputTarget,
};
pub use transition::{Envelope, Point, TempoMode, TransitionPlan, TransitionState, NEXT_LEAD};
