//! Bit-perfect PCM output to audio devices.
//!
//! The crate opens an output device **exclusively** at exactly the stream's
//! rate, in an integer format the device takes natively, so the samples
//! reach the DAC unchanged: no OS mixer, resampler or software volume in
//! between. It is not tied to any decoder; anything that produces integer
//! PCM can use it.
//!
//! | OS | Backend |
//! |---|---|
//! | Windows | WASAPI exclusive mode, event-driven |
//! | macOS | Core Audio HAL: hog mode, mixing off, the device's physical integer format at the nominal rate, our own IOProc |
//! | Linux | ALSA `hw:` devices with resampling off and the exact rate |
//!
//! There is no shared-mode fallback: a device that cannot be opened
//! exclusively in a requested format is an error.
//!
//! # Use
//!
//! 1. [`devices`] lists the devices; [`capabilities`] probes what one takes
//!    exclusively (rates × formats × channels) without playing anything.
//! 2. [`open`] opens one with an [`OpenRequest`]: rate, channels and a
//!    format preference list. The first format the device accepts is used
//!    ([`choose_format`] is the rule).
//! 3. The [`Output`] handle takes the engine's samples with
//!    [`Output::write`]: `i32` words holding 24-bit values (see [`pack`]
//!    for the exact contract and what happens to values that do not fit).
//!    A backend thread drains the ring into the device.
//! 4. [`Output::position`] is the device clock in stream frames;
//!    [`Output::latency`] is how far behind the writer the DAC is.
//!
//! ```no_run
//! use jodel_output::{open, default_device, OpenRequest, Output};
//! use std::time::Duration;
//!
//! let device = default_device()?;
//! let mut out = open(&device.id, &OpenRequest::new(96_000, 2))?;
//! println!("playing {}", out.format());
//! let silence = vec![0i32; 2 * 9_600];
//! for _ in 0..10 {
//!     out.write(&silence, None)?;
//! }
//! out.drain(Some(Duration::from_secs(2)))?;
//! out.close()?;
//! # Ok::<(), jodel_output::Error>(())
//! ```
//!
//! # A microphone listened to
//!
//! [`input`] captures a microphone (shared mode, at its own rate) to hear
//! the singer, never to play: nothing of it reaches the output. Through
//! [`Output::set_tap`] it sees what the device plays, buffer by buffer and
//! lined up with what it captured at that moment, so [`aec`] can take the
//! music's echo out of it. A tap cannot change a buffer: the device gets
//! exactly the bytes written, a microphone open or not.
//!
//! # Several players on one device
//!
//! [`mixer`] owns the device and mixes the outputs opened on its inputs,
//! each on one of the user's strips: bit for bit when one is heard as
//! written, else summed with gains that are ramps on the mixer's clock. It
//! meters its strips and the device, and taps the device for the
//! microphone ([`mixer::Mixer::set_mic`]). A player's outputs can come in
//! one stream as [`lanes`]: main (the file, passed through as written) and
//! parts that make it, each on a strip of its own.
//!
//! # Tests without a device
//!
//! [`FileOutput`] has the same API and writes exactly the bytes a device
//! would receive to a file, memory or any writer, on a simulated clock
//! ([`SimClock`]). [`NullOutput`] discards them.
//!
//! # Examples
//!
//! `cargo run -p jodel-output --example devices` prints every device's
//! exclusive capability matrix; `cargo run -p jodel-output --example silence
//! -- default 48000 2` plays two seconds of digital silence through the
//! exclusive path.

#![warn(missing_docs)]

pub mod aec;
mod backend;
mod caps;
mod device;
mod dither;
mod error;
mod file;
mod format;
pub mod input;
pub mod isolate;
pub mod lanes;
pub mod mixer;
mod output;
pub mod pack;
mod ring;
mod shared;
mod tap;
mod timeline;

pub use caps::{candidates, choose_format, Capabilities, PROBE_RATES};
pub use device::{
    capabilities, default_device, devices, find_device, open, set_volume, volume, Device,
    DeviceOutput, Volume, BACKEND,
};
pub use dither::Requantizer;
pub use error::Error;
pub use file::{FileOutput, MemorySink, NullOutput, SimClock};
pub use format::{SampleFormat, StreamFormat};
pub use input::{
    default_input, find_input, input_devices, open_input, open_input_with, EchoRecording, Input,
    InputInfo, MicFeed, MicStatus, MicTap, Mono, VoiceBlock,
};
pub use output::{Clock, OpenRequest, Output, OutputTarget, PullSignal, Stats, StreamInfo};
pub use pack::{PackError, PackMode, PackReport};
pub use tap::{Tap, TAP_CHUNK};
