//! PCM output to audio devices, shared with every other application.
//!
//! The crate opens an output device **shared**, through the system's
//! mixer, in the system's mix format: the rate the system runs the device
//! at (its sound settings choose it, not this crate) and, as a rule,
//! 32-bit float. Other applications play beside it, and a call's screen
//! share hears it. It is not tied to any decoder; anything that produces
//! integer PCM can use it.
//!
//! | OS | Backend |
//! |---|---|
//! | Windows | WASAPI shared mode, event-driven, the engine's mix format |
//! | macOS | Core Audio HAL, shared: our own IOProc in the device's IO format at its nominal rate, nothing about the device changed |
//! | Linux | ALSA `default` (the sound server, PipeWire or PulseAudio, else `dmix`), and the servers' own `pipewire` and `pulse` |
//!
//! There is no exclusive mode: nothing holds a device for itself.
//!
//! # Use
//!
//! 1. [`devices`] lists the devices; [`mix_format`] reads the one format
//!    one takes (the system's rate, stereo, as a rule 32-bit float)
//!    without playing anything.
//! 2. [`open`] opens one with an [`OpenRequest`] at that format: its rate,
//!    channels and a format preference list that holds its sample format.
//! 3. The [`Output`] handle takes the engine's samples with
//!    [`Output::write`]: `i32` words holding 24-bit values (see [`pack`]
//!    for the exact contract and what happens to values that do not fit).
//!    A backend thread drains the ring into the device.
//! 4. [`Output::position`] is the device clock in stream frames;
//!    [`Output::latency`] is how far behind the writer the DAC is.
//!
//! ```no_run
//! use jodel_output::{default_device, mix_format, open, OpenRequest, Output};
//! use std::time::Duration;
//!
//! let device = default_device()?;
//! let f = mix_format(&device.id)?;
//! let mut out = open(&device.id, &OpenRequest::new(f.rate, f.channels))?;
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
//! parts that make it, each on a strip of its own. Its effects input
//! ([`mixer::Mixer::effects`]) plays short sounds (a game's) on a strip of
//! their own, each started on a frame of an input's stream.
//!
//! # Tests without a device
//!
//! [`FileOutput`] has the same API and writes exactly the bytes a device
//! would receive to a file, memory or any writer, on a simulated clock
//! ([`SimClock`]). [`NullOutput`] discards them.
//!
//! # Examples
//!
//! `cargo run -p jodel-output --example devices` prints the mix format
//! every device takes; `cargo run -p jodel-output --example silence --
//! default mix 2` plays two seconds of digital silence at the system's rate,
//! and `--example tone` three seconds of 440 Hz with how the device's clock
//! runs.

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

pub use caps::{candidates, choose_format, Capabilities};
pub use device::{
    default_device, devices, find_device, mix_format, open, Device, DeviceOutput, BACKEND,
};
pub use dither::Requantizer;
pub use error::Error;
pub use file::{FileOutput, MemorySink, NullOutput, SimClock};
pub use format::{SampleFormat, StreamFormat};
pub use input::{
    default_input, find_input, input_devices, open_input, open_input_with, EchoRecording, Input,
    InputInfo, MicFeed, MicStatus, MicTap, Mono, VoiceBlock,
};
pub use output::{Clock, OpenRequest, Output, OutputTarget, Stats, StreamInfo};
pub use pack::{PackError, PackMode, PackReport};
pub use tap::{Tap, TAP_CHUNK};
