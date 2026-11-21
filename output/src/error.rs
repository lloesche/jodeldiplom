//! The crate's error type.

use crate::pack::PackError;
use std::fmt;
use std::io;

/// Why an output call failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// This platform, or this request on it, has no exclusive output path.
    Unsupported(String),
    /// The system has no output device.
    NoDevice,
    /// No device has this id (or matches this name).
    DeviceNotFound(String),
    /// The device accepts none of the requested formats at this rate and
    /// channel count in exclusive mode.
    FormatNotAccepted {
        /// The requested rate.
        rate: u32,
        /// The requested channel count.
        channels: u16,
    },
    /// Another application holds the device (exclusively, or the sound
    /// server holds the `hw:` device).
    DeviceBusy,
    /// The user has disallowed exclusive use of this device.
    ExclusiveNotAllowed,
    /// The device went away (unplugged or disabled) while open.
    DeviceLost,
    /// A malformed argument: the message says which.
    InvalidArgument(String),
    /// A sample could not be packed in [`crate::PackMode::Exact`].
    Pack(PackError),
    /// The device or its thread did not answer in time; the text names the
    /// step.
    Timeout(&'static str),
    /// The output is closed.
    Closed,
    /// Writing a [`crate::FileOutput`] sink failed.
    Io {
        /// The I/O error kind.
        kind: io::ErrorKind,
        /// The I/O error text.
        message: String,
    },
    /// An OS audio call failed with a code this crate has no meaning for.
    Backend {
        /// The call that failed.
        call: &'static str,
        /// The OS result code (HRESULT, OSStatus or negative errno).
        code: i64,
        /// The OS text for it, where there is one.
        message: String,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unsupported(what) => write!(f, "not supported: {what}"),
            Error::NoDevice => f.write_str("no output device"),
            Error::DeviceNotFound(id) => write!(f, "no output device {id:?}"),
            Error::FormatNotAccepted { rate, channels } => write!(
                f,
                "the device accepts none of the requested formats at {rate} Hz, {channels} ch, exclusively"
            ),
            Error::DeviceBusy => f.write_str("the device is in use by another application"),
            Error::ExclusiveNotAllowed => f.write_str("exclusive use of the device is not allowed"),
            Error::DeviceLost => f.write_str("the device was removed"),
            Error::InvalidArgument(what) => write!(f, "invalid argument: {what}"),
            Error::Pack(e) => write!(f, "{e}"),
            Error::Timeout(step) => write!(f, "timed out: {step}"),
            Error::Closed => f.write_str("the output is closed"),
            Error::Io { message, .. } => write!(f, "write failed: {message}"),
            Error::Backend {
                call,
                code,
                message,
            } => {
                // HRESULTs and four-character codes read best in hex, errno values in decimal.
                if code.unsigned_abs() > 0xFFFF {
                    write!(f, "{call} failed: 0x{:08x}", *code as u32)?;
                } else {
                    write!(f, "{call} failed: {code}")?;
                }
                if !message.is_empty() {
                    write!(f, " ({message})")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for Error {}

impl From<PackError> for Error {
    fn from(e: PackError) -> Self {
        Error::Pack(e)
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io {
            kind: e.kind(),
            message: e.to_string(),
        }
    }
}
