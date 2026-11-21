//! The crate's error type.

use std::fmt;
use std::path::PathBuf;

use crate::rate::NoRoute;

/// Why something in the playback chain failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A file could not be opened, read, decoded or seeked.
    Source {
        /// The file.
        path: PathBuf,
        /// What went wrong.
        message: String,
    },
    /// No output format fits this file on this device (see [`NoRoute`]).
    NoRoute(NoRoute),
    /// The output failed: opening, probing or writing.
    Output(jodel_output::Error),
    /// The engine thread has stopped.
    Stopped,
}

impl Error {
    /// Whether playback can go on with the next track: everything except a
    /// failed output and a stopped engine.
    pub fn is_skippable(&self) -> bool {
        matches!(self, Error::Source { .. } | Error::NoRoute(_))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Source { path, message } => write!(f, "{}: {message}", path.display()),
            Error::NoRoute(no_route) => write!(f, "{no_route}"),
            Error::Output(e) => write!(f, "output: {e}"),
            Error::Stopped => f.write_str("the playback engine has stopped"),
        }
    }
}

impl std::error::Error for Error {}

impl From<jodel_output::Error> for Error {
    fn from(e: jodel_output::Error) -> Self {
        Error::Output(e)
    }
}

impl From<NoRoute> for Error {
    fn from(e: NoRoute) -> Self {
        Error::NoRoute(e)
    }
}
