//! The platform backends. Each provides `BACKEND`, `devices`,
//! `mix_format` and `open`, and for capture `INPUT_BACKEND`,
//! `input_devices` and `open_input`; exactly one is compiled in. Every one
//! opens its device shared, in the system's mix format, beside every other
//! application.

mod priority;
pub(crate) use priority::AudioPriority;

#[cfg(any(windows, target_os = "linux"))]
mod thread;

#[cfg(windows)]
mod wasapi;
#[cfg(windows)]
mod wasapi_input;
#[cfg(windows)]
pub(crate) use wasapi::{devices, mix_format, open, BACKEND};
#[cfg(windows)]
pub(crate) use wasapi_input::{input_devices, open_input, INPUT_BACKEND};

#[cfg(target_os = "macos")]
mod coreaudio;
#[cfg(target_os = "macos")]
mod coreaudio_hal;
#[cfg(target_os = "macos")]
mod coreaudio_input;
#[cfg(target_os = "macos")]
pub(crate) use coreaudio::{devices, mix_format, open, BACKEND};
#[cfg(target_os = "macos")]
pub(crate) use coreaudio_input::{input_devices, open_input, INPUT_BACKEND};

#[cfg(target_os = "linux")]
mod alsa_input;
#[cfg(target_os = "linux")]
mod alsa_output;
#[cfg(target_os = "linux")]
pub(crate) use alsa_input::{input_devices, open_input, INPUT_BACKEND};
#[cfg(target_os = "linux")]
pub(crate) use alsa_output::{devices, mix_format, open, BACKEND};

#[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
mod none {
    use crate::device::Device;
    use crate::error::Error;
    use crate::format::StreamFormat;
    use crate::input::{Input, Mono};
    use crate::output::{Core, OpenRequest};

    pub(crate) const BACKEND: &str = "none";
    pub(crate) const INPUT_BACKEND: &str = "none";

    fn unsupported() -> Error {
        Error::Unsupported("no output backend for this platform".into())
    }

    pub(crate) fn devices() -> Result<Vec<Device>, Error> {
        Err(unsupported())
    }

    pub(crate) fn mix_format(_id: &str) -> Result<StreamFormat, Error> {
        Err(unsupported())
    }

    pub(crate) fn open(_id: &str, _request: &OpenRequest) -> Result<Core, Error> {
        Err(unsupported())
    }

    pub(crate) fn input_devices() -> Result<Vec<Device>, Error> {
        Err(unsupported())
    }

    pub(crate) fn open_input(_id: &str, _mono: Mono) -> Result<Input, Error> {
        Err(unsupported())
    }
}
#[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
pub(crate) use none::{
    devices, input_devices, mix_format, open, open_input, BACKEND, INPUT_BACKEND,
};
