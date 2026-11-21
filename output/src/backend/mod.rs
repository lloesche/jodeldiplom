//! The platform backends. Each provides `BACKEND`, `devices`,
//! `capabilities` and `open`, and for capture `INPUT_BACKEND`,
//! `input_devices` and `open_input`; exactly one is compiled in. Core Audio also
//! provides `volume` and `set_volume` (the device's own volume control);
//! elsewhere the system's volume keys reach an exclusive stream's device,
//! and these say there is no control to use.

#[cfg(any(windows, target_os = "linux"))]
mod thread;

#[cfg(target_os = "macos")]
pub(crate) use coreaudio::{set_volume, volume};

#[cfg(not(target_os = "macos"))]
pub(crate) fn volume(_id: &str) -> Result<Option<crate::device::Volume>, crate::error::Error> {
    Ok(None)
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn set_volume(
    _id: &str,
    _level: f32,
) -> Result<crate::device::Volume, crate::error::Error> {
    Err(crate::error::Error::Unsupported(
        "the device's volume is the system's to set here".into(),
    ))
}

#[cfg(windows)]
mod wasapi;
/// Raises the thread that holds it for audio work (the mixer's), for as
/// long as it is held: MMCSS "Pro Audio" on Windows; elsewhere nothing.
#[cfg(windows)]
pub(crate) use wasapi::Mmcss as AudioPriority;
#[cfg(windows)]
mod wasapi_input;
#[cfg(windows)]
pub(crate) use wasapi::{capabilities, devices, open, BACKEND};
#[cfg(windows)]
pub(crate) use wasapi_input::{input_devices, open_input, INPUT_BACKEND};

#[cfg(target_os = "macos")]
mod coreaudio;
#[cfg(target_os = "macos")]
mod coreaudio_input;
#[cfg(target_os = "macos")]
pub(crate) use coreaudio::{capabilities, devices, open, BACKEND};
#[cfg(target_os = "macos")]
pub(crate) use coreaudio_input::{input_devices, open_input, INPUT_BACKEND};

#[cfg(target_os = "linux")]
mod alsa_hw;
#[cfg(target_os = "linux")]
mod alsa_input;
#[cfg(target_os = "linux")]
pub(crate) use alsa_hw::{capabilities, devices, open, BACKEND};
#[cfg(target_os = "linux")]
pub(crate) use alsa_input::{input_devices, open_input, INPUT_BACKEND};

#[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
mod none {
    use crate::caps::Capabilities;
    use crate::device::Device;
    use crate::error::Error;
    use crate::input::{Input, Mono};
    use crate::output::{Core, OpenRequest};

    pub(crate) const BACKEND: &str = "none";
    pub(crate) const INPUT_BACKEND: &str = "none";

    fn unsupported() -> Error {
        Error::Unsupported("no exclusive output backend for this platform".into())
    }

    pub(crate) fn devices() -> Result<Vec<Device>, Error> {
        Err(unsupported())
    }

    pub(crate) fn capabilities(_id: &str) -> Result<Capabilities, Error> {
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
    capabilities, devices, input_devices, open, open_input, BACKEND, INPUT_BACKEND,
};

/// Raises nothing here (MMCSS on Windows).
#[cfg(not(windows))]
pub(crate) struct AudioPriority;

#[cfg(not(windows))]
impl AudioPriority {
    pub(crate) fn enter() -> AudioPriority {
        AudioPriority
    }
}
