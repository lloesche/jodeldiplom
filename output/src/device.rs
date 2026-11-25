//! Listing the system's output devices, the format each takes, and opening them.

use crate::backend;
use crate::error::Error;
use crate::format::StreamFormat;
use crate::output::{delegate_output, Core, OpenRequest};

/// The backend this build uses: `"WASAPI (shared)"`, `"Core Audio
/// (shared)"`, `"ALSA default (shared)"`, or `"none"`.
pub const BACKEND: &str = backend::BACKEND;

/// An output device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Device {
    /// A stable id to store in a config and pass to [`open`]: the WASAPI
    /// endpoint id, the Core Audio device UID, or an ALSA PCM name
    /// (`default`).
    pub id: String,
    /// A name for people.
    pub name: String,
    /// Whether this is the system's default output (ALSA's `default`).
    pub is_default: bool,
}

/// The output devices that are present now.
pub fn devices() -> Result<Vec<Device>, Error> {
    backend::devices()
}

/// The default output device.
pub fn default_device() -> Result<Device, Error> {
    devices()?
        .into_iter()
        .find(|d| d.is_default)
        .ok_or(Error::NoDevice)
}

/// Finds a device by `query`: `"default"`, an exact id, an exact name, or a
/// case-insensitive part of exactly one device's name.
pub fn find_device(query: &str) -> Result<Device, Error> {
    if query.eq_ignore_ascii_case("default") {
        return default_device();
    }
    let all = devices()?;
    pick(all, query)
}

pub(crate) fn pick(all: Vec<Device>, query: &str) -> Result<Device, Error> {
    if let Some(d) = all.iter().find(|d| d.id == query || d.name == query) {
        return Ok(d.clone());
    }
    let q = query.to_lowercase();
    let mut matches = all
        .into_iter()
        .filter(|d| d.name.to_lowercase().contains(&q));
    match (matches.next(), matches.next()) {
        (Some(d), None) => Ok(d),
        (Some(_), Some(_)) => Err(Error::InvalidArgument(format!(
            "{query:?} matches more than one device"
        ))),
        _ => Err(Error::DeviceNotFound(query.into())),
    }
}

/// The one format the device `id` takes, opened shared: stereo, at the
/// rate the system runs it at (its sound settings, not this crate, set
/// it), in the sample format the system mixes in (32-bit float as a rule).
/// WASAPI's is the engine's mix format; Core Audio's the IO format of the
/// device's first output stream of two channels or more; ALSA's 32-bit
/// float at the sound server's rate. Nothing is played and nothing about
/// the device is changed. Ask it again after the device was lost: the
/// system may run it at another rate by then.
pub fn mix_format(id: &str) -> Result<StreamFormat, Error> {
    backend::mix_format(id)
}

/// Opens the device `id` shared, beside every other application, at its
/// [`mix_format`]: a request at another rate or channel count, or that
/// does not list its sample format, is [`Error::FormatNotAccepted`]. The
/// device does not start until the prefill is written. A device takes
/// plain streams only: more than one lane is [`Error::InvalidArgument`].
pub fn open(id: &str, request: &OpenRequest) -> Result<DeviceOutput, Error> {
    request.check_lanes(1)?;
    Ok(DeviceOutput {
        core: backend::open(id, request)?,
    })
}

/// A shared device stream (see [`crate::Output`]).
pub struct DeviceOutput {
    core: Core,
}

delegate_output!(DeviceOutput, core);

#[cfg(test)]
mod tests {
    use super::*;

    fn list() -> Vec<Device> {
        ["Speakers (USB DAC)", "Headphones (USB DAC)", "HDMI"]
            .iter()
            .enumerate()
            .map(|(i, n)| Device {
                id: format!("id{i}"),
                name: n.to_string(),
                is_default: i == 2,
            })
            .collect()
    }

    #[test]
    fn find_by_id_name_or_unique_part() {
        assert_eq!(pick(list(), "id1").unwrap().name, "Headphones (USB DAC)");
        assert_eq!(pick(list(), "HDMI").unwrap().id, "id2");
        assert_eq!(pick(list(), "speak").unwrap().id, "id0");
        assert!(matches!(
            pick(list(), "usb dac"),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            pick(list(), "nothing"),
            Err(Error::DeviceNotFound(_))
        ));
    }
}
