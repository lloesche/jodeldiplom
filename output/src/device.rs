//! Listing, probing and opening the system's output devices.

use crate::backend;
use crate::caps::Capabilities;
use crate::error::Error;
use crate::output::{delegate_output, Core, OpenRequest};

/// The exclusive backend this build uses: `"WASAPI exclusive"`,
/// `"Core Audio HAL (hog mode)"`, `"ALSA hw"`, or `"none"`.
pub const BACKEND: &str = backend::BACKEND;

/// An output device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Device {
    /// A stable id to store in a config and pass to [`open`]: the WASAPI
    /// endpoint id, the Core Audio device UID, or an ALSA
    /// `hw:CARD=<id>,DEV=<n>` name.
    pub id: String,
    /// A name for people.
    pub name: String,
    /// Whether this is the system's default output. ALSA has no default
    /// `hw:` device; there the first card's first playback device is
    /// marked.
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

/// What the device `id` accepts exclusively over [`crate::PROBE_RATES`]
/// and every [`crate::SampleFormat`], in stereo and in the device's own
/// channel layout. Nothing is played and the device is not started:
/// WASAPI and ALSA answer format queries without starting a stream, and
/// Core Audio lists its physical formats.
pub fn capabilities(id: &str) -> Result<Capabilities, Error> {
    backend::capabilities(id)
}

/// Opens the device `id` exclusively at exactly `request.rate` and
/// `request.channels`, in the first of `request.formats` it accepts. The
/// device does not start until the prefill is written. A device takes plain
/// streams only: more than one lane is [`Error::InvalidArgument`].
pub fn open(id: &str, request: &OpenRequest) -> Result<DeviceOutput, Error> {
    request.check_lanes(1)?;
    Ok(DeviceOutput {
        core: backend::open(id, request)?,
    })
}

/// A device's own output volume control: the one the system's volume
/// slider moves. The device applies it after it has received the samples,
/// so the stream stays bit-perfect; at `level` 1.0 the control passes the
/// signal unchanged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Volume {
    /// From 0.0 (silent) to 1.0 (full), as the system's slider shows it.
    pub level: f32,
    /// The gain at that level in dB (0 at full), when the device says.
    pub db: Option<f32>,
    /// Whether the output is muted, when the device has a mute control.
    pub muted: Option<bool>,
}

/// The volume control of the device `id`, while anything may hold it
/// (including an exclusive stream of this process). `None`: the device
/// has none this process can set (on Windows and Linux the system's
/// volume keys reach the device of an exclusive stream themselves, and
/// this always says `None`).
///
/// macOS needs it: the system's volume keys do not reach a device that a
/// process holds in hog mode, so the level stays where it was when the
/// stream opened.
pub fn volume(id: &str) -> Result<Option<Volume>, Error> {
    backend::volume(id)
}

/// Sets the volume control of the device `id` to `level` (clamped to
/// 0.0–1.0; every channel to the same level where there is no master),
/// unmutes it when `level` is above 0, and returns the control as it is
/// then. An error where [`volume`] says `None`.
pub fn set_volume(id: &str, level: f32) -> Result<Volume, Error> {
    backend::set_volume(id, level.clamp(0.0, 1.0))
}

/// An exclusive device stream (see [`crate::Output`]).
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
