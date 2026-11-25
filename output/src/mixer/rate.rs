//! The desk's format ([`desk_format`]): what the device is opened at, once,
//! for as long as the mixer runs, and what every output on an input plays
//! at (each player resamples to it). A device offers one format, its mix
//! format (see [`crate::mix_format`]): the system's rate, stereo; a
//! stand-in offers one rate too. The desk runs at it, in the first sample
//! format of [`SampleFormat::PREFERENCE`] the target takes in stereo.

use crate::caps::Capabilities;
use crate::format::{SampleFormat, StreamFormat};

/// The channels the desk mixes.
pub const DESK_CHANNELS: u16 = 2;

/// The desk's format on a target that takes `caps`: `None` where it takes
/// no stereo stream at all.
pub fn desk_format(caps: &Capabilities) -> Option<StreamFormat> {
    SampleFormat::PREFERENCE.into_iter().find_map(|sample| {
        caps.iter()
            .copied()
            .find(|f| f.channels == DESK_CHANNELS && f.sample == sample)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(formats: &[(u32, u16, SampleFormat)]) -> Capabilities {
        formats
            .iter()
            .map(|&(r, c, f)| StreamFormat::new(r, c, f))
            .collect()
    }

    /// The target's rate, whatever it is; the most bits it takes there in
    /// stereo; none without stereo.
    #[test]
    fn the_targets_one_rate_in_its_best_format() {
        use SampleFormat::{F32, S16, S32};
        let at = |c: &Capabilities| desk_format(c).map(|f| (f.rate, f.sample));
        assert_eq!(at(&caps(&[(44_100, 2, F32)])), Some((44_100, F32)));
        assert_eq!(at(&caps(&[(96_000, 2, F32)])), Some((96_000, F32)));
        let several = caps(&[(48_000, 2, S16), (48_000, 2, S32), (48_000, 1, F32)]);
        assert_eq!(at(&several), Some((48_000, S32)));
        assert_eq!(desk_format(&caps(&[(48_000, 1, S16)])), None);
        assert_eq!(desk_format(&Capabilities::new()), None);
    }
}
