//! The desk's format ([`desk_format`]): what the device is opened at, once,
//! for as long as the mixer runs, and what every output on an input plays
//! at (each player resamples to it). Stereo, at the rate set for the device
//! ([`DESK_RATE`] unless set otherwise) where the device takes it, else
//! 44.1 kHz, else the lowest rate it takes (a device that takes only
//! 96 kHz runs at 96 kHz); in the first sample format of
//! [`SampleFormat::PREFERENCE`] it takes at that rate, the most bits.

use crate::caps::Capabilities;
use crate::format::{SampleFormat, StreamFormat};

/// The rate a device runs at unless it is set otherwise.
pub const DESK_RATE: u32 = 48_000;

/// The rate a device runs at where it does not take the one set.
pub const FALLBACK_RATE: u32 = 44_100;

/// The channels the desk mixes.
pub const DESK_CHANNELS: u16 = 2;

/// The rate a device that takes `rates` (in stereo, lowest first) runs at,
/// `rate` set for it: that, else [`FALLBACK_RATE`], else its lowest; `None`
/// where it takes none.
pub fn desk_rate(rates: &[u32], rate: u32) -> Option<u32> {
    [rate, FALLBACK_RATE]
        .into_iter()
        .find(|r| rates.contains(r))
        .or_else(|| rates.iter().copied().min())
}

/// The desk's format on a device that takes `caps`, `rate` set for it:
/// `None` where it takes no stereo stream at all.
pub fn desk_format(caps: &Capabilities, rate: u32) -> Option<StreamFormat> {
    let rate = desk_rate(&caps.rates(DESK_CHANNELS), rate)?;
    let sample = SampleFormat::PREFERENCE
        .into_iter()
        .find(|&f| caps.supports(&StreamFormat::new(rate, DESK_CHANNELS, f)))?;
    Some(StreamFormat::new(rate, DESK_CHANNELS, sample))
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

    /// The rate set where the device takes it, else 44.1 kHz, else its
    /// lowest; the most bits it takes there; none without stereo.
    #[test]
    fn the_rate_set_then_44_1_then_the_lowest() {
        use SampleFormat::{S24In32, S16, S32};
        let all = caps(&[
            (44_100, 2, S16),
            (48_000, 2, S16),
            (48_000, 2, S32),
            (96_000, 2, S24In32),
            (192_000, 1, S32),
        ]);
        let at = |rate| desk_format(&all, rate).map(|f| (f.rate, f.sample));
        assert_eq!(at(DESK_RATE), Some((48_000, S32)));
        assert_eq!(at(96_000), Some((96_000, S24In32)));
        // Not taken (in stereo): 44.1 kHz.
        assert_eq!(at(192_000), Some((44_100, S16)));
        assert_eq!(at(88_200), Some((44_100, S16)));
        // No 44.1 kHz either: the lowest it takes.
        let high = caps(&[(96_000, 2, S24In32), (192_000, 2, S24In32)]);
        let f = desk_format(&high, DESK_RATE).unwrap();
        assert_eq!((f.rate, f.channels), (96_000, 2));
        // No stereo at all: none.
        assert_eq!(desk_format(&caps(&[(48_000, 1, S16)]), DESK_RATE), None);
        assert_eq!(desk_format(&Capabilities::new(), DESK_RATE), None);
    }
}
