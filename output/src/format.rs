//! Sample containers and stream formats.

use std::fmt;
use std::time::Duration;

/// How one sample is laid out in device memory. Every format is signed
/// little-endian integer PCM, interleaved by frame.
///
/// The engine's samples are 24-bit values in `i32` words (see
/// [`crate::pack`]); the table says how each format holds one.
///
/// | Format | Bytes | Valid bits | Layout of a 24-bit value *v* |
/// |---|---|---|---|
/// | [`S16`](Self::S16) | 2 | 16 | the top 16 bits, `v >> 8` |
/// | [`S24Packed`](Self::S24Packed) | 3 | 24 | the three low bytes of *v* |
/// | [`S24In32`](Self::S24In32) | 4 | 24 | MSB-aligned, `v << 8` (WASAPI and Core Audio "aligned high") |
/// | [`S24In32Low`](Self::S24In32Low) | 4 | 24 | low-aligned, *v* sign-extended (ALSA `S24_LE`) |
/// | [`S32`](Self::S32) | 4 | 32 | `v << 8` |
///
/// `S24In32` and `S32` carry the same bytes for 24-bit content; they differ
/// in what the device is told about the valid bits, and some devices accept
/// only one of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SampleFormat {
    /// Signed 16-bit.
    S16,
    /// Signed 24-bit in three bytes.
    S24Packed,
    /// Signed 24-bit in the upper three bytes of a 32-bit word.
    S24In32,
    /// Signed 24-bit in the lower three bytes of a 32-bit word.
    S24In32Low,
    /// Signed 32-bit.
    S32,
}

impl SampleFormat {
    /// Every format, narrowest first.
    pub const ALL: [SampleFormat; 5] = [
        SampleFormat::S16,
        SampleFormat::S24Packed,
        SampleFormat::S24In32,
        SampleFormat::S24In32Low,
        SampleFormat::S32,
    ];

    /// The default preference order for opening a device: every format that
    /// holds 24 bits losslessly first, 16-bit last.
    pub const PREFERENCE: [SampleFormat; 5] = [
        SampleFormat::S24In32,
        SampleFormat::S32,
        SampleFormat::S24In32Low,
        SampleFormat::S24Packed,
        SampleFormat::S16,
    ];

    /// Bytes per sample in device memory.
    pub const fn bytes(self) -> usize {
        match self {
            SampleFormat::S16 => 2,
            SampleFormat::S24Packed => 3,
            SampleFormat::S24In32 | SampleFormat::S24In32Low | SampleFormat::S32 => 4,
        }
    }

    /// The precision the device is told about.
    pub const fn valid_bits(self) -> u32 {
        match self {
            SampleFormat::S16 => 16,
            SampleFormat::S24Packed | SampleFormat::S24In32 | SampleFormat::S24In32Low => 24,
            SampleFormat::S32 => 32,
        }
    }

    /// Whether every 24-bit value packs into this format unchanged.
    pub const fn holds_24_bits(self) -> bool {
        self.valid_bits() >= 24
    }

    /// A short name for tables and logs.
    pub const fn name(self) -> &'static str {
        match self {
            SampleFormat::S16 => "S16",
            SampleFormat::S24Packed => "S24_3",
            SampleFormat::S24In32 => "S24in32",
            SampleFormat::S24In32Low => "S24in32lo",
            SampleFormat::S32 => "S32",
        }
    }
}

impl fmt::Display for SampleFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A rate, a channel count and a sample format: what a device stream runs at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StreamFormat {
    /// Frames per second.
    pub rate: u32,
    /// Samples per frame.
    pub channels: u16,
    /// The container of each sample.
    pub sample: SampleFormat,
}

impl StreamFormat {
    /// A stream format.
    pub const fn new(rate: u32, channels: u16, sample: SampleFormat) -> Self {
        StreamFormat {
            rate,
            channels,
            sample,
        }
    }

    /// Bytes per frame in device memory.
    pub const fn frame_bytes(&self) -> usize {
        self.sample.bytes() * self.channels as usize
    }

    /// How long `frames` frames play.
    pub fn frames_to_duration(&self, frames: u64) -> Duration {
        if self.rate == 0 {
            return Duration::ZERO;
        }
        let nanos = u128::from(frames) * 1_000_000_000 / u128::from(self.rate);
        Duration::from_nanos(nanos.min(u128::from(u64::MAX)) as u64)
    }

    /// How many frames play in `duration`, rounded down.
    pub fn duration_to_frames(&self, duration: Duration) -> u64 {
        let frames = duration.as_nanos() * u128::from(self.rate) / 1_000_000_000;
        frames.min(u128::from(u64::MAX)) as u64
    }
}

impl fmt::Display for StreamFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} Hz {} ch {}", self.rate, self.channels, self.sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_and_bits() {
        let table: Vec<(usize, u32, bool)> = SampleFormat::ALL
            .iter()
            .map(|f| (f.bytes(), f.valid_bits(), f.holds_24_bits()))
            .collect();
        assert_eq!(
            table,
            [
                (2, 16, false),
                (3, 24, true),
                (4, 24, true),
                (4, 24, true),
                (4, 32, true)
            ]
        );
    }

    #[test]
    fn preference_lists_every_format_once() {
        let mut sorted = SampleFormat::PREFERENCE;
        sorted.sort();
        assert_eq!(sorted, SampleFormat::ALL);
        assert_eq!(SampleFormat::PREFERENCE[4], SampleFormat::S16);
    }

    #[test]
    fn frame_arithmetic() {
        let f = StreamFormat::new(192_000, 2, SampleFormat::S24Packed);
        assert_eq!(f.frame_bytes(), 6);
        assert_eq!(f.frames_to_duration(96_000), Duration::from_millis(500));
        assert_eq!(f.duration_to_frames(Duration::from_millis(10)), 1920);
        assert_eq!(f.to_string(), "192000 Hz 2 ch S24_3");
    }
}
