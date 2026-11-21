//! Packing engine samples into device formats.
//!
//! The engine's sample is an `i32` holding a signed 24-bit value,
//! right-aligned and sign-extended: full scale is ±2²³. A processing stage
//! may deliver values outside 24 bits unclipped in the word, so a packer
//! must decide what to do with a value that does not fit.
//!
//! A value *fits* a format when the format holds it unchanged:
//!
//! - every format with 24 or more valid bits holds every value in
//!   −2²³ ..= 2²³ − 1;
//! - [`SampleFormat::S16`] holds a value only when it is in that range and
//!   its low 8 bits are zero (16-bit content scaled to 24 bits).
//!
//! Every format has the same full scale, so a value outside 24 bits fits
//! none of them; `S32` has no headroom above the 24-bit range either.
//!
//! [`PackMode`] says what happens to a value that does not fit:
//!
//! - [`PackMode::Exact`]: the whole call fails with [`PackError`], naming
//!   the first such sample, and writes nothing.
//! - [`PackMode::Saturate`]: a value outside 24 bits is clamped to the
//!   nearest end of the range and counted in [`PackReport::clipped`]; for
//!   `S16` an in-range value keeps its top 16 bits (`v >> 8`, rounding
//!   toward −∞) and is counted in [`PackReport::truncated`].
//!
//! No mode wraps: a packed sample is always the value, or the clamped value,
//! in the format's layout.

use crate::format::SampleFormat;
use std::fmt;

/// The smallest 24-bit value.
pub const SAMPLE_MIN: i32 = -(1 << 23);
/// The largest 24-bit value.
pub const SAMPLE_MAX: i32 = (1 << 23) - 1;

/// What to do with a sample that a format cannot hold unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PackMode {
    /// Fail the call and write nothing.
    Exact,
    /// Clamp values outside 24 bits, drop low bits for 16-bit output, and
    /// count both.
    #[default]
    Saturate,
}

/// What [`pack`] changed in [`PackMode::Saturate`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PackReport {
    /// Samples outside 24 bits that were clamped.
    pub clipped: u64,
    /// In-range samples whose low 8 bits were dropped for `S16`.
    pub truncated: u64,
}

impl PackReport {
    /// Whether every sample was packed unchanged.
    pub fn is_exact(&self) -> bool {
        self.clipped == 0 && self.truncated == 0
    }

    /// Adds another report's counts.
    pub fn add(&mut self, other: PackReport) {
        self.clipped += other.clipped;
        self.truncated += other.truncated;
    }
}

/// Why [`pack`] failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackError {
    /// In [`PackMode::Exact`], the sample at `index` is outside 24 bits.
    OutOfRange {
        /// Position in the input slice.
        index: usize,
        /// The sample.
        value: i32,
    },
    /// In [`PackMode::Exact`], the sample at `index` has low bits the
    /// format cannot hold (`S16`).
    LowBitsLost {
        /// Position in the input slice.
        index: usize,
        /// The sample.
        value: i32,
    },
    /// The output buffer is not exactly `samples × bytes` long.
    BufferSize {
        /// The length needed.
        expected: usize,
        /// The length given.
        actual: usize,
    },
}

impl fmt::Display for PackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            PackError::OutOfRange { index, value } => {
                write!(f, "sample {index} ({value}) is outside 24 bits")
            }
            PackError::LowBitsLost { index, value } => {
                write!(
                    f,
                    "sample {index} ({value}) has bits below 16-bit precision"
                )
            }
            PackError::BufferSize { expected, actual } => {
                write!(f, "output buffer is {actual} bytes, expected {expected}")
            }
        }
    }
}

impl std::error::Error for PackError {}

/// Whether `format` holds `value` unchanged (see the module docs).
pub fn fits(value: i32, format: SampleFormat) -> bool {
    in_range(value) && (format != SampleFormat::S16 || value & 0xFF == 0)
}

fn in_range(value: i32) -> bool {
    (SAMPLE_MIN..=SAMPLE_MAX).contains(&value)
}

/// Checks that `format` holds every sample unchanged, as
/// [`PackMode::Exact`] requires, and names the first one it does not.
pub fn check(samples: &[i32], format: SampleFormat) -> Result<(), PackError> {
    match samples.iter().enumerate().find(|(_, &v)| !fits(v, format)) {
        None => Ok(()),
        Some((index, &value)) if in_range(value) => Err(PackError::LowBitsLost { index, value }),
        Some((index, &value)) => Err(PackError::OutOfRange { index, value }),
    }
}

/// Bytes [`pack`] writes for `samples` samples.
pub fn packed_len(format: SampleFormat, samples: usize) -> usize {
    samples * format.bytes()
}

/// Packs `samples` into `out` in `format`. `out` must be exactly
/// `samples.len() * format.bytes()` long.
///
/// In [`PackMode::Exact`] the input is checked first, so on error `out` is
/// untouched. See the module docs for what each mode does with a sample
/// that does not fit.
pub fn pack(
    samples: &[i32],
    format: SampleFormat,
    mode: PackMode,
    out: &mut [u8],
) -> Result<PackReport, PackError> {
    let expected = packed_len(format, samples.len());
    if out.len() != expected {
        return Err(PackError::BufferSize {
            expected,
            actual: out.len(),
        });
    }
    if mode == PackMode::Exact {
        check(samples, format)?;
    }
    let mut report = PackReport::default();
    let mut clamp = |v: i32| -> i32 {
        if in_range(v) {
            v
        } else {
            report.clipped += 1;
            v.clamp(SAMPLE_MIN, SAMPLE_MAX)
        }
    };
    match format {
        SampleFormat::S16 => {
            let mut truncated = 0;
            for (s, o) in samples.iter().zip(out.chunks_exact_mut(2)) {
                let in_range_value = in_range(*s);
                let v = clamp(*s);
                if in_range_value && v & 0xFF != 0 {
                    truncated += 1;
                }
                o.copy_from_slice(&((v >> 8) as i16).to_le_bytes());
            }
            report.truncated = truncated;
        }
        SampleFormat::S24Packed => {
            for (s, o) in samples.iter().zip(out.chunks_exact_mut(3)) {
                let b = clamp(*s).to_le_bytes();
                o.copy_from_slice(&b[..3]);
            }
        }
        SampleFormat::S24In32 | SampleFormat::S32 => {
            for (s, o) in samples.iter().zip(out.chunks_exact_mut(4)) {
                o.copy_from_slice(&(clamp(*s) << 8).to_le_bytes());
            }
        }
        SampleFormat::S24In32Low => {
            for (s, o) in samples.iter().zip(out.chunks_exact_mut(4)) {
                o.copy_from_slice(&clamp(*s).to_le_bytes());
            }
        }
    }
    Ok(report)
}

/// Reads one packed sample back as a 24-bit value: the inverse of [`pack`]
/// for every value that fits. `bytes` must be `format.bytes()` long.
///
/// `S16` comes back scaled to 24 bits (`s << 8`). For `S32` and `S24In32`
/// the low byte is dropped (`w >> 8`), and for `S24In32Low` the top byte is
/// ignored, as devices do.
pub fn unpack_sample(bytes: &[u8], format: SampleFormat) -> i32 {
    match format {
        SampleFormat::S16 => i32::from(i16::from_le_bytes([bytes[0], bytes[1]])) << 8,
        SampleFormat::S24Packed => i32::from_le_bytes([0, bytes[0], bytes[1], bytes[2]]) >> 8,
        SampleFormat::S24In32 | SampleFormat::S32 => {
            i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) >> 8
        }
        SampleFormat::S24In32Low => i32::from_le_bytes([0, bytes[0], bytes[1], bytes[2]]) >> 8,
    }
}

/// Reads packed bytes back as 24-bit values (see [`unpack_sample`]). A
/// trailing partial sample is ignored.
pub fn unpack(bytes: &[u8], format: SampleFormat) -> Vec<i32> {
    bytes
        .chunks_exact(format.bytes())
        .map(|b| unpack_sample(b, format))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EDGES: [i32; 9] = [
        0,
        1,
        -1,
        SAMPLE_MAX,
        SAMPLE_MIN,
        0x12_3456,
        -0x12_3456,
        0x7F_FF00,
        -0x80_0000 + 0x100,
    ];

    fn packed(
        samples: &[i32],
        format: SampleFormat,
        mode: PackMode,
    ) -> Result<(Vec<u8>, PackReport), PackError> {
        let mut out = vec![0xAA; packed_len(format, samples.len())];
        pack(samples, format, mode, &mut out).map(|r| (out, r))
    }

    #[test]
    fn layouts_of_one_value() {
        let v = [0x12_3456, -2];
        assert_eq!(
            packed(&v, SampleFormat::S24Packed, PackMode::Exact)
                .unwrap()
                .0,
            [0x56, 0x34, 0x12, 0xFE, 0xFF, 0xFF]
        );
        assert_eq!(
            packed(&v, SampleFormat::S24In32, PackMode::Exact)
                .unwrap()
                .0,
            [0x00, 0x56, 0x34, 0x12, 0x00, 0xFE, 0xFF, 0xFF]
        );
        assert_eq!(
            packed(&v, SampleFormat::S32, PackMode::Exact).unwrap().0,
            [0x00, 0x56, 0x34, 0x12, 0x00, 0xFE, 0xFF, 0xFF]
        );
        assert_eq!(
            packed(&v, SampleFormat::S24In32Low, PackMode::Exact)
                .unwrap()
                .0,
            [0x56, 0x34, 0x12, 0x00, 0xFE, 0xFF, 0xFF, 0xFF]
        );
        let s16 = [0x12_3400, -0x100];
        assert_eq!(
            packed(&s16, SampleFormat::S16, PackMode::Exact).unwrap().0,
            [0x34, 0x12, 0xFF, 0xFF]
        );
    }

    #[test]
    fn every_24_bit_format_round_trips_the_edges() {
        for format in SampleFormat::ALL.into_iter().filter(|f| f.holds_24_bits()) {
            for mode in [PackMode::Exact, PackMode::Saturate] {
                let (bytes, report) = packed(&EDGES, format, mode).unwrap();
                assert!(report.is_exact(), "{format}");
                assert_eq!(unpack(&bytes, format), EDGES, "{format} {mode:?}");
            }
        }
    }

    #[test]
    fn every_24_bit_format_round_trips_a_sweep() {
        let sweep: Vec<i32> = (SAMPLE_MIN..=SAMPLE_MAX)
            .step_by(997)
            .chain([SAMPLE_MAX])
            .collect();
        for format in SampleFormat::ALL.into_iter().filter(|f| f.holds_24_bits()) {
            let (bytes, report) = packed(&sweep, format, PackMode::Exact).unwrap();
            assert!(report.is_exact());
            assert_eq!(unpack(&bytes, format), sweep, "{format}");
        }
    }

    #[test]
    fn s16_round_trips_16_bit_content() {
        let content: Vec<i32> = (i16::MIN..=i16::MAX).map(|s| i32::from(s) << 8).collect();
        let (bytes, report) = packed(&content, SampleFormat::S16, PackMode::Exact).unwrap();
        assert!(report.is_exact());
        assert_eq!(unpack(&bytes, SampleFormat::S16), content);
    }

    #[test]
    fn exact_mode_rejects_and_writes_nothing() {
        for format in SampleFormat::ALL {
            let input = [0, SAMPLE_MAX + 1, 5];
            let mut out = vec![0xAA; packed_len(format, 3)];
            assert_eq!(
                pack(&input, format, PackMode::Exact, &mut out),
                Err(PackError::OutOfRange {
                    index: 1,
                    value: SAMPLE_MAX + 1
                })
            );
            assert!(out.iter().all(|&b| b == 0xAA));
            let input = [SAMPLE_MIN - 1];
            let mut out = vec![0; packed_len(format, 1)];
            assert_eq!(
                pack(&input, format, PackMode::Exact, &mut out),
                Err(PackError::OutOfRange {
                    index: 0,
                    value: SAMPLE_MIN - 1
                })
            );
        }
        let mut out = [0; 4];
        assert_eq!(
            pack(
                &[0x100, 0x101],
                SampleFormat::S16,
                PackMode::Exact,
                &mut out
            ),
            Err(PackError::LowBitsLost {
                index: 1,
                value: 0x101
            })
        );
    }

    #[test]
    fn saturate_clamps_and_counts_never_wraps() {
        let input = [
            i32::MAX,
            i32::MIN,
            SAMPLE_MAX + 1,
            SAMPLE_MIN - 1,
            0x0100_0000,
            7,
        ];
        for format in SampleFormat::ALL {
            let (bytes, report) = packed(&input, format, PackMode::Saturate).unwrap();
            assert_eq!(report.clipped, 5, "{format}");
            let back = unpack(&bytes, format);
            let top = if format == SampleFormat::S16 {
                0x7F_FF00
            } else {
                SAMPLE_MAX
            };
            assert_eq!(
                back[..5],
                [top, SAMPLE_MIN, top, SAMPLE_MIN, top],
                "{format}"
            );
            // Positive overflow never comes back negative and vice versa.
            assert!(back[0] > 0 && back[1] < 0);
        }
    }

    #[test]
    fn saturate_truncates_s16_toward_minus_infinity() {
        let input = [0x1FF, -1, -0x101, 0x7F_FFFF];
        let (bytes, report) = packed(&input, SampleFormat::S16, PackMode::Saturate).unwrap();
        assert_eq!(
            report,
            PackReport {
                clipped: 0,
                truncated: 4
            }
        );
        assert_eq!(
            unpack(&bytes, SampleFormat::S16),
            [0x100, -0x100, -0x200, 0x7F_FF00]
        );
        // 16-bit content is not counted.
        let (_, report) =
            packed(&[0x100, SAMPLE_MIN], SampleFormat::S16, PackMode::Saturate).unwrap();
        assert!(report.is_exact());
    }

    #[test]
    fn buffer_size_is_checked() {
        let mut out = [0; 5];
        assert_eq!(
            pack(
                &[1, 2],
                SampleFormat::S24Packed,
                PackMode::Saturate,
                &mut out
            ),
            Err(PackError::BufferSize {
                expected: 6,
                actual: 5
            })
        );
    }

    #[test]
    fn fits_matches_the_rules() {
        assert!(fits(SAMPLE_MAX, SampleFormat::S24Packed));
        assert!(!fits(SAMPLE_MAX + 1, SampleFormat::S32));
        assert!(fits(-0x100, SampleFormat::S16));
        assert!(!fits(0x80, SampleFormat::S16));
    }

    #[test]
    fn unpack_ignores_the_unused_byte() {
        assert_eq!(
            unpack_sample(&[0x56, 0x34, 0x92, 0x00], SampleFormat::S24In32Low),
            -0x6D_CBAA
        );
        assert_eq!(
            unpack_sample(&[0xFF, 0x56, 0x34, 0x12], SampleFormat::S24In32),
            0x12_3456
        );
    }
}
