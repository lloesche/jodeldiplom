//! WAV files as bytes: written with the plain 44-byte header (16-bit or
//! 24-bit PCM, or 32-bit float), or with the extensible header a
//! recording of floats takes ([`float_header`]); and read back
//! ([`parse`]). Writing them to disk is the caller's, through `durable`.

/// How the samples are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    /// 16-bit integers.
    Pcm16,
    /// 24-bit integers, three bytes each.
    Pcm24,
    /// 32-bit IEEE floats.
    Float32,
}

impl Encoding {
    /// The bytes of a sample.
    pub fn bytes(self) -> u16 {
        match self {
            Encoding::Pcm16 => 2,
            Encoding::Pcm24 => 3,
            Encoding::Float32 => 4,
        }
    }

    /// The format tag: 1, PCM; 3, IEEE float.
    fn tag(self) -> u16 {
        match self {
            Encoding::Pcm16 | Encoding::Pcm24 => 1,
            Encoding::Float32 => 3,
        }
    }
}

/// A file's rate, channels and encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Format {
    /// Frames a second.
    pub rate: u32,
    /// Channels, interleaved.
    pub channels: u16,
    /// How each sample is stored.
    pub encoding: Encoding,
}

/// The bytes of [`header`].
pub const HEADER: usize = 44;

/// The plain header (a 16-byte `fmt ` chunk) of a file in `format` that
/// holds `samples` samples, every channel's counted.
pub fn header(format: Format, samples: usize) -> Vec<u8> {
    let bytes = format.encoding.bytes();
    let data = (samples * usize::from(bytes)) as u32;
    let block = bytes * format.channels;
    let mut h = Vec::with_capacity(HEADER);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&(36 + data).to_le_bytes());
    h.extend_from_slice(b"WAVEfmt ");
    h.extend_from_slice(&16u32.to_le_bytes());
    h.extend_from_slice(&format.encoding.tag().to_le_bytes());
    h.extend_from_slice(&format.channels.to_le_bytes());
    h.extend_from_slice(&format.rate.to_le_bytes());
    h.extend_from_slice(&(format.rate * u32::from(block)).to_le_bytes());
    h.extend_from_slice(&block.to_le_bytes());
    h.extend_from_slice(&(8 * bytes).to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&data.to_le_bytes());
    h
}

/// A whole file of 16-bit `samples`, `channels` interleaved, at `rate`.
pub fn pcm16(rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
    let format = Format {
        rate,
        channels,
        encoding: Encoding::Pcm16,
    };
    let mut out = header(format, samples.len());
    out.reserve(2 * samples.len());
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// A whole file of 24-bit `samples` (the low three bytes of each),
/// `channels` interleaved, at `rate`.
pub fn pcm24(rate: u32, channels: u16, samples: &[i32]) -> Vec<u8> {
    let format = Format {
        rate,
        channels,
        encoding: Encoding::Pcm24,
    };
    let mut out = header(format, samples.len());
    out.reserve(3 * samples.len());
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes()[..3]);
    }
    out
}

/// A whole file of 32-bit float `samples`, `channels` interleaved, at
/// `rate`.
pub fn float32(rate: u32, channels: u16, samples: &[f32]) -> Vec<u8> {
    let format = Format {
        rate,
        channels,
        encoding: Encoding::Float32,
    };
    let mut out = header(format, samples.len());
    out.reserve(4 * samples.len());
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// The bytes of [`float_header`].
pub const FLOAT_HEADER: usize = 80;

/// The header of `frames` of `channels` channels of 32-bit float at
/// `rate`, as players everywhere read one (macOS's Quick Look among them):
/// WAVE_FORMAT_EXTENSIBLE, saying which speaker each channel is for (the
/// mask `speakers`), and a `fact` chunk with the frame count, which a WAV
/// file of floats should have. The plain 16-byte header of float data on
/// three channels was refused by some.
pub fn float_header(channels: u16, speakers: u32, rate: u32, frames: u64) -> Vec<u8> {
    /// KSDATAFORMAT_SUBTYPE_IEEE_FLOAT.
    const FLOAT: [u8; 16] = [
        0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B,
        0x71,
    ];
    let block = u32::from(channels) * 4;
    let data = (frames * u64::from(block)).min(u64::from(u32::MAX - FLOAT_HEADER as u32)) as u32;
    let mut h = Vec::with_capacity(FLOAT_HEADER);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&(FLOAT_HEADER as u32 - 8 + data).to_le_bytes());
    h.extend_from_slice(b"WAVE");
    h.extend_from_slice(b"fmt ");
    h.extend_from_slice(&40u32.to_le_bytes());
    h.extend_from_slice(&0xFFFEu16.to_le_bytes());
    h.extend_from_slice(&channels.to_le_bytes());
    h.extend_from_slice(&rate.to_le_bytes());
    h.extend_from_slice(&(rate * block).to_le_bytes());
    h.extend_from_slice(&(block as u16).to_le_bytes());
    h.extend_from_slice(&32u16.to_le_bytes());
    h.extend_from_slice(&22u16.to_le_bytes());
    h.extend_from_slice(&32u16.to_le_bytes());
    h.extend_from_slice(&speakers.to_le_bytes());
    h.extend_from_slice(&FLOAT);
    h.extend_from_slice(b"fact");
    h.extend_from_slice(&4u32.to_le_bytes());
    h.extend_from_slice(&(frames.min(u64::from(u32::MAX)) as u32).to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&data.to_le_bytes());
    debug_assert_eq!(h.len(), FLOAT_HEADER);
    h
}

/// A WAV file read: its format and its samples' bytes.
#[derive(Clone, Copy, Debug)]
pub struct Wav<'a> {
    /// How the samples are stored.
    pub format: Format,
    /// The `data` chunk (cut short where the file is).
    pub data: &'a [u8],
}

impl Wav<'_> {
    /// The samples, if 16-bit.
    pub fn pcm16(&self) -> Option<Vec<i16>> {
        (self.format.encoding == Encoding::Pcm16).then(|| {
            self.data
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]))
                .collect()
        })
    }

    /// The samples, if 32-bit float.
    pub fn float32(&self) -> Option<Vec<f32>> {
        (self.format.encoding == Encoding::Float32).then(|| {
            self.data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        })
    }
}

/// `bytes` as a WAV file of 16-bit or 24-bit PCM or 32-bit float, plain or
/// extensible; `None` for anything else.
pub fn parse(bytes: &[u8]) -> Option<Wav<'_>> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    let u16_at = |i: usize| Some(u16::from_le_bytes(bytes.get(i..i + 2)?.try_into().ok()?));
    let u32_at = |i: usize| Some(u32::from_le_bytes(bytes.get(i..i + 4)?.try_into().ok()?));
    let mut format = None;
    let mut at = 12;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let len = u32_at(at + 4)? as usize;
        let body = at + 8;
        if id == b"fmt " {
            let mut tag = u16_at(body)?;
            if tag == 0xFFFE {
                // The sub-format's first two bytes are the plain tag.
                tag = u16_at(body + 24)?;
            }
            let encoding = match (tag, u16_at(body + 14)?) {
                (1, 16) => Encoding::Pcm16,
                (1, 24) => Encoding::Pcm24,
                (3, 32) => Encoding::Float32,
                _ => return None,
            };
            format = Some(Format {
                rate: u32_at(body + 4)?,
                channels: u16_at(body + 2)?,
                encoding,
            });
        } else if id == b"data" {
            let end = body.saturating_add(len).min(bytes.len());
            return Some(Wav {
                format: format?,
                data: &bytes[body..end],
            });
        }
        at = body.checked_add(len)?.checked_add(len & 1)?;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_reads_back_as_written() {
        let samples = [0i16, 1, -1, i16::MAX, i16::MIN, 1234];
        let bytes = pcm16(16_000, 2, &samples);
        assert_eq!(bytes.len(), HEADER + 2 * samples.len());
        let wav = parse(&bytes).unwrap();
        assert_eq!(
            wav.format,
            Format {
                rate: 16_000,
                channels: 2,
                encoding: Encoding::Pcm16
            }
        );
        assert_eq!(wav.pcm16().unwrap(), samples);
        let floats = [0.5f32, -0.25, 1.0];
        let wav = float32(48_000, 1, &floats);
        assert_eq!(parse(&wav).unwrap().float32().unwrap(), floats);
    }

    #[test]
    fn the_plain_header_is_the_standard_one() {
        let h = header(
            Format {
                rate: 44_100,
                channels: 2,
                encoding: Encoding::Pcm24,
            },
            10,
        );
        assert_eq!(h.len(), HEADER);
        assert_eq!(&h[..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(h[4..8].try_into().unwrap()), 36 + 30);
        assert_eq!(&h[8..16], b"WAVEfmt ");
        assert_eq!(u16::from_le_bytes([h[20], h[21]]), 1);
        assert_eq!(
            u32::from_le_bytes(h[28..32].try_into().unwrap()),
            44_100 * 6
        );
        assert_eq!(u16::from_le_bytes([h[32], h[33]]), 6);
        assert_eq!(u16::from_le_bytes([h[34], h[35]]), 24);
        assert_eq!(&h[36..40], b"data");
        assert_eq!(u32::from_le_bytes(h[40..44].try_into().unwrap()), 30);
        let mut bytes = pcm24(44_100, 2, &[0x12_3456, -2]);
        assert_eq!(&bytes[HEADER..], &[0x56, 0x34, 0x12, 0xfe, 0xff, 0xff]);
        bytes.truncate(HEADER + 3);
        assert_eq!(parse(&bytes).unwrap().data.len(), 3);
    }

    #[test]
    fn an_extensible_float_file_reads() {
        let mut bytes = float_header(3, 0x7, 48_000, 2);
        assert_eq!(bytes.len(), FLOAT_HEADER);
        for v in [0.0f32, 0.1, 0.2, 0.3, 0.4, 0.5] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let wav = parse(&bytes).unwrap();
        assert_eq!((wav.format.channels, wav.format.rate), (3, 48_000));
        assert_eq!(wav.float32().unwrap().len(), 6);
    }

    #[test]
    fn other_files_are_not_read() {
        assert!(parse(b"RIFF\0\0\0\0WAVE").is_none());
        assert!(parse(b"FORM\0\0\0\0AIFF").is_none());
    }
}
