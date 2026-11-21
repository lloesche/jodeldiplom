//! What codec headers say that the demuxers leave out: the bit rate of
//! lossy streams (an MPEG audio file's first frame and its Xing header, a
//! Vorbis stream's identification header) and the bit depth of ALAC.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Bytes searched for the first MPEG frame after the ID3v2 tag.
const SEARCH: usize = 64 * 1024;

/// kbit/s per bitrate index: MPEG-1 layers I, II, III, then MPEG-2/2.5
/// layer I and layers II/III.
const KBPS: [[u16; 15]; 5] = [
    [
        0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448,
    ],
    [
        0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384,
    ],
    [
        0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
    ],
    [
        0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256,
    ],
    [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
];

/// One MPEG audio frame header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Frame {
    /// MPEG-1 (else 2 or 2.5).
    mpeg1: bool,
    /// 1, 2 or 3.
    layer: u8,
    /// bit/s.
    bitrate: u32,
    rate: u32,
    mono: bool,
}

impl Frame {
    fn parse(h: [u8; 4]) -> Option<Frame> {
        if h[0] != 0xFF || h[1] & 0xE0 != 0xE0 {
            return None;
        }
        let version = (h[1] >> 3) & 3; // 0: 2.5, 2: 2, 3: 1
        let layer = match (h[1] >> 1) & 3 {
            1 => 3,
            2 => 2,
            3 => 1,
            _ => return None,
        };
        let index = usize::from(h[2] >> 4);
        let rate_index = usize::from((h[2] >> 2) & 3);
        if version == 1 || index == 0 || index == 15 || rate_index == 3 {
            return None;
        }
        let mpeg1 = version == 3;
        let table = match (mpeg1, layer) {
            (true, 1) => 0,
            (true, 2) => 1,
            (true, _) => 2,
            (false, 1) => 3,
            (false, _) => 4,
        };
        let base = [44_100, 48_000, 32_000][rate_index];
        let rate = match version {
            3 => base,
            2 => base / 2,
            _ => base / 4,
        };
        Some(Frame {
            mpeg1,
            layer,
            bitrate: u32::from(KBPS[table][index]) * 1000,
            rate,
            mono: h[3] >> 6 == 3,
        })
    }

    fn samples(self) -> u32 {
        match (self.layer, self.mpeg1) {
            (1, _) => 384,
            (2, _) | (3, true) => 1152,
            _ => 576,
        }
    }

    /// Where a Xing/Info header starts in a layer III frame.
    fn xing_offset(self) -> usize {
        4 + match (self.mpeg1, self.mono) {
            (true, false) => 32,
            (true, true) | (false, false) => 17,
            (false, true) => 9,
        }
    }
}

/// An MPEG audio file's bit rate: for a VBR file (a `Xing` header with a
/// frame and a byte count) the average, else the first frame's rate.
pub(crate) fn mpeg(path: &Path) -> Option<u32> {
    let mut file = File::open(path).ok()?;
    let mut id3 = [0u8; 10];
    file.read_exact(&mut id3).ok()?;
    let mut start = 0u64;
    if &id3[..3] == b"ID3" {
        let size = id3[6..10]
            .iter()
            .fold(0u64, |n, &b| (n << 7) | u64::from(b & 0x7F));
        let footer = if id3[5] & 0x10 != 0 { 10 } else { 0 };
        start = 10 + size + footer;
    }
    file.seek(SeekFrom::Start(start)).ok()?;
    let mut data = Vec::with_capacity(SEARCH);
    file.take(SEARCH as u64).read_to_end(&mut data).ok()?;
    mpeg_in(&data)
}

fn mpeg_in(data: &[u8]) -> Option<u32> {
    let (at, frame) = (0..data.len().saturating_sub(4)).find_map(|at| {
        let h = [data[at], data[at + 1], data[at + 2], data[at + 3]];
        Frame::parse(h).map(|f| (at, f))
    })?;
    if frame.layer == 3 {
        let x = at + frame.xing_offset();
        let be = |i: usize| {
            data.get(i..i + 4)
                .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
        };
        if data.get(x..x + 4) == Some(b"Xing") {
            let flags = be(x + 4)?;
            if flags & 3 == 3 {
                let frames = u64::from(be(x + 8)?);
                let bytes = u64::from(be(x + 12)?);
                let samples = frames * u64::from(frame.samples());
                if let Some(bps) = (bytes * 8 * u64::from(frame.rate)).checked_div(samples) {
                    return u32::try_from(bps).ok();
                }
            }
        }
    }
    Some(frame.bitrate)
}

/// A Vorbis stream's nominal bit rate, from the identification header that
/// starts `extra` (`None` when the encoder left it unset).
pub(crate) fn vorbis(extra: &[u8]) -> Option<u32> {
    if extra.len() < 28 || &extra[..7] != b"\x01vorbis" {
        return None;
    }
    let nominal = i32::from_le_bytes([extra[20], extra[21], extra[22], extra[23]]);
    u32::try_from(nominal).ok().filter(|&b| b > 0)
}

/// An ALAC stream's bits per sample, from its magic cookie (the
/// `ALACSpecificConfig`, perhaps behind `frma` and `alac` atom headers).
pub(crate) fn alac_bits(mut cookie: &[u8]) -> Option<u32> {
    for atom in [b"frma", b"alac"] {
        if cookie.get(4..8) == Some(&atom[..]) {
            cookie = cookie.get(12..)?;
        }
    }
    (cookie.len() >= 24)
        .then(|| u32::from(cookie[5]))
        .filter(|&bits| (1..=32).contains(&bits))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_headers_parse() {
        // MPEG-1 layer III, 128 kbit/s, 44.1 kHz, joint stereo.
        let f = Frame::parse([0xFF, 0xFB, 0x90, 0x44]).unwrap();
        assert_eq!(
            (f.mpeg1, f.layer, f.bitrate, f.rate),
            (true, 3, 128_000, 44_100)
        );
        assert_eq!(f.samples(), 1152);
        assert_eq!(f.xing_offset(), 36);
        // MPEG-2 layer III, 64 kbit/s, 22.05 kHz, mono.
        let f = Frame::parse([0xFF, 0xF3, 0x80, 0xC4]).unwrap();
        assert_eq!(
            (f.mpeg1, f.layer, f.bitrate, f.rate),
            (false, 3, 64_000, 22_050)
        );
        assert_eq!(f.samples(), 576);
        assert_eq!(f.xing_offset(), 13);
        // MPEG-1 layer II, 192 kbit/s, 48 kHz.
        let f = Frame::parse([0xFF, 0xFD, 0xA4, 0x00]).unwrap();
        assert_eq!((f.layer, f.bitrate, f.rate), (2, 192_000, 48_000));
        // No sync, the free and the bad bitrate index, a reserved rate.
        assert_eq!(Frame::parse([0x00, 0xFB, 0x90, 0x44]), None);
        assert_eq!(Frame::parse([0xFF, 0xFB, 0x00, 0x44]), None);
        assert_eq!(Frame::parse([0xFF, 0xFB, 0xF0, 0x44]), None);
        assert_eq!(Frame::parse([0xFF, 0xFB, 0x9C, 0x44]), None);
    }

    #[test]
    fn a_xing_header_gives_the_average() {
        let mut data = vec![0u8; 3];
        data.extend([0xFF, 0xFB, 0x90, 0x44]);
        data.extend([0u8; 32]);
        data.extend(b"Xing");
        data.extend(3u32.to_be_bytes());
        data.extend(100u32.to_be_bytes()); // frames
        data.extend(400_000u32.to_be_bytes()); // bytes
        data.resize(600, 0);
        // 400 000 bytes over 100 · 1152 samples at 44.1 kHz.
        assert_eq!(mpeg_in(&data), Some(1_225_000));
        // Without the counts: the frame's own rate.
        let at = 3 + 36 + 4;
        data[at..at + 4].copy_from_slice(&0u32.to_be_bytes());
        assert_eq!(mpeg_in(&data), Some(128_000));
    }

    #[test]
    fn alac_bit_depth() {
        let mut config = vec![0u8; 24];
        config[5] = 24;
        assert_eq!(alac_bits(&config), Some(24));
        let mut atom = vec![0, 0, 0, 36];
        atom.extend(b"alac");
        atom.extend([0; 4]);
        atom.extend(&config);
        assert_eq!(alac_bits(&atom), Some(24));
        config[5] = 0;
        assert_eq!(alac_bits(&config), None);
        assert_eq!(alac_bits(&config[..10]), None);
    }

    #[test]
    fn vorbis_nominal_rate() {
        let mut header = b"\x01vorbis".to_vec();
        header.extend(0u32.to_le_bytes());
        header.push(2);
        header.extend(44_100u32.to_le_bytes());
        header.extend(0i32.to_le_bytes());
        header.extend(160_000i32.to_le_bytes());
        header.extend(0i32.to_le_bytes());
        assert_eq!(vorbis(&header), Some(160_000));
        header[20..24].copy_from_slice(&(-1i32).to_le_bytes());
        assert_eq!(vorbis(&header), None);
        assert_eq!(vorbis(b"\x01vorbis"), None);
    }
}
