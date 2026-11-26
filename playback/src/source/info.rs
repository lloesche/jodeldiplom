//! What a file is: its container and codec, its tags and cover, and the
//! facts about its audio ([`TrackInfo`]).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// The file container.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Container {
    /// FLAC.
    Flac,
    /// RIFF WAVE (and RF64).
    Wav,
    /// AIFF and AIFF-C.
    Aiff,
    /// Core Audio Format.
    Caf,
    /// ISO base media (MP4, M4A).
    Mp4,
    /// An MPEG audio elementary stream (MP3, MP2, MP1).
    Mpeg,
    /// Ogg.
    Ogg,
}

impl Container {
    /// Its usual name: `FLAC`, `WAV`, `AIFF`, `CAF`, `MP4`, `MPEG`, `Ogg`.
    pub fn name(self) -> &'static str {
        match self {
            Container::Flac => "FLAC",
            Container::Wav => "WAV",
            Container::Aiff => "AIFF",
            Container::Caf => "CAF",
            Container::Mp4 => "MP4",
            Container::Mpeg => "MPEG",
            Container::Ogg => "Ogg",
        }
    }
}

/// The audio codec inside the container.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Codec {
    /// FLAC.
    Flac,
    /// Uncompressed PCM (integer or floating point).
    Pcm,
    /// Apple Lossless.
    Alac,
    /// MPEG-1/2 Audio Layer I.
    Mp1,
    /// MPEG-1/2 Audio Layer II.
    Mp2,
    /// MPEG-1/2 Audio Layer III.
    Mp3,
    /// Vorbis.
    Vorbis,
    /// AAC-LC (in MP4).
    Aac,
}

impl Codec {
    /// Its usual name: `FLAC`, `PCM`, `ALAC`, `MP1`, `MP2`, `MP3`, `Vorbis`,
    /// `AAC`.
    pub fn name(self) -> &'static str {
        match self {
            Codec::Flac => "FLAC",
            Codec::Pcm => "PCM",
            Codec::Alac => "ALAC",
            Codec::Mp1 => "MP1",
            Codec::Mp2 => "MP2",
            Codec::Mp3 => "MP3",
            Codec::Vorbis => "Vorbis",
            Codec::Aac => "AAC",
        }
    }

    /// Whether the codec is lossy: what it decodes to is not the signal
    /// that was encoded.
    pub fn is_lossy(self) -> bool {
        matches!(
            self,
            Codec::Mp1 | Codec::Mp2 | Codec::Mp3 | Codec::Vorbis | Codec::Aac
        )
    }
}

/// The tags a player shows, plus every tag as read.
///
/// The names come from Vorbis comments (FLAC, Ogg), ID3v2 frames (MP3,
/// AIFF, WAV), MP4 atoms and RIFF INFO chunks alike.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tags {
    /// The title.
    pub title: Option<String>,
    /// The artist.
    pub artist: Option<String>,
    /// The album.
    pub album: Option<String>,
    /// The track number, as written (it may be `4/12`).
    pub track_number: Option<String>,
    /// Every tag as (key, value): the newest metadata block first, each in
    /// file order. Keys keep their case.
    pub all: Vec<(String, String)>,
}

impl Tags {
    /// The value of the first tag named `key` (ASCII case ignored).
    pub fn get(&self, key: &str) -> Option<&str> {
        self.all
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(_, v)| v.as_str())
    }
}

/// Where a cover came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoverOrigin {
    /// A picture embedded in the file (a FLAC or Vorbis picture block, an
    /// ID3v2 `APIC` frame, an MP4 `covr` atom).
    Embedded,
    /// An image file in the track's folder.
    Folder(PathBuf),
}

/// A cover image, still encoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cover {
    /// The encoded image (JPEG, PNG, ...).
    pub data: Arc<[u8]>,
    /// Its MIME type (`image/jpeg`, `image/png`, ...), from the file or
    /// sniffed from the bytes; `application/octet-stream` when unknown.
    pub mime: String,
    /// Where it came from.
    pub origin: CoverOrigin,
}

/// What a file is: format, length, tags and cover.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrackInfo {
    /// The file.
    pub path: PathBuf,
    /// The container.
    pub container: Container,
    /// The codec.
    pub codec: Codec,
    /// Frames per second.
    pub rate: u32,
    /// Channels per frame.
    pub channels: u16,
    /// Bits per sample the file stores (16, 24, ...); 0 when unknown and
    /// for lossy codecs.
    pub bits: u32,
    /// Whether the samples are floating point (floating-point PCM, and
    /// every lossy codec: their decoders give floating-point samples).
    pub float: bool,
    /// For lossy codecs, the bit rate in bits per second, when known: an
    /// MP3 file's constant rate or the average of a VBR file (from its
    /// Xing header), a Vorbis stream's nominal rate, the average an AAC
    /// stream's descriptor states.
    pub bitrate: Option<u32>,
    /// The length in frames, when the file states it. For lossy codecs it
    /// leaves out the encoder delay and padding.
    pub frames: Option<u64>,
    /// The tags.
    pub tags: Tags,
    /// The cover: the embedded front cover (else the first embedded
    /// picture), else `cover.*`, `folder.*` or `front.*` in the folder.
    pub cover: Option<Cover>,
}

impl TrackInfo {
    /// The length, when known.
    pub fn duration(&self) -> Option<Duration> {
        let frames = self.frames?;
        if self.rate == 0 {
            return None;
        }
        Some(Duration::from_nanos(
            (u128::from(frames) * 1_000_000_000 / u128::from(self.rate)) as u64,
        ))
    }

    /// Whether every sample is an integer of at most 24 bits, so the
    /// engine's 24-bit samples hold it exactly.
    pub fn fits_24_bits(&self) -> bool {
        !self.float && self.bits <= 24
    }

    /// Whether the codec is lossy.
    pub fn is_lossy(&self) -> bool {
        self.codec.is_lossy()
    }

    /// The format's short name: the codec, or the container for PCM
    /// (`FLAC`, `ALAC`, `MP3`, `Vorbis`, `WAV`, `AIFF`, `CAF`, ...).
    pub fn format_name(&self) -> &'static str {
        match self.codec {
            Codec::Pcm => self.container.name(),
            codec => codec.name(),
        }
    }
}
