//! Reading audio files: PCM, sample-accurate seeking, tags and the cover.
//!
//! [`Source::open`] finds the container (FLAC, WAV, AIFF, CAF, MP4/M4A,
//! MPEG audio, Ogg) from the file's first bytes and reads its tags and
//! cover.
//!
//! [`Source::read`] returns the PCM packet by packet as interleaved signed
//! 24-bit values right-aligned in `i32` words, the output's sample
//! contract:
//!
//! - integer sources of at most 24 bits pass exactly (16-bit samples are
//!   shifted up, 24-bit ones are kept as they are);
//! - everything else (floating-point PCM, integer PCM above 24 bits, and
//!   every lossy decoder, which gives floating-point samples) is
//!   requantized to 24 bits once, with TPDF dither
//!   ([`crate::resample::Requantizer`]), counting the samples that clip
//!   ([`Source::clipped`]). A packet whose samples all lie on the 24-bit
//!   grid already (a 32-bit file that holds 24-bit audio, digital silence)
//!   passes exactly, without dither. The dither of each packet is seeded
//!   from the packet's first frame index, so a track read from a seek point
//!   gives the same samples as a track read from its start.
//!
//! [`Source::read_unquantized`] gives the same samples before
//! requantization, for the resampler, so that a resampled lossy track is
//! quantized once.
//!
//! Lossy tracks play gapless: the encoder delay and padding the container
//! states (the LAME/Xing header of an MP3 file, the granule positions of an
//! Ogg stream) are trimmed by the decoder, and those of AAC in an MP4 file
//! (its `iTunSMPB` tag, else its edit list) here ([`aac`]), so a
//! track starts with its first sample, ends with its last and has exactly
//! the length the header says. [`Source::seek`] moves to an exact frame on
//! every format.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use symphonia::core::audio::sample::SampleFormat as SymphoniaSampleFormat;
use symphonia::core::codecs::audio::well_known as codec_ids;
use symphonia::core::codecs::audio::{AudioCodecParameters, AudioDecoder, AudioDecoderOptions};
use symphonia::core::codecs::CodecParameters;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::well_known as format_ids;
use symphonia::core::formats::{FormatOptions, FormatReader, SeekMode, SeekTo, Track};
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::units::{TimeBase, Timestamp};

use crate::error::Error;
use crate::headers;
use crate::resample::Requantizer;

use meta::{folder_cover, read_metadata};

mod aac;
mod meta;

/// Frames a lossy track is decoded ahead of a seek target, so that the
/// decoder's state (an MP3 bit reservoir and overlap, a Vorbis window) is
/// built up again before the target. What is decoded before the target is
/// dropped.
pub const SEEK_PREROLL: u64 = 8_192;

/// One 24-bit LSB in the `[-1, 1)` floating-point scale.
pub(crate) const SCALE_24: f64 = 8_388_608.0;

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

/// An open file, read packet by packet.
pub struct Source {
    info: Arc<TrackInfo>,
    reader: Box<dyn FormatReader>,
    decoder: Box<dyn AudioDecoder>,
    /// What the decoder is made from (again at every seek).
    params: AudioCodecParameters,
    track_id: u32,
    /// The track's timestamps per frame, when they are not frames.
    time_base: Option<(u64, u64)>,
    /// Decoded frames before the track's first frame (the encoder delay
    /// of AAC in MP4, which the decoder does not trim).
    delay: u64,
    /// The track's length when frames decoded after it are dropped here
    /// (the padding of AAC in MP4).
    end: Option<u64>,
    /// The frame index of the next frame [`Source::read`] returns.
    position: u64,
    /// Frames before this index are dropped (after a seek).
    skip_to: u64,
    /// A seek happened and its first packet has not been checked yet.
    after_seek: bool,
    /// The last packet as 24-bit integers, or as `f64` in 24-bit units for
    /// sources that are requantized.
    scratch: Vec<i32>,
    wide: Vec<f64>,
    requantizer: Requantizer,
    /// What [`Source::read`] scales requantized samples by (headroom).
    gain: f64,
    ended: bool,
}

/// A packet decoded: its first decoded frame index (the dither seed), the
/// frames to drop from its start, and its frames.
struct Block {
    start: u64,
    skip: usize,
    frames: usize,
}

/// A track's container, codec and parameters, and its reader.
struct Opened {
    reader: Box<dyn FormatReader>,
    track: Track,
    params: AudioCodecParameters,
    info: TrackInfo,
    /// The frames to play of what is decoded, when the decoder does not
    /// trim them itself.
    trim: Option<aac::Trim>,
}

/// Which covers [`open_reader`] looks for.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Covers {
    /// The embedded one, else one in the folder.
    EmbeddedOrFolder,
    /// The embedded one only (the folder is not listed).
    Embedded,
}

fn open_reader(path: &Path, covers: Covers) -> Result<Opened, Error> {
    let failed = |message: String| Error::Source {
        path: path.to_path_buf(),
        message,
    };
    let file = File::open(path).map_err(|e| failed(e.to_string()))?;
    let stream = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let mut reader = symphonia::default::get_probe()
        .probe(
            &hint,
            stream,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| {
            failed(format!(
                "not a FLAC, WAV, AIFF, CAF, MP4/M4A, MP3 or Ogg file ({e})"
            ))
        })?;
    let container = match reader.format_info().format {
        format_ids::FORMAT_ID_FLAC => Container::Flac,
        format_ids::FORMAT_ID_WAVE => Container::Wav,
        format_ids::FORMAT_ID_AIFF => Container::Aiff,
        format_ids::FORMAT_ID_CAF => Container::Caf,
        format_ids::FORMAT_ID_ISOMP4 => Container::Mp4,
        format_ids::FORMAT_ID_MP1 | format_ids::FORMAT_ID_MP2 | format_ids::FORMAT_ID_MP3 => {
            Container::Mpeg
        }
        format_ids::FORMAT_ID_OGG => Container::Ogg,
        _ => {
            let name = reader.format_info().long_name;
            return Err(failed(format!("the {name} container is not supported")));
        }
    };
    let track = reader
        .tracks()
        .iter()
        .find(|t| matches!(t.codec_params, Some(CodecParameters::Audio(_))))
        .ok_or_else(|| failed("no audio track".into()))?
        .clone();
    let Some(CodecParameters::Audio(params)) = &track.codec_params else {
        return Err(failed("the track has no audio parameters".into()));
    };
    let params = params.clone();
    let codec = codec_of(&params).map_err(failed)?;
    let (tags, embedded) = read_metadata(reader.as_mut());
    let aac = (codec == Codec::Aac)
        .then(|| aac::open_track(path, container, &track, &params, &tags))
        .transpose()
        .map_err(failed)?;
    let rate = aac
        .as_ref()
        .map(|a| a.rate)
        .or(params.sample_rate)
        .filter(|&r| r > 0)
        .ok_or_else(|| failed("no sample rate".into()))?;
    let channels = params.channels.as_ref().map_or(0, |c| c.count());
    if channels == 0 || channels > usize::from(u16::MAX) {
        return Err(failed("no channels".into()));
    }
    let lossy = codec.is_lossy();
    let float = lossy
        || matches!(
            params.sample_format,
            Some(SymphoniaSampleFormat::F32 | SymphoniaSampleFormat::F64)
        )
        || matches!(
            params.codec,
            codec_ids::CODEC_ID_PCM_F32LE
                | codec_ids::CODEC_ID_PCM_F32LE_PLANAR
                | codec_ids::CODEC_ID_PCM_F32BE
                | codec_ids::CODEC_ID_PCM_F32BE_PLANAR
                | codec_ids::CODEC_ID_PCM_F64LE
                | codec_ids::CODEC_ID_PCM_F64LE_PLANAR
                | codec_ids::CODEC_ID_PCM_F64BE
                | codec_ids::CODEC_ID_PCM_F64BE_PLANAR
        );
    let bits = if lossy {
        0
    } else {
        params
            .bits_per_sample
            .filter(|&b| b > 0)
            .or_else(|| {
                (codec == Codec::Alac)
                    .then(|| params.extra_data.as_deref().and_then(headers::alac_bits))
                    .flatten()
            })
            .unwrap_or(if float { 32 } else { 0 })
    };
    let bitrate = match codec {
        Codec::Mp1 | Codec::Mp2 | Codec::Mp3 => headers::mpeg(path),
        Codec::Vorbis => params.extra_data.as_deref().and_then(headers::vorbis),
        Codec::Aac => aac.as_ref().and_then(|a| a.bitrate),
        _ => None,
    };
    let cover = match covers {
        Covers::EmbeddedOrFolder => embedded.or_else(|| folder_cover(path)),
        Covers::Embedded => embedded,
    };
    let info = TrackInfo {
        path: path.to_path_buf(),
        container,
        codec,
        rate,
        channels: channels as u16,
        bits,
        float,
        bitrate,
        frames: match &aac {
            Some(a) => a.frames,
            None => track.num_frames,
        },
        tags,
        cover,
    };
    Ok(Opened {
        reader,
        track,
        params,
        info,
        trim: aac.and_then(|a| a.trim),
    })
}

fn make_decoder(
    path: &Path,
    params: &AudioCodecParameters,
    info: &TrackInfo,
) -> Result<Box<dyn AudioDecoder>, Error> {
    // Gapless: the decoder drops the encoder delay and padding the
    // container states.
    symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default().gapless(true))
        .map_err(|e| Error::Source {
            path: path.to_path_buf(),
            message: format!("cannot decode this {} stream ({e})", info.format_name()),
        })
}

/// The codec of a track, or why it does not play.
fn codec_of(params: &AudioCodecParameters) -> Result<Codec, String> {
    Ok(match params.codec {
        codec_ids::CODEC_ID_FLAC => Codec::Flac,
        codec_ids::CODEC_ID_ALAC => Codec::Alac,
        codec_ids::CODEC_ID_MP1 => Codec::Mp1,
        codec_ids::CODEC_ID_MP2 => Codec::Mp2,
        codec_ids::CODEC_ID_MP3 => Codec::Mp3,
        codec_ids::CODEC_ID_VORBIS => Codec::Vorbis,
        codec_ids::CODEC_ID_AAC => Codec::Aac,
        codec_ids::CODEC_ID_OPUS => return Err("Opus audio is not supported".into()),
        // Of the decoders built in, only PCM is left.
        id if symphonia::default::get_codecs()
            .get_audio_decoder(id)
            .is_some() =>
        {
            Codec::Pcm
        }
        _ => return Err("this audio codec is not supported".into()),
    })
}

impl Source {
    /// Opens `path` (the container chosen by its first bytes) and reads its
    /// tags and cover.
    pub fn open(path: impl AsRef<Path>) -> Result<Source, Error> {
        let path = path.as_ref();
        let Opened {
            reader,
            track,
            params,
            info,
            trim,
        } = open_reader(path, Covers::EmbeddedOrFolder)?;
        let decoder = make_decoder(path, &params, &info)?;
        let time_base = track
            .time_base
            .and_then(|tb| frames_per_tick(tb, info.rate));
        Ok(Source {
            info: Arc::new(info),
            reader,
            decoder,
            params,
            track_id: track.id,
            time_base,
            delay: trim.map_or(0, |t| t.delay),
            end: trim.map(|t| t.length),
            position: 0,
            skip_to: 0,
            after_seek: false,
            scratch: Vec::new(),
            wide: Vec::new(),
            requantizer: Requantizer::new(),
            gain: 1.0,
            ended: false,
        })
    }

    /// What the file is.
    pub fn info(&self) -> &Arc<TrackInfo> {
        &self.info
    }

    /// The frame index of the next frame [`Source::read`] returns.
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Samples that clipped when requantized to 24 bits (saturated), since
    /// the file was opened or last sought.
    pub fn clipped(&self) -> u64 {
        self.requantizer.clipped()
    }

    /// Scales the samples [`Source::read`] requantizes (a lossy,
    /// floating-point or wider than 24-bit file's) by `gain` first:
    /// headroom, so that a decoded signal over full scale need not clip.
    /// Integer samples of at most 24 bits pass exactly whatever it is, and
    /// [`Source::read_unquantized`] is never scaled.
    pub fn set_gain(&mut self, gain: f64) {
        self.gain = gain;
    }

    /// Appends the next packet's frames to `out` as interleaved 24-bit
    /// samples (`channels` per frame), and returns how many frames it
    /// appended: 0 at the end of the file.
    pub fn read(&mut self, out: &mut Vec<i32>) -> Result<usize, Error> {
        let channels = usize::from(self.info.channels);
        let Some(block) = self.next_block()? else {
            return Ok(0);
        };
        if self.info.fits_24_bits() {
            out.extend_from_slice(&self.scratch[block.skip * channels..]);
        } else {
            if self.gain != 1.0 {
                for y in &mut self.wide {
                    *y *= self.gain;
                }
            }
            let exact = self.wide.iter().all(|&y| on_grid(y));
            if exact {
                out.extend(self.wide[block.skip * channels..].iter().map(|&y| y as i32));
            } else {
                // The whole packet is requantized, from a seed of its own,
                // and the frames before a seek target dropped after.
                self.requantizer.reseed(block.start);
                let skipped = block.skip * channels;
                self.requantizer.skip(skipped);
                let requantizer = &mut self.requantizer;
                out.extend(
                    self.wide[skipped..]
                        .iter()
                        .map(|&y| requantizer.quantize(y)),
                );
            }
        }
        Ok(block.frames - block.skip)
    }

    /// Appends the next packet's frames to `out` as interleaved samples in
    /// 24-bit units (full scale ±2²³), before any requantization, and
    /// returns how many frames it appended: 0 at the end of the file.
    /// Integer sources of at most 24 bits give their exact values.
    pub fn read_unquantized(&mut self, out: &mut Vec<f64>) -> Result<usize, Error> {
        let channels = usize::from(self.info.channels);
        let Some(block) = self.next_block()? else {
            return Ok(0);
        };
        if self.info.fits_24_bits() {
            out.extend(
                self.scratch[block.skip * channels..]
                    .iter()
                    .map(|&s| f64::from(s)),
            );
        } else {
            out.extend_from_slice(&self.wide[block.skip * channels..]);
        }
        Ok(block.frames - block.skip)
    }

    /// The timestamp of decoded frame `frame` (counted from the encoder
    /// delay's start) in the track's time base.
    fn frame_to_ts(&self, frame: u64) -> i64 {
        match self.time_base {
            None => frame as i64,
            Some((num, den)) => (u128::from(frame) * u128::from(den) / u128::from(num)) as i64,
        }
    }

    /// The frame of timestamp `ts`.
    fn ts_to_frame(&self, ts: i64) -> i64 {
        match self.time_base {
            None => ts,
            Some((num, den)) => (i128::from(ts) * i128::from(num) / i128::from(den.max(1))) as i64,
        }
    }

    /// Decodes the next packet with frames to keep into `scratch` (24-bit
    /// integers) or `wide` (24-bit units), whole but for frames after the
    /// track's end; `None` at the end.
    fn next_block(&mut self) -> Result<Option<Block>, Error> {
        let channels = usize::from(self.info.channels);
        while !self.ended {
            let packet = match self.reader.next_packet() {
                Ok(Some(packet)) => packet,
                Ok(None) => {
                    self.ended = true;
                    break;
                }
                Err(_) if self.after_seek => {
                    // Some readers cannot read on from where they sought
                    // to: decode from the start instead.
                    self.after_seek = false;
                    self.rewind()?;
                    continue;
                }
                Err(e) => return Err(self.failed(format!("reading failed ({e})"))),
            };
            if packet.track_id != self.track_id {
                continue;
            }
            // The packet's timestamp is that of its first decoded frame;
            // the decoder drops the encoder delay (`trim_start`) from it.
            let ts = packet
                .pts
                .get()
                .saturating_add(packet.trim_start.get() as i64);
            let decoded_at = self.ts_to_frame(ts).max(0) as u64;
            // Its first frame in the track: before 0 in an encoder delay
            // that is trimmed here.
            let start = decoded_at as i64 - self.delay as i64;
            if self.end.is_some_and(|end| start >= end as i64) {
                self.ended = true;
                break;
            }
            let decoded = self.decoder.decode(&packet).map_err(|e| Error::Source {
                path: self.info.path.clone(),
                message: format!("decoding failed ({e})"),
            })?;
            let mut frames = decoded.frames();
            if frames == 0 {
                continue;
            }
            // The rate the track states may not be what its decoder gives
            // (an AAC stream's header and its container may disagree).
            let rate = decoded.spec().rate();
            if rate != self.info.rate {
                return Err(self.failed(format!(
                    "the stream decodes at {rate} Hz, not the {} Hz it states",
                    self.info.rate
                )));
            }
            if self.info.fits_24_bits() {
                decoded.copy_to_vec_interleaved(&mut self.scratch);
                // Full-scale 32-bit samples; an arithmetic shift
                // right-aligns 24 bits and keeps every bit of 16- and
                // 24-bit sources.
                for s in &mut self.scratch {
                    *s >>= 8;
                }
            } else {
                decoded.copy_to_vec_interleaved(&mut self.wide);
                for y in &mut self.wide {
                    *y *= SCALE_24;
                }
            }
            if self.after_seek {
                self.after_seek = false;
                if start > self.skip_to as i64 {
                    // The reader landed after the target: decode from the
                    // start of the track instead, dropping up to the
                    // target.
                    self.rewind()?;
                    continue;
                }
            }
            // The padding after the track's end is dropped.
            if let Some(end) = self.end {
                let keep = (end as i64 - start).clamp(0, frames as i64) as usize;
                if keep < frames {
                    frames = keep;
                    self.scratch.truncate(frames * channels);
                    self.wide.truncate(frames * channels);
                }
            }
            // After a seek (or in the encoder delay) the packet may start
            // before the frame asked for.
            let skip = (self.skip_to as i64 - start).clamp(0, frames as i64) as usize;
            if skip == frames {
                continue;
            }
            debug_assert_eq!(
                if self.info.fits_24_bits() {
                    self.scratch.len()
                } else {
                    self.wide.len()
                },
                frames * channels
            );
            self.position = (start + frames as i64) as u64;
            return Ok(Some(Block {
                start: decoded_at,
                skip,
                frames,
            }));
        }
        Ok(None)
    }

    /// Opens the file again, to decode it from the start (the frames up
    /// to `skip_to` are then dropped).
    fn rewind(&mut self) -> Result<(), Error> {
        let opened = open_reader(&self.info.path, Covers::Embedded)?;
        self.decoder = make_decoder(&self.info.path, &opened.params, &self.info)?;
        self.params = opened.params;
        self.reader = opened.reader;
        self.track_id = opened.track.id;
        self.ended = false;
        Ok(())
    }

    /// Moves so that the next [`Source::read`] starts exactly at `frame`
    /// (clamped to the length when it is known). Returns the frame.
    ///
    /// Lossy tracks are decoded from [`SEEK_PREROLL`] frames before the
    /// target. When the container cannot seek there, lands after the
    /// target or cannot read on from where it landed, the track is decoded
    /// from its start instead.
    pub fn seek(&mut self, frame: u64) -> Result<u64, Error> {
        let frame = self.info.frames.map_or(frame, |total| frame.min(total));
        self.skip_to = frame;
        self.position = frame;
        self.ended = false;
        self.requantizer.reset();
        if self.info.frames == Some(frame) && frame > 0 {
            // At the end: nothing more to read.
            self.ended = true;
            return Ok(frame);
        }
        let preroll = if self.info.is_lossy() {
            SEEK_PREROLL
        } else {
            0
        };
        // From before the target in decoded frames: into the encoder delay
        // when the target is near the start.
        let from = (frame + self.delay).saturating_sub(preroll);
        let sought = self.reader.seek(
            SeekMode::Accurate,
            SeekTo::Timestamp {
                ts: Timestamp::new(self.frame_to_ts(from)),
                track_id: self.track_id,
            },
        );
        if sought.is_err() {
            // Decode from the start instead.
            self.rewind()?;
            return Ok(frame);
        }
        // A new decoder, as at the start: none of its state (an AAC
        // decoder's noise generator for noise-substituted bands, say) is
        // carried over, so a seek to a frame always gives the same samples.
        self.decoder = make_decoder(&self.info.path, &self.params, &self.info)?;
        self.after_seek = true;
        Ok(frame)
    }

    fn failed(&self, message: String) -> Error {
        Error::Source {
            path: self.info.path.clone(),
            message,
        }
    }
}

/// Whether `y` (in 24-bit units) is a 24-bit integer.
fn on_grid(y: f64) -> bool {
    y == y.trunc() && (-SCALE_24..SCALE_24).contains(&y)
}

/// Frames per timestamp tick as a ratio `(num, den)`, when a tick is not a
/// frame.
fn frames_per_tick(tb: TimeBase, rate: u32) -> Option<(u64, u64)> {
    let num = u64::from(tb.numer.get()) * u64::from(rate);
    let den = u64::from(tb.denom.get());
    (num != den).then_some((num, den))
}

/// Opens `path` and returns what it is (with a cover from its folder when
/// it has none of its own), without keeping it open.
pub fn probe(path: impl AsRef<Path>) -> Result<TrackInfo, Error> {
    let source = Source::open(path)?;
    Ok((**source.info()).clone())
}

/// What `path` is from its headers alone: the container, codec, rate,
/// length, tags and embedded cover, without decoding any audio (and
/// without looking for a cover in the folder).
pub fn probe_header(path: impl AsRef<Path>) -> Result<TrackInfo, Error> {
    Ok(open_reader(path.as_ref(), Covers::Embedded)?.info)
}
