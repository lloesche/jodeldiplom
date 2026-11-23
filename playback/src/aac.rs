//! What an AAC track in an MP4 file needs that the demuxer leaves out.
//!
//! - **The profile.** The decoder decodes AAC-LC only, mono or stereo, in
//!   1024-frame packets. [`config`] reads the stream's
//!   `AudioSpecificConfig` itself, including the SBR and PS sync extensions
//!   that announce HE-AAC behind an LC header, so that a stream the decoder
//!   would play without its SBR band (at the core rate) is refused instead.
//! - **The gapless trim.** The demuxer hands out every decoded frame, the
//!   encoder delay and padding included. [`trim`] takes the track's first
//!   frame and length from the iTunes `iTunSMPB` tag, else from the track's
//!   edit list (`elst`).
//! - **The bit rate**, from the elementary stream descriptor (`esds`).
//!
//! [`mp4_track`] reads the atoms it needs from the file's `moov` atom; the
//! file is only read.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// The largest `moov` atom read (its sample tables grow with the length).
const MAX_MOOV: u64 = 64 * 1024 * 1024;

/// Rates by sampling frequency index.
const RATES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];

/// Audio object types.
const AOT_LC: u8 = 2;
const AOT_SBR: u8 = 5;
const AOT_PS: u8 = 29;
const AOT_ER_BSAC: u8 = 22;

/// An `AudioSpecificConfig`, as far as it decides whether the stream
/// plays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AacConfig {
    /// The core audio object type (2 for LC), under any SBR or PS layer.
    pub object_type: u8,
    /// The core sampling rate: what the decoder outputs.
    pub rate: u32,
    /// The channel configuration (0: a program config element).
    pub channels: u8,
    /// Whether SBR is signalled present (`Some(true)`), signalled absent
    /// (`Some(false)`), or not signalled at all.
    pub sbr: Option<bool>,
    /// Whether parametric stereo is signalled.
    pub ps: bool,
    /// Packets of 960 frames instead of 1024.
    pub frame_960: bool,
}

/// An MSB-first bit reader.
struct Bits<'a> {
    data: &'a [u8],
    at: usize,
}

impl Bits<'_> {
    fn left(&self) -> usize {
        (self.data.len() * 8).saturating_sub(self.at)
    }

    fn read(&mut self, n: usize) -> Option<u32> {
        if n > self.left() {
            return None;
        }
        let mut value = 0u32;
        for _ in 0..n {
            let bit = (self.data[self.at / 8] >> (7 - self.at % 8)) & 1;
            value = value << 1 | u32::from(bit);
            self.at += 1;
        }
        Some(value)
    }

    fn object_type(&mut self) -> Option<u8> {
        let aot = self.read(5)?;
        Some(if aot == 31 { 32 + self.read(6)? } else { aot } as u8)
    }

    fn rate(&mut self) -> Option<u32> {
        let index = self.read(4)?;
        if index == 15 {
            return self.read(24);
        }
        RATES.get(index as usize).copied()
    }
}

/// Reads an `AudioSpecificConfig` (ISO/IEC 14496-3, 1.6.2.1): the object
/// type, rate and channels, explicit SBR and PS signalling (hierarchical,
/// or the backward-compatible sync extensions after an LC header) and the
/// frame length. `None` when it is cut short or the rate is reserved.
pub(crate) fn config(asc: &[u8]) -> Option<AacConfig> {
    let mut bits = Bits { data: asc, at: 0 };
    let mut object_type = bits.object_type()?;
    let rate = bits.rate()?;
    let channels = bits.read(4)? as u8;
    let mut sbr = None;
    let mut ps = false;
    let hierarchical = object_type == AOT_SBR || object_type == AOT_PS;
    if hierarchical {
        sbr = Some(true);
        ps = object_type == AOT_PS;
        bits.rate()?;
        object_type = bits.object_type()?;
        if object_type == AOT_ER_BSAC {
            bits.read(4)?;
        }
    }
    let mut config = AacConfig {
        object_type,
        rate,
        channels,
        sbr,
        ps,
        frame_960: false,
    };
    // The general audio types: their GASpecificConfig.
    if !matches!(object_type, 1..=4 | 6 | 7 | 17 | 19..=23) {
        return Some(config);
    }
    config.frame_960 = bits.read(1)? == 1;
    if bits.read(1)? == 1 {
        bits.read(14)?; // coreCoderDelay
    }
    let extension = bits.read(1)? == 1;
    if channels == 0 {
        // A program config element follows; it is refused anyway.
        return Some(config);
    }
    if object_type == 6 || object_type == 20 {
        bits.read(3)?;
    }
    if extension {
        match object_type {
            AOT_ER_BSAC => {
                bits.read(16)?;
            }
            17 | 19 | 20 | 23 => {
                bits.read(3)?;
            }
            _ => {}
        }
        bits.read(1)?;
    }
    if matches!(object_type, 17 | 19..=27) && bits.read(2)? >= 2 {
        return Some(config);
    }
    // The backward-compatible signalling, after an LC header.
    if !hierarchical && bits.left() >= 16 && bits.read(11)? == 0x2b7 {
        match bits.object_type()? {
            AOT_SBR => {
                let present = bits.read(1)? == 1;
                config.sbr = Some(present);
                if present {
                    bits.rate()?;
                    if bits.left() >= 12 && bits.read(11)? == 0x548 {
                        config.ps = bits.read(1)? == 1;
                    }
                }
            }
            AOT_ER_BSAC => {
                let present = bits.read(1)? == 1;
                config.sbr = Some(present);
                if present {
                    bits.rate()?;
                }
            }
            _ => {}
        }
    }
    Some(config)
}

/// The name of an AAC profile, for the refusal messages.
fn profile_name(object_type: u8) -> String {
    match object_type {
        1 => "AAC Main".into(),
        3 => "AAC SSR".into(),
        4 => "AAC LTP".into(),
        6 => "scalable AAC".into(),
        17 => "ER AAC-LC".into(),
        19 => "ER AAC LTP".into(),
        20 => "ER scalable AAC".into(),
        22 => "ER BSAC".into(),
        23 => "AAC-LD".into(),
        39 => "AAC-ELD".into(),
        42 => "xHE-AAC (USAC)".into(),
        aot => format!("MPEG-4 audio object type {aot}"),
    }
}

/// Why a stream with this configuration does not play, if it does not:
/// `entry_rate` and `media_timescale` are the rates the container states
/// (the sample entry's and the track's).
pub(crate) fn refusal(
    config: &AacConfig,
    entry_rate: Option<u32>,
    media_timescale: Option<u32>,
) -> Option<String> {
    const ONLY: &str = "only AAC-LC plays";
    if config.ps {
        return Some(format!(
            "HE-AAC v2 (AAC with SBR and parametric stereo) is not supported: {ONLY}"
        ));
    }
    if config.sbr == Some(true) {
        return Some(format!("HE-AAC (AAC with SBR) is not supported: {ONLY}"));
    }
    if config.object_type != AOT_LC {
        return Some(format!(
            "{} is not supported: {ONLY}",
            profile_name(config.object_type)
        ));
    }
    // SBR left unsignalled (implicit): the container states the output
    // rate, twice the core rate the header states.
    let doubled = |rate: Option<u32>| rate == Some(2 * config.rate);
    if config.sbr.is_none()
        && config.rate <= 24_000
        && (doubled(entry_rate) || doubled(media_timescale))
    {
        return Some(format!(
            "this looks like HE-AAC (AAC with SBR): the file plays at {} Hz but \
             its AAC core at {} Hz; {ONLY}",
            2 * config.rate,
            config.rate
        ));
    }
    if config.channels == 0 {
        return Some(format!(
            "AAC with a program config element (a custom channel layout) is not supported: \
             {ONLY}, mono or stereo"
        ));
    }
    if config.channels > 2 {
        return Some("multichannel AAC is not supported: AAC plays in mono or stereo".into());
    }
    if config.frame_960 {
        return Some("AAC with 960-frame packets is not supported".into());
    }
    None
}

/// One edit list entry, in the movie's (duration) and the track's (media
/// time) timescales.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Edit {
    pub duration: u64,
    /// -1 for an empty edit.
    pub media_time: i64,
    /// The integer part of the rate.
    pub rate: i16,
}

/// What the `moov` atom says about one track.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Mp4Track {
    /// The movie timescale (`mvhd`), for edit durations.
    pub movie_timescale: u32,
    /// The track's timescale (`mdhd`).
    pub media_timescale: u32,
    /// The rate in the sample entry, when it states one.
    pub entry_rate: Option<u32>,
    /// The length of all samples in the track's timescale (`stts`).
    pub media_duration: u64,
    /// The edit list.
    pub edits: Vec<Edit>,
    /// The average bit rate the `esds` states.
    pub avg_bitrate: Option<u32>,
}

fn be_u16(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes(b.get(at..at + 2)?.try_into().ok()?))
}

fn be_u32(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

fn be_u64(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_be_bytes(b.get(at..at + 8)?.try_into().ok()?))
}

/// The atoms in `data`, as (type, body).
fn atoms(data: &[u8]) -> impl Iterator<Item = ([u8; 4], &[u8])> {
    let mut at = 0usize;
    std::iter::from_fn(move || {
        let size = be_u32(data, at)? as usize;
        let kind: [u8; 4] = data.get(at + 4..at + 8)?.try_into().ok()?;
        let (header, size) = match size {
            0 => (8, data.len() - at),
            1 => (16, usize::try_from(be_u64(data, at + 8)?).ok()?),
            n => (8, n),
        };
        let body = data.get(at + header..at.checked_add(size)?)?;
        at += size;
        Some((kind, body))
    })
}

fn child<'a>(data: &'a [u8], kind: &[u8; 4]) -> Option<&'a [u8]> {
    atoms(data).find(|(k, _)| k == kind).map(|(_, b)| b)
}

/// A full atom's timescale: after version and flags, two times (32 or 64
/// bits by the version).
fn timescale(body: &[u8]) -> Option<u32> {
    be_u32(body, if body.first()? == &1 { 20 } else { 12 })
}

/// Reads `path`'s `moov` atom and what it says about the track `track_id`.
pub(crate) fn mp4_track(path: &Path, track_id: u32) -> Option<Mp4Track> {
    let moov = read_moov(path)?;
    let movie_timescale = timescale(child(&moov, b"mvhd")?)?;
    let trak = atoms(&moov)
        .filter(|(k, _)| k == b"trak")
        .find_map(|(_, trak)| {
            let tkhd = child(trak, b"tkhd")?;
            let id = be_u32(tkhd, if tkhd.first()? == &1 { 20 } else { 12 })?;
            (id == track_id).then_some(trak)
        })?;
    let mdia = child(trak, b"mdia")?;
    let stbl = child(child(mdia, b"minf")?, b"stbl")?;
    let mut track = Mp4Track {
        movie_timescale,
        media_timescale: timescale(child(mdia, b"mdhd")?)?,
        media_duration: stts_total(child(stbl, b"stts")?)?,
        edits: child(trak, b"edts")
            .and_then(|e| child(e, b"elst"))
            .map(elst)
            .unwrap_or_default(),
        ..Mp4Track::default()
    };
    if let Some(entry) = child(stbl, b"stsd").and_then(|s| s.get(8..)) {
        if let Some((_, mp4a)) = atoms(entry).next() {
            sample_entry(mp4a, &mut track);
        }
    }
    Some(track)
}

fn read_moov(path: &Path) -> Option<Vec<u8>> {
    let mut file = File::open(path).ok()?;
    let length = file.metadata().ok()?.len();
    let mut at = 0u64;
    while at + 8 <= length {
        file.seek(SeekFrom::Start(at)).ok()?;
        let mut header = [0u8; 16];
        file.read_exact(&mut header[..8]).ok()?;
        let (skip, size) = match be_u32(&header, 0)? {
            0 => (8, length - at),
            1 => {
                file.read_exact(&mut header[8..]).ok()?;
                (16, be_u64(&header, 8)?)
            }
            n => (8, u64::from(n)),
        };
        if size < skip {
            return None;
        }
        if &header[4..8] == b"moov" {
            let body = size - skip;
            if body > MAX_MOOV {
                return None;
            }
            let mut moov = vec![0u8; body as usize];
            file.read_exact(&mut moov).ok()?;
            return Some(moov);
        }
        at = at.checked_add(size)?;
    }
    None
}

fn stts_total(body: &[u8]) -> Option<u64> {
    let count = be_u32(body, 4)? as usize;
    (0..count).try_fold(0u64, |sum, i| {
        let samples = u64::from(be_u32(body, 8 + 8 * i)?);
        let delta = u64::from(be_u32(body, 12 + 8 * i)?);
        sum.checked_add(samples * delta)
    })
}

/// The entries of an `elst` atom's body.
pub(crate) fn elst(body: &[u8]) -> Vec<Edit> {
    let wide = body.first() == Some(&1);
    let count = be_u32(body, 4).unwrap_or(0) as usize;
    let size = if wide { 20 } else { 12 };
    (0..count)
        .map_while(|i| {
            let at = 8 + size * i;
            let (duration, media_time, rate_at) = if wide {
                (be_u64(body, at)?, be_u64(body, at + 8)? as i64, at + 16)
            } else {
                (
                    u64::from(be_u32(body, at)?),
                    i64::from(be_u32(body, at + 4)? as i32),
                    at + 8,
                )
            };
            Some(Edit {
                duration,
                media_time,
                rate: be_u16(body, rate_at)? as i16,
            })
        })
        .collect()
}

/// The rate and the `esds` bit rate of an `mp4a` sample entry (QuickTime
/// sound descriptions of version 0, 1 and 2).
fn sample_entry(body: &[u8], track: &mut Mp4Track) {
    let version = be_u16(body, 8).unwrap_or(0);
    let (rate, children) = match version {
        1 => (be_u32(body, 24).map(|r| r >> 16), 44),
        2 => (
            body.get(32..40)
                .and_then(|b| b.try_into().ok())
                .map(f64::from_be_bytes)
                .filter(|r| r.is_finite() && *r >= 1.0 && *r <= f64::from(u32::MAX))
                .map(|r| r.round() as u32),
            64,
        ),
        _ => (be_u32(body, 24).map(|r| r >> 16), 28),
    };
    track.entry_rate = rate.filter(|&r| r > 0);
    let Some(children) = body.get(children..) else {
        return;
    };
    let esds = child(children, b"esds").or_else(|| child(child(children, b"wave")?, b"esds"));
    track.avg_bitrate = esds.and_then(esds_avg_bitrate);
}

/// A descriptor's tag, its body and what follows it.
fn descriptor(data: &[u8]) -> Option<(u8, &[u8], &[u8])> {
    let tag = *data.first()?;
    let mut length = 0usize;
    let mut at = 1;
    loop {
        let byte = *data.get(at)?;
        at += 1;
        length = length << 7 | usize::from(byte & 0x7f);
        if byte & 0x80 == 0 || at > 4 {
            break;
        }
    }
    let body = data.get(at..at.checked_add(length)?)?;
    Some((tag, body, &data[at + length..]))
}

/// The average bit rate of an `esds` body's DecoderConfigDescriptor.
fn esds_avg_bitrate(body: &[u8]) -> Option<u32> {
    let (tag, es, _) = descriptor(body.get(4..)?)?;
    if tag != 0x03 {
        return None;
    }
    let flags = *es.get(2)?;
    let mut at = 3;
    if flags & 0x80 != 0 {
        at += 2;
    }
    if flags & 0x40 != 0 {
        at += 1 + usize::from(*es.get(at)?);
    }
    if flags & 0x20 != 0 {
        at += 2;
    }
    let (tag, config, _) = descriptor(es.get(at..)?)?;
    if tag != 0x04 {
        return None;
    }
    be_u32(config, 9).filter(|&b| b > 0)
}

/// The encoder delay, padding and length (in frames) of an iTunes
/// `iTunSMPB` tag: hexadecimal fields, the second to the fourth.
pub(crate) fn itunsmpb(text: &str) -> Option<(u64, u64, u64)> {
    let fields: Vec<u64> = text
        .split_whitespace()
        .take(4)
        .map(|f| u64::from_str_radix(f, 16).ok())
        .collect::<Option<_>>()?;
    match fields[..] {
        [_, delay, padding, length] => Some((delay, padding, length)),
        _ => None,
    }
}

/// The frames of a track to play: the decoded frames from `delay` on,
/// `length` of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Trim {
    pub delay: u64,
    pub length: u64,
}

/// `ticks` of `timescale` in frames at `rate`, rounded.
pub(crate) fn frames(ticks: u64, timescale: u32, rate: u32) -> u64 {
    let timescale = u128::from(timescale.max(1));
    ((u128::from(ticks) * u128::from(rate) + timescale / 2) / timescale) as u64
}

/// Where the track starts and how long it is, at `rate` frames per second:
/// from the `iTunSMPB` tag when it fits in the track, else from a single
/// edit (after any empty ones), else `None` (every decoded frame plays).
pub(crate) fn trim(rate: u32, track: &Mp4Track, smpb: Option<(u64, u64, u64)>) -> Option<Trim> {
    let to_frames = |ticks: u64, timescale: u32| frames(ticks, timescale, rate);
    let total = to_frames(track.media_duration, track.media_timescale);
    if let Some((delay, _, length)) = smpb {
        if length > 0 && delay.checked_add(length).is_some_and(|end| end <= total) {
            return Some(Trim { delay, length });
        }
    }
    let mut edits = track.edits.iter().skip_while(|e| e.media_time == -1);
    let edit = edits.next()?;
    if edits.next().is_some() || edit.media_time < 0 || edit.rate != 1 {
        return None;
    }
    let delay = to_frames(edit.media_time as u64, track.media_timescale);
    let available = total.checked_sub(delay)?;
    let length = if edit.duration == 0 {
        available
    } else {
        // An edit's length is in movie ticks; a coarse movie timescale
        // rounds it, so within a tick the track's own length is taken.
        let stated = to_frames(edit.duration, track.movie_timescale);
        let tick = u64::from(rate).div_ceil(u64::from(track.movie_timescale.max(1)));
        if stated.abs_diff(available) <= tick {
            available
        } else {
            stated.min(available)
        }
    };
    (delay > 0 || length < total).then_some(Trim { delay, length })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An `AudioSpecificConfig` from (value, bits) fields.
    fn asc(fields: &[(u32, usize)]) -> Vec<u8> {
        let mut bits = Vec::new();
        for &(value, n) in fields {
            bits.extend((0..n).rev().map(|i| (value >> i) & 1 == 1));
        }
        bits.chunks(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u8, |b, (i, &x)| b | (u8::from(x) << (7 - i)))
            })
            .collect()
    }

    #[test]
    fn lc_headers_play() {
        // What an encoder writes: LC, 44.1 kHz, stereo, and the
        // backward-compatible extension saying there is no SBR.
        let ffmpeg = config(&[0x12, 0x10, 0x56, 0xe5, 0x00]).unwrap();
        assert_eq!(
            ffmpeg,
            AacConfig {
                object_type: 2,
                rate: 44_100,
                channels: 2,
                sbr: Some(false),
                ps: false,
                frame_960: false,
            }
        );
        assert_eq!(refusal(&ffmpeg, Some(44_100), Some(44_100)), None);
        // The bare two bytes, mono at 48 kHz.
        let bare = config(&[0x11, 0x88]).unwrap();
        assert_eq!((bare.rate, bare.channels, bare.sbr), (48_000, 1, None));
        assert_eq!(refusal(&bare, Some(48_000), Some(48_000)), None);
        // LC at 22.05 kHz where the container agrees: it plays.
        let low = config(&asc(&[(2, 5), (7, 4), (2, 4), (0, 3)])).unwrap();
        assert_eq!(refusal(&low, Some(22_050), Some(22_050)), None);
        // A 96 kHz entry rate does not fit 16.16 bits: any entry rate goes.
        let high = config(&asc(&[(2, 5), (0, 4), (2, 4), (0, 3)])).unwrap();
        assert_eq!(refusal(&high, Some(30_464), Some(96_000)), None);
    }

    #[test]
    fn he_aac_is_refused_however_it_is_signalled() {
        let he = |c: &AacConfig| refusal(c, None, None).unwrap();
        // Hierarchical: AOT 5, core 22.05 kHz, extension 44.1 kHz, LC.
        let explicit = config(&asc(&[(5, 5), (7, 4), (2, 4), (4, 4), (2, 5), (0, 3)])).unwrap();
        assert_eq!((explicit.object_type, explicit.rate), (2, 22_050));
        assert!(he(&explicit).starts_with("HE-AAC (AAC with SBR)"));
        // AOT 29: HE-AAC v2.
        let v2 = config(&asc(&[(29, 5), (8, 4), (1, 4), (3, 4), (2, 5), (0, 3)])).unwrap();
        assert!(he(&v2).starts_with("HE-AAC v2"));
        // Backward compatible: LC, then sync 0x2b7, AOT 5, present, 44.1 kHz.
        let compatible = asc(&[
            (2, 5),
            (7, 4),
            (2, 4),
            (0, 3),
            (0x2b7, 11),
            (5, 5),
            (1, 1),
            (4, 4),
        ]);
        assert!(he(&config(&compatible).unwrap()).starts_with("HE-AAC (AAC with SBR)"));
        // ... and PS behind it, sync 0x548 (a mono core).
        let with_ps = asc(&[
            (2, 5),
            (7, 4),
            (1, 4),
            (0, 3),
            (0x2b7, 11),
            (5, 5),
            (1, 1),
            (4, 4),
            (0x548, 11),
            (1, 1),
        ]);
        assert!(he(&config(&with_ps).unwrap()).starts_with("HE-AAC v2"));
        // Implicit: an LC header at 24 kHz in a 48 kHz track.
        let implicit = config(&asc(&[(2, 5), (6, 4), (2, 4), (0, 3)])).unwrap();
        assert!(refusal(&implicit, Some(48_000), Some(48_000))
            .unwrap()
            .starts_with("this looks like HE-AAC"));
        assert!(refusal(&implicit, None, Some(48_000)).is_some());
        assert_eq!(refusal(&implicit, Some(24_000), Some(24_000)), None);
    }

    #[test]
    fn other_profiles_and_layouts_are_refused() {
        let refused = |fields: &[(u32, usize)]| refusal(&config(&asc(fields)).unwrap(), None, None);
        assert!(refused(&[(1, 5), (4, 4), (2, 4), (0, 3)])
            .unwrap()
            .starts_with("AAC Main"));
        assert!(refused(&[(4, 5), (4, 4), (2, 4), (0, 3)])
            .unwrap()
            .starts_with("AAC LTP"));
        assert!(refused(&[(23, 5), (4, 4), (2, 4), (0, 3), (0, 2)])
            .unwrap()
            .starts_with("AAC-LD"));
        // USAC: an escaped object type (31, then 42 - 32).
        assert!(refused(&[(31, 5), (10, 6), (4, 4), (2, 4)])
            .unwrap()
            .starts_with("xHE-AAC"));
        // 5.1, a program config element, 960-frame packets.
        assert!(refused(&[(2, 5), (4, 4), (6, 4), (0, 3)])
            .unwrap()
            .starts_with("multichannel"));
        assert!(refused(&[(2, 5), (4, 4), (0, 4), (0, 3)])
            .unwrap()
            .contains("program config element"));
        assert!(refused(&[(2, 5), (4, 4), (2, 4), (4, 3)])
            .unwrap()
            .contains("960"));
        // An escaped rate; a reserved rate index; too short.
        let escaped = config(&asc(&[(2, 5), (15, 4), (44_100, 24), (2, 4), (0, 3)])).unwrap();
        assert_eq!(escaped.rate, 44_100);
        assert_eq!(config(&asc(&[(2, 5), (13, 4), (2, 4), (0, 3)])), None);
        assert_eq!(config(&[0x12]), None);
    }

    #[test]
    fn itunsmpb_fields() {
        let text = " 00000000 00000840 000001CA 00000000003F31F6 00000000 00000000";
        assert_eq!(itunsmpb(text), Some((2112, 458, 0x3F31F6)));
        assert_eq!(itunsmpb("00000000 00000840"), None);
        assert_eq!(itunsmpb("x y z w"), None);
    }

    #[test]
    fn edit_lists_parse() {
        // Version 0: one edit of 19 999 ticks from media time 1024.
        let body = [
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0x4e, 0x1f, 0, 0, 4, 0, 0, 1, 0, 0,
        ];
        assert_eq!(
            elst(&body),
            [Edit {
                duration: 19_999,
                media_time: 1024,
                rate: 1
            }]
        );
        // Version 1, an empty edit then one.
        let mut wide = vec![1, 0, 0, 0, 0, 0, 0, 2];
        wide.extend(500u64.to_be_bytes());
        wide.extend((-1i64).to_be_bytes());
        wide.extend([0, 1, 0, 0]);
        wide.extend(1000u64.to_be_bytes());
        wide.extend(2112i64.to_be_bytes());
        wide.extend([0, 1, 0, 0]);
        let edits = elst(&wide);
        assert_eq!(edits.len(), 2);
        assert_eq!((edits[0].media_time, edits[1].media_time), (-1, 2112));
        // Cut short: the whole entries only.
        assert_eq!(elst(&body[..16]), []);
    }

    fn track(edits: Vec<Edit>, movie_timescale: u32, media_duration: u64) -> Mp4Track {
        Mp4Track {
            movie_timescale,
            media_timescale: 44_100,
            entry_rate: Some(44_100),
            media_duration,
            edits,
            avg_bitrate: None,
        }
    }

    fn edit(duration: u64, media_time: i64) -> Edit {
        Edit {
            duration,
            media_time,
            rate: 1,
        }
    }

    #[test]
    fn trims_come_from_the_tag_else_the_edit() {
        // 22 packets of 1024 frames: 22 528.
        let total = 22 * 1024;
        // An exact edit (movie timescale = rate) whose last packet is short.
        let exact = track(vec![edit(19_999, 1024)], 44_100, 1024 + 19_999);
        assert_eq!(
            trim(44_100, &exact, None),
            Some(Trim {
                delay: 1024,
                length: 19_999
            })
        );
        // A coarse movie timescale (ms): within a tick, the track's length.
        let coarse = track(vec![edit(453, 1024)], 1000, 1024 + 19_999);
        assert_eq!(trim(44_100, &coarse, None).unwrap().length, 19_999);
        // Padding beyond the tick: the edit's length, rounded.
        let padded = track(vec![edit(20_000, 2112)], 44_100, total);
        assert_eq!(
            trim(44_100, &padded, None),
            Some(Trim {
                delay: 2112,
                length: 20_000
            })
        );
        // The tag wins over the edit, when it fits in the track.
        assert_eq!(
            trim(44_100, &padded, Some((2112, 0, 19_000))),
            Some(Trim {
                delay: 2112,
                length: 19_000
            })
        );
        assert_eq!(
            trim(44_100, &padded, Some((2112, 0, 30_000)))
                .unwrap()
                .length,
            20_000
        );
        // Leading empty edits go; two edits, a rate of 0 or nothing to cut:
        // no trim.
        let empty_first = track(vec![edit(100, -1), edit(20_000, 2112)], 44_100, total);
        assert_eq!(trim(44_100, &empty_first, None).unwrap().delay, 2112);
        let two = track(vec![edit(100, 0), edit(20_000, 2112)], 44_100, total);
        assert_eq!(trim(44_100, &two, None), None);
        let mut dwell = edit(20_000, 2112);
        dwell.rate = 0;
        assert_eq!(trim(44_100, &track(vec![dwell], 44_100, total), None), None);
        let whole = track(vec![edit(total, 0)], 44_100, total);
        assert_eq!(trim(44_100, &whole, None), None);
        assert_eq!(trim(44_100, &track(vec![], 44_100, total), None), None);
    }
}
