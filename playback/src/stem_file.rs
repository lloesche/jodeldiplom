//! A track's stems kept on disk, so they are separated once.
//!
//! Separating stems is by far the most work the player does: a 6 s chunk
//! is about half a second of a laptop GPU. A track's stems are separated
//! once, when it is got ready (with its analysis), and kept in a file of
//! this format; playing on stems, a jump, a transition or the Transition tab
//! read them back instead of separating again ([`StemStore`]).
//!
//! # The format
//!
//! The four stems in [`Stem::ALL`](crate::Stem::ALL) order, each stereo, at
//! [`RATE`] (the separator's rate), from the track's start, in blocks of
//! [`BLOCK`] frames that can be read on their own:
//!
//! - a header: [`MAGIC`], [`FORMAT`], the rate, the track file's own rate,
//!   the frames, and the separator's name;
//! - the blocks, each the eight channels (stem by stem, left then right),
//!   each channel as 16-bit values times a power of two of its own (from
//!   2^-16 to 2^4: a quiet stretch keeps its detail and a loud one never
//!   clips), coded losslessly: silent, constant, or a fixed predictor of
//!   order 0 to 3 and its residuals in a Rice code;
//! - an index of the blocks' offsets, and a footer pointing to it.
//!
//! The values are within half a step of 16 bits of each channel's block
//! peak (about 96 dB below it): far below what four stems summed again,
//! faded and resampled, can show. A file is written to a temporary name and
//! renamed once complete, so a file that is there is whole.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::stems::{RATE, STEMS};

/// What a stems file starts with.
pub const MAGIC: &[u8; 8] = b"JDLSTEMS";
/// The format's version; another is not read.
pub const FORMAT: u32 = 1;
/// Frames in a block.
pub const BLOCK: usize = 4096;
/// The channels in a block: each stem's two.
const CHANNELS: usize = STEMS * 2;
/// What the footer ends with.
const END: &[u8; 8] = b"STEMSEND";
/// The largest value a sample takes (16 bits, symmetric).
const FULL: f32 = 32_767.0;
/// The exponents a channel's block may have.
const EXPONENTS: std::ops::RangeInclusive<i32> = -16..=4;
/// Rice quotients from this on are written as an escape and the value raw.
const ESCAPE: u32 = 32;
/// Bits of a raw value after an escape (enough for any residual).
const RAW_BITS: u32 = 24;

/// The stems of a track, `[stem][channel]` in [`Stem::ALL`](crate::Stem::ALL)
/// order.
pub type Stems = [[Vec<f32>; 2]; STEMS];

fn empty() -> Stems {
    std::array::from_fn(|_| [Vec::new(), Vec::new()])
}

/// Where the stems of tracks are kept: the engine asks it for a track's
/// file before it separates the track itself.
pub trait StemStore: Send + Sync {
    /// The stems of `path`, when they are kept (and whole).
    fn open(&self, path: &Path) -> Option<StemFile>;
}

impl std::fmt::Debug for dyn StemStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("StemStore")
    }
}

/// Writes a stems file as the stems come, in order from the track's start.
pub struct StemWriter {
    /// Under a temporary name beside where it goes, until it is whole.
    file: Option<durable::AtomicFile>,
    /// Frames waiting for a whole block.
    pending: [Vec<f32>; CHANNELS],
    offsets: Vec<u64>,
    at: u64,
    frames: u64,
    frames_at: u64,
}

impl StemWriter {
    /// Starts the stems file for `target` (of a track at `source_rate`,
    /// separated by `separator`), under a temporary name beside it.
    pub fn create(target: &Path, source_rate: u32, separator: &str) -> io::Result<StemWriter> {
        let mut file = durable::AtomicFile::create(target)?;
        let name = separator.as_bytes();
        let mut header = Vec::new();
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&FORMAT.to_le_bytes());
        header.extend_from_slice(&RATE.to_le_bytes());
        header.extend_from_slice(&source_rate.to_le_bytes());
        // The frames, filled in at the end.
        let frames_at = header.len() as u64;
        header.extend_from_slice(&0u64.to_le_bytes());
        header.extend_from_slice(&(name.len().min(255) as u8).to_le_bytes());
        header.extend_from_slice(&name[..name.len().min(255)]);
        file.write_all(&header)?;
        Ok(StemWriter {
            file: Some(file),
            pending: std::array::from_fn(|_| Vec::with_capacity(BLOCK)),
            offsets: Vec::new(),
            at: header.len() as u64,
            frames: 0,
            frames_at,
        })
    }

    /// Adds the next frames: `[stem][channel]`, all the same length.
    pub fn push(&mut self, stems: [[&[f32]; 2]; STEMS]) -> io::Result<()> {
        let n = stems[0][0].len();
        let mut done = 0;
        while done < n {
            let room = BLOCK - self.pending[0].len();
            let take = room.min(n - done);
            for (c, pending) in self.pending.iter_mut().enumerate() {
                pending.extend_from_slice(&stems[c / 2][c % 2][done..done + take]);
            }
            done += take;
            if self.pending[0].len() == BLOCK {
                self.flush_block()?;
            }
        }
        Ok(())
    }

    fn flush_block(&mut self) -> io::Result<()> {
        let frames = self.pending[0].len();
        if frames == 0 {
            return Ok(());
        }
        let mut bytes = Vec::with_capacity(frames * 2 * CHANNELS / 2);
        bytes.extend_from_slice(&(frames as u32).to_le_bytes());
        for channel in &self.pending {
            encode_channel(channel, &mut bytes);
        }
        let file = self.file.as_mut().ok_or_else(closed)?;
        file.write_all(&bytes)?;
        self.offsets.push(self.at);
        self.at += bytes.len() as u64;
        self.frames += frames as u64;
        for pending in &mut self.pending {
            pending.clear();
        }
        Ok(())
    }

    /// Writes what is left, the index and the footer, and gives the file
    /// its name.
    pub fn finish(mut self) -> io::Result<()> {
        self.flush_block()?;
        let mut file = self.file.take().ok_or_else(closed)?;
        let index_at = self.at;
        for offset in &self.offsets {
            file.write_all(&offset.to_le_bytes())?;
        }
        file.write_all(&(self.offsets.len() as u64).to_le_bytes())?;
        file.write_all(&index_at.to_le_bytes())?;
        file.write_all(END)?;
        file.seek(SeekFrom::Start(self.frames_at))?;
        file.write_all(&self.frames.to_le_bytes())?;
        // Its bytes, then its name, made durable. Dropped unfinished, the
        // temporary goes and nothing is left behind.
        file.commit()
    }
}

fn closed() -> io::Error {
    io::Error::other("the stems file is closed")
}

fn bad(what: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("not a stems file: {what}"),
    )
}

/// A stems file open for reading.
pub struct StemFile {
    file: File,
    source_rate: u32,
    frames: u64,
    separator: String,
    offsets: Vec<u64>,
    /// The last block decoded.
    decoded: Option<(usize, [Vec<f32>; CHANNELS])>,
}

impl StemFile {
    /// Opens the stems file at `path`; an error if it is not one of this
    /// format, or not whole.
    pub fn open(path: &Path) -> io::Result<StemFile> {
        let mut file = File::open(path)?;
        let mut head = [0u8; 8 + 4 + 4 + 4 + 8 + 1];
        file.read_exact(&mut head)?;
        if &head[..8] != MAGIC {
            return Err(bad("its start"));
        }
        let u32_at =
            |at: usize| u32::from_le_bytes(head[at..at + 4].try_into().unwrap_or_default());
        if u32_at(8) != FORMAT || u32_at(12) != RATE {
            return Err(bad("another version"));
        }
        let source_rate = u32_at(16);
        let frames = u64::from_le_bytes(head[20..28].try_into().unwrap_or_default());
        let mut name = vec![0u8; usize::from(head[28])];
        file.read_exact(&mut name)?;
        let separator = String::from_utf8_lossy(&name).into_owned();
        let end = file.seek(SeekFrom::End(-24))?;
        let mut foot = [0u8; 24];
        file.read_exact(&mut foot)?;
        if &foot[16..] != END {
            return Err(bad("its end"));
        }
        let count = u64::from_le_bytes(foot[..8].try_into().unwrap_or_default());
        let index_at = u64::from_le_bytes(foot[8..16].try_into().unwrap_or_default());
        if index_at + count * 8 != end || count > end / 4 {
            return Err(bad("its index"));
        }
        file.seek(SeekFrom::Start(index_at))?;
        let mut index = vec![0u8; count as usize * 8];
        file.read_exact(&mut index)?;
        let offsets: Vec<u64> = index
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap_or_default()))
            .collect();
        if frames.div_ceil(BLOCK as u64) != count {
            return Err(bad("its length"));
        }
        Ok(StemFile {
            file,
            source_rate,
            frames,
            separator,
            offsets,
            decoded: None,
        })
    }

    /// Its frames at [`RATE`].
    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// The rate of the track it was separated from.
    pub fn source_rate(&self) -> u32 {
        self.source_rate
    }

    /// The separator that made it.
    pub fn separator(&self) -> &str {
        &self.separator
    }

    /// Up to `len` frames from frame `from` (fewer at the end).
    pub fn read(&mut self, from: u64, len: usize) -> io::Result<Stems> {
        let mut out = empty();
        let end = (from + len as u64).min(self.frames);
        let mut at = from;
        while at < end {
            let block = (at / BLOCK as u64) as usize;
            self.decode(block)?;
            let Some((_, channels)) = &self.decoded else {
                return Err(bad("a block"));
            };
            let first = (at - block as u64 * BLOCK as u64) as usize;
            let take = (channels[0].len() - first).min((end - at) as usize);
            if take == 0 {
                return Err(bad("a short block"));
            }
            for (c, channel) in channels.iter().enumerate() {
                out[c / 2][c % 2].extend_from_slice(&channel[first..first + take]);
            }
            at += take as u64;
        }
        Ok(out)
    }

    fn decode(&mut self, block: usize) -> io::Result<()> {
        if self.decoded.as_ref().is_some_and(|(b, _)| *b == block) {
            return Ok(());
        }
        let start = *self
            .offsets
            .get(block)
            .ok_or_else(|| bad("a block past its end"))?;
        let next = self
            .offsets
            .get(block + 1)
            .copied()
            .unwrap_or(start + (BLOCK * CHANNELS * 4) as u64 + 64);
        self.file.seek(SeekFrom::Start(start))?;
        let mut bytes = Vec::new();
        (&mut self.file)
            .take(next.saturating_sub(start))
            .read_to_end(&mut bytes)?;
        let frames = u32::from_le_bytes(
            bytes
                .get(..4)
                .ok_or_else(|| bad("a block's length"))?
                .try_into()
                .unwrap_or_default(),
        ) as usize;
        if frames == 0 || frames > BLOCK {
            return Err(bad("a block's length"));
        }
        let mut at = 4;
        let mut channels: [Vec<f32>; CHANNELS] = std::array::from_fn(|_| Vec::new());
        for channel in &mut channels {
            *channel = decode_channel(&bytes, &mut at, frames).ok_or_else(|| bad("a channel"))?;
        }
        self.decoded = Some((block, channels));
        Ok(())
    }
}

/// The fixed predictors' residuals of `q` at `order` (from sample
/// `order` on).
fn residuals(q: &[i32], order: usize, out: &mut Vec<i32>) {
    out.clear();
    for i in order..q.len() {
        let p = match order {
            0 => 0,
            1 => q[i - 1],
            2 => 2 * q[i - 1] - q[i - 2],
            _ => 3 * q[i - 1] - 3 * q[i - 2] + q[i - 3],
        };
        out.push(q[i] - p);
    }
}

fn zigzag(v: i32) -> u32 {
    ((v << 1) ^ (v >> 31)) as u32
}

fn unzigzag(u: u32) -> i32 {
    ((u >> 1) as i32) ^ -((u & 1) as i32)
}

/// The Rice parameter for values of mean `mean`.
fn rice_k(mean: f64) -> u32 {
    if mean < 1.0 {
        0
    } else {
        (mean.log2().floor() as u32).min(RAW_BITS - 1)
    }
}

/// Channel kinds.
const SILENT: u8 = 0;
const CONSTANT: u8 = 1;
const PREDICTED: u8 = 2;

/// One channel of a block, appended to `out`.
fn encode_channel(x: &[f32], out: &mut Vec<u8>) {
    let peak = x
        .iter()
        .map(|v| if v.is_finite() { v.abs() } else { 0.0 })
        .fold(0.0f32, f32::max);
    if peak == 0.0 {
        out.push(SILENT);
        return;
    }
    let exponent = (peak.log2().ceil() as i32).clamp(*EXPONENTS.start(), *EXPONENTS.end());
    let scale = FULL / 2f32.powi(exponent);
    let q: Vec<i32> = x
        .iter()
        .map(|&v| {
            let v = if v.is_finite() { v } else { 0.0 };
            (v * scale).round().clamp(-FULL, FULL) as i32
        })
        .collect();
    if q.iter().all(|&v| v == q[0]) {
        out.push(CONSTANT);
        out.push(exponent as i8 as u8);
        out.extend_from_slice(&(q[0] as i16).to_le_bytes());
        return;
    }
    // The predictor leaving the least.
    let mut best = (u64::MAX, 0usize, Vec::new());
    let mut r = Vec::with_capacity(q.len());
    for order in 0..=3usize.min(q.len() - 1) {
        residuals(&q, order, &mut r);
        let cost: u64 = r.iter().map(|&v| u64::from(v.unsigned_abs())).sum();
        if cost < best.0 {
            best = (cost, order, r.clone());
        }
    }
    let (cost, order, r) = best;
    let k = rice_k(cost as f64 / r.len().max(1) as f64);
    out.push(PREDICTED);
    out.push(exponent as i8 as u8);
    out.push(order as u8);
    out.push(k as u8);
    for &w in &q[..order] {
        out.extend_from_slice(&(w as i16).to_le_bytes());
    }
    let mut bits = BitWriter::default();
    for &v in &r {
        let u = zigzag(v);
        let quotient = u >> k;
        if quotient < ESCAPE {
            bits.ones(quotient);
            bits.bit(false);
            bits.put(u & ((1 << k) - 1), k);
        } else {
            bits.ones(ESCAPE);
            bits.put(u, RAW_BITS);
        }
    }
    let bytes = bits.finish();
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&bytes);
}

/// One channel of a block of `frames` from `bytes` at `at`.
fn decode_channel(bytes: &[u8], at: &mut usize, frames: usize) -> Option<Vec<f32>> {
    let mut take = |n: usize| {
        let s = bytes.get(*at..*at + n)?;
        *at += n;
        Some(s)
    };
    let kind = take(1)?[0];
    if kind == SILENT {
        return Some(vec![0.0; frames]);
    }
    let exponent = i32::from(take(1)?[0] as i8);
    if !EXPONENTS.contains(&exponent) {
        return None;
    }
    let unit = 2f32.powi(exponent) / FULL;
    if kind == CONSTANT {
        let v = i16::from_le_bytes(take(2)?.try_into().ok()?);
        return Some(vec![f32::from(v) * unit; frames]);
    }
    if kind != PREDICTED {
        return None;
    }
    let order = usize::from(take(1)?[0]);
    let k = u32::from(take(1)?[0]);
    if order > 3 || order >= frames || k >= RAW_BITS {
        return None;
    }
    let mut q: Vec<i32> = Vec::with_capacity(frames);
    for _ in 0..order {
        q.push(i32::from(i16::from_le_bytes(take(2)?.try_into().ok()?)));
    }
    let len = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
    let mut bits = BitReader::new(take(len)?);
    while q.len() < frames {
        let quotient = bits.ones(ESCAPE)?;
        let u = if quotient < ESCAPE {
            (quotient << k) | bits.get(k)?
        } else {
            bits.get(RAW_BITS)?
        };
        let i = q.len();
        let p = match order {
            0 => 0,
            1 => q[i - 1],
            2 => 2 * q[i - 1] - q[i - 2],
            _ => 3 * q[i - 1] - 3 * q[i - 2] + q[i - 3],
        };
        q.push(p + unzigzag(u));
    }
    Some(q.into_iter().map(|v| v as f32 * unit).collect())
}

#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    word: u64,
    used: u32,
}

impl BitWriter {
    fn put(&mut self, value: u32, bits: u32) {
        if bits == 0 {
            return;
        }
        self.word = (self.word << bits) | u64::from(value & ((1u64 << bits) - 1) as u32);
        self.used += bits;
        while self.used >= 8 {
            self.used -= 8;
            self.bytes.push((self.word >> self.used) as u8);
        }
        self.word &= (1u64 << self.used) - 1;
    }

    fn bit(&mut self, one: bool) {
        self.put(u32::from(one), 1);
    }

    fn ones(&mut self, n: u32) {
        let mut left = n;
        while left > 0 {
            let take = left.min(24);
            self.put((1u32 << take) - 1, take);
            left -= take;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.used > 0 {
            let pad = 8 - self.used;
            self.put(0, pad);
        }
        self.bytes
    }
}

struct BitReader<'a> {
    bytes: &'a [u8],
    bit: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> BitReader<'a> {
        BitReader { bytes, bit: 0 }
    }

    fn one(&mut self) -> Option<bool> {
        let byte = *self.bytes.get(self.bit / 8)?;
        let b = (byte >> (7 - self.bit % 8)) & 1 == 1;
        self.bit += 1;
        Some(b)
    }

    /// Ones up to a zero (taken) or `most` of them (no zero taken).
    fn ones(&mut self, most: u32) -> Option<u32> {
        let mut n = 0;
        while n < most {
            if !self.one()? {
                return Some(n);
            }
            n += 1;
        }
        Some(n)
    }

    fn get(&mut self, bits: u32) -> Option<u32> {
        let mut v = 0u32;
        for _ in 0..bits {
            v = (v << 1) | u32::from(self.one()?);
        }
        Some(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn temp(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("jodel-stem-file-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Four stems of `frames`: a sine, noise-like, silence then a tone, and
    /// a loud one past full scale.
    fn stems(frames: usize) -> Stems {
        let mut s = empty();
        let mut seed = 1u32;
        for i in 0..frames {
            let t = i as f32 / RATE as f32;
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            let values = [
                0.5 * (t * 2.0 * std::f32::consts::PI * 110.0).sin(),
                0.3 * noise,
                if i < frames / 2 {
                    0.0
                } else {
                    0.001 * (t * 6000.0).sin()
                },
                1.7 * (t * 900.0).sin(),
            ];
            for (stem, &v) in values.iter().enumerate() {
                s[stem][0].push(v);
                s[stem][1].push(-v * 0.5);
            }
        }
        s
    }

    fn write(path: &Path, s: &Stems, piece: usize) {
        let mut w = StemWriter::create(path, 96_000, "test").unwrap();
        let n = s[0][0].len();
        let mut at = 0;
        while at < n {
            let end = (at + piece).min(n);
            let part: [[&[f32]; 2]; STEMS] =
                std::array::from_fn(|st| [&s[st][0][at..end], &s[st][1][at..end]]);
            w.push(part).unwrap();
            at = end;
        }
        w.finish().unwrap();
    }

    /// Written in odd pieces and read back at any frame, every value is
    /// within half a step of its channel's block scale; silence stays
    /// silent; the file is smaller than the values as 16 bits.
    #[test]
    fn stems_read_back_as_written_from_any_frame() {
        let dir = temp("round");
        let path = dir.join("track.stems");
        let frames = BLOCK * 5 + 1234;
        let s = stems(frames);
        write(&path, &s, 3000);
        let mut f = StemFile::open(&path).unwrap();
        assert_eq!(f.frames(), frames as u64);
        assert_eq!(f.source_rate(), 96_000);
        assert_eq!(f.separator(), "test");
        for &(from, len) in &[
            (0u64, frames),
            (4095, 10),
            (BLOCK as u64 * 3 + 7, 9000),
            (frames as u64 - 5, 100),
        ] {
            let back = f.read(from, len).unwrap();
            let n = back[0][0].len();
            assert_eq!(n as u64, (from + len as u64).min(frames as u64) - from);
            for stem in 0..STEMS {
                for ch in 0..2 {
                    for i in 0..n {
                        let want = s[stem][ch][from as usize + i];
                        let got = back[stem][ch][i];
                        // Half a step of the loudest stem's scale (2^1 for
                        // 1.7) at most.
                        assert!(
                            (want - got).abs() <= 2.0 / FULL,
                            "{stem} {ch} {i}: {want} {got}"
                        );
                    }
                }
            }
        }
        // The quiet half of the third stem: exact silence, then fine detail.
        let back = f.read(0, frames).unwrap();
        assert!(back[2][0][..frames / 2 - BLOCK].iter().all(|&v| v == 0.0));
        let quiet_error = (frames / 2 + BLOCK..frames)
            .map(|i| (s[2][0][i] - back[2][0][i]).abs())
            .fold(0.0f32, f32::max);
        assert!(quiet_error < 0.001 / 30_000.0, "{quiet_error}");
        let raw = frames as u64 * CHANNELS as u64 * 2;
        let size = fs::metadata(&path).unwrap().len();
        assert!(size < raw * 9 / 10, "{size} of {raw}");
        let _ = fs::remove_dir_all(&dir);
    }

    /// A file not finished is not left behind; one cut short, or of another
    /// version, is not read.
    #[test]
    fn only_whole_files_are_read() {
        let dir = temp("whole");
        let path = dir.join("track.stems");
        {
            let mut w = StemWriter::create(&path, 44_100, "test").unwrap();
            let s = stems(BLOCK * 2);
            let part: [[&[f32]; 2]; STEMS] =
                std::array::from_fn(|st| [&s[st][0][..], &s[st][1][..]]);
            w.push(part).unwrap();
            // Dropped without finish.
        }
        assert!(!path.exists());
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 0, "no temporary left");
        write(&path, &stems(BLOCK * 2 + 10), 5000);
        let bytes = fs::read(&path).unwrap();
        fs::write(&path, &bytes[..bytes.len() - 30]).unwrap();
        assert!(StemFile::open(&path).is_err());
        let mut other = bytes.clone();
        other[8] = 9;
        fs::write(&path, &other).unwrap();
        assert!(StemFile::open(&path).is_err());
        fs::write(&path, &bytes).unwrap();
        assert!(StemFile::open(&path).is_ok());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn residuals_round_trip_through_the_rice_code() {
        for (i, x) in [
            vec![0.25f32; 100],
            (0..BLOCK)
                .map(|i| ((i * 7919) % 2001) as f32 / 1000.0 - 1.0)
                .collect(),
        ]
        .iter()
        .enumerate()
        {
            let mut bytes = Vec::new();
            encode_channel(x, &mut bytes);
            let mut at = 0;
            let back = decode_channel(&bytes, &mut at, x.len()).unwrap();
            assert_eq!(at, bytes.len(), "case {i}");
            for (a, b) in x.iter().zip(&back) {
                assert!((a - b).abs() <= 1.0 / FULL, "case {i}: {a} {b}");
            }
        }
    }
}
