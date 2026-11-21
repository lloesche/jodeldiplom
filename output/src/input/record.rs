//! A recording of what the echo canceller works with
//! ([`super::Input::record_echo`]): its blocks written as a WAV file of
//! floats.

use super::capture::MicShared;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

/// A recording of the canceller's channels ([`Input::record_echo`]).
pub(super) const RECORD_CHANNELS: usize = 3;
/// What a recording of the canceller queues at most: two seconds of its
/// channels at its highest rate.
pub(super) const RECORD_QUEUE: usize = RECORD_CHANNELS * 56_000 * 2;

/// A recording of the echo canceller ([`Input::record_echo`]).
pub struct EchoRecording {
    pub(super) thread: Option<std::thread::JoinHandle<Result<PathBuf, String>>>,
}

impl EchoRecording {
    /// The file written, once done (`Some` once, then `None`).
    pub fn done(&mut self) -> Option<Result<PathBuf, String>> {
        if !self.thread.as_ref()?.is_finished() {
            return None;
        }
        let thread = self.thread.take()?;
        Some(
            thread
                .join()
                .unwrap_or_else(|_| Err("the recording stopped".into())),
        )
    }
}

/// The bytes of [`wav_header`].
const WAV_HEADER: usize = 80;

/// A WAV header for `frames` of `channels` channels of 32-bit float at
/// `rate`, as players everywhere read one (macOS's Quick Look among them):
/// WAVE_FORMAT_EXTENSIBLE, saying which speaker each channel is for (front
/// left, front right, centre, then on), and a `fact` chunk with the frame
/// count, which a WAV file of floats should have. The plain 16-byte header
/// of float data on three channels was refused by some.
fn wav_header(channels: u16, rate: u32, frames: u64) -> Vec<u8> {
    /// KSDATAFORMAT_SUBTYPE_IEEE_FLOAT.
    const FLOAT: [u8; 16] = [
        0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B,
        0x71,
    ];
    let block = u32::from(channels) * 4;
    let data = (frames * u64::from(block)).min(u64::from(u32::MAX - WAV_HEADER as u32)) as u32;
    let mask: u32 = if channels >= 32 {
        u32::MAX
    } else {
        (1u32 << channels) - 1
    };
    let mut h = Vec::with_capacity(WAV_HEADER);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&(WAV_HEADER as u32 - 8 + data).to_le_bytes());
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
    h.extend_from_slice(&mask.to_le_bytes());
    h.extend_from_slice(&FLOAT);
    h.extend_from_slice(b"fact");
    h.extend_from_slice(&4u32.to_le_bytes());
    h.extend_from_slice(&(frames.min(u64::from(u32::MAX)) as u32).to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&data.to_le_bytes());
    debug_assert_eq!(h.len(), WAV_HEADER);
    h
}

/// Writes the canceller's queued blocks to `path` as they come, for
/// `seconds` of them (or until none comes for three seconds), under a
/// temporary name that becomes `path` only once the recording is whole
/// ([`durable::AtomicFile`]).
pub(super) fn write_recording(shared: &MicShared, path: &Path, seconds: f64) -> Result<(), String> {
    use std::io::{Seek, SeekFrom, Write};
    let fail = |e: std::io::Error| format!("{}: {e}", path.display());
    let ring = &shared.record;
    // What was queued before is not this recording's.
    ring.skip(ring.len());
    let mut file = durable::AtomicFile::create(path).map_err(fail)?;
    file.write_all(&[0u8; WAV_HEADER]).map_err(fail)?;
    let mut buf = vec![0.0f32; RECORD_CHANNELS * 4096];
    let mut bytes = Vec::with_capacity(buf.len() * 4);
    let mut frames = 0u64;
    let mut rate = 0u32;
    let mut last = Instant::now();
    loop {
        let n = ring.pop(&mut buf);
        if n == 0 {
            if last.elapsed() > Duration::from_secs(3) {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
            continue;
        }
        last = Instant::now();
        if rate == 0 {
            rate = shared.echo_rate.load(Ordering::Relaxed);
        }
        bytes.clear();
        for v in &buf[..n] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        file.write_all(&bytes).map_err(fail)?;
        frames += (n / RECORD_CHANNELS) as u64;
        if rate > 0 && frames as f64 >= seconds * f64::from(rate) {
            break;
        }
    }
    if frames == 0 {
        // Dropped: the temporary goes, nothing takes the name.
        return Err("nothing came from the echo canceller (is music playing?)".into());
    }
    // The header, now the length is known; then the name.
    file.seek(SeekFrom::Start(0)).map_err(fail)?;
    file.write_all(&wav_header(RECORD_CHANNELS as u16, rate, frames))
        .map_err(fail)?;
    file.commit().map_err(fail)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_recording_has_the_header_players_read() {
        let h = wav_header(3, 44_100, 1000);
        assert_eq!(h.len(), WAV_HEADER);
        let u16_at = |i: usize| u16::from_le_bytes([h[i], h[i + 1]]);
        let u32_at = |i: usize| u32::from_le_bytes([h[i], h[i + 1], h[i + 2], h[i + 3]]);
        assert_eq!(&h[0..4], b"RIFF");
        // The file's length less 8: the header's 72 and the samples'.
        assert_eq!(u32_at(4), 72 + 1000 * 12);
        assert_eq!(&h[8..16], b"WAVEfmt ");
        assert_eq!(u32_at(16), 40);
        assert_eq!(u16_at(20), 0xFFFE);
        assert_eq!((u16_at(22), u32_at(24)), (3, 44_100));
        assert_eq!((u32_at(28), u16_at(32), u16_at(34)), (44_100 * 12, 12, 32));
        assert_eq!((u16_at(36), u16_at(38)), (22, 32));
        // Front left, front right, centre.
        assert_eq!(u32_at(40), 0b111);
        // The float sub-format, then the frames in `fact`, then the data.
        assert_eq!(&h[44..48], &[3, 0, 0, 0]);
        assert_eq!(&h[56..60], &[0x00, 0x38, 0x9B, 0x71]);
        assert_eq!(&h[60..64], b"fact");
        assert_eq!((u32_at(64), u32_at(68)), (4, 1000));
        assert_eq!(&h[72..76], b"data");
        assert_eq!(u32_at(76), 1000 * 12);
        // A file of it, for a reader of its own (`JODEL_WAV_SAMPLE`).
        if let Some(out) = std::env::var_os("JODEL_WAV_SAMPLE") {
            let mut bytes = h.clone();
            for i in 0..3000u32 {
                bytes.extend_from_slice(&((i % 3) as f32 * 0.25).to_le_bytes());
            }
            std::fs::write(out, bytes).unwrap();
        }
    }
}
