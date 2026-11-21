//! A song's singing kept, for tuning the game on real singing: the voice
//! as the echo canceller cleaned it (as the game heard it), where in the
//! song each part of it was sung, and the song's lyrics and melody as the
//! game held it against them. Kept while Karaoke mode sings a song with
//! `karaoke_keep` on, saved when the song ends or the mode is left: a
//! folder `recordings/sing-<seconds>` beside the settings file, its files
//! ([`VOICE`], [`TIMES`], [`LYRICS`], [`MELODY`], [`ABOUT`]) each written
//! durably. The `sing_replay` example reads one back.

use std::fmt::Write as _;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The voice: mono, 16-bit WAV at the rate it was heard at.
pub const VOICE: &str = "voice.wav";
/// Where in the song the voice was sung: a line a mark, the sample (of
/// [`VOICE`]) and the song's time then in seconds (as the music was
/// played, not yet less how late the microphone hears it).
pub const TIMES: &str = "times.txt";
/// The lyrics with our word times (Enhanced LRC).
pub const LYRICS: &str = "lyrics.lrc";
/// The song's waveform and melody, as the analysis keeps them.
pub const MELODY: &str = "song.wave";
/// What was sung: the song, the level, how late the microphone heard.
pub const ABOUT: &str = "about.txt";

/// The most seconds of voice kept for one song (an hour).
const MOST_SECONDS: usize = 3600;
/// A mark at least every this many samples' worth of seconds.
const MARK_EVERY: f64 = 1.0;

/// A song's singing as it is kept.
pub struct Kept {
    /// The song.
    pub song: PathBuf,
    rate: u32,
    samples: Vec<i16>,
    marks: Vec<(usize, f64)>,
    lyrics: Arc<jodel_lyrics::Lyrics>,
    waveform: Arc<jodel_rhythm::waveform::Waveform>,
    level: &'static str,
    /// How late the microphone heard the music, as the game took it, each
    /// time it changed: the sample, the seconds.
    late: Vec<(usize, f64)>,
}

impl Kept {
    /// For `song`, sung at `level` against `lyrics` and `waveform`.
    pub fn new(
        song: PathBuf,
        lyrics: Arc<jodel_lyrics::Lyrics>,
        waveform: Arc<jodel_rhythm::waveform::Waveform>,
        level: &'static str,
    ) -> Kept {
        Kept {
            song,
            rate: 0,
            samples: Vec::new(),
            marks: Vec::new(),
            lyrics,
            waveform,
            level,
            late: Vec::new(),
        }
    }

    /// A block of `samples` at `rate`, its first sung at `at` seconds into
    /// the song; `continues`: right after the block before. `late`: how
    /// late the microphone hears the music now.
    pub fn push(&mut self, samples: &[f32], rate: u32, at: f64, continues: bool, late: f64) {
        if self.rate == 0 {
            self.rate = rate;
        }
        if rate != self.rate || self.samples.len() >= MOST_SECONDS * rate as usize {
            return;
        }
        let n = self.samples.len();
        let due = self
            .marks
            .last()
            .is_none_or(|&(k, _)| (n - k) as f64 >= MARK_EVERY * f64::from(rate));
        if !continues || due {
            self.marks.push((n, at));
        }
        if self
            .late
            .last()
            .is_none_or(|&(_, l)| (l - late).abs() > 1e-4)
        {
            self.late.push((n, late));
        }
        self.samples.extend(
            samples
                .iter()
                .map(|&x| (x.clamp(-1.0, 1.0) * 32767.0).round() as i16),
        );
    }

    /// Whether anything was heard.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Saves it into a new folder in `dir`, durably; its path.
    pub fn save(&self, dir: &Path) -> io::Result<PathBuf> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let mut folder = dir.join(format!("sing-{stamp}"));
        let mut k = 1;
        while folder.exists() {
            folder = dir.join(format!("sing-{stamp}-{k}"));
            k += 1;
        }
        std::fs::create_dir_all(&folder)?;
        durable::write_atomic(&folder.join(VOICE), &wav(&self.samples, self.rate))?;
        let mut times = String::new();
        for (k, t) in &self.marks {
            let _ = writeln!(times, "{k} {t:.6}");
        }
        durable::write_atomic(&folder.join(TIMES), times.as_bytes())?;
        durable::write_atomic(&folder.join(LYRICS), self.lyrics.to_lrc().as_bytes())?;
        durable::write_atomic(&folder.join(MELODY), &self.waveform.to_bytes())?;
        let mut about = String::new();
        let _ = writeln!(about, "song {}", self.song.display());
        let _ = writeln!(about, "level {}", self.level);
        let _ = writeln!(about, "rate {}", self.rate);
        for (k, l) in &self.late {
            let _ = writeln!(about, "late {k} {l:.4}");
        }
        durable::write_atomic(&folder.join(ABOUT), about.as_bytes())?;
        Ok(folder)
    }
}

/// `samples` at `rate` as a mono 16-bit WAV file.
pub fn wav(samples: &[i16], rate: u32) -> Vec<u8> {
    let data = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data.to_le_bytes());
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// A mono 16-bit WAV file's samples (±1) and rate, as [`wav`] writes them.
pub fn read_wav(bytes: &[u8]) -> Option<(Vec<f32>, u32)> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    let rate = u32::from_le_bytes(bytes[24..28].try_into().ok()?);
    let data = bytes.get(44..)?;
    Some((
        data.as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from(i16::from_le_bytes(*b)) / 32767.0)
            .collect(),
        rate,
    ))
}

/// A kept song's marks ([`TIMES`]): sample, seconds.
pub fn read_times(text: &str) -> Vec<(usize, f64)> {
    text.lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_voice_goes_to_wav_and_back_with_its_times() {
        let lyrics = Arc::new(jodel_lyrics::Lyrics::parse(
            "la la",
            jodel_lyrics::Source::Unknown,
        ));
        let waveform = Arc::new(jodel_rhythm::waveform::Waveform::default());
        let mut kept = Kept::new("/a.flac".into(), lyrics, waveform, "easy");
        kept.push(&[0.5; 100], 1000, 2.0, false, 0.05);
        kept.push(&[-0.5; 100], 1000, 2.1, true, 0.05);
        kept.push(&[0.25; 100], 1000, 5.0, false, 0.06);
        // A mark where it starts, where it jumped; the later block runs on.
        assert_eq!(kept.marks, [(0, 2.0), (200, 5.0)]);
        assert_eq!(kept.late, [(0, 0.05), (200, 0.06)]);
        let (back, rate) = read_wav(&wav(&kept.samples, kept.rate)).unwrap();
        assert_eq!((back.len(), rate), (300, 1000));
        assert!((back[0] - 0.5).abs() < 1e-3 && (back[150] + 0.5).abs() < 1e-3);
        assert_eq!(read_times("0 2.000000\n200 5.000000\n"), kept.marks);
    }
}
