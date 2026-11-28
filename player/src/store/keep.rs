//! A song's singing kept (a take: a recording of a session), for tuning
//! the game on real singing: the voice
//! as the echo canceller cleaned it (as the game heard it), where in the
//! song each part of it was sung, and the song's lyrics and melody as the
//! game held it against them. Kept while Karaoke mode sings a song with
//! `karaoke_keep` on, saved when the song ends or the mode is left: a
//! folder `recordings/sing-<seconds>` beside the settings file, its files
//! ([`VOICE`], [`TIMES`], [`LYRICS`], [`MELODY`], [`ABOUT`]) written into
//! it under a temporary name and the folder named whole, durably
//! ([`durable::AtomicDir`]): a save cut short leaves at most a temporary,
//! which is no take ([`is_take`]) and is swept once a day old
//! ([`durable::STALE`]). The `sing_replay`
//! example reads one back. A song the demo key sang in
//! ([`crate::core::singing::Demo`]) is not saved: it was not all sung. Nor
//! is one without a voice in it (the microphone muted or silent, nothing
//! sung): under [`VOICED_ENOUGH`] of the pitch tracker's frames had a
//! pitch.

use std::fmt::Write as _;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use jodel_dsp::wav;

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
/// What was sung: the song, how late the voice came as the
/// game took it and where that came from (`late <sample> <seconds>
/// <source>`, [`crate::core::lag::Source::key`]), and the key the song was
/// heard in where it was not its own (`key <sample> <semitones>`, from that
/// sample on; [`read_keys`]).
pub const ABOUT: &str = "about.txt";

/// How a block of the voice was heard: how late it came as the game took
/// it, in seconds, where that came from, and the key the song was heard
/// in, in semitones from its own.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sung {
    /// How late, in seconds.
    pub late: f64,
    /// Where that came from ([`crate::core::lag::Source::key`]).
    pub source: &'static str,
    /// The key, in semitones from the song's own.
    pub key: i8,
}

impl Sung {
    /// How a block was heard: `late`, from `source`, in `key`.
    pub fn new(late: f64, source: &'static str, key: i8) -> Sung {
        Sung { late, source, key }
    }
}

/// The most seconds of voice kept for one song (an hour).
const MOST_SECONDS: usize = 3600;
/// A mark at least every this many samples' worth of seconds.
const MARK_EVERY: f64 = 1.0;
/// How far the delay moves before it is written down again, in seconds.
const LATE_STEP: f64 = 0.0005;
/// The least singing a song kept is saved with: seconds of the pitch
/// tracker's frames that had a pitch ([`jodel_rhythm::yin`]: a frame under
/// its quiet level has none). About two words sung; a room's noise or a
/// muted microphone gives a stray frame at most.
pub const VOICED_ENOUGH: f64 = 1.0;

/// A song's singing as it is kept.
pub struct Kept {
    /// The song.
    pub song: PathBuf,
    rate: u32,
    samples: Vec<i16>,
    marks: Vec<(usize, f64)>,
    lyrics: Arc<jodel_lyrics::Lyrics>,
    waveform: Arc<jodel_rhythm::waveform::Waveform>,
    /// How late the voice came, as the game took it, each time it changed:
    /// the sample, the seconds, where that came from.
    late: Vec<(usize, f64, &'static str)>,
    /// The key the song was heard in, each time it changed: the sample,
    /// the semitones from its own.
    keys: Vec<(usize, i8)>,
    /// The pitch tracker's frames of the voice that had a pitch.
    voiced: usize,
    /// The demo key sang in the song.
    demo: bool,
}

impl Kept {
    /// For `song`, sung against `lyrics` and `waveform`.
    pub fn new(
        song: PathBuf,
        lyrics: Arc<jodel_lyrics::Lyrics>,
        waveform: Arc<jodel_rhythm::waveform::Waveform>,
    ) -> Kept {
        Kept {
            song,
            rate: 0,
            samples: Vec::new(),
            marks: Vec::new(),
            lyrics,
            waveform,
            late: Vec::new(),
            keys: Vec::new(),
            voiced: 0,
            demo: false,
        }
    }

    /// A block of `samples` at `rate`, its first sung at `at` seconds into
    /// the song; `continues`: right after the block before; `how`: how it
    /// is heard now.
    pub fn push(&mut self, samples: &[f32], rate: u32, at: f64, continues: bool, how: Sung) {
        let (late, key) = ((how.late, how.source), how.key);
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
        // Mid-song it moves smoothly: a line each half millisecond moved.
        if self
            .late
            .last()
            .is_none_or(|&(_, l, s)| (l - late.0).abs() >= LATE_STEP || s != late.1)
        {
            self.late.push((n, late.0, late.1));
        }
        if self.keys.last().map_or(key != 0, |&(_, k)| k != key) {
            self.keys.push((n, key));
        }
        self.samples.extend(
            samples
                .iter()
                .map(|&x| (x.clamp(-1.0, 1.0) * 32767.0).round() as i16),
        );
    }

    /// `frames` more of the pitch tracker's frames of the voice had a pitch.
    pub fn voiced(&mut self, frames: usize) {
        self.voiced += frames;
    }

    /// The demo key sang in the song: it is not saved.
    pub fn sung_by_demo(&mut self) {
        self.demo = true;
    }

    /// Whether it is to be saved: a voice was heard in it (at least
    /// [`VOICED_ENOUGH`] of frames with a pitch), and the demo key sang
    /// none of the song.
    pub fn worth_saving(&self) -> bool {
        let voiced = self.voiced as f64 * jodel_rhythm::yin::HOP;
        !self.samples.is_empty() && voiced >= VOICED_ENOUGH && !self.demo
    }

    /// Saves it into a new folder in `dir`, whole or not at all, durably;
    /// its path. What an earlier save cut short left goes first, once old.
    pub fn save(&self, dir: &Path) -> io::Result<PathBuf> {
        durable::sweep(dir, durable::STALE);
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let mut folder = dir.join(format!("{TAKE}{stamp}"));
        let mut k = 1;
        while folder.exists() {
            folder = dir.join(format!("{TAKE}{stamp}-{k}"));
            k += 1;
        }
        // Cut short (an error, a panic), the take goes with what is in it.
        let take = durable::AtomicDir::create(&folder)?;
        self.write_into(take.temporary_path())?;
        take.commit()?;
        Ok(folder)
    }

    /// Writes its files into `folder`.
    fn write_into(&self, folder: &Path) -> io::Result<()> {
        durable::write_atomic(
            &folder.join(VOICE),
            &wav::pcm16(self.rate, 1, &self.samples),
        )?;
        let mut times = String::new();
        for (k, t) in &self.marks {
            let _ = writeln!(times, "{k} {t:.6}");
        }
        durable::write_atomic(&folder.join(TIMES), times.as_bytes())?;
        durable::write_atomic(&folder.join(LYRICS), self.lyrics.to_lrc().as_bytes())?;
        durable::write_atomic(&folder.join(MELODY), &self.waveform.to_bytes())?;
        let mut about = String::new();
        let _ = writeln!(about, "song {}", self.song.display());
        let _ = writeln!(about, "rate {}", self.rate);
        for (k, l, source) in &self.late {
            let _ = writeln!(about, "late {k} {l:.4} {source}");
        }
        for (k, key) in &self.keys {
            let _ = writeln!(about, "key {k} {key}");
        }
        durable::write_atomic(&folder.join(ABOUT), about.as_bytes())
    }
}

/// How a kept song's folder is named: `sing-<seconds>`.
const TAKE: &str = "sing-";

/// Whether `folder` is a kept song, saved whole: named as one, not a
/// temporary a save cut short left.
pub fn is_take(folder: &Path) -> bool {
    folder
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with(TAKE) && !durable::is_temporary(n))
}

/// A mono 16-bit WAV file's samples (±1) and rate, as a take's voice is
/// kept ([`jodel_dsp::wav::pcm16`]).
pub fn read_wav(bytes: &[u8]) -> Option<(Vec<f32>, u32)> {
    let file = wav::parse(bytes)?;
    let samples = file.pcm16()?;
    Some((
        samples.iter().map(|&s| f32::from(s) / 32767.0).collect(),
        file.format.rate,
    ))
}

/// The key a kept song was heard in ([`ABOUT`]'s `key` lines): from each
/// sample on, the semitones from its own (none: its own all along).
pub fn read_keys(about: &str) -> Vec<(usize, i8)> {
    about
        .lines()
        .filter_map(|l| {
            let mut it = l.strip_prefix("key ")?.split_whitespace();
            Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
        })
        .collect()
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

    /// A song's singing as kept, against `la la`.
    fn kept() -> Kept {
        let lyrics = Arc::new(jodel_lyrics::Lyrics::parse(
            "la la",
            jodel_lyrics::Source::Unknown,
        ));
        let waveform = Arc::new(jodel_rhythm::waveform::Waveform::default());
        Kept::new("/a.flac".into(), lyrics, waveform)
    }

    #[test]
    fn the_voice_goes_to_wav_and_back_with_its_times() {
        let mut kept = kept();
        kept.push(&[0.5; 100], 1000, 2.0, false, Sung::new(0.15, "default", 0));
        kept.push(&[-0.5; 100], 1000, 2.1, true, Sung::new(0.15, "default", 2));
        kept.push(
            &[0.25; 100],
            1000,
            5.0,
            false,
            Sung::new(0.16, "singing", 2),
        );
        // A mark where it starts, where it jumped; the later block runs on.
        assert_eq!(kept.marks, [(0, 2.0), (200, 5.0)]);
        assert_eq!(kept.late, [(0, 0.15, "default"), (200, 0.16, "singing")]);
        // The key from where it changed, back as it was written.
        assert_eq!(kept.keys, [(100, 2)]);
        let dir = jodel_testkit::TempDir::new("keep-keys");
        let folder = kept.save(&dir).unwrap();
        let about = std::fs::read_to_string(folder.join(ABOUT)).unwrap();
        assert_eq!(read_keys(&about), [(100, 2)]);
        let (back, rate) = read_wav(&wav::pcm16(kept.rate, 1, &kept.samples)).unwrap();
        assert_eq!((back.len(), rate), (300, 1000));
        assert!((back[0] - 0.5).abs() < 1e-3 && (back[150] + 0.5).abs() < 1e-3);
        assert_eq!(read_times("0 2.000000\n200 5.000000\n"), kept.marks);
    }

    #[test]
    fn a_save_cut_short_leaves_no_take() {
        let mut kept = kept();
        kept.push(&[0.5; 100], 1000, 2.0, false, Sung::new(0.15, "default", 0));
        let dir = jodel_testkit::TempDir::new("keep-cut");
        let takes = || -> Vec<PathBuf> {
            std::fs::read_dir(&dir)
                .unwrap()
                .flatten()
                .map(|e| e.path())
                .filter(|p| is_take(p))
                .collect()
        };
        // Cut short by an error: nothing left.
        let take = durable::AtomicDir::create(&dir.join("sing-1")).unwrap();
        kept.write_into(take.temporary_path()).unwrap();
        drop(take);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        // Cut short by a crash (nothing dropped): its temporary is no take.
        let take = durable::AtomicDir::create(&dir.join("sing-2")).unwrap();
        kept.write_into(take.temporary_path()).unwrap();
        let part = take.temporary_path().to_path_buf();
        std::mem::forget(take);
        assert!(!is_take(&part));
        assert_eq!(takes(), Vec::<PathBuf>::new());
        // The next save is a take, whole; the one cut short goes once old.
        let folder = kept.save(&dir).unwrap();
        assert_eq!(takes(), std::slice::from_ref(&folder));
        for name in [VOICE, TIMES, LYRICS, MELODY, ABOUT] {
            assert!(folder.join(name).is_file(), "{name}");
        }
        assert_eq!(durable::sweep(&dir, std::time::Duration::ZERO), 1);
        assert!(!part.exists());
    }

    /// The pitch tracker's frames in `seconds`.
    fn frames(seconds: f64) -> usize {
        (seconds / jodel_rhythm::yin::HOP).round() as usize
    }

    #[test]
    fn a_take_without_a_voice_is_not_kept() {
        // The microphone muted: a second heard, silence, no pitch.
        let mut kept = kept();
        kept.push(
            &[0.0; 16_000],
            16_000,
            2.0,
            false,
            Sung::new(0.15, "default", 0),
        );
        assert!(!kept.worth_saving(), "nothing sung");
        // A stray frame or two of the room's noise is no singing.
        kept.voiced(3);
        assert!(!kept.worth_saving(), "a stray pitch");
        // Sung: kept.
        kept.voiced(frames(VOICED_ENOUGH));
        assert!(kept.worth_saving());
    }

    #[test]
    fn a_song_the_demo_sang_in_is_not_kept() {
        let mut kept = kept();
        assert!(!kept.worth_saving(), "nothing heard");
        kept.push(&[0.5; 100], 1000, 2.0, false, Sung::new(0.15, "default", 0));
        kept.voiced(frames(VOICED_ENOUGH));
        assert!(kept.worth_saving());
        kept.sung_by_demo();
        kept.push(&[0.5; 100], 1000, 9.0, false, Sung::new(0.15, "default", 0));
        assert!(!kept.worth_saving(), "the demo sang in it");
    }
}
