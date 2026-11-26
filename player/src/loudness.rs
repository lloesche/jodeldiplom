//! Each song's loudness as the karaoke plays it, for the game's sounds to
//! follow ([`crate::sfx::conductor::calibration`]): measured once, on a
//! thread of its own (never the audio's, never holding up play), when a
//! song is sung with the game on and has none kept, and kept in
//! `loudness` in the data folder ([`crate::cache::LOUDNESS`]), a small
//! file a song, named by a hash of its path, size and modification time.
//! Nothing else is analysed again for it.
//!
//! What the karaoke plays is the song's stems, the voice at the Karaoke
//! tab's default level (the instrumental) where they are kept; else the
//! file as it is, which is what plays without stems. It is measured as
//! ITU-R BS.1770 has it ([`jodel_rhythm::loudness`]): the kept stems at
//! their rate, else the file at its own (a mono one on both sides). One
//! measured from the file is measured again once the song's stems are
//! kept, and one from the stems again when the voice's default level is
//! another; a file kept in another schema version is measured again and
//! written over.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use jodel_playback::stems::{Stem, RATE};
use jodel_rhythm::loudness::Meter;

use crate::browser::Notify;
use crate::cache::LOUDNESS;

/// What a song's loudness was measured from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Played {
    /// Its stems, the voice at this level (dB).
    Stems {
        /// The voice's level.
        voice_db: f64,
    },
    /// The file as it is.
    File,
}

/// A song's loudness as measured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Measured {
    /// Its integrated loudness, LUFS (none: silent).
    pub lufs: Option<f64>,
    /// What it was measured from.
    pub from: Played,
}

/// The voice's level the karaoke plays a song at as it starts: the Karaoke
/// tab's default (dB).
pub fn default_voice_db() -> f64 {
    crate::core::sliders::voice_db(crate::config::Config::default().karaoke_voice)
}

/// `measured` as the file holds it, its header in front.
pub fn to_text(measured: &Measured) -> String {
    let lufs = measured
        .lufs
        .map_or_else(|| "none".to_owned(), |l| format!("{l:.3}"));
    let from = match measured.from {
        Played::Stems { voice_db } => format!("stems voice {voice_db}"),
        Played::File => "file".to_owned(),
    };
    LOUDNESS.wrap(&format!("lufs {lufs}\nfrom {from}\n"))
}

/// The loudness in a file's `text`: none for another schema version, or
/// one not read.
pub fn from_text(text: &str) -> Option<Measured> {
    let mut lines = LOUDNESS.body(text)?.lines();
    let lufs = match lines.next()?.strip_prefix("lufs ")? {
        "none" => None,
        l => Some(l.parse().ok().filter(|l: &f64| l.is_finite())?),
    };
    let from = match lines.next()?.strip_prefix("from ")? {
        "file" => Played::File,
        stems => Played::Stems {
            voice_db: stems.strip_prefix("stems voice ")?.parse().ok()?,
        },
    };
    Some(Measured { lufs, from })
}

/// The folder the files are kept in: `loudness` beside the analyses'
/// folder `analyses`.
pub fn beside(analyses: &Path) -> PathBuf {
    analyses.parent().unwrap_or(analyses).join("loudness")
}

/// The file of `path` in `dir` (whether or not it is there); `None` when
/// the song cannot be read.
pub fn file_for(dir: &Path, path: &Path) -> Option<PathBuf> {
    let key = format!("{} loudness", crate::cache::identity(path)?);
    Some(dir.join(format!("{}.txt", crate::cache::name(&key))))
}

/// The loudness kept for `path` in `dir`, where there is one of this
/// version.
pub fn load(dir: &Path, path: &Path) -> Option<Measured> {
    let file = crate::mover::locate(&file_for(dir, path)?);
    from_text(&std::fs::read_to_string(file).ok()?)
}

/// `measured` kept for `path` in `dir`, durably.
pub fn store(dir: &Path, path: &Path, measured: &Measured) -> std::io::Result<()> {
    let dir = crate::mover::current(dir);
    let file = file_for(&dir, path).ok_or_else(|| std::io::Error::other("the song is gone"))?;
    durable::write_atomic(&file, to_text(measured).as_bytes())
}

/// Whether `kept` still says what the karaoke plays of the song: measured
/// from its stems at the voice's default level, or from the file while no
/// stems are kept.
fn current(kept: &Measured, stems_kept: bool) -> bool {
    match kept.from {
        Played::Stems { voice_db } => voice_db == default_voice_db(),
        Played::File => !stems_kept,
    }
}

/// The loudness of the stems in `file`, the voice at `voice` (a gain).
fn of_stems(
    file: &mut jodel_playback::stem_file::StemFile,
    voice: f32,
) -> Result<Measured, String> {
    let mut meter = Meter::new(RATE);
    let (mut left, mut right) = (Vec::new(), Vec::new());
    let mut at = 0;
    while at < file.frames() {
        let stems = file
            .read(at, RATE as usize * 10)
            .map_err(|e| e.to_string())?;
        let n = stems[Stem::Bass][0].len();
        if n == 0 {
            break;
        }
        at += n as u64;
        for (side, out) in [&mut left, &mut right].into_iter().enumerate() {
            out.clear();
            out.resize(n, 0.0);
            for (s, stem) in stems.iter() {
                let gain = if s == Stem::Vocals { voice } else { 1.0 };
                for (o, x) in out.iter_mut().zip(&stem[side]) {
                    *o += gain * x;
                }
            }
        }
        meter.push(&left, &right);
    }
    Ok(Measured {
        lufs: meter.integrated(),
        from: Played::Stems {
            voice_db: default_voice_db(),
        },
    })
}

/// The loudness of the file `path` as it is, at its own rate.
fn of_file(path: &Path) -> Result<Measured, String> {
    let mut source = jodel_playback::Source::open(path).map_err(|e| e.to_string())?;
    source.make_stereo();
    let mut meter = Meter::new(source.info().rate);
    // The decoder's samples are in 24-bit units.
    let scale = 1.0 / jodel_dsp::FULL_SCALE;
    let (mut block, mut left, mut right) = (Vec::new(), Vec::new(), Vec::new());
    loop {
        block.clear();
        let frames = source
            .read_unquantized(&mut block)
            .map_err(|e| e.to_string())?;
        if frames == 0 {
            break;
        }
        left.clear();
        right.clear();
        for frame in block.as_chunks::<2>().0 {
            left.push((frame[0] * scale) as f32);
            right.push((frame[1] * scale) as f32);
        }
        meter.push(&left, &right);
    }
    Ok(Measured {
        lufs: meter.integrated(),
        from: Played::File,
    })
}

/// Where the worker reads and keeps: the loudness files and the kept
/// stems (none: nothing kept, the stems not read).
#[derive(Clone, Debug, Default)]
pub struct Folders {
    /// The loudness files.
    pub loudness: Option<PathBuf>,
    /// The stems.
    pub stems: Option<PathBuf>,
}

impl Folders {
    /// Those beside the analyses' folder `analyses`.
    pub fn beside(analyses: &Path) -> Folders {
        Folders {
            loudness: Some(beside(analyses)),
            stems: Some(crate::stems_cache::beside(analyses)),
        }
    }
}

/// The loudness of `path` as the karaoke plays it: kept in `folders`
/// while it still says so ([`current`]), else measured and kept.
pub fn loudness_of(folders: &Folders, path: &Path) -> Result<Measured, String> {
    let stems = folders.stems.as_deref();
    let stems_kept = stems.is_some_and(|d| crate::stems_cache::is_kept(d, path));
    let dir = folders.loudness.as_deref();
    if let Some(kept) = dir.and_then(|d| load(d, path)) {
        if current(&kept, stems_kept) {
            return Ok(kept);
        }
    }
    let kept_stems = stems.and_then(|d| crate::stems_cache::open_waiting(d, path));
    let measured = match kept_stems {
        Some(mut file) => {
            let voice = crate::core::desk::taper::gain(default_voice_db()) as f32;
            of_stems(&mut file, voice)?
        }
        None => of_file(path)?,
    };
    if let Some(dir) = dir {
        if let Err(e) = store(dir, path, &measured) {
            crate::trace::note(format!("loudness: not kept for {}: {e}", path.display()));
        }
    }
    Ok(measured)
}

/// A song's loudness found (none: it could not be read), LUFS.
pub struct Found {
    /// The song.
    pub path: PathBuf,
    /// Its loudness as played (none: silent, or not read).
    pub lufs: Option<f64>,
}

/// The thread measuring songs' loudness, the last asked for at a time.
pub struct LoudnessWorker {
    tx: Sender<PathBuf>,
    rx: Receiver<Found>,
    /// What was last asked for.
    asked: Option<PathBuf>,
}

impl LoudnessWorker {
    /// A worker reading and keeping in `folders`, calling `notify` as each
    /// is done.
    pub fn new(folders: Folders, notify: Notify) -> LoudnessWorker {
        let (tx, wants) = mpsc::channel::<PathBuf>();
        let (found, rx) = mpsc::channel();
        thread::spawn(move || {
            while let Ok(mut path) = wants.recv() {
                // Only the last asked for.
                while let Ok(later) = wants.try_recv() {
                    path = later;
                }
                let started = std::time::Instant::now();
                let lufs = match loudness_of(&folders, &path) {
                    Ok(m) => {
                        crate::trace::note(format!(
                            "loudness: {} at {} as played ({:?}), in {:.1} s",
                            path.display(),
                            m.lufs
                                .map_or_else(|| "silence".to_owned(), |l| format!("{l:.1} LUFS")),
                            m.from,
                            started.elapsed().as_secs_f64()
                        ));
                        m.lufs
                    }
                    Err(e) => {
                        crate::trace::note(format!("loudness: none for {}: {e}", path.display()));
                        None
                    }
                };
                if found.send(Found { path, lufs }).is_err() {
                    return;
                }
                notify();
            }
        });
        LoudnessWorker {
            tx,
            rx,
            asked: None,
        }
    }

    /// The loudness wanted for `path` (asked once in a row).
    pub fn want(&mut self, path: &Path) {
        if self.asked.as_deref() == Some(path) {
            return;
        }
        self.asked = Some(path.to_owned());
        let _ = self.tx.send(path.to_owned());
    }

    /// A loudness found since the last call, if one was.
    pub fn take(&self) -> Option<Found> {
        self.rx.try_recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_kept_loudness_reads_back_and_says_what_it_was_measured_from() {
        for measured in [
            Measured {
                lufs: Some(-14.96),
                from: Played::Stems {
                    voice_db: f64::NEG_INFINITY,
                },
            },
            Measured {
                lufs: None,
                from: Played::File,
            },
        ] {
            assert_eq!(from_text(&to_text(&measured)), Some(measured));
        }
        assert_eq!(from_text("lufs -9.0\nfrom file\n"), None);
    }

    /// One from the file holds while no stems are kept; one from the stems
    /// while the voice's default level is the one it was measured at.
    #[test]
    fn a_kept_loudness_is_measured_again_when_it_no_longer_says_what_plays() {
        let file = Measured {
            lufs: Some(-9.0),
            from: Played::File,
        };
        assert!(current(&file, false));
        assert!(!current(&file, true));
        let stems = |voice_db| Measured {
            lufs: Some(-11.0),
            from: Played::Stems { voice_db },
        };
        assert!(current(&stems(default_voice_db()), true));
        assert!(!current(&stems(-6.0), true));
    }

    /// A song is measured from its file (no stems kept), kept, and read
    /// back from its file in the folder after.
    #[test]
    fn a_song_is_measured_once_and_kept() {
        let dir = std::env::temp_dir().join(format!("jodel-loudness-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let song = dir.join("sine.wav");
        // Five seconds of a stereo sine at 1 kHz peaking at -20 dBFS, as
        // 16-bit WAV.
        let frames: u32 = 5 * 48_000;
        let samples: Vec<i16> = (0..frames)
            .flat_map(|i| {
                let x = 0.1 * (std::f64::consts::TAU * 1_000.0 * f64::from(i) / 48_000.0).sin();
                let s = (x * 32_767.0).round() as i16;
                [s, s]
            })
            .collect();
        durable::write_atomic(&song, &jodel_dsp::wav::pcm16(48_000, 2, &samples)).unwrap();
        let folders = Folders {
            loudness: Some(dir.join("loudness")),
            stems: None,
        };
        let m = loudness_of(&folders, &song).unwrap();
        assert_eq!(m.from, Played::File);
        assert!((m.lufs.unwrap() + 20.0).abs() < 0.05, "{m:?}");
        let kept = load(&dir.join("loudness"), &song).unwrap();
        assert_eq!(kept.from, m.from);
        assert!(
            (kept.lufs.unwrap() - m.lufs.unwrap()).abs() < 1e-3,
            "{kept:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
