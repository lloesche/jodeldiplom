//! The original singer's phrases ([`crate::core::shine::reference`]) for
//! the song in Karaoke mode, worked out on a thread of their own (never
//! the audio's, never holding up play) and kept in `shine` beside the
//! analyses ([`crate::cache::SHINE`]), a file a song's notes: named by a
//! hash of its path, its words as timed and its melody; read only in the
//! schema's version and the measuring's ([`from_text`]: one of another is
//! measured again, its file written over). A song without them has no
//! shine until they come.
//!
//! The voice is the vocals of the song's kept stems, at their rate; a
//! song without kept stems is separated for it (on the GPU, its voice at
//! 16 kHz), as its words were timed.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;

use jodel_lyrics::Lyrics;
use jodel_rhythm::waveform::Waveform;

use crate::browser::Notify;
use crate::cache::SHINE;
use crate::core::shine::reference::Original;
use crate::core::shine::Reference;
use crate::core::singing::Notes;

/// A reference as the file holds it, its header in front.
pub fn to_text(reference: &Reference) -> String {
    SHINE.wrap(&reference.to_text())
}

/// The reference in a file's `text`: none for another schema version, one
/// measured another way, or one not read.
pub fn from_text(text: &str) -> Option<Reference> {
    Reference::from_text(SHINE.body(text)?)
}

/// What names a song's file: its path, its sung lines' words as timed
/// (each word's text, times, letters, whether a backing one and, when it
/// is, that it is not to be sung) and its melody, hashed.
pub fn key(path: &Path, lyrics: &Lyrics, waveform: &Waveform) -> String {
    use std::fmt::Write as _;
    let mut text = format!("{} shine\n", path.display());
    for line in lyrics.lines.iter().filter(|l| !l.words.is_empty()) {
        for w in &line.words {
            let _ = write!(
                text,
                "{}|{:?}|{:?}|{}|{:?};",
                w.text,
                w.start.map(f64::to_bits),
                w.end.map(f64::to_bits),
                w.backing,
                w.letters.iter().map(|l| l.to_bits()).collect::<Vec<_>>()
            );
            if w.not_sung {
                text.push_str("not sung;");
            }
        }
        text.push('\n');
    }
    if let Some(m) = &waveform.melody {
        let cents = m
            .cents
            .iter()
            .fold(0u64, |h, &c| h.rotate_left(5) ^ u64::from(c));
        let sure = m
            .sure
            .iter()
            .fold(0u64, |h, &s| h.rotate_left(3) ^ u64::from(s));
        let _ = write!(text, "melody {} {cents:x} {sure:x}", m.cents.len());
    }
    crate::cache::name(&text)
}

/// The folder the files are kept in: `shine` beside the analyses' folder
/// `analyses`.
pub fn beside(analyses: &Path) -> PathBuf {
    analyses.parent().unwrap_or(analyses).join("shine")
}

/// The reference kept for `key` in `dir`, where there is one of this
/// version.
pub fn load(dir: &Path, key: &str) -> Option<Reference> {
    let file = crate::mover::locate(&dir.join(format!("{key}.txt")));
    from_text(&std::fs::read_to_string(file).ok()?)
}

/// `reference` kept for `key` in `dir`, durably.
pub fn store(dir: &Path, key: &str, reference: &Reference) -> std::io::Result<()> {
    let dir = crate::mover::current(dir);
    durable::write_atomic(
        &dir.join(format!("{key}.txt")),
        to_text(reference).as_bytes(),
    )
}

/// The voice of `path`, mono: the vocals of its stems kept in `stems` at
/// their rate (once its analysis has them in their file, if it is
/// separating them now: [`crate::stems_cache::open_waiting`]), else
/// separated now at 16 kHz; and its rate.
pub fn voice_of(stems: Option<&Path>, path: &Path) -> Result<(Vec<f32>, u32), String> {
    use jodel_playback::stems::{Stem, RATE};
    let kept = stems.and_then(|dir| crate::stems_cache::open_waiting(dir, path));
    let Some(mut file) = kept else {
        return Ok((crate::lyrics::timer::separate_16k(path, false)?.0, 16_000));
    };
    let vocals = Stem::ALL
        .iter()
        .position(|&s| s == Stem::Vocals)
        .unwrap_or(2);
    let mut voice = Vec::with_capacity(file.frames() as usize);
    let mut at = 0;
    while at < file.frames() {
        let stems = file
            .read(at, RATE as usize * 10)
            .map_err(|e| e.to_string())?;
        let [l, r] = &stems[vocals];
        if l.is_empty() {
            break;
        }
        at += l.len() as u64;
        voice.extend(l.iter().zip(r).map(|(a, b)| (a + b) / 2.0));
    }
    Ok((voice, RATE))
}

/// Where the worker reads and keeps: the analyses' folder (the references
/// kept beside it, the stems too); none: nothing kept, the stems not read.
#[derive(Clone, Debug, Default)]
pub struct Folders {
    /// The references.
    pub shine: Option<PathBuf>,
    /// The stems.
    pub stems: Option<PathBuf>,
}

impl Folders {
    /// Those beside the analyses' folder `analyses`.
    pub fn beside(analyses: &Path) -> Folders {
        Folders {
            shine: Some(beside(analyses)),
            stems: Some(crate::stems_cache::beside(analyses)),
        }
    }
}

/// The reference of `path`'s singing to `notes` (made from its `lyrics`
/// and `waveform`): kept in `folders` (none: not kept), else worked out
/// from its voice and kept.
pub fn reference_of(
    folders: &Folders,
    path: &Path,
    notes: &Notes,
    (lyrics, waveform): (&Lyrics, &Waveform),
) -> Result<Reference, String> {
    let key = key(path, lyrics, waveform);
    let dir = folders.shine.as_deref();
    if let Some(kept) = dir.and_then(|d| load(d, &key)) {
        if kept.phrases.len() == notes.lines.len() {
            return Ok(kept);
        }
    }
    let (voice, rate) = voice_of(folders.stems.as_deref(), path)?;
    let reference = Original::new(notes, &voice, rate).reference();
    if let Some(dir) = dir {
        if let Err(e) = store(dir, &key, &reference) {
            crate::trace::note(format!("shine: not kept for {}: {e}", path.display()));
        }
    }
    Ok(reference)
}

/// A song's notes, wanted with their reference: the song, the lyrics and
/// the waveform they are made from.
#[derive(Clone, Debug)]
pub struct Made {
    /// The song.
    pub path: PathBuf,
    /// Its lyrics.
    pub lyrics: Arc<Lyrics>,
    /// Its waveform (its melody).
    pub waveform: Arc<Waveform>,
}

impl Made {
    /// Whether it is of these (the same lyrics and waveform, not copies).
    fn is(&self, path: &Path, lyrics: &Arc<Lyrics>, waveform: &Arc<Waveform>) -> bool {
        self.path == path
            && Arc::ptr_eq(&self.lyrics, lyrics)
            && Arc::ptr_eq(&self.waveform, waveform)
    }
}

/// A reference worked out (or not: none), for the notes `made` of.
pub struct Found {
    /// What the notes were made of.
    pub made: Made,
    /// The reference.
    pub reference: Option<Arc<Reference>>,
}

/// The thread working out references, the last asked for at a time.
pub struct ShineWorker {
    tx: Sender<Made>,
    rx: Receiver<Found>,
    /// What was last asked for.
    asked: Option<Made>,
}

impl ShineWorker {
    /// A worker reading and keeping in `folders`, calling `notify` as
    /// each is done.
    pub fn new(folders: Folders, notify: Notify) -> ShineWorker {
        let (tx, wants) = mpsc::channel::<Made>();
        let (found, rx) = mpsc::channel();
        thread::spawn(move || {
            // Its GPU work (separating a song without kept stems) gives way
            // to the window's frames.
            jodel_gpu::pace::background();
            while let Ok(mut want) = wants.recv() {
                // Only the last asked for.
                while let Ok(later) = wants.try_recv() {
                    want = later;
                }
                let done = work(&folders, want);
                if found.send(done).is_err() {
                    return;
                }
                notify();
            }
        });
        ShineWorker {
            tx,
            rx,
            asked: None,
        }
    }

    /// The reference wanted for `path`'s notes made from `lyrics` and
    /// `waveform` (asked once).
    pub fn want(&mut self, path: &Path, lyrics: &Arc<Lyrics>, waveform: &Arc<Waveform>) {
        if self
            .asked
            .as_ref()
            .is_some_and(|a| a.is(path, lyrics, waveform))
        {
            return;
        }
        let made = Made {
            path: path.to_owned(),
            lyrics: Arc::clone(lyrics),
            waveform: Arc::clone(waveform),
        };
        self.asked = Some(made.clone());
        let _ = self.tx.send(made);
    }

    /// A reference worked out since the last call, if one was.
    pub fn take(&self) -> Option<Found> {
        self.rx.try_recv().ok()
    }
}

/// `made`'s reference, kept in `folders`.
fn work(folders: &Folders, made: Made) -> Found {
    let Some(melody) = &made.waveform.melody else {
        return Found {
            made,
            reference: None,
        };
    };
    let notes = Notes::new(&made.lyrics, melody);
    // A song without kept stems is separated on the background separator
    // (this thread does background work).
    let separator = jodel_gpu::GpuSeparator::background()
        .filter(|_| crate::trace::on())
        .map(|g| (g, g.tally()));
    let started = std::time::Instant::now();
    let reference = match reference_of(folders, &made.path, &notes, (&made.lyrics, &made.waveform))
    {
        Ok(r) => {
            crate::trace::note(format!(
                "shine: {} phrases of {} to shine on, in {:.1} s{}",
                r.shining(),
                made.path.display(),
                started.elapsed().as_secs_f64(),
                separator
                    .map(|(g, before)| g.tally().since(&before))
                    .filter(|t| t.submissions > 0)
                    .map_or_else(String::new, |t| format!(
                        " (the separator, other work on it meanwhile included: {t})"
                    ))
            ));
            Some(Arc::new(r))
        }
        Err(e) => {
            crate::trace::note(format!("shine: none for {}: {e}", made.path.display()));
            None
        }
    };
    Found { made, reference }
}
