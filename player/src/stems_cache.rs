//! The tracks' stems kept on disk ([`jodel_playback::stem_file`]), so a
//! track is separated once: when it is got ready (its analysis writes
//! them), not each time it plays on stems.
//!
//! They are kept in `stems` in the data folder, beside the analyses,
//! one file per track, named by a hash of its path, size and modification
//! time and the stems format. The folder is kept under [`BUDGET`]: after a
//! file is written, the least recently used go first (a file read for
//! playing counts as used).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use jodel_playback::stem_file::{self, StemFile, StemStore};

/// How much the stems may take on disk, at most (a four-minute song takes
/// about 60 to 120 MB).
pub const BUDGET: u64 = 30 * 1024 * 1024 * 1024;

/// The folder: `stems` beside the analyses.
pub fn dir() -> Option<PathBuf> {
    crate::rhythm::cache_dir().map(|d| beside(&d))
}

/// The stems folder beside the analyses' folder `analyses`.
pub fn beside(analyses: &Path) -> PathBuf {
    analyses.parent().unwrap_or(analyses).join("stems")
}

/// The stems file of `path` in `dir` (whether or not it is there); `None`
/// when the file cannot be read.
pub fn file_for(dir: &Path, path: &Path) -> Option<PathBuf> {
    let meta = fs::metadata(path).ok()?;
    let modified = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    let key = format!(
        "{} {} {modified} stems v{}",
        path.display(),
        meta.len(),
        stem_file::FORMAT
    );
    Some(dir.join(format!("{:016x}.stems", crate::rhythm::hash(&key))))
}

/// Whether the stems of `path` are kept in `dir` (whole).
pub fn is_kept(dir: &Path, path: &Path) -> bool {
    file_for(dir, path).is_some_and(|f| crate::mover::locate(&f).is_file())
}

/// The tracks whose stems are being written into their files now
/// ([`Keeping`]).
static SEPARATING: (Mutex<Vec<PathBuf>>, Condvar) = (Mutex::new(Vec::new()), Condvar::new());

/// The stems of a track written into their file as they are separated
/// (by its analysis). While it lives, what wants them waits for the file
/// ([`open_waiting`]) rather than separate them again beside it, which
/// would halve the speed of both on the one background separator; a write
/// that fails ends it (dropped), so no one waits for a file that will not
/// come.
pub struct Keeping {
    writer: jodel_playback::StemWriter,
    _separating: Separating,
}

impl Keeping {
    /// The stems of `path` (at `rate`, separated by `separator`) to be
    /// written into the file `to`; `None` when it cannot be made (they are
    /// separated live then).
    pub fn create(to: &Path, path: &Path, rate: u32, separator: &str) -> Option<Keeping> {
        let writer = jodel_playback::StemWriter::create(to, rate, separator).ok()?;
        Some(Keeping {
            writer,
            _separating: Separating::new(path),
        })
    }

    /// The next frames ([`jodel_playback::StemWriter::push`]).
    pub fn push(
        &mut self,
        stems: [[&[f32]; 2]; jodel_playback::stems::STEMS],
    ) -> std::io::Result<()> {
        self.writer.push(stems)
    }

    /// The file made whole: kept, and those waiting for it told.
    pub fn finish(self) -> std::io::Result<()> {
        self.writer.finish()
    }
}

/// A track marked as having its stems written now ([`Keeping`]).
struct Separating(PathBuf);

impl Separating {
    fn new(path: &Path) -> Separating {
        let (list, _) = &SEPARATING;
        list.lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(path.to_owned());
        Separating(path.to_owned())
    }
}

impl Drop for Separating {
    fn drop(&mut self) {
        let (list, changed) = &SEPARATING;
        let mut list = list.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(at) = list.iter().position(|p| *p == self.0) {
            list.swap_remove(at);
        }
        changed.notify_all();
    }
}

/// Waits while the stems of `path` are being written into their file
/// ([`Keeping`]).
fn wait_kept(path: &Path) {
    let (list, changed) = &SEPARATING;
    let list = list.lock().unwrap_or_else(|e| e.into_inner());
    let _done = changed
        .wait_while(list, |list| list.iter().any(|p| p == path))
        .unwrap_or_else(|e| e.into_inner());
}

/// The stems of `path` kept in `dir`, opened ([`open`]); when none are,
/// but they are being written into their file now ([`Keeping`]), once
/// they are.
pub fn open_waiting(dir: &Path, path: &Path) -> Option<StemFile> {
    open(dir, path).or_else(|| {
        wait_kept(path);
        open(dir, path)
    })
}

/// The stems of `path` kept in `dir`, opened (and marked as used).
#[expect(
    clippy::disallowed_methods,
    reason = "marks the kept stems as used: their modified time only, no bytes written"
)]
pub fn open(dir: &Path, path: &Path) -> Option<StemFile> {
    // Where it is now: while the data folder moves, not yet moved is where
    // it was.
    let file = crate::mover::locate(&file_for(dir, path)?);
    let opened = StemFile::open(&file).ok()?;
    if let Ok(f) = fs::OpenOptions::new().write(true).open(&file) {
        let _ = f.set_modified(SystemTime::now());
    }
    Some(opened)
}

/// Deletes the least recently used stems files in `dir` until they take
/// `budget` bytes at most.
pub fn prune(dir: &Path, budget: u64) {
    let dir = &crate::mover::current(dir);
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut files: Vec<(SystemTime, u64, PathBuf)> = entries
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "stems"))
        .filter_map(|e| {
            let meta = e.metadata().ok()?;
            Some((meta.modified().ok()?, meta.len(), e.path()))
        })
        .collect();
    let mut total: u64 = files.iter().map(|f| f.1).sum();
    files.sort_by_key(|f| f.0);
    for (_, len, path) in files {
        if total <= budget {
            break;
        }
        if fs::remove_file(&path).is_ok() {
            total -= len;
        }
    }
}

/// The engine's way to the kept stems.
pub struct Store {
    dir: PathBuf,
}

impl Store {
    /// The store over `dir`, for the engine.
    pub fn shared(dir: PathBuf) -> Arc<dyn StemStore> {
        Arc::new(Store { dir })
    }
}

impl StemStore for Store {
    fn open(&self, path: &Path) -> Option<StemFile> {
        open(&self.dir, path)
    }
}

#[cfg(test)]
#[expect(
    clippy::disallowed_methods,
    reason = "tests: stem files and tracks in a temporary folder, their times set"
)]
mod tests {
    use super::*;

    #[test]
    fn the_least_recently_used_go_first() {
        let dir = std::env::temp_dir().join(format!("jodel-stems-prune-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let now = SystemTime::now();
        for (i, name) in ["old", "middle", "new"].iter().enumerate() {
            let path = dir.join(format!("{name}.stems"));
            fs::write(&path, vec![0u8; 1000]).unwrap();
            let f = fs::OpenOptions::new().write(true).open(&path).unwrap();
            f.set_modified(now - std::time::Duration::from_secs(100 * (3 - i as u64)))
                .unwrap();
        }
        fs::write(dir.join("other.txt"), vec![0u8; 5000]).unwrap();
        prune(&dir, 2000);
        let mut left: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        left.sort();
        assert_eq!(left, ["middle.stems", "new.stems", "other.txt"]);
        // A track's file changes name with the track.
        let track = dir.join("song.flac");
        fs::write(&track, b"one").unwrap();
        let a = file_for(&dir, &track).unwrap();
        fs::write(&track, b"longer").unwrap();
        assert_ne!(file_for(&dir, &track).unwrap(), a);
        let _ = fs::remove_dir_all(&dir);
    }

    /// What wants a track's stems waits while its analysis separates them
    /// into their file, and for nothing else.
    #[test]
    fn stems_being_separated_are_waited_for() {
        let (song, other) = (Path::new("waited.flac"), Path::new("other.flac"));
        let separating = Separating::new(song);
        let started = std::time::Instant::now();
        wait_kept(other);
        assert!(started.elapsed() < std::time::Duration::from_millis(50));
        let done = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(80));
            drop(separating);
        });
        wait_kept(song);
        assert!(started.elapsed() >= std::time::Duration::from_millis(60));
        done.join().unwrap();
    }
}
