//! Files written so that a crash or a power cut leaves the old version or
//! the new one whole, never a torn mix of both: every file this workspace
//! writes goes through here, so they all behave alike.
//!
//! - [`write_atomic`]: bytes at once. They go to a temporary file beside
//!   the file, are made durable (`fsync`), then renamed over it (atomic
//!   within a folder), and the folder is made durable too, so the rename
//!   itself survives.
//! - [`AtomicFile`]: a file written piece by piece (and seeked: a header
//!   filled in at the end), under a temporary name beside it, that takes its
//!   name only on [`AtomicFile::commit`], made durable the same way. Dropped
//!   without it, the temporary goes: nothing half written ever has the
//!   file's name.
//! - [`rename_durable`]: a rename, then both folders made durable.
//! - [`sync_dir`]: a folder's entries made durable (a directory handle's
//!   flush on Windows, `fsync` of the folder elsewhere).
//! - [`is_temporary`], [`sweep`]: the temporaries a crash left behind
//!   (never a file's name, so nothing reads them), removed once old.

use std::fs;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// The suffix of every temporary name.
pub const TEMPORARY: &str = ".tmp";

/// Makes `dir`'s entries (names made, renamed, removed) durable.
///
/// On Windows through a directory handle (`FILE_FLAG_BACKUP_SEMANTICS`);
/// where the system will not flush one, NTFS has journaled the change
/// already and nothing more can be done: no error then.
pub fn sync_dir(dir: &Path) -> io::Result<()> {
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
        if let Ok(handle) = fs::OpenOptions::new()
            .write(true)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
            .open(dir)
        {
            let _ = handle.sync_all();
        }
        Ok(())
    }
    #[cfg(not(windows))]
    {
        fs::File::open(dir)?.sync_all()
    }
}

/// A temporary name beside `path`, unique to this process and call:
/// `NAME.PID-N.tmp`.
pub fn temporary(path: &Path) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(
        ".{}-{}{TEMPORARY}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    path.with_file_name(name)
}

/// Whether `name` is a temporary of [`temporary`]'s (or of an older
/// writer's: `.partial`, `.moving`, `.recording`, `.tmpPID`).
pub fn is_temporary(name: &str) -> bool {
    name.ends_with(TEMPORARY)
        || name.ends_with(".partial")
        || name.ends_with(".moving")
        || name.ends_with(".recording")
        || name
            .rsplit_once(".tmp")
            .is_some_and(|(_, pid)| !pid.is_empty() && pid.bytes().all(|b| b.is_ascii_digit()))
}

/// Removes the temporaries in `dir` (not below it) last written more than
/// `older_than` ago: what a crash left (one younger may be another
/// player's, being written). How many went.
pub fn sweep(dir: &Path, older_than: Duration) -> usize {
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    let mut gone = 0;
    for entry in entries.flatten() {
        let name = entry.file_name();
        if !is_temporary(&name.to_string_lossy()) {
            continue;
        }
        let old = entry
            .metadata()
            .ok()
            .filter(|m| m.is_file())
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|age| age > older_than);
        if old && fs::remove_file(entry.path()).is_ok() {
            gone += 1;
        }
    }
    gone
}

fn parent(path: &Path) -> Option<&Path> {
    path.parent().filter(|d| !d.as_os_str().is_empty())
}

/// Writes `bytes` to `path` whole or not at all (see the crate's docs),
/// making its folder first if need be.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = AtomicFile::create(path)?;
    file.write_all(bytes)?;
    file.commit()
}

/// Renames `from` to `to` and makes both folders durable.
pub fn rename_durable(from: &Path, to: &Path) -> io::Result<()> {
    fs::rename(from, to)?;
    if let Some(dir) = parent(to) {
        sync_dir(dir)?;
    }
    if let (Some(a), Some(b)) = (parent(from), parent(to)) {
        if a != b {
            sync_dir(a)?;
        }
    }
    Ok(())
}

/// A file being written under a temporary name beside `path`, buffered;
/// it takes `path`'s name only on [`AtomicFile::commit`].
pub struct AtomicFile {
    file: Option<BufWriter<fs::File>>,
    part: PathBuf,
    path: PathBuf,
}

impl AtomicFile {
    /// Starts `path` (its folder made if need be).
    pub fn create(path: &Path) -> io::Result<AtomicFile> {
        if let Some(dir) = parent(path) {
            fs::create_dir_all(dir)?;
        }
        let part = temporary(path);
        let file = fs::File::create(&part)?;
        Ok(AtomicFile {
            file: Some(BufWriter::with_capacity(1 << 20, file)),
            part,
            path: path.to_path_buf(),
        })
    }

    /// Where it goes.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Where it is being written.
    pub fn temporary_path(&self) -> &Path {
        &self.part
    }

    fn file(&mut self) -> &mut BufWriter<fs::File> {
        self.file
            .as_mut()
            .expect("an AtomicFile is used after commit")
    }

    /// Done: its bytes made durable, then its name (over what was there),
    /// then the folder.
    pub fn commit(mut self) -> io::Result<()> {
        let file = self.file.take().expect("committed once");
        let file = file.into_inner().map_err(|e| e.into_error())?;
        file.sync_all()?;
        drop(file);
        rename_durable(&self.part, &self.path)
    }
}

impl Write for AtomicFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.file().write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file().flush()
    }
}

impl Seek for AtomicFile {
    fn seek(&mut self, to: SeekFrom) -> io::Result<u64> {
        self.file().seek(to)
    }
}

impl Drop for AtomicFile {
    fn drop(&mut self) {
        // Not committed: never takes the name.
        if self.file.take().is_some() {
            let _ = fs::remove_file(&self.part);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("durable-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn a_file_is_replaced_whole_and_nothing_is_left_beside_it() {
        let dir = temp("whole");
        let path = dir.join("deep/config.toml");
        write_atomic(&path, b"one").unwrap();
        write_atomic(&path, b"two, longer").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"two, longer");
        let left: Vec<_> = fs::read_dir(dir.join("deep")).unwrap().flatten().collect();
        assert_eq!(left.len(), 1, "no temporary left: {left:?}");
        rename_durable(&path, &dir.join("moved.toml")).unwrap();
        assert_eq!(fs::read(dir.join("moved.toml")).unwrap(), b"two, longer");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_file_written_piece_by_piece_has_its_name_only_once_committed() {
        let dir = temp("pieces");
        let path = dir.join("song.stems");
        let mut file = AtomicFile::create(&path).unwrap();
        file.write_all(&[0u8; 4]).unwrap();
        file.write_all(b"body").unwrap();
        // The header filled in at the end.
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(b"head").unwrap();
        assert!(!path.exists(), "no name before the commit");
        assert!(file.temporary_path().exists());
        file.commit().unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"headbody");
        // Dropped without a commit: nothing.
        let other = dir.join("other.stems");
        let mut file = AtomicFile::create(&other).unwrap();
        file.write_all(b"half").unwrap();
        let part = file.temporary_path().to_path_buf();
        drop(file);
        assert!(!other.exists() && !part.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn temporaries_are_told_apart_and_old_ones_swept() {
        for name in [
            "a.flac.1234-0.tmp",
            "x.partial",
            "y.stems.moving",
            "r.wav.recording",
            "k.tmp4321",
        ] {
            assert!(is_temporary(name), "{name}");
        }
        for name in ["song.stems", "lyrics.txt", "a.tmp.flac", "temp", "x.tmpdir"] {
            assert!(!is_temporary(name), "{name}");
        }
        let dir = temp("sweep");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("left.1-0.tmp"), b"x").unwrap();
        fs::write(dir.join("kept.txt"), b"x").unwrap();
        assert_eq!(sweep(&dir, Duration::from_secs(3600)), 0, "too young");
        assert_eq!(sweep(&dir, Duration::ZERO), 1);
        assert!(dir.join("kept.txt").exists());
        let _ = fs::remove_dir_all(&dir);
    }
}
