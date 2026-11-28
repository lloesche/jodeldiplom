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
//! - [`AtomicDir`]: a folder of files filled under a temporary name beside
//!   it, that takes its name whole only on [`AtomicDir::commit`]; dropped
//!   without it, the temporary goes with what is in it.
//! - [`rename_durable`]: a rename, then both folders made durable.
//! - [`create_dir_all`]: the folders missing made, each one's entry made
//!   durable in its parent (so a file's first folder survives with it).
//! - [`sync_dir`]: a folder's entries made durable (a directory handle's
//!   flush on Windows, `fsync` of the folder elsewhere).
//! - [`Resumable`]: a file filled over more than one run (a download that
//!   may be cut off) under a fixed name beside it, `NAME.partial`, kept on
//!   purpose so that the next run goes on from it, and never swept; it
//!   takes the file's name only on [`Resumable::commit`].
//! - [`make_replaceable`]: a file's read-only mark taken off, so that a
//!   rename over it or its removal is not refused (Windows).
//! - [`is_temporary`], [`sweep`]: the temporaries a crash left behind
//!   (never a file's name, so nothing reads them), removed once old
//!   ([`STALE`]).

// The one place the writing calls clippy refuses elsewhere (clippy.toml)
// are made.
#![allow(clippy::disallowed_methods)]

use std::fs;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime};

mod resumable;

pub use resumable::{Resumable, RESUMABLE};

/// Takes the read-only mark off the file at `path`, where there is one, so
/// that it can be renamed over ([`write_atomic`], [`Resumable::commit`]) or
/// removed: Windows refuses either for a read-only file, the other systems
/// ask only the folder (nothing is done there). For a file the caller owns
/// and is about to replace; no other permission is widened.
pub fn make_replaceable(path: &Path) {
    #[cfg(windows)]
    if let Ok(meta) = fs::metadata(path) {
        let mut permissions = meta.permissions();
        if permissions.readonly() {
            #[expect(
                clippy::permissions_set_readonly_false,
                reason = "Windows: clears the read-only attribute, nothing more"
            )]
            permissions.set_readonly(false);
            let _ = fs::set_permissions(path, permissions);
        }
    }
    #[cfg(not(windows))]
    let _ = path;
}

/// The suffix of every temporary name.
pub const TEMPORARY: &str = ".tmp";

/// How old a temporary is before a sweep takes it for one a crash left
/// ([`sweep`]): a day. One younger may be another player's, being written.
pub const STALE: Duration = Duration::from_secs(24 * 3600);

/// Makes `dir`'s entries (names made, renamed, removed) durable.
///
/// On Windows through a directory handle (`FILE_FLAG_BACKUP_SEMANTICS`);
/// where the system will not flush one, NTFS has journaled the change
/// already and nothing more can be done: no error then. Elsewhere a file
/// system that cannot `fsync` a folder (`EINVAL`, `ENOTSUP`: some network
/// and FUSE ones) has nothing more to do either.
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
        match fs::File::open(dir)?.sync_all() {
            Err(e) if cannot_flush_folders(&e) => Ok(()),
            done => done,
        }
    }
}

/// Whether a folder's `fsync` failed because the file system cannot flush
/// one: `EINVAL`, or `ENOTSUP`/`EOPNOTSUPP` (one number on Linux, two on
/// Apple's systems).
#[cfg(not(windows))]
fn cannot_flush_folders(e: &io::Error) -> bool {
    const EINVAL: i32 = 22;
    #[cfg(target_vendor = "apple")]
    const NOT_SUPPORTED: [i32; 2] = [45, 102];
    #[cfg(not(target_vendor = "apple"))]
    const NOT_SUPPORTED: [i32; 1] = [95];
    e.raw_os_error()
        .is_some_and(|code| code == EINVAL || NOT_SUPPORTED.contains(&code))
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

/// Whether `name` is a temporary of [`temporary`]'s, or one of the
/// player's copies under way that do not go through here yet: a moved
/// file's `.moving`. A [`Resumable`]'s `.partial` is not one: it is kept on
/// purpose, to be gone on from.
pub fn is_temporary(name: &str) -> bool {
    name.ends_with(TEMPORARY) || name.ends_with(".moving")
}

/// Removes the temporaries in `dir` (not below it) last written more than
/// `older_than` ago ([`STALE`] at the players' sweeps): what a crash left
/// (one younger may be another player's, being written). A temporary
/// folder ([`AtomicDir`]'s) goes with what is in it; it was last written
/// when it or anything directly in it was (a folder filled deeper is not
/// looked into). How many went.
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
        // Not followed through a link: a link is removed, not what it is to.
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        let old = last_written(&entry.path(), &meta)
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|age| age > older_than);
        if !old {
            continue;
        }
        let removed = if meta.is_dir() {
            fs::remove_dir_all(entry.path())
        } else {
            fs::remove_file(entry.path())
        };
        if removed.is_ok() {
            gone += 1;
        }
    }
    gone
}

/// When the file at `path` was last written; for a folder, when it or
/// anything directly in it was.
fn last_written(path: &Path, meta: &fs::Metadata) -> Option<SystemTime> {
    let own = meta.modified().ok();
    if !meta.is_dir() {
        return own;
    }
    fs::read_dir(path)
        .ok()?
        .flatten()
        .filter_map(|e| e.metadata().ok()?.modified().ok())
        .chain(own)
        .max()
}

fn parent(path: &Path) -> Option<&Path> {
    path.parent().filter(|d| !d.as_os_str().is_empty())
}

/// Makes `dir` and the folders above it that are missing, each new one's
/// entry made durable in its parent: a folder only made (as
/// [`fs::create_dir_all`] does) can vanish after a power cut, and every
/// file in it with it, however durable the file.
pub fn create_dir_all(dir: &Path) -> io::Result<()> {
    let mut missing = Vec::new();
    let mut at = dir;
    while !at.as_os_str().is_empty() && !at.is_dir() {
        missing.push(at);
        at = at.parent().unwrap_or(Path::new(""));
    }
    for level in missing.into_iter().rev() {
        match fs::create_dir(level) {
            Ok(()) => {}
            // Made meanwhile, by another writer.
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists && level.is_dir() => {}
            Err(e) => return Err(e),
        }
        sync_dir(parent(level).unwrap_or(Path::new(".")))?;
    }
    Ok(())
}

/// Writes `bytes` to `path` whole or not at all (see the crate's docs),
/// making its folder first if need be. Unbuffered: the bytes go in one
/// write.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = AtomicFile::start(path, 0)?;
    file.write_all(bytes)?;
    file.commit()
}

/// Renames `from` to `to` and makes both folders durable.
pub fn rename_durable(from: &Path, to: &Path) -> io::Result<()> {
    fs::rename(from, to)?;
    sync_both(from, to)
}

/// How long a rename refused for a while is tried again: on Windows a
/// scanner or an indexer holding a file inside a folder refuses its
/// rename until it lets go.
const RENAME_PATIENCE: Duration = Duration::from_secs(2);

/// [`fs::rename`], tried again with growing pauses while it is refused
/// ([`io::ErrorKind::PermissionDenied`], or a sharing violation on
/// Windows), for [`RENAME_PATIENCE`]; then the last refusal.
fn rename_patiently(from: &Path, to: &Path) -> io::Result<()> {
    const SHARING_VIOLATION: i32 = 32;
    let started = Instant::now();
    let mut pause = Duration::from_millis(5);
    loop {
        match fs::rename(from, to) {
            Err(e)
                if (e.kind() == io::ErrorKind::PermissionDenied
                    || (cfg!(windows) && e.raw_os_error() == Some(SHARING_VIOLATION)))
                    && started.elapsed() < RENAME_PATIENCE =>
            {
                std::thread::sleep(pause);
                pause = (pause * 2).min(Duration::from_millis(250));
            }
            done => return done,
        }
    }
}

/// After a rename from `from` to `to`: both folders made durable.
fn sync_both(from: &Path, to: &Path) -> io::Result<()> {
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
    /// The temporary took `path`'s name: there is none to remove.
    committed: bool,
}

/// The buffer of a file written piece by piece.
const BUFFER: usize = 1 << 20;

impl AtomicFile {
    /// Starts `path` (its folder made if need be, [`create_dir_all`]).
    pub fn create(path: &Path) -> io::Result<AtomicFile> {
        AtomicFile::start(path, BUFFER)
    }

    /// Starts `path` with a buffer of `buffer` bytes (0: every write goes
    /// straight to the file, and nothing is allocated for it).
    fn start(path: &Path, buffer: usize) -> io::Result<AtomicFile> {
        if let Some(dir) = parent(path) {
            create_dir_all(dir)?;
        }
        let part = temporary(path);
        let file = fs::File::create(&part)?;
        Ok(AtomicFile {
            file: Some(BufWriter::with_capacity(buffer, file)),
            part,
            path: path.to_path_buf(),
            committed: false,
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
    /// then the folder. Failed before the name, the temporary goes.
    pub fn commit(mut self) -> io::Result<()> {
        let file = self.file.take().expect("committed once");
        let file = file.into_inner().map_err(|e| e.into_error())?;
        file.sync_all()?;
        drop(file);
        rename_durable(&self.part, &self.path)?;
        self.committed = true;
        Ok(())
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
        // Not committed (or the commit failed): never takes the name. The
        // buffer is let go of unwritten, the file closed, then removed.
        if !self.committed {
            drop(self.file.take().map(BufWriter::into_parts));
            let _ = fs::remove_file(&self.part);
        }
    }
}

/// A folder being filled under a temporary name beside `path` (which must
/// not exist yet); it takes `path`'s name, whole, only on
/// [`AtomicDir::commit`]. Its files are written into
/// [`AtomicDir::temporary_path`] the durable way ([`write_atomic`],
/// [`AtomicFile`]).
pub struct AtomicDir {
    part: PathBuf,
    path: PathBuf,
    /// The temporary took `path`'s name: there is none to remove.
    committed: bool,
}

impl AtomicDir {
    /// Starts `path` (the folders above it made if need be,
    /// [`create_dir_all`]).
    pub fn create(path: &Path) -> io::Result<AtomicDir> {
        let part = temporary(path);
        create_dir_all(&part)?;
        Ok(AtomicDir {
            part,
            path: path.to_path_buf(),
            committed: false,
        })
    }

    /// Where it goes.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Where it is being filled.
    pub fn temporary_path(&self) -> &Path {
        &self.part
    }

    /// Done: its entries made durable, then its name, then the folder it
    /// is in. A rename refused for a while (a scanner holding a file in
    /// it) is tried again for a couple of seconds before it fails; failed,
    /// the temporary goes.
    pub fn commit(mut self) -> io::Result<()> {
        sync_dir(&self.part)?;
        rename_patiently(&self.part, &self.path)?;
        self.committed = true;
        sync_both(&self.part, &self.path)
    }
}

impl Drop for AtomicDir {
    fn drop(&mut self) {
        // Not committed (or the commit failed), or a panic unwinds: never
        // takes the name.
        if !self.committed {
            let _ = fs::remove_dir_all(&self.part);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> jodel_testkit::TempDir {
        jodel_testkit::TempDir::new(&format!("durable-{name}"))
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
    }

    #[test]
    fn a_failed_commit_leaves_no_temporary() {
        let dir = temp("failed");
        let path = dir.join("taken");
        let mut file = AtomicFile::create(&path).unwrap();
        file.write_all(b"never named").unwrap();
        let part = file.temporary_path().to_path_buf();
        // A folder with something in it has the name: the rename fails.
        create_dir_all(&path.join("inside")).unwrap();
        assert!(file.commit().is_err());
        assert!(!part.exists(), "the temporary is gone");
        let left: Vec<_> = fs::read_dir(&dir).unwrap().flatten().collect();
        assert_eq!(left.len(), 1, "only the folder: {left:?}");
    }

    #[test]
    fn a_folder_filled_has_its_name_only_once_committed() {
        let dir = temp("folder");
        let path = dir.join("take");
        let take = AtomicDir::create(&path).unwrap();
        write_atomic(&take.temporary_path().join("voice.wav"), b"voice").unwrap();
        write_atomic(&take.temporary_path().join("about.txt"), b"about").unwrap();
        assert!(!path.exists(), "no name before the commit");
        take.commit().unwrap();
        assert_eq!(fs::read(path.join("voice.wav")).unwrap(), b"voice");
        assert_eq!(fs::read(path.join("about.txt")).unwrap(), b"about");
        let left: Vec<_> = fs::read_dir(&dir).unwrap().flatten().collect();
        assert_eq!(left.len(), 1, "no temporary left: {left:?}");
        // Dropped without a commit (cut short, a panic): nothing.
        let other = AtomicDir::create(&dir.join("other")).unwrap();
        write_atomic(&other.temporary_path().join("voice.wav"), b"half").unwrap();
        let part = other.temporary_path().to_path_buf();
        drop(other);
        assert!(!part.exists() && !dir.join("other").exists());
        // Its name taken already: the commit fails, the temporary goes.
        let again = AtomicDir::create(&path).unwrap();
        write_atomic(&again.temporary_path().join("voice.wav"), b"two").unwrap();
        let part = again.temporary_path().to_path_buf();
        assert!(again.commit().is_err());
        assert!(!part.exists());
        assert_eq!(fs::read(path.join("voice.wav")).unwrap(), b"voice");
    }

    /// A file held open inside a folder refuses the folder's rename on
    /// Windows (as a scanner's does): the commit waits it out.
    #[cfg(windows)]
    #[test]
    fn a_folder_rename_refused_for_a_while_is_tried_again() {
        let dir = temp("refused");
        let path = dir.join("take");
        let take = AtomicDir::create(&path).unwrap();
        let held = take.temporary_path().join("voice.wav");
        write_atomic(&held, b"voice").unwrap();
        let handle = fs::File::open(&held).unwrap();
        let refused = fs::rename(take.temporary_path(), &path).unwrap_err();
        assert_eq!(refused.kind(), io::ErrorKind::PermissionDenied, "{refused}");
        let letting_go = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            drop(handle);
        });
        take.commit().unwrap();
        letting_go.join().unwrap();
        assert_eq!(fs::read(path.join("voice.wav")).unwrap(), b"voice");
    }

    #[test]
    fn folders_are_made_level_by_level() {
        let dir = temp("folders");
        let deep = dir.join("a/b/c");
        create_dir_all(&deep).unwrap();
        assert!(deep.is_dir());
        // There already: nothing to do.
        create_dir_all(&deep).unwrap();
        create_dir_all(&dir).unwrap();
        // A file in the way.
        write_atomic(&dir.join("a/file"), b"x").unwrap();
        assert!(create_dir_all(&dir.join("a/file/d")).is_err());
    }

    #[test]
    fn temporaries_are_told_apart_and_old_ones_swept() {
        for name in ["a.flac.1234-0.tmp", "y.stems.moving"] {
            assert!(is_temporary(name), "{name}");
        }
        // A resumable download's is kept on purpose.
        for name in [
            "song.stems",
            "lyrics.txt",
            "a.tmp.flac",
            "temp",
            "x.tmpdir",
            "x.partial",
        ] {
            assert!(!is_temporary(name), "{name}");
        }
        let dir = temp("sweep");
        write_atomic(&dir.join("left.1-0.tmp"), b"x").unwrap();
        write_atomic(&dir.join("kept.txt"), b"x").unwrap();
        // A folder filled under a temporary name, never renamed into place.
        write_atomic(&dir.join("take.1-1.tmp/voice.wav"), b"x").unwrap();
        write_atomic(&dir.join("take/voice.wav"), b"x").unwrap();
        assert_eq!(sweep(&dir, Duration::from_secs(3600)), 0, "too young");
        assert_eq!(sweep(&dir, Duration::ZERO), 2);
        assert!(dir.join("kept.txt").exists() && dir.join("take/voice.wav").exists());
        let left: Vec<_> = fs::read_dir(&dir).unwrap().flatten().collect();
        assert_eq!(left.len(), 2, "{left:?}");
        // A folder is as young as the youngest file directly in it.
        let file = dir.join("take/voice.wav");
        let later = SystemTime::now() + Duration::from_secs(3600);
        fs::File::options()
            .write(true)
            .open(&file)
            .unwrap()
            .set_modified(later)
            .unwrap();
        let folder = dir.join("take");
        let meta = fs::metadata(&folder).unwrap();
        let written = last_written(&folder, &meta).unwrap();
        assert!(written + Duration::from_secs(1) > later, "{written:?}");
    }
}
