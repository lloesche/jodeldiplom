//! A file filled over more than one run ([`Resumable`]).

use std::fs;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use super::{create_dir_all, make_replaceable, parent, rename_durable};

/// The suffix of a [`Resumable`]'s name.
pub const RESUMABLE: &str = ".partial";

/// A file filled over more than one run (a download that may be cut off)
/// under a fixed name beside `path`, `NAME.partial`: opened again, by this
/// run or a later one, it holds what was kept, to be gone on from. It is
/// kept on purpose (dropped, it stays; [`super::sweep`] leaves it: its
/// owner removes one that nothing goes on from any more), and
/// takes `path`'s name only on [`Resumable::commit`]. What it holds is the
/// caller's to check before that: bytes kept may be torn at their end or
/// from another copy of the file (a download checks the whole against its
/// length and SHA-256, which is what makes going on from them safe).
pub struct Resumable {
    file: fs::File,
    part: PathBuf,
    path: PathBuf,
}

impl Resumable {
    /// The resumable file of `path` (its folder made if need be), as an
    /// earlier run left it, or empty: read from its start, written at its
    /// end.
    pub fn open(path: &Path) -> io::Result<Resumable> {
        if let Some(dir) = parent(path) {
            create_dir_all(dir)?;
        }
        let part = Resumable::part_of(path);
        let mut file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&part)?;
        file.seek(SeekFrom::Start(0))?;
        Ok(Resumable {
            file,
            part,
            path: path.to_path_buf(),
        })
    }

    /// Where `path`'s resumable file is: `NAME.partial` beside it.
    pub fn part_of(path: &Path) -> PathBuf {
        let mut name = path.file_name().unwrap_or_default().to_os_string();
        name.push(RESUMABLE);
        path.with_file_name(name)
    }

    /// The bytes it holds.
    pub fn kept(&self) -> io::Result<u64> {
        Ok(self.file.metadata()?.len())
    }

    /// Only its first `n` bytes kept, written on from there.
    pub fn keep_only(&mut self, n: u64) -> io::Result<()> {
        self.file.set_len(n)?;
        self.file.seek(SeekFrom::Start(n))?;
        Ok(())
    }

    /// What it holds made durable (to be gone on from, the run cut off).
    pub fn sync(&self) -> io::Result<()> {
        self.file.sync_all()
    }

    /// Done: its bytes made durable, then its name (over a file there,
    /// made replaceable first), then the folder.
    pub fn commit(self) -> io::Result<()> {
        self.file.sync_all()?;
        drop(self.file);
        make_replaceable(&self.path);
        rename_durable(&self.part, &self.path)
    }

    /// Given up: removed (what it held was not the file).
    pub fn discard(self) -> io::Result<()> {
        drop(self.file);
        fs::remove_file(&self.part)
    }
}

impl Read for Resumable {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.file.read(buf)
    }
}

impl Write for Resumable {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.file.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Opened again, it holds what was kept; cut back, it goes on from
    /// there; committed, it takes the name over a read-only file; given
    /// up, it goes.
    #[test]
    fn a_resumable_file_is_gone_on_from_then_takes_its_name() {
        let dir = jodel_testkit::TempDir::new("durable-resumable");
        let path = dir.join("deep").join("model.bin");
        let mut part = Resumable::open(&path).unwrap();
        part.write_all(b"the first half, ").unwrap();
        part.sync().unwrap();
        drop(part);
        assert!(!path.exists());
        let mut part = Resumable::open(&path).unwrap();
        assert_eq!(part.kept().unwrap(), 16);
        let mut kept = String::new();
        part.read_to_string(&mut kept).unwrap();
        assert_eq!(kept, "the first half, ");
        part.keep_only(10).unwrap();
        part.write_all(b"half, then the rest").unwrap();
        fs::write(&path, b"old").unwrap();
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_readonly(true);
        fs::set_permissions(&path, permissions).unwrap();
        part.commit().unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"the first half, then the rest");
        assert!(!Resumable::part_of(&path).exists());
        let mut part = Resumable::open(&path).unwrap();
        part.write_all(b"not it").unwrap();
        part.discard().unwrap();
        assert!(!Resumable::part_of(&path).exists());
    }
}
