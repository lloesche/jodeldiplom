//! Whether a file is the one expected: at its length, then stamped as it
//! is (a line in [`STAMP`] in the store), else hashed and stamped where it
//! matches. Every path here is read through [`crate::store::mover::locate`]
//! (while the data folder moves, a file not yet moved is where it was):
//! the callers pass paths as setup puts them, and never locate before
//! calling these.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use super::status::{failure, Failure, Status};
use super::Sum;
use crate::work::download::Sha256;
use crate::work::Teller;

/// The stamp's file in the store: a line a file checked, `SHA-256 length
/// written-at path` (written-at in ns since 1970).
pub(super) const STAMP: &str = "checked.txt";

/// What to do when the disk let a step down.
const DISK: &str = "Check that the disk can be read, then try again.";

/// What to do when the store could not be written to.
pub(super) const STORE: &str = "Check that the data folder can be written to, then try again.";

/// Where `path` is now.
fn located(path: &Path) -> PathBuf {
    crate::store::mover::locate(path)
}

/// When `meta`'s file was last written, in ns since 1970.
fn written(meta: &fs::Metadata) -> u128 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos())
}

fn stamp_lines(store: &Path) -> Vec<String> {
    fs::read_to_string(store.join(STAMP))
        .map(|t| t.lines().map(str::to_owned).collect())
        .unwrap_or_default()
}

fn stamp_line(sha256: &str, path: &Path) -> Option<String> {
    let meta = fs::metadata(path).ok()?;
    Some(format!(
        "{sha256} {} {} {}",
        meta.len(),
        written(&meta),
        path.display()
    ))
}

/// Records the file at `path` as checked against `sha256`: its earlier
/// lines gone, and those of files no longer there.
pub(super) fn stamp(store: &Path, sha256: &str, path: &Path) -> Result<(), Failure> {
    let path = located(path);
    let Some(line) = stamp_line(sha256, &path) else {
        return Err(failure(format!("{}: gone", path.display()), DISK));
    };
    let tail = format!(" {}", path.display());
    let mut lines: Vec<String> = stamp_lines(store)
        .into_iter()
        .filter(|l| !l.ends_with(&tail))
        .filter(|l| {
            l.splitn(4, ' ')
                .nth(3)
                .is_some_and(|file| Path::new(file).is_file())
        })
        .collect();
    lines.push(line);
    write_stamps(store, &lines)
}

/// Keeps `lines` as the store's stamps (a stamp file made read-only made
/// writable again: it is the player's own).
fn write_stamps(store: &Path, lines: &[String]) -> Result<(), Failure> {
    let file = store.join(STAMP);
    durable::make_replaceable(&file);
    let text = lines.iter().map(|l| format!("{l}\n")).collect::<String>();
    durable::write_atomic(&file, text.as_bytes())
        .map_err(|e| failure(format!("{}: {e}", file.display()), STORE))
}

/// Forgets the stamps of the files at `paths`: each is hashed again the
/// next time it is looked at.
pub(super) fn forget(store: &Path, paths: &[PathBuf]) -> Result<(), Failure> {
    let tails: Vec<String> = (paths.iter())
        .map(|p| format!(" {}", located(p).display()))
        .collect();
    let lines = stamp_lines(store);
    let kept: Vec<String> = (lines.iter())
        .filter(|l| !tails.iter().any(|t| l.ends_with(t.as_str())))
        .cloned()
        .collect();
    if kept.len() == lines.len() {
        return Ok(());
    }
    write_stamps(store, &kept)
}

/// Whether a file is at `path` at `sum`'s length.
pub(super) fn present(path: &Path, sum: &Sum) -> bool {
    fs::metadata(located(path)).is_ok_and(|m| m.is_file() && m.len() == sum.bytes)
}

/// Whether the file at `path` is known to be `sum`'s without hashing it:
/// at its length, and stamped as it is.
pub(super) fn known(store: &Path, path: &Path, sum: &Sum) -> bool {
    present(path, sum)
        && stamp_line(&sum.sha256, &located(path))
            .is_some_and(|line| stamp_lines(store).contains(&line))
}

/// Whether the file at `path` is `sum`'s: [`known`], else hashed (its
/// step's bytes before it `before`) and stamped where it matches. A file
/// that cannot be read stops setup: the disk is at fault, and downloading
/// again would not mend it.
pub(super) fn verified(
    store: &Path,
    path: &Path,
    sum: &Sum,
    before: u64,
    teller: &mut Teller<'_, Status>,
) -> Result<bool, Failure> {
    if known(store, path, sum) {
        return Ok(true);
    }
    if !present(path, sum) {
        return Ok(false);
    }
    if hashes_to(path, &sum.sha256, before, teller)?.is_some() {
        return Ok(false);
    }
    stamp(store, &sum.sha256, path)?;
    Ok(true)
}

/// Hashes the file at `path` (its step's bytes before it `before`): `None`
/// where it is `sha256`, else its SHA-256.
pub(super) fn hashes_to(
    path: &Path,
    sha256: &str,
    before: u64,
    teller: &mut Teller<'_, Status>,
) -> Result<Option<String>, Failure> {
    let path = located(path);
    let digest = hash_file(&path, &mut |done| {
        teller.set(false, |s| s.done = before + done)
    })
    .map_err(|e| failure(e, DISK))?;
    Ok((digest != sha256).then_some(digest))
}

/// Hashes `path`, telling `progress` the bytes so far.
fn hash_file(path: &Path, progress: &mut dyn FnMut(u64)) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut done = 0u64;
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
        done += n as u64;
        progress(done);
    }
    Ok(hash.hex())
}
