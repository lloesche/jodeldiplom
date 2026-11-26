//! Moving folders without losing anything to an interruption: the data
//! folder when the settings choose another.
//!
//! A move is written down before it begins: a journal in the settings
//! folder ([`JOURNAL`]), a line `from<TAB>to` for each folder to move. Then
//! each file goes over on its own:
//!
//! 1. A file already where it goes is the newer (the player writes only
//!    there once a move has begun): the old copy is dropped.
//! 2. Else it is renamed, where both places are on one disk (at once).
//! 3. Else it is copied to a `.moving` file beside where it goes, made
//!    durable, renamed to its name, and only then removed where it was.
//!
//! Interrupted at any point (the player quit, the system rebooted), the
//! next start finds the journal and goes on: nothing is lost (a file is
//! removed only once its copy is whole) and nothing is done twice. A full
//! disk stops the move, said with what is left, until [`Mover::retry`]. A
//! file in use (Windows will not remove a file a reader holds open) is left
//! for a later pass. When a folder is empty it goes, and its line is struck
//! from the journal.
//!
//! While a move is under way the player writes where things go
//! ([`current`]) and reads from there, or from where a file still is if it
//! has not come over yet ([`locate`]): nothing is made again. That a file
//! written where it goes is the newer (step 1) holds because of two rules:
//!
//! - A file kept for something (an analysis, a song's lyrics, a cover) is
//!   written from what made it, never from the old copy: the new one is
//!   the old one made again, or better.
//! - A file the player adds to (the best scores, the record of the last
//!   analyses) is written only from what was read through the mover
//!   ([`read`], [`locate`]): the new one holds everything the old one did.
//!
//! A file removed goes from both places ([`remove`]): left where it was, it
//! would come over again.

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use crate::core::DataMove;
use crate::outbox::Notify;
use crate::work::Job;

/// The journal's file, in the settings folder.
pub const JOURNAL: &str = "moving.txt";

/// The moves known this run (under way, or done): what [`current`] and
/// [`locate`] go by.
static MOVES: RwLock<Vec<(PathBuf, PathBuf)>> = RwLock::new(Vec::new());

fn moves() -> Vec<(PathBuf, PathBuf)> {
    MOVES.read().unwrap_or_else(|e| e.into_inner()).clone()
}

fn register(from: &Path, to: &Path) {
    let mut moves = MOVES.write().unwrap_or_else(|e| e.into_inner());
    if !moves.iter().any(|(f, t)| f == from && t == to) {
        moves.push((from.to_path_buf(), to.to_path_buf()));
    }
}

/// Where `path` is to be written: under the folder it was moved to, if it
/// is under one being (or that was) moved.
pub fn current(path: &Path) -> PathBuf {
    for (from, to) in moves() {
        if let Ok(rest) = path.strip_prefix(&from) {
            return to.join(rest);
        }
    }
    path.to_path_buf()
}

/// Where `path` is to be read: where it goes ([`current`]) when it is
/// there, else where it was, if it has not come over yet.
pub fn locate(path: &Path) -> PathBuf {
    let now = current(path);
    if now.exists() {
        return now;
    }
    old_places(path)
        .into_iter()
        .find(|old| old.exists())
        .unwrap_or(now)
}

/// Where `path` was before the moves: each place a copy not yet moved may
/// still be.
fn old_places(path: &Path) -> Vec<PathBuf> {
    let now = current(path);
    moves()
        .into_iter()
        .filter_map(|(from, to)| Some(from.join(now.strip_prefix(&to).ok()?)))
        .collect()
}

/// The bytes of the file `path` names, wherever it is now: where it goes,
/// else where it was. One moved between the two looks is found where it
/// went (a file is in one place or the other at every moment: the old copy
/// goes only once the new one is whole), so a file there is never taken for
/// none, as [`locate`] then a read could.
pub fn read(path: &Path) -> io::Result<Vec<u8>> {
    let now = current(path);
    let found =
        |r: &io::Result<Vec<u8>>| !matches!(r, Err(e) if e.kind() == io::ErrorKind::NotFound);
    let first = fs::read(&now);
    if found(&first) {
        return first;
    }
    for old in old_places(path) {
        let there = fs::read(old);
        if found(&there) {
            return there;
        }
    }
    fs::read(&now)
}

/// Removes the file `path` names where it goes and where it was: a copy
/// left in the old place would come over later as if it were kept. Not
/// there at all is no error.
pub fn remove(path: &Path) -> io::Result<()> {
    let mut result = Ok(());
    for file in std::iter::once(current(path)).chain(old_places(path)) {
        match fs::remove_file(&file) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => result = Err(e),
            _ => {}
        }
    }
    result
}

/// The moves the journal in `settings` holds.
pub fn journal(settings: &Path) -> Vec<(PathBuf, PathBuf)> {
    fs::read_to_string(settings.join(JOURNAL))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| {
            let (from, to) = l.split_once('\t')?;
            Some((PathBuf::from(from), PathBuf::from(to)))
        })
        .collect()
}

fn write_journal(settings: &Path, moves: &[(PathBuf, PathBuf)]) -> io::Result<()> {
    let file = settings.join(JOURNAL);
    if moves.is_empty() {
        return match fs::remove_file(&file) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
            _ => Ok(()),
        };
    }
    let text: String = moves
        .iter()
        .map(|(f, t)| format!("{}\t{}\n", f.display(), t.display()))
        .collect();
    // Whole or not at all, and on the disk before anything moves: a reboot
    // mid-write leaves the journal as it was before.
    durable::write_atomic(&file, text.as_bytes())
}

/// Writes `moves` into the journal (those whose folder is there, not
/// already in it), and registers them for [`current`] and [`locate`]:
/// from now on, what is written goes where it goes.
pub fn begin(settings: &Path, moves: &[(PathBuf, PathBuf)]) -> io::Result<()> {
    let mut all = journal(settings);
    for (from, to) in moves {
        if from != to && !gone(from) && !all.iter().any(|(f, _)| f == from) {
            all.push((from.clone(), to.clone()));
        }
    }
    write_journal(settings, &all)?;
    // Every move redirects what is written from now on, a folder not there
    // yet (nothing kept in it so far) too.
    for (from, to) in all.iter().chain(moves) {
        if from != to {
            register(from, to);
        }
    }
    Ok(())
}

/// Registers the moves the journal in `settings` holds (unfinished, from
/// an earlier run) for [`current`] and [`locate`], leaving the journal as
/// it is.
pub fn resume(settings: &Path) {
    for (from, to) in journal(settings) {
        if from != to {
            register(&from, &to);
        }
    }
}

/// Where a move stands, for the settings page.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Status {
    /// Bytes moved, and in all.
    pub done: u64,
    /// See [`Status::done`].
    pub total: u64,
    /// Files still to move.
    pub files_left: u64,
    /// Files in use, left for a later pass.
    pub waiting: u64,
    /// Why it stopped: it waits for [`Mover::retry`].
    pub failure: Option<String>,
    /// Every move in the journal is done.
    pub finished: bool,
}

impl Status {
    /// The share moved (0 to 1).
    pub fn share(&self) -> f64 {
        if self.total == 0 {
            if self.finished {
                1.0
            } else {
                0.0
            }
        } else {
            (self.done as f64 / self.total as f64).clamp(0.0, 1.0)
        }
    }
}

/// The journal's moves under way, on a thread of their own.
pub struct Mover {
    job: Arc<Job<Status>>,
}

impl Mover {
    /// Starts the moves in `settings`'s journal; `None` when there are none.
    pub fn start(settings: PathBuf, notify: Notify) -> Option<Mover> {
        let moves = journal(&settings);
        if moves.is_empty() {
            return None;
        }
        for (from, to) in &moves {
            register(from, to);
        }
        let job = Job::new(Status::default(), notify);
        let worker = Arc::clone(&job);
        std::thread::Builder::new()
            .name("mover".into())
            .spawn(move || run(&settings, &worker))
            .expect("a thread for the mover");
        Some(Mover { job })
    }

    /// Where it stands.
    pub fn status(&self) -> Status {
        self.job.status()
    }

    /// After a failure (a full disk): goes on.
    pub fn retry(&self) {
        self.job.retry();
    }
}

/// A move of the data folder as the settings page shows it
/// ([`DataMove`]): the move under way, where it goes, and where it stood
/// when last shown.
pub struct MoveView {
    mover: Option<Arc<Mover>>,
    to: Option<PathBuf>,
    seen: Option<Status>,
}

impl MoveView {
    /// Showing `mover` (one found under way at the start), going to `to`.
    pub fn new(mover: Option<Arc<Mover>>, to: Option<PathBuf>) -> MoveView {
        MoveView {
            mover,
            to,
            seen: None,
        }
    }

    /// Whether a move is still under way.
    pub fn busy(&self) -> bool {
        self.mover.as_ref().is_some_and(|m| !m.status().finished)
    }

    /// Moves the folder `from` to `to`: written down in the journal in
    /// `settings` first (from now on what is written goes there), then the
    /// files moved in the background, telling `notify`; shown in `shown`.
    /// Why it could not be written down.
    pub fn begin(
        &mut self,
        settings: &Path,
        (from, to): (PathBuf, PathBuf),
        notify: Notify,
        shown: &mut Option<DataMove>,
    ) -> Result<(), String> {
        begin(settings, &[(from, to.clone())])
            .map_err(|e| format!("Could not note the move in {}: {e}", settings.display()))?;
        self.to = Some(to.clone());
        self.seen = None;
        self.mover = Mover::start(settings.to_owned(), notify).map(Arc::new);
        *shown = Some(DataMove {
            to,
            // Nothing there to move: it is where it goes already.
            share: if self.mover.is_none() { 1.0 } else { 0.0 },
            finished: self.mover.is_none(),
            ..Default::default()
        });
        Ok(())
    }

    /// Where the move stands, into `shown` when that changed: whether it
    /// did.
    pub fn pump(&mut self, shown: &mut Option<DataMove>) -> bool {
        let Some(mover) = &self.mover else {
            return false;
        };
        let status = mover.status();
        if self.seen.as_ref() == Some(&status) {
            return false;
        }
        *shown = Some(DataMove {
            to: self.to.clone().unwrap_or_default(),
            share: status.share(),
            done: status.done,
            total: status.total,
            failure: status.failure.clone(),
            finished: status.finished,
        });
        self.seen = Some(status);
        true
    }

    /// After a failure (a full disk): goes on.
    pub fn retry(&self) {
        if let Some(mover) = &self.mover {
            mover.retry();
        }
    }
}

/// A file to move: where it is, where it goes, its length.
struct Item {
    from: PathBuf,
    to: PathBuf,
    bytes: u64,
}

/// Every file under `from`, with where it goes under `to`.
fn items(from: &Path, to: &Path) -> Vec<Item> {
    let mut out = Vec::new();
    let mut stack = vec![from.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(meta) = fs::symlink_metadata(&path) else {
                continue;
            };
            if meta.is_dir() {
                stack.push(path);
            } else if let Ok(rest) = path.strip_prefix(from) {
                out.push(Item {
                    to: to.join(rest),
                    from: path,
                    bytes: meta.len(),
                });
            }
        }
    }
    out
}

/// Removes the empty folders under `dir`, and `dir` if it is left empty.
fn prune(dir: &Path) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                prune(&path);
            }
        }
    }
    let _ = fs::remove_dir(dir);
}

/// Whether path is surely gone: the system says it is not found. Any
/// other answer (in use, being scanned, no access) says nothing, and a
/// file or folder is never taken for moved, or struck from the journal, on
/// it: `Path::exists` would say no then.
fn gone(path: &Path) -> bool {
    matches!(fs::symlink_metadata(path), Err(e) if e.kind() == io::ErrorKind::NotFound)
}

/// Tests: the moves in the journal in `settings` made to the end.
#[cfg(test)]
pub(crate) fn finish(settings: &Path) {
    let told = crate::told::Told::new();
    let mover = Mover::start(settings.to_owned(), told.notify()).expect("a move written down");
    let s = told.until("the move settles", || {
        let s = mover.status();
        (s.finished || s.failure.is_some()).then_some(s)
    });
    assert!(s.finished && s.failure.is_none(), "{s:?}");
}

/// Tests copy, as between two disks, where renaming would do.
#[cfg(test)]
static COPY_ONLY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn copy_only() -> bool {
    #[cfg(test)]
    return COPY_ONLY.load(std::sync::atomic::Ordering::SeqCst);
    #[cfg(not(test))]
    false
}

/// What moving one file came to.
enum Moved {
    Done,
    /// In use, or gone missing for now: a later pass.
    Later,
    /// The disk is full (or another error that needs the user).
    Stop(String),
}

fn full(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::StorageFull | io::ErrorKind::QuotaExceeded
    ) || e.raw_os_error() == Some(if cfg!(windows) { 112 } else { 28 })
}

/// Moves one file (see the module's steps), telling `progress` the bytes
/// copied so far.
#[expect(
    clippy::disallowed_methods,
    reason = "debt (W-B): the data move's rename on one disk, and across two its copy (fsync, then rename); to move onto durable"
)]
fn move_one(item: &Item, progress: &mut dyn FnMut(u64)) -> Moved {
    if gone(&item.from) {
        return Moved::Done;
    }
    use durable::sync_dir;
    let from_dir = item.from.parent().unwrap_or(Path::new("."));
    let to_dir = item.to.parent().unwrap_or(Path::new("."));
    let there = match fs::symlink_metadata(&item.to) {
        Ok(_) => true,
        Err(e) if e.kind() == io::ErrorKind::NotFound => false,
        // Cannot be told now (in use, scanned): later.
        Err(_) => return Moved::Later,
    };
    if there {
        return match fs::remove_file(&item.from) {
            Ok(()) => {
                let _ = sync_dir(from_dir);
                Moved::Done
            }
            Err(_) => Moved::Later,
        };
    }
    if let Err(e) = durable::create_dir_all(to_dir) {
        return Moved::Stop(format!("{}: {e}", to_dir.display()));
    }
    // One disk: renamed (atomic), the new name then the old one's going
    // made durable.
    if !copy_only() && fs::rename(&item.from, &item.to).is_ok() {
        let _ = sync_dir(to_dir);
        let _ = sync_dir(from_dir);
        progress(item.bytes);
        return Moved::Done;
    }
    let mut name = item.to.file_name().unwrap_or_default().to_os_string();
    name.push(".moving");
    let part = item.to.with_file_name(name);
    let copied = (|| -> io::Result<()> {
        let mut src = fs::File::open(&item.from)?;
        let mut dst = fs::File::create(&part)?;
        let mut buf = vec![0u8; 1 << 20];
        let mut done = 0u64;
        loop {
            let n = src.read(&mut buf)?;
            if n == 0 {
                break;
            }
            dst.write_all(&buf[..n])?;
            done += n as u64;
            progress(done);
        }
        // The copy's bytes on the disk, then its name: only once both are
        // there may the old one go.
        dst.sync_all()?;
        drop(dst);
        fs::rename(&part, &item.to)?;
        sync_dir(to_dir)
    })();
    match copied {
        Ok(()) => match fs::remove_file(&item.from) {
            // The copy is whole; the old one goes when it is no longer held.
            Ok(()) => {
                let _ = sync_dir(from_dir);
                Moved::Done
            }
            Err(_) => Moved::Later,
        },
        Err(e) => {
            let _ = fs::remove_file(&part);
            if full(&e) {
                Moved::Stop(format!(
                    "The disk {} is on is full.",
                    item.to.parent().unwrap_or(&item.to).display()
                ))
            } else if gone(&item.from) {
                // It went meanwhile (another writer's newer copy won).
                Moved::Done
            } else {
                Moved::Later
            }
        }
    }
}

fn run(settings: &Path, job: &Job<Status>) {
    // Tells the settings page, at most every 100 ms between files' ends.
    let mut teller = job.teller(Duration::from_millis(100));
    loop {
        let moves = journal(settings);
        if moves.is_empty() {
            teller.set(true, |s| s.finished = true);
            return;
        }
        let mut all: Vec<Item> = moves.iter().flat_map(|(f, t)| items(f, t)).collect();
        let total: u64 = all.iter().map(|i| i.bytes).sum();
        teller.set(true, |s| {
            // Over several starts the share counts what is left.
            *s = Status {
                total,
                files_left: all.len() as u64,
                ..Status::default()
            }
        });
        let mut done_bytes = 0u64;
        let mut stopped = None;
        let mut passes = 0;
        while !all.is_empty() && stopped.is_none() {
            let mut later = Vec::new();
            for item in all {
                let before = done_bytes;
                let moved = move_one(&item, &mut |bytes| {
                    teller.set(false, |s| s.done = before + bytes);
                });
                match moved {
                    Moved::Done => {
                        done_bytes = before + item.bytes;
                        teller.set(false, |s| {
                            s.done = done_bytes;
                            s.files_left -= 1;
                        });
                    }
                    Moved::Later => later.push(item),
                    Moved::Stop(why) => {
                        stopped = Some(why);
                        break;
                    }
                }
            }
            all = later;
            if !all.is_empty() && stopped.is_none() {
                passes += 1;
                let waiting = all.len() as u64;
                teller.set(true, |s| s.waiting = waiting);
                // Files held open: a while, then again.
                std::thread::sleep(Duration::from_secs(if passes < 5 { 1 } else { 5 }));
            }
        }
        if let Some(why) = stopped {
            let left = all
                .iter()
                .map(|i| i.bytes)
                .sum::<u64>()
                .saturating_sub(done_bytes);
            let failure = format!(
                "{why} {} MB are still to move. Free some space there, then try again.",
                left.div_ceil(1_000_000)
            );
            teller.set(true, |s| s.failure = Some(failure));
            job.wait_for_retry();
            continue;
        }
        // Every file over: the folders go, the journal is done with.
        for (from, _) in &moves {
            prune(from);
        }
        let left: Vec<_> = journal(settings)
            .into_iter()
            .filter(|(f, _)| !gone(f))
            .collect();
        let _ = write_journal(settings, &left);
        if !left.is_empty() && done_bytes == 0 {
            // A round that moved nothing, with folders still to go (what is
            // in them cannot be listed now): a while before the next.
            std::thread::sleep(Duration::from_secs(5));
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::disallowed_methods,
    reason = "tests: the folders to move, and a journal and part-moved files as a crash left them"
)]
mod tests {
    use super::*;
    use crate::told::Told;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("mover-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The move started on the journal in `settings`, and what it tells.
    fn start(settings: &Path) -> (Mover, Arc<Told>) {
        let told = Told::new();
        let mover = Mover::start(settings.to_owned(), told.notify()).unwrap();
        (mover, told)
    }

    /// Where the move ends: finished or failed.
    fn settle(mover: &Mover, told: &Told) -> Status {
        told.until("the move settles", || {
            let s = mover.status();
            (s.finished || s.failure.is_some()).then_some(s)
        })
    }

    #[test]
    fn a_folder_moves_whole_and_the_journal_is_done_with() {
        let dir = temp("whole");
        let settings = dir.join("settings");
        let (from, to) = (dir.join("old"), dir.join("new/data"));
        fs::create_dir_all(from.join("stems")).unwrap();
        fs::write(from.join("stems/a.stems"), vec![1u8; 3000]).unwrap();
        fs::write(from.join("b.txt"), b"bee").unwrap();
        begin(&settings, &[(from.clone(), to.clone())]).unwrap();
        assert_eq!(journal(&settings).len(), 1);
        // Written from now on: into the new place.
        assert_eq!(current(&from.join("c.txt")), to.join("c.txt"));
        // Read before it moved: from the old place.
        assert_eq!(locate(&from.join("b.txt")), from.join("b.txt"));
        let (mover, told) = start(&settings);
        let s = settle(&mover, &told);
        assert!(s.finished && s.failure.is_none(), "{s:?}");
        assert_eq!(fs::read(to.join("stems/a.stems")).unwrap().len(), 3000);
        assert_eq!(fs::read(to.join("b.txt")).unwrap(), b"bee");
        assert!(!from.exists(), "the old folder is gone");
        assert!(journal(&settings).is_empty());
        assert!(!settings.join(JOURNAL).exists());
        assert_eq!(locate(&from.join("b.txt")), to.join("b.txt"));
    }

    #[test]
    fn resuming_registers_the_journal_and_leaves_it_as_it_is() {
        let dir = temp("registered");
        let settings = dir.join("settings");
        let (from, to) = (dir.join("old"), dir.join("new"));
        fs::create_dir_all(&settings).unwrap();
        // Nothing written down: nothing to register, no journal made.
        resume(&settings);
        assert!(!settings.join(JOURNAL).exists());
        // An earlier run's journal: registered as it is.
        let text = format!("{}\t{}\n", from.display(), to.display());
        fs::write(settings.join(JOURNAL), &text).unwrap();
        resume(&settings);
        assert_eq!(current(&from.join("a.txt")), to.join("a.txt"));
        assert_eq!(fs::read_to_string(settings.join(JOURNAL)).unwrap(), text);
    }

    #[test]
    fn an_interrupted_move_goes_on_and_the_newer_copy_wins() {
        let dir = temp("resume");
        let settings = dir.join("settings");
        let (from, to) = (dir.join("old"), dir.join("new"));
        fs::create_dir_all(&from).unwrap();
        fs::create_dir_all(&to).unwrap();
        for name in ["one", "two", "three"] {
            fs::write(from.join(name), name.repeat(100)).unwrap();
        }
        begin(&settings, &[(from.clone(), to.clone())]).unwrap();
        // As if it stopped mid-way: "one" moved; "two" copied but not yet
        // removed where it was; a half copy of "three" left behind; and the
        // player wrote a newer "two" since.
        fs::rename(from.join("one"), to.join("one")).unwrap();
        fs::write(to.join("two"), b"newer").unwrap();
        fs::write(to.join("three.moving"), b"thr").unwrap();
        let (mover, told) = start(&settings);
        let s = settle(&mover, &told);
        assert!(s.finished, "{s:?}");
        assert_eq!(
            fs::read(to.join("two")).unwrap(),
            b"newer",
            "the newer stays"
        );
        assert_eq!(
            fs::read(to.join("three")).unwrap(),
            "three".repeat(100).as_bytes()
        );
        assert_eq!(
            fs::read(to.join("one")).unwrap(),
            "one".repeat(100).as_bytes()
        );
        assert!(!from.exists());
        assert!(journal(&settings).is_empty());
    }

    /// Between two disks a file is copied: whole, its bytes as they were,
    /// the old one removed only after; nothing `.moving` left.
    #[test]
    fn between_disks_a_file_is_copied_whole_then_removed() {
        let dir = temp("copy");
        let (from, to) = (dir.join("old"), dir.join("new"));
        fs::create_dir_all(from.join("deep/er")).unwrap();
        let big: Vec<u8> = (0..3_000_000u32).map(|i| (i * 7 % 251) as u8).collect();
        fs::write(from.join("deep/er/big.stems"), &big).unwrap();
        fs::write(from.join("small.txt"), b"s").unwrap();
        COPY_ONLY.store(true, std::sync::atomic::Ordering::SeqCst);
        let mut seen = Vec::new();
        let items = items(&from, &to);
        for item in &items {
            assert!(matches!(move_one(item, &mut |b| seen.push(b)), Moved::Done));
        }
        COPY_ONLY.store(false, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(fs::read(to.join("deep/er/big.stems")).unwrap(), big);
        assert_eq!(fs::read(to.join("small.txt")).unwrap(), b"s");
        assert!(!from.join("deep/er/big.stems").exists());
        assert!(seen.contains(&3_000_000), "the bytes copied told");
        let moving = items.iter().any(|i| {
            let mut n = i.to.file_name().unwrap().to_os_string();
            n.push(".moving");
            i.to.with_file_name(n).exists()
        });
        assert!(!moving, "no half copy left");
    }

    /// Only "not found" is gone: a file or folder that cannot be looked at
    /// is never taken for moved.
    #[test]
    fn only_not_found_is_gone() {
        let dir = temp("gone");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("here"), b"x").unwrap();
        assert!(!gone(&dir.join("here")));
        assert!(!gone(&dir));
        assert!(gone(&dir.join("not here")));
        // Under a file (no folder there): not found either way.
        assert!(gone(&dir.join("not here").join("deeper")));
    }

    /// A file another program holds (as a virus scanner does, for a moment)
    /// is left for a later pass: the move is not done, and the journal stays,
    /// until it has moved too.
    #[cfg(windows)]
    #[test]
    fn a_file_held_open_waits_and_the_journal_stays_until_it_moves() {
        use std::os::windows::fs::OpenOptionsExt;
        let dir = temp("held");
        let settings = dir.join("settings");
        let (from, to) = (dir.join("old"), dir.join("new"));
        fs::create_dir_all(&from).unwrap();
        for name in ["a", "b", "held"] {
            fs::write(from.join(name), name).unwrap();
        }
        // Nobody else may open it, read or rename it while this is held.
        let held = fs::OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(from.join("held"))
            .unwrap();
        begin(&settings, &[(from.clone(), to.clone())]).unwrap();
        let (mover, told) = start(&settings);
        // Its first pass over: the held one waits.
        let s = told.until("the first pass", || {
            let s = mover.status();
            (s.waiting > 0 || s.finished || s.failure.is_some()).then_some(s)
        });
        assert!(!s.finished, "not done while a file is held: {s:?}");
        assert_eq!(s.waiting, 1, "{s:?}");
        assert!(to.join("a").exists() && to.join("b").exists());
        assert!(from.join("held").exists());
        assert_eq!(journal(&settings).len(), 1, "the journal stays");
        drop(held);
        let s = settle(&mover, &told);
        assert!(s.finished, "{s:?}");
        assert_eq!(fs::read(to.join("held")).unwrap(), b"held");
        assert!(gone(&from));
        assert!(journal(&settings).is_empty());
    }

    /// While a move is under way a file is read in either place, and one
    /// removed goes from both: it does not come over afterwards.
    #[test]
    fn a_file_is_read_in_either_place_and_removed_from_both() {
        let dir = temp("either");
        let settings = dir.join("settings");
        let (from, to) = (dir.join("old"), dir.join("new"));
        fs::create_dir_all(&from).unwrap();
        fs::create_dir_all(&to).unwrap();
        fs::write(from.join("old.txt"), b"not moved yet").unwrap();
        fs::write(from.join("gone.txt"), b"stale").unwrap();
        fs::write(to.join("gone.txt"), b"newer").unwrap();
        begin(&settings, &[(from.clone(), to.clone())]).unwrap();
        // By the new place's name and by the old one's.
        assert_eq!(read(&to.join("old.txt")).unwrap(), b"not moved yet");
        assert_eq!(read(&from.join("old.txt")).unwrap(), b"not moved yet");
        assert_eq!(read(&to.join("gone.txt")).unwrap(), b"newer");
        assert_eq!(
            read(&to.join("none.txt")).unwrap_err().kind(),
            io::ErrorKind::NotFound
        );
        remove(&to.join("gone.txt")).unwrap();
        remove(&to.join("none.txt")).unwrap();
        finish(&settings);
        assert_eq!(fs::read(to.join("old.txt")).unwrap(), b"not moved yet");
        assert!(!to.join("gone.txt").exists(), "the stale copy stayed away");
    }

    #[test]
    fn nothing_in_the_journal_starts_nothing() {
        let dir = temp("none");
        assert!(Mover::start(dir.join("settings"), Arc::new(|| {})).is_none());
        // A folder that is not there is not written down.
        begin(
            &dir.join("settings"),
            &[(dir.join("absent"), dir.join("to"))],
        )
        .unwrap();
        assert!(journal(&dir.join("settings")).is_empty());
    }
}
