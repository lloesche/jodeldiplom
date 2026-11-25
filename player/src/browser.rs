//! The file browser: one folder at a time, no database.
//!
//! - [`list_dir`] lists a folder's subfolders and playable files (FLAC,
//!   WAV, AIFF, CAF, M4A/MP4, MP3/MP2/MP1, Ogg: [`PLAYABLE`]), folders
//!   first, in natural order (`2` before `10`), with a parent row when there
//!   is one.
//! - [`read_details`] reads what a row shows about a file (the format chip
//!   such as `FLAC 44.1`, `ALAC 96` or `MP3 320k`, the title
//!   and artist from the tags, the length). [`quick_details`] reads it from
//!   the file's headers alone; [`DetailsWorker`] does it in the background,
//!   for the rows on screen only; the results are kept in memory for the
//!   session.
//! - [`search`] walks below a folder for names containing a text, streaming
//!   what it finds; [`walk_playable`] collects every playable file below a
//!   folder, for queueing it.

use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use jodel_playback::TrackInfo;

use crate::core::text::khz_short;
use crate::core::Action;

/// Something to call after a result was sent, so a sleeping UI loop wakes
/// up (the window frontend posts an event to its loop).
pub type Notify = Arc<dyn Fn() + Send + Sync>;

/// What a row is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EntryKind {
    /// `..`: the folder above.
    Parent,
    /// A folder.
    Dir,
    /// A playable file.
    File,
    /// A playlist (M3U or M3U8): its tracks play, or are queued.
    Playlist,
}

/// One row of a folder or of the search results.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// The file or folder name (`..` for the parent row).
    pub name: String,
    /// Where it is.
    pub path: PathBuf,
    /// What it is.
    pub kind: EntryKind,
    /// For search results: the path below the searched folder.
    pub relative: Option<String>,
}

impl Entry {
    fn new(path: PathBuf, kind: EntryKind) -> Entry {
        let name = match kind {
            EntryKind::Parent => "..".to_owned(),
            _ => path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.to_string_lossy().into_owned()),
        };
        Entry {
            name,
            path,
            kind,
            relative: None,
        }
    }

    /// The `..` row leading to `parent`.
    pub fn parent(parent: PathBuf) -> Entry {
        Entry::new(parent, EntryKind::Parent)
    }

    /// The text a list row starts with: the relative path for search
    /// results, else the name.
    pub fn label(&self) -> &str {
        self.relative.as_deref().unwrap_or(&self.name)
    }
}

/// What a row shows about a file, read from it once per session.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Details {
    /// The format chip ([`chip`]): `FLAC 44.1`, `WAV 96`, `AIFF 96`,
    /// `ALAC 44.1`, `MP3 320k`, `VORBIS 160k`; `?` when the file could not
    /// be read.
    pub chip: String,
    /// The title tag.
    pub title: Option<String>,
    /// The artist tag.
    pub artist: Option<String>,
    /// The length in seconds.
    pub seconds: Option<f64>,
    /// Why the file could not be read.
    pub error: Option<String>,
}

impl Details {
    /// The details of an opened file.
    pub fn from_info(info: &TrackInfo) -> Details {
        Details {
            chip: chip(info),
            title: info.tags.title.clone().filter(|t| !t.is_empty()),
            artist: info.tags.artist.clone().filter(|t| !t.is_empty()),
            seconds: info.duration().map(|d| d.as_secs_f64()),
            error: None,
        }
    }

    /// The details of a file that could not be read.
    pub fn unreadable(error: String) -> Details {
        Details {
            chip: "?".into(),
            error: Some(error),
            ..Details::default()
        }
    }
}

/// The format chip a row shows for a file:
///
/// - lossy: the codec and the bit rate in kbit/s (`MP3 320k`, `VORBIS
///   160k`; for a VBR MP3 file its average), else the codec and the rate
///   (`MP3 44.1`);
/// - lossless: the codec, or the container for PCM, and the rate in kHz
///   (`FLAC 44.1`, `ALAC 96`, `WAV 48`, `AIFF 96`, `CAF 192`).
pub fn chip(info: &TrackInfo) -> String {
    let name = info.format_name().to_uppercase();
    match info.bitrate.filter(|_| info.is_lossy()) {
        Some(bitrate) => format!("{name} {}k", (bitrate + 500) / 1000),
        None => format!("{name} {}", khz_short(info.rate)),
    }
}

/// The extensions the browser lists, in lower case: FLAC; WAV; AIFF; CAF;
/// MP4 audio (ALAC and AAC-LC play; HE-AAC is reported as not supported);
/// MPEG audio layers III, II and I; Ogg Vorbis.
pub const PLAYABLE: [&str; 13] = [
    "flac", "wav", "aif", "aiff", "aifc", "caf", "m4a", "mp4", "mp3", "mp2", "mp1", "ogg", "oga",
];

/// Whether the player can play `path` (by its extension, [`PLAYABLE`]).
pub fn is_playable(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| PLAYABLE.iter().any(|p| e.eq_ignore_ascii_case(p)))
}

fn is_hidden(name: &str) -> bool {
    name.starts_with('.')
}

/// Compares names the way people count: case ignored, runs of digits by
/// their value (`track 2` before `track 10`).
pub fn natural_cmp(a: &str, b: &str) -> Ordering {
    let (mut a, mut b) = (a.chars().peekable(), b.chars().peekable());
    loop {
        match (a.peek().copied(), b.peek().copied()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(x), Some(y)) if x.is_ascii_digit() && y.is_ascii_digit() => {
                let take = |it: &mut std::iter::Peekable<std::str::Chars>| {
                    let mut digits = String::new();
                    while let Some(c) = it.peek().copied().filter(char::is_ascii_digit) {
                        digits.push(c);
                        it.next();
                    }
                    digits
                };
                let (da, db) = (take(&mut a), take(&mut b));
                let (ta, tb) = (da.trim_start_matches('0'), db.trim_start_matches('0'));
                let order = ta.len().cmp(&tb.len()).then_with(|| ta.cmp(tb));
                if order != Ordering::Equal {
                    return order;
                }
            }
            (Some(x), Some(y)) => {
                let order = x.to_lowercase().cmp(y.to_lowercase());
                if order != Ordering::Equal {
                    return order;
                }
                a.next();
                b.next();
            }
        }
    }
}

fn sort_entries(entries: &mut [Entry]) {
    entries.sort_by(|x, y| {
        let rank = |e: &Entry| match e.kind {
            EntryKind::Parent => 0,
            EntryKind::Dir => 1,
            EntryKind::Playlist => 2,
            EntryKind::File => 3,
        };
        rank(x)
            .cmp(&rank(y))
            .then_with(|| natural_cmp(x.label(), y.label()))
    });
}

/// The folder above `dir`. On Windows the root of a drive has the list of
/// drives above it, which is the empty path (see [`is_drives`]).
pub fn parent_of(dir: &Path) -> Option<PathBuf> {
    if let Some(parent) = dir.parent() {
        return Some(parent.to_path_buf());
    }
    if cfg!(windows) && !dir.as_os_str().is_empty() {
        return Some(PathBuf::new());
    }
    None
}

/// Whether `dir` is the list of drives (Windows; the empty path).
pub fn is_drives(dir: &Path) -> bool {
    cfg!(windows) && dir.as_os_str().is_empty()
}

/// What a path typed after `/` names, if it exists: a folder to show, or a
/// file's folder with the file to select. Relative paths start at `from`;
/// `~` is the home folder; on Windows `D:` is the root of drive D.
pub fn resolve_jump(text: &str, from: &Path) -> Option<(PathBuf, Option<PathBuf>)> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let mut path = PathBuf::from(text);
    if let Some(rest) = text
        .strip_prefix('~')
        .filter(|rest| rest.is_empty() || rest.starts_with(['/', '\\']))
    {
        let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
        path = PathBuf::from(home).join(rest.trim_start_matches(['/', '\\']));
    }
    if cfg!(windows) {
        let bytes = text.as_bytes();
        if bytes.len() == 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
            path = PathBuf::from(format!("{text}\\"));
        }
    }
    if path.is_relative() {
        path = from.join(path);
    }
    let path = normalize(&path);
    let meta = fs::metadata(&path).ok()?;
    if meta.is_dir() {
        Some((path, None))
    } else if meta.is_file() {
        let dir = path.parent()?.to_path_buf();
        Some((dir, Some(path)))
    } else {
        None
    }
}

/// `path` with `.` dropped and `..` applied to the part before it.
fn normalize(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for part in path.components() {
        match part {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push(part);
                }
            }
            other => out.push(other),
        }
    }
    out
}

/// The drives that exist, `A:\` to `Z:\`, as folders.
fn list_drives() -> Vec<Entry> {
    (b'A'..=b'Z')
        .map(|letter| PathBuf::from(format!("{}:\\", letter as char)))
        .filter(|root| fs::metadata(root).is_ok_and(|m| m.is_dir()))
        .map(|root| Entry::new(root, EntryKind::Dir))
        .collect()
}

/// The subfolders and playable files of `dir`, folders first, with a
/// parent row when `dir` has a parent. Hidden names (starting with `.`) are
/// left out. On Windows the empty path lists the drives.
pub fn list_dir(dir: &Path) -> io::Result<Vec<Entry>> {
    if is_drives(dir) {
        return Ok(list_drives());
    }
    let mut entries = Vec::new();
    if let Some(parent) = parent_of(dir) {
        entries.push(Entry::new(parent, EntryKind::Parent));
    }
    for item in fs::read_dir(dir)? {
        let Ok(item) = item else { continue };
        let name = item.file_name().to_string_lossy().into_owned();
        if is_hidden(&name) {
            continue;
        }
        let path = item.path();
        // The listing itself says what each entry is; asking the file
        // system again per entry costs a network round trip each on a NAS.
        // Only links are followed (a linked folder is browsed like a folder).
        let Ok(mut kind) = item.file_type() else {
            continue;
        };
        if kind.is_symlink() {
            let Ok(meta) = fs::metadata(&path) else {
                continue;
            };
            kind = meta.file_type();
        }
        if kind.is_dir() {
            entries.push(Entry::new(path, EntryKind::Dir));
        } else if kind.is_file() && is_playable(&path) {
            entries.push(Entry::new(path, EntryKind::File));
        } else if kind.is_file() && crate::session::is_playlist(&path) {
            entries.push(Entry::new(path, EntryKind::Playlist));
        }
    }
    sort_entries(&mut entries);
    Ok(entries)
}

/// Whether an entry matches a filter text: its name, or the title read for
/// it, contains the text (case ignored). The parent row never matches.
pub fn matches(entry: &Entry, text: &str, details: Option<&Details>) -> bool {
    if entry.kind == EntryKind::Parent {
        return false;
    }
    let needle = text.to_lowercase();
    entry.label().to_lowercase().contains(&needle)
        || details
            .and_then(|d| d.title.as_ref())
            .is_some_and(|t| t.to_lowercase().contains(&needle))
}

/// Every playable file below `root` (or `root` itself, if it is one): in
/// each folder its files first, then its subfolders, in natural order.
pub fn walk_playable(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if root.is_file() {
        if is_playable(root) {
            out.push(root.to_path_buf());
        }
        return out;
    }
    walk_into(root, &mut out, 0);
    out
}

fn walk_into(dir: &Path, out: &mut Vec<PathBuf>, depth: usize) {
    if depth > 64 {
        return;
    }
    let Ok(entries) = list_dir(dir) else { return };
    let mut dirs = Vec::new();
    for entry in entries {
        match entry.kind {
            EntryKind::File => out.push(entry.path),
            EntryKind::Dir => dirs.push(entry.path),
            EntryKind::Parent | EntryKind::Playlist => {}
        }
    }
    for sub in dirs {
        walk_into(&sub, out, depth + 1);
    }
}

/// How many hits a search reports at most.
pub const SEARCH_LIMIT: usize = 5_000;

/// Walks below `root` (breadth first) for folders and playable files whose
/// name contains `query` (case ignored), handing batches of hits to
/// `found` as it goes. Stops early when `cancel` is set or after
/// [`SEARCH_LIMIT`] hits.
pub fn search(root: &Path, query: &str, cancel: &AtomicBool, mut found: impl FnMut(Vec<Entry>)) {
    let needle = query.to_lowercase();
    let mut queue = VecDeque::from([root.to_path_buf()]);
    let mut batch = Vec::new();
    let mut total = 0;
    let mut last_send = Instant::now();
    while let Some(dir) = queue.pop_front() {
        if cancel.load(AtomicOrdering::Relaxed) {
            return;
        }
        let Ok(entries) = list_dir(&dir) else {
            continue;
        };
        for mut entry in entries {
            if entry.kind == EntryKind::Parent {
                continue;
            }
            if entry.kind == EntryKind::Dir {
                queue.push_back(entry.path.clone());
            }
            if entry.name.to_lowercase().contains(&needle) {
                let relative = entry
                    .path
                    .strip_prefix(root)
                    .unwrap_or(&entry.path)
                    .to_string_lossy()
                    .replace('\\', "/");
                entry.relative = Some(relative);
                batch.push(entry);
                total += 1;
                if total >= SEARCH_LIMIT {
                    found(std::mem::take(&mut batch));
                    return;
                }
            }
        }
        if !batch.is_empty()
            && (batch.len() >= 64 || last_send.elapsed() > Duration::from_millis(60))
        {
            found(std::mem::take(&mut batch));
            last_send = Instant::now();
        }
    }
    if !batch.is_empty() {
        found(batch);
    }
}

/// Reads the details of one file.
pub fn read_details(path: &Path) -> Details {
    match jodel_playback::probe(path) {
        Ok(info) => Details::from_info(&info),
        Err(error) => Details::unreadable(error.to_string()),
    }
}

/// What a FLAC file's metadata blocks say, read without its audio or its
/// pictures (they are skipped, not read).
#[derive(Clone, Debug, Default, PartialEq)]
struct FlacHeader {
    rate: u32,
    total_frames: u64,
    title: Option<String>,
    artist: Option<String>,
}

fn read_flac_header(path: &Path) -> io::Result<FlacHeader> {
    use std::io::{BufReader, Read};
    let invalid = |what: &str| io::Error::new(io::ErrorKind::InvalidData, what.to_owned());
    let mut file = BufReader::with_capacity(16 * 1024, fs::File::open(path)?);
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != b"fLaC" {
        return Err(invalid("not a FLAC file"));
    }
    let mut header = FlacHeader::default();
    let mut seen_info = false;
    loop {
        let mut block = [0u8; 4];
        file.read_exact(&mut block)?;
        let last = block[0] & 0x80 != 0;
        let kind = block[0] & 0x7f;
        let len = u32::from_be_bytes([0, block[1], block[2], block[3]]) as usize;
        match kind {
            // STREAMINFO: rate (20 bits) and total frames (36 bits).
            0 if len >= 18 => {
                let mut info = vec![0u8; len];
                file.read_exact(&mut info)?;
                header.rate = (u32::from(info[10]) << 12)
                    | (u32::from(info[11]) << 4)
                    | (u32::from(info[12]) >> 4);
                header.total_frames = (u64::from(info[13] & 0x0f) << 32)
                    | u64::from(u32::from_be_bytes([info[14], info[15], info[16], info[17]]));
                seen_info = true;
            }
            // VORBIS_COMMENT: little-endian lengths, KEY=value entries.
            4 => {
                let mut data = vec![0u8; len];
                file.read_exact(&mut data)?;
                read_comments(&data, &mut header);
            }
            _ => file.seek_relative(len as i64)?,
        }
        if last {
            break;
        }
    }
    if !seen_info || header.rate == 0 {
        return Err(invalid("no stream info"));
    }
    Ok(header)
}

fn read_comments(data: &[u8], header: &mut FlacHeader) {
    let u32_at = |at: usize| {
        data.get(at..at + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
    };
    let Some(vendor) = u32_at(0) else { return };
    let mut at = 4 + vendor;
    let Some(count) = u32_at(at) else { return };
    at += 4;
    for _ in 0..count {
        let Some(len) = u32_at(at) else { return };
        at += 4;
        let Some(entry) = data.get(at..at + len) else {
            return;
        };
        at += len;
        let entry = String::from_utf8_lossy(entry);
        let Some((key, value)) = entry.split_once('=') else {
            continue;
        };
        let value = value.trim().to_owned();
        match key.to_ascii_uppercase().as_str() {
            "TITLE" if header.title.is_none() && !value.is_empty() => header.title = Some(value),
            "ARTIST" if header.artist.is_none() && !value.is_empty() => header.artist = Some(value),
            _ => {}
        }
    }
}

/// The details a row shows, from the file's headers alone, without
/// decoding audio: FLAC metadata is read directly (a few kilobytes at the
/// start of the file, pictures skipped); the other formats through the
/// demuxer's headers ([`jodel_playback::probe_header`]: rate, length, tags
/// from ID3v2, MP4 atoms or Vorbis comments).
pub fn quick_details(path: &Path) -> Details {
    let flac = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("flac"));
    if !flac {
        return match jodel_playback::probe_header(path) {
            Ok(info) => Details::from_info(&info),
            Err(error) => Details::unreadable(error.to_string()),
        };
    }
    match read_flac_header(path) {
        Ok(header) => Details {
            chip: format!("FLAC {}", khz_short(header.rate)),
            title: header.title,
            artist: header.artist,
            seconds: (header.total_frames > 0)
                .then(|| header.total_frames as f64 / f64::from(header.rate)),
            error: None,
        },
        Err(_) => read_details(path),
    }
}

struct Wanted {
    /// Files whose rows are on screen, most wanted first.
    queue: VecDeque<PathBuf>,
    stop: bool,
}

/// Reads file details on background threads, most recently wanted first.
///
/// [`DetailsWorker::want`] replaces what is wanted: files no longer wanted
/// that were not read yet are dropped (and reported back as
/// [`Action::DetailsDropped`], so they can be asked for again later).
pub struct DetailsWorker {
    shared: Arc<(Mutex<Wanted>, Condvar)>,
    sender: Sender<Action>,
    notify: Notify,
    threads: Vec<thread::JoinHandle<()>>,
}

impl DetailsWorker {
    /// Starts `threads` readers that send [`Action::Details`] to `sender`.
    pub fn new(threads: usize, sender: Sender<Action>, notify: Notify) -> DetailsWorker {
        let shared = Arc::new((
            Mutex::new(Wanted {
                queue: VecDeque::new(),
                stop: false,
            }),
            Condvar::new(),
        ));
        let threads = (0..threads.max(1))
            .map(|_| {
                let shared = shared.clone();
                let sender = sender.clone();
                let notify = notify.clone();
                thread::spawn(move || loop {
                    let path = {
                        let (lock, ready) = &*shared;
                        let mut wanted = lock.lock().unwrap_or_else(|e| e.into_inner());
                        loop {
                            if wanted.stop {
                                return;
                            }
                            if let Some(path) = wanted.queue.pop_front() {
                                break path;
                            }
                            wanted = ready.wait(wanted).unwrap_or_else(|e| e.into_inner());
                        }
                    };
                    let details = quick_details(&path);
                    if sender.send(Action::Details(path, details)).is_err() {
                        return;
                    }
                    notify();
                })
            })
            .collect();
        DetailsWorker {
            shared,
            sender,
            notify,
            threads,
        }
    }

    /// Wants exactly `paths` now, in this order.
    pub fn want(&self, paths: Vec<PathBuf>) {
        let (lock, ready) = &*self.shared;
        let mut wanted = lock.lock().unwrap_or_else(|e| e.into_inner());
        let dropped: Vec<PathBuf> = wanted
            .queue
            .drain(..)
            .filter(|p| !paths.contains(p))
            .collect();
        wanted.queue = paths.into();
        ready.notify_all();
        drop(wanted);
        if !dropped.is_empty() && self.sender.send(Action::DetailsDropped(dropped)).is_ok() {
            (self.notify)();
        }
    }
}

impl Drop for DetailsWorker {
    fn drop(&mut self) {
        {
            let (lock, ready) = &*self.shared;
            let mut wanted = lock.lock().unwrap_or_else(|e| e.into_inner());
            wanted.stop = true;
            ready.notify_all();
        }
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::disallowed_methods,
    reason = "tests: tracks and files in a temporary folder to browse"
)]
mod tests {
    use super::*;

    #[cfg(windows)]
    #[test]
    fn a_drive_root_has_the_drives_above_it() {
        assert_eq!(parent_of(Path::new("C:\\")), Some(PathBuf::new()));
        assert_eq!(
            parent_of(Path::new("C:\\Music")),
            Some(PathBuf::from("C:\\"))
        );
        assert_eq!(parent_of(Path::new("")), None);
        let drives = list_dir(Path::new("")).unwrap();
        assert!(drives
            .iter()
            .any(|e| e.kind == EntryKind::Dir && e.path == Path::new("C:\\")));
        assert!(drives.iter().all(|e| e.kind == EntryKind::Dir));
        let root = list_dir(Path::new("C:\\")).unwrap();
        assert_eq!(root[0].kind, EntryKind::Parent);
        assert!(is_drives(&root[0].path));
    }

    #[test]
    fn typed_paths_resolve_to_folders_and_files() {
        let root = std::env::temp_dir().join(format!("jodeldiplom-jump-{}", std::process::id()));
        let album = root.join("Album");
        fs::create_dir_all(&album).unwrap();
        let file = album.join("01 a.flac");
        fs::write(&file, b"x").unwrap();
        let root = normalize(&root);
        // Absolute and relative folders; `..`; a file selects itself.
        assert_eq!(
            resolve_jump(&album.to_string_lossy(), Path::new("/")),
            Some((normalize(&album), None))
        );
        assert_eq!(
            resolve_jump("Album", &root),
            Some((root.join("Album"), None))
        );
        assert_eq!(
            resolve_jump("Album/../Album/./01 a.flac", &root),
            Some((
                root.join("Album"),
                Some(root.join("Album").join("01 a.flac"))
            ))
        );
        // Anything that doesn't exist stays a filter.
        assert_eq!(resolve_jump("live", &root), None);
        assert_eq!(resolve_jump("  ", &root), None);
        if cfg!(windows) {
            let drive = &root.to_string_lossy()[..2];
            assert_eq!(
                resolve_jump(drive, &album),
                Some((PathBuf::from(format!("{drive}\\")), None))
            );
        }
        fs::remove_dir_all(&root).unwrap();
    }

    /// A FLAC header: STREAMINFO, a large picture block, then the comments.
    fn flac_header(rate: u32, frames: u64, comments: &[&str]) -> Vec<u8> {
        let mut out = b"fLaC".to_vec();
        let mut info = [0u8; 34];
        info[10] = (rate >> 12) as u8;
        info[11] = (rate >> 4) as u8;
        info[12] = ((rate & 0x0f) << 4) as u8 | (1 << 1);
        info[13] = 0x70 | ((frames >> 32) as u8 & 0x0f);
        info[14..18].copy_from_slice(&(frames as u32).to_be_bytes());
        out.extend([0, 0, 0, 34]);
        out.extend(info);
        let picture = vec![0xabu8; 300_000];
        let len = picture.len() as u32;
        out.extend([6, (len >> 16) as u8, (len >> 8) as u8, len as u8]);
        out.extend(picture);
        let mut c = Vec::new();
        c.extend(3u32.to_le_bytes());
        c.extend(b"enc");
        c.extend((comments.len() as u32).to_le_bytes());
        for entry in comments {
            c.extend((entry.len() as u32).to_le_bytes());
            c.extend(entry.as_bytes());
        }
        let len = c.len() as u32;
        out.extend([0x84, (len >> 16) as u8, (len >> 8) as u8, len as u8]);
        out.extend(c);
        out
    }

    #[test]
    fn rows_come_from_the_flac_header_alone() {
        let dir = std::env::temp_dir().join(format!("jodeldiplom-header-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let tagged = dir.join("tagged.flac");
        fs::write(
            &tagged,
            flac_header(
                48_000,
                48_000 * 90,
                &["TITLE=Song", "artist=Band", "ORIGINALSAMPLERATE=192000"],
            ),
        )
        .unwrap();
        let details = quick_details(&tagged);
        assert_eq!(details.chip, "FLAC 48");
        assert_eq!(details.title.as_deref(), Some("Song"));
        assert_eq!(details.artist.as_deref(), Some("Band"));
        assert_eq!(details.seconds, Some(90.0));
        let plain = dir.join("plain.flac");
        fs::write(&plain, flac_header(44_100, 44_100 * 3, &["TITLE=Other"])).unwrap();
        let details = quick_details(&plain);
        assert_eq!(details.chip, "FLAC 44.1");
        assert_eq!(details.title.as_deref(), Some("Other"));
        assert_eq!(details.seconds, Some(3.0));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_header_agrees_with_the_full_probe_on_the_fixtures() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/pcm");
        for name in ["ramp-44100-16bit.flac", "ramp-48000-24bit.flac"] {
            let path = fixtures.join(name);
            let full = read_details(&path);
            let quick = quick_details(&path);
            assert_eq!(quick.chip, full.chip, "{name}");
            assert_eq!(quick.title, full.title, "{name}");
            let (quick, full) = (quick.seconds.unwrap(), full.seconds.unwrap());
            assert!((quick - full).abs() < 1e-6, "{name}: {quick} {full}");
        }
    }

    #[test]
    fn natural_order_counts_numbers() {
        let mut names = vec!["Track 10", "track 2", "Track 1", "b", "A", "track 02x"];
        names.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(
            names,
            vec!["A", "b", "Track 1", "track 2", "track 02x", "Track 10"]
        );
    }

    #[test]
    fn the_supported_extensions_play() {
        for name in [
            "a/b.FLAC", "x.wav", "x.aif", "x.AIFF", "x.aifc", "x.caf", "x.m4a", "x.mp4", "x.MP3",
            "x.mp2", "x.mp1", "x.ogg", "x.oga",
        ] {
            assert!(is_playable(Path::new(name)), "{name}");
        }
        for name in [
            "cover.jpg",
            "noext",
            "x.opus",
            "x.aac",
            "x.wma",
            "x.m4b",
            "x.mkv",
        ] {
            assert!(!is_playable(Path::new(name)), "{name}");
        }
    }

    #[test]
    fn filters_match_names_and_titles() {
        let entry = Entry::new(PathBuf::from("/m/01 intro.flac"), EntryKind::File);
        let details = Details {
            title: Some("Morning Song".into()),
            ..Details::default()
        };
        assert!(matches(&entry, "INTRO", None));
        assert!(matches(&entry, "morning", Some(&details)));
        assert!(!matches(&entry, "evening", Some(&details)));
        let parent = Entry::new(PathBuf::from("/"), EntryKind::Parent);
        assert!(!matches(&parent, "", None));
    }
}
