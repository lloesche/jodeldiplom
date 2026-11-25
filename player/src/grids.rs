//! Beat grids fixed by hand, kept per track.
//!
//! The Transition tab's fixes ([`crate::core::grid`]) are the user's work, so
//! they live with the settings (`grids` next to `config.toml`), not with the
//! cached analyses, and outlast any new analysis: a track's grid is found by
//! its path and size only. One small text file per track: the beats as tick
//! deltas and the indices of the beats that start a bar, as in the rhythm
//! cache. [`GridStore`] writes them on a thread of its own, one at a time and
//! in order, so the last fix is the one kept.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::thread;

use crate::browser::Notify;
use crate::cache;
use crate::core::grid::Grid;
use crate::core::Action;
use crate::rhythm::{beats_text, parse_beats};

/// Where the grids are kept: `grids` in the settings folder (fixed by hand,
/// they are not data the player could make again).
pub fn dir() -> Option<PathBuf> {
    crate::dirs::settings_dir().map(|d| d.join("grids"))
}

/// What finds a track's grid: its path and size.
fn key(path: &Path) -> Option<String> {
    let size = fs::metadata(path).ok()?.len();
    Some(format!("{} {size}", path.display()))
}

/// The file of the grid stored under `key` in `dir`.
fn file(dir: &Path, key: &str) -> PathBuf {
    dir.join(format!("{}.txt", crate::cache::name(key)))
}

/// A grid as its file's text, under `key`, in [`cache::GRID`]'s version.
pub fn to_text(key: &str, grid: &Grid) -> String {
    let mut s = format!("jodel-grid 1\nkey {}\nbeats", key.replace('\n', " "));
    beats_text(&mut s, &grid.beats);
    s.push_str("\ndownbeats");
    for i in &grid.downbeats {
        let _ = write!(s, " {i}");
    }
    s.push('\n');
    cache::GRID.wrap(&s)
}

/// The grid in a file's text, if it was stored under `key` (in any
/// version of [`cache::GRID`] up to this player's: the user's work is
/// kept).
pub fn from_text(key: &str, text: &str) -> Option<Grid> {
    let (version, text) = cache::GRID.open(text)?;
    if version > cache::GRID.version {
        return None;
    }
    let mut lines = text.lines();
    if lines.next()? != "jodel-grid 1" {
        return None;
    }
    if lines.next()?.strip_prefix("key ")? != key.replace('\n', " ") {
        return None;
    }
    let mut grid = Grid::default();
    for line in lines {
        let (word, rest) = line.split_once(' ').unwrap_or((line, ""));
        match word {
            "beats" => grid.beats = parse_beats(rest)?,
            "downbeats" => {
                for i in rest.split_whitespace() {
                    let i: usize = i.parse().ok()?;
                    if i >= grid.beats.len() {
                        return None;
                    }
                    grid.downbeats.push(i);
                }
            }
            _ => return None,
        }
    }
    Some(grid)
}

/// The grid fixed by hand for `path`, if there is one.
pub fn load(dir: &Path, path: &Path) -> Option<Grid> {
    let key = key(path)?;
    from_text(&key, &fs::read_to_string(file(dir, &key)).ok()?)
}

/// Keeps `grid` for `path` (written whole, then renamed into place).
pub fn save(dir: &Path, path: &Path, grid: &Grid) -> std::io::Result<()> {
    let Some(key) = key(path) else {
        return Ok(());
    };
    // The grids are fixed by hand: whole or not at all, and durable.
    durable::write_atomic(&file(dir, &key), to_text(&key, grid).as_bytes())
}

/// Forgets the grid fixed by hand for `path`.
pub fn remove(dir: &Path, path: &Path) -> std::io::Result<()> {
    let Some(key) = key(path) else {
        return Ok(());
    };
    match fs::remove_file(file(dir, &key)) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}

/// What the store's thread does next.
enum Job {
    Save(PathBuf, Grid),
    Remove(PathBuf),
}

/// Keeps and forgets grids in a folder, on a thread of its own: in the
/// order asked, one at a time. A failure is said with
/// [`Action::Failed`].
pub struct GridStore {
    jobs: Sender<Job>,
}

impl GridStore {
    /// A store in `dir`, saying failures to `sender`.
    pub fn new(dir: PathBuf, sender: Sender<Action>, notify: Notify) -> GridStore {
        let (jobs, rx) = mpsc::channel();
        thread::spawn(move || {
            for job in rx {
                let failed = match job {
                    Job::Save(path, grid) => save(&dir, &path, &grid)
                        .err()
                        .map(|e| format!("The grid was not saved: {e}")),
                    Job::Remove(path) => remove(&dir, &path)
                        .err()
                        .map(|e| format!("The fixed grid was not removed: {e}")),
                };
                if let Some(text) = failed {
                    if sender.send(Action::Failed(text)).is_err() {
                        return;
                    }
                    notify();
                }
            }
        });
        GridStore { jobs }
    }

    /// Keeps `grid` for `path`.
    pub fn save(&self, path: PathBuf, grid: Grid) {
        let _ = self.jobs.send(Job::Save(path, grid));
    }

    /// Forgets the grid of `path`.
    pub fn remove(&self, path: PathBuf) {
        let _ = self.jobs.send(Job::Remove(path));
    }
}

#[cfg(test)]
#[expect(
    clippy::disallowed_methods,
    reason = "tests: a track whose size and bytes change under its grid"
)]
mod tests {
    use super::*;

    #[test]
    fn a_grid_is_kept_by_path_and_size() {
        let dir = std::env::temp_dir().join(format!("jodel-grids-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let track = dir.join("track.flac");
        fs::write(&track, b"0123456789").unwrap();
        let grid = Grid {
            beats: vec![0.5, 1.0, 1.5, 2.0001, 2.5],
            downbeats: vec![1],
        };
        let store = dir.join("grids");
        assert_eq!(load(&store, &track), None);
        save(&store, &track, &grid).unwrap();
        let back = load(&store, &track).unwrap();
        assert_eq!(back.downbeats, grid.downbeats);
        assert!(back
            .beats
            .iter()
            .zip(&grid.beats)
            .all(|(a, b)| (a - b).abs() < 1e-4));
        // A new modification time keeps it; another size does not.
        fs::write(&track, b"9876543210").unwrap();
        assert!(load(&store, &track).is_some());
        fs::write(&track, b"longer than before").unwrap();
        assert_eq!(load(&store, &track), None);
        fs::write(&track, b"0123456789").unwrap();
        remove(&store, &track).unwrap();
        assert_eq!(load(&store, &track), None);
        remove(&store, &track).unwrap();
        assert_eq!(
            from_text("k", "jodel-grid 1\nkey k\nbeats 1\ndownbeats 3\n"),
            None
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
