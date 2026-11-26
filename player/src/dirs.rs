//! Where the player keeps things: two folders.
//!
//! - **Settings** ([`settings_dir`]): the settings file, the session and
//!   the queues, the beat grids fixed by hand, and the record of a data move
//!   under way. Small, and what one would hate to lose. Where the system
//!   keeps an app's settings, and not to be changed: `%APPDATA%\Jodeldiplom`,
//!   `~/Library/Application Support/Jodeldiplom`, `~/.config/Jodeldiplom`.
//! - **Data** ([`Data`]): what the player makes or fetches and could make
//!   again: the speech models, the beat analyses and waveforms, the stems
//!   kept, the lyrics found and our timings, the covers taken apart; and
//!   the best scores sung. Large. It is the folder [`DATA`] inside a parent
//!   the settings name (`data_folder`), by default the system's place for
//!   an app's own data ([`default_data_parent`]): `%LOCALAPPDATA%`,
//!   `~/Library/Application Support`, `~/.local/share`.
//!
//! A player started with `--config` has the settings folder of that file,
//! and the data folder its settings choose. Both are decided once, at the
//! start ([`Folders`]), and handed to every part that reads or writes
//! there; while the data folder moves, each goes through
//! [`crate::mover`].

use std::path::{Path, PathBuf};

/// The player's name, as its settings folder is named.
pub const NAME: &str = "Jodeldiplom";
/// The data folder's name, inside the parent chosen.
pub const DATA: &str = "Jodeldiplom Data";
/// The settings file's name.
pub const SETTINGS_FILE: &str = "config.toml";

fn var(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn home() -> Option<PathBuf> {
    var("HOME").or_else(|| var("USERPROFILE"))
}

/// The system's folder for apps' settings.
fn settings_base() -> Option<PathBuf> {
    if cfg!(windows) {
        var("APPDATA")
    } else if cfg!(target_os = "macos") {
        home().map(|h| h.join("Library/Application Support"))
    } else {
        var("XDG_CONFIG_HOME")
            .filter(|p| p.is_absolute())
            .or_else(|| home().map(|h| h.join(".config")))
    }
}

/// The settings folder.
pub fn settings_dir() -> Option<PathBuf> {
    settings_base().map(|b| b.join(NAME))
}

/// The settings file, in [`settings_dir`].
pub fn settings_file() -> Option<PathBuf> {
    settings_dir().map(|d| d.join(SETTINGS_FILE))
}

/// Where the data folder is made when the settings name none.
pub fn default_data_parent() -> Option<PathBuf> {
    if cfg!(windows) {
        var("LOCALAPPDATA")
    } else if cfg!(target_os = "macos") {
        home().map(|h| h.join("Library/Application Support"))
    } else {
        var("XDG_DATA_HOME")
            .filter(|p| p.is_absolute())
            .or_else(|| home().map(|h| h.join(".local/share")))
    }
}

/// The data folder, and what is in it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Data(pub PathBuf);

impl Data {
    /// The data folder inside `parent`.
    pub fn in_parent(parent: &Path) -> Data {
        Data(parent.join(DATA))
    }

    /// The data folder the settings choose (`data_folder`, the parent),
    /// else the default one.
    pub fn chosen(parent: Option<&Path>) -> Option<Data> {
        match parent {
            Some(p) => Some(Data::in_parent(p)),
            None => default_data_parent().map(|p| Data::in_parent(&p)),
        }
    }

    /// The speech models.
    pub fn models(&self) -> PathBuf {
        self.0.join("models")
    }

    /// The beat analyses and waveforms.
    pub fn analyses(&self) -> PathBuf {
        self.0.join("analyses")
    }

    /// The stems kept.
    pub fn stems(&self) -> PathBuf {
        self.0.join("stems")
    }

    /// The lyrics found and our timings.
    pub fn lyrics(&self) -> PathBuf {
        self.0.join("lyrics")
    }

    /// The covers taken apart, and which is each song's
    /// ([`crate::cover::CoverStore`]).
    pub fn covers(&self) -> PathBuf {
        self.0.join("covers")
    }

    /// The songs' loudness as the karaoke plays them
    /// ([`crate::loudness`]).
    pub fn loudness(&self) -> PathBuf {
        self.0.join("loudness")
    }

    /// The karaoke game's sounds made, a rate each
    /// ([`crate::sfx::store::BankStore`]).
    pub fn sounds(&self) -> PathBuf {
        self.0.join("sounds")
    }

    /// The original singer's phrases worked out ([`crate::shine_cache`]).
    pub fn shine(&self) -> PathBuf {
        self.0.join("shine")
    }

    /// The best scores sung.
    pub fn scores(&self) -> PathBuf {
        self.0.join("scores.txt")
    }

    /// The folders of what the player makes and can make again (but the
    /// covers', [`crate::cover::CoverStore::dirs_in`]), and what each
    /// holds: what [`forget_cache`] forgets.
    fn caches(&self) -> [(PathBuf, &'static str); 6] {
        [
            (
                self.analyses(),
                "beat analyses (an analysis and its waveform each)",
            ),
            (self.stems(), "kept stems"),
            (self.lyrics(), "lyrics"),
            (self.loudness(), "songs' loudness"),
            (self.sounds(), "game sounds (a rate each)"),
            (self.shine(), "original singer's phrases"),
        ]
    }

    /// The folders a crash may have left a file half written in (never
    /// under a file's name, so never read): the caches', the covers', the
    /// models'.
    pub fn swept(&self) -> Vec<PathBuf> {
        let mut dirs: Vec<PathBuf> = self.caches().into_iter().map(|(d, _)| d).collect();
        dirs.push(self.models());
        dirs.extend(crate::cover::CoverStore::dirs_in(&self.covers()));
        dirs
    }
}

/// Where one player keeps things, decided once at its start from its
/// settings file (the usual one, or `--config`'s) and given to every part
/// that reads or writes there: nothing looks a folder up on its own, so two
/// parts never disagree, and a player with settings of its own touches
/// nothing of the usual one's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Folders {
    /// The settings folder: the settings file's.
    pub settings: PathBuf,
    /// The data folder its settings choose.
    pub data: Data,
}

impl Folders {
    /// The folders of the settings file `file`, whose settings name the
    /// data folder's parent `data_parent` (`None`: the system's place).
    pub fn new(file: &Path, data_parent: Option<&Path>) -> Option<Folders> {
        Some(Folders {
            settings: file.parent()?.to_path_buf(),
            data: Data::chosen(data_parent)?,
        })
    }

    /// The session kept between runs ([`crate::session`]).
    pub fn session(&self) -> PathBuf {
        self.settings.join("session.m3u8")
    }

    /// The queues saved ([`crate::session`]).
    pub fn queues(&self) -> PathBuf {
        self.settings.join("queues")
    }

    /// The beat grids fixed by hand ([`crate::grids`]).
    pub fn grids(&self) -> PathBuf {
        self.settings.join("grids")
    }

    /// The data folders of these settings: the one in use, and the one a
    /// move written down in the settings folder's journal is still taking
    /// it from ([`crate::mover`]).
    fn every_data(&self) -> Vec<Data> {
        let mut all = vec![self.data.clone()];
        for (from, to) in crate::mover::journal(&self.settings) {
            if to == self.data.0 && from != to {
                all.push(Data(from));
            }
        }
        all
    }
}

/// The record of the last analyses `--diag` shows, in the analyses' folder.
const RECENT: &str = "recent-analyses.txt";

/// Forgets what the player made and can make again (`--clean-cache`): the
/// beat analyses (not the record `--diag` shows), the stems kept, the
/// lyrics, the songs' loudness, the game's sounds, the original singer's
/// phrases, the covers taken apart; in the data folder, and in the one a
/// move is still taking it from (left there, it would come over). The
/// models stay. What it did, a line a folder.
pub fn forget_cache(folders: &Folders) -> Vec<String> {
    let mut said = Vec::new();
    for data in folders.every_data() {
        for (dir, what) in data.caches() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                said.push(format!("No {what} in {}", dir.display()));
                continue;
            };
            let mut count = 0;
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file()
                    && path.file_name().is_some_and(|n| n != RECENT)
                    && std::fs::remove_file(&path).is_ok()
                {
                    count += 1;
                }
            }
            said.push(format!(
                "Forgot {count} files of {what} in {}",
                dir.display()
            ));
        }
        let covers = crate::cover::CoverStore::new(data.covers());
        said.push(format!(
            "Forgot {} files of covers taken apart in {}",
            covers.forget(),
            data.covers().display()
        ));
    }
    said
}

/// Forgets the record of the last analyses `--diag` shows (`--clean-diag`),
/// in the data folder and in the one a move is still taking it from. What
/// it did, a line a folder.
pub fn forget_diag(folders: &Folders) -> Vec<String> {
    folders
        .every_data()
        .iter()
        .map(|data| {
            let file = data.analyses().join(RECENT);
            match std::fs::remove_file(&file) {
                Ok(()) => format!(
                    "Forgot the record of the last analyses in {}",
                    file.display()
                ),
                Err(_) => format!("No record of the last analyses in {}", file.display()),
            }
        })
        .collect()
}

#[cfg(test)]
#[expect(
    clippy::disallowed_methods,
    reason = "tests: a data folder's caches as a run left them"
)]
mod tests {
    use super::*;

    /// `--clean-cache` forgets every cache, the original singer's phrases
    /// too, in the data folder and in the one a move is still taking it
    /// from; the models and the record `--diag` shows stay.
    #[test]
    fn clean_cache_forgets_every_cache_in_both_places() {
        let dir = std::env::temp_dir().join(format!("jodel-forget-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let folders = Folders {
            settings: dir.join("settings"),
            data: Data(dir.join("new")),
        };
        let old = Data(dir.join("old"));
        let mut kept = Vec::new();
        for data in [&folders.data, &old] {
            for (cache, _) in data.caches() {
                std::fs::create_dir_all(&cache).unwrap();
                std::fs::write(cache.join("a.txt"), b"made").unwrap();
            }
            std::fs::create_dir_all(data.covers().join("songs")).unwrap();
            std::fs::write(data.covers().join("songs/a.txt"), b"made").unwrap();
            std::fs::create_dir_all(data.models()).unwrap();
            std::fs::write(data.models().join("m.bin"), b"model").unwrap();
            std::fs::write(data.analyses().join(RECENT), b"record").unwrap();
            kept.extend([data.models().join("m.bin"), data.analyses().join(RECENT)]);
        }
        let shine = folders.data.shine().join("a.txt");
        assert!(shine.is_file());
        crate::mover::begin(
            &folders.settings,
            &[(old.0.clone(), folders.data.0.clone())],
        )
        .unwrap();
        let said = forget_cache(&folders);
        assert!(said.iter().any(|l| l.contains("original singer's phrases")));
        assert!(!shine.exists());
        for data in [&folders.data, &old] {
            for (cache, what) in data.caches() {
                assert!(!cache.join("a.txt").exists(), "{what}: {}", cache.display());
            }
            assert!(!data.covers().join("songs/a.txt").exists());
        }
        for file in kept {
            assert!(file.is_file(), "{}", file.display());
        }
        // The crash sweep goes over the phrases too.
        assert!(folders.data.swept().contains(&folders.data.shine()));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
