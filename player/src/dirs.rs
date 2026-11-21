//! Where the player keeps things: two folders.
//!
//! - **Settings** ([`settings_dir`]): the settings file, the session and
//!   the queues, the beat grids fixed by hand, and the record of a data move
//!   under way. Small, and what one would hate to lose. Where the system
//!   keeps an app's settings, and not to be changed: `%APPDATA%\Jodeldiplom`,
//!   `~/Library/Application Support/Jodeldiplom`, `~/.config/Jodeldiplom`.
//! - **Data** ([`Data`]): what the player makes or fetches and could make
//!   again: the speech models, the beat analyses and waveforms, the stems
//!   kept, the lyrics found and our timings. Large. It is the folder
//!   [`DATA`] inside a parent the settings name (`data_folder`), by default
//!   the system's place for an app's own data ([`default_data_parent`]):
//!   `%LOCALAPPDATA%`, `~/Library/Application Support`, `~/.local/share`.

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

    /// The data folder of the settings in the usual place (for examples
    /// and `--diag`, which read no command line's settings).
    pub fn current() -> Option<Data> {
        let parent = settings_file()
            .and_then(|f| crate::config::Config::load(&f).ok())
            .and_then(|(c, _)| c.data_folder);
        Data::chosen(parent.as_deref())
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

    /// The best scores sung.
    pub fn scores(&self) -> PathBuf {
        self.0.join("scores.txt")
    }
}

/// The record of the last analyses `--diag` shows, in the analyses' folder.
const RECENT: &str = "recent-analyses.txt";

/// Forgets what the player made and can make again (`--clean-cache`): the
/// beat analyses (not the record `--diag` shows), the stems kept, the
/// lyrics. The models stay. What it did, a line a folder.
pub fn forget_cache(data: &Data) -> Vec<String> {
    let mut said = Vec::new();
    for (dir, what) in [
        (
            data.analyses(),
            "beat analyses (an analysis and its waveform each)",
        ),
        (data.stems(), "kept stems"),
        (data.lyrics(), "lyrics"),
    ] {
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
    said
}

/// Forgets the record of the last analyses `--diag` shows (`--clean-diag`).
pub fn forget_diag(data: &Data) -> String {
    let file = data.analyses().join(RECENT);
    match std::fs::remove_file(&file) {
        Ok(()) => format!(
            "Forgot the record of the last analyses in {}",
            file.display()
        ),
        Err(_) => format!("No record of the last analyses in {}", file.display()),
    }
}
