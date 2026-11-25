//! One player at a time. Two players would play over each other on the
//! audio device and fight over the settings file, so a
//! player holds a lock on a file in the settings folder for as long as it
//! runs (released by the system when it ends, however it ends), and a
//! second one started meanwhile says so and goes.
//!
//! The lock file holds nothing: it is only opened, never written.

use std::fs::{File, OpenOptions, TryLockError};
use std::path::Path;

/// The lock file, in the settings folder.
pub const LOCK_FILE: &str = "jodeldiplom.lock";

/// The lock, held while it lives.
#[derive(Debug)]
pub struct Only {
    _file: File,
}

/// Whether this player may run.
#[derive(Debug)]
pub enum Start {
    /// It holds the lock (none when the lock could not be taken for
    /// another reason than a player holding it: then it runs unguarded).
    Alone(Option<Only>),
    /// Another player holds it.
    Running,
}

/// Takes the lock in `dir` (made if missing).
#[expect(
    clippy::disallowed_methods,
    reason = "the one-player lock file: opened to be locked, never written"
)]
pub fn take(dir: &Path) -> Start {
    let file = durable::create_dir_all(dir).and_then(|()| {
        OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(dir.join(LOCK_FILE))
    });
    let Ok(file) = file else {
        return Start::Alone(None);
    };
    match file.try_lock() {
        Ok(()) => Start::Alone(Some(Only { _file: file })),
        Err(TryLockError::WouldBlock) => Start::Running,
        Err(TryLockError::Error(_)) => Start::Alone(None),
    }
}

/// Tells that a player is running already: on the console it was started
/// from, if any, and on Windows in a message box too (started from
/// Explorer it has no console to print to).
pub fn say_running(bin: &str) {
    let text = "Jodeldiplom is already running.";
    eprintln!("{bin}: {text}");
    #[cfg(windows)]
    {
        use windows::core::HSTRING;
        use windows::Win32::UI::WindowsAndMessaging::{MessageBoxW, MB_ICONINFORMATION, MB_OK};
        // SAFETY: plain strings and no owner window; the call only shows a
        // box and returns when it is closed.
        unsafe {
            MessageBoxW(
                None,
                &HSTRING::from(text),
                &HSTRING::from("Jodeldiplom"),
                MB_OK | MB_ICONINFORMATION,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_second_player_finds_the_first_running_and_runs_once_it_is_gone() {
        let dir = std::env::temp_dir().join(format!("jodeldiplom-single-{}", std::process::id()));
        let first = take(&dir);
        assert!(matches!(first, Start::Alone(Some(_))), "{first:?}");
        assert!(matches!(take(&dir), Start::Running));
        drop(first);
        assert!(matches!(take(&dir), Start::Alone(Some(_))));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
