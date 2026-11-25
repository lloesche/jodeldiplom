//! The singer's delays kept: `delays.txt` beside the settings file
//! ([`crate::cache::DELAYS`]), an output device and a microphone each (see
//! [`crate::core::lag`]). The singer's own record, not a cache: every
//! version of the file is read.
//!
//! After the header, a line a pair: the delay measured in ms (`-`: none,
//! only a nudge), the nudge in ms (`-`: none; version 1 wrote 0 for none,
//! before a nudge of 0 meant anything), the output device's id and the
//! microphone's, tab apart.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::cache::DELAYS;
use crate::core::lag::{Delay, Delays, NUDGE_LEAST, NUDGE_MOST};

/// The file's name, in the settings folder.
pub const FILE: &str = "delays.txt";

/// `kept` as the file holds them, in order (by output, then microphone).
pub fn to_text(kept: &Delays) -> String {
    let mut keys: Vec<_> = kept.keys().collect();
    keys.sort();
    let mut body = String::new();
    for key in keys {
        let d = &kept[key];
        let measured = d
            .measured_ms
            .map_or_else(|| "-".to_owned(), |m| m.to_string());
        let nudge = d.nudge_ms.map_or_else(|| "-".to_owned(), |n| n.to_string());
        let _ = writeln!(body, "{measured}\t{nudge}\t{}\t{}", key.0, key.1);
    }
    DELAYS.wrap(&body)
}

/// The delays in a file's `text` (every version; lines not read skipped,
/// a nudge out of its range brought into it).
pub fn from_text(text: &str) -> Delays {
    let (version, body) = DELAYS.open(text).unwrap_or((0, ""));
    let mut out = Delays::new();
    for line in body.lines() {
        let mut parts = line.splitn(4, '\t');
        let nudge = |n: &str| match (n, version) {
            ("-", _) => Some(None),
            ("0", 1) => Some(None),
            (n, _) => n.parse::<i32>().ok().map(Some),
        };
        let (Some(measured), Some(nudge), Some(output), Some(mic)) = (
            parts.next(),
            parts.next().and_then(nudge),
            parts.next(),
            parts.next(),
        ) else {
            continue;
        };
        let measured_ms = match measured {
            "-" => None,
            m => match m.parse() {
                Ok(m) => Some(m),
                Err(_) => continue,
            },
        };
        out.insert(
            (output.to_owned(), mic.to_owned()),
            Delay {
                measured_ms,
                nudge_ms: nudge.map(|n| n.clamp(NUDGE_LEAST, NUDGE_MOST)),
            },
        );
    }
    out
}

/// The delays in the file at `path` (none when there is none).
pub fn load(path: &Path) -> Delays {
    std::fs::read_to_string(path)
        .map(|t| from_text(&t))
        .unwrap_or_default()
}

/// Writes `kept` to `path`, durably.
pub fn save(path: &Path, kept: &Delays) -> std::io::Result<()> {
    durable::write_atomic(path, to_text(kept).as_bytes())
}

/// Where the delays are kept for the settings file `config_file`: beside
/// it (none without one: a run with no settings file keeps them for
/// itself).
pub fn beside(config_file: Option<&Path>) -> Option<PathBuf> {
    config_file.and_then(Path::parent).map(|dir| dir.join(FILE))
}

/// The delays kept beside the settings file `config_file`.
pub fn load_beside(config_file: Option<&Path>) -> Delays {
    beside(config_file).map(|f| load(&f)).unwrap_or_default()
}

/// Saves `kept` beside the settings file `config_file` (nowhere without
/// one); what to say when that failed.
pub fn save_beside(config_file: Option<&Path>, kept: &Delays) -> Option<String> {
    let file = beside(config_file)?;
    save(&file, kept)
        .err()
        .map(|e| format!("The delays were not saved: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_delays_go_to_a_file_and_back_a_pair_each() {
        let dir = std::env::temp_dir().join(format!("jodel-delays-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let config = dir.join("config.toml");
        let mut kept = Delays::new();
        let headset = ("{0.0.0}.{abc}".to_owned(), "{0.0.1}.{def}".to_owned());
        let other_mic = ("{0.0.0}.{abc}".to_owned(), "USB mic 2".to_owned());
        kept.insert(
            headset.clone(),
            Delay {
                measured_ms: Some(152),
                nudge_ms: Some(-20),
            },
        );
        kept.insert(
            other_mic.clone(),
            Delay {
                measured_ms: None,
                nudge_ms: Some(30),
            },
        );
        assert_eq!(save_beside(Some(&config), &kept), None);
        assert!(dir.join(FILE).is_file());
        let back = load_beside(Some(&config));
        assert_eq!(back, kept);
        assert_eq!(back[&headset].measured_ms, Some(152));
        // A line not read is skipped, the rest kept; a nudge out of range
        // is brought into it; no file, none; no settings file, nowhere.
        let text = std::fs::read_to_string(dir.join(FILE)).unwrap();
        assert_eq!(from_text(&text.replace("152\t", "x\t")).len(), 1);
        let far = from_text(&text.replace("\t30\t", "\t900\t"));
        assert_eq!(far[&other_mic].nudge_ms, Some(NUDGE_MOST));
        assert!(load(&dir.join("none.txt")).is_empty());
        // Version 1 wrote 0 for no nudge (before a nudge of 0 meant
        // anything): read as none; version 2 tells them apart.
        let v1 = "jodeldiplom-cache delays 1 by x\n120\t0\tout\tmic\n140\t-30\tout\tusb\n";
        let old = from_text(v1);
        let pair = |m: &str| ("out".to_owned(), m.to_owned());
        assert_eq!(old[&pair("mic")].nudge_ms, None);
        assert_eq!(old[&pair("usb")].nudge_ms, Some(-30));
        let v2 = "jodeldiplom-cache delays 2 by x\n120\t0\tout\tmic\n120\t-\tout\tusb\n";
        let new = from_text(v2);
        assert_eq!(new[&pair("mic")].nudge_ms, Some(0));
        assert_eq!(new[&pair("usb")].nudge_ms, None);
        assert!(load_beside(None).is_empty() && save_beside(None, &kept).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
