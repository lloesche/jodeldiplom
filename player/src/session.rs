//! The session kept between runs (the queue, the track heard in it, and
//! where the browser was), and queues saved as playlists.
//!
//! Both are M3U8 files (UTF-8, one path a line), so other players read
//! them too: the session adds its state in comment lines of its own.
//! They live beside `config.toml`: `session.m3u8`, and the saved queues
//! in `queues/`.

use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// The session kept between runs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Session {
    /// The queue.
    pub queue: Vec<PathBuf>,
    /// The track heard in it.
    pub current: Option<usize>,
    /// The folder the browser showed.
    pub browser: Option<PathBuf>,
    /// The row selected there.
    pub selected: Option<PathBuf>,
}

const CURRENT: &str = "#JODELDIPLOM-CURRENT:";
const BROWSER: &str = "#JODELDIPLOM-BROWSER:";
const SELECTED: &str = "#JODELDIPLOM-SELECTED:";

/// Where the session is kept.
pub fn path() -> Option<PathBuf> {
    crate::dirs::settings_dir().map(|d| d.join("session.m3u8"))
}

/// Where queues are saved to.
pub fn queues_dir() -> Option<PathBuf> {
    crate::dirs::settings_dir().map(|d| d.join("queues"))
}

impl Session {
    /// Writes it to `path` (whole or not at all).
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut extra = Vec::new();
        if let Some(i) = self.current {
            extra.push(format!("{CURRENT}{i}"));
        }
        if let Some(dir) = &self.browser {
            extra.push(format!("{BROWSER}{}", dir.display()));
        }
        if let Some(selected) = &self.selected {
            extra.push(format!("{SELECTED}{}", selected.display()));
        }
        write_playlist(path, &self.queue, &extra)
    }

    /// The session in `path`, if there is one.
    pub fn load(path: &Path) -> Option<Session> {
        let text = fs::read(path).ok()?;
        let text = String::from_utf8_lossy(&text);
        let mut session = Session {
            queue: tracks(&text, path.parent()),
            ..Session::default()
        };
        for line in text.lines() {
            if let Some(i) = line.strip_prefix(CURRENT) {
                session.current = i.trim().parse().ok();
            } else if let Some(dir) = line.strip_prefix(BROWSER) {
                session.browser = Some(PathBuf::from(dir.trim_end()));
            } else if let Some(selected) = line.strip_prefix(SELECTED) {
                session.selected = Some(PathBuf::from(selected.trim_end()));
            }
        }
        session.current = session.current.filter(|&i| i < session.queue.len());
        Some(session)
    }
}

/// Writes `tracks` to `path` as an M3U8 playlist, with `extra` comment
/// lines after its header; through a file beside it, so a crash leaves
/// the old one.
pub fn write_playlist(path: &Path, tracks: &[PathBuf], extra: &[String]) -> io::Result<()> {
    let mut text = String::from("#EXTM3U\n");
    for line in extra {
        text.push_str(line);
        text.push('\n');
    }
    for track in tracks {
        text.push_str(&track.display().to_string());
        text.push('\n');
    }
    durable::write_atomic(path, text.as_bytes())
}

/// The tracks of the playlist at `path` (M3U or M3U8).
pub fn read_playlist(path: &Path) -> io::Result<Vec<PathBuf>> {
    let bytes = fs::read(path)?;
    Ok(tracks(&String::from_utf8_lossy(&bytes), path.parent()))
}

/// A playlist's tracks: every line that is not a comment, a relative one
/// from the playlist's folder, a `file://` one as its path.
fn tracks(text: &str, base: Option<&Path>) -> Vec<PathBuf> {
    text.lines()
        .map(|l| l.trim_start_matches('\u{feff}').trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            if let Some(url) = l.strip_prefix("file://") {
                return from_url(url);
            }
            // Relative ones from the playlist's folder (one with a root,
            // `\x` on Windows, stays as it is).
            let path = PathBuf::from(l);
            match base {
                Some(base) if path.is_relative() && !path.has_root() => base.join(path),
                _ => path,
            }
        })
        .collect()
}

/// The path of a `file://` URL's rest (`/C:/x` on Windows is `C:/x`),
/// its `%XX` escapes decoded.
fn from_url(url: &str) -> PathBuf {
    let bytes = url.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex = |b: u8| (b as char).to_digit(16);
        match (
            bytes[i],
            bytes.get(i + 1).copied().and_then(hex),
            bytes.get(i + 2).copied().and_then(hex),
        ) {
            (b'%', Some(h), Some(l)) => {
                out.push((h * 16 + l) as u8);
                i += 3;
            }
            (b, _, _) => {
                out.push(b);
                i += 1;
            }
        }
    }
    let text = String::from_utf8_lossy(&out).into_owned();
    // A drive after the leading slash.
    let b = text.as_bytes();
    if b.len() >= 3 && b[0] == b'/' && b[1].is_ascii_alphabetic() && b[2] == b':' {
        return PathBuf::from(&text[1..]);
    }
    PathBuf::from(text)
}

/// Whether `path` is a playlist (by its extension).
pub fn is_playlist(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("m3u8") || e.eq_ignore_ascii_case("m3u"))
}

/// The name a queue is saved under in `dir`: its first track's folder
/// (and how many other folders it has), and its length, made unique
/// there: `Evolve – 12 tracks.m3u8`, `Evolve and 2 more – 40 tracks 2.m3u8`.
pub fn queue_name(dir: &Path, tracks: &[PathBuf]) -> PathBuf {
    let folder = |p: &PathBuf| p.parent().map(Path::to_path_buf);
    let first = tracks
        .first()
        .and_then(folder)
        .and_then(|f| f.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "Queue".into());
    let folders: HashSet<PathBuf> = tracks.iter().filter_map(folder).collect();
    let more = match folders.len() {
        0 | 1 => String::new(),
        n => format!(" and {} more", n - 1),
    };
    let count = match tracks.len() {
        1 => "1 track".to_owned(),
        n => format!("{n} tracks"),
    };
    // Characters a file name cannot have on some system.
    let base: String = format!("{first}{more} – {count}")
        .chars()
        .map(|c| {
            if "<>:\"/\\|?*".contains(c) || c.is_control() {
                '_'
            } else {
                c
            }
        })
        .collect();
    (1..)
        .map(|k| match k {
            1 => dir.join(format!("{base}.m3u8")),
            k => dir.join(format!("{base} {k}.m3u8")),
        })
        .find(|p| !p.exists())
        .expect("a free name")
}

#[cfg(test)]
#[expect(
    clippy::disallowed_methods,
    reason = "tests: session and playlist files as written by hand"
)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("jodel-session-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_session_comes_back_as_it_was_saved() {
        let dir = temp("round");
        let path = dir.join("session.m3u8");
        let session = Session {
            queue: vec!["/m/a/01.flac".into(), "/m/a/02 ä.flac".into()],
            current: Some(1),
            browser: Some("/m/a".into()),
            selected: Some("/m/a/02 ä.flac".into()),
        };
        session.save(&path).unwrap();
        assert_eq!(Session::load(&path), Some(session.clone()));
        // Another player sees the tracks only.
        assert_eq!(read_playlist(&path).unwrap(), session.queue);
        // A current track past the queue is none.
        fs::write(&path, "#EXTM3U\n#JODELDIPLOM-CURRENT:5\n/m/x.flac\n").unwrap();
        assert_eq!(Session::load(&path).unwrap().current, None);
        assert_eq!(Session::load(&dir.join("none.m3u8")), None);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn playlists_read_relative_and_url_lines() {
        let dir = temp("read");
        let path = dir.join("list.m3u");
        fs::write(
            &path,
            "\u{feff}#EXTM3U\n#EXTINF:123,Some song\nsub/01.flac\n\nfile:///abs/02%20two.flac\nfile:///C:/m/03.flac\n",
        )
        .unwrap();
        let got = read_playlist(&path).unwrap();
        assert_eq!(
            got,
            [
                dir.join("sub/01.flac"),
                PathBuf::from("/abs/02 two.flac"),
                PathBuf::from("C:/m/03.flac")
            ]
        );
        assert!(is_playlist(&path) && is_playlist(Path::new("x.M3U8")));
        assert!(!is_playlist(Path::new("x.flac")));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_queue_is_named_by_its_folders_and_length() {
        let dir = temp("name");
        let one = vec![
            PathBuf::from("/m/Evolve/01.flac"),
            PathBuf::from("/m/Evolve/02.flac"),
        ];
        assert_eq!(queue_name(&dir, &one), dir.join("Evolve – 2 tracks.m3u8"));
        fs::write(dir.join("Evolve – 2 tracks.m3u8"), "").unwrap();
        assert_eq!(queue_name(&dir, &one), dir.join("Evolve – 2 tracks 2.m3u8"));
        let mixed = vec![
            PathBuf::from("/m/Evolve/01.flac"),
            PathBuf::from("/m/Other/01.flac"),
            PathBuf::from("/m/Third: Live/01.flac"),
        ];
        assert_eq!(
            queue_name(&dir, &mixed),
            dir.join("Evolve and 2 more – 3 tracks.m3u8")
        );
        assert_eq!(
            queue_name(&dir, &mixed[2..]),
            dir.join("Third_ Live – 1 track.m3u8")
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
