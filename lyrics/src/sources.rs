//! The text of a track from the track itself: a lyrics tag in the file,
//! or a `.lrc` file beside it.

use std::fs;
use std::path::{Path, PathBuf};

use crate::{lrc, Found, Lyrics, LyricsSource, Source, Track};

/// The tag keys that hold lyrics, in upper case: Vorbis comments and APE
/// (`LYRICS`, `UNSYNCEDLYRICS`), ID3v2 frames (`USLT`, `SYLT`), MP4's
/// `©lyr`, and the names decoders give them.
const LYRICS_KEYS: [&str; 8] = [
    "LYRICS",
    "UNSYNCEDLYRICS",
    "UNSYNCED LYRICS",
    "SYNCEDLYRICS",
    "USLT",
    "SYLT",
    "©LYR",
    "LYR",
];

/// Whether `key` names a lyrics tag: as a whole, or before its first `:`
/// (`USLT:eng`) or after its last (`----:com.apple.iTunes:LYRICS`).
fn is_lyrics_key(key: &str) -> bool {
    let upper = key.trim().to_uppercase();
    let first = upper.split(':').next().unwrap_or("");
    let last = upper.rsplit(':').next().unwrap_or("");
    [upper.as_str(), first, last]
        .iter()
        .any(|k| LYRICS_KEYS.contains(&k.trim()))
}

/// The lyrics among `tags` (key, value), if any: an untimed one before a
/// timed one (whose times are dropped anyway), else the first.
pub fn lyrics_tag(tags: &[(String, String)]) -> Option<&str> {
    let mut found = tags
        .iter()
        .filter(|(k, v)| is_lyrics_key(k) && !v.trim().is_empty())
        .map(|(_, v)| v.as_str());
    let first = found.next()?;
    if !lrc::is_synced(first) {
        return Some(first);
    }
    Some(found.find(|v| !lrc::is_synced(v)).unwrap_or(first))
}

/// The lyrics tag of a track.
#[derive(Clone, Copy, Debug, Default)]
pub struct TagSource;

impl LyricsSource for TagSource {
    fn name(&self) -> &str {
        "tag"
    }

    fn find(&self, track: &Track) -> Result<Found, String> {
        Ok(found(lyrics_tag(&track.tags), Source::Tag))
    }
}

/// The `.lrc` files that may sit beside `path`: `song.lrc`, then
/// `song.flac.lrc`.
pub fn sidecars(path: &Path) -> [PathBuf; 2] {
    let mut beside = path.as_os_str().to_owned();
    beside.push(".lrc");
    [path.with_extension("lrc"), PathBuf::from(beside)]
}

/// Text from a file: UTF-8 (a byte-order mark passed over), else Latin-1.
pub fn read_text(bytes: &[u8]) -> String {
    let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    match std::str::from_utf8(bytes) {
        Ok(text) => text.to_owned(),
        Err(_) => bytes.iter().map(|&b| char::from(b)).collect(),
    }
}

/// A `.lrc` file beside the track.
#[derive(Clone, Copy, Debug, Default)]
pub struct SidecarSource;

impl LyricsSource for SidecarSource {
    fn name(&self) -> &str {
        "sidecar"
    }

    fn find(&self, track: &Track) -> Result<Found, String> {
        for file in sidecars(&track.path) {
            if let Ok(bytes) = fs::read(&file) {
                let found = found(Some(&read_text(&bytes)), Source::Sidecar);
                if found != Found::NotFound {
                    return Ok(found);
                }
            }
        }
        Ok(Found::NotFound)
    }
}

/// `text` as lyrics from `source`, when it has words: the words only, any
/// times in it dropped ([`Lyrics::parse_text`]).
fn found(text: Option<&str>, source: Source) -> Found {
    match text.map(|t| Lyrics::parse_text(t, source)) {
        Some(lyrics) if !lyrics.is_empty() => Found::Lyrics(lyrics),
        _ => Found::NotFound,
    }
}

#[cfg(test)]
#[expect(
    clippy::disallowed_methods,
    reason = "tests: sidecar lyrics files in a temporary folder"
)]
mod tests {
    use super::*;

    fn tags(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|&(k, v)| (k.to_owned(), v.to_owned()))
            .collect()
    }

    #[test]
    fn lyrics_tags_go_by_many_names() {
        for key in [
            "LYRICS",
            "lyrics",
            "UNSYNCEDLYRICS",
            "USLT",
            "USLT:eng",
            "SYLT",
            "©lyr",
            "----:com.apple.iTunes:LYRICS",
        ] {
            assert_eq!(
                lyrics_tag(&tags(&[("TITLE", "x"), (key, "la la")])),
                Some("la la"),
                "{key}"
            );
        }
        assert_eq!(
            lyrics_tag(&tags(&[("TITLE", "lyrics"), ("COMMENT", "x")])),
            None
        );
        assert_eq!(lyrics_tag(&tags(&[("LYRICS", "  ")])), None);
        // An untimed one wins.
        let both = tags(&[("LYRICS", "[00:01.00]la la"), ("UNSYNCEDLYRICS", "la la")]);
        assert_eq!(lyrics_tag(&both), Some("la la"));
    }

    #[test]
    fn the_tag_and_the_sidecar_are_found() {
        let dir = std::env::temp_dir().join(format!("jodel-lyrics-sidecar-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let track = Track {
            path: dir.join("song.flac"),
            tags: tags(&[("LYRICS", "[00:01.00]From the tag")]),
            ..Track::default()
        };
        let Ok(Found::Lyrics(l)) = TagSource.find(&track) else {
            panic!("no tag");
        };
        assert_eq!(l.source, Source::Tag);
        // Its words only: the time it came with is gone.
        assert_eq!(l.lines[0].time, None);
        assert_eq!(l.lines[0].text(), "From the tag");
        assert_eq!(SidecarSource.find(&track), Ok(Found::NotFound));
        // Latin-1 beside it, under the second name.
        fs::write(dir.join("song.flac.lrc"), b"[00:02.00]Caf\xe9\n").unwrap();
        let Ok(Found::Lyrics(l)) = SidecarSource.find(&track) else {
            panic!("no sidecar");
        };
        assert_eq!(l.lines[0].text(), "Café");
        assert_eq!(l.source, Source::Sidecar);
        // The first name wins.
        fs::write(dir.join("song.lrc"), "\u{feff}[00:03.00]First\n").unwrap();
        let Ok(Found::Lyrics(l)) = SidecarSource.find(&track) else {
            panic!("no sidecar");
        };
        assert_eq!(l.lines[0].text(), "First");
        let _ = fs::remove_dir_all(&dir);
    }
}
