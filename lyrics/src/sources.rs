//! The text of a track from the track itself: a lyrics tag in the file,
//! or a `.lrc` file beside it; and how a text found, untimed or synced or
//! both, becomes lyrics ([`found_from`], LRCLIB's too).

use std::fs;
use std::path::{Path, PathBuf};

use crate::{lrc, scaffold, Found, Lyrics, LyricsSource, Source, Track};

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

/// A track's text both ways, as found: untimed and synced (LRC).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Texts<'a> {
    /// The first untimed one.
    pub plain: Option<&'a str>,
    /// The first synced one.
    pub synced: Option<&'a str>,
}

impl<'a> Texts<'a> {
    /// The first untimed and the first synced of `texts` (the blank ones
    /// passed over).
    pub fn of(texts: impl IntoIterator<Item = &'a str>) -> Texts<'a> {
        let mut out = Texts::default();
        for text in texts.into_iter().filter(|t| !t.trim().is_empty()) {
            let slot = if lrc::is_synced(text) {
                &mut out.synced
            } else {
                &mut out.plain
            };
            slot.get_or_insert(text);
        }
        out
    }
}

/// The lyrics among `tags` (key, value), untimed and synced.
pub fn lyrics_tags(tags: &[(String, String)]) -> Texts<'_> {
    Texts::of(
        tags.iter()
            .filter(|(k, _)| is_lyrics_key(k))
            .map(|(_, v)| v.as_str()),
    )
}

/// The lyrics tag of a track.
#[derive(Clone, Copy, Debug, Default)]
pub struct TagSource;

impl LyricsSource for TagSource {
    fn name(&self) -> &str {
        "tag"
    }

    fn find(&self, track: &Track) -> Result<Found, String> {
        Ok(found_from(lyrics_tags(&track.tags), Source::Tag))
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
                let text = read_text(&bytes);
                let found = found_from(Texts::of([text.as_str()]), Source::Sidecar);
                if found != Found::NotFound {
                    return Ok(found);
                }
            }
        }
        Ok(Found::NotFound)
    }
}

/// `texts` as lyrics from `source`, when they have words
/// ([`Lyrics::parse_text`]: no word times): the untimed text, with the
/// synced one's line times laid on it as the scaffold where it is the same
/// text ([`scaffold::attach`]); else (none, or one without words) the
/// synced one, its line times the scaffold.
pub fn found_from(texts: Texts, source: Source) -> Found {
    let parse = |text: &str| Lyrics::parse_text(text, source.clone());
    let plain = texts.plain.map(parse).filter(|l| !l.is_empty());
    let lyrics = match (plain, texts.synced.map(parse)) {
        (Some(mut plain), Some(synced)) => {
            scaffold::attach(&mut plain, &synced);
            plain
        }
        (Some(lyrics), None) | (None, Some(lyrics)) => lyrics,
        (None, None) => return Found::NotFound,
    };
    if lyrics.is_empty() {
        Found::NotFound
    } else {
        Found::Lyrics(lyrics)
    }
}

#[cfg(test)]
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
                lyrics_tags(&tags(&[("TITLE", "x"), (key, "la la")])).plain,
                Some("la la"),
                "{key}"
            );
        }
        let none = Texts::default();
        assert_eq!(
            lyrics_tags(&tags(&[("TITLE", "lyrics"), ("COMMENT", "x")])),
            none
        );
        assert_eq!(lyrics_tags(&tags(&[("LYRICS", "  ")])), none);
        // Both kinds, each the first of its kind.
        let both = tags(&[
            ("LYRICS", "[00:01.00]la la"),
            ("UNSYNCEDLYRICS", "la la"),
            ("USLT", "other"),
        ]);
        let texts = Texts {
            plain: Some("la la"),
            synced: Some("[00:01.00]la la"),
        };
        assert_eq!(lyrics_tags(&both), texts);
    }

    #[test]
    fn a_plain_tag_keeps_the_synced_tags_lines_as_its_scaffold() {
        let track = Track {
            tags: tags(&[
                ("UNSYNCEDLYRICS", "Look at the sky\nAnd the sea"),
                (
                    "SYNCEDLYRICS",
                    "[00:01.50]Look at the sky\n[00:04.00]And the sea\n",
                ),
            ]),
            ..Track::default()
        };
        let Ok(Found::Lyrics(l)) = TagSource.find(&track) else {
            panic!("no tag");
        };
        assert_eq!(l.lines[0].text(), "Look at the sky");
        let synced: Vec<_> = l.lines.iter().map(|l| l.synced.map(|s| s.start)).collect();
        assert_eq!(synced, [Some(1.5), Some(4.0)]);
        assert!(l.lines.iter().all(|l| l.time.is_none()));
    }

    #[test]
    fn a_plain_text_without_words_gives_way_to_the_synced_one() {
        let texts = Texts {
            plain: Some("[Chorus]\n"),
            synced: Some("[00:01.50]Look at the sky\n[00:04.00]And the sea\n"),
        };
        let Found::Lyrics(l) = found_from(texts, Source::Lrclib) else {
            panic!("no lyrics");
        };
        assert_eq!(l.lines[0].text(), "Look at the sky");
        assert_eq!(l.lines[0].synced.map(|s| s.start), Some(1.5));
        let none = Texts {
            plain: Some("[Chorus]\n"),
            synced: None,
        };
        assert_eq!(found_from(none, Source::Lrclib), Found::NotFound);
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
        durable::write_atomic(&dir.join("song.flac.lrc"), b"[00:02.00]Caf\xe9\n").unwrap();
        let Ok(Found::Lyrics(l)) = SidecarSource.find(&track) else {
            panic!("no sidecar");
        };
        assert_eq!(l.lines[0].text(), "Café");
        assert_eq!(l.source, Source::Sidecar);
        // The first name wins.
        durable::write_atomic(
            &dir.join("song.lrc"),
            "\u{feff}[00:03.00]First\n".as_bytes(),
        )
        .unwrap();
        let Ok(Found::Lyrics(l)) = SidecarSource.find(&track) else {
            panic!("no sidecar");
        };
        assert_eq!(l.lines[0].text(), "First");
        let _ = fs::remove_dir_all(&dir);
    }
}
