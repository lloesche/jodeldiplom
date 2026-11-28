//! A file's tags and cover: the tags of every metadata block, the
//! embedded pictures, and a cover image in the track's folder.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use symphonia::core::formats::FormatReader;
use symphonia::core::meta::{
    MetadataRevision, RawValue, StandardTag, StandardVisualKey, Tag, Visual,
};

use super::{Cover, CoverOrigin, Tags};

fn value_text(value: &RawValue) -> Option<String> {
    match value {
        RawValue::String(s) => Some(s.as_str().to_owned()),
        RawValue::StringList(list) => Some(list.join("; ")),
        RawValue::UnsignedInt(n) => Some(n.to_string()),
        RawValue::SignedInt(n) => Some(n.to_string()),
        RawValue::Float(f) => Some(f.to_string()),
        RawValue::Boolean(b) => Some(b.to_string()),
        _ => None,
    }
}

/// The name of a user-defined ID3v2 text frame (`TXXX`): its
/// description, which names it (`ORIGINALSAMPLERATE`, ...).
fn user_key(tag: &Tag) -> Option<String> {
    if !tag.raw.key.eq_ignore_ascii_case("TXXX") {
        return None;
    }
    tag.raw
        .sub_fields
        .as_deref()?
        .iter()
        .find(|f| f.field.eq_ignore_ascii_case("DESCRIPTION"))
        .and_then(|f| value_text(&f.value))
        .filter(|d| !d.is_empty())
}

/// A tag key without the prefix of a user-defined ID3v2 frame (`TXXX:`)
/// or an MP4 freeform atom (`----:com.apple.iTunes:`).
pub(super) fn bare_key(key: &str) -> &str {
    let lower = key.to_ascii_lowercase();
    for prefix in ["txxx:", "----:com.apple.itunes:"] {
        if lower.starts_with(prefix) {
            return &key[prefix.len()..];
        }
    }
    key
}

pub(super) fn read_metadata(reader: &mut dyn FormatReader) -> (Tags, Option<Cover>) {
    let mut tags = Tags::default();
    let mut visuals: Vec<Visual> = Vec::new();
    let mut all: Vec<Tag> = Vec::new();
    // Every revision, newest first: tags in front of the container (an
    // ID3v2 tag before MP3 frames), the container's own, trailing ones
    // (ID3v1, APE).
    let mut revisions: Vec<MetadataRevision> = Vec::new();
    {
        let mut metadata = reader.metadata();
        while let Some(old) = metadata.pop() {
            revisions.push(old);
        }
        if let Some(latest) = metadata.current() {
            revisions.push(latest.clone());
        }
    }
    for revision in revisions.iter().rev() {
        all.extend(revision.media.tags.iter().cloned());
        visuals.extend(revision.media.visuals.iter().cloned());
        for track in &revision.per_track {
            all.extend(track.metadata.tags.iter().cloned());
            visuals.extend(track.metadata.visuals.iter().cloned());
        }
    }
    for tag in &all {
        let Some(value) = value_text(&tag.raw.value) else {
            continue;
        };
        let value = value.trim().to_owned();
        let key = user_key(tag).unwrap_or_else(|| tag.raw.key.clone());
        let upper = bare_key(&key).to_ascii_uppercase();
        let slot = match (&tag.std, upper.as_str()) {
            (Some(StandardTag::TrackTitle(_)), _) | (_, "TITLE") => Some(&mut tags.title),
            (Some(StandardTag::Artist(_)), _) | (_, "ARTIST") => Some(&mut tags.artist),
            (Some(StandardTag::Album(_)), _) | (_, "ALBUM") => Some(&mut tags.album),
            (Some(StandardTag::TrackNumber(_)), _) | (_, "TRACKNUMBER") => {
                Some(&mut tags.track_number)
            }
            _ => None,
        };
        if let Some(slot) = slot {
            if !value.is_empty() && slot.is_none() {
                *slot = Some(value.clone());
            }
        }
        tags.all.push((key, value));
    }
    let cover = visuals
        .iter()
        .find(|v| v.usage == Some(StandardVisualKey::FrontCover))
        .or_else(|| visuals.first())
        .filter(|v| !v.data.is_empty())
        .map(|v| Cover {
            data: Arc::from(&v.data[..]),
            mime: v
                .media_type
                .clone()
                .filter(|m| !m.is_empty())
                .unwrap_or_else(|| sniff_mime(&v.data).into()),
            origin: CoverOrigin::Embedded,
        });
    (tags, cover)
}

/// The image types a folder cover may have, and their MIME types.
const COVER_EXTENSIONS: [(&str, &str); 6] = [
    ("jpg", "image/jpeg"),
    ("jpeg", "image/jpeg"),
    ("png", "image/png"),
    ("webp", "image/webp"),
    ("gif", "image/gif"),
    ("bmp", "image/bmp"),
];

/// The folder cover names, in the order they are preferred.
const COVER_NAMES: [&str; 3] = ["cover", "folder", "front"];

/// `cover.*`, else `folder.*`, else `front.*` beside `track` (names and
/// extensions in any case).
pub(super) fn folder_cover(track: &Path) -> Option<Cover> {
    let folder = track.parent()?;
    let folder = if folder.as_os_str().is_empty() {
        Path::new(".")
    } else {
        folder
    };
    let mut found: Vec<(usize, PathBuf, &'static str)> = fs::read_dir(folder)
        .ok()?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let stem = path.file_stem()?.to_str()?.to_ascii_lowercase();
            let ext = path.extension()?.to_str()?.to_ascii_lowercase();
            let rank = COVER_NAMES.iter().position(|n| *n == stem)?;
            let mime = COVER_EXTENSIONS.iter().find(|(e, _)| *e == ext)?.1;
            path.is_file().then_some((rank, path, mime))
        })
        .collect();
    found.sort();
    found.into_iter().find_map(|(_, path, mime)| {
        let data = fs::read(&path).ok().filter(|d| !d.is_empty())?;
        Some(Cover {
            data: Arc::from(data),
            mime: mime.into(),
            origin: CoverOrigin::Folder(path),
        })
    })
}

fn sniff_mime(data: &[u8]) -> &'static str {
    if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        "image/jpeg"
    } else if data.starts_with(b"\x89PNG") {
        "image/png"
    } else if data.starts_with(b"GIF8") {
        "image/gif"
    } else if data.len() > 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WEBP" {
        "image/webp"
    } else if data.starts_with(b"BM") {
        "image/bmp"
    } else {
        "application/octet-stream"
    }
}

#[cfg(test)]
#[expect(
    clippy::disallowed_methods,
    reason = "tests: cover image files beside a track, in a temporary folder"
)]
mod tests {
    use super::*;

    #[test]
    fn sniffs_common_images() {
        assert_eq!(sniff_mime(&[0xFF, 0xD8, 0xFF, 0xE0]), "image/jpeg");
        assert_eq!(sniff_mime(b"\x89PNG\r\n"), "image/png");
        assert_eq!(sniff_mime(b"nothing"), "application/octet-stream");
    }

    #[test]
    fn folder_cover_prefers_cover_then_folder_then_front() {
        let dir = jodel_testkit::TempDir::new("playback-cover");
        let track = dir.join("01.flac");
        assert_eq!(folder_cover(&track), None);
        fs::write(dir.join("Front.PNG"), b"front").unwrap();
        fs::write(dir.join("notes.txt"), b"x").unwrap();
        let cover = folder_cover(&track).unwrap();
        assert_eq!(&*cover.data, b"front");
        assert_eq!(cover.mime, "image/png");
        fs::write(dir.join("folder.jpg"), b"folder").unwrap();
        assert_eq!(&*folder_cover(&track).unwrap().data, b"folder");
        fs::write(dir.join("COVER.jpeg"), b"cover").unwrap();
        let cover = folder_cover(&track).unwrap();
        assert_eq!(&*cover.data, b"cover");
        assert_eq!(cover.mime, "image/jpeg");
        assert_eq!(cover.origin, CoverOrigin::Folder(dir.join("COVER.jpeg")));
    }
}
