//! The covers taken apart, on disk in two levels (`covers/` in the data
//! folder):
//!
//! - **The cover's result**, once ([`crate::cache::COVER_LAYERS`],
//!   `<pixels>.layers`): its parts as PNGs and a thumbnail of the cover,
//!   keyed by the hash of its decoded pixels, with what made it (the models
//!   and way of taking apart, [`super::VERSION`], compared when it is
//!   read: one made otherwise is made again).
//! - **A song's entry** ([`crate::cache::COVER_SONG`], `songs/<song>.txt`):
//!   which cover result is the song's (or that it has no cover). Whatever
//!   about the cover comes to depend on the song itself goes here later.
//! - **An album's covers** ([`crate::cache::COVER_ALBUM`],
//!   `albums/<album>.txt`): each cover result made for it with its
//!   perceptual hash, so the same art encoded again (another quality or
//!   size, a `folder.jpg` beside embedded art) is taken from there, once its
//!   thumbnail agrees with the cover's pixel for pixel (a templated single
//!   or another edition's cover that only looks alike is not).
//!
//! A song whose entry leads to a current result is ready at once. The
//! covers' step works one song at a time, so a second song of an album
//! finds what the first's made.

use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use image::imageops::FilterType;
use image::{GrayImage, ImageBuffer, ImageFormat, Luma, RgbImage};

use super::hash::{perceptual, pixel_hash, Perceptual};
use super::{Apart, CoverLayers, Layers, NearMap, Swatch, VERSION};
use crate::cache;

/// What takes a cover apart (the models on the GPU, or a stand-in).
pub trait Splitter: Send + Sync {
    /// `image` taken apart.
    fn split(&self, image: &RgbImage) -> Result<CoverLayers, String>;
    /// Nothing more to take apart for a while: what it holds may go.
    fn rest(&self) {}
}

/// A song's cover, as its preparation finds it.
#[derive(Clone, Debug)]
pub struct SongCover {
    /// The song's key ([`super::song_key`]).
    pub song: String,
    /// The keys its album is known by ([`super::album_keys`]).
    pub albums: Vec<String>,
    /// Its decoded cover; `None` for a song without one.
    pub image: Option<RgbImage>,
}

/// A cover's result as kept: its layers, a thumbnail of the cover (to tell
/// a cover encoded again from one that only looks alike), and what made it.
#[derive(Clone, Debug, PartialEq)]
pub struct Kept {
    /// What made it ([`super::VERSION`] when this player did).
    pub made: String,
    /// The cover at [`THUMB`] × [`THUMB`].
    pub thumb: RgbImage,
    /// The cover taken apart.
    pub layers: CoverLayers,
}

/// The side of a cover's thumbnail.
pub const THUMB: u32 = 64;

/// The thumbnail of `image`.
pub fn thumb(image: &RgbImage) -> RgbImage {
    image::imageops::resize(image, THUMB, THUMB, FilterType::Triangle)
}

/// Whether two thumbnails are of one picture: their pixels' mean
/// difference at most 4 levels (the same art encoded again differs by
/// about 1; a cover that only looks alike, by far more).
pub fn same_picture(a: &RgbImage, b: &RgbImage) -> bool {
    let n = a.as_raw().len().max(1) as f64;
    a.dimensions() == b.dimensions()
        && a.as_raw()
            .iter()
            .zip(b.as_raw())
            .map(|(&x, &y)| f64::from(x.abs_diff(y)))
            .sum::<f64>()
            / n
            <= 4.0
}

/// The covers taken apart, in a folder: while the data folder moves, each
/// file read where it is and written where it goes ([`crate::mover`]).
pub struct CoverStore {
    dir: PathBuf,
}

const MAGIC: &[u8; 4] = b"JDLC";

impl CoverStore {
    /// The store in `dir` (made when first written to).
    pub fn new(dir: impl Into<PathBuf>) -> CoverStore {
        CoverStore { dir: dir.into() }
    }

    /// The folders a store in `dir` keeps its files in (for the sweep of
    /// unfinished writes and for forgetting).
    pub fn dirs_in(dir: &Path) -> [PathBuf; 3] {
        [dir.to_path_buf(), dir.join("songs"), dir.join("albums")]
    }

    /// Its folders ([`CoverStore::dirs_in`]).
    pub fn dirs(&self) -> [PathBuf; 3] {
        CoverStore::dirs_in(&self.dir)
    }

    /// Forgets every cover taken apart and every song's and album's entry
    /// (they are made again): the files it removed.
    pub fn forget(&self) -> usize {
        let mut removed = 0;
        for dir in self.dirs() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && std::fs::remove_file(&path).is_ok() {
                    removed += 1;
                }
            }
        }
        removed
    }

    fn result_file(&self, hash: &str) -> PathBuf {
        self.dir.join(format!("{hash}.layers"))
    }

    fn song_file(&self, song: &str) -> PathBuf {
        self.dir.join("songs").join(format!("{song}.txt"))
    }

    fn album_file(&self, album: &str) -> PathBuf {
        self.dir.join("albums").join(format!("{album}.txt"))
    }

    /// The song's entry: `Some(Some(pixels))` for its cover's result,
    /// `Some(None)` for a song without a cover, `None` without an entry.
    pub fn linked(&self, song: &str) -> Option<Option<String>> {
        let text = std::fs::read_to_string(crate::mover::locate(&self.song_file(song))).ok()?;
        let body = cache::COVER_SONG.body(&text)?;
        let line = body.lines().find_map(|l| l.strip_prefix("cover "))?;
        Some((line != "none").then(|| line.to_owned()))
    }

    /// Whether the song is ready: its entry leads to a current result, or
    /// says it has no cover (without decoding anything).
    pub fn ready(&self, song: &str) -> bool {
        match self.linked(song) {
            Some(None) => true,
            Some(Some(hash)) => self.current(&hash),
            None => false,
        }
    }

    /// Whether the cover's result is there, of this layout, and made as
    /// this player makes them (its header only).
    fn current(&self, hash: &str) -> bool {
        let Ok(mut f) = std::fs::File::open(crate::mover::locate(&self.result_file(hash))) else {
            return false;
        };
        let mut head = vec![0u8; 9];
        if f.read_exact(&mut head).is_err() {
            return false;
        }
        let Some(len) = meta_len(&head) else {
            return false;
        };
        head.resize(9 + len, 0);
        f.read_exact(&mut head[9..]).is_ok() && made(&head).as_deref() == Some(VERSION)
    }

    /// The cover result `hash`, when there and made as this player makes
    /// them.
    pub fn load(&self, hash: &str) -> Option<Kept> {
        let kept = from_bytes(&std::fs::read(crate::mover::locate(&self.result_file(hash))).ok()?)?;
        (kept.made == VERSION).then_some(kept)
    }

    /// The song's cover's layers: from the song's entry where it leads to a
    /// current result; else found (the same pixels, or the same picture in
    /// the album) or made by `splitter`, kept, and the song's entry written.
    /// `None` for a song without a cover (its entry says so).
    pub fn layers(
        &self,
        song: &SongCover,
        splitter: &dyn Splitter,
    ) -> Result<Option<Layers>, String> {
        if let Some(Some(hash)) = self.linked(&song.song) {
            if let Some(k) = self.load(&hash) {
                return Ok(Some(Arc::new(Apart::new(hash, k.layers))));
            }
        }
        let Some(image) = &song.image else {
            self.link(&song.song, None)?;
            return Ok(None);
        };
        let hash = pixel_hash(image);
        // The same pixels, taken apart before.
        if let Some(k) = self.load(&hash) {
            self.link(&song.song, Some(&hash))?;
            return Ok(Some(Arc::new(Apart::new(hash, k.layers))));
        }
        // The same picture in the album: alike to its perceptual hash, and
        // pixel for pixel at a thumbnail's size.
        let seen = perceptual(image);
        let small = thumb(image);
        for album in &song.albums {
            for (other, p) in self.album(album) {
                if !p.same(&seen) {
                    continue;
                }
                if let Some(k) = self.load(&other).filter(|k| same_picture(&k.thumb, &small)) {
                    self.link(&song.song, Some(&other))?;
                    return Ok(Some(Arc::new(Apart::new(other, k.layers))));
                }
            }
        }
        let layers = splitter.split(image)?;
        let kept = Kept {
            made: VERSION.to_owned(),
            thumb: small,
            layers,
        };
        self.write(&self.result_file(&hash), &to_bytes(&kept))?;
        for album in &song.albums {
            self.add_to_album(album, &hash, &seen)?;
        }
        self.link(&song.song, Some(&hash))?;
        Ok(Some(Arc::new(Apart::new(hash, kept.layers))))
    }

    fn write(&self, path: &Path, bytes: &[u8]) -> Result<(), String> {
        let path = &crate::mover::current(path);
        durable::write_atomic(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Writes the song's entry: its cover's result, or none.
    pub fn link(&self, song: &str, hash: Option<&str>) -> Result<(), String> {
        let body = format!("cover {}\n", hash.unwrap_or("none"));
        self.write(
            &self.song_file(song),
            cache::COVER_SONG.wrap(&body).as_bytes(),
        )
    }

    /// The album's covers: each result's pixels and perceptual hash.
    pub fn album(&self, album: &str) -> Vec<(String, Perceptual)> {
        let Ok(text) = std::fs::read_to_string(crate::mover::locate(&self.album_file(album)))
        else {
            return Vec::new();
        };
        let Some(body) = cache::COVER_ALBUM.body(&text) else {
            return Vec::new();
        };
        body.lines()
            .filter_map(|l| {
                let words: Vec<&str> = l.split_whitespace().collect();
                let (hash, rest) = words.split_first()?;
                Some(((*hash).to_owned(), Perceptual::parse(rest)?))
            })
            .collect()
    }

    /// Adds the cover result `hash` (its perceptual hash `seen`) to the
    /// album's covers.
    pub fn add_to_album(&self, album: &str, hash: &str, seen: &Perceptual) -> Result<(), String> {
        let mut lines: Vec<String> = self
            .album(album)
            .into_iter()
            .filter(|(h, _)| h != hash)
            .map(|(h, p)| format!("{h} {}", p.to_text()))
            .collect();
        lines.push(format!("{hash} {}", seen.to_text()));
        let body = lines.join("\n") + "\n";
        self.write(
            &self.album_file(album),
            cache::COVER_ALBUM.wrap(&body).as_bytes(),
        )
    }
}

/// The length of a result's header text, from its first 9 bytes, when
/// they are of this layout.
fn meta_len(head: &[u8]) -> Option<usize> {
    (head.len() >= 9 && &head[..4] == MAGIC && u32::from(head[4]) == cache::COVER_LAYERS.version)
        .then(|| u32::from_le_bytes([head[5], head[6], head[7], head[8]]) as usize)
}

/// What made a result, from its header (its first bytes, the header
/// text whole).
fn made(bytes: &[u8]) -> Option<String> {
    let len = meta_len(bytes)?;
    let text = std::str::from_utf8(bytes.get(9..9 + len)?).ok()?;
    text.lines()
        .find_map(|l| l.strip_prefix("made "))
        .map(str::to_owned)
}

fn png(encode: impl FnOnce(&mut Cursor<&mut Vec<u8>>) -> image::ImageResult<()>) -> Vec<u8> {
    let mut out = Vec::new();
    encode(&mut Cursor::new(&mut out)).expect("PNG into memory");
    out
}

fn near_png(n: &NearMap) -> Vec<u8> {
    let img: ImageBuffer<Luma<u16>, Vec<u16>> =
        ImageBuffer::from_raw(n.width, n.height, n.near.clone()).expect("its size");
    png(|c| img.write_to(c, ImageFormat::Png))
}

fn near_from(bytes: &[u8]) -> Option<NearMap> {
    let img = image::load_from_memory_with_format(bytes, ImageFormat::Png)
        .ok()?
        .to_luma16();
    Some(NearMap {
        width: img.width(),
        height: img.height(),
        near: img.into_raw(),
    })
}

/// A result as kept: `JDLC`, the version byte, the length of a text of
/// what made it and the palette, that text, then the parts as PNGs each
/// after its length (the plate, the subject or nothing, the depth, the
/// plate's depth, the rim, the thumbnail).
pub fn to_bytes(k: &Kept) -> Vec<u8> {
    let l = &k.layers;
    let palette: Vec<String> = l
        .palette
        .iter()
        .map(|s| format!("{},{},{},{:.4}", s.rgb[0], s.rgb[1], s.rgb[2], s.share))
        .collect();
    let meta = format!("made {}\npalette {}\n", k.made, palette.join(" "));
    let mut out = MAGIC.to_vec();
    out.push(cache::COVER_LAYERS.version as u8);
    out.extend((meta.len() as u32).to_le_bytes());
    out.extend(meta.as_bytes());
    let parts = [
        png(|c| l.plate.write_to(c, ImageFormat::Png)),
        l.subject
            .as_ref()
            .map_or_else(Vec::new, |s| png(|c| s.write_to(c, ImageFormat::Png))),
        near_png(&l.depth),
        near_png(&l.plate_depth),
        png(|c| l.outline.write_to(c, ImageFormat::Png)),
        png(|c| k.thumb.write_to(c, ImageFormat::Png)),
    ];
    for p in parts {
        out.extend((p.len() as u32).to_le_bytes());
        out.extend(p);
    }
    out
}

/// A result from [`to_bytes`]'s bytes, when of this layout (whatever made
/// it).
pub fn from_bytes(bytes: &[u8]) -> Option<Kept> {
    let made = made(bytes)?;
    let mut at = 5;
    let mut take = || -> Option<&[u8]> {
        let len = u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?) as usize;
        let part = bytes.get(at + 4..at + 4 + len)?;
        at += 4 + len;
        Some(part)
    };
    let meta = std::str::from_utf8(take()?).ok()?.to_owned();
    let decode = |b: &[u8]| image::load_from_memory_with_format(b, ImageFormat::Png).ok();
    let plate: RgbImage = decode(take()?)?.to_rgb8();
    let subject_bytes = take()?;
    let subject = if subject_bytes.is_empty() {
        None
    } else {
        Some(decode(subject_bytes)?.to_rgba8())
    };
    let depth = near_from(take()?)?;
    let plate_depth = near_from(take()?)?;
    let outline: GrayImage = decode(take()?)?.to_luma8();
    let thumb = decode(take()?)?.to_rgb8();
    let palette = meta
        .lines()
        .find_map(|l| l.strip_prefix("palette"))?
        .split_whitespace()
        .filter_map(|s| {
            let v: Vec<&str> = s.split(',').collect();
            let [r, g, b, share] = v[..] else {
                return None;
            };
            Some(Swatch {
                rgb: [r.parse().ok()?, g.parse().ok()?, b.parse().ok()?],
                share: share.parse().ok()?,
            })
        })
        .collect();
    Some(Kept {
        made,
        thumb,
        layers: CoverLayers {
            plate,
            subject,
            depth,
            plate_depth,
            outline,
            palette,
        },
    })
}

#[cfg(test)]
#[expect(
    clippy::disallowed_methods,
    reason = "tests: a kept result, written as an older player left it"
)]
pub(crate) mod tests;
