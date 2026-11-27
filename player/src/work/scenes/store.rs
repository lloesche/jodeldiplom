//! Where a data folder's story scenes are ([`SceneStore`]): each song's,
//! as the story's drawing step leaves them (a folder a song in `scenes/`,
//! named by the song's key: its pictures and their [`super::FILE`]), and
//! the parts derived from each picture, a cache of their own
//! (`scene-parts/`, a file a picture: `<hash>.parts`). The parts are kept
//! by the picture's file's bytes ([`super::picture_hash`]), so a picture is
//! taken apart once whichever song shows it, and one drawn again is taken
//! apart again; parts made otherwise ([`super::VERSION`], in the file) are
//! made again. (A cover's result is kept by its decoded pixels instead,
//! `crate::work::cover::pixel_hash`: a song's cover comes re-encoded, a
//! scene's picture never.)

use std::path::{Path, PathBuf};

use image::ImageFormat;

use super::split::{Cuts, Derived, Head, Split};
use super::{Scenes, VERSION};
use crate::store::cache;
use crate::work::picture::codec::{near_from, near_png, palette_from, palette_text, png};

/// A data folder's story scenes and their parts kept: while the data
/// folder moves, each file read where it is and written where it goes
/// ([`crate::store::mover`]).
#[derive(Clone, Debug)]
pub struct SceneStore {
    scenes: PathBuf,
    parts: PathBuf,
}

const MAGIC: &[u8; 4] = b"JDLS";

impl SceneStore {
    /// The songs' scenes in `scenes`, their pictures' parts kept in `parts`
    /// (each made when first written to).
    pub fn new(scenes: impl Into<PathBuf>, parts: impl Into<PathBuf>) -> SceneStore {
        SceneStore {
            scenes: scenes.into(),
            parts: parts.into(),
        }
    }

    /// The folder the song of `song_key` ([`crate::work::cover::song_key`])
    /// has its scenes in.
    pub fn folder(&self, song_key: &str) -> PathBuf {
        self.scenes.join(song_key)
    }

    /// The scenes of the song at `song`, where the drawing step left them;
    /// none where it left none. Looks at the song's file (its key): not for
    /// the window's thread.
    pub fn scenes_of(&self, song: &Path) -> Option<Scenes> {
        Scenes::read(&self.folder(&crate::work::cover::song_key(song)))
    }

    fn file(&self, hash: &str) -> PathBuf {
        self.parts.join(format!("{hash}.parts"))
    }

    /// The parts kept for the picture of `hash`, where made as this player
    /// makes them.
    pub fn load(&self, hash: &str) -> Option<Derived> {
        let bytes = std::fs::read(crate::store::mover::locate(&self.file(hash))).ok()?;
        let (made, parts) = from_bytes(&bytes)?;
        (made == VERSION && parts.picture == hash).then_some(parts)
    }

    /// Keeps `parts`.
    pub fn keep(&self, parts: &Derived) -> Result<(), String> {
        let path = crate::store::mover::current(&self.file(&parts.picture));
        durable::write_atomic(&path, &to_bytes(VERSION, parts))
            .map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// Parts as kept ([`cache::frame`]d, `JDLS`): a text of what made them,
/// the picture's hash, its split and heads (each number as it reads back
/// exactly) and palette; then its nearness (16 bits), and where in layers
/// its layers' shares and plate, as PNGs.
pub fn to_bytes(made: &str, p: &Derived) -> Vec<u8> {
    let split = match &p.split {
        Split::Warp => "warp".to_owned(),
        Split::Layers { cuts, .. } => match cuts {
            Cuts::Two(c) => format!("two {c}"),
            Cuts::Three(a, b) => format!("three {a} {b}"),
        },
    };
    let mut meta = format!("made {made}\npicture {}\nsplit {split}\n", p.picture);
    for head in &p.heads {
        let [x, y, w, h] = head.face;
        meta += &format!("head {x} {y} {w} {h} {}\n", head.near);
    }
    meta += &format!("palette {}\n", palette_text(&p.palette));
    let mut parts = vec![meta.into_bytes(), near_png(&p.near)];
    if let Split::Layers { alphas, plate, .. } = &p.split {
        parts.push(png(|c| alphas.write_to(c, ImageFormat::Png)));
        parts.push(png(|c| plate.write_to(c, ImageFormat::Png)));
    }
    let parts: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    cache::frame(MAGIC, &cache::SCENE_PARTS, &parts)
}

/// What made parts and the parts, from [`to_bytes`]'s bytes, when of this
/// layout.
pub fn from_bytes(bytes: &[u8]) -> Option<(String, Derived)> {
    let parts = cache::unframe(bytes, MAGIC, &cache::SCENE_PARTS)?;
    let (meta, rest) = parts.split_first()?;
    let meta = std::str::from_utf8(meta).ok()?;
    let field = |name: &str| {
        meta.lines()
            .find_map(|l| l.strip_prefix(name)?.strip_prefix(' '))
    };
    let made = field("made")?.to_owned();
    let picture = field("picture")?.to_owned();
    let numbers = |text: &str| -> Option<Vec<f32>> {
        text.split_whitespace().map(|w| w.parse().ok()).collect()
    };
    let heads = meta
        .lines()
        .filter_map(|l| l.strip_prefix("head "))
        .map(|l| {
            let [x, y, w, h, near] = numbers(l)?[..] else {
                return None;
            };
            Some(Head {
                face: [x, y, w, h],
                near,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    let decode = |b: &[u8]| image::load_from_memory_with_format(b, ImageFormat::Png).ok();
    let (kind, cuts) = field("split")?.split_once(' ').unwrap_or(("warp", ""));
    let cuts = match (kind, &numbers(cuts)?[..]) {
        ("warp", []) => None,
        ("two", &[c]) => Some(Cuts::Two(c)),
        ("three", &[a, b]) => Some(Cuts::Three(a, b)),
        _ => return None,
    };
    let split = match (cuts, rest) {
        (None, [_]) => Split::Warp,
        (Some(cuts), [_, alphas, plate]) => Split::Layers {
            cuts,
            alphas: decode(alphas)?.to_luma_alpha8(),
            plate: decode(plate)?.to_rgb8(),
        },
        _ => return None,
    };
    Some((
        made,
        Derived {
            picture,
            near: near_from(rest.first()?)?,
            heads,
            split,
            palette: palette_from(field("palette")?),
        },
    ))
}
