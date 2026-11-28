//! Cover art: decoding (on a thread: [`ArtWorker`], the track heard's,
//! or a song's read from its file), fitting into a pixel rect, and a
//! placeholder.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;

use image::{imageops::FilterType, Rgb, RgbImage};
use jodel_playback::Cover;

use crate::work::outbox::Notify;

/// A decoded cover with an id that changes when the cover does.
#[derive(Clone, Debug)]
pub struct Art {
    /// Changes with every new cover (for caches).
    pub id: u64,
    /// The pixels.
    pub image: Arc<RgbImage>,
}

/// Which cover a song brings, as the window's thread can tell cheaply: a
/// hash of all its bytes (where they came from does not count, so the same
/// picture embedded or beside the file is the same cover). A cover is
/// decoded again only where it changes.
pub fn identity(cover: &Cover) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    cover.data.hash(&mut h);
    h.finish()
}

/// Which cover an [`ArtWorker`] was asked for last.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Wanted {
    /// A cover given, by its [`identity`].
    Cover(u64),
    /// The cover of the song at this path, read from its file.
    Song(PathBuf),
}

/// A cover decoded on a thread of its own, the track heard's or a song's
/// read from its file: the one asked for last is the one shown.
pub struct ArtWorker {
    art: Option<Art>,
    /// The cover asked for last (none: none).
    wanted: Option<Wanted>,
    tx: Sender<(u64, Option<RgbImage>)>,
    rx: Receiver<(u64, Option<RgbImage>)>,
    /// The id of the cover asked for last.
    next: u64,
    notify: Notify,
}

impl ArtWorker {
    /// No cover yet; each one decoded told through `notify`.
    pub fn new(notify: Notify) -> ArtWorker {
        let (tx, rx) = mpsc::channel();
        ArtWorker {
            art: None,
            wanted: None,
            tx,
            rx,
            next: 0,
            notify,
        }
    }

    /// The cover shown now is the song's at `path`, read from its file
    /// and decoded, unless it is the one shown already (none: none shown).
    pub fn want_song(&mut self, path: Option<&Path>) {
        let wanted = path.map(|p| Wanted::Song(p.to_owned()));
        let path = path.map(Path::to_owned);
        self.ask(wanted, move || {
            let cover = jodel_playback::source::probe(path?).ok()?.cover?;
            decode(&cover.data)
        });
    }

    /// The cover of the track heard now is `cover`: decoded unless it is
    /// the one shown already (none: none shown).
    pub fn want(&mut self, cover: Option<&Cover>) {
        let wanted = cover.map(|c| Wanted::Cover(identity(c)));
        let data = cover.map(|c| c.data.clone());
        self.ask(wanted, move || decode(&data?));
    }

    /// `wanted` shown now, made by `decoded` on a thread of its own, unless
    /// it is the one asked for already (none: none shown).
    fn ask(
        &mut self,
        wanted: Option<Wanted>,
        decoded: impl FnOnce() -> Option<RgbImage> + Send + 'static,
    ) {
        if wanted == self.wanted && (wanted.is_some() || self.art.is_none()) {
            return;
        }
        self.art = None;
        self.next += 1;
        self.wanted = wanted;
        if self.wanted.is_none() {
            return;
        }
        let id = self.next;
        let (tx, notify) = (self.tx.clone(), Arc::clone(&self.notify));
        thread::spawn(move || {
            if tx.send((id, decoded())).is_ok() {
                notify();
            }
        });
    }

    /// Takes in the covers decoded: whether the one asked for last came.
    pub fn pump(&mut self) -> bool {
        let mut any = false;
        while let Ok((id, image)) = self.rx.try_recv() {
            if id == self.next {
                any = true;
                self.art = image.map(|image| Art {
                    id,
                    image: Arc::new(image),
                });
            }
        }
        any
    }

    /// Sets the cover directly, or none (tests and screenshots).
    pub fn set(&mut self, image: Option<RgbImage>) {
        self.next += 1;
        self.wanted = None;
        self.art = image.map(|image| Art {
            id: self.next,
            image: Arc::new(image),
        });
    }

    /// The cover, once decoded.
    pub fn art(&self) -> Option<&Art> {
        self.art.as_ref()
    }
}

/// Decodes an encoded cover (JPEG or PNG). Very large images are
/// shrunk to 1,024 pixels on their long side.
pub fn decode(data: &[u8]) -> Option<RgbImage> {
    let image = image::load_from_memory(data).ok()?;
    let image = if image.width().max(image.height()) > 1024 {
        image.resize(1024, 1024, FilterType::Triangle)
    } else {
        image
    };
    Some(image.to_rgb8())
}

/// Scales `image` to fit `width` × `height` with its aspect ratio kept,
/// and returns it with the offset that centres it.
pub fn fit(image: &RgbImage, width: u32, height: u32) -> (RgbImage, u32, u32) {
    let (w, h) = (image.width().max(1), image.height().max(1));
    let scale = (f64::from(width) / f64::from(w)).min(f64::from(height) / f64::from(h));
    let (fw, fh) = (
        ((f64::from(w) * scale).round() as u32).clamp(1, width.max(1)),
        ((f64::from(h) * scale).round() as u32).clamp(1, height.max(1)),
    );
    let scaled = image::imageops::resize(image, fw, fh, FilterType::CatmullRom);
    (scaled, (width - fw) / 2, (height - fh) / 2)
}

/// A generated square cover for tests and screenshots, and the Shader
/// tab's picture where a song has none: concentric rings over a diagonal
/// gradient.
pub fn sample_cover(size: u32) -> RgbImage {
    RgbImage::from_fn(size, size, |x, y| {
        let (fx, fy) = (x as f32 / size as f32, y as f32 / size as f32);
        let (dx, dy) = (fx - 0.62, fy - 0.42);
        let d = (dx * dx + dy * dy).sqrt();
        let ring = ((d * 28.0).sin() * 0.5 + 0.5).powf(3.0);
        let r = 30.0 + 200.0 * fx * (1.0 - 0.4 * ring);
        let g = 40.0 + 90.0 * (1.0 - fy) + 100.0 * ring * (1.0 - d);
        let b = 90.0 + 140.0 * fy;
        Rgb([
            r.min(255.0) as u8,
            g.clamp(0.0, 255.0) as u8,
            b.min(255.0) as u8,
        ])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fitting_keeps_the_aspect_and_centres() {
        let image = RgbImage::new(400, 200);
        let (scaled, x, y) = fit(&image, 100, 100);
        assert_eq!((scaled.width(), scaled.height(), x, y), (100, 50, 0, 25));
        let (scaled, x, y) = fit(&sample_cover(64), 300, 120);
        assert_eq!((scaled.width(), scaled.height(), x, y), (120, 120, 90, 0));
    }

    /// Two embedded covers of the same length that differ in one byte
    /// (any byte: all of them count) are two covers.
    #[test]
    fn covers_of_one_length_and_other_bytes_are_told_apart() {
        let embedded = |data: Vec<u8>| Cover {
            data: data.into(),
            mime: "image/jpeg".into(),
            origin: jodel_playback::CoverOrigin::Embedded,
        };
        let a: Vec<u8> = (0..4096u32).map(|i| (i * 7) as u8).collect();
        for at in [1, 63, 2000, 4095] {
            let mut b = a.clone();
            b[at] ^= 1;
            assert_ne!(
                identity(&embedded(a.clone())),
                identity(&embedded(b)),
                "{at}"
            );
        }
        assert_eq!(identity(&embedded(a.clone())), identity(&embedded(a)));
    }

    #[test]
    fn covers_decode_from_png() {
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(sample_cover(32))
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        let back = decode(&bytes).unwrap();
        assert_eq!(back.dimensions(), (32, 32));
        assert!(decode(b"not an image").is_none());
    }
}
