//! How a cover is known again: by its pixels (the same art embedded in
//! every track of an album, or beside them as `folder.jpg`), by a
//! perceptual hash among the covers of one album (the same art encoded
//! again, at another quality or size), and the keys an album is known by.

use std::path::Path;

use image::imageops::FilterType;
use image::RgbImage;

use crate::download::Sha256;

/// The side covers are brought to before their pixels are hashed, so the
/// same picture decoded at two sizes from one file is one cover.
const SIDE: u32 = 256;

/// The cover's key: a hash of its decoded pixels at 256 × 256.
pub fn pixel_hash(image: &RgbImage) -> String {
    let small = image::imageops::resize(image, SIDE, SIDE, FilterType::Triangle);
    let mut sha = Sha256::default();
    sha.update(small.as_raw());
    sha.hex()[..20].to_owned()
}

/// A cover's perceptual hash (64 bits of its brightness's low
/// frequencies, above or below their median) and its mean colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Perceptual {
    /// The 64 bits.
    pub bits: u64,
    /// The mean colour.
    pub mean: [u8; 3],
}

impl Perceptual {
    /// Whether two covers are the same picture: at most 8 of the 64 bits
    /// apart and their mean colours within 16 of each other.
    pub fn same(&self, other: &Perceptual) -> bool {
        (self.bits ^ other.bits).count_ones() <= 8
            && self
                .mean
                .iter()
                .zip(other.mean)
                .all(|(&a, b)| a.abs_diff(b) <= 16)
    }

    /// As kept: the bits in hex, then the mean colour.
    pub fn to_text(&self) -> String {
        format!(
            "{:016x} {} {} {}",
            self.bits, self.mean[0], self.mean[1], self.mean[2]
        )
    }

    /// From [`Perceptual::to_text`]'s words.
    pub fn parse(words: &[&str]) -> Option<Perceptual> {
        let [bits, r, g, b] = words else {
            return None;
        };
        Some(Perceptual {
            bits: u64::from_str_radix(bits, 16).ok()?,
            mean: [r.parse().ok()?, g.parse().ok()?, b.parse().ok()?],
        })
    }
}

/// The perceptual hash of `image`: its brightness at 32 × 32, the 2-D DCT,
/// the 8 × 8 lowest frequencies but the constant one against their median.
pub fn perceptual(image: &RgbImage) -> Perceptual {
    let small = image::imageops::resize(image, 32, 32, FilterType::Triangle);
    let mut mean = [0u64; 3];
    let luma: Vec<f64> = small
        .pixels()
        .map(|p| {
            for c in 0..3 {
                mean[c] += u64::from(p[c]);
            }
            0.299 * f64::from(p[0]) + 0.587 * f64::from(p[1]) + 0.114 * f64::from(p[2])
        })
        .collect();
    let n = 32.0;
    let dct = |u: usize, v: usize| -> f64 {
        let mut sum = 0.0;
        for y in 0..32 {
            for x in 0..32 {
                sum += luma[y * 32 + x]
                    * (std::f64::consts::PI * (2.0 * x as f64 + 1.0) * u as f64 / (2.0 * n)).cos()
                    * (std::f64::consts::PI * (2.0 * y as f64 + 1.0) * v as f64 / (2.0 * n)).cos();
            }
        }
        sum
    };
    let coefficients: Vec<f64> = (0..64)
        .map(|i| if i == 0 { 0.0 } else { dct(i % 8, i / 8) })
        .collect();
    let mut sorted = coefficients[1..].to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    let bits = coefficients
        .iter()
        .enumerate()
        .filter(|&(i, &c)| i > 0 && c > median)
        .fold(0u64, |b, (i, _)| b | 1 << i);
    Perceptual {
        bits,
        mean: mean.map(|m| (m / 1024) as u8),
    }
}

/// The keys a track's album is known by: its album tag with its album
/// artist (where the album is tagged), and its folder. Two tracks whose
/// covers are compared share one of them.
pub fn album_keys(path: &Path, tags: Option<&jodel_playback::Tags>) -> Vec<String> {
    let mut keys = Vec::new();
    if let Some(tags) = tags {
        if let Some(album) = tags.album.as_deref().filter(|a| !a.trim().is_empty()) {
            let artist = [
                "ALBUMARTIST",
                "ALBUM ARTIST",
                "ALBUM_ARTIST",
                "TPE2",
                "aART",
            ]
            .iter()
            .find_map(|k| tags.get(k))
            .unwrap_or("");
            let text = format!(
                "album {}\u{1f}{}",
                album.trim().to_lowercase(),
                artist.trim().to_lowercase()
            );
            keys.push(format!("a{:016x}", crate::rhythm::hash(&text)));
        }
    }
    if let Some(dir) = path.parent() {
        let text = format!("folder {}", dir.display());
        keys.push(format!("f{:016x}", crate::rhythm::hash(&text)));
    }
    keys
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::art::sample_cover;

    #[test]
    fn the_same_picture_encoded_again_is_the_same_perceptually() {
        let cover = sample_cover(400);
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 60)
            .encode_image(&cover)
            .unwrap();
        let again = image::load_from_memory(&jpeg).unwrap().to_rgb8();
        let smaller = image::imageops::resize(&again, 300, 300, FilterType::Lanczos3);
        assert_ne!(pixel_hash(&cover), pixel_hash(&smaller));
        assert!(perceptual(&cover).same(&perceptual(&smaller)));
        // Another picture is not.
        let other = image::imageops::flip_horizontal(&image::imageops::rotate90(&cover));
        assert!(!perceptual(&cover).same(&perceptual(&other)));
        let p = perceptual(&cover);
        let words: Vec<String> = p.to_text().split(' ').map(str::to_owned).collect();
        let words: Vec<&str> = words.iter().map(String::as_str).collect();
        assert_eq!(Perceptual::parse(&words), Some(p));
    }

    #[test]
    fn an_album_is_known_by_its_tags_and_its_folder() {
        let mut tags = jodel_playback::Tags {
            album: Some("Fallen".into()),
            ..Default::default()
        };
        tags.all.push(("ALBUMARTIST".into(), "Evanescence".into()));
        let a = album_keys(Path::new("/m/a/01.flac"), Some(&tags));
        let b = album_keys(Path::new("/m/b/02.flac"), Some(&tags));
        assert_eq!(a.len(), 2);
        assert_eq!(a[0], b[0]);
        assert_ne!(a[1], b[1]);
        assert_eq!(
            album_keys(Path::new("/m/a/03.flac"), None),
            vec![a[1].clone()]
        );
    }
}
