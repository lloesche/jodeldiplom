//! Pictures and their parts as kept in the caches: PNGs ([`png`]), a depth
//! map in 16 bits ([`near_png`], [`near_from`]) and a palette as text
//! ([`palette_text`], [`palette_from`]).

use std::io::Cursor;

use image::{ImageBuffer, ImageFormat, Luma};

use super::{NearMap, Swatch};

/// A palette as kept: each colour `r,g,b,share`, a space between.
pub fn palette_text(palette: &[Swatch]) -> String {
    let swatches: Vec<String> = palette
        .iter()
        .map(|s| format!("{},{},{},{:.4}", s.rgb[0], s.rgb[1], s.rgb[2], s.share))
        .collect();
    swatches.join(" ")
}

/// A palette from its [`palette_text`] (a colour amiss left out).
pub fn palette_from(text: &str) -> Vec<Swatch> {
    text.split_whitespace()
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
        .collect()
}

/// A PNG of what `encode` writes.
pub fn png(encode: impl FnOnce(&mut Cursor<&mut Vec<u8>>) -> image::ImageResult<()>) -> Vec<u8> {
    let mut out = Vec::new();
    encode(&mut Cursor::new(&mut out)).expect("PNG into memory");
    out
}

/// A depth map as a 16-bit PNG.
pub fn near_png(n: &NearMap) -> Vec<u8> {
    let img: ImageBuffer<Luma<u16>, Vec<u16>> =
        ImageBuffer::from_raw(n.width, n.height, n.near.clone()).expect("its size");
    png(|c| img.write_to(c, ImageFormat::Png))
}

/// A depth map from a [`near_png`].
pub fn near_from(bytes: &[u8]) -> Option<NearMap> {
    let img = image::load_from_memory_with_format(bytes, ImageFormat::Png)
        .ok()?
        .to_luma16();
    Some(NearMap {
        width: img.width(),
        height: img.height(),
        near: img.into_raw(),
    })
}
