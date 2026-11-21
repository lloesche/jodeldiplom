//! Cover art: decoding, fitting into a pixel rect, and a placeholder.

use std::sync::Arc;

use image::{imageops::FilterType, Rgb, RgbImage};

/// A decoded cover with an id that changes when the cover does.
#[derive(Clone, Debug)]
pub struct Art {
    /// Changes with every new cover (for caches).
    pub id: u64,
    /// The pixels.
    pub image: Arc<RgbImage>,
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

/// A generated square cover for tests and screenshots: concentric
/// rings over a diagonal gradient.
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
