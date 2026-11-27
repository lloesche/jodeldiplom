//! Karaoke mode's ground under everything: the cover cropped to the
//! screen's shape, shrunk, blurred and shaded (a plain dark gradient
//! without one), made small ([`small`]): the GPU scales it up to the
//! window ([`super::Visualiser`]). Without the game it is the backdrop
//! (the cover, [`super::Shows::Blur`]); with it, the plain gradient under
//! the story's scenes ([`super::Shows::Story`]).

use image::imageops::{self, FilterType};
use image::RgbImage;

use crate::view::palette;
use crate::work::art::Art;

/// How dark the layer over the blurred cover is (0 clear, 1 black).
const SHADE: f32 = 0.55;
/// The long side of the cover before it is blurred, in pixels: the blur
/// is done small and scaled up, so it is cheap at any screen size.
const SMALL: u32 = 48;

/// The ground for `art` (none: a plain dark gradient) on a screen of
/// `width` × `height`, as small as it is made, shaded: to be scaled up to
/// the screen.
pub fn small(art: Option<&Art>, (width, height): (u32, u32)) -> RgbImage {
    let (w, h) = (width.max(1), height.max(1));
    match art {
        Some(art) => {
            let mut small = blur_small(&art.image, w, h);
            small.pixels_mut().for_each(|p| p.0 = shade(p.0, SHADE));
            small
        }
        None => {
            let (sw, sh) = small_size(w, h);
            gradient(sw, sh)
        }
    }
}

/// The size the cover is blurred at for a screen of `w` × `h`.
fn small_size(w: u32, h: u32) -> (u32, u32) {
    if w >= h {
        (SMALL, (SMALL * h / w).max(1))
    } else {
        ((SMALL * w / h).max(1), SMALL)
    }
}

/// `image` cropped to the screen's shape (`w` × `h`), shrunk and blurred
/// three times.
fn blur_small(image: &RgbImage, w: u32, h: u32) -> RgbImage {
    // The middle of the cover in the screen's shape (it fills the screen).
    let (iw, ih) = (image.width().max(1), image.height().max(1));
    let screen = f64::from(w) / f64::from(h);
    let (cw, ch) = if f64::from(iw) / f64::from(ih) > screen {
        (((f64::from(ih) * screen).round() as u32).clamp(1, iw), ih)
    } else {
        (iw, ((f64::from(iw) / screen).round() as u32).clamp(1, ih))
    };
    let crop = imageops::crop_imm(image, (iw - cw) / 2, (ih - ch) / 2, cw, ch).to_image();
    let (sw, sh) = small_size(w, h);
    let mut small = imageops::resize(&crop, sw, sh, FilterType::Triangle);
    for _ in 0..3 {
        box_blur(&mut small, 2);
    }
    small
}

/// A plain ground without a cover, `w` × `h`: the window's background, a
/// little lighter towards the middle.
fn gradient(w: u32, h: u32) -> RgbImage {
    let base = palette::BG;
    RgbImage::from_fn(w, h, |_, y| {
        let t = 1.0 - (2.0 * f64::from(y) / f64::from(h.max(1)) - 1.0).abs();
        let lift = (18.0 * t) as u8;
        image::Rgb(base.map(|v| v.saturating_add(lift)))
    })
}

/// A box blur of `radius` pixels, across and then down, edges held.
fn box_blur(image: &mut RgbImage, radius: i64) {
    let (w, h) = (i64::from(image.width()), i64::from(image.height()));
    let n = (2 * radius + 1) as u32;
    let get = |img: &RgbImage, x: i64, y: i64| {
        img.get_pixel(x.clamp(0, w - 1) as u32, y.clamp(0, h - 1) as u32)
            .0
    };
    let source = image.clone();
    for y in 0..h {
        for x in 0..w {
            let mut sum = [0u32; 3];
            for d in -radius..=radius {
                let p = get(&source, x + d, y);
                (0..3).for_each(|i| sum[i] += u32::from(p[i]));
            }
            image.put_pixel(x as u32, y as u32, image::Rgb(sum.map(|s| (s / n) as u8)));
        }
    }
    let source = image.clone();
    for y in 0..h {
        for x in 0..w {
            let mut sum = [0u32; 3];
            for d in -radius..=radius {
                let p = get(&source, x, y + d);
                (0..3).for_each(|i| sum[i] += u32::from(p[i]));
            }
            image.put_pixel(x as u32, y as u32, image::Rgb(sum.map(|s| (s / n) as u8)));
        }
    }
}

fn shade(c: [u8; 3], dark: f32) -> [u8; 3] {
    c.map(|v| (f32::from(v) * (1.0 - dark)).round() as u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn the_ground_fills_the_screen_shaded_and_blurred() {
        let art = Art {
            id: 7,
            image: Arc::new(crate::work::art::sample_cover(64)),
        };
        let s = small(Some(&art), (320, 180));
        assert_eq!(s.dimensions(), (48, 27));
        // Shaded: nothing near white is left.
        assert!(s.pixels().all(|p| p.0.iter().all(|&v| v < 160)));
        // Blurred: neighbours differ little.
        let step = (0..s.height())
            .flat_map(|y| (1..s.width()).map(move |x| (x, y)))
            .map(|(x, y)| {
                let (a, c) = (s.get_pixel(x - 1, y).0, s.get_pixel(x, y).0);
                (0..3).map(|i| a[i].abs_diff(c[i])).max().unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        assert!(step < 40, "{step}");
        // Without a cover: the gradient, darkest at the edges.
        let plain = small(None, (320, 180));
        assert_eq!(plain.dimensions(), (48, 27));
        assert!(plain.get_pixel(0, 13).0[0] > plain.get_pixel(0, 0).0[0]);
    }
}
