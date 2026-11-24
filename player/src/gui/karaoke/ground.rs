//! Karaoke mode's ground under the lyrics: the cover cropped to the
//! screen's shape, shrunk, blurred and shaded (a plain dark gradient
//! without one). Without a GPU it is painted into the pixmap at the
//! screen's size ([`Backdrop::new`]). On the GPU the pixmap's ground is
//! all see-through ([`Backdrop::through`]): the GPU draws the ground itself
//! from the small blur ([`small`]), with the cover's layers over it
//! ([`crate::gui::backdrop`]).

use image::imageops::{self, FilterType};
use image::RgbImage;

use super::rgb;
use crate::art::Art;
use crate::gui::raster::{pack, SEE_THROUGH};
use crate::view::colors;

/// How dark the layer over the blurred cover is (0 clear, 1 black).
const SHADE: f32 = 0.55;
/// The long side of the cover before it is blurred, in pixels: the blur
/// is done small and scaled up, so it is cheap at any screen size.
const SMALL: u32 = 48;

/// The screen's ground in the pixmap, at the screen's size: the cover
/// blurred and shaded, or all see-through where the GPU draws it.
pub struct Backdrop {
    /// The cover it is blurred from (0: none); `None` where it is all
    /// see-through.
    pub art: Option<u64>,
    /// Its size.
    pub size: (usize, usize),
    /// The pixels.
    pub pixels: Vec<u32>,
}

impl Backdrop {
    /// The ground for `art` (none: a plain dark gradient) at `width` ×
    /// `height`, `0x00RRGGBB` words.
    pub fn new(art: Option<&Art>, width: usize, height: usize) -> Backdrop {
        let (w, h) = (width.max(1) as u32, height.max(1) as u32);
        let pixels = match art {
            Some(art) => blurred(&art.image, w, h),
            None => gradient(w, h),
        };
        Backdrop {
            art: Some(art.map_or(0, |a| a.id)),
            size: (width, height),
            pixels,
        }
    }

    /// A ground of `width` × `height` all see-through: the GPU draws it.
    pub fn through(width: usize, height: usize) -> Backdrop {
        Backdrop {
            art: None,
            size: (width, height),
            pixels: vec![SEE_THROUGH; width * height],
        }
    }
}

/// The ground for `art` (none: the gradient) on a screen of `width` ×
/// `height`, as small as it is made: what [`Backdrop::new`] scales up to
/// the screen, shaded alike (the GPU scales it up).
pub fn small(art: Option<&Art>, (width, height): (u32, u32)) -> RgbImage {
    let (w, h) = (width.max(1), height.max(1));
    let (sw, sh) = small_size(w, h);
    match art {
        Some(art) => {
            let mut small = blur_small(&art.image, w, h);
            small.pixels_mut().for_each(|p| p.0 = shade(p.0, SHADE));
            small
        }
        None => {
            let words = gradient(sw, sh);
            RgbImage::from_fn(sw, sh, |x, y| {
                let w = words[(y * sw + x) as usize];
                image::Rgb([(w >> 16) as u8, (w >> 8) as u8, w as u8])
            })
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

/// `image` blurred small ([`blur_small`]), scaled up to `w` × `h` and
/// shaded.
fn blurred(image: &RgbImage, w: u32, h: u32) -> Vec<u32> {
    let big = imageops::resize(&blur_small(image, w, h), w, h, FilterType::Triangle);
    big.pixels().map(|p| pack(shade(p.0, SHADE))).collect()
}

/// A plain backdrop without a cover: the window's background, a little
/// lighter towards the middle.
fn gradient(w: u32, h: u32) -> Vec<u32> {
    let base = rgb(colors::BG);
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        let t = 1.0 - (2.0 * f64::from(y) / f64::from(h.max(1)) - 1.0).abs();
        let lift = (18.0 * t) as u8;
        let c = base.map(|v| v.saturating_add(lift));
        out.extend(std::iter::repeat_n(pack(c), w as usize));
    }
    out
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
    use crate::gui::raster::unpack;
    use std::sync::Arc;

    #[test]
    fn the_ground_fills_the_screen_shaded_and_blurred() {
        let art = Art {
            id: 7,
            image: Arc::new(crate::art::sample_cover(64)),
        };
        let b = Backdrop::new(Some(&art), 320, 180);
        assert_eq!(
            (b.art, b.size, b.pixels.len()),
            (Some(7), (320, 180), 320 * 180)
        );
        // Shaded: nothing near white is left.
        assert!(b.pixels.iter().all(|&p| unpack(p).iter().all(|&v| v < 160)));
        // Blurred: neighbours differ little.
        let step = b
            .pixels
            .chunks(320)
            .flat_map(|row| row.windows(2))
            .map(|w| {
                let (a, c) = (unpack(w[0]), unpack(w[1]));
                (0..3).map(|i| a[i].abs_diff(c[i])).max().unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        assert!(step < 40, "{step}");
        assert_eq!(Backdrop::new(None, 10, 10).art, Some(0));
        // Small for the GPU: the same picture before it is scaled up.
        let s = small(Some(&art), (320, 180));
        assert_eq!(s.dimensions(), (48, 27));
        let big = |x: u32, y: u32| unpack(b.pixels[(y * 320 + x) as usize]);
        let near = s.get_pixel(24, 13).0;
        let mid = big(160, 80);
        assert!(
            (0..3).all(|i| near[i].abs_diff(mid[i]) < 12),
            "{near:?} {mid:?}"
        );
        // On the GPU: all see-through.
        let through = Backdrop::through(4, 3);
        assert_eq!((through.art, through.pixels), (None, vec![SEE_THROUGH; 12]));
    }
}
