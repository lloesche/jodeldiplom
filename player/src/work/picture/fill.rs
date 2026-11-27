//! Filling a hole in a picture from what is around it, cheaply: push-pull
//! (a pyramid of averages of the known pixels down to one, then each level's
//! unknown pixels from the coarser one, up again), and grain on top as fine
//! as the picture's around it, so the fill does not look smeared flat; a
//! picture's plate behind what is taken out of it ([`plate`], [`hole`]).
//! And the image helpers the taking apart uses: box blurs, dilation,
//! resizing.

use image::RgbImage;

use super::mask::Size;

/// A picture of `channels` floats a pixel, rows top to bottom.
#[derive(Clone, Debug, PartialEq)]
pub struct Planes {
    /// Its width in pixels.
    pub width: usize,
    /// Its height in pixels.
    pub height: usize,
    /// The floats a pixel.
    pub channels: usize,
    /// The pixels, each its channels in turn.
    pub data: Vec<f32>,
}

impl Planes {
    /// A picture of `width × height`, `channels` a pixel, all 0.
    pub fn new(width: usize, height: usize, channels: usize) -> Planes {
        Planes {
            width,
            height,
            channels,
            data: vec![0.0; width * height * channels],
        }
    }

    /// Its pixels as an RGB picture (three channels, 0..255, rounded).
    pub fn to_rgb(&self) -> RgbImage {
        let w = self.width;
        RgbImage::from_fn(self.width as u32, self.height as u32, |x, y| {
            let i = (y as usize * w + x as usize) * 3;
            image::Rgb([0, 1, 2].map(|k| self.data[i + k].round().clamp(0.0, 255.0) as u8))
        })
    }

    fn at(&self, x: usize, y: usize, c: usize) -> f32 {
        self.data[(y * self.width + x) * self.channels + c]
    }
}

/// One level of the pyramid: the known pixels' sums and their weights.
struct Level {
    width: usize,
    height: usize,
    sum: Vec<f32>,
    weight: Vec<f32>,
}

/// `picture` with its pixels of weight 0 in `known` (0..=1 each) filled by
/// push-pull; pixels of weight 1 kept as they are, partly known ones
/// blended. Without a known pixel at all, it is returned as it is.
pub fn push_pull(picture: &Planes, known: &[f32]) -> Planes {
    let ch = picture.channels;
    assert_eq!(known.len(), picture.width * picture.height);
    if known.iter().all(|&k| k <= 0.0) {
        return picture.clone();
    }
    let mut levels = vec![Level {
        width: picture.width,
        height: picture.height,
        sum: picture
            .data
            .iter()
            .enumerate()
            .map(|(i, &v)| v * known[i / ch])
            .collect(),
        weight: known.to_vec(),
    }];
    while levels.last().is_some_and(|l| l.width > 1 || l.height > 1) {
        let l = levels.last().unwrap();
        let (w, h) = (l.width.div_ceil(2), l.height.div_ceil(2));
        let mut sum = vec![0.0; w * h * ch];
        let mut weight = vec![0.0; w * h];
        for y in 0..l.height {
            for x in 0..l.width {
                let to = (y / 2) * w + x / 2;
                let from = y * l.width + x;
                weight[to] += l.weight[from] * 0.25;
                for c in 0..ch {
                    sum[to * ch + c] += l.sum[from * ch + c] * 0.25;
                }
            }
        }
        levels.push(Level {
            width: w,
            height: h,
            sum,
            weight,
        });
    }
    // Down to one pixel, then back up.
    let top = levels.last().unwrap();
    let mut colour: Vec<f32> = (0..ch)
        .map(|c| top.sum[c] / top.weight[0].max(1e-12))
        .collect();
    let (mut cw, mut chh) = (1, 1);
    for l in levels.iter().rev().skip(1) {
        let mut next = vec![0.0; l.width * l.height * ch];
        for y in 0..l.height {
            for x in 0..l.width {
                // The coarser level, bilinear at this pixel's centre.
                let fx = ((x as f32 + 0.5) / 2.0 - 0.5).clamp(0.0, (cw - 1) as f32);
                let fy = ((y as f32 + 0.5) / 2.0 - 0.5).clamp(0.0, (chh - 1) as f32);
                let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
                let (x1, y1) = ((x0 + 1).min(cw - 1), (y0 + 1).min(chh - 1));
                let (ax, ay) = (fx - x0 as f32, fy - y0 as f32);
                let i = y * l.width + x;
                let trust = (l.weight[i] * 2.0).min(1.0);
                for c in 0..ch {
                    let g = |xx: usize, yy: usize| colour[(yy * cw + xx) * ch + c];
                    let up = (1.0 - ay) * ((1.0 - ax) * g(x0, y0) + ax * g(x1, y0))
                        + ay * ((1.0 - ax) * g(x0, y1) + ax * g(x1, y1));
                    let mine = l.sum[i * ch + c] / l.weight[i].max(1e-12);
                    next[i * ch + c] = mine * trust + up * (1.0 - trust);
                }
            }
        }
        colour = next;
        (cw, chh) = (l.width, l.height);
    }
    Planes {
        data: colour,
        ..picture.clone()
    }
}

/// `image`'s pixels as a texture takes them: four bytes a pixel, opaque.
pub fn rgba(image: &RgbImage) -> Vec<u8> {
    image
        .pixels()
        .flat_map(|p| [p.0[0], p.0[1], p.0[2], 255])
        .collect()
}

impl From<&RgbImage> for Planes {
    /// Its pixels, three channels, 0..255.
    fn from(image: &RgbImage) -> Planes {
        Planes {
            width: image.width() as usize,
            height: image.height() as usize,
            channels: 3,
            data: image.as_raw().iter().map(|&v| f32::from(v)).collect(),
        }
    }
}

/// A box blur of radius `r` (a square of 2r + 1 a side, its edges clamped)
/// of each channel.
pub fn box_blur(p: &Planes, r: usize) -> Planes {
    let pass = |src: &Planes, horizontal: bool| -> Planes {
        let mut out = Planes::new(src.width, src.height, src.channels);
        let (n, m) = if horizontal {
            (src.width, src.height)
        } else {
            (src.height, src.width)
        };
        let ch = src.channels;
        let mut line = vec![0.0f64; n + 1];
        for j in 0..m {
            for c in 0..ch {
                for i in 0..n {
                    let v = if horizontal {
                        src.at(i, j, c)
                    } else {
                        src.at(j, i, c)
                    };
                    line[i + 1] = line[i] + f64::from(v);
                }
                for i in 0..n {
                    let lo = i.saturating_sub(r);
                    let hi = (i + r + 1).min(n);
                    let v = ((line[hi] - line[lo]) / (hi - lo) as f64) as f32;
                    let at = if horizontal {
                        j * src.width + i
                    } else {
                        i * src.width + j
                    };
                    out.data[at * ch + c] = v;
                }
            }
        }
        out
    };
    pass(&pass(p, true), false)
}

/// `mask` (one value a pixel) grown by `r` pixels: each the largest within
/// a square of 2r + 1.
pub fn dilate(mask: &[f32], width: usize, height: usize, r: usize) -> Vec<f32> {
    let pass = |src: &[f32], horizontal: bool| -> Vec<f32> {
        let mut out = vec![0.0; src.len()];
        for y in 0..height {
            for x in 0..width {
                let mut most = 0.0f32;
                if horizontal {
                    for xx in x.saturating_sub(r)..(x + r + 1).min(width) {
                        most = most.max(src[y * width + xx]);
                    }
                } else {
                    for yy in y.saturating_sub(r)..(y + r + 1).min(height) {
                        most = most.max(src[yy * width + x]);
                    }
                }
                out[y * width + x] = most;
            }
        }
        out
    };
    pass(&pass(mask, true), false)
}

/// `p` resized to `width × height`, bilinear (pixel centres aligned).
pub fn resize(p: &Planes, width: usize, height: usize) -> Planes {
    let mut out = Planes::new(width, height, p.channels);
    for y in 0..height {
        let fy = ((y as f32 + 0.5) * p.height as f32 / height as f32 - 0.5)
            .clamp(0.0, (p.height - 1) as f32);
        let (y0, ay) = (fy.floor() as usize, fy - fy.floor());
        let y1 = (y0 + 1).min(p.height - 1);
        for x in 0..width {
            let fx = ((x as f32 + 0.5) * p.width as f32 / width as f32 - 0.5)
                .clamp(0.0, (p.width - 1) as f32);
            let (x0, ax) = (fx.floor() as usize, fx - fx.floor());
            let x1 = (x0 + 1).min(p.width - 1);
            for c in 0..p.channels {
                let v = (1.0 - ay) * ((1.0 - ax) * p.at(x0, y0, c) + ax * p.at(x1, y0, c))
                    + ay * ((1.0 - ax) * p.at(x0, y1, c) + ax * p.at(x1, y1, c));
                out.data[(y * width + x) * p.channels + c] = v;
            }
        }
    }
    out
}

/// A repeatable noise value in -1..1 for a pixel.
fn noise(x: usize, y: usize) -> f32 {
    let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h & 0xFFFF) as f32 / 32767.5 - 1.0
}

/// Repeatable smooth noise in -1..1: [`noise`] on a grid `cell` pixels
/// apart, bilinear between.
fn smooth_noise(x: usize, y: usize, cell: usize) -> f32 {
    let (gx, gy) = (x / cell, y / cell);
    let (fx, fy) = (
        (x % cell) as f32 / cell as f32,
        (y % cell) as f32 / cell as f32,
    );
    let n = |a: usize, b: usize| noise(a + 7919, b + 104_729);
    (1.0 - fy) * ((1.0 - fx) * n(gx, gy) + fx * n(gx + 1, gy))
        + fy * ((1.0 - fx) * n(gx, gy + 1) + fx * n(gx + 1, gy + 1))
}

/// `picture` (RGB in 0..255) with the pixels where `hole` is above 0 filled
/// from around them ([`push_pull`]), the fill grained at two scales as the
/// picture around the hole is: fine (its pixels' difference from their
/// 3 × 3 mean) and mottled (that mean's from the 9 × 9 one), each filled in
/// the same way. Pixels partly in the hole blend the fill in by how much
/// they are.
pub fn fill(picture: &Planes, hole: &[f32]) -> Planes {
    let (w, h) = (picture.width, picture.height);
    let known: Vec<f32> = hole.iter().map(|&a| 1.0 - a.clamp(0.0, 1.0)).collect();
    let mut filled = push_pull(picture, &known);
    // How grainy the picture is around, as brightness, where it is known.
    let fine = box_blur(picture, 1);
    let coarse = box_blur(picture, 4);
    let mut grain = Planes::new(w, h, 2);
    for i in 0..w * h {
        let diff = |a: &Planes, b: &Planes| -> f32 {
            (0..3)
                .map(|c| (a.data[i * 3 + c] - b.data[i * 3 + c]).abs())
                .sum::<f32>()
                / 3.0
        };
        grain.data[i * 2] = diff(picture, &fine);
        grain.data[i * 2 + 1] = diff(&fine, &coarse);
    }
    let solid: Vec<f32> = known
        .iter()
        .map(|&k| if k >= 1.0 { 1.0 } else { 0.0 })
        .collect();
    let grain = push_pull(&grain, &solid);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let a = hole[i].clamp(0.0, 1.0);
            let n = noise(x, y) * grain.data[i * 2] * 1.6
                + smooth_noise(x, y, 6) * grain.data[i * 2 + 1] * 1.6;
            for c in 0..3 {
                let v = filled.data[i * 3 + c] + n * a;
                filled.data[i * 3 + c] =
                    picture.data[i * 3 + c] * (1.0 - a) + v.clamp(0.0, 255.0) * a;
            }
        }
    }
    filled
}

/// The least share of a pixel that is taken out with the rest.
const HOLE_TOUCHED: f32 = 0.03;
/// How far the hole is grown past what is taken out (pixels at a 1024
/// picture's size; so many at least).
const HOLE_GROWN: (f32, usize) = (8.0, 3);
/// How softly its edge goes (pixels at a 1024 picture's size).
const HOLE_SOFT: f32 = 2.0;
/// How much of its softened edge is all hole (the share there times this).
const HOLE_FULL: f32 = 1.5;

/// The hole what is taken out leaves in a `w × h` picture, from its share
/// of each pixel (`alpha`): all it touches ([`HOLE_TOUCHED`]), grown
/// ([`HOLE_GROWN`]: its glow and the matte's soft edge go with it), the
/// edge softened ([`HOLE_SOFT`]); 1 where the plate is all fill.
pub fn hole(alpha: &[f32], w: usize, h: usize) -> Vec<f32> {
    let size = Size::of(w, h);
    let touched: Vec<f32> = alpha
        .iter()
        .map(|&a| if a > HOLE_TOUCHED { 1.0 } else { 0.0 })
        .collect();
    let grown = dilate(&touched, w, h, size.px(HOLE_GROWN.0).max(HOLE_GROWN.1));
    let soft = box_blur(
        &Planes {
            width: w,
            height: h,
            channels: 1,
            data: grown,
        },
        size.px(HOLE_SOFT),
    );
    soft.data
        .iter()
        .map(|&v| (v * HOLE_FULL).min(1.0))
        .collect()
}

/// `image` with what is taken out of it (`alpha`, its share of each pixel,
/// the image's size) gone and the hole filled from round it ([`fill`]): the
/// plate behind, and the hole ([`hole`]).
pub fn plate(image: &RgbImage, alpha: &[f32]) -> (RgbImage, Vec<f32>) {
    let (w, h) = (image.width() as usize, image.height() as usize);
    let hole = hole(alpha, w, h);
    (fill(&Planes::from(image), &hole).to_rgb(), hole)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_pull_keeps_the_known_and_fills_the_hole_from_around() {
        // Left half 10, right half 200, a hole in the middle.
        let (w, h) = (16, 8);
        let mut p = Planes::new(w, h, 1);
        let mut known = vec![1.0; w * h];
        for y in 0..h {
            for x in 0..w {
                p.data[y * w + x] = if x < 8 { 10.0 } else { 200.0 };
                if (6..10).contains(&x) {
                    known[y * w + x] = 0.0;
                    p.data[y * w + x] = 999.0;
                }
            }
        }
        let f = push_pull(&p, &known);
        assert_eq!(f.data[0], 10.0);
        assert_eq!(f.data[w - 1], 200.0);
        let mid = f.data[3 * w + 6];
        assert!((10.0..=200.0).contains(&mid), "{mid}");
        // Nearer the dark side, darker.
        assert!(f.data[3 * w + 6] < f.data[3 * w + 9]);
    }

    #[test]
    fn a_flat_picture_fills_flat_without_grain() {
        let (w, h) = (20, 20);
        let mut p = Planes::new(w, h, 3);
        p.data.iter_mut().for_each(|v| *v = 77.0);
        let mut hole = vec![0.0; w * h];
        hole[10 * w + 10] = 1.0;
        let f = fill(&p, &hole);
        assert!(f.data.iter().all(|&v| (v - 77.0).abs() < 1e-3));
    }

    #[test]
    fn dilation_grows_a_point_into_a_square() {
        let mut m = vec![0.0; 25];
        m[12] = 1.0;
        let d = dilate(&m, 5, 5, 1);
        assert_eq!(d.iter().filter(|&&v| v == 1.0).count(), 9);
    }
}
