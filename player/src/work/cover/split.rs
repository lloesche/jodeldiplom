//! A cover taken apart from what the models found in it ([`Parts`]: its
//! depth, its subject's matte, its lettering and faces): the subject (the
//! matte cleaned of lettering, `clean.rs`) with its colours freed of the old
//! background at the edges, the plate with the subject removed (its matte
//! grown a little first, so no outline of it stays) and filled, the depth
//! of both, the subject's rim, the palette.

use image::{GrayImage, Luma, RgbImage, Rgba, RgbaImage};
use jodel_gpu::vision::{Depth, Matte};

use super::faces::Face;
use super::fill::{box_blur, dilate, fill, push_pull, resize, Planes};
use super::text::TextMap;
use super::{CoverLayers, NearMap, Swatch};

/// What the models found in a cover.
#[derive(Clone, Debug, PartialEq)]
pub struct Parts {
    /// Depth Anything 3 Small's depth.
    pub depth: Depth,
    /// BEN2's matte of the subject.
    pub matte: Matte,
    /// PP-OCRv5's map of the lettering.
    pub text: TextMap,
    /// YuNet's faces.
    pub faces: Vec<Face>,
}

/// The subject's colours freed of the background they were over (the blur
/// fusion foreground estimate, as BEN2's own code makes it: the means of
/// the subject and of the background around, twice, wide then narrow).
fn foreground(image: &Planes, alpha: &[f32]) -> Planes {
    let (w, h) = (image.width, image.height);
    let scale = w.max(h) as f32 / 1024.0;
    let mut unit = image.clone();
    unit.data.iter_mut().for_each(|v| *v /= 255.0);
    let pass = |f: &Planes, b: &Planes, r: usize| -> (Planes, Planes) {
        let mut fa = Planes::new(w, h, 3);
        let mut ba = Planes::new(w, h, 3);
        let mut a1 = Planes::new(w, h, 1);
        for i in 0..w * h {
            a1.data[i] = alpha[i];
            for c in 0..3 {
                fa.data[i * 3 + c] = f.data[i * 3 + c] * alpha[i];
                ba.data[i * 3 + c] = b.data[i * 3 + c] * (1.0 - alpha[i]);
            }
        }
        let (fa, ba, ab) = (box_blur(&fa, r), box_blur(&ba, r), box_blur(&a1, r));
        let mut out = Planes::new(w, h, 3);
        let mut blur_b = Planes::new(w, h, 3);
        for i in 0..w * h {
            let a = alpha[i];
            for c in 0..3 {
                let bf = fa.data[i * 3 + c] / (ab.data[i] + 1e-5);
                let bb = ba.data[i * 3 + c] / (1.0 - ab.data[i] + 1e-5);
                blur_b.data[i * 3 + c] = bb;
                let v = bf + a * (unit.data[i * 3 + c] - a * bf - (1.0 - a) * bb);
                out.data[i * 3 + c] = v.clamp(0.0, 1.0);
            }
        }
        (out, blur_b)
    };
    let wide = ((45.0 * scale) as usize).max(2);
    let narrow = ((3.0 * scale) as usize).max(1);
    let (f, b) = pass(&unit, &unit, wide);
    let (mut f, _) = pass(&f, &b, narrow);
    f.data.iter_mut().for_each(|v| *v *= 255.0);
    f
}

/// Nearness from depth (its reciprocal, the 1st to 99th percentile spread
/// over 0..=1).
fn nearness(depth: &Depth) -> Planes {
    let disparity: Vec<f32> = depth.values.iter().map(|&d| 1.0 / d.max(1e-6)).collect();
    let mut sorted = disparity.clone();
    sorted.sort_by(f32::total_cmp);
    let pick = |q: f32| sorted[((sorted.len() - 1) as f32 * q) as usize];
    let (lo, hi) = (pick(0.01), pick(0.99));
    let span = (hi - lo).max(1e-9);
    Planes {
        width: depth.width,
        height: depth.height,
        channels: 1,
        data: disparity
            .iter()
            .map(|&d| ((d - lo) / span).clamp(0.0, 1.0))
            .collect(),
    }
}

/// Nearness as a straight share of the depth's range (0 farthest, 1
/// nearest) at `w × h`: what a subject is grown through from a face.
fn depth_near(depth: &Depth, w: usize, h: usize) -> Vec<f32> {
    let (lo, hi) = depth
        .values
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let span = (hi - lo).max(1e-9);
    let p = Planes {
        width: depth.width,
        height: depth.height,
        channels: 1,
        data: depth
            .values
            .iter()
            .map(|&d| 1.0 - (d - lo) / span)
            .collect(),
    };
    resize(&p, w, h).data
}

fn near_map(p: &Planes) -> NearMap {
    NearMap {
        width: p.width as u32,
        height: p.height as u32,
        near: p
            .data
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 65535.0).round() as u16)
            .collect(),
    }
}

/// `p` (one channel in 0..1) as an 8-bit image.
fn gray(p: &Planes) -> GrayImage {
    GrayImage::from_fn(p.width as u32, p.height as u32, |x, y| {
        Luma([(p.data[y as usize * p.width + x as usize].clamp(0.0, 1.0) * 255.0).round() as u8])
    })
}

/// The edges of `p` (one channel): its Sobel gradient's size, its 99th
/// percentile to 1.
fn edges(p: &Planes) -> Planes {
    let (w, h) = (p.width, p.height);
    let at = |x: isize, y: isize| {
        p.data[(y.clamp(0, h as isize - 1) as usize) * w + x.clamp(0, w as isize - 1) as usize]
    };
    let mut out = Planes::new(w, h, 1);
    for y in 0..h as isize {
        for x in 0..w as isize {
            let gx = at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1)
                - at(x - 1, y - 1)
                - 2.0 * at(x - 1, y)
                - at(x - 1, y + 1);
            let gy = at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1)
                - at(x - 1, y - 1)
                - 2.0 * at(x, y - 1)
                - at(x + 1, y - 1);
            out.data[y as usize * w + x as usize] = (gx * gx + gy * gy).sqrt();
        }
    }
    let mut sorted = out.data.clone();
    sorted.sort_by(f32::total_cmp);
    let top = sorted[(sorted.len() - 1) * 99 / 100].max(1e-6);
    out.data.iter_mut().for_each(|v| *v = (*v / top).min(1.0));
    out
}

/// The cover's main colours: k-means of five over its pixels at 64 × 64,
/// started from colours far apart, the largest share first.
fn palette(image: &RgbImage) -> Vec<Swatch> {
    let small = image::imageops::resize(image, 64, 64, image::imageops::FilterType::Triangle);
    let pixels: Vec<[f32; 3]> = small
        .pixels()
        .map(|p| [f32::from(p[0]), f32::from(p[1]), f32::from(p[2])])
        .collect();
    let dist = |a: &[f32; 3], b: &[f32; 3]| (0..3).map(|c| (a[c] - b[c]).powi(2)).sum::<f32>();
    let n = pixels.len() as f32;
    let mean = pixels.iter().fold([0.0; 3], |m, p| {
        [m[0] + p[0] / n, m[1] + p[1] / n, m[2] + p[2] / n]
    });
    let mut centres: Vec<[f32; 3]> = Vec::new();
    let first = *pixels
        .iter()
        .max_by(|a, b| dist(a, &mean).total_cmp(&dist(b, &mean)))
        .unwrap();
    centres.push(first);
    while centres.len() < 5 {
        let far = *pixels
            .iter()
            .max_by(|a, b| {
                let da = centres.iter().map(|c| dist(a, c)).fold(f32::MAX, f32::min);
                let db = centres.iter().map(|c| dist(b, c)).fold(f32::MAX, f32::min);
                da.total_cmp(&db)
            })
            .unwrap();
        centres.push(far);
    }
    let mut counts = vec![0usize; 5];
    for _ in 0..12 {
        let mut sums = [[0.0f32; 3]; 5];
        counts = vec![0; 5];
        for p in &pixels {
            let k = (0..5)
                .min_by(|&a, &b| dist(p, &centres[a]).total_cmp(&dist(p, &centres[b])))
                .unwrap();
            counts[k] += 1;
            for c in 0..3 {
                sums[k][c] += p[c];
            }
        }
        for k in 0..5 {
            if counts[k] > 0 {
                centres[k] = sums[k].map(|s| s / counts[k] as f32);
            }
        }
    }
    let mut out: Vec<Swatch> = (0..5)
        .filter(|&k| counts[k] > 0)
        .map(|k| Swatch {
            rgb: centres[k].map(|v| v.round().clamp(0.0, 255.0) as u8),
            share: counts[k] as f32 / n,
        })
        .collect();
    out.sort_by(|a, b| b.share.total_cmp(&a.share));
    out
}

/// The hole the subject leaves in a `w × h` cover, from its matte: all it
/// touches, grown by 8 pixels a 1024 (its glow and the matte's soft edge go
/// with it), the edge softened over 2; 1 where the plate is all fill.
pub fn hole(alpha: &[f32], w: usize, h: usize) -> Vec<f32> {
    let scale = w.max(h) as f32 / 1024.0;
    let touched: Vec<f32> = alpha
        .iter()
        .map(|&a| if a > 0.03 { 1.0 } else { 0.0 })
        .collect();
    let grown = dilate(&touched, w, h, ((8.0 * scale) as usize).max(3));
    let soft = box_blur(
        &Planes {
            width: w,
            height: h,
            channels: 1,
            data: grown,
        },
        ((2.0 * scale) as usize).max(1),
    );
    soft.data.iter().map(|&v| (v * 1.5).min(1.0)).collect()
}

/// `image` taken apart by what the models found in it.
pub fn split(image: &RgbImage, parts: &Parts) -> CoverLayers {
    let (w, h) = (image.width() as usize, image.height() as usize);
    let pixels = Planes {
        width: w,
        height: h,
        channels: 3,
        data: image.as_raw().iter().map(|&v| f32::from(v)).collect(),
    };
    let matte = Planes {
        width: parts.matte.width,
        height: parts.matte.height,
        channels: 1,
        data: parts.matte.values.clone(),
    };
    let alpha = resize(&matte, w, h).data;
    let near = nearness(&parts.depth);
    let depth = near_map(&near);
    let scale = w.max(h) as f32 / 1024.0;
    let Ok(alpha) = super::clean::subject(
        &alpha,
        &depth_near(&parts.depth, w, h),
        w,
        h,
        &parts.text,
        &parts.faces,
    ) else {
        let rim = resize(&edges(&near), w, h);
        return CoverLayers {
            plate: image.clone(),
            subject: None,
            plate_depth: depth.clone(),
            depth,
            outline: gray(&rim),
            palette: palette(image),
        };
    };
    // The subject: its matte, its colours freed of the old background.
    let colours = foreground(&pixels, &alpha);
    let subject = RgbaImage::from_fn(w as u32, h as u32, |x, y| {
        let i = y as usize * w + x as usize;
        let c = |k: usize| colours.data[i * 3 + k].round().clamp(0.0, 255.0) as u8;
        Rgba([
            c(0),
            c(1),
            c(2),
            (alpha[i] * 255.0).round().clamp(0.0, 255.0) as u8,
        ])
    });
    let hole = hole(&alpha, w, h);
    let plate_px = fill(&pixels, &hole);
    let plate = RgbImage::from_fn(w as u32, h as u32, |x, y| {
        let i = (y as usize * w + x as usize) * 3;
        image::Rgb([0, 1, 2].map(|k| plate_px.data[i + k].round().clamp(0.0, 255.0) as u8))
    });
    // The plate's depth: under the hole, from around it.
    let small_hole = resize(
        &Planes {
            width: w,
            height: h,
            channels: 1,
            data: hole,
        },
        near.width,
        near.height,
    );
    let known: Vec<f32> = small_hole
        .data
        .iter()
        .map(|&a| 1.0 - a.clamp(0.0, 1.0))
        .collect();
    let filled = push_pull(&near, &known);
    let plate_near = Planes {
        data: near
            .data
            .iter()
            .zip(&filled.data)
            .zip(&known)
            .map(|((&n, &f), &k)| n * k + f * (1.0 - k))
            .collect(),
        ..near.clone()
    };
    // The rim: the matte where its blur falls away, inside the edge.
    let a1 = Planes {
        width: w,
        height: h,
        channels: 1,
        data: alpha.clone(),
    };
    let blurred = box_blur(&a1, ((10.0 * scale) as usize).max(2));
    let rim = Planes {
        data: alpha
            .iter()
            .zip(&blurred.data)
            .map(|(&a, &b)| (a * (1.0 - b) * 2.0).min(1.0))
            .collect(),
        ..a1
    };
    CoverLayers {
        plate,
        subject: Some(subject),
        depth,
        plate_depth: near_map(&plate_near),
        outline: gray(&rim),
        palette: palette(image),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grey cover with a red disc, its matte the disc, its depth nearer
    /// there.
    fn disc(side: usize) -> (RgbImage, Parts) {
        let inside = |x: usize, y: usize| {
            let (dx, dy) = (x as f32 - side as f32 / 2.0, y as f32 - side as f32 / 2.0);
            (dx * dx + dy * dy).sqrt() < side as f32 / 4.0
        };
        let image = RgbImage::from_fn(side as u32, side as u32, |x, y| {
            if inside(x as usize, y as usize) {
                image::Rgb([220, 20, 30])
            } else {
                image::Rgb([90, 90, 90])
            }
        });
        let values = (0..side * side)
            .map(|i| if inside(i % side, i / side) { 1.0 } else { 0.0 })
            .collect();
        let depth = (0..side * side)
            .map(|i| if inside(i % side, i / side) { 1.0 } else { 4.0 })
            .collect();
        let parts = Parts {
            depth: Depth {
                width: side,
                height: side,
                values: depth,
            },
            matte: Matte {
                width: side,
                height: side,
                values,
            },
            text: TextMap::none(),
            faces: Vec::new(),
        };
        (image, parts)
    }

    #[test]
    fn a_subject_comes_off_and_its_hole_is_filled_with_the_background() {
        let (image, parts) = disc(128);
        let l = split(&image, &parts);
        let s = l.subject.as_ref().expect("a subject");
        // Its centre is the disc's colour, fully there; a corner not at all.
        assert_eq!(s.get_pixel(64, 64).0, [220, 20, 30, 255]);
        assert_eq!(s.get_pixel(2, 2).0[3], 0);
        // The plate shows no red where the disc was, nor at its edge.
        for (x, y) in [(64, 64), (64, 33), (96, 64)] {
            let p = l.plate.get_pixel(x, y).0;
            assert!(p[0] < 110 && p[1] > 70, "({x}, {y}): {p:?}");
        }
        // Nearer in the middle as it is; the plate's depth there as around.
        assert!(l.depth.at(0.5, 0.5) > 0.9 && l.depth.at(0.05, 0.05) < 0.1);
        assert!(l.plate_depth.at(0.5, 0.5) < 0.1);
        // The rim lights the disc's edge, not its middle.
        assert!(l.outline.get_pixel(64, 33).0[0] > 100);
        assert_eq!(l.outline.get_pixel(64, 64).0[0], 0);
        assert_eq!(l.palette[0].rgb, [90, 90, 90]);
    }

    #[test]
    fn a_cover_without_a_clear_subject_stays_whole() {
        let (image, mut parts) = disc(64);
        parts.matte.values.iter_mut().for_each(|v| *v = 0.5);
        let l = split(&image, &parts);
        assert!(l.subject.is_none());
        assert_eq!(l.plate, image);
    }
}
