//! The networks' inputs made from a picture as their authors' code makes
//! them: OpenCV's area resampling (Depth Anything 3's input processor),
//! Pillow's Lanczos (BEN2's), each to 8 bits as those libraries round, then
//! ImageNet's normalisation.

/// ImageNet's means and deviations, red, green, blue.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// An 8-bit RGB picture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgb {
    pub width: usize,
    pub height: usize,
    /// Rows of pixels, three bytes each.
    pub data: Vec<u8>,
}

impl Rgb {
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Rgb {
        assert_eq!(data.len(), width * height * 3);
        Rgb {
            width,
            height,
            data,
        }
    }
}

/// `image` normalised (`[y][x][rgb]`).
pub fn normalise(image: &Rgb) -> Vec<f32> {
    image
        .data
        .chunks_exact(3)
        .flat_map(|p| (0..3).map(move |c| (f32::from(p[c]) / 255.0 - MEAN[c]) / STD[c]))
        .collect()
}

/// One axis's taps: for each output, its first input and weights.
type Taps = Vec<(usize, Vec<f64>)>;

/// OpenCV's `INTER_AREA` taps when shrinking `from` to `to` (each output
/// the mean of the inputs its cell covers, partly covered ones by their
/// share), as `computeResizeAreaTab`.
fn area_taps(from: usize, to: usize) -> Taps {
    let scale = from as f64 / to as f64;
    (0..to)
        .map(|dx| {
            let fsx1 = dx as f64 * scale;
            let fsx2 = fsx1 + scale;
            let cell = scale.min(from as f64 - fsx1);
            let mut sx1 = fsx1.ceil() as usize;
            let sx2 = (fsx2.floor() as usize).min(from - 1);
            sx1 = sx1.min(sx2);
            let mut taps = Vec::new();
            let mut first = sx1;
            if sx1 as f64 - fsx1 > 1e-3 {
                first = sx1 - 1;
                taps.push((sx1 as f64 - fsx1) / cell);
            }
            for _ in sx1..sx2 {
                taps.push(1.0 / cell);
            }
            if fsx2 - sx2 as f64 > 1e-3 {
                taps.push((fsx2 - sx2 as f64).min(1.0).min(cell) / cell);
            }
            (first, taps)
        })
        .collect()
}

/// Pillow's Lanczos taps (support 3, widened when shrinking), its weights
/// in fixed point of 22 bits as Pillow's 8-bit resampling keeps them.
fn lanczos_taps(from: usize, to: usize) -> Taps {
    fn sinc(x: f64) -> f64 {
        if x == 0.0 {
            1.0
        } else {
            let x = x * std::f64::consts::PI;
            x.sin() / x
        }
    }
    let lanczos = |x: f64| {
        if x.abs() < 3.0 {
            sinc(x) * sinc(x / 3.0)
        } else {
            0.0
        }
    };
    let scale = from as f64 / to as f64;
    let filterscale = scale.max(1.0);
    let support = 3.0 * filterscale;
    (0..to)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = ((center + support + 0.5) as i64).min(from as i64) as usize;
            let mut w: Vec<f64> = (xmin..xmax)
                .map(|x| lanczos((x as f64 - center + 0.5) / filterscale))
                .collect();
            let total: f64 = w.iter().sum();
            for v in &mut w {
                *v /= total;
                // Fixed point, rounded away from zero.
                let k = *v * f64::from(1 << 22);
                *v = if k < 0.0 {
                    (k - 0.5).trunc()
                } else {
                    (k + 0.5).trunc()
                };
            }
            (xmin, w)
        })
        .collect()
}

/// How an axis's sums become bytes.
#[derive(Clone, Copy)]
enum Round {
    /// To nearest, halves to even (OpenCV's `saturate_cast`).
    Even,
    /// Pillow's fixed point: (sum + 2²¹) >> 22.
    Fixed,
}

fn to_byte(v: f64, round: Round) -> u8 {
    match round {
        Round::Even => v.round_ties_even().clamp(0.0, 255.0) as u8,
        Round::Fixed => ((v as i64 + (1 << 21)) >> 22).clamp(0, 255) as u8,
    }
}

/// `image` resampled along x by `xs` then along y by `ys` (each pass to
/// bytes, as Pillow does; OpenCV's area pass sums both axes before it
/// rounds, `both` at once).
fn resample(image: &Rgb, xs: &Taps, ys: &Taps, round: Round, both: bool) -> Rgb {
    let (w, h) = (xs.len(), ys.len());
    let mut wide = vec![0f64; image.height * w * 3];
    for y in 0..image.height {
        for (x, (first, taps)) in xs.iter().enumerate() {
            for c in 0..3 {
                let mut sum = 0.0;
                for (i, k) in taps.iter().enumerate() {
                    sum += k * f64::from(image.data[(y * image.width + first + i) * 3 + c]);
                }
                wide[(y * w + x) * 3 + c] = if both {
                    sum
                } else {
                    f64::from(to_byte(sum, round))
                };
            }
        }
    }
    let mut out = vec![0u8; w * h * 3];
    for (y, (first, taps)) in ys.iter().enumerate() {
        for x in 0..w {
            for c in 0..3 {
                let mut sum = 0.0;
                for (i, k) in taps.iter().enumerate() {
                    sum += k * wide[((first + i) * w + x) * 3 + c];
                }
                out[(y * w + x) * 3 + c] = to_byte(sum, round);
            }
        }
    }
    Rgb::new(w, h, out)
}

/// `image` at `width × height` by OpenCV's area resampling (for shrinking:
/// a side that grows takes the nearest pixel, as OpenCV's area does).
pub fn area(image: &Rgb, width: usize, height: usize) -> Rgb {
    let axis = |from: usize, to: usize| {
        if to <= from {
            area_taps(from, to)
        } else {
            (0..to).map(|i| (i * from / to, vec![1.0])).collect()
        }
    };
    resample(
        image,
        &axis(image.width, width),
        &axis(image.height, height),
        Round::Even,
        true,
    )
}

/// OpenCV's `INTER_CUBIC` taps (A = -0.75, pixel centres aligned, the edge
/// pixel repeated past the edge) from `from` to `to`.
fn cubic_taps(from: usize, to: usize) -> Taps {
    const A: f64 = -0.75;
    let scale = from as f64 / to as f64;
    (0..to)
        .map(|x| {
            let f = (x as f64 + 0.5) * scale - 0.5;
            let sx = f.floor();
            let t = f - sx;
            let w0 = ((A * (t + 1.0) - 5.0 * A) * (t + 1.0) + 8.0 * A) * (t + 1.0) - 4.0 * A;
            let w1 = ((A + 2.0) * t - (A + 3.0)) * t * t + 1.0;
            let w2 = ((A + 2.0) * (1.0 - t) - (A + 3.0)) * (1.0 - t) * (1.0 - t) + 1.0;
            let w = [w0, w1, w2, 1.0 - w0 - w1 - w2];
            // Each tap onto its pixel, those past the edge onto the edge's.
            let at = |k: usize| (sx as i64 - 1 + k as i64).clamp(0, from as i64 - 1) as usize;
            let first = at(0);
            let mut taps = vec![0.0; at(3) - first + 1];
            for (k, wk) in w.iter().enumerate() {
                taps[at(k) - first] += wk;
            }
            (first, taps)
        })
        .collect()
}

/// `image` at `width × height` by OpenCV's bicubic resampling (for
/// enlarging; rounded once at the end, so within a level of OpenCV's fixed
/// point).
pub fn cubic(image: &Rgb, width: usize, height: usize) -> Rgb {
    resample(
        image,
        &cubic_taps(image.width, width),
        &cubic_taps(image.height, height),
        Round::Even,
        true,
    )
}

/// `image` at `width × height` as Depth Anything 3's input processor
/// resizes: bicubic where it grows (either side), else by area.
fn depth_resize(image: &Rgb, width: usize, height: usize) -> Rgb {
    if width > image.width || height > image.height {
        cubic(image, width, height)
    } else {
        area(image, width, height)
    }
}

/// `image` at `width × height` by Pillow's Lanczos.
pub fn lanczos(image: &Rgb, width: usize, height: usize) -> Rgb {
    let xs = lanczos_taps(image.width, width);
    let ys = lanczos_taps(image.height, height);
    resample(image, &xs, &ys, Round::Fixed, false)
}

/// The size Depth Anything 3 takes `width × height` at: its longest side
/// 504, then each side to its nearest multiple of 14.
pub fn depth_size(width: usize, height: usize) -> (usize, usize) {
    let s = 504.0 / width.max(height) as f64;
    let (w, h) = (
        ((width as f64 * s).round() as usize).max(1),
        ((height as f64 * s).round() as usize).max(1),
    );
    let near = |x: usize| {
        let down = x / 14 * 14;
        let up = down + 14;
        (if up - x < x - down { up } else { down }).max(14)
    };
    (near(w), near(h))
}

/// Depth Anything 3's input from `image`: its size ([`depth_size`]) as its
/// processor makes it (the longest side to 504, then each side to a multiple
/// of 14 where that changes it; by area where it shrinks, bicubic where it
/// grows), normalised.
pub fn depth_input(image: &Rgb) -> (Vec<f32>, usize, usize) {
    let s = 504.0 / image.width.max(image.height) as f64;
    let (w, h) = (
        ((image.width as f64 * s).round() as usize).max(1),
        ((image.height as f64 * s).round() as usize).max(1),
    );
    let first = depth_resize(image, w, h);
    let (tw, th) = depth_size(image.width, image.height);
    let last = if (tw, th) == (w, h) {
        first
    } else {
        depth_resize(&first, tw, th)
    };
    (normalise(&last), tw, th)
}

/// BEN2's input from `image`: 1024 × 1024 by Lanczos, normalised.
pub fn matte_input(image: &Rgb) -> Vec<f32> {
    normalise(&lanczos(image, 1024, 1024))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halving_by_area_averages_two_by_two() {
        let image = Rgb::new(2, 2, vec![0, 10, 20, 2, 10, 20, 4, 10, 20, 7, 10, 21]);
        let out = area(&image, 1, 1);
        // (0 + 2 + 4 + 7) / 4 = 3.25 → 3; 20.25 → 20.
        assert_eq!(out.data, vec![3, 10, 20]);
    }

    #[test]
    fn enlarging_bicubically_keeps_flat_flat_and_the_corners() {
        let flat = Rgb::new(5, 3, vec![90; 5 * 3 * 3]);
        assert!(cubic(&flat, 12, 7).data.iter().all(|&v| v == 90));
        // A ramp stays a ramp (rising past its last pixel a little, as
        // OpenCV's bicubic does at an edge it repeats).
        let ramp = Rgb::new(
            4,
            1,
            [0u8, 80, 160, 240].iter().flat_map(|&v| [v; 3]).collect(),
        );
        let up = cubic(&ramp, 8, 1);
        let reds: Vec<u8> = up.data.chunks(3).map(|p| p[0]).collect();
        assert!(reds.windows(2).all(|w| w[0] <= w[1]), "{reds:?}");
        assert_eq!(reds[0], 0);
        assert!((240..=250).contains(&reds[7]), "{reds:?}");
    }

    #[test]
    fn a_flat_picture_stays_flat_through_lanczos() {
        let image = Rgb::new(37, 23, vec![77; 37 * 23 * 3]);
        let out = lanczos(&image, 16, 16);
        assert!(out.data.iter().all(|&v| v == 77));
    }

    #[test]
    fn depth_sizes_are_multiples_of_14_near_504() {
        assert_eq!(depth_size(1280, 1280), (504, 504));
        assert_eq!(depth_size(1000, 1500), (336, 504));
        assert_eq!(depth_size(1500, 600), (504, 196));
    }
}
