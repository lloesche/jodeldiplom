//! A picture's main colours ([`palette`]).

use image::RgbImage;

/// A colour of a picture's palette and its share of the picture.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Swatch {
    /// The colour.
    pub rgb: [u8; 3],
    /// Its share of the picture's pixels, 0..=1.
    pub share: f32,
}

/// A picture's main colours: k-means of five over its pixels at 64 × 64,
/// started from colours far apart, the largest share first.
pub fn palette(image: &RgbImage) -> Vec<Swatch> {
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
