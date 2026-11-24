//! Covers taken apart, to look at: `cover_layers OUT COVER...` takes each
//! cover image apart on the GPU as the player does (decoded and shrunk to
//! 1024 at most, the depth and cut-out models, then `jodeldiplom::cover`)
//! and writes to `OUT`:
//!
//! - `NAME.sheet.png`: the cover, its depth (near bright), the subject over
//!   a checkerboard, the plate, the rim and the palette; under them five
//!   frames of the layers moved as the background might: the plate and the
//!   subject shifted by their depth one way and the other (parallax), the
//!   subject faded to 30 %, moved far off, and 10 % larger, each over the
//!   plate.
//! - `NAME.plate.png`, `NAME.hole.png` (where the plate is fill) and
//!   `NAME.cover.png` (the cover as taken apart), for comparing fills.
//!
//! Times each cover's models and taking apart. The weights are found as the
//! player finds them.

use std::path::{Path, PathBuf};
use std::time::Instant;

use image::{GrayImage, Rgb, RgbImage, RgbaImage};
use jodeldiplom::cover::{self, CoverLayers, GpuSplitter, NearMap};

/// Nearness as a heat colour (dark far, bright near).
fn heat(n: &NearMap, w: u32, h: u32) -> RgbImage {
    RgbImage::from_fn(w, h, |x, y| {
        let v = n.at((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
        Rgb([
            (255.0 * (v * 1.6).min(1.0)) as u8,
            (255.0 * ((v - 0.35) * 1.6).clamp(0.0, 1.0)) as u8,
            (255.0 * ((v - 0.7) * 3.0).clamp(0.0, 1.0)) as u8,
        ])
    })
}

fn over_checker(s: &RgbaImage) -> RgbImage {
    RgbImage::from_fn(s.width(), s.height(), |x, y| {
        let p = s.get_pixel(x, y).0;
        let a = f32::from(p[3]) / 255.0;
        let bg = if (x / 32 + y / 32) % 2 == 0 {
            200.0
        } else {
            120.0
        };
        Rgb([0, 1, 2].map(|c| (f32::from(p[c]) * a + bg * (1.0 - a)) as u8))
    })
}

fn gray_rgb(g: &GrayImage) -> RgbImage {
    RgbImage::from_fn(g.width(), g.height(), |x, y| {
        let v = g.get_pixel(x, y).0[0];
        Rgb([v, v, v])
    })
}

fn palette_strip(l: &CoverLayers, w: u32, h: u32) -> RgbImage {
    let mut out = RgbImage::new(w, h);
    let mut x0 = 0.0;
    for s in &l.palette {
        let x1 = x0 + s.share * w as f32;
        for x in x0 as u32..(x1 as u32).min(w) {
            for y in 0..h {
                out.put_pixel(x, y, Rgb(s.rgb));
            }
        }
        x0 = x1;
    }
    out
}

/// A frame: the plate shifted by its depth (`shift` pixels at the
/// nearest), the subject shifted by `subject_shift` and scaled by `scale`
/// about its middle, at `opacity`.
fn frame(
    l: &CoverLayers,
    shift: f32,
    subject_shift: (f32, f32),
    scale: f32,
    opacity: f32,
) -> RgbImage {
    let (w, h) = l.size();
    let mut out = RgbImage::from_fn(w, h, |x, y| {
        let n = l
            .plate_depth
            .at((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
        let sx = (x as f32 - shift * n).clamp(0.0, (w - 1) as f32) as u32;
        *l.plate.get_pixel(sx, y)
    });
    if let Some(s) = &l.subject {
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        for y in 0..h {
            for x in 0..w {
                let sx = (x as f32 - subject_shift.0 - cx) / scale + cx;
                let sy = (y as f32 - subject_shift.1 - cy) / scale + cy;
                if sx < 0.0 || sy < 0.0 || sx >= w as f32 || sy >= h as f32 {
                    continue;
                }
                let p = s.get_pixel(sx as u32, sy as u32).0;
                let a = f32::from(p[3]) / 255.0 * opacity;
                let q = out.get_pixel_mut(x, y);
                for c in 0..3 {
                    q.0[c] = (f32::from(p[c]) * a + f32::from(q.0[c]) * (1.0 - a)) as u8;
                }
            }
        }
    }
    out
}

fn sheet(cover: &RgbImage, l: &CoverLayers) -> RgbImage {
    let t = 384;
    let fit =
        |i: &RgbImage| image::imageops::resize(i, t, t, image::imageops::FilterType::Triangle);
    let (w, h) = l.size();
    let subject = l
        .subject
        .as_ref()
        .map_or_else(|| RgbImage::new(w, h), over_checker);
    let near = l.subject.as_ref().map_or(0.6, |_| 0.9);
    let top = [
        fit(cover),
        fit(&heat(&l.depth, w, h)),
        fit(&subject),
        fit(&l.plate),
        fit(&gray_rgb(&l.outline)),
        palette_strip(l, t, t),
    ];
    let big = w as f32 * 0.04;
    let bottom = [
        fit(&frame(l, -big, (-big * near, 0.0), 1.0, 1.0)),
        fit(&frame(l, big, (big * near, 0.0), 1.0, 1.0)),
        fit(&frame(l, 0.0, (0.0, 0.0), 1.0, 0.3)),
        fit(&frame(l, 0.0, (w as f32 * 0.45, h as f32 * 0.1), 1.0, 1.0)),
        fit(&frame(l, 0.0, (0.0, 0.0), 1.1, 1.0)),
    ];
    let mut out = RgbImage::from_pixel(6 * t, 2 * t, Rgb([20, 20, 20]));
    for (i, img) in top.iter().enumerate() {
        image::imageops::replace(&mut out, img, i as i64 * t as i64, 0);
    }
    for (i, img) in bottom.iter().enumerate() {
        image::imageops::replace(&mut out, img, i as i64 * t as i64, t as i64);
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((out, covers)) = args.split_first() else {
        eprintln!("usage: cover_layers OUT COVER...");
        return;
    };
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out).unwrap();
    let models = jodeldiplom::dirs::Data::current()
        .expect("a data folder")
        .models();
    let splitter = GpuSplitter::new(cover::CoverModels {
        depth: models.join(cover::DEPTH_FILE),
        matte: models.join(cover::MATTE_FILE),
    });
    for path in covers {
        let path = Path::new(path);
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let Some(image) = jodeldiplom::art::decode(&std::fs::read(path).unwrap()) else {
            eprintln!("{name}: not an image");
            continue;
        };
        let started = Instant::now();
        let parts = match splitter.parts(&image) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{name}: {e}");
                return;
            }
        };
        let models = started.elapsed();
        let started = Instant::now();
        let l = cover::split(&image, &parts);
        let apart = started.elapsed();
        println!(
            "{name}: models {} ms, taking apart {} ms, subject {}",
            models.as_millis(),
            apart.as_millis(),
            if l.subject.is_some() { "yes" } else { "no" }
        );
        sheet(&image, &l)
            .save(out.join(format!("{name}.sheet.png")))
            .unwrap();
        l.plate.save(out.join(format!("{name}.plate.png"))).unwrap();
        image.save(out.join(format!("{name}.cover.png"))).unwrap();
        if let Some(s) = &l.subject {
            let (w, h) = (s.width() as usize, s.height() as usize);
            let alpha: Vec<f32> = s.pixels().map(|p| f32::from(p[3]) / 255.0).collect();
            let hole = cover::hole(&alpha, w, h);
            GrayImage::from_fn(w as u32, h as u32, |x, y| {
                image::Luma([(hole[y as usize * w + x as usize] * 255.0) as u8])
            })
            .save(out.join(format!("{name}.hole.png")))
            .unwrap();
        }
    }
}
