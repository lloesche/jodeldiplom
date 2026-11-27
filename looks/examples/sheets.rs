//! Contact sheets of the looks, to judge them by eye: for each family of
//! passes a sheet (`passes-<family>.png`) with a row a pass, each image
//! given in a column three times, the pass subtle, middling and strong
//! (its `amount`, or its first knob, a quarter, three fifths and all the
//! way up its range, the rest at their defaults); and the presets
//! (`presets-<family>.png`), a row each over every image. Each row's name
//! in `<sheet>.txt` beside it. Drawn with the passes' maths on the
//! processor (which the shader check holds the GPU to), at a moment of a
//! made-up song at 120 a minute, the lyric band kept as in the game.
//! The images are read where they are, never copied.
//!
//! ```text
//! cargo run --release -p jodel-looks --example sheets -- OUT_DIR IMAGE...
//! ```

use std::path::{Path, PathBuf};

use jodel_looks::render::Image;
use jodel_looks::{demo, guard, passes, presets, Layer, Look, Moment};

/// A tile's size.
const TILE: (usize, usize) = (320, 180);
/// The gap between tiles.
const GAP: usize = 4;

/// The picture at `path`, cut to the tiles' shape (its middle) and made
/// their size.
fn read(path: &Path) -> Image {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .to_rgb8();
    let (w, h) = img.dimensions();
    let want = TILE.0 as f32 / TILE.1 as f32;
    let (cw, ch) = if w as f32 / h as f32 > want {
        ((h as f32 * want) as u32, h)
    } else {
        (w, (w as f32 / want) as u32)
    };
    let cut = image::imageops::crop_imm(&img, (w - cw) / 2, (h - ch) / 2, cw, ch).to_image();
    Image::from_rgb(cw as usize, ch as usize, cut.as_raw()).resized(TILE.0, TILE.1)
}

/// `look` drawn over `img` at `t` seconds into the made-up song.
fn drawn(look: &Look, img: &Image, t: f64) -> Image {
    let checked = look.check();
    let curves = demo::Song::new(60.0, 120.0).curves();
    let moment = Moment {
        size: (img.w as u32, img.h as u32),
        t,
        seed: jodel_looks::seed(&look.id, "sheets"),
        dt: 1.0 / 30.0,
        band: Some(jodel_looks::BAND),
        ..Moment::MEASURING
    };
    let signals = curves.at(t);
    let mut prev = None;
    let mut out = img.clone();
    // A few frames, so the feedback has an echo to show.
    for f in 0..4 {
        let m = Moment {
            t: t + f64::from(f) / 30.0,
            forget: f == 0,
            ..moment
        };
        out = jodel_looks::render::draw(
            &checked.frame(&m, &signals, Some(&curves)),
            img,
            prev.as_ref(),
        );
        prev = Some(out.clone());
    }
    out
}

/// A sheet of `rows` of tiles, written as a PNG to `file` with the rows'
/// names beside it.
fn sheet(file: &Path, rows: &[(String, Vec<Image>)]) {
    let cols = rows.iter().map(|(_, r)| r.len()).max().unwrap_or(0);
    let (w, h) = (
        cols * (TILE.0 + GAP) + GAP,
        rows.len() * (TILE.1 + GAP) + GAP,
    );
    let mut px = vec![24u8; w * h * 3];
    for (r, (_, tiles)) in rows.iter().enumerate() {
        for (c, tile) in tiles.iter().enumerate() {
            let (x0, y0) = (GAP + c * (TILE.0 + GAP), GAP + r * (TILE.1 + GAP));
            let rgb = tile.to_rgb();
            for y in 0..TILE.1 {
                let from = &rgb[y * TILE.0 * 3..(y + 1) * TILE.0 * 3];
                let at = ((y0 + y) * w + x0) * 3;
                px[at..at + TILE.0 * 3].copy_from_slice(from);
            }
        }
    }
    let img = image::RgbImage::from_raw(w as u32, h as u32, px).expect("a sheet");
    let mut bytes = std::io::Cursor::new(Vec::new());
    img.write_to(&mut bytes, image::ImageFormat::Png)
        .expect("a PNG");
    durable::write_atomic(file, bytes.get_ref()).expect("written");
    let names: String = rows
        .iter()
        .enumerate()
        .map(|(i, (n, _))| format!("{}: {n}\n", i + 1))
        .collect();
    durable::write_atomic(&file.with_extension("txt"), names.as_bytes()).expect("written");
    println!("{}", file.display());
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = PathBuf::from(args.first().expect("OUT_DIR"));
    let images: Vec<Image> = args[1..].iter().map(|p| read(Path::new(p))).collect();
    assert!(!images.is_empty(), "IMAGE...");
    let t = 21.0;
    for family in jodel_looks::Family::ALL {
        let mut rows = Vec::new();
        for p in passes::PASSES.iter().filter(|p| p.family == family) {
            let strength = p.knob("amount").or_else(|| p.knobs.first().map(|k| (0, k)));
            let mut tiles = Vec::new();
            for img in &images {
                for share in [0.25, 0.6, 1.0] {
                    let mut layer = Layer::of(p.name);
                    if let Some((_, k)) = strength {
                        layer
                            .knobs
                            .insert(k.name.to_owned(), k.lo + (k.hi - k.lo) * share);
                    }
                    let look = Look {
                        id: p.name.to_owned(),
                        name: String::new(),
                        about: String::new(),
                        tags: Vec::new(),
                        layers: vec![layer],
                    };
                    tiles.push(drawn(&look, img, t));
                }
            }
            rows.push((p.name.to_owned(), tiles));
        }
        sheet(&out.join(format!("passes-{}.png", family.name())), &rows);
    }
    for f in presets::families() {
        let rows: Vec<(String, Vec<Image>)> = f
            .looks
            .iter()
            .map(|l| {
                let tiles: Vec<Image> = images.iter().map(|img| drawn(l, img, t)).collect();
                let lights: Vec<String> = tiles
                    .iter()
                    .map(|i| format!("{:.2}", guard::light(&i.px)))
                    .collect();
                (format!("{} (light {})", l.id, lights.join(" ")), tiles)
            })
            .collect();
        sheet(&out.join(format!("presets-{}.png", f.family)), &rows);
    }
}
