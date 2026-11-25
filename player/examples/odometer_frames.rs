//! Karaoke mode's score rolling up, frame by frame, as the window paints
//! it without a GPU (over a sample cover's blur): a word's points at ×4
//! (12 400 to 12 800) and at ×32 (to 15 600), and a jackpot (to 186 000),
//! each at several moments into its roll. Saved as PNGs into `OUT_DIR`: the top right
//! corner five times as large (`<roll>-<ms>.png`) and the whole frame
//! mid-jackpot (`jackpot-whole.png`).
//!
//! ```text
//! cargo run --release -p jodeldiplom --example odometer_frames -- OUT_DIR
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use jodeldiplom::art::{sample_cover, Art};
use jodeldiplom::gui::big_text::Faces;
use jodeldiplom::gui::karaoke::{paint, Backdrop, Layer};
use jodeldiplom::gui::raster::{unpack, Target};
use jodeldiplom::view::karaoke::{KaraokePicture, Score};
use ratatui::layout::Rect;

// The fonts build.rs found: `EMBEDDED_FONTS`.
include!(concat!(env!("OUT_DIR"), "/fonts.rs"));

const W: usize = 1600;
const H: usize = 900;
/// A frame, in seconds.
const FRAME: f32 = 1.0 / 60.0;
/// The moments shown, in milliseconds after the points came.
const MOMENTS: [u32; 9] = [0, 33, 83, 150, 250, 350, 450, 600, 900];
/// The corner shown: its size, and how much larger it is saved.
const CORNER: (usize, usize) = (330, 110);
const ZOOM: usize = 5;

/// The picture with the score `points` (its multiplier under it) and no
/// lines.
fn picture(points: u32) -> KaraokePicture {
    KaraokePicture {
        area: Rect::new(0, 0, 100, 30),
        bar: Rect::new(0, 27, 100, 3),
        lines: Vec::new(),
        count: None,
        cue: None,
        controls: None,
        start: None,
        before: Vec::new(),
        moved: 1.0,
        score: Some(Score { points, streak: 25 }),
        result: None,
    }
}

fn save(file: &Path, img: &image::RgbImage) {
    let mut png = std::io::Cursor::new(Vec::new());
    img.write_to(&mut png, image::ImageFormat::Png)
        .expect("encoded");
    durable::write_atomic(file, png.get_ref()).expect("written");
    println!("{}", file.display());
}

fn main() {
    jodeldiplom::gui::fonts::use_fonts(EMBEDDED_FONTS);
    let out = PathBuf::from(std::env::args().nth(1).expect("OUT_DIR"));
    let art = Art {
        id: 1,
        image: Arc::new(sample_cover(256)),
    };
    let backdrop = Backdrop::new(Some(&art), W, H);
    let rolls = [
        ("word", 12_400, 12_800),
        ("x32", 12_400, 15_600),
        ("jackpot", 12_400, 186_000),
    ];
    for (roll, from, to) in rolls {
        let mut faces = Faces::default();
        let mut layer = Layer::default();
        let mut pixels = vec![0u32; W * H];
        // Each frame painted as the window paints it, at its time.
        let t0 = Instant::now();
        let mut frame = |layer: &mut Layer, pixels: &mut [u32], f: usize, points: u32| {
            layer.at(t0 + Duration::from_secs_f32(f as f32 * FRAME));
            let mut target = Target {
                pixels,
                width: W,
                height: H,
            };
            let area = (0, 0, W, H);
            paint(
                &mut target,
                &mut faces,
                layer,
                &backdrop,
                &picture(points),
                area,
                None,
                true,
            );
        };
        // The points before, then frame by frame from the one they came in.
        frame(&mut layer, &mut pixels, 0, from);
        let mut f = 0;
        for &moment in &MOMENTS {
            let due = 1 + (moment as f32 / 1000.0 / FRAME).round() as usize;
            while f < due {
                f += 1;
                frame(&mut layer, &mut pixels, f, to);
            }
            let (cw, ch) = CORNER;
            let x0 = W - cw;
            let corner =
                image::RgbImage::from_fn((cw * ZOOM) as u32, (ch * ZOOM) as u32, |x, y| {
                    let (x, y) = (x as usize / ZOOM, y as usize / ZOOM);
                    image::Rgb(unpack(pixels[y * W + x0 + x]))
                });
            save(&out.join(format!("{roll}-{moment:03}.png")), &corner);
            if roll == "jackpot" && moment == 150 {
                let whole = image::RgbImage::from_fn(W as u32, H as u32, |x, y| {
                    image::Rgb(unpack(pixels[y as usize * W + x as usize]))
                });
                save(&out.join("jackpot-whole.png"), &whole);
            }
        }
    }
}
