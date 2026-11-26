//! Karaoke mode's score rolling up, frame by frame, as the GPU draws it
//! (the stage over a sample cover's blur): a word's points at ×4
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

use jodeldiplom::art::{sample_cover, Art};
use jodeldiplom::gui::backdrop::{Cue, Visualiser};
use jodeldiplom::gui::pixels::unpack;
use jodeldiplom::gui::present::Offscreen;
use jodeldiplom::gui::stage::{Song, Stage, Tint};
use jodeldiplom::view::karaoke::{KaraokePicture, Score};

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
        bar: 3.0 / 30.0,
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
    let (w, h) = (W as u32, H as u32);
    let mut gpu = Offscreen::new(false).expect("a GPU");
    let cue = Cue {
        art: Some(art),
        layers: None,
        heard: Default::default(),
    };
    let ground = Visualiser::default().step(&cue, (w, h), 0.0);
    gpu.set_backdrop(Some(&ground), w, h);
    let singing = jodeldiplom::core::singing::Singing::default();
    let song = Song::new(&singing, Vec::new(), None);
    let area = (0.0, 0.0, W as f32, H as f32);
    let rolls = [
        ("word", 12_400, 12_800),
        ("x32", 12_400, 15_600),
        ("jackpot", 12_400, 186_000),
    ];
    for (roll, from, to) in rolls {
        let mut stage = Stage::default();
        // Each frame stepped as the window steps it, a frame apart.
        let frame = |stage: &mut Stage, points: u32| {
            stage.scene(
                (&song, false),
                &picture(points),
                area,
                (0.0, FRAME),
                &Tint::STAGE,
            )
        };
        // The points before, then frame by frame from the one they came in.
        let mut last = frame(&mut stage, from);
        let mut f = 0;
        for &moment in &MOMENTS {
            let due = 1 + (moment as f32 / 1000.0 / FRAME).round() as usize;
            while f < due {
                f += 1;
                last = frame(&mut stage, to);
            }
            gpu.set_stage(last.as_ref(), w, h);
            let pixels = gpu.show_frame(w, h).expect("a frame");
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
