//! Frames of Karaoke mode's result screen (the diploma headed
//! "Jodeldiplom") on the GPU, to look at its reveal: a Silver run short
//! of its best, a Bronze one with a long Japanese title, a Gold run with
//! a new best, long numbers and several phrases at the top tier (each
//! over a made-up picture, a story of one scene, the diploma tinted by
//! its palette), a run without a medal (over the plain ground), a run
//! not sung from its start (over another picture) and a
//! silent one (nothing scored), each at moments of its reveal; in Karaoke
//! mode's own fonts (as the player embeds them), the stage drawn offscreen
//! and the frames saved as PNGs into `OUT_DIR` (`<run>-<seconds>.png`,
//! written whole).
//!
//! ```text
//! cargo run --release -p jodeldiplom --example result_frames -- OUT_DIR [W H] [--at SECONDS ...]
//! ```

use std::sync::Arc;

use jodeldiplom::core::karaoke::result::{Medal, Shown, SongResult};
use jodeldiplom::core::shine::{Shone, Tier};
use jodeldiplom::core::singing::Singing;
use jodeldiplom::gui::backdrop::{Cue, Shows, Visualiser};
use jodeldiplom::gui::present::Offscreen;
use jodeldiplom::gui::stage::{Song, Stage, Tint};
use jodeldiplom::view::karaoke::{plain_lines, KaraokePicture, Next, ResultShown};

#[path = "../tests/common/sung.rs"]
mod sung;

// The fonts build.rs found: `EMBEDDED_FONTS`.
include!(concat!(env!("OUT_DIR"), "/fonts.rs"));

/// The moments looked at when none are asked for (seconds into the
/// reveal); and well after its end.
const LOOKS: [f64; 9] = [0.3, 1.2, 2.0, 2.9, 3.6, 4.4, 4.7, 5.6, 30.0];

/// A run's phrases, by tier: its row's place among the tiers (`0` the
/// first), `.` none; a digit past the top's is the top.
fn phrases(tiers: &str) -> Vec<Shone> {
    tiers
        .chars()
        .enumerate()
        .map(|(line, c)| Shone {
            line,
            end: line as f64 * 4.0,
            value: 0.8,
            top: 0.9,
            tier: c
                .to_digit(10)
                .map(|i| Tier::ALL[(i as usize).min(Tier::COUNT - 1)]),
        })
        .collect()
}

/// A made-up picture: a gradient of two colours with a disc of a third.
fn cover(a: [u8; 3], b: [u8; 3], c: [u8; 3]) -> image::RgbImage {
    image::RgbImage::from_fn(256, 256, |x, y| {
        let t = (x + y) as f32 / 510.0;
        let (dx, dy) = (x as f32 - 150.0, y as f32 - 110.0);
        if dx * dx + dy * dy < 60.0 * 60.0 {
            return image::Rgb(c);
        }
        image::Rgb([0, 1, 2].map(|k| (f32::from(a[k]) * (1.0 - t) + f32::from(b[k]) * t) as u8))
    })
}

/// The runs: a name, the result, and its picture.
fn runs() -> Vec<(&'static str, SongResult, Option<image::RgbImage>)> {
    let gold = SongResult {
        title: "Dancing in the Moonlight".into(),
        points: 486_350,
        hit: 187,
        words: 243,
        streak: 112,
        phrases: phrases("413042413.0432114032"),
        medal: Some(Medal::Gold),
        best: Some(98_500),
        whole: true,
        demo: false,
    };
    let none = SongResult {
        title: "Bohemian Rhapsody".into(),
        points: 2_150,
        hit: 17,
        words: 52,
        streak: 6,
        phrases: phrases("..0...0..0.."),
        medal: None,
        best: Some(3_400),
        whole: true,
        demo: false,
    };
    let part = SongResult {
        title: "こんにちは世界".into(),
        points: 5_300,
        hit: 22,
        words: 26,
        streak: 14,
        phrases: phrases("1301"),
        medal: None,
        best: None,
        whole: false,
        demo: false,
    };
    let silver = SongResult {
        title: "Take On Me".into(),
        points: 7_820,
        hit: 34,
        words: 45,
        streak: 11,
        phrases: phrases("10321.0103"),
        medal: Some(Medal::Silver),
        best: Some(8_400),
        whole: true,
        demo: false,
    };
    let long = SongResult {
        title: "君の名は。〜 夢灯籠 〜 前前前世 (movie ver.) スパークル 〜 なんでもないや".into(),
        points: 4_210,
        hit: 25,
        words: 45,
        streak: 9,
        phrases: phrases("01.030"),
        medal: Some(Medal::Bronze),
        best: None,
        whole: true,
        demo: false,
    };
    vec![
        (
            "silver",
            silver,
            Some(cover([120, 30, 30], [200, 120, 40], [250, 230, 200])),
        ),
        (
            "long-cjk",
            long,
            Some(cover([10, 30, 80], [60, 20, 90], [250, 180, 90])),
        ),
        (
            "gold",
            gold,
            Some(cover([40, 20, 90], [10, 70, 110], [230, 90, 160])),
        ),
        ("no-medal", none, None),
        (
            "silent",
            SongResult {
                title: "Stille Nacht".into(),
                points: 0,
                hit: 0,
                words: 0,
                streak: 0,
                phrases: Vec::new(),
                medal: None,
                best: Some(4_200),
                whole: true,
                demo: false,
            },
            Some(cover([30, 30, 70], [70, 50, 110], [200, 190, 240])),
        ),
        (
            "not-whole",
            part,
            Some(cover([20, 70, 40], [90, 120, 30], [240, 200, 60])),
        ),
    ]
}

fn main() {
    jodeldiplom::gui::fonts::use_fonts(EMBEDDED_FONTS);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = std::path::PathBuf::from(args.first().expect("OUT_DIR"));
    let (w, h) = match (
        args.get(1).and_then(|a| a.parse().ok()),
        args.get(2).and_then(|a| a.parse().ok()),
    ) {
        (Some(w), Some(h)) => (w, h),
        _ => (1600usize, 900usize),
    };
    let mut wanted: Vec<f64> = args
        .iter()
        .skip_while(|a| *a != "--at")
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    if wanted.is_empty() {
        wanted = LOOKS.to_vec();
    }
    let mut gpu = Offscreen::new(false).expect("a GPU");
    for (id, (name, result, art)) in (1..).zip(runs()) {
        let next = "Take On Me";
        let lines = plain_lines(&result, next);
        let shown = Arc::new(Shown::new(result, id, 0));
        let mut stage = Stage::default();
        let singing = Singing::default();
        // Over its picture as a story of one scene (none: the plain
        // ground), as the window's backdrop.
        let cue = Cue {
            shows: Shows::Story(art.as_ref().map(sung::one_scene)),
            ..Cue::default()
        };
        let ground = Visualiser::default().step(&cue, (w as u32, h as u32), 0.0);
        // Tinted as the app tints it: by the scene showing's palette.
        let tint = Tint::of(ground.palette());
        gpu.set_backdrop(Some(&ground), w as u32, h as u32);
        let end = shown.reveal.end();
        println!(
            "{name}: ends at {end:.2} s, beats {:?}",
            shown.reveal.beats()
        );
        for &t in &wanted {
            let picture = KaraokePicture {
                bar: 5.0 / 30.0,
                reach: 0.15,
                lines: lines.clone(),
                count: None,
                cue: None,
                controls: None,
                next: Some(Next { ready: true }),
                before: Vec::new(),
                moved: 1.0,
                score: None,
                result: Some(ResultShown {
                    shown: shown.clone(),
                    t,
                    next: next.into(),
                    again: false,
                }),
            };
            let area = (0.0, 0.0, w as f32, h as f32);
            let song = Song::new(&singing, Vec::new(), None);
            let frame = stage.scene((&song, false), &picture, area, (0.0, 0.0), &tint);
            let frame = frame.expect("a result");
            gpu.set_stage(Some(&frame), w as u32, h as u32);
            let shown_px = gpu.show_frame(w as u32, h as u32).expect("a frame");
            let mut img = image::RgbImage::new(w as u32, h as u32);
            for (i, p) in shown_px.iter().enumerate() {
                img.put_pixel(
                    (i % w) as u32,
                    (i / w) as u32,
                    image::Rgb([(p >> 16) as u8, (p >> 8) as u8, *p as u8]),
                );
            }
            let file = out.join(format!("{name}-{t:05.2}.png"));
            let mut png = std::io::Cursor::new(Vec::new());
            img.write_to(&mut png, image::ImageFormat::Png).unwrap();
            durable::write_atomic(&file, png.get_ref()).unwrap();
            println!("{}", file.display());
        }
    }
}
