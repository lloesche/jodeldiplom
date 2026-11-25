//! The effects no tier's row uses yet ([`jodeldiplom::gui::fx`]: the
//! screen's edges on fire, the marquee bulbs round the lane, the line sung
//! on fire and frosted, the coins' fountain, rain and flick) seen and
//! timed over a frame of Karaoke mode as the window draws it: a made-up
//! song sung on its notes up to ten seconds in, over a cover taken apart
//! (its `.layers` file), in a window of 3024 × 1964 (a 16-inch MacBook
//! Pro's screen) by default. Each effect is put into the frame as the
//! stage puts it (by the effects' board, from a row listing it: its screen
//! pass, its text in its treatment, its bursts), at a few strengths and
//! moments. Its GPU time is what it adds to the window pass's step it
//! draws in (`fx`, `stage` or `bursts`), the median over a few frames.
//! `--png OUT_DIR`: each frame whole as a PNG (`<effect>.png`).
//!
//! ```text
//! cargo run --release -p jodeldiplom --example fx_frames --
//!     COVER.layers [--size WxH] [--density D] [--png OUT_DIR]
//! ```

use std::path::Path;
use std::sync::Arc;

use jodeldiplom::art::Art;
use jodeldiplom::core::singing::Singing;
use jodeldiplom::cover::{Apart, Layers};
use jodeldiplom::gui::backdrop::{BackdropFrame, Cue, Visualiser};
use jodeldiplom::gui::big_text::Faces;
use jodeldiplom::gui::fx::{Effects, Fx, Use};
use jodeldiplom::gui::karaoke::{paint, Backdrop, Layer};
use jodeldiplom::gui::present::Offscreen;
use jodeldiplom::gui::raster::Target;
use jodeldiplom::gui::stage::{without_lyrics, Song, Stage, Tint};
use jodeldiplom::gui::stage_gpu::StageFrame;

#[path = "../tests/common/sung.rs"]
mod sung;
use sung::{cover_of, levels, picture_at, song, with_effects, FPS, LENGTH};

// The fonts build.rs found: `EMBEDDED_FONTS`.
include!(concat!(env!("OUT_DIR"), "/fonts.rs"));

/// The moment the effects are drawn over.
const AT: f64 = 10.0;
/// Each frame timed this many times.
const TIMES: usize = 7;
/// The frame at [`AT`]: the song sung on its notes up to it, frame by
/// frame as the window draws it (into `gpu`); its backdrop, its stage and
/// the pixmap under them.
fn frame_at(
    gpu: &mut Offscreen,
    (art, layers): (&Art, &Layers),
    (w, h): (u32, u32),
) -> (BackdropFrame, StageFrame, Vec<u32>) {
    let (lyrics, melody) = song();
    let sung: Vec<&jodel_lyrics::Line> = lyrics.lines.iter().collect();
    let mut singing = Singing::default();
    singing.prepare(
        Path::new("song"),
        &lyrics,
        &jodeldiplom::core::singing::melodic(&melody),
    );
    let mut stage = Stage::default();
    let mut visualiser = Visualiser::default();
    let (mut faces, mut layer) = (Faces::default(), Layer::default());
    let under = Backdrop::through(w as usize, h as usize);
    let mut pixels = vec![0u32; (w * h) as usize];
    let mut heard = 0usize;
    let mut last = None;
    for f in 0.. {
        let now = f as f64 / FPS;
        if now > AT + 1e-9 {
            break;
        }
        while (heard as f64) / 100.0 <= now {
            let t = heard as f64 / 100.0;
            let c = singing.word_at(t).map(|(_, c)| c as f32);
            singing.hear(t, c);
            heard += 1;
        }
        let cue = Cue {
            art: Some(art.clone()),
            layers: Some(layers.clone()),
            levels: levels(),
            glow: jodeldiplom::gui::streak::level(&singing, true),
            section: false,
        };
        let backdrop = visualiser.step(&cue, (w, h), (1.0 / FPS) as f32);
        let picture = picture_at(&sung, now, &singing);
        let song = Song::new(&singing, sung.clone(), Some(LENGTH));
        let area = (0.0, 0.0, w as f32, h as f32);
        let frame = stage
            .scene(
                (&song, true),
                &picture,
                area,
                (now, (1.0 / FPS) as f32),
                &Tint::STAGE,
            )
            .expect("a stage");
        let mut target = Target {
            pixels: &mut pixels,
            width: w as usize,
            height: h as usize,
        };
        paint(
            &mut target,
            &mut faces,
            &mut layer,
            &under,
            &without_lyrics(&picture),
            (0, 0, w as usize, h as usize),
            None,
            f == 0,
        );
        gpu.set_backdrop(Some(&backdrop), w, h);
        if f == 0 {
            gpu.show_pixels(&pixels, w, h).expect("a frame");
        }
        gpu.set_stage(Some(&frame), w, h);
        last = Some((backdrop, frame));
    }
    let (backdrop, frame) = last.expect("a frame");
    (backdrop, frame, pixels)
}

/// An effect put into a frame: its name as shown, the effect, a row
/// holding it at its strength (and firing it), and how long ago it fired
/// (bursts; none: held, the word hit then).
struct Shot {
    name: String,
    fx: Fx,
    row: Effects,
    ago: f64,
}

impl Shot {
    fn new(fx: Fx, strength: f32, ago: f64) -> Shot {
        let uses: &'static [Use] = Vec::leak(vec![Use::of(fx, strength)]);
        Shot {
            name: format!("{}-{strength}-{ago}", fx.spec().ident),
            fx,
            row: Effects {
                held: uses,
                on_phrase: uses,
                ..Effects::NONE
            },
            ago,
        }
    }

    /// The window pass's step the effect draws in.
    fn step(&self) -> &'static str {
        let spec = self.fx.spec();
        if spec.is_screen() {
            "fx"
        } else if spec.lyric.is_some() {
            "stage"
        } else {
            "bursts"
        }
    }

    /// `base` with the effect in it, over the stage in `area`.
    fn into(&self, base: &StageFrame, area: (f32, f32, f32, f32)) -> StageFrame {
        with_effects(base, area, &self.row, f64::from(base.clock) - self.ago)
    }
}
/// The median of `v`.
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

/// `frame` timed [`TIMES`] times, each just after `base` (the frame
/// without the effect): its `step`'s median, and the median of what the
/// effect adds to it, ms.
fn cost(
    gpu: &mut Offscreen,
    (backdrop, base): (&BackdropFrame, &StageFrame),
    frame: &StageFrame,
    (size, step): ((u32, u32), &str),
) -> (f64, f64) {
    for _ in 0..TIMES {
        for f in [base, frame] {
            gpu.time_frame(Some(backdrop), Some(f), size.0, size.1)
                .expect("timed");
        }
    }
    let costs = gpu.costs().expect("the times");
    let at = |f: &Vec<(&str, f64)>| f.iter().find(|s| s.0 == step).map_or(0.0, |s| s.1);
    let pairs: Vec<(f64, f64)> = (costs.as_chunks::<2>().0.iter())
        .map(|p| (at(&p[0]), at(&p[1])))
        .collect();
    (
        median(pairs.iter().map(|p| p.1).collect()),
        median(pairs.iter().map(|p| p.1 - p.0).collect()),
    )
}

/// `pixels` (`0x00RRGGBB`, `w` × `h`) as a PNG at `file`.
fn write_png(file: &Path, pixels: &[u32], (w, h): (u32, u32)) {
    let mut img = image::RgbImage::new(w, h);
    for (i, p) in pixels.iter().enumerate() {
        img.put_pixel(
            i as u32 % w,
            i as u32 / w,
            image::Rgb([(p >> 16) as u8, (p >> 8) as u8, *p as u8]),
        );
    }
    let mut bytes = std::io::Cursor::new(Vec::new());
    img.write_to(&mut bytes, image::ImageFormat::Png)
        .expect("a PNG");
    durable::write_atomic(file, bytes.get_ref()).expect("written");
}

fn main() {
    jodeldiplom::gui::fonts::use_fonts(EMBEDDED_FONTS);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cover = args.first().expect("COVER.layers");
    let after = |flag: &str| args.iter().skip_while(|a| *a != flag).nth(1);
    let (w, h) = after("--size").map_or((3024, 1964), |s| {
        let (w, h) = s.split_once('x').expect("WxH");
        (w.parse().expect("W"), h.parse().expect("H"))
    });
    let png = after("--png").map(std::path::PathBuf::from);
    let density: f32 = after("--density").map_or(2.0, |d| d.parse().expect("D"));
    let bytes = std::fs::read(cover).expect("the layers");
    let layers = jodeldiplom::cover::from_bytes(&bytes)
        .expect("layers")
        .layers;
    let art = Art {
        id: 1,
        image: Arc::new(cover_of(&layers)),
    };
    let key = Path::new(cover)
        .file_stem()
        .map_or(String::new(), |s| s.to_string_lossy().into_owned());
    let layers: Layers = Arc::new(Apart::new(key, layers));
    let mut gpu = Offscreen::timed().expect("a GPU that times");
    gpu.set_density(density);
    println!("{}, {w}x{h}, {density} pixels to a point", gpu.name);
    let (backdrop, base, pixmap) = frame_at(&mut gpu, (&art, &layers), (w, h));
    let area = (0.0, 0.0, w as f32, h as f32);
    let shots = [
        Shot::new(Fx::EdgeFire, 0.3, 0.0),
        Shot::new(Fx::EdgeFire, 1.0, 0.0),
        Shot::new(Fx::Bulbs, 0.3, 0.0),
        Shot::new(Fx::Bulbs, 1.0, 0.0),
        Shot::new(Fx::LyricFire, 0.3, 0.0),
        Shot::new(Fx::LyricFire, 1.0, 0.0),
        Shot::new(Fx::LyricFrost, 0.6, 0.0),
        Shot::new(Fx::LyricFrost, 1.0, 0.0),
        Shot::new(Fx::Fountain, 1.0, 0.5),
        Shot::new(Fx::Rain, 1.0, 1.0),
        Shot::new(Fx::Flick, 1.0, 0.15),
    ];
    println!("GPU ms, median: the window pass's step the effect draws in (what it adds)");
    for shot in &shots {
        let frame = shot.into(&base, area);
        let step = shot.step();
        let (ms, added) = cost(&mut gpu, (&backdrop, &base), &frame, ((w, h), step));
        println!("  {:<22} {step:<7} {ms:7.3}  ({added:+.3})", shot.name);
        if let Some(dir) = &png {
            gpu.set_stage(Some(&frame), w, h);
            let pixels = gpu.show_pixels(&pixmap, w, h).expect("a frame");
            write_png(&dir.join(format!("{}.png", shot.name)), &pixels, (w, h));
        }
    }
}
