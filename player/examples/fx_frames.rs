//! The effects ([`jodeldiplom::gui::fx`]: the screen's edges on fire, the
//! marquee bulbs round the lane, the line sung on fire and frosted, the
//! coins' fountain, rain and flick, a word slammed down, and the marquee
//! on fire: the lane's glow and flames, the bulbs and the edges on fire
//! at once, as Diva's row has them) seen and timed over a frame of
//! Karaoke mode as the window
//! draws it: a made-up song sung on its notes up to ten seconds in, over a
//! cover taken apart (its `.layers` file), in a window of 3024 × 1964 (a
//! 16-inch MacBook Pro's screen) by default. Each effect is put into the
//! frame as the stage puts it (by the effects' board, from a row listing
//! it: its screen pass, its text in its treatment, its bursts, its words),
//! at a few strengths and moments: with no music, and moving with a
//! steady beat (120 a minute, bars of four, the grid sure) at a few places
//! in the beat and the bar (the bulbs' chase and flash, the fire's flare
//! and breath, the marquee on fire on a bar's first beat and off the beat,
//! the rain's bursts, the slam landing on a bar's first beat, shouting the
//! top tier's name or the multiplier). Its GPU time is what
//! it adds to the window pass's step it draws in (`fx`, `stage`, `bursts`
//! or `top`), the median over a few frames. `--png OUT_DIR`: each frame
//! whole as a PNG (`<effect>-<what>.png`).
//!
//! ```text
//! cargo run --release -p jodeldiplom --example fx_frames --
//!     COVER.layers [--size WxH] [--density D] [--png OUT_DIR]
//! ```

use std::path::Path;
use std::sync::Arc;

use jodeldiplom::art::Art;
use jodeldiplom::core::shine::{Reference, Tier};
use jodeldiplom::core::singing::Singing;
use jodeldiplom::cover::{Apart, Layers};
use jodeldiplom::gui::backdrop::music::Hearing;
use jodeldiplom::gui::backdrop::{BackdropFrame, Cue, Visualiser};
use jodeldiplom::gui::fx::{By, Effects, Fx, Layer, Music, Use};
use jodeldiplom::gui::present::Offscreen;
use jodeldiplom::gui::stage::{Song, Stage, Tint};
use jodeldiplom::gui::stage_gpu::StageFrame;

#[path = "../tests/common/sung.rs"]
mod sung;
use sung::{cover_of, levels, picture_at, song, with_effects, Fired, FPS, LENGTH};

// The fonts build.rs found: `EMBEDDED_FONTS`.
include!(concat!(env!("OUT_DIR"), "/fonts.rs"));

/// The moment the effects are drawn over.
const AT: f64 = 10.0;
/// Each frame timed this many times.
const TIMES: usize = 15;
/// The steady beat's tempo, beats a minute.
const TEMPO: f64 = 120.0;

/// The frame at [`AT`]: the song sung on its notes up to it, frame by
/// frame as the window draws it (into `gpu`); its backdrop, its stage and
/// the stage that made it.
fn frame_at(
    gpu: &mut Offscreen,
    (art, layers): (&Art, &Layers),
    (w, h): (u32, u32),
) -> (BackdropFrame, StageFrame, Stage) {
    let (lyrics, melody) = song();
    let sung: Vec<&jodel_lyrics::Line> = lyrics.lines.iter().collect();
    let mut singing = Singing::default();
    singing.prepare(
        Path::new("song"),
        &lyrics,
        &jodeldiplom::core::singing::melodic(&melody),
    );
    // A reference to sing against: the meter shows, as in a game.
    let phrases = vec![None; singing.notes.lines.len()];
    singing.set_reference(Some(Arc::new(Reference { phrases })));
    let mut stage = Stage::default();
    let mut visualiser = Visualiser::default();
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
            heard: Hearing {
                levels: levels(),
                glow: jodeldiplom::gui::streak::level(&singing, true),
                ..Hearing::default()
            },
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
        gpu.set_backdrop(Some(&backdrop), w, h);
        gpu.set_stage(Some(&frame), w, h);
        last = Some((backdrop, frame));
    }
    let (backdrop, frame) = last.expect("a frame");
    (backdrop, frame, stage)
}

/// An effect put into a frame: its name as shown, the effect, a row
/// holding it at its strength (and firing it), how long ago it fired
/// (bursts, a word; none: held, the word hit then), what fired it, and
/// the music it moves with: none, or a steady beat (the beats counted at
/// the frame).
struct Shot {
    name: String,
    label: &'static str,
    fx: Fx,
    row: Effects,
    ago: f64,
    by: By,
    beats: Option<f64>,
}

impl Shot {
    fn new(fx: Fx, strength: f32, ago: f64) -> Shot {
        let uses: &'static [Use] = Vec::leak(vec![Use::of(fx, strength)]);
        Shot {
            name: format!("{}-{strength}-{ago}", fx.spec().ident),
            label: fx.spec().ident,
            fx,
            row: Effects {
                held: uses,
                on_phrase: uses,
                on_rung: uses,
                ..Effects::NONE
            },
            ago,
            by: By::Phrase(0),
            beats: None,
        }
    }

    /// The marquee on fire: the lane's glow and flames, the bulbs and the
    /// edges on fire, each at full strength (timed by the bulbs' step).
    fn marquee() -> Shot {
        static ROW: [Use; 4] = [
            Use::of(Fx::Glow, 1.0),
            Use::of(Fx::Flames, 1.0),
            Use::of(Fx::EdgeFire, 1.0),
            Use::of(Fx::Bulbs, 1.0),
        ];
        Shot {
            name: "marquee".to_owned(),
            label: "marquee",
            fx: Fx::Bulbs,
            row: Effects {
                held: &ROW,
                ..Effects::NONE
            },
            ago: 0.0,
            by: By::Phrase(0),
            beats: None,
        }
    }

    /// `self` moving with a steady beat, `beats` beats in at the frame,
    /// named for it.
    fn on_beat(self, beats: f64, what: &str) -> Shot {
        Shot {
            name: format!("{}-beat-{what}", self.label),
            beats: Some(beats),
            ..self
        }
    }

    /// `self` fired by the top row (its phrase, or its rung reached).
    fn by_top(self, rung: bool) -> Shot {
        let top = Tier::TOP.index();
        Shot {
            by: if rung { By::Rung(top) } else { By::Phrase(top) },
            ..self
        }
    }

    /// The window pass's step the effect draws in.
    fn step(&self) -> &'static str {
        let spec = self.fx.spec();
        if spec.is_screen() {
            "fx"
        } else if spec.at.contains(&Layer::Over) {
            "top"
        } else if spec.lyric.is_some() {
            "stage"
        } else {
            "bursts"
        }
    }

    /// `base` (made by `stage`) with the effect in it, over the stage in
    /// `area`: fired [`Shot::ago`] before the frame, to its music going on
    /// from then (a steady grid, at the frame its beats counted).
    fn into(&self, stage: &mut Stage, base: &StageFrame, area: (f32, f32, f32, f32)) -> StageFrame {
        let clock = f64::from(base.clock);
        let music = |t: f64| {
            self.beats.map_or(Music::NEUTRAL, |b| {
                Music::steady(b + (t - clock) * TEMPO / 60.0, TEMPO, 4, 1.0, t)
            })
        };
        let fired = Fired {
            at: clock - self.ago,
            music: &music,
            by: self.by,
        };
        with_effects(stage, base, area, &self.row, fired)
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

/// What is shown: each effect with no music as made, then moving with the
/// beat at places in it.
fn shots() -> Vec<Shot> {
    let mut shots = vec![
        Shot::new(Fx::EdgeFire, 0.3, 0.0),
        Shot::new(Fx::EdgeFire, 1.0, 0.0),
        Shot::new(Fx::Bulbs, 0.3, 0.0),
        Shot::new(Fx::Bulbs, 1.0, 0.0),
        Shot::new(Fx::Bulbs, 0.6, 0.0),
        Shot::new(Fx::Glow, 1.0, 0.0),
        Shot::new(Fx::Flames, 1.0, 0.0),
        Shot::new(Fx::LyricFire, 0.3, 0.0),
        Shot::new(Fx::LyricFire, 1.0, 0.0),
        Shot::new(Fx::LyricFrost, 0.6, 0.0),
        Shot::new(Fx::LyricFrost, 1.0, 0.0),
        Shot::new(Fx::Fountain, 1.0, 0.5),
        Shot::new(Fx::Rain, 1.0, 1.0),
        Shot::new(Fx::Flick, 1.0, 0.15),
        Shot::new(Fx::Slam, 1.0, 0.3).by_top(false),
    ];
    // The bulbs: on the one, after it, between beats, the next beats.
    for (beats, what) in [
        (8.0, "one"),
        (8.2, "one-after"),
        (8.5, "half"),
        (9.0, "two"),
        (9.5, "two-half"),
        (10.0, "three"),
    ] {
        shots.push(Shot::new(Fx::Bulbs, 1.0, 0.0).on_beat(beats, what));
    }
    shots.push(Shot::new(Fx::Bulbs, 0.6, 0.0).on_beat(9.0, "two-0.6"));
    // The fire: flaring on the one, falling back, between.
    for (beats, what) in [
        (8.0, "one"),
        (8.15, "one-after"),
        (8.5, "half"),
        (9.0, "two"),
    ] {
        shots.push(Shot::new(Fx::EdgeFire, 1.0, 0.0).on_beat(beats, what));
    }
    // The marquee on fire: still, on the one, off the beat.
    shots.push(Shot::marquee());
    for (beats, what) in [(8.0, "one"), (8.5, "off"), (9.0, "two")] {
        shots.push(Shot::marquee().on_beat(beats, what));
    }
    // The rain: fired just before a beat, its bursts on the beats after.
    for ago in [0.3, 0.8, 1.3, 2.0] {
        let beats = 7.75 + ago * TEMPO / 60.0;
        let what = format!("{ago}s");
        shots.push(Shot::new(Fx::Rain, 1.0, ago).on_beat(beats, &what));
    }
    // The slam: fired two and a half beats into a bar, landing on the
    // next bar's first beat 0.75 s on; a moment before it, on it, after.
    for u in [-0.14, -0.1, -0.04, 0.0, 0.03, 0.08, 0.2, 0.45, 0.9, 1.05] {
        let ago = 0.75 + u;
        let beats = 6.5 + ago * TEMPO / 60.0;
        let what = format!("diva-{u}s");
        shots.push(
            Shot::new(Fx::Slam, 1.0, ago)
                .by_top(false)
                .on_beat(beats, &what),
        );
    }
    for u in [0.0, 0.2] {
        let ago = 0.75 + u;
        let what = format!("x32-{u}s");
        let beats = 6.5 + ago * TEMPO / 60.0;
        shots.push(
            Shot::new(Fx::Slam, 1.0, ago)
                .by_top(true)
                .on_beat(beats, &what),
        );
    }
    shots
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
    let (backdrop, base, mut stage) = frame_at(&mut gpu, (&art, &layers), (w, h));
    let area = (0.0, 0.0, w as f32, h as f32);
    println!("GPU ms, median: the window pass's step the effect draws in (what it adds)");
    for shot in &shots() {
        let frame = shot.into(&mut stage, &base, area);
        let step = shot.step();
        let (ms, added) = cost(&mut gpu, (&backdrop, &base), &frame, ((w, h), step));
        println!("  {:<26} {step:<7} {ms:7.3}  ({added:+.3})", shot.name);
        if let Some(dir) = &png {
            gpu.set_stage(Some(&frame), w, h);
            let pixels = gpu.show_frame(w, h).expect("a frame");
            write_png(&dir.join(format!("{}.png", shot.name)), &pixels, (w, h));
        }
    }
}
