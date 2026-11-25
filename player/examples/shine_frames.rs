//! Frames of Karaoke mode's stage on the GPU as the singing shines, to
//! look at: a made-up song of eight lines, its original singer a clean
//! voice, sung line by line worse or better ([`WOBBLE`]: the pitch
//! wobbling that far off the notes, silence for none) so that its phrases
//! reach each tier as the game scores them, the stage drawn offscreen at
//! 60 a second and the frames asked for saved as PNGs (`OUT` with the
//! frame's time before `.png`).
//!
//! ```text
//! cargo run --release -p jodeldiplom --example shine_frames -- OUT.png [W H] [--at SECONDS ...]
//! ```

use std::path::Path;
use std::sync::Arc;

use jodeldiplom::core::shine::reference::Original;
use jodeldiplom::core::shine::Tier;
use jodeldiplom::core::singing::{Cue, Singing};
use jodeldiplom::gui::big_text::Faces;
use jodeldiplom::gui::karaoke::{paint, Backdrop, Layer};
use jodeldiplom::gui::present::Offscreen;
use jodeldiplom::gui::raster::Target;
use jodeldiplom::gui::stage::{without_lyrics, Song, Stage};
use jodeldiplom::view::karaoke::{timed_lines, KaraokePicture};
use ratatui::layout::Rect;

/// How far off its notes each line is sung, the pitch wobbling (cents;
/// none: not sung at all).
const WOBBLE: [Option<f32>; 8] = [
    None,
    Some(14.0),
    Some(6.0),
    Some(2.0),
    Some(0.0),
    Some(0.0),
    Some(6.0),
    Some(0.0),
];

/// The song: eight lines of four words, each half a second on a note of a
/// tune, three seconds a line; and its melody.
fn song() -> (Arc<jodel_lyrics::Lyrics>, jodel_rhythm::waveform::Melody) {
    const TUNE: [u16; 8] = [6000, 6200, 6400, 6700, 6400, 6200, 6500, 6000];
    let n = 30 * 100;
    let mut m = jodel_rhythm::waveform::Melody {
        cents: vec![0; n],
        sure: vec![0; n],
    };
    let words = ["shine", "on", "the", "stage"];
    let lines = (0..8)
        .map(|l| {
            let from = 2.0 + 3.0 * l as f64;
            jodel_lyrics::Line {
                words: (0..4)
                    .map(|k| {
                        let s = from + 0.5 * k as f64;
                        for i in (s * 100.0) as usize..((s + 0.45) * 100.0) as usize {
                            m.cents[i] = TUNE[(l + k) % 8];
                            m.sure[i] = 255;
                        }
                        jodel_lyrics::Word {
                            start: Some(s),
                            end: Some(s + 0.45),
                            confidence: 0.9,
                            ..jodel_lyrics::Word::new(words[k])
                        }
                    })
                    .collect(),
                ..jodel_lyrics::Line::default()
            }
        })
        .collect();
    let lyrics = jodel_lyrics::Lyrics {
        lines,
        source: jodel_lyrics::Source::Unknown,
        timing: jodel_lyrics::Timing::default(),
    };
    (Arc::new(lyrics), m)
}

/// A voice singing `notes`' notes at `rate`, a sine at each.
fn voice(notes: &Singing, rate: u32, seconds: f64) -> Vec<f32> {
    (0..(f64::from(rate) * seconds) as usize)
        .map(|i| {
            let t = i as f64 / f64::from(rate);
            notes.word_at(t).map_or(0.0, |(_, c)| {
                let hz = 440.0 * 2f64.powf((c - 6900.0) / 1200.0);
                (0.3 * (std::f64::consts::TAU * hz * t).sin()) as f32
            })
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = args.first().expect("OUT.png").clone();
    let (w, h) = match (
        args.get(1).and_then(|a| a.parse().ok()),
        args.get(2).and_then(|a| a.parse().ok()),
    ) {
        (Some(w), Some(h)) => (w, h),
        _ => (1600usize, 900usize),
    };
    let wanted: Vec<f64> = args
        .iter()
        .skip_while(|a| *a != "--at")
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let (lyrics, melody) = song();
    let mut singing = Singing::default();
    singing.prepare(
        Path::new("song"),
        &lyrics,
        &jodeldiplom::core::singing::melodic(&melody),
    );
    let rate = 16_000;
    let original = voice(&singing, rate, 30.0);
    let reference = Original::new(&singing.notes, &original, rate).reference();
    singing.set_reference(Some(Arc::new(reference)));
    let sung: Vec<&jodel_lyrics::Line> = lyrics.lines.iter().collect();
    let mut gpu = Offscreen::new(false).expect("a GPU");
    let mut stage = Stage::default();
    let mut big = Faces::default();
    let backdrop = Backdrop::new(None, w, h);
    let last = wanted.iter().copied().fold(0.0, f64::max);
    let mut heard = 0usize;
    let mut f = 0usize;
    loop {
        let now = f as f64 / 60.0;
        if now > last + 1e-9 {
            break;
        }
        f += 1;
        // The singer on the notes, each line wobbling as `WOBBLE` says (a
        // frame each 10 ms).
        while (heard as f64) / 100.0 <= now {
            let t = heard as f64 / 100.0;
            let line = ((t - 2.0) / 3.0).floor().clamp(0.0, 7.0) as usize;
            let off = WOBBLE[line].map(|w| [w, 0.0, -w][heard % 3]);
            let c = singing.word_at(t).and_then(|(_, c)| Some(c as f32 + off?));
            singing.hear(t, c);
            heard += 1;
            if let Some(Cue::Phrase(p)) = singing.moments().last().map(|m| m.cue) {
                if singing.moments().last().is_some_and(|m| m.at == t) {
                    println!(
                        "phrase {} at {t:.2}: {:.2} of {}, {:?}",
                        p.line,
                        p.of_top(),
                        Tier::TOP.name(),
                        p.tier
                    );
                }
            }
        }
        if !wanted.iter().any(|&t| (t - now).abs() < 0.5 / 60.0) {
            // Only the stage moves between the frames looked at.
            let song = Song::new(&singing, sung.clone(), Some(30.0));
            let picture = picture_at(&sung, now, &singing);
            // Its text still goes to the atlas, as the window sends every frame.
            let frame = stage.frame(
                &song,
                &picture,
                (0.0, 0.0, w as f32, h as f32),
                now,
                1.0 / 60.0,
            );
            gpu.set_stage(Some(&frame), w as u32, h as u32);
            continue;
        }
        let picture = picture_at(&sung, now, &singing);
        let mut pixels = vec![0u32; w * h];
        let mut target = Target {
            pixels: &mut pixels,
            width: w,
            height: h,
        };
        paint(
            &mut target,
            &mut big,
            &mut Layer::default(),
            &backdrop,
            &without_lyrics(&picture),
            (0, 0, w, h),
            None,
            true,
        );
        let song = Song::new(&singing, sung.clone(), Some(30.0));
        let frame = stage.frame(
            &song,
            &picture,
            (0.0, 0.0, w as f32, h as f32),
            now,
            1.0 / 60.0,
        );
        gpu.set_stage(Some(&frame), w as u32, h as u32);
        let shown = gpu
            .show_pixels(&pixels, w as u32, h as u32)
            .expect("a frame");
        let mut img = image::RgbImage::new(w as u32, h as u32);
        for (i, p) in shown.iter().enumerate() {
            img.put_pixel(
                (i % w) as u32,
                (i / w) as u32,
                image::Rgb([(p >> 16) as u8, (p >> 8) as u8, *p as u8]),
            );
        }
        let name = out.replace(".png", &format!("{now:05.2}.png"));
        let mut png = durable::AtomicFile::create(Path::new(&name)).unwrap();
        img.write_to(&mut png, image::ImageFormat::Png).unwrap();
        png.commit().unwrap();
        println!(
            "{name}: tier {:?}, {} lights, {} light bursts",
            singing.last_phrase().and_then(|p| p.tier),
            frame.lights.len(),
            frame.light_sprays.len()
        );
    }
}

fn picture_at(sung: &[&jodel_lyrics::Line], now: f64, singing: &Singing) -> KaraokePicture {
    KaraokePicture {
        area: Rect::new(0, 0, 100, 24),
        bar: jodeldiplom::core::karaoke::controls::bar(Rect::new(0, 0, 100, 24)),
        lines: timed_lines(sung, now).unwrap_or_default(),
        count: None,
        cue: None,
        controls: None,
        start: None,
        before: Vec::new(),
        moved: 1.0,
        score: Some(jodeldiplom::view::karaoke::Score {
            points: singing.tally.score,
            tier: singing.last_phrase().and_then(|p| p.tier),
            streak: singing.tally.streak,
        }),
        result: None,
    }
}
