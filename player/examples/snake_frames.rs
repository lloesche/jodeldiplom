//! Frames of Karaoke mode's stage on the GPU, to look at its word bubbles
//! and its rescues: a made-up song of words on one pitch and words
//! snaking over several (a melisma), a word missed alone then rescued by
//! the next one's shot (on the lane, and once long gone off it), and a
//! line in Japanese; sung as the notes go (the words missed sung well
//! off), in Karaoke mode's own fonts (as the player embeds them), the
//! stage drawn offscreen at 60 a second and the frames asked for saved as
//! PNGs (`OUT` with the frame's time before `.png`).
//!
//! ```text
//! cargo run --release -p jodeldiplom --example snake_frames -- OUT.png [W H] [--at SECONDS ...]
//! ```

use std::path::Path;
use std::sync::Arc;

use jodeldiplom::core::shine::{Shone, Tier};
use jodeldiplom::core::singing::{Cue, Singing};
use jodeldiplom::gui::big_text::Faces;
use jodeldiplom::gui::karaoke::{paint, Backdrop, Layer};
use jodeldiplom::gui::present::Offscreen;
use jodeldiplom::gui::raster::Target;
use jodeldiplom::gui::stage::{without_lyrics, Song, Stage};
use jodeldiplom::view::karaoke::{timed_lines, KaraokePicture, Score};
use ratatui::layout::Rect;

// The fonts build.rs found: `EMBEDDED_FONTS`.
include!(concat!(env!("OUT_DIR"), "/fonts.rs"));

/// A word: its text, when (from, to, seconds), its notes' pitches (an
/// even share of its time each, hundredths of a MIDI note), and whether
/// it is missed (sung well off).
type Word = (&'static str, (f64, f64), &'static [u16], bool);

/// The song's lines.
const LINES: [&[Word]; 5] = [
    &[
        ("Sun", (2.0, 2.6), &[6000], false),
        ("beautiful", (2.7, 4.5), &[6400, 6900, 6700, 6200], false),
        ("day", (4.6, 5.4), &[6400], false),
    ],
    &[
        ("la", (6.2, 6.8), &[6000], false),
        ("miss", (6.9, 7.5), &[6200], true),
        ("hey", (7.6, 8.3), &[6700], false),
        ("yeah", (8.4, 9.6), &[6700, 7200, 6900], false),
    ],
    &[
        ("so", (10.0, 10.6), &[6200], false),
        ("gone", (10.7, 11.5), &[6400], true),
    ],
    &[
        ("back", (15.0, 15.7), &[6000], false),
        ("again", (15.8, 16.8), &[6200, 6500], false),
    ],
    &[
        ("こんにちは", (17.5, 18.5), &[6400, 6700], false),
        ("世界", (18.6, 19.5), &[6200], false),
    ],
];

/// The song's lyrics and melody.
fn song() -> (Arc<jodel_lyrics::Lyrics>, jodel_rhythm::waveform::Melody) {
    let n = 21 * 100;
    let mut m = jodel_rhythm::waveform::Melody {
        cents: vec![0; n],
        sure: vec![0; n],
    };
    let lines = LINES
        .iter()
        .map(|words| jodel_lyrics::Line {
            words: words
                .iter()
                .map(|&(text, (s, e), notes, _)| {
                    let each = (e - s) / notes.len() as f64;
                    for (k, &c) in notes.iter().enumerate() {
                        let from = ((s + each * k as f64) * 100.0) as usize;
                        let to = ((s + each * (k + 1) as f64) * 100.0) as usize;
                        for i in from..to {
                            m.cents[i] = c;
                            m.sure[i] = 255;
                        }
                    }
                    jodel_lyrics::Word {
                        start: Some(s),
                        end: Some(e),
                        confidence: 0.9,
                        ..jodel_lyrics::Word::new(text)
                    }
                })
                .collect(),
            ..jodel_lyrics::Line::default()
        })
        .collect();
    let lyrics = jodel_lyrics::Lyrics {
        lines,
        source: jodel_lyrics::Source::Unknown,
        timing: jodel_lyrics::Timing::default(),
    };
    (Arc::new(lyrics), m)
}

/// Whether the singer misses the word sung at `t`.
fn missed(t: f64) -> bool {
    LINES
        .iter()
        .flat_map(|l| l.iter())
        .any(|&(_, (s, e), _, miss)| miss && (s..e).contains(&t))
}

/// The frames looked at when none are asked for: a word on one pitch, a
/// melisma's tube sweeping full and bursting, a shot onto a word missed
/// on the lane, one off its edge, a shout, the Japanese line.
const LOOKS: [f64; 17] = [
    2.15, 2.85, 2.95, 3.05, 3.12, 3.2, 7.95, 8.05, 8.15, 8.3, 8.42, 10.3, 15.3, 15.38, 15.46, 15.6,
    18.0,
];

fn main() {
    jodeldiplom::gui::fonts::use_fonts(EMBEDDED_FONTS);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = args.first().expect("OUT.png").clone();
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
    let (lyrics, melody) = song();
    let mut singing = Singing::default();
    singing.prepare(
        Path::new("song"),
        &lyrics,
        &jodeldiplom::core::singing::melodic(&melody),
    );
    let sung: Vec<&jodel_lyrics::Line> = lyrics.lines.iter().collect();
    let mut gpu = Offscreen::new(false).expect("a GPU");
    let mut stage = Stage::default();
    let mut faces = Faces::default();
    let backdrop = Backdrop::new(None, w, h);
    let last = wanted.iter().copied().fold(0.0, f64::max);
    let (mut heard, mut f, mut told) = (0usize, 0usize, 0usize);
    loop {
        let now = f as f64 / 60.0;
        if now > last + 1e-9 {
            break;
        }
        f += 1;
        // The singer on the notes, the words missed a fourth and more off.
        while (heard as f64) / 100.0 <= now {
            let t = heard as f64 / 100.0;
            let off = if missed(t) { 450.0 } else { 0.0 };
            let c = singing.word_at(t).map(|(_, c)| (c + off) as f32);
            singing.hear(t, c);
            heard += 1;
            // The second line shines Amazing as it ends (its shout over
            // the meter).
            if (t - 9.9).abs() < 1e-6 {
                singing.decided(t, amazing());
            }
        }
        for m in &singing.moments()[told..] {
            match m.cue {
                Cue::Shot { from, to, .. } => {
                    let lands = m.at + jodeldiplom::core::bubbles::FLIGHT;
                    println!("{:.2}: shot {from:?} -> {to:?}, lands {lands:.2}", m.at)
                }
                Cue::Pop(b) => {
                    // A word rescued: a shot to it before.
                    let to = (b.line, b.word);
                    let shot = |c: &Cue| matches!(c, Cue::Shot { to: t, .. } if *t == to);
                    let rescued = singing.moments().iter().any(|m| shot(&m.cue));
                    let rescued = if rescued { " (rescued)" } else { "" };
                    println!("{:.2}: pop {to:?}{rescued}", m.at)
                }
                _ => {}
            }
        }
        told = singing.moments().len();
        let picture = picture_at(&sung, now, &singing);
        let song = Song::new(&singing, sung.clone(), Some(21.0));
        let frame = stage.frame(
            &song,
            &picture,
            (0.0, 0.0, w as f32, h as f32),
            now,
            1.0 / 60.0,
        );
        gpu.set_stage(Some(&frame), w as u32, h as u32);
        if !wanted.iter().any(|&t| (t - now).abs() < 0.5 / 60.0) {
            continue;
        }
        let mut pixels = vec![0u32; w * h];
        let mut target = Target {
            pixels: &mut pixels,
            width: w,
            height: h,
        };
        paint(
            &mut target,
            &mut faces,
            &mut Layer::default(),
            &backdrop,
            &without_lyrics(&picture),
            (0, 0, w, h),
            None,
            true,
        );
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
            "{name}: score {}, streak {}, rescued {}",
            singing.tally.score, singing.tally.streak, singing.tally.rescued
        );
    }
}

/// The second line's shine: Amazing.
fn amazing() -> Shone {
    Shone {
        line: 1,
        end: 9.6,
        value: 0.85,
        top: 0.9,
        tier: Some(Tier::named("Amazing")),
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
        score: Some(Score {
            points: singing.tally.score,
            tier: singing.last_phrase().and_then(|p| p.tier),
            streak: singing.tally.streak,
        }),
        result: None,
    }
}
