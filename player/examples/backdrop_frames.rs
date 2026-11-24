//! Frames of Karaoke mode's backdrop on the GPU, with the stage over it, to
//! judge its look: covers already taken apart (`.layers` files from the
//! player's data folder), each stem quiet and loud, a section starting,
//! the singer at Diva, a cover fading in over its blur and one cross-fading
//! into the next, the lyrics in large type without the stage, and tall and
//! wide windows. The music is made up (each stem's level a function of
//! time, the drums hitting on the beat), the backdrop and the stage stepped
//! at 60 frames a second up to the moment looked at, drawn offscreen as the
//! window draws them, and saved as PNGs into `OUT_DIR` (`<scene>.png`,
//! written whole). `--bare`: the backdrop alone, no stage and no text
//! (`<scene>-bare.png`), to tell what the music does to it.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example backdrop_frames -- OUT_DIR COVER.layers [NEXT.layers] [--only SCENE] [--bare]
//! ```

use std::path::Path;
use std::sync::Arc;

use jodeldiplom::art::Art;
use jodeldiplom::core::shine::{Shone, Tier};
use jodeldiplom::core::singing::Singing;
use jodeldiplom::cover::{Apart, CoverLayers, Layers};
use jodeldiplom::gui::backdrop::music::Levels;
use jodeldiplom::gui::backdrop::{Cue, Visualiser};
use jodeldiplom::gui::big_text::Faces;
use jodeldiplom::gui::karaoke::{self, paint, Backdrop, Layer};
use jodeldiplom::gui::present::Offscreen;
use jodeldiplom::gui::raster::Target;
use jodeldiplom::gui::stage::{text_band, without_lyrics, Song, Stage};
use jodeldiplom::view::karaoke::{timed_lines, KaraokePicture, Score};
use ratatui::layout::Rect;

// The fonts build.rs found: `EMBEDDED_FONTS`.
include!(concat!(env!("OUT_DIR"), "/fonts.rs"));

const FPS: f64 = 60.0;

/// The song's words: text, from, to (seconds), pitch (hundredths of a MIDI
/// note), line by line, again every 8 s.
const WORDS: [&[(&str, f64, f64, u16)]; 2] = [
    &[
        ("Shine", 0.4, 1.2, 6200),
        ("on", 1.3, 1.8, 6400),
        ("tonight", 1.9, 3.4, 6700),
    ],
    &[
        ("we", 4.2, 4.7, 6400),
        ("sing", 4.8, 5.6, 6900),
        ("along", 5.7, 7.4, 6600),
    ],
];
const LOOP: f64 = 8.0;
const LENGTH: f64 = 40.0;

fn song() -> (Arc<jodel_lyrics::Lyrics>, jodel_rhythm::waveform::Melody) {
    let n = (LENGTH * 100.0) as usize;
    let mut m = jodel_rhythm::waveform::Melody {
        cents: vec![0; n],
        sure: vec![0; n],
    };
    let mut lines = Vec::new();
    let mut at = 0.0;
    while at + LOOP <= LENGTH {
        for words in WORDS {
            let words = words
                .iter()
                .map(|&(text, s, e, c)| {
                    for i in ((at + s) * 100.0) as usize..((at + e) * 100.0) as usize {
                        m.cents[i] = c;
                        m.sure[i] = 255;
                    }
                    jodel_lyrics::Word {
                        start: Some(at + s),
                        end: Some(at + e),
                        confidence: 0.9,
                        ..jodel_lyrics::Word::new(text)
                    }
                })
                .collect();
            lines.push(jodel_lyrics::Line {
                words,
                ..jodel_lyrics::Line::default()
            });
        }
        at += LOOP;
    }
    let lyrics = jodel_lyrics::Lyrics {
        lines,
        source: jodel_lyrics::Source::Unknown,
        timing: jodel_lyrics::Timing::default(),
    };
    (Arc::new(lyrics), m)
}

/// The drums on the beat (120 a minute): a hit for 40 ms, then quiet.
fn beat(t: f64) -> f32 {
    if (t * 2.0).fract() < 0.08 {
        1.0
    } else {
        0.1
    }
}

/// A scene: its name, the window, the music at a time, the shine tier
/// from when, a section starting when, which cover shows from when (the
/// second from `next`, if any), whether the stage shows, and when it is
/// looked at.
#[derive(Clone, Copy)]
struct Scene {
    name: &'static str,
    size: (usize, usize),
    music: fn(f64) -> Levels,
    diva: Option<f64>,
    section: Option<f64>,
    arrives: f64,
    next: Option<f64>,
    staged: bool,
    at: f64,
}

fn quiet(_: f64) -> Levels {
    Levels {
        drums: 0.05,
        bass: 0.05,
        vocals: 0.0,
        other: 0.05,
    }
}

fn drums(t: f64) -> Levels {
    Levels {
        drums: beat(t),
        ..quiet(t)
    }
}

fn bass(t: f64) -> Levels {
    Levels {
        bass: 1.0,
        ..quiet(t)
    }
}

fn vocals(t: f64) -> Levels {
    Levels {
        vocals: 1.0,
        ..quiet(t)
    }
}

fn other(t: f64) -> Levels {
    Levels {
        other: 1.0,
        ..quiet(t)
    }
}

fn loud(t: f64) -> Levels {
    Levels {
        drums: beat(t),
        bass: 0.85,
        vocals: 0.9,
        other: 0.8,
    }
}

fn middling(t: f64) -> Levels {
    Levels {
        drums: beat(t) * 0.7,
        bass: 0.5,
        vocals: 0.6,
        other: 0.5,
    }
}

fn scenes() -> Vec<Scene> {
    let base = Scene {
        name: "",
        size: (1600, 900),
        music: quiet,
        diva: None,
        section: None,
        arrives: 0.0,
        next: None,
        staged: true,
        at: 10.0,
    };
    let s = |name, music, at| Scene {
        name,
        music,
        at,
        ..base
    };
    vec![
        s("quiet", quiet, 10.0),
        s("drums-quiet", drums, 10.2),
        s("drums-hit", drums, 10.03),
        s("bass-loud", bass, 10.0),
        s("vocals-loud", vocals, 10.0),
        s("other-loud", other, 10.0),
        s("other-loud-later", other, 19.0),
        s("all-loud", loud, 10.03),
        Scene {
            section: Some(9.4),
            ..s("section", middling, 10.0)
        },
        Scene {
            diva: Some(4.0),
            ..s("diva", middling, 10.03)
        },
        Scene {
            size: (900, 1600),
            ..s("tall", loud, 10.03)
        },
        Scene {
            size: (2560, 1080),
            ..s("wide", loud, 10.03)
        },
        Scene {
            arrives: 9.4,
            ..s("fading-in", middling, 10.0)
        },
        Scene {
            next: Some(9.4),
            ..s("cross-fade", middling, 10.0)
        },
        Scene {
            staged: false,
            ..s("lyrics-no-stage", loud, 10.03)
        },
    ]
}

/// The cover as it was: the subject over the plate.
fn cover_of(l: &CoverLayers) -> image::RgbImage {
    let mut cover = l.plate.clone();
    if let Some(s) = &l.subject {
        for (p, q) in cover.pixels_mut().zip(s.pixels()) {
            let a = f32::from(q.0[3]) / 255.0;
            for c in 0..3 {
                p.0[c] = (f32::from(p.0[c]) * (1.0 - a) + f32::from(q.0[c]) * a).round() as u8;
            }
        }
    }
    cover
}

/// The layers in `path` (its file name the key), saying how much of the
/// cover its subject is.
fn load(path: &Path) -> Layers {
    let bytes = std::fs::read(path).expect("the layers");
    let layers = jodeldiplom::cover::from_bytes(&bytes)
        .expect("layers")
        .layers;
    let share = layers.subject.as_ref().map_or(0.0, |s| {
        let a: f64 = s.pixels().map(|p| f64::from(p.0[3]) / 255.0).sum();
        a / f64::from(s.width() * s.height())
    });
    println!(
        "{}: {:?}, subject {:.0} %",
        path.display(),
        layers.size(),
        share * 100.0
    );
    let key = path
        .file_stem()
        .map_or(String::new(), |s| s.to_string_lossy().into_owned());
    Arc::new(Apart::new(key, layers))
}

fn picture_at(sung: &[&jodel_lyrics::Line], now: f64, singing: &Singing) -> KaraokePicture {
    KaraokePicture {
        area: Rect::new(0, 0, 100, 24),
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

fn main() {
    jodeldiplom::gui::fonts::use_fonts(EMBEDDED_FONTS);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = std::path::PathBuf::from(args.first().expect("OUT_DIR"));
    let only = args.iter().skip_while(|a| *a != "--only").nth(1).cloned();
    let bare = args.iter().any(|a| a == "--bare");
    let files: Vec<&String> = args[1..]
        .iter()
        .take_while(|a| !a.starts_with("--"))
        .collect();
    let covers: Vec<Layers> = files.iter().map(|f| load(Path::new(f))).collect();
    let first = covers.first().expect("a COVER.layers").clone();
    let next = covers.get(1).cloned().unwrap_or_else(|| first.clone());
    let arts: Vec<Art> = [&first, &next]
        .iter()
        .enumerate()
        .map(|(k, l)| Art {
            id: k as u64 + 1,
            image: Arc::new(cover_of(&l.layers)),
        })
        .collect();
    let (lyrics, melody) = song();
    let sung: Vec<&jodel_lyrics::Line> = lyrics.lines.iter().collect();
    let mut gpu = Offscreen::new(false).expect("a GPU");
    for scene in scenes() {
        if only.as_deref().is_some_and(|o| o != scene.name) {
            continue;
        }
        let (w, h) = scene.size;
        let mut singing = Singing::default();
        singing.prepare(
            Path::new("song"),
            &lyrics,
            &jodeldiplom::core::singing::melodic(&melody),
        );
        let mut stage = Stage::default();
        let mut visualiser = Visualiser::default();
        let mut faces = Faces::default();
        let (mut heard, mut f) = (0usize, 0usize);
        let (mut frame, mut picture) = (None, None);
        loop {
            let now = f as f64 / FPS;
            if now > scene.at + 1e-9 {
                break;
            }
            f += 1;
            // The singer on the notes.
            while (heard as f64) / 100.0 <= now {
                let t = heard as f64 / 100.0;
                let c = singing.word_at(t).map(|(_, c)| c as f32);
                singing.hear(t, c);
                heard += 1;
            }
            if scene.diva.is_some_and(|d| (now - d).abs() < 0.5 / FPS) {
                singing.decided(
                    now,
                    Shone {
                        line: 0,
                        end: now,
                        value: 1.0,
                        diva: 0.9,
                        tier: Some(Tier::Diva),
                    },
                );
            }
            let showing = usize::from(scene.next.is_some_and(|n| now >= n));
            let layers = (now >= scene.arrives).then(|| [&first, &next][showing].clone());
            let cue = Cue {
                art: Some(arts[showing].clone()),
                layers,
                levels: (scene.music)(now),
                tier: jodeldiplom::gui::streak::level(&singing, true),
                section: scene.section.is_some_and(|s| (now - s).abs() < 0.5 / FPS),
                text: if scene.staged {
                    text_band()
                } else {
                    karaoke::text_band(h)
                },
            };
            frame = Some(visualiser.step(&cue, (w as u32, h as u32), (1.0 / FPS) as f32));
            let p = picture_at(&sung, now, &singing);
            if scene.staged && !bare {
                let song = Song::new(&singing, sung.clone(), Some(LENGTH));
                let area = (0.0, 0.0, w as f32, h as f32);
                let frame = stage.frame(&song, &p, area, now, (1.0 / FPS) as f32);
                // Every frame: the text it puts into the atlas is sent once.
                gpu.set_stage(Some(&frame), w as u32, h as u32);
            }
            picture = Some(p);
        }
        let picture = picture.expect("a frame");
        gpu.set_backdrop(frame.as_ref(), w as u32, h as u32);
        if !scene.staged || bare {
            gpu.set_stage(None, w as u32, h as u32);
        }
        let backdrop = Backdrop::through(w, h);
        let mut pixels = vec![0u32; w * h];
        let mut target = Target {
            pixels: &mut pixels,
            width: w,
            height: h,
        };
        let drawn = if bare {
            KaraokePicture {
                lines: Vec::new(),
                score: None,
                ..picture
            }
        } else if scene.staged {
            without_lyrics(&picture)
        } else {
            picture
        };
        paint(
            &mut target,
            &mut faces,
            &mut Layer::default(),
            &backdrop,
            &drawn,
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
        let bare_name = if bare { "-bare" } else { "" };
        let file = out.join(format!("{}{bare_name}.png", scene.name));
        let mut png = std::io::Cursor::new(Vec::new());
        img.write_to(&mut png, image::ImageFormat::Png).unwrap();
        durable::write_atomic(&file, png.get_ref()).unwrap();
        // What the music lit: the pulse, the vocals, the rest, the bass; a
        // section, the glow, the rays, the lift.
        let look = frame.and_then(|f| f.slots.last().map(|s| s.look));
        let felt = look.map(|l| (l.music, l.moments));
        println!("{} (felt {felt:.2?})", file.display());
    }
}
