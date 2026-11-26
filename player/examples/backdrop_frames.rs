//! Frames of Karaoke mode's backdrop on the GPU, with the stage over it, to
//! judge its look: covers already taken apart (`.layers` files from the
//! player's data folder) or pictures (a JPEG or PNG, taken apart without
//! the models: no subject), each stem quiet and loud, a section starting,
//! the singer at Diva, a cover fading in over its blur and one cross-fading
//! into the next, each combination of the game and the notes on or off
//! (the notes off: the lyrics in large type without the stage; neither:
//! the blurred cover alone, [`Mode`]) and the switch to neither mid-song,
//! and tall and wide windows; and with a steady beat (`beat-…`), the
//! singer at no tier, the lowest, Amazing and Diva, each on a bar's first
//! beat and off the beat, and Diva's phrase just landed. The music is made
//! up (each stem's level a
//! function of time), the backdrop and the stage stepped at 60 frames a
//! second up to the moment looked at, drawn offscreen as the window draws
//! them, and saved as PNGs into `OUT_DIR` (`<scene>.png`,
//! written whole). `--bare`: the backdrop alone, no stage and no text
//! (`<scene>-bare.png`), to tell what the music does to it. `--size WxH`:
//! every wide scene in a window of that size (the tall one `H` × 9 / 16
//! wide and `H` high). `--shrink SIDE`: the cover as a small, coarsely
//! coded one would be (its layers made `SIDE` pixels square, the plate
//! through JPEG at quality 60). `--art IMAGE`: the first cover's picture
//! itself (a JPEG or PNG, as a song's cover is decoded); without it, the
//! picture is put back together from the layers (the subject over the
//! plate). `--time`: how long the GPU takes to draw a frame of the
//! backdrop at each scene's size. `--crop`: a cut of each frame pixel for
//! pixel too (`<scene>-100.png`), to look for pixels and blocks.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example backdrop_frames --
//!     OUT_DIR COVER.layers|COVER.jpg [NEXT] [--only SCENE] [--bare] [--size WxH]
//!     [--shrink SIDE] [--art IMAGE] [--time] [--crop]
//! ```

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use jodel_gpu::vision::{Depth, Matte};
use jodeldiplom::art::Art;
use jodeldiplom::config::Config;
use jodeldiplom::core::shine::{Shone, Tier};
use jodeldiplom::core::singing::Singing;
use jodeldiplom::cover::{Apart, CoverLayers, Layers, Parts};
use jodeldiplom::gui::backdrop::music::{Hearing, Levels};
use jodeldiplom::gui::backdrop::{Cue, Mode, Visualiser};
use jodeldiplom::gui::big_text::Faces;
use jodeldiplom::gui::fx::Music;
use jodeldiplom::gui::karaoke::{paint, Backdrop, Layer};
use jodeldiplom::gui::pixels::Target;
use jodeldiplom::gui::present::Offscreen;
use jodeldiplom::gui::stage::{without_lyrics, Song, Stage};
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

/// A scene: its name, the window, the music at a time (and whether it
/// has a steady beat, a bar's first every two seconds), the shine tier
/// its phrase reached and when
/// it was decided, a section starting when, which cover shows from when (the
/// second from `next`, if any), whether the game and the notes are on
/// (from `switched`, if any: both on before), and when it is looked at.
#[derive(Clone, Copy)]
struct Scene {
    name: &'static str,
    size: (usize, usize),
    music: fn(f64) -> Levels,
    beat: bool,
    tier: Option<(Tier, f64)>,
    section: Option<f64>,
    arrives: f64,
    next: Option<f64>,
    game: bool,
    notes: bool,
    switched: Option<f64>,
    at: f64,
}

impl Scene {
    /// Karaoke mode's settings at `now`.
    fn config(&self, now: f64) -> Config {
        let on = self.switched.is_none_or(|s| now >= s);
        Config {
            karaoke_game: self.game || !on,
            karaoke_notes: self.notes || !on,
            ..Config::default()
        }
    }
}

fn quiet(_: f64) -> Levels {
    Levels {
        bass: 0.05,
        vocals: 0.0,
        other: 0.05,
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

fn loud(_: f64) -> Levels {
    Levels {
        bass: 0.85,
        vocals: 0.9,
        other: 0.8,
    }
}

fn middling(_: f64) -> Levels {
    Levels {
        bass: 0.5,
        vocals: 0.6,
        other: 0.5,
    }
}

/// What every scene is but for what it says: wide, quiet, no beat, the
/// game and the notes on, looked at 10 s in.
fn base_scene() -> Scene {
    Scene {
        name: "",
        size: (1600, 900),
        music: quiet,
        beat: false,
        tier: None,
        section: None,
        arrives: 0.0,
        next: None,
        game: true,
        notes: true,
        switched: None,
        at: 10.0,
    }
}

fn scenes() -> Vec<Scene> {
    let base = base_scene();
    let s = |name, music, at| Scene {
        name,
        music,
        at,
        ..base
    };
    vec![
        s("quiet", quiet, 10.0),
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
            tier: Some((Tier::TOP, 4.0)),
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
            game: false,
            ..s("notes-no-game", loud, 10.03)
        },
        Scene {
            notes: false,
            ..s("game-no-notes", loud, 10.03)
        },
        Scene {
            game: false,
            notes: false,
            ..s("plain", loud, 10.03)
        },
        Scene {
            game: false,
            notes: false,
            switched: Some(9.4),
            ..s("to-plain", loud, 10.0)
        },
        // With the beat: no tier, and the lowest, Amazing and the top,
        // each on a bar's first beat and just before the next beat; the
        // top's phrase just landed on one.
        Scene {
            beat: true,
            ..s("beat-none-on", middling, 10.02)
        },
        tiered("beat-good-on", "Good", 10.02),
        tiered("beat-good-off", "Good", 10.45),
        tiered("beat-amazing-on", "Amazing", 10.02),
        tiered("beat-amazing-off", "Amazing", 10.45),
        tiered("beat-diva-on", "Diva", 10.02),
        tiered("beat-diva-off", "Diva", 10.45),
        Scene {
            tier: Some((Tier::TOP, 9.85)),
            ..tiered("beat-diva-landed", "Diva", 10.02)
        },
    ]
}

/// A scene with the beat, its phrase reaching the tier `named` 4 s in,
/// looked at `at`.
fn tiered(name: &'static str, named: &str, at: f64) -> Scene {
    let base = base_scene();
    Scene {
        name,
        at,
        beat: true,
        tier: Some((Tier::named(named), 4.0)),
        music: middling,
        ..base
    }
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

/// `layers` as a cover `side` pixels square would have them, its plate
/// coded as a JPEG of quality 60 (the depth kept: it is the model's).
fn shrink(layers: &CoverLayers, side: u32) -> CoverLayers {
    use image::imageops::{resize, FilterType};
    let small = resize(&layers.plate, side, side, FilterType::Triangle);
    let mut jpeg = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 60)
        .encode_image(&small)
        .expect("a JPEG");
    let plate = image::load_from_memory(&jpeg).expect("the JPEG").to_rgb8();
    CoverLayers {
        plate,
        subject: layers
            .subject
            .as_ref()
            .map(|s| resize(s, side, side, FilterType::Triangle)),
        outline: resize(&layers.outline, side, side, FilterType::Triangle),
        ..layers.clone()
    }
}

/// A picture taken apart without the models: no subject, its depth nearer
/// towards its bottom (as a photo's ground comes nearer).
fn picture_apart(image: &image::RgbImage) -> CoverLayers {
    let side = 64;
    let depth = Depth {
        width: side,
        height: side,
        values: (0..side * side)
            .map(|i| 3.0 - 2.0 * (i / side) as f32 / (side - 1) as f32)
            .collect(),
    };
    let matte = Matte {
        width: 1,
        height: 1,
        values: vec![0.0],
    };
    jodeldiplom::cover::split(image, &Parts { depth, matte })
}

/// The layers in `path` (its file name the key; made `shrink` pixels
/// square if asked), saying how much of the cover its subject is. A JPEG
/// or PNG is the picture itself, decoded as a song's cover is and taken
/// apart without the models ([`picture_apart`]).
fn load(path: &Path, shrunk: Option<u32>) -> Layers {
    let bytes = std::fs::read(path).expect("the layers");
    let picture = ["jpg", "jpeg", "png"]
        .iter()
        .any(|e| path.extension().is_some_and(|x| x.eq_ignore_ascii_case(e)));
    let layers = if picture {
        picture_apart(&jodeldiplom::art::decode(&bytes).expect("a JPEG or PNG"))
    } else {
        jodeldiplom::cover::from_bytes(&bytes)
            .expect("layers")
            .layers
    };
    let layers = match shrunk {
        Some(side) => shrink(&layers, side),
        None => layers,
    };
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

/// The picture at `now` of the lines `sung`, scored as `singing` has it
/// with the `game` on (off: no score).
fn picture_at(
    sung: &[&jodel_lyrics::Line],
    now: f64,
    singing: &Singing,
    game: bool,
) -> KaraokePicture {
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
        score: game.then_some(Score {
            points: singing.tally.score,
            streak: singing.tally.streak,
        }),
        result: None,
    }
}

/// `img` as a PNG in `file`, written whole.
fn save(img: &image::RgbImage, file: &Path) {
    let mut png = std::io::Cursor::new(Vec::new());
    img.write_to(&mut png, image::ImageFormat::Png)
        .expect("a PNG");
    durable::write_atomic(file, png.get_ref()).expect("written");
}

/// How long the GPU takes to draw `frame`'s backdrop in a window of
/// `size`: frames drawn one after the other (each a little later, so none
/// is the last again) and read back once, less the reading back alone.
fn time(
    gpu: &mut Offscreen,
    frame: &jodeldiplom::gui::backdrop::BackdropFrame,
    size: (usize, usize),
) {
    let (w, h) = (size.0 as u32, size.1 as u32);
    let pixels = vec![jodeldiplom::gui::pixels::SEE_THROUGH; size.0 * size.1];
    let mut later = frame.clone();
    let mut run = |gpu: &mut Offscreen, n: usize| {
        let start = Instant::now();
        for _ in 0..n {
            for s in &mut later.slots {
                s.look.size[2] += 0.01;
            }
            gpu.set_backdrop(Some(&later), w, h);
        }
        gpu.show_pixels(&pixels, w, h).expect("a frame");
        start.elapsed().as_secs_f64()
    };
    run(gpu, 3);
    let once = (0..3).map(|_| run(gpu, 0)).fold(f64::MAX, f64::min);
    let n = 40;
    let many = (0..3).map(|_| run(gpu, n)).fold(f64::MAX, f64::min);
    println!(
        "{w}x{h}: {:.2} ms a frame ({})",
        (many - once) / n as f64 * 1000.0,
        gpu.name
    );
}

fn main() {
    jodeldiplom::gui::fonts::use_fonts(EMBEDDED_FONTS);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = std::path::PathBuf::from(args.first().expect("OUT_DIR"));
    let only = args.iter().skip_while(|a| *a != "--only").nth(1).cloned();
    let bare = args.iter().any(|a| a == "--bare");
    let timed = args.iter().any(|a| a == "--time");
    let cropped = args.iter().any(|a| a == "--crop");
    let after = |flag: &str| args.iter().skip_while(|a| *a != flag).nth(1).cloned();
    let size: Option<(usize, usize)> = after("--size").map(|s| {
        let (w, h) = s.split_once('x').expect("WxH");
        (w.parse().expect("W"), h.parse().expect("H"))
    });
    let shrunk: Option<u32> = after("--shrink").map(|s| s.parse().expect("SIDE"));
    let picture = after("--art").map(|f| {
        let bytes = std::fs::read(&f).expect("the picture");
        jodeldiplom::art::decode(&bytes).expect("a JPEG or PNG")
    });
    let files: Vec<&String> = args[1..]
        .iter()
        .take_while(|a| !a.starts_with("--"))
        .collect();
    let covers: Vec<Layers> = files.iter().map(|f| load(Path::new(f), shrunk)).collect();
    let first = covers.first().expect("a COVER.layers").clone();
    let next = covers.get(1).cloned().unwrap_or_else(|| first.clone());
    let arts: Vec<Art> = [&first, &next]
        .iter()
        .enumerate()
        .map(|(k, l)| Art {
            id: k as u64 + 1,
            image: Arc::new(match (&picture, k) {
                (Some(p), 0) => p.clone(),
                _ => cover_of(&l.layers),
            }),
        })
        .collect();
    let (lyrics, melody) = song();
    let sung: Vec<&jodel_lyrics::Line> = lyrics.lines.iter().collect();
    let mut gpu = Offscreen::new(false).expect("a GPU");
    for scene in scenes() {
        if only.as_deref().is_some_and(|o| o != scene.name) {
            continue;
        }
        let (w, h) = match size {
            Some((_, h)) if scene.size.0 < scene.size.1 => (h * 9 / 16, h),
            Some(size) => size,
            None => scene.size,
        };
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
        let (mut frame, mut picture, mut staged) = (None, None, true);
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
            let landed = scene.tier.filter(|&(_, d)| (now - d).abs() < 0.5 / FPS);
            if let Some((tier, _)) = landed {
                singing.decided(
                    now,
                    Shone {
                        line: 0,
                        end: now,
                        value: 1.0,
                        top: 0.9,
                        tier: Some(tier),
                    },
                );
            }
            let showing = usize::from(scene.next.is_some_and(|n| now >= n));
            let config = scene.config(now);
            let layers = (now >= scene.arrives).then(|| [&first, &next][showing].clone());
            let music = if scene.beat {
                Music::steady(now * 2.0, 120.0, 4, 0.9, now)
            } else {
                Music::NEUTRAL
            };
            let cue = Cue {
                art: Some(arts[showing].clone()),
                layers: Mode::of(&config).cover(layers),
                heard: Hearing {
                    levels: (scene.music)(now),
                    glow: jodeldiplom::gui::streak::level(&singing, config.karaoke_game),
                    section: scene.section.is_some_and(|s| (now - s).abs() < 0.5 / FPS),
                    phrase: landed.is_some() && config.karaoke_game,
                    music,
                },
            };
            frame = Some(visualiser.step(&cue, (w as u32, h as u32), (1.0 / FPS) as f32));
            let p = picture_at(&sung, now, &singing, config.karaoke_game);
            staged = config.karaoke_notes;
            if staged && !bare {
                let mut song = Song::new(&singing, sung.clone(), Some(LENGTH));
                song.game = config.karaoke_game;
                let area = (0.0, 0.0, w as f32, h as f32);
                let frame = stage.frame(&song, &p, area, now, (1.0 / FPS) as f32);
                // Every frame: the text it puts into the atlas is sent once.
                gpu.set_stage(Some(&frame), w as u32, h as u32);
            }
            picture = Some(p);
        }
        let picture = picture.expect("a frame");
        if timed {
            time(&mut gpu, frame.as_ref().expect("a frame"), (w, h));
        }
        gpu.set_backdrop(frame.as_ref(), w as u32, h as u32);
        if !staged || bare {
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
        } else if staged {
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
        save(&img, &file);
        if cropped {
            // Pixel for pixel: 1280 × 720 round the card's upper middle.
            let (cw, ch) = (1280.min(w as u32), 720.min(h as u32));
            let x = (w as u32 - cw) / 2;
            let y = ((h as f32 * 0.4) as u32)
                .saturating_sub(ch / 2)
                .min(h as u32 - ch);
            let cut = image::imageops::crop_imm(&img, x, y, cw, ch).to_image();
            save(
                &cut,
                &out.join(format!("{}{bare_name}-100.png", scene.name)),
            );
        }
        // What the music lit: the vocals, the rest, the bass; a section,
        // the glow, the rays, the lift.
        let look = frame.and_then(|f| f.slots.last().map(|s| s.look));
        let felt = look.map(|l| (l.music, l.moments));
        println!("{} (felt {felt:.2?})", file.display());
    }
}
