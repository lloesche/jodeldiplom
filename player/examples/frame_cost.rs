//! What a frame of Karaoke mode costs, step by step: the backdrop (a cover
//! taken apart, its `.layers` file from the player's data folder) and the
//! stage over it, stepped at 60 frames a second through a made-up song
//! sung on the notes, drawn offscreen as the window draws them, and each
//! step's GPU time read from timestamps (the window's pass drawn a step a
//! pass): a normal frame (the singing at no tier), and the Diva peak (two
//! phrases Diva, the second just decided: its jackpot flying, angels
//! coming down, the flames, the rays, the rainbow), and the Jodeldiplom
//! after that song (a new best: its confetti), timed over its whole
//! reveal. The singing scenes are timed over their second up to the moment
//! looked at, each in every size asked for
//! (`--size WxH`, again for more; the default a 16-inch MacBook Pro's
//! screen, 3024 × 1964), `--density D` pixels to a point (the window's
//! scale; 2, a Retina screen's, by default); each step's median and mean,
//! the frame's total (and its worst), and what the processor spent making each frame's
//! stage and backdrop. The GPU must write timestamps (Vulkan, DirectX 12,
//! Metal on Apple's chips). `--png OUT_DIR`: the last frame of each as a PNG
//! (`<scene>-<W>x<H>.png`, written whole), to hold the look to.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example frame_cost --
//!     COVER.layers [--size WxH ...] [--density D] [--png OUT_DIR]
//! ```

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use jodeldiplom::art::Art;
use jodeldiplom::core::shine::{Shone, Tier};
use jodeldiplom::core::singing::{Shown, Singing, SongResult};
use jodeldiplom::cover::{Apart, CoverLayers, Layers};
use jodeldiplom::gui::backdrop::music::Levels;
use jodeldiplom::gui::backdrop::{BackdropFrame, Cue, Visualiser};
use jodeldiplom::gui::big_text::Faces;
use jodeldiplom::gui::karaoke::{paint, Backdrop, Layer};
use jodeldiplom::gui::present::Offscreen;
use jodeldiplom::gui::raster::Target;
use jodeldiplom::gui::stage::{without_lyrics, Song, Stage, Tint};
use jodeldiplom::gui::stage_gpu::StageFrame;
use jodeldiplom::view::karaoke::{timed_lines, KaraokePicture, ResultShown, Score};
use ratatui::layout::Rect;

// The fonts build.rs found: `EMBEDDED_FONTS`.
include!(concat!(env!("OUT_DIR"), "/fonts.rs"));

const FPS: f64 = 60.0;
/// The song's words: text, from, to (seconds), pitch (hundredths of a MIDI
/// note), line by line, again every [`LOOP`] seconds.
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
/// How long before the moment looked at each scene is timed.
const TIMED: f64 = 1.0;

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

/// The music: every stem middling.
fn music() -> Levels {
    Levels {
        bass: 0.5,
        vocals: 0.6,
        other: 0.5,
    }
}

/// A scene: its name, the singing's phrases decided Diva when, the
/// moment looked at, and whether the song's diploma is timed from there
/// (its whole reveal) rather than the singing up to it.
struct Scene {
    name: &'static str,
    diva: &'static [f64],
    at: f64,
    diploma: bool,
}

const SCENES: [Scene; 3] = [
    Scene {
        name: "normal",
        diva: &[],
        at: 10.0,
        diploma: false,
    },
    Scene {
        name: "diva",
        diva: &[4.0, 9.2],
        at: 10.0,
        diploma: false,
    },
    Scene {
        name: "diploma",
        diva: &[4.0, 9.2],
        at: 10.0,
        diploma: true,
    },
];

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

/// The stage's shapes by kind ([`jodeldiplom::gui::stage_gpu::kind`]),
/// as the tool names them.
const KINDS: [&str; 25] = [
    "pill",
    "fill",
    "segment",
    "spark",
    "disc",
    "arrow",
    "text",
    "tube",
    "liquid",
    "bands",
    "swirl",
    "rim",
    "meter",
    "ring",
    "mark",
    "glow",
    "blaze",
    "badge",
    "tube liquid",
    "rays",
    "star",
    "seal",
    "paper",
    "stamp",
    "signature",
];

/// What the stage drew of each kind, in the frames looked at kind by
/// kind: how many in all, the screens their quads covered in all, and
/// the stage's step drawn with that kind alone, frame by frame.
#[derive(Default)]
struct Kinds {
    frames: usize,
    each: std::collections::BTreeMap<u32, (usize, f64, Vec<f64>)>,
}

impl Kinds {
    /// Each kind's shapes and the screens they covered a frame, and the
    /// stage with it alone (median, ms), the dearest first.
    fn print(&self) {
        let n = self.frames.max(1) as f64;
        let mut rows: Vec<(f64, String)> = (self.each.iter())
            .map(|(&k, (count, screens, ms))| {
                let mut ms = ms.clone();
                ms.sort_by(f64::total_cmp);
                let median = ms.get(ms.len() / 2).copied().unwrap_or(0.0);
                let name = KINDS.get(k as usize).copied().unwrap_or("?");
                let row = format!(
                    "  {name:<12} alone {median:7.3}  ({:.1} a frame, {:.2} screens)",
                    *count as f64 / n,
                    screens / n
                );
                (median, row)
            })
            .collect();
        rows.sort_by(|a, b| b.0.total_cmp(&a.0));
        println!("  the stage, kind by kind ({} frames):", self.frames);
        for (_, row) in rows {
            println!("{row}");
        }
    }
}

/// `frame` over `backdrop` timed, read back: its steps. With `kinds`, the
/// frame drawn again with each kind of the stage's shapes alone, the
/// stage's step of each counted there.
fn time(
    gpu: &mut Offscreen,
    backdrop: &BackdropFrame,
    frame: &StageFrame,
    (w, h): (u32, u32),
    kinds: Option<&mut Kinds>,
) -> Vec<(&'static str, f64)> {
    gpu.time_frame(Some(backdrop), Some(frame), w, h)
        .expect("timed");
    let Some(kinds) = kinds else {
        return gpu.costs().expect("the times").remove(0);
    };
    let kind = |i: &jodeldiplom::gui::stage_gpu::Instance| i.params[0].round() as u32;
    let mut present: Vec<u32> = frame.instances.iter().chain(&frame.top).map(kind).collect();
    present.sort_unstable();
    present.dedup();
    for &k in &present {
        let only = StageFrame {
            instances: frame
                .instances
                .iter()
                .filter(|i| kind(i) == k)
                .copied()
                .collect(),
            top: frame.top.iter().filter(|i| kind(i) == k).copied().collect(),
            ..frame.clone()
        };
        gpu.time_frame(Some(backdrop), Some(&only), w, h)
            .expect("timed");
    }
    // The whole frame again over the next (and in the picture saved).
    gpu.set_stage(Some(frame), w, h);
    let mut costs = gpu.costs().expect("the times").into_iter();
    let whole = costs.next().expect("the frame");
    kinds.frames += 1;
    for (&k, steps) in present.iter().zip(costs) {
        let shapes: Vec<_> = frame
            .instances
            .iter()
            .chain(&frame.top)
            .filter(|i| kind(i) == k)
            .collect();
        let screens: f64 = (shapes.iter())
            .map(|i| f64::from((i.quad[2] - i.quad[0]) * (i.quad[3] - i.quad[1])))
            .sum::<f64>()
            / f64::from(w * h);
        let stage = steps.iter().find(|s| s.0 == "stage").map_or(0.0, |s| s.1);
        let e = kinds.each.entry(k).or_default();
        e.0 += shapes.len();
        e.1 += screens;
        e.2.push(stage);
    }
    whole
}

/// What makes a scene's frames in one size: the stage, the backdrop and
/// the pixmap under them, as the window keeps them.
struct Rig<'a> {
    stage: Stage,
    visualiser: Visualiser,
    pixels: Vec<u32>,
    faces: Faces,
    layer: Layer,
    under: Backdrop,
    art: &'a Art,
    layers: &'a Layers,
    sung: &'a [&'a jodel_lyrics::Line],
    size: (u32, u32),
    /// Frames made so far.
    made: usize,
}

impl Rig<'_> {
    /// The frame at `now` (`result`: the diploma shown over it), and what
    /// the processor took for its stage and its pixmap, ms.
    fn frame(
        &mut self,
        singing: &Singing,
        now: f64,
        result: Option<ResultShown>,
    ) -> (BackdropFrame, StageFrame, [f64; 2]) {
        let (w, h) = self.size;
        self.made += 1;
        let started = Instant::now();
        let cue = Cue {
            art: Some(self.art.clone()),
            layers: Some(self.layers.clone()),
            levels: music(),
            tier: jodeldiplom::gui::streak::level(singing, true),
            section: false,
        };
        let backdrop = self.visualiser.step(&cue, (w, h), (1.0 / FPS) as f32);
        let picture = KaraokePicture {
            result,
            ..picture_at(self.sung, now, singing)
        };
        let song = Song::new(singing, self.sung.to_vec(), Some(LENGTH));
        let area = (0.0, 0.0, w as f32, h as f32);
        // As the window has it: the result screen while one shows, else
        // the lane.
        let frame = self
            .stage
            .scene(
                (&song, true),
                &picture,
                area,
                (now, (1.0 / FPS) as f32),
                &Tint::STAGE,
            )
            .expect("a stage");
        let stage_ms = started.elapsed().as_secs_f64() * 1000.0;
        // The pixmap as the window paints it: only what changed.
        let started = Instant::now();
        let mut target = Target {
            pixels: &mut self.pixels,
            width: w as usize,
            height: h as usize,
        };
        paint(
            &mut target,
            &mut self.faces,
            &mut self.layer,
            &self.under,
            &without_lyrics(&picture),
            (0, 0, w as usize, h as usize),
            None,
            self.made == 1,
        );
        let pixmap_ms = started.elapsed().as_secs_f64() * 1000.0;
        (backdrop, frame, [stage_ms, pixmap_ms])
    }
}

/// The frames timed (each its steps' GPU times) and what the processor
/// took for each (the stage and the backdrop made, the pixmap painted):
/// each step's median and mean, the GPU's total, and the processor's, in
/// milliseconds.
fn print(title: &str, frames: &[Vec<(&'static str, f64)>], cpu: &[[f64; 2]]) {
    let stats = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        let median = v.get(v.len() / 2).copied().unwrap_or(0.0);
        (median, v.iter().sum::<f64>() / v.len().max(1) as f64)
    };
    println!("{title}: {} frames, ms (median / mean)", frames.len());
    let names = frames
        .first()
        .map_or(Vec::new(), |f| f.iter().map(|s| s.0).collect());
    for name in names {
        let mut v: Vec<f64> = frames
            .iter()
            .flat_map(|f| f.iter().filter(|s| s.0 == name).map(|s| s.1))
            .collect();
        let (median, mean) = stats(&mut v);
        println!("  {name:<10} {median:7.3} {mean:7.3}");
    }
    let mut total: Vec<f64> = frames.iter().map(|f| f.iter().map(|s| s.1).sum()).collect();
    let (median, mean) = stats(&mut total);
    let worst = total.last().copied().unwrap_or(0.0);
    println!(
        "  {:<10} {median:7.3} {mean:7.3}  (worst {worst:.3})",
        "GPU total"
    );
    for (k, name) in ["CPU stage", "CPU pixmap"].into_iter().enumerate() {
        let (median, mean) = stats(&mut cpu.iter().map(|c| c[k]).collect());
        println!("  {name:<10} {median:7.3} {mean:7.3}");
    }
}

fn main() {
    jodeldiplom::gui::fonts::use_fonts(EMBEDDED_FONTS);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cover = args.first().expect("COVER.layers");
    let mut sizes: Vec<(u32, u32)> = args
        .iter()
        .zip(args.iter().skip(1))
        .filter(|(a, _)| *a == "--size")
        .map(|(_, s)| {
            let (w, h) = s.split_once('x').expect("WxH");
            (w.parse().expect("W"), h.parse().expect("H"))
        })
        .collect();
    if sizes.is_empty() {
        sizes.push((3024, 1964));
    }
    let after = |flag: &str| args.iter().skip_while(|a| *a != flag).nth(1);
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
    let (lyrics, melody) = song();
    let sung: Vec<&jodel_lyrics::Line> = lyrics.lines.iter().collect();
    let mut gpu = Offscreen::timed().expect("a GPU that times");
    gpu.set_density(density);
    println!("{}, {density} pixels to a point", gpu.name);
    for &(w, h) in &sizes {
        for scene in &SCENES {
            let mut singing = Singing::default();
            singing.prepare(
                Path::new("song"),
                &lyrics,
                &jodeldiplom::core::singing::melodic(&melody),
            );
            let mut rig = Rig {
                stage: Stage::default(),
                visualiser: Visualiser::default(),
                // The pixmap under it all (the picture without its lyrics:
                // the stage draws them), sent over once.
                pixels: vec![0u32; (w * h) as usize],
                faces: Faces::default(),
                layer: Layer::default(),
                under: Backdrop::through(w as usize, h as usize),
                art: &art,
                layers: &layers,
                sung: &sung,
                size: (w, h),
                made: 0,
            };
            let (mut cpu_ms, mut frames) = (Vec::new(), Vec::new());
            let mut kinds = Kinds::default();
            let mut heard = 0usize;
            for f in 0.. {
                let now = f as f64 / FPS;
                if now > scene.at + 1e-9 {
                    break;
                }
                while (heard as f64) / 100.0 <= now {
                    let t = heard as f64 / 100.0;
                    let c = singing.word_at(t).map(|(_, c)| c as f32);
                    singing.hear(t, c);
                    heard += 1;
                }
                if scene.diva.iter().any(|d| (now - d).abs() < 0.5 / FPS) {
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
                let (backdrop, frame, cpu) = rig.frame(&singing, now, None);
                if rig.made == 1 {
                    gpu.set_backdrop(Some(&backdrop), w, h);
                    gpu.show_pixels(&rig.pixels, w, h).expect("a frame");
                }
                if !scene.diploma && now > scene.at - TIMED {
                    frames.push(time(&mut gpu, &backdrop, &frame, (w, h), Some(&mut kinds)));
                    cpu_ms.push(cpu);
                } else {
                    gpu.set_backdrop(Some(&backdrop), w, h);
                    gpu.set_stage(Some(&frame), w, h);
                }
            }
            if scene.diploma {
                // The song over: its result, a first score (a new best),
                // revealed frame by frame until it is still.
                let (tally, phrases, notes) = singing.take();
                let result = SongResult::new("Shine on".into(), &tally, phrases, notes, None);
                let shown = Arc::new(Shown::new(result, 1, 0));
                let still = shown.reveal.still();
                for f in 0.. {
                    let t = f as f64 / FPS;
                    if t > still {
                        break;
                    }
                    let result = ResultShown {
                        shown: shown.clone(),
                        t,
                        next: String::new(),
                        again: false,
                    };
                    let (backdrop, frame, cpu) = rig.frame(&singing, scene.at, Some(result));
                    // Kind by kind every tenth frame: the reveal is long.
                    let by_kind = (f % 10 == 0).then_some(&mut kinds);
                    frames.push(time(&mut gpu, &backdrop, &frame, (w, h), by_kind));
                    cpu_ms.push(cpu);
                }
            }
            print(&format!("{} {w}x{h}", scene.name), &frames, &cpu_ms);
            kinds.print();
            if let Some(dir) = &png {
                let shown = gpu.show_pixels(&rig.pixels, w, h).expect("a frame");
                let mut img = image::RgbImage::new(w, h);
                for (i, p) in shown.iter().enumerate() {
                    img.put_pixel(
                        i as u32 % w,
                        i as u32 / w,
                        image::Rgb([(p >> 16) as u8, (p >> 8) as u8, *p as u8]),
                    );
                }
                let mut bytes = std::io::Cursor::new(Vec::new());
                img.write_to(&mut bytes, image::ImageFormat::Png)
                    .expect("a PNG");
                let file = dir.join(format!("{}-{w}x{h}.png", scene.name));
                durable::write_atomic(&file, bytes.get_ref()).expect("written");
            }
        }
    }
}
