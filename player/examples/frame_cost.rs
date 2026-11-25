//! What a frame of Karaoke mode costs, step by step: the backdrop (a cover
//! taken apart, its `.layers` file from the player's data folder) and the
//! stage over it, stepped at 60 frames a second through a made-up song
//! sung on the notes (`tests/common/sung.rs`), drawn offscreen as the
//! window draws them, and each step's GPU time read from timestamps (the
//! window's pass drawn a step a pass): a normal frame (the singing at no
//! tier), the Diva peak (two phrases Diva, the second just decided: its
//! jackpot flying, angels coming down, the flames, the rays, the rainbow),
//! the worst the effects can make (`all`: the Diva peak with every
//! registered effect in it at full strength at once, the edges on fire,
//! the bulbs, the coins too, and the lyrics' lines in every lyric
//! treatment in turn), and the Jodeldiplom after that song (a new best:
//! its confetti), timed over its whole reveal. The singing scenes are
//! timed over their second up to the moment looked at, each in every size
//! asked for (`--size WxH`, again for more; the default a 16-inch MacBook
//! Pro's screen, 3024 × 1964), `--density D` pixels to a point (the
//! window's scale; 2, a Retina screen's, by default); each step's median,
//! mean and worst, the frame's total (and its worst), and what the
//! processor spent making each frame's stage and backdrop. The GPU must
//! write timestamps (Vulkan, DirectX 12, Metal on Apple's chips). `--png
//! OUT_DIR`: the last frame of each as a PNG (`<scene>-<W>x<H>.png`,
//! written whole), to hold the look to. Each scene's stage is also timed
//! kind by kind (the frame drawn with that kind of shape alone: its stage,
//! screen-pass and top steps) and effect by effect (the frame with that
//! effect's shapes alone, its light and its text in its treatment too: its
//! stage, light, screen-pass and top steps). First, how long the device
//! and the window's pipelines take to make.
//!
//! `--budget MS`: exits with 1 (after printing it all) where any scene's
//! worst frame took the GPU longer than that in all, or any step did;
//! `--step-budget MS`: where any step's worst did (the budgets to hold
//! to, and on which machine: PERFORMANCE.md).
//!
//! ```text
//! cargo run --release -p jodeldiplom --example frame_cost --
//!     COVER.layers [--size WxH ...] [--density D] [--png OUT_DIR]
//!     [--budget MS] [--step-budget MS]
//! ```

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use jodeldiplom::art::Art;
use jodeldiplom::core::shine::{Shone, Tier};
use jodeldiplom::core::singing::{Shown, Singing, SongResult};
use jodeldiplom::cover::{Apart, Layers};
use jodeldiplom::gui::backdrop::{BackdropFrame, Cue, Visualiser};
use jodeldiplom::gui::big_text::Faces;
use jodeldiplom::gui::fx::{Effects, Fx, Treat, Use};
use jodeldiplom::gui::karaoke::{paint, Backdrop, Layer};
use jodeldiplom::gui::present::Offscreen;
use jodeldiplom::gui::raster::Target;
use jodeldiplom::gui::stage::{without_lyrics, Song, Stage, Tint};
use jodeldiplom::gui::stage_gpu::{kind, Instance, StageFrame};
use jodeldiplom::view::karaoke::{KaraokePicture, ResultShown};

#[path = "../tests/common/sung.rs"]
mod sung;
use sung::{cover_of, levels, picture_at, song, with_effects, FPS, LENGTH};

// The fonts build.rs found: `EMBEDDED_FONTS`.
include!(concat!(env!("OUT_DIR"), "/fonts.rs"));

/// How long before the moment looked at each scene is timed.
const TIMED: f64 = 1.0;

/// A scene: its name, the singing's phrases decided Diva when, the
/// moment looked at, whether the song's diploma is timed from there (its
/// whole reveal) rather than the singing up to it, and whether every
/// effect is in its frames at once ([`every_effect`]).
struct Scene {
    name: &'static str,
    diva: &'static [f64],
    at: f64,
    diploma: bool,
    every: bool,
}

const SCENES: [Scene; 4] = [
    Scene {
        name: "normal",
        diva: &[],
        at: 10.0,
        diploma: false,
        every: false,
    },
    Scene {
        name: "diva",
        diva: &[4.0, 9.2],
        at: 10.0,
        diploma: false,
        every: false,
    },
    Scene {
        name: "all",
        diva: &[4.0, 9.2],
        at: 10.0,
        diploma: false,
        every: true,
    },
    Scene {
        name: "diploma",
        diva: &[4.0, 9.2],
        at: 10.0,
        diploma: true,
        every: false,
    },
];

/// The effects Diva's row does not list, each at full strength, held and
/// fired by a phrase: with Diva's own, every registered effect at once.
fn the_rest() -> Effects {
    let diva = &Tier::TOP.spec().fx;
    let rest: Vec<Use> = (Fx::ALL.into_iter())
        .filter(|&fx| !diva.lists_fx(fx))
        .map(|fx| Use::of(fx, 1.0))
        .collect();
    let rest: &'static [Use] = Vec::leak(rest);
    Effects {
        held: rest,
        on_phrase: rest,
        ..Effects::NONE
    }
}

/// `frame` (the singing at Diva, over the stage in `area`) with every
/// effect in it at full strength ([`the_rest`], fired at `fired`), the
/// lyrics' lines in every lyric treatment in turn (each its own pipeline
/// drawing): the dearest frame the effects can make.
fn every_effect(
    frame: &StageFrame,
    area: (f32, f32, f32, f32),
    (rest, fired): (&Effects, f64),
) -> StageFrame {
    let mut f = with_effects(frame, area, rest, fired);
    let codes: Vec<f32> = (Fx::ALL.into_iter())
        .filter(|fx| fx.spec().lyric.is_some())
        .map(Fx::treatment_code)
        .collect();
    let lyric = |i: &Instance| i.params[0] == kind::TEXT && i.params[1].is_finite();
    let mut tops: Vec<u32> = (f.instances.iter())
        .filter(|i| lyric(i))
        .map(|i| i.quad[1].to_bits())
        .collect();
    tops.sort_unstable();
    tops.dedup();
    for i in f.instances.iter_mut().filter(|i| lyric(i)) {
        let line = tops.binary_search(&i.quad[1].to_bits()).unwrap_or(0);
        let code = codes[line % codes.len()];
        Treat {
            code,
            strain: [1.0, 1.0, 0.0],
        }
        .apply(i);
    }
    f
}
/// What the stage drew of each kind (or effect), in the frames looked at
/// one by one: how many shapes in all, the screens their quads covered in
/// all, and the steps it costs drawn alone, frame by frame.
#[derive(Default)]
struct Alone {
    frames: usize,
    each: std::collections::BTreeMap<&'static str, (usize, f64, Vec<f64>)>,
}

impl Alone {
    /// Each one's shapes and the screens they covered a frame, and its
    /// steps alone (median, ms), the dearest first, under `title`.
    fn print(&self, title: &str) {
        let n = self.frames.max(1) as f64;
        let mut rows: Vec<(f64, String)> = (self.each.iter())
            .map(|(name, (count, screens, ms))| {
                let mut ms = ms.clone();
                ms.sort_by(f64::total_cmp);
                let median = ms.get(ms.len() / 2).copied().unwrap_or(0.0);
                let row = format!(
                    "  {name:<14} alone {median:7.3}  ({:.1} a frame, {:.2} screens)",
                    *count as f64 / n,
                    screens / n
                );
                (median, row)
            })
            .collect();
        rows.sort_by(|a, b| b.0.total_cmp(&a.0));
        println!("  {title} ({} frames):", self.frames);
        for (_, row) in rows {
            println!("{row}");
        }
    }
}

/// The effect a shape is of: an effect's own kind, or text in its
/// treatment (none: the stage's own).
fn effect_of(i: &Instance) -> Option<Fx> {
    Fx::of_kind(i.params[0])
        .map(|(fx, _)| fx)
        .or_else(|| Treat::of(i).map(|(fx, _)| fx))
}

/// The steps a frame drawn with one kind alone is timed by, and one
/// effect alone.
const KIND_STEPS: [&str; 3] = ["stage", "fx", "top"];
const EFFECT_STEPS: [&str; 6] = ["light", "bloom", "stage", "composite", "fx", "top"];

/// `frame` over `backdrop` timed, read back: its steps. With `alone`,
/// the frame drawn again with each kind of the stage's shapes alone (its
/// [`KIND_STEPS`] counted), then with each effect's alone (its
/// [`EFFECT_STEPS`]).
fn time(
    gpu: &mut Offscreen,
    backdrop: &BackdropFrame,
    frame: &StageFrame,
    (w, h): (u32, u32),
    alone: Option<(&mut Alone, &mut Alone)>,
) -> Vec<(&'static str, f64)> {
    gpu.time_frame(Some(backdrop), Some(frame), w, h)
        .expect("timed");
    let Some((kinds, effects)) = alone else {
        return gpu.costs().expect("the times").remove(0);
    };
    let shapes = |f: &StageFrame| -> Vec<Instance> {
        (f.instances
            .iter()
            .chain(&f.top)
            .chain(&f.screen)
            .chain(&f.lights))
        .copied()
        .collect()
    };
    let all = shapes(frame);
    let only = |keep: &dyn Fn(&Instance) -> bool| StageFrame {
        instances: frame
            .instances
            .iter()
            .filter(|i| keep(i))
            .copied()
            .collect(),
        top: frame.top.iter().filter(|i| keep(i)).copied().collect(),
        screen: frame.screen.iter().filter(|i| keep(i)).copied().collect(),
        lights: frame.lights.iter().filter(|i| keep(i)).copied().collect(),
        ..frame.clone()
    };
    let mut names: Vec<&'static str> = all.iter().map(|i| kind::name(i.params[0])).collect();
    names.sort_unstable();
    names.dedup();
    let mut fxs: Vec<Fx> = all.iter().filter_map(effect_of).collect();
    fxs.sort_unstable();
    fxs.dedup();
    for &name in &names {
        let f = only(&|i| kind::name(i.params[0]) == name);
        gpu.time_frame(Some(backdrop), Some(&f), w, h)
            .expect("timed");
    }
    for &fx in &fxs {
        let f = only(&|i| effect_of(i) == Some(fx));
        gpu.time_frame(Some(backdrop), Some(&f), w, h)
            .expect("timed");
    }
    // The whole frame again over the next (and in the picture saved).
    gpu.set_stage(Some(frame), w, h);
    let mut costs = gpu.costs().expect("the times").into_iter();
    let whole = costs.next().expect("the frame");
    let mut count =
        |into: &mut Alone, name: &'static str, keep: &dyn Fn(&Instance) -> bool, steps: &[&str]| {
            let costs = costs.next().expect("timed alone");
            let mine: Vec<&Instance> = all.iter().filter(|i| keep(i)).collect();
            let screens: f64 = (mine.iter())
                .map(|i| f64::from((i.quad[2] - i.quad[0]) * (i.quad[3] - i.quad[1])))
                .sum::<f64>()
                / f64::from(w * h);
            let ms = (costs.iter())
                .filter(|s| steps.contains(&s.0))
                .map(|s| s.1)
                .sum();
            let e = into.each.entry(name).or_default();
            e.0 += mine.len();
            e.1 += screens;
            e.2.push(ms);
        };
    for &name in &names {
        count(
            kinds,
            name,
            &|i| kind::name(i.params[0]) == name,
            &KIND_STEPS,
        );
    }
    for &fx in &fxs {
        count(
            effects,
            fx.spec().name,
            &|i| effect_of(i) == Some(fx),
            &EFFECT_STEPS,
        );
    }
    kinds.frames += 1;
    effects.frames += 1;
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
            levels: levels(),
            glow: jodeldiplom::gui::streak::level(singing, true),
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
/// each step's median, mean and worst, the GPU's total, and the
/// processor's, in milliseconds. Gives back each step's worst and the
/// worst frame's total (`GPU total`).
fn print(
    title: &str,
    frames: &[Vec<(&'static str, f64)>],
    cpu: &[[f64; 2]],
) -> Vec<(&'static str, f64)> {
    let stats = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        let median = v.get(v.len() / 2).copied().unwrap_or(0.0);
        let worst = v.last().copied().unwrap_or(0.0);
        (median, v.iter().sum::<f64>() / v.len().max(1) as f64, worst)
    };
    println!(
        "{title}: {} frames, ms (median / mean / worst)",
        frames.len()
    );
    let names = frames
        .first()
        .map_or(Vec::new(), |f| f.iter().map(|s| s.0).collect());
    let mut worsts = Vec::new();
    for name in names {
        let mut v: Vec<f64> = frames
            .iter()
            .flat_map(|f| f.iter().filter(|s| s.0 == name).map(|s| s.1))
            .collect();
        let (median, mean, worst) = stats(&mut v);
        println!("  {name:<10} {median:7.3} {mean:7.3} {worst:7.3}");
        worsts.push((name, worst));
    }
    let mut total: Vec<f64> = frames.iter().map(|f| f.iter().map(|s| s.1).sum()).collect();
    let (median, mean, worst) = stats(&mut total);
    println!("  {:<10} {median:7.3} {mean:7.3} {worst:7.3}", "GPU total");
    worsts.push(("GPU total", worst));
    for (k, name) in ["CPU stage", "CPU pixmap"].into_iter().enumerate() {
        let (median, mean, worst) = stats(&mut cpu.iter().map(|c| c[k]).collect());
        println!("  {name:<10} {median:7.3} {mean:7.3} {worst:7.3}");
    }
    worsts
}

/// What went over the budgets (`--budget`: the worst frame's GPU total
/// and every step; `--step-budget`: every step), scene by scene: each
/// as a line saying what, by how much.
fn over(
    scenes: &[(String, Vec<(&'static str, f64)>)],
    (budget, step_budget): (Option<f64>, Option<f64>),
) -> Vec<String> {
    let mut over = Vec::new();
    for (scene, worsts) in scenes {
        for &(name, worst) in worsts {
            let step = step_budget.filter(|_| name != "GPU total");
            for limit in [budget, step].into_iter().flatten() {
                if worst > limit {
                    over.push(format!("{scene}: {name} {worst:.3} ms over {limit} ms"));
                }
            }
        }
    }
    over
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
    let ms = |flag: &str| after(flag).map(|m| m.parse::<f64>().expect("milliseconds"));
    let budgets = (ms("--budget"), ms("--step-budget"));
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
    // The adapter chosen first: what is timed is the device and the
    // window's pipelines.
    jodel_gpu::chosen().adapter.as_ref().expect("a GPU");
    let started = Instant::now();
    let mut gpu = Offscreen::timed().expect("a GPU that times");
    let made = started.elapsed().as_secs_f64() * 1000.0;
    gpu.set_density(density);
    println!("{}, {density} pixels to a point", gpu.name);
    println!("the device and the window's pipelines made in {made:.1} ms");
    let rest = the_rest();
    let mut worsts = Vec::new();
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
            let (mut kinds, mut effects) = (Alone::default(), Alone::default());
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
                            top: 0.9,
                            tier: Some(Tier::TOP),
                        },
                    );
                }
                let (backdrop, mut frame, cpu) = rig.frame(&singing, now, None);
                if scene.every {
                    let area = (0.0, 0.0, w as f32, h as f32);
                    frame = every_effect(&frame, area, (&rest, scene.at - TIMED));
                }
                if rig.made == 1 {
                    gpu.set_backdrop(Some(&backdrop), w, h);
                    gpu.show_pixels(&rig.pixels, w, h).expect("a frame");
                }
                if !scene.diploma && now > scene.at - TIMED {
                    let alone = Some((&mut kinds, &mut effects));
                    frames.push(time(&mut gpu, &backdrop, &frame, (w, h), alone));
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
                    let alone = (f % 10 == 0).then_some((&mut kinds, &mut effects));
                    frames.push(time(&mut gpu, &backdrop, &frame, (w, h), alone));
                    cpu_ms.push(cpu);
                }
            }
            let title = format!("{} {w}x{h}", scene.name);
            worsts.push((title.clone(), print(&title, &frames, &cpu_ms)));
            kinds.print("the stage, kind by kind");
            effects.print("the effects, one by one");
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
    let over = over(&worsts, budgets);
    for line in &over {
        eprintln!("over budget: {line}");
    }
    if !over.is_empty() {
        std::process::exit(1);
    }
}
