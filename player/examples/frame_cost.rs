//! What a frame of Karaoke mode costs, step by step: the backdrop (a cover
//! taken apart, its `.layers` file from the player's data folder, moving
//! with a steady beat) and the
//! stage over it, stepped at 60 frames a second through a made-up song
//! sung on the notes (`tests/common/sung.rs`), drawn offscreen as the
//! window draws them, and each step's GPU time read from timestamps (the
//! window's pass drawn a step a pass; the first, `ground`, the backdrop
//! over the window): a normal frame (the singing at no tier), the lyrics
//! set in the middle (`lyrics`: no lane, as with the notes off), each
//! tier's peak (a scene a tier, named after it: two phrases
//! at it, the second just decided, the streak just reaching its rung of
//! the multiplier; each with every effect its row lists, held, fired by
//! the phrase and by the rung; at Diva's its jackpot flying, angels coming
//! down, the flames, the rays, the rainbow, the coins raining, the words
//! slammed, ×32), the worst the effects can make
//! (`all`: the Diva peak with every registered effect in it at full
//! strength at once, moving with a steady beat, the edges on fire, the
//! bulbs, the coins, a word slammed down too, and the
//! lyrics' lines in every lyric treatment in turn), and the Jodeldiplom
//! after that song (a new best: its confetti), timed over its whole
//! reveal, twice ([`Judge`]). The singing scenes are
//! timed over their second up to the moment looked at, each in every size
//! asked for (`--size WxH`, again for more; the default a 16-inch MacBook
//! Pro's screen, 3024 × 1964), `--density D` pixels to a point (the
//! window's scale; 2, a Retina screen's, by default); each step's median,
//! mean, 99th percentile and worst, the same of the frames' totals, and
//! what the processor spent making each frame's stage and backdrop; which
//! frame was the worst (its number, its time in the scene, the beats that
//! had just landed or the phrase just decided, and how much it drew),
//! and how many frames went over the budgets. The `story` scene is the
//! normal one over the story's scenes instead of the cover (the cover's
//! picture and depth taken for three scenes, each in three layers, a head
//! held flat in the middle: the dearest a scene draws), one cut in and the
//! next scene sent over within the second timed. The GPU must write
//! timestamps (Vulkan, DirectX 12, Metal on Apple's chips). `--png
//! OUT_DIR`: the last frame of each as a PNG (`<scene>-<W>x<H>.png`,
//! written whole), to hold the look to. Each scene's stage is also timed
//! kind by kind (the frame drawn with that kind of shape alone: its stage,
//! screen-pass and top steps) and effect by effect (the frame with that
//! effect's shapes alone, its light and its text in its treatment too: its
//! stage, light, screen-pass and top steps); the diploma's only once its
//! sheet has unrolled, its paper baked (the paper alone is its bake read).
//! First, how long the device and the window's pipelines take to make.
//!
//! `--look ID`: every scene with that look over its backdrop (a preset's
//! id, [`jodel_looks::presets`]; `--look heaviest`: each of the presets
//! that cost the most in turn, two scenes each over the normal scene's
//! singing: over the cover and over the story's scenes), moving with a made-up song at the steady grid's tempo
//! ([`jodel_looks::demo`]); its passes are the `look` step.
//!
//! `--theme NAME`: the game in that theme ([`jodeldiplom::theme`]: its
//! colours, its effects' looks, what it holds throughout and its grade
//! over the backdrop; classic by default). The `all` scene's effects
//! beyond Diva's row are in their own looks either way.
//!
//! `--budget MS`: exits with 1 (after printing it all) where any scene's
//! worst frame took the GPU longer than that in all, or any step did;
//! `--step-budget MS`: where any step's worst did. The diploma's frames
//! are each the lesser of its two runs' ([`Judge::MinOfTwo`]): a hitch
//! at the same moment both times counts, the machine's own in one does
//! not (the budgets to hold to, and on which machine: PERFORMANCE.md).
//!
//! ```text
//! cargo run --release -p jodeldiplom --example frame_cost --
//!     COVER.layers [--size WxH ...] [--density D] [--png OUT_DIR]
//!     [--budget MS] [--step-budget MS] [--theme NAME] [--look ID]
//! ```

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use jodeldiplom::core::karaoke::result::{Reveal, Shown, SongResult, UNROLL};
use jodeldiplom::core::shine::{Reference, Shone, Tier};
use jodeldiplom::core::singing::Singing;
use jodeldiplom::gui::backdrop::looks::LookCue;
use jodeldiplom::gui::backdrop::music::Hearing;
use jodeldiplom::gui::backdrop::scenes::Showing;
use jodeldiplom::gui::backdrop::{BackdropFrame, Cue, Visualiser};
use jodeldiplom::gui::fx::{By, Effects, Fx, Music, Treat, Use};
use jodeldiplom::gui::present::Offscreen;
use jodeldiplom::gui::stage::gpu::{Instance, Shape, StageFrame};
use jodeldiplom::gui::stage::{Song, Stage, Tint};
use jodeldiplom::theme::Theme;
use jodeldiplom::view::karaoke::{KaraokePicture, ResultShown};
use jodeldiplom::work::art::Art;
use jodeldiplom::work::cover::{Apart, CoverLayers, Layers};
use jodeldiplom::work::picture::Planes;
use jodeldiplom::work::scenes::{alphas, half, Cuts, Derived, Head, Ready, Split, Story};

#[path = "../tests/common/sung.rs"]
mod sung;
use sung::{cover_of, levels, picture_at, song, with_effects, Fired, FPS, LENGTH};

// The fonts build.rs found: `EMBEDDED_FONTS`.
include!(concat!(env!("OUT_DIR"), "/fonts.rs"));

/// How long before the moment looked at each scene is timed.
const TIMED: f64 = 1.0;

/// The moment each scene is looked at, and when its two phrases are
/// decided.
const AT: f64 = 10.0;
const DECIDED: [f64; 2] = [4.0, 9.2];

/// When a tier's scene sets its streak a word short of the tier's rung
/// (none before): the next word hit (scored 9.5 s in) reaches the rung
/// with its moment, so what reaching it fires is in the second timed.
const CLIMBS: f64 = AT - TIMED;

/// A scene: its name, the tier its phrases are decided at (none: no
/// phrase decided; the streak reaching the tier's rung: [`CLIMBS`]), whether
/// the song's diploma is timed from the moment looked at (its whole
/// reveal) rather than the singing up to it, whether every effect is in
/// its frames at once ([`every_effect`]), and how its frames are judged
/// against the budgets; whether the stage draws the song's lane (else the
/// lyrics set in the middle, as with the notes off); the look over its
/// backdrop (none: none); whether the story's scenes show instead of the
/// cover.
struct Scene {
    name: String,
    tier: Option<Tier>,
    diploma: bool,
    every: bool,
    judge: Judge,
    lane: bool,
    look: Option<Arc<jodel_looks::Checked>>,
    story: bool,
}

/// How many of the dearest presets `--look heaviest` times.
const HEAVIEST: usize = 4;

/// The looks `--look` asks for (none: none): a preset by its id, or the
/// [`HEAVIEST`] presets that cost the most (`heaviest`), each made safe
/// as the player makes it.
fn looks_asked(asked: Option<&str>) -> Option<Vec<Arc<jodel_looks::Checked>>> {
    let asked = asked?;
    if asked == "heaviest" {
        let mut all: Vec<jodel_looks::Checked> = jodel_looks::presets::all()
            .iter()
            .map(|l| l.check())
            .collect();
        all.sort_by(|a, b| b.cost().total_cmp(&a.cost()).then(a.id.cmp(&b.id)));
        return Some(all.into_iter().take(HEAVIEST).map(Arc::new).collect());
    }
    let look = jodel_looks::presets::named(asked).unwrap_or_else(|| panic!("no preset {asked:?}"));
    Some(vec![Arc::new(look.check())])
}

/// The scenes: the normal one, the normal one over the story's scenes, the
/// lyrics set in the middle (no lane), a
/// tier's each, every effect at once and the diploma (at the top, judged by
/// the lesser of two runs).
fn scenes() -> Vec<Scene> {
    let scene = |name: &str, tier, diploma, every| Scene {
        name: name.to_owned(),
        tier,
        diploma,
        every,
        judge: Judge::Worst,
        lane: true,
        look: None,
        story: false,
    };
    let story = Scene {
        story: true,
        ..scene("story", None, false, false)
    };
    let lyrics = Scene {
        lane: false,
        ..scene("lyrics", None, false, false)
    };
    let tiers =
        (Tier::ALL.into_iter()).map(|t| scene(&t.name().to_lowercase(), Some(t), false, false));
    let diploma = Scene {
        judge: Judge::MinOfTwo,
        ..scene("diploma", Some(Tier::TOP), true, false)
    };
    [scene("normal", None, false, false), story, lyrics]
        .into_iter()
        .chain(tiers)
        .chain([scene("all", Some(Tier::TOP), false, true), diploma])
        .collect()
}

/// The effects Diva's row does not list nor `theme` holds, each at full
/// strength, held and fired by a phrase: with Diva's own and the theme's,
/// every registered effect at once.
fn the_rest(theme: &Theme) -> Effects {
    let diva = &Tier::TOP.spec().fx;
    let rest: Vec<Use> = (Fx::ALL.into_iter())
        .filter(|&fx| !diva.lists_fx(fx) && !theme.holds(fx))
        .map(|fx| Use::of(fx, 1.0))
        .collect();
    let rest: &'static [Use] = Vec::leak(rest);
    Effects {
        held: rest,
        on_phrase: rest,
        ..Effects::NONE
    }
}

/// The music the effects move with in the `all` scene: a steady grid at
/// 120 a minute, bars of four, sure, `now` seconds in (a bar's first beat
/// a quarter of a second after each two seconds).
fn steady(now: f64) -> Music {
    Music::steady(now * 2.0 + 0.5, 120.0, 4, 1.0, now)
}

/// `frame` (the singing at Diva, over the stage in `area`, made by
/// `stage`) with every effect in it at full strength ([`the_rest`], fired
/// at `fired`, moving with a steady grid's music: [`steady`]), the lyrics'
/// lines in every lyric treatment in turn (each its own pipeline drawing):
/// the dearest frame the effects can make.
fn every_effect(
    stage: &mut Stage,
    frame: &StageFrame,
    area: (f32, f32, f32, f32),
    (rest, fired): (&Effects, f64),
) -> StageFrame {
    let fired = Fired {
        at: fired,
        music: &steady,
        by: By::Phrase(0),
    };
    let mut f = with_effects(stage, frame, area, rest, fired);
    let codes: Vec<f32> = (Fx::ALL.into_iter())
        .filter(|fx| fx.spec().lyric.is_some())
        .map(Fx::treatment_code)
        .collect();
    let lyric = |i: &Instance| i.is(Shape::Text) && i.params[1].is_finite();
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
    let mut names: Vec<&'static str> = all.iter().map(|i| Shape::name_of(i.params[0])).collect();
    names.sort_unstable();
    names.dedup();
    let mut fxs: Vec<Fx> = all.iter().filter_map(effect_of).collect();
    fxs.sort_unstable();
    fxs.dedup();
    for &name in &names {
        let f = only(&|i| Shape::name_of(i.params[0]) == name);
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
            &|i| Shape::name_of(i.params[0]) == name,
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
/// What makes a scene's frames in one size: the stage and the backdrop
/// under it, as the window keeps them.
struct Rig<'a> {
    stage: Stage,
    visualiser: Visualiser,
    art: &'a Art,
    layers: &'a Layers,
    sung: &'a [&'a jodel_lyrics::Line],
    size: (u32, u32),
    /// Whether the stage draws the song's lane.
    lane: bool,
    /// The game's theme.
    theme: &'static Theme,
    /// Frames made so far.
    made: usize,
    /// When the last phrase seen ended.
    phrase: Option<f64>,
    /// The look over the backdrop, and the made-up song it moves with.
    look: Option<(Arc<jodel_looks::Checked>, Arc<jodel_looks::Curves>)>,
    /// The story's scenes instead of the cover (none: the cover).
    story: Option<Arc<Showing>>,
}

impl Rig<'_> {
    /// The frame at `now` (`result`: the diploma shown over it), and what
    /// the processor took for its stage, ms.
    fn frame(
        &mut self,
        singing: &Singing,
        now: f64,
        result: Option<ResultShown>,
    ) -> (BackdropFrame, StageFrame, f64) {
        let (w, h) = self.size;
        self.made += 1;
        let started = Instant::now();
        // A phrase landing flares the subject; the backdrop moves with the
        // steady grid's beat.
        let last = singing.last_phrase().map(|p| p.end);
        let phrase = self.made > 1 && last.is_some() && last != self.phrase;
        self.phrase = last;
        let cue = Cue {
            art: Some(self.art.clone()),
            layers: Some(self.layers.clone()),
            heard: Hearing {
                levels: levels(),
                glow: jodeldiplom::gui::streak::level(singing, true),
                section: false,
                phrase,
                music: steady(now),
            },
            grade: self.theme.grade,
            look: self.look.as_ref().map(|(look, curves)| LookCue {
                look: Arc::clone(look),
                strength: 1.0,
                t: now,
                signals: curves.at(now),
                curves: Some(Arc::clone(curves)),
                seed: jodel_looks::seed(&look.id, "frame_cost"),
                band: Some((0.68, 0.86)),
            }),
            story: self.story.clone(),
            at: now,
        };
        let backdrop = self.visualiser.step(&cue, (w, h), (1.0 / FPS) as f32);
        let picture = KaraokePicture {
            result,
            ..picture_at(self.sung, now, singing)
        };
        let song = Song {
            theme: self.theme,
            ..Song::new(singing, self.sung.to_vec(), Some(LENGTH))
        };
        let area = (0.0, 0.0, w as f32, h as f32);
        // As the window has it: the result screen while one shows, else
        // the lane (or the lyrics set in the middle) and the score.
        let frame = self
            .stage
            .scene(
                (&song, self.lane),
                &picture,
                area,
                (now, (1.0 / FPS) as f32),
                &Tint::STAGE,
            )
            .expect("a stage");
        let stage_ms = started.elapsed().as_secs_f64() * 1000.0;
        (backdrop, frame, stage_ms)
    }
}

/// The story's scenes the `story` scene shows: the cover's `picture` and
/// its depth (of its `layers`) three times, each cut in three layers with a
/// head held flat in the middle, the cover's plate behind; the second cut
/// in 9.4 s into the song, the third sent over just after (within the
/// second timed).
fn story_of(layers: &CoverLayers, picture: &image::RgbImage) -> Arc<Story> {
    let near = Planes::from(&layers.depth);
    let cuts = Cuts::Three(0.35, 0.65);
    let parts = Derived {
        picture: "cover".into(),
        near: layers.depth.clone(),
        heads: vec![Head {
            face: [0.42, 0.3, 0.16, 0.2],
            near: 0.8,
        }],
        split: Split::Layers {
            cuts,
            alphas: alphas(&near, &cuts, half(picture.dimensions()), None),
            plate: image::imageops::resize(
                &layers.plate,
                picture.width(),
                picture.height(),
                image::imageops::FilterType::Triangle,
            ),
        },
        palette: layers.palette.clone(),
    };
    let scenes = [(0.0, 9.4), (9.4, 12.3), (12.3, LENGTH)]
        .map(|span| Ready::new(span, picture, &parts))
        .to_vec();
    let hashes = ["cover".to_owned(), "cover".to_owned(), "cover".to_owned()];
    let look = jodel_looks::presets::all()[0].clone();
    Arc::new(Story::new(&hashes, scenes, look))
}

/// How a scene's frames are judged against the budgets: by its worst
/// frame, or (a scene that can be shown again: the diploma's reveal) timed
/// twice and each frame's steps taken at the lesser of the two runs', its
/// worst frame judged then: a hitch at the same moment in both runs stays,
/// the machine's own, in one, goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Judge {
    Worst,
    MinOfTwo,
}

impl Judge {
    /// How many times its frames are timed.
    fn runs(self) -> usize {
        match self {
            Judge::Worst => 1,
            Judge::MinOfTwo => 2,
        }
    }
}

/// What a frame drew: its shapes, its bursts, its light (shapes and
/// bursts) and its text put into the atlas.
#[derive(Clone, Copy, Debug)]
struct Drew {
    shapes: usize,
    bursts: usize,
    lights: usize,
    uploads: usize,
}

impl Drew {
    fn of(frame: &StageFrame) -> Drew {
        Drew {
            shapes: frame.instances.len() + frame.top.len() + frame.screen.len(),
            bursts: frame.sprays.len(),
            lights: frame.lights.len() + frame.light_sprays.len(),
            uploads: frame.uploads.len(),
        }
    }
}

/// A frame timed: its steps' GPU times and what the processor took for
/// its stage (ms), its time in the scene (s) and what it drew.
struct Timed {
    steps: Vec<(&'static str, f64)>,
    cpu: f64,
    at: f64,
    drew: Drew,
}

impl Timed {
    fn total(&self) -> f64 {
        self.steps.iter().map(|s| s.1).sum()
    }
}

/// The same frames timed in two runs: each step of each frame (and the
/// processor's times) the lesser of the two runs'.
fn lesser(a: Vec<Timed>, b: &[Timed]) -> Vec<Timed> {
    (a.into_iter().zip(b))
        .map(|(mut a, b)| {
            for (s, t) in a.steps.iter_mut().zip(&b.steps) {
                s.1 = s.1.min(t.1);
            }
            a.cpu = a.cpu.min(b.cpu);
            a
        })
        .collect()
}

/// A series' 99th percentile (the nearest rank) and worst.
#[derive(Clone, Copy, Debug, Default)]
struct Spread {
    p99: f64,
    worst: f64,
}

/// The `frames` timed: each step's median, mean, 99th percentile and
/// worst, the same of the GPU's totals and of the processor's times, in
/// milliseconds; the worst frame (its number and, from `moment` at its
/// time, what was on screen then) and how many frames went over the
/// `budgets` ([`over`]'s). Gives back each step's spread and the totals'
/// (`GPU total`).
fn print(
    title: &str,
    frames: &[Timed],
    moment: &dyn Fn(f64) -> String,
    (budget, step_budget): (Option<f64>, Option<f64>),
) -> Vec<(&'static str, Spread)> {
    let stats = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        let median = v.get(v.len() / 2).copied().unwrap_or(0.0);
        let rank = (v.len() as f64 * 0.99).ceil() as usize;
        let spread = Spread {
            p99: v.get(rank.saturating_sub(1)).copied().unwrap_or(0.0),
            worst: v.last().copied().unwrap_or(0.0),
        };
        let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
        println!(
            "{median:7.3} {mean:7.3} {:7.3} {:7.3}",
            spread.p99, spread.worst
        );
        spread
    };
    println!(
        "{title}: {} frames, ms (median / mean / p99 / worst)",
        frames.len()
    );
    let names = frames
        .first()
        .map_or(Vec::new(), |f| f.steps.iter().map(|s| s.0).collect());
    let mut spreads = Vec::new();
    for name in names {
        let v = (frames.iter())
            .flat_map(|f| f.steps.iter().filter(|s| s.0 == name).map(|s| s.1))
            .collect();
        print!("  {name:<10} ");
        spreads.push((name, stats(v)));
    }
    print!("  {:<10} ", "GPU total");
    spreads.push((
        "GPU total",
        stats(frames.iter().map(Timed::total).collect()),
    ));
    print!("  {:<10} ", "CPU stage");
    stats(frames.iter().map(|f| f.cpu).collect());
    let worst = (frames.iter().enumerate()).max_by(|a, b| a.1.total().total_cmp(&b.1.total()));
    if let Some((n, f)) = worst {
        let d = f.drew;
        println!(
            "  worst frame: #{n}, {}: {} shapes, {} bursts, {} lights, {} text uploads",
            moment(f.at),
            d.shapes,
            d.bursts,
            d.lights,
            d.uploads
        );
    }
    if let Some(budget) = budget {
        let over = frames.iter().filter(|f| f.total() > budget).count();
        println!("  {over} frames over {budget} ms in all");
    }
    if let Some(limit) = step_budget {
        let over = (frames.iter())
            .filter(|f| f.steps.iter().any(|s| s.1 > limit))
            .count();
        println!("  {over} frames with a step over {limit} ms");
    }
    spreads
}

/// What went over the budgets (`--budget`: the frames' GPU total and
/// every step; `--step-budget`: every step), scene by scene, each scene's
/// worst frame (of the series its [`Judge`] makes): each as a line saying
/// what, by how much.
fn over(
    scenes: &[(String, Vec<(&'static str, Spread)>)],
    (budget, step_budget): (Option<f64>, Option<f64>),
) -> Vec<String> {
    let mut over = Vec::new();
    for (scene, spreads) in scenes {
        for &(name, spread) in spreads {
            let step = step_budget.filter(|_| name != "GPU total");
            for limit in [budget, step].into_iter().flatten() {
                if spread.worst > limit {
                    let worst = spread.worst;
                    over.push(format!("{scene}: {name} {worst:.3} ms over {limit} ms"));
                }
            }
        }
    }
    over
}

/// When `now` is into the song (its phrases decided at `tier`, none:
/// none decided, [`DECIDED`]).
fn in_song(now: f64, tier: Option<Tier>) -> String {
    let decided = tier.and_then(|tier| {
        let d = DECIDED.iter().rev().find(|&&d| d <= now + 1e-9)?;
        Some(format!(
            ", {:.2} s after a phrase decided {}",
            now - d,
            tier.name()
        ))
    });
    format!("{now:.3} s into the song{}", decided.unwrap_or_default())
}

/// When `t` is into the diploma's `reveal`: the beats that landed in the
/// half second before.
fn in_reveal(reveal: &Reveal, t: f64) -> String {
    let beats: Vec<String> = (reveal.between(t - 0.5, t))
        .map(|(at, beat)| format!("{beat:?} {:.2} s before", t - at))
        .collect();
    let landed = if beats.is_empty() {
        "no beat in the half second before".to_owned()
    } else {
        beats.join(", ")
    };
    format!("{t:.3} s into the reveal, {landed}")
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
    let theme = after("--theme").map_or(&jodeldiplom::theme::CLASSIC, |name| {
        Theme::named(name).unwrap_or_else(|| panic!("no theme named {name:?}"))
    });
    let bytes = std::fs::read(cover).expect("the layers");
    let layers = jodeldiplom::work::cover::from_bytes(&bytes)
        .expect("layers")
        .layers;
    let art = Art {
        id: 1,
        image: Arc::new(cover_of(&layers)),
    };
    let key = Path::new(cover)
        .file_stem()
        .map_or(String::new(), |s| s.to_string_lossy().into_owned());
    let story = story_of(&layers, &art.image);
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
    println!(
        "{}, {density} pixels to a point, the {} theme",
        gpu.name, theme.name
    );
    println!("the device and the window's pipelines made in {made:.1} ms");
    let rest = the_rest(theme);
    let curves = Arc::new(jodel_looks::demo::Song::new(LENGTH, 120.0).curves());
    let looks = looks_asked(after("--look").map(String::as_str));
    let scenes: Vec<Scene> = match &looks {
        None => scenes(),
        Some(looks) if looks.len() == 1 => scenes()
            .into_iter()
            .map(|s| Scene {
                look: Some(Arc::clone(&looks[0])),
                ..s
            })
            .collect(),
        Some(looks) => looks
            .iter()
            .flat_map(|l| {
                scenes().into_iter().take(2).map(|s| Scene {
                    name: format!("{} look {}", s.name, l.id),
                    look: Some(Arc::clone(l)),
                    ..s
                })
            })
            .collect(),
    };
    let mut worsts = Vec::new();
    for &(w, h) in &sizes {
        for scene in &scenes {
            let mut singing = Singing::default();
            singing.prepare(
                Path::new("song"),
                &lyrics,
                &jodeldiplom::core::singing::melodic(&melody),
            );
            // A reference to sing against: the meter shows, as in a game.
            let phrases = vec![None; singing.notes.lines.len()];
            singing.set_reference(Some(Arc::new(Reference { phrases })));
            let mut rig = Rig {
                stage: Stage::default(),
                visualiser: Visualiser::default(),
                art: &art,
                layers: &layers,
                sung: &sung,
                size: (w, h),
                lane: scene.lane,
                theme,
                made: 0,
                phrase: None,
                look: scene.look.clone().map(|l| (l, Arc::clone(&curves))),
                story: scene
                    .story
                    .then(|| Arc::new(Showing::new(Arc::clone(&story), None))),
            };
            let mut frames = Vec::new();
            let (mut kinds, mut effects) = (Alone::default(), Alone::default());
            let mut heard = 0usize;
            let mut climbing = false;
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
                if let Some(tier) = scene.tier {
                    if DECIDED.iter().any(|d| (now - d).abs() < 0.5 / FPS) {
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
                    let streak = tier.spec().streak;
                    if now < CLIMBS {
                        singing.tally.streak = 0;
                    } else if !climbing {
                        singing.tally.streak = streak - 1;
                        climbing = true;
                    }
                    singing.tally.best_streak = singing.tally.best_streak.max(streak);
                }
                let (backdrop, mut frame, cpu) = rig.frame(&singing, now, None);
                if scene.every {
                    let area = (0.0, 0.0, w as f32, h as f32);
                    let fired = (&rest, AT - TIMED);
                    frame = every_effect(&mut rig.stage, &frame, area, fired);
                }
                if !scene.diploma && now > AT - TIMED {
                    let alone = Some((&mut kinds, &mut effects));
                    let steps = time(&mut gpu, &backdrop, &frame, (w, h), alone);
                    let drew = Drew::of(&frame);
                    frames.push(Timed {
                        steps,
                        cpu,
                        at: now,
                        drew,
                    });
                } else {
                    gpu.set_backdrop(Some(&backdrop), w, h);
                    gpu.set_stage(Some(&frame), w, h);
                }
            }
            let mut revealed = None;
            if scene.diploma {
                // The song over: its result, a first score (a new best),
                // revealed frame by frame until it is still, as many times
                // as it is judged by.
                let (tally, phrases, notes) = singing.take();
                let result = SongResult::new("Shine on".into(), &tally, phrases, notes, None);
                let shown = Arc::new(Shown::new(result, 1, 0));
                let still = shown.reveal.still();
                for run in 0..scene.judge.runs() {
                    // Shown afresh each time, as after a song: the stage
                    // gone (its paper's bake let go) and back, its bake's
                    // texture made then, before the reveal.
                    gpu.set_stage(None, w, h);
                    let (_, lane, _) = rig.frame(&singing, AT, None);
                    gpu.set_stage(Some(&lane), w, h);
                    let mut timed = Vec::new();
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
                        let (backdrop, frame, cpu) = rig.frame(&singing, AT, Some(result));
                        // Kind by kind every tenth frame of the first run
                        // (the reveal is long), once the sheet has unrolled
                        // and its paper is baked: the paper alone is its
                        // bake read.
                        let alone = run == 0 && f % 10 == 0 && t >= UNROLL + 0.1;
                        let alone = alone.then_some((&mut kinds, &mut effects));
                        let steps = time(&mut gpu, &backdrop, &frame, (w, h), alone);
                        let drew = Drew::of(&frame);
                        timed.push(Timed {
                            steps,
                            cpu,
                            at: t,
                            drew,
                        });
                    }
                    frames = if run == 0 {
                        timed
                    } else {
                        lesser(frames, &timed)
                    };
                }
                revealed = Some(shown);
            }
            let judged = match scene.judge {
                Judge::Worst => "",
                Judge::MinOfTwo => ", each frame the lesser of two runs",
            };
            let title = format!("{} {w}x{h}{judged}", scene.name);
            let moment = |at: f64| match &revealed {
                Some(shown) => in_reveal(&shown.reveal, at),
                None => in_song(at, scene.tier),
            };
            let spreads = print(&title, &frames, &moment, budgets);
            worsts.push((title, spreads));
            kinds.print("the stage, kind by kind");
            effects.print("the effects, one by one");
            if let Some(dir) = &png {
                let shown = gpu.show_frame(w, h).expect("a frame");
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
