//! What bursting bubbles cost the processor (and the GPU, where there is
//! one): Karaoke mode's picture painted frame by frame at 1600 × 900 with
//! 20 bubbles holding liquid and with 20 bursting at once (and as the
//! GPU draws it, the stage setting the lyrics in the middle), and the GPU's
//! stage made frame by frame (and shown, offscreen) with none and with 20
//! words filling and bursting together.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example burst_load
//! ```

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use jodeldiplom::core::singing::{Note, Singing};
use jodeldiplom::gui::big_text::Faces;
use jodeldiplom::gui::karaoke::{paint, Backdrop, Layer};
use jodeldiplom::gui::pixels::Target;
use jodeldiplom::gui::stage::{Song, Stage, Tint};
use jodeldiplom::view::bubbles::{Liquid, Part};
use jodeldiplom::view::karaoke::{KaraokePicture, LineKind, LyricLine};

const FRAMES: usize = 120;
const WORDS: usize = 20;

fn main() {
    let (w, h) = (1600usize, 900usize);
    println!("{FRAMES} frames at {w} × {h}, {WORDS} words; ms a frame (median, 90 %):");
    let shown = |gpu: Option<(f64, f64)>| {
        gpu.map_or_else(
            || "no GPU".to_owned(),
            |(m, p)| format!("{m:.3} ms, {p:.3} ms"),
        )
    };
    for bursting in [false, true] {
        let ((median, p90), gpu) = picture(w, h, bursting);
        println!(
            "  picture, {:<12} CPU {median:.3} ms, {p90:.3} ms · shown offscreen (with read-back) {}",
            if bursting { "20 bursting" } else { "20 liquid" },
            shown(gpu)
        );
    }
    for (bursting, shine, name) in [
        (false, false, "none"),
        (true, false, "20 bursting"),
        (true, true, "20 bursting, top shine, a step up every 0.25 s"),
    ] {
        let (cpu, gpu) = stage(w, h, bursting, shine);
        let gpu = shown(gpu);
        println!(
            "  stage,   {name:<12} CPU {:.3} ms, {:.3} ms · shown offscreen (with read-back) {gpu}",
            cpu.0, cpu.1
        );
    }
}

fn quantiles(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(f64::total_cmp);
    (v[v.len() / 2], v[(v.len() - 1) * 9 / 10])
}

/// The picture's line of 20 words, each on a note, all bursting (or none)
/// as the frames go by; each frame painted where it changed.
fn picture(w: usize, h: usize, bursting: bool) -> ((f64, f64), Option<(f64, f64)>) {
    let text = vec!["la"; WORDS].join(" ");
    let notes: Vec<Note> = (0..WORDS)
        .map(|k| Note {
            from: k as f32 * 3.0,
            to: k as f32 * 3.0 + 2.0,
            height: ((k % 5) as f32 - 2.0) / 2.0,
            start: k as f64,
            end: k as f64 + 1.0,
            cents: 6000.0,
            word: k,
        })
        .collect();
    let backdrop = Backdrop::new(None, w, h);
    let (mut text_big, mut layer) = (Faces::default(), Layer::default());
    let mut pixels = vec![0u32; w * h];
    let (mut took, mut shown) = (Vec::new(), Vec::new());
    let mut gpu = jodeldiplom::gui::present::Offscreen::new(false).ok();
    let (mut staged, nobody, blank) = (Stage::default(), Singing::default(), vec![0u32; w * h]);
    for f in 0..FRAMES {
        let age = (f as f32 / 60.0 / 0.5).fract();
        let line = LyricLine {
            text: text.clone(),
            fill: 1.0,
            kind: LineKind::Current,
            id: Some(0),
            alpha: 1.0,
            notes: notes.clone(),
            hits: vec![(0.0, text.chars().count() as f32)],
            // Each word's bubble (one a note here) bursting, or with its
            // liquid in it (60 %).
            liquid: notes
                .iter()
                .enumerate()
                .map(|(k, n)| {
                    let level = if bursting { 1.0 } else { 0.6 };
                    Liquid {
                        word: k,
                        parts: vec![Part {
                            note: k,
                            from: n.from,
                            to: n.to,
                            time: (n.start, n.end),
                            height: n.height,
                            level,
                            jumped: f64::NEG_INFINITY,
                        }],
                        time: (n.start, n.end),
                        level,
                        pops_at: bursting.then_some(f as f64 / 60.0 - f64::from(age) * 0.5),
                        age: bursting.then_some(f64::from(age) * 0.5),
                        held: 0.0,
                        pumped: None,
                        seed: k as u32,
                    }
                })
                .collect(),
            trail: Vec::new(),
            voice: None,
        };
        let picture = KaraokePicture {
            bar: 5.0 / 24.0,
            lines: vec![line],
            count: None,
            cue: None,
            controls: None,
            start: None,
            before: Vec::new(),
            moved: 1.0,
            score: None,
            result: None,
        };
        let mut target = Target {
            pixels: &mut pixels,
            width: w,
            height: h,
        };
        let started = Instant::now();
        paint(
            &mut target,
            &mut text_big,
            &mut layer,
            &backdrop,
            &picture,
            (0, 0, w, h),
            None,
            f == 0,
        );
        if f > 0 {
            took.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        // As the GPU draws it: the stage's lyrics set in the middle, its
        // bubbles' liquid and bursts over their bars.
        if let Some(gpu) = gpu.as_mut() {
            let song = Song::new(&nobody, Vec::new(), None);
            let area = (0.0, 0.0, w as f32, h as f32);
            let at = (f as f64 / 60.0, 1.0 / 60.0);
            let frame = staged.scene((&song, false), &picture, area, at, &Tint::STAGE);
            let started = Instant::now();
            gpu.set_stage(frame.as_ref(), w as u32, h as u32);
            let _ = gpu.show_pixels(&blank, w as u32, h as u32);
            shown.push(started.elapsed().as_secs_f64() * 1000.0);
        }
    }
    (
        quantiles(took),
        (!shown.is_empty()).then(|| quantiles(shown)),
    )
}

/// The stage of a line of 20 short words sung all through (each
/// counts at once: they burst together), or not sung; frames at 60 a
/// second through the bursts. With the streak's `shine` at its top, a step
/// up every 15 frames (the tally set as the scoring would).
fn stage(w: usize, h: usize, bursting: bool, shine: bool) -> ((f64, f64), Option<(f64, f64)>) {
    let word = |k: usize| jodel_lyrics::Word {
        start: Some(1.0 + k as f64 * 0.04),
        end: Some(1.0 + (k + 1) as f64 * 0.04),
        confidence: 0.9,
        ..jodel_lyrics::Word::new("la")
    };
    let lyrics = Arc::new(jodel_lyrics::Lyrics {
        lines: vec![jodel_lyrics::Line {
            words: (0..WORDS).map(word).collect(),
            ..jodel_lyrics::Line::default()
        }],
        source: jodel_lyrics::Source::Unknown,
        timing: jodel_lyrics::Timing::default(),
    });
    let mut melody = jodel_rhythm::waveform::Melody {
        cents: vec![0; 400],
        sure: vec![0; 400],
    };
    for i in 100..200 {
        melody.cents[i] = 6000 + 100 * ((i / 4) % 3) as u16;
        melody.sure[i] = 255;
    }
    let mut singing = Singing::default();
    singing.prepare(
        Path::new("load"),
        &lyrics,
        &jodeldiplom::core::singing::melodic(&melody),
    );
    let mut gpu = jodeldiplom::gui::present::Offscreen::new(false).ok();
    let blank = vec![0u32; w * h];
    let mut stage = Stage::default();
    let (mut cpu, mut shown) = (Vec::new(), Vec::new());
    let mut heard = 0usize;
    for f in 0..FRAMES {
        let now = 0.9 + f as f64 / 60.0;
        while (heard as f64) / 100.0 <= now {
            let t = heard as f64 / 100.0;
            let c = singing
                .word_at(t)
                .map(|(_, c)| c as f32)
                .filter(|_| bursting);
            singing.hear(t, c);
            heard += 1;
        }
        if shine {
            singing.tally.streak = 20;
            if f % 15 == 0 {
                singing.tally.last_scored = Some(jodeldiplom::core::singing::Scored {
                    end: now,
                    hit: true,
                    points: 40,
                    streak: 20,
                });
            }
        }
        let song = Song::new(&singing, lyrics.lines.iter().collect(), Some(4.0));
        let picture = KaraokePicture {
            bar: 5.0 / 24.0,
            lines: Vec::new(),
            count: None,
            cue: None,
            controls: None,
            start: None,
            before: Vec::new(),
            moved: 1.0,
            score: None,
            result: None,
        };
        let started = Instant::now();
        let frame = stage.frame(
            &song,
            &picture,
            (0.0, 0.0, w as f32, h as f32),
            now,
            1.0 / 60.0,
        );
        cpu.push(started.elapsed().as_secs_f64() * 1000.0);
        if let Some(gpu) = gpu.as_mut() {
            let started = Instant::now();
            gpu.set_stage(Some(&frame), w as u32, h as u32);
            let _ = gpu.show_pixels(&blank, w as u32, h as u32);
            shown.push(started.elapsed().as_secs_f64() * 1000.0);
        }
    }
    let full = jodeldiplom::core::bubbles::words_full(&singing, f64::INFINITY).0;
    println!("  (stage: {full} words full)");
    (
        quantiles(cpu),
        (!shown.is_empty()).then(|| quantiles(shown)),
    )
}
