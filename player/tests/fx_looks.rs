//! What the effects' shaders read besides their shapes reaches the GPU: a
//! look the music moves (a row holding the line sung on fire, its look's
//! flicker speed moved by the tempo, [`Mod`]) and the theme's dress (the
//! same line in Spooky's green fire) each draw the line otherwise than as
//! made, through their uniform alone (the shapes the same), and only
//! where the line is.

use std::path::Path;

use jodel_looks::Driver;
use jodeldiplom::core::singing::Singing;
use jodeldiplom::gui::fx::{Effects, Fx, Mod, Param, Use};
use jodeldiplom::gui::present::Offscreen;
use jodeldiplom::gui::stage::gpu::{Shape, StageFrame};
use jodeldiplom::gui::stage::{Song, Stage};
use jodeldiplom::theme::{CLASSIC, SPOOKY};

#[path = "common/sung.rs"]
mod sung;
use sung::{picture_at, song, with_effects, Fired, LENGTH};

/// The line on fire, its look as made, or flickering twice as fast moved
/// by the tempo (1 at the free tempo, with no music).
const STILL: [Use; 1] = [Use::of(Fx::LyricFire, 1.0)];
const MOVED: [Use; 1] = [Use {
    mods: &[Mod {
        param: Param::Look(1),
        by: Driver::Tempo,
        amount: 1.0,
    }],
    ..Use::of(Fx::LyricFire, 1.0)
}];

/// The window's size the frames are drawn at.
const SIZE: (u32, u32) = (800, 450);

/// The GPU to draw on (none: skipped).
fn gpu() -> Option<Offscreen> {
    Offscreen::new(false)
        .inspect_err(|e| eprintln!("no GPU ({e}): skipped"))
        .ok()
}

/// The stage's frame with the line sung on fire as `uses` hold it, and
/// where the line on fire is.
fn on_fire(uses: &'static [Use]) -> (StageFrame, Vec<[f32; 4]>) {
    let area = (0.0, 0.0, SIZE.0 as f32, SIZE.1 as f32);
    let (lyrics, melody) = song();
    let sung: Vec<&jodel_lyrics::Line> = lyrics.lines.iter().collect();
    let mut singing = Singing::default();
    singing.prepare(
        Path::new("song"),
        &lyrics,
        &jodeldiplom::core::singing::melodic(&melody),
    );
    let now = 9.0;
    let picture = picture_at(&sung, now, &singing);
    let song = Song::new(&singing, sung.clone(), Some(LENGTH));
    let mut stage = Stage::default();
    let base = stage.frame(&song, &picture, area, now, 1.0 / 60.0);
    let row = Effects {
        held: uses,
        ..Effects::NONE
    };
    let frame = with_effects(&mut stage, &base, area, &row, Fired::phrase(now - 5.0));
    let lines: Vec<[f32; 4]> = (frame.instances.iter())
        .filter(|i| i.is(Shape::Text) && i.params[3] == Fx::LyricFire.treatment_code())
        .map(|i| i.quad)
        .collect();
    assert!(!lines.is_empty(), "the line sung on fire");
    (frame, lines)
}

/// `a` and `b` drawn on `gpu`: the pixels that differ, each near one of
/// `lines` (the line on fire), more than a few.
fn only_the_line_differs(
    gpu: &mut Offscreen,
    (a, b): (&StageFrame, &StageFrame),
    lines: &[[f32; 4]],
) {
    let (w, h) = SIZE;
    let mut show = |f: &StageFrame| {
        gpu.set_stage(Some(f), w, h);
        gpu.show_frame(w, h).expect("a frame")
    };
    let (first, second) = (show(a), show(b));
    let mut changed = 0;
    for (n, (p, q)) in first.iter().zip(&second).enumerate() {
        if p == q {
            continue;
        }
        changed += 1;
        let (x, y) = ((n as u32 % w) as f32, (n as u32 / w) as f32);
        let near = lines.iter().any(|r| {
            (r[0] - 4.0..=r[2] + 4.0).contains(&x) && (r[1] - 4.0..=r[3] + 4.0).contains(&y)
        });
        assert!(near, "({x}, {y}) changed away from the line on fire");
    }
    assert!(changed > 200, "the line moved: {changed} pixels");
}

#[test]
fn a_look_moved_by_the_music_reaches_the_lyric_treatment() {
    let Some(mut gpu) = gpu() else {
        return;
    };
    let ((a, lines), (b, _)) = (on_fire(&STILL), on_fire(&MOVED));
    let fire = Fx::LyricFire.index();
    assert_eq!(a.looks.0[fire][1], 1.0, "as made");
    assert_eq!(b.looks.0[fire][1], 2.0, "moved by the tempo");
    assert_eq!(
        a.instances, b.instances,
        "the same shapes: only the looks differ"
    );
    only_the_line_differs(&mut gpu, (&a, &b), &lines);
}

/// The theme's dress reaches the shaders through its uniform: the same
/// shapes in Spooky draw the line on fire green, nothing else otherwise.
#[test]
fn a_theme_s_dress_reaches_the_lyric_treatment() {
    let Some(mut gpu) = gpu() else {
        return;
    };
    let (classic, lines) = on_fire(&STILL);
    assert_eq!(classic.theme, &CLASSIC);
    let spooky = StageFrame {
        theme: &SPOOKY,
        ..classic.clone()
    };
    only_the_line_differs(&mut gpu, (&classic, &spooky), &lines);
}
