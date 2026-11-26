//! A look the music moves reaches the GPU: a row holding the line sung on
//! fire, its look's first number (classic orange to a ghostly green)
//! moved by the tempo ([`Mod`]), draws the line in another colour than the
//! row without the move, through the looks' uniform alone (the shapes the
//! same), and only where the line is.

use std::path::Path;

use jodeldiplom::core::singing::Singing;
use jodeldiplom::gui::fx::{Effects, Fx, Mod, Param, Signal, Use};
use jodeldiplom::gui::present::Offscreen;
use jodeldiplom::gui::stage::{Song, Stage};
use jodeldiplom::gui::stage_gpu::{kind, StageFrame};

#[path = "common/sung.rs"]
mod sung;
use sung::{picture_at, song, with_effects, Fired, LENGTH};

/// The line on fire, its look as made, or moved all the way to green by
/// the tempo (1 at the free tempo, with no music).
const STILL: [Use; 1] = [Use::of(Fx::LyricFire, 1.0)];
const MOVED: [Use; 1] = [Use {
    mods: &[Mod {
        param: Param::Look(0),
        by: Signal::Tempo,
        amount: 1.0,
    }],
    ..Use::of(Fx::LyricFire, 1.0)
}];

#[test]
fn a_look_moved_by_the_music_reaches_the_lyric_treatment() {
    let mut gpu = match Offscreen::new(false) {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("no GPU ({e}): skipped");
            return;
        }
    };
    let (w, h) = (800u32, 450u32);
    let area = (0.0, 0.0, w as f32, h as f32);
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
    let row = |uses: &'static [Use]| Effects {
        held: uses,
        ..Effects::NONE
    };
    let (still, moved) = (row(&STILL), row(&MOVED));
    let fired = now - 5.0;
    let neutral = Fired::phrase(fired);
    let (a, b) = (
        with_effects(&mut stage, &base, area, &still, neutral),
        with_effects(&mut stage, &base, area, &moved, neutral),
    );
    let fire = Fx::LyricFire.index();
    assert_eq!(a.looks.0[fire][0], 0.0, "as made");
    assert_eq!(b.looks.0[fire][0], 1.0, "moved by the tempo");
    assert_eq!(
        a.instances, b.instances,
        "the same shapes: only the looks differ"
    );
    let on_fire: Vec<[f32; 4]> = (a.instances.iter())
        .filter(|i| i.params[0] == kind::TEXT && i.params[3] == Fx::LyricFire.treatment_code())
        .map(|i| i.quad)
        .collect();
    assert!(!on_fire.is_empty(), "the line sung on fire");
    let mut show = |f: &StageFrame| {
        gpu.set_stage(Some(f), w, h);
        gpu.show_frame(w, h).expect("a frame")
    };
    let (orange, green) = (show(&a), show(&b));
    let mut changed = 0;
    for (n, (p, q)) in orange.iter().zip(&green).enumerate() {
        if p == q {
            continue;
        }
        changed += 1;
        let (x, y) = ((n as u32 % w) as f32, (n as u32 / w) as f32);
        let near = on_fire.iter().any(|r| {
            (r[0] - 4.0..=r[2] + 4.0).contains(&x) && (r[1] - 4.0..=r[3] + 4.0).contains(&y)
        });
        assert!(near, "({x}, {y}) changed away from the line on fire");
    }
    assert!(changed > 200, "the line's colour moved: {changed} pixels");
}
