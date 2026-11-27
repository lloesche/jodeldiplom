//! The backdrop by the game's setting: without it the blurred cover alone,
//! with it the story's scenes (the plain ground before the song's story is
//! there, a scene all through the song once it is), graded by the theme
//! of its frame, the look over it taking the scene's palette; and on the
//! GPU the blurred cover, the plain ground and the scene drawn, nothing
//! held once it is gone.

use std::sync::Arc;

use super::scenes::Showing;
use super::*;
use crate::work::scenes::{Moved, Ready, Story};

const DT: f32 = 1.0 / 60.0;
const WINDOW: (u32, u32) = (320, 180);

/// A story of two scenes, each a picture of one `colours`' colour (the
/// first from 10 s to 20 s, the second from 25 s to 40 s), whole (no
/// layers), its palette its colour.
fn story(colours: [[u8; 3]; 2]) -> Arc<Showing> {
    let (w, h) = (32u32, 18u32);
    let scene = |span, rgb: [u8; 3]| Ready {
        span,
        size: (w, h),
        picture: (0..w * h)
            .flat_map(|_| [rgb[0], rgb[1], rgb[2], 255])
            .collect(),
        shape_size: (w / 2, h / 2),
        shape: vec![128; (w / 2 * h / 2 * 4) as usize],
        moved: Moved::Warp,
        focus: [0.5, 0.5],
        palette: vec![Swatch { rgb, share: 1.0 }],
    };
    let scenes = vec![
        scene((10.0, 20.0), colours[0]),
        scene((25.0, 40.0), colours[1]),
    ];
    let story = Story::new(&["a".into(), "b".into()], scenes);
    Arc::new(Showing::new(Arc::new(story), None))
}

/// A song's cover, `rgb` all over.
fn art(rgb: [u8; 3]) -> Art {
    Art {
        id: 7,
        image: Arc::new(RgbImage::from_pixel(64, 64, image::Rgb(rgb))),
    }
}

/// A cue with the story showing at `at` seconds, `story` (none: not there
/// yet).
fn cue(story: Option<&Arc<Showing>>, at: f64) -> Cue {
    Cue {
        shows: Shows::Story(story.cloned()),
        at,
        ..Cue::default()
    }
}

/// A cue with `art` blurred showing, at 15 s.
fn blurred(art: &Art) -> Cue {
    Cue {
        shows: Shows::Blur(Some(art.clone())),
        at: 15.0,
        ..Cue::default()
    }
}

/// Before the song's story is there, the plain ground (no scene); once it
/// is, a scene all through the song: the first from the start and on over
/// the gap till the second's cut, the second on after its end.
#[test]
fn the_backdrop_is_the_story_s_scenes_and_the_plain_ground_before_it_comes() {
    let mut v = Visualiser::default();
    let plain = v.step(&cue(None, 12.0), WINDOW, DT);
    assert!(plain.scene.is_none() && plain.picture(WINDOW).is_none());
    assert!(plain.palette().is_empty());
    assert_eq!(plain.ground.of, (0, WINDOW), "the gradient");
    let s = story([[200, 40, 40], [40, 40, 200]]);
    let drawn = |v: &mut Visualiser, at: f64| {
        let f = v.step(&cue(Some(&s), at), WINDOW, DT);
        f.scene.expect("a scene").drawn.0
    };
    assert_eq!(
        [0.0, 15.0, 22.0, 25.0, 30.0, 400.0].map(|t| drawn(&mut v, t)),
        [0, 0, 0, 1, 1, 1]
    );
    let f = v.step(&cue(Some(&s), 15.0), WINDOW, DT);
    assert_eq!(f.picture(WINDOW), Some(((320.0, 180.0), (32, 18))));
}

/// The game on, where stories are made: the story's scenes (over the plain
/// gradient, the cover not blurred under them); off, with the notes or
/// without (plain lyrics karaoke), or where no stories are made (no
/// story's models, no data folder): the blurred cover alone, no scene.
/// Switched mid-song, the next frame cuts to the other.
#[test]
fn plain_karaoke_has_the_blur_alone_and_the_game_its_story() {
    use crate::core::{Job, State};
    let shown = |game, notes, made: bool| {
        let mut state = State::new("/music".into(), crate::config::Config::default());
        (state.config.karaoke_game, state.config.karaoke_notes) = (game, notes);
        if made {
            state.preparing.push(Job::Story);
        }
        state.show().story
    };
    assert!(shown(true, true, true) && shown(true, false, true));
    assert!(!shown(false, true, true) && !shown(false, false, true));
    assert!(!shown(true, true, false), "no stories made here: the blur");
    let s = story([[200, 40, 40], [40, 40, 200]]);
    let red = art([200, 40, 40]);
    let mut v = Visualiser::default();
    let blur = v.step(&blurred(&red), WINDOW, DT);
    assert!(blur.scene.is_none() && blur.palette().is_empty());
    assert_eq!(blur.ground.of, (red.id, WINDOW), "the cover blurred");
    let story = v.step(&cue(Some(&s), 15.0), WINDOW, DT);
    assert_eq!(story.scene.expect("a scene").drawn.0, 0);
    assert_eq!(story.ground.of, (0, WINDOW), "the gradient under it");
    let back = v.step(&blurred(&red), WINDOW, DT);
    assert!(back.scene.is_none() && back.ground == blur.ground);
}

/// The scene is graded as the frame's cue says (the theme in force then):
/// another grade is the next frame's, whole.
#[test]
fn a_scene_is_graded_by_the_theme_of_its_frame() {
    use crate::theme::{CLASSIC, SPOOKY};
    let s = story([[200, 40, 40], [40, 40, 200]]);
    let graded = |grade| Cue {
        grade,
        ..cue(Some(&s), 15.0)
    };
    let mut v = Visualiser::default();
    let grade = |f: BackdropFrame| f.scene.expect("a scene").drawn.1.grade;
    assert_eq!(
        grade(v.step(&graded(CLASSIC.grade.uniform()), WINDOW, DT)),
        CLASSIC.grade.uniform()
    );
    assert_eq!(
        grade(v.step(&graded(SPOOKY.grade.uniform()), WINDOW, DT)),
        SPOOKY.grade.uniform()
    );
    assert_eq!(
        grade(v.step(&graded(CLASSIC.grade.uniform()), WINDOW, DT)),
        CLASSIC.grade.uniform()
    );
}

/// The look over the backdrop takes the palette of the scene showing.
#[test]
fn the_look_takes_the_palette_of_the_scene_showing() {
    let s = story([[200, 40, 40], [40, 40, 200]]);
    let mut v = Visualiser::default();
    let first = v.step(&cue(Some(&s), 15.0), WINDOW, DT);
    assert_eq!(first.palette()[0].rgb, [200, 40, 40]);
    let second = v.step(&cue(Some(&s), 30.0), WINDOW, DT);
    assert_eq!(second.palette()[0].rgb, [40, 40, 200]);
}

/// The GPU's frames: without the game the red cover blurred and dark; with
/// it the plain ground (dark, not black) before the story, its blue scene
/// once it is there; black and nothing held once the backdrop is gone.
#[test]
fn the_backdrop_draws_its_scene_on_the_gpu_and_the_plain_ground_without() {
    let Ok(mut gpu) = crate::gui::present::Offscreen::new(false) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let (w, h) = (256u32, 144u32);
    let mut shown = |f: Option<&BackdropFrame>| {
        gpu.set_backdrop(f, w, h);
        gpu.show_frame(w, h).expect("a frame")
    };
    let rgb = |p: u32| [(p >> 16) & 255, (p >> 8) & 255, p & 255];
    let middle = (h / 2 * w + w / 2) as usize;
    let mut v = Visualiser::default();
    let s = story([[40, 40, 220], [40, 40, 220]]);
    let red = blurred(&art([220, 40, 40]));
    let [r, g, b] = rgb(shown(Some(&v.step(&red, (w, h), DT)))[middle]);
    assert!(r > 2 * g && r > 2 * b && r < 160, "{:?}", [r, g, b]);
    let [r, g, b] = rgb(shown(Some(&v.step(&cue(None, 12.0), (w, h), DT)))[middle]);
    assert!(
        (10..60).contains(&r) && r.abs_diff(b) < 10 && g < 60,
        "{:?}",
        [r, g, b]
    );
    let [r, g, b] = rgb(shown(Some(&v.step(&cue(Some(&s), 15.0), (w, h), DT)))[middle]);
    assert!(b > 2 * g && b > 2 * r && b > 60, "{:?}", [r, g, b]);
    let gone = shown(None);
    assert_eq!(gone[middle] & 0x00ff_ffff, 0);
}
