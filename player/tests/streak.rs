//! A streak's shine as the window draws it: on the stage, rainbow bands
//! from the playhead (more and brighter a level up, the whole lane at the
//! top), a swirl round the arrow, the multiplier in rainbow; a burst from
//! the playhead each step up, once; the bands collapsing into the playhead
//! when the streak is lost; in the picture, a rainbow edge on the bar
//! under the word sung, and a smaller burst a step up. The streak is set
//! on the singing's tally as the scoring would. With `JODEL_FRAMES` set to
//! a folder, the frames are saved there as PNGs.

mod common;

use jodeldiplom::core::singing::{Hit, Level, Scored, Singing};
use jodeldiplom::gui::karaoke::{paint, Backdrop, BigText, Layer, Target};
use jodeldiplom::gui::notes::Over;
use jodeldiplom::gui::particles::Kind;
use jodeldiplom::gui::present::Offscreen;
use jodeldiplom::gui::rainbow::shine_of;
use jodeldiplom::gui::stage::{Song, Stage};
use jodeldiplom::gui::stage_gpu::{kind, with_picture, StageFrame};
use jodeldiplom::gui::streak::{Moment, PictureShine, Sung};
use ratatui::layout::Rect;

/// `pixels` (`0x00RRGGBB`, `w` wide) as a PNG at `name` in the folder
/// `JODEL_FRAMES` names, where it is set.
fn save_frame(name: &str, pixels: &[u32], w: usize) {
    let Some(dir) = std::env::var_os("JODEL_FRAMES") else {
        return;
    };
    let h = pixels.len() / w;
    let mut img = image::RgbImage::new(w as u32, h as u32);
    for (i, p) in pixels.iter().enumerate() {
        let rgb = image::Rgb([(p >> 16) as u8, (p >> 8) as u8, *p as u8]);
        img.put_pixel((i % w) as u32, (i / w) as u32, rgb);
    }
    img.save(std::path::Path::new(&dir).join(name)).unwrap();
}

/// A streak at a moment: its length, and the word that took it there
/// (when it ended), if one just did.
type At = (u32, Option<f64>);

/// [`common::bubble_song`] sung on "la"'s note all along, frame by frame
/// at 60 a second up to `until`, the streak as `streak` says at each
/// moment (the tally set as the scoring sets it): the stage's frames.
fn stage_frames(streak: impl Fn(f64) -> At, until: f64) -> Vec<(f64, StageFrame)> {
    let (mut singing, lyrics) = common::bubble_song(Level::Easy);
    let picture = common::bubble_picture(&singing, &lyrics, (0.0, 0.0), Rect::new(0, 0, 80, 24));
    let mut stage = Stage::new();
    let (mut t, mut heard, mut out) = (0.0, 0u32, Vec::new());
    while t <= until + 1e-9 {
        while f64::from(heard) / 100.0 <= t.min(0.95) {
            singing.hear(f64::from(heard) / 100.0, Some(6000.0));
            heard += 1;
        }
        set(&mut singing, streak(t));
        let song = Song::new(&singing, lyrics.lines.iter().collect(), Some(3.0));
        let frame = stage.frame(&song, &picture, (0.0, 0.0, 800.0, 450.0), t, 1.0 / 60.0);
        out.push((t, frame));
        t += 1.0 / 60.0;
    }
    out
}

/// The tally of `singing` with a streak `at`.
fn set(singing: &mut Singing, (streak, scored): At) {
    singing.tally.streak = streak;
    if let Some(end) = scored {
        singing.tally.last_scored = Some(Scored {
            end,
            hit: Some(Hit::Full),
            points: 10,
            streak,
        });
    }
}

/// The light a stage frame's instances of `k` add (summed over its
/// pixels), drawn on the GPU against the same frame without them.
fn light_of(gpu: &mut Offscreen, frame: &StageFrame, k: f32) -> (u64, Vec<u32>) {
    let blank = vec![0u32; 800 * 450];
    let mut show = |f: &StageFrame| {
        gpu.set_stage(Some(f), 800, 450);
        gpu.show_pixels(&blank, 800, 450).expect("a frame")
    };
    let lit = show(frame);
    let mut without = frame.clone();
    without.instances.retain(|i| i.params[0] != k);
    let bare = show(&without);
    let lum = |p: u32| u64::from((p >> 16) & 255) + u64::from((p >> 8) & 255) + u64::from(p & 255);
    let sum = lit
        .iter()
        .zip(&bare)
        .map(|(a, b)| lum(*a).saturating_sub(lum(*b)))
        .sum();
    (sum, lit)
}

fn gpu() -> Option<Offscreen> {
    Offscreen::new(false)
        .map_err(|e| eprintln!("no GPU ({e}): skipped"))
        .ok()
}

/// Each level its bands: none at ×1, a soft shimmer at ×2, more at ×3,
/// the whole lane at ×4, each level more light than the one below; the
/// swirl round the arrow and the multiplier in rainbow while it shines.
#[test]
fn the_bands_shine_by_the_level() {
    let mut lights = Vec::new();
    let mut gpu = gpu();
    for (streak, level, name) in [
        (0, 0.0, ""),
        (5, 1.0, "x2"),
        (10, 2.0, ""),
        (20, 3.0, "top"),
    ] {
        let frames = stage_frames(|_| (streak, None), 0.6);
        let (_, frame) = frames.last().unwrap();
        let shines = level > 0.0;
        let bands = frame.instances.iter().find(|i| i.params[0] == kind::BANDS);
        assert_eq!(bands.is_some(), shines, "bands while it shines");
        if let Some(shine) = bands.and_then(shine_of) {
            assert_eq!(shine.level, level);
        }
        let swirls = frame
            .instances
            .iter()
            .filter(|i| i.params[0] == kind::SWIRL);
        assert_eq!(swirls.count(), usize::from(shines), "round the arrow");
        let rainbow = frame
            .instances
            .iter()
            .filter(|i| i.params[0] == kind::TEXT && shine_of(i).is_some());
        assert_eq!(
            rainbow.count(),
            usize::from(level > 0.0),
            "the multiplier in rainbow"
        );
        if let Some(gpu) = gpu.as_mut() {
            let (light, pixels) = light_of(gpu, frame, kind::BANDS);
            lights.push(light);
            if !name.is_empty() {
                save_frame(&format!("streak_{name}.png"), &pixels, 800);
            }
        }
    }
    if !lights.is_empty() {
        assert_eq!(lights[0], 0, "none at ×1");
        assert!(lights.windows(2).all(|w| w[1] > w[0]), "{lights:?}");
        assert!(lights[3] > 3 * lights[1], "the top far more: {lights:?}");
    }
}

/// Each step up, one burst from the playhead (a step-up record, from its
/// start for its life, once however many frames show it; none for a word
/// that keeps the level); the multiplier pops.
#[test]
fn a_step_up_bursts_once_a_step() {
    // ×1 to 0.2 s; 5 in a row at 0.2 s (×2), 6 at 0.4 s (still ×2), 10 at
    // 0.6 s (×3).
    let streak = |t: f64| match t {
        t if t < 0.2 => (4, Some(0.1)),
        t if t < 0.4 => (5, Some(0.2)),
        t if t < 0.6 => (6, Some(0.4)),
        _ => (10, Some(0.6)),
    };
    let frames = stage_frames(streak, 1.1);
    let mut ups: Vec<(u32, u32)> = Vec::new();
    for (t, frame) in &frames {
        let here: Vec<_> = frame
            .sprays
            .iter()
            .filter(|s| s.kind == Kind::StepUp)
            .collect();
        for s in &here {
            assert!(s.alive(*t), "{t}");
            if !ups.contains(&(s.start.to_bits(), s.seed)) {
                ups.push((s.start.to_bits(), s.seed));
            }
        }
        let keys: std::collections::HashSet<_> =
            here.iter().map(|s| (s.start.to_bits(), s.seed)).collect();
        assert_eq!(keys.len(), here.len(), "once in a frame at {t}");
    }
    assert_eq!(ups.len(), 2, "two steps up: {ups:?}");
    let scored = frames
        .iter()
        .flat_map(|(_, f)| f.sprays.iter().filter(|s| s.kind == Kind::Scored))
        .map(|s| s.start.to_bits())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(scored.len(), 4, "each word its sparks where it hit");
    if let Some(mut gpu) = gpu() {
        let (_, frame) = frames.iter().find(|(t, _)| *t > 0.6 + 0.25).unwrap();
        let (_, pixels) = light_of(&mut gpu, frame, -1.0);
        save_frame("streak_step_up.png", &pixels, 800);
    }
}

/// Lost, the bands collapse into the playhead and drain: less light each
/// moment, none after [`jodeldiplom::gui::rainbow::COLLAPSE`].
#[test]
fn a_lost_streak_collapses_into_the_playhead() {
    let streak = |t: f64| if t < 0.6 { (20, None) } else { (0, None) };
    let frames = stage_frames(streak, 1.3);
    let at = |t: f64| &frames.iter().find(|(f, _)| *f >= t).unwrap().1;
    let bands = at(0.7)
        .instances
        .iter()
        .find(|i| i.params[0] == kind::BANDS);
    let shine = bands.and_then(shine_of).expect("the bands");
    assert_eq!((shine.level, shine.from), (0.0, 3.0));
    let Some(mut gpu) = gpu() else {
        return;
    };
    let lights: Vec<u64> = [0.55, 0.65, 0.8, 0.95, 1.25]
        .iter()
        .map(|&t| light_of(&mut gpu, at(t), kind::BANDS).0)
        .collect();
    assert!(lights.windows(2).all(|w| w[1] < w[0]), "{lights:?}");
    assert_eq!(lights[4], 0, "gone: {lights:?}");
    save_frame(
        "streak_collapse.png",
        &light_of(&mut gpu, at(0.75), kind::BANDS).1,
        800,
    );
}

/// `picture` painted for a window with a GPU, and what the GPU draws over
/// it.
fn painted(
    picture: &jodeldiplom::view::karaoke::KaraokePicture,
    (w, h): (usize, usize),
) -> (Vec<u32>, Over) {
    let backdrop = Backdrop::new(None, w, h);
    let mut pixels = vec![0u32; w * h];
    let mut target = Target {
        pixels: &mut pixels,
        width: w,
        height: h,
    };
    let mut layer = Layer::default();
    layer.over.gpu = true;
    let mut text = BigText::default();
    paint(
        &mut target,
        &mut text,
        &mut layer,
        &backdrop,
        picture,
        (0, 0, w, h),
        None,
        true,
    );
    (pixels, layer.over)
}

/// The picture while the streak shines: a rainbow edge on the bar under
/// the word sung (none at ×1), a step up's burst from it, smaller than the
/// stage's, once; drawn round the bar on the GPU.
#[test]
fn the_picture_s_bar_gets_a_rainbow_edge() {
    let (mut singing, lyrics) = common::bubble_song(Level::Medium);
    let mut shine = PictureShine::default();
    let (w, h) = (800usize, 450usize);
    let mut ups = std::collections::HashSet::new();
    let mut last = None;
    for f in 0..40 {
        let t = 0.3 + f64::from(f) / 60.0;
        singing.hear(t, Some(6000.0));
        set(
            &mut singing,
            if t < 0.5 {
                (4, Some(0.2))
            } else {
                (5, Some(0.5))
            },
        );
        let picture = common::bubble_picture(&singing, &lyrics, (t, 0.0), Rect::new(0, 0, 80, 24));
        let (pixels, over) = painted(&picture, (w, h));
        let rim = over.rim.expect("the bar under the word sung");
        let song = Sung {
            singing: &singing,
            game: true,
        };
        let (edge, sprays) = shine.frame(Some(&over), song, Moment { now: t, clock: t });
        let (bubbles, sprays): (Vec<_>, Vec<_>) = (edge.into_iter().collect(), sprays.to_vec());
        if (t - 0.5).abs() > 0.02 {
            assert_eq!(bubbles.len(), usize::from(t > 0.5), "{t}");
        }
        for s in sprays.iter().filter(|s| s.kind == Kind::StepUp) {
            assert_eq!(s.at, rim.0, "from the bar");
            assert!(s.size < 47.0, "smaller than the stage's");
            ups.insert(s.start.to_bits());
        }
        last = Some((pixels, bubbles, sprays, t));
    }
    assert_eq!(ups.len(), 1, "one step up");
    let (pixels, bubbles, sprays, t) = last.unwrap();
    let Some(mut gpu) = gpu() else {
        return;
    };
    let mut show = |b: &[_], s: &[_]| {
        let frame = with_picture(None, b, s, t as f32);
        gpu.set_stage(frame.as_ref(), w as u32, h as u32);
        gpu.show_pixels(&pixels, w as u32, h as u32)
            .expect("a frame")
    };
    let (lit, plain) = (show(&bubbles, &sprays), show(&[], &[]));
    save_frame("picture_rim.png", &lit, w);
    let rim = bubbles.iter().find(|i| i.params[0] == kind::RIM).unwrap();
    let (c, half) = ((rim.shape[0], rim.shape[1]), (rim.shape[2], rim.shape[3]));
    // Lit along the bar's top edge, its colours not all one.
    let row = (c.1 - half.1).round() as usize;
    let along: Vec<u32> = (0..10)
        .map(|k| (c.0 - half.0 * 0.8 + half.0 * 1.6 * k as f32 / 9.0) as usize)
        .map(|x| lit[row * w + x])
        .collect();
    let lum = |p: u32| ((p >> 16) & 255) + ((p >> 8) & 255) + (p & 255);
    let brighter = along
        .iter()
        .zip((0..10).map(|k| {
            plain[row * w + (c.0 - half.0 * 0.8 + half.0 * 1.6 * k as f32 / 9.0) as usize]
        }))
        .filter(|(a, b)| lum(**a) > lum(*b) + 30)
        .count();
    assert!(brighter >= 8, "{brighter}");
    let hues: std::collections::HashSet<u32> = along.iter().map(|p| p & 0xE0E0E0).collect();
    assert!(hues.len() > 3, "rainbow: {hues:?}");
}
