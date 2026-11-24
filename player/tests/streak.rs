//! The singing's shine as the window draws it: on the stage, rainbow
//! bands from the playhead (more and brighter a tier up, the whole lane
//! at the top), a swirl round the arrow, the multiplier in rainbow; a
//! burst from the playhead each tier up, once; the bands collapsing into
//! the playhead when a phrase reaches no tier; in the picture, a rainbow
//! edge on the bar under the word sung, and a smaller burst a step up. The
//! tier is set on the singing's tally as the scoring would. With
//! `JODEL_FRAMES` set to a folder, the frames are saved there as PNGs.

mod common;

use jodeldiplom::core::shine::{Shone, Tier};
use jodeldiplom::core::singing::Singing;
use jodeldiplom::gui::big_text::Faces;
use jodeldiplom::gui::karaoke::{paint, Backdrop, Layer};
use jodeldiplom::gui::notes::Over;
use jodeldiplom::gui::particles::Kind;
use jodeldiplom::gui::present::Offscreen;
use jodeldiplom::gui::rainbow::shine_of;
use jodeldiplom::gui::raster::Target;
use jodeldiplom::gui::stage::{Song, Stage};
use jodeldiplom::gui::stage_gpu::{kind, with_picture, StageFrame};
use jodeldiplom::gui::streak::{PictureShine, Sung};
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

/// The shine at a moment: the tier the last phrase reached, and when that
/// phrase ended.
type At = (Option<Tier>, f64);

/// [`common::bubble_song`] sung on "la"'s note all along, frame by frame
/// at 60 a second up to `until`, the last phrase as `at` says at each
/// moment (the tally set as the scoring sets it), a streak of five words
/// (×2): the stage's frames.
fn stage_frames(at: impl Fn(f64) -> At, until: f64) -> Vec<(f64, StageFrame)> {
    let (mut singing, lyrics) = common::bubble_song();
    let picture = common::bubble_picture(&singing, &lyrics, (0.0, 0.0), Rect::new(0, 0, 80, 24));
    let mut stage = Stage::default();
    let (mut t, mut heard, mut out) = (0.0, 0u32, Vec::new());
    while t <= until + 1e-9 {
        while f64::from(heard) / 100.0 <= t.min(0.95) {
            singing.hear(f64::from(heard) / 100.0, Some(6000.0));
            heard += 1;
        }
        set(&mut singing, at(t));
        singing.tally.streak = 5;
        let song = Song::new(&singing, lyrics.lines.iter().collect(), Some(3.0));
        let frame = stage.frame(&song, &picture, (0.0, 0.0, 800.0, 450.0), t, 1.0 / 60.0);
        out.push((t, frame));
        t += 1.0 / 60.0;
    }
    out
}

/// `singing` with its last phrase `at`: decided (at the last frame heard)
/// where it is another.
fn set(singing: &mut Singing, (tier, end): At) {
    if singing.last_phrase().map(|p| (p.tier, p.end)) == Some((tier, end)) {
        return;
    }
    let shone = Shone {
        line: 0,
        end,
        value: 0.8,
        diva: 0.9,
        tier,
    };
    singing.decided(singing.last_heard().unwrap_or(0.0), shone);
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

/// Each tier its bands: none without one, a soft shimmer at Good, more
/// at Great, the whole lane from Amazing, each more light than the one
/// below; the swirl round the arrow and the multiplier in rainbow while it
/// shines.
#[test]
fn the_bands_shine_by_the_tier() {
    let mut lights = Vec::new();
    let mut gpu = gpu();
    for (tier, level, name) in [
        (None, 0.0, ""),
        (Some(Tier::Good), 1.0, "good"),
        (Some(Tier::Great), 2.0, ""),
        (Some(Tier::Amazing), 3.0, "amazing"),
    ] {
        let frames = stage_frames(|_| (tier, 0.1), 0.6);
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
            usize::from(shines),
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
        assert_eq!(lights[0], 0, "none without a tier");
        assert!(lights.windows(2).all(|w| w[1] > w[0]), "{lights:?}");
        assert!(lights[3] > 3 * lights[1], "the top far more: {lights:?}");
    }
}

/// Each tier up, one burst from the playhead (a step-up record, from its
/// start for its life, once however many frames show it; none for a
/// phrase that keeps the tier).
#[test]
fn a_step_up_bursts_once_a_tier() {
    // None to 0.2 s; Good at 0.2 s; Good again at 0.4 s; Amazing at 0.6 s
    // (two up).
    let at = |t: f64| match t {
        t if t < 0.2 => (None, 0.1),
        t if t < 0.4 => (Some(Tier::Good), 0.2),
        t if t < 0.6 => (Some(Tier::Good), 0.4),
        _ => (Some(Tier::Amazing), 0.6),
    };
    let frames = stage_frames(at, 1.1);
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
    assert_eq!(ups.len(), 3, "three steps up: {ups:?}");
    if let Some(mut gpu) = gpu() {
        let (_, frame) = frames.iter().find(|(t, _)| *t > 0.6 + 0.25).unwrap();
        let (_, pixels) = light_of(&mut gpu, frame, -1.0);
        save_frame("streak_step_up.png", &pixels, 800);
    }
}

/// A phrase reaching no tier: the bands collapse into the playhead and
/// drain, less light each moment, none after
/// [`jodeldiplom::gui::rainbow::COLLAPSE`].
#[test]
fn a_phrase_without_a_tier_collapses_the_shine() {
    let at = |t: f64| {
        if t < 0.6 {
            (Some(Tier::Amazing), 0.1)
        } else {
            (None, 0.6)
        }
    };
    let frames = stage_frames(at, 1.3);
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
    let mut text = Faces::default();
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

/// The picture while the singing shines: a rainbow edge on the bar under
/// the word sung (none without a tier), a step up's burst from it, smaller
/// than the stage's, once; drawn round the bar on the GPU.
#[test]
fn the_picture_s_bar_gets_a_rainbow_edge() {
    let (mut singing, lyrics) = common::bubble_song();
    let mut shine = PictureShine::default();
    let (w, h) = (800usize, 450usize);
    let mut ups = std::collections::HashSet::new();
    let mut last = None;
    for f in 0..40 {
        let t = 0.3 + f64::from(f) / 60.0;
        // Heard, not sung: the word's bar stays (sung, it would count and
        // burst).
        singing.hear(t, None);
        set(
            &mut singing,
            if t < 0.5 {
                (None, 0.2)
            } else {
                (Some(Tier::Good), 0.5)
            },
        );
        let picture = common::bubble_picture(&singing, &lyrics, (t, 0.0), Rect::new(0, 0, 80, 24));
        let (pixels, over) = painted(&picture, (w, h));
        let rim = over.rim.expect("the bar under the word sung");
        let song = Sung {
            singing: &singing,
            game: true,
        };
        let (edge, sprays) = shine.frame(Some(&over), song, t);
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
