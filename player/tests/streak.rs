//! The singing's shine as the window draws it: on the stage, rainbow
//! bands from the playhead (more and brighter a tier up, the whole lane
//! at the top), a swirl round the arrow, the multiplier in rainbow; a
//! burst from the playhead each tier up, once; the bands collapsing into
//! the playhead when a phrase reaches no tier; without the lane, a rainbow
//! edge on the bar under the word sung, and a smaller burst a step up. The
//! tier is set on the singing's tally as the scoring would. With
//! `JODEL_FRAMES` set to a folder, the frames are saved there as PNGs.

mod common;

use jodeldiplom::core::shine::{Shone, Tier};
use jodeldiplom::core::singing::Singing;
use jodeldiplom::gui::fx::rainbow::{shine_of, BANDS, RIM, SWIRL};
use jodeldiplom::gui::particles::Kind;
use jodeldiplom::gui::present::Offscreen;
use jodeldiplom::gui::stage::gpu::{Shape, StageFrame};
use jodeldiplom::gui::stage::{Song, Stage};

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
    let mut png = durable::AtomicFile::create(&std::path::Path::new(&dir).join(name)).unwrap();
    img.write_to(&mut png, image::ImageFormat::Png).unwrap();
    png.commit().unwrap();
}

/// The shine at a moment: the tier the last phrase reached, and when that
/// phrase ended.
type At = (Option<Tier>, f64);

/// [`common::bubble_song`] sung on "la"'s note all along, frame by frame
/// at 60 a second up to `until`, the last phrase as `at` says at each
/// moment (the tally set as the scoring sets it), a streak at the first
/// rung (×2): the stage's frames.
fn stage_frames(at: impl Fn(f64) -> At, until: f64) -> Vec<(f64, StageFrame)> {
    let (mut singing, lyrics) = common::bubble_song();
    let picture = common::bubble_picture(&singing, &lyrics, (0.0, 0.0));
    let mut stage = Stage::default();
    let (mut t, mut heard, mut out) = (0.0, 0u32, Vec::new());
    while t <= until + 1e-9 {
        while f64::from(heard) / 100.0 <= t.min(0.95) {
            singing.hear(f64::from(heard) / 100.0, Some(6000.0));
            heard += 1;
        }
        set(&mut singing, at(t));
        singing.tally.streak = Tier::ALL[0].spec().streak;
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
        top: 0.9,
        tier,
    };
    singing.decided(singing.last_heard().unwrap_or(0.0), shone);
}

/// The light a stage frame's instances of `k` add (summed over its
/// pixels), drawn on the GPU against the same frame without them.
fn light_of(gpu: &mut Offscreen, frame: &StageFrame, k: f32) -> (u64, Vec<u32>) {
    let mut show = |f: &StageFrame| {
        gpu.set_stage(Some(f), 800, 450);
        gpu.show_frame(800, 450).expect("a frame")
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
    for (tier, name) in [
        (None, ""),
        (Some(Tier::named("Good")), "good"),
        (Some(Tier::named("Great")), ""),
        (Some(Tier::named("Amazing")), "amazing"),
    ] {
        let level = tier.map_or(0.0, |t| t.spec().glow);
        let frames = stage_frames(|_| (tier, 0.1), 0.6);
        let (_, frame) = frames.last().unwrap();
        let shines = level > 0.0;
        let bands = frame.instances.iter().find(|i| i.params[0] == BANDS);
        assert_eq!(bands.is_some(), shines, "bands while it shines");
        if let Some(shine) = bands.and_then(shine_of) {
            assert_eq!(shine.level, level);
        }
        let swirls = frame.instances.iter().filter(|i| i.params[0] == SWIRL);
        assert_eq!(swirls.count(), usize::from(shines), "round the arrow");
        // The multiplier is over everything.
        let rainbow = (frame.instances.iter().chain(&frame.top))
            .filter(|i| i.is(Shape::Text) && shine_of(i).is_some());
        assert_eq!(
            rainbow.count(),
            usize::from(shines),
            "the multiplier in rainbow"
        );
        if let Some(gpu) = gpu.as_mut() {
            let (light, pixels) = light_of(gpu, frame, BANDS);
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
    // None to 0.2 s; Good at 0.2 s (one up); Good again at 0.4 s (none);
    // Amazing at 0.6 s (up from Good to it): one a tier up to Amazing.
    let (good, amazing) = (Tier::named("Good"), Tier::named("Amazing"));
    let at = |t: f64| match t {
        t if t < 0.2 => (None, 0.1),
        t if t < 0.4 => (Some(good), 0.2),
        t if t < 0.6 => (Some(good), 0.4),
        _ => (Some(amazing), 0.6),
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
    assert_eq!(ups.len(), amazing.index() + 1, "a step a tier: {ups:?}");
    if let Some(mut gpu) = gpu() {
        let (_, frame) = frames.iter().find(|(t, _)| *t > 0.6 + 0.25).unwrap();
        let (_, pixels) = light_of(&mut gpu, frame, -1.0);
        save_frame("streak_step_up.png", &pixels, 800);
    }
}

/// A phrase reaching no tier: the bands collapse into the playhead and
/// drain, less light each moment, none after
/// [`jodeldiplom::gui::shine::COLLAPSE`].
#[test]
fn a_phrase_without_a_tier_collapses_the_shine() {
    let amazing = Some(Tier::named("Amazing"));
    let at = |t: f64| if t < 0.6 { (amazing, 0.1) } else { (None, 0.6) };
    let frames = stage_frames(at, 1.3);
    let at = |t: f64| &frames.iter().find(|(f, _)| *f >= t).unwrap().1;
    let bands = at(0.7).instances.iter().find(|i| i.params[0] == BANDS);
    let shine = bands.and_then(shine_of).expect("the bands");
    let was = amazing.map(|t| t.spec().glow);
    assert_eq!((shine.level, Some(shine.from)), (0.0, was));
    let Some(mut gpu) = gpu() else {
        return;
    };
    let lights: Vec<u64> = [0.55, 0.65, 0.8, 0.95, 1.25]
        .iter()
        .map(|&t| light_of(&mut gpu, at(t), BANDS).0)
        .collect();
    assert!(lights.windows(2).all(|w| w[1] < w[0]), "{lights:?}");
    assert_eq!(lights[4], 0, "gone: {lights:?}");
    save_frame(
        "streak_collapse.png",
        &light_of(&mut gpu, at(0.75), BANDS).1,
        800,
    );
}

/// The lyrics set in the middle (the notes shown without the lane) while
/// the singing shines: a rainbow edge on the bar under the word sung (none
/// without a tier), a step up's burst from it, smaller than the stage's,
/// once; drawn round the bar on the GPU.
#[test]
fn the_picture_s_bar_gets_a_rainbow_edge() {
    let (mut singing, lyrics) = common::bubble_song();
    let mut stage = Stage::default();
    let (w, h) = (800usize, 450usize);
    let area = (0.0, 0.0, w as f32, h as f32);
    let mut ups = Vec::new();
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
                (Some(Tier::named("Good")), 0.5)
            },
        );
        let picture = common::bubble_picture(&singing, &lyrics, (t, 0.0));
        let song = Song::new(&singing, lyrics.lines.iter().collect(), Some(3.0));
        let tint = &jodeldiplom::gui::stage::Tint::STAGE;
        let frame = stage.scene((&song, false), &picture, area, (t, 1.0 / 60.0), tint);
        let frame = frame.expect("the lyrics");
        let rims = frame
            .instances
            .iter()
            .filter(|i| i.params[0] == RIM)
            .count();
        if (t - 0.5).abs() > 0.02 {
            assert_eq!(rims, usize::from(t > 0.5), "{t}");
        }
        for s in frame.sprays.iter().filter(|s| s.kind == Kind::StepUp) {
            assert!(s.size < 47.0, "smaller than the stage's");
            if !ups
                .iter()
                .any(|u: &(u32, (f32, f32))| u.0 == s.start.to_bits())
            {
                ups.push((s.start.to_bits(), s.at));
            }
        }
        last = Some(frame);
    }
    assert_eq!(ups.len(), 1, "one step up");
    let frame = last.unwrap();
    let rim = *frame.instances.iter().find(|i| i.params[0] == RIM).unwrap();
    let (c, half) = ((rim.shape[0], rim.shape[1]), (rim.shape[2], rim.shape[3]));
    assert_eq!(ups[0].1, c, "from the bar");
    let Some(mut gpu) = gpu() else {
        return;
    };
    let mut show = |f: &StageFrame| {
        gpu.set_stage(Some(f), w as u32, h as u32);
        gpu.show_frame(w as u32, h as u32).expect("a frame")
    };
    let mut plain = frame.clone();
    plain.instances.retain(|i| i.params[0] != RIM);
    plain.sprays.clear();
    let (lit, plain) = (show(&frame), show(&plain));
    save_frame("picture_rim.png", &lit, w);
    // Lit along the bar's top edge, its colours not all one.
    let row = (c.1 - half.1).round() as usize;
    let xs: Vec<usize> = (0..10)
        .map(|k| (c.0 - half.0 * 0.8 + half.0 * 1.6 * k as f32 / 9.0) as usize)
        .collect();
    let along: Vec<u32> = xs.iter().map(|&x| lit[row * w + x]).collect();
    let lum = |p: u32| ((p >> 16) & 255) + ((p >> 8) & 255) + (p & 255);
    let brighter = along
        .iter()
        .zip(xs.iter().map(|&x| plain[row * w + x]))
        .filter(|(a, b)| lum(**a) > lum(*b) + 30)
        .count();
    assert!(brighter >= 8, "{brighter}");
    let hues: std::collections::HashSet<u32> = along.iter().map(|p| p & 0xE0E0E0).collect();
    assert!(hues.len() > 3, "rainbow: {hues:?}");
}
