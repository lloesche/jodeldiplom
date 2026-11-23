//! The note bubbles of Karaoke mode as the window draws them: each filling
//! with liquid as it is sung, a word counted on easy swept full,
//! straining, then bursting whole (it overfills, tears, its liquid
//! splashing out, and is gone), in the picture painted on the processor
//! (its liquid and its burst drawn by the GPU over it, the bubble gone from
//! it once it bursts), and on the GPU's stage; the voice heard as it comes
//! or 264 ms late (the bubbles go by the singing's own time, so the sweep
//! and the burst are seen whole either way). A bubble only partly filled
//! never strains or bursts; one no longer full after a seek back shows
//! again. With `JODEL_FRAMES` set to a folder, the window's frames are
//! saved there as PNGs.

mod common;

use jodeldiplom::core::singing::{Level, Singing};
use jodeldiplom::gui::karaoke::{paint, Backdrop, BigText, Layer, Target};
use jodeldiplom::gui::notes::Over;
use jodeldiplom::gui::particles::{Kind, Spray};
use jodeldiplom::gui::stage::{Song, Stage};
use jodeldiplom::gui::stage_gpu::{kind, Instance, StageFrame};
use jodeldiplom::view::bubbles::{BURST, SWEEP, SWELL};
use jodeldiplom::view::karaoke::KaraokePicture;
use ratatui::layout::Rect;

/// When "la" of [`common::bubble_song`] counts on easy: its 11th frame hit
/// (a tenth of the 100 it has, and one more).
const COUNTED: f64 = 0.10;

/// `pixels` (`0x00RRGGBB`, `w` wide) as a PNG at `name` in the folder
/// `JODEL_FRAMES` names, where it is set (to look at a frame).
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

/// The user's delay: how late the voice comes.
const LATE: f64 = 0.264;

/// `singing` of [`common::bubble_song`] heard up to `heard` seconds into
/// the song as heard, the voice coming `late`: its frames up to then less
/// that (from `next`, the first not yet heard); the next to hear.
fn hear_to(singing: &mut Singing, next: f64, heard: f64, late: f64) -> f64 {
    common::sing_bubbles(singing, next, heard - late);
    (((heard - late) * 100.0).floor() + 1.0).max(0.0) / 100.0
}

/// `singing` hears its frames from `from` to `to` seconds into the song
/// (going back to `from` first, as after a seek), sung on "la"'s note
/// before `until`, silent after.
fn hear_until(singing: &mut Singing, (from, to): (f64, f64), until: f64) {
    let mut i = (from * 100.0).round() as u32;
    while f64::from(i) / 100.0 <= to {
        let t = f64::from(i) / 100.0;
        singing.hear(t, (t < until - 1e-9).then_some(6000.0));
        i += 1;
    }
}

/// `picture` painted afresh at `w` × `h` (for a window with a GPU, `gpu`,
/// or without), and what the GPU draws over it.
fn painted(picture: &KaraokePicture, (w, h): (usize, usize), gpu: bool) -> (Vec<u32>, Over) {
    let backdrop = Backdrop::new(None, w, h);
    let mut pixels = vec![0u32; w * h];
    let mut target = Target {
        pixels: &mut pixels,
        width: w,
        height: h,
    };
    let mut layer = Layer::default();
    layer.over.gpu = gpu;
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

/// `picture` with its line being sung's bubbles undrawn (`notes`) or
/// none with liquid or bursting.
fn without(picture: &KaraokePicture, notes: bool) -> KaraokePicture {
    let mut p = picture.clone();
    for l in &mut p.lines {
        if notes {
            l.notes.clear();
        }
        l.liquid.clear();
    }
    p
}

/// Whether `picture` paints "la"'s bar: it differs from the same without
/// it (and with nothing bursting).
fn la_bar(picture: &KaraokePicture) -> bool {
    let mut no_la = without(picture, false);
    no_la.lines[0].notes.remove(0);
    painted(picture, (640, 360), false).0 != painted(&no_la, (640, 360), false).0
}

/// A stage of [`common::bubble_song`] at `now` (the voice `late`), made
/// afresh (its pills are as the bubbles are each frame).
fn stage_at(singing: &Singing, lyrics: &jodel_lyrics::Lyrics, now: f64, late: f64) -> StageFrame {
    let picture = common::bubble_picture(singing, lyrics, (now, late), Rect::new(0, 0, 80, 24));
    let mut song = Song::new(singing, lyrics.lines.iter().collect(), Some(3.0));
    song.late = late;
    Stage::new().frame(&song, &picture, (0.0, 0.0, 800.0, 450.0), now, 1.0 / 60.0)
}

/// The words' bubbles on a stage, those overfilling, and the most any
/// strains.
fn word_pills(frame: &StageFrame) -> (usize, usize, f32) {
    let bubbles: Vec<_> = frame
        .instances
        .iter()
        .filter(|i| i.params[0] == kind::BUBBLE)
        .map(|i| i.strained())
        .collect();
    let swelling = bubbles.iter().filter(|s| s.popping.is_some()).count();
    let strain = bubbles.iter().map(|s| s.pressure).fold(0.0, f32::max);
    (bubbles.len(), swelling, strain)
}

/// The liquids on a stage (or over a picture) whose front is living.
fn fronts(instances: &[Instance]) -> usize {
    instances.iter().filter(|i| i.living()).count()
}

/// A stage's bursts of bubbles (not words scored).
fn bubble_sprays(frame: &StageFrame) -> Vec<&Spray> {
    let scored = |s: &&Spray| matches!(s.kind, Kind::Scored | Kind::StepUp);
    frame.sprays.iter().filter(|s| !scored(s)).collect()
}

/// A word counted on easy in the window's picture, the voice 264 ms late,
/// with a GPU and without: the bubble's bar fills (seen part way: painted
/// flat without a GPU, its liquid handed to the GPU with one, over the bar
/// painted empty); from its burst on it is no longer painted (the picture
/// changes once), its burst handed to the GPU
/// ([`jodeldiplom::gui::particles`]) for as long as it lasts, nothing
/// after. Frames drawn only where they changed are the frames drawn
/// afresh all through.
#[test]
fn a_bubble_bursts_over_the_picture_and_is_gone_from_it() {
    for gpu in [false, true] {
        let (mut singing, lyrics) = common::bubble_song(Level::Easy);
        let (w, h) = (640usize, 360usize);
        let backdrop = Backdrop::new(None, w, h);
        let area = Rect::new(0, 0, 80, 24);
        let (mut kept, mut text, mut layer) =
            (vec![0u32; w * h], BigText::default(), Layer::default());
        layer.over.gpu = gpu;
        let (mut heard, mut next, mut part_way, mut k) = (0.0, 0.0, false, 0);
        let (mut seen, mut after, mut liquid) = (None, None, 0);
        while heard < 0.95 + LATE {
            heard += 0.015;
            next = hear_to(&mut singing, next, heard, LATE);
            let picture = common::bubble_picture(&singing, &lyrics, (heard, LATE), area);
            let mut target = Target {
                pixels: &mut kept,
                width: w,
                height: h,
            };
            paint(
                &mut target,
                &mut text,
                &mut layer,
                &backdrop,
                &picture,
                (0, 0, w, h),
                None,
                k == 0,
            );
            let (afresh, over) = painted(&picture, (w, h), gpu);
            let differ = kept.iter().zip(&afresh).filter(|(a, b)| a != b).count();
            assert_eq!(differ, 0, "gpu {gpu} at {heard}");
            assert_eq!(layer.over, over);
            let line = &picture.lines[0];
            let filled: f32 = line.hits.iter().map(|h| h.1 - h.0).sum();
            part_way |= filled > 0.6 && filled < 1.9;
            match line.liquid.first() {
                Some(b) if !b.begun() => {
                    liquid += 1;
                    assert!(over.sprays.is_empty());
                    // Its liquid, with a GPU (once there is a pixel of it):
                    // as high as its level.
                    assert!(over.bubbles.len() <= usize::from(gpu));
                    if gpu && b.level > 0.1 {
                        assert_eq!(over.bubbles.len(), 1, "{}", b.level);
                    }
                    if let Some(l) = over.bubbles.first() {
                        let share = (l.uv[1] - l.uv[0]) / (2.0 * l.shape[2]);
                        assert!((share - b.level as f32).abs() < 1e-3);
                        assert_eq!(l.living(), b.level < 1.0);
                    }
                }
                Some(b) if b.bursting().is_some() && seen.is_none() => {
                    seen = Some(picture.clone());
                    // One burst, as the bubble's bar is: from its burst's
                    // start on the singing's clock.
                    assert_eq!(over.sprays.len(), 1);
                    assert_eq!(over.sprays[0].kind, Kind::Burst);
                    let start = f64::from(over.sprays[0].start);
                    assert!((start - (COUNTED + SWEEP)).abs() < 0.011, "{start}");
                }
                Some(b) if b.bursting().is_none() && after.is_none() => {
                    assert!(over.sprays.is_empty() && over.bubbles.is_empty(), "over");
                    after = Some(picture.clone());
                }
                _ => {}
            }
            k += 1;
        }
        assert!(part_way, "the sweep seen part way");
        assert!(liquid > 5, "its liquid while it fills: {liquid}");
        for picture in [seen.expect("a burst"), after.expect("over")] {
            let (a, _) = painted(&picture, (w, h), gpu);
            let (b, none) = painted(&without(&picture, false), (w, h), gpu);
            let (c, _) = painted(&without(&picture, true), (w, h), gpu);
            // Its bar not painted: as with no bars at all but "lo"'s.
            let gone = a.iter().zip(&b).filter(|(a, b)| a != b).count();
            assert!(gone > 20, "the bar gone: {gone}");
            assert!(none.sprays.is_empty() && none.bubbles.is_empty());
            assert!(a != c, "\"lo\"'s bar still painted");
        }
    }
}

/// A frame of the picture of [`common::bubble_song`] at `at` (the voice
/// as it comes) sung at `level` on "la" until `until`, for a window with a
/// GPU: painted, and what the GPU draws over it.
fn picture_at(level: Level, until: f64, at: f64, size: (usize, usize)) -> (Vec<u32>, Over) {
    let (mut singing, lyrics) = common::bubble_song(level);
    hear_until(&mut singing, (0.0, at), until);
    let picture = common::bubble_picture(&singing, &lyrics, (at, 0.0), Rect::new(0, 0, 80, 24));
    painted(&picture, size, true)
}

/// The picture's bubble drawn by the GPU over it (offscreen): partly
/// filled (its liquid gold to its level over its bar painted empty, the
/// bar as painted past its front), straining just before full (larger than
/// its bar), overfilling (larger still), splashing and after. Its liquid
/// lights the picture where its instance says; each part of its burst (its
/// drops, its bits of skin, its sparkles) where the processor's reference
/// of the shader ([`jodeldiplom::gui::particles::sprite`]) puts it;
/// nothing far from them changes, and after, nothing at all. Skipped (with
/// a note) without a GPU.
#[test]
fn the_gpu_draws_the_bubbles_over_the_picture() {
    use jodeldiplom::gui::particles::{sprite, Look, QUADS};
    use jodeldiplom::gui::present::Offscreen;
    use jodeldiplom::gui::stage_gpu::with_picture;
    let mut gpu = match Offscreen::new(false) {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("no GPU ({e}): skipped");
            return;
        }
    };
    let (w, h) = (800usize, 450usize);
    let lum = |p: u32| ((p >> 16) & 255) + ((p >> 8) & 255) + (p & 255);
    let popped = COUNTED + SWEEP;
    for (name, level, until, at) in [
        ("partial", Level::Medium, 0.6, 0.8),
        ("pressure", Level::Easy, 0.15, popped - 0.012),
        ("bulge", Level::Easy, 0.15, popped + SWELL * 0.6),
        ("splash", Level::Easy, 0.15, popped + 0.29),
        ("after", Level::Easy, 0.15, popped + BURST + 0.1),
    ] {
        let (pixels, over) = picture_at(level, until, at, (w, h));
        let bursting = matches!(name, "bulge" | "splash");
        let liquid = matches!(name, "partial" | "pressure" | "bulge");
        assert_eq!(over.sprays.len(), usize::from(bursting), "{name}");
        assert_eq!(over.bubbles.len(), usize::from(liquid), "{name}");
        let mut sprays = over.sprays.clone();
        if name == "splash" {
            // And two words scored, the second as the multiplier goes up.
            for (k, kind) in [(0, Kind::Scored), (1, Kind::StepUp)] {
                sprays.push(Spray {
                    at: (200.0 + 400.0 * k as f32, 330.0),
                    half: (0.0, 0.0),
                    start: (at - 0.2) as f32,
                    drift: -150.0,
                    size: 20.0,
                    seed: 41 + k,
                    rgb: [1.0, 0.72, 0.2],
                    kind,
                });
            }
        }
        let show = |gpu: &mut Offscreen, bubbles: &[Instance], sprays: &[Spray]| {
            let frame = with_picture(None, bubbles, sprays, at as f32);
            gpu.set_stage(frame.as_ref(), w as u32, h as u32);
            gpu.show_pixels(&pixels, w as u32, h as u32)
                .expect("a frame")
        };
        let lit = show(&mut gpu, &over.bubbles, &sprays);
        let plain = show(&mut gpu, &[], &[]);
        save_frame(&format!("picture_{name}.png"), &lit, w);
        let mut near = vec![false; w * h];
        let mut mark = |(x, y): (f32, f32), reach: f32| {
            let (x, y, reach) = (x as usize, y as usize, reach as usize);
            for yy in y.saturating_sub(reach)..(y + reach).min(h) {
                for xx in x.saturating_sub(reach)..(x + reach).min(w) {
                    near[yy * w + xx] = true;
                }
            }
        };
        for b in &over.bubbles {
            mark((b.shape[0], b.shape[1]), b.shape[2] * 1.5 + 8.0);
            liquid_drawn(b, &lit, &plain, w, name);
        }
        let (mut sparkles, mut bits, mut drops) = (0, 0, 0);
        for s in &sprays {
            for q in 0..QUADS {
                let Some(sp) = sprite(s, q, at as f32) else {
                    continue;
                };
                mark(sp.at, sp.half.0.max(sp.half.1) * 2.5 + 6.0);
                let (x, y) = (sp.at.0.floor() as usize, sp.at.1.floor() as usize);
                let what = format!("{name}: {:?} {q} at {x},{y}", sp.look);
                let count = match sp.look {
                    Look::Sparkle | Look::Note | Look::Bit => &mut sparkles,
                    Look::Shard => &mut bits,
                    Look::Drop => &mut drops,
                    Look::Sheen => continue,
                };
                if sp.alpha > 0.3 {
                    *count += 1;
                    let i = y * w + x;
                    assert!(lum(lit[i]) > lum(plain[i]) + 60, "{what}");
                }
            }
        }
        match name {
            "bulge" => {
                // Overfilling: larger than its bar, all of it.
                let b = &over.bubbles[0];
                assert!(b.strained().popping.is_some());
                let x = (b.shape[0] + b.shape[2] * 1.12) as usize;
                let i = b.shape[1] as usize * w + x;
                assert!(lum(lit[i]) > lum(plain[i]) + 60, "bulging");
            }
            "splash" => {
                let found = format!("{sparkles} sparkles, {bits} bits, {drops} drops");
                assert!(sparkles >= 10 && bits >= 4 && drops >= 6, "{found}");
            }
            "after" => assert!(lit == plain, "nothing after"),
            _ => {}
        }
        let stray = (0..w * h)
            .filter(|&i| !near[i] && lum(lit[i]) > lum(plain[i]) + 8)
            .count();
        assert_eq!(stray, 0, "{name}: light only where the bubble is");
    }
}

/// A bubble's liquid `b` as the GPU drew it (`lit`, over `plain`, `w`
/// wide): gold well inside its level; not straining, the bar as painted
/// well past its front; straining, larger than its bar (its start out
/// past the bar's).
fn liquid_drawn(b: &Instance, lit: &[u32], plain: &[u32], w: usize, what: &str) {
    let (c, half, span) = (
        (b.shape[0], b.shape[1]),
        (b.shape[2], b.shape[3]),
        (b.uv[0], b.uv[1]),
    );
    let y = c.1.floor() as usize;
    let px = |x: f32| y * w + x.floor() as usize;
    let inside = px(span.0 + (span.1 - span.0) * 0.4);
    let rgb = |p: u32| ((p >> 16) & 255, (p >> 8) & 255, p & 255);
    let (r, g, bl) = rgb(lit[inside]);
    assert!(
        r > 180 && g > 120 && bl < g,
        "{what}: gold liquid {:?}",
        (r, g, bl)
    );
    let strain = b.strained();
    if strain.pressure > 0.3 || strain.popping.is_some() {
        let past = px(c.0 - half.0 - 1.0);
        assert!(lit[past] != plain[past], "{what}: bulging past its bar");
    } else {
        let amp = (half.1 * 0.35).max(1.0);
        let x = span.1 + amp * 2.2 + 3.0;
        if x < c.0 + half.0 - 1.0 {
            assert_eq!(lit[px(x)], plain[px(x)], "{what}: the bar past it");
        }
    }
}

/// The stage on the GPU, the voice 264 ms late: "la" sung partly on medium
/// keeps its liquid at its level, its pill whole; counted on easy, its
/// bubble (its word's one) strains just before full, overfills, then is
/// gone, its drops, bits and sparks flying (one burst); after, only "lo"'s
/// pill. Skipped (with a note) without a GPU.
#[test]
fn the_stage_bursts_a_word_counted() {
    use jodeldiplom::gui::present::Offscreen;
    let mut gpu = match Offscreen::new(false) {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("no GPU ({e}): skipped");
            return;
        }
    };
    let (w, h) = (800u32, 450u32);
    let area = (0.0, 0.0, w as f32, h as f32);
    let mut frame_at = |level: Level, until: f64, now: f64| {
        let (mut singing, lyrics) = common::bubble_song(level);
        let mut stage = Stage::new();
        let picture =
            common::bubble_picture(&singing, &lyrics, (now, LATE), Rect::new(0, 0, 80, 24));
        // Frames at 60 a second up to now, as the window draws them.
        let (mut t, mut next, mut frame) = (0.0, 0.0, None);
        while t <= now + 1e-9 {
            if t - LATE >= next {
                hear_until(&mut singing, (next, t - LATE), until);
                next = ((t - LATE) * 100.0).floor() / 100.0 + 0.01;
            }
            let mut song = Song::new(&singing, lyrics.lines.iter().collect(), Some(3.0));
            song.late = LATE;
            frame = Some(stage.frame(&song, &picture, area, t, 1.0 / 60.0));
            t += 1.0 / 60.0;
        }
        let mut frame = frame.expect("a frame");
        let blank = vec![0u32; (w * h) as usize];
        gpu.set_stage(Some(&frame), w, h);
        let shown = gpu.show_pixels(&blank, w, h).expect("a frame");
        let bubbles: Vec<Kind> = bubble_sprays(&frame).iter().map(|s| s.kind).collect();
        // The pixels its bursts light.
        let fills = fronts(&frame.instances);
        frame.sprays.clear();
        gpu.set_stage(Some(&frame), w, h);
        let bare = gpu.show_pixels(&blank, w, h).expect("a frame");
        let lum = |p: u32| ((p >> 16) & 255) + ((p >> 8) & 255) + (p & 255);
        let lit = shown.iter().zip(&bare);
        let lit = lit.filter(|(a, b)| lum(**a) > lum(**b) + 60).count();
        (shown, word_pills(&frame), bubbles, lit, fills)
    };
    let popped = LATE + COUNTED + SWEEP;
    let (partial, pills, sprays, _, fronts) = frame_at(Level::Medium, 0.6, LATE + 0.8);
    assert_eq!(
        (pills.0, pills.1, pills.2, sprays),
        (2, 0, 0.0, vec![]),
        "partly"
    );
    assert_eq!(fronts, 1, "its liquid's front");
    let (strain, pills, sprays, _, _) = frame_at(Level::Easy, 0.15, popped - 0.012);
    assert_eq!((pills.0, pills.1, sprays), (2, 0, vec![]), "straining");
    assert!(pills.2 > 0.5, "straining: {}", pills.2);
    let (bulge, pills, sprays, _, _) = frame_at(Level::Easy, 0.15, popped + SWELL * 0.9);
    assert_eq!(
        (pills.0, pills.1, sprays),
        (2, 1, vec![Kind::Burst]),
        "overfilling"
    );
    let (splash, pills, sprays, lit, _) = frame_at(Level::Easy, 0.15, popped + 0.29);
    assert_eq!(
        (pills.0, pills.1, sprays),
        (1, 0, vec![Kind::Burst]),
        "splashing"
    );
    assert!(lit > 400, "its drops, bits and sparks: {lit}");
    let (after, pills, sprays, _, _) = frame_at(Level::Easy, 0.15, popped + BURST + 0.1);
    assert_eq!((pills.0, pills.1, sprays), (1, 0, vec![]), "gone");
    let (calm, pills, sprays, lit, _) = frame_at(Level::Medium, 0.15, popped + 0.29);
    assert_eq!(
        (pills.0, pills.1, sprays, lit),
        (2, 0, vec![], 0),
        "no burst"
    );
    save_frame("stage_partial.png", &partial, w as usize);
    save_frame("stage_pressure.png", &strain, w as usize);
    save_frame("stage_bulge.png", &bulge, w as usize);
    save_frame("stage_splash.png", &splash, w as usize);
    save_frame("stage_after.png", &after, w as usize);
    save_frame("stage_calm.png", &calm, w as usize);
    // Overfilling: the pill larger and brighter than straining.
    let bright = |v: &[u32]| {
        v.iter()
            .filter(|&&p| (p >> 16) & 255 > 200 && (p >> 8) & 255 > 150)
            .count()
    };
    let (b, s) = (bright(&bulge), bright(&strain));
    assert!(b > s + 300, "overfilling: {b} against {s}");
}

/// "lala" a second, its first half on C4 and its second on G4: one word
/// of two notes, two bubbles in a row on its pill. At `level`, sung on its
/// first note all through (it fills) and on its second for half of it
/// (it does not), heard up to `upto`.
fn two_bubbles(level: Level, upto: f64) -> (Singing, std::sync::Arc<jodel_lyrics::Lyrics>) {
    let lyrics = std::sync::Arc::new(jodel_lyrics::Lyrics {
        lines: vec![jodel_lyrics::Line {
            words: vec![jodel_lyrics::Word {
                start: Some(0.0),
                end: Some(1.0),
                confidence: 0.9,
                ..jodel_lyrics::Word::new("lala")
            }],
            ..jodel_lyrics::Line::default()
        }],
        source: jodel_lyrics::Source::Unknown,
        timing: jodel_lyrics::Timing::default(),
    });
    let mut melody = jodel_rhythm::waveform::Melody {
        cents: vec![0; 300],
        sure: vec![0; 300],
    };
    for i in 0..100 {
        melody.cents[i] = if i < 50 { 6000 } else { 6700 };
        melody.sure[i] = 255;
    }
    let mut singing = Singing::default();
    singing.prepare(std::path::Path::new("b"), &lyrics, &melody, 1, level);
    let mut i = 0u32;
    while f64::from(i) / 100.0 <= upto {
        let t = f64::from(i) / 100.0;
        let cents = match t {
            t if t < 0.5 => Some(6000.0),
            t if t < 0.75 => Some(6700.0),
            _ => None,
        };
        singing.hear(t, cents);
        i += 1;
    }
    (singing, lyrics)
}

/// The stage, a word of two bubbles on medium: its first full bursts on
/// its own (one burst, from its place: the pill's first half) and is gone,
/// while its second, half filled, stays with its liquid (its front
/// living), and never bursts. Its frames (first overfilling, then
/// splashing, the second still filling) saved, with a GPU.
#[test]
fn the_stage_bursts_each_bubble_of_a_word_on_its_own() {
    use jodeldiplom::core::bubbles;
    use jodeldiplom::gui::present::Offscreen;
    let full_at = |s: &Singing| bubbles::line(s, 0, f64::INFINITY).notes[0].full_at;
    let (singing, _) = two_bubbles(Level::Medium, 2.0);
    let start = full_at(&singing).expect("the first fills") + SWEEP;
    assert_eq!(bubbles::line(&singing, 0, 9.0).notes[1].full_at, None);
    let mut gpu = Offscreen::new(false).ok();
    for (name, at) in [
        ("before", start - 0.05),
        ("overfilling", start + SWELL * 0.6),
        ("splashing", start + 0.15),
        ("after", start + BURST + 0.1),
    ] {
        let (singing, lyrics) = two_bubbles(Level::Medium, at);
        let frame = stage_at(&singing, &lyrics, at, 0.0);
        let sprays = bubble_sprays(&frame);
        let bubbles = word_pills(&frame);
        let fronts = fronts(&frame.instances);
        match name {
            "before" => assert_eq!((bubbles.0, sprays.len()), (2, 0), "{name}"),
            "overfilling" => assert_eq!((bubbles.0, bubbles.1), (2, 1), "{name}"),
            "after" => assert_eq!((bubbles.0, sprays.len()), (1, 0), "{name}"),
            _ => assert_eq!(bubbles.0, 1, "{name}: the first gone"),
        }
        if name == "overfilling" || name == "splashing" {
            assert_eq!(sprays.len(), 1, "{name}: one burst, the first's");
            assert_eq!(sprays[0].kind, Kind::Burst);
            // From the pill's first half: left of the bubble that stays.
            let stays = frame
                .instances
                .iter()
                .find(|i| i.params[0] == kind::BUBBLE && i.strained().popping.is_none())
                .expect("the second bubble");
            assert!(sprays[0].at.0 < stays.shape[0], "{name}");
        }
        // The second's liquid, its front living (the first's too, maybe,
        // before it is swept full).
        assert!(
            fronts == 1 || (name == "before" && fronts >= 1),
            "{name}: {fronts}"
        );
        if let Some(gpu) = gpu.as_mut() {
            gpu.set_stage(Some(&frame), 800, 450);
            let shown = gpu.show_pixels(&vec![0u32; 800 * 450], 800, 450);
            save_frame(
                &format!("stage_two_{name}.png"),
                &shown.expect("a frame"),
                800,
            );
        }
    }
}

/// A bubble filled about 60 % on medium (its note sung for 0.6 s of its
/// second) is never full: after its time it is still drawn with its
/// liquid at its level, on every frontend, and nothing strains, overfills
/// or bursts.
#[test]
fn a_bubble_partly_filled_never_pops() {
    use jodeldiplom::core::bubbles;
    let (mut singing, lyrics) = common::bubble_song(Level::Medium);
    hear_until(&mut singing, (0.0, 1.6), 0.6);
    let b = bubbles::line(&singing, 0, f64::INFINITY);
    assert_eq!(b.notes[0].full_at, None);
    let filled: f64 = b.notes[0].spans.iter().map(|s| s.1 - s.0).sum();
    assert!((0.5..0.7).contains(&filled), "{filled}");
    for heard in [0.8, 1.1, 1.3] {
        // The picture: its bar painted, its liquid at its level for the
        // GPU, unstrained, no burst.
        let picture =
            common::bubble_picture(&singing, &lyrics, (heard, 0.0), Rect::new(0, 0, 80, 24));
        let liquid = picture.lines[0].liquid.first().expect("its liquid");
        assert_eq!(
            (liquid.pops_at, liquid.begun(), liquid.pressure()),
            (None, false, 0.0)
        );
        assert!((liquid.level - 0.6).abs() < 0.06, "{}", liquid.level);
        let over = painted(&picture, (640, 360), true).1;
        assert!(over.sprays.is_empty() && over.bubbles.len() == 1);
        let strain = over.bubbles[0].strained();
        assert_eq!((strain.pressure, strain.popping), (0.0, None));
        assert!(la_bar(&picture), "{heard}: its bar painted");
        // The stage: both pills, "la"'s liquid in it, nothing straining or
        // bursting.
        let frame = stage_at(&singing, &lyrics, heard, 0.0);
        assert_eq!(word_pills(&frame), (2, 0, 0.0), "{heard}");
        assert!(bubble_sprays(&frame).is_empty(), "{heard}");
        let liquids = frame
            .instances
            .iter()
            .filter(|i| i.params[0] == kind::LIQUID);
        assert!(liquids.count() >= 1, "{heard}");
    }
}

/// A bubble burst, then sought back before what filled it and not sung
/// again: no longer full, it shows again as it is on every frontend (its
/// bar, its pill) and does not burst; sung again, it bursts again, as it
/// did.
#[test]
fn a_bubble_unpopped_by_a_seek_back_shows_again() {
    let (mut singing, lyrics) = common::bubble_song(Level::Easy);
    let now = 0.9;
    // How "la" shows at `now`: its liquid's level, whether the picture
    // paints its bar (and its bursts for the GPU), the stage's word pills
    // and bursts of bubbles, and when its burst started.
    let shows = |singing: &Singing| {
        let picture = common::bubble_picture(singing, &lyrics, (now, 0.0), Rect::new(0, 0, 80, 24));
        let over = painted(&picture, (640, 360), true).1;
        let frame = stage_at(singing, &lyrics, now, 0.0);
        let starts: Vec<u32> = bubble_sprays(&frame)
            .iter()
            .map(|s| s.start.to_bits())
            .collect();
        let liquid = picture.lines[0].liquid.first();
        (
            liquid.map(|b| (b.level * 100.0).round() as u32),
            la_bar(&picture),
            over.sprays.len(),
            word_pills(&frame).0,
            starts,
            liquid.filter(|b| b.begun()).and_then(|b| b.pops_at),
        )
    };
    hear_until(&mut singing, (0.0, now), 0.15);
    let popped = shows(&singing);
    assert_eq!(popped.0, Some(100), "full");
    assert!(!popped.1, "its bar gone");
    assert_eq!(popped.3, 1, "its pill gone");
    // Back to 0.05 s, before it counted, and silent on.
    hear_until(&mut singing, (0.05, now), 0.0);
    let back = shows(&singing);
    assert_eq!(back.0, Some(5), "its liquid as sung, 5 % of it");
    assert!(back.1, "its bar again");
    assert_eq!((back.2, back.3, back.4.len(), back.5), (0, 2, 0, None));
    // Back again and sung as before: it bursts again, as it did.
    hear_until(&mut singing, (0.05, now), 0.15);
    assert_eq!(shows(&singing), popped);
}

/// The stage, the voice 264 ms late: the singer's arrow shows at the
/// playhead while they sing (by the singing's time, as the picture's dot).
#[test]
fn the_arrow_shows_while_singing_however_late_the_voice() {
    let (mut singing, lyrics) = common::bubble_song(Level::Easy);
    let mut stage = Stage::new();
    let area = (0.0, 0.0, 800.0, 450.0);
    let picture = common::bubble_picture(&singing, &lyrics, (0.0, LATE), Rect::new(0, 0, 80, 24));
    let (mut t, mut next, mut shown) = (0.0, 0.0, Vec::new());
    while t <= LATE + 0.4 {
        next = hear_to(&mut singing, next, t, LATE);
        let mut song = Song::new(&singing, lyrics.lines.iter().collect(), Some(3.0));
        song.late = LATE;
        let frame = stage.frame(&song, &picture, area, t, 1.0 / 60.0);
        let arrow = frame.instances.iter().any(|i| i.params[0] == kind::ARROW);
        shown.push((t, arrow));
        t += 1.0 / 60.0;
    }
    // "la" sung from the start for 0.15 s: the arrow from a frame or two
    // after that is heard, for as long as it is sung.
    let at = |from: f64, to: f64| shown.iter().filter(move |s| s.0 >= from && s.0 <= to);
    assert!(at(LATE + 0.04, LATE + 0.14).all(|s| s.1), "{shown:?}");
    assert!(at(LATE + 0.45, LATE + 0.5).all(|s| !s.1));
}

/// The stage sought back a little (under a second) past a word scored and
/// its bubble's burst, and on again: no burst shows twice in a frame, and
/// none comes again (no ghost of the word scored, its bubble bursting
/// once).
#[test]
fn a_seek_back_brings_no_ghost_bursts_on_the_stage() {
    let (mut singing, lyrics) = common::bubble_song(Level::Easy);
    let mut stage = Stage::new();
    let area = (0.0, 0.0, 800.0, 450.0);
    let picture = common::bubble_picture(&singing, &lyrics, (0.0, 0.0), Rect::new(0, 0, 80, 24));
    // On to 1.6 s ("la" scored at 1.3 s), back to 1.0 s, on to 2.2 s.
    let times = (0..=96).chain(60..=132).map(|f| f as f64 / 60.0);
    let (mut next, mut bubble, mut scored) = (0.0, Vec::new(), Vec::new());
    for t in times {
        if next > t + 0.02 {
            next = (t * 100.0).floor() / 100.0;
        }
        next = hear_to(&mut singing, next, t, 0.0);
        let song = Song::new(&singing, lyrics.lines.iter().collect(), Some(3.0));
        let frame = stage.frame(&song, &picture, area, t, 1.0 / 60.0);
        let keys: Vec<(u32, u32)> = frame
            .sprays
            .iter()
            .map(|s| (s.start.to_bits(), s.seed))
            .collect();
        let mut unique = keys.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), keys.len(), "twice in a frame at {t}");
        for s in &frame.sprays {
            let seen = match s.kind {
                Kind::Scored | Kind::StepUp => &mut scored,
                _ => &mut bubble,
            };
            if !seen.contains(&(s.start.to_bits(), s.seed)) {
                seen.push((s.start.to_bits(), s.seed));
            }
        }
    }
    assert_eq!(bubble.len(), 1, "its bubble bursts once: {bubble:?}");
    assert_eq!(scored.len(), 1, "the word scored once: {scored:?}");
}
