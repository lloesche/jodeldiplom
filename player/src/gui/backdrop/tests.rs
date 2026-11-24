//! The backdrop's covers coming and going (fading in over the ground,
//! cross-fading at a song change, fading out to it, two at most, the same
//! cover read again the same), where the subject is, and on the GPU: the
//! ground and the covers under the pixmap as far as its words let them
//! through, calmer under the text, lit by the stems, their textures
//! dropped with them.

use std::sync::Arc;

use image::{Rgb, RgbImage};
use jodel_gpu::vision::{Depth, Matte};

use super::music::Levels;
use super::*;
use crate::cover::{split, Apart, Parts};

/// A cover `side` square: a grey ground with a disc of `colour` round
/// (`cx`, `cy`) (shares), the disc its subject, nearer; its key in the
/// store `key`.
pub(crate) fn disc_layers(side: usize, colour: [u8; 3], (cx, cy): (f32, f32)) -> Layers {
    let inside = |x: usize, y: usize| {
        let (dx, dy) = (x as f32 - cx * side as f32, y as f32 - cy * side as f32);
        (dx * dx + dy * dy).sqrt() < side as f32 / 5.0
    };
    let image = RgbImage::from_fn(side as u32, side as u32, |x, y| {
        if inside(x as usize, y as usize) {
            Rgb(colour)
        } else {
            Rgb([120, 110, 100])
        }
    });
    let at = |f: &dyn Fn(bool) -> f32| {
        (0..side * side)
            .map(|i| f(inside(i % side, i / side)))
            .collect()
    };
    let parts = Parts {
        depth: Depth {
            width: side,
            height: side,
            values: at(&|i| if i { 1.0 } else { 4.0 }),
        },
        matte: Matte {
            width: side,
            height: side,
            values: at(&|i| if i { 1.0 } else { 0.0 }),
        },
    };
    let key = format!("{side}-{colour:?}-{cx}-{cy}");
    Arc::new(Apart::new(key, split(&image, &parts)))
}

const DT: f32 = 1.0 / 60.0;
const WINDOW: (u32, u32) = (320, 180);

fn cue(layers: Option<&Layers>) -> Cue {
    Cue {
        layers: layers.cloned(),
        text: [-1.0; 2],
        ..Cue::default()
    }
}

/// `n` frames of `cue`: the last.
fn run(v: &mut Visualiser, cue: &Cue, n: usize) -> BackdropFrame {
    let mut last = None;
    for _ in 0..n {
        last = Some(v.step(cue, WINDOW, DT));
    }
    last.expect("a frame")
}

fn frames(secs: f32) -> usize {
    (secs / DT).round() as usize
}

/// How much of each cover a frame shows in the end, its ground having the
/// rest: each drawn over what is under it by its alpha.
fn shares(f: &BackdropFrame) -> Vec<f32> {
    let mut out: Vec<f32> = Vec::new();
    for s in &f.slots {
        let a = s.look.cam[3];
        out.iter_mut().for_each(|x| *x *= 1.0 - a);
        out.push(a);
    }
    out
}

#[test]
fn a_cover_fades_in_over_the_ground_and_back_out_to_it() {
    let a = disc_layers(64, [200, 40, 40], (0.5, 0.5));
    let mut v = Visualiser::default();
    // Still being prepared: the ground alone.
    assert!(run(&mut v, &cue(None), 10).slots.is_empty());
    let half = run(&mut v, &cue(Some(&a)), frames(FADE / 2.0));
    assert_eq!(half.slots.len(), 1);
    assert!((shares(&half)[0] - 0.5).abs() < 0.05, "{:?}", shares(&half));
    // Its frames come by themselves while it fades, not once it is still.
    assert!(v.moving());
    let whole = run(&mut v, &cue(Some(&a)), frames(FADE));
    assert_eq!(shares(&whole), vec![1.0]);
    assert!(!v.moving());
    // Gone (another song without its layers yet): out to the ground.
    let going = run(&mut v, &cue(None), frames(FADE / 2.0));
    let left = shares(&going)[0];
    assert!(left < 0.6 && left > 0.4, "{left}");
    assert!(run(&mut v, &cue(None), frames(FADE)).slots.is_empty());
    assert_eq!(v.holds(), 0);
}

#[test]
fn a_song_change_cross_fades_one_cover_into_the_next() {
    let a = disc_layers(64, [200, 40, 40], (0.5, 0.5));
    let b = disc_layers(64, [40, 40, 200], (0.4, 0.4));
    let mut v = Visualiser::default();
    run(&mut v, &cue(Some(&a)), frames(FADE) + 1);
    let mid = run(&mut v, &cue(Some(&b)), frames(FADE / 2.0));
    // Both: the old behind, the new over it, half each, none of the ground.
    assert_eq!(mid.slots.len(), 2);
    assert_eq!(mid.slots[0].layers.key, a.key);
    assert_eq!(mid.slots[1].layers.key, b.key);
    let s = shares(&mid);
    assert!(
        (s[0] - 0.5).abs() < 0.05 && (s[1] - 0.5).abs() < 0.05,
        "{s:?}"
    );
    assert!((s[0] + s[1] - 1.0).abs() < 1e-3, "{s:?}");
    let done = run(&mut v, &cue(Some(&b)), frames(FADE));
    assert_eq!(done.slots.len(), 1);
    assert_eq!(done.slots[0].layers.key, b.key);
}

#[test]
fn the_same_cover_read_again_is_the_same_and_two_at_most_show() {
    let a = disc_layers(48, [200, 40, 40], (0.5, 0.5));
    // The same cover result, read again for the album's next song.
    let again = Arc::new((*a).clone());
    let b = disc_layers(48, [40, 200, 40], (0.5, 0.5));
    let c = disc_layers(48, [40, 40, 200], (0.5, 0.5));
    let mut v = Visualiser::default();
    run(&mut v, &cue(Some(&a)), frames(FADE) + 1);
    let same = run(&mut v, &cue(Some(&again)), 1);
    assert_eq!(shares(&same), vec![1.0]);
    assert!(!v.moving());
    run(&mut v, &cue(Some(&b)), frames(FADE / 4.0));
    // Back to a: it comes forward from most of the way shown.
    let back = run(&mut v, &cue(Some(&a)), 1);
    assert_eq!(back.slots[1].layers.key, a.key);
    assert!(shares(&back)[1] > 0.6, "{:?}", shares(&back));
    // A third: the one showing least makes way.
    let third = run(&mut v, &cue(Some(&c)), 1);
    assert_eq!(third.slots.len(), 2);
    assert_eq!(third.slots[0].layers.key, a.key);
    assert_eq!(third.slots[1].layers.key, c.key);
}

#[test]
fn the_subject_is_found_and_the_cut_keeps_it_in_a_tall_window() {
    let a = disc_layers(100, [200, 40, 40], (0.75, 0.4));
    let s = Subject::of(&a.layers);
    assert!((s.focus.0 - 0.75).abs() < 0.03, "{:?}", s.focus);
    assert!(s.light.1 < 0.4 && s.light.1 > 0.2, "{:?}", s.light);
    let fit = look::fit((600.0, 1200.0), (1.0, 1.0), s.focus, 1.04);
    assert!(fit[2] < 0.75 && fit[2] + fit[0] > 0.75, "{fit:?}");
    // A cover without one: its middle.
    let mut plain = a.layers.clone();
    plain.subject = None;
    assert_eq!(Subject::of(&plain).focus, (0.5, 0.45));
}

#[test]
fn the_palette_lights_are_its_vivid_colours_at_full_strength() {
    let swatch = |rgb, share| crate::cover::Swatch { rgb, share };
    let palette = [
        swatch([60, 60, 64], 0.6),
        swatch([220, 40, 120], 0.2),
        swatch([30, 120, 200], 0.15),
    ];
    let [accent, second, ground] = lights(&palette);
    assert!(accent[0] == 1.0 && accent[1] < 0.1, "{accent:?}");
    assert!(second[2] == 1.0, "{second:?}");
    assert!(ground.iter().all(|&c| c > 0.8), "{ground:?}");
    assert_eq!(lights(&[])[0], [1.0, 0.85, 0.65]);
}

/// The GPU's frames of a red disc cover: the ground and the cover under the
/// pixmap where its words let them through, calmer under the text, lit by
/// the vocals; its textures dropped when it is gone.
#[test]
fn the_backdrop_draws_on_the_gpu_and_drops_its_textures() {
    let Ok(mut gpu) = crate::gui::present::Offscreen::new(false) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let (w, h) = (256u32, 256u32);
    let a = disc_layers(128, [220, 40, 40], (0.5, 0.5));
    let see = vec![crate::gui::raster::SEE_THROUGH; (w * h) as usize];
    let frame_of = |layers: Option<&Layers>, levels: Levels, text: [f32; 2]| {
        let mut v = Visualiser::default();
        let cue = Cue {
            layers: layers.cloned(),
            levels,
            text,
            ..Cue::default()
        };
        let mut f = None;
        for _ in 0..frames(FADE * 2.0) {
            f = Some(v.step(&cue, (w, h), DT));
        }
        f.expect("a frame")
    };
    let shown = |gpu: &mut crate::gui::present::Offscreen, f: Option<&BackdropFrame>| {
        gpu.set_backdrop(f, w, h);
        gpu.show_pixels(&see, w, h).expect("a frame")
    };
    let mean = |px: &[u32], rows: std::ops::Range<u32>, cols: std::ops::Range<u32>| {
        let mut sum = 0.0;
        let mut n = 0.0;
        for y in rows {
            for x in cols.clone() {
                let p = px[(y * w + x) as usize];
                sum += f64::from((p >> 16) & 255) + f64::from((p >> 8) & 255) + f64::from(p & 255);
                n += 3.0;
            }
        }
        sum / n
    };
    let none = [-1.0; 2];
    // The ground alone (no cover, the gradient): dark, not black.
    let ground = shown(&mut gpu, Some(&frame_of(None, Levels::default(), none)));
    assert_eq!(gpu.backdrop_held(), 0);
    let g = mean(&ground, 100..156, 100..156);
    assert!(g > 10.0 && g < 60.0, "{g}");
    let quiet = shown(&mut gpu, Some(&frame_of(Some(&a), Levels::default(), none)));
    assert_eq!(gpu.backdrop_held(), 1);
    // The cover shows, and its subject's middle is red.
    let corner = mean(&quiet, 8..40, 8..40);
    assert!(corner > 30.0, "{corner}");
    let mid = quiet[(128 * w + 128) as usize];
    assert!((mid >> 16) & 255 > 2 * ((mid >> 8) & 255), "{mid:08x}");
    // Calmer under text across the top half.
    let calm = shown(
        &mut gpu,
        Some(&frame_of(Some(&a), Levels::default(), [0.0, 0.5])),
    );
    let (top, top_quiet) = (mean(&calm, 20..60, 20..236), mean(&quiet, 20..60, 20..236));
    assert!(top < top_quiet * 0.8, "{top} {top_quiet}");
    // The vocals light the subject; under the text they do not.
    let sung = Levels {
        vocals: 1.0,
        ..Levels::default()
    };
    let lit = shown(&mut gpu, Some(&frame_of(Some(&a), sung, none)));
    let disc = |px: &[u32]| mean(px, 112..144, 112..144);
    assert!(
        disc(&lit) > disc(&quiet) * 1.15,
        "{} {}",
        disc(&lit),
        disc(&quiet)
    );
    let whole = [0.0, 1.0];
    let calm_quiet = shown(
        &mut gpu,
        Some(&frame_of(Some(&a), Levels::default(), whole)),
    );
    let calm_lit = shown(&mut gpu, Some(&frame_of(Some(&a), sung, whole)));
    assert!(
        disc(&calm_lit) <= disc(&calm_quiet) + 1.0,
        "{} {}",
        disc(&calm_lit),
        disc(&calm_quiet)
    );
    // A pixmap word over it: its own colour with as much of the backdrop as
    // its top byte lets through.
    let half: Vec<u32> = (0..w * h).map(|_| 0x8000_0000 | 0x0020_2020).collect();
    gpu.set_backdrop(Some(&frame_of(Some(&a), Levels::default(), none)), w, h);
    let over = gpu.show_pixels(&half, w, h).expect("a frame");
    for at in [(128u32, 128u32), (20, 20)] {
        let i = (at.1 * w + at.0) as usize;
        for shift in [16, 8, 0] {
            let (o, q) = ((over[i] >> shift) & 255, (quiet[i] >> shift) & 255);
            let want = 0x20 as f32 + q as f32 * 128.0 / 255.0;
            assert!((o as f32 - want).abs() <= 3.0, "{at:?}: {o} {want}");
        }
    }
    // Gone: the pixmap alone, nothing held.
    let gone = shown(&mut gpu, None);
    assert_eq!(gpu.backdrop_held(), 0);
    assert_eq!(gone[(10 * w + 10) as usize] & 0x00ff_ffff, 0);
}
