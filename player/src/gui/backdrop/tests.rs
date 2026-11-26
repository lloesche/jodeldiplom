//! The backdrop's covers coming and going (fading in over the ground,
//! cross-fading at a song change, fading out to it, two at most, the same
//! cover read again the same), the song's cover kept with them once it is
//! seen to be theirs, where the subject is, and on the GPU: the ground and
//! the covers under the pixmap as far as its words let them through, the
//! whole cover on its card over its blur (at rest the cover itself, not
//! the plate's fill), lit by the stems, their textures dropped with them.

use std::sync::Arc;

use image::{Rgb, RgbImage};
use jodel_gpu::vision::{Depth, Matte};

use super::music::{Hearing, Levels};
use super::*;
use crate::cover::{split, Apart, Parts};

/// A cover `side` square: a grey ground with a disc of `colour` round
/// (`cx`, `cy`) (shares), the disc its subject, nearer; its key in the
/// store `key`.
fn disc_layers(side: usize, colour: [u8; 3], centre: (f32, f32)) -> Layers {
    disc_cover(side, colour, centre, false).1
}

/// A cover as [`disc_layers`] has it, and its picture; with `lettering`, a
/// white bar low on its left that the matte touches faintly (a stray speck
/// of it: the plate there is filled).
fn disc_cover(
    side: usize,
    colour: [u8; 3],
    (cx, cy): (f32, f32),
    lettering: bool,
) -> (RgbImage, Layers) {
    let n = side as f32;
    let inside = |x: usize, y: usize| {
        let (dx, dy) = (x as f32 - cx * n, y as f32 - cy * n);
        (dx * dx + dy * dy).sqrt() < n / 5.0
    };
    let bar = |x: usize, y: usize| {
        let (u, v) = (x as f32 / n, y as f32 / n);
        lettering && (0.12..0.38).contains(&u) && (0.72..0.84).contains(&v)
    };
    let image = RgbImage::from_fn(side as u32, side as u32, |x, y| {
        let (x, y) = (x as usize, y as usize);
        match (inside(x, y), bar(x, y)) {
            (true, _) => Rgb(colour),
            (_, true) => Rgb([250, 250, 250]),
            _ => Rgb([120, 110, 100]),
        }
    });
    let at =
        |f: &dyn Fn(usize, usize) -> f32| (0..side * side).map(|i| f(i % side, i / side)).collect();
    let parts = Parts {
        depth: Depth {
            width: side,
            height: side,
            values: at(&|x, y| if inside(x, y) { 1.0 } else { 4.0 }),
        },
        matte: Matte {
            width: side,
            height: side,
            values: at(&|x, y| match (inside(x, y), bar(x, y)) {
                (true, _) => 1.0,
                (_, true) => 0.1,
                _ => 0.0,
            }),
        },
    };
    let key = format!("{side}-{colour:?}-{cx}-{cy}-{lettering}");
    let layers = Arc::new(Apart::new(key, split(&image, &parts)));
    (image, layers)
}

const DT: f32 = 1.0 / 60.0;
const WINDOW: (u32, u32) = (320, 180);

fn cue(layers: Option<&Layers>) -> Cue {
    Cue {
        layers: layers.cloned(),
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

/// The game or the notes on: the cover over its blur; neither (plain
/// lyrics karaoke): the blur alone. Switched mid-song, the card fades out
/// to the blur and back in, as a cover comes and goes.
#[test]
fn plain_karaoke_has_the_blur_alone_and_a_switch_fades() {
    let mode = |game, notes| {
        Mode::of(&crate::config::Config {
            karaoke_game: game,
            karaoke_notes: notes,
            ..crate::config::Config::default()
        })
    };
    assert_eq!(mode(true, true), Mode::Cover);
    assert_eq!(mode(true, false), Mode::Cover);
    assert_eq!(mode(false, true), Mode::Cover);
    assert_eq!(mode(false, false), Mode::Blur);
    let a = disc_layers(64, [200, 40, 40], (0.5, 0.5));
    let shown = |m: Mode| cue(m.cover(Some(Arc::clone(&a))).as_ref());
    assert!(Mode::Blur.cover(Some(Arc::clone(&a))).is_none());
    let mut v = Visualiser::default();
    // Plain from the start: the blur alone, still.
    assert!(run(&mut v, &shown(Mode::Blur), frames(FADE))
        .slots
        .is_empty());
    assert!(!v.moving());
    let whole = run(&mut v, &shown(Mode::Cover), frames(FADE) + 1);
    assert_eq!(shares(&whole), vec![1.0]);
    // Both off mid-song: halfway out to the blur, then gone.
    let half = run(&mut v, &shown(Mode::Blur), frames(FADE / 2.0));
    let left = shares(&half)[0];
    assert!(left > 0.4 && left < 0.6, "{left}");
    assert!(v.moving());
    assert!(run(&mut v, &shown(Mode::Blur), frames(FADE))
        .slots
        .is_empty());
    // One on again: the card fades back in.
    let back = run(&mut v, &shown(Mode::Cover), frames(FADE / 2.0));
    let s = shares(&back)[0];
    assert!(s > 0.4 && s < 0.6, "{s}");
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
fn the_subject_is_found_and_its_rays_fan_from_its_middle_on_the_card() {
    let a = disc_layers(100, [200, 40, 40], (0.75, 0.4));
    let s = Subject::of(&a.layers);
    assert!((s.focus.0 - 0.75).abs() < 0.03, "{:?}", s.focus);
    assert!(s.focus.1 < 0.4 && s.focus.1 > 0.2, "{:?}", s.focus);
    let (dx, dy) = (s.centre.0 - 0.75, s.centre.1 - 0.4);
    assert!(dx.abs() < 0.02 && dy.abs() < 0.02, "{:?}", s.centre);
    // The disc is the nearest: what is in focus.
    assert!(s.near > 0.9, "{}", s.near);
    // On the card: right of its middle, above it.
    let mut v = Visualiser::default();
    let f = run(&mut v, &cue(Some(&a)), 2);
    let look = f.slots[0].look;
    let card = look.card;
    let want = card[0] + 0.5 * card[2];
    assert!(
        (look.light[0] - want).abs() < 0.03 * card[2] * 2.0,
        "{look:?}"
    );
    assert!(look.light[1] < 0.5 && look.read[3] == 1.0, "{look:?}");
    // What is round it blurred, the disc in focus; the beat pushing it out
    // round its middle.
    assert!(look.defocus[3] == 1.0 && look.defocus[0] > 0.9, "{look:?}");
    assert!(look.defocus[1] > 0.0 && look.focus[2] > 0.7, "{look:?}");
    // A cover without one: its middle, nothing in front, nothing blurred.
    let mut plain = a.layers.clone();
    plain.subject = None;
    let none = Subject::of(&plain);
    assert_eq!((none.focus, none.near), ((0.5, 0.45), 1.0));
}

/// The cue for a cover heard as `heard` (its music, the singer's glow, a
/// phrase landing).
fn heard(layers: &Layers, heard: Hearing) -> Cue {
    Cue {
        heard,
        ..cue(Some(layers))
    }
}

/// The subject pushed out on the beat (none off it, none with the grid
/// unsure), more on a bar's first, its rim lit and the rays swelling then;
/// all of it harder at the top tier's glow, and a phrase landing flaring it.
#[test]
fn the_beat_and_the_tiers_move_the_subject() {
    let a = disc_layers(64, [200, 40, 40], (0.5, 0.5));
    let pulse_at = |beats: f64, confidence: f64, glow: f32| {
        let mut v = Visualiser::default();
        let at = |music| Hearing {
            glow,
            music,
            ..Hearing::default()
        };
        // The glow eased to its row's first.
        run(&mut v, &heard(&a, at(crate::gui::fx::Music::NEUTRAL)), 400);
        let music = crate::gui::fx::Music::steady(beats, 120.0, 4, confidence, 0.0);
        run(&mut v, &heard(&a, at(music)), 1).slots[0].look.pulse
    };
    let bar = pulse_at(8.04, 0.9, 0.0);
    let beat = pulse_at(9.04, 0.9, 0.0);
    let off = pulse_at(8.95, 0.9, 0.0);
    assert!(
        bar[0] > 1.3 * beat[0] && beat[0] > 4.0 * off[0],
        "{bar:?} {beat:?} {off:?}"
    );
    assert!(
        bar[1] > 4.0 * beat[1] && bar[2] > 2.0 && off[2] < 1.35,
        "{bar:?} {beat:?}"
    );
    assert!(pulse_at(8.04, 0.1, 0.0)[0] < 1e-4);
    let top = pulse_at(8.04, 0.9, 1.0);
    assert!(
        top[0] > 1.9 * bar[0] && top[1] > 2.5 * bar[1],
        "{top:?} {bar:?}"
    );
    // A phrase landing: the subject flares for a moment.
    let mut v = Visualiser::default();
    run(&mut v, &cue(Some(&a)), 2);
    let landed = Hearing {
        phrase: true,
        ..Hearing::default()
    };
    run(&mut v, &heard(&a, landed), 1);
    let flared = run(&mut v, &cue(Some(&a)), 8).slots[0].look.pulse;
    assert!(flared[3] > 0.2 && flared[0] > 0.0, "{flared:?}");
    assert!(v.moving());
}

/// A cover `side` square: a red disc (its subject, nearer) in the middle
/// over fine grey stripes two texels wide.
fn striped_layers(side: usize) -> Layers {
    let n = side as f32;
    let inside = |x: usize, y: usize| {
        let (dx, dy) = (x as f32 - n / 2.0, y as f32 - n / 2.0);
        (dx * dx + dy * dy).sqrt() < n / 5.0
    };
    let image = RgbImage::from_fn(side as u32, side as u32, |x, y| {
        if inside(x as usize, y as usize) {
            Rgb([220, 40, 40])
        } else if (x / 2) % 2 == 0 {
            Rgb([200, 200, 200])
        } else {
            Rgb([60, 60, 60])
        }
    });
    let at =
        |f: &dyn Fn(usize, usize) -> f32| (0..side * side).map(|i| f(i % side, i / side)).collect();
    let parts = Parts {
        depth: Depth {
            width: side,
            height: side,
            values: at(&|x, y| if inside(x, y) { 1.0 } else { 4.0 }),
        },
        matte: Matte {
            width: side,
            height: side,
            values: at(&|x, y| if inside(x, y) { 1.0 } else { 0.0 }),
        },
    };
    Arc::new(Apart::new(format!("stripes-{side}"), split(&image, &parts)))
}

/// On the GPU: the stripes round the subject blurred away (the subject cut
/// out), the disc kept sharp and red to its edge (no halo of the plate's
/// fill, no grey ring), and wider on the bar's first beat than off it.
#[test]
fn round_the_subject_is_blurred_and_it_grows_on_the_beat() {
    let Ok(mut gpu) = crate::gui::present::Offscreen::new(false) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let (w, h) = (1024u32, 1024u32);
    let a = striped_layers(256);
    let see = vec![crate::gui::pixels::SEE_THROUGH; (w * h) as usize];
    let mut shown = |beats: Option<f64>| {
        let mut v = Visualiser::default();
        let music = beats.map_or(crate::gui::fx::Music::NEUTRAL, |b| {
            crate::gui::fx::Music::steady(b, 120.0, 4, 0.9, 0.0)
        });
        let cue = heard(
            &a,
            Hearing {
                music,
                ..Hearing::default()
            },
        );
        let mut f = None;
        for _ in 0..frames(FADE * 2.0) {
            f = Some(v.step(&cue, (w, h), DT));
        }
        let f = f.expect("a frame");
        gpu.set_backdrop(Some(&f), w, h);
        (
            gpu.show_pixels(&see, w, h).expect("a frame"),
            f.slots[0].look,
        )
    };
    let (off, look) = shown(None);
    let (on, _) = shown(Some(8.04));
    let rgb = |p: u32| [(p >> 16) & 255, (p >> 8) & 255, p & 255].map(|c| c as i32);
    // The card's pixel at the cover's (u, v).
    let at = |u: f32, v: f32| {
        let x = (look.card[0] - look.card[2] + 2.0 * look.card[2] * u) * w as f32;
        let y = (look.card[1] - look.card[3] + 2.0 * look.card[3] * v) * h as f32;
        (y as u32 * w + x as u32) as usize
    };
    // Stripes four texels a period, over three window pixels a texel:
    // across a row of the plate, next to no difference left.
    let row: Vec<i32> = (0..40)
        .map(|k| rgb(off[at(0.1 + k as f32 / 512.0, 0.15)])[1])
        .collect();
    let swing = row.iter().max().unwrap_or(&0) - row.iter().min().unwrap_or(&0);
    assert!(swing < 25, "{row:?}");
    // The disc red to its edge: just inside it, red, not grey.
    let inner = rgb(off[at(0.5 + 0.19, 0.5)]);
    assert!(inner[0] > 2 * inner[1] && inner[0] > 60, "{inner:?}");
    // Along its middle row, red further out on the bar's first beat (a
    // pixel and a half each side).
    let red = |px: &[u32]| {
        (0..w)
            .filter(|&x| {
                let c = rgb(px[(h / 2 * w + x) as usize]);
                c[0] > c[1] + 60
            })
            .count()
    };
    let (wide, narrow) = (red(&on), red(&off));
    assert!(wide >= narrow + 3, "{wide} {narrow}");
}

#[test]
fn the_song_s_cover_is_kept_once_it_is_seen_to_be_theirs() {
    let (picture, a) = disc_cover(96, [200, 40, 40], (0.5, 0.5), true);
    let art = |id: u64, image: RgbImage| crate::art::Art {
        id,
        image: Arc::new(image),
    };
    assert!(subject::is_of(&picture, &a.layers));
    // The same picture coded again, smaller: theirs too.
    let smaller = image::imageops::resize(&picture, 64, 64, image::imageops::FilterType::Triangle);
    assert!(subject::is_of(&smaller, &a.layers));
    // Another song's cover, or another shape: not.
    let other = RgbImage::from_pixel(96, 96, Rgb([30, 60, 160]));
    let wide = RgbImage::from_pixel(120, 60, Rgb([120, 110, 100]));
    assert!(!subject::is_of(&other, &a.layers) && !subject::is_of(&wide, &a.layers));
    let mut v = Visualiser::default();
    let with = |art: Option<crate::art::Art>| Cue {
        art,
        ..cue(Some(&a))
    };
    // The last song's cover still showing as theirs come: not kept...
    let f = run(&mut v, &with(Some(art(1, other))), 2);
    assert!(f.slots[0].art.is_none());
    // ...the song's own, once it comes: kept, and kept after.
    let f = run(&mut v, &with(Some(art(2, picture))), 1);
    assert_eq!(f.slots[0].art.as_ref().map(|a| a.id), Some(2));
    let f = run(&mut v, &cue(Some(&a)), 1);
    assert_eq!(f.slots[0].art.as_ref().map(|a| a.id), Some(2));
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
/// pixmap where its words let them through, the card brighter than the blur
/// round it, lit by the vocals; its textures dropped when it is gone.
#[test]
fn the_backdrop_draws_on_the_gpu_and_drops_its_textures() {
    let Ok(mut gpu) = crate::gui::present::Offscreen::new(false) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let (w, h) = (256u32, 256u32);
    let a = disc_layers(128, [220, 40, 40], (0.5, 0.5));
    let see = vec![crate::gui::pixels::SEE_THROUGH; (w * h) as usize];
    let frame_of = |layers: Option<&Layers>, levels: Levels| {
        let mut v = Visualiser::default();
        let cue = Cue {
            layers: layers.cloned(),
            heard: Hearing {
                levels,
                ..Hearing::default()
            },
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
    // The ground alone (no cover, the gradient): dark, not black.
    let ground = shown(&mut gpu, Some(&frame_of(None, Levels::default())));
    assert_eq!(gpu.backdrop_held(), 0);
    let g = mean(&ground, 100..156, 100..156);
    assert!(g > 10.0 && g < 60.0, "{g}");
    let quiet = shown(&mut gpu, Some(&frame_of(Some(&a), Levels::default())));
    assert_eq!(gpu.backdrop_held(), 1);
    // The card shows the cover whole, its subject's middle red; round it
    // the blur, darker.
    let mid = quiet[(128 * w + 128) as usize];
    assert!((mid >> 16) & 255 > 2 * ((mid >> 8) & 255), "{mid:08x}");
    let (on_card, round) = (mean(&quiet, 36..56, 36..56), mean(&quiet, 2..12, 2..12));
    assert!(round > 4.0 && on_card > round * 1.5, "{on_card} {round}");
    // The vocals light the subject.
    let sung = Levels {
        vocals: 1.0,
        ..Levels::default()
    };
    let lit = shown(&mut gpu, Some(&frame_of(Some(&a), sung)));
    let disc = |px: &[u32]| mean(px, 112..144, 112..144);
    assert!(
        disc(&lit) > disc(&quiet) * 1.15,
        "{} {}",
        disc(&lit),
        disc(&quiet)
    );
    // A pixmap word over it: its own colour with as much of the backdrop as
    // its top byte lets through.
    let half: Vec<u32> = (0..w * h).map(|_| 0x8000_0000 | 0x0020_2020).collect();
    gpu.set_backdrop(Some(&frame_of(Some(&a), Levels::default())), w, h);
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

/// A cover whose matte touched its lettering faintly (the plate filled
/// there): at rest the card shows the lettering from the cover itself,
/// where the plate alone smudges it away.
#[test]
fn the_card_at_rest_is_the_cover_itself() {
    let Ok(mut gpu) = crate::gui::present::Offscreen::new(false) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let (w, h) = (256u32, 256u32);
    let (picture, a) = disc_cover(128, [220, 40, 40], (0.6, 0.4), true);
    let see = vec![crate::gui::pixels::SEE_THROUGH; (w * h) as usize];
    let mut bright = |art: Option<RgbImage>| {
        let mut v = Visualiser::default();
        let cue = Cue {
            art: art.map(|image| crate::art::Art {
                id: 1,
                image: Arc::new(image),
            }),
            ..cue(Some(&a))
        };
        let mut f = None;
        for _ in 0..frames(FADE * 2.0) {
            f = Some(v.step(&cue, (w, h), DT));
        }
        gpu.set_backdrop(f.as_ref(), w, h);
        let px = gpu.show_pixels(&see, w, h).expect("a frame");
        // The bar's middle on the card (its cover units 0.25, 0.78).
        let card = f.expect("a frame").slots[0].look.card;
        let at = |c: usize, u: f32| {
            let (mid, half, size) = (card[c], card[c + 2], [w, h][c] as f32);
            ((mid - half + 2.0 * half * u) * size) as u32
        };
        let p = px[(at(1, 0.78) * w + at(0, 0.25)) as usize];
        ((p >> 16) & 255) + ((p >> 8) & 255) + (p & 255)
    };
    let real = bright(Some(picture));
    let filled = bright(None);
    assert!(real > filled + 90, "{real} {filled}");
}
