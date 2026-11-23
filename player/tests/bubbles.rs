//! The note bubbles of Karaoke mode as each frontend draws them: a word
//! counted on easy swept full and bursting, in the terminal (a snapshot in
//! `tests/snapshots/`, `UPDATE_SNAPSHOTS=1` to rewrite it), in the picture
//! painted on the processor (its bursts drawn by the GPU over it), and on
//! the GPU's stage; the voice heard as it comes or 264 ms late (the
//! bubbles go by the singing's own time, so the sweep and the burst are
//! seen whole either way). With `JODEL_FRAMES`
//! set to a folder, the window's frames are saved there as PNGs.

mod common;

use std::path::PathBuf;

use jodeldiplom::core::singing::{Level, Singing};
use jodeldiplom::view::bubbles::{BURST, SWEEP};
use jodeldiplom::view::karaoke::KaraokePicture;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;

/// When "la" of [`common::bubble_song`] counts on easy: its 11th frame hit
/// (a tenth of the 100 it has, and one more).
const COUNTED: f64 = 0.10;

fn snapshot(name: &str, text: &str) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/snapshots")
        .join(format!("{name}.txt"));
    if std::env::var_os("UPDATE_SNAPSHOTS").is_some() {
        std::fs::write(&path, text).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("no snapshot {}", path.display()))
        .replace("\r\n", "\n");
    assert_eq!(expected, text, "{name} differs from its snapshot");
}

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

fn rows(buf: &Buffer) -> String {
    let area = buf.area;
    (0..area.height)
        .map(|y| {
            let row: String = (0..area.width).map(|x| buf[(x, y)].symbol()).collect();
            format!("{}\n", row.trim_end())
        })
        .collect()
}

/// A word counted on easy in the terminal, the voice heard as it comes
/// and 264 ms late: its bubble swept full in gold (seen part way, its
/// first half gold, then whole), then bursting, brighter, a flourish
/// either side (✦, then ✧, then ·); once over, full and gold, the
/// flourish gone. The word not sung stays as it was.
#[test]
fn a_bubble_fills_and_bursts_in_the_terminal() {
    use jodeldiplom::view::karaoke::{karaoke_cells, HIT, NOTE_AHEAD};
    for late in [0.0, LATE] {
        let (mut singing, lyrics) = common::bubble_song(Level::Easy);
        let area = Rect::new(0, 0, 24, 5);
        let cells = |singing: &Singing, heard: f64| {
            let picture = common::bubble_picture(singing, &lyrics, (heard, late), area);
            let mut buf = Buffer::empty(area);
            karaoke_cells(&mut buf, &picture);
            buf
        };
        let notes_row =
            |buf: &Buffer| rows(buf).lines().position(|r| r.contains("la lo")).unwrap() as u16 + 1;
        // Half way through the sweep, as heard: the voice `late` behind.
        let half = late + COUNTED + SWEEP / 2.0 + 0.005;
        let next = hear_to(&mut singing, 0.0, half, late);
        let sweeping = cells(&singing, half);
        let y = notes_row(&sweeping);
        let row: Vec<&str> = (0..area.width).map(|x| sweeping[(x, y)].symbol()).collect();
        // "la" at the line's middle (the singer's dot on its first cell),
        // "lo" higher: half swept, its second cell not yet gold.
        let first = row.iter().position(|s| *s == "─" || *s == "●").unwrap();
        let x = first as u16;
        assert_eq!(row[first], "●", "late {late}");
        assert_ne!(sweeping[(x + 1, y)].fg, HIT, "late {late}: not yet");
        // Nearly swept (the singer gone quiet): both gold, no burst yet.
        let nearly = late + COUNTED + SWEEP * 0.97;
        let next = hear_to(&mut singing, next, nearly, late);
        let swept = cells(&singing, nearly);
        assert_eq!(
            (swept[(x, y)].fg, swept[(x + 1, y)].fg),
            (HIT, HIT),
            "late {late}"
        );
        assert_eq!(swept[(x - 1, y)].symbol(), " ");
        // Bursting, a fifth through.
        let at = late + COUNTED + SWEEP + 0.2 * BURST;
        hear_to(&mut singing, next, at, late);
        let bursting = cells(&singing, at);
        let row: Vec<&str> = (0..area.width).map(|x| bursting[(x, y)].symbol()).collect();
        assert_eq!((row[first - 1], row[first + 2]), ("✦", "✦"), "{row:?}");
        // Brighter than gold while it pops.
        assert_ne!(bursting[(x, y)].fg, HIT);
        // Over: full, in gold; "lo" still to come.
        let next = ((at - late) * 100.0).floor() / 100.0 + 0.01;
        hear_to(&mut singing, next, 0.95 + late, late);
        let full = cells(&singing, 0.95 + late);
        assert_eq!((full[(x, y)].fg, full[(x + 1, y)].fg), (HIT, HIT));
        assert_eq!(full[(x - 1, y)].symbol(), " ");
        assert_eq!(full[(x + 3, y)].fg, NOTE_AHEAD);
        if late == 0.0 {
            let text = format!("bursting:\n{}full:\n{}", rows(&bursting), rows(&full));
            snapshot("karaoke_burst", &text);
        }
    }
}

/// A word counted on easy in the window's picture, the voice 264 ms late:
/// the bubble's bar sweeps full (seen part way), and its burst is handed
/// to the GPU ([`jodeldiplom::gui::particles`]): the picture itself is the
/// same with or without it. Frames drawn only where they changed are the
/// frames drawn afresh all through.
#[test]
fn a_bubble_bursts_over_the_picture_not_in_it() {
    use jodeldiplom::gui::karaoke::{paint, Backdrop, BigText, Layer, Target};
    let (mut singing, lyrics) = common::bubble_song(Level::Easy);
    let (w, h) = (640usize, 360usize);
    let backdrop = Backdrop::new(None, w, h);
    let area = Rect::new(0, 0, 80, 24);
    let fresh = |picture: &KaraokePicture| {
        let mut pixels = vec![0u32; w * h];
        let mut target = Target {
            pixels: &mut pixels,
            width: w,
            height: h,
        };
        let mut layer = Layer::default();
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
        (pixels, layer.sprays().to_vec())
    };
    let (mut kept, mut text, mut layer) = (vec![0u32; w * h], BigText::default(), Layer::default());
    let (mut heard, mut next, mut seen, mut part_way, mut k) = (0.0, 0.0, None, false, 0);
    while heard < 0.95 + LATE {
        heard += 0.015;
        next = hear_to(&mut singing, next, heard, LATE);
        let picture = common::bubble_picture(&singing, &lyrics, (heard, LATE), area);
        let mut target = Target {
            pixels: &mut kept,
            width: w,
            height: h,
        };
        let full = k == 0;
        paint(
            &mut target,
            &mut text,
            &mut layer,
            &backdrop,
            &picture,
            (0, 0, w, h),
            None,
            full,
        );
        let (afresh, sprays) = fresh(&picture);
        let differ = kept.iter().zip(&afresh).filter(|(a, b)| a != b).count();
        assert_eq!(differ, 0, "at {heard}");
        assert_eq!(layer.sprays(), &sprays[..]);
        let line = &picture.lines[0];
        let filled: f32 = line.hits.iter().map(|h| h.1 - h.0).sum();
        part_way |= filled > 0.6 && filled < 1.9;
        if !line.bursts.is_empty() && seen.is_none() {
            seen = Some(picture.clone());
            // One burst, as the bubble's bar is: gold, from its burst's
            // start on the singing's clock.
            assert_eq!(sprays.len(), 1);
            let start = f64::from(sprays[0].start);
            assert!((start - (COUNTED + SWEEP)).abs() < 0.011, "{start}");
        }
        k += 1;
    }
    assert!(part_way, "the sweep seen part way");
    let picture = seen.expect("a burst");
    let mut calm = picture.clone();
    calm.lines.iter_mut().for_each(|l| l.bursts.clear());
    let ((a, sprays), (b, none)) = (fresh(&picture), fresh(&calm));
    assert!(a == b, "the picture is the same with its burst");
    assert!(!sprays.is_empty() && none.is_empty());
}

/// The picture's burst drawn by the GPU over it (offscreen): each of its
/// sparkles lights the picture where the processor's reference of the
/// shader ([`jodeldiplom::gui::particles::sprite`]) puts it, and nothing
/// far from them changes. Skipped (with a note) without a GPU.
#[test]
fn the_gpu_draws_the_bursts_over_the_picture() {
    use jodeldiplom::gui::karaoke::{paint, Backdrop, BigText, Layer, Target};
    use jodeldiplom::gui::particles::{sprite, with_sprays, QUADS};
    use jodeldiplom::gui::present::Offscreen;
    let mut gpu = match Offscreen::new(false) {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("no GPU ({e}): skipped");
            return;
        }
    };
    let (mut singing, lyrics) = common::bubble_song(Level::Easy);
    let (w, h) = (800usize, 450usize);
    let at = COUNTED + SWEEP + 0.3 * BURST;
    common::sing_bubbles(&mut singing, 0.0, at);
    let picture = common::bubble_picture(&singing, &lyrics, (at, 0.0), Rect::new(0, 0, 80, 24));
    let backdrop = Backdrop::new(None, w, h);
    let mut pixels = vec![0u32; w * h];
    let mut target = Target {
        pixels: &mut pixels,
        width: w,
        height: h,
    };
    let mut layer = Layer::default();
    let mut text = BigText::default();
    paint(
        &mut target,
        &mut text,
        &mut layer,
        &backdrop,
        &picture,
        (0, 0, w, h),
        None,
        true,
    );
    let mut sprays = layer.sprays().to_vec();
    assert_eq!(sprays.len(), 1, "the word's bubble");
    // And two words scored, the second as the multiplier goes up.
    use jodeldiplom::gui::particles::{Kind, Spray};
    for (k, kind) in [(0, Kind::Scored), (1, Kind::ScoredUp)] {
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
    let show = |gpu: &mut Offscreen, sprays: &[jodeldiplom::gui::particles::Spray]| {
        let frame = with_sprays(None, sprays, at as f32);
        gpu.set_stage(frame.as_ref(), w as u32, h as u32);
        gpu.show_pixels(&pixels, w as u32, h as u32)
            .expect("a frame")
    };
    let (lit, plain) = (show(&mut gpu, &sprays), show(&mut gpu, &[]));
    save_frame("picture_burst.png", &lit, w);
    let lum = |p: u32| ((p >> 16) & 255) + ((p >> 8) & 255) + (p & 255);
    let mut near = vec![false; w * h];
    let mut sparkles = 0;
    for s in &sprays {
        for q in 0..QUADS {
            let Some(sp) = sprite(s, q, at as f32) else {
                continue;
            };
            let (x, y) = (sp.at.0.round() as usize, sp.at.1.round() as usize);
            let reach = (sp.half.0.max(sp.half.1) * 2.5 + 3.0) as usize;
            for yy in y.saturating_sub(reach)..(y + reach).min(h) {
                for xx in x.saturating_sub(reach)..(x + reach).min(w) {
                    near[yy * w + xx] = true;
                }
            }
            if !sp.pop && sp.alpha > 0.3 {
                sparkles += 1;
                let i = y * w + x;
                assert!(lum(lit[i]) > lum(plain[i]) + 60, "sparkle {q} at {x},{y}");
            }
        }
    }
    assert!(sparkles >= 12, "{sparkles}");
    let stray = (0..w * h)
        .filter(|&i| !near[i] && lum(lit[i]) > lum(plain[i]) + 8)
        .count();
    assert_eq!(stray, 0, "light only where the sprites are");
}

/// The stage on the GPU, the voice 264 ms late: a word counted on easy
/// pops its pill, white hot, and sheds sparks; sung alike on medium (not
/// enough to fill it), it does not. Skipped (with a note) without a GPU.
#[test]
fn the_stage_bursts_a_word_counted() {
    use jodeldiplom::gui::present::Offscreen;
    use jodeldiplom::gui::stage::{Song, Stage};
    let mut gpu = match Offscreen::new(false) {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("no GPU ({e}): skipped");
            return;
        }
    };
    let (w, h) = (800u32, 450u32);
    let area = (0.0, 0.0, w as f32, h as f32);
    let now = LATE + COUNTED + SWEEP + 0.25 * BURST;
    let mut frame_at = |level: Level| {
        let (mut singing, lyrics) = common::bubble_song(level);
        let mut stage = Stage::new();
        let picture =
            common::bubble_picture(&singing, &lyrics, (now, LATE), Rect::new(0, 0, 80, 24));
        // Frames at 60 a second up to now, as the window draws them.
        let (mut t, mut next, mut frame) = (0.0, 0.0, None);
        while t <= now + 1e-9 {
            next = hear_to(&mut singing, next, t, LATE);
            let mut song = Song::new(&singing, lyrics.lines.iter().collect(), Some(3.0));
            song.late = LATE;
            frame = Some(stage.frame(&song, &picture, area, t, 1.0 / 60.0));
            t += 1.0 / 60.0;
        }
        let frame = frame.expect("a frame");
        gpu.set_stage(Some(&frame), w, h);
        let blank = vec![0u32; (w * h) as usize];
        (
            gpu.show_pixels(&blank, w, h).expect("a frame"),
            frame.sprays.len(),
        )
    };
    let (burst, sprays) = frame_at(Level::Easy);
    let (calm, none) = frame_at(Level::Medium);
    assert_eq!((sprays, none), (1, 0), "the word's bubble's sparkles");
    save_frame("stage_burst.png", &burst, w as usize);
    save_frame("stage_calm.png", &calm, w as usize);
    let bright = |v: &[u32]| {
        v.iter()
            .filter(|&&p| (p >> 16) & 255 > 230 && (p >> 8) & 255 > 200 && p & 255 > 120)
            .count()
    };
    let (b, c) = (bright(&burst), bright(&calm));
    assert!(b > c + 200, "{b} against {c}");
}

/// The stage, the voice 264 ms late: the singer's arrow shows at the
/// playhead while they sing (by the singing's time, as the picture's dot).
#[test]
fn the_arrow_shows_while_singing_however_late_the_voice() {
    use jodeldiplom::gui::stage::{Song, Stage};
    use jodeldiplom::gui::stage_gpu::kind;
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
    use jodeldiplom::gui::particles::Kind;
    use jodeldiplom::gui::stage::{Song, Stage};
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
                Kind::Scored | Kind::ScoredUp => &mut scored,
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
