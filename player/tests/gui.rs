//! The window's drawing path without a window: the shared widgets
//! rasterised into pixels, the cover and the spectrum painted at their
//! cell rects.

mod common;

use std::sync::Arc;

use iced::keyboard::{key::Named, Key as IcedKey, Modifiers};
use image::Rgb;
use jodeldiplom::art::{sample_cover, Art};
use jodeldiplom::core::layout::{self, Layout};
use jodeldiplom::core::{update, Screen, State};
use jodeldiplom::gui::{self, font_px, Canvas};
use jodeldiplom::icon::Style;
use jodeldiplom::spectrum::{sine, tap_frames_for, Analyzer, SpectrumFrame};
use jodeldiplom::view::{self, colors, Plot, Scene};
use ratatui::layout::Rect;
use ratatui::style::Color;

fn rgb(color: Color) -> [u8; 3] {
    match color {
        Color::Rgb(r, g, b) => [r, g, b],
        _ => unreachable!(),
    }
}

fn bg() -> Rgb<u8> {
    Rgb(rgb(colors::BG))
}

fn spectrum(rate: u32, hz: f64) -> SpectrumFrame {
    // The tone over a low noise floor, so the curve runs across the plot.
    let mut seed = 7u32;
    let frames: Vec<_> = sine(rate, hz, 0.7, tap_frames_for(rate))
        .into_iter()
        .map(|[l, r]| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let noise = (seed as i32) >> 20;
            [l + noise, r + noise]
        })
        .collect();
    let mut analyzer = Analyzer::new();
    analyzer.analyse(rate, &frames, 0.0);
    analyzer.frame()
}

fn draw(canvas: &mut Canvas, state: &State, spectrum: &SpectrumFrame, art: Option<&Art>) {
    let scene = Scene {
        state,
        spectrum,
        pointer: None,
        heard: 0.0,
    };
    canvas.draw(&scene, art);
}

fn sized(canvas: &mut Canvas, w: u32, h: u32) -> State {
    let grid = canvas.fit(w, h);
    let mut state = common::playing_state(grid.0, grid.1);
    state.size = grid;
    state
}

fn pixel_rect(canvas: &Canvas, rect: Rect) -> (u32, u32, u32, u32) {
    let (cw, ch) = canvas.cell();
    (
        u32::from(rect.x) * cw as u32,
        u32::from(rect.y) * ch as u32,
        u32::from(rect.width) * cw as u32,
        u32::from(rect.height) * ch as u32,
    )
}

#[test]
fn the_cover_and_the_spectrum_are_painted_in_their_cells() {
    let mut canvas = Canvas::new(80, 24, font_px(1.0));
    let state = sized(&mut canvas, 1280, 800);
    let art = Art {
        id: 7,
        image: Arc::new(sample_cover(256)),
    };
    let spectrum = spectrum(176_400, 1_000.0);
    draw(&mut canvas, &state, &spectrum, Some(&art));
    let image = canvas.image();
    let (cw, ch) = canvas.cell();
    assert_eq!(image.width() as usize, usize::from(state.size.0) * cw);
    assert_eq!(image.height() as usize, usize::from(state.size.1) * ch);
    let Layout::Main(m) = layout::compute(&state) else {
        panic!("not the main screen")
    };

    // The cover fills its square from under the list pane's top border
    // line (half a cell down), at the left: its centre pixel is the
    // cover's centre, and above it there is only background.
    let (x, y, w, h) = pixel_rect(&canvas, m.cover.unwrap());
    let top = y + ch as u32 / 2;
    let side = w.min(h - ch as u32 / 2);
    let center = image.get_pixel(x + side / 2, top + side / 2);
    let expected = sample_cover(side).get_pixel(side / 2, side / 2).0;
    let close = |a: [u8; 3], b: [u8; 3]| a.iter().zip(b).all(|(p, q)| p.abs_diff(q) <= 24);
    assert!(close(center.0, expected), "{center:?} vs {expected:?}");
    let covered = (0..side)
        .step_by(4)
        .filter(|&i| *image.get_pixel(x + i, top + i) != bg())
        .count();
    assert!(covered as u32 >= side / 4 - 2, "{covered}");
    for py in y..top {
        assert_eq!(*image.get_pixel(x + side / 2, py), bg(), "row {py}");
    }
    assert_ne!(*image.get_pixel(x + side / 2, top + 1), bg());

    // The spectrum is a dotted curve in the zone's colour, not a filled
    // area: near the tone it reaches the top quarter, under the curve
    // there is only background and grid, and the curve covers a small part
    // of the plot.
    let plot = Plot::new(
        m.bottom.spectrum.unwrap(),
        m.bottom.labels,
        &spectrum,
        view::zones(&state),
    );
    let (sx, sy, sw, sh) = pixel_rect(&canvas, plot.area);
    assert!(
        plot.area.x > m.bottom.spectrum.unwrap().x,
        "room for the dB labels"
    );
    let curve = rgb(colors::SOURCE);
    let near = |p: Rgb<u8>, c: [u8; 3]| p.0.iter().zip(c).all(|(a, b)| a.abs_diff(b) <= 40);
    let on_curve = |col: u32, rows: std::ops::Range<u32>| {
        rows.filter(|&r| near(*image.get_pixel(sx + col, sy + r), curve))
            .count() as u32
    };
    let tone = spectrum.x_of(1_000.0, sw as usize).round() as u32;
    let peak: u32 = (tone.saturating_sub(4)..tone + 4)
        .map(|col| on_curve(col, 0..sh / 4))
        .sum();
    assert!(peak > 0, "the tone's peak is drawn");
    let under: u32 = (tone.saturating_sub(4)..tone + 4)
        .map(|col| on_curve(col, sh * 3 / 4..sh))
        .sum();
    assert_eq!(under, 0, "nothing is filled under the curve");
    let all: u32 = (0..sw).map(|col| on_curve(col, 0..sh)).sum();
    assert!(all > sw / 4 && all < sw * sh / 10, "{all} of {}", sw * sh);
    // No sound at the far right: nothing drawn there but grid.
    assert_eq!(on_curve(sw - 2, 0..sh), 0);
    let grid = rgb(colors::GRID);
    let grid_pixels = (0..sw)
        .flat_map(|x| (0..sh).map(move |y| (x, y)))
        .filter(|&(x, y)| image.get_pixel(sx + x, sy + y).0 == grid)
        .count();
    assert!(grid_pixels > 100, "{grid_pixels}");
    // The axes are labelled.
    let text = canvas.text();
    for label in ["0 dB", "-40", "-120", "20 Hz", "1k", "10k", "20k", "88k"] {
        assert!(text.contains(label), "missing {label:?}");
    }

    // Text has ink: the title row in the right column.
    let info = m.info.unwrap();
    let (ix, iy, iw, _) = pixel_rect(&canvas, Rect::new(info.x, info.y, info.width, 1));
    let ink = (ix..ix + iw)
        .flat_map(|px| (iy..iy + ch as u32).map(move |py| (px, py)))
        .filter(|&(px, py)| *image.get_pixel(px, py) != bg())
        .count();
    assert!(ink > 100, "{ink}");
    assert!(canvas.text().contains("Morning Light"));
}

#[test]
fn leaving_a_screen_removes_its_pictures() {
    let mut canvas = Canvas::new(80, 24, font_px(1.0));
    let mut state = sized(&mut canvas, 1280, 800);
    let art = Art {
        id: 1,
        image: Arc::new(sample_cover(128)),
    };
    let spectrum = spectrum(96_000, 3_000.0);
    draw(&mut canvas, &state, &spectrum, Some(&art));
    let Layout::Main(m) = layout::compute(&state) else {
        panic!()
    };
    let (x, y, w, h) = pixel_rect(&canvas, m.cover.unwrap());
    let probe = (x + w / 2, y + h / 2);
    assert_ne!(*canvas.image().get_pixel(probe.0, probe.1), bg());
    // Settings has no cover there: the old pixels are gone.
    state.screen = Screen::Settings;
    draw(&mut canvas, &state, &spectrum, Some(&art));
    let after = *canvas.image().get_pixel(probe.0, probe.1);
    assert!(
        after == bg() || after.0.iter().all(|&c| c < 90),
        "{after:?}"
    );
    // Now playing paints the cover bigger.
    state.screen = Screen::NowPlaying;
    draw(&mut canvas, &state, &spectrum, Some(&art));
    let Layout::NowPlaying(n) = layout::compute(&state) else {
        panic!()
    };
    let big = n.cover;
    assert!(big.height > m.cover.unwrap().height);
    let (x, y, w, h) = pixel_rect(&canvas, big);
    assert_ne!(*canvas.image().get_pixel(x + w / 2, y + h / 2), bg());
}

#[test]
fn now_playing_is_the_cover_with_the_caption_over_it() {
    let mut canvas = Canvas::new(80, 24, font_px(1.0));
    let mut state = sized(&mut canvas, 1280, 800);
    state.screen = Screen::NowPlaying;
    // A light cover, so the text needs the darkening behind it.
    let light = image::RgbImage::from_pixel(64, 64, Rgb([235, 225, 210]));
    let art = Art {
        id: 3,
        image: Arc::new(light),
    };
    let spectrum = spectrum(176_400, 1_000.0);
    draw(&mut canvas, &state, &spectrum, Some(&art));
    let Layout::NowPlaying(n) = layout::compute(&state) else {
        panic!()
    };
    let image = canvas.image();
    let (x, y, w, h) = pixel_rect(&canvas, n.cover);
    // The cover fills the screen's height at full resolution, centred.
    assert_eq!(h as usize, usize::from(state.size.1) * canvas.cell().1);
    assert_eq!(image.get_pixel(x + w / 2, y + h / 4).0, [235, 225, 210]);
    assert_eq!(*image.get_pixel(x / 2, y + h / 2), bg());
    // The caption: title, artist and album, the status light; nothing else.
    let text = canvas.text();
    for part in ["Morning Light", "The Ramps · Synthetic Sessions", "○ PCM"] {
        assert!(text.contains(part), "missing {part:?} in\n{text}");
    }
    for gone in ["file 44.1", "settings", "20 Hz", "esc", "4:05", "FLAC"] {
        assert!(!text.contains(gone), "{gone:?} on Now playing");
    }
    // Under the text the cover is darkened, and the text is drawn on it.
    let (cx, cy, cw, ch) = pixel_rect(&canvas, n.caption);
    assert!(cx > x && cy + ch <= y + h);
    let darkest = (cx..cx + cw)
        .map(|px| image.get_pixel(px, cy + ch - 2).0[0])
        .min()
        .unwrap();
    assert!(darkest < 120, "{darkest}");
    // Without a cover: a placeholder and the caption under it.
    draw(&mut canvas, &state, &spectrum, None);
    let text = canvas.text();
    assert!(text.contains('\u{266a}') && text.contains("Morning Light"));
}

/// Frames drawn one after another (pictures kept where nothing under them
/// changed, cells redrawn only where they did) end in the same pixels as
/// the last frame drawn afresh: through covers that stay, change and go,
/// a caption over one, a moving spectrum and the Transition tab.
#[test]
fn frames_in_a_row_end_as_the_last_drawn_afresh() {
    let first = Art {
        id: 1,
        image: Arc::new(sample_cover(128)),
    };
    let second = Art {
        id: 2,
        image: Arc::new(image::RgbImage::from_pixel(64, 64, Rgb([235, 225, 210]))),
    };
    let mut canvas = Canvas::new(80, 24, font_px(1.0));
    let main = sized(&mut canvas, 1280, 800);
    let screen = |s: Screen| {
        let mut state = main.clone();
        state.screen = s;
        state
    };
    let mut later = main.clone();
    later.playing.position += 7.0;
    let mut tab = common::transition_state(main.size.0, main.size.1);
    tab.drawn_heard = Some(tab.playing.position);
    let mut tab_later = tab.clone();
    tab_later.playing.position += 0.5;
    tab_later.drawn_heard = Some(tab_later.playing.position);
    let tones = [spectrum(96_000, 1_000.0), spectrum(96_000, 5_000.0)];
    let steps: Vec<(&str, State, Option<&Art>, usize)> = vec![
        ("main", main.clone(), Some(&first), 0),
        ("main, the spectrum moved", main.clone(), Some(&first), 1),
        ("main, later", later, Some(&first), 0),
        ("now playing", screen(Screen::NowPlaying), Some(&first), 1),
        (
            "now playing, again",
            screen(Screen::NowPlaying),
            Some(&first),
            0,
        ),
        (
            "now playing, another cover",
            screen(Screen::NowPlaying),
            Some(&second),
            0,
        ),
        ("now playing, no cover", screen(Screen::NowPlaying), None, 1),
        ("settings", screen(Screen::Settings), Some(&first), 0),
        ("main again", main.clone(), Some(&first), 1),
        ("transition", tab, Some(&first), 0),
        ("transition, later", tab_later, Some(&first), 1),
        ("main after transition", main.clone(), Some(&first), 0),
    ];
    for (name, state, art, tone) in &steps {
        draw(&mut canvas, state, &tones[*tone], *art);
        let mut fresh = Canvas::new(80, 24, font_px(1.0));
        fresh.fit(1280, 800);
        draw(&mut fresh, state, &tones[*tone], *art);
        let (got, want) = (canvas.image(), fresh.image());
        let differ = got
            .pixels()
            .zip(want.pixels())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(differ, 0, "{name}: {differ} pixels differ");
    }
}

/// The Transition tab's lanes stay in the pixmap from frame to frame, shifted
/// as the view scrolls and painted again only where their marks changed: a run
/// of frames (the view following the playhead by fractions of a pixel and by
/// several, back after a seek, held with the playhead going across it, the
/// cursor and a selection moving, a transition's curves shown, a zoom) gives
/// at each frame the pixels of that frame drawn afresh, at scale 2; a frame
/// where nothing moved changes no pixel.
#[test]
fn transition_frames_in_a_row_are_as_drawn_afresh() {
    let (width, height, scale) = (2400, 1500, 2.0);
    let mut canvas = Canvas::new(80, 24, font_px(scale));
    let grid = canvas.fit(width, height);
    let tone = spectrum(96_000, 1_000.0);
    let steps = transition_steps(grid);
    for (name, state) in &steps {
        draw_heard(&mut canvas, state, &tone, None);
        let mut fresh = Canvas::new(80, 24, font_px(scale));
        fresh.fit(width, height);
        draw_heard(&mut fresh, state, &tone, None);
        let (got, want) = (canvas.image(), fresh.image());
        let differ = got
            .pixels()
            .zip(want.pixels())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(differ, 0, "{name}: {differ} pixels differ");
    }
    let (name, last) = steps.last().unwrap();
    draw_heard(&mut canvas, last, &tone, None);
    assert!(!canvas.changed, "{name}, again: a pixel changed");
    // The words were there to be drawn: without them the frame differs.
    let (_, first) = &steps[0];
    let mut bare = first.clone();
    bare.karaoke.lyrics.clear();
    let picture = |state: &State| {
        let mut c = Canvas::new(80, 24, font_px(scale));
        c.fit(width, height);
        draw_heard(&mut c, state, &tone, None);
        c.image()
    };
    let differ = picture(first)
        .pixels()
        .zip(picture(&bare).pixels())
        .filter(|(a, b)| a != b)
        .count();
    assert!(differ > 500, "the words: {differ} pixels");
}

/// A frame of `state` at what it says is heard.
fn draw_heard(canvas: &mut Canvas, state: &State, tone: &SpectrumFrame, art: Option<&Art>) {
    let scene = Scene {
        state,
        spectrum: tone,
        pointer: None,
        heard: state.heard(),
    };
    canvas.draw(&scene, art);
}

/// The window's frames through the GPU, off screen: the pixels that
/// changed go over frame by frame (as [`Canvas::take_changes`] says), and
/// each frame read back is the canvas, with the background beyond it,
/// through the Transition tab's steps, the browser with a cover and a moving
/// spectrum, Now playing, a window resized, and a skipped take; into a
/// frame that takes the bytes as they are and into one that encodes them
/// as sRGB. Skipped (with a note) without a GPU.
#[test]
fn the_gpu_window_shows_each_frame_as_the_canvas() {
    for srgb in [false, true] {
        gpu_frames(srgb);
    }
}

fn gpu_frames(srgb: bool) {
    use jodeldiplom::gui::present::Offscreen;
    let mut gpu = match Offscreen::new(srgb) {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("no GPU ({e}): skipped");
            return;
        }
    };
    gpu.name += if srgb { ", sRGB" } else { "" };
    let scale = 2.0;
    let art = Art {
        id: 3,
        image: Arc::new(sample_cover(128)),
    };
    let tones = [spectrum(96_000, 1_000.0), spectrum(96_000, 5_000.0)];
    let mut canvas = Canvas::new(80, 24, font_px(scale));
    // Not a whole number of cells: the background shows beyond them.
    let (mut width, mut height) = (2411, 1507);
    let grid = canvas.fit(width, height);
    let main = {
        let mut s = common::playing_state(grid.0, grid.1);
        s.size = grid;
        s
    };
    let mut steps: Vec<(String, State, usize)> = vec![
        ("main".into(), main.clone(), 0),
        ("main, the spectrum moved".into(), main.clone(), 1),
        ("main, still".into(), main.clone(), 1),
    ];
    let mut now_playing = main.clone();
    now_playing.screen = Screen::NowPlaying;
    steps.push(("now playing".into(), now_playing, 0));
    steps.extend(transition_steps(grid).into_iter().map(|(n, s)| (n, s, 0)));
    steps.push(("main after transition".into(), main.clone(), 1));
    for (i, (name, state, tone)) in steps.iter().enumerate() {
        if i == 5 {
            // The window a little wider: the grid stays, the margin grows.
            width += 7;
        }
        draw_heard(&mut canvas, state, &tones[*tone], Some(&art));
        // One take skipped: what it held must come with the next.
        if i == 8 {
            continue;
        }
        let changes = canvas.take_changes();
        let got = gpu.show(&canvas, &changes, width, height, None).unwrap();
        let mut want = vec![0u32; (width * height) as usize];
        canvas.write_xrgb(&mut want, width as usize, height as usize);
        let differ = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(differ, 0, "{name} on {}: {differ} pixels differ", gpu.name);
    }
    // A smaller window: a smaller grid.
    (width, height) = (1900, 1200);
    let grid = canvas.fit(width, height);
    let mut small = main.clone();
    small.size = grid;
    draw_heard(&mut canvas, &small, &tones[0], Some(&art));
    let changes = canvas.take_changes();
    let got = gpu.show(&canvas, &changes, width, height, None).unwrap();
    let mut want = vec![0u32; (width * height) as usize];
    canvas.write_xrgb(&mut want, width as usize, height as usize);
    assert!(got == want, "smaller on {}: the pixels differ", gpu.name);
}

/// The Karaoke tab's voice lane as the layer the window scrolls by itself:
/// each frame through the GPU is the frame the canvas paints with the lane
/// in its pixmap (the playhead apart, which the layer draws over the
/// words), as the song goes on, tiles coming in and the word heard
/// changing; also frames that move the layer alone, between frames of the
/// rest. Into a frame that takes the bytes as they are and one that
/// encodes them as sRGB. Skipped (with a note) without a GPU.
#[test]
fn the_karaoke_lane_scrolled_by_the_gpu_is_the_lane_painted() {
    for srgb in [false, true] {
        layer_frames(srgb);
    }
}

fn layer_frames(srgb: bool) {
    let Some(mut gpu) = layer_gpu(srgb) else {
        return;
    };
    let (mut flat, mut layered, grid) = layer_canvases();
    let mut state = common::transition_state(grid.0, grid.1);
    state.side = jodeldiplom::core::Side::Karaoke;
    state.size = grid;
    common::sung_every(&mut state, 0.5);
    // In C major: the scale's lines across the lane, the notes on them.
    for r in state.rhythm.values_mut() {
        if let jodeldiplom::core::Rhythm::Ready(x) = r {
            let mut keyed = (**x).clone();
            let mut a = (*keyed.analysis).clone();
            a.key = Some(jodel_rhythm::key::Key::on(0, false));
            keyed.analysis = std::sync::Arc::new(a);
            *r = jodeldiplom::core::Rhythm::Ready(std::sync::Arc::new(keyed));
        }
    }
    for (i, &time) in [30.0, 30.4, 31.7, 36.0, 52.3, 52.31, 75.0]
        .iter()
        .enumerate()
    {
        same_through_layer(
            &mut gpu,
            &mut flat,
            &mut layered,
            &mut state,
            time,
            &format!("frame {i}"),
        );
    }
}

/// The Transition tab's playing track following the playhead as the layer the
/// window scrolls by itself (its bar numbers and lanes, the words on the
/// voice's): each frame through the GPU is the frame the canvas paints with
/// them in its pixmap (the playhead apart), across bars (the cursor going
/// along), with a cue set and a transition's curves shown while it
/// follows. Into both kinds of frame; skipped without a GPU.
#[test]
fn the_transition_lanes_scrolled_by_the_gpu_are_the_lanes_painted() {
    use jodeldiplom::core::keymap::Op;
    use jodeldiplom::core::{update, Action};
    for srgb in [false, true] {
        let Some(mut gpu) = layer_gpu(srgb) else {
            return;
        };
        let (mut flat, mut layered, grid) = layer_canvases();
        let mut state = common::transition_state(grid.0, grid.1);
        state.size = grid;
        sung(&mut state);
        let mut step = |state: &mut State, time: f64, name: &str| {
            same_through_layer(&mut gpu, &mut flat, &mut layered, state, time, name);
            assert!(layered.has_layer(), "{name}: a layer");
        };
        for (i, &time) in [30.0, 30.2, 31.1, 33.9, 34.0, 41.5].iter().enumerate() {
            step(&mut state, time, &format!("following, {i}"));
        }
        update(&mut state, Action::Op(Op::Accept));
        step(&mut state, 42.0, "a cue set");
        update(&mut state, Action::Op(Op::TransitionPreset));
        step(&mut state, 42.3, "a transition's curves");
        step(&mut state, 47.0, "later");
    }
}

fn layer_gpu(srgb: bool) -> Option<jodeldiplom::gui::present::Offscreen> {
    match jodeldiplom::gui::present::Offscreen::new(srgb) {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("no GPU ({e}): skipped");
            None
        }
    }
}

const LAYER_SIZE: (u32, u32) = (2411, 1507);

/// Two canvases for one window: the lanes in the pixmap, and in a layer.
fn layer_canvases() -> (Canvas, Canvas, (u16, u16)) {
    let mut flat = Canvas::new(80, 24, font_px(2.0));
    let mut layered = Canvas::new(80, 24, font_px(2.0));
    layered.set_gpu(true);
    let grid = flat.fit(LAYER_SIZE.0, LAYER_SIZE.1);
    layered.fit(LAYER_SIZE.0, LAYER_SIZE.1);
    (flat, layered, grid)
}

/// `state` near `time` drawn both ways, the layer shown through the GPU:
/// the same pixels (but the playhead's, which the layer draws over the
/// words). The layer is moved alone first (frames between frames of the
/// rest), then to where it stands on a whole column (as the pixmap's lanes
/// do); every frame goes over, as the window's do.
fn same_through_layer(
    gpu: &mut jodeldiplom::gui::present::Offscreen,
    flat: &mut Canvas,
    layered: &mut Canvas,
    state: &mut State,
    time: f64,
    name: &str,
) {
    let (width, height) = LAYER_SIZE;
    let tone = spectrum(96_000, 1_000.0);
    let at = |state: &mut State, seconds: f64| {
        state.playing.position = seconds;
        state.drawn_heard = Some(seconds);
    };
    let mut heard = time;
    at(state, heard);
    draw_heard(layered, state, &tone, None);
    for _ in 0..3 {
        let layer = layered.layer_frame(heard).expect("a layer");
        let step = f64::from(layer.frac) * layer.px;
        let none = jodeldiplom::gui::canvas::Changes::default();
        gpu.show(layered, &none, width, height, Some(layer))
            .unwrap();
        heard -= step;
    }
    at(state, heard);
    draw_heard(flat, state, &tone, None);
    draw_heard(layered, state, &tone, None);
    let layer = layered.layer_frame(heard).expect("a layer");
    let (x0, y0, w, h) = layer.rect;
    let top = y0 + layer.head_top;
    let head = layer.playhead.map(|(x, w)| (x.floor() as usize, w));
    let changes = layered.take_changes();
    let got = gpu
        .show(layered, &changes, width, height, Some(layer))
        .unwrap();
    let mut want = vec![0u32; (width * height) as usize];
    flat.write_xrgb(&mut want, width as usize, height as usize);
    let mut differ = 0;
    for (n, (a, b)) in got.iter().zip(&want).enumerate() {
        let (x, y) = (n % width as usize, n / width as usize);
        let under_head = x >= x0 && x < x0 + w && y >= top && y < y0 + h;
        let on_head = head.is_some_and(|(hx, hw)| {
            under_head && x >= (x0 + hx).saturating_sub(1) && x <= x0 + hx + hw
        });
        if a != b && !on_head {
            if differ < 5 {
                eprintln!("{name}: ({x}, {y}) {a:06x} against {b:06x}; the layer at ({x0}, {y0}) {w}x{h}, rows from {top}, head {head:?}");
            }
            differ += 1;
        }
    }
    assert_eq!(
        differ, 0,
        "{name} at {heard:.3} s on {}: {differ} pixels differ",
        gpu.name
    );
}
/// The Transition tab for a run of frames: the view following the playhead by
/// fractions of a pixel and by several, back after a seek, held with the
/// playhead going across it, the cursor and a selection moving, a
/// transition's curves shown, a zoom.
fn transition_steps(grid: (u16, u16)) -> Vec<(String, State)> {
    use jodeldiplom::core::keymap::Op;
    use jodeldiplom::core::{update, Action};
    let mut state = common::transition_state(grid.0, grid.1);
    sung(&mut state);
    let at = |state: &mut State, seconds: f64| {
        state.playing.position = seconds;
        state.drawn_heard = Some(seconds);
    };
    let op = |state: &mut State, op: Op| {
        update(state, Action::Op(op));
    };
    let mut steps: Vec<(String, State)> = Vec::new();
    for i in 0..10 {
        at(&mut state, 30.0 + 0.003 * f64::from(i));
        steps.push((format!("following, {i} small steps"), state.clone()));
    }
    for i in 0..6 {
        at(&mut state, 30.1 + 0.37 * f64::from(i));
        steps.push((format!("following, {i} larger steps"), state.clone()));
    }
    at(&mut state, 12.0);
    steps.push(("back after a seek".into(), state.clone()));
    op(&mut state, Op::TransitionBarForward);
    steps.push(("the cursor moved: the view held".into(), state.clone()));
    for i in 0..6 {
        at(&mut state, 12.0 + 0.11 * f64::from(i));
        steps.push((format!("held, the playhead on {i}"), state.clone()));
    }
    op(&mut state, Op::TransitionSelectForward);
    op(&mut state, Op::TransitionSelectForward);
    steps.push(("a selection".into(), state.clone()));
    op(&mut state, Op::TransitionPreset);
    steps.push(("a transition's curves".into(), state.clone()));
    op(&mut state, Op::TransitionFollow);
    at(&mut state, 12.9);
    steps.push(("following again".into(), state.clone()));
    op(&mut state, Op::TransitionZoomOut);
    steps.push(("zoomed out".into(), state.clone()));
    at(&mut state, 13.4);
    steps.push(("zoomed out, later".into(), state.clone()));
    steps
}

/// Words timed on the playing and the next track, every 0.45 s from 5 s to
/// 60 s, every other one with its letters' times: on the voice's lane in
/// every Transition frame, the one heard changing as the playhead goes.
fn sung(state: &mut State) {
    use jodel_lyrics::{Line, Lyrics, Source, Timing, Word};
    use jodeldiplom::core::karaoke::SongLyrics;
    let texts = ["Mister", "rock", "and", "roll,", "take", "me", "home"];
    let words: Vec<Word> = (0..122)
        .map(|k| {
            let start = 5.0 + 0.45 * f64::from(k);
            let text = texts[k as usize % texts.len()];
            let n = text.chars().count();
            Word {
                start: Some(start),
                end: Some(start + 0.3),
                letters: if k % 2 == 0 {
                    (0..n)
                        .map(|i| (0.3 * (i * i) as f32) / (n * n) as f32)
                        .collect()
                } else {
                    Vec::new()
                },
                ..Word::new(text)
            }
        })
        .collect();
    let lyrics = Arc::new(Lyrics {
        lines: words
            .chunks(7)
            .map(|w| Line {
                words: w.to_vec(),
                ..Line::default()
            })
            .collect(),
        source: Source::Unknown,
        timing: Timing {
            detail: jodel_lyrics::Detail::Words,
            by: Some(jodel_lyrics::Maker::new("ctc:test", 2)),
            fit: None,
        },
    });
    for index in [1, 2] {
        let path = state.queue.paths[index].clone();
        state.karaoke.lyrics.insert(
            path,
            SongLyrics::Found {
                lyrics: Arc::clone(&lyrics),
                shown: Arc::clone(&lyrics),
            },
        );
    }
}

#[test]
fn hidpi_doubles_the_cells() {
    let small = Canvas::new(10, 10, font_px(1.0));
    let large = Canvas::new(10, 10, font_px(2.0));
    let (a, b) = (small.cell(), large.cell());
    assert!(b.0 >= 2 * a.0 - 1 && b.0 <= 2 * a.0 + 1, "{a:?} {b:?}");
    assert!(b.1 >= 2 * a.1 - 1 && b.1 <= 2 * a.1 + 1, "{a:?} {b:?}");
    // The same window at twice the scale has about the same grid.
    let mut c1 = Canvas::new(10, 10, font_px(1.0));
    let mut c2 = Canvas::new(10, 10, font_px(2.0));
    let g1 = c1.fit(1280, 800);
    let g2 = c2.fit(2560, 1600);
    assert!(
        g1.0.abs_diff(g2.0) <= 2 && g1.1.abs_diff(g2.1) <= 2,
        "{g1:?} {g2:?}"
    );
    // Without a cover the placeholder is drawn in cells.
    let mut state = common::playing_state(g1.0, g1.1);
    state.size = g1;
    let empty = SpectrumFrame::default();
    draw(&mut c1, &state, &empty, None);
    assert!(c1.text().contains('♪'));
}

/// The icon file built into the window player is the one the code draws
/// (rewrite it with `cargo run -p jodeldiplom --example icon`), and is a
/// well-formed icon with every size.
#[test]
fn the_icon_file_matches_the_drawing() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/jodeldiplom.ico");
    let file = std::fs::read(&path).unwrap();
    assert!(
        file == jodeldiplom::icon::ico(),
        "assets/jodeldiplom.ico is stale: run cargo run -p jodeldiplom --example icon"
    );
    assert_eq!(&file[..4], &[0, 0, 1, 0]);
    let count = u16::from_le_bytes([file[4], file[5]]) as usize;
    assert_eq!(count, jodeldiplom::icon::ICO_SIZES.len());
    for (i, size) in jodeldiplom::icon::ICO_SIZES.iter().enumerate() {
        let entry = &file[6 + 16 * i..6 + 16 * (i + 1)];
        let side = if *size >= 256 { 0 } else { *size as u8 };
        assert_eq!((entry[0], entry[1]), (side, side));
        let bytes = u32::from_le_bytes(entry[8..12].try_into().unwrap()) as usize;
        let offset = u32::from_le_bytes(entry[12..16].try_into().unwrap()) as usize;
        assert!(offset + bytes <= file.len());
    }
    // Transparent corners, an opaque middle.
    let rgba = jodeldiplom::icon::rgba(Style::Windows, 32);
    assert_eq!(rgba[3], 0);
    assert_eq!(rgba[((16 * 32 + 16) * 4 + 3) as usize], 255);
}

/// The macOS icon file is the one the code draws, a well-formed `icns`
/// whose entries are PNGs of their sizes; the macOS style leaves Apple's
/// margin around the square (with a faint shadow under it).
#[test]
fn the_macos_icon_file_matches_the_drawing() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/jodeldiplom.icns");
    let file = std::fs::read(&path).unwrap();
    assert!(
        file == jodeldiplom::icon::icns(),
        "assets/jodeldiplom.icns is stale: run cargo run -p jodeldiplom --example icon"
    );
    assert_eq!(&file[..4], b"icns");
    assert_eq!(
        u32::from_be_bytes(file[4..8].try_into().unwrap()) as usize,
        file.len()
    );
    let mut at = 8;
    for (kind, size) in jodeldiplom::icon::ICNS_ENTRIES {
        assert_eq!(&file[at..at + 4], kind);
        let len = u32::from_be_bytes(file[at + 4..at + 8].try_into().unwrap()) as usize;
        let png = image::load_from_memory(&file[at + 8..at + len]).unwrap();
        assert_eq!((png.width(), png.height()), (size, size), "{kind:?}");
        at += len;
    }
    assert_eq!(at, file.len());
    let size = 128;
    let rgba = jodeldiplom::icon::rgba(Style::MacOs, size);
    let alpha = |x: u32, y: u32| rgba[((y * size + x) * 4 + 3) as usize];
    // The margin is clear at the top and the sides, the middle opaque.
    assert_eq!(alpha(size / 2, 2), 0);
    assert_eq!(alpha(2, size / 2), 0);
    assert_eq!(alpha(size / 2, size / 2), 255);
    // A little shadow below the square.
    let below = alpha(size / 2, size - 11);
    assert!(below > 0 && below < 128, "{below}");
}

#[test]
fn the_transition_lanes_are_painted_in_pixels() {
    let mut canvas = Canvas::new(80, 24, font_px(1.0));
    let grid = canvas.fit(1600, 1000);
    let mut state = common::transition_state(grid.0, grid.1);
    state.size = grid;
    let spectrum = spectrum(176_400, 1_000.0);
    draw(&mut canvas, &state, &spectrum, None);
    let image = canvas.image();
    let Layout::Main(m) = layout::compute(&state) else {
        panic!("the main screen");
    };
    let tab = m.transition.expect("the Transition tab");
    assert_eq!(tab.blocks.len(), 2);
    assert!(m.bottom.spectrum.is_none());
    // The vocals of the playing track come in at 60 s: before, nothing is
    // painted in their lane but its background; the drums lane is busy.
    let lane_pixels = |i: usize| {
        let (_, _, area) = tab.blocks[0].lanes[i];
        let (x, y, w, h) = pixel_rect(&canvas, area);
        let mut lit = 0;
        for yy in y..y + h {
            for xx in x + 8..x + w {
                if *image.get_pixel(xx, yy) != bg() {
                    lit += 1;
                }
            }
        }
        lit
    };
    assert!(lane_pixels(1) > 1000, "drums {}", lane_pixels(1));
    // The vocals lane, from 0 s with the playhead at 0: bar lines only.
    assert!(
        lane_pixels(4) < lane_pixels(1) / 4,
        "vocals {}",
        lane_pixels(4)
    );
}

#[test]
fn the_bar_numbers_start_at_their_bars_to_the_pixel() {
    use jodeldiplom::core::lanes::{self, Block};
    use jodeldiplom::core::transition::viewport;
    let mut canvas = Canvas::new(80, 24, font_px(1.0));
    let grid = canvas.fit(1600, 1000);
    let mut state = common::transition_state(grid.0, grid.1);
    state.size = grid;
    // A view that starts between cells.
    state.transition.now.path = lanes::path(&state, Block::Now).cloned();
    state.transition.now.start = Some(10.123);
    let spectrum = spectrum(176_400, 1_000.0);
    draw(&mut canvas, &state, &spectrum, None);
    let image = canvas.image();
    let Layout::Main(m) = layout::compute(&state) else {
        panic!("the main screen");
    };
    let ruler = m.transition.unwrap().blocks[0].ruler;
    let (x0, y0, w, h) = pixel_rect(&canvas, ruler);
    let (start, end) = viewport::window(&state, Block::Now, 0.0);
    let a = lanes::rhythmed(&state, Block::Now)
        .unwrap()
        .analysis
        .clone();
    let cw = canvas.cell().0 as u32;
    let ink = |x: u32| (y0..y0 + h).any(|y| *image.get_pixel(x, y) != bg());
    let mut checked = 0;
    for &d in &a.downbeats {
        let t = a.beats[d];
        if t <= start || t >= end {
            continue;
        }
        let x = x0 + ((t - start) / (end - start) * f64::from(w)).floor() as u32;
        if x + 2 * cw >= x0 + w {
            continue;
        }
        // Ink within its first cell, none in the few pixels before it.
        assert!((x..x + cw).any(ink), "bar at {t} s (x {x})");
        assert!(!(x - 3..x - 1).any(ink), "bar at {t} s (x {x}) starts late");
        checked += 1;
    }
    assert!(checked >= 8, "{checked}");
}

#[test]
fn after_a_track_change_the_next_block_is_blank_until_its_analysis_is_in() {
    use jodeldiplom::core::{update, Action};
    let mut canvas = Canvas::new(80, 24, font_px(1.0));
    let grid = canvas.fit(1600, 1000);
    let mut state = common::transition_state(grid.0, grid.1);
    state.size = grid;
    let spectrum = spectrum(176_400, 1_000.0);
    draw(&mut canvas, &state, &spectrum, None);
    // The next track starts; the one after it is not analysed yet.
    let track = state.playing.track.clone().unwrap();
    update(
        &mut state,
        Action::Engine(jodel_playback::Event::TrackStarted(Box::new(
            jodel_playback::TrackStart {
                index: 2,
                head: 0,
                info: track.info.clone(),
                route: track.route.clone(),
                output: track.output,
                device: "DAC".into(),
                bit_perfect: false,
                output_frame: 0,
                from_frame: 0,
            },
        ))),
    );
    state.drawn_heard = Some(0.0);
    draw(&mut canvas, &state, &spectrum, None);
    let image = canvas.image();
    let Layout::Main(m) = layout::compute(&state) else {
        panic!("the main screen");
    };
    let tab = m.transition.unwrap();
    assert_eq!(tab.blocks.len(), 2);
    for (_, _, area) in &tab.blocks[1].lanes {
        let (x, y, w, h) = pixel_rect(&canvas, *area);
        let lit = (y..y + h)
            .flat_map(|yy| (x..x + w).map(move |xx| (xx, yy)))
            .filter(|&(xx, yy)| *image.get_pixel(xx, yy) != bg())
            .count();
        assert_eq!(lit, 0, "the next block's lanes show nothing yet");
    }
}

/// The window's keys on the help: `?`, `/`, then "volume" typed (a capital
/// too), as iced reports them. The window shows only the device volume's
/// keys under the search line; esc clears the search, a second leaves.
#[test]
fn the_window_searches_the_help() {
    let mut canvas = Canvas::new(80, 24, font_px(1.0));
    let mut state = sized(&mut canvas, 1280, 800);
    let press = |state: &mut State, key: IcedKey| {
        let key = gui::key(&key, Modifiers::empty()).unwrap();
        if let Some(action) = jodeldiplom::core::keymap::action(state, key) {
            update(state, action);
        }
    };
    let ch = |c: &str| IcedKey::Character(c.into());
    press(&mut state, ch("?"));
    press(&mut state, ch("/"));
    for c in ["V", "o", "l", "u", "m", "e"] {
        press(&mut state, ch(c));
    }
    assert_eq!(state.screen, Screen::Help);
    assert!(!state.quit);
    let spectrum = spectrum(176_400, 1_000.0);
    draw(&mut canvas, &state, &spectrum, None);
    let text = canvas.text();
    for part in [
        "/ Volume",
        "alt+↑",
        "device volume a step up",
        "alt+↓",
        "device volume a step down",
    ] {
        assert!(text.contains(part), "missing {part:?} in\n{text}");
    }
    for gone in ["play / pause", "Browser and queue"] {
        assert!(!text.contains(gone), "{gone:?} in\n{text}");
    }
    press(&mut state, IcedKey::Named(Named::Escape));
    draw(&mut canvas, &state, &spectrum, None);
    assert!(canvas.text().contains("play / pause"));
    assert_eq!(state.screen, Screen::Help);
    press(&mut state, IcedKey::Named(Named::Escape));
    assert_eq!(state.screen, Screen::Main);
}

/// Karaoke mode's stage on the GPU: a word ahead of the playhead shows as
/// its pill, one sung and hit as gold, the singer's arrow at the playhead,
/// over the pixmap as it is elsewhere. Skipped (with a note) without a GPU.
#[test]
fn the_stage_draws_the_lane_over_the_pixmap() {
    use jodeldiplom::core::singing::Singing;
    use jodeldiplom::gui::present::Offscreen;
    use jodeldiplom::gui::stage::{Lane, Song, Stage};
    use jodeldiplom::view::karaoke::KaraokePicture;
    let mut gpu = match Offscreen::new(false) {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("no GPU ({e}): skipped");
            return;
        }
    };
    // "la la", a second each, on C4 then G4.
    let word = |text: &str, s: f64| jodel_lyrics::Word {
        start: Some(s),
        end: Some(s + 1.0),
        confidence: 0.9,
        ..jodel_lyrics::Word::new(text)
    };
    let line = jodel_lyrics::Line {
        words: vec![word("la", 0.0), word("lo", 1.0)],
        ..jodel_lyrics::Line::default()
    };
    let lyrics = Arc::new(jodel_lyrics::Lyrics {
        lines: vec![line],
        source: jodel_lyrics::Source::Unknown,
        timing: jodel_lyrics::Timing::default(),
    });
    let mut melody = jodel_rhythm::waveform::Melody {
        cents: vec![0; 300],
        sure: vec![0; 300],
    };
    for i in 0..200 {
        melody.cents[i] = if i < 100 { 6000 } else { 6700 };
        melody.sure[i] = 255;
    }
    let mut singing = Singing::default();
    singing.prepare(
        std::path::Path::new("a"),
        &lyrics,
        &jodeldiplom::core::singing::melodic(&melody),
    );
    // Sung on the note for its first 0.06 s (too little to count: "la"
    // keeps its liquid there), then heard up to 0.5 s.
    for i in 0..=50 {
        singing.hear(f64::from(i) / 100.0, (i <= 5).then_some(6000.0));
    }
    let (w, h) = (800u32, 450u32);
    let song = Song::new(&singing, lyrics.lines.iter().collect(), Some(3.0));
    let picture = KaraokePicture {
        area: Rect::new(0, 0, 80, 24),
        bar: jodeldiplom::core::karaoke::controls::bar(Rect::new(0, 0, 80, 24)),
        lines: Vec::new(),
        count: None,
        cue: None,
        controls: None,
        start: None,
        before: Vec::new(),
        moved: 1.0,
        score: None,
        result: None,
    };
    let now = 0.55;
    let area = (0.0, 0.0, w as f32, h as f32);
    let mut stage = Stage::default();
    let frame = stage.frame(&song, &picture, area, now, 1.0 / 60.0);
    gpu.set_stage(Some(&frame), w, h);
    let shown = gpu
        .show_pixels(&vec![0u32; (w * h) as usize], w, h)
        .expect("a frame");
    let at = |x: f32, y: f32| shown[y as usize * w as usize + x as usize];
    let rgb = |c: u32| ((c >> 16) & 255, (c >> 8) & 255, c & 255);
    let lane = Lane::new(area, now);
    let height = |cents: f64| singing.notes.lines[0].height(cents);
    // "lo" ahead: its pill's glass, grey.
    let (r, g, b) = rgb(at(lane.x(1.5), lane.y(height(6700.0)) + 2.0));
    assert!(
        r > 15 && (r as i32 - b as i32).abs() < 12,
        "the pill ahead: {r} {g} {b}"
    );
    // "la" sung on its note: gold behind the playhead.
    let (r, g, b) = rgb(at(lane.x(0.03), lane.y(height(6000.0)) + 4.0));
    assert!(r > 180 && g > 100 && b < 120, "the hit: {r} {g} {b}");
    // Nothing far above the lane.
    assert_eq!(at(w as f32 / 2.0, 2.0), 0);
}

/// The Mixer tab's rows are left to the window's own widgets
/// ([`jodeldiplom::gui::mixer`]): the canvas paints its box and its keys,
/// and nothing inside.
#[test]
fn the_canvas_leaves_the_mixer_tab_to_the_window() {
    let mut canvas = Canvas::new(80, 24, font_px(1.0));
    let grid = canvas.fit(1280, 800);
    let mut state = common::desk_state(grid.0, grid.1);
    state.size = grid;
    draw(&mut canvas, &state, &spectrum(176_400, 1_000.0), None);
    let Layout::Main(m) = layout::compute(&state) else {
        panic!("the main screen");
    };
    assert!(m.desk.is_some(), "the desk is laid out");
    let image = canvas.image();
    let (x, y, w, h) = pixel_rect(&canvas, m.rows);
    let lit = (y..y + h)
        .flat_map(|yy| (x..x + w).map(move |xx| (xx, yy)))
        .filter(|&(xx, yy)| *image.get_pixel(xx, yy) != bg())
        .count();
    assert_eq!(lit, 0, "the desk's rows are the window's");
    assert!(
        canvas.text().contains("m mute"),
        "the keys in the box's border"
    );
}
