//! The window's drawing path without a window: the shared widgets
//! rasterised into pixels, the pictures painted at their cell rects, the
//! covers drawn where the canvas leaves their cells.

mod common;
#[path = "../examples/headless/mod.rs"]
mod headless;

use std::sync::Arc;
use std::time::Instant;

use headless::Headless;
use iced::advanced::image::{Handle, Id as ImageId};
use iced::keyboard::{key::Named, Key as IcedKey, Modifiers};
use image::{Rgb, RgbaImage};
use jodeldiplom::art::{sample_cover, Art};
use jodeldiplom::core::{update, Screen, State};
use jodeldiplom::gui::lanes::{Lanes, Offscreen as LaneOffscreen};
use jodeldiplom::gui::pixels::PixelRect;
use jodeldiplom::gui::widgets::cover::{Align, Caption, Covers, Spot};
use jodeldiplom::gui::{self, font_px, Canvas};
use jodeldiplom::icon::Style;
use jodeldiplom::view::layout::{self, Grid, Layout};
use jodeldiplom::view::{palette, Scene};
use ratatui::layout::Rect;

fn bg() -> Rgb<u8> {
    Rgb(palette::BG)
}

fn draw(canvas: &mut Canvas, state: &State, art: Option<&Art>) {
    let scene = Scene {
        state,
        grid: canvas.grid(),
        pointer: None,
        heard: 0.0,
    };
    canvas.draw(&scene, art);
}

fn sized(canvas: &mut Canvas, w: u32, h: u32) -> State {
    let grid = canvas.fit(w, h);
    common::playing_state(grid.cols, grid.rows)
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

/// The cells are rasterised to the grid's size, text with ink; the
/// bottom bar's cells are left to the window (background alone), as are
/// the cover's.
#[test]
fn the_cells_are_painted_and_the_bar_left_to_the_window() {
    let mut canvas = Canvas::new(80, 24, font_px(1.0));
    let state = sized(&mut canvas, 1280, 800);
    draw(&mut canvas, &state, None);
    let image = canvas.image();
    let (cw, ch) = canvas.cell();
    assert_eq!(image.width() as usize, usize::from(canvas.grid().cols) * cw);
    assert_eq!(
        image.height() as usize,
        usize::from(canvas.grid().rows) * ch
    );
    let Layout::Main(m) = layout::compute(&state, canvas.grid()) else {
        panic!("not the main screen")
    };

    // The bottom bar's cells: background alone.
    let (bx, by, bw, bh) = pixel_rect(&canvas, m.bottom.area);
    let lit = (by..by + bh)
        .flat_map(|py| (bx..bx + bw).map(move |px| (px, py)))
        .filter(|&(px, py)| *image.get_pixel(px, py) != bg())
        .count();
    assert_eq!(lit, 0, "the bar's cells are blank");

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

/// `words` (`0x00RRGGBB`, `width` × `height`) as an image.
fn rgb_image(words: &[u32], width: u32, height: u32) -> image::RgbImage {
    let bytes = words
        .iter()
        .flat_map(|&w| jodeldiplom::gui::pixels::unpack(w))
        .collect();
    image::RgbImage::from_raw(width, height, bytes).expect("the picture's size")
}

/// `rect` in pixels as the covers' places have it.
fn place(canvas: &Canvas, rect: Rect) -> PixelRect {
    let (x, y, w, h) = pixel_rect(canvas, rect);
    (x as usize, y as usize, w as usize, h as usize)
}

/// `covers` drawn by iced's headless renderer in a window of `size`
/// pixels at `scale`, with nothing under them.
fn drawn(headless: &mut Headless, covers: &Covers, size: (u32, u32), scale: f32) -> RgbaImage {
    let window = iced::Size::new(size.0 as f32 / scale, size.1 as f32 / scale);
    headless.draw(covers.view(), window, Instant::now());
    headless.screenshot(scale)
}

/// Whether two colours are within `by` of each other in every channel.
fn near(a: &[u8], b: &[u8], by: u8) -> bool {
    a.iter().zip(b).all(|(p, q)| p.abs_diff(*q) <= by)
}

/// The canvas leaves a cover's cells blank and says where it goes, for
/// the window to draw it over them: on the main screen in its square from
/// under the list pane's top line (half a cell down) at the left; on Now
/// playing as large as the screen allows, in the middle, the caption over
/// it. Without art, or on a screen without one, there is none.
#[test]
fn the_covers_go_where_the_canvas_leaves_their_cells() {
    let mut canvas = Canvas::new(80, 24, font_px(1.0));
    let mut state = sized(&mut canvas, 1280, 800);
    let art = Art {
        id: 1,
        image: Arc::new(sample_cover(128)),
    };
    draw(&mut canvas, &state, Some(&art));
    let Layout::Main(m) = layout::compute(&state, canvas.grid()) else {
        panic!()
    };
    let ch = canvas.cell().1;
    let (x, y, w, h) = place(&canvas, m.cover.unwrap());
    let [spot] = canvas.covers() else {
        panic!("{:?}", canvas.covers())
    };
    assert_eq!(spot.place, (x, y + ch / 2, w, h - ch / 2));
    assert_eq!(spot.align, Align::TopLeft);
    assert!(spot.caption.is_none());
    let blank = canvas.image();
    assert_eq!(
        *blank.get_pixel((x + w / 2) as u32, (y + h / 2) as u32),
        bg()
    );

    state.screen = Screen::Settings;
    draw(&mut canvas, &state, Some(&art));
    assert!(canvas.covers().is_empty());

    state.screen = Screen::NowPlaying;
    draw(&mut canvas, &state, Some(&art));
    let Layout::NowPlaying(n) = layout::compute(&state, canvas.grid()) else {
        panic!()
    };
    assert!(n.cover.height > m.cover.unwrap().height);
    let [spot] = canvas.covers() else {
        panic!("{:?}", canvas.covers())
    };
    assert_eq!(spot.place, place(&canvas, n.cover));
    assert_eq!(spot.align, Align::Center);
    let caption = spot.caption.as_ref().expect("the caption over it");
    assert_eq!(caption.rect, place(&canvas, n.caption));
    assert_eq!(caption.lead, 4 * ch);

    draw(&mut canvas, &state, None);
    assert!(canvas.covers().is_empty());
}

/// A cover drawn where the canvas left it, at scale 1 and 2, in the first
/// frame it shows (a large image too, which iced would otherwise send to
/// the GPU in the background): the art fitted into its place at the
/// pixels it shows at, its corners rounded.
#[test]
fn a_cover_is_drawn_fitted_with_its_corners_rounded() {
    let mut headless = Headless::new();
    for scale in [1.0, 2.0] {
        let size = ((1280.0 * scale) as u32, (800.0 * scale) as u32);
        let mut canvas = Canvas::new(80, 24, font_px(f64::from(scale)));
        let state = sized(&mut canvas, size.0, size.1);
        let art = Art {
            id: 7,
            image: Arc::new(sample_cover(256)),
        };
        draw(&mut canvas, &state, Some(&art));
        let mut covers = Covers::default();
        covers.show(Some(&art), canvas.covers(), scale);
        let shot = drawn(&mut headless, &covers, size, scale);
        let (x, y, w, h) = canvas.covers()[0].place;
        let side = w.min(h);
        let cover = sample_cover(side as u32);
        let at = |dx: usize, dy: usize| shot.get_pixel((x + dx) as u32, (y + dy) as u32).0;
        let want = |dx: usize, dy: usize| cover.get_pixel(dx as u32, dy as u32).0;
        let (mid, r) = (side / 2, (10.0 * scale) as usize);
        assert!(near(&at(mid, mid), &want(mid, mid), 24), "scale {scale}");
        // The corner is cut off; a radius in along the diagonal is cover.
        assert!(!near(&at(0, 0), &want(0, 0), 40), "scale {scale}");
        assert!(near(&at(r, r), &want(r, r), 24), "scale {scale}");
        assert!(!near(&at(side - 1, 0), &want(side - 1, 0), 40));
    }
}

/// Now playing: the cover in the middle of the screen (here a light one),
/// darkened towards its bottom under the caption, and the caption's text
/// drawn over it; the screen's text is the caption alone.
#[test]
fn now_playing_is_the_cover_with_the_caption_over_it() {
    let mut canvas = Canvas::new(80, 24, font_px(1.0));
    let mut state = sized(&mut canvas, 1280, 800);
    state.screen = Screen::NowPlaying;
    let light = image::RgbImage::from_pixel(64, 64, Rgb([235, 225, 210]));
    let art = Art {
        id: 3,
        image: Arc::new(light),
    };
    draw(&mut canvas, &state, Some(&art));
    // The place fills the screen's height.
    let [spot] = canvas.covers() else {
        panic!("{:?}", canvas.covers())
    };
    let (x, y, w, h) = spot.place;
    assert_eq!(h, usize::from(canvas.grid().rows) * canvas.cell().1);
    // The caption: title, artist and album, the status light; nothing else.
    let text = canvas.text();
    for part in ["Morning Light", "The Ramps · Synthetic Sessions", "○ PCM"] {
        assert!(text.contains(part), "missing {part:?} in\n{text}");
    }
    for gone in ["file 44.1", "settings", "20 Hz", "esc", "4:05", "FLAC"] {
        assert!(!text.contains(gone), "{gone:?} on Now playing");
    }

    let mut headless = Headless::new();
    let mut covers = Covers::default();
    covers.show(Some(&art), canvas.covers(), 1.0);
    let shot = drawn(&mut headless, &covers, (1280, 800), 1.0);
    let side = w.min(h);
    let (ix, iy) = (x + (w - side) / 2, y + (h - side) / 2);
    let at = |px: usize, py: usize| shot.get_pixel(px as u32, py as u32).0;
    // Its top is the cover as it is; beside it the ground.
    assert!(near(
        &at(ix + side / 2, iy + side / 4),
        &[235, 225, 210, 255],
        2
    ));
    assert_eq!(at(ix / 2, iy + side / 2), at(1, 1));
    // Under the caption, at the cover's bottom, it is darkened.
    let caption = spot.caption.clone().expect("the caption");
    let (cx, cy, cw, chh) = caption.rect;
    assert!(cx > ix && cy < iy + side);
    let darkest = (cx..cx + cw)
        .map(|px| at(px, iy + side - 3)[0])
        .min()
        .unwrap();
    assert!(darkest < 120, "{darkest}");
    // The caption's text is drawn over it: without it, its pixels differ.
    let mut bare = spot.clone();
    let (tw, th) = (cw as u32, chh as u32);
    bare.caption = Some(Caption {
        text: Handle::from_rgba(tw, th, vec![0; (tw * th * 4) as usize]),
        ..caption
    });
    covers.show(Some(&art), &[bare], 1.0);
    let without = drawn(&mut headless, &covers, (1280, 800), 1.0);
    let ink = (cx..cx + cw)
        .flat_map(|px| (cy..cy + chh).map(move |py| (px as u32, py as u32)))
        .filter(|&(px, py)| shot.get_pixel(px, py) != without.get_pixel(px, py))
        .count();
    assert!(ink > 200, "{ink}");

    // Without a cover: a placeholder and the caption under it.
    draw(&mut canvas, &state, None);
    let text = canvas.text();
    assert!(text.contains('\u{266a}') && text.contains("Morning Light"));
}

/// A cover's art is scaled once for its art and size: the same art at
/// the same size keeps its image (wherever it goes), another art or size
/// is fitted anew, and no art shows nothing.
#[test]
fn a_covers_art_is_kept_by_art_id_and_size() {
    let art = Art {
        id: 5,
        image: Arc::new(sample_cover(64)),
    };
    /// The image `art` shows with in a cover at `place`.
    fn image(covers: &mut Covers, art: Option<&Art>, place: PixelRect) -> Option<ImageId> {
        let spot = Spot {
            place,
            align: Align::Center,
            caption: None,
        };
        covers.show(art, &[spot], 1.0);
        covers.images().next().map(Handle::id)
    }
    let mut covers = Covers::default();
    let first = image(&mut covers, Some(&art), (10, 10, 200, 200));
    assert!(first.is_some());
    assert_eq!(image(&mut covers, Some(&art), (10, 10, 200, 200)), first);
    assert_eq!(image(&mut covers, Some(&art), (300, 40, 200, 200)), first);
    let other = Art {
        id: 6,
        image: art.image.clone(),
    };
    let anew = image(&mut covers, Some(&other), (300, 40, 200, 200));
    assert!(anew.is_some() && anew != first);
    let bigger = image(&mut covers, Some(&other), (300, 40, 240, 240));
    assert!(bigger.is_some() && bigger != anew);
    assert_eq!(image(&mut covers, None, (300, 40, 240, 240)), None);
    covers.show(Some(&art), &[], 1.0);
    assert_eq!(covers.images().count(), 0);
}

/// Frames drawn one after another (pictures kept where nothing under them
/// changed, cells redrawn only where they did) end in the same pixels as
/// the last frame drawn afresh: through covers that stay, change and go
/// (their cells left blank for the window), a caption over one and the
/// Transition tab.
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
    let mut tab = common::transition_state(canvas.grid().cols, canvas.grid().rows);
    tab.drawn_heard = Some(tab.playing.position);
    let mut tab_later = tab.clone();
    tab_later.playing.position += 0.5;
    tab_later.drawn_heard = Some(tab_later.playing.position);
    let steps: Vec<(&str, State, Option<&Art>)> = vec![
        ("main", main.clone(), Some(&first)),
        ("main, later", later, Some(&first)),
        ("now playing", screen(Screen::NowPlaying), Some(&first)),
        (
            "now playing, again",
            screen(Screen::NowPlaying),
            Some(&first),
        ),
        (
            "now playing, another cover",
            screen(Screen::NowPlaying),
            Some(&second),
        ),
        ("now playing, no cover", screen(Screen::NowPlaying), None),
        ("settings", screen(Screen::Settings), Some(&first)),
        ("main again", main.clone(), Some(&first)),
        ("transition", tab, Some(&first)),
        ("transition, later", tab_later, Some(&first)),
        ("main after transition", main.clone(), Some(&first)),
    ];
    for (name, state, art) in &steps {
        draw(&mut canvas, state, *art);
        let mut fresh = Canvas::new(80, 24, font_px(1.0));
        fresh.fit(1280, 800);
        draw(&mut fresh, state, *art);
        let (got, want) = (canvas.image(), fresh.image());
        let differ = got
            .pixels()
            .zip(want.pixels())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(differ, 0, "{name}: {differ} pixels differ");
    }
}

/// The Transition tab's lanes kept in tiles from frame to frame, painted
/// again only where their marks changed: a run of frames (the view
/// following the playhead by fractions of a pixel and by several, back
/// after a seek, held with the playhead going across it, the cursor and a
/// selection moving, a transition's curves shown, a zoom) shows at each
/// frame the lanes of that frame painted afresh, at scale 2; the canvas
/// under them changes no pixel in a frame where nothing moved.
#[test]
fn transition_frames_in_a_row_are_as_drawn_afresh() {
    let (width, height, scale) = (2400, 1500, 2.0f32);
    let mut canvas = Canvas::new(80, 24, font_px(f64::from(scale)));
    let grid = canvas.fit(width, height);
    let mut lanes = Lanes::default();
    let steps = transition_steps(grid);
    let picture = |state: &State| {
        let mut c = Canvas::new(80, 24, font_px(f64::from(scale)));
        c.fit(width, height);
        let mut l = Lanes::default();
        lanes_picture(&mut c, &mut l, state, (width, height, scale))
    };
    for (name, state) in &steps {
        let got = lanes_picture(&mut canvas, &mut lanes, state, (width, height, scale));
        let want = picture(state);
        let differ = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(differ, 0, "{name}: {differ} pixels differ");
    }
    let (name, last) = steps.last().unwrap();
    draw_heard(&mut canvas, last, None);
    assert!(!canvas.changed, "{name}, again: a pixel changed");
    // The words were there to be drawn: without them the lanes differ.
    let (_, first) = &steps[0];
    let mut bare = first.clone();
    bare.karaoke.lyrics.clear();
    let differ = picture(first)
        .iter()
        .zip(&picture(&bare))
        .filter(|(a, b)| a != b)
        .count();
    assert!(differ > 500, "the words: {differ} pixels");
}

/// A frame of `state` at what it says is heard drawn on `canvas`, and its
/// lanes framed by `lanes` as the processor shows them (no GPU) in a
/// window of `(width, height, scale)`: their picture over the window's
/// ground.
fn lanes_picture(
    canvas: &mut Canvas,
    lanes: &mut Lanes,
    state: &State,
    (width, height, scale): (u32, u32, f32),
) -> Vec<u32> {
    draw_heard(canvas, state, None);
    lanes.show(canvas.take_lanes(), scale, false);
    lanes.frame(state.heard());
    lanes.picture(width as usize, height as usize)
}

/// A frame of `state` at what it says is heard.
fn draw_heard(canvas: &mut Canvas, state: &State, art: Option<&Art>) {
    let scene = Scene {
        state,
        grid: canvas.grid(),
        pointer: None,
        heard: state.heard(),
    };
    canvas.draw(&scene, art);
}

/// The window's frames through the GPU, off screen: the pixels that
/// changed go over frame by frame (as [`Canvas::take_changes`] says), and
/// each frame read back is the canvas, with the background beyond it,
/// through the Transition tab's steps, the browser with a cover and a moving
/// frame drawn again, Now playing, a window resized, and a skipped take; into a
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
    let mut canvas = Canvas::new(80, 24, font_px(scale));
    // Not a whole number of cells: the background shows beyond them.
    let (mut width, mut height) = (2411, 1507);
    let grid = canvas.fit(width, height);
    let main = common::playing_state(grid.cols, grid.rows);
    let mut steps: Vec<(String, State)> = vec![
        ("main".into(), main.clone()),
        ("main, again".into(), main.clone()),
        ("main, still".into(), main.clone()),
    ];
    let mut now_playing = main.clone();
    now_playing.screen = Screen::NowPlaying;
    steps.push(("now playing".into(), now_playing));
    steps.extend(transition_steps(grid));
    steps.push(("main after transition".into(), main.clone()));
    for (i, (name, state)) in steps.iter().enumerate() {
        if i == 5 {
            // The window a little wider: the grid stays, the margin grows.
            width += 7;
        }
        draw_heard(&mut canvas, state, Some(&art));
        // One take skipped: what it held must come with the next.
        if i == 8 {
            continue;
        }
        let changes = canvas.take_changes();
        let got = gpu.show(&canvas, &changes, width, height).unwrap();
        let mut want = vec![0u32; (width * height) as usize];
        canvas.write_xrgb(&mut want, width as usize, height as usize);
        let differ = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(differ, 0, "{name} on {}: {differ} pixels differ", gpu.name);
    }
    // A smaller window: a smaller grid.
    (width, height) = (1900, 1200);
    let grid = canvas.fit(width, height);
    let mut small = main.clone();
    small.fit = layout::fit(&small, grid);
    draw_heard(&mut canvas, &small, Some(&art));
    let changes = canvas.take_changes();
    let got = gpu.show(&canvas, &changes, width, height).unwrap();
    let mut want = vec![0u32; (width * height) as usize];
    canvas.write_xrgb(&mut want, width as usize, height as usize);
    assert!(got == want, "smaller on {}: the pixels differ", gpu.name);
}

/// The Karaoke tab's voice lane on the GPU ([`Lanes`] through
/// [`LaneOffscreen`]): each frame read back is the lane painted afresh and
/// shown by the processor, as the song goes on, tiles coming in and the
/// word heard changing; also frames that move the lane alone, between
/// frames of the rest. Into a frame that takes the bytes as they are and
/// one that encodes them as sRGB. Skipped (with a note) without a GPU.
#[test]
fn the_karaoke_lane_scrolled_by_the_gpu_is_the_lane_painted() {
    for srgb in [false, true] {
        let Some(mut gpu) = lanes_gpu(srgb) else {
            return;
        };
        let (mut canvas, grid) = lanes_canvas();
        let mut lanes = Lanes::default();
        let mut state = common::transition_state(grid.cols, grid.rows);
        state.side = jodeldiplom::core::Side::Karaoke;
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
            let name = format!("frame {i}");
            same_through_gpu(&mut gpu, (&mut canvas, &mut lanes), &mut state, time, &name);
        }
    }
}

/// The Transition tab's lanes on the GPU (the playing track's bar numbers
/// and lanes following the playhead, the words on the voice's; the next
/// track's, the overviews): each frame read back is the lanes painted
/// afresh and shown by the processor, across bars (the cursor going
/// along), with a cue set and a transition's curves shown while it
/// follows. Into both kinds of frame; skipped without a GPU. The tiles of
/// the lanes gone are let go of.
#[test]
fn the_transition_lanes_scrolled_by_the_gpu_are_the_lanes_painted() {
    use jodeldiplom::core::keymap::Op;
    use jodeldiplom::core::{update, Action};
    for srgb in [false, true] {
        let Some(mut gpu) = lanes_gpu(srgb) else {
            return;
        };
        let (mut canvas, grid) = lanes_canvas();
        let mut lanes = Lanes::default();
        let mut state = common::transition_state(grid.cols, grid.rows);
        sung(&mut state);
        let mut step = |state: &mut State, time: f64, name: &str| {
            same_through_gpu(&mut gpu, (&mut canvas, &mut lanes), state, time, name);
        };
        for (i, &time) in [30.0, 30.2, 31.1, 33.9, 34.0, 41.5].iter().enumerate() {
            step(&mut state, time, &format!("following, {i}"));
        }
        update(&mut state, Action::Op(Op::Accept));
        step(&mut state, 42.0, "a cue set");
        update(&mut state, Action::Op(Op::TransitionPreset));
        step(&mut state, 42.3, "a transition's curves");
        step(&mut state, 47.0, "later");
        assert!(lanes.following(), "the playing track follows");
        assert!(gpu.held() >= 10, "{} lanes held", gpu.held());
        // The browser: no lanes, and none held once a frame went by.
        state.side = jodeldiplom::core::Side::Browser;
        draw_heard(&mut canvas, &state, None);
        lanes.show(canvas.take_lanes(), 2.0, true);
        lanes.frame(state.heard());
        let (width, height) = LANES_SIZE;
        gpu.show(&lanes, width, height).unwrap();
        assert_eq!(gpu.held(), 0, "the lanes' tiles let go of");
    }
}

fn lanes_gpu(srgb: bool) -> Option<LaneOffscreen> {
    match LaneOffscreen::new(srgb) {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("no GPU ({e}): skipped");
            None
        }
    }
}

const LANES_SIZE: (u32, u32) = (2411, 1507);

/// A canvas for a window of [`LANES_SIZE`] at scale 2.
fn lanes_canvas() -> (Canvas, Grid) {
    let mut canvas = Canvas::new(80, 24, font_px(2.0));
    let grid = canvas.fit(LANES_SIZE.0, LANES_SIZE.1);
    (canvas, grid)
}

/// `state` near `time` shown by `lanes` on the GPU, and by lanes made
/// afresh on the processor: the same pixels but at the cards' rounded
/// corners (blended apart). The lanes are moved alone first (frames
/// between frames of the rest), then to where those following the
/// playhead stand on a whole column; every frame goes over, as the
/// window's do.
fn same_through_gpu(
    gpu: &mut LaneOffscreen,
    (canvas, lanes): (&mut Canvas, &mut Lanes),
    state: &mut State,
    time: f64,
    name: &str,
) {
    let (width, height) = LANES_SIZE;
    let at = |state: &mut State, seconds: f64| {
        state.playing.position = seconds;
        state.drawn_heard = Some(seconds);
    };
    let mut heard = time;
    at(state, heard);
    draw_heard(canvas, state, None);
    lanes.show(canvas.take_lanes(), 2.0, true);
    for _ in 0..3 {
        lanes.frame(heard);
        gpu.show(lanes, width, height).unwrap();
        // Back to a whole column of the lanes that follow.
        let following = lanes.stands().iter().find(|s| s.1 > 0.0);
        heard -= following.map_or(0.0, |&(_, frac, px)| f64::from(frac) * px);
    }
    at(state, heard);
    draw_heard(canvas, state, None);
    lanes.show(canvas.take_lanes(), 2.0, true);
    lanes.frame(heard);
    assert!(
        lanes.stands().iter().all(|s| s.1 == 0.0),
        "{name}: not on a whole column: {:?}",
        lanes.stands()
    );
    let got = gpu.show(lanes, width, height).unwrap();
    let mut fresh_canvas = lanes_canvas().0;
    let want = lanes_picture(
        &mut fresh_canvas,
        &mut Lanes::default(),
        state,
        (width, height, 2.0),
    );
    let radius = (2.0 * jodeldiplom::gui::look::radius::PANEL).ceil() as usize;
    let corner = |x: usize, y: usize| {
        lanes.places().iter().any(|&(x0, y0, w, h)| {
            let near = |v: usize, from: usize, len: usize| {
                v >= from && v < from + len && (v < from + radius || v >= from + len - radius)
            };
            near(x, x0, w) && near(y, y0, h)
        })
    };
    let mut differ = 0;
    for (n, (a, b)) in got.iter().zip(&want).enumerate() {
        let (x, y) = (n % width as usize, n / width as usize);
        if a != b && !corner(x, y) {
            if differ < 5 {
                eprintln!("{name}: ({x}, {y}) {a:06x} against {b:06x}");
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
fn transition_steps(grid: Grid) -> Vec<(String, State)> {
    use jodeldiplom::core::keymap::Op;
    use jodeldiplom::core::{update, Action};
    let mut state = common::transition_state(grid.cols, grid.rows);
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
        g1.cols.abs_diff(g2.cols) <= 2 && g1.rows.abs_diff(g2.rows) <= 2,
        "{g1:?} {g2:?}"
    );
    // Without a cover the placeholder is drawn in cells.
    let state = common::playing_state(g1.cols, g1.rows);
    draw(&mut c1, &state, None);
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

/// The Transition tab's lanes are the window's, over the cells the canvas
/// leaves blank for them: each painted where its cells are.
#[test]
fn the_transition_lanes_are_painted_in_pixels() {
    let mut canvas = Canvas::new(80, 24, font_px(1.0));
    let grid = canvas.fit(1600, 1000);
    let mut state = common::transition_state(grid.cols, grid.rows);
    state.drawn_heard = Some(0.0);
    let picture = lanes_picture(
        &mut canvas,
        &mut Lanes::default(),
        &state,
        (1600, 1000, 1.0),
    );
    let image = rgb_image(&picture, 1600, 1000);
    let Layout::Main(m) = layout::compute(&state, canvas.grid()) else {
        panic!("the main screen");
    };
    let tab = m.transition.expect("the Transition tab");
    assert_eq!(tab.blocks.len(), 2);
    assert_eq!(m.bottom.spectrum, 0);
    // The vocals of the playing track come in at 60 s: before, nothing is
    // painted in their lane but its card; the drums lane is busy.
    let card = Rgb(jodeldiplom::gui::look::bytes(
        jodeldiplom::gui::look::SURFACE,
    ));
    let lane_pixels = |i: usize| {
        let (_, _, area) = tab.blocks[0].lanes[i];
        let (x, y, w, h) = pixel_rect(&canvas, area);
        let mut lit = 0;
        for yy in y + 10..y + h - 10 {
            for xx in x + 10..x + w - 10 {
                if *image.get_pixel(xx, yy) != card {
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
    let mut state = common::transition_state(grid.cols, grid.rows);
    // A view that starts between cells.
    state.transition.now.path = lanes::path(&state, Block::Now).cloned();
    state.transition.now.start = Some(10.123);
    state.drawn_heard = Some(0.0);
    let picture = lanes_picture(
        &mut canvas,
        &mut Lanes::default(),
        &state,
        (1600, 1000, 1.0),
    );
    let image = rgb_image(&picture, 1600, 1000);
    let Layout::Main(m) = layout::compute(&state, canvas.grid()) else {
        panic!("the main screen");
    };
    let ruler = m.transition.unwrap().blocks[0].ruler;
    let (x0, y0, w, h) = pixel_rect(&canvas, ruler);
    let (start, end) = viewport::window(&state, Block::Now, state.heard());
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
    let mut state = common::transition_state(grid.cols, grid.rows);
    draw(&mut canvas, &state, None);
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
                output_frame: 0,
                from_frame: 0,
            },
        ))),
    );
    state.drawn_heard = Some(0.0);
    let picture = lanes_picture(
        &mut canvas,
        &mut Lanes::default(),
        &state,
        (1600, 1000, 1.0),
    );
    let image = rgb_image(&picture, 1600, 1000);
    let Layout::Main(m) = layout::compute(&state, canvas.grid()) else {
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
    draw(&mut canvas, &state, None);
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
    draw(&mut canvas, &state, None);
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
        bar: 5.0 / 24.0,
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
    let state = common::desk_state(grid.cols, grid.rows);
    draw(&mut canvas, &state, None);
    let Layout::Main(m) = layout::compute(&state, canvas.grid()) else {
        panic!("the main screen");
    };
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
