//! The window's drawing path without a window: the old canvas (Karaoke
//! mode alone), the covers in the places the screens make for them, the
//! lanes where the screens lay them out.

mod common;
#[path = "../examples/headless/mod.rs"]
mod headless;

use std::sync::Arc;
use std::time::Instant;

use headless::Headless;
use iced::advanced::image::{Handle, Id as ImageId};
use iced::Size;
use image::{Rgb, RgbaImage};
use jodel_output::SimClock;
use jodeldiplom::app::{App, AppOptions, Output};
use jodeldiplom::art::{sample_cover, Art};
use jodeldiplom::core::{Screen, Side, State};
use jodeldiplom::gui::lanes::{Lanes, Offscreen as LaneOffscreen};
use jodeldiplom::gui::widgets::cover::Covers;
use jodeldiplom::gui::window::{Gui, Message};
use jodeldiplom::gui::{font_px, lane_spots, metrics, transition, Canvas};
use jodeldiplom::icon::Style;

use jodeldiplom::view::{palette, Scene};

fn bg() -> Rgb<u8> {
    Rgb(palette::BG)
}

fn draw(canvas: &mut Canvas, state: &State) {
    let scene = Scene {
        state,
        grid: canvas.grid(),
        heard: 0.0,
    };
    canvas.draw(&scene);
}

fn sized(canvas: &mut Canvas, w: u32, h: u32) -> State {
    let grid = canvas.fit(w, h);
    common::playing_state(grid.cols, grid.rows)
}

/// The canvas draws no screen any more: on the main screen (every side),
/// the screens the window lays out whole and Karaoke mode it leaves every
/// pixel to the window's own widgets.
#[test]
fn the_canvas_draws_no_screen() {
    let mut canvas = Canvas::new(80, 24, font_px(1.0));
    let main = sized(&mut canvas, 1280, 800);
    for side in [
        Side::Browser,
        Side::Queue,
        Side::Transition,
        Side::Karaoke,
        Side::Mixer,
    ] {
        for screen in [
            Screen::Main,
            Screen::NowPlaying,
            Screen::Settings,
            Screen::Help,
            Screen::Karaoke,
        ] {
            let mut state = main.clone();
            (state.side, state.screen) = (side, screen);
            draw(&mut canvas, &state);
            let image = canvas.image();
            assert!(image.pixels().all(|p| *p == bg()), "{side:?}, {screen:?}");
            assert!(canvas.text().trim().is_empty(), "{side:?}, {screen:?}");
        }
    }
}

/// `words` (`0x00RRGGBB`, `width` × `height`) as an image.
fn rgb_image(words: &[u32], width: u32, height: u32) -> image::RgbImage {
    let bytes = words
        .iter()
        .flat_map(|&w| jodeldiplom::gui::pixels::unpack(w))
        .collect();
    image::RgbImage::from_raw(width, height, bytes).expect("the picture's size")
}

/// Whether two colours are within `by` of each other in every channel.
fn near(a: &[u8], b: &[u8], by: u8) -> bool {
    a.iter().zip(b).all(|(p, q)| p.abs_diff(*q) <= by)
}

/// A player with no output, no devices and no microphone.
fn player() -> App {
    let dir = common::temp_dir("gui-window");
    let mut options = AppOptions::new(dir);
    options.output = Output::Null(SimClock::Realtime);
    options.devices = Arc::new(|| Ok(Vec::new()));
    options.mic = Arc::new(|_| Err("no microphone".into()));
    options.mics = Arc::new(|| Ok(Vec::new()));
    App::new(options)
}

/// A window of `size` at scale 1 showing `state` with `art` as the cover,
/// a frame drawn.
fn window(state: State, art: Option<image::RgbImage>, size: Size) -> Gui {
    let mut gui = Gui::headless(player());
    let event = |e| Message::Event(iced::Event::Window(e), iced::window::Id::unique());
    let _ = gui.update(event(iced::window::Event::Resized(size)));
    let _ = gui.update(event(iced::window::Event::Rescaled(1.0)));
    gui.show(state);
    gui.set_art(art);
    let _ = gui.update(Message::Frame(Instant::now()));
    gui
}

/// The covers go where the screens make room for them: on the main
/// screen at the top left of the right column's place for it, a card with
/// a note on it there without art; on Now playing in the middle of the
/// window; none on the settings.
#[test]
fn the_covers_go_where_the_screens_make_room_for_them() {
    let size = Size::new(1280.0, 800.0);
    let mut headless = Headless::new();
    let state = common::playing_state(160, 42);
    let place = metrics::main(size, Side::Browser).cover.expect("a cover");
    let side = place.width.min(place.height);
    let cover = sample_cover(side as u32);
    let middle = cover.get_pixel(side as u32 / 2, side as u32 / 2).0;
    let shot = |gui: &Gui, headless: &mut Headless| {
        headless.frame(gui, size);
        headless.settled(gui, size);
        headless.screenshot(1.0)
    };
    let at = |image: &RgbaImage, x: f32, y: f32| {
        let p = image.get_pixel(x as u32, y as u32).0;
        [p[0], p[1], p[2]]
    };
    let gui = window(state.clone(), Some(sample_cover(512)), size);
    let image = shot(&gui, &mut headless);
    let (cx, cy) = (place.x + side / 2.0, place.y + side / 2.0);
    assert!(
        near(&at(&image, cx, cy), &middle, 24),
        "the right column's cover"
    );
    // Without art: the card, a note in its middle.
    let gui = window(state.clone(), None, size);
    let image = shot(&gui, &mut headless);
    let card = jodeldiplom::gui::look::bytes(jodeldiplom::gui::look::RAISED);
    assert!(near(&at(&image, place.x + 8.0, place.y + 8.0), &card, 2));
    // Now playing: in the middle of the window.
    let mut now = state.clone();
    now.screen = Screen::NowPlaying;
    let gui = window(now, Some(sample_cover(512)), size);
    let image = shot(&gui, &mut headless);
    let big = sample_cover(800 - 2 * 24);
    let mid = big.get_pixel(big.width() / 2, big.height() / 2).0;
    assert!(
        near(&at(&image, 640.0, 400.0), &mid, 24),
        "Now playing's cover"
    );
    // The settings: none.
    let mut settings = state;
    settings.screen = Screen::Settings;
    let gui = window(settings, Some(sample_cover(512)), size);
    let image = shot(&gui, &mut headless);
    assert!(
        !near(&at(&image, cx, cy), &middle, 24),
        "no cover on the settings"
    );
}

/// A cover drawn in its place, at scale 1 and 2, in the first frame it
/// shows (a large image too, which iced would otherwise send to the GPU in
/// the background): the art fitted into its place at the pixels it shows
/// at, its corners rounded.
#[test]
fn a_cover_is_drawn_fitted_with_its_corners_rounded() {
    let mut headless = Headless::new();
    for scale in [1.0, 2.0] {
        let size = ((1280.0 * scale) as u32, (800.0 * scale) as u32);
        let art = Art {
            id: 7,
            image: Arc::new(sample_cover(256)),
        };
        let place = ((600.0 * scale) as u32, (500.0 * scale) as u32);
        let mut covers = Covers::default();
        covers.show(Some(&art), &[place], scale);
        let window = Size::new(size.0 as f32 / scale, size.1 as f32 / scale);
        headless.draw(covers.art(0).expect("art"), window, Instant::now());
        let shot = headless.screenshot(scale);
        let side = place.0.min(place.1) as usize;
        let cover = sample_cover(side as u32);
        let at = |dx: usize, dy: usize| shot.get_pixel(dx as u32, dy as u32).0;
        let want = |dx: usize, dy: usize| cover.get_pixel(dx as u32, dy as u32).0;
        let (mid, r) = (side / 2, (10.0 * scale) as usize);
        assert!(near(&at(mid, mid), &want(mid, mid), 24), "scale {scale}");
        // The corner is cut off; a radius in along the diagonal is cover.
        assert!(!near(&at(0, 0), &want(0, 0), 40), "scale {scale}");
        assert!(near(&at(r, r), &want(r, r), 24), "scale {scale}");
        assert!(!near(&at(side - 1, 0), &want(side - 1, 0), 40));
    }
}

/// A cover's art is scaled once for its art and size: the same art at
/// the same size keeps its image, another art or size is fitted anew, and
/// no art shows nothing.
#[test]
fn a_covers_art_is_kept_by_art_id_and_size() {
    let art = Art {
        id: 5,
        image: Arc::new(sample_cover(64)),
    };
    /// The image `art` shows with in a cover in a place of `place`.
    fn image(covers: &mut Covers, art: Option<&Art>, place: (u32, u32)) -> Option<ImageId> {
        covers.show(art, &[place], 1.0);
        covers.images().next().map(Handle::id)
    }
    let mut covers = Covers::default();
    let first = image(&mut covers, Some(&art), (200, 200));
    assert!(first.is_some());
    assert_eq!(image(&mut covers, Some(&art), (200, 200)), first);
    let other = Art {
        id: 6,
        image: art.image.clone(),
    };
    let anew = image(&mut covers, Some(&other), (200, 200));
    assert!(anew.is_some() && anew != first);
    let bigger = image(&mut covers, Some(&other), (240, 240));
    assert!(bigger.is_some() && bigger != anew);
    assert_eq!(image(&mut covers, None, (240, 240)), None);
    covers.show(Some(&art), &[], 1.0);
    assert_eq!(covers.images().count(), 0);
}

/// The Transition tab's lanes kept in tiles from frame to frame, painted
/// again only where their marks changed: a run of frames (the view
/// following the playhead by fractions of a pixel and by several, back
/// after a seek, held with the playhead going across it, the cursor and a
/// selection moving, a transition's curves shown, a zoom) shows at each
/// frame the lanes of that frame painted afresh, at scale 2.
#[test]
fn transition_frames_in_a_row_are_as_drawn_afresh() {
    let (width, height, scale) = (2400, 1500, 2.0f32);
    let mut lanes = Lanes::default();
    let steps = transition_steps(Size::new(1200.0, 750.0));
    let picture = |state: &State| {
        let mut l = Lanes::default();
        lanes_picture(&mut l, state, (width, height, scale))
    };
    for (name, state) in &steps {
        let got = lanes_picture(&mut lanes, state, (width, height, scale));
        let want = picture(state);
        let differ = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(differ, 0, "{name}: {differ} pixels differ");
    }
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

/// The sizes the lanes are painted at in a window of `scale`: the
/// canvas's cells'.
fn lane_metrics(scale: f32) -> jodeldiplom::gui::lanes::Metrics {
    Canvas::new(80, 24, font_px(f64::from(scale))).metrics()
}

/// The lanes `state` shows at what it says is heard, framed by `lanes` on
/// the GPU (`gpu`) or the processor, in a window of
/// `(width, height, scale)` pixels.
fn frame_lanes(
    lanes: &mut Lanes,
    state: &State,
    (width, height, scale): (u32, u32, f32),
    gpu: bool,
) {
    let size = Size::new(width as f32 / scale, height as f32 / scale);
    let heard = state.heard();
    lanes.show(
        lane_spots(state, size, scale, lane_metrics(scale), heard),
        scale,
        gpu,
    );
    lanes.frame(heard);
}

/// The lanes of `state` at what it says is heard framed by `lanes` as the
/// processor shows them (no GPU) in a window of `(width, height, scale)`:
/// their picture over the window's ground.
fn lanes_picture(lanes: &mut Lanes, state: &State, window: (u32, u32, f32)) -> Vec<u32> {
    frame_lanes(lanes, state, window, false);
    lanes.picture(window.0 as usize, window.1 as usize)
}

/// A frame of `state` at what it says is heard.
fn draw_heard(canvas: &mut Canvas, state: &State) {
    let scene = Scene {
        state,
        grid: canvas.grid(),
        heard: state.heard(),
    };
    canvas.draw(&scene);
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
    steps.extend(transition_steps(common::window(grid.cols, grid.rows)));
    steps.push(("main after transition".into(), main.clone()));
    for (i, (name, state)) in steps.iter().enumerate() {
        if i == 5 {
            // The window a little wider: the grid stays, the margin grows.
            width += 7;
        }
        draw_heard(&mut canvas, state);
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
    common::fit(&mut small, grid.cols, grid.rows);
    draw_heard(&mut canvas, &small);
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
        let mut lanes = Lanes::default();
        let (cols, rows) = LANES_CELLS;
        let mut state = common::transition_state(cols, rows);
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
            same_through_gpu(&mut gpu, &mut lanes, &mut state, time, &name);
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
        let mut lanes = Lanes::default();
        let (cols, rows) = LANES_CELLS;
        let mut state = common::transition_state(cols, rows);
        sung(&mut state);
        let mut step = |state: &mut State, time: f64, name: &str| {
            same_through_gpu(&mut gpu, &mut lanes, state, time, name);
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
        let (width, height) = LANES_SIZE;
        frame_lanes(&mut lanes, &state, (width, height, 2.0), true);
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

/// The lanes' window, in pixels at scale 2.
const LANES_SIZE: (u32, u32) = (2411, 1507);

/// The old cells that window's state is fitted to (8 × 19 points each).
const LANES_CELLS: (u16, u16) = (150, 39);

/// `state` near `time` shown by `lanes` on the GPU, and by lanes made
/// afresh on the processor: the same pixels but at the cards' rounded
/// corners (blended apart). The lanes are moved alone first (frames
/// between frames of the rest), then to where those following the
/// playhead stand on a whole column; every frame goes over, as the
/// window's do.
fn same_through_gpu(
    gpu: &mut LaneOffscreen,
    lanes: &mut Lanes,
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
    let window = (width, height, 2.0);
    frame_lanes(lanes, state, window, true);
    for _ in 0..3 {
        lanes.frame(heard);
        gpu.show(lanes, width, height).unwrap();
        // Back to a whole column of the lanes that follow.
        let following = lanes.stands().iter().find(|s| s.1 > 0.0);
        heard -= following.map_or(0.0, |&(_, frac, px)| f64::from(frac) * px);
    }
    at(state, heard);
    frame_lanes(lanes, state, window, true);
    assert!(
        lanes.stands().iter().all(|s| s.1 == 0.0),
        "{name}: not on a whole column: {:?}",
        lanes.stands()
    );
    let got = gpu.show(lanes, width, height).unwrap();
    let want = lanes_picture(&mut Lanes::default(), state, window);
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
fn transition_steps(size: Size) -> Vec<(String, State)> {
    use jodeldiplom::core::keymap::Op;
    use jodeldiplom::core::{update, Action};
    let cells = |points: f32, cell: f32| (points / cell) as u16;
    let mut state = common::transition_state(cells(size.width, 8.0), cells(size.height, 19.0));
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

/// The Transition tab as the window lays it out in a window of `size`
/// at a scale of 1 (its points the pixels).
fn transition_plan(state: &State, size: Size) -> transition::Plan {
    let room = transition::Room::of(state, size, 1.0).expect("the Transition tab");
    transition::plan(state, &room)
}

/// `rect`'s pixels at a scale of 1.
fn pixels_of(rect: iced::Rectangle) -> (u32, u32, u32, u32) {
    let px = |v: f32| v.round() as u32;
    (px(rect.x), px(rect.y), px(rect.width), px(rect.height))
}

/// A window of 1600 × 1000 at scale 1, and the Transition tab's state
/// fitted to it.
fn transition_window() -> (Size, State) {
    let size = Size::new(1600.0, 1000.0);
    (size, common::transition_state(200, 52))
}

/// The Transition tab's lanes are the window's, where the tab leaves room
/// for them: each painted there.
#[test]
fn the_transition_lanes_are_painted_in_pixels() {
    let (size, mut state) = transition_window();
    state.drawn_heard = Some(0.0);
    let picture = lanes_picture(&mut Lanes::default(), &state, (1600, 1000, 1.0));
    let image = rgb_image(&picture, 1600, 1000);
    let tab = transition_plan(&state, size);
    assert_eq!(tab.blocks.len(), 2);
    assert_eq!(metrics::main(size, Side::Transition).spectrum, 0.0);
    // The vocals of the playing track come in at 60 s: before, nothing is
    // painted in their lane but its card; the drums lane is busy.
    let card = Rgb(jodeldiplom::gui::look::bytes(
        jodeldiplom::gui::look::RAISED,
    ));
    let lane_pixels = |i: usize| {
        let (x, y, w, h) = pixels_of(tab.blocks[0].lanes[i].rect);
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
    let (size, mut state) = transition_window();
    // A view that starts between cells.
    state.transition.now.path = lanes::path(&state, Block::Now).cloned();
    state.transition.now.start = Some(10.123);
    state.drawn_heard = Some(0.0);
    let picture = lanes_picture(&mut Lanes::default(), &state, (1600, 1000, 1.0));
    let image = rgb_image(&picture, 1600, 1000);
    let ruler = transition_plan(&state, size).blocks[0].ruler.unwrap();
    let (x0, y0, w, h) = pixels_of(ruler);
    let (start, end) = viewport::window(&state, Block::Now, state.heard());
    let a = lanes::rhythmed(&state, Block::Now)
        .unwrap()
        .analysis
        .clone();
    let cw = lane_metrics(1.0).cell.0 as u32;
    // On the panel's ground.
    let ground = Rgb(jodeldiplom::gui::look::bytes(
        jodeldiplom::gui::look::SURFACE,
    ));
    let ink = |x: u32| (y0..y0 + h).any(|y| *image.get_pixel(x, y) != ground);
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
    let (size, mut state) = transition_window();
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
    let picture = lanes_picture(&mut Lanes::default(), &state, (1600, 1000, 1.0));
    let image = rgb_image(&picture, 1600, 1000);
    let tab = transition_plan(&state, size);
    assert_eq!(tab.blocks.len(), 2);
    for lane in &tab.blocks[1].lanes {
        let (x, y, w, h) = pixels_of(lane.rect);
        let lit = (y..y + h)
            .flat_map(|yy| (x..x + w).map(move |xx| (xx, yy)))
            .filter(|&(xx, yy)| *image.get_pixel(xx, yy) != bg())
            .count();
        assert_eq!(lit, 0, "the next block's lanes show nothing yet");
    }
}

/// Karaoke mode's stage on the GPU: a word ahead of the playhead shows as
/// its pill, one sung and hit as gold, the singer's arrow at the playhead,
/// over its ground (black, without a backdrop). Skipped (with a note)
/// without a GPU.
#[test]
fn the_stage_draws_the_lane_over_its_ground() {
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
    let shown = gpu.show_frame(w, h).expect("a frame");
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
