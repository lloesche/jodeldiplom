//! Times the window's frames headless, as the window draws them while a track
//! plays (a fixture through the null output, its spectrum moving, the
//! Transition tab's playhead going on): per frame the widgets with their
//! cells, the pictures, and the copy into the window's buffer (for the frames
//! that changed a pixel: the window shows only those), what that would cost at
//! 30 frames a second, and how many frames changed. "transition held" is the
//! Transition tab with its view held (the cursor moved), the playhead going
//! across it.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example frames -- [WIDTH HEIGHT SCALE] [FRAMES]
//! ```
//!
//! Default: 3024 × 1890 pixels at scale 2 (a 16-inch MacBook Pro's
//! window, maximised), 240 frames per screen.

#[path = "../tests/common/mod.rs"]
mod common;

use std::time::{Duration, Instant};

use jodel_output::SimClock;
use jodeldiplom::app::{App, AppOptions, Output};
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::{Action, Screen, State};
use jodeldiplom::gui::{font_px, Canvas};
use jodeldiplom::view::{colors, Links, Scene};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let number = |i: usize, default: f64| -> f64 {
        args.get(i).and_then(|a| a.parse().ok()).unwrap_or(default)
    };
    let (width, height, scale) = (
        number(0, 3024.0) as usize,
        number(1, 1890.0) as usize,
        number(2, 2.0),
    );
    let frames = number(3, 240.0) as usize;

    let tree = common::music_tree("frames");
    let mut options = AppOptions::new(tree.album.clone());
    options.output = Output::Null(SimClock::Realtime);
    options.saved = options.config.clone();
    let mut app = App::new(options);
    let listed = common::wait_for(&mut app, Duration::from_secs(10), |app| {
        !app.state.browser.loading && !app.state.details.is_empty()
    });
    assert!(listed, "the folder was not listed");
    app.dispatch(Action::SelectRow(2));
    app.dispatch(Action::Op(Op::Open));
    let playing = common::wait_for(&mut app, Duration::from_secs(10), |app| {
        app.state.playing.track.is_some() && app.state.playing.position >= 0.05
    });
    assert!(playing, "the track did not start");

    println!(
        "{width} × {height} pixels at scale {scale}, {frames} frames per screen (ms per frame; a core's share at 30 fps, the rate while a track plays)"
    );
    for screen in ["main", "nowplaying", "transition", "transition held"] {
        let mut canvas = Canvas::new(80, 24, font_px(scale));
        let grid = canvas.fit(width as u32, height as u32);
        let mut buffer = vec![0u32; width * height];
        let mut tab = common::transition_state(grid.0, grid.1);
        if screen == "transition held" {
            // The view stays where it is (its cursor moved); the playhead
            // goes across it.
            tab.transition.now.path = Some(tab.queue.paths[1].clone());
            tab.transition.now.start = Some(tab.playing.position - 5.0);
        }
        let (mut widgets, mut pictures, mut copy) =
            (Duration::ZERO, Duration::ZERO, Duration::ZERO);
        let mut shown = 0;
        for frame in 0..frames {
            app.pump();
            let spectrum = app.spectrum();
            let state: State = match screen {
                "transition" | "transition held" => {
                    tab.playing.position += 1.0 / 60.0;
                    tab.drawn_heard = Some(tab.playing.position);
                    tab.clone()
                }
                _ => {
                    let mut s = app.state.clone();
                    s.size = grid;
                    if screen == "nowplaying" {
                        s.screen = Screen::NowPlaying;
                    }
                    s
                }
            };
            let scene = Scene {
                state: &state,
                spectrum: &spectrum,
                background: colors::BG,
                links: Links::OnHover(None),
                heard: state.heard(),
            };
            canvas.draw(&scene, app.art());
            let started = Instant::now();
            // As the window: a frame that changed no pixel is not shown.
            if canvas.changed {
                canvas.write_xrgb(&mut buffer, width, height);
            }
            // The first frames draw everything; the rest only what changes.
            if frame >= 10 {
                widgets += canvas.last_draw.widgets;
                pictures += canvas.last_draw.pictures;
                copy += started.elapsed();
                shown += usize::from(canvas.changed);
            }
            std::hint::black_box(&buffer);
            std::thread::sleep(Duration::from_millis(2));
        }
        let n = (frames.saturating_sub(10)).max(1) as f64;
        let ms = |d: Duration| d.as_secs_f64() * 1e3 / n;
        let total = ms(widgets) + ms(pictures) + ms(copy);
        println!(
            "  {screen:10}  widgets {:6.2}  pictures {:6.2}  copy {:6.2}  = {total:6.2} ms  ({:.0} % of a core), {:.0} % of frames shown",
            ms(widgets),
            ms(pictures),
            ms(copy),
            total * 30.0 / 10.0,
            100.0 * shown as f64 / n
        );
    }
    app.shutdown();
    let _ = std::fs::remove_dir_all(&tree.root);
}
