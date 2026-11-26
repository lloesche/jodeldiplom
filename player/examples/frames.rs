//! Times the window's frames headless, as the window draws them while a
//! track plays (a fixture through the null output, the Transition tab's
//! playhead going on): per frame what the window's frame does beside
//! iced's view (whose cost is `view_cost`'s): what its size does to the
//! keys ([`metrics::fit`]) and the lanes ([`Lanes`]: their tiles painted
//! and their frame, for the GPU or, without one ("cpu"), as the
//! processor's images), and what that would cost at 30 frames a second.
//! Karaoke mode is `frame_cost`'s.
//! "transition held" is the Transition tab with its view held (the cursor
//! moved), the playhead going across it; "karaoke tab" has the voice's
//! lane going by.
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

use iced::Size;
use jodel_output::SimClock;
use jodeldiplom::app::{AppOptions, Output};
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::{Action, Screen, Side, State};
use jodeldiplom::gui::lanes::{Lanes, Metrics};
use jodeldiplom::gui::{lane_spots, metrics};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let number = |i: usize, default: f64| -> f64 {
        args.get(i).and_then(|a| a.parse().ok()).unwrap_or(default)
    };
    let (width, height, scale) = (
        number(0, 3024.0) as f32,
        number(1, 1890.0) as f32,
        number(2, 2.0) as f32,
    );
    let frames = number(3, 240.0) as usize;
    let size = Size::new(width / scale, height / scale);
    let lane_metrics = Metrics::at(scale);
    let letter = lane_metrics.cell.0 as f32 / scale;
    let cells = ((size.width / 8.0) as u16, (size.height / 19.0) as u16);

    let tree = common::music_tree("frames");
    let mut options = AppOptions::new(tree.album.clone());
    options.output = Output::Null(SimClock::Realtime);
    options.saved = options.config.clone();
    let mut app = common::fitted_app(options, cells.0, cells.1);
    let listed = common::wait_for(&mut app, Duration::from_secs(10), |app| {
        !app.state.browser.loading && !app.state.details.is_empty()
    });
    assert!(listed, "the folder was not listed");
    app.dispatch(Action::SelectRow(2));
    app.dispatch(Action::Op(Op::OpenReplace));
    let playing = common::wait_for(&mut app, Duration::from_secs(10), |app| {
        app.state.playing.track.is_some() && app.state.playing.position >= 0.05
    });
    assert!(playing, "the track did not start");

    println!(
        "{width} × {height} pixels at scale {scale}, {frames} frames per screen (ms per frame; a core's share at 30 fps, the rate while a track plays)"
    );
    let screens = [
        ("main", true),
        ("nowplaying", true),
        ("transition", true),
        ("transition held", true),
        ("karaoke tab", true),
        ("transition", false),
        ("transition held", false),
        ("karaoke tab", false),
    ];
    for (screen, gpu) in screens {
        let mut lanes = Lanes::default();
        let mut tab = common::transition_state(cells.0, cells.1);
        if screen == "transition held" {
            // The view stays where it is (its cursor moved); the playhead
            // goes across it.
            tab.transition.now.path = Some(tab.queue.paths[1].clone());
            tab.transition.now.start = Some(tab.playing.position - 5.0);
        }
        if screen == "karaoke tab" {
            tab.side = Side::Karaoke;
            common::sung_every(&mut tab, 0.5);
        }
        let (mut fitted, mut framed) = (Duration::ZERO, Duration::ZERO);
        for frame in 0..frames {
            app.pump();
            let state: State = match screen {
                "transition" | "transition held" | "karaoke tab" => {
                    tab.playing.position += 1.0 / 60.0;
                    tab.drawn_heard = Some(tab.playing.position);
                    tab.clone()
                }
                _ => {
                    let mut s = app.state.clone();
                    if screen == "nowplaying" {
                        s.screen = Screen::NowPlaying;
                    }
                    s
                }
            };
            let started = Instant::now();
            std::hint::black_box(metrics::fit(size, &state, letter));
            let fit_took = started.elapsed();
            let started = Instant::now();
            let spots = lane_spots(&state, size, scale, lane_metrics, state.heard());
            lanes.show(spots, scale, gpu);
            lanes.frame(state.heard());
            let lanes_took = started.elapsed();
            // The first frames paint everything; the rest only what changes.
            if frame >= 10 {
                fitted += fit_took;
                framed += lanes_took;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        let n = (frames.saturating_sub(10)).max(1) as f64;
        let ms = |d: Duration| d.as_secs_f64() * 1e3 / n;
        let total = ms(fitted) + ms(framed);
        let on = if gpu { "gpu" } else { "cpu" };
        println!(
            "  {screen:15} {on}  fit {:6.3}  lanes {:6.2}  = {total:6.2} ms  ({:.0} % of a core)",
            ms(fitted),
            ms(framed),
            total * 30.0 / 10.0,
        );
    }
    app.shutdown();
    let _ = std::fs::remove_dir_all(&tree.root);
}
