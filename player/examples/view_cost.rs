//! What the window's view costs the processor a frame, as the window
//! builds it every tick: [`Gui::view`] for a screen with a fixture track
//! playing through the null output, its widgets' tree brought up to date
//! and laid out, and drawn by iced's headless renderer (its lists of what
//! to draw; the GPU is not waited for), at 1512 × 982 at scale 2 (a
//! 14-inch MacBook Pro's window). Each frame is ticked first, as in the
//! window (the lanes framed: not timed). For each screen, each part's
//! median and worst, and the same of the frames' totals.
//!
//! The main screen, laid out by iced (the tabs' header, the browser's
//! rows, the right column with the cover and its text: links, tags, the
//! pulse; the bottom bar: the spectrum, the seek bar and the buttons), the
//! same on a folder of 2,000 files (the cost must not grow with the
//! folder), the queue, the Mixer tab, the Transition and Karaoke tabs
//! (their lanes) on generated tracks with stems, Karaoke mode on them (its
//! stage and its controls' bar), and the screens drawn whole: Now playing,
//! the settings (the device list open) and the help.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example view_cost -- [FRAMES] [--budget MEDIAN_MS WORST_MS]
//! ```
//!
//! 600 frames by default, the first 30 not counted. `--budget`: exits
//! with 1 (after printing it all) where a screen's median or worst frame
//! took longer in all (the budgets to hold to, and on which machine:
//! PERFORMANCE.md).

#[path = "../tests/common/mod.rs"]
mod common;
mod headless;

use std::sync::Arc;
use std::time::{Duration, Instant};

use iced::{window, Size};
use jodel_output::SimClock;
use jodeldiplom::app::{AppOptions, Output};
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::{Action, Screen, Side, State};
use jodeldiplom::gui::window::{Gui, Message};

use headless::{Headless, Took};

/// The window's size (logical pixels) and scale.
const SIZE: Size = Size::new(1512.0, 982.0);
const SCALE: f32 = 2.0;

/// The frames at the start not counted (the caches filling).
const WARM: usize = 30;

/// The median and the worst of `times`, in milliseconds.
fn spread(times: &mut [Duration]) -> (f64, f64) {
    times.sort();
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    (ms(times[times.len() / 2]), ms(times[times.len() - 1]))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let frames = args
        .first()
        .and_then(|a| a.parse().ok())
        .unwrap_or(600usize)
        .max(WARM + 1);
    let budget = args.iter().position(|a| a == "--budget").map(|i| {
        let ms = |k: usize| -> f64 {
            args.get(i + k)
                .and_then(|a| a.parse().ok())
                .expect("--budget MEDIAN_MS WORST_MS")
        };
        (ms(1), ms(2))
    });

    let tree = common::music_tree("view_cost");
    let mut options = AppOptions::new(tree.album.clone());
    options.output = Output::Null(SimClock::Realtime);
    options.saved = options.config.clone();
    // No microphone of this machine opened on the Mixer tab.
    options.mic = Arc::new(|_| Err("no microphone".into()));
    options.mics = Arc::new(|| Ok(Vec::new()));
    // Fitted as the window would tell it, so the rows shown get their details.
    let mut app = common::fitted_app(options, 120, 36);
    let listed = common::wait_for(&mut app, Duration::from_secs(10), |app| {
        !app.state.browser.loading && !app.state.details.is_empty()
    });
    assert!(listed, "the folder was not listed");
    app.dispatch(Action::SelectRow(2));
    app.dispatch(Action::Op(Op::OpenReplace));
    let playing = common::wait_for(&mut app, Duration::from_secs(10), |app| {
        app.state.playing.track.is_some()
            && app.state.playing.position >= 0.05
            && app.art().is_some()
    });
    assert!(playing, "the track did not start (with its cover)");

    // A folder of `n` files, the selection in the middle: the list makes
    // only the rows in view, a page whatever the folder.
    let folder = |n: usize| {
        let mut big = app.state.clone();
        let dir = big.browser.dir.clone();
        big.browser.entries.truncate(1);
        big.browser
            .entries
            .extend((0..n).map(|i| jodeldiplom::browser::Entry {
                name: format!("{i:04} Track.flac"),
                path: dir.join(format!("{i:04} Track.flac")),
                kind: jodeldiplom::browser::EntryKind::File,
                relative: None,
            }));
        jodeldiplom::core::update(&mut big, Action::SelectRow(n / 2));
        big
    };
    let screens: Vec<(&str, State)> = vec![
        ("main", app.state.clone()),
        ("main, 50 entries", folder(50)),
        ("main, 2,000 entries", folder(2_000)),
        ("queue", {
            let mut q = app.state.clone();
            q.side = Side::Queue;
            q
        }),
        ("mixer", {
            // The player at 96 kHz, a preview crossfading over it, the
            // microphone's strip selected and hearing (as the shots).
            let mut d = app.state.clone();
            d.side = Side::Mixer;
            d.desk.selected = jodeldiplom::core::desk::Channel::Mic;
            common::fill_desk(&mut d, 96_000);
            d
        }),
        // The Transition tab on generated tracks with stems and waveforms
        // (its lanes the window's widget), and the Karaoke tab on it (the
        // voice's lane).
        ("transition", common::transition_state(120, 36)),
        ("karaoke tab", {
            let mut k = common::transition_state(120, 36);
            k.side = Side::Karaoke;
            k
        }),
        // Karaoke mode on it: its stage (a widget of its own), and its
        // controls' bar faded in over it.
        ("karaoke mode", {
            let mut k = common::transition_state(120, 36);
            k.side = Side::Karaoke;
            k.screen = Screen::Karaoke;
            k
        }),
        ("karaoke mode, controls", {
            let mut k = common::transition_state(120, 36);
            k.side = Side::Karaoke;
            k.screen = Screen::Karaoke;
            jodeldiplom::core::update(&mut k, Action::PointerMoved { from_bottom: 0.0 });
            let shown = k.now_ms + jodeldiplom::core::karaoke::controls::FADE_MS;
            jodeldiplom::core::update(&mut k, Action::Tick(shown));
            k
        }),
        ("nowplaying", {
            let mut n = app.state.clone();
            n.screen = Screen::NowPlaying;
            n
        }),
        ("settings", {
            let mut s = app.state.clone();
            s.screen = Screen::Settings;
            s.settings.picking = true;
            s
        }),
        ("help", {
            let mut h = app.state.clone();
            h.screen = Screen::Help;
            h
        }),
    ];
    let mut gui = Gui::headless(app);
    let mut headless = Headless::new();
    let id = window::Id::unique();
    let event = |e| Message::Event(iced::Event::Window(e), id);
    let _ = gui.update(event(window::Event::Resized(SIZE)));
    let _ = gui.update(event(window::Event::Rescaled(SCALE)));
    println!(
        "{} × {} at scale {SCALE}, drawn by {}, {frames} frames per screen, the first {WARM} not counted (ms per frame: median, worst)",
        SIZE.width,
        SIZE.height,
        headless.name()
    );
    let mut over = false;
    for (name, state) in screens {
        gui.show(state);
        let mut took: Vec<Took> = Vec::with_capacity(frames);
        for _ in 0..frames {
            let _ = gui.update(Message::Frame(Instant::now()));
            took.push(headless.frame(&gui, SIZE));
        }
        let took = &took[WARM..];
        let parts = |t: &Took| [t.view, t.layout, t.draw, t.view + t.layout + t.draw];
        println!("{name}:");
        for (k, part) in ["view", "layout", "draw", "in all"].into_iter().enumerate() {
            let mut times: Vec<Duration> = took.iter().map(|t| parts(t)[k]).collect();
            let (median, worst) = spread(&mut times);
            println!("  {part:<7} {median:6.3} {worst:6.3}");
            if part == "in all" {
                if let Some((m, w)) = budget {
                    if median > m || worst > w {
                        println!("  over the budget ({m} ms median, {w} ms worst)");
                        over = true;
                    }
                }
            }
        }
    }
    drop(gui);
    let _ = std::fs::remove_dir_all(&tree.root);
    if over {
        std::process::exit(1);
    }
}
