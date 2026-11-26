//! The screens drawn in iced's own widgets, through the window's view in
//! iced's simulator: what their buttons, rows, tabs, keys, drags and
//! wheel do.

mod common;
#[path = "../examples/headless/mod.rs"]
mod headless;
#[path = "common/setup.rs"]
mod setup_web;

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use headless::Headless;
use iced::keyboard::key::Named;
use iced::keyboard::Key;
use iced::mouse::{self, ScrollDelta};
use iced::widget::Id;
use iced::{window, Point, Size};
use iced_test::selector::{Bounded, Selector};
use iced_test::simulator::{press_key, Simulator};
use jodel_output::SimClock;
use jodeldiplom::app::{App, AppOptions, Output};
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::{update, Action, Effect, Gpu, Screen, Side, State};
use jodeldiplom::gui::pointer::Grab;
use jodeldiplom::gui::window::{Gui, Message};
use jodeldiplom::view::setup::Choice;
use setup_web::{start, when, Made, Web};

/// The window's size in the tests.
const SIZE: Size = Size::new(1280.0, 800.0);

/// The headless renderers (iced's simulator, the shots' renderer), one
/// test at a time: each makes a GPU device of its own, and several made
/// and dropped at once on other threads can take the whole test process
/// down (on Windows, with no panic to say so).
fn gpu() -> MutexGuard<'static, ()> {
    static GPU: Mutex<()> = Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}
/// The messages a click on what `selector` finds in `gui`'s view sends
/// (an error where it is not shown).
fn click<S>(gui: &Gui, selector: S) -> Result<Vec<Message>, iced_test::Error>
where
    S: Selector + Send,
    S::Output: Bounded + Clone + Send + Sync + 'static,
{
    let mut sim = Simulator::with_size(jodeldiplom::gui::iced_settings(), SIZE, gui.view());
    sim.click(selector)?;
    Ok(sim.into_messages().collect())
}

/// `event` as the window hears it.
fn hear(gui: &mut Gui, event: iced::Event) {
    let _ = gui.update(Message::Event(event, window::Id::unique()));
}

/// A player with no output, no devices and no microphone.
fn player() -> App {
    let dir = common::temp_dir("screens-window");
    let mut options = AppOptions::new(dir);
    options.output = Output::Null(SimClock::Realtime);
    options.devices = Arc::new(|| Ok(Vec::new()));
    options.mic = Arc::new(|_| Err("no microphone".into()));
    options.mics = Arc::new(|| Ok(Vec::new()));
    App::new(options)
}

/// A window of [`SIZE`] at scale 1 round a player with no output, showing
/// `state`, a frame drawn (the bottom bar where the cell layout leaves it).
fn window(state: State) -> Gui {
    window_of(player(), state)
}

/// [`window`] round `app`.
fn window_of(app: App, state: State) -> Gui {
    window_sized(app, state, SIZE)
}

/// [`window_of`] of `size`.
fn window_sized(app: App, state: State, size: Size) -> Gui {
    let mut gui = Gui::headless(app);
    hear(&mut gui, iced::Event::Window(window::Event::Resized(size)));
    hear(&mut gui, iced::Event::Window(window::Event::Rescaled(1.0)));
    gui.show(state);
    let _ = gui.update(Message::Frame(Instant::now()));
    gui
}

/// What `messages` ask of the core, the only messages there.
fn acts(messages: &[Message]) -> Vec<Action> {
    messages
        .iter()
        .map(|m| match m {
            Message::Act(action) => action.clone(),
            other => panic!("{other:?}"),
        })
        .collect()
}

/// The bottom bar's buttons each do what their key does, a stem's switch
/// too (with a GPU); a click is the whole of it, press and release.
#[test]
fn the_bars_buttons_are_hit() {
    let _gpu = gpu();
    let mut state = common::playing_state(160, 42);
    state.gpu = Gpu::Present("a GPU".into());
    let gui = window(state);
    let op = |messages: &[Message]| match acts(messages)[..] {
        [Action::Op(op)] => op,
        ref other => panic!("{other:?}"),
    };
    for (id, wanted) in [
        ("previous", Op::Previous),
        ("pause", Op::PlayPause),
        ("next", Op::Next),
        ("stop", Op::Stop),
        ("settings", Op::Settings),
        ("help", Op::Help),
    ] {
        assert_eq!(op(&click(&gui, Id::new(id)).unwrap()), wanted, "{id}");
    }
    assert_eq!(op(&click(&gui, "drums").unwrap()), Op::ToggleDrums);
}

/// A press on the seek bar shows where it would go; the pointer then
/// moves it from anywhere in the window (off the bar, off the window's
/// side); the release seeks there, once. With nothing playing a press
/// starts nothing.
#[test]
fn dragging_the_bar_seeks_on_release() {
    let _gpu = gpu();
    let state = common::playing_state(160, 42);
    let length = state.playing.duration().unwrap();
    let mut gui = window(state);
    let mut sim = Simulator::with_size(jodeldiplom::gui::iced_settings(), SIZE, gui.view());
    let track = sim.find(Id::new("seek")).unwrap().bounds();
    // Pressed at the middle: a grab of the track, the drag there.
    sim.point_at(Point::new(track.x + track.width / 2.0, track.center_y()));
    sim.simulate([iced::Event::Mouse(mouse::Event::ButtonPressed(
        mouse::Button::Left,
    ))]);
    let messages: Vec<Message> = sim.into_messages().collect();
    assert!(
        matches!(messages[..], [Message::Grab(Grab::Seek(b), _)] if b == track),
        "{messages:?}"
    );
    for message in messages {
        let _ = gui.update(message);
    }
    let held = |gui: &Gui| gui.state().unwrap().seek_drag;
    assert!(held(&gui).is_some_and(|f| (f - 0.5).abs() < 0.01));
    // Moved far off the bar, past the window's right edge: the end.
    let moved = |gui: &mut Gui, x: f32, y: f32| {
        hear(
            gui,
            iced::Event::Mouse(mouse::Event::CursorMoved {
                position: Point::new(x, y),
            }),
        );
    };
    moved(&mut gui, SIZE.width + 40.0, 12.0);
    assert_eq!(held(&gui), Some(1.0));
    // A quarter of the way, up over the list: there.
    moved(&mut gui, track.x + track.width / 4.0, 200.0);
    assert!(held(&gui).is_some_and(|f| (f - 0.25).abs() < 0.01));
    // Let go of there: sought, and nothing held.
    hear(
        &mut gui,
        iced::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)),
    );
    let state = gui.state().unwrap();
    assert_eq!(state.seek_drag, None);
    assert!((state.playing.position - length / 4.0).abs() < length / 100.0);
    // Another release or move does nothing more.
    let position = state.playing.position;
    moved(&mut gui, track.x, track.center_y());
    hear(
        &mut gui,
        iced::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)),
    );
    assert_eq!(gui.state().unwrap().playing.position, position);

    // Nothing playing: the bar takes no press.
    let mut stopped = common::playing_state(160, 42);
    stopped.playing.track = None;
    let gui = window(stopped);
    let mut sim = Simulator::with_size(jodeldiplom::gui::iced_settings(), SIZE, gui.view());
    sim.point_at(Point::new(track.x + track.width / 2.0, track.center_y()));
    sim.simulate([iced::Event::Mouse(mouse::Event::ButtonPressed(
        mouse::Button::Left,
    ))]);
    assert_eq!(sim.into_messages().count(), 0);
}

/// The wheel over the bar scrolls the list, as over the list.
#[test]
fn the_wheel_over_the_bar_scrolls_the_list() {
    let _gpu = gpu();
    let state = common::playing_state(160, 42);
    let cursor = state.browser.cursor;
    let mut gui = window(state);
    let mut sim = Simulator::with_size(jodeldiplom::gui::iced_settings(), SIZE, gui.view());
    let track = sim.find(Id::new("seek")).unwrap().bounds();
    sim.point_at(Point::new(track.center_x(), track.y - 20.0));
    sim.scroll(ScrollDelta::Lines { x: 0.0, y: -1.0 });
    let messages: Vec<Message> = sim.into_messages().collect();
    assert!(matches!(messages[..], [Message::Wheel(_)]), "{messages:?}");
    for message in messages {
        let _ = gui.update(message);
    }
    assert!(gui.state().unwrap().browser.cursor > cursor);
}

/// The setup choice among `messages`, the only message there.
fn chosen(messages: &[Message]) -> Option<Choice> {
    match messages {
        [Message::Setup(choice)] => Some(*choice),
        _ => None,
    }
}

/// `key` pressed in the window.
fn press(gui: &mut Gui, key: Key) {
    let _ = gui.update(Message::Event(press_key(key, None), window::Id::unique()));
}

/// Whether setup asks for `n` downloads in all within two seconds.
fn downloads_come_to(made: &Made, n: usize) -> bool {
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(2) {
        if made.downloads.load(Ordering::SeqCst) == n {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    false
}

/// K8: after a failure the screen offers only Try again and Quit, each a
/// button and a key (Enter or r, q or Esc); while setup goes on, only
/// Quit. Try again goes on from where it stopped; once Quit is taken,
/// nothing more is tried.
#[test]
fn the_setup_screen_offers_try_again_and_quit_by_click_and_key() {
    let _gpu = gpu();
    let dir = std::env::temp_dir().join(format!("jodeldiplom-screens-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    // Under way: Quit alone.
    let stalls = Web::Stalls {
        share: 0.5,
        after: Duration::ZERO,
    };
    let (setup, _made) = start(stalls, &dir.join("stalls"));
    when(&setup, |s| s.current == Some(1) && s.done > 0);
    let gui = Gui::headless_setup(setup, AppOptions::new(dir.clone()));
    assert!(click(&gui, "Try again").is_err());
    assert_eq!(chosen(&click(&gui, "Quit").unwrap()), Some(Choice::Quit));

    // Stopped: Try again by click, by Enter and by r, each going on from
    // where setup stopped (the download that failed asked for again).
    let (setup, made) = start(Web::Offline, &dir.join("offline"));
    when(&setup, |s| s.failure.is_some());
    let mut gui = Gui::headless_setup(setup, AppOptions::new(dir.clone()));
    assert_eq!(made.downloads.load(Ordering::SeqCst), 2);
    let retry = click(&gui, "Try again").unwrap();
    assert_eq!(chosen(&retry), Some(Choice::Retry));
    for message in retry {
        let _ = gui.update(message);
    }
    assert!(downloads_come_to(&made, 3), "Try again was not taken");
    press(&mut gui, Key::Named(Named::Enter));
    assert!(downloads_come_to(&made, 4), "Enter did not try again");
    press(&mut gui, Key::Character("r".into()));
    assert!(downloads_come_to(&made, 5), "r did not try again");

    // Quit by click; then nothing more is tried, whatever is pressed.
    let quit = click(&gui, "Quit").unwrap();
    assert_eq!(chosen(&quit), Some(Choice::Quit));
    for message in quit {
        let _ = gui.update(message);
    }
    press(&mut gui, Key::Named(Named::Enter));
    assert!(!downloads_come_to(&made, 6), "tried again after Quit");

    // q and Esc quit as well.
    for key in [Key::Character("q".into()), Key::Named(Named::Escape)] {
        let (setup, made) = start(Web::Offline, &dir.join("offline-keys"));
        when(&setup, |s| s.failure.is_some());
        let mut gui = Gui::headless_setup(setup, AppOptions::new(dir.clone()));
        press(&mut gui, key.clone());
        press(&mut gui, Key::Named(Named::Enter));
        assert!(!downloads_come_to(&made, 3), "tried again after {key:?}");
        let _ = std::fs::remove_dir_all(dir.join("offline-keys"));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The simulator over `gui`'s view.
fn simulator(gui: &Gui) -> Simulator<'_, Message> {
    Simulator::with_size(jodeldiplom::gui::iced_settings(), SIZE, gui.view())
}

/// The browser's row titled `title` in `state`.
fn row_of(state: &State, title: &str) -> usize {
    state
        .browser
        .entries
        .iter()
        .position(|e| e.name.contains(title))
        .unwrap()
}

/// Each tab shows its side, by a click on its name; the queue's carries
/// its length.
#[test]
fn the_tabs_show_each_side() {
    let _gpu = gpu();
    let gui = window(common::playing_state(160, 42));
    for (name, side) in [
        ("Browser", Side::Browser),
        ("Queue", Side::Queue),
        ("Transition", Side::Transition),
        ("Karaoke", Side::Karaoke),
        ("Mixer", Side::Mixer),
    ] {
        let shown = acts(&click(&gui, name).unwrap());
        assert!(
            matches!(shown[..], [Action::ShowSide(s)] if s == side),
            "{name}: {shown:?}"
        );
    }
    assert!(simulator(&gui).find("5").is_ok(), "the queue's length");
}

/// A click on a browser row is the core's click on it: two on one row
/// make a double click, which is enter on it (queued, nothing played);
/// clicks far apart only select. The wheel over the list scrolls it, a
/// right press queues the row.
#[test]
fn a_double_click_on_a_row_is_enter_and_the_wheel_scrolls() {
    let _gpu = gpu();
    let state = common::playing_state(160, 42);
    let row = row_of(&state, "Undertow");
    let gui = window(state);
    let mut sim = simulator(&gui);
    sim.click("Undertow").unwrap();
    sim.click("Undertow").unwrap();
    let clicks = acts(&sim.into_messages().collect::<Vec<_>>());
    assert!(
        matches!(clicks[..], [Action::ClickRow(a), Action::ClickRow(b)] if a == row && b == row),
        "{clicks:?}"
    );
    // The core's double click is enter on the row.
    let engine = |fx: &[Effect]| -> Vec<jodel_playback::Command> {
        fx.iter()
            .filter_map(|e| match e {
                Effect::Engine(c) => Some(c.clone()),
                _ => None,
            })
            .collect()
    };
    let mut s = gui.state().unwrap().clone();
    update(&mut s, Action::Tick(1_000));
    update(&mut s, clicks[0].clone());
    update(&mut s, Action::Tick(1_100));
    let fx = update(&mut s, clicks[1].clone());
    let mut enter = gui.state().unwrap().clone();
    enter.browser.cursor = row;
    let by_key = update(&mut enter, Action::Op(Op::Open));
    assert_eq!(engine(&fx), engine(&by_key));
    assert!(matches!(
        engine(&fx)[..],
        [jodel_playback::Command::Enqueue(ref p)] if p.len() == 1
    ));
    // Far apart: only selected.
    update(&mut s, Action::Tick(5_000));
    update(&mut s, clicks[0].clone());
    update(&mut s, Action::Tick(6_000));
    assert!(engine(&update(&mut s, clicks[0].clone())).is_empty());

    // A right press queues the row.
    let mut sim = simulator(&gui);
    let at = sim.find("Undertow").unwrap().bounds().center();
    sim.point_at(at);
    sim.simulate([iced::Event::Mouse(mouse::Event::ButtonPressed(
        mouse::Button::Right,
    ))]);
    let queued = acts(&sim.into_messages().collect::<Vec<_>>());
    assert!(
        matches!(queued[..], [Action::EnqueueRow(r)] if r == row),
        "{queued:?}"
    );

    // The wheel over the rows scrolls the list: three rows a notch.
    let mut gui = gui;
    let before = gui.state().unwrap().browser.cursor;
    let mut sim = simulator(&gui);
    sim.point_at(at);
    sim.scroll(ScrollDelta::Lines { x: 0.0, y: -1.0 });
    let messages: Vec<Message> = sim.into_messages().collect();
    assert!(matches!(messages[..], [Message::Wheel(_)]), "{messages:?}");
    for message in messages {
        let _ = gui.update(message);
    }
    let after = gui.state().unwrap().browser.cursor;
    assert_eq!(
        after,
        (before + 3).min(gui.state().unwrap().browser.entries.len() - 1)
    );
}

/// Q4: a double click on a queue row is two clicks on it, each holding
/// the row (as a drag would): the core's double click, which plays it.
#[test]
fn a_double_click_on_a_queue_row_is_two_clicks_on_it() {
    let _gpu = gpu();
    let mut state = common::playing_state(160, 42);
    update(&mut state, Action::ShowSide(Side::Queue));
    let mut gui = window(state);
    let mut sim = simulator(&gui);
    sim.click("Glass Rooms").unwrap();
    sim.click("Glass Rooms").unwrap();
    let messages: Vec<Message> = sim.into_messages().collect();
    let clicks: Vec<&Action> = messages
        .iter()
        .filter_map(|m| match m {
            Message::Act(action) => Some(action),
            _ => None,
        })
        .collect();
    assert!(
        matches!(clicks[..], [Action::ClickRow(3), Action::ClickRow(3)]),
        "{messages:?}"
    );
    let grabs = messages
        .iter()
        .filter(|m| matches!(m, Message::Grab(Grab::QueueRow { .. }, _)))
        .count();
    assert_eq!(grabs, 2, "{messages:?}");
    // Through the window: held while pressed, let go of on release.
    let up = || iced::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left));
    let mut pressed = messages.into_iter();
    for _ in 0..2 {
        for message in pressed.by_ref().take(2) {
            let _ = gui.update(message);
        }
        assert!(gui.state().unwrap().queue.dragging);
        hear(&mut gui, up());
        assert!(!gui.state().unwrap().queue.dragging);
    }
    assert_eq!(gui.state().unwrap().queue.cursor, 3);
}

/// A queue row held goes where the pointer takes it: over another row
/// its track moves there (the selection with it); let go of, it stays.
#[test]
fn dragging_a_queue_row_moves_its_track() {
    let _gpu = gpu();
    let mut state = common::playing_state(160, 42);
    update(&mut state, Action::ShowSide(Side::Queue));
    let first = state.queue.paths[0].clone();
    let mut gui = window(state);
    let mut sim = simulator(&gui);
    let to = sim.find("Glass Rooms").unwrap().bounds().center();
    sim.click("Opening").unwrap();
    let messages: Vec<Message> = sim.into_messages().take(2).collect();
    for message in messages {
        let _ = gui.update(message);
    }
    assert!(gui.state().unwrap().queue.dragging);
    hear(
        &mut gui,
        iced::Event::Mouse(mouse::Event::CursorMoved { position: to }),
    );
    let q = &gui.state().unwrap().queue;
    assert_eq!(q.paths[3], first, "{:?}", q.paths);
    assert_eq!(q.cursor, 3);
    // Off the rows: it stays where it was taken.
    hear(
        &mut gui,
        iced::Event::Mouse(mouse::Event::CursorMoved {
            position: Point::new(SIZE.width - 4.0, 4.0),
        }),
    );
    assert_eq!(gui.state().unwrap().queue.paths[3], first);
    hear(
        &mut gui,
        iced::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)),
    );
    assert!(!gui.state().unwrap().queue.dragging);
}

/// With Ctrl held for a preview, the pointer moving over a browser row
/// selects it (the preview follows); without Ctrl it does not.
#[test]
fn ctrl_over_a_row_selects_it() {
    let _gpu = gpu();
    let state = common::playing_state(160, 42);
    let row = row_of(&state, "Coda");
    let mut gui = window(state);
    let moved_over = |gui: &Gui| {
        let mut sim = simulator(gui);
        let at = sim.find("Coda").unwrap().bounds().center();
        sim.point_at(at);
        sim.simulate([iced::Event::Mouse(mouse::Event::CursorMoved {
            position: at,
        })]);
        // The old canvas under the pane hears every move too (as off it).
        sim.into_messages()
            .filter_map(|m| match m {
                Message::Act(action) => Some(action),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    assert!(moved_over(&gui).is_empty());
    hear(
        &mut gui,
        iced::Event::Keyboard(iced::keyboard::Event::ModifiersChanged(
            iced::keyboard::Modifiers::CTRL,
        )),
    );
    assert!(gui.state().unwrap().preview.ctrl);
    let selected = moved_over(&gui);
    assert!(
        matches!(selected[..], [Action::SelectRow(r)] if r == row),
        "{selected:?}"
    );
}

/// The list shows the core's page: as many rows as the window told the
/// core, from the first in view; the keys page by them, and the last
/// page ends on the last row.
#[test]
fn the_list_shows_the_page_the_keys_turn() {
    let _gpu = gpu();
    let mut state = common::playing_state(160, 42);
    let parent = state.browser.entries[0].clone();
    state.browser.entries = std::iter::once(parent)
        .chain((0..200).map(|i| jodeldiplom::browser::Entry {
            name: format!("Track {i:03}.flac"),
            path: format!("/music/Album/Track {i:03}.flac").into(),
            kind: jodeldiplom::browser::EntryKind::File,
            relative: None,
        }))
        .collect();
    state.browser.cursor = 0;
    let mut gui = window(state);
    let rows = usize::from(gui.state().unwrap().fit.list_rows);
    assert!((8..30).contains(&rows), "{rows} rows");
    let shown = |gui: &Gui, i: usize| simulator(gui).find(format!("Track {i:03}.flac")).is_ok();
    // The first page: the parent row and the tracks after it.
    assert!(shown(&gui, rows - 2) && !shown(&gui, rows - 1));
    // A page down moves a page less a row.
    press(&mut gui, Key::Named(Named::PageDown));
    assert_eq!(gui.state().unwrap().browser.cursor, rows - 1);
    // The end: the last row last.
    press(&mut gui, Key::Named(Named::End));
    let s = gui.state().unwrap();
    assert_eq!(s.browser.cursor, 200);
    assert_eq!(s.browser.scroll, 201 - rows);
    assert!(shown(&gui, 199) && shown(&gui, 200 - rows) && !shown(&gui, 199 - rows));
}

/// The command palette over everything: a click on a command runs it; a
/// press beside the card closes the palette, and nothing under it takes
/// the press.
#[test]
fn the_palette_runs_a_click_and_closes_beside_it() {
    let _gpu = gpu();
    let mut state = common::playing_state(160, 42);
    update(&mut state, Action::Op(Op::Palette));
    for c in "ne".chars() {
        update(&mut state, Action::Type(c));
    }
    let gui = window(state);
    let run = acts(&click(&gui, "Next track").unwrap());
    assert!(matches!(run[..], [Action::PaletteRun(_)]), "{run:?}");
    let mut sim = simulator(&gui);
    // Over a browser row, under the palette's shade.
    sim.point_at(Point::new(40.0, SIZE.height - 120.0));
    sim.simulate(iced_test::simulator::click());
    let closed = acts(&sim.into_messages().collect::<Vec<_>>());
    assert!(matches!(closed[..], [Action::Op(Op::Back)]), "{closed:?}");
}

/// The messages `events` send with the pointer moved to `at` first (as a
/// pointer does: a link knows it is under the pointer once it has moved
/// there).
fn at_point(gui: &Gui, at: Point, events: &[iced::Event]) -> Vec<Message> {
    let mut sim = Simulator::with_size(jodeldiplom::gui::iced_settings(), SIZE, gui.view());
    sim.point_at(at);
    sim.simulate([iced::Event::Mouse(mouse::Event::CursorMoved {
        position: at,
    })]);
    sim.simulate(events.iter().cloned());
    sim.into_messages().collect()
}

/// A press and release of the main button.
fn a_click() -> [iced::Event; 2] {
    [
        iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)),
        iced::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)),
    ]
}

/// Where `text` shows in `gui`'s view.
fn bounds_of(gui: &Gui, text: &str) -> iced::Rectangle {
    let mut sim = Simulator::with_size(jodeldiplom::gui::iced_settings(), SIZE, gui.view());
    sim.find(text)
        .unwrap_or_else(|e| panic!("{text:?}: {e}"))
        .bounds()
}

/// Whether `text` shows in `gui`'s view.
fn shows(gui: &Gui, text: &str) -> bool {
    let mut sim = Simulator::with_size(jodeldiplom::gui::iced_settings(), SIZE, gui.view());
    sim.find(text).is_ok()
}

/// On Now playing a click returns, as the key does; the caption's title,
/// artist and album are links, each going where its key goes.
#[test]
fn a_click_on_now_playing_returns_and_its_links_jump() {
    let _gpu = gpu();
    let mut state = common::playing_state(160, 42);
    state.screen = Screen::NowPlaying;
    let mut app = player();
    app.set_art(image::RgbImage::from_pixel(
        64,
        64,
        image::Rgb([40, 60, 90]),
    ));
    let gui = window_of(app, state);
    let op = |messages: &[Message]| match acts(messages)[..] {
        [Action::Op(op)] => op,
        ref other => panic!("{other:?}"),
    };
    // The cover's middle, and the ground beside it.
    let middle = Point::new(SIZE.width / 2.0, SIZE.height / 3.0);
    assert_eq!(op(&at_point(&gui, middle, &a_click())), Op::NowPlaying);
    assert_eq!(
        op(&at_point(&gui, Point::new(8.0, 8.0), &a_click())),
        Op::NowPlaying
    );
    let title = bounds_of(&gui, "Morning Light");
    assert_eq!(op(&at_point(&gui, title.center(), &a_click())), Op::GoTrack);
    let by = bounds_of(&gui, "The Ramps · Synthetic Sessions");
    let artist = Point::new(by.x + 6.0, by.center_y());
    assert_eq!(op(&at_point(&gui, artist, &a_click())), Op::GoArtist);
    let album = Point::new(by.x + by.width - 6.0, by.center_y());
    assert_eq!(op(&at_point(&gui, album, &a_click())), Op::GoAlbum);
}

/// Now playing: the cover in the middle of the window (here a light one),
/// as tall as the window but for its margin, darkened towards its bottom
/// under the caption, and the caption's text drawn over it; without a
/// cover, the caption under a placeholder.
#[test]
fn now_playing_is_the_cover_with_the_caption_over_it() {
    let _gpu = gpu();
    let light = image::RgbImage::from_pixel(64, 64, image::Rgb([235, 225, 210]));
    let shot = |title: &str| {
        let mut state = common::playing_state(160, 42);
        state.screen = Screen::NowPlaying;
        let track = state.playing.track.as_mut().unwrap();
        let mut info = (*track.info).clone();
        info.tags.title = Some(title.into());
        track.info = Arc::new(info);
        let mut app = player();
        app.set_art(light.clone());
        let gui = window_of(app, state);
        let mut headless = Headless::new();
        headless.frame(&gui, SIZE);
        headless.screenshot(1.0)
    };
    let first = shot("Morning Light");
    let at = |px: f32, py: f32| first.get_pixel(px as u32, py as u32).0;
    // Square, as tall as the window but for the margin, in the middle.
    let side = SIZE.height - 2.0 * 24.0;
    let left = (SIZE.width - side) / 2.0;
    let near =
        |a: [u8; 4], b: [u8; 3], by: u8| a[..3].iter().zip(b).all(|(p, q)| p.abs_diff(q) <= by);
    assert!(near(
        at(SIZE.width / 2.0, 24.0 + side / 4.0),
        [235, 225, 210],
        2
    ));
    assert_eq!(at(left / 2.0, SIZE.height / 2.0), at(1.0, 1.0));
    // At its bottom, under the caption, it is darkened.
    let bottom = 24.0 + side - 3.0;
    let darkest = (0..side as u32 / 2)
        .map(|dx| at(left + 8.0 + dx as f32, bottom)[0])
        .min()
        .unwrap();
    assert!(darkest < 120, "{darkest}");
    // The caption's text is drawn there: another title, other pixels.
    let second = shot("Wavelength Arrives");
    let differ = (0..side as u32)
        .flat_map(|dx| (0..(side / 3.0) as u32).map(move |dy| (dx, dy)))
        .filter(|&(dx, dy)| {
            let (x, y) = ((left as u32) + dx, (24.0 + side * 2.0 / 3.0) as u32 + dy);
            first.get_pixel(x, y) != second.get_pixel(x, y)
        })
        .count();
    assert!(differ > 500, "{differ}");

    // Without a cover: the caption all the same.
    let mut state = common::playing_state(160, 42);
    state.screen = Screen::NowPlaying;
    let gui = window(state);
    assert!(shows(&gui, "Morning Light"));
    assert!(shows(&gui, "♪"));
}

/// The settings' rows and controls take clicks: a row does what enter
/// does on it (the old screen's value), a segment of a choice steps its
/// row there (and the one set takes none), a device in the open list is
/// chosen.
#[test]
fn settings_rows_step_values() {
    let _gpu = gpu();
    use jodel_playback::Headroom;
    let mut state = common::playing_state(160, 42);
    state.screen = Screen::Settings;
    let mut gui = window(state.clone());
    // The headroom's third choice, 1 dB: two steps on.
    let messages = click(&gui, Id::new("setting-1-2")).unwrap();
    assert!(
        matches!(acts(&messages)[..], [Action::SettingsClick(1, 2)]),
        "{messages:?}"
    );
    for message in messages {
        let _ = gui.update(message);
    }
    assert_eq!(gui.state().unwrap().config.headroom, Headroom(10));
    // Back to none: two steps back. The one set takes no click.
    let messages = click(&gui, Id::new("setting-1-0")).unwrap();
    assert!(matches!(
        acts(&messages)[..],
        [Action::SettingsClick(1, -2)]
    ));
    assert!(click(&gui, Id::new("setting-1-2")).unwrap().is_empty());
    // The tempo's row: a click turns it off, as enter does.
    let bpm = gui.state().unwrap().config.bpm;
    let messages = click(&gui, Id::new("setting-2")).unwrap();
    assert!(matches!(acts(&messages)[..], [Action::SettingsClick(2, 0)]));
    for message in messages {
        let _ = gui.update(message);
    }
    assert_eq!(gui.state().unwrap().config.bpm, !bpm);
    // The device's row opens the list; a device in it is chosen.
    state.settings.picking = true;
    state.settings.devices = Some(vec![
        jodeldiplom::core::DeviceChoice {
            id: None,
            name: "System default".into(),
            summary: String::new(),
        },
        jodeldiplom::core::DeviceChoice {
            id: Some("usb".into()),
            name: "Speakers (USB Audio DAC)".into(),
            summary: "44.1 48 kHz · 2 ch".into(),
        },
    ]);
    let gui = window(state);
    let messages = click(&gui, "Speakers (USB Audio DAC)").unwrap();
    assert!(
        matches!(acts(&messages)[..], [Action::ChooseDevice(1)]),
        "{messages:?}"
    );
}

/// The window's keys on the help: `?`, `/`, then "volume" typed (a
/// capital too), as iced reports them. The window shows only the device
/// volume's keys under the search; esc clears the search, a second
/// leaves.
#[test]
fn the_window_searches_the_help() {
    let _gpu = gpu();
    let mut gui = window(common::playing_state(160, 42));
    let ch = |c: &str| Key::Character(c.into());
    press(&mut gui, ch("?"));
    press(&mut gui, ch("/"));
    for c in ["V", "o", "l", "u", "m", "e"] {
        press(&mut gui, ch(c));
    }
    let _ = gui.update(Message::Frame(Instant::now()));
    let state = gui.state().unwrap();
    assert_eq!(state.screen, Screen::Help);
    assert!(!state.quit);
    for part in ["Volume", "alt+↑", "alt+↓", "device volume a step down"] {
        assert!(shows(&gui, part), "missing {part:?}");
    }
    for gone in ["play / pause", "Browser and queue"] {
        assert!(!shows(&gui, gone), "{gone:?} shows");
    }
    press(&mut gui, Key::Named(Named::Escape));
    assert!(shows(&gui, "play / pause"));
    assert_eq!(gui.state().unwrap().screen, Screen::Help);
    press(&mut gui, Key::Named(Named::Escape));
    assert_eq!(gui.state().unwrap().screen, Screen::Main);
}

/// The wheel over the help scrolls it three lines a step, as over the
/// old screen; the help shows as many lines as the window reports to
/// the core.
#[test]
fn the_wheel_scrolls_the_help() {
    let _gpu = gpu();
    let mut state = common::playing_state(160, 42);
    state.screen = Screen::Help;
    let mut gui = window(state);
    let rows = gui.state().unwrap().fit.help_rows;
    // 800 points: the page over the bar, in lines of 26.
    assert!((20..=30).contains(&rows), "{rows}");
    let mut sim = Simulator::with_size(jodeldiplom::gui::iced_settings(), SIZE, gui.view());
    sim.point_at(Point::new(SIZE.width / 2.0, SIZE.height / 2.0));
    sim.scroll(ScrollDelta::Lines { x: 0.0, y: -1.0 });
    let messages: Vec<Message> = sim.into_messages().collect();
    assert!(matches!(messages[..], [Message::Wheel(_)]), "{messages:?}");
    for message in messages {
        let _ = gui.update(message);
    }
    assert_eq!(gui.state().unwrap().help_scroll, 3);
}

/// The settings follow the keys: in a window too short for every row,
/// down to the last row, the cards scroll so it shows whole (as the
/// window does after each key: the operation run on the view, its second
/// pass where it asks for one); back to the first, they scroll back.
#[test]
fn the_settings_keep_the_row_the_keys_are_on_in_view() {
    use iced::advanced::widget::operation::Outcome;
    use iced::advanced::widget::Operation;
    use jodeldiplom::gui::widgets::Reveal;
    let _gpu = gpu();
    let small = Size::new(1000.0, 420.0);
    let mut state = common::playing_state(160, 42);
    state.screen = Screen::Settings;
    let mut gui = window_sized(player(), state, small);
    let last = jodeldiplom::core::SettingsRow::ALL.len() - 1;
    for _ in 0..last {
        press(&mut gui, Key::Named(Named::ArrowDown));
    }
    assert_eq!(gui.state().unwrap().settings.row, last);
    let mut sim = Simulator::with_size(jodeldiplom::gui::iced_settings(), small, gui.view());
    // Whether the row `i` shows whole.
    let whole = |sim: &mut Simulator<'_, Message>, i: usize| {
        let row = sim.find(Id::from(format!("setting-{i}"))).unwrap();
        row.visible_bounds()
            .is_some_and(|v| (v.height - row.bounds().height).abs() < 0.5)
    };
    let run = |sim: &mut Simulator<'_, Message>, mut reveal: Reveal| {
        sim.operate(&mut reveal);
        if let Outcome::Chain(mut next) = Operation::<()>::finish(&reveal) {
            sim.operate(next.as_mut());
        }
        // The scroll takes effect at the next redraw, as in the window.
        let redraw = window::Event::RedrawRequested(Instant::now());
        sim.simulate([iced::Event::Window(redraw)]);
    };
    assert!(!whole(&mut sim, last), "the window is too short already");
    run(
        &mut sim,
        gui.in_view().expect("the settings follow the keys"),
    );
    assert!(whole(&mut sim, last), "the last row is out of view");
    assert!(!whole(&mut sim, 0), "scrolled: the first row went up");
    run(&mut sim, Reveal::new("settings", "setting-0"));
    assert!(whole(&mut sim, 0), "the first row is out of view");
}
