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
use jodeldiplom::core::lanes::Block;
use jodeldiplom::core::sliders::Slider;
use jodeldiplom::core::{lag, update, Action, Effect, Gpu, Screen, Side, State};
use jodeldiplom::gui::metrics;
use jodeldiplom::gui::pointer::{Grab, WheelOn};
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
    let gui = Gui::headless_setup(setup);
    assert!(click(&gui, "Try again").is_err());
    assert_eq!(chosen(&click(&gui, "Quit").unwrap()), Some(Choice::Quit));

    // Stopped: Try again by click, by Enter and by r, each going on from
    // where setup stopped (the download that failed asked for again).
    let (setup, made) = start(Web::Offline, &dir.join("offline"));
    when(&setup, |s| s.failure.is_some());
    let mut gui = Gui::headless_setup(setup);
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
        let mut gui = Gui::headless_setup(setup);
        press(&mut gui, key.clone());
        press(&mut gui, Key::Named(Named::Enter));
        assert!(!downloads_come_to(&made, 3), "tried again after {key:?}");
        let _ = std::fs::remove_dir_all(dir.join("offline-keys"));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// K8: the window's first view is the setup screen, at once, saying what
/// is being done: what is slow at the start (the graphics card chosen and
/// the files looked for, here a manifest that waits until it is let go;
/// then the player's own start) goes on in the background, its steps on
/// that screen, and the player shows once it has started.
#[test]
fn the_first_view_comes_before_the_slow_start() {
    use jodeldiplom::app::Starting;
    use jodeldiplom::view::setup::LOOKING;
    let _gpu = gpu();
    let dir = common::temp_dir("slow-start");
    let mut options = AppOptions::new(dir.clone());
    options.output = Output::Null(SimClock::Realtime);
    options.devices = Arc::new(|| Ok(Vec::new()));
    options.mic = Arc::new(|_| Err("no microphone".into()));
    options.mics = Arc::new(|| Ok(Vec::new()));
    options.data = Some(jodeldiplom::dirs::Data::in_parent(&dir));
    let (go, slow) = std::sync::mpsc::channel::<()>();
    let manifest = move || {
        let _ = slow.recv();
        Vec::new()
    };
    let mut gui = Gui::headless_starting(options, manifest, Arc::new(jodeldiplom::setup::Real));
    // The setup screen at once, the slow work not done: the files looked
    // for, the player's own steps to come, no player yet.
    assert!(gui.app().is_none());
    for shown in [LOOKING, Starting::Session.title(), Starting::Device.title()] {
        assert!(simulator(&gui).find(shown).is_ok(), "{shown} not shown");
    }
    // Let go of: the player starts, and shows.
    go.send(()).unwrap();
    let started = Instant::now();
    while gui.app().is_none() {
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "the player never showed"
        );
        let _ = gui.update(Message::Wake);
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(simulator(&gui).find(LOOKING).is_err());
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

/// The main screen as iced lays it out is where the window's measures
/// say ([`metrics::main`]): the side pane, the right column beside it, the
/// voice's lane under it, at the old 120- and 80-column windows, narrower
/// and on the wide tabs; so what the core is told of the window (the
/// list's page, the desk's width) is what shows.
#[test]
fn the_main_screen_is_laid_out_where_the_measures_say() {
    let _gpu = gpu();
    let near = |a: iced::Rectangle, b: iced::Rectangle| {
        let d = [a.x - b.x, a.y - b.y, a.width - b.width, a.height - b.height];
        d.iter().all(|v| v.abs() < 0.01)
    };
    for (size, side) in [
        (Size::new(960.0, 684.0), Side::Browser),
        (Size::new(640.0, 456.0), Side::Queue),
        (Size::new(512.0, 418.0), Side::Browser),
        (SIZE, Side::Karaoke),
        (SIZE, Side::Mixer),
        (SIZE, Side::Transition),
    ] {
        let mut state = common::playing_state(160, 42);
        state.side = side;
        let gui = window_sized(player(), state, size);
        let m = metrics::main(size, side);
        let mut sim = Simulator::with_size(jodeldiplom::gui::iced_settings(), size, gui.view());
        let mut found = |id: &'static str| sim.find(Id::new(id)).ok().map(|t| t.bounds());
        let pane = found("pane").expect("the pane");
        assert!(
            near(pane, m.pane),
            "{side:?} in {size:?}: {pane:?} against {:?}",
            m.pane
        );
        match (found("right"), m.right) {
            (Some(got), Some(want)) => assert!(near(got, want), "{got:?} against {want:?}"),
            (None, None) => {}
            other => panic!("{side:?} in {size:?}: the right column {other:?}"),
        }
        match (found("voice"), m.voice) {
            (Some(got), Some(want)) => assert!(near(got, want), "{got:?} against {want:?}"),
            (None, None) => {}
            other => panic!("{side:?} in {size:?}: the voice's lane {other:?}"),
        }
        let fit = gui.state().unwrap().fit;
        if side == Side::Mixer {
            assert_eq!(fit.desk_width, pane.width);
        }
    }
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
    app.set_art(Some(image::RgbImage::from_pixel(
        64,
        64,
        image::Rgb([40, 60, 90]),
    )));
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
        app.set_art(Some(light.clone()));
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

/// `state` with the playing track titled `title` (the browser's row keeps
/// its own).
fn titled(mut state: State, title: &str) -> State {
    let track = state.playing.track.as_mut().unwrap();
    let mut info = (*track.info).clone();
    info.tags.title = Some(title.into());
    track.info = Arc::new(info);
    state
}

/// The right column under the cover: a click on the title, the artist or
/// the album goes to where it is in the browser; a click on the device
/// opens its setting; the wheel over the column scrolls the list.
#[test]
fn clicking_the_title_artist_or_album_jumps() {
    let _gpu = gpu();
    let gui = window(titled(
        common::playing_state(160, 42),
        "Morning Light (Live)",
    ));
    for (words, op) in [
        ("Morning Light (Live)", Op::GoTrack),
        ("The Ramps", Op::GoArtist),
        ("Synthetic Sessions", Op::GoAlbum),
    ] {
        let at = bounds_of(&gui, words).center();
        let sent = at_point(&gui, at, &a_click());
        assert!(
            matches!(&sent[..], [Message::Act(Action::Op(o))] if *o == op),
            "{words}: {sent:?}"
        );
    }
    let device = acts(&click(&gui, "USB Audio DAC").unwrap());
    assert!(
        matches!(
            device[..],
            [Action::OpenSettings(jodeldiplom::core::SettingsRow::Device)]
        ),
        "{device:?}"
    );
    let wheel = ScrollDelta::Lines { x: 0.0, y: -1.0 };
    let at = bounds_of(&gui, "PCM").center();
    let scrolled = at_point(
        &gui,
        at,
        &[iced::Event::Mouse(mouse::Event::WheelScrolled {
            delta: wheel,
        })],
    );
    assert!(matches!(scrolled[..], [Message::Wheel(_)]), "{scrolled:?}");
}

/// On a window too narrow for the right column, the summary under the
/// list: a click on it opens Now playing.
#[test]
fn the_summary_opens_now_playing() {
    let _gpu = gpu();
    let narrow = Size::new(512.0, 418.0);
    let state = titled(common::playing_state(64, 22), "Morning Light (Live)");
    let gui = window_sized(player(), state, narrow);
    let mut sim = Simulator::with_size(jodeldiplom::gui::iced_settings(), narrow, gui.view());
    sim.click("Morning Light (Live) — The Ramps").unwrap();
    let sent = acts(&sim.into_messages().collect::<Vec<_>>());
    assert!(matches!(sent[..], [Action::Op(Op::NowPlaying)]), "{sent:?}");
}

/// Where the Transition tab's `what` (`lanes` or `overview`) of `block`
/// is in `gui`'s view.
fn transition_area(gui: &Gui, what: &str, block: Block) -> iced::Rectangle {
    let id = jodeldiplom::gui::transition::id(what, block);
    simulator(gui).find(Id::from(id)).unwrap().bounds()
}

/// The messages a press of `button` at `at` over `gui`'s view sends,
/// each taken by the window.
fn press_at(gui: &mut Gui, at: Point, button: mouse::Button) -> Vec<Message> {
    let mut sim = simulator(gui);
    sim.point_at(at);
    sim.simulate([iced::Event::Mouse(mouse::Event::ButtonPressed(button))]);
    let messages: Vec<Message> = sim.into_messages().collect();
    for message in messages.clone() {
        let _ = gui.update(message);
    }
    messages
}

/// The pointer moved to `at`, as the window hears it.
fn moved_to(gui: &mut Gui, at: Point) {
    hear(
        gui,
        iced::Event::Mouse(mouse::Event::CursorMoved { position: at }),
    );
}

/// `button` let go of, as the window hears it.
fn let_go(gui: &mut Gui, button: mouse::Button) {
    hear(
        gui,
        iced::Event::Mouse(mouse::Event::ButtonReleased(button)),
    );
}

/// The window's state (there is a player).
fn shown(gui: &Gui) -> &State {
    gui.state().unwrap()
}

/// A press on a lane puts the cursor on the beat at or before the
/// pointer, holding the lanes; a second on that beat soon after cues its
/// bar, as enter does, and playback stays where it is. The wheel over the
/// lanes zooms that block (the tab's zoom stays).
#[test]
fn a_click_on_a_lane_puts_the_cursor_and_a_double_click_cues_there() {
    let _gpu = gpu();
    use jodeldiplom::core::transition::{tab, viewport};
    let mut gui = window(common::transition_state(160, 42));
    let lanes = transition_area(&gui, "lanes", Block::Now);
    let at = Point::new(lanes.center_x(), lanes.y + lanes.height * 0.4);
    let (start, end) = viewport::window(shown(&gui), Block::Now, shown(&gui).heard());
    let time = start + (end - start) * f64::from((at.x - lanes.x) / lanes.width);
    let position = shown(&gui).playing.position;
    // Two clicks there, one after the other.
    let mut sim = simulator(&gui);
    sim.point_at(at);
    sim.simulate(iced_test::simulator::click());
    sim.simulate(iced_test::simulator::click());
    let messages: Vec<Message> = sim.into_messages().collect();
    let [first @ Message::Grab(
        Grab::Lanes {
            block: Block::Now,
            bounds,
        },
        _,
    ), second] = &messages[..]
    else {
        panic!("{messages:?}");
    };
    assert_eq!(*bounds, lanes);
    let _ = gui.update(first.clone());
    let_go(&mut gui, mouse::Button::Left);
    let state = shown(&gui);
    let a = lanes_analysis(state, Block::Now);
    let beat = viewport::cursor(state, Block::Now, state.heard()).unwrap();
    assert!(a.beats[beat] <= time && time - a.beats[beat] < 60.0 / 128.0);
    assert!(tab::block(state, Block::Now).cue.is_none());
    // The same beat again: its bar cued, playback where it was.
    let _ = gui.update(second.clone());
    let_go(&mut gui, mouse::Button::Left);
    let state = shown(&gui);
    let cue = tab::block(state, Block::Now).cue.expect("cued");
    assert!(a.downbeats.contains(&cue.beat) && cue.beat <= beat);
    assert_eq!(state.playing.position, position);
    // The wheel over the lanes zooms that block.
    let zoom = state.transition.zoom;
    let mut sim = simulator(&gui);
    sim.point_at(at);
    sim.simulate([iced::Event::Mouse(mouse::Event::WheelScrolled {
        delta: ScrollDelta::Lines { x: 0.0, y: 1.0 },
    })]);
    let messages: Vec<Message> = sim.into_messages().collect();
    for message in messages {
        let _ = gui.update(message);
    }
    assert_eq!(viewport::zoom_of(shown(&gui), Block::Now), zoom - 1);
    assert_eq!(shown(&gui).transition.zoom, zoom);
}

/// A press on the lanes dragged across them selects whole bars, from the
/// beat pressed to the one the pointer is over, also past their edge; a
/// tremble within the beat pressed selects nothing.
#[test]
fn dragging_across_the_lanes_selects_bars() {
    let _gpu = gpu();
    use jodeldiplom::core::transition::viewport;
    let mut gui = window(common::transition_state(160, 42));
    let lanes = transition_area(&gui, "lanes", Block::Now);
    let at = Point::new(lanes.x + lanes.width * 0.3, lanes.center_y());
    press_at(&mut gui, at, mouse::Button::Left);
    moved_to(&mut gui, Point::new(at.x + 0.5, at.y + 1.0));
    let heard = shown(&gui).heard();
    assert_eq!(viewport::selection(shown(&gui), Block::Now, heard), None);
    let beat = |gui: &Gui| viewport::cursor(shown(gui), Block::Now, heard).unwrap();
    let pressed = beat(&gui);
    // A quarter of the way on (four bars at the tab's 16): bars selected
    // from the one pressed to the one there.
    moved_to(&mut gui, Point::new(at.x + lanes.width * 0.25, at.y));
    let (from, to) = viewport::selection(shown(&gui), Block::Now, heard).expect("a selection");
    let a = lanes_analysis(shown(&gui), Block::Now);
    assert!(from <= a.beats[pressed] && to > a.beats[beat(&gui)]);
    assert!(beat(&gui) > pressed);
    // Off the lanes' right edge: their last beat in view.
    moved_to(&mut gui, Point::new(SIZE.width + 50.0, 10.0));
    let (_, end) = viewport::window(shown(&gui), Block::Now, heard);
    assert!(a.beats[beat(&gui)] < end && end - a.beats[beat(&gui)] < 60.0 / 128.0 * 1.01);
    let_go(&mut gui, mouse::Button::Left);
    assert!(viewport::selection(shown(&gui), Block::Now, heard).is_some());
}

/// The secondary button drags a block's view: dragged left, later times
/// come in, by the span over the lanes' width a point; it follows the
/// pointer off the lanes too, until let go. On the whole track the span
/// shown goes with the pointer.
#[test]
fn the_secondary_button_drags_a_transition_view() {
    let _gpu = gpu();
    use jodeldiplom::core::transition::viewport;
    let mut gui = window(common::transition_state(160, 42));
    let lanes = transition_area(&gui, "lanes", Block::Now);
    let heard = shown(&gui).heard();
    let (start, end) = viewport::window(shown(&gui), Block::Now, heard);
    let per = (end - start) / f64::from(lanes.width);
    let at = Point::new(lanes.x + 400.0, lanes.center_y());
    let messages = press_at(&mut gui, at, mouse::Button::Right);
    assert!(
        matches!(
            messages[..],
            [Message::Grab(
                Grab::Pan {
                    block: Block::Now,
                    overview: false,
                    ..
                },
                _
            )]
        ),
        "{messages:?}"
    );
    moved_to(&mut gui, Point::new(at.x - 80.0, at.y));
    let (moved, _) = viewport::window(shown(&gui), Block::Now, heard);
    assert!(
        (moved - (start + 80.0 * per)).abs() < 1e-6,
        "dragged left: later"
    );
    // Off the lanes, to the window's top.
    moved_to(&mut gui, Point::new(at.x - 40.0, 2.0));
    let_go(&mut gui, mouse::Button::Right);
    assert!(shown(&gui).transition.pan.is_none());
    let (kept, _) = viewport::window(shown(&gui), Block::Now, heard);
    assert!((kept - (start + 40.0 * per)).abs() < 1e-6);
    // Moved after it was let go of: it stays.
    moved_to(&mut gui, Point::new(at.x - 300.0, at.y));
    assert_eq!(viewport::window(shown(&gui), Block::Now, heard).0, kept);
    // On the whole track, the span shown follows the pointer.
    let whole = transition_area(&gui, "overview", Block::Now);
    let at = Point::new(whole.x + 100.0, whole.center_y());
    press_at(&mut gui, at, mouse::Button::Right);
    moved_to(&mut gui, Point::new(at.x + 20.0, at.y));
    let_go(&mut gui, mouse::Button::Right);
    let (now, _) = viewport::window(shown(&gui), Block::Now, heard);
    let r = jodeldiplom::core::lanes::rhythmed(shown(&gui), Block::Now).unwrap();
    let length = r
        .waveform
        .as_ref()
        .map_or(0.0, |w| w.duration())
        .max(r.analysis.duration);
    assert!((now - (kept + 20.0 * length / f64::from(whole.width))).abs() < 1e-6);
}

/// A press on the playing track's whole track seeks there (its view
/// follows the playhead again); on the next track's, only its view goes
/// there. The wheel over the whole track scrolls the view by tenths of
/// the span.
#[test]
fn a_press_on_the_whole_track_seeks_and_the_wheel_there_scrolls() {
    let _gpu = gpu();
    use jodeldiplom::core::transition::{tab, viewport};
    let mut gui = window(common::transition_state(160, 42));
    let whole = transition_area(&gui, "overview", Block::Now);
    let r = jodeldiplom::core::lanes::rhythmed(shown(&gui), Block::Now).unwrap();
    let length = r
        .waveform
        .as_ref()
        .map_or(0.0, |w| w.duration())
        .max(r.analysis.duration);
    let at = Point::new(whole.x + whole.width * 0.25, whole.center_y());
    let messages = press_at(&mut gui, at, mouse::Button::Left);
    let [Message::Act(Action::TransitionJump { block, time })] = messages[..] else {
        panic!("{messages:?}");
    };
    assert_eq!(block, Block::Now);
    assert!((time - 0.25 * length).abs() < 1e-6);
    assert!((shown(&gui).playing.position - time).abs() < 1e-9);
    assert!(tab::block(shown(&gui), Block::Now).start.is_none());
    // The next track's: its view goes there, playback stays.
    let next = transition_area(&gui, "overview", Block::Next);
    let position = shown(&gui).playing.position;
    press_at(&mut gui, next.center(), mouse::Button::Left);
    assert_eq!(shown(&gui).playing.position, position);
    let heard = shown(&gui).heard();
    let (start, end) = viewport::window(shown(&gui), Block::Next, heard);
    assert!(start > 0.0 && end > start);
    // The wheel down over the next track's whole track: a tenth later.
    let mut sim = simulator(&gui);
    sim.point_at(next.center());
    sim.simulate([iced::Event::Mouse(mouse::Event::WheelScrolled {
        delta: ScrollDelta::Lines { x: 0.0, y: -1.0 },
    })]);
    let messages: Vec<Message> = sim.into_messages().collect();
    for message in messages {
        let _ = gui.update(message);
    }
    let (later, _) = viewport::window(shown(&gui), Block::Next, heard);
    assert!((later - (start + 0.1 * (end - start))).abs() < 1e-6);
}

/// The footer's button zooms the track selected to fit its lyrics, as `z`
/// does.
#[test]
fn the_fit_lyrics_button_is_z() {
    let _gpu = gpu();
    let gui = window(common::transition_state(160, 42));
    let clicked = acts(&click(&gui, "fit lyrics").unwrap());
    assert!(
        matches!(clicked[..], [Action::Op(Op::TransitionFitLyrics)]),
        "{clicked:?}"
    );
}

/// A block's analysis.
fn lanes_analysis(state: &State, block: Block) -> jodel_rhythm::Analysis {
    (*jodeldiplom::core::lanes::rhythmed(state, block)
        .unwrap()
        .analysis)
        .clone()
}

/// Where `slider`'s track is on the Karaoke tab.
fn track(gui: &Gui, slider: Slider) -> iced::Rectangle {
    let id = jodeldiplom::view::karaoke::track_id(slider);
    simulator(gui).find(Id::new(id)).unwrap().bounds()
}

/// `messages` taken by the window.
fn send(gui: &mut Gui, messages: Vec<Message>) {
    for message in messages {
        let _ = gui.update(message);
    }
}

/// A Karaoke slider's track takes a press (the slider there, held); the
/// pointer then moves it from anywhere in the window, and the release
/// lets go of it. The wheel over its row steps it. The button into the
/// mode is enter (with shift held, in where the song is); the switches
/// and the sync's button are their keys.
#[test]
fn the_sliders_and_buttons_take_clicks() {
    let _gpu = gpu();
    let mut gui = window(common::karaoke_tab());
    let voice = track(&gui, Slider::Voice);
    // Pressed at the track's right end: the most, held.
    let at = Point::new(voice.x + voice.width - 0.5, voice.center_y());
    let pressed = press_at(&mut gui, at, mouse::Button::Left);
    assert!(
        matches!(pressed[..], [Message::Grab(Grab::Slider(Slider::Voice, b), _)] if b == voice),
        "{pressed:?}"
    );
    assert_eq!(shown(&gui).config.karaoke_voice, 100);
    assert_eq!(shown(&gui).karaoke.dragging, Some(Slider::Voice));
    // Moved far left of the track, over another row: the least.
    moved_to(&mut gui, Point::new(4.0, 300.0));
    assert_eq!(shown(&gui).config.karaoke_voice, 0);
    let_go(&mut gui, mouse::Button::Left);
    assert_eq!(shown(&gui).karaoke.dragging, None);
    assert_eq!(shown(&gui).karaoke.focus, Slider::Voice);

    // The wheel over the delay's row, off its track: a step a notch (its
    // nudge's, 10 ms), no drag; the arrow keys move it next.
    let delay = track(&gui, Slider::Delay);
    let mut sim = simulator(&gui);
    sim.point_at(Point::new(delay.x - 20.0, delay.center_y()));
    sim.scroll(ScrollDelta::Lines { x: 0.0, y: 1.0 });
    let wheeled: Vec<Message> = sim.into_messages().collect();
    assert!(
        matches!(
            wheeled[..],
            [Message::Steps(WheelOn::Slider(Slider::Delay), _, _)]
        ),
        "{wheeled:?}"
    );
    send(&mut gui, wheeled);
    assert_eq!(Slider::Delay.value(shown(&gui)), 160);
    assert_eq!(shown(&gui).karaoke.focus, Slider::Delay);
    assert_eq!(shown(&gui).karaoke.dragging, None);

    // The button into the mode: enter, or with shift in where the song is.
    let enter = click(&gui, "Enter Karaoke Mode").unwrap();
    assert!(
        matches!(
            enter[..],
            [Message::OpShifted(Op::KaraokeEnter, Op::KaraokeEnterHere)]
        ),
        "{enter:?}"
    );
    let mut shifted = window(common::karaoke_tab());
    hear(
        &mut shifted,
        iced::Event::Keyboard(iced::keyboard::Event::ModifiersChanged(
            iced::keyboard::Modifiers::SHIFT,
        )),
    );
    send(&mut shifted, enter.clone());
    assert_eq!(shown(&shifted).screen, Screen::Karaoke);
    assert_eq!(shown(&shifted).karaoke.held, None, "in where the song is");
    send(&mut gui, enter);
    assert!(shown(&gui).karaoke.held.is_some(), "in at the song's start");

    // The switches and the sync's button.
    let gui = window(common::karaoke_tab());
    for (label, wanted) in [
        ("Game", Op::KaraokeGame),
        ("Notes", Op::KaraokeNotes),
        ("Keep", Op::KaraokeKeep),
        ("Measure", Op::KaraokeSync),
    ] {
        let done = acts(&click(&gui, label).unwrap());
        assert!(
            matches!(done[..], [Action::Op(o)] if o == wanted),
            "{label}: {done:?}"
        );
    }
}

/// K4: a press or a drag on a Karaoke slider lands on a whole percent of
/// its travel (the delay on its steps of 10 ms), whatever the pixel:
/// across the delay's track pixel by pixel every value comes by, from 0
/// to the most its nudge reaches; across the voice's, every whole percent.
#[test]
fn a_slider_under_the_pointer_lands_on_whole_percents() {
    let _gpu = gpu();
    let mut gui = window(common::karaoke_tab());
    for (slider, most, step) in [(Slider::Delay, 550, 10), (Slider::Voice, 100, 1)] {
        let t = track(&gui, slider);
        let mut seen = std::collections::BTreeSet::new();
        let at = |x: f32| Point::new(x, t.center_y());
        send(
            &mut gui,
            vec![Message::Grab(Grab::Slider(slider, t), at(t.x))],
        );
        for px in 0..=(t.width as i32) {
            moved_to(&mut gui, at(t.x + px as f32));
            seen.insert(slider.value(shown(&gui)));
        }
        let_go(&mut gui, mouse::Button::Left);
        assert_eq!(seen.first(), Some(&0), "{slider:?}");
        assert_eq!(seen.last(), Some(&most), "{slider:?}");
        assert!(seen.iter().all(|v| v % step == 0), "{slider:?}: {seen:?}");
        if slider == Slider::Voice {
            assert_eq!(seen, (0..=100).collect(), "every whole percent");
        }
        // A quarter of the way: a quarter of its range, on a step.
        send(
            &mut gui,
            vec![Message::Grab(
                Grab::Slider(slider, t),
                at(t.x + t.width / 4.0),
            )],
        );
        let_go(&mut gui, mouse::Button::Left);
        let span = slider.most() - slider.least();
        let quarter = slider.least() + (span / 4 / slider.unit()) * slider.unit();
        assert!(
            (slider.value(shown(&gui)) - quarter).abs() <= slider.unit(),
            "{slider:?}"
        );
    }
    // A quarter of the delay's range, 150 ms here, is no nudge.
    assert_eq!(Slider::Delay.value(shown(&gui)), 150);
    let t = track(&gui, Slider::Delay);
    send(
        &mut gui,
        vec![Message::Grab(
            Grab::Slider(Slider::Delay, t),
            Point::new(t.x + t.width / 4.0, t.center_y()),
        )],
    );
    let_go(&mut gui, mouse::Button::Left);
    assert_eq!(Slider::Delay.value(shown(&gui)), 150);
    assert_eq!(lag::nudge_ms(shown(&gui)), 0);
}

/// K18: while a sync shows, only its panel's buttons take a press:
/// cancel at once, keep and again once it has a result. Nothing under it
/// does, the tab's switches, its tabs and the bottom bar too.
#[test]
fn the_panels_buttons_take_clicks_and_nothing_else_does() {
    use jodeldiplom::core::sync::{outcome, Session, CLICKS};
    let _gpu = gpu();
    let mut state = common::karaoke_tab();
    state.karaoke.mic.name = Some("Headset".into());
    lag::set_pair(&mut state, "out", "headset");
    state.karaoke.sync = Some(Session::new(("out".into(), "headset".into())));
    let gui = window(state.clone());
    let has = |gui: &Gui, id: &'static str| simulator(gui).find(Id::new(id)).is_ok();
    assert!(has(&gui, "sync-cancel"));
    assert!(!has(&gui, "sync-keep") && !has(&gui, "sync-again"));
    let done = acts(&click(&gui, Id::new("sync-cancel")).unwrap());
    assert!(matches!(done[..], [Action::Op(Op::Back)]), "{done:?}");
    // The Keep switch under the panel's shade takes none.
    assert!(click(&gui, "Keep").unwrap().is_empty());
    // Elsewhere: a slider's track, the tabs, the bar's buttons.
    for at in [
        Point::new(700.0, 200.0),
        Point::new(40.0, 8.0),
        Point::new(SIZE.width - 60.0, SIZE.height - 20.0),
        Point::new(SIZE.width / 2.0, SIZE.height - 20.0),
    ] {
        let mut sim = simulator(&gui);
        sim.point_at(at);
        sim.simulate(iced_test::simulator::click());
        let pressed: Vec<Message> = sim.into_messages().collect();
        assert!(pressed.is_empty(), "a press at {at:?}: {pressed:?}");
    }

    // Every click answered: the delay measured, kept by a press.
    let mut session = Session::new(("out".into(), "headset".into()));
    session.played = CLICKS;
    session.answers = vec![Some(0.14); CLICKS];
    session.outcome = Some(outcome(&session.answers));
    state.karaoke.sync = Some(session);
    let mut gui = window(state);
    for id in ["sync-keep", "sync-again", "sync-cancel"] {
        assert!(has(&gui, id), "{id}");
    }
    let keep = click(&gui, Id::new("sync-keep")).unwrap();
    assert!(
        matches!(acts(&keep)[..], [Action::Op(Op::SyncKeep)]),
        "{keep:?}"
    );
    assert!(shown(&gui).karaoke.sync.is_some());
    send(&mut gui, keep);
    // Kept (for the pair the sync measured: `sync.rs`), the panel gone.
    assert!(shown(&gui).karaoke.sync.is_none());
}

/// The Karaoke tab, every song's words timed: one every second from 5 s,
/// each sung for 0.3 s.
fn lyrics_state() -> State {
    let mut state = common::karaoke_tab();
    common::sung_every(&mut state, 1.0);
    state
}

/// `key` pressed in the window with `modifiers` held.
fn press_held(gui: &mut Gui, key: Key, modifiers: iced::keyboard::Modifiers) {
    let mut event = press_key(key, None);
    if let iced::Event::Keyboard(iced::keyboard::Event::KeyPressed { modifiers: m, .. }) =
        &mut event
    {
        *m = modifiers;
    }
    let _ = gui.update(Message::Event(event, window::Id::unique()));
}

/// The lyrics editor's word at `line`, `word`.
fn lyric_word(gui: &Gui, line: usize, word: usize) -> jodel_lyrics::Word {
    shown(gui)
        .karaoke
        .editor
        .as_ref()
        .expect("the editor")
        .lyrics
        .lines[line]
        .words[word]
        .clone()
}

/// Where the lyrics editor's `part` of the word at `line`, `word` takes a
/// press.
fn lyric_part(
    gui: &Gui,
    line: usize,
    word: usize,
    part: jodeldiplom::core::lyric_edit::screen::Part,
) -> iced::Rectangle {
    use jodeldiplom::core::lyric_edit::WordAt;
    let id = jodeldiplom::gui::lyrics::part_id(WordAt { line, word }, part);
    simulator(gui).find(Id::from(id)).unwrap().bounds()
}

/// K27: the lyrics editor opens from the Karaoke tab by its button, as `e`
/// does, on its screen with the words over the voice; `esc` closes it,
/// back to the tab, Done as well.
#[test]
fn the_lyrics_editor_opens_from_the_karaoke_tab_and_closes() {
    let _gpu = gpu();
    let mut gui = window(lyrics_state());
    let pressed = click(&gui, "Edit lyrics").unwrap();
    assert!(
        matches!(acts(&pressed)[..], [Action::Op(Op::LyricsEditor)]),
        "{pressed:?}"
    );
    send(&mut gui, pressed);
    let _ = gui.update(Message::Frame(Instant::now()));
    assert_eq!(shown(&gui).screen, Screen::Lyrics);
    // The words in view as boxes, the song's title over them.
    assert!(shows(&gui, "Mister") && shows(&gui, "rock"));
    assert!(shows(&gui, "Morning Light"));
    assert!(simulator(&gui)
        .find(Id::new(jodeldiplom::gui::lyrics::LANE))
        .is_ok());
    press(&mut gui, Key::Named(Named::Escape));
    assert_eq!(
        (shown(&gui).screen, shown(&gui).side),
        (Screen::Main, Side::Karaoke)
    );
    assert!(shown(&gui).karaoke.editor.is_none());
    press(&mut gui, Key::Character("e".into()));
    assert_eq!(shown(&gui).screen, Screen::Lyrics);
    let done = click(&gui, "Done").unwrap();
    send(&mut gui, done);
    assert_eq!(shown(&gui).screen, Screen::Main);
}

/// K27: a click on a word selects it; its end dragged follows the pointer
/// (also past the lane), the word set by hand, one step of the history
/// once let go of; a tremble on a press changes nothing.
#[test]
fn dragging_a_words_end_moves_it_and_sets_it_by_hand() {
    use jodeldiplom::core::lyric_edit::screen::{window as shown_span, Part};
    let _gpu = gpu();
    let mut gui = window(lyrics_state());
    press(&mut gui, Key::Character("e".into()));
    let _ = gui.update(Message::Frame(Instant::now()));
    let lane = simulator(&gui)
        .find(Id::new(jodeldiplom::gui::lyrics::LANE))
        .unwrap()
        .bounds();
    let (start, end) = shown_span(shown(&gui)).unwrap();
    let per = (end - start) / f64::from(lane.width);
    let before = lyric_word(&gui, 0, 1);
    // A press on its middle, trembling a point: selected, nothing moved.
    let middle = lyric_part(&gui, 0, 1, Part::Middle);
    let pressed = press_at(&mut gui, middle.center(), mouse::Button::Left);
    assert!(
        matches!(
            pressed[..],
            [Message::Grab(
                Grab::LyricWord {
                    part: Part::Middle,
                    ..
                },
                _
            )]
        ),
        "{pressed:?}"
    );
    moved_to(
        &mut gui,
        Point::new(middle.center_x() + 1.0, middle.center_y()),
    );
    let_go(&mut gui, mouse::Button::Left);
    let editor = shown(&gui).karaoke.editor.as_ref().unwrap();
    assert_eq!(editor.selected.map(|w| (w.line, w.word)), Some((0, 1)));
    assert_eq!(lyric_word(&gui, 0, 1), before);
    assert!(!editor.can_undo());
    // Its end, 30 points on: there, set by hand, its start where it was.
    let edge = lyric_part(&gui, 0, 1, Part::End);
    press_at(&mut gui, edge.center(), mouse::Button::Left);
    moved_to(&mut gui, Point::new(edge.center_x() + 30.0, 300.0));
    let moved = lyric_word(&gui, 0, 1);
    let to = before.end.unwrap() + 30.0 * per;
    assert!((moved.end.unwrap() - to).abs() < 2.0 * per, "{moved:?}");
    assert_eq!(moved.start, before.start);
    let_go(&mut gui, mouse::Button::Left);
    let after = lyric_word(&gui, 0, 1);
    assert!(after.pinned);
    assert!(shown(&gui).karaoke.editor.as_ref().unwrap().can_undo());
    // Undone with ctrl+z, as it was.
    press_held(
        &mut gui,
        Key::Character("z".into()),
        iced::keyboard::Modifiers::CTRL,
    );
    assert_eq!(lyric_word(&gui, 0, 1), before);
    // The wheel over the lane zooms the view round the pointer.
    let mut sim = simulator(&gui);
    sim.point_at(Point::new(lane.center_x(), lane.center_y()));
    sim.scroll(ScrollDelta::Lines { x: 0.0, y: 1.0 });
    let wheeled: Vec<Message> = sim.into_messages().collect();
    send(&mut gui, wheeled);
    let (a, b) = shown_span(shown(&gui)).unwrap();
    assert!(b - a < end - start);
}

/// K27: the editor's keys: → selects the next word, shift+→ its start
/// later, alt+← its end earlier, `n` marks it not sung (dimmed, said
/// so), `+` the voice louder over it, ctrl+z and ctrl+y undo and redo.
#[test]
fn the_lyrics_editors_keys_nudge_mark_and_undo() {
    use iced::keyboard::Modifiers;
    use jodeldiplom::core::lyric_edit::screen::NUDGE;
    let _gpu = gpu();
    let mut gui = window(lyrics_state());
    press(&mut gui, Key::Character("e".into()));
    let _ = gui.update(Message::Frame(Instant::now()));
    let right = || Key::Named(Named::ArrowRight);
    press(&mut gui, right());
    let before = lyric_word(&gui, 0, 1);
    press_held(&mut gui, right(), Modifiers::SHIFT);
    press_held(&mut gui, Key::Named(Named::ArrowLeft), Modifiers::ALT);
    let w = lyric_word(&gui, 0, 1);
    assert!((w.start.unwrap() - before.start.unwrap() - NUDGE).abs() < 1e-9);
    assert!((before.end.unwrap() - w.end.unwrap() - NUDGE).abs() < 1e-9);
    assert!(w.pinned);
    press(&mut gui, Key::Character("n".into()));
    assert!(lyric_word(&gui, 0, 1).not_sung);
    assert!(shows(&gui, "not sung"));
    press(&mut gui, Key::Character("+".into()));
    assert!(lyric_word(&gui, 0, 1).voice_db.is_some());
    press_held(&mut gui, Key::Character("z".into()), Modifiers::CTRL);
    assert_eq!(lyric_word(&gui, 0, 1).voice_db, None);
    press_held(&mut gui, Key::Character("y".into()), Modifiers::CTRL);
    assert!(lyric_word(&gui, 0, 1).voice_db.is_some());
    // The buttons do what their keys do.
    let louder = acts(&click(&gui, "Louder").unwrap());
    assert!(
        matches!(louder[..], [Action::Op(Op::LyricsLouder)]),
        "{louder:?}"
    );
}

/// K27: enter writes the word selected in a field: what is typed there
/// kept with enter; esc leaves it as it was; the keys meanwhile type, they
/// do not edit.
#[test]
fn a_word_is_written_in_the_lyrics_editors_field() {
    let _gpu = gpu();
    let mut gui = window(lyrics_state());
    press(&mut gui, Key::Character("e".into()));
    press(&mut gui, Key::Named(Named::ArrowRight));
    press(&mut gui, Key::Named(Named::Enter));
    let _ = gui.update(Message::Frame(Instant::now()));
    let field = Id::new(jodeldiplom::gui::lyrics::FIELD);
    let mut sim = simulator(&gui);
    let at = sim.find(field.clone()).unwrap().bounds();
    sim.point_at(Point::new(at.x + at.width - 4.0, at.center_y()));
    sim.simulate(iced_test::simulator::click());
    sim.typewrite("s");
    let typed: Vec<Message> = sim.into_messages().collect();
    send(&mut gui, typed);
    // `n` while writing types: the word is not marked.
    press(&mut gui, Key::Character("n".into()));
    assert!(!lyric_word(&gui, 0, 1).not_sung);
    press(&mut gui, Key::Named(Named::Enter));
    assert_eq!(lyric_word(&gui, 0, 1).text, "rocks");
    // Written again and left with esc: as it was, the editor still open.
    press(&mut gui, Key::Named(Named::Enter));
    let _ = gui.update(Message::Frame(Instant::now()));
    send(
        &mut gui,
        vec![Message::Act(Action::LyricsEdit(
            jodeldiplom::core::lyric_edit::LyricsEdit::Screen(
                jodeldiplom::core::lyric_edit::screen::Input::Draft("roll".into()),
            ),
        ))],
    );
    press(&mut gui, Key::Named(Named::Escape));
    assert_eq!(lyric_word(&gui, 0, 1).text, "rocks");
    assert_eq!(shown(&gui).screen, Screen::Lyrics);
}

/// K27: space plays the song from just before the word selected, the view
/// following the playhead a page at a time; again, it pauses.
#[test]
fn space_plays_from_the_word_and_the_view_follows() {
    use jodel_playback::{Command, PlayState};
    use jodeldiplom::core::lyric_edit::screen::{window as shown_span, PAGE, PREROLL};
    let mut state = lyrics_state();
    state.playing.state = PlayState::Paused;
    update(&mut state, Action::Op(Op::LyricsEditor));
    let fx = update(&mut state, Action::Op(Op::LyricsPlay));
    let from = 5.0 - PREROLL;
    assert!(fx.contains(&Effect::Engine(Command::Seek(from))), "{fx:?}");
    assert!(fx.contains(&Effect::Engine(Command::Resume)));
    state.playing.state = PlayState::Playing;
    let (start, end) = shown_span(&state).unwrap();
    let span = end - start;
    state.drawn_heard = Some(start + 2.5 * span);
    let (a, b) = shown_span(&state).unwrap();
    let heard = state.heard();
    assert!(heard >= a && heard < a + PAGE * span && (b - a - span).abs() < 1e-9);
    let fx = update(&mut state, Action::Op(Op::LyricsPlay));
    assert!(fx.contains(&Effect::Engine(Command::Pause)));
}
/// A player with no output, no devices and no microphone, timing by
/// `clock`.
fn mode_player(clock: &jodeldiplom::app::AppClock) -> App {
    let dir = common::temp_dir("controls-window");
    let mut options = AppOptions::new(dir);
    options.output = Output::Null(SimClock::Realtime);
    options.devices = Arc::new(|| Ok(Vec::new()));
    options.mic = Arc::new(|_| Err("no microphone".into()));
    options.mics = Arc::new(|| Ok(Vec::new()));
    options.clock = clock.clone();
    App::new(options)
}

/// A window of [`SIZE`] at scale 1 showing `state`, a frame drawn; its
/// player's clock stepped by hand.
fn mode_window(state: State) -> (Gui, jodeldiplom::app::AppClock) {
    let clock = jodeldiplom::app::AppClock::stepped();
    let mut gui = Gui::headless(mode_player(&clock));
    hear(&mut gui, iced::Event::Window(window::Event::Resized(SIZE)));
    hear(&mut gui, iced::Event::Window(window::Event::Rescaled(1.0)));
    mode_shown((&mut gui, &clock), state);
    (gui, clock)
}

/// `gui` showing `state`, a frame drawn, its player's `clock` moved on to
/// the state's time (the core's clock is the player's).
fn mode_shown((gui, clock): (&mut Gui, &jodeldiplom::app::AppClock), state: State) {
    let at = gui.state().map_or(0, |s| s.drawn_ms.unwrap_or(0));
    clock.step(std::time::Duration::from_millis(
        state.now_ms.saturating_sub(at),
    ));
    gui.show(state);
    let _ = gui.update(Message::Frame(Instant::now()));
}

/// The messages `events` send with the pointer at `at` in `gui`'s view.
fn mode_sent(gui: &Gui, at: Point, events: Vec<iced::Event>) -> Vec<Message> {
    let mut sim = simulator(gui);
    sim.point_at(at);
    sim.simulate(events);
    sim.into_messages().collect()
}

/// The pointer moving to `at`.
fn mode_moved(at: Point) -> Vec<iced::Event> {
    vec![iced::Event::Mouse(mouse::Event::CursorMoved {
        position: at,
    })]
}

/// Where the widget `id` is in `gui`'s view, if it shows.
fn mode_bounds(gui: &Gui, id: &'static str) -> Option<iced::Rectangle> {
    simulator(gui).find(Id::new(id)).ok().map(|f| f.bounds())
}

/// The core's actions among `messages`, the pointer's moves left out.
fn mode_acts(messages: &[Message]) -> Vec<Action> {
    messages
        .iter()
        .filter_map(|m| match m {
            Message::Act(Action::PointerMoved { .. }) => None,
            Message::Act(action) => Some(action.clone()),
            _ => None,
        })
        .collect()
}

/// Where the pointer is heard to be among `messages`: its share of the
/// window's height from the bottom.
fn heard_at(messages: &[Message]) -> Option<f32> {
    messages.iter().find_map(|m| match m {
        Message::Act(Action::PointerMoved { from_bottom }) => Some(*from_bottom),
        _ => None,
    })
}

/// In the mode the controls come up only with the pointer near the bottom
/// (staying while it is there; going 1.5 s after it leaves, the window
/// too), fading in and out, over the picture: the lines stay where they
/// are. The window hears every move of the pointer over the mode, a
/// button held or not, as a share of its height. In the bar the seek bar
/// seeks, except while the song is held for START; the transport, the
/// voice slider and the way out do what their keys do. They have no
/// START: that is the stage's pill.
#[test]
fn the_controls_come_up_near_the_bottom_over_the_lines_with_a_seek_bar() {
    use jodeldiplom::view::karaoke::picture;
    let _gpu = gpu();
    use jodel_playback::PlayState;
    use jodeldiplom::core::karaoke::controls::{self, FADE_MS, LINGER_MS};
    let mut state = common::karaoke_tab();
    update(&mut state, Action::Op(Op::KaraokeEnter));
    // A song playing (nothing waits for START), the controls gone.
    state.karaoke.held = None;
    state.karaoke.countdown = None;
    state.playing.track = Some(common::now_track(44_100, state.config.settings(), 2));
    state.playing.state = PlayState::Playing;
    state.playing.position = 30.0;
    let later = state.now_ms + 10_000;
    update(&mut state, Action::Tick(later));
    assert!(!controls::shown(&state));
    let (mut gui, clock) = mode_window(state.clone());
    assert_eq!(mode_bounds(&gui, "karaoke-seek"), None);
    // The controls reach as far up as the window says: their bar and two
    // lines, more than the least share.
    let reach = gui.state().unwrap().fit.controls_reach;
    assert!(reach > 0.15 && reach < 0.25, "{reach}");
    state.fit.controls_reach = reach;
    let hidden = picture(&state, 0.12, 30.0);
    // Above the bottom: heard there, nothing comes up.
    let middle = mode_sent(
        &gui,
        Point::new(640.0, 400.0),
        mode_moved(Point::new(640.0, 400.0)),
    );
    let at = heard_at(&middle).expect("a move heard");
    assert!((at - 0.5).abs() < 0.01, "{at}");
    update(&mut state, Action::PointerMoved { from_bottom: at });
    assert!(!controls::shown(&state) && !controls::fading(&state));
    // A button held, the move is heard all the same.
    let mut sim = simulator(&gui);
    sim.point_at(Point::new(640.0, 400.0));
    sim.simulate([iced::Event::Mouse(mouse::Event::ButtonPressed(
        mouse::Button::Right,
    ))]);
    let to = Point::new(10.0, 300.0);
    sim.point_at(to);
    sim.simulate(mode_moved(to));
    let dragged: Vec<Message> = sim.into_messages().collect();
    let last = dragged.iter().rev().find_map(|m| match m {
        Message::Act(Action::PointerMoved { from_bottom }) => Some(*from_bottom),
        _ => None,
    });
    assert!(
        last.is_some_and(|f| (f - 0.625).abs() < 0.01),
        "{dragged:?}"
    );
    // Near it: the controls fading in over the bottom (laid out, and taking
    // clicks, as soon as they show at all); the lines where they were.
    let low = Point::new(640.0, SIZE.height - 10.0);
    let near = heard_at(&mode_sent(&gui, low, mode_moved(low))).expect("a move heard");
    update(&mut state, Action::PointerMoved { from_bottom: near });
    assert!(!controls::shown(&state) && controls::fading(&state));
    let at = state.now_ms + FADE_MS / 2;
    update(&mut state, Action::Tick(at));
    let shown_now = picture(&state, 0.12, 30.0);
    assert_eq!(shown_now.controls, Some(0.5));
    assert_eq!(shown_now.start, None);
    assert_eq!(hidden.lines, shown_now.lines);
    let faded_in = state.now_ms + FADE_MS / 2;
    update(&mut state, Action::Tick(faded_in));
    assert_eq!(picture(&state, 0.12, 30.0).controls, Some(1.0));
    // They stay while the pointer rests there.
    update(&mut state, Action::Tick(faded_in + 60_000));
    assert!(controls::shown(&state) && !controls::fading(&state));
    mode_shown((&mut gui, &clock), state.clone());
    // The bar over the window's bottom: the seek bar and the voice beside
    // it, the transport and the way out.
    let seek = mode_bounds(&gui, "karaoke-seek").expect("the seek bar");
    let voice = mode_bounds(&gui, "karaoke-voice").expect("the voice slider");
    let bottom = SIZE.height * (1.0 - reach);
    assert!(seek.y > bottom && voice.y > bottom, "{seek:?} {voice:?}");
    assert!(
        seek.width > 200.0 && voice.x > seek.x + seek.width,
        "{seek:?} {voice:?}"
    );
    for (id, op) in [
        ("previous", Op::Previous),
        ("pause", Op::PlayPause),
        ("next", Op::Next),
        ("karaoke-leave", Op::Back),
    ] {
        let mut sim = simulator(&gui);
        sim.click(Id::new(id)).unwrap();
        let messages: Vec<Message> = sim.into_messages().collect();
        assert!(
            matches!(mode_acts(&messages)[..], [Action::Op(o)] if o == op),
            "{id}: {messages:?}"
        );
    }
    // No START among them (the hint names its keys, no button).
    assert!(mode_bounds(&gui, "start").is_none() && mode_bounds(&gui, "play").is_none());
    // The voice: a press grabs its track, the wheel steps it.
    let press = vec![iced::Event::Mouse(mouse::Event::ButtonPressed(
        mouse::Button::Left,
    ))];
    let grabbed = mode_sent(&gui, voice.center(), press.clone());
    assert!(
        grabbed.iter().any(
            |m| matches!(m, Message::Grab(Grab::Slider(Slider::Voice, b), _) if b.width > 100.0)
        ),
        "{grabbed:?}"
    );
    let wheel = vec![iced::Event::Mouse(mouse::Event::WheelScrolled {
        delta: ScrollDelta::Lines { x: 0.0, y: 1.0 },
    })];
    let wheeled = mode_sent(&gui, voice.center(), wheel);
    assert!(wheeled
        .iter()
        .any(|m| matches!(m, Message::Steps(WheelOn::Slider(Slider::Voice), ..))));
    // The seek bar: a press grabs it, the release seeks there.
    let length = state.playing.duration().unwrap();
    let pressed = mode_sent(&gui, seek.center(), press.clone());
    let grab = pressed
        .into_iter()
        .find(|m| matches!(m, Message::Grab(Grab::Seek(b), _) if *b == seek))
        .expect("the seek bar grabbed");
    let _ = gui.update(grab);
    assert!(gui.state().unwrap().seek_drag.is_some());
    hear(
        &mut gui,
        iced::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)),
    );
    let after = gui.state().unwrap();
    assert_eq!(after.seek_drag, None);
    assert!((after.playing.position - length / 2.0).abs() < length / 50.0);
    // Held for START: the bar takes no press.
    let mut waiting = state.clone();
    waiting.karaoke.held = Some(2);
    mode_shown((&mut gui, &clock), waiting);
    let pressed = mode_sent(&gui, seek.center(), press);
    assert!(!pressed.iter().any(|m| matches!(m, Message::Grab(..))));
    // Back up above the bottom: they stay a while, then fade out; gone,
    // they are no longer in the view.
    update(&mut state, Action::PointerMoved { from_bottom: 0.9 });
    let left = state.now_ms;
    assert_eq!(state.due(), Some(left + LINGER_MS));
    update(&mut state, Action::Tick(left + LINGER_MS - 1));
    assert!(controls::shown(&state) && !controls::fading(&state));
    update(&mut state, Action::Tick(left + LINGER_MS + 100));
    assert!(controls::fading(&state));
    update(&mut state, Action::Tick(left + LINGER_MS + FADE_MS));
    assert!(!controls::shown(&state) && controls::due(&state).is_none());
    mode_shown((&mut gui, &clock), state.clone());
    assert_eq!(mode_bounds(&gui, "karaoke-seek"), None);
    // Off the window: heard as the pointer leaving it.
    let mut sim = simulator(&gui);
    sim.point_at(low);
    sim.simulate(mode_moved(low));
    let outside = Point::new(-20.0, -20.0);
    sim.point_at(outside);
    sim.simulate(mode_moved(outside));
    let messages: Vec<Message> = sim.into_messages().collect();
    assert!(
        messages
            .iter()
            .any(|m| matches!(m, Message::Act(Action::PointerLeft))),
        "{messages:?}"
    );
}
