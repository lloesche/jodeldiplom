//! The screens drawn in iced's own widgets, through the window's view in
//! iced's simulator: what their buttons, rows, tabs, keys, drags and
//! wheel do.

mod common;
#[path = "common/setup.rs"]
mod setup_web;

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

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
use jodeldiplom::core::{update, Action, Effect, Gpu, Side, State};
use jodeldiplom::gui::pointer::Grab;
use jodeldiplom::gui::window::{Gui, Message};
use jodeldiplom::view::setup::Choice;
use setup_web::{start, when, Made, Web};

/// The window's size in the tests.
const SIZE: Size = Size::new(1280.0, 800.0);

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

/// A window of [`SIZE`] at scale 1 round a player with no output, showing
/// `state`, a frame drawn (the bottom bar where the cell layout leaves it).
fn window(state: State) -> Gui {
    let dir = common::temp_dir("screens-window");
    let mut options = AppOptions::new(dir);
    options.output = Output::Null(SimClock::Realtime);
    options.devices = Arc::new(|| Ok(Vec::new()));
    options.mic = Arc::new(|_| Err("no microphone".into()));
    options.mics = Arc::new(|| Ok(Vec::new()));
    let mut gui = Gui::headless(App::new(options));
    hear(&mut gui, iced::Event::Window(window::Event::Resized(SIZE)));
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
