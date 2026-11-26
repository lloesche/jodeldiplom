//! The screens drawn in iced's own widgets, through the window's view in
//! iced's simulator: what their buttons, keys, drags and wheel do.

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
use jodeldiplom::core::{Action, Gpu, State};
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
