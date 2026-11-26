//! The screens drawn in iced's own widgets, through the window's view in
//! iced's simulator: what their buttons and keys do.

#[path = "common/setup.rs"]
mod setup_web;

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use iced::keyboard::key::Named;
use iced::keyboard::Key;
use iced::{window, Size};
use iced_test::simulator::{press_key, Simulator};
use jodeldiplom::app::AppOptions;
use jodeldiplom::gui::window::{Gui, Message};
use jodeldiplom::view::setup::Choice;
use setup_web::{start, when, Made, Web};

/// The messages a click on the text `label` in `gui`'s view sends (an
/// error where it is not shown).
fn click(gui: &Gui, label: &str) -> Result<Vec<Message>, iced_test::Error> {
    let mut sim = Simulator::with_size(
        jodeldiplom::gui::iced_settings(),
        Size::new(1280.0, 800.0),
        gui.view(),
    );
    sim.click(label)?;
    Ok(sim.into_messages().collect())
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
