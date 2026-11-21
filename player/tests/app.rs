//! The whole player headless: the engine plays into a null output on a
//! real-time clock (no device), driven by the same actions the keys and
//! the mouse produce.

mod common;

use std::sync::Arc;
use std::time::Duration;

use jodel_output::SimClock;
use jodel_playback::{Headroom, PlayState, Stage};
use jodeldiplom::app::{App, AppOptions, Output};
use jodeldiplom::config::Config;
use jodeldiplom::core::keymap::{lookup, Key, KeyCode, Meaning, Op};
use jodeldiplom::core::text;
use jodeldiplom::core::{Action, Effect, Screen};

const WAIT: Duration = Duration::from_secs(30);

fn options(tree: &common::Tree) -> AppOptions {
    let mut options = AppOptions::new(tree.album.clone());
    options.output = Output::Null(SimClock::Realtime);
    options.size = (120, 36);
    options.devices = Arc::new(|| Ok(Vec::new()));
    options
}

fn press(app: &mut App, key: Key) {
    match lookup(&app.state, key) {
        Some(Meaning::Op(op)) => app.dispatch(Action::Op(op)),
        Some(Meaning::Type(c)) => app.dispatch(Action::Type(c)),
        None => panic!("{key:?} does nothing"),
    }
}

/// `alt+enter`: play with the folder, in place of the queue.
fn alt_enter() -> Key {
    Key {
        code: KeyCode::Enter,
        ctrl: false,
        alt: true,
        shift: false,
    }
}

fn path_text(app: &App) -> String {
    text::signal_path(&app.state)
        .unwrap_or_default()
        .iter()
        .map(|s| s.text.clone())
        .collect()
}

#[test]
fn alt_enter_plays_the_folder_through_the_null_output() {
    let tree = common::music_tree("play");
    let mut app = App::new(options(&tree));
    assert!(common::wait_for(&mut app, WAIT, |a| !a
        .state
        .browser
        .loading));
    // Down to the first file: enter queues it, and nothing plays.
    press(&mut app, Key::plain(KeyCode::Down));
    press(&mut app, Key::plain(KeyCode::Enter));
    assert!(common::wait_for(&mut app, WAIT, |a| a
        .state
        .queue
        .paths
        .len()
        == 1));
    assert!(app.state.playing.track.is_none());
    // alt+enter: the folder in place of the queue, playing from the file.
    press(&mut app, alt_enter());
    assert!(common::wait_for(&mut app, WAIT, |a| a
        .state
        .playing
        .track
        .is_some()));
    assert_eq!(app.state.queue.paths.len(), 4);
    assert_eq!(app.state.queue.current, Some(0));
    let track = app.state.playing.track.clone().unwrap();
    assert_eq!(track.route.stage, Stage::Passthrough);
    assert_eq!(
        path_text(&app),
        "file 44.1 → null output (not exclusive), not bit-perfect"
    );
    // The cover is the folder's cover.png.
    assert!(common::wait_for(&mut app, WAIT, |a| a.art().is_some()));
    assert_eq!(app.art().unwrap().image.width(), 300);
    // The spectrum shows the output rate.
    assert_eq!(app.spectrum().rate, 44_100);
    // The fixtures are short: the next track (48 kHz) follows by itself,
    // then the WAV plays as plain PCM at its own rate.
    assert!(common::wait_for(&mut app, WAIT, |a| a
        .state
        .playing
        .track
        .as_ref()
        .is_some_and(|t| t.index == 1)));
    assert!(path_text(&app).starts_with("file 48 → null output"));
    press(&mut app, Key::ch('N'));
    assert!(common::wait_for(&mut app, WAIT, |a| a
        .state
        .playing
        .track
        .as_ref()
        .is_some_and(|t| t.index >= 2)));
    if app.state.playing.track.as_ref().unwrap().index == 2 {
        assert!(path_text(&app).starts_with("file 96 → null output"));
        // Pause and resume with space.
        press(&mut app, Key::ch(' '));
        assert!(common::wait_for(&mut app, WAIT, |a| a.state.playing.state
            == PlayState::Paused));
        press(&mut app, Key::ch(' '));
        assert!(common::wait_for(&mut app, WAIT, |a| a.state.playing.state
            == PlayState::Playing));
        // Seek forward 5 s: past the end of the 3-second WAV.
        press(&mut app, Key::plain(KeyCode::Right));
    }
    // The queue plays out and the player stops.
    assert!(common::wait_for(&mut app, WAIT, |a| a.state.playing.state
        == PlayState::Stopped));
    assert!(app.state.playing.track.is_none());
    // The spectrum starts over: no rate, so no frequency labels.
    let spectrum = app.spectrum();
    assert!(spectrum.is_empty(), "rate {}", spectrum.rate);
    app.shutdown();
    std::fs::remove_dir_all(&tree.root).unwrap();
}

/// Paused, the picture stands still: the spectrum is frozen and the app
/// asks for no more frames (the window stops drawing); playing again, it
/// moves.
#[test]
fn paused_the_window_has_nothing_to_redraw() {
    let tree = common::music_tree("pause");
    let mut app = App::new(options(&tree));
    assert!(common::wait_for(&mut app, WAIT, |a| !a
        .state
        .browser
        .loading));
    press(&mut app, Key::plain(KeyCode::Down));
    press(&mut app, alt_enter());
    assert!(common::wait_for(&mut app, WAIT, |a| a
        .state
        .playing
        .position
        > 0.2));
    app.spectrum();
    assert!(app.animating(), "playing");
    app.dispatch(Action::Op(Op::PlayPause));
    assert!(common::wait_for(&mut app, WAIT, |a| a.state.playing.state
        == PlayState::Paused));
    // A frame or two to see that it no longer changes.
    app.spectrum();
    std::thread::sleep(Duration::from_millis(50));
    app.spectrum();
    assert!(!app.animating(), "paused, nothing moves");
    app.dispatch(Action::Op(Op::PlayPause));
    assert!(common::wait_for(&mut app, WAIT, |a| a.state.playing.state
        == PlayState::Playing));
    assert!(app.animating(), "playing again");
    app.shutdown();
    let _ = std::fs::remove_dir_all(&tree.root);
}

#[test]
fn a_file_argument_plays_at_once_and_settings_are_saved() {
    let tree = common::music_tree("autoplay");
    let config_file = tree.root.join("settings").join("config.toml");
    std::fs::create_dir_all(config_file.parent().unwrap()).unwrap();
    std::fs::write(&config_file, "max_rate = 192000\n").unwrap();
    let mut options = options(&tree);
    options.autoplay = Some(tree.album.join("02 Ramp 48.flac"));
    options.config_file = Some(config_file.clone());
    options.saved = Config::load(&config_file).unwrap().0;
    // Microphones made up, fed by nothing.
    options.mic = Arc::new(|_| Ok(jodel_output::input::Input::feed(48_000, 240).0));
    options.mics = Arc::new(|| Ok(vec![("a".into(), "Mic A".into())]));
    // A command-line override for this run only.
    options.config = Config {
        headroom: Headroom(10),
        ..options.saved.clone()
    };
    let mut app = App::new(options);
    assert!(common::wait_for(&mut app, WAIT, |a| a
        .state
        .playing
        .track
        .is_some()));
    let track = app.state.playing.track.clone().unwrap();
    // The whole folder is queued, from the file given.
    assert_eq!(track.index, 1);
    assert!(track.info.path.ends_with("02 Ramp 48.flac"));
    // At its own rate, under the cap.
    assert_eq!(track.route.output_rate, 48_000);
    assert_eq!(app.state.queue.paths.len(), 4);
    // Settings: the rate cap down by one step is saved; the headroom from
    // the command line is not.
    app.dispatch(Action::Op(Op::Settings));
    assert_eq!(app.state.screen, Screen::Settings);
    app.dispatch(Action::SettingsClick(1, 1));
    assert_eq!(app.state.config.max_rate, Some(176_400));
    let (saved, _) = Config::load(&config_file).unwrap();
    assert_eq!(saved.max_rate, Some(176_400));
    assert_eq!(saved.headroom, Headroom::NONE);
    // Every setting changed in the player is saved: the Transition tab's preset
    // and lyrics, the Karaoke tab's voice level.
    app.dispatch(Action::ShowSide(jodeldiplom::core::Side::Transition));
    app.dispatch(Action::Op(Op::TransitionPreset));
    app.dispatch(Action::Op(Op::TransitionLyrics));
    app.dispatch(Action::ShowSide(jodeldiplom::core::Side::Karaoke));
    app.dispatch(Action::Op(Op::SliderUp));
    // The microphone chosen (`i`).
    app.dispatch(Action::Op(Op::MicNext));
    assert!(common::wait_for(&mut app, WAIT, |a| a
        .state
        .config
        .karaoke_mic_device
        .is_some()));
    let (saved, _) = Config::load(&config_file).unwrap();
    assert_eq!(saved.transition_preset, app.state.config.transition_preset);
    assert_ne!(saved.transition_preset, Config::default().transition_preset);
    assert!(!saved.transition_lyrics);
    assert_eq!(saved.karaoke_voice, 5);
    assert_eq!(saved.karaoke_mic_device.as_deref(), Some("a"));
    assert_eq!(saved.headroom, Headroom::NONE);
    app.dispatch(Action::ShowSide(jodeldiplom::core::Side::Browser));
    // Choosing a device under --null keeps the null output.
    app.dispatch(Action::Devices(Ok(Vec::new())));
    app.dispatch(Action::SettingsClick(0, 0));
    app.dispatch(Action::ChooseDevice(0));
    assert!(app.state.null_output);
    app.run(Effect::SwitchDevice(Some("anything".into())));
    assert!(app.tap().is_some(), "the null engine still runs");
    app.dispatch(Action::Op(Op::Quit));
    assert!(app.state.quit);
    app.shutdown();
    std::fs::remove_dir_all(&tree.root).unwrap();
}

/// The session: the queue, the track in it and the browser's folder and
/// row are saved on quitting and come back at the next start (nothing
/// plays); `ctrl+s` saves the queue as a playlist, `ctrl+l` shows the saved
/// queues, and alt+enter on one plays it in place of the queue.
#[test]
fn the_session_and_saved_queues_come_back() {
    let tree = common::music_tree("session");
    let keep = tree.album.parent().unwrap().join("kept");
    let session = keep.join("session.m3u8");
    let queues = keep.join("queues");
    let with_session = |restore: bool| {
        let mut o = options(&tree);
        o.session = Some(session.clone());
        o.queues = Some(queues.clone());
        o.restore_browser = restore;
        o
    };
    let mut app = App::new(with_session(true));
    assert!(common::wait_for(&mut app, WAIT, |a| !a
        .state
        .browser
        .loading));
    // Every file queued (enter on each), the third row of the browser
    // selected.
    for _ in 0..4 {
        press(&mut app, Key::plain(KeyCode::Down));
        press(&mut app, Key::plain(KeyCode::Enter));
    }
    assert!(common::wait_for(&mut app, WAIT, |a| a
        .state
        .queue
        .paths
        .len()
        == 4));
    let queued = app.state.queue.paths.clone();
    press(&mut app, Key::plain(KeyCode::Up));
    let selected = app.state.browser.cursor;
    // Saved as a playlist.
    press(&mut app, Key::ctrl('s'));
    assert!(common::wait_for(&mut app, WAIT, |_| {
        std::fs::read_dir(&queues).is_ok_and(|mut d| d.next().is_some())
    }));
    app.shutdown();
    drop(app);
    assert!(session.is_file());
    // Back as it was, nothing playing.
    let mut app = App::new(with_session(true));
    assert!(common::wait_for(&mut app, WAIT, |a| !a
        .state
        .browser
        .loading
        && a.state.queue.paths.len() == 4));
    assert_eq!(app.state.queue.paths, queued);
    assert_eq!(app.state.browser.dir, tree.album);
    assert_eq!(app.state.browser.cursor, selected);
    assert!(app.state.playing.track.is_none());
    // The saved queues, and one played.
    press(&mut app, Key::ctrl('l'));
    assert!(common::wait_for(&mut app, WAIT, |a| a.state.browser.dir
        == queues
        && !a.state.browser.loading));
    let rows = app.state.browser.entries.len();
    assert!(rows >= 2, "the parent row and the playlist");
    press(&mut app, Key::plain(KeyCode::End));
    press(&mut app, alt_enter());
    assert!(common::wait_for(&mut app, WAIT, |a| a
        .state
        .playing
        .track
        .is_some()));
    assert_eq!(app.state.queue.paths, queued);
    app.shutdown();
    let _ = std::fs::remove_dir_all(&keep);
}

/// alt+↑ / alt+↓ move the volume of the device the engine plays on (its
/// id), not the system default looked up anew: on macOS that can name
/// another device or none while the player hogs its own.
#[test]
fn the_device_volume_moves_the_device_playing() {
    use jodeldiplom::app::volume_device;
    let default = Output::Device(None);
    let named = Output::Device(Some("USB DAC".into()));
    assert_eq!(
        volume_device(&default, Some("BuiltInSpeakerDevice")),
        Some(Some("BuiltInSpeakerDevice".into()))
    );
    assert_eq!(
        volume_device(&named, Some("AppleUSBAudioEngine:DAC")),
        Some(Some("AppleUSBAudioEngine:DAC".into()))
    );
    // Before the engine has found a device: the one asked for.
    assert_eq!(volume_device(&default, None), Some(None));
    assert_eq!(volume_device(&named, None), Some(Some("USB DAC".into())));
    // The null output has none.
    assert_eq!(
        volume_device(&Output::Null(SimClock::Realtime), Some("x")),
        None
    );
}
