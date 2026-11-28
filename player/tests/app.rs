//! The whole player headless: the engine plays into a null output on a
//! real-time clock (no device), driven by the same actions the keys and
//! the mouse produce.

mod common;

use std::sync::Arc;
use std::time::Duration;

use jodel_output::SimClock;
use jodel_playback::{Headroom, PlayState};
use jodeldiplom::app::{App, AppOptions, Output};
use jodeldiplom::config::Config;
use jodeldiplom::core::keymap::{lookup, Key, KeyCode, Meaning, Op};
use jodeldiplom::core::text;
use jodeldiplom::core::{Action, Effect, Screen};

const WAIT: Duration = Duration::from_secs(30);

fn options(tree: &common::Tree) -> AppOptions {
    let mut options = AppOptions::new(tree.album.clone());
    options.output = Output::Null(SimClock::Realtime);
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
    let _guard = jodel_testkit::real_time();
    let tree = common::music_tree("play");
    let mut app = common::fitted_app(options(&tree), 120, 36);
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
    // Resampled to the rate the device runs at (48 kHz).
    assert_eq!(
        path_text(&app),
        "file 44.1 → resample 48 (160/147) → null output"
    );
    // The cover is the folder's cover.png.
    assert!(common::wait_for(&mut app, WAIT, |a| a.art().is_some()));
    assert_eq!(app.art().unwrap().image.width(), 300);
    // The spectrum shows the output rate.
    assert_eq!(app.spectrum().rate, 48_000);
    // The fixtures are short: the next track (48 kHz) follows by itself,
    // as it is, then the WAV (96 kHz) resampled.
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
        assert!(path_text(&app).starts_with("file 96 → resample 48 (1/2) → null output"));
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
}

/// Paused, the picture stands still: the spectrum is frozen and the app
/// asks for no more frames (the window stops drawing); playing again, it
/// moves.
#[test]
fn paused_the_window_has_nothing_to_redraw() {
    let _guard = jodel_testkit::real_time();
    let tree = common::music_tree("pause");
    let mut app = common::fitted_app(options(&tree), 120, 36);
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
}

/// Paused, another queue row double-clicked plays at once: no more presses
/// to hear it. The same after a pause in Karaoke mode and `esc`, and with
/// `enter`. Both songs are of one format, so the output paused is kept
/// for the next.
#[test]
fn a_song_picked_while_paused_plays() {
    use jodeldiplom::core::karaoke;
    use jodeldiplom::core::Side;
    let _guard = jodel_testkit::real_time();
    let dir = jodel_testkit::TempDir::new("paused-pick");
    for name in ["a", "b"] {
        common::write_wav(&dir.join(format!("{name}.wav")), 44_100, 8.0);
        durable::write_atomic(&dir.join(format!("{name}.lrc")), b"la la la\nla la\n").unwrap();
    }
    let mut options = AppOptions::new(dir.to_path_buf());
    options.output = Output::Null(SimClock::Realtime);
    options.devices = Arc::new(|| Ok(Vec::new()));
    options.mic = Arc::new(|_| Ok(jodel_output::input::Input::feed(48_000, 240).0));
    options.mics = Arc::new(|| Ok(Vec::new()));
    options.separator = || None;
    let mut app = common::fitted_app(options, 120, 36);
    assert!(common::wait_for(&mut app, WAIT, |a| !a
        .state
        .browser
        .loading));
    let row = app
        .state
        .browser
        .entries
        .iter()
        .position(|e| e.path.ends_with("a.wav"))
        .unwrap();
    app.dispatch(Action::SelectRow(row));
    press(&mut app, alt_enter());
    assert!(common::wait_for(&mut app, WAIT, |a| a
        .state
        .queue
        .paths
        .len()
        == 2));
    // Heard: that song, playing, from past where it was.
    let heard = |index: usize| {
        move |a: &App| {
            a.state.playing.state == PlayState::Playing
                && a.state
                    .playing
                    .track
                    .as_ref()
                    .is_some_and(|t| t.index == index)
                && a.state.playing.position > 0.3
        }
    };
    let pause = |app: &mut App| {
        app.dispatch(Action::Op(Op::PlayPause));
        assert!(common::wait_for(app, WAIT, |a| a.state.playing.state == PlayState::Paused));
    };
    assert!(common::wait_for(&mut app, WAIT, heard(0)));
    // Paused, the second row double-clicked in the queue.
    pause(&mut app);
    app.dispatch(Action::ShowSide(Side::Queue));
    app.dispatch(Action::ClickRow(1));
    app.dispatch(Action::ClickRow(1));
    assert!(
        common::wait_for(&mut app, WAIT, heard(1)),
        "the double click"
    );
    // Paused again, the first row with enter.
    pause(&mut app);
    app.dispatch(Action::SelectRow(0));
    press(&mut app, Key::plain(KeyCode::Enter));
    assert!(common::wait_for(&mut app, WAIT, heard(0)), "enter");
    // In Karaoke mode: the song started (after its count-in), paused,
    // the mode left, and the second row double-clicked.
    app.dispatch(Action::ShowSide(Side::Karaoke));
    assert!(common::wait_for(&mut app, WAIT, |a| (0..2)
        .all(|i| karaoke::not_singable(&a.state, i).is_none())));
    app.dispatch(Action::Op(Op::KaraokeEnter));
    assert_eq!(app.state.screen, Screen::Karaoke);
    app.dispatch(Action::Op(Op::KaraokeNext));
    assert!(app.state.karaoke.countdown.is_some());
    assert!(
        common::wait_for(&mut app, WAIT, |a| a.state.karaoke.countdown.is_none()
            && heard(0)(a)),
        "Next"
    );
    pause(&mut app);
    app.dispatch(Action::Op(Op::Back));
    assert_eq!(app.state.screen, Screen::Main);
    app.dispatch(Action::ShowSide(Side::Queue));
    app.dispatch(Action::ClickRow(1));
    app.dispatch(Action::ClickRow(1));
    assert!(
        common::wait_for(&mut app, WAIT, heard(1)),
        "the double click after Karaoke mode"
    );
    app.shutdown();
}

/// No microphone (none opens), Karaoke mode entered where the song plays,
/// four words a second each coming up on C4: `p` held, the demo sings
/// them as the frames are drawn, each frame up to its own time (within a
/// frame of the pitch tracker's), every word hit and the song marked as the
/// demo's; let go of, nothing more is heard.
#[test]
fn the_demo_sings_with_no_microphone() {
    use jodeldiplom::core::karaoke::{self, SongLyrics};
    use jodeldiplom::core::{Rhythm, Rhythmed, Side};
    use std::time::Instant;
    let _guard = jodel_testkit::real_time();
    let dir = jodel_testkit::TempDir::new("demo-no-mic");
    common::write_wav(&dir.join("a.wav"), 44_100, 30.0);
    durable::write_atomic(&dir.join("a.lrc"), b"la la la la\n").unwrap();
    let mut options = AppOptions::new(dir.to_path_buf());
    options.output = Output::Null(SimClock::Realtime);
    // The beats by the flux alone, and of the mix: the networks' inference
    // (the beat network's, and the separator's where there is a GPU, as on
    // the macOS runners), unoptimised, takes longer than the wait on a slow
    // machine, and nothing here hears the stems.
    options.config.beat_voting = false;
    options.config.beat_method = jodel_rhythm::Method::Flux;
    options.config.stems = false;
    options.devices = Arc::new(|| Ok(Vec::new()));
    options.mic = Arc::new(|_| Err("no microphone".into()));
    options.mics = Arc::new(|| Ok(Vec::new()));
    options.separator = || None;
    let mut app = common::fitted_app(options, 120, 36);
    assert!(common::wait_for(&mut app, WAIT, |a| !a
        .state
        .browser
        .loading));
    let path = dir.join("a.wav");
    let row = app
        .state
        .browser
        .entries
        .iter()
        .position(|e| e.path == path)
        .unwrap();
    app.dispatch(Action::SelectRow(row));
    press(&mut app, alt_enter());
    let ready = |a: &App| {
        a.state.playing.state == PlayState::Playing
            && matches!(a.state.rhythm.get(&path), Some(Rhythm::Ready(_)))
            && karaoke::not_singable(&a.state, 0).is_none()
    };
    app.dispatch(Action::ShowSide(Side::Karaoke));
    assert!(
        common::wait_for(&mut app, WAIT, ready),
        "{:?} {:?} {:?} {:?}",
        app.state.playing.state,
        app.state.rhythm.get(&path).map(std::mem::discriminant),
        karaoke::not_singable(&app.state, 0),
        app.state.queue.paths
    );
    app.dispatch(Action::Op(Op::KaraokeEnterHere));
    assert_eq!(app.state.screen, Screen::Karaoke);
    assert!(app.state.karaoke.mic.name.is_none(), "no microphone");
    // The words from a second after where the song is, the melody under
    // them.
    let from = app.state.playing.position.ceil() + 1.0;
    let Some(Rhythm::Ready(r)) = app.state.rhythm.get(&path).cloned() else {
        unreachable!()
    };
    let frames = 3100;
    let mut melody = jodel_rhythm::waveform::Melody {
        cents: vec![0; frames],
        sure: vec![0; frames],
    };
    for i in (from as usize * 100)..((from as usize + 4) * 100) {
        (melody.cents[i], melody.sure[i]) = (6000, 255);
    }
    let waveform = jodel_rhythm::waveform::Waveform {
        melody: Some(melody),
        ..Default::default()
    };
    let rhythm = Rhythmed::new((*r.analysis).clone(), Some(waveform));
    app.state
        .rhythm
        .insert(path.clone(), Rhythm::Ready(Arc::new(rhythm)));
    let mut lyrics = jodel_lyrics::Lyrics::parse_text("la la la la", jodel_lyrics::Source::Lrclib);
    for (k, w) in lyrics.lines[0].words.iter_mut().enumerate() {
        (w.start, w.end) = (Some(from + k as f64), Some(from + k as f64 + 1.0));
    }
    let found = jodeldiplom::work::lyrics::Answer::Found {
        lyrics,
        timed: None,
    };
    let found = SongLyrics::from_answer(found);
    app.state.karaoke.lyrics.insert(path.clone(), found);
    press(&mut app, Key::ch('p'));
    assert!(app.state.karaoke.demo.held());
    // Drawn frame by frame through the last word; while it is held, the
    // demo has sung up to the frame's own time, within a frame of the
    // pitch tracker's (never the frame before's).
    let draw_until = |app: &mut App, t: f64| {
        let end = Instant::now() + WAIT;
        while Instant::now() < end && app.state.playing.position < t {
            app.pump_wait(Duration::from_millis(16));
            let heard = app.heard_for_frame();
            let sung = app.state.karaoke.singing.last_heard();
            if let Some(sung) = sung.filter(|_| app.state.karaoke.demo.held()) {
                let hop = jodel_rhythm::yin::HOP;
                assert!((sung - heard).abs() <= hop + 1e-9, "{sung} at {heard}");
            }
        }
    };
    draw_until(&mut app, from + 4.6);
    let singing = &app.state.karaoke.singing;
    let tally = singing.tally;
    assert!(singing.is_of(&path));
    assert_eq!((tally.hit_words, tally.scored_words), (4, 4), "{tally:?}");
    assert!(tally.demo && !singing.heard.is_empty());
    // Let go of: nothing heard after.
    app.dispatch(Action::Released(jodeldiplom::core::keymap::Op::KaraokeDemo));
    let last = app.state.karaoke.singing.last_heard();
    draw_until(&mut app, from + 5.5);
    assert_eq!(app.state.karaoke.singing.last_heard(), last);
    app.shutdown();
}

#[test]
fn a_file_argument_plays_at_once_and_settings_are_saved() {
    let _guard = jodel_testkit::real_time();
    let tree = common::music_tree("autoplay");
    let config_file = tree.root.join("settings").join("config.toml");
    durable::write_atomic(&config_file, b"bpm = true\n").unwrap();
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
    let mut app = common::fitted_app(options, 120, 40);
    assert!(common::wait_for(&mut app, WAIT, |a| a
        .state
        .playing
        .track
        .is_some()));
    let track = app.state.playing.track.clone().unwrap();
    // The whole folder is queued, from the file given.
    assert_eq!(track.index, 1);
    assert!(track.info.path.ends_with("02 Ramp 48.flac"));
    // At the rate the device runs at, as it is.
    assert_eq!(track.route.output_rate, 48_000);
    assert!(track.route.resample.is_none());
    assert_eq!(app.state.queue.paths.len(), 4);
    // Settings: the tempo turned off is saved; the headroom from the
    // command line is not.
    app.dispatch(Action::Op(Op::Settings));
    assert_eq!(app.state.screen, Screen::Settings);
    app.dispatch(Action::SettingsClick(2, 1));
    assert!(!app.state.config.bpm);
    let (saved, _) = Config::load(&config_file).unwrap();
    assert!(!saved.bpm);
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
    // The voice from off a step up: the taper's last mark, −45 dB, 10 % of
    // its travel.
    assert_eq!(saved.karaoke_voice, 10);
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
}

/// The session: the queue, the track in it and the browser's folder and
/// row are saved on quitting and come back at the next start (nothing
/// plays); `ctrl+s` saves the queue as a playlist, `ctrl+l` shows the saved
/// queues, and alt+enter on one plays it in place of the queue.
#[test]
fn the_session_and_saved_queues_come_back() {
    let _guard = jodel_testkit::real_time();
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
    let mut app = common::fitted_app(with_session(true), 120, 40);
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
    let mut app = common::fitted_app(with_session(true), 120, 40);
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
}

/// No separator: the stems are never made (a transition fades its tracks
/// whole), and no GPU is set up for a test.
fn no_separator() -> Option<&'static dyn jodel_playback::stems::Separator> {
    None
}

/// Each stem a quarter of the mix.
struct Quarter;

impl jodel_playback::stems::Separator for Quarter {
    fn name(&self) -> String {
        "quarter".into()
    }

    fn masks(&self, _frames: usize, stft: &[f32]) -> Result<Vec<f32>, String> {
        Ok((0..4 * stft.len())
            .map(|i| if i % 2 == 0 { 0.25 } else { 0.0 })
            .collect())
    }
}

static QUARTER: Quarter = Quarter;

fn quarter() -> Option<&'static dyn jodel_playback::stems::Separator> {
    Some(&QUARTER)
}

const MIX_RATE: u32 = 44_100;

/// The app on a memory desk with two tracks (8 s and 12 s), the first
/// playing, stems by `separator` (on a "GPU" when there is one), and a
/// transition sent to the engine and armed: from 4 s of the first, over
/// two seconds, into the second's first second, a fade with the beats at
/// one tempo. Its folder goes with it.
fn armed_transition(
    tag: &str,
    separator: fn() -> Option<&'static dyn jodel_playback::stems::Separator>,
) -> (App, jodel_testkit::TempDir) {
    use jodel_output::{Capabilities, SampleFormat, StreamFormat};
    use jodel_playback::{Command, Envelope, TempoMode, TransitionPlan};
    use jodeldiplom::app::MemoryOutput;
    use jodeldiplom::core::transition::Status;
    let dir = jodel_testkit::TempDir::new(tag);
    let (a, b) = (dir.join("a.wav"), dir.join("b.wav"));
    common::write_wav(&a, MIX_RATE, 8.0);
    common::write_wav(&b, MIX_RATE, 12.0);
    let mut caps = Capabilities::new();
    caps.insert(StreamFormat::new(MIX_RATE, 2, SampleFormat::S24In32Low));
    let mut options = AppOptions::new(dir.to_path_buf());
    options.output = Output::Memory(MemoryOutput::new(caps, SimClock::Paced));
    options.devices = Arc::new(|| Ok(Vec::new()));
    options.separator = separator;
    let mut app = common::fitted_app(options, 120, 40);
    if separator().is_some() {
        assert!(common::wait_for(&mut app, WAIT, |a| a.state.gpu
            != jodeldiplom::core::Gpu::Looking));
        app.dispatch(Action::Gpu(Ok("the quarter".into())));
    }
    app.run(Effect::Engine(Command::Load(vec![a, b])));
    app.run(Effect::Engine(Command::Play(0)));
    assert!(common::wait_for(&mut app, WAIT, |a| a
        .state
        .playing
        .anchor
        .is_some_and(|p| p.index == 0)));
    let fade = |from: f64, to: f64| Envelope::fade(from, to, 0.0, 1.0);
    let plan = TransitionPlan {
        from: 0,
        into: 1,
        start: 4.0,
        length: 2.0,
        entry: 1.0,
        beats: (0..5)
            .map(|i| (4.0 + 0.5 * f64::from(i), 1.0 + 0.5 * f64::from(i)))
            .collect(),
        lanes: std::array::from_fn(|i| {
            if i < 4 {
                fade(1.0, 0.0)
            } else {
                fade(0.0, 1.0)
            }
        }),
        tempo: TempoMode::Varispeed,
        settle: 0.5,
        glide: false,
    };
    app.run(Effect::Engine(Command::Transition(Box::new(plan))));
    assert!(common::wait_for(&mut app, WAIT, |a| a
        .state
        .transition
        .draft
        .status
        == Status::Armed));
    (app, dir)
}

/// Handed over: the second track on the other head, and two positions
/// told of it (the frame of a change is worked out from them).
fn on_b(app: &App) -> bool {
    let p = &app.state.playing;
    p.speed.is_some()
        && p.anchor
            .is_some_and(|p| p.index == 1 && p.at.head == 1 && p.seconds > 3.5)
}

/// The app's engine plays on two heads of its desk: a transition sent to
/// it is mixed there (not called off for want of a desk), its next track
/// starts on the other head, and a stem's strip muted from a frame of
/// that head's output after the hand-over lands on that frame (worked out
/// from the positions told of it), not at once.
#[test]
fn a_transition_plays_on_the_desk_and_a_mute_after_it_lands_on_the_new_heads_frame() {
    let _guard = jodel_testkit::real_time();
    use jodeldiplom::core::desk::{DeskChange, StripKind};
    let (mut app, _dir) = armed_transition("app-transition", no_separator);
    assert!(
        common::wait_for(&mut app, WAIT, on_b),
        "{:?}",
        app.state.message
    );
    let said = app.state.message.as_ref().map(|m| m.text.clone());
    assert!(!said.as_deref().unwrap_or("").contains("desk"), "{said:?}");
    // A second on, on the second head's frames.
    let anchor = app.state.playing.anchor.unwrap();
    let on = jodeldiplom::core::HeadFrame {
        frame: anchor.at.frame + u64::from(MIX_RATE),
        ..anchor.at
    };
    app.run(Effect::Desk(DeskChange::MuteAt(StripKind::Drums, true, on)));
    let muted = |a: &App| {
        a.desk().is_some_and(|d| {
            d.strips
                .iter()
                .any(|s| s.name == StripKind::Drums.name() && s.mute)
        })
    };
    assert!(common::wait_for(&mut app, WAIT, muted));
    let heard_then = app.state.playing.anchor.unwrap().seconds;
    assert!(
        (heard_then - (anchor.seconds + 1.0)).abs() < 0.25,
        "muted with {heard_then} s heard, the frame of {} s",
        anchor.seconds + 1.0
    );
    app.shutdown();
}

/// The app's state with the desk as it is now looked at (the meters left
/// out): what the Mixer tab's PLAYER strip says.
fn mixer_tab(app: &App) -> String {
    use jodeldiplom::core::desk::{self, Channel, Look, StripKind};
    use jodeldiplom::gui::mixer::said;
    let view_now = app.desk().expect("a desk");
    let ids = StripKind::ALL.map(|kind| {
        view_now
            .strips
            .iter()
            .find(|s| s.name == kind.name())
            .map_or(0, |s| s.id)
    });
    let mut state = app.state.clone();
    state.side = jodeldiplom::core::Side::Mixer;
    let look = Look::of(view_now, jodel_output::mixer::Meters::default(), ids);
    let now = state.now_ms;
    desk::feed(&mut state, look, now);
    said::text(&state, Channel::Strip(StripKind::Player)).join("\n")
}

/// How many of the player's heads have an output open on the desk, and
/// the Lead's vocals lane's gain as last mixed (with it heard).
fn heads_and_voice(app: &App) -> (usize, Option<f64>) {
    use jodel_output::mixer::Role;
    use jodeldiplom::core::desk::StripKind;
    let desk = app.desk().expect("a desk");
    let vocals = desk
        .strips
        .iter()
        .find(|s| s.name == StripKind::Vocals.name())
        .map(|s| s.id);
    let Some(player) = desk.strips.iter().find(|s| s.name == "Player") else {
        return (0, None);
    };
    let open = player.inputs.iter().filter(|i| i.format.is_some()).count();
    let voice = player
        .inputs
        .iter()
        .find(|i| i.role == Role::Lead)
        .and_then(|i| i.lanes.iter().find(|l| Some(l.strip) == vocals && l.heard))
        .map(|l| l.gain);
    (open, voice)
}

/// Through a transition on the real desk: the Karaoke voice (the vocals
/// lane) set before it is on the next track's head too, and heard there
/// after the hand-over; the Mixer tab shows the next track's head (NEXT)
/// while it plays beside the playing one, and no more once the head handed
/// over from has played out and closed after the mix.
#[test]
fn the_voice_and_the_mixer_tab_follow_a_transition_on_the_desk() {
    let _guard = jodel_testkit::real_time();
    use jodeldiplom::core::desk::{DeskChange, StripKind};
    use jodeldiplom::core::transition::Status;
    let (mut app, _dir) = armed_transition("app-transition-voice", quarter);
    app.run(Effect::Desk(DeskChange::Lane(StripKind::Vocals, 0.25)));
    // Both heads open: the next track's beside the playing one.
    assert!(common::wait_for(&mut app, WAIT, |a| heads_and_voice(a).0 == 2));
    let text = mixer_tab(&app);
    assert!(text.contains("NEXT"), "{text}");
    // Handed over, the mix done, the head handed over from closed.
    assert!(common::wait_for(&mut app, WAIT, on_b));
    assert!(common::wait_for(&mut app, WAIT, |a| {
        a.state.transition.draft.status == Status::Draft && heads_and_voice(a).0 == 1
    }));
    let text = mixer_tab(&app);
    assert!(!text.contains("NEXT"), "{text}");
    // The voice on the next track's head, now on air.
    let voice = |a: &App| {
        heads_and_voice(a)
            .1
            .is_some_and(|g| (g - 0.25).abs() < 1e-6)
    };
    assert!(
        common::wait_for(&mut app, WAIT, voice),
        "{:?}",
        heads_and_voice(&app)
    );
    app.shutdown();
}
