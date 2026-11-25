//! The Mixer tab: its faders, buttons and keys, what they send to the
//! mixer, and the settings made again on a new mixer. The window's Mixer
//! tab sends the desk's actions ([`DeskAction`]): these are what its
//! faders, buttons and strips send.

mod common;

use std::sync::Arc;
use std::time::Duration;

use jodel_output::input::Input;
use jodel_output::SimClock;
use jodeldiplom::app::{App, AppOptions, Output};
use jodeldiplom::core::desk::{taper, Channel, DeskAction, DeskButton, DeskChange, StripKind};
use jodeldiplom::core::keymap::{lookup, Key, KeyCode, Meaning, Op};
use jodeldiplom::core::{update, Action, Effect, Side, State, DOUBLE_CLICK_MS};

fn tab() -> State {
    common::desk_state(120, 36)
}

/// What the Mixer tab's widgets send, carried out.
fn desk(state: &mut State, action: DeskAction) -> Vec<Effect> {
    update(state, Action::Desk(action))
}

fn changes(fx: &[Effect]) -> Vec<DeskChange> {
    fx.iter()
        .filter_map(|e| match e {
            Effect::Desk(c) => Some(*c),
            _ => None,
        })
        .collect()
}

fn db(state: &State, strip: StripKind) -> f64 {
    state.desk.setting(strip).db
}

fn grab(strip: StripKind, position: f64) -> DeskAction {
    DeskAction::Grab { strip, position }
}

#[test]
fn a_held_fader_follows_the_pointer_to_unity_exactly() {
    let mut state = tab();
    let player = StripKind::Player;
    let unity = taper::position_of(0.0);
    // Taken where the cap is: it does not move.
    let fx = desk(&mut state, grab(player, unity));
    assert_eq!(db(&state, player), 0.0);
    assert!(changes(&fx).is_empty(), "{fx:?}");
    assert!(state.desk.dragging.is_some());
    // Down a little at a time: every step a lower level, sent to the mixer.
    let mut last = 0.0;
    for k in 1..=24 {
        let position = unity - f64::from(k) / 100.0;
        let fx = desk(&mut state, DeskAction::Slide(position));
        let now = db(&state, player);
        assert_eq!(now, taper::snap(taper::db_at(position)), "step {k}");
        assert!(now < last, "step {k}: {now} after {last}");
        assert_eq!(changes(&fx), [DeskChange::Level(player, taper::gain(now))]);
        last = now;
    }
    // Back at unity's position: exactly 0 dB, unity gain, bit-perfect.
    desk(&mut state, DeskAction::Slide(unity));
    assert_eq!(db(&state, player), 0.0);
    assert_eq!(taper::gain(db(&state, player)), 1.0);
    // The ends: off, and the top.
    desk(&mut state, DeskAction::Slide(0.0));
    assert_eq!(db(&state, player), f64::NEG_INFINITY);
    desk(&mut state, DeskAction::Slide(1.0));
    assert_eq!(db(&state, player), taper::TOP_DB);
    desk(&mut state, DeskAction::Release);
    assert!(state.desk.dragging.is_none());
    // Let go, a slide moves nothing.
    desk(&mut state, DeskAction::Slide(unity));
    assert_eq!(db(&state, player), taper::TOP_DB);
}

#[test]
fn a_press_on_the_track_brings_the_cap_and_a_double_click_is_unity() {
    let mut state = tab();
    let preview = StripKind::Preview;
    // Low on the track: the cap comes there, and the strip is selected.
    desk(&mut state, grab(preview, 0.15));
    let low = db(&state, preview);
    assert!(low < -30.0, "{low}");
    assert_eq!(state.desk.selected, Channel::Strip(preview));
    desk(&mut state, DeskAction::Release);
    // A second press soon after: back to 0 dB, nothing held.
    state.now_ms += DOUBLE_CLICK_MS / 2;
    let fx = desk(&mut state, grab(preview, 0.15));
    assert_eq!(db(&state, preview), 0.0);
    assert_eq!(changes(&fx), [DeskChange::Level(preview, 1.0)]);
    assert!(state.desk.dragging.is_none());
    // Long after, a press is a press.
    state.now_ms += DOUBLE_CLICK_MS * 4;
    desk(&mut state, grab(preview, 0.15));
    assert_eq!(db(&state, preview), low);
}

/// A stem's fader held while the strips change under it (the stems gone,
/// the window narrower: its widget is not where it was, or not at all) is
/// let go of on the next release of the left button, wherever that is:
/// nothing stays held. A release with no fader held does nothing.
#[test]
fn a_fader_held_while_the_strips_change_is_let_go_of_on_any_release() {
    use jodeldiplom::core::desk::let_go;
    let mut state = tab();
    assert_eq!(let_go(&state), None);
    let drums = StripKind::Drums;
    desk(&mut state, grab(drums, taper::position_of(0.0)));
    assert!(state.desk.dragging.is_some());
    // The stems go from the player's head, and the window narrows.
    let player = state.desk.views[StripKind::Player.index()]
        .as_mut()
        .unwrap();
    for input in &mut player.inputs {
        input.parts = None;
    }
    assert!(!state.desk.has_stems());
    update(&mut state, Action::Resize(80, 24));
    let release = let_go(&state).expect("a release");
    desk(&mut state, release);
    assert!(state.desk.dragging.is_none());
    assert_eq!(let_go(&state), None);
}

#[test]
fn the_wheel_moves_a_fader_by_a_db_and_the_buttons_toggle() {
    let mut state = tab();
    let player = StripKind::Player;
    desk(&mut state, DeskAction::Wheel(Channel::Strip(player), 1));
    assert_eq!(db(&state, player), 1.0);
    let fx = desk(&mut state, DeskAction::Wheel(Channel::Strip(player), -1));
    assert_eq!(db(&state, player), 0.0);
    assert_eq!(changes(&fx), [DeskChange::Level(player, 1.0)]);
    // [M] and [S].
    for (button, change) in [
        (DeskButton::Mute(player), DeskChange::Mute(player, true)),
        (DeskButton::Solo(player), DeskChange::Solo(player, true)),
    ] {
        let fx = desk(&mut state, DeskAction::Press(button));
        assert_eq!(changes(&fx), [change]);
    }
    let s = state.desk.setting(player);
    assert!(s.mute && s.solo);
    // A click on the master selects it; the wheel there steps the
    // device's own volume, never a gain of ours.
    desk(&mut state, DeskAction::Select(Channel::Main));
    assert_eq!(state.desk.selected, Channel::Main);
    let fx = desk(&mut state, DeskAction::Wheel(Channel::Main, 1));
    assert!(changes(&fx).is_empty());
    assert!(fx.contains(&Effect::DeviceVolume(
        jodeldiplom::core::VolumeChange::Step(1)
    )));
    // The microphone's ECHO turns its echo canceller off, and saves it.
    let fx = desk(&mut state, DeskAction::Press(DeskButton::Echo));
    assert!(!state.config.karaoke_echo);
    assert!(fx.contains(&Effect::SaveConfig));
}

#[test]
fn the_keys_of_the_mixer_tab_leave_the_global_capitals_alone() {
    let state = tab();
    let op = |key| match lookup(&state, key) {
        Some(Meaning::Op(op)) => Some(op),
        _ => None,
    };
    for (key, want) in [
        (Key::plain(KeyCode::Left), Op::DeskPrevStrip),
        (Key::plain(KeyCode::Right), Op::DeskNextStrip),
        (Key::plain(KeyCode::Up), Op::Up),
        (Key::plain(KeyCode::Down), Op::Down),
        (Key::plain(KeyCode::PageUp), Op::PageUp),
        (Key::plain(KeyCode::PageDown), Op::PageDown),
        (Key::plain(KeyCode::Home), Op::Top),
        (Key::plain(KeyCode::End), Op::Bottom),
        (Key::ch('m'), Op::DeskMute),
        (Key::ch('s'), Op::DeskSolo),
        (Key::ch('e'), Op::DeskEcho),
        (Key::ch('i'), Op::MicNext),
        // Still what they are everywhere.
        (Key::ch('M'), Op::ToggleBpm),
        (Key::ch('S'), Op::Stop),
        (Key::ch(' '), Op::PlayPause),
        (Key::plain(KeyCode::Tab), Op::ToggleSide),
        (Key::shift(KeyCode::Right), Op::Next),
    ] {
        assert_eq!(op(key), Some(want), "{key:?}");
    }
    // Tab goes round all five, the Mixer tab last.
    let mut s = tab();
    s.side = Side::Karaoke;
    update(&mut s, Action::Op(Op::ToggleSide));
    assert_eq!(s.side, Side::Mixer);
    update(&mut s, Action::Op(Op::ToggleSide));
    assert_eq!(s.side, Side::Browser);
}

/// Waits until `done`, pumping the app.
fn until(app: &mut App, mut done: impl FnMut(&App) -> bool) -> bool {
    for _ in 0..400 {
        app.pump();
        if done(app) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    false
}

#[test]
fn the_faders_are_set_on_the_mixer_and_again_on_a_new_one() {
    let _guard = common::real_time();
    let tree = common::music_tree("desk");
    let mut options = AppOptions::new(tree.album.clone());
    options.output = Output::Null(SimClock::Realtime);
    options.devices = Arc::new(|| Ok(Vec::new()));
    options.mic = Arc::new(|_| Ok(Input::feed(48_000, 240).0));
    let mut app = App::new(options);
    let level = |app: &App, kind: StripKind| {
        let desk = app.desk().expect("an engine");
        let s = desk.strips.iter().find(|s| s.name == kind.name()).unwrap();
        (s.level, s.mute, s.solo)
    };
    // Every start at 0 dB: nothing added; the game's effects on a strip
    // of their own.
    assert_eq!(level(&app, StripKind::Player), (1.0, false, false));
    assert_eq!(level(&app, StripKind::Effects), (1.0, false, false));
    app.dispatch(Action::ShowSide(Side::Mixer));
    // The tab keeps drawing; the microphone stays closed until its strip
    // is selected (K6), and closes when another one is.
    assert!(app.animating());
    assert!(!until(&mut app, |a| a.state.karaoke.mic.name.is_some()));
    app.dispatch(Action::Desk(DeskAction::Select(Channel::Mic)));
    assert!(until(&mut app, |a| a.state.karaoke.mic.name.is_some()));
    app.dispatch(Action::Desk(DeskAction::Select(Channel::Strip(
        StripKind::Player,
    ))));
    assert!(app.state.karaoke.mic.name.is_none());
    for _ in 0..6 {
        app.dispatch(Action::Op(Op::Down));
    }
    app.dispatch(Action::Op(Op::DeskMute));
    app.dispatch(Action::Op(Op::DeskNextStrip));
    app.dispatch(Action::Op(Op::DeskSolo));
    let want = taper::gain(-6.0);
    let (l, mute, solo) = level(&app, StripKind::Player);
    assert!(
        (l - want).abs() < 1e-12 && mute && !solo,
        "{l} {mute} {solo}"
    );
    assert_eq!(level(&app, StripKind::Preview), (1.0, false, true));
    // A frame looks at the desk: it is in the state, the meters too.
    app.heard_for_frame();
    assert!(app.state.desk.master.is_some());
    assert!(app.state.desk.view(StripKind::Player).is_some());
    // The engine fails and starts again on a new mixer: the same settings.
    app.dispatch(Action::EngineGone(1));
    assert!(app.desk().is_some());
    let (l, mute, _) = level(&app, StripKind::Player);
    assert!(
        (l - want).abs() < 1e-12 && mute,
        "after the restart: {l} {mute}"
    );
    assert_eq!(level(&app, StripKind::Preview), (1.0, false, true));
    assert_eq!(level(&app, StripKind::Effects), (1.0, false, false));
    // Away from the tab and Karaoke, the microphone closes.
    app.dispatch(Action::ShowSide(Side::Browser));
    assert!(app.state.karaoke.mic.name.is_none());
    app.shutdown();
    let _ = std::fs::remove_dir_all(&tree.root);
}

#[test]
fn a_stems_strip_takes_its_fader_and_its_buttons() {
    let mut state = tab();
    assert!(state.desk.has_stems());
    // The vocals' [M] (muted on that desk) unmutes them, at once.
    let vocals = StripKind::Vocals;
    let fx = desk(&mut state, DeskAction::Press(DeskButton::Mute(vocals)));
    assert_eq!(changes(&fx), [DeskChange::Mute(vocals, false)]);
    assert_eq!(state.desk.selected, Channel::Strip(vocals));
    // The drums' fader taken and moved down; the wheel over the bass.
    let drums = StripKind::Drums;
    let unity = taper::position_of(0.0);
    desk(&mut state, grab(drums, unity));
    let fx = desk(&mut state, DeskAction::Slide(unity - 0.1));
    let level = db(&state, drums);
    assert!(level < 0.0, "{level}");
    assert_eq!(changes(&fx), [DeskChange::Level(drums, taper::gain(level))]);
    desk(&mut state, DeskAction::Release);
    let bass = StripKind::Bass;
    let fx = desk(&mut state, DeskAction::Wheel(Channel::Strip(bass), 1));
    assert_eq!(changes(&fx), [DeskChange::Level(bass, taper::gain(1.0))]);
    // The stems touched: the engine is told nothing but to make them (the
    // desk plays them, a transition's two tracks too).
    for e in &fx {
        assert!(
            !matches!(e, Effect::Engine(c) if !matches!(c, jodel_playback::Command::MakeStems(_))),
            "{e:?}"
        );
    }
}

/// Each stem a quarter of the mix, so the four add up to it.
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

/// The samples `common::write_wav` writes, frame `i` at `rate`.
fn chord(i: usize, rate: u32) -> i32 {
    let t = i as f64 / f64::from(rate);
    let x = [220.0, 277.18, 329.63, 3_000.0]
        .iter()
        .zip([0.20, 0.15, 0.12, 0.02])
        .map(|(hz, a)| a * (std::f64::consts::TAU * hz * t).sin())
        .sum::<f64>();
    (x * 8_388_607.0 * 0.5) as i32
}

/// Through a real mixer: a stem's strip muted from a frame of the player's
/// output switches the device to the other three stems on that frame
/// (three quarters of the file here), and unmuted from another, the file
/// again bit for bit; after a seek, a frame worked out from the position
/// told lands where that position says. The device plays at about a
/// device's pace but only what the mixer gives it ([`SimClock::Paced`]),
/// and the mixer waits for the player: one put off on a busy machine is
/// late, never a gap of silence in what is checked bit for bit.
#[test]
fn a_stem_muted_on_its_frame_is_heard_and_unmuted_the_file_is_again() {
    let _guard = common::real_time();
    use jodel_output::lanes::Parts;
    use jodel_output::{Capabilities, SampleFormat, StreamFormat};
    use jodeldiplom::app::MemoryOutput;
    // At the separator's rate, so neither main nor the stems are
    // resampled (a debug build keeps up in real time).
    const RATE: u32 = 44_100;
    const SEEK: f64 = 1.0;
    let dir = common::temp_dir("desk-stems");
    let file = dir.join("chord.wav");
    common::write_wav(&file, RATE, 10.0);
    let mut caps = Capabilities::new();
    caps.insert(StreamFormat::new(RATE, 2, SampleFormat::S24In32Low));
    let memory = MemoryOutput::new(caps, SimClock::Paced);
    let mut options = AppOptions::new(dir.clone());
    options.output = Output::Memory(memory.clone());
    options.devices = Arc::new(|| Ok(Vec::new()));
    options.separator = quarter;
    options.autoplay = Some(file);
    let mut app = App::new(options);
    // The Mixer tab shows: the stems are made, on the "GPU" (the quarter).
    app.dispatch(Action::ShowSide(Side::Mixer));
    assert!(until(&mut app, |a| a.state.gpu != jodeldiplom::core::Gpu::Looking));
    app.dispatch(Action::Gpu(Some("the quarter".into())));
    let heard = |a: &App| a.state.playing.anchor.map_or(0, |p| p.at.frame);
    // Once the stems are there: muted half a second on (the mixer is a
    // few periods ahead of what is heard), for a second.
    let present = |a: &App| {
        a.desk().is_some_and(|d| {
            d.strips[0]
                .inputs
                .iter()
                .any(|i| i.parts.is_some_and(|p| p.state == Parts::Present))
        })
    };
    assert!(common::wait_for(&mut app, Duration::from_secs(30), present));
    let on = heard(&app) + u64::from(RATE) / 2;
    let off = on + u64::from(RATE);
    // All of it, to half a second past the unmute (where the seek comes),
    // within the first chunk's stems (4.8 s of the file: its own share of
    // the 6 s separated), however slow the separation of the next.
    assert!(
        off + u64::from(RATE) / 2 < 9 * u64::from(RATE) / 2,
        "too late to set it: the stems there at {} s",
        heard(&app) as f64 / f64::from(RATE)
    );
    app.run(Effect::Desk(DeskChange::MuteAt(
        StripKind::Drums,
        true,
        jodeldiplom::core::HeadFrame { head: 0, frame: on },
    )));
    // A strip has one change of its mute waiting: the next once this one
    // has landed.
    assert!(common::wait_for(
        &mut app,
        Duration::from_secs(30),
        |a| heard(a) > on
    ));
    app.run(Effect::Desk(DeskChange::MuteAt(
        StripKind::Drums,
        false,
        jodeldiplom::core::HeadFrame {
            head: 0,
            frame: off,
        },
    )));
    // Then a seek back to 1 s, and the drums muted a second after the
    // first position told from there: Position's output frame and the
    // mixer's stream frame are the same count after a seek too. (From
    // here, at least 2 s in: no position told before the seek is one of
    // its first second.)
    assert!(common::wait_for(
        &mut app,
        Duration::from_secs(30),
        |a| heard(a) > off + u64::from(RATE) / 2
    ));
    assert!(heard(&app) as f64 >= (SEEK + 1.0) * f64::from(RATE));
    app.run(Effect::Engine(jodel_playback::Command::Seek(SEEK)));
    // Two positions told from there (as the stem keys wait for: the first
    // can come before the mixer has mixed any of it again).
    let fresh = |a: &App| {
        let p = &a.state.playing;
        p.speed.is_some()
            && p.anchor
                .is_some_and(|p| (SEEK..SEEK + 1.0).contains(&p.seconds))
    };
    assert!(common::wait_for(&mut app, Duration::from_secs(30), fresh));
    let anchor = app.state.playing.anchor.unwrap();
    let (frame, seconds) = (anchor.at.frame, anchor.seconds);
    let again = frame + u64::from(RATE);
    app.run(Effect::Desk(DeskChange::MuteAt(
        StripKind::Drums,
        true,
        jodeldiplom::core::HeadFrame {
            head: anchor.at.head,
            frame: again,
        },
    )));
    // Where the song is heard when the mixer takes the mute: a second on
    // (the positions come every 100 ms; the mixer runs a little ahead).
    let muted = |a: &App| {
        a.desk().is_some_and(|d| {
            d.strips
                .iter()
                .any(|s| s.name == StripKind::Drums.name() && s.mute)
        })
    };
    assert!(common::wait_for(&mut app, Duration::from_secs(30), muted));
    let heard_then = app.state.playing.anchor.unwrap().seconds;
    assert!(
        (heard_then - (seconds + 1.0)).abs() < 0.25,
        "muted with {heard_then} s heard, the frame of {} s",
        seconds + 1.0
    );
    let ended = |a: &App| a.state.playing.state == jodel_playback::PlayState::Stopped;
    assert!(common::wait_for(
        &mut app,
        Duration::from_secs(30),
        |a| heard(a) > again && ended(a)
    ));
    app.shutdown();
    let captures = memory.captures();
    let streams = captures.last().expect("a device").list();
    let bytes = streams[0].sink.bytes();
    let got: Vec<i32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| (i32::from_le_bytes(*b) << 8) >> 8)
        .collect();
    let (on, off) = (on as usize, off as usize);
    // The file as it is up to the frame, both channels.
    for f in 0..on {
        assert_eq!(got[2 * f], chord(f, RATE), "frame {f}, before");
    }
    // The switch on that very frame: from it, main falls to the parts
    // over the switch as the drums' strip falls over its ramp, and what is
    // heard is main less a quarter of it by the product of the two. (The
    // first frame that differs from the file is no measure: the two start
    // at 0 together, so where the chord is quiet the first steps round to
    // it.) Started a frame off, it is hundreds of steps off.
    let switch = jodel_output::mixer::SWITCH_FADE.as_secs_f64() * f64::from(RATE);
    let ramp = jodel_output::mixer::STRIP_RAMP.as_secs_f64() * f64::from(RATE);
    let model_off = |start: usize| {
        (start..start + ramp as usize)
            .map(|f| {
                let k = (f - start) as f64;
                let (s, g) = ((k / switch).min(1.0), (1.0 - k / ramp).max(0.0));
                let want = f64::from(chord(f, RATE)) * (1.0 - s * (1.0 - g) / 4.0);
                (f64::from(got[2 * f]) - want).abs()
            })
            .fold(0.0, f64::max)
    };
    let offs = [on - 1, on, on + 1].map(model_off);
    assert!(
        offs[1] < 16.0,
        "the switch from frame {on} is {} steps off (from a frame before and after: {} \
         and {})",
        offs[1],
        offs[0],
        offs[2]
    );
    // Three quarters of it once the switch and the strip's ramp are over
    // (within the stems' filters' few steps).
    let worst = (on + 2_000..off)
        .map(|f| (f64::from(got[2 * f]) - 0.75 * f64::from(chord(f, RATE))).abs())
        .fold(0.0, f64::max);
    assert!(worst < 2_000.0, "{worst} steps off three quarters");
    // The file again, bit for bit, once the strip's ramp back (20 ms) has
    // settled by the end of a period (10 ms) and the switch back (10 ms)
    // has run, from the period after: 50 ms on.
    let back = off + 5 * RATE as usize / 100;
    let seek_from = off + RATE as usize / 2;
    for f in back..seek_from {
        assert_eq!(got[2 * f], chord(f, RATE), "frame {f}, after");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The device is opened once, as the player starts, at the rate set (48
/// kHz by default), and stays open while songs at 44.1, 48 and 96 kHz play
/// one into the next: each is routed to its rate, the 44.1 and 96 kHz ones
/// resampled, the 48 kHz one as it is; the device is never opened again.
#[test]
fn the_device_opens_once_for_songs_at_any_rate() {
    let _guard = common::real_time();
    use jodel_output::{Capabilities, SampleFormat, StreamFormat};
    use jodeldiplom::app::MemoryOutput;
    let dir = common::temp_dir("one-rate");
    let rates = [44_100, 48_000, 96_000];
    for (k, rate) in rates.into_iter().enumerate() {
        common::write_wav(&dir.join(format!("{k}.wav")), rate, 1.0);
    }
    let mut caps = Capabilities::new();
    for rate in rates {
        caps.insert(StreamFormat::new(rate, 2, SampleFormat::S24In32Low));
    }
    let memory = MemoryOutput::new(caps, SimClock::Realtime);
    let mut options = AppOptions::new(dir.clone());
    options.output = Output::Memory(memory.clone());
    options.devices = Arc::new(|| Ok(Vec::new()));
    options.separator = || None;
    options.autoplay = Some(dir.join("0.wav"));
    let mut app = App::new(options);
    // Each song's route as it plays, until the last has ended.
    let mut routes = vec![None; rates.len()];
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(t) = &app.state.playing.track {
            routes[t.index] = Some(t.route.clone());
        }
        let stopped = app.state.playing.state == jodel_playback::PlayState::Stopped;
        if stopped && routes.iter().all(Option::is_some) {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "{routes:?}");
        app.pump_wait(Duration::from_millis(5));
    }
    let tb = app.desk().unwrap().master.timebase;
    assert_eq!((tb.epoch, tb.rate), (1, 48_000));
    app.shutdown();
    let engines = memory.captures();
    assert_eq!(engines.len(), 1);
    let streams = engines[0].list();
    assert_eq!(streams.len(), 1, "opened once");
    assert_eq!(streams[0].format.rate, 48_000);
    for (route, rate) in routes.into_iter().flatten().zip(rates) {
        assert_eq!((route.input_rate, route.output_rate), (rate, 48_000));
        assert_eq!(route.resample.is_some(), rate != 48_000, "{rate}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Another rate set for the device opens it again once, on a new mixer at
/// that rate: the song paused where it was stays paused there, the same
/// song.
#[test]
fn another_rate_opens_the_device_again_where_the_song_was() {
    let _guard = common::real_time();
    use jodel_output::{Capabilities, SampleFormat, StreamFormat};
    use jodel_playback::PlayState;
    use jodeldiplom::app::MemoryOutput;
    let dir = common::temp_dir("reopen-rate");
    common::write_wav(&dir.join("a.wav"), 48_000, 8.0);
    let mut caps = Capabilities::new();
    for rate in [48_000, 96_000] {
        caps.insert(StreamFormat::new(rate, 2, SampleFormat::S24In32Low));
    }
    let memory = MemoryOutput::new(caps, SimClock::Realtime);
    let mut options = AppOptions::new(dir.clone());
    options.output = Output::Memory(memory.clone());
    options.devices = Arc::new(|| Ok(Vec::new()));
    options.separator = || None;
    options.autoplay = Some(dir.join("a.wav"));
    let mut app = App::new(options);
    assert_eq!(app.state.opened_rate, Some(48_000));
    let wait = Duration::from_secs(30);
    let played = |a: &App| a.state.playing.position > 1.5;
    assert!(common::wait_for(&mut app, wait, played));
    app.dispatch(Action::Op(Op::PlayPause));
    let paused = |a: &App| a.state.playing.state == PlayState::Paused;
    assert!(common::wait_for(&mut app, wait, paused));
    let (index, at) = (app.state.queue.current, app.state.playing.position);
    app.state.config.set_rate(96_000);
    app.run(Effect::ReopenOutput);
    let back = |a: &App| {
        a.state.playing.track.is_some()
            && a.state.playing.state == PlayState::Paused
            && (a.state.playing.position - at).abs() < 0.5
    };
    assert!(
        common::wait_for(&mut app, wait, back),
        "paused where it was"
    );
    assert_eq!(app.state.queue.current, index);
    assert_eq!(app.state.opened_rate, Some(96_000));
    let route = app.state.playing.track.as_ref().unwrap().route.clone();
    assert_eq!(route.output_rate, 96_000);
    app.shutdown();
    let engines = memory.captures();
    assert_eq!(engines.len(), 2, "a new mixer");
    let streams = engines[1].list();
    assert_eq!(streams.len(), 1);
    assert_eq!(streams[0].format.rate, 96_000);
    let _ = std::fs::remove_dir_all(&dir);
}
