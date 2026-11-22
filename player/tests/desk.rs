//! The Mix tab: its faders, buttons and keys, what they send to the
//! mixer, and the settings made again on a new mixer.

mod common;

use std::sync::Arc;
use std::time::Duration;

use jodel_output::input::Input;
use jodel_output::SimClock;
use jodeldiplom::app::{App, AppOptions, Output};
use jodeldiplom::core::desk::mouse::{cap_y, position_at, DETENT_ROWS};
use jodeldiplom::core::desk::{taper, Channel, DeskAction, DeskButton, DeskChange, StripKind};
use jodeldiplom::core::keymap::{lookup, Key, KeyCode, Meaning, Op};
use jodeldiplom::core::layout::desk::ChannelLayout;
use jodeldiplom::core::layout::{self, Layout};
use jodeldiplom::core::mouse::{self, MouseButton, MouseInput, MouseKind};
use jodeldiplom::core::{update, Action, Effect, Side, State, DOUBLE_CLICK_MS};

fn tab() -> State {
    common::desk_state(120, 36)
}

fn strip_layout(state: &State, channel: Channel) -> ChannelLayout {
    let Layout::Main(m) = layout::compute(state) else {
        panic!("the main screen")
    };
    m.desk.expect("the desk").channel(channel).unwrap().clone()
}

/// What the pointer does at (`x`, `y`) cells, carried out.
#[track_caller]
fn point(state: &mut State, kind: MouseKind, x: f64, y: f64) -> Vec<Effect> {
    let input = MouseInput {
        kind,
        col: x as u16,
        row: y as u16,
    };
    match mouse::action_at(state, input, x, y) {
        Some(action) => update(state, action),
        None => Vec::new(),
    }
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

const LEFT: MouseKind = MouseKind::Down(MouseButton::Left);
const DRAG: MouseKind = MouseKind::Drag(MouseButton::Left);
const UP: MouseKind = MouseKind::Up(MouseButton::Left);

#[test]
fn a_fader_follows_the_pointer_by_eighths_of_a_cell() {
    let mut state = tab();
    let player = Channel::Strip(StripKind::Player);
    let track = strip_layout(&state, player).track.unwrap();
    let x = f64::from(track.x) + 1.5;
    // Taken by the cap, a little off its middle: it does not jump.
    let at = cap_y(track, taper::position_of(0.0));
    let fx = point(&mut state, LEFT, x, at + 0.3);
    assert_eq!(db(&state, StripKind::Player), 0.0);
    assert!(changes(&fx).is_empty(), "{fx:?}");
    assert!(state.desk.dragging.is_some());
    // Down an eighth of a cell at a time, the cap's middle staying under
    // the pointer where it was taken: within the detent (0.4 rows) it
    // stays at exactly 0 dB, then every step is a lower level.
    let mut last = 0.0;
    for k in 1..=24 {
        let y = at + 0.3 + f64::from(k) / 8.0;
        let fx = point(&mut state, DRAG, x, y);
        let now = db(&state, StripKind::Player);
        let want = taper::snap(taper::db_at(position_at(track, y - 0.3)));
        assert_eq!(now, want, "step {k}");
        if f64::from(k) / 8.0 < DETENT_ROWS {
            assert_eq!(now, 0.0, "step {k}: in the detent");
            assert!(changes(&fx).is_empty());
            continue;
        }
        assert!(now < last, "step {k}: {now} after {last}");
        assert_eq!(
            changes(&fx),
            [DeskChange::Level(StripKind::Player, taper::gain(now))]
        );
        last = now;
    }
    // Back up into the detent: exactly 0 dB, unity, bit-perfect.
    point(&mut state, DRAG, x, at + 0.3 + 0.2);
    assert_eq!(db(&state, StripKind::Player), 0.0);
    assert_eq!(taper::gain(db(&state, StripKind::Player)), 1.0);
    // The pointer may leave the track while held: far below, off; far
    // above (and beside), the top.
    point(&mut state, DRAG, x + 30.0, 500.0);
    assert_eq!(db(&state, StripKind::Player), f64::NEG_INFINITY);
    point(&mut state, DRAG, 0.5, 0.5);
    assert_eq!(db(&state, StripKind::Player), taper::TOP_DB);
    point(&mut state, UP, 0.5, 0.5);
    assert!(state.desk.dragging.is_none());
    // Let go, the pointer moves nothing.
    point(&mut state, DRAG, x, at);
    assert_eq!(db(&state, StripKind::Player), taper::TOP_DB);
}

#[test]
fn a_press_on_the_track_brings_the_cap_and_a_double_click_is_unity() {
    let mut state = tab();
    let preview = Channel::Strip(StripKind::Preview);
    let track = strip_layout(&state, preview).track.unwrap();
    let x = f64::from(track.x) + 1.5;
    // Low on the track: the cap comes there, and the strip is selected.
    let y = f64::from(track.bottom()) - 3.5;
    point(&mut state, LEFT, x, y);
    let low = db(&state, StripKind::Preview);
    assert!(low < -30.0, "{low}");
    assert_eq!(state.desk.selected, preview);
    point(&mut state, UP, x, y);
    // A second press soon after: back to 0 dB, nothing held.
    state.now_ms += DOUBLE_CLICK_MS / 2;
    let fx = point(&mut state, LEFT, x, y);
    assert_eq!(db(&state, StripKind::Preview), 0.0);
    assert_eq!(changes(&fx), [DeskChange::Level(StripKind::Preview, 1.0)]);
    assert!(state.desk.dragging.is_none());
    // Long after, a press is a press.
    state.now_ms += DOUBLE_CLICK_MS * 4;
    point(&mut state, LEFT, x, y);
    assert_eq!(db(&state, StripKind::Preview), low);
}

#[test]
fn the_wheel_moves_a_fader_by_a_db_and_the_buttons_toggle() {
    let mut state = tab();
    let player = strip_layout(&state, Channel::Strip(StripKind::Player));
    let track = player.track.unwrap();
    let (x, y) = (f64::from(track.x) + 0.5, f64::from(track.y) + 2.5);
    point(&mut state, MouseKind::ScrollUp, x, y);
    assert_eq!(db(&state, StripKind::Player), 1.0);
    // Anywhere over the strip.
    let name = player.name;
    let fx = point(
        &mut state,
        MouseKind::ScrollDown,
        f64::from(name.x) + 1.5,
        f64::from(name.y) + 0.5,
    );
    assert_eq!(db(&state, StripKind::Player), 0.0);
    assert_eq!(changes(&fx), [DeskChange::Level(StripKind::Player, 1.0)]);
    // [M] and [S].
    for (button, change) in [
        (
            DeskButton::Mute(StripKind::Player),
            DeskChange::Mute(StripKind::Player, true),
        ),
        (
            DeskButton::Solo(StripKind::Player),
            DeskChange::Solo(StripKind::Player, true),
        ),
    ] {
        let rect = player.buttons.iter().find(|b| b.1 == button).unwrap().0;
        let fx = point(
            &mut state,
            LEFT,
            f64::from(rect.x) + 1.5,
            f64::from(rect.y) + 0.5,
        );
        assert_eq!(changes(&fx), [change]);
    }
    let s = state.desk.setting(StripKind::Player);
    assert!(s.mute && s.solo);
    // A click on the master selects it; the wheel there steps the
    // device's own volume, never a gain of ours.
    let main = strip_layout(&state, Channel::Main);
    let (mx, my) = (f64::from(main.name.x) + 2.5, f64::from(main.name.y) + 0.5);
    point(&mut state, LEFT, mx, my);
    assert_eq!(state.desk.selected, Channel::Main);
    let fx = point(&mut state, MouseKind::ScrollUp, mx, my);
    assert!(changes(&fx).is_empty());
    assert!(fx.contains(&Effect::DeviceVolume(
        jodeldiplom::core::VolumeChange::Step(1)
    )));
    // The microphone's ECHO turns its echo canceller off, and saves it.
    let mic = strip_layout(&state, Channel::Mic);
    let echo = mic
        .buttons
        .iter()
        .find(|b| b.1 == DeskButton::Echo)
        .unwrap()
        .0;
    let fx = point(
        &mut state,
        LEFT,
        f64::from(echo.x) + 1.5,
        f64::from(echo.y) + 0.5,
    );
    assert!(!state.config.karaoke_echo);
    assert!(fx.contains(&Effect::SaveConfig));
}

#[test]
fn the_keys_of_the_mix_tab_leave_the_global_capitals_alone() {
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
    // Tab goes round all five, the Mix tab last.
    let mut s = tab();
    s.side = Side::Karaoke;
    update(&mut s, Action::Op(Op::ToggleSide));
    assert_eq!(s.side, Side::Mix);
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
    // Every start at 0 dB: nothing added.
    assert_eq!(level(&app, StripKind::Player), (1.0, false, false));
    app.dispatch(Action::ShowSide(Side::Mix));
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
    // Away from the tab and Karaoke, the microphone closes.
    app.dispatch(Action::ShowSide(Side::Browser));
    assert!(app.state.karaoke.mic.name.is_none());
    app.shutdown();
    let _ = std::fs::remove_dir_all(&tree.root);
}

#[test]
fn a_stems_strip_takes_the_pointer_and_its_buttons() {
    let mut state = tab();
    assert!(state.desk.has_stems());
    // The vocals' [M] (muted on that desk) unmutes them, at once.
    let vocals = strip_layout(&state, Channel::Strip(StripKind::Vocals));
    let mute = vocals
        .buttons
        .iter()
        .find(|b| b.1 == DeskButton::Mute(StripKind::Vocals))
        .unwrap()
        .0;
    let fx = point(
        &mut state,
        LEFT,
        f64::from(mute.x) + f64::from(mute.width) / 2.0,
        f64::from(mute.y) + 0.5,
    );
    assert_eq!(changes(&fx), [DeskChange::Mute(StripKind::Vocals, false)]);
    assert_eq!(state.desk.selected, Channel::Strip(StripKind::Vocals));
    // The drums' fader follows the pointer down; the wheel over the bass.
    let track = strip_layout(&state, Channel::Strip(StripKind::Drums))
        .track
        .unwrap();
    let x = f64::from(track.x) + 1.5;
    let at = cap_y(track, taper::position_of(0.0));
    point(&mut state, LEFT, x, at);
    let fx = point(&mut state, DRAG, x, at + 3.0);
    let drums = db(&state, StripKind::Drums);
    assert!(drums < 0.0, "{drums}");
    assert_eq!(
        changes(&fx),
        [DeskChange::Level(StripKind::Drums, taper::gain(drums))]
    );
    point(&mut state, UP, x, at + 3.0);
    let bass = strip_layout(&state, Channel::Strip(StripKind::Bass)).name;
    let fx = point(
        &mut state,
        MouseKind::ScrollUp,
        f64::from(bass.x) + 1.5,
        f64::from(bass.y) + 0.5,
    );
    assert_eq!(
        changes(&fx),
        [DeskChange::Level(StripKind::Bass, taper::gain(1.0))]
    );
    // The stems touched: the transitions are told the stems' gains.
    assert!(fx.iter().any(|e| matches!(
        e,
        Effect::Engine(jodel_playback::Command::StemMask { at: None, .. })
    )));
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
/// told lands where that position says.
#[test]
fn a_stem_muted_on_its_frame_is_heard_and_unmuted_the_file_is_again() {
    use jodel_output::lanes::Parts;
    use jodel_output::{Capabilities, SampleFormat, StreamFormat};
    use jodeldiplom::app::MemoryOutput;
    // At the separator's rate, so neither main nor the stems are
    // resampled (a debug build keeps up in real time).
    const RATE: u32 = 44_100;
    const SEEK: f64 = 2.0;
    let dir = common::temp_dir("desk-stems");
    let file = dir.join("chord.wav");
    common::write_wav(&file, RATE, 10.0);
    let mut caps = Capabilities::new();
    caps.insert(StreamFormat::new(RATE, 2, SampleFormat::S24In32Low));
    let memory = MemoryOutput::new(caps, SimClock::Realtime);
    let mut options = AppOptions::new(dir.clone());
    options.output = Output::Memory(memory.clone());
    options.devices = Arc::new(|| Ok(Vec::new()));
    options.separator = quarter;
    options.autoplay = Some(file);
    let mut app = App::new(options);
    // The Mix tab shows: the stems are made, on the "GPU" (the quarter).
    app.dispatch(Action::ShowSide(Side::Mix));
    assert!(until(&mut app, |a| a.state.gpu != jodeldiplom::core::Gpu::Looking));
    app.dispatch(Action::Gpu(Some("the quarter".into())));
    let heard = |a: &App| a.state.playing.anchor.map_or(0, |(_, frame, _)| frame);
    // Once the stems are there: muted a second on, for a second and a half.
    let present = |a: &App| {
        a.desk().is_some_and(|d| {
            d.strips[0]
                .inputs
                .iter()
                .any(|i| i.parts.is_some_and(|p| p.state == Parts::Present))
        })
    };
    assert!(common::wait_for(&mut app, Duration::from_secs(30), present));
    let on = heard(&app) + u64::from(RATE);
    let off = on + 3 * u64::from(RATE) / 2;
    assert!(
        off + u64::from(RATE) < 5 * u64::from(RATE),
        "too late to set it"
    );
    app.run(Effect::Desk(DeskChange::MuteAt(StripKind::Drums, true, on)));
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
        off,
    )));
    // Then a seek back to 2 s, and the drums muted a second after the
    // first position told from there: Position's output frame and the
    // mixer's stream frame are the same count after a seek too.
    assert!(common::wait_for(
        &mut app,
        Duration::from_secs(30),
        |a| heard(a) > off + u64::from(RATE) / 2
    ));
    app.run(Effect::Engine(jodel_playback::Command::Seek(SEEK)));
    // Two positions told from there (as the stem keys wait for: the first
    // can come before the mixer has mixed any of it again).
    let fresh = |a: &App| {
        let p = &a.state.playing;
        p.speed.is_some()
            && p.anchor
                .is_some_and(|(_, _, seconds)| (SEEK..SEEK + 1.0).contains(&seconds))
    };
    assert!(common::wait_for(&mut app, Duration::from_secs(30), fresh));
    let (_, frame, seconds) = app.state.playing.anchor.unwrap();
    let again = frame + u64::from(RATE);
    app.run(Effect::Desk(DeskChange::MuteAt(
        StripKind::Drums,
        true,
        again,
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
    let (_, _, heard_then) = app.state.playing.anchor.unwrap();
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
    // The switch on that very frame (its first step there).
    let main_to = (on..off).find(|&f| got[2 * f] != chord(f, RATE));
    assert!(
        main_to.is_some_and(|f| f.abs_diff(on) <= 1),
        "main to {main_to:?}, the frame {on}"
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
