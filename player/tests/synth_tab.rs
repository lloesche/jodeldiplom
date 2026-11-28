//! The Synth tab through the window, in iced's simulator: it opens from
//! the tabs; a click on a sound plays it (the effects input takes its
//! notes); a slider moves its parameter in the patch shown, kept then
//! reverted; the keyboard plays a note held by the pointer and by the
//! computer's keys; and through the core: the tab in the tab cycle, the
//! sounds and the keys stepped by the keys.

mod common;

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use iced::keyboard::Key;
use iced::mouse;
use iced::widget::Id;
use iced::{window, Point, Size};
use iced_test::simulator::{press_key, release_key, Simulator};
use jodel_output::SimClock;
use jodeldiplom::app::{App, AppOptions, Output};
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::synth_tab::{self, Bench, SynthAction};
use jodeldiplom::core::{update, Action, Effect, Side, State};
use jodeldiplom::gui::pointer::Grab;
use jodeldiplom::gui::synth;
use jodeldiplom::gui::window::{Gui, Message};
use jodeldiplom::sfx::bench::{Held, What, SOUNDS};

const SIZE: Size = Size::new(1280.0, 800.0);

/// One headless renderer at a time (as the window's other tests).
fn gpu() -> MutexGuard<'static, ()> {
    static GPU: Mutex<()> = Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

/// A player playing into the null output, no devices, no microphone.
fn player() -> App {
    let mut options = AppOptions::new(jodel_testkit::thread_dir("synth-tab"));
    options.output = Output::Null(SimClock::Realtime);
    options.devices = Arc::new(|| Ok(Vec::new()));
    options.mic = Arc::new(|_| Err("no microphone".into()));
    options.mics = Arc::new(|| Ok(Vec::new()));
    App::new(options)
}

/// A window of [`SIZE`] round a player showing `state`, a frame drawn.
fn window(state: State) -> Gui {
    let mut gui = Gui::headless(player());
    hear(&mut gui, iced::Event::Window(window::Event::Resized(SIZE)));
    hear(&mut gui, iced::Event::Window(window::Event::Rescaled(1.0)));
    gui.show(state);
    let _ = gui.update(Message::Frame(Instant::now()));
    gui
}

/// The Synth tab shown.
fn synth_tab() -> State {
    let mut state = common::playing_state(120, 36);
    state.side = Side::Synth;
    state
}

fn hear(gui: &mut Gui, event: iced::Event) {
    let _ = gui.update(Message::Event(event, window::Id::unique()));
}

fn simulator(gui: &Gui) -> Simulator<'_, Message> {
    Simulator::with_size(jodeldiplom::gui::iced_settings(), SIZE, gui.view())
}

/// Where the widget `id` is.
fn bounds(gui: &Gui, id: impl Into<Id>) -> iced::Rectangle {
    simulator(gui).find(id.into()).unwrap().bounds()
}

/// The messages a click on the widget `id` sends, taken by the window.
fn click(gui: &mut Gui, id: impl Into<Id>) -> Vec<Message> {
    let mut sim = simulator(gui);
    sim.click(id.into()).unwrap();
    let messages: Vec<Message> = sim.into_messages().collect();
    send(gui, messages.clone());
    messages
}

/// The messages a press at `at` sends, taken by the window.
fn press_at(gui: &mut Gui, at: Point) -> Vec<Message> {
    let mut sim = simulator(gui);
    sim.point_at(at);
    sim.simulate([iced::Event::Mouse(mouse::Event::ButtonPressed(
        mouse::Button::Left,
    ))]);
    let messages: Vec<Message> = sim.into_messages().collect();
    send(gui, messages.clone());
    messages
}

fn send(gui: &mut Gui, messages: Vec<Message>) {
    for message in messages {
        let _ = gui.update(message);
    }
}

fn moved_to(gui: &mut Gui, at: Point) {
    hear(
        gui,
        iced::Event::Mouse(mouse::Event::CursorMoved { position: at }),
    );
}

fn let_go(gui: &mut Gui) {
    hear(
        gui,
        iced::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)),
    );
}

fn shown(gui: &Gui) -> &State {
    gui.state().unwrap()
}

/// The commands the effects input took.
fn taken(gui: &Gui) -> u64 {
    gui.app().and_then(App::effects).map_or(0, |v| v.commands)
}

/// Frames drawn until `done` (at most five seconds).
fn frames_until(gui: &mut Gui, what: &str, done: impl Fn(&Gui) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !done(gui) {
        assert!(Instant::now() < deadline, "{what}");
        let _ = gui.update(Message::Frame(Instant::now()));
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// The patch the tab shows.
fn patch_shown(state: &State) -> jodel_synth::Patch {
    synth_tab::patch(state, synth_tab::atom(state))
}

/// The tab opens from its tab; a click on a sound selects it and plays
/// it, the effects input taking its notes.
#[test]
fn the_synth_tab_opens_and_a_click_plays_its_sound() {
    let _gpu = gpu();
    let mut gui = window(common::playing_state(120, 36));
    let mut sim = simulator(&gui);
    sim.click("Synth").unwrap();
    let messages: Vec<Message> = sim.into_messages().collect();
    assert!(
        matches!(&messages[..], [Message::Act(Action::ShowSide(Side::Synth))]),
        "{messages:?}"
    );
    send(&mut gui, messages);
    assert_eq!(shown(&gui).side, Side::Synth);
    // The synth on the effects input, the device open.
    frames_until(&mut gui, "the effects' synth live", |g| {
        g.app().and_then(App::effects).is_some_and(|v| v.live > 0)
    });
    // A sound in view: the run of words to the first rung.
    let run = (SOUNDS.iter()).position(|s| s.what == What::Run).unwrap();
    let before = taken(&gui);
    let clicked = click(&mut gui, synth::sound_id(run));
    assert!(
        matches!(
            &clicked[..],
            [Message::Act(Action::Synth(SynthAction::Pick(k)))] if *k == run
        ),
        "{clicked:?}"
    );
    assert_eq!(shown(&gui).synth.sound, run);
    frames_until(&mut gui, "its notes taken", |g| taken(g) > before + 5);
}

/// A parameter's slider takes a press (the parameter there, held); the
/// pointer then moves it from anywhere, the patch shown edited; Keep keeps
/// it, Revert brings the built-in one back.
#[test]
fn a_synth_slider_edits_the_patch_kept_then_reverted() {
    let _gpu = gpu();
    let mut gui = window(synth_tab());
    let built = patch_shown(shown(&gui));
    let knob = built.find("envelope decay").unwrap();
    let (decay, spec) = (knob.k, knob.spec);
    let track = bounds(&gui, synth::knob_id(&spec));
    let at = Point::new(track.x + track.width - 0.5, track.center_y());
    let pressed = press_at(&mut gui, at);
    assert!(
        matches!(&pressed[..], [Message::Grab(Grab::SynthKnob(k, b), _)] if *k == decay && *b == track),
        "{pressed:?}"
    );
    assert_eq!(shown(&gui).synth.dragging, Some(decay));
    assert!(patch_shown(shown(&gui)).adsr.decay > 7.9);
    moved_to(&mut gui, Point::new(4.0, 300.0));
    assert_eq!(patch_shown(shown(&gui)).adsr.decay, 0.0);
    moved_to(&mut gui, Point::new(track.x + track.width / 2.0, 10.0));
    let_go(&mut gui);
    assert_eq!(shown(&gui).synth.dragging, None);
    let edited = patch_shown(shown(&gui));
    assert!(
        (edited.adsr.decay - 2.0).abs() < 0.05,
        "{}",
        edited.adsr.decay
    );
    assert_eq!(edited.adsr.attack, built.adsr.attack);
    // Kept: written down as it is.
    let _ = gui.update(Message::Frame(Instant::now()));
    click(&mut gui, "synth-keep");
    let state = shown(&gui);
    let (theme, atom) = (state.bench_theme(), synth_tab::atom(state));
    assert_eq!(state.patches.kept(theme, atom), Some(edited));
    // Reverted: as built, nothing kept.
    let _ = gui.update(Message::Frame(Instant::now()));
    click(&mut gui, "synth-revert");
    assert!(shown(&gui).patches.is_empty());
    assert_eq!(patch_shown(shown(&gui)), built);
}

/// The keyboard plays the patch shown: a press holds the key under the
/// pointer (sliding to the next as it moves) until it is let go of; a
/// computer key holds its note until it is let go of; the effects input
/// takes them.
#[test]
fn the_synth_keyboard_plays_a_note() {
    let _gpu = gpu();
    let mut gui = window(synth_tab());
    frames_until(&mut gui, "the effects' synth live", |g| {
        g.app().and_then(App::effects).is_some_and(|v| v.live > 0)
    });
    let before = taken(&gui);
    let keys = bounds(&gui, synth::PIANO);
    let white = keys.width / 14.0;
    let low = keys.y + keys.height * 0.9;
    let pressed = press_at(&mut gui, Point::new(keys.x + white / 2.0, low));
    assert!(
        matches!(&pressed[..], [Message::Grab(Grab::Piano(_), _)]),
        "{pressed:?}"
    );
    assert_eq!(shown(&gui).synth.mouse, Some(60.0));
    moved_to(&mut gui, Point::new(keys.x + white * 1.5, low));
    assert_eq!(shown(&gui).synth.mouse, Some(62.0));
    let_go(&mut gui);
    assert_eq!(shown(&gui).synth.mouse, None);
    hear(&mut gui, press_key(Key::Character("d".into()), None));
    assert_eq!(shown(&gui).synth.held, [(4, 64.0)]);
    hear(&mut gui, release_key(Key::Character("d".into())));
    assert!(shown(&gui).synth.held.is_empty());
    frames_until(&mut gui, "the notes taken", |g| taken(g) >= before + 6);
}

/// Tab goes round to the Synth tab after the Mixer; its keys step the
/// sounds and the moments, Enter plays the sound, a computer key held is
/// let go of as the tab is left, and the key and the tempo step round
/// and back to the song's.
#[test]
fn the_synth_tab_is_in_the_tab_cycle_and_its_keys_step() {
    let mut state = common::playing_state(120, 36);
    state.side = Side::Mixer;
    update(&mut state, Action::Op(Op::ToggleSide));
    assert_eq!(state.side, Side::Synth);
    update(&mut state, Action::Op(Op::Down));
    assert_eq!(state.synth.sound, 1);
    update(&mut state, Action::Op(Op::PageDown));
    let first_rung = SOUNDS.iter().position(|s| matches!(s.what, What::Rung(_)));
    assert_eq!(Some(state.synth.sound), first_rung);
    let fx = update(&mut state, Action::Op(Op::Open));
    assert!(
        fx.iter()
            .any(|e| matches!(e, Effect::Bench(Bench::Play { .. }))),
        "{fx:?}"
    );
    let fx = update(&mut state, Action::Op(Op::SynthNote(0)));
    let atom = synth_tab::atom(&state);
    let theme = state.bench_theme();
    let held = Held::Key(0);
    let down = Bench::Down {
        atom,
        note: 60.0,
        held,
        theme,
    };
    assert!(fx.contains(&Effect::Bench(down)), "{fx:?}");
    update(&mut state, Action::Op(Op::SynthOctave(1)));
    let fx = update(&mut state, Action::Op(Op::ToggleSide));
    assert_eq!(state.side, Side::Shader);
    assert!(fx.contains(&Effect::Bench(Bench::Up { held })), "{fx:?}");
    state.side = Side::Synth;
    update(&mut state, Action::Op(Op::SynthKey(1)));
    assert_eq!(
        synth_tab::key(&state).map(|k| k.name()),
        Some("C major".into())
    );
    update(&mut state, Action::Op(Op::SynthKey(0)));
    assert_eq!(state.synth.key, None);
    let song = synth_tab::tempo(&state);
    update(&mut state, Action::Op(Op::SynthTempo(1)));
    assert_eq!(state.synth.tempo, Some((song / 5.0).round() * 5.0 + 5.0));
    update(&mut state, Action::Op(Op::SynthTempo(0)));
    assert_eq!(state.synth.tempo, None);
    update(&mut state, Action::Op(Op::SynthTheme));
    assert_eq!(state.bench_theme().name, "spooky");
}

/// A key held while the theme compared changes is let go of all the
/// same, its note alone: what lets it go names its key, not the theme nor
/// the patch it was pressed on.
#[test]
fn a_key_held_across_a_theme_switch_is_let_go_of() {
    let mut state = common::playing_state(120, 36);
    state.side = Side::Synth;
    update(&mut state, Action::Op(Op::SynthNote(4)));
    update(&mut state, Action::Op(Op::SynthNote(7)));
    update(&mut state, Action::Op(Op::SynthTheme));
    let fx = update(&mut state, Action::Released(Op::SynthNote(4)));
    let ups: Vec<&Effect> = (fx.iter())
        .filter(|e| matches!(e, Effect::Bench(Bench::Up { .. })))
        .collect();
    assert_eq!(ups, [&Effect::Bench(Bench::Up { held: Held::Key(4) })]);
    assert_eq!(state.synth.held, [(7, 67.0)]);
}
