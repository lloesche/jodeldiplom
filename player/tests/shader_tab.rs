//! The Shader tab through the core and the window, in iced's simulator:
//! it is in the tab cycle after the Synth tab; its keys pick the presets
//! (searched by `/`), select, move and take out the stack's passes, copy
//! the stack as JSON and keep it as the user's own; a knob's slider moves
//! its knob and its driver's amount; the look in force is the tab's while
//! it shows; and on the GPU: the look draws over the backdrop, the same
//! look the same bytes, none the backdrop as it was.

mod common;

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use iced::mouse;
use iced::widget::Id;
use iced::{window, Point, Size};
use iced_test::simulator::Simulator;
use jodel_looks::{presets, Driver, Look};
use jodel_output::SimClock;
use jodeldiplom::app::{App, AppOptions, Output};
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::shader_tab::{self, ShaderAction};
use jodeldiplom::core::{update, Action, Effect, Side, State};
use jodeldiplom::gui::backdrop::looks::LookCue;
use jodeldiplom::gui::backdrop::{Cue, Visualiser};
use jodeldiplom::gui::pointer::Grab;
use jodeldiplom::gui::shader;
use jodeldiplom::gui::window::{Gui, Message};

const SIZE: Size = Size::new(1280.0, 800.0);

/// One headless renderer at a time (as the window's other tests).
fn gpu() -> MutexGuard<'static, ()> {
    static GPU: Mutex<()> = Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

/// The Shader tab shown.
fn shader_tab() -> State {
    let mut state = common::playing_state(120, 36);
    state.side = Side::Shader;
    state
}

/// Tab goes round to the Shader tab after the Synth tab; the look in force
/// is the tab's while it shows (none in the classic theme elsewhere); its
/// keys pick the presets one by one and by family, select, move and take
/// out the stack's passes; `/` searches the presets as it is typed, enter
/// keeps the search and picks the first found, esc clears it; the stack
/// is copied as JSON (reading back as it is) and kept as the user's own,
/// under an id of its own.
#[test]
fn the_shader_tab_is_in_the_tab_cycle_and_its_keys_edit_the_stack() {
    let mut state = common::playing_state(120, 36);
    state.side = Side::Synth;
    assert_eq!(state.look(), None);
    update(&mut state, Action::Op(Op::ToggleSide));
    assert_eq!(state.side, Side::Shader);
    assert_eq!(state.look(), Some(&presets::all()[0]));
    update(&mut state, Action::Op(Op::Down));
    assert_eq!(state.shader.preset, 1);
    assert_eq!(state.look(), Some(&presets::all()[1]));
    // The stack's passes: the next selected, moved earlier, taken out.
    let before: Vec<String> = shader_tab::look(&state)
        .layers
        .iter()
        .map(|l| l.pass.clone())
        .collect();
    update(&mut state, Action::Op(Op::ShaderLayer(1)));
    assert_eq!(state.shader.layer, 1);
    update(&mut state, Action::Op(Op::ShaderMove(-1)));
    let moved: Vec<String> = shader_tab::look(&state)
        .layers
        .iter()
        .map(|l| l.pass.clone())
        .collect();
    assert_eq!((&moved[0], &moved[1]), (&before[1], &before[0]));
    assert_eq!(state.shader.layer, 0);
    update(&mut state, Action::Op(Op::ShaderRemove));
    assert_eq!(shader_tab::look(&state).layers.len(), before.len() - 1);
    // The search, typed.
    update(&mut state, Action::Op(Op::Filter));
    for c in "noir".chars() {
        update(&mut state, Action::Type(c));
    }
    let found: Vec<&str> = shader_tab::listed(&state)
        .iter()
        .map(|(_, l)| l.id.as_str())
        .collect();
    assert_eq!(found, ["noir"]);
    update(&mut state, Action::Op(Op::Accept));
    assert_eq!(shader_tab::look(&state).id, "noir");
    update(&mut state, Action::Op(Op::Back));
    assert_eq!(state.shader.search, None);
    // Copied, and kept.
    let fx = update(&mut state, Action::Op(Op::ShaderCopy));
    let copied = fx.iter().find_map(|e| match e {
        Effect::CopyText(t) => Some(t.clone()),
        _ => None,
    });
    assert_eq!(
        Look::from_json(&copied.expect("copied")).ok().as_ref(),
        Some(shader_tab::look(&state))
    );
    let fx = update(&mut state, Action::Op(Op::ShaderKeep));
    let kept = fx.iter().find_map(|e| match e {
        Effect::KeepLook(l) => Some(l.clone()),
        _ => None,
    });
    let kept = kept.expect("kept");
    assert_eq!(kept.id, "my-noir");
    assert_eq!(state.shader.kept, vec![kept.clone()]);
    assert_eq!(shader_tab::listed(&state)[0], (shader_tab::YOURS, &kept));
    // The tab left, the look in force is the theme's again; a song's own,
    // where it has one.
    update(&mut state, Action::Op(Op::ToggleSide));
    assert_eq!(state.look(), None);
    let own = Look {
        id: "made-for-this-song".into(),
        ..presets::named("noir").unwrap().clone()
    };
    state.karaoke.look = Some(Arc::new(own.clone()));
    assert_eq!(state.look(), Some(&own));
}

/// A knob's driver steps round (none, then every driver) and its amount
/// is set by its slider; a pass added goes after the one selected.
#[test]
fn a_knob_s_driver_is_chosen_and_a_pass_added() {
    let mut state = shader_tab();
    let act = |s: &mut State, a| update(s, Action::Shader(a));
    act(&mut state, ShaderAction::Pick(5));
    assert_eq!(shader_tab::look(&state).id, "noir");
    act(&mut state, ShaderAction::Layer(2));
    let (_, layer, pass) = shader_tab::layer(&state).unwrap();
    assert_eq!(pass.name, "grain");
    let natural = shader_tab::drive(layer, pass, 0);
    assert_eq!(natural, Some((Driver::Bass, 0.5)), "the pass's own");
    act(&mut state, ShaderAction::Driver { knob: 0, by: 1 });
    let (_, layer, pass) = shader_tab::layer(&state).unwrap();
    let next = Driver::ALL[Driver::Bass as usize + 1];
    assert_eq!(shader_tab::drive(layer, pass, 0), Some((next, 0.5)));
    act(
        &mut state,
        ShaderAction::Amount {
            knob: 0,
            share: 0.0,
        },
    );
    let (_, layer, pass) = shader_tab::layer(&state).unwrap();
    assert_eq!(shader_tab::drive(layer, pass, 0), Some((next, -1.0)));
    act(&mut state, ShaderAction::SlideEnd);
    act(&mut state, ShaderAction::Add("snow"));
    let look = shader_tab::look(&state);
    assert_eq!(look.layers[3].pass, "snow");
    assert_eq!(state.shader.layer, 3);
    assert!(look.check().notes.is_empty());
}

fn player() -> App {
    let dir = common::temp_dir("shader-tab");
    let mut options = AppOptions::new(dir);
    options.output = Output::Null(SimClock::Realtime);
    options.devices = Arc::new(|| Ok(Vec::new()));
    options.mic = Arc::new(|_| Err("no microphone".into()));
    options.mics = Arc::new(|| Ok(Vec::new()));
    App::new(options)
}

fn hear(gui: &mut Gui, event: iced::Event) {
    let _ = gui.update(Message::Event(event, window::Id::unique()));
}

fn simulator(gui: &Gui) -> Simulator<'_, Message> {
    Simulator::with_size(jodeldiplom::gui::iced_settings(), SIZE, gui.view())
}

/// A knob's slider takes a press (the knob there, held); the pointer then
/// moves it from anywhere; let go, it is no longer held.
#[test]
fn a_shader_slider_moves_its_knob() {
    let _gpu = gpu();
    let mut gui = Gui::headless(player());
    hear(&mut gui, iced::Event::Window(window::Event::Resized(SIZE)));
    hear(&mut gui, iced::Event::Window(window::Event::Rescaled(1.0)));
    let mut state = shader_tab();
    state.shader.look = presets::named("noir").cloned();
    state.shader.layer = 2;
    gui.show(state);
    let _ = gui.update(Message::Frame(Instant::now()));
    let id: Id = shader::knob_id(0, false).into();
    let track = simulator(&gui).find(id).unwrap().bounds();
    let mut sim = simulator(&gui);
    sim.point_at(Point::new(track.x + track.width - 0.5, track.center_y()));
    sim.simulate([iced::Event::Mouse(mouse::Event::ButtonPressed(
        mouse::Button::Left,
    ))]);
    let pressed: Vec<Message> = sim.into_messages().collect();
    assert!(
        matches!(&pressed[..], [Message::Grab(Grab::ShaderKnob(0, false, b), _)] if *b == track),
        "{pressed:?}"
    );
    for m in pressed {
        let _ = gui.update(m);
    }
    let amount = |g: &Gui| {
        let s = g.state().unwrap();
        let (_, layer, pass) = shader_tab::layer(s).unwrap();
        shader_tab::value(layer, pass, 0)
    };
    assert_eq!(gui.state().unwrap().shader.dragging, Some((0, false)));
    assert!((amount(&gui) - 1.0).abs() < 0.01, "{}", amount(&gui));
    hear(
        &mut gui,
        iced::Event::Mouse(mouse::Event::CursorMoved {
            position: Point::new(2.0, 300.0),
        }),
    );
    assert_eq!(amount(&gui), 0.0);
    hear(
        &mut gui,
        iced::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)),
    );
    assert_eq!(gui.state().unwrap().shader.dragging, None);
}

/// On the GPU: the backdrop without a look, then with the noir look over
/// it (another picture: grey), the same look again (the same bytes), and
/// without one again (the backdrop as it was, byte for byte).
#[test]
fn a_look_draws_over_the_backdrop_and_none_leaves_it_as_it_was() {
    let _gpu = gpu();
    let Ok(mut gpu) = jodeldiplom::gui::present::Offscreen::new(false) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let (w, h) = (320u32, 180u32);
    let noir = Arc::new(presets::named("noir").unwrap().check());
    let curves = Arc::new(jodel_looks::demo::Song::new(30.0, 120.0).curves());
    let frame = |look: Option<&Arc<jodel_looks::Checked>>,
                 gpu: &mut jodeldiplom::gui::present::Offscreen| {
        let mut v = Visualiser::default();
        let mut last = None;
        for f in 0..3 {
            let t = 5.0 + f64::from(f) / 60.0;
            let cue = Cue {
                look: look.map(|l| LookCue {
                    look: Arc::clone(l),
                    t,
                    signals: curves.at(t),
                    curves: Some(Arc::clone(&curves)),
                    seed: jodel_looks::seed(&l.id, "a song"),
                    band: Some((0.68, 0.86)),
                }),
                ..Cue::default()
            };
            last = Some(v.step(&cue, (w, h), 1.0 / 60.0));
        }
        gpu.set_backdrop(last.as_ref(), w, h);
        gpu.set_stage(None, w, h);
        gpu.show_frame(w, h).expect("a frame")
    };
    let plain = frame(None, &mut gpu);
    let looked = frame(Some(&noir), &mut gpu);
    assert_ne!(plain, looked);
    let grey = looked.iter().all(|p| {
        let (r, g, b) = ((p >> 16) & 255, (p >> 8) & 255, p & 255);
        r.abs_diff(g) <= 6 && g.abs_diff(b) <= 6
    });
    assert!(grey, "noir is black and white");
    assert_eq!(
        frame(Some(&noir), &mut gpu),
        looked,
        "the same look, the same bytes"
    );
    assert_eq!(frame(None, &mut gpu), plain, "none: the backdrop as it was");
}
