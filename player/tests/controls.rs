//! Karaoke mode's controls (K3b): the voice slider, the seek bar and the
//! keys over the picture's bottom, near the pointer only, fading in and
//! out; never START (the stage's pill, K3c).

mod common;

use common::karaoke_tab as tab;
use jodel_playback::{Command, PlayState};
use jodeldiplom::core::karaoke::controls::{self, FADE_MS, LINGER_MS, ROWS};
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::layout::{self, Layout};
use jodeldiplom::core::mouse::{self, MouseButton, MouseInput, MouseKind, Target};
use jodeldiplom::core::{update, Action, Effect, State};

fn op(state: &mut State, op: Op) -> Vec<Effect> {
    update(state, Action::Op(op))
}

fn engine(fx: &[Effect]) -> Vec<&Command> {
    fx.iter()
        .filter_map(|e| match e {
            Effect::Engine(c) => Some(c),
            _ => None,
        })
        .collect()
}

/// In the mode the controls come up only with the pointer near the bottom
/// (staying while it is there; going 1.5 s after it leaves, the window
/// too), fading in and out, over the picture: the lines stay where they
/// are. Under the slider, the seek bar seeks, except while the song is
/// held for START. They have no START: that is the stage's pill.
#[test]
fn the_controls_come_up_near_the_bottom_over_the_lines_with_a_seek_bar() {
    use jodeldiplom::view::karaoke::{picture, KaraokePicture};
    let mut state = tab();
    op(&mut state, Op::KaraokeEnter);
    // A song playing (nothing waits for START), the controls gone.
    state.karaoke.held = None;
    state.karaoke.countdown = None;
    state.playing.track = Some(common::now_track(44_100, state.engine_settings(), 2));
    state.playing.state = PlayState::Playing;
    state.playing.position = 30.0;
    let later = state.now_ms + 10_000;
    update(&mut state, Action::Tick(later));
    assert!(!controls::shown(&state));
    let rows = state.size.1;
    let moved = |row: u16| MouseInput {
        kind: MouseKind::Moved,
        col: 10,
        row,
    };
    let point = |state: &mut State, input: MouseInput| {
        if let Some(a) = mouse::action(state, input) {
            update(state, a)
        } else {
            Vec::new()
        }
    };
    let pictured = |state: &State| -> (KaraokePicture, Option<u16>) {
        let Layout::Karaoke(k) = layout::compute(state) else {
            panic!("the mode")
        };
        (picture(state, &k, 30.0), k.controls.map(|c| c.bar.y))
    };
    let (hidden, none) = pictured(&state);
    assert_eq!(none, None);
    // Above the bottom: nothing.
    point(&mut state, moved(rows / 2));
    assert!(!controls::shown(&state) && !controls::fading(&state));
    // Near it: the controls fading in over the bottom (laid out, and taking
    // clicks, as soon as they show at all); the lines where they were. No
    // START among them while a song plays (nor waits).
    point(&mut state, moved(rows - 2));
    assert!(!controls::shown(&state) && controls::fading(&state));
    let at = state.now_ms + FADE_MS / 2;
    update(&mut state, Action::Tick(at));
    let (shown, bar) = pictured(&state);
    assert_eq!(bar, Some(rows - ROWS));
    assert_eq!(shown.bar.y, rows - ROWS);
    assert_eq!(shown.controls, Some(0.5));
    assert_eq!(shown.start, None);
    let faded_in = state.now_ms + FADE_MS / 2;
    update(&mut state, Action::Tick(faded_in));
    assert_eq!(pictured(&state).0.controls, Some(1.0));
    // They stay while the pointer rests there.
    update(&mut state, Action::Tick(faded_in + 60_000));
    assert!(controls::shown(&state) && !controls::fading(&state));
    // The same lines in the same picture: the window lays them out on the
    // whole screen, the controls never move them.
    assert_eq!((hidden.area, &hidden.lines), (shown.area, &shown.lines));
    // The seek bar under the slider: a click and release seeks there.
    let Layout::Karaoke(k) = layout::compute(&state) else {
        panic!()
    };
    let c = k.controls.unwrap();
    assert!(c.seek.y > c.sliders[0].row.y && c.seek.width > 10, "{c:?}");
    let mid = c.seek.x + c.seek.width / 2;
    assert_eq!(mouse::hit(&state, mid, c.seek.y), Some(Target::Bar));
    let click = |kind| MouseInput {
        kind,
        col: mid,
        row: c.seek.y,
    };
    point(&mut state, click(MouseKind::Down(MouseButton::Left)));
    assert!(state.seek_drag.is_some());
    let fx = point(&mut state, click(MouseKind::Up(MouseButton::Left)));
    assert!(
        engine(&fx).iter().any(|c| matches!(c, Command::Seek(_))),
        "{fx:?}"
    );
    // Held for START: the bar does nothing.
    state.karaoke.held = Some(2);
    point(&mut state, click(MouseKind::Down(MouseButton::Left)));
    let fx = point(&mut state, click(MouseKind::Up(MouseButton::Left)));
    assert!(!engine(&fx).iter().any(|c| matches!(c, Command::Seek(_))));
    state.karaoke.held = None;
    // Back up above the bottom: they stay a while, then fade out; gone,
    // they take no clicks.
    point(&mut state, moved(1));
    let left = state.now_ms;
    assert_eq!(state.due(), Some(left + LINGER_MS));
    update(&mut state, Action::Tick(left + LINGER_MS - 1));
    assert!(controls::shown(&state) && !controls::fading(&state));
    update(&mut state, Action::Tick(left + LINGER_MS + 100));
    assert!(controls::fading(&state));
    assert_eq!(pictured(&state).0.controls, Some(0.5));
    update(&mut state, Action::Tick(left + LINGER_MS + FADE_MS));
    assert!(!controls::shown(&state) && state.due().is_none());
    assert_eq!(mouse::hit(&state, mid, c.seek.y), None);
    // Off the window from near the bottom: the same.
    point(&mut state, moved(rows - 1));
    let at = state.now_ms + FADE_MS;
    update(&mut state, Action::Tick(at));
    assert!(controls::shown(&state));
    let out = MouseInput {
        kind: MouseKind::Out,
        col: 10,
        row: rows - 1,
    };
    assert!(matches!(
        mouse::action(&state, out),
        Some(Action::PointerLeft)
    ));
    point(&mut state, out);
    let at = state.now_ms + LINGER_MS + FADE_MS;
    update(&mut state, Action::Tick(at));
    assert!(!controls::shown(&state));
}
