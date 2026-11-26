//! The key the song plays in (`core::key`): `-`, `+` and `0` on the
//! Karaoke tab and the Mixer tab change it, shown at once (the tab's Key
//! slider, the PLAYER strip) and sent to the engine; the engine's word on
//! it is taken (a change refused, a song picked putting it back at 0); the
//! song's notes and the scoring follow the key heard.

mod common;

use std::sync::Arc;

use iced::{Point, Rectangle, Size};
use jodel_playback::{Command, Event};
use jodeldiplom::core::desk::{Channel, StripKind};
use jodeldiplom::core::keymap::{self, Key, KeyCode, Meaning, Op};
use jodeldiplom::core::sliders::Slider;
use jodeldiplom::core::{update, Action, Effect, Side, State};
use jodeldiplom::gui::mixer::said;
use jodeldiplom::gui::pointer::Grab;
use jodeldiplom::view::karaoke::{Control, Tab};

fn op(state: &mut State, op: Op) -> Vec<Effect> {
    update(state, Action::Op(op))
}

/// The keys sent to the engine.
fn keys_sent(fx: &[Effect]) -> Vec<i8> {
    fx.iter()
        .filter_map(|e| match e {
            Effect::Engine(Command::Key { semitones }) => Some(*semitones),
            _ => None,
        })
        .collect()
}

/// What `c` does where `state` is.
fn meant(state: &State, c: char) -> Option<Op> {
    match keymap::lookup(state, Key::plain(KeyCode::Char(c))) {
        Some(Meaning::Op(op)) => Some(op),
        _ => None,
    }
}

/// On the Karaoke tab `-` `+` `0` move the key a semitone or put the
/// song's own back, within ±6: shown on the tab's Key slider at once and
/// sent; the engine's word is what stays.
#[test]
fn the_keys_change_the_key_on_the_karaoke_tab_and_show_it() {
    let mut state = common::transition_state(120, 40);
    state.side = Side::Karaoke;
    for (c, want) in [('-', Op::KeyDown), ('+', Op::KeyUp), ('=', Op::KeyUp)] {
        assert_eq!(meant(&state, c), Some(want), "{c}");
    }
    assert_eq!(meant(&state, '0'), Some(Op::KeyReset));
    let fx = op(&mut state, Op::KeyUp);
    assert_eq!(keys_sent(&fx), [1]);
    let fx = op(&mut state, Op::KeyUp);
    assert_eq!(keys_sent(&fx), [2]);
    assert_eq!(state.playing.key, 2);
    assert_eq!(Slider::Key.text(&state).trim(), "+2");
    let shown = Tab::of(&state).to_string();
    assert!(
        shown.contains("[slider] Key 67 % +2"),
        "the Key slider shows: {shown}"
    );
    // Never past ±6: up there it says so and sends nothing.
    for _ in 0..6 {
        op(&mut state, Op::KeyUp);
    }
    assert_eq!(state.playing.key, 6);
    assert!(keys_sent(&op(&mut state, Op::KeyUp)).is_empty());
    let fx = op(&mut state, Op::KeyReset);
    assert_eq!(keys_sent(&fx), [0]);
    let fx = op(&mut state, Op::KeyDown);
    assert_eq!(keys_sent(&fx), [-1]);
    assert_eq!(Slider::Key.text(&state).trim(), "−1");
    // The engine refused it (a transition set up): its word stays.
    update(&mut state, Action::Engine(Event::Key { semitones: 0 }));
    assert_eq!(state.playing.key, 0);
}

/// The Key slider is the tab's third row (below the delay's sync): the
/// arrows pick it and step it a semitone, and a click on its track puts
/// the key there.
#[test]
fn the_key_slider_is_a_row_of_the_tab() {
    let mut state = common::transition_state(120, 40);
    state.side = Side::Karaoke;
    op(&mut state, Op::Down);
    op(&mut state, Op::Down);
    assert_eq!(state.karaoke.focus, Slider::Key);
    let fx = op(&mut state, Op::SliderUp);
    assert_eq!(keys_sent(&fx), [1]);
    let fx = op(&mut state, Op::SliderFineDown);
    assert_eq!(keys_sent(&fx), [0]);
    op(&mut state, Op::Up);
    assert_eq!(state.karaoke.focus, Slider::Delay);
    // Its row: the last, after the voice, the delay and the sync's.
    let rows: Vec<Option<Slider>> = Tab::of(&state)
        .controls
        .iter()
        .map(|c| match c {
            Control::Slider(s) => Some(s.slider),
            Control::Sync(_) => None,
        })
        .collect();
    assert_eq!(
        rows,
        [
            Some(Slider::Voice),
            Some(Slider::Delay),
            None,
            Some(Slider::Key)
        ]
    );
    // A press at its track's right end: +6.
    let track = Rectangle::new(Point::new(400.0, 200.0), Size::new(300.0, 16.0));
    let press = Grab::Slider(Slider::Key, track).moved(Point::new(700.0, 208.0), &state);
    let fx = update(&mut state, press.expect("an action"));
    assert_eq!(keys_sent(&fx), [6]);
    update(&mut state, Action::KaraokeSlideEnd);
}

/// On the Mixer tab the same keys change it, and the PLAYER strip shows
/// the key on the head on air while it is not the song's own.
#[test]
fn the_mixer_tab_changes_the_key_and_its_player_strip_shows_it() {
    let mut state = common::desk_state(120, 36);
    assert_eq!(meant(&state, '+'), Some(Op::KeyUp));
    assert_eq!(meant(&state, '0'), Some(Op::KeyReset));
    let lead = |state: &State| {
        let player = Channel::Strip(StripKind::Player);
        said::text(state, player)[1].clone()
    };
    assert_eq!(lead(&state), "● LEAD");
    let fx = op(&mut state, Op::KeyDown);
    assert_eq!(keys_sent(&fx), [-1]);
    op(&mut state, Op::KeyDown);
    assert_eq!(lead(&state), "● LEAD −2");
    op(&mut state, Op::KeyReset);
    assert_eq!(lead(&state), "● LEAD");
}

/// The song's notes and the scoring are in the key heard: a position
/// heard two semitones up moves the notes there, so singing in the new key
/// hits them; and picking another song (the engine's key back at 0) moves
/// them back.
#[test]
fn the_notes_and_the_scoring_follow_the_key_heard() {
    let mut state = common::transition_state(120, 40);
    state.side = Side::Karaoke;
    let (lyrics, melody) = two_note_song();
    let singing = &mut state.karaoke.singing;
    singing.prepare(
        std::path::Path::new("a"),
        &lyrics,
        &jodeldiplom::core::singing::melodic(&melody),
    );
    let note = |state: &State| state.karaoke.singing.word_at(0.5).map(|(_, c)| c);
    assert_eq!(note(&state), Some(6000.0));
    let index = state.playing.track.as_ref().map_or(0, |t| t.index);
    let heard = |key: i8| {
        Action::Engine(Event::Position(jodel_playback::Position {
            index,
            head: 0,
            seconds: 0.5,
            frame: 22_050,
            output_frame: 22_050,
            key,
        }))
    };
    update(&mut state, heard(2));
    assert_eq!(state.karaoke.singing.shift(), 2);
    assert_eq!(note(&state), Some(6200.0));
    for i in 0..250 {
        let t = f64::from(i) / 100.0;
        let sung = if t < 1.0 { 6200.0 } else { 6600.0 };
        state.karaoke.singing.hear(t, Some(sung));
    }
    assert_eq!(
        state.karaoke.singing.tally.hit_words, 2,
        "sung in the new key"
    );
    // Another song picked: the engine says 0, and the next positions too.
    update(&mut state, Action::Engine(Event::Key { semitones: 0 }));
    update(&mut state, heard(0));
    assert_eq!((state.playing.key, note(&state)), (0, Some(6000.0)));
}

/// "la la", a second each, on C4 and E4.
fn two_note_song() -> (
    Arc<jodel_lyrics::Lyrics>,
    Arc<jodel_rhythm::waveform::Melody>,
) {
    let mut melody = jodel_rhythm::waveform::Melody {
        cents: vec![6000; 300],
        sure: vec![255; 300],
    };
    melody.cents[100..200].fill(6400);
    let word = |k: f64| jodel_lyrics::Word {
        start: Some(k),
        end: Some(k + 1.0),
        confidence: 0.9,
        ..jodel_lyrics::Word::new("la")
    };
    let line = jodel_lyrics::Line {
        words: vec![word(0.0), word(1.0)],
        ..jodel_lyrics::Line::default()
    };
    let lyrics = jodel_lyrics::Lyrics {
        lines: vec![line],
        source: jodel_lyrics::Source::Unknown,
        timing: jodel_lyrics::Timing::default(),
    };
    (Arc::new(lyrics), Arc::new(melody))
}

/// Another output or the engine started again: the key asked for stays,
/// and the new engine is told it after the queue, for the song played next.
#[test]
fn a_new_engine_is_told_the_key_after_the_queue() {
    let mut state = common::transition_state(120, 40);
    op(&mut state, Op::KeyUp);
    op(&mut state, Op::KeyUp);
    let queue = state.queue.paths.clone();
    let first = jodeldiplom::core::key::for_new_engine(&mut state, queue.clone());
    assert_eq!(
        first,
        [Command::Load(queue.clone()), Command::Key { semitones: 2 }]
    );
    assert_eq!(state.playing.key, 2);
    assert!(state.playing.track.is_none(), "nothing plays on it yet");
    op(&mut state, Op::KeyReset);
    let first = jodeldiplom::core::key::for_new_engine(&mut state, queue.clone());
    assert_eq!(first, [Command::Load(queue)]);
}
