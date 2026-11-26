//! Which song Karaoke is about once a song's end held the next one: the
//! one held stays the song through `esc` and back in (rule K2), until
//! another plays.

mod common;

use jodel_playback::{Command, Event, PlayState, TrackStart};
use jodeldiplom::core::karaoke::{self, SongLyrics};
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::{update, Action, Effect, Screen, State};

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

/// In the mode, the song playing (track 1) sung to its end: the engine
/// holds the next (2) and stops.
fn held_after_the_song(state: &mut State) {
    op(state, Op::KaraokeEnter);
    assert_eq!(state.karaoke.held, Some(1));
    update(state, Action::Engine(Event::Held { index: 2 }));
    update(state, Action::Engine(Event::State(PlayState::Stopped)));
    // The engine names no new current track for a hold.
    assert_eq!(state.queue.current, Some(1));
}

/// Held, then esc, then enter: the song held stays the one after the song
/// just sung, the one the tab showed; it is the one checked for its lyrics
/// (the song just sung has none any more here), and START plays it.
#[test]
fn held_then_esc_then_enter_keeps_the_song_held() {
    let mut state = common::karaoke_tab();
    held_after_the_song(&mut state);
    op(&mut state, Op::Back);
    assert_eq!(state.screen, Screen::Main);
    assert_eq!(karaoke::song_index(&state), Some(2), "the tab shows it");
    let just_sung = state.queue.paths[1].clone();
    state.karaoke.lyrics.insert(just_sung, SongLyrics::NotFound);
    let fx = op(&mut state, Op::KaraokeEnter);
    assert_eq!(state.screen, Screen::Karaoke, "{:?}", state.message);
    assert_eq!(state.karaoke.held, Some(2));
    assert!(!engine(&fx).contains(&&Command::Stop), "{fx:?}");
    let fx = op(&mut state, Op::KaraokeStart);
    assert!(
        engine(&fx)
            .iter()
            .any(|c| matches!(c, Command::PlayIn { index: 2, .. })),
        "{fx:?}"
    );
}

/// A song played on the tab after the mode puts the one held away: going
/// back in holds the song played, not the one held before.
#[test]
fn a_song_played_after_the_mode_is_the_one_held_next() {
    let mut state = common::karaoke_tab();
    held_after_the_song(&mut state);
    op(&mut state, Op::Back);
    let track = common::now_track(44_100, state.engine_settings(), 0);
    update(
        &mut state,
        Action::Engine(Event::TrackStarted(Box::new(TrackStart {
            index: 0,
            head: 0,
            info: track.info.clone(),
            route: track.route.clone(),
            output: track.output.clone(),
            device: "DAC".into(),
            output_frame: 0,
            from_frame: 0,
        }))),
    );
    assert_eq!(state.karaoke.held, None);
    assert_eq!(karaoke::song_index(&state), Some(0));
    // Its lyrics (asked for again, as it came to be wanted) found.
    common::singable(&mut state);
    op(&mut state, Op::KaraokeEnter);
    assert_eq!(state.screen, Screen::Karaoke, "{:?}", state.message);
    assert_eq!(state.karaoke.held, Some(0));
}

/// `N` twice in the mode, the engine not yet stopped: each goes on from the
/// song held, not from the one still heard.
#[test]
fn next_twice_goes_on_from_the_song_held() {
    let mut state = common::karaoke_tab();
    op(&mut state, Op::KaraokeEnter);
    update(&mut state, Action::Engine(Event::State(PlayState::Stopped)));
    let heard = common::now_track(44_100, state.engine_settings(), 0);
    state.playing.track = Some(heard);
    state.playing.state = PlayState::Playing;
    state.karaoke.held = None;
    op(&mut state, Op::Next);
    assert_eq!(state.karaoke.held, Some(1));
    op(&mut state, Op::Next);
    assert_eq!(state.karaoke.held, Some(2));
}
