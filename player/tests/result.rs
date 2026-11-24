//! A song's result in Karaoke mode (K3): shown when a song is sung to its
//! end, its medal and best kept, its reveal completed by the first START
//! and the next song started by the one after, its sounds on the frames'
//! clock.

mod common;

use common::karaoke_tab as tab;
use jodel_playback::{Command, Event};
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::singing::Motion;
use jodeldiplom::core::{karaoke, update, Action, Effect, State};

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

/// The first song made one line of four words, a second each on C4, its
/// singing ready to be scored.
fn four_words(state: &mut State) {
    use std::sync::Arc;
    let mut melody = jodel_rhythm::waveform::Melody {
        cents: vec![0; 500],
        sure: vec![0; 500],
    };
    for i in 0..400 {
        melody.cents[i] = 6000;
        melody.sure[i] = 255;
    }
    let line = jodel_lyrics::Line {
        words: (0..4)
            .map(|k| jodel_lyrics::Word {
                start: Some(f64::from(k)),
                end: Some(f64::from(k) + 1.0),
                ..jodel_lyrics::Word::new("la")
            })
            .collect(),
        ..jodel_lyrics::Line::default()
    };
    let lyrics = Arc::new(jodel_lyrics::Lyrics {
        lines: vec![line],
        source: jodel_lyrics::Source::Unknown,
        timing: jodel_lyrics::Timing::default(),
    });
    let path = state.queue.paths[0].clone();
    state.karaoke.singing.prepare(
        &path,
        &lyrics,
        &jodeldiplom::core::singing::melodic(&melody),
    );
}

/// The four words sung, those before `hit` seconds in tune.
fn sing_four(state: &mut State, hit: f64) {
    for i in 0..450 {
        let t = f64::from(i) / 100.0;
        state.karaoke.singing.hear(t, (t < hit).then_some(6000.0));
    }
}

#[test]
fn a_song_sung_to_its_end_shows_its_medal_and_keeps_its_best() {
    use jodeldiplom::core::singing::Medal;
    let mut state = tab();
    // Three of the four words sung.
    four_words(&mut state);
    let path = state.queue.paths[0].clone();
    sing_four(&mut state, 3.0);
    update(&mut state, Action::Engine(Event::Held { index: 1 }));
    let result = state
        .karaoke
        .result
        .as_ref()
        .expect("a result")
        .result
        .clone();
    assert_eq!((result.points, result.hit, result.words), (300, 3, 4));
    assert_eq!(result.medal, Some(Medal::Silver));
    assert!(result.new_best() && state.karaoke.bests_changed);
    assert_eq!(state.karaoke.bests[&path.display().to_string()].points, 300);
    // Sung worse next time: the best stays.
    state.karaoke.bests_changed = false;
    sing_four(&mut state, 0.0);
    update(&mut state, Action::Engine(Event::Held { index: 1 }));
    let result = state
        .karaoke
        .result
        .as_ref()
        .expect("a result")
        .result
        .clone();
    assert_eq!(
        (result.points, result.medal, result.best),
        (0, None, Some(300))
    );
    assert!(!result.new_best() && !state.karaoke.bests_changed);
    // The game off: no result.
    state.karaoke.result = None;
    state.config.karaoke_game = false;
    update(&mut state, Action::Engine(Event::Held { index: 1 }));
    assert!(state.karaoke.result.is_none());
}

/// In the mode, a song sung to its end: its result shows while the next
/// waits; START during the reveal completes it (and starts nothing), the
/// next START starts the song, the result gone. Once the reveal is over,
/// START starts the song at once.
#[test]
fn start_during_the_reveal_completes_it_and_the_next_starts_the_song() {
    let sung = || {
        let mut state = tab();
        op(&mut state, Op::KaraokeEnter);
        four_words(&mut state);
        sing_four(&mut state, 4.0);
        update(&mut state, Action::Engine(Event::Held { index: 1 }));
        state
    };
    let starts = |fx: &[Effect]| {
        engine(fx)
            .iter()
            .any(|c| matches!(c, Command::PlayIn { index: 1, .. }))
    };
    let mut state = sung();
    let shown = karaoke::result_shown(&state).expect("shown").clone();
    assert!(shown.revealing(state.now_ms));
    assert_eq!(karaoke::result_motion(&state), Some(Motion::Revealing));
    state.now_ms += 500;
    let fx = op(&mut state, Op::KaraokeStart);
    assert!(engine(&fx).is_empty() && state.karaoke.countdown.is_none());
    let skipped = karaoke::result_shown(&state).expect("still shown");
    assert!(skipped.time(state.now_ms) >= skipped.reveal.settled() - 1e-9);
    assert_eq!(karaoke::result_motion(&state), Some(Motion::Twinkling));
    let fx = op(&mut state, Op::KaraokeStart);
    assert!(starts(&fx));
    assert!(karaoke::result_shown(&state).is_none() && state.karaoke.result.is_none());

    // The reveal left to its end: START starts the next song.
    let mut state = sung();
    let end = karaoke::result_shown(&state).unwrap().reveal.end();
    state.now_ms += (end * 1000.0) as u64 + 1;
    assert!(starts(&op(&mut state, Op::KaraokeStart)));
}

/// The reveal's sounds go by the frames' clock, as the picture does: the
/// app ticks the core every 250 ms (`now_ms`) and draws a frame every
/// 1/60 s (`drawn_ms`), and each beat sounds once, on the first frame at
/// or past it, none bunched up to the next tick nor dropped as stale.
#[test]
fn the_reveal_sounds_on_the_frames_clock() {
    use jodeldiplom::sfx::reveal::{sound_of, RevealSounds};
    let mut state = tab();
    op(&mut state, Op::KaraokeEnter);
    four_words(&mut state);
    sing_four(&mut state, 4.0);
    let start = 10_000;
    update(&mut state, Action::Tick(start));
    state.drawn_ms = Some(start);
    update(&mut state, Action::Engine(Event::Held { index: 1 }));
    let shown = karaoke::result_shown(&state).expect("shown").clone();
    let mut sounds = RevealSounds::default();
    let mut heard: Vec<f64> = Vec::new();
    let mut tick = start;
    for f in 1..(60.0 * (shown.reveal.end() + 1.0)) as u64 {
        let now = start + f * 1000 / 60;
        if now >= tick + 250 {
            tick += 250;
            update(&mut state, Action::Tick(tick));
        }
        state.drawn_ms = Some(now);
        let (s, t) = karaoke::result_now(&state).expect("still shown");
        // The picture's clock and the sounds' are one.
        assert_eq!(t, s.time(now));
        let cues = sounds.frame(Some((s, t)));
        heard.extend(cues.iter().map(|_| t));
    }
    let due: Vec<f64> = shown
        .reveal
        .beats()
        .iter()
        .filter(|&&(_, b)| sound_of(b, &shown).is_some())
        .map(|&(at, _)| at)
        .collect();
    assert_eq!(heard.len(), due.len(), "{heard:?}\n{due:?}");
    for (t, at) in heard.iter().zip(&due) {
        assert!(
            *t >= *at && *t < at + 1.0 / 60.0 + 1e-3,
            "due {at}, heard {t}"
        );
    }
}
