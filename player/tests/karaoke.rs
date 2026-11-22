//! The Karaoke tab and Karaoke mode: the rules K1 to K5 of
//! `INTERACTIONS.md`.

mod common;

use jodel_playback::{Command, Event, PlayState};
use jodeldiplom::core::desk::{DeskChange, StripKind};
use jodeldiplom::core::karaoke::{self, Slider};
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::layout::{self, Layout};
use jodeldiplom::core::mouse::{self, MouseButton, MouseInput, MouseKind, Target};
use jodeldiplom::core::{update, Action, Effect, Rhythm, Rhythmed, Screen, Side, State};

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

/// The gain the vocals' lane was last given on the desk (the voice's
/// level), if it was.
fn voice_sent(fx: &[Effect]) -> Option<f64> {
    fx.iter().rev().find_map(|e| match e {
        Effect::Desk(DeskChange::Lane(StripKind::Vocals, gain)) => Some(*gain),
        _ => None,
    })
}

/// The Karaoke tab shown, a track playing, a GPU for the stems, the voice
/// at 20 %, and every song's lyrics found (their words not timed: there is
/// no speech model), so each can be sung.
fn tab() -> State {
    let mut state = common::transition_state(120, 40);
    state.side = Side::Karaoke;
    state.config.karaoke_voice = 20;
    singable(&mut state);
    state
}

/// Every song in the queue has its lyrics: found, as they are (kept for
/// whichever karaoke comes to want).
fn singable(state: &mut State) {
    use jodel_lyrics::{Lyrics, Source};
    use jodeldiplom::core::karaoke::SongLyrics;
    use jodeldiplom::lyrics::Answer;
    for path in state.queue.paths.clone() {
        let lyrics = Lyrics::parse_text("la la la\nla la", Source::Lrclib);
        state.karaoke.lyrics.insert(
            path,
            SongLyrics::from_answer(Answer::Found {
                lyrics,
                timed: None,
            }),
        );
    }
}

/// Track `index` analysed with a beat of `beat` seconds from `first`.
fn beats_for(state: &mut State, index: usize, first: f64, beat: f64) {
    let beats: Vec<f64> = (0..200).map(|k| first + beat * f64::from(k)).collect();
    let mut a = jodel_rhythm::Analysis {
        duration: 200.0,
        downbeats: (0..beats.len()).step_by(4).collect(),
        beats,
        confidence: 0.9,
        ..Default::default()
    };
    a.derive();
    let path = state.queue.paths[index].clone();
    state.rhythm.insert(
        path,
        Rhythm::Ready(std::sync::Arc::new(Rhythmed::new(a, None))),
    );
}

/// Nothing playing (the engine stopped).
fn stopped(state: &mut State) {
    update(state, Action::Engine(Event::State(PlayState::Stopped)));
}

#[test]
fn the_tab_plays_the_voice_at_its_level_and_esc_does_nothing_there() {
    let mut state = tab();
    // On the tab the voice plays at the slider's level, moved at once: the
    // vocals' lane on the desk, the strips as they were.
    let fx = op(&mut state, Op::SliderUp);
    assert_eq!(state.config.karaoke_voice, 25);
    assert_eq!(voice_sent(&fx), Some(0.25));
    assert!(fx.contains(&Effect::SaveConfig));
    // Down: the delay's nudge, 10 ms a step (kept with the delays, not in
    // the settings); up: the voice again.
    op(&mut state, Op::Down);
    let fx = op(&mut state, Op::SliderFineDown);
    assert_eq!(state.config.karaoke_voice, 25);
    assert_eq!(jodeldiplom::core::lag::nudge_ms(&state), -10);
    assert!(!fx.contains(&Effect::SaveConfig) && state.karaoke.lag.changed);
    op(&mut state, Op::Up);
    let fx = op(&mut state, Op::SliderFineDown);
    assert_eq!(state.config.karaoke_voice, 24);
    assert_eq!(voice_sent(&fx), Some(0.24));
    // Esc stays on the tab and does nothing.
    let fx = op(&mut state, Op::Back);
    assert!(fx.is_empty(), "{fx:?}");
    assert_eq!((state.screen, state.side), (Screen::Main, Side::Karaoke));
    assert!(!state.desk.stems_touched());
    // Leaving the tab puts the lane back at 1; coming back plays the voice
    // at its level again.
    let fx = op(&mut state, Op::ToggleSide);
    assert_ne!(state.side, Side::Karaoke);
    assert_eq!(voice_sent(&fx), Some(1.0));
    let mut back = Vec::new();
    while state.side != Side::Karaoke {
        back = op(&mut state, Op::ToggleSide);
    }
    assert_eq!(voice_sent(&back), Some(0.24));
}

/// Without a GPU the voice still sets its lane: a song whose stems are
/// kept has them from the cache, and the engine is asked for those only.
#[test]
fn without_a_gpu_the_voice_turns_down_the_kept_stems() {
    use jodel_playback::MakeStems;
    use jodeldiplom::core::stems::make_wanted;
    let mut state = tab();
    state.gpu = jodeldiplom::core::Gpu::Absent;
    let fx = op(&mut state, Op::SliderUp);
    assert_eq!(state.config.karaoke_voice, 25);
    assert_eq!(voice_sent(&fx), Some(0.25), "{fx:?}");
    assert_eq!(make_wanted(&state), MakeStems::Kept);
    state.gpu = jodeldiplom::core::Gpu::Present("a GPU".into());
    assert_eq!(make_wanted(&state), MakeStems::Separate);
}

#[test]
fn the_mode_plays_the_voice_at_its_level_and_esc_keeps_it_on_the_tab() {
    let mut state = tab();
    let fx = op(&mut state, Op::KaraokeEnter);
    assert_eq!(state.screen, Screen::Karaoke);
    assert!(engine(&fx).contains(&&Command::HoldAtEnd(true)));
    assert_eq!(voice_sent(&fx), Some(0.2));
    // The song playing stops and is held at its start for START: nothing
    // plays without a count-in.
    let playing = state.playing.track.as_ref().map(|t| t.index);
    assert!(engine(&fx).contains(&&Command::Stop), "{fx:?}");
    assert!(!engine(&fx).iter().any(|c| matches!(
        c,
        Command::Play(_) | Command::PlayIn { .. } | Command::Seek(_)
    )));
    assert_eq!(state.karaoke.held, playing);
    assert!(karaoke::waiting(&state), "START shows at once");
    // The voice's level is heard at once in the mode.
    let fx = op(&mut state, Op::SliderUp);
    assert_eq!(voice_sent(&fx), Some(0.25));
    // Tab and the other screens are not reached from the mode.
    for o in [
        Op::ToggleSide,
        Op::Settings,
        Op::Help,
        Op::NowPlaying,
        Op::Filter,
    ] {
        op(&mut state, o);
        assert_eq!(state.screen, Screen::Karaoke, "{o:?}");
    }
    // Esc: back to the tab, the voice still at its level (nothing sent),
    // songs run on.
    let fx = op(&mut state, Op::Back);
    assert_eq!((state.screen, state.side), (Screen::Main, Side::Karaoke));
    assert!(engine(&fx).contains(&&Command::HoldAtEnd(false)));
    assert_eq!(voice_sent(&fx), None, "{fx:?}");
    assert!(!engine(&fx).iter().any(|c| matches!(c, Command::Stop)));
    // Leaving the tab then puts them back.
    let fx = op(&mut state, Op::ToggleSide);
    assert_eq!(voice_sent(&fx), Some(1.0));
}

#[test]
fn start_counts_in_three_seconds_whatever_the_song() {
    for (first, beat) in [(0.3, 0.5), (3.0, 0.5)] {
        let mut state = tab();
        op(&mut state, Op::KaraokeEnter);
        // The song held after the last one ended.
        update(&mut state, Action::Engine(Event::Held { index: 2 }));
        stopped(&mut state);
        assert!(karaoke::waiting(&state));
        // Its beats play no part: a number a second, then its start.
        beats_for(&mut state, 2, first, beat);
        let fx = op(&mut state, Op::KaraokeStart);
        assert!(
            engine(&fx).iter().any(|c| matches!(
                c,
                Command::PlayIn { index: 2, lead } if (lead - 3.0).abs() < 1e-9
            )),
            "{fx:?}"
        );
        let countdown = state.karaoke.countdown.expect("a count-in");
        assert_eq!(countdown.at(-3.0).map(|(n, _)| n), Some(3));
        assert_eq!(countdown.at(-0.5).map(|(n, _)| n), Some(1));
        assert!(!karaoke::waiting(&state), "START goes while it counts");
    }
}

/// Space in the mode never lets a song play without its count-in: entering
/// holds the song, space is START from there; a song paused there goes on
/// only after the same count-in.
#[test]
fn space_never_plays_a_song_without_its_count_in() {
    let mut state = tab();
    op(&mut state, Op::KaraokeEnter);
    // Entered while the song played: space is START, not pause or play on.
    let fx = op(&mut state, Op::PlayPause);
    assert!(
        engine(&fx)
            .iter()
            .any(|c| matches!(c, Command::PlayIn { lead, .. } if (lead - 3.0).abs() < 1e-9)),
        "{fx:?}"
    );
    // Counted in and playing: space pauses.
    state.karaoke.countdown = None;
    state.karaoke.held = None;
    state.playing.state = PlayState::Playing;
    let fx = op(&mut state, Op::PlayPause);
    assert_eq!(engine(&fx), [&Command::Pause]);
    update(&mut state, Action::Engine(Event::State(PlayState::Paused)));
    assert!(karaoke::waiting(&state));
    // START (or space) on a paused song: its count-in first, no resume yet.
    let fx = op(&mut state, Op::PlayPause);
    assert!(engine(&fx).is_empty(), "{fx:?}");
    let resume = state.karaoke.resume_at.expect("a count-in");
    assert_eq!(resume, state.now_ms + 3_000);
    assert_eq!(
        karaoke::resume_count(&state, state.now_ms).map(|(n, _)| n),
        Some(3)
    );
    assert_eq!(
        karaoke::resume_count(&state, resume - 500).map(|(n, _)| n),
        Some(1)
    );
    assert!(!karaoke::waiting(&state));
    // Before it is over, nothing; once it is, the song goes on.
    let fx = update(&mut state, Action::Tick(resume - 1));
    assert!(engine(&fx).is_empty(), "{fx:?}");
    let fx = update(&mut state, Action::Tick(resume));
    assert_eq!(engine(&fx), [&Command::Resume]);
    assert_eq!(state.karaoke.resume_at, None);
}

/// Track `index`'s words timed by us, lines starting at `starts` seconds.
fn timed(state: &mut State, index: usize, starts: &[f64]) {
    use jodel_lyrics::{Lyrics, Source};
    use jodeldiplom::lyrics::Answer;
    let lrc: String = starts
        .iter()
        .enumerate()
        .map(|(k, &s)| {
            format!(
                "[{:02}:{:05.2}]line {k} goes on\n",
                (s / 60.0) as u32,
                s % 60.0
            )
        })
        .collect();
    let mut ours = jodel_lyrics::spread::split_lines(&Lyrics::parse(&lrc, Source::Lrclib));
    ours.timing.by = Some(jodel_lyrics::Maker::new("ctc:test", 1));
    let path = state.queue.paths[index].clone();
    if !state.karaoke.lyrics_wanted.contains(&path) {
        state.karaoke.lyrics_wanted.push(path.clone());
    }
    update(
        state,
        Action::Lyrics(
            path,
            Answer::Found {
                lyrics: Lyrics::parse_text(&lrc, Source::Lrclib),
                timed: Some(ours),
            },
        ),
    );
}

/// The count-in ends as the first word is sung: after a long intro the
/// song plays at once and the count comes in it (space pauses there, esc
/// leaves it playing); sung from 1.2 s in, it starts 1.8 s late, and the
/// title's time later still.
#[test]
fn the_count_in_ends_as_the_first_word_is_sung() {
    let mut state = tab();
    op(&mut state, Op::KaraokeEnter);
    update(&mut state, Action::Engine(Event::Held { index: 2 }));
    stopped(&mut state);
    timed(&mut state, 2, &[30.0, 34.0]);
    assert_eq!(karaoke::first_sung(&state, 2), Some(30.0));
    let fx = op(&mut state, Op::KaraokeStart);
    assert!(
        engine(&fx).iter().any(|c| matches!(
            c,
            Command::PlayIn { index: 2, lead } if lead.abs() < 1e-9
        )),
        "{fx:?}"
    );
    let countdown = state.karaoke.countdown.expect("a count-in");
    assert_eq!(countdown.at(20.0), None, "the intro plays first");
    assert_eq!(countdown.at(27.0).map(|(n, _)| n), Some(3));
    assert_eq!(countdown.at(29.5).map(|(n, _)| n), Some(1));
    assert_eq!(countdown.at(30.0), None);
    // In the intro: space pauses (it is the song, not the silence before).
    let start = common::now_track(44_100, state.engine_settings(), 2);
    state.playing.track = Some(start);
    state.playing.state = PlayState::Playing;
    state.playing.position = 5.0;
    let fx = op(&mut state, Op::PlayPause);
    assert_eq!(engine(&fx), [&Command::Pause]);
    // Esc in the intro: the song plays on.
    let fx = op(&mut state, Op::Back);
    assert!(!engine(&fx).contains(&&Command::Stop), "{fx:?}");
    // Sung almost at once: the song starts late enough for its title to
    // show and fade, then the count.
    let mut state = tab();
    op(&mut state, Op::KaraokeEnter);
    update(&mut state, Action::Engine(Event::Held { index: 2 }));
    stopped(&mut state);
    timed(&mut state, 2, &[1.2, 5.0]);
    let title = karaoke::TITLE_HOLD + karaoke::TITLE_FADE;
    let fx = op(&mut state, Op::KaraokeStart);
    assert!(
        engine(&fx).iter().any(|c| matches!(
            c,
            Command::PlayIn { index: 2, lead } if (lead - 1.8 - title).abs() < 1e-9
        )),
        "{fx:?}"
    );
    let countdown = state.karaoke.countdown.unwrap();
    assert_eq!(countdown.at(-1.8).map(|(n, _)| n), Some(3));
    assert_eq!(countdown.at(1.0).map(|(n, _)| n), Some(1));
}

/// A song waiting for START shows its title and artist (asked for, as the
/// queue's rows are), the same as START opens it: nothing changes or moves
/// until they fade.
#[test]
fn the_title_and_artist_show_while_the_song_waits_and_stay_as_it_starts() {
    use jodeldiplom::view::karaoke::lines_now;
    let mut state = tab();
    op(&mut state, Op::KaraokeEnter);
    update(&mut state, Action::Engine(Event::Held { index: 2 }));
    stopped(&mut state);
    state.playing.track = None;
    timed(&mut state, 2, &[1.0, 5.0]);
    let path = state.queue.paths[2].clone();
    state.details.remove(&path);
    state.pending.remove(&path);
    let fx = update(&mut state, Action::Engine(Event::Held { index: 2 }));
    assert!(
        fx.iter()
            .any(|e| matches!(e, Effect::WantDetails(p) if p.contains(&path))),
        "{fx:?}"
    );
    state.details.insert(
        path.clone(),
        jodeldiplom::browser::Details {
            title: Some("A Song".into()),
            artist: Some("Someone".into()),
            ..Default::default()
        },
    );
    let shown = |state: &State| -> Vec<(String, f64)> {
        lines_now(state, 0.2)
            .into_iter()
            .map(|l| (l.text, l.alpha))
            .collect()
    };
    let waiting = shown(&state);
    assert_eq!(
        waiting,
        [("A Song".to_owned(), 1.0), ("Someone".to_owned(), 1.0)]
    );
    // START: the track opens (its own tags whatever they are), heard from
    // its lead's start; the same two lines, whole.
    op(&mut state, Op::KaraokeStart);
    let lead = state.karaoke.countdown.unwrap().lead();
    assert!(lead > 3.0);
    state.playing.track = Some(common::now_track(44_100, state.engine_settings(), 2));
    state.playing.state = PlayState::Playing;
    assert_eq!(
        lines_now(&state, -lead + 0.2)
            .into_iter()
            .map(|l| (l.text, l.alpha))
            .collect::<Vec<_>>(),
        waiting
    );
}

/// The title shows once, as the song starts: not again when the count-in
/// is over (its record gone), which hid the first lines being sung. Here
/// with lyrics heard by machine learning, whose warning goes with it.
#[test]
fn the_title_shows_once_and_not_again_after_the_count_in() {
    use jodel_lyrics::{Lyrics, Maker, Source};
    use jodeldiplom::core::karaoke::SongLyrics;
    use jodeldiplom::lyrics::Answer;
    use jodeldiplom::view::karaoke::{lines_now, LineKind};
    let mut state = tab();
    let lrc = "[00:00.42]<00:00.42>Ja,<00:00.70> <00:01.50>Dynamite<00:02.08>\n\
               [00:05.54]<00:05.54>null<00:05.78> <00:05.94>null<00:06.16>\n";
    let mut timed = Lyrics::parse(lrc, Source::Transcribed(Maker::new("asr", 2)));
    timed.timing.by = Some(Maker::new("ctc:test", 1));
    let mut text = timed.clone();
    text.drop_times();
    for path in state.queue.paths.clone() {
        let answer = Answer::Found {
            lyrics: text.clone(),
            timed: Some(timed.clone()),
        };
        state
            .karaoke
            .lyrics
            .insert(path, SongLyrics::from_answer(answer));
    }
    op(&mut state, Op::KaraokeEnter);
    update(&mut state, Action::Engine(Event::Held { index: 2 }));
    stopped(&mut state);
    op(&mut state, Op::KaraokeStart);
    let lead = state.karaoke.countdown.unwrap().lead();
    state.playing.track = Some(common::now_track(44_100, state.engine_settings(), 2));
    state.playing.state = PlayState::Playing;
    // Heard from the lead's start to 10 s in, a tick every quarter second.
    let mut titled_after_count = Vec::new();
    let mut warned = false;
    let mut t = -lead;
    while t < 10.0 {
        state.playing.position = t;
        state.now_ms += 250;
        let now = state.now_ms;
        update(&mut state, Action::Tick(now));
        let lines = lines_now(&state, t);
        let title = lines.iter().any(|l| l.kind == LineKind::Warning);
        warned |= lines.iter().any(|l| l.kind == LineKind::Warning);
        if title && t > 0.0 {
            titled_after_count.push(t);
        }
        t += 0.25;
    }
    assert!(warned, "the warning shows with the title");
    assert!(state.karaoke.countdown.is_none(), "the count-in is over");
    assert!(
        titled_after_count.is_empty(),
        "the title again at {titled_after_count:?}"
    );
}

/// Shift+enter (or shift+click on the button) goes into the mode where the
/// song is: nothing stops, nothing counts in, the voice down at once.
#[test]
fn shift_enter_goes_into_the_mode_where_the_song_is() {
    use jodeldiplom::core::keymap::{lookup, Key, KeyCode, Meaning};
    let mut state = tab();
    assert_eq!(
        lookup(&state, Key::shift(KeyCode::Enter)),
        Some(Meaning::Op(Op::KaraokeEnterHere))
    );
    assert!(matches!(
        mouse::shifted(Action::Op(Op::KaraokeEnter)),
        Action::Op(Op::KaraokeEnterHere)
    ));
    let fx = op(&mut state, Op::KaraokeEnterHere);
    assert_eq!(state.screen, Screen::Karaoke);
    assert!(!engine(&fx).iter().any(|c| matches!(
        c,
        Command::Stop | Command::Play(_) | Command::PlayIn { .. } | Command::Seek(_)
    )));
    assert_eq!(state.karaoke.held, None);
    assert_eq!(state.karaoke.countdown, None);
    assert!(!karaoke::waiting(&state), "the song plays on: no START");
    assert_eq!(voice_sent(&fx), Some(0.2));
    // Space pauses it as any song playing in the mode.
    let fx = op(&mut state, Op::PlayPause);
    assert_eq!(engine(&fx), [&Command::Pause]);
}

/// `r` records what the echo canceller works with, on the tab and in the
/// mode (the app writes it).
#[test]
fn r_records_the_echo_canceller() {
    let mut state = tab();
    assert!(op(&mut state, Op::KaraokeRecordEcho).contains(&Effect::RecordEcho));
    op(&mut state, Op::KaraokeEnterHere);
    assert!(op(&mut state, Op::KaraokeRecordEcho).contains(&Effect::RecordEcho));
}

#[test]
fn the_mode_arms_no_transitions_and_makes_no_jumps() {
    let mut state = tab();
    op(&mut state, Op::KaraokeEnter);
    for o in [
        Op::TransitionGo,
        Op::JumpBar(0),
        Op::JumpBarAtEnd(1),
        Op::SeekForward,
        Op::SeekBack,
    ] {
        let fx = op(&mut state, o);
        assert!(engine(&fx).is_empty(), "{o:?}: {fx:?}");
    }
}

#[test]
fn esc_during_a_count_in_calls_the_start_off() {
    let mut state = tab();
    op(&mut state, Op::KaraokeEnter);
    update(&mut state, Action::Engine(Event::Held { index: 2 }));
    stopped(&mut state);
    op(&mut state, Op::KaraokeStart);
    // The count-in is heard: the track started, a second before its start.
    let start = common::now_track(44_100, state.engine_settings(), 2);
    state.playing.track = Some(start);
    state.playing.state = PlayState::Playing;
    state.playing.position = -1.0;
    let fx = op(&mut state, Op::Back);
    assert!(engine(&fx).contains(&&Command::Stop), "{fx:?}");
    assert_eq!(state.karaoke.held, Some(2));
    assert_eq!(state.screen, Screen::Main);
}

#[test]
fn next_in_the_mode_holds_the_next_song_for_start() {
    let mut state = tab();
    op(&mut state, Op::KaraokeEnter);
    let fx = op(&mut state, Op::Next);
    assert_eq!(engine(&fx), [&Command::Stop]);
    assert_eq!(state.karaoke.held, Some(2));
}

#[test]
fn lyrics_are_asked_for_on_the_tab_and_follow_the_song() {
    use jodel_lyrics::{Lyrics, Source};
    use jodeldiplom::lyrics::Answer;
    use jodeldiplom::view::karaoke::{lines_now, LineKind};
    let mut state = common::transition_state(120, 40);
    state.side = Side::Browser;
    // Not asked for away from Karaoke.
    let fx = update(&mut state, Action::Tick(1));
    assert!(!fx.iter().any(|e| matches!(e, Effect::WantLyrics(_))));
    let fx = update(&mut state, Action::ShowSide(Side::Karaoke));
    let asked = fx.iter().find_map(|e| match e {
        Effect::WantLyrics(paths) => Some(paths.clone()),
        _ => None,
    });
    let playing = state.playing.track.as_ref().unwrap().index;
    assert_eq!(
        asked.as_deref(),
        Some(&state.queue.paths[playing..(playing + 3).min(state.queue.paths.len())])
    );
    // Words found, then timed by us (here: a stand-in that puts the
    // lines at 10, 14 and 30 s and splits them by syllables).
    let lrc = "[00:10.00]one two three\n[00:14.00]four five\n[00:30.00]six\n";
    let mut ours = jodel_lyrics::spread::split_lines(&Lyrics::parse(lrc, Source::Lrclib));
    ours.timing.by = Some(jodel_lyrics::Maker::new("ctc:test", 1));
    let path = state.queue.paths[playing].clone();
    update(
        &mut state,
        Action::Lyrics(
            path,
            Answer::Found {
                lyrics: Lyrics::parse_text(lrc, Source::Lrclib),
                timed: Some(ours),
            },
        ),
    );
    // As the song starts (heard from 0 s here): its title and artist
    // alone, then fading out, then the first line alone until it is sung.
    state.playing.position = 0.0;
    let now = state.now_ms + 1_000;
    update(&mut state, Action::Tick(now));
    let title = |t: f64| -> Vec<(String, f64)> {
        lines_now(&state, t)
            .into_iter()
            .map(|l| (l.text, l.alpha))
            .collect()
    };
    let opening = title(0.5);
    assert!(
        !opening.is_empty()
            && opening
                .iter()
                .all(|(t, a)| t != "one two three" && *a == 1.0)
    );
    let fading = title(2.5);
    assert!(
        fading.iter().all(|(_, a)| (*a - 0.5).abs() < 1e-9),
        "{fading:?}"
    );
    for t in [3.5, 8.5] {
        let alone = lines_now(&state, t);
        assert_eq!(alone.len(), 1, "{alone:?}");
        assert_eq!(
            (alone[0].text.as_str(), alone[0].kind, alone[0].fill),
            ("one two three", LineKind::Current, 0.0)
        );
    }
    // Sung: filling from the left at a sung pace (three syllables,
    // 0.9 s), the next lines after it.
    let now = lines_now(&state, 10.5);
    let current = now.iter().find(|l| l.kind == LineKind::Current).unwrap();
    assert_eq!(current.text, "one two three");
    assert!(current.fill > 0.2 && current.fill < 1.0, "{}", current.fill);
    assert!(now
        .iter()
        .any(|l| l.text == "four five" && l.kind == LineKind::Next));
    // A long pause: a note, the line before it sung.
    let pause = lines_now(&state, 20.0);
    assert!(pause.iter().any(|l| l.text == "♪"));
    assert!(pause
        .iter()
        .any(|l| l.text == "four five" && l.kind == LineKind::Previous));
    // Within two seconds of the next line, the pause ends: it moves up to
    // be sung, not filled yet, the line before staying where it was.
    let next = lines_now(&state, 29.0);
    assert!(next
        .iter()
        .any(|l| l.text == "six" && l.kind == LineKind::Current && l.fill == 0.0));
    assert!(next
        .iter()
        .any(|l| l.text == "four five" && l.kind == LineKind::Previous));
    // The mode draws the words `karaoke_lead` ahead of what is heard (0.2 s
    // by default), to be read before they are sung.
    assert_eq!(state.config.karaoke_lead, 200);
    let mut shown = state.clone();
    shown.screen = Screen::Karaoke;
    let Layout::Karaoke(k) = layout::compute(&shown) else {
        panic!("the mode's layout")
    };
    let p = jodeldiplom::view::karaoke::picture(&shown, &k, 10.3);
    let ahead = lines_now(&shown, 10.5);
    assert_eq!(p.lines, ahead);
    assert_ne!(p.lines, lines_now(&shown, 10.3), "not where the voice is");
    // The tab says what was found.
    let lyrics = karaoke::lyrics_of(&state, playing).unwrap();
    assert_eq!(lyrics.describe(), "from LRCLIB, words timed (ctc:test)");
}

#[test]
fn the_microphone_is_open_while_karaoke_shows() {
    let _guard = common::real_time();
    use jodel_output::input::Input;
    use jodel_output::SimClock;
    use jodeldiplom::app::{App, AppOptions, Output};
    use std::sync::{Arc, Mutex};
    let tree = common::music_tree("karaoke-mic");
    let mut options = AppOptions::new(tree.album.clone());
    options.output = Output::Null(SimClock::Realtime);
    options.devices = Arc::new(|| Ok(Vec::new()));
    // Each microphone opened: the id asked for, and the input.
    type Opened = Vec<(Option<String>, Input)>;
    let opened: Arc<Mutex<Opened>> = Arc::default();
    let log = opened.clone();
    options.mic = Arc::new(move |id| {
        let (input, _feed) = Input::feed(48_000, 240);
        log.lock()
            .unwrap()
            .push((id.map(str::to_owned), input.clone()));
        Ok(input)
    });
    options.mics = Arc::new(|| {
        Ok(vec![
            ("a".into(), "Mic A".into()),
            ("b".into(), "Mic B".into()),
        ])
    });
    let mut app = App::new(options);
    assert!(opened.lock().unwrap().is_empty());
    // Opened in the background: the tab shows at once, the microphone
    // comes after.
    let open = |app: &mut App| {
        for _ in 0..200 {
            app.pump();
            if app.state.karaoke.mic.name.is_some() {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        false
    };
    app.dispatch(Action::ShowSide(Side::Karaoke));
    assert!(open(&mut app));
    assert_eq!(opened.lock().unwrap().len(), 1);
    // Nothing plays, yet the window keeps drawing: the meter moves with
    // the voice (K14).
    assert_ne!(app.state.playing.state, jodel_playback::PlayState::Playing);
    assert!(app.animating());
    // It is listened to, never played: the sliders have nothing to do with it.
    let input = opened.lock().unwrap()[0].1.clone();
    assert!(input.echo_cancel(), "its echo canceller on (karaoke_echo)");
    // `i`: the next one, kept in the settings, opened at once.
    app.dispatch(Action::Op(Op::MicNext));
    assert!(open(&mut app));
    assert_eq!(app.state.config.karaoke_mic_device.as_deref(), Some("a"));
    assert_eq!(
        opened.lock().unwrap().last().unwrap().0.as_deref(),
        Some("a")
    );
    // Away from Karaoke it closes.
    app.dispatch(Action::ShowSide(Side::Browser));
    assert!(app.state.karaoke.mic.name.is_none());
    let opens = opened.lock().unwrap().len();
    app.dispatch(Action::Tick(1));
    assert_eq!(opened.lock().unwrap().len(), opens);
    app.shutdown();
    let _ = std::fs::remove_dir_all(&tree.root);
}

#[test]
fn the_speech_model_is_found_where_the_build_put_it() {
    use jodeldiplom::lyrics::{model_in, MODEL_DIR, MODEL_FILE};
    let root = std::env::temp_dir().join(format!("jodel-model-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let put = |dir: &std::path::Path| {
        std::fs::create_dir_all(dir.join(MODEL_DIR)).unwrap();
        std::fs::write(dir.join(MODEL_DIR).join(MODEL_FILE), b"x").unwrap();
        dir.join(MODEL_DIR).join(MODEL_FILE)
    };
    let exe = root.join("release");
    let settings = root.join("settings");
    std::fs::create_dir_all(&exe).unwrap();
    assert_eq!(model_in(Some(&exe), Some(&settings)), None);
    // Where earlier players downloaded it.
    let old = put(&settings);
    assert_eq!(model_in(Some(&exe), Some(&settings)), Some(old));
    // In the app bundle: Contents/MacOS/../Resources/models.
    let bundle = root.join("Contents");
    std::fs::create_dir_all(bundle.join("MacOS")).unwrap();
    put(&bundle.join("Resources"));
    let found = model_in(Some(&bundle.join("MacOS")), Some(&settings)).unwrap();
    assert!(found.starts_with(bundle.join("MacOS")), "{found:?}");
    // Beside the binary comes first.
    let beside = put(&exe);
    assert_eq!(model_in(Some(&exe), Some(&settings)), Some(beside));
    let _ = std::fs::remove_dir_all(&root);
    // Nothing to download in the player: `d` on the tab is the game's
    // level, not a download.
    let state = tab();
    let d = jodeldiplom::core::keymap::lookup(&state, jodeldiplom::core::keymap::Key::ch('d'));
    assert!(
        matches!(
            d,
            Some(jodeldiplom::core::keymap::Meaning::Op(Op::KaraokeLevel))
        ),
        "{d:?}"
    );
}

/// What `input` does with the pointer at `x` cells, done.
#[track_caller]
fn point(state: &mut State, input: MouseInput, x: f64) {
    let action = mouse::action_at(state, input, x, f64::from(input.row) + 0.5).expect("an action");
    update(state, action);
}

#[test]
fn a_slider_under_the_pointer_lands_on_whole_percents() {
    let mut state = tab();
    let Layout::Main(m) = layout::compute(&state) else {
        panic!()
    };
    let sliders = m.karaoke.expect("the tab's layout").sliders;
    // The delay in force (150 ms, the default), from 0 to 600 ms in steps
    // of 10, the wheel a step a notch: what moves is its nudge (−200 to
    // +400 ms), so it reaches from 0 to 550 here.
    let delay = sliders[1];
    assert_eq!(delay.slider, Slider::Delay);
    let at = |kind, col| MouseInput {
        kind,
        col,
        row: delay.track.y,
    };
    let mut seen = std::collections::BTreeSet::new();
    for px in 0..=delay.track.width * 8 {
        let x = f64::from(delay.track.x) + f64::from(px) / 8.0;
        point(
            &mut state,
            at(MouseKind::Down(MouseButton::Left), delay.track.x),
            x,
        );
        seen.insert(delay.slider.value(&state));
        update(&mut state, Action::KaraokeSlideEnd);
    }
    assert_eq!(seen.first(), Some(&0));
    assert_eq!(seen.last(), Some(&550));
    assert!(seen.iter().all(|v| v % 10 == 0), "{seen:?}");
    let zero = f64::from(delay.track.x) + 0.5 + f64::from(delay.track.width - 1) / 4.0;
    point(
        &mut state,
        at(MouseKind::Down(MouseButton::Left), delay.track.x),
        zero,
    );
    update(&mut state, Action::KaraokeSlideEnd);
    assert_eq!(delay.slider.value(&state), 150);
    assert_eq!(jodeldiplom::core::lag::nudge_ms(&state), 0);
    point(&mut state, at(MouseKind::ScrollUp, delay.track.x), 0.0);
    assert_eq!(delay.slider.value(&state), 160);
    assert_eq!(delay.slider.text(&state).trim(), "160 ms");
    assert_eq!(jodeldiplom::core::lag::nudge_ms(&state), 10);
    for s in sliders.into_iter().filter(|s| s.slider == Slider::Voice) {
        let most = s.slider.most();
        let input = |kind, col| MouseInput {
            kind,
            col,
            row: s.track.y,
        };
        let press = input(MouseKind::Down(MouseButton::Left), s.track.x);
        let drag = input(MouseKind::Drag(MouseButton::Left), 0);
        // A press on the track, then a drag along it a pixel at a time
        // (the GUI, 8 pixels a cell): every whole percent comes by.
        point(&mut state, press, f64::from(s.track.x));
        let mut seen = std::collections::BTreeSet::new();
        for px in 0..=s.track.width * 8 {
            point(&mut state, drag, f64::from(s.track.x) + f64::from(px) / 8.0);
            seen.insert(s.slider.value(&state));
        }
        assert_eq!(seen, (0..=most).collect(), "{:?}", s.slider);
        // 25 % where the knob is drawn for it.
        let quarter = f64::from(s.track.x) + 0.5 + 0.25 * f64::from(s.track.width - 1);
        point(&mut state, drag, quarter);
        assert_eq!(s.slider.value(&state), most / 4, "{:?}", s.slider);
        update(&mut state, Action::KaraokeSlideEnd);
        // A terminal knows only the cell: the nearest whole percent to
        // the cell's middle.
        for i in 0..s.track.width {
            let click = input(MouseKind::Down(MouseButton::Left), s.track.x + i);
            let action = mouse::action(&state, click).expect("a slide");
            update(&mut state, action);
            let share = f64::from(i) / f64::from(s.track.width - 1);
            let nearest = (share * f64::from(most)).round() as i32;
            assert_eq!(s.slider.value(&state), nearest, "{:?}", s.slider);
            update(&mut state, Action::KaraokeSlideEnd);
        }
        // The wheel: whole steps of 5 %, from a level the pointer set. It
        // starts no drag: the next notch steps again, and the pointer
        // moving after it leaves the level where it is.
        point(&mut state, press, quarter);
        update(&mut state, Action::KaraokeSlideEnd);
        let wheel = input(MouseKind::ScrollUp, s.track.x);
        point(&mut state, wheel, 0.0);
        point(&mut state, wheel, 0.0);
        assert_eq!(s.slider.value(&state), most / 4 + 10, "{:?}", s.slider);
        assert_eq!(state.karaoke.dragging, None);
        let moved = input(MouseKind::Moved, s.track.x);
        assert!(mouse::action(&state, moved).is_none());
        assert_eq!(s.slider.value(&state), most / 4 + 10, "{:?}", s.slider);
    }
}

#[test]
fn the_speech_recogniser_is_found_beside_the_speech_model() {
    use jodeldiplom::lyrics::{asr_model_in, ASR_FILE, MODEL_DIR};
    let root = std::env::temp_dir().join(format!("jodel-asr-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let exe = root.join("Contents").join("MacOS");
    let settings = root.join("settings");
    std::fs::create_dir_all(&exe).unwrap();
    assert_eq!(asr_model_in(Some(&exe), Some(&settings)), None);
    let resources = root.join("Contents").join("Resources").join(MODEL_DIR);
    std::fs::create_dir_all(&resources).unwrap();
    std::fs::write(resources.join(ASR_FILE), b"x").unwrap();
    let found = asr_model_in(Some(&exe), Some(&settings)).unwrap();
    assert!(found.ends_with(ASR_FILE), "{found:?}");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_sliders_and_buttons_take_clicks() {
    let mut state = tab();
    let Layout::Main(m) = layout::compute(&state) else {
        panic!()
    };
    let k = m.karaoke.expect("the tab's layout");
    let voice = k.sliders[0];
    assert_eq!(voice.slider, Slider::Voice);
    // A click at the track's end: the most.
    let at = MouseInput {
        kind: MouseKind::Down(MouseButton::Left),
        col: voice.track.right() - 1,
        row: voice.track.y,
    };
    let action = mouse::action(&state, at).expect("a slide");
    update(&mut state, action);
    assert_eq!(state.config.karaoke_voice, 100);
    // Dragged to its start, wherever the pointer goes.
    let drag = MouseInput {
        kind: MouseKind::Drag(MouseButton::Left),
        col: 0,
        row: 0,
    };
    let action = mouse::action(&state, drag).expect("a drag");
    update(&mut state, action);
    assert_eq!(state.config.karaoke_voice, 0);
    let up = MouseInput {
        kind: MouseKind::Up(MouseButton::Left),
        col: 0,
        row: 0,
    };
    let action = mouse::action(&state, up).expect("the end");
    update(&mut state, action);
    assert_eq!(state.karaoke.dragging, None);
    // The button into the mode.
    assert_eq!(
        mouse::hit(&state, k.enter.x + 2, k.enter.y + 1),
        Some(Target::Button(Op::KaraokeEnter))
    );
    // In the mode, waiting: START.
    op(&mut state, Op::KaraokeEnter);
    stopped(&mut state);
    let Layout::Karaoke(km) = layout::compute(&state) else {
        panic!()
    };
    let start = km
        .controls
        .as_ref()
        .expect("controls while waiting")
        .start
        .expect("START");
    assert_eq!(
        mouse::hit(&state, start.x + 1, start.y + 1),
        Some(Target::Button(Op::KaraokeStart))
    );
    // The game is set on the tab, not here: a song is sung at one level
    // from its start to its end. No keys for it, and the ops do nothing.
    use jodeldiplom::core::keymap::{lookup, Key, Meaning};
    let before = state.config.clone();
    for (key, o) in [
        ('g', Op::KaraokeGame),
        ('d', Op::KaraokeLevel),
        ('p', Op::KaraokeNotes),
    ] {
        assert!(!matches!(lookup(&state, Key::ch(key)), Some(Meaning::Op(x)) if x == o));
        op(&mut state, o);
    }
    assert_eq!(state.config, before);
    // On the tab: the game, each level and the notes take clicks.
    use jodeldiplom::core::singing::Level;
    op(&mut state, Op::Back);
    let Layout::Main(m) = layout::compute(&state) else {
        panic!()
    };
    let k = m.karaoke.expect("the tab");
    assert_eq!(
        mouse::hit(&state, k.game.x + 13, k.game.y),
        Some(Target::Button(Op::KaraokeGame))
    );
    assert_eq!(
        mouse::hit(&state, k.show_notes.x + 13, k.show_notes.y),
        Some(Target::Button(Op::KaraokeNotes))
    );
    let levels: Vec<Level> = k.levels.iter().map(|l| l.1).collect();
    assert_eq!(levels, Level::ALL);
    let (hard, _) = k.levels[2];
    let target = mouse::hit(&state, hard.x + 1, hard.y);
    assert_eq!(target, Some(Target::KaraokeLevel(Level::Hard)));
    update(&mut state, Action::KaraokeLevel(Level::Hard));
    assert_eq!(state.config.karaoke_level, Level::Hard);
}

/// In the mode the controls come up only with the pointer near the bottom
/// (and go at once above it), over the picture: the lines stay where they
/// are. Under the slider, the seek bar seeks, except while the song is
/// held for START.
#[test]
fn the_controls_come_up_near_the_bottom_over_the_lines_with_a_seek_bar() {
    use jodeldiplom::view::karaoke::{karaoke_cells, picture};
    use ratatui::buffer::Buffer;
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
    assert!(!karaoke::controls_shown(&state));
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
    let cells = |state: &State| -> (Buffer, Option<u16>) {
        let Layout::Karaoke(k) = layout::compute(state) else {
            panic!("the mode")
        };
        let p = picture(state, &k, 30.0);
        let mut buf = Buffer::empty(p.area);
        karaoke_cells(&mut buf, &p);
        (buf, k.controls.map(|c| c.bar.y))
    };
    let (hidden, none) = cells(&state);
    assert_eq!(none, None);
    // Above the bottom quarter: nothing.
    point(&mut state, moved(rows / 2));
    assert!(!karaoke::controls_shown(&state));
    // In it: the controls, over the bottom; the lines where they were.
    point(&mut state, moved(rows - 2));
    assert!(karaoke::controls_shown(&state));
    let (shown, bar) = cells(&state);
    let bar = bar.expect("the controls' bar");
    for y in 0..bar {
        for x in 0..state.size.0 {
            assert_eq!(
                hidden[(x, y)].symbol(),
                shown[(x, y)].symbol(),
                "({x}, {y})"
            );
        }
    }
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
    // Back up above the bottom: they go at once.
    point(&mut state, moved(1));
    assert!(!karaoke::controls_shown(&state));
}

/// A song cannot be sung until its lyrics are ready: while they are looked
/// for, while their words are timed, or when there are none (none found,
/// an instrumental) the mode does not open for it, START does not play it,
/// and the status line (and the greyed button) says why.
#[test]
fn no_karaoke_for_a_song_until_its_lyrics_are_ready() {
    use jodeldiplom::lyrics::Answer;
    let none = |state: &mut State, index: usize, answer: Answer| {
        let path = state.queue.paths[index].clone();
        if !state.karaoke.lyrics_wanted.contains(&path) {
            state.karaoke.lyrics_wanted.push(path.clone());
        }
        update(state, Action::Lyrics(path, answer));
    };
    let mut state = tab();
    let playing = state.playing.track.as_ref().unwrap().index;
    none(&mut state, playing, Answer::NotFound);
    assert!(karaoke::not_singable(&state, playing).is_some());
    let fx = op(&mut state, Op::KaraokeEnter);
    assert_eq!(state.screen, Screen::Main, "not into the mode");
    assert!(engine(&fx).is_empty(), "nothing stopped: {fx:?}");
    let said = state
        .message
        .as_ref()
        .map(|m| m.text.clone())
        .unwrap_or_default();
    assert!(said.starts_with("No karaoke for "), "{said}");
    // Still looked for (a big folder queued, not got to yet): not yet.
    use jodeldiplom::core::karaoke::{ModelState, SongLyrics};
    let mut state = tab();
    let path = state.queue.paths[playing].clone();
    state
        .karaoke
        .lyrics
        .insert(path.clone(), SongLyrics::Pending);
    assert_eq!(
        karaoke::not_singable(&state, playing),
        Some("still looking for its lyrics")
    );
    op(&mut state, Op::KaraokeEnter);
    assert_eq!(state.screen, Screen::Main);
    // Found, the speech model there to time them: not until they are.
    let lyrics = jodel_lyrics::Lyrics::parse_text("la la", jodel_lyrics::Source::Lrclib);
    state.karaoke.lyrics.insert(
        path.clone(),
        SongLyrics::from_answer(Answer::Found {
            lyrics,
            timed: None,
        }),
    );
    state.karaoke.model = ModelState::Ready;
    assert_eq!(
        karaoke::not_singable(&state, playing),
        Some("its words are still being timed")
    );
    // The timing over (it failed: sung as they are), or no model: in.
    update(
        &mut state,
        Action::Ready {
            path,
            job: jodeldiplom::core::Job::Lyrics,
            facts: None,
        },
    );
    assert_eq!(karaoke::not_singable(&state, playing), None);
    op(&mut state, Op::KaraokeEnter);
    assert_eq!(state.screen, Screen::Karaoke);
    // In the mode, the song held is an instrumental: START says why.
    update(&mut state, Action::Engine(Event::Held { index: 2 }));
    stopped(&mut state);
    none(&mut state, 2, Answer::Instrumental);
    let fx = op(&mut state, Op::KaraokeStart);
    assert!(
        !engine(&fx)
            .iter()
            .any(|c| matches!(c, Command::PlayIn { .. })),
        "{fx:?}"
    );
    let said = state
        .message
        .as_ref()
        .map(|m| m.text.clone())
        .unwrap_or_default();
    assert!(
        said.contains("instrumental") && said.contains("N for the next song"),
        "{said}"
    );
}

#[test]
fn the_tab_says_how_well_the_text_fits_the_song() {
    use jodel_lyrics::{Fit, Lyrics, Source};
    use jodeldiplom::core::karaoke::SongLyrics;
    use jodeldiplom::lyrics::Answer;
    let said = |fit: Option<Fit>| {
        let lyrics = Lyrics::parse_text("One line\nTwo lines", Source::Lrclib);
        let mut timed = lyrics.clone();
        timed.timing.detail = jodel_lyrics::Detail::Words;
        timed.timing.by = Some(jodel_lyrics::Maker::new("ctc:test", 1));
        timed.timing.fit = fit;
        SongLyrics::from_answer(Answer::Found {
            lyrics,
            timed: Some(timed),
        })
        .describe()
    };
    assert_eq!(said(None), "from LRCLIB, words timed (ctc:test)");
    assert!(said(Some(Fit::new(0.78, false))).ends_with("; fits the song (78 % heard)"));
    assert!(said(Some(Fit::new(0.64, true)))
        .ends_with("; its first text did not fit the song, this one does (64 % heard)"));
    assert!(said(Some(Fit::new(0.31, false))).ends_with("; may not fit the song (only 31 % heard)"));
}

#[test]
fn g_d_and_p_set_the_game_and_are_saved() {
    use jodeldiplom::core::singing::Level;
    let mut state = tab();
    assert!(state.config.karaoke_game && state.config.karaoke_notes);
    assert_eq!(state.config.karaoke_level, Level::Medium);
    let fx = op(&mut state, Op::KaraokeGame);
    assert!(!state.config.karaoke_game);
    assert!(fx.iter().any(|e| matches!(e, Effect::SaveConfig)));
    op(&mut state, Op::KaraokeLevel);
    assert_eq!(state.config.karaoke_level, Level::Hard);
    op(&mut state, Op::KaraokeLevel);
    assert_eq!(state.config.karaoke_level, Level::Easy);
    op(&mut state, Op::KaraokeNotes);
    assert!(!state.config.karaoke_notes);
}

#[test]
fn a_song_sung_to_its_end_shows_its_medal_and_keeps_its_best() {
    use jodeldiplom::core::singing::{Level, Medal};
    use std::sync::Arc;
    let mut state = tab();
    // A song of four words, a second each on C4: three of them sung.
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
    let singing = &mut state.karaoke.singing;
    singing.prepare(&path, &lyrics, &melody, 1, Level::Medium);
    for i in 0..450 {
        let t = f64::from(i) / 100.0;
        singing.hear(t, (t < 3.0).then_some(6000.0));
    }
    update(&mut state, Action::Engine(Event::Held { index: 1 }));
    let result = state.karaoke.result.clone().expect("a result");
    assert_eq!((result.points, result.hit, result.words), (300, 3, 4));
    assert_eq!(result.medal, Some(Medal::Silver));
    assert!(result.new_best() && state.karaoke.bests_changed);
    let key = (path.display().to_string(), Level::Medium);
    assert_eq!(state.karaoke.bests[&key].points, 300);
    // Sung worse next time: the best stays.
    state.karaoke.bests_changed = false;
    let singing = &mut state.karaoke.singing;
    for i in 0..450 {
        singing.hear(f64::from(i) / 100.0, None);
    }
    update(&mut state, Action::Engine(Event::Held { index: 1 }));
    let result = state.karaoke.result.clone().expect("a result");
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
