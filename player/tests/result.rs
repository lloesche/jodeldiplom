//! A song's result in Karaoke mode (K3): shown when a song is played to
//! its end (the next held, or the queue played out; sought on or not,
//! nothing scored or not), its medal and best kept, its reveal completed
//! by the first START and the next song started by the one after, its
//! sounds on the frames' clock.

mod common;

use common::karaoke_tab as tab;
use jodel_playback::{Command, Event};
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::singing::Motion;
use jodeldiplom::core::{karaoke, update, Action, Effect, State};
use jodeldiplom::gui::stage_gpu::{Instance, StageFrame};
use jodeldiplom::gui::{font_px, Canvas};
use jodeldiplom::view::mouse::{self, MouseButton, MouseInput, MouseKind};

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
    op(&mut state, Op::KaraokeEnter);
    // Three of the four words sung.
    four_words(&mut state);
    let path = state.queue.paths[0].clone();
    sing_four(&mut state, 3.0);
    ended(&mut state, 0);
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
    ended(&mut state, 0);
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
    ended(&mut state, 0);
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
        ended(&mut state, 0);
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

/// The window at 1600 × 900 on the GPU's path, its frame of `state` drawn:
/// the canvas (what the pointer hits) and the stage drawn, if any.
fn drawn(state: &mut State) -> (Canvas, Option<StageFrame>) {
    let mut canvas = Canvas::new(80, 24, font_px(1.0));
    canvas.set_gpu(true);
    let grid = canvas.fit(1600, 900);
    state.fit = jodeldiplom::view::layout::fit(state, grid);
    let spectrum = jodeldiplom::spectrum::SpectrumFrame::default();
    let scene = jodeldiplom::view::Scene {
        state,
        grid,
        spectrum: &spectrum,
        pointer: None,
        heard: 0.0,
    };
    canvas.draw(&scene, None);
    let tint = jodeldiplom::gui::stage::Tint::STAGE;
    let frame = canvas.stage_frame((state, None), 0.0, 0.0, &tint);
    (canvas, frame)
}

/// What a press of the main button at (`x`, `y`) (window pixels) does in
/// `canvas`: the target drawn there in pixels, over the cells.
fn press_at(canvas: &Canvas, state: &State, (x, y): (f32, f32)) -> Option<Action> {
    let (cw, ch) = canvas.cell();
    let input = MouseInput {
        kind: MouseKind::Down(MouseButton::Left),
        col: (x / cw as f32) as u16,
        row: (y / ch as f32) as u16,
    };
    let over = canvas.target_at(x, y);
    mouse::action_at(state, canvas.grid(), input, f64::from(x) / cw as f64, over)
}

/// Where the stage drew START's pill in `canvas` (window pixels).
fn start_box(canvas: &Canvas) -> [f32; 4] {
    let start = Op::KaraokeStart;
    let mut found = None;
    for y in (0..900).step_by(2) {
        for x in (0..1600).step_by(2) {
            if canvas.target_at(x as f32, y as f32) == Some(start) {
                let (x, y) = (x as f32, y as f32);
                let b: [f32; 4] = found.unwrap_or([x, y, x, y]);
                found = Some([b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)]);
            }
        }
    }
    found.expect("START's pill")
}

/// START's pill is START by click, as `enter` is: a press inside it gives
/// START, one just outside nothing; drawn where the pointer finds it (one
/// place says where), and never moved by the controls coming up. On a
/// result a press during the reveal completes the reveal, the next starts
/// the song; once the reveal is over the first starts it. Entering the
/// mode, it is there before any result.
#[test]
fn the_start_pill_is_start_by_click_during_and_after_the_reveal() {
    let press = |state: &mut State, at: (f32, f32)| {
        let (canvas, _) = drawn(state);
        let action = press_at(&canvas, state, at).expect("START");
        assert!(matches!(action, Action::Op(Op::KaraokeStart)), "{action:?}");
        update(state, action)
    };
    let starts = |fx: &[Effect]| {
        engine(fx)
            .iter()
            .any(|c| matches!(c, Command::PlayIn { index: 1, .. }))
    };
    let sung = || {
        let mut state = tab();
        op(&mut state, Op::KaraokeEnter);
        four_words(&mut state);
        sing_four(&mut state, 4.0);
        ended(&mut state, 0);
        state
    };
    let mut state = sung();
    state.now_ms += 500;
    let (canvas, frame) = drawn(&mut state);
    let [x0, y0, x1, y1] = start_box(&canvas);
    assert!(x0 > 0.0 && x1 < 1600.0 && y0 > 450.0 && y1 < 900.0 && x1 - x0 > 40.0);
    let middle = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    // The pill drawn is round that box.
    let round = |i: &&Instance| {
        let [a, b, c, d] = i.quad;
        a <= x0 && b <= y0 && c >= x1 && d >= y1 && c - a < (x1 - x0) * 2.0
    };
    let frame = frame.expect("the stage");
    assert!(frame.instances.iter().any(|i| round(&i)), "the pill");
    // Just outside it: nothing (the picture under it takes no clicks).
    for outside in [
        (x0 - 4.0, middle.1),
        (x1 + 4.0, middle.1),
        (middle.0, y0 - 4.0),
    ] {
        assert!(press_at(&canvas, &state, outside).is_none(), "{outside:?}");
    }
    // The controls up: START stays where it was.
    let rows = canvas.grid().rows;
    let from_bottom = mouse::from_bottom(rows, rows - 1);
    update(&mut state, Action::PointerMoved { from_bottom });
    let now = state.now_ms + karaoke::controls::FADE_MS;
    update(&mut state, Action::Tick(now));
    assert!(karaoke::controls::shown(&state));
    assert_eq!(start_box(&drawn(&mut state).0), [x0, y0, x1, y1]);
    // Pressed during the reveal: the reveal completed, nothing played.
    let fx = press(&mut state, middle);
    assert!(engine(&fx).is_empty() && state.karaoke.countdown.is_none());
    assert_eq!(karaoke::result_motion(&state), Some(Motion::Twinkling));
    // Pressed again: the song; then no START to press.
    assert!(starts(&press(&mut state, middle)));
    assert!(karaoke::result_shown(&state).is_none());
    let (canvas, _) = drawn(&mut state);
    assert!(press_at(&canvas, &state, middle).is_none());

    // After the reveal: the first press starts it.
    let mut state = sung();
    let end = karaoke::result_shown(&state).unwrap().reveal.end();
    state.now_ms += (end * 1000.0) as u64 + 1;
    assert!(starts(&press(&mut state, middle)));

    // Just entered, no result: START alone, a press starts the song held.
    let mut state = tab();
    op(&mut state, Op::KaraokeEnter);
    let held = state.karaoke.held.expect("a song held");
    let (canvas, _) = drawn(&mut state);
    let [x0, y0, x1, y1] = start_box(&canvas);
    let fx = press(&mut state, ((x0 + x1) / 2.0, (y0 + y1) / 2.0));
    assert!(engine(&fx)
        .iter()
        .any(|c| matches!(c, Command::PlayIn { index, .. } if *index == held)));
}

/// The reveal's sounds go by the frames' clock, as the picture does: the
/// app ticks the core every 250 ms (`now_ms`) and draws a frame every
/// 1/60 s (`drawn_ms`), and each beat sounds once, on the first frame at
/// or past it, none bunched up to the next tick nor dropped as stale.
#[test]
fn the_reveal_sounds_on_the_frames_clock() {
    use jodeldiplom::sfx::reveal::{sounds_of, RevealSounds};
    let mut state = tab();
    op(&mut state, Op::KaraokeEnter);
    four_words(&mut state);
    sing_four(&mut state, 4.0);
    let start = 10_000;
    update(&mut state, Action::Tick(start));
    state.drawn_ms = Some(start);
    ended(&mut state, 0);
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
        .flat_map(|&(at, b)| sounds_of(b, &shown).map(move |_| at))
        .collect();
    assert_eq!(heard.len(), due.len(), "{heard:?}\n{due:?}");
    for (t, at) in heard.iter().zip(&due) {
        assert!(
            *t >= *at && *t < at + 1.0 / 60.0 + 1e-3,
            "due {at}, heard {t}"
        );
    }
}

/// Track `index` made a song of 245 s: a line of four words a second each
/// on C4 from its start, and another from 236 s, its singing ready.
fn sung_song(state: &mut State, index: usize) {
    use std::sync::Arc;
    let fps = 100;
    let mut melody = jodel_rhythm::waveform::Melody {
        cents: vec![0; 245 * fps],
        sure: vec![0; 245 * fps],
    };
    let line = |from: f64| jodel_lyrics::Line {
        words: (0..4)
            .map(|k| jodel_lyrics::Word {
                start: Some(from + f64::from(k)),
                end: Some(from + f64::from(k) + 1.0),
                ..jodel_lyrics::Word::new("la")
            })
            .collect(),
        ..jodel_lyrics::Line::default()
    };
    for from in [0, 236] {
        for i in from * fps..(from + 4) * fps {
            melody.cents[i] = 6000;
            melody.sure[i] = 255;
        }
    }
    let lyrics = Arc::new(jodel_lyrics::Lyrics {
        lines: vec![line(0.0), line(236.0)],
        source: jodel_lyrics::Source::Unknown,
        timing: jodel_lyrics::Timing::default(),
    });
    let path = state.queue.paths[index].clone();
    state.karaoke.singing.prepare(
        &path,
        &lyrics,
        &jodeldiplom::core::singing::melodic(&melody),
    );
}

/// The singer heard from `from` to `to` seconds, a frame every 10 ms, in
/// tune before `tune` seconds, silent after; the engine telling where the
/// song is as it goes.
fn hear(state: &mut State, from: f64, to: f64, tune: f64) {
    let mut i = (from * 100.0).round() as u32;
    while f64::from(i) / 100.0 <= to {
        let t = f64::from(i) / 100.0;
        state.karaoke.singing.hear(t, (t < tune).then_some(6000.0));
        state.playing.position = t;
        i += 1;
    }
}

/// In the mode, the game on, two songs queued: song `index` held for
/// START, started, heard from its start (its count-in over).
fn started(index: usize) -> State {
    use jodel_playback::PlayState;
    let mut state = tab();
    state.queue.paths.truncate(2);
    state.queue.current = Some(index);
    state.playing.track = None;
    state.playing.state = PlayState::Stopped;
    op(&mut state, Op::KaraokeEnter);
    assert_eq!(state.karaoke.held, Some(index));
    sung_song(&mut state, index);
    let fx = op(&mut state, Op::KaraokeStart);
    assert!(engine(&fx)
        .iter()
        .any(|c| matches!(c, Command::PlayIn { index: i, .. } if *i == index)));
    state.playing.track = Some(common::now_track(44_100, state.engine_settings(), index));
    update(&mut state, Action::Engine(Event::State(PlayState::Playing)));
    state.playing.position = 0.0;
    let now = state.now_ms + 1_000;
    update(&mut state, Action::Tick(now));
    assert!(state.karaoke.countdown.is_none() && state.karaoke.held.is_none());
    state
}

/// The engine plays track `index` to its end: the next held for START, or
/// (the last) the queue ended.
fn ended(state: &mut State, index: usize) {
    use jodel_playback::PlayState;
    update(
        state,
        Action::Engine(Event::TrackEnded {
            index,
            finished: true,
        }),
    );
    if index + 1 < state.queue.paths.len() {
        update(state, Action::Engine(Event::Held { index: index + 1 }));
    } else {
        update(state, Action::Engine(Event::QueueEnded));
    }
    update(state, Action::Engine(Event::State(PlayState::Stopped)));
}

/// The result shown stays, its reveal running to its end and on, until
/// START: the first completes it, the next starts `next` (the song held,
/// or after the last the last again), the result gone.
fn stays_until_start(state: &mut State, next: usize) {
    let shown = karaoke::result_shown(state).expect("shown").clone();
    assert!(shown.revealing(karaoke::clock_ms(state)));
    for _ in 0..8 {
        let now = state.now_ms + 250;
        update(state, Action::Tick(now));
        assert_eq!(karaoke::result_shown(state), Some(&shown), "it stays");
    }
    assert_eq!(karaoke::result_motion(state), Some(Motion::Revealing));
    let fx = op(state, Op::KaraokeStart);
    assert!(engine(&fx).is_empty(), "the reveal completed, no more");
    assert_eq!(karaoke::result_motion(state), Some(Motion::Twinkling));
    let fx = op(state, Op::KaraokeStart);
    assert!(
        engine(&fx)
            .iter()
            .any(|c| matches!(c, Command::PlayIn { index, .. } if *index == next)),
        "{fx:?}"
    );
    assert!(karaoke::result_shown(state).is_none() && state.karaoke.result.is_none());
}

/// As the user sang it: three words hit at the song's start (300 points),
/// sought on to a few seconds before its end, and played out. Its result
/// shows while the next song is held, and as well after the last song of
/// the queue (no next to hold); played on without the seek, the same. Sought
/// past its second line, the song was not sung whole: no medal, no best.
#[test]
fn a_song_sought_on_and_played_out_shows_its_result() {
    use jodeldiplom::core::singing::Verdict;
    for (index, seek) in [(0, true), (1, true), (0, false), (1, false)] {
        let mut state = started(index);
        hear(&mut state, 0.0, 4.5, 3.0);
        assert_eq!(state.karaoke.singing.tally.score, 300);
        if seek {
            let fx = update(&mut state, Action::SeekTo(240.0 / 245.0));
            assert!(engine(&fx).iter().any(|c| matches!(c, Command::Seek(_))));
            update(
                &mut state,
                Action::Engine(Event::Seeked {
                    index,
                    seconds: 240.0,
                }),
            );
            hear(&mut state, 240.0, 245.0, 0.0);
        } else {
            hear(&mut state, 4.5, 245.0, 0.0);
        }
        ended(&mut state, index);
        let shown = karaoke::result_shown(&state)
            .unwrap_or_else(|| panic!("a result for song {index}, sought {seek}"))
            .clone();
        let r = &shown.result;
        assert_eq!(
            (r.points, r.hit, r.words),
            (300, 3, if seek { 4 } else { 8 })
        );
        let verdict = if seek { Verdict::Whole } else { Verdict::Cheer };
        assert_eq!(r.verdict(), verdict, "song {index}, sought {seek}");
        assert_eq!(r.new_best(), !seek);
        // After the last, START sings it again, and says so.
        let next = jodeldiplom::view::karaoke::result_now(&state).unwrap();
        let words = ["Next: Morning Light", "Again: Morning Light"][index];
        assert_eq!(next.next_words(), words);
        stays_until_start(&mut state, 1);
    }
}

/// Nothing scored (not a note heard, or no notes made for the song): the
/// song played out shows its result all the same, a wink at the microphone,
/// with no medal and no best (a best kept before stays as it was); its
/// reveal counts nothing up, fills no bar, pops no gems and cheers for
/// none of it, and runs to its end as ever.
#[test]
fn a_song_played_out_with_nothing_scored_shows_its_result() {
    use jodeldiplom::core::singing::verdict_words;
    use jodeldiplom::core::singing::{Beat, Verdict};
    use jodeldiplom::sfx::conductor::{Diploma, Entry, Sheet};
    use jodeldiplom::sfx::reveal::sounds_of;
    for (notes, best) in [(true, None), (true, Some(1_200)), (false, None)] {
        let mut state = started(0);
        let key = state.queue.paths[0].display().to_string();
        if let Some(points) = best {
            let kept = jodeldiplom::scores::Best {
                points,
                hit: 4,
                words: 8,
            };
            state.karaoke.bests.insert(key.clone(), kept);
        }
        if !notes {
            // The singing is of another song: none was made for this one.
            sung_song(&mut state, 1);
        }
        state.playing.position = 245.0;
        ended(&mut state, 0);
        let shown = karaoke::result_shown(&state).expect("a result").clone();
        let r = &shown.result;
        assert_eq!((r.points, r.hit, r.words, r.streak), (0, 0, 0, 0));
        assert_eq!(
            (r.verdict(), r.medal, r.best),
            (Verdict::Silent, None, best)
        );
        assert!(!r.new_best() && !state.karaoke.bests_changed);
        assert_eq!(state.karaoke.bests.get(&key).map(|b| b.points), best);
        assert_eq!(verdict_words(r).0, "Silent night?");
        let reveal = &shown.reveal;
        assert!(!reveal
            .beats()
            .iter()
            .any(|(_, b)| matches!(b, Beat::Tick(_) | Beat::Gem(..) | Beat::NewBest)));
        assert_eq!(reveal.points_at(reveal.end()), 0);
        // Nothing cheered: no ding, the pen alone for the words.
        assert!(reveal.when(Beat::Points).is_some() && reveal.when(Beat::Words).is_some());
        assert_eq!(sounds_of(Beat::Points, &shown).count(), 0);
        let words: Vec<Diploma> = sounds_of(Beat::Words, &shown).collect();
        let pen = Diploma::Sheet {
            sound: Sheet::Scribbled(Entry::Words),
            gain: 1.0,
            pan: 0.0,
        };
        assert_eq!(words, [pen]);
        assert!(sounds_of(Beat::Verdict(Verdict::Silent), &shown).count() > 0);
        let lines = jodeldiplom::view::karaoke::lines_now(&state, 0.0);
        assert!(
            lines.iter().any(|l| l.text.contains("0 of 0 words")),
            "{lines:?}"
        );
        stays_until_start(&mut state, 1);
    }
}

/// The engine's word that track `index` started (from its start), on the
/// output `now_track` has.
fn track_started(state: &mut State, index: usize) {
    let track = common::now_track(44_100, state.engine_settings(), index);
    let start = jodel_playback::TrackStart {
        index,
        head: 0,
        info: track.info,
        route: track.route,
        output: track.output,
        device: track.device,
        output_frame: 0,
        from_frame: 0,
    };
    update(state, Action::Engine(Event::TrackStarted(Box::new(start))));
}

/// No result where no song was heard to its end in the mode: one left for
/// the next (`N`) or stopped there; one played out away from the mode, or
/// after leaving it mid-song. One shown goes when the mode is left and
/// entered again, and when the next track is heard (a mix run into it).
#[test]
fn no_result_where_no_song_was_played_out_in_the_mode() {
    use jodel_playback::PlayState;
    let none = |state: &State| state.karaoke.result.is_none();
    let cut = |state: &mut State, index| {
        let event = Event::TrackEnded {
            index,
            finished: false,
        };
        update(state, Action::Engine(event));
        update(state, Action::Engine(Event::State(PlayState::Stopped)));
    };
    // `N`: the next held, this one stopped short.
    let mut state = started(0);
    hear(&mut state, 0.0, 4.5, 3.0);
    assert_eq!(engine(&op(&mut state, Op::Next)), [&Command::Stop]);
    cut(&mut state, 0);
    assert!(none(&state));
    // Stopped in the mode.
    let mut state = started(0);
    hear(&mut state, 0.0, 4.5, 3.0);
    cut(&mut state, 0);
    assert!(none(&state));
    // Played out away from the mode.
    let mut state = tab();
    sung_song(&mut state, 1);
    hear(&mut state, 0.0, 245.0, 3.0);
    let event = Event::TrackEnded {
        index: 1,
        finished: true,
    };
    update(&mut state, Action::Engine(event));
    assert!(none(&state));
    // Left mid-song, then played out.
    let mut state = started(0);
    hear(&mut state, 0.0, 4.5, 3.0);
    op(&mut state, Op::Back);
    hear(&mut state, 4.5, 245.0, 0.0);
    let event = Event::TrackEnded {
        index: 0,
        finished: true,
    };
    update(&mut state, Action::Engine(event));
    track_started(&mut state, 1);
    assert!(none(&state));
    // Left with a result shown, then entered again: gone.
    let mut state = started(0);
    hear(&mut state, 0.0, 245.0, 3.0);
    ended(&mut state, 0);
    assert!(karaoke::result_shown(&state).is_some());
    op(&mut state, Op::Back);
    assert!(karaoke::result_shown(&state).is_none());
    op(&mut state, Op::KaraokeEnter);
    assert!(none(&state) && karaoke::result_shown(&state).is_none());
    // A mix hands over to the next song in the mode (A heard to the end of
    // its window, B starting on that frame): no result once B is heard.
    let mut state = started(0);
    hear(&mut state, 0.0, 200.0, 3.0);
    let event = Event::TrackEnded {
        index: 0,
        finished: true,
    };
    update(&mut state, Action::Engine(event));
    assert!(!none(&state), "A's end is an end as the engine tells it");
    track_started(&mut state, 1);
    assert!(none(&state) && karaoke::result_shown(&state).is_none());
}

/// `p` as the window sends it in the mode.
fn demo_key(state: &mut State) {
    use jodeldiplom::core::keymap::{lookup, Key, Meaning};
    let Some(Meaning::Op(op)) = lookup(state, Key::ch('p')) else {
        panic!("p does nothing");
    };
    update(state, Action::Op(op));
}

/// With `p` held (no microphone open: none is in a test), the demo sings
/// the four words, the window's frames asking every 16 ms; then the song
/// is played out. Its result shows as any song's, every word hit, but it
/// is no best: none kept where there was none, the one before staying,
/// the best's line saying why.
#[test]
fn a_song_the_demo_sang_shows_its_result_and_sets_no_best() {
    use jodeldiplom::core::singing::Medal;
    for before in [None, Some(100)] {
        let mut state = tab();
        op(&mut state, Op::KaraokeEnter);
        four_words(&mut state);
        let key = state.queue.paths[0].display().to_string();
        if let Some(points) = before {
            let best = jodeldiplom::scores::Best {
                points,
                hit: 1,
                words: 4,
            };
            state.karaoke.bests.insert(key.clone(), best);
        }
        demo_key(&mut state);
        let mut t = 0.0;
        while t <= 4.5 {
            let k = &mut state.karaoke;
            k.demo.sing(&mut k.singing, t);
            t += 0.016;
        }
        ended(&mut state, 0);
        let r = &state.karaoke.result.as_ref().expect("a result").result;
        assert_eq!((r.points, r.hit, r.words), (400, 4, 4));
        assert_eq!(r.medal, Some(Medal::Gold));
        assert!(r.demo && !r.new_best());
        assert_eq!(state.karaoke.bests.get(&key).map(|b| b.points), before);
        assert!(!state.karaoke.bests_changed);
        assert!(jodeldiplom::view::karaoke::best_words(r).contains("demo"));
    }
}

/// `p` is the demo key in the mode only (on the tab it shows or hides the
/// notes), held while it repeats; let go of (as the window tells it: the
/// key alone, whatever is held with it), or the mode left, it stops.
#[test]
fn the_demo_key_is_held_in_the_mode_only() {
    use iced::keyboard::Key as IcedKey;
    use jodeldiplom::gui::keys::lets_go_of_demo;
    let mut state = tab();
    let notes = state.config.karaoke_notes;
    demo_key(&mut state);
    assert!(!state.karaoke.demo.held() && state.config.karaoke_notes != notes);
    op(&mut state, Op::KaraokeEnter);
    demo_key(&mut state);
    demo_key(&mut state);
    assert!(state.karaoke.demo.held());
    assert!(!lets_go_of_demo(&state, &IcedKey::Character("x".into())));
    assert!(lets_go_of_demo(&state, &IcedKey::Character("p".into())));
    update(&mut state, Action::DemoLetGo);
    assert!(!state.karaoke.demo.held());
    assert!(!lets_go_of_demo(&state, &IcedKey::Character("p".into())));
    demo_key(&mut state);
    op(&mut state, Op::Back);
    assert!(!state.karaoke.demo.held(), "the mode left");
}
