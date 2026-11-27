//! A song's preparation as it goes: what the workers tell of each step
//! ([`Heard`]) kept for the song and shown, the queue row's caption
//! following the step under way, and the preparation view resting when
//! nothing changes.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use jodel_rhythm::key::Key;
use jodel_rhythm::waveform::Lane;
use jodeldiplom::core::prep::{
    BeatsFound, Preview, Separating, Story, Words, CAPTION_MS, HOLD_MS, REVEAL_MS, SHOWN_MS,
};
use jodeldiplom::core::{update, Action, Heard, Job, Side, State};
use jodeldiplom::view::list::{Body, Pane};
use jodeldiplom::view::prep::{self, Hero, Kind, PrepView, Stand};

/// The playing state on the queue, its second song selected, lyrics and
/// covers got ready too (as the app), at `now`; that song.
fn on_the_queue(now: u64) -> (State, PathBuf) {
    let mut state = common::playing_state(120, 40);
    state.preparing.extend([Job::Lyrics, Job::Cover]);
    state.side = Side::Queue;
    state.queue.cursor = 1;
    update(&mut state, Action::Tick(now));
    let path = state.queue.paths[1].clone();
    (state, path)
}

fn working(state: &mut State, path: &Path, job: Job, on: bool) {
    update(
        state,
        Action::Working {
            path: path.to_owned(),
            job,
            on,
        },
    );
}

fn heard(state: &mut State, path: &Path, part: Heard) {
    update(
        state,
        Action::Heard {
            path: path.to_owned(),
            part,
        },
    );
}

/// The clock at `now`, as the frame drawn then.
fn at(state: &mut State, now: u64) {
    update(state, Action::Tick(now));
    state.drawn_ms = Some(now);
}

fn stage(view: &PrepView, kind: Kind) -> (Stand, String) {
    let s = view.stages.iter().find(|s| s.kind == kind).expect("listed");
    (s.stand, s.line.clone())
}

/// A lane of `frames` at `level`.
fn lane(frames: usize, level: u8) -> Lane {
    Lane {
        peak: vec![level; frames],
        rms: vec![level / 2; frames],
    }
}

/// An analysis of a 60 s song at 120 BPM, a bar every four beats.
fn analysis() -> jodel_rhythm::Analysis {
    let beats: Vec<f64> = (0..120).map(|k| 0.5 * f64::from(k)).collect();
    jodel_rhythm::Analysis {
        duration: 60.0,
        downbeats: (0..120).step_by(4).collect(),
        local_bpm: vec![120.0; beats.len()],
        summary: Some(jodel_rhythm::Summary {
            bpm: 123.0,
            low: 123.0,
            high: 123.0,
            steady: true,
        }),
        beats,
        ..jodel_rhythm::Analysis::default()
    }
}

/// Two lines of words, timed from 10 s.
fn timed_words() -> Preview {
    let mut lyrics = jodel_lyrics::Lyrics::parse(
        "[00:10.00]one two three\n[00:12.00]four five six\n",
        jodel_lyrics::Source::Lrclib,
    );
    for (l, line) in lyrics.lines.iter_mut().enumerate() {
        for (k, word) in line.words.iter_mut().enumerate() {
            let start = 10.0 + 2.0 * l as f64 + 0.5 * k as f64;
            (word.start, word.end) = (Some(start), Some(start + 0.4));
        }
    }
    Preview::of(&lyrics).expect("timed")
}

/// What a song's work tells as it goes is kept for the song and shown:
/// the key found, the stems as they are separated (their share and their
/// lanes so far), the beat with its tempo and its ticks coming in, the
/// words looked up, then timed and lit in time.
#[test]
fn partial_results_fill_the_songs_preparation_and_its_view() {
    let (mut state, path) = on_the_queue(1_000);
    assert!(PrepView::of(&state).is_none(), "nothing worked on yet");
    working(&mut state, &path, Job::Rhythm, true);
    let view = PrepView::of(&state).expect("shown");
    assert_eq!(
        stage(&view, Kind::Key),
        (Stand::Running, "Listening for the key…".into())
    );
    assert_eq!(view.hero, Hero::Waiting);
    // The key, then the stems so far: half the song separated.
    heard(&mut state, &path, Heard::Key(Some(Key::on(0, true))));
    let half = [
        &lane(3_000, 200),
        &lane(3_000, 120),
        &lane(3_000, 90),
        &lane(3_000, 60),
    ];
    heard(
        &mut state,
        &path,
        Heard::Separating(Arc::new(Separating::of(half, 6_000))),
    );
    let view = PrepView::of(&state).unwrap();
    assert_eq!(
        stage(&view, Kind::Key),
        (Stand::Found, "Found it: C minor".into())
    );
    assert_eq!(
        stage(&view, Kind::Stems),
        (Stand::Running, "Taking the song apart… 50 %".into())
    );
    // The key just found shows big for a while, on a keyboard.
    let Hero::Key(key) = &view.hero else {
        panic!("{view}");
    };
    assert_eq!(key.lit, [0, 3, 7]);
    // After it, the stems as they are separated: the voice's lane filled
    // halfway, the rest still to come.
    at(&mut state, 1_000 + HOLD_MS);
    let view = PrepView::of(&state).unwrap();
    let Hero::Stems(stems) = &view.hero else {
        panic!("{view}");
    };
    assert_eq!(stems.share, 0.5);
    let (name, voice) = &stems.lanes[0];
    assert_eq!(*name, "Voice");
    assert!(voice[..31].iter().all(|&h| h > 0.2) && voice[33..].iter().all(|&h| h == 0.0));
    heard(&mut state, &path, Heard::Stems);
    let view = PrepView::of(&state).unwrap();
    assert_eq!(stage(&view, Kind::Stems).0, Stand::Found);
    assert_eq!(
        stage(&view, Kind::Beat),
        (Stand::Running, "Finding the downbeat…".into())
    );
    // The beat: its tempo, its ticks coming in left to right.
    let t = 1_000 + HOLD_MS;
    let beats = BeatsFound::of(&analysis(), &lane(6_000, 180));
    assert_eq!(beats.from, 20.0, "from the first downbeat a third in");
    assert_eq!(beats.beats.len(), 16);
    heard(&mut state, &path, Heard::Beats(Arc::new(beats)));
    working(&mut state, &path, Job::Rhythm, false);
    let view = PrepView::of(&state).unwrap();
    assert_eq!(
        stage(&view, Kind::Beat),
        (Stand::Found, "Found it: 123 BPM".into())
    );
    assert_eq!(view.headline, "Found it: 123 BPM");
    let ticks = |state: &State| match PrepView::of(state).unwrap().hero {
        Hero::Beat(b) => b.ticks.len(),
        other => panic!("{other:?}"),
    };
    assert_eq!(ticks(&state), 1);
    at(&mut state, t + REVEAL_MS / 2);
    let halfway = ticks(&state);
    assert!((2..16).contains(&halfway), "{halfway}");
    at(&mut state, t + REVEAL_MS);
    assert_eq!(ticks(&state), 16);
    // The words: looked up, then timed, lit in time as they will be sung.
    working(&mut state, &path, Job::Lyrics, true);
    heard(&mut state, &path, Heard::Lyrics(Words::Looking));
    let view = PrepView::of(&state).unwrap();
    assert_eq!(
        stage(&view, Kind::Words),
        (Stand::Running, "Looking up the words…".into())
    );
    heard(&mut state, &path, Heard::Lyrics(Words::Timing));
    assert_eq!(
        stage(&PrepView::of(&state).unwrap(), Kind::Words),
        (Stand::Running, "Timing the words…".into())
    );
    let t = t + REVEAL_MS;
    heard(
        &mut state,
        &path,
        Heard::Lyrics(Words::Timed(Arc::new(timed_words()))),
    );
    working(&mut state, &path, Job::Lyrics, false);
    // A quarter second in: the first word half lit, the next not yet.
    at(&mut state, t + 200);
    let view = PrepView::of(&state).unwrap();
    assert_eq!(
        stage(&view, Kind::Words),
        (Stand::Found, "Timed, word by word".into())
    );
    let Hero::Words(words) = &view.hero else {
        panic!("{view}");
    };
    assert_eq!(words.lines.len(), 2);
    let (first, lit) = &words.lines[0][0];
    assert_eq!(first, "one");
    assert!((lit - 0.5).abs() < 0.01, "{lit}");
    assert_eq!(words.lines[0][1].1, 0.0);
    // Everything but the cover done: still shown, waiting for it.
    let view = PrepView::of(&state).unwrap();
    assert_eq!(stage(&view, Kind::Cover).0, Stand::Waiting);
    assert!(!view.done);
}

/// A queue row's caption says the step under way: the key listened for,
/// the song taken apart, the beat found, the words looked up, heard or
/// timed, the cover cut out; each result found shows instead for a
/// moment, with a tick; with nothing under way, none.
#[test]
fn a_rows_caption_follows_the_step_running() {
    let (mut state, path) = on_the_queue(1_000);
    let caption = |state: &State| {
        let Body::Queue(q) = Pane::of(state, 960.0).body else {
            panic!("the queue");
        };
        let row = q.rows.iter().find(|r| r.index == 1).unwrap();
        row.caption.clone().map(|c| (c.text, c.found))
    };
    let running = |text: &str| Some((text.to_owned(), false));
    let found = |text: &str| Some((text.to_owned(), true));
    assert_eq!(caption(&state), None);
    working(&mut state, &path, Job::Rhythm, true);
    assert_eq!(caption(&state), running("listening for the key…"));
    heard(&mut state, &path, Heard::Key(Some(Key::on(9, true))));
    assert_eq!(caption(&state), found("A minor ✓"));
    // The found result for a moment, then the step under way again.
    let mut now = 1_000 + CAPTION_MS;
    at(&mut state, now);
    assert_eq!(caption(&state), running("finding the beat…"));
    let so_far = [
        &lane(10, 100),
        &lane(10, 100),
        &lane(10, 100),
        &lane(10, 100),
    ];
    heard(
        &mut state,
        &path,
        Heard::Separating(Arc::new(Separating::of(so_far, 100))),
    );
    assert_eq!(caption(&state), running("taking the song apart…"));
    heard(&mut state, &path, Heard::Stems);
    assert_eq!(caption(&state), found("stems ✓"));
    now += CAPTION_MS;
    at(&mut state, now);
    assert_eq!(caption(&state), running("finding the beat…"));
    let beats = BeatsFound::of(&analysis(), &lane(6_000, 180));
    heard(&mut state, &path, Heard::Beats(Arc::new(beats)));
    working(&mut state, &path, Job::Rhythm, false);
    assert_eq!(caption(&state), found("123 BPM ✓"));
    now += CAPTION_MS;
    at(&mut state, now);
    assert_eq!(caption(&state), None, "nothing under way");
    // The words: looked up, heard (none found), timed.
    working(&mut state, &path, Job::Lyrics, true);
    assert_eq!(caption(&state), running("looking up the words…"));
    heard(&mut state, &path, Heard::Lyrics(Words::Hearing));
    assert_eq!(caption(&state), running("listening for the words…"));
    heard(&mut state, &path, Heard::Lyrics(Words::Timing));
    assert_eq!(caption(&state), running("timing the words…"));
    // The cover, started last, says its step.
    working(&mut state, &path, Job::Cover, true);
    assert_eq!(caption(&state), running("cutting out the cover…"));
    working(&mut state, &path, Job::Cover, false);
    assert_eq!(caption(&state), running("timing the words…"));
}

/// The preparation view moves only while it shows and something changes:
/// a worker at its song, a result coming in and held, its song just
/// ready. Its song waiting for the rest of its work with nothing new, it
/// rests (the window draws no frames for it, and nothing is due), as it
/// does once its song has been ready a while (the view goes). Its story
/// slot shows what is told of it, the scene drawn last its headline.
#[test]
fn an_idle_preparation_view_rests() {
    let (mut state, path) = on_the_queue(1_000);
    assert!(!prep::moving(&state));
    working(&mut state, &path, Job::Rhythm, true);
    assert!(prep::moving(&state), "a worker at it");
    heard(&mut state, &path, Heard::Key(Some(Key::on(4, false))));
    working(&mut state, &path, Job::Rhythm, false);
    assert!(prep::moving(&state), "a result held");
    // Past the hold, waiting for its lyrics and its cover: at rest, and
    // nothing due but what is to come of it.
    at(&mut state, 1_000 + HOLD_MS);
    assert!(PrepView::of(&state).is_some(), "still shown");
    assert!(!prep::moving(&state), "at rest");
    assert_eq!(jodeldiplom::core::prep::due(&state), None);
    // A story told: the slot shows it, as it is written.
    let picture = Arc::new(image::RgbaImage::new(4, 4));
    for part in [
        Story::Words("Once upon".into()),
        Story::Words(" a time".into()),
        Story::Scene {
            index: 0,
            of: 3,
            picture,
        },
    ] {
        heard(&mut state, &path, Heard::Story(part));
    }
    let view = PrepView::of(&state).unwrap();
    let Hero::Story(story) = view.hero else {
        panic!("the story shown");
    };
    assert_eq!(story.text, "Once upon a time");
    assert_eq!(story.scenes.len(), 1);
    assert_eq!(view.headline, "Scene 1 of 3 drawn");
    // Every step done: it shows for a while (and moves), then goes.
    let now = 1_000 + 2 * HOLD_MS;
    at(&mut state, now);
    use jodeldiplom::core::{Facts, LyricsFacts, RhythmFacts};
    let analysed = Facts::Rhythm(RhythmFacts {
        bpm: Some(120.0),
        key: Some("12B".into()),
        stems: false,
    });
    for (job, facts) in [
        (Job::Rhythm, analysed),
        (Job::Lyrics, Facts::Lyrics(LyricsFacts::Timed)),
        (Job::Cover, Facts::Cover(true)),
    ] {
        update(
            &mut state,
            Action::Ready {
                path: path.to_owned(),
                job,
                facts: Some(facts),
            },
        );
    }
    let view = PrepView::of(&state).unwrap();
    assert!(view.done && prep::moving(&state));
    assert_eq!(view.headline, "Ready");
    assert_eq!(jodeldiplom::core::prep::due(&state), Some(now + SHOWN_MS));
    at(&mut state, now + SHOWN_MS);
    assert!(PrepView::of(&state).is_none() && !prep::moving(&state));
    assert_eq!(jodeldiplom::core::prep::due(&state), None);
}

/// Through the app: the queue shown, its selected song's preparation at
/// rest (waiting for the rest of its work, nothing new) asks for no
/// frames; a worker at it again, it moves.
#[test]
fn an_idle_preparation_view_asks_the_app_for_no_frames() {
    let _guard = jodel_testkit::real_time();
    let tree = common::music_tree("prep-rest");
    let mut options = jodeldiplom::app::AppOptions::new(tree.album.clone());
    options.output = jodeldiplom::app::Output::Null(jodel_output::SimClock::Realtime);
    options.devices = Arc::new(|| Ok(Vec::new()));
    let mut app = common::fitted_app(options, 120, 36);
    assert!(common::wait_for(&mut app, Duration::from_secs(20), |a| !a
        .state
        .browser
        .loading));
    // A song elsewhere, queued and selected, its key found a while ago.
    let song = tree.root.join("elsewhere.flac");
    app.state.queue.paths = vec![song.clone()];
    app.state.queue.cursor = 0;
    app.state.side = Side::Queue;
    let now = app.now_ms();
    app.dispatch(Action::Tick(now));
    let tell = |app: &mut jodeldiplom::app::App, action| app.dispatch(action);
    tell(
        &mut app,
        Action::Working {
            path: song.clone(),
            job: Job::Rhythm,
            on: true,
        },
    );
    assert!(app.animating(), "a worker at it");
    tell(
        &mut app,
        Action::Heard {
            path: song.clone(),
            part: Heard::Key(Some(Key::on(4, false))),
        },
    );
    tell(
        &mut app,
        Action::Working {
            path: song.clone(),
            job: Job::Rhythm,
            on: false,
        },
    );
    // Its lyrics still to come: shown, at rest once the result's hold is
    // over.
    app.dispatch(Action::Tick(now + HOLD_MS + 1));
    assert!(PrepView::of(&app.state).is_some());
    assert!(!app.animating(), "at rest");
    assert!(app.state.due().is_none());
    app.shutdown();
    let _ = std::fs::remove_dir_all(&tree.root);
}
