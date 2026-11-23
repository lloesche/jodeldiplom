//! The UI core without IO: browsing, playing, the filter and the search,
//! queueing, lazy details, engine events, settings, the keymap.

mod common;

use std::path::PathBuf;

use jodel_playback::{Command, Event, PlayState};
use jodel_rhythm::Method;
use jodeldiplom::browser::{Details, Entry, EntryKind};
use jodeldiplom::config::Config;
use jodeldiplom::core::keymap::{self, Context, Key, KeyCode, Meaning, Op, KEYMAP};
use jodeldiplom::core::layout::{self, Layout};
use jodeldiplom::core::text::{self, width, Seg, Tone};
use jodeldiplom::core::{
    browser_rows, update, Action, DeviceChoice, Effect, Gpu, Placing, Rhythm, RhythmRecipe, Row,
    Screen, SettingsRow, Side, State, WalkPurpose,
};

fn entry(path: &str, kind: EntryKind) -> Entry {
    let path = PathBuf::from(path);
    Entry {
        name: if kind == EntryKind::Parent {
            "..".into()
        } else {
            path.file_name().unwrap().to_string_lossy().into_owned()
        },
        path,
        kind,
        relative: None,
    }
}

/// `/m/Album` listed: a parent row, a folder, three files.
fn listed() -> State {
    let mut state = State::new("/m/Album".into(), Config::default(), (100, 30));
    let fx = update(
        &mut state,
        Action::Listed {
            dir: "/m/Album".into(),
            result: Ok(vec![
                entry("/m", EntryKind::Parent),
                entry("/m/Album/Bonus", EntryKind::Dir),
                entry("/m/Album/01 a.flac", EntryKind::File),
                entry("/m/Album/02 b.flac", EntryKind::File),
                entry("/m/Album/03 c.wav", EntryKind::File),
            ]),
        },
    );
    // The files on screen are wanted at once.
    assert_eq!(
        fx,
        vec![Effect::WantDetails(vec![
            "/m/Album/01 a.flac".into(),
            "/m/Album/02 b.flac".into(),
            "/m/Album/03 c.wav".into()
        ])]
    );
    state
}

fn op(state: &mut State, op: Op) -> Vec<Effect> {
    update(state, Action::Op(op))
}

fn engine(fx: &[Effect]) -> Vec<Command> {
    fx.iter()
        .filter_map(|e| match e {
            Effect::Engine(c) => Some(c.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn playing_a_file_queues_its_whole_folder_and_starts_there() {
    let mut state = listed();
    update(&mut state, Action::SelectRow(3));
    // Enter only queues it; shift+enter queues and plays it, the queue
    // kept; alt+enter puts its folder in place of the queue.
    assert_eq!(
        engine(&op(&mut state, Op::Open)),
        vec![Command::Enqueue(vec!["/m/Album/02 b.flac".into()])]
    );
    assert_eq!(
        engine(&op(&mut state, Op::QueuePlay)),
        vec![
            Command::Enqueue(vec!["/m/Album/02 b.flac".into()]),
            Command::Play(0)
        ]
    );
    let fx = op(&mut state, Op::OpenReplace);
    assert_eq!(
        engine(&fx),
        vec![
            Command::Load(vec![
                "/m/Album/01 a.flac".into(),
                "/m/Album/02 b.flac".into(),
                "/m/Album/03 c.wav".into()
            ]),
            Command::Play(1)
        ]
    );
    // With a filter that hides 03, the whole folder is still queued.
    let mut state = listed();
    op(&mut state, Op::Filter);
    for c in "a.flac".chars() {
        update(&mut state, Action::Type(c));
    }
    assert_eq!(browser_rows(&state), vec![Row::Entry(2), Row::SearchBelow]);
    let fx = op(&mut state, Op::OpenReplace);
    assert!(matches!(&engine(&fx)[0], Command::Load(p) if p.len() == 3));
}

#[test]
fn folders_open_and_going_up_reselects_where_we_came_from() {
    let mut state = listed();
    update(&mut state, Action::SelectRow(1));
    let fx = op(&mut state, Op::Open);
    assert!(fx.contains(&Effect::ListDir("/m/Album/Bonus".into())));
    assert_eq!(state.browser.dir, PathBuf::from("/m/Album/Bonus"));
    // While it lists, only `..` is there, so we can always go back.
    assert!(state.browser.loading);
    assert_eq!(state.browser.entries.len(), 1);
    assert_eq!(state.browser.entries[0].kind, EntryKind::Parent);
    assert_eq!(state.browser.entries[0].path, PathBuf::from("/m/Album"));
    update(
        &mut state,
        Action::Listed {
            dir: "/m/Album/Bonus".into(),
            result: Ok(vec![entry("/m/Album", EntryKind::Parent)]),
        },
    );
    // A listing of a folder we already left is ignored.
    update(
        &mut state,
        Action::Listed {
            dir: "/m/Album".into(),
            result: Ok(vec![]),
        },
    );
    assert_eq!(state.browser.entries.len(), 1);
    let fx = op(&mut state, Op::Parent);
    assert!(fx.contains(&Effect::ListDir("/m/Album".into())));
    update(
        &mut state,
        Action::Listed {
            dir: "/m/Album".into(),
            result: Ok(vec![
                entry("/m", EntryKind::Parent),
                entry("/m/Album/Bonus", EntryKind::Dir),
                entry("/m/Album/01 a.flac", EntryKind::File),
            ]),
        },
    );
    assert_eq!(state.browser.cursor, 1, "Bonus is selected again");
    // A listing error shows instead of the rows.
    update(&mut state, Action::SelectRow(1));
    op(&mut state, Op::Open);
    update(
        &mut state,
        Action::Listed {
            dir: "/m/Album/Bonus".into(),
            result: Err("denied".into()),
        },
    );
    assert_eq!(state.browser.error.as_deref(), Some("denied"));
    // A folder that can't be listed still has its `..` row.
    assert_eq!(state.browser.entries.len(), 1);
    assert_eq!(state.browser.entries[0].kind, EntryKind::Parent);
    op(&mut state, Op::Open);
    assert_eq!(state.browser.dir, PathBuf::from("/m/Album"));
}

#[test]
fn esc_with_nothing_to_clear_goes_up_and_reselects_the_folder() {
    let mut state = listed();
    update(&mut state, Action::SelectRow(1));
    op(&mut state, Op::Open);
    update(
        &mut state,
        Action::Listed {
            dir: "/m/Album/Bonus".into(),
            result: Ok(vec![entry("/m/Album", EntryKind::Parent)]),
        },
    );
    let fx = op(&mut state, Op::Back);
    assert!(fx.contains(&Effect::ListDir("/m/Album".into())));
    update(
        &mut state,
        Action::Listed {
            dir: "/m/Album".into(),
            result: Ok(vec![
                entry("/m", EntryKind::Parent),
                entry("/m/Album/Bonus", EntryKind::Dir),
                entry("/m/Album/01 a.flac", EntryKind::File),
            ]),
        },
    );
    assert_eq!(state.browser.cursor, 1, "Bonus is selected again");
}

#[test]
fn opening_the_parent_row_reselects_the_folder_we_came_from() {
    let mut state = listed();
    // Into Bonus, then out again through its `..` row.
    update(&mut state, Action::SelectRow(1));
    op(&mut state, Op::Open);
    update(
        &mut state,
        Action::Listed {
            dir: "/m/Album/Bonus".into(),
            result: Ok(vec![entry("/m/Album", EntryKind::Parent)]),
        },
    );
    update(&mut state, Action::SelectRow(0));
    let fx = op(&mut state, Op::Open);
    assert!(fx.contains(&Effect::ListDir("/m/Album".into())));
    update(
        &mut state,
        Action::Listed {
            dir: "/m/Album".into(),
            result: Ok(vec![
                entry("/m", EntryKind::Parent),
                entry("/m/Album/Bonus", EntryKind::Dir),
                entry("/m/Album/01 a.flac", EntryKind::File),
            ]),
        },
    );
    assert_eq!(state.browser.cursor, 1, "Bonus is selected again");
}

#[test]
fn enter_after_slash_jumps_to_a_path_that_exists() {
    let mut state = listed();
    op(&mut state, Op::Filter);
    for c in "D:".chars() {
        update(&mut state, Action::Type(c));
    }
    let fx = op(&mut state, Op::Accept);
    // The app checks the path; the text stays a filter until it answers.
    assert!(fx.contains(&Effect::Jump {
        text: "D:".into(),
        from: "/m/Album".into()
    }));
    assert!(state.browser.filter.is_some());
    let fx = update(
        &mut state,
        Action::JumpTo {
            dir: "/x/Other".into(),
            select: Some("/x/Other/05 e.flac".into()),
        },
    );
    assert!(fx.contains(&Effect::ListDir("/x/Other".into())));
    assert!(state.browser.filter.is_none());
    update(
        &mut state,
        Action::Listed {
            dir: "/x/Other".into(),
            result: Ok(vec![
                entry("/x", EntryKind::Parent),
                entry("/x/Other/04 d.flac", EntryKind::File),
                entry("/x/Other/05 e.flac", EntryKind::File),
            ]),
        },
    );
    assert_eq!(state.browser.cursor, 2, "the typed file is selected");
}

#[test]
fn enter_searches_below_when_the_text_is_no_path() {
    let mut state = listed();
    op(&mut state, Op::Filter);
    for c in "deep".chars() {
        update(&mut state, Action::Type(c));
    }
    let fx = op(&mut state, Op::Accept);
    assert!(fx.contains(&Effect::Jump {
        text: "deep".into(),
        from: "/m/Album".into()
    }));
    // No such path: the search below starts.
    let fx = update(
        &mut state,
        Action::JumpMissed {
            text: "deep".into(),
        },
    );
    assert!(fx
        .iter()
        .any(|e| matches!(e, Effect::Search { query, .. } if query == "deep")));
    assert!(state.browser.search.is_some());
    // An answer for text no longer typed does nothing.
    let fx = update(&mut state, Action::JumpMissed { text: "old".into() });
    assert!(!fx.iter().any(|e| matches!(e, Effect::Search { .. })));
    // Tab switches to the queue and ends typing; the filter stays.
    op(&mut state, Op::Filter);
    op(&mut state, Op::ToggleSide);
    assert_eq!(state.side, Side::Queue);
    assert!(state.browser.filter.as_ref().is_some_and(|f| !f.editing));
}

#[test]
fn the_filter_keeps_the_selected_row() {
    // Rows: .., Bonus, 01 a.flac, 02 b.flac, 03 c.wav
    let mut state = listed();
    update(&mut state, Action::SelectRow(3));
    op(&mut state, Op::Filter);
    assert_eq!(state.browser.cursor, 3, "/ alone leaves the cursor");
    update(&mut state, Action::Type('b'));
    // Bonus and 02 b.flac match; 02 b.flac stays selected.
    let rows = browser_rows(&state);
    assert_eq!(rows[state.browser.cursor], Row::Entry(3));
    // A filter that hides it moves to the first match.
    update(&mut state, Action::Type('o'));
    assert_eq!(state.browser.cursor, 0);
    // Clearing the filter keeps what is selected then.
    op(&mut state, Op::Back);
    assert_eq!(browser_rows(&state)[state.browser.cursor], Row::Entry(1));
}

#[test]
fn the_filter_narrows_and_the_search_goes_below() {
    let mut state = listed();
    op(&mut state, Op::Filter);
    for c in "B".chars() {
        update(&mut state, Action::Type(c));
    }
    // Names match case-insensitively: Bonus and 02 b.flac.
    assert_eq!(
        browser_rows(&state),
        vec![Row::Entry(1), Row::Entry(3), Row::SearchBelow]
    );
    // Titles read in the background match too.
    state.details.insert(
        "/m/Album/03 c.wav".into(),
        Details {
            title: Some("Bossa".into()),
            ..Details::default()
        },
    );
    assert_eq!(browser_rows(&state).len(), 4);
    let fx = op(&mut state, Op::SearchBelow);
    let id = match &fx[..] {
        [Effect::Search { id, root, query }, ..] => {
            assert_eq!((root.to_str().unwrap(), query.as_str()), ("/m/Album", "B"));
            *id
        }
        other => panic!("{other:?}"),
    };
    assert!(browser_rows(&state).is_empty());
    let mut hit = entry("/m/Album/Deep/Blue.flac", EntryKind::File);
    hit.relative = Some("Deep/Blue.flac".into());
    update(
        &mut state,
        Action::SearchHits {
            id,
            hits: vec![hit],
        },
    );
    // Hits of an older search are dropped.
    update(
        &mut state,
        Action::SearchHits {
            id: id + 7,
            hits: vec![entry("/x.flac", EntryKind::File)],
        },
    );
    update(&mut state, Action::SearchDone { id });
    assert_eq!(browser_rows(&state), vec![Row::Hit(0)]);
    assert!(state.browser.search.as_ref().unwrap().done);
    let fx = op(&mut state, Op::OpenReplace);
    assert_eq!(
        engine(&fx),
        vec![
            Command::Load(vec!["/m/Album/Deep/Blue.flac".into()]),
            Command::Play(0)
        ]
    );
    // esc leaves the results, then clears the filter.
    let fx = op(&mut state, Op::Back);
    assert_eq!(fx.first(), Some(&Effect::CancelSearch));
    assert!(state.browser.filter.is_some());
    op(&mut state, Op::Back);
    assert!(state.browser.filter.is_none());
    assert_eq!(browser_rows(&state).len(), 5);
    // Searching with no filter text says what to do instead.
    let fx = op(&mut state, Op::SearchBelow);
    assert!(fx.is_empty() && state.message.is_some());
}

#[test]
fn queueing_a_file_or_a_whole_folder() {
    let mut state = listed();
    update(&mut state, Action::SelectRow(2));
    let fx = op(&mut state, Op::Enqueue);
    assert_eq!(
        engine(&fx),
        vec![Command::Enqueue(vec!["/m/Album/01 a.flac".into()])]
    );
    let fx = update(&mut state, Action::EnqueueRow(1));
    assert_eq!(
        fx,
        vec![Effect::Walk {
            root: "/m/Album/Bonus".into(),
            purpose: WalkPurpose::Enqueue {
                name: "Bonus".into(),
                how: Placing::Append,
            }
        }]
    );
    let fx = update(
        &mut state,
        Action::Walked {
            purpose: WalkPurpose::Enqueue {
                name: "Bonus".into(),
                how: Placing::Append,
            },
            paths: vec![
                "/m/Album/Bonus/x.flac".into(),
                "/m/Album/Bonus/y.flac".into(),
            ],
        },
    );
    assert!(matches!(&engine(&fx)[..], [Command::Enqueue(p)] if p.len() == 2));
    assert_eq!(
        state.message.as_ref().unwrap().text,
        "Queued Bonus (2 tracks)"
    );
    // The queue side shows what the engine reports.
    update(
        &mut state,
        Action::Engine(Event::QueueChanged {
            queue: vec!["a".into(), "b".into(), "c".into()],
            current: Some(1),
        }),
    );
    op(&mut state, Op::ToggleSide);
    assert_eq!(state.side, Side::Queue);
    update(&mut state, Action::SelectRow(2));
    assert_eq!(engine(&op(&mut state, Op::Open)), vec![Command::Play(2)]);
}

#[test]
fn only_files_on_screen_are_read() {
    let mut state = State::new("/big".into(), Config::default(), (100, 16));
    let entries: Vec<Entry> = (0..100)
        .map(|i| entry(&format!("/big/{i:03}.flac"), EntryKind::File))
        .collect();
    let fx = update(
        &mut state,
        Action::Listed {
            dir: "/big".into(),
            result: Ok(entries),
        },
    );
    let Effect::WantDetails(first) = &fx[0] else {
        panic!("{fx:?}")
    };
    let height = first.len();
    assert!((5..20).contains(&height), "{height}");
    assert_eq!(first[0], PathBuf::from("/big/000.flac"));
    // Arrived details are not asked for again; moving within the screen
    // asks for nothing new.
    update(
        &mut state,
        Action::Details(first[0].clone(), Details::default()),
    );
    assert!(op(&mut state, Op::Down).is_empty());
    // Paging down wants the next screen, which replaces the old wish.
    let fx = op(&mut state, Op::PageDown);
    let Effect::WantDetails(next) = &fx[0] else {
        panic!("{fx:?}")
    };
    assert!(!next.contains(&first[0]));
    let cursor = state.browser.cursor;
    assert!(cursor >= height);
    assert!(next.contains(&PathBuf::from(format!("/big/{cursor:03}.flac"))));
    // What the worker dropped can be asked for again later.
    update(&mut state, Action::DetailsDropped(first[1..].to_vec()));
    assert!(first[1..].iter().all(|p| !state.pending.contains(p)));
    let fx = op(&mut state, Op::Top);
    assert!(matches!(&fx[0], Effect::WantDetails(again) if again.contains(&first[1])));
}

#[test]
fn engine_events_drive_the_now_playing_state() {
    let mut state = listed();
    let track = common::now_track(44_100, state.config.settings(), 0);
    update(
        &mut state,
        Action::Engine(Event::TrackStarted(Box::new(jodel_playback::TrackStart {
            index: 0,
            head: 0,
            info: track.info.clone(),
            route: track.route.clone(),
            output: track.output,
            device: "DAC".into(),
            bit_perfect: true,
            output_frame: 0,
            from_frame: 0,
        }))),
    );
    update(&mut state, Action::Engine(Event::State(PlayState::Playing)));
    let text = jodeldiplom::core::text::signal_path(&state)
        .unwrap()
        .iter()
        .map(|s| s.text.clone())
        .collect::<String>();
    assert_eq!(text, "file 44.1 → DAC (exclusive), bit-perfect");
    assert_eq!(jodeldiplom::core::text::status_light(&state).1, "PCM");
    // Seeking moves the position at once and asks the engine.
    state.playing.position = 20.0;
    assert_eq!(
        engine(&op(&mut state, Op::SeekBack)),
        vec![Command::Seek(15.0)]
    );
    assert_eq!(engine(&op(&mut state, Op::PlayPause)), vec![Command::Pause]);
    update(&mut state, Action::Engine(Event::State(PlayState::Paused)));
    assert_eq!(
        engine(&op(&mut state, Op::PlayPause)),
        vec![Command::Resume]
    );
    // A click on the bar seeks to that fraction of the track.
    let duration = state.playing.duration().unwrap();
    assert_eq!(
        engine(&update(&mut state, Action::SeekTo(0.5))),
        vec![Command::Seek(duration / 2.0)]
    );
    // A click on the Transition tab's overview of the playing track seeks there
    // too, and its view follows the playhead again; on the next track's,
    // only the view moves.
    use jodeldiplom::core::lanes::Block;
    assert_eq!(
        engine(&update(
            &mut state,
            Action::TransitionJump {
                block: Block::Now,
                time: 12.5
            }
        )),
        vec![Command::Seek(12.5)]
    );
    assert_eq!(state.playing.position, 12.5);
    assert!(
        jodeldiplom::core::transition::tab::block(&state, Block::Now)
            .start
            .is_none()
    );
    assert!(engine(&update(
        &mut state,
        Action::TransitionJump {
            block: Block::Next,
            time: 30.0
        }
    ))
    .is_empty());
    update(
        &mut state,
        Action::Engine(Event::Error {
            index: Some(0),
            error: jodel_playback::Error::NoRoute(jodel_playback::NoRoute {
                reason: "gone".into(),
            }),
            skipped: true,
        }),
    );
    assert!(state.message.as_ref().unwrap().text.contains("(skipped)"));
    update(&mut state, Action::Engine(Event::State(PlayState::Stopped)));
    assert!(state.playing.track.is_none());
    // Stopped with nothing queued, space plays the selected file.
    update(&mut state, Action::SelectRow(4));
    assert!(matches!(
        &engine(&op(&mut state, Op::PlayPause))[..],
        [Command::Load(_), Command::Play(_)]
    ));
}

/// A lossy file is never bit-perfect: the path says what it was decoded
/// from and the status light names the codec and its bit rate.
#[test]
fn a_lossy_track_says_what_it_was_decoded_from() {
    use jodeldiplom::core::text::{signal_path, status_light, Tone};
    let mut state = listed();
    let track = common::now_track(44_100, state.config.settings(), 0);
    let mut info = (*track.info).clone();
    info.container = jodel_playback::Container::Mpeg;
    info.codec = jodel_playback::Codec::Mp3;
    info.bits = 0;
    info.float = true;
    info.bitrate = Some(320_000);
    let facts = jodel_playback::FileFacts::from(&info);
    assert_eq!(facts.decoded_from, Some("MP3"));
    let route = jodel_playback::choose_route(
        &facts,
        &jodel_playback::any_capabilities(),
        &state.config.settings(),
    )
    .unwrap();
    assert!(!route.lossless);
    update(
        &mut state,
        Action::Engine(Event::TrackStarted(Box::new(jodel_playback::TrackStart {
            index: 0,
            head: 0,
            info: std::sync::Arc::new(info),
            route,
            output: track.output,
            device: "DAC".into(),
            bit_perfect: false,
            output_frame: 0,
            from_frame: 0,
        }))),
    );
    let segs = signal_path(&state).unwrap();
    let text: String = segs.iter().map(|s| s.text.as_str()).collect();
    assert_eq!(text, "file 44.1 → DAC (exclusive), decoded from MP3");
    assert!(!text.contains("bit-perfect"));
    let tail = segs.iter().find(|s| s.text.contains("decoded")).unwrap();
    assert_eq!(tail.tone, Tone::Dim);
    let (light, label, tone) = status_light(&state);
    assert_eq!(
        (light, label.as_str(), tone),
        ("○", "MP3 320 kbps · lossy", Tone::Dim)
    );
}

/// A device that takes only 48, 96 and 192 kHz plays a 44.1 kHz file
/// resampled: the path shows the step and says so in the warning colour,
/// with the clip count when there is one.
#[test]
fn a_resampled_route_shows_its_step_and_its_clips() {
    use jodel_output::{Capabilities, SampleFormat, StreamFormat};
    use jodeldiplom::core::text::{signal_path, Tone};
    let mut state = listed();
    let mut track = common::now_track(44_100, state.config.settings(), 0);
    let mut gc7 = Capabilities::new();
    for rate in [48_000, 96_000, 192_000] {
        for f in [SampleFormat::S16, SampleFormat::S24Packed] {
            gc7.insert(StreamFormat::new(rate, 2, f));
        }
    }
    let facts = jodel_playback::FileFacts::from(&*track.info);
    track.route = jodel_playback::choose_route(&facts, &gc7, &state.config.settings()).unwrap();
    track.output.format = track.route.format;
    update(
        &mut state,
        Action::Engine(Event::TrackStarted(Box::new(jodel_playback::TrackStart {
            index: 0,
            head: 0,
            info: track.info.clone(),
            route: track.route.clone(),
            output: track.output,
            device: "DAC".into(),
            bit_perfect: false,
            output_frame: 0,
            from_frame: 0,
        }))),
    );
    let text = |state: &State| -> String {
        signal_path(state)
            .unwrap()
            .iter()
            .map(|s| s.text.clone())
            .collect()
    };
    assert_eq!(
        text(&state),
        "file 44.1 → resample 48 (160/147) → DAC (exclusive), not bit-perfect"
    );
    let segs = signal_path(&state).unwrap();
    let tone = |needle: &str| segs.iter().find(|s| s.text.contains(needle)).unwrap().tone;
    assert_eq!(tone("resample"), Tone::Warn);
    assert_eq!(tone("not bit-perfect"), Tone::Warn);
    // With headroom the step says so.
    let mut with = state.clone();
    if let Some(t) = with.playing.track.as_mut() {
        let step = t.route.resample.unwrap();
        t.route.resample = Some(step.with_headroom(jodel_playback::Headroom(10)));
    }
    assert_eq!(
        text(&with),
        "file 44.1 → resample 48 (160/147, -1 dB) → DAC (exclusive), not bit-perfect"
    );
    update(
        &mut state,
        Action::Engine(Event::Clipped {
            index: 0,
            samples: 12,
            output_frame: 4_800,
        }),
    );
    assert!(text(&state).ends_with("not bit-perfect, 12 samples clipped"));
    // A seek starts the count again.
    update(
        &mut state,
        Action::Engine(Event::Seeked {
            index: 0,
            seconds: 1.0,
        }),
    );
    assert!(text(&state).ends_with("not bit-perfect"));
}

#[test]
fn a_bit_perfect_route_is_not_while_the_mixer_sums() {
    let mut state = common::playing_state(120, 40);
    let text = |state: &State| -> String {
        text::signal_path(state)
            .unwrap()
            .iter()
            .map(|s| s.text.clone())
            .collect()
    };
    state.playing.track.as_mut().unwrap().bit_perfect = true;
    assert!(text(&state).ends_with(", bit-perfect"));
    state.playing.mixed = true;
    assert!(text(&state).ends_with("not bit-perfect"));
}

#[test]
fn settings_change_the_engine_and_the_file() {
    let mut state = listed();
    let fx = op(&mut state, Op::Settings);
    assert_eq!(state.screen, Screen::Settings);
    assert_eq!(fx, vec![Effect::ListDevices]);
    op(&mut state, Op::Down);
    let fx = op(&mut state, Op::ValueDown);
    assert_eq!(state.config.max_rate, Some(44_100));
    // Saved at once; the engine gets it after the pause (see
    // `settings_reach_the_engine_after_a_pause`).
    assert_eq!(fx, vec![Effect::SaveConfig]);
    // Resample when needed: on by default, toggles, reaches the engine.
    assert!(state.config.resample);
    op(&mut state, Op::Down);
    let fx = op(&mut state, Op::ValueUp);
    assert!(!state.config.resample);
    assert!(!state.config.settings().resample);
    assert_eq!(fx, vec![Effect::SaveConfig]);
    let due = state.settings_due.unwrap();
    assert_eq!(
        engine(&update(&mut state, Action::Tick(due))),
        vec![Command::SetSettings(state.engine_settings())]
    );
    op(&mut state, Op::Open);
    assert!(state.config.resample);
    // The device list.
    update(
        &mut state,
        Action::Devices(Ok(vec![DeviceChoice {
            id: Some("dac".into()),
            name: "DAC".into(),
            summary: "44.1 kHz".into(),
        }])),
    );
    update(&mut state, Action::SettingsClick(0, 0));
    assert!(state.settings.picking);
    op(&mut state, Op::Down);
    let fx = op(&mut state, Op::Open);
    assert_eq!(state.config.device.as_deref(), Some("dac"));
    assert_eq!(
        fx,
        vec![Effect::SaveConfig, Effect::SwitchDevice(Some("dac".into()))]
    );
    // With --null the choice is saved but the output stays.
    state.null_output = true;
    update(&mut state, Action::SettingsClick(0, 0));
    op(&mut state, Op::Up);
    let fx = op(&mut state, Op::Open);
    assert_eq!(fx, vec![Effect::SaveConfig]);
    assert_eq!(state.config.device, None);
    op(&mut state, Op::Back);
    assert_eq!(state.screen, Screen::Main);
}

fn key(code: KeyCode) -> Key {
    Key::plain(code)
}

#[test]
fn typing_matches_the_titles_shown_and_works_in_the_queue() {
    // Rows: .., Bonus, 01 a.flac, 02 b.flac, 03 c.wav; 02 has a title.
    let mut state = listed();
    update(
        &mut state,
        Action::Details(
            "/m/Album/02 b.flac".into(),
            Details {
                title: Some("Zebra".into()),
                ..Details::default()
            },
        ),
    );
    update(&mut state, Action::Tick(5_000));
    update(&mut state, Action::Type('z'));
    assert_eq!(state.browser.cursor, 3, "the title, not the file name");
    // The queue: titles where read, file names otherwise.
    update(
        &mut state,
        Action::Engine(Event::QueueChanged {
            queue: vec![
                "/m/Album/01 a.flac".into(),
                "/m/Album/02 b.flac".into(),
                "/m/Album/03 c.wav".into(),
            ],
            current: None,
        }),
    );
    op(&mut state, Op::ToggleSide);
    update(&mut state, Action::Tick(10_000));
    update(&mut state, Action::Type('z'));
    assert_eq!(state.queue.cursor, 1);
    update(&mut state, Action::Tick(15_000));
    update(&mut state, Action::Type('0'));
    update(&mut state, Action::Type('3'));
    assert_eq!(state.queue.cursor, 2);
}

#[test]
fn typing_in_the_browser_jumps_to_names() {
    // Rows: .., Bonus, 01 a.flac, 02 b.flac, 03 c.wav
    let mut state = listed();
    let mut tick = 0;
    let mut type_at = |state: &mut State, ms: u64, text: &str| {
        tick += ms;
        update(state, Action::Tick(tick));
        for c in text.chars() {
            update(state, Action::Type(c));
        }
    };
    type_at(&mut state, 10, "b");
    assert_eq!(state.browser.cursor, 1, "Bonus");
    type_at(&mut state, 2000, "02");
    assert_eq!(state.browser.cursor, 3, "02 b.flac");
    // After a pause a new prefix starts; `..` is never a match.
    type_at(&mut state, 2000, "0");
    assert_eq!(state.browser.cursor, 4, "the next name starting with 0");
    type_at(&mut state, 100, "0");
    assert_eq!(state.browser.cursor, 2, "0 again wraps around");
    type_at(&mut state, 2000, "zz");
    assert_eq!(state.browser.cursor, 2, "no match: the cursor stays");
}

#[test]
fn any_other_key_or_a_new_folder_starts_the_typing_afresh() {
    // Rows: .., Bonus, 01 a.flac, 02 b.flac, 03 c.wav
    let mut state = listed();
    update(&mut state, Action::Tick(10));
    update(&mut state, Action::Type('b'));
    assert_eq!(state.browser.cursor, 1, "Bonus");
    // A key in between, at once: "0" is a new prefix, not "b0" (which
    // matches nothing): from 01 a.flac it moves on to the next 0.
    op(&mut state, Op::Down);
    assert_eq!(state.browser.cursor, 2, "01 a.flac");
    update(&mut state, Action::Type('0'));
    assert_eq!(state.browser.cursor, 3, "02 b.flac");
    // Opening a folder, at once: what was typed in the last one is gone.
    update(&mut state, Action::Tick(3000));
    update(&mut state, Action::Type('b'));
    assert_eq!(state.browser.cursor, 1, "Bonus");
    op(&mut state, Op::Open);
    assert!(state.browser.dir.ends_with("Bonus"));
    assert_eq!(state.browser.typed, "");
}

#[test]
fn space_is_part_of_a_name_being_typed_and_play_pause_otherwise() {
    // Rows: .., Bonus, 01 a.flac, 02 b.flac, 03 c.wav
    let mut state = listed();
    let space = Key::ch(' ');
    update(&mut state, Action::Tick(10));
    assert_eq!(
        keymap::lookup(&state, space),
        Some(Meaning::Op(Op::PlayPause)),
        "nothing typed"
    );
    for c in "02".chars() {
        update(&mut state, Action::Type(c));
    }
    update(&mut state, Action::Tick(500));
    assert_eq!(keymap::lookup(&state, space), Some(Meaning::Type(' ')));
    update(&mut state, Action::Type(' '));
    update(&mut state, Action::Type('b'));
    assert_eq!(state.browser.cursor, 3, "02 b.flac");
    assert_eq!(state.browser.typed, "02 b");
    // After the pause, space plays and pauses again.
    update(&mut state, Action::Tick(2000));
    assert_eq!(
        keymap::lookup(&state, space),
        Some(Meaning::Op(Op::PlayPause))
    );
}

#[test]
fn keys_mean_different_things_in_each_context() {
    let mut state = listed();
    let look = |s: &State, k: Key| keymap::lookup(s, k);
    // In the browser, lowercase letters jump to names; capitals are commands.
    assert_eq!(look(&state, Key::ch('j')), Some(Meaning::Type('j')));
    assert_eq!(look(&state, Key::ch('0')), Some(Meaning::Type('0')));
    assert_eq!(look(&state, Key::ch('Q')), Some(Meaning::Op(Op::Quit)));
    assert_eq!(look(&state, Key::ch('N')), Some(Meaning::Op(Op::Next)));
    // A no longer queues: enter does (shift+enter plays it too).
    assert_eq!(look(&state, Key::ch('A')), None);
    assert_eq!(
        look(&state, Key::shift(KeyCode::Enter)),
        Some(Meaning::Op(Op::QueuePlay))
    );
    assert_eq!(
        look(
            &state,
            Key {
                code: KeyCode::Enter,
                ctrl: false,
                alt: true,
                shift: false
            }
        ),
        Some(Meaning::Op(Op::OpenReplace))
    );
    assert_eq!(look(&state, Key::ch(' ')), Some(Meaning::Op(Op::PlayPause)));
    assert_eq!(look(&state, Key::ch('/')), Some(Meaning::Op(Op::Filter)));
    assert_eq!(
        look(&state, key(KeyCode::Left)),
        Some(Meaning::Op(Op::SeekBack))
    );
    // The queue types too; commands stay on capitals and other keys.
    op(&mut state, Op::ToggleSide);
    assert_eq!(look(&state, Key::ch('j')), Some(Meaning::Type('j')));
    assert_eq!(look(&state, Key::ch('n')), Some(Meaning::Type('n')));
    assert_eq!(look(&state, Key::ch('N')), Some(Meaning::Op(Op::Next)));
    assert_eq!(
        look(&state, key(KeyCode::Down)),
        Some(Meaning::Op(Op::Down))
    );
    // Arrows seek; with shift they skip tracks.
    assert_eq!(
        look(&state, Key::shift(KeyCode::Right)),
        Some(Meaning::Op(Op::Next))
    );
    assert_eq!(
        look(&state, Key::shift(KeyCode::Left)),
        Some(Meaning::Op(Op::Previous))
    );
    assert_eq!(
        look(&state, key(KeyCode::Right)),
        Some(Meaning::Op(Op::SeekForward))
    );
    op(&mut state, Op::ToggleSide);
    // Typing a filter: letters are text, ctrl+c still quits.
    op(&mut state, Op::Filter);
    assert_eq!(look(&state, Key::ch('j')), Some(Meaning::Type('j')));
    assert_eq!(look(&state, Key::ch('q')), Some(Meaning::Type('q')));
    assert_eq!(look(&state, Key::ctrl('c')), Some(Meaning::Op(Op::Quit)));
    assert_eq!(
        look(&state, key(KeyCode::Tab)),
        Some(Meaning::Op(Op::ToggleSide))
    );
    op(&mut state, Op::Accept);
    // Settings: ←→ change values instead of seeking.
    op(&mut state, Op::Settings);
    assert_eq!(
        look(&state, key(KeyCode::Right)),
        Some(Meaning::Op(Op::ValueUp))
    );
    assert_eq!(look(&state, Key::ch(' ')), Some(Meaning::Op(Op::PlayPause)));
    // The palette runs commands by name.
    op(&mut state, Op::Back);
    op(&mut state, Op::Palette);
    for c in "help".chars() {
        update(&mut state, Action::Type(c));
    }
    op(&mut state, Op::Accept);
    assert_eq!(state.screen, Screen::Help);
    assert!(state.palette.is_none());
}

#[test]
fn help_and_palette_come_from_the_keymap() {
    let help = keymap::help_sections();
    for binding in KEYMAP {
        if let Op::JumpBar(_) | Op::JumpBarAtEnd(_) = binding.op {
            // Eighteen keys in two lines.
            let section = help.iter().find(|s| s.title == "Transition tab").unwrap();
            let line = if matches!(binding.op, Op::JumpBarAtEnd(_)) {
                "1 … 9"
            } else {
                "shift+1 … 9"
            };
            assert!(section.lines.iter().any(|(keys, _)| keys == line), "{line}");
            continue;
        }
        let section = help
            .iter()
            .find(|s| s.title == binding.context.title())
            .unwrap();
        assert!(
            section
                .lines
                .iter()
                .any(|(keys, text)| *keys == keymap::keys_label(binding) && *text == binding.help),
            "{binding:?} missing from the help"
        );
    }
    let palette = keymap::palette_entries();
    for entry in &palette {
        assert!(KEYMAP.iter().any(|b| b.op == entry.op));
        assert_eq!(palette.iter().filter(|e| e.op == entry.op).count(), 1);
    }
    for op in [Op::PlayPause, Op::Quit, Op::Settings, Op::QueuePlay] {
        assert!(palette.iter().any(|e| e.op == op), "{op:?}");
    }
    assert!(palette.iter().all(|e| e.op != Op::Down));
    // The Mixer tab: its keys in the help, its commands in the palette.
    assert!(help.iter().any(|s| s.title == "Mixer tab"));
    for op in [Op::DeskMute, Op::DeskSolo, Op::DeskEcho] {
        assert!(palette.iter().any(|e| e.op == op), "{op:?}");
    }
    assert!(palette.iter().all(|e| e.op != Op::DeskNextStrip));
    // No key means two things in one context.
    for (i, a) in KEYMAP.iter().enumerate() {
        for b in &KEYMAP[i + 1..] {
            if a.context == b.context {
                assert!(
                    a.keys.iter().all(|k| !b.keys.contains(k)),
                    "{a:?} and {b:?} share a key"
                );
            }
        }
    }
    assert_eq!(keymap::palette_matches("next track").len(), 1);
    assert!(keymap::palette_matches("zzz").is_empty());
    let _ = Context::Global;
}

/// A key pressed: what it means now, carried out.
fn press(state: &mut State, key: Key) -> Vec<Effect> {
    match keymap::lookup(state, key) {
        Some(Meaning::Op(o)) => op(state, o),
        Some(Meaning::Type(c)) => update(state, Action::Type(c)),
        None => Vec::new(),
    }
}

fn press_text(state: &mut State, text: &str) -> Vec<Effect> {
    text.chars()
        .flat_map(|c| press(state, Key::ch(c)))
        .collect()
}

/// On the help, `/` searches it: "volume" leaves the device volume's keys
/// under their section's title and drops the rest; the letters typed are
/// only typed; esc clears the search, and a second esc leaves the help.
#[test]
fn the_help_is_searched_after_a_slash() {
    let mut state = listed();
    press(&mut state, Key::ch('?'));
    assert_eq!(state.screen, Screen::Help);
    let whole = jodeldiplom::core::help_line_count(&state);
    press(&mut state, Key::ch('/'));
    assert!(state.help_search.as_ref().is_some_and(|s| s.editing));
    // While typing, keys that are commands elsewhere are letters.
    for c in ['Q', 'N', ' ', '?', ',', ':', '/'] {
        assert_eq!(
            keymap::lookup(&state, Key::ch(c)),
            Some(Meaning::Type(c)),
            "{c:?}"
        );
    }
    // Case is ignored; typing runs nothing (no volume step, no seek).
    let fx = press_text(&mut state, "VOLume");
    assert!(fx.is_empty(), "{fx:?}");
    assert_eq!(state.help_search.as_ref().unwrap().text, "VOLume");
    assert_eq!(state.screen, Screen::Help);
    assert!(!state.quit);
    let shown = jodeldiplom::core::help_shown(&state);
    let anywhere = shown.iter().find(|s| s.title == "Anywhere").unwrap();
    let keys: Vec<&str> = anywhere.lines.iter().map(|(k, _)| k.as_str()).collect();
    assert!(
        keys.contains(&"alt+↑") && keys.contains(&"alt+↓"),
        "{keys:?}"
    );
    for section in &shown {
        assert!(!section.lines.is_empty(), "{} is empty", section.title);
        for (keys, text) in &section.lines {
            assert!(
                format!("{keys} {text}").to_lowercase().contains("volume"),
                "{keys} {text}"
            );
        }
    }
    let all: Vec<&str> = shown
        .iter()
        .flat_map(|s| s.lines.iter().map(|(_, t)| *t))
        .collect();
    assert!(!all.contains(&"play / pause"));
    assert!(!shown.iter().any(|s| s.title == "Mouse"));
    assert!(jodeldiplom::core::help_line_count(&state) < whole);
    // Backspace edits; enter keeps the search and gives the keys back.
    press(&mut state, key(KeyCode::Backspace));
    assert_eq!(state.help_search.as_ref().unwrap().text, "VOLum");
    press(&mut state, key(KeyCode::Enter));
    let search = state.help_search.as_ref().unwrap();
    assert!(!search.editing && search.text == "VOLum");
    assert_eq!(
        keymap::lookup(&state, key(KeyCode::Down)),
        Some(Meaning::Op(Op::Down))
    );
    // `/` again goes on typing it.
    press(&mut state, Key::ch('/'));
    press(&mut state, Key::ch('e'));
    assert_eq!(state.help_search.as_ref().unwrap().text, "VOLume");
    // Esc clears the search (the help is whole again), a second leaves.
    press(&mut state, key(KeyCode::Esc));
    assert_eq!(state.help_search, None);
    assert_eq!(state.screen, Screen::Help);
    assert_eq!(jodeldiplom::core::help_line_count(&state), whole);
    press(&mut state, Key::ch('/'));
    press_text(&mut state, "volume");
    press(&mut state, key(KeyCode::Enter));
    press(&mut state, key(KeyCode::Esc));
    assert_eq!(state.help_search, None);
    assert_eq!(state.screen, Screen::Help);
    press(&mut state, key(KeyCode::Esc));
    assert_eq!(state.screen, Screen::Main);
    // The browser's `/` is untouched by it.
    assert!(state.browser.filter.is_none());
    press(&mut state, Key::ch('/'));
    assert!(state.browser.filter.as_ref().is_some_and(|f| f.editing));
}

/// The help searched still scrolls and pages, within what the search
/// leaves; typing starts again at the top. The help opens whole.
#[test]
fn the_help_searched_scrolls_within_the_matches() {
    let mut state = listed();
    press(&mut state, Key::ch('?'));
    press(&mut state, Key::ch('/'));
    press(&mut state, Key::ch('d'));
    let Layout::Help(h) = layout::compute(&state) else {
        panic!("not the help")
    };
    let visible = usize::from(h.text.height);
    let lines = jodeldiplom::core::help_line_count(&state);
    assert!(h.search.is_some());
    assert!(lines > visible, "{lines} lines, {visible} shown");
    press(&mut state, key(KeyCode::PageDown));
    assert!(state.help_scroll > 0);
    for _ in 0..100 {
        press(&mut state, key(KeyCode::PageDown));
    }
    assert_eq!(state.help_scroll, lines - visible);
    press(&mut state, key(KeyCode::Up));
    assert_eq!(state.help_scroll, lines - visible - 1);
    // Typing narrows it from the top; what fits does not scroll.
    press_text(&mut state, "evice vol");
    assert_eq!(state.help_scroll, 0);
    press(&mut state, key(KeyCode::Enter));
    press(&mut state, key(KeyCode::PageDown));
    assert_eq!(state.help_scroll, 0);
    // Left with the search kept (for the settings) and opened again, the
    // help is whole.
    press(&mut state, Key::ch(','));
    assert_eq!(state.screen, Screen::Settings);
    press(&mut state, Key::ch('?'));
    assert_eq!(state.screen, Screen::Help);
    assert_eq!(state.help_search, None);
    let Layout::Help(h) = layout::compute(&state) else {
        panic!("not the help")
    };
    assert!(h.search.is_none());
}

fn line_text(line: &[Seg]) -> String {
    line.iter().map(|s| s.text.as_str()).collect()
}

/// Checks wrapped signal path lines: within `max`, a new line only at a
/// step's arrow or the device's tail, the tail in one piece, no lone word.
fn check_path_lines(lines: &[Vec<Seg>], max: usize, device: &str) {
    let texts: Vec<String> = lines.iter().map(|l| line_text(l)).collect();
    // A device name wider than a line has to break between its words.
    let fits = width(device) + 2 <= max;
    for (i, text) in texts.iter().enumerate() {
        assert!(width(text) <= max, "{text:?} wider than {max}");
        assert!(
            text.split_whitespace().count() >= 2,
            "a lone word: {texts:?}"
        );
        if i > 0 && fits {
            assert!(
                text.starts_with("→ ") || text.starts_with('('),
                "a break inside a step: {texts:?}"
            );
        }
    }
    let tail = texts
        .iter()
        .find(|t| t.contains("bit-perfect"))
        .expect("the tail");
    assert!(tail.contains("(exclusive), "), "{texts:?}");
    if fits {
        assert!(
            texts.iter().any(|t| t.contains(device)),
            "the device split: {texts:?}"
        );
    }
}

#[test]
fn the_right_column_fits_the_signal_path() {
    let names = [
        "Speakers (Sound Blaster GC7)",
        "Headset (Sound Blaster GC7)",
        "Digital Output (High Definition Audio 2)",
    ];
    assert_eq!(names[2].len(), 40);
    for name in names {
        for cols in [72u16, 96, 120, 160, 200, 240] {
            let mut state = common::playing_state(cols, 36);
            state.playing.track.as_mut().unwrap().device = name.into();
            let Layout::Main(m) = layout::compute(&state) else {
                panic!("not the main screen at {cols}")
            };
            let right = m.right.unwrap().width;
            let need = text::info_width(&state) as u16 + 1;
            assert!(right >= layout::right_width(cols), "{cols}");
            assert!(right <= (cols / 2).max(layout::right_width(cols)), "{cols}");
            let info = m.info.unwrap();
            let lines = text::info_lines(&state, usize::from(info.width));
            let path = &lines[lines
                .iter()
                .position(|l| line_text(l).starts_with("file"))
                .unwrap()..];
            if need <= cols / 2 {
                // Room for every line: the path is one line.
                assert_eq!(right, need, "{name} at {cols}");
                assert_eq!(path.len(), 1, "{name} at {cols}: {path:?}");
            }
            check_path_lines(path, usize::from(info.width), name);
            // The list keeps the rest.
            assert_eq!(m.list.width + right, cols);
        }
    }
}

#[test]
fn the_signal_path_wraps_between_steps() {
    let seg = |text: &str, tone: Tone| Seg {
        text: text.into(),
        tone,
    };
    let arrow = || seg(" → ", Tone::Dim);
    let device = "Speakers (Sound Blaster GC7)";
    let resampled = vec![
        seg("file 44.1", Tone::Source),
        arrow(),
        seg("resample 48 (160/147)", Tone::Warn),
        arrow(),
        seg(device, Tone::Device),
        seg(" (exclusive)", Tone::Dim),
        seg(", ", Tone::Dim),
        seg("not bit-perfect", Tone::Warn),
    ];
    let whole = line_text(&resampled);
    for max in [30, 36, 45, 56, 60, 70, 80, width(&whole)] {
        let lines = text::wrap_path(&resampled, max);
        check_path_lines(&lines, max, device);
        let joined: String = lines
            .iter()
            .map(|l| line_text(l))
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(joined, whole, "nothing lost at {max}");
    }
    let texts = |max| -> Vec<String> {
        text::wrap_path(&resampled, max)
            .iter()
            .map(|l| line_text(l))
            .collect()
    };
    assert_eq!(texts(width(&whole)), std::slice::from_ref(&whole));
    // The device keeps its tail when both fit on a line.
    assert_eq!(
        texts(60),
        [
            "file 44.1 → resample 48 (160/147)",
            "→ Speakers (Sound Blaster GC7) (exclusive), not bit-perfect",
        ]
    );
    // Else the tail goes on its own line, whole.
    assert_eq!(
        texts(40),
        [
            "file 44.1 → resample 48 (160/147)",
            "→ Speakers (Sound Blaster GC7)",
            "(exclusive), not bit-perfect",
        ]
    );
    // The bit-perfect case at the column's usual width.
    let plain = vec![
        seg("file 48", Tone::Source),
        arrow(),
        seg(device, Tone::Device),
        seg(" (exclusive)", Tone::Dim),
        seg(", ", Tone::Dim),
        seg("bit-perfect", Tone::Good),
    ];
    assert_eq!(
        texts_of(&plain, 33),
        [
            "file 48",
            "→ Speakers (Sound Blaster GC7)",
            "(exclusive), bit-perfect"
        ]
    );
    assert_eq!(
        texts_of(&plain, 63),
        ["file 48 → Speakers (Sound Blaster GC7) (exclusive), bit-perfect"]
    );
}

fn texts_of(path: &[Seg], max: usize) -> Vec<String> {
    text::wrap_path(path, max)
        .iter()
        .map(|l| line_text(l))
        .collect()
}

/// A state playing `path`, on another screen and the queue side.
fn playing_at(path: &str) -> State {
    let mut state = common::playing_state(120, 36);
    let track = state.playing.track.as_mut().unwrap();
    let mut info = (*track.info).clone();
    info.path = PathBuf::from(path);
    track.info = std::sync::Arc::new(info);
    state.screen = Screen::NowPlaying;
    state.side = Side::Queue;
    state
}

fn jumped(state: &State, fx: &[Effect], dir: &str, select: &str) {
    assert_eq!(state.screen, Screen::Main);
    assert_eq!(state.side, Side::Browser);
    assert_eq!(state.browser.dir, PathBuf::from(dir));
    assert!(fx.contains(&Effect::ListDir(PathBuf::from(dir))), "{fx:?}");
    assert_eq!(
        state.browser.last_child.get(&PathBuf::from(dir)),
        Some(&PathBuf::from(select)),
        "selects {select}"
    );
}

#[test]
fn title_artist_and_album_go_to_the_playing_track() {
    let track = "/music/Artist/Album/02 Song.flac";
    let mut state = playing_at(track);
    let fx = update(&mut state, Action::Op(Op::GoTrack));
    jumped(&state, &fx, "/music/Artist/Album", track);

    let mut state = playing_at(track);
    let fx = update(&mut state, Action::Op(Op::GoAlbum));
    jumped(&state, &fx, "/music/Artist/Album", track);

    let mut state = playing_at(track);
    let fx = update(&mut state, Action::Op(Op::GoArtist));
    jumped(&state, &fx, "/music/Artist", "/music/Artist/Album");

    // An album right under the root has no artist folder: its album.
    let mut state = playing_at("/Album/01.flac");
    let fx = update(&mut state, Action::Op(Op::GoArtist));
    jumped(&state, &fx, "/Album", "/Album/01.flac");
    #[cfg(windows)]
    {
        let mut state = playing_at(r"D:\Album\01.flac");
        let fx = update(&mut state, Action::Op(Op::GoArtist));
        jumped(&state, &fx, r"D:\Album", r"D:\Album\01.flac");
    }

    // Nothing playing: a note, no jump.
    let mut state = listed();
    let dir = state.browser.dir.clone();
    let fx = update(&mut state, Action::Op(Op::GoArtist));
    assert!(fx.is_empty() && state.browser.dir == dir);
    assert!(state.message.is_some());

    // Keys: capitals, so typing ahead in the browser keeps the letters.
    let mut state = playing_at(track);
    state.screen = Screen::Main;
    state.side = Side::Browser;
    for (c, op) in [('T', Op::GoTrack), ('B', Op::GoAlbum), ('R', Op::GoArtist)] {
        let key = Key::ch(c);
        assert_eq!(keymap::lookup(&state, key), Some(Meaning::Op(op)), "{c}");
    }
    for name in ["Go to playing track", "Go to album", "Go to artist"] {
        assert_eq!(keymap::palette_matches(name).len(), 1, "{name}");
    }
}

#[test]
fn clicking_the_title_artist_or_album_jumps() {
    use jodeldiplom::core::mouse::{self, MouseButton, MouseInput, MouseKind, Target};
    let track = "/music/Artist/Album/02 Song.flac";
    let mut state = playing_at(track);
    state.screen = Screen::Main;
    let Layout::Main(m) = layout::compute(&state) else {
        panic!()
    };
    let info = m.info.unwrap();
    // Title, artist, album: the first three lines of the panel.
    for (row, op) in [(0, Op::GoTrack), (1, Op::GoArtist), (2, Op::GoAlbum)] {
        assert_eq!(
            mouse::hit(&state, info.x + 1, info.y + row),
            Some(Target::Link(op)),
            "row {row}"
        );
    }
    // Past the end of the title there is no link.
    assert_eq!(mouse::hit(&state, info.x + 40, info.y), Some(Target::Info));
    let click = MouseInput {
        kind: MouseKind::Down(MouseButton::Left),
        col: info.x + 2,
        row: info.y + 1,
    };
    let action = mouse::action(&state, click).unwrap();
    let fx = update(&mut state, action);
    jumped(&state, &fx, "/music/Artist", "/music/Artist/Album");

    // On Now playing: the caption's links, the rest of the screen returns.
    let state = playing_at(track);
    let Layout::NowPlaying(n) = layout::compute(&state) else {
        panic!()
    };
    let c = n.caption;
    assert_eq!(
        mouse::hit(&state, c.x, c.y),
        Some(Target::Link(Op::GoTrack))
    );
    assert_eq!(
        mouse::hit(&state, c.x, c.y + 1),
        Some(Target::Link(Op::GoArtist))
    );
    let album_x = c.x + "The Ramps · ".chars().count() as u16;
    assert_eq!(
        mouse::hit(&state, album_x, c.y + 1),
        Some(Target::Link(Op::GoAlbum))
    );
    assert_eq!(mouse::hit(&state, c.x, c.y + 2), Some(Target::Cover));
    assert_eq!(mouse::hit(&state, 0, 0), Some(Target::Cover));
}

/// Steps through settings reach the engine once, 400 ms after the last
/// one, with what the screen shows by then; quitting sends what is pending.
#[test]
fn settings_reach_the_engine_after_a_pause() {
    let mut state = listed();
    update(&mut state, Action::Tick(1_000));
    op(&mut state, Op::Settings);
    state.settings.row = 1;
    // Three steps of the rate cap, 100 ms apart: shown at once, not sent.
    for (i, at) in [1_000u64, 1_100, 1_200].into_iter().enumerate() {
        update(&mut state, Action::Tick(at));
        let fx = op(&mut state, Op::ValueUp);
        assert_eq!(fx, vec![Effect::SaveConfig], "step {i}");
        assert_eq!(state.config.max_rate, jodeldiplom::config::RATE_CAPS[i + 1]);
    }
    assert_eq!(
        state.settings_due,
        Some(1_200 + jodeldiplom::core::SETTINGS_DELAY_MS)
    );
    // Nothing before the pause ends ...
    for at in [1_300, 1_500, 1_599] {
        assert!(engine(&update(&mut state, Action::Tick(at))).is_empty());
    }
    // ... then once, with the last value.
    let fx = update(&mut state, Action::Tick(1_600));
    assert_eq!(
        engine(&fx),
        vec![Command::SetSettings(state.engine_settings())]
    );
    assert_eq!(state.engine_settings().max_rate, Some(192_000));
    assert_eq!(state.settings_due, None);
    assert!(engine(&update(&mut state, Action::Tick(5_000))).is_empty());
    // Quitting first sends what is pending.
    let fx = op(&mut state, Op::ValueUp);
    assert_eq!(fx, vec![Effect::SaveConfig]);
    let fx = op(&mut state, Op::Quit);
    assert_eq!(
        fx,
        vec![
            Effect::Engine(Command::SetSettings(state.engine_settings())),
            Effect::Quit
        ]
    );
}

/// The rate caps offered are those the device can satisfy; a saved cap
/// below every rate the device takes is shown as the lowest cap offered,
/// and the engine plays with that, while the file keeps the saved one.
#[test]
fn the_rate_caps_follow_the_device() {
    use jodeldiplom::config::RATE_CAPS;
    let mut state = listed();
    // Not known yet: every cap.
    assert_eq!(state.rate_caps(), RATE_CAPS.to_vec());
    // A device with 48, 96 and 192 kHz only.
    update(
        &mut state,
        Action::DeviceRates(Some(vec![192_000, 48_000, 96_000])),
    );
    assert_eq!(state.device_rates, Some(vec![48_000, 96_000, 192_000]));
    let caps = state.rate_caps();
    assert_eq!(
        caps,
        [
            None,
            Some(384_000),
            Some(352_800),
            Some(192_000),
            Some(176_400),
            Some(96_000),
            Some(88_200),
            Some(48_000)
        ]
    );
    // Stepping goes round the offered caps only.
    op(&mut state, Op::Settings);
    state.settings.row = 1;
    let mut seen = vec![state.config.max_rate];
    for _ in 0..caps.len() {
        op(&mut state, Op::ValueUp);
        seen.push(state.config.max_rate);
    }
    assert_eq!(&seen[..caps.len()], &caps[..]);
    assert_eq!(seen[caps.len()], None);
    op(&mut state, Op::ValueDown);
    assert_eq!(state.config.max_rate, Some(48_000));
    op(&mut state, Op::ValueDown);
    assert_eq!(state.config.max_rate, Some(88_200));
    // A cap saved for another device: shown as the lowest offered, which
    // the engine gets; the saved value stays.
    let mut state = listed();
    state.config.max_rate = Some(44_100);
    update(
        &mut state,
        Action::DeviceRates(Some(vec![48_000, 96_000, 192_000])),
    );
    assert_eq!(state.config.max_rate, Some(44_100));
    assert_eq!(state.effective_max_rate(), Some(48_000));
    assert_eq!(state.engine_settings().max_rate, Some(48_000));
    let message = state.message.as_ref().unwrap().text.clone();
    assert!(message.contains("using 48 kHz"), "{message}");
    op(&mut state, Op::Settings);
    // One step up from what is in force.
    update(&mut state, Action::SettingsClick(1, 1));
    assert_eq!(state.config.max_rate, None);
    // Back on a device that takes 44.1 kHz, the saved cap is in force.
    let mut state = listed();
    state.config.max_rate = Some(44_100);
    update(&mut state, Action::DeviceRates(Some(vec![44_100, 48_000])));
    assert_eq!(state.effective_max_rate(), Some(44_100));
    assert!(state.message.is_none());
}

/// A notice from the engine is shown as a message; nothing stops.
#[test]
fn engine_notices_are_shown() {
    let mut state = listed();
    let fx = update(
        &mut state,
        Action::Engine(Event::Notice {
            index: Some(0),
            message: "No route for this track with these settings on DAC; keeping 44.1 kHz".into(),
        }),
    );
    assert!(engine(&fx).is_empty());
    let message = state.message.unwrap();
    assert!(message.text.starts_with("No route for this track"));
    assert_eq!(message.level, jodeldiplom::core::Level::Info);
}

/// The headroom row steps through none, 0.5, 1, 2 and 3 dB, debounced like
/// the other settings.
#[test]
fn the_headroom_setting_steps_and_reaches_the_engine() {
    use jodel_playback::Headroom;
    let mut state = listed();
    op(&mut state, Op::Settings);
    state.settings.row = 3;
    assert_eq!(
        state.settings_row(),
        jodeldiplom::core::SettingsRow::Headroom
    );
    let mut seen = Vec::new();
    for _ in 0..5 {
        assert_eq!(op(&mut state, Op::ValueUp), vec![Effect::SaveConfig]);
        seen.push(state.config.headroom);
    }
    assert_eq!(
        seen,
        [
            Headroom(5),
            Headroom(10),
            Headroom(20),
            Headroom(30),
            Headroom(0)
        ]
    );
    op(&mut state, Op::ValueDown);
    assert_eq!(state.config.headroom, Headroom(30));
    let due = state.settings_due.unwrap();
    let fx = update(&mut state, Action::Tick(due));
    assert_eq!(
        engine(&fx),
        vec![Command::SetSettings(state.engine_settings())]
    );
    assert_eq!(state.engine_settings().headroom, Headroom(30));
}

fn rhythm_asked(fx: &[Effect]) -> Option<Vec<PathBuf>> {
    fx.iter().find_map(|e| match e {
        Effect::WantRhythm(paths, _) => Some(paths.clone()),
        _ => None,
    })
}

/// Every beat method, the default.
fn all() -> Vec<Method> {
    Method::ALL.to_vec()
}

/// `methods` without stems (the tests have no GPU).
fn recipe(methods: Vec<Method>) -> RhythmRecipe {
    RhythmRecipe {
        methods,
        stems: false,
    }
}

fn bpm_text(state: &State) -> Option<String> {
    text::bpm_line(state).map(|line| line.iter().map(|s| s.text.as_str()).collect())
}

/// An analysis of `beats` whose regions have `confidence`.
fn analysis(beats: Vec<f64>, confidence: impl Fn(usize) -> f64) -> jodel_rhythm::Analysis {
    let mut a = jodel_rhythm::Analysis {
        regions: beats
            .chunks(16)
            .enumerate()
            .map(|(i, c)| jodel_rhythm::Region {
                start: c[0],
                end: c[c.len() - 1],
                confidence: confidence(i),
            })
            .collect(),
        beats,
        ..Default::default()
    };
    a.derive();
    a
}

#[test]
fn the_playing_track_and_the_next_two_are_analysed_and_shown() {
    let mut state = common::playing_state(120, 36);
    state.rhythm.clear();
    state.rhythm_wanted.clear();
    let paths = state.queue.paths.clone();
    // Any update notices what is wanted: the playing track, the next two.
    let fx = update(&mut state, Action::Tick(1_000));
    assert_eq!(rhythm_asked(&fx).unwrap(), paths[1..4].to_vec());
    assert_eq!(bpm_text(&state).unwrap(), "♩ … BPM");
    assert!(rhythm_asked(&update(&mut state, Action::Tick(2_000))).is_none());
    // Results arrive; one for a file no longer wanted is dropped.
    let steady = analysis((0..200).map(|i| f64::from(i) * 0.5).collect(), |_| 0.9);
    update(
        &mut state,
        Action::Rhythm(
            paths[1].clone(),
            recipe(all()),
            Ok(std::sync::Arc::new(jodeldiplom::core::Rhythmed::new(
                steady, None,
            ))),
        ),
    );
    assert_eq!(bpm_text(&state).unwrap(), "♩ 120 BPM");
    update(
        &mut state,
        Action::Rhythm(paths[2].clone(), recipe(all()), Err("bad".into())),
    );
    update(
        &mut state,
        Action::Rhythm("/x.flac".into(), recipe(all()), Err("x".into())),
    );
    assert!(!state.rhythm.contains_key(&PathBuf::from("/x.flac")));
    // The next track: the one that comes into view is asked for with the
    // one still pending; the failed one is not asked again, the finished
    // one is forgotten.
    let fx = update(
        &mut state,
        Action::Engine(Event::QueueChanged {
            queue: paths.clone(),
            current: Some(2),
        }),
    );
    assert_eq!(rhythm_asked(&fx).unwrap(), paths[3..5].to_vec());
    assert!(!state.rhythm.contains_key(&paths[1]));
}

#[test]
fn a_varying_tempo_shows_the_tempo_here_and_the_range_and_m_turns_it_off() {
    let mut state = common::playing_state(120, 36);
    let path = state.queue.paths[1].clone();
    let mut beats = Vec::new();
    let mut t = 0.0;
    for i in 0..400 {
        beats.push(t);
        t += 60.0 / (90.0 + 0.1 * f64::from(i));
    }
    let set = |state: &mut State, a: jodel_rhythm::Analysis| {
        state.rhythm.insert(
            path.clone(),
            jodeldiplom::core::Rhythm::Ready(std::sync::Arc::new(
                jodeldiplom::core::Rhythmed::new(a, None),
            )),
        );
    };
    // Playing at 1:23.
    let a = analysis(beats.clone(), |_| 0.9);
    let s = a.summary.unwrap();
    assert!(!s.steady);
    let here = a.bpm_at(83.0).unwrap();
    let want = format!("♩ {here:.0} BPM · {:.0}–{:.0}", s.low, s.high);
    set(&mut state, a.clone());
    assert_eq!(bpm_text(&state).unwrap(), want);
    // Unclear here: the range only.
    let unclear = a.region_at(83.0).map(|r| r.start).unwrap();
    let b = analysis(beats.clone(), |i| {
        if (a.regions[i].start - unclear).abs() < 1e-9 {
            0.2
        } else {
            0.9
        }
    });
    let s = b.summary.unwrap();
    set(&mut state, b);
    assert_eq!(
        bpm_text(&state).unwrap(),
        format!("♩ {:.0}–{:.0} BPM", s.low, s.high)
    );
    // Clear nowhere: a dash.
    set(&mut state, analysis(beats, |_| 0.2));
    assert_eq!(bpm_text(&state).unwrap(), "♩ – BPM");
    // M turns it off: no line, nothing analysed, and the file keeps it.
    let fx = op(&mut state, Op::ToggleBpm);
    assert!(bpm_text(&state).is_none());
    assert_eq!(rhythm_asked(&fx), Some(Vec::new()));
    assert!(state.rhythm.is_empty());
    assert!(fx.contains(&Effect::SaveConfig));
    assert!(!state.config.bpm);
}

/// Selects settings row `row` and steps it once.
fn step_row(state: &mut State, row: SettingsRow) -> Vec<Effect> {
    let index = SettingsRow::ALL.iter().position(|r| *r == row).unwrap();
    update(state, Action::SettingsClick(index, 1))
}

/// What the next-track key does is a row of the settings: cut or mix in,
/// saved, `X` switching it too; and a row under it, which preset it mixes
/// by: the Transition tab's (the default), or each preset, stepped either way.
#[test]
fn the_next_track_mix_is_a_setting() {
    use jodeldiplom::core::transition::Preset;
    let mut state = common::playing_state(120, 36);
    assert!(!state.config.skip_mix, "cut by default");
    assert_eq!(
        state.config.skip_preset, None,
        "the Transition tab's by default"
    );
    let row = |r: SettingsRow| SettingsRow::ALL.iter().position(|x| *x == r).unwrap();
    let fx = update(
        &mut state,
        Action::SettingsClick(row(SettingsRow::SkipMix), 1),
    );
    assert!(state.config.skip_mix);
    assert!(fx.contains(&Effect::SaveConfig));
    update(
        &mut state,
        Action::SettingsClick(row(SettingsRow::SkipMix), -1),
    );
    assert!(!state.config.skip_mix);
    let by = row(SettingsRow::SkipPreset);
    let fx = update(&mut state, Action::SettingsClick(by, 1));
    assert_eq!(state.config.skip_preset, Some(Preset::Fade));
    assert!(fx.contains(&Effect::SaveConfig));
    update(&mut state, Action::SettingsClick(by, -1));
    assert_eq!(state.config.skip_preset, None);
    update(&mut state, Action::SettingsClick(by, -1));
    assert_eq!(state.config.skip_preset, Some(*Preset::ALL.last().unwrap()));
    let (back, _) = Config::parse(&state.config.to_text()).unwrap();
    assert_eq!(
        (back.skip_mix, back.skip_preset),
        (state.config.skip_mix, state.config.skip_preset)
    );
    // X only switches cut and mix; the preset stays.
    op(&mut state, Op::SkipMix);
    assert!(state.config.skip_mix);
    op(&mut state, Op::SkipMix);
    assert!(!state.config.skip_mix);
    assert_eq!(state.config.skip_preset, Some(*Preset::ALL.last().unwrap()));
}

/// The data folder is chosen in a folder list: it opens on the folder the
/// data folder is in, folders open and `backspace` goes up, and the folder
/// shown is chosen by its first row: the app moves the data there. Not the
/// data folder itself, nor a folder inside it; while a move that stopped
/// waits, the row tries it again.
#[test]
fn the_data_folder_is_chosen_in_a_folder_list() {
    use jodeldiplom::core::{DataMove, FolderRow};
    use std::path::PathBuf;
    let root = if cfg!(windows) { r"D:\" } else { "/srv/" };
    let at = |p: &str| PathBuf::from(format!("{root}{p}"));
    let mut state = common::playing_state(120, 40);
    state.screen = jodeldiplom::core::Screen::Settings;
    state.data_dir = Some(at("old/Jodeldiplom Data"));
    let row = SettingsRow::ALL
        .iter()
        .position(|r| *r == SettingsRow::DataFolder)
        .unwrap();
    let fx = update(&mut state, Action::SettingsClick(row, 0));
    assert_eq!(fx, vec![Effect::ListFolders(Some(at("old")))]);
    update(
        &mut state,
        Action::FoldersListed(
            Some(at("old")),
            vec![at("old/music"), at("old/Jodeldiplom Data")],
        ),
    );
    let pick = state.settings.folders.clone().unwrap();
    assert_eq!(
        pick.rows(),
        vec![
            FolderRow::Choose(at("old")),
            FolderRow::Up,
            FolderRow::Folder(at("old/music")),
            FolderRow::Folder(at("old/Jodeldiplom Data")),
        ]
    );
    // Into music, and back up.
    op(&mut state, Op::Down);
    op(&mut state, Op::Down);
    let fx = op(&mut state, Op::Open);
    assert_eq!(fx, vec![Effect::ListFolders(Some(at("old/music")))]);
    let fx = op(&mut state, Op::Parent);
    assert_eq!(fx, vec![Effect::ListFolders(Some(at("old")))]);
    // The folder the data folder is in already: nothing to move.
    update(&mut state, Action::FoldersListed(Some(at("old")), vec![]));
    let fx = op(&mut state, Op::Open);
    assert!(fx.is_empty(), "{fx:?}");
    assert!(state.settings.folders.is_none(), "the list closes");
    // Inside the data folder: refused.
    update(&mut state, Action::SettingsClick(row, 0));
    update(&mut state, Action::FolderRow(0));
    update(&mut state, Action::SettingsClick(row, 0));
    let inside = at("old/Jodeldiplom Data/stems");
    state.settings.folders.as_mut().unwrap().dir = Some(inside.clone());
    let fx = update(&mut state, Action::FolderRow(0));
    assert!(fx.is_empty(), "{fx:?}");
    // Elsewhere: moved there.
    update(&mut state, Action::SettingsClick(row, 0));
    state.settings.folders.as_mut().unwrap().dir = Some(at("new"));
    let fx = update(&mut state, Action::FolderRow(0));
    assert_eq!(fx, vec![Effect::MoveData(at("new"))]);
    // A move that stopped: the row tries it again, no list.
    state.data_move = Some(DataMove {
        to: at("new/Jodeldiplom Data"),
        failure: Some("full".into()),
        ..Default::default()
    });
    let fx = update(&mut state, Action::SettingsClick(row, 0));
    assert_eq!(fx, vec![Effect::RetryMove]);
    assert!(state.settings.folders.is_none());
    // Esc closes the list.
    state.data_move = None;
    update(&mut state, Action::SettingsClick(row, 0));
    op(&mut state, Op::Back);
    assert!(state.settings.folders.is_none());
}

/// The Transition tab's preset and tempo mode are remembered: a new run starts
/// from them, and changing either saves it.
#[test]
fn the_transition_preset_and_tempo_are_remembered() {
    use jodel_playback::TempoMode;
    use jodeldiplom::core::transition::Preset;
    let config = Config {
        transition_preset: Preset::BassSwap,
        transition_tempo: TempoMode::Stretch,
        ..Config::default()
    };
    let state = State::new(std::env::temp_dir(), config, (120, 40));
    assert_eq!(state.transition.draft.preset, Preset::BassSwap);
    assert_eq!(state.transition.draft.tempo, TempoMode::Stretch);
    let mut state = common::transition_state(120, 40);
    let fx = op(&mut state, Op::TransitionPreset);
    assert_eq!(
        state.config.transition_preset,
        state.transition.draft.preset
    );
    assert_ne!(state.config.transition_preset, Preset::default());
    assert!(fx.contains(&Effect::SaveConfig));
    let fx = op(&mut state, Op::TransitionTempo);
    assert_eq!(state.config.transition_tempo, state.transition.draft.tempo);
    assert!(fx.contains(&Effect::SaveConfig));
    let (back, _) = Config::parse(&state.config.to_text()).unwrap();
    assert_eq!(
        (back.transition_preset, back.transition_tempo),
        (
            state.config.transition_preset,
            state.config.transition_tempo
        )
    );
}

#[test]
fn stems_need_a_gpu_and_change_the_recipe() {
    let mut state = common::playing_state(120, 36);
    let paths = state.queue.paths.clone();
    // No GPU: the row cannot be switched, and the cues listen to the mix.
    step_row(&mut state, SettingsRow::Stems);
    assert!(state.config.stems);
    assert!(!state.rhythm_recipe.stems);
    // Still looking: nothing is asked for while stems are on.
    let mut looking = common::playing_state(120, 36);
    looking.gpu = Gpu::Looking;
    looking.rhythm_wanted.clear();
    looking.rhythm.clear();
    let fx = update(&mut looking, Action::Tick(1));
    assert!(!fx.iter().any(|e| matches!(e, Effect::WantRhythm(..))));
    // Found: asked for again with stems.
    let fx = update(&mut looking, Action::Gpu(Some("GPU".into())));
    assert!(fx.iter().any(|e| *e
        == Effect::WantRhythm(
            paths[1..4].to_vec(),
            RhythmRecipe {
                methods: all(),
                stems: true
            }
        )));
    // Switched off: the mix again, and saved.
    let fx = step_row(&mut looking, SettingsRow::Stems);
    assert!(!looking.config.stems);
    assert!(fx.contains(&Effect::SaveConfig));
    assert!(fx
        .iter()
        .any(|e| *e == Effect::WantRhythm(paths[1..4].to_vec(), recipe(all()))));
}

#[test]
fn beat_methods_are_checkboxes_while_voting_and_a_radio_without() {
    let mut state = common::playing_state(120, 36);
    let paths = state.queue.paths.clone();
    assert_eq!(state.config.methods_in_force(), all());
    let flux = SettingsRow::Method(Method::Flux);
    let network = SettingsRow::Method(Method::Network);

    // Voting: unchecking the network leaves flux; the analyses so far are
    // dropped and asked for again with flux alone.
    let fx = step_row(&mut state, network);
    assert_eq!(state.config.beat_methods, vec![Method::Flux]);
    assert!(fx.contains(&Effect::SaveConfig));
    assert!(fx
        .iter()
        .any(|e| *e == Effect::WantRhythm(paths[1..4].to_vec(), recipe(vec![Method::Flux]))));
    assert!(state.rhythm.values().all(|r| matches!(r, Rhythm::Pending)));
    // A result for the methods before is dropped; for these it is kept.
    let a = analysis((0..100).map(|i| f64::from(i) * 0.5).collect(), |_| 0.9);
    update(
        &mut state,
        Action::Rhythm(
            paths[1].clone(),
            recipe(all()),
            Ok(std::sync::Arc::new(jodeldiplom::core::Rhythmed::new(
                a.clone(),
                None,
            ))),
        ),
    );
    assert!(matches!(state.rhythm[&paths[1]], Rhythm::Pending));
    update(
        &mut state,
        Action::Rhythm(
            paths[1].clone(),
            recipe(vec![Method::Flux]),
            Ok(std::sync::Arc::new(jodeldiplom::core::Rhythmed::new(
                a, None,
            ))),
        ),
    );
    assert!(matches!(state.rhythm[&paths[1]], Rhythm::Ready(_)));
    // The last checked method stays.
    step_row(&mut state, flux);
    assert_eq!(state.config.beat_methods, vec![Method::Flux]);
    assert!(state.message.as_ref().unwrap().text.contains("stays on"));
    step_row(&mut state, network);
    assert_eq!(state.config.beat_methods, all());

    // Without voting, a method row picks the one method.
    step_row(&mut state, SettingsRow::Voting);
    assert!(!state.config.beat_voting);
    assert_eq!(state.config.methods_in_force(), vec![Method::Network]);
    step_row(&mut state, flux);
    assert_eq!(state.config.methods_in_force(), vec![Method::Flux]);
    step_row(&mut state, flux);
    assert_eq!(state.config.beat_method, Method::Flux);
    // The checkboxes are kept for when voting comes back.
    assert_eq!(state.config.beat_methods, all());
    step_row(&mut state, SettingsRow::Voting);
    assert_eq!(state.config.methods_in_force(), all());

    // The tempo row is the M key's setting.
    step_row(&mut state, SettingsRow::Tempo);
    assert!(!state.config.bpm);
    assert!(state.rhythm_wanted.is_empty());
}

/// The stems: alt+1..4 mute and unmute a stem's strip on the desk, alt+0
/// puts them all back; without bars there they change at once, and the
/// engine's transitions are told; the buttons by the transport buttons show
/// and switch them (with a GPU: without one only kept stems play, and there
/// are no buttons).
#[test]
fn stems_switch_from_keys_and_buttons_with_a_gpu() {
    use jodel_playback::Stem;
    use jodeldiplom::core::desk::{DeskChange, StripKind};
    let mut state = listed();
    state.gpu = Gpu::Absent;
    let alt = |c| Key {
        code: KeyCode::Char(c),
        ctrl: false,
        alt: true,
        shift: false,
    };
    assert_eq!(
        keymap::lookup(&state, alt('1')),
        Some(Meaning::Op(Op::ToggleDrums))
    );
    assert_eq!(
        keymap::lookup(&state, alt('0')),
        Some(Meaning::Op(Op::AllStems))
    );
    let desk = |fx: &[Effect]| -> Vec<DeskChange> {
        fx.iter()
            .filter_map(|e| match e {
                Effect::Desk(c) => Some(*c),
                _ => None,
            })
            .collect()
    };
    // Without a GPU the drums' strip mutes all the same (kept stems play).
    let fx = op(&mut state, Op::ToggleDrums);
    assert_eq!(desk(&fx), [DeskChange::Mute(StripKind::Drums, true)]);
    assert_eq!(
        engine(&fx),
        // A stem touched: the kept ones made.
        [Command::MakeStems(jodel_playback::MakeStems::Kept)]
    );
    assert!(state.message.as_ref().unwrap().text.contains("no GPU"));
    op(&mut state, Op::ToggleDrums);
    let stem_buttons = |state: &State| {
        let Layout::Main(m) = layout::compute(state) else {
            panic!()
        };
        m.bottom
            .buttons
            .iter()
            .filter(|b| b.op.stem().is_some())
            .cloned()
            .collect::<Vec<_>>()
    };
    assert!(stem_buttons(&state).is_empty());

    state.gpu = Gpu::Present("a GPU".into());
    // The vocals off: their strip muted at once, nothing sent to the engine
    // but to make the stems (no settings, no route).
    let fx = op(&mut state, Op::ToggleVocals);
    assert!(state.desk.setting(StripKind::Vocals).mute);
    assert_eq!(desk(&fx), [DeskChange::Mute(StripKind::Vocals, true)]);
    assert_eq!(
        engine(&fx),
        [Command::MakeStems(jodel_playback::MakeStems::Separate)]
    );
    assert!(state
        .message
        .as_ref()
        .unwrap()
        .text
        .contains("not bit-perfect"));
    // Four buttons in the order people name them, by initial at 87
    // columns; a click switches one.
    update(&mut state, Action::Resize(87, 30));
    let buttons = stem_buttons(&state);
    let labels: Vec<&str> = buttons.iter().map(|b| b.label.trim()).collect();
    assert_eq!(labels, ["D", "B", "V", "O"]);
    let drums = &buttons[0];
    let click = jodeldiplom::core::mouse::MouseInput {
        kind: jodeldiplom::core::mouse::MouseKind::Down(
            jodeldiplom::core::mouse::MouseButton::Left,
        ),
        col: drums.rect.x + 1,
        row: drums.rect.y,
    };
    let action = jodeldiplom::core::mouse::action(&state, click).unwrap();
    assert!(matches!(action, Action::Op(Op::ToggleDrums)));
    update(&mut state, action);
    assert!(jodeldiplom::core::stems::is_off(&state, Stem::Drums));
    // By name where there is room.
    update(&mut state, Action::Resize(140, 30));
    let named: Vec<String> = stem_buttons(&state)
        .iter()
        .map(|b| b.label.trim().to_owned())
        .collect();
    assert_eq!(named, ["drums", "bass", "vocals", "other"]);
    // All on: every stem's strip as it starts (a fader moved too), the file
    // as it is.
    state.desk.strips[StripKind::Bass.index()].db = -6.0;
    let fx = op(&mut state, Op::AllStems);
    assert!(!state.desk.stems_touched());
    let changes = desk(&fx);
    for c in [
        DeskChange::Level(StripKind::Bass, 1.0),
        DeskChange::Mute(StripKind::Drums, false),
        DeskChange::Mute(StripKind::Vocals, false),
    ] {
        assert!(changes.contains(&c), "{changes:?}");
    }
    assert_eq!(
        engine(&fx),
        // Nothing touched away from the tabs that show stems: none made.
        [Command::MakeStems(jodel_playback::MakeStems::Never)]
    );
    assert!(state.message.as_ref().unwrap().text.contains("all on"));
    // The palette lists the five.
    assert_eq!(keymap::palette_matches("stems").len(), 5);
}

/// A playback engine that failed is started again by the app, which makes
/// the desk again as it was set (M2); the state says so.
#[test]
fn a_failed_engine_is_started_again_and_the_desk_kept() {
    use jodeldiplom::core::desk::StripKind;
    let mut state = listed();
    op(&mut state, Op::ToggleDrums);
    let fx = update(&mut state, Action::EngineGone(1));
    assert!(fx.is_empty(), "the app starts the engine");
    assert!(state.desk.setting(StripKind::Drums).mute);
    assert!(state
        .message
        .as_ref()
        .unwrap()
        .text
        .contains("started again"));
}

#[test]
fn ctrl_v_pastes_into_the_filter_and_the_palette() {
    let mut state = listed();
    // On the list, ctrl+v reads the clipboard and opens a filter with it.
    let ctrl_v = Key::ctrl('v');
    assert_eq!(keymap::lookup(&state, ctrl_v), Some(Meaning::Op(Op::Paste)));
    assert_eq!(op(&mut state, Op::Paste), vec![Effect::ReadClipboard]);
    update(&mut state, Action::Paste("b.fl".into()));
    let filter = state.browser.filter.clone().unwrap();
    assert_eq!((filter.text.as_str(), filter.editing), ("b.fl", true));
    assert_eq!(browser_rows(&state), vec![Row::Entry(3), Row::SearchBelow]);
    // While typing, more goes on the end; line breaks and tabs become
    // spaces and the ends are trimmed.
    assert_eq!(keymap::lookup(&state, ctrl_v), Some(Meaning::Op(Op::Paste)));
    update(&mut state, Action::Paste("ac\r\n".into()));
    assert_eq!(state.browser.filter.as_ref().unwrap().text, "b.flac");
    // A path pasted and accepted is jumped to.
    op(&mut state, Op::Back);
    update(&mut state, Action::Paste("  D:\\Music\tLive\n".into()));
    let fx = op(&mut state, Op::Accept);
    assert!(fx.contains(&Effect::Jump {
        text: "D:\\Music Live".into(),
        from: "/m/Album".into()
    }));
    // In the palette it goes into the palette's text.
    op(&mut state, Op::Back);
    op(&mut state, Op::Palette);
    assert_eq!(op(&mut state, Op::Paste), vec![Effect::ReadClipboard]);
    update(&mut state, Action::Paste("stems".into()));
    assert_eq!(state.palette.as_ref().unwrap().text, "stems");
    // Nothing but blanks pastes nothing; other screens take no paste.
    op(&mut state, Op::Back);
    update(&mut state, Action::Paste(" \n ".into()));
    assert!(state.palette.is_none());
    op(&mut state, Op::Help);
    assert!(op(&mut state, Op::Paste).is_empty());
}

/// The playing track's analysis as the state has it.
fn block_analysis(state: &State, block: jodeldiplom::core::lanes::Block) -> jodel_rhythm::Analysis {
    (*jodeldiplom::core::lanes::rhythmed(state, block)
        .unwrap()
        .analysis)
        .clone()
}

/// On the Transition tab the digits jump the playing track to the start of a
/// bar around the one heard (5 its own, 6 the next, 4 the one before …) when
/// the bar heard ends; with shift (or the symbol a US keyboard's shift makes,
/// as a terminal sends it) at once. Past the bars found they go on at the
/// bar's length; none before the track starts, or during a transition.
#[test]
fn digits_on_the_transition_tab_jump_to_bars_around_the_one_heard() {
    use jodel_playback::JumpAt;
    use jodeldiplom::core::lanes::Block;
    let mut state = common::transition_state(120, 40);
    let a = block_analysis(&state, Block::Now);
    let starts: Vec<f64> = a.downbeats.iter().map(|&d| a.beats[d]).collect();
    let heard = (starts[10] + starts[11]) / 2.0;
    state.drawn_heard = Some(heard);
    let index = state.queue.current.unwrap();
    let jump = |state: &mut State, key: Key| {
        let meaning = keymap::lookup(state, key);
        let Some(Meaning::Op(o)) = meaning else {
            panic!("{key:?}: {meaning:?}");
        };
        match &engine(&op(state, o))[..] {
            [Command::Jump { index: i, to, at }] => {
                assert_eq!(*i, index);
                (*to, *at)
            }
            other => panic!("{key:?}: {other:?}"),
        }
    };
    let shifted = [
        ('5', '%', 10),
        ('6', '^', 11),
        ('4', '$', 9),
        ('9', '(', 14),
        ('1', '!', 6),
    ];
    for (c, symbol, bar) in shifted {
        for key in [Key::shifted(c), Key::ch(symbol)] {
            assert_eq!(jump(&mut state, key), (starts[bar], JumpAt::Now), "{key:?}");
        }
    }
    assert_eq!(
        jump(&mut state, Key::ch('5')),
        (starts[10], JumpAt::At(starts[11]))
    );
    assert_eq!(
        jump(&mut state, Key::ch('7')),
        (starts[12], JumpAt::At(starts[11]))
    );
    // At the last bar found, ahead at its length.
    let n = starts.len();
    state.drawn_heard = Some(starts[n - 1] + 0.1);
    let bar = (starts[n - 1] - starts[0]) / (n - 1) as f64;
    let (to, _) = jump(&mut state, Key::shifted('6'));
    assert!((to - (starts[n - 1] + bar)).abs() < 1e-9, "{to}");
    // Before the track starts: nothing.
    state.drawn_heard = Some(starts[0] + 0.1);
    let fx = op(&mut state, Op::JumpBar(-4));
    assert!(engine(&fx).is_empty() || starts[0] - 4.0 * bar >= 0.0);
    // A transition set up: none, and a message.
    state.drawn_heard = Some(heard);
    state.transition.draft.status = jodeldiplom::core::transition::Status::Armed;
    let fx = op(&mut state, Op::JumpBar(0));
    assert!(engine(&fx).is_empty());
    assert!(state.message.is_some());
}

#[test]
fn tab_cycles_through_browser_queue_transition_karaoke_and_mixer() {
    let mut state = common::transition_state(120, 40);
    state.side = Side::Browser;
    op(&mut state, Op::ToggleSide);
    assert_eq!(state.side, Side::Queue);
    op(&mut state, Op::ToggleSide);
    assert_eq!(state.side, Side::Transition);
    // On the Transition tab, keys are its own; letters do not jump to names.
    assert_eq!(
        keymap::lookup(&state, key(KeyCode::Left)),
        Some(Meaning::Op(Op::TransitionBarBack))
    );
    assert_eq!(
        keymap::lookup(&state, Key::ch('[')),
        Some(Meaning::Op(Op::GridEarlier))
    );
    assert_eq!(
        keymap::lookup(&state, Key::ch(' ')),
        Some(Meaning::Op(Op::PlayPause))
    );
    assert_eq!(keymap::lookup(&state, Key::ch('a')), None);
    op(&mut state, Op::ToggleSide);
    assert_eq!(state.side, Side::Karaoke);
    op(&mut state, Op::ToggleSide);
    assert_eq!(state.side, Side::Mixer);
    op(&mut state, Op::ToggleSide);
    assert_eq!(state.side, Side::Browser);
}

#[test]
fn the_transition_cursor_moves_by_bars_and_cues_one() {
    use jodeldiplom::core::lanes::Block;
    use jodeldiplom::core::transition::{tab, viewport};
    let mut state = common::transition_state(120, 40);
    let heard = state.playing.position;
    let a = block_analysis(&state, Block::Now);
    // At first the cursor is on the bar under the playhead (83 s).
    let start = viewport::cursor(&state, Block::Now, heard).unwrap();
    assert!(a.downbeats.contains(&start));
    let bar = viewport::bar_of(&a, start);
    assert!(a.beats[start] <= heard && heard < a.beats[start] + 4.0 * 60.0 / 128.0 + 1e-9);
    op(&mut state, Op::TransitionBarForward);
    op(&mut state, Op::TransitionBarForward);
    let c = viewport::cursor(&state, Block::Now, heard).unwrap();
    assert_eq!(viewport::bar_of(&a, c), bar + 2);
    op(&mut state, Op::TransitionBeatForward);
    op(&mut state, Op::TransitionBarBack);
    assert_eq!(
        viewport::cursor(&state, Block::Now, heard).unwrap(),
        c,
        "back to its bar"
    );
    // Enter sets the cue on the cursor's bar; playback stays as it is.
    let fx = op(&mut state, Op::Accept);
    assert!(engine(&fx).is_empty(), "{fx:?}");
    assert_eq!(
        tab::block(&state, Block::Now).cue,
        Some(tab::Cue {
            beat: c,
            bars: jodeldiplom::core::transition::DEFAULT_BARS,
            chosen: false
        })
    );
    // Selecting takes whole bars, from where the cursor was.
    let heard = state.playing.position;
    op(&mut state, Op::TransitionSelectForward);
    op(&mut state, Op::TransitionSelectForward);
    let (from, to) = viewport::selection(&state, Block::Now, heard).unwrap();
    let bar_len = 4.0 * 60.0 / 128.0;
    assert!((from - a.beats[c]).abs() < 1e-9);
    assert!((to - (a.beats[c] + 3.0 * bar_len)).abs() < 1e-6, "{to}");
    // Esc clears the selection, then the cue, and never leaves the tab
    // (tab does).
    op(&mut state, Op::Back);
    assert!(viewport::selection(&state, Block::Now, heard).is_none());
    assert!(tab::block(&state, Block::Now).cue.is_some());
    op(&mut state, Op::Back);
    assert!(tab::block(&state, Block::Now).cue.is_none());
    op(&mut state, Op::Back);
    assert_eq!(state.side, Side::Transition);
    // An armed transition: esc calls it off.
    cue_both(&mut state);
    let fx = op(&mut state, Op::TransitionGo);
    assert!(
        matches!(engine(&fx)[..], [Command::Transition(_)]),
        "{fx:?}"
    );
    update(
        &mut state,
        Action::Engine(Event::Transition(jodel_playback::TransitionState::Armed)),
    );
    let fx = op(&mut state, Op::Back);
    assert_eq!(engine(&fx), vec![Command::CancelTransition]);
    assert_eq!(
        state.transition.draft.status,
        jodeldiplom::core::transition::Status::Draft
    );
    assert_eq!(state.side, Side::Transition);
    // A late word that a window started, of one called off, is not taken.
    update(
        &mut state,
        Action::Engine(Event::Transition(
            jodel_playback::TransitionState::Started {
                head: 0,
                output_frame: 9,
                origin: 0,
            },
        )),
    );
    assert_eq!(
        state.transition.draft.status,
        jodeldiplom::core::transition::Status::Draft
    );
}

#[test]
fn the_next_track_gets_a_cue_bar_and_zoom_is_shared() {
    use jodeldiplom::core::lanes::{self, Block};
    use jodeldiplom::core::transition::{tab, viewport};
    let mut state = common::transition_state(120, 40);
    let heard = state.playing.position;
    op(&mut state, Op::Down);
    assert_eq!(state.transition.focus, Block::Next);
    let a = block_analysis(&state, Block::Next);
    // The next track starts at its first bar.
    assert_eq!(
        viewport::cursor(&state, Block::Next, heard),
        Some(a.downbeats[0])
    );
    op(&mut state, Op::TransitionBarForward);
    assert!(op(&mut state, Op::Accept).is_empty());
    assert_eq!(
        tab::block(&state, Block::Next)
            .cue
            .map(|c| (c.beat, c.bars)),
        Some((a.downbeats[1], jodeldiplom::core::transition::DEFAULT_BARS))
    );
    // 16 bars of the playing track (128 BPM) by default; zooming halves
    // and doubles it, out to whole tracks.
    let bar = 4.0 * 60.0 / 128.0;
    assert!((viewport::span(&state, lanes::Block::Now) - 16.0 * bar).abs() < 1e-6);
    op(&mut state, Op::TransitionZoomIn);
    assert!((viewport::span(&state, lanes::Block::Now) - 8.0 * bar).abs() < 1e-6);
    for _ in 0..6 {
        op(&mut state, Op::TransitionZoomOut);
    }
    assert!(viewport::whole(&state, lanes::Block::Now));
    assert_eq!(viewport::window(&state, Block::Next, heard), (0.0, 252.0));
    // A new next track starts afresh.
    state.queue.paths.swap(2, 3);
    assert_eq!(tab::block(&state, Block::Next).cue, None);
}

#[test]
fn grid_fixes_change_the_bars_and_are_kept() {
    use jodeldiplom::core::lanes::{self, Block};
    let mut state = common::transition_state(120, 40);
    let path = state.queue.paths[1].clone();
    let before = block_analysis(&state, Block::Now);
    let fx = op(&mut state, Op::GridLater);
    let after = block_analysis(&state, Block::Now);
    assert_eq!(after.downbeats[0], before.downbeats[0] + 1);
    assert!(lanes::rhythmed(&state, Block::Now).unwrap().fixed());
    assert!(
        matches!(&fx[..], [Effect::SaveGrid(p, g)] if *p == path && g.downbeats == after.downbeats)
    );
    // The one at the cursor, a nudge, half and double tempo.
    op(&mut state, Op::GridOne);
    let one = block_analysis(&state, Block::Now);
    assert!(one.downbeats.windows(2).all(|w| w[1] - w[0] == 4));
    op(&mut state, Op::GridNudgeForward);
    let nudged = block_analysis(&state, Block::Now);
    assert!((nudged.beats[0] - one.beats[0] - 0.005).abs() < 1e-9);
    op(&mut state, Op::GridDouble);
    let double = block_analysis(&state, Block::Now);
    assert!((double.summary.unwrap().bpm - 256.0).abs() < 1.0);
    op(&mut state, Op::GridHalf);
    assert!((block_analysis(&state, Block::Now).summary.unwrap().bpm - 128.0).abs() < 1.0);
    // Undo: the analysis as found, and the kept grid forgotten.
    let fx = op(&mut state, Op::GridReset);
    assert_eq!(fx, vec![Effect::ResetGrid(path)]);
    assert!(!lanes::rhythmed(&state, Block::Now).unwrap().fixed());
    assert_eq!(
        block_analysis(&state, Block::Now).downbeats,
        before.downbeats
    );
    // Off the Transition tab the grid keys say where they belong (the
    // engine told no stems are wanted there any more).
    state.side = Side::Browser;
    let fx = op(&mut state, Op::GridLater);
    let stems = |e: &Effect| matches!(e, Effect::Engine(Command::MakeStems(_)));
    assert!(fx.iter().all(stems), "{fx:?}");
    assert!(state.message.is_some());
}

#[test]
fn a_click_on_a_lane_puts_the_cursor_and_a_double_click_cues_there() {
    use jodeldiplom::core::lanes::Block;
    use jodeldiplom::core::layout::{self, Layout};
    use jodeldiplom::core::mouse::{self, MouseButton, MouseInput, MouseKind};
    use jodeldiplom::core::transition::{tab, viewport};
    let mut state = common::transition_state(120, 40);
    let Layout::Main(m) = layout::compute(&state) else {
        panic!("the main screen");
    };
    let now = &m.transition.unwrap().blocks[0];
    let lane = now.lanes[1].2;
    let press = MouseInput {
        kind: MouseKind::Down(MouseButton::Left),
        col: lane.x + lane.width / 2,
        row: lane.y,
    };
    let action = mouse::action(&state, press).unwrap();
    let Action::TransitionPoint {
        block,
        time,
        extend,
    } = action.clone()
    else {
        panic!("{action:?}");
    };
    assert_eq!((block, extend), (Block::Now, false));
    let (start, end) = viewport::window(&state, Block::Now, state.playing.position);
    assert!(time > start && time < end);
    assert!(update(&mut state, action.clone()).is_empty());
    let a = block_analysis(&state, Block::Now);
    let beat = viewport::cursor(&state, Block::Now, state.playing.position).unwrap();
    assert!(a.beats[beat] <= time && time - a.beats[beat] < 60.0 / 128.0);
    // The same beat again soon after cues its bar, as enter does; playback
    // stays as it is.
    let fx = update(&mut state, action);
    assert!(engine(&fx).is_empty(), "{fx:?}");
    let cue = tab::block(&state, Block::Now).cue.unwrap();
    assert!(a.downbeats.contains(&cue.beat) && cue.beat <= beat);
    // The wheel over the lanes zooms that block (the tab's zoom stays).
    let zoom = state.transition.zoom;
    let wheel = MouseInput {
        kind: MouseKind::ScrollUp,
        ..press
    };
    let action = mouse::action(&state, wheel).unwrap();
    update(&mut state, action);
    assert_eq!(viewport::zoom_of(&state, Block::Now), zoom - 1);
    assert_eq!(state.transition.zoom, zoom);
}

#[test]
fn a_tempo_fix_keeps_the_cursor_where_it_was_in_time() {
    use jodeldiplom::core::lanes::Block;
    use jodeldiplom::core::transition::viewport;
    let mut state = common::transition_state(120, 40);
    for _ in 0..10 {
        op(&mut state, Op::TransitionBarForward);
    }
    let heard = state.heard();
    let at = |state: &State| {
        let a = block_analysis(state, Block::Now);
        a.beats[viewport::cursor(state, Block::Now, heard).unwrap()]
    };
    let before = at(&state);
    op(&mut state, Op::GridDouble);
    assert!(
        (at(&state) - before).abs() < 1e-9,
        "{} {before}",
        at(&state)
    );
    op(&mut state, Op::GridHalf);
    assert!((at(&state) - before).abs() < 0.3, "{} {before}", at(&state));
    op(&mut state, Op::GridReset);
    assert!((at(&state) - before).abs() < 0.3);
}

#[test]
fn a_paste_on_the_transition_tab_goes_to_the_browser_filter() {
    let mut state = common::transition_state(120, 40);
    update(&mut state, Action::Paste("Coda".into()));
    assert_eq!(state.side, Side::Browser);
    assert!(state
        .browser
        .filter
        .as_ref()
        .is_some_and(|f| f.editing && f.text == "Coda"));
    // Enter then accepts the filter rather than playing from a bar.
    let fx = op(&mut state, Op::Accept);
    assert!(engine(&fx).is_empty());
    assert!(fx.iter().any(|e| matches!(e, Effect::Jump { .. })));
}

/// Cues both tracks where `enter` puts them (the playing track's, its
/// cursor following the playhead, at the first bar the engine can get
/// ready for), the focus left where it was.
fn cue_both(state: &mut State) {
    use jodeldiplom::core::lanes::Block;
    let focus = state.transition.focus;
    op(state, Op::Up);
    op(state, Op::Accept);
    op(state, Op::Down);
    op(state, Op::Accept);
    if focus == Block::Now {
        op(state, Op::Up);
    }
}

#[test]
fn g_arms_a_transition_from_the_cue_into_the_cue() {
    use jodel_playback::{TempoMode, TransitionState};
    use jodeldiplom::core::lanes::Block;
    use jodeldiplom::core::transition::draft;
    use jodeldiplom::core::transition::{Preset, Status};
    let mut state = common::transition_state(120, 40);
    // Nothing armed without both cues, and the keys say so.
    assert!(engine(&op(&mut state, Op::TransitionGo)).is_empty());
    assert!(state
        .message
        .as_ref()
        .is_some_and(|m| m.text.contains("cue")));
    // The cursor's bar and the two after it cued on the playing track, two
    // bars ahead; the next track's two bars from its second.
    op(&mut state, Op::TransitionBarForward);
    op(&mut state, Op::TransitionBarForward);
    op(&mut state, Op::TransitionSelectForward);
    op(&mut state, Op::TransitionSelectForward);
    assert!(engine(&op(&mut state, Op::Accept)).is_empty());
    assert!(engine(&op(&mut state, Op::TransitionGo)).is_empty());
    op(&mut state, Op::Down);
    op(&mut state, Op::TransitionBarForward);
    op(&mut state, Op::TransitionSelectForward);
    op(&mut state, Op::Accept);
    op(&mut state, Op::Up);
    // A preset only names itself in the header until it is armed.
    let fx = op(&mut state, Op::TransitionPreset);
    assert!(engine(&fx).is_empty());
    assert_eq!(state.transition.draft.preset, Preset::DrumsFirst);
    op(&mut state, Op::TransitionTempo);
    assert_eq!(state.transition.draft.tempo, TempoMode::Stretch);
    let fx = op(&mut state, Op::TransitionGo);
    let [Effect::Engine(Command::Transition(plan))] = &fx[..] else {
        panic!("{fx:?}");
    };
    let a = block_analysis(&state, Block::Now);
    let b = block_analysis(&state, Block::Next);
    let (start, end, bars) = draft::draft_window(&state).unwrap();
    assert_eq!(bars, 3, "the longer of the two selected cues");
    assert_eq!(
        Some(start),
        draft::cue_window(&state, Block::Now).map(|w| w.0),
        "at the cue"
    );
    assert_eq!((plan.from, plan.into), (1, 2));
    assert_eq!(plan.start, start);
    assert!((plan.length - (end - start)).abs() < 1e-9);
    assert_eq!(plan.entry, b.beats[b.downbeats[1]]);
    assert_eq!(plan.tempo, TempoMode::Stretch);
    assert_eq!(plan.beats[0], (start, plan.entry));
    assert_eq!(plan.beats.len(), 3 * 4 + 1);
    assert!(a.downbeats.iter().any(|&d| a.beats[d] == plan.start));
    // The engine takes it, plays it and finishes.
    update(
        &mut state,
        Action::Engine(Event::Transition(TransitionState::Armed)),
    );
    assert_eq!(state.transition.draft.status, Status::Armed);
    update(
        &mut state,
        Action::Engine(Event::Transition(TransitionState::Started {
            head: 0,
            output_frame: 7,
            origin: 0,
        })),
    );
    assert_eq!(
        state.transition.draft.status,
        Status::Mixing(jodeldiplom::core::transition::MixStart {
            start: jodeldiplom::core::HeadFrame { head: 0, frame: 7 },
            origin: 0,
        })
    );
    update(
        &mut state,
        Action::Engine(Event::Transition(TransitionState::Finished)),
    );
    assert_eq!(state.transition.draft.status, Status::Draft);
    // Armed again, g calls it off.
    op(&mut state, Op::TransitionGo);
    update(
        &mut state,
        Action::Engine(Event::Transition(TransitionState::Armed)),
    );
    let fx = op(&mut state, Op::TransitionGo);
    assert_eq!(engine(&fx), vec![Command::CancelTransition]);
}

/// Cues of different lengths: a selected one outweighs one of the default
/// bars, else the longer one sets the window; a cue already heard is
/// refused, and an armed transition moves to a new cue.
#[test]
fn the_cues_decide_the_window_and_an_armed_one_follows_them() {
    use jodel_playback::TransitionState;
    use jodeldiplom::core::transition::draft;
    let mut state = common::transition_state(120, 40);
    op(&mut state, Op::TransitionBarForward);
    op(&mut state, Op::TransitionBarForward);
    op(&mut state, Op::TransitionSelectForward);
    op(&mut state, Op::Accept);
    op(&mut state, Op::Down);
    op(&mut state, Op::Accept);
    assert_eq!(
        draft::draft_window(&state).unwrap().2,
        2,
        "selected: 2 bars"
    );
    // Twelve bars selected on the next track: the longer.
    for _ in 0..11 {
        op(&mut state, Op::TransitionSelectForward);
    }
    op(&mut state, Op::Accept);
    assert_eq!(draft::draft_window(&state).unwrap().2, 12);
    // Armed, a new cue on the playing track moves it there.
    op(&mut state, Op::TransitionGo);
    update(
        &mut state,
        Action::Engine(Event::Transition(TransitionState::Armed)),
    );
    let armed = state.transition.draft.sent.clone().unwrap().plan;
    op(&mut state, Op::Up);
    op(&mut state, Op::TransitionBarForward);
    let fx = op(&mut state, Op::Accept);
    let [Command::Transition(moved)] = &engine(&fx)[..] else {
        panic!("{fx:?}");
    };
    assert!(moved.start > armed.start);
    assert_eq!(moved.entry, armed.entry);
    // A cue already heard: g says so and arms nothing.
    op(&mut state, Op::TransitionGo);
    later(&mut state, 60.0);
    assert!(engine(&op(&mut state, Op::TransitionGo)).is_empty());
    assert!(state
        .message
        .as_ref()
        .is_some_and(|m| m.text.contains("passed")));
}

#[test]
fn e_edits_the_fades_bar_by_bar() {
    use jodel_playback::TransitionState;
    use jodeldiplom::core::transition::{point_at, Preset};
    let mut state = common::transition_state(120, 40);
    // Not without a cue to show the fades on.
    op(&mut state, Op::TransitionEdit);
    assert!(state.transition.draft.edit.is_none());
    cue_both(&mut state);
    op(&mut state, Op::TransitionEdit);
    assert_eq!(
        keymap::lookup(&state, key(KeyCode::Left)),
        Some(Meaning::Op(Op::EditBack))
    );
    assert_eq!(
        keymap::lookup(&state, Key::ch('c')),
        Some(Meaning::Op(Op::EditCut))
    );
    // Lane 5 (the next track's drums), bar 2: off, then a cut.
    for _ in 0..5 {
        op(&mut state, Op::Down);
    }
    op(&mut state, Op::EditForward);
    op(&mut state, Op::EditForward);
    op(&mut state, Op::EditOff);
    op(&mut state, Op::EditCut);
    let lanes = state.transition.draft.lanes.clone().unwrap();
    let p = point_at(&lanes, 5, 2, 8).unwrap();
    assert_eq!((p.gain, p.cut), (0.0, true));
    assert_ne!(lanes, Preset::Fade.lanes(8));
    // Raised a quarter at a time.
    op(&mut state, Op::EditRaise);
    op(&mut state, Op::EditRaise);
    assert_eq!(
        point_at(state.transition.draft.lanes.as_ref().unwrap(), 5, 2, 8)
            .unwrap()
            .gain,
        0.5
    );
    // While armed, an edit goes to the engine at once, on the window armed.
    let fx = op(&mut state, Op::TransitionGo);
    let [Effect::Engine(Command::Transition(armed))] = &fx[..] else {
        panic!("{fx:?}");
    };
    let armed = armed.clone();
    update(
        &mut state,
        Action::Engine(Event::Transition(TransitionState::Armed)),
    );
    later(&mut state, 30.0);
    let fx = op(&mut state, Op::EditFull);
    let [Effect::Engine(Command::Transition(plan))] = &fx[..] else {
        panic!("{fx:?}");
    };
    assert_eq!((plan.start, plan.length), (armed.start, armed.length));
    assert_ne!(plan.lanes, armed.lanes);
    // Esc stops editing; the edits stay.
    op(&mut state, Op::Back);
    assert!(state.transition.draft.edit.is_none());
    assert!(state.transition.draft.lanes.is_some());
    // A preset starts afresh.
    op(&mut state, Op::TransitionPreset);
    assert!(state.transition.draft.lanes.is_none());
}

/// The track plays on `seconds`.
fn later(state: &mut State, seconds: f64) {
    let t = state.heard() + seconds;
    state.playing.position = t;
    state.drawn_heard = Some(t);
}

#[test]
fn following_the_playhead_a_transition_starts_where_the_engine_can_get_ready() {
    use jodel_playback::TransitionState;
    use jodeldiplom::core::lanes::Block;
    use jodeldiplom::core::transition::draft::{self, GO_AFTER};
    let mut state = common::transition_state(120, 40);
    let heard = state.heard();
    // No cue, no window.
    assert_eq!(draft::draft_window(&state), None);
    cue_both(&mut state);
    let (start, _, bars) = draft::draft_window(&state).unwrap();
    assert_eq!(bars, 8);
    let a = block_analysis(&state, Block::Now);
    let first = a
        .downbeats
        .iter()
        .map(|&d| a.beats[d])
        .find(|&t| t >= heard + GO_AFTER);
    assert_eq!(Some(start), first, "the first bar it can start on");
    // Armed, the window shown stays at the cue as the track plays.
    op(&mut state, Op::TransitionGo);
    update(
        &mut state,
        Action::Engine(Event::Transition(TransitionState::Armed)),
    );
    later(&mut state, 20.0);
    let (shown, _) = draft::shown_plan(&state).unwrap();
    assert_eq!(shown.start, start);
    assert_eq!(draft::draft_window(&state).unwrap().0, start);
    // While it mixes, g does not start another.
    update(
        &mut state,
        Action::Engine(Event::Transition(TransitionState::Started {
            head: 0,
            output_frame: 1,
            origin: 0,
        })),
    );
    assert!(engine(&op(&mut state, Op::TransitionGo)).is_empty());
    // Called off by the engine: the set-up window again.
    update(
        &mut state,
        Action::Engine(Event::Transition(TransitionState::Cancelled {
            reason: "a seek".into(),
        })),
    );
    assert!(state.transition.draft.sent.is_none());
    assert_eq!(
        draft::shown_plan(&state).unwrap().0.start,
        draft::draft_window(&state).unwrap().0
    );
}

/// The playing track (index 1) hands over to the next: as the engine says
/// when a transition's window ends.
fn hand_over(state: &mut State, seconds: f64) {
    let track = state.playing.track.clone().unwrap();
    update(
        state,
        Action::Engine(Event::TrackStarted(Box::new(jodel_playback::TrackStart {
            index: 2,
            head: 1,
            info: track.info.clone(),
            route: track.route.clone(),
            output: track.output,
            device: "DAC".into(),
            bit_perfect: false,
            output_frame: 0,
            from_frame: (seconds * f64::from(track.info.rate)) as u64,
        }))),
    );
}

/// A zoom of the next track's own (the wheel over it, or `z`) goes with it
/// when it starts playing: its block, following the playhead; the track
/// after it starts with the tab's zoom.
#[test]
fn the_next_track_keeps_its_zoom_when_it_starts_playing() {
    use jodeldiplom::core::lanes::Block;
    use jodeldiplom::core::transition::{tab, viewport};
    let mut state = common::transition_state(120, 60);
    let tab = viewport::zoom_of(&state, Block::Now);
    update(
        &mut state,
        Action::TransitionWheel {
            block: Block::Next,
            steps: -1,
            scroll: false,
            at: None,
        },
    );
    let own = viewport::zoom_of(&state, Block::Next);
    assert_ne!(own, tab);
    hand_over(&mut state, 10.0);
    assert_eq!(state.playing.track.as_ref().unwrap().index, 2);
    assert_eq!(viewport::zoom_of(&state, Block::Now), own, "its zoom kept");
    assert_eq!(tab::block(&state, Block::Now).start, None, "following");
    assert_eq!(viewport::zoom_of(&state, Block::Next), tab);
}
#[test]
fn the_next_track_follows_its_playhead_in_and_slides_up_when_it_takes_over() {
    use jodel_playback::TransitionState;
    use jodeldiplom::core::lanes::{self, Block};
    use jodeldiplom::core::transition::{
        draft,
        viewport::{self, SLIDE_MS},
    };
    let mut state = common::transition_state(120, 60);
    // The next track cued 16 bars in, the playing track where the engine
    // can get ready.
    op(&mut state, Op::Down);
    op(&mut state, Op::PageDown);
    op(&mut state, Op::PageDown);
    cue_both(&mut state);
    op(&mut state, Op::Up);
    let b = block_analysis(&state, Block::Next);
    let fx = op(&mut state, Op::TransitionGo);
    let [Effect::Engine(Command::Transition(plan))] = &fx[..] else {
        panic!("{fx:?}");
    };
    assert_eq!(plan.entry, b.beats[b.downbeats[16]]);
    let plan = (**plan).clone();
    let map = plan.time_map();
    update(
        &mut state,
        Action::Engine(Event::Transition(TransitionState::Armed)),
    );
    // Armed, the next track's view runs along where it will be heard,
    // under the playing track's playhead; it has no playhead yet.
    let heard = state.heard();
    let span = viewport::span(&state, lanes::Block::Now);
    assert_eq!(viewport::playhead(&state, Block::Next, heard), None);
    let (start, _) = viewport::window(&state, Block::Next, heard);
    assert!((start - (map.at(heard) - span / 3.0)).abs() < 1e-9);
    // Mixing, it has: where the map puts it.
    later(&mut state, plan.start - heard + 1.0);
    update(
        &mut state,
        Action::Engine(Event::Transition(TransitionState::Started {
            head: 0,
            output_frame: 1,
            origin: 0,
        })),
    );
    let heard = state.heard();
    assert_eq!(
        viewport::playhead(&state, Block::Next, heard),
        Some(map.at(heard))
    );
    // It takes over: the track left keeps its view and slides up out of
    // sight, the new one up into its place.
    let (left_start, _) = viewport::window(&state, Block::Now, heard);
    state.drawn_ms = Some(10_000);
    let at = map.at(plan.start + plan.length);
    hand_over(&mut state, at);
    state.drawn_heard = Some(at);
    assert_eq!(state.transition.slide, Some(10_000));
    assert_eq!(
        lanes::path(&state, Block::Prev),
        Some(&state.queue.paths[1])
    );
    assert!(
        lanes::rhythmed(&state, Block::Prev).is_some(),
        "kept while it slides"
    );
    assert_eq!(viewport::window(&state, Block::Prev, at).0, left_start);
    let blocks = |state: &State| -> Vec<(Block, u16, bool)> {
        let Layout::Main(m) = layout::compute(state) else {
            panic!()
        };
        m.transition
            .unwrap()
            .blocks
            .iter()
            .map(|b| (b.block, b.header.height, b.top_cut > 0))
            .collect()
    };
    state.drawn_ms = Some(10_000 + SLIDE_MS / 2);
    let mid = blocks(&state);
    assert_eq!(mid[0].0, Block::Prev);
    assert_eq!(mid[0].1, 0, "its header is above the view: {mid:?}");
    assert_eq!(mid[1].0, Block::Now);
    // The new playing track's fades are those it came in by.
    let (shown, _) = draft::shown_plan(&state).unwrap();
    assert_eq!(shown, plan);
    // Over: the two blocks as always, and the slide is forgotten.
    state.drawn_ms = Some(10_000 + SLIDE_MS);
    let done: Vec<Block> = blocks(&state).iter().map(|b| b.0).collect();
    assert_eq!(done, vec![Block::Now, Block::Next]);
    update(&mut state, Action::Tick(10_000 + SLIDE_MS));
    assert_eq!(state.transition.slide, None);
}

/// `N` mixes in by the Transition tab's preset, as it is set when `N` is
/// pressed (`p` there chose drums first: so does `N`), unless the settings
/// give it one of its own.
#[test]
fn n_mixes_into_the_next_track_by_the_transition_tabs_preset() {
    use jodel_playback::TransitionState;
    use jodeldiplom::core::transition::draft::GO_AFTER;
    use jodeldiplom::core::transition::Preset;
    let mut state = common::transition_state(120, 40);
    // The Transition tab's preset set to drums first (p, from fade).
    state.side = Side::Transition;
    assert_eq!(state.transition.draft.preset, Preset::Fade);
    op(&mut state, Op::TransitionPreset);
    assert_eq!(state.transition.draft.preset, Preset::DrumsFirst);
    state.side = Side::Queue;
    assert_eq!(
        keymap::lookup(&state, Key::ch('X')),
        Some(Meaning::Op(Op::SkipMix))
    );
    // By default the next track key cuts, as ever.
    let next = engine(&op(&mut state, Op::Next));
    // (Away from the Transition tab now: no stems made.)
    assert_eq!(
        next,
        vec![
            Command::Next,
            Command::MakeStems(jodel_playback::MakeStems::Never)
        ]
    );
    let fx = op(&mut state, Op::SkipMix);
    assert!(fx.contains(&Effect::SaveConfig));
    assert!(state.config.skip_mix);
    // Its own preset, set and unset again in the settings: the Transition
    // tab's.
    state.config.skip_preset = Some(Preset::BassSwap);
    let fx = op(&mut state, Op::Next);
    let [Effect::Engine(Command::Transition(plan))] = &fx[..] else {
        panic!("{fx:?}");
    };
    assert_eq!(plan.lanes, Preset::BassSwap.lanes(8));
    op(&mut state, Op::Next); // cut at once
    let mut state = common::transition_state(120, 40);
    state.side = Side::Transition;
    op(&mut state, Op::TransitionPreset);
    state.side = Side::Queue;
    state.config.skip_mix = true;
    let fx = op(&mut state, Op::Next);
    // (With the engine told that no stems are wanted off the tabs.)
    let [Effect::Engine(Command::Transition(plan)), ..] = &fx[..] else {
        panic!("{fx:?}");
    };
    assert_eq!(plan.lanes, Preset::DrumsFirst.lanes(8));
    assert_eq!((plan.from, plan.into), (1, 2));
    assert!(plan.start >= state.heard() + GO_AFTER);
    update(
        &mut state,
        Action::Engine(Event::Transition(TransitionState::Armed)),
    );
    assert_eq!(
        state.transition.draft.sent.as_ref().unwrap().preset,
        Preset::DrumsFirst
    );
    // Again while it waits: cut at once.
    assert_eq!(engine(&op(&mut state, Op::Next)), vec![Command::Next]);
    // While it mixes, the key cuts into the next track (the engine plays
    // it on alone from where it is heard).
    update(
        &mut state,
        Action::Engine(Event::Transition(TransitionState::Started {
            head: 0,
            output_frame: 1,
            origin: 0,
        })),
    );
    assert_eq!(engine(&op(&mut state, Op::Next)), vec![Command::Next]);
    assert!(state
        .message
        .as_ref()
        .is_some_and(|m| m.text.contains("Cut into the next track")));
    // The Transition tab still says drums first.
    assert_eq!(state.transition.draft.preset, Preset::DrumsFirst);
    op(&mut state, Op::SkipMix);
    assert!(!state.config.skip_mix, "back to cutting");
}

#[test]
fn the_secondary_button_drags_a_transition_view() {
    use jodeldiplom::core::lanes::{self, Block};
    use jodeldiplom::core::mouse::{action, MouseButton, MouseInput, MouseKind};
    use jodeldiplom::core::transition::viewport;
    let mut state = common::transition_state(120, 40);
    let Layout::Main(m) = layout::compute(&state) else {
        panic!()
    };
    let now = m.transition.unwrap().blocks[0].clone();
    let lane = now.lanes[1].2;
    let heard = state.heard();
    let (start, end) = viewport::window(&state, Block::Now, heard);
    let per = (end - start) / f64::from(lane.width);
    let at = |kind, col, row| MouseInput { kind, col, row };
    let press = |state: &mut State, input| {
        let a = action(state, input).expect("an action");
        update(state, a);
    };
    press(
        &mut state,
        at(MouseKind::Down(MouseButton::Right), lane.x + 50, lane.y),
    );
    press(
        &mut state,
        at(MouseKind::Drag(MouseButton::Right), lane.x + 40, lane.y),
    );
    let (moved, _) = viewport::window(&state, Block::Now, heard);
    assert!(
        (moved - (start + 10.0 * per)).abs() < 1e-9,
        "dragged left: later"
    );
    // It follows the pointer off the lanes too, until let go.
    press(
        &mut state,
        at(MouseKind::Drag(MouseButton::Right), lane.x + 45, 0),
    );
    press(
        &mut state,
        at(MouseKind::Up(MouseButton::Right), lane.x + 45, 0),
    );
    assert!(state.transition.pan.is_none());
    let (kept, _) = viewport::window(&state, Block::Now, heard);
    assert!((kept - (start + 5.0 * per)).abs() < 1e-9);
    // On the whole track, the span shown goes with the pointer.
    let duration = block_analysis(&state, Block::Now).duration;
    let o = now.overview;
    press(
        &mut state,
        at(MouseKind::Down(MouseButton::Right), o.x + 10, o.y),
    );
    press(
        &mut state,
        at(MouseKind::Drag(MouseButton::Right), o.x + 20, o.y),
    );
    let (whole, _) = viewport::window(&state, Block::Now, heard);
    let rhythmed = lanes::rhythmed(&state, Block::Now).unwrap();
    let length = rhythmed
        .waveform
        .as_ref()
        .map_or(0.0, |w| w.duration())
        .max(duration);
    assert!((whole - (kept + 10.0 * length / f64::from(o.width))).abs() < 1e-9);
}

/// A stem's mute lands on the first beat of the next bar at least 0.1 s
/// ahead (D6): on that frame of the player's output, worked out from the
/// last position told, on the head it was told of; nothing goes to the
/// engine. A second press before the bar takes it back there; alt+0 is on
/// the bar too; with the setting at once (or no position yet) it is at
/// once; with a transition armed, 0.1 s ahead all the same.
#[test]
fn stems_switch_on_the_next_bar_as_mutes_of_their_strips() {
    use jodeldiplom::core::desk::{DeskChange, StripKind};
    use jodeldiplom::core::lanes::Block;
    use jodeldiplom::core::stems::DESK_SOON;
    let mut state = common::transition_state(120, 40);
    assert!(state.config.stems_on_bar, "on the next bar by default");
    let index = state.playing.track.as_ref().unwrap().index;
    let rate = f64::from(state.playing.track.as_ref().unwrap().output.format.rate);
    let heard = state.heard();
    // The engine told it is heard there, at output frame 1 000 000, the
    // output going through the track at its own rate.
    let anchor = jodeldiplom::core::Anchor {
        index,
        at: jodeldiplom::core::HeadFrame {
            head: 0,
            frame: 1_000_000,
        },
        seconds: heard,
    };
    state.playing.anchor = Some(anchor);
    state.playing.speed = Some(rate);
    let a = block_analysis(&state, Block::Now);
    let (at, bar) = a
        .downbeats
        .iter()
        .enumerate()
        .map(|(k, &d)| (a.beats[d], k + 1))
        .find(|&(b, _)| b >= heard + DESK_SOON)
        .unwrap();
    let frame = (1_000_000.0 + (at - heard) * rate).round() as u64;
    let desk = |fx: &[Effect]| -> Vec<DeskChange> {
        fx.iter()
            .filter_map(|e| match e {
                Effect::Desk(c) => Some(*c),
                _ => None,
            })
            .collect()
    };
    for side in [Side::Browser, Side::Transition] {
        state.side = side;
        let fx = op(&mut state, Op::ToggleDrums);
        assert_eq!(
            desk(&fx),
            [DeskChange::MuteAt(
                StripKind::Drums,
                true,
                jodeldiplom::core::HeadFrame { head: 0, frame }
            )]
        );
        assert!(
            engine(&fx)
                .iter()
                .all(|c| matches!(c, Command::MakeStems(_))),
            "{fx:?}"
        );
        assert!(state
            .message
            .as_ref()
            .unwrap()
            .text
            .contains(&format!("from bar {bar}")));
        // Again before the bar: unmuted on that same frame, so nothing
        // changes there.
        let fx = op(&mut state, Op::ToggleDrums);
        assert_eq!(
            desk(&fx),
            [DeskChange::MuteAt(
                StripKind::Drums,
                false,
                jodeldiplom::core::HeadFrame { head: 0, frame }
            )]
        );
        assert!(!state.desk.stems_touched());
        // Untouched, but the stems still made until that bar has played.
        assert_ne!(
            jodeldiplom::core::stems::make_wanted(&state),
            jodel_playback::MakeStems::Never
        );
    }
    // Two stems off, and alt+0 brings both back on the bar.
    op(&mut state, Op::ToggleDrums);
    op(&mut state, Op::ToggleBass);
    let fx = op(&mut state, Op::AllStems);
    assert_eq!(
        desk(&fx),
        [
            DeskChange::MuteAt(
                StripKind::Drums,
                false,
                jodeldiplom::core::HeadFrame { head: 0, frame }
            ),
            DeskChange::MuteAt(
                StripKind::Bass,
                false,
                jodeldiplom::core::HeadFrame { head: 0, frame }
            ),
        ]
    );
    // Switched in the settings to at once: as the key is pressed.
    state.config.stems_on_bar = false;
    let fx = op(&mut state, Op::ToggleOther);
    assert_eq!(desk(&fx), [DeskChange::Mute(StripKind::Other, true)]);
    // No position told yet: at once too.
    state.config.stems_on_bar = true;
    state.playing.anchor = None;
    let fx = op(&mut state, Op::ToggleOther);
    assert_eq!(desk(&fx), [DeskChange::Mute(StripKind::Other, false)]);
    // No stems made before the key (away from the tabs, nothing touched),
    // but kept (the analysis has them): the same bar, 0.1 s ahead (the
    // engine starts them on the frame heard and writes again what it holds
    // with them).
    state.playing.anchor = Some(anchor);
    state.playing.speed = Some(rate);
    state.stems_sent.make = jodel_playback::MakeStems::Never;
    let first = jodeldiplom::core::stems::bar(&state).unwrap();
    assert_eq!((first.seconds, first.at.frame), (at, frame));
    // A transition armed or mixed: the desk mixes it, on the same bar.
    for status in [
        jodeldiplom::core::transition::Status::Armed,
        jodeldiplom::core::transition::Status::Mixing(jodeldiplom::core::transition::MixStart {
            start: jodeldiplom::core::HeadFrame {
                head: 0,
                frame: 900_000,
            },
            origin: 0,
        }),
    ] {
        state.transition.draft.status = status;
        let same = jodeldiplom::core::stems::bar(&state).unwrap();
        assert_eq!((same.seconds, same.at.head, same.at.frame), (at, 0, frame));
    }
    state.transition.draft.status = jodeldiplom::core::transition::Status::Draft;
    // A seek (or a jump, another track): no position to work the frame out
    // from until the next.
    update(
        &mut state,
        Action::Engine(jodel_playback::Event::Seeked {
            index,
            seconds: heard,
        }),
    );
    assert!(jodeldiplom::core::stems::bar(&state).is_none());
    // One position from there is not enough (the mixer may not have mixed
    // any of it yet); two are, and the output's speed through the track
    // between them (2 % fast here, a next track still gliding) times the bar.
    let position = |seconds: f64, output_frame: u64| {
        Action::Engine(jodel_playback::Event::Position(jodel_playback::Position {
            index,
            head: 0,
            seconds,
            frame: (seconds * rate) as u64,
            output_frame,
            key: 0,
        }))
    };
    update(&mut state, position(heard, 2_000_000));
    assert!(jodeldiplom::core::stems::bar(&state).is_none());
    let fast = 1.02 * rate;
    let next = 2_000_000 + (0.1 * fast).round() as u64;
    update(&mut state, position(heard + 0.1, next));
    state.drawn_heard = Some(heard + 0.1);
    let bar = jodeldiplom::core::stems::bar(&state).expect("a bar");
    let want = next as f64 + (bar.seconds - (heard + 0.1)) * (next - 2_000_000) as f64 / 0.1;
    assert!(
        (bar.at.frame as f64 - want).abs() <= 1.0,
        "{} {want}",
        bar.at.frame
    );
    // The setting is a row of the settings, saved.
    state.config.stems_on_bar = false;
    let (back, _) = Config::parse(&state.config.to_text()).unwrap();
    assert!(!back.stems_on_bar);
}

/// A first touch with no stems made lands on a bar the stems will meet:
/// with the track's stems kept (the queue's facts, or its analysis) or
/// there already, 0.1 s ahead; with stems the GPU must separate first, past
/// its first chunk; with neither a GPU nor kept stems (or with the track's
/// stems failed) they never come (the key says so), and nothing is waited
/// for.
#[test]
fn a_first_touch_lands_on_a_bar_its_stems_will_meet() {
    use jodel_playback::MakeStems;
    use jodeldiplom::core::desk::{DeskChange, StripKind};
    use jodeldiplom::core::lanes::Block;
    use jodeldiplom::core::stems::{bar, DESK_SOON, SEPARATE_SOON};
    use jodeldiplom::core::{Heard, Rhythmed};
    let mut state = common::transition_state(120, 40);
    state.side = Side::Browser;
    let track = state.playing.track.clone().unwrap();
    let rate = f64::from(track.output.format.rate);
    // Heard a little before a bar: DESK_SOON lands on it, SEPARATE_SOON on
    // the next.
    let a = block_analysis(&state, Block::Now);
    let bars: Vec<f64> = a.downbeats.iter().map(|&d| a.beats[d]).collect();
    let k = bars.iter().position(|&b| b > 10.0).unwrap();
    let heard = bars[k] - 0.3;
    assert!(bars[k + 1] >= heard + SEPARATE_SOON && bars[k] < heard + SEPARATE_SOON);
    state.drawn_heard = Some(heard);
    state.playing.anchor = Some(jodeldiplom::core::Anchor {
        index: track.index,
        at: jodeldiplom::core::HeadFrame {
            head: 0,
            frame: 1_000_000,
        },
        seconds: heard,
    });
    state.playing.speed = Some(rate);
    // The playing track analysed without its stems kept, none made.
    let path = track.info.path.clone();
    let Some(Rhythm::Ready(r)) = state.rhythm.get(&path).cloned() else {
        panic!("the playing track is analysed");
    };
    let mut unkept = (*r.analysis).clone();
    unkept.stems = false;
    let waveform = r.waveform.as_deref().cloned();
    state.rhythm.insert(
        path.clone(),
        Rhythm::Ready(std::sync::Arc::new(Rhythmed::new(unkept, waveform))),
    );
    state.stems_sent.make = MakeStems::Never;
    let soon = bar(&state).unwrap();
    assert_eq!(
        soon.seconds,
        bars[k + 1],
        "past the separator's first chunk"
    );
    let frame = (1_000_000.0 + (bars[k + 1] - heard) * rate).round() as u64;
    assert_eq!(soon.at.frame, frame);
    // The key itself mutes on that bar's frame.
    let fx = op(&mut state, Op::ToggleDrums);
    assert!(fx.contains(&Effect::Desk(DeskChange::MuteAt(
        StripKind::Drums,
        true,
        jodeldiplom::core::HeadFrame { head: 0, frame }
    ))));
    assert!(engine(&fx).contains(&Command::MakeStems(MakeStems::Separate)));
    op(&mut state, Op::ToggleDrums);
    // Separated, but not on the desk yet (the first chunk under way): still
    // past it.
    state.stems_sent.make = MakeStems::Separate;
    assert_eq!(bar(&state).unwrap().seconds, bars[k + 1]);
    // There on the desk: the next bar 0.1 s ahead.
    common::fill_desk(&mut state, track.output.format.rate);
    state.drawn_heard = Some(heard);
    assert_eq!(bar(&state).unwrap().seconds, bars[k]);
    assert!(bars[k] >= heard + DESK_SOON);
    // None made again (the desk still shows its last): past the first chunk.
    state.stems_sent.make = MakeStems::Never;
    assert_eq!(bar(&state).unwrap().seconds, bars[k + 1]);
    // Kept, as the work on the file told: 0.1 s ahead.
    update(
        &mut state,
        Action::Heard {
            path: path.clone(),
            part: Heard::Stems,
        },
    );
    // (The update told the engine to make them for the change still to
    // come; none made yet.)
    state.stems_sent.make = MakeStems::Never;
    assert_eq!(bar(&state).unwrap().seconds, bars[k]);
    // Not kept, and the stems of the track failed: none come.
    state.facts.get_mut(&path).unwrap().stems = Some(false);
    state.desk.parts_failed = vec![track.index];
    assert_eq!(bar(&state).unwrap().seconds, bars[k]);
    state.desk.parts_failed.clear();
    assert_eq!(bar(&state).unwrap().seconds, bars[k + 1]);
    // Not kept and no GPU: they never come either, and the key says so.
    state.gpu = Gpu::Absent;
    assert_eq!(bar(&state).unwrap().seconds, bars[k]);
    op(&mut state, Op::ToggleBass);
    let said = &state.message.as_ref().unwrap().text;
    assert!(said.contains("no GPU: kept stems only"), "{said}");
}

/// D7: the engine makes the stems while a stem's strip is touched or the
/// Mixer, Karaoke or Transition tab (or the mode) shows, from the stems kept
/// without a GPU; elsewhere none.
#[test]
fn the_stems_are_made_while_touched_or_a_tab_shows_them() {
    use jodel_playback::MakeStems;
    use jodeldiplom::core::desk::StripKind;
    use jodeldiplom::core::stems::make_wanted;
    let mut state = listed();
    state.gpu = Gpu::Present("a GPU".into());
    state.side = Side::Browser;
    assert_eq!(make_wanted(&state), MakeStems::Never);
    for side in [Side::Mixer, Side::Karaoke, Side::Transition] {
        state.side = side;
        assert_eq!(make_wanted(&state), MakeStems::Separate, "{side:?}");
    }
    state.side = Side::Browser;
    state.desk.strips[StripKind::Vocals.index()].db = -3.0;
    assert_eq!(make_wanted(&state), MakeStems::Separate);
    state.gpu = Gpu::Absent;
    assert_eq!(make_wanted(&state), MakeStems::Kept);
    state.desk.strips[StripKind::Vocals.index()].db = 0.0;
    assert_eq!(make_wanted(&state), MakeStems::Never);
    // The player's own strip touched is no reason.
    state.desk.strips[StripKind::Player.index()].mute = true;
    assert_eq!(make_wanted(&state), MakeStems::Never);
    // As effects: told once, as tabs show and go.
    let makes = |fx: &[Effect]| -> Vec<MakeStems> {
        engine(fx)
            .into_iter()
            .filter_map(|c| match c {
                Command::MakeStems(m) => Some(m),
                _ => None,
            })
            .collect()
    };
    state.gpu = Gpu::Present("a GPU".into());
    let fx = update(&mut state, Action::ShowSide(Side::Mixer));
    assert_eq!(makes(&fx), [MakeStems::Separate]);
    let fx = update(&mut state, Action::ShowSide(Side::Transition));
    assert!(makes(&fx).is_empty());
    let fx = update(&mut state, Action::ShowSide(Side::Queue));
    assert_eq!(makes(&fx), [MakeStems::Never]);
    // A new engine is told what is due.
    state.side = Side::Mixer;
    let mut fx = Vec::new();
    jodeldiplom::core::stems::engine_started(&mut state, &mut fx);
    assert_eq!(makes(&fx), [MakeStems::Separate]);
}

/// alt+↑ and alt+↓ step the device's own volume control (the app sets it
/// and answers); a track starting reads it; the signal path shows it, and
/// a device whose volume is the system's to set says so.
#[test]
fn the_device_volume_steps_reads_and_shows() {
    use jodeldiplom::core::{volume_step, DeviceVolume, VolumeChange};
    // Sixteen steps, as macOS's keys, from the nearest one.
    assert_eq!(volume_step(0.062, 1), 0.125);
    assert_eq!(volume_step(0.5, -1), 0.4375);
    assert_eq!(volume_step(1.0, 1), 1.0);
    assert_eq!(volume_step(0.0, -1), 0.0);

    let mut state = listed();
    let alt = |code| Key {
        code,
        ctrl: false,
        alt: true,
        shift: false,
    };
    for (code, op) in [(KeyCode::Up, Op::VolumeUp), (KeyCode::Down, Op::VolumeDown)] {
        assert_eq!(
            keymap::lookup(&state, alt(code)),
            Some(Meaning::Op(op)),
            "{code:?}"
        );
    }
    assert_eq!(
        op(&mut state, Op::VolumeUp),
        vec![Effect::DeviceVolume(VolumeChange::Step(1))]
    );
    assert_eq!(
        op(&mut state, Op::VolumeDown),
        vec![Effect::DeviceVolume(VolumeChange::Step(-1))]
    );

    // A track starting reads the volume.
    let track = common::now_track(44_100, state.config.settings(), 0);
    let fx = update(
        &mut state,
        Action::Engine(Event::TrackStarted(Box::new(jodel_playback::TrackStart {
            index: 0,
            head: 0,
            info: track.info.clone(),
            route: track.route.clone(),
            output: track.output.clone(),
            device: "Speakers".into(),
            bit_perfect: true,
            output_frame: 0,
            from_frame: 0,
        }))),
    );
    assert!(
        fx.contains(&Effect::DeviceVolume(VolumeChange::Read)),
        "{fx:?}"
    );

    // Read: shown in the signal path, nothing said.
    let half = DeviceVolume {
        level: 0.5,
        db: Some(-12.0),
        muted: Some(false),
    };
    update(
        &mut state,
        Action::DeviceVolume {
            result: Ok(Some(half)),
            changed: false,
        },
    );
    assert_eq!(state.device_volume, Some(half));
    assert!(state.message.is_none());
    let path: String = text::signal_path(&state)
        .unwrap()
        .iter()
        .map(|s| s.text.as_str())
        .collect();
    assert!(
        path.contains("Speakers (exclusive, volume 50 % (−12.0 dB)), bit-perfect"),
        "{path}"
    );

    // Changed by a key: said.
    let full = DeviceVolume {
        level: 1.0,
        db: Some(0.0),
        muted: Some(false),
    };
    update(
        &mut state,
        Action::DeviceVolume {
            result: Ok(Some(full)),
            changed: true,
        },
    );
    assert_eq!(state.message.as_ref().unwrap().text, "Device volume 100 %");

    // A device whose volume the system's keys set: nothing shown, said so.
    update(
        &mut state,
        Action::DeviceVolume {
            result: Ok(None),
            changed: true,
        },
    );
    assert_eq!(state.device_volume, None);
    assert_eq!(
        state.message.as_ref().unwrap().text,
        "The system's volume keys set this device's volume"
    );
    let path: String = text::signal_path(&state)
        .unwrap()
        .iter()
        .map(|s| s.text.as_str())
        .collect();
    assert!(path.contains("Speakers (exclusive), bit-perfect"), "{path}");
}

/// The playing track at 128 BPM and the next at 64: beats pair two to
/// one, the playing track's cursor moves two bars at a time (one of the
/// next's), its window is a whole number of the next track's bars, the
/// plan pairs every other playing beat with one of the next's and glides;
/// `r` sets the pairing by hand, and an armed transition takes it.
#[test]
fn beats_pair_two_to_one_where_the_next_track_is_at_half_tempo() {
    use jodel_playback::TransitionState;
    use jodeldiplom::core::lanes::Block;
    use jodeldiplom::core::transition::draft;
    use jodeldiplom::core::transition::{pairing_for, Pairing};
    assert_eq!(pairing_for(Some(128.0), Some(64.0)), Pairing::Two);
    assert_eq!(pairing_for(Some(128.0), Some(250.0)), Pairing::Half);
    assert_eq!(pairing_for(Some(128.0), Some(100.0)), Pairing::One);
    assert_eq!(pairing_for(None, Some(64.0)), Pairing::One);
    let mut state = common::transition_state(120, 40);
    common::next_at(&mut state, 64.0);
    assert_eq!(draft::pairing(&state), Pairing::Two);
    let a = block_analysis(&state, Block::Now);
    let b = block_analysis(&state, Block::Next);
    let bar = |beat: usize| a.downbeats.iter().position(|&d| d == beat).unwrap();
    // Two bars a step on the playing track.
    op(&mut state, Op::TransitionBarForward);
    let first = bar(state.transition.now.cursor.unwrap());
    op(&mut state, Op::TransitionBarForward);
    assert_eq!(bar(state.transition.now.cursor.unwrap()), first + 2);
    // A selection from there one step on cued: four bars, two of the
    // next's.
    op(&mut state, Op::TransitionSelectForward);
    op(&mut state, Op::Accept);
    op(&mut state, Op::Down);
    op(&mut state, Op::Accept);
    op(&mut state, Op::Up);
    let (_, _, bars) = draft::draft_window(&state).unwrap();
    assert_eq!(bars, 4);
    let fx = op(&mut state, Op::TransitionGo);
    let [Effect::Engine(Command::Transition(plan))] = &fx[..] else {
        panic!("{fx:?}");
    };
    assert!(plan.glide);
    let (pa, pb) = (60.0 / 128.0, 60.0 / 64.0);
    for w in plan.beats.windows(2) {
        assert!(
            (w[1].0 - w[0].0 - 2.0 * pa).abs() < 1e-9,
            "{:?}",
            plan.beats
        );
        assert!((w[1].1 - w[0].1 - pb).abs() < 1e-9, "{:?}", plan.beats);
    }
    assert_eq!(
        plan.beats.len(),
        bars * 2 + 1,
        "every other beat, and the last"
    );
    assert!(b.downbeats.iter().any(|&d| b.beats[d] == plan.entry));
    // Armed, `r` pairs them one to one by hand: sent again from the same
    // window start and entry.
    update(
        &mut state,
        Action::Engine(Event::Transition(TransitionState::Armed)),
    );
    let fx = op(&mut state, Op::TransitionPairing);
    assert_eq!(state.transition.draft.pairing, Some(Pairing::One));
    let [Effect::Engine(Command::Transition(again))] = &fx[..] else {
        panic!("{fx:?}");
    };
    assert_eq!((again.start, again.entry), (plan.start, plan.entry));
    assert!((again.beats[1].0 - again.beats[0].0 - pa).abs() < 1e-9);
    assert!((again.beats[1].1 - again.beats[0].1 - pb).abs() < 1e-9);
    assert_eq!(draft::pairing(&state), Pairing::One);
    // Round again to automatic.
    op(&mut state, Op::TransitionPairing);
    op(&mut state, Op::TransitionPairing);
    op(&mut state, Op::TransitionPairing);
    assert_eq!(state.transition.draft.pairing, None);
    assert_eq!(draft::pairing(&state), Pairing::Two);
}

/// The next track at double tempo: its cursor moves two of its bars a
/// step, and the plan pairs every other one of its beats.
#[test]
fn beats_pair_one_to_two_where_the_next_track_is_at_double_tempo() {
    use jodeldiplom::core::lanes::Block;
    use jodeldiplom::core::transition::draft;
    use jodeldiplom::core::transition::Pairing;
    let mut state = common::transition_state(120, 40);
    common::next_at(&mut state, 250.0);
    assert_eq!(draft::pairing(&state), Pairing::Half);
    let b = block_analysis(&state, Block::Next);
    let bar = |beat: usize| b.downbeats.iter().position(|&d| d == beat).unwrap();
    op(&mut state, Op::Down);
    op(&mut state, Op::TransitionBarForward);
    let first = bar(state.transition.next.cursor.unwrap());
    op(&mut state, Op::TransitionBarForward);
    assert_eq!(bar(state.transition.next.cursor.unwrap()), first + 2);
    cue_both(&mut state);
    op(&mut state, Op::Up);
    let fx = op(&mut state, Op::TransitionGo);
    let [Effect::Engine(Command::Transition(plan))] = &fx[..] else {
        panic!("{fx:?}");
    };
    let (pa, pb) = (60.0 / 128.0, 60.0 / 250.0);
    assert!((plan.beats[1].0 - plan.beats[0].0 - pa).abs() < 1e-9);
    assert!((plan.beats[1].1 - plan.beats[0].1 - 2.0 * pb).abs() < 1e-9);
}

/// In the queue: del and backspace take the selected track out, `K` and
/// `J` (or shift with ↑ ↓) move it, here at once and in the engine; a
/// queue the engine sends from before those changes is not taken; a row
/// dragged with the mouse moves under the pointer.
#[test]
fn queue_tracks_are_removed_and_moved() {
    use jodeldiplom::core::Side;
    let mut state = common::playing_state(120, 40);
    state.side = Side::Queue;
    let p = state.queue.paths.clone();
    let look = |s: &State, k: Key| keymap::lookup(s, k);
    assert_eq!(
        look(&state, Key::ch('K')),
        Some(Meaning::Op(Op::QueueMoveUp))
    );
    assert_eq!(
        look(&state, Key::ch('J')),
        Some(Meaning::Op(Op::QueueMoveDown))
    );
    assert_eq!(
        look(&state, Key::shift(KeyCode::Up)),
        Some(Meaning::Op(Op::QueueMoveUp))
    );
    assert_eq!(
        look(&state, key(KeyCode::Delete)),
        Some(Meaning::Op(Op::QueueRemove))
    );
    assert_eq!(
        look(&state, key(KeyCode::Backspace)),
        Some(Meaning::Op(Op::QueueRemove))
    );
    assert_eq!(look(&state, Key::ch('k')), Some(Meaning::Type('k')));
    // Row 3 up twice: before the playing track (row 1), which moves down.
    update(&mut state, Action::SelectRow(3));
    let fx = op(&mut state, Op::QueueMoveUp);
    assert_eq!(engine(&fx), vec![Command::Move { from: 3, to: 2 }]);
    op(&mut state, Op::QueueMoveUp);
    assert_eq!(state.queue.cursor, 1);
    assert_eq!(
        state.queue.paths,
        [
            p[0].clone(),
            p[3].clone(),
            p[1].clone(),
            p[2].clone(),
            p[4].clone()
        ]
    );
    assert_eq!(state.queue.current, Some(2));
    assert_eq!(state.playing.track.as_ref().unwrap().index, 2);
    // The engine's queue after the first move only: not taken; after the
    // second: taken.
    let after_first = vec![
        p[0].clone(),
        p[1].clone(),
        p[3].clone(),
        p[2].clone(),
        p[4].clone(),
    ];
    update(
        &mut state,
        Action::Engine(Event::QueueChanged {
            queue: after_first,
            current: Some(1),
        }),
    );
    assert_eq!(state.queue.paths[1], p[3]);
    let after_second = state.queue.paths.clone();
    update(
        &mut state,
        Action::Engine(Event::QueueChanged {
            queue: after_second.clone(),
            current: Some(2),
        }),
    );
    assert_eq!(state.queue.paths, after_second);
    assert_eq!(state.queue.pending, 0);
    // The top row cannot go up.
    update(&mut state, Action::SelectRow(0));
    assert!(engine(&op(&mut state, Op::QueueMoveUp)).is_empty());
    // Removed: the row below takes its place under the cursor.
    let fx = op(&mut state, Op::QueueRemove);
    assert_eq!(engine(&fx), vec![Command::Remove(0)]);
    assert_eq!(state.queue.paths.len(), 4);
    assert_eq!(state.queue.paths[0], p[3]);
    assert_eq!(state.queue.current, Some(1));
    // Dragged from the last row to the first.
    update(&mut state, Action::ClickRow(3));
    assert!(state.queue.dragging);
    let fx = update(&mut state, Action::QueueDrag(2));
    assert_eq!(engine(&fx), vec![Command::Move { from: 3, to: 2 }]);
    let fx = update(&mut state, Action::QueueDrag(0));
    assert_eq!(engine(&fx), vec![Command::Move { from: 2, to: 0 }]);
    assert_eq!(state.queue.paths[0], p[4]);
    update(&mut state, Action::QueueDragEnd);
    assert!(!state.queue.dragging);
    // In the browser backspace is still the parent folder.
    state.side = Side::Browser;
    assert_eq!(
        look(&state, key(KeyCode::Backspace)),
        Some(Meaning::Op(Op::Parent))
    );
}

// The rules of how the controls meet a transition (INTERACTIONS.md).

/// The Transition tab with a transition armed (and, with `mixing`, under way).
fn with_transition(mixing: bool) -> State {
    use jodel_playback::TransitionState;
    let mut state = common::transition_state(120, 40);
    cue_both(&mut state);
    op(&mut state, Op::TransitionGo);
    update(
        &mut state,
        Action::Engine(Event::Transition(TransitionState::Armed)),
    );
    if mixing {
        update(
            &mut state,
            Action::Engine(Event::Transition(TransitionState::Started {
                head: 0,
                output_frame: 7,
                origin: 0,
            })),
        );
    }
    state
}

/// S1: stems switched (on a bar or at once, one or all) in every state of
/// a transition only ever say which stems to play: the transition is not
/// sent again, called off or changed.
#[test]
fn stem_toggles_never_touch_the_transition() {
    for mixing in [false, true] {
        for on_bar in [true, false] {
            let mut state = with_transition(mixing);
            state.config.stems_on_bar = on_bar;
            let before = (
                state.transition.draft.status.clone(),
                state.transition.draft.sent.clone(),
            );
            for toggle in [
                Op::ToggleDrums,
                Op::ToggleVocals,
                Op::AllStems,
                Op::ToggleBass,
            ] {
                let fx = op(&mut state, toggle);
                for command in engine(&fx) {
                    assert!(
                        matches!(command, Command::MakeStems(_)),
                        "{toggle:?} sent {command:?} (mixing {mixing}, on a bar {on_bar})"
                    );
                }
                assert_eq!(
                    (
                        state.transition.draft.status.clone(),
                        state.transition.draft.sent.clone()
                    ),
                    before,
                    "{toggle:?} (mixing {mixing}, on a bar {on_bar})"
                );
            }
        }
    }
}

/// T1, T2: during the mix the transition keys change nothing and send
/// nothing (the mix under way plays as it is); armed, they are taken.
#[test]
fn transition_keys_wait_while_mixing() {
    let mut state = with_transition(true);
    let before = state.transition.draft.clone();
    for key in [
        Op::TransitionGo,
        Op::Accept,
        Op::Back,
        Op::TransitionPreset,
        Op::TransitionTempo,
        Op::TransitionPairing,
        Op::TransitionEdit,
    ] {
        let fx = op(&mut state, key);
        assert!(engine(&fx).is_empty(), "{key:?} sent {:?}", engine(&fx));
        assert_eq!(state.transition.draft, before, "{key:?}");
    }
    assert!(state
        .message
        .as_ref()
        .is_some_and(|m| m.text.contains("under way")));
    // Armed, a preset is sent again (same window and entry).
    let mut state = with_transition(false);
    let sent = state.transition.draft.sent.clone().unwrap();
    let fx = op(&mut state, Op::TransitionPreset);
    let [Command::Transition(plan)] = &engine(&fx)[..] else {
        panic!("{:?}", engine(&fx));
    };
    assert_eq!(
        (plan.start, plan.length, plan.entry),
        (sent.plan.start, sent.plan.length, sent.plan.entry)
    );
}

/// J1: no bar jumps while a transition is armed or under way.
#[test]
fn bar_jumps_wait_for_no_transition() {
    for mixing in [false, true] {
        let mut state = with_transition(mixing);
        for jump in [Op::JumpBar(0), Op::JumpBar(-1), Op::JumpBarAtEnd(1)] {
            let fx = op(&mut state, jump);
            assert!(
                !engine(&fx)
                    .iter()
                    .any(|c| matches!(c, Command::Jump { .. })),
                "{jump:?} (mixing {mixing})"
            );
        }
    }
}

/// T5: a new engine (the device switched, the old one gone) leaves no
/// transition armed or mixing behind.
#[test]
fn an_engine_restart_leaves_no_transition_behind() {
    use jodeldiplom::core::transition::Status;
    for mixing in [false, true] {
        let mut state = with_transition(mixing);
        jodeldiplom::core::transition::draft::engine_restarted(&mut state);
        assert_eq!(state.transition.draft.status, Status::Draft);
        assert!(state.transition.draft.sent.is_none());
        // A transition can be armed again, at the cues kept.
        let fx = op(&mut state, Op::TransitionGo);
        assert!(matches!(engine(&fx)[..], [Command::Transition(_)]));
    }
}

#[test]
fn a_remembered_folder_that_cannot_be_listed_gives_way_to_the_start_one() {
    let mut state = State::new("/gone/folder".into(), Config::default(), (120, 40));
    state.browser.fallback = Some("/music".into());
    let fx = update(
        &mut state,
        Action::Listed {
            dir: "/gone/folder".into(),
            result: Err("not there".into()),
        },
    );
    assert_eq!(state.browser.dir, std::path::PathBuf::from("/music"));
    assert!(fx.contains(&Effect::ListDir("/music".into())), "{fx:?}");
    // The start folder failing too stays, with its error.
    update(
        &mut state,
        Action::Listed {
            dir: "/music".into(),
            result: Err("not there either".into()),
        },
    );
    assert_eq!(state.browser.dir, std::path::PathBuf::from("/music"));
    assert!(state.browser.error.is_some());
}

#[test]
fn the_wheel_zooms_the_block_under_the_pointer_around_it() {
    use jodeldiplom::core::lanes::Block;
    use jodeldiplom::core::transition::{tab, viewport};
    let mut state = common::transition_state(120, 40);
    let heard = state.heard();
    let (start, end) = viewport::window(&state, Block::Now, heard);
    let next_before = viewport::window(&state, Block::Next, heard);
    // The pointer a quarter of the way across the playing track's lanes.
    let at = start + 0.25 * (end - start);
    update(
        &mut state,
        Action::TransitionWheel {
            block: Block::Now,
            steps: -1,
            scroll: false,
            at: Some(at),
        },
    );
    let (s2, e2) = viewport::window(&state, Block::Now, heard);
    // Half the span, the time under the pointer where it was.
    assert!(((e2 - s2) - 0.5 * (end - start)).abs() < 1e-6);
    assert!(((at - s2) / (e2 - s2) - 0.25).abs() < 1e-6);
    // The other block is as it was.
    assert_eq!(viewport::window(&state, Block::Next, heard), next_before);
    // The keys zoom both again, from the tab's zoom.
    update(&mut state, Action::Op(Op::TransitionZoomOut));
    assert_eq!(tab::block(&state, Block::Now).zoom, None);
    assert!(
        (viewport::span(&state, Block::Now) - viewport::span(&state, Block::Next)).abs() < 1e-9
    );
}

/// `z` on the Transition tab picks, for the track selected only, the widest
/// zoom whose lanes leave out no word of its lyrics; the next wider one would
/// leave some out. The other track keeps its zoom. Without lyrics it says so
/// and changes nothing.
#[test]
fn z_zooms_the_selected_track_to_fit_its_lyrics() {
    use jodeldiplom::core::lanes::{self, Block};
    use jodeldiplom::core::transition::{
        tab,
        viewport::{self, ZOOMS},
    };
    let mut state = common::transition_state(120, 40);
    let span = |state: &State, b| viewport::span(state, b);
    let (now_before, next_before) = (span(&state, Block::Now), span(&state, Block::Next));
    op(&mut state, Op::TransitionFitLyrics);
    assert_eq!(span(&state, Block::Now), now_before, "nothing to fit");
    assert!(state
        .message
        .as_ref()
        .is_some_and(|m| m.text.contains("No timed lyrics")));
    common::sung_every(&mut state, 1.2);
    // The playing track, following the playhead: it still follows.
    op(&mut state, Op::TransitionFitLyrics);
    assert_eq!(
        tab::block(&state, Block::Now).start,
        None,
        "still following"
    );
    let heard = state.heard();
    let (start, end) = viewport::window(&state, Block::Now, heard);
    assert!(((heard - start) / (end - start) - 1.0 / 3.0).abs() < 1e-6);
    op(&mut state, Op::TransitionZoomOut);
    op(&mut state, Op::TransitionZoomIn);
    // The next track selected: it alone is zoomed.
    let now_before = span(&state, Block::Now);
    op(&mut state, Op::Down);
    op(&mut state, Op::TransitionFitLyrics);
    assert_eq!(
        span(&state, Block::Now),
        now_before,
        "the playing track keeps its zoom"
    );
    let z = viewport::zoom_of(&state, Block::Next);
    let Layout::Main(m) = layout::compute(&state) else {
        panic!()
    };
    let width = m.transition.unwrap().blocks[0].lanes[0].2.width;
    let lyrics = lanes::lyrics(&state, Block::Next).unwrap();
    let words = lanes::timed_words(&lyrics);
    let left_at = |state: &State| {
        let per = f64::from(width) / span(state, Block::Next);
        lanes::words_left_out(&words, per, 1)
    };
    assert_eq!(left_at(&state), 0, "at {} bars", ZOOMS[z]);
    assert!(z + 1 < ZOOMS.len());
    // The next wider zoom, as the tab's (the playing track has none of its
    // own, so it shows that span).
    let mut wider = state.clone();
    wider.transition.zoom = z + 1;
    let wider_left = {
        let per = f64::from(width) / viewport::span(&wider, Block::Now);
        lanes::words_left_out(&words, per, 1)
    };
    assert!(wider_left > 0, "{} bars fit too", ZOOMS[z + 1]);
    assert_ne!(span(&state, Block::Next), next_before);
    assert!(state
        .message
        .as_ref()
        .is_some_and(|m| m.text.contains("next track to fit its lyrics")));
}
