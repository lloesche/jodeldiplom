//! Karaoke waits for everything a song needs: a GPU, and the song fully
//! prepared (every job the queue's icons show for it done): the rules K1b
//! and K1c of `INTERACTIONS.md`.

mod common;

use common::karaoke_tab as tab;

use jodel_playback::{Command, Event, PlayState};
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::{karaoke, update, Action, Effect, Gpu, Job, Screen, Side, State};
use jodeldiplom::view::karaoke::{lines_now, picture, Tab};

fn op(state: &mut State, op: Op) -> Vec<Effect> {
    update(state, Action::Op(op))
}

/// Whether `fx` plays a song (Next's count-in).
fn plays(fx: &[Effect]) -> bool {
    fx.iter()
        .any(|e| matches!(e, Effect::Engine(Command::PlayIn { .. })))
}

/// What the status line says.
fn said(state: &State) -> String {
    (state.message.as_ref())
        .map(|m| m.text.clone())
        .unwrap_or_default()
}

/// The Karaoke tab, the queue got ready by every job the app has (the
/// analysis, the lyrics, the stories), all of it done for every song.
fn prepared_tab() -> State {
    let mut state = tab();
    state.preparing = vec![Job::Rhythm, Job::Lyrics, Job::Story];
    for path in state.queue.paths.clone() {
        state.ready.insert(path, state.preparing.clone());
    }
    state
}

/// Every way into karaoke from the tab (its button's ops, `enter`, the
/// palette's from another tab), each refused while `state` cannot sing:
/// the mode never opens and nothing plays or stops.
fn every_way_in_is_refused(state: &State) {
    for (side, o) in [
        (Side::Karaoke, Op::KaraokeEnter),
        (Side::Karaoke, Op::KaraokeEnterHere),
        (Side::Karaoke, Op::Accept),
        (Side::Karaoke, Op::Open),
        (Side::Browser, Op::KaraokeEnter),
        (Side::Queue, Op::KaraokeEnterHere),
    ] {
        let mut state = state.clone();
        state.side = side;
        let fx = op(&mut state, o);
        assert_eq!(state.screen, Screen::Main, "{o:?} from {side:?}");
        let moved = |e: &Effect| {
            matches!(
                e,
                Effect::Engine(Command::Stop | Command::PlayIn { .. } | Command::HoldAtEnd(_))
            )
        };
        assert!(!fx.iter().any(moved), "{o:?}: {fx:?}");
        assert!(said(&state).starts_with("No karaoke"), "{}", said(&state));
    }
}

/// K1b: without a GPU there is no karaoke at all, by any way in, and the
/// tab says why in one line: karaoke needs a graphics card, and why none
/// was found; while one is still looked for, it says so.
#[test]
fn no_karaoke_without_a_gpu() {
    let mut state = prepared_tab();
    state.gpu = Gpu::Absent("only Basic Render, a software renderer".into());
    let why = Tab::of(&state).enter.why;
    assert_eq!(
        why.as_deref(),
        Some("karaoke needs a graphics card: only Basic Render, a software renderer")
    );
    every_way_in_is_refused(&state);
    let mut tried = state.clone();
    op(&mut tried, Op::KaraokeEnter);
    assert_eq!(
        said(&tried),
        "No karaoke: karaoke needs a graphics card: only Basic Render, a software renderer"
    );
    // Nothing queued: still no karaoke, still said.
    let mut empty = state.clone();
    empty.queue.paths.clear();
    empty.queue.current = None;
    empty.karaoke.held = None;
    empty.playing.track = None;
    assert!(Tab::of(&empty).enter.why.is_some());
    op(&mut empty, Op::KaraokeEnter);
    assert_eq!(empty.screen, Screen::Main);
    // Still looked for at the start: the tab says so.
    state.gpu = Gpu::Looking;
    assert_eq!(
        Tab::of(&state).enter.why.as_deref(),
        Some("still looking for a graphics card (karaoke needs one)")
    );
    every_way_in_is_refused(&state);
    // Found: in.
    update(&mut state, Action::Gpu(Ok("Test GPU".into())));
    assert_eq!(Tab::of(&state).enter.why, None);
    op(&mut state, Op::KaraokeEnter);
    assert_eq!(state.screen, Screen::Karaoke);
}

/// K1b: a song with a job still to come (one at work on it, or not got
/// to yet) cannot start karaoke by any way in; the tab says what is still
/// being done for it, job by job in the words of the queue's icons, the
/// button disabled; once the last job is done it starts.
#[test]
fn no_karaoke_for_a_song_until_it_is_fully_prepared() {
    let mut state = prepared_tab();
    let song = karaoke::song_index(&state).unwrap();
    let path = state.queue.paths[song].clone();
    // Its analysis under way; its lyrics not timed yet (the speech model
    // there); its story not made yet.
    state.ready.insert(path.clone(), Vec::new());
    state.working.insert(path.clone(), vec![Job::Rhythm]);
    state.karaoke.model = karaoke::ModelState::Ready;
    let tab = Tab::of(&state);
    assert_eq!(
        tab.enter.why.as_deref(),
        Some(
            "still being prepared: its analysis, waveforms and stems: \
             listening for the key…; its lyrics, found or heard, and timed; its story"
        )
    );
    assert!(tab.to_string().contains("[disabled] Enter Karaoke Mode"));
    every_way_in_is_refused(&state);
    let mut tried = state.clone();
    op(&mut tried, Op::KaraokeEnter);
    assert!(
        said(&tried).contains(": still being prepared: its analysis"),
        "{}",
        said(&tried)
    );
    // The jobs done one by one: the tab follows; refused until the last.
    let done = |state: &mut State, job| {
        update(
            state,
            Action::Working {
                path: path.clone(),
                job,
                on: false,
            },
        );
        update(
            state,
            Action::Ready {
                path: path.clone(),
                job,
                facts: None,
            },
        )
    };
    done(&mut state, Job::Rhythm);
    done(&mut state, Job::Lyrics);
    assert_eq!(
        Tab::of(&state).enter.why.as_deref(),
        Some("still being prepared: its story")
    );
    every_way_in_is_refused(&state);
    done(&mut state, Job::Story);
    assert_eq!(Tab::of(&state).enter.why, None);
    assert!(Tab::of(&state)
        .to_string()
        .contains("[primary] Enter Karaoke Mode"));
    op(&mut state, Op::KaraokeEnter);
    assert_eq!(state.screen, Screen::Karaoke);
    // Next plays it, after its count-in.
    update(&mut state, Action::Engine(Event::State(PlayState::Stopped)));
    assert!(plays(&op(&mut state, Op::KaraokeNext)));
}

/// K1c: in the mode, the next song still being prepared is not started:
/// Next greyed, the stage says what is still to come; Next (or space,
/// or the demo) plays nothing, and once pressed the song starts with its
/// count-in as soon as its last job is done. `N` calls the wait off.
#[test]
fn in_the_mode_the_next_song_waits_until_it_is_prepared() {
    let mut state = prepared_tab();
    op(&mut state, Op::KaraokeEnter);
    assert_eq!(state.screen, Screen::Karaoke);
    // The song ends; the engine holds the next, its story not made yet.
    let next = karaoke::song_index(&state).unwrap() + 1;
    let path = state.queue.paths[next].clone();
    state
        .ready
        .insert(path.clone(), vec![Job::Rhythm, Job::Lyrics]);
    update(&mut state, Action::Engine(Event::Held { index: next }));
    update(&mut state, Action::Engine(Event::State(PlayState::Stopped)));
    assert_eq!(karaoke::song_index(&state), Some(next));
    let button = picture(&state, 0.1, 0.0).next.expect("Next shows");
    assert!(!button.ready, "Next greyed");
    let shows =
        |state: &State, words: &str| lines_now(state, 0.0).iter().any(|l| l.text.contains(words));
    assert!(shows(&state, "Being prepared: its story"));
    // The demo plays nothing; Next and space wait for it.
    assert!(!plays(&op(&mut state, Op::KaraokeDemo)));
    assert!(!plays(&op(&mut state, Op::KaraokeNext)));
    assert!(!plays(&op(&mut state, Op::PlayPause)));
    assert_eq!(state.karaoke.start_when_ready, Some(next));
    assert!(said(&state).contains("starts as soon as it is prepared"));
    assert!(shows(&state, "Starts as soon as it is prepared: its story"));
    // Anything else coming in starts nothing.
    assert!(!plays(&update(&mut state, Action::Tick(1))));
    // Its story made: it starts at once, with its count-in.
    let fx = update(
        &mut state,
        Action::Ready {
            path,
            job: Job::Story,
            facts: None,
        },
    );
    assert!(plays(&fx), "{fx:?}");
    assert!(state.karaoke.countdown.is_some());
    assert_eq!(state.karaoke.start_when_ready, None);
    // N calls a wait off: the song after, not prepared, waits for Next.
    let mut state = prepared_tab();
    op(&mut state, Op::KaraokeEnter);
    let song = karaoke::song_index(&state).unwrap();
    for path in &state.queue.paths.clone()[song + 1..] {
        state.ready.insert(path.clone(), Vec::new());
    }
    op(&mut state, Op::Skip);
    update(&mut state, Action::Engine(Event::State(PlayState::Stopped)));
    op(&mut state, Op::KaraokeNext);
    assert_eq!(state.karaoke.start_when_ready, Some(song + 1));
    op(&mut state, Op::Skip);
    assert_eq!(state.karaoke.start_when_ready, None);
    // esc too.
    op(&mut state, Op::KaraokeNext);
    assert!(state.karaoke.start_when_ready.is_some());
    op(&mut state, Op::Back);
    assert_eq!(state.karaoke.start_when_ready, None);
}

/// K1b: what cannot be done on this machine holds no song up, as the
/// queue's icons have it: lyrics that cannot be timed (no speech model),
/// lyrics heard from the voice (none found), no analysis while the tempo
/// is off, no stories where they are not made. A song with nothing
/// to sing (an instrumental) is refused at once, never waited for.
#[test]
fn what_cannot_be_done_here_holds_no_song_up() {
    use jodel_lyrics::{Lyrics, Maker, Source};
    use jodeldiplom::core::karaoke::SongLyrics;
    use jodeldiplom::work::lyrics::Answer;
    let mut state = prepared_tab();
    let song = karaoke::song_index(&state).unwrap();
    let path = state.queue.paths[song].clone();
    // No word of its lyrics told done, no stories made here.
    state.preparing = vec![Job::Rhythm, Job::Lyrics];
    state.ready.insert(path.clone(), vec![Job::Rhythm]);
    // Not timed, and no speech model to time them: sung as they are.
    state.karaoke.model = karaoke::ModelState::Missing;
    assert_eq!(karaoke::not_singable(&state, song), None);
    // Heard from the voice (no source had them), timed.
    let mut heard = Lyrics::parse_text("la la la", Source::Transcribed(Maker::new("asr", 1)));
    heard.timing.detail = jodel_lyrics::Detail::Words;
    state.karaoke.model = karaoke::ModelState::Ready;
    state.karaoke.lyrics.insert(
        path.clone(),
        SongLyrics::from_answer(Answer::Found {
            lyrics: heard.clone(),
            timed: Some(heard),
        }),
    );
    assert_eq!(karaoke::not_singable(&state, song), None);
    // The tempo off: no analysis to wait for.
    state.ready.insert(path.clone(), Vec::new());
    state.prepared.recipe = None;
    assert_eq!(karaoke::not_singable(&state, song), None);
    // An instrumental: nothing to sing, said at once, never waited for.
    let mut state = prepared_tab();
    op(&mut state, Op::KaraokeEnter);
    let next = karaoke::song_index(&state).unwrap() + 1;
    let next_path = state.queue.paths[next].clone();
    state
        .karaoke
        .lyrics
        .insert(next_path, SongLyrics::Instrumental);
    op(&mut state, Op::Skip);
    update(&mut state, Action::Engine(Event::State(PlayState::Stopped)));
    assert!(!plays(&op(&mut state, Op::KaraokeNext)));
    assert_eq!(state.karaoke.start_when_ready, None);
    assert!(said(&state).contains("instrumental"), "{}", said(&state));
}

/// Every song of the queue is wanted for its story once its lyrics are
/// settled, wherever the player is (no Karaoke mode): in the order the
/// queue is got ready (from the song playing to the end, then from the
/// start), asked for once as that changes; the same lyrics settled again
/// (as every track change has them told) ask for nothing, its story made
/// stays made; other lyrics (edited, timed anew) ask for it again, its
/// story to be made again, a story told of the old ones dropped. Lyrics
/// settled otherwise with the same fingerprint (a look-up that failed,
/// then found none) keep the story told. Without the stories' worker,
/// none is.
#[test]
fn every_queued_song_is_wanted_for_its_story_once_its_lyrics_are_settled() {
    use jodeldiplom::core::{job_done, Facts, LyricsFacts, SettledLyrics, StoryFact};
    let mut state = common::transition_state(120, 40);
    state.side = Side::Queue;
    state.preparing = vec![Job::Rhythm, Job::Lyrics, Job::Story];
    state.ready.clear();
    state.karaoke.lyrics.clear();
    let paths = state.queue.paths.clone();
    let n = paths.len();
    assert!(n >= 3, "a queue of a few songs");
    let from = state.playing.track.as_ref().unwrap().index;
    let wanted = |fx: &[Effect]| {
        fx.iter().find_map(|e| match e {
            Effect::WantStories(songs) => Some(
                (songs.iter())
                    .map(|s| (s.path.clone(), s.lyrics))
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
    };
    let settle_as = |state: &mut State, k: usize, how: LyricsFacts, print: u64| {
        update(
            state,
            Action::Ready {
                path: paths[k].clone(),
                job: Job::Lyrics,
                facts: Some(Facts::Lyrics(SettledLyrics { how, print })),
            },
        )
    };
    let settle =
        |state: &mut State, k: usize, print: u64| settle_as(state, k, LyricsFacts::Timed, print);
    let story = |state: &mut State, k: usize, lyrics: u64, fact: StoryFact| {
        update(
            state,
            Action::Ready {
                path: paths[k].clone(),
                job: Job::Story,
                facts: Some(Facts::Story { lyrics, fact }),
            },
        )
    };
    assert_eq!(wanted(&update(&mut state, Action::Tick(1))), None);
    // Settled out of order: wanted in the queue's, from the song playing.
    let (next, first) = ((from + 1) % n, (from + 2) % n);
    settle(&mut state, first, 1);
    settle(&mut state, next, 1);
    let fx = settle(&mut state, from, 1);
    let order = |ks: &[usize]| {
        (ks.iter())
            .map(|&k| (paths[k].clone(), 1))
            .collect::<Vec<_>>()
    };
    assert_eq!(wanted(&fx), Some(order(&[from, next, first])));
    assert_eq!(wanted(&update(&mut state, Action::Tick(2))), None, "once");
    // Its story made, then its lyrics settled again: wanted again, of
    // these lyrics, its story still to come; told made of the old ones
    // (they changed while it was made), it is not.
    story(&mut state, next, 1, StoryFact::Made);
    assert!(job_done(&state, &paths[next], Job::Story));
    assert_eq!(wanted(&settle(&mut state, next, 1)), None, "the same");
    assert!(job_done(&state, &paths[next], Job::Story));
    let again = wanted(&settle(&mut state, next, 2)).expect("asked again");
    assert_eq!(again[1], (paths[next].clone(), 2));
    assert!(!job_done(&state, &paths[next], Job::Story));
    story(&mut state, next, 1, StoryFact::Made);
    assert!(!job_done(&state, &paths[next], Job::Story), "of the old");
    story(&mut state, next, 2, StoryFact::Made);
    assert!(job_done(&state, &paths[next], Job::Story));
    // No lyrics: a look-up that failed, then found none, is the same
    // fingerprint (none), its story told stays.
    settle_as(&mut state, first, LyricsFacts::Failed, 0);
    story(&mut state, first, 0, StoryFact::NoStory);
    let fx = settle_as(&mut state, first, LyricsFacts::NotFound, 0);
    assert_eq!(wanted(&fx), None, "nothing asked");
    assert!(job_done(&state, &paths[first], Job::Story));
    // Without the stories' worker: none.
    let mut none = common::transition_state(120, 40);
    none.preparing = vec![Job::Rhythm, Job::Lyrics];
    for path in &paths {
        let fx = update(
            &mut none,
            Action::Ready {
                path: path.clone(),
                job: Job::Lyrics,
                facts: None,
            },
        );
        assert_eq!(wanted(&fx), None);
    }
}

/// K1b: the story is wanted for the game alone: with the game off a song
/// is sung once the rest is done, its story still being made; with it on
/// it waits for its story, the step its story is at said in the tab's why
/// and the queue's caption ("its story: drawing scene 4 of 23…"), one
/// source for every screen; a song left with no story is refused in the
/// game.
#[test]
fn the_game_waits_for_the_story_and_says_where_it_is() {
    use jodeldiplom::core::prep::{Picture, Story};
    use jodeldiplom::core::{Facts, Heard, StoryFact};
    let mut state = prepared_tab();
    let song = karaoke::song_index(&state).unwrap();
    let path = state.queue.paths[song].clone();
    state
        .ready
        .insert(path.clone(), vec![Job::Rhythm, Job::Lyrics]);
    let tell = |state: &mut State, story: Story| {
        update(
            state,
            Action::Heard {
                path: path.clone(),
                part: Heard::Story(story),
            },
        );
    };
    let working = |state: &mut State, on: bool| {
        update(
            state,
            Action::Working {
                path: path.clone(),
                job: Job::Story,
                on,
            },
        );
    };
    working(&mut state, true);
    tell(&mut state, Story::Planning);
    let why = |state: &State| Tab::of(state).enter.why;
    assert_eq!(
        why(&state).as_deref(),
        Some("still being prepared: its story: planning the story…")
    );
    tell(&mut state, Story::Drawing(Picture::Scene(4, 23)));
    assert_eq!(
        why(&state).as_deref(),
        Some("still being prepared: its story: drawing scene 4 of 23…")
    );
    let caption = jodeldiplom::view::prep::caption(&state, &path).expect("a caption");
    assert_eq!(caption.text, "drawing scene 4 of 23…");
    // The game off: sung on the blurred cover, its story still coming.
    let mut plain = state.clone();
    plain.config.karaoke_game = false;
    assert_eq!(karaoke::not_singable(&plain, song), None);
    // Between its turns: its story still to come, said alone.
    working(&mut state, false);
    assert_eq!(
        why(&state).as_deref(),
        Some("still being prepared: its story")
    );
    let ready = |state: &mut State, fact: StoryFact| {
        update(
            state,
            Action::Ready {
                path: path.clone(),
                job: Job::Story,
                facts: Some(Facts::Story { lyrics: 0, fact }),
            },
        );
    };
    // Not made now: still to come, why said; tried again.
    ready(&mut state, StoryFact::NotNow("no GPU".into()));
    assert_eq!(
        why(&state).as_deref(),
        Some("still being prepared: its story: not made now (no GPU)")
    );
    // Left with no story for good: refused in the game, said why.
    ready(&mut state, StoryFact::NoStory);
    assert_eq!(
        karaoke::not_singable(&state, song),
        Some(karaoke::NotSingable::Nothing(karaoke::NO_STORY))
    );
    let mut plain = state.clone();
    plain.config.karaoke_game = false;
    assert_eq!(karaoke::not_singable(&plain, song), None);
    // Made: sung.
    ready(&mut state, StoryFact::Made);
    assert_eq!(karaoke::not_singable(&state, song), None);
}
