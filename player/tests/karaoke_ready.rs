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

/// Whether `fx` plays a song (START's count-in).
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
/// analysis, the lyrics, the covers), all of it done for every song.
fn prepared_tab() -> State {
    let mut state = tab();
    state.preparing = vec![Job::Rhythm, Job::Lyrics, Job::Cover];
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
    // there); its cover not taken apart yet.
    state.ready.insert(path.clone(), Vec::new());
    state.working.insert(path.clone(), vec![Job::Rhythm]);
    state.karaoke.model = karaoke::ModelState::Ready;
    let tab = Tab::of(&state);
    assert_eq!(
        tab.enter.why.as_deref(),
        Some(
            "still being prepared: its analysis, waveforms and stems; \
             its lyrics, found or heard, and timed; its cover taken apart"
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
        Some("still being prepared: its cover taken apart")
    );
    every_way_in_is_refused(&state);
    done(&mut state, Job::Cover);
    assert_eq!(Tab::of(&state).enter.why, None);
    assert!(Tab::of(&state)
        .to_string()
        .contains("[primary] Enter Karaoke Mode"));
    op(&mut state, Op::KaraokeEnter);
    assert_eq!(state.screen, Screen::Karaoke);
    // START plays it, after its count-in.
    update(&mut state, Action::Engine(Event::State(PlayState::Stopped)));
    assert!(plays(&op(&mut state, Op::KaraokeStart)));
}

/// K1c: in the mode, the next song still being prepared is not started:
/// START greyed, the stage says what is still to come; START (or space,
/// or the demo) plays nothing, and once pressed the song starts with its
/// count-in as soon as its last job is done. `N` calls the wait off.
#[test]
fn in_the_mode_the_next_song_waits_until_it_is_prepared() {
    let mut state = prepared_tab();
    op(&mut state, Op::KaraokeEnter);
    assert_eq!(state.screen, Screen::Karaoke);
    // The song ends; the engine holds the next, its cover not done yet.
    let next = karaoke::song_index(&state).unwrap() + 1;
    let path = state.queue.paths[next].clone();
    state
        .ready
        .insert(path.clone(), vec![Job::Rhythm, Job::Lyrics]);
    update(&mut state, Action::Engine(Event::Held { index: next }));
    update(&mut state, Action::Engine(Event::State(PlayState::Stopped)));
    assert_eq!(karaoke::song_index(&state), Some(next));
    let start = picture(&state, 0.1, 0.0).start.expect("START shows");
    assert!(!start.ready, "START greyed");
    let shows =
        |state: &State, words: &str| lines_now(state, 0.0).iter().any(|l| l.text.contains(words));
    assert!(shows(&state, "Being prepared: its cover taken apart"));
    // The demo plays nothing; START and space wait for it.
    assert!(!plays(&op(&mut state, Op::KaraokeDemo)));
    assert!(!plays(&op(&mut state, Op::KaraokeStart)));
    assert!(!plays(&op(&mut state, Op::PlayPause)));
    assert_eq!(state.karaoke.start_when_ready, Some(next));
    assert!(said(&state).contains("starts as soon as it is prepared"));
    assert!(shows(&state, "Starts as soon as it is prepared: its cover"));
    // Anything else coming in starts nothing.
    assert!(!plays(&update(&mut state, Action::Tick(1))));
    // Its cover taken apart: it starts at once, with its count-in.
    let fx = update(
        &mut state,
        Action::Ready {
            path,
            job: Job::Cover,
            facts: None,
        },
    );
    assert!(plays(&fx), "{fx:?}");
    assert!(state.karaoke.countdown.is_some());
    assert_eq!(state.karaoke.start_when_ready, None);
    // N calls a wait off: the song after, not prepared, waits for START.
    let mut state = prepared_tab();
    op(&mut state, Op::KaraokeEnter);
    let song = karaoke::song_index(&state).unwrap();
    for path in &state.queue.paths.clone()[song + 1..] {
        state.ready.insert(path.clone(), Vec::new());
    }
    op(&mut state, Op::Next);
    update(&mut state, Action::Engine(Event::State(PlayState::Stopped)));
    op(&mut state, Op::KaraokeStart);
    assert_eq!(state.karaoke.start_when_ready, Some(song + 1));
    op(&mut state, Op::Next);
    assert_eq!(state.karaoke.start_when_ready, None);
    // esc too.
    op(&mut state, Op::KaraokeStart);
    assert!(state.karaoke.start_when_ready.is_some());
    op(&mut state, Op::Back);
    assert_eq!(state.karaoke.start_when_ready, None);
}

/// K1b: what cannot be done on this machine holds no song up, as the
/// queue's icons have it: lyrics that cannot be timed (no speech model),
/// lyrics heard from the voice (none found), no analysis while the tempo
/// is off, no covers where they are not taken apart. A song with nothing
/// to sing (an instrumental) is refused at once, never waited for.
#[test]
fn what_cannot_be_done_here_holds_no_song_up() {
    use jodel_lyrics::{Lyrics, Maker, Source};
    use jodeldiplom::core::karaoke::SongLyrics;
    use jodeldiplom::work::lyrics::Answer;
    let mut state = prepared_tab();
    let song = karaoke::song_index(&state).unwrap();
    let path = state.queue.paths[song].clone();
    // No word of its lyrics told done, the covers not taken apart here.
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
    op(&mut state, Op::Next);
    update(&mut state, Action::Engine(Event::State(PlayState::Stopped)));
    assert!(!plays(&op(&mut state, Op::KaraokeStart)));
    assert_eq!(state.karaoke.start_when_ready, None);
    assert!(said(&state).contains("instrumental"), "{}", said(&state));
}
