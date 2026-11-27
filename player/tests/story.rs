//! Songs' karaoke stories made the whole way on the real models, from the
//! player itself, in its preparation: the songs queued in the normal
//! player (never Karaoke mode), their lyrics found and timed, their
//! stories planned, drawn and their scenes taken apart into their folders
//! (or none, for good), each song with a story then singable in the game;
//! the tracks changed, every song still ready and nothing made again; the
//! first with a story played, its look the song's, its scenes made ready
//! once Karaoke mode shows them.

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use jodel_output::SimClock;
use jodeldiplom::app::{AppOptions, Output};
use jodeldiplom::core::karaoke::{NotSingable, NO_STORY};
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::{job_done, karaoke, Action, Job, Placing, Side, StoryFact};
use jodeldiplom::store::dirs::Data;
use jodeldiplom::work::scenes::{song_key, taken_apart, SceneStore, Scenes};

/// How long the songs may take to be made ready, their stories with the
/// rest: the language model's plans and a few minutes of pictures each.
const WAIT: Duration = Duration::from_secs(60 * 60);

/// How often where each song is is said.
const SAY: Duration = Duration::from_secs(30);

/// How long every song is watched staying ready after the tracks change:
/// past the story's retry (a minute), so a song planned again shows.
const HOLD: Duration = Duration::from_secs(75);

/// `JODEL_STORY_DATA` names a data folder with the story's and the
/// scenes' models (setup's `models/`) and `JODEL_STORY_SONGS` the songs to
/// queue, their paths joined by `|`; skipped where either is not there.
/// The songs are queued on the queue's side, nothing playing, and got
/// ready as the player does (their lyrics looked up on LRCLIB too, and
/// timed), each story made in the background (kept in the data folder: a
/// song made before is taken as it is): each song is told made, its
/// scenes drawn and taken apart, and singable in the game, or told it has
/// none (for good: not singable in the game, and not tried again). Then
/// another song plays, and the first with a story after it: every song
/// stays ready, nothing made again, for longer than a retry would wait;
/// its look is the song's at once, its scenes made ready only once
/// Karaoke mode is entered (the game's backdrop shows them). How long
/// each song took is printed; the trace (`JODELDIPLOM_TRACE`) has each
/// model's turn (the plan's thought and answer, their tokens and times)
/// and each picture's time.
#[test]
#[ignore = "runs the story's models: JODEL_STORY_DATA, JODEL_STORY_SONGS"]
fn queued_songs_get_their_stories_made_in_the_background() {
    let (Some(data), Some(songs)) = (
        std::env::var_os("JODEL_STORY_DATA").map(|d| Data(PathBuf::from(d))),
        std::env::var("JODEL_STORY_SONGS").ok(),
    ) else {
        eprintln!("skipped: JODEL_STORY_DATA and JODEL_STORY_SONGS name a data folder and songs");
        return;
    };
    let songs: Vec<PathBuf> = songs.split('|').map(PathBuf::from).collect();
    let models = data.models();
    let paths: jodeldiplom::setup::Paths = (jodeldiplom::setup::manifest_for(true).iter())
        .map(|need| (need.file.clone(), need.at(&models)))
        .collect();
    let folder = songs[0].parent().expect("a folder").to_owned();
    let mut options = AppOptions::new(folder);
    options.output = Output::Null(SimClock::Realtime);
    options.devices = Arc::new(|| Ok(Vec::new()));
    options.mic = Arc::new(|_| Err("no microphone".into()));
    options.mics = Arc::new(|| Ok(Vec::new()));
    let mut lyrics = jodeldiplom::work::lyrics::Options::new(true);
    lyrics.cache = Some(data.lyrics());
    lyrics.edits = Some(data.lyric_edits());
    options.lyrics = Some(lyrics);
    options.rhythm_cache = Some(data.analyses());
    options.data = Some(data.clone());
    jodeldiplom::cli::set_models(&mut options, &paths);
    assert!(
        options.story.is_some() && options.scenes.is_some(),
        "the story's and the scenes' models in {}",
        models.display()
    );
    let started = Instant::now();
    let mut app = common::fitted_app(options, 120, 36);
    // Queued as a playlist read is, after the (empty) queue: nothing plays.
    app.dispatch(Action::PlaylistRead {
        path: PathBuf::from("the songs.m3u8"),
        tracks: Ok(songs.clone()),
        how: Placing::Append,
    });
    assert!(common::wait_for(&mut app, Duration::from_secs(60), |a| {
        a.state.queue.paths == songs
    }));
    app.dispatch(Action::ShowSide(Side::Queue));
    app.dispatch(Action::Tick(app.now_ms()));
    let mut made: Vec<Option<f64>> = vec![None; songs.len()];
    let mut said = Instant::now();
    let all = common::wait_for(&mut app, WAIT, |a| {
        let now = started.elapsed().as_secs_f64();
        for (k, song) in songs.iter().enumerate() {
            if made[k].is_none() && job_done(&a.state, song, Job::Story) {
                println!("{now:.0} s: {} made", song.display());
                made[k] = Some(now);
            }
        }
        if said.elapsed() >= SAY {
            for (k, song) in songs.iter().enumerate() {
                let not = karaoke::not_singable(&a.state, k);
                println!("{now:.0} s: {}: {not:?}", song.display());
            }
            said = Instant::now();
        }
        made.iter().all(Option::is_some)
    });
    assert!(all, "made: {made:?}");
    let store = SceneStore::new(data.scenes(), data.scene_parts());
    let facts = |a: &jodeldiplom::app::App| -> Vec<Option<StoryFact>> {
        (songs.iter())
            .map(|song| a.state.facts.get(song).and_then(|f| f.story.clone()))
            .collect()
    };
    let told = facts(&app);
    for (k, song) in songs.iter().enumerate() {
        match &told[k] {
            Some(StoryFact::Made) => {
                let scenes = Scenes::read(&store.folder(&song_key(song))).expect("its scenes");
                assert_eq!(taken_apart(&scenes, &store), Ok(true));
                println!(
                    "{}: {} scenes, made after {:.0} s",
                    song.display(),
                    scenes.scenes.len(),
                    made[k].unwrap_or_default()
                );
            }
            Some(StoryFact::NoStory) => println!(
                "{}: no story, after {:.0} s",
                song.display(),
                made[k].unwrap_or_default()
            ),
            other => panic!("{}: made or none for good, not {other:?}", song.display()),
        }
    }
    let with_story: Vec<usize> = (0..songs.len())
        .filter(|&k| told[k] == Some(StoryFact::Made))
        .collect();
    let first = *with_story.first().expect("a song with a story");
    // Every song prepared (its lyrics and its story); those whose lyrics
    // the tab reads (the current song and the next few) singable in the
    // game, a song with no story not.
    app.dispatch(Action::ShowSide(Side::Karaoke));
    let sung = |a: &jodeldiplom::app::App| -> Vec<Option<NotSingable>> {
        (0..songs.len())
            .map(|k| karaoke::not_singable(&a.state, k))
            .collect()
    };
    let as_told = |a: &jodeldiplom::app::App| {
        let read = karaoke::lyrics_wanted(&a.state);
        let not = sung(a);
        (0..songs.len()).all(|k| {
            let prepared = [Job::Lyrics, Job::Story]
                .into_iter()
                .all(|job| job_done(&a.state, &songs[k], job));
            let singable = match told[k] {
                Some(StoryFact::Made) => not[k].is_none(),
                _ => not[k] == Some(NotSingable::Nothing(NO_STORY)),
            };
            prepared && (singable || !read.contains(&songs[k]))
        })
    };
    let singable = common::wait_for(&mut app, WAIT, as_told);
    assert!(singable, "{:?}", sung(&app));
    // Another song played, then the first with a story: the tracks change,
    // every song stays ready, nothing is made again.
    app.dispatch(Action::ShowSide(Side::Queue));
    let playing = |a: &jodeldiplom::app::App| a.state.playing.track.as_ref().map(|t| t.index);
    for row in [usize::from(first == 0), first] {
        app.dispatch(Action::ClickRow(row));
        app.dispatch(Action::ClickRow(row));
        assert!(common::wait_for(&mut app, Duration::from_secs(60), |a| {
            playing(a) == Some(row)
        }));
    }
    let held = common::wait_for(&mut app, HOLD, |a| {
        let ready = songs.iter().all(|s| job_done(&a.state, s, Job::Story));
        assert!(
            ready && facts(a) == told,
            "a song made again: {:?}",
            facts(a)
        );
        assert!(as_told(a), "not as told: {:?}", sung(a));
        false
    });
    assert!(!held, "watched to the end");
    println!(
        "every song still ready {:.0} s after the tracks changed",
        HOLD.as_secs_f64()
    );
    // Its look the song's at once, its scenes not made ready (no game's
    // backdrop shows); in Karaoke mode, they are.
    let scenes = Scenes::read(&store.folder(&song_key(&songs[first]))).expect("its scenes");
    assert!(common::wait_for(&mut app, Duration::from_secs(60), |a| {
        a.state.karaoke.look.as_deref() == Some(&scenes.look)
    }));
    assert!(app.story().is_none(), "no texels outside the game");
    // Entered once the tab has its lyrics again (read as they are wanted).
    app.dispatch(Action::ShowSide(Side::Karaoke));
    assert!(common::wait_for(&mut app, Duration::from_secs(60), |a| {
        karaoke::not_singable(&a.state, first).is_none()
    }));
    app.dispatch(Action::Op(Op::KaraokeEnterHere));
    let held = Instant::now();
    assert!(common::wait_for(&mut app, Duration::from_secs(60), |a| {
        a.story().is_some()
    }));
    let story = app.story().expect("held");
    assert_eq!(story.scenes.len(), scenes.scenes.len());
    println!(
        "the first song's scenes made ready {:.1} s after Karaoke mode was entered",
        held.elapsed().as_secs_f64()
    );
    app.shutdown();
}
