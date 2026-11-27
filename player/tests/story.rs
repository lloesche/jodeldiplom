//! A song's karaoke story made the whole way on the real models, from the
//! player itself: the song played in Karaoke mode, its lyrics found and
//! timed, its story planned and drawn into its folder, and the covers'
//! worker holding it (and its look) for the backdrop.

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use jodel_output::SimClock;
use jodeldiplom::app::{App, AppOptions, Output};
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::{karaoke, Action, Screen, Side};
use jodeldiplom::store::dirs::Data;
use jodeldiplom::work::scenes::{SceneStore, Scenes};

/// How long the song may take to be got ready for karaoke (analysed, its
/// words found and timed, its cover taken apart).
const READY: Duration = Duration::from_secs(10 * 60);

/// How long its story may take then: the language model's plan and a few
/// minutes of pictures.
const WAIT: Duration = Duration::from_secs(30 * 60);

/// How often what the song waits for is said.
const SAY: Duration = Duration::from_secs(30);

/// `JODEL_STORY_DATA` names a data folder with the story's models (setup's
/// `models/story/`) and `JODEL_STORY_SONG` a song in it to play (Fast Car,
/// say); skipped where either is not there. The song plays on a null
/// output in Karaoke mode, its lyrics looked up (on LRCLIB too) and timed
/// as the player does, then its story is planned and drawn by the
/// stories' thread (kept in the data folder: a song planned and drawn
/// before is taken as it is), and the covers' worker takes the scenes up:
/// the app holds them for the backdrop, the plan's look the song's. The
/// trace (`JODELDIPLOM_TRACE`) has each model's turn and each picture's
/// time.
#[test]
#[ignore = "runs the story's models: JODEL_STORY_DATA, JODEL_STORY_SONG"]
fn a_song_played_in_karaoke_mode_gets_its_story_drawn() {
    let (Some(data), Some(song)) = (
        std::env::var_os("JODEL_STORY_DATA").map(|d| Data(PathBuf::from(d))),
        std::env::var_os("JODEL_STORY_SONG").map(PathBuf::from),
    ) else {
        eprintln!("skipped: JODEL_STORY_DATA and JODEL_STORY_SONG name a data folder and a song");
        return;
    };
    let models = data.models();
    let paths: jodeldiplom::setup::Paths = (jodeldiplom::setup::manifest_for(true).iter())
        .map(|need| (need.file.clone(), need.at(&models)))
        .collect();
    let mut options = AppOptions::new(song.parent().expect("a folder").to_owned());
    options.autoplay = Some(song.clone());
    options.output = Output::Null(SimClock::Realtime);
    options.devices = Arc::new(|| Ok(Vec::new()));
    options.mic = Arc::new(|_| Err("no microphone".into()));
    options.mics = Arc::new(|| Ok(Vec::new()));
    let mut lyrics = jodeldiplom::work::lyrics::Options::new(true);
    lyrics.cache = Some(data.lyrics());
    options.lyrics = Some(lyrics);
    options.rhythm_cache = Some(data.analyses());
    options.data = Some(data.clone());
    jodeldiplom::cli::set_models(&mut options, &paths);
    assert!(
        options.story.is_some(),
        "the story's models in {}",
        models.display()
    );
    let started = Instant::now();
    // A window's size, so the Karaoke tab shows (and wants its lyrics).
    let mut app = common::fitted_app(options, 120, 36);
    // Once the song is got ready for karaoke (its analysis, its words
    // timed, its cover; what it waits for said now and then), played from
    // its row, into the mode where it is, and paused there: still the
    // song heard when its story lands, however long that takes.
    app.dispatch(Action::ShowSide(Side::Karaoke));
    let queued = |a: &App| a.state.queue.paths.iter().position(|p| *p == song);
    assert!(common::wait_for(&mut app, READY, |a| queued(a).is_some()));
    let index = queued(&app).expect("queued");
    let mut said = Instant::now();
    let singable = common::wait_for(&mut app, READY, |a| {
        let not = karaoke::not_singable(&a.state, index);
        if said.elapsed() >= SAY {
            println!("{:.0} s: {not:?}", started.elapsed().as_secs_f64());
            said = Instant::now();
        }
        not.is_none()
    });
    assert!(singable, "{:?}", karaoke::not_singable(&app.state, index));
    println!("singable after {:.0} s", started.elapsed().as_secs_f64());
    let playing = |a: &App| a.state.playing.track.as_ref().map(|t| t.index);
    if playing(&app) != Some(index) {
        app.dispatch(Action::ShowSide(Side::Queue));
        app.dispatch(Action::ClickRow(index));
        app.dispatch(Action::ClickRow(index));
        assert!(common::wait_for(&mut app, READY, |a| playing(a) == Some(index)));
        app.dispatch(Action::ShowSide(Side::Karaoke));
    }
    app.dispatch(Action::Op(Op::KaraokeEnterHere));
    assert_eq!(app.state.screen, Screen::Karaoke);
    app.dispatch(Action::Op(Op::PlayPause));
    let drawn = common::wait_for(&mut app, WAIT, |a| {
        a.story().is_some() && a.state.karaoke.look.is_some()
    });
    let took = started.elapsed().as_secs_f64();
    assert!(drawn, "no story after {took:.0} s");
    let story = app.story().expect("held");
    let store = SceneStore::new(data.scenes(), data.scene_parts());
    let scenes = Scenes::read(&store.folder(&jodeldiplom::work::cover::song_key(&song)))
        .expect("its scenes described");
    assert_eq!(story.scenes.len(), scenes.scenes.len());
    assert_eq!(*story.look, scenes.look);
    assert_eq!(app.state.look(), Some(&scenes.look));
    println!(
        "{} scenes in {} after {took:.0} s",
        scenes.scenes.len(),
        store
            .folder(&jodeldiplom::work::cover::song_key(&song))
            .display()
    );
    app.shutdown();
}
