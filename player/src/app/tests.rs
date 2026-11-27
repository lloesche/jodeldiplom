//! The app's own glue, with no GPU: the song's look follows the story
//! held for the track heard.

use super::*;
use crate::core::keymap::Op;
use crate::store::dirs::Data;
use crate::work::scenes::{drawn, look, take_apart, ByRed, SceneStore};

/// How long the app may take to hold a story or let it go.
const WAIT: Duration = Duration::from_secs(60);

/// Pumps `app` until `done`, for [`WAIT`] at most; whether it was.
fn until(app: &mut App, done: impl Fn(&App) -> bool) -> bool {
    let end = Instant::now() + WAIT;
    while Instant::now() < end && !done(app) {
        app.pump_wait(Duration::from_millis(20));
    }
    done(app)
}

/// The track heard with a story has the story's look as Karaoke's own;
/// the next, with none, has none (the theme's). The story's scenes were
/// taken apart in its preparation: the song playing's story is made ready
/// from what was kept, no model asked.
#[test]
fn the_song_s_look_follows_its_story() {
    let dir = jodel_testkit::TempDir::new("app-look");
    let music = dir.join("music");
    let (a, b) = (music.join("a.wav"), music.join("b.wav"));
    // Long enough to be heard all through the test.
    let silence = vec![0i16; 8_000 * 60];
    for song in [&a, &b] {
        durable::write_atomic(song, &jodel_dsp::wav::pcm16(8_000, 1, &silence)).unwrap();
    }
    let data = Data(dir.join("data"));
    let store = SceneStore::new(data.scenes(), data.scene_parts());
    drawn(&store, &a);
    let scenes = store.scenes_of(&a).expect("its scenes");
    take_apart(&scenes, &store, &ByRed::default(), &|| true).unwrap();
    let mut options = AppOptions::new(music);
    options.autoplay = Some(a);
    options.output = Output::Null(jodel_output::SimClock::Realtime);
    options.devices = Arc::new(|| Ok(Vec::new()));
    options.mic = Arc::new(|_| Err("no microphone".into()));
    options.mics = Arc::new(|| Ok(Vec::new()));
    options.separator = || None;
    options.data = Some(data);
    let mut app = App::new(options);
    assert!(until(&mut app, |a| a.state.karaoke.look.is_some()), "held");
    assert_eq!(app.state.karaoke.look.as_deref(), Some(&look()));
    app.dispatch(Action::Op(Op::Skip));
    let heard = |a: &App| {
        a.state
            .playing
            .track
            .as_ref()
            .is_some_and(|t| t.info.path == b)
    };
    assert!(until(&mut app, |a| heard(a) && a.state.karaoke.look.is_none()));
    app.shutdown();
}
