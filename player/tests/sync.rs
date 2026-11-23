//! The singer's delay: synced on the Karaoke tab (`s`), nudged there, and
//! the one in force (measured, from a steady echo, from the singing, or
//! the default), moving smoothly mid-song.

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use jodel_output::input::Input;
use jodel_output::mixer::MicPort;
use jodel_output::{Clock, SimClock};
use jodel_playback::{Command, PlayState};
use jodeldiplom::app::{App, AppOptions, Output};
use jodeldiplom::core::desk::StripKind;
use jodeldiplom::core::keymap::{lookup, Key, KeyCode, Meaning, Op};
use jodeldiplom::core::lag::{self, Delay, Source, DEFAULT_LATE, SLEW};
use jodeldiplom::core::layout::{self, Layout};
use jodeldiplom::core::mouse::{self, MouseButton, MouseInput, MouseKind};
use jodeldiplom::core::sync::{self, Outcome, SyncCommand, SyncEvent, CLICKS};
use jodeldiplom::core::{update, Action, Effect, Screen, Side, State};

fn op(state: &mut State, op: Op) -> Vec<Effect> {
    update(state, Action::Op(op))
}

fn key(state: &mut State, key: Key) -> Vec<Effect> {
    match lookup(state, key) {
        Some(Meaning::Op(o)) => op(state, o),
        other => panic!("{key:?} does {other:?}"),
    }
}

fn pair(mic: &str) -> (String, String) {
    ("out".to_owned(), mic.to_owned())
}

/// The Karaoke tab shown, a track playing, on an output with a microphone
/// open (`mic`: its id; empty: none).
fn tab(mic: &str) -> State {
    let mut state = common::transition_state(120, 40);
    state.side = Side::Karaoke;
    lag::set_pair(&mut state, "out", mic);
    if !mic.is_empty() {
        state.karaoke.mic.name = Some("Headset".into());
    }
    state
}

/// The sync under way told that every click was answered `answers`
/// seconds after it, then that it ended.
fn answered(state: &mut State, answers: &[Option<f64>]) {
    update(
        state,
        Action::Sync(SyncEvent::Progress {
            played: CLICKS,
            answers: answers.to_vec(),
        }),
    );
    update(state, Action::Sync(SyncEvent::Ended));
}

#[test]
fn s_stops_the_song_and_plays_the_clicks_with_a_microphone_open() {
    let mut state = tab("headset");
    assert_eq!(state.playing.state, PlayState::Playing);
    let fx = key(&mut state, Key::ch('s'));
    assert!(fx.contains(&Effect::Engine(Command::Stop)));
    assert!(fx.contains(&Effect::Sync(SyncCommand::Start)));
    let session = state.karaoke.sync.clone().unwrap();
    assert_eq!((session.pair, session.played), (pair("headset"), 0));
    // The dots light up as the clicks play and are answered.
    let mut answers = vec![None; CLICKS];
    answers[0] = Some(0.15);
    update(
        &mut state,
        Action::Sync(SyncEvent::Progress { played: 2, answers }),
    );
    let session = state.karaoke.sync.as_ref().unwrap();
    assert_eq!((session.played, session.heard()), (2, 1));
    // Esc calls it off.
    let fx = key(&mut state, Key::plain(KeyCode::Esc));
    assert!(fx.contains(&Effect::Sync(SyncCommand::Stop)));
    assert!(state.karaoke.sync.is_none());
}

/// It is the microphone's delay that is measured: without one open (or
/// one still opening) there is no sync, and it says so; nor while the
/// clicks would not be heard.
#[test]
fn no_sync_without_a_microphone_or_where_the_clicks_are_not_heard() {
    let mut state = tab("");
    let fx = key(&mut state, Key::ch('s'));
    assert!(!fx.contains(&Effect::Sync(SyncCommand::Start)) && state.karaoke.sync.is_none());
    assert!(state
        .message
        .as_ref()
        .unwrap()
        .text
        .contains("needs a microphone"));
    // Shown, but not open in the app yet.
    state.karaoke.mic.name = Some("Headset".into());
    assert!(!key(&mut state, Key::ch('s')).contains(&Effect::Sync(SyncCommand::Start)));
    // Open, but the player's strip muted on the Mixer tab.
    let mut state = tab("headset");
    state.desk.strips[StripKind::Player.index()].mute = true;
    assert!(!key(&mut state, Key::ch('s')).contains(&Effect::Sync(SyncCommand::Start)));
    assert!(state.message.as_ref().unwrap().text.contains("muted"));
    // A stem soloed: it silences the player's own inputs, the clicks too.
    let mut state = tab("headset");
    state.desk.strips[StripKind::Vocals.index()].solo = true;
    assert!(!key(&mut state, Key::ch('s')).contains(&Effect::Sync(SyncCommand::Start)));
    assert!(state.message.as_ref().unwrap().text.contains("soloed"));
    // A stem muted leaves them alone; the player soloed with it keeps them.
    let mut state = tab("headset");
    state.desk.strips[StripKind::Drums.index()].mute = true;
    assert!(key(&mut state, Key::ch('s')).contains(&Effect::Sync(SyncCommand::Start)));
    let mut state = tab("headset");
    state.desk.strips[StripKind::Vocals.index()].solo = true;
    state.desk.strips[StripKind::Player.index()].solo = true;
    assert!(key(&mut state, Key::ch('s')).contains(&Effect::Sync(SyncCommand::Start)));
}

#[test]
fn a_delay_measured_is_kept_for_its_output_and_microphone_the_nudge_back_to_0() {
    let mut state = tab("headset");
    // A nudge first: ↓ to the delay's row, → three steps.
    key(&mut state, Key::plain(KeyCode::Down));
    for _ in 0..3 {
        key(&mut state, Key::plain(KeyCode::Right));
    }
    assert_eq!(lag::nudge_ms(&state), 30);
    assert!(std::mem::take(&mut state.karaoke.lag.changed));
    key(&mut state, Key::ch('s'));
    // The answers: 150 ms, one far off.
    let mut answers = vec![Some(0.15); CLICKS];
    answers[3] = Some(0.42);
    answers[5] = Some(0.155);
    answered(&mut state, &answers);
    let session = state.karaoke.sync.as_ref().unwrap();
    assert_eq!(session.outcome, Some(Outcome::Measured(0.15)));
    // Enter keeps it.
    key(&mut state, Key::plain(KeyCode::Enter));
    assert!(state.karaoke.sync.is_none());
    assert!(state.karaoke.lag.changed);
    assert_eq!(
        state.karaoke.lag.kept[&pair("headset")],
        Delay {
            measured_ms: Some(150),
            nudge_ms: 0
        }
    );
    assert_eq!(lag::measured(&state), Some(0.15));
    // Another microphone has none of it.
    lag::set_pair(&mut state, "out", "usb mic");
    assert_eq!(lag::measured(&state), None);
}

/// The microphone changed (or the device) while the clicks played: the
/// delay is kept for the pair it was measured with.
#[test]
fn a_delay_is_kept_for_the_pair_it_was_measured_with() {
    let mut state = tab("headset");
    key(&mut state, Key::ch('s'));
    lag::set_pair(&mut state, "out", "usb mic");
    answered(&mut state, &[Some(0.12); CLICKS]);
    key(&mut state, Key::plain(KeyCode::Enter));
    assert_eq!(
        state.karaoke.lag.kept[&pair("headset")].measured_ms,
        Some(120)
    );
    assert!(!state.karaoke.lag.kept.contains_key(&pair("usb mic")));
    assert!(state
        .message
        .as_ref()
        .unwrap()
        .text
        .contains("measured with"));
}

#[test]
fn too_few_answers_ask_for_another_go_and_leaving_the_tab_calls_it_off() {
    let mut state = tab("headset");
    key(&mut state, Key::ch('s'));
    let mut answers = vec![None; CLICKS];
    for a in answers.iter_mut().take(3) {
        *a = Some(0.1);
    }
    answered(&mut state, &answers);
    assert_eq!(
        state.karaoke.sync.as_ref().unwrap().outcome,
        Some(Outcome::TooFew(3))
    );
    // Enter keeps nothing; r goes again.
    key(&mut state, Key::plain(KeyCode::Enter));
    assert!(state.karaoke.lag.kept.is_empty());
    let fx = key(&mut state, Key::ch('r'));
    assert!(fx.contains(&Effect::Sync(SyncCommand::Start)));
    assert_eq!(state.karaoke.sync.as_ref().unwrap().outcome, None);
    // Elsewhere, it is called off.
    let fx = update(&mut state, Action::ShowSide(Side::Browser));
    assert!(fx.contains(&Effect::Sync(SyncCommand::Stop)));
    assert!(state.karaoke.sync.is_none());
    // The palette's sync goes to the tab first.
    let fx = op(&mut state, Op::KaraokeSync);
    assert_eq!((state.screen, state.side), (Screen::Main, Side::Karaoke));
    assert!(fx.contains(&Effect::Sync(SyncCommand::Start)));
}

/// While a sync shows only its panel's buttons take a click: cancel at
/// once, keep and again once it has a result.
#[test]
fn the_panels_buttons_take_clicks_and_nothing_else_does() {
    let mut state = tab("headset");
    key(&mut state, Key::ch('s'));
    let buttons = |state: &State| {
        let Layout::Main(m) = layout::compute(state) else {
            panic!()
        };
        m.karaoke.unwrap().panel_buttons
    };
    let click = |state: &State, col: u16, row: u16| {
        mouse::action(
            state,
            MouseInput {
                kind: MouseKind::Down(MouseButton::Left),
                col,
                row,
            },
        )
    };
    let now = buttons(&state);
    assert_eq!(now.iter().map(|b| b.1).collect::<Vec<_>>(), [Op::Back]);
    assert!(click(&state, 2, 2).is_none(), "a click elsewhere");
    answered(&mut state, &[Some(0.14); CLICKS]);
    let now = buttons(&state);
    assert_eq!(
        now.iter().map(|b| b.1).collect::<Vec<_>>(),
        [Op::SyncKeep, Op::SyncAgain, Op::Back]
    );
    let keep = now[0].0;
    let Some(action) = click(&state, keep.x + 1, keep.y) else {
        panic!("keep takes the click")
    };
    update(&mut state, action);
    assert_eq!(lag::measured(&state), Some(0.14));
}

#[test]
fn the_delay_in_force_is_the_first_there_is_and_says_where_it_comes_from() {
    let mut state = tab("headset");
    let lag = |s: &State| (s.karaoke.lag.late, s.karaoke.lag.source);
    // Nothing known: the default.
    lag::follow(&mut state, None, 0.0);
    assert_eq!(lag(&state), (DEFAULT_LATE, Source::Default));
    // Estimated from the singing.
    state.karaoke.lag.estimate.lag = Some(0.2);
    lag::follow(&mut state, None, 0.1);
    assert_eq!(lag(&state), (0.2, Source::Singing));
    // The echo canceller's: only once steady for 3 s.
    let mut t = 0.1;
    while t < 3.0 {
        t += 0.05;
        lag::follow(&mut state, Some((true, Some(0.08))), t);
        assert_eq!(state.karaoke.lag.source, Source::Singing, "at {t}");
    }
    lag::follow(&mut state, Some((true, Some(0.08))), 3.2);
    assert_eq!(lag(&state), (0.08, Source::Echo));
    // A jumping one is not taken.
    for (i, d) in [0.16, 0.34, 0.16, 0.34].iter().enumerate() {
        lag::follow(&mut state, Some((true, Some(*d))), 4.0 + 2.0 * i as f64);
        assert_eq!(state.karaoke.lag.source, Source::Singing);
    }
    // Measured: before any of them, the nudge on it.
    state.karaoke.lag.kept.insert(
        pair("headset"),
        Delay {
            measured_ms: Some(120),
            nudge_ms: -20,
        },
    );
    lag::follow(&mut state, None, 20.0);
    let (late, source) = lag(&state);
    assert!(
        (late - 0.1).abs() < 1e-9 && source == Source::Measured,
        "{late}"
    );
    // The tab says it: the Delay row the value in force, the sync row
    // where it comes from.
    use jodeldiplom::core::karaoke::Slider;
    assert_eq!(Slider::Delay.text(&state).trim(), "100 ms");
    assert_eq!(
        jodeldiplom::view::sync::source_text(&state),
        "measured · nudged −20 ms"
    );
}

/// Another output or microphone: the echo's wait and the estimate start
/// over (they were the other's); the same pair again keeps them.
#[test]
fn another_pair_starts_the_estimate_over() {
    let mut state = tab("headset");
    state.karaoke.lag.estimate.lag = Some(0.2);
    lag::set_pair(&mut state, "out", "headset");
    assert_eq!(state.karaoke.lag.estimate.lag, Some(0.2));
    lag::set_pair(&mut state, "out", "usb mic");
    assert_eq!(state.karaoke.lag.estimate.lag, None);
    lag::follow(&mut state, None, 0.0);
    assert_eq!(state.karaoke.lag.source, Source::Default);
}

#[test]
fn mid_song_the_delay_never_jumps_but_a_nudge_or_a_sync_does_at_once() {
    let mut state = tab("headset");
    state.screen = Screen::Karaoke;
    assert_eq!(state.playing.state, PlayState::Playing);
    lag::follow(&mut state, None, 0.0);
    assert_eq!(state.karaoke.lag.late, DEFAULT_LATE);
    // The singing says 350 ms mid-song: a frame at a time, at most the
    // slew.
    state.karaoke.lag.estimate.lag = Some(0.35);
    let mut was = state.karaoke.lag.late;
    for i in 1..=120 {
        lag::follow(&mut state, None, f64::from(i) / 60.0);
        let late = state.karaoke.lag.late;
        assert!(
            (late - was).abs() <= SLEW / 60.0 + 1e-12,
            "frame {i}: {late}"
        );
        was = late;
    }
    assert!((was - (DEFAULT_LATE + 2.0 * SLEW)).abs() < 1e-9, "{was}");
    // A nudge: at once.
    state.karaoke.focus = jodeldiplom::core::karaoke::Slider::Delay;
    state.screen = Screen::Main;
    op(&mut state, Op::SliderUp);
    state.screen = Screen::Karaoke;
    lag::follow(&mut state, None, 2.1);
    assert!((state.karaoke.lag.late - 0.36).abs() < 1e-9);
    // A sync kept: at once.
    state.karaoke.lag.kept.insert(
        pair("headset"),
        Delay {
            measured_ms: Some(140),
            nudge_ms: 0,
        },
    );
    lag::follow(&mut state, None, 2.2);
    assert!((state.karaoke.lag.late - 0.14).abs() < 1e-9);
}

/// A microphone fed in real time on a thread of its own: quiet noise, and
/// a "ta" (120 ms of loud noise) 150 ms after each click is played. That
/// thread listens too: as soon as the clicks' clock ([`Singer::listen`])
/// has played a click's frame, it puts its ta 150 ms after the moment that
/// frame was played, on its own clock (from when it started), however late
/// the clicks came. It sends 5 ms packets ([`PACKET`]) as a microphone of
/// 20 ms periods ([`PERIOD`]): the bridge holds two periods, 40 ms, so a
/// thread that wakes late now and then (a loaded runner) leaves it short
/// of none, while a packet's samples are at most 5 ms old when sent.
struct Singer {
    input: Input,
    /// The clicks' clock, once they play.
    clicks: Arc<Mutex<Option<Clock>>>,
    /// Each click as heard: when, on the singer's clock, the clock's
    /// position then, and the ta's start put for it.
    heard: Arc<Mutex<Vec<(f64, u64, f64)>>>,
    /// The loud stretches sent, packet by packet: the singer's times of the
    /// first and the last loud sample of each.
    loud: Arc<Mutex<Vec<f64>>>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// The ta's lag after its click.
const TA: f64 = 0.15;
/// The frames of a packet the singer sends (5 ms at 48 kHz).
const PACKET: usize = 240;
/// The capture period it says it has (20 ms).
const PERIOD: u32 = 960;

impl Singer {
    fn new() -> Singer {
        let (input, mut feed) = Input::feed(48_000, PERIOD);
        input.set_echo_cancel(false);
        let clicks: Arc<Mutex<Option<Clock>>> = Arc::default();
        let heard: Arc<Mutex<Vec<(f64, u64, f64)>>> = Arc::default();
        let loud: Arc<Mutex<Vec<f64>>> = Arc::default();
        let stop = Arc::new(AtomicBool::new(false));
        let start = Instant::now();
        let thread = {
            let (clicks, heard, stop, sent_loud) =
                (clicks.clone(), heard.clone(), stop.clone(), loud.clone());
            std::thread::spawn(move || {
                let mut sent = 0usize;
                let mut seed = 7u32;
                while !stop.load(Ordering::Relaxed) {
                    let t = start.elapsed();
                    feed.set_time(t);
                    let now = t.as_secs_f64();
                    // Each click played since: when its frame was (the
                    // position steps a device period at a time, the frames
                    // past it played since).
                    if let Some(clock) = clicks.lock().unwrap().as_ref() {
                        let rate = clock.format().rate;
                        let frames = sync::click_frames(rate);
                        let position = clock.position();
                        let mut heard = heard.lock().unwrap();
                        while heard.len() < CLICKS && position >= frames[heard.len()] {
                            let past = (position - frames[heard.len()]) as f64 / f64::from(rate);
                            heard.push((now, position, now - past + TA));
                        }
                    }
                    let due = (now * 48_000.0) as usize;
                    while sent + PACKET <= due {
                        // Loud sample by sample: a ta starts where it is
                        // put, not where a packet does.
                        let tas: Vec<f64> = heard.lock().unwrap().iter().map(|h| h.2).collect();
                        let mut loud_at = Vec::new();
                        let packet: Vec<f32> = (sent..sent + PACKET)
                            .map(|i| {
                                let at = i as f64 / 48_000.0;
                                let loud = tas.iter().any(|&s| at >= s && at < s + 0.12);
                                if loud {
                                    loud_at.push(at);
                                }
                                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                                let n = f64::from(seed) / f64::from(u32::MAX) - 0.5;
                                (n * if loud { 0.5 } else { 0.001 }) as f32
                            })
                            .collect();
                        if let (Some(&a), Some(&b)) = (loud_at.first(), loud_at.last()) {
                            sent_loud.lock().unwrap().extend([a, b]);
                        }
                        feed.push(&packet);
                        sent += PACKET;
                    }
                    feed.clean();
                    std::thread::sleep(Duration::from_millis(1));
                }
            })
        };
        Singer {
            input,
            clicks,
            heard,
            loud,
            stop,
            thread: Some(thread),
        }
    }

    /// Listens to the clicks on `clock`.
    fn listen(&self, clock: &Clock) {
        self.clicks
            .lock()
            .unwrap()
            .get_or_insert_with(|| clock.clone());
    }

    /// What it heard and sang, for a failure's message.
    fn told(&self) -> String {
        let loud = self.loud.lock().unwrap().clone();
        let runs = loud
            .iter()
            .fold(Vec::new(), |mut runs: Vec<(f64, f64)>, &t| {
                match runs.last_mut() {
                    Some(r) if t - r.1 < 0.03 => r.1 = t,
                    _ => runs.push((t, t)),
                }
                runs
            });
        format!(
            "each click heard (when, the position then, the ta put): {:?}; sung loud (from, \
             to): {runs:?}; the microphone {:?}",
            self.heard.lock().unwrap(),
            self.input.status()
        )
    }
}

impl Drop for Singer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// The whole player on the null output with `singer`'s microphone, the
/// Karaoke tab shown and the microphone open.
fn app_with(tree: &common::Tree, singer: &Singer) -> App {
    let mut options = AppOptions::new(tree.album.clone());
    options.output = Output::Null(SimClock::Realtime);
    options.devices = Arc::new(|| Ok(Vec::new()));
    let input = singer.input.clone();
    options.mic = Arc::new(move |_| Ok(input.clone()));
    options.mics = Arc::new(|| Ok(vec![("feed".into(), "Feed".into())]));
    let mut app = App::new(options);
    assert!(common::wait_for(&mut app, Duration::from_secs(10), |a| {
        !a.state.browser.loading
    }));
    app.dispatch(Action::ShowSide(Side::Karaoke));
    assert!(common::wait_for(&mut app, Duration::from_secs(10), |a| a
        .state
        .karaoke
        .mic
        .name
        .is_some()));
    app.heard_for_frame();
    app
}

/// Frames until the sync shows an outcome (or `give_up`); `each` runs
/// every frame with the clicks' clock once open.
fn until_outcome(app: &mut App, mut each: impl FnMut(&Clock)) -> Outcome {
    let begun = Instant::now();
    loop {
        let Some(session) = app.state.karaoke.sync.as_ref() else {
            panic!("no sync: {:?}", app.state.message);
        };
        if let Some(outcome) = session.outcome.clone() {
            return outcome;
        }
        assert!(
            begun.elapsed() < Duration::from_secs(30),
            "the sync never ended"
        );
        app.pump();
        if let Some(clock) = app.sync_clock() {
            each(&clock);
        }
        app.heard_for_frame();
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// The clicks play on the null output, and a microphone that says "ta"
/// 150 ms after each click is played (the frame the output is at when it
/// is heard, as the voice is scored) measures 150 ms, and the microphone's
/// own bridge (its two capture periods, the voice's real lateness through
/// it). After the sync, the microphone's tap is back on the song's output.
#[test]
fn the_app_plays_the_clicks_and_hears_a_ta_after_each() {
    let _guard = common::real_time();
    let tree = common::music_tree("sync-app");
    let singer = Singer::new();
    let mut app = app_with(&tree, &singer);
    // A song playing first: the sync stops it.
    app.dispatch(Action::ShowSide(Side::Browser));
    app.dispatch(Action::SelectRow(1));
    app.dispatch(Action::Op(Op::OpenReplace));
    assert!(common::wait_for(&mut app, Duration::from_secs(10), |a| a
        .state
        .playing
        .state
        == PlayState::Playing));
    // Back on the tab, the microphone opens again (in the background).
    app.dispatch(Action::ShowSide(Side::Karaoke));
    assert!(common::wait_for(&mut app, Duration::from_secs(10), |a| a
        .state
        .karaoke
        .mic
        .name
        .is_some()));
    app.dispatch(Action::Op(Op::KaraokeSync));
    let outcome = until_outcome(&mut app, |clock| singer.listen(clock));
    let status = singer.input.status();
    let bridge = f64::from(status.target) / f64::from(status.rate);
    let what = format!(
        "{outcome:?}: the answers {:?}, the bridge {bridge}; {}",
        app.state.karaoke.sync.as_ref().map(|s| &s.answers),
        singer.told()
    );
    let Outcome::Measured(d) = outcome else {
        panic!("{what}");
    };
    assert!((d - (TA + bridge)).abs() <= 0.015, "{d}: {what}");
    // Kept; then the song again: its output is the only one on the
    // Player's strip (the clicks' input gone, the player's two heads
    // there, one of them open), and the microphone's tap is on it.
    app.dispatch(Action::Op(Op::SyncKeep));
    assert!(app.state.karaoke.sync.is_none());
    assert!(app
        .state
        .karaoke
        .lag
        .kept
        .values()
        .any(|k| k.measured_ms.is_some()));
    app.dispatch(Action::Op(Op::PlayPause));
    let playing = common::wait_for(&mut app, Duration::from_secs(10), |a| {
        a.state.playing.state == PlayState::Playing
    });
    assert!(
        playing,
        "{:?} {:?} {:?}",
        app.state.playing.state, app.state.queue.current, app.state.message
    );
    assert!(common::wait_for(&mut app, Duration::from_secs(10), |a| {
        let desk = a.desk().unwrap();
        let player = desk.strips.iter().find(|s| s.name == "Player").unwrap();
        let open = player.inputs.iter().filter(|i| i.format.is_some()).count();
        player.inputs.len() == 2 && open == 1 && desk.mic.port == MicPort::Hearing
    }));
    drop(singer);
    app.shutdown();
    let _ = std::fs::remove_dir_all(&tree.root);
}

/// The engine started again (another output, a failure) while the clicks
/// play: the sync ends, saying why, rather than listen on.
#[test]
fn the_output_changing_mid_sync_ends_it() {
    let _guard = common::real_time();
    let tree = common::music_tree("sync-restart");
    let singer = Singer::new();
    let mut app = app_with(&tree, &singer);
    app.dispatch(Action::Op(Op::KaraokeSync));
    assert!(common::wait_for(&mut app, Duration::from_secs(10), |a| a
        .sync_clock()
        .is_some()));
    // The first engine's start is the first: it failed.
    app.dispatch(Action::EngineGone(1));
    let outcome = app.state.karaoke.sync.as_ref().unwrap().outcome.clone();
    assert!(
        matches!(&outcome, Some(Outcome::Failed(why)) if why.contains("output changed")),
        "{outcome:?}"
    );
    assert!(!app.animating() || app.sync_clock().is_none());
    drop(singer);
    app.shutdown();
    let _ = std::fs::remove_dir_all(&tree.root);
}
