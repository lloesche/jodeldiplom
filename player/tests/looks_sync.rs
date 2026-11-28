//! The song's time as the frames draw it, and the looks with it, keep time
//! with the song as it is heard: the whole player on a memory device
//! playing on the test's time (each window frame 1/60 s, or 1/120 s, of
//! the app's clock and as much played by the device). The time each frame
//! is drawn at ([`App::heard_for_frame`], what the stage, the arrow and the
//! lyrics go by) moves on with the device; the Shader tab's look
//! is read at it over a song whose beats are known, through a pause and a
//! seek.

mod common;

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use jodel_looks::Driver;
use jodel_output::mixer::Timebase;
use jodel_output::{Capabilities, SampleFormat, SimClock, SimTime, StreamFormat};
use jodel_playback::{Captures, PlayState};
use jodeldiplom::app::{App, AppClock, AppOptions, MemoryOutput, Output};
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::{Action, Rhythmed, Side};
use jodeldiplom::gui::backdrop::music::Song;
use jodeldiplom::gui::backdrop::song_look::Looks;

const RATE: u32 = 48_000;
const GUARD: Duration = Duration::from_secs(60);

/// Held by each test for its whole: one player at a time. The engine tells
/// the position from its own thread, on the wall clock, and a machine
/// loaded (by the other test's player) delays a report past the frames'
/// tolerance; nothing else is shared between them.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// [`ONE_AT_A_TIME`], held (a test that failed holding it still lets the
/// next go).
fn one_at_a_time() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

/// Waits until the mixer has filled the device for its next period.
fn mixer_waits(device: &SimTime, timebase: impl Fn() -> Timebase) {
    let guard = Instant::now() + GUARD;
    while !timebase().waits_for(device) {
        assert!(Instant::now() < guard, "the mixer fills the device");
        std::thread::yield_now();
    }
}

/// The time of the device `captures` opened first.
fn device_time(captures: &Captures) -> SimTime {
    let guard = Instant::now() + GUARD;
    loop {
        if let Some(c) = captures.list().first() {
            return c.time.clone().expect("a device on the test's time");
        }
        assert!(Instant::now() < guard, "the device opened");
        std::thread::yield_now();
    }
}

/// The window's frames on the test's time, each `frame` device frames
/// long.
struct Window {
    clock: AppClock,
    time: SimTime,
    frame: u64,
}

impl Window {
    /// A frame: its time passes, the device plays it, the app draws:
    /// where the song is heard.
    fn frame(&mut self, app: &mut App) -> f64 {
        let timebase = |a: &App| a.desk().unwrap().master.timebase;
        mixer_waits(&self.time, || timebase(app));
        self.clock.step(Duration::from_nanos(
            self.frame * 1_000_000_000 / u64::from(RATE),
        ));
        self.time.advance(self.frame).unwrap();
        mixer_waits(&self.time, || timebase(app));
        app.pump();
        app.heard_for_frame()
    }
}

/// The whole player on a memory device on the test's time, a 30 s chord
/// (`tag` names its folder) playing, its window's frames `frame` device
/// frames long; what the device captures.
fn playing(tag: &str, frame: u64) -> (App, Window, Captures) {
    let dir = jodel_testkit::thread_dir(tag);
    let file = dir.join("a.wav");
    common::write_wav(&file, RATE, 30.0);
    let mut caps = Capabilities::new();
    caps.insert(StreamFormat::new(RATE, 2, SampleFormat::S24In32Low));
    let memory = MemoryOutput::new(caps, SimClock::Manual);
    let mut options = AppOptions::new(dir.clone());
    let clock = AppClock::stepped();
    options.clock = clock.clone();
    options.output = Output::Memory(memory.clone());
    options.devices = Arc::new(|| Ok(Vec::new()));
    options.mic = Arc::new(|_| Err("no microphone".into()));
    options.mics = Arc::new(|| Ok(Vec::new()));
    options.separator = || None;
    options.config.beat_voting = false;
    options.config.beat_method = jodel_rhythm::Method::Flux;
    options.config.stems = false;
    options.autoplay = Some(file);
    let mut app = common::fitted_app(options, 120, 36);
    let engine = memory.captures().last().expect("an engine").clone();
    let mut window = Window {
        clock,
        time: device_time(&engine),
        frame,
    };
    let guard = Instant::now() + GUARD;
    while app.state.playing.state != PlayState::Playing {
        assert!(Instant::now() < guard, "playing");
        window.frame(&mut app);
    }
    (app, window, engine)
}

/// The song's time as each frame draws it ([`App::heard_for_frame`], what
/// the stage, the arrow and the lyrics go by) against the song as the
/// device plays it (from its first frame of the chord, in the bytes it
/// played): past its first two seconds, at 60 and at 120 frames a second,
/// every frame moves on by a frame's time (within a quarter of one: no
/// step at the reports), and it is never two frames from where the song
/// is heard (the engine tells the position on its own thread, so a report
/// may come a frame after the one it was made in).
#[test]
fn the_song_s_time_as_drawn_moves_by_the_frame_with_the_device() {
    let _one = one_at_a_time();
    for fps in [60, 120] {
        let frame = RATE as u64 / fps;
        let (mut app, mut window, engine) = playing("song-time", frame);
        let device = |a: &App| a.desk().unwrap().master.timebase.heard;
        let drawn: Vec<(f64, u64)> = (0..5 * fps)
            .map(|_| {
                let heard = window.frame(&mut app);
                (heard, device(&app))
            })
            .collect();
        let bytes = engine.list()[0].sink.bytes();
        let frame_bytes = StreamFormat::new(RATE, 2, SampleFormat::S24In32Low).frame_bytes();
        let first = (bytes.chunks_exact(frame_bytes))
            .position(|f| f.iter().any(|&b| b != 0))
            .expect("the chord played") as u64;
        let dt = 1.0 / fps as f64;
        let settled = &drawn[2 * fps as usize..];
        for (k, pair) in settled.windows(2).enumerate() {
            let ((was, _), (heard, played)) = (pair[0], pair[1]);
            let step = heard - was;
            assert!(
                (step - dt).abs() < dt / 4.0,
                "{fps}: frame {k}: a step of {step:.4} s"
            );
            let truth = played.saturating_sub(first) as f64 / f64::from(RATE);
            let off = heard - truth;
            assert!(
                off.abs() < 2.0 * dt,
                "{fps}: frame {k}: {off:+.4} s from the song heard"
            );
        }
        app.shutdown();
    }
}

/// A frame of the look: where the song is heard, and the downbeat's and
/// the kick's pulses its passes are moved by.
#[derive(Clone, Copy, Debug)]
struct Seen {
    heard: f64,
    downbeat: f32,
    kick: f32,
}

/// The frames of `seen` in runs of the song heard on (none paused or
/// sought within them).
fn runs(seen: &[Seen]) -> Vec<&[Seen]> {
    let mut out = Vec::new();
    let mut from = 0;
    for i in 1..=seen.len() {
        let step = seen.get(i).map(|s| s.heard - seen[i - 1].heard);
        if step.is_none_or(|d| !(0.001..0.1).contains(&d)) {
            if i - from > 10 {
                out.push(&seen[from..i]);
            }
            from = i;
        }
    }
    out
}

/// Every moment of `moments` well inside `run` peaks on the frame whose
/// heard time is nearest it, give or take one frame, by `pulse`.
fn lands(run: &[Seen], moments: &[f64], pulse: impl Fn(&Seen) -> f32) -> usize {
    let (first, last) = (run[2].heard, run[run.len() - 3].heard);
    let mut landed = 0;
    for &m in moments.iter().filter(|&&m| m > first && m < last) {
        let near = (0..run.len())
            .min_by(|&a, &b| {
                (run[a].heard - m)
                    .abs()
                    .total_cmp(&(run[b].heard - m).abs())
            })
            .unwrap();
        let window = near.saturating_sub(4)..(near + 5).min(run.len());
        let peak = window
            .clone()
            .max_by(|&a, &b| pulse(&run[a]).total_cmp(&pulse(&run[b])))
            .unwrap();
        assert!(
            peak.abs_diff(near) <= 1,
            "the moment at {m:.3} s peaks at {:.3} s, the nearest frame {:.3} s",
            run[peak].heard,
            run[near].heard
        );
        landed += 1;
    }
    landed
}

/// The Shader tab's look is read at each frame's heard time, the same
/// time the stage draws at, with the song's curves at that time exactly:
/// every bar's first beat and every kick peaks on the frame whose heard
/// time is nearest it (within one frame), before a pause, after it and
/// after a seek; while paused the look holds still.
#[test]
fn the_looks_land_on_the_heard_beats_across_a_seek_and_a_pause() {
    let _one = one_at_a_time();
    let (mut app, mut window, _) = playing("looks-sync", RATE as u64 / 60);
    app.dispatch(Action::ShowSide(Side::Shader));
    // The song's beats as the look hears them: a steady grid at 120 a
    // minute, bars of four, from the song's start.
    let made = jodel_looks::demo::Song::new(30.0, 120.0);
    let rhythmed = Rhythmed::new(made.analysis.clone(), Some(made.waveform.clone()));
    let song = Song::of(&rhythmed);
    let curves =
        jodel_looks::Curves::new(&made.analysis, Some(&made.waveform), &song.sections, &[]);
    let mut looks = Looks::default();
    let mut seen = Vec::new();
    let mut look_at = |app: &App, heard: f64, seen: &mut Vec<Seen>| {
        let song_heard = looks.hear(&app.state, Some(&song), heard);
        let cue = looks
            .cue(
                &app.state,
                (None, &song_heard),
                (heard, 1.0 / 60.0),
                (320, 180),
            )
            .expect("the tab's look");
        assert_eq!(cue.t, heard, "the look at the time heard");
        let want = curves.at(heard);
        for d in [Driver::Downbeat, Driver::Kick, Driver::Bass, Driver::Chorus] {
            assert_eq!(cue.signals.get(d), want.get(d), "{d:?} at {heard}");
        }
        seen.push(Seen {
            heard,
            downbeat: cue.signals.get(Driver::Downbeat),
            kick: cue.signals.get(Driver::Kick),
        });
    };
    for _ in 0..240 {
        let heard = window.frame(&mut app);
        look_at(&app, heard, &mut seen);
    }
    // A pause: once the engine has said so, the look holds still.
    app.dispatch(Action::Op(Op::PlayPause));
    let guard = Instant::now() + GUARD;
    let mut paused = Vec::new();
    while app.state.playing.state != PlayState::Paused {
        assert!(Instant::now() < guard, "paused");
        let heard = window.frame(&mut app);
        look_at(&app, heard, &mut paused);
    }
    let from = paused.len();
    for _ in 0..30 {
        let heard = window.frame(&mut app);
        look_at(&app, heard, &mut paused);
    }
    let held = paused[from];
    assert!(
        paused[from..]
            .iter()
            .all(|s| s.heard == held.heard && s.downbeat == held.downbeat),
        "still while paused"
    );
    app.dispatch(Action::Op(Op::PlayPause));
    for _ in 0..240 {
        let heard = window.frame(&mut app);
        look_at(&app, heard, &mut seen);
    }
    // A seek forward (the engine's, on its own thread, while the frames go
    // on: the device plays only as they step it).
    let before = seen.last().unwrap().heard;
    app.dispatch(Action::Op(Op::SeekForward));
    for _ in 0..300 {
        let heard = window.frame(&mut app);
        look_at(&app, heard, &mut seen);
    }
    assert!(seen.last().unwrap().heard > before + 7.0, "sought forward");
    let runs = runs(&seen);
    // The pause's frames were looked at apart: the song heard goes on from
    // where it held, one run with what came before it; the seek starts
    // another.
    assert!(runs.len() >= 2, "before and after the seek: {}", runs.len());
    let bar_starts: Vec<f64> = (made.analysis.downbeats.iter())
        .map(|&i| made.analysis.beats[i])
        .collect();
    let (mut bars, mut kicks) = (0, 0);
    for run in &runs {
        bars += lands(run, &bar_starts, |s| s.downbeat);
        kicks += lands(run, curves.kicks(), |s| s.kick);
    }
    assert!(bars >= 4 && kicks >= 8, "{bars} bars, {kicks} kicks");
}
