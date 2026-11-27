//! The looks keep time with the song as it is heard: the whole player on
//! a memory device playing on the test's time (each window frame 1/60 s
//! of the app's clock and as much played by the device), the Shader tab's
//! look read at each frame's heard time ([`App::heard_for_frame`], the
//! time the stage and the lyrics are drawn at) over a song whose beats are
//! known, through a pause and a seek.

mod common;

use std::sync::Arc;
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
/// The device frames of a window's frame.
const FRAME: u64 = RATE as u64 / 60;

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

/// The window's frames on the test's time.
struct Window {
    clock: AppClock,
    time: SimTime,
}

impl Window {
    /// A frame: its time passes, the device plays it, the app draws:
    /// where the song is heard.
    fn frame(&mut self, app: &mut App) -> f64 {
        let timebase = |a: &App| a.desk().unwrap().master.timebase;
        mixer_waits(&self.time, || timebase(app));
        self.clock.step(Duration::from_nanos(
            FRAME * 1_000_000_000 / u64::from(RATE),
        ));
        self.time.advance(FRAME).unwrap();
        mixer_waits(&self.time, || timebase(app));
        app.pump();
        app.heard_for_frame()
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
    let dir = common::temp_dir("looks-sync");
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
    options.autoplay = Some(file.clone());
    let mut app = common::fitted_app(options, 120, 36);
    let engine = memory.captures().last().expect("an engine").clone();
    let mut window = Window {
        clock,
        time: device_time(&engine),
    };
    let guard = Instant::now() + GUARD;
    while app.state.playing.state != PlayState::Playing {
        assert!(Instant::now() < guard, "playing");
        window.frame(&mut app);
    }
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
