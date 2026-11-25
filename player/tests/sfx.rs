//! The karaoke game's sounds: the song's frames a time of it is heard on
//! ([`Playing::frame_at`]), what they are cued on (a moment at a time of
//! the singing, plus how late the voice comes) and a stem's change on a
//! bar, through the last position told and the output's speed through the
//! track; and the result's reveal heard after the song on the device kept
//! open, through the whole player on a memory device.

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use jodel_output::{Capabilities, SampleFormat, SimClock, StreamFormat};
use jodel_playback::PlayState;
use jodeldiplom::app::{App, AppOptions, MemoryOutput, Output};
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::singing::Beat;
use jodeldiplom::core::{karaoke, Action, Rhythm, Side};
use jodeldiplom::core::{Anchor, HeadFrame, Playing};
use jodeldiplom::sfx::diploma::Diploma;
use jodeldiplom::sfx::Sound;

/// Runs `app` as the window does (its events taken, a frame drawn every
/// 1/60 s, the game's sounds cued with each) until `done`, for `timeout`
/// at most; whether it came.
fn frames(app: &mut App, timeout: Duration, mut done: impl FnMut(&App) -> bool) -> bool {
    let end = Instant::now() + timeout;
    while Instant::now() < end {
        if done(app) {
            return true;
        }
        app.pump_wait(Duration::from_millis(16));
        app.heard_for_frame();
    }
    done(app)
}

/// Where `sound` (its left side) is heard in `left` (the device's left
/// channel, full scale 1) near the device frames `near` (from the first
/// to the last, within 60 ms): the frame its first `span` frames line up
/// best from, and how well (normalized correlation, 1: alone and whole),
/// against how well at its best 200 ms after (where it is not).
fn heard_near(left: &[f32], sound: Sound, near: (usize, usize), span: usize) -> (usize, f64, f64) {
    let made: Vec<f32> = sound
        .render(RATE)
        .interleaved()
        .into_iter()
        .step_by(2)
        .collect();
    let made = &made[..span];
    let made_energy: f64 = made.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let at = |from: usize| {
        let got = &left[from..from + span];
        let dot: f64 = got
            .iter()
            .zip(made)
            .map(|(&g, &m)| f64::from(g) * f64::from(m))
            .sum();
        let energy: f64 = got.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        dot / (made_energy * energy).sqrt().max(1e-12)
    };
    let reach = RATE as usize * 6 / 100;
    let best = |(first, last): (usize, usize)| {
        (first.saturating_sub(reach)..last + reach)
            .map(|f| (f, at(f)))
            .fold((0, f64::MIN), |b, x| if x.1 > b.1 { x } else { b })
    };
    let (frame, fit) = best(near);
    let after = near.1 + RATE as usize / 5;
    let (_, away) = best((after, after));
    (frame, fit, away)
}

const RATE: u32 = 48_000;

/// A frame of the device's stream (stereo, 24 bits in 32).
const FRAME_BYTES: usize = 8;

/// A song sung to its end in Karaoke mode, the game on, through the whole
/// player on a memory device playing in real time: the device is opened
/// once, as the player starts, and stays open; the song over and its
/// output closed, the result's reveal is heard on it, its paper's rustle
/// from its first beat and the examiner's pen from its last, each where
/// it was cued.
#[test]
fn the_result_is_heard_after_the_song() {
    let _guard = common::real_time();
    let dir = common::temp_dir("reveal-heard");
    let song = dir.join("a.wav");
    common::write_wav(&song, RATE, 3.0);
    durable::write_atomic(&dir.join("a.lrc"), b"la la la\nla la\n").unwrap();
    let mut caps = Capabilities::new();
    caps.insert(StreamFormat::new(RATE, 2, SampleFormat::S24In32Low));
    let memory = MemoryOutput::new(caps, SimClock::Realtime);
    let mut options = AppOptions::new(dir.clone());
    options.output = Output::Memory(memory.clone());
    options.size = (120, 36);
    options.devices = Arc::new(|| Ok(Vec::new()));
    options.mic = Arc::new(|_| Ok(jodel_output::input::Input::feed(48_000, 240).0));
    options.mics = Arc::new(|| Ok(Vec::new()));
    options.separator = || None;
    // The beats by the flux alone, of the mix: the networks' inference
    // (the beat network's, and the separator's where there is a GPU, as on
    // the macOS runners), unoptimised, would take the processors from the
    // mixer as it plays in real time on a small machine.
    options.config.beat_voting = false;
    options.config.beat_method = jodel_rhythm::Method::Flux;
    options.config.stems = false;
    options.autoplay = Some(song.clone());
    let mut app = App::new(options);
    let wait = Duration::from_secs(30);
    let playing = |a: &App| a.state.playing.state == PlayState::Playing;
    assert!(frames(&mut app, wait, playing));
    // Its frames drawn as it plays: the sounds' bank made at the device's
    // rate meanwhile, and the song analysed before it is sung.
    app.dispatch(Action::ShowSide(Side::Karaoke));
    assert!(frames(&mut app, wait, |a| karaoke::not_singable(
        &a.state, 0
    )
    .is_none()
        && matches!(a.state.rhythm.get(&song), Some(Rhythm::Ready(_)))));
    frames(&mut app, Duration::from_millis(1_500), |_| false);
    app.dispatch(Action::Op(Op::KaraokeEnter));
    app.dispatch(Action::Op(Op::KaraokeStart));
    let started = |a: &App| a.state.karaoke.countdown.is_none() && playing(a);
    assert!(frames(&mut app, wait, started), "START");
    // To its end, and the reveal after it drawn frame by frame, where the
    // mixer is noted as its first beat and its last are cued: from just
    // before the frame's sounds are cued to just after (the frames the
    // mixer went through meanwhile, however long this thread was held up).
    // A frame of the mixer's is a frame of what the device played once the
    // silence the device filled underruns with so far is added (played,
    // but none of the mixer's frames).
    let played = memory.captures().last().expect("a device").list()[0]
        .sink
        .clone();
    let mixing = |a: &App| {
        let clock = a.desk().unwrap().master.timebase;
        let device = (played.len() / FRAME_BYTES) as u64;
        (clock.frame + device.saturating_sub(clock.heard)) as usize
    };
    // The signature's beat is reached on the frame whose reveal time
    // passes its end from the frame before's, as the sounds read it.
    let (mut paper, mut signed, mut until, mut before) = (None, None, None, None);
    let deadline = Instant::now() + wait;
    while until.is_none_or(|u| Instant::now() < u) {
        assert!(Instant::now() < deadline, "the song ends");
        app.pump_wait(Duration::from_millis(16));
        let cueing = mixing(&app);
        app.heard_for_frame();
        let Some((shown, after)) = karaoke::result_now(&app.state) else {
            continue;
        };
        let cued = (cueing, mixing(&app));
        let (end, settled) = (
            shown.reveal.when(Beat::End).unwrap(),
            shown.reveal.settled(),
        );
        if paper.is_none() {
            paper = Some(cued);
            until = Some(Instant::now() + Duration::from_secs_f64(settled + 1.0));
        }
        if signed.is_none() && after >= end && before.is_some_and(|t| t < end) {
            signed = Some(cued);
        }
        before = Some(after);
    }
    let (paper, signed) = (paper.unwrap(), signed.expect("signed"));
    let desk = app.desk().unwrap();
    assert_eq!(desk.master.timebase.epoch, 1, "never opened again");
    app.shutdown();
    let streams = memory.captures().last().expect("a device").list();
    assert_eq!(streams.len(), 1, "the device opened once");
    let left: Vec<f32> = streams[0]
        .sink
        .bytes()
        .as_chunks::<4>()
        .0
        .iter()
        .step_by(2)
        .map(|b| ((i32::from_le_bytes(*b) << 8) >> 8) as f32 / 8_388_608.0)
        .collect();
    // The paper's rustle, alone before the title lands (its first 0.1 s),
    // and the pen's scratch (its first 0.3 s, a lullaby over it): each
    // heard where it was cued, and not elsewhere.
    let rustle = heard_near(&left, Sound::Diploma(Diploma::Unroll), paper, 4_000);
    assert!(rustle.1 > 0.8 && rustle.2 < 0.3, "the paper: {rustle:?}");
    let pen = heard_near(&left, Sound::Diploma(Diploma::Signature), signed, 14_400);
    assert!(
        pen.1 > 0.12 && pen.1 > 8.0 * pen.2,
        "the signature: {pen:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_time_lands_on_the_frame_it_is_heard_on() {
    let mut state = common::playing_state(120, 36);
    // None before a position is told.
    assert_eq!(state.playing.frame_at(84.0), None);
    let index = state.playing.track.as_ref().unwrap().index;
    state.playing.anchor = Some(Anchor {
        index,
        at: HeadFrame {
            head: 1,
            frame: 1_000_000,
        },
        seconds: 83.0,
    });
    // Nor with one: the output's speed through the track is not known
    // before two.
    assert_eq!(state.playing.frame_at(84.0), None);
    // At its speed through the track once it is (a varispeed).
    state.playing.speed = Some(44_100.0 * 1.25);
    let at = |frame| Some(HeadFrame { head: 1, frame });
    assert_eq!(state.playing.frame_at(84.0), at(1_055_125));
    // Before the output's first frame: none.
    assert_eq!(state.playing.frame_at(0.0), None);
    // A position of another track: none.
    state.playing.anchor.as_mut().unwrap().index = index + 1;
    assert_eq!(state.playing.frame_at(84.0), None);
    // Forgotten (a seek, a jump): none, and counted, so what was worked
    // out from it is known to be off.
    let before = state.playing.anchors_forgotten;
    state.playing.forget_anchor();
    assert_eq!(state.playing.frame_at(84.0), None);
    assert_eq!(state.playing.anchors_forgotten, before + 1);
    assert_eq!(Playing::default().anchors_forgotten, 0);
}
