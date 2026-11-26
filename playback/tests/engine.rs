//! The engine end to end, without a device: what reaches the output, byte
//! for byte, against the file's own samples.

#![allow(
    clippy::disallowed_methods,
    reason = "tests: a cover file that is not an image"
)]

mod common;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use common::*;
use jodel_output::{Capabilities, OpenRequest, Output, SampleFormat, SimClock, StreamFormat};
use jodel_playback::{
    Captures, Error, Event, Headroom, MemoryTarget, OutputTarget, PlayState, Player, Settings,
    Source, TapMiss,
};
use jodel_testkit::{fixture, TempDir};
use SampleFormat::*;

fn play_all(caps: Capabilities, files: &[&str]) -> (Vec<(StreamFormat, Vec<u8>)>, Vec<Event>) {
    let target = MemoryTarget::new(caps, SimClock::Instant);
    let captures = target.captures();
    play_on(Box::new(target), &captures, files)
}

fn play_on(
    target: Box<dyn OutputTarget>,
    captures: &Captures,
    files: &[&str],
) -> (Vec<(StreamFormat, Vec<u8>)>, Vec<Event>) {
    let (player, events) = Player::new(target, player_options());
    player
        .load(files.iter().map(|f| fixture(f)).collect())
        .unwrap();
    player.play(0).unwrap();
    let seen = until_queue_end(&events);
    let stopped = collect_until(&events, |e| matches!(e, Event::State(PlayState::Stopped)));
    drop(player);
    let mut all = seen;
    all.extend(stopped);
    (captured(captures), all)
}

/// A target whose one format is at rate `from` until its first output
/// opens, and at `to` from then on: a device the system moved to another
/// rate. `asked` is set once it is asked for its format after that.
struct Moving {
    inner: MemoryTarget,
    rates: (u32, u32),
    opened: bool,
    asked: Arc<AtomicBool>,
}

impl Moving {
    /// The target, its streams and its `asked`.
    fn new(rates: (u32, u32), clock: SimClock) -> (Moving, Captures, Arc<AtomicBool>) {
        let inner = MemoryTarget::new(caps(&[rates.0, rates.1], &[S24In32Low]), clock);
        let (captures, asked) = (inner.captures(), Arc::new(AtomicBool::new(false)));
        let target = Moving {
            inner,
            rates,
            opened: false,
            asked: asked.clone(),
        };
        (target, captures, asked)
    }
}

impl OutputTarget for Moving {
    fn name(&self) -> String {
        self.inner.name()
    }

    fn capabilities(&mut self) -> Result<Capabilities, jodel_output::Error> {
        self.asked.store(self.opened, Ordering::SeqCst);
        let rate = if self.opened {
            self.rates.1
        } else {
            self.rates.0
        };
        Ok(caps(&[rate], &[S24In32Low]))
    }

    fn open(&mut self, request: &OpenRequest) -> Result<Box<dyn Output>, jodel_output::Error> {
        self.opened = true;
        self.inner.open(request)
    }
}

#[test]
fn fixtures_open_with_their_format() {
    for (name, rate, bits) in [(F44, 44_100, 16), (F48, 48_000, 24)] {
        let source = Source::open(fixture(name)).unwrap();
        let info = source.info();
        assert_eq!(info.rate, rate);
        assert_eq!(info.channels, 2);
        assert_eq!(info.bits, bits);
        assert_eq!(info.frames, Some(FIXTURE_FRAMES as u64));
        assert_eq!(read_all(&fixture(name)).len(), 2 * FIXTURE_FRAMES);
    }
}

/// The engine's bytes are the file's samples, at the file's rate.
#[test]
fn engine_output_equals_the_file() {
    for (name, rate) in [(F44, 44_100u32), (F48, 48_000)] {
        let (streams, events) = play_all(caps(&[rate], &[S24In32Low]), &[name]);
        assert_eq!(streams.len(), 1, "{name}");
        assert_eq!(streams[0].0, StreamFormat::new(rate, 2, S24In32Low));
        assert!(
            streams[0].1 == device_bytes(&read_all(&fixture(name)), S24In32Low),
            "{name}: bytes differ"
        );
        let start = started(&events)[0];
        assert_eq!(start.output.format.rate, rate);
        assert!(start.route.lossless);
    }
}

/// The same samples in the device's own container (packed 24-bit here).
#[test]
fn packed_device_gets_the_same_samples() {
    let (streams, _) = play_all(caps(&[48_000], &[S24Packed]), &[F48]);
    assert_eq!(streams[0].0.sample, S24Packed);
    assert!(streams[0].1 == device_bytes(&read_all(&fixture(F48)), S24Packed));
}

#[test]
fn two_tracks_at_one_format_play_gaplessly_into_one_stream() {
    let (streams, events) = play_all(caps(&[44_100], &[S24In32Low]), &[F44, F44]);
    assert_eq!(streams.len(), 1, "the output was reopened");
    let mut want = read_all(&fixture(F44));
    want.extend(read_all(&fixture(F44)));
    assert!(streams[0].1 == device_bytes(&want, S24In32Low));

    // Track events in order, the second track starting at the first's end.
    let order: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            Event::TrackStarted(s) => Some(format!("start {}", s.index)),
            Event::TrackEnded { index, finished } => Some(format!("end {index} {finished}")),
            Event::QueueEnded => Some("queue end".into()),
            Event::State(s) => Some(format!("{s:?}")),
            _ => None,
        })
        .collect();
    assert_eq!(
        order,
        [
            "start 0",
            "Playing",
            "end 0 true",
            "start 1",
            "end 1 true",
            "queue end",
            "Stopped"
        ]
    );
    let ended_at: Vec<u64> = events
        .windows(2)
        .filter_map(|w| match (&w[0], &w[1]) {
            (Event::Position(p), Event::TrackEnded { .. }) => Some(p.output_frame),
            _ => None,
        })
        .collect();
    assert_eq!(ended_at, [FIXTURE_FRAMES as u64, 2 * FIXTURE_FRAMES as u64]);
}

/// The next track routes to the target's format as it is then: at
/// another, the output plays out and opens again at it.
#[test]
fn a_format_change_between_tracks_reopens_the_output() {
    let (target, captures, _) = Moving::new((44_100, 48_000), SimClock::Instant);
    let (streams, events) = play_on(Box::new(target), &captures, &[F44, F48]);
    assert_eq!(streams.len(), 2);
    assert_eq!(streams[0].0.rate, 44_100);
    assert_eq!(streams[1].0.rate, 48_000);
    assert!(streams[0].1 == device_bytes(&read_all(&fixture(F44)), S24In32Low));
    assert!(streams[1].1 == device_bytes(&read_all(&fixture(F48)), S24In32Low));
    let starts = started(&events);
    assert_eq!(starts.len(), 2);
    assert_eq!(starts[1].output.format.rate, 48_000);
}

/// Lets a period of the device's time pass on the newest stream a
/// [`SimClock::Manual`] target opened (none yet: nothing).
fn play_a_period(captures: &Captures) {
    if let Some(time) = captures.list().last().and_then(|c| c.time.clone()) {
        // A stream closed meanwhile plays nothing more.
        let _ = time.advance(480);
    }
}

/// New settings while the output plays out before it opens again at the
/// target's new format: the next track takes them, there.
#[test]
fn new_settings_during_the_play_out_reach_the_next_track() {
    let dir = TempDir::new("engine-play-out");
    let path = dir.join("ramp.wav");
    // A second, so the play-out lasts the output's ring.
    ramp_wav(&path, 48_000, 48_000);
    // The device's time passes only as the test lets it, so the play-out
    // lasts until the engine has the settings, however slowly it runs.
    let (target, captures, asked) = Moving::new((48_000, 44_100), SimClock::Manual);
    let (player, events) = Player::new(Box::new(target), player_options());
    player.load(vec![path.clone(), path]).unwrap();
    player.play(0).unwrap();
    // The next track is routed (to 44.1 kHz) as the first one is all
    // written: the output plays out from then.
    let deadline = std::time::Instant::now() + TIMEOUT;
    while !asked.load(Ordering::SeqCst) {
        assert!(std::time::Instant::now() < deadline, "never asked again");
        play_a_period(&captures);
        std::thread::sleep(Duration::from_millis(1));
    }
    // The device stands still. The settings, then a command answered
    // after them: once its answer comes, the engine has them.
    let mut seen: Vec<Event> = events.try_iter().collect();
    let headroom = Headroom(10);
    player.set_settings(Settings { headroom }).unwrap();
    player.enqueue(Vec::new()).unwrap();
    seen.extend(collect_until(&events, |e| {
        matches!(e, Event::QueueChanged { .. })
    }));
    // The play-out, the next track and the end of the queue, a period at
    // a time.
    loop {
        assert!(
            std::time::Instant::now() < deadline,
            "the queue never ended; seen: {seen:#?}"
        );
        play_a_period(&captures);
        match events.recv_timeout(Duration::from_millis(1)) {
            Ok(event) => {
                let end = matches!(event, Event::QueueEnded);
                seen.push(event);
                if end {
                    break;
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(e) => panic!("{e}; seen: {seen:#?}"),
        }
    }
    drop(player);
    assert!(
        !seen
            .iter()
            .any(|e| matches!(e, Event::Error { .. } | Event::Notice { .. })),
        "{seen:#?}"
    );
    let next = started(&seen)
        .into_iter()
        .find(|s| s.index == 1)
        .expect("the next track");
    assert_eq!(next.route.output_rate, 44_100);
    assert_eq!(next.route.resample.map(|r| r.headroom), Some(headroom));
    let streams = captured(&captures);
    let rates: Vec<u32> = streams.iter().map(|s| s.0.rate).collect();
    assert_eq!(rates, [48_000, 44_100]);
}

/// A 16-bit target: a 24-bit file has no route.
#[test]
fn no_route_is_reported_and_skipped() {
    let (streams, events) = play_all(caps(&[44_100], &[S16]), &[F48, F44]);
    let errors: Vec<(Option<usize>, &Error, bool)> = events
        .iter()
        .filter_map(|e| match e {
            Event::Error {
                index,
                error,
                skipped,
            } => Some((*index, error, *skipped)),
            _ => None,
        })
        .collect();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].0, Some(0));
    assert!(errors[0].2);
    match errors[0].1 {
        Error::NoRoute(none) => {
            assert!(none.reason.contains("48 kHz"), "{none}");
            assert!(none.reason.contains("16 bits only"), "{none}");
        }
        other => panic!("{other}"),
    }
    // The error comes before the track that plays instead.
    let error_at = events.iter().position(|e| matches!(e, Event::Error { .. }));
    let start_at = events
        .iter()
        .position(|e| matches!(e, Event::TrackStarted(_)));
    assert!(error_at < start_at);
    assert_eq!(started(&events)[0].index, 1);
    assert_eq!(streams.len(), 1);
    assert!(streams[0].1 == device_bytes(&read_all(&fixture(F44)), S16));
}

#[test]
fn positions_follow_the_track_start() {
    let (_, events) = play_all(caps(&[44_100], &[S24In32Low]), &[F44]);
    let start = events
        .iter()
        .position(|e| matches!(e, Event::TrackStarted(_)))
        .unwrap();
    let positions: Vec<_> = events[start..]
        .iter()
        .filter_map(|e| match e {
            Event::Position(p) => Some(*p),
            _ => None,
        })
        .collect();
    assert!(!positions.is_empty());
    assert!(positions
        .windows(2)
        .all(|w| w[0].output_frame <= w[1].output_frame));
    let end = positions.last().unwrap();
    assert_eq!(end.frame, FIXTURE_FRAMES as u64);
    assert!((end.seconds - FIXTURE_FRAMES as f64 / 44_100.0).abs() < 1e-9);
}

/// Seeking: the bytes before the seek, then exactly the file from the
/// target frame.
#[test]
fn seek_plays_on_from_the_target_frame() {
    for (name, rate) in [(F44, 44_100u32), (F48, 48_000)] {
        let (target, captures, gate) = GatedTarget::new(caps(&[rate], &[S24In32Low]));
        let (player, events) = Player::new(Box::new(target), player_options());
        player.load(vec![fixture(name)]).unwrap();
        let before = 6_000u64;
        gate.allow(before);
        player.play(0).unwrap();
        gate.wait_written(before);
        let target_frame = 3_000usize;
        player.seek(target_frame as f64 / f64::from(rate)).unwrap();
        let seen = collect_until(&events, |e| matches!(e, Event::Seeked { .. }));
        let Some(Event::Seeked { seconds, .. }) = seen.last() else {
            unreachable!()
        };
        assert!((seconds - target_frame as f64 / f64::from(rate)).abs() < 1e-9);
        gate.allow_all();
        until_queue_end(&events);
        drop(player);

        let all = read_all(&fixture(name));
        let mut want = all[..2 * before as usize].to_vec();
        want.extend(&all[2 * target_frame..]);
        let streams = captured(&captures);
        assert_eq!(streams.len(), 1);
        assert!(
            streams[0].1 == device_bytes(&want, S24In32Low),
            "{name}: {} bytes, want {}",
            streams[0].1.len(),
            want.len() * 4
        );
    }
}

#[test]
fn pause_and_resume_do_not_change_the_bytes() {
    let (target, captures, gate) = GatedTarget::new(caps(&[44_100], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), player_options());
    player.load(vec![fixture(F44)]).unwrap();
    gate.allow(5_000);
    player.play(0).unwrap();
    gate.wait_written(5_000);
    player.pause().unwrap();
    collect_until(&events, |e| matches!(e, Event::State(PlayState::Paused)));
    gate.allow_all();
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(gate.written(), 5_000, "wrote while paused");
    player.resume().unwrap();
    collect_until(&events, |e| matches!(e, Event::State(PlayState::Playing)));
    until_queue_end(&events);
    drop(player);
    assert!(captured(&captures)[0].1 == device_bytes(&read_all(&fixture(F44)), S24In32Low));
}

/// Another track played while paused plays, as does the next one: the
/// output, kept for it (the same format), is paused no more. A seek stays
/// paused.
#[test]
fn another_track_played_while_paused_plays() {
    let (target, captures, gate) = GatedTarget::new(caps(&[44_100], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), player_options());
    player
        .load(vec![fixture(F44), fixture(F44), fixture(F44)])
        .unwrap();
    let pause = || {
        player.pause().unwrap();
        collect_until(&events, |e| matches!(e, Event::State(PlayState::Paused)));
    };
    let started = |index: usize| {
        collect_until(
            &events,
            |e| matches!(e, Event::TrackStarted(s) if s.index == index),
        );
    };
    gate.allow(5_000);
    player.play(0).unwrap();
    gate.wait_written(5_000);
    pause();
    // A seek while paused writes nothing.
    player.seek(0.05).unwrap();
    gate.allow(10_000);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(gate.written(), 5_000, "wrote after a seek while paused");
    // Played while paused: heard (its start is told only once it is).
    player.play(1).unwrap();
    started(1);
    gate.wait_written(10_000);
    pause();
    gate.allow_all();
    player.next().unwrap();
    started(2);
    until_queue_end(&events);
    drop(player);
    // One stream (the output kept), the last track whole at its end.
    let streams = captured(&captures);
    assert_eq!(streams.len(), 1);
    let whole = device_bytes(&read_all(&fixture(F44)), S24In32Low);
    assert!(streams[0].1.ends_with(&whole), "the last track not heard");
}

/// Plain PCM plays untouched, and its seek is sample-accurate.
#[test]
fn plain_pcm_plays_untouched_and_seeks_exactly() {
    let dir = TempDir::new("wav");
    let wav = dir.join("ramp.wav");
    let samples = ramp_wav(&wav, 48_000, 50_000);
    std::fs::write(dir.join("cover.png"), b"\x89PNG not really").unwrap();
    let info = jodel_playback::probe(&wav).unwrap();
    assert_eq!(
        (info.rate, info.bits, info.frames),
        (48_000, 16, Some(50_000))
    );
    let cover = info.cover.unwrap();
    assert_eq!(cover.mime, "image/png");

    let (target, captures, gate) = GatedTarget::new(caps(&[48_000], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), player_options());
    player.load(vec![wav.clone()]).unwrap();
    gate.allow(7_777);
    player.play(0).unwrap();
    gate.wait_written(7_777);
    player.seek(30_001.0 / 48_000.0).unwrap();
    let seen = collect_until(&events, |e| matches!(e, Event::Seeked { .. }));
    let start = started(&seen)[0];
    assert_eq!(start.output.format.sample, S24In32Low);
    gate.allow_all();
    until_queue_end(&events);
    drop(player);
    let mut want = samples[..2 * 7_777].to_vec();
    want.extend(&samples[2 * 30_001..]);
    assert!(captured(&captures)[0].1 == device_bytes(&want, S24In32Low));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn next_previous_and_stop() {
    let (target, _captures, gate) = GatedTarget::new(caps(&[44_100], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), player_options());
    player
        .load(vec![fixture(F44), fixture(F44), fixture(F44)])
        .unwrap();
    collect_until(&events, |e| matches!(e, Event::QueueChanged { .. }));
    gate.allow(1_000);
    player.play(0).unwrap();
    gate.wait_written(1_000);
    player.next().unwrap();
    let seen = collect_until(
        &events,
        |e| matches!(e, Event::TrackStarted(s) if s.index == 1),
    );
    assert!(seen.iter().any(|e| matches!(
        e,
        Event::TrackEnded {
            index: 0,
            finished: false
        }
    )));
    player.previous().unwrap();
    collect_until(
        &events,
        |e| matches!(e, Event::TrackStarted(s) if s.index == 0),
    );
    player.stop().unwrap();
    let seen = collect_until(&events, |e| matches!(e, Event::State(PlayState::Stopped)));
    assert!(seen.iter().any(|e| matches!(
        e,
        Event::TrackEnded {
            index: 0,
            finished: false
        }
    )));
    // Play past the end stops.
    player.play(3).unwrap();
    collect_until(&events, |e| matches!(e, Event::State(PlayState::Stopped)));
}

#[test]
fn enqueue_after_the_end_continues_and_unreadable_files_are_skipped() {
    let (target, captures, gate) = GatedTarget::new(caps(&[44_100], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), player_options());
    player
        .load(vec![PathBuf::from("does-not-exist.flac"), fixture(F44)])
        .unwrap();
    gate.allow(2_000);
    player.play(0).unwrap();
    let seen = collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    assert!(seen.iter().any(|e| matches!(
        e,
        Event::Error {
            index: Some(0),
            error: Error::Source { .. },
            skipped: true
        }
    )));
    player.enqueue(vec![fixture(F44)]).unwrap();
    gate.allow_all();
    let seen = until_queue_end(&events);
    assert_eq!(
        started(&seen).iter().map(|s| s.index).collect::<Vec<_>>(),
        [2]
    );
    drop(player);
    let streams = captured(&captures);
    assert_eq!(streams.len(), 1);
    assert_eq!(streams[0].1.len(), 2 * FIXTURE_FRAMES * 2 * 4);
}

/// New headroom moves a resampled track to its new route in the same
/// output.
#[test]
fn settings_change_reroutes_the_track_being_heard() {
    let (target, captures, gate) = GatedTarget::new(caps(&[44_100], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), player_options());
    player.load(vec![fixture(F48)]).unwrap();
    gate.allow(3_000);
    player.play(0).unwrap();
    gate.wait_written(3_000);
    let first = collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    assert_eq!(started(&first)[0].route.headroom, Headroom::NONE);
    let headroom = Headroom(10);
    player.set_settings(Settings { headroom }).unwrap();
    let seen = trickle_until_started(&gate, &events);
    let start = started(&seen)[0];
    assert_eq!(start.route.output_rate, 44_100);
    assert_eq!(start.route.resample.map(|r| r.headroom), Some(headroom));
    gate.allow_all();
    until_queue_end(&events);
    drop(player);
    assert_eq!(captured(&captures).len(), 1);
}

/// A spectrum tap as a mixer's monitor holds what the device plays,
/// tagged with device frame indices, and the window heard now ends at the
/// device's clock.
#[test]
fn the_tap_holds_the_most_recent_output() {
    let (target, captures, gate, mixer) = gated_mixer(caps(&[44_100], &[S24In32Low]));
    let tap = jodel_playback::SpectrumTap::new(65_536);
    mixer.set_monitor(Some(Box::new(tap.clone())));
    let (player, events) = Player::new(Box::new(target), player_options());
    player.load(vec![fixture(F44)]).unwrap();
    // Whole periods of the mixer (10 ms at 44.1 kHz), past the output's
    // prefill (100 ms), from which it plays.
    let at = 11 * 441;
    gate.allow(at as u64);
    player.play(0).unwrap();
    gate.wait_written(at as u64);
    wait_captured(&captures, at as u64);
    let want = read_all(&fixture(F44));
    let frames = |from: usize, to: usize| -> Vec<[i32; 2]> {
        want[2 * from..2 * to]
            .chunks_exact(2)
            .map(|p| [p[0], p[1]])
            .collect()
    };
    let mut window = vec![[0; 2]; 1024];
    let heard = tap.read_heard(&mut window).unwrap();
    assert_eq!(
        (heard.start, heard.silence, heard.rate),
        (at as u64 - 1024, 0, 44_100)
    );
    assert_eq!(window, frames(at - 1024, at));
    let generation = heard.generation;
    gate.allow_all();
    until_queue_end(&events);
    let total = want.len() / 2;
    assert_eq!(tap.written(), total as u64);
    let got = tap.read(generation, total as u64, &mut window).unwrap();
    assert_eq!(got.start, total as u64 - 1024);
    assert_eq!(window, frames(total - 1024, total));
    collect_until(&events, |e| matches!(e, Event::State(PlayState::Stopped)));
    drop(player);
    // The device stays open with the engine's output closed, and closes
    // with the mixer.
    assert!(tap.read_heard(&mut window).is_ok());
    drop(mixer);
    assert_eq!(tap.read_heard(&mut window), Err(TapMiss::NoStream));
}

#[test]
fn handles_cross_threads() {
    fn send<T: Send>() {}
    fn shared<T: Send + Sync>() {}
    send::<Player>();
    send::<Event>();
    shared::<jodel_playback::SpectrumTap>();
    shared::<jodel_playback::TrackInfo>();
}
