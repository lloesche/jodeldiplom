//! Jumps within the playing track ([`Command::Jump`]): at once or at a time
//! of the track, from the stretch it keeps ([`Command::JumpWindow`]),
//! without a gap. The track is a ramp, so every sample says which frame of
//! the track it is.

mod common;

use std::time::{Duration, Instant};

use common::*;
use jodel_output::{SampleFormat, SimClock};
use jodel_playback::{Event, JumpAt, MemoryTarget, Player, PlayerOptions, Settings};

const RATE: u32 = 44_100;
/// The ramp's period: its values go round every this many frames.
const PERIOD: u64 = 30_000;

fn options(fade: Duration) -> PlayerOptions {
    PlayerOptions {
        settings: Settings::default(),
        position_interval: Duration::from_millis(1),
        switch_fade: fade,
        ..PlayerOptions::default()
    }
}

/// The 24-bit samples of a stream the device took as `S24In32Low`.
fn samples(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|b| (i32::from_le_bytes([b[0], b[1], b[2], b[3]]) << 8) >> 8)
        .collect()
}

/// The left channel's ramp value for track frame `frame`.
fn value(frame: u64) -> i32 {
    ((frame % PERIOD) as i32) << 8
}

/// A ramp of `seconds` and a player on a gated device at its rate, with a
/// window of `back` and `ahead` seconds.
fn setup(
    name: &str,
    seconds: u32,
    window: (f64, f64),
    fade: Duration,
) -> (
    TempDir,
    Player,
    std::sync::mpsc::Receiver<Event>,
    jodel_playback::Captures,
    std::sync::Arc<Gate>,
) {
    let dir = temp_dir(name);
    let path = dir.join("ramp.wav");
    ramp_wav(&path, RATE, RATE * seconds);
    let (target, captures, gate) = GatedTarget::new(caps(&[RATE], &[SampleFormat::S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options(fade));
    player.jump_window(window.0, window.1).unwrap();
    player.load(vec![path]).unwrap();
    (dir, player, events, captures, gate)
}

/// The left channel of the one stream played.
fn left(captures: &jodel_playback::Captures) -> Vec<i32> {
    let streams = captured(captures);
    assert_eq!(streams.len(), 1, "one stream");
    samples(&streams[0].1).iter().step_by(2).copied().collect()
}

/// Where the ramp in `got` does not go on by one: output frame, the track
/// frame (within the ramp's period) before and after.
fn breaks(got: &[i32]) -> Vec<(usize, i32, i32)> {
    (1..got.len())
        .filter(|&i| {
            let (a, b) = (got[i - 1] >> 8, got[i] >> 8);
            b != (a + 1) % PERIOD as i32
        })
        .map(|i| (i, got[i - 1] >> 8, got[i] >> 8))
        .take(8)
        .collect()
}

/// Frames of the track `got[from..]` is, from `frame` on, as far as it goes.
fn assert_track_from(got: &[i32], from: usize, frame: u64, what: &str) {
    let mismatch = got[from..]
        .iter()
        .enumerate()
        .find(|&(i, &v)| v != value(frame + i as u64));
    assert!(
        mismatch.is_none(),
        "{what}: at {:?} (from output frame {from}, track frame {frame}); breaks {:?}",
        mismatch.map(|(i, v)| (from + i, *v >> 8, value(frame + i as u64) >> 8)),
        breaks(got)
    );
}

/// Where the stream jumped: the first break in the ramp, at or after the
/// frame the jump was asked for (the write under way when it was asked for
/// goes out first: up to one write later).
fn cut(got: &[i32], asked: usize) -> usize {
    let at = (1..got.len())
        .find(|&i| got[i] != value(i as u64))
        .expect("a jump");
    assert!(
        (asked..asked + 8_192).contains(&at),
        "jumped at {at}, asked for at {asked}"
    );
    at
}

fn jumped(events: &[Event]) -> Vec<f64> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::Jumped { seconds, .. } => Some(*seconds),
            _ => None,
        })
        .collect()
}

/// At once, back two seconds: the track up to the frame being written,
/// then from the target to its end, one stream with nothing lost or
/// doubled; the position goes back with it.
#[test]
fn a_jump_now_plays_on_from_the_target() {
    let (_dir, player, events, captures, gate) = setup("now", 6, (4.0, 4.0), Duration::ZERO);
    let at = 3 * u64::from(RATE);
    gate.allow(at);
    player.play(0).unwrap();
    gate.wait_written(at);
    let mut seen = collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    player.jump(0, 1.0, JumpAt::Now).unwrap();
    gate.allow_all();
    seen.extend(until_queue_end(&events));
    drop(player);
    assert_eq!(jumped(&seen), [1.0]);
    assert!(!seen
        .iter()
        .any(|e| matches!(e, Event::Seeked { .. } | Event::Error { .. })));
    let got = left(&captures);
    let to = u64::from(RATE);
    let at = cut(&got, at as usize);
    assert_track_from(&got, at, to, "after");
    assert_eq!(got.len() as u64, at as u64 + 6 * u64::from(RATE) - to);
    // Positions after the jump are from the target on.
    let after: Vec<f64> = seen
        .iter()
        .skip_while(|e| !matches!(e, Event::Jumped { .. }))
        .filter_map(|e| match e {
            Event::Position(p) => Some(p.seconds),
            _ => None,
        })
        .collect();
    assert!(
        after.first().is_some_and(|&s| (1.0..1.5).contains(&s)),
        "{after:?}"
    );
}

/// Jumps one after another (the same key struck again and again): each
/// from where the last one landed, the second to before where the first
/// one went (from what the track kept from before it).
#[test]
fn jumps_one_after_another_each_play_from_their_target() {
    let (_dir, player, events, captures, gate) = setup("again", 6, (4.0, 4.0), Duration::ZERO);
    let rate = u64::from(RATE);
    gate.allow(3 * rate);
    player.play(0).unwrap();
    gate.wait_written(3 * rate);
    let mut seen = collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    player.jump(0, 2.0, JumpAt::Now).unwrap();
    seen.extend(collect_until(&events, |e| {
        matches!(e, Event::Jumped { .. })
    }));
    // Half a second on from 2 s, then back to 1 s.
    gate.allow(3 * rate + rate / 2);
    gate.wait_written(3 * rate + rate / 2);
    player.jump(0, 1.0, JumpAt::Now).unwrap();
    gate.allow_all();
    seen.extend(until_queue_end(&events));
    drop(player);
    assert_eq!(jumped(&seen), [2.0, 1.0]);
    let got = left(&captures);
    let first = cut(&got, 3 * rate as usize);
    let second = (first + 1..got.len())
        .find(|&i| got[i] != value(2 * rate + (i - first) as u64))
        .expect("a second jump");
    assert_track_from(&got[..second], first, 2 * rate, "after the first");
    assert_track_from(&got, second, rate, "after the second");
}

/// At a time of the track still to be written: written up to it, then the
/// target.
#[test]
fn a_jump_at_a_time_is_heard_at_that_frame() {
    let (_dir, player, events, captures, gate) = setup("at", 6, (4.0, 4.0), Duration::ZERO);
    let written = 2 * u64::from(RATE);
    gate.allow(written);
    player.play(0).unwrap();
    gate.wait_written(written);
    let mut seen = collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    player.jump(0, 0.5, JumpAt::At(2.5)).unwrap();
    gate.allow_all();
    seen.extend(until_queue_end(&events));
    drop(player);
    assert_eq!(jumped(&seen), [0.5]);
    let got = left(&captures);
    // Written up to it, so exactly there.
    let at = (2.5 * f64::from(RATE)) as usize;
    assert_eq!(cut(&got, at), at);
    assert_track_from(&got, at, u64::from(RATE) / 2, "after");
}

/// At a time already written (and played: nothing to take back here): the
/// jump lands as far after the target as it is late, so the beat goes on.
#[test]
fn a_late_jump_lands_as_late_after_the_target() {
    let (_dir, player, events, captures, gate) = setup("late", 6, (4.0, 4.0), Duration::ZERO);
    let written = 3 * u64::from(RATE);
    gate.allow(written);
    player.play(0).unwrap();
    gate.wait_written(written);
    let mut seen = collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    player.jump(0, 0.5, JumpAt::At(2.0)).unwrap();
    gate.allow_all();
    seen.extend(until_queue_end(&events));
    drop(player);
    assert_eq!(jumped(&seen).len(), 1);
    let got = left(&captures);
    let at = cut(&got, written as usize);
    // As late after 0.5 s as it is after 2 s.
    let late = at as u64 - 2 * u64::from(RATE);
    assert_track_from(&got, at, u64::from(RATE) / 2 + late, "after");
}

/// Ahead, into what was decoded ahead and beyond: from the target.
#[test]
fn a_jump_ahead_plays_from_the_target() {
    let (_dir, player, events, captures, gate) = setup("ahead", 8, (2.0, 2.0), Duration::ZERO);
    let at = u64::from(RATE);
    gate.allow(at);
    player.play(0).unwrap();
    gate.wait_written(at);
    let mut seen = collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    player.jump(0, 3.0, JumpAt::Now).unwrap();
    gate.allow_all();
    seen.extend(until_queue_end(&events));
    drop(player);
    assert_eq!(jumped(&seen), [3.0]);
    let got = left(&captures);
    let at = cut(&got, at as usize);
    assert_track_from(&got, at, 3 * u64::from(RATE), "after");
}

/// With the crossfade: exact outside it, between the two inside it.
#[test]
fn a_jump_crossfades_from_what_would_have_played() {
    let fade = Duration::from_millis(20);
    let (_dir, player, events, captures, gate) = setup("fade", 6, (4.0, 4.0), fade);
    let at = 3 * u64::from(RATE);
    gate.allow(at);
    player.play(0).unwrap();
    gate.wait_written(at);
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    player.jump(0, 1.0, JumpAt::Now).unwrap();
    gate.allow_all();
    until_queue_end(&events);
    drop(player);
    let got = left(&captures);
    let (to, n) = (u64::from(RATE), 882usize);
    // Where it jumped: from there the track plays to its end. (Not from the
    // first sample that differs: the crossfade's first ones round to what
    // would have played, and the output may have given a frame back.)
    let jumped_at = got.len() - (6 * u64::from(RATE) - to) as usize;
    assert!(
        (at as usize - 8_192..at as usize + 8_192).contains(&jumped_at),
        "jumped at {jumped_at}, asked for at {at}"
    );
    let at = jumped_at;
    assert_track_from(&got, at + n, to + n as u64, "after the fade");
    for i in 0..n {
        let (old, new) = (value(at as u64 + i as u64), value(to + i as u64));
        let (lo, hi) = (old.min(new), old.max(new));
        assert!(
            (lo - 1..=hi + 1).contains(&got[at + i]),
            "{i}: {}",
            got[at + i]
        );
    }
}

/// Outside the stretch kept (no window): the target is read again and
/// takes over once it is ready, without a gap (and no seek).
#[test]
fn without_a_window_a_jump_reads_its_target_again() {
    let (_dir, player, events, captures, gate) = setup("read", 4, (0.0, 0.0), Duration::ZERO);
    let at = 2 * u64::from(RATE);
    gate.allow(at);
    player.play(0).unwrap();
    gate.wait_written(at);
    let mut seen = collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    player.jump(0, 0.5, JumpAt::Now).unwrap();
    gate.allow_all();
    seen.extend(until_queue_end(&events));
    drop(player);
    assert!(
        !seen.iter().any(|e| matches!(e, Event::Seeked { .. })),
        "{seen:#?}"
    );
    assert_eq!(jumped(&seen), [0.5]);
    let got = left(&captures);
    let at = cut(&got, at as usize);
    assert_track_from(&got, at, u64::from(RATE) / 2, "after");
}

/// The same at a time of the track: read ahead of it, met at that frame.
#[test]
fn without_a_window_a_jump_at_a_time_is_heard_at_that_frame() {
    let (_dir, player, events, captures, gate) = setup("read-at", 6, (0.0, 0.0), Duration::ZERO);
    let written = 2 * u64::from(RATE);
    gate.allow(written);
    player.play(0).unwrap();
    gate.wait_written(written);
    let mut seen = collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    player.jump(0, 0.5, JumpAt::At(3.0)).unwrap();
    gate.allow_all();
    seen.extend(until_queue_end(&events));
    drop(player);
    assert_eq!(jumped(&seen), [0.5]);
    let got = left(&captures);
    let at = 3 * RATE as usize;
    assert_eq!(cut(&got, at), at);
    assert_track_from(&got, at, u64::from(RATE) / 2, "after");
}

/// In real time: the output gives back what it has not played, so the
/// jump is heard within a few device buffers of the command, not after the
/// whole ring; the stream has no gap.
#[test]
fn a_jump_now_in_real_time_is_heard_at_once() {
    let _guard = real_time();
    let dir = temp_dir("realtime");
    let path = dir.join("ramp.wav");
    ramp_wav(&path, RATE, RATE * 3);
    let target = MemoryTarget::new(
        caps(&[RATE], &[SampleFormat::S24In32Low]),
        SimClock::Realtime,
    );
    let captures = target.captures();
    let (player, events) = Player::new(Box::new(target), options(Duration::ZERO));
    player.jump_window(2.0, 2.0).unwrap();
    player.load(vec![path]).unwrap();
    player.play(0).unwrap();
    // A second in.
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut heard = 0;
    while heard < u64::from(RATE) {
        assert!(Instant::now() < deadline, "no position");
        if let Ok(Event::Position(p)) = events.recv_timeout(Duration::from_secs(1)) {
            heard = p.frame;
        }
    }
    player.jump(0, 0.25, JumpAt::Now).unwrap();
    let seen = until_queue_end(&events);
    drop(player);
    assert_eq!(jumped(&seen).len(), 1, "{seen:#?}");
    let got = left(&captures);
    // Where the ramp stops going on: the jump.
    let at = (1..got.len())
        .find(|&i| got[i] != value(i as u64))
        .expect("a jump");
    // From the target to the track's end; then only the silence the
    // device plays until the player has closed it (on a busy machine,
    // periods of it).
    let from = u64::from(RATE) / 4;
    let end = at + (u64::from(RATE * 3) - from) as usize;
    assert!(
        got.len() >= end,
        "{} frames, the track ends at {end}",
        got.len()
    );
    assert_track_from(&got[..end], at, from, "after");
    let sound = got[end..].iter().position(|&v| v != 0);
    assert_eq!(sound.map(|i| end + i), None, "sound after the end");
    // Within 150 ms of the frame heard when it was asked for (the ring is
    // 500 ms).
    let late = at as u64 - heard;
    assert!(
        late < u64::from(RATE) * 15 / 100,
        "heard {late} frames after"
    );
}
