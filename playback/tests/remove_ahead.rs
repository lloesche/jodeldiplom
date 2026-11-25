//! The track after the one heard removed while the heard one runs out
//! ([`Command::Remove`]): it was decoded ahead (written after the heard
//! track's end, or waiting for the output to play out), never heard. The
//! heard track plays to its last frame and ends finished; the track after
//! the removed one follows it gaplessly (its stems too), or is held there
//! with [`Command::HoldAtEnd`]; the removed one never starts or ends. What
//! the device keeps of it when the output gives it back is heard between
//! the two.

mod common;

use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use common::lanes::{options, quarter, Wide};
use common::*;
use jodel_output::{SampleFormat, SimClock};
use jodel_playback::{Captures, Command, Event, MakeStems, MemoryTarget, Player, PlayerOptions};

const RATE: u32 = 44_100;

/// Frames the held device lets the engine write ahead of what it played.
const ROOM: u64 = RATE as u64 / 4;

/// Frames played before the track after the heard one is removed: its
/// last eighth of a second is still to be heard.
const PLAYED: u64 = RATE as u64 * 7 / 8;

/// The first track's frames the device holds then.
const A_HELD: u64 = RATE as u64 - PLAYED;

/// `frames` stereo frames of one value, as a 24-bit WAV at `rate`.
fn level_wav(path: &std::path::Path, rate: u32, frames: usize, value: i32) -> Vec<i32> {
    let samples = vec![value; frames * 2];
    write_wav(path, rate, 2, &samples);
    samples
}

/// Waits until the device has played `played` frames and holds `held`.
fn wait_held(hold: &Hold, played: u64, held: u64) {
    let deadline = Instant::now() + TIMEOUT;
    while hold.played() < played || (hold.held_words().len() as u64) < held {
        assert!(Instant::now() < deadline, "not played and held");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// A player on a held device, its tracks and what it heard.
struct Run {
    _dir: TempDir,
    paths: Vec<PathBuf>,
    /// The tracks' samples.
    samples: Vec<Vec<i32>>,
    player: Player,
    events: Receiver<Event>,
    captures: Captures,
    hold: std::sync::Arc<Hold>,
}

/// The tracks `names` of the three (a ramp a second long, a level a second
/// long at `b_rate`, another level half a second long) queued on a held
/// device, the first playing, its first `PLAYED` frames let through.
fn run(name: &str, names: &[&str], b_rate: u32, lanes: bool, options: PlayerOptions) -> Run {
    let dir = temp_dir(name);
    let paths: Vec<PathBuf> = names.iter().map(|n| dir.join(format!("{n}.wav"))).collect();
    let samples = names
        .iter()
        .zip(&paths)
        .map(|(n, path)| match *n {
            "a" => ramp_wav(path, RATE, RATE),
            "b" => level_wav(path, b_rate, b_rate as usize, 3 << 20),
            _ => level_wav(path, RATE, RATE as usize / 2, -(5 << 20)),
        })
        .collect();
    let caps = caps(&[RATE, 48_000], &[SampleFormat::S24In32Low]);
    let device = if lanes {
        MemoryTarget::with_lanes(caps, SimClock::Instant, 5)
    } else {
        MemoryTarget::new(caps, SimClock::Instant)
    };
    let captures = device.captures();
    let (target, hold) = HoldingTarget::around(Box::new(device), ROOM);
    let (player, events) = Player::new(Box::new(target), options);
    if lanes {
        player.make_stems(MakeStems::Separate).unwrap();
    }
    player.load(paths.clone()).unwrap();
    player.play(0).unwrap();
    hold.play(PLAYED);
    Run {
        _dir: dir,
        paths,
        samples,
        player,
        events,
        captures,
        hold,
    }
}

/// The three tracks queued, the second as fast as the first or not.
fn three(name: &str, b_rate: u32) -> Run {
    run(
        name,
        &["a", "b", "c"],
        b_rate,
        false,
        PlayerOptions::default(),
    )
}

/// Removes the second track; returns the events until the removal is
/// done. The queue is said before what was written is taken back, so a
/// move that changes nothing follows: its queue said, the removal is done
/// (the device must not play on before, or it would play what is taken
/// back).
fn remove_second(r: &Run) -> Vec<Event> {
    r.player.remove(1).unwrap();
    r.player.move_track(0, 0).unwrap();
    let said = std::cell::Cell::new(0);
    collect_until(&r.events, |e| {
        if matches!(e, Event::QueueChanged { queue, .. } if queue.len() == 2) {
            said.set(said.get() + 1);
        }
        said.get() == 2
    })
}

fn started(seen: &[Event]) -> Vec<(usize, PathBuf)> {
    seen.iter()
        .filter_map(|e| match e {
            Event::TrackStarted(s) => Some((s.index, s.info.path.clone())),
            _ => None,
        })
        .collect()
}

fn ended(seen: &[Event]) -> Vec<(usize, bool)> {
    seen.iter()
        .filter_map(|e| match e {
            Event::TrackEnded { index, finished } => Some((*index, *finished)),
            _ => None,
        })
        .collect()
}

/// Checks the one stream the device got against `heard` (24-bit
/// samples): nothing cut or padded.
fn heard_exactly(captures: &Captures, heard: &[i32]) {
    let streams = captured(captures);
    assert_eq!(streams.len(), 1, "one stream, never reopened");
    let bytes = &streams[0].1;
    let frame = 8;
    let want = device_bytes(heard, SampleFormat::S24In32Low);
    let first = bytes.iter().zip(&want).position(|(x, y)| x != y);
    assert_eq!(
        first.map(|i| i / frame),
        None,
        "the first frame heard wrong"
    );
    let after: Vec<u8> = bytes.iter().skip(want.len()).take(16).copied().collect();
    assert_eq!(
        bytes.len() / frame,
        heard.len() / 2,
        "nothing cut or padded (then {after:?})"
    );
}

/// Removes the second track, lets everything play, and checks what was
/// heard: the first track whole, the first `kept` frames of the second
/// (what the device kept of it), then the third, on one stream.
fn remove_second_and_check(r: Run, kept: usize) {
    let mut seen = remove_second(&r);
    r.hold.play_all();
    seen.extend(until_queue_end(&r.events));
    drop(r.player);
    assert_eq!(
        started(&seen),
        [(0, r.paths[0].clone()), (1, r.paths[2].clone())],
        "the removed track never starts"
    );
    assert_eq!(
        ended(&seen),
        [(0, true), (1, true)],
        "the heard track ends finished, the removed one never ends"
    );
    let mut heard = r.samples[0].clone();
    heard.extend(&r.samples[1][..kept * 2]);
    heard.extend(&r.samples[2]);
    heard_exactly(&r.captures, &heard);
}

#[test]
fn the_next_track_removed_during_the_run_in_leaves_the_heard_one_whole() {
    let r = three("remove-ahead-written", RATE);
    // The first track's last eighth of a second held, and the second's
    // first eighth after it: written, not heard.
    wait_held(&r.hold, PLAYED, ROOM);
    remove_second_and_check(r, 0);
}

#[test]
fn the_next_track_waiting_for_another_rate_removed_leaves_the_heard_one_whole() {
    let r = three("remove-ahead-waiting", 48_000);
    // The first track written to its end, its last eighth of a second held;
    // the second waits for the output to play out to open at its rate.
    wait_held(&r.hold, PLAYED, A_HELD);
    remove_second_and_check(r, 0);
}

#[test]
fn what_the_device_keeps_of_the_removed_track_is_heard_before_the_next() {
    let r = three("remove-ahead-kept", RATE);
    wait_held(&r.hold, PLAYED, ROOM);
    // The device keeps what it plays next: the first track's rest, and
    // the first thousand frames of the second.
    let kept = 1_000;
    r.hold.keep_on_retract(A_HELD + kept as u64);
    remove_second_and_check(r, kept);
}

#[test]
fn held_at_the_end_the_track_after_the_removed_one_waits() {
    let r = three("remove-ahead-held", RATE);
    wait_held(&r.hold, PLAYED, ROOM);
    // Held from now: the second, decoded ahead already, is removed; the
    // third waits to be started.
    r.player.send(Command::HoldAtEnd(true)).unwrap();
    let mut seen = remove_second(&r);
    r.hold.play_all();
    seen.extend(collect_until(&r.events, |e| {
        matches!(e, Event::Held { .. })
    }));
    drop(r.player);
    assert!(
        matches!(seen.last(), Some(Event::Held { index: 1 })),
        "{:?}",
        seen.last()
    );
    assert_eq!(started(&seen), [(0, r.paths[0].clone())]);
    assert_eq!(ended(&seen), [(0, true)]);
    heard_exactly(&r.captures, &r.samples[0]);
}

/// Plays the tracks `names` on a held device of lanes, stems separated
/// (and waited for), removing the second during the first's run-in when
/// `remove`; returns the lane stream.
fn wide(name: &str, names: &[&str], remove: bool) -> Wide {
    let r = run(name, names, RATE, true, options(false, true, quarter));
    wait_held(&r.hold, PLAYED, ROOM);
    if remove {
        remove_second(&r);
    }
    r.hold.play_all();
    until_queue_end(&r.events);
    drop(r.player);
    Wide::of(&r.captures)
}

#[test]
fn the_stems_of_the_track_after_the_removed_one_carry_on_as_in_a_run_in() {
    let _heavy = heavy();
    let plain = wide("remove-ahead-stems-plain", &["a", "c"], false);
    let removed = wide("remove-ahead-stems", &["a", "b", "c"], true);
    assert_eq!(removed.frames(), plain.frames());
    assert_eq!(removed.main, plain.main, "main as in a run-in");
    // In a run-in the parts are there across the seam.
    let seam = RATE as usize;
    assert!(
        plain.control[seam - 4..seam + 4].iter().all(|&w| w == 1),
        "the parts across the seam"
    );
    let differ = |x: &[i32], y: &[i32]| x.iter().zip(y).position(|(a, b)| a != b);
    assert_eq!(
        differ(&removed.control, &plain.control),
        None,
        "the control words as in a run-in"
    );
    for (k, (r, p)) in removed.parts.iter().zip(&plain.parts).enumerate() {
        assert_eq!(differ(r, p).map(|i| i / 2), None, "stem {k}");
    }
}
