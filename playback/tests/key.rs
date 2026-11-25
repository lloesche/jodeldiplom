//! The key shift ([`Command::Key`]) on an output of lanes: a sine two
//! semitones up comes out at its new frequency, as long as the file and
//! with no click at the change; the drums keep their key and the other
//! stems are shifted; without stems the whole mix is; back at 0 the file
//! plays bit for bit; two keys in a row crossfade as one does; the key
//! carries into the track the queue runs into and goes at a track picked;
//! one asked while nothing plays is the next song's (after its count-in
//! too), one asked during a count-in is heard from the song's first frame;
//! a seek keeps it; a head on a time map takes it on its map; a change is
//! heard within a tenth of a second on a device, and there stems that do
//! not come give way to the whole mix shifted; and no transition while the
//! key is shifted (nor a key while one is set up).

mod common;

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use common::lanes::{caps_of, quarter, wide_target_on, Gated, Wide};
use common::transition::{beats, lanes as plan_lanes, Desk};
use common::*;
use jodel_output::{SampleFormat, SimClock};
use jodel_playback::stems::Separator;
use jodel_playback::warp::TimeMap;
use jodel_playback::{
    Command, Envelope, Event, MakeStems, Player, PlayerOptions, SeparatorSource, Settings,
    TempoMode, TransitionPlan, TransitionState,
};

const RATE: u32 = 44_100;

/// Full scale, 24 bits.
const FULL: f64 = 8_388_607.0;

/// The sine's level: half of full scale.
const LEVEL: f64 = 0.5;

/// 440 Hz two semitones up.
const UP_TWO: f64 = 493.883;

/// 440 Hz three semitones up.
const UP_THREE: f64 = 523.251;

/// A stereo file of a 440 Hz sine, `seconds` long; and its samples.
fn sine(dir: &Path, name: &str, seconds: f64) -> (PathBuf, Vec<i32>) {
    let path = dir.join(name);
    let frames = (seconds * f64::from(RATE)) as usize;
    let samples: Vec<i32> = (0..frames)
        .flat_map(|n| {
            let t = n as f64 / f64::from(RATE);
            let v = (LEVEL * FULL * (2.0 * std::f64::consts::PI * 440.0 * t).sin()).round() as i32;
            [v, v]
        })
        .collect();
    write_wav(&path, RATE, 2, &samples);
    (path, samples)
}

/// The frequency of the strongest bin of a channel's 16384 frames from
/// frame `from` (a Hann window; the peak found to a hundredth of a bin).
fn peak_hz(samples: &[i32], channels: usize, from: usize) -> f64 {
    const N: usize = 16_384;
    let x: Vec<f64> = (0..N)
        .map(|i| {
            let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N as f64).cos();
            f64::from(samples[(from + i) * channels]) * w
        })
        .collect();
    let bin = |k: f64| {
        let (mut re, mut im) = (0.0, 0.0);
        for (i, &v) in x.iter().enumerate() {
            let a = 2.0 * std::f64::consts::PI * k * i as f64 / N as f64;
            re += v * a.cos();
            im -= v * a.sin();
        }
        re.hypot(im)
    };
    let hz = |k: f64| k * f64::from(RATE) / N as f64;
    // Between 300 and 700 Hz, then finely around the best.
    let (lo, hi) = (
        (300.0 * N as f64 / f64::from(RATE)) as usize,
        (700.0 * N as f64 / f64::from(RATE)) as usize,
    );
    let best = (lo..hi)
        .max_by(|&a, &b| bin(a as f64).total_cmp(&bin(b as f64)))
        .unwrap() as f64;
    let mut top = (best, bin(best));
    let mut k = best - 1.0;
    while k <= best + 1.0 {
        let m = bin(k);
        if m > top.1 {
            top = (k, m);
        }
        k += 0.01;
    }
    hz(top.0)
}

/// How a file is played with keys.
struct Run {
    separator: Option<SeparatorSource>,
    make: MakeStems,
    /// The commands sent, each once the output second given is heard
    /// (those of the same second together).
    sent: Vec<(f64, Command)>,
}

/// [`Command::Key`] at output second `at`.
fn key(at: f64, semitones: i8) -> (f64, Command) {
    (at, Command::Key { semitones })
}

impl Run {
    fn options(&self) -> PlayerOptions {
        PlayerOptions {
            position_interval: Duration::from_millis(1),
            separator: self.separator,
            offline: true,
            ..PlayerOptions::default()
        }
    }

    /// Plays `paths` offline on a lane device, the output let through a
    /// piece at a time, each key sent once its second is heard; the
    /// capture and the events.
    fn play(&self, paths: Vec<PathBuf>) -> (Wide, Vec<Event>) {
        let (target, captures, gate) = wide_target_on(caps_of(&[RATE], 2), SimClock::Instant);
        let (player, events) = Player::new(Box::new(target), self.options());
        player.make_stems(self.make).unwrap();
        gate.allow(4_096);
        player.load(paths).unwrap();
        player.play(0).unwrap();
        let mut sent = self.sent.iter().peekable();
        let seen = step_to_end(&gate, &events, &mut |e| {
            let Event::Position(p) = e else {
                return;
            };
            let heard = p.output_frame as f64 / f64::from(RATE);
            while let Some((_, command)) = sent.next_if(|(at, _)| heard > *at) {
                player.send(command.clone()).unwrap();
            }
        });
        drop(player);
        (Wide::of(&captures), seen)
    }
}

/// Lets the output through 4096 frames at a time (the next piece once the
/// last is written), telling `on` of each event, until the queue ends.
fn step_to_end(gate: &Gate, events: &Receiver<Event>, on: &mut dyn FnMut(&Event)) -> Vec<Event> {
    let deadline = Instant::now() + TIMEOUT * 2;
    let mut seen = Vec::new();
    loop {
        let target = gate.written() + 4_096;
        gate.allow(target);
        loop {
            assert!(Instant::now() < deadline, "the queue did not end");
            match events.recv_timeout(Duration::from_millis(2)) {
                Ok(e) => {
                    on(&e);
                    let end = matches!(e, Event::QueueEnded);
                    seen.push(e);
                    if end {
                        return seen;
                    }
                }
                Err(RecvTimeoutError::Timeout) if gate.written() >= target => break,
                Err(RecvTimeoutError::Timeout) => {}
                Err(e) => panic!("{e}"),
            }
        }
    }
}

/// The first output frame heard in `key`.
fn heard_in(seen: &[Event], key: i8) -> usize {
    seen.iter()
        .find_map(|e| match e {
            Event::Position(p) if p.key == key => Some(p.output_frame as usize),
            _ => None,
        })
        .unwrap_or_else(|| panic!("never heard in {key}"))
}

/// No click about `at` in a lane of a sine at `level` (of full scale) and
/// `hz` at most: from a tenth of a second before to a fifth after, no step
/// between two frames of the first channel larger than the sine's steepest
/// (with a margin), and no 10 ms (a window each ms) under half the sine's
/// level (a hole: a −14 dB dip, a dropout) or much over it. A crossfade
/// between two pitches beats: an ideal one between these sines, 20 ms of
/// raised cosine, is 0.59 of the level at its least over 10 ms (0.54 two
/// in a row), 0.47 over 5 ms, so windows shorter than 10 ms would take the
/// crossfade itself for a hole.
fn no_click(samples: &[i32], channels: usize, at: usize, level: f64, hz: f64) {
    let steepest = level * FULL * 2.0 * std::f64::consts::PI * hz / f64::from(RATE);
    let from = at - RATE as usize / 10;
    let to = at + RATE as usize / 5;
    let x: Vec<f64> = (from..to)
        .map(|f| f64::from(samples[f * channels]))
        .collect();
    let worst = x
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0, f64::max);
    assert!(
        worst < 1.3 * steepest,
        "a step of {worst} (the sine's steepest {steepest})"
    );
    let rms = level * FULL / 2f64.sqrt();
    let (window, step) = (RATE as usize / 100, RATE as usize / 1000);
    for k in (0..x.len() - window).step_by(step) {
        let piece = &x[k..k + window];
        let got = (piece.iter().map(|v| v * v).sum::<f64>() / window as f64).sqrt() / rms;
        assert!(
            (0.5..1.4).contains(&got),
            "{got:.3} of the level over 10 ms from {:.1} ms after the change",
            (k as f64 - (RATE / 10) as f64) * 1000.0 / f64::from(RATE)
        );
    }
}

/// Without stems the whole mix is shifted: a 440 Hz sine two semitones up
/// comes out at 493.9 Hz, the file's length, with no click at the change.
#[test]
fn a_sine_two_up_comes_out_at_its_new_frequency_as_long_and_without_a_click() {
    let dir = temp_dir("key-sine");
    let (path, samples) = sine(&dir, "sine.wav", 4.0);
    let run = Run {
        separator: None,
        make: MakeStems::Never,
        sent: vec![key(1.0, 2)],
    };
    let (wide, seen) = run.play(vec![path]);
    assert_eq!(wide.main.len(), samples.len(), "as long as the file");
    let at = heard_in(&seen, 2);
    let before = peak_hz(&wide.main, 2, RATE as usize / 4);
    assert!((before - 440.0).abs() < 1.0, "{before} Hz before");
    let after = peak_hz(&wide.main, 2, at + RATE as usize / 4);
    assert!((after - UP_TWO).abs() < 1.0, "{after} Hz after");
    // (The parts come only with the key, none asked for before: main is
    // what is heard across the change.)
    no_click(&wide.main, 2, at, LEVEL, UP_TWO);
    // The whole mix is the other instruments' part, the rest silent.
    let from = (at + RATE as usize / 4) * 2;
    assert!(wide.parts[..3]
        .iter()
        .all(|p| p[from..].iter().all(|&s| s == 0)));
    assert_eq!(wide.parts[3][from..], wide.main[from..]);
    // (No route switch: a key of its own keeps the head on its map.)
    let starts = seen
        .iter()
        .filter(|e| matches!(e, Event::TrackStarted(_)))
        .count();
    assert_eq!(starts, 1);
}

/// With stems the drums keep their key and the others are shifted; main
/// is their sum.
#[test]
fn the_drums_keep_their_key_and_the_other_stems_are_shifted() {
    let dir = temp_dir("key-stems");
    let (path, _) = sine(&dir, "sine.wav", 4.0);
    let run = Run {
        separator: Some(quarter),
        make: MakeStems::Separate,
        sent: vec![key(1.0, 2)],
    };
    let (wide, seen) = run.play(vec![path]);
    let at = heard_in(&seen, 2) + RATE as usize / 4;
    let drums = peak_hz(&wide.parts[0], 2, at);
    assert!((drums - 440.0).abs() < 1.0, "the drums at {drums} Hz");
    for (k, part) in wide.parts[1..].iter().enumerate() {
        let hz = peak_hz(part, 2, at);
        assert!((hz - UP_TWO).abs() < 1.0, "stem {} at {hz} Hz", k + 1);
    }
    let (worst, there) = wide.worst(at..at + RATE as usize);
    assert!(
        there > 0 && worst <= 4,
        "main {worst} steps off the parts' sum"
    );
    let change = heard_in(&seen, 2);
    no_click(&wide.main, 2, change, LEVEL, UP_TWO);
    for part in &wide.parts {
        no_click(part, 2, change, LEVEL / 4.0, UP_TWO);
    }
}

/// Back at 0 the head goes back onto its file: past the switch's crossfade
/// it is the file, bit for bit.
#[test]
fn back_at_0_the_file_plays_bit_for_bit() {
    let dir = temp_dir("key-back");
    let (path, samples) = sine(&dir, "sine.wav", 5.0);
    let run = Run {
        separator: Some(quarter),
        make: MakeStems::Separate,
        sent: vec![key(1.0, 3), key(2.0, 0)],
    };
    let (wide, seen) = run.play(vec![path]);
    let starts: Vec<(u64, u64)> = seen
        .iter()
        .filter_map(|e| match e {
            Event::TrackStarted(s) => Some((s.output_frame, s.from_frame)),
            _ => None,
        })
        .collect();
    assert_eq!(starts.len(), 2, "started, and back on its file: {starts:?}");
    let (back, frame) = starts[1];
    assert!(back as f64 / f64::from(RATE) > 2.0, "back at {back}");
    assert_eq!(back, frame, "where it was: the tempo never changed");
    // Past the switch's crossfade.
    let from = (back as usize + RATE as usize / 50) * 2;
    assert!(
        wide.main[from..] == samples[from..],
        "the file, bit for bit"
    );
    // Heard in 3 between.
    let shifted = heard_in(&seen, 3);
    assert!(shifted < back as usize);
}

/// The key stays for the track the queue runs into (gaplessly: the two as
/// long as the files), and goes back to 0 when a track is picked.
#[test]
fn the_key_carries_into_the_next_track_and_goes_at_a_track_picked() {
    let dir = temp_dir("key-next");
    let (a, a_samples) = sine(&dir, "a.wav", 2.0);
    let (b, b_samples) = sine(&dir, "b.wav", 2.0);
    let run = Run {
        separator: None,
        make: MakeStems::Never,
        sent: vec![key(0.5, 2)],
    };
    let (wide, seen) = run.play(vec![a.clone(), b.clone()]);
    assert_eq!(wide.main.len(), a_samples.len() + b_samples.len());
    let b_from = a_samples.len() / 2;
    let hz = peak_hz(&wide.main, 2, b_from + RATE as usize / 2);
    assert!((hz - UP_TWO).abs() < 1.0, "B at {hz} Hz");
    // Heard in 2 on B too.
    assert!(seen.iter().any(|e| matches!(e,
        Event::Position(p) if p.index == 1 && p.key == 2)));
    // A track picked: its own key.
    let (target, captures, gate) = wide_target_on(caps_of(&[RATE], 2), SimClock::Instant);
    let (player, events) = Player::new(Box::new(target), run.options());
    gate.allow(4_096);
    player.load(vec![a, b]).unwrap();
    player.play(0).unwrap();
    player.send(Command::Key { semitones: 2 }).unwrap();
    collect_until(&events, |e| matches!(e, Event::Key { semitones: 2 }));
    // The same track picked again keeps it, heard in it; another resets it.
    player.play(0).unwrap();
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    gate.allow(gate.written() + 8_192);
    collect_until(
        &events,
        |e| matches!(e, Event::Position(p) if p.index == 0 && p.key == 2),
    );
    player.play(1).unwrap();
    let seen = collect_until(&events, |e| matches!(e, Event::Key { .. }));
    assert!(matches!(seen.last(), Some(Event::Key { semitones: 0 })));
    gate.allow_all();
    let seen = until_queue_end(&events);
    assert!(seen
        .iter()
        .all(|e| !matches!(e, Event::Position(p) if p.index == 1 && p.key != 0)));
    drop(player);
    drop(captures);
}

/// On a device a key change is heard within a tenth of a second of the
/// frame heard when it was asked for: the output gives back what it has
/// not played and the head starts again there in the new key.
#[test]
fn on_a_device_a_key_change_is_heard_within_a_tenth_of_a_second() {
    let _real_time = real_time();
    let dir = temp_dir("key-device");
    let (path, _) = sine(&dir, "sine.wav", 6.0);
    let run = Run {
        separator: Some(quarter),
        make: MakeStems::Separate,
        sent: Vec::new(),
    };
    let options = PlayerOptions {
        offline: false,
        ..run.options()
    };
    let (target, captures, gate) = wide_target_on(caps_of(&[RATE], 2), SimClock::Paced);
    let (player, events) = Player::new(Box::new(target), options);
    player.make_stems(run.make).unwrap();
    gate.allow_all();
    player.load(vec![path]).unwrap();
    player.play(0).unwrap();
    let mut latencies = Vec::new();
    for (at, key) in [(1.0, 2i8), (1.6, -3), (2.2, 0)] {
        let seen = collect_until(
            &events,
            |e| matches!(e, Event::Position(p) if p.output_frame as f64 / f64::from(RATE) > at),
        );
        let Some(Event::Position(heard)) = seen.last() else {
            unreachable!()
        };
        let asked = heard.output_frame;
        player.send(Command::Key { semitones: key }).unwrap();
        let seen = collect_until(&events, |e| matches!(e, Event::Position(p) if p.key == key));
        let Some(Event::Position(p)) = seen.last() else {
            unreachable!()
        };
        latencies.push((p.output_frame - asked) as f64 / f64::from(RATE));
    }
    drop(player);
    println!("keys heard after {latencies:.3?} s");
    // Measured between positions, told every tenth of a second: a key in
    // force by the next one told after it was asked for came within it.
    for latency in latencies {
        assert!(latency <= 0.1, "heard {latency} s after it was asked for");
    }
    drop(captures);
}

/// No transition while the key is shifted, and no key change while one is
/// set up: each refused, saying why.
#[test]
fn no_transition_in_a_key_and_no_key_in_a_transition() {
    let dir = temp_dir("key-transition");
    let (a, _) = sine(&dir, "a.wav", 10.0);
    let (b, _) = sine(&dir, "b.wav", 10.0);
    let caps = common::caps(&[RATE], &[SampleFormat::S24In32Low]);
    let desk = Desk::new(caps, common::transition::options());
    desk.play(vec![a, b]);
    desk.gate.allow(u64::from(RATE));
    let plan = || TransitionPlan {
        from: 0,
        into: 1,
        start: 6.0,
        length: 2.0,
        entry: 1.0,
        beats: beats(6.0, 0.5, 1.0, 0.5, 5),
        lanes: plan_lanes(
            Envelope::fade(1.0, 0.0, 0.0, 1.0),
            Envelope::fade(0.0, 1.0, 0.0, 1.0),
        ),
        tempo: TempoMode::Varispeed,
        settle: 1.0,
        glide: false,
    };
    desk.player.send(Command::Key { semitones: 2 }).unwrap();
    collect_until(&desk.events, |e| matches!(e, Event::Key { semitones: 2 }));
    desk.player.transition(plan()).unwrap();
    let seen = collect_until(&desk.events, |e| matches!(e, Event::Transition(_)));
    let refused = matches!(seen.last(),
        Some(Event::Transition(TransitionState::Cancelled { reason })) if reason.contains("key"));
    assert!(refused, "{:?}", seen.last());
    // Back at 0, and once back on its own route, it may.
    desk.player.send(Command::Key { semitones: 0 }).unwrap();
    collect_until(&desk.events, |e| matches!(e, Event::Key { semitones: 0 }));
    desk.gate.allow(2 * u64::from(RATE));
    collect_until(&desk.events, |e| matches!(e, Event::TrackStarted(_)));
    desk.player.transition(plan()).unwrap();
    collect_until(&desk.events, |e| {
        matches!(e, Event::Transition(TransitionState::Armed))
    });
    desk.player.send(Command::Key { semitones: 1 }).unwrap();
    let seen = collect_until(&desk.events, |e| matches!(e, Event::Key { .. }));
    assert!(matches!(seen.last(), Some(Event::Key { semitones: 0 })));
    assert!(seen.iter().any(|e| matches!(e,
        Event::Notice { message, .. } if message.contains("transition"))));
    drop(desk);
}

/// Two keys in a row (a slider dragged, a key held): the second joins the
/// first's start as the first joined what played, crossfaded: no click on
/// main nor on any part.
#[test]
fn two_keys_in_a_row_crossfade_as_one_does() {
    let dir = temp_dir("key-two");
    let (path, _) = sine(&dir, "sine.wav", 3.0);
    let run = Run {
        separator: Some(quarter),
        make: MakeStems::Separate,
        sent: vec![key(1.0, 2), key(1.0, 3)],
    };
    let (wide, seen) = run.play(vec![path]);
    let at = heard_in(&seen, 3);
    no_click(&wide.main, 2, at, LEVEL, UP_THREE);
    for part in &wide.parts {
        no_click(part, 2, at, LEVEL / 4.0, UP_THREE);
    }
    let hz = peak_hz(&wide.main, 2, at + RATE as usize / 4);
    assert!((hz - UP_THREE).abs() < 1.0, "{hz} Hz");
}

/// The output frame of the first position of a track's own time (its
/// count-in over), and whether every position from it is in `key`.
fn song_start(seen: &[Event], key: i8) -> (usize, bool) {
    let heard: Vec<_> = seen
        .iter()
        .filter_map(|e| match e {
            Event::Position(p) if p.seconds >= 0.0 => Some(*p),
            _ => None,
        })
        .collect();
    let first = heard.first().expect("positions in the song").output_frame;
    (first as usize, heard.iter().all(|p| p.key == key))
}

/// A player offline on a lane device, the stems made, `paths` loaded, a
/// few frames let through; the capture, the gate and the events.
fn offline_player(
    paths: Vec<PathBuf>,
) -> (
    Player,
    Receiver<Event>,
    jodel_playback::Captures,
    std::sync::Arc<Gate>,
) {
    let run = Run {
        separator: Some(quarter),
        make: MakeStems::Separate,
        sent: Vec::new(),
    };
    let (target, captures, gate) = wide_target_on(caps_of(&[RATE], 2), SimClock::Instant);
    let (player, events) = Player::new(Box::new(target), run.options());
    player.make_stems(run.make).unwrap();
    gate.allow(4_096);
    player.load(paths).unwrap();
    (player, events, captures, gate)
}

/// A key asked while nothing plays is the next song's: Karaoke's START
/// (a count-in, then the song) plays it in that key from its first frame.
#[test]
fn a_key_asked_while_nothing_plays_is_the_next_songs() {
    let dir = temp_dir("key-stopped");
    let (a, _) = sine(&dir, "a.wav", 2.0);
    let (b, _) = sine(&dir, "b.wav", 2.0);
    let (player, events, captures, gate) = offline_player(vec![a, b]);
    // A song played and stopped: the key is still for whatever comes.
    player.play(0).unwrap();
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    player.stop().unwrap();
    player.send(Command::Key { semitones: 2 }).unwrap();
    collect_until(&events, |e| matches!(e, Event::Key { semitones: 2 }));
    player.play_in(1, 0.5).unwrap();
    gate.allow_all();
    let seen = until_queue_end(&events);
    drop(player);
    assert!(!seen.iter().any(|e| matches!(e, Event::Key { .. })), "kept");
    let (start, keyed) = song_start(&seen, 2);
    assert!(keyed, "every position of the song in +2");
    let wide = Wide::nth(&captures, captures.list().len() - 1);
    let hz = peak_hz(&wide.main, 2, start + RATE as usize / 4);
    assert!((hz - UP_TWO).abs() < 1.0, "{hz} Hz");
}

/// A key asked during a count-in is heard from the song's first frame.
#[test]
fn a_key_asked_during_a_count_in_is_heard_from_the_songs_first_frame() {
    let dir = temp_dir("key-count-in");
    let (a, _) = sine(&dir, "a.wav", 2.0);
    let (player, events, captures, gate) = offline_player(vec![a]);
    player.play_in(0, 1.0).unwrap();
    // The count-in's silence begins to be heard; the song is not yet.
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    player.send(Command::Key { semitones: 2 }).unwrap();
    collect_until(&events, |e| matches!(e, Event::Key { semitones: 2 }));
    gate.allow_all();
    let seen = until_queue_end(&events);
    drop(player);
    let (start, keyed) = song_start(&seen, 2);
    assert!(keyed, "every position of the song in +2");
    assert!(
        start >= RATE as usize * 9 / 10,
        "after the count-in: {start}"
    );
    let wide = Wide::of(&captures);
    let hz = peak_hz(&wide.main, 2, start + RATE as usize / 10);
    assert!((hz - UP_TWO).abs() < 1.0, "{hz} Hz");
}

/// A seek keeps the key: the song plays on from there in it.
#[test]
fn a_seek_keeps_the_key() {
    let dir = temp_dir("key-seek");
    let (path, _) = sine(&dir, "sine.wav", 5.0);
    let run = Run {
        separator: Some(quarter),
        make: MakeStems::Separate,
        sent: vec![key(1.0, 2), (2.0, Command::Seek(0.5))],
    };
    let (wide, seen) = run.play(vec![path]);
    let seeked = seen
        .iter()
        .position(|e| matches!(e, Event::Seeked { .. }))
        .expect("seeked");
    let after: Vec<_> = seen[seeked..]
        .iter()
        .filter_map(|e| match e {
            Event::Position(p) => Some(*p),
            _ => None,
        })
        .collect();
    assert!(after.iter().all(|p| p.key == 2), "in +2 after the seek");
    let p = after[0];
    assert!((p.seconds - 0.5).abs() < 0.05, "from 0.5 s: {p:?}");
    let hz = peak_hz(&wide.main, 2, p.output_frame as usize + RATE as usize / 4);
    assert!((hz - UP_TWO).abs() < 1.0, "{hz} Hz");
}

/// A head read along a time map takes the key on its map: its positions
/// stay the map's, and its pitch is the map's speed times the key's.
#[test]
fn a_head_on_a_time_map_takes_the_key_on_it() {
    let dir = temp_dir("key-map");
    let (path, _) = sine(&dir, "sine.wav", 7.0);
    // Straight to 1 s, 5 % faster by 2 s and on at that to 4 s.
    let map = std::sync::Arc::new(TimeMap::from_knots(
        vec![(1.0, 1.0), (2.0, 2.05), (4.0, 4.15)],
        vec![1.0, 1.05, 1.05],
        1.0,
    ));
    let along = Command::Along {
        map: std::sync::Arc::clone(&map),
        from: u64::from(RATE),
        tempo: TempoMode::Varispeed,
    };
    let run = Run {
        separator: Some(quarter),
        make: MakeStems::Separate,
        sent: vec![(0.5, along), key(2.2, 2)],
    };
    let (wide, seen) = run.play(vec![path]);
    let at = heard_in(&seen, 2);
    let rate = f64::from(RATE);
    for e in &seen {
        if let Event::Position(p) = e {
            let o = p.output_frame as f64 / rate;
            if p.key == 2 && o < 4.0 {
                let want = map.at(o);
                assert!(
                    (p.seconds - want).abs() < 1e-6,
                    "{} s at {o}, the map's {want}",
                    p.seconds
                );
            }
        }
    }
    let want = 440.0 * 1.05 * jodel_playback::ratio_of(2);
    let hz = peak_hz(&wide.main, 2, at + RATE as usize / 4);
    assert!(
        (hz - want).abs() < 1.5,
        "{hz} Hz, the map's and the key's {want}"
    );
}

/// A separator that separates nothing: the stems never come.
static NEVER: Gated = Gated::new();

fn never() -> Option<&'static dyn Separator> {
    Some(&NEVER)
}

/// On a device, stems that are never separated give way to the whole mix,
/// shifted: the song goes on in the key, the drums with the rest.
#[test]
fn on_a_device_stems_that_do_not_come_give_way_to_the_whole_mix_in_the_key() {
    let _real_time = real_time();
    let dir = temp_dir("key-whole");
    let (path, _) = sine(&dir, "sine.wav", 6.0);
    let run = Run {
        separator: Some(never),
        make: MakeStems::Separate,
        sent: Vec::new(),
    };
    let options = PlayerOptions {
        offline: false,
        ..run.options()
    };
    let (target, captures, gate) = wide_target_on(caps_of(&[RATE], 2), SimClock::Paced);
    let (player, events) = Player::new(Box::new(target), options);
    player.make_stems(run.make).unwrap();
    gate.allow_all();
    player.load(vec![path]).unwrap();
    player.play(0).unwrap();
    collect_until(
        &events,
        |e| matches!(e, Event::Position(p) if p.output_frame > u64::from(RATE) / 2),
    );
    player.send(Command::Key { semitones: 2 }).unwrap();
    let mut seen = collect_until(&events, |e| matches!(e, Event::Position(p) if p.key == 2));
    let Some(Event::Position(heard)) = seen.last() else {
        unreachable!()
    };
    let at = heard.output_frame as usize;
    seen.extend(collect_until(
        &events,
        |e| matches!(e, Event::Position(p) if p.output_frame as usize > at + RATE as usize),
    ));
    drop(player);
    NEVER.allow(usize::MAX);
    let gave_way = seen
        .iter()
        .any(|e| matches!(e, Event::PartsStopped { reason, .. } if reason.contains("whole mix")));
    assert!(gave_way, "the stems gave way to the whole mix");
    let wide = Wide::of(&captures);
    let hz = peak_hz(&wide.main, 2, at + RATE as usize / 4);
    assert!((hz - UP_TWO).abs() < 1.0, "{hz} Hz");
}

/// While the key is shifted a jump is refused and new settings wait, each
/// saying why.
#[test]
fn shifted_a_jump_is_refused_and_new_settings_wait_saying_so() {
    let dir = temp_dir("key-refusals");
    let (path, _) = sine(&dir, "sine.wav", 3.0);
    let (player, events, _captures, gate) = offline_player(vec![path]);
    player.play(0).unwrap();
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    player.send(Command::Key { semitones: 2 }).unwrap();
    collect_until(&events, |e| matches!(e, Event::Key { semitones: 2 }));
    player.jump(0, 0.5, jodel_playback::JumpAt::Now).unwrap();
    let seen = collect_until(&events, |e| matches!(e, Event::Notice { .. }));
    let said = |seen: &[Event], what: &str| {
        seen.iter()
            .any(|e| matches!(e, Event::Notice { message, .. } if message.contains(what)))
    };
    assert!(said(&seen, "No jumps while the key is shifted"), "{seen:?}");
    let settings = Settings {
        headroom: jodel_playback::Headroom(10),
    };
    player.set_settings(settings).unwrap();
    let seen = collect_until(&events, |e| matches!(e, Event::Notice { .. }));
    assert!(said(&seen, "back in its own key"), "{seen:?}");
    gate.allow_all();
    until_queue_end(&events);
}

/// Two keys during a count-in (one set before START, then nudged while it
/// counts): the song is heard in the second from its first frame, where
/// the count-in ends, its positions as they were.
#[test]
fn two_keys_during_a_count_in_are_heard_from_the_songs_first_frame() {
    let dir = temp_dir("key-count-in-two");
    let (a, _) = sine(&dir, "a.wav", 2.0);
    let (player, events, captures, gate) = offline_player(vec![a]);
    player.play_in(0, 1.0).unwrap();
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    player.send(Command::Key { semitones: 2 }).unwrap();
    collect_until(&events, |e| matches!(e, Event::Key { semitones: 2 }));
    player.send(Command::Key { semitones: 3 }).unwrap();
    collect_until(&events, |e| matches!(e, Event::Key { semitones: 3 }));
    gate.allow_all();
    let seen = until_queue_end(&events);
    drop(player);
    let (start, keyed) = song_start(&seen, 3);
    assert!(keyed, "every position of the song in +3");
    let rate = f64::from(RATE);
    for e in &seen {
        if let Event::Position(p) = e {
            let want = p.output_frame as f64 / rate - 1.0;
            assert!(
                (p.seconds - want).abs() < 1e-3,
                "{} s heard at {}: the count-in is a second",
                p.seconds,
                p.output_frame
            );
        }
    }
    assert!(start >= RATE as usize, "the song from {start}");
    let wide = Wide::of(&captures);
    assert!(
        wide.main[..RATE as usize * 2 * 9 / 10]
            .iter()
            .all(|&s| s == 0),
        "silence through the count-in"
    );
    let hz = peak_hz(&wide.main, 2, start + RATE as usize / 10);
    assert!((hz - UP_THREE).abs() < 1.0, "{hz} Hz");
}
