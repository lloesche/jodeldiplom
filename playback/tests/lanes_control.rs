//! The control word of the stem parts ([`jodel_output::lanes::Parts`]):
//! none until the stems are separated, there from the first frame from a
//! store, quietly none for a track not kept, ended before they run out and
//! there again after, before a next track that has none, given back by a
//! jump with the promise kept, and offline the writing waits for them
//! without ever waiting for good.

mod common;

use std::path::Path;
use std::time::{Duration, Instant};

use common::lanes::*;
use common::*;
use jodel_output::lanes::ending_frames;
use jodel_output::{SampleFormat, SimClock};
use jodel_playback::{
    Event, JumpAt, MakeStems, MemoryTarget, Player, PlayerOptions, Settings, Source, StemStore,
};
use SampleFormat::*;

#[test]
fn the_parts_are_there_once_separated() {
    let dir = temp_dir("lanes-control");
    let file = tone_file(&dir, "tones.wav", RATE, 10 * RATE as usize);
    let (target, captures, gate) = wide_target(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options(false, false, quarter));
    player.make_stems(MakeStems::Separate).unwrap();
    player.load(vec![file]).unwrap();
    player.play(0).unwrap();
    pace_once_present(&gate, &events, &captures, 2_048);
    drop(player);
    let wide = Wide::of(&captures);
    assert_eq!(wide.control[0], 0, "none before the first stems");
    let first = wide.control.iter().position(|&c| c == 1).expect("parts");
    wide.keeps_the_promise(&[]).unwrap();
    let (worst, _) = wide.worst(first + 1..wide.frames());
    assert!(worst <= CLOSE, "{worst} steps off");
}

#[test]
fn kept_stems_are_there_from_the_first_frame() {
    let dir = temp_dir("lanes-kept");
    let samples = tones(RATE, 3 * RATE as usize);
    let file = dir.join("tones.wav");
    write_wav(&file, RATE, 2, &samples);
    let store = kept(&dir, &file, &samples);
    // Offline: there from the first frame.
    let options = |offline| PlayerOptions {
        stem_store: Some(store.clone()),
        ..options(false, offline, broken)
    };
    let (target, captures, gate) = wide_target(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options(true));
    player.make_stems(MakeStems::Kept).unwrap();
    player.load(vec![file.clone()]).unwrap();
    gate.allow_all();
    player.play(0).unwrap();
    let seen = until_queue_end(&events);
    drop(player);
    assert!(!seen.iter().any(|e| matches!(e, Event::PartsStopped { .. })));
    let wide = Wide::of(&captures);
    assert!(wide.control.iter().all(|&c| c == 1), "there from frame 0");
    // Kept at 16 bits a block: within a step of that, four times.
    let (worst, _) = wide.worst(0..wide.frames());
    assert!(worst <= 4 * 256, "{worst} steps off");
    // On a device: there as soon as half a second of them is read, and in
    // step.
    let (target, captures, gate) = wide_target(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options(false));
    player.make_stems(MakeStems::Kept).unwrap();
    player.load(vec![file]).unwrap();
    player.play(0).unwrap();
    pace_once_present(&gate, &events, &captures, 1_024);
    drop(player);
    let wide = Wide::of(&captures);
    wide.keeps_the_promise(&[]).unwrap();
    let first = wide.saying(1)[0];
    assert!(wide.control[first..].iter().all(|&c| c == 1));
    assert!(wide.worst(first..wide.frames()).0 <= 4 * 256);
}

/// Kept only: a track whose stems are not kept has none, quietly.
#[test]
fn a_track_not_kept_has_no_parts_and_nothing_is_said() {
    let dir = temp_dir("lanes-not-kept");
    let file = tone_file(&dir, "tones.wav", RATE, RATE as usize);
    let options = PlayerOptions {
        stem_store: Some(kept(&dir, &dir.join("other.wav"), &tones(RATE, 100))),
        ..options(false, true, quarter)
    };
    let (target, captures, gate) = wide_target(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options);
    player.make_stems(MakeStems::Kept).unwrap();
    player.load(vec![file]).unwrap();
    gate.allow_all();
    player.play(0).unwrap();
    let seen = until_queue_end(&events);
    drop(player);
    assert!(!seen.iter().any(|e| matches!(e, Event::PartsStopped { .. })));
    assert!(Wide::of(&captures).control.iter().all(|&c| c == 0));
}

/// Stems separated slower than they are played: they run out (ending over
/// a switch's length), are none, and are there again once they are ahead;
/// a jump back after they ran out finds them kept.
#[test]
fn parts_that_run_out_end_and_come_back() {
    let dir = temp_dir("lanes-run-out");
    let file = tone_file(&dir, "tones.wav", RATE, 16 * RATE as usize);
    RUN_OUT.allow(1);
    let (target, captures, gate) = wide_target(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options(false, false, run_out));
    player.make_stems(MakeStems::Separate).unwrap();
    player.jump_window(4.0, 0.0).unwrap();
    player.load(vec![file]).unwrap();
    player.play(0).unwrap();
    // The first chunk's stems only: they run out before 7 s.
    let mut seen = pace_to(&gate, &events, 512, 8 * u64::from(RATE));
    // Back to 5 s, then the rest separated.
    player.jump(0, 5.0, JumpAt::Now).unwrap();
    seen.extend(collect_until(&events, |e| {
        matches!(e, Event::Jumped { .. })
    }));
    let jumped = gate.written() as usize;
    RUN_OUT.allow(usize::MAX);
    seen.extend(pace(&gate, &events, 512));
    drop(player);
    let wide = Wide::of(&captures);
    wide.keeps_the_promise(&[]).unwrap();
    let present = wide.saying(1);
    let ending = wide.saying(2);
    assert!(
        !present.is_empty() && !ending.is_empty(),
        "there, then ending"
    );
    let first_end = ending[0];
    assert!(present[0] < first_end && first_end < jumped);
    let none_after = (first_end..wide.frames()).find(|&f| wide.control[f] == 0);
    let back = present.iter().find(|&&f| f > first_end).copied();
    assert!(
        none_after.is_some() && back.is_some(),
        "none, then there again"
    );
    assert!(none_after.unwrap() < back.unwrap());
    let (worst, _) = wide.worst(0..wide.frames());
    assert!(worst <= CLOSE, "{worst} steps off");
}

/// The next track's stems fail: this track's end over a switch's length
/// before its last frame, and the next has none.
#[test]
fn a_next_track_without_stems_has_this_ones_end_before_it() {
    let dir = temp_dir("lanes-next-fails");
    let one = tone_file(&dir, "one.wav", RATE, 2 * RATE as usize);
    let two = dir.join("two.wav");
    write_wav(&two, RATE, 2, &vec![0; 4 * RATE as usize]);
    let (wide, seen) = play_wide(
        caps(&[RATE], &[S24In32Low]),
        vec![one, two],
        options(false, true, fails_on_silence),
    );
    let end = 2 * RATE as usize;
    let e = ending_frames(RATE) as usize;
    assert!(wide.control[..end - e].iter().all(|&c| c == 1));
    assert!(wide.control[end - e..end].iter().all(|&c| c == 2));
    assert!(wide.control[end..].iter().all(|&c| c == 0));
    assert!(seen
        .iter()
        .any(|e| matches!(e, Event::PartsStopped { index: 1, .. })));
}

/// A file whose length is not stated: offline its parts are waited for
/// as it plays, and it plays to its end.
#[test]
fn a_file_of_no_stated_length_plays_to_its_end_offline() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/formats/lossy.mp2");
    let source = Source::open(&path).unwrap();
    let info = source.info().clone();
    assert_eq!(info.frames, None, "a file of no stated length");
    let (wide, _) = play_wide(
        caps_of(&[info.rate], info.channels),
        vec![path.clone()],
        options(true, true, quarter),
    );
    let frames = read_all(&path).len() / usize::from(info.channels);
    assert_eq!(wide.frames(), frames, "the whole file");
    assert!(!wide.saying(1).is_empty());
}

#[test]
fn a_failing_separator_stops_the_parts_and_says_so() {
    let dir = temp_dir("lanes-broken");
    let file = tone_file(&dir, "tones.wav", RATE, 2 * RATE as usize);
    let (wide, seen) = play_wide(
        caps(&[RATE], &[S24In32Low]),
        vec![file.clone()],
        options(false, true, broken),
    );
    assert!(wide.control.iter().all(|&c| c == 0));
    assert!(seen.iter().any(|e| matches!(
        e,
        Event::PartsStopped { reason, .. } if reason == "out of memory"
    )));
    assert!(wide.main == read_all(&file), "main is the file");
}

/// A store whose stems for a track run on past its end (kept for a longer
/// version), and the next track's: offline the next one's are waited for
/// without the first run ever being read to its end.
#[test]
fn stems_kept_longer_than_the_track_do_not_hold_the_next_one_up() {
    let dir = temp_dir("lanes-kept-longer");
    let one = dir.join("one.wav");
    write_wav(&one, RATE, 2, &tones(RATE, RATE as usize));
    let two = tone_file(&dir, "two.wav", RATE, RATE as usize);
    let long = dir.join("long");
    std::fs::create_dir_all(&long).unwrap();
    let longer = kept(&long, &one, &tones(RATE, 12 * RATE as usize));
    let next = kept(&dir, &two, &tones(RATE, RATE as usize));
    let store: std::sync::Arc<dyn StemStore> = std::sync::Arc::new(Both(longer, next));
    let options = PlayerOptions {
        stem_store: Some(store),
        ..options(false, true, broken)
    };
    let (target, captures, gate) = wide_target(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options);
    player.make_stems(MakeStems::Kept).unwrap();
    player.load(vec![one, two]).unwrap();
    gate.allow_all();
    player.play(0).unwrap();
    let seen = until_queue_end(&events);
    drop(player);
    let wide = Wide::of(&captures);
    wide.keeps_the_promise(&track_starts(&seen)).unwrap();
    assert_eq!(wide.frames(), 2 * RATE as usize);
    assert!(wide.control.iter().all(|&c| c == 1), "there across both");
}

/// The parts run out while the output holds what was written: a jump
/// gives back the frames that said they end, and the parts go on from the
/// stems kept, crossfaded, the promise kept.
#[test]
fn a_jump_taking_back_an_end_of_the_parts_keeps_the_promise() {
    let dir = temp_dir("lanes-take-back");
    let file = tone_file(&dir, "tones.wav", RATE, 12 * RATE as usize);
    TAKEN_BACK.allow(1);
    let device = MemoryTarget::with_lanes(caps(&[RATE], &[S24In32Low]), SimClock::Instant, 5);
    let captures = device.captures();
    let (target, hold) = HoldingTarget::around(Box::new(device), u64::from(RATE) / 2);
    let (player, events) = Player::new(Box::new(target), options(false, false, taken_back));
    player.make_stems(MakeStems::Separate).unwrap();
    player.jump_window(4.0, 0.0).unwrap();
    player.load(vec![file]).unwrap();
    player.play(0).unwrap();
    // Played on until the frames held say the parts end, well behind what
    // may still be played (what was let through) before the jump.
    let deadline = Instant::now() + TIMEOUT;
    let ending_held = |words: Vec<i32>| {
        words
            .iter()
            .position(|&w| w == 2)
            .is_some_and(|i| i > 2_048)
    };
    while !ending_held(hold.held_words()) {
        assert!(Instant::now() < deadline, "no end of the parts held");
        hold.play(256);
        std::thread::sleep(Duration::from_millis(2));
    }
    player.jump(0, 2.0, JumpAt::Now).unwrap();
    let mut seen = collect_until(&events, |e| matches!(e, Event::Jumped { .. }));
    // Everything held was given back: the jump is heard after what played.
    // (A lane frame here: eleven words of four bytes.)
    let jumped = captures.list()[0].sink.len() / (11 * 4);
    TAKEN_BACK.allow(usize::MAX);
    hold.play_all();
    seen.extend(until_queue_end(&events));
    drop(player);
    let wide = Wide::of(&captures);
    wide.keeps_the_promise(&[]).unwrap();
    // The end given back never played: the parts went on over the jump
    // (they may run out again later, ahead of the separation).
    assert!(
        !wide.control[..jumped].contains(&2),
        "the end given back played"
    );
    assert_eq!(wide.control[jumped - 1], 1);
    assert_eq!(wide.control[jumped], 1, "on over the jump");
    let (worst, there) = wide.worst(0..wide.frames());
    assert!(worst <= CLOSE, "{worst} steps off");
    assert!(there > wide.frames() / 2, "there after the jump");
    assert!(!seen.iter().any(|e| matches!(e, Event::PartsStopped { .. })));
}

/// A slow 48 kHz signal (100 Hz left, 150 Hz right, half scale), `frames`
/// long: a level step in it stands out from its own steps.
fn slow_tones(frames: usize) -> Vec<i32> {
    let w = |f: f64, n: usize| {
        ((2.0 * std::f64::consts::PI * f * n as f64 / 48_000.0).sin() * 4_194_304.0) as i32
    };
    (0..frames)
        .flat_map(|n| [w(100.0, n), w(150.0, n)])
        .collect()
}

/// Plays a 48 kHz file resampled to 44.1 kHz, then (new settings, half a
/// second in) on its own route at 48 kHz: the output opens again at the
/// other format, the old route fading out and the new one in.
fn switch_format(target: GatedTarget, gate: &Gate, path: &Path) -> Vec<Event> {
    let settings = |max_rate| Settings {
        max_rate,
        resample: true,
        ..Settings::default()
    };
    let options = PlayerOptions {
        settings: settings(Some(44_100)),
        ..options(true, true, quarter)
    };
    let (player, events) = Player::new(Box::new(target), options);
    player.make_stems(MakeStems::Separate).unwrap();
    player.load(vec![path.to_path_buf()]).unwrap();
    gate.allow(22_050);
    player.play(0).unwrap();
    gate.wait_written(22_050);
    let mut seen = collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    player.set_settings(settings(None)).unwrap();
    seen.extend(trickle_until_started(gate, &events));
    gate.allow_all();
    seen.extend(until_queue_end(&events));
    drop(player);
    seen
}

/// A route moved to another format mid-track: the parts end over a
/// mixer's switch's length before the old route fades out, are none over
/// the fade and over the new route's rise, and only then there again; on
/// the mixer with the drums muted, nothing jumps (only the fades and the
/// mixer's own switch move the level).
#[test]
fn a_switch_to_another_format_ends_the_parts_before_its_fades() {
    let dir = temp_dir("lanes-format");
    let path = dir.join("slow.wav");
    write_wav(&path, 48_000, 2, &slow_tones(2 * 48_000));
    let caps = caps(&[44_100, 48_000], &[S24In32Low]);
    // The lane streams as written.
    let device = MemoryTarget::with_lanes(caps.clone(), SimClock::Instant, 5);
    let captures = device.captures();
    let (target, gate) = GatedTarget::around(Box::new(device));
    switch_format(target, &gate, &path);
    assert_eq!(captures.list().len(), 2, "two streams");
    let (old, new) = (Wide::nth(&captures, 0), Wide::nth(&captures, 1));
    assert_eq!((old.rate, new.rate), (44_100, 48_000));
    old.keeps_the_promise(&[]).unwrap();
    new.keeps_the_promise(&[]).unwrap();
    // The old route: there, ending, then none over its 20 ms fade.
    let (n, fade, e) = (old.frames(), 882, ending_frames(44_100) as usize);
    assert!(old.control[..n - fade - e].contains(&1));
    assert!(old.control[n - fade - e..n - fade].iter().all(|&c| c == 2));
    assert!(old.control[n - fade..].iter().all(|&c| c == 0));
    // The new one: none over its rise, then there.
    let rise = 960;
    assert!(new.control[..rise].iter().all(|&c| c == 0));
    let back = new.saying(1);
    assert_eq!(back.first(), Some(&rise), "there from the end of the rise");
    // (A 48 kHz file's stems go through two filters: where the file cuts
    // out, its last 5 ms, they ring apart from main.)
    let (worst, _) = new.worst(back[0]..new.frames() - 240);
    assert!(worst <= CLOSE, "{worst} steps off after the switch");
    // Through the mixer, the drums muted.
    let (target, captures, gate, lanes) = gated_mixer_lanes(caps);
    lanes.parts[0].set_mute(true);
    let seen = switch_format(target, &gate, &path);
    assert!(!seen.iter().any(|e| matches!(e, Event::Error { .. })));
    drop(lanes);
    let streams = captured(&captures);
    assert_eq!(streams.len(), 2, "two streams");
    // The left channel's 100 Hz moves at most this much a frame; a step of
    // the level (three quarters to all, all to nothing) at once would be
    // over ten times that.
    let most = 2.0 * std::f64::consts::PI * 100.0 / 44_100.0 * 4_194_304.0;
    for (format, bytes) in &streams {
        let got: Vec<i32> = bytes
            .chunks_exact(4)
            .map(|b| (i32::from_le_bytes([b[0], b[1], b[2], b[3]]) << 8) >> 8)
            .collect();
        let left: Vec<i32> = got.iter().step_by(2).copied().collect();
        let step = left
            .windows(2)
            .map(|w| f64::from((w[1] - w[0]).abs()))
            .fold(0.0, f64::max);
        assert!(
            step <= 1.5 * most,
            "{format}: a step of {step} (at most {most})"
        );
    }
    // Three quarters of the file before the switch and after it.
    let left = |i: usize| {
        let b = &streams[1].1[8 * i..8 * i + 4];
        (i32::from_le_bytes([b[0], b[1], b[2], b[3]]) << 8) >> 8
    };
    let first = streams[0].1.len() / 8;
    let from = (first as f64 * 48_000.0 / 44_100.0).round() as usize;
    let file = slow_tones(2 * 48_000);
    let off = (2_000..streams[1].1.len() / 8 - 240)
        .map(|i| (f64::from(left(i)) - 0.75 * f64::from(file[2 * (from + i)])).abs())
        .fold(0.0, f64::max);
    assert!(
        off <= 2.0 * CLOSE as f64,
        "{off} steps off three quarters after the switch"
    );
}
