//! The stems as parts of the output ([`Command::MakeStems`],
//! [`jodel_output::lanes`]): main stays the file bit for bit through a
//! mixer's lane input with stems made beside it; the parts add up to main
//! at every output rate; the control word says when they are there; seeks,
//! jumps, the next track and a failing separator keep them in step; mono
//! and multichannel files have them in their channels; they end with the
//! track; and a part strip touched on the mixer is heard.

mod common;

use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use common::lanes::*;
use common::*;
use jodel_output::lanes::Parts;
use jodel_output::{Capabilities, SampleFormat};
use jodel_playback::{
    Captures, Command, Event, JumpAt, MakeStems, PlayState, Player, PlayerOptions,
};
use SampleFormat::*;

// ---- 1. The bit-perfect proof -------------------------------------------

type Script<'a> = &'a dyn Fn(&Player, &Receiver<Event>, &Gate, &Captures);

/// What the device gets from `script` directly, and through a lane input
/// with the quarter's stems made beside main: the same bytes. Offline, so
/// the parts are there on every frame (waited for), untouched.
fn same_through_lanes(caps: Capabilities, options: PlayerOptions, script: Script) {
    let direct = {
        let (target, captures, gate) = GatedTarget::new(caps.clone());
        let (player, events) = Player::new(Box::new(target), options.clone());
        script(&player, &events, &gate, &captures);
        drop(player);
        captured(&captures)
    };
    let laned = {
        let (target, captures, gate, lanes) = gated_mixer_lanes(caps);
        let (player, events) = Player::new(Box::new(target), options);
        player.make_stems(MakeStems::Separate).unwrap();
        script(&player, &events, &gate, &captures);
        // The parts were there to the last frame mixed, and not heard.
        let desk = lanes.mixer.desk();
        let input = &desk.strips[0].inputs[0];
        let parts = input.parts.as_ref().expect("an input of lanes");
        assert_eq!((parts.state, parts.share), (Parts::Present, 0.0));
        drop(player);
        drop(lanes);
        captured(&captures)
    };
    assert!(!direct.is_empty() && direct.iter().all(|(_, b)| !b.is_empty()));
    assert_eq!(direct.len(), laned.len(), "streams");
    for ((format, a), (format_laned, b)) in direct.iter().zip(&laned) {
        assert_eq!(format, format_laned);
        assert!(
            a == b,
            "{format}: {} bytes directly, {} through the lanes",
            a.len(),
            b.len()
        );
    }
}

fn to_the_end(files: Vec<PathBuf>) -> impl Fn(&Player, &Receiver<Event>, &Gate, &Captures) {
    move |player, events, gate, _| {
        gate.allow_all();
        player.load(files.clone()).unwrap();
        player.play(0).unwrap();
        until_queue_end(events);
    }
}

#[test]
fn main_through_a_lane_input_is_bit_exact_with_stems_made() {
    let exact = options(false, true, quarter);
    let resampled = options(true, true, quarter);
    let cases = [
        (caps(&[44_100], &[S24In32Low]), vec![F44], &exact),
        (caps(&[48_000], &[S16, S24Packed]), vec![F48], &exact),
        (caps(&[44_100], &[S24In32Low]), vec![F44, F44], &exact),
        (caps(&[48_000], &[S24In32Low]), vec![F44, F44], &resampled),
        (
            caps(&[44_100, 48_000], &[S24In32Low]),
            vec![F44, F48],
            &exact,
        ),
    ];
    for (caps, files, options) in cases {
        let files = files.into_iter().map(fixture).collect();
        same_through_lanes(caps, options.clone(), &to_the_end(files));
    }
    let dir = temp_dir("lanes-pcm");
    let (w48, w44) = (dir.join("48.wav"), dir.join("44.wav"));
    ramp_wav(&w48, 48_000, 30_000);
    ramp_wav(&w44, 44_100, 30_000);
    same_through_lanes(
        caps(&[48_000], &[S16, S24In32Low]),
        options(false, true, quarter),
        &to_the_end(vec![w48]),
    );
    same_through_lanes(
        caps(&[48_000], &[S24In32]),
        options(true, true, quarter),
        &to_the_end(vec![w44]),
    );
    // A share of the length in.
    let wav = dir.join("third.wav");
    ramp_wav(&wav, 44_100, 44_101);
    let play_at = |player: &Player, events: &Receiver<Event>, gate: &Gate, _: &Captures| {
        gate.allow_all();
        player.load(vec![wav.clone()]).unwrap();
        let share = 1.0 / 3.0;
        player.send(Command::PlayAt { index: 0, share }).unwrap();
        until_queue_end(events);
    };
    same_through_lanes(
        caps(&[44_100], &[S24In32Low]),
        options(false, true, quarter),
        &play_at,
    );
}

#[test]
fn seek_pause_and_jump_through_a_lane_input_are_bit_exact() {
    let dir = temp_dir("lanes-moves");
    let wav = dir.join("ramp.wav");
    ramp_wav(&wav, 44_100, 3 * 44_100);
    let up_to = |gate: &Gate, captures: &Captures, frames: u64| {
        gate.allow(frames);
        gate.wait_written(frames);
        wait_captured(captures, frames);
    };
    let script = |player: &Player, events: &Receiver<Event>, gate: &Gate, captures: &Captures| {
        player.load(vec![wav.clone()]).unwrap();
        gate.allow(45 * 441);
        player.play(0).unwrap();
        up_to(gate, captures, 45 * 441);
        player.pause().unwrap();
        collect_until(events, |e| matches!(e, Event::State(PlayState::Paused)));
        player.resume().unwrap();
        collect_until(events, |e| matches!(e, Event::State(PlayState::Playing)));
        up_to(gate, captures, 90 * 441);
        player.seek(2.0).unwrap();
        collect_until(events, |e| matches!(e, Event::Seeked { .. }));
        up_to(gate, captures, 136 * 441);
        player.jump(0, 0.5, JumpAt::Now).unwrap();
        // Taken before anything more is let through: the rest of the
        // packet under way is not to go out on one run and not the other.
        collect_until(events, |e| matches!(e, Event::Jumped { .. }));
        gate.allow_all();
        until_queue_end(events);
    };
    same_through_lanes(
        caps(&[44_100], &[S24In32Low]),
        options(false, true, quarter),
        &script,
    );
    let script = |player: &Player, events: &Receiver<Event>, gate: &Gate, captures: &Captures| {
        player.load(vec![fixture(F44)]).unwrap();
        gate.allow(11 * 441);
        player.play(0).unwrap();
        up_to(gate, captures, 11 * 441);
        player.seek(3_000.0 / 44_100.0).unwrap();
        collect_until(events, |e| matches!(e, Event::Seeked { .. }));
        gate.allow_all();
        until_queue_end(events);
    };
    same_through_lanes(
        caps(&[44_100], &[S24In32Low]),
        options(false, true, quarter),
        &script,
    );
}

#[test]
fn the_parts_add_up_to_main_at_every_rate() {
    let dir = temp_dir("lanes-sum");
    let file = tone_file(&dir, "tones.wav", RATE, 2 * RATE as usize);
    for out in [44_100, 48_000, 96_000, 192_000] {
        let (wide, _) = play_wide(
            caps(&[out], &[S24In32Low]),
            vec![file.clone()],
            options(true, true, quarter),
        );
        let n = wide.frames();
        assert_eq!(n as u64, output_frames(RATE, out, 2 * u64::from(RATE)));
        assert!(wide.control.iter().all(|&c| c == 1), "{out}: all there");
        // The first frames of a separation fade in.
        let (worst, there) = wide.worst(out as usize / 1_000..n);
        println!("{out} Hz: parts within {worst} steps of main");
        assert!(worst <= CLOSE, "{out} Hz: {worst} steps off");
        assert!(there > n * 9 / 10);
    }
}

// ---- 3. The control word ---------------------------------------------------

// ---- 4. Seeks, jumps, the next track, a failing separator -----------------

#[test]
fn after_a_seek_the_parts_start_from_its_frame() {
    let dir = temp_dir("lanes-seek");
    let file = tone_file(&dir, "tones.wav", RATE, 3 * RATE as usize);
    for (out, window) in [(44_100, true), (48_000, true), (48_000, false)] {
        let (target, captures, gate) = wide_target(caps(&[out], &[S24In32Low]));
        let (player, events) = Player::new(Box::new(target), options(true, true, quarter));
        player.make_stems(MakeStems::Separate).unwrap();
        if window {
            player.jump_window(4.0, 4.0).unwrap();
        }
        player.load(vec![file.clone()]).unwrap();
        let at = u64::from(out) / 2;
        gate.allow(at);
        player.play(0).unwrap();
        gate.wait_written(at);
        player.seek(2.0).unwrap();
        collect_until(&events, |e| matches!(e, Event::Seeked { .. }));
        gate.allow_all();
        until_queue_end(&events);
        drop(player);
        let wide = Wide::of(&captures);
        let at = at as usize;
        wide.keeps_the_promise(&[at]).unwrap();
        assert!(wide.control.iter().all(|&c| c == 1), "{out}: there");
        // A separation fades in on its first frame: a new run's are left
        // out; the stems kept start on the seek's very frame, as main does.
        let skip = out as usize / 1_000;
        let after = if window { at } else { at + skip };
        let (before, _) = wide.worst(skip..at);
        let (after, _) = wide.worst(after..wide.frames());
        println!("{out} Hz, window {window}: {before} steps before the seek, {after} after");
        assert!(before.max(after) <= CLOSE, "{out}: {before}, {after}");
    }
}

#[test]
fn after_a_jump_the_parts_are_crossfaded_and_in_step() {
    let dir = temp_dir("lanes-jump");
    let file = tone_file(&dir, "tones.wav", RATE, 2 * RATE as usize);
    // At a frame every 397 (less than a crossfade's 882 less a switch's
    // 441) over more than a piece of parts (4,096): wherever the parts
    // made ahead end, what would have played next is there for the
    // crossfade. At 48 kHz, once.
    let at = |k: u32| 1.0 + f64::from(k * 397) / f64::from(RATE);
    let runs = (0..12).map(|k| (44_100, k)).chain([(48_000, 0)]);
    for (out, k) in runs {
        let (target, captures, gate) = wide_target(caps(&[out], &[S24In32Low]));
        let (player, events) = Player::new(Box::new(target), options(true, true, quarter));
        player.make_stems(MakeStems::Separate).unwrap();
        player.jump_window(4.0, 4.0).unwrap();
        player.load(vec![file.clone()]).unwrap();
        gate.allow(u64::from(out) / 2);
        player.play(0).unwrap();
        gate.wait_written(u64::from(out) / 2);
        player.jump(0, 0.25, JumpAt::At(at(k))).unwrap();
        gate.allow_all();
        let seen = until_queue_end(&events);
        drop(player);
        assert!(seen.iter().any(|e| matches!(e, Event::Jumped { .. })));
        let wide = Wide::of(&captures);
        wide.keeps_the_promise(&[]).unwrap();
        let at = (at(k) * f64::from(out)).round() as usize;
        let fade = out as usize / 50;
        let (seam, _) = wide.worst(at - 10..at + fade + 10);
        let (after, there) = wide.worst(at + fade..wide.frames());
        let (all, _) = wide.worst(out as usize / 1_000..wide.frames());
        println!("{out} Hz, at {at}: {seam} steps over the jump's crossfade, {after} after it");
        assert!(
            there > (wide.frames() - at) * 9 / 10,
            "{out}: there after it"
        );
        assert!(all <= CLOSE, "{out}: {all} steps off");
    }
}

#[test]
fn a_jump_on_a_96_khz_file_played_at_48_keeps_the_parts_in_step() {
    let dir = temp_dir("lanes-jump-96");
    let file = tone_file(&dir, "tones.wav", 96_000, 3 * 96_000);
    let (target, captures, gate) = wide_target(caps(&[48_000], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options(true, true, quarter));
    player.make_stems(MakeStems::Separate).unwrap();
    player.jump_window(4.0, 4.0).unwrap();
    player.load(vec![file]).unwrap();
    gate.allow(24_000);
    player.play(0).unwrap();
    gate.wait_written(24_000);
    player.jump(0, 0.25, JumpAt::At(1.0)).unwrap();
    gate.allow_all();
    let seen = until_queue_end(&events);
    drop(player);
    assert!(seen.iter().any(|e| matches!(e, Event::Jumped { .. })));
    let wide = Wide::of(&captures);
    wide.keeps_the_promise(&[]).unwrap();
    assert!(wide.control.iter().all(|&c| c == 1));
    // The stems of a file at another rate than the separator's go through
    // two filters (to it and from it), main through one: where the file
    // cuts in and out (its first and last 4 ms here) they ring apart.
    let edge = 192;
    let (worst, _) = wide.worst(edge..wide.frames() - edge);
    println!("96 kHz at 48: {worst} steps off over the jump");
    assert!(worst <= CLOSE, "{worst} steps off");
}

#[test]
fn the_next_track_follows_with_its_parts() {
    let dir = temp_dir("lanes-next");
    let one = tone_file(&dir, "one.wav", RATE, 2 * RATE as usize);
    let two = tone_file(&dir, "two.wav", RATE, 2 * RATE as usize);
    let (wide, seen) = play_wide(
        caps(&[48_000], &[S24In32Low]),
        vec![one, two],
        options(true, true, quarter),
    );
    let starts = seen
        .iter()
        .filter(|e| matches!(e, Event::TrackStarted(_)))
        .count();
    assert_eq!(starts, 2);
    let end = output_frames(RATE, 48_000, 2 * u64::from(RATE)) as usize;
    assert_eq!(wide.frames(), 2 * end, "gapless");
    assert!(
        wide.control.iter().all(|&c| c == 1),
        "there across the change"
    );
    let (worst, _) = wide.worst(48..wide.frames());
    assert!(worst <= CLOSE, "{worst} steps off");
}

// ---- 5. Mono and multichannel ----------------------------------------------

#[test]
fn mono_and_six_channel_files_have_parts_in_their_channels() {
    let dir = temp_dir("lanes-channels");
    let stereo = tones(RATE, RATE as usize);
    for channels in [1u16, 6] {
        let ch = usize::from(channels);
        let samples: Vec<i32> = stereo
            .chunks_exact(2)
            .flat_map(|f| (0..ch).map(move |c| if c < 2 { f[c] } else { f[0] / 3 }))
            .collect();
        let path = dir.join(format!("{channels}.wav"));
        write_wav(&path, RATE, channels, &samples);
        let (wide, _) = play_wide(
            caps_of(&[RATE], channels),
            vec![path],
            options(false, true, quarter),
        );
        assert_eq!(wide.channels, ch);
        assert!(wide.control.iter().all(|&c| c == 1));
        for f in 0..wide.frames() {
            for c in 2..ch {
                assert!(wide.parts.iter().all(|p| p[f * ch + c] == 0), "{ch}: rest");
            }
        }
        // The front pair (or the one channel) adds up.
        let front = ch.min(2);
        let mut worst = 0i64;
        for f in 50..wide.frames() {
            for c in 0..front {
                let i = f * ch + c;
                let sum: i64 = wide.parts.iter().map(|p| i64::from(p[i])).sum();
                worst = worst.max((sum - i64::from(wide.main[i])).abs());
            }
        }
        assert!(worst <= CLOSE, "{channels} channels: {worst} steps off");
    }
}

// ---- 6. The end -------------------------------------------------------------

#[test]
fn the_parts_end_with_the_track() {
    let dir = temp_dir("lanes-end");
    for (frames, out) in [(RATE as usize + 7, 192_000), (RATE as usize + 3, 48_000)] {
        let file = tone_file(&dir, &format!("{frames}-{out}.wav"), RATE, frames);
        let (wide, _) = play_wide(
            caps(&[out], &[S24In32Low]),
            vec![file],
            options(true, true, quarter),
        );
        let n = output_frames(RATE, out, frames as u64) as usize;
        assert_eq!(wide.frames(), n);
        assert!(wide.control.iter().all(|&c| c == 1), "to the last frame");
        let (worst, _) = wide.worst(n - out as usize / 100..n);
        assert!(worst <= CLOSE, "{out}: {worst} steps off at the end");
    }
}

// ---- 7. Through the mixer -------------------------------------------------

#[test]
fn a_part_touched_on_the_mixer_is_heard_and_untouched_is_main_again() {
    let dir = temp_dir("lanes-mixer");
    let samples = tones(RATE, 3 * RATE as usize);
    let file = dir.join("tones.wav");
    write_wav(&file, RATE, 2, &samples);
    let (target, captures, gate, lanes) = gated_mixer_lanes(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options(false, true, quarter));
    player.make_stems(MakeStems::Separate).unwrap();
    // The drums muted, then not after a second.
    lanes.parts[0].set_mute(true);
    player.load(vec![file]).unwrap();
    gate.allow(u64::from(RATE));
    player.play(0).unwrap();
    gate.wait_written(u64::from(RATE));
    // (The mixer's limiter holds 2 ms back while it mixes.)
    wait_captured(&captures, u64::from(RATE) - 441);
    lanes.parts[0].set_mute(false);
    gate.allow_all();
    until_queue_end(&events);
    drop(player);
    drop(lanes);
    let streams = captured(&captures);
    assert_eq!(streams.len(), 1);
    let got = device_samples(&streams[0].1);
    assert_eq!(got.len(), samples.len());
    // Three quarters of the file while the drums are off.
    let off = (100..RATE as usize - 1_000)
        .flat_map(|f| [2 * f, 2 * f + 1])
        .map(|i| (f64::from(got[i]) - 0.75 * f64::from(samples[i])).abs())
        .fold(0.0, f64::max);
    assert!(off <= CLOSE as f64, "{off} steps off three quarters");
    // The file again, bit for bit, once back.
    let back = 2 * (RATE as usize + RATE as usize / 2);
    assert!(got[back..] == samples[back..], "main again");
}
