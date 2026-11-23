//! What karaoke asks of the engine: a track started after a count-in of
//! silence ([`Command::PlayIn`]), heard from its end with positions
//! running negative through it, the voice muted on the desk after it, and
//! holding at the end of each track ([`Command::HoldAtEnd`]).

mod common;

use std::path::PathBuf;
use std::time::Duration;

use common::lanes::quarter;
use common::*;
use jodel_output::mixer::{Mixer, MixerOptions, Role};
use jodel_output::{SampleFormat, SimClock};
use jodel_playback::stems::Stem;
use jodel_playback::{
    Event, MakeStems, MemoryTarget, PlayState, Player, PlayerOptions, Settings, StemDesk,
};

const RATE: u32 = 44_100;

fn options() -> PlayerOptions {
    PlayerOptions {
        position_interval: Duration::from_millis(1),
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

/// `n` tracks of a second each, as WAV ramps.
fn tracks(dir: &TempDir, n: usize) -> (Vec<PathBuf>, Vec<Vec<i32>>) {
    (0..n)
        .map(|i| {
            let path = dir.join(format!("{i}.wav"));
            let samples = ramp_wav(&path, RATE, RATE);
            (path, samples)
        })
        .unzip()
}

#[test]
fn a_count_in_is_silence_then_the_track_heard_from_its_end() {
    let dir = temp_dir("karaoke-lead");
    let (paths, files) = tracks(&dir, 1);
    let (target, captures, gate) = GatedTarget::new(caps(&[RATE], &[SampleFormat::S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options());
    player.load(paths).unwrap();
    // Three quarters of a second of silence, a quarter of it heard first.
    let lead = (0.75 * f64::from(RATE)) as u64;
    gate.allow(lead / 3);
    player.play_in(0, 0.75).unwrap();
    gate.wait_written(lead / 3);
    let early = collect_until(
        &events,
        |e| matches!(e, Event::Position(p) if p.seconds < -0.3),
    );
    // The track starts to be heard as the silence does, from its end.
    let start = early
        .iter()
        .find_map(|e| match e {
            Event::TrackStarted(s) => Some(s.output_frame),
            _ => None,
        })
        .expect("started");
    assert_eq!(start, lead);
    let Some(Event::Position(p)) = early.last() else {
        panic!("a position");
    };
    // Seconds to go: what is left of the silence.
    let left = (lead - p.output_frame) as f64 / f64::from(RATE);
    assert!((p.seconds + left).abs() < 1e-9, "{} vs -{left}", p.seconds);
    assert_eq!(p.frame, 0);
    gate.allow_all();
    let seen = until_queue_end(&events);
    drop(player);
    // Positions run on through zero into the track.
    assert!(seen
        .iter()
        .any(|e| matches!(e, Event::Position(p) if p.seconds > 0.5)));
    let streams = captured(&captures);
    assert_eq!(streams.len(), 1);
    let got = samples(&streams[0].1);
    let lead = lead as usize;
    assert_eq!(got.len(), 2 * lead + files[0].len());
    assert!(got[..2 * lead].iter().all(|&s| s == 0), "silence first");
    assert_eq!(&got[2 * lead..], &files[0][..], "then the track as it is");
}

/// The karaoke mix after a count-in, the voice muted on its strip of a
/// mixer: the count-in's silence on every lane, then the rest of the band
/// (three quarters of the file, from the stems made beside it) from the
/// track's first frames, after the mixer's switch.
#[test]
fn a_count_in_into_the_voice_muted_on_the_desk_plays_the_band_after_it() {
    // It separates stems: not beside the test in real time.
    let _guard = heavy();
    let dir = temp_dir("karaoke-lead-stems");
    let path = dir.join("tones.wav");
    let file = tones(RATE, 3 * RATE as usize);
    write_wav(&path, RATE, 2, &file);
    let (target, captures, gate, lanes) =
        gated_mixer_lanes(caps(&[RATE], &[SampleFormat::S24In32Low]));
    let options = PlayerOptions {
        settings: Settings {
            resample: false,
            ..Settings::default()
        },
        separator: Some(quarter),
        // The parts waited for: there from the track's first frame.
        offline: true,
        ..options()
    };
    let (player, events) = Player::new(Box::new(target), options);
    player.make_stems(MakeStems::Separate).unwrap();
    // The vocals' strip (drums, bass, vocals, other).
    lanes.parts[2].set_mute(true);
    player.load(vec![path]).unwrap();
    gate.allow_all();
    player.play_in(0, 1.0).unwrap();
    let seen = until_queue_end(&events);
    drop(player);
    drop(lanes);
    assert!(!seen
        .iter()
        .any(|e| matches!(e, Event::Error { .. } | Event::PartsStopped { .. })));
    let streams = captured(&captures);
    assert_eq!(streams.len(), 1);
    let got = samples(&streams[0].1);
    let lead = RATE as usize;
    assert_eq!(got.len(), 2 * lead + file.len());
    assert!(got[..2 * lead].iter().all(|&s| s == 0), "silence first");
    // From the mixer's switch (10 ms) on.
    let switched = RATE as usize / 100 + 8;
    let worst = (2 * switched..file.len())
        .map(|i| (f64::from(got[2 * lead + i]) - 0.75 * f64::from(file[i])).abs())
        .fold(0.0, f64::max);
    assert!(worst <= 64.0, "{worst} steps off");
}

/// The same on a device playing in real time: the voice's strip muted, the
/// stems are made while the count-in plays (two seconds of it: time for a
/// slow machine to separate the first stretch), so the band (not the file)
/// is heard from the track's first frames on, after the mixer's switch.
///
/// What is held to: the first stretch of the track, as much as the parts
/// made ahead of it in the count-in cover. A machine too slow to play in
/// real time (a small runner, a debug build, tests side by side) leaves the
/// device short now and then, whether stems are made or not: the mixer pads
/// those periods with silence, and they are left out here (nothing is lost
/// with them, the track goes on after). Later, such a machine may let the
/// stems fall behind, and the file is heard where they did: that is not
/// what this test is about.
#[test]
fn in_real_time_a_count_in_has_the_band_ready_at_its_end() {
    let _guard = real_time();
    let dir = temp_dir("karaoke-lead-stems-real");
    let path = dir.join("tones.wav");
    // The tones, the left a little off zero: no frame of it is silent (the
    // tones alone are, every 10 ms, where they could not be told from the
    // device's padding).
    let file: Vec<i32> = tones(RATE, 3 * RATE as usize)
        .chunks_exact(2)
        .flat_map(|f| [f[0] + 100_000, f[1]])
        .collect();
    write_wav(&path, RATE, 2, &file);
    let device = MemoryTarget::new(
        caps(&[RATE], &[SampleFormat::S24In32Low]),
        SimClock::Realtime,
    );
    let captures = device.captures();
    let mixer = Mixer::open(Box::new(device), MixerOptions::default());
    let (desk, head) = StemDesk::new(&mixer, "Player", Role::Lead, 1.0);
    desk.stem(Stem::Vocals).set_mute(true);
    let options = PlayerOptions {
        settings: Settings {
            resample: false,
            ..Settings::default()
        },
        separator: Some(quarter),
        ..options()
    };
    let (player, events) = Player::new(head.target, options);
    player.make_stems(MakeStems::Separate).unwrap();
    player.load(vec![path]).unwrap();
    player.play_in(0, 2.0).unwrap();
    let seen = until_queue_end(&events);
    drop(player);
    drop(desk);
    drop(mixer);
    assert!(
        !seen
            .iter()
            .any(|e| matches!(e, Event::Error { .. } | Event::PartsStopped { .. })),
        "{seen:?}"
    );
    let streams = captured(&captures);
    let got = samples(&streams[0].1);
    // Frames of the track held to (0.4 s): within the half second of parts
    // made ahead before it starts.
    let held = 4 * RATE as usize / 10;
    // The frames of a track that starts at output frame `start` as heard,
    // the device's padding left out, from the mixer's switch (10 ms) on, to
    // `held`: the largest distance from the band (three quarters of the
    // file).
    let switched = RATE as usize / 100 + 8;
    let silent = |f: usize| got.get(2 * f..2 * f + 2) == Some(&[0, 0][..]);
    let off = |start: usize| {
        let (mut f, mut k, mut worst) = (start, 0, 0.0f64);
        while k < held && 2 * f + 1 < got.len() {
            // A run of silence (more than a zero crossing): padding.
            if (0..3).all(|d| silent(f + d)) {
                while silent(f) {
                    f += 1;
                }
                continue;
            }
            if k >= switched {
                for c in 0..2 {
                    let want = 0.75 * f64::from(file[2 * k + c]);
                    worst = worst.max((f64::from(got[2 * f + c]) - want).abs());
                }
            }
            (f, k) = (f + 1, k + 1);
        }
        assert_eq!(k, held, "the track's first frames were played");
        worst
    };
    // The track starts about where the sound does.
    let sound = got.iter().position(|&s| s != 0).expect("sound") / 2;
    let worst = (sound.saturating_sub(2)..sound + 3)
        .map(off)
        .fold(f64::INFINITY, f64::min);
    assert!(worst <= 64.0, "{worst} steps off the band");
}

#[test]
fn holding_stops_after_a_track_and_names_the_next() {
    let dir = temp_dir("karaoke-hold");
    let (paths, files) = tracks(&dir, 3);
    let (target, captures, gate) = GatedTarget::new(caps(&[RATE], &[SampleFormat::S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options());
    player.load(paths).unwrap();
    player.hold_at_end(true).unwrap();
    gate.allow_all();
    player.play(0).unwrap();
    let seen = collect_until(&events, |e| matches!(e, Event::State(PlayState::Stopped)));
    assert!(seen.iter().any(|e| matches!(
        e,
        Event::TrackEnded {
            index: 0,
            finished: true
        }
    )));
    let held: Vec<usize> = seen
        .iter()
        .filter_map(|e| match e {
            Event::Held { index } => Some(*index),
            _ => None,
        })
        .collect();
    assert_eq!(held, [1]);
    assert!(!seen
        .iter()
        .any(|e| matches!(e, Event::TrackStarted(s) if s.index != 0)));
    // Started with a count-in, the next one plays, and holds again.
    player.play_in(1, 0.1).unwrap();
    let seen = collect_until(&events, |e| matches!(e, Event::Held { .. }));
    assert!(seen.iter().any(|e| matches!(e, Event::Held { index: 2 })));
    // The last one ends the queue as usual.
    player.play(2).unwrap();
    until_queue_end(&events);
    drop(player);
    let streams = captured(&captures);
    let lengths: Vec<usize> = streams.iter().map(|(_, b)| samples(b).len()).collect();
    let lead = 2 * (0.1 * f64::from(RATE)).round() as usize;
    assert_eq!(
        lengths,
        [files[0].len(), lead + files[1].len(), files[2].len()]
    );
}
