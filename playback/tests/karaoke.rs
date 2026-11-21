//! What karaoke asks of the engine: a track started after a count-in of
//! silence ([`Command::PlayIn`]), heard from its end with positions
//! running negative through it, and holding at the end of each track
//! ([`Command::HoldAtEnd`]).

mod common;

use std::path::PathBuf;
use std::time::Duration;

use common::*;
use jodel_output::SampleFormat;
use jodel_playback::stems::{Separator, Stem, StemMix};
use jodel_playback::{Event, PlayState, Player, PlayerOptions, Settings};

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

/// Each stem a quarter of the mix.
struct Quarter;

impl Separator for Quarter {
    fn name(&self) -> String {
        "quarter".into()
    }

    fn masks(&self, frames: usize, stft: &[f32]) -> Result<Vec<f32>, String> {
        let per_stem = stft.len();
        assert_eq!(per_stem % frames, 0);
        Ok((0..4 * per_stem)
            .map(|i| if i % 2 == 0 { 0.25 } else { 0.0 })
            .collect())
    }
}

static QUARTER: Quarter = Quarter;

fn quarter() -> Option<&'static dyn Separator> {
    Some(&QUARTER)
}

/// The karaoke mix (the voice off) after a count-in: the stems are
/// separated while the silence plays, and follow it without a gap.
#[test]
fn a_count_in_into_stems_has_them_ready_at_its_end() {
    let dir = temp_dir("karaoke-lead-stems");
    let path = dir.join("tones.wav");
    let file = tones(RATE, 3 * RATE as usize);
    write_wav(&path, RATE, 2, &file);
    let (target, captures, gate) = GatedTarget::new(caps(&[RATE], &[SampleFormat::S24In32Low]));
    let options = PlayerOptions {
        settings: Settings {
            resample: false,
            stems: StemMix::ALL.with_level(Stem::Vocals, 0),
            ..Settings::default()
        },
        separator: Some(quarter),
        ..options()
    };
    let (player, events) = Player::new(Box::new(target), options);
    player.load(vec![path]).unwrap();
    // Half the silence taken, then the output full for a while: the
    // track's stems are made behind it.
    let lead = RATE as usize;
    gate.allow(lead as u64 / 2);
    player.play_in(0, 1.0).unwrap();
    gate.wait_written(lead as u64 / 2);
    std::thread::sleep(Duration::from_millis(300));
    gate.allow_all();
    let seen = until_queue_end(&events);
    drop(player);
    assert!(!seen
        .iter()
        .any(|e| matches!(e, Event::Error { .. } | Event::StemsStopped { .. })));
    let streams = captured(&captures);
    let got = samples(&streams[0].1);
    assert_eq!(got.len(), 2 * lead + file.len());
    assert!(got[..2 * lead].iter().all(|&s| s == 0));
    let worst = (2..file.len())
        .map(|i| (f64::from(got[2 * lead + i]) - 0.75 * f64::from(file[i])).abs())
        .fold(0.0, f64::max);
    assert!(worst <= 64.0, "{worst} steps off");
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
