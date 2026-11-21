//! Stem playback through the engine with the separator on the GPU: kicks
//! over a held chord, played with the drums only and with the other stem
//! only. Skipped without a GPU.

use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use jodel_gpu::GpuSeparator;
use jodel_output::{Capabilities, SampleFormat, SimClock, StreamFormat};
use jodel_playback::stems::{Separator, Stem, StemMix};
use jodel_playback::{Event, MemoryTarget, Player, PlayerOptions, Settings, Stage};

const RATE: u32 = 44_100;
const SECONDS: usize = 8;

fn gpu() -> Option<&'static dyn Separator> {
    GpuSeparator::shared().map(|g| g as &dyn Separator)
}

/// A kick every half second over a held A major chord, 24-bit stereo.
fn kicks_and_chord() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("jodel-gpu-playback-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("kicks.wav");
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: RATE,
        bits_per_sample: 24,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec).unwrap();
    for n in 0..SECONDS * RATE as usize {
        let t = n as f64 / f64::from(RATE);
        let k = t % 0.5;
        let kick = 0.6 * (2.0 * PI * 55.0 * k).sin() * (-k / 0.06).exp();
        let chord: f64 = [220.0, 277.18, 329.63]
            .iter()
            .map(|f| 0.1 * (2.0 * PI * f * t).sin())
            .sum();
        let v = ((kick + chord) * 8_388_607.0) as i32;
        writer.write_sample(v).unwrap();
        writer.write_sample(v).unwrap();
    }
    writer.finalize().unwrap();
    path
}

/// The left channel of a played stream (`S24In32Low`), full scale ±1.
fn played(path: &Path, stems: StemMix) -> Vec<f64> {
    let mut caps = Capabilities::new();
    caps.insert(StreamFormat::new(RATE, 2, SampleFormat::S24In32Low));
    let target = MemoryTarget::new(caps, SimClock::Instant);
    let captures = target.captures();
    let options = PlayerOptions {
        settings: Settings {
            stems,
            ..Settings::default()
        },
        separator: Some(gpu),
        ..PlayerOptions::default()
    };
    let (player, events) = Player::new(Box::new(target), options);
    player.load(vec![path.to_path_buf()]).unwrap();
    player.play(0).unwrap();
    loop {
        match events.recv_timeout(Duration::from_secs(60)).unwrap() {
            Event::TrackStarted(s) => assert_eq!(s.route.stage, Stage::Stems),
            Event::Error { error, .. } => panic!("{error}"),
            Event::Notice { message, .. } => panic!("{message}"),
            Event::StemsStopped { reason, .. } => panic!("the stems stopped: {reason}"),
            Event::QueueEnded => break,
            _ => {}
        }
    }
    drop(player);
    let bytes = captures.list().remove(0).sink.bytes();
    bytes
        .chunks_exact(8)
        .map(|b| f64::from((i32::from_le_bytes([b[0], b[1], b[2], b[3]]) << 8) >> 8) / 8_388_608.0)
        .collect()
}

/// The energy right on the kicks over the energy between them.
fn on_kicks_over_between(left: &[f64]) -> f64 {
    let half = RATE as usize / 2;
    let (mut on, mut off) = (0.0, 0.0);
    for (n, x) in left.iter().enumerate().skip(half) {
        let k = n % half;
        if k < half / 8 {
            on += x * x;
        } else if k > half / 2 && k < half * 7 / 8 {
            off += x * x;
        }
    }
    on / off.max(1e-12)
}

#[test]
fn the_drums_keep_the_kicks_and_the_other_stem_the_chord() {
    if gpu().is_none() {
        eprintln!("skipped: no GPU");
        return;
    }
    let path = kicks_and_chord();
    let started = Instant::now();
    let drums = played(&path, StemMix::NONE.toggle(Stem::Drums));
    let took = started.elapsed();
    let other = played(&path, StemMix::NONE.toggle(Stem::Other));
    assert_eq!(drums.len(), SECONDS * RATE as usize);
    assert_eq!(other.len(), drums.len());
    let (d, o) = (on_kicks_over_between(&drums), on_kicks_over_between(&other));
    eprintln!("drums {d:.1}, other {o:.1}; {SECONDS} s separated and played in {took:?}");
    assert!(d > 10.0 * o, "drums {d}, other {o}");
}
