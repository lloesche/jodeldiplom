//! Headroom on routes that are not bit-perfect: a gain before their one
//! requantization (inside the resampler, or where a floating-point or lossy
//! file is turned into 24 bits) that keeps overs from clipping, and that
//! never touches a bit-perfect route.

mod common;

use std::path::Path;
use std::time::Duration;

use common::*;
use jodel_output::{Capabilities, SampleFormat, SimClock};
use jodel_playback::{
    Event, Headroom, MemoryTarget, Player, PlayerOptions, Resampler, Settings, Stage,
};
use SampleFormat::*;

fn options(headroom: Headroom) -> PlayerOptions {
    PlayerOptions {
        settings: Settings {
            headroom,
            ..Settings::default()
        },
        position_interval: Duration::from_millis(1),
        ..PlayerOptions::default()
    }
}

/// Plays `wav` on a device that takes `caps`: the bytes of its one stream,
/// and the events.
fn play(caps: Capabilities, wav: &Path, headroom: Headroom) -> (Vec<u8>, Vec<Event>) {
    let target = MemoryTarget::new(caps, SimClock::Instant);
    let captures = target.captures();
    let (player, events) = Player::new(Box::new(target), options(headroom));
    player.load(vec![wav.to_path_buf()]).unwrap();
    player.play(0).unwrap();
    let seen = until_queue_end(&events);
    drop(player);
    let mut streams = captured(&captures);
    assert_eq!(streams.len(), 1);
    (streams.remove(0).1, seen)
}

/// `[A, A, -A, -A]` per channel: a 11.025 kHz sine at 44.1 kHz sampled
/// 45° off its peaks, whose true peak is √2·A. At 0.74 of full scale that
/// is about 1.05 of full scale between the samples.
fn overs(frames: usize) -> Vec<i32> {
    let a = (0.74 * 8_388_607.0) as i32;
    (0..frames)
        .flat_map(|n| {
            let v = if n % 4 < 2 { a } else { -a };
            [v, v]
        })
        .collect()
}

fn resample(input: &[i32], gain: f64) -> (Vec<i32>, u64) {
    let mut r = Resampler::new(44_100, 48_000, 2);
    r.set_gain(gain);
    let mut out = Vec::new();
    r.process(input, &mut out);
    r.finish(&mut out);
    (out, r.clipped())
}

/// The gain is the resample of the input scaled by it, exactly: no other
/// rounding than the one requantization.
#[test]
fn the_gain_is_applied_exactly() {
    let input = tones(44_100, 20_000);
    for headroom in Headroom::CHOICES {
        let gain = headroom.gain();
        let (with, _) = resample(&input, gain);
        let scaled: Vec<f64> = input.iter().map(|&s| f64::from(s) * gain).collect();
        let mut r = Resampler::new(44_100, 48_000, 2);
        let mut want = Vec::new();
        r.process_float(&scaled, &mut want);
        r.finish(&mut want);
        assert_eq!(with, want, "{headroom}");
    }
    // And it is a gain: -1 dB is about 0.891 of the level.
    let (none, _) = resample(&input, 1.0);
    let (one, _) = resample(&input, Headroom(10).gain());
    let peak = |s: &[i32]| s.iter().map(|v| v.unsigned_abs()).max().unwrap();
    let ratio = f64::from(peak(&one)) / f64::from(peak(&none));
    assert!((ratio - 0.8913).abs() < 1e-3, "{ratio}");
}

/// Through the engine: inter-sample overs clip without headroom and not
/// with 1 dB; the bytes are the resample of the file scaled by the gain.
#[test]
fn headroom_keeps_inter_sample_overs_from_clipping() {
    let dir = temp_dir("headroom-overs");
    let wav = dir.join("overs.wav");
    let samples = overs(20_000);
    write_wav(&wav, 44_100, 2, &samples);
    let gc7 = caps(&[48_000, 96_000, 192_000], &[S24Packed]);
    let clipped = |events: &[Event]| {
        events
            .iter()
            .filter_map(|e| match e {
                Event::Clipped { samples, .. } => Some(*samples),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    };
    // The resampler alone: the overs clip at full scale.
    let (_, clips) = resample(&samples, 1.0);
    assert!(clips > 1_000, "{clips}");
    let (bytes, events) = play(gc7.clone(), &wav, Headroom::NONE);
    // Counted as heard (at most four reports a second of output).
    assert!(clipped(&events) > 0 && clipped(&events) <= clips);
    assert!(bytes == device_bytes(&resample(&samples, 1.0).0, S24Packed));
    // 1 dB: no clip, and the bytes are the scaled resample.
    let one = Headroom(10);
    let (bytes, events) = play(gc7, &wav, one);
    assert_eq!(clipped(&events), 0);
    let (want, clips) = resample(&samples, one.gain());
    assert_eq!(clips, 0);
    assert!(bytes == device_bytes(&want, S24Packed));
    let start = events
        .iter()
        .find_map(|e| match e {
            Event::TrackStarted(s) => Some(s),
            _ => None,
        })
        .unwrap();
    assert_eq!(start.route.resample.unwrap().headroom, one);
    assert!(start.route.describe().ends_with("(160/147, -1 dB)"));
    std::fs::remove_dir_all(dir).unwrap();
}

/// A floating-point file over full scale, played at its own rate: it clips
/// without headroom, and not with 2 dB, which its route says.
#[test]
fn headroom_keeps_a_floating_point_file_from_clipping() {
    let dir = temp_dir("headroom-float");
    let wav = dir.join("loud.wav");
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 44_100,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(&wav, spec).unwrap();
    // A tone peaking at +1 dB.
    for n in 0..20_000 {
        let v = 1.12 * (n as f32 * 0.05).sin();
        writer.write_sample(v).unwrap();
        writer.write_sample(v).unwrap();
    }
    writer.finalize().unwrap();
    let device = caps(&[44_100], &[S24In32Low]);
    let clipped = |events: &[Event]| events.iter().any(|e| matches!(e, Event::Clipped { .. }));
    let (_, events) = play(device.clone(), &wav, Headroom::NONE);
    assert!(clipped(&events));
    let (bytes, events) = play(device, &wav, Headroom(20));
    assert!(!clipped(&events));
    let start = events
        .iter()
        .find_map(|e| match e {
            Event::TrackStarted(s) => Some(s),
            _ => None,
        })
        .unwrap();
    assert_eq!(start.route.resample, None);
    assert_eq!(start.route.headroom, Headroom(20));
    assert_eq!(start.route.describe(), "44.1 kHz (-2 dB)");
    // About 1.12 × −2 dB of full scale at the peak.
    let peak = bytes
        .chunks_exact(4)
        .map(|b| (i32::from_le_bytes([b[0], b[1], b[2], b[3]]) << 8 >> 8).unsigned_abs())
        .max()
        .unwrap();
    let ratio = f64::from(peak) / 8_388_608.0;
    assert!((ratio - 1.12 * Headroom(20).gain()).abs() < 1e-3, "{ratio}");
    std::fs::remove_dir_all(dir).unwrap();
}

/// A bit-perfect route plays the file's samples whatever the headroom.
#[test]
fn bit_perfect_routes_have_no_headroom() {
    let dir = temp_dir("headroom-exact");
    let wav = dir.join("overs.wav");
    let samples = overs(10_000);
    write_wav(&wav, 44_100, 2, &samples);
    let (bytes, events) = play(caps(&[44_100], &[S24In32Low]), &wav, Headroom(30));
    assert!(bytes == device_bytes(&samples, S24In32Low));
    assert!(!events.iter().any(|e| matches!(e, Event::Clipped { .. })));
    let start = events
        .iter()
        .find_map(|e| match e {
            Event::TrackStarted(s) => Some(s),
            _ => None,
        })
        .unwrap();
    assert_eq!(start.route.stage, Stage::Passthrough);
    assert_eq!(start.route.resample, None);
    // A FLAC file played bit-perfect: its own samples.
    let target = MemoryTarget::new(caps(&[44_100], &[S24In32Low]), SimClock::Instant);
    let captures = target.captures();
    let (player, events) = Player::new(Box::new(target), options(Headroom(30)));
    player.load(vec![fixture(F44)]).unwrap();
    player.play(0).unwrap();
    until_queue_end(&events);
    drop(player);
    let streams = captured(&captures);
    assert!(streams[0].1 == device_bytes(&read_all(&fixture(F44)), S24In32Low));
    std::fs::remove_dir_all(dir).unwrap();
}

/// A headroom change while a track plays moves a resampled track to the new
/// gain without a break, and leaves a bit-perfect one alone.
#[test]
fn a_headroom_change_reroutes_resampled_tracks_only() {
    let dir = temp_dir("headroom-change");
    let wav = dir.join("tones.wav");
    write_wav(&wav, 44_100, 2, &tones(44_100, 30_000));
    for (rates, rerouted) in [(&[48_000u32][..], true), (&[44_100][..], false)] {
        let (target, captures, gate) = GatedTarget::new(caps(rates, &[S24In32Low]));
        let (player, events) = Player::new(Box::new(target), options(Headroom::NONE));
        player.load(vec![wav.clone()]).unwrap();
        gate.allow(3_000);
        player.play(0).unwrap();
        gate.wait_written(3_000);
        collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
        player
            .set_settings(Settings {
                headroom: Headroom(20),
                ..Settings::default()
            })
            .unwrap();
        // The new gain is decoded ahead and crossfaded in after what was
        // written.
        let mut seen = if rerouted {
            trickle_until_started(&gate, &events)
        } else {
            Vec::new()
        };
        gate.allow_all();
        seen.extend(until_queue_end(&events));
        let starts: Vec<_> = seen
            .iter()
            .filter_map(|e| match e {
                Event::TrackStarted(s) => Some(s.route.resample.map(|r| r.headroom)),
                _ => None,
            })
            .collect();
        assert!(!seen.iter().any(|e| matches!(e, Event::Error { .. })));
        if rerouted {
            assert_eq!(starts, [Some(Headroom(20))]);
        } else {
            assert!(starts.is_empty());
        }
        drop(player);
        // Same format: the output stays open.
        assert_eq!(captured(&captures).len(), 1);
    }
    std::fs::remove_dir_all(dir).unwrap();
}
