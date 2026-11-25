//! The engine with resampling, without a device: a device that takes only
//! 48, 96 and 192 kHz in packed 24 and 16 bit (the GC7 of the phase A
//! probe) plays 44.1 kHz-family files through the resampler, and the bytes
//! equal a fresh resample of the same stream, per track and per seek.

mod common;

use std::path::PathBuf;
use std::time::Duration;

use common::*;
use jodel_output::{Capabilities, SampleFormat, SimClock, StreamFormat};
use jodel_playback::rate::Resample;
use jodel_playback::resample::output_frames;
use jodel_playback::{Error, Event, MemoryTarget, PlayState, Player, PlayerOptions, Settings};
use SampleFormat::*;

fn gc7() -> Capabilities {
    caps(&[48_000, 96_000, 192_000], &[S16, S24Packed])
}

fn options(max_rate: Option<u32>) -> PlayerOptions {
    PlayerOptions {
        settings: Settings {
            max_rate,
            ..Settings::default()
        },
        position_interval: Duration::from_millis(1),
        ..PlayerOptions::default()
    }
}

fn play_all(caps: Capabilities, files: Vec<PathBuf>) -> (Vec<(StreamFormat, Vec<u8>)>, Vec<Event>) {
    let target = MemoryTarget::new(caps, SimClock::Instant);
    let captures = target.captures();
    let (player, events) = Player::new(Box::new(target), options(None));
    player.load(files).unwrap();
    player.play(0).unwrap();
    let mut all = until_queue_end(&events);
    all.extend(collect_until(&events, |e| {
        matches!(e, Event::State(PlayState::Stopped))
    }));
    drop(player);
    (captured(&captures), all)
}

fn started(events: &[Event]) -> Vec<&jodel_playback::TrackStart> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::TrackStarted(s) => Some(&**s),
            _ => None,
        })
        .collect()
}

/// The position reported with each track end: (output frame, file frame,
/// seconds).
fn ends(events: &[Event]) -> Vec<(u64, u64, f64)> {
    let mut out = Vec::new();
    let mut last = None;
    for e in events {
        match e {
            Event::Position(p) => last = Some((p.output_frame, p.frame, p.seconds)),
            Event::TrackEnded { finished: true, .. } => out.push(last.unwrap()),
            _ => {}
        }
    }
    out
}

#[test]
fn pcm_at_44k_plays_resampled_to_48k() {
    let dir = temp_dir("resample-pcm");
    let wav = dir.join("tones.wav");
    let frames = 20_000;
    let samples = tones(44_100, frames);
    write_wav(&wav, 44_100, 2, &samples);
    let (streams, events) = play_all(gc7(), vec![wav]);
    assert_eq!(streams.len(), 1);
    assert_eq!(streams[0].0, StreamFormat::new(48_000, 2, S24Packed));
    let want = reference_resample(44_100, 48_000, 2, &samples);
    assert_eq!(
        want.len() as u64,
        2 * output_frames(44_100, 48_000, frames as u64)
    );
    assert!(
        streams[0].1 == device_bytes(&want, S24Packed),
        "bytes differ"
    );

    let start = started(&events)[0];
    assert_eq!(start.route.resample, Some(Resample::new(44_100, 48_000)));
    assert_eq!(
        start.route.describe(),
        "44.1 kHz → resample 48 kHz (160/147)"
    );
    assert!(!start.route.lossless);
    // Positions are in the file's frames and seconds.
    let (output_frame, frame, seconds) = ends(&events)[0];
    assert_eq!(output_frame, 21_769);
    assert!(frame.abs_diff(frames as u64) <= 1, "{frame}");
    assert!((seconds - frames as f64 / 44_100.0).abs() < 1.0 / 44_100.0);
    // Half-scale tones do not clip.
    assert!(!events.iter().any(|e| matches!(e, Event::Clipped { .. })));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_flac_file_at_44k_plays_resampled_to_48k() {
    let (streams, events) = play_all(gc7(), vec![fixture(F44)]);
    assert_eq!(streams.len(), 1);
    assert_eq!(streams[0].0, StreamFormat::new(48_000, 2, S24Packed));
    let want = reference_resample(44_100, 48_000, 2, &read_all(&fixture(F44)));
    assert!(
        streams[0].1 == device_bytes(&want, S24Packed),
        "bytes differ"
    );
    let start = started(&events)[0];
    assert_eq!(
        start.route.describe(),
        "44.1 kHz → resample 48 kHz (160/147)"
    );
    let (output_frame, frame, _) = ends(&events)[0];
    assert_eq!(output_frame, want.len() as u64 / 2);
    assert!(frame.abs_diff(FIXTURE_FRAMES as u64) <= 1, "{frame}");
}

/// Seeking: the bytes before the seek, then exactly a fresh resample of the
/// track from the target frame.
#[test]
fn seek_equals_a_fresh_resample_from_the_target() {
    let dir = temp_dir("resample-seek");
    let wav = dir.join("tones.wav");
    let samples = tones(44_100, 30_000);
    write_wav(&wav, 44_100, 2, &samples);
    let (target, captures, gate) = GatedTarget::new(gc7());
    let (player, events) = Player::new(Box::new(target), options(None));
    player.load(vec![wav]).unwrap();
    let before = 7_000u64;
    gate.allow(before);
    player.play(0).unwrap();
    gate.wait_written(before);
    let target_frame = 12_345usize;
    player.seek(target_frame as f64 / 44_100.0).unwrap();
    let seen = collect_until(&events, |e| matches!(e, Event::Seeked { .. }));
    let Some(Event::Seeked { seconds, .. }) = seen.last() else {
        unreachable!()
    };
    assert!((seconds - target_frame as f64 / 44_100.0).abs() < 1e-9);
    gate.allow_all();
    let rest = until_queue_end(&events);
    drop(player);
    // The first position after the seek is at the target.
    let first = rest.iter().find_map(|e| match e {
        Event::Position(p) => Some(*p),
        _ => None,
    });
    assert!(first.unwrap().frame >= target_frame as u64);

    let full = reference_resample(44_100, 48_000, 2, &samples);
    let fresh = reference_resample(44_100, 48_000, 2, &samples[2 * target_frame..]);
    let mut want = full[..2 * before as usize].to_vec();
    want.extend(&fresh);
    let streams = captured(&captures);
    assert_eq!(streams.len(), 1);
    assert!(
        streams[0].1 == device_bytes(&want, S24Packed),
        "bytes differ"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

/// Tracks on the same resampled route play into one stream, each resampled
/// on its own: the lengths add up exactly.
#[test]
fn gapless_resampled_tracks_keep_their_exact_lengths() {
    let dir = temp_dir("resample-gapless");
    let (a, b) = (dir.join("a.wav"), dir.join("b.wav"));
    let (sa, sb) = (tones(44_100, 10_001), tones(44_100, 7_777));
    write_wav(&a, 44_100, 2, &sa);
    write_wav(&b, 44_100, 2, &sb);
    let (streams, events) = play_all(gc7(), vec![a, b]);
    assert_eq!(streams.len(), 1, "the output was reopened");
    let mut want = reference_resample(44_100, 48_000, 2, &sa);
    let first = want.len() as u64 / 2;
    want.extend(reference_resample(44_100, 48_000, 2, &sb));
    assert_eq!(first, output_frames(44_100, 48_000, 10_001));
    assert_eq!(
        want.len() as u64 / 2,
        first + output_frames(44_100, 48_000, 7_777)
    );
    assert!(
        streams[0].1 == device_bytes(&want, S24Packed),
        "bytes differ"
    );
    let ends = ends(&events);
    assert_eq!(ends.len(), 2);
    assert_eq!(ends[0].0, first);
    assert_eq!(ends[1].0, want.len() as u64 / 2);
    std::fs::remove_dir_all(dir).unwrap();
}

/// A full-scale square wave rings past full scale: the output saturates
/// and the clip count arrives as events, rising, timed to the output.
#[test]
fn clipping_is_counted_and_reported() {
    let dir = temp_dir("resample-clip");
    let wav = dir.join("square.wav");
    let samples: Vec<i32> = (0..44_100)
        .flat_map(|n| {
            let s = if (n / 50) % 2 == 0 {
                8_388_607
            } else {
                -8_388_608
            };
            [s, 0]
        })
        .collect();
    write_wav(&wav, 44_100, 2, &samples);
    let (streams, events) = play_all(gc7(), vec![wav]);
    let want = reference_resample(44_100, 48_000, 2, &samples);
    assert!(streams[0].1 == device_bytes(&want, S24Packed));
    let clips: Vec<(u64, u64)> = events
        .iter()
        .filter_map(|e| match e {
            Event::Clipped {
                samples,
                output_frame,
                index: 0,
            } => Some((*samples, *output_frame)),
            _ => None,
        })
        .collect();
    assert!(clips.len() >= 2, "{clips:?}");
    assert!(clips.windows(2).all(|w| w[0].0 < w[1].0 && w[0].1 < w[1].1));
    // At most four a second of output.
    assert!(clips.windows(2).all(|w| w[1].1 - w[0].1 >= 12_000));
    assert!(clips.len() <= 5, "{clips:?}");
    std::fs::remove_dir_all(dir).unwrap();
}

/// Turning resampling off while a resampled track plays leaves it without
/// a route: it keeps playing as it was (a notice says so), to its end. The
/// next track, which has no route either, is skipped.
#[test]
fn turning_resampling_off_keeps_the_track_playing() {
    let (target, captures, gate) = GatedTarget::new(gc7());
    let (player, events) = Player::new(Box::new(target), options(None));
    player.load(vec![fixture(F44), fixture(F44)]).unwrap();
    gate.allow(3_000);
    player.play(0).unwrap();
    gate.wait_written(3_000);
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    player
        .set_settings(Settings {
            resample: false,
            ..Settings::default()
        })
        .unwrap();
    let seen = collect_until(&events, |e| matches!(e, Event::Notice { .. }));
    match seen.last() {
        Some(Event::Notice {
            index: Some(0),
            message,
        }) => {
            assert!(message.contains("resampling is off"), "{message}");
            assert!(message.contains("keeping 44.1 kHz"), "{message}");
        }
        other => panic!("{other:?}"),
    }
    gate.allow_all();
    let seen = until_queue_end(&events);
    let mut order = Vec::new();
    for event in &seen {
        match event {
            Event::TrackEnded { index, finished } => order.push(format!("end {index} {finished}")),
            Event::Error {
                index,
                error: Error::NoRoute(_),
                skipped,
            } => order.push(format!("error {index:?} {skipped}")),
            Event::TrackStarted(s) => order.push(format!("start {}", s.index)),
            _ => {}
        }
    }
    assert_eq!(order, ["end 0 true", "error Some(1) true"]);
    drop(player);
    // One stream, the whole track, as it started.
    let streams = captured(&captures);
    assert_eq!(streams.len(), 1);
    assert_eq!(streams[0].0.rate, 48_000);
}
