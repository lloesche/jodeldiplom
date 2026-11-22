//! Switches of route while a track plays, without a break: the new route
//! reads ahead and meets the old one at an exact frame. At one output
//! format they share one stream (at one frame with no fade, or
//! crossfaded); at another the old route fades out at that frame and the
//! new one, in the next stream, fades in from it.
//!
//! The switches here meet early in the track, within every route's
//! run-up, so each new route reads from the start of the track and each
//! piece of a stream is a slice of a whole resample.

mod common;

use std::time::Duration;

use common::*;
use jodel_output::{SampleFormat, SimClock};
use jodel_playback::{
    Event, Headroom, MemoryTarget, Player, PlayerOptions, Resampler, Settings, Stage, TrackStart,
};
use SampleFormat::*;

fn settings(max_rate: Option<u32>, headroom: Headroom) -> Settings {
    Settings {
        max_rate,
        resample: true,
        headroom,
    }
}

fn options(max_rate: Option<u32>, fade: Duration) -> PlayerOptions {
    PlayerOptions {
        settings: settings(max_rate, Headroom::NONE),
        position_interval: Duration::from_millis(1),
        switch_fade: fade,
        ..PlayerOptions::default()
    }
}

/// The default crossfade.
const FADE: Duration = Duration::from_millis(20);

/// The 24-bit samples of a stream the device took as `S24In32Low`.
fn samples(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|b| (i32::from_le_bytes([b[0], b[1], b[2], b[3]]) << 8) >> 8)
        .collect()
}

fn starts(events: &[Event]) -> Vec<&TrackStart> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::TrackStarted(s) => Some(&**s),
            _ => None,
        })
        .collect()
}

/// The whole of `all` (stereo) resampled from `from` to `to` Hz with
/// `headroom`.
fn resampled(from: u32, to: u32, all: &[i32], headroom: Headroom) -> Vec<i32> {
    let mut resampler = Resampler::new(from, to, 2);
    resampler.set_gain(headroom.gain());
    let mut out = Vec::new();
    resampler.process(all, &mut out);
    resampler.finish(&mut out);
    out
}

/// No error, no track left unfinished, one end of the track and of the
/// queue, and the position never goes back.
fn assert_unbroken(events: &[Event]) {
    assert!(
        !events.iter().any(|e| matches!(
            e,
            Event::Error { .. }
                | Event::TrackEnded {
                    finished: false,
                    ..
                }
                | Event::Notice { .. }
        )),
        "{events:#?}"
    );
    let ends = events
        .iter()
        .filter(|e| matches!(e, Event::TrackEnded { finished: true, .. }))
        .count();
    assert_eq!(ends, 1);
    let frames: Vec<u64> = events
        .iter()
        .filter_map(|e| match e {
            Event::Position(p) => Some(p.frame),
            _ => None,
        })
        .collect();
    assert!(frames.windows(2).all(|w| w[0] <= w[1]), "{frames:?}");
}

/// Headroom changes on a resampled track at one output rate: one stream,
/// and every piece of it exactly the route's own output from the frame it
/// took over at. One of the switches is made while paused.
#[test]
fn switches_at_one_rate_are_one_exact_stream() {
    let all = read_all(&fixture(F44));
    let (target, captures, gate) = GatedTarget::new(caps(&[48_000], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options(None, Duration::ZERO));
    player.load(vec![fixture(F44)]).unwrap();
    gate.allow(1_000);
    player.play(0).unwrap();
    gate.wait_written(1_000);
    let mut seen = collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    for (n, headroom) in [Headroom(20), Headroom(10), Headroom::NONE]
        .into_iter()
        .enumerate()
    {
        let paused = n == 2;
        if paused {
            player.pause().unwrap();
        }
        player.set_settings(settings(None, headroom)).unwrap();
        if paused {
            player.resume().unwrap();
        }
        seen.extend(trickle_until_started(&gate, &events));
    }
    gate.allow_all();
    seen.extend(until_queue_end(&events));
    drop(player);
    assert_unbroken(&seen);

    let legs = starts(&seen);
    let gains: Vec<Option<Headroom>> = legs
        .iter()
        .map(|s| s.route.resample.map(|r| r.headroom))
        .collect();
    assert_eq!(
        gains,
        [
            Some(Headroom::NONE),
            Some(Headroom(20)),
            Some(Headroom(10)),
            Some(Headroom::NONE)
        ]
    );
    let streams = captured(&captures);
    assert_eq!(streams.len(), 1, "one stream throughout");
    let got = samples(&streams[0].1);
    let want = pieces(&legs, &all);
    assert_eq!(got.len(), want.len());
    assert!(got == want, "the stream is the routes' outputs, end to end");
}

/// What a stream at 48 kHz holds for `legs` of the 44.1 kHz track `all`:
/// each route's whole output, from the output frame it took over at to the
/// one the next took over at.
fn pieces(legs: &[&TrackStart], all: &[i32]) -> Vec<i32> {
    let mut want = Vec::new();
    for (i, leg) in legs.iter().enumerate() {
        // Each switch meets on the 147 : 160 grid of the two rates.
        assert_eq!(147 * leg.output_frame, 160 * leg.from_frame, "{i}");
        assert_eq!(leg.route.stage, Stage::Passthrough);
        let headroom = leg.route.resample.expect("resampled").headroom;
        let whole = resampled(44_100, 48_000, all, headroom);
        let from = 2 * leg.output_frame as usize;
        let to = legs
            .get(i + 1)
            .map_or(whole.len(), |next| 2 * next.output_frame as usize);
        want.extend_from_slice(&whole[from..to]);
    }
    want
}

/// Two changes at once: the second replaces the first if it has not met
/// the old route yet, or follows it once it has taken over. Either way the
/// track ends on the last one, in one exact stream.
#[test]
fn a_second_change_replaces_or_follows_the_first() {
    let all = read_all(&fixture(F44));
    let (target, captures, gate) = GatedTarget::new(caps(&[48_000], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options(None, Duration::ZERO));
    player.load(vec![fixture(F44)]).unwrap();
    gate.allow(1_000);
    player.play(0).unwrap();
    gate.wait_written(1_000);
    let mut seen = collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    player.set_settings(settings(None, Headroom(20))).unwrap();
    player.set_settings(settings(None, Headroom(10))).unwrap();
    let last = |s: &TrackStart| s.route.resample.map(|r| r.headroom) == Some(Headroom(10));
    loop {
        let more = trickle_until_started(&gate, &events);
        let done = starts(&more).last().is_some_and(|s| last(s));
        seen.extend(more);
        if done {
            break;
        }
    }
    gate.allow_all();
    seen.extend(until_queue_end(&events));
    drop(player);
    assert_unbroken(&seen);
    let legs = starts(&seen);
    assert!(legs.last().is_some_and(|s| last(s)));
    let streams = captured(&captures);
    assert_eq!(streams.len(), 1);
    assert!(samples(&streams[0].1) == pieces(&legs, &all));
}

/// With the default crossfade: exact outside it, and inside it every
/// sample between the two routes' (and their rounding).
#[test]
fn a_crossfade_is_exact_outside_its_window() {
    let all = read_all(&fixture(F44));
    let (target, captures, gate) = GatedTarget::new(caps(&[48_000], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options(None, FADE));
    player.load(vec![fixture(F44)]).unwrap();
    gate.allow(2_000);
    player.play(0).unwrap();
    gate.wait_written(2_000);
    let mut seen = collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    player.set_settings(settings(None, Headroom(30))).unwrap();
    seen.extend(trickle_until_started(&gate, &events));
    gate.allow_all();
    seen.extend(until_queue_end(&events));
    drop(player);
    assert_unbroken(&seen);
    let legs = starts(&seen);
    assert_eq!(
        legs[1].route.resample.map(|r| r.headroom),
        Some(Headroom(30))
    );
    let at = 2 * legs[1].output_frame as usize;
    let fade = 2 * 960;
    let before = resampled(44_100, 48_000, &all, Headroom::NONE);
    let after = resampled(44_100, 48_000, &all, Headroom(30));
    let streams = captured(&captures);
    assert_eq!(streams.len(), 1);
    let got = samples(&streams[0].1);
    assert_eq!(got.len(), after.len());
    assert!(got[..at] == before[..at], "the old gain up to {at}");
    assert!(
        got[at + fade..] == after[at + fade..],
        "the new gain after the fade"
    );
    for i in at..at + fade {
        let (lo, hi) = (before[i].min(after[i]), before[i].max(after[i]));
        assert!((lo - 1..=hi + 1).contains(&got[i]), "{i}");
    }
}

/// From a resampled route to the file as it is, on a device that takes
/// both rates: the resampled route fades out at the frame they meet at,
/// the output opens again at the file's rate, and the file plays from that
/// frame, faded in.
#[test]
fn another_format_fades_out_and_in_at_one_frame() {
    let all = read_all(&fixture(F48));
    let total = all.len() / 2;
    let (target, captures, gate) = GatedTarget::new(caps(&[44_100, 48_000], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options(Some(44_100), FADE));
    player.load(vec![fixture(F48)]).unwrap();
    gate.allow(1_500);
    player.play(0).unwrap();
    gate.wait_written(1_500);
    let mut seen = collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    player.set_settings(settings(None, Headroom::NONE)).unwrap();
    seen.extend(trickle_until_started(&gate, &events));
    gate.allow_all();
    seen.extend(until_queue_end(&events));
    drop(player);
    assert_unbroken(&seen);
    let legs = starts(&seen);
    assert_eq!(legs[0].route.output_rate, 44_100);
    let down = resampled(48_000, 44_100, &all, Headroom::NONE);
    let exact = &legs[1];
    assert_eq!(exact.route.stage, Stage::Passthrough);
    assert_eq!(exact.route.resample, None);
    assert_eq!(exact.output_frame, 0, "a new stream");
    let from = exact.from_frame as usize;
    let streams = captured(&captures);
    assert_eq!(streams.len(), 2);
    assert_eq!((streams[0].0.rate, streams[1].0.rate), (44_100, 48_000));
    let first = samples(&streams[0].1);
    let second = samples(&streams[1].1);
    // The resampled track up to that frame, the last 20 ms faded out.
    let (out, rise) = (2 * 882, 2 * 960);
    assert_eq!(147 * from % 160, 0, "{from}");
    assert_eq!(first.len(), 2 * 147 * from / 160);
    assert!(first[..first.len() - out] == down[..first.len() - out]);
    for i in first.len() - out..first.len() {
        assert!(first[i].abs() <= down[i].abs() + 1, "{i}");
    }
    // The file from there, the first 20 ms faded in.
    assert_eq!(second.len(), 2 * (total - from));
    assert!(second[rise..] == all[2 * from + rise..]);
    for i in 0..rise {
        assert!(second[i].abs() <= all[2 * from + i].abs() + 1, "{i}");
    }
}

/// A seek while a switch is under way plays the new route from the frame
/// sought.
#[test]
fn a_seek_during_a_switch_plays_the_new_route_from_there() {
    let all = read_all(&fixture(F48));
    let (target, captures, gate) = GatedTarget::new(caps(&[44_100, 48_000], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options(Some(44_100), FADE));
    player.load(vec![fixture(F48)]).unwrap();
    gate.allow(2_000);
    player.play(0).unwrap();
    gate.wait_written(2_000);
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    // The gate stays shut: the resampled route cannot reach where the two
    // meet (a fade after what is written) before the seek.
    player.set_settings(settings(None, Headroom::NONE)).unwrap();
    player.seek(0.1).unwrap();
    let seen = collect_until(&events, |e| matches!(e, Event::Seeked { .. }));
    let last = starts(&seen).last().copied().cloned().expect("a start");
    assert_eq!(last.route.resample, None);
    assert_eq!(last.from_frame, 4_800);
    gate.allow_all();
    let rest = until_queue_end(&events);
    assert!(!rest.iter().any(|e| matches!(e, Event::Error { .. })));
    drop(player);
    let streams = captured(&captures);
    assert_eq!(streams.len(), 2);
    assert_eq!(streams[0].1.len() / 8, 2_000);
    assert!(samples(&streams[1].1) == all[2 * 4_800..]);
}

/// A switch on a device that plays in real time: the old route plays on
/// while the new one decodes ahead, so nothing is padded. Every frame up to
/// where they meet is the resample at the old gain, and every frame after
/// it the resample at the new gain (to within dither: the new route's
/// resampler started a little earlier, so its filter is full when heard).
#[test]
fn a_switch_in_real_time_leaves_no_gap() {
    let dir = temp_dir("switch-realtime");
    let wav = dir.join("tones.wav");
    let frames = 3 * 44_100;
    let file = tones(44_100, frames);
    write_wav(&wav, 44_100, 2, &file);
    let target = MemoryTarget::new(caps(&[48_000], &[S24In32Low]), SimClock::Realtime);
    let captures = target.captures();
    let (player, events) = Player::new(
        Box::new(target),
        PlayerOptions {
            switch_fade: Duration::ZERO,
            ..PlayerOptions::default()
        },
    );
    player.load(vec![wav]).unwrap();
    player.play(0).unwrap();
    std::thread::sleep(Duration::from_millis(1_200));
    player
        .set_settings(Settings {
            headroom: Headroom(20),
            ..Settings::default()
        })
        .unwrap();
    let seen = until_queue_end(&events);
    drop(player);
    assert!(!seen.iter().any(|e| matches!(e, Event::Error { .. })));
    let legs = starts(&seen);
    assert_eq!(legs.len(), 2);
    assert_eq!(
        legs[1].route.resample.map(|r| r.headroom),
        Some(Headroom(20))
    );
    let at = 2 * legs[1].output_frame as usize;
    let before = reference_resample(44_100, 48_000, 2, &file);
    let mut resampler = Resampler::new(44_100, 48_000, 2);
    resampler.set_gain(Headroom(20).gain());
    let mut after = Vec::new();
    resampler.process(&file, &mut after);
    resampler.finish(&mut after);
    let streams = captured(&captures);
    assert_eq!(streams.len(), 1);
    let got = samples(&streams[0].1);
    assert!(got.len() >= before.len());
    assert!(got[..at] == before[..at], "the old gain up to {at}");
    let worst = (at..before.len())
        .map(|i| (got[i] - after[i]).abs())
        .max()
        .unwrap_or(0);
    assert!(worst <= 2, "the new gain from {at}: off by {worst}");
    // Only the device's silence after the end.
    assert!(got[before.len()..].iter().all(|&s| s == 0));
    std::fs::remove_dir_all(dir).unwrap();
}
