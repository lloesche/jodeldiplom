//! Settings changes while a track plays, the file's own rate on a device
//! that takes several, and opening a device that is busy for a moment.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::*;
use jodel_output::{
    Capabilities, Error as OutputError, OpenRequest, Output, SampleFormat, SimClock, StreamFormat,
};
use jodel_playback::{
    Error, Event, MemoryTarget, OutputTarget, Player, PlayerOptions, Settings, Stage,
    OPEN_BUSY_WAIT,
};
use SampleFormat::*;

fn settings(max_rate: Option<u32>, resample: bool) -> Settings {
    Settings {
        max_rate,
        resample,
        ..Settings::default()
    }
}

/// The routes of a switch meet at one frame (no fades), so every stream is
/// exactly some route's output.
fn options() -> PlayerOptions {
    PlayerOptions {
        settings: settings(None, false),
        position_interval: Duration::from_millis(1),
        switch_fade: Duration::ZERO,
        ..PlayerOptions::default()
    }
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

fn play_all(caps: Capabilities, files: &[&str]) -> (Vec<(StreamFormat, Vec<u8>)>, Vec<Event>) {
    let target = MemoryTarget::new(caps, SimClock::Instant);
    let captures = target.captures();
    let (player, events) = Player::new(Box::new(target), options());
    player
        .load(files.iter().map(|f| fixture(f)).collect())
        .unwrap();
    player.play(0).unwrap();
    let seen = until_queue_end(&events);
    drop(player);
    (captured(&captures), seen)
}

/// The 48, 96 and 192 kHz device of the rate policy's tests, packed 24 bit.
fn gc7() -> Capabilities {
    caps(&[48_000, 96_000, 192_000], &[S16, S24Packed])
}

/// A file plays at its own rate, as it is, on a device that takes others
/// too.
#[test]
fn the_file_plays_at_its_own_rate() {
    for (name, rate) in [(F44, 44_100u32), (F48, 48_000)] {
        let (streams, events) = play_all(caps(&[rate, 2 * rate, 4 * rate], &[S24In32Low]), &[name]);
        let start = started(&events)[0];
        assert_eq!(start.route.stage, Stage::Passthrough);
        assert!(start.route.lossless);
        assert_eq!(streams.len(), 1);
        assert_eq!(streams[0].0, StreamFormat::new(rate, 2, S24In32Low));
        assert!(
            streams[0].1 == device_bytes(&read_all(&fixture(name)), S24In32Low),
            "{name}: bytes differ"
        );
    }
}

/// A switch of route mid-track (resampled under a cap, then back to the
/// file's rate) moves the track to its new route after the frame written
/// up to: no skip, no end, and the streams before and after the resampled
/// one are the file's samples.
#[test]
fn a_route_switch_mid_track_keeps_the_position() {
    let all = read_all(&fixture(F48));
    let total = all.len() / 2;
    let (target, captures, gate) = GatedTarget::new(caps(&[44_100, 48_000, 96_000], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options());
    player.load(vec![fixture(F48)]).unwrap();
    gate.allow(3_000);
    player.play(0).unwrap();
    gate.wait_written(3_000);
    let first = collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    assert_eq!(started(&first)[0].route.output_rate, 48_000);
    for (cap, rate) in [(Some(44_100), 44_100), (None, 48_000)] {
        player.set_settings(settings(cap, true)).unwrap();
        let seen = trickle_until_started(&gate, &events);
        let start = started(&seen)[0];
        assert_eq!((start.index, start.route.output_rate), (0, rate));
        assert!(
            !seen.iter().any(|e| matches!(
                e,
                Event::Error { .. } | Event::TrackEnded { .. } | Event::QueueEnded
            )),
            "{seen:#?}"
        );
        let written = gate.written();
        gate.allow(written + 1_000);
        gate.wait_written(written + 1_000);
    }
    gate.allow_all();
    let rest = until_queue_end(&events);
    assert!(!rest.iter().any(|e| matches!(e, Event::Error { .. })));
    assert_eq!(
        rest.iter()
            .filter(|e| matches!(
                e,
                Event::TrackEnded {
                    index: 0,
                    finished: true
                }
            ))
            .count(),
        1
    );
    drop(player);
    let streams = captured(&captures);
    assert_eq!(streams.len(), 3);
    assert_eq!(
        streams.iter().map(|s| s.0.rate).collect::<Vec<_>>(),
        [48_000, 44_100, 48_000]
    );
    // Bit-perfect before and after: the file's samples, the first stream
    // from its start, the last one to its end.
    let first = streams[0].1.len() / 8;
    let last = streams[2].1.len() / 8;
    assert!(first >= 3_000 && last > 0);
    assert!(streams[0].1 == device_bytes(&all[..2 * first], S24In32Low));
    assert!(streams[2].1 == device_bytes(&all[2 * (total - last)..], S24In32Low));
    // The resampled stream in between covers the rest of the track.
    let between = (total - last - first) as f64 * 44_100.0 / 48_000.0;
    let got = (streams[1].1.len() / 8) as f64;
    assert!((got - between).abs() <= 2.0, "{got} frames, want {between}");
}

/// A rate cap below every rate the device takes, set while a track plays:
/// the track keeps its route and plays to its end; a notice says why. The
/// next track has no route with that cap and is skipped as before.
#[test]
fn no_route_on_a_settings_change_keeps_playing() {
    let (target, captures, gate) = GatedTarget::new(gc7());
    let (player, events) = Player::new(Box::new(target), options());
    player.load(vec![fixture(F48), fixture(F48)]).unwrap();
    gate.allow(4_000);
    player.play(0).unwrap();
    gate.wait_written(4_000);
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    player.set_settings(settings(Some(44_100), false)).unwrap();
    let seen = collect_until(&events, |e| matches!(e, Event::Notice { .. }));
    let Some(Event::Notice { index, message }) = seen.last() else {
        unreachable!()
    };
    assert_eq!(*index, Some(0));
    assert!(message.starts_with("No route for this track"), "{message}");
    assert!(message.contains("44.1 kHz cap"), "{message}");
    assert!(message.contains("keeping 48 kHz"), "{message}");
    gate.allow_all();
    let rest = until_queue_end(&events);
    let skipped: Vec<_> = rest
        .iter()
        .filter_map(|e| match e {
            Event::Error { index, skipped, .. } => Some((*index, *skipped)),
            _ => None,
        })
        .collect();
    assert_eq!(skipped, [(Some(1), true)]);
    assert!(rest.iter().any(|e| matches!(
        e,
        Event::TrackEnded {
            index: 0,
            finished: true
        }
    )));
    drop(player);
    // One stream, as it started: the whole file at 48 kHz.
    let streams = captured(&captures);
    assert_eq!(streams.len(), 1);
    assert_eq!(streams[0].0, StreamFormat::new(48_000, 2, S24Packed));
    assert!(streams[0].1 == device_bytes(&read_all(&fixture(F48)), S24Packed));
}

/// A target that refuses some opens: busy for the first `busy` opens, and
/// never at `refuse_rate`.
struct Flaky<T: OutputTarget> {
    inner: T,
    busy: usize,
    refuse_rate: Option<u32>,
    opens: Arc<AtomicUsize>,
}

impl<T: OutputTarget> OutputTarget for Flaky<T> {
    fn name(&self) -> String {
        "flaky".into()
    }

    fn capabilities(&mut self) -> Result<Capabilities, OutputError> {
        self.inner.capabilities()
    }

    fn open(&mut self, request: &OpenRequest) -> Result<Box<dyn Output>, OutputError> {
        let n = self.opens.fetch_add(1, Ordering::SeqCst);
        if n < self.busy {
            return Err(OutputError::DeviceBusy);
        }
        if self.refuse_rate == Some(request.rate) {
            return Err(OutputError::FormatNotAccepted {
                rate: request.rate,
                channels: request.channels,
            });
        }
        self.inner.open(request)
    }
}

/// A device busy for its first opens is opened once it is free; one busy
/// for longer than [`OPEN_BUSY_WAIT`] fails and playback stops.
#[test]
fn a_busy_device_is_tried_again() {
    let opens = Arc::new(AtomicUsize::new(0));
    let target = Flaky {
        inner: MemoryTarget::new(caps(&[44_100], &[S24In32Low]), SimClock::Instant),
        busy: 3,
        refuse_rate: None,
        opens: opens.clone(),
    };
    let (player, events) = Player::new(Box::new(target), options());
    player.load(vec![fixture(F44)]).unwrap();
    player.play(0).unwrap();
    let seen = until_queue_end(&events);
    assert!(!seen.iter().any(|e| matches!(e, Event::Error { .. })));
    assert_eq!(started(&seen).len(), 1);
    assert_eq!(opens.load(Ordering::SeqCst), 4);
    drop(player);

    let opens = Arc::new(AtomicUsize::new(0));
    let target = Flaky {
        inner: MemoryTarget::new(caps(&[44_100], &[S24In32Low]), SimClock::Instant),
        busy: usize::MAX,
        refuse_rate: None,
        opens: opens.clone(),
    };
    let (player, events) = Player::new(Box::new(target), options());
    player.load(vec![fixture(F44)]).unwrap();
    let at = Instant::now();
    player.play(0).unwrap();
    let seen = collect_until(&events, |e| matches!(e, Event::Error { .. }));
    assert!(at.elapsed() >= OPEN_BUSY_WAIT);
    assert!(seen.iter().any(|e| matches!(
        e,
        Event::Error {
            error: Error::Output(OutputError::DeviceBusy),
            skipped: false,
            ..
        }
    )));
    assert!(opens.load(Ordering::SeqCst) >= 5);
}

/// When the new route's output does not open, the track goes back to the
/// route it played on, at the same position, with a notice.
#[test]
fn a_new_route_that_does_not_open_falls_back() {
    let opens = Arc::new(AtomicUsize::new(0));
    let (gated, captures, gate) = GatedTarget::new(caps(&[44_100, 48_000], &[S24In32Low]));
    let target = Flaky {
        inner: gated,
        busy: 0,
        refuse_rate: Some(44_100),
        opens,
    };
    let (player, events) = Player::new(Box::new(target), options());
    player.load(vec![fixture(F48)]).unwrap();
    gate.allow(3_000);
    player.play(0).unwrap();
    gate.wait_written(3_000);
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    player.set_settings(settings(Some(44_100), true)).unwrap();
    // The old route is let through to where the two would meet.
    let seen = trickle_until(&gate, &events, |e| matches!(e, Event::Notice { .. }));
    let Some(Event::Notice { message, .. }) = seen.last() else {
        unreachable!()
    };
    assert!(message.contains("did not open"), "{message}");
    let start = started(&seen);
    assert_eq!(start.len(), 1);
    assert_eq!(start[0].route.output_rate, 48_000);
    assert_eq!(start[0].route.resample, None);
    gate.allow_all();
    let rest = until_queue_end(&events);
    assert!(!rest.iter().any(|e| matches!(e, Event::Error { .. })));
    drop(player);
    // The file before, then the file again from where it was heard.
    let streams = captured(&captures);
    assert_eq!(streams.len(), 2);
    assert!(streams.iter().all(|s| s.0.rate == 48_000));
    let heard = streams[0].1.len() / 8;
    assert!(heard >= 3_000, "{heard}");
    let all = read_all(&fixture(F48));
    assert!(streams[0].1 == device_bytes(&all[..2 * heard], S24In32Low));
    assert!(streams[1].1 == device_bytes(&all[2 * heard..], S24In32Low));
}
