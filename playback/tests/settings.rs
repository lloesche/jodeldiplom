//! A file at the target's rate plays as it is, and opening a device that
//! is busy fails at once.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use common::*;
use jodel_output::{
    Capabilities, Error as OutputError, OpenRequest, Output, SampleFormat, SimClock, StreamFormat,
};
use jodel_playback::{Error, Event, MemoryTarget, OutputTarget, Player, PlayerOptions};
use SampleFormat::*;

fn options() -> PlayerOptions {
    PlayerOptions {
        position_interval: Duration::from_millis(1),
        ..PlayerOptions::default()
    }
}

/// A file at the target's rate plays as it is, a straight copy.
#[test]
fn the_file_plays_as_it_is_at_the_targets_rate() {
    for (name, rate) in [(F44, 44_100u32), (F48, 48_000)] {
        let target = MemoryTarget::new(caps(&[rate], &[S24In32Low]), SimClock::Instant);
        let captures = target.captures();
        let (player, events) = Player::new(Box::new(target), options());
        player.load(vec![fixture(name)]).unwrap();
        player.play(0).unwrap();
        let seen = until_queue_end(&events);
        drop(player);
        let start = seen.iter().find_map(|e| match e {
            Event::TrackStarted(s) => Some(s),
            _ => None,
        });
        assert!(start.unwrap().route.lossless);
        let streams = captured(&captures);
        assert_eq!(streams.len(), 1);
        assert_eq!(streams[0].0, StreamFormat::new(rate, 2, S24In32Low));
        assert!(
            streams[0].1 == device_bytes(&read_all(&fixture(name)), S24In32Low),
            "{name}: bytes differ"
        );
    }
}

/// A target that is busy for every open.
struct Busy<T: OutputTarget> {
    inner: T,
    opens: Arc<AtomicUsize>,
}

impl<T: OutputTarget> OutputTarget for Busy<T> {
    fn name(&self) -> String {
        "busy".into()
    }

    fn capabilities(&mut self) -> Result<Capabilities, OutputError> {
        self.inner.capabilities()
    }

    fn open(&mut self, _: &OpenRequest) -> Result<Box<dyn Output>, OutputError> {
        self.opens.fetch_add(1, Ordering::SeqCst);
        Err(OutputError::DeviceBusy)
    }
}

/// A device another application holds for itself fails the open at once,
/// tried once, and playback stops.
#[test]
fn a_busy_device_fails_the_open() {
    let opens = Arc::new(AtomicUsize::new(0));
    let target = Busy {
        inner: MemoryTarget::new(caps(&[44_100], &[S24In32Low]), SimClock::Instant),
        opens: opens.clone(),
    };
    let (player, events) = Player::new(Box::new(target), options());
    player.load(vec![fixture(F44)]).unwrap();
    player.play(0).unwrap();
    let seen = collect_until(&events, |e| matches!(e, Event::Error { .. }));
    assert!(seen.iter().any(|e| matches!(
        e,
        Event::Error {
            error: Error::Output(OutputError::DeviceBusy),
            skipped: false,
            ..
        }
    )));
    assert_eq!(opens.load(Ordering::SeqCst), 1);
}
