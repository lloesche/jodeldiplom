//! A device the system starts running at another rate (its format changed
//! in the sound settings): the stream fails as the device is lost, and the
//! next track routes to the device's new rate, the engine asking the desk
//! what it takes anew rather than keeping what it was told.

mod common;

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::*;
use jodel_output::mixer::{Mixer, MixerOptions, Role};
use jodel_output::{
    Capabilities, Clock, Error, OpenRequest, Output, OutputTarget, SampleFormat, SimClock, Stats,
    StreamInfo,
};
use jodel_playback::{Event, MemoryTarget, Player, PlayerOptions};

const FORMAT: SampleFormat = SampleFormat::S24In32Low;

/// A device in real time that takes one rate, `rate` as it is now, and
/// whose streams fail while `fail` is on, as a device lost.
#[derive(Clone)]
struct Moving {
    rate: Arc<AtomicU32>,
    fail: Arc<AtomicBool>,
}

impl OutputTarget for Moving {
    fn name(&self) -> String {
        "moving device".into()
    }

    fn capabilities(&mut self) -> Result<Capabilities, Error> {
        Ok(caps(&[self.rate.load(Ordering::SeqCst)], &[FORMAT]))
    }

    fn open(&mut self, request: &OpenRequest) -> Result<Box<dyn Output>, Error> {
        let mut inner = MemoryTarget::new(self.capabilities()?, SimClock::Realtime);
        let out = inner.open(request)?;
        Ok(Box::new(Failing(out, self.fail.clone())))
    }
}

struct Failing(Box<dyn Output>, Arc<AtomicBool>);

impl Output for Failing {
    fn info(&self) -> &StreamInfo {
        self.0.info()
    }
    fn write(&mut self, samples: &[i32], timeout: Option<Duration>) -> Result<usize, Error> {
        if self.1.load(Ordering::SeqCst) {
            return Err(Error::DeviceLost);
        }
        self.0.write(samples, timeout)
    }
    fn drain(&mut self, timeout: Option<Duration>) -> Result<bool, Error> {
        self.0.drain(timeout)
    }
    fn position(&self) -> Result<u64, Error> {
        self.0.position()
    }
    fn written(&self) -> u64 {
        self.0.written()
    }
    fn latency(&self) -> Duration {
        self.0.latency()
    }
    fn pause(&mut self) -> Result<(), Error> {
        self.0.pause()
    }
    fn resume(&mut self) -> Result<(), Error> {
        self.0.resume()
    }
    fn is_paused(&self) -> bool {
        self.0.is_paused()
    }
    fn flush(&mut self) -> Result<(), Error> {
        self.0.flush()
    }
    fn stats(&self) -> Stats {
        self.0.stats()
    }
    fn clock(&self) -> Clock {
        self.0.clock()
    }
    fn close(&mut self) -> Result<(), Error> {
        self.0.close()
    }
}

fn started_rate(seen: &[Event], index: usize) -> Option<u32> {
    seen.iter().find_map(|e| match e {
        Event::TrackStarted(s) if s.index == index => Some(s.route.output_rate),
        _ => None,
    })
}

#[test]
fn a_device_back_at_another_rate_routes_the_next_track_there() {
    let _guard = real_time();
    let device = Moving {
        rate: Arc::new(AtomicU32::new(48_000)),
        fail: Arc::new(AtomicBool::new(false)),
    };
    let mixer = Mixer::open(Box::new(device.clone()), MixerOptions::default());
    let strip = mixer.strip("Player");
    let (target, _fader) = mixer.input(&strip, Role::Lead, 1.0);
    let (player, events) = Player::new(Box::new(target), PlayerOptions::default());
    player.load(vec![fixture(F48), fixture(F44)]).unwrap();
    player.play(0).unwrap();
    let seen = collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    assert_eq!(started_rate(&seen, 0), Some(48_000));
    // The system runs the device at 44.1 kHz now: its stream is lost.
    device.rate.store(44_100, Ordering::SeqCst);
    device.fail.store(true, Ordering::SeqCst);
    let deadline = Instant::now() + TIMEOUT;
    while mixer.format().is_some() {
        assert!(Instant::now() < deadline, "the device never failed");
        std::thread::sleep(Duration::from_millis(5));
    }
    device.fail.store(false, Ordering::SeqCst);
    player.play(1).unwrap();
    let seen = collect_until(
        &events,
        |e| matches!(e, Event::TrackStarted(s) if s.index == 1),
    );
    assert_eq!(started_rate(&seen, 1), Some(44_100));
    assert_eq!(mixer.format().map(|f| f.rate), Some(44_100));
}
