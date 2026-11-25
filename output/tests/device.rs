//! The desk's device as it comes and goes, on a simulated device that
//! takes what it is given at once: one not there as the mixer starts has
//! no format and refuses outputs, then, once there, is asked what it
//! takes, opened and played by the next output; one that fails fails every
//! output on it with its error, and the next output opens it again at the
//! same format (another epoch), a fader's ramp under way ending on its
//! target and a change for a time of the last epoch never coming.

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use common::*;
use jodel_output::mixer::{Curve, Mixer, MixerOptions, Ramp, Role, Start};
use jodel_output::{
    Capabilities, Clock, Error, OpenRequest, Output, OutputTarget, Stats, StreamFormat, StreamInfo,
};

/// A device that may be away (`present` off: it has no capabilities and
/// opens nothing) or fail (`fail` on: what is written fails, as a device
/// unplugged).
#[derive(Clone)]
struct Flaky {
    device: Device,
    present: Arc<AtomicBool>,
    fail: Arc<AtomicBool>,
}

impl Flaky {
    fn new(present: bool) -> Flaky {
        Flaky {
            device: Device {
                instant: true,
                ..Device::default()
            },
            present: Arc::new(AtomicBool::new(present)),
            fail: Arc::new(AtomicBool::new(false)),
        }
    }

    fn there(&self) -> Result<(), Error> {
        if self.present.load(Ordering::SeqCst) {
            Ok(())
        } else {
            Err(Error::NoDevice)
        }
    }

    fn mixer(&self) -> Mixer {
        let options = MixerOptions {
            buffer: Duration::from_millis(200),
            ..MixerOptions::default()
        };
        Mixer::open(Box::new(self.clone()), options)
    }
}

impl OutputTarget for Flaky {
    fn name(&self) -> String {
        "flaky device".into()
    }
    fn capabilities(&mut self) -> Result<Capabilities, Error> {
        self.there()?;
        self.device.capabilities()
    }
    fn open(&mut self, request: &OpenRequest) -> Result<Box<dyn Output>, Error> {
        self.there()?;
        let inner = self.device.open(request)?;
        Ok(Box::new(Failing(inner, self.fail.clone())))
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

#[test]
fn a_device_not_there_at_the_start_is_opened_once_it_is() {
    let flaky = Flaky::new(false);
    let mixer = flaky.mixer();
    assert_eq!(mixer.format(), None);
    let strip = mixer.strip("Player");
    let (mut lead, _fader) = mixer.input(&strip, Role::Lead, 1.0);
    assert_eq!(lead.capabilities().err(), Some(Error::NoDevice));
    assert!(matches!(lead.open(&request(RATE)), Err(Error::NoDevice)));
    // There now: asked again, and opened by the first output.
    flaky.present.store(true, Ordering::SeqCst);
    let desk = StreamFormat::new(RATE, 2, FORMAT);
    let offered: Vec<StreamFormat> = lead.capabilities().unwrap().iter().copied().collect();
    assert_eq!(offered, [desk]);
    assert_eq!(mixer.format(), Some(desk));
    let mut out = open_started(&mut lead, RATE, 1_000);
    out.write(&frames(1, 4_799, |_| 1_000), None).unwrap();
    out.drain(None).unwrap();
    drop(out);
    drop(mixer);
    let left = flaky.device.left(0);
    assert_eq!(left.len(), 4_800);
    assert!(left.iter().all(|&v| v == 1_000));
}

#[test]
fn a_device_that_fails_is_opened_again_by_the_next_output() {
    let flaky = Flaky::new(true);
    let mixer = flaky.mixer();
    let group = mixer.strip("Player");
    let (mut lead, fader) = mixer.input(&group, Role::Lead, 1.0);
    let level = 1_000_000;
    let mut out = open_started(&mut lead, RATE, level);
    out.write(&frames(1, 4_799, |_| level), None).unwrap();
    wait_until("mixed", || mixer.timebase().frame == 4_800);
    let tb = mixer.timebase();
    assert_eq!(tb.epoch, 1);
    // A fade under way, and a mute for a frame of this epoch, to come.
    fader.set(Ramp {
        start: Start::Now,
        to: 0.5,
        over: Duration::from_secs(2),
        curve: Curve::Linear,
    });
    out.write(&frames(0, 480, |_| level), None).unwrap();
    wait_until("mixed", || mixer.timebase().frame == 5_280);
    assert!(fader.gain() > 0.5 && fader.gain() < 1.0, "{}", fader.gain());
    group.set_mute_at(true, tb.at(48_000));
    taken(&mut lead);
    assert_eq!(group.pending_mute(), Some(true));
    // The device fails: the output fails with its error.
    flaky.fail.store(true, Ordering::SeqCst);
    let failed = (0..1_000)
        .map(|_| {
            std::thread::sleep(Duration::from_millis(1));
            out.write(&frames(0, 480, |_| level), None)
        })
        .find_map(Result::err);
    assert_eq!(failed, Some(Error::DeviceLost));
    wait_until("closed", || mixer.desk().master.stream.is_none());
    // Back: the next output opens it again, at the same format, another
    // epoch; the fade ended on its target, the mute never comes.
    flaky.fail.store(false, Ordering::SeqCst);
    let mut out = open_started(&mut lead, RATE, level);
    out.write(&frames(1, 9_599, |_| level), None).unwrap();
    out.drain(None).unwrap();
    let tb = mixer.timebase();
    assert_eq!((tb.epoch, tb.rate), (2, RATE));
    assert_eq!(mixer.format(), Some(StreamFormat::new(RATE, 2, FORMAT)));
    assert!(fader.settled() && fader.gain() == 0.5);
    assert!(!group.mute() && group.pending_mute().is_none());
    drop(out);
    drop(mixer);
    let streams = flaky.device.streams.lock().unwrap().len();
    assert_eq!(streams, 2);
    let left = flaky.device.left(1);
    assert_eq!(left.len(), 9_600);
    let half = f64::from(level) * 0.5;
    assert!(left.iter().all(|&v| (f64::from(v) - half).abs() < 1.5));
}
