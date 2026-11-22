//! What the mixer's and the desk's tests share: a simulated device that
//! keeps what it played, mixers on it, and outputs written and read back.

#![allow(dead_code)]

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use jodel_output::mixer::{InputTarget, Mixer, MixerOptions};
use jodel_output::pack::unpack;
use jodel_output::{
    Capabilities, Clock, Error, FileOutput, MemorySink, OpenRequest, Output, OutputTarget,
    SampleFormat, SimClock, StreamFormat,
};

pub const RATE: u32 = 48_000;
pub const FORMAT: SampleFormat = SampleFormat::S32;

/// A device that keeps what it played: every stream opened, with its
/// clock. In real time, or (`instant`) taking what it is given at once.
#[derive(Clone, Default)]
pub struct Device {
    pub streams: Arc<Mutex<Vec<(MemorySink, Clock)>>>,
    pub instant: bool,
}

impl OutputTarget for Device {
    fn name(&self) -> String {
        "test device".into()
    }

    fn capabilities(&mut self) -> Result<Capabilities, Error> {
        Ok([RATE, 96_000]
            .into_iter()
            .map(|r| StreamFormat::new(r, 2, FORMAT))
            .collect())
    }

    fn open(&mut self, request: &OpenRequest) -> Result<Box<dyn Output>, Error> {
        let sink = MemorySink::default();
        let caps = self.capabilities()?;
        let clock = if self.instant {
            SimClock::Instant
        } else {
            SimClock::Realtime
        };
        let out = FileOutput::open(sink.clone(), &caps, request, clock)?;
        self.streams.lock().unwrap().push((sink, out.clock()));
        Ok(Box::new(out))
    }
}

impl Device {
    /// The left channel of stream `i`.
    pub fn left(&self, i: usize) -> Vec<i32> {
        let (sink, _) = self.streams.lock().unwrap()[i].clone();
        unpack(&sink.bytes(), FORMAT)
            .into_iter()
            .step_by(2)
            .collect()
    }

    pub fn underruns(&self) -> u64 {
        let streams = self.streams.lock().unwrap();
        streams.iter().map(|(_, c)| c.stats().underruns).sum()
    }
}

pub fn mixer() -> (Mixer, Device) {
    mixer_on(Device::default())
}

/// A mixer on a device that takes what it is given at once: it mixes a
/// period once every running output holds one, so what is heard together
/// is what was written together, however the test thread is scheduled.
pub fn instant_mixer() -> (Mixer, Device) {
    mixer_on(Device {
        instant: true,
        ..Device::default()
    })
}

pub fn mixer_on(device: Device) -> (Mixer, Device) {
    let options = MixerOptions {
        // Room for a busy test machine; the device period is the default.
        buffer: Duration::from_millis(200),
        ..MixerOptions::default()
    };
    (Mixer::open(Box::new(device.clone()), options), device)
}

/// Writes `a` and `b` to their outputs and drains both, each on a thread
/// of its own: on a device that takes what it is given, one waits for the
/// other (what the limiter holds back goes once both are drained).
pub fn play_both(
    mut oa: Box<dyn Output>,
    mut ob: Box<dyn Output>,
    a: Vec<i32>,
    b: Vec<i32>,
) -> (Box<dyn Output>, Box<dyn Output>) {
    let tb = std::thread::spawn(move || {
        ob.write(&b, None).unwrap();
        ob.drain(None).unwrap();
        ob
    });
    oa.write(&a, None).unwrap();
    oa.drain(None).unwrap();
    (oa, tb.join().unwrap())
}

/// An output on `target` running with its first frame (left `first`)
/// written and no more: short of a period, the mixer waits for it.
pub fn open_started(target: &mut InputTarget, rate: u32, first: i32) -> Box<dyn Output> {
    let mut out = target.open(&request(rate).prefill(Duration::ZERO)).unwrap();
    out.write(&[first, 0], None).unwrap();
    out
}

pub fn request(rate: u32) -> OpenRequest {
    OpenRequest::new(rate, 2).formats(&[FORMAT])
}

pub fn open(target: &mut InputTarget, rate: u32) -> Box<dyn Output> {
    target.open(&request(rate)).unwrap()
}

/// `frames` stereo frames, the left one `f(frame)`, the right zero.
pub fn frames(from: u64, n: u64, f: impl Fn(u64) -> i32) -> Vec<i32> {
    (from..from + n).flat_map(|k| [f(k), 0]).collect()
}

pub fn wait_until(what: &str, done: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !done() {
        assert!(Instant::now() < deadline, "timed out: {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

pub fn frames_of(n: u64) -> Duration {
    Duration::from_secs_f64(n as f64 / f64::from(RATE))
}

/// Writes each of `data` to its output of `outs` and drains them all, each
/// on a thread of its own (as [`play_both`] does for two).
pub fn play_all(outs: Vec<Box<dyn Output>>, data: Vec<Vec<i32>>) -> Vec<Box<dyn Output>> {
    let threads: Vec<_> = outs
        .into_iter()
        .zip(data)
        .map(|(mut out, d)| {
            std::thread::spawn(move || {
                out.write(&d, None).unwrap();
                out.drain(None).unwrap();
                out
            })
        })
        .collect();
    threads.into_iter().map(|t| t.join().unwrap()).collect()
}

/// The plateaus `left` holds (runs of at least 2 000 frames within 2 of a
/// value of `levels`) from the first at `outer` to the last (before and
/// after, the inputs start and end one after the other), and the largest
/// step between frames from the first to the last.
pub fn plateaus(left: &[i32], levels: &[i32], outer: i32) -> (Vec<i32>, i32) {
    let mut runs: Vec<(i32, usize, usize)> = Vec::new();
    let mut at: Option<(i32, usize)> = None;
    for (k, &v) in left.iter().enumerate().chain([(left.len(), &i32::MIN)]) {
        let near = levels.iter().copied().find(|&l| v.abs_diff(l) <= 2);
        match (at, near) {
            (Some((l, _)), Some(n)) if l == n => {}
            _ => {
                if let Some((l, from)) = at.take() {
                    if k - from >= 2_000 {
                        runs.push((l, from, k));
                    }
                }
                at = near.map(|n| (n, k));
            }
        }
    }
    let first = runs.iter().position(|r| r.0 == outer).unwrap();
    let last = runs.iter().rposition(|r| r.0 == outer).unwrap();
    let runs = &runs[first..=last];
    let (from, to) = (runs[0].2, runs[runs.len() - 1].1);
    let step = left[from..to]
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .max()
        .unwrap_or(0);
    (runs.iter().map(|r| r.0).collect(), step)
}
