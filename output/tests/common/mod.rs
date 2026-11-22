//! What the mixer's and the desk's tests share: a simulated device that
//! keeps what it played, mixers on it, and outputs written and read back.

#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use jodel_output::mixer::{InputTarget, Mixer, MixerOptions};
use jodel_output::pack::unpack;
use jodel_output::{
    Capabilities, Clock, Error, FileOutput, Input, MemorySink, OpenRequest, Output, OutputTarget,
    SampleFormat, SimClock, StreamFormat, VoiceBlock,
};

pub const RATE: u32 = 48_000;
pub const FORMAT: SampleFormat = SampleFormat::S32;

/// A device that keeps what it played: every stream opened, with its
/// clock. In real time, or (`instant`) taking what it is given at once; in
/// [`FORMAT`], or `sample` when given; stereo, or `channels` when given;
/// in real time with a buffer of its own of `buffered` frames when given
/// ([`FileOutput::with_device_buffer`]: hardware's).
#[derive(Clone, Default)]
pub struct Device {
    pub streams: Arc<Mutex<Vec<(MemorySink, Clock)>>>,
    pub instant: bool,
    pub sample: Option<SampleFormat>,
    pub channels: Option<u16>,
    pub buffered: Option<u64>,
}

impl OutputTarget for Device {
    fn name(&self) -> String {
        "test device".into()
    }

    fn capabilities(&mut self) -> Result<Capabilities, Error> {
        Ok([RATE, 96_000]
            .into_iter()
            .map(|r| StreamFormat::new(r, self.channels(), self.format()))
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
        let mut out = FileOutput::open(sink.clone(), &caps, request, clock)?;
        if let Some(frames) = self.buffered {
            out = out.with_device_buffer(frames);
        }
        self.streams.lock().unwrap().push((sink, out.clock()));
        Ok(Box::new(out))
    }
}

impl Device {
    /// Its sample format.
    pub fn format(&self) -> SampleFormat {
        self.sample.unwrap_or(FORMAT)
    }

    /// Its channels.
    pub fn channels(&self) -> u16 {
        self.channels.unwrap_or(2)
    }

    /// The left channel of stream `i`.
    pub fn left(&self, i: usize) -> Vec<i32> {
        let (sink, _) = self.streams.lock().unwrap()[i].clone();
        unpack(&sink.bytes(), self.format())
            .into_iter()
            .step_by(usize::from(self.channels()))
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

/// Returns once the mixer has taken everything sent to it before (a
/// question asked of it through `target` comes after them): what is
/// written next is mixed after the changes set so far.
pub fn taken(target: &mut InputTarget) {
    target.capabilities().unwrap();
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

/// A microphone fed in real time (a [`MicFeed`] on a thread of its own,
/// its cleaner run there too) singing `tone`, until dropped.
pub struct Singer {
    pub input: Input,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Singer {
    pub fn new(tone: fn(usize) -> f32) -> Singer {
        let (input, mut feed) = Input::feed(48_000, 240);
        input.set_echo_cancel(false);
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let thread = std::thread::spawn(move || {
            let start = Instant::now();
            let mut sent = 0;
            while !stopped.load(Ordering::Relaxed) {
                let t = start.elapsed();
                feed.set_time(t);
                let due = (t.as_secs_f64() * 48_000.0) as usize;
                while sent + 240 <= due {
                    let packet: Vec<f32> = (sent..sent + 240).map(tone).collect();
                    feed.push(&packet);
                    sent += 240;
                }
                feed.clean();
                std::thread::sleep(Duration::from_millis(1));
            }
        });
        Singer {
            input,
            stop,
            thread: Some(thread),
        }
    }

    /// The voice given since the last call.
    pub fn voice(&self) -> Vec<VoiceBlock> {
        let mut voice = Vec::new();
        self.input.take_voice(&mut voice);
        voice
    }
}

impl Drop for Singer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

pub fn loud(i: usize) -> f32 {
    0.8 * (i as f32 * 0.13).sin()
}

/// Held by a test that plays on a device in real time (a simulated clock
/// that plays a period every period of wall-clock time) for as long as it
/// runs: one such test of a binary at a time. Side by side on a small,
/// loaded machine (a debug build, every test of a binary at once), their
/// threads would starve each other of the periods they must keep.
pub fn real_time() -> std::sync::RwLockWriteGuard<'static, ()> {
    REAL_TIME.write().unwrap_or_else(|e| e.into_inner())
}

static REAL_TIME: std::sync::RwLock<()> = std::sync::RwLock::new(());
