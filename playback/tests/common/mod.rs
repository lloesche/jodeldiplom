//! Test support: the PCM fixtures, a target whose outputs only accept
//! frames as the test allows (to place commands at exact output frames),
//! event collection and reference resamples.

#![allow(dead_code)]

pub mod lanes;

use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use jodel_output::lanes::lane_width;
use jodel_output::mixer::{Fader, InputTarget, Mixer, MixerOptions, Role, Strip};
use jodel_output::{
    Capabilities, Clock, Error as OutputError, OpenRequest, Output, SampleFormat, SimClock, Stats,
    StreamFormat, StreamInfo,
};
use jodel_playback::{Captures, Event, MemoryTarget, OutputTarget, Resampler, Source};

pub const TIMEOUT: Duration = Duration::from_secs(30);

pub fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/fixtures/pcm")
        .join(name)
}

/// 44.1 kHz, 16 bits, 8,193 stereo frames.
pub const F44: &str = "ramp-44100-16bit.flac";
/// 48 kHz, 24 bits, 8,193 stereo frames.
pub const F48: &str = "ramp-48000-24bit.flac";
/// The fixtures' length in frames.
pub const FIXTURE_FRAMES: usize = 8_193;

/// The bytes a device in `format` gets for 24-bit `samples`.
pub fn device_bytes(samples: &[i32], format: SampleFormat) -> Vec<u8> {
    let mut out = vec![0u8; samples.len() * format.bytes()];
    jodel_output::pack::pack(samples, format, jodel_output::PackMode::Exact, &mut out).unwrap();
    out
}

/// Every frame of a file, as the engine reads it.
pub fn read_all(path: &Path) -> Vec<i32> {
    let mut source = Source::open(path).unwrap();
    let mut all = Vec::new();
    while source.read(&mut all).unwrap() > 0 {}
    all
}

pub fn caps(rates: &[u32], formats: &[SampleFormat]) -> Capabilities {
    let mut caps = Capabilities::new();
    for &rate in rates {
        for &f in formats {
            caps.insert(StreamFormat::new(rate, 2, f));
        }
    }
    caps
}

/// Collects events until one matches `done` (included). Panics on timeout.
pub fn collect_until(events: &Receiver<Event>, done: impl Fn(&Event) -> bool) -> Vec<Event> {
    let deadline = Instant::now() + TIMEOUT;
    let mut seen = Vec::new();
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match events.recv_timeout(left) {
            Ok(event) => {
                let stop = done(&event);
                seen.push(event);
                if stop {
                    return seen;
                }
            }
            Err(e) => panic!("no matching event ({e}); seen: {seen:#?}"),
        }
    }
}

/// Lets frames through a few at a time until a track starts: a switch of
/// route meets the old one a little after what was written, so the old one
/// must be let through to there. Returns the events up to it.
pub fn trickle_until_started(gate: &Gate, events: &Receiver<Event>) -> Vec<Event> {
    trickle_until(gate, events, |e| matches!(e, Event::TrackStarted(_)))
}

/// The same, until an event matches `done` (included).
pub fn trickle_until(
    gate: &Gate,
    events: &Receiver<Event>,
    done: impl Fn(&Event) -> bool,
) -> Vec<Event> {
    let deadline = Instant::now() + TIMEOUT;
    let mut seen = Vec::new();
    let mut last = Instant::now();
    loop {
        assert!(
            Instant::now() < deadline,
            "no matching event; seen: {seen:#?}"
        );
        if last.elapsed() >= Duration::from_millis(20) {
            gate.allow(gate.written() + 256);
            last = Instant::now();
        }
        match events.recv_timeout(Duration::from_millis(5)) {
            Ok(event) => {
                let stop = done(&event);
                seen.push(event);
                if stop {
                    return seen;
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(e) => panic!("{e}; seen: {seen:#?}"),
        }
    }
}

pub fn until_queue_end(events: &Receiver<Event>) -> Vec<Event> {
    collect_until(events, |e| matches!(e, Event::QueueEnded))
}

/// The bytes of every stream a memory target opened.
pub fn captured(captures: &Captures) -> Vec<(StreamFormat, Vec<u8>)> {
    captures
        .list()
        .into_iter()
        .map(|c| (c.format, c.sink.bytes()))
        .collect()
}

/// How many output frames the test lets through, across every stream.
#[derive(Default)]
pub struct Gate {
    state: Mutex<(u64, u64)>, // (allowed, written)
    changed: Condvar,
}

impl Gate {
    pub fn allow(&self, frames: u64) {
        let mut s = self.state.lock().unwrap();
        s.0 = frames;
        self.changed.notify_all();
    }

    pub fn allow_all(&self) {
        self.allow(u64::MAX);
    }

    pub fn written(&self) -> u64 {
        self.state.lock().unwrap().1
    }

    /// Waits until `frames` have gone through.
    pub fn wait_written(&self, frames: u64) {
        let mut s = self.state.lock().unwrap();
        let deadline = Instant::now() + TIMEOUT;
        while s.1 < frames {
            let left = deadline.saturating_duration_since(Instant::now());
            assert!(!left.is_zero(), "only {} of {frames} frames written", s.1);
            s = self.changed.wait_timeout(s, left).unwrap().0;
        }
    }

    /// Waits up to `timeout` until `frames` have gone through; whether they
    /// have.
    pub fn written_within(&self, frames: u64, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut s = self.state.lock().unwrap();
        while s.1 < frames {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return false;
            }
            s = self.changed.wait_timeout(s, left).unwrap().0;
        }
        true
    }

    fn take(&self, frames: u64, timeout: Option<Duration>) -> u64 {
        let mut s = self.state.lock().unwrap();
        let wait = timeout.unwrap_or(Duration::from_millis(5));
        if s.0 <= s.1 {
            s = self.changed.wait_timeout(s, wait).unwrap().0;
        }
        frames.min(s.0.saturating_sub(s.1))
    }

    fn wrote(&self, frames: u64) {
        let mut s = self.state.lock().unwrap();
        s.1 += frames;
        self.changed.notify_all();
    }

    /// The output gave back `frames` ([`Output::retract`]): they count as
    /// not written.
    fn unwrote(&self, frames: u64) {
        let mut s = self.state.lock().unwrap();
        s.1 -= frames;
        self.changed.notify_all();
    }
}

/// A target whose outputs accept only what the [`Gate`] allows.
pub struct GatedTarget {
    inner: Box<dyn OutputTarget>,
    pub gate: Arc<Gate>,
}

impl GatedTarget {
    /// A gated [`MemoryTarget`].
    pub fn new(caps: Capabilities) -> (GatedTarget, Captures, Arc<Gate>) {
        let inner = MemoryTarget::new(caps, SimClock::Instant);
        let captures = inner.captures();
        let (target, gate) = GatedTarget::around(Box::new(inner));
        (target, captures, gate)
    }

    /// `inner` gated.
    pub fn around(inner: Box<dyn OutputTarget>) -> (GatedTarget, Arc<Gate>) {
        let gate = Arc::new(Gate::default());
        (
            GatedTarget {
                inner,
                gate: gate.clone(),
            },
            gate,
        )
    }
}

/// A [`GatedTarget`] on the Lead input of a mixer on a [`MemoryTarget`]
/// (a device that plays what it is given at once: the mixer mixes a
/// period as soon as the engine has written one), the device's streams,
/// and the mixer (dropped after the player).
pub fn gated_mixer(caps: Capabilities) -> (GatedTarget, Captures, Arc<Gate>, Mixer) {
    let (lead, captures, mixer) = mixer_lead(caps);
    let (target, gate) = GatedTarget::around(Box::new(lead));
    (target, captures, gate, mixer)
}

/// The Lead input of a mixer on a [`MemoryTarget`], the device's streams,
/// and the mixer.
pub fn mixer_lead(caps: Capabilities) -> (InputTarget, Captures, Mixer) {
    let device = MemoryTarget::new(caps, SimClock::Instant);
    let captures = device.captures();
    let mixer = Mixer::open(Box::new(device), MixerOptions::default());
    let (lead, _fader) = mixer.input(&mixer.strip("Player"), Role::Lead, 1.0);
    (lead, captures, mixer)
}

/// A [`GatedTarget`] on a lane input of a mixer on a [`MemoryTarget`]:
/// main on the Player strip, the four stems on strips inside it (in lane
/// order), the device's streams, the mixer and the part strips and faders.
pub fn gated_mixer_lanes(caps: Capabilities) -> (GatedTarget, Captures, Arc<Gate>, MixerLanes) {
    let device = MemoryTarget::new(caps, SimClock::Instant);
    let captures = device.captures();
    let mixer = Mixer::open(Box::new(device), MixerOptions::default());
    let player = mixer.strip("Player");
    let parts: Vec<Strip> = ["Drums", "Bass", "Vocals", "Other"]
        .into_iter()
        .map(|name| mixer.strip_in(&player, name))
        .collect();
    let refs: Vec<&Strip> = parts.iter().collect();
    let (input, _fader, faders) = mixer.input_lanes(&player, &refs, Role::Lead, 1.0);
    let (target, gate) = GatedTarget::around(Box::new(input));
    (
        target,
        captures,
        gate,
        MixerLanes {
            parts,
            faders,
            mixer,
        },
    )
}

/// What [`gated_mixer_lanes`] keeps: the part strips and faders, and the
/// mixer (dropped after the player).
pub struct MixerLanes {
    pub parts: Vec<Strip>,
    pub faders: Vec<Fader>,
    pub mixer: Mixer,
}

/// The lanes of a lane stream's bytes ([`jodel_output::lanes::LANE_SAMPLE`],
/// `channels` a lane, `lanes` lanes and the control word a frame): each
/// lane's samples (interleaved, 24-bit), then the control words.
pub fn lanes_of(bytes: &[u8], channels: usize, lanes: usize) -> (Vec<Vec<i32>>, Vec<i32>) {
    let width = channels * lanes + 1;
    let words: Vec<i32> = bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) >> 8)
        .collect();
    let mut out = vec![Vec::new(); lanes];
    let mut control = Vec::new();
    for frame in words.chunks_exact(width) {
        for (l, lane) in out.iter_mut().enumerate() {
            lane.extend_from_slice(&frame[l * channels..(l + 1) * channels]);
        }
        control.push(frame[width - 1]);
    }
    (out, control)
}

/// The frames a memory target's streams have taken, together.
pub fn captured_frames(captures: &Captures) -> u64 {
    captures
        .list()
        .iter()
        .map(|c| (c.sink.len() / c.format.frame_bytes()) as u64)
        .sum()
}

/// Waits until a memory target's streams have taken `frames` together.
pub fn wait_captured(captures: &Captures, frames: u64) {
    let deadline = Instant::now() + TIMEOUT;
    while captured_frames(captures) < frames {
        assert!(
            Instant::now() < deadline,
            "only {} of {frames} frames reached the device",
            captured_frames(captures)
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

impl OutputTarget for GatedTarget {
    fn name(&self) -> String {
        "gated memory".into()
    }

    fn capabilities(&mut self) -> Result<Capabilities, OutputError> {
        self.inner.capabilities()
    }

    fn lanes(&mut self) -> u16 {
        self.inner.lanes()
    }

    fn open(&mut self, request: &OpenRequest) -> Result<Box<dyn Output>, OutputError> {
        Ok(Box::new(GatedOutput {
            inner: self.inner.open(request)?,
            gate: self.gate.clone(),
        }))
    }
}

struct GatedOutput {
    inner: Box<dyn Output>,
    gate: Arc<Gate>,
}

impl Output for GatedOutput {
    fn info(&self) -> &StreamInfo {
        self.inner.info()
    }
    fn lanes(&self) -> u16 {
        self.inner.lanes()
    }
    fn write(&mut self, samples: &[i32], timeout: Option<Duration>) -> Result<usize, OutputError> {
        let channels = lane_width(self.inner.format().channels, self.inner.lanes());
        let frames = (samples.len() / channels) as u64;
        let n = self.gate.take(frames, timeout) as usize;
        if n == 0 {
            return Ok(0);
        }
        let done = self.inner.write(&samples[..n * channels], timeout)?;
        self.gate.wrote(done as u64);
        Ok(done)
    }
    fn drain(&mut self, timeout: Option<Duration>) -> Result<bool, OutputError> {
        self.inner.drain(timeout)
    }
    fn position(&self) -> Result<u64, OutputError> {
        self.inner.position()
    }
    fn written(&self) -> u64 {
        self.inner.written()
    }
    fn latency(&self) -> Duration {
        self.inner.latency()
    }
    fn pause(&mut self) -> Result<(), OutputError> {
        self.inner.pause()
    }
    fn resume(&mut self) -> Result<(), OutputError> {
        self.inner.resume()
    }
    fn is_paused(&self) -> bool {
        self.inner.is_paused()
    }
    fn flush(&mut self) -> Result<(), OutputError> {
        self.inner.flush()
    }
    fn retract(&mut self, frames: u64) -> Result<u64, OutputError> {
        let n = self.inner.retract(frames)?;
        self.gate.unwrote(n);
        Ok(n)
    }
    fn stats(&self) -> Stats {
        self.inner.stats()
    }
    fn clock(&self) -> Clock {
        self.inner.clock()
    }
    fn close(&mut self) -> Result<(), OutputError> {
        self.inner.close()
    }
}

/// What a [`HoldingTarget`]'s outputs hold: up to `room` frames written
/// and not yet played, played as far as the test lets them.
#[derive(Default)]
pub struct Hold {
    state: Mutex<Held>,
}

#[derive(Default)]
struct Held {
    /// The samples held, a frame `width` of them.
    samples: std::collections::VecDeque<i32>,
    width: usize,
    room: u64,
    /// Frames let through to the device, and those it got.
    let_through: u64,
    played: u64,
}

impl Hold {
    /// Lets `frames` more be played.
    pub fn play(&self, frames: u64) {
        let mut h = self.state.lock().unwrap();
        h.let_through = h.let_through.saturating_add(frames);
    }

    /// Lets everything be played.
    pub fn play_all(&self) {
        self.play(u64::MAX);
    }

    /// The last sample of each frame held (a lane stream's control word).
    pub fn held_words(&self) -> Vec<i32> {
        let h = self.state.lock().unwrap();
        let w = h.width.max(1);
        let n = h.samples.len() / w;
        (0..n).map(|f| h.samples[f * w + w - 1]).collect()
    }
}

/// A target whose outputs hold what is written (up to a room's worth) until
/// the test lets it be played ([`Hold::play`]): what they hold can be given
/// back ([`Output::retract`]), all of it, and the test sees it.
pub struct HoldingTarget {
    inner: Box<dyn OutputTarget>,
    pub hold: Arc<Hold>,
    room: u64,
}

impl HoldingTarget {
    /// `inner` held, `room` frames at most.
    pub fn around(inner: Box<dyn OutputTarget>, room: u64) -> (HoldingTarget, Arc<Hold>) {
        let hold = Arc::new(Hold::default());
        let target = HoldingTarget {
            inner,
            hold: hold.clone(),
            room,
        };
        (target, hold)
    }
}

impl OutputTarget for HoldingTarget {
    fn name(&self) -> String {
        "held memory".into()
    }

    fn capabilities(&mut self) -> Result<Capabilities, OutputError> {
        self.inner.capabilities()
    }

    fn lanes(&mut self) -> u16 {
        self.inner.lanes()
    }

    fn open(&mut self, request: &OpenRequest) -> Result<Box<dyn Output>, OutputError> {
        let inner = self.inner.open(request)?;
        let width = lane_width(inner.format().channels, inner.lanes());
        {
            let mut h = self.hold.state.lock().unwrap();
            // Less than the device's ring, so the writer never takes the
            // output for full and keeps writing (which plays what is let).
            let ring = inner.info().ring_frames;
            *h = Held {
                width,
                room: self.room.min(ring.saturating_sub(1)),
                let_through: h.let_through,
                ..Held::default()
            };
        }
        Ok(Box::new(HoldingOutput {
            inner,
            hold: self.hold.clone(),
        }))
    }
}

struct HoldingOutput {
    inner: Box<dyn Output>,
    hold: Arc<Hold>,
}

impl HoldingOutput {
    /// Plays what the test lets through.
    fn play(&mut self) -> Result<(), OutputError> {
        let mut h = self.hold.state.lock().unwrap();
        let w = h.width.max(1);
        let n = (h.let_through - h.played.min(h.let_through)).min((h.samples.len() / w) as u64);
        if n > 0 {
            let out: Vec<i32> = h.samples.drain(..n as usize * w).collect();
            h.played += n;
            drop(h);
            self.inner.write(&out, None)?;
        }
        Ok(())
    }

    fn held(&self) -> u64 {
        let h = self.hold.state.lock().unwrap();
        (h.samples.len() / h.width.max(1)) as u64
    }
}

impl Output for HoldingOutput {
    fn info(&self) -> &StreamInfo {
        self.inner.info()
    }
    fn lanes(&self) -> u16 {
        self.inner.lanes()
    }
    fn write(&mut self, samples: &[i32], _: Option<Duration>) -> Result<usize, OutputError> {
        self.play()?;
        let mut h = self.hold.state.lock().unwrap();
        let w = h.width.max(1);
        let held = (h.samples.len() / w) as u64;
        let n = ((samples.len() / w) as u64).min(h.room.saturating_sub(held)) as usize;
        h.samples.extend(&samples[..n * w]);
        drop(h);
        if n == 0 {
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok(n)
    }
    fn drain(&mut self, timeout: Option<Duration>) -> Result<bool, OutputError> {
        self.play()?;
        if self.held() > 0 {
            std::thread::sleep(Duration::from_millis(1));
            return Ok(false);
        }
        self.inner.drain(timeout)
    }
    fn position(&self) -> Result<u64, OutputError> {
        self.inner.position()
    }
    fn written(&self) -> u64 {
        self.inner.written() + self.held()
    }
    fn latency(&self) -> Duration {
        self.inner.latency()
    }
    fn pause(&mut self) -> Result<(), OutputError> {
        self.inner.pause()
    }
    fn resume(&mut self) -> Result<(), OutputError> {
        self.inner.resume()
    }
    fn is_paused(&self) -> bool {
        self.inner.is_paused()
    }
    fn flush(&mut self) -> Result<(), OutputError> {
        let mut h = self.hold.state.lock().unwrap();
        h.samples.clear();
        drop(h);
        self.inner.flush()
    }
    fn retract(&mut self, frames: u64) -> Result<u64, OutputError> {
        self.play()?;
        let mut h = self.hold.state.lock().unwrap();
        let w = h.width.max(1);
        let n = frames.min((h.samples.len() / w) as u64);
        let keep = h.samples.len() - n as usize * w;
        h.samples.truncate(keep);
        Ok(n)
    }
    fn stats(&self) -> Stats {
        self.inner.stats()
    }
    fn clock(&self) -> Clock {
        self.inner.clock()
    }
    fn close(&mut self) -> Result<(), OutputError> {
        self.play()?;
        self.inner.close()
    }
}

/// A 16-bit stereo WAV of a ramp (left = n, right = -n, in 24-bit terms
/// `n << 8`), `frames` long.
pub fn ramp_wav(path: &Path, rate: u32, frames: u32) -> Vec<i32> {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    let mut samples = Vec::new();
    for n in 0..frames {
        let v = (n % 30_000) as i16;
        writer.write_sample(v).unwrap();
        writer.write_sample(-v).unwrap();
        samples.push(i32::from(v) << 8);
        samples.push(-(i32::from(v) << 8));
    }
    writer.finalize().unwrap();
    samples
}

/// A 24-bit WAV of `samples` (24-bit, interleaved).
pub fn write_wav(path: &Path, rate: u32, channels: u16, samples: &[i32]) {
    let spec = hound::WavSpec {
        channels,
        sample_rate: rate,
        bits_per_sample: 24,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for &s in samples {
        writer.write_sample(s).unwrap();
    }
    writer.finalize().unwrap();
}

/// Two tones (left 1 kHz, right 5 kHz) at half scale, `frames` long.
pub fn tones(rate: u32, frames: usize) -> Vec<i32> {
    let w = |f: f64, n: usize| {
        ((2.0 * std::f64::consts::PI * f * n as f64 / f64::from(rate)).sin() * 4_194_304.0) as i32
    };
    (0..frames)
        .flat_map(|n| [w(1_000.0, n), w(5_000.0, n)])
        .collect()
}

/// A fresh resample of a whole stream, as one track.
pub fn reference_resample(from: u32, to: u32, channels: usize, samples: &[i32]) -> Vec<i32> {
    let mut resampler = Resampler::new(from, to, channels);
    let mut out = Vec::new();
    resampler.process(samples, &mut out);
    resampler.finish(&mut out);
    out
}

/// A fresh directory under the system temp dir for this test, removed
/// when it goes out of scope (with the test's audio files in it: left
/// behind, they filled a small temp file system run after run).
pub fn temp_dir(name: &str) -> TempDir {
    let dir = std::env::temp_dir().join(format!("jodel-playback-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    TempDir(dir)
}

/// A directory removed when dropped ([`temp_dir`]).
pub struct TempDir(PathBuf);

impl std::ops::Deref for TempDir {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for TempDir {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Held by a test that plays on a device in real time (a simulated clock
/// that plays a period every period of wall-clock time) for as long as it
/// runs: one such test of a binary at a time, and none beside one that
/// holds [`heavy`]. Side by side on a small, loaded machine (a debug
/// build, every test of a binary at once), their threads would starve each
/// other of the periods they must keep.
pub fn real_time() -> std::sync::RwLockWriteGuard<'static, ()> {
    REAL_TIME.write().unwrap_or_else(|e| e.into_inner())
}

static REAL_TIME: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Held by a test that keeps the processor busy (separating stems, say) in
/// a binary with [`real_time`] tests: any number of them at once, but none
/// while one of those runs.
pub fn heavy() -> std::sync::RwLockReadGuard<'static, ()> {
    REAL_TIME.read().unwrap_or_else(|e| e.into_inner())
}
