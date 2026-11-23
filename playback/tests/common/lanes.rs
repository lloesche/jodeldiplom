//! Test support for the stem lanes ([`jodel_playback::MakeStems`]): the
//! separators, a lane stream captured and read back ([`Wide`]), pacing a
//! device, and stores of kept stems.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use jodel_output::lanes::ending_frames;
use jodel_output::{Capabilities, SampleFormat, SimClock, StreamFormat};
use jodel_playback::stems::Separator;
use jodel_playback::{
    Captures, Event, MakeStems, MemoryTarget, Player, PlayerOptions, SeparatorSource, Settings,
    StemFile, StemStore, StemWriter,
};

use super::*;

pub const RATE: u32 = 44_100;

/// Each stem a quarter of the mix, so the four add up to it.
pub struct Quarter;

impl Separator for Quarter {
    fn name(&self) -> String {
        "quarter".into()
    }

    fn masks(&self, frames: usize, stft: &[f32]) -> Result<Vec<f32>, String> {
        assert_eq!(stft.len() % frames, 0);
        Ok((0..4 * stft.len())
            .map(|i| if i % 2 == 0 { 0.25 } else { 0.0 })
            .collect())
    }
}

/// A separator that always fails.
pub struct Broken;

impl Separator for Broken {
    fn name(&self) -> String {
        "broken".into()
    }

    fn masks(&self, _: usize, _: &[f32]) -> Result<Vec<f32>, String> {
        Err("out of memory".into())
    }
}

/// The quarter, failing on silence (a second track of nothing).
pub struct FailsOnSilence;

impl Separator for FailsOnSilence {
    fn name(&self) -> String {
        "fails on silence".into()
    }

    fn masks(&self, frames: usize, stft: &[f32]) -> Result<Vec<f32>, String> {
        if stft.iter().all(|&x| x == 0.0) {
            return Err("nothing to separate".into());
        }
        Quarter.masks(frames, stft)
    }
}

/// The quarter, separating no more chunks than it is let (one for each
/// test that uses it, as they run side by side).
pub struct Gated {
    allowed: AtomicUsize,
    separated: AtomicUsize,
}

impl Gated {
    pub const fn new() -> Gated {
        Gated {
            allowed: AtomicUsize::new(0),
            separated: AtomicUsize::new(0),
        }
    }

    pub fn allow(&self, chunks: usize) {
        self.allowed.store(chunks, Ordering::SeqCst);
    }
}

impl Separator for Gated {
    fn name(&self) -> String {
        "gated".into()
    }

    fn masks(&self, frames: usize, stft: &[f32]) -> Result<Vec<f32>, String> {
        while self.separated.load(Ordering::SeqCst) >= self.allowed.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        self.separated.fetch_add(1, Ordering::SeqCst);
        Quarter.masks(frames, stft)
    }
}

pub static QUARTER: Quarter = Quarter;
pub static BROKEN: Broken = Broken;
pub static FAILS_ON_SILENCE: FailsOnSilence = FailsOnSilence;
pub static RUN_OUT: Gated = Gated::new();
pub static TAKEN_BACK: Gated = Gated::new();
pub static FIRST_CHUNK: Gated = Gated::new();

pub fn fails_on_silence() -> Option<&'static dyn Separator> {
    Some(&FAILS_ON_SILENCE)
}

pub fn run_out() -> Option<&'static dyn Separator> {
    Some(&RUN_OUT)
}

pub fn taken_back() -> Option<&'static dyn Separator> {
    Some(&TAKEN_BACK)
}

pub fn first_chunk() -> Option<&'static dyn Separator> {
    Some(&FIRST_CHUNK)
}

pub fn quarter() -> Option<&'static dyn Separator> {
    Some(&QUARTER)
}

pub fn broken() -> Option<&'static dyn Separator> {
    Some(&BROKEN)
}

pub fn options(resample: bool, offline: bool, separator: SeparatorSource) -> PlayerOptions {
    PlayerOptions {
        settings: Settings {
            max_rate: None,
            resample,
            ..Settings::default()
        },
        position_interval: Duration::from_millis(1),
        separator: Some(separator),
        offline,
        ..PlayerOptions::default()
    }
}

/// Within this many 24-bit steps the parts add up to main: the
/// separator's transforms, four requantizations against main's one.
pub const CLOSE: i64 = 64;

/// A device's 24-bit samples (`S24In32Low`).
pub fn device_samples(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|b| (i32::from_le_bytes([b[0], b[1], b[2], b[3]]) << 8) >> 8)
        .collect()
}

pub fn caps_of(rates: &[u32], channels: u16) -> Capabilities {
    let mut caps = Capabilities::new();
    for &rate in rates {
        caps.insert(StreamFormat::new(rate, channels, SampleFormat::S24In32Low));
    }
    caps
}

/// A lane stream as captured: main, the four parts, the control words.
pub struct Wide {
    pub rate: u32,
    pub channels: usize,
    pub main: Vec<i32>,
    pub parts: Vec<Vec<i32>>,
    pub control: Vec<i32>,
}

impl Wide {
    /// The one stream captured.
    pub fn of(captures: &Captures) -> Wide {
        assert_eq!(captures.list().len(), 1, "one stream");
        Wide::nth(captures, 0)
    }

    /// Stream `n` of those captured.
    pub fn nth(captures: &Captures, n: usize) -> Wide {
        let list = captures.list();
        let channels = usize::from(list[n].format.channels);
        let (mut lanes, control) = lanes_of(&list[n].sink.bytes(), channels, 5);
        let parts = lanes.split_off(1);
        Wide {
            rate: list[n].format.rate,
            channels,
            main: lanes.remove(0),
            parts,
            control,
        }
    }

    pub fn frames(&self) -> usize {
        self.control.len()
    }

    /// The largest step between main and the parts' sum on the frames
    /// `range` whose parts are there (and how many those are).
    pub fn worst(&self, range: std::ops::Range<usize>) -> (i64, usize) {
        let ch = self.channels;
        let mut worst = (0i64, 0usize);
        for f in range {
            if self.control[f] == 0 {
                continue;
            }
            worst.1 += 1;
            for c in 0..ch {
                let i = f * ch + c;
                let sum: i64 = self.parts.iter().map(|p| i64::from(p[i])).sum();
                worst.0 = worst.0.max((sum - i64::from(self.main[i])).abs());
            }
        }
        worst
    }

    /// Whether the control words keep their promise: after `Present` the
    /// parts end with `Ending` for a mixer's switch's length before `None`
    /// (unless at one of the frames `gaps`, where the stream starts again
    /// or main cuts to a new track: there it may go straight to `None`).
    pub fn keeps_the_promise(&self, gaps: &[usize]) -> Result<(), String> {
        let e = ending_frames(self.rate) as usize;
        let mut ending = 0;
        for (i, &w) in self.control.iter().enumerate() {
            if w == 0 && i > 0 && !gaps.contains(&i) {
                match self.control[i - 1] {
                    1 => return Err(format!("None right after Present at {i}")),
                    2 if ending < e => {
                        return Err(format!("None after {ending} frames of Ending at {i}"))
                    }
                    _ => {}
                }
            }
            ending = if w == 2 { ending + 1 } else { 0 };
        }
        Ok(())
    }

    /// The frames whose word is `word`.
    pub fn saying(&self, word: i32) -> Vec<usize> {
        (0..self.frames())
            .filter(|&f| self.control[f] == word)
            .collect()
    }
}

/// A lane target taking `caps` (a memory device keeping the lane stream),
/// gated, and its captures.
pub fn wide_target(caps: Capabilities) -> (GatedTarget, Captures, std::sync::Arc<Gate>) {
    wide_target_on(caps, SimClock::Instant)
}

/// [`wide_target`] on a device of `clock`: [`SimClock::Paced`] for one
/// that plays what it holds no faster than real time, its ring filled
/// ahead as a device's is (an instant one never holds anything: a player
/// on a device takes it for one about to run dry).
pub fn wide_target_on(
    caps: Capabilities,
    clock: SimClock,
) -> (GatedTarget, Captures, std::sync::Arc<Gate>) {
    let device = MemoryTarget::with_lanes(caps, clock, 5);
    let captures = device.captures();
    let (target, gate) = GatedTarget::around(Box::new(device));
    (target, captures, gate)
}

/// Plays `files` on a lane target to the end, offline (the parts waited
/// for), with stems made; returns the capture and the events.
pub fn play_wide(
    caps: Capabilities,
    files: Vec<PathBuf>,
    options: PlayerOptions,
) -> (Wide, Vec<Event>) {
    let (target, captures, gate) = wide_target(caps);
    let (player, events) = Player::new(Box::new(target), options);
    player.make_stems(MakeStems::Separate).unwrap();
    gate.allow_all();
    player.load(files).unwrap();
    player.play(0).unwrap();
    let seen = until_queue_end(&events);
    drop(player);
    let wide = Wide::of(&captures);
    wide.keeps_the_promise(&track_starts(&seen)).unwrap();
    (wide, seen)
}

/// The output frames tracks started on.
pub fn track_starts(seen: &[Event]) -> Vec<usize> {
    seen.iter()
        .filter_map(|e| match e {
            Event::TrackStarted(s) => Some(s.output_frame as usize),
            _ => None,
        })
        .collect()
}

pub fn tone_file(dir: &Path, name: &str, rate: u32, frames: usize) -> PathBuf {
    let path = dir.join(name);
    write_wav(&path, rate, 2, &tones(rate, frames));
    path
}

pub fn output_frames(from: u32, to: u32, n: u64) -> u64 {
    jodel_playback::resample::output_frames(from, to, n)
}

/// Lets frames through `step` at a time every couple of milliseconds until
/// the queue has ended, as a device would, many times faster.
pub fn pace(gate: &Gate, events: &Receiver<Event>, step: u64) -> Vec<Event> {
    let deadline = Instant::now() + TIMEOUT;
    let mut seen = Vec::new();
    loop {
        assert!(Instant::now() < deadline, "no end; seen: {seen:#?}");
        gate.allow(gate.written() + step);
        if let Ok(event) = events.recv_timeout(Duration::from_millis(2)) {
            let end = matches!(event, Event::QueueEnded);
            seen.push(event);
            if end {
                return seen;
            }
        }
    }
}

/// Lets frames through in real time (after a first `step`) until a frame
/// captured says the parts are there, as a device would while the
/// separator or the store gives them in its own time; then as [`pace`]
/// does, to the end. Returns the events.
pub fn pace_once_present(
    gate: &Gate,
    events: &Receiver<Event>,
    captures: &Captures,
    step: u64,
) -> Vec<Event> {
    let deadline = Instant::now() + TIMEOUT;
    let start = Instant::now();
    let mut seen = Vec::new();
    loop {
        assert!(
            Instant::now() < deadline,
            "no parts in {} frames; seen: {seen:#?}",
            gate.written()
        );
        let present = captures.list().first().is_some_and(|c| {
            let channels = usize::from(c.format.channels);
            let bytes = c.sink.bytes();
            let frame = (channels * 5 + 1) * 4;
            let whole = bytes.len() / frame * frame;
            whole > 0 && lanes_of(&bytes[whole - frame..whole], channels, 5).1[0] == 1
        });
        if present {
            seen.extend(pace(gate, events, step));
            return seen;
        }
        let real = (start.elapsed().as_secs_f64() * f64::from(RATE)) as u64;
        gate.allow(step + real);
        if let Ok(event) = events.recv_timeout(Duration::from_millis(5)) {
            let end = matches!(event, Event::QueueEnded);
            seen.push(event);
            assert!(!end, "the queue ended without the parts; seen: {seen:#?}");
        }
    }
}

/// [`pace`] until `frames` have gone through.
pub fn pace_to(gate: &Gate, events: &Receiver<Event>, step: u64, frames: u64) -> Vec<Event> {
    let deadline = Instant::now() + TIMEOUT;
    let mut seen = Vec::new();
    while gate.written() < frames {
        assert!(Instant::now() < deadline, "only {} frames", gate.written());
        gate.allow(gate.written() + step);
        if let Ok(event) = events.recv_timeout(Duration::from_millis(2)) {
            seen.push(event);
        }
    }
    gate.allow(gate.written());
    seen
}

/// A store with one track's stems kept.
pub struct Kept {
    track: PathBuf,
    stems: PathBuf,
}

impl StemStore for Kept {
    fn open(&self, path: &Path) -> Option<StemFile> {
        (path == self.track)
            .then(|| StemFile::open(&self.stems).ok())
            .flatten()
    }
}

/// Stems of `samples` (stereo at [`RATE`]) kept for `track`: a quarter
/// each.
pub fn kept(dir: &Path, track: &Path, samples: &[i32]) -> std::sync::Arc<dyn StemStore> {
    let stems = dir.join("kept.stems");
    let mut w = StemWriter::create(&stems, RATE, "quarter").unwrap();
    let full = f32::powi(2.0, 23);
    let side = |c: usize| -> Vec<f32> {
        samples
            .iter()
            .skip(c)
            .step_by(2)
            .map(|&s| s as f32 / full / 4.0)
            .collect()
    };
    let (l, r) = (side(0), side(1));
    w.push(std::array::from_fn(|_| [&l[..], &r[..]])).unwrap();
    w.finish().unwrap();
    std::sync::Arc::new(Kept {
        track: track.to_path_buf(),
        stems,
    })
}

/// Two stores asked in turn.
pub struct Both(
    pub std::sync::Arc<dyn StemStore>,
    pub std::sync::Arc<dyn StemStore>,
);

impl StemStore for Both {
    fn open(&self, path: &Path) -> Option<StemFile> {
        self.0.open(path).or_else(|| self.1.open(path))
    }
}
