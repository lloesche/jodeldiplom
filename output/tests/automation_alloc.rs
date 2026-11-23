//! What the mixer thread allocates and frees while it evaluates automation:
//! nothing. A counting allocator (this binary's alone) counts every call on
//! the thread named `jodel-mixer` while a lane input plays with every one
//! of its curves moving on every frame, so each gain is worked out frame by
//! frame; the simulated device's own memory, which grows as it plays on
//! that thread, is left out.
//!
//! A set replaced (`Fader::automate` again) is freed on the mixer thread,
//! as every message it takes is (the channel's blocks, a crossfade's list,
//! an input forgotten): that is not counted here, only the mixing.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use common::*;
use jodel_output::lanes::{lane_width, Parts};
use jodel_output::mixer::{Automation, GainCurve, Mixer, MixerOptions, Role};
use jodel_output::{
    Capabilities, Clock, Error, OpenRequest, Output, OutputTarget, Stats, StreamInfo,
};

/// Counts what the mixer thread allocates and frees while counting is on.
struct Counting;

static COUNTING: AtomicBool = AtomicBool::new(false);
static MIXER_ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// 0: not known yet; 1: being found out; 2: the mixer's; 3: another.
    static THREAD: Cell<u8> = const { Cell::new(0) };
    /// Inside the simulated device: not the mixer's to count.
    static IN_DEVICE: Cell<bool> = const { Cell::new(false) };
}

fn on_the_mixer_thread() -> bool {
    THREAD
        .try_with(|t| match t.get() {
            0 => {
                t.set(1);
                let mixer = std::thread::current().name() == Some("jodel-mixer");
                t.set(if mixer { 2 } else { 3 });
                mixer
            }
            k => k == 2,
        })
        .unwrap_or(false)
}

fn count() {
    let device = IN_DEVICE.try_with(Cell::get).unwrap_or(true);
    if COUNTING.load(Ordering::Relaxed) && !device && on_the_mixer_thread() {
        MIXER_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    }
}

// SAFETY: every call goes to the system allocator as it came.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        count();
        System.dealloc(ptr, layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count();
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count();
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// `f` run as the simulated device's, its allocations not counted.
fn in_device<T>(f: impl FnOnce() -> T) -> T {
    IN_DEVICE.with(|d| d.set(true));
    let t = f();
    IN_DEVICE.with(|d| d.set(false));
    t
}

/// The test's device, its own work kept out of the mixer's count.
struct Quiet(Box<dyn Output>);

impl Output for Quiet {
    fn info(&self) -> &StreamInfo {
        self.0.info()
    }
    fn write(&mut self, samples: &[i32], timeout: Option<Duration>) -> Result<usize, Error> {
        in_device(|| self.0.write(samples, timeout))
    }
    fn drain(&mut self, timeout: Option<Duration>) -> Result<bool, Error> {
        in_device(|| self.0.drain(timeout))
    }
    fn position(&self) -> Result<u64, Error> {
        in_device(|| self.0.position())
    }
    fn written(&self) -> u64 {
        self.0.written()
    }
    fn latency(&self) -> Duration {
        self.0.latency()
    }
    fn pause(&mut self) -> Result<(), Error> {
        in_device(|| self.0.pause())
    }
    fn resume(&mut self) -> Result<(), Error> {
        in_device(|| self.0.resume())
    }
    fn is_paused(&self) -> bool {
        self.0.is_paused()
    }
    fn flush(&mut self) -> Result<(), Error> {
        in_device(|| self.0.flush())
    }
    fn stats(&self) -> Stats {
        in_device(|| self.0.stats())
    }
    fn clock(&self) -> Clock {
        self.0.clock()
    }
    fn close(&mut self) -> Result<(), Error> {
        in_device(|| self.0.close())
    }
}

struct QuietDevice(Device);

impl OutputTarget for QuietDevice {
    fn name(&self) -> String {
        self.0.name()
    }
    fn capabilities(&mut self) -> Result<Capabilities, Error> {
        self.0.capabilities()
    }
    fn open(&mut self, request: &OpenRequest) -> Result<Box<dyn Output>, Error> {
        Ok(Box::new(Quiet(self.0.open(request)?)))
    }
}

const PARTS: u16 = 4;

/// `n` lane frames: main `x` on the left, each part a quarter, the parts
/// there.
fn lane_frames(n: u64) -> Vec<i32> {
    let x = 4_000_000;
    let mut v = Vec::with_capacity(n as usize * lane_width(2, 1 + PARTS));
    for _ in 0..n {
        v.extend([x, 0]);
        for _ in 0..PARTS {
            v.extend([x / 4, 0]);
        }
        v.push(Parts::Present.word());
    }
    v
}

#[test]
fn evaluating_curves_allocates_nothing_on_the_mixer_thread() {
    let device = Device {
        instant: true,
        ..Device::default()
    };
    let options = MixerOptions {
        buffer: Duration::from_millis(200),
        ..MixerOptions::default()
    };
    let mixer = Mixer::open(Box::new(QuietDevice(device.clone())), options);
    let player = mixer.strip("Player");
    let parts: Vec<_> = ["Drums", "Bass", "Vocals", "Other"]
        .iter()
        .map(|name| mixer.strip_in(&player, name))
        .collect();
    let refs: Vec<_> = parts.iter().collect();
    let (mut target, fader, _parts) = mixer.input_lanes(&player, &refs, Role::Lead, 1.0);
    let request = OpenRequest::new(RATE, 2)
        .formats(&[device.format()])
        .lanes(1 + PARTS)
        .prefill(Duration::ZERO);
    let mut out = target.open(&request).unwrap();
    out.write(&lane_frames(1), None).unwrap();
    // Every curve moving on every frame (one with a cut), so every gain is
    // one a frame.
    let moving = |to: f64| GainCurve::new(&[(0, 1.0), (48_000, to)]).unwrap();
    let cut = GainCurve::new(&[(0, 1.0), (10_000, 0.9), (10_000, 0.5), (48_000, 0.1)]).unwrap();
    let mut auto = Automation::new().input(moving(0.5)).main(cut);
    for k in 0..usize::from(PARTS) {
        auto = auto.part(k, moving(0.1 * k as f64));
    }
    fader.automate(auto).unwrap();
    taken(&mut target);
    // Each gain's buffer grown to a chunk first.
    out.write(&lane_frames(4_799), None).unwrap();
    wait_until("mixed", || mixer.timebase().frame == 4_800);
    let data = lane_frames(24_000 - 4_800);
    MIXER_ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    out.write(&data, None).unwrap();
    wait_until("mixed", || mixer.timebase().frame == 24_000);
    COUNTING.store(false, Ordering::Relaxed);
    assert_eq!(MIXER_ALLOCATIONS.load(Ordering::Relaxed), 0);
    assert!(!mixer.passthrough(), "the curves were mixed");
    out.drain(None).unwrap();
}
