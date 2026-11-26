//! What the mixer thread allocates and frees, counted: a test binary makes
//! [`count`] its allocator's (`jodel_testkit::Counting(count)`), turns
//! [`COUNTING`] on and reads [`MIXER_ALLOCATIONS`]. Every call on the thread
//! named `jodel-mixer` counts, except the simulated device's own
//! ([`QuietDevice`]), whose memory grows as it plays on that thread.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use jodel_output::{
    Capabilities, Clock, Error, OpenRequest, Output, OutputTarget, Stats, StreamInfo,
};
use jodel_testkit::Call;

use super::Device;

/// Whether the calls count.
pub static COUNTING: AtomicBool = AtomicBool::new(false);
/// The mixer thread's calls counted.
pub static MIXER_ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

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

/// Counts a call of the mixer thread's, outside the device, while
/// [`COUNTING`] is on.
pub fn count(_: Call) {
    let device = IN_DEVICE.try_with(Cell::get).unwrap_or(true);
    if COUNTING.load(Ordering::Relaxed) && !device && on_the_mixer_thread() {
        MIXER_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    }
}

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

/// A simulated device whose outputs' work is not counted.
pub struct QuietDevice(pub Device);

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
