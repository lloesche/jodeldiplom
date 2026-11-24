//! What the mixer thread allocates and frees while the effects input plays:
//! nothing, from the very first sound after the device opens, on each
//! opening (here a second one at another rate, with more channels). A
//! counting allocator (this binary's alone) counts every call on the
//! thread named `jodel-mixer` while sounds are cued on stream frames, wait
//! for them, start (more at once than there are voices, so one takes
//! another's place) and are summed with the input; the simulated device's
//! own memory, which grows as it plays on that thread, is left out. The
//! bank, the effects input and the device's opening are messages (freed
//! and reserved as the mixer thread takes them), the cues are not: they
//! come through a queue of fixed size. The song plays a little before the
//! sounds, so its own stream's buffers are at their sizes.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use common::*;
use jodel_output::mixer::{EffectCue, Mixer, MixerOptions, Role, SoundBank, VOICES};
use jodel_output::{
    Capabilities, Clock, Error, OpenRequest, Output, OutputTarget, Stats, StreamFormat, StreamInfo,
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

/// The test's devices: stereo, and six channels, by what is asked; their
/// own work kept out of the mixer's count.
struct QuietDevices {
    stereo: Device,
    six: Device,
}

impl OutputTarget for QuietDevices {
    fn name(&self) -> String {
        self.stereo.name()
    }
    fn capabilities(&mut self) -> Result<Capabilities, Error> {
        let mut formats = Vec::new();
        for d in [&self.stereo, &self.six] {
            for rate in [RATE, 96_000] {
                formats.push(StreamFormat::new(rate, d.channels(), d.format()));
            }
        }
        Ok(formats.into_iter().collect())
    }
    fn open(&mut self, request: &OpenRequest) -> Result<Box<dyn Output>, Error> {
        let device = if request.channels == 6 {
            &mut self.six
        } else {
            &mut self.stereo
        };
        Ok(Box::new(Quiet(device.open(request)?)))
    }
}

/// `n` frames of `channels` channels from stream frame `from`: the first
/// channel the frame's number.
fn wide(from: u64, n: u64, channels: usize) -> Vec<i32> {
    (from..from + n)
        .flat_map(|k| std::iter::once(k as i32).chain(std::iter::repeat_n(0, channels - 1)))
        .collect()
}

/// A bank at `rate` of one sound, 2 400 frames.
fn bank(rate: u32) -> SoundBank {
    let mut bank = SoundBank::new(rate);
    bank.add(
        (0..2_400)
            .flat_map(|k| [0.1 * (k as f32 * 0.05).sin(); 2])
            .collect(),
    );
    bank
}

#[test]
fn playing_effects_allocates_nothing_on_the_mixer_thread() {
    let stereo = Device {
        instant: true,
        ..Device::default()
    };
    let six = Device {
        channels: Some(6),
        ..stereo.clone()
    };
    let options = MixerOptions {
        buffer: Duration::from_millis(200),
        ..MixerOptions::default()
    };
    let devices = QuietDevices {
        stereo: stereo.clone(),
        six: six.clone(),
    };
    let mixer = Mixer::open(Box::new(devices), options);
    let player = mixer.strip("Player");
    let strip = mixer.strip("Effects");
    let effects = mixer.effects(&strip);
    strip.set_level(0.5);
    let (mut target, fader) = mixer.input(&player, Role::Lead, 1.0);
    let mut played = 0;
    for (rate, channels, device) in [(RATE, 2, &stereo), (96_000, 6, &six)] {
        effects.load(bank(rate));
        let request = OpenRequest::new(rate, channels as u16)
            .formats(&[device.format()])
            .prefill(Duration::ZERO);
        let mut out = target.open(&request).unwrap();
        taken(&mut target);
        // The song alone first: its stream's buffers at their sizes.
        out.write(&wide(0, 1_920, channels), None).unwrap();
        wait_until("mixed", || mixer.timebase().frame == 1_920);
        MIXER_ALLOCATIONS.store(0, Ordering::Relaxed);
        COUNTING.store(true, Ordering::Relaxed);
        // The first sound of this opening; then more at once than there
        // are voices; then one every 1 000 frames.
        effects.play_at(EffectCue::new(0), &fader, 2_000);
        for _ in 0..VOICES + 4 {
            effects.play_at(EffectCue::new(0), &fader, 5_000);
        }
        for k in 0..18 {
            effects.play_at(EffectCue::new(0), &fader, 6_000 + 1_000 * k);
        }
        out.write(&wide(1_920, 22_080, channels), None).unwrap();
        wait_until("mixed", || mixer.timebase().frame == 24_000);
        COUNTING.store(false, Ordering::Relaxed);
        assert_eq!(MIXER_ALLOCATIONS.load(Ordering::Relaxed), 0, "{rate}");
        played += 1 + VOICES as u64 + 4 + 18;
        let view = effects.view();
        assert_eq!(view.played, played, "{rate}: {view:?}");
        assert_eq!(view.rate, rate);
        out.drain(None).unwrap();
    }
}
