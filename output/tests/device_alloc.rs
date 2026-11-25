//! What the real-time threads allocate and free after they are made:
//! nothing. A counting allocator (this binary's alone) counts every call
//! on a simulated device's thread and on the mixer's, each marked as such
//! by code that runs there (a tap on the device thread, a monitor on the
//! mixer's):
//!
//! - a device output in real time, written, flushed (a seek), starved
//!   (underruns), paused and resumed, with a tap set;
//! - a mixer on such a device, with a microphone tapping it, through
//!   twenty rounds of outputs opened and closed on its inputs, flushed,
//!   their roles swapped (the microphone's tap set on each new Lead) and
//!   the microphone replaced; nothing it hands its helper thread has to
//!   wait past the room kept for it.
//!
//! What the mixer thread takes in a message is freed there (the channel's
//! blocks): frees are counted apart, and only the device thread's must be
//! none.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use common::*;
use jodel_output::mixer::{MicPort, Mixer, MixerOptions, Monitor, Role};
use jodel_output::{
    Capabilities, Clock, Error, FileOutput, OpenRequest, Output, OutputTarget, SimClock,
    StreamFormat, StreamInfo, Tap,
};

struct Counting;

static COUNTING: AtomicBool = AtomicBool::new(false);
static DEVICE_ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static DEVICE_FREES: AtomicU64 = AtomicU64::new(0);
static MIXER_ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static MIXER_FREES: AtomicU64 = AtomicU64::new(0);

/// What a thread was marked as.
const OTHER: u8 = 0;
const DEVICE: u8 = 1;
const MIXER: u8 = 2;

thread_local! {
    static THREAD: Cell<u8> = const { Cell::new(OTHER) };
}

/// Marks this thread as `kind`'s.
fn mark(kind: u8) {
    THREAD.with(|t| t.set(kind));
}

fn count(free: bool) {
    if !COUNTING.load(Ordering::Relaxed) {
        return;
    }
    let kind = THREAD.try_with(Cell::get).unwrap_or(OTHER);
    let counter = match (kind, free) {
        (DEVICE, false) => &DEVICE_ALLOCATIONS,
        (DEVICE, true) => &DEVICE_FREES,
        (MIXER, false) => &MIXER_ALLOCATIONS,
        (MIXER, true) => &MIXER_FREES,
        _ => return,
    };
    counter.fetch_add(1, Ordering::Relaxed);
}

// SAFETY: every call goes to the system allocator as it came.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(false);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        count(true);
        System.dealloc(ptr, layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(false);
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(false);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// The counts of `f`: the device thread's allocations and frees, the
/// mixer thread's.
fn counted(f: impl FnOnce()) -> [u64; 4] {
    let all = [
        &DEVICE_ALLOCATIONS,
        &DEVICE_FREES,
        &MIXER_ALLOCATIONS,
        &MIXER_FREES,
    ];
    for c in all {
        c.store(0, Ordering::Relaxed);
    }
    COUNTING.store(true, Ordering::SeqCst);
    f();
    COUNTING.store(false, Ordering::SeqCst);
    all.map(|c| c.load(Ordering::Relaxed))
}

fn format() -> StreamFormat {
    StreamFormat::new(RATE, 2, FORMAT)
}

/// Marks the device thread it runs on, and counts what it sees.
struct Marks(Arc<AtomicU64>);

impl Tap for Marks {
    fn begin(&mut self, _: usize, _: u64, _: usize, _: StreamFormat) {
        mark(DEVICE);
    }

    fn see(&mut self, samples: &[i32], _: StreamFormat) {
        self.0.fetch_add(samples.len() as u64, Ordering::Relaxed);
    }
}

/// A tap that allocates as it looks: what the count must see.
struct Allocates;

impl Tap for Allocates {
    fn see(&mut self, samples: &[i32], _: StreamFormat) {
        std::hint::black_box(samples.to_vec());
    }
}

#[test]
fn a_device_thread_allocates_and_frees_nothing() {
    let _guard = real_time();
    let mut out = FileOutput::new(std::io::sink(), format(), SimClock::Realtime);
    let tenth = frames(0, 4_800, |k| k as i32);
    let seen = Arc::new(AtomicU64::new(0));
    out.set_tap(Some(Box::new(Marks(seen.clone())))).unwrap();
    out.write(&tenth, None).unwrap();
    wait_until("it plays", || seen.load(Ordering::Relaxed) > 0);
    // The count sees the device thread: a tap that allocates there.
    out.set_tap(Some(Box::new(Allocates))).unwrap();
    let control = counted(|| {
        out.write(&tenth, None).unwrap();
        out.drain(None).unwrap();
    });
    assert!(control[0] > 0 && control[1] > 0, "{control:?}");
    out.set_tap(Some(Box::new(Marks(seen.clone())))).unwrap();
    let counts = counted(|| {
        for round in 0..8 {
            // Played through, then starved.
            out.write(&tenth, None).unwrap();
            out.write(&tenth, None).unwrap();
            let written = out.written();
            wait_until("played out", || out.position().unwrap() == written);
            std::thread::sleep(Duration::from_millis(30));
            // A seek: what is queued goes.
            out.write(&tenth, None).unwrap();
            out.flush().unwrap();
            assert_eq!(out.position().unwrap(), out.written(), "round {round}");
            out.write(&tenth, None).unwrap();
            out.pause().unwrap();
            out.resume().unwrap();
        }
        out.drain(None).unwrap();
    });
    assert_eq!(
        counts[..2],
        [0, 0],
        "the device thread's allocations and frees"
    );
    let stats = out.stats();
    assert!(stats.underruns >= 8, "{stats:?}");
}

/// A device in real time that plays into nothing, its thread marked by a
/// tap until the microphone's takes its place.
#[derive(Clone, Default)]
struct Sink(Arc<AtomicU64>);

impl OutputTarget for Sink {
    fn name(&self) -> String {
        "sink".into()
    }

    fn capabilities(&mut self) -> Result<Capabilities, Error> {
        Ok([format()].into_iter().collect())
    }

    fn open(&mut self, request: &OpenRequest) -> Result<Box<dyn Output>, Error> {
        let caps = self.capabilities()?;
        let mut out = FileOutput::open(std::io::sink(), &caps, request, SimClock::Realtime)?;
        out.set_tap(Some(Box::new(Marks(self.0.clone()))))?;
        Ok(Box::new(out))
    }
}

/// Marks the mixer thread it runs on.
struct MarksMixer(Arc<AtomicBool>);

impl Monitor for MarksMixer {
    fn opened(&mut self, _: &StreamInfo, _: Clock) {}

    fn mixed(&mut self, _: &[i32], _: usize, _: u64) {
        mark(MIXER);
        self.0.store(true, Ordering::Relaxed);
    }

    fn closed(&mut self) {}
}

/// An output on `target`, started with a tenth of a second.
fn started(target: &mut jodel_output::mixer::InputTarget) -> Box<dyn Output> {
    let mut out = target.open(&request(RATE).prefill(Duration::ZERO)).unwrap();
    out.write(&frames(0, 4_800, |_| 1_000), None).unwrap();
    out
}

#[test]
fn the_mixer_thread_allocates_nothing_for_outputs_and_the_microphone() {
    let _guard = real_time();
    let options = MixerOptions {
        buffer: Duration::from_millis(200),
        ..MixerOptions::default()
    };
    let device = Sink::default();
    let mixer = Mixer::open(Box::new(device.clone()), options);
    let marked = Arc::new(AtomicBool::new(false));
    mixer.set_monitor(Some(Box::new(MarksMixer(marked.clone()))));
    wait_until("both threads marked", || {
        marked.load(Ordering::Relaxed) && device.0.load(Ordering::Relaxed) > 0
    });
    let strip = mixer.strip("Player");
    let singers = [Singer::new(loud), Singer::new(loud)];
    mixer.set_mic(Some(singers[0].input.clone()));
    let (mut lead, _lead_fader) = mixer.input(&strip, Role::Lead, 1.0);
    let (mut guest, guest_fader) = mixer.input(&strip, Role::Guest, 0.5);
    let port = |want: MicPort| wait_until("the port", || mixer.mic().port == want);
    let mut round = |k: usize| {
        // Each Lead output opened is tapped for the microphone.
        let mut out = started(&mut lead);
        port(MicPort::Hearing);
        let g = started(&mut guest);
        // A seek on the Lead: the device keeps the guest's frames.
        out.flush().unwrap();
        out.write(&frames(0, 4_800, |_| 1_000), None).unwrap();
        // The guest made Lead (the tap goes to it), and back.
        guest_fader.set_role(Role::Lead).unwrap();
        guest_fader.set_role(Role::Guest).unwrap();
        // Another microphone.
        mixer.set_mic(Some(singers[k % 2].input.clone()));
        let written = g.written();
        wait_until("the guest plays", || g.position().unwrap() == written);
        drop(g);
        drop(out);
        port(MicPort::Idle);
    };
    // The inputs' and the strips' room is made as they first play.
    for k in 0..2 {
        round(k);
    }
    let counts = counted(|| {
        for k in 0..20 {
            round(k);
        }
    });
    let [device, device_frees, mixer_allocations, mixer_frees] = counts;
    println!("mixer thread: {mixer_allocations} allocations, {mixer_frees} frees");
    assert_eq!(mixer_allocations, 0, "the mixer thread's allocations");
    assert_eq!([device, device_frees], [0, 0], "the device thread's");
    assert_eq!(mixer.helper_overflows(), 0);
    assert_eq!(mixer.mic().refused, 0);
}
