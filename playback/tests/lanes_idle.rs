//! The engine on a lane input whose ring is full waits for room as it does
//! on a plain output, blocked in its write: through a mixer on a device
//! that plays in real time, with a separator configured, it asks the output
//! a few hundred times a second whether stems are asked for or not, not
//! the millions of a thread that spins.

mod common;

use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::lanes::*;
use common::*;
use jodel_output::lanes::lane_width;
use jodel_output::mixer::{Mixer, MixerOptions, Role};
use jodel_output::{
    Capabilities, Clock, Error as OutputError, OpenRequest, Output, SampleFormat, SimClock, Stats,
    StreamInfo,
};
use jodel_playback::{MakeStems, MemoryTarget, OutputTarget, Player, StemDesk};
use jodel_testkit::{real_time, TempDir};
use SampleFormat::*;

/// What the engine did on a [`Counted`] target's outputs.
#[derive(Default)]
struct Calls {
    /// Every call of the output, writes or not.
    all: AtomicU64,
    /// The control word of the last frame written.
    word: AtomicI32,
}

/// A target whose outputs count what the engine asks of them.
struct Counted {
    inner: Box<dyn OutputTarget>,
    calls: Arc<Calls>,
}

impl OutputTarget for Counted {
    fn name(&self) -> String {
        "counted".into()
    }

    fn capabilities(&mut self) -> Result<Capabilities, OutputError> {
        self.inner.capabilities()
    }

    fn lanes(&mut self) -> u16 {
        self.inner.lanes()
    }

    fn open(&mut self, request: &OpenRequest) -> Result<Box<dyn Output>, OutputError> {
        Ok(Box::new(CountedOutput {
            inner: self.inner.open(request)?,
            calls: self.calls.clone(),
        }))
    }
}

struct CountedOutput {
    inner: Box<dyn Output>,
    calls: Arc<Calls>,
}

impl CountedOutput {
    fn count(&self) {
        self.calls.all.fetch_add(1, Ordering::Relaxed);
    }
}

impl Output for CountedOutput {
    fn info(&self) -> &StreamInfo {
        self.count();
        self.inner.info()
    }
    fn lanes(&self) -> u16 {
        self.count();
        self.inner.lanes()
    }
    fn write(&mut self, samples: &[i32], timeout: Option<Duration>) -> Result<usize, OutputError> {
        self.count();
        let width = lane_width(self.inner.format().channels, self.inner.lanes());
        let done = self.inner.write(samples, timeout)?;
        if done > 0 {
            let word = samples[done * width - 1];
            self.calls.word.store(word, Ordering::Relaxed);
        }
        Ok(done)
    }
    fn drain(&mut self, timeout: Option<Duration>) -> Result<bool, OutputError> {
        self.count();
        self.inner.drain(timeout)
    }
    fn position(&self) -> Result<u64, OutputError> {
        self.count();
        self.inner.position()
    }
    fn written(&self) -> u64 {
        self.count();
        self.inner.written()
    }
    fn latency(&self) -> Duration {
        self.count();
        self.inner.latency()
    }
    fn pause(&mut self) -> Result<(), OutputError> {
        self.inner.pause()
    }
    fn resume(&mut self) -> Result<(), OutputError> {
        self.inner.resume()
    }
    fn is_paused(&self) -> bool {
        self.count();
        self.inner.is_paused()
    }
    fn flush(&mut self) -> Result<(), OutputError> {
        self.inner.flush()
    }
    fn retract(&mut self, frames: u64) -> Result<u64, OutputError> {
        self.count();
        self.inner.retract(frames)
    }
    fn stats(&self) -> Stats {
        self.count();
        self.inner.stats()
    }
    fn clock(&self) -> Clock {
        self.count();
        self.inner.clock()
    }
    fn close(&mut self) -> Result<(), OutputError> {
        self.inner.close()
    }
}

/// The output's frame rate.
const OUT: u32 = 48_000;

/// Calls of the output in a second of steady playback at the most: a write
/// blocks for up to 10 ms and the mixer frees a period at a time, so the
/// engine asks a few hundred times a second; one that spins asks some
/// million times, even on a slow machine.
const CALLS: u64 = 20_000;

/// Plays a ten-second file through a mixer's lane input on a device that
/// plays in real time, with stems made as `make` says by a separator that
/// takes a quarter of the mix each, and returns the calls of the output in
/// a second of steady playback (from `present`: once the stems are there).
fn calls_in_a_second(make: MakeStems, present: bool) -> u64 {
    let dir = TempDir::new(&format!("idle-{make:?}"));
    let file = tone_file(&dir, "tones.wav", OUT, 10 * OUT as usize);
    let device = MemoryTarget::new(caps(&[OUT], &[S24In32Low]), SimClock::Realtime);
    let mixer = Mixer::open(Box::new(device), MixerOptions::default());
    let (desk, head) = StemDesk::new(&mixer, "Player", Role::Lead, 1.0);
    let calls = Arc::new(Calls::default());
    let target = Counted {
        inner: head.target,
        calls: calls.clone(),
    };
    let (player, events) = Player::new(Box::new(target), options(false, quarter));
    player.make_stems(make).unwrap();
    player.load(vec![file]).unwrap();
    player.play(0).unwrap();
    // The ring filled: a second played (and the stems there, when asked).
    let deadline = Instant::now() + TIMEOUT;
    let mut started = None;
    loop {
        assert!(Instant::now() < deadline, "no steady playback");
        while let Ok(event) = events.try_recv() {
            if matches!(event, jodel_playback::Event::Error { .. }) {
                panic!("{event:?}");
            }
        }
        let there = !present || calls.word.load(Ordering::Relaxed) == 1;
        match started {
            None if there => started = Some(Instant::now()),
            Some(at) if there && at.elapsed() >= Duration::from_secs(1) => break,
            _ if !there => started = None,
            _ => {}
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let (from, at) = (calls.all.load(Ordering::Relaxed), Instant::now());
    std::thread::sleep(Duration::from_secs(1));
    let n = calls.all.load(Ordering::Relaxed) - from;
    let per_second = (n as f64 / at.elapsed().as_secs_f64()) as u64;
    drop(player);
    drop(desk);
    drop(mixer);
    per_second
}

#[test]
fn a_full_lane_input_with_no_stems_asked_for_is_waited_on() {
    let _guard = real_time();
    let n = calls_in_a_second(MakeStems::Never, false);
    println!("no stems asked for: {n} calls of the output a second");
    assert!(
        n < CALLS,
        "{n} calls of the output a second: the engine spins"
    );
}

#[test]
fn a_full_lane_input_with_its_stems_there_is_waited_on() {
    let _guard = real_time();
    let n = calls_in_a_second(MakeStems::Separate, true);
    println!("stems separated and there: {n} calls of the output a second");
    assert!(
        n < CALLS,
        "{n} calls of the output a second: the engine spins"
    );
}
