//! Microphone input: the singer heard, never played.
//!
//! Nothing of the microphone reaches the output. It is listened to, to
//! measure the singing: its voice, the music's echo taken out, is kept
//! block by block with where in the stream it was sung.
//!
//! To take the echo out, the canceller needs what played at the moment the
//! microphone heard it. An engine writes far ahead of the DAC (up to the
//! output's ring, 500 ms by default), so that is taken where the device
//! takes its buffers: a [`crate::Tap`] in the device thread sees each one
//! (and cannot change it), and lines it up with the microphone's samples of
//! that moment. The canceller itself runs on a thread of its own.
//!
//! # Use
//!
//! 1. [`input_devices`] lists the capture devices; [`open_input`] opens one
//!    in shared mode at its own rate, with a period of about 5 ms where the
//!    system allows it. Capture runs on its own thread from then on, mixed
//!    down to mono ([`Mono`]) into a lock-free ring of `f32`.
//! 2. [`Input::tap`] makes the [`MicTap`] to hand to
//!    [`Output::set_tap`](crate::Output::set_tap) (a mixer does it itself:
//!    [`crate::mixer::Mixer::set_mic`]). In the device thread it
//!    resamples the voice to the output's rate and queues it, with what
//!    plays (left and right), for the cleaner: a thread that takes the
//!    music's echo out ([`crate::aec`]) and suppresses what the canceller
//!    left of it ([`crate::aec::Suppressor`]).
//! 3. [`Input::take_voice`] gives the voice cleaned, block by block
//!    ([`VoiceBlock`]); [`Input::voice_peak`] is a meter of the singer
//!    isolated from it ([`crate::isolate`]: the band the canceller missed
//!    and the room's sounds taken out too) and [`Input::peak`] one of the
//!    voice as captured; [`Input::status`] tells
//!    how the two clocks are held together.
//!
//! Capture stops when the [`Input`] and every tap made from it are dropped.
//!
//! # Two clocks
//!
//! The microphone and the DAC run on different crystals. The tap's
//! resampler (cubic, varispeed) runs at the nominal ratio of the two rates
//! corrected by at most ±0.2 %: a control loop holds the ring's fill at
//! about two capture periods. Measured at the moment a device buffer is
//! made, the fill would rise and fall with where the last capture packet
//! fell; the time since that packet is counted in, so the loop sees the
//! fill as if the voice arrived continuously.
//!
//! When the ring runs dry (the capture stalled) the voice drops out and
//! starts again once two periods are queued. When the fill is held more
//! than a period over its target (a burst of capture, as devices deliver
//! when they start, or the device thread stalled), the excess is skipped
//! at once rather than drained at the loop's limit. Both are counted in
//! [`MicStatus`].
//!
//! # Tests
//!
//! [`Input::feed`] makes an input fed by the caller instead of a device, on
//! a simulated clock, its cleaner run by the caller ([`MicFeed::clean`]):
//! with [`crate::FileOutput`] the whole path runs without hardware.
//! [`Input::feed_in_real_time`] is one on the system's clock, for a caller
//! feeding it as a device would, from a thread of its own.
//!
//! # Parts
//!
//! The capture side (`capture`: the ring, what all sides share, the
//! mixdown), the clock bridge (`bridge`), the tap in the device thread
//! (`tap`), the cleaner (`clean`: the echo canceller with what played as
//! its sidechain, then the singer isolated) and the canceller's recording
//! (`record`).

mod bridge;
mod capture;
mod clean;
mod record;
mod tap;

pub(crate) use capture::{MicClock, MicShared, Opened, Producer, Wire};
pub use clean::VoiceBlock;
pub use record::EchoRecording;
pub use tap::MicTap;

use crate::backend;
use crate::device::Device;
use crate::error::Error;
use clean::Cleaner;
use record::write_recording;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

/// The capture backend this build uses: `"WASAPI (shared)"`, `"Core Audio
/// (shared)"`, `"ALSA default (shared)"`, or `"none"`.
pub const INPUT_BACKEND: &str = backend::INPUT_BACKEND;

/// The capture period asked for.
pub const CAPTURE_PERIOD: Duration = Duration::from_millis(5);

/// The furthest the resampler's ratio moves from the nominal one.
pub const MAX_ADJUST: f64 = 0.002;

/// How a multichannel input becomes one voice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mono {
    /// One channel as it is, the mean of two, the first of more.
    #[default]
    Auto,
    /// The first channel.
    First,
    /// The mean of the first two channels (the first, with one).
    Mean,
    /// This channel (from 0); the first, if there is no such channel.
    Channel(u16),
}

impl Mono {
    fn pick(self, channels: usize) -> Pick {
        match self {
            Mono::Auto if channels == 2 => Pick::Mean,
            Mono::Mean if channels >= 2 => Pick::Mean,
            Mono::Channel(c) if usize::from(c) < channels => Pick::One(usize::from(c)),
            _ => Pick::One(0),
        }
    }
}

#[derive(Clone, Copy)]
enum Pick {
    One(usize),
    Mean,
}

/// The capture devices that are present now. `Device::id` is what
/// [`open_input`] takes.
pub fn input_devices() -> Result<Vec<Device>, Error> {
    backend::input_devices()
}

/// The system's default capture device.
pub fn default_input() -> Result<Device, Error> {
    input_devices()?
        .into_iter()
        .find(|d| d.is_default)
        .ok_or(Error::NoDevice)
}

/// Finds a capture device by `query`: `"default"`, an exact id, an exact
/// name, or a case-insensitive part of exactly one device's name.
pub fn find_input(query: &str) -> Result<Device, Error> {
    if query.eq_ignore_ascii_case("default") {
        return default_input();
    }
    crate::device::pick(input_devices()?, query)
}

/// Opens the capture device `id` (shared, at its own rate) and starts
/// capturing, mixed down as [`Mono::Auto`].
pub fn open_input(id: &str) -> Result<Input, Error> {
    open_input_with(id, Mono::Auto)
}

/// [`open_input`] with the mixdown given.
pub fn open_input_with(id: &str, mono: Mono) -> Result<Input, Error> {
    backend::open_input(id, mono)
}

/// What an open input runs at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputInfo {
    /// The device's id.
    pub id: String,
    /// The device's name.
    pub name: String,
    /// Which backend captures it.
    pub backend: &'static str,
    /// The rate it captures at, as opened ([`Input::rate`] follows a change).
    pub rate: u32,
    /// The device's channels.
    pub channels: u16,
    /// The capture period the system gave, in frames.
    pub period_frames: u32,
    /// How the channels become one voice.
    pub mono: Mono,
}

/// How the microphone path is doing; see [`Input::status`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MicStatus {
    /// The capture rate.
    pub rate: u32,
    /// The capture period the system gave, in frames.
    pub period_frames: u32,
    /// The largest capture packet seen, in frames.
    pub packet_frames: u32,
    /// The ring's fill as the control loop sees it (smoothed), in capture
    /// frames.
    pub fill: f64,
    /// The fill it holds: two capture periods (of the larger of the
    /// period and the largest packet).
    pub target: u32,
    /// The correction to the nominal ratio of the two clocks: positive when
    /// the microphone runs fast against the DAC. Within ±[`MAX_ADJUST`].
    pub adjust: f64,
    /// Whether a tap is consuming the voice now.
    pub active: bool,
    /// Times the ring ran dry and the voice dropped out.
    pub underruns: u64,
    /// Capture frames lost because the ring was full while the clock
    /// bridge read the voice (it fell a second behind). Before it reads
    /// (no tap yet, or warming up), the ring keeps the first second that
    /// came and drops the rest, and the bridge starts from the newest of
    /// that: nothing anyone wanted is lost, and nothing is counted.
    pub overruns: u64,
    /// Times the fill was held more than a period over its target and the
    /// excess was skipped.
    pub resyncs: u64,
    /// Blocks lost on the way to the cleaner (its thread fell behind).
    pub blocks_dropped: u64,
    /// Cleaned blocks lost because nothing took them ([`Input::take_voice`]).
    pub voice_dropped: u64,
}

impl MicStatus {
    /// The fill as time.
    pub fn buffered(&self) -> Duration {
        if self.rate == 0 {
            return Duration::ZERO;
        }
        Duration::from_secs_f64(self.fill.max(0.0) / f64::from(self.rate))
    }
}

// ---------------------------------------------------------------------------
// The handle

struct InputInner {
    shared: Arc<MicShared>,
    info: InputInfo,
    /// The backend's stream; dropping it stops capture.
    stream: Mutex<Option<Box<dyn Send>>>,
    /// The cleaner's thread (none for a [`MicFeed`], whose caller runs it).
    cleaner: Option<CleanerThread>,
}

/// The thread a device's input cleans on, and what tells it to stop.
struct CleanerThread {
    stop: Arc<AtomicBool>,
    thread: JoinHandle<()>,
}

impl Drop for InputInner {
    fn drop(&mut self) {
        let stream = self
            .stream
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        drop(stream);
        if let Some(cleaner) = self.cleaner.take() {
            cleaner.stop.store(true, Ordering::Release);
            let _ = cleaner.thread.join();
        }
    }
}

/// An open capture device (or a [`MicFeed`]). Cheap to clone; capture stops
/// when the last clone and every [`MicTap`] made from it are dropped.
#[derive(Clone)]
pub struct Input {
    inner: Arc<InputInner>,
}

impl Input {
    /// A device's input, as its backend opened it: its cleaner runs on a
    /// thread of its own.
    pub(crate) fn new(info: InputInfo, opened: Opened, stream: Box<dyn Send>) -> Input {
        let Opened {
            shared,
            mut cleaner,
        } = opened;
        let stop = Arc::new(AtomicBool::new(false));
        let cleaner = {
            let shared = Arc::clone(&shared);
            let stopped = Arc::clone(&stop);
            std::thread::Builder::new()
                .name("jodel-mic-clean".into())
                .spawn(move || {
                    while !stopped.load(Ordering::Acquire) {
                        if !cleaner.run(&shared) {
                            std::thread::sleep(Duration::from_millis(5));
                        }
                    }
                })
                .ok()
                .map(|thread| CleanerThread { stop, thread })
        };
        Input::wrap_running(info, shared, stream, cleaner)
    }

    /// The handle over a capture that runs: its stream, and its cleaner's
    /// thread if it has one.
    fn wrap_running(
        info: InputInfo,
        shared: Arc<MicShared>,
        stream: Box<dyn Send>,
        cleaner: Option<CleanerThread>,
    ) -> Input {
        Input {
            inner: Arc::new(InputInner {
                shared,
                info,
                stream: Mutex::new(Some(stream)),
                cleaner,
            }),
        }
    }

    /// An input the caller feeds ([`MicFeed::push`]) instead of a device,
    /// at `rate` with packets of `period_frames`, on the caller's clock
    /// ([`MicFeed::set_time`]). For tests and simulations.
    pub fn feed(rate: u32, period_frames: u32) -> (Input, MicFeed) {
        Self::fed(rate, period_frames, MicClock::caller())
    }

    /// An input the caller feeds as [`Input::feed`] does, but on the
    /// system's clock, as a device's ([`MicFeed::set_time`] is not needed):
    /// for a caller pushing its packets in real time from a thread of its
    /// own. The clock bridge then counts in the time since the last packet,
    /// as it does for a device, however late that thread wakes; on a
    /// simulated clock, set only when the thread wakes, it never sees any.
    pub fn feed_in_real_time(rate: u32, period_frames: u32) -> (Input, MicFeed) {
        Self::fed(rate, period_frames, MicClock::system())
    }

    fn fed(rate: u32, period_frames: u32, clock: MicClock) -> (Input, MicFeed) {
        let info = InputInfo {
            id: "feed".into(),
            name: "feed".into(),
            backend: "feed",
            rate,
            channels: 1,
            period_frames,
            mono: Mono::Auto,
        };
        let (producer, Opened { shared, cleaner }) =
            Opened::new(rate, period_frames, Mono::Auto, clock);
        let feed = MicFeed { producer, cleaner };
        (Input::wrap_running(info, shared, Box::new(()), None), feed)
    }

    fn shared(&self) -> &MicShared {
        &self.inner.shared
    }

    /// What it was opened at.
    pub fn info(&self) -> &InputInfo {
        &self.inner.info
    }

    /// The capture rate now.
    pub fn rate(&self) -> u32 {
        self.shared().capture.rate.load(Ordering::Acquire)
    }

    /// The voice's recent peak as captured: 0.0 to 1.0 (full scale),
    /// falling by about 30 dB a second.
    pub fn peak(&self) -> f32 {
        f32::from_bits(self.shared().capture.peak.load(Ordering::Relaxed))
    }

    /// Turns the echo canceller on or off ([`crate::aec`]): on (as it
    /// opens), what the output plays is taken out of the voice the cleaner
    /// gives; off, the voice is the microphone as it is. Any thread.
    pub fn set_echo_cancel(&self, on: bool) {
        self.shared().echo_on.store(on, Ordering::Relaxed);
    }

    /// Whether the echo canceller is on.
    pub fn echo_cancel(&self) -> bool {
        self.shared().echo_on.load(Ordering::Relaxed)
    }

    /// The recent peak of the singer isolated: the music's echo taken out,
    /// then what the canceller left of the band and the room's sounds
    /// ([`crate::isolate`]), 11 to 21 ms late. It falls like
    /// [`Input::peak`]; `None` while the cleaner has given
    /// none lately (no output tapped: nothing plays to take out, and
    /// [`Input::peak`] is then the voice).
    pub fn voice_peak(&self) -> Option<f32> {
        let s = self.shared();
        let at = s.cleaned.voice_at.load(Ordering::Relaxed);
        let fresh = at != 0 && s.now_ns().saturating_sub(at - 1) < 500_000_000;
        fresh.then(|| f32::from_bits(s.cleaned.voice_peak.load(Ordering::Relaxed)))
    }

    /// Takes the voice cleaned since the last call into `out` (appended),
    /// oldest first. The cleaner keeps the last few seconds; what nobody
    /// takes in time is dropped ([`MicStatus::voice_dropped`]). One clone
    /// takes at a time: while another one takes, this takes nothing (and
    /// leaves it for the next call).
    pub fn take_voice(&self, out: &mut Vec<VoiceBlock>) {
        if let Some(mut voice) = self.shared().cleaned.voice.consumer() {
            while let Some(v) = voice.pop() {
                out.push(v.block());
            }
        }
    }

    /// How the echo canceller is doing: the echo it removes in dB, and
    /// whether it has learnt the echo's way (see [`crate::aec::EchoStatus`]).
    pub fn echo_status(&self) -> crate::aec::EchoStatus {
        let c = &self.shared().cleaned;
        crate::aec::EchoStatus {
            removed_db: f32::from_bits(c.echo_removed.load(Ordering::Relaxed)),
            filter_db: f32::from_bits(c.echo_filter.load(Ordering::Relaxed)),
            learnt: c.echo_learnt.load(Ordering::Relaxed),
            delay: Some(f32::from_bits(c.echo_delay.load(Ordering::Relaxed))).filter(|d| *d >= 0.0),
        }
    }

    /// Records what the echo canceller works with for `seconds` (of its
    /// own time) to `path`, a WAV file of five channels at its rate,
    /// 32-bit float (WAVE_FORMAT_EXTENSIBLE): the microphone as it hears it,
    /// what plays (the music going to the left and the right speaker), the
    /// voice the canceller gives, and that voice with its leftover
    /// suppressed (as the game hears it). For tuning it to a
    /// room. The cleaner queues its blocks; a thread of its own writes
    /// them. The handle says when it is done and how it went. It stops
    /// early, with what it has, when no block comes for a few seconds (no
    /// output tapped). One recording at a time.
    pub fn record_echo(&self, path: &Path, seconds: f64) -> Result<EchoRecording, String> {
        let shared = Arc::clone(&self.inner.shared);
        let record = shared
            .cleaned
            .record
            .consumer()
            .ok_or("a recording is under way")?;
        let path = path.to_owned();
        let thread = std::thread::Builder::new()
            .name("jodel-echo-record".into())
            .spawn(move || write_recording(&shared, record, &path, seconds).map(|()| path))
            .map_err(|e| e.to_string())?;
        Ok(EchoRecording {
            thread: Some(thread),
        })
    }

    /// Whether capture has run for at least `after` and every sample was
    /// exactly zero. A real microphone always has some noise, so this is
    /// a muted device or, on macOS, a process the system has not allowed
    /// to use the microphone (the HAL then delivers silence, no error).
    pub fn only_silence(&self, after: Duration) -> bool {
        let s = self.shared();
        let first = s.capture.first_at.load(Ordering::Relaxed);
        first != 0
            && !s.capture.signal.load(Ordering::Relaxed)
            && s.now_ns().saturating_sub(first - 1) >= after.as_nanos() as u64
    }

    /// How the path is doing.
    pub fn status(&self) -> MicStatus {
        let s = self.shared();
        let (c, b) = (&s.capture, &s.bridge);
        MicStatus {
            rate: c.rate.load(Ordering::Acquire),
            period_frames: c.period.load(Ordering::Relaxed),
            packet_frames: c.packet.load(Ordering::Relaxed),
            fill: f64::from_bits(b.fill.load(Ordering::Relaxed)),
            target: b.target.load(Ordering::Relaxed),
            adjust: f64::from_bits(b.adjust.load(Ordering::Relaxed)),
            active: c.ring.has_consumer(),
            underruns: b.underruns.load(Ordering::Relaxed),
            overruns: c.overruns.load(Ordering::Relaxed),
            resyncs: b.resyncs.load(Ordering::Relaxed),
            blocks_dropped: b.blocks_dropped.load(Ordering::Relaxed),
            voice_dropped: s.cleaned.voice_dropped.load(Ordering::Relaxed),
        }
    }

    /// The error that stopped capture, if one did.
    pub fn failure(&self) -> Option<Error> {
        self.shared().failure()
    }

    /// A tap that lines this input up with what an output plays, for the
    /// cleaner (see [`MicTap`]). It keeps the capture running while it
    /// lives.
    pub fn tap(&self) -> MicTap {
        MicTap::new(self.clone())
    }
}

impl std::fmt::Debug for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Input").field("info", self.info()).finish()
    }
}

/// The caller's end of an [`Input::feed`].
pub struct MicFeed {
    producer: Producer,
    /// The cleaner, run when the caller says ([`MicFeed::clean`]).
    cleaner: Cleaner,
}

impl MicFeed {
    /// Queues mono samples (full scale ±1) as one capture packet, at the
    /// time last set. Returns how many fitted (the ring holds a second).
    pub fn push(&mut self, samples: &[f32]) -> usize {
        let room = self.producer.free();
        let bytes: &[u8] = bytemuck_f32(samples);
        self.producer.push(bytes, Wire::F32, 1, samples.len());
        samples.len().min(room)
    }

    /// Sets the simulated clock that both the feed and the tap read
    /// (nothing reads it on [`Input::feed_in_real_time`]'s, whose clock is
    /// the system's).
    pub fn set_time(&self, t: Duration) {
        self.producer.shared().set_time(t);
    }

    /// Runs the cleaner over every block the tap has queued (a device's
    /// input has a thread of its own for it).
    pub fn clean(&mut self) {
        while self.cleaner.run(self.producer.shared()) {}
    }
}

/// The bytes of `f32` samples, as the wire reads them (little-endian
/// hosts; elsewhere each sample is converted).
fn bytemuck_f32(samples: &[f32]) -> &[u8] {
    #[cfg(target_endian = "little")]
    {
        // SAFETY: f32 has no padding and any byte is a valid u8.
        unsafe {
            std::slice::from_raw_parts(samples.as_ptr().cast(), std::mem::size_of_val(samples))
        }
    }
    #[cfg(not(target_endian = "little"))]
    {
        compile_error!("jodel-output assumes a little-endian host")
    }
}

#[cfg(test)]
mod tests {
    use super::tap::{Packet, Stamp};
    use super::*;
    use crate::aec::BLOCK;
    use crate::format::{SampleFormat, StreamFormat};
    use crate::tap::Tap;

    const PERIOD: usize = 240;

    /// A feed and the device buffers a tap sees, 5 ms each, side by side.
    struct Bench {
        input: Input,
        feed: MicFeed,
        /// Buffers so far.
        step: u64,
    }

    impl Bench {
        fn new() -> Bench {
            let (input, feed) = Input::feed(48_000, PERIOD as u32);
            input.set_echo_cancel(false);
            Bench {
                input,
                feed,
                step: 0,
            }
        }

        /// One capture packet, then one buffer of silence through `tap`.
        fn run(&mut self, tap: &mut MicTap) {
            self.feed.set_time(Duration::from_millis(5 * self.step));
            self.feed.push(&[0.25; PERIOD]);
            self.see(tap);
            self.feed.clean();
        }

        /// One buffer through `tap`, nothing captured.
        fn see(&mut self, tap: &mut MicTap) {
            let format = StreamFormat::new(48_000, 2, SampleFormat::S24In32);
            tap.begin(PERIOD, self.step * PERIOD as u64, PERIOD, format);
            tap.see(&[0; 2 * PERIOD], format);
            self.step += 1;
        }

        fn voice(&self) -> Vec<VoiceBlock> {
            let mut voice = Vec::new();
            self.input.take_voice(&mut voice);
            voice
        }
    }

    /// A second tap of an input queues nothing while the first lives, and
    /// takes nothing of the capture ring; once the first is dropped it
    /// takes over, its first block following none. A tap that finds the
    /// cleaner's queue taken (half its ends free) gives the ring back
    /// untouched.
    #[test]
    fn a_second_tap_takes_over_when_the_first_is_dropped() {
        let mut bench = Bench::new();
        let mut first = bench.input.tap();
        for _ in 0..60 {
            bench.run(&mut first);
        }
        assert!(bench.input.status().active);
        let given = bench.voice();
        assert!(!given.is_empty() && !given[0].continues);
        assert!(given[1..].iter().all(|v| v.continues));

        let mut second = bench.input.tap();
        let room = bench.feed.producer.free();
        for _ in 0..30 {
            bench.see(&mut second);
        }
        bench.feed.clean();
        assert_eq!(bench.feed.producer.free(), room, "the ring is the first's");
        assert!(bench.voice().is_empty(), "the second queued nothing");

        drop(first);
        let blocks = bench.input.shared().bridge.blocks.producer().unwrap();
        bench.run(&mut second);
        assert!(!bench.input.status().active, "the ring was given back");
        assert_eq!(bench.feed.producer.free(), room - PERIOD);
        drop(blocks);

        for _ in 0..60 {
            bench.run(&mut second);
        }
        let given = bench.voice();
        assert!(!given.is_empty() && !given[0].continues);
        assert!(given[1..].iter().all(|v| v.continues));
    }

    /// Two clones taking the voice at once while the cleaner gives it:
    /// every block is taken once, whole, by one of them, and each one's
    /// come in order.
    #[test]
    fn two_clones_take_the_voice_at_once() {
        const BLOCKS: u64 = 3000;
        let (input, mut feed) = Input::feed(48_000, 240);
        input.set_echo_cancel(false);
        let mut tap_end = input.shared().bridge.blocks.producer().unwrap();
        let done = Arc::new(AtomicBool::new(false));
        let takers: Vec<_> = (0..2)
            .map(|_| {
                let input = input.clone();
                let done = Arc::clone(&done);
                std::thread::spawn(move || {
                    let mut taken = Vec::new();
                    while !done.load(Ordering::Acquire) {
                        input.take_voice(&mut taken);
                        std::thread::yield_now();
                    }
                    taken
                })
            })
            .collect();
        for k in 0..BLOCKS {
            // A click at a place of its own in each block.
            let mut mic = [0.0; BLOCK];
            mic[k as usize % BLOCK] = 0.5;
            let packet = Packet {
                stamp: Stamp {
                    stream_frame: k * BLOCK as u64,
                    rate: 48_000,
                    continues: k > 0,
                },
                mic,
                left: [0.0; BLOCK],
                right: [0.0; BLOCK],
            };
            while !tap_end.push(packet) {
                feed.clean();
            }
            if k % 8 == 0 {
                feed.clean();
            }
        }
        feed.clean();
        done.store(true, Ordering::Release);
        let mut all = Vec::new();
        for taker in takers {
            let taken = taker.join().unwrap();
            assert!(taken
                .windows(2)
                .all(|w| w[0].stream_frame < w[1].stream_frame));
            all.extend(taken);
        }
        // What neither took before they stopped.
        input.take_voice(&mut all);
        all.sort_by_key(|v| v.stream_frame);
        // The newest block is still with the suppressor.
        assert_eq!(all.len() as u64, BLOCKS - 1);
        for (k, v) in all.iter().enumerate() {
            assert_eq!(
                (v.stream_frame, v.rate, v.continues, v.samples.len()),
                ((k * BLOCK) as u64, 48_000, k > 0, BLOCK)
            );
            for (i, s) in v.samples.iter().enumerate() {
                let want = if i == k % BLOCK { 0.5 } else { 0.0 };
                assert!((s - want).abs() < 1e-5, "block {k} sample {i}: {s}");
            }
        }
        assert_eq!(input.status().voice_dropped, 0);
    }
}
