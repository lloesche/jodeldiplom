//! The microphone path without hardware: taps on a `FileOutput`, and a
//! `MicFeed` standing in for a capture device on a simulated clock.

use jodel_output::pack::{pack, unpack};
use jodel_output::{
    Capabilities, FileOutput, Input, MemorySink, OpenRequest, Output, PackMode, SampleFormat,
    SimClock, StreamFormat, Tap, VoiceBlock,
};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

fn ramp(frames: usize, channels: usize) -> Vec<i32> {
    (0..frames * channels)
        .map(|i| ((i as i64 * 40_009) % (1 << 24) - (1 << 23)) as i32)
        .collect()
}

fn packed(samples: &[i32], format: SampleFormat) -> Vec<u8> {
    let mut out = vec![0; samples.len() * format.bytes()];
    pack(samples, format, PackMode::Saturate, &mut out).unwrap();
    out
}

/// A manual-clock output: ring 500 ms, prefill 100 ms, period 10 ms.
fn manual(rate: u32, channels: u16, format: SampleFormat) -> (FileOutput, MemorySink) {
    let request = OpenRequest::new(rate, channels).formats(&[format]);
    let caps: Capabilities = [StreamFormat::new(rate, channels, format)]
        .into_iter()
        .collect();
    let sink = MemorySink::default();
    let out = FileOutput::open(sink.clone(), &caps, &request, SimClock::Manual).unwrap();
    (out, sink)
}

/// Sees every buffer, and says when it is called, what it saw, and when it
/// is dropped.
struct Probe {
    calls: Arc<AtomicU64>,
    /// The sum of every sample seen.
    sum: Arc<AtomicU64>,
    dropped: Arc<AtomicBool>,
}

impl Tap for Probe {
    fn see(&mut self, samples: &[i32], _format: StreamFormat) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let s: i64 = samples.iter().map(|&v| i64::from(v)).sum();
        self.sum.fetch_add(s as u64, Ordering::Relaxed);
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

fn probe() -> (Box<Probe>, Arc<AtomicU64>, Arc<AtomicBool>) {
    let calls = Arc::new(AtomicU64::new(0));
    let dropped = Arc::new(AtomicBool::new(false));
    (
        Box::new(Probe {
            calls: calls.clone(),
            sum: Arc::new(AtomicU64::new(0)),
            dropped: dropped.clone(),
        }),
        calls,
        dropped,
    )
}

/// Plays `input` through a manual output in 10 ms periods, calling
/// `between` before each period; returns the bytes played.
fn play(
    format: StreamFormat,
    input: &[i32],
    mut between: impl FnMut(&mut FileOutput, usize),
) -> Vec<u8> {
    play_heard(format, input, |out, k, _| between(out, k))
}

/// [`play`], `between` also given what has played so far.
fn play_heard(
    format: StreamFormat,
    input: &[i32],
    mut between: impl FnMut(&mut FileOutput, usize, &MemorySink),
) -> Vec<u8> {
    let (mut out, sink) = manual(format.rate, format.channels, format.sample);
    let ch = usize::from(format.channels);
    let period = (format.rate / 100) as usize;
    let frames = input.len() / ch;
    // The 100 ms prefill, then one period written per period played.
    let mut written = (10 * period).min(frames);
    out.write(&input[..written * ch], None).unwrap();
    let mut k = 0;
    while (out.position().unwrap() as usize) < frames {
        between(&mut out, k, &sink);
        k += 1;
        let n = period.min(frames - written);
        out.write(&input[written * ch..(written + n) * ch], None)
            .unwrap();
        written += n;
        out.advance(period.min(frames - out.position().unwrap() as usize) as u64)
            .unwrap();
    }
    out.close().unwrap();
    sink.bytes()
}

/// Two 5 ms packets of `tone(i)` per 10 ms period `k`, at the times they
/// would arrive; `sent` the samples sent so far.
fn feed_period(
    feed: &mut jodel_output::MicFeed,
    k: usize,
    sent: &mut usize,
    tone: impl Fn(usize) -> f32,
) {
    for p in 0..2 {
        feed.set_time(Duration::from_micros(10_000 * k as u64 + 5_000 * p));
        let packet: Vec<f32> = (*sent..*sent + 240).map(&tone).collect();
        *sent += feed.push(&packet);
    }
    feed.set_time(Duration::from_micros(10_000 * (k as u64 + 1)));
}

/// The rule of the microphone: a tap only sees. With a tap, a microphone
/// tapped and singing, or none, the device gets exactly the bytes written,
/// in every format.
#[test]
fn a_tap_never_changes_what_plays() {
    for sample in SampleFormat::ALL {
        let format = StreamFormat::new(48_000, 2, sample);
        let input: Vec<i32> = ramp(48_000, 2)
            .into_iter()
            .map(|v| {
                if sample == SampleFormat::S16 {
                    v & !0xFF
                } else {
                    v
                }
            })
            .collect();
        let want = packed(&input, sample);
        assert_eq!(play(format, &input, |_, _| {}), want, "{sample}: none set");
        let (tap, calls, _) = probe();
        let bytes = play(format, &input, |out, k| {
            if k == 10 {
                out.set_tap(Some(tap_once(&tap))).unwrap();
            }
            if k == 50 {
                out.set_tap(None).unwrap();
            }
        });
        assert_eq!(bytes, want, "{sample}: a tap set and taken away");
        assert!(calls.load(Ordering::Relaxed) > 0, "the tap saw the buffers");
        // A microphone singing loudly into it, the echo canceller on and
        // off: not a byte changes.
        for cancel in [true, false] {
            let (input_mic, mut feed) = Input::feed(48_000, 240);
            input_mic.set_echo_cancel(cancel);
            let mut sent = 0;
            let bytes = play(format, &input, |out, k| {
                if k == 0 {
                    out.set_tap(Some(Box::new(input_mic.tap()))).unwrap();
                }
                feed_period(&mut feed, k, &mut sent, |i| (i as f32 * 0.13).sin());
                feed.clean();
            });
            assert_eq!(
                bytes, want,
                "{sample}: a microphone tapped, cancel {cancel}"
            );
            let mut voice = Vec::new();
            input_mic.take_voice(&mut voice);
            assert!(!voice.is_empty(), "{sample}: the voice was heard");
        }
    }
}

/// A probe the test keeps the counters of: a fresh box sharing them.
fn tap_once(p: &Probe) -> Box<dyn Tap> {
    Box::new(Probe {
        calls: p.calls.clone(),
        sum: p.sum.clone(),
        dropped: p.dropped.clone(),
    })
}

/// The voice of a `MicFeed` (a 1 kHz tone at half of full scale) tapped
/// on an output at 48 kHz, the canceller off: what the cleaner gives is
/// the tone as it came in, in whole blocks that follow each other, each
/// where it was in the stream.
#[test]
fn a_mic_tap_gives_the_voice_where_it_was_sung() {
    let rate = 48_000;
    let format = StreamFormat::new(rate, 2, SampleFormat::S24In32);
    let (input, mut feed) = Input::feed(rate, 240);
    input.set_echo_cancel(false);
    let frames = 3 * rate as usize;
    let engine = vec![-1_000_000; 2 * frames];
    let mut sent = 0usize;
    let mut voice: Vec<VoiceBlock> = Vec::new();
    play(format, &engine, |out, k| {
        if k == 0 {
            out.set_tap(Some(Box::new(input.tap()))).unwrap();
        }
        feed_period(&mut feed, k, &mut sent, |i| {
            0.5 * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48_000.0).sin()
        });
        feed.clean();
        input.take_voice(&mut voice);
    });
    let status = input.status();
    assert_eq!(
        (status.underruns, status.overruns, status.resyncs),
        (0, 0, 0)
    );
    assert_eq!((status.blocks_dropped, status.voice_dropped), (0, 0));
    assert!(
        voice.len() > 2 * rate as usize / 128,
        "{} blocks",
        voice.len()
    );
    // Each block follows the one before, 128 frames on.
    for w in voice.windows(2) {
        assert!(w[1].continues, "{:?}", w[1].stream_frame);
        assert_eq!(w[1].stream_frame, w[0].stream_frame + 128);
        assert_eq!(w[1].rate, rate);
    }
    // The last two seconds: the tone at half of full scale, 1 kHz.
    let samples: Vec<f32> = voice.iter().flat_map(|b| b.samples.clone()).collect();
    let tail = &samples[samples.len() - 2 * rate as usize..];
    let peak = tail.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!((peak - 0.5).abs() < 0.005, "peak {peak}");
    let crossings = tail
        .windows(2)
        .filter(|w| w[0] < 0.0 && w[1] >= 0.0)
        .count();
    assert!(
        (1999..=2001).contains(&crossings),
        "{crossings} cycles in 2 s"
    );
    // The meter hears the singer isolated (jodel_output::isolate), which
    // keeps a steady tone like a held note.
    let meter = input.voice_peak().expect("the meter has the voice");
    assert!((meter - 0.5).abs() < 0.05, "meter {meter}");
}

/// Voice not taken for longer than it is kept: the blocks that do not fit
/// are lost, and the first kept after them says it does not follow the
/// one before (a reader must see the gap, even when the block lost
/// started a run).
#[test]
fn after_voice_lost_to_a_full_queue_the_next_block_starts_anew() {
    let rate = 48_000;
    let format = StreamFormat::new(rate, 2, SampleFormat::S24In32);
    let (input, mut feed) = Input::feed(rate, 240);
    input.set_echo_cancel(false);
    // 14 s, the voice taken only from 12 s on (the queue keeps about 11).
    let frames = 14 * rate as usize;
    let engine = vec![-1_000_000; 2 * frames];
    let mut sent = 0usize;
    let mut voice: Vec<VoiceBlock> = Vec::new();
    play(format, &engine, |out, k| {
        if k == 0 {
            out.set_tap(Some(Box::new(input.tap()))).unwrap();
        }
        feed_period(&mut feed, k, &mut sent, |i| 0.1 * (i as f32 * 0.05).sin());
        feed.clean();
        if k >= 1_200 {
            input.take_voice(&mut voice);
        }
    });
    let status = input.status();
    assert!(status.voice_dropped > 0);
    let gap = voice
        .windows(2)
        .position(|w| w[1].stream_frame != w[0].stream_frame + 128)
        .expect("a gap")
        + 1;
    assert!(!voice[gap].continues, "the block after the gap continues");
    assert_eq!(voice[1..].iter().filter(|b| !b.continues).count(), 1);
}

#[test]
fn a_running_device_takes_a_new_tap_without_a_lock() {
    let format = StreamFormat::new(48_000, 2, SampleFormat::S24In32);
    let request = OpenRequest::new(48_000, 2)
        .formats(&[format.sample])
        .prefill(Duration::from_millis(10));
    let caps: Capabilities = [format].into_iter().collect();
    let mut out = FileOutput::open(std::io::sink(), &caps, &request, SimClock::Realtime).unwrap();
    out.write(&vec![0; 2 * 4_800], None).unwrap();
    // The device thread runs when it is let: what it must do is waited for
    // (a guard, however busy the machine).
    let until = |what: &str, done: &dyn Fn() -> bool| {
        let guard = std::time::Instant::now() + Duration::from_secs(10);
        while !done() {
            assert!(std::time::Instant::now() < guard, "{what}");
            std::thread::yield_now();
        }
    };
    let (a, a_calls, a_dropped) = probe();
    out.set_tap(Some(a)).unwrap();
    until("the device runs the tap", &|| {
        a_calls.load(Ordering::Relaxed) > 0
    });
    let (b, b_calls, b_dropped) = probe();
    out.set_tap(Some(b)).unwrap();
    // The device took the new one; the old one was dropped here, on return.
    assert!(a_dropped.load(Ordering::SeqCst));
    let a_after = a_calls.load(Ordering::Relaxed);
    until("the device runs the new tap", &|| {
        b_calls.load(Ordering::Relaxed) > 0
    });
    assert_eq!(a_calls.load(Ordering::Relaxed), a_after);
    // Paused, the swap happens on this thread.
    out.pause().unwrap();
    out.set_tap(None).unwrap();
    assert!(b_dropped.load(Ordering::SeqCst));
    out.close().unwrap();
}

/// Ten simulated minutes of a microphone whose clock is `offset` off the
/// DAC's, into an output at `out_rate`: checks the loop holds the fill at
/// its target and settles on the offset.
fn hold_the_fill(out_rate: u32, mic_rate: u32, offset: f64, seconds: u64) {
    let format = StreamFormat::new(out_rate, 2, SampleFormat::S24In32);
    let request = OpenRequest::new(out_rate, 2).formats(&[format.sample]);
    let caps: Capabilities = [format].into_iter().collect();
    let mut out = FileOutput::open(std::io::sink(), &caps, &request, SimClock::Manual).unwrap();
    let packet = mic_rate as usize / 200; // 5 ms
    let (input, mut feed) = Input::feed(mic_rate, packet as u32);
    let period = out_rate as u64 / 100; // 10 ms
    let silence = vec![0i32; 2 * period as usize];
    for _ in 0..10 {
        out.write(&silence, None).unwrap();
    }
    out.set_tap(Some(Box::new(input.tap()))).unwrap();
    // Event times in ns: the next output period and the next mic packet
    // (on the mic's own clock, `offset` fast).
    let packet_ns = packet as f64 * 1e9 / (f64::from(mic_rate) * (1.0 + offset));
    let mut next_mic = 1u64;
    let mut next_out = 0u64;
    let mut sent = 0usize;
    let mut worst_fill = 0.0f64;
    let mut worst_adjust = 0.0f64;
    let end = seconds * 1_000_000_000;
    let mut tone = vec![0f32; packet];
    let mut checked_at = 0;
    while next_out < end {
        let mic_t = (next_mic as f64 * packet_ns) as u64;
        if mic_t <= next_out {
            feed.set_time(Duration::from_nanos(mic_t));
            for (k, v) in tone.iter_mut().enumerate() {
                *v = 0.3 * ((sent + k) as f32 * 0.05).sin();
            }
            sent += feed.push(&tone);
            next_mic += 1;
            continue;
        }
        feed.set_time(Duration::from_nanos(next_out));
        out.write(&silence, None).unwrap();
        out.advance(period).unwrap();
        next_out += 10_000_000;
        let s = input.status();
        let t = next_out / 1_000_000_000;
        if t >= 60 && t != checked_at {
            checked_at = t;
            worst_fill = worst_fill.max((s.fill - f64::from(s.target)).abs());
            worst_adjust = worst_adjust.max((s.adjust - offset).abs());
        }
    }
    let s = input.status();
    assert_eq!(s.target as usize, 2 * packet);
    assert_eq!((s.underruns, s.overruns, s.resyncs), (0, 0, 0), "{s:?}");
    assert_eq!(out.stats().underruns, 0);
    // After the first minute: within 0.05 ms of the target, the ratio
    // within 5 ppm of the clocks' offset.
    let tolerance = f64::from(mic_rate) * 0.000_05;
    assert!(
        worst_fill < tolerance,
        "fill off by {worst_fill} frames: {s:?}"
    );
    assert!(worst_adjust < 5e-6, "adjust off by {worst_adjust}: {s:?}");
    out.close().unwrap();
}

#[test]
fn the_loop_holds_a_fast_mic_over_ten_minutes() {
    hold_the_fill(44_100, 48_000, 0.001, 600);
}

#[test]
fn the_loop_holds_a_slow_mic_at_352_8_khz() {
    hold_the_fill(352_800, 48_000, -0.001, 90);
}

#[test]
fn the_voice_comes_back_after_a_capture_gap_and_a_device_stall() {
    let rate = 48_000;
    let format = StreamFormat::new(rate, 2, SampleFormat::S24In32);
    let request = OpenRequest::new(rate, 2).formats(&[format.sample]);
    let caps: Capabilities = [format].into_iter().collect();
    let mut out = FileOutput::open(std::io::sink(), &caps, &request, SimClock::Manual).unwrap();
    let (input, mut feed) = Input::feed(rate, 240);
    let silence = vec![0i32; 2 * 480];
    for _ in 0..10 {
        out.write(&silence, None).unwrap();
    }
    out.set_tap(Some(Box::new(input.tap()))).unwrap();
    let packet = vec![0.1f32; 240];
    let mut t = 0u64; // ms
                      // Each 10 ms: two packets (unless the capture is stalled), then one
                      // output period (unless the device is); `burst` packets more at once
                      // first.
    let mut run = |out: &mut FileOutput, ms: u64, capture: bool, device: bool, burst: usize| {
        for _ in 0..burst {
            feed.push(&packet);
        }
        for _ in 0..ms / 10 {
            if capture {
                for p in 0..2 {
                    feed.set_time(Duration::from_millis(t + 5 * p));
                    feed.push(&packet);
                }
            }
            t += 10;
            feed.set_time(Duration::from_millis(t));
            if device {
                out.write(&silence, None).unwrap();
                out.advance(480).unwrap();
            }
        }
    };
    run(&mut out, 5_000, true, true, 0);
    let steady = input.status();
    assert_eq!((steady.underruns, steady.resyncs), (0, 0));
    // The capture delivers nothing for 100 ms: the voice drops out once.
    run(&mut out, 100, false, true, 0);
    run(&mut out, 5_000, true, true, 0);
    let s = input.status();
    assert_eq!((s.underruns, s.resyncs), (1, 0), "{s:?}");
    assert!((s.fill - f64::from(s.target)).abs() < 5.0, "{s:?}");
    // The device takes nothing for 200 ms: the backlog is skipped.
    run(&mut out, 200, true, false, 0);
    run(&mut out, 5_000, true, true, 0);
    let s = input.status();
    assert_eq!((s.underruns, s.resyncs), (1, 1), "{s:?}");
    assert!((s.fill - f64::from(s.target)).abs() < 5.0, "{s:?}");
    assert!(s.adjust.abs() < 1e-4, "{s:?}");
    // The capture delivers 30 ms at once (as devices do when they start):
    // skipped within half a second, not drained for seconds.
    run(&mut out, 500, true, true, 6);
    let s = input.status();
    assert_eq!((s.underruns, s.resyncs), (1, 2), "{s:?}");
    run(&mut out, 1_000, true, true, 0);
    let s = input.status();
    assert!((s.fill - f64::from(s.target)).abs() < 5.0, "{s:?}");
    out.close().unwrap();
}

/// Exact zeros for a while are reported as only silence (a muted device,
/// or on macOS a process not allowed to use the microphone); any noise at
/// all is not.
#[test]
fn a_microphone_giving_only_zeros_is_told_apart() {
    let wait = Duration::from_millis(1_500);
    let (input, mut feed) = Input::feed(48_000, 240);
    assert!(!input.only_silence(wait), "nothing came in yet");
    for k in 0..400u64 {
        feed.set_time(Duration::from_millis(5 * k));
        feed.push(&[0.0; 240]);
    }
    assert!(input.only_silence(wait));
    let (input, mut feed) = Input::feed(48_000, 240);
    for k in 0..400u64 {
        feed.set_time(Duration::from_millis(5 * k));
        let mut packet = [0.0f32; 240];
        packet[7] = 1e-5;
        feed.push(&packet);
    }
    assert!(!input.only_silence(wait));
    // Zeros for less than the wait are not yet told.
    let (input, mut feed) = Input::feed(48_000, 240);
    for k in 0..100u64 {
        feed.set_time(Duration::from_millis(5 * k));
        feed.push(&[0.0; 240]);
    }
    assert!(!input.only_silence(wait));
}

/// The echo canceller behind the tap: the engine plays music, the
/// microphone hears it 30 ms later at half its level (and, from 6 s on, a
/// voice over it). The voice the cleaner gives: once learnt, the music's
/// echo is far down in it, and the voice sung over it comes through whole.
/// What plays is never touched.
#[test]
fn the_cleaner_takes_the_echo_of_what_plays_out_of_the_voice() {
    let rate = 48_000u32;
    let format = StreamFormat::new(rate, 2, SampleFormat::S24In32);
    let frames = 10 * rate as usize;
    // Music-like: coloured noise.
    let mut seed = 5u32;
    let mut lp = 0.0f32;
    let music: Vec<f32> = (0..frames)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let n = (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            lp += (n - lp) * 0.25;
            0.5 * lp
        })
        .collect();
    let full = 8_388_608.0f32;
    let engine: Vec<i32> = music
        .iter()
        .flat_map(|&m| {
            let v = (m * full) as i32;
            [v, v]
        })
        .collect();
    let voice_from = 6 * rate as usize;
    let voice = |i: usize| -> f32 {
        if i < voice_from {
            0.0
        } else {
            0.2 * (2.0 * std::f32::consts::PI * 330.0 * i as f32 / rate as f32).sin()
        }
    };
    // The echo: what played, 30 ms later, at half its level. The
    // microphone's sample i is heard with output frame i at the same moment.
    let delay = (rate as usize * 30) / 1000;
    let (input, mut feed) = Input::feed(rate, 240);
    assert!(input.echo_cancel(), "on as it opens");
    // A recording of what the canceller works with, from the start.
    let dir = std::env::temp_dir().join(format!("jodel-echo-record-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let recorded = dir.join("echo.wav");
    let mut recording = input.record_echo(&recorded, 0.5).unwrap();
    assert!(input.record_echo(&recorded, 0.5).is_err(), "one at a time");
    let mut sent = 0usize;
    // What has played (its first channel).
    let mut heard: Vec<f32> = Vec::new();
    let frame_bytes = 2 * format.sample.bytes();
    let mut blocks: Vec<VoiceBlock> = Vec::new();
    let bytes = play_heard(format, &engine, |out, k, sink| {
        if k == 0 {
            out.set_tap(Some(Box::new(input.tap()))).unwrap();
        }
        let so_far = sink.bytes();
        let from = heard.len() * frame_bytes;
        let to = so_far.len() / frame_bytes * frame_bytes;
        let new = unpack(&so_far[from..to], format.sample);
        heard.extend(new.chunks_exact(2).map(|f| f[0] as f32 / full));
        let heard = &heard;
        feed_period(&mut feed, k, &mut sent, |i| {
            let echo = if i >= delay && i - delay < heard.len() {
                0.5 * heard[i - delay]
            } else if i >= delay && i - delay < frames {
                0.5 * music[i - delay]
            } else {
                0.0
            };
            echo + voice(i)
        });
        feed.clean();
        input.take_voice(&mut blocks);
    });
    assert_eq!(
        bytes,
        packed(&engine, format.sample),
        "what plays is untouched"
    );
    // The voice cleaned, laid out where it was in the stream.
    let mut cleaned = vec![0.0f32; frames];
    for b in &blocks {
        for (i, &v) in b.samples.iter().enumerate() {
            if let Some(c) = cleaned.get_mut(b.stream_frame as usize + i) {
                *c = v;
            }
        }
    }
    let power = |s: &[f32]| s.iter().map(|v| v * v).sum::<f32>() / s.len().max(1) as f32;
    let db = |p: f32| 10.0 * p.max(1e-20).log10();
    let s = rate as usize;
    // Music alone, learnt: the echo's power was a quarter of the music's.
    let echo = power(&music[3 * s..6 * s]) / 4.0;
    let left = power(&cleaned[3 * s..6 * s]);
    let removed = db(echo) - db(left);
    // Singing over it: what comes out is about the voice's power.
    let sung = 0.2f32 * 0.2 / 2.0;
    let kept = db(power(&cleaned[7 * s..10 * s])) - db(sung);
    eprintln!("behind the tap: echo removed {removed:.1} dB, the voice at {kept:.1} dB");
    assert!(removed > 25.0, "echo removed {removed:.1} dB");
    assert!(kept.abs() < 1.5, "the voice at {kept:.1} dB");
    let status = input.echo_status();
    assert!(status.learnt && status.removed_db > 10.0, "{status:?}");
    assert!(input.voice_peak().is_some());
    // The recording: half a second at the canceller's rate, five float
    // channels; the second and third are what played (the music, as the
    // engine sent it, left and right), the first the microphone (its echo,
    // at half the level).
    let done = (0..500).find_map(|_| {
        std::thread::sleep(Duration::from_millis(10));
        recording.done()
    });
    let path = done.expect("the recording ends").unwrap();
    let wav = std::fs::read(&path).unwrap();
    assert_eq!(&wav[..4], b"RIFF");
    let read16 = |at: usize| u16::from_le_bytes([wav[at], wav[at + 1]]);
    let read32 = |at: usize| u32::from_le_bytes([wav[at], wav[at + 1], wav[at + 2], wav[at + 3]]);
    // WAVE_FORMAT_EXTENSIBLE, float (its sub-format), the samples after
    // the `fact` chunk.
    assert_eq!((read16(20), read16(22), read32(24)), (0xFFFE, 5, rate));
    assert_eq!(read16(44), 3);
    assert_eq!((&wav[60..64], &wav[72..76]), (&b"fact"[..], &b"data"[..]));
    let samples: Vec<f32> = wav[80..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let frames_recorded = samples.len() / 5;
    assert!(
        frames_recorded >= rate as usize / 2,
        "{frames_recorded} frames"
    );
    let channel = |c: usize| -> Vec<f32> { samples.iter().skip(c).step_by(5).copied().collect() };
    let (mic, played) = (channel(0), channel(1));
    assert_eq!(played, channel(2), "left and right of a mono output");
    let ratio = db(power(&played)) - db(power(&music[..frames_recorded]));
    assert!(
        ratio.abs() < 1.0,
        "what played, {ratio:.1} dB off the music"
    );
    let echo_level = db(power(&mic)) - db(power(&played));
    assert!(
        (echo_level + 6.0).abs() < 1.5,
        "the microphone at {echo_level:.1} dB"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The echo canceller learns the room once: song after song on the same
/// device (each its own output and tap, at the one rate), what it has
/// learnt carries over, and the next song's echo is taken out from its
/// first second.
#[test]
fn what_the_echo_canceller_learnt_carries_over_to_the_next_output() {
    let rate = 48_000u32;
    let format = StreamFormat::new(rate, 2, SampleFormat::S24In32);
    let s = rate as usize;
    let music = |seed: u32, frames: usize| -> Vec<f32> {
        let (mut seed, mut lp) = (seed, 0.0f32);
        (0..frames)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let n = (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
                lp += (n - lp) * 0.25;
                0.5 * lp
            })
            .collect()
    };
    let full = 8_388_608.0f32;
    let delay = (rate as usize * 30) / 1000;
    let (input, mut feed) = Input::feed(rate, 240);
    let (mut sent, mut periods) = (0usize, 0usize);
    let frame_bytes = 2 * format.sample.bytes();
    // Two songs, one output each; the microphone runs on through both,
    // hearing each 30 ms late at half its level.
    let songs = [music(5, 6 * s), music(9, 3 * s)];
    let mut second: Vec<VoiceBlock> = Vec::new();
    let mut learnt_as_the_second_began = None;
    for (n, song) in songs.iter().enumerate() {
        let engine: Vec<i32> = song
            .iter()
            .flat_map(|&m| {
                let v = (m * full) as i32;
                [v, v]
            })
            .collect();
        let first_sent = sent;
        let mut heard: Vec<f32> = Vec::new();
        let mut blocks = Vec::new();
        play_heard(format, &engine, |out, k, sink| {
            if k == 0 {
                out.set_tap(Some(Box::new(input.tap()))).unwrap();
            }
            let so_far = sink.bytes();
            let from = heard.len() * frame_bytes;
            let to = so_far.len() / frame_bytes * frame_bytes;
            let new = unpack(&so_far[from..to], format.sample);
            heard.extend(new.chunks_exact(2).map(|f| f[0] as f32 / full));
            let heard = &heard;
            feed_period(&mut feed, periods, &mut sent, |i| {
                let i = i - first_sent;
                if i >= delay && i - delay < heard.len() {
                    0.5 * heard[i - delay]
                } else {
                    0.0
                }
            });
            periods += 1;
            feed.clean();
            input.take_voice(&mut blocks);
            if n == 1 && k == 20 {
                learnt_as_the_second_began = Some(input.echo_status().learnt);
            }
        });
        if n == 1 {
            second = blocks;
        }
    }
    assert_eq!(learnt_as_the_second_began, Some(true), "still learnt");
    // The second song's first second, cleaned: its echo far down.
    let mut cleaned = vec![0.0f32; s];
    for b in &second {
        for (i, &v) in b.samples.iter().enumerate() {
            if let Some(c) = cleaned.get_mut(b.stream_frame as usize + i) {
                *c = v;
            }
        }
    }
    let power = |x: &[f32]| x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32;
    let db = |p: f32| 10.0 * p.max(1e-20).log10();
    let echo = power(&songs[1][s / 4..s]) / 4.0;
    let removed = db(echo) - db(power(&cleaned[s / 4..s]));
    assert!(
        removed > 20.0,
        "the second song's echo removed {removed:.1} dB"
    );
}
