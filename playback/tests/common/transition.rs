//! Test support for transitions: the quarter separator, one transition
//! test at a time, tone and click files, the output let through a step at a
//! time, and the plan's lanes and beats.

use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use jodel_playback::stems::{Separator, STEMS};
use jodel_playback::{Envelope, Event, PlayerOptions, Settings, TrackStart, TransitionState};

use super::*;

pub const RATE: u32 = 44_100;
pub const FULL: f64 = 8_388_608.0;

/// Each stem a quarter of the mix.
pub struct Quarter;

impl Separator for Quarter {
    fn name(&self) -> String {
        "quarter".into()
    }

    fn masks(&self, _frames: usize, stft: &[f32]) -> Result<Vec<f32>, String> {
        Ok((0..4 * stft.len())
            .map(|i| if i % 2 == 0 { 0.25 } else { 0.0 })
            .collect())
    }
}

pub static QUARTER: Quarter = Quarter;

/// One test at a time: each runs two tracks' stems at several times real
/// time, and side by side (debug-built above all) they leave each other
/// too little time, so the engine's real-time fallbacks set in (a next
/// track that falls behind is left out of the mix, a route switch that
/// does not catch up is not made).
pub static SERIAL: Mutex<()> = Mutex::new(());

pub fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn quarter() -> Option<&'static dyn Separator> {
    Some(&QUARTER)
}

pub fn options() -> PlayerOptions {
    PlayerOptions {
        settings: Settings {
            resample: false,
            ..Settings::default()
        },
        position_interval: Duration::from_millis(5),
        separator: Some(quarter),
        // The gated output runs faster than real time: what the window
        // needs is waited for, however slow the machine.
        offline: true,
        ..PlayerOptions::default()
    }
}

/// A stereo WAV of `seconds` from a sample function of time.
pub fn wav(
    dir: &Path,
    name: &str,
    seconds: f64,
    f: impl Fn(f64, usize) -> f64,
) -> (PathBuf, Vec<i32>) {
    let path = dir.join(name);
    let frames = (seconds * f64::from(RATE)) as usize;
    let samples: Vec<i32> = (0..frames)
        .flat_map(|n| {
            let t = n as f64 / f64::from(RATE);
            [0, 1].map(|c| (f(t, c) * FULL).round() as i32)
        })
        .collect();
    write_wav(&path, RATE, 2, &samples);
    (path, samples)
}

/// Clicks every `period` seconds from `first`: short bursts at `hz`.
pub fn clicks(period: f64, first: f64, hz: f64) -> impl Fn(f64, usize) -> f64 {
    move |t, _| {
        if t < first {
            return 0.0;
        }
        let since = (t - first) % period;
        if since > 0.004 {
            return 0.0;
        }
        0.8 * (2.0 * std::f64::consts::PI * hz * since).sin() * (-since / 0.001).exp()
    }
}

/// The 24-bit samples of a stream the device took as `S24In32Low`.
pub fn samples(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|b| (i32::from_le_bytes([b[0], b[1], b[2], b[3]]) << 8) >> 8)
        .collect()
}

pub fn starts(events: &[Event]) -> Vec<&TrackStart> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::TrackStarted(s) => Some(&**s),
            _ => None,
        })
        .collect()
}

pub fn transitions(events: &[Event]) -> Vec<&TransitionState> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::Transition(s) => Some(s),
            _ => None,
        })
        .collect()
}

/// Lets the output through a step at a time until the queue ends: the next
/// step once the engine has written the last, after a pause with the gate
/// shut.
///
/// The gated output reports what it took as played at once, so it never
/// holds half its buffer, which is how the engine knows it has time to
/// spare: a route switch (and the next track) get their turns only after a
/// write the gate refused, which takes the engine's write wait (10 ms). The
/// pause is well over that wait, also where timers tick about every 16 ms
/// (Windows), so each step leaves the switch time to decode however long
/// the engine took to write it, and the pace follows the build. At a fixed
/// pace, a debug build of the oldest supported compiler spent nearly all
/// its time writing the playing track's stems, hardly a write was refused,
/// and the switch back to the file never caught up with the frame being
/// written before the track ended.
pub fn step_to_end(gate: &Gate, events: &Receiver<Event>) -> Vec<Event> {
    step_until(gate, events, |e| matches!(e, Event::QueueEnded))
}

/// [`step_to_end`] until an event `done` says so (or the queue ends).
pub fn step_until(
    gate: &Gate,
    events: &Receiver<Event>,
    done: impl Fn(&Event) -> bool,
) -> Vec<Event> {
    const STEP: u64 = 8_192;
    const PAUSE: Duration = Duration::from_millis(30);
    let deadline = Instant::now() + TIMEOUT * 4;
    let mut seen = Vec::new();
    // Takes the events so far; whether the queue ended.
    let ended = |seen: &mut Vec<Event>| {
        for e in events.try_iter() {
            let end = matches!(e, Event::QueueEnded) || done(&e);
            seen.push(e);
            if end {
                return true;
            }
        }
        assert!(Instant::now() < deadline, "the queue did not end");
        false
    };
    loop {
        let allowed = gate.written() + STEP;
        gate.allow(allowed);
        while !gate.written_within(allowed, Duration::from_millis(5)) {
            if ended(&mut seen) {
                return seen;
            }
        }
        std::thread::sleep(PAUSE);
        if ended(&mut seen) {
            return seen;
        }
    }
}

/// Lanes: every stem of the playing track by `from`, the next by `into`.
pub fn lanes(from: Envelope, into: Envelope) -> [Envelope; 2 * STEMS] {
    std::array::from_fn(|i| {
        if i < STEMS {
            from.clone()
        } else {
            into.clone()
        }
    })
}

/// Beats every `pa` seconds of the playing track from `a`, paired with
/// every `pb` of the next from `b`.
pub fn beats(a: f64, pa: f64, b: f64, pb: f64, count: usize) -> Vec<(f64, f64)> {
    (0..count)
        .map(|i| (a + pa * i as f64, b + pb * i as f64))
        .collect()
}

/// Every stem as it is.
pub const ALL: [f64; STEMS] = [1.0; STEMS];

/// The drums and the bass (in [`jodel_playback::stems::LANES`]' order).
pub fn drums_bass() -> [f64; STEMS] {
    [1.0, 1.0, 0.0, 0.0]
}

pub fn drums_only() -> [f64; STEMS] {
    [1.0, 0.0, 0.0, 0.0]
}
