//! Test support for transitions, on a desk: a player with two heads on a
//! mixer's desk over a memory device ([`Desk`]), the quarter separator, one
//! transition test at a time, tone and click files, the output let through
//! a step at a time, the stems' strips muted as a test has them, and the
//! plan's lanes and beats.

use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use jodel_output::mixer::Mixer;
use jodel_output::{Capabilities, SimClock, StreamFormat};
use jodel_playback::stems::{Separator, STEMS};
use jodel_playback::{
    Captures, Envelope, Event, HeadFaders, Player, PlayerOptions, StemDesk, TransitionState,
    NEXT_LEAD,
};

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
/// too little time.
pub static SERIAL: Mutex<()> = Mutex::new(());

pub fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn quarter() -> Option<&'static dyn Separator> {
    Some(&QUARTER)
}

pub fn options() -> PlayerOptions {
    PlayerOptions {
        position_interval: Duration::from_millis(5),
        separator: Some(quarter),
        // The gated output runs faster than real time: the stems are
        // waited for, however slow the machine.
        offline: true,
        ..PlayerOptions::default()
    }
}

/// The playing track's stream frame that the next track's frame 0 is
/// linked to, for a window from `start` seconds of it (at [`RATE`]): the
/// output frames reported after the hand-over are the next track's, these
/// fewer.
pub fn origin(start: f64) -> u64 {
    ((start - NEXT_LEAD) * f64::from(RATE)).round() as u64
}

/// A player with two heads on a desk ([`StemDesk::two_heads`]) over a
/// memory device: what the test drives, hears and looks at. Dropped (or
/// [`Desk::close`]d), the player goes before the mixer.
pub struct Desk {
    pub player: Player,
    pub events: Receiver<Event>,
    /// Both heads' frames, let through together.
    pub gate: Arc<Gate>,
    pub captures: Captures,
    pub desk: StemDesk,
    /// The faders of the head the player starts on (head 0).
    pub lead: HeadFaders,
    pub mixer: Mixer,
    /// The device plays in real time, the heads let through.
    paced: bool,
}

impl Desk {
    /// On a device that plays what it is given at once, the heads gated.
    pub fn new(caps: Capabilities, options: PlayerOptions) -> Desk {
        Desk::on(caps, options, SimClock::Instant)
    }

    /// On a device that plays in real time, holding what is written ahead
    /// of it as a device does; the heads let through.
    pub fn paced(caps: Capabilities, options: PlayerOptions) -> Desk {
        let mut desk = Desk::on(caps, options, SimClock::Paced);
        desk.gate.allow_all();
        desk.paced = true;
        desk
    }

    /// On a device that plays a period every period of wall-clock time,
    /// whatever it is given (an output short of frames is heard short), the
    /// heads let through.
    pub fn real_time(caps: Capabilities, options: PlayerOptions) -> Desk {
        let mut desk = Desk::on(caps, options, SimClock::Realtime);
        desk.gate.allow_all();
        desk.paced = true;
        desk
    }

    fn on(caps: Capabilities, options: PlayerOptions, clock: SimClock) -> Desk {
        let (lead, guest, captures, gate, desk, mixer) = gated_desk(caps, clock);
        let faders = lead.faders.clone();
        let (player, events) = Player::on_desk(lead, guest, options);
        Desk {
            player,
            events,
            gate,
            captures,
            desk,
            lead: faders,
            mixer,
            paced: false,
        }
    }

    /// Loads `files` and plays the first, a little let through, until it
    /// starts.
    pub fn play(&self, files: Vec<PathBuf>) {
        self.player.load(files).unwrap();
        if !self.paced {
            self.gate.allow(2_048);
        }
        self.player.play(0).unwrap();
        collect_until(&self.events, |e| matches!(e, Event::TrackStarted(_)));
    }

    /// Each stem's strip muted where `gains` (in
    /// [`jodel_playback::stems::LANES`]' order) has 0, at once.
    pub fn mask(&self, gains: [f64; STEMS]) {
        for (strip, gain) in self.desk.stems.iter().zip(gains) {
            strip.set_mute(gain == 0.0);
        }
    }

    /// The device's streams, once the player and the mixer are gone.
    pub fn close(self) -> Vec<(StreamFormat, Vec<u8>)> {
        let Desk {
            player,
            mixer,
            captures,
            ..
        } = self;
        drop(player);
        drop(mixer);
        captured(&captures)
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

/// Two tones, one a channel, at 0.4 of full scale.
pub fn tone(hz: [f64; 2]) -> impl Fn(f64, usize) -> f64 {
    move |t, c| 0.4 * (2.0 * std::f64::consts::PI * hz[c] * t).sin()
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

/// Where clicks start in `got` (stereo), in seconds: the left channel's
/// first sample past 15 % of full scale at least 0.2 s after the last
/// click's.
pub fn onsets(got: &[i32]) -> Vec<f64> {
    let rate = f64::from(RATE);
    let mut out: Vec<f64> = Vec::new();
    for f in 0..got.len() / 2 {
        let t = f as f64 / rate;
        let v = f64::from(got[2 * f]).abs() / FULL;
        if v > 0.15 && out.last().is_none_or(|&last| t - last > 0.2) {
            out.push(t);
        }
    }
    out
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
/// shut (so each step leaves the next track's head time to write ahead,
/// however long the engine took to write it, and the pace follows the
/// build).
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
