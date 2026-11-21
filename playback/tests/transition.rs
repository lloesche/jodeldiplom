//! Transitions: the playing track and the next one heard together over a
//! window, each faded by its lanes, the next one's beats on the playing
//! one's, then the next one on its own and back on its own route. The
//! separator gives each stem a quarter of the mix, so the stems of a track
//! add up to the track, and what comes out can be worked out.

mod common;

use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use common::*;
use jodel_output::SampleFormat;
use jodel_playback::stems::{Separator, StemMix, STEMS};
use jodel_playback::{
    Envelope, Event, Player, PlayerOptions, Settings, Stage, TempoMode, TrackStart, TransitionPlan,
    TransitionState,
};
use SampleFormat::*;

const RATE: u32 = 44_100;
const FULL: f64 = 8_388_608.0;

/// Each stem a quarter of the mix.
struct Quarter;

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

static QUARTER: Quarter = Quarter;

/// One test at a time: each runs two tracks' stems at several times real
/// time, and side by side (debug-built above all) they leave each other
/// too little time, so the engine's real-time fallbacks set in (a next
/// track that falls behind is left out of the mix, a route switch that
/// does not catch up is not made).
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn quarter() -> Option<&'static dyn Separator> {
    Some(&QUARTER)
}

fn options() -> PlayerOptions {
    PlayerOptions {
        settings: Settings {
            resample: false,
            stems: StemMix::ALL,
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
fn wav(dir: &Path, name: &str, seconds: f64, f: impl Fn(f64, usize) -> f64) -> (PathBuf, Vec<i32>) {
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
fn clicks(period: f64, first: f64, hz: f64) -> impl Fn(f64, usize) -> f64 {
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
fn samples(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|b| (i32::from_le_bytes([b[0], b[1], b[2], b[3]]) << 8) >> 8)
        .collect()
}

fn starts(events: &[Event]) -> Vec<&TrackStart> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::TrackStarted(s) => Some(&**s),
            _ => None,
        })
        .collect()
}

fn transitions(events: &[Event]) -> Vec<&TransitionState> {
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
fn trickle_to_end(gate: &Gate, events: &Receiver<Event>) -> Vec<Event> {
    trickle_until(gate, events, |e| matches!(e, Event::QueueEnded))
}

/// [`trickle_to_end`] until an event `done` says so (or the queue ends).
fn trickle_until(
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
fn lanes(from: Envelope, into: Envelope) -> [Envelope; 2 * STEMS] {
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
fn beats(a: f64, pa: f64, b: f64, pb: f64, count: usize) -> Vec<(f64, f64)> {
    (0..count)
        .map(|i| (a + pa * i as f64, b + pb * i as f64))
        .collect()
}

#[test]
fn a_fade_at_one_tempo_mixes_the_two_and_hands_over() {
    let _serial = serial();
    let dir = temp_dir("transition-fade");
    let tone =
        |hz: [f64; 2]| move |t: f64, c: usize| 0.4 * (2.0 * std::f64::consts::PI * hz[c] * t).sin();
    let (a_path, a) = wav(&dir, "a.wav", 12.0, tone([1_000.0, 5_000.0]));
    let (b_path, b) = wav(&dir, "b.wav", 12.0, tone([2_000.0, 3_000.0]));
    let (target, captures, gate) = GatedTarget::new(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options());
    player.load(vec![a_path, b_path]).unwrap();
    gate.allow(2_048);
    player.play(0).unwrap();
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    // From 5 s of A, over 2 s, into B at 1 s: B's time is A's less 4 s.
    let plan = TransitionPlan {
        from: 0,
        into: 1,
        start: 5.0,
        length: 2.0,
        entry: 1.0,
        beats: beats(5.0, 0.5, 1.0, 0.5, 5),
        lanes: lanes(
            Envelope::fade(1.0, 0.0, 0.0, 1.0),
            Envelope::fade(0.0, 1.0, 0.0, 1.0),
        ),
        tempo: TempoMode::Varispeed,
        settle: 1.0,
        glide: false,
    };
    player.transition(plan.clone()).unwrap();
    let seen = trickle_to_end(&gate, &events);
    drop(player);
    let states = transitions(&seen);
    assert_eq!(states[0], &TransitionState::Armed, "{states:?}");
    let w0 = 5 * RATE as u64;
    assert!(
        states.contains(&&TransitionState::Started { output_frame: w0 }),
        "{states:?}"
    );
    assert!(states.contains(&&TransitionState::Finished), "{states:?}");
    let legs = starts(&seen);
    let into = legs.iter().find(|s| s.index == 1).expect("B started");
    assert_eq!(into.output_frame, 7 * u64::from(RATE));
    assert_eq!(into.from_frame, 3 * u64::from(RATE));
    let streams = captured(&captures);
    assert_eq!(streams.len(), 1, "one stream throughout");
    let got = samples(&streams[0].1);
    let rate = RATE as usize;
    // A as it is before it moves onto its stems.
    assert!(got[..2 * rate / 2] == a[..2 * rate / 2]);
    // Inside the window: each faded by its lanes (equal power).
    let offset = 4 * rate;
    let (mut worst, mut at) = (0.0f64, 0);
    for f in (5 * rate + 200)..(7 * rate - 200) {
        let share = plan.share(f as f64 / f64::from(RATE));
        let (ga, gb) = (plan.lanes[0].gain(share), plan.lanes[STEMS].gain(share));
        for c in 0..2 {
            let want = ga * f64::from(a[2 * f + c]) + gb * f64::from(b[2 * (f - offset) + c]);
            let off = (f64::from(got[2 * f + c]) - want).abs();
            if off > worst {
                (worst, at) = (off, 2 * f + c);
            }
        }
    }
    // The gains follow the lanes in short steps, ramped (so a cut does not
    // click): within half a percent of full scale where a fade is steepest.
    let stages: Vec<_> = legs
        .iter()
        .map(|s| (s.index, s.route.stage, s.output_frame, s.from_frame))
        .collect();
    assert!(
        worst < 0.005 * FULL,
        "off by {worst} in the window at {:.4} s, channel {}; {states:?}; starts {stages:?}; A there {}, B there {}, got {}",
        (at / 2) as f64 / f64::from(RATE),
        at % 2,
        a[at],
        b[at - 2 * offset],
        got[at]
    );
    // B on its own route at the end, exactly the file.
    let end = got.len() / 2;
    assert_eq!(end, 16 * rate, "A to 7 s, then B from 3 s to its end");
    let tail = (end - rate)..end;
    let shift = (-3i64..=3).find(|&d| {
        tail.clone()
            .all(|f| got[2 * f] == b[(2 * (f - offset) as i64 + 2 * d) as usize])
    });
    let stages: Vec<_> = legs
        .iter()
        .map(|s| (s.index, s.route.stage, s.output_frame))
        .collect();
    assert_eq!(
        shift,
        Some(0),
        "B at the end, shifted by {shift:?}; starts {stages:?}"
    );
    assert!(
        tail.clone().all(|f| got[2 * f] == b[2 * (f - offset)] && got[2 * f + 1] == b[2 * (f - offset) + 1]),
        "B is bit-perfect again at the end"
    );
    assert_eq!(legs.last().unwrap().route.stage, Stage::Passthrough);
}

#[test]
fn the_next_tracks_beats_fall_on_the_playing_ones() {
    let _serial = serial();
    beats_meet(TempoMode::Varispeed, 0.001);
}

#[test]
fn stretched_the_next_tracks_beats_fall_on_the_playing_ones_too() {
    let _serial = serial();
    beats_meet(TempoMode::Stretch, 0.004);
}

/// B's clicks on A's beats (within `within` seconds) in the window, and
/// back at B's own tempo at the end, with `tempo`.
fn beats_meet(tempo: TempoMode, within: f64) {
    let dir = temp_dir(&format!("transition-beats-{tempo:?}"));
    // A at 120 BPM, B at 125: B plays 4 % slower in the window.
    let (a_path, _) = wav(&dir, "a.wav", 12.0, clicks(0.5, 0.0, 3_000.0));
    let (b_path, _) = wav(&dir, "b.wav", 12.0, clicks(0.48, 0.0, 5_000.0));
    let (target, captures, gate) = GatedTarget::new(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options());
    player.load(vec![a_path, b_path]).unwrap();
    gate.allow(2_048);
    player.play(0).unwrap();
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    // Only B is heard from the window's start (A cut, B cut in).
    let plan = TransitionPlan {
        from: 0,
        into: 1,
        start: 5.0,
        length: 2.0,
        entry: 1.92,
        beats: beats(5.0, 0.5, 1.92, 0.48, 5),
        lanes: lanes(Envelope::cut(1.0, 0.0, 0.0), Envelope::cut(0.0, 1.0, 0.0)),
        tempo,
        settle: 2.0,
        glide: false,
    };
    player.transition(plan).unwrap();
    let seen = trickle_to_end(&gate, &events);
    drop(player);
    assert!(transitions(&seen).contains(&&TransitionState::Finished));
    let got = samples(&captured(&captures)[0].1);
    // The clicks' starts: the left channel's first sample past 15 % of
    // full scale at least 0.2 s after the last click's.
    let rate = f64::from(RATE);
    let onsets: Vec<f64> = {
        let mut out: Vec<f64> = Vec::new();
        for f in 0..got.len() / 2 {
            let t = f as f64 / rate;
            let v = f64::from(got[2 * f]).abs() / FULL;
            if v > 0.15 && out.last().is_none_or(|&last| t - last > 0.2) {
                out.push(t);
            }
        }
        out
    };
    // In the window, on A's beats (5.0, 5.5, ...) within a millisecond.
    let in_window: Vec<f64> = onsets
        .iter()
        .copied()
        .filter(|&t| (5.1..7.0).contains(&t))
        .collect();
    assert!(in_window.len() >= 3, "{onsets:?}");
    for t in &in_window {
        let beat = ((t - 5.0) / 0.5).round() * 0.5 + 5.0;
        assert!(
            (t - beat).abs() < within,
            "click at {t}, beat at {beat}: {onsets:?}"
        );
    }
    // Back at its own tempo at the end: 0.48 s apart.
    let late: Vec<f64> = onsets.iter().copied().filter(|&t| t > 11.0).collect();
    for w in late.windows(2) {
        assert!((w[1] - w[0] - 0.48).abs() < within, "{late:?}");
    }
}

#[test]
fn gliding_the_tempo_meets_the_beats_and_ends_at_the_next_ones_own() {
    let _serial = serial();
    glide_meets(TempoMode::Varispeed, 0.001, StemMix::ALL);
}

#[test]
fn stretched_gliding_the_tempo_meets_the_beats_too() {
    let _serial = serial();
    glide_meets(TempoMode::Stretch, 0.004, StemMix::ALL);
}

/// The same with a stem off: the playing track is on its stems from the
/// start, and moves onto a fresh stems track that glides.
#[test]
fn gliding_from_a_track_on_its_stems_meets_the_beats_too() {
    use jodel_playback::stems::Stem;
    let _serial = serial();
    glide_meets(
        TempoMode::Varispeed,
        0.001,
        StemMix::NONE.toggle(Stem::Drums).toggle(Stem::Bass),
    );
}

/// A at 120 BPM, B at 126, the tempo gliding across a 4 s window: A alone
/// through its first half, B alone through its second. Every click (A's,
/// then B's) is where A's beat is heard along the glide, within `within`
/// seconds; A's are closer together as it goes; after the window B is at
/// its own tempo at once; and the position heard follows the glide. With
/// `stems` on (half of them: the clicks at half their level).
fn glide_meets(tempo: TempoMode, within: f64, stems: StemMix) {
    let dir = temp_dir(&format!("transition-glide-{tempo:?}-{}", stems.is_full()));
    let b_period = 60.0 / 126.0;
    let (a_path, _) = wav(&dir, "a.wav", 14.0, clicks(0.5, 0.0, 3_000.0));
    let (b_path, _) = wav(&dir, "b.wav", 14.0, clicks(b_period, 0.0, 5_000.0));
    let (target, captures, gate) = GatedTarget::new(caps(&[RATE], &[S24In32Low]));
    let options = PlayerOptions {
        settings: Settings {
            stems,
            ..options().settings
        },
        ..options()
    };
    let (player, events) = Player::new(Box::new(target), options);
    player.load(vec![a_path, b_path]).unwrap();
    gate.allow(2_048);
    player.play(0).unwrap();
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    let entry = 4.0 * b_period;
    let plan = TransitionPlan {
        from: 0,
        into: 1,
        start: 5.0,
        length: 4.0,
        entry,
        beats: beats(5.0, 0.5, entry, b_period, 9),
        lanes: lanes(Envelope::cut(1.0, 0.0, 0.5), Envelope::cut(0.0, 1.0, 0.5)),
        tempo,
        settle: 0.5,
        glide: true,
    };
    let glide = plan.glide_of(&plan.time_map());
    assert!((glide.end_speed() - 1.05).abs() < 1e-3, "{glide:?}");
    player.transition(plan).unwrap();
    // The output held at 2.5 s for a while, as a device playing in real
    // time leaves A's move onto its stems (from 1 s on) the time it needs
    // (the trickle goes several times faster than that).
    gate.allow(5 * u64::from(RATE) / 2);
    gate.wait_written(5 * u64::from(RATE) / 2);
    std::thread::sleep(Duration::from_millis(1_500));
    let seen = trickle_to_end(&gate, &events);
    assert!(
        !seen.iter().any(|e| matches!(e, Event::Notice { .. })),
        "{seen:?}"
    );
    drop(player);
    assert!(
        transitions(&seen).contains(&&TransitionState::Finished),
        "{:?}",
        transitions(&seen)
    );
    let got = samples(&captured(&captures)[0].1);
    let rate = f64::from(RATE);
    let onsets: Vec<f64> = {
        let mut out: Vec<f64> = Vec::new();
        for f in 0..got.len() / 2 {
            let t = f as f64 / rate;
            let v = f64::from(got[2 * f]).abs() / FULL;
            if v > 0.15 && out.last().is_none_or(|&last| t - last > 0.2) {
                out.push(t);
            }
        }
        out
    };
    // A's beats as heard along the glide, and the window's end.
    let heard: Vec<f64> = (0..=8)
        .map(|k| glide.inverse(5.0 + 0.5 * f64::from(k)))
        .collect();
    let end = glide.start() + glide.length();
    let in_window: Vec<f64> = onsets
        .iter()
        .copied()
        .filter(|&t| (5.1..end - 0.1).contains(&t))
        .collect();
    assert!(in_window.len() >= 6, "{onsets:?}");
    for &t in &in_window {
        let nearest = heard
            .iter()
            .copied()
            .min_by(|a, b| (a - t).abs().total_cmp(&(b - t).abs()))
            .unwrap();
        assert!(
            (t - nearest).abs() < within,
            "click at {t}, beat heard at {nearest}: {onsets:?}"
        );
    }
    // Closer together as A speeds up.
    let gaps: Vec<f64> = in_window.windows(2).map(|w| w[1] - w[0]).collect();
    assert!(gaps.first().unwrap() > gaps.last().unwrap(), "{gaps:?}");
    // After the window, B at its own tempo straight away.
    let after: Vec<f64> = onsets.iter().copied().filter(|&t| t > end + 0.1).collect();
    assert!(after.len() >= 4, "{onsets:?}");
    for w in after.windows(2) {
        assert!((w[1] - w[0] - b_period).abs() < within, "{after:?}");
    }
    // The position heard in the window: A's time along the glide.
    let positions: Vec<(f64, f64)> = seen
        .iter()
        .filter_map(|e| match e {
            Event::Position(p) if p.index == 0 => Some((p.output_frame as f64 / rate, p.seconds)),
            _ => None,
        })
        .filter(|&(u, _)| (5.2..end - 0.2).contains(&u))
        .collect();
    assert!(!positions.is_empty());
    for (u, s) in positions {
        assert!(
            (s - glide.at(u)).abs() < 0.005,
            "at {u}: {s}, glide {}",
            glide.at(u)
        );
    }
}
/// A gliding transition called off in its window (its next track taken
/// out of the queue):
/// the playing track goes on from where the glide had it, at its own
/// speed, without a jump (its clicks keep their place and then their
/// spacing).
#[test]
fn a_glide_called_off_in_its_window_goes_on_without_a_jump() {
    let _serial = serial();
    let dir = temp_dir("transition-glide-off");
    let b_period = 60.0 / 126.0;
    let (a_path, _) = wav(&dir, "a.wav", 14.0, clicks(0.5, 0.0, 3_000.0));
    let (b_path, _) = wav(&dir, "b.wav", 14.0, clicks(b_period, 0.0, 5_000.0));
    let (target, captures, gate) = GatedTarget::new(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options());
    player.load(vec![a_path, b_path]).unwrap();
    gate.allow(2_048);
    player.play(0).unwrap();
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    let entry = 4.0 * b_period;
    let plan = TransitionPlan {
        from: 0,
        into: 1,
        start: 5.0,
        length: 4.0,
        entry,
        beats: beats(5.0, 0.5, entry, b_period, 9),
        // Only A is heard.
        lanes: lanes(Envelope::flat(1.0), Envelope::flat(0.0)),
        tempo: TempoMode::Varispeed,
        settle: 0.5,
        glide: true,
    };
    let glide = plan.glide_of(&plan.time_map());
    player.transition(plan).unwrap();
    gate.allow(5 * u64::from(RATE) / 2);
    gate.wait_written(5 * u64::from(RATE) / 2);
    std::thread::sleep(Duration::from_millis(1_500));
    // Into the window, then other settings call it off.
    let mut seen = trickle_until(
        &gate,
        &events,
        |e| matches!(e, Event::Position(p) if p.output_frame >= 7 * u64::from(RATE)),
    );
    player.remove(1).unwrap();
    seen.extend(trickle_to_end(&gate, &events));
    drop(player);
    assert!(
        transitions(&seen)
            .iter()
            .any(|s| matches!(s, TransitionState::Cancelled { .. })),
        "{:?}",
        transitions(&seen)
    );
    let got = samples(&captured(&captures)[0].1);
    let rate = f64::from(RATE);
    let onsets: Vec<f64> = {
        let mut out: Vec<f64> = Vec::new();
        for f in 0..got.len() / 2 {
            let t = f as f64 / rate;
            let v = f64::from(got[2 * f]).abs() / FULL;
            if v > 0.15 && out.last().is_none_or(|&last| t - last > 0.2) {
                out.push(t);
            }
        }
        out
    };
    // Along the glide up to where it was called off, then half a second
    // apart (the gap across the change between the glide's and that).
    let mut gliding = true;
    let mut last = 0.0;
    for &t in onsets.iter().filter(|&&t| t > 5.1 && t < 12.0) {
        let on_glide = (0..=8)
            .map(|k| glide.inverse(5.0 + 0.5 * f64::from(k)))
            .any(|u| (u - t).abs() < 0.002);
        if gliding && on_glide {
            last = t;
            continue;
        }
        gliding = false;
        let gap = t - last;
        assert!(
            (0.5 / glide.end_speed() - 0.003..0.503).contains(&gap),
            "a gap of {gap} s at {t}: {onsets:?}"
        );
        last = t;
    }
    assert!(!gliding, "it went on gliding: {onsets:?}");
}
#[test]
fn a_transition_is_called_off_by_a_seek_and_when_it_does_not_fit() {
    let _serial = serial();
    let dir = temp_dir("transition-off");
    let tone = |t: f64, _c: usize| 0.3 * (2.0 * std::f64::consts::PI * 440.0 * t).sin();
    let (a_path, _) = wav(&dir, "a.wav", 12.0, tone);
    let (b_path, _) = wav(&dir, "b.wav", 12.0, tone);
    let (target, _captures, gate) = GatedTarget::new(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options());
    player.load(vec![a_path, b_path]).unwrap();
    gate.allow(2_048);
    player.play(0).unwrap();
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    let plan = TransitionPlan {
        from: 0,
        into: 1,
        start: 8.0,
        length: 2.0,
        entry: 0.0,
        beats: Vec::new(),
        lanes: lanes(
            Envelope::fade(1.0, 0.0, 0.0, 1.0),
            Envelope::fade(0.0, 1.0, 0.0, 1.0),
        ),
        tempo: TempoMode::Varispeed,
        settle: 0.0,
        glide: false,
    };
    // Not the playing track: called off at once.
    player
        .transition(TransitionPlan {
            from: 1,
            into: 2,
            ..plan.clone()
        })
        .unwrap();
    let seen = collect_until(&events, |e| matches!(e, Event::Transition(_)));
    assert!(matches!(
        transitions(&seen)[0],
        TransitionState::Cancelled { .. }
    ));
    // Armed, then a seek calls it off.
    player.transition(plan).unwrap();
    collect_until(&events, |e| {
        matches!(e, Event::Transition(TransitionState::Armed))
    });
    player.seek(2.0).unwrap();
    let seen = collect_until(&events, |e| matches!(e, Event::Transition(_)));
    assert!(
        matches!(transitions(&seen)[0], TransitionState::Cancelled { reason } if reason == "a seek"),
        "{seen:?}"
    );
    gate.allow_all();
    let rest = until_queue_end(&events);
    // Without it, the tracks follow each other as usual.
    assert_eq!(starts(&rest).iter().filter(|s| s.index == 1).count(), 1);
    assert!(transitions(&rest).is_empty());
}

#[test]
fn without_a_separator_the_tracks_fade_whole() {
    let _serial = serial();
    let dir = temp_dir("transition-whole");
    let tone =
        |hz: [f64; 2]| move |t: f64, c: usize| 0.4 * (2.0 * std::f64::consts::PI * hz[c] * t).sin();
    let (a_path, a) = wav(&dir, "a.wav", 10.0, tone([1_000.0, 5_000.0]));
    let (b_path, b) = wav(&dir, "b.wav", 10.0, tone([2_000.0, 3_000.0]));
    let (target, captures, gate) = GatedTarget::new(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(
        Box::new(target),
        PlayerOptions {
            separator: None,
            ..options()
        },
    );
    player.load(vec![a_path, b_path]).unwrap();
    gate.allow(2_048);
    player.play(0).unwrap();
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    let plan = TransitionPlan {
        from: 0,
        into: 1,
        start: 5.0,
        length: 2.0,
        entry: 1.0,
        beats: beats(5.0, 0.5, 1.0, 0.5, 5),
        lanes: lanes(
            Envelope::fade(1.0, 0.0, 0.0, 1.0),
            Envelope::fade(0.0, 1.0, 0.0, 1.0),
        ),
        tempo: TempoMode::Varispeed,
        settle: 1.0,
        glide: false,
    };
    player.transition(plan.clone()).unwrap();
    let seen = trickle_to_end(&gate, &events);
    drop(player);
    assert!(transitions(&seen).contains(&&TransitionState::Finished));
    let got = samples(&captured(&captures)[0].1);
    let rate = RATE as usize;
    let offset = 4 * rate;
    let mut worst = 0.0f64;
    for f in (5 * rate + 200)..(7 * rate - 200) {
        let share = plan.share(f as f64 / f64::from(RATE));
        let (ga, gb) = (plan.lanes[0].gain(share), plan.lanes[STEMS].gain(share));
        for c in 0..2 {
            let want = ga * f64::from(a[2 * f + c]) + gb * f64::from(b[2 * (f - offset) + c]);
            worst = worst.max((f64::from(got[2 * f + c]) - want).abs());
        }
    }
    assert!(worst < 0.005 * FULL, "off by {worst}");
}

#[test]
fn the_next_track_is_heard_where_the_map_puts_it() {
    let _serial = serial();
    use jodel_playback::warp::TimeMap;
    let dir = temp_dir("transition-heard");
    let (a_path, _) = wav(&dir, "a.wav", 12.0, clicks(0.5, 0.0, 3_000.0));
    let (b_path, _) = wav(&dir, "b.wav", 12.0, clicks(0.48, 0.0, 5_000.0));
    let (target, _captures, gate) = GatedTarget::new(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options());
    player.load(vec![a_path, b_path]).unwrap();
    gate.allow(2_048);
    player.play(0).unwrap();
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    let pairs = beats(5.0, 0.5, 1.92, 0.48, 5);
    let plan = TransitionPlan {
        from: 0,
        into: 1,
        start: 5.0,
        length: 2.0,
        entry: 1.92,
        beats: pairs.clone(),
        lanes: lanes(Envelope::cut(1.0, 0.0, 0.0), Envelope::cut(0.0, 1.0, 0.0)),
        tempo: TempoMode::Varispeed,
        settle: 2.0,
        glide: false,
    };
    player.transition(plan).unwrap();
    let seen = trickle_to_end(&gate, &events);
    drop(player);
    let map = TimeMap::new(&pairs, 2.0);
    // While B glides back to its tempo (from 7 s to 9 s of A's time), the
    // positions reported are those of the map; after that, a straight line.
    let mut checked = 0;
    for e in &seen {
        let Event::Position(p) = e else { continue };
        if p.index != 1 {
            continue;
        }
        let a_time = p.output_frame as f64 / f64::from(RATE);
        if a_time < 7.05 {
            continue;
        }
        let want = map.at(a_time);
        assert!(
            (p.seconds - want).abs() < 0.002,
            "at {a_time}: {} {want}",
            p.seconds
        );
        checked += 1;
    }
    assert!(checked > 20, "{checked} positions");
}

#[test]
fn next_while_a_transition_mixes_cuts_into_the_next_track_where_it_is() {
    let _serial = serial();
    use jodel_playback::warp::TimeMap;
    let dir = temp_dir("transition-cut");
    let (a_path, _) = wav(&dir, "a.wav", 14.0, clicks(0.5, 0.0, 3_000.0));
    let (b_path, _) = wav(&dir, "b.wav", 14.0, clicks(0.48, 0.0, 5_000.0));
    let (target, _captures, gate) = GatedTarget::new(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options());
    player.load(vec![a_path, b_path]).unwrap();
    gate.allow(2_048);
    player.play(0).unwrap();
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    // A window of 4 s from 5 s of A, B entering at 1.92 s.
    let pairs = beats(5.0, 0.5, 1.92, 0.48, 9);
    let plan = TransitionPlan {
        from: 0,
        into: 1,
        start: 5.0,
        length: 4.0,
        entry: 1.92,
        beats: pairs.clone(),
        lanes: lanes(
            Envelope::fade(1.0, 0.0, 0.0, 1.0),
            Envelope::fade(0.0, 1.0, 0.0, 1.0),
        ),
        tempo: TempoMode::Varispeed,
        settle: 0.0,
        glide: false,
    };
    player.transition(plan).unwrap();
    // A second into the window, next.
    trickle_until(
        &gate,
        &events,
        |e| matches!(e, Event::Position(p) if p.index == 0 && p.seconds >= 6.0),
    );
    player.next().unwrap();
    let seen = trickle_to_end(&gate, &events);
    drop(player);
    assert!(
        transitions(&seen).iter().any(
            |t| matches!(t, TransitionState::Cancelled { reason } if reason == "cut to the next track")
        ),
        "{:?}",
        transitions(&seen)
    );
    // B goes on where the mix had it (not from its start), and A is gone.
    let map = TimeMap::new(&pairs, 0.0);
    let first_b = seen
        .iter()
        .position(|e| matches!(e, Event::Position(p) if p.index == 1))
        .expect("B heard");
    let Event::Position(p) = &seen[first_b] else {
        unreachable!()
    };
    let at_cut = map.at(6.0);
    assert!(
        p.seconds >= at_cut - 0.05 && p.seconds < at_cut + 1.0,
        "B from {} where the mix had it at {at_cut}",
        p.seconds
    );
    assert!(
        !seen[first_b..]
            .iter()
            .any(|e| matches!(e, Event::Position(p) if p.index == 0)),
        "A heard after the cut"
    );
}

#[test]
fn a_transition_called_off_while_it_prepares_leaves_the_track_as_it_was() {
    let _serial = serial();
    let dir = temp_dir("transition-cancel");
    let tone = |t: f64, _c: usize| 0.3 * (2.0 * std::f64::consts::PI * 440.0 * t).sin();
    let (a_path, a) = wav(&dir, "a.wav", 12.0, tone);
    let (b_path, _) = wav(&dir, "b.wav", 12.0, tone);
    let (target, captures, gate) = GatedTarget::new(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options());
    player.load(vec![a_path, b_path]).unwrap();
    gate.allow(2_048);
    player.play(0).unwrap();
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    player
        .transition(TransitionPlan {
            from: 0,
            into: 1,
            start: 6.0,
            length: 2.0,
            entry: 0.0,
            beats: Vec::new(),
            lanes: lanes(
                Envelope::fade(1.0, 0.0, 0.0, 1.0),
                Envelope::fade(0.0, 1.0, 0.0, 1.0),
            ),
            tempo: TempoMode::Varispeed,
            settle: 0.0,
            glide: false,
        })
        .unwrap();
    // Into the preparation (from 2 s of A), then called off.
    while gate.written() < 3 * u64::from(RATE) {
        gate.allow(gate.written() + 4_096);
        std::thread::sleep(Duration::from_millis(15));
    }
    player.cancel_transition().unwrap();
    let seen = trickle_to_end(&gate, &events);
    drop(player);
    assert!(
        transitions(&seen)
            .iter()
            .any(|s| matches!(s, TransitionState::Cancelled { .. })),
        "{seen:?}"
    );
    // A plays on to its end, back on its own route: the file exactly at
    // its end, and nothing faded where the window would have been.
    let got = samples(&captured(&captures)[0].1);
    assert_eq!(got.len(), 2 * a.len(), "A to its end, then B");
    let rate = RATE as usize;
    let peak = |from: usize, to: usize| {
        (2 * from..2 * to)
            .map(|i| got[i].unsigned_abs())
            .max()
            .unwrap_or(0)
    };
    assert!(
        peak(7 * rate, 8 * rate) > 2_000_000,
        "A is heard where the window was"
    );
    let tail = (11 * rate)..(12 * rate);
    assert!(
        tail.clone().all(|f| got[2 * f] == a[2 * f]),
        "A as it is at the end"
    );
}

/// Some stems off through a transition, and another off after it: the next
/// track takes the change (its stems no longer follow the transition's
/// fades, which had the stems on then in them).
#[test]
fn after_a_transition_the_next_tracks_stems_follow_the_settings() {
    use jodel_playback::stems::Stem;
    let _serial = serial();
    let dir = temp_dir("transition-stems");
    let tone =
        |hz: [f64; 2]| move |t: f64, c: usize| 0.4 * (2.0 * std::f64::consts::PI * hz[c] * t).sin();
    let (a_path, _) = wav(&dir, "a.wav", 10.0, tone([1_000.0, 5_000.0]));
    let (b_path, b) = wav(&dir, "b.wav", 10.0, tone([2_000.0, 3_000.0]));
    let settings = |stems: StemMix| Settings {
        stems,
        ..options().settings
    };
    let two = StemMix::NONE.toggle(Stem::Drums).toggle(Stem::Bass);
    let (target, captures, gate) = GatedTarget::new(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(
        Box::new(target),
        PlayerOptions {
            settings: settings(two),
            ..options()
        },
    );
    player.load(vec![a_path, b_path]).unwrap();
    gate.allow(2_048);
    player.play(0).unwrap();
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    // From 5 s of A, over 2 s, into B at 1 s: B's time is A's less 4 s.
    let plan = TransitionPlan {
        from: 0,
        into: 1,
        start: 5.0,
        length: 2.0,
        entry: 1.0,
        beats: beats(5.0, 0.5, 1.0, 0.5, 5),
        lanes: lanes(
            Envelope::fade(1.0, 0.0, 0.0, 1.0),
            Envelope::fade(0.0, 1.0, 0.0, 1.0),
        ),
        tempo: TempoMode::Varispeed,
        settle: 1.0,
        glide: false,
    };
    player.transition(plan).unwrap();
    let mut seen = trickle_until(&gate, &events, |e| {
        matches!(e, Event::Transition(TransitionState::Finished))
    });
    assert!(
        transitions(&seen).contains(&&TransitionState::Finished),
        "{:?}",
        transitions(&seen)
    );
    // The drums alone from here.
    player
        .set_settings(settings(StemMix::NONE.toggle(Stem::Drums)))
        .unwrap();
    seen.extend(trickle_to_end(&gate, &events));
    drop(player);
    let streams = captured(&captures);
    assert_eq!(streams.len(), 1, "one stream throughout");
    let got = samples(&streams[0].1);
    let rate = RATE as usize;
    // A to 7 s, then B from 3 s to its end; its last second is a quarter
    // of B (the drums alone), not a half (drums and bass).
    let (end, offset) = (got.len() / 2, 4 * rate);
    assert_eq!(end, 14 * rate);
    let worst = ((end - rate)..(end - 10))
        .flat_map(|f| (0..2).map(move |c| (f, c)))
        .map(|(f, c)| (f64::from(got[2 * f + c]) - 0.25 * f64::from(b[2 * (f - offset) + c])).abs())
        .fold(0.0, f64::max);
    assert!(worst < 64.0, "B's end is {worst} steps off a quarter of B");
}

/// Drums and bass through a transition, the bass switched off in its
/// window: the transition goes on (into its fades, not called off), and
/// after it the next track plays the drums alone.
#[test]
fn stems_switched_during_a_transition_go_into_its_fades() {
    use jodel_playback::stems::Stem;
    let _serial = serial();
    let dir = temp_dir("transition-stems-during");
    let tone =
        |hz: [f64; 2]| move |t: f64, c: usize| 0.4 * (2.0 * std::f64::consts::PI * hz[c] * t).sin();
    let (a_path, _) = wav(&dir, "a.wav", 10.0, tone([1_000.0, 5_000.0]));
    let (b_path, b) = wav(&dir, "b.wav", 10.0, tone([2_000.0, 3_000.0]));
    let settings = |stems: StemMix| Settings {
        stems,
        ..options().settings
    };
    let two = StemMix::NONE.toggle(Stem::Drums).toggle(Stem::Bass);
    let (target, captures, gate) = GatedTarget::new(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(
        Box::new(target),
        PlayerOptions {
            settings: settings(two),
            ..options()
        },
    );
    player.load(vec![a_path, b_path]).unwrap();
    gate.allow(2_048);
    player.play(0).unwrap();
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    let plan = TransitionPlan {
        from: 0,
        into: 1,
        start: 5.0,
        length: 2.0,
        entry: 1.0,
        beats: beats(5.0, 0.5, 1.0, 0.5, 5),
        lanes: lanes(
            Envelope::fade(1.0, 0.0, 0.0, 1.0),
            Envelope::fade(0.0, 1.0, 0.0, 1.0),
        ),
        tempo: TempoMode::Varispeed,
        settle: 1.0,
        glide: false,
    };
    player.transition(plan).unwrap();
    let mut seen = trickle_until(&gate, &events, |e| {
        matches!(e, Event::Transition(TransitionState::Started { .. }))
    });
    player
        .set_settings(settings(StemMix::NONE.toggle(Stem::Drums)))
        .unwrap();
    seen.extend(trickle_to_end(&gate, &events));
    drop(player);
    let states = transitions(&seen);
    assert!(states.contains(&&TransitionState::Finished), "{states:?}");
    assert!(
        !states
            .iter()
            .any(|s| matches!(s, TransitionState::Cancelled { .. })),
        "{states:?}"
    );
    let got = samples(&captured(&captures)[0].1);
    let rate = RATE as usize;
    let (end, offset) = (got.len() / 2, 4 * rate);
    assert_eq!(end, 14 * rate);
    let worst = ((end - rate)..(end - 10))
        .flat_map(|f| (0..2).map(move |c| (f, c)))
        .map(|(f, c)| (f64::from(got[2 * f + c]) - 0.25 * f64::from(b[2 * (f - offset) + c])).abs())
        .fold(0.0, f64::max);
    assert!(worst < 64.0, "B's end is {worst} steps off a quarter of B");
}

// The rules of how other controls meet a transition (player/INTERACTIONS.md).

/// Two tone tracks (10 s), a player playing the first with `stems` on, and
/// the second's samples.
fn rules_setup(
    name: &str,
    stems: StemMix,
) -> (
    TempDir,
    Vec<i32>,
    Player,
    Receiver<Event>,
    jodel_playback::Captures,
    std::sync::Arc<Gate>,
) {
    let dir = temp_dir(name);
    let tone =
        |hz: [f64; 2]| move |t: f64, c: usize| 0.4 * (2.0 * std::f64::consts::PI * hz[c] * t).sin();
    let (a_path, _) = wav(&dir, "a.wav", 10.0, tone([1_000.0, 5_000.0]));
    let (b_path, b) = wav(&dir, "b.wav", 10.0, tone([2_000.0, 3_000.0]));
    let (target, captures, gate) = GatedTarget::new(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(
        Box::new(target),
        PlayerOptions {
            settings: rules_settings(stems),
            ..options()
        },
    );
    player.load(vec![a_path, b_path]).unwrap();
    gate.allow(2_048);
    player.play(0).unwrap();
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    (dir, b, player, events, captures, gate)
}

fn rules_settings(stems: StemMix) -> Settings {
    Settings {
        stems,
        ..options().settings
    }
}

fn drums_bass() -> StemMix {
    use jodel_playback::stems::Stem;
    StemMix::NONE.toggle(Stem::Drums).toggle(Stem::Bass)
}

fn drums_only() -> StemMix {
    use jodel_playback::stems::Stem;
    StemMix::NONE.toggle(Stem::Drums)
}

/// From 5 s of A over 2 s into B at 1 s (B's time is A's less 4 s), by
/// `lanes`, settling over `settle`.
fn rules_plan(lanes: [Envelope; 2 * STEMS], settle: f64) -> TransitionPlan {
    TransitionPlan {
        from: 0,
        into: 1,
        start: 5.0,
        length: 2.0,
        entry: 1.0,
        beats: beats(5.0, 0.5, 1.0, 0.5, 5),
        lanes,
        tempo: TempoMode::Varispeed,
        settle,
        glide: false,
    }
}

fn fade_lanes() -> [Envelope; 2 * STEMS] {
    lanes(
        Envelope::fade(1.0, 0.0, 0.0, 1.0),
        Envelope::fade(0.0, 1.0, 0.0, 1.0),
    )
}

/// The window started where it was planned (5 s), it finished, and
/// nothing called it off.
fn assert_ran_as_planned(seen: &[Event]) {
    let states = transitions(seen);
    assert!(
        states.contains(&&TransitionState::Started {
            output_frame: 5 * u64::from(RATE)
        }),
        "{states:?}"
    );
    assert!(states.contains(&&TransitionState::Finished), "{states:?}");
    assert!(
        !states
            .iter()
            .any(|s| matches!(s, TransitionState::Cancelled { .. })),
        "{states:?}"
    );
}

/// The worst difference, in 24-bit steps, between output frames `range`
/// and `scale` times B where B's frame is the output's less `offset`.
fn off_b(got: &[i32], b: &[i32], range: std::ops::Range<usize>, offset: usize, scale: f64) -> f64 {
    range
        .flat_map(|f| (0..2).map(move |c| (f, c)))
        .map(|(f, c)| {
            (f64::from(got[2 * f + c]) - scale * f64::from(b[2 * (f - offset) + c])).abs()
        })
        .fold(0.0, f64::max)
}

/// S1: stems switched while armed (and again while preparing) leave the
/// transition where it was: its window starts on time, it runs, it ends.
#[test]
fn stems_switched_while_armed_leave_the_transition_where_it_was() {
    let _serial = serial();
    let (_dir, _b, player, events, _captures, gate) = rules_setup("rules-armed", StemMix::ALL);
    player.transition(rules_plan(fade_lanes(), 1.0)).unwrap();
    // Armed (the window is 5 s off): the first stem off, on a bar.
    let mut seen = trickle_until(
        &gate,
        &events,
        |e| matches!(e, Event::Position(p) if p.output_frame >= u64::from(RATE) / 2),
    );
    player.set_stems_at(drums_bass(), 1.5).unwrap();
    // Preparing (within 4 s of it): another off, at once.
    seen.extend(trickle_until(
        &gate,
        &events,
        |e| matches!(e, Event::Position(p) if p.output_frame >= 7 * u64::from(RATE) / 2),
    ));
    player.set_settings(rules_settings(drums_only())).unwrap();
    seen.extend(trickle_to_end(&gate, &events));
    drop(player);
    assert_ran_as_planned(&seen);
}

/// S1, S2: a stem switched on a bar inside the window changes there, on
/// the next track too (here alone in the window), and nothing else does.
#[test]
fn stems_switched_at_a_bar_in_the_window_change_there() {
    let _serial = serial();
    let (_dir, b, player, events, captures, gate) = rules_setup("rules-at-bar", drums_bass());
    // A out at once, B in at once.
    let plan = rules_plan(
        lanes(Envelope::cut(1.0, 0.0, 0.0), Envelope::cut(0.0, 1.0, 0.0)),
        1.0,
    );
    player.transition(plan).unwrap();
    let mut seen = trickle_until(&gate, &events, |e| {
        matches!(e, Event::Transition(TransitionState::Started { .. }))
    });
    // The bass out at 6 s of the playing track: the middle of the window.
    player.set_stems_at(drums_only(), 6.0).unwrap();
    seen.extend(trickle_to_end(&gate, &events));
    drop(player);
    assert_ran_as_planned(&seen);
    let got = samples(&captured(&captures)[0].1);
    let (rate, offset) = (RATE as usize, 4 * RATE as usize);
    // Drums and bass (half of B) up to 6 s, the drums (a quarter) after
    // (past the stems' 10 ms fade).
    let before = off_b(&got, &b, 5 * rate + rate / 20..6 * rate - 64, offset, 0.5);
    let after = off_b(&got, &b, 6 * rate + rate / 40..7 * rate - 64, offset, 0.25);
    assert!(before < 0.01 * FULL, "before the change {before}");
    assert!(after < 0.01 * FULL, "after the change {after}");
}

/// S1, S4: stems switched while the next track settles after the window
/// go into its fades; the transition ends as planned, and the next track
/// plays them after.
#[test]
fn stems_switched_while_settling_go_into_the_next_tracks_fades() {
    let _serial = serial();
    let (_dir, b, player, events, captures, gate) = rules_setup("rules-settling", drums_bass());
    player.transition(rules_plan(fade_lanes(), 2.0)).unwrap();
    // The window is over at 7 s: the next track settles.
    let mut seen = trickle_until(
        &gate,
        &events,
        |e| matches!(e, Event::Position(p) if p.index == 1 && p.output_frame >= 15 * u64::from(RATE) / 2),
    );
    player.set_settings(rules_settings(drums_only())).unwrap();
    seen.extend(trickle_to_end(&gate, &events));
    drop(player);
    assert_ran_as_planned(&seen);
    let got = samples(&captured(&captures)[0].1);
    let rate = RATE as usize;
    let end = got.len() / 2;
    assert_eq!(end, 14 * rate);
    let tail = off_b(&got, &b, end - rate..end - 10, 4 * rate, 0.25);
    assert!(tail < 64.0, "the next track's end is {tail} off its drums");
}

/// T2: calling off a mix under way does nothing: it plays to its end.
#[test]
fn calling_off_a_mix_under_way_does_nothing() {
    let _serial = serial();
    let (_dir, _b, player, events, _captures, gate) = rules_setup("rules-cancel-mix", StemMix::ALL);
    player.transition(rules_plan(fade_lanes(), 1.0)).unwrap();
    let mut seen = trickle_until(&gate, &events, |e| {
        matches!(e, Event::Transition(TransitionState::Started { .. }))
    });
    player.cancel_transition().unwrap();
    // A new plan neither replaces it.
    player.transition(rules_plan(fade_lanes(), 1.0)).unwrap();
    seen.extend(trickle_to_end(&gate, &events));
    drop(player);
    assert_ran_as_planned(&seen);
}

/// T3: pausing while preparing does not call it off: it runs on time
/// after the pause.
#[test]
fn a_paused_transition_goes_on_after_the_pause() {
    let _serial = serial();
    let (_dir, _b, player, events, _captures, gate) = rules_setup("rules-pause", StemMix::ALL);
    player.transition(rules_plan(fade_lanes(), 1.0)).unwrap();
    let mut seen = trickle_until(
        &gate,
        &events,
        |e| matches!(e, Event::Position(p) if p.output_frame >= 7 * u64::from(RATE) / 2),
    );
    player.pause().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    player.resume().unwrap();
    seen.extend(trickle_to_end(&gate, &events));
    drop(player);
    assert_ran_as_planned(&seen);
}

/// T4: other settings during the mix do not call it off; they wait for
/// its end, and a notice says so.
#[test]
fn other_settings_during_a_mix_wait_for_its_end() {
    let _serial = serial();
    let (_dir, _b, player, events, _captures, gate) = rules_setup("rules-settings", StemMix::ALL);
    player.transition(rules_plan(fade_lanes(), 1.0)).unwrap();
    let mut seen = trickle_until(&gate, &events, |e| {
        matches!(e, Event::Transition(TransitionState::Started { .. }))
    });
    player
        .set_settings(Settings {
            resample: true,
            ..options().settings
        })
        .unwrap();
    seen.extend(trickle_to_end(&gate, &events));
    drop(player);
    assert_ran_as_planned(&seen);
    assert!(
        seen.iter().any(|e| matches!(e, Event::Notice { message, .. } if message.contains("once the transition is over"))),
        "no notice"
    );
}

/// T1: another plan for the same window while it prepares (another preset,
/// another tempo mode) keeps the window where it was.
#[test]
fn a_plan_changed_while_preparing_keeps_its_window() {
    let _serial = serial();
    let (_dir, _b, player, events, _captures, gate) = rules_setup("rules-replan", drums_bass());
    player.transition(rules_plan(fade_lanes(), 1.0)).unwrap();
    let mut seen = trickle_until(
        &gate,
        &events,
        |e| matches!(e, Event::Position(p) if p.output_frame >= 7 * u64::from(RATE) / 2),
    );
    let mut again = rules_plan(
        lanes(Envelope::cut(1.0, 0.0, 0.5), Envelope::cut(0.0, 1.0, 0.5)),
        1.0,
    );
    again.tempo = TempoMode::Stretch;
    player.transition(again).unwrap();
    seen.extend(trickle_to_end(&gate, &events));
    drop(player);
    assert_ran_as_planned(&seen);
}

/// Q1: tracks moved around the transition's two: it follows them; its
/// next track taken out: it is called off, and the playing track goes on.
#[test]
fn a_transition_follows_its_tracks_through_the_queue() {
    let _serial = serial();
    let (dir, _b, player, events, _captures, gate) = rules_setup("rules-queue", StemMix::ALL);
    // A third track after the two, moved before the playing one: the
    // transition's tracks are 1 and 2 then.
    let (c_path, _) = wav(&dir, "c.wav", 1.0, |_, _| 0.0);
    player.enqueue(vec![c_path]).unwrap();
    player.transition(rules_plan(fade_lanes(), 1.0)).unwrap();
    let mut seen = trickle_until(&gate, &events, |e| {
        matches!(e, Event::Transition(TransitionState::Armed))
    });
    player.move_track(2, 0).unwrap();
    seen.extend(trickle_until(&gate, &events, |e| {
        matches!(e, Event::Transition(TransitionState::Finished))
    }));
    assert!(transitions(&seen).contains(&&TransitionState::Started {
        output_frame: 5 * u64::from(RATE)
    }));
    assert!(seen
        .iter()
        .any(|e| matches!(e, Event::TrackStarted(s) if s.index == 2)));
    drop(player);
    // Again, its next track taken out.
    let (_dir, _b, player, events, _captures, gate) = rules_setup("rules-queue-out", StemMix::ALL);
    player.transition(rules_plan(fade_lanes(), 1.0)).unwrap();
    let mut seen = trickle_until(&gate, &events, |e| {
        matches!(e, Event::Transition(TransitionState::Armed))
    });
    player.remove(1).unwrap();
    seen.extend(trickle_to_end(&gate, &events));
    drop(player);
    assert!(
        transitions(&seen).iter().any(
            |s| matches!(s, TransitionState::Cancelled { reason } if reason.contains("queue"))
        ),
        "{:?}",
        transitions(&seen)
    );
}
