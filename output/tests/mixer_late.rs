//! Inputs that lag on a device in real time ([`jodel_output::mixer`]'s
//! pacing): one late by less than the device's ring holds beyond its low
//! water loses nothing, a later one is silence until it is back (and is
//! waited for again once it holds a period), another plays on meanwhile,
//! and on a device with a buffer of its own (hardware's) the wait never
//! runs the ring dry.

mod common;

use std::time::{Duration, Instant};

use common::*;
use jodel_output::mixer::{Mixer, MixerOptions, Role};

/// The frames of `left` that are not silence, in order, and the runs of
/// silence between the first and the last of them (where, how long).
fn heard_and_gaps(left: &[i32]) -> (Vec<i32>, Vec<(usize, usize)>) {
    let first = left.iter().position(|&v| v != 0).unwrap_or(left.len());
    let last = left.iter().rposition(|&v| v != 0).map_or(first, |l| l + 1);
    let mut gaps = Vec::new();
    let mut at = None;
    for (f, &v) in left.iter().enumerate().take(last).skip(first) {
        match (v == 0, at) {
            (true, None) => at = Some(f),
            (false, Some(from)) => {
                gaps.push((from, f - from));
                at = None;
            }
            _ => {}
        }
    }
    let heard = left[first..last]
        .iter()
        .copied()
        .filter(|&v| v != 0)
        .collect();
    (heard, gaps)
}

/// Frames `from..from + n` of a counting stream: frame k is k + 1 (left),
/// so silence stands out.
fn counting(from: u64, n: u64) -> Vec<i32> {
    frames(from, n, |k| k as i32 + 1)
}

/// An output whose writer is late by less than the device's ring holds:
/// on a device in real time the mix waits for it, and nothing is lost.
#[test]
fn a_briefly_lagging_input_on_a_real_time_device_leaves_no_gap() {
    let _guard = real_time();
    let (mixer, device) = mixer();
    let strip = mixer.strip("Player");
    let (mut lead, _fader) = mixer.input(&strip, Role::Lead, 1.0);
    let mut out = open(&mut lead, RATE);
    // Each 100 ms goes into the device's ring (200 ms) at once and the
    // writer is away for 40 ms: its own ring is empty while the device's
    // still holds far more than that.
    let chunk = 4_800;
    for c in 0..4 {
        out.write(&counting(c * chunk, chunk), None).unwrap();
        std::thread::sleep(Duration::from_millis(40));
    }
    out.drain(None).unwrap();
    let late = out.stats().underruns;
    drop(out);
    drop(mixer);
    let (heard, gaps) = heard_and_gaps(&device.left(0));
    assert_eq!(gaps, [], "{late} underruns of the output");
    let want: Vec<i32> = (1..=4 * chunk as i32).collect();
    assert!(heard == want, "{} frames heard", heard.len());
    assert_eq!(device.underruns(), 0);
}

/// Later than the device can wait: the output is silence until it is back,
/// its frames then go on in order, none lost; and once it holds a period
/// again it is waited for again (a brief lateness after costs nothing).
#[test]
fn an_input_later_than_the_device_can_wait_is_silence_until_back_then_waited_for_again() {
    let _guard = real_time();
    let (mixer, device) = mixer();
    let strip = mixer.strip("Player");
    let (mut lead, _fader) = mixer.input(&strip, Role::Lead, 1.0);
    let mut out = open(&mut lead, RATE);
    let chunk = 4_800;
    out.write(&counting(0, chunk), None).unwrap();
    std::thread::sleep(Duration::from_millis(300));
    out.write(&counting(chunk, chunk), None).unwrap();
    std::thread::sleep(Duration::from_millis(40));
    out.write(&counting(2 * chunk, chunk), None).unwrap();
    out.drain(None).unwrap();
    drop(out);
    drop(mixer);
    let (heard, gaps) = heard_and_gaps(&device.left(0));
    let want: Vec<i32> = (1..=3 * chunk as i32).collect();
    assert!(heard == want, "{} frames heard", heard.len());
    // One gap: the long absence; the short one after went unheard.
    assert_eq!(gaps.len(), 1, "{gaps:?}");
    assert_eq!(device.underruns(), 0);
}

/// While one output is late, another plays on, bit for bit: the mix waits
/// only while the device has frames to spare, which it plays meanwhile.
#[test]
fn another_input_plays_on_while_one_is_late() {
    let _guard = real_time();
    let (mixer, device) = mixer();
    played_on_while_a_guest_is_away(mixer, &device, 300);
}

/// The same on a device with a buffer of its own (hardware's: four
/// periods taken from the ring ahead of what it plays) and the mixer's own
/// short ring (40 ms): what is written and not played is more than what
/// the ring has left, and only the ring counts for the wait. A guest that
/// gives a period every 50 ms lags again and again (each time it holds a
/// period it is waited for anew), longer than the ring lasts: the device
/// never runs dry, and the song is every frame of it, in order. (Counting
/// the device's own buffer, the wait would run the ring dry every time.)
#[test]
fn a_device_with_a_buffer_of_its_own_never_runs_dry_for_a_lagging_input() {
    let _guard = real_time();
    let period = u64::from(RATE / 100);
    let device = Device {
        buffered: Some(4 * period),
        ..Device::default()
    };
    let mixer = Mixer::open(Box::new(device.clone()), MixerOptions::default());
    let strip = mixer.strip("Player");
    let (mut lead, _lead_fader) = mixer.input(&strip, Role::Lead, 1.0);
    let (mut guest, _guest_fader) = mixer.input(&strip, Role::Guest, 0.0);
    let mut out = open(&mut lead, RATE);
    let total = 2 * u64::from(RATE);
    let writer = std::thread::spawn(move || {
        out.write(&counting(0, total), None).unwrap();
        out.drain(None).unwrap();
    });
    wait_until("the song plays", || mixer.timebase().heard > 4_800);
    let mut g = open(&mut guest, RATE);
    g.write(&frames(0, 4_800, |_| 0), None).unwrap();
    for k in 0..16 {
        std::thread::sleep(Duration::from_millis(50));
        g.write(&frames(4_800 + k * period, period, |_| 0), None)
            .unwrap();
    }
    g.drain(None).unwrap();
    writer.join().unwrap();
    drop(g);
    drop(mixer);
    assert_eq!(device.underruns(), 0);
    let (heard, gaps) = heard_and_gaps(&device.left(0));
    assert_eq!(gaps, []);
    let want: Vec<i32> = (1..=total as i32).collect();
    assert!(heard == want, "{} frames heard", heard.len());
}

/// A song on the Lead, two seconds written at once, and a guest at 0 (the
/// Lead alone heard, bit for bit) that gives 100 ms, is away for `away`
/// ms without draining, and comes back: the song plays on at the device's
/// pace meanwhile, every frame of it in order, and the device never runs
/// dry.
fn played_on_while_a_guest_is_away(mixer: Mixer, device: &Device, away: u64) {
    let strip = mixer.strip("Player");
    let (mut lead, _lead_fader) = mixer.input(&strip, Role::Lead, 1.0);
    let (mut guest, _guest_fader) = mixer.input(&strip, Role::Guest, 0.0);
    let mut out = open(&mut lead, RATE);
    let clock = out.clock();
    let total = 2 * u64::from(RATE);
    let writer = std::thread::spawn(move || {
        out.write(&counting(0, total), None).unwrap();
        out.drain(None).unwrap();
    });
    wait_until("the song plays", || mixer.timebase().heard > 4_800);
    let mut g = open(&mut guest, RATE);
    g.write(&frames(0, 4_800, |_| 0), None).unwrap();
    wait_until("the guest is dry", || g.position().unwrap() >= 4_000);
    let (from, at) = (clock.position(), Instant::now());
    std::thread::sleep(Duration::from_millis(away));
    let played = clock.position() - from;
    let wall = at.elapsed();
    g.write(&frames(4_800, 4_800, |_| 0), None).unwrap();
    g.drain(None).unwrap();
    writer.join().unwrap();
    drop(g);
    drop(mixer);
    let due = wall.as_secs_f64() * f64::from(RATE);
    assert!(
        played as f64 > due - f64::from(RATE) / 20.0,
        "{played} frames of the song played in {wall:?}"
    );
    let (heard, gaps) = heard_and_gaps(&device.left(0));
    assert_eq!(gaps, []);
    let want: Vec<i32> = (1..=total as i32).collect();
    assert!(heard == want, "{} frames heard", heard.len());
    assert_eq!(device.underruns(), 0);
}
