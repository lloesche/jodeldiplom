//! Lane inputs on a simulated device, with a synthetic writer: main is `x`,
//! each of the four parts a quarter of it, and a control word per test.
//! Untouched, the device gets main bit for bit; a part moved switches to
//! the parts' sum from that frame, and back to main bit for bit once the
//! part's ramp has settled; the control word's changes land on their
//! frames; the input's fader scales every lane, and a part's below 1
//! counts as touched; flush, retract and pause act on every lane; the
//! microphone's tap is told the lane stream's frames; on a 16-bit device a
//! part is always mixed and dithered while main still passes through; and
//! `Fader::at_stream` lands a change on its stream frame.

mod common;

use std::time::Duration;

use common::*;
use jodel_output::lanes::{lane_width, Parts};
use jodel_output::mixer::{
    Fader, InputTarget, MicPort, Mixer, PartsView, Ramp, Role, Start, Strip, SWITCH_FADE,
};
use jodel_output::{OpenRequest, Output, OutputTarget, SampleFormat};

const PARTS: usize = 4;
const LANES: u16 = 1 + PARTS as u16;
/// The switch's frames (10 ms) and a strip's ramp's (20 ms).
const SWITCH: usize = 480;
const RAMP: usize = 960;
/// Main, constant, where the tests follow gains.
const X: i32 = 4_000_000;

/// A lane input on the desk: main on Player, a part on each of its four
/// children.
struct Desk {
    mixer: Mixer,
    device: Device,
    player: Strip,
    parts: Vec<Strip>,
    target: InputTarget,
    duck: Fader,
    faders: Vec<Fader>,
}

fn desk_on(device: Device) -> Desk {
    let (mixer, device) = mixer_on(device);
    let player = mixer.strip("Player");
    let parts: Vec<Strip> = ["Drums", "Bass", "Vocals", "Other"]
        .iter()
        .map(|name| mixer.strip_in(&player, name))
        .collect();
    let refs: Vec<&Strip> = parts.iter().collect();
    let (target, duck, faders) = mixer.input_lanes(&player, &refs, Role::Lead, 1.0);
    Desk {
        mixer,
        device,
        player,
        parts,
        target,
        duck,
        faders,
    }
}

fn desk() -> Desk {
    desk_on(Device {
        instant: true,
        ..Device::default()
    })
}

type Main<'a> = &'a dyn Fn(u64) -> i32;
type Control<'a> = &'a dyn Fn(u64) -> Parts;

fn constant(_: u64) -> i32 {
    X
}

fn present(_: u64) -> Parts {
    Parts::Present
}

/// `n` lane frames from stream frame `from`: main's left `main(k)` (its
/// right 0), every part a quarter of main, and the control word
/// `control(k)`.
fn lanes(from: u64, n: u64, main: Main, control: Control) -> Vec<i32> {
    let mut v = Vec::with_capacity(n as usize * lane_width(2, LANES));
    for k in from..from + n {
        let x = main(k);
        v.extend([x, 0]);
        for _ in 0..PARTS {
            v.extend([x / 4, 0]);
        }
        v.push(control(k).word());
    }
    v
}

/// A lane request in `sample`.
fn lane_request(sample: SampleFormat) -> OpenRequest {
    OpenRequest::new(RATE, 2).formats(&[sample]).lanes(LANES)
}

impl Desk {
    /// A lane output running with frame 0 written and no more: short of a
    /// period, the mixer waits for it.
    fn open(&mut self, main: Main, control: Control) -> Box<dyn Output> {
        let request = lane_request(self.device.format()).prefill(Duration::ZERO);
        let mut out = self.target.open(&request).unwrap();
        out.write(&lanes(0, 1, main, control), None).unwrap();
        out
    }

    /// Writes stream frames `from..to` once the mixer has taken what was
    /// set before, and waits until it has mixed up to frame `to`.
    fn play(
        &mut self,
        out: &mut Box<dyn Output>,
        from: u64,
        to: u64,
        main: Main,
        control: Control,
    ) {
        taken(&mut self.target);
        out.write(&lanes(from, to - from, main, control), None)
            .unwrap();
        wait_until("mixed", || self.mixer.timebase().frame == to);
    }

    /// Drains `out`, closes the mixer and gives the left channel of the
    /// device's first opening.
    fn finish(self, mut out: Box<dyn Output>) -> Vec<i32> {
        out.drain(None).unwrap();
        drop(out);
        let Desk { mixer, device, .. } = self;
        drop(mixer);
        device.left(0)
    }
}

/// The first frame of `left[from..to]` not within 1.5 of `want` (by frame).
fn off(left: &[i32], from: usize, to: usize, want: impl Fn(usize) -> f64) -> Option<(usize, i32)> {
    (from..to)
        .find(|&f| (f64::from(left[f]) - want(f)).abs() >= 1.5)
        .map(|f| (f, left[f]))
}

/// The first frame of `left[from..to]` that is not `want(frame)`.
fn wrong(
    left: &[i32],
    from: usize,
    to: usize,
    want: impl Fn(usize) -> i32,
) -> Option<(usize, i32)> {
    (from..to)
        .find(|&f| left[f] != want(f))
        .map(|f| (f, left[f]))
}

/// The period (480 frames) frame `f` falls in starts at.
fn period(f: usize) -> usize {
    f / 480 * 480
}

/// The switch's share `k` frames into it, and a 20 ms ramp's from 1 to 0.
fn rise(k: usize) -> f64 {
    (k as f64 / SWITCH as f64).min(1.0)
}

fn fall(k: usize) -> f64 {
    (1.0 - k as f64 / RAMP as f64).max(0.0)
}

#[test]
fn untouched_the_device_gets_main_bit_for_bit() {
    let mut d = desk();
    assert_eq!(d.target.lanes(), LANES);
    let main = |k: u64| 4 * ((k % 1_000) as i32 * 997 - 400_000);
    let mut out = d.open(&main, &present);
    assert_eq!((out.lanes(), out.format().channels), (LANES, 2));
    out.write(&lanes(1, 47_999, &main, &present), None).unwrap();
    out.drain(None).unwrap();
    let desk = d.mixer.desk();
    assert!(desk.master.passthrough);
    let input = &desk.strips[0].inputs[0];
    let strips: Vec<u64> = input.lanes.iter().map(|l| l.strip).collect();
    let mut want = vec![d.player.id()];
    want.extend(d.parts.iter().map(Strip::id));
    assert_eq!(strips, want);
    assert!(input.lanes[1..].iter().all(|l| !l.heard && l.gain == 0.0));
    let parts = PartsView {
        state: Parts::Present,
        share: 0.0,
    };
    assert_eq!(input.parts, Some(parts));
    let left = d.finish(out);
    assert_eq!(left.len(), 48_000);
    assert_eq!(wrong(&left, 0, 48_000, |f| main(f as u64)), None);
}

#[test]
fn a_part_moved_switches_to_the_parts_and_back_to_main_bit_for_bit() {
    let mut d = desk();
    let mut out = d.open(&constant, &present);
    d.play(&mut out, 1, 4_800, &constant, &present);
    // The drums muted at 4 800: the switch from that frame, the drums
    // down over 20 ms.
    d.parts[0].set_mute(true);
    d.play(&mut out, 4_800, 9_600, &constant, &present);
    assert!(!d.mixer.passthrough());
    let share = d.mixer.desk().strips[0].inputs[0].parts.unwrap().share;
    assert_eq!(share, 1.0);
    // Unmuted at 9 600: the drums back up; the switch holds until they
    // have settled, then returns to main.
    d.parts[0].set_mute(false);
    d.play(&mut out, 9_600, 14_400, &constant, &present);
    assert!(d.mixer.passthrough());
    let left = d.finish(out);
    let x = f64::from(X);
    assert_eq!(wrong(&left, 0, 4_800, |_| X), None);
    let switched = |f: usize| {
        let (s, g) = (rise(f - 4_800), fall(f - 4_800));
        x * (1.0 - s) + s * (0.75 * x + g * x / 4.0)
    };
    assert_eq!(off(&left, 4_800, 9_600, switched), None);
    let back = |f: usize| 0.75 * x + (f - 9_600).min(RAMP) as f64 / RAMP as f64 * x / 4.0;
    assert_eq!(off(&left, 9_600, 11_520, back), None);
    // Settled by the end of the period after the ramp (10 560..11 040),
    // the switch back over 10 ms from 11 040: main as written from 11 520.
    assert_eq!(wrong(&left, 11_520, 14_400, |_| X), None);
}

#[test]
fn a_part_at_another_level_is_heard_at_it() {
    let mut d = desk();
    let mut out = d.open(&constant, &present);
    d.play(&mut out, 1, 4_800, &constant, &present);
    d.parts[1].set_level(0.5);
    d.play(&mut out, 4_800, 9_600, &constant, &present);
    let left = d.finish(out);
    assert_eq!(off(&left, 4_800 + RAMP, 9_600, |_| 3_500_000.0), None);
}

#[test]
fn the_control_word_switches_on_its_frames() {
    let mut d = desk();
    // The parts from 5 123, ending from 9 777; the drums muted throughout.
    let (on, end) = (5_123usize, 9_777usize);
    let control = move |k: u64| match k as usize {
        k if k < on => Parts::None,
        k if k < end => Parts::Present,
        _ => Parts::Ending,
    };
    d.parts[0].set_mute(true);
    let mut out = d.open(&constant, &control);
    d.play(&mut out, 1, 4_800, &constant, &control);
    d.play(&mut out, 4_800, 14_400, &constant, &control);
    assert!(d.mixer.passthrough());
    let left = d.finish(out);
    let x = f64::from(X);
    // No parts: main, as written, however touched.
    assert_eq!(wrong(&left, 0, period(on), |_| X), None);
    assert_eq!(off(&left, period(on), on, |_| x), None);
    // From `on`, the switch to the parts (the drums silent).
    let up = |f: usize| x - rise(f - on) * x / 4.0;
    assert_eq!(off(&left, on, end, up), None);
    // From `end`, back to main over the switch; as written from the
    // period after.
    let down = |f: usize| 0.75 * x + rise(f - end) * x / 4.0;
    assert_eq!(off(&left, end, end + SWITCH, down), None);
    let after = period(end + SWITCH) + 480;
    assert_eq!(off(&left, end + SWITCH, after, |_| x), None);
    assert_eq!(wrong(&left, after, 14_400, |_| X), None);
    assert_eq!(SWITCH_FADE, Duration::from_millis(10));
}

#[test]
fn the_inputs_fader_scales_every_lane() {
    let mut d = desk();
    let mut out = d.open(&constant, &present);
    d.play(&mut out, 1, 4_800, &constant, &present);
    d.duck.set(Ramp::set(0.5));
    d.play(&mut out, 4_800, 9_600, &constant, &present);
    // Switched to the parts too.
    d.parts[0].set_mute(true);
    d.play(&mut out, 9_600, 14_400, &constant, &present);
    assert_eq!(d.duck.gain(), 0.5);
    let left = d.finish(out);
    assert_eq!(off(&left, 4_800, 9_600, |_| 2_000_000.0), None);
    assert_eq!(off(&left, 9_600 + RAMP, 14_400, |_| 1_500_000.0), None);
}

#[test]
fn a_parts_fader_below_1_counts_as_touched() {
    let mut d = desk();
    let mut out = d.open(&constant, &present);
    d.play(&mut out, 1, 4_800, &constant, &present);
    d.faders[1].set(Ramp::set(0.5));
    d.play(&mut out, 4_800, 9_600, &constant, &present);
    assert!(d.faders[1].settled() && d.faders[1].gain() == 0.5);
    let input = d.mixer.desk().strips[0].inputs[0].clone();
    assert_eq!(input.lanes[2].gain, 0.5);
    assert_eq!(input.lanes[0].gain, 0.0);
    // Back at 1: back to main, as written two periods on.
    d.faders[1].set(Ramp::set(1.0));
    d.play(&mut out, 9_600, 14_400, &constant, &present);
    assert!(d.mixer.passthrough());
    let left = d.finish(out);
    let x = f64::from(X);
    let switched = |f: usize| x - rise(f - 4_800) * x / 8.0;
    assert_eq!(off(&left, 4_800, 9_600, switched), None);
    assert_eq!(off(&left, 9_600, 9_600 + 2 * SWITCH, |_| x), None);
    assert_eq!(wrong(&left, 9_600 + 2 * SWITCH, 14_400, |_| X), None);
}

#[test]
fn a_flush_on_a_lane_stream_drops_what_the_device_holds_of_it() {
    let mut d = desk_on(Device::default());
    let mut out = d.target.open(&lane_request(FORMAT)).unwrap();
    let (a, b) = (|_: u64| 444, |_: u64| 888);
    out.write(&lanes(0, 24_000, &a, &present), None).unwrap();
    wait_until("it plays", || d.mixer.timebase().heard > 9_600);
    out.flush().unwrap();
    let flushed = d.mixer.timebase().heard as usize;
    assert_eq!(out.position().unwrap(), out.written());
    out.write(&lanes(0, 9_600, &b, &present), None).unwrap();
    let left = d.finish(out);
    let last = left.iter().rposition(|&v| v == 444).unwrap();
    assert!(last < flushed + 2 * 480, "{last} {flushed}");
    let first = left.iter().position(|&v| v == 888).unwrap();
    assert!(first > last);
    assert_eq!(left[first..].iter().filter(|&&v| v == 888).count(), 9_600);
}

#[test]
fn a_retract_on_a_lane_stream_takes_back_every_lane() {
    let mut d = desk_on(Device::default());
    let mut out = d.target.open(&lane_request(FORMAT)).unwrap();
    let (a, b) = (|_: u64| 444, |_: u64| 888);
    out.write(&lanes(0, 20_000, &a, &present), None).unwrap();
    wait_until("it plays", || d.mixer.timebase().heard > 4_800);
    let back = out.retract(12_000).unwrap();
    assert!(back > 0);
    assert_eq!(out.written(), 20_000 - back);
    out.write(&lanes(0, 4_800, &b, &present), None).unwrap();
    let left = d.finish(out);
    let count = |v: i32| left.iter().filter(|&&s| s == v).count() as u64;
    assert_eq!((count(444), count(888)), (20_000 - back, 4_800));
    let last = left.iter().rposition(|&v| v == 444).unwrap();
    let first = left.iter().position(|&v| v == 888).unwrap();
    assert!(last < first);
}

#[test]
fn a_paused_lane_stream_goes_on_where_it_was() {
    let mut d = desk_on(Device::default());
    let mut out = d.target.open(&lane_request(FORMAT)).unwrap();
    let main = |k: u64| 4 * (k as i32 + 1);
    out.write(&lanes(0, 9_600, &main, &present), None).unwrap();
    wait_until("it plays", || d.mixer.timebase().heard > 4_800);
    out.pause().unwrap();
    std::thread::sleep(Duration::from_millis(50));
    out.resume().unwrap();
    out.write(&lanes(9_600, 9_600, &main, &present), None)
        .unwrap();
    let left = d.finish(out);
    let heard: Vec<i32> = left.into_iter().filter(|&v| v != 0).collect();
    let want: Vec<i32> = (0..19_200).map(main).collect();
    assert_eq!(heard, want);
}

#[test]
fn the_microphone_is_told_a_lane_streams_frames() {
    let mut d = desk_on(Device::default());
    let singer = Singer::new(loud);
    d.mixer.set_mic(Some(singer.input.clone()));
    // A guest plays (silent) first, so the device is ahead of the Lead.
    let preview = d.mixer.strip("Preview");
    let (mut guest, _guest_fader) = d.mixer.input(&preview, Role::Guest, 0.0);
    let mut g = open(&mut guest, RATE);
    g.write(&frames(0, 9_600, |k| k as i32), None).unwrap();
    wait_until("the guest plays", || d.mixer.timebase().heard > 9_600);
    let mut out = d.target.open(&lane_request(FORMAT)).unwrap();
    let total = u64::from(RATE);
    out.write(&lanes(0, total, &constant, &present), None)
        .unwrap();
    out.drain(None).unwrap();
    assert_eq!(d.mixer.mic().port, MicPort::Hearing);
    let voice = singer.voice();
    assert!(voice.len() > 100, "{} blocks", voice.len());
    // Told in the lane stream's frames: the device's run 9 600 and more
    // ahead.
    let last = voice.iter().map(|b| b.stream_frame).max().unwrap();
    assert!(last <= total && last > total - 4_800, "{last}");
    let within =
        |w: &&[jodel_output::VoiceBlock]| w[0].stream_frame > 0 && w[1].stream_frame < total - 128;
    for w in voice.windows(2).filter(|w| w[1].continues).filter(within) {
        assert_eq!(w[1].stream_frame, w[0].stream_frame + 128);
    }
    d.mixer.set_mic(None);
}

#[test]
fn on_a_16_bit_device_a_part_is_mixed_and_main_passes_through() {
    let mut d = desk_on(Device {
        instant: true,
        sample: Some(SampleFormat::S16),
        ..Device::default()
    });
    // 16-bit content (and so is a quarter of it).
    let main = |k: u64| 1_024 * ((k % 500) as i32 - 250);
    let mut out = d.open(&main, &present);
    d.play(&mut out, 1, 4_800, &main, &present);
    assert!(d.mixer.passthrough());
    // The vocals soloed: they alone, mixed and dithered at 16 bits.
    d.parts[2].set_solo(true);
    d.play(&mut out, 4_800, 14_400, &main, &present);
    assert!(!d.mixer.passthrough());
    let left = d.finish(out);
    assert_eq!(wrong(&left, 0, 4_800, |f| main(f as u64)), None);
    let solo = &left[4_800 + RAMP..14_400];
    assert!(solo.iter().all(|&v| v & 0xFF == 0), "16-bit samples");
    let quarter = |f: usize| main((4_800 + RAMP + f) as u64) / 4;
    assert!(solo
        .iter()
        .enumerate()
        .all(|(f, &v)| (v - quarter(f)).abs() <= 256));
    let dithered = solo
        .iter()
        .enumerate()
        .filter(|&(f, &v)| v != quarter(f))
        .count();
    assert!(dithered > 100, "{dithered} of {} dithered", solo.len());
}

#[test]
fn at_stream_lands_a_change_on_its_stream_frame() {
    let mut d = desk();
    // A silent guest first, so the lane stream's frames are 4 800 behind
    // the device's.
    let preview = d.mixer.strip("Preview");
    let (mut guest, _guest_fader) = d.mixer.input(&preview, Role::Guest, 0.0);
    let mut g = open_started(&mut guest, RATE, 0);
    g.write(&frames(1, 4_799, |_| 0), None).unwrap();
    wait_until("mixed", || d.mixer.timebase().frame == 4_800);
    assert_eq!(d.duck.at_stream(100), Start::Now);
    let mut out = d.open(&constant, &present);
    g.write(&frames(0, 4_800, |_| 0), None).unwrap();
    out.write(&lanes(1, 4_799, &constant, &present), None)
        .unwrap();
    wait_until("mixed", || d.mixer.timebase().frame == 9_600);
    let at = 6_123usize;
    let start = d.duck.at_stream(at as u64);
    assert_eq!(start, d.mixer.timebase().at(at as u64 + 4_800));
    d.player.set_mute_at(true, start);
    taken(&mut d.target);
    g.write(&frames(0, 9_600, |_| 0), None).unwrap();
    out.write(&lanes(4_800, 9_600, &constant, &present), None)
        .unwrap();
    wait_until("mixed", || d.mixer.timebase().frame == 19_200);
    drop(g);
    let left = d.finish(out);
    let device = at + 4_800;
    assert_eq!(wrong(&left, 4_800, period(device), |_| X), None);
    let x = f64::from(X);
    assert_eq!(off(&left, period(device), device, |_| x), None);
    assert_eq!(
        off(&left, device, device + RAMP, |f| x * fall(f - device)),
        None
    );
    let silent = period(device + RAMP) + 480;
    assert_eq!(wrong(&left, silent, 19_200, |_| 0), None);
}

/// Every frame of `left` that is not silence, from `from` on, within 2 of
/// three quarters of `X` (the drums muted): the first that is not.
fn not_muted_drums(left: &[i32], from: usize) -> Option<(usize, i32)> {
    let want = 0.75 * f64::from(X);
    (from..left.len())
        .filter(|&f| left[f] != 0)
        .find(|&f| (f64::from(left[f]) - want).abs() > 2.0)
        .map(|f| (f, left[f]))
}

#[test]
fn touched_from_the_first_frame_no_main_is_heard() {
    let mut d = desk();
    d.parts[0].set_mute(true);
    let mut out = d.open(&constant, &present);
    d.play(&mut out, 1, 4_800, &constant, &present);
    let left = d.finish(out);
    assert_eq!(not_muted_drums(&left, 0), None);
    assert_eq!(left.iter().filter(|&&v| v == 0).count(), 0);
}

#[test]
fn touched_across_a_pause_a_flush_and_a_new_output_no_main_is_heard() {
    let mut d = desk_on(Device::default());
    d.parts[0].set_mute(true);
    let mut out = d.target.open(&lane_request(FORMAT)).unwrap();
    out.write(&lanes(0, 9_600, &constant, &present), None)
        .unwrap();
    wait_until("it plays", || d.mixer.timebase().heard > 4_800);
    // Paused and resumed.
    out.pause().unwrap();
    std::thread::sleep(Duration::from_millis(50));
    out.resume().unwrap();
    out.write(&lanes(9_600, 9_600, &constant, &present), None)
        .unwrap();
    wait_until("it plays on", || d.mixer.timebase().heard > 14_400);
    // Flushed (a seek).
    out.flush().unwrap();
    let w = out.written();
    out.write(&lanes(w, 9_600, &constant, &present), None)
        .unwrap();
    out.drain(None).unwrap();
    // A new output (the next track) on the input.
    let mut next = d.target.open(&lane_request(FORMAT)).unwrap();
    next.write(&lanes(0, 9_600, &constant, &present), None)
        .unwrap();
    next.drain(None).unwrap();
    drop((out, next));
    let Desk { mixer, device, .. } = d;
    drop(mixer);
    // (The new output, the input's only one, opened the device again.)
    let streams = device.streams.lock().unwrap().len();
    for i in 0..streams {
        let left = device.left(i);
        assert_eq!(not_muted_drums(&left, 0), None, "opening {i}");
    }
    let heard: usize = (0..streams)
        .map(|i| device.left(i).iter().filter(|&&v| v != 0).count())
        .sum();
    // At least all of what was written after the flush and to the new
    // output (the flush dropped what was queued before it).
    assert!(heard >= 19_200, "{heard} frames heard");
}

#[test]
fn an_underrun_in_the_middle_of_the_switch_comes_back_switched() {
    let mut d = desk();
    // A silent guest keeps the periods coming while the lane stream is dry.
    let preview = d.mixer.strip("Preview");
    let (mut guest, _guest_fader) = d.mixer.input(&preview, Role::Guest, 0.0);
    let mut g = open_started(&mut guest, RATE, 0);
    let mut out = d.open(&constant, &present);
    g.write(&frames(1, 4_799, |_| 0), None).unwrap();
    d.play(&mut out, 1, 4_800, &constant, &present);
    // The drums muted at 4 800; the lane stream gives 100 frames and runs
    // dry, 380 frames before the end of the switch's 480.
    d.parts[0].set_mute(true);
    taken(&mut d.target);
    let writer = std::thread::spawn(move || {
        out.write(&lanes(4_800, 100, &constant, &present), None)
            .unwrap();
        out.drain(None).unwrap();
        out
    });
    g.write(&frames(0, 4_800, |_| 0), None).unwrap();
    wait_until("mixed", || d.mixer.timebase().frame == 9_600);
    let mut out = writer.join().unwrap();
    // Back: switched at once, no crossfade from main.
    g.write(&frames(0, 4_800, |_| 0), None).unwrap();
    out.write(&lanes(4_900, 4_800, &constant, &present), None)
        .unwrap();
    wait_until("mixed", || d.mixer.timebase().frame == 14_400);
    drop(g);
    let left = d.finish(out);
    let x = f64::from(X);
    let switching = |f: usize| {
        let (s, g) = (rise(f - 4_800), fall(f - 4_800));
        x * (1.0 - s) + s * (0.75 * x + g * x / 4.0)
    };
    assert_eq!(off(&left, 4_800, 4_900, switching), None);
    assert_eq!(off(&left, 4_900, 5_280, |_| 0.0), None);
    assert_eq!(wrong(&left, 5_280, 9_600, |_| 0), None);
    assert_eq!(off(&left, 9_600, 14_400, |_| 0.75 * x), None);
}

#[test]
fn ending_then_present_again_switches_back_in_from_that_frame() {
    let mut d = desk();
    let (end, again) = (5_123usize, 7_777usize);
    let control = move |k: u64| match k as usize {
        k if k < end => Parts::Present,
        k if k < again => Parts::Ending,
        _ => Parts::Present,
    };
    d.parts[0].set_mute(true);
    let mut out = d.open(&constant, &control);
    d.play(&mut out, 1, 4_800, &constant, &control);
    d.play(&mut out, 4_800, 14_400, &constant, &control);
    let left = d.finish(out);
    let x = f64::from(X);
    assert_eq!(off(&left, 0, end, |_| 0.75 * x), None);
    let down = |f: usize| 0.75 * x + rise(f - end) * x / 4.0;
    assert_eq!(off(&left, end, end + SWITCH, down), None);
    assert_eq!(off(&left, end + SWITCH, again, |_| x), None);
    let up = |f: usize| x - rise(f - again) * x / 4.0;
    assert_eq!(off(&left, again, 14_400, up), None);
}

#[test]
fn on_a_16_bit_device_main_is_bit_for_bit_again_once_a_solo_is_released() {
    let mut d = desk_on(Device {
        instant: true,
        sample: Some(SampleFormat::S16),
        ..Device::default()
    });
    let main = |k: u64| 1_024 * ((k % 500) as i32 - 250);
    let mut out = d.open(&main, &present);
    d.play(&mut out, 1, 4_800, &main, &present);
    d.parts[2].set_solo(true);
    d.play(&mut out, 4_800, 9_600, &main, &present);
    d.parts[2].set_solo(false);
    d.play(&mut out, 9_600, 14_400, &main, &present);
    assert!(d.mixer.passthrough());
    let left = d.finish(out);
    // The gates back over 20 ms (settled by the end of the period after),
    // then the switch back over 10 ms: main as written from 11 520.
    assert_eq!(wrong(&left, 11_520, 14_400, |f| main(f as u64)), None);
    assert!(left[9_600..11_520].iter().all(|&v| v & 0xFF == 0));
}

/// `n` frames of a stream of `parts` parts at `channels` channels: main's
/// first channel `x` (the others 0), each part's `x / parts`, the word
/// `Present`.
fn wide(channels: u16, parts: usize, n: u64, x: i32) -> Vec<i32> {
    let ch = usize::from(channels);
    let mut v = Vec::new();
    for _ in 0..n {
        for lane in 0..=parts {
            let value = if lane == 0 { x } else { x / parts as i32 };
            v.push(value);
            v.extend(std::iter::repeat_n(0, ch - 1));
        }
        v.push(Parts::Present.word());
    }
    v
}

/// A lane input of `parts` parts on a device of `channels` channels:
/// untouched, main bit for bit; the first part muted, the others' sum.
fn lanes_at(channels: u16, parts: usize) {
    let device = Device {
        instant: true,
        channels: Some(channels),
        ..Device::default()
    };
    let (mixer, device) = mixer_on(device);
    let player = mixer.strip("Player");
    let strips: Vec<Strip> = (0..parts)
        .map(|k| mixer.strip_in(&player, &format!("Part {k}")))
        .collect();
    let refs: Vec<&Strip> = strips.iter().collect();
    let (mut target, _duck, _faders) = mixer.input_lanes(&player, &refs, Role::Lead, 1.0);
    let lanes = 1 + parts as u16;
    assert_eq!(target.lanes(), lanes);
    let width = usize::from(channels) * (1 + parts) + 1;
    assert_eq!(lane_width(channels, lanes), width);
    let request = OpenRequest::new(RATE, channels)
        .formats(&[FORMAT])
        .lanes(lanes)
        .prefill(Duration::ZERO);
    let mut out = target.open(&request).unwrap();
    let x = 1_200_000;
    out.write(&wide(channels, parts, 4_800, x), None).unwrap();
    wait_until("mixed", || mixer.timebase().frame == 4_800);
    assert!(mixer.passthrough());
    strips[0].set_mute(true);
    taken(&mut target);
    out.write(&wide(channels, parts, 4_800, x), None).unwrap();
    out.drain(None).unwrap();
    drop(out);
    drop(mixer);
    let left = device.left(0);
    assert_eq!(left.len(), 9_600);
    assert_eq!(wrong(&left, 0, 4_800, |_| x), None, "{channels} {parts}");
    let rest = f64::from(x) * (parts - 1) as f64 / parts as f64;
    let after = off(&left, 4_800 + RAMP, 9_600, |_| rest);
    assert_eq!(after, None, "{channels} {parts}");
}

#[test]
fn mono_and_odd_lane_widths_mix_as_wide_ones() {
    lanes_at(1, 4);
    lanes_at(2, 2);
    lanes_at(1, 1);
    lanes_at(3, 3);
}

#[test]
fn the_parts_strips_meter_the_parts_while_main_is_heard() {
    let mut d = desk();
    let mut out = d.open(&constant, &present);
    d.play(&mut out, 1, 4_800, &constant, &present);
    let meters = d.mixer.meters();
    let full = 8_388_608.0f32;
    let peak = |id: u64| meters.strips.iter().find(|s| s.0 == id).unwrap().1[0];
    // Main alone on the group, not main and the parts.
    assert_eq!(peak(d.player.id()), X as f32 / full);
    for part in &d.parts {
        assert_eq!(peak(part.id()), (X / 4) as f32 / full);
    }
    assert!(d.mixer.passthrough());
    drop(d.finish(out));
}

#[test]
fn lanes_are_asked_of_targets_that_take_them() {
    let caps = [jodel_output::StreamFormat::new(RATE, 2, FORMAT)]
        .into_iter()
        .collect();
    let lanes = OpenRequest::new(RATE, 2).formats(&[FORMAT]).lanes(LANES);
    let sink = jodel_output::MemorySink::default();
    let clock = jodel_output::SimClock::Instant;
    let refused = jodel_output::FileOutput::open(sink.clone(), &caps, &lanes, clock);
    assert!(matches!(
        refused,
        Err(jodel_output::Error::InvalidArgument(_))
    ));
    let wide = jodel_output::FileOutput::open_lanes(sink, &caps, &lanes, clock).unwrap();
    assert_eq!(wide.lanes(), LANES);
    // A plain input takes one lane; a lane input one or all of them.
    let mut d = desk();
    let other = d.mixer.strip("Other");
    let (mut plain, _fader) = d.mixer.input(&other, Role::Guest, 1.0);
    assert_eq!(plain.lanes(), 1);
    assert!(matches!(
        plain.open(&lane_request(FORMAT).lanes(2)),
        Err(jodel_output::Error::InvalidArgument(_))
    ));
    assert!(matches!(
        d.target.open(&lane_request(FORMAT).lanes(3)),
        Err(jodel_output::Error::InvalidArgument(_))
    ));
    let one = d.target.open(&lane_request(FORMAT).lanes(1)).unwrap();
    assert_eq!(one.lanes(), 1);
}

#[test]
#[should_panic(expected = "a part's strip is a child of main's")]
fn a_parts_strip_is_a_child_of_mains() {
    let (mixer, _device) = instant_mixer();
    let (main, stray) = (mixer.strip("Player"), mixer.strip("Stray"));
    let _ = mixer.input_lanes(&main, &[&stray], Role::Lead, 1.0);
}

#[test]
fn after_a_flush_at_stream_waits_for_frames_mixed_again() {
    let mut d = desk();
    let mut out = d.open(&constant, &present);
    d.play(&mut out, 1, 4_800, &constant, &present);
    assert_ne!(d.duck.at_stream(5_000), Start::Now);
    out.flush().unwrap();
    assert_eq!(d.duck.at_stream(5_000), Start::Now);
    let w = out.written();
    out.write(&lanes(w, 960, &constant, &present), None)
        .unwrap();
    wait_until("mixed", || d.mixer.timebase().frame == 5_760);
    // Its frame `w` went to device frame 4 800.
    let start = d.duck.at_stream(w + 100);
    assert_eq!(start, d.mixer.timebase().at(4_900));
    drop(d.finish(out));
}

#[test]
fn changes_set_before_the_device_opens_are_in_force_from_its_first_frame() {
    let mut d = desk();
    // A level and a fade due now, set while nothing is open: no ramp at
    // the first audio.
    d.player.set_level(0.5);
    d.duck.set(Ramp {
        start: Start::Now,
        to: 0.5,
        over: Duration::from_millis(20),
        curve: jodel_output::mixer::Curve::Linear,
    });
    let mut out = d.open(&constant, &present);
    d.play(&mut out, 1, 4_800, &constant, &present);
    let left = d.finish(out);
    assert_eq!(off(&left, 0, 4_800, |_| f64::from(X) / 4.0), None);
}

#[test]
fn a_part_soloed_keeps_main_while_main_stands_for_the_parts() {
    let mut d = desk();
    // The vocals soloed; no parts until 5 123, then they are there.
    let on = 5_123usize;
    let control = move |k: u64| {
        if (k as usize) < on {
            Parts::None
        } else {
            Parts::Present
        }
    };
    d.parts[2].set_solo(true);
    let mut out = d.open(&constant, &control);
    d.play(&mut out, 1, 4_800, &constant, &control);
    d.play(&mut out, 4_800, 14_400, &constant, &control);
    let left = d.finish(out);
    let x = f64::from(X);
    // Main heard in the parts' place: the song, not silence.
    assert_eq!(wrong(&left, 0, period(on), |_| X), None);
    // Once the parts are there, the vocals alone (a quarter of main).
    let up = |f: usize| x - rise(f - on) * 0.75 * x;
    assert_eq!(off(&left, on, on + SWITCH, up), None);
    assert_eq!(off(&left, on + SWITCH, 14_400, |_| x / 4.0), None);
}
