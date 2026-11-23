//! Automation and roles on a simulated device. A curve lands on its
//! stream frames, frame by frame: on a plain input (passthrough back once
//! it ends at 1), across a pause that parts its stream frames from the
//! device's, and on a follower across its leader's underrun and pause (the
//! device running on). Replaced or cleared mid-curve, each gain goes
//! linearly from where it was heard, with no step larger than that
//! smoothing's. A set ends with its output, and one given with none open
//! is the next one's. On a lane input the parts' curves shape each part
//! while the parts are there and main's curve shapes main while they are
//! not; a part's curve through exactly 1 does not let the switch go, the
//! input's own curve moving does not hold it, and curves that end at 1
//! bring main back bit for bit. A guest made Lead takes the microphone's
//! tap and is asked again what it takes, with no new opening of the
//! device; the only Lead playing cannot be made a guest. (What the mixer
//! thread allocates is counted in `automation_alloc.rs`.)

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use common::*;
use jodel_output::lanes::{lane_width, Parts};
use jodel_output::mixer::{
    equal_power, Automation, AutomationView, Fader, GainCurve, InputTarget, MicPort, Mixer, Role,
};
use jodel_output::pack::unpack;
use jodel_output::{Error, OpenRequest, Output, OutputTarget};

/// A level that reads back exactly and leaves room under full scale.
const L: i32 = 2_000_000;
const PARTS: usize = 4;
const LANES: u16 = 1 + PARTS as u16;
/// Main, and its parts (which add up to it).
const X: i32 = 4_000_000;
const P: [i32; PARTS] = [X / 2, X / 4, X / 8, X / 8];
/// [`jodel_output::mixer::CUT_FADE`] at [`RATE`].
const FADE: usize = 480;

fn curve(points: &[(u64, f64)]) -> GainCurve {
    GainCurve::new(points).unwrap()
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// The first frame of `got[from..to]` not within 1.5 of `want` (by frame).
fn off(got: &[i32], from: usize, to: usize, want: impl Fn(usize) -> f64) -> Option<(usize, i32)> {
    (from..to)
        .find(|&f| (f64::from(got[f]) - want(f)).abs() >= 1.5)
        .map(|f| (f, got[f]))
}

/// The first frame of `got[from..to]` that is not `want(frame)`.
fn wrong(got: &[i32], from: usize, to: usize, want: impl Fn(usize) -> i32) -> Option<(usize, i32)> {
    (from..to).find(|&f| got[f] != want(f)).map(|f| (f, got[f]))
}

/// The largest step between frames of `got[from..to]`.
fn largest_step(got: &[i32], from: usize, to: usize) -> f64 {
    got[from..to]
        .windows(2)
        .map(|w| (f64::from(w[1]) - f64::from(w[0])).abs())
        .fold(0.0, f64::max)
}

/// Once the mixer has taken what was set before, writes `data` and waits
/// until it has mixed up to frame `to`.
fn play(mixer: &Mixer, target: &mut InputTarget, out: &mut Box<dyn Output>, data: &[i32], to: u64) {
    taken(target);
    out.write(data, None).unwrap();
    wait_until("mixed", || mixer.timebase().frame == to);
}

/// The automation of the first input on the desk's first strip.
fn automation_view(mixer: &Mixer) -> Option<AutomationView> {
    mixer.desk().strips[0].inputs[0].automation.clone()
}

/// A plain input on an instant device, its output running with frame 0
/// (at `L`) written.
struct Plain {
    mixer: Mixer,
    device: Device,
    target: InputTarget,
    fader: Fader,
    out: Box<dyn Output>,
    written: u64,
}

fn plain() -> Plain {
    let (mixer, device) = instant_mixer();
    let strip = mixer.strip("Player");
    let (mut target, fader) = mixer.input(&strip, Role::Lead, 1.0);
    let out = open_started(&mut target, RATE, L);
    Plain {
        mixer,
        device,
        target,
        fader,
        out,
        written: 1,
    }
}

impl Plain {
    /// Plays its constant up to frame `to`.
    fn play_to(&mut self, to: u64) {
        let data = frames(self.written, to - self.written, |_| L);
        play(&self.mixer, &mut self.target, &mut self.out, &data, to);
        self.written = to;
    }

    /// The left channel of the device's opening `i`, once all is played.
    fn finish(self, i: usize) -> Vec<i32> {
        let Plain {
            mixer,
            device,
            mut out,
            ..
        } = self;
        out.drain(None).unwrap();
        drop(out);
        drop(mixer);
        device.left(i)
    }
}

#[test]
fn a_curve_lands_on_its_stream_frames_and_passthrough_comes_back() {
    let mut p = plain();
    // Down to a quarter from 1 000 to 9 000, held, back up from 12 000
    // to 14 000: at 1 from there, settled.
    let points = [(1_000, 1.0), (9_000, 0.25), (12_000, 0.25), (14_000, 1.0)];
    let set = Automation::new().input(curve(&points));
    p.fader.automate(set).unwrap();
    p.play_to(10_080);
    let view = automation_view(&p.mixer).unwrap();
    assert_eq!((view.main, view.parts.len(), view.settled), (1.0, 0, false));
    assert_eq!(view.input, 0.25);
    assert!(!p.mixer.passthrough());
    p.play_to(24_000);
    let view = automation_view(&p.mixer).unwrap();
    assert!(view.settled && view.input == 1.0, "{view:?}");
    assert!(p.mixer.passthrough());
    let left = p.finish(0);
    assert_eq!(left.len(), 24_000);
    let g = |f: usize| match f {
        f if f < 1_000 => 1.0,
        f if f < 9_000 => equal_power(1.0, 0.25, (f - 1_000) as f64 / 8_000.0),
        f if f < 12_000 => 0.25,
        f if f < 14_000 => equal_power(0.25, 1.0, (f - 12_000) as f64 / 2_000.0),
        _ => 1.0,
    };
    assert_eq!(off(&left, 0, 24_000, |f| f64::from(L) * g(f)), None);
    // As written from the period after the curve's end.
    assert_eq!(wrong(&left, 14_880, 24_000, |_| L), None);
}

#[test]
fn a_set_replaced_mid_curve_is_smoothed_from_the_gain_heard() {
    let mut p = plain();
    // Down from 1 towards silence at 48 000, power-linearly.
    let a = |f: usize| (1.0 - f as f64 / 48_000.0).sqrt();
    let first = curve(&[(0, 1.0), (48_000, 0.0)]);
    p.fader.automate(Automation::new().input(first)).unwrap();
    p.play_to(12_000);
    // Mid-curve, replaced by one well away from it at the frame it lands
    // on: from where the first was heard to it over 10 ms.
    let b = |f: usize| equal_power(0.3, 1.0, f as f64 / 30_000.0);
    let second = curve(&[(0, 0.3), (30_000, 1.0)]);
    p.fader.automate(Automation::new().input(second)).unwrap();
    p.play_to(40_800);
    let left = p.finish(0);
    let heard = a(11_999);
    let g = |f: usize| match f {
        f if f < 12_000 => a(f),
        f if f < 12_000 + FADE => lerp(heard, b(f), (f - 12_000) as f64 / FADE as f64),
        f if f < 30_000 => b(f),
        _ => 1.0,
    };
    assert_eq!(off(&left, 0, 40_800, |f| f64::from(L) * g(f)), None);
    // No step larger than the smoothing's (and the curves' own slopes).
    let smoothing = f64::from(L) * (heard - b(12_000)).abs() / FADE as f64;
    let step = largest_step(&left, 11_000, 13_000);
    assert!(step <= smoothing + 50.0, "a step of {step}");
    assert!(heard - b(12_000) > 0.15, "the curves differ there");
    assert_eq!(wrong(&left, 30_720, 40_800, |_| L), None);
}

#[test]
fn a_set_cleared_goes_back_to_1_over_the_smoothing() {
    let mut p = plain();
    let half = Automation::new().input(curve(&[(0, 0.5)]));
    p.fader.automate(half).unwrap();
    p.play_to(4_800);
    p.fader.clear_automation();
    p.play_to(9_600);
    assert_eq!(automation_view(&p.mixer), None);
    assert!(p.mixer.passthrough());
    let left = p.finish(0);
    let g = |f: usize| match f {
        f if f < 4_800 => 0.5,
        f if f < 4_800 + FADE => lerp(0.5, 1.0, (f - 4_800) as f64 / FADE as f64),
        _ => 1.0,
    };
    assert_eq!(off(&left, 0, 9_600, |f| f64::from(L) * g(f)), None);
    let smoothing = f64::from(L) * 0.5 / FADE as f64;
    assert!(largest_step(&left, 4_000, 6_000) <= smoothing + 2.0);
    assert_eq!(wrong(&left, 5_760, 9_600, |_| L), None);
}

#[test]
fn a_set_ends_with_its_output_and_one_given_with_none_open_is_the_nexts() {
    let (mixer, device) = instant_mixer();
    let strip = mixer.strip("Player");
    let (mut target, fader) = mixer.input(&strip, Role::Lead, 1.0);
    let set = |g: f64| Automation::new().input(curve(&[(0, g)]));
    let rest = frames(1, 4_799, |_| L);
    let mut out = open_started(&mut target, RATE, L);
    fader.automate(set(0.5)).unwrap();
    play(&mixer, &mut target, &mut out, &rest, 4_800);
    out.drain(None).unwrap();
    // Its output closed (the device with it): the set is gone. One given
    // now is the next output's, from its first frame.
    drop(out);
    wait_until("gone", || automation_view(&mixer).is_none());
    fader.automate(set(0.25)).unwrap();
    let mut out = open_started(&mut target, RATE, L);
    play(&mixer, &mut target, &mut out, &rest, 4_800);
    assert_eq!(automation_view(&mixer).map(|v| v.input), Some(0.25));
    out.drain(None).unwrap();
    // Another output on the input (the old one closed first, the device
    // opens again): the set ends with the one it was for.
    let mut out = open_started(&mut target, RATE, L);
    wait_until("ended", || automation_view(&mixer).is_none());
    play(&mixer, &mut target, &mut out, &rest, 4_800);
    out.drain(None).unwrap();
    drop(out);
    drop(mixer);
    let opened: Vec<Vec<i32>> = (0..3).map(|i| device.left(i)).collect();
    let lengths: Vec<usize> = opened.iter().map(Vec::len).collect();
    assert_eq!(lengths, [4_800; 3]);
    assert_eq!(off(&opened[0], 0, 4_800, |_| f64::from(L) * 0.5), None);
    assert_eq!(off(&opened[1], 0, 4_800, |_| f64::from(L) * 0.25), None);
    assert_eq!(wrong(&opened[2], 0, 4_800, |_| L), None);
}

#[test]
fn a_curve_holds_through_a_pause_while_the_device_plays_on() {
    let (mixer, device) = instant_mixer();
    let (sa, sb) = (mixer.strip("A"), mixer.strip("B"));
    let (mut a, fa) = mixer.input(&sa, Role::Lead, 1.0);
    let (mut b, _fb) = mixer.input(&sb, Role::Guest, 1.0);
    let right = |n: u64| -> Vec<i32> { (0..n).flat_map(|_| [0, L]).collect() };
    // Both running before either holds a period: an output not started
    // yet is not waited for, and A would be mixed alone.
    let mut oa = open_started(&mut a, RATE, L);
    let mut ob = b.open(&request(RATE).prefill(Duration::ZERO)).unwrap();
    ob.write(&right(1), None).unwrap();
    // Down from 1 to a fifth over A's frames 0..9 600.
    let down = curve(&[(0, 1.0), (9_600, 0.2)]);
    fa.automate(Automation::new().input(down.clone())).unwrap();
    taken(&mut a);
    oa.write(&frames(1, 4_799, |_| L), None).unwrap();
    ob.write(&right(4_799), None).unwrap();
    wait_until("mixed", || mixer.timebase().frame == 4_800);
    // A paused for 4 800 device frames, B playing on alone.
    oa.pause().unwrap();
    ob.write(&right(4_800), None).unwrap();
    wait_until("mixed", || mixer.timebase().frame == 9_600);
    oa.resume().unwrap();
    oa.write(&frames(4_800, 9_600, |_| L), None).unwrap();
    ob.write(&right(9_600), None).unwrap();
    wait_until("mixed", || mixer.timebase().frame == 19_200);
    drop(play_both(oa, ob, Vec::new(), Vec::new()));
    drop(mixer);
    let (sink, _) = device.streams.lock().unwrap()[0].clone();
    let heard = unpack(&sink.bytes(), device.format());
    let (left, right): (Vec<i32>, Vec<i32>) = heard.chunks_exact(2).map(|f| (f[0], f[1])).unzip();
    assert_eq!(left.len(), 19_200);
    // A's frame k is at device frame k before the pause, k + 4 800 after:
    // its curve on its own frames, held through the pause.
    let stream = |d: usize| if d < 4_800 { d } else { d - 4_800 };
    let want = |d: usize| f64::from(L) * down.gain(stream(d) as u64, RATE);
    assert_eq!(off(&left, 0, 4_800, want), None);
    assert_eq!(off(&left, 4_800, 9_600, |_| 0.0), None);
    assert_eq!(off(&left, 9_600, 19_200, want), None);
    assert_eq!(off(&right, 0, 19_200, |_| f64::from(L)), None);
}

/// The leader's frame `k` on the left, `(k + 1) * 16`; read back, `k`.
fn code(v: i32) -> Option<u64> {
    (v.abs() >= 8).then(|| (f64::from(v) / 16.0).round() as u64 - 1)
}

fn open_with(target: &mut InputTarget, buffer: Duration) -> Box<dyn Output> {
    let request = request(RATE).prefill(Duration::ZERO).buffer(buffer);
    target.open(&request).unwrap()
}

#[test]
fn a_followers_curve_holds_through_its_leaders_underrun_and_pause() {
    let _guard = real_time();
    let (mixer, device) = mixer();
    let (sa, sb, sc) = (mixer.strip("A"), mixer.strip("B"), mixer.strip("C"));
    let (mut a, fa) = mixer.input(&sa, Role::Lead, 1.0);
    let (mut b, fb) = mixer.input(&sb, Role::Guest, 1.0);
    // Another input keeps the device running while the leader is paused.
    let (mut c, _fc) = mixer.input(&sc, Role::Guest, 1.0);
    let origin = 4_923u64;
    let mut oa = open_with(&mut a, Duration::from_millis(200));
    let mut ob = open_with(&mut b, Duration::from_secs(2));
    let mut oc = open_with(&mut c, Duration::from_millis(200));
    fb.follow(&fa, origin).unwrap();
    // On the follower's own frames: down from 1 to a tenth over its run.
    let n = 60_000u64;
    let down = curve(&[(0, 1.0), (55_000, 0.1)]);
    fb.automate(Automation::new().input(down.clone())).unwrap();
    let right: Vec<i32> = (0..n).flat_map(|_| [0, L]).collect();
    ob.write(&right, None).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let stopped = stop.clone();
    let silence = std::thread::spawn(move || {
        while !stopped.load(Ordering::Relaxed) {
            oc.write(&frames(0, 960, |_| 0), None).unwrap();
        }
        oc
    });
    let lead = |from: u64, n: u64| frames(from, n, |k| (k as i32 + 1) * 16);
    // A second, a gap that runs the leader dry, more, a pause, the rest.
    oa.write(&lead(0, 24_000), None).unwrap();
    std::thread::sleep(Duration::from_millis(600));
    oa.write(&lead(24_000, 24_000), None).unwrap();
    oa.pause().unwrap();
    std::thread::sleep(Duration::from_millis(400));
    oa.resume().unwrap();
    oa.write(&lead(48_000, 24_000), None).unwrap();
    oa.drain(None).unwrap();
    ob.drain(None).unwrap();
    stop.store(true, Ordering::Relaxed);
    drop(silence.join().unwrap());
    assert!(oa.stats().underruns > 0, "the leader ran short");
    assert_eq!((ob.stats().dropped, fb.dropped()), (0, 0));
    drop(mixer);
    let (sink, _) = device.streams.lock().unwrap()[0].clone();
    let heard = unpack(&sink.bytes(), device.format());
    // Every follower frame heard is at the curve's gain at its own frame.
    let mut followed = Vec::new();
    for (d, f) in heard.chunks_exact(2).enumerate() {
        let Some(k) = code(f[0]).filter(|&k| k >= origin && k - origin < n) else {
            continue;
        };
        let m = k - origin;
        let want = f64::from(L) * down.gain(m, RATE);
        assert!(
            (f64::from(f[1]) - want).abs() < 1.5,
            "frame {m} at {d}: {} for {want}",
            f[1]
        );
        followed.push((d, m));
    }
    assert_eq!(followed.len() as u64, n);
    assert!(followed.windows(2).all(|w| w[1].1 == w[0].1 + 1));
    // Held twice on a running device: the underrun and the pause.
    let held = followed.windows(2).filter(|w| w[1].0 - w[0].0 > 4_800);
    assert!(held.count() >= 2);
}

/// A lane input: main on Player, a part on each of its four children.
struct Lanes {
    mixer: Mixer,
    device: Device,
    target: InputTarget,
    fader: Fader,
    out: Box<dyn Output>,
}

/// `n` lane frames from stream frame `from`: main's left `main(k)`, part
/// j's left its share of it ([`P`]), and the control word `control`.
fn lane_frames(from: u64, n: u64, main: fn(u64) -> i32, control: Parts) -> Vec<i32> {
    let mut v = Vec::with_capacity(n as usize * lane_width(2, LANES));
    for k in from..from + n {
        let x = main(k);
        v.extend([x, 0]);
        for p in P {
            v.extend([(i64::from(x) * i64::from(p) / i64::from(X)) as i32, 0]);
        }
        v.push(control.word());
    }
    v
}

fn constant(_: u64) -> i32 {
    X
}

/// The lane input, its output running with frame 0 written.
fn lanes_on(main: fn(u64) -> i32, control: Parts) -> Lanes {
    let (mixer, device) = instant_mixer();
    let player = mixer.strip("Player");
    let parts: Vec<_> = ["Drums", "Bass", "Vocals", "Other"]
        .iter()
        .map(|name| mixer.strip_in(&player, name))
        .collect();
    let refs: Vec<_> = parts.iter().collect();
    let (mut target, fader, _parts) = mixer.input_lanes(&player, &refs, Role::Lead, 1.0);
    let request = OpenRequest::new(RATE, 2)
        .formats(&[device.format()])
        .lanes(LANES)
        .prefill(Duration::ZERO);
    let mut out = target.open(&request).unwrap();
    out.write(&lane_frames(0, 1, main, control), None).unwrap();
    Lanes {
        mixer,
        device,
        target,
        fader,
        out,
    }
}

impl Lanes {
    fn play(&mut self, from: u64, to: u64, main: fn(u64) -> i32, control: Parts) {
        let data = lane_frames(from, to - from, main, control);
        play(&self.mixer, &mut self.target, &mut self.out, &data, to);
    }

    /// The parts' share in the switch, as of the last frames mixed.
    fn share(&self) -> f64 {
        self.mixer.desk().strips[0].inputs[0].parts.unwrap().share
    }

    fn finish(self) -> Vec<i32> {
        let Lanes {
            mixer,
            device,
            mut out,
            ..
        } = self;
        out.drain(None).unwrap();
        drop(out);
        drop(mixer);
        device.left(0)
    }
}

/// What `parts` (part k's curve, or 1) make of the parts at frame `f`.
fn parts_at(parts: &[Option<&GainCurve>; PARTS], f: usize) -> f64 {
    let g = |k: usize| parts[k].map_or(1.0, |c| c.gain(f as u64, RATE));
    (0..PARTS).map(|k| f64::from(P[k]) * g(k)).sum()
}

#[test]
fn with_the_parts_there_their_curves_shape_each_part() {
    let mut d = lanes_on(constant, Parts::Present);
    let input = curve(&[(0, 1.0), (10_000, 1.0), (20_000, 0.5)]);
    let drums = curve(&[(0, 0.5), (4_800, 0.5), (14_400, 0.0)]);
    let vocals = curve(&[(0, 1.0), (9_600, 1.0), (9_600, 0.0)]);
    // Main's is not heard while the parts are.
    let auto = Automation::new()
        .input(input.clone())
        .main(curve(&[(0, 1.0), (2_000, 1.0), (12_000, 0.2)]))
        .part(0, drums.clone())
        .part(2, vocals.clone());
    d.fader.automate(auto).unwrap();
    d.play(1, 24_000, constant, Parts::Present);
    let view = automation_view(&d.mixer).unwrap();
    assert_eq!(view.parts, [0.0, 1.0, 0.0, 1.0]);
    assert_eq!((view.input, view.main), (0.5, 0.2));
    assert_eq!(d.share(), 1.0);
    let left = d.finish();
    // The drums at 0.5 from the first frame: the parts at once.
    let parts = [Some(&drums), None, Some(&vocals), None];
    let want = |f: usize| input.gain(f as u64, RATE) * parts_at(&parts, f);
    assert_eq!(off(&left, 0, 24_000, want), None);
    // The drums' ramp, and the vocals' cut smoothed linearly.
    let ramp = equal_power(0.5, 0.0, (9_000 - 4_800) as f64 / 9_600.0);
    assert_eq!(drums.gain(9_000, RATE), ramp);
    assert_eq!(
        vocals.gain(9_700, RATE),
        lerp(1.0, 0.0, 100.0 / FADE as f64)
    );
}

#[test]
fn with_no_parts_mains_curve_shapes_main() {
    let mut d = lanes_on(constant, Parts::None);
    let input = curve(&[(0, 1.0), (10_000, 1.0), (20_000, 0.5)]);
    let main = curve(&[(0, 1.0), (2_000, 1.0), (12_000, 0.2)]);
    // The parts' curves are not heard while the parts are not there.
    let auto = Automation::new()
        .input(input.clone())
        .main(main.clone())
        .part(0, curve(&[(0, 0.5)]));
    d.fader.automate(auto).unwrap();
    d.play(1, 24_000, constant, Parts::None);
    assert_eq!(d.share(), 0.0);
    let left = d.finish();
    let want = |f: usize| f64::from(X) * input.gain(f as u64, RATE) * main.gain(f as u64, RATE);
    assert_eq!(off(&left, 0, 24_000, want), None);
    // Untouched by its curves before 2 000: main as written.
    assert_eq!(wrong(&left, 0, 1_920, |_| X), None);
}

#[test]
fn a_parts_curve_through_exactly_1_keeps_the_switch() {
    let mut d = lanes_on(constant, Parts::Present);
    // The drums at exactly 1 on the last frame of a period (4 799) and
    // away again after; main's curve at a quarter, so main heard would
    // show.
    let drums = curve(&[(0, 0.5), (4_799, 1.0), (9_600, 0.5)]);
    let auto = Automation::new()
        .main(curve(&[(0, 0.25)]))
        .part(0, drums.clone());
    d.fader.automate(auto).unwrap();
    d.play(1, 4_800, constant, Parts::Present);
    d.play(4_800, 14_400, constant, Parts::Present);
    assert_eq!(d.share(), 1.0);
    let left = d.finish();
    let parts = [Some(&drums), None, None, None];
    assert_eq!(off(&left, 0, 14_400, |f| parts_at(&parts, f)), None);
}

#[test]
fn the_inputs_own_curve_moving_does_not_hold_the_parts() {
    let mut d = lanes_on(constant, Parts::Present);
    // The drums back at 1 from 4 800, the input's curve moving on.
    let input = curve(&[(0, 1.0), (48_000, 0.2)]);
    let auto = Automation::new()
        .input(input.clone())
        .part(0, curve(&[(0, 0.5), (4_800, 1.0)]));
    d.fader.automate(auto).unwrap();
    d.play(1, 4_800, constant, Parts::Present);
    assert_eq!(d.share(), 1.0);
    d.play(4_800, 14_400, constant, Parts::Present);
    assert_eq!(d.share(), 0.0, "back on main");
    let view = automation_view(&d.mixer).unwrap();
    assert!(!view.settled, "the input's curve still moves");
    let left = d.finish();
    let want = |f: usize| f64::from(X) * input.gain(f as u64, RATE);
    assert_eq!(off(&left, 5_760, 14_400, want), None);
}

/// Main varying, so passthrough is seen to give it as written.
fn varying(k: u64) -> i32 {
    4 * ((k % 1_000) as i32 * 997 - 400_000)
}

#[test]
fn curves_that_end_at_1_bring_main_back_bit_for_bit() {
    let mut d = lanes_on(varying, Parts::Present);
    // The drums away from 1 from 4 800 to 14 400, and main's curve at 1.
    let auto = Automation::new()
        .part(0, curve(&[(4_800, 1.0), (9_600, 0.5), (14_400, 1.0)]))
        .main(curve(&[(0, 1.0)]));
    d.fader.automate(auto).unwrap();
    d.play(1, 9_600, varying, Parts::Present);
    assert!(!d.mixer.passthrough());
    assert_eq!(d.share(), 1.0);
    d.play(9_600, 24_000, varying, Parts::Present);
    assert!(d.mixer.passthrough());
    let view = automation_view(&d.mixer).unwrap();
    assert!(view.settled);
    let left = d.finish();
    assert_eq!(wrong(&left, 0, 4_800, |f| varying(f as u64)), None);
    // Settled by the end of the period 14 400 is in, the switch back over
    // 10 ms from the next: main as written from 15 360.
    assert_eq!(wrong(&left, 15_360, 24_000, |f| varying(f as u64)), None);
}

#[test]
fn automation_is_refused_for_a_part_the_input_lacks() {
    let (mixer, _device) = instant_mixer();
    let strip = mixer.strip("Player");
    let (_target, fader) = mixer.input(&strip, Role::Lead, 1.0);
    let parts = Automation::new().part(0, curve(&[(0, 0.5)]));
    assert!(matches!(
        fader.automate(parts),
        Err(Error::InvalidArgument(_))
    ));
    fader
        .automate(Automation::new().main(curve(&[(0, 0.5)])))
        .unwrap();
    fader.clear_automation();
    fader.close();
    let main = Automation::new().main(curve(&[(0, 0.5)]));
    assert!(matches!(fader.automate(main), Err(Error::Closed)));
}

#[test]
fn a_guest_made_lead_takes_the_microphone_and_is_asked_again() {
    let _guard = real_time();
    let (mixer, device) = mixer();
    let (sa, sb) = (mixer.strip("A"), mixer.strip("B"));
    let singer = Singer::new(loud);
    mixer.set_mic(Some(singer.input.clone()));
    let (mut a, fa) = mixer.input(&sa, Role::Lead, 1.0);
    let (mut b, fb) = mixer.input(&sb, Role::Guest, 1.0);
    let mut oa = open_with(&mut a, Duration::from_secs(3));
    oa.write(&frames(0, 96_000, |_| 0), None).unwrap();
    wait_until("the lead plays", || mixer.timebase().heard > 24_000);
    // A guest while the device is open: offered its format alone.
    assert_eq!(b.capabilities().unwrap().iter().count(), 1);
    let mut ob = open_with(&mut b, Duration::from_secs(3));
    ob.write(&frames(0, 48_000, |_| 0), None).unwrap();
    wait_until("the guest plays", || ob.position().unwrap() > 4_800);
    singer.voice();
    std::thread::sleep(Duration::from_millis(100));
    let lead_frames = singer.voice().iter().map(|v| v.stream_frame).max();
    assert!(lead_frames > Some(24_000), "{lead_frames:?}");
    // Made Lead: the tap is on its output (the newest Lead's), told in its
    // frames, and it is asked again what it takes.
    fb.set_role(Role::Lead).unwrap();
    assert_eq!(b.capabilities().unwrap().iter().count(), 2);
    singer.voice();
    std::thread::sleep(Duration::from_millis(200));
    let told = singer.voice().iter().map(|v| v.stream_frame).max().unwrap();
    let (played, guest) = (oa.position().unwrap(), ob.position().unwrap());
    // In the new lead's frames (at most where it has played to, plus a
    // period or two in flight), well behind the old lead's however slowly
    // a runner goes: the guest started 24 000 frames after the old lead.
    assert!(
        told <= guest + 4_800 && told + 12_000 < played,
        "{told} with the new lead at {guest} and the old at {played}"
    );
    assert_eq!(mixer.mic().port, MicPort::Hearing);
    let roles: Vec<Role> = mixer
        .desk()
        .strips
        .iter()
        .map(|s| s.inputs[0].role)
        .collect();
    assert_eq!(roles, [Role::Lead, Role::Lead]);
    // The old lead made a guest: another Lead plays. The new one cannot
    // be: it is the only Lead open.
    fa.set_role(Role::Guest).unwrap();
    assert_eq!(a.capabilities().unwrap().iter().count(), 1);
    assert!(matches!(
        fb.set_role(Role::Guest),
        Err(Error::InvalidArgument(_))
    ));
    // Nothing opened the device again.
    oa.drain(None).unwrap();
    ob.drain(None).unwrap();
    assert_eq!(device.streams.lock().unwrap().len(), 1);
    assert_eq!(mixer.mic().refused, 0);
    // With its output closed, it may be a guest.
    drop(ob);
    fb.set_role(Role::Guest).unwrap();
}
