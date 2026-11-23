//! Desk links: an input that follows another (`Fader::follow`). The
//! follower's frame 0 lands on the leader's frame `origin`, inside a
//! period, sample for sample (a plain input and a lane input), and a fade
//! for its first audio starts there; a follower linked after it gave frames
//! starts its frames where they are due; a leader short of frames or paused
//! holds its follower, which loses nothing (and the device pauses with the
//! leader); a leader at its end lets its follower play on; a follower short
//! of frames drops its late ones to land on its frames again, counts them
//! and plays them in its position; offline the render is the same however
//! the writers are scheduled, and a follower paused holds nothing up; a
//! retract that writes the same frames again keeps the link; a flush, a
//! close or a new output ends it; and a follower's `at_stream` is its
//! leader's before it has been mixed.
//!
//! The leader writes its frame k on the left channel as `(k + 1) * 16`, the
//! follower its frame m on the right as `±(m + 1) * 16` (the sign tells two
//! versions apart): summed and dithered, each is still read back exactly.

mod common;

use std::time::{Duration, Instant};

use common::*;
use jodel_output::lanes::{lane_width, Parts};
use jodel_output::mixer::{Curve, Fader, InputTarget, Mixer, Ramp, Role, Start};
use jodel_output::pack::unpack;
use jodel_output::{Error, Output, OutputTarget};

/// The leader's frames `from..from + n`.
fn lead(from: u64, n: u64) -> Vec<i32> {
    (from..from + n)
        .flat_map(|k| [(k as i32 + 1) * 16, 0])
        .collect()
}

/// The follower's frames `from..from + n`, of version `sign` (±1).
fn follow(from: u64, n: u64, sign: i32) -> Vec<i32> {
    (from..from + n)
        .flat_map(|m| [0, sign * (m as i32 + 1) * 16])
        .collect()
}

/// A frame number read back (`None`: silence), signed as written.
fn code(v: i32) -> Option<i64> {
    (v.abs() >= 8).then(|| (f64::from(v) / 16.0).round() as i64)
}

/// Both channels of the device's opening `i`.
fn heard(device: &Device, i: usize) -> Vec<(i32, i32)> {
    let (sink, _) = device.streams.lock().unwrap()[i].clone();
    let samples = unpack(&sink.bytes(), device.format());
    samples.chunks_exact(2).map(|f| (f[0], f[1])).collect()
}

/// A frame of the follower heard: the device frame, its frame, and the
/// version it was written in.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Heard {
    device: usize,
    frame: u64,
    sign: i64,
}

/// The follower's frames heard, in order, each checked to land on the
/// leader's frame `origin` on.
fn followed(frames: &[(i32, i32)], origin: u64) -> Vec<Heard> {
    let mut out = Vec::new();
    for (d, &(l, r)) in frames.iter().enumerate() {
        let Some(b) = code(r) else {
            continue;
        };
        let m = b.unsigned_abs() - 1;
        let a = code(l).unwrap_or_else(|| panic!("follower frame {m} at {d} without its leader"));
        assert_eq!(
            a as u64 - 1,
            m + origin,
            "follower frame {m} at device frame {d}"
        );
        out.push(Heard {
            device: d,
            frame: m,
            sign: b.signum(),
        });
    }
    out
}

/// The frames of `heard` follow one another with none missing.
fn unbroken(heard: &[Heard]) {
    for (k, h) in heard.iter().enumerate() {
        assert_eq!(h.frame, k as u64, "at device frame {}", h.device);
    }
}

/// A leader and a follower, plain inputs on strips of their own.
struct Pair {
    mixer: Mixer,
    device: Device,
    a: InputTarget,
    fa: Fader,
    b: InputTarget,
    fb: Fader,
}

fn pair_on((mixer, device): (Mixer, Device)) -> Pair {
    let (sa, sb) = (mixer.strip("A"), mixer.strip("B"));
    let (a, fa) = mixer.input(&sa, Role::Lead, 1.0);
    let (b, fb) = mixer.input(&sb, Role::Guest, 1.0);
    Pair {
        mixer,
        device,
        a,
        fa,
        b,
        fb,
    }
}

/// An output that starts on its first frame written, its ring `buffer`.
fn open_with(target: &mut InputTarget, buffer: Duration) -> Box<dyn Output> {
    let request = request(RATE).prefill(Duration::ZERO).buffer(buffer);
    target.open(&request).unwrap()
}

fn drain(out: &mut Box<dyn Output>) {
    assert!(out.drain(Some(Duration::from_secs(20))).unwrap(), "drained");
}

#[test]
fn a_followers_frame_0_lands_on_the_leaders_origin_inside_a_period() {
    let mut p = pair_on(instant_mixer());
    let origin = 1_234;
    let oa = open_with(&mut p.a, Duration::from_millis(500));
    let mut ob = open_with(&mut p.b, Duration::from_millis(500));
    p.fb.follow(&p.fa, origin).unwrap();
    // The follower runs first, held: nothing is mixed until its leader
    // runs, so the periods start on the leader's frame 0 and the origin
    // falls inside one.
    ob.write(&follow(0, 3_000, 1), None).unwrap();
    let (mut oa, mut ob) = play_both(oa, ob, lead(0, 6_000), Vec::new());
    drain(&mut oa);
    drain(&mut ob);
    let frames = heard(&p.device, 0);
    let b = followed(&frames, origin);
    unbroken(&b);
    assert_eq!(b.len(), 3_000);
    assert_eq!(b[0].device, origin as usize);
    assert_ne!(origin % 480, 0, "inside a period");
    assert_eq!(code(frames[origin as usize - 1].1), None);
    assert_eq!((ob.stats().dropped, p.fb.dropped()), (0, 0));
    assert_eq!(ob.position().unwrap(), 3_000);
    drop(p.mixer);
}

#[test]
fn a_lane_input_follows_on_its_parts() {
    let (mixer, device) = instant_mixer();
    let (sa, sb) = (mixer.strip("A"), mixer.strip("B"));
    let part = mixer.strip_in(&sb, "Part");
    let (mut a, fa) = mixer.input(&sa, Role::Lead, 1.0);
    let (mut b, fb, parts) = mixer.input_lanes(&sb, &[&part], Role::Guest, 1.0);
    // The part touched: heard in main's place, from the follower's first
    // frame (main is silent; the part twice the frame, at half).
    parts[0].set(Ramp {
        start: Start::Now,
        to: 0.5,
        over: Duration::ZERO,
        curve: Curve::Linear,
    });
    let origin = 2_003;
    let oa = open_with(&mut a, Duration::from_millis(500));
    let request = request(RATE).prefill(Duration::ZERO).lanes(2);
    let mut ob = b.open(&request).unwrap();
    fb.follow(&fa, origin).unwrap();
    let n = 2_500u64;
    let mut lanes = Vec::with_capacity(n as usize * lane_width(2, 2));
    for m in 0..n {
        let x = (m as i32 + 1) * 16;
        lanes.extend([0, 0, 0, 2 * x, Parts::Present.word()]);
    }
    ob.write(&lanes, None).unwrap();
    let (mut oa, mut ob) = play_both(oa, ob, lead(0, 6_000), Vec::new());
    drain(&mut oa);
    drain(&mut ob);
    drop(mixer);
    let b = followed(&heard(&device, 0), origin);
    unbroken(&b);
    assert_eq!((b.len(), b[0].device), (n as usize, origin as usize));
}

#[test]
fn a_leader_short_of_frames_holds_its_follower() {
    let _guard = real_time();
    let mut p = pair_on(mixer());
    let origin = 4_923;
    let mut oa = open_with(&mut p.a, Duration::from_millis(200));
    let mut ob = open_with(&mut p.b, Duration::from_millis(500));
    p.fb.follow(&p.fa, origin).unwrap();
    // The follower queued before its leader runs: in real time the leader
    // is at the origin at once.
    let total = 36_000;
    ob.write(&follow(0, 24_000, 1), None).unwrap();
    let writer = std::thread::spawn(move || {
        ob.write(&follow(24_000, total - 24_000, 1), None).unwrap();
        drain(&mut ob);
        ob
    });
    // Half a second, a gap that runs the leader's ring dry, then a second.
    oa.write(&lead(0, 24_000), None).unwrap();
    std::thread::sleep(Duration::from_millis(600));
    oa.write(&lead(24_000, 48_000), None).unwrap();
    drain(&mut oa);
    let ob = writer.join().unwrap();
    assert!(oa.stats().underruns > 0, "the leader ran short");
    drop(p.mixer);
    let b = followed(&heard(&p.device, 0), origin);
    unbroken(&b);
    assert_eq!(b.len() as u64, total);
    assert_eq!(ob.stats().dropped, 0);
    // The leader's gap is inside the follower's frames: they paused there.
    let gap = b.windows(2).map(|w| w[1].device - w[0].device).max();
    assert!(gap > Some(4_800), "{gap:?}");
}

#[test]
fn a_late_follower_drops_its_frames_up_to_where_it_is_due() {
    let _guard = real_time();
    let mut p = pair_on(mixer());
    let origin = 2_400;
    let mut oa = open_with(&mut p.a, Duration::from_millis(500));
    let mut ob = open_with(&mut p.b, Duration::from_secs(1));
    p.fb.follow(&p.fa, origin).unwrap();
    let leader = std::thread::spawn(move || {
        oa.write(&lead(0, 4 * u64::from(RATE)), None).unwrap();
        drain(&mut oa);
        oa
    });
    // The follower starts a tenth of a second late, and later runs dry
    // for a while. Its position is taken as it goes, between two looks at
    // what the device played.
    wait_until("past the origin", || {
        p.mixer.timebase().frame > origin + 4_800
    });
    let (first, second) = (48_000, 48_000);
    ob.write(&follow(0, first, 1), None).unwrap();
    wait_until("dropped", || p.fb.dropped() > 0);
    let until = Instant::now() + Duration::from_millis(1_300);
    let mut samples = Vec::new();
    while Instant::now() < until {
        let before = p.mixer.timebase().heard;
        let position = ob.position().unwrap();
        std::thread::sleep(Duration::from_millis(2));
        samples.push((before, position, p.mixer.timebase().heard));
        std::thread::sleep(Duration::from_millis(20));
    }
    ob.write(&follow(first, second, 1), None).unwrap();
    drain(&mut ob);
    let written = first + second;
    // Every frame is played or dropped.
    assert_eq!(ob.position().unwrap(), written);
    let dropped = ob.stats().dropped;
    leader.join().unwrap();
    assert_eq!(p.fb.dropped(), dropped);
    drop(p.mixer);
    let b = followed(&heard(&p.device, 0), origin);
    assert!(b.windows(2).all(|w| w[1].frame > w[0].frame));
    assert_eq!(b.len() as u64 + dropped, written);
    // Late at the start, and once more after running dry.
    let skips = b.windows(2).filter(|w| w[1].frame > w[0].frame + 1).count();
    assert!(b[0].frame > 0 && skips >= 1, "{} then {skips}", b[0].frame);
    assert_eq!(b.last().unwrap().frame, written - 1);
    // As it went: at least the frames heard before (and those dropped
    // before them), never the one the device had not played yet.
    let played_by = |device: u64| {
        let mut before = b.iter().filter(|h| (h.device as u64) < device);
        before.next_back().map_or(0, |h| h.frame + 1)
    };
    let next_at = |device: u64| {
        let next = b.iter().find(|h| h.device as u64 >= device);
        next.map_or(written, |h| h.frame)
    };
    assert!(samples.len() > 20);
    for (before, position, after) in samples {
        assert!(
            played_by(before) <= position && position <= next_at(after),
            "position {position}: heard {before} then {after}"
        );
    }
}

#[test]
fn a_paused_leader_holds_its_follower_and_the_device_pauses() {
    let _guard = real_time();
    let mut p = pair_on(mixer());
    let origin = 4_800;
    let mut oa = open_with(&mut p.a, Duration::from_secs(3));
    let mut ob = open_with(&mut p.b, Duration::from_secs(2));
    p.fb.follow(&p.fa, origin).unwrap();
    let total = 72_000;
    ob.write(&follow(0, total, 1), None).unwrap();
    oa.write(&lead(0, 96_000), None).unwrap();
    wait_until("following", || p.mixer.timebase().frame > origin + 9_600);
    oa.pause().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    let (mixed, played) = (p.mixer.timebase().frame, ob.position().unwrap());
    std::thread::sleep(Duration::from_millis(300));
    // Nothing is mixed: the device is paused; the follower holds.
    assert_eq!(p.mixer.timebase().frame, mixed);
    assert_eq!(ob.position().unwrap(), played);
    assert!(played < total);
    oa.resume().unwrap();
    drain(&mut oa);
    drain(&mut ob);
    drop(p.mixer);
    let b = followed(&heard(&p.device, 0), origin);
    unbroken(&b);
    assert_eq!(b.len() as u64, total);
    assert_eq!(ob.stats().dropped, 0);
}

#[test]
fn a_fade_for_a_followers_first_audio_starts_on_the_origin() {
    let (mixer, device) = instant_mixer();
    let (sa, sb) = (mixer.strip("A"), mixer.strip("B"));
    let (mut a, fa) = mixer.input(&sa, Role::Lead, 1.0);
    let (mut b, fb) = mixer.input(&sb, Role::Guest, 0.0);
    let over = 480;
    fb.set(Ramp {
        start: Start::FirstAudio,
        to: 1.0,
        over: frames_of(over),
        curve: Curve::Linear,
    });
    let origin = 1_234;
    let oa = open_with(&mut a, Duration::from_millis(500));
    let mut ob = open_with(&mut b, Duration::from_millis(500));
    fb.follow(&fa, origin).unwrap();
    let x = 1_000_000;
    let constant: Vec<i32> = (0..3_000).flat_map(|_| [0, x]).collect();
    ob.write(&constant, None).unwrap();
    let (mut oa, mut ob) = play_both(oa, ob, lead(0, 6_000), Vec::new());
    drain(&mut oa);
    drain(&mut ob);
    drop(mixer);
    let right: Vec<i32> = heard(&device, 0).iter().map(|f| f.1).collect();
    let o = origin as usize;
    assert!(right[..o].iter().all(|v| v.abs() <= 1));
    for k in 0..over as usize {
        let want = f64::from(x) * k as f64 / over as f64;
        assert!(
            (f64::from(right[o + k]) - want).abs() <= 2.0,
            "frame {k}: {}",
            right[o + k]
        );
    }
    assert!(right[o + over as usize..o + 3_000]
        .iter()
        .all(|v| (v - x).abs() <= 1));
}

#[test]
fn a_follower_linked_after_it_gave_frames_starts_where_they_are_due() {
    let mut p = pair_on(instant_mixer());
    let mut oa = open_with(&mut p.a, Duration::from_millis(500));
    let mut ob = open_with(&mut p.b, Duration::from_millis(500));
    // Both play 4 800 frames side by side (linked at 0, to start them
    // together).
    p.fb.follow(&p.fa, 0).unwrap();
    ob.write(&follow(0, 4_800, 1), None).unwrap();
    oa.write(&lead(0, 4_800), None).unwrap();
    wait_until("mixed", || p.mixer.timebase().frame == 4_800);
    // Linked again so that its next frame, 4 800, is due on the leader's
    // 5 800: a thousand frames on.
    let origin = 1_000;
    p.fb.follow(&p.fa, origin).unwrap();
    let (mut oa, mut ob) = play_both(oa, ob, lead(4_800, 9_600), follow(4_800, 4_800, -1));
    drain(&mut oa);
    drain(&mut ob);
    drop(p.mixer);
    let frames = heard(&p.device, 0);
    let linked: Vec<(i32, i32)> = frames
        .iter()
        .map(|&(l, r)| {
            if code(r).is_some_and(|m| m < 0) {
                (l, r)
            } else {
                (l, 0)
            }
        })
        .collect();
    let b = followed(&linked, origin);
    assert_eq!(b.len(), 4_800);
    for (k, h) in b.iter().enumerate() {
        assert_eq!(h.frame, 4_800 + k as u64);
    }
    assert_eq!(b[0].device, 5_800);
    // Nothing of it between its frames before and after the link.
    assert!(frames[4_800..5_800].iter().all(|f| code(f.1).is_none()));
    assert_eq!(ob.stats().dropped, 0);
}

#[test]
fn a_leader_at_its_end_lets_its_follower_play_on() {
    let mut p = pair_on(instant_mixer());
    let origin = 1_000;
    let oa = open_with(&mut p.a, Duration::from_millis(500));
    let mut ob = open_with(&mut p.b, Duration::from_millis(500));
    p.fb.follow(&p.fa, origin).unwrap();
    let total = 10_000;
    ob.write(&follow(0, total, 1), None).unwrap();
    let (mut oa, mut ob) = play_both(oa, ob, lead(0, 6_000), Vec::new());
    drain(&mut oa);
    drain(&mut ob);
    // Its clock is where it played on, as it was due.
    let tb = p.mixer.timebase();
    assert_eq!(p.fb.at_stream(7_000), tb.at(7_000 + origin));
    drop(p.mixer);
    let frames = heard(&p.device, 0);
    let b: Vec<(usize, i64)> = frames
        .iter()
        .enumerate()
        .filter_map(|(d, f)| code(f.1).map(|m| (d, m - 1)))
        .collect();
    assert_eq!(b.len() as u64, total);
    for (k, &(d, m)) in b.iter().enumerate() {
        assert_eq!((m, d), (k as i64, k + origin as usize));
    }
    assert_eq!(ob.stats().dropped, 0);
}

#[test]
fn offline_a_follower_paused_after_its_origin_holds_nothing_up() {
    let mut p = pair_on(instant_mixer());
    let origin = 1_000;
    let mut oa = open_with(&mut p.a, Duration::from_millis(500));
    let mut ob = open_with(&mut p.b, Duration::from_millis(500));
    p.fb.follow(&p.fa, origin).unwrap();
    ob.write(&follow(0, 9_600, 1), None).unwrap();
    oa.write(&lead(0, 4_800), None).unwrap();
    wait_until("mixed", || p.mixer.timebase().frame == 4_800);
    ob.pause().unwrap();
    taken(&mut p.b);
    // The leader plays on to its end; the follower holds.
    oa.write(&lead(4_800, 9_600), None).unwrap();
    drain(&mut oa);
    assert_eq!(ob.position().unwrap(), 3_800);
    drop(p.mixer);
    let frames = heard(&p.device, 0);
    let b = followed(&frames, origin);
    unbroken(&b);
    assert_eq!(b.len(), 3_800);
    let a = frames.iter().filter(|f| code(f.0).is_some()).count();
    assert_eq!(a, 14_400);
}

/// An offline render of a link: the follower written `late` after its
/// leader (the leader held at the origin meanwhile). Returns what the
/// device got and what the follower dropped.
fn render(late: Duration) -> (Vec<(i32, i32)>, u64) {
    let mut p = pair_on(instant_mixer());
    let origin = 7_011;
    let mut oa = open_with(&mut p.a, Duration::from_millis(500));
    let mut ob = open_with(&mut p.b, Duration::from_millis(500));
    p.fb.follow(&p.fa, origin).unwrap();
    let writer = std::thread::spawn(move || {
        std::thread::sleep(late);
        ob.write(&follow(0, 24_000, 1), None).unwrap();
        drain(&mut ob);
        ob
    });
    oa.write(&lead(0, 48_000), None).unwrap();
    drain(&mut oa);
    let ob = writer.join().unwrap();
    drop(p.mixer);
    (heard(&p.device, 0), ob.stats().dropped)
}

#[test]
fn offline_a_link_renders_the_same_however_the_writers_run() {
    let (first, dropped) = render(Duration::ZERO);
    assert_eq!(dropped, 0);
    for late in [Duration::from_millis(50), Duration::from_millis(5)] {
        let (again, dropped) = render(late);
        assert_eq!(dropped, 0);
        assert!(again == first, "late by {late:?}");
    }
    // Every frame of the follower, on its leader's.
    let b: Vec<Heard> = first
        .iter()
        .enumerate()
        .filter_map(|(d, &(_, r))| {
            code(r).map(|m| Heard {
                device: d,
                frame: m as u64 - 1,
                sign: 1,
            })
        })
        .collect();
    unbroken(&b);
    assert_eq!((b.len(), b[0].device), (24_000, 7_011));
}

#[test]
fn a_retract_that_writes_the_same_frames_again_keeps_the_link() {
    let mut p = pair_on(instant_mixer());
    let origin = 1_000;
    let mut oa = open_with(&mut p.a, Duration::from_millis(500));
    let mut ob = open_with(&mut p.b, Duration::from_millis(500));
    p.fb.follow(&p.fa, origin).unwrap();
    ob.write(&follow(0, 20_000, 1), None).unwrap();
    oa.write(&lead(0, 6_000), None).unwrap();
    // Mixed while the leader holds a period.
    wait_until("mixed", || p.mixer.timebase().frame == 5_760);
    // The follower takes back what it had not given and writes those
    // frames again, another version of them.
    let back = ob.retract(12_000).unwrap();
    assert_eq!(back, 12_000);
    let from = ob.written();
    assert_eq!(from, 8_000);
    let total = 30_000;
    let (mut oa, mut ob) = play_both(oa, ob, lead(6_000, 42_000), follow(from, total - from, -1));
    drain(&mut oa);
    drain(&mut ob);
    drop(p.mixer);
    let b = followed(&heard(&p.device, 0), origin);
    unbroken(&b);
    assert_eq!(b.len() as u64, total);
    for h in &b {
        let sign = if h.frame < from { 1 } else { -1 };
        assert_eq!(h.sign, sign, "frame {}", h.frame);
    }
    assert_eq!(ob.stats().dropped, 0);
}

#[test]
fn a_flush_of_either_side_ends_the_link() {
    let mut p = pair_on(instant_mixer());
    let far = 1_000_000;
    let mut oa = open_with(&mut p.a, Duration::from_millis(500));
    let mut ob = open_with(&mut p.b, Duration::from_millis(500));
    p.fb.follow(&p.fa, far).unwrap();
    ob.write(&follow(0, 4_800, 1), None).unwrap();
    oa.write(&lead(0, 9_600), None).unwrap();
    wait_until("mixed", || p.mixer.timebase().frame == 9_600);
    // Far before its origin, the follower gave nothing.
    assert_eq!(ob.position().unwrap(), 0);
    // The leader seeks: the follower plays on its own, from its frame 0.
    oa.flush().unwrap();
    drain(&mut ob);
    let frames = heard(&p.device, 0);
    let b: Vec<u64> = frames
        .iter()
        .filter_map(|f| code(f.1))
        .map(|m| m as u64 - 1)
        .collect();
    assert_eq!(b, (0..4_800).collect::<Vec<_>>());
    let last_a = frames.iter().rposition(|f| code(f.0).is_some());
    let first_b = frames.iter().position(|f| code(f.1).is_some());
    assert!(first_b > last_a, "{first_b:?} after {last_a:?}");
    // Linked again, and held by a leader that gives nothing: its own flush
    // ends that link too.
    p.fb.follow(&p.fa, far).unwrap();
    ob.write(&follow(4_800, 4_800, -1), None).unwrap();
    taken(&mut p.b);
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(ob.position().unwrap(), 4_800);
    ob.flush().unwrap();
    ob.write(&follow(9_600, 4_800, 1), None).unwrap();
    drain(&mut ob);
    assert_eq!(ob.position().unwrap(), 14_400);
    // Closing the leader does too.
    p.fb.follow(&p.fa, far).unwrap();
    drop(oa);
    ob.write(&follow(14_400, 4_800, 1), None).unwrap();
    drain(&mut ob);
    drop(p.mixer);
    let frames = heard(&p.device, 0);
    let after: Vec<i64> = frames.iter().filter_map(|f| code(f.1)).collect();
    assert_eq!(after.len(), 4_800 * 3);
    assert!(
        after.iter().all(|&m| m > 0),
        "the version flushed is never heard"
    );
}

#[test]
fn a_new_output_on_either_input_ends_the_link() {
    let mut p = pair_on(instant_mixer());
    let far = 1_000_000;
    let mut oa = open_with(&mut p.a, Duration::from_millis(500));
    let mut ob = open_with(&mut p.b, Duration::from_millis(500));
    p.fb.follow(&p.fa, far).unwrap();
    ob.write(&follow(0, 4_800, 1), None).unwrap();
    oa.write(&lead(0, 9_600), None).unwrap();
    wait_until("mixed", || p.mixer.timebase().frame == 9_600);
    assert_eq!(ob.position().unwrap(), 0);
    // A new output on the leader's input: the follower plays on its own.
    let _oa = open_with(&mut p.a, Duration::from_millis(500));
    drain(&mut ob);
    assert_eq!(ob.position().unwrap(), 4_800);
    // Linked again, and a new output on the follower's input: the leader
    // leads no one now, so it may follow that one.
    p.fb.follow(&p.fa, far).unwrap();
    let _ob = open_with(&mut p.b, Duration::from_millis(500));
    p.fa.follow(&p.fb, 0).unwrap();
}

#[test]
fn links_are_one_level_and_need_their_outputs() {
    let mut p = pair_on(instant_mixer());
    let strip = p.mixer.strip("C");
    let (mut c, fc) = p.mixer.input(&strip, Role::Guest, 1.0);
    let invalid = |r: Result<(), Error>| matches!(r, Err(Error::InvalidArgument(_)));
    // No outputs open yet.
    assert!(invalid(p.fb.follow(&p.fa, 0)));
    let _oa = open(&mut p.a, RATE);
    let _ob = open(&mut p.b, RATE);
    let _oc = open(&mut c, RATE);
    assert!(invalid(p.fb.follow(&p.fb, 0)));
    p.fb.follow(&p.fa, 10).unwrap();
    // No chains: B leads no one, and A follows no one while B follows it.
    assert!(invalid(fc.follow(&p.fb, 0)));
    assert!(invalid(p.fa.follow(&p.fb, 0)));
    // Unlinked, B may lead.
    p.fb.unfollow();
    fc.follow(&p.fb, 0).unwrap();
}

#[test]
fn a_followers_clock_is_its_leaders_before_it_is_mixed() {
    let mut p = pair_on(instant_mixer());
    let mut oa = open_with(&mut p.a, Duration::from_millis(500));
    oa.write(&lead(0, 2_000), None).unwrap();
    wait_until("mixed", || p.mixer.timebase().frame == 1_920);
    let _ob = open(&mut p.b, RATE);
    assert_eq!(p.fb.at_stream(5), Start::Now);
    let origin = 777;
    p.fb.follow(&p.fa, origin).unwrap();
    for n in [0, 5, 1_000, 48_000] {
        assert_eq!(p.fb.at_stream(n), p.fa.at_stream(n + origin), "frame {n}");
        assert_ne!(p.fb.at_stream(n), Start::Now);
    }
    // Unlinked before it gave a frame: it has no clock of its own.
    p.fb.unfollow();
    taken(&mut p.b);
    assert_eq!(p.fb.at_stream(5), Start::Now);
}
