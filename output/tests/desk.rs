//! The desk's groups and scheduled changes, on a simulated device: a
//! group's level and mute act on its children and its own inputs, ramped;
//! a child's solo silences its siblings, its group's own inputs and every
//! other strip, a group's solo keeps its children, and solo off brings all
//! back; a mute and a level set for a desk frame start their ramps on that
//! very frame; a group mute for the first audio starts on the first frame
//! of an output opened since on a child, not on an older one's; a change
//! for a time of an epoch gone never comes; a lone input on a child of a
//! group, all at 0 dB, reaches the device bit for bit, and again bit for
//! bit once a group's ramp has played out; and a group meters its subtree.

mod common;

use common::*;
use jodel_output::mixer::{InputTarget, Mixer, Role, Start};
use jodel_output::Output;

/// Writes each of `outs` its constant of `levels` up to frame `to` (from
/// `written`; a few periods at most, so no ring fills), once the mixer has
/// taken what was set before ([`taken`], through `sync`), and waits until
/// it has mixed them.
fn play_to(
    mixer: &Mixer,
    sync: &mut InputTarget,
    outs: &mut [Box<dyn Output>],
    levels: &[i32],
    written: &mut u64,
    to: u64,
) {
    taken(sync);
    let n = to - *written;
    for (out, &level) in outs.iter_mut().zip(levels) {
        out.write(&frames(0, n, |_| level), None).unwrap();
    }
    *written = to;
    wait_until("mixed", || mixer.timebase().frame == to);
}

/// Drains `outs` together.
fn drain(outs: Vec<Box<dyn Output>>) {
    let none = vec![Vec::new(); outs.len()];
    drop(play_all(outs, none));
}

/// Each change, the one set at the end of step `k` (from 0, `step` frames
/// a step), from the frame it was set at over 20 ms: from the level
/// before, through the middle, to the level after.
fn ramps_from_their_frames(left: &[i32], step: usize, changes: &[(i32, i32)]) {
    for (k, &(from, to)) in changes.iter().enumerate() {
        let at = (k + 1) * step;
        let near = |i: usize, v: i32| left[i].abs_diff(v) <= 2;
        assert!(near(at - 1, from), "{k}: {}", left[at - 1]);
        assert!(near(at + 480, (from + to) / 2), "{k}: {}", left[at + 480]);
        assert!(near(at + 960, to), "{k}: {}", left[at + 960]);
    }
}

#[test]
fn a_groups_level_and_mute_act_on_its_children() {
    let (mixer, device) = instant_mixer();
    let group = mixer.strip("Player");
    let child = mixer.strip_in(&group, "Drums");
    let other = mixer.strip("Preview");
    let (mut a, _fa) = mixer.input(&child, Role::Lead, 1.0);
    let (mut b, _fb) = mixer.input(&group, Role::Guest, 1.0);
    let (mut c, _fc) = mixer.input(&other, Role::Guest, 1.0);
    let levels = [1_000_000, 2_000_000, 4_000_000];
    let mut outs = vec![
        open_started(&mut a, RATE, levels[0]),
        open_started(&mut b, RATE, levels[1]),
        open_started(&mut c, RATE, levels[2]),
    ];
    let step = 4_800;
    let mut written = 1;
    let mut to = 0;
    let mut play = |outs: &mut [Box<dyn Output>]| {
        to += step;
        play_to(&mixer, &mut a, outs, &levels, &mut written, to);
    };
    play(&mut outs);
    group.set_level(0.5);
    play(&mut outs);
    group.set_mute(true);
    play(&mut outs);
    assert!(mixer.desk().strips[0].mute && group.mute());
    group.set_mute(false);
    play(&mut outs);
    child.set_level(0.5);
    play(&mut outs);
    group.set_level(1.0);
    child.set_level(1.0);
    play(&mut outs);
    drain(outs);
    drop(mixer);
    let left = device.left(0);
    let all = 7_000_000;
    let (seen, largest) = plateaus(&left, &[all, 5_500_000, 4_000_000, 5_250_000], all);
    assert_eq!(
        seen,
        [all, 5_500_000, 4_000_000, 5_500_000, 5_250_000, all],
        "all; the group at -6 dB (its child's input and its own); muted \
         (only the other strip); unmuted; the child at -6 dB too; all"
    );
    let changes = [
        (all, 5_500_000),
        (5_500_000, 4_000_000),
        (4_000_000, 5_500_000),
        (5_500_000, 5_250_000),
    ];
    ramps_from_their_frames(&left, step as usize, &changes);
    // Ramped: 1 500 000 over 960 frames is about 1 563 a frame; the last
    // change, the group's and the child's ramps multiplied, is steepest at
    // its end (2 × 1 000 000 × 0.5 + 2 000 000 × 0.5 over 960: 2 083).
    assert!(largest < 2_200, "a step of {largest}");
}

#[test]
fn a_childs_solo_silences_the_rest_and_a_groups_keeps_its_children() {
    let (mixer, device) = instant_mixer();
    let group = mixer.strip("Player");
    let (c1, c2) = (
        mixer.strip_in(&group, "Drums"),
        mixer.strip_in(&group, "Bass"),
    );
    let other = mixer.strip("Preview");
    let strips = [&c1, &c2, &group, &other];
    let levels = [1_000_000, 2_000_000, 500_000, 4_000_000];
    let mut faders = Vec::new();
    let mut outs = Vec::new();
    for (k, (strip, &level)) in strips.iter().zip(&levels).enumerate() {
        let role = if k == 0 { Role::Lead } else { Role::Guest };
        let (mut target, fader) = mixer.input(strip, role, 1.0);
        outs.push(open_started(&mut target, RATE, level));
        faders.push((target, fader));
    }
    let step = 4_800;
    let mut written = 1;
    let mut to = 0;
    let mut play = |outs: &mut [Box<dyn Output>]| {
        to += step;
        play_to(&mixer, &mut faders[0].0, outs, &levels, &mut written, to);
    };
    play(&mut outs);
    c1.set_solo(true);
    play(&mut outs);
    let solos: Vec<bool> = mixer.desk().strips.iter().map(|s| s.solo).collect();
    assert_eq!(solos, [false, true, false, false], "{:?}", mixer.desk());
    c1.set_solo(false);
    play(&mut outs);
    group.set_solo(true);
    play(&mut outs);
    group.set_solo(false);
    play(&mut outs);
    drain(outs);
    drop(mixer);
    let left = device.left(0);
    let all = 7_500_000;
    let (seen, largest) = plateaus(&left, &[all, 1_000_000, 3_500_000], all);
    assert_eq!(
        seen,
        [all, 1_000_000, all, 3_500_000, all],
        "all; a child soloed (alone); all; its group soloed (the group's \
         own input and both children); all"
    );
    let changes = [
        (all, 1_000_000),
        (1_000_000, all),
        (all, 3_500_000),
        (3_500_000, all),
    ];
    ramps_from_their_frames(&left, step as usize, &changes);
    // 6 500 000 over 960 frames is about 6 771 a frame.
    assert!(largest < 7_000, "a step of {largest}");
}

#[test]
fn a_mute_and_a_level_set_for_a_frame_ramp_from_that_frame() {
    let (mixer, device) = instant_mixer();
    let group = mixer.strip("Player");
    let child = mixer.strip_in(&group, "Vocals");
    let (mut a, _fader) = mixer.input(&child, Role::Lead, 1.0);
    let level = 1_000_000;
    let mut out = open_started(&mut a, RATE, level);
    out.write(&frames(1, 4_799, |_| level), None).unwrap();
    wait_until("mixed", || mixer.timebase().frame == 4_800);
    // Off the periods (480 frames): the group muted at `mute`, in place of
    // a mute set for earlier; the child to -6 dB at `down`; once the mute
    // is in force, the group unmuted at `back`.
    let tb = mixer.timebase();
    let (mute, back, down): (usize, usize, usize) = (9_723, 19_211, 22_222);
    let at = |f: usize| tb.at(f as u64);
    group.set_mute_at(true, at(9_000));
    group.set_mute_at(true, at(mute));
    child.set_level_at(0.5, at(down));
    assert!(!group.mute());
    taken(&mut a);
    out.write(&frames(4_800, 9_600, |_| level), None).unwrap();
    wait_until("mixed", || mixer.timebase().frame == 14_400);
    assert!(group.mute() && child.level() == 1.0);
    group.set_mute_at(false, at(back));
    taken(&mut a);
    out.write(&frames(14_400, 9_600, |_| level), None).unwrap();
    out.drain(None).unwrap();
    assert!(!group.mute() && child.level() == 0.5);
    drop(out);
    drop(mixer);
    let left = device.left(0);
    assert_eq!(left.len(), 24_000);
    let period = |f: usize| f / 480 * 480;
    let within = |from: usize, to: usize, want: &dyn Fn(usize) -> f64| {
        let wrong = (from..to).find(|&f| (f64::from(left[f]) - want(f)).abs() >= 1.5);
        assert_eq!(wrong.map(|f| (f, left[f])), None, "{from}..{to}");
    };
    let exactly = |from: usize, to: usize, v: i32| {
        let wrong = (from..to).find(|&f| left[f] != v);
        assert_eq!(wrong.map(|f| (f, left[f])), None, "{from}..{to}");
    };
    let over = 960; // 20 ms
    let ramp = |at: usize, from: f64, to: f64| {
        move |f: usize| f64::from(level) * (from + (to - from) * (f - at) as f64 / over as f64)
    };
    // As written up to the period the mute starts in; then, from that very
    // frame, down over 20 ms; then silence (exactly, from the next period).
    exactly(0, period(mute), level);
    within(period(mute), mute, &|_| f64::from(level));
    within(mute, mute + over, &ramp(mute, 1.0, 0.0));
    within(mute + over, period(mute + over) + 480, &|_| 0.0);
    exactly(period(mute + over) + 480, period(back), 0);
    // Back up from its frame, then as written again.
    within(back, back + over, &ramp(back, 0.0, 1.0));
    exactly(period(back + over) + 480, period(down), level);
    // The child's level from its frame.
    within(period(down), down, &|_| f64::from(level));
    within(down, down + over, &ramp(down, 1.0, 0.5));
    within(down + over, 24_000, &|_| f64::from(level) * 0.5);
}

#[test]
fn a_lone_input_on_a_child_of_a_group_at_0_db_is_bit_for_bit() {
    let (mixer, device) = instant_mixer();
    let group = mixer.strip("Player");
    let child = mixer.strip_in(&group, "Main");
    let preview = mixer.strip("Preview");
    // Down and back, muted and back, before anything plays: exactly 0 dB.
    group.set_level(0.5);
    group.set_level(1.0);
    child.set_mute(true);
    child.set_mute(false);
    let (mut lead, _fader) = mixer.input(&child, Role::Lead, 1.0);
    // An input on the group with nothing open, and a silent guest.
    let (_direct, _direct_fader) = mixer.input(&group, Role::Guest, 1.0);
    let (_guest, _guest_fader) = mixer.input(&preview, Role::Guest, 0.0);
    let mut out = open(&mut lead, RATE);
    let total = u64::from(RATE);
    let sample = |k: u64| (k as i32) * 97 - 2_000_000;
    out.write(&frames(0, total, sample), None).unwrap();
    out.drain(None).unwrap();
    let desk = mixer.desk();
    assert!(desk.master.passthrough);
    let strips: Vec<(&str, Option<u64>)> = desk
        .strips
        .iter()
        .map(|s| (s.name.as_str(), s.parent))
        .collect();
    assert_eq!(
        strips,
        [
            ("Player", None),
            ("Main", Some(group.id())),
            ("Preview", None)
        ]
    );
    assert_eq!((child.parent(), group.parent()), (Some(group.id()), None));
    assert_eq!(desk.strips[1].inputs.len(), 1);
    assert_eq!(desk.strips[0].inputs[0].format, None);
    drop(out);
    drop(mixer);
    let left = device.left(0);
    assert_eq!(left.len(), total as usize);
    let wrong = (0..total).find(|&k| left[k as usize] != sample(k));
    assert_eq!(wrong, None);
}

#[test]
fn a_group_meters_its_subtree() {
    let (mixer, _device) = instant_mixer();
    let group = mixer.strip("Player");
    let child = mixer.strip_in(&group, "Drums");
    let other = mixer.strip("Preview");
    child.set_level(0.5);
    let (mut a, _fa) = mixer.input(&child, Role::Lead, 1.0);
    let (mut b, _fb) = mixer.input(&group, Role::Guest, 1.0);
    let (mut c, _fc) = mixer.input(&other, Role::Guest, 1.0);
    let levels = [1_000_000, 2_000_000, 500_000];
    let mut outs = vec![
        open_started(&mut a, RATE, levels[0]),
        open_started(&mut b, RATE, levels[1]),
        open_started(&mut c, RATE, levels[2]),
    ];
    // Past the child's ramp to -6 dB, then a fresh look.
    let mut written = 1;
    play_to(&mixer, &mut a, &mut outs, &levels, &mut written, 4_800);
    mixer.meters();
    let data = levels.iter().map(|&l| frames(0, 19_200, |_| l)).collect();
    let _outs = play_all(outs, data);
    let desk = mixer.desk();
    let meters = mixer.meters();
    let full = 8_388_608.0f32;
    let ids: Vec<u64> = desk.strips.iter().map(|s| s.id).collect();
    assert_eq!(ids, [group.id(), child.id(), other.id()]);
    assert_eq!(
        meters.strips,
        [
            // The child's input at -6 dB and the group's own.
            (group.id(), vec![2_500_000.0 / full, 0.0]),
            (child.id(), vec![500_000.0 / full, 0.0]),
            (other.id(), vec![500_000.0 / full, 0.0]),
        ]
    );
    let master = meters.master[0];
    assert!(
        (master - 3_000_000.0 / full).abs() <= 2.0 / full,
        "{master}"
    );
}

/// The first frame of `left[from..to]` not within 1.5 of `want` (by frame).
fn off(left: &[i32], from: usize, to: usize, want: impl Fn(usize) -> f64) -> Option<(usize, i32)> {
    (from..to)
        .find(|&f| (f64::from(left[f]) - want(f)).abs() >= 1.5)
        .map(|f| (f, left[f]))
}

#[test]
fn a_group_mute_for_the_first_audio_starts_on_a_new_childs_first_frame() {
    let (mixer, device) = instant_mixer();
    let group = mixer.strip("Player");
    let (c1, c2) = (
        mixer.strip_in(&group, "Main"),
        mixer.strip_in(&group, "Vocals"),
    );
    let (mut a, _fa) = mixer.input(&c1, Role::Lead, 1.0);
    let (mut b, _fb) = mixer.input(&c2, Role::Guest, 1.0);
    let (la, lb) = (1_000_000, 2_000_000);
    let mut oa = open_started(&mut a, RATE, la);
    oa.write(&frames(1, 4_799, |_| la), None).unwrap();
    wait_until("mixed", || mixer.timebase().frame == 4_800);
    group.set_mute_at(true, Start::FirstAudio);
    taken(&mut a);
    // The older output plays on: nothing yet.
    oa.write(&frames(0, 4_800, |_| la), None).unwrap();
    wait_until("mixed", || mixer.timebase().frame == 9_600);
    assert!(!group.mute() && group.pending_mute() == Some(true));
    // A new output on the other child: from its first frame, 9 600.
    let mut ob = open_started(&mut b, RATE, lb);
    oa.write(&frames(0, 4_800, |_| la), None).unwrap();
    ob.write(&frames(1, 4_799, |_| lb), None).unwrap();
    wait_until("mixed", || mixer.timebase().frame == 14_400);
    assert!(group.mute() && group.pending_mute().is_none());
    drain(vec![oa, ob]);
    drop(mixer);
    let left = device.left(0);
    let start = 9_600;
    assert_eq!(left[..start].iter().position(|&v| v != la), None);
    let both = f64::from(la + lb);
    let fall = |f: usize| both * (1.0 - (f - start) as f64 / 960.0);
    assert_eq!(off(&left, start, start + 960, fall), None);
    assert_eq!(left[start + 960..14_400].iter().position(|&v| v != 0), None);
}

#[test]
fn passthrough_comes_back_exactly_after_a_group_ramp() {
    let (mixer, device) = instant_mixer();
    let group = mixer.strip("Player");
    let child = mixer.strip_in(&group, "Main");
    let (mut lead, _fader) = mixer.input(&child, Role::Lead, 1.0);
    let sample = |k: u64| (k as i32) * 97 - 2_000_000;
    let mut out = open_started(&mut lead, RATE, sample(0));
    out.write(&frames(1, 4_799, sample), None).unwrap();
    wait_until("mixed", || mixer.timebase().frame == 4_800);
    group.set_level(0.5);
    taken(&mut lead);
    out.write(&frames(4_800, 4_800, sample), None).unwrap();
    wait_until("mixed", || mixer.timebase().frame == 9_600);
    assert!(!mixer.passthrough());
    group.set_level(1.0);
    taken(&mut lead);
    out.write(&frames(9_600, 4_800, sample), None).unwrap();
    out.drain(None).unwrap();
    assert!(mixer.passthrough());
    drop(out);
    drop(mixer);
    let left = device.left(0);
    assert_eq!(left.len(), 14_400);
    let wrong = |from: usize, to: usize| (from..to).find(|&k| left[k] != sample(k as u64));
    // As written; down to -6 dB and back, mixed; then (the ramp back ends
    // on a period's edge, 9 600 + 960) as written again.
    assert_eq!(wrong(0, 4_800), None);
    let half = |f: usize| 0.5 * f64::from(sample(f as u64));
    assert_eq!(off(&left, 4_800 + 960, 9_600, half), None);
    assert_eq!(wrong(10_560, 14_400), None);
}

#[test]
fn a_change_for_a_time_of_an_epoch_gone_never_comes() {
    let (mixer, device) = instant_mixer();
    let group = mixer.strip("Player");
    let child = mixer.strip_in(&group, "Main");
    let (mut lead, _fader) = mixer.input(&child, Role::Lead, 1.0);
    let level = 1_000_000;
    let mut first = open_started(&mut lead, RATE, level);
    first.write(&frames(1, 4_799, |_| level), None).unwrap();
    wait_until("mixed", || mixer.timebase().frame == 4_800);
    let tb = mixer.timebase();
    group.set_mute_at(true, tb.at(20_000));
    taken(&mut lead);
    assert_eq!(group.pending_mute(), Some(true));
    // The Lead at another rate: a new epoch, and the mute waiting for a
    // frame of the last one goes; one set for it now never comes either.
    let mut out = open(&mut lead, 96_000);
    wait_until("a new epoch", || mixer.timebase().epoch == 2);
    assert_eq!(group.pending_mute(), None);
    group.set_mute_at(true, tb.at(10_000));
    taken(&mut lead);
    out.write(&frames(0, 48_000, |_| level), None).unwrap();
    out.drain(None).unwrap();
    assert!(!group.mute() && group.pending_mute().is_none());
    drop((first, out));
    drop(mixer);
    let left = device.left(1);
    assert_eq!(left.len(), 48_000);
    assert_eq!(left.iter().position(|&v| v != level), None);
}
