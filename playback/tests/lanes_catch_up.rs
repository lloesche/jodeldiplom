//! The stems asked for mid-track (the first touch of a stem's strip): the
//! frames the output holds, written without parts, are written again with
//! them from the first frame it gives back, as soon as they are made (kept
//! stems at once, separated ones once the first chunk is), main bit for
//! bit and the control word's promise kept across, also when the output
//! keeps some of what it holds; never over a jump's crossfade; through a
//! mixer, a stem muted a quarter of a second after what is heard is heard
//! muted there; and a jump back into what is kept for that plays as it
//! does without lanes.

mod common;

use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use common::lanes::*;
use common::*;
use jodel_output::mixer::{Mixer, MixerOptions, Role, STRIP_RAMP};
use jodel_output::{SampleFormat, SimClock};
use jodel_playback::stems::Stem;
use jodel_playback::{
    Captures, Event, JumpAt, MakeStems, MemoryTarget, Player, PlayerOptions, StemDesk,
};
use jodel_testkit::{heavy, real_time, TempDir};
use SampleFormat::*;

/// Frames a held device lets the engine write ahead of what it played.
const ROOM: u64 = RATE as u64 / 4;

/// Waits until `played` frames have been played and the device holds a
/// room's worth more.
fn wait_held(hold: &Hold, played: u64) {
    let deadline = Instant::now() + TIMEOUT;
    while hold.played() < played || (hold.held_words().len() as u64) < ROOM {
        assert!(Instant::now() < deadline, "not played and held");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Waits until frame `k` of those the device holds says the parts are
/// there; returns how long that took.
fn wait_present(hold: &Hold, k: usize) -> Duration {
    let start = Instant::now();
    while hold.held_words().get(k) != Some(&1) {
        assert!(start.elapsed() < TIMEOUT, "the parts never came");
        std::thread::sleep(Duration::from_millis(1));
    }
    start.elapsed()
}

/// A player on a held device of lanes playing `file`, a second of it
/// played and the device holding a room's worth more.
fn held_player(
    file: &std::path::Path,
    options: PlayerOptions,
) -> (Player, Receiver<Event>, Captures, std::sync::Arc<Hold>) {
    let device = MemoryTarget::with_lanes(caps(&[RATE], &[S24In32Low]), SimClock::Instant, 5);
    let captures = device.captures();
    let (target, hold) = HoldingTarget::around(Box::new(device), ROOM);
    let (player, events) = Player::new(Box::new(target), options);
    player.load(vec![file.to_path_buf()]).unwrap();
    player.play(0).unwrap();
    hold.play(u64::from(RATE));
    wait_held(&hold, u64::from(RATE));
    (player, events, captures, hold)
}

/// Kept stems for `samples` (written to a file in `dir`), a broken
/// separator besides.
fn kept_options(dir: &std::path::Path, samples: &[i32]) -> (std::path::PathBuf, PlayerOptions) {
    let file = dir.join("tones.wav");
    write_wav(&file, RATE, 2, samples);
    let options = PlayerOptions {
        stem_store: Some(kept(dir, &file, samples)),
        ..options(false, broken)
    };
    (file, options)
}

/// Kept stems asked for a second in: what the device holds is written again
/// with them, from the first frame it gives back (all it holds, or all but
/// the `keep` frames it plays next, as a mixer's input keeps a few
/// periods).
#[test]
fn kept_stems_asked_for_mid_track_are_there_from_the_first_frame_given_back() {
    let _heavy = heavy();
    for keep in [0, 3 * 1_024] {
        let dir = TempDir::new(&format!("catch-up-kept-{keep}"));
        let samples = tones(RATE, 4 * RATE as usize);
        let (file, options) = kept_options(&dir, &samples);
        let (player, events, captures, hold) = held_player(&file, options);
        assert!(hold.held_words().iter().all(|&w| w == 0), "no parts yet");
        hold.keep_on_retract(keep);
        player.make_stems(MakeStems::Kept).unwrap();
        let took = wait_present(&hold, keep as usize);
        println!(
            "keeping {keep}: kept stems there {:.1} ms after they were asked for",
            took.as_secs_f64() * 1e3
        );
        let words = hold.held_words();
        let keep = keep as usize;
        assert!(
            words[..keep].iter().all(|&w| w == 0),
            "{keep}: kept as it was"
        );
        assert!(
            words[keep..].iter().all(|&w| w == 1),
            "{keep}: written again"
        );
        hold.play_all();
        let seen = until_queue_end(&events);
        drop(player);
        assert!(!seen.iter().any(|e| matches!(e, Event::PartsStopped { .. })));
        let wide = Wide::of(&captures);
        assert!(wide.main == samples, "{keep}: main is the file");
        wide.keeps_the_promise(&[]).unwrap();
        // (Past what was held the parts may run out on a loaded machine, the
        // device playing all at once.)
        let first = RATE as usize + keep;
        assert!(wide.control[..first].iter().all(|&c| c == 0));
        let held = RATE as usize + ROOM as usize;
        assert!(wide.control[first..held].iter().all(|&c| c == 1));
        let (worst, _) = wide.worst(first..wide.frames());
        assert!(worst <= 4 * 256, "{keep}: {worst} steps off");
    }
}

#[test]
fn separated_stems_are_there_once_the_first_chunk_is() {
    let _heavy = heavy();
    let dir = TempDir::new("catch-up-separated");
    let file = tone_file(&dir, "tones.wav", RATE, 10 * RATE as usize);
    FIRST_CHUNK.allow(0);
    let options = options(false, first_chunk);
    let (player, events, captures, hold) = held_player(&file, options);
    player.make_stems(MakeStems::Separate).unwrap();
    // The separation held up: a quarter of a second more plays, and what
    // is written meanwhile has no parts.
    let heard = u64::from(RATE) * 5 / 4;
    hold.play(heard - u64::from(RATE));
    wait_held(&hold, heard);
    assert!(hold.held_words().iter().all(|&w| w == 0), "no parts yet");
    // Once separated, all that is held is written again with them.
    FIRST_CHUNK.allow(usize::MAX);
    let took = wait_present(&hold, 0);
    println!(
        "separated stems there {:.1} ms after the separator was let go",
        took.as_secs_f64() * 1e3
    );
    assert!(hold.held_words().iter().all(|&w| w == 1));
    hold.play_all();
    let seen = until_queue_end(&events);
    drop(player);
    assert!(!seen.iter().any(|e| matches!(e, Event::PartsStopped { .. })));
    let wide = Wide::of(&captures);
    assert!(wide.main == read_all(&file), "main is the file");
    wide.keeps_the_promise(&[]).unwrap();
    let heard = heard as usize;
    assert!(wide.control[..heard].iter().all(|&c| c == 0));
    assert_eq!(wide.control[heard], 1, "there from the frame heard next");
    // (A separation fades in over its first frames.)
    let (worst, _) = wide.worst(heard + RATE as usize / 100..wide.frames());
    assert!(worst <= CLOSE, "{worst} steps off");
}

/// A jump back met from the buffer (its first frames crossfaded), then kept
/// stems asked for while the device holds that crossfade: the frames
/// written again start after it, so main is what it is with no stems asked
/// for.
#[test]
fn what_a_jump_crossfaded_is_not_written_again() {
    let _heavy = heavy();
    let dir = TempDir::new("catch-up-crossfade");
    let samples = tones(RATE, 3 * RATE as usize);
    let (file, options) = kept_options(&dir, &samples);
    let run = |ask: bool| {
        let (player, events, captures, hold) = held_player(&file, options.clone());
        player.jump(0, 0.9, JumpAt::Now).unwrap();
        collect_until(&events, |e| matches!(e, Event::Jumped { .. }));
        wait_held(&hold, u64::from(RATE));
        if ask {
            player.make_stems(MakeStems::Kept).unwrap();
            // (The crossfade, 20 ms, is held first: then the parts.)
            wait_present(&hold, RATE as usize / 50);
            assert_eq!(hold.held_words()[0], 0, "the crossfade as it was");
        }
        hold.play_all();
        until_queue_end(&events);
        drop(player);
        captures
    };
    let without = Wide::of(&run(false));
    let with = Wide::of(&run(true));
    assert!(without.control.iter().all(|&c| c == 0));
    assert!(with.main == without.main, "the same main");
    with.keeps_the_promise(&[]).unwrap();
}

/// Through a mixer on a device that plays in real time: kept stems not
/// made until a stem's strip is touched, the drums muted a quarter of a
/// second after what is heard then (on a small loaded machine the parts
/// take a few tens of milliseconds, and the mixer mixes ahead). The file
/// is heard bit for bit up to that frame, and from there (after the
/// mute's ramp) the band without the drums: the touch is heard on its
/// frame, not after the engine's ring and the parts made ahead (a second).
#[test]
fn a_first_touch_is_heard_on_its_frame_through_a_mixer() {
    let _guard = real_time();
    let dir = TempDir::new("catch-up-mixer");
    let samples = tones(RATE, 3 * RATE as usize);
    let (file, options) = kept_options(&dir, &samples);
    let device = MemoryTarget::new(caps(&[RATE], &[S24In32Low]), SimClock::Paced);
    let captures = device.captures();
    let mixer = Mixer::open(Box::new(device), MixerOptions::default());
    let (desk, head) = StemDesk::new(&mixer, "Player", Role::Lead, 1.0);
    let (player, events) = Player::new(head.target, options);
    player.load(vec![file]).unwrap();
    player.play(0).unwrap();
    let heard = |e: &Event| match e {
        Event::Position(p) if p.output_frame >= u64::from(RATE) / 2 => Some(p.output_frame),
        _ => None,
    };
    let seen = collect_until(&events, |e| heard(e).is_some());
    let now = seen.iter().rev().find_map(heard).expect("a position");
    player.make_stems(MakeStems::Kept).unwrap();
    let at = now + u64::from(RATE) / 4;
    desk.stem(Stem::Drums)
        .set_mute_at(true, head.faders.input.at_stream(at));
    let seen = until_queue_end(&events);
    drop(player);
    drop(desk);
    drop(mixer);
    assert!(
        !seen
            .iter()
            .any(|e| matches!(e, Event::PartsStopped { .. } | Event::Error { .. })),
        "{seen:?}"
    );
    let streams = captured(&captures);
    assert_eq!(streams.len(), 1);
    let got = device_samples(&streams[0].1);
    assert_eq!(got.len(), samples.len());
    let at = at as usize;
    assert!(
        got[..2 * at] == samples[..2 * at],
        "the file up to the touch"
    );
    // Three quarters of the file once the mute's ramp is over (the mixer's
    // switch to the parts goes with it).
    let switched = at + frames_of(STRIP_RAMP);
    let off = (2 * switched..got.len())
        .map(|i| (f64::from(got[i]) - 0.75 * f64::from(samples[i])).abs())
        .fold(0.0, f64::max);
    assert!(
        off <= 4.0 * 256.0,
        "{off} steps off three quarters from the touch"
    );
}

/// What the track keeps behind what it wrote for the lanes to write again
/// meets a jump a little back without a jump window (an output without
/// lanes reads the track there again): main is the same either way.
#[test]
fn a_jump_back_into_what_is_kept_for_the_lanes_plays_the_same() {
    let _heavy = heavy();
    let dir = TempDir::new("catch-up-jumps");
    let samples = tones(RATE, 3 * RATE as usize);
    let (file, options) = kept_options(&dir, &samples);
    let run = |target: GatedTarget, gate: &Gate| {
        let (player, events) = Player::new(Box::new(target), options.clone());
        player.load(vec![file.clone()]).unwrap();
        gate.allow(u64::from(RATE));
        player.play(0).unwrap();
        gate.wait_written(u64::from(RATE));
        player.jump(0, 0.7, JumpAt::Now).unwrap();
        collect_until(&events, |e| matches!(e, Event::Jumped { .. }));
        gate.allow_all();
        until_queue_end(&events);
    };
    let (target, captures, gate) = GatedTarget::new(caps(&[RATE], &[S24In32Low]));
    run(target, &gate);
    let direct = device_samples(&captured(&captures)[0].1);
    let (target, captures, gate) = wide_target(caps(&[RATE], &[S24In32Low]));
    run(target, &gate);
    let laned = Wide::of(&captures);
    assert!(laned.control.iter().all(|&c| c == 0), "no stems asked for");
    assert_eq!(direct.len(), laned.main.len());
    assert!(direct == laned.main, "the same jump");
}

fn frames_of(d: Duration) -> usize {
    (d.as_secs_f64() * f64::from(RATE)).round() as usize
}
