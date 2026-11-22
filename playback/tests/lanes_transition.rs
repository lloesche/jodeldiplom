//! A transition while a mixer plays the stems at other levels
//! ([`Command::StemMask`]): the two tracks are mixed inside the engine and
//! heard as main, with the mixer's stem gains as their mask, so nothing
//! jumps where the playing track's parts end (it moves onto its stems) nor
//! where the next track's come back (it moves back onto its file). The
//! separator gives each stem a quarter of the mix; the drums are muted on
//! the mixer and masked in the engine, so what is heard throughout is three
//! quarters of what the transition would play; a stem soloed, a quarter of
//! it; and a stem muted on a bar inside the window, half of it from there.

mod common;

use std::f64::consts::PI;
use std::path::{Path, PathBuf};

use common::lanes::*;
use common::*;
use jodel_output::lanes::Parts;
use jodel_output::SampleFormat::S24In32Low;
use jodel_playback::stems::STEMS;
use jodel_playback::{
    Envelope, Event, MakeStems, Player, Stage, TempoMode, TransitionPlan, TransitionState,
};

const FULL: f64 = 8_388_608.0;

/// A stereo WAV of `seconds` of two tones.
fn tone_wav(dir: &Path, name: &str, seconds: f64, hz: [f64; 2]) -> (PathBuf, Vec<i32>) {
    let path = dir.join(name);
    let frames = (seconds * f64::from(RATE)) as usize;
    let samples: Vec<i32> = (0..frames)
        .flat_map(|n| {
            let t = n as f64 / f64::from(RATE);
            [0, 1].map(|c| (0.4 * (2.0 * PI * hz[c] * t).sin() * FULL).round() as i32)
        })
        .collect();
    write_wav(&path, RATE, 2, &samples);
    (path, samples)
}

fn plan() -> TransitionPlan {
    let lanes = std::array::from_fn(|i| {
        if i < STEMS {
            Envelope::fade(1.0, 0.0, 0.0, 1.0)
        } else {
            Envelope::fade(0.0, 1.0, 0.0, 1.0)
        }
    });
    TransitionPlan {
        from: 0,
        into: 1,
        start: 5.0,
        length: 2.0,
        entry: 1.0,
        beats: (0..5)
            .map(|i| (5.0 + 0.5 * i as f64, 1.0 + 0.5 * i as f64))
            .collect(),
        lanes,
        tempo: TempoMode::Varispeed,
        settle: 1.0,
        glide: false,
    }
}

/// How the stems stand on the mixer and in the engine.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Stems {
    /// The drums muted throughout.
    DrumsMuted,
    /// The vocals soloed throughout.
    VocalsSoloed,
    /// The drums muted, and the vocals too from 6 s of the playing track,
    /// a bar inside the window.
    VocalsOnABar,
}

/// The second of the playing track the vocals go off at in
/// [`Stems::VocalsOnABar`].
const BAR: f64 = 6.0;

/// The share of the transition heard at output frame `f`.
fn share_heard(stems: Stems, f: usize) -> f64 {
    match stems {
        Stems::DrumsMuted => 0.75,
        Stems::VocalsSoloed => 0.25,
        Stems::VocalsOnABar if (f as f64) < BAR * f64::from(RATE) => 0.75,
        Stems::VocalsOnABar => 0.5,
    }
}

/// Plays the transition with the stems as `stems` has them, and checks
/// every frame against that share of what it plays.
fn never_jumps(stems: Stems) {
    let dir = temp_dir("lanes-transition");
    let (a_path, a) = tone_wav(&dir, "a.wav", 12.0, [1_000.0, 5_000.0]);
    let (b_path, b) = tone_wav(&dir, "b.wav", 12.0, [2_000.0, 3_000.0]);
    let (target, captures, gate, lanes) = gated_mixer_lanes(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options(false, true, quarter));
    player.make_stems(MakeStems::Separate).unwrap();
    // As on the mixer, so in what the engine mixes.
    match stems {
        Stems::DrumsMuted | Stems::VocalsOnABar => {
            lanes.parts[0].set_mute(true);
            player.stem_mask([0.0, 1.0, 1.0, 1.0], None).unwrap();
        }
        Stems::VocalsSoloed => {
            lanes.parts[2].set_solo(true);
            player.stem_mask([0.0, 0.0, 1.0, 0.0], None).unwrap();
        }
    }
    player.load(vec![a_path, b_path]).unwrap();
    gate.allow(2_048);
    player.play(0).unwrap();
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    let plan = plan();
    player.transition(plan.clone()).unwrap();
    let mut seen = Vec::new();
    if stems == Stems::VocalsOnABar {
        // In the window, before the bar is written: the vocals off from
        // the bar in the engine's mix, and on the mixer for after it.
        seen = pace_to(&gate, &events, 8_192, (5.2 * f64::from(RATE)) as u64);
        player.stem_mask([0.0, 1.0, 0.0, 1.0], Some(BAR)).unwrap();
        lanes.parts[2].set_mute(true);
    }
    seen.extend(pace(&gate, &events, 8_192));
    // The next track's stems are parts again at the end, and heard.
    let desk = lanes.mixer.desk();
    let input = &desk.strips[0].inputs[0];
    let parts = input.parts.expect("an input of lanes");
    assert_eq!(
        (parts.state, parts.share),
        (Parts::Present, 1.0),
        "{stems:?}"
    );
    drop(player);
    drop(lanes);
    let states: Vec<&TransitionState> = seen
        .iter()
        .filter_map(|e| match e {
            Event::Transition(s) => Some(s),
            _ => None,
        })
        .collect();
    assert!(states.contains(&&TransitionState::Finished), "{states:?}");
    let starts: Vec<_> = seen
        .iter()
        .filter_map(|e| match e {
            Event::TrackStarted(s) => Some((s.index, s.route.stage, s.output_frame)),
            _ => None,
        })
        .collect();
    assert_eq!(
        starts.last().map(|s| (s.0, s.1)),
        Some((1, Stage::Passthrough)),
        "B back on its file: {starts:?}"
    );
    let streams = captured(&captures);
    assert_eq!(streams.len(), 1, "one stream throughout");
    let got = device_samples(&streams[0].1);
    let rate = RATE as usize;
    assert_eq!(got.len() / 2, 16 * rate, "A to 7 s, then B from 3 s");
    // The share of the transition, on every frame: A alone, A on its
    // stems (from 1 s), the window, B on its stems, B on its file again.
    let offset = 4 * rate;
    let (mut worst, mut at) = (0.0f64, 0);
    // (Not where the window starts and ends: its steps there are as
    // without stems muted, `tests/transition.rs`; nor over the 10 ms a
    // stem goes off in on its bar.)
    let bar = (BAR * f64::from(RATE)) as usize;
    let edge = |f: usize| {
        [5 * rate, 7 * rate].iter().any(|&w| f.abs_diff(w) < 200)
            || (stems == Stems::VocalsOnABar && f.abs_diff(bar) < 600)
    };
    for f in (200..got.len() / 2 - 200).filter(|&f| !edge(f)) {
        let share = plan.share(f as f64 / f64::from(RATE));
        let (ga, gb) = (plan.lanes[0].gain(share), plan.lanes[STEMS].gain(share));
        for c in 0..2 {
            let a = a.get(2 * f + c).map_or(0.0, |&s| f64::from(s));
            let b = (f >= offset).then(|| f64::from(b[2 * (f - offset) + c]));
            let want = share_heard(stems, f) * (ga * a + gb * b.unwrap_or(0.0));
            let off = (f64::from(got[2 * f + c]) - want).abs();
            if off > worst {
                (worst, at) = (off, f);
            }
        }
    }
    // Within what the window's stepped gains leave (half a percent of full
    // scale): a stem's quarter heard where it should not be, for even a few
    // milliseconds, is ten times that.
    assert!(
        worst < 0.005 * FULL,
        "{stems:?}: off by {:.4} of full scale at {:.4} s; starts {starts:?}",
        worst / FULL,
        at as f64 / f64::from(RATE)
    );
}

#[test]
fn a_transition_with_a_stem_muted_never_jumps() {
    never_jumps(Stems::DrumsMuted);
}

#[test]
fn a_transition_with_a_stem_soloed_never_goes_silent() {
    never_jumps(Stems::VocalsSoloed);
}

#[test]
fn a_stem_muted_on_a_bar_in_the_window_goes_off_there() {
    never_jumps(Stems::VocalsOnABar);
}

/// The next track goes back onto its file only once its stems have been
/// parts of the output for a mixer's switch and its end: the mixer has
/// switched to them by then, so the move is not heard.
#[test]
fn the_next_tracks_return_waits_until_its_stems_are_switched_to() {
    use jodel_output::lanes::{ending_frames, SWITCH_FADE};
    let dir = temp_dir("lanes-transition-return");
    let (a_path, _) = tone_wav(&dir, "a.wav", 12.0, [1_000.0, 5_000.0]);
    let (b_path, _) = tone_wav(&dir, "b.wav", 12.0, [2_000.0, 3_000.0]);
    let (target, captures, gate) = wide_target(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options(false, true, quarter));
    player.make_stems(MakeStems::Separate).unwrap();
    player.stem_mask([0.0, 1.0, 1.0, 1.0], None).unwrap();
    player.load(vec![a_path, b_path]).unwrap();
    gate.allow(2_048);
    player.play(0).unwrap();
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    player.transition(plan()).unwrap();
    let seen = pace(&gate, &events, 8_192);
    drop(player);
    let back = seen
        .iter()
        .filter_map(|e| match e {
            Event::TrackStarted(s) if s.route.stage == Stage::Passthrough => {
                Some(s.output_frame as usize)
            }
            _ => None,
        })
        .next_back()
        .expect("B back on its file");
    let wide = Wide::of(&captures);
    // The parts again after the window (7 s), on B's stems.
    let present = (7 * RATE as usize..wide.frames())
        .find(|&f| wide.control[f] == 1)
        .expect("B's parts again");
    let settled = ending_frames(RATE) as usize
        + (SWITCH_FADE.as_secs_f64() * f64::from(RATE)).round() as usize;
    assert!(
        back >= present + settled,
        "back on its file at {back}, the parts there from {present}"
    );
    assert!(
        wide.control[present..].iter().all(|&w| w == 1),
        "and there from then on"
    );
}
