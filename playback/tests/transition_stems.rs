//! The stems a transition mixes, where they can go wrong: a separator that
//! fails (the transitions after it fade their tracks whole, until the mix
//! changes), and a high output rate (the stems resampled to it, a track on
//! them ending inside the window on its own last frame).

mod common;

use common::lanes::fails_on_silence;
use common::transition::*;
use common::*;
use jodel_output::{SampleFormat, SimClock};
use jodel_playback::stems::STEMS;
use jodel_playback::{
    Event, MemoryTarget, Player, PlayerOptions, TempoMode, TransitionPlan, TransitionState,
};
use SampleFormat::*;

/// From `start` s of track `from` over 2 s into the next at 1 s, faded,
/// settling over a second.
fn plan(from: usize, start: f64) -> TransitionPlan {
    TransitionPlan {
        from,
        into: from + 1,
        start,
        length: 2.0,
        entry: 1.0,
        beats: beats(start, 0.5, 1.0, 0.5, 5),
        lanes: fade_lanes(),
        tempo: TempoMode::Varispeed,
        settle: 1.0,
        glide: false,
    }
}

fn fade_lanes() -> [jodel_playback::Envelope; 2 * STEMS] {
    use jodel_playback::Envelope;
    lanes(
        Envelope::fade(1.0, 0.0, 0.0, 1.0),
        Envelope::fade(0.0, 1.0, 0.0, 1.0),
    )
}

/// The tracks whose stems stopped, and whether a transition finished and
/// none was called off, in `seen`.
fn stopped_and_finished(seen: &[Event]) -> (Vec<usize>, bool) {
    let stopped = seen
        .iter()
        .filter_map(|e| match e {
            Event::StemsStopped { index, .. } => Some(*index),
            _ => None,
        })
        .collect();
    let states = transitions(seen);
    let cancelled = states
        .iter()
        .any(|s| matches!(s, TransitionState::Cancelled { .. }));
    let finished = states.contains(&&TransitionState::Finished);
    (stopped, finished && !cancelled)
}

/// Three transitions with a separator that fails on silence. The first:
/// the playing track's stems fail before its window (it is silent there),
/// it goes back onto its file and fades whole, and the transition
/// finishes. The second takes both its tracks whole, so the next one,
/// silent, separates nothing and fails nothing. The mix changed after it:
/// the third separates again, and the silent track's stems fail as it
/// moves onto them; that transition finishes too.
#[test]
fn after_a_separator_fails_transitions_take_their_tracks_whole_until_the_mix_changes() {
    let _serial = serial();
    let dir = temp_dir("transition-stems-failed");
    let tone = |t: f64, _c: usize| 0.3 * (2.0 * std::f64::consts::PI * 440.0 * t).sin();
    let (a, _) = wav(
        &dir,
        "a.wav",
        12.0,
        move |t, c| {
            if t < 4.8 {
                tone(t, c)
            } else {
                0.0
            }
        },
    );
    let (b, _) = wav(&dir, "b.wav", 12.0, tone);
    let (c, _) = wav(&dir, "c.wav", 12.0, |_, _| 0.0);
    let (d, _) = wav(&dir, "d.wav", 12.0, tone);
    let (target, _captures, gate) = GatedTarget::new(caps(&[RATE], &[S24In32Low]));
    let options = PlayerOptions {
        separator: Some(fails_on_silence),
        ..options()
    };
    let (player, events) = Player::new(Box::new(target), options);
    player.load(vec![a, b, c, d]).unwrap();
    gate.allow(2_048);
    player.play(0).unwrap();
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    let finished = |e: &Event| matches!(e, Event::Transition(TransitionState::Finished));
    player.transition(plan(0, 7.0)).unwrap();
    let first = step_until(&gate, &events, finished);
    assert_eq!(stopped_and_finished(&first), (vec![0], true), "{first:?}");
    player.transition(plan(1, 7.0)).unwrap();
    let second = step_until(&gate, &events, finished);
    assert_eq!(stopped_and_finished(&second), (vec![], true), "{second:?}");
    // Another mix, outside a transition: stems are tried again.
    player.stem_mask([1.0, 1.0, 0.0, 0.0], None).unwrap();
    player.transition(plan(2, 7.0)).unwrap();
    let third = step_to_end(&gate, &events);
    drop(player);
    assert_eq!(stopped_and_finished(&third), (vec![2], true), "{third:?}");
}

/// The output's rate.
const HIGH: u32 = 192_000;

/// A transition at 192 kHz whose playing track ends inside the window: its
/// move onto its stems waits for their first chunk (its run-up, a second,
/// is more than a switch keeps of a route that has not caught up), and on
/// them it ends on its own last frame, where the next track goes on, at
/// lengths whose stems once came out two frames short and two over.
#[test]
fn at_a_high_rate_a_track_on_its_stems_ends_on_its_last_frame() {
    let _serial = serial();
    let dir = temp_dir("transition-stems-high");
    for frames in [
        4 * HIGH as usize,
        4 * HIGH as usize - 512,
        4 * HIGH as usize - 320,
    ] {
        let a = dir.join(format!("a-{frames}.wav"));
        write_wav(&a, HIGH, 2, &tones(HIGH, frames));
        let b = dir.join(format!("b-{frames}.wav"));
        write_wav(&b, HIGH, 2, &tones(HIGH, 4 * HIGH as usize));
        let target = MemoryTarget::new(caps(&[HIGH], &[S24In32Low]), SimClock::Instant);
        let options = PlayerOptions {
            settings: jodel_playback::Settings::default(),
            ..options()
        };
        let (player, events) = Player::new(Box::new(target), options);
        player.load(vec![a, b]).unwrap();
        player.play(0).unwrap();
        collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
        player
            .transition(TransitionPlan {
                start: 2.5,
                entry: 0.0,
                beats: Vec::new(),
                settle: 0.0,
                ..plan(0, 2.5)
            })
            .unwrap();
        let seen = until_queue_end(&events);
        drop(player);
        let (stopped, finished) = stopped_and_finished(&seen);
        assert!(stopped.is_empty() && finished, "{frames}: {seen:?}");
        assert!(!seen.iter().any(|e| matches!(e, Event::Error { .. })));
        let next = starts(&seen)
            .into_iter()
            .find(|s| s.index == 1)
            .expect("the next track");
        assert_eq!(next.output_frame as usize, frames, "{frames} frames");
    }
}
