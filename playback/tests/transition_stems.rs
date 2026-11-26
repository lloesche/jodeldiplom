//! A transition's stems where they can go wrong: a separator that fails
//! (the track's parts stop, and the desk fades it whole by its lanes'
//! mean), and a high output rate (the next track's stems resampled to it,
//! the playing track ending inside the window on its own last frame).

mod common;

use common::lanes::fails_on_silence;
use common::transition::*;
use common::*;
use jodel_output::{SampleFormat, SimClock};
use jodel_playback::stems::STEMS;
use jodel_playback::{
    Event, MakeStems, Player, PlayerOptions, TempoMode, TransitionPlan, TransitionState,
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

/// The tracks whose parts stopped, and whether a transition finished and
/// none was called off, in `seen`.
fn stopped_and_finished(seen: &[Event]) -> (Vec<usize>, bool) {
    let mut stopped: Vec<usize> = seen
        .iter()
        .filter_map(|e| match e {
            Event::PartsStopped { index, .. } => Some(*index),
            _ => None,
        })
        .collect();
    stopped.dedup();
    let states = transitions(seen);
    let cancelled = states
        .iter()
        .any(|s| matches!(s, TransitionState::Cancelled { .. }));
    let finished = states.contains(&&TransitionState::Finished);
    (stopped, finished && !cancelled)
}

/// Two transitions with a separator that fails on silence. The first: the
/// playing track's stems fail (it starts silent): its parts stop, the desk
/// hears it on main, faded by the mean of its lanes, and the transition
/// finishes. The second goes into a silent track, whose stems fail as it is
/// read along its map: it plays on the whole mix as one part, and that
/// transition finishes too.
#[test]
fn a_separator_that_fails_leaves_the_transitions_as_planned() {
    let _serial = serial();
    let dir = temp_dir("transition-stems-failed");
    let sine = |t: f64, _c: usize| 0.3 * (2.0 * std::f64::consts::PI * 440.0 * t).sin();
    // Silent through the separator's first chunk (6 s), then a tone.
    let (a, a_samples) = wav(
        &dir,
        "a.wav",
        14.0,
        move |t, c| if t < 6.5 { 0.0 } else { sine(t, c) },
    );
    let (b, _) = wav(&dir, "b.wav", 12.0, sine);
    let (c, _) = wav(&dir, "c.wav", 12.0, |_, _| 0.0);
    let options = PlayerOptions {
        separator: Some(fails_on_silence),
        ..options()
    };
    let desk = Desk::new(caps(&[RATE], &[S24In32Low]), options);
    desk.player.make_stems(MakeStems::Separate).unwrap();
    desk.play(vec![a, b, c]);
    let finished = |e: &Event| matches!(e, Event::Transition(TransitionState::Finished));
    // A's drums fade out, its other stems stay; B is not heard.
    use jodel_playback::Envelope;
    let mut first = plan(0, 9.0);
    first.lanes = std::array::from_fn(|i| match i {
        0 => Envelope::fade(1.0, 0.0, 0.0, 1.0),
        i if i < STEMS => Envelope::flat(1.0),
        _ => Envelope::flat(0.0),
    });
    desk.player.transition(first.clone()).unwrap();
    let seen = step_until(&desk.gate, &desk.events, finished);
    assert_eq!(stopped_and_finished(&seen), (vec![0], true), "{seen:?}");
    desk.player.transition(plan(1, 7.0)).unwrap();
    let second = step_to_end(&desk.gate, &desk.events);
    let streams = desk.close();
    assert_eq!(stopped_and_finished(&second), (vec![2], true), "{second:?}");
    assert!(!second.iter().any(|e| matches!(e, Event::Error { .. })));
    // In the first window A is heard at its lanes' mean.
    let got = samples(&streams[0].1);
    let rate = RATE as usize;
    let off = (9 * rate + 200..11 * rate - 200)
        .flat_map(|f| (0..2).map(move |c| (f, c)))
        .map(|(f, c)| {
            let share = first.share(f as f64 / f64::from(RATE));
            let mean = (0..STEMS).map(|k| first.lanes[k].gain(share)).sum::<f64>() / 4.0;
            (f64::from(got[2 * f + c]) - mean * f64::from(a_samples[2 * f + c])).abs()
        })
        .fold(0.0, f64::max);
    assert!(off < 0.005 * FULL, "A off its lanes' mean by {off}");
}

/// The output's rate.
const HIGH: u32 = 192_000;

/// A transition at 192 kHz whose playing track ends inside the window: the
/// next track, its stems resampled to the output's rate along its map,
/// goes on on the playing one's last frame, at lengths whose stems once
/// came out two frames short and two over.
#[test]
fn at_a_high_rate_the_playing_track_ends_on_its_last_frame() {
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
        let (lead, guest, _captures, gate, _desk, mixer) =
            gated_desk(caps(&[HIGH], &[S24In32Low]), SimClock::Instant);
        gate.allow_all();
        let options = PlayerOptions {
            settings: jodel_playback::Settings::default(),
            ..options()
        };
        let (player, events) = Player::on_desk(lead, guest, options);
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
        drop(mixer);
        let (stopped, finished) = stopped_and_finished(&seen);
        assert!(stopped.is_empty() && finished, "{frames}: {seen:?}");
        assert!(!seen.iter().any(|e| matches!(e, Event::Error { .. })));
        let next = started(&seen)
            .into_iter()
            .find(|s| s.index == 1)
            .expect("the next track");
        // Its frames counted from its first, 1.5 s of A's in.
        let origin = (1.5 * f64::from(HIGH)) as u64;
        assert_eq!(
            (next.output_frame + origin) as usize,
            frames,
            "{frames} frames"
        );
    }
}
