//! Transitions: the playing track and the next one heard together over a
//! window, each faded by its lanes, the next one's beats on the playing
//! one's, then the next one on its own and back on its own route. The
//! separator gives each stem a quarter of the mix, so the stems of a track
//! add up to the track, and what comes out can be worked out.

mod common;

use std::time::Duration;

use common::transition::*;
use common::*;
use jodel_output::SampleFormat;
use jodel_playback::stems::STEMS;
use jodel_playback::{
    Envelope, Event, Player, PlayerOptions, Stage, TempoMode, TransitionPlan, TransitionState,
};
use SampleFormat::*;

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
    let seen = step_to_end(&gate, &events);
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
    let seen = step_to_end(&gate, &events);
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
    glide_meets(TempoMode::Varispeed, 0.001, ALL, false);
}

#[test]
fn stretched_gliding_the_tempo_meets_the_beats_too() {
    let _serial = serial();
    glide_meets(TempoMode::Stretch, 0.004, ALL, false);
}

/// The same with two stems muted on a mixer's strips
/// ([`jodel_playback::Command::StemMask`]): the clicks at half their level.
#[test]
fn gliding_with_stems_muted_on_a_mixer_meets_the_beats_too() {
    let _serial = serial();
    glide_meets(TempoMode::Varispeed, 0.001, drums_bass(), false);
}

/// The same with the plan replaced after A has moved onto its stems for one
/// without a glide: A goes onto a fresh stems track read along the glide.
#[test]
fn a_glide_planned_once_on_its_stems_meets_the_beats_too() {
    let _serial = serial();
    glide_meets(TempoMode::Varispeed, 0.001, ALL, true);
}

/// A at 120 BPM, B at 126, the tempo gliding across a 4 s window: A alone
/// through its first half, B alone through its second. Every click (A's,
/// then B's) is where A's beat is heard along the glide, within `within`
/// seconds; A's are closer together as it goes; after the window B is at
/// its own tempo at once; and the position heard follows the glide. Each
/// stem at its gain in `mask`, as a mixer plays them; with `replan`, the
/// plan comes once A is on its stems for a plan without the glide.
fn glide_meets(tempo: TempoMode, within: f64, mask: [f64; STEMS], replan: bool) {
    let dir = temp_dir(&format!(
        "transition-glide-{tempo:?}-{}-{replan}",
        mask == ALL
    ));
    let b_period = 60.0 / 126.0;
    let (a_path, _) = wav(&dir, "a.wav", 14.0, clicks(0.5, 0.0, 3_000.0));
    let (b_path, _) = wav(&dir, "b.wav", 14.0, clicks(b_period, 0.0, 5_000.0));
    let (target, captures, gate) = GatedTarget::new(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options());
    player.stem_mask(mask, None).unwrap();
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
    if replan {
        // First without the glide, until A is on its stems.
        let flat = TransitionPlan {
            glide: false,
            ..plan.clone()
        };
        player.transition(flat).unwrap();
        step_until(
            &gate,
            &events,
            |e| matches!(e, Event::TrackStarted(s) if s.index == 0 && s.route.stage == Stage::Stems),
        );
    }
    player.transition(plan).unwrap();
    // The output held at 2.5 s for a while, as a device playing in real
    // time leaves A's move onto its stems (from 1 s on) the time it needs
    // (the trickle goes several times faster than that).
    gate.allow(5 * u64::from(RATE) / 2);
    gate.wait_written(5 * u64::from(RATE) / 2);
    std::thread::sleep(Duration::from_millis(1_500));
    let seen = step_to_end(&gate, &events);
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
    let mut seen = step_until(
        &gate,
        &events,
        |e| matches!(e, Event::Position(p) if p.output_frame >= 7 * u64::from(RATE)),
    );
    player.remove(1).unwrap();
    seen.extend(step_to_end(&gate, &events));
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
    let seen = step_to_end(&gate, &events);
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
    let seen = step_to_end(&gate, &events);
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
    step_until(
        &gate,
        &events,
        |e| matches!(e, Event::Position(p) if p.index == 0 && p.seconds >= 6.0),
    );
    player.next().unwrap();
    let seen = step_to_end(&gate, &events);
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
    let seen = step_to_end(&gate, &events);
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
