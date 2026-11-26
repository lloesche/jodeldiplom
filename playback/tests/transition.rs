//! Transitions on a desk: the playing track and the next one, each on a
//! head of its own, heard together over a window, each faded by its lanes
//! (the desk's automation), the next one's beats on the playing one's, then
//! the next one on its own and back on its own route. The separator gives
//! each stem a quarter of the mix, so the stems of a track add up to the
//! track, and what comes out can be worked out.

mod common;

use std::time::Duration;

use common::transition::*;
use common::*;
use jodel_output::{SampleFormat, SimClock};
use jodel_playback::stems::STEMS;
use jodel_playback::warp::TimeMap;
use jodel_playback::{
    Envelope, Event, MakeStems, MemoryTarget, Player, TempoMode, TransitionPlan, TransitionState,
};
use SampleFormat::*;

/// From 5 s of A over 2 s into B at 1 s (B's time is A's less 4 s), by
/// `lanes`.
fn fade_plan(lanes: [Envelope; 2 * STEMS], settle: f64) -> TransitionPlan {
    TransitionPlan {
        from: 0,
        into: 1,
        start: 5.0,
        length: 2.0,
        entry: 1.0,
        beats: beats(5.0, 0.5, 1.0, 0.5, 5),
        lanes,
        tempo: TempoMode::Varispeed,
        settle,
        glide: false,
    }
}

fn fades() -> [Envelope; 2 * STEMS] {
    lanes(
        Envelope::fade(1.0, 0.0, 0.0, 1.0),
        Envelope::fade(0.0, 1.0, 0.0, 1.0),
    )
}

/// The worst difference in the window (5 s to 7 s, but its edges) between
/// `got` and A and B faded by `plan`'s first lanes, B's frame the output's
/// less 4 s; and where.
fn off_in_window(plan: &TransitionPlan, got: &[i32], a: &[i32], b: &[i32]) -> (f64, usize) {
    let rate = RATE as usize;
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
    (worst, at)
}

#[test]
fn a_fade_at_one_tempo_mixes_the_two_and_hands_over() {
    let _serial = serial();
    let dir = temp_dir("transition-fade");
    let (a_path, a) = wav(&dir, "a.wav", 12.0, tone([1_000.0, 5_000.0]));
    let (b_path, b) = wav(&dir, "b.wav", 12.0, tone([2_000.0, 3_000.0]));
    let desk = Desk::new(caps(&[RATE], &[S24In32Low]), options());
    desk.play(vec![a_path, b_path]);
    let plan = fade_plan(fades(), 1.0);
    desk.player.transition(plan.clone()).unwrap();
    let seen = step_to_end(&desk.gate, &desk.events);
    let streams = desk.close();
    let states = transitions(&seen);
    assert_eq!(states[0], &TransitionState::Armed, "{states:?}");
    let w0 = 5 * RATE as u64;
    assert!(
        states.contains(&&TransitionState::Started {
            head: 0,
            output_frame: w0,
            origin: origin(5.0),
        }),
        "{states:?}"
    );
    assert!(states.contains(&&TransitionState::Finished), "{states:?}");
    // B heard from the window's end, its own output's frames counted from
    // its first (a second before the window).
    let legs = started(&seen);
    let into = legs.iter().find(|s| s.index == 1).expect("B started");
    assert_eq!(into.output_frame + origin(5.0), 7 * u64::from(RATE));
    assert_eq!(into.from_frame, 3 * u64::from(RATE));
    assert_eq!(streams.len(), 1, "one stream throughout");
    let got = samples(&streams[0].1);
    let rate = RATE as usize;
    // A as it is before the next track is heard with it.
    assert!(got[..2 * rate] == a[..2 * rate]);
    // Inside the window: each faded by its lanes (equal power), on the
    // desk's automation.
    let (worst, at) = off_in_window(&plan, &got, &a, &b);
    assert!(
        worst < 0.005 * FULL,
        "off by {worst} in the window at {:.4} s",
        (at / 2) as f64 / f64::from(RATE)
    );
    // B on its own route at the end, exactly the file.
    let end = got.len() / 2;
    assert_eq!(end, 16 * rate, "A to 7 s, then B from 3 s to its end");
    let offset = 4 * rate;
    let tail = (end - rate)..end;
    assert!(
        tail.clone()
            .all(|f| got[2 * f] == b[2 * (f - offset)] && got[2 * f + 1] == b[2 * (f - offset) + 1]),
        "B is lossless again at the end"
    );
    let back = legs.last().unwrap();
    assert_eq!((back.index, back.route.resample), (1, None));
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
    let desk = Desk::new(caps(&[RATE], &[S24In32Low]), options());
    desk.play(vec![a_path, b_path]);
    // Only B is heard from the window's start (A cut, B cut in).
    let plan = TransitionPlan {
        entry: 1.92,
        beats: beats(5.0, 0.5, 1.92, 0.48, 5),
        lanes: lanes(Envelope::cut(1.0, 0.0, 0.0), Envelope::cut(0.0, 1.0, 0.0)),
        tempo,
        ..fade_plan(fades(), 2.0)
    };
    desk.player.transition(plan).unwrap();
    let seen = step_to_end(&desk.gate, &desk.events);
    let streams = desk.close();
    assert!(transitions(&seen).contains(&&TransitionState::Finished));
    let onsets = onsets(&samples(&streams[0].1));
    // In the window, on A's beats (5.0, 5.5, ...).
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
    assert!(late.len() >= 2, "{onsets:?}");
    for w in late.windows(2) {
        assert!((w[1] - w[0] - 0.48).abs() < within, "{late:?}");
    }
}

/// B's map at its head's first frame (a second before the window) falls
/// before B's first frame (B comes in at its first click, 0.2 s in): B
/// starts that much silence later, so stretched as at a changing speed
/// its clicks land on A's beats from the first (once they were hundreds
/// of ms late, crept back at the stretchers' steering).
#[test]
fn a_next_track_whose_map_starts_before_it_lands_on_the_beats_from_the_first() {
    let _serial = serial();
    for tempo in [TempoMode::Varispeed, TempoMode::Stretch] {
        let dir = temp_dir(&format!("transition-early-{tempo:?}"));
        // A at 120 BPM, B at 130 from 0.2 s in.
        let b_period = 60.0 / 130.0;
        let (a_path, _) = wav(&dir, "a.wav", 14.0, clicks(0.5, 0.0, 3_000.0));
        let (b_path, _) = wav(&dir, "b.wav", 14.0, clicks(b_period, 0.2, 5_000.0));
        let desk = Desk::new(caps(&[RATE], &[S24In32Low]), options());
        desk.play(vec![a_path, b_path]);
        let plan = TransitionPlan {
            start: 4.0,
            length: 4.0,
            entry: 0.2,
            beats: beats(4.0, 0.5, 0.2, b_period, 8),
            lanes: lanes(Envelope::cut(1.0, 0.0, 0.0), Envelope::cut(0.0, 1.0, 0.0)),
            tempo,
            ..fade_plan(fades(), 2.0)
        };
        desk.player.transition(plan).unwrap();
        let seen = step_to_end(&desk.gate, &desk.events);
        let streams = desk.close();
        assert!(transitions(&seen).contains(&&TransitionState::Finished));
        let onsets = onsets(&samples(&streams[0].1));
        let in_window: Vec<f64> = onsets
            .iter()
            .copied()
            .filter(|&t| (3.9..7.9).contains(&t))
            .collect();
        assert_eq!(in_window.len(), 8, "{tempo:?}: {onsets:?}");
        for (i, t) in in_window.iter().enumerate() {
            let beat = 4.0 + 0.5 * i as f64;
            assert!(
                (t - beat).abs() < 0.002,
                "{tempo:?}: click {i} at {t}, beat at {beat}: {in_window:?}"
            );
        }
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

/// The same with two stems' strips muted on the desk (both tracks' stems
/// on them): the clicks at half their level.
#[test]
fn gliding_with_stems_muted_on_the_desk_meets_the_beats_too() {
    let _serial = serial();
    glide_meets(TempoMode::Varispeed, 0.001, drums_bass(), false);
}

/// The same with a plan without the glide replaced, once its next track is
/// being prepared, by one with it: the playing track goes along the glide.
#[test]
fn a_glide_planned_once_the_next_track_is_prepared_meets_the_beats_too() {
    let _serial = serial();
    glide_meets(TempoMode::Varispeed, 0.001, ALL, true);
}

/// The plan of [`glide_meets`]: A at 120 BPM, B at 126, a 4 s window from
/// 5 s, A alone through its first half, B alone through its second.
fn glide_plan(tempo: TempoMode, b_period: f64) -> TransitionPlan {
    let entry = 4.0 * b_period;
    TransitionPlan {
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
    }
}

/// A at 120 BPM, B at 126, the tempo gliding across a 4 s window: every
/// click (A's, then B's) is where A's beat is heard along the glide, within
/// `within` seconds; A's are closer together as it goes; after the window
/// B is at its own tempo at once; and the position heard follows the
/// glide. The stems' strips muted where `mask` has 0; with `replan`, the
/// plan comes once the next track is prepared for a plan without the
/// glide.
fn glide_meets(tempo: TempoMode, within: f64, mask: [f64; STEMS], replan: bool) {
    let dir = temp_dir(&format!(
        "transition-glide-{tempo:?}-{}-{replan}",
        mask == ALL
    ));
    let b_period = 60.0 / 126.0;
    let (a_path, _) = wav(&dir, "a.wav", 14.0, clicks(0.5, 0.0, 3_000.0));
    let (b_path, _) = wav(&dir, "b.wav", 14.0, clicks(b_period, 0.0, 5_000.0));
    let desk = Desk::new(caps(&[RATE], &[S24In32Low]), options());
    if mask != ALL {
        desk.player.make_stems(MakeStems::Separate).unwrap();
        desk.mask(mask);
    }
    desk.play(vec![a_path, b_path]);
    let plan = glide_plan(tempo, b_period);
    let glide = plan.glide_of(&plan.time_map());
    assert!((glide.end_speed() - 1.05).abs() < 1e-3, "{glide:?}");
    if replan {
        // First without the glide, until the next track is prepared (from
        // 1 s on).
        let flat = TransitionPlan {
            glide: false,
            ..plan.clone()
        };
        desk.player.transition(flat).unwrap();
        step_until(
            &desk.gate,
            &desk.events,
            |e| matches!(e, Event::Position(p) if p.output_frame >= 3 * u64::from(RATE) / 2),
        );
    }
    desk.player.transition(plan).unwrap();
    let seen = step_to_end(&desk.gate, &desk.events);
    let streams = desk.close();
    assert!(
        !seen.iter().any(|e| matches!(e, Event::Notice { .. })),
        "{seen:?}"
    );
    assert!(
        transitions(&seen).contains(&&TransitionState::Finished),
        "{:?}",
        transitions(&seen)
    );
    let onsets = onsets(&samples(&streams[0].1));
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
    let rate = f64::from(RATE);
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
/// out of the queue): the playing track goes on from where the glide had
/// it, at its own speed, without a jump (its clicks keep their place and
/// then their spacing).
#[test]
fn a_glide_called_off_in_its_window_goes_on_without_a_jump() {
    let _serial = serial();
    let dir = temp_dir("transition-glide-off");
    let b_period = 60.0 / 126.0;
    let (a_path, _) = wav(&dir, "a.wav", 14.0, clicks(0.5, 0.0, 3_000.0));
    let (b_path, _) = wav(&dir, "b.wav", 14.0, clicks(b_period, 0.0, 5_000.0));
    let desk = Desk::new(caps(&[RATE], &[S24In32Low]), options());
    desk.play(vec![a_path, b_path]);
    // Only A is heard.
    let plan = TransitionPlan {
        lanes: lanes(Envelope::flat(1.0), Envelope::flat(0.0)),
        ..glide_plan(TempoMode::Varispeed, b_period)
    };
    let glide = plan.glide_of(&plan.time_map());
    desk.player.transition(plan).unwrap();
    // Into the window, then the next track taken out calls it off.
    let mut seen = step_until(
        &desk.gate,
        &desk.events,
        |e| matches!(e, Event::Position(p) if p.output_frame >= 7 * u64::from(RATE)),
    );
    desk.player.remove(1).unwrap();
    seen.extend(step_to_end(&desk.gate, &desk.events));
    let streams = desk.close();
    assert!(
        transitions(&seen)
            .iter()
            .any(|s| matches!(s, TransitionState::Cancelled { .. })),
        "{:?}",
        transitions(&seen)
    );
    let onsets = onsets(&samples(&streams[0].1));
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
    let desk = Desk::new(caps(&[RATE], &[S24In32Low]), options());
    desk.play(vec![a_path, b_path]);
    let plan = TransitionPlan {
        start: 8.0,
        entry: 0.0,
        beats: Vec::new(),
        settle: 0.0,
        ..fade_plan(fades(), 0.0)
    };
    // Not the playing track: called off at once.
    desk.player
        .transition(TransitionPlan {
            from: 1,
            into: 2,
            ..plan.clone()
        })
        .unwrap();
    let seen = collect_until(&desk.events, |e| matches!(e, Event::Transition(_)));
    assert!(matches!(
        transitions(&seen)[0],
        TransitionState::Cancelled { .. }
    ));
    // Armed, then a seek calls it off.
    desk.player.transition(plan).unwrap();
    collect_until(&desk.events, |e| {
        matches!(e, Event::Transition(TransitionState::Armed))
    });
    desk.player.seek(2.0).unwrap();
    let seen = collect_until(&desk.events, |e| matches!(e, Event::Transition(_)));
    assert!(
        matches!(transitions(&seen)[0], TransitionState::Cancelled { reason } if reason == "a seek"),
        "{seen:?}"
    );
    desk.gate.allow_all();
    let rest = until_queue_end(&desk.events);
    // Without it, the tracks follow each other as usual.
    assert_eq!(started(&rest).iter().filter(|s| s.index == 1).count(), 1);
    assert!(transitions(&rest).is_empty());
}

/// A player with no second head (not on a desk): a transition is called
/// off, as it needs one.
#[test]
fn a_transition_without_a_second_head_is_called_off() {
    let dir = temp_dir("transition-one-head");
    let (a_path, _) = wav(&dir, "a.wav", 3.0, tone([440.0, 440.0]));
    let (b_path, _) = wav(&dir, "b.wav", 3.0, tone([440.0, 440.0]));
    let target = MemoryTarget::new(caps(&[RATE], &[S24In32Low]), SimClock::Instant);
    let (player, events) = Player::new(Box::new(target), options());
    player.load(vec![a_path, b_path]).unwrap();
    player.play(0).unwrap();
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    player.transition(fade_plan(fades(), 1.0)).unwrap();
    let seen = collect_until(&events, |e| matches!(e, Event::Transition(_)));
    assert!(
        matches!(transitions(&seen)[0], TransitionState::Cancelled { reason } if reason == "needs a desk"),
        "{seen:?}"
    );
}

#[test]
fn without_a_separator_the_tracks_fade_whole() {
    let _serial = serial();
    let dir = temp_dir("transition-whole");
    let (a_path, a) = wav(&dir, "a.wav", 10.0, tone([1_000.0, 5_000.0]));
    let (b_path, b) = wav(&dir, "b.wav", 10.0, tone([2_000.0, 3_000.0]));
    let options = jodel_playback::PlayerOptions {
        separator: None,
        ..options()
    };
    let desk = Desk::new(caps(&[RATE], &[S24In32Low]), options);
    desk.play(vec![a_path, b_path]);
    let plan = fade_plan(fades(), 1.0);
    desk.player.transition(plan.clone()).unwrap();
    let seen = step_to_end(&desk.gate, &desk.events);
    let streams = desk.close();
    assert!(transitions(&seen).contains(&&TransitionState::Finished));
    let got = samples(&streams[0].1);
    let (worst, _) = off_in_window(&plan, &got, &a, &b);
    assert!(worst < 0.005 * FULL, "off by {worst}");
}

#[test]
fn the_next_track_is_heard_where_the_map_puts_it() {
    let _serial = serial();
    let dir = temp_dir("transition-heard");
    let (a_path, _) = wav(&dir, "a.wav", 12.0, clicks(0.5, 0.0, 3_000.0));
    let (b_path, _) = wav(&dir, "b.wav", 12.0, clicks(0.48, 0.0, 5_000.0));
    let desk = Desk::new(caps(&[RATE], &[S24In32Low]), options());
    desk.play(vec![a_path, b_path]);
    let pairs = beats(5.0, 0.5, 1.92, 0.48, 5);
    let plan = TransitionPlan {
        entry: 1.92,
        beats: pairs.clone(),
        lanes: lanes(Envelope::cut(1.0, 0.0, 0.0), Envelope::cut(0.0, 1.0, 0.0)),
        ..fade_plan(fades(), 2.0)
    };
    desk.player.transition(plan).unwrap();
    let seen = step_to_end(&desk.gate, &desk.events);
    drop(desk);
    let map = TimeMap::new(&pairs, 2.0);
    // While B glides back to its tempo (from 7 s to 9 s of A's time), the
    // positions reported are those of the map; after that, a straight line.
    // B's output frames are A's less the link's origin.
    let mut checked = 0;
    for e in &seen {
        let Event::Position(p) = e else { continue };
        if p.index != 1 {
            continue;
        }
        let a_time = (p.output_frame + origin(5.0)) as f64 / f64::from(RATE);
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
    let dir = temp_dir("transition-cut");
    let (a_path, _) = wav(&dir, "a.wav", 14.0, clicks(0.5, 0.0, 3_000.0));
    let (b_path, _) = wav(&dir, "b.wav", 14.0, clicks(0.48, 0.0, 5_000.0));
    let desk = Desk::new(caps(&[RATE], &[S24In32Low]), options());
    desk.play(vec![a_path, b_path]);
    // A window of 4 s from 5 s of A, B entering at 1.92 s.
    let pairs = beats(5.0, 0.5, 1.92, 0.48, 9);
    let plan = TransitionPlan {
        length: 4.0,
        entry: 1.92,
        beats: pairs.clone(),
        ..fade_plan(fades(), 0.0)
    };
    desk.player.transition(plan).unwrap();
    // A second into the window, next.
    step_until(
        &desk.gate,
        &desk.events,
        |e| matches!(e, Event::Position(p) if p.index == 0 && p.seconds >= 6.0),
    );
    desk.player.next().unwrap();
    let seen = step_to_end(&desk.gate, &desk.events);
    drop(desk);
    assert!(
        transitions(&seen).iter().any(
            |t| matches!(t, TransitionState::Cancelled { reason } if reason == "cut to the next track")
        ),
        "{:?}",
        transitions(&seen)
    );
    assert!(!transitions(&seen).contains(&&TransitionState::Finished));
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

/// `Next` in the window: A fades out over a few milliseconds and B goes
/// to its end values, so nothing clicks: no step from one sample to the
/// next beyond what the two low tones make (a cut at full scale would be
/// ten times that).
#[test]
fn next_mid_mix_does_not_click() {
    let _serial = serial();
    let dir = temp_dir("transition-cut-click");
    let (a_path, _) = wav(&dir, "a.wav", 12.0, tone([200.0, 230.0]));
    let (b_path, _) = wav(&dir, "b.wav", 12.0, tone([170.0, 190.0]));
    let desk = Desk::new(caps(&[RATE], &[S24In32Low]), options());
    desk.play(vec![a_path, b_path]);
    desk.player.transition(fade_plan(fades(), 0.0)).unwrap();
    step_until(
        &desk.gate,
        &desk.events,
        |e| matches!(e, Event::Position(p) if p.index == 0 && p.seconds >= 5.8),
    );
    desk.player.next().unwrap();
    let seen = step_to_end(&desk.gate, &desk.events);
    let streams = desk.close();
    assert!(transitions(&seen).iter().any(
        |t| matches!(t, TransitionState::Cancelled { reason } if reason == "cut to the next track")
    ));
    let got = samples(&streams[0].1);
    let rate = RATE as usize;
    // Two tones of 0.4 move by 0.4 · 2π · 230 / 44100 of full scale a
    // sample at most, each.
    let most = 2.0 * 0.4 * 2.0 * std::f64::consts::PI * 230.0 / f64::from(RATE) * FULL;
    let steepest = (4 * rate..8 * rate)
        .flat_map(|f| (0..2).map(move |c| (f, c)))
        .map(|(f, c)| (f64::from(got[2 * (f + 1) + c]) - f64::from(got[2 * f + c])).abs())
        .fold(0.0, f64::max);
    assert!(
        steepest < 1.2 * most,
        "a step of {steepest} (the tones make {most} at most)"
    );
}

#[test]
fn a_transition_called_off_while_it_prepares_leaves_the_track_as_it_was() {
    let _serial = serial();
    let dir = temp_dir("transition-cancel");
    let tone = |t: f64, _c: usize| 0.3 * (2.0 * std::f64::consts::PI * 440.0 * t).sin();
    let (a_path, a) = wav(&dir, "a.wav", 12.0, tone);
    let (b_path, _) = wav(&dir, "b.wav", 12.0, tone);
    let desk = Desk::new(caps(&[RATE], &[S24In32Low]), options());
    desk.play(vec![a_path, b_path]);
    desk.player
        .transition(TransitionPlan {
            start: 6.0,
            entry: 0.0,
            beats: Vec::new(),
            ..fade_plan(fades(), 0.0)
        })
        .unwrap();
    // Into the preparation (from 2 s of A), then called off.
    step_until(
        &desk.gate,
        &desk.events,
        |e| matches!(e, Event::Position(p) if p.output_frame >= 3 * u64::from(RATE)),
    );
    desk.player.cancel_transition().unwrap();
    let seen = step_to_end(&desk.gate, &desk.events);
    let streams = desk.close();
    assert!(
        transitions(&seen)
            .iter()
            .any(|s| matches!(s, TransitionState::Cancelled { .. })),
        "{seen:?}"
    );
    // A plays on to its end, as it is: the file exactly at its end, and
    // nothing faded where the window would have been.
    let got = samples(&streams[0].1);
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

/// Every track at the desk's one rate, through a mix, after it and when
/// picked: the first (44.1 kHz) and the next two (48 kHz) on a desk running
/// at 48 kHz, on the one device stream, never opened again.
#[test]
fn every_track_plays_at_the_desks_one_rate() {
    let _serial = serial();
    let dir = temp_dir("transition-rate");
    let (a_path, _) = wav(&dir, "a.wav", 8.0, tone([1_000.0, 5_000.0]));
    let tone48 = |seconds: f64, name: &str| {
        let path = dir.join(name);
        write_wav(
            &path,
            48_000,
            2,
            &tones(48_000, (seconds * 48_000.0) as usize),
        );
        path
    };
    let (b_path, c_path) = (tone48(8.0, "b.wav"), tone48(2.0, "c.wav"));
    let options = jodel_playback::PlayerOptions {
        settings: jodel_playback::Settings::default(),
        ..options()
    };
    let desk = Desk::new(caps(&[48_000], &[S24In32Low]), options);
    desk.play(vec![a_path, b_path, c_path]);
    desk.player.transition(fade_plan(fades(), 0.5)).unwrap();
    let seen = step_until(
        &desk.gate,
        &desk.events,
        |e| matches!(e, Event::TrackStarted(s) if s.index == 2),
    );
    assert!(transitions(&seen).contains(&&TransitionState::Finished));
    let rates: Vec<(usize, u32)> = started(&seen)
        .iter()
        .map(|s| (s.index, s.output.format.rate))
        .collect();
    assert!(rates.iter().all(|&(_, r)| r == 48_000), "{rates:?}");
    // Picked: at the same rate.
    desk.player.play(2).unwrap();
    desk.gate.allow_all();
    let seen = until_queue_end(&desk.events);
    let picked = started(&seen)
        .into_iter()
        .find(|s| s.index == 2)
        .expect("C");
    assert_eq!(picked.output.format.rate, 48_000);
    let streams = desk.close();
    assert_eq!(streams.len(), 1, "one stream: {rates:?}");
    assert_eq!(streams[0].0.rate, 48_000);
}

/// On a device in real time: a stem's strip muted on a bar 0.1 s ahead in
/// the window is heard off on its frame, on both tracks; a pause in the
/// window holds the next track with the playing one (its link): after it
/// the two are mixed where they were, nothing of either dropped.
#[test]
fn in_real_time_a_stem_muted_on_its_bar_and_a_pause_keep_the_mix_in_place() {
    let _serial = serial();
    let _real_time = real_time();
    let dir = temp_dir("transition-paced");
    let (a_path, a) = wav(&dir, "a.wav", 10.0, tone([1_000.0, 5_000.0]));
    let (b_path, b) = wav(&dir, "b.wav", 10.0, tone([2_000.0, 3_000.0]));
    let options = jodel_playback::PlayerOptions {
        offline: false,
        position_interval: Duration::from_millis(10),
        ..options()
    };
    let desk = Desk::paced(caps(&[RATE], &[S24In32Low]), options);
    desk.player.make_stems(MakeStems::Separate).unwrap();
    desk.play(vec![a_path, b_path]);
    let plan = fade_plan(fades(), 1.0);
    desk.player.transition(plan.clone()).unwrap();
    let heard_past = |seconds: f64| {
        let seen = collect_until(
            &desk.events,
            |e| matches!(e, Event::Position(p) if p.index == 0 && p.seconds >= seconds),
        );
        match seen.last() {
            Some(Event::Position(p)) => p.output_frame,
            _ => unreachable!(),
        }
    };
    // The drums off on a bar a tenth of a second ahead of what is heard.
    let bar = heard_past(5.2) + u64::from(RATE) / 10;
    desk.desk.stems[0].set_mute_at(true, desk.lead.input.at_stream(bar));
    // Paused for a while later in the window.
    heard_past(6.0);
    desk.player.pause().unwrap();
    std::thread::sleep(Duration::from_millis(400));
    desk.player.resume().unwrap();
    let seen = until_queue_end(&desk.events);
    assert!(transitions(&seen).contains(&&TransitionState::Finished));
    let dropped: u64 = desk
        .mixer
        .desk()
        .strips
        .iter()
        .flat_map(|s| s.inputs.iter().map(|i| i.dropped))
        .sum();
    assert_eq!(dropped, 0, "frames dropped");
    let streams = desk.close();
    assert_eq!(streams.len(), 1);
    let got = samples(&streams[0].1);
    let rate = RATE as usize;
    let offset = 4 * rate;
    // What the window plays at A's frame `f`, a share `share` of it.
    let want = |f: usize, c: usize, share: f64| {
        let s = plan.share(f as f64 / f64::from(RATE));
        share
            * (plan.lanes[0].gain(s) * f64::from(a[2 * f + c])
                + plan.lanes[STEMS].gain(s) * f64::from(b[2 * (f - offset) + c]))
    };
    // How far `got` is off that over A's frames `range`, the device's
    // frames `shift` later.
    let off = |range: std::ops::Range<usize>, shift: usize, share: f64| {
        range
            .flat_map(|f| (0..2).map(move |c| (f, c)))
            .map(|(f, c)| (f64::from(got[2 * (f + shift) + c]) - want(f, c, share)).abs())
            .fold(0.0, f64::max)
    };
    // Where the device has A's frames before the pause, and after it.
    let found = |at: usize, share: f64| {
        (0..2 * rate)
            .find(|&d| off(at..at + 64, d, share) < 0.002 * FULL)
            .unwrap_or_else(|| panic!("the mix at {at} not found"))
    };
    let before = found(5 * rate + rate / 10, 1.0);
    let bar = bar as usize;
    let unmuted = off(5 * rate + rate / 20..bar - 600, before, 1.0);
    let muted = off(bar + rate / 40..6 * rate - rate / 20, before, 0.75);
    assert!(unmuted < 0.01 * FULL, "before the bar, off by {unmuted}");
    assert!(muted < 0.01 * FULL, "after the bar, off by {muted}");
    // After the pause, both still where the window has them.
    let after = found(6 * rate + rate / 2, 0.75);
    let held = off(6 * rate + rate / 2..7 * rate - 200, after, 0.75);
    assert!(held < 0.01 * FULL, "after the pause, off by {held}");
}
