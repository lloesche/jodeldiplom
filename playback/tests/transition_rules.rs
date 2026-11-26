//! The rules of how other controls meet a transition
//! (player/INTERACTIONS.md), on a desk: stems' strips muted while it is
//! armed, prepared, mixed and settling (both tracks on those strips, on
//! their frames), a mix called off, paused, other settings, a plan
//! changed, the queue changed; and the next track back on its own route
//! after it.

mod common;

use std::time::Duration;

use common::transition::*;
use common::*;
use jodel_dsp::FULL_SCALE;
use jodel_output::SampleFormat;
use jodel_playback::stems::STEMS;
use jodel_playback::{
    Envelope, Event, MakeStems, Settings, TempoMode, TransitionPlan, TransitionState,
};
use jodel_testkit::TempDir;
use SampleFormat::*;

/// Two tone tracks (10 s) on a desk, the first playing, its stems made and
/// their strips muted where `mask` has 0; and the second's samples.
fn rules_setup(name: &str, mask: [f64; STEMS]) -> (TempDir, Vec<i32>, Desk) {
    let dir = TempDir::new(name);
    let (a_path, _) = wav(&dir, "a.wav", 10.0, tone([1_000.0, 5_000.0]));
    let (b_path, b) = wav(&dir, "b.wav", 10.0, tone([2_000.0, 3_000.0]));
    let desk = Desk::new(caps(&[RATE], &[S24In32Low]), options());
    desk.player.make_stems(MakeStems::Separate).unwrap();
    desk.mask(mask);
    desk.play(vec![a_path, b_path]);
    (dir, b, desk)
}

/// From 5 s of A over 2 s into B at 1 s (B's time is A's less 4 s), by
/// `lanes`, settling over `settle`.
fn rules_plan(lanes: [Envelope; 2 * STEMS], settle: f64) -> TransitionPlan {
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

fn fade_lanes() -> [Envelope; 2 * STEMS] {
    lanes(
        Envelope::fade(1.0, 0.0, 0.0, 1.0),
        Envelope::fade(0.0, 1.0, 0.0, 1.0),
    )
}

/// The window started where it was planned (5 s), it finished, and
/// nothing called it off.
fn assert_ran_as_planned(seen: &[Event]) {
    let states = transitions(seen);
    assert!(
        states.contains(&&TransitionState::Started {
            head: 0,
            output_frame: 5 * u64::from(RATE),
            origin: origin(5.0),
        }),
        "{states:?}"
    );
    assert!(states.contains(&&TransitionState::Finished), "{states:?}");
    assert!(
        !states
            .iter()
            .any(|s| matches!(s, TransitionState::Cancelled { .. })),
        "{states:?}"
    );
}

/// The worst difference, in 24-bit steps, between output frames `range`
/// and `scale` times B where B's frame is the output's less `offset`.
fn off_b(got: &[i32], b: &[i32], range: std::ops::Range<usize>, offset: usize, scale: f64) -> f64 {
    range
        .flat_map(|f| (0..2).map(move |c| (f, c)))
        .map(|(f, c)| {
            (f64::from(got[2 * f + c]) - scale * f64::from(b[2 * (f - offset) + c])).abs()
        })
        .fold(0.0, f64::max)
}

/// Two stems' strips muted through a transition, and every strip put back
/// after it: the next track goes back onto its own route once the
/// transition is over, and the desk plays the file as it is there.
#[test]
fn after_a_transition_the_next_track_plays_its_own_route_again() {
    let _serial = serial();
    let (_dir, b, desk) = rules_setup("transition-stems", drums_bass());
    desk.player
        .transition(rules_plan(fade_lanes(), 1.0))
        .unwrap();
    let mut seen = step_until(&desk.gate, &desk.events, |e| {
        matches!(e, Event::Transition(TransitionState::Finished))
    });
    assert_ran_as_planned(&seen);
    desk.mask(ALL);
    seen.extend(step_to_end(&desk.gate, &desk.events));
    let streams = desk.close();
    assert_eq!(streams.len(), 1, "one stream throughout");
    let got = samples(&streams[0].1);
    let rate = RATE as usize;
    // A to 7 s, then B from 3 s to its end: its last second is B exactly.
    let end = got.len() / 2;
    assert_eq!(end, 14 * rate);
    let last = 2 * (end - rate);
    assert!(
        got[last..] == b[last - 2 * 4 * rate..],
        "B's end is not the file"
    );
    let back = started(&seen).last().copied().expect("B's return");
    assert_eq!((back.index, back.route.resample), (1, None));
}

/// Drums and bass through a transition, the bass muted in its window: the
/// transition goes on, and the next track plays the drums alone while it
/// settles and after.
#[test]
fn stems_muted_during_a_transition_are_heard_on_both_tracks() {
    let _serial = serial();
    let (_dir, b, desk) = rules_setup("transition-stems-during", drums_bass());
    desk.player
        .transition(rules_plan(fade_lanes(), 1.0))
        .unwrap();
    let mut seen = step_until(&desk.gate, &desk.events, |e| {
        matches!(e, Event::Transition(TransitionState::Started { .. }))
    });
    desk.mask(drums_only());
    seen.extend(step_to_end(&desk.gate, &desk.events));
    let streams = desk.close();
    assert_ran_as_planned(&seen);
    let got = samples(&streams[0].1);
    let rate = RATE as usize;
    let (end, offset) = (got.len() / 2, 4 * rate);
    assert_eq!(end, 14 * rate);
    // In the window, from a while after the drums alone: a quarter of A
    // and of B, each faded by its lanes.
    let a = read_all(&_dir.join("a.wav"));
    let plan = rules_plan(fade_lanes(), 1.0);
    let window = (5 * rate + 3 * rate / 10..7 * rate - 200)
        .flat_map(|f| (0..2).map(move |c| (f, c)))
        .map(|(f, c)| {
            let share = plan.share(f as f64 / f64::from(RATE));
            let (ga, gb) = (plan.lanes[0].gain(share), plan.lanes[STEMS].gain(share));
            let want = ga * f64::from(a[2 * f + c]) + gb * f64::from(b[2 * (f - offset) + c]);
            (f64::from(got[2 * f + c]) - 0.25 * want).abs()
        })
        .fold(0.0, f64::max);
    assert!(
        window < 0.005 * FULL_SCALE,
        "in the window, {window} off the drums of both"
    );
    // Settling (to 8.3 s), and on its own route after: a quarter of B.
    let settling = off_b(
        &got,
        &b,
        7 * rate + rate / 10..8 * rate + rate / 5,
        offset,
        0.25,
    );
    assert!(
        settling < 96.0,
        "B settling is {settling} steps off its drums"
    );
    let last = off_b(&got, &b, end - rate..end, offset, 0.25);
    assert!(last < 96.0, "B at its end is {last} steps off its drums");
}

/// S1: stems muted while armed (on a bar) and again while preparing (at
/// once) leave the transition where it was: its window starts on time, it
/// runs, it ends.
#[test]
fn stems_switched_while_armed_leave_the_transition_where_it_was() {
    let _serial = serial();
    let (_dir, _b, desk) = rules_setup("rules-armed", ALL);
    desk.player
        .transition(rules_plan(fade_lanes(), 1.0))
        .unwrap();
    // Armed (the window is 5 s off): the vocals and the other instruments
    // off on a bar.
    let mut seen = step_until(
        &desk.gate,
        &desk.events,
        |e| matches!(e, Event::Position(p) if p.output_frame >= u64::from(RATE) / 2),
    );
    let bar = desk.lead.input.at_stream(3 * u64::from(RATE) / 2);
    for strip in &desk.desk.stems[2..] {
        strip.set_mute_at(true, bar);
    }
    // Preparing (within 4 s of it): the bass off too, at once.
    seen.extend(step_until(
        &desk.gate,
        &desk.events,
        |e| matches!(e, Event::Position(p) if p.output_frame >= 7 * u64::from(RATE) / 2),
    ));
    desk.mask(drums_only());
    seen.extend(step_to_end(&desk.gate, &desk.events));
    drop(desk);
    assert_ran_as_planned(&seen);
}

/// S1, S2: a stem muted on a bar inside the window goes off there, on the
/// next track too (here alone in the window), and nothing else does.
#[test]
fn stems_switched_at_a_bar_in_the_window_change_there() {
    let _serial = serial();
    let (_dir, b, desk) = rules_setup("rules-at-bar", drums_bass());
    // A out at once, B in at once.
    let plan = rules_plan(
        lanes(Envelope::cut(1.0, 0.0, 0.0), Envelope::cut(0.0, 1.0, 0.0)),
        1.0,
    );
    desk.player.transition(plan).unwrap();
    let mut seen = step_until(&desk.gate, &desk.events, |e| {
        matches!(e, Event::Transition(TransitionState::Started { .. }))
    });
    // The bass off at 6 s of the playing track: the middle of the window.
    let bar = desk.lead.input.at_stream(6 * u64::from(RATE));
    desk.desk.stems[1].set_mute_at(true, bar);
    seen.extend(step_to_end(&desk.gate, &desk.events));
    let streams = desk.close();
    assert_ran_as_planned(&seen);
    let got = samples(&streams[0].1);
    let (rate, offset) = (RATE as usize, 4 * RATE as usize);
    // Drums and bass (half of B) up to 6 s, the drums (a quarter) after
    // (past the strip's 20 ms ramp).
    let before = off_b(&got, &b, 5 * rate + rate / 20..6 * rate - 64, offset, 0.5);
    let after = off_b(&got, &b, 6 * rate + rate / 40..7 * rate - 64, offset, 0.25);
    assert!(before < 0.01 * FULL_SCALE, "before the change {before}");
    assert!(after < 0.01 * FULL_SCALE, "after the change {after}");
}

/// S1, S4: a stem muted while the next track settles after the window is
/// heard on it; the transition ends as planned, and the next track plays on
/// so once it is back on its own route.
#[test]
fn stems_switched_while_settling_go_into_the_next_tracks_fades() {
    let _serial = serial();
    let (_dir, b, desk) = rules_setup("rules-settling", drums_bass());
    desk.player
        .transition(rules_plan(fade_lanes(), 2.0))
        .unwrap();
    // The window is over at 7 s: the next track settles (its frames A's
    // less the link's origin).
    let settling = 15 * u64::from(RATE) / 2 - origin(5.0);
    let mut seen = step_until(
        &desk.gate,
        &desk.events,
        |e| matches!(e, Event::Position(p) if p.index == 1 && p.output_frame >= settling),
    );
    desk.mask(drums_only());
    seen.extend(step_to_end(&desk.gate, &desk.events));
    let streams = desk.close();
    assert_ran_as_planned(&seen);
    let got = samples(&streams[0].1);
    let rate = RATE as usize;
    let end = got.len() / 2;
    assert_eq!(end, 14 * rate);
    // Settling to 9.3 s: from half a second after the change, the drums.
    let settled = off_b(&got, &b, 8 * rate..9 * rate + rate / 5, 4 * rate, 0.25);
    assert!(settled < 96.0, "settling, {settled} steps off its drums");
    let last = off_b(&got, &b, end - rate..end, 4 * rate, 0.25);
    assert!(last < 96.0, "at its end, {last} steps off its drums");
}

/// T2: calling off a mix under way does nothing: it plays to its end.
#[test]
fn calling_off_a_mix_under_way_does_nothing() {
    let _serial = serial();
    let (_dir, _b, desk) = rules_setup("rules-cancel-mix", ALL);
    desk.player
        .transition(rules_plan(fade_lanes(), 1.0))
        .unwrap();
    let mut seen = step_until(&desk.gate, &desk.events, |e| {
        matches!(e, Event::Transition(TransitionState::Started { .. }))
    });
    desk.player.cancel_transition().unwrap();
    // A new plan neither replaces it.
    desk.player
        .transition(rules_plan(fade_lanes(), 1.0))
        .unwrap();
    seen.extend(step_to_end(&desk.gate, &desk.events));
    drop(desk);
    assert_ran_as_planned(&seen);
}

/// T3: pausing while preparing does not call it off: it runs on time
/// after the pause.
#[test]
fn a_paused_transition_goes_on_after_the_pause() {
    let _serial = serial();
    let (_dir, _b, desk) = rules_setup("rules-pause", ALL);
    desk.player
        .transition(rules_plan(fade_lanes(), 1.0))
        .unwrap();
    let mut seen = step_until(
        &desk.gate,
        &desk.events,
        |e| matches!(e, Event::Position(p) if p.output_frame >= 7 * u64::from(RATE) / 2),
    );
    desk.player.pause().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    desk.player.resume().unwrap();
    seen.extend(step_to_end(&desk.gate, &desk.events));
    drop(desk);
    assert_ran_as_planned(&seen);
}

/// T3: pausing in the window pauses the playing track, and the link holds
/// the next one with it: the mix goes on as planned after the pause.
#[test]
fn a_mix_paused_in_its_window_goes_on_after_the_pause() {
    let _serial = serial();
    let (_dir, _b, desk) = rules_setup("rules-pause-mix", ALL);
    desk.player
        .transition(rules_plan(fade_lanes(), 1.0))
        .unwrap();
    let mut seen = step_until(
        &desk.gate,
        &desk.events,
        |e| matches!(e, Event::Position(p) if p.output_frame >= 6 * u64::from(RATE)),
    );
    desk.player.pause().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    desk.player.resume().unwrap();
    seen.extend(step_to_end(&desk.gate, &desk.events));
    drop(desk);
    assert_ran_as_planned(&seen);
}

/// T4: other settings during the mix do not call it off; they wait for
/// its end, and a notice says so.
#[test]
fn other_settings_during_a_mix_wait_for_its_end() {
    let _serial = serial();
    let (_dir, _b, desk) = rules_setup("rules-settings", ALL);
    desk.player
        .transition(rules_plan(fade_lanes(), 1.0))
        .unwrap();
    let mut seen = step_until(&desk.gate, &desk.events, |e| {
        matches!(e, Event::Transition(TransitionState::Started { .. }))
    });
    desk.player
        .set_settings(Settings {
            headroom: jodel_playback::Headroom(10),
        })
        .unwrap();
    seen.extend(step_to_end(&desk.gate, &desk.events));
    drop(desk);
    assert_ran_as_planned(&seen);
    assert!(
        seen.iter().any(|e| matches!(e, Event::Notice { message, .. } if message.contains("once the transition is over"))),
        "no notice"
    );
}

/// T1: another plan for the same window while it prepares (another preset,
/// another tempo mode) keeps the window where it was.
#[test]
fn a_plan_changed_while_preparing_keeps_its_window() {
    let _serial = serial();
    let (_dir, _b, desk) = rules_setup("rules-replan", drums_bass());
    desk.player
        .transition(rules_plan(fade_lanes(), 1.0))
        .unwrap();
    let mut seen = step_until(
        &desk.gate,
        &desk.events,
        |e| matches!(e, Event::Position(p) if p.output_frame >= 7 * u64::from(RATE) / 2),
    );
    let mut again = rules_plan(
        lanes(Envelope::cut(1.0, 0.0, 0.5), Envelope::cut(0.0, 1.0, 0.5)),
        1.0,
    );
    again.tempo = TempoMode::Stretch;
    desk.player.transition(again).unwrap();
    seen.extend(step_to_end(&desk.gate, &desk.events));
    drop(desk);
    assert_ran_as_planned(&seen);
}

/// Q1: tracks moved around the transition's two: it follows them; its
/// next track taken out: it is called off, and the playing track goes on.
#[test]
fn a_transition_follows_its_tracks_through_the_queue() {
    let _serial = serial();
    let (dir, _b, desk) = rules_setup("rules-queue", ALL);
    // A third track after the two, moved before the playing one: the
    // transition's tracks are 1 and 2 then.
    let (c_path, _) = wav(&dir, "c.wav", 1.0, |_, _| 0.0);
    desk.player.enqueue(vec![c_path]).unwrap();
    desk.player
        .transition(rules_plan(fade_lanes(), 1.0))
        .unwrap();
    let mut seen = step_until(&desk.gate, &desk.events, |e| {
        matches!(e, Event::Transition(TransitionState::Armed))
    });
    desk.player.move_track(2, 0).unwrap();
    seen.extend(step_until(&desk.gate, &desk.events, |e| {
        matches!(e, Event::Transition(TransitionState::Finished))
    }));
    assert!(transitions(&seen).contains(&&TransitionState::Started {
        head: 0,
        output_frame: 5 * u64::from(RATE),
        origin: origin(5.0),
    }));
    assert!(seen
        .iter()
        .any(|e| matches!(e, Event::TrackStarted(s) if s.index == 2)));
    drop(desk);
    // Again, its next track taken out.
    let (_dir, _b, desk) = rules_setup("rules-queue-out", ALL);
    desk.player
        .transition(rules_plan(fade_lanes(), 1.0))
        .unwrap();
    let mut seen = step_until(&desk.gate, &desk.events, |e| {
        matches!(e, Event::Transition(TransitionState::Armed))
    });
    desk.player.remove(1).unwrap();
    seen.extend(step_to_end(&desk.gate, &desk.events));
    drop(desk);
    assert!(
        transitions(&seen).iter().any(
            |s| matches!(s, TransitionState::Cancelled { reason } if reason.contains("queue"))
        ),
        "{:?}",
        transitions(&seen)
    );
}
