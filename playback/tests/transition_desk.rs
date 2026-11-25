//! A transition's two heads on the desk as playback moves around it: next
//! while it is prepared and while it settles, stop, play and previous in
//! its window, its next track moved away in its window (it fades out, the
//! playing track comes back whole), the heads' roles over two hand-overs
//! in a row, a head read along a map refusing one, and on a
//! device a next track whose stems come late (it waits while it is silent,
//! then plays its whole mix).

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::lanes::Gated;
use common::transition::*;
use common::*;
use jodel_output::mixer::{MicPort, Role};
use jodel_output::{Input, SampleFormat};
use jodel_playback::stems::{Separator, STEMS};
use jodel_playback::warp::TimeMap;
use jodel_playback::{
    Command, Envelope, Event, MakeStems, PlayState, PlayerOptions, TempoMode, TransitionPlan,
    TransitionState,
};
use SampleFormat::*;

/// From `start` s of track `from` over 2 s into the next at 1 s (beats at
/// one tempo), faded, settling over `settle`.
fn plan(from: usize, start: f64, settle: f64) -> TransitionPlan {
    TransitionPlan {
        from,
        into: from + 1,
        start,
        length: 2.0,
        entry: 1.0,
        beats: beats(start, 0.5, 1.0, 0.5, 5),
        lanes: lanes(
            Envelope::fade(1.0, 0.0, 0.0, 1.0),
            Envelope::fade(0.0, 1.0, 0.0, 1.0),
        ),
        tempo: TempoMode::Varispeed,
        settle,
        glide: false,
    }
}

/// Three tone tracks of `seconds`, on a desk, the first playing.
fn three(name: &str, seconds: f64, options: PlayerOptions) -> (TempDir, Vec<Vec<i32>>, Desk) {
    let dir = temp_dir(name);
    let mut files = Vec::new();
    let mut samples = Vec::new();
    for (k, hz) in [[200.0, 230.0], [170.0, 190.0], [260.0, 290.0]]
        .into_iter()
        .enumerate()
    {
        let (path, s) = wav(&dir, &format!("{k}.wav"), seconds, tone(hz));
        files.push(path);
        samples.push(s);
    }
    let desk = Desk::new(caps(&[RATE], &[S24In32Low]), options);
    desk.play(files);
    (dir, samples, desk)
}

/// Steps the output on until the playing track (`index`) is heard at
/// `seconds`.
fn heard_at(desk: &Desk, index: usize, seconds: f64) -> Vec<Event> {
    step_until(
        &desk.gate,
        &desk.events,
        |e| matches!(e, Event::Position(p) if p.index == index && p.seconds >= seconds),
    )
}

/// Lets the output through until the player's guest head has no output
/// open (the head handed over from, or a next track faded out, has played
/// out what it held and closed); whether it closed in time.
fn guest_closes(desk: &Desk) -> bool {
    let guest_open = || {
        desk.mixer.desk().strips.iter().any(|s| {
            s.name == "Player"
                && s.inputs
                    .iter()
                    .any(|i| i.role == Role::Guest && i.format.is_some())
        })
    };
    for _ in 0..200 {
        if !guest_open() {
            return true;
        }
        desk.gate.allow(desk.gate.written() + 4_096);
        std::thread::sleep(Duration::from_millis(10));
    }
    !guest_open()
}

fn cancelled(seen: &[Event]) -> Option<&str> {
    transitions(seen).into_iter().find_map(|s| match s {
        TransitionState::Cancelled { reason } => Some(reason.as_str()),
        _ => None,
    })
}

/// Next while the next track is prepared plays it from its start (the
/// transition called off); next while it settles plays the track after it.
#[test]
fn next_while_prepared_plays_the_next_track_and_while_settling_the_one_after() {
    let _serial = serial();
    let (_dir, _, desk) = three("desk-next-prepared", 12.0, options());
    desk.player.transition(plan(0, 5.0, 1.0)).unwrap();
    heard_at(&desk, 0, 2.0);
    desk.player.next().unwrap();
    let seen = step_until(
        &desk.gate,
        &desk.events,
        |e| matches!(e, Event::TrackStarted(s) if s.index == 1),
    );
    drop(desk);
    assert_eq!(
        cancelled(&seen),
        Some("playback moved elsewhere"),
        "{seen:?}"
    );
    let start = starts(&seen).into_iter().find(|s| s.index == 1).unwrap();
    assert_eq!(start.from_frame, 0);
    assert!(!transitions(&seen)
        .iter()
        .any(|s| matches!(s, TransitionState::Started { .. })));
    // Settling: the track after.
    let (_dir, _, desk) = three("desk-next-settling", 12.0, options());
    desk.player.transition(plan(0, 5.0, 3.0)).unwrap();
    let mut seen = step_until(
        &desk.gate,
        &desk.events,
        |e| matches!(e, Event::TrackStarted(s) if s.index == 1),
    );
    desk.player.next().unwrap();
    seen.extend(step_until(
        &desk.gate,
        &desk.events,
        |e| matches!(e, Event::TrackStarted(s) if s.index == 2),
    ));
    drop(desk);
    assert!(transitions(&seen).contains(&&TransitionState::Finished));
    let third = starts(&seen).into_iter().find(|s| s.index == 2).unwrap();
    assert_eq!(third.from_frame, 0);
}

/// Stop, play and previous in the window drop the next track: its head's
/// output closes, and it is never started.
#[test]
fn stop_play_and_previous_in_the_window_drop_the_next_track() {
    let _serial = serial();
    for (k, command) in [Command::Stop, Command::Play(0), Command::Previous]
        .into_iter()
        .enumerate()
    {
        let (_dir, _, desk) = three(&format!("desk-moved-{k}"), 12.0, options());
        desk.player.transition(plan(0, 5.0, 1.0)).unwrap();
        let mut seen = heard_at(&desk, 0, 6.0);
        desk.player.send(command.clone()).unwrap();
        seen.extend(match command {
            Command::Stop => collect_until(&desk.events, |e| {
                matches!(e, Event::State(PlayState::Stopped))
            }),
            _ => step_until(
                &desk.gate,
                &desk.events,
                |e| matches!(e, Event::TrackStarted(s) if s.index == 0),
            ),
        });
        std::thread::sleep(Duration::from_millis(50));
        let open = desk
            .mixer
            .desk()
            .strips
            .iter()
            .map(|s| s.inputs.iter().filter(|i| i.format.is_some()).count())
            .sum::<usize>();
        drop(desk);
        assert_eq!(
            cancelled(&seen),
            Some("playback moved elsewhere"),
            "{command:?}: {seen:?}"
        );
        assert!(
            !seen
                .iter()
                .any(|e| matches!(e, Event::TrackStarted(s) if s.index == 1)),
            "{command:?}"
        );
        let want = usize::from(command != Command::Stop);
        assert_eq!(open, want, "{command:?}: outputs open on the desk");
    }
}

/// The next track moved away in the window: the transition is called off,
/// the next track fades out and the playing one comes back whole, without
/// a click, and plays on into what follows it now.
#[test]
fn the_next_track_moved_away_in_the_window_fades_out() {
    let _serial = serial();
    let (_dir, samples, desk) = three("desk-moved-away", 12.0, options());
    desk.player.transition(plan(0, 5.0, 1.0)).unwrap();
    let mut seen = heard_at(&desk, 0, 6.0);
    desk.player.move_track(1, 2).unwrap();
    seen.extend(step_until(
        &desk.gate,
        &desk.events,
        |e| matches!(e, Event::TrackStarted(s) if s.index == 1),
    ));
    // The next track's head, faded out, closes.
    assert!(guest_closes(&desk), "the next track's output is still open");
    let streams = desk.close();
    assert_eq!(cancelled(&seen), Some("the queue changed"), "{seen:?}");
    let got = samples_of(&streams);
    let (a, rate) = (&samples[0], RATE as usize);
    // A whole from a little after the move (its frame at 6 s and the steps
    // let through after it), to its end.
    let whole = (6 * rate + rate / 2..12 * rate - 64)
        .flat_map(|f| (0..2).map(move |c| (f, c)))
        .map(|(f, c)| (f64::from(got[2 * f + c]) - f64::from(a[2 * f + c])).abs())
        .fold(0.0, f64::max);
    assert!(whole < 0.002 * FULL, "A off by {whole} once back");
    assert!(
        steepest(&got, 5 * rate..7 * rate) < 1.2 * most_step(),
        "a click"
    );
    // The track after A now (the third file) follows it.
    let next = starts(&seen).into_iter().find(|s| s.index == 1).unwrap();
    assert!(next.info.path.ends_with("2.wav"), "{:?}", next.info.path);
}

/// The device's samples of one stream.
fn samples_of(streams: &[(jodel_output::StreamFormat, Vec<u8>)]) -> Vec<i32> {
    assert_eq!(streams.len(), 1);
    samples(&streams[0].1)
}

/// The most two low tones of 0.4 move from one sample to the next, each.
fn most_step() -> f64 {
    2.0 * 0.4 * 2.0 * std::f64::consts::PI * 290.0 / f64::from(RATE) * FULL
}

/// The largest step from one sample to the next over `frames`.
fn steepest(got: &[i32], frames: std::ops::Range<usize>) -> f64 {
    frames
        .flat_map(|f| (0..2).map(move |c| (f, c)))
        .map(|(f, c)| (f64::from(got[2 * (f + 1) + c]) - f64::from(got[2 * f + c])).abs())
        .fold(0.0, f64::max)
}

/// Two hand-overs in a row: each next track starts on the other head,
/// whose input is made the desk's Lead (the one handed over from a guest
/// again), and the microphone hears the device throughout.
#[test]
fn two_hand_overs_in_a_row_swap_the_heads_roles() {
    let _serial = serial();
    let (_dir, _, desk) = three("desk-roles", 10.0, options());
    let (mic, _feed) = Input::feed(48_000, 240);
    desk.mixer.set_mic(Some(mic));
    let roles = |desk: &Desk| -> Vec<Role> {
        let view = desk.mixer.desk();
        let group = view
            .strips
            .iter()
            .find(|s| s.name == "Player")
            .expect("the group");
        group.inputs.iter().map(|i| i.role).collect()
    };
    assert_eq!(roles(&desk), [Role::Lead, Role::Guest]);
    desk.player.transition(plan(0, 4.0, 0.5)).unwrap();
    let mut seen = step_until(
        &desk.gate,
        &desk.events,
        |e| matches!(e, Event::TrackStarted(s) if s.index == 1),
    );
    assert_eq!(roles(&desk), [Role::Guest, Role::Lead]);
    assert_eq!(desk.mixer.mic().port, MicPort::Hearing);
    seen.extend(step_until(&desk.gate, &desk.events, |e| {
        matches!(e, Event::Transition(TransitionState::Finished))
    }));
    // The head handed over from has played out what it held and closed.
    assert!(
        guest_closes(&desk),
        "the head handed over from is still open"
    );
    // B started at its 3 s: its 6 s comes 3 s on.
    desk.player.transition(plan(1, 6.0, 0.5)).unwrap();
    seen.extend(step_until(
        &desk.gate,
        &desk.events,
        |e| matches!(e, Event::TrackStarted(s) if s.index == 2),
    ));
    assert_eq!(roles(&desk), [Role::Lead, Role::Guest]);
    assert_eq!(desk.mixer.mic().port, MicPort::Hearing);
    assert_eq!(desk.mixer.mic().refused, 0);
    drop(desk);
    let heads: Vec<(usize, usize)> = starts(&seen).iter().map(|s| (s.index, s.head)).collect();
    assert!(
        heads.contains(&(1, 1)) && heads.contains(&(2, 0)),
        "{heads:?}"
    );
}

/// A head read along a map of its own refuses a transition on a desk too.
#[test]
fn a_head_read_along_a_map_refuses_a_transition() {
    let _serial = serial();
    let (_dir, _, desk) = three("desk-along", 8.0, options());
    let map = Arc::new(TimeMap::from_knots(vec![(0.0, 0.0)], vec![1.0], 0.0));
    let along = Command::Along {
        map,
        from: 0,
        tempo: TempoMode::Varispeed,
    };
    desk.player.send(along).unwrap();
    desk.player.transition(plan(0, 5.0, 1.0)).unwrap();
    let seen = collect_until(&desk.events, |e| matches!(e, Event::Transition(_)));
    drop(desk);
    assert_eq!(cancelled(&seen), Some("a track is read along a time map"));
}

/// The separator for [`a_next_track_whose_stems_are_late_waits_then_plays_whole`]:
/// separating nothing until let.
static LATE: Gated = Gated::new();

fn late() -> Option<&'static dyn Separator> {
    Some(&LATE)
}

/// On a device in real time, a next track whose stems are not separated in
/// time: it waits for them while what it would play is silent (before the
/// window), then plays its whole mix as one part, heard in the window.
#[test]
fn a_next_track_whose_stems_are_late_waits_then_plays_whole() {
    let _serial = serial();
    let _real_time = real_time();
    let dir = temp_dir("desk-late-stems");
    let (a, _) = wav(&dir, "a.wav", 9.0, tone([1_000.0, 5_000.0]));
    let (b, b_samples) = wav(&dir, "b.wav", 9.0, tone([2_000.0, 3_000.0]));
    let options = PlayerOptions {
        separator: Some(late),
        offline: false,
        position_interval: Duration::from_millis(10),
        ..options()
    };
    let desk = Desk::real_time(caps(&[RATE], &[S24In32Low]), options);
    desk.player.make_stems(MakeStems::Separate).unwrap();
    desk.play(vec![a, b]);
    // Only B from the window on.
    let mut cut = plan(0, 4.0, 0.0);
    cut.lanes = lanes(Envelope::cut(1.0, 0.0, 0.0), Envelope::cut(0.0, 1.0, 0.0));
    desk.player.transition(cut).unwrap();
    let seen = collect_until(&desk.events, |e| {
        matches!(e, Event::PartsStopped { index: 1, .. }) || matches!(e, Event::QueueEnded)
    });
    let rest = until_queue_end(&desk.events);
    LATE.allow(usize::MAX);
    let streams = desk.close();
    // It gave way to the whole mix only once its window was near: A heard
    // past its 3.7 s then.
    let heard = seen
        .iter()
        .filter_map(|e| match e {
            Event::Position(p) if p.index == 0 => Some(p.seconds),
            _ => None,
        })
        .fold(0.0, f64::max);
    assert!(
        matches!(seen.last(), Some(Event::PartsStopped { index: 1, .. })),
        "{seen:?}"
    );
    assert!(heard > 3.5, "gave way with A heard at {heard}");
    assert!(transitions(&rest).contains(&&TransitionState::Finished));
    // B, whole, in the window.
    let got = samples_of(&streams);
    let rate = RATE as usize;
    let offset = 3 * rate;
    let off = (4 * rate + rate / 2..6 * rate)
        .flat_map(|f| (0..2).map(move |c| (f, c)))
        .map(|(f, c)| {
            (f64::from(got[2 * f + c]) - f64::from(b_samples[2 * (f - offset) + c])).abs()
        })
        .fold(0.0, f64::max);
    assert!(off < 0.01 * FULL, "B off its whole mix by {off}");
    let _ = STEMS;
}

/// On a device in real time, next in the window: the playing track fades
/// out from where the desk mixes it, so nothing clicks however far ahead of
/// what is heard the desk has mixed.
#[test]
fn in_real_time_next_mid_mix_does_not_click() {
    let _serial = serial();
    let _real_time = real_time();
    let dir = temp_dir("desk-next-paced");
    let (a, _) = wav(&dir, "a.wav", 9.0, tone([200.0, 230.0]));
    let (b, _) = wav(&dir, "b.wav", 9.0, tone([170.0, 190.0]));
    let options = PlayerOptions {
        offline: false,
        position_interval: Duration::from_millis(10),
        ..options()
    };
    let desk = Desk::paced(caps(&[RATE], &[S24In32Low]), options);
    desk.play(vec![a, b]);
    desk.player.transition(plan(0, 4.0, 0.0)).unwrap();
    collect_until(
        &desk.events,
        |e| matches!(e, Event::Position(p) if p.index == 0 && p.seconds >= 4.8),
    );
    desk.player.next().unwrap();
    let seen = until_queue_end(&desk.events);
    let streams = desk.close();
    assert_eq!(cancelled(&seen), Some("cut to the next track"), "{seen:?}");
    let got = samples_of(&streams);
    let rate = RATE as usize;
    let step = steepest(&got, 3 * rate..7 * rate);
    assert!(step < 1.2 * most_step(), "a step of {step}");
}
