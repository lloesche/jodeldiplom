//! The stems through a transition: the desk mixes its two tracks, each on
//! a head of the player's own, so a stem's key lands on its bar 0.1 s
//! ahead as at any other time, on a frame of the head on air (worked out
//! from the positions told of it): the playing track's head through the
//! mix, the next track's from its hand-over on. The stems are made while a
//! transition is armed or mixed, whatever shows.

mod common;

use jodel_playback::{Command, Event, MakeStems, Position, TrackStart, TransitionState};
use jodeldiplom::core::desk::{DeskChange, StripKind};
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::lanes::{self, Block};
use jodeldiplom::core::stems::{bar, make_wanted, DESK_SOON};
use jodeldiplom::core::transition::{MixStart, Status};
use jodeldiplom::core::{update, Action, Effect, HeadFrame, Side, State};

fn op(state: &mut State, op: Op) -> Vec<Effect> {
    update(state, Action::Op(op))
}

fn desk(fx: &[Effect]) -> Vec<DeskChange> {
    fx.iter()
        .filter_map(|e| match e {
            Effect::Desk(c) => Some(*c),
            _ => None,
        })
        .collect()
}

fn engine(fx: &[Effect]) -> Vec<Command> {
    fx.iter()
        .filter_map(|e| match e {
            Effect::Engine(c) => Some(c.clone()),
            _ => None,
        })
        .collect()
}

/// The Transition tab with both tracks cued and the transition armed:
/// mixing too (its window heard), with `mixing`.
fn with_transition(mixing: bool) -> State {
    let mut state = common::transition_state(120, 40);
    let focus = state.transition.focus;
    for key in [Op::Up, Op::Accept, Op::Down, Op::Accept] {
        op(&mut state, key);
    }
    if focus == Block::Now {
        op(&mut state, Op::Up);
    }
    op(&mut state, Op::TransitionGo);
    update(
        &mut state,
        Action::Engine(Event::Transition(TransitionState::Armed)),
    );
    if mixing {
        update(
            &mut state,
            Action::Engine(Event::Transition(TransitionState::Started {
                head: 0,
                output_frame: 7,
                origin: 0,
            })),
        );
    }
    state
}

/// Track `index` told heard `seconds` in, at output frame `frame` of
/// `head`, the file at `rate`.
fn position(index: usize, head: usize, seconds: f64, frame: u64, rate: f64) -> Action {
    Action::Engine(Event::Position(Position {
        index,
        head,
        seconds,
        frame: (seconds * rate) as u64,
        output_frame: frame,
        key: 0,
    }))
}

/// Two positions of the playing track on `head`, 0.1 s apart, the second
/// at `frame` (its output at the track's own rate between them), heard
/// there; the bar a change lands on then (the first at least
/// [`DESK_SOON`] on), and its frame.
fn told(state: &mut State, head: usize, seconds: f64, frame: u64) -> (f64, u64) {
    let track = state.playing.track.clone().unwrap();
    let rate = f64::from(track.output.format.rate);
    let step = (0.1 * rate).round() as u64;
    update(
        state,
        position(track.index, head, seconds - 0.1, frame - step, rate),
    );
    update(state, position(track.index, head, seconds, frame, rate));
    state.drawn_heard = Some(seconds);
    let a = &lanes::rhythmed(state, Block::Now).unwrap().analysis;
    let at = a
        .downbeats
        .iter()
        .map(|&d| a.beats[d])
        .find(|&t| t >= seconds + DESK_SOON)
        .unwrap();
    let speed = step as f64 / 0.1;
    (at, (frame as f64 + (at - seconds) * speed).round() as u64)
}

#[test]
fn the_stems_are_made_while_a_transition_is_armed_or_mixed() {
    for mixing in [false, true] {
        let mut state = with_transition(mixing);
        // No tab that shows them, nothing touched: still made, both tracks
        // faded by their lanes stem by stem.
        state.side = Side::Browser;
        assert!(!state.desk.stems_touched());
        assert_eq!(make_wanted(&state), MakeStems::Separate, "mixing {mixing}");
        // Done: none.
        update(
            &mut state,
            Action::Engine(Event::Transition(TransitionState::Finished)),
        );
        assert_eq!(make_wanted(&state), MakeStems::Never, "mixing {mixing}");
    }
}

/// A stem's key during a mix: on its bar 0.1 s ahead (the desk plays both
/// tracks' stems, no engine mix to wait for), a frame of the playing
/// track's head; after the hand-over, of the next track's head, from its
/// own positions; nothing but the desk is told.
#[test]
fn a_stem_key_during_a_mix_lands_on_its_bar_of_the_head_on_air() {
    let mut state = with_transition(true);
    state.side = Side::Browser;
    let index = state.playing.track.as_ref().unwrap().index;
    let heard = state.heard();
    let (at, frame) = told(&mut state, 0, heard, 1_000_000);
    assert_eq!(bar(&state).map(|b| (b.seconds, b.at.head)), Some((at, 0)));
    let fx = op(&mut state, Op::ToggleDrums);
    assert_eq!(
        desk(&fx),
        [DeskChange::MuteAt(
            StripKind::Drums,
            true,
            HeadFrame { head: 0, frame }
        )]
    );
    assert!(engine(&fx).is_empty(), "{fx:?}");
    op(&mut state, Op::ToggleDrums);
    // Handed over: the next track on the other head, its frames counted
    // from that head's first. No bar until two positions of it are told.
    let track = state.playing.track.clone().unwrap();
    update(
        &mut state,
        Action::Engine(Event::TrackStarted(Box::new(TrackStart {
            index: index + 1,
            head: 1,
            info: track.info.clone(),
            route: track.route.clone(),
            output: track.output.clone(),
            device: "DAC".into(),
            bit_perfect: false,
            output_frame: 88_200,
            from_frame: (20.0 * f64::from(track.info.rate)) as u64,
        }))),
    );
    assert_eq!(state.playing.track.as_ref().unwrap().index, index + 1);
    assert!(matches!(state.transition.draft.status, Status::Mixing(_)));
    assert!(bar(&state).is_none());
    let (at, frame) = told(&mut state, 1, 20.1, 100_000);
    let fx = op(&mut state, Op::ToggleBass);
    assert_eq!(
        desk(&fx),
        [DeskChange::MuteAt(
            StripKind::Bass,
            true,
            HeadFrame { head: 1, frame }
        )]
    );
    assert!(
        (20.1 + DESK_SOON..20.1 + DESK_SOON + 2.5).contains(&at),
        "{at}"
    );
}

/// The Mixer tab's PLAYER strip during a mix: both heads, read only, the
/// one on air (LEAD) and the next track's (NEXT), each with the gain its
/// automation puts on it, and the late frames the next track's dropped by
/// the ring.
#[test]
fn the_player_strip_shows_both_heads_during_a_mix() {
    use jodel_output::mixer::{AutomationView, Role};
    use jodeldiplom::view::{self, CellPictures, Scene};
    let mut state = common::desk_state(120, 36);
    let automation = |main: f64| AutomationView {
        input: 1.0,
        main,
        parts: vec![main; 4],
        settled: false,
    };
    let player = state.desk.views[StripKind::Player.index()]
        .as_mut()
        .unwrap();
    let lead = &mut player.inputs[0];
    lead.automation = Some(automation(0.5));
    let mut next = lead.clone();
    next.role = Role::Guest;
    next.automation = Some(automation(0.25));
    next.dropped = 12;
    player.inputs.push(next);
    let draw = |state: &State| {
        let (cols, rows) = state.size;
        let backend = ratatui::backend::TestBackend::new(cols, rows);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        let spectrum = jodeldiplom::spectrum::SpectrumFrame::default();
        let scene = Scene {
            state,
            spectrum: &spectrum,
            pointer: None,
            heard: 0.0,
        };
        terminal
            .draw(|frame| view::draw(frame, &scene, &mut CellPictures))
            .unwrap();
        view::buffer_text(terminal.backend().buffer())
    };
    let text = draw(&state);
    for want in ["● LEAD −6.0", "● NEXT −12", "↓12"] {
        assert!(text.contains(want), "no {want:?} in\n{text}");
    }
    // Once the mix is over: the head on air alone, as before.
    let player = state.desk.views[StripKind::Player.index()]
        .as_mut()
        .unwrap();
    player.inputs.truncate(1);
    player.inputs[0].automation = None;
    let text = draw(&state);
    assert!(text.contains("│● LEAD     │"), "{text}");
    assert!(text.contains("│ring ▮▮▮▮▯▯│"), "{text}");
    assert!(!text.contains("NEXT"), "{text}");
}

/// A mix under way, from frame `w0` of head 0, the next track's frame 0 on
/// its `origin`.
fn mixing_from(state: &mut State, w0: u64, origin: u64) {
    state.transition.draft.status = Status::Mixing(MixStart {
        start: HeadFrame { head: 0, frame: w0 },
        origin,
    });
}

/// Between the engine's hand-over and the next track's start being heard,
/// the positions told are still the playing track's but on the next
/// track's head (its frames, `origin` fewer): no speed is worked out across
/// the two heads, and a stem's bar is on the new head's frames.
#[test]
fn positions_on_the_new_head_before_its_track_starts_time_bars_on_it() {
    let mut state = with_transition(true);
    state.side = Side::Browser;
    let track = state.playing.track.clone().unwrap();
    let rate = f64::from(track.output.format.rate);
    let heard = state.heard();
    told(&mut state, 0, heard, 1_000_000);
    let speed = state.playing.speed.unwrap();
    // Swapped: the same track told on head 1, its frames rebased by a
    // tenth of a second only, so they still run on past the last told.
    let origin = (0.1 * rate) as u64;
    let rebased = 1_000_000 + (0.2 * rate) as u64 - origin;
    update(
        &mut state,
        position(track.index, 1, heard + 0.2, rebased, rate),
    );
    state.drawn_heard = Some(heard + 0.2);
    assert_eq!(state.playing.speed, Some(speed), "a speed across heads");
    let b = bar(&state).expect("a bar");
    assert_eq!(b.at.head, 1);
    let want = rebased as f64 + (b.seconds - (heard + 0.2)) * speed;
    assert!((b.at.frame as f64 - want).abs() <= 1.0, "{b:?} {want}");
    let fx = op(&mut state, Op::ToggleDrums);
    assert_eq!(
        desk(&fx),
        [DeskChange::MuteAt(StripKind::Drums, true, b.at)]
    );
}

/// The voice's blocks around a hand-over: the microphone's tap moves to
/// the next track's head before the positions do (and blocks of the old
/// head can come after them), so a block is timed through whichever head
/// puts it near what is heard, and one near neither is dropped.
#[test]
fn voice_blocks_are_timed_through_their_head_around_a_hand_over() {
    use jodeldiplom::core::voice_time::VoiceClock;
    let mut state = with_transition(true);
    let track = state.playing.track.clone().unwrap();
    let rate = f64::from(track.output.format.rate);
    let heard = state.heard();
    let origin = 2_000_000;
    told(&mut state, 0, heard, 3_000_000);
    mixing_from(&mut state, 2_500_000, origin);
    let clock = VoiceClock::of(&state, track.index).unwrap();
    let sung = 3_000_000.0 - 0.5 * rate;
    // The playing head's frame, and the same moment on the next track's.
    let on_a = clock.seconds(sung).unwrap();
    assert!((on_a - (heard - 0.5)).abs() < 1e-9, "{on_a}");
    let on_b = clock.seconds(sung - origin as f64).unwrap();
    assert!((on_b - on_a).abs() < 1e-9, "{on_b} {on_a}");
    // Far from either: none, and the block after it does not follow on.
    assert_eq!(clock.seconds(sung + 20.0 * rate), None);
    let block = |frame: f64| jodel_output::VoiceBlock {
        stream_frame: frame as u64,
        rate: 48_000,
        continues: true,
        samples: Vec::new(),
    };
    let blocks = [block(sung), block(sung + 20.0 * rate), block(sung + 512.0)];
    let placed: Vec<bool> = clock.place(&blocks).iter().map(|p| p.2).collect();
    assert_eq!(placed, [true, false]);
    // Told on the next track's head now (positions of the same track,
    // rebased): a block of the old head comes back by `origin` the other
    // way.
    told(
        &mut state,
        1,
        heard + 0.2,
        3_000_000 + (0.2 * rate) as u64 - origin,
    );
    let clock = VoiceClock::of(&state, track.index).unwrap();
    let late = clock.seconds(sung).unwrap();
    assert!((late - on_a).abs() < 1e-6, "{late} {on_a}");
    // No mix: one head, and a block from elsewhere is dropped.
    state.transition.draft.status = Status::Draft;
    let clock = VoiceClock::of(&state, track.index).unwrap();
    assert_eq!(clock.seconds(sung), None);
}
