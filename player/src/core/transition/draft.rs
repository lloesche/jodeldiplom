//! The transition set up on the Transition tab, from its cues to the
//! engine: the cues `enter` sets, the window and the bars they make, how
//! the beats pair, the plan made of them, and how it is armed, sent again,
//! called off, skipped into and handed over as the engine reports how it
//! goes ([`Status`]).

use jodel_playback::{Command, TransitionPlan, TransitionState};
use jodel_rhythm::Analysis;

use super::tab::{block, block_mut, BlockState, Cue};
use super::viewport::{bar_of, cursor, frame_ms, window};
use crate::core::lanes::{self, rhythmed, Block};
use crate::core::transition::{self as tr, Draft, Status};
use crate::core::update::say_info;
use crate::core::{Effect, State};

/// The least time before a cue's first bar that `g` arms a transition at
/// it, in seconds after what is heard (the engine writes ahead of it).
pub const SOONEST: f64 = 1.0;

/// Seconds after what is heard that a skip mixes in at the soonest, and
/// where `enter` cues the playing track while its cursor follows: the
/// engine gets the tracks ready over [`jodel_playback`]'s lead.
pub const GO_AFTER: f64 = 4.5;

/// The transition armed or under way from the playing track into the
/// next, as sent.
pub fn transition_into(state: &State) -> Option<&tr::Sent> {
    let sent = state.transition.draft.in_force()?;
    let playing = state.playing.track.as_ref()?.index;
    (sent.plan.from == playing).then_some(sent)
}

/// The cue `enter` sets on a block: its selection's bars, else
/// [`tr::DEFAULT_BARS`] from its cursor's bar (the playing track's, while
/// its cursor follows the playhead: from the first bar the engine can still
/// get ready for, [`GO_AFTER`] on).
pub fn cue_here(state: &State, which: Block, heard: f64) -> Option<Cue> {
    let a = &rhythmed(state, which)?.analysis;
    let b = block(state, which);
    let cursor = cursor(state, which, heard)?;
    let bar_start = |beat: usize| {
        bar_of(a, beat)
            .checked_sub(1)
            .map_or(beat, |k| a.downbeats[k])
    };
    match (b.anchor, b.cursor, which) {
        (Some(anchor), _, _) => {
            let (lo, hi) = (anchor.min(cursor), anchor.max(cursor));
            let bars = (bar_of(a, hi) + 1)
                .saturating_sub(bar_of(a, lo).max(1))
                .max(1);
            Some(Cue {
                beat: bar_start(lo),
                bars,
                chosen: true,
            })
        }
        (None, None, Block::Now) => Some(Cue {
            beat: first_bar_after(a, heard + GO_AFTER)?,
            bars: tr::DEFAULT_BARS,
            chosen: false,
        }),
        _ => Some(Cue {
            beat: bar_start(cursor),
            bars: tr::DEFAULT_BARS,
            chosen: false,
        }),
    }
}

/// The seconds a block's cue covers on its own track, and its bars.
pub fn cue_window(state: &State, which: Block) -> Option<(f64, f64, usize)> {
    let a = &rhythmed(state, which)?.analysis;
    let cue = block(state, which).cue?;
    tr::window(a, cue.beat, cue.bars)
}

/// The bars the transition mixes over, in the playing track's bars, with
/// `now` the playing track's cue: the longer of the two cues (the next
/// track's counted in the playing track's bars, by how the beats pair),
/// of those selected if either was; in whole bars of the slower track.
fn mixed_bars(state: &State, now: Option<Cue>) -> usize {
    let per = pairing(state);
    let next = block(state, Block::Next).cue.map(|c| Cue {
        bars: (c.bars * per.bars_per_step()).div_ceil(per.next_bars_per_step()),
        ..c
    });
    let cues = [now, next];
    let chosen = cues.iter().flatten().any(|c| c.chosen);
    let bars = cues
        .iter()
        .flatten()
        .filter(|c| c.chosen || !chosen)
        .map(|c| c.bars)
        .max()
        .unwrap_or(tr::DEFAULT_BARS);
    let every = per.bars_per_step();
    bars.max(1).div_ceil(every) * every
}

/// The window on the playing track: from its cue's first bar, over the
/// bars of the longer cue ([`mixed_bars`]). `None` without a cue there.
pub fn draft_window(state: &State) -> Option<(f64, f64, usize)> {
    let a = &rhythmed(state, Block::Now)?.analysis;
    let cue = block(state, Block::Now).cue?;
    tr::window(a, cue.beat, mixed_bars(state, Some(cue)))
}

/// `enter`: the focused block's cue, from its selection or its cursor
/// ([`cue_here`]). Playback stays as it is; a transition armed takes the
/// new cue.
pub(super) fn set_cue(state: &mut State, fx: &mut Vec<Effect>, heard: f64) {
    let which = state.transition.focus;
    if which == Block::Prev {
        return;
    }
    if matches!(state.transition.draft.status, Status::Mixing(_)) {
        return say_info(state, "A transition is under way: it plays as it is");
    }
    let Some(cue) = cue_here(state, which, heard) else {
        return say_info(state, "No bars here yet");
    };
    let bar = rhythmed(state, which).map_or(0, |r| bar_of(&r.analysis, cue.beat));
    let b = block_mut(state, which);
    b.cue = Some(cue);
    b.anchor = None;
    let track = match which {
        Block::Now => "the playing track",
        _ => "the next track",
    };
    let bars = cue.bars;
    let last = bar + bars - 1;
    let what = if bars == 1 {
        format!("bar {bar}")
    } else {
        format!("bars {bar}–{last}")
    };
    let other = match which {
        Block::Now => Block::Next,
        _ => Block::Now,
    };
    let then = if block(state, other).cue.is_none() {
        let there = match other {
            Block::Now => "the playing track",
            _ => "the next track",
        };
        format!("; set one in {there} too")
    } else if state.transition.draft.status == Status::Draft {
        "; g arms the transition".to_owned()
    } else {
        String::new()
    };
    say_info(state, format!("Cue on {track}: {what}{then}"));
    if state.transition.draft.status == Status::Armed {
        // The transition armed moves to the new cue (if it can).
        match cued_plan(state) {
            Ok(sent) => send(state, fx, sent),
            Err(why) => say_info(state, format!("The armed one stays: {why}")),
        }
    }
}

/// How the beats pair: for the transition sent while it is armed or
/// mixing, else as `r` set it, else by the two tracks' tempi.
pub fn pairing(state: &State) -> tr::Pairing {
    match state.transition.draft.in_force() {
        Some(sent) => sent.pairing,
        None => chosen_pairing(state),
    }
}

/// How the beats are to pair: as `r` set it, else by the two tracks'
/// tempi.
fn chosen_pairing(state: &State) -> tr::Pairing {
    if let Some(p) = state.transition.draft.pairing {
        return p;
    }
    let (a, b) = meeting_tempi(state);
    tr::pairing_for(a, b)
}

/// The two tracks' tempi where they meet: the playing track's from its
/// cursor's bar (or what is heard), the next one's from its cue (or its
/// cursor, or its first bar), over [`tr::TEMPO_SPAN`].
pub fn meeting_tempi(state: &State) -> (Option<f64>, Option<f64>) {
    let heard = state.heard();
    let near = |which: Block, at: Option<usize>, or: f64| {
        rhythmed(state, which).and_then(|r| {
            let a = &r.analysis;
            let from = at.and_then(|i| a.beats.get(i).copied()).unwrap_or(or);
            tr::beat_tempo_near(a, from, tr::TEMPO_SPAN)
        })
    };
    let now = block(state, Block::Now);
    let next = block(state, Block::Next);
    let first_b = rhythmed(state, Block::Next)
        .and_then(|r| r.analysis.downbeats.first().map(|&d| r.analysis.beats[d]))
        .unwrap_or(0.0);
    let cued = |b: &BlockState| b.cue.map(|c| c.beat).or(b.anchor).or(b.cursor);
    (
        near(Block::Now, cued(&now), heard),
        near(Block::Next, cued(&next), first_b),
    )
}

/// The first beat of the first bar at or after `t`.
fn first_bar_after(a: &Analysis, t: f64) -> Option<usize> {
    a.downbeats.iter().copied().find(|&d| a.beats[d] >= t)
}

/// The bars of the transition shown: the plan sent while it is armed or
/// mixing, else the window set up.
pub fn draft_bars(state: &State) -> usize {
    if let Some(sent) = state.transition.draft.in_force() {
        return sent.bars;
    }
    match draft_window(state) {
        Some(w) => w.2,
        None if block(state, Block::Next).cue.is_some() => mixed_bars(state, None),
        None => tr::DEFAULT_BARS,
    }
}

/// The transition shown and its bars: the plan sent while it is armed or
/// mixing, else the one set up.
pub fn shown_plan(state: &State) -> Option<(TransitionPlan, usize)> {
    match state.transition.draft.in_force() {
        Some(sent) => Some((sent.plan.clone(), sent.bars)),
        None => draft_plan(state, None).ok(),
    }
}

/// The plan of the transition set up and its window's bars: the window
/// ([`draft_window`], or from the first bar at least `soon` seconds after
/// what is heard), the next track's cue (or, with `soon`, its cursor or
/// first bar), the lanes and the tempo mode.
pub fn draft_plan(state: &State, soon: Option<f64>) -> Result<(TransitionPlan, usize), String> {
    let from = state
        .playing
        .track
        .as_ref()
        .ok_or("nothing is playing")?
        .index;
    let a = &rhythmed(state, Block::Now)
        .ok_or("the playing track is not analysed yet")?
        .analysis;
    let b = &rhythmed(state, Block::Next)
        .ok_or("the next track is not analysed yet")?
        .analysis;
    if b.beats.is_empty() || a.downbeats.is_empty() {
        return Err("no bars to mix on".into());
    }
    let window = match soon {
        Some(after) => {
            let bars = draft_window(state).map_or(tr::DEFAULT_BARS, |w| w.2);
            let first = first_bar_after(a, state.heard() + after)
                .ok_or("the playing track has no bar left to start on")?;
            tr::window(a, first, bars)
        }
        None => Some(draft_window(state).ok_or("no cue in the playing track")?),
    }
    .ok_or("no bars to mix on")?;
    let next = block(state, Block::Next);
    let entry = match (next.cue, soon) {
        (Some(cue), _) => cue.beat,
        (None, None) => return Err("no cue in the next track".into()),
        // A skip: where its cursor was put, else its first bar.
        (None, Some(_)) => next
            .cursor
            .unwrap_or_else(|| b.downbeats.first().copied().unwrap_or(0)),
    };
    let lanes = state.transition.draft.lanes(window.2);
    let per = pairing(state);
    let plan = tr::plan(
        from,
        a,
        b,
        window,
        entry,
        lanes,
        state.transition.draft.tempo,
        per,
    );
    Ok((plan, window.2))
}

/// The transition at the two cues, ready to send.
fn cued_plan(state: &State) -> Result<tr::Sent, String> {
    let missing = |b: Block| block(state, b).cue.is_none();
    match (missing(Block::Now), missing(Block::Next)) {
        (true, true) => return Err("set a cue in both tracks first (enter on their bars)".into()),
        (true, false) => return Err("set a cue in the playing track first".into()),
        (false, true) => return Err("set a cue in the next track first".into()),
        (false, false) => {}
    }
    let (plan, bars) = draft_plan(state, None)?;
    if plan.start < state.heard() + SOONEST {
        return Err("the playing track's cue has passed: set a later one".into());
    }
    Ok(tr::Sent::new(
        plan,
        bars,
        state.transition.draft.preset,
        pairing(state),
    ))
}

/// `g`: arms the transition at the two cues, or calls the armed one off.
pub(super) fn go(state: &mut State, fx: &mut Vec<Effect>) {
    match state.transition.draft.status {
        // Calling it off does not cut a mix under way off (moving playback
        // does).
        Status::Mixing(_) => say_info(state, "A transition is under way"),
        Status::Armed => call_off(state, fx),
        Status::Draft => match cued_plan(state) {
            Ok(sent) => {
                let at = sent.plan.start - state.heard();
                send(state, fx, sent);
                say_info(state, format!("Transition armed: it starts in {at:.0} s"));
            }
            Err(why) => say_info(state, format!("No transition: {why}")),
        },
    }
}

/// Calls off the transition armed (its cues stay).
pub(super) fn call_off(state: &mut State, fx: &mut Vec<Effect>) {
    fx.push(Effect::Engine(Command::CancelTransition));
    state.transition.draft.status = Status::Draft;
    state.transition.draft.sent = None;
    say_info(state, "Transition called off");
}

/// Sends `sent` to the engine, and keeps it.
fn send(state: &mut State, fx: &mut Vec<Effect>, sent: tr::Sent) {
    fx.push(Effect::Engine(Command::Transition(Box::new(
        sent.plan.clone(),
    ))));
    if state.transition.draft.status == Status::Draft {
        // Newly armed: the next track's view runs along with it.
        block_mut(state, Block::Next).start = None;
    }
    state.transition.draft.sent = Some(sent);
}

/// The fades (`lanes`) or the tempo mode changed while armed: the engine
/// takes the plan it has with them (its window and entry stay).
pub(super) fn rearm(state: &mut State, fx: &mut Vec<Effect>, lanes: bool) {
    let d = &state.transition.draft;
    if d.status != Status::Armed {
        return;
    }
    let Some(mut sent) = d.sent.clone() else {
        return;
    };
    if lanes {
        sent.plan.lanes = d.lanes(sent.bars);
        sent.preset = d.preset;
    }
    sent.plan.tempo = d.tempo;
    send(state, fx, sent);
}

/// The pairing changed (`r`): an armed transition is sent again with its
/// beats paired the new way, from the same window start and entry (the
/// window made whole bars of the slower track).
pub(super) fn repair(state: &mut State, fx: &mut Vec<Effect>) {
    let d = &state.transition.draft;
    if let (Status::Armed, Some(sent)) = (&d.status, d.sent.clone()) {
        let per = chosen_pairing(state);
        resend(state, fx, sent, per);
    }
}

/// `sent` again, its beats paired by `per`.
fn resend(state: &mut State, fx: &mut Vec<Effect>, sent: tr::Sent, per: tr::Pairing) {
    let (Some(a), Some(b)) = (rhythmed(state, Block::Now), rhythmed(state, Block::Next)) else {
        return;
    };
    let (a, b) = (&a.analysis, &b.analysis);
    let first = a.beats.partition_point(|&t| t < sent.plan.start - 1e-6);
    let entry = b.beats.partition_point(|&t| t < sent.plan.entry - 1e-6);
    let bars = sent.bars.div_ceil(per.bars_per_step()) * per.bars_per_step();
    let Some(window) = tr::window(a, first, bars) else {
        return;
    };
    let lanes = state.transition.draft.lanes(window.2);
    let plan = tr::plan(
        sent.plan.from,
        a,
        b,
        window,
        entry,
        lanes,
        state.transition.draft.tempo,
        per,
    );
    send(state, fx, tr::Sent::new(plan, window.2, sent.preset, per));
}

/// The next track key: with a transition armed, cut to the next track
/// now (the engine drops the transition); with one mixing, cut into the
/// next track where it is heard (the engine plays it on alone from there,
/// the playing track cut off); else, by the setting, mix into it from the
/// first bar the engine can get ready for, or cut to it.
pub(crate) fn skip(state: &mut State, fx: &mut Vec<Effect>) {
    match state.transition.draft.status {
        Status::Mixing(_) if transition_into(state).is_some() => {
            say_info(state, "Cut into the next track where it is");
        }
        Status::Armed => {}
        _ => {
            if state.config.skip_mix {
                match skip_plan(state, state.config.skip_preset) {
                    Ok(sent) => {
                        let at = sent.plan.start - state.heard();
                        let name = crate::core::update::skip_preset_name(state);
                        say_info(
                            state,
                            format!(
                                "Mixing into the next track in {at:.0} s ({name}); N again cuts"
                            ),
                        );
                        return send(state, fx, sent);
                    }
                    Err(why) => {
                        say_info(state, format!("Cut to the next track: {why}"));
                    }
                }
            }
        }
    }
    fx.push(Effect::Engine(Command::Next));
}

/// The transition a skip starts: from the first bar at least [`GO_AFTER`]
/// on, over the bars set up (or [`tr::DEFAULT_BARS`]), into the next
/// track's cue (or first bar), faded by `preset` (`None`: the Transition tab's
/// preset as it is set, its fades as edited).
fn skip_plan(state: &State, preset: Option<tr::Preset>) -> Result<tr::Sent, String> {
    let (mut plan, bars) = draft_plan(state, Some(GO_AFTER))?;
    let d = &state.transition.draft;
    let (lanes, preset) = match preset {
        Some(p) => (p.lanes(bars), p),
        None => (d.lanes(bars), d.preset),
    };
    plan.lanes = lanes;
    Ok(tr::Sent::new(plan, bars, preset, pairing(state)))
}

/// A new engine: nothing of a transition the old one had is left (it will
/// say nothing more of it).
pub fn engine_restarted(state: &mut State) {
    let d = &mut state.transition.draft;
    d.status = Status::Draft;
    d.sent = None;
}

/// The engine says how the transition goes.
pub(crate) fn transition_event(state: &mut State, event: TransitionState) {
    let d = &mut state.transition.draft;
    match event {
        TransitionState::Armed => d.status = Status::Armed,
        // Only for one it has: a late word of one called off is not taken.
        TransitionState::Started {
            head,
            output_frame,
            origin,
        } => {
            if d.sent.is_some() {
                let start = crate::core::HeadFrame {
                    head,
                    frame: output_frame,
                };
                d.status = Status::Mixing(tr::MixStart { start, origin });
            }
        }
        TransitionState::Finished => {
            *d = Draft {
                preset: d.preset,
                tempo: d.tempo,
                ..Draft::default()
            };
            say_info(state, "Transition done");
        }
        TransitionState::Cancelled { reason } => {
            d.status = Status::Draft;
            d.sent = None;
            // Playback moved on by the user's own key or click: no news.
            if reason != "playback moved elsewhere" {
                say_info(state, format!("Transition called off: {reason}"));
            }
        }
    }
}

/// Track `index` starts: when it is the next track (handed over to, or
/// skipped to), its block (its own zoom with it) becomes the playing
/// track's, following the playhead, its cue (where it was mixed in) gone.
/// When a transition hands over to it, the playing track's block is kept
/// as it is seen now (its view held), and the next track's block slides up
/// into its place ([`super::viewport::slide`]).
pub(crate) fn track_started(state: &mut State, index: usize) {
    let carried = (lanes::index(state, Block::Next) == Some(index)).then(|| BlockState {
        start: None,
        cursor: None,
        anchor: None,
        cue: None,
        ..block(state, Block::Next)
    });
    let handing_over = transition_into(state).is_some_and(|s| s.plan.into == index)
        && matches!(state.transition.draft.status, Status::Mixing(_));
    if handing_over {
        hand_over(state);
    }
    if let Some(now) = carried {
        state.transition.now = now;
        state.transition.next = BlockState::default();
    }
}

/// A transition hands over: the playing track's block kept as it is seen
/// now, the next one's sliding up into its place.
fn hand_over(state: &mut State) {
    let heard = state.heard();
    let (start, _) = window(state, Block::Now, heard);
    state.transition.prev = BlockState {
        start: Some(start),
        cursor: None,
        anchor: None,
        ..block(state, Block::Now)
    };
    state.transition.slide = Some(frame_ms(state));
    state.transition.focus = Block::Now;
    state.transition.pan = None;
}
