//! The Transition tab: the track playing and the next one, their waveforms
//! and bars over time, a cursor on the beats, a selection of bars, and the
//! fixes to a beat grid ([`crate::core::grid`]). The lanes themselves, and
//! which track each block shows, are [`crate::core::lanes`].
//!
//! Here are the tab's state, its keys and its grid fixes. What each block
//! shows, and how its cursor and view move, is [`viewport`]; the
//! transition set up at the cues, armed and sent to the engine, is
//! [`draft`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

use jodel_rhythm::Analysis;

use super::{draft, viewport};
use crate::core::grid::Grid;
use crate::core::keymap::{self, Context, Op};
use crate::core::lanes::{path, rhythmed, Block};
use crate::core::transition::{self as tr, Draft, Edit, Status};
use crate::core::{DoubleClick, Effect, Rhythm, Rhythmed, Side, State};
use jodel_playback::stems::STEMS;
use jodel_playback::TempoMode;

/// Bars a page moves.
pub const PAGE_BARS: isize = 8;
/// What a nudge moves the grid by, in seconds.
pub const NUDGE: f64 = 0.005;

/// A drag of a block's view with the secondary button.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pan {
    /// The block.
    pub block: Block,
    /// Seconds a unit across is worth, in the window's units (negative:
    /// the view follows the pointer, as on the whole track; positive: the
    /// waveform does).
    pub per_unit: f64,
    /// Where across it started.
    pub at: f32,
    /// The view's start then.
    pub start: f64,
}

/// A transition's cue on a track: the bars it is to mix over there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cue {
    /// The first beat of its first bar.
    pub beat: usize,
    /// Its bars (at least one).
    pub bars: usize,
    /// Its bars were selected (else [`tr::DEFAULT_BARS`] from the cursor).
    pub chosen: bool,
}

/// What one block is set to, for the track it was set for.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BlockState {
    /// The track (a block for another track starts afresh).
    pub path: Option<PathBuf>,
    /// The beat the cursor is on; `None` until it moves (at the playhead's
    /// bar for the playing track, the first bar for the next).
    pub cursor: Option<usize>,
    /// Where a selection started (a beat), while there is one.
    pub anchor: Option<usize>,
    /// The time the view starts at; `None`: following (or, for the next
    /// track, from its first bar).
    pub start: Option<f64>,
    /// The bars a transition is to mix over on this track, once set
    /// (`enter`).
    pub cue: Option<Cue>,
    /// Its own zoom (an index into [`viewport::ZOOMS`]), set by the wheel
    /// over it; `None`: the tab's ([`TransitionTab::zoom`]).
    pub zoom: Option<usize>,
}

/// The Transition tab's state.
#[derive(Clone, Debug, PartialEq)]
pub struct TransitionTab {
    /// The block the keys act on.
    pub focus: Block,
    /// The playing track's block.
    pub now: BlockState,
    /// The next track's block.
    pub next: BlockState,
    /// The index into [`viewport::ZOOMS`].
    pub zoom: usize,
    /// The last press on the lanes, its block and beat, to tell a double
    /// click.
    pub last_press: DoubleClick<(Block, usize)>,
    /// The transition being set up.
    pub draft: Draft,
    /// The track before the playing one, as its block was when a
    /// transition handed over from it (its view held where it was).
    pub prev: BlockState,
    /// When ([`viewport::frame_ms`]) a transition handed over: the next
    /// track's block slides up into the playing track's place from then.
    pub slide: Option<u64>,
    /// A drag of a view with the secondary button, under way.
    pub pan: Option<Pan>,
}

impl Default for TransitionTab {
    fn default() -> TransitionTab {
        TransitionTab {
            focus: Block::Now,
            now: BlockState::default(),
            next: BlockState::default(),
            zoom: viewport::DEFAULT_ZOOM,
            last_press: DoubleClick::default(),
            draft: Draft::default(),
            prev: BlockState::default(),
            slide: None,
            pan: None,
        }
    }
}

impl TransitionTab {
    /// A Transition tab whose transitions start from `preset` and `tempo` (as
    /// the settings remember them).
    pub fn with(preset: tr::Preset, tempo: TempoMode) -> TransitionTab {
        let mut tab = TransitionTab::default();
        tab.draft.preset = preset;
        tab.draft.tempo = tempo;
        tab
    }
}

impl TransitionTab {
    /// A block's state, as stored (see [`block`] for it as it applies).
    pub fn get(&self, block: Block) -> &BlockState {
        match block {
            Block::Now => &self.now,
            Block::Next => &self.next,
            Block::Prev => &self.prev,
        }
    }

    fn get_mut(&mut self, block: Block) -> &mut BlockState {
        match block {
            Block::Now => &mut self.now,
            Block::Next => &mut self.next,
            Block::Prev => &mut self.prev,
        }
    }
}

/// A block's state as it applies to the track it shows now: what was set
/// for that track, else nothing yet.
pub fn block(state: &State, which: Block) -> BlockState {
    let stored = state.transition.get(which);
    let path = path(state, which);
    if stored.path.as_ref() == path {
        stored.clone()
    } else {
        BlockState {
            path: path.cloned(),
            ..BlockState::default()
        }
    }
}

/// A block's state to change, started afresh if it was for another track.
pub(super) fn block_mut(state: &mut State, which: Block) -> &mut BlockState {
    let path = path(state, which).cloned();
    let stored = state.transition.get_mut(which);
    if stored.path != path {
        *stored = BlockState {
            path,
            ..BlockState::default()
        };
    }
    stored
}

/// The tracks whose lyrics the Transition tab wants while it shows (their
/// words go on the voice's lane): its blocks', unless that is turned off
/// (`transition_lyrics`). `None` while it does not show.
pub fn lyrics_wanted(state: &State) -> Option<Vec<PathBuf>> {
    if !state.showing(Side::Transition) {
        return None;
    }
    if !state.config.transition_lyrics {
        return Some(Vec::new());
    }
    let mut paths: Vec<PathBuf> = Vec::new();
    for which in [Block::Prev, Block::Now, Block::Next] {
        if let Some(path) = path(state, which).filter(|p| !paths.contains(p)) {
            paths.push(path.clone());
        }
    }
    Some(paths)
}

/// A grid fix on the focused block's track: the analysis in the state
/// takes it at once, and the fixed grid is saved.
fn fix(state: &mut State, fx: &mut Vec<Effect>, heard: f64, how: impl Fn(&Grid, usize) -> Grid) {
    let which = state.transition.focus;
    let (Some(path), Some(r)) = (path(state, which).cloned(), rhythmed(state, which).cloned())
    else {
        state.info("No beats to fix here yet");
        return;
    };
    let Some(at) = viewport::cursor(state, which, heard) else {
        return;
    };
    let grid = how(&Grid::of(&r.analysis), at);
    regrid(state, which, &path, &r, Arc::new(grid.apply(&r.analysis)));
    fx.push(Effect::SaveGrid(path, grid));
}

/// Back to the grid the analysis found.
fn reset(state: &mut State, fx: &mut Vec<Effect>) {
    let which = state.transition.focus;
    let (Some(path), Some(r)) = (path(state, which).cloned(), rhythmed(state, which).cloned())
    else {
        return;
    };
    if !r.fixed() {
        state.info("The grid is as the analysis found it");
        return;
    }
    regrid(state, which, &path, &r, Arc::clone(&r.found));
    fx.push(Effect::ResetGrid(path));
    state.info("The grid is back as the analysis found it");
}

/// A block's track (at `path`, analysed as `r`) takes `analysis` as its
/// grid, the grid the analysis found kept: its cursor and cue go to the
/// beats of the new grid nearest their times (half and double tempo number
/// the beats afresh), and its selection is cleared.
fn regrid(state: &mut State, which: Block, path: &Path, r: &Rhythmed, analysis: Arc<Analysis>) {
    let (old, new) = (&r.analysis.beats, &analysis.beats);
    let same = |beat: Option<usize>| beat.and_then(|b| old.get(b)).map(|&t| nearest(new, t));
    state.rhythm.insert(
        path.to_path_buf(),
        Rhythm::Ready(Arc::new(Rhythmed {
            analysis: Arc::clone(&analysis),
            found: Arc::clone(&r.found),
            waveform: r.waveform.clone(),
        })),
    );
    let b = block_mut(state, which);
    b.cursor = same(b.cursor);
    b.cue = b
        .cue
        .and_then(|c| same(Some(c.beat)).map(|beat| Cue { beat, ..c }));
    b.anchor = None;
}

/// The index of the beat of `beats` nearest `time` (0 without beats).
fn nearest(beats: &[f64], time: f64) -> usize {
    let i = beats.partition_point(|&b| b < time);
    match (i.checked_sub(1), beats.get(i)) {
        (Some(before), Some(&after)) if time - beats[before] <= after - time => before,
        (Some(before), None) => before,
        _ => i.min(beats.len().saturating_sub(1)),
    }
}

/// Carries out `op` on the Transition tab; `false` when it is not one of the
/// tab's.
pub(crate) fn apply(state: &mut State, op: Op, fx: &mut Vec<Effect>) -> bool {
    let heard = state.heard();
    if let Some(at) = state.transition.draft.edit.as_mut() {
        match op {
            Op::Up => {
                at.lane = at.lane.saturating_sub(1);
                return true;
            }
            Op::Down => {
                at.lane = (at.lane + 1).min(2 * STEMS - 1);
                return true;
            }
            Op::Back => {
                state.transition.draft.edit = None;
                return true;
            }
            _ => {}
        }
    }
    match op {
        Op::Up => state.transition.focus = Block::Now,
        Op::Down => state.transition.focus = Block::Next,
        Op::TransitionBarBack => viewport::step(state, -1, 0, false, heard),
        Op::TransitionBarForward => viewport::step(state, 1, 0, false, heard),
        Op::TransitionBeatBack => viewport::step(state, 0, -1, false, heard),
        Op::TransitionBeatForward => viewport::step(state, 0, 1, false, heard),
        Op::TransitionSelectBack => viewport::step(state, -1, 0, true, heard),
        Op::TransitionSelectForward => viewport::step(state, 1, 0, true, heard),
        Op::PageUp => viewport::step(state, -PAGE_BARS, 0, false, heard),
        Op::PageDown => viewport::step(state, PAGE_BARS, 0, false, heard),
        Op::TransitionZoomIn => viewport::zoom(state, -1, heard),
        Op::TransitionZoomOut => viewport::zoom(state, 1, heard),
        Op::TransitionFitLyrics => viewport::fit_lyrics(state, heard),
        Op::TransitionLyrics => {
            state.config.transition_lyrics = !state.config.transition_lyrics;
            fx.push(Effect::SaveConfig);
            let now = if state.config.transition_lyrics {
                "on"
            } else {
                "off"
            };
            state.info(format!("Lyrics on the vocals lane: {now}"));
        }
        Op::TransitionFollow => {
            let b = block_mut(state, Block::Now);
            b.start = None;
            b.cursor = None;
            b.anchor = None;
            state.transition.focus = Block::Now;
        }
        // Enter never moves playback: it sets the focused track's cue.
        Op::Accept => draft::set_cue(state, fx, heard),
        // Esc stays on the tab (tab and its buttons leave it): it clears
        // the selection, else calls off a transition armed, else clears the
        // focused track's cue.
        Op::Back => {
            let which = state.transition.focus;
            let b = block_mut(state, which);
            if b.anchor.is_some() {
                b.anchor = None;
            } else {
                match state.transition.draft.status {
                    Status::Armed => draft::call_off(state, fx),
                    Status::Mixing(_) => {
                        state.info("A transition is under way");
                    }
                    Status::Draft => {
                        let b = block_mut(state, which);
                        if b.cue.take().is_some() {
                            state.transition.draft.edit = None;
                            state.info("Cue cleared");
                        }
                    }
                }
            }
        }
        Op::GridEarlier => fix(state, fx, heard, |g, _| g.shifted(-1)),
        Op::GridLater => fix(state, fx, heard, |g, _| g.shifted(1)),
        Op::GridNudgeBack => fix(state, fx, heard, |g, _| g.nudged(-NUDGE)),
        Op::GridNudgeForward => fix(state, fx, heard, |g, _| g.nudged(NUDGE)),
        Op::GridHalf => fix(state, fx, heard, |g, _| g.halved()),
        Op::GridDouble => fix(state, fx, heard, |g, _| g.doubled()),
        Op::GridOne => fix(state, fx, heard, |g, at| g.with_one_at(at)),
        Op::GridReset => reset(state, fx),
        Op::TransitionGo => draft::go(state, fx),
        // During the mix the plan heard stays as it is.
        Op::TransitionPreset
        | Op::TransitionTempo
        | Op::TransitionPairing
        | Op::TransitionEdit
        | Op::EditBack
        | Op::EditForward
        | Op::EditRaise
        | Op::EditLower
        | Op::EditOff
        | Op::EditFull
        | Op::EditCut
            if matches!(state.transition.draft.status, Status::Mixing(_)) =>
        {
            state.info("A transition is under way: it plays as it is");
        }
        Op::TransitionPreset => {
            let d = &mut state.transition.draft;
            d.preset = d.preset.next();
            d.lanes = None;
            let name = d.preset.name();
            // Remembered: the next run starts from it.
            state.config.transition_preset = state.transition.draft.preset;
            fx.push(Effect::SaveConfig);
            state.info(format!("Transition: {name}"));
            draft::rearm(state, fx, true);
        }
        Op::TransitionTempo => {
            let d = &mut state.transition.draft;
            d.tempo = match d.tempo {
                TempoMode::Varispeed => TempoMode::Stretch,
                TempoMode::Stretch => TempoMode::Varispeed,
            };
            let what = tr::tempo_name(d.tempo);
            state.config.transition_tempo = state.transition.draft.tempo;
            fx.push(Effect::SaveConfig);
            state.info(format!("Transition: {what}"));
            draft::rearm(state, fx, false);
        }
        Op::TransitionPairing => {
            let d = &mut state.transition.draft;
            d.pairing = match d.pairing {
                None => Some(tr::Pairing::One),
                Some(tr::Pairing::One) => Some(tr::Pairing::Two),
                Some(tr::Pairing::Two) => Some(tr::Pairing::Half),
                Some(tr::Pairing::Half) => None,
            };
            let label = draft::pairing(state).label();
            let what = match state.transition.draft.pairing {
                None => format!("{label} (by the tempi)"),
                Some(_) => label.to_owned(),
            };
            state.info(format!("Transition: beats paired {what}"));
            draft::repair(state, fx);
        }
        Op::TransitionEdit => {
            let cued = Block::BOTH.iter().any(|&b| block(state, b).cue.is_some());
            if !cued && state.transition.draft.status == Status::Draft {
                let enter = keymap::hint(Context::Transition, Op::Accept);
                state.info(format!("Set a cue first: {enter} on the bars to mix over"));
                return true;
            }
            let bars = draft::draft_bars(state);
            let d = &mut state.transition.draft;
            if d.edit.take().is_none() {
                d.lanes = Some(d.lanes(bars));
                d.edit = Some(Edit::default());
            }
        }
        Op::EditBack
        | Op::EditForward
        | Op::EditRaise
        | Op::EditLower
        | Op::EditOff
        | Op::EditFull
        | Op::EditCut => edit(state, op, fx),
        _ => return false,
    }
    true
}

/// An editing key on the lanes.
fn edit(state: &mut State, op: Op, fx: &mut Vec<Effect>) {
    let bars = draft::draft_bars(state);
    let d = &mut state.transition.draft;
    let Some(mut at) = d.edit else {
        return;
    };
    let lanes = d.lanes.get_or_insert_with(|| d.preset.lanes(bars));
    let share = at.bar as f64 / bars as f64;
    let now = lanes[at.lane].gain(share);
    let cut = tr::point_at(lanes, at.lane, at.bar, bars).is_some_and(|p| p.cut);
    match op {
        Op::EditBack => at.bar = at.bar.saturating_sub(1),
        Op::EditForward => at.bar = (at.bar + 1).min(bars),
        Op::EditRaise => tr::set_point(lanes, at.lane, at.bar, bars, now + tr::GAIN_STEP, cut),
        Op::EditLower => tr::set_point(lanes, at.lane, at.bar, bars, now - tr::GAIN_STEP, cut),
        Op::EditOff => tr::set_point(lanes, at.lane, at.bar, bars, 0.0, cut),
        Op::EditFull => tr::set_point(lanes, at.lane, at.bar, bars, 1.0, cut),
        Op::EditCut => tr::set_point(lanes, at.lane, at.bar, bars, now, !cut),
        _ => {}
    }
    d.edit = Some(at);
    draft::rearm(state, fx, true);
}
