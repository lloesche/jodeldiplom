//! What a block of the Transition tab shows, and how it moves: the span of
//! time in view at each zoom, the playhead, the cursor and the selection,
//! and the keys, clicks, drags and the wheel that move them.
//!
//! Both blocks show the same span of time, so their tempi compare at a
//! glance: [`ZOOMS`] bars of the playing track, or the whole of the longer
//! track. The playing track's block follows what is heard (a third from
//! its left) until its cursor moves; the next one's starts at its first
//! bar. The cursor is a beat; it moves by bars (or beats), and the
//! selection runs from where it was when a selecting move began to where
//! it is, in whole bars.

use jodel_playback::Command;
use jodel_rhythm::Analysis;

use super::draft::{pairing, transition_into};
use super::tab::{block, block_mut, Pan};
use crate::core::lanes::{self, rhythmed, Block};
use crate::core::transition::Status;
use crate::core::{Effect, State};

/// The spans the view shows, in bars of the playing track; 0 is the whole
/// of the longer track.
pub const ZOOMS: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 0];
/// The zoom at the start (16 bars).
pub const DEFAULT_ZOOM: usize = 4;
/// A bar's length where the playing track's is not known, in seconds.
const FALLBACK_BAR: f64 = 2.0;
/// The share of the span shown before the playhead while following, and
/// before the cursor when it is brought into view.
pub const LEAD: f64 = 1.0 / 3.0;

/// How long the next track's block takes to slide up into the playing
/// track's after a transition hands over, in milliseconds.
pub const SLIDE_MS: u64 = 700;

/// The milliseconds of the frame being drawn (the clock between ticks).
pub fn frame_ms(state: &State) -> u64 {
    state.drawn_ms.unwrap_or(state.now_ms)
}

/// How far the slide after a hand-over has come, 0 to 1 (eased), while
/// it is under way.
pub fn slide(state: &State) -> Option<f64> {
    let since = state.transition.slide?;
    let p = frame_ms(state).saturating_sub(since) as f64 / SLIDE_MS as f64;
    (p < 1.0).then(|| 1.0 - (1.0 - p).powi(3))
}

/// The clock: a slide that has ended is forgotten.
pub(crate) fn tick(state: &mut State) {
    if state.transition.slide.is_some() && slide(state).is_none() {
        state.transition.slide = None;
    }
}

/// What is heard of a block's track, with `heard` the position heard in
/// the playing track: that; the next track's while a transition mixes it
/// in.
pub fn playhead(state: &State, which: Block, heard: f64) -> Option<f64> {
    state.playing.track.as_ref()?;
    match which {
        Block::Now => Some(heard),
        Block::Next => {
            let sent = transition_into(state)?;
            matches!(state.transition.draft.status, Status::Mixing(_)).then(|| sent.map.at(heard))
        }
        Block::Prev => None,
    }
}

/// The median length of the bars of `a`, in seconds.
fn bar_seconds(a: &Analysis) -> Option<f64> {
    let mut lens: Vec<f64> = a
        .downbeats
        .windows(2)
        .map(|w| a.beats[w[1]] - a.beats[w[0]])
        .collect();
    if lens.is_empty() {
        return None;
    }
    lens.sort_by(f64::total_cmp);
    Some(lens[lens.len() / 2])
}

/// The length of a block's track, as far as known.
fn duration(state: &State, which: Block) -> f64 {
    rhythmed(state, which).map_or(0.0, |r| {
        r.analysis
            .duration
            .max(r.waveform.as_ref().map_or(0.0, |w| w.duration()))
    })
}

/// A block's zoom: its own (the wheel's), else the tab's.
pub fn zoom_of(state: &State, which: Block) -> usize {
    block(state, which)
        .zoom
        .unwrap_or(state.transition.zoom)
        .min(ZOOMS.len() - 1)
}

/// Whether a block shows its whole track.
pub fn whole(state: &State, which: Block) -> bool {
    ZOOMS[zoom_of(state, which)] == 0
}

/// The seconds a block shows.
pub fn span(state: &State, which: Block) -> f64 {
    span_at(state, zoom_of(state, which))
}

/// The seconds zoom `zoom` (an index into [`ZOOMS`]) shows.
fn span_at(state: &State, zoom: usize) -> f64 {
    let bars = ZOOMS[zoom.min(ZOOMS.len() - 1)];
    if bars == 0 {
        return duration(state, Block::Now)
            .max(duration(state, Block::Next))
            .max(1.0);
    }
    let bar = Block::BOTH
        .iter()
        .find_map(|&b| rhythmed(state, b).and_then(|r| bar_seconds(&r.analysis)))
        .unwrap_or(FALLBACK_BAR);
    bars as f64 * bar
}

/// The time of a block's first bar, 0 without one.
fn first_bar(a: &Analysis) -> f64 {
    a.downbeats
        .first()
        .and_then(|&d| a.beats.get(d))
        .copied()
        .unwrap_or(0.0)
}

/// The seconds a block shows, `(start, end)`, with `heard` the position
/// heard in the playing track.
pub fn window(state: &State, which: Block, heard: f64) -> (f64, f64) {
    let span = span(state, which);
    if whole(state, which) {
        return (0.0, span);
    }
    let b = block(state, which);
    let into = transition_into(state).map(|s| s.map.at(heard));
    let start = match (b.start, which, into) {
        (Some(start), _, _) => start,
        (None, Block::Now, _) => heard - LEAD * span,
        // Mixed in (or about to be): it follows its own playhead, which
        // stays under the playing track's.
        (None, Block::Next, Some(at)) => at - LEAD * span,
        (None, Block::Next | Block::Prev, _) => {
            rhythmed(state, which).map_or(0.0, |r| first_bar(&r.analysis) - 0.05 * span)
        }
    };
    let start = start.max(0.0);
    (start, start + span)
}

/// The bar a beat is in, counted from 1 (0 before the first bar).
pub fn bar_of(a: &Analysis, beat: usize) -> usize {
    a.downbeats.partition_point(|&d| d <= beat)
}

/// The beat the cursor is on: where it was put, else the playing track's
/// bar under the playhead, or the next track's first bar.
pub fn cursor(state: &State, which: Block, heard: f64) -> Option<usize> {
    let a = &rhythmed(state, which)?.analysis;
    if a.beats.is_empty() {
        return None;
    }
    let b = block(state, which);
    let beat = match (b.cursor, which) {
        (Some(c), _) => c,
        (None, Block::Now) => {
            let at = a.beat_at(heard).unwrap_or(0);
            let bar = bar_of(a, at);
            bar.checked_sub(1).map_or(at, |k| a.downbeats[k])
        }
        (None, Block::Next) => a.downbeats.first().copied().unwrap_or(0),
        (None, Block::Prev) => return None,
    };
    Some(beat.min(a.beats.len() - 1))
}

/// The seconds a block's selection covers, in whole bars, if it has one.
pub fn selection(state: &State, which: Block, heard: f64) -> Option<(f64, f64)> {
    let a = &rhythmed(state, which)?.analysis;
    let b = block(state, which);
    let (anchor, cursor) = (b.anchor?, cursor(state, which, heard)?);
    let (lo, hi) = (anchor.min(cursor), anchor.max(cursor));
    let start = match bar_of(a, lo).checked_sub(1) {
        Some(k) => a.beats[a.downbeats[k]],
        None => *a.beats.first()?,
    };
    let end = a
        .downbeats
        .get(bar_of(a, hi))
        .and_then(|&d| a.beats.get(d))
        .copied()
        .unwrap_or(a.duration);
    Some((start, end))
}

/// The beat `bars` bars from `from` (on a bar's first beat: whole bars;
/// within a bar: to the bar's first beat first, going back). Without bars,
/// four beats a bar.
fn by_bars(a: &Analysis, from: usize, bars: isize) -> usize {
    let n = a.beats.len();
    if a.downbeats.is_empty() {
        return (from as isize + 4 * bars).clamp(0, n as isize - 1) as usize;
    }
    let started = a.downbeats.partition_point(|&d| d <= from) as isize;
    let on_bar = started > 0 && a.downbeats[started as usize - 1] == from;
    // Back from within a bar, its own first beat is the first step.
    let target = started - 1 + bars + isize::from(bars < 0 && !on_bar);
    let k = target.clamp(0, a.downbeats.len() as isize - 1) as usize;
    a.downbeats[k]
}

/// Puts a block's cursor on `beat` (selecting from where it was when
/// `select`), and brings it into view.
fn put_cursor(state: &mut State, which: Block, beat: usize, select: bool, heard: f64) {
    let old = cursor(state, which, heard);
    let span = span(state, which);
    let (start, end) = window(state, which, heard);
    let Some(r) = rhythmed(state, which).cloned() else {
        return;
    };
    let time = r.analysis.beats.get(beat).copied().unwrap_or(0.0);
    let whole = whole(state, which);
    let following = which == Block::Now || transition_into(state).is_some();
    let b = block_mut(state, which);
    if select {
        if b.anchor.is_none() {
            b.anchor = old;
        }
    } else {
        b.anchor = None;
    }
    b.cursor = Some(beat);
    if !whole && (time < start || time >= end) {
        b.start = Some((time - LEAD * span).max(0.0));
    } else if b.start.is_none() && following {
        // The view stops following where it is.
        b.start = Some(start);
    }
}

/// Moves a block's cursor by bars or beats.
pub(super) fn step(state: &mut State, bars: isize, beats: isize, select: bool, heard: f64) {
    let which = state.transition.focus;
    let Some(r) = rhythmed(state, which).cloned() else {
        return;
    };
    let a = &r.analysis;
    let Some(from) = cursor(state, which, heard) else {
        return;
    };
    // By whole bars of the slower track where the beats pair two to one.
    let per = pairing(state);
    let every = match which {
        Block::Now => per.bars_per_step(),
        _ => per.next_bars_per_step(),
    } as isize;
    let to = if bars != 0 {
        by_bars(a, from, bars * every)
    } else {
        (from as isize + beats).clamp(0, a.beats.len() as isize - 1) as usize
    };
    put_cursor(state, which, to, select, heard);
}

/// Zooms the tab by `by` levels ([`set_zoom`]).
pub(super) fn zoom(state: &mut State, by: isize, heard: f64) {
    let z = (state.transition.zoom as isize + by).clamp(0, ZOOMS.len() as isize - 1) as usize;
    set_zoom(state, z, heard);
}

/// Sets the tab's zoom (both blocks, their own zooms dropped), keeping the
/// middle of each block's view where it is.
fn set_zoom(state: &mut State, z: usize, heard: f64) {
    let old: Vec<(f64, f64)> = Block::BOTH
        .iter()
        .map(|&b| window(state, b, heard))
        .collect();
    state.transition.zoom = z;
    for (i, which) in Block::BOTH.into_iter().enumerate() {
        block_mut(state, which).zoom = None;
        let span = span(state, which);
        let mid = 0.5 * (old[i].0 + old[i].1);
        let b = block_mut(state, which);
        if b.start.is_some() {
            b.start = Some((mid - 0.5 * span).max(0.0));
        }
    }
}

/// The widest zoom at which the focused track's lanes leave out no word of
/// its lyrics (no more than they must at any zoom): set as that track's own
/// zoom (the other keeps its own), and said. A view following the playhead
/// goes on following it; one moved by hand keeps its middle.
pub(super) fn fit_lyrics(state: &mut State, heard: f64) {
    let which = state.transition.focus;
    let width = state.fit.lane_letters;
    if width == 0 {
        return;
    }
    let lyrics = lanes::lyrics(state, which);
    let words = lyrics
        .as_deref()
        .map(lanes::fitted_words)
        .unwrap_or_default();
    if words.is_empty() {
        state.info("No timed lyrics to fit the zoom to");
        return;
    }
    let least = lanes::words_left_out(&words, lanes::LYRICS_MOST_PER, 1);
    let fits = |z: usize| {
        let per = f64::from(width) / span_at(state, z).max(1e-3);
        lanes::words_left_out(&words, per, 1) <= least
    };
    let fitting = (0..ZOOMS.len()).rev().find(|&z| fits(z));
    let z = fitting.unwrap_or(0);
    let from = zoom_of(state, which);
    // A view following the playhead (or, the next track's, at its first
    // bar) goes on doing so; one moved by hand stays around its middle.
    let following = block(state, which).start.is_none();
    zoom_block(state, which, z as isize - from as isize, None, heard);
    if following {
        block_mut(state, which).start = None;
    }
    let shown = match ZOOMS[z] {
        0 => "the whole track".to_owned(),
        1 => "1 bar".to_owned(),
        bars => format!("{bars} bars"),
    };
    let track = match which {
        Block::Next => "the next track",
        _ => "the playing track",
    };
    let said = if fitting.is_some() {
        format!("Zoomed {track} to fit its lyrics: {shown}")
    } else {
        format!("Zoomed {track} in all the way ({shown}); some words still do not fit")
    };
    state.info(said);
}

/// Zooms one block (the wheel over it), keeping `at` (seconds into its
/// track, under the pointer) where it is on screen; without `at`, its
/// middle.
fn zoom_block(state: &mut State, which: Block, by: isize, at: Option<f64>, heard: f64) {
    let (start, end) = window(state, which, heard);
    let from = zoom_of(state, which);
    let z = (from as isize + by).clamp(0, ZOOMS.len() as isize - 1) as usize;
    if z == from {
        return;
    }
    let at = at.unwrap_or(0.5 * (start + end)).clamp(start, end);
    // Where the time is across the view, 0 to 1.
    let share = if end > start {
        (at - start) / (end - start)
    } else {
        0.5
    };
    block_mut(state, which).zoom = Some(z);
    let span = span(state, which);
    if !whole(state, which) {
        block_mut(state, which).start = Some((at - share * span).max(0.0));
    }
}

/// A point of a block's lanes was pressed (`extend`: dragged to, which
/// selects): the cursor goes to the beat at or before `time`. Whether it
/// was a double click (a second press on that beat within
/// [`crate::core::DOUBLE_CLICK_MS`]).
pub(crate) fn point(state: &mut State, which: Block, time: f64, extend: bool) -> bool {
    state.transition.focus = which;
    let heard = state.heard();
    let Some(r) = rhythmed(state, which).cloned() else {
        return false;
    };
    let a = &r.analysis;
    if a.beats.is_empty() {
        return false;
    }
    let beat = a.beat_at(time).unwrap_or(0);
    // A drag is no press: the one after it is a first.
    let presses = &mut state.transition.last_press;
    let double = if extend {
        presses.forget();
        false
    } else {
        presses.press((which, beat), state.now_ms)
    };
    if extend {
        let b = block_mut(state, which);
        if b.anchor.is_none() {
            b.anchor = b.cursor;
        }
    }
    put_cursor(state, which, beat, extend, heard);
    double
}

/// The playing track's overview was clicked at `time`: playback goes there
/// (unless Karaoke mode holds it), and the view follows the playhead.
/// With nothing to seek, the view goes there.
pub(crate) fn seek(state: &mut State, fx: &mut Vec<Effect>, time: f64) {
    let Some(duration) = state.playing.duration() else {
        return jump(state, Block::Now, time);
    };
    if crate::core::karaoke::seek_refused(state) {
        return jump(state, Block::Now, time);
    }
    let seconds = time.clamp(0.0, duration);
    state.playing.position = seconds;
    fx.push(Effect::Engine(Command::Seek(seconds)));
    block_mut(state, Block::Now).start = None;
}

/// The next track's overview was clicked at `time`: its view goes there.
pub(crate) fn jump(state: &mut State, which: Block, time: f64) {
    if whole(state, which) {
        return;
    }
    let span = span(state, which);
    block_mut(state, which).start = Some((time - 0.5 * span).max(0.0));
}

/// The view scrolls by a share of the span (the wheel with shift).
pub(crate) fn scroll(state: &mut State, which: Block, share: f64) {
    if whole(state, which) {
        return;
    }
    let heard = state.heard();
    let (start, _) = window(state, which, heard);
    let span = span(state, which);
    block_mut(state, which).start = Some((start + share * span).max(0.0));
}

/// The secondary button went down over a block (`overview`: over its
/// whole track) at `at` across, of an area `width` wide (in the window's
/// units): a drag of its view starts.
pub(crate) fn pan_start(state: &mut State, which: Block, at: f32, width: f32, overview: bool) {
    if whole(state, which) || width <= 0.0 || which == Block::Prev {
        return;
    }
    let heard = state.heard();
    let (start, end) = window(state, which, heard);
    let per_unit = if overview {
        -duration(state, which) / f64::from(width)
    } else {
        (end - start) / f64::from(width)
    };
    state.transition.focus = which;
    state.transition.pan = Some(Pan {
        block: which,
        per_unit,
        at,
        start,
    });
}

/// The pointer is at `at` across with a drag of a view under way: the
/// view moves with it (dragged right, the waveform goes right, to earlier
/// times; on the whole track, the span shown follows the pointer).
pub(crate) fn pan_to(state: &mut State, at: f32) {
    let Some(pan) = state.transition.pan else {
        return;
    };
    let moved = f64::from(at) - f64::from(pan.at);
    let start = (pan.start - moved * pan.per_unit).max(0.0);
    block_mut(state, pan.block).start = Some(start);
}

/// The drag of a view ends.
pub(crate) fn pan_end(state: &mut State) {
    state.transition.pan = None;
}

/// The wheel without shift over a block: that block zooms in (`by` < 0)
/// or out, around `at` (seconds into its track, under the pointer).
pub(crate) fn wheel_zoom(state: &mut State, which: Block, by: isize, at: Option<f64>) {
    let heard = state.heard();
    zoom_block(state, which, by, at, heard);
}
