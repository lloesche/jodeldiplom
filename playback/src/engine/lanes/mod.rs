//! The playing track's stems as parts of its output ([`jodel_output::lanes`]):
//! drums, bass, vocals and other beside main, on the frames main is heard
//! on, and a control word that says whether they are there.
//!
//! [`StemLanes`] keeps a stem run for the playing track (and one for the
//! next, so a gapless change finds its stems ready) and makes the parts a
//! little ahead of what is written: the run's stems resampled to the
//! output's rate and laid out in the track's channels ([`stream`]), started
//! on an output frame a run frame falls on exactly, so they add nothing of
//! their own: they add up to main as closely as the separator's stems add
//! up to the song.
//! What is made and not yet written waits in a [`buffer`], with what the
//! output may still give back behind it.
//!
//! The control word ([`control`]) follows what is there. On a device the
//! parts are never waited for: before a write only what it needs is made
//! (a seam's length past it, nothing while they are absent), the rest after
//! it, a few pieces at most; while its ring is full they are made on, and
//! once none are left to make the write waits for room, as on a plain
//! output. They are said to be there once [`AHEAD`] of
//! them is made ahead (and no sooner again once they ran out), and they end
//! over a mixer's switch's length ([`ending_frames`]) while still valid, so
//! a mixer hearing them fades back to main. Offline the writing waits for
//! them. At a track's end they run to its last frame when the next track's
//! are ready to follow (or nothing follows), else they end before it. They
//! end before a route fades out for another format, and are there again
//! only after the new route's rise. Parts that start while the output holds
//! frames written without them start on the frame heard, and those frames
//! are written again with them once they are made ([`catch_up`]).
//!
//! A new segment ([`SegStart::Cold`]: a new track, a seek) has its parts
//! start from silence on the segment's first frame, as main's resampler
//! starts there: from the stems kept when their grid has that frame, else
//! from a new run from the segment's track frame. A segment joined to what
//! played ([`SegStart::Joined`]: a jump, a route met at another gain) has
//! them start from the stems kept, a resampler's run-up before it, their
//! first frames crossfaded from what would have played next over the
//! frames main's are.
//!
//! A head read along a time map ([`along`]) has parts whatever was asked:
//! its stems read along the map are its parts, and its main is their sum
//! (with no separator, the whole mix as the other instruments' part).
//!
//! The parts' requantization is not limited and its clips are not reported
//! (the desk shows them). The next track's run starts here once this one's
//! file is read; a transition's next track, on a head of its own, starts
//! its own ahead of the window.

mod along;
mod buffer;
mod catch_up;
mod control;
mod stream;
#[cfg(test)]
mod tests;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use jodel_output::lanes::{ending_frames, Parts};
use jodel_output::StreamFormat;

use crate::error::Error;
use crate::stem_file::StemStore;
use crate::stems::{SeparatorSource, StemRun, Unkept, RATE as STEM_RATE, STEMS};

use super::api::Event;
use super::track::{frames_of, Progress, SegStart, Track};
use super::{Engine, PREROLL_RESAMPLE, WRITE_WAIT};
use buffer::Buffer;
use control::{Plan, State, Supply, Timeline};
use stream::{run_start, Grid, Line, Made, Stream};

/// When a player makes its tracks' stems as parts of its output
/// ([`crate::Command::MakeStems`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MakeStems {
    /// Never: the output is main alone.
    #[default]
    Never,
    /// From the stems kept ([`crate::StemStore`]) only: a track whose
    /// stems are not kept has none, and nothing is separated.
    Kept,
    /// From the stems kept, else separated as the track plays.
    Separate,
}

/// Seconds of parts made ahead before a device is told they are there.
const AHEAD: f64 = 0.5;

/// Frames of an output of lanes written at a time.
const LANE_CHUNK: usize = 4096;

/// The most pieces of stems made around a write on a device (behind after
/// a stall, they catch up over a few writes).
const PIECES: usize = 4;

/// The track after the one written, for the parts at its end.
#[derive(Clone, Copy, Debug)]
pub(super) enum Next<'a> {
    /// None follows (the queue ends, or playback holds at the end).
    Nothing,
    /// This file is run into.
    Track(&'a PathBuf),
    /// One takes over that has no parts here (a transition's, on a head of
    /// its own).
    Unable,
}

/// What the writer is about to write, for [`StemLanes::before`].
pub(super) struct Want<'a> {
    pub(super) track: &'a Track,
    /// The output frame written next, and the frames of main to go.
    pub(super) from: u64,
    pub(super) frames: usize,
    /// Output frames kept behind what is written (the output may give
    /// them back).
    pub(super) back: u64,
    /// What a run keeps behind the frame it hands on next, and takes ahead
    /// of it (at the separator's rate).
    pub(super) keep: (u64, u64),
    pub(super) next: Next<'a>,
    /// The track may have parts now: its plain route, or along a map.
    pub(super) able: bool,
    /// Frames of parts made past what is written at the least: a mixer's
    /// switch's length, or the longest crossfade main makes at a seam, so
    /// a seam finds what would have played next.
    pub(super) margin: u64,
    /// The output frame after main's last, once all of it is made.
    pub(super) main_end: Option<u64>,
    /// No parts before this output frame (a route's rise after a reopen),
    /// nor from this one on (its fade out for another format).
    pub(super) rise: u64,
    pub(super) cut: Option<u64>,
    /// The first output frame the track may write again once its parts are
    /// made ([`Engine::rewrite_floor`]; `from` when none may be).
    pub(super) floor: u64,
}

/// What [`StemLanes::before`] found.
pub(super) struct Followed {
    /// The parts there, for the control word.
    supply: Supply,
    /// Whether the parts promised are owed on this write (not on a track's
    /// first frame, where main itself cuts).
    owed: bool,
    /// Offline, the frames of main that may be written now (the rest
    /// waits for its parts).
    limit: Option<u64>,
    /// The stems of the track just failed, and why.
    failed: Option<String>,
}

/// What [`StemLanes::make_to`] did: whether it made parts, and whether it
/// stopped for stems not there yet.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Making {
    pub(super) made: bool,
    pub(super) waiting: bool,
}

/// Why a run has no more stems for its track.
enum Stopped {
    Failed(String),
    /// Not kept, and none are separated: quietly none.
    Missing,
}

/// How the next track's parts stand, at the end of this one's.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Following {
    /// Nothing follows, or they are ready.
    Fine,
    /// They are being made (offline, waited for).
    Coming,
    /// There will be none.
    Not,
}

/// The playing track's stems as parts of the output: see the module.
pub(super) struct StemLanes {
    make: MakeStems,
    separator: Option<SeparatorSource>,
    store: Option<Arc<dyn StemStore>>,
    offline: bool,
    /// The run of the track the parts are made of, of the next one, and
    /// one started for a move onto a map ([`StemLanes::prepare_along`]).
    run: Option<StemRun>,
    prestart: Option<StemRun>,
    ahead: Option<StemRun>,
    /// A track (by index and file) whose stems failed or are not kept: it
    /// has none until they are asked for again.
    off: Option<(usize, PathBuf)>,
    stream: Option<Stream>,
    buffer: Buffer,
    timeline: Timeline,
    /// The output's rate.
    rate: u32,
    plan: Plan,
    wide: Vec<i32>,
}

impl StemLanes {
    pub(super) fn new(
        separator: Option<SeparatorSource>,
        store: Option<Arc<dyn StemStore>>,
        offline: bool,
    ) -> StemLanes {
        StemLanes {
            make: MakeStems::Never,
            separator,
            store,
            offline,
            run: None,
            prestart: None,
            ahead: None,
            off: None,
            stream: None,
            buffer: Buffer::new(STEMS * 2),
            timeline: Timeline::new(0),
            rate: 1,
            plan: Plan::default(),
            wide: Vec::new(),
        }
    }

    /// Another, for another head of the same player: the same separator,
    /// store and choice, nothing made yet.
    pub(super) fn sibling(&self) -> StemLanes {
        let mut lanes = StemLanes::new(self.separator, self.store.clone(), self.offline);
        lanes.make = self.make;
        lanes
    }

    /// [`crate::Command::MakeStems`]: stems that failed are tried again,
    /// and the runs start again for the new choice (kept only: nothing
    /// separates any more).
    pub(super) fn set_make(&mut self, make: MakeStems) {
        if make != self.make {
            self.run = None;
            self.prestart = None;
            self.ahead = None;
            self.stream = None;
        }
        self.make = make;
        self.off = None;
    }

    /// An output at `format` opened: its frames count from 0.
    pub(super) fn open(&mut self, format: StreamFormat) {
        self.rate = format.rate.max(1);
        self.buffer = Buffer::new(STEMS * usize::from(format.channels));
        self.timeline = Timeline::new(ending_frames(self.rate));
        self.stream = None;
    }

    /// The output dropped what it had not played and goes on from output
    /// frame `at` (a seek, a track installed, an output closed): nothing
    /// before binds what follows.
    pub(super) fn flushed(&mut self, at: u64) {
        self.stream = None;
        self.buffer.clear(at);
        self.timeline.clear(at);
    }

    /// Playback stopped: the runs go too.
    pub(super) fn stop(&mut self) {
        self.flushed(0);
        self.run = None;
        self.prestart = None;
        self.ahead = None;
        self.off = None;
    }

    /// No more parts are made: what is there stays only as far as it is
    /// promised ([`Timeline::promised`]), so they end from then.
    fn end_at(&mut self, from: u64) {
        self.stream = None;
        let promised = self.timeline.promised(from);
        self.buffer.restart(from, promised, 0);
    }

    /// Whether the stems of `track` failed or are not kept.
    fn is_off(&self, track: &Track) -> bool {
        self.off
            .as_ref()
            .is_some_and(|(i, p)| *i == track.index && *p == track.info.path)
    }

    /// The track has no parts any more: none until asked for again (with
    /// `why`, when its stems failed).
    fn give_up(&mut self, track: &Track, from: u64) {
        self.off = Some((track.index, track.info.path.clone()));
        self.run = None;
        self.end_at(from);
    }

    /// Where a run gets the stems the store does not keep, as the stems
    /// are asked for. `None`: no runs.
    fn unkept(&self) -> Option<Unkept> {
        match self.make {
            MakeStems::Never => None,
            MakeStems::Kept => Some(Unkept::Missing),
            MakeStems::Separate => Some(self.separator.into()),
        }
    }

    /// Where a run for `want` gets the stems not kept: as they are asked
    /// for (`None`: no runs, also while the track may have no parts or
    /// they failed for it). A head read along a map has them whatever was
    /// asked, as its main is their sum: separated when they are asked for
    /// and there is a separator, else (and once they failed) the whole mix.
    fn unkept_for(&self, want: &Want) -> Option<Unkept> {
        if want.track.along.is_some() {
            return Some(self.unkept_along(want.track));
        }
        let off = self.is_off(want.track);
        self.unkept().filter(|_| want.able && !off)
    }

    /// Where a run for `track` read along a map gets the stems not kept.
    fn unkept_along(&self, track: &Track) -> Unkept {
        match (self.make, self.separator) {
            (MakeStems::Separate, Some(separator)) if !self.is_off(track) => {
                Unkept::Separate(separator)
            }
            _ => Unkept::Whole,
        }
    }

    /// Before a write: the runs started, kept or stopped as the stems are
    /// asked for, the parts started again where the track's line changed,
    /// and what the write needs of them made (waited for offline, for a
    /// moment).
    pub(super) fn before(&mut self, want: &Want) -> Followed {
        let (track, from) = (want.track, want.from);
        if from < self.timeline.written() {
            self.timeline.given_back(from);
        }
        let behind = from.saturating_sub(want.back);
        self.timeline.trim(behind);
        self.buffer.trim(behind);
        let line = Line::of(track);
        let owed = !(line.begun == SegStart::Cold && from == line.seg_out);
        let quiet = |lanes: &StemLanes, failed| Followed {
            supply: lanes.supply(want, false),
            owed,
            limit: None,
            failed,
        };
        let Some(unkept) = self.unkept_for(want) else {
            if self.stream.is_some() {
                self.end_at(from);
            }
            if self.make == MakeStems::Never {
                self.run = None;
                self.prestart = None;
            }
            return quiet(self, None);
        };
        if !self.stream.as_ref().is_some_and(|s| s.line.same(&line)) {
            self.start(line, want, unkept);
        }
        // What this write needs; on a device nothing while there are none
        // (they could not be said to be there in it anyway).
        let need = from + want.frames as u64 + want.margin;
        let absent = !self.offline && self.timeline.state(from) == State::Absent;
        let target = if absent { 0 } else { need };
        match self.make_to(want, target, self.budget()) {
            Err(Stopped::Failed(why)) => {
                self.give_up(track, from);
                return quiet(self, Some(why));
            }
            Err(Stopped::Missing) => {
                self.give_up(track, from);
                return quiet(self, None);
            }
            Ok(_) => {}
        }
        let following = self.follow_next(want);
        let done = self.stream.as_ref().is_some_and(|s| s.done);
        let there = self.buffer.there(from);
        let limit = if !self.offline {
            None
        } else if !done {
            // Made a seam's length past what is written, so the parts are
            // there with what follows them.
            Some(there.saturating_sub(from + want.margin))
        } else if following == Following::Coming && there >= from + want.frames as u64 {
            Some(0)
        } else {
            None
        };
        Followed {
            supply: self.supply(want, done && following == Following::Fine),
            owed,
            limit,
            failed: None,
        }
    }

    /// After a write that took the output to frame `head`: on a device, the
    /// parts made on toward [`AHEAD`]. Returns whether any were made (more
    /// may be made at once), or why the stems failed, when they just did.
    pub(super) fn after(&mut self, want: &Want, head: u64) -> Result<bool, String> {
        if self.offline || self.stream.is_none() {
            return Ok(false);
        }
        let ahead = (AHEAD * f64::from(self.rate)).round() as u64;
        match self.make_to(want, head + ahead, self.budget()) {
            Err(Stopped::Failed(why)) => {
                self.give_up(want.track, head);
                Err(why)
            }
            Err(Stopped::Missing) => {
                self.give_up(want.track, head);
                Ok(false)
            }
            Ok(making) => Ok(making.made),
        }
    }

    /// Where the parts end, when they are made to the track's end and it
    /// comes before output frame `to`.
    pub(super) fn near_end(&self, to: u64) -> Option<u64> {
        let end = self.buffer.end();
        (self.stream.as_ref().is_some_and(|s| s.done) && end <= to).then_some(end)
    }

    /// The parts there from `want.from` on, for the control word.
    fn supply(&self, want: &Want, through: bool) -> Supply {
        let from = want.from;
        let mut there = self.buffer.there(from);
        if let Some(cut) = want.cut {
            there = there.min(cut.max(from));
        }
        Supply {
            there,
            coming: self.stream.is_some() && from >= want.rise,
            hold: if self.offline {
                0
            } else {
                (AHEAD * f64::from(self.rate)).round() as u64
            },
            through,
            main_end: want.main_end,
        }
    }

    /// Starts the parts on `line` from output frame `want.from`: from
    /// silence on its first frame for a new segment, else a resampler's
    /// run-up before it, into what is there (crossfaded as main is, for a
    /// joined one); always on an output frame a run frame falls on. When
    /// the frames written up to there have none, from as far back as they
    /// may be written again ([`Want::floor`]), so they are once the parts
    /// are made ([`StemLanes::rewind_to`]).
    fn start(&mut self, line: Line, want: &Want, unkept: Unkept) {
        if line.along.is_some() {
            return self.start_along(line, want, unkept);
        }
        let track = want.track;
        let from = self
            .timeline
            .quiet_since(want.from)
            .map_or(want.from, |quiet| quiet.max(want.floor).min(want.from));
        let promised = self.timeline.promised(from);
        let cold = line.begun == SegStart::Cold && from <= line.seg_out;
        let fade = match line.begun {
            SegStart::Joined { fade } => fade,
            _ => 0,
        };
        self.buffer
            .restart(from, promised.max(from + fade as u64), fade);
        let preroll = if cold {
            0
        } else {
            (PREROLL_RESAMPLE * f64::from(line.out)).round() as i128
        };
        let (here, mut x) = (i128::from(from), i128::from(from) - preroll);
        // Parts that ended a while ago, within what the output may give
        // back: the new ones start no later than they ended, so the frames
        // kept stay one stretch (a jump back there finds them).
        let end = self.buffer.end();
        if !self.buffer.is_empty() && end < from && end + want.back >= from {
            x = x.min(i128::from(end));
        }
        let path = &track.info.path;
        // The run kept, the next track's (from its start), or a new one.
        let usable = |run: &StemRun| -> Option<(i128, i128)> {
            if &run.path != path || run.whole() || run.failed() || run.missing() {
                return None;
            }
            let grid = Grid::of(&line, run.start)?;
            let (mut o, mut r) = grid.at_or_before(x);
            let low = i128::from(run.kept_from());
            if cold && (o != here || r < low) {
                return None;
            }
            if r < low {
                (o, r) = grid.first_at_run(low);
            }
            let reach = i128::from(run.kept_end() + want.keep.1.max(u64::from(STEM_RATE)));
            (o <= here && r <= reach).then_some((o, r))
        };
        let (mut run, (o, r)) = self.take_run(usable).unwrap_or_else(|| {
            let start = run_start(&line, line.frame_at(x));
            let run = StemRun::start(path.clone(), start, unkept, self.store.clone());
            let grid = Grid::of(&line, start).expect("a run starts on the line's grid");
            let (o, r) = grid.at_or_before(x);
            (run, if r < 0 { grid.first_at_run(0) } else { (o, r) })
        });
        run.seek(r as u64);
        run.set_keep(want.keep.0, want.keep.1);
        let channels = usize::from(track.route.channels);
        self.stream = Some(Stream::new(line, o as i64, channels));
        self.run = Some(run);
    }

    /// The run whose stems `usable` takes (with what it found there): the
    /// track's, the next one's or one started for a move onto a map, taken
    /// out of its place.
    fn take_run<T>(&mut self, usable: impl Fn(&StemRun) -> Option<T>) -> Option<(StemRun, T)> {
        for slot in [&mut self.run, &mut self.prestart, &mut self.ahead] {
            if let Some(found) = slot.as_ref().and_then(&usable) {
                return slot.take().map(|run| (run, found));
            }
        }
        None
    }

    /// How long a write waits for stems, and how many pieces it makes at
    /// most: on a device none and [`PIECES`], offline a moment and all.
    fn budget(&self) -> (Duration, usize) {
        if self.offline {
            (WRITE_WAIT, usize::MAX)
        } else {
            (Duration::ZERO, PIECES)
        }
    }

    /// Makes parts until they reach output frame `target`, or the run has
    /// no more yet: `pieces` at most, the first waiting up to `wait` for its
    /// stems. Returns what it did, or why the run has none, when it has
    /// none for the track.
    fn make_to(
        &mut self,
        want: &Want,
        target: u64,
        (wait, pieces): (Duration, usize),
    ) -> Result<Making, Stopped> {
        let mut making = Making::default();
        let (Some(stream), Some(run)) = (self.stream.as_mut(), self.run.as_mut()) else {
            return Ok(making);
        };
        run.set_keep(want.keep.0, want.keep.1);
        let buffer = &mut self.buffer;
        let (mut wait, mut left) = (wait, pieces);
        while buffer.end() < target && left > 0 {
            left -= 1;
            match stream.make(run, wait, &mut |frame, parts| buffer.push(frame, parts)) {
                Made::Frames => (wait, making.made) = (Duration::ZERO, true),
                Made::Waiting => {
                    making.waiting = true;
                    break;
                }
                Made::Done => break,
                Made::Failed(why) => return Err(Stopped::Failed(why)),
                Made::Missing => return Err(Stopped::Missing),
            }
        }
        Ok(making)
    }

    /// The next track's run, started once this one's file is read (or its
    /// parts are made to its end), and how its parts stand (none when none
    /// are asked for: a head read along a map has its parts whatever was).
    fn follow_next(&mut self, want: &Want) -> Following {
        let (next, unkept) = match (want.next, self.unkept()) {
            (Next::Nothing, _) => {
                self.prestart = None;
                return Following::Fine;
            }
            (Next::Unable, _) | (_, None) => {
                self.prestart = None;
                return Following::Not;
            }
            (Next::Track(next), Some(unkept)) => (next, unkept),
        };
        if self.prestart.as_ref().is_some_and(|p| &p.path != next) {
            self.prestart = None;
        }
        // A run whose parts are made to the track's end takes no more: its
        // thread may hold the last of the file unread.
        let read = self.run.as_ref().is_some_and(StemRun::read_all)
            || self.stream.as_ref().is_some_and(|s| s.done);
        if self.prestart.is_none() && read {
            let store = self.store.clone();
            self.prestart = Some(StemRun::start(next.clone(), 0, unkept, store));
        }
        let Some(p) = self.prestart.as_mut() else {
            return Following::Coming;
        };
        let need = (AHEAD * f64::from(STEM_RATE)) as u64;
        let wait = if self.offline {
            WRITE_WAIT
        } else {
            Duration::ZERO
        };
        p.fetch(need, wait);
        if p.failed() || p.missing() {
            Following::Not
        } else if p.kept_end() >= need || p.received() {
            Following::Fine
        } else {
            Following::Coming
        }
    }

    /// The lane stream's frames for `main` (whole frames of `channels`,
    /// heard from output frame `from`): each main, then its parts (or
    /// nothing where they are not there), then its control word.
    fn lay_out(&mut self, main: &[i32], channels: usize, from: u64, followed: &Followed) -> &[i32] {
        let supply = followed.supply;
        // What is promised, up to main's end when the parts run to it.
        let promised = match supply.main_end {
            Some(end) if supply.through => self.timeline.promised(from).min(end),
            _ => self.timeline.promised(from),
        };
        debug_assert!(
            !followed.owed || supply.there >= promised,
            "parts promised to {promised} are there to {} only (from {from})",
            supply.there
        );
        let frames = main.len() / channels.max(1);
        self.plan = self.timeline.plan(from, frames as u64, supply);
        let width = self.buffer.width();
        self.wide.clear();
        self.wide.reserve(frames * (channels + width + 1));
        for (i, frame) in main.chunks_exact(channels).enumerate() {
            let o = from + i as u64;
            // Never a word without its parts (should what was promised not
            // be there).
            let word = if o < supply.there {
                self.plan.word(o)
            } else {
                Parts::None
            };
            self.wide.extend_from_slice(frame);
            if word == Parts::None {
                self.wide.resize(self.wide.len() + width, 0);
            } else {
                self.buffer.copy(o, &mut self.wide);
            }
            self.wide.push(word.word());
        }
        &self.wide
    }

    /// `took` frames of the last lay-out were written.
    fn commit(&mut self, from: u64, took: usize) {
        self.timeline.commit(&self.plan, from, took as u64);
    }
}

/// What a write of lanes knows besides the track and the next one, to make
/// its [`Want`].
#[derive(Clone, Copy)]
pub(super) struct Ask {
    from: u64,
    frames: usize,
    back: u64,
    keep: (u64, u64),
    able: bool,
    margin: u64,
    floor: u64,
}

impl Ask {
    pub(super) fn want<'a>(&self, track: &'a Track, next: Next<'a>) -> Want<'a> {
        Want {
            track,
            from: self.from,
            frames: self.frames,
            back: self.back,
            keep: self.keep,
            next,
            able: self.able,
            margin: self.margin,
            main_end: track.made_all().then_some(track.seg_out + track.emitted),
            rise: track.seg_out + track.parts_rise,
            cut: track.parts_cut,
            floor: self.floor,
        }
    }
}

impl Engine {
    /// Whether `track` may have parts now: on its plain route, or read
    /// along a map (its main is its parts' sum).
    pub(super) fn parts_able(&self, track: &Track) -> bool {
        track.plain() || track.along.is_some()
    }

    /// What a write of `frames` of `track` from output frame `from` asks
    /// of the stem lanes.
    pub(super) fn lane_ask(&self, track: &Track, from: u64, frames: usize) -> Ask {
        let (rate, ring) = self
            .head
            .out
            .as_ref()
            .map_or((1, 0), |o| (o.info.format.rate, o.info.ring_frames));
        // A jump window's stems, and what the output may give back, the
        // parts made ahead and a second besides.
        let held = ring as f64 / f64::from(rate.max(1)) + AHEAD + 1.0;
        let (back, ahead) = self.stem_keep(track);
        let keep = (back.max((held * f64::from(STEM_RATE)) as u64), ahead);
        // (Frames to come, a count-in's track's first, have none to write
        // again before them.)
        let floor = self
            .rewrite_floor(track)
            .filter(|&f| f < from)
            .unwrap_or(from);
        Ask {
            from,
            frames,
            back: ring,
            keep,
            able: self.parts_able(track),
            margin: self.lane_margin(rate),
            floor,
        }
    }

    /// Frames of parts made past what is written at the least, at `rate`:
    /// a mixer's switch's length past main's longest crossfade, so a track
    /// taking over without parts finds them there through its crossfade.
    pub(super) fn lane_margin(&self, rate: u32) -> u64 {
        ending_frames(rate) + frames_of(self.switch_fade, rate)
    }

    /// A count-in's silence is being written before `track`: its parts are
    /// made ahead from its first frame meanwhile (on a device, as after a
    /// write), so they are there as it starts.
    pub(super) fn lanes_ahead(&mut self, track: &Track) {
        if self.head.out.as_ref().is_none_or(|o| o.output.lanes() <= 1) {
            return;
        }
        let from = track.seg_out;
        let ask = self.lane_ask(track, from, 0);
        let next = next_of(&self.queue, track, self.hold, self.transition.is_none());
        let failed = self.head.lanes.before(&ask.want(track, next)).failed;
        let failed = failed.or_else(|| {
            self.head
                .lanes
                .after(&ask.want(track, Next::Nothing), from)
                .err()
        });
        if let Some(reason) = failed {
            let index = track.index;
            self.emit(Event::PartsStopped { index, reason });
        }
    }

    /// Writes `track`'s output from `out_at` up to sample `limit` (at most
    /// [`LANE_CHUNK`] frames of it) on an output of lanes, and returns the
    /// frames it took: each main as it is, then the parts the stem lanes
    /// have for it and its control word; offline, no further than the
    /// parts are made. While a device's ring is full the parts are made on
    /// first, and once there are none to make now the write waits up to
    /// `wait` for room, as on a plain output (else the thread would spin on
    /// the full ring).
    /// Parts made behind what is written have what the output has not
    /// played given back first, and written again with them
    /// ([`Engine::rewrite_lanes`]).
    pub(super) fn write_lanes(
        &mut self,
        track: &mut Track,
        limit: usize,
        wait: Duration,
    ) -> Result<usize, Error> {
        self.rewrite_lanes(track)?;
        let out = self.head.out.as_ref().ok_or(Error::Stopped)?;
        let channels = usize::from(out.info.format.channels);
        let ring = out.info.ring_frames;
        let from = out.output.written();
        let full = !self.offline && ring > 0 && from - out.clock.position().min(from) >= ring;
        let mut frames = ((limit - track.out_at) / channels).min(LANE_CHUNK);
        // A route's rise ends a write, so the parts may be there from its
        // very end.
        let rise = track.seg_out + track.parts_rise;
        if from < rise {
            frames = frames.min((rise - from) as usize);
        }
        let ask = self.lane_ask(track, from, frames);
        let (able, margin) = (ask.able, ask.margin);
        let next = next_of(&self.queue, track, self.hold, self.transition.is_none());
        if full {
            match self.head.lanes.after(&ask.want(track, Next::Nothing), from) {
                // None to make now: the write below waits for room.
                Ok(false) => {}
                Ok(true) => return Ok(0),
                Err(reason) => {
                    let index = track.index;
                    self.emit(Event::PartsStopped { index, reason });
                    return Ok(0);
                }
            }
        }
        let mut followed = self.head.lanes.before(&ask.want(track, next));
        if let Some(reason) = followed.failed.take() {
            let index = track.index;
            self.emit(Event::PartsStopped { index, reason });
        }
        // Near the end of its parts main is decoded on until its own end is
        // known (as far as a seam past theirs: a longer main ends them
        // before it anyway), so the parts may run to it.
        let parts_end = self.head.lanes.near_end(from + frames as u64 + margin);
        while let Some(end) = parts_end.filter(|_| able && !track.made_all()) {
            if track.seg_out + track.emitted > end + margin {
                break;
            }
            if let Progress::Complete = self.produce(track, Duration::ZERO)? {
                break;
            }
        }
        followed.supply.main_end = ask.want(track, Next::Nothing).main_end;
        let frames = followed.limit.map_or(frames, |l| frames.min(l as usize));
        if frames == 0 {
            return Ok(0);
        }
        let main = &track.out_buf[track.out_at..track.out_at + frames * channels];
        let wide = self.head.lanes.lay_out(main, channels, from, &followed);
        let out = self.head.out.as_mut().ok_or(Error::Stopped)?;
        let took = out.output.write(wide, Some(wait))?;
        self.head.lanes.commit(from, took);
        if let Err(reason) = self
            .head
            .lanes
            .after(&ask.want(track, Next::Nothing), from + took as u64)
        {
            let index = track.index;
            self.emit(Event::PartsStopped { index, reason });
        }
        track.out_at += took * channels;
        Ok(took)
    }
}

/// The track `track` runs into in `queue`, for the parts at its end
/// (`hold`: playback holds at the end; `able`: no transition, whose next
/// track plays on a head of its own).
pub(super) fn next_of<'a>(queue: &'a [PathBuf], track: &Track, hold: bool, able: bool) -> Next<'a> {
    match queue.get(track.index + 1) {
        _ if hold => Next::Nothing,
        None => Next::Nothing,
        Some(path) if able => Next::Track(path),
        Some(_) => Next::Unable,
    }
}
