//! A track's stems: drums, bass, vocals and other, separated as it plays
//! or read from where they are kept.
//!
//! A stem run separates one track in a thread of its own, ahead of the
//! playhead: it decodes the file as it is stored, makes it stereo at the
//! separator's rate ([`StereoFeed`]), runs [`Splitter`] and hands the
//! finished stems to the engine. The engine hands them on beside the file
//! as parts of its output (in [`LANES`]' order, for a mixer to play on its
//! strips), or (a transition, which mixes its two tracks itself) sums them
//! faded by the transition's lanes on a [`Stage::Stems`](crate::Stage::Stems)
//! route.

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TryRecvError};
use std::sync::Arc;
use std::time::Duration;

pub use jodel_rhythm::stems::{Separator, Stem, RATE, STEMS};
use jodel_rhythm::stems::{Splitter, StemBlock};

use crate::resample::Resampler;
use crate::source::{Source, SCALE_24 as FULL_SCALE};
use crate::stem_file::{StemFile, StemStore};

/// The stems in the order a player hands them on as parts of its output
/// (lanes 1 to 4, after main), the order people name them in: drums, bass,
/// vocals, other.
pub const LANES: [Stem; STEMS] = [Stem::Drums, Stem::Bass, Stem::Vocals, Stem::Other];

/// Where a stem run gets its separator: called in the run's thread, so
/// setting a GPU up never holds up playback. `None`: there is none.
pub type SeparatorSource = fn() -> Option<&'static dyn Separator>;

/// Where a stem run gets the stems of a track that are not kept.
#[derive(Clone, Copy)]
pub(crate) enum Unkept {
    /// Nowhere: the run has none ([`Pull::Missing`]).
    Missing,
    /// Separated by this separator.
    Separate(SeparatorSource),
    /// The whole mix as the other instruments' stem, the rest silent: the
    /// file as it is, in a stem (no transform, so nothing of it changes).
    Whole,
}

impl From<Option<SeparatorSource>> for Unkept {
    fn from(separator: Option<SeparatorSource>) -> Unkept {
        separator.map_or(Unkept::Missing, Unkept::Separate)
    }
}

/// `stem`'s place in [`Stem::ALL`], the order a run keeps the stems in.
fn slot(stem: Stem) -> usize {
    Stem::ALL.iter().position(|&s| s == stem).unwrap_or(0)
}

/// Decoded blocks as stereo at the separator's rate ([`RATE`]), handed on
/// in pieces of a second: the first two channels (a mono file's one
/// twice), resampled when the file has another rate.
pub struct StereoFeed {
    resampler: Option<Resampler>,
    stereo: Vec<f64>,
    out: Vec<f64>,
    left: Vec<f32>,
    right: Vec<f32>,
}

/// Samples per piece [`StereoFeed`] hands on.
const PIECE: usize = RATE as usize;

impl StereoFeed {
    /// A feed for a file at `rate`.
    pub fn new(rate: u32) -> StereoFeed {
        StereoFeed {
            resampler: (rate != RATE).then(|| Resampler::new(rate, RATE, 2)),
            stereo: Vec::new(),
            out: Vec::new(),
            left: Vec::new(),
            right: Vec::new(),
        }
    }

    /// Takes a decoded block (`channels` interleaved, in 24-bit units, as
    /// [`Source::read_unquantized`] gives them), handing each full piece
    /// to `put` as left and right (full scale ±1).
    pub fn push(
        &mut self,
        block: &[f64],
        channels: usize,
        put: &mut dyn FnMut(Vec<f32>, Vec<f32>),
    ) {
        self.stereo.clear();
        for frame in block.chunks_exact(channels.max(1)) {
            let left = frame[0] / FULL_SCALE;
            let right = frame.get(1).map_or(left, |r| r / FULL_SCALE);
            self.stereo.extend([left, right]);
        }
        let ready = match self.resampler.as_mut() {
            Some(r) => {
                self.out.clear();
                r.process_f64(&self.stereo, &mut self.out);
                &self.out
            }
            None => &self.stereo,
        };
        for frame in ready.chunks_exact(2) {
            self.left.push(frame[0] as f32);
            self.right.push(frame[1] as f32);
        }
        while self.left.len() >= PIECE {
            let (left, right) = (self.left.split_off(PIECE), self.right.split_off(PIECE));
            put(
                std::mem::replace(&mut self.left, left),
                std::mem::replace(&mut self.right, right),
            );
        }
    }

    /// Hands on the rest.
    pub fn finish(mut self, put: &mut dyn FnMut(Vec<f32>, Vec<f32>)) {
        if let Some(r) = self.resampler.as_mut() {
            self.out.clear();
            r.finish_f64(&mut self.out);
            for frame in self.out.chunks_exact(2) {
                self.left.push(frame[0] as f32);
                self.right.push(frame[1] as f32);
            }
        }
        if !self.left.is_empty() {
            put(self.left, self.right);
        }
    }
}

/// Finished stems handed on at a time (about 4.8 s each): how far a run
/// separates ahead of what the engine has taken.
const AHEAD: usize = 4;

/// A stretch of finished stems: `[stem][channel]` samples, in the order of
/// [`Stem::ALL`].
struct Separated {
    stems: Vec<[Vec<f32>; 2]>,
    /// Made by the separator (not read where they are kept).
    separated: bool,
}

impl Separated {
    fn of(block: &StemBlock) -> Separated {
        Separated {
            stems: block
                .stems
                .iter()
                .map(|[l, r]| [l.to_vec(), r.to_vec()])
                .collect(),
            separated: true,
        }
    }
}

enum Message {
    Stems(Separated),
    End,
    Failed(String),
    /// The stems are not kept, and the run has no separator.
    Missing,
}

/// What [`StemRun::mix`] or [`StemRun::take`] gave.
pub(crate) enum Pull {
    /// Frames were added.
    Frames,
    /// Nothing is separated yet.
    Waiting,
    /// The track's stems have all been taken.
    End,
    /// The separation failed.
    Failed(String),
    /// The track's stems are not kept, and the run separates none.
    Missing,
}

/// One track separated from a frame on, in a thread of its own. Dropping it
/// stops the thread (it is not waited for).
///
/// The separated stems are kept for a while ([`StemRun::set_keep`]): some
/// behind the frame handed on next, and as much as the thread has separated
/// ahead of it, so the run can go on from another frame of that stretch at
/// once ([`StemRun::seek`]): the parts of a jump within the track.
pub(crate) struct StemRun {
    /// The file.
    pub(crate) path: PathBuf,
    /// The track frame it starts at.
    pub(crate) start: u64,
    /// Its stems are the whole mix ([`Unkept::Whole`]), not kept nor
    /// separated (unless the store had them after all).
    pub(crate) whole: bool,
    stems: Receiver<Message>,
    /// The whole file has been read (the rest is being separated).
    read: Arc<AtomicBool>,
    /// Set when the run is dropped: its thread stops before the next chunk.
    cancel: Arc<AtomicBool>,
    /// The stems kept, `[stem][channel]` in the order of [`Stem::ALL`],
    /// from run frame `kept_from` (frames at [`RATE`] from `start`).
    kept: Vec<[Vec<f32>; 2]>,
    kept_from: u64,
    /// Every stem has come from the thread (or it failed: `failed`).
    received: bool,
    failed: Option<String>,
    /// The stems are not kept and none are separated.
    missing: bool,
    /// The separator has handed stems on since [`StemRun::take_separated`]
    /// was last asked.
    separated: bool,
    /// Frames kept behind `mixed`, and taken from the thread ahead of it.
    keep: (u64, u64),
    /// Each stem's gain now, in the order of [`Stem::ALL`]; `None` before
    /// the first mix.
    gains: Option<[f64; STEMS]>,
    /// The run frame mixed next.
    mixed: u64,
    /// Keeps the mix under full scale.
    limiter: Limiter,
}

/// The highest a stem mix goes, of full scale (−0.2 dBFS).
pub const CEILING: f64 = 0.977;
/// How far the limiter looks ahead, and how long its gain takes to come
/// down to what a peak needs (2 ms).
const LOOKAHEAD: usize = RATE as usize / 500;
/// How fast its gain comes back up, per frame (from nothing to whole in
/// 0.2 s).
const RELEASE: f64 = 5.0 / RATE as f64;

/// Keeps a stem mix under [`CEILING`]. Stems separated and summed again go
/// over full scale, the more so without one of them (a singer cuts through
/// some of the band's peaks): over 24 songs the band without its voice
/// peaked up to 2.15 dB over (11 of them over), all four up to 0.33 dB,
/// which the requantization to 24 bits clipped.
///
/// It looks ahead in the stems the run already has (no delay: the mix
/// comes out at the frames it always did). Each frame needs a gain that
/// keeps it under the ceiling; the gain taken is the least needed over the
/// next [`LOOKAHEAD`] frames, averaged over the last [`LOOKAHEAD`] (so it
/// comes down smoothly, and is never more than any frame needs), and it
/// goes back up at [`RELEASE`]. Where nothing goes over, it is exactly 1
/// and the mix is what it was: a track moving between its file and its
/// stems still sounds the same.
#[derive(Debug)]
struct Limiter {
    /// The gain now.
    gain: f64,
    /// The last `LOOKAHEAD + 1` least gains ahead, and how many are under 1.
    mins: std::collections::VecDeque<f64>,
    under: usize,
    /// Nothing went through it yet (a run's start, or where it went on
    /// from): it starts at the gain its first frames need, nothing before
    /// them to come down from.
    fresh: bool,
}

impl Limiter {
    fn new() -> Limiter {
        Limiter {
            gain: 1.0,
            mins: std::iter::repeat_n(1.0, LOOKAHEAD + 1).collect(),
            under: 0,
            fresh: true,
        }
    }

    /// The gain a frame of `left` and `right` (in 24-bit units) needs.
    fn need(left: f64, right: f64) -> f64 {
        let peak = left.abs().max(right.abs());
        let top = CEILING * FULL_SCALE;
        if peak > top {
            top / peak
        } else {
            1.0
        }
    }

    /// Scales `frames` (interleaved stereo, 24-bit units), the frames after
    /// them needing `ahead` (as far as they are known).
    fn apply(&mut self, frames: &mut [f64], ahead: &[f64]) {
        let n = frames.len() / 2;
        let needs: Vec<f64> = frames
            .chunks_exact(2)
            .map(|f| Self::need(f[0], f[1]))
            .chain(ahead.iter().copied())
            .collect();
        // Nothing over, nothing coming down from before: as it is.
        if self.gain == 1.0 && self.under == 0 && needs.iter().all(|&g| g == 1.0) {
            self.fresh &= n == 0;
            return;
        }
        let least_from = |j: usize| {
            needs[j..needs.len().min(j + LOOKAHEAD + 1)]
                .iter()
                .copied()
                .fold(1.0f64, f64::min)
        };
        if self.fresh && n > 0 {
            // As if the frames before had needed what the first ones do.
            let first = least_from(0);
            self.mins.iter_mut().for_each(|m| *m = first);
            self.under = if first < 1.0 { self.mins.len() } else { 0 };
            self.gain = first;
            self.fresh = false;
        }
        for j in 0..n {
            let least = least_from(j);
            if let Some(old) = self.mins.pop_front() {
                self.under -= usize::from(old < 1.0);
            }
            self.mins.push_back(least);
            self.under += usize::from(least < 1.0);
            // The average of the least gains ahead (exactly 1 once none is
            // under it, never drifting off it).
            let average = if self.under == 0 {
                1.0
            } else {
                self.mins.iter().sum::<f64>() / self.mins.len() as f64
            };
            self.gain = (self.gain + RELEASE).min(average).min(1.0);
            frames[2 * j] *= self.gain;
            frames[2 * j + 1] *= self.gain;
        }
    }
}

/// Each stem's gain at a time of its track (seconds), in [`Stem::ALL`]
/// order: a transition's fades.
pub(crate) type Gains = Arc<dyn Fn(f64) -> [f64; STEMS] + Send + Sync>;

/// Frames over which a [`Gains`] is read once (1.5 ms; the gains ramp to
/// it at [`RAMP`]).
const GAIN_BLOCK: usize = 64;

/// A separator that puts the whole music in the other instruments' stem:
/// what a transition plays through without a stem separator, so that its
/// tracks fade as one.
pub(crate) struct WholeMix;

impl Separator for WholeMix {
    fn name(&self) -> String {
        "the whole mix".into()
    }

    fn masks(&self, frames: usize, stft: &[f32]) -> Result<Vec<f32>, String> {
        let block = stft.len();
        let mut masks = vec![0.0f32; STEMS * block];
        let other = Stem::ALL
            .iter()
            .position(|&s| s == Stem::Other)
            .unwrap_or(STEMS - 1);
        for v in masks[other * block..][..block].chunks_mut(2) {
            v[0] = 1.0;
        }
        debug_assert_eq!(block % frames.max(1), 0);
        Ok(masks)
    }
}

/// The [`WholeMix`] separator, as a [`SeparatorSource`].
pub(crate) fn whole_mix() -> Option<&'static dyn Separator> {
    static WHOLE: WholeMix = WholeMix;
    Some(&WHOLE)
}

/// Frames a stem's gain takes at most to reach a new one (10 ms), so a
/// transition's fades never click.
const RAMP: f64 = RATE as f64 / 100.0;

impl StemRun {
    /// Starts the stems of `path` from track frame `start`: read from
    /// `store` where it keeps them, else as `unkept` says. The store is
    /// asked in the run's thread.
    pub(crate) fn start(
        path: PathBuf,
        start: u64,
        unkept: Unkept,
        store: Option<Arc<dyn StemStore>>,
    ) -> StemRun {
        let (tx, rx) = mpsc::sync_channel(AHEAD);
        let read = Arc::new(AtomicBool::new(false));
        let cancel = Arc::new(AtomicBool::new(false));
        let worker = {
            let (path, read, cancel) = (path.clone(), read.clone(), cancel.clone());
            move || {
                let kept = store.as_ref().and_then(|s| s.open(&path));
                let done = match (kept, unkept) {
                    (Some(file), _) => read_kept(file, start, &tx, &read, &cancel),
                    (None, Unkept::Separate(separator)) => {
                        separate(&path, start, separator, &tx, &read, &cancel)
                    }
                    (None, Unkept::Whole) => whole(&path, start, &tx, &read, &cancel),
                    (None, Unkept::Missing) => {
                        let _ = tx.send(Message::Missing);
                        return;
                    }
                };
                let message = match done {
                    Ok(()) => Message::End,
                    Err(e) => Message::Failed(e),
                };
                let _ = tx.send(message);
            }
        };
        let spawned = std::thread::Builder::new()
            .name("jodel-stems".into())
            .spawn(worker);
        let stems = match spawned {
            Ok(_) => rx,
            Err(e) => {
                let (tx, rx) = mpsc::sync_channel(1);
                let _ = tx.send(Message::Failed(format!("no thread for the stems: {e}")));
                rx
            }
        };
        StemRun {
            path,
            start,
            whole: matches!(unkept, Unkept::Whole),
            stems,
            read,
            cancel,
            kept: (0..STEMS).map(|_| [Vec::new(), Vec::new()]).collect(),
            kept_from: 0,
            received: false,
            failed: None,
            missing: false,
            separated: false,
            keep: (0, 0),
            gains: None,
            mixed: 0,
            limiter: Limiter::new(),
        }
    }

    /// Whether the whole file has been read, so another run can start
    /// without falling behind this one.
    pub(crate) fn read_all(&self) -> bool {
        self.read.load(Ordering::Relaxed)
    }

    /// Keeps `back` frames (at [`RATE`]) behind the frame mixed next, and
    /// takes up to `ahead` from the thread ahead of it (none: the stems go
    /// once mixed, and come as they are needed).
    pub(crate) fn set_keep(&mut self, back: u64, ahead: u64) {
        self.keep = (back, ahead);
    }

    /// The first run frame kept.
    pub(crate) fn kept_from(&self) -> u64 {
        self.kept_from
    }

    /// The run frame after the last one kept.
    pub(crate) fn kept_end(&self) -> u64 {
        self.kept_from + self.kept[0][0].len() as u64
    }

    /// Every stem has come from the thread: nothing past
    /// [`StemRun::kept_end`] will.
    pub(crate) fn received(&self) -> bool {
        self.received
    }

    /// Whether the separation failed (what it kept still plays).
    pub(crate) fn failed(&self) -> bool {
        self.failed.is_some()
    }

    /// Whether the track's stems turned out not kept, with none separated.
    pub(crate) fn missing(&self) -> bool {
        self.missing
    }

    /// Whether the separator has handed stems on (not read where they are
    /// kept) since this was last asked: it works.
    pub(crate) fn take_separated(&mut self) -> bool {
        std::mem::take(&mut self.separated)
    }

    /// Mixes from run frame `frame` next, if it is kept or still to come
    /// (what comes before it is dropped); whether it does.
    pub(crate) fn seek(&mut self, frame: u64) -> bool {
        if frame < self.kept_from {
            return false;
        }
        self.mixed = frame;
        self.limiter = Limiter::new();
        true
    }

    /// Takes what the thread has separated, as far as [`StemRun::keep`]
    /// ahead of the frame mixed next, waiting up to `wait` while the stems
    /// end at or before frame `need`.
    pub(crate) fn fetch(&mut self, need: u64, wait: Duration) {
        let mut wait = wait;
        while !self.received {
            let short = self.kept_end() <= need;
            if !short && self.kept_end() >= self.mixed + self.keep.1 {
                break;
            }
            let message = if short {
                match self.stems.recv_timeout(wait) {
                    Ok(m) => m,
                    Err(RecvTimeoutError::Timeout) => break,
                    Err(RecvTimeoutError::Disconnected) => {
                        Message::Failed("the stems stopped".into())
                    }
                }
            } else {
                match self.stems.try_recv() {
                    Ok(m) => m,
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => Message::Failed("the stems stopped".into()),
                }
            };
            // One wait a call.
            wait = Duration::ZERO;
            match message {
                Message::Stems(s) => {
                    self.separated |= s.separated;
                    for (kept, [l, r]) in self.kept.iter_mut().zip(s.stems) {
                        kept[0].extend_from_slice(&l);
                        kept[1].extend_from_slice(&r);
                    }
                }
                Message::End => self.received = true,
                Message::Failed(e) => {
                    self.received = true;
                    self.failed = Some(e);
                }
                Message::Missing => {
                    self.received = true;
                    self.missing = true;
                }
            }
        }
    }

    /// Drops what is kept before `keep.0` frames behind the frame mixed
    /// next (a second at a time).
    fn trim(&mut self) {
        let from = self.mixed.saturating_sub(self.keep.0);
        let len = self.kept[0][0].len() as u64;
        let drop = from.saturating_sub(self.kept_from).min(len);
        // In steps of a quarter of what is kept behind (a second at
        // least): each moves all that is kept.
        let step = (self.keep.0 / 4).max(u64::from(RATE));
        if drop >= step || (drop == len && drop > 0) {
            for kept in &mut self.kept {
                for channel in kept {
                    channel.drain(..drop as usize);
                }
            }
            self.kept_from += drop;
        }
    }

    /// Adds up to `frames` frames of the stems summed to `out` (interleaved
    /// stereo, in 24-bit units), waiting up to `wait` for the next stems
    /// when none are left: each stem at its gain in `gains` (with the
    /// seconds of the run's first frame in its track), else every one at
    /// one (the whole track).
    pub(crate) fn mix(
        &mut self,
        frames: usize,
        gains: Option<(&Gains, f64)>,
        out: &mut Vec<f64>,
        wait: Duration,
    ) -> Pull {
        // The limiter looks LOOKAHEAD frames past what is mixed: while more
        // stems are to come, those stay unmixed until they have.
        let (first, n) = match self.span(frames, LOOKAHEAD as u64, wait) {
            Ok(span) => span,
            Err(pull) => return pull,
        };
        let end = self.kept_end();
        let target_at = |frame: u64| match gains {
            Some((g, origin)) => g(origin + frame as f64 / f64::from(RATE)),
            None => [1.0; STEMS],
        };
        let mut target = target_at(self.mixed);
        let mixed = self.mixed;
        let gains = self.gains.get_or_insert(target);
        let begin = out.len();
        out.reserve(2 * n);
        for i in first..first + n {
            let k = (i - first) as u64;
            // The gains are read every block.
            if k > 0 && k % GAIN_BLOCK as u64 == 0 {
                target = target_at(mixed + k);
            }
            let (mut l, mut r) = (0.0f64, 0.0f64);
            for (s, [sl, sr]) in self.kept.iter().enumerate() {
                let g = &mut gains[s];
                // Towards its target over at most RAMP frames, stopping on
                // it (a level between off and on too).
                if *g != target[s] {
                    *g = if target[s] > *g {
                        (*g + 1.0 / RAMP).min(target[s])
                    } else {
                        (*g - 1.0 / RAMP).max(target[s])
                    };
                }
                if *g > 0.0 {
                    l += *g * f64::from(sl[i]);
                    r += *g * f64::from(sr[i]);
                }
            }
            out.extend([l * FULL_SCALE, r * FULL_SCALE]);
        }
        // What the frames after these need, as far as the run has them (at
        // the gains due then): the limiter looks there.
        let ahead_frames = (end - (mixed + n as u64)).min(LOOKAHEAD as u64) as usize;
        let ahead: Vec<f64> = (0..ahead_frames)
            .map(|a| {
                let i = first + n + a;
                let g = target_at(mixed + (n + a) as u64);
                let (mut l, mut r) = (0.0f64, 0.0f64);
                for (s, [sl, sr]) in self.kept.iter().enumerate() {
                    l += g[s] * f64::from(sl[i]);
                    r += g[s] * f64::from(sr[i]);
                }
                Limiter::need(l * FULL_SCALE, r * FULL_SCALE)
            })
            .collect();
        self.limiter.apply(&mut out[begin..], &ahead);
        self.mixed += n as u64;
        self.trim();
        Pull::Frames
    }
}

impl StemRun {
    /// The stretch handed on next, up to `frames`: its first frame's index
    /// in what is kept and its length, waiting up to `wait` for the next
    /// stems when none are left; or what to say instead (none yet, the end,
    /// the failure). While more stems are to come, the `hold` frames after
    /// what is kept stay unmixed until they have (what a look ahead needs).
    fn span(&mut self, frames: usize, hold: u64, wait: Duration) -> Result<(usize, usize), Pull> {
        self.fetch(self.mixed + hold, wait);
        let end = self.kept_end();
        if self.mixed >= end {
            return Err(match (&self.failed, self.received) {
                (Some(e), _) => Pull::Failed(e.clone()),
                (None, true) if self.missing => Pull::Missing,
                (None, true) => Pull::End,
                (None, false) => Pull::Waiting,
            });
        }
        let mixable = if self.received {
            end
        } else {
            end.saturating_sub(hold)
        };
        if self.mixed >= mixable {
            return Err(Pull::Waiting);
        }
        let first = (self.mixed - self.kept_from) as usize;
        Ok((first, frames.min((mixable - self.mixed) as usize)))
    }

    /// Appends up to `frames` frames of the four stems as they are, not
    /// mixed: each frame the stems in [`LANES`] order, each left then right
    /// (eight samples, in 24-bit units), waiting up to `wait` for the next
    /// stems when none are left. Nothing limits them: they add up to the
    /// track.
    pub(crate) fn take(&mut self, frames: usize, out: &mut Vec<f64>, wait: Duration) -> Pull {
        let (first, n) = match self.span(frames, 0, wait) {
            Ok(span) => span,
            Err(pull) => return pull,
        };
        let slots = LANES.map(slot);
        out.reserve(2 * STEMS * n);
        for i in first..first + n {
            for &s in &slots {
                let [l, r] = &self.kept[s];
                out.extend([f64::from(l[i]) * FULL_SCALE, f64::from(r[i]) * FULL_SCALE]);
            }
        }
        self.mixed += n as u64;
        self.trim();
        Pull::Frames
    }
}

impl Drop for StemRun {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// Frames of kept stems sent on at a time (a second).
const KEPT_PIECE: usize = RATE as usize;

/// The run's thread with the stems kept ([`StemStore`]): reads them from
/// track frame `start` on and sends them until they end or the run is
/// dropped (`cancel`). Nothing is separated.
fn read_kept(
    mut file: StemFile,
    start: u64,
    tx: &SyncSender<Message>,
    read: &AtomicBool,
    cancel: &AtomicBool,
) -> Result<(), String> {
    let mut at = crate::resample::output_frames(file.source_rate(), RATE, start);
    while at < file.frames() {
        if cancel.load(Ordering::Relaxed) {
            return Ok(());
        }
        let stems = file.read(at, KEPT_PIECE).map_err(|e| e.to_string())?;
        let n = stems[0][0].len() as u64;
        if n == 0 {
            break;
        }
        at += n;
        let piece = Separated {
            stems: Vec::from(stems),
            separated: false,
        };
        if tx.send(Message::Stems(piece)).is_err() {
            return Ok(());
        }
    }
    read.store(true, Ordering::Relaxed);
    Ok(())
}

/// The run's thread with no separator ([`Unkept::Whole`]): decodes the
/// file and sends it on as the other instruments' stem, the others silent,
/// until it ends or the run is dropped (`cancel`).
fn whole(
    path: &std::path::Path,
    start: u64,
    tx: &SyncSender<Message>,
    read: &AtomicBool,
    cancel: &AtomicBool,
) -> Result<(), String> {
    let mut source = Source::open(path).map_err(|e| e.to_string())?;
    if start > 0 {
        source.seek(start).map_err(|e| e.to_string())?;
    }
    let channels = usize::from(source.info().channels).max(1);
    let mut feed = StereoFeed::new(source.info().rate);
    let other = slot(Stem::Other);
    let gone = Cell::new(false);
    let stopped = || gone.get() || cancel.load(Ordering::Relaxed);
    let mut put = |left: Vec<f32>, right: Vec<f32>| {
        if stopped() {
            return;
        }
        let n = left.len();
        let mut stems: Vec<[Vec<f32>; 2]> =
            (0..STEMS).map(|_| [vec![0.0; n], vec![0.0; n]]).collect();
        stems[other] = [left, right];
        let piece = Separated {
            stems,
            separated: false,
        };
        if tx.send(Message::Stems(piece)).is_err() {
            gone.set(true);
        }
    };
    let mut block = Vec::new();
    loop {
        block.clear();
        if source
            .read_unquantized(&mut block)
            .map_err(|e| e.to_string())?
            == 0
        {
            break;
        }
        feed.push(&block, channels, &mut put);
        if stopped() {
            return Ok(());
        }
    }
    read.store(true, Ordering::Relaxed);
    feed.finish(&mut put);
    Ok(())
}

/// The run's thread: decodes, separates and sends the stems on until the
/// file ends or the run is dropped (`cancel`).
fn separate(
    path: &std::path::Path,
    start: u64,
    separator: SeparatorSource,
    tx: &SyncSender<Message>,
    read: &AtomicBool,
    cancel: &AtomicBool,
) -> Result<(), String> {
    let separator = separator().ok_or("no GPU for the stems")?;
    let mut source = Source::open(path).map_err(|e| e.to_string())?;
    if start > 0 {
        source.seek(start).map_err(|e| e.to_string())?;
    }
    let channels = usize::from(source.info().channels).max(1);
    let mut feed = StereoFeed::new(source.info().rate);
    let mut splitter = Splitter::new(separator);
    // The run was dropped (stop), or the separator failed.
    let gone = Cell::new(false);
    let stopped = || gone.get() || cancel.load(Ordering::Relaxed);
    let failed = RefCell::new(None);
    let mut sink = |b: &StemBlock| {
        if !gone.get() && tx.send(Message::Stems(Separated::of(b))).is_err() {
            gone.set(true);
        }
    };
    let mut put = |left: Vec<f32>, right: Vec<f32>| {
        if failed.borrow().is_none() && !stopped() {
            if let Err(e) = splitter.push(&left, &right, &mut sink) {
                *failed.borrow_mut() = Some(e);
            }
        }
    };
    let mut block = Vec::new();
    loop {
        block.clear();
        if source
            .read_unquantized(&mut block)
            .map_err(|e| e.to_string())?
            == 0
        {
            break;
        }
        feed.push(&block, channels, &mut put);
        if failed.borrow().is_some() || stopped() {
            break;
        }
    }
    read.store(true, Ordering::Relaxed);
    if failed.borrow().is_none() && !stopped() {
        feed.finish(&mut put);
    }
    if let Some(e) = failed.take() {
        return Err(e);
    }
    if stopped() {
        return Ok(());
    }
    splitter.finish(&mut sink)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mix going twice over full scale for a while, between quiet music,
    /// through the limiter in pieces as the run mixes it (each looking at
    /// the start of the next): nothing over the ceiling; the quiet music
    /// before and after, past the limiter's lookahead and release, exactly
    /// as it was.
    #[test]
    fn the_limiter_keeps_a_mix_under_full_scale_and_leaves_the_rest() {
        let rate = RATE as usize;
        let n = 2 * rate;
        let (loud_from, loud_to) = (rate / 2, rate);
        let signal: Vec<f64> = (0..n)
            .flat_map(|i| {
                let t = i as f64 / rate as f64;
                let level = if (loud_from..loud_to).contains(&i) {
                    2.0
                } else {
                    0.3
                };
                let v = level * FULL_SCALE * (2.0 * std::f64::consts::PI * 220.0 * t).sin();
                [v, -0.5 * v]
            })
            .collect();
        let mut out = signal.clone();
        let mut limiter = Limiter::new();
        let piece = 1000;
        for start in (0..n).step_by(piece) {
            let end = (start + piece).min(n);
            let ahead: Vec<f64> = (end..(end + LOOKAHEAD).min(n))
                .map(|i| Limiter::need(signal[2 * i], signal[2 * i + 1]))
                .collect();
            limiter.apply(&mut out[2 * start..2 * end], &ahead);
        }
        let peak = out.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(
            peak <= CEILING * FULL_SCALE + 1e-6,
            "{} dBFS",
            20.0 * (peak / FULL_SCALE).log10()
        );
        // Untouched before the lookahead reaches the loud part, and once
        // the release is over after it.
        let before = 2 * (loud_from - LOOKAHEAD - 1);
        assert!(out[..before] == signal[..before]);
        let after = 2 * (loud_to + (1.0 / RELEASE) as usize + LOOKAHEAD);
        assert!(out[after..] == signal[after..]);
        // The loud part held near the ceiling, not far under it.
        let held = out[2 * (loud_from + rate / 10)..2 * loud_to]
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(held > 0.9 * CEILING * FULL_SCALE, "{held}");
    }

    #[test]
    fn the_lanes_are_every_stem_once() {
        let mut slots = LANES.map(slot);
        slots.sort_unstable();
        assert_eq!(slots, [0, 1, 2, 3]);
        assert_eq!(Stem::ALL[slot(Stem::Vocals)], Stem::Vocals);
    }

    #[test]
    fn the_feed_hands_on_stereo_seconds() {
        let mut pieces = Vec::new();
        let mut feed = StereoFeed::new(RATE);
        // 1.5 s of mono at half scale.
        let block = vec![FULL_SCALE / 2.0; RATE as usize * 3 / 2];
        feed.push(&block, 1, &mut |l, r| pieces.push((l, r)));
        assert_eq!(pieces.len(), 1);
        feed.finish(&mut |l, r| pieces.push((l, r)));
        assert_eq!(pieces.len(), 2);
        assert_eq!(pieces[0].0.len(), RATE as usize);
        assert_eq!(pieces[1].1.len(), RATE as usize / 2);
        assert!(pieces
            .iter()
            .all(|(l, r)| l == r && l.iter().all(|&x| x == 0.5)));
    }
}
