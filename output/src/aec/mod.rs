//! Echo cancellation for the microphone: the music the speakers play,
//! heard again by the microphone, taken out of it, so what is left is the
//! singer.
//!
//! The microphone is only listened to, never played (see
//! [`crate::input`]), so there is no loop: the echo is the music alone,
//! and the music is known exactly. The canceller is given, block by block,
//! what the microphone took in and what went to the speakers at the same
//! moment (the reference: left and right). One rule above the others:
//! **the singer is never taken out.** Some of the music's echo may be left
//! in the voice; the voice is never lost.
//!
//! It works in three parts:
//!
//! 1. **Finding the delay** ([`delay`]): the microphone's bins against the
//!    reference's, their phases only, over up to [`MAX_DELAY`]; the
//!    filter's window is moved to where the echo starts. Most echoes start
//!    well inside the window as it is (the output's and the input's
//!    latency, a few tens of ms); a wireless speaker's do not.
//! 2. **A linear echo canceller**: an adaptive filter learns how the
//!    reference reaches the microphone (the speakers, the room) and
//!    subtracts its prediction of the echo. It is a partitioned-block
//!    frequency-domain adaptive filter ([`TAIL`] long) on two channels of
//!    the reference, what both speakers play alike and what they play apart
//!    ([`reference::CHANNELS`]): a laptop's microphone sits nearer one
//!    speaker, and from the mean of the two alone the best linear filter
//!    took only 6 dB of a MacBook's echo out, however long, whatever its
//!    model of the speakers' distortion (recordings of left and right
//!    apart are to show how much the two channels take). Each bin is
//!    normalised by the reference's power ([`filter::adapt`]). The
//!    estimate of the filter in use is scaled by a gain that follows what
//!    of it the microphone holds ([`gain`]: a speaker's limiter riding the
//!    music's loudness). Two filters: a shadow that learns while nobody
//!    sings, and the filter in use, which takes the shadow over only once
//!    it leaves clearly less. Singing never teaches either (a held note
//!    would draw them towards taking the note out). A room that changes
//!    while someone seems to sing (the volume turned up: the leftover jumps
//!    as a voice's would) is learnt by a trial filter apart, which takes
//!    over only once it has left far less for a second running. The
//!    filters learn along the reference prewhitened (each frequency's
//!    steady part, a held tone, taken out): learnt along a tone, the path
//!    is right for that note only. A filter that went beyond numbers, or
//!    whose estimate is far louder than all the microphone heard (no echo
//!    can be), is started over at once, never left to spoil the canceller
//!    until the player starts again; its estimate takes nothing out. One
//!    that leaves more than the microphone heard, its gain not mending
//!    that, for a while gives way to the shadow or the trial if either does
//!    better, else is started over: a speaker's limiter riding a loud
//!    passage makes a working filter overshoot for moments, and started
//!    over at once it was learning again every few seconds. Once learnt,
//!    the window follows the echo's loudest part only where the delay
//!    finder hears the echo too.
//! 3. **A residual suppressor** ([`Suppressor`], run after the canceller by
//!    its caller, given what the canceller subtracted: [`Echo`]): what the
//!    filter could not predict of the echo, taken down where it outweighs
//!    the rest, for the pitch the game hears. The canceller's own output is
//!    the microphone less the echo it predicts, nothing more taken out:
//!    what the music does not account for (the singer) passes exactly as it
//!    came.
//!
//! Of a song in mono (left and right alike) the second channel is silent
//! and its filter never moves: the canceller works as one on the mean
//! would, but for a little regularisation of the two channels' power, the
//! gain, and the delay finder.
//!
//! Each block is [`BLOCK`] samples at the canceller's rate (about 44.1 or
//! 48 kHz); the voice of a block comes out with it. [`Whole`] runs both
//! over whole signals, offline.

mod delay;
mod fft;
mod filter;
mod gain;
mod reference;
mod suppress;
#[cfg(test)]
mod tests;
mod whole;

use delay::{DelayFinder, LEARN_FIRST};
use fft::Fft;
use filter::{Along, Filter};
use gain::Gain;
use reference::{Frame, History, Whitener, CHANNELS};
pub use suppress::Suppressor;
pub use whole::{Cleaned, Whole};

/// Samples in a block.
pub const BLOCK: usize = 128;
/// The echo the filter learns, in seconds from where its window starts:
/// the way through the air and the room's tail.
pub const TAIL: f32 = 0.2;
/// The longest delay before the echo starts that the window is moved to,
/// in seconds.
pub const MAX_DELAY: f32 = 1.0;

/// Blocks after the last one sung over before the meter measures again
/// (about 100 ms).
const HOLD: u32 = 40;
/// A block leaving this many times the echo the filter in use usually
/// leaves is sung over.
const SINGING: f32 = 10.0;
/// While someone sings, the trial filter takes over only after leaving this
/// share of what the filter in use leaves (−6 dB) for this many blocks
/// running (about a second): a room that changed (the volume turned up)
/// does that soon; a voice, which the music does not predict, never does,
/// while a held note may draw the trial near it for moments.
const CLEARLY: f32 = 0.25;
const CLEARLY_FOR: u32 = 375;
/// An echo estimate over this many times the microphone's power (+20 dB)
/// in a block is no echo (the echo is part of what the microphone hears):
/// the filter that made it has gone astray. Within a block of a working
/// filter an estimate runs up to 9 dB over (a singer's sound against the
/// echo's); started over at +6 dB, a song through a laptop lost a third of
/// what was taken out. One gone astray ran 30 to 45 dB over.
const ASTRAY: f32 = 100.0;
/// Blocks between copies of one filter to the other (about 20 ms).
const SWAP_EVERY: u32 = 8;
/// The smoothing of the powers the detector and the meter use, per
/// block (about 30 ms).
const SMOOTH: f32 = 0.1;
/// Blocks running a filter in use must be worse than nothing to be dropped
/// (about 0.2 s): on a MacBook's speakers a working filter left up to
/// twice the microphone's power for moments as a loud passage began.
const WORSE_FOR: u32 = 75;
/// Blocks between looks at where the window should be (about 170 ms).
const PLACE_EVERY: u32 = 64;
/// Partitions the window starts before the echo's first sound.
const MARGIN: usize = 6;
/// Blocks the filter's loudest part and the delay finder's delay may be
/// apart and still be the same echo.
const AGREE: usize = 3;
/// A power that is nothing (silence, a block of rounding).
const QUIET: f32 = 1e-9;

/// How the canceller is doing, for a meter.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EchoStatus {
    /// How much quieter the echo is after the canceller than before, in
    /// dB, while only the reference is heard (0 before it has learnt).
    pub removed_db: f32,
    /// How much of it the filter takes out, as it measures that to tell
    /// singing from the echo it leaves, in dB.
    pub filter_db: f32,
    /// It has learnt the echo's way (the filter is converged).
    pub learnt: bool,
    /// Seconds after the reference that the echo comes, once learnt: as
    /// the delay finder hears it (else where the filter has the most).
    pub delay: Option<f32>,
}

/// What the canceller did with a block ([`EchoCanceller::process`]), for
/// the suppressor after it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Echo {
    /// The echo it subtracted.
    pub estimate: [f32; BLOCK],
    /// The block had only the echo in it, as far as the canceller can tell
    /// (learnt, the music playing, nobody singing for a while).
    pub echo_only: bool,
    /// It has learnt the echo's way.
    pub learnt: bool,
    /// Which filter the estimate is of: it changes when the canceller
    /// starts over or takes a trial filter over, and what was learnt of
    /// the filter before (what it leaves) is then no longer true.
    pub generation: u32,
}

/// The filter in use: its partitions, what it leaves (smoothed: with its
/// gain, and as it predicts without), and its gain ([`gain`]).
struct InUse {
    filter: Filter,
    error_level: f32,
    plain_level: f32,
    gain: Gain,
}

impl InUse {
    /// `filter` in use, leaving `level` as far as is known; its gain
    /// starts over.
    fn from(filter: &Filter, level: f32, fft: &Fft) -> InUse {
        let mut filter = filter.clone();
        filter::constrain(fft, &mut filter);
        InUse {
            filter,
            error_level: level,
            plain_level: level,
            gain: Gain::new(),
        }
    }

    /// A filter of `partitions` that predicts nothing, leaving the
    /// microphone's `level`.
    fn silent(partitions: usize, level: f32) -> InUse {
        InUse {
            filter: filter::silent(partitions),
            error_level: level,
            plain_level: level,
            gain: Gain::new(),
        }
    }
}

/// The trial filter, while someone seems to sing: its partitions, what it
/// leaves (smoothed), and the blocks running it has left [`CLEARLY`] less
/// than the filter in use.
struct Trial {
    filter: Filter,
    level: f32,
    better_for: u32,
}

/// One block's signals and powers, as the filters left them
/// ([`EchoCanceller::predict`]).
struct Step {
    mic: [f32; BLOCK],
    /// The reference's bins (for the delay finder) and the microphone's.
    reference: Frame,
    mic_bins: fft::Bins,
    /// What the filter in use subtracts (at its gain), and what that leaves.
    estimate: [f32; BLOCK],
    error: [f32; BLOCK],
    /// What the shadow leaves.
    shadow_error: [f32; BLOCK],
    /// Powers: the microphone's, the error's, what the filter in use
    /// leaves as it predicts (no gain), the shadow's error, its estimate's
    /// as it predicts it (no gain), the reference's.
    pm: f32,
    pe: f32,
    plain: f32,
    ps: f32,
    py: f32,
    px: f32,
    /// The filter in use's estimate, and the shadow's, has gone astray
    /// ([`ASTRAY`]).
    in_use_astray: bool,
    shadow_astray: bool,
}

/// The canceller ([`crate::aec`]).
pub struct EchoCanceller {
    fft: Fft,
    /// Its rate (samples a second), for the delay it reports.
    rate: u32,
    /// The filter in use.
    in_use: InUse,
    /// The filter that learns, and what it leaves (smoothed); blocks
    /// since one was copied to the other.
    shadow: Filter,
    shadow_level: f32,
    since_swap: u32,
    /// The trial filter, while someone seems to sing.
    trial: Option<Trial>,
    /// Bumped whenever the filter in use is no longer the one before
    /// ([`Echo::generation`]).
    generation: u32,
    /// The reference's bins, each the transform of two blocks, as far back
    /// as the window may go; where the window starts, in blocks back.
    history: History,
    offset: usize,
    /// The reference whitened (a whitener per channel), and its bins like
    /// `history`'s: what the filters learn along.
    whiteners: [Whitener; CHANNELS],
    white_history: History,

    /// Finds where the window should start.
    finder: DelayFinder,
    /// Blocks with the reference playing since the window last moved;
    /// blocks since the last look at it; the partition where the filter
    /// in use has the most.
    placed: u32,
    since_place: u32,
    peak: usize,
    /// The last reference (its channels) and microphone blocks (the
    /// transforms take two).
    last_reference: [[f32; BLOCK]; CHANNELS],
    last_mic: [f32; BLOCK],
    /// The partition whose gradient is constrained next.
    turn: usize,

    /// Smoothed block powers: microphone, echo estimate, reference.
    mic_level: f32,
    echo_level: f32,
    reference_level: f32,
    /// What is left, smoothed.
    out_level: f32,
    /// Blocks since the last one sung over; blocks running the filter in use
    /// has been worse than nothing.
    since_singing: u32,
    worse_for: u32,
    /// The last block had only the echo in it ([`Echo::echo_only`]).
    echo_only: bool,
    /// The echo removed, as the meter shows it and as the filter measures
    /// it, in dB.
    removed_db: f32,
    filter_db: f32,
    learnt: bool,
}

impl EchoCanceller {
    /// A canceller at `rate` (samples a second; it sets the window's and
    /// the delay's length in blocks).
    pub fn new(rate: u32) -> EchoCanceller {
        let blocks =
            |seconds: f32| ((seconds * rate as f32) / BLOCK as f32).ceil().max(1.0) as usize;
        let partitions = blocks(TAIL);
        let lags = blocks(MAX_DELAY);

        EchoCanceller {
            fft: Fft::new(),
            rate,
            in_use: InUse::silent(partitions, 0.0),
            shadow: filter::silent(partitions),
            shadow_level: 0.0,
            since_swap: 0,
            trial: None,
            generation: 0,
            history: History::new(lags + partitions),
            offset: 0,
            whiteners: [Whitener::new(), Whitener::new()],
            white_history: History::new(lags + partitions),

            finder: DelayFinder::new(lags),
            placed: 0,
            since_place: 0,
            peak: 0,
            last_reference: [[0.0; BLOCK]; CHANNELS],
            last_mic: [0.0; BLOCK],
            turn: 0,

            mic_level: 0.0,
            echo_level: 0.0,
            reference_level: 0.0,
            out_level: 0.0,
            since_singing: u32::MAX,
            worse_for: 0,
            echo_only: false,
            removed_db: 0.0,
            filter_db: 0.0,
            learnt: false,
        }
    }

    /// The rate it runs at (only the delay it reports changes).
    pub fn set_rate(&mut self, rate: u32) {
        self.rate = rate;
    }

    /// How it is doing.
    pub fn status(&self) -> EchoStatus {
        EchoStatus {
            removed_db: self.removed_db,
            filter_db: if self.learnt { self.filter_db } else { 0.0 },
            learnt: self.learnt,
            delay: self.learnt.then(|| {
                let blocks = self.finder.lag().unwrap_or(self.offset + self.peak);
                (blocks * BLOCK) as f32 / self.rate.max(1) as f32
            }),
        }
    }

    /// Forgets what it learnt, and where the echo was (a new output).
    pub fn reset(&mut self) {
        self.forget();
        self.history.clear();
        self.white_history.clear();
        self.finder.clear();
        self.offset = 0;
        self.placed = 0;
    }

    /// Forgets the filters (the echo's way has changed), and what they
    /// left: with nothing learnt they leave the microphone as it is, so
    /// their levels start over from its level (a level kept from before,
    /// a filter gone astray's above all, would keep the shadow from ever
    /// taking over).
    fn forget(&mut self) {
        let mic = if self.mic_level.is_finite() {
            self.mic_level
        } else {
            0.0
        };
        let partitions = self.shadow.len();
        self.mic_level = mic;
        self.in_use = InUse::silent(partitions, mic);
        filter::clear(&mut self.shadow);
        self.shadow_level = mic;
        self.trial = None;
        self.generation = self.generation.wrapping_add(1);
        self.echo_level = 0.0;
        self.since_swap = 0;
        self.learnt = false;
        self.echo_only = false;
        self.removed_db = 0.0;
        self.filter_db = 0.0;
        self.peak = 0;
    }

    /// One block: `mic` what the microphone took in, `left` and `right`
    /// what went to the speakers at the same moment (the same for one
    /// speaker); the voice without the echo goes to `out`. Gives what it
    /// subtracted, for the suppressor after it.
    pub fn process(&mut self, mic: &[f32], left: &[f32], right: &[f32], out: &mut [f32]) -> Echo {
        assert!(
            mic.len() == BLOCK && left.len() == BLOCK && right.len() == BLOCK && out.len() == BLOCK
        );
        let mut step = self.predict(mic, left, right);
        self.guard(&mut step);
        let reference_on = self.reference_level > QUIET;
        if reference_on {
            self.finder.push(&step.reference[0], &step.mic_bins);
            self.placed = self.placed.saturating_add(1);
        }
        // Double talk: once learnt, the filter in use leaves about the
        // share of the echo its own cut says (of the echo as it predicts
        // it, which singing does not change); a block leaving several
        // times that is something else in the microphone (someone
        // singing).
        let cut = 10f32.powf(self.filter_db / 10.0);
        let usual = self.echo_level.max(step.py) / cut;
        let singing = self.learnt && step.pe > SINGING * usual + 10.0 * QUIET;
        self.since_swap = self.since_swap.saturating_add(1);
        if self.since_swap > SWAP_EVERY && reference_on && !singing {
            self.swap();
        }
        if singing && reference_on {
            self.try_while_singing(&step.mic);
        } else {
            self.trial = None;
        }
        self.drop_if_worse(step.pm);
        self.measure(&step, reference_on, singing, usual);
        if reference_on && !singing {
            let along = Along {
                plain: &self.history,
                white: &self.white_history,
                offset: self.offset,
            };
            filter::adapt(
                &self.fft,
                &mut self.shadow,
                &step.shadow_error,
                &along,
                &mut self.turn,
            );
        }
        self.since_place = self.since_place.saturating_add(1);
        if self.since_place >= PLACE_EVERY {
            self.since_place = 0;
            self.place();
        }
        // The voice: the microphone less the echo predicted, and nothing
        // more taken out.
        out.copy_from_slice(&step.error);
        Echo {
            estimate: step.estimate,
            echo_only: self.echo_only,
            learnt: self.learnt,
            generation: self.generation,
        }
    }

    /// The block's signals: the reference into the histories, and what the
    /// filter in use (at its gain) and the shadow leave of the microphone.
    fn predict(&mut self, mic: &[f32], left: &[f32], right: &[f32]) -> Step {
        // Nothing but numbers goes in (a device's glitch must not stick).
        let clean = |s: &[f32], i: usize| if s[i].is_finite() { s[i] } else { 0.0 };
        let mic: [f32; BLOCK] = std::array::from_fn(|i| clean(mic, i));
        let reference: [[f32; BLOCK]; CHANNELS] = [
            std::array::from_fn(|i| 0.5 * (clean(left, i) + clean(right, i))),
            std::array::from_fn(|i| 0.5 * (clean(left, i) - clean(right, i))),
        ];
        // The reference's newest two blocks into the history, whitened
        // beside it; the microphone's for the delay finder.
        let x: Frame =
            std::array::from_fn(|c| self.fft.real(&self.last_reference[c], &reference[c]));
        let mic_bins = self.fft.real(&self.last_mic, &mic);
        self.last_reference = reference;
        self.last_mic = mic;
        self.history.push(x);
        let before = *self.history.at(2);
        let xw: Frame = std::array::from_fn(|c| self.whiteners[c].reference(&x[c], &before[c]));
        self.white_history.push(xw);
        // The echo estimates: each filter over its window, the filter in
        // use's at the loudness the speakers play it now.
        let predicted = filter::predict(&self.fft, &self.in_use.filter, &self.history, self.offset);
        let predicted = &predicted[BLOCK..];
        let mean = |s: &[f32]| s.iter().map(|v| v * v).sum::<f32>() / s.len() as f32;
        let cross = mic.iter().zip(predicted).map(|(m, y)| m * y).sum::<f32>() / BLOCK as f32;
        let sung = self.since_singing <= HOLD;
        let gain = self
            .in_use
            .gain
            .follow(cross, mean(predicted), sung, self.in_use.error_level);
        let estimate: [f32; BLOCK] = std::array::from_fn(|i| gain * predicted[i]);
        let error: [f32; BLOCK] = std::array::from_fn(|i| mic[i] - estimate[i]);
        let plain = mic
            .iter()
            .zip(predicted)
            .map(|(m, y)| (m - y) * (m - y))
            .sum::<f32>()
            / BLOCK as f32;
        let shadow = filter::predict(&self.fft, &self.shadow, &self.history, self.offset);
        let shadow_error: [f32; BLOCK] = std::array::from_fn(|i| mic[i] - shadow[BLOCK + i]);
        let pm = mean(&mic);
        let astray = |s: &[f32]| !s.iter().all(|v| v.is_finite()) || mean(s) > ASTRAY * pm + QUIET;
        Step {
            mic,
            reference: x,
            mic_bins,
            estimate,
            error,
            shadow_error,
            pm,
            pe: mean(&error),
            plain,
            ps: mean(&shadow_error),
            py: mean(predicted),
            px: mean(&reference[0]) + mean(&reference[1]),
            in_use_astray: astray(predicted),
            shadow_astray: astray(&shadow[BLOCK..]),
        }
    }

    /// An estimate beyond numbers, or far louder than all the microphone
    /// heard (the echo is part of that), is no echo: the filter that made
    /// it has gone astray (learnt from a voice over music near silence, it
    /// meets the music at full level). The filter in use's takes nothing
    /// out (the voice is never lost to it) and is started over: the
    /// shadow from the filter in use, the trial dropped, the filter in use
    /// (and everything learnt) forgotten. What the shadow left this block
    /// goes with it (it is then the filter in use's), or its next step
    /// would spoil it again at once. Then the levels of the detector and
    /// the meter (an estimate gone astray counts for nothing).
    fn guard(&mut self, step: &mut Step) {
        if step.in_use_astray {
            step.error = step.mic;
            step.estimate = [0.0; BLOCK];
            step.pe = step.pm;
            step.plain = step.pm;
            step.py = 0.0;
        }
        let smooth = |a: &mut f32, v: f32| *a += (v - *a) * SMOOTH;
        smooth(&mut self.mic_level, step.pm);
        smooth(&mut self.in_use.error_level, step.pe);
        smooth(&mut self.in_use.plain_level, step.plain);
        smooth(&mut self.echo_level, step.py);
        smooth(&mut self.reference_level, step.px);
        smooth(&mut self.shadow_level, step.ps);
        let in_use_spoilt = !(self.in_use.error_level.is_finite() && self.echo_level.is_finite());
        if step.in_use_astray || in_use_spoilt {
            self.forget();
            step.shadow_error = step.mic;
        } else if step.shadow_astray || !self.shadow_level.is_finite() {
            self.take_in_use();
            step.shadow_error = step.error;
        }
        if self.trial.as_ref().is_some_and(|t| !t.level.is_finite()) {
            self.trial = None;
        }
    }

    /// The shadow learns while the music plays and nobody sings (a held
    /// note would draw it towards taking the note out: the music has the
    /// note's frequency in it too). The filter in use takes it over once
    /// it leaves clearly less, and gives it back what it has when it has
    /// gone astray. The shadow is held against the filter in use as it
    /// predicts, without the gain (which would hold the shadow back the
    /// longer it does well).
    fn swap(&mut self) {
        if self.shadow_level < 0.7 * self.in_use.plain_level && self.shadow_level < self.mic_level {
            self.take_shadow();
        } else if self.shadow_level > 4.0 * self.in_use.error_level + QUIET {
            self.take_in_use();
        }
    }

    /// Worse than nothing for a while ([`WORSE_FOR`]; its estimate as it
    /// predicts it, and the gain not mending that: a gain near 0 is a
    /// filter that no longer fits): the filter in use is dropped. The
    /// shadow takes its place if it does better than nothing, else the
    /// trial filter if that does (while the leftover looks like singing
    /// only the trial learns); else, once learnt, the echo's way changed
    /// completely and everything starts over. Before, the filter in use
    /// never fitted it (taken over from the shadow while it fitted a voice
    /// for moments): it is dropped, and the shadow learns on (forgetting it
    /// as well cost a song through a laptop a third of what was taken out).
    fn drop_if_worse(&mut self, pm: f32) {
        let worse = self.in_use.plain_level > 2.0 * self.mic_level
            && self.in_use.error_level > 0.5 * self.mic_level;
        self.worse_for = if worse && pm > QUIET {
            self.worse_for.saturating_add(1)
        } else {
            0
        };
        if self.worse_for < WORSE_FOR {
            return;
        }
        self.worse_for = 0;
        if self.shadow_level < self.mic_level {
            self.take_shadow();
        } else if self
            .trial
            .as_ref()
            .is_some_and(|t| t.level < self.mic_level)
        {
            self.take_trial();
        } else if self.learnt {
            self.forget();
        } else {
            self.in_use = InUse::silent(self.shadow.len(), self.mic_level);
            self.echo_level = 0.0;
        }
    }

    /// The meters. The filter's own cut, while the reference plays and
    /// nobody sings over it: once learnt, only blocks whose leftover is
    /// about the `usual` (singing never is) count. Not until a while after
    /// the last block sung: the smoothed levels still carry its tail, and a
    /// cut that looks smaller makes singing look like the usual leftover.
    /// Then what is left against the microphone's level, the echo removed.
    fn measure(&mut self, step: &Step, reference_on: bool, singing: bool, usual: f32) {
        let echo_only = !self.learnt || step.pe < 4.0 * usual;
        let settled = !self.learnt || self.since_singing > HOLD;
        let heard = self.mic_level > QUIET;
        if reference_on && !singing && echo_only && settled && heard {
            let db = 10.0 * (self.mic_level / self.in_use.error_level.max(1e-12)).log10();
            self.filter_db += (db.clamp(0.0, 60.0) - self.filter_db) * SMOOTH;
            if self.filter_db > 6.0 {
                self.learnt = true;
            }
        }
        self.out_level += (step.pe - self.out_level) * SMOOTH;
        self.since_singing = if singing {
            0
        } else {
            self.since_singing.saturating_add(1)
        };
        let settled = self.since_singing > HOLD;
        self.echo_only = self.learnt && reference_on && settled && echo_only && heard;
        if reference_on && settled && echo_only && heard {
            let db = 10.0 * (self.mic_level / self.out_level.max(1e-12)).log10();
            self.removed_db += (db.clamp(0.0, 80.0) - self.removed_db) * SMOOTH;
        }
    }

    /// The filter in use takes the shadow over.
    fn take_shadow(&mut self) {
        self.in_use = InUse::from(&self.shadow, self.shadow_level, &self.fft);
        self.since_swap = 0;
    }

    /// The shadow starts again from the filter in use.
    fn take_in_use(&mut self) {
        self.shadow.clone_from(&self.in_use.filter);
        self.shadow_level = self.in_use.plain_level;
        self.since_swap = 0;
    }

    /// The filter in use and the shadow take the trial filter over; its
    /// cut is measured again.
    fn take_trial(&mut self) {
        let Some(trial) = self.trial.take() else {
            return;
        };
        self.in_use = InUse::from(&trial.filter, trial.level, &self.fft);
        self.shadow.clone_from(&self.in_use.filter);
        self.shadow_level = trial.level;
        self.generation = self.generation.wrapping_add(1);
        self.filter_db = 0.0;
        self.since_swap = 0;
    }

    /// While someone sings, a trial filter learns from where the filter
    /// in use is, apart: a room that changed (the volume turned up)
    /// looks like singing to the detector for good, and would never be
    /// learnt. The trial takes over only once it has left far less than
    /// the filter in use for a second running, which a room that changed
    /// does soon and a voice never does (the music does not predict
    /// it). It starts from the filter in use at its gain, and is held
    /// against it as it predicts without the gain: the gain mends a louder
    /// room's level at once, a little of it, and against that the trial
    /// would wait for the room to settle. It is dropped when the singing
    /// stops.
    fn try_while_singing(&mut self, mic: &[f32; BLOCK]) {
        let in_use = &self.in_use;
        let trial = self.trial.get_or_insert_with(|| {
            let mut filter = in_use.filter.clone();
            filter::scale(&mut filter, in_use.gain.now());
            Trial {
                filter,
                level: in_use.error_level,
                better_for: 0,
            }
        });
        let predicted = filter::predict(&self.fft, &trial.filter, &self.history, self.offset);
        let error: [f32; BLOCK] = std::array::from_fn(|i| mic[i] - predicted[BLOCK + i]);
        let p = error.iter().map(|v| v * v).sum::<f32>() / BLOCK as f32;
        trial.level += (p - trial.level) * SMOOTH;
        trial.better_for = if trial.level < CLEARLY * self.in_use.plain_level {
            trial.better_for.saturating_add(1)
        } else {
            0
        };
        if trial.better_for >= CLEARLY_FOR && trial.level < self.mic_level {
            self.take_trial();
        } else {
            let along = Along {
                plain: &self.history,
                white: &self.white_history,
                offset: self.offset,
            };
            filter::adapt(&self.fft, &mut trial.filter, &error, &along, &mut self.turn);
        }
    }

    /// Where the window should start. Once learnt, the echo's loudest part
    /// is kept near its start (so the room's tail fits after it), where the
    /// delay finder hears the echo too: a filter that learnt something
    /// else has its most anywhere (one taken over from the trial on a
    /// laptop's distorted echo had it at the window's end), and the window
    /// moved there left the echo behind it for good, the delay it told
    /// jumping from 30 to 190 ms. Before, the delay finder's delay, if the
    /// window does not reach it.
    fn place(&mut self) {
        let partitions = self.shadow.len();
        let furthest = self.history.len() - partitions;
        self.peak = filter::peak(&self.in_use.filter);
        self.finder.look();
        let to = if self.learnt {
            let at = self.offset + self.peak;
            let heard = self
                .finder
                .lag()
                .is_some_and(|lag| lag.abs_diff(at) <= AGREE);
            if self.peak <= partitions / 3 || !heard {
                return;
            }
            at - MARGIN
        } else {
            if self.placed < LEARN_FIRST {
                return;
            }
            let Some(lag) = self.finder.lag() else {
                return;
            };
            if lag >= self.offset && lag <= self.offset + partitions / 2 {
                return;
            }
            lag.saturating_sub(MARGIN)
        };
        self.move_to(to.min(furthest));
    }

    /// Moves the window to start `to` blocks back, the filters with it
    /// (what they learnt of the delays both windows hold is kept).
    fn move_to(&mut self, to: usize) {
        if to == self.offset {
            return;
        }
        let partitions = self.shadow.len();
        for w in [&mut self.in_use.filter, &mut self.shadow] {
            if to > self.offset {
                let by = (to - self.offset).min(partitions);
                w.rotate_left(by);
                w[partitions - by..].fill(reference::SILENT);
            } else {
                let by = (self.offset - to).min(partitions);
                w.rotate_right(by);
                w[..by].fill(reference::SILENT);
            }
        }
        self.peak = self.peak.saturating_sub(to.saturating_sub(self.offset));
        self.offset = to;
        self.placed = 0;
    }
}
