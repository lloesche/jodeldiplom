//! How late the singer's voice comes, one value for everything that holds
//! the voice against the song (the scoring, what Keep records): the time
//! from a frame of the music going to the device to the microphone hearing
//! the singer sing it (the output's buffers, the ear, the microphone's).
//!
//! It is the first of these there is ([`resolve`]):
//!
//! 1. the delay **measured** for the output and the microphone in use (a
//!    sync, [`crate::core::sync`]);
//! 2. the echo canceller's delay, once it is **steady** ([`Steady`]: learnt
//!    and within [`STEADY_WITHIN`] of itself for [`STEADY_FOR`]; with a
//!    headset there is no echo, and it never is);
//! 3. the delay **estimated from the singing** ([`estimate`]);
//! 4. [`DEFAULT_LATE`].
//!
//! A measured or echo delay is still corrected by the singing: where the
//! estimate from the singing is within [`CORRECT_WITHIN`] of it, the
//! estimate is taken ([`correction`]; a sync's clicks are answered a little
//! sooner than words are sung). The estimate is held against that delay
//! before it is taken: it must fit the singing clearly better
//! ([`estimate::Estimator::anchor`]), so silence or a few windows never
//! move it. The singer's nudge is added to any of them, and wins over the
//! correction: a delay nudged (even to 0) is not corrected ([`target`]).
//! [`Lag::settle`] is the one step of all this. What is kept of it, a pair
//! of an output and a microphone each ([`Delays`]: the delay measured and
//! the nudge), is here too; `crate::delays` keeps it on disk. Another pair
//! starts the echo's wait and the estimate over: they were of the other.
//!
//! Mid-song the value in force moves at most [`SLEW`] a second towards a
//! new one, so the scoring never jumps; a nudge or a sync is taken at once,
//! and so is anything while no song is sung ([`Lag::step`]).

pub mod estimate;

use std::collections::HashMap;

use crate::core::State;

/// The delay taken when nothing better is known, in seconds: what a USB
/// headset's singing fitted best.
pub const DEFAULT_LATE: f64 = 0.15;
/// The furthest the delay in force moves in a second of a song, in seconds.
pub const SLEW: f64 = 0.02;
/// How close to itself the echo canceller's delay stays to be steady, each
/// way, in seconds.
pub const STEADY_WITHIN: f64 = 0.010;
/// How long it stays there, in seconds.
pub const STEADY_FOR: f64 = 3.0;
/// How far from a measured or echo delay the estimate from the singing may
/// be and still correct it, each way, in seconds: further off, one of the
/// two is wrong, and the one measured stands.
pub const CORRECT_WITHIN: f64 = 0.06;
/// The least nudge on a delay, in ms.
pub const NUDGE_LEAST: i32 = -200;
/// The most.
pub const NUDGE_MOST: i32 = 400;
/// Its step.
pub const NUDGE_STEP: i32 = 10;

/// An output device and a microphone, by their ids.
pub type Pair = (String, String);

/// What is kept of the delay for a pair: the one measured (none yet) and
/// the nudge on whatever delay is in force, in ms (none: not nudged, the
/// singing may correct it; 0 is a nudge, the delay as it is).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Delay {
    /// Measured by a sync.
    pub measured_ms: Option<i32>,
    /// The singer's nudge.
    pub nudge_ms: Option<i32>,
}

/// What is kept of the delay, by pair.
pub type Delays = HashMap<Pair, Delay>;

/// Where the delay in force comes from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Source {
    /// Measured by a sync.
    Measured,
    /// The echo canceller's, steady.
    Echo,
    /// Estimated from the singing.
    Singing,
    /// [`DEFAULT_LATE`].
    #[default]
    Default,
}

impl Source {
    /// As the Karaoke tab says it.
    pub fn name(self) -> &'static str {
        match self {
            Source::Measured => "measured",
            Source::Echo => "from the echo",
            Source::Singing => "from your singing",
            Source::Default => "default",
        }
    }

    /// As a kept singing's `about.txt` says it: one word.
    pub fn key(self) -> &'static str {
        match self {
            Source::Measured => "measured",
            Source::Echo => "echo",
            Source::Singing => "singing",
            Source::Default => "default",
        }
    }
}

/// The echo canceller's delay, held until it is steady.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Steady {
    /// The value it has stayed near, and since when (seconds).
    anchor: Option<(f64, f64)>,
}

impl Steady {
    /// The canceller read at `now` seconds: whether it has learnt the
    /// echo's way, and the delay it found (none: none, or no canceller
    /// running).
    pub fn hear(&mut self, echo: Option<(bool, Option<f64>)>, now: f64) {
        let Some((true, Some(delay))) = echo else {
            self.anchor = None;
            return;
        };
        match self.anchor {
            Some((at, _)) if (delay - at).abs() <= STEADY_WITHIN => {}
            _ => self.anchor = Some((delay, now)),
        }
    }

    /// The delay it has stayed near, once it has been steady for
    /// [`STEADY_FOR`] at `now` (the same while it stays: what the singing
    /// corrects holds still).
    pub fn steady(&self, now: f64) -> Option<f64> {
        self.anchor
            .filter(|&(_, since)| now - since >= STEADY_FOR)
            .map(|(at, _)| at)
    }
}

/// The delay before the nudge and where it comes from: the first there is
/// of a measured one, a steady echo, one estimated from the singing, and
/// the default.
pub fn resolve(measured: Option<f64>, echo: Option<f64>, sung: Option<f64>) -> (f64, Source) {
    measured
        .map(|m| (m, Source::Measured))
        .or(echo.map(|e| (e, Source::Echo)))
        .or(sung.map(|s| (s, Source::Singing)))
        .unwrap_or((DEFAULT_LATE, Source::Default))
}

/// How far the singing's estimate `sung` moves `base` (from `source`), in
/// seconds: to it, where `base` was measured or is the echo's and `sung`
/// is within [`CORRECT_WITHIN`] of it; else not at all.
pub fn correction(base: f64, source: Source, sung: Option<f64>) -> f64 {
    match (source, sung) {
        (Source::Measured | Source::Echo, Some(s)) if (s - base).abs() <= CORRECT_WITHIN + 1e-9 => {
            s - base
        }
        _ => 0.0,
    }
}

/// The delay to take and its parts ([`target`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Target {
    /// How late the voice comes, in seconds: all of it.
    pub late: f64,
    /// The first there is ([`resolve`]), in seconds.
    pub base: f64,
    /// The singing's correction on it ([`correction`]), in seconds.
    pub corrected: f64,
    /// Where `base` comes from.
    pub source: Source,
}

/// The delay to take from the one `measured`, the `echo`'s once steady,
/// the one estimated from the singing (`sung`) and the singer's `nudge`
/// (seconds; none: not nudged): the first there is, the singing's
/// correction on a measured or echo one, the nudge on top. A nudge wins
/// over the correction: with one, even of 0, there is none (the singer set
/// the delay by ear).
pub fn target(
    measured: Option<f64>,
    echo: Option<f64>,
    sung: Option<f64>,
    nudge: Option<f64>,
) -> Target {
    let (base, source) = resolve(measured, echo, sung);
    let corrected = match nudge {
        None => correction(base, source, sung),
        Some(_) => 0.0,
    };
    Target {
        late: base + corrected + nudge.unwrap_or(0.0),
        base,
        corrected,
        source,
    }
}

/// What the delay in force follows at a moment ([`Lag::settle`]): the
/// delay measured, the echo's once steady, the nudge (seconds; none: none).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Known {
    /// Measured by a sync.
    pub measured: Option<f64>,
    /// The echo canceller's, steady.
    pub echo: Option<f64>,
    /// The singer's nudge.
    pub nudge: Option<f64>,
}

/// The singer's delay: what is known of it, and the one in force.
#[derive(Clone, Debug)]
pub struct Lag {
    /// How late the voice comes, in seconds: what the scoring takes.
    pub late: f64,
    /// Where the value it goes to comes from.
    pub source: Source,
    /// That value before the singing's correction and the nudge, in
    /// seconds.
    pub base: f64,
    /// How far the singing corrected a measured or echo delay in it, in
    /// seconds ([`correction`]).
    pub corrected: f64,
    /// The nudge in it, in seconds (none: none).
    pub nudge: Option<f64>,
    /// The echo canceller's delay, until steady.
    pub echo: Steady,
    /// The delay estimated from the singing.
    pub estimate: estimate::Estimator,
    /// What is kept of the delay, by pair.
    pub kept: Delays,
    /// The output and microphone in use (the microphone's id empty with
    /// none open).
    pub pair: Pair,
    /// Whether what is kept changed since it was last saved.
    pub changed: bool,
    /// The measured delay and the nudge it was last given, in ms: a
    /// change of either is taken at once.
    given: Option<(Option<i32>, Option<i32>)>,
    /// When it last stepped, in seconds.
    at: Option<f64>,
    /// What it last followed ([`Lag::settle`]).
    known: Known,
}

/// The singing's own time at `heard` seconds into the song as heard, the
/// voice coming `late`: what its bubbles' sweep and burst, its sparks and
/// the singer's dot go by.
pub fn sung_clock(heard: f64, late: f64) -> f64 {
    heard - late
}

impl Default for Lag {
    fn default() -> Self {
        Lag {
            late: DEFAULT_LATE,
            source: Source::Default,
            base: DEFAULT_LATE,
            corrected: 0.0,
            nudge: None,
            echo: Steady::default(),
            estimate: estimate::Estimator::default(),
            kept: Delays::new(),
            pair: Pair::default(),
            changed: false,
            given: None,
            at: None,
            known: Known::default(),
        }
    }
}

impl Lag {
    /// A step of the clock to `now` seconds towards `target` from `source`:
    /// at once when `jump` (or the first time), else by at most [`SLEW`] a
    /// second since the last step.
    pub fn step(&mut self, target: f64, source: Source, now: f64, jump: bool) {
        let since = self.at.map_or(f64::INFINITY, |at| (now - at).max(0.0));
        self.at = Some(now);
        self.source = source;
        if jump || since.is_infinite() {
            self.late = target;
            return;
        }
        let most = SLEW * since;
        self.late += (target - self.late).clamp(-most, most);
    }

    /// The delay in force followed at `now` from what is `known`: the
    /// estimate held against the delay it would correct
    /// ([`estimate::Estimator::anchor`]), the [`target`] of it all, and a
    /// [`Lag::step`] towards it (at once when `jump`). The game, the tests
    /// and `sing_replay` all go by this.
    pub fn settle(&mut self, known: Known, now: f64, jump: bool) -> Target {
        self.known = known;
        self.estimate.anchor(known.measured.or(known.echo));
        let t = target(known.measured, known.echo, self.estimate.lag, known.nudge);
        (self.base, self.corrected, self.nudge) = (t.base, t.corrected, known.nudge);
        self.step(t.late, t.source, now, jump);
        t
    }

    /// The delay the scoring goes to now, in seconds, as the Delay slider
    /// shows it: the [`target`] of what is known, with the nudge as it is
    /// set now (`nudge_ms`), not as last followed.
    pub fn in_force(&self, nudge_ms: Option<i32>) -> f64 {
        let k = self.known;
        let nudge = nudge_ms.map(|ms| f64::from(ms) / 1000.0);
        target(k.measured, k.echo, self.estimate.lag, nudge).late
    }

    /// The singing's own time at `heard` seconds into the song as heard:
    /// its frames of a moment come [`Lag::late`] after it ([`sung_clock`]).
    pub fn sung_clock(&self, heard: f64) -> f64 {
        sung_clock(heard, self.late)
    }
}

/// The output and microphone now in use: another pair than before starts
/// the echo's wait and the estimate over (they were the other pair's).
pub fn set_pair(state: &mut State, output: &str, mic: &str) {
    let lag = &mut state.karaoke.lag;
    if lag.pair.0 == output && lag.pair.1 == mic {
        return;
    }
    lag.pair = (output.to_owned(), mic.to_owned());
    lag.echo = Steady::default();
    lag.estimate = estimate::Estimator::default();
}

/// The delay measured for the output and microphone in use, in seconds.
pub fn measured(state: &State) -> Option<f64> {
    let lag = &state.karaoke.lag;
    lag.kept
        .get(&lag.pair)
        .and_then(|d| d.measured_ms)
        .map(|ms| f64::from(ms) / 1000.0)
}

/// The nudge for the output and microphone in use, in ms (none: not
/// nudged).
pub fn nudged(state: &State) -> Option<i32> {
    let lag = &state.karaoke.lag;
    lag.kept.get(&lag.pair).and_then(|d| d.nudge_ms)
}

/// [`nudged`], 0 for none.
pub fn nudge_ms(state: &State) -> i32 {
    nudged(state).unwrap_or(0)
}

/// Sets the nudge for the output and microphone in use (kept), within
/// [`NUDGE_LEAST`] to [`NUDGE_MOST`].
pub fn set_nudge(state: &mut State, ms: i32) {
    let ms = ms.clamp(NUDGE_LEAST, NUDGE_MOST);
    if nudged(state) == Some(ms) {
        return;
    }
    let lag = &mut state.karaoke.lag;
    lag.kept.entry(lag.pair.clone()).or_default().nudge_ms = Some(ms);
    lag.changed = true;
}

/// The nudge for the output and microphone in use gone: the delay not
/// nudged (the singing may correct it) again (kept).
pub fn unset_nudge(state: &mut State) {
    if nudged(state).is_none() {
        return;
    }
    let lag = &mut state.karaoke.lag;
    lag.kept.entry(lag.pair.clone()).or_default().nudge_ms = None;
    lag.changed = true;
}

/// Keeps `ms` as the delay measured for `pair`, its nudge gone.
pub fn keep_measured(state: &mut State, pair: &Pair, ms: i32) {
    let lag = &mut state.karaoke.lag;
    lag.kept.insert(
        pair.clone(),
        Delay {
            measured_ms: Some(ms),
            nudge_ms: None,
        },
    );
    lag.changed = true;
}

/// The singer's delay followed at `now` seconds: `echo` is the echo
/// canceller as read (whether it has learnt the echo's way, its delay),
/// none without one running. Mid-song it moves smoothly
/// ([`Lag::step`]); a nudge or a sync is taken at once.
pub fn follow(state: &mut State, echo: Option<(bool, Option<f64>)>, now: f64) {
    let measured = measured(state);
    let nudge_ms = nudged(state);
    let singing = crate::core::karaoke::in_mode(state)
        && state.playing.state == jodel_playback::PlayState::Playing;
    let lag = &mut state.karaoke.lag;
    lag.echo.hear(echo, now);
    let known = Known {
        measured,
        echo: lag.echo.steady(now),
        nudge: nudge_ms.map(|ms| f64::from(ms) / 1000.0),
    };
    let given = (measured.map(|m| (m * 1000.0).round() as i32), nudge_ms);
    let jump = !singing || lag.given != Some(given);
    lag.given = Some(given);
    lag.settle(known, now, jump);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_delay_is_the_first_there_is_measured_echo_singing_default() {
        assert_eq!(
            resolve(Some(0.12), Some(0.2), Some(0.3)),
            (0.12, Source::Measured)
        );
        assert_eq!(resolve(None, Some(0.2), Some(0.3)), (0.2, Source::Echo));
        assert_eq!(resolve(None, None, Some(0.3)), (0.3, Source::Singing));
        assert_eq!(resolve(None, None, None), (DEFAULT_LATE, Source::Default));
    }

    #[test]
    fn the_singing_corrects_a_measured_or_echo_delay_near_it() {
        // 36 ms short: corrected; 80 ms off: one of them is wrong, none.
        assert!((correction(0.264, Source::Measured, Some(0.3)) - 0.036).abs() < 1e-9);
        assert!((correction(0.08, Source::Echo, Some(0.05)) + 0.03).abs() < 1e-9);
        assert_eq!(correction(0.12, Source::Measured, Some(0.2)), 0.0);
        assert_eq!(correction(0.264, Source::Measured, None), 0.0);
        // The estimate itself, or the default: nothing to correct.
        assert_eq!(correction(0.2, Source::Singing, Some(0.2)), 0.0);
        assert_eq!(correction(DEFAULT_LATE, Source::Default, Some(0.18)), 0.0);
    }

    #[test]
    fn only_a_steady_echo_is_taken() {
        let mut s = Steady::default();
        // Learnt, near 80 ms, for 3 s (a frame every 20 ms): steady.
        for i in 0..=150 {
            let t = f64::from(i) * 0.02;
            s.hear(Some((true, Some(0.08 + 0.004 * (t * 7.0).sin()))), t);
        }
        assert!(s.steady(3.0).is_some_and(|d| (d - 0.08).abs() < 0.005));
        // One that jumps (a headset's, learnt from leakage now and then) is
        // never taken.
        let mut j = Steady::default();
        for i in 0..=500 {
            let t = f64::from(i) * 0.02;
            let d = if (i / 60) % 2 == 0 { 0.16 } else { 0.34 };
            j.hear(Some((true, Some(d))), t);
            assert_eq!(j.steady(t), None, "at {t}");
        }
        // Not learnt, or none: not taken, and the wait starts over.
        s.hear(Some((false, Some(0.08))), 3.1);
        assert_eq!(s.steady(3.1), None);
        s.hear(Some((true, Some(0.08))), 3.2);
        assert_eq!(s.steady(6.1), None);
        assert!(s.steady(6.2).is_some());
        s.hear(None, 6.3);
        assert_eq!(s.steady(6.3), None);
    }

    #[test]
    fn mid_song_the_delay_moves_smoothly_and_a_jump_is_taken_at_once() {
        let mut lag = Lag::default();
        lag.step(0.15, Source::Default, 0.0, false);
        assert_eq!(lag.late, 0.15);
        // A new value mid-song: no frame moves it more than the slew.
        let mut was = lag.late;
        for i in 1..=300 {
            let t = f64::from(i) / 60.0;
            lag.step(0.35, Source::Singing, t, false);
            assert!((lag.late - was).abs() <= SLEW / 60.0 + 1e-12, "at {t}");
            was = lag.late;
        }
        // 5 s at 20 ms a second: 100 ms of the 200.
        assert!((lag.late - 0.25).abs() < 1e-9, "{}", lag.late);
        // A nudge or a sync: at once.
        lag.step(0.12, Source::Measured, 5.1, true);
        assert_eq!((lag.late, lag.source), (0.12, Source::Measured));
    }
}
