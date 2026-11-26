//! Syncing (latency calibration): measuring how late the singer's voice
//! comes, as rhythm games
//! do. Started from the Karaoke tab (`s`) with a microphone open, it stops
//! the song, plays [`CLICKS`] clicks at [`BPM`] after [`LEAD_IN`] of
//! silence through the output, and listens: the singer says "ta" (or
//! claps) with each click. The start of each sound ([`Onsets`]) is told in
//! the stream frames of the clicks' own output, as the voice is when it is
//! scored, so its offset from its click is the whole way the scoring sees:
//! the output, the ear and the microphone.
//!
//! Each click's answer is the strongest sound within [`EARLIEST`] to
//! [`LATEST`] of it ([`answers`]); the delay is the median of those within
//! [`AGREE`] of the median of them all, once [`ENOUGH`] agree
//! ([`outcome`]). Kept (`enter`), it is the delay measured for the output
//! and microphone the sync was started with ([`crate::core::lag`]), the
//! nudge on it set back to 0.
//!
//! The app plays the clicks and hears the answers (`crate::sync`); here is
//! what they mean.

mod clicks;
mod onsets;

pub use clicks::{
    answers, click_frames, click_time, length, outcome, Outcome, AGREE, BPM, CLICKS, EARLIEST,
    ENOUGH, LATEST, LEAD_IN, TAIL,
};
pub use onsets::Onsets;

use jodel_playback::{Command, PlayState};

use crate::core::desk::StripKind;
use crate::core::keymap::Op;
use crate::core::lag::{self, Pair};
use crate::core::{Effect, Side, State};

/// What the app tells of the sync under way.
#[derive(Clone, Debug, PartialEq)]
pub enum SyncEvent {
    /// The clicks heard so far, and each click's answer so far (seconds
    /// after it; none: none yet).
    Progress {
        /// The clicks heard.
        played: usize,
        /// The answers.
        answers: Vec<Option<f64>>,
    },
    /// Every click played and listened to.
    Ended,
    /// The clicks could not be played, or were stopped.
    Failed(String),
}

/// What the app is to do for a sync.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyncCommand {
    /// Play the clicks and listen (any sync under way is called off).
    Start,
    /// Stop.
    Stop,
}

/// A sync under way, or ended and shown.
#[derive(Clone, Debug, PartialEq)]
pub struct Session {
    /// The output and microphone it measures (what it is kept for, should
    /// either change meanwhile).
    pub pair: Pair,
    /// The clicks heard so far.
    pub played: usize,
    /// Each click's answer so far, in seconds after it.
    pub answers: Vec<Option<f64>>,
    /// How it ended, once it has.
    pub outcome: Option<Outcome>,
}

impl Session {
    /// One starting for `pair`.
    pub fn new(pair: Pair) -> Session {
        Session {
            pair,
            played: 0,
            answers: vec![None; CLICKS],
            outcome: None,
        }
    }

    /// The clicks answered so far.
    pub fn heard(&self) -> usize {
        self.answers.iter().flatten().count()
    }

    /// The panel's buttons now, each its label and what it does (its key is
    /// the keymap's):
    /// keep once a delay is measured, again once it has ended, cancel
    /// always.
    pub fn buttons(&self) -> Vec<(&'static str, Op)> {
        let mut out = Vec::new();
        if matches!(self.outcome, Some(Outcome::Measured(_))) {
            out.push(("Keep", Op::SyncKeep));
        }
        if self.outcome.is_some() {
            out.push(("Again", Op::SyncAgain));
        }
        out.push(("Cancel", Op::Back));
        out
    }
}

/// How far down the Player's strip may be on the Mixer tab for the clicks
/// to be heard, in dB.
pub const QUIET_DB: f64 = -30.0;

/// Why the clicks would not be heard, if they would not: they play on the
/// Player's strip (an input of its own, beside the song's), so a strip
/// muted, turned well down, or another soloed on the Mixer tab silences them.
/// A stem's strip soloed is another: solo in place along the path silences
/// its group's own inputs (unless the player's strip is soloed too), and a
/// stem muted or turned down does nothing to them.
fn unheard(state: &State) -> Option<&'static str> {
    let player = state.desk.setting(StripKind::Player);
    let other_solo = StripKind::ALL
        .iter()
        .any(|&k| k != StripKind::Player && state.desk.setting(k).solo);
    if player.mute {
        Some("the player's strip is muted on the Mixer tab")
    } else if other_solo && !player.solo {
        Some("another strip is soloed on the Mixer tab")
    } else if player.db < QUIET_DB {
        Some("the player's fader is far down on the Mixer tab")
    } else {
        None
    }
}

/// Starts a sync (`s` on the Karaoke tab) for the output and microphone in
/// use: the song stopped, the clicks played. Not without a microphone open
/// (it is the microphone's delay that is measured), nor where the clicks
/// would not be heard: it says why instead.
pub(crate) fn begin(state: &mut State, fx: &mut Vec<Effect>) {
    let pair = state.karaoke.lag.pair.clone();
    if state.karaoke.mic.name.is_none() || pair.1.is_empty() {
        let why = "A sync needs a microphone to hear you, and none is open (i: choose one)";
        return state.info(why);
    }
    if let Some(why) = unheard(state) {
        return state.info(format!(
            "No sync: the clicks play on the player's strip, and {why}"
        ));
    }
    if state.playing.state != PlayState::Stopped {
        fx.push(Effect::Engine(Command::Stop));
    }
    state.karaoke.sync = Some(Session::new(pair));
    fx.push(Effect::Sync(SyncCommand::Start));
}

/// Carries out `op` while a sync shows: keep (`enter`), again (`r`),
/// cancel (`esc`); quitting and the device's volume as anywhere; nothing
/// else. `false` with none shown.
pub(crate) fn apply(state: &mut State, op: Op, fx: &mut Vec<Effect>) -> bool {
    if state.karaoke.sync.is_none() {
        return false;
    }
    match op {
        Op::SyncKeep | Op::Accept | Op::Open => keep(state),
        Op::SyncAgain => {
            stop(state, fx);
            begin(state, fx);
        }
        Op::Back => stop(state, fx),
        Op::Quit | Op::VolumeUp | Op::VolumeDown => return false,
        _ => {}
    }
    true
}

/// The sync called off.
fn stop(state: &mut State, fx: &mut Vec<Effect>) {
    state.karaoke.sync = None;
    fx.push(Effect::Sync(SyncCommand::Stop));
}

/// The delay measured kept for the output and microphone the sync was
/// started with, the nudge back to 0; nothing before one is measured.
fn keep(state: &mut State) {
    let Some(session) = &state.karaoke.sync else {
        return;
    };
    let Some(Outcome::Measured(delay)) = session.outcome else {
        return;
    };
    let pair = session.pair.clone();
    let ms = (delay * 1000.0).round() as i32;
    lag::keep_measured(state, &pair, ms);
    state.karaoke.sync = None;
    let what = if pair == state.karaoke.lag.pair {
        "for this output and microphone"
    } else {
        "for the output and microphone it was measured with"
    };
    state.info(format!("Delay kept: {ms} ms {what}"));
}

/// What the app tells of the sync under way.
pub(crate) fn event(state: &mut State, event: SyncEvent) {
    let Some(session) = &mut state.karaoke.sync else {
        return;
    };
    if session.outcome.is_some() {
        return;
    }
    match event {
        SyncEvent::Progress { played, answers } => {
            session.played = played.min(CLICKS);
            if answers.len() == CLICKS {
                session.answers = answers;
            }
        }
        SyncEvent::Ended => {
            session.played = CLICKS;
            session.outcome = Some(outcome(&session.answers));
        }
        SyncEvent::Failed(why) => session.outcome = Some(Outcome::Failed(why)),
    }
}

/// After any action: a sync is called off once the Karaoke tab is left.
pub(crate) fn follow(state: &mut State, fx: &mut Vec<Effect>) {
    if state.karaoke.sync.is_some() && !state.showing(Side::Karaoke) {
        stop(state, fx);
    }
}
