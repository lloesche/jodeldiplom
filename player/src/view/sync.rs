//! The sync on the Karaoke tab ([`crate::core::sync`]) as the window
//! shows it: where the delay in force comes from ([`source_text`], said on
//! the tab's sync row), the keys while a sync shows ([`keys`]), and its
//! panel over the tab ([`Panel`]): what to do, the clicks as dots lighting
//! up as they play (in the accent once answered), how many were heard,
//! and at the end the delay measured (or why there is none), with its
//! buttons.

use std::fmt;

use crate::core::keymap::{self, Context, Op};
use crate::core::sync::{Outcome, CLICKS};
use crate::core::text::signed_ms;
use crate::core::State;
use crate::view::karaoke::Hint;

/// The panel's title.
pub const TITLE: &str = "Sync: how late your voice comes";
/// What to do.
pub const SAY: &str = "Say “ta” (or clap) with each click.";
/// Why.
pub const HEARS: &str = "The microphone hears you: this is the delay the scoring sees.";

/// Where the delay in force comes from as the tab says it, with the
/// singing's correction and the nudge on it where there are: `from your
/// singing`, `measured · your singing +30 ms · nudged −20 ms`.
pub fn source_text(state: &State) -> String {
    let lag = &state.karaoke.lag;
    let corrected = (lag.corrected * 1000.0).round() as i32;
    let corrected = if corrected == 0 {
        String::new()
    } else {
        format!(" · your singing {}", signed_ms(corrected))
    };
    let nudged = match lag.nudge {
        Some(n) => format!(" · nudged {}", signed_ms((n * 1000.0).round() as i32)),
        None => String::new(),
    };
    format!("{}{corrected}{nudged}", lag.source.name())
}

/// The keys while a sync shows, for the pane's foot: keep only once there
/// is a delay to keep, again once it has ended.
pub fn keys(state: &State) -> Vec<Hint> {
    let key = |op, what| Hint::new(keymap::hint(Context::Sync, op), what);
    match state.karaoke.sync.as_ref().and_then(|s| s.outcome.as_ref()) {
        None => vec![
            Hint::new("", "say “ta” with each click"),
            key(Op::Back, "cancel"),
        ],
        Some(Outcome::Measured(_)) => vec![
            key(Op::SyncKeep, "keep"),
            key(Op::SyncAgain, "again"),
            key(Op::Back, "cancel"),
        ],
        Some(_) => vec![key(Op::SyncAgain, "again"), key(Op::Back, "cancel")],
    }
}

/// A click, as its dot shows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dot {
    /// Still to play.
    Coming,
    /// Played, no answer heard.
    Played,
    /// Played and answered.
    Answered,
}

/// How a sync ended, as the panel says it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Said {
    /// The delay measured, in ms.
    Measured(i32),
    /// Too few answers agreed: again?
    TooFew(String),
    /// The clicks stopped.
    Failed(String),
}

/// A button of the panel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PanelButton {
    /// What it says.
    pub label: &'static str,
    /// Its key, from the keymap.
    pub key: String,
    /// What it does.
    pub op: Op,
    /// The one to take (keep).
    pub primary: bool,
}

impl PanelButton {
    /// The id it is found by in the window.
    pub fn id(&self) -> &'static str {
        match self.op {
            Op::SyncKeep => "sync-keep",
            Op::SyncAgain => "sync-again",
            _ => "sync-cancel",
        }
    }
}

/// The sync's panel as it is now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Panel {
    /// The clicks.
    pub dots: Vec<Dot>,
    /// How many were answered, of those played: `2 of 3 heard`.
    pub heard: String,
    /// How it ended, once it has.
    pub said: Option<Said>,
    /// Its buttons: keep, again, cancel, as the sync stands.
    pub buttons: Vec<PanelButton>,
}

impl Panel {
    /// The panel, while a sync shows.
    pub fn of(state: &State) -> Option<Panel> {
        let session = state.karaoke.sync.as_ref()?;
        let dots = (0..CLICKS)
            .map(|k| {
                if k >= session.played {
                    Dot::Coming
                } else if session.answers.get(k).copied().flatten().is_some() {
                    Dot::Answered
                } else {
                    Dot::Played
                }
            })
            .collect();
        let said = session.outcome.as_ref().map(|o| match o {
            Outcome::Measured(d) => Said::Measured((d * 1000.0).round() as i32),
            Outcome::TooFew(n) => Said::TooFew(format!(
                "Didn't hear enough ({n} of {CLICKS} together): again?"
            )),
            Outcome::Failed(why) => Said::Failed(format!("The clicks stopped: {why}")),
        });
        let buttons = session
            .buttons()
            .into_iter()
            .map(|(label, op)| PanelButton {
                label,
                key: keymap::hint(Context::Sync, op),
                op,
                primary: op == Op::SyncKeep,
            })
            .collect();
        Some(Panel {
            dots,
            heard: format!("{} of {} heard", session.heard(), session.played),
            said,
            buttons,
        })
    }
}

/// An outline of the panel, one line a piece with its tone in brackets:
/// for the snapshots.
impl fmt::Display for Panel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "[panel] {TITLE}")?;
        writeln!(f, "[bold] {SAY}")?;
        writeln!(f, "[dim] {HEARS}")?;
        let dots: String = self
            .dots
            .iter()
            .map(|d| match d {
                Dot::Coming => '○',
                Dot::Played => '●',
                Dot::Answered => '◉',
            })
            .collect();
        writeln!(f, "[dots] {dots}  [dim] {}", self.heard)?;
        match &self.said {
            None => {}
            Some(Said::Measured(ms)) => writeln!(f, "[accent] Your delay: {ms} ms")?,
            Some(Said::TooFew(text)) => writeln!(f, "[warn] {text}")?,
            Some(Said::Failed(text)) => writeln!(f, "[fail] {text}")?,
        }
        for b in &self.buttons {
            let kind = if b.primary { "primary" } else { "button" };
            writeln!(f, "[{kind}] {} ({})", b.label, b.key)?;
        }
        Ok(())
    }
}
