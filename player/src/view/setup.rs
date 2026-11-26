//! The setup screen ([`crate::setup`]) as it reads: what the player does
//! before it can be used, step by step (the files it gets, then its own
//! start), how far the step under way is and how far the whole; on a
//! failure, what went wrong, what to do, and two choices: Try again and
//! Quit (there is no skipping: without its files the player is not what
//! it should be). It is the first thing the window shows at every start;
//! with the files there, only the player's own start's steps show. The
//! window draws it (`gui::screens::setup`).

use std::fmt;

use crate::setup::{Status, StepState, Unit};

/// The screen's title.
pub const TITLE: &str = "Getting the player ready";

/// What setup is, under the title, when it gets files.
pub const INTRO: &str = "This happens once. The player needs these files and checks each \
                         against a checksum built into it. Once they are here it never needs \
                         the network for them again.";

/// What shows under the title when only the player's own start is to come.
pub const STARTING: &str = "A moment while the player starts.";

/// What shows while the files are looked for.
pub const LOOKING: &str = "Looking for the graphics card and the files…";

/// The whole's label.
pub const WHOLE: &str = "All of it";

/// What quitting keeps, while files are got.
pub const KEPT: &str = "Quitting keeps what is downloaded whole for the next start.";

/// What the person chose on the setup screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Choice {
    /// Try again, after a failure.
    Retry,
    /// Quit the player.
    Quit,
}

/// What a key means on the setup screen: `Enter` or `r` try again (after a
/// failure), `q` or `Esc` quit. `key` is the key's name as the window gives
/// it: one character, `"enter"` or `"esc"`.
pub fn choice(key: &str, failed: bool) -> Option<Choice> {
    match key {
        "enter" | "r" | "R" if failed => Some(Choice::Retry),
        "q" | "Q" | "esc" => Some(Choice::Quit),
        _ => None,
    }
}

/// A choice as a button shows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Offer {
    /// What it does.
    pub choice: Choice,
    /// What it says.
    pub label: &'static str,
    /// The key that does the same.
    pub key: &'static str,
    /// The one to take: it stands out.
    pub primary: bool,
}

/// Where a step stands, as its mark shows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mark {
    /// Done.
    Done,
    /// Under way.
    Working,
    /// Under way when setup stopped.
    Failed,
    /// Still to come.
    ToDo,
}

/// A step's line.
#[derive(Clone, Debug, PartialEq)]
pub struct StepLine {
    /// What it does: "Downloading the word-timing model (wav2vec2 by Meta,
    /// 378 MB)".
    pub title: String,
    /// Where it stands.
    pub mark: Mark,
}

/// The step under way: why it is needed, how far it is.
#[derive(Clone, Debug, PartialEq)]
pub struct Current {
    /// Its place among the steps.
    pub index: usize,
    /// Why the player needs it.
    pub why: String,
    /// Its share done (0 to 1); `None` for a step of the player's own
    /// start, which is not measured.
    pub share: Option<f64>,
    /// What it has done: "812 MB of 1.88 GB · 48 MB/s · about 23 s left".
    pub detail: String,
}

/// Why setup stopped, and what to do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stopped {
    /// What went wrong.
    pub what: String,
    /// What to do about it.
    pub advice: String,
}

/// What the setup screen says now.
#[derive(Clone, Debug, PartialEq)]
pub struct Screen {
    /// The files are looked for ([`LOOKING`] shows above the steps).
    pub looking: bool,
    /// Some step gets a file: [`INTRO`] shows, and [`KEPT`] (else
    /// [`STARTING`]).
    pub files: bool,
    /// The steps, in order.
    pub steps: Vec<StepLine>,
    /// The step under way, while setup goes on.
    pub current: Option<Current>,
    /// The whole's share done (0 to 1).
    pub whole: f64,
    /// Why it stopped, waiting for Try again.
    pub stopped: Option<Stopped>,
}

impl Screen {
    /// The screen for `status`.
    pub fn of(status: &Status) -> Screen {
        let failed = status.failure.is_some();
        let steps = status
            .steps
            .iter()
            .enumerate()
            .map(|(i, step)| StepLine {
                title: step.title.clone(),
                mark: match step.state {
                    StepState::Done => Mark::Done,
                    StepState::Working if failed && status.current == Some(i) => Mark::Failed,
                    StepState::Working => Mark::Working,
                    StepState::ToDo => Mark::ToDo,
                },
            })
            .collect();
        let current = status
            .current
            .filter(|_| !failed)
            .and_then(|i| Some((i, status.steps.get(i)?)))
            .map(|(index, step)| Current {
                index,
                why: step.why.clone(),
                share: (!step.start).then(|| status.step_share()),
                detail: detail(status),
            });
        Screen {
            looking: status.looking && !failed,
            files: status.files(),
            steps,
            current,
            whole: status.overall(),
            stopped: status.failure.as_ref().map(|f| Stopped {
                what: f.what.clone(),
                advice: f.advice.clone(),
            }),
        }
    }

    /// What shows under the title.
    pub fn intro(&self) -> &'static str {
        if self.files {
            INTRO
        } else {
            STARTING
        }
    }

    /// The choices offered: Try again and Quit after a failure, Quit
    /// while setup goes on.
    pub fn offers(&self) -> Vec<Offer> {
        let quit = Offer {
            choice: Choice::Quit,
            label: "Quit",
            key: "Q",
            primary: false,
        };
        if self.stopped.is_some() {
            let retry = Offer {
                choice: Choice::Retry,
                label: "Try again",
                key: "Enter",
                primary: true,
            };
            vec![quit, retry]
        } else {
            vec![quit]
        }
    }
}

/// A share as whole percent: "43 %".
pub fn percent(share: f64) -> String {
    format!("{:.0} %", share.clamp(0.0, 1.0) * 100.0)
}

/// The outline: one line per thing shown, its tone in brackets.
impl fmt::Display for Screen {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "[title] {TITLE}")?;
        writeln!(f, "[dim] {}", self.intro())?;
        if self.looking {
            writeln!(f, "[dim] {LOOKING}")?;
        }
        for (i, step) in self.steps.iter().enumerate() {
            let mark = match step.mark {
                Mark::Done => "done",
                Mark::Working => "working",
                Mark::Failed => "failed",
                Mark::ToDo => "to do",
            };
            writeln!(f, "[{mark}] {}", step.title)?;
            if let Some(c) = self.current.as_ref().filter(|c| c.index == i) {
                writeln!(f, "  [dim] {}", c.why)?;
                if let Some(share) = c.share {
                    writeln!(f, "  [bar accent] {}", percent(share))?;
                    writeln!(f, "  [dim] {}", c.detail)?;
                }
            }
        }
        writeln!(f, "[label] {WHOLE}: {}", percent(self.whole))?;
        writeln!(f, "[bar good] {}", percent(self.whole))?;
        if let Some(stopped) = &self.stopped {
            writeln!(f, "[fail] {}", stopped.what)?;
            writeln!(f, "[text] {}", stopped.advice)?;
        } else if self.files {
            writeln!(f, "[faint] {KEPT}")?;
        }
        for offer in self.offers() {
            let tone = if offer.primary { "primary" } else { "button" };
            writeln!(f, "[{tone}] {} ({})", offer.label, offer.key)?;
        }
        Ok(())
    }
}

fn size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1e9)
    } else {
        format!("{} MB", bytes / 1_000_000)
    }
}

/// What the step under way has done: "812 MB of 1.88 GB · 48 MB/s · about
/// 23 s left", "tensor 120 of 401".
fn detail(status: &Status) -> String {
    match status.unit {
        Some(Unit::Bytes) if status.total > 0 => {
            let mut text = format!("{} of {}", size(status.done), size(status.total));
            if let Some(rate) = status.rate.filter(|r| *r > 0.0) {
                let left = (status.total.saturating_sub(status.done)) as f64 / rate;
                text.push_str(&format!(" · {:.0} MB/s", rate / 1e6));
                if left >= 1.0 {
                    let left = left.round() as u64;
                    text.push_str(&if left >= 90 {
                        format!(" · about {} min left", left.div_ceil(60))
                    } else {
                        format!(" · about {left} s left")
                    });
                }
            }
            text
        }
        Some(Unit::Tensors) if status.total > 0 => {
            format!("tensor {} of {}", status.done, status.total)
        }
        Some(Unit::Tensors) => "reading the download…".to_owned(),
        _ => String::new(),
    }
}
