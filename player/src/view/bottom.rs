//! The bottom bar's model ([`Bar`]): the time heard, the seek bar's share
//! and the length; the buttons (the transport, the stems' switches with a
//! GPU, the settings and the help); and the message. The window draws it
//! ([`crate::gui`]'s bottom bar) over the place the cell layout leaves for
//! it, under the spectrum ([`super::plot`]).
//!
//! What fits is decided here, on the bar's width in points and the
//! measures the window draws it with ([`measure`]): the stems by name,
//! by initial or not at all, so the message keeps its room; the settings
//! and the help with their names or as icons alone.

use std::fmt;

use crate::core::keymap::Op;
use crate::core::text::clock;
use crate::core::{Gpu, Level, Screen, State};
use jodel_playback::stems::LANES;
use jodel_playback::PlayState;

/// The bar's measures, in points, as the window draws it: what fits is
/// decided on them.
pub mod measure {
    /// A button with an icon alone, and the gap after it.
    pub const ICON: f32 = 38.0;
    /// A letter of a label, on average.
    pub const LETTER: f32 = 7.5;
    /// A labelled button's icon, padding and gap, beside its letters.
    pub const LABELLED: f32 = 52.0;
    /// A stem's chip's padding and the gap after it, beside its letters.
    pub const CHIP: f32 = 24.0;
    /// The least room the message keeps.
    pub const MESSAGE: f32 = 128.0;
    /// The narrowest bar whose settings and help show their names.
    pub const NAMED: f32 = 560.0;
}

/// A button on the bar.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Press {
    /// What it does.
    pub op: Op,
    /// What it says (or, for an icon alone, its name).
    pub label: String,
    /// Its stem is switched off (a stem's switch).
    pub off: bool,
}

impl Press {
    fn new(op: Op, label: impl Into<String>) -> Press {
        Press {
            op,
            label: label.into(),
            off: false,
        }
    }
}

/// How the message is coloured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tone {
    /// What was just said.
    Text,
    /// The play state and the hints.
    Dim,
    /// An error.
    Fail,
}

/// The time heard, the seek bar and the length, as they are now (the
/// bar's, and Karaoke mode's controls').
#[derive(Clone, Debug, PartialEq)]
pub struct Seek {
    /// The time heard (where a drag would go, while the bar is held).
    pub elapsed: String,
    /// The length (`--:--` without one).
    pub total: String,
    /// The share of the track heard (or where the drag is): the head.
    pub share: f64,
    /// A track is loaded: the times are bright and the head shows.
    pub track: bool,
    /// The seek bar is held.
    pub held: bool,
}

impl Seek {
    /// The seek bar for `state`.
    pub fn of(state: &State) -> Seek {
        let duration = state.playing.duration();
        let position = match (state.seek_drag, duration) {
            (Some(f), Some(d)) => f * d,
            _ => state.playing.position.max(0.0),
        };
        let share = match duration {
            Some(d) if d > 0.0 => (position / d).clamp(0.0, 1.0),
            _ => 0.0,
        };
        Seek {
            elapsed: clock(position),
            total: duration.map_or_else(|| "--:--".to_owned(), clock),
            share,
            track: state.playing.track.is_some(),
            held: state.seek_drag.is_some(),
        }
    }
}

/// The bottom bar as it is now.
#[derive(Clone, Debug, PartialEq)]
pub struct Bar {
    /// The time heard, the seek bar and the length.
    pub seek: Seek,
    /// The transport: previous, play or pause, next, stop.
    pub transport: Vec<Press>,
    /// The stems' switches (with a GPU, where they fit), by name or
    /// initial.
    pub stems: Vec<Press>,
    /// The settings and the help, at the right.
    pub ends: Vec<Press>,
    /// Whether the settings and the help show their names.
    pub named_ends: bool,
    /// The message, the preview's line or the play state with the hints.
    pub message: String,
    /// Its colour.
    pub tone: Tone,
}

impl Bar {
    /// The bar for `state`, `width` points wide.
    pub fn of(state: &State, width: f32) -> Bar {
        let play = match state.playing.state {
            PlayState::Playing => "pause",
            _ => "play",
        };
        let transport = vec![
            Press::new(Op::Previous, "previous"),
            Press::new(Op::PlayPause, play),
            Press::new(Op::Next, "next"),
            Press::new(Op::Stop, "stop"),
        ];
        let named_ends = width >= measure::NAMED;
        let ends = vec![
            Press::new(Op::Settings, "Settings"),
            Press::new(Op::Help, "Help"),
        ];
        let ends_w: f32 = if named_ends {
            ends.iter()
                .map(|p| letters(&p.label) + measure::LABELLED)
                .sum()
        } else {
            ends.len() as f32 * measure::ICON
        };
        let room = width - transport.len() as f32 * measure::ICON - ends_w;
        let stems = stems(state, room);
        let (message, tone) = message(state);
        Bar {
            seek: Seek::of(state),
            transport,
            stems,
            ends,
            named_ends,
            message,
            tone,
        }
    }
}

/// `text`'s width in letters of [`measure::LETTER`].
fn letters(text: &str) -> f32 {
    text.chars().count() as f32 * measure::LETTER
}

/// The stems' switches, with a GPU, where they leave the message its room
/// in `room` points: by name, else by initial, else none.
fn stems(state: &State, room: f32) -> Vec<Press> {
    if !matches!(state.gpu, Gpu::Present(_)) {
        return Vec::new();
    }
    [true, false]
        .into_iter()
        .map(|named| {
            LANES
                .iter()
                .map(|&stem| {
                    let name = stem.id();
                    let label = if named {
                        name.to_owned()
                    } else {
                        name[..1].to_uppercase()
                    };
                    Press {
                        op: Op::for_stem(stem),
                        label,
                        off: crate::core::stems::is_off(state, stem),
                    }
                })
                .collect::<Vec<_>>()
        })
        .find(|stems| {
            let w: f32 = stems
                .iter()
                .map(|p| letters(&p.label) + measure::CHIP)
                .sum();
            w + measure::MESSAGE <= room
        })
        .unwrap_or_default()
}

/// The message line: what was said last (an error in its tone), else a
/// preview's line while it lasts, else the play state with the hints on
/// the main screen.
fn message(state: &State) -> (String, Tone) {
    match (&state.message, crate::core::preview::line(state)) {
        (Some(m), _) => (
            m.text.clone(),
            if m.level == Level::Error {
                Tone::Fail
            } else {
                Tone::Text
            },
        ),
        (None, Some(line)) => (line, Tone::Text),
        (None, None) => {
            let mut text = match state.playing.state {
                PlayState::Playing => "playing",
                PlayState::Paused => "paused",
                PlayState::Stopped => "stopped",
            }
            .to_owned();
            if state.screen == Screen::Main {
                use crate::core::keymap::{hint, Context};
                let key = |op| hint(Context::Global, op);
                text.push_str(&format!(
                    "  ·  {} commands  {} filter  {} keys",
                    key(Op::Palette),
                    key(Op::Filter),
                    key(Op::Help)
                ));
            }
            (text, Tone::Dim)
        }
    }
}

/// An outline of the bar, one line a piece with its tone in brackets:
/// for the snapshots.
impl fmt::Display for Bar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let seek = &self.seek;
        let time = if seek.track { "time" } else { "time faint" };
        writeln!(f, "[{time}] {}", seek.elapsed)?;
        let held = if seek.held { " held" } else { "" };
        writeln!(f, "[seek{held}] {:.0} %", seek.share * 100.0)?;
        writeln!(f, "[{time}] {}", seek.total)?;
        for p in &self.transport {
            writeln!(f, "[icon] {}", p.label)?;
        }
        for p in &self.stems {
            let off = if p.off { " off" } else { "" };
            writeln!(f, "[chip{off}] {}", p.label)?;
        }
        let tone = match self.tone {
            Tone::Text => "text",
            Tone::Dim => "dim",
            Tone::Fail => "fail",
        };
        writeln!(f, "[{tone}] {}", self.message)?;
        let kind = if self.named_ends { "button" } else { "icon" };
        for p in &self.ends {
            writeln!(f, "[{kind}] {}", p.label)?;
        }
        Ok(())
    }
}
