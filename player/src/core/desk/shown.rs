//! Which of the desk's strips are shown in its width, in points
//! ([`crate::core::Fit::desk_width`]): every strip where it fits, the
//! player's stems while the player has them; where it is too narrow, the
//! microphone goes first, then the karaoke game's effects' strip, then the
//! preview, then the stems. The arrow keys go through the strips shown,
//! and the window lays out the same ones (`gui::mixer::layout`).
//!
//! The widths are the cell desk's, a cell 8 points (the fixed face's at a
//! scale of 1), so the strips go at the same window widths as they did:
//! every one at 120 columns (960 points), all but the microphone's at 80
//! (640).

use crate::core::desk::{Channel, StripKind};

/// A strip's widths in points: on a narrow desk, the least it grows from,
/// and the widest.
pub fn widths(channel: Channel) -> (f32, f32, f32) {
    match channel {
        Channel::Strip(
            StripKind::Drums | StripKind::Bass | StripKind::Vocals | StripKind::Other,
        ) => (56.0, 56.0, 80.0),
        // The preview and the effects without a scale of their own where it
        // is narrow.
        Channel::Strip(StripKind::Preview | StripKind::Effects) => (72.0, 96.0, 128.0),
        Channel::Strip(StripKind::Player) => (96.0, 96.0, 128.0),
        Channel::Mic => (144.0, 144.0, 176.0),
        // The master's lines whole: its backend, format and volume.
        Channel::Main => (192.0, 208.0, 288.0),
    }
}

/// The room round the desk, each side, in points.
pub const EDGE: f32 = 8.0;

/// The gap between strips, and before the master (one at the narrow
/// widths); the stems' strips stand side by side under their bracket.
const GAP: f32 = 8.0;
const MASTER_GAP: f32 = 24.0;

/// The gap before `next`, after `prev`, in points.
pub fn gap(prev: Channel, next: Channel, tight: bool) -> f32 {
    match (prev, next) {
        (_, Channel::Main) if tight => GAP,
        (_, Channel::Main) => MASTER_GAP,
        (a, b) if a.is_part() && b.is_part() => 0.0,
        _ => GAP,
    }
}

/// The gaps between the strips of `set`, in points.
pub fn gaps(set: &[Channel], tight: bool) -> f32 {
    set.windows(2).map(|w| gap(w[0], w[1], tight)).sum()
}

/// The room for the strips on a desk `width` points wide.
pub fn room(width: f32) -> f32 {
    (width - 2.0 * EDGE).max(0.0)
}

/// The widths the strips shown are at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Widths {
    /// From their least widths (the second of [`widths`]) growing into the
    /// room up to their widest.
    Least,
    /// At their narrow widths (the first), with the narrow gaps.
    Narrow,
    /// The master alone, as wide as the room.
    Room,
}

impl Widths {
    /// Whether the narrow gaps go between the strips.
    pub fn tight(self) -> bool {
        self != Widths::Least
    }
}

/// The strips shown, left to right, the master last.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shown {
    /// Which.
    pub strips: Vec<Channel>,
    /// At what widths.
    pub widths: Widths,
}

/// The strips shown on a desk `width` points wide; `stems`: the player has
/// its stems on the desk. The first set that fits its [`room`] at its
/// least widths, or else at its narrow ones; where not even the player and
/// the master fit, the master alone in the room.
pub fn strips_shown(width: f32, stems: bool) -> Shown {
    let room = room(width);
    let player = Channel::Strip(StripKind::Player);
    let preview = Channel::Strip(StripKind::Preview);
    let effects = Channel::Strip(StripKind::Effects);
    let parts: Vec<Channel> = if stems {
        StripKind::PARTS.map(Channel::Strip).to_vec()
    } else {
        Vec::new()
    };
    let with = |rest: &[Channel]| -> Vec<Channel> {
        std::iter::once(player)
            .chain(parts.iter().copied())
            .chain(rest.iter().copied())
            .collect()
    };
    let mut sets: Vec<Vec<Channel>> = vec![
        with(&[preview, effects, Channel::Mic, Channel::Main]),
        with(&[preview, effects, Channel::Main]),
        with(&[preview, Channel::Main]),
    ];
    if stems {
        sets.push(with(&[Channel::Main]));
    }
    sets.push(vec![player, Channel::Main]);
    for strips in sets {
        let least: f32 = strips.iter().map(|&c| widths(c).1).sum::<f32>() + gaps(&strips, false);
        if room >= least {
            return Shown {
                strips,
                widths: Widths::Least,
            };
        }
        let narrow: f32 = strips.iter().map(|&c| widths(c).0).sum::<f32>() + gaps(&strips, true);
        if room >= narrow {
            return Shown {
                strips,
                widths: Widths::Narrow,
            };
        }
    }
    Shown {
        strips: vec![Channel::Main],
        widths: Widths::Room,
    }
}
