//! Which of the desk's strips are shown in its room, in letters of the
//! fixed face ([`crate::core::Fit::desk_letters`]): every strip where it
//! fits, the player's stems while the player has them; where it is too
//! narrow, the microphone goes first, then the karaoke game's effects'
//! strip, then the preview, then the stems. The arrow keys go through the
//! strips shown, and the window lays out the same ones.

use crate::core::desk::{Channel, StripKind};

/// A strip's widths in letters: on a narrow desk, the least it grows from,
/// and the widest.
pub fn widths(channel: Channel) -> (u16, u16, u16) {
    match channel {
        Channel::Strip(
            StripKind::Drums | StripKind::Bass | StripKind::Vocals | StripKind::Other,
        ) => (7, 7, 10),
        // The preview and the effects without a scale of their own where it
        // is narrow.
        Channel::Strip(StripKind::Preview | StripKind::Effects) => (9, 12, 16),
        Channel::Strip(StripKind::Player) => (12, 12, 16),
        Channel::Mic => (18, 18, 22),
        // The master's lines whole: its backend, format and volume.
        Channel::Main => (24, 26, 36),
    }
}

/// The gap between strips, and before the master (one at the narrow
/// widths); the stems' strips stand side by side under their bracket.
const GAP: u16 = 1;
const MASTER_GAP: u16 = 3;

/// The gap before `next`, after `prev`, in letters.
pub fn gap(prev: Channel, next: Channel, tight: bool) -> u16 {
    match (prev, next) {
        (_, Channel::Main) if tight => GAP,
        (_, Channel::Main) => MASTER_GAP,
        (a, b) if a.is_part() && b.is_part() => 0,
        _ => GAP,
    }
}

/// The gaps between the strips of `set`, in letters.
pub fn gaps(set: &[Channel], tight: bool) -> u16 {
    set.windows(2).map(|w| gap(w[0], w[1], tight)).sum()
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

/// The strips shown in `room` letters; `stems`: the player has its stems
/// on the desk. The first set that fits at its least widths, or else at
/// its narrow ones; where not even the player and the master fit, the
/// master alone in the room.
pub fn strips_shown(room: u16, stems: bool) -> Shown {
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
        let least: u16 = strips.iter().map(|&c| widths(c).1).sum::<u16>() + gaps(&strips, false);
        if room >= least {
            return Shown {
                strips,
                widths: Widths::Least,
            };
        }
        let narrow: u16 = strips.iter().map(|&c| widths(c).0).sum::<u16>() + gaps(&strips, true);
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
