//! The desk's strips as the Mixer tab names them: the player's group with
//! its four stems inside it, the preview's strip, the microphone's and the
//! master.

use jodel_playback::stems::{Stem, LANES};

/// A channel strip with a fader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StripKind {
    /// The song's player: a group, its main on it and its stems inside it.
    Player,
    /// The song's drums, inside the player's group.
    Drums,
    /// Its bass.
    Bass,
    /// Its vocals.
    Vocals,
    /// Everything else of it.
    Other,
    /// The preview's players.
    Preview,
}

impl StripKind {
    /// How many there are.
    pub const COUNT: usize = 6;

    /// Every one, left to right on the desk.
    pub const ALL: [StripKind; StripKind::COUNT] = [
        StripKind::Player,
        StripKind::Drums,
        StripKind::Bass,
        StripKind::Vocals,
        StripKind::Other,
        StripKind::Preview,
    ];

    /// The player's stems, in the order of their lanes ([`LANES`]).
    pub const PARTS: [StripKind; 4] = [
        StripKind::Drums,
        StripKind::Bass,
        StripKind::Vocals,
        StripKind::Other,
    ];

    /// The name of the strip on the mixer: a stem's is the stem's own, as
    /// the player's desk names it ([`jodel_playback::StemDesk`]).
    pub fn name(self) -> &'static str {
        match self {
            StripKind::Player => "Player",
            StripKind::Preview => "Preview",
            part => part.stem().map_or("", Stem::id),
        }
    }

    /// Its name on its scribble strip.
    pub fn label(self) -> &'static str {
        match self {
            StripKind::Player => "PLAYER",
            StripKind::Drums => "DRUMS",
            StripKind::Bass => "BASS",
            StripKind::Vocals => "VOCALS",
            StripKind::Other => "OTHER",
            StripKind::Preview => "PREVIEW",
        }
    }

    /// Its name on a narrow scribble strip (five cells at most).
    pub fn short_label(self) -> &'static str {
        match self {
            StripKind::Player => "PLAY",
            StripKind::Vocals => "VOX",
            StripKind::Preview => "PREV",
            other => other.label(),
        }
    }

    /// Where it is in arrays of every one ([`StripKind::ALL`]'s order).
    pub fn index(self) -> usize {
        match self {
            StripKind::Player => 0,
            StripKind::Drums => 1,
            StripKind::Bass => 2,
            StripKind::Vocals => 3,
            StripKind::Other => 4,
            StripKind::Preview => 5,
        }
    }

    /// Whether it is one of the player's stems.
    pub fn is_part(self) -> bool {
        self.stem().is_some()
    }

    /// The group it sits in: the player's, for a stem.
    pub fn parent(self) -> Option<StripKind> {
        self.is_part().then_some(StripKind::Player)
    }

    /// The stem it plays, for one of the player's stems.
    pub fn stem(self) -> Option<Stem> {
        match self {
            StripKind::Drums => Some(Stem::Drums),
            StripKind::Bass => Some(Stem::Bass),
            StripKind::Vocals => Some(Stem::Vocals),
            StripKind::Other => Some(Stem::Other),
            StripKind::Player | StripKind::Preview => None,
        }
    }

    /// The strip of `stem`.
    pub fn for_stem(stem: Stem) -> StripKind {
        match stem {
            Stem::Drums => StripKind::Drums,
            Stem::Bass => StripKind::Bass,
            Stem::Vocals => StripKind::Vocals,
            Stem::Other => StripKind::Other,
        }
    }

    /// Where a stem's strip is among the lanes ([`LANES`], [`StripKind::PARTS`]).
    pub fn lane(self) -> Option<usize> {
        let stem = self.stem()?;
        LANES.iter().position(|&s| s == stem)
    }
}

/// A strip of the desk, as the arrow keys go through them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Channel {
    /// A strip with a fader.
    Strip(StripKind),
    /// The microphone's strip: listened to, never played.
    Mic,
    /// The master: what goes to the device.
    Main,
}

impl Channel {
    /// Left to right.
    pub const ALL: [Channel; 8] = [
        Channel::Strip(StripKind::Player),
        Channel::Strip(StripKind::Drums),
        Channel::Strip(StripKind::Bass),
        Channel::Strip(StripKind::Vocals),
        Channel::Strip(StripKind::Other),
        Channel::Strip(StripKind::Preview),
        Channel::Mic,
        Channel::Main,
    ];

    /// Whether it is one of the player's stems.
    pub fn is_part(self) -> bool {
        matches!(self, Channel::Strip(kind) if kind.is_part())
    }
}

impl Default for Channel {
    /// The player's strip, as the tab opens.
    fn default() -> Self {
        Channel::Strip(StripKind::Player)
    }
}

/// A button on a strip.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeskButton {
    /// Mute.
    Mute(StripKind),
    /// Solo.
    Solo(StripKind),
    /// The microphone's echo canceller on or off.
    Echo,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stems_sit_in_the_player_in_the_order_of_their_lanes() {
        for (k, kind) in StripKind::ALL.into_iter().enumerate() {
            assert_eq!(kind.index(), k);
        }
        for (k, kind) in StripKind::PARTS.into_iter().enumerate() {
            assert_eq!(kind.lane(), Some(k));
            assert_eq!(kind.parent(), Some(StripKind::Player));
            assert_eq!(StripKind::for_stem(kind.stem().unwrap()), kind);
            assert_eq!(LANES[k], kind.stem().unwrap());
        }
        assert!(!StripKind::Player.is_part() && StripKind::Player.parent().is_none());
        assert!(!StripKind::Preview.is_part());
    }
}
