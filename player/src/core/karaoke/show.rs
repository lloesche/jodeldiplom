//! What Karaoke mode shows of the game ([`Show`]), worked out once from
//! its two settings: the game ([`crate::config::Config::karaoke_game`]:
//! the singing scored) and the notes
//! ([`crate::config::Config::karaoke_notes`]: the lane of them). Every part
//! of the mode reads a field of it ([`State::show`]) rather than putting
//! the two together itself.

use crate::config::Config;
use crate::core::State;

/// What Karaoke mode shows of the game: the cover behind the stage (else
/// the blurred cover alone: plain lyrics karaoke), the lane of notes the
/// singer sings to, the singing scored (the score, the effects' tiers,
/// the diploma) and the game's sounds (its bubbles popping: scored on the
/// lane).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Show {
    /// The cover behind the stage.
    pub cover: bool,
    /// The lane of notes.
    pub lane: bool,
    /// The singing scored.
    pub scored: bool,
    /// The game's sounds.
    pub sounds: bool,
}

impl Show {
    /// What `config`'s settings show: the cover with either on, the lane
    /// with the notes, the scoring with the game, the sounds with both.
    pub fn of(config: &Config) -> Show {
        let (game, notes) = (config.karaoke_game, config.karaoke_notes);
        Show {
            cover: game || notes,
            lane: notes,
            scored: game,
            sounds: game && notes,
        }
    }
}

impl State {
    /// What Karaoke mode shows of the game with the settings now
    /// ([`Show::of`]).
    pub fn show(&self) -> Show {
        Show::of(&self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four ways the two settings go, each what it shows.
    #[test]
    fn the_two_settings_show_the_game_four_ways() {
        let show = |karaoke_game, karaoke_notes| {
            Show::of(&Config {
                karaoke_game,
                karaoke_notes,
                ..Config::default()
            })
        };
        let all = |cover, lane, scored, sounds| Show {
            cover,
            lane,
            scored,
            sounds,
        };
        assert_eq!(show(true, true), all(true, true, true, true));
        assert_eq!(show(true, false), all(true, false, true, false));
        assert_eq!(show(false, true), all(true, true, false, false));
        assert_eq!(show(false, false), all(false, false, false, false));
    }
}
