//! What Karaoke mode shows of the game ([`Show`]), worked out once from
//! its two settings, the game ([`crate::config::Config::karaoke_game`]:
//! the singing scored) and the notes
//! ([`crate::config::Config::karaoke_notes`]: the lane of them), and
//! whether stories are made here. Every part of the mode reads a field of
//! it ([`State::show`]) rather than putting them together itself.

use crate::config::Config;
use crate::core::{Job, State};

/// What Karaoke mode shows of the game: the song's story behind the stage
/// (else the blurred cover alone), the lane of notes the singer sings to,
/// the singing scored (the score, the effects' tiers, the diploma) and the
/// game's sounds (its bubbles popping: scored on the lane).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Show {
    /// The song's story behind the stage.
    pub story: bool,
    /// The lane of notes.
    pub lane: bool,
    /// The singing scored.
    pub scored: bool,
    /// The game's sounds.
    pub sounds: bool,
}

impl Show {
    /// What `config`'s settings show where stories are made or not
    /// (`stories`): the story with the game, where they are (else the
    /// blurred cover: plain lyrics karaoke, the notes on or not, or the
    /// game on a machine that makes no stories), the lane with the notes,
    /// the scoring with the game, the sounds with both.
    pub fn of(config: &Config, stories: bool) -> Show {
        let (game, notes) = (config.karaoke_game, config.karaoke_notes);
        Show {
            story: game && stories,
            lane: notes,
            scored: game,
            sounds: game && notes,
        }
    }
}

impl State {
    /// What Karaoke mode shows of the game with the settings now, where
    /// stories are made as they are ([`Show::of`]).
    pub fn show(&self) -> Show {
        Show::of(&self.config, self.preparing.contains(&Job::Story))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four ways the two settings go, each what it shows; with no
    /// stories made here, no story.
    #[test]
    fn the_two_settings_show_the_game_four_ways() {
        let show = |karaoke_game, karaoke_notes, stories| {
            let config = Config {
                karaoke_game,
                karaoke_notes,
                ..Config::default()
            };
            Show::of(&config, stories)
        };
        let all = |story, lane, scored, sounds| Show {
            story,
            lane,
            scored,
            sounds,
        };
        assert_eq!(show(true, true, true), all(true, true, true, true));
        assert_eq!(show(true, false, true), all(true, false, true, false));
        assert_eq!(show(false, true, true), all(false, true, false, false));
        assert_eq!(show(false, false, true), all(false, false, false, false));
        assert_eq!(show(true, true, false), all(false, true, true, true));
    }
}
