//! The scoring of a song's singing as it goes ([`super::Singing::tally`]):
//! the points, the words hit, half hit and scored, the streak.

use super::{multiplier, Hit, Level, Scored, WORD_POINTS};

/// The scoring at a moment.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Tally {
    /// The words scored so far (of the song's words with notes, by when
    /// they end).
    pub(super) next_word: usize,
    /// The score.
    pub score: u32,
    /// The words hit.
    pub hit_words: u32,
    /// The words half hit.
    pub half_words: u32,
    /// The words scored (hit or not).
    pub scored_words: u32,
    /// The words hit in a row up to now (a half hit keeps it).
    pub streak: u32,
    /// The longest streak.
    pub best_streak: u32,
    /// The last word scored.
    pub last_scored: Option<Scored>,
}

impl Tally {
    /// A word ending at `end` scored at `level` by its frames `counts` (on
    /// its notes, those that hit, those that half hit): none heard (sought
    /// past), not scored.
    pub(super) fn score_word(
        &mut self,
        level: Level,
        (noted, hit, half): (u32, u32, u32),
        end: f64,
    ) {
        if noted == 0 {
            return;
        }
        self.scored_words += 1;
        let got = level
            .credit(f64::from(noted), f64::from(hit), f64::from(half))
            .hit();
        let base = match got {
            Some(Hit::Full) => {
                self.hit_words += 1;
                self.streak += 1;
                WORD_POINTS
            }
            Some(Hit::Half) => {
                self.half_words += 1;
                WORD_POINTS / 2
            }
            None => {
                self.streak = 0;
                0
            }
        };
        self.best_streak = self.best_streak.max(self.streak);
        let points = base * multiplier(self.streak);
        self.score += points;
        self.last_scored = Some(Scored {
            end,
            hit: got,
            points,
            streak: self.streak,
        });
    }
}
