//! The scoring of a song's singing as it goes ([`super::Singing::tally`]):
//! the points, the words hit and scored, the streak, and how far the words
//! and phrases are scored. What each phrase shone is in the moments
//! ([`super::Singing::shone`]).

use super::{multiplier, Scored, WORD_POINTS};

/// The scoring at a moment.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Tally {
    /// The words scored so far (of the song's words with notes, by when
    /// they end).
    pub(super) next_word: usize,
    /// The phrases looked at so far (by when they end).
    pub(super) next_phrase: usize,
    /// The score.
    pub score: u32,
    /// The words hit.
    pub hit_words: u32,
    /// The words scored (hit or not).
    pub scored_words: u32,
    /// The words hit in a row up to now.
    pub streak: u32,
    /// The longest streak.
    pub best_streak: u32,
    /// The last word scored.
    pub last_scored: Option<Scored>,
}

impl Tally {
    /// A word ending at `end` scored by its frames `(noted, hit)` (on its
    /// notes, those that hit); whether it was (none heard, sought past: not
    /// scored).
    pub(super) fn score_word(&mut self, (noted, hit): (u32, u32), end: f64) -> bool {
        if noted == 0 {
            return false;
        }
        self.scored_words += 1;
        let got = super::judge::credit(f64::from(noted), f64::from(hit)) >= 1.0;
        let base = if got {
            self.hit_words += 1;
            self.streak += 1;
            WORD_POINTS
        } else {
            self.streak = 0;
            0
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
        true
    }
}
