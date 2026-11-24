//! The scoring of a song's singing as it goes ([`super::Singing::tally`]):
//! the points, the words hit, rescued and scored, the streak, and how far
//! the words and phrases are scored. What each phrase shone is in the
//! moments ([`super::Singing::shone`]).
//!
//! A single word missed does not end the streak at once: it waits
//! ([`Run::Waiting`]) for the very next word to burst, which rescues it
//! ([`Tally::rescue`]: scored as if sung, the streak going on). A second
//! word missed in a row ends the streak ([`Run::Broken`]), as the first
//! then does; a word missed after those waits for no rescue. A word passed
//! unheard (sought past) ends the wait without a rescue: a seek is not a
//! word sung wrong, so nothing is lost either.

use super::{multiplier, Scored, WORD_POINTS};

/// How the last words scored left the streak.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Run {
    /// Going on: the last word hit (or rescued), or none scored yet.
    #[default]
    On,
    /// A single word missed (its place among the song's words, by when
    /// they end), waiting for the next to burst to rescue it.
    Waiting(usize),
    /// Two words missed in a row: the streak lost.
    Broken,
}

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
    /// The words hit (those rescued among them).
    pub hit_words: u32,
    /// The words missed and rescued by a shot: scored as if sung, so they
    /// count among [`Tally::hit_words`] (and so for the medal) too.
    pub rescued: u32,
    /// The words scored (hit or not).
    pub scored_words: u32,
    /// The words hit in a row up to now.
    pub streak: u32,
    /// The longest streak.
    pub best_streak: u32,
    /// The last word scored.
    pub last_scored: Option<Scored>,
    /// How the last words scored left the streak.
    pub run: Run,
}

impl Tally {
    /// Word `order` (its place among the song's words) ending at `end`
    /// scored by its frames `(noted, hit)` (on its notes, those that hit);
    /// whether it was (none heard, sought past: not scored, and the word
    /// waiting, if any, waits no more).
    pub(super) fn score_word(&mut self, (noted, hit): (u32, u32), end: f64, order: usize) -> bool {
        if noted == 0 {
            if let Run::Waiting(_) = self.run {
                self.run = Run::On;
            }
            return false;
        }
        self.scored_words += 1;
        let got = super::judge::counts((noted, hit));
        let points = if got {
            self.hit()
        } else {
            // A single miss waits for its rescue; a second in a row ends
            // the streak, the first with it.
            self.run = match self.run {
                Run::On => Run::Waiting(order),
                Run::Waiting(_) | Run::Broken => {
                    self.streak = 0;
                    Run::Broken
                }
            };
            0
        };
        self.last_scored = Some(Scored {
            end,
            hit: got,
            points,
            streak: self.streak,
        });
        true
    }

    /// The word waiting ([`Run::Waiting`]) rescued: scored as if sung.
    pub(super) fn rescue(&mut self) {
        self.rescued += 1;
        self.hit();
    }

    /// A word hit: its points (times the streak's multiplier with it), the
    /// streak going on.
    fn hit(&mut self) -> u32 {
        self.hit_words += 1;
        self.streak += 1;
        self.best_streak = self.best_streak.max(self.streak);
        self.run = Run::On;
        let points = WORD_POINTS * multiplier(self.streak);
        self.score += points;
        points
    }
}
