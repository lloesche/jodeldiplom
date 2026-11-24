//! How a song was sung, shown when it ends ([`SongResult`]), and its medal.

use crate::core::shine::{Shone, Tier};

/// A medal for a song sung: by the words hit (not by the points, which
/// the streaks multiply).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Medal {
    /// Half of them.
    Bronze,
    /// 70 %.
    Silver,
    /// 85 %.
    Gold,
}

impl Medal {
    /// The medal for `hit` of `words` words, if any.
    pub fn of(hit: u32, words: u32) -> Option<Medal> {
        if words == 0 {
            return None;
        }
        let share = f64::from(hit) / f64::from(words);
        if share >= 0.85 {
            Some(Medal::Gold)
        } else if share >= 0.7 {
            Some(Medal::Silver)
        } else if share >= 0.5 {
            Some(Medal::Bronze)
        } else {
            None
        }
    }

    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Medal::Bronze => "Bronze",
            Medal::Silver => "Silver",
            Medal::Gold => "Gold",
        }
    }
}

/// How a song was sung, shown when it ends.
#[derive(Clone, Debug, PartialEq)]
pub struct SongResult {
    /// The song's title.
    pub title: String,
    /// The points.
    pub points: u32,
    /// The words hit.
    pub hit: u32,
    /// The words scored.
    pub words: u32,
    /// The most words hit in a row.
    pub streak: u32,
    /// Each phrase that shone, in order: its shine and tier.
    pub phrases: Vec<Shone>,
    /// The medal won, if any.
    pub medal: Option<Medal>,
    /// The best score of the song before (none: the first).
    pub best: Option<u32>,
    /// Whether the song was sung whole (from its start): only then a medal
    /// and a best.
    pub whole: bool,
}

impl SongResult {
    /// Whether it beat the best before (or is the first).
    pub fn new_best(&self) -> bool {
        self.best.is_none_or(|b| self.points > b)
    }

    /// The best tier a phrase reached, if any.
    pub fn best_tier(&self) -> Option<Tier> {
        self.phrases.iter().filter_map(|p| p.tier).max()
    }

    /// How many phrases reached each tier (as [`Tier::ALL`]: Good first).
    pub fn tier_counts(&self) -> [u32; Tier::ALL.len()] {
        let mut out = [0; Tier::ALL.len()];
        for t in self.phrases.iter().filter_map(|p| p.tier) {
            out[t as usize] += 1;
        }
        out
    }
}
